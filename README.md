# Nova — Single-Host Video Ingest + VLM Captioning

## Overview

Nova now uses a two-service pipeline designed for one machine and one GPU:

```mermaid
flowchart LR
    Sources[RTSPAndFileSources] --> IngestApi[videoIngestApi]
    IngestApi --> IngestWorker[videoIngestWorker]
    IngestWorker --> BatchFs[sharedFrameBatchFs]
    IngestWorker --> FrameBatchStream[nova_frame_batches]
    FrameBatchStream --> Captioner[vlmCaptioner]
    BatchFs --> Captioner
    Captioner --> CaptionStream[nova_captions]
    Captioner --> InsightStream[openclaw_insights]
    InsightStream --> Skills[skills]
    Skills --> AlertStream[openclaw_alerts]
    AlertStream --> Consumers[consumers]
```

The key design rules are:

- `video-ingest` owns decode, chunking, frame sampling, frame-batch storage, and control-plane event publication.
- `vlm-captioner` owns VLM inference, idempotent caption persistence, and publication of `caption_ready` plus compatibility `openclaw:insights` events.
- Redis carries metadata and control only. Frames and optional audio stay on a shared local filesystem.
- Frame batches are the canonical unit of work.

## Repository Layout

- `app/video_ingest/`: ingest API routes, Celery ingest tasks, media pipeline helpers, frame-batch storage.
- `app/vlm_captioner/`: Redis consumer loop, caption processing, caption/insight publishers.
- `app/state/`: Redis-backed job, stream, insight, and caption persistence.
- `app/services/`: reusable FFmpeg, Ollama, and compatibility helpers.
- `common/contracts/`: shared frame-batch and caption event/manifest models.
- `common/event_bus.py`: shared Redis Streams helper used by services, skills, and consumers.
- `skills/`: event-driven detectors that subscribe to `openclaw:insights`.
- `consumers/`: downstream alert sinks.

## Services

### `video-ingest-api`

FastAPI control-plane service for:

- `GET /v1/health`
- `POST /v1/jobs`
- `GET /v1/jobs/{job_id}`
- `GET /v1/jobs/{job_id}/results`
- `POST /v1/streams`
- `GET /v1/streams`
- `GET /v1/streams/{stream_id}`
- `DELETE /v1/streams/{stream_id}`
- `GET /v1/insights`
- `GET /v1/streams/{stream_id}/insights`
- `POST /v1/chat/completions` only when `ENABLE_DIRECT_CHAT_COMPLETIONS=true` for explicit compatibility

### `video-ingest-worker`

Celery worker that:

- validates RTSP/file sources
- samples frames once using FFmpeg/NVDEC-aware helpers
- writes canonical frame-batch directories under `FRAME_BATCH_ROOT`
- optionally preserves chunk and audio artifacts
- publishes `frame_batch_ready` metadata to `nova:frame_batches`
- enforces bounded producer backpressure via Redis job/stream state

### `vlm-captioner`

Long-running Redis Streams consumer that:

- claims or consumes `frame_batch_ready` events
- loads the referenced manifest and frames from disk
- runs VLM inference with the existing Ollama/OpenAI-compat code
- stores idempotent caption records keyed by `batch_id`
- writes `completion.json` beside each captioned manifest
- publishes `caption_ready` to `nova:captions`
- publishes compatibility insight events to `openclaw:insights`

### Skills and consumers

Existing skills and consumers remain event-driven:

- skills still register RTSP streams through `POST /v1/streams`
- skills still consume `openclaw:insights`
- consumers still consume `openclaw:alerts`

## Canonical Frame-Batch Layout

Each produced batch lives under:

```text
<FRAME_BATCH_ROOT>/<jobs|streams>/<source_id>/<batch_id>/
  manifest.json
  frames/
    frame_000000.jpg
    frame_000001.jpg
    ...
  debug/
    source.json
  chunk_000000.mp4            # optional
  audio/
    audio_000000.m4a          # optional
```

`manifest.json` records:

- source identity: job/stream IDs, source kind, sanitized source URI
- sequencing: source index, chunk index, chunk window timestamps
- artifact pointers: ordered frame paths, optional chunk path, optional audio path
- replay/debug metadata: creation time, attempt count, cleanup deadline, status

## Redis Streams

- `nova:frame_batches`: produced by ingest, consumed by `vlm-captioner`
- `nova:captions`: produced by `vlm-captioner`, includes metadata plus `completion_path` instead of embedding large completion payloads in Redis
- `openclaw:insights`: compatibility stream for existing skills
- `openclaw:alerts`: emitted by skills and consumed by notifiers

## Audio Policy

- Ingest probes each source for audio and records `has_audio` in the manifest/event.
- Audio is metadata-only by default.
- If `PRESERVE_AUDIO_ARTIFACTS=true`, ingest extracts one aligned audio artifact per batch when audio is present and a chunk artifact is available.
- The current captioner ignores audio by default; it is preserved for future STT/omni consumers.

## Single-GPU Runtime Notes

- `vlm-captioner` is the single inference owner and should run with concurrency `1`.
- `video-ingest-worker` defaults to Celery concurrency `2` in Compose so multiple RTSP streams can make progress; reduce it if NVDEC or local decode competes with inference on your GPU.
- Redis is the control plane only; do not store image/audio bytes in Redis.
- Backpressure is bounded through per-job and per-stream `pending_batches` counters in Redis.
- Ingest waits only up to `MAX_BACKPRESSURE_WAIT_SECONDS` before failing a stuck producer instead of sleeping forever.

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Redis via Compose
- [Ollama](https://github.com/ollama/ollama) running on the host with a vision model, for example:

```bash
ollama pull llava
```

### Configure skills and consumers

```bash
cp skills/package_delivery_alert/.env.example skills/package_delivery_alert/.env
cp consumers/discord_notifier/.env.example consumers/discord_notifier/.env
```

Set at least:

```dotenv
# skills/package_delivery_alert/.env
RTSP_URL=rtsp://user:password@192.168.1.100:554/stream1
```

```dotenv
# consumers/discord_notifier/.env
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN
```

### Start the stack

```bash
docker compose up --build
```

Services:

- `video-ingest-api`: [http://localhost:8000](http://localhost:8000)
- `redis`: `localhost:6379`
- `video-ingest-worker`: background ingest producer
- `vlm-captioner`: background caption consumer
- `package_delivery_alert`: skill example
- `discord_notifier`: consumer example

### Verify

```bash
curl -sS http://localhost:8000/v1/health
curl -sS http://localhost:8000/v1/streams
redis-cli XLEN nova:frame_batches
redis-cli XLEN nova:captions
redis-cli XLEN openclaw:insights
redis-cli XLEN openclaw:alerts
```

## Important Configuration

Key application settings live in `app/config.py`.

- `REDIS_URL`: Redis for job, stream, insight, caption, and stream state
- `CELERY_BROKER_URL`: Celery broker for ingest tasks
- `OLLAMA_BASE_URL`: Ollama base URL used by `vlm-captioner` and only by `video-ingest-api` when direct chat compatibility is explicitly enabled
- `FRAME_BATCH_ROOT`: shared filesystem root for canonical frame batches
- `FRAME_BATCH_READY_STREAM`: Redis stream for ingest-to-captioner work
- `CAPTION_READY_STREAM`: Redis stream for caption outputs
- `FRAME_BATCH_STREAM_GROUP`: captioner consumer group name
- `MAX_INFLIGHT_BATCHES_PER_STREAM`: RTSP backlog cap
- `MAX_INFLIGHT_BATCHES_PER_JOB`: batch-job backlog cap
- `CAPTION_RETRY_LIMIT`: bounded caption retry count before terminal failure
- `CAPTION_TTL_SECONDS`: retention for caption records and one-shot publication markers
- `FRAME_BATCH_RETENTION_SECONDS`: successful batch retention window
- `FRAME_BATCH_FAILED_RETENTION_SECONDS`: failed batch retention window
- `ENABLE_DIRECT_CHAT_COMPLETIONS`: opt-in compatibility switch for `POST /v1/chat/completions` on `video-ingest-api`
- `PRESERVE_AUDIO_ARTIFACTS`: optional per-batch audio extraction
- `PRESERVE_CHUNK_ARTIFACTS`: optional chunk video retention
- `OPENCLAW_INSIGHTS_STREAM` and `OPENCLAW_ALERTS_STREAM`: skill/consumer compatibility stream names

## Local Development

Start Redis:

```bash
redis-server
```

Start the ingest API:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Start the ingest worker:

```bash
celery -A app.worker.celery_app worker --loglevel=info --concurrency=2
```

Start the captioner:

```bash
OPENCLAW_BUS_ENABLED=true python -m app.vlm_captioner.consumer
```

Run a skill from the repo root:

```bash
cd skills/package_delivery_alert
PYTHONPATH=../.. RTSP_URL=rtsp://... python skill.py
```

Run a consumer from the repo root:

```bash
cd consumers/discord_notifier
PYTHONPATH=../.. DISCORD_WEBHOOK_URL=https://... python consumer.py
```

## Adding New Downstream Consumers

For new detectors or sinks:

1. Reuse `common/event_bus.py` for consumer-group management.
2. Read `openclaw:insights` if you need compatibility with the current skills model.
3. Prefer `nova:captions` for new services that want the denser caption contract.
4. Publish alerts to `openclaw:alerts` if you want to interoperate with the existing notification consumers.

## Further Reading

- FastAPI docs once running: [http://localhost:8000/docs](http://localhost:8000/docs)
- [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Redis Streams](https://redis.io/docs/data-types/streams/)
