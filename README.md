# OpenClaw — Home Automation Platform

## Overview

OpenClaw is a mono-repo for event-driven home automation built around a **vision-language model (VLM)** microservice. Cameras stream video to the VLM analyzer, which publishes analysis results to a **Redis Streams** message bus. Skills subscribe to the bus, detect events of interest, and publish alerts. Consumers deliver those alerts to notification channels (Discord, etc.).

```
[IP Cameras / RTSP Streams]
         │
         ▼
[vlm_analyzer service]          ← FastAPI + Celery + Ollama
  Chunks video, runs VLM
         │
         ▼  xadd
  Redis Stream: openclaw:insights
         │
         │  xreadgroup (fan-out, one consumer group per skill)
   ┌─────┴──────────────────┐
   ▼                        ▼
[package_delivery_alert]  [vehicle_exit_monitor]  ...future skills
   Detect → publish alert
         │
         ▼  xadd
  Redis Stream: openclaw:alerts
         │
         ▼  xreadgroup
[discord_notifier]        ...future consumers (Slack, SMS, Home Assistant)
```

---

## Repository Layout

| Path | Role |
|------|------|
| `app/` | **vlm_analyzer** FastAPI service (API, worker, state, services) |
| `common/` | Shared SDK: `EventBus` class, event schema TypedDicts |
| `skills/` | One directory per OpenClaw skill |
| `skills/package_delivery_alert/` | Detects package deliveries on a front-yard camera |
| `consumers/` | One directory per notification consumer |
| `consumers/discord_notifier/` | Forwards alerts from any skill to a Discord webhook |
| `Dockerfile` | Image for vlm_analyzer API + Celery worker |
| `docker-compose.yml` | Full stack: Redis, vlm_analyzer, skills, consumers |

---

## Services

### vlm_analyzer (`app/`)

FastAPI microservice that processes video — live RTSP streams or local files — using a VLM via [Ollama](https://github.com/ollama/ollama). Celery workers handle async chunking and inference.

When `OPENCLAW_BUS_ENABLED=true` the worker publishes every VLM completion to the `openclaw:insights` Redis Stream immediately after storing it. This is opt-in so the service can run standalone without OpenClaw.

**Key API routes**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/v1/health` | Liveness and Redis connectivity |
| POST | `/v1/jobs` | Queue a file-based video batch job |
| GET | `/v1/jobs/{job_id}` | Job status and results |
| POST | `/v1/streams` | Register an RTSP stream for continuous analysis |
| GET | `/v1/streams` | List active streams |
| GET | `/v1/streams/{stream_id}` | Stream detail |
| DELETE | `/v1/streams/{stream_id}` | Stop a stream |
| GET | `/v1/insights` | Paginated insight list (optional `stream_id` filter) |
| GET | `/v1/streams/{stream_id}/insights` | Insights for one stream |
| POST | `/v1/chat/completions` | OpenAI-compatible vision endpoint (proxied to Ollama) |

### common/

Shared Python package used by all skills and consumers.

- **`event_bus.py`** — `EventBus` class wrapping Redis Streams (`xadd`, `xreadgroup`, `xack`, consumer group management).
- **`events.py`** — `InsightEvent` and `AlertEvent` TypedDicts documenting the fields present in each stream entry.

### Skills (`skills/`)

Each skill:
1. Registers its camera(s) with vlm_analyzer on startup via `POST /v1/streams`.
2. Subscribes to `openclaw:insights` as a dedicated consumer group.
3. Filters insights by its own `stream_id`, runs detection logic.
4. Publishes an `AlertEvent` to `openclaw:alerts` when an event is detected.
5. Deregisters its streams on shutdown.

**Available skills**

| Skill | Camera | Event detected |
|-------|--------|----------------|
| `package_delivery_alert` | Front yard / porch | Package left at the door |

### Consumers (`consumers/`)

Consumers subscribe to `openclaw:alerts` and forward alerts to external channels. They are decoupled from detection — swapping Discord for another channel requires only a new consumer, not changes to any skill.

**Available consumers**

| Consumer | Destination |
|----------|-------------|
| `discord_notifier` | Discord channel via incoming webhook |

---

## Message Bus Streams

| Stream | Producer | Consumer(s) | Content |
|--------|----------|-------------|---------|
| `openclaw:insights` | vlm_analyzer worker | Skills (one consumer group each) | VLM completion for every processed video chunk |
| `openclaw:alerts` | Skills | Consumers (one consumer group each) | Detected event with score, matched signals, and VLM excerpt |

---

## Quick Start (Docker Compose)

### Prerequisites

- Docker and Docker Compose
- [Ollama](https://github.com/ollama/ollama) running on the host with at least one vision model pulled:
  ```bash
  ollama pull llava
  ```

### 1. Configure skills and consumers

```bash
cp skills/package_delivery_alert/.env.example skills/package_delivery_alert/.env
cp consumers/discord_notifier/.env.example    consumers/discord_notifier/.env
```

Edit each `.env` file. Minimum required values:

**`skills/package_delivery_alert/.env`**
```
RTSP_URL=rtsp://user:password@192.168.1.100:554/stream1
```

**`consumers/discord_notifier/.env`**
```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN
```

### 2. Start the stack

```bash
docker compose up --build
```

Services started:

| Service | Address |
|---------|---------|
| vlm_analyzer API | http://localhost:8000 |
| Redis | localhost:6379 |
| package_delivery_alert | (no port — event-driven) |
| discord_notifier | (no port — event-driven) |

### 3. Verify

```bash
# API health
curl -sS http://localhost:8000/v1/health

# Active streams (the skill registers one on startup)
curl -sS http://localhost:8000/v1/streams

# Recent insights from the bus
redis-cli XLEN openclaw:insights

# Recent alerts
redis-cli XLEN openclaw:alerts
```

---

## Configuration Reference

### vlm_analyzer (`app/config.py`)

| Variable | Purpose | Default |
|----------|---------|---------|
| `REDIS_URL` | Redis for job/stream/insight state | `redis://localhost:6379/0` |
| `CELERY_BROKER_URL` | Celery broker | `redis://redis:6379/1` |
| `OLLAMA_BASE_URL` | Ollama HTTP base | `http://127.0.0.1:11434` |
| `OPENCLAW_BUS_ENABLED` | Publish insights to Redis Streams | `false` |
| `OPENCLAW_INSIGHTS_STREAM` | Stream key for VLM completions | `openclaw:insights` |
| `OPENCLAW_ALERTS_STREAM` | Stream key for detected alerts | `openclaw:alerts` |
| `OPENCLAW_STREAM_MAXLEN` | Maximum entries kept per stream | `10000` |
| `TEMP_DIR` | Temp workspace for video chunks | `/tmp/vlm_jobs` |
| `VIDEO_MOUNT` | Root for validated file sources | `/data/videos` |
| `ENABLE_NVDEC` | NVIDIA hardware decode | `false` |

### package_delivery_alert skill

| Variable | Purpose | Default |
|----------|---------|---------|
| `RTSP_URL` | Camera RTSP URL | **required** |
| `VLM_BASE_URL` | vlm_analyzer API URL | `http://localhost:8000` |
| `VLM_MODEL` | Ollama model | `llava` |
| `CHUNK_SECONDS` | Video window per analysis | `30` |
| `FRAMES_PER_CHUNK` | Frames sampled per chunk | `3` |
| `REDIS_URL` | Redis for event bus | `redis://localhost:6379/0` |
| `DETECTION_THRESHOLD` | Minimum score to fire an alert | `2` |
| `COOLDOWN_SECONDS` | Suppress duplicate alerts | `300` |

### discord_notifier consumer

| Variable | Purpose | Default |
|----------|---------|---------|
| `DISCORD_WEBHOOK_URL` | Discord incoming webhook | **required** |
| `REDIS_URL` | Redis for event bus | `redis://localhost:6379/0` |
| `DISCORD_MENTION` | Mention string prepended to alerts | *(none)* |

---

## Local Development (without Docker)

1. Start Redis:
   ```bash
   redis-server
   ```

2. Install vlm_analyzer dependencies and start the API:
   ```bash
   pip install -r requirements.txt
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. Start a Celery worker:
   ```bash
   OPENCLAW_BUS_ENABLED=true celery -A app.worker.celery_app worker --loglevel=info
   ```

4. Run a skill (from repo root so `common/` is on the path):
   ```bash
   cd skills/package_delivery_alert
   PYTHONPATH=../.. RTSP_URL=rtsp://... python skill.py
   ```

5. Run a consumer (from repo root):
   ```bash
   cd consumers/discord_notifier
   PYTHONPATH=../.. DISCORD_WEBHOOK_URL=https://... python consumer.py
   ```

---

## Adding a New Skill

1. Create `skills/<skill_name>/` with `skill.py`, `detector.py`, `config.py`, `.env.example`, `Dockerfile`.
2. In `skill.py`:
   - Register camera(s) with vlm_analyzer via `POST /v1/streams`.
   - Use `EventBus.ensure_consumer_group("openclaw:insights", "<skill_name>")`.
   - Loop on `EventBus.consume(...)`, filter by `stream_id`, run detection.
   - On detection call `EventBus.publish("openclaw:alerts", AlertEvent(...))`.
3. Add `AlertEvent` field `"skill": "<skill_name>"` so consumers can label it.
4. Add the service to `docker-compose.yml` with build context set to the repo root.
5. Add a row to the skills table in this README.

---

## Further Reading

- Interactive API docs (when the server is running): http://localhost:8000/docs
- [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Redis Streams](https://redis.io/docs/data-types/streams/)
