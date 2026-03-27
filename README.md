# Video VLM analyzer (microservice)

## Overview

This repository contains a **FastAPI** microservice that processes video with a **vision-language model (VLM)** backed by **[Ollama](https://github.com/ollama/ollama)**. It supports:

- **Batch jobs** over local video files (chunked, with per-chunk VLM completions).
- **RTSP streams** registered for continuous chunked analysis.
- **Insights** persisted in **Redis** and listed via HTTP.
- An **OpenAI-compatible** `POST /v1/chat/completions` path for chat or single-image vision calls proxied to Ollama.

Asynchronous work uses **Celery** workers with **Redis** as the broker and result backend.

## Status

Active development. APIs and behavior may change without a formal versioning policy beyond the FastAPI app version (`0.1.0` in `app/main.py`).

## Contact

For questions, bugs, or requests, use your team’s usual issue tracker or chat channel for this repository. If none is published yet, open a discussion or issue in the hosting platform’s issue tracker and tag the maintainers you work with.

## Further documentation

- **Interactive API docs** (when the server is running): `http://localhost:8000/docs` (Swagger UI) and `http://localhost:8000/redoc`.
- **FastAPI**: https://fastapi.tiangolo.com/
- **Ollama API**: https://github.com/ollama/ollama/blob/main/docs/api.md

## Prerequisites

- **Docker** and **Docker Compose** (recommended), or Python **3.12+** with dependencies from `requirements.txt`.
- **Ollama** running somewhere reachable from the API and worker (default: `http://127.0.0.1:11434`). The provided `docker-compose.yml` points at `http://host.docker.internal:11434` so the host’s Ollama is used from containers.
- **Redis** for job/stream state and Celery (included in Compose).

## Quick start (Docker Compose)

From the repository root:

```bash
docker compose up --build
```

Services:

- API: `http://localhost:8000`
- Redis: `localhost:6379`

The Compose file mounts `./samples` read-only at `/data/videos` inside the API and worker. Place test videos there or adjust the `VIDEO_MOUNT` / volume mapping.

### Health check

```bash
curl -sS http://localhost:8000/v1/health
```

You should see JSON with `status` and `redis`. If Redis is unreachable, `status` may be `degraded`.

### Example: start a file-based job

Paths in `sources` must be **absolute paths inside the worker container** under `VIDEO_MOUNT` (default `/data/videos` in Docker). With the default Compose mount, a file `samples/clips/demo.mp4` on the host is `/data/videos/clips/demo.mp4` in the container.

```bash
curl -sS -X POST http://localhost:8000/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "sources": [{"uri": "/data/videos/clips/demo.mp4", "kind": "file"}],
    "model": "llava",
    "chunk_seconds": 10,
    "chunk_format": "jpg",
    "prompt": "Describe what you see in this video segment."
  }'
```

Poll job status (replace `JOB_ID` from the response):

```bash
curl -sS "http://localhost:8000/v1/jobs/JOB_ID"
```

## Configuration

Settings are loaded from the environment (and optionally a `.env` file). Names map from the fields in `app/config.py` (for example `redis_url` → `REDIS_URL`).

| Variable | Purpose | Default |
|----------|---------|---------|
| `REDIS_URL` | Redis DB for job/stream/insight state | `redis://localhost:6379/0` |
| `CELERY_BROKER_URL` | Celery broker | `redis://localhost:6379/1` |
| `OLLAMA_BASE_URL` | Ollama HTTP base | `http://127.0.0.1:11434` |
| `TEMP_DIR` | Temp workspace for chunks | `/tmp/vlm_jobs` |
| `VIDEO_MOUNT` | Root directory validated for file sources | `/data/videos` |
| `ENABLE_NVDEC` | Hardware decode toggle (see code) | `false` |

Optional tuning (limits, TTLs, insight list caps) is also defined in `app/config.py`.

## Local development (without Docker)

1. Start Redis locally (`redis-server` or a container).
2. Install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Ensure Ollama is running and models are pulled (for example `llava` or your chosen VLM).
4. Run the API:

   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. In another terminal, run a worker from the same environment and working directory:

   ```bash
   celery -A app.worker.celery_app worker --loglevel=info
   ```

## Project layout

| Path | Role |
|------|------|
| `app/main.py` | FastAPI application entry |
| `app/api/routes.py` | HTTP routes (`/v1/*`) |
| `app/worker/` | Celery app and tasks |
| `app/services/` | VLM/Ollama, chunking, OpenAI-compat helpers |
| `app/state/` | Redis-backed stores |
| `app/schemas/` | Pydantic request/response models |
| `docker-compose.yml` | API, worker, Redis stack |
| `Dockerfile` | API/worker image (Python 3.12, FFmpeg) |

## API summary

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/health` | Liveness and Redis connectivity |
| `POST` | `/v1/jobs` | Queue file-based video job |
| `GET` | `/v1/jobs/{job_id}` | Job status and results metadata |
| `GET` | `/v1/jobs/{job_id}/results` | Job results |
| `POST` | `/v1/streams` | Register RTSP stream processing |
| `GET` | `/v1/streams` | List active streams |
| `GET` | `/v1/streams/{stream_id}` | Stream detail |
| `DELETE` | `/v1/streams/{stream_id}` | Stop stream |
| `GET` | `/v1/insights` | List insights (optional `stream_id` filter) |
| `GET` | `/v1/streams/{stream_id}/insights` | Insights for one stream |
| `POST` | `/v1/chat/completions` | OpenAI-style chat/vision (non-streaming) |

Prefer `/docs` for authoritative request bodies and response shapes.
