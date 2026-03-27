from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.concurrency import run_in_threadpool

from app.config import Settings, get_settings
from app.schemas.chat import ChatCompletionRequest
from app.schemas.insights_schema import (
    InsightRecord,
    InsightsListResponse,
    JobResultsResponse,
)
from app.schemas.jobs import (
    JobAcceptedResponse,
    JobDetailResponse,
    JobStatus,
    MediaSource,
    SourceKind,
    StartProcessingRequest,
)
from app.schemas.streams import (
    CreateStreamRequest,
    StreamAcceptedResponse,
    StreamDetailResponse,
    StreamListItem,
)
from app.services.openai_compat import ollama_to_openai_chat_completion
from app.services.vlm import (
    extract_text_and_image_b64_from_openai_messages,
    ollama_chat_vision,
)
from app.state.insights import InsightStore
from app.state.jobs import JobStore
from app.state.streams import StreamStore
from app.worker.tasks import process_rtsp_stream, process_video_job

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/v1/health")
async def health(settings: Settings = Depends(get_settings)) -> dict:
    store = JobStore(settings)
    redis_ok = store.ping()
    return {"status": "ok" if redis_ok else "degraded", "redis": redis_ok}


@router.post(
    "/v1/jobs",
    response_model=JobAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_job(
    body: StartProcessingRequest,
    response: Response,
    settings: Settings = Depends(get_settings),
) -> JobAcceptedResponse:
    if len(body.sources) > settings.max_sources_per_job:
        raise HTTPException(status_code=400, detail="too many sources for one job")
    if (
        body.chunk_seconds > settings.max_chunk_seconds
        or body.chunk_seconds < settings.min_chunk_seconds
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"chunk_seconds must be between {settings.min_chunk_seconds} "
                f"and {settings.max_chunk_seconds}"
            ),
        )
    for src in body.sources:
        if src.kind != SourceKind.file:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Only file sources are allowed on /v1/jobs; "
                    "use POST /v1/streams for RTSP."
                ),
            )

    store = JobStore(settings)
    initial = {
        "status": JobStatus.queued.value,
        "model": body.model,
        "prompt": body.prompt,
        "chunk_seconds": body.chunk_seconds,
        "chunk_format": body.chunk_format,
        "sources": [s.model_dump(mode="json") for s in body.sources],
        "ollama_options": body.ollama_options,
        "max_chunks_per_source": body.max_chunks_per_source,
        "chunks_total": 0,
        "chunks_done": 0,
        "results": [],
        "error": None,
    }
    job_id = store.create_job(initial)
    process_video_job.delay(job_id)
    response.headers["Location"] = f"/v1/jobs/{job_id}"
    return JobAcceptedResponse(job_id=job_id)


@router.get("/v1/jobs/{job_id}", response_model=JobDetailResponse)
async def get_job(job_id: str, settings: Settings = Depends(get_settings)) -> JobDetailResponse:
    store = JobStore(settings)
    data = store.get(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="job not found")
    return JobDetailResponse(
        job_id=data["job_id"],
        status=JobStatus(data["status"]),
        model=data["model"],
        prompt=data["prompt"],
        chunk_seconds=float(data["chunk_seconds"]),
        chunk_format=data["chunk_format"],
        sources=[MediaSource.model_validate(s) for s in data["sources"]],
        chunks_total=int(data.get("chunks_total", 0)),
        chunks_done=int(data.get("chunks_done", 0)),
        results=list(data.get("results", [])),
        error=data.get("error"),
    )


@router.get("/v1/jobs/{job_id}/results", response_model=JobResultsResponse)
async def get_job_results(
    job_id: str,
    settings: Settings = Depends(get_settings),
) -> JobResultsResponse:
    store = JobStore(settings)
    data = store.get(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="job not found")
    return JobResultsResponse(
        job_id=job_id,
        results=list(data.get("results", [])),
    )


def _chunk_bounds(settings: Settings, chunk_seconds: float) -> None:
    if (
        chunk_seconds > settings.max_chunk_seconds
        or chunk_seconds < settings.min_chunk_seconds
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"chunk_seconds must be between {settings.min_chunk_seconds} "
                f"and {settings.max_chunk_seconds}"
            ),
        )


@router.post(
    "/v1/streams",
    response_model=StreamAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def register_stream(
    body: CreateStreamRequest,
    response: Response,
    settings: Settings = Depends(get_settings),
) -> StreamAcceptedResponse:
    if not body.rtsp_url.lower().startswith("rtsp://"):
        raise HTTPException(
            status_code=400,
            detail="rtsp_url must start with rtsp://",
        )
    _chunk_bounds(settings, body.chunk_seconds)

    stream_store = StreamStore(settings)
    stream_id = stream_store.create_stream(
        {
            "rtsp_uri": body.rtsp_url,
            "status": "active",
            "model": body.model,
            "prompt": body.prompt,
            "chunk_seconds": body.chunk_seconds,
            "chunk_format": body.chunk_format,
            "ollama_options": body.ollama_options,
        }
    )
    process_rtsp_stream.delay(stream_id)
    response.headers["Location"] = f"/v1/streams/{stream_id}"
    return StreamAcceptedResponse(stream_id=stream_id)


@router.get("/v1/streams", response_model=list[StreamListItem])
async def list_streams(settings: Settings = Depends(get_settings)) -> list[StreamListItem]:
    stream_store = StreamStore(settings)
    stream_store.reconcile_active_set()
    items: list[StreamListItem] = []
    for sid in stream_store.list_active_ids():
        s = stream_store.get(sid)
        if not s:
            continue
        if s.get("status") not in ("active", "stopping"):
            continue
        items.append(
            StreamListItem(
                stream_id=s["stream_id"],
                rtsp_uri=s["rtsp_uri"],
                status=s["status"],
                chunk_seq=int(s.get("chunk_seq", 0)),
                last_chunk_at=s.get("last_chunk_at"),
                last_error=s.get("last_error"),
            )
        )
    return items


@router.get("/v1/streams/{stream_id}", response_model=StreamDetailResponse)
async def get_stream(
    stream_id: str,
    settings: Settings = Depends(get_settings),
) -> StreamDetailResponse:
    stream_store = StreamStore(settings)
    s = stream_store.get(stream_id)
    if not s:
        raise HTTPException(status_code=404, detail="stream not found")
    return StreamDetailResponse(
        stream_id=s["stream_id"],
        rtsp_uri=s["rtsp_uri"],
        status=s["status"],
        model=s["model"],
        prompt=s["prompt"],
        chunk_seconds=float(s["chunk_seconds"]),
        chunk_format=s["chunk_format"],
        chunk_seq=int(s.get("chunk_seq", 0)),
        last_chunk_at=s.get("last_chunk_at"),
        created_at=float(s.get("created_at", 0)),
        updated_at=float(s.get("updated_at", 0)),
        last_error=s.get("last_error"),
    )


@router.delete("/v1/streams/{stream_id}", status_code=status.HTTP_204_NO_CONTENT)
async def stop_stream(
    stream_id: str,
    settings: Settings = Depends(get_settings),
) -> Response:
    stream_store = StreamStore(settings)
    s = stream_store.get(stream_id)
    if not s:
        raise HTTPException(status_code=404, detail="stream not found")
    if s["status"] == "active":
        s["status"] = "stopping"
        stream_store.save(s)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _build_insights_response(
    *,
    stream_id: str | None,
    limit: int,
    offset: int,
    settings: Settings,
) -> InsightsListResponse:
    store = InsightStore(settings)
    rows, _ = store.list_insights(stream_id=stream_id, limit=limit, offset=offset)
    recs: list[InsightRecord] = []
    for r in rows:
        try:
            recs.append(InsightRecord.model_validate(r))
        except Exception:
            continue
    return InsightsListResponse(insights=recs, total_returned=len(recs))


@router.get("/v1/insights", response_model=InsightsListResponse)
async def list_insights(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    stream_id: str | None = None,
    settings: Settings = Depends(get_settings),
) -> InsightsListResponse:
    return _build_insights_response(
        stream_id=stream_id,
        limit=limit,
        offset=offset,
        settings=settings,
    )


@router.get(
    "/v1/streams/{stream_id}/insights",
    response_model=InsightsListResponse,
)
async def list_stream_insights(
    stream_id: str,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    settings: Settings = Depends(get_settings),
) -> InsightsListResponse:
    stream_store = StreamStore(settings)
    if not stream_store.get(stream_id):
        raise HTTPException(status_code=404, detail="stream not found")
    return _build_insights_response(
        stream_id=stream_id,
        limit=limit,
        offset=offset,
        settings=settings,
    )


def _ollama_option_overrides(body: ChatCompletionRequest) -> dict | None:
    opts: dict = {}
    if body.temperature is not None:
        opts["temperature"] = body.temperature
    if body.max_tokens is not None:
        opts["num_predict"] = body.max_tokens
    return opts or None


def _chat_completion_sync(body: ChatCompletionRequest, settings: Settings) -> dict:
    if body.stream:
        raise ValueError("stream not supported")

    messages = [m.model_dump(mode="json") for m in body.messages]
    text, image_b64 = extract_text_and_image_b64_from_openai_messages(messages)
    if not text:
        raise ValueError("user message text is required")

    overrides = _ollama_option_overrides(body)

    if image_b64:
        ollama_body = ollama_chat_vision(
            base_url=settings.ollama_base_url,
            model=body.model,
            prompt=text,
            image_b64=image_b64,
            timeout_seconds=settings.ollama_timeout_seconds,
            options=overrides,
        )
    else:
        payload: dict = {
            "model": body.model,
            "messages": [{"role": "user", "content": text}],
            "stream": False,
        }
        if overrides:
            for k, v in overrides.items():
                payload[k] = v
        with httpx.Client(timeout=settings.ollama_timeout_seconds) as client:
            r = client.post(
                f"{settings.ollama_base_url.rstrip('/')}/api/chat",
                json=payload,
            )
            r.raise_for_status()
            ollama_body = r.json()

    return ollama_to_openai_chat_completion(
        ollama_body=ollama_body,
        model=body.model,
        completion_id_prefix="chatcmpl",
    )


@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    settings: Settings = Depends(get_settings),
) -> dict:
    if body.stream:
        raise HTTPException(status_code=400, detail="stream=false is required")

    try:
        return await run_in_threadpool(_chat_completion_sync, body, settings)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except httpx.HTTPError as e:
        logger.exception("chat completion failed (HTTP)")
        raise HTTPException(status_code=502, detail=f"upstream error: {e}") from e
    except Exception as e:
        logger.exception("chat completion failed")
        raise HTTPException(status_code=502, detail=f"upstream error: {e}") from e
