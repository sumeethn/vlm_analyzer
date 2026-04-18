from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.services.chunker import source_has_audio
from app.state.jobs import JobStore, validate_file_under_mount
from app.state.streams import StreamStore
from app.video_ingest.media_pipeline import create_file_batches, create_stream_batch
from app.video_ingest.storage import FrameBatchStorage
from app.worker.celery_app import celery_app
from common.event_bus import EventBus

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

logger = logging.getLogger(__name__)


def _verify_opencv_can_open(path: str) -> None:
    if cv2 is None:
        return
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise RuntimeError("OpenCV could not open video file for reading")
    finally:
        cap.release()


def _publish_frame_batch(event_fields: dict[str, str]) -> None:
    settings = get_settings()
    bus = EventBus(settings.redis_url, stream_maxlen=settings.frame_batch_stream_maxlen)
    bus.publish(settings.frame_batch_ready_stream, event_fields)


def _wait_for_job_capacity(
    store: JobStore,
    settings: Any,
    job_id: str,
    max_inflight: int,
) -> dict[str, Any] | None:
    """
    Poll until inflight batch count drops below *max_inflight*.

    Raises ``TimeoutError`` if the backpressure deadline is exceeded; this is
    appropriate for finite jobs because exceeding the deadline indicates a
    stalled captioner that should not accept more work.
    """
    deadline = time.monotonic() + settings.max_backpressure_wait_seconds
    while True:
        job = store.get(job_id)
        if not job:
            return None
        if job.get("status") in {"failed", "completed"}:
            return job
        if int(job.get("pending_batches", 0)) < max_inflight:
            return job
        if time.monotonic() >= deadline:
            raise TimeoutError(f"job {job_id} exceeded backpressure wait limit")
        time.sleep(settings.backpressure_poll_seconds)


def _wait_for_stream_capacity(
    stream_store: StreamStore,
    settings: Any,
    stream_id: str,
    max_inflight: int,
    chunk_seconds: float,
) -> dict[str, Any] | None:
    """
    Poll until inflight batch count drops below *max_inflight*.

    For RTSP streams this never raises — if the captioner is backlogged we
    log a warning and return the current stream state so the caller can skip
    the current chunk interval rather than killing the stream worker.
    """
    deadline = time.monotonic() + settings.max_backpressure_wait_seconds
    while True:
        stream = stream_store.get(stream_id)
        if not stream:
            return None
        if stream.get("status") in {"stopping", "stopped", "failed"}:
            return stream
        if int(stream.get("pending_batches", 0)) < max_inflight:
            return stream
        if time.monotonic() >= deadline:
            logger.warning(
                "stream %s backpressure timeout reached (%ss); skipping batch for this cycle",
                stream_id,
                settings.max_backpressure_wait_seconds,
            )
            # Return the current state tagged so the caller knows to skip.
            return {**stream, "_backpressure_skip": True}
        time.sleep(min(max(chunk_seconds, settings.backpressure_poll_seconds), 5.0))


def _finalize_stream_stopped(
    stream_store: StreamStore,
    stream_id: str,
) -> None:
    stream_store.update(
        stream_id,
        lambda existing: {**existing, "status": "stopped"},
    )
    stream_store.remove_from_active(stream_id)
    wr = Path(get_settings().temp_dir) / "streams" / stream_id
    if wr.exists():
        shutil.rmtree(wr, ignore_errors=True)


# ── Publication counter mutator (hoisted out of the publish loop) ─────────────

def _record_publication_mutator(existing: dict[str, Any]) -> dict[str, Any]:
    if existing.get("status") == "failed":
        return existing
    existing["pending_batches"] = int(existing.get("pending_batches", 0)) + 1
    existing["published_batches"] = int(existing.get("published_batches", 0)) + 1
    return existing


@celery_app.task(name="process_video_job")
def process_video_job(job_id: str) -> None:
    settings = get_settings()
    store = JobStore(settings)
    storage = FrameBatchStorage(settings)
    job = store.get(job_id)
    if not job:
        logger.error("job not found: %s", job_id)
        return

    work_root = Path(settings.temp_dir) / job_id
    try:
        storage.cleanup_expired_batches()

        def _mark_running(existing: dict[str, Any]) -> dict[str, Any]:
            existing["status"] = "running"
            existing.setdefault("results", [])
            existing.setdefault("pending_batches", 0)
            existing.setdefault("published_batches", 0)
            existing.setdefault("chunks_total", 0)
            existing.setdefault("chunks_done", 0)
            existing["error"] = None
            return existing

        # Use the returned updated dict so that default-initialised fields
        # (e.g. pending_batches, results) are present for the rest of the task.
        updated_job = store.update(job_id, _mark_running)
        if updated_job is None:
            logger.error("job %s disappeared after initial get", job_id)
            return
        job = updated_job

        sources = job["sources"]
        chunk_seconds = float(job["chunk_seconds"])
        chunk_format = job["chunk_format"]
        max_chunks = int(job["max_chunks_per_source"])
        frames_per_chunk = int(job.get("frames_per_chunk") or 1)
        processing_config = {
            "model": job["model"],
            "prompt": job["prompt"],
            "ollama_options": job.get("ollama_options") or {},
        }

        manifests_by_source: list[list[Any]] = []
        total_batches = 0
        for si, source in enumerate(sources):
            kind = source["kind"]
            uri = source["uri"]
            if kind == "file":
                uri = validate_file_under_mount(uri, settings.video_mount)
                _verify_opencv_can_open(uri)
            manifests = create_file_batches(
                settings=settings,
                storage=storage,
                source_uri=uri,
                source_kind=kind,
                source_index=si,
                job_id=job_id,
                chunk_seconds=chunk_seconds,
                frames_per_chunk=frames_per_chunk,
                max_chunks=max_chunks,
                chunk_format=chunk_format,
                work_dir=work_root,
                processing_config=processing_config,
            )
            manifests_by_source.append(manifests)
            total_batches += len(manifests)

        def _set_total(existing: dict[str, Any]) -> dict[str, Any]:
            existing["chunks_total"] = total_batches
            if total_batches == 0 and existing.get("status") != "failed":
                existing["status"] = "completed"
            return existing

        store.update(job_id, _set_total)

        for manifests in manifests_by_source:
            for manifest in manifests:
                job = _wait_for_job_capacity(
                    store,
                    settings,
                    job_id,
                    settings.max_inflight_batches_per_job,
                )
                if not job:
                    return
                if job.get("status") in {"failed", "completed"}:
                    return
                _publish_frame_batch(storage.manifest_to_event(manifest).to_stream_fields())
                store.update(job_id, _record_publication_mutator)
    except Exception as e:
        logger.exception("job %s failed", job_id)

        def _mark_failed(existing: dict[str, Any]) -> dict[str, Any]:
            existing["status"] = "failed"
            existing["error"] = str(e)
            return existing

        store.update(job_id, _mark_failed)
        raise
    finally:
        if work_root.exists():
            shutil.rmtree(work_root, ignore_errors=True)


@celery_app.task(name="process_rtsp_stream")
def process_rtsp_stream(stream_id: str) -> None:
    settings = get_settings()
    stream_store = StreamStore(settings)
    storage = FrameBatchStorage(settings)
    work_root = Path(settings.temp_dir) / "streams" / stream_id

    try:
        storage.cleanup_expired_batches()

        # ── Probe audio once per task invocation ─────────────────────────────
        # Running ffprobe in an async FastAPI handler would block the event
        # loop.  We do it here instead, before the main loop, and cache the
        # result in a local variable for the lifetime of this task.
        initial_s = stream_store.get(stream_id)
        if not initial_s:
            logger.info("stream %s not found on startup, exiting", stream_id)
            return
        uri = initial_s["rtsp_uri"]
        has_audio = source_has_audio(uri, "rtsp")
        logger.info("stream %s audio detected: %s", stream_id, has_audio)
        # Persist so the stored record stays accurate (e.g. for the API GET).
        stream_store.update(stream_id, lambda s: {**s, "has_audio": has_audio})

        while True:
            s = stream_store.get(stream_id)
            if not s:
                logger.info("stream %s deleted, exiting worker", stream_id)
                stream_store.remove_from_active(stream_id)
                return

            if s["status"] == "stopped":
                stream_store.remove_from_active(stream_id)
                return

            if s["status"] == "failed":
                stream_store.remove_from_active(stream_id)
                return

            if s["status"] == "stopping":
                _finalize_stream_stopped(stream_store, stream_id)
                return

            if s["status"] != "active":
                return

            chunk_seconds = float(s["chunk_seconds"])
            s = _wait_for_stream_capacity(
                stream_store,
                settings,
                stream_id,
                settings.max_inflight_batches_per_stream,
                chunk_seconds,
            )
            if not s:
                return
            if s.get("_backpressure_skip"):
                # Captioner is backlogged: sleep one chunk interval and retry.
                time.sleep(chunk_seconds)
                continue
            if s.get("status") in {"stopping", "stopped", "failed"}:
                if s.get("status") == "stopping":
                    _finalize_stream_stopped(stream_store, stream_id)
                return

            chunk_format = s["chunk_format"]
            frames_per_chunk = int(s.get("frames_per_chunk") or 1)
            seq = int(s["chunk_seq"])
            processing_config = {
                "model": s["model"],
                "prompt": s["prompt"],
                "ollama_options": s.get("ollama_options") or {},
            }

            manifest = None
            exc: Exception | None = None
            for attempt in range(3):
                try:
                    manifest = create_stream_batch(
                        settings=settings,
                        storage=storage,
                        stream_id=stream_id,
                        source_uri=uri,
                        chunk_seconds=chunk_seconds,
                        frames_per_chunk=frames_per_chunk,
                        chunk_format=chunk_format,
                        chunk_index=seq,
                        work_dir=work_root / f"iter_{seq}",
                        has_audio=has_audio,
                        processing_config=processing_config,
                    )
                    exc = None
                    break
                except Exception as err:
                    exc = err
                    if attempt < 2:
                        time.sleep(2**attempt)

            if exc is not None:
                logger.exception("stream %s ingest failed after retries", stream_id)

                def _mark_stream_failed(existing: dict[str, Any]) -> dict[str, Any]:
                    existing["status"] = "failed"
                    existing["last_error"] = str(exc)
                    return existing

                stream_store.update(stream_id, _mark_stream_failed)
                stream_store.remove_from_active(stream_id)
                return

            if manifest is None:
                logger.warning("stream %s seq=%s produced no frame batch", stream_id, seq)
                time.sleep(max(chunk_seconds, 0.5))
                continue

            _publish_frame_batch(storage.manifest_to_event(manifest).to_stream_fields())

            def _record_stream_publication(existing: dict[str, Any]) -> dict[str, Any]:
                existing["chunk_seq"] = max(int(existing.get("chunk_seq", 0)), seq + 1)
                existing["last_chunk_at"] = time.time()
                existing["last_error"] = None
                existing["pending_batches"] = int(existing.get("pending_batches", 0)) + 1
                return existing

            if stream_store.update(stream_id, _record_stream_publication) is None:
                return

    finally:
        if work_root.exists():
            shutil.rmtree(work_root, ignore_errors=True)
