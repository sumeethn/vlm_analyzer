from __future__ import annotations

import logging
import signal
import time
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.services.openai_compat import ollama_to_openai_chat_completion
from app.services.vlm import file_to_base64, ollama_chat_vision
from app.state.captions import CaptionStore
from app.state.jobs import JobStore
from app.state.streams import StreamStore
from app.video_ingest.storage import FrameBatchStorage
from app.vlm_captioner.publisher import publish_caption_ready, publish_legacy_insight
from common.contracts.caption import CaptionReadyEvent, CaptionRecord
from common.contracts.frame_batch import FrameBatchManifest, FrameBatchReadyEvent
from common.event_bus import EventBus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("vlm_captioner")


def _caption_text(completion: dict[str, Any]) -> str:
    try:
        return str(completion["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        return ""


def _runtime_config_for_batch(
    settings: Settings,
    manifest: FrameBatchManifest,
    event: FrameBatchReadyEvent,
) -> tuple[str, str, dict[str, Any] | None]:
    processing_config = manifest.metadata.get("processing_config")
    if isinstance(processing_config, dict):
        model = processing_config.get("model")
        prompt = processing_config.get("prompt")
        if isinstance(model, str) and isinstance(prompt, str):
            options = processing_config.get("ollama_options")
            return model, prompt, options if isinstance(options, dict) else {}
    if event.job_id:
        job = JobStore(settings).get(event.job_id)
        if not job:
            raise RuntimeError(f"job not found for batch {event.batch_id}")
        return job["model"], job["prompt"], job.get("ollama_options") or {}
    if event.stream_id:
        stream = StreamStore(settings).get(event.stream_id)
        if not stream:
            raise RuntimeError(f"stream not found for batch {event.batch_id}")
        return stream["model"], stream["prompt"], stream.get("ollama_options") or {}
    raise RuntimeError(f"batch {event.batch_id} has neither job_id nor stream_id")


def _append_job_result(
    settings: Settings,
    manifest: FrameBatchManifest,
    record: CaptionRecord,
) -> CaptionRecord:
    if not manifest.job_id or record.job_result_recorded or not record.completion:
        return record
    store = JobStore(settings)
    completion = record.completion

    def _mutate(job: dict[str, Any]) -> dict[str, Any]:
        results = list(job.get("results", []))
        if not any(item.get("batch_id") == manifest.batch_id for item in results):
            results.append(
                {
                    "batch_id": manifest.batch_id,
                    "source_index": manifest.source_index,
                    "chunk_index": manifest.chunk_index,
                    "artifact_path": manifest.frame_paths[0] if manifest.frame_paths else "",
                    "artifact_paths": list(manifest.frame_paths),
                    "manifest_path": manifest.manifest_path,
                    "completion": completion,
                }
            )
            job["pending_batches"] = max(0, int(job.get("pending_batches", 0)) - 1)
        job["results"] = results
        job["chunks_done"] = max(int(job.get("chunks_done", 0)), len(results))
        if int(job.get("chunks_total", 0)) and int(job["chunks_done"]) >= int(job["chunks_total"]):
            job["status"] = "completed"
            job["error"] = None
        return job

    if store.update(manifest.job_id, _mutate):
        record.job_result_recorded = True
    return record


def _mark_job_failed(settings: Settings, job_id: str, record: CaptionRecord, error: str) -> CaptionRecord:
    if record.job_failure_recorded:
        return record
    store = JobStore(settings)
    marker = f"{settings.caption_key_prefix}job-failure:{record.batch_id}"

    def _mutate(job: dict[str, Any]) -> dict[str, Any]:
        job["status"] = "failed"
        job["error"] = error
        job["pending_batches"] = max(0, int(job.get("pending_batches", 0)) - 1)
        return job

    if store.update_once(
        job_id,
        marker=marker,
        marker_ttl_seconds=settings.caption_ttl_seconds,
        mutator=_mutate,
    )[1]:
        record.job_failure_recorded = True
    return record


def _mark_stream_batch_done(settings: Settings, stream_id: str, record: CaptionRecord) -> CaptionRecord:
    if record.stream_result_recorded:
        return record
    store = StreamStore(settings)
    marker = f"{settings.caption_key_prefix}stream-done:{record.batch_id}"

    def _mutate(stream: dict[str, Any]) -> dict[str, Any]:
        stream["pending_batches"] = max(0, int(stream.get("pending_batches", 0)) - 1)
        stream["last_error"] = None
        return stream

    if store.update_once(
        stream_id,
        marker=marker,
        marker_ttl_seconds=settings.caption_ttl_seconds,
        mutator=_mutate,
    )[1]:
        record.stream_result_recorded = True
    return record


def _mark_stream_batch_failed(
    settings: Settings,
    stream_id: str,
    record: CaptionRecord,
    error: str,
) -> CaptionRecord:
    if record.stream_failure_recorded:
        return record
    store = StreamStore(settings)
    marker = f"{settings.caption_key_prefix}stream-failure:{record.batch_id}"

    def _mutate(stream: dict[str, Any]) -> dict[str, Any]:
        stream["pending_batches"] = max(0, int(stream.get("pending_batches", 0)) - 1)
        stream["last_error"] = error
        return stream

    if store.update_once(
        stream_id,
        marker=marker,
        marker_ttl_seconds=settings.caption_ttl_seconds,
        mutator=_mutate,
    )[1]:
        record.stream_failure_recorded = True
    return record


def _infer_completion(
    settings: Settings,
    manifest: FrameBatchManifest,
    event: FrameBatchReadyEvent,
) -> tuple[dict[str, Any], str]:
    model, prompt, options = _runtime_config_for_batch(settings, manifest, event)
    image_paths = [Path(path) for path in manifest.frame_paths]
    images_b64 = [file_to_base64(path) for path in image_paths]
    ollama_body = ollama_chat_vision(
        base_url=settings.ollama_base_url,
        model=model,
        prompt=prompt,
        images_b64=images_b64,
        timeout_seconds=settings.ollama_timeout_seconds,
        options=options,
    )
    completion = ollama_to_openai_chat_completion(
        ollama_body=ollama_body,
        model=model,
        completion_id_prefix="chatcmpl-frame-batch",
    )
    return completion, model


def _ensure_publications(
    settings: Settings,
    captions: CaptionStore,
    storage: FrameBatchStorage,
    manifest: FrameBatchManifest,
    record: CaptionRecord,
) -> CaptionRecord:
    if not record.completion:
        raise RuntimeError(f"caption record {record.batch_id} is missing completion")
    completion = record.completion
    caption_text = record.caption_text or _caption_text(completion)
    if not record.completion_path:
        record.completion_path = storage.write_completion(manifest, completion)
        captions.save(record)

    if not record.caption_event_published:
        publish_caption_ready(
            settings,
            CaptionReadyEvent(
                batch_id=record.batch_id,
                job_id=record.job_id,
                stream_id=record.stream_id,
                source_index=record.source_index,
                chunk_index=record.chunk_index,
                model=record.model,
                manifest_path=record.manifest_path,
                caption_text=caption_text,
                completion_path=record.completion_path,
                created_at=time.time(),
            ),
        )
        record.caption_event_published = True
        captions.save(record)

    if not record.legacy_insight_published:
        publish_legacy_insight(
            settings,
            batch_id=record.batch_id,
            job_id=record.job_id,
            stream_id=record.stream_id,
            source_index=record.source_index,
            chunk_index=record.chunk_index,
            completion=completion,
        )
        record.legacy_insight_published = True
        captions.save(record)

    return record


def process_frame_batch(
    settings: Settings,
    event: FrameBatchReadyEvent,
) -> None:
    storage = FrameBatchStorage(settings)
    captions = CaptionStore(settings)
    manifest = FrameBatchManifest.read_json(Path(event.manifest_path))
    existing = captions.get(event.batch_id)
    if existing and existing.status == "completed":
        existing = _ensure_publications(settings, captions, storage, manifest, existing)
        if event.job_id and existing.completion:
            existing = _append_job_result(settings, manifest, existing)
        if event.stream_id:
            existing = _mark_stream_batch_done(settings, event.stream_id, existing)
        captions.save(existing)
        return

    if existing and existing.status == "failed" and existing.attempt >= settings.caption_retry_limit:
        if event.job_id:
            existing = _mark_job_failed(
                settings,
                event.job_id,
                existing,
                existing.error or "caption failed",
            )
        if event.stream_id:
            existing = _mark_stream_batch_failed(
                settings,
                event.stream_id,
                existing,
                existing.error or "caption failed",
            )
        captions.save(existing)
        return

    attempt = int(existing.attempt if existing else 0) + 1
    record = existing or CaptionRecord(
        batch_id=event.batch_id,
        manifest_path=event.manifest_path,
        model="",
        job_id=event.job_id,
        stream_id=event.stream_id,
        source_index=event.source_index,
        chunk_index=event.chunk_index,
        attempt=0,
        created_at=time.time(),
        updated_at=time.time(),
    )
    record.status = "processing"
    record.attempt = attempt
    captions.save(record)

    try:
        completion, model = _infer_completion(settings, manifest, event)
        record.model = model
        record.status = "completed"
        record.caption_text = _caption_text(completion)
        record.completion = completion
        record.completion_path = storage.write_completion(manifest, completion)
        record.error = None
        captions.save(record)
        record = _ensure_publications(settings, captions, storage, manifest, record)

        storage.mark_manifest(
            event.manifest_path,
            status="captioned",
            cleanup_after_ts=time.time() + settings.frame_batch_retention_seconds,
            caption_completed_at=time.time(),
            attempt=record.attempt,
        )
        if event.job_id:
            record = _append_job_result(settings, manifest, record)
        if event.stream_id:
            record = _mark_stream_batch_done(settings, event.stream_id, record)
        captions.save(record)
    except Exception as exc:
        if record.status == "completed" and record.completion:
            record.error = str(exc)
            captions.save(record)
            raise
        record.status = "failed"
        record.error = str(exc)
        captions.save(record)
        if attempt >= settings.caption_retry_limit:
            storage.mark_manifest(
                event.manifest_path,
                status="failed",
                cleanup_after_ts=time.time() + settings.frame_batch_failed_retention_seconds,
                attempt=record.attempt,
            )
            if event.job_id:
                record = _mark_job_failed(settings, event.job_id, record, str(exc))
            if event.stream_id:
                record = _mark_stream_batch_failed(settings, event.stream_id, record, str(exc))
            captions.save(record)
        raise


def run_consumer(settings: Settings) -> None:
    bus = EventBus(settings.redis_url, stream_maxlen=settings.frame_batch_stream_maxlen)
    bus.ensure_consumer_group(
        settings.frame_batch_ready_stream,
        settings.frame_batch_stream_group,
        start_id="0",
    )
    shutdown_requested = False
    claim_cursor = "0-0"

    def _handle_signal(sig: int, _frame: Any) -> None:
        nonlocal shutdown_requested
        logger.info("Shutdown signal received (%s).", signal.Signals(sig).name)
        shutdown_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    while not shutdown_requested:
        claim_cursor, claimed = bus.claim_stale(
            settings.frame_batch_ready_stream,
            settings.frame_batch_stream_group,
            settings.frame_batch_stream_consumer,
            min_idle_ms=settings.caption_claim_idle_ms,
            start_id=claim_cursor,
            count=10,
        )
        processed = False
        for msg_id, fields in claimed:
            processed = True
            event = FrameBatchReadyEvent.from_stream_fields(fields)
            try:
                process_frame_batch(settings, event)
            except Exception:
                logger.exception("caption processing failed for batch %s", event.batch_id)
                continue
            bus.ack(settings.frame_batch_ready_stream, settings.frame_batch_stream_group, msg_id)

        for msg_id, fields in bus.consume(
            settings.frame_batch_ready_stream,
            settings.frame_batch_stream_group,
            settings.frame_batch_stream_consumer,
            block_ms=settings.caption_poll_block_ms,
            count=1,
        ):
            processed = True
            event = FrameBatchReadyEvent.from_stream_fields(fields)
            try:
                process_frame_batch(settings, event)
            except Exception:
                logger.exception("caption processing failed for batch %s", event.batch_id)
                continue
            bus.ack(settings.frame_batch_ready_stream, settings.frame_batch_stream_group, msg_id)

        if not processed:
            time.sleep(0.1)


def main() -> None:
    run_consumer(get_settings())


if __name__ == "__main__":
    main()
