from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from app.config import get_settings
from app.services.chunker import (
    extract_representative_jpeg,
    segment_to_jpg,
    segment_to_mp4,
)
from app.services.openai_compat import ollama_to_openai_chat_completion
from app.services.vlm import file_to_base64, ollama_chat_vision
from app.state.insights import InsightStore
from app.state.jobs import JobStore, validate_file_under_mount
from app.state.streams import StreamStore
from app.worker.celery_app import celery_app

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


def _append_insight(
    *,
    job_id: str | None,
    stream_id: str | None,
    source_index: int,
    chunk_index: int,
    completion: dict,
) -> None:
    settings = get_settings()
    InsightStore(settings).append(
        stream_id=stream_id,
        job_id=job_id,
        source_index=source_index,
        chunk_index=chunk_index,
        completion=completion,
    )


@celery_app.task(name="process_video_job")
def process_video_job(job_id: str) -> None:
    settings = get_settings()
    store = JobStore(settings)
    job = store.get(job_id)
    if not job:
        logger.error("job not found: %s", job_id)
        return

    work_root = Path(settings.temp_dir) / job_id
    try:
        job["status"] = "running"
        store.save(job)

        results: list[dict] = []
        chunks_total = 0
        chunks_done = 0
        use_nvdec = settings.enable_nvdec

        sources = job["sources"]
        chunk_seconds = float(job["chunk_seconds"])
        chunk_format = job["chunk_format"]
        max_chunks = int(job["max_chunks_per_source"])
        model = job["model"]
        prompt = job["prompt"]
        options = job.get("ollama_options") or {}

        for si, source in enumerate(sources):
            kind = source["kind"]
            uri = source["uri"]
            if kind == "file":
                uri = validate_file_under_mount(uri, settings.video_mount)
                _verify_opencv_can_open(uri)

            src_dir = work_root / str(si)
            if chunk_format == "jpg":
                chunk_paths = segment_to_jpg(
                    uri=uri,
                    kind=kind,
                    out_dir=src_dir,
                    chunk_seconds=chunk_seconds,
                    use_nvdec=use_nvdec,
                    max_chunks=max_chunks,
                )
            else:
                chunk_paths = segment_to_mp4(
                    uri=uri,
                    kind=kind,
                    out_dir=src_dir,
                    chunk_seconds=chunk_seconds,
                    use_nvdec=use_nvdec,
                    max_chunks=max_chunks,
                )

            chunk_paths = chunk_paths[:max_chunks]
            chunks_total += len(chunk_paths)
            job["chunks_total"] = chunks_total
            job["chunks_done"] = chunks_done
            store.save(job)

            for ci, chunk_path in enumerate(chunk_paths):
                if chunk_format == "jpg":
                    image_path = chunk_path
                else:
                    image_path = chunk_path.with_suffix(".vlm.jpg")
                    extract_representative_jpeg(
                        Path(chunk_path),
                        Path(image_path),
                        use_nvdec=use_nvdec,
                    )

                image_b64 = file_to_base64(Path(image_path))
                ollama_body = ollama_chat_vision(
                    base_url=settings.ollama_base_url,
                    model=model,
                    prompt=prompt,
                    image_b64=image_b64,
                    timeout_seconds=settings.ollama_timeout_seconds,
                    options=options,
                )
                completion = ollama_to_openai_chat_completion(
                    ollama_body=ollama_body,
                    model=model,
                    completion_id_prefix="chatcmpl-chunk",
                )
                entry = {
                    "source_index": si,
                    "chunk_index": ci,
                    "artifact_path": str(chunk_path),
                    "completion": completion,
                }
                results.append(entry)
                chunks_done += 1
                job["results"] = results
                job["chunks_done"] = chunks_done
                store.save(job)
                _append_insight(
                    job_id=job_id,
                    stream_id=None,
                    source_index=si,
                    chunk_index=ci,
                    completion=completion,
                )

        job["status"] = "completed"
        job["error"] = None
        store.save(job)
    except Exception as e:
        logger.exception("job %s failed", job_id)
        job = store.get(job_id)
        if job:
            job["status"] = "failed"
            job["error"] = str(e)
            store.save(job)
        raise
    finally:
        if work_root.exists():
            shutil.rmtree(work_root, ignore_errors=True)


def _finalize_stream_stopped(stream_store: StreamStore, stream_id: str) -> None:
    s = stream_store.get(stream_id)
    if s:
        s["status"] = "stopped"
        stream_store.save(s)
    stream_store.remove_from_active(stream_id)
    wr = Path(get_settings().temp_dir) / "streams" / stream_id
    if wr.exists():
        shutil.rmtree(wr, ignore_errors=True)


@celery_app.task(name="process_rtsp_stream")
def process_rtsp_stream(stream_id: str) -> None:
    settings = get_settings()
    stream_store = StreamStore(settings)
    work_root = Path(settings.temp_dir) / "streams" / stream_id

    try:
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

            uri = s["rtsp_uri"]
            chunk_seconds = float(s["chunk_seconds"])
            chunk_format = s["chunk_format"]
            model = s["model"]
            prompt = s["prompt"]
            options = s.get("ollama_options") or {}
            use_nvdec = settings.enable_nvdec
            seq = int(s["chunk_seq"])

            iter_dir = work_root / f"iter_{seq}"
            if iter_dir.exists():
                shutil.rmtree(iter_dir, ignore_errors=True)
            iter_dir.mkdir(parents=True, exist_ok=True)

            try:
                if chunk_format == "jpg":
                    chunk_paths = segment_to_jpg(
                        uri=uri,
                        kind="rtsp",
                        out_dir=iter_dir,
                        chunk_seconds=chunk_seconds,
                        use_nvdec=use_nvdec,
                        max_chunks=1,
                    )
                else:
                    chunk_paths = segment_to_mp4(
                        uri=uri,
                        kind="rtsp",
                        out_dir=iter_dir,
                        chunk_seconds=chunk_seconds,
                        use_nvdec=use_nvdec,
                        max_chunks=1,
                    )
            except Exception as e:
                logger.exception("stream %s ffmpeg failed", stream_id)
                s["status"] = "failed"
                s["last_error"] = str(e)
                stream_store.save(s)
                stream_store.remove_from_active(stream_id)
                return

            if not chunk_paths:
                err = "no media captured from RTSP in this window"
                logger.error("stream %s: %s", stream_id, err)
                s["status"] = "failed"
                s["last_error"] = err
                stream_store.save(s)
                stream_store.remove_from_active(stream_id)
                return

            chunk_path = chunk_paths[0]
            if chunk_format == "jpg":
                image_path = chunk_path
            else:
                image_path = chunk_path.with_suffix(".vlm.jpg")
                extract_representative_jpeg(
                    Path(chunk_path),
                    Path(image_path),
                    use_nvdec=use_nvdec,
                )

            try:
                image_b64 = file_to_base64(Path(image_path))
                ollama_body = ollama_chat_vision(
                    base_url=settings.ollama_base_url,
                    model=model,
                    prompt=prompt,
                    image_b64=image_b64,
                    timeout_seconds=settings.ollama_timeout_seconds,
                    options=options,
                )
                completion = ollama_to_openai_chat_completion(
                    ollama_body=ollama_body,
                    model=model,
                    completion_id_prefix="chatcmpl-stream",
                )
            except Exception as e:
                logger.exception("stream %s VLM failed", stream_id)
                s["status"] = "failed"
                s["last_error"] = str(e)
                stream_store.save(s)
                stream_store.remove_from_active(stream_id)
                return
            finally:
                shutil.rmtree(iter_dir, ignore_errors=True)

            s = stream_store.get(stream_id)
            if not s:
                return
            if s["status"] == "stopping":
                _finalize_stream_stopped(stream_store, stream_id)
                return

            s["chunk_seq"] = seq + 1
            s["last_chunk_at"] = time.time()
            s["last_error"] = None
            stream_store.save(s)

            _append_insight(
                job_id=None,
                stream_id=stream_id,
                source_index=0,
                chunk_index=seq,
                completion=completion,
            )

    finally:
        if work_root.exists():
            shutil.rmtree(work_root, ignore_errors=True)
