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
    extract_spaced_jpegs_from_mp4,
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
        frames_per_chunk = int(job.get("frames_per_chunk") or 1)

        for si, source in enumerate(sources):
            kind = source["kind"]
            uri = source["uri"]
            if kind == "file":
                uri = validate_file_under_mount(uri, settings.video_mount)
                _verify_opencv_can_open(uri)

            src_dir = work_root / str(si)
            if chunk_format == "jpg":
                frame_paths = segment_to_jpg(
                    uri=uri,
                    kind=kind,
                    out_dir=src_dir,
                    chunk_seconds=chunk_seconds,
                    use_nvdec=use_nvdec,
                    frames_per_chunk=frames_per_chunk,
                    max_chunks=max_chunks,
                )
                groups: list[list[Path]] = []
                for i in range(0, len(frame_paths), frames_per_chunk):
                    g = frame_paths[i : i + frames_per_chunk]
                    if len(g) == frames_per_chunk:
                        groups.append(g)
                groups = groups[:max_chunks]
            else:
                mp4_paths = segment_to_mp4(
                    uri=uri,
                    kind=kind,
                    out_dir=src_dir,
                    chunk_seconds=chunk_seconds,
                    use_nvdec=use_nvdec,
                    max_chunks=max_chunks,
                )
                groups = []
                for ci, mp4_path in enumerate(mp4_paths):
                    frame_dir = src_dir / f"chunk_{ci:06d}_frames"
                    jpgs = extract_spaced_jpegs_from_mp4(
                        Path(mp4_path),
                        out_dir=frame_dir,
                        stem="f",
                        n=frames_per_chunk,
                        use_nvdec=use_nvdec,
                    )
                    groups.append(jpgs)

            chunks_total += len(groups)
            job["chunks_total"] = chunks_total
            job["chunks_done"] = chunks_done
            store.save(job)

            for ci, frame_group in enumerate(groups):
                images_b64 = [file_to_base64(p) for p in frame_group]
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
                    completion_id_prefix="chatcmpl-chunk",
                )
                entry = {
                    "source_index": si,
                    "chunk_index": ci,
                    "artifact_path": str(frame_group[0]),
                    "artifact_paths": [str(p) for p in frame_group],
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
            frames_per_chunk = int(s.get("frames_per_chunk") or 1)
            seq = int(s["chunk_seq"])

            iter_dir = work_root / f"iter_{seq}"
            if iter_dir.exists():
                shutil.rmtree(iter_dir, ignore_errors=True)
            iter_dir.mkdir(parents=True, exist_ok=True)

            try:
                if chunk_format == "jpg":
                    frame_paths = segment_to_jpg(
                        uri=uri,
                        kind="rtsp",
                        out_dir=iter_dir,
                        chunk_seconds=chunk_seconds,
                        use_nvdec=use_nvdec,
                        frames_per_chunk=frames_per_chunk,
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

            if chunk_format == "jpg":
                if len(frame_paths) < frames_per_chunk:
                    err = (
                        f"expected {frames_per_chunk} frame(s) from RTSP in this window, "
                        f"got {len(frame_paths)}"
                    )
                    logger.error("stream %s: %s", stream_id, err)
                    s["status"] = "failed"
                    s["last_error"] = err
                    stream_store.save(s)
                    stream_store.remove_from_active(stream_id)
                    return
                frame_group = frame_paths[:frames_per_chunk]
            else:
                if not chunk_paths:
                    err = "no media captured from RTSP in this window"
                    logger.error("stream %s: %s", stream_id, err)
                    s["status"] = "failed"
                    s["last_error"] = err
                    stream_store.save(s)
                    stream_store.remove_from_active(stream_id)
                    return
                frame_group = extract_spaced_jpegs_from_mp4(
                    Path(chunk_paths[0]),
                    out_dir=iter_dir / "vlm_frames",
                    stem="f",
                    n=frames_per_chunk,
                    use_nvdec=use_nvdec,
                )

            try:
                images_b64 = [file_to_base64(Path(p)) for p in frame_group]
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
