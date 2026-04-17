from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path

from app.config import Settings
from app.services.chunker import (
    extract_spaced_jpegs_from_mp4,
    segment_to_jpg,
    segment_to_mp4,
    source_has_audio,
)
from common.contracts.frame_batch import FrameBatchManifest

from app.video_ingest.storage import FrameBatchStorage

logger = logging.getLogger(__name__)


def _extract_audio_from_chunk(mp4_path: Path, audio_path: Path) -> Path | None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-i",
            str(mp4_path),
            "-vn",
            "-acodec",
            "aac",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if r.returncode != 0:
        logger.warning("audio extract failed for %s: %s", mp4_path, (r.stderr or "").strip())
        return None
    return audio_path


def _source_scope_ids(job_id: str | None, stream_id: str | None) -> tuple[str, str]:
    if stream_id:
        return "streams", stream_id
    if job_id:
        return "jobs", job_id
    raise ValueError("either job_id or stream_id is required")


def _optional_chunk_artifact_needed(settings: Settings, chunk_format: str) -> bool:
    return settings.preserve_chunk_artifacts or chunk_format == "mp4" or settings.preserve_audio_artifacts


def _frame_groups(frame_paths: list[Path], frames_per_chunk: int, max_chunks: int | None) -> list[list[Path]]:
    groups: list[list[Path]] = []
    for i in range(0, len(frame_paths), frames_per_chunk):
        group = frame_paths[i : i + frames_per_chunk]
        if len(group) == frames_per_chunk:
            groups.append(group)
    if max_chunks is not None:
        groups = groups[:max_chunks]
    return groups


def _chunk_frames_from_mp4(
    *,
    chunk_paths: list[Path],
    frames_per_chunk: int,
    frames_root: Path,
    use_nvdec: bool,
) -> list[list[Path]]:
    groups: list[list[Path]] = []
    for chunk_index, chunk_path in enumerate(chunk_paths):
        group = extract_spaced_jpegs_from_mp4(
            chunk_path,
            out_dir=frames_root / f"chunk_{chunk_index:06d}",
            stem="frame",
            n=frames_per_chunk,
            use_nvdec=use_nvdec,
        )
        groups.append(group)
    return groups


def create_file_batches(
    *,
    settings: Settings,
    storage: FrameBatchStorage,
    source_uri: str,
    source_kind: str,
    source_index: int,
    job_id: str,
    chunk_seconds: float,
    frames_per_chunk: int,
    max_chunks: int,
    chunk_format: str,
    work_dir: Path,
    processing_config: dict[str, object],
) -> list[FrameBatchManifest]:
    source_dir = work_dir / str(source_index)
    source_dir.mkdir(parents=True, exist_ok=True)

    chunk_paths: list[Path] = []
    groups: list[list[Path]]
    if _optional_chunk_artifact_needed(settings, chunk_format):
        chunk_paths = segment_to_mp4(
            uri=source_uri,
            kind=source_kind,
            out_dir=source_dir / "chunks_raw",
            chunk_seconds=chunk_seconds,
            use_nvdec=settings.enable_nvdec,
            max_chunks=max_chunks,
        )
        groups = _chunk_frames_from_mp4(
            chunk_paths=chunk_paths,
            frames_per_chunk=frames_per_chunk,
            frames_root=source_dir / "frames_raw",
            use_nvdec=settings.enable_nvdec,
        )
    else:
        frame_paths = segment_to_jpg(
            uri=source_uri,
            kind=source_kind,
            out_dir=source_dir / "frames_raw",
            chunk_seconds=chunk_seconds,
            use_nvdec=settings.enable_nvdec,
            frames_per_chunk=frames_per_chunk,
            max_chunks=max_chunks,
        )
        groups = _frame_groups(frame_paths, frames_per_chunk, max_chunks)

    manifests: list[FrameBatchManifest] = []
    source_scope, source_id = _source_scope_ids(job_id, None)
    ingest_started_at = time.time()
    has_audio = source_has_audio(source_uri, source_kind)
    for chunk_index, frame_group in enumerate(groups):
        chunk_path = chunk_paths[chunk_index] if chunk_index < len(chunk_paths) else None
        audio_path = None
        batch_has_audio = has_audio
        if settings.preserve_audio_artifacts and has_audio and chunk_path is not None:
            audio_path = _extract_audio_from_chunk(
                chunk_path,
                source_dir / f"audio_{chunk_index:06d}.m4a",
            )
            batch_has_audio = audio_path is not None
        manifest = storage.write_batch(
            source_scope=source_scope,
            source_id=source_id,
            source_type=source_kind,
            source_uri=source_uri,
            source_index=source_index,
            chunk_index=chunk_index,
            chunk_seconds=chunk_seconds,
            chunk_start_ts=ingest_started_at + (chunk_index * chunk_seconds),
            chunk_end_ts=ingest_started_at + ((chunk_index + 1) * chunk_seconds),
            frame_paths=frame_group,
            job_id=job_id,
            stream_id=None,
            has_audio=batch_has_audio,
            audio_source=audio_path,
            chunk_source=chunk_path,
            metadata={
                "chunk_format": chunk_format,
                "chunk_offset_seconds": chunk_index * chunk_seconds,
                "chunk_time_basis": "source_offset",
                "processing_config": processing_config,
                "audio_detected": batch_has_audio,
            },
        )
        manifests.append(manifest)
    return manifests


def create_stream_batch(
    *,
    settings: Settings,
    storage: FrameBatchStorage,
    stream_id: str,
    source_uri: str,
    chunk_seconds: float,
    frames_per_chunk: int,
    chunk_format: str,
    chunk_index: int,
    work_dir: Path,
    has_audio: bool,
    processing_config: dict[str, object],
) -> FrameBatchManifest | None:
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    chunk_path = None
    groups: list[list[Path]]
    if _optional_chunk_artifact_needed(settings, chunk_format):
        chunk_paths = segment_to_mp4(
            uri=source_uri,
            kind="rtsp",
            out_dir=work_dir / "chunks_raw",
            chunk_seconds=chunk_seconds,
            use_nvdec=settings.enable_nvdec,
            max_chunks=1,
        )
        chunk_path = chunk_paths[0] if chunk_paths else None
        if chunk_path is None:
            return None
        groups = _chunk_frames_from_mp4(
            chunk_paths=[chunk_path],
            frames_per_chunk=frames_per_chunk,
            frames_root=work_dir / "frames_raw",
            use_nvdec=settings.enable_nvdec,
        )
    else:
        frame_paths = segment_to_jpg(
            uri=source_uri,
            kind="rtsp",
            out_dir=work_dir / "frames_raw",
            chunk_seconds=chunk_seconds,
            use_nvdec=settings.enable_nvdec,
            frames_per_chunk=frames_per_chunk,
            max_chunks=1,
        )
        groups = _frame_groups(frame_paths, frames_per_chunk, 1)
    if not groups:
        return None

    audio_path = None
    if settings.preserve_audio_artifacts and has_audio and chunk_path is not None:
        audio_path = _extract_audio_from_chunk(chunk_path, work_dir / "audio.m4a")

    now = time.time()
    source_scope, source_id = _source_scope_ids(None, stream_id)
    return storage.write_batch(
        source_scope=source_scope,
        source_id=source_id,
        source_type="rtsp",
        source_uri=source_uri,
        source_index=0,
        chunk_index=chunk_index,
        chunk_seconds=chunk_seconds,
        chunk_start_ts=now,
        chunk_end_ts=now + chunk_seconds,
        frame_paths=groups[0],
        job_id=None,
        stream_id=stream_id,
        has_audio=has_audio,
        audio_source=audio_path,
        chunk_source=chunk_path,
        metadata={
            "chunk_format": chunk_format,
            "chunk_time_basis": "wall_clock",
            "processing_config": processing_config,
            "audio_detected": has_audio,
        },
    )
