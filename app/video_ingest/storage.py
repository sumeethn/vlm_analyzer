from __future__ import annotations

import json
import shutil
import time
from urllib.parse import urlsplit, urlunsplit
import uuid
from pathlib import Path
from typing import Any

from app.config import Settings
from common.contracts.frame_batch import (
    FrameBatchManifest,
    FrameBatchReadyEvent,
    dump_debug_json,
)


class FrameBatchStorage:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._root = Path(settings.frame_batch_root)

    @property
    def root(self) -> Path:
        return self._root

    def build_batch_dir(self, source_scope: str, source_id: str, batch_id: str) -> Path:
        return self._root / source_scope / source_id / batch_id

    @staticmethod
    def sanitize_source_uri(source_uri: str) -> str:
        parts = urlsplit(source_uri)
        if not parts.username and not parts.password:
            return source_uri
        hostname = parts.hostname or ""
        port = f":{parts.port}" if parts.port else ""
        netloc = f"{hostname}{port}"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))

    def write_batch(
        self,
        *,
        source_scope: str,
        source_id: str,
        source_type: str,
        source_uri: str,
        source_index: int,
        chunk_index: int,
        chunk_seconds: float,
        chunk_start_ts: float,
        chunk_end_ts: float,
        frame_paths: list[Path],
        job_id: str | None,
        stream_id: str | None,
        has_audio: bool,
        audio_source: Path | None,
        chunk_source: Path | None,
        metadata: dict[str, Any],
    ) -> FrameBatchManifest:
        batch_id = uuid.uuid4().hex
        batch_dir = self.build_batch_dir(source_scope, source_id, batch_id)
        frames_dir = batch_dir / "frames"
        debug_dir = batch_dir / "debug"
        frames_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)

        stored_frames: list[str] = []
        for idx, src in enumerate(frame_paths):
            dest = frames_dir / f"frame_{idx:06d}{src.suffix.lower() or '.jpg'}"
            shutil.move(str(src), dest)
            stored_frames.append(str(dest))

        audio_path: str | None = None
        if audio_source and audio_source.exists():
            audio_dir = batch_dir / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            audio_dest = audio_dir / audio_source.name
            shutil.move(str(audio_source), audio_dest)
            audio_path = str(audio_dest)

        chunk_path: str | None = None
        if chunk_source and chunk_source.exists():
            chunk_dest = batch_dir / chunk_source.name
            shutil.move(str(chunk_source), chunk_dest)
            chunk_path = str(chunk_dest)

        created_at = time.time()
        manifest_path = batch_dir / "manifest.json"
        manifest = FrameBatchManifest(
            batch_id=batch_id,
            source_scope=source_scope,
            source_id=source_id,
            source_type=source_type,
            source_uri=self.sanitize_source_uri(source_uri),
            source_index=source_index,
            chunk_index=chunk_index,
            chunk_seconds=chunk_seconds,
            chunk_start_ts=chunk_start_ts,
            chunk_end_ts=chunk_end_ts,
            created_at=created_at,
            frame_count=len(stored_frames),
            sampling_fps=len(stored_frames) / max(chunk_seconds, 0.5),
            frame_paths=stored_frames,
            manifest_path=str(manifest_path),
            frames_dir=str(frames_dir),
            job_id=job_id,
            stream_id=stream_id,
            has_audio=has_audio,
            audio_path=audio_path,
            chunk_path=chunk_path,
            cleanup_after_ts=created_at + self._settings.frame_batch_retention_seconds,
            metadata=metadata,
        )
        manifest.write_json(manifest_path)
        dump_debug_json(
            debug_dir / "source.json",
            {
                "job_id": job_id,
                "stream_id": stream_id,
                "source_scope": source_scope,
                "source_id": source_id,
                "source_index": source_index,
                "chunk_index": chunk_index,
                "source_uri": source_uri,
                "metadata": metadata,
            },
        )
        return manifest

    def write_completion(
        self,
        manifest: FrameBatchManifest,
        completion: dict[str, Any],
    ) -> str:
        completion_path = Path(manifest.manifest_path or "").with_name("completion.json")
        completion_path.write_text(
            json.dumps(completion, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return str(completion_path)

    def mark_manifest(
        self,
        manifest_path: str,
        *,
        status: str,
        cleanup_after_ts: float,
        caption_completed_at: float | None = None,
        attempt: int | None = None,
    ) -> FrameBatchManifest:
        manifest = FrameBatchManifest.read_json(Path(manifest_path))
        manifest.status = status
        manifest.cleanup_after_ts = cleanup_after_ts
        if caption_completed_at is not None:
            manifest.caption_completed_at = caption_completed_at
        if attempt is not None:
            manifest.attempt = attempt
        manifest.write_json(Path(manifest_path))
        return manifest

    def cleanup_expired_batches(self) -> int:
        removed = 0
        if not self._root.exists():
            return removed
        now = time.time()
        for manifest_path in self._root.glob("*/*/*/manifest.json"):
            try:
                manifest = FrameBatchManifest.read_json(manifest_path)
            except Exception:
                continue
            if manifest.cleanup_after_ts is None or manifest.cleanup_after_ts > now:
                continue
            if manifest.status not in {"captioned", "failed"}:
                continue
            batch_dir = manifest_path.parent
            shutil.rmtree(batch_dir, ignore_errors=True)
            removed += 1
        return removed

    @staticmethod
    def manifest_to_event(manifest: FrameBatchManifest) -> FrameBatchReadyEvent:
        return FrameBatchReadyEvent.from_manifest(manifest)
