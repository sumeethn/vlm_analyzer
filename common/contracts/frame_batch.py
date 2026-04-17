from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


def _as_str(value: str | None) -> str:
    return value or ""


def _parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


class FrameBatchManifest(BaseModel):
    batch_id: str
    source_scope: Literal["jobs", "streams"]
    source_id: str
    source_type: Literal["file", "rtsp"]
    source_uri: str
    source_index: int = 0
    chunk_index: int = 0
    chunk_seconds: float
    chunk_start_ts: float
    chunk_end_ts: float
    created_at: float
    frame_count: int
    sampling_fps: float
    frame_paths: list[str]
    manifest_path: str | None = None
    frames_dir: str | None = None
    job_id: str | None = None
    stream_id: str | None = None
    has_audio: bool = False
    audio_path: str | None = None
    chunk_path: str | None = None
    status: Literal["ready", "captioned", "failed"] = "ready"
    caption_completed_at: float | None = None
    cleanup_after_ts: float | None = None
    attempt: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def read_json(cls, path: Path) -> "FrameBatchManifest":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


class FrameBatchReadyEvent(BaseModel):
    event_type: Literal["frame_batch_ready"] = "frame_batch_ready"
    batch_id: str
    source_type: Literal["file", "rtsp"]
    source_id: str
    job_id: str | None = None
    stream_id: str | None = None
    source_index: int = 0
    chunk_index: int = 0
    chunk_start_ts: float
    chunk_end_ts: float
    manifest_path: str
    frames_dir: str
    frame_count: int
    has_audio: bool = False
    audio_path: str | None = None
    attempt: int = 0
    created_at: float

    def to_stream_fields(self) -> dict[str, str]:
        return {
            "event_type": self.event_type,
            "batch_id": self.batch_id,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "job_id": _as_str(self.job_id),
            "stream_id": _as_str(self.stream_id),
            "source_index": str(self.source_index),
            "chunk_index": str(self.chunk_index),
            "chunk_start_ts": str(self.chunk_start_ts),
            "chunk_end_ts": str(self.chunk_end_ts),
            "manifest_path": self.manifest_path,
            "frames_dir": self.frames_dir,
            "frame_count": str(self.frame_count),
            "has_audio": "1" if self.has_audio else "0",
            "audio_path": _as_str(self.audio_path),
            "attempt": str(self.attempt),
            "created_at": str(self.created_at),
        }

    @classmethod
    def from_stream_fields(cls, fields: dict[str, str]) -> "FrameBatchReadyEvent":
        return cls(
            event_type="frame_batch_ready",
            batch_id=fields["batch_id"],
            source_type=fields["source_type"],
            source_id=fields["source_id"],
            job_id=fields.get("job_id") or None,
            stream_id=fields.get("stream_id") or None,
            source_index=int(fields.get("source_index", "0")),
            chunk_index=int(fields.get("chunk_index", "0")),
            chunk_start_ts=float(fields["chunk_start_ts"]),
            chunk_end_ts=float(fields["chunk_end_ts"]),
            manifest_path=fields["manifest_path"],
            frames_dir=fields["frames_dir"],
            frame_count=int(fields["frame_count"]),
            has_audio=_parse_bool(fields.get("has_audio")),
            audio_path=fields.get("audio_path") or None,
            attempt=int(fields.get("attempt", "0")),
            created_at=float(fields["created_at"]),
        )

    @classmethod
    def from_manifest(cls, manifest: FrameBatchManifest) -> "FrameBatchReadyEvent":
        if not manifest.manifest_path or not manifest.frames_dir:
            raise ValueError("manifest_path and frames_dir are required on the manifest")
        return cls(
            batch_id=manifest.batch_id,
            source_type=manifest.source_type,
            source_id=manifest.source_id,
            job_id=manifest.job_id,
            stream_id=manifest.stream_id,
            source_index=manifest.source_index,
            chunk_index=manifest.chunk_index,
            chunk_start_ts=manifest.chunk_start_ts,
            chunk_end_ts=manifest.chunk_end_ts,
            manifest_path=manifest.manifest_path,
            frames_dir=manifest.frames_dir,
            frame_count=manifest.frame_count,
            has_audio=manifest.has_audio,
            audio_path=manifest.audio_path,
            attempt=manifest.attempt,
            created_at=manifest.created_at,
        )


def dump_debug_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
