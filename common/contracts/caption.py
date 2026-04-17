from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel


def _as_str(value: str | None) -> str:
    return value or ""


class CaptionReadyEvent(BaseModel):
    event_type: Literal["caption_ready"] = "caption_ready"
    batch_id: str
    job_id: str | None = None
    stream_id: str | None = None
    source_index: int = 0
    chunk_index: int = 0
    model: str
    manifest_path: str
    caption_text: str
    completion_json: str
    created_at: float

    def to_stream_fields(self) -> dict[str, str]:
        return {
            "event_type": self.event_type,
            "batch_id": self.batch_id,
            "job_id": _as_str(self.job_id),
            "stream_id": _as_str(self.stream_id),
            "source_index": str(self.source_index),
            "chunk_index": str(self.chunk_index),
            "model": self.model,
            "manifest_path": self.manifest_path,
            "caption_text": self.caption_text,
            "completion_json": self.completion_json,
            "created_at": str(self.created_at),
        }

    @classmethod
    def from_stream_fields(cls, fields: dict[str, str]) -> "CaptionReadyEvent":
        return cls(
            batch_id=fields["batch_id"],
            job_id=fields.get("job_id") or None,
            stream_id=fields.get("stream_id") or None,
            source_index=int(fields.get("source_index", "0")),
            chunk_index=int(fields.get("chunk_index", "0")),
            model=fields["model"],
            manifest_path=fields["manifest_path"],
            caption_text=fields.get("caption_text", ""),
            completion_json=fields.get("completion_json", "{}"),
            created_at=float(fields["created_at"]),
        )


class CaptionRecord(BaseModel):
    batch_id: str
    status: Literal["processing", "completed", "failed"] = "processing"
    manifest_path: str
    model: str
    job_id: str | None = None
    stream_id: str | None = None
    source_index: int = 0
    chunk_index: int = 0
    attempt: int = 0
    created_at: float
    updated_at: float
    caption_text: str | None = None
    completion: dict[str, Any] | None = None
    error: str | None = None
    caption_event_published: bool = False
    legacy_insight_published: bool = False

    def to_json(self) -> str:
        return self.model_dump_json()

    @classmethod
    def from_json(cls, payload: str) -> "CaptionRecord":
        return cls.model_validate_json(payload)

    def completion_json(self) -> str:
        return json.dumps(self.completion or {})
