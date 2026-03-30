from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class SourceKind(str, Enum):
    rtsp = "rtsp"
    file = "file"


class MediaSource(BaseModel):
    uri: str = Field(..., description="RTSP URL or absolute file path visible in the worker container")
    kind: SourceKind


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class StartProcessingRequest(BaseModel):
    sources: list[MediaSource] = Field(..., min_length=1)
    chunk_seconds: float = Field(10.0, ge=0.5, le=600.0)
    chunk_format: Literal["mp4", "jpg"] = "jpg"
    model: str = Field(..., min_length=1)
    prompt: str = Field(
        default="Describe what you see in this video segment.",
        min_length=1,
    )
    max_chunks_per_source: int = Field(200, ge=1, le=10_000)
    frames_per_chunk: int | None = Field(
        default=None,
        description="Override server default; samples this many frames per chunk window for the VLM.",
    )
    ollama_options: dict[str, Any] | None = None


class JobAcceptedResponse(BaseModel):
    job_id: str


class JobChunkResult(BaseModel):
    source_index: int
    chunk_index: int
    artifact_path: str
    completion: dict[str, Any]


class JobDetailResponse(BaseModel):
    job_id: str
    status: JobStatus
    model: str
    prompt: str
    chunk_seconds: float
    chunk_format: str
    frames_per_chunk: int
    sources: list[MediaSource]
    chunks_total: int
    chunks_done: int
    results: list[dict[str, Any]]
    error: str | None = None
