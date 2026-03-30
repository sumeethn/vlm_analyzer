from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CreateStreamRequest(BaseModel):
    rtsp_url: str = Field(..., min_length=8, description="RTSP URL, e.g. rtsp://...")
    chunk_seconds: float = Field(10.0, ge=0.5, le=600.0)
    chunk_format: Literal["mp4", "jpg"] = "jpg"
    model: str = Field(..., min_length=1)
    prompt: str = Field(
        default="Describe what you see in this video segment.",
        min_length=1,
    )
    frames_per_chunk: int | None = Field(
        default=None,
        description="Override server default; samples this many frames per chunk window for the VLM.",
    )
    ollama_options: dict[str, Any] | None = None


class StreamAcceptedResponse(BaseModel):
    stream_id: str


class StreamListItem(BaseModel):
    stream_id: str
    rtsp_uri: str
    status: str
    chunk_seq: int
    last_chunk_at: float | None
    last_error: str | None


class StreamDetailResponse(BaseModel):
    stream_id: str
    rtsp_uri: str
    status: str
    model: str
    prompt: str
    chunk_seconds: float
    chunk_format: str
    frames_per_chunk: int
    chunk_seq: int
    last_chunk_at: float | None
    created_at: float
    updated_at: float
    last_error: str | None
