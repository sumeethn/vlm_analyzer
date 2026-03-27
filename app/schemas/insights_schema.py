from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class InsightRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    insight_id: str
    ts: float
    stream_id: str | None = None
    job_id: str | None = None
    source_index: int = 0
    chunk_index: int = 0
    completion: dict[str, Any]


class JobResultsResponse(BaseModel):
    job_id: str
    results: list[dict[str, Any]]


class InsightsListResponse(BaseModel):
    insights: list[InsightRecord]
    total_returned: int
