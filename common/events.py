"""
Shared event schemas for the Nova message bus.

All values in Redis Stream entries are strings; these TypedDicts document
the expected fields so skills and consumers can reference them by name.
"""
from __future__ import annotations

from typing import TypedDict


class InsightEvent(TypedDict):
    """
    Published to  nova:insights  by vlm_analyzer after each VLM completion.

    Fields
    ------
    insight_id:      UUID assigned by InsightStore.
    ts:              Unix timestamp (float) as string.
    stream_id:       vlm_analyzer stream_id, or "" for batch jobs.
    job_id:          vlm_analyzer job_id, or "" for live streams.
    source_index:    Source index within the job/stream (usually "0").
    chunk_index:     Sequential chunk number.
    model:           Ollama model name used for this completion.
    content:         The assistant text from the VLM response.
    completion_json: Full OpenAI-format completion object, JSON-serialized.
    """

    insight_id: str
    ts: str
    stream_id: str
    job_id: str
    source_index: str
    chunk_index: str
    model: str
    content: str
    completion_json: str


class AlertEvent(TypedDict):
    """
    Published to  nova:alerts  by a skill when it detects an event.

    Fields
    ------
    skill:       Skill identifier (e.g. "package_delivery_alert").
    stream_id:   The vlm_analyzer stream_id that triggered the alert.
    chunk_index: Chunk that triggered the alert.
    score:       Detection confidence score (integer, as string).
    matched:     Comma-separated list of matched signal keywords.
    excerpt:     Short excerpt of the VLM response text (≤300 chars).
    """

    skill: str
    stream_id: str
    chunk_index: str
    score: str
    matched: str
    excerpt: str
