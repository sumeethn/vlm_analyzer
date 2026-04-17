from __future__ import annotations

from app.config import Settings
from app.services.event_bus import publish_insight
from app.state.insights import InsightStore
from common.contracts.caption import CaptionReadyEvent
from common.event_bus import EventBus


def publish_caption_ready(settings: Settings, event: CaptionReadyEvent) -> None:
    bus = EventBus(settings.redis_url, stream_maxlen=settings.frame_batch_stream_maxlen)
    bus.publish(settings.caption_ready_stream, event.to_stream_fields())


def publish_legacy_insight(
    settings: Settings,
    *,
    job_id: str | None,
    stream_id: str | None,
    source_index: int,
    chunk_index: int,
    completion: dict,
) -> dict:
    record = InsightStore(settings).append(
        stream_id=stream_id,
        job_id=job_id,
        source_index=source_index,
        chunk_index=chunk_index,
        completion=completion,
    )
    publish_insight(settings, record)
    return record
