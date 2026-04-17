from __future__ import annotations

import json
import time
from typing import Any

import redis

from app.config import Settings
from common.contracts.caption import CaptionReadyEvent


def _redis(settings: Settings) -> redis.Redis:
    return redis.from_url(settings.redis_url, decode_responses=True)


def publish_caption_ready(settings: Settings, event: CaptionReadyEvent) -> bool:
    marker_key = f"{settings.caption_key_prefix}published:caption:{event.batch_id}"
    client = _redis(settings)
    with client.pipeline() as pipe:
        while True:
            try:
                pipe.watch(marker_key)
                if pipe.exists(marker_key):
                    pipe.unwatch()
                    return False
                pipe.multi()
                pipe.xadd(
                    settings.caption_ready_stream,
                    event.to_stream_fields(),
                    maxlen=settings.frame_batch_stream_maxlen,
                    approximate=True,
                )
                pipe.setex(marker_key, settings.caption_ttl_seconds, "1")
                pipe.execute()
                return True
            except redis.WatchError:
                continue
            finally:
                pipe.reset()


def publish_legacy_insight(
    settings: Settings,
    *,
    batch_id: str,
    job_id: str | None,
    stream_id: str | None,
    source_index: int,
    chunk_index: int,
    completion: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    marker_key = f"{settings.caption_key_prefix}published:legacy-insight:{batch_id}"
    record = {
        "insight_id": batch_id,
        "batch_id": batch_id,
        "ts": time.time(),
        "stream_id": stream_id,
        "job_id": job_id,
        "source_index": source_index,
        "chunk_index": chunk_index,
        "completion": completion,
    }
    content = ""
    try:
        content = str(completion["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError):
        pass

    payload = json.dumps(record)
    client = _redis(settings)
    with client.pipeline() as pipe:
        while True:
            try:
                pipe.watch(marker_key)
                if pipe.exists(marker_key):
                    pipe.unwatch()
                    return record, False
                pipe.multi()
                pipe.lpush(settings.insights_global_list_key, payload)
                pipe.ltrim(settings.insights_global_list_key, 0, settings.insights_max_per_list - 1)
                if stream_id:
                    stream_key = f"{settings.insights_stream_list_prefix}{stream_id}"
                    pipe.lpush(stream_key, payload)
                    pipe.ltrim(stream_key, 0, settings.insights_max_per_list - 1)
                if job_id:
                    job_key = f"{settings.insights_job_list_prefix}{job_id}"
                    pipe.lpush(job_key, payload)
                    pipe.ltrim(job_key, 0, settings.insights_max_per_list - 1)
                if settings.openclaw_bus_enabled:
                    pipe.xadd(
                        settings.openclaw_insights_stream,
                        {
                            "insight_id": record["insight_id"],
                            "batch_id": batch_id,
                            "ts": str(record["ts"]),
                            "stream_id": stream_id or "",
                            "job_id": job_id or "",
                            "source_index": str(source_index),
                            "chunk_index": str(chunk_index),
                            "model": completion.get("model", ""),
                            "content": content,
                            "completion_json": json.dumps(completion),
                        },
                        maxlen=settings.openclaw_stream_maxlen,
                        approximate=True,
                    )
                pipe.setex(marker_key, settings.caption_ttl_seconds, "1")
                pipe.execute()
                return record, True
            except redis.WatchError:
                continue
            finally:
                pipe.reset()
