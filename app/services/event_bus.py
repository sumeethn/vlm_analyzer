"""
Publish insight events to the OpenClaw Redis Stream.

Only active when OPENCLAW_BUS_ENABLED=true.  Failures are logged and
swallowed so a bus outage never disrupts normal vlm_analyzer operation.
"""
from __future__ import annotations

import json
import logging

import redis as redis_lib

from app.config import Settings

logger = logging.getLogger(__name__)


def publish_insight(settings: Settings, record: dict) -> None:
    """
    Publish an InsightRecord to the openclaw:insights Redis Stream.

    Parameters
    ----------
    settings:   Application settings (checked for openclaw_bus_enabled).
    record:     The dict returned by InsightStore.append().
    """
    if not settings.openclaw_bus_enabled:
        return

    try:
        content = ""
        try:
            content = record["completion"]["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            pass

        r = redis_lib.from_url(settings.redis_url, decode_responses=True)
        r.xadd(
            settings.openclaw_insights_stream,
            {
                "insight_id": record.get("insight_id", ""),
                "ts": str(record.get("ts", "")),
                "stream_id": record.get("stream_id") or "",
                "job_id": record.get("job_id") or "",
                "source_index": str(record.get("source_index", 0)),
                "chunk_index": str(record.get("chunk_index", 0)),
                "model": record.get("completion", {}).get("model", ""),
                "content": content,
                "completion_json": json.dumps(record.get("completion", {})),
            },
            maxlen=settings.openclaw_stream_maxlen,
            approximate=True,
        )
        logger.debug(
            "Published insight %s to %s",
            record.get("insight_id"),
            settings.openclaw_insights_stream,
        )
    except Exception:
        logger.warning("Failed to publish insight to OpenClaw bus", exc_info=True)
