from __future__ import annotations

import json
import time
import uuid
from typing import Any

import redis

from app.config import Settings


class InsightStore:
    def __init__(self, settings: Settings) -> None:
        self._r = redis.from_url(settings.redis_url, decode_responses=True)
        self._gk = settings.insights_global_list_key
        self._sp = settings.insights_stream_list_prefix
        self._jp = settings.insights_job_list_prefix
        self._max = settings.insights_max_per_list

    def append(
        self,
        *,
        stream_id: str | None,
        job_id: str | None,
        source_index: int,
        chunk_index: int,
        completion: dict[str, Any],
    ) -> dict[str, Any]:
        record = {
            "insight_id": str(uuid.uuid4()),
            "ts": time.time(),
            "stream_id": stream_id,
            "job_id": job_id,
            "source_index": source_index,
            "chunk_index": chunk_index,
            "completion": completion,
        }
        payload = json.dumps(record)
        pipe = self._r.pipeline()
        pipe.lpush(self._gk, payload)
        pipe.ltrim(self._gk, 0, self._max - 1)
        if stream_id:
            sk = f"{self._sp}{stream_id}"
            pipe.lpush(sk, payload)
            pipe.ltrim(sk, 0, self._max - 1)
        if job_id:
            jk = f"{self._jp}{job_id}"
            pipe.lpush(jk, payload)
            pipe.ltrim(jk, 0, self._max - 1)
        pipe.execute()
        return record

    def list_insights(
        self,
        *,
        stream_id: str | None,
        limit: int,
        offset: int,
    ) -> tuple[list[dict[str, Any]], int]:
        key = f"{self._sp}{stream_id}" if stream_id else self._gk
        stop = offset + limit - 1
        if stop < offset:
            return [], 0
        raw_items = self._r.lrange(key, offset, stop)
        parsed: list[dict[str, Any]] = []
        for raw in raw_items:
            try:
                parsed.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return parsed, len(parsed)
