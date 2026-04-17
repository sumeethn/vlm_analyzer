from __future__ import annotations

import time

import redis

from app.config import Settings
from common.contracts.caption import CaptionRecord


class CaptionStore:
    def __init__(self, settings: Settings) -> None:
        self._r = redis.from_url(settings.redis_url, decode_responses=True)
        self._prefix = settings.caption_key_prefix
        self._ttl = settings.caption_ttl_seconds

    def _key(self, batch_id: str) -> str:
        return f"{self._prefix}{batch_id}"

    def get(self, batch_id: str) -> CaptionRecord | None:
        raw = self._r.get(self._key(batch_id))
        if not raw:
            return None
        return CaptionRecord.from_json(raw)

    def save(self, record: CaptionRecord) -> None:
        record.updated_at = time.time()
        self._r.setex(self._key(record.batch_id), self._ttl, record.to_json())

    def exists(self, batch_id: str) -> bool:
        return bool(self._r.exists(self._key(batch_id)))
