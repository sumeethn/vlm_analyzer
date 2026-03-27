from __future__ import annotations

import json
import time
import uuid
from typing import Any

import redis

from app.config import Settings


class StreamStore:
    def __init__(self, settings: Settings) -> None:
        self._r = redis.from_url(settings.redis_url, decode_responses=True)
        self._p = settings.stream_key_prefix
        self._active = settings.streams_active_set_key

    def _key(self, stream_id: str) -> str:
        return f"{self._p}{stream_id}"

    def create_stream(self, initial: dict[str, Any]) -> str:
        stream_id = str(uuid.uuid4())
        now = time.time()
        data = {
            "stream_id": stream_id,
            "created_at": now,
            "updated_at": now,
            "chunk_seq": 0,
            "last_chunk_at": None,
            "last_error": None,
            **initial,
        }
        self._r.set(self._key(stream_id), json.dumps(data))
        self._r.sadd(self._active, stream_id)
        return stream_id

    def get(self, stream_id: str) -> dict[str, Any] | None:
        raw = self._r.get(self._key(stream_id))
        if not raw:
            return None
        return json.loads(raw)

    def save(self, data: dict[str, Any]) -> None:
        data["updated_at"] = time.time()
        sid = data["stream_id"]
        self._r.set(self._key(sid), json.dumps(data))

    def delete(self, stream_id: str) -> None:
        self._r.delete(self._key(stream_id))
        self._r.srem(self._active, stream_id)

    def remove_from_active(self, stream_id: str) -> None:
        self._r.srem(self._active, stream_id)

    def add_to_active(self, stream_id: str) -> None:
        self._r.sadd(self._active, stream_id)

    def list_active_ids(self) -> list[str]:
        ids = list(self._r.smembers(self._active))
        return sorted(ids)

    def reconcile_active_set(self) -> None:
        """Drop stale members whose stream keys are missing."""
        for sid in self.list_active_ids():
            if not self._r.exists(self._key(sid)):
                self._r.srem(self._active, sid)
