from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import Any

import redis

from app.config import Settings


class JobStore:
    def __init__(self, settings: Settings) -> None:
        self._r = redis.from_url(settings.redis_url, decode_responses=True)
        self.prefix = settings.job_key_prefix
        self.ttl = settings.job_ttl_seconds

    def _key(self, job_id: str) -> str:
        return f"{self.prefix}{job_id}"

    def create_job(self, initial: dict[str, Any]) -> str:
        job_id = str(uuid.uuid4())
        data = {"job_id": job_id, **initial}
        self._r.setex(self._key(job_id), self.ttl, json.dumps(data))
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        raw = self._r.get(self._key(job_id))
        if not raw:
            return None
        return json.loads(raw)

    def save(self, data: dict[str, Any]) -> None:
        job_id = data["job_id"]
        key = self._key(job_id)
        blob = json.dumps(data)
        try:
            if self._r.exists(key):
                ttl = self._r.ttl(key)
                if ttl is not None and ttl > 0:
                    self._r.setex(key, ttl, blob)
                else:
                    self._r.setex(key, self.ttl, blob)
            else:
                self._r.setex(key, self.ttl, blob)
        except redis.RedisError:
            self._r.setex(key, self.ttl, blob)

    def update(
        self,
        job_id: str,
        mutator: Callable[[dict[str, Any]], dict[str, Any] | None],
    ) -> dict[str, Any] | None:
        """
        Optimistic-lock read-modify-write on a job key.

        Reads the current TTL while in WATCH mode and preserves it via
        setex inside the MULTI block.  This avoids keepttl=True (Redis 6.0+
        only) and the dead try/except pattern that could never fire inside a
        MULTI block.
        """
        key = self._key(job_id)
        with self._r.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(key)
                    raw = pipe.get(key)
                    if not raw:
                        pipe.unwatch()
                        return None
                    current = json.loads(raw)
                    updated = mutator(current)
                    if updated is None:
                        pipe.unwatch()
                        return current
                    # Read TTL while still in watch (immediate) mode so we
                    # can replicate it inside MULTI without keepttl=True.
                    ttl = pipe.ttl(key)
                    effective_ttl = ttl if (ttl is not None and ttl > 0) else self.ttl
                    pipe.multi()
                    pipe.setex(key, effective_ttl, json.dumps(updated))
                    pipe.execute()
                    return updated
                except redis.WatchError:
                    continue
                finally:
                    pipe.reset()

    def update_once(
        self,
        job_id: str,
        *,
        marker: str,
        marker_ttl_seconds: int,
        mutator: Callable[[dict[str, Any]], dict[str, Any] | None],
    ) -> tuple[dict[str, Any] | None, bool]:
        """
        Idempotent read-modify-write: executes the mutation exactly once,
        gated by *marker*.  Returns ``(final_state, did_execute)``.
        """
        key = self._key(job_id)
        with self._r.pipeline() as pipe:
            while True:
                try:
                    pipe.watch(key, marker)
                    raw = pipe.get(key)
                    if not raw:
                        pipe.unwatch()
                        return None, False
                    if pipe.exists(marker):
                        pipe.unwatch()
                        return json.loads(raw), False
                    current = json.loads(raw)
                    updated = mutator(current)
                    if updated is None:
                        pipe.unwatch()
                        return current, False
                    ttl = pipe.ttl(key)
                    effective_ttl = ttl if (ttl is not None and ttl > 0) else self.ttl
                    payload = json.dumps(updated)
                    pipe.multi()
                    pipe.setex(key, effective_ttl, payload)
                    pipe.setex(marker, marker_ttl_seconds, "1")
                    pipe.execute()
                    return updated, True
                except redis.WatchError:
                    continue
                finally:
                    pipe.reset()

    def ping(self) -> bool:
        try:
            return bool(self._r.ping())
        except Exception:
            return False


def validate_file_under_mount(uri: str, video_mount: str) -> str:
    from pathlib import Path

    base = Path(video_mount).resolve()
    p = Path(uri).resolve()
    try:
        p.relative_to(base)
    except ValueError as e:
        raise ValueError("file path must resolve under VIDEO_MOUNT") from e
    if not p.is_file():
        raise ValueError("file does not exist or is not a file")
    return str(p)
