from __future__ import annotations

import json
import uuid
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
        self._r.setex(self._key(job_id), self.ttl, json.dumps(data))

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
