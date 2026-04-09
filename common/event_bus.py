"""
Shared Redis Streams helpers for OpenClaw skills and consumers.

Usage pattern for a skill
-------------------------
    from common.event_bus import EventBus

    bus = EventBus(redis_url="redis://localhost:6379/0")
    bus.ensure_consumer_group("openclaw:insights", "my_skill")

    for msg_id, fields in bus.consume("openclaw:insights", "my_skill", "worker-1"):
        process(fields)
        bus.ack("openclaw:insights", "my_skill", msg_id)

Usage pattern for publishing an alert
--------------------------------------
    bus.publish("openclaw:alerts", {"skill": "my_skill", ...})
"""
from __future__ import annotations

import logging
from typing import Iterator

import redis as redis_lib

logger = logging.getLogger(__name__)

# How long (ms) to block waiting for new stream entries before returning an
# empty result.  Keeps the consumer loop responsive to shutdown signals.
_DEFAULT_BLOCK_MS = 5_000


class EventBus:
    def __init__(self, redis_url: str, stream_maxlen: int = 10_000) -> None:
        self._r: redis_lib.Redis = redis_lib.from_url(redis_url, decode_responses=True)
        self._maxlen = stream_maxlen

    # ------------------------------------------------------------------
    # Consumer group management
    # ------------------------------------------------------------------

    def ensure_consumer_group(self, stream: str, group: str) -> None:
        """
        Create the consumer group if it doesn't exist.

        Uses id="$" so a freshly started skill only receives events produced
        *after* startup — it does not replay historical insights.
        Pass id="0" if you want to replay from the beginning of the stream.
        """
        try:
            self._r.xgroup_create(stream, group, id="$", mkstream=True)
            logger.info("Created consumer group '%s' on stream '%s'.", group, stream)
        except redis_lib.exceptions.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    # ------------------------------------------------------------------
    # Consuming
    # ------------------------------------------------------------------

    def consume(
        self,
        stream: str,
        group: str,
        consumer: str,
        block_ms: int = _DEFAULT_BLOCK_MS,
        count: int = 10,
    ) -> Iterator[tuple[str, dict[str, str]]]:
        """
        Yield (msg_id, fields) for pending messages in *stream* for *group*.

        Blocks up to *block_ms* milliseconds then returns an empty iterator,
        allowing the caller to check a shutdown flag between calls.
        """
        try:
            results = self._r.xreadgroup(
                group,
                consumer,
                {stream: ">"},
                count=count,
                block=block_ms,
            )
        except redis_lib.exceptions.ResponseError as exc:
            # Consumer group may not exist yet on first call; caller should
            # call ensure_consumer_group() before the loop.
            logger.error("xreadgroup failed: %s", exc)
            return

        if not results:
            return

        for _, entries in results:
            for msg_id, fields in entries:
                yield msg_id, fields

    def ack(self, stream: str, group: str, msg_id: str) -> None:
        """Acknowledge a consumed message so it is not redelivered."""
        self._r.xack(stream, group, msg_id)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, stream: str, fields: dict[str, str]) -> str:
        """
        Append *fields* to *stream* and return the assigned entry ID.
        """
        entry_id: str = self._r.xadd(
            stream,
            fields,
            maxlen=self._maxlen,
            approximate=True,
        )
        return entry_id
