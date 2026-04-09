"""
Discord notification via incoming webhook.

Sends a rich embed when a package delivery is detected.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from dataclasses import dataclass

from detector import DetectionResult

logger = logging.getLogger(__name__)


@dataclass
class NotifierConfig:
    webhook_url: str
    mention: str = ""  # e.g. "<@123456789>" or "@here"


def send_delivery_alert(
    config: NotifierConfig,
    stream_id: str,
    chunk_index: int,
    result: DetectionResult,
) -> bool:
    """
    POST a Discord embed to the configured webhook.

    Returns True on success, False on failure.
    """
    ts = int(time.time())
    description = (
        f"**VLM says:**\n> {result.excerpt}\n\n"
        f"**Matched signals:** {', '.join(result.matched_groups)}\n"
        f"**Score:** {result.score}"
    )

    content = config.mention if config.mention else None

    payload: dict = {
        "embeds": [
            {
                "title": "📦 Package Delivery Detected",
                "description": description,
                "color": 0xF4A015,  # amber
                "fields": [
                    {"name": "Stream", "value": stream_id, "inline": True},
                    {"name": "Chunk", "value": str(chunk_index), "inline": True},
                ],
                "footer": {"text": "package_delivery_alert skill"},
                "timestamp": _iso_utc(ts),
            }
        ]
    }
    if content:
        payload["content"] = content

    return _post_webhook(config.webhook_url, payload)


def _post_webhook(url: str, payload: dict) -> bool:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status not in (200, 204):
                logger.error("Discord webhook returned HTTP %s", resp.status)
                return False
        return True
    except Exception as exc:
        logger.error("Discord webhook request failed: %s", exc)
        return False


def _iso_utc(epoch: int) -> str:
    """Format epoch seconds as an ISO-8601 UTC string for Discord embeds."""
    t = time.gmtime(epoch)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", t)
