"""
discord_notifier — Nova consumer

Subscribes to the  nova:alerts  Redis Stream and posts a Discord embed
for every alert event, regardless of which skill produced it.

Decoupling notifications from detection means:
- Skills stay focused on detection logic only.
- Swapping Discord for another channel (Slack, SMS, etc.) only requires
  a new consumer — no skill changes needed.

Usage
-----
Set environment variables (see .env.example), then:

    python consumer.py
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
import urllib.request
from typing import Any

# common/ is on PYTHONPATH (set by Dockerfile / local runner)
from common.event_bus import EventBus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("discord_notifier")

_CONSUMER_GROUP = "discord_notifier"
_CONSUMER_NAME = "worker-1"

# Human-readable skill labels for the embed title
_SKILL_LABELS: dict[str, str] = {
    "package_delivery_alert": "📦 Package Delivery Detected",
    "vehicle_exit_monitor": "🚗 Vehicle Exiting Premises",
    "license_plate_gate": "🔍 License Plate Recognised at Gate",
}
_DEFAULT_LABEL = "🔔 Nova Alert"

_SKILL_COLORS: dict[str, int] = {
    "package_delivery_alert": 0xF4A015,   # amber
    "vehicle_exit_monitor":   0x3498DB,   # blue
    "license_plate_gate":     0x2ECC71,   # green
}
_DEFAULT_COLOR = 0x95A5A6  # grey


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _require(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        raise KeyError(key)
    return val


def _load_dotenv() -> None:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# ---------------------------------------------------------------------------
# Discord webhook
# ---------------------------------------------------------------------------

def _post_discord(webhook_url: str, mention: str, fields: dict[str, str]) -> bool:
    skill = fields.get("skill", "unknown")
    title = _SKILL_LABELS.get(skill, _DEFAULT_LABEL)
    color = _SKILL_COLORS.get(skill, _DEFAULT_COLOR)

    description = (
        f"**VLM says:**\n> {fields.get('excerpt', '')}\n\n"
        f"**Matched signals:** {fields.get('matched', '')}\n"
        f"**Score:** {fields.get('score', '?')}"
    )

    payload: dict[str, Any] = {
        "embeds": [{
            "title": title,
            "description": description,
            "color": color,
            "fields": [
                {"name": "Skill",    "value": skill,                          "inline": True},
                {"name": "Stream",   "value": fields.get("stream_id", "?"),   "inline": True},
                {"name": "Chunk",    "value": fields.get("chunk_index", "?"), "inline": True},
            ],
            "footer": {"text": "Nova · discord_notifier"},
            "timestamp": _iso_utc(int(time.time())),
        }]
    }
    if mention:
        payload["content"] = mention

    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        webhook_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status not in (200, 204):
                logger.error("Discord returned HTTP %s", resp.status)
                return False
        return True
    except Exception as exc:
        logger.error("Discord request failed: %s", exc)
        return False


def _iso_utc(epoch: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(
    redis_url: str,
    alerts_stream: str,
    discord_webhook_url: str,
    discord_mention: str,
    stream_maxlen: int,
) -> None:
    bus = EventBus(redis_url=redis_url, stream_maxlen=stream_maxlen)
    bus.ensure_consumer_group(alerts_stream, _CONSUMER_GROUP)

    shutdown_requested = False

    def _handle_signal(sig: int, _frame: Any) -> None:
        nonlocal shutdown_requested
        logger.info("Shutdown signal received (%s).", signal.Signals(sig).name)
        shutdown_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info("Subscribed to %s — forwarding alerts to Discord.", alerts_stream)

    while not shutdown_requested:
        for msg_id, fields in bus.consume(alerts_stream, _CONSUMER_GROUP, _CONSUMER_NAME):
            if shutdown_requested:
                bus.ack(alerts_stream, _CONSUMER_GROUP, msg_id)
                break

            skill = fields.get("skill", "unknown")
            logger.info(
                "Alert from skill '%s' (stream=%s chunk=%s score=%s).",
                skill,
                fields.get("stream_id"),
                fields.get("chunk_index"),
                fields.get("score"),
            )

            ok = _post_discord(discord_webhook_url, discord_mention, fields)
            if ok:
                logger.info("Discord notification sent.")
            else:
                logger.error("Discord notification failed — message will NOT be redelivered.")

            bus.ack(alerts_stream, _CONSUMER_GROUP, msg_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _load_dotenv()
    try:
        redis_url         = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        alerts_stream     = os.environ.get("NOVA_ALERTS_STREAM", "nova:alerts")
        discord_webhook   = _require("DISCORD_WEBHOOK_URL")
        discord_mention   = os.environ.get("DISCORD_MENTION", "")
        stream_maxlen     = int(os.environ.get("NOVA_STREAM_MAXLEN", "10000"))
    except KeyError as exc:
        logger.error("Missing required environment variable: %s", exc)
        sys.exit(1)

    run(redis_url, alerts_stream, discord_webhook, discord_mention, stream_maxlen)
