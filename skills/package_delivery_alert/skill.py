"""
package_delivery_alert — Nova skill

Registers a front-yard camera RTSP stream with the vlm_analyzer microservice,
then subscribes to the  nova:insights  Redis Stream.  When the VLM
response for this camera scores above the detection threshold the skill
publishes an alert event to  nova:alerts  for any downstream consumer
(e.g. discord_notifier) to handle.

Usage
-----
Set the required environment variables (see .env.example), then:

    python skill.py

The skill runs until interrupted (Ctrl-C or SIGTERM).
"""
from __future__ import annotations

import json
import logging
import signal
import sys
import time
import urllib.request
from typing import Any

from config import SkillConfig, load_config
from detector import detect_package_delivery

# common/ is on PYTHONPATH (set by Dockerfile / local runner)
from common.event_bus import EventBus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("package_delivery_alert")

_SKILL_ID = "package_delivery_alert"
_CONSUMER_NAME = "worker-1"


# ---------------------------------------------------------------------------
# vlm_analyzer stream registration
# ---------------------------------------------------------------------------

def _api_request(method: str, url: str, payload: dict | None = None) -> dict[str, Any]:
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def register_stream(cfg: SkillConfig) -> str:
    """POST /v1/streams to register the RTSP camera. Returns stream_id."""
    url = f"{cfg.vlm_base_url}/v1/streams"
    payload = {
        "rtsp_url": cfg.rtsp_url,
        "chunk_seconds": cfg.chunk_seconds,
        "chunk_format": "jpg",
        "model": cfg.vlm_model,
        "prompt": cfg.vlm_prompt,
        "frames_per_chunk": cfg.frames_per_chunk,
    }
    logger.info("Registering RTSP stream %s with vlm_analyzer …", cfg.rtsp_url)
    resp = _api_request("POST", url, payload)
    stream_id: str = resp["stream_id"]
    logger.info("Stream registered → stream_id=%s", stream_id)
    return stream_id


def stop_stream(base_url: str, stream_id: str) -> None:
    """DELETE /v1/streams/{stream_id} to stop processing."""
    url = f"{base_url}/v1/streams/{stream_id}"
    try:
        req = urllib.request.Request(url, method="DELETE")
        with urllib.request.urlopen(req, timeout=10):
            pass
        logger.info("Stream %s stopped.", stream_id)
    except Exception as exc:
        logger.warning("Could not stop stream %s: %s", stream_id, exc)


# ---------------------------------------------------------------------------
# Core event loop
# ---------------------------------------------------------------------------

def run(cfg: SkillConfig) -> None:
    bus = EventBus(redis_url=cfg.redis_url, stream_maxlen=cfg.stream_maxlen)
    last_alert_ts: float = 0.0
    stream_id: str | None = None

    shutdown_requested = False

    def _handle_signal(sig: int, _frame: Any) -> None:
        nonlocal shutdown_requested
        logger.info("Shutdown signal received (%s).", signal.Signals(sig).name)
        shutdown_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Register camera stream with vlm_analyzer
    while not shutdown_requested:
        try:
            stream_id = register_stream(cfg)
            break
        except Exception as exc:
            logger.error("Failed to register stream: %s — retrying in 30s", exc)
            _interruptible_sleep(30, lambda: shutdown_requested)

    if shutdown_requested or stream_id is None:
        return

    # Create consumer group on the insights stream.
    # id="$" means: only process insights produced after this skill starts.
    bus.ensure_consumer_group(cfg.insights_stream, _SKILL_ID)

    logger.info(
        "Subscribed to %s as consumer group '%s' (detection threshold=%d, cooldown=%.0fs).",
        cfg.insights_stream,
        _SKILL_ID,
        cfg.detection_threshold,
        cfg.cooldown_seconds,
    )

    try:
        while not shutdown_requested:
            for msg_id, fields in bus.consume(cfg.insights_stream, _SKILL_ID, _CONSUMER_NAME):
                if shutdown_requested:
                    bus.ack(cfg.insights_stream, _SKILL_ID, msg_id)
                    break

                # Filter: only process insights from our registered camera stream
                if fields.get("stream_id") != stream_id:
                    bus.ack(cfg.insights_stream, _SKILL_ID, msg_id)
                    continue

                content = fields.get("content", "")
                chunk_index = fields.get("chunk_index", "-1")

                result = detect_package_delivery(content, cfg.detection_threshold)

                logger.info(
                    "Insight (chunk=%s): score=%d detected=%s matched=%s",
                    chunk_index,
                    result.score,
                    result.detected,
                    result.matched_groups,
                )

                if result.detected:
                    now = time.monotonic()
                    if now - last_alert_ts < cfg.cooldown_seconds:
                        logger.info(
                            "Delivery detected but suppressed (cooldown, %.0fs remaining).",
                            cfg.cooldown_seconds - (now - last_alert_ts),
                        )
                    else:
                        bus.publish(cfg.alerts_stream, {
                            "skill": _SKILL_ID,
                            "stream_id": stream_id,
                            "chunk_index": chunk_index,
                            "score": str(result.score),
                            "matched": ",".join(result.matched_groups),
                            "excerpt": result.excerpt,
                        })
                        logger.info("Alert published to %s.", cfg.alerts_stream)
                        last_alert_ts = now

                bus.ack(cfg.insights_stream, _SKILL_ID, msg_id)

    finally:
        if stream_id:
            stop_stream(cfg.vlm_base_url, stream_id)


def _interruptible_sleep(seconds: float, stop: "Callable[[], bool]") -> None:
    deadline = time.monotonic() + seconds
    while not stop():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(remaining, 0.5))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        cfg = load_config()
    except KeyError as exc:
        logger.error("Missing required environment variable: %s", exc)
        sys.exit(1)

    run(cfg)
