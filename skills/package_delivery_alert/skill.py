"""
package_delivery_alert — OpenClaw skill

Registers a front-yard camera RTSP stream with the vlm_analyzer microservice,
polls for new insights, and sends a Discord notification whenever a package
delivery is detected in the VLM response.

Usage
-----
Set the required environment variables (see config.py or .env.example), then:

    python skill.py

The skill runs until interrupted (Ctrl-C or SIGTERM).
"""
from __future__ import annotations

import json
import logging
import signal
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from config import SkillConfig, load_config
from detector import detect_package_delivery
from notifier import NotifierConfig, send_delivery_alert

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("package_delivery_alert")


# ---------------------------------------------------------------------------
# vlm_analyzer API helpers
# ---------------------------------------------------------------------------

def _api_request(
    method: str,
    url: str,
    payload: dict | None = None,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Perform a JSON HTTP request and return the parsed response body."""
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def register_stream(cfg: SkillConfig) -> str:
    """
    POST /v1/streams to register the RTSP stream.
    Returns the stream_id assigned by vlm_analyzer.
    """
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
    """DELETE /v1/streams/{stream_id} to gracefully stop the stream."""
    url = f"{base_url}/v1/streams/{stream_id}"
    try:
        req = urllib.request.Request(url, method="DELETE")
        with urllib.request.urlopen(req, timeout=10):
            pass
        logger.info("Stream %s stopped.", stream_id)
    except Exception as exc:
        logger.warning("Could not stop stream %s: %s", stream_id, exc)


def fetch_insights(
    base_url: str,
    stream_id: str,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """
    GET /v1/streams/{stream_id}/insights and return the insight list.
    Insights are returned newest-first (Redis LPUSH order).
    """
    url = (
        f"{base_url}/v1/streams/{stream_id}/insights"
        f"?limit={limit}&offset={offset}"
    )
    resp = _api_request("GET", url)
    return resp.get("insights", [])


# ---------------------------------------------------------------------------
# State: track which insight_ids we have already evaluated
# ---------------------------------------------------------------------------

class _SeenSet:
    """In-memory set of processed insight_ids, bounded to avoid unbounded growth."""

    _MAX = 10_000

    def __init__(self) -> None:
        self._ids: set[str] = set()
        self._ordered: list[str] = []

    def seen(self, insight_id: str) -> bool:
        return insight_id in self._ids

    def mark(self, insight_id: str) -> None:
        if insight_id in self._ids:
            return
        if len(self._ids) >= self._MAX:
            evict = self._ordered.pop(0)
            self._ids.discard(evict)
        self._ids.add(insight_id)
        self._ordered.append(insight_id)


# ---------------------------------------------------------------------------
# Core polling loop
# ---------------------------------------------------------------------------

def _extract_text(insight: dict[str, Any]) -> str | None:
    """Pull the assistant text out of an InsightRecord completion."""
    try:
        return insight["completion"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def run(cfg: SkillConfig) -> None:
    notifier_cfg = NotifierConfig(
        webhook_url=cfg.discord_webhook_url,
        mention=cfg.discord_mention,
    )
    seen = _SeenSet()
    last_alert_ts: float = 0.0
    stream_id: str | None = None

    # --- graceful shutdown ---
    shutdown_requested = False

    def _handle_signal(sig: int, _frame: Any) -> None:
        nonlocal shutdown_requested
        logger.info("Shutdown signal received (%s).", signal.Signals(sig).name)
        shutdown_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # --- register stream ---
    while not shutdown_requested:
        try:
            stream_id = register_stream(cfg)
            break
        except Exception as exc:
            logger.error("Failed to register stream: %s — retrying in 30s", exc)
            _interruptible_sleep(30, lambda: shutdown_requested)

    if shutdown_requested or stream_id is None:
        return

    logger.info(
        "Polling insights every %.0fs (detection threshold=%d, cooldown=%.0fs) …",
        cfg.poll_interval_seconds,
        cfg.detection_threshold,
        cfg.cooldown_seconds,
    )

    # --- polling loop ---
    try:
        while not shutdown_requested:
            try:
                insights = fetch_insights(cfg.vlm_base_url, stream_id)
            except Exception as exc:
                logger.warning("Failed to fetch insights: %s", exc)
                _interruptible_sleep(cfg.poll_interval_seconds, lambda: shutdown_requested)
                continue

            new_count = 0
            for insight in insights:
                insight_id = insight.get("insight_id", "")
                if seen.seen(insight_id):
                    continue
                seen.mark(insight_id)
                new_count += 1

                text = _extract_text(insight)
                if not text:
                    logger.debug("Insight %s has no text content, skipping.", insight_id)
                    continue

                chunk_index = insight.get("chunk_index", -1)
                result = detect_package_delivery(text, cfg.detection_threshold)

                logger.info(
                    "Insight %s (chunk=%s): score=%d detected=%s matched=%s",
                    insight_id,
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
                        continue

                    ok = send_delivery_alert(notifier_cfg, stream_id, chunk_index, result)
                    if ok:
                        logger.info("Discord alert sent for insight %s.", insight_id)
                        last_alert_ts = now
                    else:
                        logger.error("Discord alert failed for insight %s.", insight_id)

            if new_count:
                logger.debug("Processed %d new insight(s).", new_count)

            _interruptible_sleep(cfg.poll_interval_seconds, lambda: shutdown_requested)

    finally:
        if stream_id:
            stop_stream(cfg.vlm_base_url, stream_id)


def _interruptible_sleep(seconds: float, stop: "Callable[[], bool]") -> None:
    """Sleep for *seconds*, waking early if *stop()* becomes True."""
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
