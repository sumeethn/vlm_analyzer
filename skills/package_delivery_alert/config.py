"""
Configuration for the package_delivery_alert skill.
All settings are loaded from environment variables (or a .env file).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class SkillConfig:
    # --- vlm_analyzer ---
    vlm_base_url: str = field(
        default_factory=lambda: os.environ.get("VLM_BASE_URL", "http://localhost:8000")
    )
    rtsp_url: str = field(
        default_factory=lambda: os.environ["RTSP_URL"]  # required
    )
    vlm_model: str = field(
        default_factory=lambda: os.environ.get("VLM_MODEL", "llava")
    )
    vlm_prompt: str = field(
        default_factory=lambda: os.environ.get(
            "VLM_PROMPT",
            (
                "You are a security camera monitoring a front yard and porch. "
                "Describe what you see. Focus on: any people present, vehicles, "
                "packages or boxes being delivered or already on the porch, and "
                "delivery personnel (UPS, FedEx, USPS, Amazon, DHL, etc.)."
            ),
        )
    )
    chunk_seconds: float = field(
        default_factory=lambda: float(os.environ.get("CHUNK_SECONDS", "30"))
    )
    frames_per_chunk: int = field(
        default_factory=lambda: int(os.environ.get("FRAMES_PER_CHUNK", "3"))
    )

    # --- polling ---
    poll_interval_seconds: float = field(
        default_factory=lambda: float(os.environ.get("POLL_INTERVAL_SECONDS", "15"))
    )

    # --- Discord ---
    discord_webhook_url: str = field(
        default_factory=lambda: os.environ["DISCORD_WEBHOOK_URL"]  # required
    )
    discord_mention: str = field(
        # Optional: a Discord user/role mention string, e.g. "<@123456789>" or "@here"
        default_factory=lambda: os.environ.get("DISCORD_MENTION", "")
    )

    # --- detection ---
    # Minimum keyword score to trigger a notification (1 = any match).
    detection_threshold: int = field(
        default_factory=lambda: int(os.environ.get("DETECTION_THRESHOLD", "2"))
    )
    # Seconds to suppress duplicate notifications after one fires.
    cooldown_seconds: float = field(
        default_factory=lambda: float(os.environ.get("COOLDOWN_SECONDS", "300"))
    )


def load_config() -> SkillConfig:
    """Load config, applying a .env file if present."""
    _load_dotenv()
    return SkillConfig()


def _load_dotenv() -> None:
    """Minimal .env loader — avoids a hard dependency on python-dotenv."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
