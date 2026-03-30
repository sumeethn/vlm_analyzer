from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500:
        return True
    return False


def file_to_base64(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii")


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(8),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def ollama_chat_vision(
    *,
    base_url: str,
    model: str,
    prompt: str,
    images_b64: list[str],
    timeout_seconds: float,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not images_b64:
        raise ValueError("at least one image is required")
    base = base_url.rstrip("/")
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "images": images_b64,
            }
        ],
        "stream": False,
    }
    if options:
        for k, v in options.items():
            if k in {"messages", "model", "stream"}:
                continue
            payload[k] = v
    with httpx.Client(timeout=timeout_seconds) as client:
        r = client.post(f"{base}/api/chat", json=payload)
        r.raise_for_status()
        return r.json()


def extract_text_and_image_b64_from_openai_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, str | None]:
    """Use the last user message: text + optional one image (url or raw base64)."""
    last_user: dict[str, Any] | None = None
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m
            break
    if last_user is None:
        raise ValueError("No user message found")

    content = last_user.get("content")
    text_parts: list[str] = []
    image_b64: str | None = None

    if isinstance(content, str):
        text_parts.append(content)
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "text":
                text_parts.append(str(part.get("text", "")))
            elif ptype == "image_url":
                iu = part.get("image_url")
                if isinstance(iu, dict):
                    url = str(iu.get("url", ""))
                else:
                    url = str(iu or "")
                if url.startswith("data:"):
                    url = url.split(",", 1)[-1]
                image_b64 = url
    else:
        text_parts.append(str(content))

    return "\n".join(text_parts).strip(), image_b64
