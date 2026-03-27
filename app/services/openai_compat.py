from __future__ import annotations

import time
import uuid
from typing import Any


def ollama_message_content(resp: dict[str, Any]) -> str:
    msg = resp.get("message") or {}
    content = msg.get("content")
    if isinstance(content, str):
        return content
    return str(content or "")


def ollama_to_openai_chat_completion(
    *,
    ollama_body: dict[str, Any],
    model: str,
    completion_id_prefix: str = "chatcmpl",
) -> dict[str, Any]:
    """Map Ollama /api/chat JSON to OpenAI chat.completion-ish object."""
    content = ollama_message_content(ollama_body)
    prompt_tokens = ollama_body.get("prompt_eval_count")
    completion_tokens = ollama_body.get("eval_count")
    total = None
    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        total = prompt_tokens + completion_tokens
    elif isinstance(prompt_tokens, int):
        total = prompt_tokens
    elif isinstance(completion_tokens, int):
        total = completion_tokens

    return {
        "id": f"{completion_id_prefix}-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total,
        },
    }


def strip_openai_prefix(s: str) -> str:
    if s.startswith("data:"):
        return s.split(",", 1)[-1]
    return s
