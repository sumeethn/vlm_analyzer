from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["system", "user", "assistant"]
    content: str | list[dict[str, Any]]
    # OpenAI compatibility: optional image_url in content parts
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: float | None = None
    stream: bool = False
    max_tokens: int | None = None
