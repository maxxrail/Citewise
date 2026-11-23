"""Placeholders for engines implemented in Module 2. They fail loudly instead of guessing."""

from __future__ import annotations

from typing import Any

from citewise.providers.base import ChatResult, JsonSchema, Message


class NotYetImplementedProvider:
    def __init__(self, name: str) -> None:
        self.name = name

    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        purpose: str = "general",
        json_schema: JsonSchema | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        raise NotImplementedError(
            f"Live '{self.name}' provider arrives in Module 2. Use CITEWISE_MODE=mock for now."
        )
