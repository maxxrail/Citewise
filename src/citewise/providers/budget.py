"""Per-command spend cap for chat calls.

`BudgetedChatProvider` wraps a (tracked) chat provider, adds up the estimated cost of
each call, and refuses new calls once the cap is reached. Refusals raise
`BudgetExceededError`, which `services.structured.complete_structured` turns into a
skipped call, so a run stops spending but keeps what it already paid for.

Embeddings are not counted: they cost ~1/1000 of a chat call at panel sizes.
"""

from __future__ import annotations

from typing import Any

from citewise.providers.base import ChatProvider, ChatResult, JsonSchema, Message
from citewise.providers.pricing import estimate_cost


class BudgetExceededError(RuntimeError):
    pass


class BudgetedChatProvider:
    def __init__(self, inner: ChatProvider, max_cost_usd: float) -> None:
        self.inner = inner
        self.name = inner.name
        self.max_cost_usd = max_cost_usd
        self.spent_usd = 0.0
        self.refused = 0

    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        purpose: str = "general",
        json_schema: JsonSchema | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        if self.spent_usd >= self.max_cost_usd:
            self.refused += 1
            raise BudgetExceededError(
                f"budget of ${self.max_cost_usd:.2f} reached (spent ${self.spent_usd:.4f})"
            )
        result = self.inner.complete(
            messages, model=model, purpose=purpose, json_schema=json_schema, tools=tools
        )
        self.spent_usd += estimate_cost(
            result.model, result.usage.tokens_in, result.usage.tokens_out
        )
        return result
