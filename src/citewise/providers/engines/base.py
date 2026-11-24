"""Answer-engine adapters: ask one question with web search on, get a normalized answer.

Every engine (OpenAI, Anthropic, Gemini, Perplexity, and the mock) returns the same
`EngineAnswer`, so the runner and Module 3 never see provider-specific shapes. `raw` keeps
the full provider payload for debugging and re-parsing.

Adapters do *not* retry: live SDK clients are built with `max_retries=0` and the runner
owns retries, backoff and rate limiting, so there is one retry policy, and every attempt
is logged to `llm_calls`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from citewise.providers.base import Citation, Usage

# HTTP statuses worth retrying: timeout, conflict/lock, rate limit, server errors and
# Anthropic's 529 "overloaded".
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


@dataclass
class EngineAnswer:
    text: str
    citations: list[Citation]
    model: str  # the model ID we asked for (pricing key); resolved snapshot stays in `raw`
    usage: Usage = field(default_factory=Usage)
    searches: int = 0  # web searches the engine ran (some providers bill per search)
    latency_ms: int = 0
    raw: dict[str, Any] = field(default_factory=dict)
    # Set when the provider reports the exact charge (Perplexity); wins over our estimate.
    reported_cost_usd: float | None = None


def get_path(data: Any, *keys: str | int, default: Any = None) -> Any:
    """Walk nested dicts/lists; accepts snake_case or camelCase alternatives as "a|b"."""
    node = data
    for key in keys:
        if isinstance(key, int):
            if not isinstance(node, list) or len(node) <= key:
                return default
            node = node[key]
            continue
        if not isinstance(node, dict):
            return default
        found = None
        for alt in key.split("|"):
            if alt in node:
                found = node[alt]
                break
        if found is None:
            return default
        node = found
    return node


class EngineError(Exception):
    """An adapter-level failure with an explicit retry decision.

    `answer` is set when the provider returned (and billed) a response that we reject, e.g.
    a truncated one, so the runner can still record what it cost.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        status: int | None = None,
        answer: EngineAnswer | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status = status
        self.answer = answer


def is_retryable(exc: BaseException) -> bool:
    """Retry rate limits, overloads, 5xx, timeouts and dropped connections; nothing else.

    Works across SDKs without importing them. openai/anthropic/perplexity put the HTTP
    status in `status_code`; google-genai's `APIError` uses `code` (int) and a *string*
    `status` like "RESOURCE_EXHAUSTED". Network errors are named *Timeout* / *Connection*.
    """
    if isinstance(exc, EngineError):
        return exc.retryable
    if isinstance(exc, TimeoutError | asyncio.TimeoutError | ConnectionError):
        return True
    for attr in ("status_code", "code", "status"):
        status = getattr(exc, attr, None)
        if isinstance(status, int) and not isinstance(status, bool):
            return status in RETRYABLE_STATUS
    name = type(exc).__name__
    return "Timeout" in name or "Connection" in name


@runtime_checkable
class EngineAdapter(Protocol):
    name: str  # "openai" | "anthropic" | "gemini" | "perplexity"
    model: str

    async def answer(self, prompt: str, *, system: str | None = None) -> EngineAnswer: ...


async def close_adapter(adapter: object) -> None:
    """Close an adapter's HTTP client if it has one (live adapters do; mocks don't)."""
    aclose = getattr(adapter, "aclose", None)
    if aclose is not None:
        await aclose()
