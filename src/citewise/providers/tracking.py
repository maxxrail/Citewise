"""The one place that records LLM calls to `llm_calls`.

Every chat/embedding call made through `registry` passes through these wrappers, which
measure wall-clock latency, estimate cost from `pricing`, and write one row per call
(including failures, with `ok=False` and the error) in its own short transaction, so a
caller's rollback never erases the cost record.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from citewise.models.llm_call import LLMCall
from citewise.providers.base import (
    ChatProvider,
    ChatResult,
    EmbeddingBackend,
    JsonSchema,
    Message,
    Usage,
)
from citewise.providers.pricing import estimate_cost, require_price

log = logging.getLogger(__name__)


class CallRecorder:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def record(
        self,
        *,
        purpose: str,
        kind: str,
        provider: str,
        model: str,
        usage: Usage,
        latency_ms: int,
        ok: bool,
        error: str | None = None,
    ) -> None:
        row = LLMCall(
            purpose=purpose,
            kind=kind,
            provider=provider,
            model=model,
            tokens_in=usage.tokens_in,
            tokens_out=usage.tokens_out,
            cost_usd=estimate_cost(model, usage.tokens_in, usage.tokens_out),
            latency_ms=latency_ms,
            ok=ok,
            error=error[:2000] if error else None,
        )
        try:
            with self.session_factory() as session:
                session.add(row)
                session.commit()
        except Exception:  # never let bookkeeping kill the actual call
            log.exception("Failed to record llm_call (purpose=%s model=%s)", purpose, model)


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


class TrackedChatProvider:
    """Wraps a chat backend; same `ChatProvider` protocol, plus `llm_calls` logging."""

    def __init__(
        self, backend: ChatProvider, recorder: CallRecorder, *, require_prices: bool = False
    ) -> None:
        self.backend = backend
        self.recorder = recorder
        self.name = backend.name
        # Live mode sets this: an unpriced model would record $0 and blind the budget guard.
        self.require_prices = require_prices

    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        purpose: str = "general",
        json_schema: JsonSchema | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        if self.require_prices:
            require_price(model)  # before the call: nothing billed, nothing to record
        start = time.perf_counter()
        try:
            result = self.backend.complete(
                messages, model=model, purpose=purpose, json_schema=json_schema, tools=tools
            )
        except Exception as exc:
            self.recorder.record(
                purpose=purpose,
                kind="chat",
                provider=self.backend.name,
                model=model,
                usage=Usage(),
                latency_ms=_elapsed_ms(start),
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        result.latency_ms = _elapsed_ms(start)
        # Truncated/failed responses still bill tokens: record cost, but flag them not ok.
        completed = result.status == "completed"
        self.recorder.record(
            purpose=purpose,
            kind="chat",
            provider=result.provider,
            model=result.model,
            usage=result.usage,
            latency_ms=result.latency_ms,
            ok=completed,
            error=None if completed else f"response status: {result.status}",
        )
        return result


class TrackedEmbedder:
    """Implements `EmbeddingProvider` over a backend, logging each batch to `llm_calls`."""

    def __init__(
        self, backend: EmbeddingBackend, recorder: CallRecorder, *, require_prices: bool = False
    ) -> None:
        self.backend = backend
        self.recorder = recorder
        self.require_prices = require_prices
        self.ident = f"{backend.name}:{backend.model}"

    def embed(self, texts: list[str], *, purpose: str = "embed") -> list[list[float]]:
        if not texts:
            return []
        if self.require_prices:
            require_price(self.backend.model)
        start = time.perf_counter()
        try:
            result = self.backend.embed_with_usage(texts)
        except Exception as exc:
            self.recorder.record(
                purpose=purpose,
                kind="embed",
                provider=self.backend.name,
                model=self.backend.model,
                usage=Usage(),
                latency_ms=_elapsed_ms(start),
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        self.recorder.record(
            purpose=purpose,
            kind="embed",
            provider=result.provider,
            model=result.model,
            usage=result.usage,
            latency_ms=_elapsed_ms(start),
            ok=True,
        )
        return result.vectors
