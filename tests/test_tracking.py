"""Acceptance: every provider call writes an `llm_calls` row with cost and latency."""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from citewise.core.config import Settings
from citewise.models import LLMCall
from citewise.providers import JsonSchema, get_chat_provider, get_embedder
from citewise.providers.base import ChatResult, Message, Usage
from citewise.providers.pricing import estimate_cost
from citewise.providers.tracking import CallRecorder, TrackedChatProvider

MSGS: list[Message] = [{"role": "user", "content": "Best CRM for startups?"}]


def _rows(db: Session) -> list[LLMCall]:
    db.expire_all()
    return list(db.scalars(select(LLMCall).order_by(LLMCall.id)))


@pytest.mark.parametrize("engine_name", ["openai", "anthropic", "gemini", "perplexity"])
def test_chat_call_writes_row_with_cost_and_latency(
    engine_name: str, settings: Settings, session_factory: sessionmaker[Session], db: Session
) -> None:
    provider = get_chat_provider(engine_name, settings=settings, session_factory=session_factory)
    result = provider.complete(MSGS, model="gpt-6-luna", purpose="test_chat")

    rows = _rows(db)
    assert len(rows) == 1
    row = rows[0]
    assert row.purpose == "test_chat"
    assert row.kind == "chat"
    assert row.provider == f"mock:{engine_name}"
    assert row.model == "gpt-6-luna"
    assert row.ok is True
    assert row.tokens_in == result.usage.tokens_in > 0
    assert row.tokens_out == result.usage.tokens_out > 0
    assert row.cost_usd == pytest.approx(estimate_cost("gpt-6-luna", row.tokens_in, row.tokens_out))
    assert row.cost_usd > 0
    assert row.latency_ms >= 0
    assert row.latency_ms == result.latency_ms


def test_structured_call_and_embedding_each_write_a_row(
    settings: Settings, session_factory: sessionmaker[Session], db: Session
) -> None:
    chat = get_chat_provider("openai", settings=settings, session_factory=session_factory)
    schema = JsonSchema(
        name="verdict",
        schema={
            "type": "object",
            "properties": {"label": {"enum": ["yes", "no"]}, "score": {"type": "number"}},
            "required": ["label", "score"],
            "additionalProperties": False,
        },
    )
    result = chat.complete(MSGS, model="gpt-6-luna", purpose="judge", json_schema=schema)
    assert result.parsed == {"label": "yes", "score": 0.0}

    vectors = get_embedder(settings=settings, session_factory=session_factory).embed(["a", "b"])
    assert len(vectors) == 2

    rows = _rows(db)
    assert [(r.kind, r.purpose) for r in rows] == [("chat", "judge"), ("embed", "embed")]
    embed_row = rows[1]
    assert embed_row.model == settings.model_embed
    assert embed_row.tokens_in > 0 and embed_row.tokens_out == 0
    assert embed_row.cost_usd > 0


def test_empty_embed_batch_makes_no_call(
    settings: Settings, session_factory: sessionmaker[Session], db: Session
) -> None:
    assert get_embedder(settings=settings, session_factory=session_factory).embed([]) == []
    assert _rows(db) == []


class _Boom:
    name = "boom"

    def complete(self, messages, *, model, purpose="general", json_schema=None, tools=None):
        raise RuntimeError("provider exploded")


def test_failed_call_is_recorded_then_reraised(
    session_factory: sessionmaker[Session], db: Session
) -> None:
    provider = TrackedChatProvider(_Boom(), CallRecorder(session_factory))
    with pytest.raises(RuntimeError, match="exploded"):
        provider.complete(MSGS, model="gpt-6-luna", purpose="fail_case")

    (row,) = _rows(db)
    assert row.ok is False
    assert row.provider == "boom"
    assert "RuntimeError: provider exploded" in (row.error or "")
    assert row.cost_usd == 0.0
    assert row.latency_ms >= 0


class _Echo:
    name = "echo"

    def complete(self, messages, *, model, purpose="general", json_schema=None, tools=None):
        return ChatResult(text="hi", model=model, provider=self.name)


def test_bookkeeping_failure_does_not_break_the_call() -> None:
    def broken_factory():
        raise RuntimeError("db down")

    provider = TrackedChatProvider(_Echo(), CallRecorder(broken_factory))  # type: ignore[arg-type]
    assert provider.complete(MSGS, model="x").text == "hi"


def test_stub_provider_failure_is_tracked_in_live_mode(
    session_factory: sessionmaker[Session], db: Session
) -> None:
    live = Settings(citewise_mode="live", _env_file=None)  # type: ignore[call-arg]
    # Gemini/Perplexity have no chat backend for Citewise's own calls (engines only).
    provider = get_chat_provider("gemini", settings=live, session_factory=session_factory)
    with pytest.raises(NotImplementedError):
        provider.complete(MSGS, model="gemini-3.5-flash-lite", purpose="engine")
    (row,) = _rows(db)
    assert row.ok is False and row.provider == "gemini"


class _Snapshot:
    """Live backends report the model they were asked for; raw keeps the dated snapshot."""

    name = "snap"

    def complete(self, messages, *, model, purpose="general", json_schema=None, tools=None):
        return ChatResult(
            text="ok",
            model=model,
            provider=self.name,
            usage=Usage(tokens_in=1000, tokens_out=1000),
            raw={"model": "gpt-6-luna-2026-11-01"},
        )


def test_dated_snapshot_ids_are_still_priced(
    session_factory: sessionmaker[Session], db: Session
) -> None:
    assert estimate_cost("gpt-6-luna-2026-11-01", 1000, 1000) == estimate_cost(
        "gpt-6-luna", 1000, 1000
    )
    provider = TrackedChatProvider(_Snapshot(), CallRecorder(session_factory))
    provider.complete(MSGS, model="gpt-6-luna")
    (row,) = _rows(db)
    assert row.cost_usd > 0


class _Truncated:
    name = "trunc"

    def complete(self, messages, *, model, purpose="general", json_schema=None, tools=None):
        return ChatResult(
            text="partial",
            model=model,
            provider=self.name,
            status="incomplete",
            usage=Usage(tokens_in=10, tokens_out=500),
        )


def test_incomplete_response_is_recorded_not_ok_but_billed(
    session_factory: sessionmaker[Session], db: Session
) -> None:
    provider = TrackedChatProvider(_Truncated(), CallRecorder(session_factory))
    assert provider.complete(MSGS, model="gpt-6-luna").text == "partial"
    (row,) = _rows(db)
    assert row.ok is False
    assert row.error == "response status: incomplete"
    assert row.tokens_out == 500 and row.cost_usd > 0
