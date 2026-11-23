from __future__ import annotations

import math

import pytest
from sqlalchemy.orm import Session, sessionmaker

from citewise.core.config import Settings
from citewise.providers import get_chat_provider
from citewise.providers.base import (
    ChatProvider,
    EmbeddingProvider,
    JsonSchema,
    Message,
    strict_schema_errors,
)
from citewise.providers.mock import MockChatProvider, fixture_key, mock_vector
from citewise.providers.openai_client import extract_url_citations
from citewise.providers.pricing import estimate_cost
from citewise.providers.registry import MissingAPIKeyError, get_embedder

FIXTURE_MSGS: list[Message] = [
    {"role": "user", "content": "Which project management tool is best for small teams?"}
]


def test_mock_chat_is_deterministic(settings: Settings) -> None:
    mock = MockChatProvider(settings.fixtures_dir)
    msgs: list[Message] = [{"role": "user", "content": "hello"}]
    a = mock.complete(msgs, model="m", purpose="p")
    b = mock.complete(msgs, model="m", purpose="p")
    c = mock.complete(msgs, model="m", purpose="other")
    assert a.text == b.text
    assert a.text != c.text


def test_mock_chat_loads_recorded_fixture(settings: Settings) -> None:
    mock = MockChatProvider(settings.fixtures_dir)
    result = mock.complete(FIXTURE_MSGS, model="m", purpose="example")
    # Generic fixture (no engine) is shared by every engine that has no specific one.
    assert result.raw == {"fixture_key": fixture_key("example", FIXTURE_MSGS), "fixture_hit": True}
    assert "Taskflow" in result.text
    assert [c.position for c in result.citations] == [1, 2]
    assert result.citations[0].url == "https://taskflow.example/pricing"


def test_mock_embedder_is_stable_unit_vectors() -> None:
    v1, v2 = mock_vector("hello"), mock_vector("hello")
    assert v1 == v2
    assert len(v1) == 1536
    assert math.isclose(sum(x * x for x in v1), 1.0, rel_tol=1e-9)
    assert mock_vector("hello") != mock_vector("world")
    # Pinned value guards against accidental changes to the seeding scheme.
    assert (round(v1[0], 6), round(v1[1], 6)) == (0.026209, -0.007888)


def test_registry_returns_protocol_conforming_wrappers(
    settings: Settings, session_factory: sessionmaker[Session]
) -> None:
    assert isinstance(get_chat_provider("openai", session_factory=session_factory), ChatProvider)
    assert isinstance(get_embedder(session_factory=session_factory), EmbeddingProvider)
    with pytest.raises(ValueError, match="Unknown provider"):
        get_chat_provider("bing", session_factory=session_factory)


def test_live_mode_without_key_fails_clearly(session_factory: sessionmaker[Session]) -> None:
    live = Settings(citewise_mode="live", openai_api_key=None, _env_file=None)  # type: ignore[call-arg]
    with pytest.raises(MissingAPIKeyError, match="OPENAI_API_KEY"):
        get_chat_provider("openai", settings=live, session_factory=session_factory)
    with pytest.raises(MissingAPIKeyError):
        get_embedder(settings=live, session_factory=session_factory)


def test_pricing() -> None:
    assert estimate_cost("gpt-6-luna", 1_000_000, 1_000_000) == pytest.approx(0.60)
    assert estimate_cost("text-embedding-3-small", 500_000, 0) == pytest.approx(0.01)
    assert estimate_cost("unknown-model", 1000, 1000) == 0.0


def test_extract_url_citations_dedupes_and_orders() -> None:
    raw = {
        "output": [
            {"type": "web_search_call"},
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [
                            {"type": "url_citation", "url": "https://a.com", "title": "A"},
                            {"type": "url_citation", "url": "https://b.com"},
                            {"type": "url_citation", "url": "https://a.com"},
                            {"type": "file_citation", "file_id": "f1"},
                        ],
                    }
                ],
            },
        ]
    }
    cites = extract_url_citations(raw)
    assert [(c.url, c.position) for c in cites] == [("https://a.com", 1), ("https://b.com", 2)]
    assert cites[0].title == "A"


def test_mock_engines_answer_differently_without_fixtures(settings: Settings) -> None:
    msgs: list[Message] = [{"role": "user", "content": "best running shoes?"}]
    texts = {
        MockChatProvider(settings.fixtures_dir, engine=e).complete(msgs, model="m").text
        for e in ("openai", "anthropic", "gemini", "perplexity")
    }
    assert len(texts) == 4


def test_engine_specific_fixture_wins_over_generic(settings: Settings, tmp_path) -> None:
    msgs: list[Message] = [{"role": "user", "content": "q"}]
    (tmp_path / "mock_chat").mkdir()
    (tmp_path / "mock_chat" / f"{fixture_key('p', msgs)}.json").write_text('{"text": "generic"}')
    specific = fixture_key("p", msgs, "gemini")
    (tmp_path / "mock_chat" / f"{specific}.json").write_text('{"text": "gemini only"}')

    assert MockChatProvider(tmp_path, "gemini").complete(msgs, model="m", purpose="p").text == (
        "gemini only"
    )
    assert MockChatProvider(tmp_path, "openai").complete(msgs, model="m", purpose="p").text == (
        "generic"
    )


def test_strict_schema_validation_matches_openai_rules() -> None:
    loose = {"type": "object", "properties": {"a": {"type": "string"}}}
    errors = strict_schema_errors(loose)
    assert any("additionalProperties" in e for e in errors)
    assert any("required" in e for e in errors)
    with pytest.raises(ValueError, match="not strict-compatible"):
        JsonSchema(name="loose", schema=loose)
    JsonSchema(name="loose_ok", schema=loose, strict=False)  # explicit opt-out allowed

    nested = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "object", "properties": {"x": {"type": "integer"}}},
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }
    assert strict_schema_errors(nested) == [
        "$.items[]: additionalProperties must be false",
        "$.items[]: required must list ['x']",
    ]
