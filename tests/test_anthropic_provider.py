"""Anthropic chat provider (Citewise's own LLM calls) and the local embedder, offline.

No network: the SDK client is replaced by a fake that records the request and returns
payloads shaped like the Messages API reference.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from pydantic import SecretStr
from sqlalchemy.orm import Session, sessionmaker

from citewise.core.config import Settings
from citewise.providers import get_chat_provider, get_embedder
from citewise.providers.anthropic_client import (
    AnthropicChatProvider,
    anthropic_schema,
    parse_response,
    split_system,
)
from citewise.providers.base import JsonSchema
from citewise.providers.registry import check_models_match_provider
from citewise.schemas.extract import MentionExtraction
from citewise.services.structured import complete_structured, schema_for

MSGS = [
    {"role": "system", "content": "You extract mentions."},
    {"role": "user", "content": "Answer: Nike is great."},
]
VALID = (
    '{"mentions": [{"entity": "Nike", "canonical": "Nike", "rank": null, "sentiment": 0.7,'
    ' "sentiment_label": "positive", "is_recommended": true, "snippet": "Nike is great."}]}'
)


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return self.payload


class FakeMessages:
    def __init__(self, *texts: str, stop: str = "end_turn") -> None:
        self.texts, self.stop, self.requests = list(texts), stop, []

    def create(self, **kwargs: Any) -> _Response:
        self.requests.append(kwargs)
        return _Response(
            {
                "model": kwargs["model"],
                "content": [
                    {"type": "thinking", "thinking": ""},  # adaptive thinking: not text
                    {"type": "text", "text": self.texts.pop(0)},
                ],
                "stop_reason": self.stop,
                "usage": {
                    "input_tokens": 100,
                    "cache_read_input_tokens": 20,
                    "output_tokens": 40,
                },
            }
        )


def _provider(fake: FakeMessages) -> AnthropicChatProvider:
    p = AnthropicChatProvider("sk-ant-test")
    p.client = types.SimpleNamespace(messages=fake)  # type: ignore[assignment]
    return p


def test_request_shape_system_and_structured_output() -> None:
    fake = FakeMessages(VALID)
    result = _provider(fake).complete(
        MSGS, model="claude-haiku-5-5", json_schema=schema_for(MentionExtraction)
    )
    [req] = fake.requests
    assert req["model"] == "claude-haiku-5-5"
    assert req["system"] == "You extract mentions."  # hoisted out of the messages
    assert [m["role"] for m in req["messages"]] == ["user"]
    fmt = req["output_config"]["format"]
    assert fmt["type"] == "json_schema" and "$defs" in fmt["schema"]  # $ref is supported
    assert req["max_tokens"] >= 4096  # adaptive thinking counts against it
    assert result.parsed is not None and result.parsed["mentions"][0]["canonical"] == "Nike"
    assert result.status == "completed" and result.provider == "anthropic"
    assert (result.usage.tokens_in, result.usage.tokens_out) == (120, 40)  # cache reads count


@pytest.mark.parametrize("stop", ["max_tokens", "refusal"])
def test_truncated_or_refused_output_is_not_completed(stop: str) -> None:
    result = parse_response(
        {
            "content": [{"type": "text", "text": '{"mentions": ['}],
            "stop_reason": stop,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        "claude-haiku-5-5",
        schema_for(MentionExtraction),
    )
    assert result.status == stop and result.parsed is None


def test_unsupported_constraints_are_stripped_but_property_names_kept() -> None:
    schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "maxLength": 10, "pattern": "^a"},
            "maximum": {"type": "integer", "minimum": 0, "maximum": 5},
            "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        },
        "required": ["pattern", "maximum", "tags"],
        "additionalProperties": False,
        "$defs": {"minimum": {"type": "number", "exclusiveMinimum": 0}},
    }
    out = anthropic_schema(schema)
    assert set(out["properties"]) == {"pattern", "maximum", "tags"}
    assert out["properties"]["pattern"] == {"type": "string"}
    assert out["properties"]["maximum"] == {"type": "integer"}
    assert out["properties"]["tags"] == {"type": "array", "items": {"type": "string"}}
    assert out["$defs"] == {"minimum": {"type": "number"}}
    assert out["additionalProperties"] is False and schema["properties"]["pattern"]["maxLength"]


def test_complete_structured_retries_with_a_user_turn_last() -> None:
    """No assistant prefill on current Claude models: the retry must end with a user turn."""
    fake = FakeMessages("not json", VALID)
    out = complete_structured(
        _provider(fake), MSGS, MentionExtraction, model="claude-haiku-5-5", purpose="t"
    )
    assert out is not None and out.mentions[0].canonical == "Nike"
    retry = fake.requests[1]["messages"]
    assert [m["role"] for m in retry] == ["user", "assistant", "user"]
    assert "failed validation" in retry[-1]["content"]


def test_split_system_joins_several_system_messages() -> None:
    system, rest = split_system(
        [{"role": "system", "content": "a"}, {"role": "system", "content": "b"}, MSGS[1]]
    )
    assert system == "a\n\nb" and rest == [MSGS[1]]
    assert split_system([MSGS[1]]) == (None, [MSGS[1]])


def test_live_registry_builds_anthropic_by_default(
    session_factory: sessionmaker[Session],
) -> None:
    live = Settings(citewise_mode="live", anthropic_api_key=SecretStr("sk-ant-test"),
                    _env_file=None)  # type: ignore[call-arg]  # fmt: skip
    chat = get_chat_provider(settings=live, session_factory=session_factory)
    assert isinstance(chat.backend, AnthropicChatProvider) and chat.require_prices
    assert get_embedder(settings=live, session_factory=session_factory).ident == (
        "local:BAAI/bge-small-en-v1.5"
    )


def test_stale_env_model_pins_fail_before_any_spend() -> None:
    # A .env copied before the switch still pins OpenAI models.
    live = Settings(citewise_mode="live", model_extract="gpt-6-luna", _env_file=None)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="MODEL_EXTRACT=gpt-6-luna"):
        check_models_match_provider(live)
    check_models_match_provider(Settings(_env_file=None))  # type: ignore[call-arg]  # defaults ok
    openai = Settings(llm_provider="openai", model_extract="gpt-6-luna", model_judge="gpt-6-luna",
                      model_generate="gpt-6.2-sol", _env_file=None)  # type: ignore[call-arg]  # fmt: skip
    check_models_match_provider(openai)


def test_local_embedder_uses_fastembed_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from citewise.providers.local_embed import LocalEmbedder

    loaded: list[str] = []

    class FakeTextEmbedding:
        def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
            loaded.append(model_name)

        def embed(self, texts: list[str], batch_size: int = 64) -> Any:
            return (np.full(384, 0.5, dtype=np.float32) for _ in texts)

    monkeypatch.setitem(
        sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTextEmbedding)
    )
    emb = LocalEmbedder()
    assert loaded == []  # nothing loaded (or downloaded) at construction
    result = emb.embed_with_usage(["a", "b c"])
    assert loaded == ["BAAI/bge-small-en-v1.5"]
    assert len(result.vectors) == 2 and len(result.vectors[0]) == 384
    assert isinstance(result.vectors[0][0], float)  # JSON-serialisable for prompts.embedding
    assert result.provider == "local" and result.usage.tokens_in > 0


def test_mock_mode_is_unchanged_by_the_provider_switch(settings: Settings) -> None:
    # Fixtures are keyed without the provider, so mock runs hit the same recordings.
    from citewise.providers.mock import MockChatProvider, fixture_key

    js = JsonSchema(name="X", schema={"type": "object", "properties": {}, "required": [],
                                      "additionalProperties": False})  # fmt: skip
    a = MockChatProvider(settings.fixtures_dir, engine="anthropic").complete(
        MSGS, model="m", json_schema=js
    )
    o = MockChatProvider(settings.fixtures_dir, engine="openai").complete(
        MSGS, model="m", json_schema=js
    )
    assert a.parsed == o.parsed and fixture_key("p", MSGS) == fixture_key("p", MSGS)
