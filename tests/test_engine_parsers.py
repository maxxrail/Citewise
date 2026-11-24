"""Live adapters' response parsers, tested offline against payloads shaped like each
provider's docs (checked 2026-11-23). No network, no keys."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from citewise.core.config import Settings
from citewise.providers.engines.base import EngineAdapter, EngineError
from citewise.providers.engines.live import (
    parse_anthropic,
    parse_gemini,
    parse_openai,
    parse_perplexity,
)
from citewise.providers.engines.mock import MockEngine
from citewise.providers.engines.registry import get_engine, parse_engines
from citewise.providers.pricing import SEARCH_FEE_USD, price_for, search_cost
from citewise.providers.registry import MissingAPIKeyError

FIXTURES = Path(__file__).parent / "fixtures" / "engine_payloads"


def _payload(name: str) -> dict:
    """Provider responses shaped like each provider's docs (checked 2026-11-23). The
    Anthropic one is the docs' own example; the others follow the documented fields."""
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


OPENAI = _payload("openai")
ANTHROPIC = _payload("anthropic")
GEMINI_REST = _payload("gemini_rest")
PERPLEXITY = _payload("perplexity")


def test_parse_openai() -> None:
    a = parse_openai(OPENAI, "gpt-6-luna")
    assert a.text == "Popular options include HubSpot and Zoho."
    assert [(c.position, c.title) for c in a.citations] == [(1, "HubSpot CRM"), (2, "Zoho CRM")]
    assert (a.usage.tokens_in, a.usage.tokens_out, a.searches) == (1834, 212, 1)
    assert a.model == "gpt-6-luna"  # priced by the requested alias, not the snapshot


def test_parse_openai_incomplete_is_retryable_error() -> None:
    with pytest.raises(EngineError) as info:
        parse_openai({**OPENAI, "status": "incomplete"}, "gpt-6-luna")
    assert info.value.retryable


def test_parse_anthropic_joins_text_and_collects_cited_sources() -> None:
    a = parse_anthropic(ANTHROPIC, "claude-haiku-5-5")
    assert a.text.endswith("born on April 30, 1916, in Petoskey, Michigan")
    assert a.text.startswith("I'll search")
    assert [c.url for c in a.citations] == ["https://en.wikipedia.org/wiki/Claude_Shannon"]
    assert (a.usage.tokens_in, a.usage.tokens_out, a.searches) == (6039, 931, 1)


def test_parse_anthropic_search_error_without_text() -> None:
    error = {"type": "web_search_tool_result_error", "error_code": "too_many_requests"}
    payload = {"content": [{"type": "web_search_tool_result", "content": error}], "usage": {}}
    with pytest.raises(EngineError) as info:
        parse_anthropic(payload, "claude-haiku-5-5")
    assert info.value.retryable
    payload["content"][0]["content"]["error_code"] = "invalid_tool_input"  # type: ignore[index]
    with pytest.raises(EngineError) as info:
        parse_anthropic(payload, "claude-haiku-5-5")
    assert not info.value.retryable


@pytest.mark.parametrize("snake", [False, True])
def test_parse_gemini_rest_and_sdk_shapes(snake: bool) -> None:
    payload = GEMINI_REST
    if snake:  # what google-genai's model_dump() returns
        import re

        def to_snake(o):  # type: ignore[no-untyped-def]
            if isinstance(o, dict):
                return {
                    re.sub(r"(?<!^)(?=[A-Z])", "_", k).lower(): to_snake(v) for k, v in o.items()
                }
            if isinstance(o, list):
                return [to_snake(x) for x in o]
            return o

        payload = to_snake(GEMINI_REST)
    a = parse_gemini(payload, "gemini-3.5-flash-lite")
    assert a.text == "Spain won Euro 2024, beating England 2-1."
    assert [c.title for c in a.citations] == ["uefa.com", "aljazeera.com"]  # first support wins
    assert a.searches == 2  # each grounding query is billed
    assert (a.usage.tokens_in, a.usage.tokens_out) == (12 + 310, 18 + 4)


def test_parse_gemini_max_tokens_is_retryable() -> None:
    payload = {"candidates": [{**GEMINI_REST["candidates"][0], "finishReason": "MAX_TOKENS"}]}  # type: ignore[dict-item]
    with pytest.raises(EngineError):
        parse_gemini(payload, "m")


def test_parse_perplexity_orders_by_markers_and_uses_reported_cost() -> None:
    a = parse_perplexity(PERPLEXITY, "perplexity/sonar")
    assert [c.title for c in a.citations] == ["Zoho CRM", "HubSpot CRM"]  # marker order, deduped
    assert a.reported_cost_usd == pytest.approx(0.003175)
    assert (a.usage.tokens_in, a.usage.tokens_out, a.searches) == (1200, 150, 1)


def test_parse_perplexity_without_markers_keeps_all_sources() -> None:
    message = {"type": "message", "content": [{"type": "output_text", "text": "No markers."}]}
    payload = {**PERPLEXITY, "output": [PERPLEXITY["output"][0], message]}
    a = parse_perplexity(payload, "perplexity/sonar")
    assert [c.title for c in a.citations] == ["HubSpot CRM", "Zoho CRM", "Unused"]


def test_registry_mock_and_live_key_checks(settings: Settings) -> None:
    for name in ("openai", "anthropic", "gemini", "perplexity"):
        engine = get_engine(name, settings)
        assert isinstance(engine, MockEngine) and isinstance(engine, EngineAdapter)
        assert engine.model == getattr(settings, f"engine_model_{name}")
    live = settings.model_copy(update={"citewise_mode": "live", "anthropic_api_key": None})
    with pytest.raises(MissingAPIKeyError):
        get_engine("anthropic", live)
    with pytest.raises(ValueError, match="Unknown engine"):
        parse_engines("openai,bing")
    assert parse_engines(" OpenAI, gemini ,openai") == ["openai", "gemini"]


def test_default_engine_models_are_priced_and_have_search_fees(settings: Settings) -> None:
    for name in ("openai", "anthropic", "gemini", "perplexity"):
        assert price_for(getattr(settings, f"engine_model_{name}")) is not None
        assert SEARCH_FEE_USD[name] > 0
    # Check the code defaults: a developer's local .env may pin older IDs (and override these).
    defaults = {k: f.default for k, f in Settings.model_fields.items()}
    assert defaults["engine_model_gemini"] == "gemini-3.5-flash-lite"
    assert defaults["engine_model_perplexity"] == "perplexity/sonar"
    assert search_cost("openai", 3) == pytest.approx(0.03)


def test_retired_perplexity_id_fails_clearly_in_live_mode(settings: Settings) -> None:
    from pydantic import SecretStr

    live = settings.model_copy(
        update={
            "citewise_mode": "live",
            "engine_model_perplexity": "sonar",
            "perplexity_api_key": SecretStr("pplx-test"),
        }
    )
    with pytest.raises(ValueError, match="perplexity/sonar"):
        get_engine("perplexity", live)
