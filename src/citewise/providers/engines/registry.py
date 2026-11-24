"""Build the answer-engine adapter for a name, mock or live, from settings."""

from __future__ import annotations

from citewise.core.config import Settings, get_settings
from citewise.providers.engines.base import EngineAdapter
from citewise.providers.engines.mock import MockEngine
from citewise.providers.pricing import require_price
from citewise.providers.registry import PROVIDERS, MissingAPIKeyError, _require_key

ENGINES: tuple[str, ...] = PROVIDERS  # openai, anthropic, gemini, perplexity
ANSWER_MAX_TOKENS = 1500


def engine_model(name: str, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    return str(getattr(settings, f"engine_model_{name}"))


def parse_engines(value: str | list[str]) -> list[str]:
    """`"openai, gemini"` -> ["openai", "gemini"], validated and de-duplicated in order."""
    names = value.split(",") if isinstance(value, str) else value
    out: list[str] = []
    for n in (x.strip().lower() for x in names):
        if not n:
            continue
        if n not in ENGINES:
            raise ValueError(f"Unknown engine {n!r}; expected some of {', '.join(ENGINES)}")
        if n not in out:
            out.append(n)
    if not out:
        raise ValueError("No engines given")
    return out


def get_engine(name: str, settings: Settings | None = None) -> EngineAdapter:
    settings = settings or get_settings()
    model = engine_model(name, settings)
    if settings.is_mock:
        return MockEngine(name, model, settings.fixtures_dir)
    require_price(model)  # fail closed before any spend we couldn't account for
    key = _require_key(settings, name)
    timeout = settings.llm_timeout_s
    from citewise.providers.engines import live

    if name == "openai":
        return live.OpenAIEngine(key, model, timeout_s=timeout)
    if name == "anthropic":
        return live.AnthropicEngine(
            key,
            model,
            timeout_s=timeout,
            max_searches=settings.scan_max_searches,
            max_tokens=ANSWER_MAX_TOKENS,
        )
    if name == "gemini":
        return live.GeminiEngine(key, model, timeout_s=timeout)
    if name == "perplexity":
        return live.PerplexityEngine(key, model, timeout_s=timeout)
    raise ValueError(f"Unknown engine {name!r}")


__all__ = ["ENGINES", "MissingAPIKeyError", "engine_model", "get_engine", "parse_engines"]
