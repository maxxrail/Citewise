"""Build the answer-engine adapter for a name, mock or live, from settings."""

from __future__ import annotations

from citewise.core.config import Settings, get_settings
from citewise.providers.engines.base import EngineAdapter
from citewise.providers.engines.mock import MockEngine
from citewise.providers.pricing import require_price
from citewise.providers.registry import PROVIDERS, MissingAPIKeyError, _require_key

ENGINES: tuple[str, ...] = PROVIDERS  # openai, anthropic, gemini, perplexity
DEFAULT_MAX_OUTPUT_TOKENS = 4096
# Old .env files copied from Module 0's .env.example still say "sonar".
RETIRED_PERPLEXITY_IDS = frozenset({"sonar", "sonar-pro", "sonar-reasoning"})


def engine_model(name: str, settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    return str(getattr(settings, f"engine_model_{name}"))


def max_output_tokens(name: str, settings: Settings) -> int:
    """Per-answer output cap; an engine missing from SCAN_MAX_OUTPUT_TOKENS gets 4096."""
    return int(settings.scan_max_output_tokens.get(name, DEFAULT_MAX_OUTPUT_TOKENS))


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
    if name == "perplexity" and model in RETIRED_PERPLEXITY_IDS:
        raise ValueError(
            f"ENGINE_MODEL_PERPLEXITY={model!r} is a retired Sonar Chat Completions ID "
            "(support ended 2026-09-27). Use 'perplexity/sonar' (Agent API) in .env."
        )
    require_price(model)  # fail closed before any spend we couldn't account for
    key = _require_key(settings, name)
    timeout = settings.llm_timeout_s
    max_out = max_output_tokens(name, settings)
    from citewise.providers.engines import live

    if name == "openai":
        return live.OpenAIEngine(key, model, timeout_s=timeout, max_output_tokens=max_out)
    if name == "anthropic":
        return live.AnthropicEngine(
            key,
            model,
            timeout_s=timeout,
            max_searches=settings.scan_max_searches,
            max_tokens=max_out,
        )
    if name == "gemini":
        return live.GeminiEngine(key, model, timeout_s=timeout, max_output_tokens=max_out)
    if name == "perplexity":
        return live.PerplexityEngine(key, model, timeout_s=timeout, max_output_tokens=max_out)
    raise ValueError(f"Unknown engine {name!r}")


__all__ = [
    "ENGINES",
    "MissingAPIKeyError",
    "engine_model",
    "get_engine",
    "max_output_tokens",
    "parse_engines",
]
