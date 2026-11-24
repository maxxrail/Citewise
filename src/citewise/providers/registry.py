"""Pick mock or live backends from config and always wrap them with call tracking."""

from __future__ import annotations

from typing import Literal

from sqlalchemy.orm import Session, sessionmaker

from citewise.core.config import Settings, get_settings
from citewise.core.db import get_sessionmaker
from citewise.providers.base import ChatProvider, EmbeddingBackend
from citewise.providers.mock import MockChatProvider, MockEmbedder
from citewise.providers.stubs import NotYetImplementedProvider
from citewise.providers.tracking import CallRecorder, TrackedChatProvider, TrackedEmbedder

ProviderName = Literal["openai", "anthropic", "gemini", "perplexity"]
PROVIDERS: tuple[str, ...] = ("openai", "anthropic", "gemini", "perplexity")


class MissingAPIKeyError(RuntimeError):
    pass


def _require_key(settings: Settings, name: str) -> str:
    secret = getattr(settings, f"{name}_api_key")
    if secret is None or not secret.get_secret_value():
        raise MissingAPIKeyError(
            f"{name.upper()}_API_KEY is not set. Add it to .env or use CITEWISE_MODE=mock."
        )
    return secret.get_secret_value()


# Which model IDs belong to which chat provider (a stale .env pinning gpt-* models while
# LLM_PROVIDER=anthropic would otherwise fail on the first paid call, mid-run).
_MODEL_PREFIXES = {"anthropic": ("claude-",), "openai": ("gpt-", "o1", "o3", "o4")}


def check_models_match_provider(settings: Settings) -> None:
    """Live mode only: every per-purpose model must belong to LLM_PROVIDER."""
    prefixes = _MODEL_PREFIXES[settings.llm_provider]
    wrong = {
        field: getattr(settings, field)
        for field in ("model_extract", "model_judge", "model_generate")
        if not str(getattr(settings, field)).startswith(prefixes)
    }
    if wrong:
        pins = ", ".join(f"{k.upper()}={v}" for k, v in wrong.items())
        raise ValueError(
            f"LLM_PROVIDER={settings.llm_provider} but {pins} in your environment/.env is not "
            f"a {settings.llm_provider} model. Remove those lines to use the defaults, or set "
            "LLM_PROVIDER to match."
        )


def get_chat_provider(
    name: ProviderName | str | None = None,
    *,
    settings: Settings | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> TrackedChatProvider:
    """The chat provider for Citewise's own LLM calls. `name` defaults to LLM_PROVIDER."""
    settings = settings or get_settings()
    name = name or settings.llm_provider
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider {name!r}; expected one of {PROVIDERS}")

    backend: ChatProvider
    if settings.is_mock:
        backend = MockChatProvider(settings.fixtures_dir, engine=name)
    elif name == "anthropic":
        check_models_match_provider(settings)
        from citewise.providers.anthropic_client import AnthropicChatProvider

        backend = AnthropicChatProvider(
            _require_key(settings, "anthropic"),
            timeout_s=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )
    elif name == "openai":
        check_models_match_provider(settings)
        from citewise.providers.openai_client import OpenAIChatProvider

        backend = OpenAIChatProvider(
            _require_key(settings, "openai"),
            timeout_s=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )
    else:
        backend = NotYetImplementedProvider(name)

    recorder = CallRecorder(session_factory or get_sessionmaker())
    return TrackedChatProvider(backend, recorder, require_prices=not settings.is_mock)


def get_embedder(
    *,
    settings: Settings | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> TrackedEmbedder:
    settings = settings or get_settings()
    backend: EmbeddingBackend
    if settings.is_mock:
        backend = MockEmbedder(settings.model_embed)
    elif settings.embed_provider == "local":
        from citewise.providers.local_embed import LocalEmbedder

        backend = LocalEmbedder(settings.model_embed)
    else:
        from citewise.providers.openai_client import OpenAIEmbedder

        backend = OpenAIEmbedder(
            _require_key(settings, "openai"),
            settings.model_embed,
            timeout_s=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )
    recorder = CallRecorder(session_factory or get_sessionmaker())
    return TrackedEmbedder(backend, recorder, require_prices=not settings.is_mock)
