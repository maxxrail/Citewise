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


def get_chat_provider(
    name: ProviderName | str = "openai",
    *,
    settings: Settings | None = None,
    session_factory: sessionmaker[Session] | None = None,
) -> TrackedChatProvider:
    settings = settings or get_settings()
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider {name!r}; expected one of {PROVIDERS}")

    backend: ChatProvider
    if settings.is_mock:
        backend = MockChatProvider(settings.fixtures_dir, engine=name)
    elif name == "openai":
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
