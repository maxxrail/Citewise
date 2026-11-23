"""Provider-agnostic types and protocols.

Two layers implement the same chat protocol:
- *backends* (mock, OpenAI, ...) talk to the outside world;
- `tracking.TrackedChatProvider` wraps a backend and writes one `llm_calls` row per call.
Callers only ever get the tracked wrapper from `registry`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class Usage:
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass(frozen=True)
class Citation:
    url: str
    title: str | None = None
    position: int = 0  # 1-based order of appearance in the answer


@dataclass
class ChatResult:
    text: str
    model: str
    provider: str
    parsed: dict[str, Any] | None = None  # set when json_schema was requested and parsed OK
    citations: list[Citation] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    latency_ms: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbedResult:
    vectors: list[list[float]]
    model: str
    provider: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: int = 0


@dataclass(frozen=True)
class JsonSchema:
    """A named JSON schema for structured output (OpenAI requires a name)."""

    name: str
    schema: dict[str, Any]


@runtime_checkable
class ChatProvider(Protocol):
    name: str

    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        purpose: str = "general",
        json_schema: JsonSchema | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult: ...


class EmbeddingBackend(Protocol):
    """What a raw embedding client implements (returns usage for cost tracking)."""

    name: str
    model: str

    def embed_with_usage(self, texts: list[str]) -> EmbedResult: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """What callers use: plain vectors in, tracking handled underneath."""

    def embed(self, texts: list[str], *, purpose: str = "embed") -> list[list[float]]: ...
