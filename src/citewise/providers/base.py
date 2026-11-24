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
    status: str = "completed"  # anything else (e.g. "incomplete") is recorded as ok=False
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


def strict_schema_errors(schema: dict[str, Any], path: str = "$") -> list[str]:
    """Problems that would make OpenAI strict structured output reject `schema`.

    Strict mode requires every object to set `additionalProperties: false` and to list
    every property in `required` (optional fields use a `["type", "null"]` union instead).
    """
    errors: list[str] = []
    if schema.get("type") == "object" or "properties" in schema:
        props = schema.get("properties", {})
        if schema.get("additionalProperties") is not False:
            errors.append(f"{path}: additionalProperties must be false")
        missing = set(props) - set(schema.get("required", []))
        if missing:
            errors.append(f"{path}: required must list {sorted(missing)}")
        for key, sub in props.items():
            errors += strict_schema_errors(sub, f"{path}.{key}")
    if isinstance(schema.get("items"), dict):
        errors += strict_schema_errors(schema["items"], f"{path}[]")
    for sub in schema.get("anyOf", []):
        errors += strict_schema_errors(sub, path)
    for name, sub in schema.get("$defs", {}).items():
        errors += strict_schema_errors(sub, f"$defs.{name}")
    return errors


@dataclass(frozen=True)
class JsonSchema:
    """A named JSON schema for structured output (OpenAI requires a name).

    Validated at construction so mock mode rejects schemas the live API would reject.
    """

    name: str
    schema: dict[str, Any]
    strict: bool = True

    def __post_init__(self) -> None:
        if self.strict and (errors := strict_schema_errors(self.schema)):
            raise ValueError(f"Schema {self.name!r} is not strict-compatible: {errors}")


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
    """What callers use: plain vectors in, tracking handled underneath.

    `ident` names the vector space ("<provider>:<model>", e.g. "mock:text-embedding-3-small"
    vs "openai:text-embedding-3-small"). Vectors are only comparable within one ident, so
    it is stored next to every embedding.
    """

    ident: str

    def embed(self, texts: list[str], *, purpose: str = "embed") -> list[list[float]]: ...
