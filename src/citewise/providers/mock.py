"""Deterministic offline providers used in tests, CI and the public demo.

Chat: the response is keyed by sha256 of (purpose, messages). If
`<fixtures_dir>/mock_chat/<key>.json` exists it is returned verbatim; otherwise a
deterministic placeholder is generated (schema-valid when a JSON schema is requested),
so new code paths work in mock mode before anyone records a fixture.

Embeddings: feature hashing of words, word bigrams and character trigrams into a signed
1536-dim vector, then unit length. Texts that share words get high cosine similarity, so
dedup and clustering behave like they would on real embeddings (random vectors would make
every prompt equidistant). Uses sha256, never `hash()`, so it is stable across runs,
platforms and Python versions.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import re
from pathlib import Path
from typing import Any

from citewise.providers import mock_content, mock_extract, mock_factcheck
from citewise.providers.base import (
    ChatResult,
    Citation,
    EmbedResult,
    JsonSchema,
    Message,
    Usage,
)

MOCK_EMBED_DIM = 1536  # same shape as text-embedding-3-small


def fixture_key(purpose: str, messages: list[Message], engine: str | None = None) -> str:
    """Stable key for a mock chat fixture.

    `engine=None` gives the generic key shared by all engines; passing an engine gives an
    engine-specific key that wins when present. Model is excluded so model swaps don't
    break fixtures.
    """
    body: dict[str, Any] = {"purpose": purpose, "messages": messages}
    if engine is not None:
        body["engine"] = engine
    payload = json.dumps(body, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) so mock calls carry realistic usage."""
    return max(1, math.ceil(len(text) / 4))


def _resolve_ref(ref: str, root: dict[str, Any]) -> dict[str, Any]:
    """Resolve a local `#/$defs/Name` pointer (what Pydantic's `model_json_schema` emits)."""
    if not ref.startswith("#/"):
        raise ValueError(f"Mock only supports local $refs, got {ref!r}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def _example_for_schema(
    schema: dict[str, Any], seed: str, root: dict[str, Any] | None = None
) -> Any:
    """Build a minimal instance satisfying a JSON schema (deterministic).

    `root` is the top-level schema, needed to resolve `$ref`s: Pydantic puts every nested
    model in `$defs`, so without this nested objects would come back as strings.
    """
    root = root if root is not None else schema
    if "$ref" in schema:
        return _example_for_schema(_resolve_ref(schema["$ref"], root), seed, root)
    if "enum" in schema:
        return schema["enum"][0]
    if "const" in schema:
        return schema["const"]
    if "anyOf" in schema:
        # Prefer a non-null branch so optional nested objects are exercised in mock mode.
        branches = schema["anyOf"]
        branch = next((b for b in branches if b.get("type") != "null"), branches[0])
        return _example_for_schema(branch, seed, root)
    typ = schema.get("type", "string")
    if isinstance(typ, list):
        typ = next((t for t in typ if t != "null"), "null")
    if typ == "object":
        props: dict[str, Any] = schema.get("properties", {})
        return {k: _example_for_schema(v, f"{seed}.{k}", root) for k, v in props.items()}
    if typ == "array":
        n = max(1, int(schema.get("minItems", 1)))
        items = schema.get("items", {})
        return [_example_for_schema(items, f"{seed}[{i}]", root) for i in range(n)]
    if typ == "integer":
        return int(schema.get("minimum", 0))
    if typ == "number":
        return float(schema.get("minimum", 0.0))
    if typ == "boolean":
        return False
    if typ == "null":
        return None
    return f"mock-{hashlib.sha256(seed.encode()).hexdigest()[:8]}"


class MockChatProvider:
    """Mock for one engine. Engines differ in mock mode, so per-engine metrics stay meaningful."""

    def __init__(self, fixtures_dir: Path, engine: str = "openai") -> None:
        self.fixtures_dir = fixtures_dir / "mock_chat"
        self.engine = engine
        self.name = f"mock:{engine}"

    def _find_fixture(self, purpose: str, messages: list[Message]) -> tuple[str, Path | None]:
        """Engine-specific fixture first, then the generic one. Returns (key, path or None)."""
        engine_key = fixture_key(purpose, messages, self.engine)
        for key in (engine_key, fixture_key(purpose, messages)):
            path = self.fixtures_dir / f"{key}.json"
            if path.exists():
                return key, path
        return engine_key, None

    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        purpose: str = "general",
        json_schema: JsonSchema | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        key, path = self._find_fixture(purpose, messages)
        prompt_text = "\n".join(m["content"] for m in messages)

        if path is not None:
            data = json.loads(path.read_text(encoding="utf-8"))
            text = data.get("text", "")
            parsed = data.get("parsed")
            citations = [Citation(**c) for c in data.get("citations", [])]
        elif json_schema is not None and purpose in mock_content.PURPOSES:
            parsed = mock_content.simulate(purpose, messages)
            text = json.dumps(parsed)
            citations = []
        elif json_schema is not None and purpose in mock_factcheck.PURPOSES:
            parsed = mock_factcheck.simulate(purpose, messages)
            text = json.dumps(parsed)
            citations = []
        elif json_schema is not None and purpose == mock_extract.PURPOSE:
            # A placeholder would make offline extraction meaningless; simulate a model.
            parsed = mock_extract.simulate_extraction(messages)
            text = json.dumps(parsed)
            citations = []
        elif json_schema is not None:
            parsed = _example_for_schema(json_schema.schema, seed=key)
            text = json.dumps(parsed)
            citations = []
        else:
            parsed = None
            text = f"[{self.name}:{purpose}:{key}] Deterministic placeholder answer."
            citations = []

        if json_schema is not None and parsed is None and text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None

        return ChatResult(
            text=text,
            model=model,
            provider=self.name,
            parsed=parsed,
            citations=citations,
            usage=Usage(tokens_in=estimate_tokens(prompt_text), tokens_out=estimate_tokens(text)),
            latency_ms=0,
            raw={"fixture_key": key, "fixture_hit": path is not None},
        )


_WORD = re.compile(r"[a-z0-9]+")
# Function words carry no topic; without this "how do i" would pull unrelated prompts together.
_STOPWORDS = frozenset(
    {
        *("a", "an", "and", "are", "as", "at", "be", "best", "can", "do", "does", "for"),
        *("from", "good", "how", "i", "in", "is", "it", "me", "my", "of", "on", "or"),
        *("should", "the", "to", "vs", "what", "whats", "which", "with", "you", "your"),
    }
)
# Share of every vector put on one reserved "anchor" dimension: cos' = A + (1 - A) * cos.
# Real embeddings (text-embedding-3-small) put paraphrases around 0.9+ and related-but-
# different questions around 0.8-0.9; lexical features alone sit lower (~0.85 / ~0.7).
# The anchor lifts the scale so the production dedup threshold (0.92) means roughly the
# same thing in mock mode. It is a constant offset, so rankings and clusters don't change.
_ANCHOR = 0.5
_WEIGHTS = {"word": 1.0, "char3": 0.8, "bigram": 0.3}


def _stem(word: str) -> str:
    """Crude suffix strip so "shoes"/"shoe" and "running"/"run" share a feature."""
    for suffix in ("ing", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _features(text: str) -> dict[str, float]:
    """Weighted sparse features: content words, character trigrams (typo-tolerant), bigrams."""
    words = [_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOPWORDS]
    feats: dict[str, float] = {}

    def add(key: str, weight: float) -> None:
        feats[key] = feats.get(key, 0.0) + weight

    for w in words:
        add(f"w:{w}", _WEIGHTS["word"])
        padded = f"_{w}_"
        for i in range(len(padded) - 2):
            add(f"c:{padded[i : i + 3]}", _WEIGHTS["char3"])
    for a, b in itertools.pairwise(words):
        add(f"b:{a}_{b}", _WEIGHTS["bigram"])
    return feats


def _random_unit_vector(text: str, dim: int) -> list[float]:
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    vec = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _hashed_vector(text: str, dim: int) -> list[float] | None:
    """Signed feature hashing into dims 1..dim-1 (dim 0 is the anchor). None if empty."""
    vec = [0.0] * dim
    for feat, weight in _features(text).items():
        digest = hashlib.sha256(feat.encode("utf-8")).digest()
        idx = 1 + int.from_bytes(digest[:4], "big") % (dim - 1)
        vec[idx] += weight if digest[4] & 1 else -weight
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:  # no content words (e.g. "how do i?") or collisions cancelled out
        return None
    return [x / norm for x in vec]


def mock_vector(text: str, dim: int = MOCK_EMBED_DIM) -> list[float]:
    """Deterministic, roughly semantic unit vector.

    Texts with no content words fall back to a random vector (they match nothing).
    """
    base = _hashed_vector(text, dim)
    if base is None:
        base = _random_unit_vector(text, dim)
        base[0] = 0.0
        norm = math.sqrt(sum(x * x for x in base)) or 1.0
        base = [x / norm for x in base]
    scale = math.sqrt(1.0 - _ANCHOR)
    vec = [x * scale for x in base]
    vec[0] = math.sqrt(_ANCHOR)
    return vec


class MockEmbedder:
    name = "mock"

    def __init__(self, model: str, dim: int = MOCK_EMBED_DIM) -> None:
        self.model = model
        self.dim = dim

    def embed_with_usage(self, texts: list[str]) -> EmbedResult:
        return EmbedResult(
            vectors=[mock_vector(t, self.dim) for t in texts],
            model=self.model,
            provider=self.name,
            usage=Usage(tokens_in=sum(estimate_tokens(t) for t in texts)),
        )
