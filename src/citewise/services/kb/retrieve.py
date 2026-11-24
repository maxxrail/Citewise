"""Hybrid retrieval over a brand's KB chunks.

- **BM25** (rank_bm25, Okapi) on lowercased word tokens: exact terms, numbers, plan names
  ("$16", "Business", "SAML"), which is what fact-checking a claim mostly needs.
- **Dense**: cosine between the query and chunk embeddings: paraphrases ("how much
  does it cost" vs "per user per month").
- **Hybrid**: reciprocal rank fusion of the two rankings, score = sum 1/(k + rank), k=60
  (Cormack et al. 2009). RRF needs no score calibration between BM25 and cosine, whose
  scales are unrelated.
- **Re-rank (optional)**: an LLM orders the fused top `pool` (20) and keeps the top k. Its
  output must be a subset of the candidates it was shown (validated); otherwise the fused
  order stands.

A brand's KB is small (tens to hundreds of chunks), so everything runs in memory per query.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict
from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session

from citewise.models.kb import KBChunk
from citewise.providers.base import ChatProvider, EmbeddingProvider
from citewise.services.prompt_library import render
from citewise.services.structured import complete_structured

Mode = Literal["bm25", "dense", "hybrid"]
RRF_K = 60
RERANK_PURPOSE = "kb.rerank"
_TOKEN = re.compile(r"[a-z0-9$€£%.]+")
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "does",
        "do",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "the",
        "to",
        "what",
        "which",
        "who",
        "with",
        "you",
        "your",
        "can",
    ]
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens; keeps "$16", "2.0" and "50,000" -> "50" "000" as terms."""
    return [t.strip(".") for t in _TOKEN.findall(text.lower()) if t.strip(".") not in _STOP]


@dataclass
class Hit:
    chunk_key: str
    text: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # method -> 1-based rank


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: each list is item ids, best first. Ties keep first-seen order."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


class _Rerank(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunk_keys: list[str]  # most relevant first


class KBIndex:
    """In-memory index of one brand's chunks (cheap to rebuild per command)."""

    def __init__(self, chunks: list[KBChunk]) -> None:
        self.keys = [c.chunk_key for c in chunks]
        self.texts = {c.chunk_key: c.text for c in chunks}
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None
        vecs = [c.embedding for c in chunks]
        self.matrix = (
            _normalize(np.array(vecs, dtype=float)) if chunks and all(v for v in vecs) else None
        )
        self.spaces = {c.embedding_model for c in chunks}

    @classmethod
    def load(cls, session: Session, brand_id: int) -> KBIndex:
        chunks = list(
            session.scalars(
                select(KBChunk).where(KBChunk.brand_id == brand_id).order_by(KBChunk.chunk_key)
            )
        )
        return cls(chunks)

    def __len__(self) -> int:
        return len(self.keys)

    def rank_bm25(self, query: str) -> list[str]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.keys)), key=lambda i: (-scores[i], i))
        return [self.keys[i] for i in order if scores[i] > 0]

    def rank_dense(self, query_vec: list[float]) -> list[str]:
        if self.matrix is None:
            return []
        sims = self.matrix @ _normalize(np.array([query_vec], dtype=float))[0]
        return [self.keys[i] for i in sorted(range(len(self.keys)), key=lambda i: (-sims[i], i))]

    def search(
        self,
        query: str,
        embedder: EmbeddingProvider | None,
        *,
        k: int = 5,
        mode: Mode = "hybrid",
        pool: int = 20,
        rerank: ChatProvider | None = None,
        rerank_model: str | None = None,
    ) -> list[Hit]:
        if not self.keys:
            return []
        bm25 = self.rank_bm25(query) if mode in ("bm25", "hybrid") else []
        dense: list[str] = []
        if mode in ("dense", "hybrid"):
            if embedder is None:
                raise ValueError("dense/hybrid retrieval needs an embedder")
            if self.spaces != {embedder.ident}:
                raise ValueError(
                    f"KB chunks are embedded with {sorted(map(str, self.spaces))} but the "
                    f"query embedder is {embedder.ident}: re-ingest with `citewise kb ingest`"
                )
            dense = self.rank_dense(embedder.embed([query], purpose="kb.query")[0])
        if mode == "bm25":
            fused = [(key, 1.0 / (RRF_K + i)) for i, key in enumerate(bm25, start=1)]
        elif mode == "dense":
            fused = [(key, 1.0 / (RRF_K + i)) for i, key in enumerate(dense, start=1)]
        else:
            fused = rrf([bm25[:pool], dense[:pool]])
        ranks = {
            "bm25": {key: i for i, key in enumerate(bm25, start=1)},
            "dense": {key: i for i, key in enumerate(dense, start=1)},
        }
        candidates = fused[:pool] if rerank is not None else fused[:k]
        if rerank is not None and rerank_model and len(candidates) > 1:
            order = self._rerank(query, [key for key, _ in candidates], rerank, rerank_model)
            if order:
                by_key = dict(candidates)
                rest = [key for key, _ in candidates if key not in order]
                candidates = [(key, by_key[key]) for key in [*order, *rest]]
        return [
            Hit(key, self.texts[key], score,
                {m: r[key] for m, r in ranks.items() if key in r})
            for key, score in candidates[:k]
        ]  # fmt: skip

    def _rerank(
        self, query: str, keys: list[str], chat: ChatProvider, model: str
    ) -> list[str] | None:
        rendered = render(
            "kb/rerank",
            "v1",
            query=query,
            passages=[{"id": key, "text": self.texts[key]} for key in keys],
        )
        out = complete_structured(
            chat, rendered.messages, _Rerank, model=model, purpose=RERANK_PURPOSE
        )
        if out is None:
            return None
        allowed = set(keys)
        order = [key for key in dict.fromkeys(out.chunk_keys) if key in allowed]
        return order or None


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return np.divide(m, norms, out=np.zeros_like(m), where=norms > 0)
