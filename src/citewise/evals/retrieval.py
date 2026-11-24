"""Retrieval eval on `evals/datasets/golden/kb_qa.jsonl`.

For each query, retrieve within its brand's KB and score where the first gold chunk lands:
recall@1, recall@5 (a hit if any gold chunk is in the top k) and MRR (1 / rank of the
best-ranked gold chunk, 0 if outside the top 20), for BM25, dense, hybrid (RRF) and
hybrid + LLM re-rank, overall and per query style (lexical / paraphrase / claim).

The index is built straight from `data/kb/<brand>/` with the same chunker and embedder
as `kb ingest`, so the eval needs no database rows and always reflects the docs on disk.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from citewise.models.kb import KBChunk
from citewise.providers.base import ChatProvider, EmbeddingProvider
from citewise.services.kb.ingest import chunk_text, collect_docs
from citewise.services.kb.retrieve import KBIndex

METHODS = ("bm25", "dense", "hybrid", "hybrid+rerank")
POOL = 20


@dataclass
class Score:
    n: int = 0
    r1: int = 0
    r5: int = 0
    rr: float = 0.0
    misses: list[str] = field(default_factory=list)

    def add(self, row_id: str, rank: int | None) -> None:
        self.n += 1
        self.r1 += rank == 1
        self.r5 += rank is not None and rank <= 5
        self.rr += 1.0 / rank if rank else 0.0
        if rank is None or rank > 5:
            self.misses.append(row_id)

    def summary(self) -> dict[str, Any]:
        n = self.n or 1
        return {"n": self.n, "recall@1": round(self.r1 / n, 4), "recall@5": round(self.r5 / n, 4),
                "mrr": round(self.rr / n, 4)}  # fmt: skip


@dataclass
class RetrievalReport:
    mode: str
    rows: int
    reviewed: int
    overall: dict[str, Score]
    by_style: dict[str, dict[str, Score]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "rows": self.rows,
            "reviewed": self.reviewed,
            "overall": {m: s.summary() for m, s in self.overall.items()},
            "by_style": {
                st: {m: s.summary() for m, s in d.items()} for st, d in self.by_style.items()
            },
        }


def build_index(
    kb_dir: Path, embedder: EmbeddingProvider, *, chunk_tokens: int = 400, overlap_tokens: int = 60
) -> KBIndex:
    """Chunked exactly like `kb ingest` with the same settings, so gold keys line up."""
    chunks: list[KBChunk] = []
    for doc in collect_docs(str(kb_dir), None, live=False):
        for piece in chunk_text(doc.text, doc.title, chunk_tokens=chunk_tokens,
                                overlap_tokens=overlap_tokens):  # fmt: skip
            chunks.append(KBChunk(chunk_key=f"{doc.slug}#{piece.ord}", text=piece.text))
    for c, v in zip(
        chunks, embedder.embed([c.text for c in chunks], purpose="kb.embed"), strict=True
    ):
        c.embedding, c.embedding_model = v, embedder.ident
    return KBIndex(chunks)


def _rank(keys: list[str], gold: set[str]) -> int | None:
    return next((i for i, k in enumerate(keys, start=1) if k in gold), None)


def evaluate_retrieval(
    rows: list[dict[str, Any]],
    kb_root: Path,
    embedder: EmbeddingProvider,
    *,
    mode: str,
    rerank: ChatProvider | None = None,
    rerank_model: str | None = None,
    chunk_tokens: int = 400,
    overlap_tokens: int = 60,
) -> RetrievalReport:
    indexes: dict[str, KBIndex] = {}
    overall = {m: Score() for m in METHODS if m != "hybrid+rerank" or rerank is not None}
    by_style: dict[str, dict[str, Score]] = defaultdict(lambda: {m: Score() for m in overall})
    for row in rows:
        brand = row["brand"]
        if brand not in indexes:
            indexes[brand] = build_index(
                kb_root / brand.lower(),
                embedder,
                chunk_tokens=chunk_tokens,
                overlap_tokens=overlap_tokens,
            )
        index, gold = indexes[brand], set(row["gold_chunk_keys"])
        for method in overall:
            if method == "hybrid+rerank":
                hits = index.search(row["query"], embedder, k=POOL, mode="hybrid", pool=POOL,
                                    rerank=rerank, rerank_model=rerank_model)  # fmt: skip
            else:
                hits = index.search(row["query"], embedder, k=POOL, mode=method, pool=POOL)  # type: ignore[arg-type]
            rank = _rank([h.chunk_key for h in hits], gold)
            overall[method].add(row["id"], rank)
            by_style[row.get("style", "other")][method].add(row["id"], rank)
    return RetrievalReport(mode, len(rows), sum(1 for r in rows if r.get("reviewed")),
                           overall, dict(by_style))  # fmt: skip


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def render_markdown(report: RetrievalReport, golden: str, date: str, embedder: str) -> str:
    methods = list(report.overall)
    lines = [
        "# KB retrieval eval",
        "",
        f"Golden set: `{golden}`: {report.rows} queries over 3 brand KBs, {report.reviewed} "
        f"hand-reviewed. Mode: **{report.mode}**, embedder `{embedder}`. Generated {date} by "
        "`citewise eval retrieval`.",
        "",
        "| Method | Recall@1 | Recall@5 | MRR |",
        "|---|---|---|---|",
    ]
    for m in methods:
        s = report.overall[m].summary()
        lines.append(f"| {m} | {s['recall@1']:.1%} | {s['recall@5']:.1%} | {s['mrr']:.3f} |")
    lines += ["", "## By query style (MRR)", "",
              "| Style | n | " + " | ".join(methods) + " |",
              "|---|---|" + "---|" * len(methods)]  # fmt: skip
    for style, scores in sorted(report.by_style.items()):
        n = next(iter(scores.values())).n
        lines.append(
            f"| {style} | {n} | "
            + " | ".join(f"{scores[m].summary()['mrr']:.3f}" for m in methods)
            + " |"
        )
    lines += ["", "## Misses (gold chunk not in the top 5)", ""]
    for m in methods:
        miss = report.overall[m].misses
        lines.append(f"- **{m}**: {', '.join(miss) if miss else 'none'}")
    return "\n".join(lines) + "\n"
