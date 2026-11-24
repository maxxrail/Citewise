"""Module 4 knowledge base: chunker boundaries, ingest idempotency, RRF fusion math,
hybrid retrieval and re-rank validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from citewise.core.config import REPO_ROOT, Settings
from citewise.models.kb import KBChunk, KBDoc
from citewise.providers import get_embedder
from citewise.providers.base import ChatResult, Usage
from citewise.providers.mock import estimate_tokens
from citewise.services.brands import seed_brands
from citewise.services.kb.ingest import (
    IngestError,
    chunk_text,
    collect_docs,
    html_to_text,
    ingest,
    parse_file,
)
from citewise.services.kb.retrieve import RRF_K, KBIndex, rrf, tokenize
from citewise.services.panel.common import resolve_brand

KB = REPO_ROOT / "data" / "kb"


# ---------------------------------------------------------------- chunker


def _para(n: int, word: str = "word") -> str:
    """A paragraph of n sentences, each ~10 tokens."""
    return " ".join(f"Sentence {i} has some {word} text in it for testing." for i in range(n))


def test_short_doc_is_one_chunk_with_title_prefix() -> None:
    [c] = chunk_text("# Pricing\n\nPro costs $12 per user.", "Taskflow plans")
    assert c.text == "Taskflow plans > Pricing\nPro costs $12 per user." and c.ord == 0


def test_chunks_respect_the_token_budget_and_never_split_sentences() -> None:
    text = _para(60)
    chunks = chunk_text(text, "T", chunk_tokens=100, overlap_tokens=20)
    assert len(chunks) > 3
    body_sentences = set(text.split(". "))
    for c in chunks:
        assert c.tokens <= 100 + estimate_tokens("T\n") + 15  # one sentence of slack at most
        body = c.text.split("\n", 1)[1]
        assert all(s.rstrip(".") + "." in text for s in body.split(". ") if s)  # whole sentences
    assert body_sentences  # sanity


def test_consecutive_chunks_overlap_within_a_section() -> None:
    chunks = chunk_text(_para(40), "T", chunk_tokens=100, overlap_tokens=30)
    first_tail = chunks[0].text.split(". ")[-1]
    assert first_tail.rstrip(".") in chunks[1].text  # the next chunk repeats the tail
    assert chunks[1].text.count("Sentence") < 12  # but still advances


def test_a_new_section_starts_a_new_chunk_without_overlap() -> None:
    text = "# Plans\n\nFree is for 3 users.\n\n# Security\n\nSOC 2 Type II audited."
    a, b = chunk_text(text, "Doc", chunk_tokens=400)
    assert a.text == "Doc > Plans\nFree is for 3 users."
    assert b.text == "Doc > Security\nSOC 2 Type II audited."  # no carry-over across sections


def test_a_sentence_longer_than_a_chunk_is_hard_wrapped() -> None:
    long_sentence = " ".join(["supercalifragilistic"] * 200) + "."
    chunks = chunk_text(long_sentence, "T", chunk_tokens=50, overlap_tokens=0)
    assert len(chunks) > 5 and all(c.tokens <= 50 + 5 for c in chunks)


def test_front_matter_and_category_validation(tmp_path: Path) -> None:
    doc = parse_file(KB / "taskflow" / "offerings.md")
    assert (doc.slug, doc.category, doc.as_of) == ("offerings", "offerings", "2026-11-01")
    assert doc.title == "Taskflow plans, pricing and integrations"
    assert doc.source == "https://taskflow.example/pricing"
    bad = tmp_path / "x.md"
    bad.write_text("# No category\n\nText.", encoding="utf-8")
    with pytest.raises(IngestError, match="category"):
        parse_file(bad)
    assert parse_file(bad, category="company").category == "company"
    with pytest.raises(IngestError, match="unsupported"):
        parse_file(tmp_path / "a.docx")


def test_url_ingest_is_refused_in_mock_mode() -> None:
    with pytest.raises(IngestError, match="live"):
        collect_docs("https://linear.app/pricing", "offerings", live=False)


def test_html_to_text_drops_scripts_and_keeps_paragraphs() -> None:
    page = "<html><script>var x=1</script><h1>Plans</h1><p>Pro &amp; Business</p></html>"
    assert html_to_text(page) == "Plans\n\nPro & Business"


# ---------------------------------------------------------------- ingest


def _ingest_brand(db: Session, settings: Settings, name: str) -> Any:
    seed_brands(db, settings.brands_dir)
    brand = resolve_brand(db, name)
    docs = collect_docs(str(KB / name.lower()), None, live=False)
    return brand, docs


def test_ingest_is_idempotent_and_replaces_changed_docs(db: Session, settings: Settings) -> None:
    brand, docs = _ingest_brand(db, settings, "Taskflow")
    embedder = get_embedder(settings=settings, session_factory=lambda: db)  # type: ignore[arg-type]
    first = ingest(db, brand, docs, embedder)
    n_chunks = db.scalar(select(func.count()).select_from(KBChunk))
    assert first.docs_added == 5 and first.chunks == n_chunks and n_chunks >= 5
    keys = set(db.scalars(select(KBChunk.chunk_key)))
    assert "offerings#0" in keys and all("#" in k for k in keys)

    again = ingest(db, brand, docs, embedder)
    assert (again.docs_added, again.docs_updated, again.docs_unchanged) == (0, 0, 5)

    docs[0].text += "\n\nNew paragraph about the team."
    changed = ingest(db, brand, docs, embedder)
    assert changed.docs_updated == 1
    assert db.scalar(select(func.count()).select_from(KBDoc)) == 5  # updated, not duplicated


# ---------------------------------------------------------------- retrieval


def test_rrf_math() -> None:
    fused = dict(rrf([["a", "b", "c"], ["b", "c", "d"]]))
    assert fused["b"] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))
    assert fused["a"] == pytest.approx(1 / (RRF_K + 1))
    assert fused["d"] == pytest.approx(1 / (RRF_K + 3))
    order = [k for k, _ in rrf([["a", "b", "c"], ["b", "c", "d"]])]
    assert order == ["b", "c", "a", "d"]  # in both lists beats first in one
    assert rrf([]) == [] and rrf([["x"]], k=0) == [("x", 1.0)]


def test_tokenize_keeps_prices_and_drops_stopwords() -> None:
    assert tokenize("How much does the Pro plan cost? $12 per user.") == [
        "much", "pro", "plan", "cost", "$12", "per", "user",
    ]  # fmt: skip


def _index(db: Session, settings: Settings) -> tuple[KBIndex, Any]:
    brand, docs = _ingest_brand(db, settings, "Taskflow")
    embedder = get_embedder(settings=settings, session_factory=lambda: db)  # type: ignore[arg-type]
    ingest(db, brand, docs, embedder, chunk_tokens=80, overlap_tokens=10)
    return KBIndex.load(db, brand.id), embedder


def test_hybrid_retrieval_finds_the_fact(db: Session, settings: Settings) -> None:
    index, embedder = _index(db, settings)
    for mode in ("bm25", "dense", "hybrid"):
        hits = index.search("How much is Taskflow Pro per user per month?", embedder, mode=mode)
        assert hits and any("$12 per user per month" in h.text for h in hits[:3]), mode
    hit = index.search("Jira importer", embedder)[0]
    assert "importers for Jira" in hit.text and set(hit.ranks) <= {"bm25", "dense"}


def test_dense_retrieval_refuses_mixed_vector_spaces(db: Session, settings: Settings) -> None:
    index, _ = _index(db, settings)

    class Other:
        ident = "openai:text-embedding-3-small"

        def embed(self, texts: list[str], *, purpose: str = "") -> list[list[float]]:
            raise AssertionError("must not be called")

    with pytest.raises(ValueError, match="re-ingest"):
        index.search("price", Other(), mode="dense")  # type: ignore[arg-type]
    assert index.search("Jira importers", None, mode="bm25")  # BM25 needs no embedder


class _Reranker:
    name = "fake"

    def __init__(self, keys: list[str]) -> None:
        self.keys = keys

    def complete(self, messages: Any, *, model: str, **kw: Any) -> ChatResult:
        return ChatResult(text="", model=model, provider="fake",
                          parsed={"chunk_keys": self.keys}, usage=Usage(1, 1))  # fmt: skip


def test_rerank_reorders_but_only_within_the_candidates(db: Session, settings: Settings) -> None:
    index, embedder = _index(db, settings)
    plain = index.search("Taskflow security and data storage", embedder, k=5)
    target = plain[-1].chunk_key
    reranked = index.search(
        "Taskflow security and data storage", embedder, k=5,
        rerank=_Reranker([target, "made-up#9"]), rerank_model="claude-haiku-5-5",
    )  # fmt: skip
    assert reranked[0].chunk_key == target  # the model's pick first
    assert "made-up#9" not in {h.chunk_key for h in reranked}  # hallucinated ids dropped
    assert len(reranked) == 5
