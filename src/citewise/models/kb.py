"""Module 4 tables: the brand knowledge base (source of truth) and fact-check results."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from citewise.core.db import Base
from citewise.models.brand import utcnow

KB_CATEGORIES = ("company", "offerings", "audience", "competitors", "policies")


class KBDoc(Base):
    """One source-of-truth document. `slug` is stable across re-ingests (from the file or
    URL name), so chunk keys referenced by evals don't change when content does."""

    __tablename__ = "kb_docs"
    __table_args__ = (UniqueConstraint("brand_id", "slug", name="uq_kb_docs_brand_slug"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    brand_id: Mapped[int] = mapped_column(
        ForeignKey("brands.id", ondelete="CASCADE"), nullable=False, index=True
    )
    slug: Mapped[str] = mapped_column(String(200), nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    category: Mapped[str] = mapped_column(String(16), nullable=False)  # KB_CATEGORIES
    source: Mapped[str] = mapped_column(Text, nullable=False)  # file path or URL
    as_of: Mapped[str | None] = mapped_column(String(32), nullable=True)  # "2026-11-24"
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)  # skip unchanged
    text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class KBChunk(Base):
    __tablename__ = "kb_chunks"
    __table_args__ = (UniqueConstraint("brand_id", "chunk_key", name="uq_kb_chunks_brand_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    doc_id: Mapped[int] = mapped_column(
        ForeignKey("kb_docs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    brand_id: Mapped[int] = mapped_column(
        ForeignKey("brands.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_key: Mapped[str] = mapped_column(String(220), nullable=False)  # "<slug>#<ord>"
    ord: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)


class AnswerFactcheck(Base):
    """Per-answer marker: which fact-check version processed it (idempotency, like
    answer_extractions). `ok=False` means an LLM step failed; it is retried next run."""

    __tablename__ = "answer_factchecks"

    answer_id: Mapped[int] = mapped_column(
        ForeignKey("answers.id", ondelete="CASCADE"), primary_key=True
    )
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    ok: Mapped[bool] = mapped_column(Boolean, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class Claim(Base):
    """An atomic factual claim an engine made about the brand, and its verdict."""

    __tablename__ = "claims"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    answer_id: Mapped[int] = mapped_column(
        ForeignKey("answers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    brand_id: Mapped[int] = mapped_column(
        ForeignKey("brands.id", ondelete="CASCADE"), nullable=False, index=True
    )
    ord: Mapped[int] = mapped_column(Integer, nullable=False)  # order within the answer
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    claim_type: Mapped[str] = mapped_column(String(16), nullable=False)
    answer_span: Mapped[str] = mapped_column(Text, nullable=False)  # verbatim from the answer
    verdict: Mapped[str] = mapped_column(
        String(16), nullable=False
    )  # supported|contradicted|not_in_kb
    rationale: Mapped[str] = mapped_column(Text, nullable=False, default="")
    evidence_chunk_keys: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    evidence_quote: Mapped[str | None] = mapped_column(Text, nullable=True)  # verbatim KB text
    # SET NULL: re-ingesting citations must not delete the verdict history.
    source_citation_id: Mapped[int | None] = mapped_column(
        ForeignKey("citations.id", ondelete="SET NULL"), nullable=True
    )
    source_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
