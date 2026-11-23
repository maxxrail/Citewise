"""Prompt panel tables: the questions buyers ask AI assistants, and the topics they form."""

from __future__ import annotations

from datetime import datetime

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


class Topic(Base):
    """One cluster of a brand's prompts. Replaced wholesale on each `panel cluster` run."""

    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    brand_id: Mapped[int] = mapped_column(
        ForeignKey("brands.id", ondelete="CASCADE"), nullable=False, index=True
    )
    label: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    algorithm: Mapped[str] = mapped_column(String(32), nullable=False)  # "kmeans" | "hdbscan"
    # "llm" when the topic-label call validated, "fallback" when top terms were used instead.
    label_source: Mapped[str] = mapped_column(String(16), nullable=False, default="llm")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class Prompt(Base):
    """A candidate buyer question. Unique per brand by normalized text, so re-runs are no-ops."""

    __tablename__ = "prompts"
    __table_args__ = (UniqueConstraint("brand_id", "text_hash", name="uq_prompts_brand_text"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    brand_id: Mapped[int] = mapped_column(
        ForeignKey("brands.id", ondelete="CASCADE"), nullable=False, index=True
    )
    text: Mapped[str] = mapped_column(String(500), nullable=False)
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False)  # "generated" | "import"
    # e.g. "panel/generate.v1"; NULL for imported prompts.
    template_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    persona: Mapped[str | None] = mapped_column(String(100), nullable=True)
    funnel_stage: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # What the generator was asked for vs. what the classifier says the prompt actually is.
    requested_intent: Mapped[str | None] = mapped_column(String(16), nullable=True)
    intent: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    intent_source: Mapped[str | None] = mapped_column(String(16), nullable=True)
    topic_id: Mapped[int | None] = mapped_column(
        ForeignKey("topics.id", ondelete="SET NULL"), nullable=True, index=True
    )
    centrality: Mapped[float | None] = mapped_column(Float, nullable=True)  # cos to centroid
    embedding: Mapped[list[float] | None] = mapped_column(JSON, nullable=True)
    # Vector space of `embedding` ("<provider>:<model>"). Vectors from different spaces are
    # meaningless together (mock vs live share a model name), so stale ones are re-embedded.
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    in_panel: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
