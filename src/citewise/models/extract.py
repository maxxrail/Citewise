"""Extraction tables (Module 3): structured mentions per answer, plus one extraction record
per answer saying which extractor version produced them and where its stages disagreed."""

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


class AnswerExtraction(Base):
    """One row per extracted answer. Its presence (at the current `version`) is what makes
    `extract run` idempotent: answers that already have one are skipped."""

    __tablename__ = "answer_extractions"

    answer_id: Mapped[int] = mapped_column(
        ForeignKey("answers.id", ondelete="CASCADE"), primary_key=True
    )
    version: Mapped[str] = mapped_column(String(64), nullable=False)  # e.g. "a2+mentions.v1"
    llm_ok: Mapped[bool] = mapped_column(Boolean, nullable=False)  # False: Stage A only
    stage_a: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    stage_b: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    disagreements: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class Mention(Base):
    """A company named in an answer, after merging both stages. One row per (answer,
    canonical entity); metrics count these."""

    __tablename__ = "mentions"
    __table_args__ = (UniqueConstraint("answer_id", "entity_key", name="uq_mentions_answer_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    answer_id: Mapped[int] = mapped_column(
        ForeignKey("answers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    entity: Mapped[str] = mapped_column(Text, nullable=False)  # surface form, first occurrence
    canonical: Mapped[str] = mapped_column(String(255), nullable=False)
    # normalize(canonical): groups unknown entities across answers ("Wrike" / "Wrike, Inc.").
    entity_key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(16), nullable=False)  # brand|competitor|unknown
    is_brand: Mapped[bool] = mapped_column(Boolean, nullable=False)
    is_competitor: Mapped[bool] = mapped_column(Boolean, nullable=False)
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sentiment: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    sentiment_label: Mapped[str] = mapped_column(String(16), nullable=False, default="neutral")
    is_recommended: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    snippet: Mapped[str] = mapped_column(Text, nullable=False, default="")
    source: Mapped[str] = mapped_column(String(8), nullable=False)  # a | b | both
