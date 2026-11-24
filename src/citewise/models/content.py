"""Module 5 tables: content opportunities, the content state machine, versions, reviews."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
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

OPPORTUNITY_KINDS = ("gap", "citation", "fact_fix")


class ContentOpportunity(Base):
    """A piece of marketing work worth doing. `key` identifies the same opportunity across
    re-runs of the finder (e.g. "gap:topic=12:intent=commercial"), so refreshing updates
    the row instead of adding a duplicate."""

    __tablename__ = "content_opportunities"
    __table_args__ = (UniqueConstraint("brand_id", "key", name="uq_content_opp_brand_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    brand_id: Mapped[int] = mapped_column(
        ForeignKey("brands.id", ondelete="CASCADE"), nullable=False, index=True
    )
    key: Mapped[str] = mapped_column(String(200), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)  # OPPORTUNITY_KINDS
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    impact: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    suggested_format: Mapped[str] = mapped_column(String(32), nullable=False)
    run_id: Mapped[int | None] = mapped_column(Integer, nullable=True)  # the scan it came from
    details: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="open")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class ContentItem(Base):
    """One piece of content moving through the pipeline. `status` is the state machine's
    current state (services/content/pipeline.py validates every transition)."""

    __tablename__ = "content_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    brand_id: Mapped[int] = mapped_column(
        ForeignKey("brands.id", ondelete="CASCADE"), nullable=False, index=True
    )
    opportunity_id: Mapped[int | None] = mapped_column(
        ForeignKey("content_opportunities.id", ondelete="SET NULL"), nullable=True, index=True
    )
    status: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(300), nullable=False, default="")
    format: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    brief: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    outline: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    loops: Mapped[int] = mapped_column(Integer, nullable=False, default=0)  # revisions done
    current_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pending_feedback: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    # Approval snapshot of the target prompts' mention rate, for before/after (Module 6).
    measurement: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class ContentVersion(Base):
    __tablename__ = "content_versions"
    __table_args__ = (UniqueConstraint("item_id", "version", name="uq_content_version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    item_id: Mapped[int] = mapped_column(
        ForeignKey("content_items.id", ondelete="CASCADE"), nullable=False, index=True
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)  # 1, 2, ...
    kind: Mapped[str] = mapped_column(String(16), nullable=False)  # draft | revision | final
    markdown: Mapped[str] = mapped_column(Text, nullable=False)
    draft: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)  # DraftOut
    scores: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    checks: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    feedback: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
    jsonld: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)  # "why this should help"
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class ContentReview(Base):
    __tablename__ = "content_reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    item_id: Mapped[int] = mapped_column(
        ForeignKey("content_items.id", ondelete="CASCADE"), nullable=False, index=True
    )
    action: Mapped[str] = mapped_column(String(16), nullable=False)  # approve | reject
    comment: Mapped[str] = mapped_column(Text, nullable=False, default="")
    version: Mapped[int] = mapped_column(Integer, nullable=False)  # the version reviewed
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
