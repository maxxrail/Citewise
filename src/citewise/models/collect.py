"""Scan tables: a run sends a fixed set of prompts to several engines; answers and their
citations are stored verbatim for Module 3 to analyse."""

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


class Run(Base):
    """One scan. `prompt_ids` and `engines` are frozen at creation so `scan resume` finishes
    exactly the work that was planned, even if the panel changes in between."""

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    brand_id: Mapped[int] = mapped_column(
        ForeignKey("brands.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # pending -> running -> completed | partial | failed | stopped_budget | interrupted
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    # Lease: a process executing the run refreshes this. Another process may only take the
    # run over once it is older than the lease (or with --force), so two workers can't
    # double-spend the same run.
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    engines: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    models: Mapped[dict[str, str]] = mapped_column(JSON, nullable=False)  # engine -> model ID
    prompt_ids: Mapped[list[int]] = mapped_column(JSON, nullable=False)
    # Optional persona / locale system message, e.g. "The user is in Toronto, Canada."
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    max_cost_usd: Mapped[float] = mapped_column(Float, nullable=False)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)  # actual spend
    total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)  # prompts x engines
    done: Mapped[int] = mapped_column(Integer, nullable=False, default=0)  # answers stored ok
    failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    mode: Mapped[str] = mapped_column(String(8), nullable=False)  # "mock" | "live"
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    @property
    def progress(self) -> float:
        return (self.done + self.failed) / self.total if self.total else 0.0


class Answer(Base):
    """One engine's answer to one prompt in one run. The unique key is the idempotency key."""

    __tablename__ = "answers"
    __table_args__ = (
        UniqueConstraint("run_id", "prompt_id", "engine", name="uq_answers_run_prompt_engine"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # SET NULL, not CASCADE: answers are the historical record (trends), so rebuilding a
    # panel must not delete them. `prompt_text` keeps what was actually asked.
    prompt_id: Mapped[int | None] = mapped_column(
        ForeignKey("prompts.id", ondelete="SET NULL"), nullable=True, index=True
    )
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    engine: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    ok: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    raw_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_in: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    searches: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # sha256 of (engine, model, system prompt, prompt text, UTC date): same-day re-asks are free.
    cache_key: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    cached_from_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class Citation(Base):
    __tablename__ = "citations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    answer_id: Mapped[int] = mapped_column(
        ForeignKey("answers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)  # 1-based, answer order
    url: Mapped[str] = mapped_column(Text, nullable=False)  # as the engine returned it
    normalized_url: Mapped[str] = mapped_column(Text, nullable=False)
    domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)


class Schedule(Base):
    """A recurring scan, run by `citewise worker` (APScheduler cron trigger)."""

    __tablename__ = "schedules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    brand_id: Mapped[int] = mapped_column(
        ForeignKey("brands.id", ondelete="CASCADE"), nullable=False, index=True
    )
    cron: Mapped[str] = mapped_column(String(100), nullable=False)  # 5-field crontab
    engines: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    size: Mapped[int] = mapped_column(Integer, nullable=False)
    max_cost_usd: Mapped[float] = mapped_column(Float, nullable=False)
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_run_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
