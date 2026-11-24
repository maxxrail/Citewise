"""scans: runs, answers, citations, schedules, llm_calls.run_id

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("brand_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("engines", sa.JSON(), nullable=False),
        sa.Column("models", sa.JSON(), nullable=False),
        sa.Column("prompt_ids", sa.JSON(), nullable=False),
        sa.Column("system_prompt", sa.Text(), nullable=True),
        sa.Column("max_cost_usd", sa.Float(), nullable=False),
        sa.Column("estimated_cost_usd", sa.Float(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False),
        sa.Column("total", sa.Integer(), nullable=False),
        sa.Column("done", sa.Integer(), nullable=False),
        sa.Column("failed", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("mode", sa.String(length=8), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["brand_id"], ["brands.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_runs_brand_id", "runs", ["brand_id"])

    op.create_table(
        "answers",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("prompt_id", sa.Integer(), nullable=True),
        sa.Column("prompt_text", sa.Text(), nullable=False),
        sa.Column("engine", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=128), nullable=False),
        sa.Column("ok", sa.Boolean(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("raw_json", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("tokens_in", sa.Integer(), nullable=False),
        sa.Column("tokens_out", sa.Integer(), nullable=False),
        sa.Column("searches", sa.Integer(), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False),
        sa.Column("cache_key", sa.String(length=64), nullable=False),
        sa.Column("cached_from_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["prompt_id"], ["prompts.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "prompt_id", "engine", name="uq_answers_run_prompt_engine"),
    )
    op.create_index("ix_answers_run_id", "answers", ["run_id"])
    op.create_index("ix_answers_prompt_id", "answers", ["prompt_id"])
    op.create_index("ix_answers_cache_key", "answers", ["cache_key"])

    op.create_table(
        "citations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("answer_id", sa.Integer(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("normalized_url", sa.Text(), nullable=False),
        sa.Column("domain", sa.String(length=255), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["answer_id"], ["answers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_citations_answer_id", "citations", ["answer_id"])
    op.create_index("ix_citations_domain", "citations", ["domain"])

    op.create_table(
        "schedules",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("brand_id", sa.Integer(), nullable=False),
        sa.Column("cron", sa.String(length=100), nullable=False),
        sa.Column("engines", sa.JSON(), nullable=False),
        sa.Column("size", sa.Integer(), nullable=False),
        sa.Column("max_cost_usd", sa.Float(), nullable=False),
        sa.Column("system_prompt", sa.Text(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("last_run_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["brand_id"], ["brands.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_schedules_brand_id", "schedules", ["brand_id"])

    with op.batch_alter_table("llm_calls") as batch:
        batch.add_column(sa.Column("run_id", sa.Integer(), nullable=True))
        batch.create_index("ix_llm_calls_run_id", ["run_id"])


def downgrade() -> None:
    with op.batch_alter_table("llm_calls") as batch:
        batch.drop_index("ix_llm_calls_run_id")
        batch.drop_column("run_id")
    op.drop_table("schedules")
    op.drop_table("citations")
    op.drop_table("answers")
    op.drop_table("runs")
