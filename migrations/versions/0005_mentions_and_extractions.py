"""extraction: mentions, answer_extractions

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "answer_extractions",
        sa.Column("answer_id", sa.Integer(), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("llm_ok", sa.Boolean(), nullable=False),
        sa.Column("stage_a", sa.JSON(), nullable=False),
        sa.Column("stage_b", sa.JSON(), nullable=True),
        sa.Column("disagreements", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["answer_id"], ["answers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("answer_id"),
    )
    op.create_table(
        "mentions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("answer_id", sa.Integer(), nullable=False),
        sa.Column("entity", sa.Text(), nullable=False),
        sa.Column("canonical", sa.String(length=255), nullable=False),
        sa.Column("entity_key", sa.String(length=255), nullable=False),
        sa.Column("entity_type", sa.String(length=16), nullable=False),
        sa.Column("is_brand", sa.Boolean(), nullable=False),
        sa.Column("is_competitor", sa.Boolean(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=True),
        sa.Column("sentiment", sa.Float(), nullable=False),
        sa.Column("sentiment_label", sa.String(length=16), nullable=False),
        sa.Column("is_recommended", sa.Boolean(), nullable=False),
        sa.Column("snippet", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=8), nullable=False),
        sa.ForeignKeyConstraint(["answer_id"], ["answers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("answer_id", "entity_key", name="uq_mentions_answer_key"),
    )
    op.create_index("ix_mentions_answer_id", "mentions", ["answer_id"])
    op.create_index("ix_mentions_entity_key", "mentions", ["entity_key"])


def downgrade() -> None:
    op.drop_index("ix_mentions_entity_key", table_name="mentions")
    op.drop_index("ix_mentions_answer_id", table_name="mentions")
    op.drop_table("mentions")
    op.drop_table("answer_extractions")
