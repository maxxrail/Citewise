"""panel: brands.personas, topics, prompts

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-23
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("brands") as batch:
        batch.add_column(
            sa.Column("personas", sa.JSON(), nullable=False, server_default=sa.text("'[]'"))
        )

    op.create_table(
        "topics",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("brand_id", sa.Integer(), nullable=False),
        sa.Column("label", sa.String(length=100), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("size", sa.Integer(), nullable=False),
        sa.Column("algorithm", sa.String(length=32), nullable=False),
        sa.Column("label_source", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["brand_id"], ["brands.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_topics_brand_id", "topics", ["brand_id"])

    op.create_table(
        "prompts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("brand_id", sa.Integer(), nullable=False),
        sa.Column("text", sa.String(length=500), nullable=False),
        sa.Column("text_hash", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("template_version", sa.String(length=64), nullable=True),
        sa.Column("persona", sa.String(length=100), nullable=True),
        sa.Column("funnel_stage", sa.String(length=16), nullable=True),
        sa.Column("requested_intent", sa.String(length=16), nullable=True),
        sa.Column("intent", sa.String(length=16), nullable=True),
        sa.Column("intent_source", sa.String(length=16), nullable=True),
        sa.Column("topic_id", sa.Integer(), nullable=True),
        sa.Column("centrality", sa.Float(), nullable=True),
        sa.Column("embedding", sa.JSON(), nullable=True),
        sa.Column("in_panel", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["brand_id"], ["brands.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["topic_id"], ["topics.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("brand_id", "text_hash", name="uq_prompts_brand_text"),
    )
    op.create_index("ix_prompts_brand_id", "prompts", ["brand_id"])
    op.create_index("ix_prompts_intent", "prompts", ["intent"])
    op.create_index("ix_prompts_topic_id", "prompts", ["topic_id"])
    op.create_index("ix_prompts_in_panel", "prompts", ["in_panel"])


def downgrade() -> None:
    op.drop_table("prompts")
    op.drop_table("topics")
    with op.batch_alter_table("brands") as batch:
        batch.drop_column("personas")
