"""content agent: content_opportunities, content_items, content_versions, content_reviews

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "content_opportunities",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("brand_id", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(length=200), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("impact", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("suggested_format", sa.String(length=32), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=True),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["brand_id"], ["brands.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("brand_id", "key", name="uq_content_opp_brand_key"),
    )
    op.create_index("ix_content_opportunities_brand_id", "content_opportunities", ["brand_id"])
    op.create_index("ix_content_opportunities_score", "content_opportunities", ["score"])
    op.create_table(
        "content_items",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("brand_id", sa.Integer(), nullable=False),
        sa.Column("opportunity_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=24), nullable=False),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("format", sa.String(length=32), nullable=False),
        sa.Column("brief", sa.JSON(), nullable=True),
        sa.Column("outline", sa.JSON(), nullable=True),
        sa.Column("loops", sa.Integer(), nullable=False),
        sa.Column("current_version", sa.Integer(), nullable=False),
        sa.Column("pending_feedback", sa.JSON(), nullable=False),
        sa.Column("measurement", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["brand_id"], ["brands.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["opportunity_id"], ["content_opportunities.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_content_items_brand_id", "content_items", ["brand_id"])
    op.create_index("ix_content_items_opportunity_id", "content_items", ["opportunity_id"])
    op.create_index("ix_content_items_status", "content_items", ["status"])
    op.create_table(
        "content_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("item_id", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("markdown", sa.Text(), nullable=False),
        sa.Column("draft", sa.JSON(), nullable=False),
        sa.Column("scores", sa.JSON(), nullable=False),
        sa.Column("checks", sa.JSON(), nullable=False),
        sa.Column("feedback", sa.JSON(), nullable=False),
        sa.Column("jsonld", sa.JSON(), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["item_id"], ["content_items.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("item_id", "version", name="uq_content_version"),
    )
    op.create_index("ix_content_versions_item_id", "content_versions", ["item_id"])
    op.create_table(
        "content_reviews",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("item_id", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(length=16), nullable=False),
        sa.Column("comment", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["item_id"], ["content_items.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_content_reviews_item_id", "content_reviews", ["item_id"])


def downgrade() -> None:
    op.drop_index("ix_content_reviews_item_id", table_name="content_reviews")
    op.drop_table("content_reviews")
    op.drop_index("ix_content_versions_item_id", table_name="content_versions")
    op.drop_table("content_versions")
    op.drop_index("ix_content_items_status", table_name="content_items")
    op.drop_index("ix_content_items_opportunity_id", table_name="content_items")
    op.drop_index("ix_content_items_brand_id", table_name="content_items")
    op.drop_table("content_items")
    op.drop_index("ix_content_opportunities_score", table_name="content_opportunities")
    op.drop_index("ix_content_opportunities_brand_id", table_name="content_opportunities")
    op.drop_table("content_opportunities")
