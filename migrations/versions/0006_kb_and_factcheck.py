"""kb + factcheck: kb_docs, kb_chunks, answer_factchecks, claims

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-24
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "kb_docs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("brand_id", sa.Integer(), nullable=False),
        sa.Column("slug", sa.String(length=200), nullable=False),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("category", sa.String(length=16), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("as_of", sa.String(length=32), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["brand_id"], ["brands.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("brand_id", "slug", name="uq_kb_docs_brand_slug"),
    )
    op.create_index("ix_kb_docs_brand_id", "kb_docs", ["brand_id"])
    op.create_table(
        "kb_chunks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("doc_id", sa.Integer(), nullable=False),
        sa.Column("brand_id", sa.Integer(), nullable=False),
        sa.Column("chunk_key", sa.String(length=220), nullable=False),
        sa.Column("ord", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("tokens", sa.Integer(), nullable=False),
        sa.Column("embedding", sa.JSON(), nullable=True),
        sa.Column("embedding_model", sa.String(length=128), nullable=True),
        sa.ForeignKeyConstraint(["doc_id"], ["kb_docs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["brand_id"], ["brands.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("brand_id", "chunk_key", name="uq_kb_chunks_brand_key"),
    )
    op.create_index("ix_kb_chunks_doc_id", "kb_chunks", ["doc_id"])
    op.create_index("ix_kb_chunks_brand_id", "kb_chunks", ["brand_id"])
    op.create_table(
        "answer_factchecks",
        sa.Column("answer_id", sa.Integer(), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("ok", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["answer_id"], ["answers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("answer_id"),
    )
    op.create_table(
        "claims",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("answer_id", sa.Integer(), nullable=False),
        sa.Column("brand_id", sa.Integer(), nullable=False),
        sa.Column("ord", sa.Integer(), nullable=False),
        sa.Column("claim_text", sa.Text(), nullable=False),
        sa.Column("claim_type", sa.String(length=16), nullable=False),
        sa.Column("answer_span", sa.Text(), nullable=False),
        sa.Column("verdict", sa.String(length=16), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column("evidence_chunk_keys", sa.JSON(), nullable=False),
        sa.Column("evidence_quote", sa.Text(), nullable=True),
        sa.Column("source_citation_id", sa.Integer(), nullable=True),
        sa.Column("source_confidence", sa.Float(), nullable=True),
        sa.Column("source_reason", sa.String(length=32), nullable=True),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["answer_id"], ["answers.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["brand_id"], ["brands.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_citation_id"], ["citations.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_claims_answer_id", "claims", ["answer_id"])
    op.create_index("ix_claims_brand_id", "claims", ["brand_id"])


def downgrade() -> None:
    op.drop_index("ix_claims_brand_id", table_name="claims")
    op.drop_index("ix_claims_answer_id", table_name="claims")
    op.drop_table("claims")
    op.drop_table("answer_factchecks")
    op.drop_index("ix_kb_chunks_brand_id", table_name="kb_chunks")
    op.drop_index("ix_kb_chunks_doc_id", table_name="kb_chunks")
    op.drop_table("kb_chunks")
    op.drop_index("ix_kb_docs_brand_id", table_name="kb_docs")
    op.drop_table("kb_docs")
