"""prompts.embedding_model: which vector space each stored embedding belongs to

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-23

Existing rows get NULL ("unknown"), so the next generate/cluster re-embeds them with the
current embedder rather than guessing their provenance.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("prompts") as batch:
        batch.add_column(sa.Column("embedding_model", sa.String(length=128), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("prompts") as batch:
        batch.drop_column("embedding_model")
