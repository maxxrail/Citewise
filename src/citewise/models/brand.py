from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from citewise.core.db import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class Brand(Base):
    """A brand we track. Aliases/competitors are JSON lists so the same model works on SQLite."""

    __tablename__ = "brands"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    domain: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(200), nullable=False)
    aliases: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    competitors: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    # Buyer personas used to generate the prompt panel (Module 1), e.g. "trail runner".
    personas: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
