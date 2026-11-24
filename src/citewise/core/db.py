"""Database engine and session helpers (SQLAlchemy 2.0)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from citewise.core.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for all models."""


@lru_cache
def get_engine() -> Engine:
    return create_engine(get_settings().database_url, pool_pre_ping=True)


@lru_cache
def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for scripts/CLI: commit on success, rollback on error."""
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a session per request."""
    session = get_sessionmaker()()
    try:
        yield session
    finally:
        session.close()


def ping(session: Session) -> bool:
    """Return True if the database answers `SELECT 1`."""
    try:
        session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def get_session_factory() -> sessionmaker[Session]:
    """FastAPI dependency for work that outlives the request (background scan runs), which
    needs its own sessions. Tests override it the same way they override `get_db`."""
    return get_sessionmaker()
