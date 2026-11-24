"""Shared fixtures: mock mode, in-memory SQLite, FastAPI client. No network, no .env needed."""

from __future__ import annotations

import os

# Set before any citewise import reads settings. Env vars beat a developer's local .env.
os.environ["CITEWISE_MODE"] = "mock"
os.environ["DATABASE_URL"] = "sqlite://"

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import citewise.models  # noqa: F401
from citewise.api.main import app
from citewise.core.config import Settings, get_settings
from citewise.core.db import Base, get_db


@pytest.fixture
def settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def engine() -> Iterator[Engine]:
    # StaticPool keeps one connection so the in-memory DB is shared across threads/sessions.
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def db(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    with session_factory() as session:
        yield session


@pytest.fixture
def cli_db(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:  # type: ignore[no-untyped-def]
    """A file SQLite DB for CLI tests (the CLI opens its own sessions via settings)."""
    from citewise.core import db as core_db

    url = f"sqlite:///{(tmp_path / 'cli.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    for cached in (get_settings, core_db.get_engine, core_db.get_sessionmaker):
        cached.cache_clear()
    Base.metadata.create_all(core_db.get_engine())
    yield url
    core_db.get_engine().dispose()
    for cached in (get_settings, core_db.get_engine, core_db.get_sessionmaker):
        cached.cache_clear()


@pytest.fixture
def client(session_factory: sessionmaker[Session]) -> Iterator[TestClient]:
    def _get_db() -> Iterator[Session]:
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
