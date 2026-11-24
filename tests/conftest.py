"""Shared fixtures: mock mode, in-memory SQLite, FastAPI client. No network, no .env needed."""

from __future__ import annotations

import os
import sqlite3

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
from citewise.core.db import Base, get_db, get_session_factory


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


@pytest.fixture(scope="session")
def panel_snapshot() -> Iterator[sqlite3.Connection]:
    """Both example brands with a clustered 50-prompt panel, built once per test session."""
    from citewise.providers import get_chat_provider, get_embedder
    from citewise.services.brands import seed_brands
    from citewise.services.panel.cluster import cluster_brand
    from citewise.services.panel.common import resolve_brand
    from citewise.services.panel.generate import generate_panel
    from citewise.services.panel.select import build_panel

    get_settings.cache_clear()
    settings = get_settings()
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(eng)
    factory = sessionmaker(bind=eng, expire_on_commit=False)
    with factory() as session:
        seed_brands(session, settings.brands_dir)
        for name in ("Stridelab", "Taskflow"):
            brand = resolve_brand(session, name)
            chat = get_chat_provider(settings=settings, session_factory=factory)
            embedder = get_embedder(settings=settings, session_factory=factory)
            generate_panel(session, brand, chat, embedder, settings)
            cluster_brand(session, brand, chat, settings, embedder=embedder)
            build_panel(session, brand, 50)
        session.commit()
    raw = eng.raw_connection().driver_connection
    assert isinstance(raw, sqlite3.Connection)
    yield raw
    eng.dispose()


@pytest.fixture
def scan_factory(panel_snapshot: sqlite3.Connection) -> Iterator[sessionmaker[Session]]:
    """A private copy of the panel DB (sqlite backup API), so each test starts clean."""
    clone = sqlite3.connect(":memory:", check_same_thread=False)
    panel_snapshot.backup(clone)
    eng = create_engine("sqlite://", creator=lambda: clone, poolclass=StaticPool)
    yield sessionmaker(bind=eng, expire_on_commit=False)
    eng.dispose()
    clone.close()


@pytest.fixture
def client(session_factory: sessionmaker[Session]) -> Iterator[TestClient]:
    def _get_db() -> Iterator[Session]:
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[get_session_factory] = lambda: session_factory
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
