"""Module 2 interfaces: API endpoints, `citewise scan/schedule` CLI, and the scheduler."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker
from typer.testing import CliRunner

from citewise.api.main import app
from citewise.core.config import Settings
from citewise.core.db import get_db, get_session_factory
from citewise.models.collect import Answer, Run, Schedule
from citewise.services.panel.common import resolve_brand
from citewise.workers.scheduler import (
    SYNC_JOB_ID,
    add_schedule,
    build_scheduler,
    parse_cron,
    run_schedule,
    sync_jobs,
)


@pytest.fixture
def scan_client(scan_factory: sessionmaker[Session]) -> Iterator[TestClient]:
    def _get_db() -> Iterator[Session]:
        with scan_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[get_session_factory] = lambda: scan_factory
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _brand_id(factory: sessionmaker[Session], name: str = "Stridelab") -> int:
    with factory() as s:
        return resolve_brand(s, name).id


# ---------------------------------------------------------------- API


def test_api_start_run_then_poll_it(
    scan_client: TestClient, scan_factory: sessionmaker[Session]
) -> None:
    bid = _brand_id(scan_factory)
    resp = scan_client.post(
        f"/brands/{bid}/runs",
        json={
            "engines": ["openai", "perplexity"],
            "size": 10,
            "max_cost": 1.0,
            "system_prompt": "The user is in Toronto, Canada.",
        },
    )
    assert resp.status_code == 202, resp.text
    body = resp.json()
    assert body["status"] == "pending" and body["total"] == 20
    assert body["estimated_cost_usd"] > 0 and body["system_prompt"].startswith("The user")

    # TestClient runs background tasks before returning, so the run has finished here.
    run = scan_client.get(f"/runs/{body['id']}").json()
    assert (run["status"], run["done"], run["progress"]) == ("completed", 20, 1.0)
    assert run["cost_usd"] > 0
    listed = scan_client.get(f"/brands/{bid}/runs").json()
    assert [r["id"] for r in listed] == [body["id"]]


def test_api_budget_and_validation_errors(
    scan_client: TestClient, scan_factory: sessionmaker[Session]
) -> None:
    bid = _brand_id(scan_factory)
    over = scan_client.post(f"/brands/{bid}/runs", json={"size": 50, "max_cost": 0.01})
    assert over.status_code == 422
    assert over.json()["detail"]["estimate_usd"] > 0.01
    with scan_factory() as s:
        assert s.scalar(select(func.count()).select_from(Run)) == 0
    assert scan_client.post(f"/brands/{bid}/runs", json={"engines": ["bing"]}).status_code == 422
    bad_system = scan_client.post(f"/brands/{bid}/runs", json={"system_prompt": "x === user ==="})
    assert bad_system.status_code == 422
    assert scan_client.post("/brands/999/runs", json={}).status_code == 404
    assert scan_client.get("/runs/999").status_code == 404
    assert scan_client.get("/brands/999/runs").status_code == 404


# ---------------------------------------------------------------- CLI


@pytest.fixture
def cli_panel_db(cli_db: str, panel_snapshot: sqlite3.Connection) -> str:
    """The CLI's file DB, pre-filled with the clustered example panels."""
    path = cli_db.removeprefix("sqlite:///")
    dst = sqlite3.connect(Path(path))
    panel_snapshot.backup(dst)
    dst.close()
    return cli_db


def test_cli_scan_run_status_list_resume(cli_panel_db: str) -> None:
    from citewise.cli import app as cli

    runner = CliRunner()
    out = runner.invoke(
        cli,
        [
            "scan",
            "run",
            "Stridelab",
            "--engines",
            "openai,gemini",
            "--size",
            "10",
            "--max-cost",
            "1",
        ],
    )
    assert out.exit_code == 0, out.output
    assert "10 prompts x 2 engines = 20 answers" in out.output
    assert "completed | 20/20 answers" in out.output

    assert "completed" in runner.invoke(cli, ["scan", "status", "1"]).output
    listing = runner.invoke(cli, ["scan", "list", "Stridelab"]).output
    assert "openai,gemini" in listing and "20/20" in listing
    assert "already completed" in runner.invoke(cli, ["scan", "resume", "1"]).output

    refused = runner.invoke(cli, ["scan", "run", "Stridelab", "--size", "50", "--max-cost", "0.01"])
    assert refused.exit_code == 1 and "Not started" in refused.output
    assert runner.invoke(cli, ["scan", "run", "Stridelab", "--engines", "bing"]).exit_code == 1
    assert runner.invoke(cli, ["scan", "status", "99"]).exit_code == 1


def test_cli_schedule_add_list_remove(cli_panel_db: str) -> None:
    from citewise.cli import app as cli

    runner = CliRunner()
    added = runner.invoke(
        cli, ["schedule", "add", "Taskflow", "--cron", "0 9 * * 1", "--size", "20"]
    )
    assert added.exit_code == 0, added.output
    assert "schedule 1: Taskflow '0 9 * * 1' (UTC); next run" in added.output
    assert "'0 9 * * 1'" in runner.invoke(cli, ["schedule", "list"]).output
    bad = runner.invoke(cli, ["schedule", "add", "Taskflow", "--cron", "every monday"])
    assert bad.exit_code == 1
    assert "removed" in runner.invoke(cli, ["schedule", "remove", "1"]).output
    assert "no schedules" in runner.invoke(cli, ["schedule", "list"]).output


# ---------------------------------------------------------------- scheduler


def test_parse_cron_validates() -> None:
    assert parse_cron("0 9 * * 1") is not None
    for bad in ("0 9 * *", "61 9 * * 1", "every monday"):
        with pytest.raises(ValueError):
            parse_cron(bad)


def test_scheduler_syncs_jobs_with_the_table_and_runs_a_schedule(
    scan_factory: sessionmaker[Session], settings: Settings
) -> None:
    with scan_factory() as s:
        sched = add_schedule(
            s,
            resolve_brand(s, "Stridelab"),
            cron="0 9 * * 1",
            engines=["openai"],
            size=5,
            max_cost=1.0,
        )
        s.commit()
        sid = sched.id

    scheduler = build_scheduler(settings, scan_factory, BackgroundScheduler(timezone="UTC"))
    ids = {j.id for j in scheduler.get_jobs()}
    assert ids == {SYNC_JOB_ID, f"schedule-{sid}"}

    run_id = run_schedule(sid, settings, scan_factory)
    assert run_id is not None
    with scan_factory() as s:
        assert s.get(Run, run_id).status == "completed"  # type: ignore[union-attr]
        assert s.get(Schedule, sid).last_run_id == run_id  # type: ignore[union-attr]
        assert s.scalar(select(func.count()).where(Answer.run_id == run_id)) == 5
        s.get(Schedule, sid).enabled = False  # type: ignore[union-attr]
        s.commit()

    assert sync_jobs(scheduler, settings, scan_factory) == 0
    assert {j.id for j in scheduler.get_jobs()} == {SYNC_JOB_ID}
    assert run_schedule(sid, settings, scan_factory) is None  # disabled schedules don't run


def test_scheduled_run_over_budget_is_skipped_not_crashed(
    scan_factory: sessionmaker[Session], settings: Settings
) -> None:
    with scan_factory() as s:
        sched = add_schedule(
            s,
            resolve_brand(s, "Stridelab"),
            cron="0 9 * * 1",
            engines=["openai", "gemini"],
            size=50,
            max_cost=0.01,
        )
        s.commit()
    assert run_schedule(sched.id, settings, scan_factory) is None
    with scan_factory() as s:
        assert s.scalar(select(func.count()).select_from(Run)) == 0
