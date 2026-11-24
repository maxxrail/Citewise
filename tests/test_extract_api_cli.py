"""Module 3 CLI (`extract run`, `metrics show|competitors`, `eval extract`) and API."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session, sessionmaker
from typer.testing import CliRunner

from citewise.api.main import app
from citewise.core.config import Settings
from citewise.core.db import get_db
from citewise.providers import get_chat_provider
from citewise.services.extract.runner import extract_run


@pytest.fixture
def cli_scanned_db(cli_db: str, scanned_snapshot: sqlite3.Connection) -> str:
    """The CLI's file DB with both brands' panels and one mock scan each (runs 1 and 2)."""
    dst = sqlite3.connect(Path(cli_db.removeprefix("sqlite:///")))
    scanned_snapshot.backup(dst)
    dst.close()
    return cli_db


def test_cli_extract_then_metrics_and_competitors(cli_scanned_db: str) -> None:
    from citewise.cli import app as cli

    runner = CliRunner()
    out = runner.invoke(cli, ["extract", "run", "1"])
    assert out.exit_code == 0, out.output
    assert "200 extracted, 0 already done" in out.output
    again = runner.invoke(cli, ["extract", "run", "1"])
    assert "0 extracted, 200 already done" in again.output  # idempotent

    shown = runner.invoke(cli, ["metrics", "show", "Stridelab"])
    assert shown.exit_code == 0, shown.output
    for text in ("mention% (95% CI)", "SoV%", "score", "openai", "anthropic", "gemini",
                 "perplexity", "ALL", "citations:"):  # fmt: skip
        assert text in shown.output, text

    data = json.loads(runner.invoke(cli, ["metrics", "show", "Stridelab", "--json"]).output)
    engines = {s["slice"]: s for s in data["slices"]}
    assert set(engines) == {"openai", "anthropic", "gemini", "perplexity"}
    for s in engines.values():
        lo, hi = s["mention_rate_ci"]
        assert s["answers"] == 50 and lo <= s["mention_rate"] <= hi
        assert 0 <= s["visibility_score"] <= 100
    # The simulator's per-engine bias must survive extraction: engines differ.
    assert len({round(s["mention_rate"], 2) for s in engines.values()}) > 1

    comp = runner.invoke(cli, ["metrics", "competitors", "Stridelab"])
    assert "EMERGING" in comp.output and "Saucony" in comp.output

    by_intent = runner.invoke(cli, ["metrics", "show", "Stridelab", "--by", "intent"])
    assert "commercial" in by_intent.output
    assert runner.invoke(cli, ["metrics", "show", "Taskflow"]).exit_code == 1  # not extracted
    assert runner.invoke(cli, ["metrics", "show", "Stridelab", "--by", "colour"]).exit_code == 1
    assert runner.invoke(cli, ["extract", "run", "99"]).exit_code == 1


def test_cli_eval_extract_meets_the_f1_bar_and_breaks_errors_down(
    cli_db: str, tmp_path: Path
) -> None:
    from citewise.cli import app as cli

    out_path = tmp_path / "extract_eval.md"
    out = CliRunner().invoke(cli, ["eval", "extract", "--out", str(out_path)])
    assert out.exit_code == 0, out.output
    report = json.loads(out_path.with_suffix(".json").read_text(encoding="utf-8"))
    merged = report["stages"]["merged"]
    assert merged["f1"] >= 0.85, merged  # acceptance criterion (mock mode)
    assert report["stages"]["A"]["sentiment_mae"] is None  # Stage A has no sentiment
    assert merged["sentiment_mae"] is not None and merged["rank_accuracy"] is not None
    md = out_path.read_text(encoding="utf-8")
    assert "## Errors by type" in md and "false_negative" in md and "## F1 by tag" in md
    assert "Mock mode" in md
    assert "merged" in out.output and "F1" in out.output


@pytest.fixture
def scanned_client(scanned_factory: sessionmaker[Session]) -> Iterator[TestClient]:
    def _get_db() -> Iterator[Session]:
        with scanned_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_api_metrics_and_competitors(
    scanned_client: TestClient, scanned_factory: sessionmaker[Session], settings: Settings
) -> None:
    assert scanned_client.get("/brands/1/metrics").status_code == 409  # nothing extracted
    extract_run(scanned_factory, 1, get_chat_provider(settings=settings,
                session_factory=scanned_factory), settings)  # fmt: skip

    body = scanned_client.get("/brands/1/metrics").json()
    [overall] = body["slices"]
    assert overall["slice"] == "all" and overall["answers"] == 200
    assert body["citations"]["total"] > 200

    by_engine = scanned_client.get("/brands/1/metrics", params={"by": "engine"}).json()
    assert {s["slice"] for s in by_engine["slices"]} == {"openai", "anthropic", "gemini",
                                                        "perplexity"}  # fmt: skip
    one = scanned_client.get("/brands/1/metrics", params={"engine": "gemini"}).json()
    assert one["slices"][0]["answers"] == 50
    topic_id = (
        by_engine["slices"]
        and scanned_client.get("/brands/1/metrics", params={"by": "topic"}).json()["slices"][0][
            "slice"
        ]
    )
    sliced = scanned_client.get("/brands/1/metrics", params={"topic": int(topic_id)}).json()
    assert 0 < sliced["slices"][0]["answers"] < 200

    comps = scanned_client.get("/brands/1/competitors").json()
    assert any(c["emerging"] and c["canonical"] in ("Saucony", "Altra") for c in comps)
    assert all(c["entity_type"] in ("competitor", "unknown") for c in comps)

    assert scanned_client.get("/brands/999/metrics").status_code == 404
    assert scanned_client.get("/brands/1/metrics", params={"by": "colour"}).status_code == 422
