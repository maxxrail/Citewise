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
    from citewise.services.panel.common import resolve_brand

    with scanned_factory() as s:
        bid = resolve_brand(s, "Stridelab").id  # ids follow seed order; don't hardcode
    base = f"/brands/{bid}"
    assert scanned_client.get(f"{base}/metrics").status_code == 409  # nothing extracted
    extract_run(scanned_factory, 1, get_chat_provider(settings=settings,
                session_factory=scanned_factory), settings)  # fmt: skip

    body = scanned_client.get(f"{base}/metrics").json()
    [overall] = body["slices"]
    assert overall["slice"] == "all" and overall["answers"] == 200
    assert body["citations"]["total"] > 200

    by_engine = scanned_client.get(f"{base}/metrics", params={"by": "engine"}).json()
    assert {s["slice"] for s in by_engine["slices"]} == {"openai", "anthropic", "gemini",
                                                        "perplexity"}  # fmt: skip
    one = scanned_client.get(f"{base}/metrics", params={"engine": "gemini"}).json()
    assert one["slices"][0]["answers"] == 50
    topic_id = (
        by_engine["slices"]
        and scanned_client.get(f"{base}/metrics", params={"by": "topic"}).json()["slices"][0][
            "slice"
        ]
    )
    sliced = scanned_client.get(f"{base}/metrics", params={"topic": int(topic_id)}).json()
    assert 0 < sliced["slices"][0]["answers"] < 200

    comps = scanned_client.get(f"{base}/competitors").json()
    assert any(c["emerging"] and c["canonical"] in ("Saucony", "Altra") for c in comps)
    assert all(c["entity_type"] in ("competitor", "unknown") for c in comps)

    assert scanned_client.get("/brands/999/metrics").status_code == 404
    assert scanned_client.get(f"{base}/metrics", params={"by": "colour"}).status_code == 422


def test_eval_extract_gates_on_min_f1(cli_db: str, tmp_path: Path) -> None:
    from citewise.cli import app as cli
    from citewise.core.config import REPO_ROOT

    # One golden row with an extra label no extractor can find: F1 drops below 1.0.
    row = json.loads(
        (REPO_ROOT / "evals/datasets/golden/mentions.jsonl").read_text("utf-8").splitlines()[0]
    )
    row["mentions"].append({"canonical": "Nonexistent Co", "entity_type": "unknown", "rank": None,
                            "sentiment": 0.0, "sentiment_label": "neutral",
                            "is_recommended": False})  # fmt: skip
    golden = tmp_path / "g.jsonl"
    golden.write_text(json.dumps(row) + "\n", encoding="utf-8")
    runner = CliRunner()
    args = ["eval", "extract", "--golden", str(golden), "--out", str(tmp_path / "r.md")]
    out = runner.invoke(cli, [*args, "--min-f1", "0.99"])
    assert out.exit_code == 1 and "below --min-f1" in out.output
    assert "Stage B:" in out.output
    assert runner.invoke(cli, [*args, "--min-f1", "0.5"]).exit_code == 0


def test_recorded_stage_b_responses_replay_in_mock_mode(settings: Settings, tmp_path: Path) -> None:
    """scripts/record_extract_fixtures.py writes real responses under the prompt's fixture
    key; the mock provider must then replay them instead of simulating."""
    import shutil

    from citewise.core.config import REPO_ROOT
    from citewise.evals.extract import evaluate, load_brand_configs, load_golden
    from citewise.providers.mock import MockChatProvider, fixture_key
    from citewise.services.extract.mentions import PURPOSE, stage_b_messages

    fixtures = tmp_path / "fixtures"
    shutil.copytree(settings.fixtures_dir / "engines", fixtures / "engines")
    (fixtures / "mock_chat").mkdir()
    rows = load_golden(REPO_ROOT / "evals/datasets/golden/mentions.jsonl")[:1]
    brands = load_brand_configs(settings.brands_dir)
    cfg = brands[rows[0]["brand"]]
    msgs = stage_b_messages(brand=cfg["name"], aliases=cfg["aliases"],
                            competitors=cfg["competitors"], category=cfg["category"],
                            text=rows[0]["answer"])  # fmt: skip
    recorded = {"mentions": [{"entity": "Nike", "canonical": "Nike", "rank": 1, "sentiment": -0.9,
                              "sentiment_label": "negative", "is_recommended": False,
                              "snippet": "x"}]}  # fmt: skip
    (fixtures / "mock_chat" / f"{fixture_key(PURPOSE, msgs)}.json").write_text(
        json.dumps({"text": json.dumps(recorded), "parsed": recorded}), encoding="utf-8"
    )
    report = evaluate(rows, brands, MockChatProvider(fixtures), model="m", mode="mock",
                      fixtures_dir=fixtures)  # fmt: skip
    assert report.stage_b_source.startswith("1/1 replayed")
    assert report.stages["B"].sentiment_abs_err  # scored on the recorded (negative) answer
    assert report.stages["B"].tp == 1  # only Nike came back


def test_cli_by_run_has_no_misleading_all_row_and_warns_on_partial_runs(
    cli_scanned_db: str,
) -> None:
    from citewise.cli import app as cli
    from citewise.core.config import get_settings
    from citewise.core.db import get_sessionmaker
    from citewise.providers.budget import BudgetedChatProvider
    from citewise.providers.registry import get_chat_provider

    settings = get_settings()
    capped = BudgetedChatProvider(get_chat_provider(settings=settings), 0.003)
    extract_run(get_sessionmaker(), 1, capped, settings)  # budget-stops partway
    runner = CliRunner()
    shown = runner.invoke(cli, ["metrics", "show", "Stridelab"])
    assert shown.exit_code == 0 and "warning" in shown.output and "extracted" in shown.output
    by_run = runner.invoke(cli, ["metrics", "show", "Stridelab", "--by", "run"])
    assert by_run.exit_code == 0 and "ALL" not in by_run.output
