from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from citewise.cli import app
from citewise.core.config import REPO_ROOT, Settings
from citewise.services.brands import list_brands, load_brand_files, seed_brands


def test_cli_help_lists_seed() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "seed" in result.output


def test_example_brand_files_are_valid(settings: Settings) -> None:
    brands = load_brand_files(settings.brands_dir)
    assert {b.name for b in brands} == {"Taskflow", "Stridelab"}
    for b in brands:
        assert 4 <= len(b.competitors) <= 6
        assert b.aliases
        assert len(b.personas) == 3


def test_seed_is_idempotent(db: Session, settings: Settings) -> None:
    assert seed_brands(db, settings.brands_dir) == (2, 0)
    db.commit()
    assert seed_brands(db, settings.brands_dir) == (0, 2)
    db.commit()
    assert sorted(b.name for b in list_brands(db)) == ["Stridelab", "Taskflow"]


def test_alembic_upgrade_and_downgrade_on_sqlite(tmp_path: Path) -> None:
    url = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", url)
    command.upgrade(cfg, "head")
    tables = set(inspect(create_engine(url)).get_table_names())
    assert {"brands", "llm_calls", "prompts", "topics"} <= tables
    command.downgrade(cfg, "base")
    assert "brands" not in inspect(create_engine(url)).get_table_names()
