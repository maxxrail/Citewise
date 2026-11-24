"""`citewise` command-line entry point."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from citewise.core.config import get_settings
from citewise.core.logging import configure_logging
from citewise.extract_cli import eval_app, extract_app, metrics_app
from citewise.panel_cli import panel_app
from citewise.scan_cli import scan_app, schedule_app, worker

app = typer.Typer(help="Citewise: AI-search visibility platform.", no_args_is_help=True)
app.add_typer(panel_app, name="panel")
app.add_typer(scan_app, name="scan")
app.add_typer(schedule_app, name="schedule")
app.add_typer(extract_app, name="extract")
app.add_typer(metrics_app, name="metrics")
app.add_typer(eval_app, name="eval")
app.command()(worker)


def _utf8_console() -> None:
    """Make stdout/stderr UTF-8 (unencodable chars replaced, never raised).

    On Windows, piped or redirected output defaults to cp1252, so printing a prompt or topic
    label in Japanese or with an emoji raised UnicodeEncodeError after the work was committed.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


@app.callback()
def main() -> None:
    _utf8_console()
    configure_logging()


@app.command()
def seed(
    directory: Path | None = typer.Option(
        None, help="Folder of brand YAMLs (default data/brands)."
    ),
) -> None:
    """Load example brands from YAML into the database (safe to re-run)."""
    from citewise.core.db import session_scope
    from citewise.services.brands import seed_brands

    directory = directory or get_settings().brands_dir
    with session_scope() as session:
        created, updated = seed_brands(session, directory)
    typer.echo(f"Seeded brands from {directory}: {created} created, {updated} updated.")


@app.command()
def config() -> None:
    """Show non-secret configuration (mode, DB host, model IDs, which keys are set)."""
    s = get_settings()
    typer.echo(f"mode: {s.citewise_mode}")
    typer.echo(f"database: {s.database_url.split('@')[-1]}")
    for field in ("model_extract", "model_judge", "model_generate", "model_embed"):
        typer.echo(f"{field}: {getattr(s, field)}")
    for key in ("openai", "anthropic", "gemini", "perplexity"):
        present = getattr(s, f"{key}_api_key") is not None
        typer.echo(f"{key.upper()}_API_KEY: {'set' if present else 'missing'}")


if __name__ == "__main__":
    app()
