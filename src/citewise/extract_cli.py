"""`citewise extract ...`, `citewise metrics ...` and `citewise eval extract` (Module 3)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import typer

from citewise.core.config import REPO_ROOT, get_settings
from citewise.panel_cli import MAX_COST_HELP, _budget_note, _budgeted_chat, _fail

extract_app = typer.Typer(help="Extract brand mentions from scan answers.", no_args_is_help=True)
metrics_app = typer.Typer(help="Visibility metrics over extracted answers.", no_args_is_help=True)
eval_app = typer.Typer(help="Evaluate LLM components against golden sets.", no_args_is_help=True)

GOLDEN_MENTIONS = REPO_ROOT / "evals" / "datasets" / "golden" / "mentions.jsonl"
EXTRACT_REPORT = REPO_ROOT / "evals" / "reports" / "extract_eval.md"


@extract_app.command("run")
def extract_run_cmd(
    run_id: int = typer.Argument(..., help="Scan run id (see `citewise scan list`)."),
    force: bool = typer.Option(False, "--force", help="Re-extract answers already done."),
    max_cost: float | None = typer.Option(None, min=0, help=MAX_COST_HELP),
) -> None:
    """Extract mentions from every ok answer in a run (idempotent)."""
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

    from citewise.core.db import get_sessionmaker
    from citewise.services.extract.runner import EXTRACTOR_VERSION, extract_run

    settings = get_settings()
    chat = _budgeted_chat(max_cost)
    with Progress(TextColumn("[bold]extract"), BarColumn(), MofNCompleteColumn(),
                  transient=True) as progress:  # fmt: skip
        task = progress.add_task("extract", total=None)

        def on_progress(i: int) -> None:
            progress.update(task, completed=i)

        try:
            report = extract_run(get_sessionmaker(), run_id, chat, settings, force=force,
                                 on_progress=on_progress)  # fmt: skip
        except ValueError as exc:
            progress.stop()
            _fail(str(exc))
            return
    typer.echo(
        f"run {run_id} [{EXTRACTOR_VERSION}]: {report.extracted} extracted, "
        f"{report.skipped} already done, {report.mentions} mentions"
    )
    if report.llm_failed:
        typer.secho(
            f"  {report.llm_failed} answers kept Stage A only (LLM failed); re-run to retry",
            fg=typer.colors.YELLOW,
        )
    if report.budget_stopped:
        typer.secho(
            f"  {report.budget_stopped} answers left for the next run (budget reached)",
            fg=typer.colors.YELLOW,
        )
    if report.disagreements:
        parts = ", ".join(f"{k} {v}" for k, v in report.disagreements.most_common())
        typer.echo(f"  stage A/B disagreements: {parts}")
    _budget_note(chat)


def _num(v: float | None, fmt: str = ".2f") -> str:
    return "-" if v is None else format(v, fmt)


@metrics_app.command("show")
def metrics_show(
    brand: str = typer.Argument(..., help="Brand name or id."),
    run: int | None = typer.Option(None, "--run", help="Run id (default: latest extracted)."),
    by: str = typer.Option("engine", help="Slice: engine, topic, intent, run, or all."),
    engine: str | None = typer.Option(None, help="Only this engine."),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable output."),
) -> None:
    """Visibility per engine (or topic/intent/run): mention rate with 95% CI, share of
    voice, average rank, sentiment and the 0-100 Visibility Score."""
    from rich.console import Console
    from rich.table import Table

    from citewise.core.db import session_scope
    from citewise.models.panel import Topic
    from citewise.services.metrics.visibility import (
        Filters,
        NoExtractionsError,
        citation_share,
        compute_metrics,
    )
    from citewise.services.panel.common import BrandNotFoundError, resolve_brand

    if by not in ("engine", "topic", "intent", "run", "all"):
        _fail("--by must be one of engine, topic, intent, run, all")
    settings = get_settings()
    try:
        with session_scope() as session:
            b = resolve_brand(session, brand)
            f = Filters(run_id=run, engine=engine)
            slice_by = None if by == "all" else by
            rows = compute_metrics(session, b.id, f, slice_by=slice_by,  # type: ignore[arg-type]
                                   bootstrap=settings.metrics_bootstrap)  # fmt: skip
            overall = compute_metrics(session, b.id, f, bootstrap=settings.metrics_bootstrap)
            cites = citation_share(session, b.id, b.domain, f) if by != "run" else None
            labels = (
                {str(t.id): t.label for t in session.query(Topic).filter_by(brand_id=b.id)}
                if by == "topic"
                else {}
            )
            name = b.name
    except (BrandNotFoundError, NoExtractionsError) as exc:
        _fail(str(exc))
        return
    if as_json:
        typer.echo(json.dumps({"brand": name, "by": by, "slices": [r.as_dict() for r in rows],
                               "overall": [r.as_dict() for r in overall]}, indent=2))  # fmt: skip
        return

    title = f"{name}: visibility by {by}" + (f" (run {run})" if run else " (latest run)")
    # Borderless, units in the headers: nine columns must fit an 80-column terminal.
    table = Table(title=title, box=None, padding=(0, 1), header_style="bold")
    for col in ("slice", "n", "mention% (95% CI)", "SoV%", "rank", "top3%", "rec%",
                "net sent", "score"):  # fmt: skip
        table.add_column(col, justify="left" if col == "slice" else "right", no_wrap=True)

    def p(v: float | None) -> str:
        return "-" if v is None else f"{100 * v:.0f}"

    # With --by run the slices *are* the runs; an ALL row would be the latest run only.
    for r in [*rows, *([] if by in ("all", "run") else overall)]:
        lo, hi = r.mention_rate_ci
        label = "ALL" if r.slice == "all" else labels.get(r.slice, r.slice)
        table.add_row(
            label[:28] + (" *" if r.small_sample else ""),
            str(r.answers),
            f"{p(r.mention_rate)} ({p(lo)}-{p(hi)})",
            p(r.share_of_voice),
            _num(r.avg_rank, ".1f"),
            p(r.top3_rate),
            p(r.recommendation_rate),
            _num(r.net_sentiment, "+.2f"),
            f"{r.visibility_score:.1f}",
        )
    console = Console()
    console.print(table)
    console.print(
        "rank = avg rank where ranked; rec = recommendation rate; score = Visibility Score 0-100."
    )
    if any(r.small_sample for r in rows):
        console.print("* fewer than 30 answers: trust the interval, not the point.")
    for r in [*rows, *overall] if by not in ("all", "run") else rows:
        if r.coverage < 1 or r.llm_ok_share < 1:
            console.print(
                f"[yellow]warning ({r.slice}): {r.coverage:.0%} of ok answers extracted, "
                f"{r.llm_ok_share:.0%} of those with Stage B; numbers describe that subset "
                "(re-run `citewise extract run`)[/yellow]"
            )
            break
    if cites is not None and cites.total:
        top = ", ".join(f"{d.domain} {d.share:.0%}" for d in cites.with_brand[:5])
        console.print(f"citations: {cites.total}, own domain {cites.own_domain_share:.1%}; "
                      f"top when {name} is mentioned: {top}")  # fmt: skip


@metrics_app.command("competitors")
def metrics_competitors(
    brand: str = typer.Argument(..., help="Brand name or id."),
    run: int | None = typer.Option(None, "--run", help="Run id (default: latest extracted)."),
) -> None:
    """Tracked competitors and emerging (untracked) companies the engines name."""
    from citewise.core.db import session_scope
    from citewise.services.metrics.visibility import Filters, NoExtractionsError, competitors
    from citewise.services.panel.common import BrandNotFoundError, resolve_brand

    try:
        with session_scope() as session:
            rows = competitors(session, resolve_brand(session, brand).id, Filters(run_id=run))
    except (BrandNotFoundError, NoExtractionsError) as exc:
        _fail(str(exc))
        return
    for r in rows:
        tag = "EMERGING" if r.entity_type == "unknown" else "tracked"
        typer.echo(f"{r.canonical:<20} {tag:<8} {r.answers_mentioning:>4} answers "
                   f"({r.mention_rate:.0%})  avg rank {_num(r.avg_rank, '.1f'):>4}  "
                   f"sentiment {r.avg_sentiment:+.2f}")  # fmt: skip


@eval_app.command("extract")
def eval_extract(
    golden: Path = typer.Option(GOLDEN_MENTIONS, help="Golden set (JSONL)."),
    out: Path = typer.Option(EXTRACT_REPORT, help="Markdown report path."),
    max_cost: float | None = typer.Option(None, min=0, help=MAX_COST_HELP),
    min_f1: float = typer.Option(
        0.85, min=0, max=1, help="Exit 1 if the merged F1 is below this (CI gate)."
    ),
) -> None:
    """Precision/recall/F1, rank accuracy and sentiment MAE for Stage A, Stage B and the
    merge, with errors broken down by type and tag."""
    from citewise.evals.extract import evaluate, load_brand_configs, load_golden, render_markdown

    settings = get_settings()
    rows = load_golden(golden)
    chat = _budgeted_chat(max_cost)
    report = evaluate(rows, load_brand_configs(settings.brands_dir), chat,
                      model=settings.model_extract, mode=settings.citewise_mode,
                      fixtures_dir=settings.fixtures_dir)  # fmt: skip
    for stage, s in report.stages.items():
        typer.echo(
            f"{stage:<7} P {s.precision:.3f}  R {s.recall:.3f}  F1 {s.f1:.3f}  "
            f"rank acc {_num(s.rate(s.rank_ok), '.3f')}  "
            f"sentiment MAE {_num(s.sentiment_mae, '.3f')}"
        )
    merged = report.stages["merged"]
    errors = ", ".join(f"{k} {v}" for k, v in merged.errors.most_common())
    typer.echo(f"merged errors by type: {errors or 'none'}")
    hand = report.handwritten.get("merged")
    if hand is not None and report.handwritten_rows:
        typer.echo(f"hand-written rows only ({report.handwritten_rows}): merged F1 {hand.f1:.3f}")
    typer.echo(f"golden rows: {report.rows} ({report.reviewed} hand-reviewed)")
    typer.echo(f"Stage B: {report.stage_b_source}")
    rel = golden.relative_to(REPO_ROOT) if golden.is_relative_to(REPO_ROOT) else golden
    out.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.now(UTC).date().isoformat()
    out.write_text(render_markdown(report, rel.as_posix(), today), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(report.as_dict(), indent=2) + "\n",
                                        encoding="utf-8")  # fmt: skip
    typer.echo(f"report: {out}")
    _budget_note(chat)
    if merged.f1 < min_f1:
        _fail(f"merged F1 {merged.f1:.3f} is below --min-f1 {min_f1:.2f}")
