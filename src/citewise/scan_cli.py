"""`citewise scan ...`, `citewise schedule ...` and `citewise worker` (Module 2)."""

from __future__ import annotations

import asyncio

import typer

from citewise.core.config import get_settings

scan_app = typer.Typer(help="Run the prompt panel across answer engines.", no_args_is_help=True)
schedule_app = typer.Typer(help="Recurring scans (run by `citewise worker`).", no_args_is_help=True)

ALL_ENGINES = "openai,anthropic,gemini,perplexity"


def _fail(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _summary(run) -> None:  # type: ignore[no-untyped-def]
    color = {"completed": typer.colors.GREEN, "partial": typer.colors.YELLOW}.get(
        run.status, typer.colors.RED
    )
    typer.secho(
        f"run {run.id} [{run.mode}]: {run.status} | {run.done}/{run.total} answers"
        f"{f', {run.failed} failed' if run.failed else ''} | "
        f"cost ${run.cost_usd:.4f} "
        f"(estimate ${run.estimated_cost_usd:.4f}, cap ${run.max_cost_usd:.2f})",
        fg=color,
    )
    if run.error:
        typer.echo(f"  {run.error}")


def _execute_with_progress(run_id: int, total: int, use_cache: bool) -> None:
    """Run with a rich progress bar; Ctrl+C leaves a resumable run."""
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

    from citewise.core.db import get_sessionmaker
    from citewise.services.collect.runner import execute_run

    settings = get_settings()
    with Progress(
        TextColumn("[bold]scan"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("${task.fields[cost]:.4f}"),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("scan", total=total, cost=0.0)

        def on_progress(run) -> None:  # type: ignore[no-untyped-def]
            progress.update(task, completed=run.done + run.failed, cost=run.cost_usd)

        try:
            final = asyncio.run(
                execute_run(
                    run_id,
                    settings,
                    get_sessionmaker(),
                    on_progress=on_progress,
                    use_cache=use_cache,
                )
            )
        except KeyboardInterrupt:
            progress.stop()
            typer.secho(
                f"Interrupted. Finish it with: citewise scan resume {run_id}",
                fg=typer.colors.YELLOW,
            )
            raise typer.Exit(code=130) from None
    _summary(final)


@scan_app.command("run")
def run_scan(
    brand: str = typer.Argument(..., help="Brand name or id."),
    engines: str = typer.Option(ALL_ENGINES, help="Comma-separated engines."),
    size: int = typer.Option(50, min=1, max=500, help="Prompts to scan (stratified panel)."),
    max_cost: float | None = typer.Option(
        None, min=0, help="USD cap (default MAX_COST_USD_PER_RUN). Checked before and during."
    ),
    system: str | None = typer.Option(
        None, help='Persona/locale system message, e.g. "The user is in Toronto, Canada."'
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Re-ask even if asked today."),
) -> None:
    """Send the brand's panel to the engines and store answers + citations."""
    from citewise.core.db import session_scope
    from citewise.providers.engines.registry import parse_engines
    from citewise.schemas.collect import clean_system_prompt
    from citewise.services.collect.runner import BudgetEstimateError, create_run
    from citewise.services.panel.common import BrandNotFoundError, resolve_brand

    settings = get_settings()
    cap = settings.max_cost_usd_per_run if max_cost is None else max_cost
    try:
        engine_list = parse_engines(engines)
        with session_scope() as session:
            b = resolve_brand(session, brand)
            run = create_run(
                session,
                b,
                engines=engine_list,
                size=size,
                max_cost=cap,
                settings=settings,
                system_prompt=clean_system_prompt(system),
            )
            run_id, total, estimate = run.id, run.total, run.estimated_cost_usd
    except BudgetEstimateError as exc:
        _fail(f"Not started: {exc}")
        return
    except (BrandNotFoundError, ValueError) as exc:
        _fail(str(exc))
        return
    typer.echo(
        f"run {run_id} [{settings.citewise_mode}]: {total // len(engine_list)} prompts x "
        f"{len(engine_list)} engines = {total} answers, estimated ${estimate:.4f} (cap ${cap:.2f})"
    )
    _execute_with_progress(run_id, total, use_cache=not no_cache)


@scan_app.command()
def resume(
    run_id: int = typer.Argument(..., help="Run id (see `citewise scan list`)."),
    max_cost: float | None = typer.Option(
        None, min=0, help="Raise the run's cap, e.g. after it stopped on budget."
    ),
) -> None:
    """Finish an interrupted, failed or budget-stopped run (no duplicates)."""
    from citewise.core.db import session_scope
    from citewise.models.collect import Run

    with session_scope() as session:
        run = session.get(Run, run_id)
        if run is None:
            _fail(f"No run {run_id}")
            return
        if run.status == "completed":
            typer.echo(f"run {run_id} is already completed.")
            return
        if max_cost is not None:
            run.max_cost_usd = max_cost
        remaining = run.total - run.done
        if run.mode != get_settings().citewise_mode:
            _fail(f"run {run_id} was a {run.mode} run; resume it with CITEWISE_MODE={run.mode}")
            return
    typer.echo(f"resuming run {run_id}: {remaining} answers to go")
    _execute_with_progress(run_id, remaining, use_cache=True)


@scan_app.command()
def status(run_id: int = typer.Argument(..., help="Run id.")) -> None:
    """Show a run's status, progress and cost."""
    from citewise.core.db import session_scope
    from citewise.models.collect import Run

    with session_scope() as session:
        run = session.get(Run, run_id)
        if run is None:
            _fail(f"No run {run_id}")
            return
        _summary(run)


@scan_app.command("list")
def list_runs(
    brand: str | None = typer.Argument(None, help="Brand name or id (default: all)."),
    limit: int = typer.Option(20, min=1),
) -> None:
    """Recent runs, newest first."""
    from sqlalchemy import select

    from citewise.core.db import session_scope
    from citewise.models.collect import Run
    from citewise.services.panel.common import BrandNotFoundError, resolve_brand

    try:
        with session_scope() as session:
            stmt = select(Run).order_by(Run.id.desc()).limit(limit)
            if brand:
                stmt = stmt.where(Run.brand_id == resolve_brand(session, brand).id)
            runs = list(session.scalars(stmt))
            for r in runs:
                typer.echo(
                    f"{r.id:>5}  {r.created_at:%Y-%m-%d %H:%M}  {r.mode:<4}  {r.status:<14} "
                    f"{r.done:>4}/{r.total:<4} ${r.cost_usd:.4f}  {','.join(r.engines)}"
                )
            if not runs:
                typer.echo("no runs yet")
    except BrandNotFoundError as exc:
        _fail(str(exc))


# ---------------------------------------------------------------- schedules


@schedule_app.command("add")
def schedule_add(
    brand: str = typer.Argument(..., help="Brand name or id."),
    cron: str = typer.Option(..., help='5-field crontab in UTC, e.g. "0 9 * * 1" (Mon 09:00).'),
    engines: str = typer.Option(ALL_ENGINES, help="Comma-separated engines."),
    size: int = typer.Option(50, min=1, max=500),
    max_cost: float | None = typer.Option(None, min=0, help="USD cap per run."),
    system: str | None = typer.Option(None, help="Persona/locale system message."),
) -> None:
    """Add a recurring scan."""
    from citewise.core.db import session_scope
    from citewise.providers.engines.registry import parse_engines
    from citewise.schemas.collect import clean_system_prompt
    from citewise.services.panel.common import BrandNotFoundError, resolve_brand
    from citewise.workers.scheduler import add_schedule, parse_cron

    settings = get_settings()
    try:
        with session_scope() as session:
            b = resolve_brand(session, brand)
            s = add_schedule(
                session,
                b,
                cron=cron,
                engines=parse_engines(engines),
                size=size,
                max_cost=settings.max_cost_usd_per_run if max_cost is None else max_cost,
                system_prompt=clean_system_prompt(system),
            )
            nxt = parse_cron(s.cron).get_next_fire_time(None, _now())
            typer.echo(
                f"schedule {s.id}: {b.name} '{s.cron}' (UTC); next run {nxt:%Y-%m-%d %H:%M} UTC"
            )
    except (BrandNotFoundError, ValueError) as exc:
        _fail(str(exc))
    typer.echo("Start `citewise worker` to run schedules.")


@schedule_app.command("list")
def schedule_list() -> None:
    """All schedules."""
    from sqlalchemy import select

    from citewise.core.db import session_scope
    from citewise.models.brand import Brand
    from citewise.models.collect import Schedule

    with session_scope() as session:
        rows = list(session.scalars(select(Schedule).order_by(Schedule.id)))
        for s in rows:
            brand = session.get(Brand, s.brand_id)
            name = brand.name if brand else str(s.brand_id)
            typer.echo(
                f"{s.id:>4}  {'on ' if s.enabled else 'off'}  {name:<12} "
                f"'{s.cron}'  size={s.size}  cap=${s.max_cost_usd:.2f}  {','.join(s.engines)}  "
                f"last_run={s.last_run_id or '-'}"
            )
        if not rows:
            typer.echo("no schedules")


@schedule_app.command("remove")
def schedule_remove(schedule_id: int = typer.Argument(...)) -> None:
    """Delete a schedule (its past runs are kept)."""
    from citewise.core.db import session_scope
    from citewise.models.collect import Schedule

    with session_scope() as session:
        s = session.get(Schedule, schedule_id)
        if s is None:
            _fail(f"No schedule {schedule_id}")
            return
        session.delete(s)
    typer.echo(f"schedule {schedule_id} removed")


def worker() -> None:
    """Run the scheduler in the foreground (Ctrl+C to stop)."""
    from citewise.core.db import get_sessionmaker
    from citewise.workers.scheduler import build_scheduler

    scheduler = build_scheduler(get_settings(), get_sessionmaker())
    jobs = [j for j in scheduler.get_jobs() if j.id.startswith("schedule-")]
    typer.echo(f"worker started with {len(jobs)} schedule(s); syncing with the DB every minute")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        typer.echo("worker stopped")


def _now():  # type: ignore[no-untyped-def]
    from datetime import UTC, datetime

    return datetime.now(UTC)
