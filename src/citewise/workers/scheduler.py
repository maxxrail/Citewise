"""Recurring scans: weekly snapshots are what make visibility *trends* possible.

Schedules live in the `schedules` table (so the API and CLI can manage them), not in an
APScheduler job store. `citewise worker` builds an APScheduler `BlockingScheduler`, adds one
cron job per enabled schedule, and re-syncs with the table every minute, so adding or
removing a schedule takes effect without restarting the worker.

APScheduler is pinned to 3.x: 4.x changed the API (async-first data stores).
Cron expressions are standard 5-field crontab, evaluated in UTC.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from apscheduler.schedulers.base import BaseScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from citewise.core.config import Settings
from citewise.models.brand import Brand
from citewise.models.collect import Schedule
from citewise.services.collect.runner import BudgetEstimateError, create_run, execute_run

log = logging.getLogger(__name__)

TIMEZONE = "UTC"
SYNC_JOB_ID = "sync-schedules"


def parse_cron(cron: str) -> CronTrigger:
    """Validate a 5-field crontab ("0 9 * * 1" = Mondays 09:00 UTC). Raises ValueError."""
    if len(cron.split()) != 5:
        raise ValueError(f"Expected 5 cron fields (min hour day month weekday), got {cron!r}")
    return CronTrigger.from_crontab(cron, timezone=TIMEZONE)


def add_schedule(
    session: Session,
    brand: Brand,
    *,
    cron: str,
    engines: list[str],
    size: int,
    max_cost: float,
    system_prompt: str | None = None,
) -> Schedule:
    parse_cron(cron)
    schedule = Schedule(
        brand_id=brand.id,
        cron=" ".join(cron.split()),
        engines=engines,
        size=size,
        max_cost_usd=max_cost,
        system_prompt=system_prompt,
        enabled=True,
    )
    session.add(schedule)
    session.flush()
    return schedule


def run_schedule(
    schedule_id: int, settings: Settings, factory: sessionmaker[Session]
) -> int | None:
    """Create and execute one run for a schedule. Returns the run id (None if skipped)."""
    with factory() as session:
        schedule = session.get(Schedule, schedule_id)
        if schedule is None or not schedule.enabled:
            return None
        brand = session.get(Brand, schedule.brand_id)
        if brand is None:
            return None
        try:
            run = create_run(
                session,
                brand,
                engines=list(schedule.engines),
                size=schedule.size,
                max_cost=schedule.max_cost_usd,
                settings=settings,
                system_prompt=schedule.system_prompt,
            )
        except (BudgetEstimateError, ValueError) as exc:
            log.error("schedule %s skipped: %s", schedule_id, exc)
            return None
        schedule.last_run_id = run.id
        session.commit()
        run_id = run.id
    final = asyncio.run(execute_run(run_id, settings, factory))
    log.info("schedule %s -> run %s %s ($%.4f)", schedule_id, run_id, final.status, final.cost_usd)
    return run_id


def sync_jobs(scheduler: BaseScheduler, settings: Settings, factory: sessionmaker[Session]) -> int:
    """Make the scheduler's cron jobs match the enabled rows. Returns the job count."""
    with factory() as session:
        rows = list(session.scalars(select(Schedule).where(Schedule.enabled.is_(True))))
    wanted = {f"schedule-{s.id}": s for s in rows}
    for job in scheduler.get_jobs():
        if job.id != SYNC_JOB_ID and job.id not in wanted:
            job.remove()
    for job_id, s in wanted.items():
        existing = scheduler.get_job(job_id)
        if existing is not None and getattr(existing, "name", "") == s.cron:
            continue
        kwargs: dict[str, Any] = {"schedule_id": s.id, "settings": settings, "factory": factory}
        try:
            trigger = parse_cron(s.cron)
        except ValueError:
            log.error("schedule %s has an invalid cron %r; skipping it", s.id, s.cron)
            continue
        scheduler.add_job(
            run_schedule,
            trigger=trigger,
            id=job_id,
            name=s.cron,  # used above to detect a changed expression
            kwargs=kwargs,
            replace_existing=True,
            max_instances=1,  # never overlap two runs of the same schedule
            coalesce=True,  # after downtime, run once rather than once per missed slot
            misfire_grace_time=3600,
        )
    return len(wanted)


def build_scheduler(
    settings: Settings,
    factory: sessionmaker[Session],
    scheduler: BaseScheduler | None = None,
) -> BaseScheduler:
    scheduler = scheduler or BlockingScheduler(timezone=TIMEZONE)
    sync_jobs(scheduler, settings, factory)
    scheduler.add_job(
        sync_jobs,
        "interval",
        minutes=1,
        id=SYNC_JOB_ID,
        args=[scheduler, settings, factory],
        replace_existing=True,
    )
    return scheduler
