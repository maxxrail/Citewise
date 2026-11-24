"""Scan runs: start one for a brand, poll its status/progress/cost."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from citewise.core.config import get_settings
from citewise.core.db import get_db, get_session_factory
from citewise.models.brand import Brand
from citewise.models.collect import Run
from citewise.schemas.collect import RunCreate, RunRead
from citewise.services.collect.runner import BudgetEstimateError, create_run, execute_run

router = APIRouter(tags=["runs"])
log = logging.getLogger(__name__)


def _execute(run_id: int, use_cache: bool, factory: sessionmaker[Session]) -> None:
    """Background task, deliberately sync: Starlette runs sync tasks in its threadpool, so
    the scan's blocking DB writes (several per answer) never stall the API's event loop.
    Failures are stored on the run (status/error) for GET /runs/{id}; here they are only
    logged, since no client is waiting."""
    try:
        asyncio.run(execute_run(run_id, get_settings(), factory, use_cache=use_cache))
    except Exception:
        log.exception("background scan run %s failed", run_id)


@router.post(
    "/brands/{brand_id}/runs", response_model=RunRead, status_code=status.HTTP_202_ACCEPTED
)
def start_run(
    brand_id: int,
    payload: RunCreate,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
    factory: sessionmaker[Session] = Depends(get_session_factory),
) -> RunRead:
    """Plan a run (budget-checked) and execute it in the background. Poll GET /runs/{id}."""
    brand = db.get(Brand, brand_id)
    if brand is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Brand not found")
    settings = get_settings()
    max_cost = settings.max_cost_usd_per_run if payload.max_cost is None else payload.max_cost
    try:
        run = create_run(
            db,
            brand,
            engines=payload.engines,
            size=payload.size,
            max_cost=max_cost,
            settings=settings,
            system_prompt=payload.system_prompt,
        )
        db.commit()
    except BudgetEstimateError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={"error": str(exc), "estimate_usd": exc.estimate, "max_cost_usd": exc.cap},
        ) from exc
    except ValueError as exc:  # no prompts / unclustered panel
        db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    background.add_task(_execute, run.id, payload.use_cache, factory)
    return RunRead.model_validate(run)


@router.get("/runs/{run_id}", response_model=RunRead)
def get_run(run_id: int, db: Session = Depends(get_db)) -> RunRead:
    run = db.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")
    return RunRead.model_validate(run)


@router.get("/brands/{brand_id}/runs", response_model=list[RunRead])
def list_runs(
    brand_id: int,
    limit: int = Query(20, ge=1, le=200),
    db: Session = Depends(get_db),
) -> list[RunRead]:
    if db.get(Brand, brand_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Brand not found")
    stmt = select(Run).where(Run.brand_id == brand_id).order_by(Run.id.desc()).limit(limit)
    return [RunRead.model_validate(r) for r in db.scalars(stmt)]
