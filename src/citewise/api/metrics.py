"""Visibility metrics and competitors for a brand (Module 3)."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from citewise.core.config import get_settings
from citewise.core.db import get_db
from citewise.models.brand import Brand
from citewise.services.metrics.visibility import (
    Filters,
    NoExtractionsError,
    citation_share,
    competitors,
    compute_metrics,
)

router = APIRouter(tags=["metrics"])


def _brand(db: Session, brand_id: int) -> Brand:
    brand = db.get(Brand, brand_id)
    if brand is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Brand not found")
    return brand


@router.get("/brands/{brand_id}/metrics")
def get_metrics(
    brand_id: int,
    run_id: int | None = Query(None, description="Default: the latest extracted run."),
    engine: str | None = None,
    topic: int | None = Query(None, description="Topic id."),
    intent: str | None = None,
    by: Literal["engine", "topic", "intent", "run", "all"] = "all",
    db: Session = Depends(get_db),
) -> dict:
    """Mention rate (with 95% bootstrap CI), share of voice, rank, recommendation rate,
    sentiment and Visibility Score, optionally sliced, plus citation share."""
    brand = _brand(db, brand_id)
    f = Filters(run_id=run_id, engine=engine, topic_id=topic, intent=intent)
    n = get_settings().metrics_bootstrap
    try:
        slices = compute_metrics(db, brand.id, f, slice_by=None if by == "all" else by,
                                 bootstrap=n)  # fmt: skip
        cites = citation_share(db, brand.id, brand.domain, f) if by != "run" else None
    except NoExtractionsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return {
        "brand_id": brand.id,
        "by": by,
        "filters": {"run_id": run_id, "engine": engine, "topic": topic, "intent": intent},
        "slices": [s.as_dict() for s in slices],
        "citations": None
        if cites is None
        else {
            "total": cites.total,
            "own_domain_share": cites.own_domain_share,
            "top_with_brand": [d.__dict__ for d in cites.with_brand],
            "top_without_brand": [d.__dict__ for d in cites.without_brand],
        },
    }


@router.get("/brands/{brand_id}/competitors")
def get_competitors(
    brand_id: int,
    run_id: int | None = Query(None, description="Default: the latest extracted run."),
    db: Session = Depends(get_db),
) -> list[dict]:
    """Tracked competitors plus emerging ones (companies engines name that the brand
    doesn't track yet), most-mentioned first."""
    brand = _brand(db, brand_id)
    try:
        rows = competitors(db, brand.id, Filters(run_id=run_id))
    except NoExtractionsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return [{**r.__dict__, "emerging": r.entity_type == "unknown"} for r in rows]
