"""FactCheck results for a brand (Module 4)."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from citewise.core.db import get_db
from citewise.models.brand import Brand
from citewise.services.factcheck.report import summarize

router = APIRouter(tags=["factcheck"])


@router.get("/brands/{brand_id}/factcheck")
def get_factcheck(
    brand_id: int,
    run_id: int | None = Query(None, description="Default: the latest fact-checked run."),
    verdict: Literal["contradicted", "supported", "not_in_kb", "all"] = "contradicted",
    limit: int = Query(200, ge=1, le=1000),
    db: Session = Depends(get_db),
) -> dict:
    """Verdict counts (overall and per engine), the domains behind wrong claims, and the
    claims themselves: text, KB evidence quote, engine, and likely source URL."""
    if db.get(Brand, brand_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Brand not found")
    s = summarize(db, brand_id, run_id=run_id, verdict=None if verdict == "all" else verdict,
                  limit=limit)  # fmt: skip
    if s.run_id is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No fact-checked runs for this brand: run `citewise factcheck run <run_id>`",
        )
    return s.as_dict()
