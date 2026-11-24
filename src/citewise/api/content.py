"""Content agent: opportunities, items awaiting approval, approve / reject (Module 5)."""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from citewise.core.config import get_settings
from citewise.core.db import get_db, get_session_factory
from citewise.models.brand import Brand
from citewise.models.content import ContentItem, ContentOpportunity
from citewise.services.content.pipeline import (
    STATES,
    ContentError,
    Deps,
    TransitionError,
    approve,
    reject,
    run_item,
)
from citewise.services.content.report import (
    item_detail,
    item_summary,
    opportunity_dict,
    versions_of,
)

router = APIRouter(tags=["content"])
log = logging.getLogger(__name__)

Status = Literal[
    "opportunity",
    "brief",
    "outline",
    "draft",
    "critique",
    "revise",
    "final",
    "awaiting_approval",
    "approved",
    "rejected",
]
assert set(Status.__args__) == set(STATES)  # type: ignore[attr-defined]


class ReviewIn(BaseModel):
    comment: str = Field("", max_length=4000)


def _item(db: Session, item_id: int) -> ContentItem:
    item = db.get(ContentItem, item_id)
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Content not found")
    return item


@router.get("/brands/{brand_id}/content/opportunities")
def list_opportunities(
    brand_id: int,
    kind: Literal["gap", "citation", "fact_fix"] | None = None,
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[dict]:
    """Stored opportunities, best first (refresh with `citewise content opportunities`)."""
    if db.get(Brand, brand_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Brand not found")
    q = select(ContentOpportunity).where(
        ContentOpportunity.brand_id == brand_id, ContentOpportunity.status != "stale"
    )
    if kind:
        q = q.where(ContentOpportunity.kind == kind)
    rows = db.scalars(
        q.order_by(ContentOpportunity.score.desc(), ContentOpportunity.id).limit(limit)
    )
    return [opportunity_dict(o) for o in rows]


@router.get("/content")
def list_content(
    status_: Status | None = Query(None, alias="status"),
    brand_id: int | None = None,
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
) -> list[dict]:
    """Content items, newest first; `?status=awaiting_approval` is the review queue."""
    q = select(ContentItem)
    if status_:
        q = q.where(ContentItem.status == status_)
    if brand_id is not None:
        q = q.where(ContentItem.brand_id == brand_id)
    items = db.scalars(
        q.order_by(ContentItem.updated_at.desc(), ContentItem.id.desc()).limit(limit)
    )
    return [item_summary(i, versions_of(db, i.id)) for i in items]


@router.get("/content/{item_id}")
def get_content(item_id: int, db: Session = Depends(get_db)) -> dict:
    """The current version (Markdown, JSON-LD, why-note, checks, scores), every earlier
    version's scores and feedback, and the review history."""
    return item_detail(db, _item(db, item_id))


@router.post("/content/{item_id}/approve")
def approve_content(item_id: int, payload: ReviewIn, db: Session = Depends(get_db)) -> dict:
    item = _item(db, item_id)
    try:
        approve(db, item, payload.comment)
    except TransitionError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    db.commit()
    return item_detail(db, item)


def _revise_in_background(item_id: int, factory: sessionmaker[Session]) -> None:
    """Sync on purpose (runs in Starlette's threadpool). Failures land on item.error."""
    from citewise.providers import get_chat_provider, get_embedder
    from citewise.providers.budget import BudgetedChatProvider

    settings = get_settings()
    try:
        chat = BudgetedChatProvider(
            get_chat_provider(session_factory=factory), settings.max_cost_usd_per_run
        )
        deps = Deps(chat, get_embedder(session_factory=factory), settings)
        run_item(factory, item_id, deps)
    except Exception:
        log.exception("content %s: background revision failed", item_id)


@router.post("/content/{item_id}/reject", status_code=status.HTTP_202_ACCEPTED)
def reject_content(
    item_id: int,
    payload: ReviewIn,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
    factory: sessionmaker[Session] = Depends(get_session_factory),
) -> dict:
    """Record the rejection and revise in the background using the comment; the item comes
    back to awaiting_approval with a new version. Poll GET /content/{id}."""
    item = _item(db, item_id)
    try:
        reject(db, item, payload.comment)
    except TransitionError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ContentError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc
    db.commit()
    background.add_task(_revise_in_background, item.id, factory)
    return {**item_summary(item), "revision": "queued"}
