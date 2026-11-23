"""Read-only panel endpoints: a brand's prompts and topics."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from citewise.core.db import get_db
from citewise.models.brand import Brand
from citewise.models.panel import Prompt, Topic
from citewise.schemas.panel import PromptRead, TopicRead

router = APIRouter(prefix="/brands/{brand_id}", tags=["panel"])

IntentFilter = Literal["informational", "commercial", "comparison", "navigational", "transactional"]


def _brand_or_404(db: Session, brand_id: int) -> Brand:
    brand = db.get(Brand, brand_id)
    if brand is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Brand not found")
    return brand


@router.get("/prompts", response_model=list[PromptRead])
def list_prompts(
    brand_id: int,
    topic_id: int | None = None,
    intent: IntentFilter | None = None,
    in_panel: bool | None = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> list[PromptRead]:
    _brand_or_404(db, brand_id)
    stmt = select(Prompt).where(Prompt.brand_id == brand_id)
    if topic_id is not None:
        stmt = stmt.where(Prompt.topic_id == topic_id)
    if intent is not None:
        stmt = stmt.where(Prompt.intent == intent)
    if in_panel is not None:
        stmt = stmt.where(Prompt.in_panel == in_panel)
    rows = db.scalars(stmt.order_by(Prompt.id).offset(offset).limit(limit))
    return [PromptRead.model_validate(p) for p in rows]


@router.get("/topics", response_model=list[TopicRead])
def list_topics(brand_id: int, db: Session = Depends(get_db)) -> list[TopicRead]:
    _brand_or_404(db, brand_id)
    stmt = select(Topic).where(Topic.brand_id == brand_id).order_by(Topic.size.desc(), Topic.id)
    return [TopicRead.model_validate(t) for t in db.scalars(stmt)]
