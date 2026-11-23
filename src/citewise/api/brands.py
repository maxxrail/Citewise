from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from citewise.core.db import get_db
from citewise.schemas.brand import BrandCreate, BrandRead
from citewise.services import brands as svc

router = APIRouter(prefix="/brands", tags=["brands"])


@router.get("", response_model=list[BrandRead])
def list_brands(db: Session = Depends(get_db)) -> list[BrandRead]:
    return [BrandRead.model_validate(b) for b in svc.list_brands(db)]


@router.post("", response_model=BrandRead, status_code=status.HTTP_201_CREATED)
def create_brand(payload: BrandCreate, db: Session = Depends(get_db)) -> BrandRead:
    try:
        brand = svc.create_brand(db, payload)
    except svc.BrandExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    db.commit()
    return BrandRead.model_validate(brand)
