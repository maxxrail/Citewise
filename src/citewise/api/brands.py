from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
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
        db.commit()
    except svc.BrandExistsError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except IntegrityError as exc:  # concurrent insert won the race on the unique name
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=f"Brand {payload.name!r} already exists"
        ) from exc
    return BrandRead.model_validate(brand)
