"""Brand CRUD and YAML seeding."""

from __future__ import annotations

from pathlib import Path

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from citewise.models.brand import Brand
from citewise.schemas.brand import BrandCreate


class BrandExistsError(ValueError):
    pass


def list_brands(session: Session) -> list[Brand]:
    return list(session.scalars(select(Brand).order_by(Brand.id)))


def get_brand_by_name(session: Session, name: str) -> Brand | None:
    return session.scalar(select(Brand).where(Brand.name == name))


def create_brand(session: Session, data: BrandCreate) -> Brand:
    if get_brand_by_name(session, data.name) is not None:
        raise BrandExistsError(f"Brand {data.name!r} already exists")
    brand = Brand(**data.model_dump())
    session.add(brand)
    session.flush()
    return brand


def upsert_brand(session: Session, data: BrandCreate) -> tuple[Brand, bool]:
    """Insert or update by name. Returns (brand, created). Makes `citewise seed` idempotent."""
    brand = get_brand_by_name(session, data.name)
    if brand is None:
        return create_brand(session, data), True
    for key, value in data.model_dump().items():
        setattr(brand, key, value)
    session.flush()
    return brand, False


def load_brand_files(directory: Path) -> list[BrandCreate]:
    """Parse and validate every `*.yaml` / `*.yml` file in `directory`, sorted by filename."""
    files = sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")])
    return [
        BrandCreate.model_validate(yaml.safe_load(f.read_text(encoding="utf-8"))) for f in files
    ]


def seed_brands(session: Session, directory: Path) -> tuple[int, int]:
    """Load brand YAMLs into the DB. Returns (created, updated)."""
    created = updated = 0
    for data in load_brand_files(directory):
        _, is_new = upsert_brand(session, data)
        created += is_new
        updated += not is_new
    return created, updated
