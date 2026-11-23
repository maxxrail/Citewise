"""Small helpers shared by the panel services."""

from __future__ import annotations

import hashlib
import re

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from citewise.models.brand import Brand
from citewise.models.panel import Prompt

MAX_PROMPT_CHARS = 500
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_TRAILING_PUNCT = re.compile(r"[\s?!.]+$")


class BrandNotFoundError(LookupError):
    pass


def clean_prompt_text(text: str) -> str | None:
    """Collapse whitespace and control chars into one line; None if empty or too long.

    Too long is rejected rather than truncated: a cut-off question is a different question.
    """
    text = " ".join(_CONTROL.sub(" ", text).split())
    if not text or len(text) > MAX_PROMPT_CHARS:
        return None
    return text


def text_hash(text: str) -> str:
    """Identity for exact-duplicate detection: case, spacing and trailing ?!. don't matter."""
    norm = _TRAILING_PUNCT.sub("", " ".join(text.lower().split()))
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def resolve_brand(session: Session, ref: str) -> Brand:
    """Find a brand by numeric id or by name (case-insensitive)."""
    brand = session.get(Brand, int(ref)) if ref.isdigit() else None
    if brand is None:
        brands = list(session.scalars(select(Brand)))
        brand = next((b for b in brands if b.name.casefold() == ref.casefold()), None)
    if brand is None:
        raise BrandNotFoundError(f"No brand {ref!r}. Run `citewise seed` or check the name.")
    return brand


def brand_prompts(session: Session, brand_id: int) -> list[Prompt]:
    stmt = select(Prompt).where(Prompt.brand_id == brand_id).order_by(Prompt.id)
    return list(session.scalars(stmt))


def as_matrix(vectors: list[list[float]]) -> np.ndarray:
    """Stack vectors into an (n, d) float array with unit-length rows."""
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("expected a non-empty list of equal-length vectors")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms
