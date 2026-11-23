from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Brand fields are interpolated into every panel/extraction prompt (Module 1+), so they are
# bounded (cost: N aliases x M competitors x every prompt) and single-line (a newline is the
# cheapest way to smuggle a fake instruction or section break into a prompt template).
MAX_LIST_ITEMS = 20
MAX_TERM_LEN = 100
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_HOSTNAME = re.compile(r"^(?=.{1,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9-]{2,63}$")


def _clean_line(v: str) -> str:
    v = v.strip()
    if not v:
        raise ValueError("must not be blank")
    if _CONTROL_CHARS.search(v):
        raise ValueError("must be a single line without control characters")
    return v


class BrandCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    domain: str = Field(min_length=1, max_length=255)
    category: str = Field(min_length=1, max_length=200)
    aliases: list[str] = Field(default_factory=list, max_length=MAX_LIST_ITEMS)
    competitors: list[str] = Field(default_factory=list, max_length=MAX_LIST_ITEMS)

    @field_validator("name", "category")
    @classmethod
    def _single_line(cls, v: str) -> str:
        return _clean_line(v)

    @field_validator("domain")
    @classmethod
    def _normalize_domain(cls, v: str) -> str:
        """Reduce a URL or bare host to `example.com` and reject anything that isn't a host."""
        v = _clean_line(v).lower()
        v = v.removeprefix("https://").removeprefix("http://")
        v = v.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0].split(":", 1)[0]
        v = v.removeprefix("www.")
        if not _HOSTNAME.match(v):
            raise ValueError(f"{v!r} is not a valid domain (expected e.g. example.com)")
        return v

    @field_validator("aliases", "competitors")
    @classmethod
    def _clean_terms(cls, v: list[str]) -> list[str]:
        """Strip, bound length, and drop case-insensitive duplicates (first spelling wins)."""
        seen: set[str] = set()
        out: list[str] = []
        for term in v:
            term = _clean_line(term)
            if len(term) > MAX_TERM_LEN:
                raise ValueError(f"terms must be at most {MAX_TERM_LEN} characters")
            if term.casefold() not in seen:
                seen.add(term.casefold())
                out.append(term)
        return out


class BrandRead(BaseModel):
    """Output model. Deliberately no input validators: rows written under older, looser
    rules must still be listable rather than 500ing GET /brands."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    domain: str
    category: str
    aliases: list[str]
    competitors: list[str]
    created_at: datetime
