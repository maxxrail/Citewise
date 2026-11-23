from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BrandCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    domain: str = Field(min_length=1, max_length=255)
    category: str = Field(min_length=1, max_length=200)
    aliases: list[str] = Field(default_factory=list)
    competitors: list[str] = Field(default_factory=list)

    @field_validator("name", "domain", "category")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be blank")
        return v

    @field_validator("domain")
    @classmethod
    def _normalize_domain(cls, v: str) -> str:
        v = v.lower().removeprefix("https://").removeprefix("http://").removeprefix("www.")
        return v.rstrip("/")


class BrandRead(BrandCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
