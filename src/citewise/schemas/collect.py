"""API schemas for scan runs."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from citewise.providers.engines.registry import ENGINES, parse_engines

MAX_SYSTEM_PROMPT_CHARS = 500


def clean_system_prompt(value: str | None) -> str | None:
    """The persona/locale line goes to every engine as a system message: one short line,
    no template markers."""
    if value is None:
        return None
    value = " ".join(value.split())
    if not value:
        return None
    if len(value) > MAX_SYSTEM_PROMPT_CHARS:
        raise ValueError(f"system prompt must be at most {MAX_SYSTEM_PROMPT_CHARS} characters")
    if "===" in value:
        raise ValueError("system prompt must not contain '==='")
    return value


class RunCreate(BaseModel):
    engines: list[str] = Field(default_factory=lambda: list(ENGINES))
    size: int = Field(default=50, ge=1, le=500)
    max_cost: float | None = Field(
        default=None, ge=0, description="USD; default MAX_COST_USD_PER_RUN"
    )
    system_prompt: str | None = Field(
        default=None, description='Optional persona/locale, e.g. "The user is in Toronto, Canada."'
    )
    use_cache: bool = True

    @field_validator("engines")
    @classmethod
    def _engines(cls, v: list[str]) -> list[str]:
        return parse_engines(v)

    @field_validator("system_prompt")
    @classmethod
    def _system(cls, v: str | None) -> str | None:
        return clean_system_prompt(v)


class RunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    brand_id: int
    status: str
    mode: str
    engines: list[str]
    models: dict[str, str]
    system_prompt: str | None
    total: int
    done: int
    failed: int
    progress: float
    max_cost_usd: float
    estimated_cost_usd: float
    cost_usd: float
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
