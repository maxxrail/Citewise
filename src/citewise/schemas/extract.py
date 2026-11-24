"""Structured output for Stage B mention extraction, and API schemas for metrics."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

SentimentLabel = Literal["positive", "neutral", "negative", "mixed"]


class _Strict(BaseModel):
    # extra="forbid" emits additionalProperties: false, which OpenAI strict mode requires.
    model_config = ConfigDict(extra="forbid")


class ExtractedMention(_Strict):
    """One company/brand the answer names. No numeric bounds in the schema (strict-mode
    support for `minimum` varies by provider); out-of-range values are clamped instead."""

    entity: str  # the surface form exactly as written in the answer
    canonical: str  # the company name, e.g. "Asana" for "Asana Inc."
    rank: int | None  # 1-based position as the subject of a list item, else null
    sentiment: float  # -1 (very negative) .. 1 (very positive)
    sentiment_label: SentimentLabel
    is_recommended: bool
    snippet: str  # the sentence or list item where it is discussed

    @field_validator("sentiment")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(-1.0, min(1.0, float(v)))

    @field_validator("rank")
    @classmethod
    def _positive(cls, v: int | None) -> int | None:
        return v if v is not None and v >= 1 else None

    @field_validator("entity", "canonical", "snippet")
    @classmethod
    def _trim(cls, v: str) -> str:
        return " ".join(v.split())[:500]


class MentionExtraction(_Strict):
    mentions: list[ExtractedMention]
