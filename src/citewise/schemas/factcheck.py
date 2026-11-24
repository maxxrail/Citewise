"""Structured outputs for FactCheck (claim extraction, verification, theme labels)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

ClaimType = Literal["pricing", "feature", "company", "comparison", "other"]
VerdictLabel = Literal["supported", "contradicted", "not_in_kb"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExtractedClaim(_Strict):
    claim: str  # one atomic, self-contained factual statement about the brand
    type: ClaimType
    answer_span: str  # the exact text in the answer the claim comes from

    @field_validator("claim", "answer_span")
    @classmethod
    def _trim(cls, v: str) -> str:
        return " ".join(v.split())[:600]


class ClaimList(_Strict):
    claims: list[ExtractedClaim]


class Verdict(_Strict):
    verdict: VerdictLabel
    evidence_chunk_ids: list[str]  # ids of the passages the verdict rests on
    evidence_quote: str | None  # verbatim sentence(s) from one of those passages
    rationale: str  # one or two sentences

    @field_validator("rationale")
    @classmethod
    def _short(cls, v: str) -> str:
        return " ".join(v.split())[:600]


class ThemeLabel(_Strict):
    label: str  # 2-6 words, e.g. "roomy toe box"

    @field_validator("label")
    @classmethod
    def _clean(cls, v: str) -> str:
        return " ".join(v.split())[:80]
