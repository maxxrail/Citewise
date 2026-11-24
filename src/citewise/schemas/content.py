"""Structured outputs for the content pipeline's LLM steps (Module 5).

Every step returns one of these, validated by Pydantic (strict: no extra fields), so a
step either produces usable data or is recorded as failed and retried on resume.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ContentFormat = Literal["faq", "comparison", "guide"]
CRITERIA = ("factuality", "coverage", "tone", "aeo_structure", "originality")
Criterion = Literal["factuality", "coverage", "tone", "aeo_structure", "originality"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BriefOut(_Strict):
    """What the model adds to the deterministic part of the brief (targets, passages)."""

    audience: str
    angle: str
    questions: list[str] = Field(description="The questions the page must answer, as headings")
    must_cover: list[str] = Field(description="chunk ids from the passages that must be used")
    comparison_points: list[str]
    format: ContentFormat
    word_count: int
    unsupported_requests: list[str] = Field(
        description="Requested claims no passage supports; they are left out"
    )


class OutlineSection(_Strict):
    heading: str
    purpose: str
    chunk_ids: list[str]


class OutlineOut(_Strict):
    title: str
    direct_answer: str = Field(description="Plan for the first two sentences")
    sections: list[OutlineSection]
    faq_questions: list[str]


class Section(_Strict):
    heading: str
    body: str = Field(description="Markdown; every factual sentence ends with [kb:<chunk id>]")


class FAQ(_Strict):
    question: str
    answer: str


class DraftOut(_Strict):
    title: str
    summary: str = Field(description="At most two sentences that answer the main question")
    sections: list[Section]
    faq: list[FAQ]


class Declined(_Strict):
    request: str
    reason: str


class RevisionOut(DraftOut):
    changes: list[str] = Field(description="What changed and which issue it fixes")
    declined: list[Declined] = Field(description="Requested changes not made, and why")


class CriterionScore(_Strict):
    criterion: Criterion
    rationale: str
    issues: list[str]
    score: int = Field(ge=1, le=5)


class CritiqueOut(_Strict):
    scores: list[CriterionScore]
