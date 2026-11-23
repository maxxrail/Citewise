"""Schemas for the prompt panel: LLM structured outputs and API read models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, field_validator

Intent = Literal["informational", "commercial", "comparison", "navigational", "transactional"]
FunnelStage = Literal["awareness", "consideration", "decision"]
INTENTS: tuple[str, ...] = get_args(Intent)
FUNNEL_STAGES: tuple[str, ...] = get_args(FunnelStage)

MAX_LABEL_WORDS = 5


class _Strict(BaseModel):
    # extra="forbid" emits additionalProperties: false, which OpenAI strict mode requires.
    model_config = ConfigDict(extra="forbid")


# ---- LLM outputs (constraints that JSON schema can't express are checked by validators) ----


class GeneratedPrompt(_Strict):
    text: str
    funnel_stage: FunnelStage


class GeneratedBatch(_Strict):
    prompts: list[GeneratedPrompt]


class IntentLabel(_Strict):
    id: int
    intent: Intent


class IntentBatch(_Strict):
    labels: list[IntentLabel]


class TopicLabel(_Strict):
    label: str
    description: str

    @field_validator("label")
    @classmethod
    def _short(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("label must not be blank")
        if len(v.split()) > MAX_LABEL_WORDS or len(v) > 60:
            raise ValueError(f"label must be at most {MAX_LABEL_WORDS} words")
        return v

    @field_validator("description")
    @classmethod
    def _one_line(cls, v: str) -> str:
        return " ".join(v.split())[:300]


# ---- API ----


class TopicRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    brand_id: int
    label: str
    description: str
    size: int
    algorithm: str
    label_source: str
    created_at: datetime


class PromptRead(BaseModel):
    """No embedding: 1536 floats per row would dwarf everything else in the response."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    brand_id: int
    text: str
    source: str
    template_version: str | None
    persona: str | None
    funnel_stage: str | None
    requested_intent: str | None
    intent: str | None
    intent_source: str | None
    topic_id: int | None
    centrality: float | None
    in_panel: bool
