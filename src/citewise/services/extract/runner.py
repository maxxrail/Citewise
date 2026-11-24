"""`citewise extract run <run_id>`: extract mentions from every ok answer in a run.

Idempotent: an answer that already has an extraction at the current EXTRACTOR_VERSION
(with a successful Stage B) is skipped, and writing an answer's extraction replaces its old
mentions in one transaction. So re-running, or running after a prompt change, never
duplicates rows. An answer whose Stage B failed is stored with Stage A only
(`llm_ok=False`) so metrics have something, and it is retried on the next run.

LLM calls happen in a small thread pool *before* any DB write, so no transaction is held
open across model latency. When the spend cap is hit, the remaining answers are left
unextracted (not stored as Stage-A-only), so the next run picks them up.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from citewise.core.config import Settings
from citewise.models.brand import Brand
from citewise.models.collect import Answer, Run
from citewise.models.extract import AnswerExtraction
from citewise.models.extract import Mention as MentionRow
from citewise.providers.base import ChatProvider
from citewise.schemas.extract import ExtractedMention
from citewise.services.extract.mentions import (
    PROMPT_NAME,
    PROMPT_VERSION,
    Mention,
    merge,
    stage_a,
    stage_b,
)
from citewise.services.extract.resolve import Resolver

log = logging.getLogger(__name__)

STAGE_A_VERSION = "a1"  # bump when Stage A or merge rules change, to re-extract
EXTRACTOR_VERSION = f"{STAGE_A_VERSION}+{PROMPT_NAME.split('/')[-1]}.{PROMPT_VERSION}"


@dataclass
class ExtractReport:
    run_id: int
    answers: int = 0  # ok answers in the run
    skipped: int = 0  # already extracted at this version
    extracted: int = 0
    llm_failed: int = 0  # stored with Stage A only
    budget_stopped: int = 0  # left for the next run
    mentions: int = 0
    disagreements: Counter[str] = field(default_factory=Counter)
    spent_usd: float = 0.0


@dataclass
class _Result:
    answer_id: int
    a: list[Mention]
    b: list[ExtractedMention] | None
    merged: list[Mention]
    disagreements: list[dict[str, Any]]
    refused: bool  # the budget refused the Stage B call


def resolver_for(brand: Brand) -> Resolver:
    return Resolver(brand.name, list(brand.aliases), list(brand.competitors))


def _store(factory: sessionmaker[Session], r: _Result) -> None:
    rows = [
        MentionRow(
            answer_id=r.answer_id,
            entity=m.entity[:500],
            canonical=m.canonical[:255],
            entity_key=m.key[:255],
            entity_type=m.entity_type,
            is_brand=m.entity_type == "brand",
            is_competitor=m.entity_type == "competitor",
            rank=m.rank,
            sentiment=m.sentiment,
            sentiment_label=m.sentiment_label,
            is_recommended=m.is_recommended,
            snippet=m.snippet,
            source=m.source,
        )
        for m in r.merged
    ]
    with factory() as session:
        session.execute(delete(MentionRow).where(MentionRow.answer_id == r.answer_id))
        session.execute(delete(AnswerExtraction).where(AnswerExtraction.answer_id == r.answer_id))
        session.add(
            AnswerExtraction(
                answer_id=r.answer_id,
                version=EXTRACTOR_VERSION,
                llm_ok=r.b is not None,
                stage_a=[m.as_dict() for m in r.a],
                stage_b=None if r.b is None else [x.model_dump() for x in r.b],
                disagreements=r.disagreements,
            )
        )
        session.add_all(rows)
        try:
            session.commit()
        except IntegrityError:  # a concurrent `extract run` stored this answer first
            session.rollback()


def extract_run(
    factory: sessionmaker[Session],
    run_id: int,
    chat: ChatProvider,
    settings: Settings,
    *,
    force: bool = False,
    on_progress: Callable[[int], None] | None = None,
) -> ExtractReport:
    with factory() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise ValueError(f"No run {run_id}")
        brand = session.get(Brand, run.brand_id)
        assert brand is not None
        resolver = resolver_for(brand)
        aliases, competitors = list(brand.aliases), list(brand.competitors)
        name, category = brand.name, brand.category
        answers = list(
            session.execute(
                select(Answer.id, Answer.text).where(Answer.run_id == run_id, Answer.ok)
            )
        )
        done = set(
            session.scalars(
                select(AnswerExtraction.answer_id)
                .join(Answer, Answer.id == AnswerExtraction.answer_id)
                .where(
                    Answer.run_id == run_id,
                    AnswerExtraction.version == EXTRACTOR_VERSION,
                    AnswerExtraction.llm_ok,
                )
            )
        )
    report = ExtractReport(run_id=run_id, answers=len(answers))
    todo = [(aid, text) for aid, text in answers if force or aid not in done]
    report.skipped = len(answers) - len(todo)
    spent_before = float(getattr(chat, "spent_usd", 0.0))

    def work(item: tuple[int, str]) -> _Result:
        aid, text = item
        refused_before = int(getattr(chat, "refused", 0))
        a = stage_a(text, resolver)
        b = stage_b(chat, model=settings.model_extract, brand=name, aliases=aliases,
                    competitors=competitors, category=category, text=text)  # fmt: skip
        refused = int(getattr(chat, "refused", 0)) > refused_before
        merged, dis = merge(text, resolver, a, b)
        return _Result(aid, a, b, merged, dis, refused and b is None)

    # Mock mode is CPU-only, and SQLite (tests) shares one connection between sessions:
    # a worker thread recording llm_calls while this thread commits would collide. So one
    # worker means *no* thread, not a pool of one that still overlaps with the writes.
    workers = 1 if settings.is_mock else settings.extract_concurrency
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = map(work, todo) if workers == 1 else pool.map(work, todo)
        for i, r in enumerate(results, start=1):
            if r.refused:
                report.budget_stopped += 1
            else:
                _store(factory, r)
                report.extracted += 1
                report.llm_failed += r.b is None
                report.mentions += len(r.merged)
                report.disagreements.update(d["type"] for d in r.disagreements)
            if on_progress is not None:
                on_progress(i)
    report.spent_usd = float(getattr(chat, "spent_usd", 0.0)) - spent_before
    return report
