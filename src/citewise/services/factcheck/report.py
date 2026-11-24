"""Read model for FactCheck results (CLI `factcheck show`, API `GET /brands/{id}/factcheck`)."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from citewise.models.collect import Answer, Citation, Run
from citewise.models.kb import AnswerFactcheck, Claim


@dataclass
class ClaimView:
    id: int
    engine: str
    prompt: str
    claim: str
    type: str
    verdict: str
    evidence_quote: str | None
    evidence_chunk_keys: list[str]
    rationale: str
    source_url: str | None
    source_domain: str | None
    source_title: str | None
    source_confidence: float | None
    source_reason: str | None


@dataclass
class FactcheckSummary:
    brand_id: int
    run_id: int | None
    claims: int = 0
    verdicts: dict[str, int] = field(default_factory=dict)
    by_engine: dict[str, dict[str, int]] = field(default_factory=dict)
    top_sources: list[dict[str, Any]] = field(default_factory=list)  # domains behind errors
    items: list[ClaimView] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def latest_factchecked_run(session: Session, brand_id: int) -> int | None:
    return session.scalar(
        select(func.max(Answer.run_id))
        .join(AnswerFactcheck, AnswerFactcheck.answer_id == Answer.id)
        .join(Run, Run.id == Answer.run_id)
        .where(Run.brand_id == brand_id)
    )


def summarize(
    session: Session,
    brand_id: int,
    *,
    run_id: int | None = None,
    verdict: str | None = None,
    limit: int = 200,
) -> FactcheckSummary:
    run_id = run_id or latest_factchecked_run(session, brand_id)
    out = FactcheckSummary(brand_id=brand_id, run_id=run_id)
    if run_id is None:
        return out
    q = (
        select(Claim, Answer.engine, Answer.prompt_text, Citation)
        .join(Answer, Answer.id == Claim.answer_id)
        .outerjoin(Citation, Citation.id == Claim.source_citation_id)
        .where(Claim.brand_id == brand_id, Answer.run_id == run_id)
        .order_by(Claim.verdict, Answer.engine, Claim.id)
    )
    rows = session.execute(q).all()
    verdicts: Counter[str] = Counter()
    by_engine: dict[str, Counter[str]] = defaultdict(Counter)
    sources: Counter[str] = Counter()
    for claim, engine, prompt, cite in rows:
        verdicts[claim.verdict] += 1
        by_engine[engine][claim.verdict] += 1
        if claim.verdict == "contradicted" and cite is not None:
            sources[cite.domain] += 1
        if verdict is None or claim.verdict == verdict:
            out.items.append(
                ClaimView(
                    claim.id, engine, prompt, claim.claim_text, claim.claim_type, claim.verdict,
                    claim.evidence_quote, list(claim.evidence_chunk_keys), claim.rationale,
                    cite.url if cite else None, cite.domain if cite else None,
                    cite.title if cite else None, claim.source_confidence, claim.source_reason,
                )
            )  # fmt: skip
    out.claims = sum(verdicts.values())
    out.verdicts = dict(verdicts)
    out.by_engine = {e: dict(c) for e, c in sorted(by_engine.items())}
    out.top_sources = [{"domain": d, "contradicted_claims": n} for d, n in sources.most_common(10)]
    out.items = out.items[:limit]
    return out
