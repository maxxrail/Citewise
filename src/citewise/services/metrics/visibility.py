"""Visibility metrics, computed in SQL over `answers` x `mentions`.

The SQL is portable (Postgres in prod, SQLite in tests): CTEs, CASE, aggregates, and no
GROUP BY on constants. Python only does the arithmetic that is awkward in SQL: bootstrap
confidence intervals and the composite Visibility Score.

Definitions (per slice of answers; a slice is all answers, or one engine/topic/intent/run):
- mention_rate: answers that mention the brand / answers.
- share_of_voice: brand mentions / mentions of all tracked entities (brand + competitors).
  Each (answer, entity) counts once.
- avg_rank: mean rank of the brand where it is ranked. top3_rate: answers where the brand
  is ranked 1-3 / answers that contain a ranking.
- recommendation_rate: answers that recommend the brand / answers.
- net_sentiment: (positive - negative brand mentions) / brand mentions, in -1..1.
- position_score: over answers containing a ranking, max(0, (6 - rank) / 5), with 0 when
  the brand isn't in the ranking. Rank 1 -> 1.0, rank 5 -> 0.2.
- visibility_score (0-100): see VISIBILITY_WEIGHTS and DECISIONS.md.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

SliceBy = Literal["engine", "topic", "intent", "run"]
_SLICE_SQL: dict[str | None, str] = {
    None: "CAST('all' AS VARCHAR(8))",
    "engine": "b.engine",
    "topic": "b.topic_id",
    "intent": "b.intent",
    "run": "b.run_id",
}
# Weights sum to 1. Mention rate and share of voice carry most of it: being in the answer
# at all is what matters most; rank, recommendation and tone refine it. See DECISIONS.md.
VISIBILITY_WEIGHTS = {
    "mention_rate": 0.35,
    "share_of_voice": 0.25,
    "position_score": 0.15,
    "recommendation_rate": 0.15,
    "sentiment": 0.10,
}


class NoExtractionsError(LookupError):
    """The brand has no extracted answers (run `citewise extract run` first)."""


@dataclass
class Filters:
    run_id: int | None = None
    engine: str | None = None
    topic_id: int | None = None
    intent: str | None = None


@dataclass
class SliceMetrics:
    slice: str
    answers: int
    mentioned: int
    mention_rate: float
    mention_rate_ci: tuple[float, float]
    share_of_voice: float | None
    share_of_voice_ci: tuple[float, float] | None
    avg_rank: float | None
    top3_rate: float | None
    recommendation_rate: float
    net_sentiment: float | None
    avg_sentiment: float | None
    position_score: float
    visibility_score: float
    small_sample: bool  # fewer than 30 answers: read the CI, not the point estimate

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DomainShare:
    domain: str
    citations: int
    share: float


@dataclass
class CitationShare:
    total: int
    own_domain_share: float
    with_brand: list[DomainShare] = field(default_factory=list)  # answers mentioning brand
    without_brand: list[DomainShare] = field(default_factory=list)


def _base_cte(f: Filters) -> tuple[str, dict[str, Any]]:
    where = ["r.brand_id = :brand_id", "a.ok"]
    params: dict[str, Any] = {}
    for col, value, name in (
        ("a.run_id", f.run_id, "run_id"),
        ("a.engine", f.engine, "engine"),
        ("p.topic_id", f.topic_id, "topic_id"),
        ("p.intent", f.intent, "intent"),
    ):
        if value is not None:
            where.append(f"{col} = :{name}")  # column names are fixed above, never input
            params[name] = value
    sql = f"""
    base AS (
        SELECT a.id AS answer_id, a.engine AS engine, a.run_id AS run_id,
               p.topic_id AS topic_id, p.intent AS intent
        FROM answers a
        JOIN runs r ON r.id = a.run_id
        JOIN answer_extractions x ON x.answer_id = a.id
        LEFT JOIN prompts p ON p.id = a.prompt_id
        WHERE {" AND ".join(where)}
    )"""
    return sql, params


def _per_answer_cte(slice_by: SliceBy | None) -> str:
    slice_expr = _SLICE_SQL[slice_by]
    return f"""
    per_answer AS (
        SELECT b.answer_id,
               MIN({slice_expr}) AS slice,
               MAX(CASE WHEN m.is_brand THEN 1 ELSE 0 END) AS mentioned,
               SUM(CASE WHEN m.is_brand OR m.is_competitor THEN 1 ELSE 0 END) AS tracked,
               MIN(CASE WHEN m.is_brand THEN m.rank END) AS brand_rank,
               MAX(CASE WHEN m.rank IS NOT NULL THEN 1 ELSE 0 END) AS has_ranking,
               MAX(CASE WHEN m.is_brand AND m.is_recommended THEN 1 ELSE 0 END) AS recommended,
               MAX(CASE WHEN m.is_brand THEN m.sentiment END) AS sentiment,
               MAX(CASE WHEN m.is_brand AND m.sentiment_label = 'positive' THEN 1 ELSE 0 END)
                   AS pos,
               MAX(CASE WHEN m.is_brand AND m.sentiment_label = 'negative' THEN 1 ELSE 0 END)
                   AS neg
        FROM base b
        LEFT JOIN mentions m ON m.answer_id = b.answer_id
        GROUP BY b.answer_id
    )"""


AGGREGATE_SQL = """
SELECT slice,
       COUNT(*) AS answers,
       SUM(mentioned) AS mentioned,
       SUM(tracked) AS tracked,
       AVG(brand_rank) AS avg_rank,
       SUM(CASE WHEN brand_rank <= 3 THEN 1 ELSE 0 END) AS top3,
       SUM(has_ranking) AS ranked_answers,
       SUM(CASE WHEN has_ranking = 1 AND brand_rank IS NOT NULL AND brand_rank < 6
                THEN (6 - brand_rank) / 5.0 ELSE 0 END) AS position_points,
       SUM(recommended) AS recommended,
       AVG(sentiment) AS avg_sentiment,
       SUM(pos) AS pos,
       SUM(neg) AS neg
FROM per_answer
GROUP BY slice
ORDER BY slice
"""


def latest_extracted_run(session: Session, brand_id: int) -> int:
    run_id = session.execute(
        text(
            """SELECT MAX(a.run_id) FROM answers a
               JOIN runs r ON r.id = a.run_id
               JOIN answer_extractions x ON x.answer_id = a.id
               WHERE r.brand_id = :brand_id"""
        ),
        {"brand_id": brand_id},
    ).scalar()
    if run_id is None:
        raise NoExtractionsError(
            "No extracted answers for this brand: run `citewise scan run` and "
            "`citewise extract run <run_id>` first."
        )
    return int(run_id)


def _resolve_filters(
    session: Session, brand_id: int, f: Filters, slice_by: SliceBy | None
) -> Filters:
    """Default to the latest extracted run, except for a per-run (trend) breakdown."""
    if f.run_id is None and slice_by != "run":
        return Filters(latest_extracted_run(session, brand_id), f.engine, f.topic_id, f.intent)
    return f


def per_answer_rows(
    session: Session, brand_id: int, f: Filters, slice_by: SliceBy | None
) -> list[dict[str, Any]]:
    base, params = _base_cte(f)
    sql = f"WITH {base}, {_per_answer_cte(slice_by)} SELECT * FROM per_answer"
    rows = session.execute(text(sql), {"brand_id": brand_id, **params}).mappings()
    return [dict(r) for r in rows]


def bootstrap_ci(
    mentioned: np.ndarray, tracked: np.ndarray, n: int, seed: int = 0
) -> tuple[tuple[float, float], tuple[float, float] | None]:
    """95% percentile CIs for mention rate (mean of `mentioned`) and share of voice
    (sum(mentioned) / sum(tracked)), resampling answers with replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(mentioned), size=(n, len(mentioned)))
    m, t = mentioned[idx], tracked[idx]
    rate = m.mean(axis=1)
    rate_ci = (float(np.percentile(rate, 2.5)), float(np.percentile(rate, 97.5)))
    tsum = t.sum(axis=1)
    if not tracked.sum():
        return rate_ci, None
    sov = np.divide(m.sum(axis=1), tsum, out=np.zeros(n), where=tsum > 0)
    return rate_ci, (float(np.percentile(sov, 2.5)), float(np.percentile(sov, 97.5)))


def visibility_score(c: dict[str, float]) -> float:
    return round(100 * sum(w * c[k] for k, w in VISIBILITY_WEIGHTS.items()), 1)


def compute_metrics(
    session: Session,
    brand_id: int,
    filters: Filters | None = None,
    *,
    slice_by: SliceBy | None = None,
    bootstrap: int = 1000,
) -> list[SliceMetrics]:
    """One SliceMetrics per slice (or a single "all" slice)."""
    f = _resolve_filters(session, brand_id, filters or Filters(), slice_by)
    base, params = _base_cte(f)
    sql = f"WITH {base}, {_per_answer_cte(slice_by)} {AGGREGATE_SQL}"
    aggregates = session.execute(text(sql), {"brand_id": brand_id, **params}).mappings().all()
    per_slice: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for r in per_answer_rows(session, brand_id, f, slice_by):
        per_slice[str(r["slice"])].append((int(r["mentioned"]), int(r["tracked"] or 0)))

    out: list[SliceMetrics] = []
    for row in aggregates:
        key = str(row["slice"])
        answers, mentioned = int(row["answers"]), int(row["mentioned"] or 0)
        tracked, ranked = int(row["tracked"] or 0), int(row["ranked_answers"] or 0)
        pos, neg = int(row["pos"] or 0), int(row["neg"] or 0)
        arr = np.array(per_slice[key], dtype=float).reshape(-1, 2)
        rate_ci, sov_ci = bootstrap_ci(arr[:, 0], arr[:, 1], bootstrap)
        mention_rate = mentioned / answers if answers else 0.0
        sov = mentioned / tracked if tracked else None
        net = (pos - neg) / mentioned if mentioned else None
        position = float(row["position_points"] or 0) / ranked if ranked else 0.0
        rec_rate = int(row["recommended"] or 0) / answers if answers else 0.0
        components = {
            "mention_rate": mention_rate,
            "share_of_voice": sov or 0.0,
            "position_score": position,
            "recommendation_rate": rec_rate,
            # A brand that is never mentioned gets 0 here, not a "neutral" 0.5.
            "sentiment": (net + 1) / 2 if net is not None else 0.0,
        }
        out.append(
            SliceMetrics(
                slice=key,
                answers=answers,
                mentioned=mentioned,
                mention_rate=mention_rate,
                mention_rate_ci=rate_ci,
                share_of_voice=sov,
                share_of_voice_ci=sov_ci,
                avg_rank=float(row["avg_rank"]) if row["avg_rank"] is not None else None,
                top3_rate=int(row["top3"] or 0) / ranked if ranked else None,
                recommendation_rate=rec_rate,
                net_sentiment=net,
                avg_sentiment=(
                    float(row["avg_sentiment"]) if row["avg_sentiment"] is not None else None
                ),
                position_score=position,
                visibility_score=visibility_score(components),
                small_sample=answers < 30,
            )
        )
    return out


def citation_share(
    session: Session,
    brand_id: int,
    brand_domain: str,
    filters: Filters | None = None,
    top: int = 10,
) -> CitationShare:
    """Which domains engines cite when they mention the brand vs when they don't, and how
    much of the citation mass points at the brand's own site."""
    f = _resolve_filters(session, brand_id, filters or Filters(), None)
    base, params = _base_cte(f)
    sql = f"""
    WITH {base}, {_per_answer_cte(None)}
    SELECT c.domain AS domain, pa.mentioned AS mentioned, COUNT(*) AS n
    FROM citations c
    JOIN per_answer pa ON pa.answer_id = c.answer_id
    GROUP BY c.domain, pa.mentioned
    """
    rows = session.execute(text(sql), {"brand_id": brand_id, **params}).mappings().all()
    groups: dict[int, dict[str, int]] = {0: {}, 1: {}}
    for r in rows:
        groups[int(r["mentioned"])][str(r["domain"])] = int(r["n"])
    total = sum(sum(g.values()) for g in groups.values())
    own = brand_domain.lower().removeprefix("www.")
    own_n = sum(
        n for g in groups.values() for d, n in g.items() if d == own or d.endswith("." + own)
    )

    def top_of(g: dict[str, int]) -> list[DomainShare]:
        s = sum(g.values())
        ranked = sorted(g.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
        return [DomainShare(d, n, n / s) for d, n in ranked]

    return CitationShare(
        total=total,
        own_domain_share=own_n / total if total else 0.0,
        with_brand=top_of(groups[1]),
        without_brand=top_of(groups[0]),
    )


@dataclass
class CompetitorRow:
    canonical: str
    entity_type: str  # competitor | unknown (= emerging)
    answers_mentioning: int
    mention_rate: float
    avg_rank: float | None
    avg_sentiment: float
    recommendation_rate: float


def competitors(
    session: Session, brand_id: int, filters: Filters | None = None
) -> list[CompetitorRow]:
    """Every other company the engines named, tracked competitors and emerging (unknown)
    ones, grouped by normalized key so "Wrike" and "Wrike, Inc." are one row."""
    f = _resolve_filters(session, brand_id, filters or Filters(), None)
    base, params = _base_cte(f)
    sql = f"""
    WITH {base}
    SELECT m.entity_key AS entity_key,
           MIN(m.entity_type) AS entity_type,
           MIN(m.canonical) AS canonical,
           COUNT(*) AS n,
           AVG(m.rank) AS avg_rank,
           AVG(m.sentiment) AS avg_sentiment,
           SUM(CASE WHEN m.is_recommended THEN 1 ELSE 0 END) AS recommended,
           (SELECT COUNT(*) FROM base) AS answers
    FROM base b
    JOIN mentions m ON m.answer_id = b.answer_id
    WHERE m.entity_type IN ('competitor', 'unknown')
    GROUP BY m.entity_key
    ORDER BY n DESC, entity_key
    """
    rows = session.execute(text(sql), {"brand_id": brand_id, **params}).mappings().all()
    return [
        CompetitorRow(
            canonical=str(r["canonical"]),
            entity_type=str(r["entity_type"]),
            answers_mentioning=int(r["n"]),
            mention_rate=int(r["n"]) / int(r["answers"]) if r["answers"] else 0.0,
            avg_rank=float(r["avg_rank"]) if r["avg_rank"] is not None else None,
            avg_sentiment=float(r["avg_sentiment"] or 0.0),
            recommendation_rate=int(r["recommended"] or 0) / int(r["n"]),
        )
        for r in rows
    ]
