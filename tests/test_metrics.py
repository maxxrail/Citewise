"""Metrics SQL against a tiny hand-built fixture. Every expected number is worked out by
hand in the comments, so a wrong JOIN or a double count shows up as a wrong number."""

from __future__ import annotations

import os
from collections.abc import Iterator

import numpy as np
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from citewise.core.db import Base
from citewise.models.brand import Brand
from citewise.models.collect import Answer, Citation, Run
from citewise.models.extract import AnswerExtraction, Mention
from citewise.models.panel import Prompt, Topic
from citewise.services.metrics.visibility import (
    VISIBILITY_WEIGHTS,
    Filters,
    NoExtractionsError,
    bootstrap_ci,
    citation_share,
    competitors,
    compute_metrics,
)

# (answer key, engine, run, topic, intent, extracted, ok)
ANSWERS = [
    ("A0", "openai", "old", "T1", "commercial", True, True),  # older run: trends only
    ("A1", "openai", "new", "T1", "commercial", True, True),
    ("A2", "openai", "new", "T1", "comparison", True, True),
    ("A3", "gemini", "new", "T2", "commercial", True, True),
    ("A4", "gemini", "new", "T2", "informational", True, True),
    ("A5", "gemini", "new", "T2", "commercial", False, True),  # not extracted: excluded
    ("A6", "gemini", "new", "T2", "commercial", True, False),  # failed answer: excluded
]
# (answer, canonical, type, rank, sentiment, label, recommended)
MENTIONS = [
    ("A0", "Acme", "brand", 1, 0.9, "positive", True),
    ("A1", "Acme", "brand", 1, 0.8, "positive", True),
    ("A1", "Beta", "competitor", 2, 0.0, "neutral", False),
    ("A2", "Beta", "competitor", 1, 0.7, "positive", True),
    ("A2", "Gamma", "competitor", 2, 0.0, "neutral", False),
    ("A3", "Acme", "brand", None, -0.6, "negative", False),
    ("A3", "Delta", "unknown", None, 0.0, "neutral", False),
    ("A5", "Acme", "brand", 1, 0.9, "positive", True),  # would inflate if A5 counted
    ("A6", "Acme", "brand", 1, 0.9, "positive", True),
]
CITATIONS = [
    ("A1", "acme.example"),
    ("A1", "g2.com"),
    ("A2", "g2.com"),
    ("A2", "reddit.com"),
    ("A3", "reddit.com"),
    ("A6", "acme.example"),  # failed answer: excluded
]


@pytest.fixture(params=["sqlite", "postgres"])
def metrics_db(request: pytest.FixtureRequest, db: Session) -> Iterator[Session]:
    """The metrics SQL must hold on both backends: SQLite here, Postgres when
    CITEWISE_TEST_PG_URL points at a throwaway database (CI sets it). The Postgres run
    drops and recreates every table, so never point it at a database you care about."""
    if request.param == "sqlite":
        yield db
        return
    url = os.environ.get("CITEWISE_TEST_PG_URL")
    if not url:
        pytest.skip("CITEWISE_TEST_PG_URL not set")
    engine = create_engine(url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine, expire_on_commit=False)() as session:
        yield session
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def fixture_db(metrics_db: Session) -> tuple[Session, int, dict[str, int]]:
    s = metrics_db
    brand = Brand(name="Acme", domain="acme.example", category="widgets", aliases=[],
                  competitors=["Beta", "Gamma"], personas=[])  # fmt: skip
    s.add(brand)
    s.flush()
    topics = {k: Topic(brand_id=brand.id, label=k, algorithm="kmeans") for k in ("T1", "T2")}
    s.add_all(topics.values())
    s.flush()
    runs = {}
    for key in ("old", "new"):  # created in this order: "new" has the higher id
        runs[key] = Run(brand_id=brand.id, status="completed", engines=["openai", "gemini"],
                        models={}, prompt_ids=[], max_cost_usd=1, mode="mock")  # fmt: skip
        s.add(runs[key])
        s.flush()
    ids: dict[str, int] = {}
    for key, engine, run, topic, intent, extracted, ok in ANSWERS:
        p = Prompt(brand_id=brand.id, text=key, text_hash=key, source="generated",
                   intent=intent, topic_id=topics[topic].id)  # fmt: skip
        s.add(p)
        s.flush()
        a = Answer(run_id=runs[run].id, prompt_id=p.id, engine=engine, model="m", ok=ok,
                   text=key, cache_key=key)  # fmt: skip
        s.add(a)
        s.flush()
        ids[key] = a.id
        if extracted:
            s.add(AnswerExtraction(answer_id=a.id, version="t", llm_ok=True, stage_a=[],
                                   stage_b=[], disagreements=[]))  # fmt: skip
    for key, canonical, typ, rank, sent, label, rec in MENTIONS:
        s.add(Mention(answer_id=ids[key], entity=canonical, canonical=canonical,
                      entity_key=canonical.lower(), entity_type=typ, is_brand=typ == "brand",
                      is_competitor=typ == "competitor", rank=rank, sentiment=sent,
                      sentiment_label=label, is_recommended=rec, snippet="",
                      source="both"))  # fmt: skip
    for i, (key, domain) in enumerate(CITATIONS):
        s.add(Citation(answer_id=ids[key], position=i, url=f"https://{domain}/",
                       normalized_url=f"https://{domain}", domain=domain))  # fmt: skip
    s.commit()
    return (
        s,
        brand.id,
        {k: r.id for k, r in runs.items()} | {f"topic:{k}": t.id for k, t in topics.items()},
    )


def test_overall_metrics_default_to_the_latest_run(
    fixture_db: tuple[Session, int, dict[str, int]],
) -> None:
    s, brand_id, _ = fixture_db
    [m] = compute_metrics(s, brand_id, bootstrap=200)
    # Latest run, extracted, ok answers: A1, A2, A3, A4.
    assert (m.slice, m.answers, m.mentioned) == ("all", 4, 2)
    assert m.mention_rate == pytest.approx(0.5)  # A1, A3
    # Tracked mentions: A1 2 + A2 2 + A3 1 (Delta is unknown, not tracked) = 5; Acme has 2.
    assert m.share_of_voice == pytest.approx(2 / 5)
    assert m.avg_rank == pytest.approx(1.0)  # ranked only in A1
    assert m.top3_rate == pytest.approx(1 / 2)  # A1 and A2 contain a ranking; Acme top-3 in A1
    assert m.position_score == pytest.approx((1.0 + 0.0) / 2)  # rank 1 in A1, absent in A2
    assert m.recommendation_rate == pytest.approx(1 / 4)  # A1 only
    assert m.net_sentiment == pytest.approx((1 - 1) / 2)  # one positive, one negative
    assert m.avg_sentiment == pytest.approx((0.8 - 0.6) / 2)
    expected = 100 * (0.35 * 0.5 + 0.25 * 0.4 + 0.15 * 0.5 + 0.15 * 0.25 + 0.10 * 0.5)
    assert m.visibility_score == pytest.approx(expected, abs=0.051)  # 43.75
    assert m.small_sample is True
    lo, hi = m.mention_rate_ci
    assert 0 <= lo <= m.mention_rate <= hi <= 1
    assert sum(VISIBILITY_WEIGHTS.values()) == pytest.approx(1.0)


def test_metrics_sliced_by_engine_topic_intent_and_run(
    fixture_db: tuple[Session, int, dict[str, int]],
) -> None:
    s, brand_id, ids = fixture_db
    by_engine = {m.slice: m for m in compute_metrics(s, brand_id, slice_by="engine", bootstrap=200)}
    assert by_engine["openai"].answers == 2 and by_engine["openai"].mention_rate == 0.5
    assert by_engine["openai"].share_of_voice == pytest.approx(1 / 4)  # A1: 1 of 2, A2: 0 of 2
    assert by_engine["gemini"].share_of_voice == pytest.approx(1.0)  # A3: Acme is the only one
    assert by_engine["gemini"].recommendation_rate == 0.0

    by_topic = {m.slice: m for m in compute_metrics(s, brand_id, slice_by="topic", bootstrap=200)}
    assert by_topic[str(ids["topic:T1"])].mention_rate == pytest.approx(1 / 2)  # A1 of A1, A2
    by_intent = {m.slice: m for m in compute_metrics(s, brand_id, slice_by="intent", bootstrap=200)}
    assert by_intent["commercial"].answers == 2 and by_intent["commercial"].mentioned == 2
    assert by_intent["informational"].mentioned == 0
    assert by_intent["informational"].visibility_score == 0.0  # never mentioned: 0, not 5

    # The run breakdown spans every run (trends), unless a run is given.
    by_run = {m.slice: m for m in compute_metrics(s, brand_id, slice_by="run", bootstrap=200)}
    assert by_run[str(ids["old"])].mention_rate == 1.0
    assert by_run[str(ids["new"])].mention_rate == 0.5
    [only] = compute_metrics(s, brand_id, Filters(run_id=ids["old"]), bootstrap=200)
    # 1 of 1: a percentile bootstrap says (1.0, 1.0); Wilson keeps the uncertainty visible.
    assert only.answers == 1 and only.mention_rate_ci == pytest.approx((0.2065, 1.0), abs=1e-3)

    [gem] = compute_metrics(s, brand_id, Filters(engine="gemini"), bootstrap=200)
    assert (gem.answers, gem.mentioned) == (2, 1)


def test_citation_share_with_and_without_the_brand(
    fixture_db: tuple[Session, int, dict[str, int]],
) -> None:
    s, brand_id, _ = fixture_db
    cs = citation_share(s, brand_id, "www.acme.example")
    assert cs.total == 5  # A6's citation is excluded with its failed answer
    assert cs.own_domain_share == pytest.approx(1 / 5)
    with_b = {d.domain: d.share for d in cs.with_brand}  # A1 + A3: acme, g2, reddit
    assert with_b == pytest.approx({"acme.example": 1 / 3, "g2.com": 1 / 3, "reddit.com": 1 / 3})
    without = {d.domain: d.share for d in cs.without_brand}  # A2: g2, reddit
    assert without == pytest.approx({"g2.com": 0.5, "reddit.com": 0.5})


def test_competitors_include_emerging_unknowns(
    fixture_db: tuple[Session, int, dict[str, int]],
) -> None:
    s, brand_id, _ = fixture_db
    rows = competitors(s, brand_id)
    assert [(r.canonical, r.entity_type, r.answers_mentioning) for r in rows] == [
        ("Beta", "competitor", 2),
        ("Delta", "unknown", 1),
        ("Gamma", "competitor", 1),
    ]
    beta = rows[0]
    assert beta.mention_rate == pytest.approx(2 / 4)
    assert beta.avg_rank == pytest.approx(1.5)
    assert beta.recommendation_rate == pytest.approx(1 / 2)


def test_no_extractions_is_a_clear_error(metrics_db: Session) -> None:
    db = metrics_db
    brand = Brand(name="Empty", domain="e.example", category="x", aliases=[], competitors=[],
                  personas=[])  # fmt: skip
    db.add(brand)
    db.commit()
    with pytest.raises(NoExtractionsError, match="extract run"):
        compute_metrics(db, brand.id)


def test_bootstrap_ci_is_deterministic_and_brackets_the_estimate() -> None:
    mentioned = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0], dtype=float)
    tracked = np.array([2, 1, 3, 1, 2, 0, 2, 1, 2, 2], dtype=float)
    a = bootstrap_ci(mentioned, tracked, 2000)
    assert a == bootstrap_ci(mentioned, tracked, 2000)
    (rlo, rhi), sov = a
    assert rlo < 0.5 < rhi and sov is not None and sov[0] < 5 / 16 < sov[1]
    assert bootstrap_ci(mentioned, np.zeros(10), 100)[1] is None  # no tracked mentions


def test_cluster_bootstrap_widens_the_interval_for_correlated_answers() -> None:
    # 20 prompts x 4 engines; engines agree within a prompt (fully correlated answers).
    per_prompt = np.array([1, 0] * 10, dtype=float)
    mentioned = np.repeat(per_prompt, 4)
    tracked = np.full_like(mentioned, 2.0)
    clusters = np.repeat(np.arange(20), 4)
    (alo, ahi), _ = bootstrap_ci(mentioned, tracked, 4000)  # pretends 80 independent answers
    (clo, chi), _ = bootstrap_ci(mentioned, tracked, 4000, clusters=clusters)
    assert clo < 0.5 < chi
    assert (chi - clo) > 1.5 * (ahi - alo)  # ~2x wider: 20 effective samples, not 80
    # One answer per cluster is the plain answer-level bootstrap.
    single = bootstrap_ci(per_prompt, np.full(20, 2.0), 500, clusters=np.arange(20))
    assert single == bootstrap_ci(per_prompt, np.full(20, 2.0), 500)


def test_wilson_interval_at_the_edges() -> None:
    from citewise.services.metrics.visibility import wilson_interval

    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.35  # 0 of 10 is not "0% for sure"
    assert wilson_interval(5, 0) == (0.0, 1.0)


def test_partial_and_degraded_extractions_are_reported_not_hidden(
    fixture_db: tuple[Session, int, dict[str, int]],
) -> None:
    from sqlalchemy import delete, update

    s, brand_id, _ = fixture_db
    a3, a4 = (s.scalar(select(Answer.id).where(Answer.text == k)) for k in ("A3", "A4"))
    s.execute(update(AnswerExtraction).where(AnswerExtraction.answer_id == a3).values(llm_ok=False))
    s.execute(delete(AnswerExtraction).where(AnswerExtraction.answer_id == a4))  # budget-stopped
    s.commit()
    [m] = compute_metrics(s, brand_id, bootstrap=200)
    # Ok answers in the latest run: A1-A5 (A6 failed) = 5; extracted A1, A2, A3 = 3.
    assert m.answers == 3 and m.coverage == pytest.approx(3 / 5)
    assert m.llm_ok_share == pytest.approx(2 / 3)  # A3 is Stage A only
    # Recommendation and sentiment use only A1, A2 (Stage B ran): A1 recommends, positive.
    assert m.recommendation_rate == pytest.approx(1 / 2)
    assert m.net_sentiment == pytest.approx(1.0)
    assert m.mention_rate == pytest.approx(2 / 3)  # detection still uses all extracted
