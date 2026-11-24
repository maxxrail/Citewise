"""Regression tests for the Module 4 code review (H1-H3, M2-M4, M6 and Lows)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from citewise.core.config import REPO_ROOT, Settings
from citewise.models.kb import AnswerFactcheck, Claim
from citewise.providers import get_chat_provider, get_embedder
from citewise.schemas.factcheck import Verdict
from citewise.services.extract.runner import extract_run
from citewise.services.factcheck.checks import (
    CitationRef,
    claim_matches_span,
    contains,
    markers_after,
    trace_source,
    validate_verdict,
)
from citewise.services.factcheck.runner import _products, factcheck_run
from citewise.services.kb.fetch import FetchError, check_public, fetch_text
from citewise.services.kb.ingest import IngestError, chunk_text, collect_docs, ingest, parse_file
from citewise.services.kb.retrieve import Hit
from citewise.services.panel.common import resolve_brand

KB = REPO_ROOT / "data" / "kb"
PRICING = Hit(
    "offerings#0",
    "Taskflow plans > Plans and pricing\nThe Pro plan costs $12 per user per month billed "
    "annually. The Enterprise plan costs $25 per user per month.",
    1.0,
)


def _v(verdict: str, quote: str | None) -> Verdict:
    return Verdict(
        verdict=verdict, evidence_chunk_ids=["offerings#0"], evidence_quote=quote, rationale="r"
    )  # type: ignore[arg-type]


# ---------------------------------------------------------------- H1


def test_products_come_only_from_this_brands_runs(
    scanned_factory: sessionmaker[Session], settings: Settings
) -> None:
    from citewise.models.collect import Answer
    from citewise.models.extract import Mention

    chat = get_chat_provider(settings=settings, session_factory=scanned_factory)
    for run_id in (1, 2):  # run 1 = Stridelab, run 2 = Taskflow, in one database
        extract_run(scanned_factory, run_id, chat, settings)
    with scanned_factory() as s:
        task_answer = s.scalar(select(Answer.id).where(Answer.run_id == 2).limit(1))
        s.add(
            Mention(
                answer_id=task_answer,
                entity="Taskflow Boards",
                canonical="Taskflow",
                entity_key="taskflow-boards",
                entity_type="brand",
                is_brand=True,
                is_competitor=False,
                snippet="",
                source="a",
            )
        )
        s.commit()
        stride = _products(s, resolve_brand(s, "Stridelab"))
        task = _products(s, resolve_brand(s, "Taskflow"))
    assert "Taskflow Boards" in task  # its own product name is offered to its extractor
    assert all("taskflow" not in p.casefold() for p in stride)  # and never to another brand's


# ---------------------------------------------------------------- H2


@pytest.mark.parametrize(
    "span",
    [
        "Taskflow Pro costs $15 per user per month on annual billing",  # changed number
        "Taskflow Pro costs $25 per user per month on annual billing",
        "Taskflow Pro does not cost $12 per user per month on annual billing",  # added negation
    ],
)
def test_fuzzy_span_check_never_accepts_a_changed_number_or_negation(span: str) -> None:
    answer = "Taskflow Pro costs $12 per user per month on annual billing [2]."
    assert not contains(answer, span, fuzzy=92)
    assert contains(answer, "Taskflow Pro costs $12 per user per month on anual billing", fuzzy=92)


# ---------------------------------------------------------------- H3


@pytest.mark.parametrize(
    ("claim", "quote", "reason"),
    [
        ("Taskflow Pro costs $25", "Pro", "quote_too_short"),  # one word
        # The heading line locates the passage; it isn't evidence.
        (
            "Taskflow plans and pricing changed",
            "Taskflow plans > Plans and pricing",
            "quote_not_in_evidence",
        ),
        # Stitched across sentences: Pro's sentence + Enterprise's price.
        (
            "Taskflow Pro costs $25 per user",
            "The Pro plan costs ... $25 per user",
            "quote_not_in_evidence",
        ),
    ],
)
def test_trivial_or_stitched_quotes_cannot_back_a_contradiction(
    claim: str, quote: str, reason: str
) -> None:
    got = validate_verdict(
        _v("contradicted", quote), [PRICING], claim=claim, brand_terms={"taskflow"}
    )
    assert got.verdict == "not_in_kb" and got.details["reason"] == reason, got


def test_a_real_quote_still_backs_a_contradiction() -> None:
    got = validate_verdict(
        _v("contradicted", "The Pro plan costs $12 per user per month billed annually."),
        [PRICING],
        claim="Taskflow Pro costs $25 per user per month",
        brand_terms={"taskflow"},
    )
    assert got.verdict == "contradicted" and got.chunk_keys == ["offerings#0"]
    assert contains(PRICING.text, "The Pro plan costs $12 ... billed annually")  # in order, near


# ---------------------------------------------------------------- M6


def test_claims_must_say_what_their_span_says() -> None:
    span = "Taskflow Pro costs $12 per user per month"
    assert claim_matches_span("Taskflow Pro costs $12 per user per month", span, {"taskflow"})
    assert not claim_matches_span("Taskflow Pro costs $25 per user per month", span, {"taskflow"})
    assert not claim_matches_span("Taskflow has no Jira importer", span, {"taskflow"})
    assert not claim_matches_span("Taskflow Pro does not cost $12 per user", span, {"taskflow"})


# ---------------------------------------------------------------- M2

CITES = [
    CitationRef(
        1, 1, "https://www.taskflow.example/", "Taskflow | Project management", "taskflow.example"
    ),
    CitationRef(2, 2, "https://www.g2.com/pm", "Best PM software | G2", "g2.com"),
    CitationRef(
        3,
        3,
        "https://www.migration-guides.example/jira",
        "Jira import guide",
        "migration-guides.example",
    ),
]


def test_markers_after_the_full_stop_and_after_abbreviations_count() -> None:
    span = "Taskflow Pro costs $25 per user per month"
    assert markers_after(f"{span}. [2] Asana is fine [1].", span) == [2]
    assert markers_after(f"{span}, e.g. for small teams [3]. Next [1].", span) == [3]
    assert markers_after(f"{span} for teams. Asana is fine [1].", span) == []


def test_the_brands_own_site_is_not_blamed_for_sharing_its_name() -> None:
    claim = "Taskflow has no Jira import"
    t = trace_source(claim, claim, f"{claim}.", CITES[:2], brand_terms={"taskflow"})
    assert t is None  # neither the homepage nor G2's title is about Jira imports
    t2 = trace_source(claim, claim, f"{claim}.", CITES, brand_terms={"taskflow"})
    assert t2 is not None and t2.citation_id == 3 and t2.reason == "title_match"


def test_perplexity_marker_numbers_map_by_url_not_position() -> None:
    claim = "Taskflow has no Jira import"
    # Perplexity said [web:7]; after dedup that page is our position 3.
    t = trace_source(
        claim,
        claim,
        f"{claim} [web:7].",
        CITES,
        brand_terms={"taskflow"},
        marker_urls={7: "https://www.migration-guides.example/jira"},
    )
    assert t is not None and (t.citation_id, t.reason) == (3, "inline_marker")


# ---------------------------------------------------------------- M3


def _checked(factory: sessionmaker[Session], settings: Settings) -> tuple[object, object]:
    chat = get_chat_provider(settings=settings, session_factory=factory)
    embedder = get_embedder(settings=settings, session_factory=factory)
    with factory() as s:
        ingest(
            s,
            resolve_brand(s, "Taskflow"),
            collect_docs(str(KB / "taskflow"), None, live=False),
            embedder,
        )
        s.commit()
    extract_run(factory, 2, chat, settings)
    factcheck_run(factory, 2, chat, embedder, settings)
    return chat, embedder


def test_a_kb_change_makes_verdicts_stale(
    scanned_factory: sessionmaker[Session], settings: Settings
) -> None:
    chat, embedder = _checked(scanned_factory, settings)
    assert factcheck_run(scanned_factory, 2, chat, embedder, settings).checked == 0  # type: ignore[arg-type]
    docs = collect_docs(str(KB / "taskflow"), None, live=False)
    docs[0].text += "\n\nTaskflow now has 70 employees."
    with scanned_factory() as s:
        ingest(s, resolve_brand(s, "Taskflow"), docs, embedder)  # type: ignore[arg-type]
        s.commit()
    again = factcheck_run(scanned_factory, 2, chat, embedder, settings)  # type: ignore[arg-type]
    assert again.checked == again.answers > 0  # every verdict re-checked against the new KB


def test_a_failed_recheck_keeps_the_previous_claims(
    scanned_factory: sessionmaker[Session], settings: Settings
) -> None:
    _, embedder = _checked(scanned_factory, settings)
    with scanned_factory() as s:
        before = len(list(s.scalars(select(Claim.id))))

    class Broken:
        name = "broken"

        def complete(self, *a: object, **k: object) -> object:
            raise ConnectionError("down")

    r = factcheck_run(scanned_factory, 2, Broken(), embedder, settings, force=True)  # type: ignore[arg-type]
    assert r.llm_failed == r.answers
    with scanned_factory() as s:
        assert len(list(s.scalars(select(Claim.id)))) == before  # nothing thrown away
        assert not any(s.scalars(select(AnswerFactcheck.ok)))  # all marked for retry


def test_an_embedding_mismatch_fails_before_any_paid_call(
    scanned_factory: sessionmaker[Session], settings: Settings
) -> None:
    _checked(scanned_factory, settings)

    class Other:
        ident = "openai:text-embedding-3-small"

        def embed(self, texts: list[str], *, purpose: str = "") -> list[list[float]]:
            raise AssertionError("not called")

    class Counting:
        name, calls = "c", 0

        def complete(self, *a: object, **k: object) -> object:
            Counting.calls += 1
            raise AssertionError("no LLM call may happen")

    with pytest.raises(ValueError, match="re-run `citewise kb ingest`"):
        factcheck_run(scanned_factory, 2, Counting(), Other(), settings, force=True)  # type: ignore[arg-type]
    assert Counting.calls == 0


def test_fetch_sources_is_live_only(
    scanned_factory: sessionmaker[Session], settings: Settings
) -> None:
    chat, embedder = _checked(scanned_factory, settings)
    with pytest.raises(ValueError, match="live"):
        factcheck_run(scanned_factory, 2, chat, embedder, settings, fetch_sources=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------- M4 (safe fetch)


def _resolver(mapping: dict[str, str]):  # type: ignore[no-untyped-def]
    return lambda host: [mapping[host]]


@pytest.mark.parametrize(
    "addr", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "::1", "fd00::1"]
)
def test_non_public_addresses_are_refused(addr: str) -> None:
    with pytest.raises(FetchError, match="non-public"):
        check_public("https://evil.example/x", _resolver({"evil.example": addr}))


def test_redirects_are_revalidated_and_bounded() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.host == "public.example":
            return httpx.Response(302, headers={"location": "http://internal.example/admin"})
        return httpx.Response(200, text="secret", headers={"content-type": "text/html"})

    resolve = _resolver({"public.example": "93.184.216.34", "internal.example": "10.0.0.7"})
    with pytest.raises(FetchError, match="non-public"):
        fetch_text(
            "https://public.example/page", resolve=resolve, transport=httpx.MockTransport(handler)
        )

    def loop(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(302, headers={"location": "/again"})

    with pytest.raises(FetchError, match="redirects"):
        fetch_text(
            "https://public.example/",
            resolve=_resolver({"public.example": "93.184.216.34"}),
            transport=httpx.MockTransport(loop),
        )


@pytest.mark.parametrize(
    ("robots", "ctype", "body", "error"),
    [
        (httpx.Response(403), "text/html", "x", "robots"),  # 403 robots.txt = disallow
        (httpx.Response(200, text="User-agent: *\nDisallow: /"), "text/html", "x", "robots"),
        (httpx.Response(404), "application/pdf", "x", "content type"),
        (httpx.Response(404), "text/html", "x" * 2_100_000, "larger than"),
    ],
)
def test_robots_content_type_and_size_limits(
    robots: httpx.Response, ctype: str, body: str, error: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return robots
        return httpx.Response(200, text=body, headers={"content-type": ctype})

    with pytest.raises(FetchError, match=error):
        fetch_text(
            "https://public.example/p",
            resolve=_resolver({"public.example": "93.184.216.34"}),
            transport=httpx.MockTransport(handler),
        )


def test_a_public_page_is_fetched() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(
            200, text="<p>Pro costs $12</p>", headers={"content-type": "text/html"}
        )

    url, body = fetch_text(
        "https://public.example/p",
        resolve=_resolver({"public.example": "93.184.216.34"}),
        transport=httpx.MockTransport(handler),
    )
    assert url == "https://public.example/p" and "Pro costs $12" in body


# ---------------------------------------------------------------- Lows


def test_overlap_must_be_under_half_the_chunk() -> None:
    with pytest.raises(ValueError, match="KB_CHUNK_OVERLAP"):
        Settings(kb_chunk_tokens=100, kb_chunk_overlap=60, _env_file=None)  # type: ignore[call-arg]


def test_duplicate_slugs_and_bom_and_bad_front_matter(tmp_path: Path) -> None:
    (tmp_path / "pricing.md").write_text("---\ncategory: offerings\n---\nA.", encoding="utf-8")
    (tmp_path / "pricing.txt").write_text("---\ncategory: offerings\n---\nB.", encoding="utf-8")
    with pytest.raises(IngestError, match="both map"):
        collect_docs(str(tmp_path), None, live=False)
    bom = tmp_path / "bom.md"
    bom.write_text("---\ncategory: company\n---\n# T\n\nText.", encoding="utf-8-sig")
    assert parse_file(bom).category == "company"
    bad = tmp_path / "bad.md"
    bad.write_text("---\n- just a list\n---\nText.", encoding="utf-8")
    with pytest.raises(IngestError, match="key: value"):
        parse_file(bad)


def test_every_golden_chunk_key_exists() -> None:
    keys: dict[str, set[str]] = {}
    for brand in ("stridelab", "taskflow", "linear"):
        keys[brand] = {
            f"{d.slug}#{c.ord}"
            for d in collect_docs(str(KB / brand), None, live=False)
            for c in chunk_text(d.text, d.title)
        }
    rows = [
        json.loads(line)
        for line in (REPO_ROOT / "evals/datasets/golden/kb_qa.jsonl")
        .read_text("utf-8")
        .splitlines()
    ]
    missing = [
        (r["id"], k)
        for r in rows
        for k in r["gold_chunk_keys"]
        if k not in keys[r["brand"].lower()]
    ]
    assert missing == []


def test_held_out_claims_evidence_hints_are_in_the_kb() -> None:
    text = {
        b: " ".join(d.text for d in collect_docs(str(KB / b), None, live=False))
        for b in ("stridelab", "taskflow", "linear")
    }
    rows = [
        json.loads(line)
        for line in (REPO_ROOT / "evals/datasets/golden/factcheck_claims.jsonl")
        .read_text("utf-8")
        .splitlines()
    ]
    assert len(rows) >= 40
    for r in rows:
        if r.get("evidence_hint"):
            assert contains(text[r["brand"].lower()], r["evidence_hint"]), r["id"]


def test_a_pronoun_quote_from_the_live_model_still_backs_a_contradiction() -> None:
    """Measured on the held-out set: a word-overlap rule threw this correct verdict away."""
    hit = Hit(
        "offerings#0",
        "Stridelab shoes and prices > Current models\nThe Stridelab Tempo is a lightweight "
        "daily trainer. It costs $140, weighs 240 grams (men's size 9).",
        1.0,
    )
    v = Verdict(
        verdict="contradicted",
        evidence_chunk_ids=["offerings#0"],
        evidence_quote="It costs $140, weighs 240 grams (men's size 9).",
        rationale="r",
    )
    got = validate_verdict(
        v, [hit], claim="The Stridelab Tempo sells for around $120", brand_terms={"stridelab"}
    )
    assert got.verdict == "contradicted"
