"""`citewise factcheck run <run_id>`: check what engines claimed about the brand.

For every ok answer in the run that mentions the brand (Module 3 mentions), extract
claims, retrieve KB evidence (hybrid, optional re-rank), verify, and trace contradicted
claims to the cited page that probably caused them.

Same operating rules as `extract run`:
- Idempotent by FACTCHECK_VERSION. Writing an answer's claims replaces them in one
  transaction. An answer whose LLM step failed is marked ok=False and retried next time.
- LLM calls happen before DB writes. When the spend cap refuses a call, the answer is
  left unchecked rather than half-stored.
- Threads in live mode, inline in mock mode (see extract/runner.py for why).
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import httpx
from sqlalchemy import delete, exists, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from citewise.core.config import Settings
from citewise.models.brand import Brand
from citewise.models.collect import Answer, Citation, Run
from citewise.models.extract import AnswerExtraction, Mention
from citewise.models.kb import AnswerFactcheck, Claim, KBDoc
from citewise.providers.base import ChatProvider, EmbeddingProvider
from citewise.providers.budget import BudgetExceededError
from citewise.services.factcheck.checks import (
    Checked,
    CitationRef,
    Trace,
    brand_terms_for,
    extract_claims,
    trace_source,
    verify_claim,
)
from citewise.services.kb.fetch import FetchError, fetch_text
from citewise.services.kb.ingest import html_to_text
from citewise.services.kb.retrieve import KBIndex
from citewise.services.panel.common import resolve_brand

log = logging.getLogger(__name__)

# Bump the leading f<N> when check rules change; the KB fingerprint is appended per run.
FACTCHECK_VERSION = "f2+claims.v1+verify.v1"
MAX_SOURCE_FETCHES = 5  # cited pages read per answer (live, --fetch-sources)


@dataclass
class FactcheckReport:
    run_id: int
    answers: int = 0  # ok answers that mention the brand
    skipped: int = 0
    checked: int = 0
    llm_failed: int = 0
    budget_stopped: int = 0
    claims: int = 0
    verdicts: Counter[str] = field(default_factory=Counter)
    traced: int = 0  # contradicted claims with a likely source
    dropped: Counter[str] = field(default_factory=Counter)
    downgraded: int = 0  # verdicts demoted to not_in_kb for lack of a real quote


@dataclass
class _Item:
    answer_id: int
    text: str
    citations: list[CitationRef]
    # Perplexity's inline [web:N] numbers are its result ids, not our deduplicated
    # citation positions: N -> URL, from the stored payload.
    marker_urls: dict[int, str] | None = None


@dataclass
class _Result:
    answer_id: int
    claims: list[dict[str, Any]]
    ok: bool
    refused: bool = False
    dropped: list[dict[str, Any]] = field(default_factory=list)


def _marker_urls(raw: dict[str, Any] | None) -> dict[int, str] | None:
    results = [
        r
        for item in (raw or {}).get("output") or []
        if isinstance(item, dict) and item.get("type") == "search_results"
        for r in item.get("results") or []
    ]
    urls = {r["id"]: r["url"] for r in results if isinstance(r.get("id"), int) and r.get("url")}
    return urls or None


def kb_fingerprint(session: Session, brand_id: int) -> str:
    """Changes whenever the brand's KB does: a verdict is only valid for the KB it used."""
    hashes = sorted(session.scalars(select(KBDoc.content_hash).where(KBDoc.brand_id == brand_id)))
    return hashlib.sha256("\n".join(hashes).encode()).hexdigest()[:12]


def _items(session: Session, run_id: int) -> list[_Item]:
    mentioned = exists().where(Mention.answer_id == Answer.id, Mention.is_brand)
    answers = list(
        session.execute(
            select(Answer.id, Answer.text, Answer.engine, Answer.raw_json).where(
                Answer.run_id == run_id, Answer.ok, mentioned
            )
        )
    )
    cites: dict[int, list[CitationRef]] = {}
    for c in session.scalars(
        select(Citation)
        .where(Citation.answer_id.in_([a for a, *_ in answers]))
        .order_by(Citation.position)
    ):
        cites.setdefault(c.answer_id, []).append(
            CitationRef(c.id, c.position, c.url, c.title, c.domain)
        )
    return [
        _Item(aid, text, cites.get(aid, []), _marker_urls(raw) if engine == "perplexity" else None)
        for aid, text, engine, raw in answers
    ]


def factcheck_run(
    factory: sessionmaker[Session],
    run_id: int,
    chat: ChatProvider,
    embedder: EmbeddingProvider,
    settings: Settings,
    *,
    force: bool = False,
    rerank: bool = False,
    fetch_sources: bool = False,
    workers: int | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> FactcheckReport:
    """`fetch_sources` (live mode only): read the pages an answer cited when one of its
    claims is contradicted, so tracing can use page text, not just titles."""
    if fetch_sources and settings.is_mock:
        raise ValueError("--fetch-sources needs CITEWISE_MODE=live (mock mode never fetches)")
    with factory() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise ValueError(f"No run {run_id}")
        brand = session.get(Brand, run.brand_id)
        assert brand is not None
        extracted = session.scalar(
            select(AnswerExtraction.answer_id)
            .join(Answer, Answer.id == AnswerExtraction.answer_id)
            .where(Answer.run_id == run_id)
            .limit(1)
        )
        if extracted is None:
            raise ValueError(
                f"Run {run_id} has no extractions: run `citewise extract run {run_id}` first"
            )
        index = KBIndex.load(session, brand.id)
        if not len(index):
            raise ValueError(
                f"{brand.name} has no knowledge base: "
                f"run `citewise kb ingest {brand.name} <path>` first"
            )
        if index.spaces != {embedder.ident}:  # fail before the first paid call, not after
            raise ValueError(
                f"{brand.name}'s KB is embedded with {sorted(map(str, index.spaces))}, not "
                f"{embedder.ident}: re-run `citewise kb ingest`"
            )
        version = f"{FACTCHECK_VERSION}+kb:{kb_fingerprint(session, brand.id)}"
        items = _items(session, run_id)
        done = set(
            session.scalars(
                select(AnswerFactcheck.answer_id)
                .join(Answer, Answer.id == AnswerFactcheck.answer_id)
                .where(Answer.run_id == run_id, AnswerFactcheck.version == version,
                       AnswerFactcheck.ok)
            )
        )  # fmt: skip
        name, brand_id = brand.name, brand.id
        aliases = [*brand.aliases, *_products(session, brand)]
    report = FactcheckReport(run_id=run_id, answers=len(items))
    todo = [it for it in items if force or it.answer_id not in done]
    report.skipped = len(items) - len(todo)
    model = settings.model_judge
    bterms = brand_terms_for(name, aliases)
    pages: dict[str, str | None] = {}  # url -> fetched text (None: failed), shared by threads
    pages_lock = threading.Lock()

    def page_texts(item: _Item) -> dict[int, str] | None:
        if not fetch_sources:
            return None
        out: dict[int, str] = {}
        for c in item.citations[:MAX_SOURCE_FETCHES]:
            with pages_lock:
                cached = c.url in pages
            if not cached:
                try:
                    _, body = fetch_text(c.url)
                    text: str | None = html_to_text(body)[:100_000]
                except (FetchError, OSError, httpx.HTTPError) as exc:
                    log.info("source page %s not fetched: %s", c.url, exc)
                    text = None
                with pages_lock:
                    pages[c.url] = text
            if pages.get(c.url):
                out[c.id] = pages[c.url] or ""
        return out

    def work(item: _Item) -> _Result:
        try:
            claims, dropped = extract_claims(chat, model=settings.model_extract, brand=name,
                                             aliases=aliases, text=item.text)  # fmt: skip
            if claims is None:
                return _Result(item.answer_id, [], ok=False)
            rows: list[dict[str, Any]] = []
            for i, c in enumerate(claims):
                reranker = chat if rerank else None
                hits = index.search(c.claim, embedder, k=5, rerank=reranker, rerank_model=model)
                checked = verify_claim(chat, model=model, brand=name, claim=c.claim,
                                       hits=hits, aliases=aliases)  # fmt: skip
                if checked is None:
                    return _Result(item.answer_id, [], ok=False, dropped=dropped)
                trace = None
                if checked.verdict == "contradicted":
                    trace = trace_source(c.claim, c.answer_span, item.text, item.citations,
                                         page_texts(item), brand_terms=bterms,
                                         marker_urls=item.marker_urls)  # fmt: skip
                rows.append(_row(i, c.claim, c.type, c.answer_span, checked, hits, trace))
            return _Result(item.answer_id, rows, ok=True, dropped=dropped)
        except BudgetExceededError:
            return _Result(item.answer_id, [], ok=False, refused=True)

    if workers is None:
        workers = 1 if settings.is_mock else settings.extract_concurrency
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = map(work, todo) if workers == 1 else pool.map(work, todo)
        for n, r in enumerate(results, start=1):
            if r.refused:
                report.budget_stopped += 1
            elif _store(factory, brand_id, r, version):
                report.checked += 1
                report.llm_failed += not r.ok
                report.claims += len(r.claims)
                report.verdicts.update(c["verdict"] for c in r.claims)
                report.traced += sum(1 for c in r.claims if c["source_citation_id"])
                report.downgraded += sum(1 for c in r.claims if c["details"].get("downgraded_from"))
                report.dropped.update(d["reason"] for d in r.dropped)
            if on_progress is not None:
                on_progress(n)
    return report


def _products(session: Session, brand: Brand) -> list[str]:
    """Product names *this brand's* mentions used ("Stridelab Tempo"), from its own runs
    only: they help the claim extractor recognise product-level facts as brand facts.
    (Unfiltered, another brand's names leaked in and its claims were checked against
    this brand's KB.)"""
    rows = session.scalars(
        select(Mention.entity)
        .join(Answer, Answer.id == Mention.answer_id)
        .join(Run, Run.id == Answer.run_id)
        .where(Run.brand_id == brand.id, Mention.is_brand, Mention.entity != brand.name)
        .group_by(Mention.entity)
        .order_by(func.count().desc(), Mention.entity)
        .limit(20)
    )
    return [r for r in rows if r.casefold() != brand.name.casefold()]


def _row(
    i: int,
    claim: str,
    ctype: str,
    span: str,
    checked: Checked,
    hits: list[Any],
    trace: Trace | None,
) -> dict[str, Any]:
    return {
        "ord": i,
        "claim_text": claim,
        "claim_type": ctype,
        "answer_span": span,
        "verdict": checked.verdict,
        "rationale": checked.rationale,
        "evidence_chunk_keys": checked.chunk_keys,
        "evidence_quote": checked.quote,
        "source_citation_id": trace.citation_id if trace else None,
        "source_confidence": trace.confidence if trace else None,
        "source_reason": trace.reason if trace else None,
        "details": {**checked.details, "retrieved": [h.chunk_key for h in hits]},
    }


def _store(factory: sessionmaker[Session], brand_id: int, r: _Result, version: str) -> bool:
    with factory() as session:
        session.execute(delete(AnswerFactcheck).where(AnswerFactcheck.answer_id == r.answer_id))
        session.add(AnswerFactcheck(answer_id=r.answer_id, version=version, ok=r.ok))
        if r.ok:
            session.execute(delete(Claim).where(Claim.answer_id == r.answer_id))
            session.add_all(Claim(answer_id=r.answer_id, brand_id=brand_id, **c) for c in r.claims)
        # A failed re-check keeps the previous claims (better than none) and is retried.
        try:
            session.commit()
        except IntegrityError as exc:  # a concurrent run stored this answer first
            session.rollback()
            log.warning("answer %s not stored: %s", r.answer_id, exc.orig)
            return False
    return True


__all__ = ["FACTCHECK_VERSION", "FactcheckReport", "factcheck_run", "resolve_brand"]
