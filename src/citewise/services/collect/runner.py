"""The scan runner: send a run's prompts to every engine, concurrently and safely.

Guarantees:
- Idempotent: one answer row per (run_id, prompt_id, engine) (unique constraint). A
  re-run or `scan resume` only does pairs without an ok answer, and a failed row is
  updated in place.
- Resumable: the plan (prompt ids, engines, models) is frozen on the run, so a run killed
  halfway (Ctrl+C, crash, reboot) finishes exactly the planned work.
- Bounded spend: the run is refused before it starts if the estimate exceeds the cap.
  During the run, each call *reserves* its estimated cost before it is sent, so in-flight
  calls can't overshoot the cap by more than estimation error. Actual cost (tokens + per-
  search fees, or the provider's own figure) is what counts against the cap.
- Polite: per-engine concurrency limits, per-engine token-bucket rate limits (off in
  mock mode, where there is nobody to be polite to), exponential backoff with full jitter
  on 429/5xx/timeouts, and a hard attempt cap. Every attempt is logged to `llm_calls`.
- Cheap re-asks: an identical (engine, model, system prompt, prompt) asked on the same
  UTC day reuses the stored answer for $0 (`cached_from_id` records where it came from).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from citewise.core.config import Settings
from citewise.models.brand import Brand
from citewise.models.collect import Answer, Citation, Run
from citewise.models.panel import Prompt
from citewise.providers.base import Citation as EngineCitation
from citewise.providers.base import Usage
from citewise.providers.engines.base import EngineAdapter, EngineAnswer, is_retryable
from citewise.providers.engines.live import is_grounding_redirect
from citewise.providers.engines.registry import engine_model, get_engine
from citewise.providers.pricing import estimate_cost, search_cost
from citewise.providers.tracking import CallRecorder
from citewise.services.collect.urls import domain_of, normalize_url
from citewise.services.panel.select import build_panel

log = logging.getLogger(__name__)

PURPOSE = "scan.answer"
ProgressCallback = Callable[[Run], None]
Sleep = Callable[[float], Awaitable[None]]


class BudgetEstimateError(ValueError):
    """The run's estimated cost is above its cap; nothing was created or spent."""

    def __init__(self, estimate: float, cap: float) -> None:
        super().__init__(
            f"Estimated cost ${estimate:.2f} is above --max-cost ${cap:.2f}. Lower --size or "
            "--engines, or raise --max-cost."
        )
        self.estimate = estimate
        self.cap = cap


# ---------------------------------------------------------------- planning


def estimate_call_cost(engine: str, model: str, settings: Settings) -> float:
    """Expected cost of one answer: typical search-heavy token use + expected searches."""
    tokens = estimate_cost(model, settings.scan_est_tokens_in, settings.scan_est_tokens_out)
    return tokens + search_cost(engine, 1) * settings.scan_est_searches.get(engine, 1.5)


def estimate_run_cost(engines: list[str], n_prompts: int, settings: Settings) -> float:
    return round(
        sum(estimate_call_cost(e, engine_model(e, settings), settings) for e in engines)
        * n_prompts,
        4,
    )


def select_prompts(session: Session, brand: Brand, size: int) -> list[Prompt]:
    """The brand's scan panel; (re)built with `panel build` logic if its size differs."""
    panel = list(
        session.scalars(
            select(Prompt)
            .where(Prompt.brand_id == brand.id, Prompt.in_panel.is_(True))
            .order_by(Prompt.id)
        )
    )
    if len(panel) != size:
        log.info("panel for %s has %d prompts; building one of %d", brand.name, len(panel), size)
        report = build_panel(session, brand, size)
        chosen = set(report.selected)
        panel = [p for p in session.scalars(select(Prompt).where(Prompt.id.in_(chosen)))]
        panel.sort(key=lambda p: p.id)
    return panel


def create_run(
    session: Session,
    brand: Brand,
    *,
    engines: list[str],
    size: int,
    max_cost: float,
    settings: Settings,
    system_prompt: str | None = None,
) -> Run:
    """Plan a run and check the budget. Raises BudgetEstimateError before writing a run."""
    prompts = select_prompts(session, brand, size)
    if not prompts:
        raise ValueError(f"{brand.name} has no prompts; run `citewise panel generate` first.")
    estimate = estimate_run_cost(engines, len(prompts), settings)
    if estimate > max_cost:
        raise BudgetEstimateError(estimate, max_cost)
    run = Run(
        brand_id=brand.id,
        status="pending",
        engines=engines,
        models={e: engine_model(e, settings) for e in engines},
        prompt_ids=[p.id for p in prompts],
        system_prompt=(system_prompt or "").strip() or None,
        max_cost_usd=max_cost,
        estimated_cost_usd=estimate,
        total=len(prompts) * len(engines),
        mode=settings.citewise_mode,
    )
    session.add(run)
    session.flush()
    return run


# ---------------------------------------------------------------- helpers


def cache_key(engine: str, model: str, system: str | None, prompt: str, day: str) -> str:
    body = "\x1f".join([engine, model, system or "", prompt, day])
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def citation_domain(url: str, title: str | None) -> str:
    """Gemini cites opaque grounding-redirect URLs; the source domain is in the title."""
    if is_grounding_redirect(url) and title and "." in title and " " not in title.strip():
        return title.strip().lower().removeprefix("www.")
    return domain_of(url)


def backoff_delay(attempt: int, settings: Settings, rng: random.Random) -> float:
    """Full-jitter exponential backoff: uniform(0, min(cap, base * 2**(attempt-1)))."""
    ceiling = min(settings.scan_backoff_max_s, settings.scan_backoff_base_s * 2 ** (attempt - 1))
    return rng.uniform(0, ceiling)


class TokenBucket:
    """Allow `rate` requests per second on average, bursts of up to `capacity`."""

    def __init__(self, rate: float, capacity: float | None = None) -> None:
        self.rate = rate
        self.capacity = capacity or max(1.0, rate)
        self.tokens = self.capacity
        self.updated = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self.lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                await asyncio.sleep((1 - self.tokens) / self.rate)


@dataclass
class _Budget:
    cap: float
    spent: float
    reserved: float = 0.0
    stopped: bool = False

    def try_reserve(self, amount: float) -> bool:
        if self.stopped or self.spent + self.reserved + amount > self.cap:
            self.stopped = True
            return False
        self.reserved += amount
        return True

    def settle(self, reserved: float, actual: float) -> None:
        self.reserved -= reserved
        self.spent += actual
        if self.spent >= self.cap:
            self.stopped = True


# ---------------------------------------------------------------- storage


def _store_answer(
    factory: sessionmaker[Session],
    run_id: int,
    prompt_id: int,
    engine: str,
    *,
    model: str,
    key: str,
    answer: EngineAnswer | None,
    cost: float,
    attempts: int,
    error: str | None = None,
    cached_from_id: int | None = None,
) -> None:
    """Upsert one answer (+ citations). An existing ok answer is never overwritten."""
    with factory() as session:
        row = session.scalar(
            select(Answer).where(
                Answer.run_id == run_id, Answer.prompt_id == prompt_id, Answer.engine == engine
            )
        )
        if row is not None and row.ok:
            return
        if row is None:
            row = Answer(run_id=run_id, prompt_id=prompt_id, engine=engine)
            session.add(row)
        else:
            session.execute(delete(Citation).where(Citation.answer_id == row.id))
        row.model = model
        row.cache_key = key
        row.attempts = attempts
        row.cached_from_id = cached_from_id
        row.cost_usd = cost
        if answer is None:
            row.ok, row.text, row.raw_json, row.error = False, "", None, (error or "")[:4000]
        else:
            row.ok, row.text, row.raw_json, row.error = True, answer.text, answer.raw, None
            row.latency_ms = answer.latency_ms
            row.tokens_in, row.tokens_out = answer.usage.tokens_in, answer.usage.tokens_out
            row.searches = answer.searches
        try:
            session.flush()
            if answer is not None:
                for c in answer.citations:
                    session.add(
                        Citation(
                            answer_id=row.id,
                            position=c.position,
                            url=c.url,
                            normalized_url=normalize_url(c.url),
                            domain=citation_domain(c.url, c.title),
                            title=c.title,
                        )
                    )
            session.commit()
        except IntegrityError:  # another worker stored this pair first: theirs stands
            session.rollback()


def _cached(factory: sessionmaker[Session], key: str, run_id: int) -> tuple[Answer, list] | None:
    with factory() as session:
        src = session.scalar(
            select(Answer)
            .where(Answer.cache_key == key, Answer.ok.is_(True), Answer.run_id != run_id)
            .order_by(Answer.id)
            .limit(1)
        )
        if src is None:
            return None
        cites = list(session.scalars(select(Citation).where(Citation.answer_id == src.id)))
        session.expunge_all()
        return src, cites


def _refresh_counts(factory: sessionmaker[Session], run_id: int) -> Run:
    with factory() as session:
        run = session.get(Run, run_id)
        assert run is not None
        counts = dict(
            session.execute(
                select(Answer.ok, func.count()).where(Answer.run_id == run_id).group_by(Answer.ok)
            ).all()
        )
        run.done = counts.get(True, 0)
        run.failed = counts.get(False, 0)
        run.cost_usd = float(
            session.scalar(
                select(func.coalesce(func.sum(Answer.cost_usd), 0.0)).where(Answer.run_id == run_id)
            )
            or 0.0
        )
        session.commit()
        session.expunge(run)
        return run


# ---------------------------------------------------------------- execution


async def execute_run(
    run_id: int,
    settings: Settings,
    factory: sessionmaker[Session],
    *,
    engines: dict[str, EngineAdapter] | None = None,
    on_progress: ProgressCallback | None = None,
    sleep: Sleep = asyncio.sleep,
    use_cache: bool = True,
    seed: int | None = None,
) -> Run:
    """Run (or resume) a planned scan. Returns the final run row (detached)."""
    with factory() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise ValueError(f"No run {run_id}")
        if run.status == "completed":
            session.expunge(run)
            return run
        prompts = {
            p.id: p.text
            for p in session.scalars(select(Prompt).where(Prompt.id.in_(run.prompt_ids)))
        }
        done_pairs = {
            (a.prompt_id, a.engine)
            for a in session.scalars(select(Answer).where(Answer.run_id == run_id, Answer.ok))
        }
        run.status = "running"
        run.started_at = run.started_at or datetime.now(UTC)
        run.error = None
        session.commit()
        plan_engines, models, system = list(run.engines), dict(run.models), run.system_prompt
        cap, spent = run.max_cost_usd, run.cost_usd

    adapters = engines or {e: get_engine(e, settings) for e in plan_engines}
    todo = [
        (pid, e)
        for pid in run.prompt_ids
        for e in plan_engines
        if pid in prompts and (pid, e) not in done_pairs
    ]
    day = datetime.now(UTC).date().isoformat()
    budget = _Budget(cap=cap, spent=spent)
    recorder = CallRecorder(factory)
    sems = {e: asyncio.Semaphore(settings.scan_concurrency) for e in plan_engines}
    limiters = {
        e: TokenBucket(settings.scan_rate_per_s.get(e, 1.0))
        for e in plan_engines
        if not settings.is_mock
    }
    rng = random.Random(seed)
    est = {e: estimate_call_cost(e, models[e], settings) for e in plan_engines}
    progress_lock = asyncio.Lock()

    async def report() -> None:
        if on_progress is not None:
            async with progress_lock:
                on_progress(_refresh_counts(factory, run_id))

    async def work(pid: int, engine: str) -> None:
        model, text = models[engine], prompts[pid]
        key = cache_key(engine, model, system, text, day)
        if use_cache and (hit := _cached(factory, key, run_id)) is not None:
            src, cites = hit

            cached = EngineAnswer(
                text=src.text,
                citations=[
                    EngineCitation(url=c.url, title=c.title, position=c.position) for c in cites
                ],
                model=src.model,
                usage=Usage(src.tokens_in, src.tokens_out),
                searches=src.searches,
                raw=src.raw_json or {},
            )
            _store_answer(factory, run_id, pid, engine, model=model, key=key, answer=cached,
                          cost=0.0, attempts=0, cached_from_id=src.id)  # fmt: skip
            await report()
            return
        async with sems[engine]:
            for attempt in range(1, settings.scan_max_attempts + 1):
                if not budget.try_reserve(est[engine]):
                    return  # budget exhausted: leave the pair for `scan resume`
                if engine in limiters:
                    await limiters[engine].acquire()
                adapter = adapters[engine]
                start = time.perf_counter()
                try:
                    ans = await asyncio.wait_for(
                        adapter.answer(text, system=system), timeout=settings.llm_timeout_s
                    )
                except asyncio.CancelledError:
                    budget.settle(est[engine], 0.0)
                    raise
                except Exception as exc:
                    budget.settle(est[engine], 0.0)
                    err = f"{type(exc).__name__}: {exc}"
                    recorder.record(purpose=PURPOSE, kind="chat", provider=adapter.name,
                                    model=model, usage=Usage(), ok=False, error=err,
                                    latency_ms=int((time.perf_counter() - start) * 1000),
                                    run_id=run_id, cost_usd=0.0)  # fmt: skip
                    if is_retryable(exc) and attempt < settings.scan_max_attempts:
                        await sleep(backoff_delay(attempt, settings, rng))
                        continue
                    _store_answer(factory, run_id, pid, engine, model=model, key=key,
                                  answer=None, cost=0.0, attempts=attempt, error=err)  # fmt: skip
                    await report()
                    return
                ans.latency_ms = ans.latency_ms or int((time.perf_counter() - start) * 1000)
                cost = (
                    ans.reported_cost_usd
                    if ans.reported_cost_usd is not None
                    else estimate_cost(model, ans.usage.tokens_in, ans.usage.tokens_out)
                    + search_cost(engine, ans.searches)
                )
                budget.settle(est[engine], cost)
                provider = "mock:" + engine if settings.is_mock else adapter.name
                recorder.record(purpose=PURPOSE, kind="chat", provider=provider, model=model,
                                usage=ans.usage, latency_ms=ans.latency_ms, ok=True,
                                run_id=run_id, cost_usd=cost)  # fmt: skip
                _store_answer(factory, run_id, pid, engine, model=model, key=key, answer=ans,
                              cost=cost, attempts=attempt)  # fmt: skip
                await report()
                return

    status, error = "completed", None
    try:
        await asyncio.gather(*(work(pid, e) for pid, e in todo))
    except asyncio.CancelledError:
        status, error = "interrupted", "run was interrupted; continue with `citewise scan resume`"
        raise
    except Exception as exc:  # a bug, not a provider error (those are caught per call)
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        log.exception("scan run %s failed", run_id)
        raise
    finally:
        final = _refresh_counts(factory, run_id)
        if status == "completed":
            if final.done < final.total and budget.stopped:
                status = "stopped_budget"
                error = (
                    f"stopped at ${final.cost_usd:.4f} of ${cap:.2f}; "
                    "raise --max-cost and run `citewise scan resume`"
                )
            elif final.failed:
                status = "partial"
                error = f"{final.failed} answers failed after retries; `scan resume` retries them"
        with factory() as session:
            run_row = session.get(Run, run_id)
            assert run_row is not None
            run_row.status, run_row.error = status, error
            run_row.finished_at = datetime.now(UTC)
            session.commit()
            session.expunge(run_row)
    return run_row
