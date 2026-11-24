"""The scan runner: send a run's prompts to every engine, concurrently and safely.

Guarantees:
- Idempotent: one answer row per (run_id, prompt_id, engine) (unique constraint). A
  re-run or `scan resume` only does pairs without an ok answer, and a failed row is
  updated in place.
- Resumable: the plan (prompt ids + texts, engines, models) is frozen on the run, so a run
  killed halfway (Ctrl+C, crash, reboot) finishes exactly the planned work.
- One executor at a time: starting a run takes a lease (`runs.heartbeat_at`, refreshed by a
  background task every HEARTBEAT_EVERY_S, so slow calls can't let it lapse). A second
  process can't execute the same run while the lease is fresh, so two terminals resuming
  one run can't double-spend it. `force=True` takes over a lease,
  e.g. after a crash, without waiting for it to expire.
- Bounded spend: the run is refused before it starts if the estimate exceeds the cap.
  During the run, each call *reserves* its estimated cost before it is sent, so in-flight
  calls can't overshoot the cap by more than estimation error. Every attempt is costed and
  logged, including ones that fail after the provider billed them: a rejected response is
  charged what it reports, and a timeout is charged its estimate because the request may
  still complete and bill server-side. The run's cost is the sum of its `llm_calls`.
- Polite: per-engine concurrency limits, per-engine token-bucket rate limits (off in
  mock mode, where there is nobody to be polite to), exponential backoff with full jitter
  on 429/5xx/timeouts, and a hard attempt cap.
- Cheap re-asks: an identical (mode, engine, model, system prompt, prompt) asked on the same
  UTC day reuses the stored answer for $0. `cached_from_id` records the source, and copies
  carry zero tokens so token aggregates don't double-count. Mock answers are never served
  to live runs.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from citewise.core.config import Settings
from citewise.models.brand import Brand
from citewise.models.collect import Answer, Citation, Run
from citewise.models.llm_call import LLMCall
from citewise.models.panel import Prompt
from citewise.providers.base import Citation as EngineCitation
from citewise.providers.base import Usage
from citewise.providers.engines.base import (
    EngineAdapter,
    EngineAnswer,
    EngineError,
    close_adapter,
    is_retryable,
    is_timeout,
)
from citewise.providers.engines.live import MAX_PAUSE_TURNS, is_grounding_redirect
from citewise.providers.engines.registry import engine_model, get_engine
from citewise.providers.pricing import estimate_cost, search_cost
from citewise.providers.tracking import CallRecorder
from citewise.services.collect.urls import domain_of, normalize_url
from citewise.services.panel.select import PanelItem, build_panel, stratified_sample

log = logging.getLogger(__name__)

PURPOSE = "scan.answer"
LEASE = timedelta(minutes=2)  # a heartbeat older than this means the executor is gone
HEARTBEAT_EVERY_S = 10.0  # well inside LEASE, even with a slow DB
DOMAIN_MAX = 255  # citations.domain column width
PROGRESS_EVERY_S = 0.5  # run-row progress refresh; the final refresh is always exact
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


class RunLeasedError(RuntimeError):
    """Another process is executing this run right now."""


# ---------------------------------------------------------------- planning


def estimate_call_cost(engine: str, model: str, settings: Settings) -> float:
    """Expected cost of one answer: typical search-heavy token use + expected searches."""
    tokens = estimate_cost(model, settings.scan_est_tokens_in, settings.scan_est_tokens_out)
    return tokens + search_cost(engine, 1) * settings.scan_est_searches.get(engine, 1.5)


def estimate_run_cost(engines: list[str], n_prompts: int, settings: Settings) -> float:
    return sum(estimate_call_cost(e, engine_model(e, settings), settings) for e in engines) * (
        n_prompts
    )


def select_prompts(session: Session, brand: Brand, size: int) -> list[Prompt]:
    """The brand's scan panel, without changing it when possible.

    A smaller run takes a stratified subset of the panel (so trend runs keep asking the
    same questions), and only a larger run rebuilds it with `panel build` logic.
    """
    panel = list(
        session.scalars(
            select(Prompt)
            .where(Prompt.brand_id == brand.id, Prompt.in_panel.is_(True))
            .order_by(Prompt.id)
        )
    )
    if len(panel) == size:
        return panel
    if size < len(panel):
        chosen = set(
            stratified_sample([PanelItem(p.id, p.topic_id, p.intent) for p in panel], size)
        )
        return [p for p in panel if p.id in chosen]
    log.info("panel for %s has %d prompts; building one of %d", brand.name, len(panel), size)
    chosen = set(build_panel(session, brand, size).selected)
    return sorted(session.scalars(select(Prompt).where(Prompt.id.in_(chosen))), key=lambda p: p.id)


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
        estimated_cost_usd=round(estimate, 6),
        total=len(prompts) * len(engines),
        mode=settings.citewise_mode,
    )
    session.add(run)
    session.flush()
    return run


# ---------------------------------------------------------------- helpers


def cache_key(mode: str, engine: str, model: str, system: str | None, prompt: str, day: str) -> str:
    """Same-day reuse key. `mode` is part of it: mock and live share model IDs."""
    body = "\x1f".join([mode, engine, model, system or "", prompt, day])
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


def answer_cost(engine: str, model: str, ans: EngineAnswer) -> float:
    if ans.reported_cost_usd is not None:
        return ans.reported_cost_usd
    return estimate_cost(model, ans.usage.tokens_in, ans.usage.tokens_out) + search_cost(
        engine, ans.searches
    )


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
    _eps: float = field(default=1e-9, repr=False)  # float noise must not refuse the last call

    def try_reserve(self, amount: float) -> bool:
        if self.stopped or self.spent + self.reserved + amount > self.cap + self._eps:
            self.stopped = True
            return False
        self.reserved += amount
        return True

    def settle(self, reserved: float, actual: float) -> None:
        self.reserved -= reserved
        self.spent += actual
        if self.spent >= self.cap - self._eps:
            self.stopped = True


# ---------------------------------------------------------------- storage


def _store_answer(
    factory: sessionmaker[Session],
    run_id: int,
    prompt_id: int,
    prompt_text: str,
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
        row.prompt_text = prompt_text
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
                            domain=citation_domain(c.url, c.title)[:DOMAIN_MAX],
                            title=c.title,
                        )
                    )
            session.commit()
        except IntegrityError:  # another worker stored this pair first: theirs stands
            session.rollback()


def _cached(
    factory: sessionmaker[Session], key: str, run_id: int
) -> tuple[Answer, list[Citation]] | None:
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


def _refresh(factory: sessionmaker[Session], run_id: int, *, heartbeat: bool = True) -> Run:
    """Recompute done/failed/cost from the DB (so GET /runs/{id} shows live progress) and
    refresh the lease."""
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
                select(func.coalesce(func.sum(LLMCall.cost_usd), 0.0)).where(
                    LLMCall.run_id == run_id, LLMCall.purpose == PURPOSE
                )
            )
            or 0.0
        )
        if heartbeat:
            run.heartbeat_at = datetime.now(UTC)
        session.commit()
        session.expunge(run)
        return run


def _beat(factory: sessionmaker[Session], run_id: int) -> None:
    with factory() as session:
        session.execute(update(Run).where(Run.id == run_id).values(heartbeat_at=datetime.now(UTC)))
        session.commit()


async def _heartbeat(factory: sessionmaker[Session], run_id: int) -> None:
    """Keep the lease fresh for as long as the executor is alive, independent of answers
    landing: a stretch of slow or timing-out calls (Anthropic can take
    llm_timeout_s * MAX_PAUSE_TURNS per answer) must not look like a dead executor."""
    while True:
        await asyncio.sleep(HEARTBEAT_EVERY_S)
        try:
            _beat(factory, run_id)
        except Exception:  # a DB blip must not kill the run; the next beat retries
            log.warning("heartbeat for run %s failed", run_id, exc_info=True)


def _claim(factory: sessionmaker[Session], run_id: int, force: bool) -> bool:
    """Atomically take the run's lease. False if another live executor holds it."""
    now = datetime.now(UTC)
    stale = now - LEASE
    with factory() as session:
        cond = [Run.id == run_id, Run.status != "completed"]
        if not force:
            cond.append(
                or_(Run.status != "running", Run.heartbeat_at.is_(None), Run.heartbeat_at < stale)
            )
        result = session.execute(
            update(Run).where(*cond).values(status="running", heartbeat_at=now, error=None)
        )
        session.commit()
        return bool(result.rowcount)  # type: ignore[attr-defined]


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
    force: bool = False,
) -> Run:
    """Run (or resume) a planned scan. Returns the final run row (detached).

    Raises RunLeasedError if another process is executing it (unless `force`).
    """
    with factory() as session:
        run = session.get(Run, run_id)
        if run is None:
            raise ValueError(f"No run {run_id}")
        if run.mode != settings.citewise_mode:
            # The cache key carries the run's mode: executing a live run with mock adapters
            # (or the reverse) would file simulated answers under the other mode's key.
            raise ValueError(
                f"run {run_id} is a {run.mode} run; execute it with CITEWISE_MODE={run.mode}"
            )
        if run.status == "completed":
            session.expunge(run)
            return run
    if not _claim(factory, run_id, force):
        raise RunLeasedError(
            f"run {run_id} is being executed by another process (heartbeat within "
            f"{int(LEASE.total_seconds())}s). Wait, or pass --force if that process is gone."
        )
    # From here on the run is ours: every exit path, including a setup failure such as a
    # missing API key, must go through the `finally` below that records the outcome and
    # releases the lease. Otherwise the run reads "running" forever.
    owned = engines is None  # close only the clients we created
    adapters: dict[str, EngineAdapter] = {}
    missing: list[int] = []
    budget = _Budget(cap=0.0, spent=0.0)
    status, error = "completed", None
    beat: asyncio.Task[None] | None = None
    try:
        with factory() as session:
            run = session.get(Run, run_id)
            assert run is not None
            prompts = {
                p.id: p.text
                for p in session.scalars(select(Prompt).where(Prompt.id.in_(run.prompt_ids)))
            }
            done_pairs = {
                (a.prompt_id, a.engine)
                for a in session.scalars(select(Answer).where(Answer.run_id == run_id, Answer.ok))
            }
            run.started_at = run.started_at or datetime.now(UTC)
            session.commit()
            plan_engines, models, system = list(run.engines), dict(run.models), run.system_prompt
            cap, mode = run.max_cost_usd, run.mode
            planned = list(run.prompt_ids)
        # Spend so far, from llm_calls rather than the run row: after a hard kill the row's
        # cost can lag the last few calls.
        spent = _refresh(factory, run_id).cost_usd
        missing = [pid for pid in planned if pid not in prompts]
        budget = _Budget(cap=cap, spent=spent)
        beat = asyncio.create_task(_heartbeat(factory, run_id))
        for e in plan_engines:
            adapters[e] = engines[e] if engines is not None else get_engine(e, settings)
        await _execute(
            run_id,
            settings,
            factory,
            todo=[
                (pid, e)
                for pid in planned
                if pid in prompts
                for e in plan_engines
                if (pid, e) not in done_pairs
            ],
            prompts=prompts,
            adapters=adapters,
            models=models,
            system=system,
            mode=mode,
            budget=budget,
            on_progress=on_progress,
            sleep=sleep,
            use_cache=use_cache,
            seed=seed,
        )
    except asyncio.CancelledError:
        status, error = "interrupted", "run was interrupted; continue with `citewise scan resume`"
        raise
    except Exception as exc:  # a bug or a setup failure, not a provider error (caught per call)
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        log.exception("scan run %s failed", run_id)
        raise
    finally:
        if beat is not None:
            beat.cancel()
        if owned:
            for adapter in adapters.values():
                try:
                    await close_adapter(adapter)
                except Exception:  # closing must never mask the run's own outcome
                    log.warning("closing %s client failed", adapter.name, exc_info=True)
        final = _refresh(factory, run_id, heartbeat=False)
        if status == "completed":
            if final.done < final.total and budget.stopped:
                status = "stopped_budget"
                error = (
                    f"stopped at ${final.cost_usd:.4f} of ${budget.cap:.2f}; "
                    "raise --max-cost and run `citewise scan resume`"
                )
            elif final.failed or missing:
                status = "partial"
                parts = []
                if final.failed:
                    parts.append(
                        f"{final.failed} answers failed after retries (`scan resume` retries them)"
                    )
                if missing:
                    parts.append(f"{len(missing)} planned prompts were deleted and can't be asked")
                error = "; ".join(parts)
        with factory() as session:
            run_row = session.get(Run, run_id)
            assert run_row is not None
            run_row.status, run_row.error = status, error
            run_row.finished_at = datetime.now(UTC)
            run_row.heartbeat_at = None  # release the lease
            session.commit()
            session.expunge(run_row)
    return run_row


async def _execute(
    run_id: int,
    settings: Settings,
    factory: sessionmaker[Session],
    *,
    todo: list[tuple[int, str]],
    prompts: dict[int, str],
    adapters: dict[str, EngineAdapter],
    models: dict[str, str],
    system: str | None,
    mode: str,
    budget: _Budget,
    on_progress: ProgressCallback | None,
    sleep: Sleep,
    use_cache: bool,
    seed: int | None,
) -> None:
    """Ask every (prompt, engine) pair in `todo`. Provider errors are handled per call; any
    other exception cancels the siblings and propagates to `execute_run`."""
    plan_engines = list(adapters)
    day = datetime.now(UTC).date().isoformat()
    recorder = CallRecorder(factory)
    sems = {e: asyncio.Semaphore(settings.scan_concurrency) for e in plan_engines}
    limiters = {
        e: TokenBucket(settings.scan_rate_per_s.get(e, 1.0))
        for e in plan_engines
        if not settings.is_mock
    }
    rng = random.Random(seed)
    est = {e: estimate_call_cost(e, models[e], settings) for e in plan_engines}
    # Anthropic may need several requests per answer (pause_turn), each with the SDK timeout.
    call_timeout = settings.llm_timeout_s * MAX_PAUSE_TURNS + 10
    last_report = [0.0]  # report the first answer straight away

    def report() -> None:
        # Progress goes to the run row (for GET /runs/{id}) at most every PROGRESS_EVERY_S:
        # a refresh is a few queries, and doing it per answer dominated a fast mock run.
        now = time.monotonic()
        if now - last_report[0] < PROGRESS_EVERY_S:
            return
        last_report[0] = now
        row = _refresh(factory, run_id, heartbeat=False)
        if on_progress is not None:
            on_progress(row)

    def log_call(
        engine: str, usage: Usage, cost: float, latency: int, ok: bool, err: str | None
    ) -> None:
        provider = f"mock:{engine}" if settings.is_mock else adapters[engine].name
        recorder.record(purpose=PURPOSE, kind="chat", provider=provider, model=models[engine],
                        usage=usage, latency_ms=latency, ok=ok, error=err, run_id=run_id,
                        cost_usd=cost)  # fmt: skip

    async def work(pid: int, engine: str) -> None:
        model, text = models[engine], prompts[pid]
        key = cache_key(mode, engine, model, system, text, day)
        if use_cache and (hit := _cached(factory, key, run_id)) is not None:
            src, cites = hit
            copy = EngineAnswer(
                text=src.text,
                citations=[EngineCitation(c.url, c.title, c.position) for c in cites],
                model=src.model,
                usage=Usage(0, 0),  # nothing was spent; don't double-count tokens
                raw=src.raw_json or {},
            )
            _store_answer(factory, run_id, pid, text, engine, model=model, key=key, answer=copy,
                          cost=0.0, attempts=0, cached_from_id=src.id)  # fmt: skip
            report()
            return
        async with sems[engine]:
            for attempt in range(1, settings.scan_max_attempts + 1):
                if not budget.try_reserve(est[engine]):
                    return  # budget exhausted: leave the pair for `scan resume`
                if engine in limiters:
                    await limiters[engine].acquire()
                start = time.perf_counter()
                try:
                    ans = await asyncio.wait_for(
                        adapters[engine].answer(text, system=system), timeout=call_timeout
                    )
                    if not ans.text.strip():  # an empty answer would read as "not mentioned"
                        raise EngineError("empty answer", retryable=False, answer=ans)
                except asyncio.CancelledError:
                    # The request was sent and may still bill server-side: count the estimate.
                    budget.settle(est[engine], est[engine])
                    latency = int((time.perf_counter() - start) * 1000)
                    log_call(engine, Usage(), est[engine], latency, False,
                             "cancelled in flight; cost is the estimate")  # fmt: skip
                    raise
                except Exception as exc:
                    latency = int((time.perf_counter() - start) * 1000)
                    billed = exc.answer if isinstance(exc, EngineError) else None
                    if billed is not None:
                        cost, usage = answer_cost(engine, model, billed), billed.usage
                    elif is_timeout(exc):  # asyncio's or the SDK's (APITimeoutError)
                        cost, usage = est[engine], Usage()  # sent, so assume it was billed
                    else:
                        cost, usage = 0.0, Usage()  # refused before any work was done
                    budget.settle(est[engine], cost)
                    err = f"{type(exc).__name__}: {exc}"
                    log_call(engine, usage, cost, latency, False, err)
                    if is_retryable(exc) and attempt < settings.scan_max_attempts:
                        await sleep(backoff_delay(attempt, settings, rng))
                        continue
                    _store_answer(factory, run_id, pid, text, engine, model=model, key=key,
                                  answer=None, cost=cost, attempts=attempt, error=err)  # fmt: skip
                    report()
                    return
                ans.latency_ms = ans.latency_ms or int((time.perf_counter() - start) * 1000)
                cost = answer_cost(engine, model, ans)
                budget.settle(est[engine], cost)
                log_call(engine, ans.usage, cost, ans.latency_ms, True, None)
                _store_answer(factory, run_id, pid, text, engine, model=model, key=key,
                              answer=ans, cost=cost, attempts=attempt)  # fmt: skip
                report()
                return

    # TaskGroup: if one task hits a bug, the siblings are cancelled instead of carrying on
    # spending after the run was marked failed.
    async with asyncio.TaskGroup() as tg:
        for pid, e in todo:
            tg.create_task(work(pid, e))
