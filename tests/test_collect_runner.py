"""Module 2 runner: speed, kill + resume, retries, budget guard, cache, citations."""

from __future__ import annotations

import asyncio
import random
import time
from collections import Counter
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from citewise.core.config import Settings
from citewise.models.collect import Answer, Citation, Run
from citewise.models.llm_call import LLMCall
from citewise.providers.base import Citation as Cite
from citewise.providers.base import Usage
from citewise.providers.engines.base import EngineAnswer, EngineError, is_retryable
from citewise.providers.engines.registry import get_engine
from citewise.services.collect.runner import (
    BudgetEstimateError,
    TokenBucket,
    backoff_delay,
    cache_key,
    citation_domain,
    create_run,
    estimate_run_cost,
    execute_run,
)
from citewise.services.panel.common import resolve_brand

ALL = ["openai", "anthropic", "gemini", "perplexity"]


def _plan(
    factory: sessionmaker[Session],
    settings: Settings,
    *,
    engines: list[str] = ALL,
    size: int = 50,
    max_cost: float = 5.0,
    system: str | None = None,
    brand: str = "Stridelab",
) -> int:
    with factory() as s:
        run = create_run(
            s,
            resolve_brand(s, brand),
            engines=engines,
            size=size,
            max_cost=max_cost,
            settings=settings,
            system_prompt=system,
        )
        s.commit()
        return run.id


def _count(factory: sessionmaker[Session], *where: Any, model: Any = Answer) -> int:
    with factory() as s:
        return int(s.scalar(select(func.count()).select_from(model).where(*where)) or 0)


class HTTPError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class FakeEngine:
    """Fails `fail_times` times with `error`, then answers. Records every call."""

    def __init__(
        self,
        name: str,
        *,
        fail_times: int = 0,
        error: Exception | None = None,
        cost: float | None = None,
    ) -> None:
        self.name = name
        self.model = "fake-model"
        self.fail_times = fail_times
        self.error = error or HTTPError(429)
        self.cost = cost
        self.calls: list[str] = []

    async def answer(self, prompt: str, *, system: str | None = None) -> EngineAnswer:
        self.calls.append(prompt)
        await asyncio.sleep(0)
        if len(self.calls) <= self.fail_times:
            raise self.error
        return EngineAnswer(
            text=f"{self.name} says: {prompt}",
            citations=[
                Cite(url="https://www.Example.com/a/?utm_source=x#frag", title="Ex", position=1)
            ],
            model=self.model,
            usage=Usage(100, 50),
            searches=1,
            reported_cost_usd=self.cost,
        )


class Yielding:
    """Wraps an adapter so every call yields to the event loop (lets a test cancel mid-run)."""

    def __init__(
        self, inner: Any, counter: Counter[str], gate: asyncio.Event, stop_at: int
    ) -> None:
        self.inner, self.name, self.model = inner, inner.name, inner.model
        self.counter, self.gate, self.stop_at = counter, gate, stop_at

    async def answer(self, prompt: str, *, system: str | None = None) -> EngineAnswer:
        await asyncio.sleep(0)
        ans = await self.inner.answer(prompt, system=system)
        self.counter["calls"] += 1
        if self.counter["calls"] >= self.stop_at:
            self.gate.set()
        await asyncio.sleep(0)
        return ans


async def _no_sleep(_: float) -> None:
    return None


# ---------------------------------------------------------------- acceptance


def test_mock_run_50_prompts_x_4_engines_under_10s(
    scan_factory: sessionmaker[Session], settings: Settings
) -> None:
    run_id = _plan(scan_factory, settings)
    start = time.perf_counter()
    final = asyncio.run(execute_run(run_id, settings, scan_factory))
    elapsed = time.perf_counter() - start

    assert elapsed < 10, f"took {elapsed:.1f}s"
    assert (final.status, final.done, final.failed, final.total) == ("completed", 200, 0, 200)
    assert _count(scan_factory, Answer.run_id == run_id, Answer.ok.is_(True)) == 200
    with scan_factory() as s:
        per_engine = Counter(s.scalars(select(Answer.engine).where(Answer.run_id == run_id)))
        cites = list(s.scalars(select(Citation)))
        calls = list(s.scalars(select(LLMCall).where(LLMCall.run_id == run_id)))
    assert per_engine == dict.fromkeys(ALL, 50)
    assert len(cites) >= 200  # every mock answer cites something
    assert all("utm_" not in c.normalized_url and "#" not in c.normalized_url for c in cites)
    assert not any(c.domain.startswith("vertexaisearch") for c in cites)  # Gemini -> real domain
    assert len(calls) == 200 and all(c.ok and c.cost_usd > 0 for c in calls)
    assert {c.provider for c in calls} == {f"mock:{e}" for e in ALL}
    assert final.cost_usd == pytest.approx(sum(c.cost_usd for c in calls))
    assert 0 < final.cost_usd <= final.estimated_cost_usd  # the estimate is conservative


def test_kill_halfway_then_resume_finishes_without_duplicates(
    scan_factory: sessionmaker[Session], settings: Settings
) -> None:
    run_id = _plan(scan_factory, settings)

    async def first_half() -> int:
        counter: Counter[str] = Counter()
        gate = asyncio.Event()
        adapters = {e: Yielding(get_engine(e, settings), counter, gate, 100) for e in ALL}
        task = asyncio.create_task(execute_run(run_id, settings, scan_factory, engines=adapters))
        await gate.wait()
        task.cancel()  # what Ctrl+C does to asyncio.run's main task
        with pytest.raises(asyncio.CancelledError):
            await task
        return counter["calls"]

    asyncio.run(first_half())
    with scan_factory() as s:
        run = s.get(Run, run_id)
        assert run is not None
        assert run.status == "interrupted"
        done_before = run.done
    assert 0 < done_before < 200

    resumed = {e: FakeEngine(e) for e in ALL}  # any adapter: only missing pairs are asked
    final = asyncio.run(
        execute_run(run_id, settings, scan_factory, engines=resumed, use_cache=False)
    )
    assert (final.status, final.done) == ("completed", 200)
    assert sum(len(f.calls) for f in resumed.values()) == 200 - done_before
    with scan_factory() as s:
        pairs = list(
            s.execute(select(Answer.prompt_id, Answer.engine).where(Answer.run_id == run_id))
        )
    assert len(pairs) == len(set(pairs)) == 200

    # Resuming a completed run is a no-op.
    again = {e: FakeEngine(e) for e in ALL}
    asyncio.run(execute_run(run_id, settings, scan_factory, engines=again))
    assert sum(len(f.calls) for f in again.values()) == 0


def test_hard_killed_run_left_running_is_resumable(
    scan_factory: sessionmaker[Session], settings: Settings
) -> None:
    """A crash leaves status 'running' and some answers; resume must still finish it."""
    run_id = _plan(scan_factory, settings, engines=["openai"], size=10)
    broken = FakeEngine("openai", fail_times=10**6, error=HTTPError(400))  # non-retryable
    first = asyncio.run(execute_run(run_id, settings, scan_factory, engines={"openai": broken}))
    assert first.status == "partial" and first.failed == 10
    with scan_factory() as s:
        run = s.get(Run, run_id)
        assert run is not None
        run.status = "running"  # simulate the process dying mid-run
        s.commit()
    healthy = FakeEngine("openai")
    final = asyncio.run(
        execute_run(run_id, settings, scan_factory, engines={"openai": healthy}, use_cache=False)
    )
    assert (final.status, final.done, final.failed) == ("completed", 10, 0)
    assert _count(scan_factory, Answer.run_id == run_id) == 10  # failed rows updated in place


def test_retry_fails_twice_then_succeeds(
    scan_factory: sessionmaker[Session], settings: Settings
) -> None:
    run_id = _plan(scan_factory, settings, engines=["anthropic"], size=1)
    flaky = FakeEngine("anthropic", fail_times=2, error=HTTPError(429))
    delays: list[float] = []

    async def record_sleep(d: float) -> None:
        delays.append(d)

    final = asyncio.run(
        execute_run(
            run_id,
            settings,
            scan_factory,
            engines={"anthropic": flaky},
            sleep=record_sleep,
            use_cache=False,
            seed=1,
        )
    )
    assert final.status == "completed"
    assert len(flaky.calls) == 3
    with scan_factory() as s:
        answer = s.scalar(select(Answer).where(Answer.run_id == run_id))
        calls = list(
            s.scalars(select(LLMCall).where(LLMCall.run_id == run_id).order_by(LLMCall.id))
        )
    assert answer is not None and answer.ok and answer.attempts == 3
    assert [c.ok for c in calls] == [False, False, True]
    assert "429" in (calls[0].error or "")
    assert len(delays) == 2
    assert 0 <= delays[0] <= settings.scan_backoff_base_s
    assert 0 <= delays[1] <= settings.scan_backoff_base_s * 2


@pytest.mark.parametrize(
    ("error", "expected_calls"),
    [(HTTPError(400), 1), (HTTPError(503), 4), (TimeoutError(), 4)],
)
def test_retry_only_retryable_errors_and_cap_attempts(
    scan_factory: sessionmaker[Session], settings: Settings, error: Exception, expected_calls: int
) -> None:
    run_id = _plan(scan_factory, settings, engines=["openai"], size=1)
    bad = FakeEngine("openai", fail_times=10**6, error=error)
    final = asyncio.run(
        execute_run(
            run_id,
            settings,
            scan_factory,
            engines={"openai": bad},
            sleep=_no_sleep,
            use_cache=False,
        )
    )
    assert len(bad.calls) == expected_calls == min(expected_calls, settings.scan_max_attempts)
    assert (final.status, final.failed) == ("partial", 1)
    with scan_factory() as s:
        answer = s.scalar(select(Answer).where(Answer.run_id == run_id))
    assert answer is not None and not answer.ok and answer.attempts == expected_calls


def test_budget_guard_aborts_before_start(
    scan_factory: sessionmaker[Session], settings: Settings
) -> None:
    estimate = estimate_run_cost(ALL, 50, settings)
    with pytest.raises(BudgetEstimateError) as info:
        _plan(scan_factory, settings, max_cost=estimate / 2)
    assert info.value.estimate == pytest.approx(estimate)
    assert _count(scan_factory, model=Run) == 0  # nothing planned...
    assert _count(scan_factory, LLMCall.purpose == "scan.answer", model=LLMCall) == 0  # ...or spent


def test_budget_guard_stops_mid_run_when_actual_spend_crosses_cap(
    scan_factory: sessionmaker[Session], settings: Settings
) -> None:
    run_id = _plan(scan_factory, settings, engines=["openai"], size=20, max_cost=2.0)
    pricey = FakeEngine("openai", cost=0.5)  # far above the per-call estimate
    final = asyncio.run(
        execute_run(run_id, settings, scan_factory, engines={"openai": pricey}, use_cache=False)
    )
    assert final.status == "stopped_budget"
    assert 4 <= final.done < 20
    # In-flight calls finish, so overshoot is bounded by the concurrency limit.
    assert final.cost_usd <= 2.0 + settings.scan_concurrency * 0.5
    assert "scan resume" in (final.error or "")

    # Raising the cap and resuming finishes the remaining pairs only.
    with scan_factory() as s:
        run = s.get(Run, run_id)
        assert run is not None
        run.max_cost_usd = 50.0
        s.commit()
    more = FakeEngine("openai", cost=0.5)
    done = asyncio.run(
        execute_run(run_id, settings, scan_factory, engines={"openai": more}, use_cache=False)
    )
    assert (done.status, done.done) == ("completed", 20)
    assert len(more.calls) == 20 - final.done


def test_same_day_reask_is_served_from_cache_for_free(
    scan_factory: sessionmaker[Session], settings: Settings
) -> None:
    first = asyncio.run(
        execute_run(
            _plan(scan_factory, settings, engines=["openai", "gemini"], size=10),
            settings,
            scan_factory,
        )
    )
    second_id = _plan(scan_factory, settings, engines=["openai", "gemini"], size=10)
    spy = {e: FakeEngine(e) for e in ("openai", "gemini")}
    second = asyncio.run(execute_run(second_id, settings, scan_factory, engines=spy))
    assert sum(len(f.calls) for f in spy.values()) == 0
    assert (second.status, second.done, second.cost_usd) == ("completed", 20, 0.0)
    assert first.cost_usd > 0
    assert (
        _count(scan_factory, Answer.run_id == second_id, Answer.cached_from_id.is_not(None)) == 20
    )
    assert _count(scan_factory, LLMCall.run_id == second_id, model=LLMCall) == 0
    with scan_factory() as s:  # citations are copied too
        n1 = s.scalar(
            select(func.count()).select_from(Citation).join(Answer).where(Answer.run_id == first.id)
        )
        n2 = s.scalar(
            select(func.count())
            .select_from(Citation)
            .join(Answer)
            .where(Answer.run_id == second_id)
        )
    assert n1 == n2 > 0

    # --no-cache re-asks.
    third_id = _plan(scan_factory, settings, engines=["openai"], size=10)
    fresh = FakeEngine("openai")
    asyncio.run(
        execute_run(third_id, settings, scan_factory, engines={"openai": fresh}, use_cache=False)
    )
    assert len(fresh.calls) == 10


def test_persona_system_prompt_is_stored_sent_and_part_of_the_cache_key(
    scan_factory: sessionmaker[Session], settings: Settings
) -> None:
    plain = asyncio.run(
        execute_run(
            _plan(scan_factory, settings, engines=["openai"], size=5), settings, scan_factory
        )
    )
    persona = "The user is in Toronto, Canada."
    local_id = _plan(scan_factory, settings, engines=["openai"], size=5, system=persona)
    local = asyncio.run(execute_run(local_id, settings, scan_factory))
    assert local.system_prompt == persona
    assert _count(scan_factory, Answer.run_id == local_id, Answer.cached_from_id.is_not(None)) == 0
    with scan_factory() as s:
        texts = list(s.scalars(select(Answer.text).where(Answer.run_id == local_id)))
    assert all("where you are" in t for t in texts)  # the mock engine honours the persona
    assert plain.status == local.status == "completed"
    day = "2026-11-23"
    assert cache_key("mock", "openai", "m", None, "q", day) != cache_key(
        "mock", "openai", "m", persona, "q", day
    )


# ---------------------------------------------------------------- units


def test_citation_domain_uses_title_for_gemini_redirects() -> None:
    redirect = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/AbC"
    assert citation_domain(redirect, "runnersworld.com") == "runnersworld.com"
    assert citation_domain(redirect, "Some page title") == "vertexaisearch.cloud.google.com"
    assert citation_domain("https://www.Nike.com/x", None) == "nike.com"


class _GenaiLikeError(Exception):
    """google-genai APIError: int `code`, string `status`."""

    def __init__(self, code: int, status: str) -> None:
        super().__init__(status)
        self.code, self.status = code, status


class APITimeoutError(Exception):
    pass


class APIConnectionError(Exception):
    pass


@pytest.mark.parametrize(
    ("exc", "retry"),
    [
        (HTTPError(429), True),
        (HTTPError(529), True),
        (HTTPError(500), True),
        (HTTPError(400), False),
        (HTTPError(401), False),
        (_GenaiLikeError(429, "RESOURCE_EXHAUSTED"), True),
        (_GenaiLikeError(404, "NOT_FOUND"), False),
        (TimeoutError(), True),
        (APITimeoutError(), True),
        (APIConnectionError(), True),
        (EngineError("x", retryable=False, status=429), False),  # explicit decision wins
        (ValueError("bug"), False),
    ],
)
def test_is_retryable(exc: Exception, retry: bool) -> None:
    assert is_retryable(exc) is retry


def test_backoff_is_bounded_full_jitter(settings: Settings) -> None:
    rng = random.Random(0)
    for attempt in range(1, 12):
        cap = min(settings.scan_backoff_max_s, settings.scan_backoff_base_s * 2 ** (attempt - 1))
        assert all(0 <= backoff_delay(attempt, settings, rng) <= cap for _ in range(50))


def test_token_bucket_spaces_out_requests() -> None:
    async def take(n: int) -> float:
        bucket = TokenBucket(rate=20.0, capacity=1)
        start = time.perf_counter()
        for _ in range(n):
            await bucket.acquire()
        return time.perf_counter() - start

    assert asyncio.run(take(5)) >= 4 / 20 * 0.9  # 1 burst token + 4 refills at 20/s


def test_estimate_scales_with_prompts_and_engines(settings: Settings) -> None:
    one = estimate_run_cost(["openai"], 10, settings)
    assert estimate_run_cost(["openai"], 20, settings) == pytest.approx(2 * one)
    assert estimate_run_cost(ALL, 50, settings) < 3.0  # the PLAN's ~$3 full-scan target
