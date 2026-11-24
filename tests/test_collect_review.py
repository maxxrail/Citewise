"""Regression tests for the Module 2 code-review findings (H1-H4, M1, M2, M5, L3, L7)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from citewise.core.config import Settings
from citewise.models.collect import Answer, Run
from citewise.models.llm_call import LLMCall
from citewise.models.panel import Prompt
from citewise.providers.base import Citation as Cite
from citewise.providers.base import Usage
from citewise.providers.engines.base import EngineAnswer, EngineError
from citewise.services.collect.runner import (
    RunLeasedError,
    create_run,
    estimate_call_cost,
    execute_run,
    select_prompts,
)
from citewise.services.panel.common import resolve_brand


def _plan(factory: sessionmaker[Session], settings: Settings, size: int = 5, **kw: Any) -> int:
    with factory() as s:
        run = create_run(
            s,
            resolve_brand(s, "Stridelab"),
            engines=kw.pop("engines", ["openai"]),
            size=size,
            max_cost=kw.pop("max_cost", 5.0),
            settings=settings,
            **kw,
        )
        s.commit()
        return run.id


class Fake:
    name, model = "openai", "fake"

    def __init__(self, behaviour: Any = None) -> None:
        self.behaviour = behaviour
        self.calls = 0

    async def answer(self, prompt: str, *, system: str | None = None) -> EngineAnswer:
        self.calls += 1
        n = self.calls  # read before yielding: concurrent calls move the counter
        await asyncio.sleep(0)
        if self.behaviour is not None:
            result = self.behaviour(n, prompt)
            if isinstance(result, BaseException):
                raise result
            if isinstance(result, EngineAnswer):
                return result
        return EngineAnswer(
            text=f"answer to {prompt}",
            citations=[],
            model=self.model,
            usage=Usage(100, 50),
            searches=1,
        )


def _calls(factory: sessionmaker[Session], run_id: int) -> list[LLMCall]:
    with factory() as s:
        return list(s.scalars(select(LLMCall).where(LLMCall.run_id == run_id).order_by(LLMCall.id)))


# ---- H1: mock answers must never be served to a live run


def test_cache_never_serves_mock_answers_to_live_runs(
    scan_factory: sessionmaker[Session], settings: Settings
) -> None:
    asyncio.run(execute_run(_plan(scan_factory, settings), settings, scan_factory))  # mock, cached
    live = settings.model_copy(update={"citewise_mode": "live"})
    live_id = _plan(scan_factory, live)
    engine = Fake()
    final = asyncio.run(execute_run(live_id, live, scan_factory, engines={"openai": engine}))
    assert engine.calls == 5  # every pair really asked
    assert final.mode == "live" and final.done == 5
    with scan_factory() as s:
        n_cached = s.scalar(
            select(func.count()).where(Answer.run_id == live_id, Answer.cached_from_id.is_not(None))
        )
    assert n_cached == 0
    assert {c.provider for c in _calls(scan_factory, live_id)} == {"openai"}


# ---- H2: progress is visible in the DB while the run is going (API / scheduler path)


def test_run_row_shows_progress_mid_run_without_a_callback(
    scan_factory: sessionmaker[Session], settings: Settings
) -> None:
    run_id = _plan(scan_factory, settings, size=10)
    seen: list[tuple[int, float]] = []

    def peek(n: int, _: str) -> None:
        if n == 8:
            with scan_factory() as s:
                run = s.get(Run, run_id)
                assert run is not None
                seen.append((run.done, run.cost_usd))

    asyncio.run(
        execute_run(run_id, settings, scan_factory, engines={"openai": Fake(peek)}, use_cache=False)
    )
    assert seen and seen[0][0] > 0 and seen[0][1] > 0


# ---- H3: billed failures are costed and count against the budget


def test_timeouts_are_charged_their_estimate(
    scan_factory: sessionmaker[Session], settings: Settings
) -> None:
    run_id = _plan(scan_factory, settings, size=1)
    engine = Fake(lambda n, p: TimeoutError())

    async def no_sleep(_: float) -> None:
        return None

    final = asyncio.run(
        execute_run(
            run_id,
            settings,
            scan_factory,
            engines={"openai": engine},
            sleep=no_sleep,
            use_cache=False,
        )
    )
    est = estimate_call_cost("openai", settings.engine_model_openai, settings)
    calls = _calls(scan_factory, run_id)
    assert len(calls) == settings.scan_max_attempts and not any(c.ok for c in calls)
    assert all(c.cost_usd == pytest.approx(est) for c in calls)
    assert final.cost_usd == pytest.approx(est * settings.scan_max_attempts)


def test_rejected_but_billed_responses_record_their_cost_and_are_not_retried(
    scan_factory: sessionmaker[Session], settings: Settings
) -> None:
    run_id = _plan(scan_factory, settings, size=1)
    truncated = EngineAnswer(
        text="Partial...",
        citations=[],
        model="fake",
        usage=Usage(10, 10),
        searches=2,
        reported_cost_usd=0.2,
    )
    engine = Fake(lambda n, p: EngineError("hit max tokens", retryable=False, answer=truncated))
    final = asyncio.run(
        execute_run(run_id, settings, scan_factory, engines={"openai": engine}, use_cache=False)
    )
    assert engine.calls == 1
    assert final.status == "partial" and final.cost_usd == pytest.approx(0.2)
    with scan_factory() as s:
        row = s.scalar(select(Answer).where(Answer.run_id == run_id))
    assert row is not None and not row.ok and row.cost_usd == pytest.approx(0.2)


def test_cancelled_in_flight_calls_are_logged_at_their_estimate(
    scan_factory: sessionmaker[Session], settings: Settings
) -> None:
    run_id = _plan(scan_factory, settings, size=10)
    gate = asyncio.Event()

    async def hang_after(n: int) -> None:
        if n >= 3:
            gate.set()
            await asyncio.sleep(3600)

    class Hanging(Fake):
        async def answer(self, prompt: str, *, system: str | None = None) -> EngineAnswer:
            self.calls += 1
            await hang_after(self.calls)
            return await Fake.answer(self, prompt, system=system)

    async def scenario() -> None:
        task = asyncio.create_task(
            execute_run(
                run_id, settings, scan_factory, engines={"openai": Hanging()}, use_cache=False
            )
        )
        await gate.wait()
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    cancelled = [c for c in _calls(scan_factory, run_id) if "cancelled" in (c.error or "")]
    assert cancelled and all(c.cost_usd > 0 for c in cancelled)
    with scan_factory() as s:
        run = s.get(Run, run_id)
        assert run is not None and run.status == "interrupted"
        assert run.cost_usd == pytest.approx(sum(c.cost_usd for c in _calls(scan_factory, run_id)))


# ---- H4: one executor per run


def test_a_run_held_by_another_process_cannot_be_executed_twice(
    scan_factory: sessionmaker[Session], settings: Settings
) -> None:
    run_id = _plan(scan_factory, settings)
    with scan_factory() as s:
        run = s.get(Run, run_id)
        assert run is not None
        run.status, run.heartbeat_at = "running", datetime.now(UTC)  # a live executor
        s.commit()
    engine = Fake()
    with pytest.raises(RunLeasedError):
        asyncio.run(execute_run(run_id, settings, scan_factory, engines={"openai": engine}))
    assert engine.calls == 0

    forced = asyncio.run(
        execute_run(
            run_id, settings, scan_factory, engines={"openai": engine}, force=True, use_cache=False
        )
    )
    assert forced.status == "completed" and forced.heartbeat_at is None  # lease released


def test_a_stale_lease_can_be_taken_over_without_force(
    scan_factory: sessionmaker[Session], settings: Settings
) -> None:
    run_id = _plan(scan_factory, settings)
    with scan_factory() as s:
        run = s.get(Run, run_id)
        assert run is not None
        run.status = "running"
        run.heartbeat_at = datetime.now(UTC) - timedelta(hours=1)  # executor died long ago
        s.commit()
    final = asyncio.run(
        execute_run(run_id, settings, scan_factory, engines={"openai": Fake()}, use_cache=False)
    )
    assert final.status == "completed"


def test_two_concurrent_executors_of_one_run_do_not_both_spend(
    scan_factory: sessionmaker[Session], settings: Settings
) -> None:
    run_id = _plan(scan_factory, settings, size=10)
    a, b = Fake(), Fake()

    async def both() -> list[Any]:
        return await asyncio.gather(
            execute_run(run_id, settings, scan_factory, engines={"openai": a}, use_cache=False),
            execute_run(run_id, settings, scan_factory, engines={"openai": b}, use_cache=False),
            return_exceptions=True,
        )

    results = asyncio.run(both())
    assert sum(isinstance(r, RunLeasedError) for r in results) == 1
    assert a.calls + b.calls == 10


# ---- M1: deleting prompts keeps history; such a run ends partial


def test_answers_keep_prompt_text_and_deleted_prompts_make_a_run_partial(
    scan_factory: sessionmaker[Session], settings: Settings
) -> None:
    run_id = _plan(scan_factory, settings, size=5)
    with scan_factory() as s:
        run = s.get(Run, run_id)
        assert run is not None
        s.execute(delete(Prompt).where(Prompt.id == run.prompt_ids[0]))
        s.commit()
    final = asyncio.run(
        execute_run(run_id, settings, scan_factory, engines={"openai": Fake()}, use_cache=False)
    )
    assert (final.status, final.done) == ("partial", 4)
    assert "deleted" in (final.error or "")
    with scan_factory() as s:
        texts = list(s.scalars(select(Answer.prompt_text).where(Answer.run_id == run_id)))
    assert len(texts) == 4 and all(texts)


# ---- M2: a bug in one task cancels the rest instead of letting them keep spending


def test_a_bug_in_one_task_stops_the_whole_run(
    scan_factory: sessionmaker[Session], settings: Settings
) -> None:
    run_id = _plan(scan_factory, settings, size=20)
    bad = EngineAnswer(
        text="ok",
        citations=[Cite(url=None, title=None, position=1)],  # type: ignore[arg-type]
        model="fake",
        usage=Usage(1, 1),
    )
    engine = Fake(lambda n, p: bad if n == 2 else None)
    with pytest.raises(BaseException):  # noqa: B017 - ExceptionGroup from the TaskGroup
        asyncio.run(
            execute_run(run_id, settings, scan_factory, engines={"openai": engine}, use_cache=False)
        )
    with scan_factory() as s:
        run = s.get(Run, run_id)
        assert run is not None and run.status == "failed" and run.heartbeat_at is None
    assert engine.calls < 20


# ---- M5: an empty answer is a failure, not "brand not mentioned"


def test_empty_answers_fail_without_retry(
    scan_factory: sessionmaker[Session], settings: Settings
) -> None:
    run_id = _plan(scan_factory, settings, size=1)
    empty = EngineAnswer(text="  ", citations=[], model="fake", usage=Usage(5, 0))
    engine = Fake(lambda n, p: empty)
    final = asyncio.run(
        execute_run(run_id, settings, scan_factory, engines={"openai": engine}, use_cache=False)
    )
    assert engine.calls == 1 and final.status == "partial"


# ---- L3 / L7


def test_smaller_runs_use_a_subset_of_the_panel_without_rebuilding_it(
    scan_factory: sessionmaker[Session], settings: Settings
) -> None:
    with scan_factory() as s:
        brand = resolve_brand(s, "Stridelab")
        before = set(
            s.scalars(select(Prompt.id).where(Prompt.brand_id == brand.id, Prompt.in_panel))
        )
        chosen = select_prompts(s, brand, 5)
        after = set(
            s.scalars(select(Prompt.id).where(Prompt.brand_id == brand.id, Prompt.in_panel))
        )
    assert before == after and len(before) == 50
    assert len(chosen) == 5 and {p.id for p in chosen} <= before


def test_cached_copies_carry_no_tokens(
    scan_factory: sessionmaker[Session], settings: Settings
) -> None:
    asyncio.run(execute_run(_plan(scan_factory, settings), settings, scan_factory))
    second = asyncio.run(execute_run(_plan(scan_factory, settings), settings, scan_factory))
    with scan_factory() as s:
        rows = list(s.scalars(select(Answer).where(Answer.run_id == second.id)))
    assert rows and all(r.cached_from_id and r.tokens_in == r.tokens_out == 0 for r in rows)
