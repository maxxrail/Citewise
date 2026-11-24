"""Regression tests for the `/review-module 02` findings (see PROGRESS.md, 2026-11-24)."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

import pytest
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from citewise.core.config import Settings
from citewise.models.collect import Answer, Citation, Run
from citewise.models.llm_call import LLMCall
from citewise.providers.base import Citation as Cite
from citewise.providers.base import Usage
from citewise.providers.engines.base import EngineAnswer, EngineError, is_timeout
from citewise.providers.engines.live import parse_perplexity
from citewise.providers.engines.registry import ENGINES, get_engine
from citewise.services.collect import runner
from citewise.services.collect.runner import create_run, estimate_call_cost, execute_run
from citewise.services.panel.common import resolve_brand


def _plan(factory: sessionmaker[Session], settings: Settings, size: int = 1, **kw: Any) -> int:
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


def _answer(prompt: str, citations: list[Cite] | None = None) -> EngineAnswer:
    return EngineAnswer(
        text=f"answer to {prompt}",
        citations=citations or [],
        model="fake",
        usage=Usage(100, 50),
        searches=1,
    )


async def _no_sleep(_: float) -> None:
    return None


def _run(factory: sessionmaker[Session], run_id: int) -> Run:
    with factory() as s:
        run = s.get(Run, run_id)
        assert run is not None
        s.expunge(run)
        return run


# ---- H1: the lease stays fresh while calls are slow, so nobody else can take the run


def test_lease_is_kept_fresh_while_no_answer_lands(
    scan_factory: sessionmaker[Session], settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Scaled down: a 0.1 s lease, beats every 0.02 s, and one call that takes 0.3 s. Before
    # the fix the heartbeat only moved when an answer landed, so mid-call the lease looked
    # stale and a second `scan resume` could claim the run and pay for every call again.
    monkeypatch.setattr(runner, "LEASE", timedelta(seconds=0.1))
    monkeypatch.setattr(runner, "HEARTBEAT_EVERY_S", 0.02)
    run_id = _plan(scan_factory, settings)
    stolen: list[bool] = []

    class Slow:
        name, model = "openai", "fake"

        async def answer(self, prompt: str, *, system: str | None = None) -> EngineAnswer:
            await asyncio.sleep(0.3)
            stolen.append(runner._claim(scan_factory, run_id, force=False))
            return _answer(prompt)

    final = asyncio.run(execute_run(run_id, settings, scan_factory, engines={"openai": Slow()}))
    assert stolen == [False]  # a second executor was refused mid-call
    assert final.status == "completed" and final.heartbeat_at is None  # lease released


# ---- M1: a setup failure after taking the lease is recorded, not left "running"


def test_missing_api_key_marks_the_run_failed_and_releases_the_lease(
    scan_factory: sessionmaker[Session], settings: Settings
) -> None:
    live = settings.model_copy(update={"citewise_mode": "live", "openai_api_key": None})
    run_id = _plan(scan_factory, live)  # the API returns 202 here, then executes in background
    with pytest.raises(Exception, match=r"(?i)key"):
        asyncio.run(execute_run(run_id, live, scan_factory))
    run = _run(scan_factory, run_id)
    assert run.status == "failed"
    assert "key" in (run.error or "").lower()
    assert run.heartbeat_at is None and run.finished_at is not None


def test_a_run_is_only_executed_in_its_own_mode(
    scan_factory: sessionmaker[Session], settings: Settings
) -> None:
    run_id = _plan(scan_factory, settings)  # a mock run
    live = settings.model_copy(update={"citewise_mode": "live"})
    with pytest.raises(ValueError, match="CITEWISE_MODE=mock"):
        asyncio.run(execute_run(run_id, live, scan_factory, engines={"openai": object()}))  # type: ignore[dict-item]
    assert _run(scan_factory, run_id).status == "pending"  # not claimed, nothing spent


# ---- M2: every engine caps its output tokens


class _Captured:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> Any:
        self.kwargs = kwargs
        payload = self.payload

        class Response:
            def model_dump(self, **_: Any) -> dict[str, Any]:
                return payload

        return Response()


def _live(settings: Settings) -> Settings:
    return settings.model_copy(
        update={
            "citewise_mode": "live",
            **{f"{e}_api_key": SecretStr("test") for e in ENGINES},
        }
    )


@pytest.mark.parametrize("engine", ["openai", "perplexity"])
def test_openai_and_perplexity_send_max_output_tokens(engine: str, settings: Settings) -> None:
    live = _live(settings)
    adapter: Any = get_engine(engine, live)
    payload = {
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    fake = _Captured(payload)
    adapter.client.responses = fake
    asyncio.run(adapter.answer("q"))
    assert fake.kwargs["max_output_tokens"] == live.scan_max_output_tokens[engine]


def test_gemini_and_anthropic_caps_come_from_settings(settings: Settings) -> None:
    live = _live(settings).model_copy(
        update={"scan_max_output_tokens": {"gemini": 1234, "anthropic": 2345}}
    )
    assert get_engine("gemini", live).max_output_tokens == 1234  # type: ignore[attr-defined]
    assert get_engine("anthropic", live).max_tokens == 2345  # type: ignore[attr-defined]


@pytest.mark.parametrize(("status", "retryable"), [("incomplete", False), ("failed", True)])
def test_perplexity_non_completed_status_is_an_error_that_keeps_billing(
    status: str, retryable: bool
) -> None:
    raw = {
        "status": status,
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "cut"}]}],
        "usage": {"input_tokens": 10, "output_tokens": 2500, "cost": {"total_cost": 0.01}},
    }
    with pytest.raises(EngineError) as info:
        parse_perplexity(raw, "perplexity/sonar")
    assert info.value.retryable is retryable
    assert info.value.answer is not None and info.value.answer.reported_cost_usd == 0.01


# ---- M3: SDK timeouts are charged their estimate, like asyncio timeouts


class APITimeoutError(Exception):
    """Same class name as openai/anthropic's SDK timeout."""


def test_is_timeout_recognises_sdk_timeouts() -> None:
    assert is_timeout(TimeoutError()) and is_timeout(APITimeoutError())
    assert not is_timeout(ValueError("bad request"))


def test_sdk_timeouts_are_charged_their_estimate(
    scan_factory: sessionmaker[Session], settings: Settings
) -> None:
    run_id = _plan(scan_factory, settings)

    class TimesOut:
        name, model = "openai", "fake"

        async def answer(self, prompt: str, *, system: str | None = None) -> EngineAnswer:
            raise APITimeoutError("Request timed out.")

    final = asyncio.run(
        execute_run(run_id, settings, scan_factory, engines={"openai": TimesOut()}, sleep=_no_sleep)
    )
    est = estimate_call_cost("openai", settings.engine_model_openai, settings)
    assert final.cost_usd == pytest.approx(est * settings.scan_max_attempts)


# ---- Low: spend so far comes from llm_calls, and long domains can't abort a run


def test_resume_counts_spend_logged_after_the_last_progress_refresh(
    scan_factory: sessionmaker[Session], settings: Settings
) -> None:
    run_id = _plan(scan_factory, settings, max_cost=1.0)
    with scan_factory() as s:  # a hard kill: the call was logged, the run row never updated
        s.add(LLMCall(purpose=runner.PURPOSE, run_id=run_id, kind="chat", provider="openai",
                      model="fake", cost_usd=0.999))  # fmt: skip
        s.commit()

    class Counting:
        name, model, calls = "openai", "fake", 0

        async def answer(self, prompt: str, *, system: str | None = None) -> EngineAnswer:
            self.calls += 1
            return _answer(prompt)

    engine = Counting()
    final = asyncio.run(execute_run(run_id, settings, scan_factory, engines={"openai": engine}))
    assert engine.calls == 0
    assert final.status == "stopped_budget"


def test_overlong_citation_domains_are_truncated_not_fatal(
    scan_factory: sessionmaker[Session], settings: Settings
) -> None:
    run_id = _plan(scan_factory, settings)
    url = "https://" + "a" * 300 + ".com/page"

    class LongHost:
        name, model = "openai", "fake"

        async def answer(self, prompt: str, *, system: str | None = None) -> EngineAnswer:
            return _answer(prompt, [Cite(url=url, title=None, position=1)])

    final = asyncio.run(execute_run(run_id, settings, scan_factory, engines={"openai": LongHost()}))
    assert final.status == "completed"
    with scan_factory() as s:
        cite = s.scalar(select(Citation).join(Answer).where(Answer.run_id == run_id))
    assert cite is not None and len(cite.domain) == runner.DOMAIN_MAX
