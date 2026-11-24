"""The content pipeline: an explicit state machine, one LLM step per state.

    opportunity -> brief -> outline -> draft -> critique -> revise -> critique ... -> final
        -> awaiting_approval -> approved | rejected -> revise -> critique ... -> awaiting_approval

`ContentItem.status` is the last state reached; `advance()` performs the one step that
leaves it and `transition()` refuses any move not in TRANSITIONS. Each step commits on
its own, so a crash, an invalid model output or the spend cap leaves the item at its last
good state with `error` set, and `run_item()` resumes from there.

Revision policy: after a critique, revise only the criteria below their pass threshold
(plus any guardrail hit and reviewer comments), at most MAX_LOOPS times per round; stop
early when everything passes. If the loops run out with a guardrail still failing, the
offending sentences are removed (`checks.redact`) before the page reaches a human, so an
unsupported claim can never be approved by accident. A rejection starts a new round.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from citewise.core.config import REPO_ROOT, Settings
from citewise.models.brand import Brand
from citewise.models.content import ContentItem, ContentOpportunity, ContentReview, ContentVersion
from citewise.models.kb import KBChunk, KBDoc
from citewise.providers.base import ChatProvider, EmbeddingProvider
from citewise.schemas.content import (
    CRITERIA,
    BriefOut,
    CriterionScore,
    CritiqueOut,
    DraftOut,
    OutlineOut,
    RevisionOut,
)
from citewise.services.content import checks as chk
from citewise.services.content.measure import baseline
from citewise.services.content.outputs import (
    faq_jsonld,
    render_markdown,
    validate_faq_jsonld,
    why_note,
)
from citewise.services.content.voice import load_voice
from citewise.services.kb.retrieve import KBIndex
from citewise.services.prompt_library import render
from citewise.services.structured import complete_structured

log = logging.getLogger(__name__)

PIPELINE_VERSION = "c1+brief.v1+outline.v1+draft.v1+critique.v1+revise.v1"
MAX_LOOPS = 2  # automatic revisions per round
MAX_PASSAGES = 12
RUBRIC_PATH = REPO_ROOT / "evals" / "rubrics" / "content.yaml"

TRANSITIONS: dict[str, set[str]] = {
    "opportunity": {"brief"},
    "brief": {"outline"},
    "outline": {"draft"},
    "draft": {"critique"},
    "critique": {"revise", "final"},
    "revise": {"critique"},
    "final": {"awaiting_approval"},
    "awaiting_approval": {"approved", "rejected"},
    "rejected": {"revise"},
    "approved": set(),
}
STATES = tuple(TRANSITIONS)
STOP_STATES = {"awaiting_approval", "approved"}


class ContentError(ValueError):
    """A request the pipeline refuses (wrong state, no KB, a pitch-only opportunity)."""


class TransitionError(ContentError):
    pass


class StepFailed(RuntimeError):
    """An LLM step returned nothing usable after its retry; the item stays put."""


@dataclass
class Deps:
    chat: ChatProvider
    embedder: EmbeddingProvider | None
    settings: Settings


def transition(item: ContentItem, new: str) -> None:
    if new not in TRANSITIONS.get(item.status, set()):
        raise TransitionError(f"content {item.id}: can't go from {item.status} to {new}")
    item.status = new


@lru_cache
def rubric() -> dict[str, Any]:
    return yaml.safe_load(RUBRIC_PATH.read_text(encoding="utf-8"))


def thresholds() -> dict[str, int]:
    r = rubric()
    default = int(r.get("default_pass_threshold", 4))
    return {c: int(r["criteria"].get(c, {}).get("pass_threshold", default)) for c in CRITERIA}


def _judge_rubric() -> dict[str, Any]:
    """What the judge sees: descriptions, anchors and checks (not the guardrail notes)."""
    return {
        c: {k: v for k, v in spec.items() if k != "pass_threshold"}
        for c, spec in rubric()["criteria"].items()
    }


# ---------------------------------------------------------------- create / run


def create_item(session: Session, opp: ContentOpportunity, *, notes: str = "") -> ContentItem:
    """Start (or return the in-progress) item for an opportunity."""
    if opp.kind == "citation":
        raise ContentError(
            f"opportunity {opp.id} is a get-listed/pitch suggestion: outreach is manual "
            f"(see its details: {opp.details.get('suggestion', '')})"
        )
    open_item = session.scalar(
        select(ContentItem)
        .where(ContentItem.opportunity_id == opp.id, ContentItem.status != "approved")
        .order_by(ContentItem.id.desc())
    )
    if open_item is not None:
        return open_item  # resume instead of starting a duplicate
    item = ContentItem(
        brand_id=opp.brand_id,
        opportunity_id=opp.id,
        status="opportunity",
        title=opp.title[:300],
        format=opp.suggested_format,
        brief={"notes": notes} if notes else None,
    )
    opp.status = "in_progress"
    session.add(item)
    session.flush()
    return item


def run_item(
    factory: sessionmaker[Session], item_id: int, deps: Deps, *, max_steps: int = 30
) -> ContentItem:
    """Advance until a human is needed (awaiting_approval/approved) or a step fails."""
    for _ in range(max_steps):
        with factory() as session:
            item = session.get(ContentItem, item_id)
            if item is None:
                raise ContentError(f"content item {item_id} not found")
            if item.status in STOP_STATES:
                return item
            try:
                advance(session, item, deps)
                item.error = None
                session.commit()
            except StepFailed as exc:
                session.rollback()
                item = session.get(ContentItem, item_id)
                assert item is not None
                item.error = str(exc)
                session.commit()
                return item
    with factory() as session:
        item = session.get(ContentItem, item_id)
        assert item is not None
        return item


def advance(session: Session, item: ContentItem, deps: Deps) -> None:
    """Perform the one step that leaves the item's current state."""
    step = {
        "opportunity": _brief,
        "brief": _outline,
        "outline": _draft,
        "draft": _critique,
        "revise": _critique,
        "critique": _after_critique,
        "rejected": _revise,
        "final": lambda s, i, d: transition(i, "awaiting_approval"),
    }.get(item.status)
    if step is None:
        raise TransitionError(f"content {item.id}: nothing to do in state {item.status}")
    step(session, item, deps)


# ---------------------------------------------------------------- context


def _brand(session: Session, item: ContentItem) -> Brand:
    brand = session.get(Brand, item.brand_id)
    assert brand is not None
    return brand


def _passages(item: ContentItem) -> dict[str, str]:
    return {p["id"]: p["text"] for p in (item.brief or {}).get("passages", [])}


def _passage_list(item: ContentItem) -> list[dict[str, str]]:
    return list((item.brief or {}).get("passages", []))


def _public_brief(item: ContentItem) -> dict[str, Any]:
    b = item.brief or {}
    return {k: v for k, v in b.items() if k not in ("passages", "opportunity", "notes")}


def _call(deps: Deps, name: str, out: type[Any], model: str, **ctx: Any) -> Any:
    rendered = render(f"content/{name}", "v1", **ctx)
    result = complete_structured(
        deps.chat,
        rendered.messages,
        out,
        model=model,
        purpose=f"content.{name}",
        raise_on_budget=True,
    )
    if result is None:
        raise StepFailed(f"content.{name}: no valid output after a retry (see llm_calls)")
    return result


def _current(session: Session, item: ContentItem) -> ContentVersion:
    v = session.scalar(
        select(ContentVersion).where(
            ContentVersion.item_id == item.id, ContentVersion.version == item.current_version
        )
    )
    if v is None:
        raise ContentError(f"content {item.id} has no version {item.current_version}")
    return v


def _new_version(
    session: Session,
    item: ContentItem,
    draft: dict[str, Any],
    kind: str,
    feedback: list[dict[str, Any]] | None = None,
) -> ContentVersion:
    item.current_version += 1
    v = ContentVersion(
        item_id=item.id,
        version=item.current_version,
        kind=kind,
        markdown=render_markdown(draft),
        draft=draft,
        scores={},
        checks={},
        feedback=feedback or [],
    )
    session.add(v)
    item.title = draft.get("title", item.title)[:300]
    return v


# ---------------------------------------------------------------- steps


def _brief(session: Session, item: ContentItem, deps: Deps) -> None:
    brand = _brand(session, item)
    opp = session.get(ContentOpportunity, item.opportunity_id) if item.opportunity_id else None
    if opp is None:
        raise ContentError(f"content {item.id}: its opportunity was deleted")
    index = KBIndex.load(session, brand.id)
    if not len(index):
        raise ContentError(f"{brand.name} has no knowledge base: run `citewise kb ingest` first")
    details = opp.details or {}
    if opp.kind == "fact_fix":
        targets = [f"Is it true that {details.get('claim', '').rstrip('.')}?"]
    else:
        targets = list(details.get("target_prompts") or [opp.title])[:6]
    rival = details.get("rival") or ""
    queries = [*targets, opp.title]
    if rival:
        queries.append(f"{brand.name} compared with {rival}")
    seen: dict[str, str] = {}
    for key in details.get("evidence_chunk_keys", []):  # the KB fact that refutes the claim
        if key in index.texts:
            seen[key] = index.texts[key]
    for q in queries:
        for hit in index.search(q, deps.embedder, k=4):
            seen.setdefault(hit.chunk_key, hit.text)
    passages = [{"id": k, "text": t} for k, t in list(seen.items())[:MAX_PASSAGES]]
    notes = (item.brief or {}).get("notes", "")
    out: BriefOut = _call(
        deps,
        "brief",
        BriefOut,
        deps.settings.model_generate,
        brand=brand.name,
        opportunity={"kind": opp.kind, "title": opp.title, "reason": opp.reason},
        format=opp.suggested_format if opp.suggested_format != "pitch" else "faq",
        rival=rival,
        targets=targets,
        notes=notes,
        passages=passages,
    )
    ids = {p["id"] for p in passages}
    item.brief = {
        "audience": out.audience,
        "angle": out.angle,
        "questions": [q for q in out.questions if q.strip()][:6] or targets,
        # A correction page must state the KB fact; code adds it, whatever the model chose.
        "must_cover": list(
            dict.fromkeys(
                [k for k in details.get("evidence_chunk_keys", []) if k in ids]
                + [k for k in out.must_cover if k in ids]
            )
        ),
        "required_facts": [details["evidence_quote"]]
        if opp.kind == "fact_fix" and details.get("evidence_quote")
        else [],
        "comparison_points": out.comparison_points[:6],
        "format": out.format,
        "word_count": max(400, min(1200, out.word_count)),
        "unsupported_requests": out.unsupported_requests,
        "targets": targets,
        "target_prompt_ids": details.get("target_prompt_ids", []),
        "rival": rival,
        "notes": notes,
        "passages": passages,
        "opportunity": {"id": opp.id, "kind": opp.kind, "key": opp.key, "details": details},
        "pipeline_version": PIPELINE_VERSION,
    }
    item.format = out.format
    transition(item, "brief")


def _outline(session: Session, item: ContentItem, deps: Deps) -> None:
    brand = _brand(session, item)
    out: OutlineOut = _call(
        deps,
        "outline",
        OutlineOut,
        deps.settings.model_generate,
        brand=brand.name,
        brief=_public_brief(item),
        passages=_passage_list(item),
    )
    ids = set(_passages(item))
    outline = out.model_dump()
    for s in outline["sections"]:
        s["chunk_ids"] = [k for k in s["chunk_ids"] if k in ids]
    item.outline = outline
    item.title = out.title[:300]
    transition(item, "outline")


def _draft(session: Session, item: ContentItem, deps: Deps) -> None:
    brand = _brand(session, item)
    voice = load_voice(brand.name, deps.settings.brands_dir)
    out: DraftOut = _call(
        deps,
        "draft",
        DraftOut,
        deps.settings.model_generate,
        brand=brand.name,
        voice=voice["text"],
        brief=_public_brief(item),
        outline=item.outline,
        passages=_passage_list(item),
    )
    _new_version(session, item, out.model_dump(), "draft")
    transition(item, "draft")


def _critique(session: Session, item: ContentItem, deps: Deps) -> None:
    brand = _brand(session, item)
    voice = load_voice(brand.name, deps.settings.brands_dir)
    version = _current(session, item)
    checks = chk.run_checks(
        version.draft,
        passages=_passages(item),
        brief=item.brief or {},
        voice=voice,
        competitors=list(brand.competitors),
    )
    comments = [f["comment"] for f in version.feedback if f.get("source") == "reviewer"]
    out: CritiqueOut = _call(
        deps,
        "critique",
        CritiqueOut,
        deps.settings.model_judge,
        rubric=_judge_rubric(),
        brand=brand.name,
        voice=voice["text"],
        brief=_public_brief(item),
        comments=comments,
        passages=_passage_list(item),
        draft=version.markdown,
    )
    version.scores = score(out, checks)
    version.checks = checks.as_dict()
    transition(item, "critique")


def score(judge: CritiqueOut, checks: chk.Checks) -> dict[str, Any]:
    """Combine the judge's scores with the deterministic caps (min of the two)."""
    by: dict[str, CriterionScore] = {s.criterion: s for s in judge.scores}
    limits = thresholds()
    criteria: dict[str, Any] = {}
    for c in CRITERIA:
        j = by.get(c)
        judged = j.score if j else 1  # a criterion the judge skipped counts as failing
        final = min(judged, checks.caps.get(c, 5))
        issues = list(dict.fromkeys([*(j.issues if j else ["not scored"]), *checks.issues[c]]))
        criteria[c] = {
            "judge": judged,
            "cap": checks.caps.get(c, 5),
            "score": final,
            "passed": final >= limits[c],
            "issues": issues if final < limits[c] or checks.issues[c] else [],
            "rationale": j.rationale if j else "",
        }
    overall = round(sum(v["score"] for v in criteria.values()) / len(criteria), 3)
    return {
        "criteria": criteria,
        "overall": overall,
        "passed": all(v["passed"] for v in criteria.values()) and checks.guardrails_ok,
        "guardrails_ok": checks.guardrails_ok,
    }


def _after_critique(session: Session, item: ContentItem, deps: Deps) -> None:
    version = _current(session, item)
    if version.scores.get("passed") or item.loops >= MAX_LOOPS:
        _finalize(session, item, deps)
    else:
        _revise(session, item, deps)


def _revise(session: Session, item: ContentItem, deps: Deps) -> None:
    brand = _brand(session, item)
    voice = load_voice(brand.name, deps.settings.brands_dir)
    version = _current(session, item)
    reviewer = [f for f in item.pending_feedback if f.get("source") == "reviewer"]
    failing = {
        c: v["issues"]
        for c, v in version.scores.get("criteria", {}).items()
        if not v["passed"] or v["issues"]
    }
    if reviewer and deps.embedder is not None:
        _add_passages_for(session, item, [f["comment"] for f in reviewer], deps)
    out: RevisionOut = _call(
        deps,
        "revise",
        RevisionOut,
        deps.settings.model_generate,
        brand=brand.name,
        voice=voice["text"],
        brief=_public_brief(item),
        issues=failing,
        comments=[f["comment"] for f in reviewer],
        passages=_passage_list(item),
        draft=version.draft,
    )
    draft = DraftOut.model_validate(out.model_dump(exclude={"changes", "declined"})).model_dump()
    feedback = [{"source": "critique", "criterion": c, "issues": i} for c, i in failing.items()]
    feedback += reviewer
    feedback.append(
        {
            "source": "writer",
            "changes": out.changes,
            "declined": [d.model_dump() for d in out.declined],
        }
    )
    _new_version(session, item, draft, "revision", feedback)
    item.loops += 1
    item.pending_feedback = []
    transition(item, "revise")


def _add_passages_for(session: Session, item: ContentItem, comments: list[str], deps: Deps) -> None:
    """A reviewer may ask for something the original passages don't cover ("mention the
    trial"): retrieve for the comment so the writer can ground it."""
    index = KBIndex.load(session, item.brand_id)
    brief = dict(item.brief or {})
    have = {p["id"] for p in brief.get("passages", [])}
    extra = []
    for c in comments:
        for hit in index.search(c, deps.embedder, k=3):
            if hit.chunk_key not in have:
                have.add(hit.chunk_key)
                extra.append({"id": hit.chunk_key, "text": hit.text})
    if extra:
        brief["passages"] = [*brief.get("passages", []), *extra][: MAX_PASSAGES + 6]
        item.brief = brief


def _finalize(session: Session, item: ContentItem, deps: Deps) -> None:
    brand = _brand(session, item)
    version = _current(session, item)
    if not version.scores.get("guardrails_ok", False):
        voice = load_voice(brand.name, deps.settings.brands_dir)
        passages = _passages(item)
        before = chk.run_checks(
            version.draft,
            passages=passages,
            brief=item.brief or {},
            voice=voice,
            competitors=list(brand.competitors),
        )
        cleaned = chk.redact(version.draft, before)
        redacted = _new_version(
            session,
            item,
            cleaned,
            "redacted",
            [{"source": "guardrail", "removed": sorted(before.offending())}],
        )
        after = chk.run_checks(
            cleaned,
            passages=passages,
            brief=item.brief or {},
            voice=voice,
            competitors=list(brand.competitors),
        )
        redacted.checks = after.as_dict()
        # The judge's view of the pre-redaction draft is the best estimate left; the caps
        # are recomputed on what will actually be published.
        redacted.scores = _rescore(version.scores, after)
        version = redacted
    jsonld = faq_jsonld(version.draft)
    errors = validate_faq_jsonld(jsonld)
    version.jsonld = jsonld if not errors else None
    opp = (item.brief or {}).get("opportunity", {})
    version.note = why_note(
        opp.get("kind", ""),
        opp.get("details", {}),
        brand.name,
        (item.brief or {}).get("targets", []),
    )
    if errors:
        version.note += " (No FAQ JSON-LD: " + "; ".join(errors) + ")"
    transition(item, "final")


def _rescore(previous: dict[str, Any], checks: chk.Checks) -> dict[str, Any]:
    limits = thresholds()
    criteria = {}
    for c, v in previous.get("criteria", {}).items():
        final = min(v["judge"], checks.caps.get(c, 5))
        criteria[c] = {
            **v,
            "cap": checks.caps.get(c, 5),
            "score": final,
            "passed": final >= limits[c],
        }
    overall = round(sum(v["score"] for v in criteria.values()) / max(len(criteria), 1), 3)
    return {
        "criteria": criteria,
        "overall": overall,
        "guardrails_ok": checks.guardrails_ok,
        "passed": all(v["passed"] for v in criteria.values()) and checks.guardrails_ok,
    }


# ---------------------------------------------------------------- human in the loop


def approve(session: Session, item: ContentItem, comment: str = "") -> ContentItem:
    if item.status != "awaiting_approval":
        raise TransitionError(f"content {item.id} is {item.status}, not awaiting_approval")
    session.add(
        ContentReview(
            item_id=item.id, action="approve", comment=comment, version=item.current_version
        )
    )
    transition(item, "approved")
    item.approved_at = datetime.now(UTC)
    item.measurement = baseline(session, item)
    if item.opportunity_id:
        opp = session.get(ContentOpportunity, item.opportunity_id)
        if opp is not None:
            opp.status = "done"
    return item


def reject(session: Session, item: ContentItem, comment: str) -> ContentItem:
    """Record the rejection; the comment becomes revision input (run_item revises next)."""
    if item.status != "awaiting_approval":
        raise TransitionError(f"content {item.id} is {item.status}, not awaiting_approval")
    if not comment.strip():
        raise ContentError("a rejection needs a comment saying what to change")
    session.add(
        ContentReview(
            item_id=item.id, action="reject", comment=comment.strip(), version=item.current_version
        )
    )
    item.pending_feedback = [
        {"source": "reviewer", "comment": comment.strip(), "version": item.current_version}
    ]
    item.loops = 0  # a new round of automatic revisions
    transition(item, "rejected")
    return item


def chunk_sources(session: Session, brand_id: int) -> dict[str, dict[str, str]]:
    """chunk key -> the KB document's title and public URL (for export footnotes)."""
    rows = session.execute(
        select(KBChunk.chunk_key, KBDoc.title, KBDoc.source)
        .join(KBDoc, KBDoc.id == KBChunk.doc_id)
        .where(KBChunk.brand_id == brand_id)
    ).all()
    return {key: {"title": title, "url": source} for key, title, source in rows}
