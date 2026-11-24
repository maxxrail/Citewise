"""Measurement hook: did approved content move the needle?

Method (see docs/DECISIONS.md, "Content measurement"):
- On approval we snapshot a baseline from the latest extracted scan, restricted to the
  item's *target prompts* (the panel prompts in the gap where the brand was missing):
  answers to those prompts, and how many mention the brand.
- For a fact fix we instead count answers repeating the wrong claim (contradicted claims
  with the same normalized text).
- `measure()` repeats the same computation on the newest extracted scan *after* approval
  and reports the difference. Same prompts, same engines, so the comparison is like for
  like; but it's one before/after pair with no control group, so a change is a signal to
  look at, not proof. Real lift also needs the page published and crawled first (weeks).
"""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from citewise.models.content import ContentItem

_MENTION_SQL = text(
    """
    SELECT a.id AS answer_id,
           MAX(CASE WHEN m.is_brand THEN 1 ELSE 0 END) AS mentioned
    FROM answers a
    JOIN answer_extractions x ON x.answer_id = a.id
    LEFT JOIN mentions m ON m.answer_id = a.id
    WHERE a.run_id = :run_id AND a.ok AND a.prompt_id IN :prompt_ids
    GROUP BY a.id
    """
).bindparams(bindparam("prompt_ids", expanding=True))

_RUNS_SQL = text(
    """SELECT DISTINCT a.run_id FROM answers a
       JOIN runs r ON r.id = a.run_id
       JOIN answer_extractions x ON x.answer_id = a.id
       WHERE r.brand_id = :brand_id ORDER BY a.run_id"""
)


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.casefold()).strip()


def _extracted_runs(session: Session, brand_id: int) -> list[int]:
    return [int(r) for r in session.execute(_RUNS_SQL, {"brand_id": brand_id}).scalars()]


def _mention_rate(session: Session, run_id: int, prompt_ids: list[int]) -> dict[str, Any]:
    if not prompt_ids:
        return {"run_id": run_id, "answers": 0, "mentioned": 0, "rate": None}
    rows = session.execute(_MENTION_SQL, {"run_id": run_id, "prompt_ids": prompt_ids}).all()
    n, hits = len(rows), sum(int(r.mentioned) for r in rows)
    return {
        "run_id": run_id,
        "answers": n,
        "mentioned": hits,
        "rate": round(hits / n, 4) if n else None,
    }


def _claim_repeats(session: Session, run_id: int, claim: str) -> dict[str, Any]:
    rows = session.execute(
        text(
            """SELECT c.claim_text FROM claims c JOIN answers a ON a.id = c.answer_id
               WHERE a.run_id = :run_id AND c.verdict = 'contradicted'"""
        ),
        {"run_id": run_id},
    ).scalars()
    return {"run_id": run_id, "repeats": sum(_norm(t) == _norm(claim) for t in rows)}


def _snapshot(session: Session, item: ContentItem, run_id: int) -> dict[str, Any]:
    brief = item.brief or {}
    opp = brief.get("opportunity", {})
    if opp.get("kind") == "fact_fix":
        claim = opp.get("details", {}).get("claim", "")
        return {"metric": "wrong_claim_repeats", **_claim_repeats(session, run_id, claim)}
    ids = [int(p) for p in brief.get("target_prompt_ids", []) if p]
    return {"metric": "mention_rate", "prompt_ids": ids, **_mention_rate(session, run_id, ids)}


def baseline(session: Session, item: ContentItem) -> dict[str, Any] | None:
    runs = _extracted_runs(session, item.brand_id)
    if not runs:
        return None
    return {"baseline": _snapshot(session, item, runs[-1])}


def measure(session: Session, item: ContentItem) -> dict[str, Any]:
    """Baseline vs the newest scan after it. `after` is None until a newer scan exists."""
    if item.status != "approved" or not item.measurement:
        return {"status": "not approved yet", "baseline": None, "after": None}
    base = item.measurement["baseline"]
    newer = [r for r in _extracted_runs(session, item.brand_id) if r > base["run_id"]]
    if not newer:
        return {"status": "waiting for a newer scan", "baseline": base, "after": None}
    after = _snapshot(session, item, newer[-1])
    out: dict[str, Any] = {"status": "measured", "baseline": base, "after": after}
    if (
        base.get("metric") == "mention_rate"
        and base["rate"] is not None
        and after["rate"] is not None
    ):
        out["delta"] = round(after["rate"] - base["rate"], 4)
    elif base.get("metric") == "wrong_claim_repeats":
        out["delta"] = after["repeats"] - base["repeats"]
    return out
