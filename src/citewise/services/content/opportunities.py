"""Find content opportunities: where writing something would most plausibly help.

Three kinds, each scored `impact x confidence` and explained in one sentence:

Impact is in one unit for all three, "answers affected", so the kinds rank together:

- **gap**: a topic x intent slice of the panel where the brand's mention rate trails the
  best-placed other company's by at least GAP_MIN.
  impact = answers in the slice x rate gap (the answers' worth of missing mentions);
  confidence = n / (n + 8) for n answers, so small slices count for less.
- **citation**: domains the engines cite in a topic's answers that *don't* mention the
  brand (review sites, listicles, Reddit threads). These are "get listed / pitch"
  suggestions; outreach is never automated.
  impact = CITATION_WEIGHT x answers citing that domain (getting listed is indirect: the
  brand still has to be picked from the listing); confidence = n / (n + 4).
- **fact_fix**: a claim Module 4 marked contradicted (verbatim KB quote) -> a correction
  page / FAQ entry. impact = answers repeating it x (1 + 0.25 per extra engine), since a
  wrong fact repeated across engines is worse than one engine's slip; confidence = 0.9.

Opportunities are upserted by a stable `key` (e.g. "gap:t12:commercial"), so refreshing
updates rows rather than duplicating them. Open ones that no longer appear are marked
`stale`; ones already in progress or done are left alone.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from citewise.models.brand import Brand
from citewise.models.collect import Answer
from citewise.models.content import ContentOpportunity
from citewise.models.kb import Claim
from citewise.models.panel import Topic
from citewise.services.factcheck.report import latest_factchecked_run
from citewise.services.metrics.visibility import latest_extracted_run

GAP_MIN = 0.2  # mention-rate gap worth writing for
MIN_ANSWERS = 4  # below this a slice says nothing
FACT_CONFIDENCE = 0.9  # a contradiction already carries a verbatim KB quote
CITATION_WEIGHT = 0.5  # a listing is an indirect route to a mention
FORMAT_BY_INTENT = {
    "comparison": "comparison",
    "commercial": "guide",
    "informational": "guide",
    "navigational": "faq",
    "transactional": "faq",
}


@dataclass
class Found:
    key: str
    kind: str
    title: str
    reason: str
    impact: float
    confidence: float
    suggested_format: str
    run_id: int | None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return round(self.impact * self.confidence, 4)


_SLICE_SQL = text(
    """
    SELECT a.id AS answer_id, a.prompt_id AS prompt_id, a.prompt_text AS prompt_text,
           p.topic_id AS topic_id, p.intent AS intent,
           m.canonical AS canonical, m.entity_type AS entity_type
    FROM answers a
    JOIN runs r ON r.id = a.run_id
    JOIN answer_extractions x ON x.answer_id = a.id
    LEFT JOIN prompts p ON p.id = a.prompt_id
    LEFT JOIN mentions m ON m.answer_id = a.id
    WHERE r.brand_id = :brand_id AND a.run_id = :run_id AND a.ok
    """
)


def _gaps(session: Session, brand: Brand, run_id: int, topics: dict[int, str]) -> list[Found]:
    answers: dict[int, dict[str, Any]] = {}
    for r in session.execute(_SLICE_SQL, {"brand_id": brand.id, "run_id": run_id}).mappings():
        a = answers.setdefault(
            r["answer_id"],
            {
                "prompt_id": r["prompt_id"],
                "prompt": r["prompt_text"],
                "topic": r["topic_id"],
                "intent": r["intent"],
                "brand": False,
                "others": set(),
            },
        )
        if r["entity_type"] == "brand":
            a["brand"] = True
        elif r["entity_type"] in ("competitor", "unknown") and r["canonical"]:
            a["others"].add(r["canonical"])
    slices: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for a in answers.values():
        if a["topic"] is not None and a["intent"]:
            slices[(a["topic"], a["intent"])].append(a)
    out = []
    for (topic_id, intent), rows in slices.items():
        n = len(rows)
        if n < MIN_ANSWERS:
            continue
        brand_hits = sum(r["brand"] for r in rows)
        brand_rate = brand_hits / n
        rivals = Counter(o for r in rows for o in r["others"])
        if not rivals:
            continue
        rival, hits = rivals.most_common(1)[0]
        gap = hits / n - brand_rate
        if gap < GAP_MIN:
            continue
        missed = [r for r in rows if not r["brand"]]
        prompts = list(dict.fromkeys(r["prompt"] for r in missed))
        label = topics.get(topic_id, f"topic {topic_id}")
        out.append(
            Found(
                key=f"gap:t{topic_id}:{intent}",
                kind="gap",
                title=f"{label} ({intent}): {brand.name} trails {rival}",
                reason=(
                    f"In '{label}' ({intent}) {brand.name} appears in {brand_hits}/{n} "
                    f"answers ({brand_rate:.0%}) vs {rival} in {hits}/{n} ({hits / n:.0%})."
                ),
                impact=round(n * gap, 4),
                confidence=round(n / (n + 8), 4),
                suggested_format=FORMAT_BY_INTENT.get(intent, "guide"),
                run_id=run_id,
                details={
                    "topic_id": topic_id,
                    "topic": label,
                    "intent": intent,
                    "brand_rate": round(brand_rate, 4),
                    "rival": rival,
                    "rival_rate": round(hits / n, 4),
                    "answers": n,
                    "target_prompt_ids": list(
                        dict.fromkeys(r["prompt_id"] for r in missed if r["prompt_id"])
                    )[:10],
                    "target_prompts": prompts[:10],
                },
            )
        )
    return out


_CITE_SQL = text(
    """
    SELECT p.topic_id AS topic_id, c.domain AS domain, c.url AS url, c.title AS title,
           a.id AS answer_id,
           MAX(CASE WHEN m.is_brand THEN 1 ELSE 0 END) AS mentioned
    FROM answers a
    JOIN runs r ON r.id = a.run_id
    JOIN answer_extractions x ON x.answer_id = a.id
    JOIN citations c ON c.answer_id = a.id
    LEFT JOIN prompts p ON p.id = a.prompt_id
    LEFT JOIN mentions m ON m.answer_id = a.id
    WHERE r.brand_id = :brand_id AND a.run_id = :run_id AND a.ok
    GROUP BY p.topic_id, c.id, c.domain, c.url, c.title, a.id
    """
)
_REDIRECT_HOSTS = ("vertexaisearch.cloud.google.com",)  # Gemini redirects carry no domain


def _citations(session: Session, brand: Brand, run_id: int, topics: dict[int, str]) -> list[Found]:
    own = brand.domain.lower().removeprefix("www.")
    absent: dict[int, Counter[str]] = defaultdict(Counter)
    examples: dict[tuple[int, str], dict[str, str]] = {}
    counted: set[tuple[int, str]] = set()
    for r in session.execute(_CITE_SQL, {"brand_id": brand.id, "run_id": run_id}).mappings():
        domain = (r["domain"] or "").lower()
        if r["topic_id"] is None or int(r["mentioned"]) or not domain:
            continue
        if domain == own or domain.endswith("." + own) or domain in _REDIRECT_HOSTS:
            continue
        if (r["answer_id"], domain) in counted:
            continue
        counted.add((r["answer_id"], domain))
        absent[r["topic_id"]][domain] += 1
        examples.setdefault((r["topic_id"], domain), {"url": r["url"], "title": r["title"] or ""})
    out = []
    for topic_id, counts in absent.items():
        label = topics.get(topic_id, f"topic {topic_id}")
        for domain, n in counts.most_common(3):
            if n < 2:
                continue
            ex = examples[(topic_id, domain)]
            out.append(
                Found(
                    key=f"citation:t{topic_id}:{domain}",
                    kind="citation",
                    title=f"Get listed on {domain} for '{label}'",
                    reason=(
                        f"Engines cited {domain} in {n} '{label}' answers that don't "
                        f'mention {brand.name}, e.g. "{ex["title"][:80]}".'
                    ),
                    impact=CITATION_WEIGHT * n,
                    confidence=round(n / (n + 4), 4),
                    suggested_format="pitch",
                    run_id=run_id,
                    details={
                        "topic_id": topic_id,
                        "topic": label,
                        "domain": domain,
                        "citations": n,
                        "example_url": ex["url"],
                        "example_title": ex["title"],
                        "suggestion": f"Ask {domain} to include {brand.name} (a listing, review "
                        "or correction). Outreach is manual.",
                    },
                )
            )
    return out


def _norm_claim(s: str) -> str:
    return re.sub(r"\W+", " ", s.casefold()).strip()


def _fact_fixes(session: Session, brand: Brand, run_id: int) -> list[Found]:
    rows = session.execute(
        select(Claim, Answer.engine)
        .join(Answer, Answer.id == Claim.answer_id)
        .where(Claim.brand_id == brand.id, Answer.run_id == run_id, Claim.verdict == "contradicted")
    ).all()
    groups: dict[str, list[tuple[Claim, str]]] = defaultdict(list)
    for claim, engine in rows:
        groups[_norm_claim(claim.claim_text)].append((claim, engine))
    out = []
    for norm, items in groups.items():
        first = items[0][0]
        engines = sorted({e for _, e in items})
        out.append(
            Found(
                key=f"fact:{norm[:180]}",
                kind="fact_fix",
                title=f'Correct: "{first.claim_text[:120]}"',
                reason=(
                    f'{len(items)} answer(s) from {", ".join(engines)} say "{first.claim_text}"; '
                    f'the knowledge base says "{(first.evidence_quote or "")[:160]}".'
                ),
                impact=round(len(items) * (1 + 0.25 * (len(engines) - 1)), 4),
                confidence=FACT_CONFIDENCE,
                suggested_format="faq",
                run_id=run_id,
                details={
                    "claim": first.claim_text,
                    "claim_ids": [c.id for c, _ in items],
                    "engines": engines,
                    "evidence_quote": first.evidence_quote,
                    "evidence_chunk_keys": list(first.evidence_chunk_keys),
                    "target_prompts": [],
                },
            )
        )
    return out


def find_opportunities(session: Session, brand: Brand, *, run_id: int | None = None) -> list[Found]:
    """All opportunities for the brand, best first (not yet stored)."""
    topics = {
        t.id: t.label for t in session.scalars(select(Topic).where(Topic.brand_id == brand.id))
    }
    run = run_id or latest_extracted_run(session, brand.id)
    found = _gaps(session, brand, run, topics) + _citations(session, brand, run, topics)
    fc_run = run_id or latest_factchecked_run(session, brand.id)
    if fc_run is not None:
        found += _fact_fixes(session, brand, fc_run)
    return sorted(found, key=lambda f: (-f.score, f.key))


def refresh_opportunities(
    session: Session, brand: Brand, *, run_id: int | None = None
) -> list[ContentOpportunity]:
    """Recompute and upsert; returns the brand's open opportunities, best first."""
    found = find_opportunities(session, brand, run_id=run_id)
    existing = {
        o.key: o
        for o in session.scalars(
            select(ContentOpportunity).where(ContentOpportunity.brand_id == brand.id)
        )
    }
    seen = set()
    for f in found:
        seen.add(f.key)
        row = existing.get(f.key) or ContentOpportunity(brand_id=brand.id, key=f.key, status="open")
        row.kind, row.title, row.reason = f.kind, f.title[:300], f.reason
        row.impact, row.confidence, row.score = f.impact, f.confidence, f.score
        row.suggested_format, row.run_id, row.details = f.suggested_format, f.run_id, f.details
        if row.status == "stale":
            row.status = "open"
        session.add(row)
    for key, row in existing.items():
        if key not in seen and row.status == "open":
            row.status = "stale"
    session.flush()
    return list(
        session.scalars(
            select(ContentOpportunity)
            .where(ContentOpportunity.brand_id == brand.id, ContentOpportunity.status != "stale")
            .order_by(ContentOpportunity.score.desc(), ContentOpportunity.id)
        )
    )
