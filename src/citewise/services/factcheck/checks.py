"""FactCheck steps: extract claims, verify each against the KB, trace contradicted ones to
the cited page that probably caused them.

Guards (an LLM can be wrong, and the answer text is untrusted web-derived content):
- **Claims must come from the answer.** Each claim's `answer_span` must appear in the
  answer as Stage B saw it (injected spans redacted), so a claim can't be invented or
  planted by an instruction in the answer.
- **No contradiction without a quote.** `supported`/`contradicted` verdicts must quote, near
  verbatim, a passage that was actually retrieved and cited by id. Otherwise the verdict is
  downgraded to `not_in_kb` and the downgrade is recorded. A false "the engine is wrong"
  is worse for a brand than a missed one.
- **Source tracing never fetches in mock mode.** It ranks the answer's citations by (1) an
  inline citation marker right after the claim, (2) overlap between the claim and the
  citation title, and (3) in live mode, overlap with the fetched page text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz

from citewise.providers.base import ChatProvider
from citewise.schemas.factcheck import ClaimList, ExtractedClaim, Verdict
from citewise.services.extract.mentions import MAX_ANSWER_CHARS, redact
from citewise.services.kb.retrieve import Hit, tokenize
from citewise.services.prompt_library import render
from citewise.services.structured import complete_structured

CLAIMS_PURPOSE = "factcheck.claims"
VERIFY_PURPOSE = "factcheck.verify"
MAX_CLAIMS_PER_ANSWER = 12
_MARKER = re.compile(r"\[(?:web:)?(\d+)\]")
_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    """For containment checks: no markdown emphasis, collapsed whitespace, casefolded,
    unified quotes."""
    s = s.replace("**", "").replace("__", "").replace("’", "'").replace("“", '"')
    s = s.replace("”", '"')
    return _WS.sub(" ", s).strip().casefold()


def contains(haystack: str, needle: str, *, fuzzy: float = 0.0) -> bool:
    """Is `needle` in `haystack` (normalized)? "..." in the needle allows elided parts."""
    h, n = _norm(haystack), _norm(needle).strip(" \"'")
    if not n:
        return False
    parts = [p.strip() for p in n.split("...") if p.strip()]
    if all(p in h for p in parts):
        return True
    return bool(fuzzy) and len(n) >= 12 and fuzz.partial_ratio(n, h) >= fuzzy


# ---------------------------------------------------------------- claims


def extract_claims(
    chat: ChatProvider, *, model: str, brand: str, aliases: list[str], text: str
) -> tuple[list[ExtractedClaim] | None, list[dict[str, Any]]]:
    """Validated claims (None if the LLM failed) and the ones dropped, with why."""
    seen_by_model = redact(text)[:MAX_ANSWER_CHARS]
    rendered = render("factcheck/claims", "v1", brand=brand, aliases=aliases, answer=seen_by_model)
    out = complete_structured(
        chat, rendered.messages, ClaimList, model=model, purpose=CLAIMS_PURPOSE,
        raise_on_budget=True,
    )  # fmt: skip
    if out is None:
        return None, []
    kept: list[ExtractedClaim] = []
    dropped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for c in out.claims:
        if not contains(seen_by_model, c.answer_span, fuzzy=92):
            dropped.append({"reason": "span_not_in_answer", "claim": c.claim})
        elif _norm(c.claim) in seen:
            dropped.append({"reason": "duplicate", "claim": c.claim})
        elif len(kept) >= MAX_CLAIMS_PER_ANSWER:
            dropped.append({"reason": "over_limit", "claim": c.claim})
        else:
            seen.add(_norm(c.claim))
            kept.append(c)
    return kept, dropped


# ---------------------------------------------------------------- verification


@dataclass
class Checked:
    verdict: str
    chunk_keys: list[str]
    quote: str | None
    rationale: str
    details: dict[str, Any] = field(default_factory=dict)


def validate_verdict(v: Verdict, hits: list[Hit]) -> Checked:
    """Enforce "never contradicted (or supported) without a real quote"."""
    texts = {h.chunk_key: h.text for h in hits}
    if v.verdict == "not_in_kb":
        return Checked("not_in_kb", [], None, v.rationale)
    cited = [k for k in dict.fromkeys(v.evidence_chunk_ids) if k in texts]
    quote = (v.evidence_quote or "").strip()
    holders = [k for k in cited if quote and contains(texts[k], quote)]
    details: dict[str, Any] = {}
    if not holders and quote:
        # The quote is real but the model cited the wrong passage id: point at the
        # retrieved passage that actually holds it, and say so.
        holders = [k for k in texts if contains(texts[k], quote)]
        if holders:
            details["evidence_ids_corrected_from"] = list(v.evidence_chunk_ids)
    if not holders:
        reason = "no_quote" if not quote else "quote_not_in_evidence"
        return Checked(
            "not_in_kb", [], None, v.rationale,
            {"downgraded_from": v.verdict, "reason": reason, "model_quote": quote or None},
        )  # fmt: skip
    return Checked(v.verdict, holders, quote, v.rationale, details)


def verify_claim(
    chat: ChatProvider, *, model: str, brand: str, claim: str, hits: list[Hit]
) -> Checked | None:
    """None when the LLM failed (both attempts); raises BudgetExceededError on refusal."""
    if not hits:
        return Checked("not_in_kb", [], None, "No knowledge-base passage matched the claim.",
                       {"reason": "no_retrieval_hits"})  # fmt: skip
    rendered = render(
        "factcheck/verify", "v1", brand=brand, claim=claim,
        passages=[{"id": h.chunk_key, "text": h.text} for h in hits],
    )  # fmt: skip
    out = complete_structured(
        chat, rendered.messages, Verdict, model=model, purpose=VERIFY_PURPOSE, raise_on_budget=True
    )
    return None if out is None else validate_verdict(out, hits)


# ---------------------------------------------------------------- source tracing


@dataclass
class CitationRef:
    id: int
    position: int  # 1-based, as cited inline: [n]
    url: str
    title: str | None
    domain: str


@dataclass
class Trace:
    citation_id: int
    confidence: float
    reason: str  # inline_marker | title_match | page_text
    scores: dict[int, float] = field(default_factory=dict)


def _overlap(claim: str, other: str) -> float:
    """Share of the claim's content terms that appear in `other`."""
    c = set(tokenize(claim))
    return len(c & set(tokenize(other))) / len(c) if c else 0.0


def markers_after(answer: str, span: str, *, window: int = 60) -> list[int]:
    """Citation numbers attached to `span`: inside it, or right after it in the same
    sentence ("... $25 per user [5].")."""
    flat = answer.replace("**", "")
    words = span.replace("**", "").split()
    if not words:
        return []
    m = re.search(r"\s+".join(re.escape(w) for w in words), flat, re.IGNORECASE)
    if m is None:
        return []
    after = flat[m.end() : m.end() + window]
    cut = re.search(r"[.!?](\s|$)|\n", after)
    # A marker before the sentence ends belongs to this claim; the next sentence's don't.
    tail = flat[m.start() : m.end()] + (after[: cut.end()] if cut else after)
    return [int(n) for n in _MARKER.findall(tail)]


def trace_source(
    claim: str,
    span: str,
    answer: str,
    citations: list[CitationRef],
    page_texts: dict[int, str] | None = None,
) -> Trace | None:
    """The citation that most likely caused a (contradicted) claim, or None."""
    if not citations:
        return None
    marked = set(markers_after(answer, span))
    scores: dict[int, tuple[float, str]] = {}
    for c in citations:
        title_sim = _overlap(claim, f"{c.title or ''} {c.domain} {c.url}")
        best = (0.15 + 0.6 * title_sim, "title_match") if title_sim > 0 else (0.0, "title_match")
        if c.position in marked:
            best = (0.95 if title_sim >= 0.3 else 0.85, "inline_marker")
        if page_texts and c.id in page_texts:
            page = 0.2 + 0.7 * _overlap(claim, page_texts[c.id])
            if page > best[0]:
                best = (page, "page_text")
        scores[c.id] = best
    cid, (conf, reason) = max(scores.items(), key=lambda kv: (kv[1][0], -kv[0]))
    if conf < 0.25:
        return None
    return Trace(cid, round(conf, 3), reason, {k: round(v[0], 3) for k, v in scores.items()})
