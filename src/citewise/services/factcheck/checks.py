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


_NUMBER = re.compile(r"\$?\d+(?:[.,]\d+)*%?")
_NEGATION = re.compile(r"\b(?:no|not|never|without|only|n't|none|neither|nor)\b")
QUOTE_MIN_WORDS = 5
_ELIDED_GAP = 200  # max characters between consecutive "..." parts of a quote


def _crosses_sentence(gap: str) -> bool:
    """An elided part of a quote may skip words, not jump to another sentence: "The Pro
    plan costs ... $25" must not stitch Pro's sentence to Enterprise's price."""
    return len(gap) > _ELIDED_GAP or re.search(r"[.!?]\s|\n", gap) is not None


def critical(s: str) -> set[str]:
    """Tokens that change a fact's meaning if altered: numbers, prices, percentages and
    negations. A fuzzy match may differ in punctuation or wording, never in these."""
    n = _norm(s)
    return set(_NUMBER.findall(n)) | set(_NEGATION.findall(n.replace("n't", " n't")))


def contains(haystack: str, needle: str, *, fuzzy: float = 0.0) -> bool:
    """Is `needle` in `haystack` (normalized)?

    - "..." in the needle elides text: the parts must appear in order, each within
      _ELIDED_GAP characters of the previous one (no stitching across a document).
    - `fuzzy` (a rapidfuzz partial_ratio) tolerates small wording or punctuation
      differences, but the matched window must contain exactly the needle's numbers and
      negations: "$15" never matches an answer that says "$12".
    """
    h, n = _norm(haystack), _norm(needle).strip(" \"'")
    if not n:
        return False
    parts = [p.strip() for p in n.split("...") if p.strip()]
    pos, prev_end = 0, None
    for part in parts:
        i = h.find(part, pos)
        if i < 0 or (prev_end is not None and _crosses_sentence(h[prev_end:i])):
            break
        pos = prev_end = i + len(part)
    else:
        return True
    if not fuzzy or len(n) < 12:
        return False
    align = fuzz.partial_ratio_alignment(n, h)
    if align is None or align.score < fuzzy:
        return False
    # Same numbers and negations in the matched part of the answer as in the needle:
    # "$15" vs "$12" or "offers" vs "does not offer" is a different fact, not a typo.
    return critical(n) == critical(h[align.dest_start : align.dest_end])


def _body(chunk_text: str) -> str:
    """A chunk without its "Doc title > Section" prefix line: the prefix locates the
    passage, it doesn't state a fact, so a quote of it proves nothing."""
    head, _, body = chunk_text.partition("\n")
    return body if body and " > " in head else chunk_text


def content_terms(text: str, brand_terms: set[str]) -> set[str]:
    return set(tokenize(text)) - brand_terms


def brand_terms_for(brand: str, aliases: list[str] | tuple[str, ...] = ()) -> set[str]:
    return {t for name in [brand, *aliases] for t in tokenize(name)}


# ---------------------------------------------------------------- claims


def claim_matches_span(claim: str, span: str, brand_terms: set[str]) -> bool:
    """The claim that gets verified must say what its span says: the same numbers and
    negations, and at least half of its content terms."""
    if not critical(claim) <= critical(span):
        return False
    terms = content_terms(claim, brand_terms)
    return not terms or len(terms & content_terms(span, brand_terms)) / len(terms) >= 0.5


def extract_claims(
    chat: ChatProvider, *, model: str, brand: str, aliases: list[str], text: str
) -> tuple[list[ExtractedClaim] | None, list[dict[str, Any]]]:
    """Validated claims (None if the LLM failed) and the ones dropped, with why."""
    bterms = brand_terms_for(brand, aliases)
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
        elif not claim_matches_span(c.claim, c.answer_span, bterms):
            dropped.append({"reason": "claim_not_in_span", "claim": c.claim})
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


def validate_verdict(
    v: Verdict, hits: list[Hit], *, claim: str = "", brand_terms: set[str] | None = None
) -> Checked:
    """Enforce "never contradicted (or supported) without a real quote": the quote must
    be at least QUOTE_MIN_WORDS words and found verbatim (ellipses within one sentence) in
    the body of a retrieved passage, not its heading.

    Deliberately *not* required: word overlap between quote and claim. It was tried, and
    on the held-out set it discarded correct contradictions from the live model, whose
    quotes use pronouns and paraphrase ("It costs $140" vs "sells for around $120",
    "founded ... in Portland" vs "got its start in Seattle"). Whether a real quote means
    what the verdict says is the model's judgement, which the held-out eval measures.
    `claim` and `brand_terms` are kept for callers and future checks."""
    texts = {h.chunk_key: _body(h.text) for h in hits}
    if v.verdict == "not_in_kb":
        return Checked("not_in_kb", [], None, v.rationale)
    cited = [k for k in dict.fromkeys(v.evidence_chunk_ids) if k in texts]
    quote = (v.evidence_quote or "").strip()
    if quote and len(quote.replace("...", " ").split()) < QUOTE_MIN_WORDS:
        return _downgrade(v, "quote_too_short", quote)
    holders = [k for k in cited if quote and contains(texts[k], quote)]
    details: dict[str, Any] = {}
    if not holders and quote:
        # The quote is real but the model cited the wrong passage id: point at the
        # retrieved passage that actually holds it, and say so.
        holders = [k for k in texts if contains(texts[k], quote)]
        if holders:
            details["evidence_ids_corrected_from"] = list(v.evidence_chunk_ids)
    if not holders:
        return _downgrade(v, "no_quote" if not quote else "quote_not_in_evidence", quote)
    return Checked(v.verdict, holders, quote, v.rationale, details)


def _downgrade(v: Verdict, reason: str, quote: str | None) -> Checked:
    return Checked(
        "not_in_kb", [], None, v.rationale,
        {"downgraded_from": v.verdict, "reason": reason, "model_quote": quote or None},
    )  # fmt: skip


def verify_claim(
    chat: ChatProvider,
    *,
    model: str,
    brand: str,
    claim: str,
    hits: list[Hit],
    aliases: list[str] | tuple[str, ...] = (),
) -> Checked | None:
    """None when the LLM failed (both attempts); raises BudgetExceededError on refusal."""
    if not hits:
        return Checked("not_in_kb", [], None, "No knowledge-base passage matched the claim.",
                       {"reason": "no_retrieval_hits"})  # fmt: skip
    rendered = render(
        "factcheck/verify", "v2", brand=brand, claim=claim,
        passages=[{"id": h.chunk_key, "text": h.text} for h in hits],
    )  # fmt: skip
    out = complete_structured(
        chat, rendered.messages, Verdict, model=model, purpose=VERIFY_PURPOSE, raise_on_budget=True
    )
    if out is None:
        return None
    return validate_verdict(out, hits, claim=claim, brand_terms=brand_terms_for(brand, aliases))


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


def _overlap(claim: str, other: str, brand_terms: set[str] | None = None) -> float:
    """Share of the claim's content terms (brand name excluded) that appear in `other`.
    Without the exclusion, the brand's own homepage title ("Taskflow | Project management")
    "matches" every claim about Taskflow."""
    c = content_terms(claim, brand_terms or set())
    return len(c & set(tokenize(other))) / len(c) if c else 0.0


_ABBREV = re.compile(r"\b(?:e\.g|i\.e|etc|vs|approx|incl)\.$", re.IGNORECASE)
_TRAILING_MARKERS = re.compile(r"^(?:\s*\[(?:web:)?\d+\])+")


def _sentence_end(after: str) -> int:
    """Offset just past the end of the current sentence, skipping "e.g." and the like."""
    for m in re.finditer(r"[.!?](?=\s|$)|\n", after):
        if m.group(0) == "." and _ABBREV.search(after[: m.end()]):
            continue
        return m.end()
    return len(after)


def markers_after(answer: str, span: str, *, window: int = 160) -> list[int]:
    """Citation numbers attached to `span`: inside it, after it in the same sentence
    ("... $25 per user [5]."), or immediately after the sentence's full stop
    ("... per month. [5]"). The next sentence's markers are not this claim's."""
    flat = answer.replace("**", "")
    words = span.replace("**", "").split()
    if not words:
        return []
    m = re.search(r"\s+".join(re.escape(w) for w in words), flat, re.IGNORECASE)
    if m is None:
        return []
    after = flat[m.end() : m.end() + window]
    end = _sentence_end(after)
    trailing = _TRAILING_MARKERS.match(after[end:])
    tail = flat[m.start() : m.end()] + after[:end] + (trailing.group(0) if trailing else "")
    return [int(n) for n in _MARKER.findall(tail)]


def trace_source(
    claim: str,
    span: str,
    answer: str,
    citations: list[CitationRef],
    page_texts: dict[int, str] | None = None,
    *,
    brand_terms: set[str] | None = None,
    marker_urls: dict[int, str] | None = None,
) -> Trace | None:
    """The citation that most likely caused a (contradicted) claim, or None.

    `marker_urls` maps an inline marker number to the URL it denotes when the provider's
    numbering differs from our deduplicated citation positions (Perplexity's [web:N] ids).
    """
    if not citations:
        return None
    markers = set(markers_after(answer, span))
    by_url = {c.url: c.id for c in citations}
    if marker_urls:
        marked_ids = {by_url[u] for n in markers if (u := marker_urls.get(n)) in by_url}
    else:
        marked_ids = {c.id for c in citations if c.position in markers}
    scores: dict[int, tuple[float, str]] = {}
    for c in citations:
        title_sim = _overlap(claim, f"{c.title or ''} {c.domain} {c.url}", brand_terms)
        best = (0.15 + 0.6 * title_sim, "title_match") if title_sim > 0 else (0.0, "title_match")
        if c.id in marked_ids:
            best = (0.95 if title_sim >= 0.3 else 0.85, "inline_marker")
        if page_texts and c.id in page_texts:
            page = 0.2 + 0.7 * _overlap(claim, page_texts[c.id], brand_terms)
            if page > best[0]:
                best = (page, "page_text")
        scores[c.id] = best
    cid, (conf, reason) = max(scores.items(), key=lambda kv: (kv[1][0], -kv[0]))
    if conf < 0.25:
        return None
    return Trace(cid, round(conf, 3), reason, {k: round(v[0], 3) for k, v in scores.items()})
