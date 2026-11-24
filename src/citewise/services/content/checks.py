"""Deterministic checks on a content draft: the guardrails and the rubric caps.

The LLM judge scores the five rubric criteria; these checks can only *lower* a score
(final = min(judge, cap)), never raise it. That way a lenient or manipulated judge can't
pass a draft with an invented price, and the checks need no model to run.

Guardrails (any hit blocks the draft from final; see evals/rubrics/content.yaml):
- unsupported_claims: every factual sentence must cite `[kb:<chunk key>]` markers from
  the brief's passages, every number in it must appear in a cited passage, rankings or
  superlatives must be stated there too, and most of its content words must come from
  the cited text (so a real marker on an invented claim still fails).
- competitor_defamation: a competitor named in the same sentence as a pejorative.
- fake_reviews_testimonials: attributed quotes, star ratings, "customers say".
- injection: instruction-like text copied from a passage ("ignore the brief", promo codes).
- max_length: over the brief's word count +20%, the voice file's cap, or 2,500 words;
  or a repeated 3-word phrase that looks like keyword stuffing.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from citewise.schemas.content import CRITERIA

HARD_MAX_WORDS = 2500
LENGTH_TOLERANCE = 1.2
SUPPORT_MIN_OVERLAP = 0.4  # share of a factual sentence's content words found in its sources
STUFFING_PER_300 = 3.0  # a 3-word phrase more often than this per 300 words is stuffing
AEO_CAP_BY_FAILURES = {0: 5, 1: 3, 2: 2}  # failed structure checks -> max aeo score

MARKER = re.compile(r"\[kb:([^\]]+)\]")
_NUMBER = re.compile(r"(?<![\w#])\$?\d[\d,]*(?:\.\d+)?%?")
_WORD = re.compile(r"[a-z0-9$%]+(?:[.,]\d+)*")
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "so",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "up",
        "us",
        "was",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
        "not",
        "no",
        "all",
        "any",
        "also",
        "more",
        "most",
        "per",
    ]
)
_FACT_CUE = re.compile(
    r"\$|\d|%|\b(?:costs?|priced?|prices|pricing|includes?|included|offers?|supports?|"
    r"integrat\w*|imports?|importers?|plans?|trials?|warranty|ships?|shipping|returns?|"
    r"refund\w*|founded|made in|manufactured|sells?|sold|available|weighs?|drop|sizes?|"
    r"certified|soc ?2|gdpr|hipaa|compliant|encrypt\w*|users?|seats?|free|discontinued|"
    r"launched|headquartered|based in|unlike|cheaper|faster|more than|fewer|only)\b",
    re.IGNORECASE,
)
_SUPERLATIVE = re.compile(
    r"#\s?1\b|\bnumber one\b|\b(?:best|top|highest)[- ]rated\b|\bmost popular\b|"
    r"\bleading\b|\bfastest\b|\bcheapest\b|\bbest\b|\bonly\b",
    re.IGNORECASE,
)
_PEJORATIVE = re.compile(
    r"\b(?:scam\w*|fraud\w*|rip-?off|liars?|lies|lying|cheats?|steals?|illegal|dangerous|"
    r"unsafe|garbage|terrible|awful|worst|junk|overcharg\w*|unreliable|shady|predatory|"
    r"falls? apart|cause[sd]? injur\w*|sells? your data|constantly down|crash(?:es)? constantly)\b",
    re.IGNORECASE,
)
_DASHES = "-" + chr(0x2013) + chr(0x2014)  # hyphen, en dash, em dash
_TESTIMONIAL = re.compile(
    r"[\"“][^\"”]{8,}[\"”]\s*(?:[" + _DASHES + r"]|,? (?:says|said|writes|wrote))\s*[A-Z]|"
    r"\b(?:customers|users|runners|teams|clients|reviewers|agency owners)\s+"
    r"(?:say|said|tell us|told us|love|rave|report|agree)\b|"
    r"★|\brated\s+\d(?:\.\d)?\s*(?:/|out of)\s*5\b|\b\d(?:\.\d)?\s*stars?\b|testimonial",
    re.IGNORECASE,
)
_INJECTION = re.compile(
    r"\bignore (?:the|all|any|previous|prior|your|these)\b|\bdisregard\b|"
    r"\bsystem (?:note|prompt|message)\b|\binstead,? write\b|\b(?:promo|discount|coupon) code\b|"
    r"\buse (?:the )?code\b",
    re.IGNORECASE,
)
_PREAMBLE = re.compile(
    r"^(?:in today'?s|when it comes to|choosing (?:a|the)|are you|have you ever|"
    r"looking for|if you'?re looking|whether you)",
    re.IGNORECASE,
)
_BOILERPLATE = re.compile(
    r"in today'?s fast-paced|in conclusion|look no further|without further ado|"
    r"at the end of the day|it'?s no secret",
    re.IGNORECASE,
)
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"“(\[A-Z0-9$])")
_TRAILING_MARKERS = re.compile(r"([.!?])((?:\s*\[kb:[^\]]+\])+)")
_MARKERS_ONLY = re.compile(r"^(?:\s*\[kb:[^\]]+\])+[.]?\s*$")


@dataclass(frozen=True)
class Unit:
    """One checkable sentence (or table row) and where it lives in the draft."""

    where: str  # "summary", "sections[1].body", "faq[0].answer", "title"
    text: str


@dataclass
class Checks:
    unsupported: list[dict[str, Any]] = field(default_factory=list)
    defamation: list[str] = field(default_factory=list)
    testimonials: list[str] = field(default_factory=list)
    injection: list[str] = field(default_factory=list)
    banned: list[str] = field(default_factory=list)
    words: int = 0
    word_limit: int = HARD_MAX_WORDS
    stuffing: list[str] = field(default_factory=list)
    reading_grade: float = 0.0
    aeo: dict[str, bool] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    caps: dict[str, int] = field(default_factory=dict)
    issues: dict[str, list[str]] = field(default_factory=dict)

    @property
    def guardrails(self) -> dict[str, bool]:
        """Guardrail -> passed."""
        return {
            "unsupported_claims": not self.unsupported,
            "competitor_defamation": not self.defamation,
            "fake_reviews_testimonials": not self.testimonials,
            "injection": not self.injection,
            "max_length": self.words <= self.word_limit and not self.stuffing,
        }

    @property
    def guardrails_ok(self) -> bool:
        return all(self.guardrails.values())

    def offending(self) -> set[str]:
        """Sentences a guardrail rejected (removed by the last-resort redaction)."""
        return (
            {u["sentence"] for u in self.unsupported}
            | set(self.defamation)
            | set(self.testimonials)
            | set(self.injection)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "guardrails": self.guardrails,
            "unsupported": self.unsupported,
            "defamation": self.defamation,
            "testimonials": self.testimonials,
            "injection": self.injection,
            "banned_words": self.banned,
            "words": self.words,
            "word_limit": self.word_limit,
            "stuffing": self.stuffing,
            "reading_grade": round(self.reading_grade, 1),
            "aeo": self.aeo,
            "coverage": self.coverage,
            "caps": self.caps,
            "issues": self.issues,
        }


# ---------------------------------------------------------------- text helpers


def _stem(w: str) -> str:
    for suffix in ("ing", "ies", "es", "ed", "s"):
        if len(w) > len(suffix) + 2 and w.endswith(suffix):
            return w[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return w


def terms(text: str) -> set[str]:
    clean = MARKER.sub(" ", text.casefold())
    return {_stem(w.strip(".,")) for w in _WORD.findall(clean) if w.strip(".,") not in _STOP}


def numbers(text: str) -> set[str]:
    """Numbers as bare values: "$1,200" -> "1200", "50%" -> "50", "60-day" -> "60"."""
    clean = MARKER.sub(" ", text)
    out = set()
    for m in _NUMBER.findall(clean):
        v = m.strip("$%").replace(",", "")
        v = v[:-2] if v.endswith(".0") else v
        if v:
            out.add(v)
    return out


def marker_keys(text: str) -> list[str]:
    keys: list[str] = []
    for m in MARKER.findall(text):
        keys += [k.strip().removeprefix("kb:") for k in m.split(",") if k.strip()]
    return keys


def strip_markers(text: str) -> str:
    return re.sub(r"\s+([.,;:!?])", r"\1", MARKER.sub("", text)).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9$%][\w$%'.,-]*", strip_markers(text)))


def split_sentences(block: str) -> list[str]:
    """Sentences of a markdown block. Table rows and list items are units of their own;
    a run of markers after a full stop belongs to the sentence before it."""
    out: list[str] = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line or re.fullmatch(r"\|?[\s:|-]+\|?", line):
            continue  # blank or a table separator row
        if line.startswith("|"):
            out.append(line)
            continue
        line = re.sub(r"^(?:[-*+]|\d+\.)\s+", "", line)
        line = _TRAILING_MARKERS.sub(r"\2\1", line)  # "fact. [kb:x]" -> "fact [kb:x]."
        for piece in _SPLIT.split(line):
            piece = piece.strip()
            if not piece:
                continue
            if _MARKERS_ONLY.match(piece) and out:
                out[-1] = f"{out[-1]} {piece}"
            else:
                out.append(piece)
    return out


def units(draft: dict[str, Any]) -> list[Unit]:
    us = [Unit("title", draft.get("title", ""))]
    us += [Unit("summary", s) for s in split_sentences(draft.get("summary", ""))]
    for i, sec in enumerate(draft.get("sections", [])):
        us.append(Unit(f"sections[{i}].heading", sec.get("heading", "")))
        us += [Unit(f"sections[{i}].body", s) for s in split_sentences(sec.get("body", ""))]
    for i, qa in enumerate(draft.get("faq", [])):
        us.append(Unit(f"faq[{i}].question", qa.get("question", "")))
        us += [Unit(f"faq[{i}].answer", s) for s in split_sentences(qa.get("answer", ""))]
    return [u for u in us if u.text.strip()]


def body_text(draft: dict[str, Any]) -> str:
    parts = [draft.get("summary", "")]
    parts += [f"{s.get('heading', '')}\n{s.get('body', '')}" for s in draft.get("sections", [])]
    parts += [f"{q.get('question', '')}\n{q.get('answer', '')}" for q in draft.get("faq", [])]
    return "\n".join(parts)


def _syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups) - (1 if w.endswith("e") and len(groups) > 1 else 0)
    return max(1, n)


def reading_grade(sentences: list[str]) -> float:
    """Flesch-Kincaid grade over prose sentences (table rows excluded)."""
    prose = [strip_markers(s) for s in sentences if not s.startswith("|")]
    words = [w for s in prose for w in re.findall(r"[A-Za-z]+", s)]
    if not prose or not words:
        return 0.0
    syllables = sum(_syllables(w) for w in words)
    return 0.39 * len(words) / len(prose) + 11.8 * syllables / len(words) - 15.59


def _overlap(a: str, b: str) -> float:
    ta = terms(a)
    return len(ta & terms(b)) / len(ta) if ta else 0.0


# ---------------------------------------------------------------- the checks


def is_factual(sentence: str) -> bool:
    text = strip_markers(sentence)
    if text.endswith("?") or (len(text.split()) < 4 and not numbers(text)):
        return False
    return bool(_FACT_CUE.search(text) or _SUPERLATIVE.search(text)) or bool(marker_keys(sentence))


def support_problem(sentence: str, passages: dict[str, str]) -> dict[str, Any] | None:
    """Why a factual sentence isn't grounded in its cited passages, or None if it is."""
    keys = marker_keys(sentence)
    text = strip_markers(sentence)
    if not keys:
        return {"sentence": sentence, "reason": "no_marker", "keys": []}
    unknown = [k for k in keys if k not in passages]
    if unknown:
        return {"sentence": sentence, "reason": "unknown_chunk", "keys": unknown}
    source = "\n".join(passages[k] for k in keys)
    missing = sorted(numbers(text) - numbers(source))
    if missing:
        return {
            "sentence": sentence,
            "reason": "number_not_in_source",
            "keys": keys,
            "numbers": missing,
        }
    for m in _SUPERLATIVE.finditer(text):
        if m.group(0).casefold() not in source.casefold():
            return {
                "sentence": sentence,
                "reason": "superlative_not_in_source",
                "keys": keys,
                "term": m.group(0),
            }
    if _overlap(text, source) < SUPPORT_MIN_OVERLAP:
        return {"sentence": sentence, "reason": "not_in_source", "keys": keys}
    return None


def _stuffing(text: str) -> list[str]:
    words = [w for w in re.findall(r"[a-z0-9]+", strip_markers(text).casefold())]
    if len(words) < 60:
        return []
    grams = Counter(
        " ".join(words[i : i + 3])
        for i in range(len(words) - 2)
        if words[i] not in _STOP and words[i + 2] not in _STOP
    )
    per = len(words) / 300
    return sorted(g for g, n in grams.items() if n >= 4 and n / max(per, 1.0) > STUFFING_PER_300)


def _matches_question(question: str, candidates: list[str]) -> bool:
    return any(_overlap(question, c) >= 0.5 for c in candidates)


def run_checks(
    draft: dict[str, Any],
    *,
    passages: dict[str, str],
    brief: dict[str, Any],
    voice: dict[str, Any],
    competitors: list[str],
) -> Checks:
    """All deterministic checks on one draft (a DraftOut dict)."""
    c = Checks()
    us = units(draft)
    rivals = [r for r in competitors if r]
    rival_re = (
        re.compile(r"\b(?:" + "|".join(re.escape(r) for r in rivals) + r")\b", re.IGNORECASE)
        if rivals
        else None
    )
    for u in us:
        text = strip_markers(u.text)
        if _INJECTION.search(text):
            c.injection.append(u.text)
            continue
        if rival_re and rival_re.search(text) and _PEJORATIVE.search(text):
            c.defamation.append(u.text)
        if _TESTIMONIAL.search(u.text):
            c.testimonials.append(u.text)
        content = u.where.endswith(("body", "answer")) or u.where == "summary"
        if content and is_factual(u.text):
            problem = support_problem(u.text, passages)
            if problem:
                problem["where"] = u.where
                c.unsupported.append(problem)

    body = body_text(draft)
    c.words = word_count(body)
    c.word_limit = min(
        HARD_MAX_WORDS,
        int(voice.get("max_words") or HARD_MAX_WORDS),
        int(int(brief.get("word_count") or HARD_MAX_WORDS) * LENGTH_TOLERANCE),
    )
    c.stuffing = _stuffing(body)
    lowered = body.casefold()
    c.banned = [w for w in voice.get("banned", []) if re.search(rf"\b{re.escape(w)}\b", lowered)]
    prose = [u.text for u in us if u.where.endswith(("body", "answer")) or u.where == "summary"]
    c.reading_grade = reading_grade(prose)

    # AEO structure
    questions = list(brief.get("questions") or [])
    headings = [s.get("heading", "") for s in draft.get("sections", [])]
    faq = draft.get("faq", [])
    faq_qs = [q.get("question", "") for q in faq]
    summary_sents = split_sentences(draft.get("summary", ""))
    primary = questions[0] if questions else draft.get("title", "")
    c.aeo = {
        "direct_answer": 1 <= len(summary_sents) <= 2
        and not _PREAMBLE.match(strip_markers(draft.get("summary", "")))
        and _overlap(primary, draft.get("summary", "")) >= 0.3,
        "headings_match_questions": bool(questions)
        and sum(_matches_question(q, headings + faq_qs) for q in questions) / len(questions)
        >= 0.75,
        "faq_section": len(faq) >= 3
        and not any(re.search(r"as mentioned|see above", q.get("answer", ""), re.I) for q in faq),
    }
    if brief.get("format") == "comparison":
        c.aeo["comparison_table"] = any(
            "|" in s.get("body", "") and re.search(r"\|\s*:?-{3,}", s.get("body", ""))
            for s in draft.get("sections", [])
        )

    # Coverage
    answered = [q for q in questions if _matches_question(q, headings + faq_qs)]
    cited = {k for u in us for k in marker_keys(u.text)}
    must = [k for k in brief.get("must_cover", []) if k in passages]
    c.coverage = {
        "questions_answered": f"{len(answered)}/{len(questions)}",
        "unanswered": [q for q in questions if q not in answered],
        "must_cover_cited": f"{len(cited & set(must))}/{len(must)}",
        "must_cover_missing": [k for k in must if k not in cited],
    }
    required = list(brief.get("required_facts") or [])
    content_units = [
        u.text for u in us if u.where.endswith(("body", "answer")) or u.where == "summary"
    ]
    stated = [f for f in required if any(_overlap(f, t) >= 0.6 for t in content_units)]
    c.coverage["required_facts_missing"] = [f for f in required if f not in stated]
    shares = [len(answered) / len(questions) if questions else 1.0]
    if required:
        shares.append(len(stated) / len(required))
    shares.append(len(cited & set(must)) / len(must) if must else 1.0)

    # Caps and the issues the revise step is told about
    issues: dict[str, list[str]] = {k: [] for k in CRITERIA}
    caps = dict.fromkeys(CRITERIA, 5)
    if c.unsupported or c.testimonials or c.injection:
        caps["factuality"] = 2 if c.unsupported or c.testimonials else 1
        issues["factuality"] += [
            f"Unsupported ({u['reason']}): {strip_markers(u['sentence'])[:160]}"
            for u in c.unsupported
        ]
        issues["factuality"] += [f"Fake review/testimonial: {t[:160]}" for t in c.testimonials]
        issues["factuality"] += [f"Instruction-like text, remove: {t[:160]}" for t in c.injection]
    coverage_score = 1 + round(4 * sum(shares) / len(shares))
    if c.words > c.word_limit:
        coverage_score = min(coverage_score, 3)
        issues["coverage"].append(f"Too long: {c.words} words, limit {c.word_limit}")
    if c.stuffing:
        coverage_score = min(coverage_score, 3)
        issues["coverage"].append(f"Keyword stuffing: {', '.join(c.stuffing[:3])}")
    if c.coverage["required_facts_missing"]:
        coverage_score = min(coverage_score, 2)  # a correction page that doesn't correct
    caps["coverage"] = coverage_score
    issues["coverage"] += [
        f"State the correct fact: {f}" for f in c.coverage["required_facts_missing"]
    ]
    issues["coverage"] += [f"Answer this question: {q}" for q in c.coverage["unanswered"]]
    issues["coverage"] += [
        f"Use must-cover passage [kb:{k}]" for k in c.coverage["must_cover_missing"]
    ]
    if c.banned or c.defamation:
        caps["tone"] = 2
        issues["tone"] += [f"Banned word: {w}" for w in c.banned]
        issues["tone"] += [f"Disparages a competitor, remove: {d[:160]}" for d in c.defamation]
    target_grade = float(voice.get("reading_level") or 99)
    if c.reading_grade > target_grade + 2:
        caps["tone"] = min(caps["tone"], 3)
        issues["tone"].append(
            f"Reading grade {c.reading_grade:.1f} is above the target {target_grade:g}"
        )
    # Rubric anchors: 5 = all structure checks pass; 3 = one lapse; worse below that.
    caps["aeo_structure"] = AEO_CAP_BY_FAILURES.get(sum(not ok for ok in c.aeo.values()), 1)
    issues["aeo_structure"] += [f"Fix: {k.replace('_', ' ')}" for k, ok in c.aeo.items() if not ok]
    # Body text only: an FAQ question that repeats a section heading is good AEO practice.
    sentences = [
        strip_markers(u.text).casefold()
        for u in us
        if u.where.endswith(("body", "answer")) or u.where == "summary"
    ]
    dupes = len(sentences) - len(set(sentences))
    if (sentences and dupes / len(sentences) > 0.15) or _BOILERPLATE.search(body):
        caps["originality"] = 3
        issues["originality"].append("Remove repeated sentences and boilerplate phrases")
    c.caps, c.issues = caps, issues
    return c


def redact(draft: dict[str, Any], checks: Checks) -> dict[str, Any]:
    """Last resort after the revision loops: drop every sentence a guardrail rejected."""
    bad = checks.offending()

    def clean(block: str) -> str:
        kept_lines = []
        for line in block.splitlines():
            if line.strip().startswith("|"):
                if line.strip() not in bad:
                    kept_lines.append(line)
                continue
            kept = [s for s in split_sentences(line) if s not in bad]
            if kept or not line.strip():
                kept_lines.append(" ".join(kept))
        return "\n".join(kept_lines).strip()

    out = {**draft, "summary": clean(draft.get("summary", ""))}
    out["sections"] = [{**s, "body": clean(s.get("body", ""))} for s in draft.get("sections", [])]
    out["sections"] = [s for s in out["sections"] if s["body"]]
    out["faq"] = [
        {**q, "answer": clean(q.get("answer", ""))}
        for q in draft.get("faq", [])
        if q.get("question", "") not in bad
    ]
    out["faq"] = [q for q in out["faq"] if q["answer"]]
    return out
