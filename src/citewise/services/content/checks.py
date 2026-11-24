"""Deterministic checks on a content draft: the guardrails and the rubric caps.

The LLM judge scores the five rubric criteria; these checks can only *lower* a score
(final = min(judge, cap)), never raise it. That way a lenient or manipulated judge can't
pass a draft with an invented price, and the checks need no model to run.

Guardrails (any hit blocks the draft from final; see evals/rubrics/content.yaml):
- unsupported_claims: every factual sentence must cite `[kb:<chunk key>]` markers from
  the brief's passages (a marker covers the sentences before it in the same paragraph),
  every number in it must appear in a cited passage, rankings or superlatives must be
  stated there too, and most of its content words must come from the cited text (so a
  real marker on an invented claim still fails). Titles and headings can't carry
  markers, so any number or superlative in them must appear in some passage.
- competitor_defamation: a named company (tracked competitor, the gap's rival, or a
  company the passages name) in the same sentence as a pejorative. Matching is
  case-sensitive so "running on worn shoes" isn't about the brand "On". Paraphrased
  disparagement ("notoriously buggy") is beyond a regex; the judge's tone score covers it.
- fake_reviews_testimonials: attributed or long quotes, blockquotes, star ratings,
  "customers say" / "one customer told us".
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
# Share of a factual sentence's content words found in its cited passages. 0.35 lets
# ordinary paraphrase through ("bring your boards over from Trello") while a claim the
# passage doesn't make shares little with it. Not yet calibrated on live drafts.
SUPPORT_MIN_OVERLAP = 0.35
STUFFING_PER_300 = 3.0  # a 3-word phrase more often than this per 300 words is stuffing
AEO_CAP_BY_FAILURES = {0: 5, 1: 3, 2: 2}  # failed structure checks -> max aeo score

MARKER = re.compile(r"\[kb:([^\]]+)\]")
_NUMBER = re.compile(r"(?<![\w#])\$?\d[\d,]*(?:\.\d+)?%?")
_WORD = re.compile(r"[a-z0-9$%]+(?:[.,]\d+)*")
_STOP = frozenset(
    (
        *("a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for"),
        *("from", "has", "have", "how", "i", "if", "in", "is", "it", "its", "of", "on"),
        *("or", "our", "so", "than", "that", "the", "their", "them", "then", "there"),
        *("these", "they", "this", "to", "up", "us", "was", "we", "what", "when", "where"),
        *("which", "who", "why", "will", "with", "you", "your", "not", "no", "all", "any"),
        *("also", "more", "most", "per", "much", "many", "get", "use", "need", "should"),
    )
)
# "one" is left out: "one of the", "no one" are not counts.
_NUM_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "ninety": 90, "hundred": 100,
}  # fmt: skip
_NUM_WORD = re.compile(r"\b(" + "|".join(_NUM_WORDS) + r")\b", re.IGNORECASE)
_FACT_CUE = re.compile(
    r"\$|\d|%|\b(?:costs?|priced?|prices|pricing|includes?|included|offers?|supports?|"
    r"integrat\w*|imports?|importers?|plans?|trials?|warranty|ships?|shipping|returns?|"
    r"refund\w*|founded|built|made in|manufactured|sells?|sold|available|weighs?|drop|"
    r"sizes?|certified|soc ?2|gdpr|hipaa|compliant|encrypt\w*|users?|seats?|free|"
    r"discontinued|launched|headquartered|based in|unlike|cheaper|faster|more than|fewer|"
    r"syncs?|native|apps?)\b",
    re.IGNORECASE,
)
_SUPERLATIVE = re.compile(
    r"#\s?1\b|\bnumber one\b|\b(?:best|top|highest)[- ]rated\b|\bmost popular\b|"
    r"\bleading\b|\bfastest\b|\bcheapest\b|\bbest\b|\blargest\b|\bbiggest\b",
    re.IGNORECASE,
)
_PEJORATIVE = re.compile(
    r"\b(?:scam\w*|fraud\w*|rip-?off|liars?|lies|lying|cheats?|steals?|illegal|dangerous|"
    r"unsafe|garbage|terrible|awful|worst|junk|overcharg\w*|unreliable|shady|predatory|"
    r"falls? apart|cause[sd]? injur\w*|sells? your data|constantly down|crash(?:es)? constantly|"
    r"buggy|nightmare|useless|incompetent)\b",
    re.IGNORECASE,
)
_DASHES = "-" + chr(0x2013) + chr(0x2014)  # hyphen, en dash, em dash
_QUOTE = re.compile(r"[\"“]([^\"”]+)[\"”]")
_TESTIMONIAL = re.compile(
    r"[\"“][^\"”]{8,}[\"”]\s*(?:[" + _DASHES + r"]|,? (?:says|said|writes|wrote))"
    r"\s*[A-Z]|"
    r"\b(?:customers|users|runners|teams|clients|reviewers|agency owners)\s+"
    r"(?:say|said|tell us|told us|love|rave)\b|"
    r"\b(?:a|one|our|another) (?:happy )?(?:customer|user|client|runner|reviewer|agency owner)s?"
    r"\b[^.]{0,80}?\b(?:said|says|told|tells|wrote|writes)\b|"
    r"\b[A-Z][a-z]+,? (?:an?|our) [a-z ]{2,40}, (?:said|says|told us|wrote)\b|"
    r"★|\brated\s+\d(?:\.\d)?\s*(?:/|out of)\s*5\b|\b\d(?:\.\d)?\s*/\s*5\b|"
    r"\b\d(?:\.\d)?\s*stars?\b|testimonial",
    re.IGNORECASE,
)
_INJECTION = re.compile(
    r"\b(?:ignore|disregard) (?:the|all|any|previous|prior|your|these) "
    r"(?:brief|instructions?|rules|above|prompt|system|guidelines)\b|"
    r"\bsystem (?:note|prompt|message)\b|\binstead,? write\b|\b(?:promo|discount|coupon) code\b",
    re.IGNORECASE,
)
_CODE = re.compile(r"\b[Uu]se (?:the )?code [A-Z0-9]{4,}\b")  # case-sensitive on purpose
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
_QUESTION_START = re.compile(
    r"^\s*(?:is|are|does|do|did|can|could|should|will|how|what|which|why|where|when|who)\b",
    re.IGNORECASE,
)
_BULLET = re.compile(r"^(\s*(?:[-*+]|\d+\.)\s+)")
# A capitalized name in running text ("Like Altra it uses...", "competes with Nike, Hoka").
_NAME_IN_TEXT = re.compile(r"(?<=[a-z,;:] )([A-Z][A-Za-z0-9]+(?: [A-Z][A-Za-z0-9]+)?)")


@dataclass(frozen=True)
class Unit:
    """One checkable sentence (or table row) and where it lives in the draft."""

    where: str  # "summary", "sections[1].body", "faq[0].answer", "title", "...heading"
    text: str
    keys: tuple[str, ...] = ()  # its own markers, or the paragraph's next/previous ones

    @property
    def is_content(self) -> bool:
        return self.where.endswith(("body", "answer")) or self.where == "summary"


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
        """Units a guardrail rejected (removed by the last-resort redaction)."""
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
    """Numbers as bare values: "$1,200" -> "1200", "$12.00" -> "12", "50%" -> "50",
    "60-day" -> "60", "two months" -> "2"."""
    clean = MARKER.sub(" ", text)
    clean = _NUM_WORD.sub(lambda m: str(_NUM_WORDS[m.group(1).casefold()]), clean)
    out = set()
    for m in _NUMBER.findall(clean):
        v = m.strip("$%").replace(",", "")
        v = re.sub(r"\.0+$", "", v)
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


def _split_line(line: str) -> list[str]:
    line = _BULLET.sub("", line)
    line = _TRAILING_MARKERS.sub(r"\2\1", line)  # "fact. [kb:x]" -> "fact [kb:x]."
    out: list[str] = []
    for piece in _SPLIT.split(line):
        piece = piece.strip()
        if not piece:
            continue
        if _MARKERS_ONLY.match(piece) and out:
            out[-1] = f"{out[-1]} {piece}"
        else:
            out.append(piece)
    return out


def split_sentences(block: str) -> list[str]:
    """Sentences of a markdown block. Table rows and list items are units of their own;
    a run of markers after a full stop belongs to the sentence before it."""
    return [s for para in paragraphs(block) for s in para]


def paragraphs(block: str) -> list[list[str]]:
    """Sentences grouped by line (a Markdown paragraph, list item or table row)."""
    out: list[list[str]] = []
    for raw in block.splitlines():
        line = raw.strip()
        if not line or re.fullmatch(r"\|?[\s:|-]+\|?", line):
            continue  # blank or a table separator row
        out.append([line] if line.startswith("|") else _split_line(line))
    return [p for p in out if p]


def _with_keys(where: str, block: str) -> list[Unit]:
    """A sentence without markers is covered by the paragraph's next marker (or the
    previous one): Claude often cites once per paragraph."""
    units: list[Unit] = []
    for para in paragraphs(block):
        own = [tuple(marker_keys(s)) for s in para]
        for i, s in enumerate(para):
            keys = own[i]
            if not keys:
                after = next((k for k in own[i + 1 :] if k), ())
                before = next((k for k in reversed(own[:i]) if k), ())
                keys = after or before
            units.append(Unit(where, s, keys))
    return units


def units(draft: dict[str, Any]) -> list[Unit]:
    us = [Unit("title", draft.get("title", ""))]
    us += _with_keys("summary", draft.get("summary", ""))
    for i, sec in enumerate(draft.get("sections", [])):
        us.append(Unit(f"sections[{i}].heading", sec.get("heading", "")))
        us += _with_keys(f"sections[{i}].body", sec.get("body", ""))
    for i, qa in enumerate(draft.get("faq", [])):
        us.append(Unit(f"faq[{i}].question", qa.get("question", "")))
        us += _with_keys(f"faq[{i}].answer", qa.get("answer", ""))
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


def _contains_word(phrase: str, text: str) -> bool:
    return (
        re.search(r"(?<!\w)" + re.escape(phrase.casefold()) + r"(?!\w)", text.casefold())
        is not None
    )


def company_names(passages: dict[str, str], extra: list[str], exclude: list[str]) -> list[str]:
    """Companies a page might mention: tracked competitors and the rival (`extra`) plus
    capitalized names the passages use in running text (e.g. "Altra")."""
    found = {m.group(1) for t in passages.values() for m in _NAME_IN_TEXT.finditer(t)}
    names = {n for n in [*extra, *found] if n}
    drop = {e.casefold() for e in exclude if e}
    return sorted(n for n in names if n.casefold() not in drop)


def _name_regex(names: list[str]) -> re.Pattern[str] | None:
    """Case-sensitive. Names of 1-2 letters ("On") also must not start the sentence."""
    if not names:
        return None
    parts = []
    for n in sorted(names, key=len, reverse=True):
        esc = re.escape(n)
        parts.append(rf"(?<=\w )(?<!\. ){esc}(?!\w)" if len(n) <= 2 else rf"(?<!\w){esc}(?!\w)")
    return re.compile("|".join(parts))


# ---------------------------------------------------------------- the checks


def is_factual(sentence: str, entity_re: re.Pattern[str] | None = None) -> bool:
    """A claim worth checking: a number, a fact cue, a superlative, a marker, or a named
    company (the brand or another) in a declarative sentence."""
    text = strip_markers(sentence)
    if text.endswith("?") or (len(text.split()) < 4 and not numbers(text)):
        return False
    if _FACT_CUE.search(text) or _SUPERLATIVE.search(text) or marker_keys(sentence):
        return True
    return bool(entity_re and entity_re.search(text))


def support_problem(
    sentence: str, passages: dict[str, str], keys: tuple[str, ...] | list[str] | None = None
) -> dict[str, Any] | None:
    """Why a factual sentence isn't grounded in its cited passages, or None if it is."""
    keys = list(keys if keys is not None else marker_keys(sentence))
    text = strip_markers(sentence)
    if not keys:
        return {"sentence": sentence, "reason": "no_marker", "keys": []}
    unknown = [k for k in keys if k not in passages]
    if unknown:
        return {"sentence": sentence, "reason": "unknown_chunk", "keys": unknown}
    source = "\n".join(passages[k] for k in keys)
    missing = sorted(numbers(text) - numbers(source))
    if missing:
        return {"sentence": sentence, "reason": "number_not_in_source", "keys": keys,
                "numbers": missing}  # fmt: skip
    for m in _SUPERLATIVE.finditer(text):
        if not _contains_word(m.group(0), source):
            return {"sentence": sentence, "reason": "superlative_not_in_source", "keys": keys,
                    "term": m.group(0)}  # fmt: skip
    if _overlap(text, source) < SUPPORT_MIN_OVERLAP:
        return {"sentence": sentence, "reason": "not_in_source", "keys": keys}
    return None


def heading_problem(text: str, passages: dict[str, str]) -> dict[str, Any] | None:
    """Titles and headings can't cite, so their numbers and rankings must be in the KB.
    A question asserts nothing ("Is it true that X discontinued Y in 2025?")."""
    if text.rstrip().endswith("?") or _QUESTION_START.match(text):
        return None
    allkb = "\n".join(passages.values())
    missing = sorted(numbers(text) - numbers(allkb))
    if missing:
        return {"sentence": text, "reason": "heading_number_not_in_kb", "keys": [],
                "numbers": missing}  # fmt: skip
    for m in _SUPERLATIVE.finditer(text):
        if not _contains_word(m.group(0), allkb):
            return {"sentence": text, "reason": "heading_superlative_not_in_kb", "keys": [],
                    "term": m.group(0)}  # fmt: skip
    return None


def _is_testimonial(text: str, passages: dict[str, str]) -> bool:
    if text.lstrip().startswith(">"):
        return True
    allkb = "\n".join(passages.values())
    long_quotes = [q for q in _QUOTE.findall(text) if len(q.split()) >= 8 and q not in allkb]
    return bool(long_quotes) or bool(_TESTIMONIAL.search(text))


def _grams(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", strip_markers(text).casefold())
    return [
        " ".join(words[i : i + 3])
        for i in range(len(words) - 2)
        if words[i] not in _STOP and words[i + 2] not in _STOP
    ]


def _stuffing(text: str, keywords: list[str]) -> list[str]:
    """Target phrases (3-word runs from the brief's questions and target prompts) repeated
    more than STUFFING_PER_300 times per 300 words. Repeating a KB fact ("United States
    and Canada") isn't stuffing; repeating "best running shoes for wide feet" is."""
    words = re.findall(r"[a-z0-9]+", strip_markers(text).casefold())
    wanted = {g for k in keywords for g in _grams(k)}
    if len(words) < 60 or not wanted:
        return []
    grams = Counter(g for g in _grams(text) if g in wanted)
    per = len(words) / 300
    return sorted(g for g, n in grams.items() if n >= 4 and n / max(per, 1.0) > STUFFING_PER_300)


def _answers(question: str, heading: str, body: str, brand_terms: set[str]) -> bool:
    """The heading matches the question AND the text under it is about the question
    (shares a content word beyond the brand's name)."""
    if _overlap(question, heading) < 0.5:
        return False
    wanted = terms(question) - brand_terms
    return bool(strip_markers(body)) and (not wanted or bool(wanted & terms(body)))


def run_checks(
    draft: dict[str, Any],
    *,
    passages: dict[str, str],
    brief: dict[str, Any],
    voice: dict[str, Any],
    competitors: list[str],
    brand_names: list[str] | None = None,
) -> Checks:
    """All deterministic checks on one draft (a DraftOut dict)."""
    c = Checks()
    us = units(draft)
    brand_names = [b for b in (brand_names or []) if b]
    brand_terms = {t for b in brand_names for t in terms(b)}
    others = company_names(passages, [*competitors, brief.get("rival") or ""], brand_names)
    others_re = _name_regex(others)
    entity_re = _name_regex(others + brand_names)
    for u in us:
        text = strip_markers(u.text)
        if _INJECTION.search(text) or _CODE.search(text):
            c.injection.append(u.text)
            continue
        if others_re and others_re.search(text) and _PEJORATIVE.search(text):
            c.defamation.append(u.text)
        if _is_testimonial(u.text, passages):
            c.testimonials.append(u.text)
        if u.is_content:
            if is_factual(u.text, entity_re):
                problem = support_problem(u.text, passages, u.keys)
                if problem:
                    problem["where"] = u.where
                    c.unsupported.append(problem)
        else:
            problem = heading_problem(text, passages)
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
    c.stuffing = _stuffing(body, [*brief.get("questions", []), *brief.get("targets", [])])
    lowered = body.casefold()
    c.banned = [w for w in voice.get("banned", []) if re.search(rf"\b{re.escape(w)}\b", lowered)]
    content = [u for u in us if u.is_content]
    c.reading_grade = reading_grade([u.text for u in content])

    # AEO structure and coverage: a question counts only if the text under it answers it.
    questions = list(brief.get("questions") or [])
    sections = draft.get("sections", [])
    faq = draft.get("faq", [])
    pairs = [(s.get("heading", ""), s.get("body", "")) for s in sections]
    pairs += [(q.get("question", ""), q.get("answer", "")) for q in faq]
    answered = [q for q in questions if any(_answers(q, h, b, brand_terms) for h, b in pairs)]
    summary_sents = split_sentences(draft.get("summary", ""))
    primary = questions[0] if questions else draft.get("title", "")
    c.aeo = {
        "direct_answer": 1 <= len(summary_sents) <= 2
        and not _PREAMBLE.match(strip_markers(draft.get("summary", "")))
        and _overlap(primary, draft.get("summary", "")) >= 0.3,
        "headings_match_questions": bool(questions) and len(answered) / len(questions) >= 0.75,
        "faq_section": len(faq) >= 3
        and not any(re.search(r"as mentioned|see above", q.get("answer", ""), re.I) for q in faq),
    }
    if brief.get("format") == "comparison":
        c.aeo["comparison_table"] = any(
            "|" in s.get("body", "") and re.search(r"\|\s*:?-{3,}", s.get("body", ""))
            for s in sections
        )

    cited = {k for u in content for k in marker_keys(u.text)}  # headings don't count
    must = [k for k in brief.get("must_cover", []) if k in passages]
    c.coverage = {
        "questions_answered": f"{len(answered)}/{len(questions)}",
        "unanswered": [q for q in questions if q not in answered],
        "must_cover_cited": f"{len(cited & set(must))}/{len(must)}",
        "must_cover_missing": [k for k in must if k not in cited],
    }
    required = list(brief.get("required_facts") or [])
    stated = [f for f in required if any(_overlap(f, u.text) >= 0.6 for u in content)]
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
    sentences = [strip_markers(u.text).casefold() for u in content]
    dupes = len(sentences) - len(set(sentences))
    if (sentences and dupes / len(sentences) > 0.15) or _BOILERPLATE.search(body):
        caps["originality"] = 3
        issues["originality"].append("Remove repeated sentences and boilerplate phrases")
    c.caps, c.issues = caps, issues
    return c


def redact(draft: dict[str, Any], checks: Checks, *, fallback_title: str = "") -> dict[str, Any]:
    """Last resort after the revision loops: drop every unit a guardrail rejected. An
    offending heading takes its section with it; an offending title is replaced."""
    bad = checks.offending()

    def clean(block: str) -> str:
        kept_lines = []
        for line in block.splitlines():
            stripped = line.strip()
            if (
                not stripped
                or stripped.startswith("|")
                or re.fullmatch(r"\|?[\s:|-]+\|?", stripped)
            ):
                if stripped not in bad:
                    kept_lines.append(line)
                continue
            m = _BULLET.match(line)
            kept = [s for s in _split_line(line) if s not in bad]
            if kept:
                kept_lines.append((m.group(1) if m else "") + " ".join(kept))
        return "\n".join(kept_lines).strip()

    title = draft.get("title", "")
    out = {**draft, "title": fallback_title if title in bad else title,
           "summary": clean(draft.get("summary", ""))}  # fmt: skip
    out["sections"] = [
        {**s, "body": clean(s.get("body", ""))}
        for s in draft.get("sections", [])
        if s.get("heading", "") not in bad
    ]
    out["sections"] = [s for s in out["sections"] if strip_markers(s["body"])]
    out["faq"] = [
        {**q, "answer": clean(q.get("answer", ""))}
        for q in draft.get("faq", [])
        if q.get("question", "") not in bad
    ]
    out["faq"] = [q for q in out["faq"] if strip_markers(q["answer"])]
    return out
