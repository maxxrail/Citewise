"""Simulated Stage B extractor for mock mode (purpose "extract.mentions").

Real Stage B is an LLM reading the rendered prompt. In mock mode there is no LLM, and a
schema-valid placeholder ("mock-1a2b") would make every offline extraction, metric and eval
meaningless. Reading the answer simulator's ground truth would make the extractor eval a
fake 100%. So this module *imitates a reasonable model* using only what a real model gets:
the rendered prompt (brand, aliases, known competitors, answer text). It reads context like
a model would (capitalization, list structure, contrast words, recommendation cues) and
uses general knowledge a real model has (which names are review sites, not products).

It is deliberately written independently of Stage A (services/extract/mentions.py), so the
two stages make different mistakes and the merge and the evals have something to measure.
Its F1 on the golden set measures this simulator, not a production model: the live number
needs `CITEWISE_MODE=live citewise eval extract`.
"""

from __future__ import annotations

import json
import re
from typing import Any

from citewise.providers.base import Message

PURPOSE = "extract.mentions"

# General knowledge a real model has: these are sources, not products.
PUBLISHERS = frozenset(
    {
        "runner's world", "runners world", "runrepeat", "reddit", "youtube", "g2", "capterra",
        "amazon", "forbes", "forbes advisor", "pcmag", "zapier", "outside", "wirecutter",
        "believe in the run", "trustpilot", "google", "wikipedia",
    }
)  # fmt: skip
# Capitalized words that start sentences or lists but are never companies.
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "as",
        "at",
        "both",
        "but",
        "by",
        "for",
        "from",
        "here",
        "however",
        "i",
        "if",
        "in",
        "it",
        "its",
        "many",
        "most",
        "my",
        "of",
        "on",
        "or",
        "overall",
        "people",
        "prices",
        "some",
        "that",
        "the",
        "their",
        "these",
        "they",
        "this",
        "those",
        "to",
        "when",
        "where",
        "which",
        "while",
        "with",
        "you",
        "your",
        "yes",
        "no",
        "also",
        "best",
        "top",
        "note",
        "tip",
        "pros",
        "cons",
        "verdict",
    ]
)
MAKER_WORDS = frozenset({"athletic", "running", "labs", "software", "sports", "apparel", "hq"})
POSITIVE = (
    "praised", "well liked", "well-liked", "solid", "strong", "favourite", "favorite",
    "popular", "versatile", "dependable", "durable", "great", "polished", "simple", "easy",
    "clean", "fast", "lively", "comfortable", "responsive", "well suited", "well regarded",
    "highly", "distinctive", "standard for", "packs many", "roomy", "max cushioning",
    "offers many", "excellent", "love", "reliable", "best", "widely adopted", "intuitive",
)  # fmt: skip
NEGATIVE = (
    "although", "however", "can feel", "expensive", "confusing", "overwhelming", "limited",
    "narrow", "less stable", "wears quickly", "steeper", "heavy for", "takes time", "snug",
    "smaller", "not the lightest", "firm", "avoid", "steer clear", "disappoint", "clunky",
    "buggy", "overpriced", "poor", "worse", "lacks", "falls apart",
)  # fmt: skip
WINNER_CUES = re.compile(
    r"(?:lean(?:s)? towards|go with|top pick is|my pick is|recommend|winner is|best bet is)"
    r"\s+(?:\*\*)?([A-Z][\w.&'’-]*(?:\s+[A-Z][\w.&'’-]*)?)",
)
DISCOURAGE = re.compile(r"avoid|steer clear|wouldn't recommend|would not recommend|skip ", re.I)
LIST_INTRO_RECOMMENDS = re.compile(r"best|recommend|top|options|picks|ways to save|consider", re.I)
_LIST_ITEM = re.compile(r"^[ \t]*(?:\d+[.)]|[-*•])[ \t]+(.*)$", re.MULTILINE)
_CAPSEQ = re.compile(r"(?<![\w'])([A-Z][\w.&'’-]*(?:[ ]+[A-Z][\w.&'’-]*)*)")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ENUM = re.compile(
    r"(?:consider|alternatives?(?: include| are| like)?|such as|like|also)\s+([^.\n]+)"
)


def _field(user: str, label: str) -> Any:
    m = re.search(rf"^{re.escape(label)}: (.*)$", user, re.MULTILINE)
    return json.loads(m.group(1)) if m else None


def _parse_prompt(messages: list[Message]) -> tuple[str, list[str], list[str], str]:
    user = messages[-1]["content"]
    brand = _field(user, "Brand") or ""
    aliases = _field(user, "Brand aliases") or []
    competitors = _field(user, "Known competitors") or []
    m = re.search(r"Answer to analyse \(a JSON string\):\n(.*)\Z", user, re.DOTALL)
    answer = json.loads(m.group(1).strip()) if m else ""
    return brand, aliases, competitors, answer


def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.casefold().replace("'s", "").replace(".com", "")).strip()


class _Found:
    def __init__(self, canonical: str, surface: str, pos: int) -> None:
        self.canonical, self.surface, self.positions = canonical, surface, [pos]


def _known_hits(answer: str, name: str, canonical: str) -> list[tuple[int, str]]:
    """Occurrences of a known name that a careful reader would count as the company."""
    hits = []
    pattern = re.compile(rf"(?<![\w]){re.escape(name)}(?:'s|’s)?(?![\w])", re.IGNORECASE)
    for m in pattern.finditer(answer):
        surface, after = m.group(0), answer[m.end() : m.end() + 15]
        before = answer[max(0, m.start() - 12) : m.start()]
        if name.casefold() in ("on", "monday", "notion"):
            if not surface[:1].isupper():
                continue  # "on", "monday", "a notion"
            if name.casefold() == "on" and re.match(r"\s+[a-z]", after):
                continue  # "On the other hand"
            if name.casefold() == "monday" and (
                re.search(r"(?:on|every|next|last|this|by|until)\s+$", before, re.I)
                or re.match(r"\s+(?:morning|night|through)", after)
            ):
                continue
        hits.append((m.start(), surface))
    return hits


def _unknown_candidates(answer: str) -> list[tuple[int, str]]:
    """Names in places where answers put products: bold spans, list subjects, and
    enumerations after "consider" / "alternatives" / "such as"."""
    spans: list[tuple[int, str]] = []
    for m in _BOLD.finditer(answer):
        spans.append((m.start(1), m.group(1)))
    for m in _LIST_ITEM.finditer(answer):
        body = m.group(1).lstrip("*")
        if c := _CAPSEQ.match(body):
            spans.append((m.start(1) + (len(m.group(1)) - len(body)), c.group(1)))
    for m in _ENUM.finditer(answer):
        offset = m.start(1)
        for part in re.split(r",\s*|\s+and\s+|\s+or\s+", m.group(1)):
            stripped = part.strip(" *")
            if c := _CAPSEQ.match(stripped):
                spans.append((offset + m.group(1).find(stripped), c.group(1)))
    out = []
    for pos, seq in spans:
        tokens = seq.replace("**", "").split()
        if not tokens or tokens[0].casefold().strip(".,:") in STOPWORDS:
            continue
        maker = (
            tokens[:2] if len(tokens) >= 2 and tokens[1].casefold() in MAKER_WORDS else tokens[:1]
        )
        name = " ".join(maker).strip(".,:;'’")
        name = re.sub(r"['’]s$", "", name)
        if len(name) >= 2 and _key(name) not in PUBLISHERS:
            out.append((pos, name))
    return out


def _contexts(answer: str, positions: list[int]) -> list[str]:
    """The list item or sentence around each occurrence."""
    out = []
    for pos in positions:
        start = answer.rfind("\n", 0, pos) + 1
        end = answer.find("\n", pos)
        line = answer[start : end if end != -1 else len(answer)]
        if not _LIST_ITEM.match(line):
            rel = pos - start
            s = line.rfind(". ", 0, rel)
            e = line.find(". ", rel)
            line = line[s + 2 if s != -1 else 0 : e + 1 if e != -1 else len(line)]
        out.append(line.strip())
    return out


def _sentiment(contexts: list[str]) -> tuple[float, str]:
    text = " ".join(contexts).casefold()
    p = sum(text.count(w) for w in POSITIVE)
    n = sum(text.count(w) for w in NEGATIVE)
    if p and n:
        return round(0.8 * (p - n) / (p + n), 2), "mixed"
    if p:
        return (0.9 if p >= 3 else 0.7), "positive"
    if n:
        return (-0.9 if n >= 2 else -0.7), "negative"
    return 0.0, "neutral"


def simulate_extraction(messages: list[Message]) -> dict[str, Any]:
    brand, aliases, competitors, answer = _parse_prompt(messages)
    known: list[tuple[str, str]] = [(n, brand) for n in [brand, *aliases]]
    known += [(c, c) for c in competitors]
    # A model knows "Monday.com" is usually written "Monday".
    known += [(re.sub(r"\.(?:com|io|so|app|ai)$", "", n), c) for n, c in known if "." in n]
    found: dict[str, _Found] = {}
    for name, canonical in known:
        for pos, surface in _known_hits(answer, name, canonical):
            f = found.setdefault(canonical, _Found(canonical, surface, pos))
            if pos not in f.positions:
                f.positions.append(pos)
    known_keys = {_key(n): c for n, c in known}
    for pos, name in _unknown_candidates(answer):
        k = _key(name)
        known_as = known_keys.get(k) or next(
            (c for n, c in known if k.startswith(_key(n) + " ")), None
        )  # "Nike Pegasus" -> Nike
        if known_as is not None:
            continue  # known names were found above, with context checks
        f = found.setdefault(name, _Found(name, name, pos))
        if pos not in f.positions:
            f.positions.append(pos)

    # Structural rank: subject (first entity) of each list item, in order.
    ranks: dict[str, int] = {}
    n_items = 0
    first_item = None
    for m in _LIST_ITEM.finditer(answer):
        s, e = m.span()
        first_item = s if first_item is None else first_item
        inside = [(min(p for p in f.positions if s <= p < e), c) for c, f in found.items()
                  if any(s <= p < e for p in f.positions)]  # fmt: skip
        if inside:
            n_items += 1
            ranks.setdefault(min(inside)[1], n_items)
    intro = answer[:first_item] if first_item is not None else ""
    list_recommends = bool(LIST_INTRO_RECOMMENDS.search(intro)) and not re.search(
        r"\bboth\b", intro, re.I
    )
    winners = {_key(m.group(1)) for m in WINNER_CUES.finditer(answer)}
    first_sentence = answer.strip().split("\n", 1)[0]

    mentions = []
    for canonical, f in sorted(found.items(), key=lambda kv: min(kv[1].positions)):
        f.positions.sort()
        contexts = _contexts(answer, f.positions)
        score, label = _sentiment(contexts)
        discouraged = any(DISCOURAGE.search(c) for c in contexts)
        rank = ranks.get(canonical)
        recommended = (
            any(w == _key(canonical) or w.startswith(_key(canonical) + " ") for w in winners)
            or (rank is not None and list_recommends and label != "negative")
            or (
                not ranks
                and first_sentence.casefold().startswith(f.surface.casefold())
                and label in ("positive", "mixed")
            )
        ) and not discouraged
        if discouraged and label != "negative":
            score, label = -0.7, "negative"
        mentions.append(
            {
                "entity": f.surface,
                "canonical": canonical,
                "rank": rank,
                "sentiment": score,
                "sentiment_label": label,
                "is_recommended": recommended,
                "snippet": contexts[0][:300],
            }
        )
    return {"mentions": mentions}
