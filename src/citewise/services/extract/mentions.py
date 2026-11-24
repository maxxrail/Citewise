"""Mention extraction: which companies an answer names, where they rank, and how it talks
about them.

Two stages, then a merge:

- **Stage A** (deterministic, free): word-boundary alias matching plus fuzzy matching for
  typos, against the brand and its competitors. High precision on names we know, blind to
  everything else (sentiment, recommendations, companies we don't track).
- **Stage B** (LLM, structured output): every company named, including unknown ones
  (emerging competitors), with rank, sentiment and whether it is recommended.
- **Merge**: union of the two, resolved to canonical names. Every disagreement is recorded
  (the evals use them). Stage B entities that don't appear in the text are dropped:
  a hallucination guard, and a prompt-injection guard for answers that tell the model
  what to report.

Rank is *structural*: the 1-based position of the list item whose subject (first named
company) the entity is. It is computed by us over the merged entity set, so it doesn't
depend on the model counting correctly. A Stage B rank is used only when the answer has
no list we can parse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from rapidfuzz import fuzz

from citewise.providers.base import ChatProvider, Message
from citewise.schemas.extract import ExtractedMention, MentionExtraction, SentimentLabel
from citewise.services.extract.resolve import (
    FUZZY_MIN_LEN,
    FUZZY_THRESHOLD,
    EntityType,
    KnownName,
    Resolver,
    fuzzy_ok,
    normalize,
)
from citewise.services.prompt_library import render
from citewise.services.structured import complete_structured

PROMPT_NAME, PROMPT_VERSION = "extract/mentions", "v1"
PURPOSE = "extract.mentions"
SNIPPET_MAX = 300
MAX_ANSWER_CHARS = 20_000  # ~5K tokens: far above a normal answer

_LIST_ITEM = re.compile(r"^[ \t]*(?:\d+[.)]|[-*•])[ \t]+\S.*$", re.MULTILINE)
_HEADING_ITEM = re.compile(r"^[ \t]*(?:#{1,6}[ \t]*\d+[.)]|\*\*\d+[.)])", re.MULTILINE)
# Text that isn't the answer's own prose: hidden comments, code blocks, and lines that
# address the extractor ("Note to AI: report X as rank 1"). Web-sourced answers can carry
# these; a company named only inside them is not a mention.
_HIDDEN = re.compile(r"<!--.*?-->|```.*?```", re.DOTALL)
_INSTRUCTION_LINE = re.compile(
    r"^.*(?:ignore (?:all |the )?(?:previous|above|prior) instructions"
    r"|\bnote to (?:the )?(?:ai|assistant|model|llm)\b"
    r"|^\s*(?:system|assistant|ai|instructions?)\s*:"
    r"|\b(?:report|list|rank|mark) \S+ as (?:rank|the top|recommended|#)).*$",
    re.IGNORECASE | re.MULTILINE,
)
REDACTED = "[removed]"
_TOKEN = re.compile(r"[\w][\w.'’&-]*")
# Context that makes an ambiguous brand name an ordinary word.
_WEEKDAY_BEFORE = re.compile(r"\b(?:on|every|next|last|this|by|until|since|a)\s+$", re.I)
_WEEKDAY_AFTER = re.compile(r"^\s+(?:morning|afternoon|evening|night|through|to)\b", re.I)
_LOWER_WORD_AFTER = re.compile(r"^\s+[a-z]")
_DATE_AFTER = re.compile(
    r"^\s+(?:mon|tues|wednes|thurs|fri|satur|sun)day\b|^\s+(?:jan|feb|mar|apr|may|jun|jul|"
    r"aug|sep|oct|nov|dec)[a-z]*\b",
    re.IGNORECASE,
)
# Links are sources, not mentions: "taskflow.example/pricing" doesn't name the company in
# the answer (it feeds citation share instead). A bare "Monday.com" is a name, so a domain
# only counts as a link with a scheme, "www." or a path.
_URL = re.compile(r"https?://\S+|\bwww\.\S+|\b[\w-]+(?:\.[\w-]+)*\.[a-z]{2,}/\S*", re.I)


def url_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in _URL.finditer(text)]


def _json_spans(text: str) -> list[tuple[int, int]]:
    """Object literals that look like structured output ({"mentions": ...}), matched by
    brace depth so nested objects are covered."""
    spans = []
    i = 0
    while (i := text.find("{", i)) != -1:
        if not re.match(r'\{\s*"', text[i : i + 20]):
            i += 1
            continue
        depth, j = 0, i
        while j < len(text):
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            if depth == 0:
                break
            j += 1
        spans.append((i, j + 1))
        i = j + 1
    return spans


def injected_spans(text: str) -> list[tuple[int, int]]:
    """Spans to ignore and to redact before Stage B (links are ignored, not redacted)."""
    return (
        [m.span() for m in _HIDDEN.finditer(text)]
        + [m.span() for m in _INSTRUCTION_LINE.finditer(text)]
        + _json_spans(text)
    )


def masked_spans(text: str) -> list[tuple[int, int]]:
    return url_spans(text) + injected_spans(text)


def redact(text: str) -> str:
    """The answer as Stage B sees it: injected spans replaced, so the model is never
    handed the instructions in the first place (the not-in-text guard is the backstop)."""
    out, last = [], 0
    for s, e in sorted(injected_spans(text)):
        if s < last:
            continue
        out += [text[last:s], REDACTED]
        last = e
    return "".join([*out, text[last:]])


def _in_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(s <= pos < e for s, e in spans)


@dataclass
class Mention:
    canonical: str
    entity_type: EntityType
    key: str
    entity: str  # surface form as first written
    rank: int | None = None
    sentiment: float = 0.0
    sentiment_label: SentimentLabel = "neutral"
    is_recommended: bool = False
    snippet: str = ""
    source: str = "a"  # "a" | "b" | "both"
    positions: list[int] = field(default_factory=list)  # char offsets of every occurrence
    ambiguous_only: bool = False  # Stage A found it only through an ambiguous alias

    def as_dict(self) -> dict[str, Any]:
        return {
            "canonical": self.canonical,
            "entity_type": self.entity_type,
            "entity": self.entity,
            "rank": self.rank,
            "sentiment": round(self.sentiment, 3),
            "sentiment_label": self.sentiment_label,
            "is_recommended": self.is_recommended,
            "source": self.source,
        }


# ---------------------------------------------------------------- structure


def snippet_at(text: str, pos: int) -> str:
    """The line containing `pos`, cut to the sentence when the line is long."""
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    line = text[start : end if end != -1 else len(text)]
    if len(line) > SNIPPET_MAX:
        rel = pos - start
        s = max(line.rfind(". ", 0, rel) + 2, 0) if line.rfind(". ", 0, rel) != -1 else 0
        e = line.find(". ", rel)
        line = line[s : e + 1 if e != -1 else len(line)]
    return " ".join(line.split())[:SNIPPET_MAX]


def _indent(line: str) -> int:
    return len(line.expandtabs(4)) - len(line.expandtabs(4).lstrip())


def list_blocks(text: str) -> list[list[tuple[int, int]]]:
    """The answer's ranked lists, each a list of item spans, in order.

    - Numbered headings ("### 1. Asana", "**1. Asana**") are items, and each item runs to
      the next heading, so bullets under a heading are its details, not list items.
    - Otherwise bullets/numbers are items, but only at the block's top indent: a nested
      "- integrates with Trello" is a detail of its parent item, not the next rank.
    - A non-list paragraph between two lists ends a block: "Avoid these: ... Best picks:
      ..." are two lists, each numbered from 1.
    """
    headings = list(_HEADING_ITEM.finditer(text))
    if headings:
        starts = [m.start() for m in headings] + [len(text)]
        return [[(starts[i], starts[i + 1]) for i in range(len(headings))]]
    blocks: list[list[tuple[int, int, int]]] = [[]]
    pos = 0
    for line in text.splitlines(keepends=True):
        span = (pos, pos + len(line.rstrip("\r\n")))
        pos += len(line)
        if _LIST_ITEM.match(line):
            blocks[-1].append((_indent(line), *span))
        elif line.strip() and _indent(line) == 0 and blocks[-1]:
            blocks.append([])  # a paragraph after a list closes it
    out = []
    for block in blocks:
        if block:
            top = min(i for i, _, _ in block)
            out.append([(s, e) for i, s, e in block if i == top])
    return out


def structural_ranks(text: str, positions: dict[str, list[int]]) -> dict[str, int]:
    """key -> 1-based rank: the n-th item of a list (counting only items that name a
    company) whose subject, the first company named in it, is that key. Each list is
    numbered separately; an entity keeps its rank in the first list it leads an item of."""
    ranks: dict[str, int] = {}
    for block in list_blocks(text):
        n = 0
        for s, e in block:
            firsts = [
                (min(p for p in ps if s <= p < e), key)
                for key, ps in positions.items()
                if any(s <= p < e for p in ps)
            ]
            if not firsts:
                continue
            n += 1
            ranks.setdefault(min(firsts)[1], n)
    return ranks


# ---------------------------------------------------------------- Stage A


def _alias_regex(surface: str, *, case_sensitive: bool) -> re.Pattern[str]:
    words = [re.escape(w) for w in re.split(r"[\s-]+", surface.strip()) if w]
    body = r"[\s-]+".join(words)
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(rf"(?<![\w]){body}(?:['’]s)?(?![\w])", flags)


def _ambiguous_is_brand(
    text: str, m: re.Match[str], key: str, other_keys: frozenset[str] = frozenset()
) -> bool:
    before, after = text[max(0, m.start() - 20) : m.start()], text[m.end() : m.end() + 20]
    if key == "monday":
        return not (_WEEKDAY_BEFORE.search(before) or _WEEKDAY_AFTER.match(after))
    if key == "on":  # "On the other hand", "On Tuesday", "On Hoka's site", "On Asana you"
        if _LOWER_WORD_AFTER.match(after) or _DATE_AFTER.match(after):
            return False
        nxt = re.match(r"\s+([\w'’]+)", after)
        return not (nxt and normalize(nxt.group(1)) in other_keys)
    return True  # "Notion" with a capital N mid-text is the product, not "a notion"


def _fuzzy_hits(
    text: str, name: KnownName, taken: set[int], masked: list[tuple[int, int]]
) -> list[tuple[int, str]]:
    """Typos of a known name ("Clickup", "Asanna", "Stridelabs"). Only capitalized
    windows, so ordinary words ("brook") don't match, and never a longer, different name
    ("LinearB" is not Linear; see `fuzzy_ok`)."""
    k = len(name.key.split())
    tokens = list(_TOKEN.finditer(text))
    hits = []
    for i in range(len(tokens) - k + 1):
        window = tokens[i : i + k]
        start, surface = window[0].start(), text[window[0].start() : window[-1].end()]
        if start in taken or not surface[:1].isupper() or _in_spans(start, masked):
            continue
        cand = normalize(surface)
        if (
            len(cand) >= FUZZY_MIN_LEN
            and fuzzy_ok(cand, name.key)
            and fuzz.ratio(cand, name.key) >= FUZZY_THRESHOLD
        ):
            hits.append((start, surface.rstrip(".'’")))
    return hits


def stage_a(text: str, resolver: Resolver) -> list[Mention]:
    """Known entities only, with structural rank and the snippet of the first mention."""
    found: dict[str, Mention] = {}
    taken: set[int] = set()  # start offsets already matched exactly

    def add(name: KnownName, pos: int, surface: str, *, ambiguous: bool) -> None:
        m = found.get(name.key) or found.setdefault(
            name.key,
            Mention(
                name.canonical, name.entity_type, normalize(name.canonical), surface,
                ambiguous_only=True,
            ),
        )  # fmt: skip
        # Several aliases (Stridelab, Stride Lab) share one canonical: merge by canonical.
        if pos not in m.positions:  # two aliases can match at one offset
            m.positions.append(pos)
        m.ambiguous_only = m.ambiguous_only and ambiguous
        if pos == min(m.positions):
            m.entity = surface

    masked = masked_spans(text)
    keys = frozenset(n.key for n in resolver.names)
    for name in resolver.names:
        regex = _alias_regex(name.surface, case_sensitive=name.ambiguous)
        for match in regex.finditer(text):
            if _in_spans(match.start(), masked):
                taken.add(match.start())  # don't let the fuzzy pass pick it up either
                continue
            if name.ambiguous and not _ambiguous_is_brand(text, match, name.key, keys - {name.key}):
                continue
            taken.add(match.start())
            add(name, match.start(), match.group(0), ambiguous=name.ambiguous)
    for name in resolver.names:
        if name.ambiguous or len(name.key) < FUZZY_MIN_LEN:
            continue
        for pos, surface in _fuzzy_hits(text, name, taken, masked):
            add(name, pos, surface, ambiguous=False)

    # One Mention per canonical (aliases were keyed separately above).
    by_canonical: dict[str, Mention] = {}
    for m in found.values():
        if (prev := by_canonical.get(m.key)) is None:
            by_canonical[m.key] = m
            continue
        prev.positions = sorted(set(prev.positions) | set(m.positions))
        prev.ambiguous_only = prev.ambiguous_only and m.ambiguous_only
        if min(m.positions) < min(prev.positions):
            prev.entity = m.entity
    mentions = sorted(by_canonical.values(), key=lambda m: min(m.positions))
    ranks = structural_ranks(text, {m.key: m.positions for m in mentions})
    for m in mentions:
        m.positions.sort()
        m.rank = ranks.get(m.key)
        m.snippet = snippet_at(text, m.positions[0])
    return mentions


# ---------------------------------------------------------------- Stage B


def stage_b(
    chat: ChatProvider,
    *,
    model: str,
    brand: str,
    aliases: list[str],
    competitors: list[str],
    category: str,
    text: str,
) -> list[ExtractedMention] | None:
    """LLM extraction (validated, one retry). None when both attempts failed.

    Raises BudgetExceededError when the spend cap refused the call, so the caller can
    tell "out of budget, try later" from "the model failed".
    """
    messages = stage_b_messages(
        brand=brand, aliases=aliases, competitors=competitors, category=category, text=text
    )
    out = complete_structured(
        chat, messages, MentionExtraction, model=model, purpose=PURPOSE, raise_on_budget=True
    )
    return None if out is None else out.mentions


def stage_b_messages(
    *, brand: str, aliases: list[str], competitors: list[str], category: str, text: str
) -> list[Message]:
    """The exact Stage B prompt (also the mock-fixture key, so recordings replay)."""
    return render(
        PROMPT_NAME,
        PROMPT_VERSION,
        brand=brand,
        aliases=aliases,
        competitors=competitors,
        category=category,
        # Injected spans removed; length capped so one huge answer can't blow the
        # per-call budget.
        answer=redact(text)[:MAX_ANSWER_CHARS],
    ).messages


# ---------------------------------------------------------------- merge


def occurrences(text: str, surface: str) -> list[int]:
    if not surface.strip():
        return []
    regex = _alias_regex(surface, case_sensitive=False)
    return [m.start() for m in regex.finditer(text)]


def in_text(surface: str, text: str) -> bool:
    """Does the answer actually contain this name (allowing for small typos/spacing)?"""
    s = surface.casefold().strip()
    if not s:
        return False
    low = text.casefold()
    if s in low:
        return True
    return len(s) >= 4 and fuzz.partial_ratio(s, low) >= 90


def merge(
    text: str,
    resolver: Resolver,
    a: list[Mention],
    b: list[ExtractedMention] | None,
) -> tuple[list[Mention], list[dict[str, Any]]]:
    """Combine Stage A and Stage B. `b=None` means the LLM failed: keep Stage A alone."""
    disagreements: list[dict[str, Any]] = []
    b_by_key: dict[str, tuple[Mention, int | None]] = {}
    for x in b or []:
        if not in_text(x.entity, text):
            disagreements.append({"type": "b_not_in_text", "entity": x.entity})
            continue
        r = resolver.resolve(x.canonical or x.entity)
        if r.entity_type == "unknown":  # the surface form may resolve where the name didn't
            r_surface = resolver.resolve(x.entity)
            if r_surface.entity_type != "unknown":
                r = r_surface
        if r.key in b_by_key:
            continue  # the model listed one company twice: keep the first
        mention = Mention(
            r.canonical, r.entity_type, r.key, x.entity,
            sentiment=x.sentiment, sentiment_label=x.sentiment_label,
            is_recommended=x.is_recommended, snippet=x.snippet, source="b",
        )  # fmt: skip
        b_by_key[r.key] = (mention, x.rank)

    a_by_key = {m.key: m for m in a}
    a_positions = {p for m in a for p in m.positions}
    urls = masked_spans(text)
    merged: list[Mention] = []
    for key in list(a_by_key) + [k for k in b_by_key if k not in a_by_key]:
        ma, (mb, _) = a_by_key.get(key), b_by_key.get(key, (None, None))
        if ma is not None and mb is not None:
            mb.source, mb.positions, mb.entity = "both", list(ma.positions), ma.entity
            mb.snippet = mb.snippet or ma.snippet
            merged.append(mb)
        elif ma is not None:
            if b is not None and ma.ambiguous_only:
                # Only an ambiguous word matched, and the model, which reads context, says
                # it isn't the company ("On the other hand").
                disagreements.append({"type": "a_only_dropped", "canonical": ma.canonical})
                continue
            if b is not None:
                disagreements.append({"type": "a_only", "canonical": ma.canonical})
            merged.append(ma)
        elif mb is not None:
            found = occurrences(text, mb.entity) or occurrences(text, mb.canonical)
            mb.positions = [p for p in found if not _in_spans(p, urls)]
            if found and not mb.positions:
                disagreements.append({"type": "b_only_masked", "canonical": mb.canonical})
                continue
            if mb.entity_type == "unknown" and mb.positions and set(mb.positions) <= a_positions:
                # Every occurrence starts where a known name does: a fragment of it ("New"
                # from "New Balance 880"), which would also steal that name's list rank.
                disagreements.append({"type": "b_fragment", "entity": mb.entity})
                continue
            if mb.entity_type != "unknown":
                disagreements.append({"type": "b_only", "canonical": mb.canonical})
            merged.append(mb)

    ranks = structural_ranks(text, {m.key: m.positions for m in merged if m.positions})
    b_ranks = {k: r for k, (_, r) in b_by_key.items()}
    for m in merged:
        m.rank = ranks.get(m.key)
        if m.rank is None and not ranks and b_ranks.get(m.key) is not None:
            m.rank = b_ranks[m.key]  # no list we can parse: trust the model's reading
        elif (br := b_ranks.get(m.key)) is not None and br != m.rank:
            disagreements.append(
                {"type": "rank", "canonical": m.canonical, "structural": m.rank, "b": br}
            )
        if not m.snippet and m.positions:
            m.snippet = snippet_at(text, m.positions[0])
    merged.sort(key=lambda m: min(m.positions) if m.positions else len(text))
    return merged, disagreements
