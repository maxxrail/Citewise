"""Entity resolution: map the many ways an answer names a company to one canonical name.

Engines write "Asana", "Asana Inc.", "asana's", "Monday.com", "monday", "HOKA", "Hoka
Clifton 9" and "ClickUp"/"Clickup". Metrics need one key per company, so every surface
form goes through `normalize()` (casing, punctuation, possessives, company suffixes, web
TLDs) and then `Resolver.resolve()`, which tries, in order:

1. an exact alias match (the brand, its aliases, its competitors);
2. a product name that starts with a known alias ("Nike Pegasus" -> Nike);
3. a fuzzy match (rapidfuzz ratio) for typos, only for keys long enough to be safe;
4. otherwise an *unknown* entity: a company we don't track yet, i.e. an emerging
   competitor. Unknowns keep a cleaned display name and are grouped by their key.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from rapidfuzz import fuzz, process

EntityType = Literal["brand", "competitor", "unknown"]

# Legal-form and filler suffixes that never distinguish two companies.
COMPANY_SUFFIXES = frozenset(
    {
        "inc", "incorporated", "ltd", "limited", "llc", "llp", "corp", "corporation", "co",
        "company", "gmbh", "plc", "ag", "sa", "bv", "pty", "holdings", "group",
    }
)  # fmt: skip
WEB_TLD = re.compile(r"\.(?:com|io|so|app|co|ai|net|org)\b")
# Brand names that are also ordinary English words. Stage A only matches these with the
# brand's own capitalization and outside contexts that make them a word ("on the",
# "on Monday"); Stage B (which reads context) has the final say.
AMBIGUOUS_KEYS = frozenset({"on", "monday", "notion"})
FUZZY_MIN_LEN = 5  # shorter keys fuzzy-match too much ("hoka" ~ "hook")
FUZZY_THRESHOLD = 88.0

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'"})
_POSSESSIVE = re.compile(r"'s\b|(?<=s)'(?=\s|$)")
_NON_WORD = re.compile(r"[^\w\s]")


def normalize(name: str) -> str:
    """Matching key: "Asana, Inc." -> "asana"; "Monday.com's" -> "monday"; "HOKA" -> "hoka"."""
    s = unicodedata.normalize("NFKC", name).translate(_APOSTROPHES).casefold().strip()
    s = _POSSESSIVE.sub("", s)
    s = WEB_TLD.sub("", s)
    s = s.replace("&", " and ").replace("-", " ").replace("_", " ")
    tokens = _NON_WORD.sub(" ", s).split()
    while len(tokens) > 1 and tokens[-1] in COMPANY_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def display_name(surface: str) -> str:
    """A readable canonical for an unknown entity: suffixes and possessives removed, the
    original casing kept ("Wrike, Inc.'s" -> "Wrike")."""
    s = unicodedata.normalize("NFKC", surface).translate(_APOSTROPHES).strip(' *_".,:;')
    s = _POSSESSIVE.sub("", s).strip(" '.,")
    tokens = s.replace(",", " ").split()
    while len(tokens) > 1 and tokens[-1].strip(".").casefold() in COMPANY_SUFFIXES:
        tokens.pop()
    return " ".join(tokens).strip(" .,") or surface.strip()


@dataclass(frozen=True)
class Resolved:
    canonical: str
    entity_type: EntityType
    key: str  # normalize(canonical) for known entities; groups unknowns across answers
    how: Literal["exact", "product", "fuzzy", "unknown"]


@dataclass(frozen=True)
class KnownName:
    surface: str  # as configured, e.g. "Monday.com", "On"
    key: str
    canonical: str
    entity_type: EntityType

    @property
    def ambiguous(self) -> bool:
        return self.key in AMBIGUOUS_KEYS


class Resolver:
    """Resolves surface forms for one tracked brand (its aliases + its competitors)."""

    def __init__(self, brand: str, aliases: list[str], competitors: list[str]) -> None:
        self.brand = brand
        self._by_key: dict[str, tuple[str, EntityType]] = {}
        self.names: list[KnownName] = []
        # The brand is added first, so it wins if a competitor shares a key.
        entries: list[tuple[str, str, str]] = [
            *((n, brand, "brand") for n in [brand, *aliases]),
            *((c, c, "competitor") for c in competitors),
        ]
        # "Monday.com" is usually written "Monday": the bare name is an implicit alias.
        entries += [
            (bare, canonical, typ)
            for surface, canonical, typ in list(entries)
            if (bare := WEB_TLD.sub("", surface)) != surface and bare.strip()
        ]
        for surface, canonical, typ in entries:
            key = normalize(surface)
            if not key:
                continue
            self._by_key.setdefault(key, (canonical, typ))  # type: ignore[arg-type]
            self.names.append(KnownName(surface, key, *self._by_key[key]))
        # Longest first, so "new balance" beats "new" as a product prefix.
        self._prefix_keys = sorted(self._by_key, key=len, reverse=True)
        self._fuzzy_keys = [k for k in self._by_key if len(k) >= FUZZY_MIN_LEN]

    def known(self, key: str) -> tuple[str, EntityType] | None:
        return self._by_key.get(key)

    def resolve(self, surface: str, *, fuzzy: bool = True) -> Resolved:
        key = normalize(surface)
        if hit := self._by_key.get(key):
            return Resolved(hit[0], hit[1], normalize(hit[0]), "exact")
        for k in self._prefix_keys:
            if key.startswith(k + " "):
                canonical, typ = self._by_key[k]
                return Resolved(canonical, typ, normalize(canonical), "product")
        if fuzzy and len(key) >= FUZZY_MIN_LEN and self._fuzzy_keys:
            match = process.extractOne(
                key, self._fuzzy_keys, scorer=fuzz.ratio, score_cutoff=FUZZY_THRESHOLD
            )
            if match is not None:
                canonical, typ = self._by_key[match[0]]
                return Resolved(canonical, typ, normalize(canonical), "fuzzy")
        name = display_name(surface)
        return Resolved(name, "unknown", normalize(name), "unknown")
