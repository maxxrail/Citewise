"""Simulated FactCheck model calls for mock mode: claim extraction, verification, KB
re-ranking and theme labels.

Same principle as `mock_extract.py`: a schema-valid placeholder would make offline
fact-checking meaningless, and reading the answer simulator's truth flags
(tests/fixtures/engines/world.yaml `true:`) would make it a fake 100%. So this imitates a
careful reader using only the rendered prompt: the claim, the answer, the passages. It
applies general reading rules (different amounts for the same thing, "no X"/"only X"/
"discontinued" against a passage that says X exists, a different place of manufacture),
and it doesn't know which claims were planted. Recorded real-model responses (fixtures
keyed by the exact prompt) replace it wherever they exist.
"""

from __future__ import annotations

import itertools
import json
import re
from collections import Counter
from typing import Any

from citewise.providers.base import Message

PURPOSES = ("factcheck.claims", "factcheck.verify", "kb.rerank", "narrative.theme")

_WORD = re.compile(r"[a-z0-9$]+(?:\.\d+)?")
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
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
        "per",
        "can",
        "you",
        "your",
        "their",
        "they",
        "there",
        "which",
        "who",
        "what",
        "how",
        "when",
        "also",
    ]
)
# "$1,200" and "$12.50" but not the comma in "costs $140, weighs ...".
_MONEY = re.compile(r"\$\s?\d+(?:,\d{3})*(?:\.\d+)?")
_FACT_CUES = re.compile(
    r"\$\d|\b\d+\b|\bonly\b|\bno\b|\bnot\b|discontinued|made in|manufactured|sells?\b|"
    r"\btrial\b|warranty|free plan|\bplan\b|integrat|import|\bapps?\b|portal|founded|"
    r"headquarter|available|costs?\b|priced",
    re.IGNORECASE,
)
_NEG_CLAIM = re.compile(r"\b(?:no|not|never|without|discontinued|ended|ends|doesn't|lacks)\b", re.I)
_ONLY = re.compile(r"\bonly\b", re.I)
_NEG_TEXT = re.compile(r"\b(?:not|no|does not|doesn't|never|isn't|aren't)\b", re.I)
_PLACE = re.compile(r"\b(?:made|manufactured|produced|built)\s+in\s+([A-Z][a-z]+)")
_MARKER = re.compile(r"\s*\[(?:web:)?\d+\]")


def _field(user: str, label: str) -> Any:
    m = re.search(rf"^{re.escape(label)}: (.*)$", user, re.MULTILINE)
    return json.loads(m.group(1)) if m else None


def _block(user: str, header: str) -> Any:
    m = re.search(re.escape(header) + r"\n(.*)\Z", user, re.DOTALL)
    return json.loads(m.group(1).strip()) if m else None


def _stem(w: str) -> str:
    """Crude suffix stripping so "apps"/"app" and "importers"/"import" match. Prices and
    numbers are left alone."""
    if w[0] == "$" or w[0].isdigit():
        return w
    for suffix in ("ers", "ing", "er", "ed", "s"):
        if len(w) - len(suffix) >= 3 and w.endswith(suffix) and not w.endswith("ss"):
            return w[: -len(suffix)]
    return w


def _terms(text: str) -> list[str]:
    return [_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP]


def _bigrams(text: str) -> set[tuple[str, str]]:
    t = _terms(text)
    return set(itertools.pairwise(t))


def _sentences(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s+", "", line).strip()
        out += [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z*])", line) if s.strip()]
    return out


# ---------------------------------------------------------------- claims


def simulate_claims(messages: list[Message]) -> dict[str, Any]:
    user = messages[-1]["content"]
    brand = _field(user, "Brand") or ""
    aliases = _field(user, "Brand aliases and products") or []
    answer = _block(user, "Answer (a JSON string):") or ""
    names = [n.casefold() for n in [brand, *aliases] if n]
    claims = []
    for sentence in _sentences(answer):
        clean = _MARKER.sub("", sentence).replace("**", "").strip().rstrip(".")
        if not any(n in clean.casefold() for n in names) or not _FACT_CUES.search(clean):
            continue
        # "Taskflow Client Portal: easy to adopt..." is a description, not a fact.
        if re.match(r"^[^:]{1,60}:\s", clean) and not _MONEY.search(clean):
            continue
        low = clean.casefold()
        if _MONEY.search(clean) or re.search(r"\b(?:cost|price|plan)\b", low):
            ctype = "pricing"
        elif re.search(r"made in|manufactur|sells|founded|headquarter|employees", low):
            ctype = "company"
        else:
            ctype = "feature"
        # The span must be verbatim from the answer: the sentence without its marker.
        span = _MARKER.sub("", sentence).replace("**", "").strip().rstrip(".")
        if span.casefold() not in answer.replace("**", "").casefold():
            continue
        claims.append({"claim": clean, "type": ctype, "answer_span": span})
    return {"claims": claims}


# ---------------------------------------------------------------- verification


def _subject_overlap(claim: str, sentence: str, brand_terms: set[str]) -> set[str]:
    return (set(_terms(claim)) - brand_terms) & set(_terms(sentence))


def _names(text: str, brand_terms: set[str]) -> set[str]:
    """Proper-noun terms (plan/product names like "Pro", "Tempo"), not counting the first
    word or the brand: what a price is *for*. "costs", "user", "month" are not subjects."""
    words = re.findall(r"[A-Za-z][\w-]*", text)
    return {w.lower() for w in words[1:] if w[0].isupper() and w.lower() not in brand_terms}


def simulate_verify(messages: list[Message]) -> dict[str, Any]:
    user = messages[-1]["content"]
    brand = _field(user, "Brand") or ""
    claim = _field(user, "Claim") or ""
    passages = _block(user, 'Passages (JSON list of {"id", "text"}):') or []
    brand_terms = set(_terms(brand))
    scored: list[tuple[int, int, str, str]] = []  # (overlap, -order, passage id, sentence)
    for _order, p in enumerate(passages):
        # Skip the "Doc title > Section" prefix line: it locates the passage, it isn't a fact.
        head, _, body = p["text"].partition("\n")
        sents = _sentences(body if body and " > " in head else p["text"])
        # Single sentences and adjacent pairs: "The Stridelab Tempo is ... It costs $140."
        # (a reader resolves "It"; the pair is still a verbatim quote of the passage).
        windows = sents + [f"{a} {b}" for a, b in itertools.pairwise(sents)]
        for sentence in windows:
            if sentence not in p["text"]:
                continue
            shared = _subject_overlap(claim, sentence, brand_terms)
            if shared:
                # Shared word pairs ("free plan") outweigh scattered single words, and a
                # shorter sentence wins a tie (it is about this, not about everything).
                score = len(shared) + 2 * len(_bigrams(claim) & _bigrams(sentence))
                scored.append((score, -len(sentence), p["id"], sentence))
    scored.sort(reverse=True)
    claim_money = set(m.replace(" ", "") for m in _MONEY.findall(claim))
    claim_terms = set(_terms(claim)) - brand_terms

    def verdict(v: str, pid: str | None, quote: str | None, why: str) -> dict[str, Any]:
        return {"verdict": v, "evidence_chunk_ids": [pid] if pid else [],
                "evidence_quote": quote, "rationale": why}  # fmt: skip

    # Agreeing evidence first: a passage giving the same price for the same plan/product
    # settles it, even if a neighbouring sentence prices something else.
    if claim_money:
        names = _names(claim, brand_terms)
        for _, _, pid, sentence in scored[:6]:
            sent_money = {m.replace(" ", "") for m in _MONEY.findall(sentence)}
            if claim_money <= sent_money and (not names or names & set(_terms(sentence))):
                return verdict("supported", pid, sentence, "The documentation gives this price.")
    for _, _, pid, sentence in scored[:4]:
        sent_money = set(m.replace(" ", "") for m in _MONEY.findall(sentence))
        shared = _subject_overlap(claim, sentence, brand_terms) - {m for m in claim_money}
        # Different amount for the same plan/product.
        # A different price only contradicts if it's the price of the same thing.
        same_subject = (_names(claim, brand_terms) & set(_terms(sentence))) or (
            not _names(claim, brand_terms) and shared
        )
        if claim_money and sent_money and not (claim_money & sent_money) and same_subject:
            return verdict("contradicted", pid, sentence,
                           f"The documentation gives {', '.join(sorted(sent_money))}, "
                           f"not {', '.join(sorted(claim_money))}.")  # fmt: skip
        # Different place of manufacture.
        cp, sp = _PLACE.search(claim), _PLACE.search(sentence)
        if cp and sp and cp.group(1) != sp.group(1):
            return verdict("contradicted", pid, sentence,
                           f"Made in {sp.group(1)}, not {cp.group(1)}.")  # fmt: skip
        # "only X" against a passage that says not-X or lists alternatives.
        lists_more = re.search(r",|\band\b|\bor\b", sentence)  # "browser and desktop and mobile"
        if _ONLY.search(claim) and shared and (_NEG_TEXT.search(sentence) or lists_more):
            why = "The documentation describes more than the one option claimed."
            return verdict("contradicted", pid, sentence, why)
        # "no X" / "discontinued X" against a passage that affirms X.
        affirms = not _NEG_TEXT.search(sentence) or re.search(r"\bstill\b", sentence, re.I)
        if _NEG_CLAIM.search(claim) and not _ONLY.search(claim) and shared and affirms:
            return verdict("contradicted", pid, sentence, "The documentation says this exists.")
    if scored:
        _, _, pid, sentence = scored[0]
        overlap = len(_subject_overlap(claim, sentence, brand_terms))  # without the pair bonus
        sent_money = set(m.replace(" ", "") for m in _MONEY.findall(sentence))
        money_ok = not claim_money or claim_money <= sent_money
        if claim_terms and overlap / len(claim_terms) >= 0.5 and money_ok:
            return verdict("supported", pid, sentence, "The documentation states this.")
    return verdict("not_in_kb", None, None, "The passages don't settle this claim.")


# ---------------------------------------------------------------- rerank, themes


def simulate_rerank(messages: list[Message]) -> dict[str, Any]:
    user = messages[-1]["content"]
    query = _field(user, "Question or statement to verify") or ""
    if not query:
        m = re.search(r"Question or statement to verify:\n(.*)\n", user)
        query = json.loads(m.group(1)) if m else ""
    passages = _block(user, 'Passages (JSON list of {"id", "text"}):') or []
    q = set(_terms(query))
    ranked = sorted(passages, key=lambda p: -len(q & set(_terms(p["text"]))))
    return {"chunk_keys": [p["id"] for p in ranked[:5]]}


def simulate_theme(messages: list[Message]) -> dict[str, Any]:
    user = messages[-1]["content"]
    brand_terms = set(_terms(_field(user, "Brand") or ""))
    snippets = _block(user, "Snippets (JSON list):") or []
    counts: Counter[str] = Counter()
    for s in snippets:
        words = [w for w in _terms(_MARKER.sub("", s)) if w not in brand_terms and not w.isdigit()]
        counts.update(" ".join(words[i : i + 2]) for i in range(len(words) - 1))
    label = counts.most_common(1)[0][0] if counts else "general mentions"
    return {"label": label}


def simulate(purpose: str, messages: list[Message]) -> dict[str, Any]:
    return {
        "factcheck.claims": simulate_claims,
        "factcheck.verify": simulate_verify,
        "kb.rerank": simulate_rerank,
        "narrative.theme": simulate_theme,
    }[purpose](messages)
