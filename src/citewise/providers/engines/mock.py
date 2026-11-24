"""Deterministic mock answer engines.

Hand-writing 400+ answers (50 prompts x 4 engines x 2 brands) isn't practical, so the mock
*simulates* an answer engine from a small fictional "world" (tests/fixtures/engines/
world.yaml): which brands exist per category, their products, praise, criticism and a few
true and false claims, the review sites engines cite, and a bias profile per engine. The
seed is (engine, model, system message, prompt), so the same question always gets the same
answer, and different engines differ in which brands they list, in what order, with what
sentiment, and which sources they cite. Module 3 metrics (share of voice, rank, cited
domains) and Module 4 fact-checking get real signal offline.

A recorded answer in `<fixtures>/mock_engine/<key>.json` (text + citations) overrides the
simulation for that prompt, like the chat mock's fixtures.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from citewise.providers.base import Citation, Usage
from citewise.providers.engines.base import EngineAnswer
from citewise.providers.mock import estimate_tokens

GROUNDING_REDIRECT = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/"
SEARCH_CONTEXT_TOKENS = 1200  # what search results add to the prompt, roughly, per search


def engine_fixture_key(engine: str, prompt: str, system: str | None) -> str:
    body = json.dumps({"engine": engine, "prompt": prompt, "system": system}, sort_keys=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]


@lru_cache
def load_world(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@dataclass
class _Brand:
    name: str
    data: dict[str, Any]

    @property
    def tracked(self) -> bool:
        return "claims" in self.data


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9.]+", text.lower()))


def _mentions(prompt: str, brand: _Brand) -> bool:
    low = f" {prompt.lower()} "
    names = [brand.name, *brand.data.get("aliases", [])]
    return any(re.search(rf"(?<![a-z0-9]){re.escape(n.lower())}(?![a-z0-9])", low) for n in names)


def _style(prompt: str, named: list[_Brand]) -> str:
    low = prompt.lower()
    if len(named) >= 2 or re.search(r"\bvs\.?\b|\bversus\b|\bcompare", low):
        return "comparison"
    if re.search(r"cheap|discount|coupon|promo|code|deal|sale|price|cost|trial|buy", low):
        return "deals"
    if len(named) == 1:
        return "single"
    return "list"


class MockEngine:
    """Implements `EngineAdapter` for one engine name ("openai", "anthropic", ...)."""

    def __init__(self, name: str, model: str, fixtures_dir: Path) -> None:
        self.name = name
        self.model = model
        self.fixtures_dir = fixtures_dir
        self.world = load_world(fixtures_dir / "engines" / "world.yaml")
        self.profile: dict[str, Any] = self.world["engines"][name]

    # ---- public ----

    async def answer(self, prompt: str, *, system: str | None = None) -> EngineAnswer:
        key = engine_fixture_key(self.name, prompt, system)
        recorded = self.fixtures_dir / "mock_engine" / f"{key}.json"
        if recorded.exists():
            data = json.loads(recorded.read_text(encoding="utf-8"))
            text, citations = data["text"], [Citation(**c) for c in data.get("citations", [])]
            raw: dict[str, Any] = {"fixture_key": key, "fixture_hit": True}
        else:
            text, citations = self._simulate(prompt, system)
            raw = {"fixture_key": key, "fixture_hit": False, "simulated": True}
        searches = 1 if citations else 0
        prompt_tokens = estimate_tokens(prompt + (system or "")) + searches * SEARCH_CONTEXT_TOKENS
        return EngineAnswer(
            text=text,
            citations=citations,
            model=self.model,
            usage=Usage(tokens_in=prompt_tokens, tokens_out=estimate_tokens(text)),
            searches=searches,
            raw=raw,
        )

    # ---- simulation ----

    def _rng(self, prompt: str, system: str | None) -> random.Random:
        seed = hashlib.sha256(f"{self.name}|{self.model}|{system}|{prompt}".encode()).digest()
        return random.Random(int.from_bytes(seed[:8], "big"))

    def _category(self, prompt: str) -> dict[str, Any]:
        words = _words(prompt)
        cats = self.world["categories"]

        def score(item: tuple[str, dict[str, Any]]) -> int:
            _name, cat = item
            brand_hits = sum(_mentions(prompt, _Brand(b, d)) for b, d in cat["brands"].items())
            return 3 * brand_hits + sum(1 for k in cat["keywords"] if k in words)

        return max(cats.items(), key=score)[1]

    def _pick_brands(self, rng: random.Random, cat: dict[str, Any], prompt: str) -> list[_Brand]:
        brands = [_Brand(n, d) for n, d in cat["brands"].items()]
        named = [b for b in brands if _mentions(prompt, b)]
        chosen = list(named)
        tracked = next((b for b in brands if b.tracked), None)
        if (
            tracked
            and tracked not in chosen
            and len(named) < 2
            and rng.random() < self.profile["own_brand_rate"]
        ):
            chosen.append(tracked)
        if len(named) < 2:  # lists and single-brand answers add a few alternatives
            pool = [b for b in brands if b not in chosen and not b.tracked]
            weights = [self.profile.get("brand_bias", {}).get(b.name, 1.0) for b in pool]
            want = rng.randint(2, 4) if len(named) == 0 else rng.randint(1, 2)
            while pool and want > 0:
                b = rng.choices(pool, weights=weights)[0]
                i = pool.index(b)
                pool.pop(i)
                weights.pop(i)
                chosen.append(b)
                want -= 1
        # Named brands keep their place; the rest are ordered by the engine's preference.
        rest = chosen[len(named) :]
        rng.shuffle(rest)
        rest.sort(key=lambda b: -self.profile.get("brand_bias", {}).get(b.name, 1.0))
        if tracked in rest and rng.random() < 0.5:  # the tracked brand lands anywhere
            rest.remove(tracked)
            rest.insert(rng.randint(0, len(rest)), tracked)
        return named + rest

    def _sentence(
        self, rng: random.Random, brand: _Brand, cite: int, avoid: str | None = None
    ) -> str:
        d = brand.data
        product = rng.choice(d["products"])
        positives = [x for x in d["positives"] if x != avoid] or d["positives"]
        text = f"**{product}**: {rng.choice(positives)}"
        if d.get("negatives") and rng.random() < self.profile["negative_rate"]:
            text += f", although {rng.choice(d['negatives'])}"
        if brand.tracked and rng.random() < self.profile["claim_rate"]:
            text += f". {rng.choice(d['claims'])['text']}"
        return f"{text} [{cite}]."

    def _citations(
        self, rng: random.Random, cat: dict[str, Any], brands: list[_Brand]
    ) -> list[Citation]:
        lo, hi = self.profile["citations"]
        n = rng.randint(lo, hi)
        pubs = list(cat["publishers"])
        bias = self.profile.get("publisher_bias", {})
        out: list[tuple[str, str]] = []
        for b in brands:
            if len(out) < n and rng.random() < self.profile["brand_site_rate"]:
                out.append((f"https://www.{b.data['domain']}/", f"{b.name} official site"))
        while pubs and len(out) < n:
            weights = [bias.get(p["domain"], 1.0) for p in pubs]
            p = rng.choices(pubs, weights=weights)[0]
            pubs.remove(p)
            out.append((f"https://www.{p['domain']}{p['path']}", p["title"]))
        cites = []
        for i, (url, title) in enumerate(out, start=1):
            if self.profile.get("utm"):
                url += ("&" if "?" in url else "?") + "utm_source=openai"
            if self.profile.get("redirect"):
                # Real Gemini grounding: opaque redirect URL; the source domain is the title.
                token = hashlib.sha256(url.encode()).hexdigest()[:24]
                title, url = url.split("/")[2].removeprefix("www."), GROUNDING_REDIRECT + token
            cites.append(Citation(url=url, title=title, position=i))
        return cites

    def _simulate(self, prompt: str, system: str | None) -> tuple[str, list[Citation]]:
        rng = self._rng(prompt, system)
        cat = self._category(prompt)
        brands = self._pick_brands(rng, cat, prompt)
        named = [b for b in brands if _mentions(prompt, b)]
        citations = self._citations(rng, cat, brands)
        n = max(1, len(citations))
        style = _style(prompt, named)
        locale = " (based on what's available where you are)" if system else ""

        if style == "comparison" and len(brands) >= 2:
            a, b = brands[0], brands[1]
            lines = [
                f"Both {a.name} and {b.name} are solid choices{locale}; "
                "it depends on what matters most to you."
            ]
            lines += [f"- {self._sentence(rng, x, rng.randint(1, n))}" for x in brands[:3]]
            winner = rng.choice([a, b])
            lines.append(
                f"Overall, many reviewers lean towards **{winner.name}** for this use case."
            )
        elif style == "single" and named:
            subject = named[0]
            opener = rng.choice(subject.data["positives"])
            lines = [f"{subject.name} is {opener}{locale}."]
            lines.append(self._sentence(rng, subject, 1, avoid=opener))
            if len(brands) > 1:
                alts = ", ".join(x.name for x in brands[1:])
                lines.append(f"If it doesn't fit your needs, people also consider {alts}.")
        elif style == "deals":
            lines = [f"Here are some ways to save{locale}:"]
            lines += [f"{i}. {self._sentence(rng, x, min(i, n))}" for i, x in enumerate(brands, 1)]
            lines.append("Prices change often, so check the retailer before buying.")
        else:
            lines = [f"Here are some of the best options{locale}:"]
            lines += [f"{i}. {self._sentence(rng, x, min(i, n))}" for i, x in enumerate(brands, 1)]
            lines.append("The best pick depends on your budget, fit and how you'll use it.")
        return "\n".join(lines), citations
