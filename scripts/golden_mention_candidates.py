"""Draw candidate answers for the mention golden set (Module 3).

Answers come from the deterministic mock engines (the same simulator a mock scan uses), for
brand-heavy prompts from the hand-written Module 1 panel sources. The sample is stratified
by brand x engine x answer style (list, comparison, single-brand, deals) so every format the
extractor meets is covered. Output has no labels: a labeler adds them (see
evals/rubrics/mentions.yaml).

    uv run python scripts/golden_mention_candidates.py --per-cell 6 --out <path>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

import yaml

from citewise.core.config import get_settings
from citewise.providers.engines.mock import MockEngine, _Brand, _mentions, _style

ENGINES = ("openai", "anthropic", "gemini", "perplexity")
BRAND_INTENTS = ("commercial", "comparison", "transactional", "navigational")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-cell", type=int, default=6, help="answers per brand x style")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    settings = get_settings()
    src = settings.fixtures_dir / "panel_source"
    rng = random.Random(args.seed)
    rows: list[dict] = []
    for brand_file in ("stridelab.yaml", "taskflow.yaml"):
        data = yaml.safe_load((src / brand_file).read_text(encoding="utf-8"))
        prompts = [
            p["text"]
            for cell in data["cells"]
            if cell["intent"] in BRAND_INTENTS
            for p in cell["prompts"]
        ]
        by_style: dict[str, list[tuple[str, str]]] = {}
        for text in prompts:
            for engine in ENGINES:
                eng = MockEngine(engine, "mock", settings.fixtures_dir)
                cat = eng._category(text)
                if cat is None:  # generic answer that names no brands: nothing to label
                    continue
                named = [
                    _Brand(n, d) for n, d in cat["brands"].items() if _mentions(text, _Brand(n, d))
                ]
                by_style.setdefault(_style(text, named), []).append((text, engine))
        for style, pairs in sorted(by_style.items()):
            for text, engine in rng.sample(pairs, min(args.per_cell, len(pairs))):
                ans = asyncio.run(MockEngine(engine, "mock", settings.fixtures_dir).answer(text))
                rows.append(
                    {
                        "id": f"{data['brand'].lower()}-{len(rows) + 1:03d}",
                        "brand": data["brand"],
                        "engine": engine,
                        "style": style,
                        "prompt": text,
                        "answer": ans.text,
                    }
                )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    print(f"wrote {len(rows)} candidates to {args.out}")


if __name__ == "__main__":
    main()
