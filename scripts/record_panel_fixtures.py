"""Re-record the Module 1 mock fixtures from the hand-authored panel sources.

    uv run python scripts/record_panel_fixtures.py

`tests/fixtures/panel_source/<brand>.yaml` holds what an LLM would have answered: the
generated prompts per (intent x persona) cell, the intent a careful labeler gives each
prompt, and the brand's topic themes. This script runs the *real* Module 1 pipeline on an
in-memory SQLite DB with a "source" chat backend that answers from those files, and writes
every answer to `tests/fixtures/mock_chat/<key>.json` under the same key the mock provider
looks up. After this, `CITEWISE_MODE=mock` reproduces the panel exactly.

Re-run it whenever a panel template, the brand YAML, the mock embedder or the clustering
changes (the fixture keys hash the rendered messages). `tests/test_panel_pipeline.py`
fails if the generation fixtures are stale.

Labels are hand-authored: generation/intents come straight from the source file. A topic
label comes from `cluster_labels` (keyed by the cluster's most central prompt, written by
reading `--verbose` output) or, failing that, from the authored theme whose keywords best
match the cluster's central prompts.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import citewise.models  # noqa: F401
from citewise.core.config import get_settings
from citewise.core.db import Base
from citewise.providers.base import ChatResult, JsonSchema, Message
from citewise.providers.mock import MockEmbedder, fixture_key
from citewise.services.brands import load_brand_files, upsert_brand
from citewise.services.panel.cluster import cluster_brand
from citewise.services.panel.common import clean_prompt_text, text_hash
from citewise.services.panel.generate import generate_panel, render_generation
from citewise.services.panel.label import fallback_label

PANEL_PURPOSES = ("panel.generate", "panel.classify", "panel.label_topic")


class SourceEmbedder:
    """Same vectors as mock mode, without call tracking."""

    def __init__(self, model: str) -> None:
        self.backend = MockEmbedder(model)

    def embed(self, texts: list[str], *, purpose: str = "embed") -> list[list[float]]:
        return self.backend.embed_with_usage(texts).vectors


def _data_block(text: str) -> Any:
    """The JSON list a panel template renders on its own line (`{{ x | data }}`)."""
    start = text.index("\n[") + 1
    value, _ = json.JSONDecoder().raw_decode(text[start:])
    return value


def _keyword_score(texts: list[str], keywords: list[str]) -> int:
    joined = " " + " ".join(re.sub(r"[^a-z0-9 ]", " ", t.lower()) for t in texts) + " "
    return sum(joined.count(f" {kw.lower()} ") for kw in keywords)


class SourceChat:
    """Answers panel calls from a source YAML and writes each answer as a fixture."""

    name = "source"

    def __init__(self, source: dict[str, Any], out_dir: Path, generation: dict[str, Any]) -> None:
        self.out_dir = out_dir
        self.generation = generation  # fixture key -> GeneratedBatch dict
        self.labels: dict[str, str] = {}
        for cell in source["cells"]:
            for p in cell["prompts"]:
                cleaned = clean_prompt_text(p["text"]) or p["text"]
                self.labels[text_hash(cleaned)] = p.get("label", cell["intent"])
        self.topics = source["topics"]
        # Hand-written labels for specific clusters, keyed by the cluster's most central
        # prompt. These win over keyword matching (which is only a first draft).
        self.cluster_labels: dict[str, dict[str, str]] = source.get("cluster_labels") or {}
        self.seen_clusters: list[tuple[list[str], str]] = []
        self.used_labels: set[str] = set()
        self.written = 0

    def _answer(self, purpose: str, messages: list[Message], key: str) -> dict[str, Any]:
        user = messages[-1]["content"]
        if purpose == "panel.generate":
            return self.generation[key]
        if purpose == "panel.classify":
            items = _data_block(user)
            missing = [i["text"] for i in items if text_hash(i["text"]) not in self.labels]
            if missing:
                raise KeyError(f"No authored intent for {missing[:3]}")
            return {
                "labels": [
                    {"id": i["id"], "intent": self.labels[text_hash(i["text"])]} for i in items
                ]
            }
        if purpose == "panel.label_topic":
            texts = _data_block(user)
            override = self.cluster_labels.get(texts[0])
            self.seen_clusters.append((texts, "override" if override else "keywords"))
            if override:
                return {"label": override["label"], "description": override["description"]}
            ranked = sorted(
                self.topics,
                key=lambda t: (-_keyword_score(texts, t["keywords"]), t["label"]),
            )
            choice = next(
                (
                    t
                    for t in ranked
                    if t["label"] not in self.used_labels and _keyword_score(texts, t["keywords"])
                ),
                None,
            )
            if choice is None:  # no authored theme fits: write the term-based label
                fb = fallback_label(texts)
                return {"label": fb.label, "description": fb.description}
            self.used_labels.add(choice["label"])
            return {"label": choice["label"], "description": choice["description"]}
        raise ValueError(f"Unexpected purpose {purpose!r}")

    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        purpose: str = "general",
        json_schema: JsonSchema | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        key = fixture_key(purpose, messages)
        parsed = self._answer(purpose, messages, key)
        text = json.dumps(parsed, ensure_ascii=False)
        payload = {"purpose": purpose, "text": text, "parsed": parsed}
        path = self.out_dir / f"{key}.json"
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        self.written += 1
        return ChatResult(text=text, model=model, provider=self.name, parsed=parsed)


def _clear_old_fixtures(out_dir: Path) -> int:
    removed = 0
    for path in out_dir.glob("*.json"):
        try:
            purpose = json.loads(path.read_text(encoding="utf-8")).get("purpose", "")
        except json.JSONDecodeError:
            continue
        if purpose in PANEL_PURPOSES:
            path.unlink()
            removed += 1
    return removed


def record_brand(brand_data: Any, source: dict[str, Any], out_dir: Path) -> None:
    settings = get_settings()
    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine, expire_on_commit=False)() as session:
        brand, _ = upsert_brand(session, brand_data)
        generation: dict[str, Any] = {}
        cells = {(c["intent"], c["persona"]): c for c in source["cells"]}
        for (intent, persona), cell in cells.items():
            rendered = render_generation(brand, intent, persona, settings.panel_prompts_per_cell)
            key = fixture_key("panel.generate", rendered.messages)
            generation[key] = {
                "prompts": [
                    {"text": p["text"], "funnel_stage": p["funnel_stage"]} for p in cell["prompts"]
                ]
            }
        chat = SourceChat(source, out_dir, generation)
        embedder = SourceEmbedder(settings.model_embed)
        report = generate_panel(session, brand, chat, embedder, settings)  # type: ignore[arg-type]
        clusters = cluster_brand(session, brand, chat, settings)  # type: ignore[arg-type]
        print(
            f"{brand.name}: {report.stored} prompts stored ({report.near_duplicates} near dupes), "
            f"{clusters.k} topics, {chat.written} fixtures written"
        )
        for (label, size, _), (central, how) in zip(
            clusters.topics, chat.seen_clusters, strict=True
        ):
            print(f"  {size:>4}  {label}  [{how}]")
            if "--verbose" in sys.argv:
                for t in central:
                    print(f"          - {t}")


def main() -> int:
    settings = get_settings()
    out_dir = settings.fixtures_dir / "mock_chat"
    source_dir = settings.fixtures_dir / "panel_source"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"removed {_clear_old_fixtures(out_dir)} old panel fixtures")
    brands = {b.name.casefold(): b for b in load_brand_files(settings.brands_dir)}
    for src_path in sorted(source_dir.glob("*.yaml")):
        source = yaml.safe_load(src_path.read_text(encoding="utf-8"))
        brand_data = brands.get(source["brand"].casefold())
        if brand_data is None:
            print(f"skip {src_path.name}: no brand YAML named {source['brand']!r}")
            continue
        record_brand(brand_data, source, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
