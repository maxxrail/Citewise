"""Generate (or import) candidate prompts, then embed, dedup and store them.

Generation makes one structured-output call per (intent x persona) cell with the versioned
template `prompts/panel/generate.<version>.jinja`, and stores the template version on
every generated prompt.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session

from citewise.core.config import Settings
from citewise.models.brand import Brand
from citewise.models.panel import Prompt
from citewise.providers.base import ChatProvider, EmbeddingProvider
from citewise.schemas.panel import FUNNEL_STAGES, INTENTS, GeneratedBatch
from citewise.services.panel.common import (
    as_matrix,
    brand_prompts,
    clean_prompt_text,
    text_hash,
)
from citewise.services.panel.dedup import dedup
from citewise.services.prompt_library import RenderedPrompt, render
from citewise.services.structured import complete_structured

log = logging.getLogger(__name__)

GENERATE_TEMPLATE = "panel/generate"
DEFAULT_PERSONA = "a typical buyer"
EMBED_BATCH = 256

INTENT_DEFINITIONS = {
    "informational": "learning or understanding something, not choosing a product yet.",
    "commercial": "researching which product or option to buy (best X for Y).",
    "comparison": "comparing two or more named options (A vs B, alternatives to A).",
    "navigational": "asking about this specific brand itself (is it good, reviews, sizing).",
    "transactional": "ready to buy or sign up (price, discount, coupon, trial, where to buy).",
}


@dataclass
class Candidate:
    text: str
    source: str  # "generated" | "import"
    template_version: str | None = None
    persona: str | None = None
    funnel_stage: str | None = None
    requested_intent: str | None = None
    intent: str | None = None  # only imports that carry a label
    intent_source: str | None = None


@dataclass
class StoreReport:
    candidates: int = 0
    invalid: int = 0  # blank or too long
    exact_duplicates: int = 0
    near_duplicates: int = 0
    stored: int = 0
    failed_calls: int = 0
    skipped_cells: int = 0  # already generated with this template version
    reembedded: int = 0  # stored prompts moved into the current embedder's vector space
    examples: list[tuple[str, str]] = field(default_factory=list)  # (dropped, kept)


def generation_cells(brand: Brand) -> list[tuple[str, str]]:
    personas = brand.personas or [DEFAULT_PERSONA]
    return [(intent, persona) for intent in INTENTS for persona in personas]


def render_generation(
    brand: Brand, intent: str, persona: str, n: int, version: str = "v1"
) -> RenderedPrompt:
    return render(
        GENERATE_TEMPLATE,
        version,
        brand=brand.name,
        category=brand.category,
        aliases=brand.aliases,
        competitors=brand.competitors,
        persona=persona,
        intent=intent,
        intent_definition=INTENT_DEFINITIONS[intent],
        n=n,
    )


def generate_candidates(
    brand: Brand,
    chat: ChatProvider,
    settings: Settings,
    version: str = "v1",
    skip: set[tuple[str, str]] | None = None,
) -> tuple[list[Candidate], int]:
    """Run every generation cell not in `skip`. Returns (candidates, cells that failed)."""
    candidates: list[Candidate] = []
    failed = 0
    for intent, persona in generation_cells(brand):
        if skip and (intent, persona) in skip:
            continue
        prompt = render_generation(brand, intent, persona, settings.panel_prompts_per_cell, version)
        batch = complete_structured(
            chat,
            prompt.messages,
            GeneratedBatch,
            model=settings.model_generate,
            purpose="panel.generate",
        )
        if batch is None:
            failed += 1
            continue
        candidates += [
            Candidate(
                text=p.text,
                source="generated",
                template_version=prompt.version,
                persona=persona if brand.personas else None,
                funnel_stage=p.funnel_stage,
                requested_intent=intent,
            )
            for p in batch.prompts
        ]
    return candidates, failed


def read_csv_candidates(path: Path) -> list[Candidate]:
    """Read prompts from a CSV with a `text` column (optional: intent, persona, funnel_stage).

    Unknown intent/stage values are dropped, so the prompt is stored unlabeled and gets
    classified like any other.
    """
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = [(h or "").strip().lower() for h in reader.fieldnames or []]
        if "text" not in headers:
            raise ValueError(f"{path} needs a 'text' column")
        out = []
        for raw in reader:
            row = {(k or "").strip().lower(): (v or "").strip() for k, v in raw.items()}
            intent = row.get("intent", "").lower()
            stage = row.get("funnel_stage", "").lower()
            labeled = intent in INTENTS
            persona = clean_prompt_text(row.get("persona", ""))
            out.append(
                Candidate(
                    text=row.get("text", ""),
                    source="import",
                    persona=persona[:100] if persona else None,
                    funnel_stage=stage if stage in FUNNEL_STAGES else None,
                    intent=intent if labeled else None,
                    intent_source="import" if labeled else None,
                )
            )
        return out


def _embed_all(embedder: EmbeddingProvider, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        vectors += embedder.embed(texts[start : start + EMBED_BATCH], purpose="panel.embed")
    return vectors


def refresh_embeddings(prompts: list[Prompt], embedder: EmbeddingProvider) -> int:
    """Re-embed prompts whose vectors are missing or from another vector space.

    Mock and live embedders share a model name but not a space, and a MODEL_EMBED change
    can change the dimension, so mixing them would silently break dedup and clustering.
    Mutates the ORM objects (the caller's session flushes). Returns how many changed.
    """
    stale = [p for p in prompts if p.embedding is None or p.embedding_model != embedder.ident]
    if not stale:
        return 0
    old = sorted({p.embedding_model or "unknown" for p in stale})
    log.info("re-embedding %d prompts (%s -> %s)", len(stale), ", ".join(old), embedder.ident)
    for p, vector in zip(stale, _embed_all(embedder, [p.text for p in stale]), strict=True):
        p.embedding, p.embedding_model = vector, embedder.ident
    return len(stale)


def store_candidates(
    session: Session,
    brand: Brand,
    candidates: Iterable[Candidate],
    embedder: EmbeddingProvider,
    threshold: float,
) -> StoreReport:
    """Clean, drop exact and near duplicates (also against stored prompts), then insert.

    Idempotent: storing the same candidates again adds nothing.
    """
    report = StoreReport()
    existing = brand_prompts(session, brand.id)
    report.reembedded = refresh_embeddings(existing, embedder)
    seen = {p.text_hash for p in existing}
    fresh: list[Candidate] = []
    for cand in candidates:
        report.candidates += 1
        cleaned = clean_prompt_text(cand.text)
        if cleaned is None:
            report.invalid += 1
            continue
        h = text_hash(cleaned)
        if h in seen:
            report.exact_duplicates += 1
            continue
        seen.add(h)
        cand.text = cleaned
        fresh.append(cand)
    if not fresh:
        return report

    vectors = _embed_all(embedder, [c.text for c in fresh])
    with_vectors = [p for p in existing if p.embedding]
    existing_matrix = (
        as_matrix([p.embedding for p in with_vectors if p.embedding]) if with_vectors else None
    )
    result = dedup([c.text for c in fresh], as_matrix(vectors), threshold, existing_matrix)

    report.near_duplicates = len(result.dropped)
    for idx, (kind, other) in list(result.dropped.items())[:5]:
        kept_text = fresh[other].text if kind == "candidate" else with_vectors[other].text
        report.examples.append((fresh[idx].text, kept_text))
    for idx in result.keep:
        c = fresh[idx]
        session.add(
            Prompt(
                brand_id=brand.id,
                text=c.text,
                text_hash=text_hash(c.text),
                source=c.source,
                template_version=c.template_version,
                persona=c.persona,
                funnel_stage=c.funnel_stage,
                requested_intent=c.requested_intent,
                intent=c.intent,
                intent_source=c.intent_source,
                embedding=vectors[idx],
                embedding_model=embedder.ident,
            )
        )
    session.flush()
    report.stored = len(result.keep)
    log.info(
        "panel store %s: %d candidates, %d invalid, %d exact dupes, %d near dupes, %d stored",
        brand.name,
        report.candidates,
        report.invalid,
        report.exact_duplicates,
        report.near_duplicates,
        report.stored,
    )
    return report


def generate_panel(
    session: Session,
    brand: Brand,
    chat: ChatProvider,
    embedder: EmbeddingProvider,
    settings: Settings,
    version: str = "v1",
    force: bool = False,
) -> StoreReport:
    """Generate and store a panel. Cells that already have prompts from this template
    version are skipped unless `force`: live output differs on every call, so blindly
    re-running would pay again and keep growing the panel.
    """
    template_version = f"{GENERATE_TEMPLATE}.{version}"
    done: set[tuple[str, str]] = set()
    if not force:
        for p in brand_prompts(session, brand.id):
            if p.template_version == template_version and p.requested_intent:
                done.add((p.requested_intent, p.persona or DEFAULT_PERSONA))
    candidates, failed = generate_candidates(brand, chat, settings, version, skip=done)
    report = store_candidates(session, brand, candidates, embedder, settings.panel_dedup_threshold)
    report.failed_calls = failed
    report.skipped_cells = len(done & set(generation_cells(brand)))
    return report


def import_csv(
    session: Session, brand: Brand, path: Path, embedder: EmbeddingProvider, settings: Settings
) -> StoreReport:
    candidates = read_csv_candidates(path)
    if len(candidates) > settings.panel_import_max_rows:
        raise ValueError(
            f"{path} has {len(candidates)} rows; the limit is {settings.panel_import_max_rows} "
            "(PANEL_IMPORT_MAX_ROWS). Each row is embedded and classified, so this caps cost."
        )
    return store_candidates(session, brand, candidates, embedder, settings.panel_dedup_threshold)
