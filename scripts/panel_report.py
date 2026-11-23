"""Write a prompt-panel report: topic table + 2D map of the prompt embeddings.

    uv run python scripts/panel_report.py Stridelab            # from the configured DB
    uv run python scripts/panel_report.py Stridelab --fresh    # run the mock pipeline in memory

Outputs `evals/reports/panel_<brand>.md` and `evals/reports/panel_<brand>.png`. The plot
uses PCA (deterministic, no extra dependency); pass `--umap` if umap-learn is installed.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed on Windows/CI
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import citewise.models  # noqa: F401
from citewise.core.config import REPO_ROOT, get_settings
from citewise.core.db import Base, get_sessionmaker
from citewise.models.brand import Brand
from citewise.models.llm_call import LLMCall
from citewise.models.panel import Prompt, Topic
from citewise.schemas.panel import FUNNEL_STAGES, INTENTS
from citewise.services.panel.common import as_matrix, brand_prompts, resolve_brand

DEFAULT_OUT = REPO_ROOT / "evals" / "reports"
SHORT = {"informational": "info", "commercial": "comm", "comparison": "comp",
         "navigational": "nav", "transactional": "trans"}  # fmt: skip


def _fresh_session(brand_name: str) -> Session:
    """Seed + generate + cluster + build (size 50) in an in-memory DB, in mock mode."""
    from citewise.providers import get_chat_provider, get_embedder
    from citewise.services.brands import seed_brands
    from citewise.services.panel.cluster import cluster_brand
    from citewise.services.panel.generate import generate_panel
    from citewise.services.panel.select import build_panel

    settings = get_settings()
    if not settings.is_mock:
        raise SystemExit("--fresh only runs in mock mode (it would spend money otherwise)")
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    seed_brands(session, settings.brands_dir)
    brand = resolve_brand(session, brand_name)
    chat = get_chat_provider(settings=settings, session_factory=factory)
    embedder = get_embedder(settings=settings, session_factory=factory)
    generate_panel(session, brand, chat, embedder, settings)
    cluster_brand(session, brand, chat, settings)
    build_panel(session, brand, 50)
    session.commit()
    return session


def _project(x: np.ndarray, use_umap: bool) -> tuple[np.ndarray, str]:
    if use_umap:
        try:
            import umap  # type: ignore[import-not-found]

            return umap.UMAP(n_components=2, random_state=0).fit_transform(x), "UMAP"
        except ImportError:
            print("umap-learn not installed; falling back to PCA", file=sys.stderr)
    pca = PCA(n_components=2, svd_solver="full")
    z = pca.fit_transform(x)
    explained = pca.explained_variance_ratio_.sum()
    return z, f"PCA (2 components, {explained:.0%} of variance)"


def _plot(prompts: list[Prompt], topics: list[Topic], path: Path, title: str, umap: bool) -> str:
    x = as_matrix([p.embedding for p in prompts if p.embedding])
    z, method = _project(x, umap)
    order = {t.id: i for i, t in enumerate(topics)}
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
    for t in topics:
        idx = [i for i, p in enumerate(prompts) if p.topic_id == t.id]
        ax.scatter(z[idx, 0], z[idx, 1], s=18, alpha=0.75, color=cmap(order[t.id] % 20),
                   label=f"{t.label} ({t.size})")  # fmt: skip
    panel = [i for i, p in enumerate(prompts) if p.in_panel]
    if panel:
        ax.scatter(z[panel, 0], z[panel, 1], s=60, facecolors="none", edgecolors="black",
                   linewidths=0.8, label="in scan panel")  # fmt: skip
    ax.set_title(f"{title}: prompt embeddings by topic ({method})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return method


def _md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return lines


def _escape(text: str) -> str:
    return text.replace("|", "\\|")


def write_report(session: Session, brand: Brand, out_dir: Path, use_umap: bool) -> Path:
    prompts = [p for p in brand_prompts(session, brand.id) if p.embedding]
    topics = list(
        session.scalars(
            select(Topic).where(Topic.brand_id == brand.id).order_by(Topic.size.desc(), Topic.id)
        )
    )
    if not prompts or not topics:
        raise SystemExit(f"{brand.name} has no clustered prompts. Run generate + cluster first.")
    slug = brand.name.lower().replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"panel_{slug}.png"
    method = _plot(prompts, topics, png, brand.name, use_umap)

    by_topic: dict[int, list[Prompt]] = {}
    for p in prompts:
        by_topic.setdefault(p.topic_id or -1, []).append(p)
    n = len(prompts)
    panel = [p for p in prompts if p.in_panel]
    sources = Counter(p.source for p in prompts)
    versions = Counter(p.template_version or "(imported)" for p in prompts)
    intent_src = Counter(p.intent_source or "none" for p in prompts)
    drift = sum(1 for p in prompts if p.requested_intent and p.intent != p.requested_intent)
    calls = list(session.scalars(select(LLMCall).where(LLMCall.purpose.like("panel.%"))))

    lines = [
        f"# Prompt panel: {brand.name}",
        "",
        f"Category: **{brand.category}**. Personas: {', '.join(brand.personas) or '(none)'}.",
        "",
        f"- **{n} prompts** after dedup ({', '.join(f'{k}: {v}' for k, v in sources.items())}); "
        f"template versions: {', '.join(f'`{k}` ({v})' for k, v in versions.items())}.",
        f"- **{len(topics)} topics** ({topics[0].algorithm}); labels from "
        f"{', '.join(f'{k}: {v}' for k, v in Counter(t.label_source for t in topics).items())}.",
        f"- Intent labels by source: {', '.join(f'{k}: {v}' for k, v in intent_src.items())}. "
        f"{drift} prompts ({drift / n:.0%}) were labeled with a different intent than the "
        "generator was asked for.",
        f"- Scan panel: **{len(panel)} prompts** covering "
        f"{len({p.topic_id for p in panel})}/{len(topics)} topics.",
        f"- Panel LLM calls: {len(calls)}, estimated ${sum(c.cost_usd for c in calls):.4f} "
        f"(mock calls are priced as if live).",
        "",
        "## Topics",
        "",
    ]
    rows = []
    for t in topics:
        members = by_topic.get(t.id, [])
        intents = Counter(p.intent for p in members)
        mix = ", ".join(f"{SHORT.get(k or '', k)} {v}" for k, v in intents.most_common(3))
        examples = sorted(members, key=lambda p: -(p.centrality or 0))[:3]
        rows.append(
            [
                _escape(t.label),
                str(t.size),
                f"{t.size / n:.0%}",
                str(sum(p.in_panel for p in members)),
                mix,
                "<br>".join(_escape(p.text) for p in examples),
            ]
        )
    lines += _md_table(
        ["Topic", "Prompts", "Share", "In panel", "Top intents", "Most central prompts"], rows
    )

    lines += ["", "## Topic x intent", ""]
    rows = []
    for t in topics:
        c = Counter(p.intent for p in by_topic.get(t.id, []))
        rows.append([_escape(t.label), *[str(c.get(i, 0)) for i in INTENTS]])
    totals = Counter(p.intent for p in prompts)
    rows.append(["**Total**", *[f"**{totals.get(i, 0)}**" for i in INTENTS]])
    lines += _md_table(["Topic", *INTENTS], rows)

    stages = Counter(p.funnel_stage for p in prompts)
    lines += [
        "",
        "## Funnel stage",
        "",
        ", ".join(f"{s}: {stages.get(s, 0)}" for s in FUNNEL_STAGES)
        + (f", unknown: {stages[None]}" if stages.get(None) else ""),
        "",
        "## Map",
        "",
        f"![{brand.name} prompt map]({png.name})",
        "",
        f"Each dot is a prompt, coloured by topic and projected with {method}. Circled dots are "
        "in the scan panel. Topics overlap in 2D because clustering ran on 20 PCA dimensions.",
        "",
        "Generated by `scripts/panel_report.py`. In mock mode the prompts and labels are "
        "hand-authored fixtures and the embeddings are a deterministic hashing embedder, so "
        "this shows the pipeline, not real search demand.",
        "",
    ]
    md = out_dir / f"panel_{slug}.md"
    md.write_text("\n".join(lines), encoding="utf-8")
    return md


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("brand", help="Brand name or id")
    parser.add_argument("--fresh", action="store_true", help="Run the mock pipeline in memory")
    parser.add_argument("--umap", action="store_true", help="Use UMAP for the plot if installed")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory")
    args = parser.parse_args()

    session = _fresh_session(args.brand) if args.fresh else get_sessionmaker()()
    try:
        md = write_report(session, resolve_brand(session, args.brand), args.out, args.umap)
    finally:
        session.close()
    print(f"wrote {md} and {md.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
