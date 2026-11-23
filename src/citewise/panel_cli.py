"""`citewise panel ...` commands (Module 1). Mounted on the main app in `cli.py`."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from citewise.core.config import REPO_ROOT, get_settings

panel_app = typer.Typer(
    help="Prompt panel: generate, cluster, classify, select.", no_args_is_help=True
)

GOLDEN_INTENTS = REPO_ROOT / "evals" / "datasets" / "golden" / "intent.jsonl"


MAX_COST_HELP = "Stop LLM calls after this estimated spend (default MAX_COST_USD_PER_RUN)."


def _fail(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _budgeted_chat(max_cost: float | None):  # type: ignore[no-untyped-def]
    """The OpenAI chat provider behind a spend cap for this command."""
    from citewise.providers import get_chat_provider
    from citewise.providers.budget import BudgetedChatProvider

    cap = get_settings().max_cost_usd_per_run if max_cost is None else max_cost
    return BudgetedChatProvider(get_chat_provider("openai"), cap)


def _budget_note(chat) -> None:  # type: ignore[no-untyped-def]
    typer.echo(f"estimated LLM spend: ${chat.spent_usd:.4f} (cap ${chat.max_cost_usd:.2f})")
    if chat.refused:
        typer.secho(
            f"  budget reached: {chat.refused} call(s) skipped; raise --max-cost and re-run",
            fg=typer.colors.YELLOW,
        )


@panel_app.command()
def generate(
    brand: str = typer.Argument(..., help="Brand name or id."),
    version: str = typer.Option("v1", help="Template version of prompts/panel/generate."),
    force: bool = typer.Option(
        False, "--force", help="Regenerate cells that already have prompts (costs money live)."
    ),
    max_cost: float | None = typer.Option(None, min=0, help=MAX_COST_HELP),
) -> None:
    """Generate candidate prompts per intent x persona, dedup them, and store them.

    Cells already generated with this template version are skipped unless --force.
    """
    from jinja2 import TemplateNotFound

    from citewise.core.db import session_scope
    from citewise.providers import get_embedder
    from citewise.services.panel.common import BrandNotFoundError, brand_prompts, resolve_brand
    from citewise.services.panel.generate import generate_panel

    settings = get_settings()
    chat = _budgeted_chat(max_cost)
    try:
        with session_scope() as session:
            b = resolve_brand(session, brand)
            report = generate_panel(session, b, chat, get_embedder(), settings, version, force)
            total = len(brand_prompts(session, b.id))
    except BrandNotFoundError as exc:
        _fail(str(exc))
        return
    except TemplateNotFound:
        _fail(f"No template prompts/panel/generate.{version}.jinja")
        return
    typer.echo(
        f"{b.name}: {report.candidates} candidates from template panel/generate.{version} "
        f"({report.failed_calls} failed cells, {report.skipped_cells} already generated)\n"
        f"  dropped: {report.invalid} invalid, {report.exact_duplicates} exact duplicates, "
        f"{report.near_duplicates} near-duplicates (cosine > {settings.panel_dedup_threshold})\n"
        f"  stored: {report.stored} new, {total} total prompts for {b.name}"
    )
    for dropped, kept in report.examples[:3]:
        typer.echo(f"  e.g. dropped {dropped!r} (kept {kept!r})")
    _budget_note(chat)


@panel_app.command("import")
def import_(
    brand: str = typer.Argument(..., help="Brand name or id."),
    file: Path = typer.Argument(..., exists=True, dir_okay=False, help="CSV with a text column."),
) -> None:
    """Import prompts from CSV (columns: text, optional intent, persona, funnel_stage)."""
    from citewise.core.db import session_scope
    from citewise.providers import get_embedder
    from citewise.services.panel.common import BrandNotFoundError, resolve_brand
    from citewise.services.panel.generate import import_csv

    try:
        with session_scope() as session:
            b = resolve_brand(session, brand)
            report = import_csv(session, b, file, get_embedder(), get_settings())
    except (BrandNotFoundError, ValueError) as exc:
        _fail(str(exc))
        return
    typer.echo(
        f"{b.name}: {report.candidates} rows, {report.invalid} invalid, "
        f"{report.exact_duplicates} exact + {report.near_duplicates} near duplicates, "
        f"{report.stored} stored."
    )


@panel_app.command()
def cluster(
    brand: str = typer.Argument(..., help="Brand name or id."),
    algorithm: str = typer.Option("kmeans", help="kmeans (silhouette-selected k) or hdbscan."),
    seed: int = typer.Option(0, help="Random seed for KMeans."),
    max_cost: float | None = typer.Option(None, min=0, help=MAX_COST_HELP),
) -> None:
    """Cluster prompts into labeled topics and classify any prompt without an intent.

    Replaces the brand's topics and clears the scan panel (re-run `panel build`).
    """
    from citewise.core.db import session_scope
    from citewise.services.panel.cluster import cluster_brand
    from citewise.services.panel.common import BrandNotFoundError, resolve_brand

    chat = _budgeted_chat(max_cost)
    try:
        with session_scope() as session:
            b = resolve_brand(session, brand)
            report = cluster_brand(session, b, chat, get_settings(), algorithm, seed)
    except (BrandNotFoundError, ValueError) as exc:
        _fail(str(exc))
        return
    if report.silhouette_by_k:
        scores = ", ".join(f"{k}:{v:.3f}" for k, v in report.silhouette_by_k.items())
        typer.echo(f"silhouette by k: {scores}")
    typer.echo(f"{b.name}: {report.k} topics ({report.algorithm})")
    typer.echo(f"  {'size':>4}  {'source':<8}  label")
    for label, size, source in report.topics:
        typer.echo(f"  {size:>4}  {source:<8}  {label}")
    typer.echo(f"intents: {dict(sorted(report.intents.items()))}")
    typer.echo(f"intent sources: {dict(sorted(report.intent_sources.items()))}")
    _budget_note(chat)
    typer.echo("scan panel cleared: run `citewise panel build` to pick a new one")


@panel_app.command("eval-intent")
def eval_intent(
    brand: str = typer.Argument(..., help="Brand name or id."),
    golden: Path | None = typer.Option(
        None, help=f"Also score both classifiers on hand labels (default {GOLDEN_INTENTS})."
    ),
    folds: int = typer.Option(5, help="Cross-validation folds."),
) -> None:
    """Agreement between the LLM and embedding intent classifiers (accuracy, macro-F1)."""
    from citewise.core.db import session_scope
    from citewise.services.panel.common import (
        BrandNotFoundError,
        as_matrix,
        brand_prompts,
        resolve_brand,
    )
    from citewise.services.panel.intent import (
        TEACHER_SOURCES,
        cross_validated_agreement,
        format_agreement,
    )

    try:
        with session_scope() as session:
            b = resolve_brand(session, brand)
            teacher = [
                p
                for p in brand_prompts(session, b.id)
                if p.intent and p.embedding and p.intent_source in TEACHER_SOURCES
            ]
            if not teacher:
                _fail(f"{b.name} has no LLM-labeled prompts. Run `citewise panel cluster` first.")
            x = as_matrix([p.embedding for p in teacher if p.embedding])
            labels = [p.intent for p in teacher if p.intent]
            agreement = cross_validated_agreement(x, labels, folds=folds)
            typer.echo(
                format_agreement(
                    agreement,
                    f"{b.name}: embedding classifier vs LLM labels ({folds}-fold cross-validated)",
                )
            )
            if golden is not None:
                _golden_report(b, x, labels, golden)
    except (BrandNotFoundError, ValueError) as exc:
        _fail(str(exc))


def _golden_report(brand, x, labels: list[str], path: Path) -> None:  # type: ignore[no-untyped-def]
    """Score both classifiers against hand labels (held out: never in the training set)."""
    from citewise.providers import get_chat_provider, get_embedder
    from citewise.services.panel.common import as_matrix
    from citewise.services.panel.intent import (
        EmbeddingIntentClassifier,
        LLMIntentClassifier,
        format_agreement,
        score,
    )

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = [r for r in rows if r.get("brand", "").casefold() == brand.name.casefold()]
    if not rows:
        typer.echo(f"\n(no golden rows for {brand.name} in {path})")
        return
    texts, gold = [r["text"] for r in rows], [r["intent"] for r in rows]
    settings = get_settings()

    clf = EmbeddingIntentClassifier().fit(x, labels)
    vectors = as_matrix(get_embedder().embed(texts, purpose="panel.embed"))
    emb_pred = [p or "none" for p in clf.predict_vectors(vectors)]
    typer.echo("\n" + format_agreement(score(gold, emb_pred), "Embedding classifier vs golden"))

    if settings.is_mock:
        typer.echo(
            "\nLLM classifier vs golden: skipped in mock mode (no recorded answers for the "
            "held-out set; run with CITEWISE_MODE=live to score it)."
        )
        return
    llm = LLMIntentClassifier(
        get_chat_provider("openai"),
        model=settings.model_extract,
        category=brand.category,
        batch_size=settings.panel_classify_batch,
    )
    llm_pred = [p or "none" for p in llm.predict(texts)]
    typer.echo("\n" + format_agreement(score(gold, llm_pred), "LLM classifier vs golden"))


@panel_app.command()
def build(
    brand: str = typer.Argument(..., help="Brand name or id."),
    size: int = typer.Option(50, min=1, help="Number of prompts in the scan panel."),
    seed: int = typer.Option(0, help="Sampling seed."),
) -> None:
    """Pick a stratified panel (topics x intents) for scanning. Re-running replaces it."""
    from citewise.core.db import session_scope
    from citewise.services.panel.common import BrandNotFoundError, resolve_brand
    from citewise.services.panel.select import build_panel

    try:
        with session_scope() as session:
            b = resolve_brand(session, brand)
            report = build_panel(session, b, size, seed)
    except (BrandNotFoundError, ValueError) as exc:
        _fail(str(exc))
        return
    typer.echo(
        f"{b.name}: panel of {len(report.selected)} prompts covering "
        f"{report.topics_covered}/{report.topics_total} topics ({report.topic_coverage:.0%})"
    )
    typer.echo(f"intents: {dict(sorted(report.intents.items()))}")
