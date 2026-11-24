"""`citewise kb ...`, `citewise factcheck ...` and `citewise eval retrieval` (Module 4)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import typer

from citewise.core.config import REPO_ROOT, get_settings
from citewise.extract_cli import eval_app
from citewise.panel_cli import MAX_COST_HELP, _budget_note, _budgeted_chat, _fail

kb_app = typer.Typer(help="Brand knowledge base (source of truth).", no_args_is_help=True)
factcheck_app = typer.Typer(help="Check engine claims against the KB.", no_args_is_help=True)

KB_QA = REPO_ROOT / "evals" / "datasets" / "golden" / "kb_qa.jsonl"
RETRIEVAL_REPORT = REPO_ROOT / "evals" / "reports" / "retrieval_eval.md"


@kb_app.command("ingest")
def kb_ingest(
    brand: str = typer.Argument(..., help="Brand name or id."),
    path: str = typer.Argument(..., help="File, folder of .md/.txt/.pdf, or URL (live mode)."),
    category: str | None = typer.Option(
        None, help="company|offerings|audience|competitors|policies (else from front matter)."
    ),
) -> None:
    """Parse, chunk, embed and store source-of-truth documents (re-runnable)."""
    from citewise.core.db import session_scope
    from citewise.providers import get_embedder
    from citewise.services.kb.ingest import IngestError, collect_docs, ingest
    from citewise.services.panel.common import BrandNotFoundError, resolve_brand

    settings = get_settings()
    try:
        docs = collect_docs(path, category, live=not settings.is_mock)
        with session_scope() as session:
            b = resolve_brand(session, brand)
            report = ingest(session, b, docs, get_embedder(), chunk_tokens=settings.kb_chunk_tokens,
                            overlap_tokens=settings.kb_chunk_overlap)  # fmt: skip
    except (IngestError, BrandNotFoundError) as exc:
        _fail(str(exc))
        return
    typer.echo(
        f"{b.name}: {report.docs_added} added, {report.docs_updated} updated, "
        f"{report.docs_unchanged} unchanged; {report.chunks} chunks written"
    )


@kb_app.command("search")
def kb_search(
    brand: str = typer.Argument(..., help="Brand name or id."),
    query: str = typer.Argument(..., help="Question or claim."),
    k: int = typer.Option(5, min=1, max=20),
    mode: str = typer.Option("hybrid", help="bm25, dense or hybrid."),
    rerank: bool = typer.Option(False, help="LLM re-rank of the top 20 (costs a call)."),
) -> None:
    """Show what the KB returns for a query (debug retrieval)."""
    from citewise.core.db import session_scope
    from citewise.providers import get_chat_provider, get_embedder
    from citewise.services.kb.retrieve import KBIndex
    from citewise.services.panel.common import BrandNotFoundError, resolve_brand

    if mode not in ("bm25", "dense", "hybrid"):
        _fail("--mode must be bm25, dense or hybrid")
    settings = get_settings()
    try:
        with session_scope() as session:
            index = KBIndex.load(session, resolve_brand(session, brand).id)
        hits = index.search(query, get_embedder(), k=k, mode=mode,  # type: ignore[arg-type]
                            rerank=get_chat_provider() if rerank else None,
                            rerank_model=settings.model_judge)  # fmt: skip
    except (BrandNotFoundError, ValueError) as exc:
        _fail(str(exc))
        return
    if not hits:
        typer.echo("no results (is the KB ingested?)")
    for i, h in enumerate(hits, start=1):
        ranks = ", ".join(f"{m} #{r}" for m, r in h.ranks.items())
        typer.echo(f"{i}. {h.chunk_key}  ({ranks})\n   {' '.join(h.text.split())[:220]}")


@factcheck_app.command("run")
def factcheck_run_cmd(
    run_id: int = typer.Argument(..., help="Scan run id (extracted with `extract run`)."),
    force: bool = typer.Option(False, "--force", help="Re-check answers already done."),
    rerank: bool = typer.Option(False, help="LLM re-rank of KB evidence (more calls)."),
    max_cost: float | None = typer.Option(None, min=0, help=MAX_COST_HELP),
) -> None:
    """Extract the brand claims in a run's answers and verify each against the KB."""
    from citewise.core.db import get_sessionmaker
    from citewise.providers import get_embedder
    from citewise.services.factcheck.runner import FACTCHECK_VERSION, factcheck_run

    settings = get_settings()
    chat = _budgeted_chat(max_cost)
    try:
        r = factcheck_run(get_sessionmaker(), run_id, chat, get_embedder(), settings,
                          force=force, rerank=rerank)  # fmt: skip
    except ValueError as exc:
        _fail(str(exc))
        return
    verdicts = ", ".join(f"{k} {v}" for k, v in r.verdicts.most_common()) or "none"
    typer.echo(
        f"run {run_id} [{FACTCHECK_VERSION}]: {r.checked} answers checked ({r.skipped} already "
        f"done) of {r.answers} mentioning the brand; {r.claims} claims: {verdicts}"
    )
    typer.echo(f"  contradicted claims traced to a likely source: {r.traced}")
    if r.downgraded:
        typer.echo(f"  {r.downgraded} verdicts downgraded to not_in_kb (no verbatim KB quote)")
    if r.llm_failed or r.budget_stopped:
        typer.secho(f"  {r.llm_failed} LLM failures, {r.budget_stopped} left for budget; "
                    "re-run to retry", fg=typer.colors.YELLOW)  # fmt: skip
    _budget_note(chat)


@factcheck_app.command("show")
def factcheck_show(
    brand: str = typer.Argument(..., help="Brand name or id."),
    run: int | None = typer.Option(None, "--run", help="Run id (default: latest checked)."),
    verdict: str = typer.Option("contradicted", help="contradicted|supported|not_in_kb|all"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Each wrong claim with the KB evidence quote, the engine, and the likely source URL."""
    from citewise.core.db import session_scope
    from citewise.services.factcheck.report import summarize
    from citewise.services.panel.common import BrandNotFoundError, resolve_brand

    try:
        with session_scope() as session:
            b = resolve_brand(session, brand)
            s = summarize(session, b.id, run_id=run, verdict=None if verdict == "all" else verdict)
            name = b.name
    except BrandNotFoundError as exc:
        _fail(str(exc))
        return
    if as_json:
        typer.echo(json.dumps(s.as_dict(), indent=2, default=str))
        return
    if s.run_id is None:
        _fail(f"{name} has no fact-checked runs: run `citewise factcheck run <run_id>` first")
    typer.echo(f"{name}, run {s.run_id}: {s.claims} claims {s.verdicts}")
    for e, counts in s.by_engine.items():
        typer.echo(f"  {e:<11} {counts}")
    if s.top_sources:
        culprits = ", ".join(f"{x['domain']} ({x['contradicted_claims']})" for x in s.top_sources)
        typer.echo(f"  sources behind wrong claims: {culprits}")
    for c in s.items:
        typer.echo(f"\n[{c.verdict}] ({c.engine}) {c.claim}")
        if c.evidence_quote:
            typer.echo(f"  KB says: \"{' '.join(c.evidence_quote.split())}\"  "
                       f"[{', '.join(c.evidence_chunk_keys)}]")  # fmt: skip
        if c.source_url:
            typer.echo(f"  likely source: {c.source_url}  "
                       f"({c.source_reason}, confidence {c.source_confidence:.2f})")  # fmt: skip


@factcheck_app.command("narrative")
def factcheck_narrative(
    brand: str = typer.Argument(..., help="Brand name or id."),
    run: int | None = typer.Option(None, "--run", help="Run id (default: latest extracted)."),
) -> None:
    """Per engine: the top positive and negative themes in how it describes the brand."""
    from citewise.core.db import session_scope
    from citewise.providers import get_chat_provider, get_embedder
    from citewise.services.factcheck.narrative import narratives
    from citewise.services.metrics.visibility import NoExtractionsError, latest_extracted_run
    from citewise.services.panel.common import BrandNotFoundError, resolve_brand

    settings = get_settings()
    try:
        with session_scope() as session:
            b = resolve_brand(session, brand)
            run_id = run or latest_extracted_run(session, b.id)
            out = narratives(session, b.id, b.name, run_id, get_embedder(), get_chat_provider(),
                             settings.model_judge)  # fmt: skip
    except (BrandNotFoundError, NoExtractionsError, ValueError) as exc:
        _fail(str(exc))
        return
    for n in out:
        typer.echo(f"\n{n.engine} ({n.mentions} brand mentions)")
        for side, themes in (("+", n.positive), ("-", n.negative)):
            for t in themes:
                typer.echo(
                    f'  {side} {t.label} ({t.count}, {t.share:.0%})  e.g. "{t.example[:90]}"'
                )


@eval_app.command("retrieval")
def eval_retrieval(
    golden: Path = typer.Option(KB_QA, help="Golden set (JSONL)."),
    out: Path = typer.Option(RETRIEVAL_REPORT, help="Markdown report path."),
    rerank: bool = typer.Option(False, help="Also score hybrid + LLM re-rank (costs calls)."),
    max_cost: float | None = typer.Option(None, min=0, help=MAX_COST_HELP),
) -> None:
    """Recall@1/@5 and MRR for BM25, dense, hybrid (and hybrid + re-rank)."""
    from citewise.evals.retrieval import evaluate_retrieval, load_rows, render_markdown
    from citewise.providers import get_embedder

    settings = get_settings()
    embedder = get_embedder()
    chat = _budgeted_chat(max_cost) if rerank else None
    report = evaluate_retrieval(load_rows(golden), settings.brands_dir.parent / "kb", embedder,
                                mode=settings.citewise_mode, rerank=chat,
                                rerank_model=settings.model_judge)  # fmt: skip
    for m, s in report.overall.items():
        x = s.summary()
        typer.echo(f"{m:<14} recall@1 {x['recall@1']:.3f}  recall@5 {x['recall@5']:.3f}  "
                   f"MRR {x['mrr']:.3f}")  # fmt: skip
    rel = golden.relative_to(REPO_ROOT) if golden.is_relative_to(REPO_ROOT) else golden
    out.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.now(UTC).date().isoformat()
    out.write_text(render_markdown(report, rel.as_posix(), today, embedder.ident), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(report.as_dict(), indent=2) + "\n",
                                        encoding="utf-8")  # fmt: skip
    typer.echo(f"report: {out}")
    if chat is not None:
        _budget_note(chat)
