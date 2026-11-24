"""`citewise content ...` (Module 5): opportunities, the content pipeline, approval."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from citewise.core.config import get_settings
from citewise.panel_cli import MAX_COST_HELP, _budget_note, _budgeted_chat, _fail

content_app = typer.Typer(help="Content agent: find gaps, draft, review.", no_args_is_help=True)


def _deps(max_cost: float | None):  # type: ignore[no-untyped-def]
    from citewise.providers import get_embedder
    from citewise.services.content.pipeline import Deps

    chat = _budgeted_chat(max_cost)
    return chat, Deps(chat, get_embedder(), get_settings())


def _print_item(item_id: int) -> None:
    from citewise.core.db import session_scope
    from citewise.models.content import ContentItem
    from citewise.services.content.report import item_detail

    with session_scope() as session:
        item = session.get(ContentItem, item_id)
        if item is None:
            _fail(f"content {item_id} not found")
            return
        d = item_detail(session, item)
    typer.echo(f"content {d['id']}: {d['title']}  [{d['status']}]  format={d['format']}")
    for h in d["history"]:
        crit = " ".join(f"{c[:4]}={s}" for c, s in h["criteria"].items())
        flag = "" if h["guardrails_ok"] else "  GUARDRAIL"
        typer.echo(f"  v{h['version']} {h['kind']:<9} overall={h['overall']}  {crit}{flag}")
        for f in h["feedback"]:
            if f.get("source") == "reviewer":
                typer.echo(f"      reviewer: {f['comment']}")
            elif f.get("source") == "writer" and f.get("declined"):
                for x in f["declined"]:
                    typer.echo(f"      declined: {x['request'][:80]} ({x['reason']})")
    if d["error"]:
        typer.secho(f"  error: {d['error']}", fg=typer.colors.RED)


def _run(item_id: int, deps, chat) -> None:  # type: ignore[no-untyped-def]
    from citewise.core.db import get_sessionmaker
    from citewise.providers.budget import BudgetExceededError
    from citewise.services.content.pipeline import ContentError, run_item

    try:
        run_item(get_sessionmaker(), item_id, deps)
    except BudgetExceededError as exc:
        typer.secho(
            f"stopped: {exc} (resume with `citewise content resume {item_id}`)",
            fg=typer.colors.YELLOW,
        )
    except ContentError as exc:
        _fail(str(exc))
    _print_item(item_id)
    _budget_note(chat)


@content_app.command("opportunities")
def opportunities(
    brand: str = typer.Argument(..., help="Brand name or id."),
    run_id: int | None = typer.Option(None, help="Scan to read (default: latest extracted)."),
    kind: str | None = typer.Option(None, help="gap, citation or fact_fix."),
    limit: int = typer.Option(15, min=1, max=500),
) -> None:
    """Find and rank content opportunities (gaps, citation gaps, fact fixes), with reasons."""
    from citewise.core.db import session_scope
    from citewise.services.content.opportunities import refresh_opportunities
    from citewise.services.metrics.visibility import NoExtractionsError
    from citewise.services.panel.common import BrandNotFoundError, resolve_brand

    try:
        with session_scope() as session:
            b = resolve_brand(session, brand)
            opps = refresh_opportunities(session, b, run_id=run_id)
            rows = [
                (o.id, o.kind, o.score, o.title, o.reason, o.status, o.suggested_format)
                for o in opps
                if kind in (None, o.kind)
            ]
    except (BrandNotFoundError, NoExtractionsError) as exc:
        _fail(str(exc))
        return
    typer.echo(f"{b.name}: {len(rows)} opportunities (score = impact x confidence)")
    for rank, (oid, k, score, title, reason, st, fmt) in enumerate(rows[:limit], start=1):
        status = "" if st == "open" else f" [{st}]"
        typer.echo(f"{rank:>2}. #{oid} {k:<8} score {score:6.2f}  {title}{status}")
        typer.echo(f"      why: {reason}")
        nxt = "pitch (manual outreach)" if fmt == "pitch" else f"citewise content create {oid}"
        typer.echo(f"      next: {nxt}  ({fmt})")


@content_app.command("create")
def create(
    opportunity_id: int = typer.Argument(..., help="Id from `content opportunities`."),
    notes: str = typer.Option("", help="Extra instructions for the brief."),
    max_cost: float | None = typer.Option(None, min=0, help=MAX_COST_HELP),
) -> None:
    """Run brief -> outline -> draft -> critique -> revise (<=2) -> final; stop for approval."""
    from citewise.core.db import session_scope
    from citewise.models.content import ContentOpportunity
    from citewise.services.content.pipeline import ContentError, create_item

    try:
        with session_scope() as session:
            opp = session.get(ContentOpportunity, opportunity_id)
            if opp is None:
                _fail(f"opportunity {opportunity_id} not found (run `content opportunities`)")
                return
            item_id = create_item(session, opp, notes=notes).id
    except ContentError as exc:
        _fail(str(exc))
        return
    chat, deps = _deps(max_cost)
    _run(item_id, deps, chat)


@content_app.command("resume")
def resume(
    item_id: int = typer.Argument(...),
    max_cost: float | None = typer.Option(None, min=0, help=MAX_COST_HELP),
) -> None:
    """Continue an item from its last good state (after an error or the spend cap)."""
    chat, deps = _deps(max_cost)
    _run(item_id, deps, chat)


@content_app.command("list")
def list_items(
    status: str | None = typer.Option(None, help="e.g. awaiting_approval"),
    brand: str | None = typer.Option(None, help="Brand name or id."),
) -> None:
    """List content items (the review queue with --status awaiting_approval)."""
    from sqlalchemy import select

    from citewise.core.db import session_scope
    from citewise.models.content import ContentItem
    from citewise.services.content.report import item_summary, versions_of
    from citewise.services.panel.common import BrandNotFoundError, resolve_brand

    with session_scope() as session:
        q = select(ContentItem).order_by(ContentItem.id)
        if status:
            q = q.where(ContentItem.status == status)
        if brand:
            try:
                q = q.where(ContentItem.brand_id == resolve_brand(session, brand).id)
            except BrandNotFoundError as exc:
                _fail(str(exc))
        rows = [item_summary(i, versions_of(session, i.id)) for i in session.scalars(q)]
    if not rows:
        typer.echo("no content items")
    for r in rows:
        typer.echo(f"#{r['id']} [{r['status']}] v{r['versions']} score={r['score']}  {r['title']}")


@content_app.command("show")
def show(
    item_id: int = typer.Argument(...),
    markdown: bool = typer.Option(True, help="Print the current version's Markdown."),
) -> None:
    """Versions with scores, reviewer feedback, then the current page and its note."""
    from citewise.core.db import session_scope
    from citewise.models.content import ContentItem
    from citewise.services.content.report import item_detail

    _print_item(item_id)
    with session_scope() as session:
        item = session.get(ContentItem, item_id)
        assert item is not None
        cur = item_detail(session, item)["current"]
    if cur and markdown:
        typer.echo("\n" + cur["markdown"])
        if cur["note"]:
            typer.echo(f"Why this should help: {cur['note']}")
        failing = {g for g, ok in cur["checks"].get("guardrails", {}).items() if not ok}
        typer.echo(f"guardrails: {'all pass' if not failing else ', '.join(sorted(failing))}")


@content_app.command("approve")
def approve_cmd(
    item_id: int = typer.Argument(...), comment: str = typer.Option("", help="Optional note.")
) -> None:
    """Approve the current version; records the target prompts' baseline for measurement."""
    from citewise.core.db import session_scope
    from citewise.models.content import ContentItem
    from citewise.services.content.pipeline import ContentError, approve

    try:
        with session_scope() as session:
            item = session.get(ContentItem, item_id)
            if item is None:
                _fail(f"content {item_id} not found")
                return
            approve(session, item, comment)
            base = (item.measurement or {}).get("baseline")
    except ContentError as exc:
        _fail(str(exc))
        return
    typer.echo(f"content {item_id} approved")
    if base:
        typer.echo(f"  baseline (run {base['run_id']}): {json.dumps(base)}")


@content_app.command("reject")
def reject_cmd(
    item_id: int = typer.Argument(...),
    comment: str = typer.Option(..., help="What to change; it drives the revision."),
    revise: bool = typer.Option(True, help="Run the revision now (else `content resume`)."),
    max_cost: float | None = typer.Option(None, min=0, help=MAX_COST_HELP),
) -> None:
    """Reject with a comment; the comment is fed back as a revision."""
    from citewise.core.db import session_scope
    from citewise.models.content import ContentItem
    from citewise.services.content.pipeline import ContentError, reject

    try:
        with session_scope() as session:
            item = session.get(ContentItem, item_id)
            if item is None:
                _fail(f"content {item_id} not found")
                return
            reject(session, item, comment)
    except ContentError as exc:
        _fail(str(exc))
        return
    typer.echo(f"content {item_id} rejected; revising with your comment")
    if revise:
        chat, deps = _deps(max_cost)
        _run(item_id, deps, chat)


@content_app.command("export")
def export(
    item_id: int = typer.Argument(...),
    out: Path = typer.Option(Path("reports/content"), help="Output folder."),
) -> None:
    """Write the page (Markdown with source footnotes), its JSON-LD and the why-note."""
    from citewise.core.db import session_scope
    from citewise.models.content import ContentItem
    from citewise.services.content.outputs import export_markdown
    from citewise.services.content.pipeline import chunk_sources
    from citewise.services.content.report import versions_of

    with session_scope() as session:
        item = session.get(ContentItem, item_id)
        if item is None or item.current_version == 0:
            _fail(f"content {item_id} has nothing to export")
            return
        current = next(
            v for v in versions_of(session, item_id) if v.version == item.current_version
        )
        md = export_markdown(current.draft, chunk_sources(session, item.brand_id))
        jsonld, note, status = current.jsonld, current.note, item.status
    out.mkdir(parents=True, exist_ok=True)
    base = out / f"content-{item_id}-v{current.version}"
    base.with_suffix(".md").write_text(md, encoding="utf-8")
    if jsonld:
        Path(f"{base}.jsonld.json").write_text(json.dumps(jsonld, indent=2), encoding="utf-8")
    if note:
        Path(f"{base}.why.txt").write_text(note + "\n", encoding="utf-8")
    if status != "approved":
        typer.secho(f"note: content {item_id} is {status}, not approved", fg=typer.colors.YELLOW)
    typer.echo(f"wrote {base}.md" + (f", {base}.jsonld.json" if jsonld else ""))


@content_app.command("measure")
def measure_cmd(item_id: int = typer.Argument(...)) -> None:
    """Before/after on the target prompts: approval baseline vs the newest scan since."""
    from citewise.core.db import session_scope
    from citewise.models.content import ContentItem
    from citewise.services.content.measure import measure

    with session_scope() as session:
        item = session.get(ContentItem, item_id)
        if item is None:
            _fail(f"content {item_id} not found")
            return
        result = measure(session, item)
    typer.echo(json.dumps(result, indent=2))
