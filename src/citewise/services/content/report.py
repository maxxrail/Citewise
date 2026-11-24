"""Read-side views of content items, shared by the API and the CLI."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from citewise.models.content import ContentItem, ContentOpportunity, ContentReview, ContentVersion


def opportunity_dict(o: ContentOpportunity) -> dict[str, Any]:
    return {
        "id": o.id,
        "kind": o.kind,
        "title": o.title,
        "reason": o.reason,
        "score": o.score,
        "impact": o.impact,
        "confidence": o.confidence,
        "suggested_format": o.suggested_format,
        "status": o.status,
        "run_id": o.run_id,
        "details": o.details,
    }


def item_summary(item: ContentItem, versions: list[ContentVersion] | None = None) -> dict[str, Any]:
    current = next((v for v in versions or [] if v.version == item.current_version), None)
    return {
        "id": item.id,
        "brand_id": item.brand_id,
        "opportunity_id": item.opportunity_id,
        "status": item.status,
        "title": item.title,
        "format": item.format,
        "versions": item.current_version,
        "loops": item.loops,
        "score": current.scores.get("overall") if current else None,
        "passed": current.scores.get("passed") if current else None,
        "error": item.error,
        "updated_at": item.updated_at.isoformat() if item.updated_at else None,
    }


def versions_of(session: Session, item_id: int) -> list[ContentVersion]:
    return list(
        session.scalars(
            select(ContentVersion)
            .where(ContentVersion.item_id == item_id)
            .order_by(ContentVersion.version)
        )
    )


def item_detail(session: Session, item: ContentItem) -> dict[str, Any]:
    versions = versions_of(session, item.id)
    reviews = session.scalars(
        select(ContentReview).where(ContentReview.item_id == item.id).order_by(ContentReview.id)
    )
    current = next((v for v in versions if v.version == item.current_version), None)
    brief = {k: v for k, v in (item.brief or {}).items() if k != "passages"}
    return {
        **item_summary(item, versions),
        "brief": brief,
        "passages": [p["id"] for p in (item.brief or {}).get("passages", [])],
        "outline": item.outline,
        "measurement": item.measurement,
        "current": None
        if current is None
        else {
            "version": current.version,
            "kind": current.kind,
            "markdown": current.markdown,
            "jsonld": current.jsonld,
            "note": current.note,
            "checks": current.checks,
            "scores": current.scores,
        },
        "history": [
            {
                "version": v.version,
                "kind": v.kind,
                "overall": v.scores.get("overall"),
                "criteria": {c: s["score"] for c, s in v.scores.get("criteria", {}).items()},
                "guardrails_ok": v.scores.get("guardrails_ok"),
                "feedback": v.feedback,
            }
            for v in versions
        ],
        "reviews": [
            {
                "action": r.action,
                "comment": r.comment,
                "version": r.version,
                "at": r.created_at.isoformat(),
            }
            for r in reviews
        ],
    }
