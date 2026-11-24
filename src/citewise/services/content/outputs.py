"""Publish-ready outputs: Markdown with source footnotes, FAQPage JSON-LD, the
"why this should help" note.

The stored draft keeps `[kb:<chunk key>]` markers (the checker needs them). Export turns
them into numbered footnotes that name the KB document and its public URL, and strips
them from the JSON-LD (search engines read the answer text verbatim).
"""

from __future__ import annotations

import re
from typing import Any

from citewise.services.content.checks import MARKER, marker_keys, strip_markers

# What Google's FAQ rich-result guidelines and schema.org require of an FAQPage: at least
# one Question, each with a name and exactly one acceptedAnswer with text. Used by the
# finalize step (validate_faq_jsonld) and by the schema test (jsonschema).
FAQ_JSONLD_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["@context", "@type", "mainEntity"],
    "properties": {
        "@context": {"const": "https://schema.org"},
        "@type": {"const": "FAQPage"},
        "name": {"type": "string", "minLength": 1},
        "mainEntity": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["@type", "name", "acceptedAnswer"],
                "additionalProperties": False,
                "properties": {
                    "@type": {"const": "Question"},
                    "name": {"type": "string", "minLength": 3},
                    "acceptedAnswer": {
                        "type": "object",
                        "required": ["@type", "text"],
                        "additionalProperties": False,
                        "properties": {
                            "@type": {"const": "Answer"},
                            # No leftover citation markers in the published answer.
                            "text": {
                                "type": "string",
                                "minLength": 1,
                                "not": {"pattern": r"\[kb:"},
                            },
                        },
                    },
                },
            },
        },
    },
}


def render_markdown(draft: dict[str, Any]) -> str:
    """The page as stored (markers kept): H1, direct answer, sections, FAQ."""
    parts = [f"# {draft.get('title', '').strip()}", draft.get("summary", "").strip()]
    for s in draft.get("sections", []):
        parts += [f"## {s.get('heading', '').strip()}", s.get("body", "").strip()]
    faq = draft.get("faq", [])
    if faq:
        parts.append("## Frequently asked questions")
        for q in faq:
            parts += [f"### {q.get('question', '').strip()}", q.get("answer", "").strip()]
    return "\n\n".join(p for p in parts if p) + "\n"


def export_markdown(draft: dict[str, Any], sources: dict[str, dict[str, str]]) -> str:
    """Markers -> numbered footnotes; a Sources list maps each to its KB document.

    `sources` is chunk key -> {"title", "url"}.
    """
    md = render_markdown(draft)
    order: dict[str, int] = {}
    for key in marker_keys(md):
        order.setdefault(key, len(order) + 1)

    def repl(m: re.Match[str]) -> str:
        keys = marker_keys(m.group(0))
        return "".join(f"[^{order[k]}]" for k in dict.fromkeys(keys) if k in order)

    body = MARKER.sub(repl, md).rstrip() + "\n"
    if order:
        lines = ["", "---", ""]
        for key, n in order.items():
            src = sources.get(key, {})
            title, url = src.get("title") or key, src.get("url") or ""
            lines.append(f"[^{n}]: {title}{f' ({url})' if url.startswith('http') else ''}")
        body += "\n".join(lines) + "\n"
    return body


def faq_jsonld(draft: dict[str, Any]) -> dict[str, Any]:
    return {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "name": strip_markers(draft.get("title", "")),
        "mainEntity": [
            {
                "@type": "Question",
                "name": strip_markers(q.get("question", "")),
                "acceptedAnswer": {"@type": "Answer", "text": strip_markers(q.get("answer", ""))},
            }
            for q in draft.get("faq", [])
            if strip_markers(q.get("answer", ""))
        ],
    }


def validate_faq_jsonld(doc: dict[str, Any]) -> list[str]:
    """The FAQ_JSONLD_SCHEMA rules without a jsonschema dependency at runtime."""
    errors = []
    if doc.get("@context") != "https://schema.org" or doc.get("@type") != "FAQPage":
        errors.append("@context/@type must be https://schema.org / FAQPage")
    entities = doc.get("mainEntity")
    if not isinstance(entities, list) or not entities:
        return [*errors, "mainEntity needs at least one Question"]
    for i, q in enumerate(entities):
        ans = q.get("acceptedAnswer") or {}
        if q.get("@type") != "Question" or len(str(q.get("name", ""))) < 3:
            errors.append(f"mainEntity[{i}] must be a Question with a name")
        if ans.get("@type") != "Answer" or not ans.get("text"):
            errors.append(f"mainEntity[{i}].acceptedAnswer must be an Answer with text")
        elif "[kb:" in ans["text"]:
            errors.append(f"mainEntity[{i}] answer still contains a citation marker")
    return errors


def why_note(kind: str, details: dict[str, Any], brand: str, targets: list[str]) -> str:
    """One paragraph tying the page to the gap it targets and how to measure it."""
    quoted = "; ".join(f'"{t}"' for t in targets[:3])
    if kind == "gap":
        why = (
            f"In the scan, engines mentioned {brand} in {details.get('brand_rate', 0):.0%} of "
            f"'{details.get('topic')}' ({details.get('intent')}) answers vs "
            f"{details.get('rival')} in {details.get('rival_rate', 0):.0%} "
            f"({details.get('answers')} answers). This page answers those prompts directly "
            "with KB-cited facts, in the structure engines quote (direct answer, question "
            "headings, FAQ)."
        )
    elif kind == "fact_fix":
        why = (
            f"Engines ({', '.join(details.get('engines', []))}) repeat a wrong claim: "
            f'"{details.get("claim")}". The KB says: "{details.get("evidence_quote")}". A '
            "page that states the correct fact plainly gives retrieval-based engines a "
            "source to cite instead."
        )
    else:
        why = "This page answers the target prompts with KB-cited facts."
    return (
        f"{why} Target prompts: {quoted}. To measure: after publishing, re-scan and compare "
        f"{brand}'s mention rate on these prompts (`citewise content measure <id>`)."
    )
