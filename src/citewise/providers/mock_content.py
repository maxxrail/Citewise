"""Simulated content-pipeline model calls for mock mode (brief, outline, draft, critique,
revise).

Like mock_factcheck.py, this reads only the rendered prompt: the brief, the passages, the
draft and the issues the critique listed. Nothing is planted. The simulated writer works
the way a hurried first pass does: it copies passage sentences verbatim with their
[kb:id] marker (so it is grounded), writes only the first two outline sections and a
two-question FAQ, and adds no comparison table. It then revises *only* what the critique
or the reviewer names: unanswered questions, missing must-cover passages, too few FAQs, a
missing table, flagged sentences, reviewer comments. So in mock mode the scores rise
because the revise loop works, not because v2 was scripted to be better. Requests for
claims no passage supports (a statistic, a quote, an insult) are declined, which is
what the prompt asks a real model to do; the deterministic checks are the backstop
either way.
"""

from __future__ import annotations

import json
import re
from typing import Any

from citewise.providers.base import Message

PURPOSES = (
    "content.brief",
    "content.outline",
    "content.draft",
    "content.critique",
    "content.revise",
)
_WORD = re.compile(r"[a-z0-9$]+")
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "with",
        "you",
        "your",
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "should",
        "about",
        "best",
        "vs",
    ]
)
_FACT = re.compile(
    r"\$|\d|\b(?:costs?|offers?|includes?|supports?|sells?|ships?|trial|"
    r"warranty|plan|import|integrat|available|made|manufactured|designed|"
    r"founded|width|drop|portal|free|returns?)",
    re.IGNORECASE,
)
_INSTRUCTION = re.compile(
    r"ignore (?:the|all|previous)|disregard|system note|instead write|"
    r"use (?:the )?code|promo code",
    re.IGNORECASE,
)
_FORBIDDEN_ASK = re.compile(
    r"\b(?:scam|fraud|rip-?off|quote|testimonial|review|stars?|rated|#\s?1|number one|"
    r"\d+ ?%|\d+ out of \d+|keyword|\d{4,} words)\b",
    re.IGNORECASE,
)


def _line_after(user: str, header: str) -> Any:
    m = re.search(rf"^{re.escape(header)}[^\n]*\n(.*)$", user, re.MULTILINE)
    return json.loads(m.group(1)) if m else None


def _field(user: str, label: str) -> Any:
    m = re.search(rf"^{re.escape(label)}: (.*)$", user, re.MULTILINE)
    return json.loads(m.group(1)) if m else None


def _user(messages: list[Message]) -> str:
    return next(m["content"] for m in reversed(messages) if m["role"] == "user")


def _terms(text: str) -> set[str]:
    return {w.rstrip("s") for w in _WORD.findall(text.casefold()) if w not in _STOP}


def _sentences(passage: str) -> list[str]:
    body = passage.split("\n", 1)[-1]  # drop the "Doc title > Section" line
    out = []
    for s in re.split(r"(?<=[.!?])\s+", " ".join(body.split())):
        s = s.strip()
        if len(s.split()) >= 4 and _FACT.search(s) and not _INSTRUCTION.search(s):
            out.append(s)
    return out


def _cited(sentence: str, key: str) -> str:
    s = sentence.rstrip()
    return f"{s[:-1] if s.endswith('.') else s} [kb:{key}]."


def _best(
    query: str, passages: list[dict[str, str]], *, n: int = 2, avoid: set[str] | None = None
) -> list[tuple[str, str]]:
    """(sentence, key) pairs from the passages that overlap the query most."""
    q = _terms(query)
    scored = []
    for p in passages:
        for i, s in enumerate(_sentences(p["text"])):
            if avoid and s in avoid:
                continue
            overlap = len(q & _terms(s))
            scored.append((overlap, -i, s, p["id"]))
    scored.sort(key=lambda t: (-t[0], -t[1]))
    return [(s, k) for o, _, s, k in scored[:n] if o > 0] or [(s, k) for _, _, s, k in scored[:1]]


def _question(text: str) -> str:
    t = text.strip().rstrip("?.")
    return (t[:1].upper() + t[1:] + "?") if t else t


# ---------------------------------------------------------------- steps


def simulate_brief(messages: list[Message]) -> dict[str, Any]:
    user = _user(messages)
    passages = _line_after(user, "Passages (JSON list") or []
    targets = _field(user, "Target prompts") or []
    rival = _field(user, "Rival") or ""
    notes = _field(user, "Notes") or ""
    fmt = _field(user, "Suggested format") or "guide"
    brand = _field(user, "Brand") or ""
    questions = [_question(t) for t in targets[:4]]
    extra = [f"What does {brand} cost?", f"What is {brand}'s return or refund policy?"]
    questions += [q for q in extra if len(questions) < 3]
    ranked = []
    for q in questions:
        ranked += [k for _, k in _best(q, passages, n=2)]
    must = list(dict.fromkeys(ranked))[:4]
    comparison = []
    if rival:
        comparison = [
            s for p in passages for s in _sentences(p["text"]) if rival.casefold() in s.casefold()
        ][:2]
    known = " ".join(p["text"] for p in passages)
    unsupported = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", notes)
        if _FORBIDDEN_ASK.search(s) and not all(n in known for n in re.findall(r"\d+", s))
    ]
    return {
        "audience": f"People asking AI assistants: {targets[0] if targets else brand}",
        "angle": f"Exactly what {brand} offers, stated from its own documentation",
        "questions": questions,
        "must_cover": must,
        "comparison_points": comparison,
        "format": fmt if fmt in ("faq", "comparison", "guide") else "guide",
        "word_count": 700,
        "unsupported_requests": unsupported,
    }


def simulate_outline(messages: list[Message]) -> dict[str, Any]:
    user = _user(messages)
    brief = _field(user, "Brief") or {}
    passages = _line_after(user, "Passages (JSON list") or []
    questions = brief.get("questions") or ["Overview"]
    sections = [
        {"heading": q, "purpose": f"Answer: {q}", "chunk_ids": [k for _, k in _best(q, passages)]}
        for q in questions[1:] or questions
    ]
    return {
        "title": questions[0].rstrip("?"),
        "direct_answer": f"Answer '{questions[0]}' in the first sentence.",
        "sections": sections,
        "faq_questions": questions[:4],
    }


def _section(heading: str, passages: list[dict[str, str]], used: set[str]) -> dict[str, str]:
    picks = _best(heading, passages, n=2, avoid=used)
    used.update(s for s, _ in picks)
    return {"heading": heading, "body": " ".join(_cited(s, k) for s, k in picks)}


def _faq(
    question: str, passages: list[dict[str, str]], used: set[str] | None = None
) -> dict[str, str]:
    [(s, k)] = _best(question, passages, n=1, avoid=used)
    if used is not None:
        used.add(s)
    return {"question": question, "answer": _cited(s, k)}


_NAMED = re.compile(r"^The ((?:[A-Z][\w-]*\s?){1,4})\b(?:is|offers|costs|has)\b")


def _table(passages: list[dict[str, str]]) -> str:
    """One row per priced option, labelled with the product the passage names."""
    rows: list[tuple[str, str, str]] = []
    for p in passages:
        name = ""
        for sent in _sentences(p["text"]):
            m = _NAMED.match(sent)
            name = m.group(1).strip() if m else name
            if "$" in sent and name:
                rows.append((name, sent, p["id"]))
    lines = ["| Option | Detail |", "| --- | --- |"]
    for name, sent, key in list(dict.fromkeys(rows))[:4]:
        lines.append(f"| {name} | {_cited(sent, key)} |")
    return "\n".join(lines) if len(lines) > 2 else ""


def simulate_draft(messages: list[Message]) -> dict[str, Any]:
    user = _user(messages)
    brief = _field(user, "Brief") or {}
    outline = _field(user, "Outline") or {}
    passages = _line_after(user, "Passages (JSON list") or []
    questions = brief.get("questions") or [outline.get("title", "")]
    used: set[str] = set()
    [(s, k)] = _best(questions[0], passages, n=1)
    used.add(s)
    sections = [_section(sec["heading"], passages, used) for sec in outline.get("sections", [])[:2]]
    faq = [_faq(q, passages, used) for q in outline.get("faq_questions", questions)[:2]]
    return {
        "title": outline.get("title") or questions[0],
        "summary": _cited(s, k),
        "sections": sections,
        "faq": faq,
    }


def simulate_critique(messages: list[Message]) -> dict[str, Any]:
    """A lenient judge: the deterministic caps do the strict checking. It does read the
    reviewer comments and marks coverage down if the draft ignores one."""
    user = _user(messages)
    comments = _field(user, "Reviewer comments") or []
    draft = _line_after(user, "Draft:") or ""
    dterms = _terms(draft)
    unaddressed = [
        c for c in comments if not _FORBIDDEN_ASK.search(c) and len(_terms(c) & dterms) < 2
    ]
    scores = []
    for c in ("factuality", "coverage", "tone", "aeo_structure", "originality"):
        issues = (
            [f"Address the reviewer comment: {x}" for x in unaddressed] if c == "coverage" else []
        )
        scores.append(
            {
                "criterion": c,
                "rationale": "mock judge",
                "issues": issues,
                "score": 3 if issues else 5,
            }
        )
    return {"scores": scores}


def simulate_revise(messages: list[Message]) -> dict[str, Any]:
    user = _user(messages)
    brief = _field(user, "Brief") or {}
    issues: dict[str, list[str]] = _field(user, "Failing criteria and issues") or {}
    comments: list[str] = _field(user, "Reviewer comments") or []
    passages = _line_after(user, "Passages (JSON list") or []
    draft: dict[str, Any] = _line_after(user, "Draft:") or {}
    sections = [dict(s) for s in draft.get("sections", [])]
    faq = [dict(q) for q in draft.get("faq", [])]
    summary = draft.get("summary", "")
    changes: list[str] = []
    declined: list[dict[str, str]] = []
    flat = [i for v in issues.values() for i in v]
    used = {s for sec in sections for s in re.split(r"(?<=\.)\s+", sec["body"])}

    for issue in flat:  # drop sentences a check rejected
        m = re.match(
            r"(?:Unsupported \([a-z_]+\)|Fake review/testimonial|Instruction-like text, "
            r"remove|Disparages a competitor, remove): (.*)",
            issue,
        )
        if m:
            bad = m.group(1)[:60]
            for sec in sections:
                kept = [s for s in re.split(r"(?<=[.!?])\s+", sec["body"]) if bad not in s]
                sec["body"] = " ".join(kept)
            changes.append(f"Removed: {bad}")
    for issue in flat:
        if issue.startswith("Answer this question: "):
            q = issue.removeprefix("Answer this question: ")
            sections.append(_section(q, passages, used))
            changes.append(f"Added a section answering: {q}")
        elif issue.startswith("Use must-cover passage [kb:"):
            key = issue.removeprefix("Use must-cover passage [kb:").rstrip("]")
            p = next((p for p in passages if p["id"] == key), None)
            if p and _sentences(p["text"]):
                s = _sentences(p["text"])[0]
                target = sections[-1] if sections else {"heading": "Details", "body": ""}
                target["body"] = f"{target['body']} {_cited(s, key)}".strip()
                if not sections:
                    sections.append(target)
                changes.append(f"Used passage {key}")
        elif issue.startswith("State the correct fact: "):
            [(s, k)] = _best(issue.removeprefix("State the correct fact: "), passages, n=1)
            summary = _cited(s, k)
            changes.append("Opened with the correct fact from the KB")
        elif issue == "Fix: faq section":
            have = {q["question"] for q in faq}
            for q in [*brief.get("questions", []), *(s["heading"] for s in sections)]:
                if len(faq) >= 4:
                    break
                if q not in have:
                    faq.append(_faq(q, passages, used))
                    have.add(q)
            changes.append("Expanded the FAQ")
        elif issue == "Fix: comparison table":
            table = _table(passages)
            if table:
                sections.append({"heading": "How the options compare", "body": table})
                changes.append("Added a comparison table")
            else:
                declined.append(
                    {
                        "request": "comparison table",
                        "reason": "no passage gives comparable priced options",
                    }
                )
        elif issue == "Fix: direct answer":
            question = (brief.get("questions") or [""])[0].rstrip("?")
            [(s, k)] = _best(question, passages, n=1)
            summary = f"On {question[:1].lower() + question[1:]}: {_cited(s, k)}"
            changes.append("Rewrote the opening as a direct answer")
        elif issue.startswith("Remove repeated sentences"):
            seen_answers: set[str] = {summary}
            for qa in faq:
                if qa["answer"] in seen_answers:
                    qa.update(_faq(qa["question"], passages, used))
                seen_answers.add(qa["answer"])
            changes.append("Replaced repeated FAQ answers")
    for comment in comments:
        if _FORBIDDEN_ASK.search(comment):
            declined.append(
                {
                    "request": comment,
                    "reason": "asks for a claim, quote, insult or "
                    "length the passages and guardrails don't allow",
                }
            )
            continue
        picks = _best(comment, passages, n=2, avoid=used)
        used.update(s for s, _ in picks)
        heading = _question(
            " ".join(
                w
                for w in comment.split()
                if w.casefold() not in {"please", "add", "mention", "include", "also"}
            )[:80]
        )
        sections.append({"heading": heading, "body": " ".join(_cited(s, k) for s, k in picks)})
        changes.append(f"Addressed reviewer comment: {comment}")
    sections = [s for s in sections if s["body"].strip()]
    return {
        "title": draft.get("title", ""),
        "summary": summary,
        "sections": sections,
        "faq": faq,
        "changes": changes,
        "declined": declined,
    }


def simulate(purpose: str, messages: list[Message]) -> dict[str, Any]:
    return {
        "content.brief": simulate_brief,
        "content.outline": simulate_outline,
        "content.draft": simulate_draft,
        "content.critique": simulate_critique,
        "content.revise": simulate_revise,
    }[purpose](messages)
