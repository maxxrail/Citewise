"""Held-out eval of the FactCheck verifier: `evals/datasets/golden/factcheck_claims.jsonl`.

Each row is a claim about a brand with the verdict a careful reader would give against the
brand's KB, written without seeing the verifier's rules or the simulator's planted claims.
For each claim we retrieve evidence (hybrid, k=5) exactly as `factcheck run` does, verify,
and compare. Reported:
- accuracy, per-verdict precision/recall, confusion matrix;
- **false-contradiction rate**: of claims that are not wrong, the share called
  contradicted. A false "the engine is wrong about you" is the costliest error, so this is
  the number to watch;
- evidence hit rate: for correct supported/contradicted verdicts, how often the quote
  contains the KB phrase the labeller marked as settling it.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from citewise.evals.retrieval import build_index, load_rows
from citewise.providers.base import ChatProvider, EmbeddingProvider
from citewise.services.factcheck.checks import contains, verify_claim
from citewise.services.kb.retrieve import KBIndex

VERDICTS = ("supported", "contradicted", "not_in_kb")


@dataclass
class FactcheckEval:
    mode: str
    rows: int
    reviewed: int
    confusion: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    evidence_hits: int = 0
    evidence_total: int = 0
    llm_failed: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        n = sum(sum(c.values()) for c in self.confusion.values())
        return sum(self.confusion[v][v] for v in VERDICTS) / n if n else 0.0

    def precision(self, v: str) -> float | None:
        predicted = sum(self.confusion[e][v] for e in VERDICTS)
        return self.confusion[v][v] / predicted if predicted else None

    def recall(self, v: str) -> float | None:
        actual = sum(self.confusion[v].values())
        return self.confusion[v][v] / actual if actual else None

    @property
    def false_contradiction_rate(self) -> float | None:
        not_wrong = sum(sum(self.confusion[e].values()) for e in ("supported", "not_in_kb"))
        wrongly = sum(self.confusion[e]["contradicted"] for e in ("supported", "not_in_kb"))
        return wrongly / not_wrong if not_wrong else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "rows": self.rows,
            "reviewed": self.reviewed,
            "accuracy": round(self.accuracy, 4),
            "false_contradiction_rate": _r(self.false_contradiction_rate),
            "evidence_hit_rate": _r(self.evidence_hits / self.evidence_total)
            if self.evidence_total
            else None,
            "per_verdict": {
                v: {"precision": _r(self.precision(v)), "recall": _r(self.recall(v))}
                for v in VERDICTS
            },
            "confusion": {e: dict(self.confusion[e]) for e in VERDICTS},
            "llm_failed": self.llm_failed,
            "errors": self.errors,
        }


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 4)


def evaluate_factcheck(
    rows: list[dict[str, Any]],
    kb_root: Path,
    embedder: EmbeddingProvider,
    chat: ChatProvider,
    *,
    model: str,
    mode: str,
    chunk_tokens: int = 400,
    overlap_tokens: int = 60,
) -> FactcheckEval:
    report = FactcheckEval(
        mode=mode, rows=len(rows), reviewed=sum(1 for r in rows if r.get("reviewed"))
    )
    indexes: dict[str, KBIndex] = {}
    for row in rows:
        brand = row["brand"]
        if brand not in indexes:
            indexes[brand] = build_index(
                kb_root / brand.lower(),
                embedder,
                chunk_tokens=chunk_tokens,
                overlap_tokens=overlap_tokens,
            )
        hits = indexes[brand].search(row["claim"], embedder, k=5)
        checked = verify_claim(chat, model=model, brand=brand, claim=row["claim"], hits=hits)
        expected = row["expected_verdict"]
        if checked is None:
            report.llm_failed += 1
            report.errors.append(f"{row['id']}: LLM failed")
            continue
        report.confusion[expected][checked.verdict] += 1
        if checked.verdict != expected:
            report.errors.append(f"{row['id']}: expected {expected}, got {checked.verdict} "
                                 f"({row.get('kind', '')}): {row['claim']}")  # fmt: skip
        elif expected != "not_in_kb" and row.get("evidence_hint"):
            report.evidence_total += 1
            report.evidence_hits += bool(
                checked.quote and contains(checked.quote, row["evidence_hint"])
            )
    return report


def render_markdown(report: FactcheckEval, golden: str, date: str, stage_note: str) -> str:
    d = report.as_dict()

    def pct(v: float | None) -> str:
        return "-" if v is None else f"{v:.1%}"

    lines = [
        "# FactCheck verifier eval (held-out claims)",
        "",
        f"Golden set: `{golden}`: {report.rows} claims, {report.reviewed} hand-reviewed. Mode: "
        f"**{report.mode}**. Generated {date} by `citewise eval factcheck`.",
        "",
        f"> {stage_note}",
        "",
        f"- **Accuracy:** {pct(d['accuracy'])}",
        f"- **False-contradiction rate** (not-wrong claims called contradicted): "
        f"{pct(d['false_contradiction_rate'])}",
        f"- **Evidence hit rate** (correct verdicts whose quote holds the settling phrase): "
        f"{pct(d['evidence_hit_rate'])}",
        "",
        "| Verdict | Precision | Recall |",
        "|---|---|---|",
    ]
    for v in VERDICTS:
        pv = d["per_verdict"][v]
        lines.append(f"| {v} | {pct(pv['precision'])} | {pct(pv['recall'])} |")
    lines += ["", "## Confusion (rows: expected, columns: predicted)", "",
              "| expected \\ predicted | " + " | ".join(VERDICTS) + " |",
              "|---|" + "---|" * len(VERDICTS)]  # fmt: skip
    for e in VERDICTS:
        lines.append(
            f"| {e} | " + " | ".join(str(d["confusion"][e].get(p, 0)) for p in VERDICTS) + " |"
        )
    lines += ["", "## Errors", ""] + ([f"- {x}" for x in report.errors] or ["- none"])
    return "\n".join(lines) + "\n"


__all__ = ["evaluate_factcheck", "load_rows", "render_markdown"]
