"""Score the mention extractor against the hand-labeled golden set.

For each golden answer we run Stage A, Stage B and the merge, then compare each stage's
entities with the gold labels, matched per answer on the normalized canonical key:

- entity detection: micro precision / recall / F1;
- on matched entities: rank accuracy (exact, null included), entity-type accuracy, and for
  stages that produce them, sentiment MAE, sentiment-label accuracy and recommendation
  accuracy (Stage A produces none of these);
- errors by type (rubric `error_types`) and F1 by tag, so a weak spot ("ambiguous_word",
  "typo") is visible instead of averaged away.

In mock mode Stage B is the simulated extractor (providers/mock_extract.py), so these
numbers measure the pipeline and the simulator, not a production model.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from citewise.providers.base import ChatProvider
from citewise.providers.mock import fixture_key
from citewise.services.extract.mentions import (
    PURPOSE,
    merge,
    stage_a,
    stage_b,
    stage_b_messages,
)
from citewise.services.extract.resolve import Resolver, normalize

STAGES = ("A", "B", "merged")


@dataclass
class Pred:
    key: str
    canonical: str
    entity_type: str
    rank: int | None
    sentiment: float | None = None
    sentiment_label: str | None = None
    is_recommended: bool | None = None


@dataclass
class StageScore:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    rank_ok: int = 0
    type_ok: int = 0
    label_ok: int = 0
    rec_ok: int = 0
    sentiment_abs_err: list[float] = field(default_factory=list)
    errors: Counter[str] = field(default_factory=Counter)
    examples: list[str] = field(default_factory=list)
    by_tag: dict[str, list[int]] = field(default_factory=lambda: defaultdict(lambda: [0, 0, 0]))

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def rate(self, ok: int) -> float | None:
        return ok / self.tp if self.tp else None

    @property
    def has_attrs(self) -> bool:  # the stage produced sentiment/recommendation at all
        return bool(self.sentiment_abs_err)

    @property
    def sentiment_mae(self) -> float | None:
        e = self.sentiment_abs_err
        return sum(e) / len(e) if e else None

    def summary(self) -> dict[str, Any]:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "rank_accuracy": _r(self.rate(self.rank_ok)),
            "type_accuracy": _r(self.rate(self.type_ok)),
            "sentiment_mae": _r(self.sentiment_mae),
            "sentiment_label_accuracy": _r(self.rate(self.label_ok)) if self.has_attrs else None,
            "recommended_accuracy": _r(self.rate(self.rec_ok)) if self.has_attrs else None,
            "errors": dict(self.errors.most_common()),
        }


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 4)


@dataclass
class EvalReport:
    rows: int
    reviewed: int
    mode: str
    stages: dict[str, StageScore]
    # Rows written by hand (engine == "handwritten"): realistic prose, not the answer
    # simulator's templates, so the better guide to real-world quality.
    handwritten: dict[str, StageScore] = field(default_factory=dict)
    handwritten_rows: int = 0
    stage_b_source: str = "live model"

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "reviewed": self.reviewed,
            "mode": self.mode,
            "stages": {k: v.summary() for k, v in self.stages.items()},
            "handwritten_rows": self.handwritten_rows,
            "stage_b_source": self.stage_b_source,
            "handwritten": {k: v.summary() for k, v in self.handwritten.items()},
        }


def load_golden(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    for r in rows:
        missing = {"id", "brand", "answer", "mentions"} - set(r)
        if missing:
            raise ValueError(f"golden row {r.get('id')} is missing {sorted(missing)}")
    return rows


def load_brand_configs(brands_dir: Path) -> dict[str, dict[str, Any]]:
    configs = {}
    for f in sorted(brands_dir.glob("*.yaml")):
        data = yaml.safe_load(f.read_text(encoding="utf-8"))
        configs[data["name"]] = data
    return configs


def _score(s: StageScore, row: dict[str, Any], preds: list[Pred], *, has_attrs: bool) -> None:
    gold = {normalize(g["canonical"]): g for g in row["mentions"]}
    pred = {p.key: p for p in preds}
    tp_keys, fp_keys, fn_keys = (
        gold.keys() & pred.keys(),
        pred.keys() - gold.keys(),
        gold.keys() - pred.keys(),
    )
    s.tp, s.fp, s.fn = s.tp + len(tp_keys), s.fp + len(fp_keys), s.fn + len(fn_keys)
    for tag in row.get("tags", []):
        counts = s.by_tag[tag]
        counts[0] += len(tp_keys)
        counts[1] += len(fp_keys)
        counts[2] += len(fn_keys)
    for k in sorted(fp_keys):
        s.errors["false_positive"] += 1
        s.examples.append(f"{row['id']}: false positive {pred[k].canonical!r}")
    for k in sorted(fn_keys):
        s.errors["false_negative"] += 1
        s.examples.append(f"{row['id']}: missed {gold[k]['canonical']!r}")
    for k in sorted(tp_keys):
        g, p = gold[k], pred[k]
        if p.rank == g["rank"]:
            s.rank_ok += 1
        else:
            s.errors["rank"] += 1
            s.examples.append(f"{row['id']}: {g['canonical']} rank {p.rank} != {g['rank']}")
        if p.entity_type == g["entity_type"]:
            s.type_ok += 1
        else:
            s.errors["wrong_type"] += 1
        if not has_attrs:
            continue
        s.sentiment_abs_err.append(abs(float(p.sentiment or 0.0) - float(g["sentiment"])))
        if p.sentiment_label == g["sentiment_label"]:
            s.label_ok += 1
        else:
            s.errors["sentiment_label"] += 1
        if p.is_recommended == g["is_recommended"]:
            s.rec_ok += 1
        else:
            s.errors["recommended"] += 1


def evaluate(
    rows: list[dict[str, Any]],
    brands: dict[str, dict[str, Any]],
    chat: ChatProvider,
    *,
    model: str,
    mode: str,
    fixtures_dir: Path | None = None,
) -> EvalReport:
    stages = {name: StageScore() for name in STAGES}
    recorded = 0
    hand = {name: StageScore() for name in STAGES}
    for row in rows:
        targets = [stages, hand] if row.get("engine") == "handwritten" else [stages]
        cfg = brands[row["brand"]]
        resolver = Resolver(cfg["name"], cfg.get("aliases", []), cfg.get("competitors", []))
        text = row["answer"]
        if mode == "mock" and fixtures_dir is not None:
            msgs = stage_b_messages(brand=cfg["name"], aliases=cfg.get("aliases", []),
                                    competitors=cfg.get("competitors", []),
                                    category=cfg["category"], text=text)  # fmt: skip
            keys = (fixture_key(PURPOSE, msgs, "openai"), fixture_key(PURPOSE, msgs))
            recorded += any((fixtures_dir / "mock_chat" / f"{k}.json").exists() for k in keys)
        a = stage_a(text, resolver)
        b = stage_b(chat, model=model, brand=cfg["name"], aliases=cfg.get("aliases", []),
                    competitors=cfg.get("competitors", []), category=cfg["category"],
                    text=text)  # fmt: skip
        merged, _ = merge(text, resolver, a, b)
        a_preds = [Pred(m.key, m.canonical, m.entity_type, m.rank) for m in a]
        for t in targets:
            _score(t["A"], row, a_preds, has_attrs=False)
        b_preds: dict[str, Pred] = {}
        for x in b or []:
            r = resolver.resolve(x.canonical or x.entity)
            pred = Pred(r.key, r.canonical, r.entity_type, x.rank, x.sentiment,
                        x.sentiment_label, x.is_recommended)  # fmt: skip
            b_preds.setdefault(r.key, pred)
        for t in targets:
            _score(t["B"], row, list(b_preds.values()), has_attrs=True)
        m_preds = [
            Pred(m.key, m.canonical, m.entity_type, m.rank, m.sentiment, m.sentiment_label,
                 m.is_recommended)
            for m in merged
        ]  # fmt: skip
        for t in targets:
            _score(t["merged"], row, m_preds, has_attrs=True)
    return EvalReport(
        rows=len(rows),
        reviewed=sum(1 for r in rows if r.get("reviewed")),
        mode=mode,
        stages=stages,
        handwritten=hand,
        handwritten_rows=sum(1 for r in rows if r.get("engine") == "handwritten"),
        stage_b_source=(
            "live model"
            if mode != "mock"
            else f"{recorded}/{len(rows)} replayed from recorded real responses, "
            f"{len(rows) - recorded} simulated"
        ),
    )


def _fmt(v: float | None, pct: bool = False) -> str:
    if v is None:
        return "-"
    return f"{v:.1%}" if pct else f"{v:.3f}"


def render_markdown(report: EvalReport, golden_path: str, date: str) -> str:
    st = report.stages
    lines = [
        "# Mention extractor eval",
        "",
        f"Golden set: `{golden_path}`: {report.rows} answers, {report.reviewed} hand-reviewed. "
        f"Mode: **{report.mode}**. Generated {date} by `citewise eval extract`.",
        "",
    ]
    if report.mode == "mock":
        lines += [
            "> **Mock mode.** Unless real responses have been recorded "
            "(`scripts/record_extract_fixtures.py`, needs a key), Stage B is the simulated "
            "extractor (`providers/mock_extract.py`), so the B and merged columns are a "
            "*smoke test of the pipeline*, not a model score. 48 of the rows are also "
            "simulator output: the hand-written table below is the fairer number, and "
            "Stage A (no LLM) is the honest offline baseline. Run `CITEWISE_MODE=live "
            "citewise eval extract` for real numbers.",
            "",
            f"Stage B responses: {report.stage_b_source}.",
            "",
        ]
    lines += [
        "| Metric | Stage A | Stage B | Merged |",
        "|---|---|---|---|",
    ]
    for label, getter, pct in [
        ("Precision", lambda s: s.precision, True),
        ("Recall", lambda s: s.recall, True),
        ("**F1**", lambda s: s.f1, True),
        ("Rank accuracy", lambda s: s.rate(s.rank_ok), True),
        ("Type accuracy", lambda s: s.rate(s.type_ok), True),
        ("Sentiment MAE", lambda s: s.sentiment_mae, False),
        (
            "Sentiment label accuracy",
            lambda s: s.rate(s.label_ok) if s.sentiment_abs_err else None,
            True,
        ),
        ("Recommended accuracy", lambda s: s.rate(s.rec_ok) if s.sentiment_abs_err else None, True),
    ]:
        lines.append(f"| {label} | " + " | ".join(_fmt(getter(st[k]), pct) for k in STAGES) + " |")
    lines += [
        "",
        "## Hand-written rows only",
        "",
        f"{report.handwritten_rows} rows of realistic prose (not the simulator's templates): "
        "the better guide to real-world quality.",
        "",
        "| Metric | Stage A | Stage B | Merged |",
        "|---|---|---|---|",
        "| F1 | "
        + " | ".join(_fmt(report.handwritten[k].f1, True) for k in STAGES if report.handwritten)
        + " |",
        "| Rank accuracy | "
        + " | ".join(
            _fmt(report.handwritten[k].rate(report.handwritten[k].rank_ok), True)
            for k in STAGES
            if report.handwritten
        )
        + " |",
        "",
        "## Errors by type",
        "",
        "| Error | Stage A | Stage B | Merged |",
        "|---|---|---|---|",
    ]
    kinds = sorted({k for s in st.values() for k in s.errors})
    for kind in kinds:
        lines.append(
            f"| {kind} | " + " | ".join(str(st[k].errors.get(kind, 0)) for k in STAGES) + " |"
        )
    lines += [
        "",
        "## F1 by tag (merged)",
        "",
        "| Tag | TP | FP | FN | F1 |",
        "|---|---|---|---|---|",
    ]
    for tag, (tp, fp, fn) in sorted(st["merged"].by_tag.items()):
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
        lines.append(f"| {tag} | {tp} | {fp} | {fn} | {f1:.1%} |")
    lines += ["", "## Merged errors (all)", ""]
    lines += [f"- {e}" for e in st["merged"].examples] or ["- none"]
    return "\n".join(lines) + "\n"
