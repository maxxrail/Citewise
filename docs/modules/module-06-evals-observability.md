# Module 6 — Evals, experiments and observability

## Why Profound cares
The posting asks directly for **eval frameworks (factuality, coverage, tone, safety, originality) using rubric-based LLM evals, human review, and red-teaming**, plus **experiments on quality, cost and latency**. This is the module that separates this project from a wrapper demo.

## Goal
One eval harness for every LLM component, a way to A/B prompt versions and models, and a CI gate that blocks quality regressions.

## Build
1. **Harness** (`src/citewise/evals/`)
   - `Dataset` (JSONL in `evals/datasets/golden/` and `evals/datasets/`), `Task` (the component under test), `Scorer` (deterministic or LLM-judge), `EvalRun` stored in `eval_runs` / `eval_results`.
   - Built-in suites: `extract` (M3), `intent` (M1), `retrieval` (M4), `factcheck` (M4), `content` (M5).
2. **Rubric LLM-judge**
   - Rubrics in YAML (`evals/rubrics/content.yaml`): criteria, 1–5 anchors with examples, pass threshold.
   - Judge best practice: score one criterion per call or with structured per-criterion output; ask for evidence before the score; randomize order in pairwise comparisons; use a different model family from the generator when possible.
   - **Judge calibration**: `evals/datasets/human_labels.jsonl` (Maxx scores ~20 items with a tiny CLI `citewise eval label`). Report judge–human agreement (Spearman, Cohen's kappa on pass/fail).
3. **Red-team set**: `evals/datasets/redteam.jsonl`: prompts meant to make the content agent invent stats, defame competitors, write fake testimonials, or follow injected instructions found inside a cited web page. Scorer = pass if the agent refuses or stays grounded.
4. **Experiments**: `citewise exp run --suite content --variants prompts/v1,prompts/v2 --models a,b`. Report quality, cost/item, p50/p95 latency, and a paired bootstrap significance on the main metric. Markdown report to `evals/reports/experiments/`.
5. **Observability**
   - Structured JSON logs with `run_id`/`trace_id`.
   - Optional OpenTelemetry export (off by default) or Langfuse if keys are set.
   - `GET /ops/llm-usage`: cost and latency by purpose/model/day (from `llm_calls`).
6. **CI gate**: GitHub Action runs `citewise eval run --suite all --mode mock --fail-under baseline.json`. Fail the build if any suite drops > 2 points from baseline.

## Acceptance criteria
- [ ] `citewise eval run --suite all` prints a summary table for all suites in mock mode.
- [ ] Judge calibration report exists (with mock human labels if Maxx hasn't labeled yet; mark clearly).
- [ ] Red-team suite has ≥ 15 cases; pass rate reported.
- [ ] Experiment report includes quality, cost, latency and a significance note.
- [ ] CI fails when a deliberately broken prompt version is used (test with a fixture).

## Out of scope
Online A/B with real traffic.
