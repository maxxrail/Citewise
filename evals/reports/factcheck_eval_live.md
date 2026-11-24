# FactCheck verifier eval (held-out claims)

Golden set: `evals/datasets/golden/factcheck_claims.jsonl`: 45 claims, 0 hand-reviewed. Mode: **live**. Generated 2026-11-24 by `citewise eval factcheck`.

> Live verifier: claude-haiku-5-5.

- **Accuracy:** 95.6%
- **False-contradiction rate** (not-wrong claims called contradicted): 0.0%
- **Evidence hit rate** (correct verdicts whose quote holds the settling phrase): 100.0%

| Verdict | Precision | Recall |
|---|---|---|
| supported | 100.0% | 86.7% |
| contradicted | 100.0% | 100.0% |
| not_in_kb | 85.7% | 100.0% |

## Confusion (rows: expected, columns: predicted)

| expected \ predicted | supported | contradicted | not_in_kb |
|---|---|---|---|
| supported | 13 | 0 | 2 |
| contradicted | 0 | 18 | 0 |
| not_in_kb | 0 | 0 | 12 |

## Errors

- fc-stridelab-08: expected supported, got not_in_kb (paraphrase_true): If you're unsure about the fit, you can run in a pair for roughly two months and still send them back at no cost if you live in the US.
- fc-linear-10: expected supported, got not_in_kb (comparison_true): On yearly billing, Linear Business costs $6 more per user per month than Basic.
