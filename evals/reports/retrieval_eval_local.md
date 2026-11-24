# KB retrieval eval

Golden set: `evals/datasets/golden/kb_qa.jsonl`: 36 queries over 3 brand KBs, 0 hand-reviewed. Mode: **live**, embedder `local:BAAI/bge-small-en-v1.5`. Generated 2026-11-24 by `citewise eval retrieval`.

| Method | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| bm25 | 69.4% | 80.6% | 0.745 |
| dense | 91.7% | 100.0% | 0.958 |
| hybrid | 83.3% | 100.0% | 0.903 |

## By query style (MRR)

| Style | n | bm25 | dense | hybrid |
|---|---|---|---|---|
| claim | 12 | 1.000 | 0.917 | 1.000 |
| lexical | 12 | 0.903 | 1.000 | 0.958 |
| paraphrase | 12 | 0.333 | 0.958 | 0.750 |

## Misses (gold chunk not in the top 5)

- **bm25**: kbqa-stridelab-05, kbqa-stridelab-06, kbqa-stridelab-07, kbqa-taskflow-05, kbqa-taskflow-06, kbqa-taskflow-07, kbqa-linear-06
- **dense**: none
- **hybrid**: none
