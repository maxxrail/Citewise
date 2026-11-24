# KB retrieval eval

Golden set: `evals/datasets/golden/kb_qa.jsonl`: 36 queries over 3 brand KBs, 0 hand-reviewed. Mode: **mock**, embedder `mock:BAAI/bge-small-en-v1.5`. Generated 2026-11-24 by `citewise eval retrieval`.

| Method | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| bm25 | 69.4% | 80.6% | 0.745 |
| dense | 75.0% | 80.6% | 0.802 |
| hybrid | 72.2% | 83.3% | 0.789 |

## By query style (MRR)

| Style | n | bm25 | dense | hybrid |
|---|---|---|---|---|
| claim | 12 | 1.000 | 0.924 | 0.931 |
| lexical | 12 | 0.903 | 1.000 | 0.958 |
| paraphrase | 12 | 0.333 | 0.482 | 0.480 |

## Misses (gold chunk not in the top 5)

- **bm25**: kbqa-stridelab-05, kbqa-stridelab-06, kbqa-stridelab-07, kbqa-taskflow-05, kbqa-taskflow-06, kbqa-taskflow-07, kbqa-linear-06
- **dense**: kbqa-stridelab-05, kbqa-stridelab-06, kbqa-taskflow-05, kbqa-taskflow-06, kbqa-taskflow-08, kbqa-linear-06, kbqa-linear-12
- **hybrid**: kbqa-stridelab-05, kbqa-stridelab-06, kbqa-taskflow-05, kbqa-taskflow-06, kbqa-linear-06, kbqa-linear-12
