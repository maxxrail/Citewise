# Mention extractor eval

Golden set: `evals/datasets/golden/mentions.jsonl`: 65 answers, 65 hand-reviewed. Mode: **mock**. Generated 2026-11-24 by `citewise eval extract`.

> **Mock mode.** Unless real responses have been recorded (`scripts/record_extract_fixtures.py`, needs a key), Stage B is the simulated extractor (`providers/mock_extract.py`), so the B and merged columns are a *smoke test of the pipeline*, not a model score. 48 of the rows are also simulator output: the hand-written table below is the fairer number, and Stage A (no LLM) is the honest offline baseline. Run `CITEWISE_MODE=live citewise eval extract` for real numbers.

Stage B responses: 65/65 replayed from recorded real responses, 0 simulated.

| Metric | Stage A | Stage B | Merged |
|---|---|---|---|
| Precision | 100.0% | 100.0% | 100.0% |
| Recall | 76.1% | 100.0% | 100.0% |
| **F1** | 86.5% | 100.0% | 100.0% |
| Rank accuracy | 93.3% | 98.3% | 98.9% |
| Type accuracy | 100.0% | 100.0% | 100.0% |
| Sentiment MAE | - | 0.148 | 0.148 |
| Sentiment label accuracy | - | 92.0% | 92.0% |
| Recommended accuracy | - | 85.2% | 85.2% |

## Hand-written rows only

17 rows of realistic prose (not the simulator's templates): the better guide to real-world quality.

| Metric | Stage A | Stage B | Merged |
|---|---|---|---|
| F1 | 85.1% | 100.0% | 100.0% |
| Rank accuracy | 94.6% | 94.0% | 100.0% |

## Errors by type

| Error | Stage A | Stage B | Merged |
|---|---|---|---|
| false_negative | 42 | 0 | 0 |
| rank | 9 | 3 | 2 |
| recommended | 0 | 26 | 26 |
| sentiment_label | 0 | 14 | 14 |

## F1 by tag (merged)

| Tag | TP | FP | FN | F1 |
|---|---|---|---|---|
| ambiguous_word | 42 | 0 | 0 | 100.0% |
| brand_absent | 84 | 0 | 0 | 100.0% |
| casing | 21 | 0 | 0 | 100.0% |
| comparison | 31 | 0 | 0 | 100.0% |
| deals | 29 | 0 | 0 | 100.0% |
| injection | 7 | 0 | 0 | 100.0% |
| legal_suffix | 13 | 0 | 0 | 100.0% |
| list | 64 | 0 | 0 | 100.0% |
| negative | 14 | 0 | 0 | 100.0% |
| off_topic | 16 | 0 | 0 | 100.0% |
| possessive | 22 | 0 | 0 | 100.0% |
| product_name | 144 | 0 | 0 | 100.0% |
| prose_ranking | 7 | 0 | 0 | 100.0% |
| publisher_not_entity | 15 | 0 | 0 | 100.0% |
| single | 28 | 0 | 0 | 100.0% |
| typo | 16 | 0 | 0 | 100.0% |
| unknown_entity | 76 | 0 | 0 | 100.0% |

## Merged errors (all)

- taskflow-026: Monday.com rank None != 2
- taskflow-028: Monday.com rank None != 2
