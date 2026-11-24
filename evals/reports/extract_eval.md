# Mention extractor eval

Golden set: `evals/datasets/golden/mentions.jsonl`: 65 answers, 0 hand-reviewed. Mode: **mock**. Generated 2026-11-24 by `citewise eval extract`.

> Mock mode: Stage B is the simulated extractor (`providers/mock_extract.py`), so the B and merged columns measure the pipeline plus that simulator, not a production model. Run `CITEWISE_MODE=live citewise eval extract` for real numbers.

| Metric | Stage A | Stage B | Merged |
|---|---|---|---|
| Precision | 98.5% | 92.3% | 94.5% |
| Recall | 76.1% | 88.1% | 88.6% |
| **F1** | 85.9% | 90.1% | 91.5% |
| Rank accuracy | 93.3% | 92.3% | 94.9% |
| Type accuracy | 100.0% | 100.0% | 100.0% |
| Sentiment MAE | - | 0.223 | 0.226 |
| Sentiment label accuracy | - | 74.8% | 74.4% |
| Recommended accuracy | - | 84.5% | 84.6% |

## Errors by type

| Error | Stage A | Stage B | Merged |
|---|---|---|---|
| false_negative | 42 | 21 | 20 |
| false_positive | 2 | 13 | 9 |
| rank | 9 | 12 | 8 |
| recommended | 0 | 24 | 24 |
| sentiment_label | 0 | 39 | 40 |

## F1 by tag (merged)

| Tag | TP | FP | FN | F1 |
|---|---|---|---|---|
| ambiguous_word | 36 | 2 | 6 | 90.0% |
| brand_absent | 69 | 6 | 15 | 86.8% |
| casing | 19 | 2 | 2 | 90.5% |
| comparison | 30 | 2 | 1 | 95.2% |
| deals | 27 | 1 | 2 | 94.7% |
| injection | 7 | 2 | 0 | 87.5% |
| legal_suffix | 12 | 1 | 1 | 92.3% |
| list | 60 | 6 | 4 | 92.3% |
| negative | 13 | 1 | 1 | 92.9% |
| off_topic | 11 | 3 | 5 | 73.3% |
| possessive | 16 | 2 | 6 | 80.0% |
| product_name | 128 | 4 | 16 | 92.8% |
| prose_ranking | 3 | 0 | 4 | 60.0% |
| publisher_not_entity | 13 | 1 | 2 | 89.7% |
| single | 23 | 0 | 5 | 90.2% |
| typo | 14 | 2 | 2 | 87.5% |
| unknown_entity | 59 | 5 | 17 | 84.3% |

## Merged errors (all)

- stridelab-020: missed 'Saucony'
- stridelab-024: missed 'Altra'
- taskflow-026: Monday.com rank None != 2
- taskflow-028: Monday.com rank None != 2
- taskflow-034: false positive 'New'
- taskflow-034: missed 'New Balance'
- taskflow-034: missed 'On'
- taskflow-038: missed 'On'
- taskflow-038: Hoka rank 3 != 4
- taskflow-039: false positive 'New'
- taskflow-039: missed 'New Balance'
- taskflow-042: false positive 'New'
- taskflow-042: missed 'New Balance'
- taskflow-045: missed 'Wrike'
- taskflow-047: missed 'Basecamp'
- taskflow-048: missed 'Wrike'
- hard-001: missed 'On'
- hard-003: false positive 'Inc'
- hard-003: missed 'Atlassian'
- hard-005: missed 'Altra'
- hard-005: missed 'Saucony'
- hard-005: missed 'Topo Athletic'
- hard-006: false positive 'Start'
- hard-006: ClickUp rank 4 != 3
- hard-006: Jira rank 2 != 1
- hard-006: Linear rank 3 != 2
- hard-007: missed 'ASICS'
- hard-007: missed 'Nike'
- hard-008: missed 'Basecamp'
- hard-009: missed 'Saucony'
- hard-011: false positive 'Look'
- hard-011: ASICS rank 2 != 1
- hard-011: Stridelab rank 3 != 2
- hard-012: missed 'Linear'
- hard-014: false positive 'Monday.com'
- inject-002: false positive 'Taskflow'
- inject-003: false positive 'Stridelab'
