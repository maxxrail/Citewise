# Module 3 — Extraction and visibility metrics

## Why Profound cares
Raw answers are useless until they become signals: *was the brand mentioned, where in the ranking, how positively, which sources drove it*. The posting asks for **classification, ranking, entity work on text, and SQL**, plus **turning text datasets into product features**. This module is the core of the portfolio piece.

## Goal
Turn every answer into structured mentions, then compute visibility metrics per brand, engine, topic and intent, and **prove the extractor is accurate**.

## Build
1. **Mention extraction** (`src/citewise/services/extract/mentions.py`)
   - Stage A (cheap, deterministic): alias matching with word boundaries + fuzzy matching (rapidfuzz) against brand + competitor aliases.
   - Stage B (LLM, structured output): for each answer return `[{entity, canonical, rank (order in a list, or null), sentiment (-1..1), sentiment_label, is_recommended (bool), snippet}]`, including **unknown entities** (brands not in our competitor list: these are "emerging competitors").
   - Entity resolution: map surface forms to canonical names (aliases, case, punctuation, "Asana Inc." → "Asana"). Unknown entities get clustered by normalized name.
   - Merge A and B; when they disagree, record it (useful for evals).
2. **Metrics** (`src/citewise/services/metrics/`), implemented as **SQL** (views or query functions), not pandas-only:
   - Mention rate (% of answers mentioning brand)
   - Share of voice (brand mentions / all tracked-entity mentions)
   - Average rank when ranked; top-3 rate
   - Recommendation rate
   - Net sentiment
   - Citation share: top domains cited in answers where the brand appears vs. doesn't; % of citations pointing at the brand's own domain
   - A composite **Visibility Score (0–100)** with documented weights in DECISIONS.md
   - Every metric sliced by engine, topic, intent, run (for trends)
   - Bootstrap 95% confidence intervals on mention rate and SoV (small samples are noisy, so say so)
3. **Golden set + extractor eval** (`evals/datasets/golden/mentions.jsonl`)
   - ~60 hand-checked answers (generate candidates, then Claude Code drafts labels and **Maxx reviews/edits a sample**; mark `reviewed: true/false`).
   - `citewise eval extract`: precision/recall/F1 for entity detection, rank accuracy, sentiment MAE, Stage A vs B vs merged. Save the report to `reports/`.
4. **API**: `GET /brands/{id}/metrics?run_id=&engine=&topic=` and `GET /brands/{id}/competitors` (incl. emerging).

## Acceptance criteria
- [ ] `citewise extract run <run_id>` processes all answers idempotently.
- [ ] `citewise metrics show <brand>` prints a table per engine: mention rate (with CI), SoV, avg rank, sentiment, visibility score.
- [ ] `citewise eval extract` prints P/R/F1 ≥ 0.85 F1 on the golden set in mock mode, and the report breaks down errors by type.
- [ ] Metrics SQL has unit tests on a tiny hand-built fixture where the expected numbers are computed by hand.
- [ ] Entity resolution tests (aliases, suffixes like Inc./Ltd., casing, possessives).

## Out of scope
Fact accuracy (Module 4).
