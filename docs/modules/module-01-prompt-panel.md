# Module 1 — Prompt panel and topic discovery

## Why Profound cares
Profound's "Prompt Volumes" shows what people actually ask AI, grouped by intent. Their AI/ML Engineer posting asks for **classification, clustering and topic discovery** on large text datasets. This module is that, at small scale.

## Goal
Given a brand (category, competitors, audience), build a realistic, de-duplicated, topic-clustered **prompt panel**: the set of questions real buyers ask AI assistants.

## Build
1. **Generation** (`src/citewise/services/panel/generate.py`)
   - Use a prompt-template library (`prompts/panel/*.jinja`, versioned: `v1`, `v2`…) to generate candidates across:
     - intents: `informational`, `commercial` ("best X for Y"), `comparison` ("A vs B"), `navigational/brand` ("is <brand> good"), `transactional` ("cheapest", "discount")
     - personas from the brand YAML (e.g. "startup founder", "marathon beginner")
     - funnel stage: awareness, consideration, decision
   - Target ~300 raw candidates per brand; write them in natural, messy user language (typos and short queries allowed in some share).
   - Also support **importing** prompts from CSV (`citewise panel import file.csv`) so real data can be dropped in.
2. **Dedup**: embed all prompts; drop near-duplicates (cosine > threshold, configurable, default 0.92). Keep the shortest/most natural one. Log how many were dropped.
3. **Clustering** (`cluster.py`): KMeans with k chosen by silhouette over a range, or HDBSCAN if installed (behind a flag). Store `topics` with sizes.
4. **Topic labeling**: for each cluster, send the 8 most central prompts to an LLM → `{label, description}` (structured output). Labels must be short (≤5 words).
5. **Intent classifier**: two implementations behind one interface:
   - `LLMIntentClassifier` (zero-shot, structured output)
   - `EmbeddingIntentClassifier` (logistic regression on embeddings, trained on the LLM-labeled set, used as the cheap path)
   - Report agreement between them (accuracy, macro-F1, confusion matrix) with `citewise panel eval-intent`.
6. **Panel selection**: `citewise panel build <brand> --size 50` picks a stratified sample across topics × intents for scanning (the full set stays in the DB).
7. **API**: `GET /brands/{id}/prompts`, `GET /brands/{id}/topics`.
8. **Fixtures**: mock fixtures so the whole pipeline runs offline and gives the same clusters every time.

## Acceptance criteria
- [ ] `uv run citewise panel generate <brand>` → ≥200 prompts stored after dedup (mock mode).
- [ ] `uv run citewise panel cluster <brand>` → topics with labels and sizes; every prompt has a `topic_id` and `intent`.
- [ ] `uv run citewise panel eval-intent <brand>` prints accuracy + macro-F1 + confusion matrix.
- [ ] `uv run citewise panel build <brand> --size 50` returns 50 prompts covering ≥80% of topics.
- [ ] Unit tests for dedup (threshold behavior), k-selection, stratified sampling.
- [ ] Template version is stored on each generated prompt.
- [ ] Notebook or script `scripts/panel_report.py` writes `reports/panel_<brand>.md` with topic table + 2D UMAP/PCA plot PNG.

## Out of scope
Real search-volume estimates (note in DECISIONS.md how you'd do it with real data).
