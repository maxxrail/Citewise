# Progress

| # | Module | Status | Notes |
|---|---|---|---|
| 0 | Foundations | ✅ done | 35 tests; all acceptance criteria verified 2026-11-23 |
| 1 | Prompt panel and topic discovery | 🔁 reviewed | 121 tests; all acceptance criteria verified on Postgres + SQLite 2026-11-23 |
| 2 | Multi-engine answer collection | ⬜ not started | |
| 3 | Extraction and visibility metrics | ⬜ not started | |
| 4 | Brand knowledge base and FactCheck | ⬜ not started | |
| 5 | Content agent | ⬜ not started | |
| 6 | Evals, experiments and observability | ⬜ not started | |
| 7 | AI crawler analytics | ⬜ not started | |
| 8 | Dashboard and ship | ⬜ not started | |

Statuses: ⬜ not started · 🟨 in progress · ✅ done (all acceptance criteria pass) · 🔁 reviewed

## Log
<!-- Claude Code: append a dated entry after each session: what was built, decisions, known gaps, what Maxx must do. -->

### 2026-11-23 - Module 0 (Foundations)
**Built:** uv/hatchling project, `citewise` CLI (`seed`, `config`), pydantic-settings config, SQLAlchemy models `brands` and `llm_calls`, Alembic migration `0001`, provider layer (protocols, mock chat and embedder, live OpenAI via the Responses API, stubs for the other three engines), `tracking.py` (one `llm_calls` row per call, including failures), `pricing.py` (checked 2026-11-23), FastAPI `/health` and `/brands`, two example brands, Dockerfile and compose, CI (ruff, mypy, pytest, docker build), pre-commit pins bumped and hook installed.

**Decisions:** see the Module 0 section of docs/DECISIONS.md (Responses API, mock cost estimation, engine-aware fixture keys, snapshot pricing, strict schema check, configurable ports).

**Code review:** fixed 1 High (live snapshot model IDs would have been priced at $0), 4 Medium (mock engines identical, fixtures missing from the image, no client timeout, strict-schema mismatch) and 2 Low (409 on a duplicate-insert race, truncated responses marked not ok).

**Known gaps / for later modules:**
- Brand names are unique only by exact case (alias/competitor lists are now deduped).
- On SQLite, `created_at` comes back without a timezone (Postgres keeps it). Add a UTC TypeDecorator before time-window metrics.
- `llm_calls` has no `run_id` yet. Module 2 must add it for the per-run budget guard and idempotency.
- Perplexity moved Sonar to its Agent API (July 2026). Verify the model ID and endpoint before the Module 2 adapter.
- Haiku 5.5 tiered pricing (>100K-token prompts) isn't modelled. Only the base tier is recorded.
- The live OpenAI path is untested against the real API (no key in this session).

**Maxx must do:** nothing is required for Module 0. Optional: put `OPENAI_API_KEY` in `.env` and run one live smoke call (README "Live mode").

### 2026-11-23 - Module 0 review (`/review-module 00`)
**Re-verified:** `uv run pytest` (53 passed, mock, SQLite), `citewise --help` lists `seed`, `docker compose up -d db` -> `alembic upgrade head` -> `citewise seed` (idempotent) -> `GET /brands` returns both brands, `GET /health` -> `{"status":"ok","db":"ok"}`, ruff check/format, mypy, `pre-commit run --all-files`, `docker build .`. Anthropic prices checked against the claude-api reference.

**Fixed:** 1 High (mock output for Pydantic `$ref` schemas was invalid) and 2 Medium (unpriced models recorded $0 in live mode, and prefix pricing could misprice sibling models; brand fields unbounded and multi-line, which is a prompt-injection and cost-amplification risk for Module 1). Tests are in `tests/test_review_fixes.py`.

**Still open (Low):** `llm_calls.run_id` (Module 2), case-insensitive brand uniqueness, `GET /brands` pagination, no Postgres `connect_timeout` on `/health`, `.env` is resolved relative to the working directory.

### 2026-11-23 - Module 1 (Prompt panel and topic discovery)
**Built:**
- Tables `prompts` and `topics`, plus `brands.personas` (migration `0002`).
- Versioned Jinja prompt library (`prompts/panel/*.v1.jinja`), with the version stored per prompt.
- `complete_structured` (Pydantic validation, one retry with error feedback, never crashes the run).
- Generation per intent x persona (15 cells x 20 prompts) and CSV import.
- Greedy cosine dedup, keeping the shortest phrasing.
- PCA(20) + KMeans with silhouette-selected k in 6–14, or sklearn HDBSCAN.
- Topic labels from the 8 most central prompts.
- LLM and embedding intent classifiers with a cross-validated agreement report and golden-set scoring.
- Sainte-Laguë stratified panel selection.
- CLI `citewise panel generate|import|cluster|eval-intent|build`, API `GET /brands/{id}/prompts|topics`, `scripts/panel_report.py`, and committed reports for both brands.
- Spend cap (`--max-cost`).

Mock mode got a calibrated semantic mock embedder and hand-written panel sources, which a recorder turns into fixtures (`scripts/record_panel_fixtures.py`). Eval data drafted by `eval-designer`: golden intent set, intent labeling rubric, 11 Module 1 red-team cases. Maxx reviewed the golden set (all 94 rows `reviewed: true`), corrected one row and added 14 hard cases (typos, negation, the brand name used as an ordinary word).

**Results (mock):**
| Brand | Stored after dedup | Topics | Embedding vs LLM, cross-validated (acc / macro-F1) | Embedding vs golden (acc / macro-F1) | Panel of 50 covers | Cost per brand |
|---|---|---|---|---|---|---|
| Stridelab | 272 of 300 | 11 | 0.717 / 0.713 | 0.761 / 0.757 (46 rows) | 11/11 topics | ~$0.075 |
| Taskflow | 276 of 300 | 13 | 0.826 / 0.821 | 0.729 / 0.724 (48 rows) | 13/13 topics | ~$0.074 |

Cost is estimated as if the calls were live.

**Decisions:** see the Module 1 section of docs/DECISIONS.md. Main ones:
- semantic mock embedder
- recorder for the fixtures
- PCA before silhouette, with a product-prior k range
- LLM calls before DB writes
- teacher/student intent labels
- coverage-first Sainte-Laguë selection
- skip cells already generated
- provider errors count as failed attempts
- spend cap

**Code review:** fixed:
- 2 High: template-marker prompt injection; a provider error aborted the run and lost paid work.
- 5 Medium:
  - live `generate` wasn't idempotent;
  - no budget guard or import cap, and silhouette was O(n²);
  - fallback labels were frozen and used for training;
  - the fallback label could raise;
  - Linux determinism was unverified. It is now verified in a container: 121/121 pass.
- Lows: build refuses unclustered prompts, re-cluster clears the stale panel, clean error for an unknown template version, `server_default` on personas.

**Known gaps:**
- LLM-vs-golden intent accuracy is only measured in live mode.
- `llm_calls` has no brand or run id, so per-brand cost in the report covers all brands. This comes with Module 2.
- Concurrent `generate` for the same brand can hit the unique constraint and roll back. Single-user CLI for now; add ON CONFLICT DO NOTHING or a per-brand lock if jobs go parallel.
- Silhouette for k=11 vs k=12 is close, so a scikit-learn upgrade could change mock clusters. The staleness test catches it, and the fix is re-recording.
- Red-team rows m1-rt-007 (HTML escaping in the dashboard) and m1-rt-010 (formula escaping on CSV export) are deferred to Module 8, when those surfaces exist.
- On the golden set, the embedding classifier misses negation ("not hoka or brooks..." -> comparison) and the brand name used as an ordinary word ("taskflow diagram" -> navigational). It only sees words. The LLM prompt now has tie-breaks for both; measure them in the live run.
- The live generation prompt hasn't been tried against the real API.

**Maxx must do:**
1. ~~Review the golden intent set~~ (done: 94 rows reviewed, `taskflow-029` reworded to remove a training near-duplicate, `stridelab-045` relabeled navigational under a new contrast/exclusion tie-break).
2. ~~Add the panel settings to `.env.example`~~ (done).
3. ~~Review the red-team rows~~ (done: all 11 reviewed; 007, 008, 010 and 011 expectations tightened).
4. Optional live smoke test (corrected in the review below): `uv run citewise panel reset Stridelab --yes` (clears the mock-mode panel; otherwise `generate` skips every cell and never calls the live model), then `CITEWISE_MODE=live uv run citewise panel generate Stridelab --max-cost 0.50`, `panel cluster Stridelab`, and `panel eval-intent Stridelab --golden evals/datasets/golden/intent.jsonl` for the LLM-vs-golden number.

### 2026-11-23 - Module 1 review (`/review-module 01`)
**Re-verified on a fresh Postgres DB:** `seed → panel generate → cluster → eval-intent --golden → build --size 50` for both brands. Same numbers as above: 272/276 prompts, 11/13 topics, 0 prompts without a topic, intent or template version, 50-prompt panels covering 100% of topics. `panel_report.py` works against Postgres. Mechanical checks were clean: every LLM and embedding call goes through the tracked providers, and no model IDs are hardcoded outside config and pricing.

**Fixed (3 Medium):**
- **Embeddings had no provenance.** Mock and live embedders share the model name `text-embedding-3-small` but not a vector space, so a live run on a DB with mock prompts would have deduped, clustered and scored the golden set across two unrelated spaces. A `MODEL_EMBED` change with a different dimension would crash. Fix: `prompts.embedding_model` (migration `0003`) and `EmbeddingProvider.ident` (`"mock:…"` / `"openai:…"`). `store`, `cluster` (CLI) and `eval-intent` re-embed stale rows first. `cluster_brand` without an embedder refuses mixed spaces. Pre-`0003` rows count as stale.
- **The documented live smoke test was a no-op:** `generate` skipped every cell the mock run had filled. Added `citewise panel reset <brand> [--yes]` (deletes that brand's prompts and topics; keeps the brand and the `llm_calls` history) and corrected the instructions.
- **Windows: piped CLI output crashed on non-cp1252 text** (Japanese, emoji) with `UnicodeEncodeError`, after the work was already committed. Reproduced, then fixed by reconfiguring stdout/stderr to UTF-8 (`errors="replace"`) in the CLI callback.

**Fixed (Low):** `eval-intent --golden` in live mode now goes through the spend cap (`--max-cost`). Its cross-validation title says "teacher labels (LLM or imported)" instead of "LLM labels".

**Not fixed (Low, documented):** the mock embedder tokenizes `[a-z0-9]` only, so mock-mode near-duplicate detection doesn't work for non-Latin text (exact duplicates still work). Live embeddings don't have this limit.

Tests: `tests/test_panel_review.py` (+6; 127 total, all mock).
