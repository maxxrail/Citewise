# Changelog

All notable changes to Citewise are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). One section per module.

## [Unreleased]

## module-01 - Prompt panel and topic discovery (2026-11-23)

### Added
- Tables `prompts` and `topics`, and `brands.personas` (migration `0002`). Prompts are unique per brand by normalised text, so re-running jobs never duplicates rows.
- Versioned prompt library (`src/citewise/services/prompt_library.py`, `prompts/panel/{generate,classify_intent,label_topic}.v1.jinja`). Brand values and prompts are JSON-encoded into templates as data. The template version is stored on every generated prompt.
- `complete_structured()`: Pydantic-validated structured output with one retry that feeds back the validation error, then a recorded failure instead of a crash.
- Panel services in `services/panel/`: generation per intent x persona, CSV import, greedy cosine dedup (default 0.92, shortest phrasing wins, also against stored prompts), PCA + KMeans with silhouette-selected k (or sklearn HDBSCAN via `--algorithm hdbscan`), topic labels (5 words or fewer) from the 8 most central prompts, LLM and embedding (logistic regression) intent classifiers with a cross-validated agreement report, and Sainte-Laguë stratified panel selection that covers every topic first.
- CLI `citewise panel generate | import | cluster | eval-intent | build`.
- API `GET /brands/{id}/prompts` and `GET /brands/{id}/topics`.
- `scripts/panel_report.py` (markdown topic table, topic x intent, PCA map PNG) and committed reports for both example brands in `evals/reports/`.
- `scripts/record_panel_fixtures.py` and hand-written sources in `tests/fixtures/panel_source/`, so the whole pipeline runs offline and gives identical clusters each run.
- Eval data: golden intent set (`evals/datasets/golden/intent.jsonl`, 80 rows), labeling rubric (`evals/rubrics/intent_labels.yaml`), Module 1 red-team cases (`evals/datasets/redteam.jsonl`).
- Settings: `PANEL_PROMPTS_PER_CELL`, `PANEL_DEDUP_THRESHOLD`, `PANEL_K_MIN`/`PANEL_K_MAX`, `PANEL_PCA_DIMS`, `PANEL_CLASSIFY_BATCH`, `PROMPTS_DIR`.
- Dependencies: numpy, scikit-learn, jinja2 (runtime) and matplotlib (dev).

### Fixed (code review, 2026-11-23)
- Prompt injection: templates are split at the `=== user ===` line before rendering, the brand category is JSON-encoded in every template, and brand fields reject `===`.
- A provider error (429, timeout) no longer aborts `generate`/`cluster` and rolls back paid work. It counts as a failed attempt and the fallback runs.
- `panel generate` skips cells already generated with the template version (`--force` regenerates), so live re-runs don't pay again or grow the panel.
- Spend cap per command (`--max-cost`, default `MAX_COST_USD_PER_RUN`), `PANEL_IMPORT_MAX_ROWS`, and a sampled silhouette for large panels.
- Fallback/embedding intent labels are retried with the LLM on later runs and never used as training labels.
- The fallback topic label is bounded. `panel build` refuses unclustered prompts. Re-clustering clears the stale scan panel. An unknown `--version` gives a clean error.

### Changed
- The mock embedder now uses calibrated feature hashing (stemmed words, char trigrams, bigrams, plus a fixed anchor dimension) instead of random vectors, so offline dedup and clustering behave like real embeddings.
- The Docker image ships `prompts/` and sets `PROMPTS_DIR`.

## module-00 - Foundations (2026-11-23)

### Added
- `pyproject.toml` (uv, hatchling src layout, ruff/mypy/pytest config), `uv.lock`, `citewise` Typer CLI with `seed` and `config` commands.
- Settings via pydantic-settings (`CITEWISE_MODE`, `DATABASE_URL`, provider keys, per-purpose and per-engine model IDs, `MAX_COST_USD_PER_RUN`). Blank env values fall back to defaults.
- SQLAlchemy 2.0 models `brands` and `llm_calls`, Alembic setup and migration `0001`.
- Provider layer: `ChatProvider` / `EmbeddingProvider` protocols, deterministic mock chat (fixture-keyed by purpose + messages, schema-aware fallback) and mock embedder, live OpenAI chat (Responses API, structured outputs, URL citations) and embeddings, stubs for Anthropic/Gemini/Perplexity.
- `tracking.py`: every call, including failures, writes an `llm_calls` row with tokens, estimated cost and latency. `pricing.py` holds the price table, last checked 2026-11-23.
- FastAPI app: `GET /health` (DB ping, 503 when the DB is down), `GET/POST /brands`.
- Example brands `Taskflow` (project management) and `Stridelab` (running shoes) in `data/brands/`.
- Multi-stage `Dockerfile` (uv, non-root, healthcheck) and `docker-compose.yml` (`db`, `api`, configurable host ports).
- CI: mypy step and a `docker build` job. Pre-commit pins bumped (ruff v0.17.6, gitleaks v8.30.1, pre-commit-hooks v6.0.0).
- Tests (mock mode, SQLite): config, providers, tracking, API, seed idempotency, CLI, Alembic upgrade/downgrade.

### Fixed (review, 2026-11-23)
- Mock structured output now resolves `$ref`/`$defs`, so Pydantic-generated nested schemas validate in mock mode.
- Live mode refuses models with no price (`UnpricedModelError`) instead of recording $0. Price lookup only strips dated snapshot suffixes, not arbitrary `-suffix` matches.
- Brand input is bounded (max 20 aliases/competitors, max 100 chars each), single-line, deduped, and the domain is normalised to a hostname.

## Kit

### Added
- Project kit: CLAUDE.md, module specs (0-8), Claude Code commands/subagents/settings, CI and gitleaks workflows, pre-commit config.
