# Changelog

All notable changes to Citewise are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). One section per module.

## [Unreleased]

## module-05 - Content agent (2026-11-24)

### Added
- Tables `content_opportunities`, `content_items`, `content_versions` and `content_reviews` (migration `0007`).
- Opportunity finder (`services/content/opportunities.py`):
  - visibility gaps by topic × intent;
  - citation gaps (get-listed / pitch suggestions; outreach stays manual);
  - fact fixes from contradicted claims;
  - ranked by impact × confidence with a reason each.
- Content pipeline state machine (`services/content/pipeline.py`):
  - brief → outline → draft → critique → revise (≤2) → final → awaiting_approval → approved/rejected;
  - every version is stored with its scores, checks and feedback;
  - resumable after errors or the spend cap.
- Deterministic guardrails and rubric caps (`services/content/checks.py`):
  - KB grounding of every factual sentence, title and heading (`[kb:<chunk>]` markers, numbers, superlatives, word overlap);
  - competitor defamation, fake reviews and testimonials, injected instructions, and length / keyword stuffing;
  - last-resort redaction.
- Outputs (`services/content/outputs.py`): Markdown with source footnotes, FAQPage JSON-LD with a JSON Schema, and a "why this should help" note.
- Measurement hook (`services/content/measure.py`): an approval baseline and before/after against later scans.
- Brand voice files `data/brands/{stridelab,taskflow,linear}/voice.md`.
- Versioned prompts `prompts/content/{brief,outline,draft,critique,revise}.v1.jinja`.
- Rubric `evals/rubrics/content.yaml` and red-team rows m5-rt-001…006.
- `providers/mock_content.py`: a simulated writer and judge for mock mode.
- API:
  - `GET /brands/{id}/content/opportunities`
  - `GET /content`
  - `GET /content/{id}`
  - `POST /content/{id}/approve|reject|resume`
- CLI: `citewise content opportunities | create | resume | list | show | approve | reject | export | measure`.
- Dev dependency `jsonschema`.

## module-04 - Brand knowledge base and FactCheck (2026-11-24)

### Added
- Knowledge base (`services/kb/`):
  - ingest of Markdown/TXT/PDF with front matter; URLs in live mode only, with robots.txt respected;
  - sentence-aware chunking with overlap and stable `chunk_key`s;
  - idempotent re-ingest;
  - hybrid retrieval: BM25 plus dense cosine, fused by RRF, with an optional validated LLM re-rank (`prompts/kb/rerank.v1`).
- FactCheck (`services/factcheck/`):
  - claim extraction (`prompts/factcheck/claims.v1`), keeping only claims whose span is in the answer;
  - verification (`verify.v1`), never contradicted without a verbatim KB quote;
  - source tracing (inline marker, title and, in live mode, page text);
  - idempotent `factcheck run`;
  - a read model;
  - per-engine sentiment narratives (`theme.v1`).
- Tables `kb_docs`, `kb_chunks`, `answer_factchecks` and `claims` (migration `0006`).
- `providers/mock_factcheck.py`: a simulated reader for claims, verify, re-rank and theme labels in mock mode.
- KBs for Stridelab and Taskflow (`data/kb/`).
- The mock world plants 8 wrong claims, each cited inline to a culprit page.
- **Linear** as a real brand (`data/brands/linear.yaml`, `data/kb/linear/` from official pages, 2026-11-24).
- Retrieval eval (`evals/retrieval.py`, golden `kb_qa.jsonl` with 36 queries; reports for mock and real local embeddings).
- CLI:
  - `citewise kb ingest | search`;
  - `factcheck run | show | narrative`;
  - `eval retrieval`.
- API: `GET /brands/{id}/factcheck`.
- Settings `KB_CHUNK_TOKENS` and `KB_CHUNK_OVERLAP`. Dependencies `rank-bm25` and `pypdf`.

## Provider switch: Claude for Citewise's own LLM calls (2026-11-24)

### Changed
- Citewise's own LLM calls now use Claude by default (`LLM_PROVIDER=anthropic`): `claude-haiku-5-5` for extraction, labelling and judging, and `claude-opus-5-5` for generation. Adds `providers/anthropic_client.py` (Messages API with structured outputs). `LLM_PROVIDER=openai` still works.
- Embeddings default to a local model (`EMBED_PROVIDER=local`, `BAAI/bge-small-en-v1.5` via fastembed, no key and no cost), because Anthropic has no embeddings API. `EMBED_PROVIDER=openai` still works.
- A live command refuses to start when a configured model doesn't belong to `LLM_PROVIDER` (a stale `.env` pinning `gpt-*` models).
- `scripts/record_extract_fixtures.py` and the panel commands use `LLM_PROVIDER`.
- Dependency `fastembed`. Prices: `claude-opus-5-5`, plus the local embedding model at $0.
- `.env.example` documents `LLM_PROVIDER`, `EMBED_PROVIDER` and the Claude model defaults.

### Added (live check)
- Recorded real `claude-haiku-5-5` Stage B responses for all 65 golden answers (`tests/fixtures/mock_chat/`, $0.034). The offline eval and CI now score a real model: merged F1 1.000, rank accuracy 98.9%, sentiment label accuracy 92.0%, recommendation accuracy 85.2%.

## module-03 - Extraction and visibility metrics (2026-11-24)

### Added
- Mention extraction (`services/extract/`):
  - Entity resolution (`resolve.py`): case, possessives, legal suffixes and TLDs; alias, product-name and fuzzy matching; unknown entities grouped by key.
  - Stage A alias + fuzzy matching with structural rank.
  - Stage B LLM extraction (`prompts/extract/mentions.v1.jinja`, strict `MentionExtraction` schema).
  - A merge with a disagreement log and guards: not in text, inside a link, fragment of a known name.
  - `extract run`: idempotent by extractor version, failed Stage B retried, clean stop at the spend cap.
- A simulated Stage B extractor for mock mode (`providers/mock_extract.py`).
- Tables `mentions` and `answer_extractions` (migration `0005`).
- Visibility metrics in SQL (`services/metrics/visibility.py`): mention rate, share of voice, average rank, top-3 rate, recommendation rate, net sentiment, position score, citation share (with vs without the brand, own-domain share) and competitors including emerging ones. Every metric can be sliced by engine, topic, intent or run. Adds prompt-clustered bootstrap 95% CIs and the 0-100 Visibility Score.
- Extractor eval (`evals/extract.py`): P/R/F1, rank accuracy, type accuracy, sentiment MAE and label accuracy, and recommendation accuracy, for Stage A, Stage B and the merge, with errors broken down by type and by tag.
- CLI `citewise extract run`, `metrics show`, `metrics competitors`, `eval extract`.
- API `GET /brands/{id}/metrics` and `GET /brands/{id}/competitors`.
- Eval data:
  - `evals/rubrics/mentions.yaml`;
  - `evals/datasets/golden/mentions.jsonl` (65 answers) plus its README;
  - red-team cases `m3-rt-001..003`;
  - `evals/reports/extract_eval.md/.json`;
  - `scripts/golden_mention_candidates.py`.
- Settings `EXTRACT_CONCURRENCY` and `METRICS_BOOTSTRAP`. Dependency `rapidfuzz`.
- Emerging brands in the mock world: Saucony, Altra, Basecamp and Wrike.

### Fixed
- The mock engine answered prompts with no category keyword from the first category, so 27 of 300 Taskflow prompts got running-shoe answers.
- Code review (extractor version is now `a2+mentions.v1`, so existing extractions re-run):
  - Nested bullets, numbered headings and multiple lists no longer produce wrong ranks.
  - Planted instructions, HTML comments, code blocks and schema-like JSON can no longer create or inflate a brand mention. They are masked from matching and redacted before Stage B.
  - Fuzzy and product-prefix resolution no longer merge distinct companies ("LinearB", "Monday Morning Co").
  - Recommendation rate and sentiment use only answers where Stage B ran. Slices report `coverage` and `llm_ok_share`, and `metrics show` warns when they are below 100%.
  - Mention-rate CIs no longer collapse to zero width at 0/n and n/n (Wilson floor).
  - The spend cap is thread-safe.
  - `--by run` shows no misleading ALL row.
- The mock extractor's lexicon no longer copies the answer simulator's phrases.

### Added (code review)
- `scripts/record_extract_fixtures.py`: record real Stage B responses so the offline eval replays a real model.
- `eval extract --min-f1` (exit 1 below it).
- The report shows a hand-written-rows table and the Stage B source (simulated vs replayed).
- CI runs the metrics SQL tests on a `postgres:16` service (`CITEWISE_TEST_PG_URL`).
- `.gitignore` ignored `evals/reports/`, so the Module 1 panel reports were never committed. They are now.

## module-02 review (2026-11-24)

### Added
- Setting `SCAN_MAX_OUTPUT_TOKENS` (per engine: OpenAI 6000, Anthropic 4096, Gemini 3000, Perplexity 2500), sent as `max_output_tokens` / `max_tokens` on every live engine call.
- `is_timeout()` in `providers/engines/base.py`.

### Changed
- The Anthropic answer cap went from a hardcoded 1500 to 4096 (configurable), because thinking tokens count against it.
- `execute_run` is split into setup/outcome (`execute_run`) and the per-call work (`_execute`).

### Fixed
- The run lease was only refreshed when an answer landed, so slow or timing-out calls let it lapse and a second executor could double-spend the run. A background task now beats every 10 s.
- A setup failure after the lease was taken (e.g. a missing API key on an API-started run) left the run `running` forever with no error. It is now `failed`, with the error, and the lease is released.
- SDK timeouts were costed at $0. They are now charged their estimate, like asyncio timeouts.
- OpenAI, Gemini and Perplexity answers had no output-token cap. Perplexity truncations (`status: incomplete`) were stored as ok answers.
- `execute_run` now refuses a run from the other mode (it was only checked by `scan resume`).
- Resume reads spend so far from `llm_calls` instead of a possibly stale `runs.cost_usd`.
- A citation domain longer than 255 characters would abort the whole run on Postgres. It is now truncated.

## module-02 - Multi-engine answer collection (2026-11-23)

### Added
- Answer-engine adapters (`providers/engines/`): a normalized `EngineAnswer` and async adapter protocol, plus live adapters for OpenAI (Responses API + `web_search`), Anthropic (Messages + `web_search_20250305`, `pause_turn` continuation), Gemini (google-genai + Google Search grounding) and Perplexity (the `perplexity/sonar` model on its Agent API). Each has a pure, offline-tested `parse_*`. API shapes were checked against the official docs on 2026-11-23.
- A deterministic mock answer simulator (`providers/engines/mock.py`, `tests/fixtures/engines/world.yaml`) with per-engine brand bias, sentiment, true and false brand claims, and realistic citations (OpenAI `utm_source`, Gemini grounding redirects).
- Scan runner (`services/collect/runner.py`): per-engine semaphores and token buckets, full-jitter exponential backoff on 429/5xx/timeouts, idempotent upserts on `(run_id, prompt_id, engine)`, resumable frozen plans, run leases, a budget estimate plus per-call cost reservations, and a same-day answer cache.
- Citation URL normalization (`services/collect/urls.py`).
- Tables `runs`, `answers`, `citations`, `schedules`, and `llm_calls.run_id` (migration `0004`).
- CLI `citewise scan run | resume | status | list`, `citewise schedule add | list | remove`, `citewise worker` (APScheduler 3.x).
- API `POST /brands/{id}/runs`, `GET /runs/{id}`, `GET /brands/{id}/runs`.
- Per-search fees in `pricing.py` (`SEARCH_FEE_USD`). Settings `SCAN_CONCURRENCY`, `SCAN_RATE_PER_S`, `SCAN_MAX_ATTEMPTS`, `SCAN_BACKOFF_BASE_S`, `SCAN_BACKOFF_MAX_S`, `SCAN_MAX_SEARCHES`, `SCAN_EST_TOKENS_IN`, `SCAN_EST_TOKENS_OUT`, `SCAN_EST_SEARCHES`.
- Dependencies: `anthropic`, `google-genai`, `apscheduler>=3.10,<4`, `rich`.

### Changed
- Default `ENGINE_MODEL_GEMINI` is now `gemini-3.5-flash-lite`, because 3.1 Flash-Lite isn't listed as supporting Search grounding. Default `ENGINE_MODEL_PERPLEXITY` is now `perplexity/sonar`, because Sonar Chat Completions ended on 2026-09-27.
- `CallRecorder.record` accepts `run_id` and an explicit `cost_usd`, so per-search fees are counted.

### Fixed (code review, 2026-11-23)
- The cache could serve mock answers to live runs. Run progress was only written at the end. Billed failures (timeouts, truncations, cancellations) were costed at $0. Two processes could execute one run. `panel reset` would have cascade-deleted historical answers. A bug in one task left sibling calls running. API scans blocked the event loop. Empty or truncated answers were stored as ok. Resume progress was miscounted. SDK clients were never closed.

## module-01 review (2026-11-23)

### Added
- `citewise panel reset <brand> [--yes]`: delete a brand's prompts and topics so the panel can be rebuilt (e.g. before the first live run).
- `prompts.embedding_model` (migration `0003`) and `EmbeddingProvider.ident`.

### Fixed
- Stored embeddings from another vector space (mock vs live, or a `MODEL_EMBED` change) are re-embedded before dedup, clustering and `eval-intent`, instead of being silently mixed.
- Piped CLI output no longer crashes on Windows for non-cp1252 text (Japanese, emoji).
- `eval-intent --golden` in live mode respects `--max-cost`. The cross-validation title now says "teacher labels".
- The live smoke-test instructions now reset the mock panel first (before, `generate` skipped every cell).

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
- Eval data: golden intent set (`evals/datasets/golden/intent.jsonl`, 94 hand-reviewed rows), labeling rubric (`evals/rubrics/intent_labels.yaml`), Module 1 red-team cases (`evals/datasets/redteam.jsonl`).
- Settings: `PANEL_PROMPTS_PER_CELL`, `PANEL_DEDUP_THRESHOLD`, `PANEL_K_MIN`/`PANEL_K_MAX`, `PANEL_PCA_DIMS`, `PANEL_CLASSIFY_BATCH`, `PROMPTS_DIR`.
- Dependencies: numpy, scikit-learn, jinja2 (runtime) and matplotlib (dev).

### Fixed (code review, 2026-11-23)
- Prompt injection: templates are split at the `=== user ===` line before rendering, the brand category is JSON-encoded in every template, and brand fields reject `===`.
- A provider error (429, timeout) no longer aborts `generate`/`cluster` and rolls back paid work. It counts as a failed attempt and the fallback runs.
- `panel generate` skips cells already generated with the template version (`--force` regenerates), so live re-runs don't pay again or grow the panel.
- Spend cap per command (`--max-cost`, default `MAX_COST_USD_PER_RUN`), `PANEL_IMPORT_MAX_ROWS`, and a sampled silhouette for large panels.
- Fallback/embedding intent labels are retried with the LLM on later runs and never used as training labels.
- Golden set review: one row reworded (it nearly duplicated a training prompt), one relabeled, 14 hard cases added. New rubric and classifier tie-breaks for contrast/exclusion ("unlike X") and for the brand name used as an ordinary word.
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
