# Progress

| # | Module | Status | Notes |
|---|---|---|---|
| 0 | Foundations | ✅ done | 35 tests; all acceptance criteria verified 2026-11-23 |
| 1 | Prompt panel and topic discovery | 🔁 reviewed | 121 tests; all acceptance criteria verified on Postgres + SQLite 2026-11-23 |
| 2 | Multi-engine answer collection | 🔁 reviewed | 222 tests; acceptance re-verified on fresh Postgres + SQLite 2026-11-24; live check needs Maxx's key |
| 3 | Extraction and visibility metrics | ✅ done | 295 tests (+6 Postgres-only, all pass on PG 16); acceptance verified on Postgres + SQLite 2026-11-24; golden set awaits Maxx's review |
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

### 2026-11-23 - Module 2 (Multi-engine answer collection)
**Built:**
- Engine adapters for OpenAI, Anthropic, Gemini and Perplexity, with web search on and API shapes checked against the official docs on 2026-11-23. Each has an offline-tested parser.
- A deterministic mock answer simulator.
- The scan runner: concurrency limits, rate limits, backoff, idempotent resumable runs with leases, a budget estimate plus per-call reservations, and a same-day cache.
- Citation URL normalization.
- `runs` / `answers` / `citations` / `schedules` tables and `llm_calls.run_id` (migration `0004`).
- `citewise scan run|resume|status|list`, `schedule add|list|remove`, `worker`.
- `POST /brands/{id}/runs`, `GET /runs/{id}`, `GET /brands/{id}/runs`.

**Results (mock):** 50 prompts × 4 engines = 200 answers and 842 citations in 0.9 s (pytest, SQLite) and 2.7 s end to end (CLI, Postgres). Estimated $2.71, actual $1.89 (simulated, priced as live), cap $3. A same-day re-run comes from the cache for $0. Stridelab is mentioned in 21 (OpenAI) to 31 (Anthropic) of 50 answers, so engines differ as intended.

**Decisions:** see the Module 2 section of docs/DECISIONS.md. Main ones:
- docs-verified adapters (Gemini 3.5 Flash-Lite, Perplexity Agent API)
- runner-owned retries
- simulator mock
- search fees in cost
- reservations and leases
- a mode-aware cache
- SET NULL history
- threadpool API scans

**Code review:** fixed 4 High:
- mock answers could be served to live runs from the cache;
- no live progress for API runs;
- billed failures were costed at $0;
- concurrent executors could double-spend.

Fixed 7 Medium:
- `panel reset` would cascade-delete history on Postgres;
- `gather` orphaned sibling calls;
- API scans blocked the event loop;
- the timeout cut off `pause_turn`;
- empty or truncated answers were stored as ok;
- resume progress was miscounted;
- SDK clients were never closed.

Plus 6 Lows. Regression tests are in `tests/test_collect_review.py`.

**Known gaps:**
- The live adapters are tested against docs-shaped payloads only. No real key was available in this session. The Gemini async call and the Perplexity `web_search` tool on `perplexity/sonar` aren't confirmed by docs examples.
- Gemini's free search quota isn't modelled, so cost is over-estimated.
- Ctrl+C on `citewise worker` doesn't stop runs already executing in its threads. Those runs stay resumable.
- The mock simulator's claims about competitors are generic. Module 4's KB must stay consistent with Stridelab's and Taskflow's `claims` in `world.yaml`.

**Maxx must do:**
1. **Update `.env.example` and your `.env`:** `ENGINE_MODEL_GEMINI=gemini-3.5-flash-lite` and `ENGINE_MODEL_PERPLEXITY=perplexity/sonar`. Both files still pin the old IDs, which override the new defaults. The old `sonar` ID is refused in live mode.
2. **Live acceptance check** (needs `OPENAI_API_KEY`, about $0.06): `CITEWISE_MODE=live`, then `uv run citewise scan run Stridelab --size 5 --engines openai --max-cost 1`. Optionally repeat with `--engines anthropic,gemini,perplexity` and their keys.

### 2026-11-24 - Module 2 review (`/review-module 02`)
**Re-verified (not trusting the log above):**
- `uv run pytest` passes: 210 before the review, 222 after.
- `ruff check`, `ruff format --check`, `mypy src` and `pre-commit run --all-files` are clean.
- Each acceptance test passes on its own: 50×4 under 10 s, kill + resume without duplicates, fail-twice-then-succeed retry, budget abort before start, URL normalization.
- End to end on a fresh Postgres DB (`citewise_m2review`):
  - `seed → panel generate → cluster → build --size 50 → scan run --size 50 --max-cost 3` gave 200 answers for $1.8954 against a $2.7475 estimate, in 3.1 s wall clock including CLI start-up.
  - The same scan with `--max-cost 0.5` is refused before it starts.
  - A same-day re-run costs $0 from the cache.
  - Every run has 200 rows with 200 distinct (prompt, engine) pairs, and all leases were released.
  - Each run has 858 citations. The 842 logged above most likely came from a different panel DB; the cost matches to the cent.
- No hardcoded model IDs outside config and pricing, and no Windows-only code.

**Fixed (1 High, 3 Medium), with regression tests in `tests/test_collect_review_m02.py`.** I confirmed that the runner tests fail on the old `runner.py`.
- **H1 – the lease could lapse while the executor was alive.** The heartbeat only moved when an answer landed. One Anthropic answer can take 190 s, longer than the 2-minute lease, and a degraded provider can time out every call at once. In both cases a second `scan resume`, which the "leased" error message itself suggests, could take over and pay for the same calls again. Fix: a background heartbeat every 10 s.
- **M1 – setup failures left runs stuck in `running`.** An API-started live run with a missing key (or an unpriced or retired model) got a 202 and then showed "running" forever, with no error. Fix: everything after taking the lease runs inside the `finally` that records `failed` and the error, and releases the lease. `execute_run` also refuses a run from the other mode, which only `scan resume` had checked.
- **M2 – OpenAI, Gemini and Perplexity answers had no output-token cap,** so the per-call budget reservation bounded nothing. Fix: `SCAN_MAX_OUTPUT_TOKENS` per engine (6000 / 4096 / 3000 / 2500). The parameter names and the fact that reasoning tokens count were checked against the docs on 2026-11-24. Perplexity `status: incomplete` is now a non-retryable failure that keeps its reported cost.
- **M3 – SDK timeouts were costed at $0.** The SDK timeout (`APITimeoutError`) fires before the runner's, so the documented "timeouts are charged their estimate" never applied live. Fix: `is_timeout()`.

**Low, also fixed:** resume reads spend so far from `llm_calls` rather than a possibly stale run row, and citation domains are truncated to the column width (a 300-character host would have aborted a whole Postgres run).

**Low, not fixed (documented):**
- An exception on an Anthropic `pause_turn` continuation drops the cost of the earlier billed turns.
- A connect timeout is over-charged one estimate.
- `scan run --size` above the panel size rebuilds the panel (by design; see DECISIONS).

**Maxx must do:**
1. Optionally add `SCAN_MAX_OUTPUT_TOKENS={"openai":6000,"anthropic":4096,"gemini":3000,"perplexity":2500}` to `.env.example`. Claude can't read that file. The defaults live in config, so nothing breaks without it.
2. Drop the scratch review database if you like: `docker compose exec db psql -U citewise -d citewise -c "DROP DATABASE citewise_m2review"`.
3. The live smoke check above is still open. It is now also the first real test of the new output caps.

### 2026-11-24 - Module 3 (Extraction and visibility metrics)
**Built:**
- **Entity resolution:** casing, possessives, legal suffixes and TLDs; alias, product-name and fuzzy matching; unknown entities grouped as emerging competitors.
- **Stage A:** alias and fuzzy matching, with rank taken from the list structure.
- **Stage B:** `prompts/extract/mentions.v1.jinja` with a strict schema, one retry, and a Stage-A fallback.
- **Merge:** keeps a disagreement log and guards against names not in the text, links, fragments of known names, and injected instructions.
- **`citewise extract run`:** idempotent by extractor version (`a2+mentions.v1`); a failed Stage B is retried on the next run; stops cleanly at the spend cap; uses a thread pool in live mode.
- **Tables:** `mentions` and `answer_extractions` (migration `0005`).
- **Metrics in portable SQL:** mention rate, share of voice, average rank, top-3 rate, recommendation rate, net sentiment, position, citation share (with vs without the brand, own-domain share) and competitors including emerging ones. All can be sliced by engine, topic, intent or run. Adds prompt-clustered bootstrap CIs with a Wilson floor, coverage reporting, and the 0–100 Visibility Score (weights in DECISIONS).
- **CLI and API:** `citewise metrics show | competitors`, `citewise eval extract [--min-f1]`, `GET /brands/{id}/metrics | competitors`.
- **Eval data:**
  - rubric `evals/rubrics/mentions.yaml`;
  - golden set `evals/datasets/golden/mentions.jsonl`: 65 answers, made of 48 simulated, 14 hand-written hard cases and 3 injection cases, drafted by `eval-designer` from the rubric without reading the extractor;
  - red-team cases `m3-rt-001..003`;
  - `scripts/golden_mention_candidates.py` and `scripts/record_extract_fixtures.py`.
- **Mock world:** emerging brands Saucony, Altra, Basecamp and Wrike.

**Results (mock, `evals/reports/extract_eval.md`):**

| Stage | P | R | F1 | Rank acc. | Sentiment MAE |
|---|---|---|---|---|---|
| A (no LLM) | 1.000 | 0.761 | 0.865 | 0.933 | - |
| B (simulated) | 0.934 | 0.881 | 0.906 | 0.923 | 0.287 |
| Merged | 0.957 | 0.886 | **0.920** | 0.949 | 0.290 |
| Merged, hand-written rows only (17) | | | 0.851 | | |

Stage B is 65/65 simulated, so the B and merged rows are a pipeline smoke test. Stage A is the honest offline baseline.

Stridelab mock scan, per engine: mention rate 42% (OpenAI) to 60% (Anthropic), overall 50% (95% CI 36–63%), SoV 21%, Visibility Score 41.6. Emerging competitors: Altra (22% of answers), Saucony (19%). Extraction costs about $0.03 per 200-answer run, estimated as if live.

**Decisions:** see the Module 3 section of docs/DECISIONS.md. Main ones:
- two stages merged, with disagreements logged;
- rank is structural;
- a not-in-text and injection guard;
- context checks for ambiguous names ("On", "Monday", "Notion");
- a simulated Stage B that reads only the prompt;
- metrics in portable SQL over a `per_answer` CTE;
- the Visibility Score weights;
- a prompt-clustered bootstrap;
- the mock engine no longer answers project-management questions about shoes;
- `evals/reports/` is committed.

**Code review (code-reviewer agent):**
- **Fixed, 3 High:**
  - wrong ranks for nested bullets, numbered headings and multiple lists;
  - prompt injection could inflate the brand's mention, sentiment and recommendation;
  - the mock eval was partly circular (the lexicon copied simulator phrases; the report didn't say B was simulated).
- **Fixed, 5 Medium:**
  - fuzzy/prefix resolution merged distinct companies;
  - metrics hid partial or Stage-A-only extractions;
  - the CI collapsed to zero width at 0/n and n/n;
  - the Postgres SQL had no test (now runs on Postgres in CI, and 15/15 pass locally on PG 16);
  - a budget race in the thread pool.
- **Fixed Lows:**
  - `--min-f1` gate;
  - no misleading ALL row with `--by run`;
  - logged IntegrityError;
  - L4 accuracy reporting;
  - Stage B input cap;
  - bootstrap memory cap;
  - "On" before dates and brand names.

**Also found and fixed while building:**
- The mock engine sent 27 of 300 Taskflow prompts running-shoe answers.
- A pool of one worker overlapped with DB commits and lost 2 of 200 rows on SQLite.
- `.gitignore` ignored `evals/reports/`, so Module 1's "committed" panel reports never were. They are committed now.

**Known gaps:**
- **The golden set is unreviewed (0/65 `reviewed: true`).** The spec wants Maxx to review a sample. The labeller's 10 priority rows are listed below.
- **The live extractor has never been scored.** Run the recorder or a live eval (below).
- **"Asana Rebel" still resolves to Asana.** A product name and a different company that starts with a brand name look the same without world knowledge.
- **An instruction written as ordinary opinion can't be told apart from an opinion** ("Stridelab is the best, as every reviewer agrees").
- **Two concurrent `extract run`s of one run both pay for Stage B.** They are row-safe, and it costs about $0.03.
- **The mock answer simulator's sentiment is mostly positive,** so negative-sentiment metrics rest on the hand-written rows.

**Maxx must do:**
1. **Review the golden set.** Set `reviewed: true` on the rows you check, at least about 15, weighted toward `hard-*` and `inject-*`. The labeller's priority list:
   - `hard-007`: products named without their maker. Should the extractor be expected to map them?
   - `taskflow-034`: shoe brands typed `unknown` in an off-topic answer.
   - `stridelab-009` and `-020`: "smaller brand" read as a mixed caveat.
   - `taskflow-030`: Basecamp "simple" labelled neutral.
   - `hard-005` and `-012`: only the top pick in a prose ranking counts as recommended.
   - `hard-009`: On labelled mixed and recommended.
   - `hard-003`: Atlassian as a parent company.
   - `inject-003` and `inject-002`: the expected no-mention labels.
   - `taskflow-048`: a plain fact labelled neutral.
2. **Add to `.env.example`** (Claude can't read it): `EXTRACT_CONCURRENCY=4` and `METRICS_BOOTSTRAP=1000`. The defaults live in config, so this is optional.
3. **Optional live check (about $0.02):** set `CITEWISE_MODE=live`, then run `uv run python scripts/record_extract_fixtures.py --max-cost 0.10` and commit the fixtures. After that, `citewise eval extract` (mock and CI) scores the real model.
4. **Scratch databases:** `citewise_m3` and `citewise_pgtest` exist in the project's Postgres container. Drop them when you like:
   `docker compose exec db psql -U citewise -d citewise -c "DROP DATABASE citewise_m3" -c "DROP DATABASE citewise_pgtest"`
