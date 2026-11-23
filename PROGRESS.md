# Progress

| # | Module | Status | Notes |
|---|---|---|---|
| 0 | Foundations | ✅ done | 35 tests; all acceptance criteria verified 2026-11-23 |
| 1 | Prompt panel and topic discovery | ⬜ not started | |
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
