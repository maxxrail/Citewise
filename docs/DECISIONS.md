# Design decisions

Format: **Decision** — why — alternative rejected.

- **Official APIs only, no UI scraping.** Stays within provider ToS and can be reproduced; API answers with web search on approximate consumer answers. Rejected: headless-browser scraping of chatgpt.com (fragile, against ToS). Limitation noted in README.
- **SQLite for tests, Postgres for dev/prod.** Tests run on Windows with no Docker. Rejected: testcontainers (slow and fiddly on Windows).
- **Embeddings stored as arrays in a regular column, similarity done in numpy.** Panel sizes are thousands, not millions. Rejected: pgvector (extra setup); revisit if scale grows.
- **Mock mode is the default.** CI and the public demo never spend money; fixtures make results deterministic.

## Module 0

- **OpenAI live adapter uses the Responses API, not Chat Completions.** OpenAI's docs (checked 2026-11-23) recommend Responses for new projects, and it returns web-search `url_citation` annotations that Module 2 needs. Rejected: Chat Completions (still supported, but older and weaker on tool and search output).
- **Mock calls record an estimated cost from the real model's price.** `llm_calls.provider = "mock"` marks them, but `cost_usd` is what the call would have cost live. Demos show realistic spend and the Module 2 budget guard can be tested offline. Rejected: cost 0 in mock mode (the cost path would never run in CI).
- **Mock fixtures are keyed by sha256(purpose, messages[, engine]), without the model.** Lookup tries the engine-specific key first, then the generic one, so the four engines give different mock answers and per-engine share of voice means something in CI. Mock rows record `provider = "mock:<engine>"`. Changing a model ID doesn't invalidate recorded fixtures. A missing fixture falls back to a deterministic placeholder, which is schema-valid when a JSON schema is requested, so new code works in mock mode before anyone records a fixture. Rejected: failing on a missing fixture (too much friction for every new prompt).
- **Mock embedder = sha256(text) seeds a PRNG, giving a Gaussian 1536-dim unit vector.** Stable across OSes and Python versions, unlike `hash()`, which is salted per process. Same shape as `text-embedding-3-small`.
- **Call tracking writes in its own session and never raises.** A caller's rollback can't erase a cost record, and a DB hiccup can't kill an LLM call. Failed calls are still logged (`ok=False`, error text) and then re-raised. Rejected: logging in the caller's transaction.
- **`brands.aliases` / `competitors` are JSON columns, not Postgres `ARRAY`.** The same model works on SQLite in tests. Rejected: ARRAY with a SQLite type decorator (more code for no gain at this size).
- **`env_ignore_empty=True` in settings.** A freshly copied `.env.example` contains blank lines like `OPENAI_API_KEY=`, which should mean "unset", not the empty string.
- **Docker host ports are configurable (`CITEWISE_DB_PORT`, `CITEWISE_API_PORT`, defaults 5432/8000).** Other local projects often already use 5432/8000.
- **The API container runs `alembic upgrade head` before uvicorn.** `docker compose up` works on a fresh clone without manual steps. Seeding stays explicit (`docker compose exec api citewise seed`).
- **Calls are priced by the requested model alias; `pricing.price_for` also accepts dated snapshots (longest `key-` prefix).** OpenAI returns resolved snapshot IDs. Exact-match pricing would record live calls as free and bypass the budget guard (found in code review).
- **`JsonSchema` is checked for OpenAI strict-mode compatibility when it is constructed.** Mock mode would otherwise accept schemas the live API rejects with a 400 (`additionalProperties: false` and all properties `required`). `strict=False` is an explicit opt-out.
- **Responses that aren't `completed` (e.g. truncated) are returned to the caller but logged `ok=False` with full token cost.** They still bill tokens. The caller decides whether to retry. Rejected: raising (would lose the partial text and the token count).
- **Mock fixtures stay in `tests/fixtures/` (per CLAUDE.md) and ship in the Docker image.** `.dockerignore` re-includes `tests/fixtures`. The image sets `FIXTURES_DIR` / `BRANDS_DIR` explicitly instead of relying on the editable-install path.
