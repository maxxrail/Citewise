# Module 0 — Foundations

## Goal
A clean, boring, production-shaped skeleton that every later module plugs into. Nothing clever yet; the point is that it works on Maxx's Windows machine and in CI on day one.

## Already in the repo (from the kit; keep and extend, don't recreate)
`.github/workflows/ci.yml` + `gitleaks.yml`, `.pre-commit-config.yaml`, `.gitignore`, `.gitattributes`, `.dockerignore`, `.gitleaksignore`, `.env.example`, `LICENSE`, `CHANGELOG.md`, and the empty folders (`src/citewise`, `prompts`, `evals`, `ops`, `scripts`, `data`, `tests`, `migrations`). Delete a `.gitkeep` once its folder has real files.

## Build
0. **Repo files this module creates**: `pyproject.toml` (src layout via hatchling, `[project.scripts] citewise = "citewise.cli:app"`, ruff/mypy/pytest config inside), `uv.lock`, `alembic.ini` at root pointing to `migrations/`, `Dockerfile` (multi-stage, uv, non-root user), `docker-compose.yml` (`db`, `api`; `worker` and `web` added later). Bump the `rev:` pins in `.pre-commit-config.yaml` to current versions and run `uv run pre-commit install`.
1. **Project setup**: `pyproject.toml` (uv), ruff, mypy (lenient), pytest, pre-commit (optional). Package `citewise` in a `src/` layout. CLI entry point `citewise` built with Typer in `src/citewise/cli.py` (`citewise --help`).
2. **Config** (`src/citewise/core/config.py`): pydantic-settings. Keys: `CITEWISE_MODE` (`mock`|`live`), `DATABASE_URL`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `PERPLEXITY_API_KEY`, per-purpose model IDs (`MODEL_EXTRACT`, `MODEL_JUDGE`, `MODEL_GENERATE`, `MODEL_EMBED`), `MAX_COST_USD_PER_RUN`. `.env.example` with every key.
3. **Database**: SQLAlchemy 2.0 models + Alembic. Initial tables: `brands`, `llm_calls`. `docker-compose.yml` with `db` (postgres:16) and `api` services. SQLite for tests via fixture.
4. **Provider layer** (`src/citewise/providers/`):
   - `base.py`: `ChatProvider` protocol → `complete(messages, *, model, json_schema=None, tools=None) -> ChatResult` (text, parsed, citations, usage, latency_ms, raw).
   - `EmbeddingProvider` protocol → `embed(texts) -> list[list[float]]`.
   - `mock.py`: deterministic responses keyed by hash of (purpose, input), loaded from `tests/fixtures/`; mock embedder = seeded hash → unit vector (stable across runs).
   - `registry.py`: `get_chat_provider(name)` / `get_embedder()` choose mock or live from config.
   - Live clients can be stubs that raise `NotImplementedError` for now, **except OpenAI**, which should work for chat + embeddings.
   - Every call is logged to `llm_calls` through one wrapper (`src/citewise/providers/tracking.py`) with a token-price table in `src/citewise/providers/pricing.py` (prices as config, with "last checked" date comment).
5. **API**: FastAPI app with `/health` (db ping) and `POST/GET /brands`.
6. **Seed data**: `data/brands/` with 2 example brands as YAML (e.g. a project-management SaaS and a running-shoe brand), each with name, domain, category, aliases, 4–6 competitors. `citewise seed` loads them.
7. **CI**: GitHub Actions: `uv sync`, `ruff check`, `pytest` in mock mode.
8. **Docs**: `README.md` (quick start for Windows + macOS/Linux), `PROGRESS.md`, `docs/DECISIONS.md`.

## Acceptance criteria
- [ ] `uv sync && uv run pytest` passes with no `.env` and no network.
- [ ] `uv run citewise --help` lists `seed`.
- [ ] `docker compose up -d db` then `uv run alembic upgrade head` then `uv run citewise seed` works; `GET /brands` returns the 2 brands.
- [ ] `GET /health` returns `{"status":"ok","db":"ok"}`.
- [ ] A test proves every provider call writes an `llm_calls` row with cost and latency.
- [ ] `ruff check .` and `ruff format --check .` are clean; `uv run pre-commit run --all-files` passes.
- [ ] `docker build .` succeeds.
- [ ] CHANGELOG.md has a `module-00` section.

## Out of scope
Frontend, real engines other than OpenAI, scheduling.
