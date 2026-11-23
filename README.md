# Citewise

**Measure how AI answer engines talk about your brand, understand why, and fix it.**

Citewise asks ChatGPT, Claude, Gemini and Perplexity the questions your buyers ask, then turns their answers into visibility metrics (mention rate, share of voice, rank, sentiment, cited sources). It fact-checks what they say against your brand's source of truth and drafts grounded content to close the gaps. Every LLM component is scored by an eval harness that gates CI.

> 🚧 Under construction: built module by module. See [PROGRESS.md](PROGRESS.md) and [docs/PLAN.md](docs/PLAN.md).

## Modules
0. Foundations · 1. Prompt panel and topic discovery · 2. Multi-engine collection · 3. Extraction and visibility metrics · 4. Knowledge base and FactCheck · 5. Content agent · 6. Evals and experiments · 7. AI crawler analytics · 8. Dashboard and deploy

## Quick start

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker Desktop (only for Postgres / the API container).

Citewise runs in **mock mode** by default: no API keys, no network, no cost. All LLM calls return deterministic fixtures.

### 1. Install and test (no Docker, no `.env` needed)

```bash
uv sync
uv run pytest
uv run citewise --help
```

### 2. Run against Postgres

Windows (PowerShell):

```powershell
Copy-Item .env.example .env        # optional; defaults work without it
docker compose up -d db
uv run alembic upgrade head
uv run citewise seed
uv run uvicorn citewise.api.main:app --reload
```

macOS / Linux:

```bash
cp .env.example .env               # optional; defaults work without it
docker compose up -d db
uv run alembic upgrade head
uv run citewise seed
uv run uvicorn citewise.api.main:app --reload
```

Then open <http://localhost:8000/health> (`{"status":"ok","db":"ok"}`), <http://localhost:8000/brands> and the interactive docs at <http://localhost:8000/docs>.

**Port already in use?** If another project already uses 5432 or 8000, set `CITEWISE_DB_PORT` / `CITEWISE_API_PORT` in `.env` (e.g. `5434`) and update the port in `DATABASE_URL` to match.

### 3. Everything in Docker

```bash
docker compose up -d --build       # db + api; the api runs migrations on start
docker compose exec api citewise seed
```

### Live mode

Set `CITEWISE_MODE=live` and `OPENAI_API_KEY` in `.env`. Only the OpenAI provider (chat + embeddings) is live in Module 0. The other engines arrive in Module 2. Model IDs are set per purpose (`MODEL_EXTRACT`, `MODEL_JUDGE`, `MODEL_GENERATE`, `MODEL_EMBED`) in `.env`.

## How it's built

- `src/citewise/providers/` is the only place that talks to LLMs. `registry.get_chat_provider()` / `get_embedder()` return mock or live backends wrapped by `tracking.py`, which writes **every** call (tokens, estimated $ cost, latency, errors) to the `llm_calls` table.
- Prices live in `src/citewise/providers/pricing.py`, with a "last checked" date.
- `citewise config` shows the active mode, models and which keys are set (never the values).

Useful commands: `uv run ruff check .`, `uv run ruff format .`, `uv run mypy`, `uv run pre-commit run --all-files`.

## Limitations
Citewise uses official APIs only (with web search where available). API answers approximate, but are not identical to, what consumer apps like chatgpt.com show.

## License
MIT
