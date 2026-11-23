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

### 4. Build a prompt panel (Module 1)

The prompt panel is the set of questions real buyers ask AI assistants about a brand's category. Every command below works in mock mode with no keys:

```bash
uv run citewise panel generate Stridelab          # ~300 candidates per brand -> dedup -> stored with embeddings
uv run citewise panel cluster Stridelab           # topics (PCA + KMeans, silhouette-picked k) + intent labels
uv run citewise panel eval-intent Stridelab --golden evals/datasets/golden/intent.jsonl
uv run citewise panel build Stridelab --size 50   # stratified topic x intent sample to scan in Module 2
uv run citewise panel import Stridelab my_prompts.csv   # bring real prompts: a `text` column (+ optional intent, persona, funnel_stage)
uv run python scripts/panel_report.py Stridelab   # evals/reports/panel_stridelab.md + .png (add --fresh to run it all in memory)
```

The API serves the results at `GET /brands/{id}/prompts` (filters: `topic_id`, `intent`, `in_panel`, `limit`, `offset`) and `GET /brands/{id}/topics`. Committed example reports: [Stridelab](evals/reports/panel_stridelab.md), [Taskflow](evals/reports/panel_taskflow.md).

Prompt templates live in `prompts/panel/*.v1.jinja`, and every generated prompt stores its template version. In mock mode the generated prompts, intent labels and topic labels come from hand-written sources in `tests/fixtures/panel_source/`. If you change a template, the mock embedder or the clustering, re-record the fixtures with `uv run python scripts/record_panel_fixtures.py`. A test fails if they go stale.

### Live mode

Set `CITEWISE_MODE=live` and `OPENAI_API_KEY` in `.env`. Only the OpenAI provider (chat + embeddings) is live so far. The other engines arrive in Module 2. A live `panel generate` + `cluster` for one brand is about 15 generation calls on `MODEL_GENERATE`, about 25 small classification and labeling calls on `MODEL_EXTRACT`, and 2 embedding batches. Check `llm_calls` for the exact cost. Model IDs are set per purpose (`MODEL_EXTRACT`, `MODEL_JUDGE`, `MODEL_GENERATE`, `MODEL_EMBED`) in `.env`.

## How it's built

- `src/citewise/providers/` is the only place that talks to LLMs. `registry.get_chat_provider()` / `get_embedder()` return mock or live backends wrapped by `tracking.py`, which writes **every** call (tokens, estimated $ cost, latency, errors) to the `llm_calls` table.
- Prices live in `src/citewise/providers/pricing.py`, with a "last checked" date.
- `citewise config` shows the active mode, models and which keys are set (never the values).

Useful commands: `uv run ruff check .`, `uv run ruff format .`, `uv run mypy`, `uv run pre-commit run --all-files`.

## Limitations
Citewise uses official APIs only (with web search where available). API answers approximate, but are not identical to, what consumer apps like chatgpt.com show.

The prompt panel has no search-volume data: every prompt counts equally. See docs/DECISIONS.md (Module 1) for how real volumes would plug in.

## License
MIT
