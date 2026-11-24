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
uv run citewise panel reset Stridelab            # delete this brand's prompts + topics to start over (asks first; --yes to skip)
```

Re-running `generate` skips cells that already have prompts (add `--force` to regenerate). `generate` and `cluster` stop making LLM calls once `--max-cost` (default `MAX_COST_USD_PER_RUN`, $3) of estimated spend is reached, and print what they spent. `cluster` replaces the topics and clears the scan panel, so run `build` again afterwards.

The API serves the results at `GET /brands/{id}/prompts` (filters: `topic_id`, `intent`, `in_panel`, `limit`, `offset`) and `GET /brands/{id}/topics`. Committed example reports: [Stridelab](evals/reports/panel_stridelab.md), [Taskflow](evals/reports/panel_taskflow.md).

Prompt templates live in `prompts/panel/*.v1.jinja`, and every generated prompt stores its template version. In mock mode the generated prompts, intent labels and topic labels come from hand-written sources in `tests/fixtures/panel_source/`. If you change a template, the mock embedder or the clustering, re-record the fixtures with `uv run python scripts/record_panel_fixtures.py`. A test fails if they go stale.

### 5. Scan the answer engines (Module 2)

A scan sends the brand's panel to ChatGPT (OpenAI), Claude (Anthropic), Gemini and Perplexity with web search on, and stores every answer, its citations, tokens, latency and cost. In mock mode the engines are deterministic simulators, so all of this works offline:

```bash
uv run citewise scan run Stridelab --engines openai,anthropic,gemini,perplexity --size 50 --max-cost 3
uv run citewise scan list Stridelab           # recent runs: status, answers, cost
uv run citewise scan status 1                 # one run
uv run citewise scan resume 1                 # finish an interrupted / budget-stopped / partial run
uv run citewise scan run Stridelab --size 20 --system "The user is in Toronto, Canada."   # persona / locale
```

- **Budget:** the run is refused before it starts if its estimate exceeds `--max-cost` (default `MAX_COST_USD_PER_RUN`). During the run it stops once actual spend reaches the cap, and `scan resume --max-cost <higher>` continues. Cost includes the engines' per-search fees, not just tokens.
- **Safe to interrupt:** Ctrl+C (or a crash) leaves a resumable run. Resuming only asks the prompt × engine pairs that have no answer yet, so nothing is duplicated. A run can only be executed by one process at a time (`scan resume --force` takes over after a crash).
- **Free re-asks:** asking the same question of the same engine on the same UTC day reuses the stored answer (`--no-cache` to re-ask).
- **Smaller scans** (`--size 20`) use a stratified subset of the existing panel, so repeated scans keep asking the same questions.

Recurring scans (weekly snapshots are what make trends possible):

```bash
uv run citewise schedule add Stridelab --cron "0 9 * * 1" --size 50 --max-cost 3   # Mondays 09:00 UTC
uv run citewise schedule list
uv run citewise worker                        # runs the schedules; picks up changes every minute
```

API: `POST /brands/{id}/runs` (body `{"engines": [...], "size": 50, "max_cost": 3, "system_prompt": null}`; 202, runs in the background; 422 with the estimate if it's over budget), `GET /runs/{id}` (status, progress, cost), `GET /brands/{id}/runs`.

### 6. Extract mentions and measure visibility (Module 3)

Extraction turns each stored answer into structured mentions: which companies it names (including ones you don't track yet), their rank in any list, sentiment, and whether the answer recommends them. Then the metrics are computed in SQL:

```bash
uv run citewise extract run 1                 # all ok answers in run 1; safe to re-run (--force to redo)
uv run citewise metrics show Stridelab        # per engine: mention rate (95% CI), SoV, rank, top-3, recommended, sentiment, score
uv run citewise metrics show Stridelab --by topic      # or intent, run (trend across scans), all; --engine gemini; --json
uv run citewise metrics competitors Stridelab # tracked competitors + EMERGING ones engines name that you don't track
uv run citewise eval extract                  # P/R/F1, rank accuracy, sentiment MAE vs the golden set -> evals/reports/extract_eval.md
```

- **Two stages.** Stage A is deterministic alias and fuzzy matching (free, precise, blind to sentiment and unknown brands). Stage B is an LLM with a strict JSON schema (`prompts/extract/mentions.v1.jinja`). They are merged, and every disagreement is stored for evals. Rank is computed from the answer's list structure, not taken from the model.
- **Metric definitions** are in `src/citewise/services/metrics/visibility.py`. The Visibility Score weights are in docs/DECISIONS.md. Confidence intervals are bootstrapped over *prompts* (the four engines' answers to one prompt aren't independent), and slices under 30 answers are starred.
- **Eval:** the golden set is `evals/datasets/golden/mentions.jsonl` (65 answers; labelling rules in `evals/rubrics/mentions.yaml`). In mock mode Stage B is a simulated extractor, so the [committed report](evals/reports/extract_eval.md) is a smoke test of the pipeline, not a model score; it reports the 17 hand-written rows separately. For real numbers run `eval extract` with `CITEWISE_MODE=live`, or record real Stage B responses once (`uv run python scripts/record_extract_fixtures.py --max-cost 0.10`, live mode) so the offline eval and CI replay them. `eval extract --min-f1 0.85` exits 1 below the bar.

API: `GET /brands/{id}/metrics` (query: `run_id`, `engine`, `topic`, `intent`, `by=engine|topic|intent|run|all`; 409 until a run is extracted) and `GET /brands/{id}/competitors`.

### 7. Knowledge base and FactCheck (Module 4)

Each brand keeps a **source of truth** (`data/kb/<brand>/*.md`, with `title`, `category`, `as_of` and `source_url` front matter). FactCheck compares what the engines claim about the brand against it and points to the cited page that probably caused each wrong claim:

```bash
uv run citewise kb ingest Taskflow data/kb/taskflow      # chunk + embed; re-runnable (also .txt/.pdf, or a URL in live mode)
uv run citewise kb search Taskflow "Jira importer"       # debug retrieval: BM25 + dense, fused (RRF); --rerank for an LLM re-rank
uv run citewise extract run <run id>                     # FactCheck uses Module 3's brand mentions
uv run citewise factcheck run <run id>                   # claims -> KB evidence -> verdict -> likely source
uv run citewise factcheck show Taskflow                  # each wrong claim: KB quote, engine, likely source URL
uv run citewise factcheck narrative Taskflow             # per engine: top positive and negative themes
uv run citewise eval retrieval                           # recall@1/@5 + MRR: BM25 vs dense vs hybrid -> evals/reports/
```

- **Verdicts:** `supported`, `contradicted` or `not_in_kb`. A verdict is never `contradicted` (or `supported`) without a verbatim quote from a retrieved KB chunk; otherwise it is downgraded and the downgrade is recorded.
- **Claims** must appear in the answer itself. Injected instructions, hidden HTML and planted JSON are stripped first, as in Module 3.
- **Source tracing** uses the inline citation marker next to the claim, the citation's title and, in live mode, the cited page's text.
- **Mock mode** plants 8 wrong claims (wrong prices, a "discontinued" product, "no Jira import", ...) in the simulated answers, each cited to the page that caused it. All 8 are flagged, each with the refuting KB quote and its source. The verifier is a simulated reader in mock mode, so for real-model numbers run with `CITEWISE_MODE=live`.
- **Linear** is a real brand with a KB built only from its official pages (dated 2026-11-24). Its facts change, so re-check `data/kb/linear/` before trusting a verdict.

API: `GET /brands/{id}/factcheck` (query: `run_id`, `verdict=contradicted|supported|not_in_kb|all`; 409 until a run is checked).

### 8. Content agent (Module 5)

The content agent turns visibility gaps and wrong facts into grounded, publish-ready pages, and a human approves each one. It needs a scanned, extracted and fact-checked run (sections 4–7) and the brand's KB:

```bash
uv run citewise content opportunities Stridelab    # ranked: gaps, citation gaps (pitch only), fact fixes; each with "why"
uv run citewise content create <opportunity id>    # brief -> outline -> draft -> critique -> revise (<=2) -> awaiting approval
uv run citewise content show <item id>             # versions with rubric scores, the page, guardrails, "why this should help"
uv run citewise content list --status awaiting_approval
uv run citewise content reject <item id> --comment "Mention the 60-day trial."   # the comment drives a revision
uv run citewise content approve <item id>          # records a baseline on the target prompts
uv run citewise content export <item id>           # Markdown with source footnotes + FAQPage JSON-LD + why-note
uv run citewise content measure <item id>          # baseline vs the newest scan since approval
uv run citewise content resume <item id>           # continue after an error or the spend cap
```

- **Opportunities** are scored as impact × confidence, in one unit ("answers affected"):
  - **gap:** a topic × intent slice where the brand's mention rate trails a rival's by ≥ 20 points;
  - **citation:** a domain the engines cite in answers that skip the brand. This is a get-listed / pitch suggestion; outreach is manual;
  - **fact fix:** a claim FactCheck marked contradicted.
- **Every factual sentence is checked against the KB automatically.** It must end with a `[kb:<chunk>]` marker to a retrieved passage. Every number in it must be in that passage, as must any ranking or superlative, and most of its words must come from it. An invented price fails, even with a real-looking marker.
- **Other guardrails:** competitor defamation, fake reviews and testimonials, text copied from an injected instruction, and max length or keyword stuffing. A page that still trips a guardrail after 2 revisions has the offending sentences removed before anyone reviews it.
- **Scoring:** an LLM judge scores the rubric in `evals/rubrics/content.yaml` (factuality, coverage, tone, AEO structure, originality). Deterministic checks can only lower those scores. Each version is stored with its scores and the feedback that produced it.
- **Brand voice** (tone, banned words, reading level, max words) lives in `data/brands/<brand>/voice.md`. Prompt templates are versioned in `prompts/content/`.
- **Mock mode:** the writer and judge are simulated. The writer's first draft is thin and only fixes what the critique names, so rising scores show the loop working, not writing quality. For a real page, run `create` with `CITEWISE_MODE=live` (about $0.15–0.30 per page).

API:
- `GET /brands/{id}/content/opportunities`
- `GET /content?status=awaiting_approval` (the review queue)
- `GET /content/{id}`
- `POST /content/{id}/approve` and `POST /content/{id}/reject`, with body `{"comment": "..."}`. Reject returns 202; the revision runs in the background. Approve returns 409 if a guardrail still fails.
- `POST /content/{id}/resume` continues after an error or the spend cap.

### Live mode

Set `CITEWISE_MODE=live` and `ANTHROPIC_API_KEY` in `.env`. That one key runs everything Citewise does itself:
- **LLM calls on Claude** (`LLM_PROVIDER=anthropic`, the default): generation on `MODEL_GENERATE` (`claude-opus-5-5`), and labelling, extraction and judging on `MODEL_EXTRACT` / `MODEL_JUDGE` (`claude-haiku-5-5`).
- **Embeddings run locally** (`EMBED_PROVIDER=local`, `MODEL_EMBED=BAAI/bge-small-en-v1.5`), because Anthropic has no embeddings API. They are free, and the ~130 MB model downloads once to `~/.cache/citewise/fastembed`.

To use OpenAI instead, set `LLM_PROVIDER=openai` and OpenAI model IDs (`MODEL_EXTRACT=gpt-6-luna`, `MODEL_GENERATE=gpt-6.2-sol`), and/or `EMBED_PROVIDER=openai` with `MODEL_EMBED=text-embedding-3-small`. A live command refuses to start if a model doesn't belong to `LLM_PROVIDER` (for example, a `.env` left pinning `gpt-*` models).

Scanning an answer engine needs *that* engine's key (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `PERPLEXITY_API_KEY`), because measuring ChatGPT means asking ChatGPT. With only a Claude key, scan with `--engines anthropic`.

A live `panel generate` + `cluster` for one brand is about 15 generation calls, about 25 small classification and labelling calls, and 2 local embedding batches. Check `llm_calls` for the exact cost. If the database already holds a mock-mode panel, run `citewise panel reset <brand>` first, because `generate` skips cells that already have prompts. Each stored embedding records its embedder (`mock:`, `local:` or `openai:`), and stale ones are re-embedded automatically. **First live run with only a Claude key (end to end, well under $0.50):**

```powershell
$env:CITEWISE_MODE="live"   # or set CITEWISE_MODE=live in .env; needs ANTHROPIC_API_KEY
uv run python scripts/record_extract_fixtures.py --max-cost 0.10   # real Stage B on the golden set
uv run citewise scan run Stridelab --size 5 --engines anthropic --max-cost 1
uv run citewise extract run <run id printed above> --max-cost 0.10
uv run citewise eval extract --max-cost 0.10            # live extractor F1 on the 65-answer golden set
```

Live answers are never served from mock answers cached the same day (the cache key includes the mode). Engine model IDs are `ENGINE_MODEL_OPENAI`, `ENGINE_MODEL_ANTHROPIC`, `ENGINE_MODEL_GEMINI` (default `gemini-3.5-flash-lite`) and `ENGINE_MODEL_PERPLEXITY` (default `perplexity/sonar`; the old `sonar` ID is refused because Perplexity retired it). Model IDs are set per purpose (`MODEL_EXTRACT`, `MODEL_JUDGE`, `MODEL_GENERATE`, `MODEL_EMBED`) in `.env`, and the providers by `LLM_PROVIDER` / `EMBED_PROVIDER`.

## How it's built

- `src/citewise/providers/` is the only place that talks to LLMs. `registry.get_chat_provider()` / `get_embedder()` return mock or live backends wrapped by `tracking.py`, which writes **every** call (tokens, estimated $ cost, latency, errors) to the `llm_calls` table.
- Prices live in `src/citewise/providers/pricing.py`, with a "last checked" date.
- `citewise config` shows the active mode, models and which keys are set (never the values).

Useful commands: `uv run ruff check .`, `uv run ruff format .`, `uv run mypy`, `uv run pre-commit run --all-files`.

## Limitations
Citewise uses official APIs only (with web search where available). API answers approximate, but are not identical to, what consumer apps like chatgpt.com show.

The prompt panel has no search-volume data: every prompt counts equally. See docs/DECISIONS.md (Module 1) for how real volumes would plug in.

Visibility metrics are only as good as the extractor. Its mock-mode scores measure a simulator; the golden set is hand-reviewed, but the live score needs a run with a key.

## License
MIT
