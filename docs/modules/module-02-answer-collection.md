# Module 2 — Multi-engine answer collection

## Why Profound cares
This is the data engine behind "Answer Engine Insights": ask many engines many prompts, reliably, cheaply, repeatedly, and keep everything (text + citations) for analysis.

## Goal
A robust async runner that sends the panel to 4 answer engines with web search turned on, stores answers and citations, and tracks cost and latency.

## Build
1. **Engine adapters** (`src/citewise/providers/engines/`), each returning a normalized `EngineAnswer(text, citations[{url,title,position}], model, usage, latency_ms, raw)`:
   - `openai` – Responses API with the web search tool
   - `anthropic` – Messages API with the web search tool
   - `gemini` – generateContent with Google Search grounding (citations from grounding metadata)
   - `perplexity` – Sonar API (OpenAI-compatible; citations/search_results field)
   - Check each provider's **current** docs for parameter names before writing the adapter; note the date checked in a comment. Keep model IDs in config.
   - Mock adapters with realistic fixture answers, including citations, for both seeded brands.
2. **Runner** (`src/citewise/services/collect/runner.py`)
   - `asyncio` + per-engine semaphores (configurable concurrency) + token-bucket rate limiting.
   - Retries with exponential backoff + jitter on 429/5xx/timeouts; give up after N and record the error.
   - Idempotency key `(run_id, prompt_id, engine)`; resumable: `citewise scan resume <run_id>`.
   - Daily cache keyed by `(engine, model, sha256(prompt), date)`.
   - **Budget guard**: estimate cost before the run; abort if over `--max-cost`; stop mid-run if actual spend crosses it.
   - Persona/locale option: optional system message, e.g. "user in Toronto, Canada" (stored on the run).
3. **Storage**: `runs`, `answers`, `citations` tables + migration. Normalize citation URLs (strip UTM, lowercase host, drop fragments) and store `domain`.
4. **Scheduling**: `src/citewise/workers/scheduler.py` with APScheduler: `citewise schedule add <brand> --cron "0 9 * * 1"`. Weekly snapshots are what make trends possible.
5. **CLI/API**: `citewise scan run <brand> --engines openai,anthropic,gemini,perplexity --size 50 --max-cost 3`; `POST /brands/{id}/runs`, `GET /runs/{id}` (status, progress, cost).
6. **Progress**: rich progress bar in CLI; `runs.progress` updated for the API.

## Acceptance criteria
- [ ] Mock run of 50 prompts × 4 engines completes in < 10 s, writes 200 answers and their citations.
- [ ] Killing a run halfway and running `citewise scan resume` finishes without duplicates (test).
- [ ] Retry logic tested with a fake adapter that fails twice then succeeds.
- [ ] Budget guard test: run aborts before start when estimate > max-cost.
- [ ] URL normalization unit tests (UTM, trailing slash, www, fragments).
- [ ] With real keys, `citewise scan run <brand> --size 5 --engines openai` works end-to-end (Maxx runs this manually; document the command in README).

## Out of scope
Parsing the answers (Module 3).
