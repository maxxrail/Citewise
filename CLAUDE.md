# CLAUDE.md — instructions for Claude Code

You are building **Citewise**, an open-source AI-search visibility platform (a small, honest version of what Profound does). The owner is Maxx. He wants **you** to do the engineering, one module at a time. Read this file, then `docs/PLAN.md`, then `PROGRESS.md`, before every session.

## How to work

1. Only build the module you were asked to build (`docs/modules/module-NN-*.md`). Don't start the next one.
2. Before writing code, read the module spec fully and restate the plan in 5–10 bullets. Then build.
3. Meet **every** item in the module's "Acceptance criteria" section. Run the commands listed there yourself. A module is not done until they pass.
4. Write tests as you go (pytest). Tests must pass in **mock mode** with no API keys and no network.
5. When the module is done:
   - Update `PROGRESS.md` (status, what was built, decisions made, known gaps).
   - Update `README.md` if the setup or usage changed.
   - Add an entry to `docs/DECISIONS.md` for any non-obvious design choice (what, why, alternative rejected).
   - Make one commit per logical step with clear messages (`module-02: add async engine runner with retries`). Do not push unless Maxx asks.
6. If a spec is ambiguous, pick the simplest option that satisfies the acceptance criteria, note it in `DECISIONS.md`, and keep going. Ask Maxx only for things you truly can't decide (spending money, deleting data, choosing between two very different products).

## Your helpers (in `.claude/`)

- Subagents: `api-docs-checker` (use **before** writing any provider adapter), `test-writer`, `eval-designer` (golden sets, rubrics, red-team), `code-reviewer` (run before finishing a module).
- Commands Maxx uses: `/next`, `/build-module NN`, `/review-module NN`, `/status`, `/fix-tests`, `/interview-prep`, `/ship-check`.
- A hook auto-formats `src/citewise/` and `tests/` with ruff after every edit. `CITEWISE_MODE=mock` is set by default in `.claude/settings.json`.
- You can't read `.env`. If you need a key's presence, check `settings` at runtime, never print secrets.

## Environment

- Maxx develops on **Windows** with VS Code and GitHub Desktop. Use cross-platform Python (`pathlib`, no bash-only scripts in the core workflow). Provide `make`-free commands; a `tasks.py` (invoke) or plain `python -m` commands are fine. If you add shell scripts, also add a PowerShell equivalent or a Python one.
- Python 3.11+, managed with **uv** (`uv sync`, `uv run pytest`).
- Postgres runs through `docker compose up -d db` (Docker Desktop). Tests use SQLite through the same SQLAlchemy models, so tests never need Docker.
- Secrets live in `.env` (never commit it). Keep `.env.example` up to date.

## Tech stack (don't swap without a DECISIONS.md entry)

| Layer | Choice |
|---|---|
| API | FastAPI + Pydantic v2 |
| DB | Postgres 16 (prod/dev), SQLite (tests), SQLAlchemy 2.0 + Alembic |
| Jobs | Simple async worker + APScheduler (no Celery/Redis unless a module needs it) |
| LLM / engines | Thin provider layer over official SDKs: OpenAI, Anthropic, Google Gemini, Perplexity (OpenAI-compatible) |
| Embeddings | OpenAI `text-embedding-3-small` (configurable); mock embedder for tests |
| ML | scikit-learn, numpy, umap-learn/hdbscan (optional), pandas |
| Frontend | Next.js (App Router) + TypeScript + Tailwind + Recharts, in `web/` |
| Tooling | uv, ruff, mypy (lenient), pytest, GitHub Actions CI |
| Deploy | Docker images; Railway or Fly.io (Module 8) |

## Non-negotiable rules

- **Mock mode first.** Every external call goes through `src/citewise/providers/`. `CITEWISE_MODE=mock` returns deterministic fixtures from `tests/fixtures/`. CI always runs in mock mode.
- **Model IDs live in config**, not in code. Model names change, so check current provider docs before hardcoding defaults and leave a comment with the date checked.
- **Track cost and latency** for every LLM call (tokens in/out, $ estimate, ms). Store them in the `llm_calls` table from Module 0 onward.
- **Structured outputs.** When an LLM must return data, use JSON schema / tool calling and validate with Pydantic. Retry once on validation failure, then record the failure; don't crash the run.
- **Idempotent jobs.** Re-running a job for the same (run_id, prompt_id, engine) must not duplicate rows.
- **Respect ToS and robots.** Use official APIs only. No scraping of chatgpt.com or other consumer UIs.
- Keep functions small, type-hinted, and documented where the *why* isn't obvious. Don't over-abstract.

## Repo layout (target)

Same structure as Maxx's frontdesk-agent repo. Keep it.

```
citewise/
  .claude/            commands, subagents, settings (don't delete)
  .github/workflows/  ci.yml (lint, tests, evals in mock mode), gitleaks.yml
  src/citewise/       the Python package (src layout)
    api/              FastAPI routers
    core/             config, logging, db session
    models/           SQLAlchemy models
    schemas/          Pydantic schemas
    providers/        LLM / engine / embedding clients + mock
    services/         business logic per module (panel, collect, extract, metrics, kb, factcheck, content, crawl)
    evals/            eval harness code (scorers, runners)
    workers/          scheduler + job runners
    cli.py            Typer entry point -> `citewise`
  prompts/            versioned prompt templates (panel/, extract/, factcheck/, content/), e.g. panel/generate.v1.jinja
  evals/              eval DATA and results
    datasets/         golden/ (hand-checked), human_labels.jsonl, redteam.jsonl
    rubrics/          YAML rubrics for LLM-judge
    baselines/        baseline.json for the CI regression gate
    reports/          committed eval + experiment reports (markdown)
  ops/                deploy + ops: railway/fly configs, runbooks, pricing snapshot, bot IP ranges
  migrations/         Alembic (alembic.ini at repo root)
  scripts/            one-off scripts (gen_logs.py, panel_report.py) - Python, not bash
  data/               sample brands (YAML), brand voice, knowledge-base docs, sample logs
  tests/              pytest; fixtures/ holds mock provider responses
  web/                Next.js dashboard (Module 8)
  docs/               PLAN, DECISIONS, INTERVIEW_PREP, modules/
  Dockerfile, docker-compose.yml, pyproject.toml, uv.lock, alembic.ini
  .env.example, .dockerignore, .gitattributes, .gitignore, .gitleaksignore, .pre-commit-config.yaml
  README.md, CHANGELOG.md, LICENSE, CLAUDE.md, PROGRESS.md, START_HERE.md
```

- Update `CHANGELOG.md` (Keep a Changelog format) at the end of each module under a `module-NN` heading.
- `pre-commit` runs ruff + gitleaks; install it in Module 0 (`uv run pre-commit install`).

## Definition of "ready to ship"

By the end of Module 8 a stranger can: clone the repo, `docker compose up`, open the dashboard, add a brand, run a visibility scan in mock mode (or with real keys), and see share-of-voice charts, citations, fact-check flags, content drafts, and eval scores. A live demo URL and a 2-minute video exist.
