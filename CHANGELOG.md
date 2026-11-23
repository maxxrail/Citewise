# Changelog

All notable changes to Citewise are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). One section per module.

## [Unreleased]

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
