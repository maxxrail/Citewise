# Citewise — project plan

## One-line pitch

An open-source platform that measures how AI answer engines (ChatGPT, Claude, Gemini, Perplexity) talk about a brand, explains why, and drafts grounded content to improve it, with an eval framework that proves the outputs are good.

## Why this project, for Profound

Profound sells "AI search visibility": Answer Engine Insights, Prompt Volumes, Agent Analytics, FactCheck, Sentiment, and an AI Marketer that drafts content. Their AI/ML Engineer posting asks for:

| Profound AI/ML Engineer requirement | Where Citewise shows it |
|---|---|
| NLP at scale: classification, ranking, clustering, topic extraction, summarization | M1 prompt clustering + intent classification, M3 mention/rank/sentiment extraction |
| End-to-end LLM content workflows: topic discovery → brief → draft → revision → publish-ready | M5 content agent |
| Prompt/template libraries, brand voice, retrieval for evidence-grounded generation with citations | M4 brand knowledge base + RAG, M5 |
| Eval frameworks: factuality, coverage, tone, safety, originality; rubric LLM evals, human review, red-teaming | M6 eval harness (plus extractor evals in M3) |
| Instrument content performance (AEO visibility), run experiments on quality/cost/latency | M2 cost tracking, M3 metrics, M6 experiments |
| Turn large text datasets into features and product signals | M3 metrics, M7 crawler analytics |
| Data pipelines, data quality, observability | M0 `llm_calls`, M2 runner, M6 tracing |
| SQL + strong Python | Throughout; M3 metrics are SQL views |

## Modules

Each module is a standalone spec in `docs/modules/`. Build them in order. Each one ends with something demoable.

| # | Module | Demo at the end | Est. Claude Code sessions |
|---|---|---|---|
| 0 | Foundations | `docker compose up`, `/health`, CI green, mock provider | 1 |
| 1 | Prompt panel and topic discovery | Given a brand + category, generate ~200 realistic prompts, cluster into topics, label intent | 1–2 |
| 2 | Multi-engine answer collection | Run the panel across 4 engines, store answers + citations + cost | 1–2 |
| 3 | Extraction and visibility metrics | Share of voice, avg rank, sentiment, top cited domains; extractor scored against a golden set | 2 |
| 4 | Brand knowledge base and FactCheck | Flags wrong claims engines make about the brand and traces them to the cited source | 1–2 |
| 5 | Content agent | Finds gap prompts, writes brief → draft → critique → revision, with citations, into an approval queue | 2 |
| 6 | Evals, experiments and observability | `citewise eval run` scores extractors + content on rubrics; prompt-version A/B report; CI regression gate | 1–2 |
| 7 | AI crawler analytics | Upload server logs → which AI bots crawl which pages, verified vs spoofed, crawl vs citation overlap | 1 |
| 8 | Dashboard and ship | Next.js dashboard, auth, deploy, demo data, README, write-up, video | 2–3 |

## Data model (grows per module)

- `brands` (name, domain, category, aliases[], competitors[])
- `prompts` (brand_id, text, intent, topic_id, source, embedding)
- `topics` (brand_id, label, description, size)
- `runs` (brand_id, started_at, engines[], status, cost_usd)
- `answers` (run_id, prompt_id, engine, model, text, raw_json, latency_ms)
- `citations` (answer_id, url, domain, title, position)
- `mentions` (answer_id, entity, is_brand, is_competitor, rank, sentiment, snippet)
- `kb_docs` / `kb_chunks` (brand knowledge base for RAG)
- `claims` (answer_id, claim_text, verdict, evidence_chunk_id, source_citation_id)
- `content_items` (brand_id, target_prompt_ids, brief, draft, versions, scores, status)
- `eval_runs` / `eval_results`
- `crawl_events` (brand_id, ts, bot, verified, path, status)
- `llm_calls` (purpose, provider, model, tokens_in, tokens_out, cost_usd, latency_ms, ok, error)

## Cost guardrails

- Mock mode by default. Real runs need `CITEWISE_MODE=live` and a `--max-cost` flag.
- Default demo scan: 50 prompts × 4 engines ≈ 200 calls. Aim for under ~$3 per full scan. Use small/cheap models for extraction and judging.
- Cache by (engine, model, prompt hash, date) so re-runs on the same day are free.

## What to say in the application

"I built a small version of Profound's core loop end-to-end: prompt discovery, multi-engine collection, NLP extraction into visibility metrics, fact-checking with RAG, a content agent, and an eval harness that gates CI. Here's the live demo, here are the eval numbers, and here's what I'd do next with your 2B-prompt dataset."
