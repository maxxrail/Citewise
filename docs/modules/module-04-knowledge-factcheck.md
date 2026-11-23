# Module 4 — Brand knowledge base and FactCheck

## Why Profound cares
Profound's FactCheck traces wrong AI claims back to their sources, and its Context Manager stores brand truth (company, audience, competitors, offerings, strategy). The posting asks for **retrieval for evidence-grounded generation** and **factuality evals**.

## Goal
Keep a brand "source of truth", check what engines claim about the brand against it, and point to the web source that probably caused each wrong claim.

## Build
1. **Knowledge base** (`src/citewise/services/kb/`)
   - Ingest Markdown/TXT/PDF/URL files (URL fetch in live mode only) into `kb_docs` → chunk (~400 tokens, overlap) → embed → `kb_chunks`.
   - Context categories, as a field on each doc: `company`, `offerings` (pricing, features, plans), `audience`, `competitors`, `policies`.
   - Sample KB per seeded brand in `data/kb/<brand>/` with deliberately specific facts (prices, plan names, founding year, HQ, integrations).
   - Retrieval: hybrid = BM25 (rank_bm25) + cosine on embeddings, merged with reciprocal rank fusion; optional LLM re-rank of top 20 → top 5.
2. **Claim extraction**: from answers that mention the brand, extract atomic factual claims about it (structured output: `claim`, `type` in {pricing, feature, company, comparison, other}, `answer_span`).
3. **Verification**: for each claim, retrieve evidence → LLM judge returns `supported | contradicted | not_in_kb` + evidence chunk IDs + short rationale. Never mark "contradicted" without quoting the KB chunk.
4. **Source tracing**: for contradicted claims, rank the answer's citations by how likely they caused the claim (fetch-free version: compare claim to citation title/snippet when available; live version: fetch page text and score similarity). Store `source_citation_id` + confidence.
5. **Sentiment narrative**: per engine, summarize the 3 most common positive and negative themes about the brand (cluster claim/snippet embeddings, label clusters).
6. **Retrieval eval**: `evals/datasets/golden/kb_qa.jsonl` (~30 questions → gold chunk IDs). Report recall@5 and MRR for BM25 vs dense vs hybrid vs hybrid+rerank.
7. **API/CLI**: `citewise kb ingest <brand> <path>`, `citewise factcheck run <run_id>`, `GET /brands/{id}/factcheck`.

## Acceptance criteria
- [ ] Mock fixtures include at least 5 planted wrong claims (e.g. wrong price, discontinued plan); `citewise factcheck run` flags ≥ 4 as contradicted with evidence.
- [ ] Retrieval eval table printed; hybrid beats BM25-only and dense-only on recall@5 in fixtures (or DECISIONS.md explains why not).
- [ ] Each contradicted claim shows: claim, KB evidence quote, engine, likely source URL.
- [ ] Tests for chunker boundaries, RRF fusion math, verdict schema validation.

## Out of scope
Auto-correcting the web (Module 5 drafts corrections).
