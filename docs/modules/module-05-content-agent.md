# Module 5 — Content agent ("AI Marketer" lite)

## Why Profound cares
Profound's AI Marketer finds the marketing work, drafts it, and asks for approval. The posting asks for **end-to-end LLM content workflows: topic discovery → briefs → drafts → revision loops → publish-ready assets**, with brand voice and citations.

## Goal
An agent that turns visibility gaps and fact errors into reviewed, grounded, publish-ready content, with a human approval step.

## Build
1. **Opportunity finder** (`src/citewise/services/content/opportunities.py`)
   - Gap prompts: topics/intents where the brand's mention rate is low and competitors' is high.
   - Citation gaps: domains engines cite for this topic where the brand is absent (e.g. listicles, review sites, Reddit). Output "get listed / pitch" suggestions; don't automate outreach.
   - Fact fixes: contradicted claims from Module 4 → "correction page / FAQ update" opportunities.
   - Score = impact (prompt count × competitor gap) × confidence; return the ranked list.
2. **Pipeline** as an explicit state machine (`src/citewise/services/content/pipeline.py`), not one mega-prompt:
   `opportunity → brief → outline → draft → critique → revise (≤2 loops) → final → awaiting_approval → approved|rejected`
   - **Brief**: target prompts, audience, angle, must-cover facts (retrieved from KB), competitor comparison points, format (FAQ, comparison page, guide), word count.
   - **Draft**: grounded only in KB chunks + allowed sources; inline citation markers `[kb:chunk_id]`.
   - **Critique**: rubric (from Module 6 rubrics if built, else a local copy) on factuality, coverage of target prompts, tone/brand voice, AEO structure (direct answer in first 2 sentences, headings matching questions, FAQ schema, tables), originality.
   - **Revise** only failing criteria. Stop early if all pass.
   - Store every version + scores in `content_items` / `content_versions`.
3. **Brand voice**: `data/brands/<brand>/voice.md` (tone, banned words, reading level). Templates versioned in `prompts/content/`.
4. **Outputs**: Markdown + suggested JSON-LD (`FAQPage` / `Product`) + a "why this should help" note linking to the gap metrics.
5. **Guardrails**: no unsupported claims (every factual sentence must map to a KB chunk, so check it automatically), no competitor defamation, no fake reviews/testimonials, max length.
6. **Human-in-the-loop**: `GET /content?status=awaiting_approval`, `POST /content/{id}/approve|reject` with comments; rejection comments get fed back as a revision.
7. **Measurement hook**: approved items record target prompts so later scans can measure before/after mention rate (document the method; real lift needs publishing time).

## Acceptance criteria
- [ ] `citewise content opportunities <brand>` prints a ranked list with reasons.
- [ ] `citewise content create <opportunity_id>` runs the full loop in mock mode and saves ≥2 versions with rising rubric scores.
- [ ] Unsupported-claim checker test: a draft with an invented price gets caught.
- [ ] Final output includes valid JSON-LD (validate with a schema test).
- [ ] Reject-with-comment triggers a revision that addresses the comment (test with mock).

## Out of scope
Auto-publishing to a CMS (mention as future work: staging to a CMS draft through an API).
