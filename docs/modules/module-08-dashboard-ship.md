# Module 8 — Dashboard and ship

## Goal
Make it something a Profound engineer can click through in 2 minutes, and something a stranger can deploy.

## Build
1. **Next.js dashboard** (`web/`, App Router, TypeScript, Tailwind, Recharts; typed API client generated from FastAPI's OpenAPI with `openapi-typescript`)
   - **Overview**: Visibility Score, mention rate, SoV, sentiment; trend over runs; engine comparison bars.
   - **Prompts and topics**: topic table with per-topic mention rate; drill into a prompt → see each engine's answer with the brand highlighted and citations listed.
   - **Competitors**: SoV leaderboard incl. emerging competitors.
   - **Citations**: top cited domains, own-domain share.
   - **FactCheck**: contradicted claims with evidence and likely source.
   - **Content**: opportunities list, approval queue with version diff and rubric scores, approve/reject.
   - **Crawlers**: bot traffic charts, verified vs spoofed, crawled vs cited.
   - **Ops**: LLM cost/latency by purpose; latest eval scores.
   - Clean, calm design (white/neutral, one accent color); use the frontend-design guidance. Loading and empty states everywhere.
2. **Auth**: simple email+password or magic link for one workspace (or a demo mode with a read-only seeded workspace). Keep it small.
3. **Demo data**: `citewise demo seed` creates 2 brands × 4 weekly runs with fixture data so charts show trends.
4. **Deploy**
   - Dockerfiles for `api` and `web`; `docker compose up` runs everything locally (db, api, worker, web).
   - Deploy to Railway or Fly.io (api + worker + Postgres) and Vercel (web). Write `docs/DEPLOY.md` with exact steps Maxx can follow; Claude Code prepares the config files.
   - The public demo runs in **mock/demo mode** so it never spends money.
5. **Polish for hiring**
   - `README.md`: problem, 30-second GIF, architecture diagram (Mermaid), eval results table, "how this maps to Profound" section, quick start, cost per scan, limitations, roadmap.
   - `docs/WRITEUP.md`: ~1,200-word technical blog post ("What I learned measuring brand visibility across 4 AI engines"), with real numbers from one live scan Maxx runs.
   - `docs/VIDEO_SCRIPT.md`: 2-minute demo walkthrough script.
   - Badge row: CI, Python version, license (MIT).

## Acceptance criteria
- [ ] `docker compose up` → dashboard at `http://localhost:3000` shows demo data on every page with no keys.
- [ ] Lighthouse accessibility ≥ 90 on the overview page.
- [ ] Playwright smoke test: login/demo → overview → prompt drill-down → approve a content item.
- [ ] `docs/DEPLOY.md` exists and has been followed once (Maxx confirms live URL in PROGRESS.md).
- [ ] README has an eval results table filled from `citewise eval run`.
