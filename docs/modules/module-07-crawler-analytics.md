# Module 7 — AI crawler analytics ("Agent Analytics" lite)

## Why Profound cares
Profound's Agent Analytics shows how AI bots crawl, cite and refer to a site. It's a classic **large-log data pipeline** problem with a security angle (spoofed bots).

## Goal
Ingest web server logs, find AI crawler and AI-referral traffic, verify the bots, and connect crawled pages to citations seen in Module 2.

## Build
1. **Ingest**: parsers for Nginx/Apache combined log format, plus a CSV format for CDN exports (Cloudflare/Vercel-like columns). Streaming parse (generators), works on 1M+ lines without loading everything into memory. `citewise crawl ingest <brand> access.log`.
2. **Bot classification**: a maintained table `data/bots.yaml` of AI user agents (OpenAI GPTBot / OAI-SearchBot / ChatGPT-User, Anthropic ClaudeBot / Claude-User / Claude-SearchBot, PerplexityBot / Perplexity-User, Google-Extended, Applebot-Extended, Bytespider, CCBot, Meta-ExternalAgent, etc.). Check the vendors' published docs for the current list and note the date. Split into categories: **training crawler**, **search indexer**, **user-triggered fetch**.
3. **Verification**: where the vendor publishes IP ranges, check that the source IP falls inside them (ranges loaded from a local JSON snapshot; live refresh optional). Label `verified | unverified | spoofed`.
4. **AI referrals**: detect human visits referred by AI products (referrer hosts like chatgpt.com, perplexity.ai, gemini.google.com, claude.ai; `utm_source=chatgpt.com`).
5. **Metrics**: crawl volume by bot/day, top crawled paths, 4xx/5xx served to bots, pages crawled but never cited vs cited but rarely crawled (join to `citations` on the brand's domain), AI referral sessions by landing page.
6. **Synthetic data generator**: `scripts/gen_logs.py` writes a realistic 500k-line log with human traffic, real-looking AI bots, spoofed bots and AI referrals, so the demo works without a real site.
7. **robots.txt helper**: given a robots.txt, report which AI bots are allowed/blocked per path.

## Acceptance criteria
- [ ] 500k-line synthetic log ingests in < 60 s on a laptop; memory stays bounded (document how you measured).
- [ ] Spoofed GPTBot lines (wrong IP) are labeled `spoofed` (test).
- [ ] `citewise crawl report <brand>` prints the metrics above.
- [ ] robots.txt parser tests for Allow/Disallow precedence and wildcards.

## Out of scope
Real-time streaming from a CDN (describe how you'd do it).
