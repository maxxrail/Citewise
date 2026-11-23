---
description: Pre-launch checklist before sharing Citewise with Profound
---

Check that Citewise is ready to put in front of a Profound hiring manager. Report ✅/❌ for each item and fix what you can:

1. Fresh clone test: in a temp folder, clone the repo, `uv sync`, `uv run pytest`, `docker compose up -d` and confirm the dashboard loads demo data with no `.env`.
2. README: pitch, GIF/screenshot, architecture diagram, eval results table with real numbers, "How this maps to Profound", quick start, cost per scan, limitations, roadmap.
3. No secrets in git history (`git log -p | grep -iE "sk-|api_key=|AIza"` finds nothing).
4. CI badge green; eval regression gate is active.
5. `docs/WRITEUP.md` and `docs/VIDEO_SCRIPT.md` exist and use real numbers from a live scan.
6. Live demo URL in README works and runs in mock/demo mode.
7. Code hygiene: no TODOs left in core paths, no dead modules, ruff clean, type hints on public functions.
8. List the 3 most impressive things to point at in an application email, with file paths.
