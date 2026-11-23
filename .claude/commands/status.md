---
description: Show where the Citewise build stands and what to do next
---

1. Read `PROGRESS.md` and `git log --oneline -15`.
2. Run `uv run pytest -q` and `uv run ruff check .` and report pass/fail counts.
3. Reply with: modules done, current module and its unmet acceptance criteria, failing tests (if any), the exact next command I should type (e.g. `/build-module 04`), and any manual task waiting on me (API keys, labeling, deploy).
Keep it under 15 lines.
