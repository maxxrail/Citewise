---
description: Get the test suite and linter green again without weakening tests
---

1. Run `uv run pytest -q` and `uv run ruff check .`.
2. For each failure, find the root cause. Fix the code, not the test, unless the test is genuinely wrong. If you change a test, explain why in the commit message.
3. Never delete a test, skip it, or loosen an assertion just to pass.
4. Re-run until green, then commit as `fix: <summary>`.
