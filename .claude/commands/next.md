---
description: Find the next unfinished Citewise module and build it
---

1. Read `PROGRESS.md` and find the lowest-numbered module that isn't ✅ or 🔁.
2. If the previous module is ✅ but not 🔁 (reviewed), run the `/review-module` steps on it first.
3. Then follow every step in `.claude/commands/build-module.md` for the next module, using its two-digit number.
4. When done, use the `code-reviewer` subagent on the new module and fix any High findings before finishing.
