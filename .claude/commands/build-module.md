---
description: Build one Citewise module end-to-end from its spec (usage: /build-module 03)
argument-hint: <module number, e.g. 00>
---

Build module $ARGUMENTS of Citewise.

1. Read `CLAUDE.md`, `docs/PLAN.md`, `PROGRESS.md`, and `docs/DECISIONS.md`.
2. Open the spec at `docs/modules/module-$ARGUMENTS-*.md` and read it fully.
3. Check that every earlier module is marked done in `PROGRESS.md`. If one isn't, stop and tell me which one.
4. Write a short plan (5–10 bullets) covering files, tables/migrations, tests. Then build it without waiting for approval.
5. Work in small steps: implement → write tests → run `uv run pytest` and `uv run ruff check .` → fix → commit with message prefix `module-$ARGUMENTS:`.
6. If this module needs golden data, rubrics or red-team cases, delegate drafting them to the `eval-designer` subagent. Before finishing, run the `code-reviewer` subagent on the module's changes and fix every High/Medium finding.
7. Go through every acceptance criterion in the spec, run its command, and show the output. If one can't pass (e.g. needs real API keys), say exactly what I must run myself.
8. Update `PROGRESS.md`, `CHANGELOG.md`, `README.md` and `docs/DECISIONS.md`.
9. Finish with: what was built, acceptance checklist with ✅/❌, the commands I should run to see it work, and anything I need to do (keys, labeling, manual checks).
