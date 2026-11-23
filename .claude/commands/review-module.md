---
description: Critically review a finished Citewise module like a Profound interviewer would (usage: /review-module 03)
argument-hint: <module number>
---

Review module $ARGUMENTS as a senior AI/ML engineer at Profound reviewing a candidate's take-home.

1. Read the spec `docs/modules/module-$ARGUMENTS-*.md` and the code it produced.
2. Re-run every acceptance-criteria command and confirm the results. Don't trust PROGRESS.md.
3. Look for: correctness bugs, missing tests, non-idempotent jobs, unbounded costs, untracked LLM calls, hardcoded model IDs, prompt-injection risks, Windows-incompatible code, over-engineering.
4. Report findings ranked by severity, then fix everything rated high or medium, with tests, and commit as `module-$ARGUMENTS: review fixes`.
5. List 3 questions an interviewer would likely ask about this module, with a strong short answer for each. Append them to `docs/INTERVIEW_PREP.md`.
