---
description: Quiz me on Citewise like a Profound interviewer, using the actual code
argument-hint: [optional module number]
---

Act as a Profound AI/ML engineer interviewing me about Citewise (focus: module $ARGUMENTS if given, otherwise the whole project).

1. Read the relevant code, `docs/DECISIONS.md`, eval reports in `reports/`, and `docs/INTERVIEW_PREP.md`.
2. Ask me one question at a time (mix of design trade-offs, metrics/eval rigor, scaling to Profound's 2B+ prompts, failure modes, and "what would you do next"). Wait for my answer.
3. After each answer, give blunt feedback and a stronger version of the answer, grounded in what the code actually does.
4. After 6 questions, summarize my weak spots and append new Q&A to `docs/INTERVIEW_PREP.md`.
