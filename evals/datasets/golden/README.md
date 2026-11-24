# Golden datasets

## mentions.jsonl (Module 3, mention extraction)

**What it measures.** Whether `citewise eval extract` finds every company an engine answer names, and gets
each one's canonical name, entity type (brand / competitor / unknown), list rank, sentiment (number and label)
and `is_recommended` right. Labels follow `evals/rubrics/mentions.yaml` (the contract). Read that first.

**Format.** One JSON object per line with these fields: `id, brand, engine, style, prompt, answer, mentions[], difficulty,
tags[], note, reviewed`. Each mention has these fields: `canonical, entity_type, rank, sentiment, sentiment_label,
is_recommended`. `entity_type` depends on the row's `brand`: a shoe brand in a Taskflow row is `unknown`.

**How it was built (65 rows).**
- 48 rows (`stridelab-*`, `taskflow-*`) are simulated engine answers from `scripts/golden_mention_candidates.py`
  (`--per-cell 6 --seed 3`; the unlabelled candidates file isn't kept). Claude Code labelled them by reading the answer text only. It did not read or run the extractor, the mock
  extractor or the Stage B prompt, so the labels stay independent of the code they score.
- 14 rows (`hard-*`, `engine: handwritten`) are hand-written prose that target the failure modes: "On", "Monday" and
  "notion" used as ordinary words, possessives, legal suffixes, casing, typos, publishers named as sources,
  discouraged brands, unknown brands, prose rankings, a list whose first item names no company, and products named without their maker.
- 3 rows (`inject-*`) are the golden versions of red-team cases `m3-rt-001..003` in `../redteam.jsonl`. They are labelled with
  what the answer really says, not what the injected text asks for.
- All rows have `reviewed: false` until Maxx checks them.

**Labelling conventions (my reading of the rubric; edit them here if you change them).**
- Comparison answers ("Both X and Y are solid choices" + bullets): the bullets count as a list, so they get ranks. Only the verdict
  winner ("reviewers lean towards X") is recommended. Winner 0.9, other 0.7. If an item also has an "although ..."
  criticism, it is mixed: 0.3 for the winner, 0.1 otherwise.
- "Ways to save" and "best options" lists: every item is recommended. One praise phrase = 0.7. Praise with an "although ..."
  criticism = mixed 0.1.
- Single-brand answers: the subject is recommended when the overall verdict is positive. Two distinct praise statements = 0.9.
  A soft caveat ("smaller brand, so sizing feedback is limited", "integration library is smaller") = mixed 0.3.
  "People also consider X, Y" = neutral 0, not recommended, rank null.
- Prose rankings ("my top pick is X, then Y"): every rank is null and only the top pick is recommended.
- Rank is the list position. It is never the `[n]` citation number (the simulator repeats numbers such as `[2]`).
- Plain facts (trial lengths, "desktop-only") are neutral. Checking whether they are true is Module 4's job.
- A brand that appears only in the prompt, a URL, an injected instruction or schema-like JSON is not a mention.

**Known biases and limits.**
- The candidate answers come from a few templates, so phrasing repeats a lot (`product_name` is on 53/65 rows).
  The 17 hand-written rows are the real stress test, so look at their scores separately.
- Sentiment leans positive (106 positive, 42 mixed, 23 neutral, 5 negative mentions). The template answers never criticise a brand outright,
  so negative-sentiment metrics rest on about 5 mentions in hard rows.
- 4 candidate rows are off-topic (shoe answers to Taskflow prompts). They test `unknown` typing, but real engines rarely do this.
  The cause was a simulator bug (no category keyword -> first category), fixed after labelling; the rows are kept because
  their stored answer text and labels are still valid test cases. Regenerating candidates now gives different answers.
- Sentiment numbers are one labeller's calibration. Score the labels by accuracy and the numbers by MAE, not exact match.
- `hard-007` (products without maker names) needs product-to-maker knowledge that Stage A alias matching cannot have.
  If it drags F1 down, report it separately rather than relabelling it.
- `taskflow-034/038/039/042` and others where the brand is absent (31 rows) test precision on the brand. They do not test mention rate.
