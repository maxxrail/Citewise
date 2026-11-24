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
- All 65 rows were reviewed by Maxx on 2026-11-24 (`reviewed: true`) and kept as drafted.

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

## kb_qa.jsonl (Module 4, KB retrieval)

**What it measures.** Whether KB retrieval (BM25, dense, hybrid RRF, hybrid + re-rank) puts the chunk that answers a
question, or settles a claim, in the top 5 of that brand's chunks. Report recall@5 and MRR per retriever, overall and
split by `style`. Retrieval is always scoped to one brand.

**Format.** One JSON object per line with these fields: `id, brand, query, gold_chunk_keys[], style, difficulty, note, reviewed`.
`gold_chunk_keys` are `<doc slug>#<chunk index>` keys of the indexed chunks (for example `pricing#0` = first chunk of
`data/kb/linear/pricing.md`). A row lists every chunk that contains the answer. Usually that is one chunk; 7 rows list two,
where the fact is split across docs or repeated (for example Linear SAML in `security#2` and `pricing#0`). For recall@5,
count a row as a hit if **any** gold key is in the top 5. MRR uses the best-ranked gold key.

**Styles (12 rows per brand, 4 of each style).**
- `lexical`: reuses the KB's own terms ("How much is the Linear Business plan per user?").
- `paraphrase`: the same need in other words, avoiding the KB's distinctive terms ("let outside contractors in" for
  guest accounts, "front of the foot" for forefoot). This is where BM25 and dense retrieval should differ.
- `claim`: an engine-style statement to fact-check. The note gives the expected verdict. 3 are true, 8 false (one partly true: `kbqa-stridelab-10`) and 1 is
  `not_in_kb` (`kbqa-linear-12`, the funding round). The gold chunk is the one a judge needs to see to reach the verdict.

**How it was built.** Claude Code wrote the rows from the KB Markdown in `data/kb/` and a dump of the indexed chunks
(brand, key, text). It did not read or run `src/citewise/services/kb/`, so the labels are not tuned to the retriever.
A script checked that every line parses, that ids are unique and that every gold key exists for that brand. All rows are
`reviewed: false` until Maxx checks them.

**Known biases and limits.**
- The KBs are tiny (10, 11 and 19 chunks), so recall@5 is easy to reach: with 10 chunks a random top 5 gets about 50%.
  MRR and the paraphrase split are the more useful signals.
- Coverage: every Stridelab and Taskflow chunk is gold at least once. 7 Linear chunks are never gold (`audience#0, billing#0,
  company#0, company#2, pricing#1, pricing#2, pricing#3`). Several of them are deliberate distractors in notes.
- Claims lean false (8 of 12) because the fact-checker's job is mostly catching wrong claims. Do not read verdict
  balance into retrieval numbers.
- Gold keys depend on the current chunking. If the chunker or a KB doc changes, re-check the keys. The build script validates
  that keys exist, not that they still hold the answer.
- One labeller wrote both the queries and the labels, so paraphrases may still carry some lexical overlap.

## factcheck_claims.jsonl (Module 4, FactCheck verifier, held out)

**What it measures.** Whether the verifier gives the right verdict (`supported | contradicted | not_in_kb`) for an
engine-style claim about a brand, judged against that brand's KB in `data/kb/<brand>/`. Report accuracy and a
3x3 confusion matrix, overall and per brand. Two error types matter most: `not_in_kb` rows marked
`contradicted`, which are false accusations, and `contradicted` rows marked `supported`, which are missed errors.
For `contradicted` predictions, also check that the quoted evidence contains `evidence_hint`.

**Format.** One JSON object per line with these fields: `id, brand, claim, expected_verdict, evidence_hint, kind, difficulty, note, reviewed`.
`evidence_hint` is a short phrase copied verbatim from the brand's KB that settles the claim. It is `null` for `not_in_kb`.
`kind` is a free-text tag for the claim's shape, for example `wrong_year`, `plan_mismatch`, `number_misattributed`,
`true_negation`, `paraphrase_true`, `high_overlap_unsettled` or `related_unsettled`.

**How it was built (45 rows, 15 per brand: 6 contradicted, 5 supported, 4 not_in_kb).** Claude Code wrote the claims
from the KB Markdown and the Module 4 spec only. It did not read the mock verifier, the factcheck services or prompts,
or the planted claims in the engine fixtures, so the rows are held out from the rules they score. The rows also avoid
the 12 `claim`-style rows in `kb_qa.jsonl`. A script checked that every line parses, that the ids are unique, and that every
`evidence_hint` is a verbatim substring of that brand's KB docs. Every row is `reviewed: false` until Maxx checks it.

**Labelling conventions.**
- `contradicted` means the KB states something incompatible with the claim. This includes plan structure ("Business adds
  SAML", so a claim that Pro has SAML is contradicted) and closed lists ("ships to the United States and Canada only").
- `not_in_kb` means the KB is silent. Silence plus related facts is still `not_in_kb`. For example, "distributed across
  North America and Europe" does not settle a headquarters city, and listing integrations does not settle which plan they are on.
- A claim with one true part and one false part is `contradicted`.
- Derived facts (a $3 price difference, "highest drop of the lineup") are `supported` when they follow from KB numbers.

**Known biases and limits.**
- One person wrote both the claims and the labels, and the claims stick close to the KB. Real engine errors are messier.
- The KBs are tiny, so retrieval rarely fails. This set mostly tests the judge, not the retriever.
- The 13 `hard` rows (misattributed numbers, plan mismatches, high-overlap `not_in_kb`) are the useful signal. Report them separately.
- Two Taskflow and Linear `not_in_kb` rows (`fc-taskflow-15`, `fc-linear-15`) resemble facts from the *other* brand's KB.
  They only test cross-brand leaks if the verifier is ever given evidence from more than one brand.
- The `contradicted`/`not_in_kb` line is a judgement call on a few rows. See the notes on `fc-taskflow-02`, `fc-linear-12` and `fc-taskflow-13`.
