# FactCheck verifier eval (held-out claims)

Golden set: `evals/datasets/golden/factcheck_claims.jsonl`: 45 claims, 0 hand-reviewed. Mode: **mock**. Generated 2026-11-24 by `citewise eval factcheck`.

> Mock mode: the verifier is the simulated reader (providers/mock_factcheck.py), unless recorded real responses exist for these prompts. These claims were written without seeing its rules, so this is the honest offline measure of it.

- **Accuracy:** 37.8%
- **False-contradiction rate** (not-wrong claims called contradicted): 29.6%
- **Evidence hit rate** (correct verdicts whose quote holds the settling phrase): 88.9%

| Verdict | Precision | Recall |
|---|---|---|
| supported | 21.1% | 26.7% |
| contradicted | 38.5% | 27.8% |
| not_in_kb | 61.5% | 66.7% |

## Confusion (rows: expected, columns: predicted)

| expected \ predicted | supported | contradicted | not_in_kb |
|---|---|---|---|
| supported | 4 | 7 | 4 |
| contradicted | 12 | 5 | 1 |
| not_in_kb | 3 | 1 | 8 |

## Errors

- fc-stridelab-01: expected contradicted, got supported (wrong_year): Stridelab was founded in 2019 by two former track coaches.
- fc-stridelab-02: expected contradicted, got not_in_kb (wrong_city): Stridelab got its start in Seattle, Washington.
- fc-stridelab-03: expected contradicted, got supported (number_misattributed): Stridelab backs its shoes with a 60-day warranty against manufacturing defects.
- fc-stridelab-07: expected supported, got contradicted (true_negation): Stridelab's warranty doesn't cover ordinary wear on the outsole.
- fc-stridelab-08: expected supported, got contradicted (paraphrase_true): If you're unsure about the fit, you can run in a pair for roughly two months and still send them back at no cost if you live in the US.
- fc-stridelab-09: expected supported, got contradicted (true_only): Stridelab only ships to customers in the United States and Canada.
- fc-stridelab-15: expected not_in_kb, got supported (high_overlap_unsettled): Stridelab founders Maya Chen and Luis Ortega were track coaches at the University of Oregon.
- fc-taskflow-01: expected contradicted, got supported (wrong_year): Taskflow was founded in 2017 by Priya Raman and Owen Brooks.
- fc-taskflow-02: expected contradicted, got supported (plan_mismatch): Taskflow's Pro plan comes with SAML single sign-on.
- fc-taskflow-03: expected contradicted, got supported (number_misattributed): New Taskflow workspaces get a 30-day free trial.
- fc-taskflow-05: expected contradicted, got supported (number_misattributed): Taskflow's Free plan is capped at 3 projects.
- fc-taskflow-06: expected contradicted, got supported (plan_mismatch): Taskflow will refund a monthly subscription in full if you cancel within 30 days.
- fc-taskflow-08: expected supported, got contradicted (paraphrase_true): Teams leaving Trello can bring their boards into Taskflow without losing tasks or comments.
- fc-taskflow-09: expected supported, got contradicted (comparison_true): Paying for Taskflow Pro month to month costs $3 more per user than paying annually.
- fc-taskflow-10: expected supported, got not_in_kb (true_negation): Taskflow won't refund monthly subscriptions.
- fc-taskflow-11: expected supported, got not_in_kb (paraphrase_true): Taskflow customers can keep their data hosted in Europe.
- fc-taskflow-13: expected not_in_kb, got contradicted (related_unsettled): Taskflow's Slack integration is only available on the Business plan.
- fc-taskflow-14: expected not_in_kb, got supported (high_overlap_unsettled): On top of its SOC 2 Type II audit, Taskflow is ISO 27001 certified.
- fc-linear-01: expected contradicted, got supported (wrong_year): Linear was founded in 2017 by Karri Saarinen, Jori Lallo and Tuomas Artman.
- fc-linear-02: expected contradicted, got supported (wrong_limit): Linear's Free plan lets you upload files up to 25 MB.
- fc-linear-03: expected contradicted, got supported (wrong_percentage): Non-profits get 50% off Linear's paid plans.
- fc-linear-04: expected contradicted, got supported (plan_mismatch): Linear's Business plan includes dashboards.
- fc-linear-05: expected contradicted, got supported (number_misattributed): On Linear's Business plan you get 5 times the standard API rate limit.
- fc-linear-07: expected supported, got not_in_kb (true_negation): If you cancel Linear partway through your subscription, you won't get money back.
- fc-linear-08: expected supported, got contradicted (true_only): Enterprise is the only Linear plan you can't pay for month to month.
- fc-linear-09: expected supported, got not_in_kb (paraphrase_true): Even Linear's free tier gives you API and webhook access.
- fc-linear-10: expected supported, got contradicted (comparison_true): On yearly billing, Linear Business costs $6 more per user per month than Basic.
- fc-linear-14: expected not_in_kb, got supported (unrelated): Linear has more than 200 employees.
