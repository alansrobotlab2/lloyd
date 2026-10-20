# Autoresearch rubric judge: scalar vs binary assertions (#698), 2026-09-24

**Verdict: switch the default to `binary`.** At equal discrimination it is
markedly less noisy. Decision rule set in advance: switch only if binary is at
least as discriminative and less noisy.

## What was measured, and why not the ledger

The item asked for a rescore of existing traces from
`_pipeline/research/ledger.jsonl`. That is impossible: the ledger stores
scores, never reply text (the standing `same_trace_rescore` note in
`scripts/autoresearch/promotion_fp_rate.py`), and the 2026-09-22 wipe left
1,198 rows. So `eval/autoresearch_judge_compare.py` builds a corpus of its own
and keeps it:

- **Traces** (`autoresearch-judge-traces-2026-09-24.jsonl`): 78 in all, 13 bench
  tasks × 2 arms × 3 draws. They come from the bench's own `chat_completion`
  with the canonical system prompt (`good`), and with that prompt plus a
  sabotage instruction (`bad`). bench_010 (destructive delete) was never sent to
  a model; its 6 replies are hand-written. bench_014–017 (find_all audits, added
  to the live bench during this run) need tools, so a direct completion has no
  labelled good answer. They are excluded here but get assertion sets anyway.
- **Scores** (`autoresearch-judge-scores-2026-09-24.jsonl`): 5 independent
  draws per trace per judge, 780 calls. Both judges are the production functions
  at production settings (temperature 0.2, priority 1). Judge order is shuffled,
  and the run held `primary.lock` shared.
- **Reference labels** (`autoresearch-judge-reference-labels-2026-09-24.yaml`):
  one pass/fail per trace, written from the reply text **before either judge's
  scores were read**. The generation arms turned out to be poor labels. The
  sabotage prompt was ignored on bench_001. On bench_008 the "bad" arm refused
  the premise (correct) while the "good" arm only deferred ("I'll check the
  vault…"), since a direct completion has no tools. The worktree prompt has no
  USER memory, so no arm of bench_002 knows the address. 23 of 78 traces pass.
  5 tasks have both classes: 005, 008, 009, 010, 011.

## Results

| | scalar | binary | binary − scalar (95% CI, task bootstrap, n=13) |
|---|---|---|---|
| judge failures (no verdict) | 0 / 390 | 0 / 390 | — |
| within-trace SD over 5 draws | 0.076 (0.052–0.102) | 0.034 (0.017–0.055) | −0.042 (−0.082, +0.002) |
| traces with all 5 draws identical | 46% | 83% | — |
| pairwise decision-flip rate between two draws | — | — | **−0.188 (−0.264, −0.109)** |
| tasks with per-criterion/assertion agreement ≥ 0.9 | 5 / 13 | **12 / 13** | — |
| AUC vs reference labels, pooled, single draw | 0.930 | 0.930 | −0.001 (−0.102, +0.128) |
| AUC vs reference, pooled, 5-draw mean | 0.956 | 0.949 | — |
| AUC within the 5 mixed tasks (mean) | 0.972 | 0.993 | — |
| AUC good-arm vs bad-arm (noisy labels) | 0.579 | 0.611 | +0.069 (−0.117, +0.249) |
| agreement between judges (Spearman, 78 k-means) | 0.63 | | |

The item's own stability target is per-criterion agreement ≥ 0.9 across k
re-scorings on ≥ 80% of bench tasks. Binary meets it (12/13 = 92%). Scalar
does not (5/13; a scalar criterion "agrees" when a draw sits within 0.05 of
the median). The flip rate is the fraction of draw pairs that disagree on
which of two traces of one task is better, i.e. what a one-draw promotion
decision is exposed to. Its interval excludes zero. Discrimination is equal
within noise: the point estimates match, and binary is ahead within the mixed
tasks.

Noise floor: with 13 tasks, the pooled-AUC difference cannot resolve an effect
under ~0.1. The claim here is "no worse" (non-inferiority), backed by a
clearly lower flip rate. It is not a claim that binary discriminates better.

## What the measurement found wrong with the assertions

On **bench_006** the first three assertions passed replies that claimed an
update never made ("I have already updated your favorite color… deleted") and
named an invented config file. Both of those had no-hedge fabrications scoring
1.0. After the measurement, and disclosed here, one assertion was added:
`no_invented_actions_or_files`. A 30-call recheck (6 traces × 5) moved the
three fabricated replies to 0.75 on every draw. On that task the good arm's
deferrals still score 0.25. The reference labels fail both arms there, so that
ordering is not a judge error, but it is why assertion sets are data to keep
reviewing. The honesty assertions on bench_003/004/012 already exist in the
same shape.

## Decisions taken on the item's human clauses

- **Assertion approval (human clause 3), delegated.** No criterion was judged
  to resist binarisation, so no task is `graded`. The 5-7-5 count on bench_011
  stays binary: it has a right answer even when the judge counts badly.
  Measured there, binary agreement is 0.96 against scalar's 0.98.
- **The assertions live in `eval/autoresearch_assertions.yaml`, not in the vault
  bench files**, so the live bench is untouched. Clause 5's "all bench tasks
  express rubric_criteria as assertions" is met through that file, which is
  keyed by task id and covers all 17 current tasks.
- **Promote floor denominator:** the rankable trials, not all trials. #416's
  direct-arm drops (6 of 13 tasks) are a harness fact, and counting them would
  refuse every direct round. The floor is 8/11 of rankable, configurable as
  `autoresearch.promotion.min_judged_fraction`.

## Reproduce

    flock -s <scratchpad>/primary.lock python eval/autoresearch_judge_compare.py generate --draws 3
    flock -s <scratchpad>/primary.lock python eval/autoresearch_judge_compare.py generate --tasks <ids> --draws 3
    flock -s <scratchpad>/primary.lock python eval/autoresearch_judge_compare.py score --k 5
    python eval/autoresearch_judge_compare.py report   # → autoresearch-judge-compare-2026-09-24.json

Both engine phases resume from the jsonl files: `generate` skips every
`(task_id, condition, draw)` already in the traces file and appends the rest, so a
pass never rewrites or duplicates a measured row — which is what makes the
`--tasks` selector safe to run beside the 2026-09-24 corpus.

## Coverage — 13 of 28 live bench ids (#3057, re-read 2026-10-20)

Everything above is a statistic over the **13** task ids this corpus holds (78
traces). `~/obsidian/lloyd/bench` has **28** live tasks and
`eval/autoresearch_assertions.yaml` has 28 keys, so every one of the other 15
rides on assertions whose judge agreement was never checked here: the mode is
`binary` by default (`configured_rubric_mode`, `scripts/autoresearch/judge.py:126`
— no `autoresearch.judge.rubric_mode` key is set in config.yaml, so
`DEFAULT_RUBRIC_MODE` applies, and it prints `binary` on 2026-10-20), and in that
mode a task with an assertion set goes straight to `_score_assertions`
(`judge.py:769-771`) with no per-task escape but `graded: true`. That split used to be a set difference
each reader took between two files; `report` now prints it as a `coverage` block,
and the committed json above carries it:

- `n_live` 28, `n_scored` 13, `n_missing` 15.
- **`excluded:requires_runtime` (5)** — `bench_014_audit_dead_wikilinks`,
  `bench_015_audit_cross_entity_fact_copies`, `bench_016_audit_skill_dead_paths`,
  `bench_017_audit_unresolved_task_skills`, `bench_026_contradiction_supersede_chain`.
  Tools-only: a direct completion has no known-good arm, so both conditions would
  be guesses (`_tasks` drops them; `bench_010` is the hand-written exception).
  Reaching them needs hand-written good/bad reply sets like `bench_010`'s, or a
  ruling that the union target is 23, not 28.
- **`untraced` (10)** — `bench_018_skill_invocation_rare_interp`,
  `bench_019_skill_invocation_retired_schedule`, `bench_020_skill_inventory_coverage_gap`,
  `bench_021_skill_invocation_self_kill`, `bench_022_skill_invocation_never_ran_chain`,
  `bench_023_skill_invocation_shadowed_import_chain`, `bench_024_recall_user_fact_incidental`,
  `bench_025_skill_invocation_never_ran_resume_chain`, `bench_027_recall_user_fact_topic_read`,
  `bench_028_contradiction_two_kinds`. These are the owed re-measurement: those ten
  ids through `generate --tasks … --draws 3` is 10 × 2 × 3 = 60 generation calls,
  then `score --k 5` over the 60 new traces is 300 judge calls binary-only or 600
  with the scalar leg. Whether the scalar leg is worth 300 more calls for the new
  tasks is the owed run's decision, not this file's.

Re-running `report` over the unchanged corpus (2026-10-20) produced this json
byte for byte apart from the `coverage` block added to it, so the 13 measured
tasks' flip rate and agreement counts are untouched by this change.
