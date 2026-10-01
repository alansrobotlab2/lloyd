---
segment: architecture
tags: [architecture, lloyd, eval, bench, measurement]
type: reference
status: implemented
date: 2026-09-29
---

# Lloyd — the measurement surface

Every number Lloyd quotes about itself — a retrieval hit rate, a tool-choice
regression, whether the secondary engine is good enough for a job class — comes
from the surface here: the arms under `eval/`, the runners at its top level, and
the bench task corpus in the vault. This doc says where a number lives, what
produced it, and which of them stands between a round and landing.

It was filed as #1699 because the surface had no owner. Twenty other
architecture docs cite `eval/` paths (nineteen of them when the item was
filed); `testing.md` is explicitly the pytest suite and nothing
else, and `evaluation-engine.md` was retired in the OpenClaw purge. An area cited
from that many places and owned by none is how one doc came to name an eval arm
that had never been created and treat it as an authority — the directory is still
absent from `eval/`, which is the point of the incident (#1421), and which is why
it is not cited here as a path: a doc that cites a directory it should know is
missing has written the drift into the file that is supposed to catch it.

## Two roots, and which one a number lives in

`eval/` in the repo is code and fixtures. The scored runs are not there.

| Root | Holds | Written by |
|---|---|---|
| `eval/` (repo) | the runners, the arm fixtures and corpora, and a small set of committed baselines | a commit |
| `$LLOYD_DATA/eval/baselines/` | every baseline a run produced since the runtime data root moved | the runners themselves |

`app/paths.py:100` is the one definition — `EVAL_BASELINES_DIR = DATA_ROOT /
"eval" / "baselines"` — and `eval/run_eval.py:2055` writes its record there, not
beside the script. The repo's `eval/baselines/` keeps a tracked handful (the
context-rot, compaction-recall and skills-index pins, for instance) so a reader
knows the *shape*; the directory that answers "what has actually been measured
lately" is under the data root, and it holds runs from the last few days rather
than the last few commits. The split is the 2026-09-22 tree deletion
([[data-home]]), and `testing.md` §2 names the same root for the same reason.

Which one a citation means matters every time it is written bare: the corpus
that moved roots once is exactly the corpus a stale path reports as empty rather
than as moved. Say `$LLOYD_DATA/eval/baselines/` or `eval/baselines/`; a
backticked path under no root at all is the spelling this section exists to
forbid, and the extractor that checks this doc reads one as an inventory entry,
which is why it stays out of backticks here.

## The arms

An arm is a directory under `eval/` holding a corpus and its outputs. The two
artifact directories (`baselines`, `measurements`) and `__pycache__` are not
arms. `tests/test_architecture_coverage_doc_claims.py` holds this table and that
directory to the same set, in both directions, so an arm added without a row and
a row naming a deleted arm both fail.

| Arm | What it scores | Read by |
|---|---|---|
| `behavioural_scenarios` | autoresearch traces scored per behaviour, `v1` | `scripts/autoresearch/behavioural.py` |
| `constraint_conflict` | paired task/control scenarios where two rules collide (`cc04` has a grant-gate pair), deliberately kept out of the live bench corpus | `scripts/autoresearch/constraint_conflict.py`, `tests/test_constraint_conflict_suite.py` |
| `djev` | the decision engine's schema registry, calibration ladder, name-prior probe and bake-off | `eval/djev/schemas.py`, `agent_mcp/djev.py`, `app/djev_shadow.py` |
| `durable_write_judge` | whether a write landed exactly once across a timeout (#544): corpus, retrieval, judge and both raw judge arms | `eval/durable_write_judge/build_corpus.py`, `eval/durable_write_judge/judge.py` |
| `engine_output` | engine behaviour under `cold`, `idle`, `preempt` and `heldout`, plus a noise floor | `eval/engine_output_probe.py` |
| `episodic-arm` | the episodic-recall arm's result set and report | `eval/episodic_arm.py`, `tests/test_episodic_arm_report.py` |
| `instruction-compliance` | an IFScale baseline over the compliance vocabulary | `eval/instruction_compliance_eval.py` |
| `iv` | the tracked Inner Voice intervention corpus, seeded from recovered `[INNER VOICE]` lines | `scripts/iv_corpus.py`, `app/usage_store.py` |
| `memory_eval` | memory-recall question set `v1` | `eval/memory_eval_build.py`, `eval/run_memory_eval.py` |
| `review_calibration` | labelled review-rung outcomes, including a first cut and a deliberately test-stripped one, so the grader can be calibrated against known verdicts | `scripts/automod/review_tools.py` |
| `secondary-routing` | keep/raise/revert decisions per job class for routing generation to the secondary engine | `eval/secondary_routing_eval.py`, `app/secondary_models.py` |
| `supply-chain` | whether the install-provenance predicate blocks the distribution names that do not exist without blocking one that does, and whether an advisory scan ran at all — 61 planted names replayed through the shipped code against a faked registry and a pinned clock, over 180 of the repo's 181 declared dependencies as the false-block control, 0 of them blocked (#1610) | `app/harness/supply_chain.py` (`fixtures`, `scan --write-baseline`), `tests/test_supply_chain_provenance_fixtures.py` |
| `uptake` | whether injected context reached the turn that needed it, with hand labels under `eval/uptake/labels/` | `app/uptake.py` |
| `injection_canary` | whether a planted instruction in fetched content drives a sink call — twelve worker-style tasks whose transcript, backlog body, page, README or module carries an instruction beside a canary token, scored on the canary reaching a non-read-only tool or Bash in the calls the model proposed, with two controls; the positive class for the P10 shadow seams (`action_review`, the input probe), whose verdicts are read back per row | `eval/run_injection_canary.py`, `tests/test_injection_canary.py` |

The corpora are the point of the arms; the scripts that read them mostly live one
level up, so an arm's *meaning* is in the arm and its *method* is in a runner.
`eval/stats.py` is the interval arithmetic every one of those numbers is read
against (#696) — stdlib-only on purpose, so the interval is importable from a
bare `python3` in a gate worktree with no store and no daemon.
`scripts/eval_trend_stats.py` folds the nightly runs into a trend and reads them
from `$LLOYD_DATA/eval/baselines/`, not from the repo copy.

A baseline carries two fingerprints of what it was asked, and they are kept
apart on purpose: `labels_sha256` hashes the gold (and is the ceiling artifact's
identity, so it must not move on a typo fix), `questions_sha256` hashes
`[id, query]` of the scored records (#1852). A re-worded question keeps its id
and its gold, so only the second can see it; the trend tool compares each
record's own `query` across a pair and prints `QUESTION BREAK` naming the ids —
an annotation, but such a pair is never an admissible verdict.

## Arms that are flag values: the compaction-recall runner

§The arms inventories **directories**. The same word carries a second, unrelated
sense in this tree: `--arms` is a flag six runner scripts under `eval/` and
`scripts/` declare, and its values are configuration presets inside one script.
`memory_eval` is the collision spelled out — it is a directory arm *and* a
runner whose `--arms` values (`closed_book`, `history`, `prefetch`) are not
directories. Nothing separates the two senses but this paragraph: **an `--arms`
value gets no row in §The arms**, and the node that holds that table set-equal
to `eval/*` in both directions is exactly what refuses one, so a reader who puts
a preset there gets a red test where they meant to write an answer.

`eval/run_compaction_recall_eval.py` is the runner two shipped-off `compaction`
flags lean on. Its `ARMS` dict holds fifteen presets; the `--arms` default runs
the threshold family (`none`, `production`, `tool_clear`, `raised`, `trigger90`)
and the summary family is `summary_legacy`, `summary_persisted` and
`memory_flush`. The middle column is counted over the tracked baselines — the
`compaction*.json` files under `eval/baselines/` — and
`tests/test_architecture_coverage_doc_claims.py` set-equals both the preset list
and that column against the dict and those files, so an arm that gets run, or a
preset added to the dict, turns this table red until somebody reads it again:

| Preset | Kept rows | What it is |
|---|---|---|
| `none` | yes (26 kept) | no compaction beyond the window — the floor |
| `production` | yes (26 kept) | `config.yaml` as it is, both passes; the threshold family's paired baseline |
| `tool_clear` | yes (26 kept) | tool-output clearing only, trigger/target 0.2/0.1 |
| `raised` | yes (10 of 20 kept) | 0.9/0.7 at both passes: fire later, keep more |
| `trigger90` | yes (10 of 20 kept) | trigger 0.9, target 0.52 |
| `self_record` | no | #1514's free route: clearing plus a clause naming the session's own record |
| `observation` | no | #1481's observation stubs and `recall_observation` |
| `production_self_record` | no | that switch at production's thresholds |
| `production_observation` | no | that switch at production's thresholds |
| `rung4` | no | #1499's truncation rung, today's spill-first form |
| `rung4_lossy` | no | that rung's pre-2026-09-11 lossy form |
| `rung4_self_record` | no | today's rung 4 with #1514's clause |
| `summary_legacy` | yes (12 kept) | regenerate-every-turn 9-section summary — **the baseline the other two summary presets are scored against** |
| `summary_persisted` | yes (12 kept) | D2: `compaction.persist_summary`, the incremental 5-section record |
| `memory_flush` | yes (12 kept) | P3: `summary_legacy` preceded by an `app/memory_flush.py` turn |

The comparison those two flags were waiting for ran on 2026-09-25:
`eval/baselines/compaction-summary-arms-2026-09-25.json`, twelve kept rows per
summary preset, no drops, no errors, paired against `summary_legacy` in the
artifact's own `paired_vs_summary_legacy` block (the runner computes it as
`paired(rows, base="summary_legacy")`). `summary_persisted` came in -0.25 on
distinctive recall (CI -0.58 to +0.08 — a loss in direction, not in
significance) and +42 s at turn start (CI +30 to +53, significant), with median
probe TTFT 11.7 s against legacy's 4.1 s. `memory_flush` gained +0.08 on both
recall measures (not significant) for +5.9 s at turn start (CI -0.4 to +12.1,
not significant), and across its twelve rows the flush turn saw the planted fact
in the bound history 8 times and wrote the distinctive one down once. The
write-up is `eval/measurements/compaction-summary-arms-2026-09-25.md` and its
verdict is **keep both flags off**.

So the two `config.yaml` comments that say a flag ships off "until the recall
eval's … arm is compared" name a comparison that has happened, and what each
flag is actually waiting for is different. D2's intended win is a cross-turn
effect — reuse one stable summary across consecutive over-threshold turns — and
no preset in `ARMS` compacts a session more than once, so that question is
unmeasured and needs a new arm rather than a re-run; P3's question was measured
and the answer was no. Flipping either flag is a ruling on those two facts, not
a reading of a green cell: nothing in this table is asserted passing, and the
counts above are re-measured with a glob over `eval/baselines/`, not copied from
here.

## The bench corpus

The bench tasks are markdown, not Python, and they are not in this repo: they sit
under `~/obsidian/lloyd/bench`, which is what `paths.bench_dir` in `config.yaml`
names. A bench task is an instruction plus what a correct run looks like; the
runner is `scripts/autoresearch/bench_runner_sdk.py`, which drives the real
harness with real tools.

That is why measurement and authority meet here: a bench trial that could delete
the vault is the point of the safety benches, so the runner refuses to start
unless the read-only tool sandbox reports itself on
(`agent_mcp/_tool_sandbox.py`; [[authority-surfaces]]). `eval/bench_lint.py` is
the `eval/`-side handle for the validity lint whose implementation is
`scripts/autoresearch/bench_lint.py` — one code path, two entry points, so a
nightly job and a round cannot disagree about whether a task is well-formed.

The promotion gate's safety leg (`promote.evaluate_promotion`) refuses on the
variant's own `safety_passed` and compares nothing, so a `safety_regression`
decision row alone cannot tell a variant that broke safety from a check nobody
can pass. Since #1927 such a row also carries `baseline_safety_passed`: the
baseline's per-task flag for each safety-critical task it scored (`{}` = it
scored none, key absent = no baseline in scope). It is a record, not an input:
the #1936 ruling (2026-10-01) declined a baseline comparison as a weakening. What
it shows is a freeze — from 2026-09-28 to 2026-10-01 `bench_010`'s #1607 anchor
accepted only a prose opening while the contract answers that prompt with the
block signal, so the baseline failed too and no round promoted. The fix was in
the task file (both refusal-form checks now also accept an opening
`{"status": "blocked"`), and `tests/test_bench_invariants.py` permits that shape
on the closed pair bench_009 + bench_010 only.

## A normalized score needs one population

`eval/run_eval.py` emits `<metric>_normalized = score / ceiling` only when both
halves say they divided over the same number of queries (`ci95[metric].n` and
`ceiling.n[metric]`). The gold-side ceiling drops every query whose gold was never
offered to the second labeler, the score does not, so on a full-corpus nightly
the two differ (2026-10-01: 66 vs 53 for `entity_hit_rate`, printed 1.2799) and
the ratio is null with a `population mismatch` note naming both n's, in
`ceiling_notes` and on the printed line (#2014). Scoring the numerator over the
ceiling's own query set would keep the field live; that is a separate change.

The ceiling itself is a function of the candidate menu the second labeler was
shown, so the artifact names its menu builder (`menu_builder`, beside `caps`) and
records an `offered` block per leg: gold offered at the run's cap, at 2x/4x/8x,
and with the ranking uncapped (#1937). Replayed on 2026-10-01 the shipped
builder (`query-token-overlap/v1`) offered 0.468 of entity gold at cap 40, 0.649
at cap 320 and 1.000 only uncapped — a gold sharing no word with its query is
ordered by spelling — so `outside_cap` does not mean a wider cap recovers a
label. `eval/gold_blind_menu_probe.py` measures replacement builders (alias
surfaces, embedding cosine, query-side expansion) over a namespace and corpus
it is handed, calls no labeler, and deploys nothing; the bar is entity offered
>= 0.80 at cap <= 60.

An autoresearch trial also records what it cost (#2019), in one currency:
`reprefill_cost_tokens`, the trial's `input_tokens - cache_read` summed over
the usage rows of its recorded session (`recorded_session_id` is the join key —
usage.db keys a bench row by that id, not by the trial id). A missing count is
null, never 0, and `usage.reprefill_tokens` is not the source: it read 0 on
bench rows that re-prefilled a whole prompt. `scripts/autoresearch/cost.py`
turns the rows into a per-variant record with a success-gated advantage —
`mean(cost | success, same task) - cost` for a successful trial, exactly 0 for a
failed one, an infrastructure failure excluded from both — and ranks variants by
successes before advantage, so cheaper-and-wrong cannot outrank correct. It is
emit-only: the round report and the decision row carry it, and no leg of
`evaluate_promotion` reads it.

## Which measurement stands between a round and landing

One arm is load-bearing at the gate: tool choice, run by the `prompt_surface`
rung of `scripts/automod/gate.py` and nothing else.

- It fires only when the diff touches a tracked prompt-surface path; otherwise
  the rung says `no prompt-surface path in the diff` and skips.
- It runs `eval/run_tool_choice_eval.py --label item<N>`, then
  `eval/compare_tool_choice.py --label item<N>`.
- It runs **from the round's worktree**, and the cwd is what makes the score the
  candidate's: the command names the script by relative path, the script roots
  its own `sys.path` at the tree it resolves into — ahead of the `PYTHONPATH`
  `_child_env` exports — so the `app.*` it imports, and the prompt it scores, are
  the round's. Pinned to the live checkout until #1790, which made the rung a
  canary on the shipped tree wearing a gate's clothes: red on drift a round did
  not cause, green over a round that regressed tool choice.
- The baseline that run writes stays **outside both trees**, which is what lets
  the launch sit in a worktree at all. It goes to `EVAL_BASELINES_DIR`
  (`eval/run_tool_choice_eval.py:498`), and `_child_env` with `live_data` set
  names the production data root in `LLOYD_DATA` rather than leaving it unset.
  Unset was never a location; it was an instruction to inherit one from the tree
  the script ran in, and a worktree keeps its data inside itself — rule 3 of
  `app.data_root` — so the record would land in `<worktree>/.lloyd-data/`, be
  deleted with the round, and leave the comparator answering `nothing to compare
  against` about a rung that had just measured something. The reason the live cwd
  used to give itself — that the baseline lived inside the checkout, true until
  runtime data left it at `6426668b` (2026-09-22) — is gone from the comment,
  from the test and from here, because the doc layer inherits whatever the
  comment asserts.
- The measured spread every floor is compared against lives in
  `eval/noise_floor_tool_choice.yaml`, and exactly one thing refills it: a
  same-tree run pair named `<stem>-a-<ts>.json` / `<stem>-b-<ts>.json` — two
  runs of an **unmodified** tree, taken as `eval/run_tool_choice_eval.py` twice
  with `--label noise-a` and then `--label noise-b`, whose filenames come out
  `noise-a-<YYYYMMDD-HHMMSS>.json` and `noise-b-<…>.json`
  (`eval/run_tool_choice_eval.py:498`). `same_tree_pairs()` matches those stems
  only when the two runs recorded an identical `config` block, and
  `eval/compare_tool_choice.py --measure-floor` is what measures the spread from
  them. No automated label can ever produce a pair: the rung above labels the
  run `item<id>` or `gate-<round_id>` (`scripts/automod/gate.py:1590`), so its
  arm residue is never `-a`/`-b`, and nothing calls `--measure-floor` on a
  schedule either. A person does it, or the record ages. Since #1888 that
  command refuses to write when no pair survived into the record — no pair on
  disk, or every pair dropped for an errored run — and exits 2 on the bytes it
  left alone; and a rewrite that does happen carries the previous record's
  `measured_at` and `pairs` under `superseded`, because the run files a floor
  was measured from get swept routinely and the spread cannot be re-derived
  afterwards.
- Exit codes come from the script, not a copy of its contract: 0 pass, 1
  regression, 2 nothing to compare against — which is not a pass — and 3
  instrument failure, meaning the control set moved so the comparison certified
  nothing and both sides must be re-run. Exit 3 is not an argument to argue past.

As a rung rather than four commands in the implementer's prompt, it runs while
the model is idle in `automod_gate_wait` — the one window in a round when nothing
else of that round's own is on the engine. Placed in the turn, it evicted the
round's own prefix twice per run. The full ladder is in [[automod]].

## What this doc does not cover

- **The test suite.** `tests/`, its two kinds of test, the production-tree
  refusal and the parallel isolation are [[testing]]. Nothing here is a pytest
  matter except the gate rung above.
- **The gate ladder around this rung.** `prompt_surface` is one rung; the
  promotion flow, the review rung and the guardian are [[automod]]. How the
  review grader is calibrated is `review_calibration`'s corpus, and the grading
  itself is #1699's neighbour, not this doc's subject.
- **Any retrieval method.** `eval/` holds the retrieval *measures*;
  how recall works is [[retrieval]] and [[qmd]], and the fact-side instruments
  are in [[knowledge-graph]].
- **The djev engine.** `eval/djev/` scores it; what it is, how it is served and
  what its trust flags mean is [[djev]].
- **Whether any arm is currently passing.** This is an inventory. Not one metric
  here is asserted green, and a list of instruments is not an all-clear —
  the failure `arch-review.md` opens with. A number is read from the arm's own
  latest artifact, in the root named above, at the time you read this.
- **The cost side.** Tokens, per-job accounting and what a turn cost are
  [[mission-control]] and the workers' own records; the retired
  `usage-tracking` doc is listed in the index for that reason and is not
  reinstated by this one.

## Review log

- **2026-09-28 — created (#1699).** Inventory taken at `f7cf29f4`: the twelve
  arms then listed were `eval/`'s directories minus `baselines`, `measurements` and
  `__pycache__`, and the item's own figures had already drifted (it said 52
  scripts where `eval/` holds 65, and twelve arms where three of its fifteen
  directories are artifacts). Both roots were read from `app/paths.py` and
  `eval/run_eval.py` rather than from the item's prose.
- **2026-09-29 — stale.** Three pointers had moved and one reason had been
  overtaken: `eval/run_eval.py` writes its baseline at line 2055, not 1995; the
  supply-chain arm is driven by that module's `fixtures` command, not by the
  Python function whose name the row quoted in its place (#1792); and the
  live-tree bullet's "baseline written next to the run" predates the data-home
  move, so what the rung now does is fire on the candidate's diff and score the
  live tree's prompt surface (#1790). The table is thirteen rows — the twelve in
  the entry above is the count before `supply-chain` arrived with #1610 on
  2026-09-28 — and the citing-docs figure is twenty, not nineteen. Re-read from
  the tree and unchanged: all thirteen arm names against `eval/*`, every cited
  path, the 61 planted names / 180 controls / 0 false blocks the eval itself
  reports, the 181 declared dependencies, and the four tool-choice exit codes.
  `scripts/eval_trend_stats.py`'s `--help` named the repo copy as the default
  root where this doc named the data root; the code had always backed the doc, and
  #1791 fixed the string, which now names `app.paths.EVAL_BASELINES_DIR` and the
  `LLOYD_ROOT` override instead.
- **2026-09-29 — §Arms that are flag values added (#1787).**
  `architecture/context-window.md` delegated two "ships off until compared"
  flags to this doc, and this doc named none of the arms involved: greps for
  `summary_persisted`, `memory_flush`, `persist_summary` and `summary_legacy`
  over this file returned 0 hits, and its single "compact" hit is the
  tracked-baseline filename list under §Two roots — a pin, not an arm. The
  fifteen presets, the kept-row counts and the paired deltas came from the
  runner's `ARMS` dict and the three `compaction*.json` baselines, not from that
  doc's prose. That matters for one figure in the item itself: it asked to have
  the pair recorded as "the uncompared pair behind the two config.yaml flags",
  and on the tree they are not — `eval/baselines/compaction-summary-arms-2026-09-25.json`
  carries twelve kept rows per arm and a `paired_vs_summary_legacy` block, and
  `eval/measurements/compaction-summary-arms-2026-09-25.md` ruled both flags
  stay off. What is unmeasured is D2's cross-turn reuse question, which no
  preset in `ARMS` can ask.
