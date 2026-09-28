---
segment: architecture
tags: [architecture, lloyd, eval, bench, measurement]
type: reference
status: implemented
date: 2026-09-28
---

# Lloyd — the measurement surface

Every number Lloyd quotes about itself — a retrieval hit rate, a tool-choice
regression, whether the secondary engine is good enough for a job class — comes
from the surface here: the arms under `eval/`, the runners at its top level, and
the bench task corpus in the vault. This doc says where a number lives, what
produced it, and which of them stands between a round and landing.

It was filed as #1699 because the surface had no owner. Nineteen architecture
docs cite `eval/` paths; `testing.md` is explicitly the pytest suite and nothing
else, and `evaluation-engine.md` was retired in the OpenClaw purge. An area cited
from nineteen places and owned by none is how one doc came to name an eval arm
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
"eval" / "baselines"` — and `eval/run_eval.py:1995` writes its record there, not
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
| `uptake` | whether injected context reached the turn that needed it, with hand labels under `eval/uptake/labels/` | `app/uptake.py` |

The corpora are the point of the arms; the scripts that read them mostly live one
level up, so an arm's *meaning* is in the arm and its *method* is in a runner.
`eval/stats.py` is the interval arithmetic every one of those numbers is read
against (#696) — stdlib-only on purpose, so the interval is importable from a
bare `python3` in a gate worktree with no store and no daemon.
`scripts/eval_trend_stats.py` folds the nightly runs into a trend and reads them
from `$LLOYD_DATA/eval/baselines/`, not from the repo copy.

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

## Which measurement stands between a round and landing

One arm is load-bearing at the gate: tool choice, run by the `prompt_surface`
rung of `scripts/automod/gate.py` and nothing else.

- It fires only when the diff touches a tracked prompt-surface path; otherwise
  the rung says `no prompt-surface path in the diff` and skips.
- It runs `eval/run_tool_choice_eval.py --label item<N>`, then
  `eval/compare_tool_choice.py --label item<N>`.
- It runs **from the live tree, not the worktree**, because the baseline is
  written next to the run and a baseline written inside a worktree is deleted
  with it (`SM_20260908_165950`'s was).
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
  arms above are `eval/`'s directories minus `baselines`, `measurements` and
  `__pycache__`, and the item's own figures had already drifted (it said 52
  scripts where `eval/` holds 65, and twelve arms where three of its fifteen
  directories are artifacts). Both roots were read from `app/paths.py` and
  `eval/run_eval.py` rather than from the item's prose.
