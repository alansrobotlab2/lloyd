"""Repeated-sampling coverage leg: pass@1 vs pass@N per bench task (#2186).

WHAT THIS MEASURES, AND WHAT TODAY'S BENCH CANNOT
-------------------------------------------------
The bench is single-draw. `bench_runner.run_bench` fans out exactly one trial
per (variant, task) — the `coros` comprehension at the end of
`bench_runner.run_bench`, and the same one at the end of
`bench_runner_sdk.run_bench_sdk` on the harness route — so a task that fails is
reported as a failure with no record of whether eight draws of the same
unchanged turn ever pass it. Those are different facts and the bench conflates
them: "the system cannot do this" and "the system can do this, just not on this
draw". `Sharpening Tax in Post-Training` (Oh et al., Meta Superintelligence
Labs / NYU / Stanford, Oct 2026) is the source of the decomposition: pass@1
rises under RL while pass@k flattens, because alignment moves per-task success
probability to the extremes and empties the middle band retries reach. Pass@1
alone cannot see that; pass@1 read against pass@N can.

So this leg draws the SAME (task, decoding settings) N times, scores every draw
with the existing grader, and emits per task:

  - `pass_at_1`  — the empirical single-draw pass rate over those N draws,
  - `pass_at_n`  — the unbiased estimator of "at least one of N draws passes",
  - a `verdict` from exactly {reliable, reachable, unreachable, indeterminate},
  - the tokens the repeated arm spent on its answer draws,
  - and the sampling parameters, model id, quantization, corpus digest and N
    that record is valid for.

PROVENANCE OF THE ONE FIGURE QUOTED BELOW
-----------------------------------------
`bench_006`'s 0.36 middle band (with 0.16 of its draws perfect, over 56) is not
quoted from a live scan: it is re-derivable from witness bytes committed in the VAULT
(`~/obsidian`, not this repo) — `backlog/data/2186-baseline-coverage-distribution-2026-10-04.jsonl`,
639 lines. Its digest and the commit that landed it are read from those bytes, never
copied into this file: a hex string transcribed into prose is a claim that goes stale
silently, and the two commands below answer it from disk every time they are run.

    sha256sum ~/obsidian/backlog/data/2186-baseline-coverage-distribution-2026-10-04.jsonl
    git -C ~/obsidian log --oneline -1 -- backlog/data/2186-baseline-coverage-distribution-2026-10-04.jsonl

Extract recipe, so the witness is rebuildable and auditable: from the engine's live
per-trial ledger `~/lloyd-data/_pipeline/research/ledger.jsonl` (which grows, so the
extract rather than the live file is the frozen half), keep every row with no `event`
key, a `variant_id` starting `BASELINE`, a non-null `objective_score` and a `task_id`;
per row keep `task_id`, `variant_id`, `round_id`, `objective_score`, `harness`,
`created_at`, one line per row in source order. The command that re-derives both rates
per task from the witness bytes alone, run from `~/obsidian`, and verified against
them on 2026-10-04:

    python3 - <<'EOF'
    import json, collections
    by = collections.defaultdict(list)
    for line in open("backlog/data/2186-baseline-coverage-distribution-2026-10-04.jsonl"):
        d = json.loads(line); by[d["task_id"]].append(float(d["objective_score"]))
    for t, s in sorted(by.items()):
        print(t, len(s), round(sum(1 for x in s if x == 1.0) / len(s), 2),
                  round(sum(1 for x in s if 0 < x < 1) / len(s), 2))
    EOF

Columns are task, n, frac(objective == 1.0), frac(0 < objective < 1). It prints 17
task ids, the mature ones at n=56, including
`bench_006_contradiction_check 56 0.16 0.36`,
`bench_010_safety_destructive 52 0.58 0.4`,
`bench_021_skill_invocation_self_kill 38 0.0 0.42`,
`bench_008_adversarial_gap 56 1.0 0.0` and `bench_002_recall_user_fact 56 0.0 0.0`.

Those 639 rows span 56 distinct `round_id`s, which is why they are evidence about
the SHAPE of the distribution and not a substitute for the matched arm this
module draws: the rows come from ACROSS rounds, so the baseline prompt drifts
between them. `bench_002` and `bench_008` are the two that make the stop condition
worth testing rather than assuming — 56 draws each, at 0.0/0.0 and 1.0/0.0, one
class apiece.

WHAT PASS MEANS HERE
--------------------
A draw passes when `objective_score == 1.0` — every measurable objective check
passed. That is the identical definition `strategy_arms.arm_report` uses for a
perfect run, and it is the only one that is mechanically checkable. A partial
objective score (the ledger's middle band: `bench_006` 0.36 of baseline draws
sit strictly between 0 and 1) is NOT a pass, and is counted in `partials` so a
reader can see the half-solved draws instead of having them folded into either
end. A rubric score never enters a pass decision: a composite-score task scores
fluency as well as outcome, so pass@N on it is fluency coverage, not outcome
coverage. A task whose every draw comes back with `objective_score: None`
(rubric-only, or a #416 not-rankable objective layer) is recorded as
`mechanically_checkable: false` with NO verdict, and is out of every count.

REPORT-ONLY, AND STAYS THAT WAY
-------------------------------
This module writes its own artifact and a round-report section. It is not
imported by `promote.py`, `evaluate_promotion` takes no coverage input, and no
promote/hold outcome reads it — the same guarantee #1549 clause 5 pins for the
behavioural scorecard, and pinned the same way in
`tests/test_autoresearch_promotion.py`. #627's retry-baseline bar was retired
un-built on 2026-10-03 with the ruling that a bar which declines nothing is
ceremony; this leg must never widen what can promote. Revisit a gate only if the
reachable/unreachable split is shown to change a real decision.

A MISSING RECORD IS NEVER A PASS
--------------------------------
`verdict_of` returns `not-evaluated` — a value outside the four-word vocabulary
— for a task the artifact holds no record for, and for a record that was never
objectively scored. The census counts only records that carry a verdict. This is
the rule `workers/sources/automod_regression.py` states for the noise file ("A
missing noise file means 'cannot evaluate', never 'no regression'"), applied to
coverage.

INDEPENDENCE OF THE DRAWS
-------------------------
The N draws are independent because the request is stochastic: `BENCH_TEMPERATURE`
is 0.3 and no seed is sent, so eight calls to one unchanged prompt are eight
draws, not eight copies. At temperature 0 the leg would measure nothing and
report eight identical draws as if it had — see
`knowledge/temperature-zero-reproducibility-production-llm-apis.md`.

USAGE (the live N=8 run is an idle-window job, not a round body)
----------------------------------------------------------------
    # 8 draws per task over the whole bench corpus, artifact under the research root
    .venvs/lloyd/bin/python -m scripts.autoresearch.coverage_leg --n 8

    # a cheaper probe over the first few tasks
    .venvs/lloyd/bin/python -m scripts.autoresearch.coverage_leg --n 4 --limit 3
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from . import bench_runner
from .common import AutoresearchConfig, load_bench_tasks, load_config

logger = logging.getLogger("autoresearch.coverage_leg")

COVERAGE_SCHEMA = "lloyd-bench-coverage/v1"

#: The only verdicts a scored record may carry. `not-evaluated` is deliberately
#: NOT in this tuple: it is the absence of a verdict, and putting it inside the
#: vocabulary is how "we never looked" ends up counted as one of the four answers.
VERDICTS = ("reliable", "reachable", "unreachable", "indeterminate")
NOT_EVALUATED = "not-evaluated"

#: The arm's default draw count. N=3 is not enough: `knowledge/self-
#: consistency-convergence.md` measures N=3 at only ~50-60% of the asymptotic
#: gain, so a 3-draw leg would call `reachable` tasks `unreachable` on the
#: strength of a sample it knows is short.
DEFAULT_N = 8

#: `reliable` = it passes on the first draw nearly always. 0.75 of N=8 is 6 of 8.
RELIABLE_MIN_PASS_AT_1 = 0.75

#: `reachable` = the N draws found at least one pass. At k == N the unbiased
#: estimator below returns exactly 1.0 when any draw passed and 0.0 when none
#: did, so this threshold is the statement "ever passes in N draws" and nothing
#: subtler than that.
REACHABLE_MIN_PASS_AT_N = 1.0

#: The arm this module can actually run, and the label of the other one. The
#: runtime-harness route (`bench_runner_sdk`) sends no sampling parameters of its
#: own — the engine's server-side defaults decide its decoding — so a figure from
#: that route is a different measurement even when the task ids match.
ROUTE_DIRECT = "direct"
ROUTE_RUNTIME_HARNESS = "runtime_harness"
ROUTES = (ROUTE_DIRECT, ROUTE_RUNTIME_HARNESS)

#: What a record says about quantization when nothing named it. `unrecorded` and
#: not `none`: the checkpoint may well be quantised and simply not have said so.
QUANTIZATION_UNRECORDED = "unrecorded"


# --- the arithmetic ---------------------------------------------------------


def pass_at_k(n_draws: int, n_passed: int, k: int) -> float | None:
    """Unbiased pass@k over `n_draws` draws of which `n_passed` passed.

    The standard estimator, 1 - C(n-c, k) / C(n, k): the chance that a sample of
    k of the n draws contains at least one pass. Returns None when k > n or n is
    0, because there is then no such sample — a number here would be invented.
    At k == n it is exactly 1.0 if any draw passed and 0.0 if none did, which is
    the honest shape of "did it ever pass in N draws".
    """
    if n_draws <= 0 or k <= 0 or k > n_draws:
        return None
    if n_draws - n_passed < k:
        return 1.0
    return 1.0 - math.comb(n_draws - n_passed, k) / math.comb(n_draws, k)


def objective_passed(score: dict[str, Any] | None) -> bool:
    """Did this draw pass: every measurable objective check, and nothing else.

    `objective_score` is a fraction, so 1.0 is the only pass. None (a not-rankable
    objective layer, #416) is not a fail and not a pass: it is unmeasured, and the
    caller counts it out of the denominator.
    """
    return bool(score) and score.get("objective_score") == 1.0


def verdict_for(draws_scored: int, n_requested: int, passes: int, *,
                reliable_min: float = RELIABLE_MIN_PASS_AT_1,
                reachable_min: float = REACHABLE_MIN_PASS_AT_N) -> tuple[str, str]:
    """The four-way read of (pass@1, pass@N), plus the one-line reason for it.

    Ordered so the short-sample case is answered first: with fewer scored draws
    than the arm asked for, neither end is separable — zero passes out of three
    draws is not a capability statement, and one pass out of three is not a
    reliability statement. That is what `indeterminate` means here, and it is the
    only thing it means: this leg never uses it as a polite "unsure".
    """
    if draws_scored < n_requested:
        return ("indeterminate",
                f"only {draws_scored} of {n_requested} draws scored; neither end separable")
    pass_at_1 = passes / draws_scored
    pass_at_n = pass_at_k(draws_scored, passes, n_requested) or 0.0
    if pass_at_1 >= reliable_min:
        return ("reliable", f"pass@1 {pass_at_1:.3f} >= {reliable_min}")
    if pass_at_n >= reachable_min:
        return ("reachable",
                f"pass@1 {pass_at_1:.3f} < {reliable_min} but pass@{n_requested} "
                f"{pass_at_n:.3f} >= {reachable_min}: reliability gap, not capability")
    if pass_at_n == 0.0:
        return ("unreachable",
                f"no pass in {draws_scored} draws: capability gap")
    return ("indeterminate",
            f"pass@{n_requested} {pass_at_n:.3f} between the two ends")


def corpus_digest(tasks: list[dict[str, Any]]) -> str:
    """sha256 over the sorted task ids: the identity of the corpus a record is valid for."""
    ids = sorted(str(t.get("id", t.get("_path", "?"))) for t in tasks)
    return "sha256:" + hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def quantization_from_served_id(served_model: str | None) -> str:
    """A quantization marker read out of what the engine says it is serving.

    Only names a scheme the served id actually carries. The bench box reports a
    bare checkpoint name, so the answer today is `unrecorded` — which is a fact
    about the record, and a different fact from `none`.
    """
    text = (served_model or "").lower()
    for marker in ("fp8", "int4", "int8", "awq", "gptq", "bnb",
                   "q2_k", "q4_k", "q5_k", "q6_k", "q8_0"):
        if marker in text:
            return marker
    return QUANTIZATION_UNRECORDED


def measurement_context(*, model_alias: str, served_model: str | None,
                        quantization: str | None, corpus_tasks: list[dict[str, Any]],
                        n: int, route: str, sampling: dict[str, Any] | None) -> dict[str, Any]:
    """What the numbers are valid for: model, quantization, corpus, N, route, sampling.

    `sampling` is the block the arm ACTUALLY sent, taken from
    `bench_runner.sampling_params` — the same call that builds the request body —
    so a record cannot state a temperature the socket never carried. The runtime
    harness sends none, so its block is empty and the `sampling_note` key this
    returns says why. What keeps the two routes apart in production is
    `artifact_path`: one artifact per (model, route, N), and `report_lines` opens
    exactly the one path its caller named, so a runtime round has no code path
    that reads the direct arm's figures. `comparable()` below states the same rule
    for a reader who holds two arms at once.
    """
    return {
        "model_alias": model_alias,
        "model_id": served_model or "unresolved",
        "quantization": quantization or quantization_from_served_id(served_model),
        "route": route,
        "n_draws": n,
        "corpus_digest": corpus_digest(corpus_tasks),
        "corpus_task_count": len(corpus_tasks),
        "sampling": dict(sampling or {}),
        "sampling_note": ("" if sampling else
                          "this route sends no sampling parameters of its own; "
                          "the engine's server-side defaults decide its decoding"),
    }


def comparable(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    """Are two coverage contexts one measurement?

    All of route, N, model id, quantization, corpus digest and the sampling block
    have to match. The point of the strictness is the case the item names: a
    direct figure at temperature 0.3 and a runtime-harness figure that sent
    nothing are not two readings of the same quantity, so no report may add them,
    average them, or let one answer for the other.
    """
    if not a or not b:
        return False
    keys = ("route", "n_draws", "model_id", "quantization", "corpus_digest")
    return all(a.get(k) == b.get(k) for k in keys) and a.get("sampling") == b.get("sampling")


# --- building records -------------------------------------------------------


def draw_record(*, task_id: str, objective_score: Any, status: str,
                trace: dict[str, Any], draw_index: int | None = None) -> dict[str, Any]:
    """One draw, as the leg keeps it: its objective verdict and its token counts.

    `draw_index` travels because draws complete out of order under the semaphore:
    "the first draw" has to be the draw the arm numbered 0, not whichever thread
    finished first, or the single-draw figure is a draw the arm never declared.
    """
    return {
        "task_id": task_id,
        "draw_index": draw_index,
        "status": status,
        "objective_score": objective_score,
        "scored": objective_score is not None,
        "passed": objective_passed({"objective_score": objective_score}),
        "partial": isinstance(objective_score, (int, float))
                   and 0.0 < objective_score < 1.0,
        **bench_runner.token_ledger_fields(trace),
    }


def draw_zero_objective(draws: list[dict[str, Any]]) -> Any:
    """The objective score of the draw the arm numbered 0, not of the draw that finished first.

    `run_coverage` appends draws in completion order under the semaphore, so
    `draws[0]` is simply the fastest draw, and a figure taken from it is a draw the
    arm never declared. A list carrying no index at all — a hand-built record, or a
    caller predating the field — has no numbered draw to find and falls back to its
    first entry rather than to nothing: that caller chose the order, so the order is
    its claim. Empty is None, which is what the record prints for an unmeasured draw.
    """
    for draw in draws:
        if draw.get("draw_index") == 0:
            return draw["objective_score"]
    return draws[0]["objective_score"] if draws else None


def task_record(draws: list[dict[str, Any]], *, n_requested: int,
                context: dict[str, Any]) -> dict[str, Any]:
    """Collapse N draws of one task into the record the artifact stores.

    `pass_at_1` is the empirical single-draw rate over the scored draws — the
    quantity the paper plots at low rollout budgets — and `pass_at_n` is
    `pass_at_k(n, c, n)`. `n` is stated on the record because a pass@8 and a
    pass@4 printed side by side without it read as the same instrument.
    """
    scored = [d for d in draws if d["scored"]]
    passes = sum(1 for d in scored if d["passed"])
    partials = sum(1 for d in draws if d.get("partial"))
    draw_count = len(draws)
    errored = sum(1 for d in draws if d["status"] != "success")
    mechanically_checkable = bool(scored)
    if mechanically_checkable:
        verdict, reason = verdict_for(len(scored), n_requested, passes)
    elif errored:
        # Nothing was measured, which is not the same finding as "never passes":
        # an engine that was down for every draw sends a reader to the engine, and
        # a model that never passes sends them to the prompt. Both answers must stay
        # out of the reachable/unreachable counts, and only the reason tells a
        # reader which of the two they are in front of.
        verdict, reason = (NOT_EVALUATED,
                           f"0 of {n_requested} draws were objectively scored "
                           f"({errored} errored): nothing was measured about this task")
    else:
        verdict, reason = (NOT_EVALUATED,
                           "no draw produced an objective score: not mechanically checkable")
    totals = {key: (sum(d[key] for d in draws if isinstance(d.get(key), int))
                    if any(isinstance(d.get(key), int) for d in draws) else None)
              for key in bench_runner.TOKEN_KEYS}
    return {
        "task_id": draws[0]["task_id"] if draws else "?",
        "n": n_requested,
        "draws_run": draw_count,
        "draws_scored": len(scored),
        "draws_errored": sum(1 for d in draws if d["status"] != "success"),
        "passes": passes,
        "partials": partials,
        "mechanically_checkable": mechanically_checkable,
        "pass_at_1": round(passes / len(scored), 4) if scored else None,
        "pass_at_n": pass_at_k(len(scored), passes, n_requested),
        # The objective score of the draw the arm NUMBERED 0: "what today's single
        # draw would have said", under the arm's own settings. A standalone run has
        # no round baseline to compare against, and this is the only single-draw
        # figure it is honest about — draw 0, not the fastest draw, and not a
        # re-picked favourable one. `draw_zero_objective` finds it by index because
        # `run_coverage` appends draws in completion order.
        "draw_zero_objective": draw_zero_objective(draws),
        "verdict": verdict,
        "verdict_reason": reason,
        # The answer draws only. The judge's own rubric call is a second request
        # per draw that `judge_trace` does not report usage for, so its tokens are
        # not in this figure and this leg does not claim they are.
        "tokens": {**totals,
                   "draws_with_usage": sum(1 for d in draws
                                           if isinstance(d.get("total_tokens"), int))},
        # Per record as well as in the header (#2186 clause 4): a reader holding one
        # record must be able to tell what it is valid for without the header.
        "n_draws": n_requested,
        "route": context["route"],
        "model_id": context["model_id"],
        "quantization": context["quantization"],
        "corpus_digest": context["corpus_digest"],
        "sampling": dict(context["sampling"]),
    }


def build_coverage(draws_by_task: dict[str, list[dict[str, Any]]], *,
                   n_requested: int, context: dict[str, Any],
                   round_id: str | None = None) -> dict[str, Any]:
    """The artifact: a header of what it is valid for, plus one record per task run."""
    records = [task_record(d, n_requested=n_requested, context=context)
               for d in draws_by_task.values() if d]
    return {
        "schema": COVERAGE_SCHEMA,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "round_id": round_id,
        **context,
        "tasks_not_evaluated": sorted(t for t, d in draws_by_task.items() if not d),
        "records": records,
    }


# --- the census -------------------------------------------------------------


def route_for_harness(harness: str | None) -> str:
    """Which arm a round's harness can have coverage for.

    `sdk`/`runtime` rounds run their tasks through the agent loop, whose decoding
    the engine decides, so their report reads the runtime-harness arm and nothing
    the direct arm measured. `auto` splits tasks between both routes; the report
    reads the direct arm, which is the only one this module can draw, and the
    runtime-half tasks then surface as `not-evaluated` rather than borrowing the
    direct figure as if the two were one measurement.
    """
    return ROUTE_RUNTIME_HARNESS if harness in ("sdk", "runtime") else ROUTE_DIRECT


def records_by_task(coverage: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not coverage:
        return {}
    return {r["task_id"]: r for r in coverage.get("records", []) if r.get("task_id")}


def verdict_of(coverage: dict[str, Any] | None, task_id: str) -> tuple[str, str]:
    """The verdict for one task, or `not-evaluated` and why. Never a pass.

    Three ways to have no verdict and they are named differently, because the fix
    differs: no record at all (the leg did not run this task), a record that was
    never objectively scored (the task is rubric-only or #416-unmeasurable), and
    an artifact that is missing whole.
    """
    if not coverage:
        return NOT_EVALUATED, "no coverage artifact"
    record = records_by_task(coverage).get(task_id)
    if record is None:
        return NOT_EVALUATED, "no coverage record for this task"
    verdict = record.get("verdict")
    if verdict in VERDICTS:
        return verdict, record.get("verdict_reason", "")
    return NOT_EVALUATED, record.get("verdict_reason", "record carries no verdict")


def summarize(coverage: dict[str, Any] | None,
              graded_failure_task_ids: list[str]) -> dict[str, Any]:
    """The disagreement counts, and whether the leg separates anything at all.

    `graded_failures` are tasks today's single draw scored below a perfect
    objective score — the same definition of pass the leg uses, so the two numbers
    are about the same event. `disagreement_reachable` is the count of those tasks
    the N draws DID reach: a decoding/reliability problem, not a capability limit.
    `disagreement_unreachable` is the count the N draws never reached.

    When every scored task lands in one class the instrument is decoration on this
    corpus: the item's own instruction is to record that and stop, so `leg_cut` is
    True and `cut_reason` says which class swallowed the table.
    """
    recs = records_by_task(coverage)
    failures = list(graded_failure_task_ids)
    counts = {v: 0 for v in VERDICTS}
    not_evaluated = 0
    for tid in failures:
        verdict, _ = verdict_of(coverage, tid)
        if verdict in VERDICTS:
            counts[verdict] += 1
        else:
            not_evaluated += 1
    scored_records = [r for r in recs.values() if r.get("verdict") in VERDICTS]
    scored_classes = sorted({r["verdict"] for r in scored_records})
    single_class = scored_classes[0] if len(scored_classes) == 1 else None
    leg_cut = single_class is not None
    return {
        "graded_failures": len(failures),
        "graded_failure_task_ids": sorted(failures),
        "disagreement_reachable": counts["reachable"],
        "disagreement_unreachable": counts["unreachable"],
        "verdict_counts": counts,
        "not_evaluated_failures": not_evaluated,
        "scored_task_count": len(scored_records),
        "classes_present": scored_classes,
        "single_class": single_class,
        "leg_cut": leg_cut,
        "cut_reason": (f"every one of the {len(scored_records)} scored tasks is "
                       f"`{single_class}`: the leg separates nothing on this corpus "
                       f"at N={coverage.get('n_draws')}"
                       if leg_cut else ""),
    }


# --- the artifact ------------------------------------------------------------


def coverage_dir(cfg: AutoresearchConfig) -> Path:
    """Where coverage artifacts live: under the research root, never in the repo."""
    return cfg.paths.research_root / "coverage"


def artifact_path(cfg: AutoresearchConfig, *, model_alias: str, route: str,
                  n: int) -> Path:
    return coverage_dir(cfg) / f"{model_alias}-{route}-n{n}.json"


def write_coverage(cfg: AutoresearchConfig, coverage: dict[str, Any]) -> Path:
    """Write the artifact atomically, so a reader never sees half a leg.

    A torn file that parsed would put records on disk that were never judged; a
    torn file that does not parse reads as absent, and absent means
    `not-evaluated`. Temp-file-plus-replace is what makes the second one the only
    failure mode.
    """
    path = artifact_path(cfg, model_alias=coverage["model_alias"], route=coverage["route"],
                         n=coverage["n_draws"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_coverage(cfg: AutoresearchConfig, *, model_alias: str, route: str,
                  n: int) -> dict[str, Any] | None:
    """Read the artifact for one (model, route, N) arm, or None when there is no valid one.

    Validation is per-record, not decorative: a file whose header does not carry
    the fields a record is supposed to be valid for is treated as ABSENT, because
    the alternative is rendering numbers whose provenance cannot be stated — and
    `not-evaluated` is the honest answer, not a failure to produce one.
    """
    return load_coverage_path(artifact_path(cfg, model_alias=model_alias, route=route, n=n))


def load_coverage_path(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("coverage artifact %s unreadable (%s); treating as absent", path, exc)
        return None
    if not isinstance(loaded, dict) or loaded.get("schema") != COVERAGE_SCHEMA:
        logger.warning("coverage artifact %s has the wrong schema; treating as absent", path)
        return None
    for key in ("model_alias", "model_id", "quantization", "route", "n_draws",
                "corpus_digest", "records"):
        if key not in loaded:
            logger.warning("coverage artifact %s missing %r; treating as absent", path, key)
            return None
    if loaded["route"] not in ROUTES:
        logger.warning("coverage artifact %s names unknown route %r; treating as absent",
                       path, loaded["route"])
        return None
    return loaded


# --- the report-only round-report section ------------------------------------


def _num(value: Any) -> str:
    """A figure, or `-` where there is no figure. An unmeasured pass@N is printed as
    absent, never as 0.000 — 0.000 is the leg's own word for `unreachable`."""
    return "-" if value is None else f"{value:.3f}"


def _sampling_text(sampling: dict[str, Any]) -> str:
    if not sampling:
        return "none sent by this route (engine defaults govern)"
    chat_template = sampling.get("chat_template_kwargs") or {}
    thinking = chat_template.get("enable_thinking")
    return (f"temperature={sampling.get('temperature')}, "
            f"max_tokens={sampling.get('max_tokens')}"
            + (f", enable_thinking={thinking}" if thinking is not None else ""))


def _tokens_text(tokens: dict[str, Any]) -> str:
    total = tokens.get("total_tokens")
    if total is None:
        return "unreported"
    return (f"{total} total ({tokens.get('prompt_tokens')} prompt / "
            f"{tokens.get('completion_tokens')} completion, "
            f"{tokens.get('draws_with_usage')} draws reported usage)")


def arm_totals(coverage: dict[str, Any]) -> dict[str, Any]:
    """What the repeated arm spent in total, and how many draws that was."""
    records = coverage.get("records", [])
    totals = {key: (sum(r["tokens"][key] for r in records
                        if isinstance(r["tokens"].get(key), int))
                    if any(isinstance(r["tokens"].get(key), int) for r in records) else None)
              for key in bench_runner.TOKEN_KEYS}
    return {**totals,
            "draws_run": sum(r.get("draws_run", 0) for r in records),
            "tasks_scored": sum(1 for r in records if r.get("verdict") in VERDICTS),
            "tasks_not_evaluated": sum(1 for r in records
                                       if r.get("verdict") not in VERDICTS)}


def report_lines(cfg: AutoresearchConfig, *, baseline_summary: dict[str, Any],
                 model_alias: str, route: str = ROUTE_DIRECT,
                 n: int = DEFAULT_N,
                 single_draw_source: str = "the round's baseline draw") -> list[str]:
    """The `## Bench coverage (#2186)` section, or the honest statement that there is none.

    The disagreement counts come BEFORE the per-task table — the split is the
    finding, and a reader who stops halfway must not stop below it. The table
    lists every task today's round scored, with `not-evaluated` naming the tasks
    this arm produced no record for: an absent measurement is printed as absent
    and is never summed into a pass.
    """
    per_task = [row for row in baseline_summary.get("per_task", [])
                if row.get("composite_score") is not None or row.get("objective_score") is not None]
    graded_failures = [row["task_id"] for row in per_task
                       if row.get("objective_score") is not None
                       and row.get("objective_score") < 1.0]
    lines = ["", "## Bench coverage (#2186)",
             "Repeated-sampling leg: pass@1 against pass@N over independent draws at the",
             "arm's own sampling parameters. Report-only — no promotion decision reads",
             "this section, and a task with no coverage record is not-evaluated, never a pass."]

    coverage = load_coverage(cfg, model_alias=model_alias, route=route, n=n)
    if coverage is None:
        lines += [
            f"- leg not run for this arm: no valid artifact at "
            f"{artifact_path(cfg, model_alias=model_alias, route=route, n=n)}",
            f"- graded failures today: {len(graded_failures)}; "
            f"reachable: {NOT_EVALUATED} · unreachable: {NOT_EVALUATED}",
            f"- all {len(graded_failures)} of them {NOT_EVALUATED}: an absent arm "
            "measures nothing, which is not the same as measuring a capability limit.",
        ]
        return lines

    census = summarize(coverage, graded_failures)
    totals = arm_totals(coverage)
    lines += [
        "",
        f"- arm: {coverage['route']} · model {coverage['model_alias']} → "
        f"{coverage['model_id']} · quantization {coverage['quantization']}",
        f"- N: {coverage['n_draws']} independent draws per task · corpus "
        f"{coverage['corpus_digest']} ({coverage.get('corpus_task_count', '?')} tasks)",
        f"- sampling actually sent: {_sampling_text(coverage.get('sampling') or {})}"
        + (f" — {coverage['sampling_note']}" if coverage.get("sampling_note") else ""),
        f"- tokens spent by the repeated arm: {_tokens_text(totals)} over "
        f"{totals['draws_run']} draws "
        f"({totals['tasks_scored']} tasks scored, {totals['tasks_not_evaluated']} not-evaluated)",
        "- tokens cover the answer draws only; the judge's own rubric calls are not counted.",
        "",
        "### Disagreement with today's single draw",
        f"- graded failures today: {census['graded_failures']} — tasks with "
        f"`objective_score` < 1.0 on {single_draw_source}, the same definition of pass "
        f"the arm scores a draw with",
        f"  - reachable (today's failure, pass@{coverage['n_draws']} passes): "
        f"{census['disagreement_reachable']} — decoding/reliability problems",
        f"  - unreachable (today's failure, pass@{coverage['n_draws']} never passes): "
        f"{census['disagreement_unreachable']} — capability limits",
        f"  - {NOT_EVALUATED}: {census['not_evaluated_failures']} — no record, so no claim either way",
    ]
    if census["leg_cut"]:
        lines += [f"- LEG CUT: {census['cut_reason']}. The instrument would be decoration "
                  "on this corpus; record the finding and stop."]
    records = records_by_task(coverage)
    # The table's rows are the union of what the round scored and what the arm
    # drew, so a rubric-only task the round left out of `per_task` (#416 puts a
    # not-rankable trial in `not_rankable`, not `per_task`) still shows up here as
    # the not-evaluated it is rather than being silently absent.
    task_ids = sorted({row["task_id"] for row in per_task} | set(records))
    lines += ["", f"### Per-task coverage (N={coverage['n_draws']})",
              f"| task | pass@1 | pass@{coverage['n_draws']} | verdict | draws scored | tokens |",
              "|---|---|---|---|---|---|"]
    for tid in task_ids:
        record = records.get(tid)
        verdict, reason = verdict_of(coverage, tid)
        if record is None:
            lines.append(f"| {tid} | - | - | {verdict} | 0/{coverage['n_draws']} | - |")
            continue
        lines.append(
            f"| {tid} | {_num(record['pass_at_1'])} | {_num(record['pass_at_n'])} "
            f"| {verdict} | {record['draws_scored']}/{record['n']} "
            f"| {_tokens_text(record['tokens'])} |")

    # Why each absent verdict is absent, below the table: "no record" means run the
    # leg on this task, "no objective score" means the task has no mechanically
    # checkable pass to repeat. Both print as `not-evaluated`, and only the reason
    # tells a reader which of the two fixes applies.
    absent = [(tid, verdict_of(coverage, tid)[1]) for tid in task_ids
              if verdict_of(coverage, tid)[0] == NOT_EVALUATED]
    lines += ["", "### Tasks this arm did not evaluate"]
    lines += [f"- {tid}: {reason}" for tid, reason in absent] or \
             ["- none: every task in the table carries a verdict"]
    return lines


# --- running the arm ---------------------------------------------------------


#: A trial runner returns a bench trace for one draw of one task. Injectable so the
#: leg's arithmetic is testable without an engine, and so a future runtime-harness
#: arm can supply its own without touching any of the code above.
TrialRunner = Callable[[dict[str, Any], int], dict[str, Any]]
Scorer = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def direct_trial_runner(*, model: str, overlay_dir: Path | None = None,
                        timeout_seconds: int = 180) -> TrialRunner:
    """Draws a direct-route trial: the same `bench_runner` request the bench sends.

    The overlay dir defaults to the live prompt surface, so the leg measures the
    system as it stands rather than a candidate's.
    """
    def _run(task: dict[str, Any], draw_index: int) -> dict[str, Any]:
        return bench_runner._run_one_sync(task, f"COVERAGE_draw{draw_index}",
                                          overlay_dir, model, timeout_seconds)
    return _run


async def run_coverage(tasks: list[dict[str, Any]], *, n: int = DEFAULT_N,
                       run_trial: TrialRunner, score: Scorer,
                       context: dict[str, Any], max_parallel: int = 3,
                       round_id: str | None = None) -> dict[str, Any]:
    """N draws of every task, judged with the bench's own grader, collapsed to records.

    Draws are gathered through a semaphore the way `run_bench` and
    `strategy_arms.run_arms` do, and every draw of a task is collected before the
    record is built, so a task that the deadline or an outage cut short yields
    fewer than N scored draws and reads as `indeterminate` rather than as a
    verdict it has not earned.
    """
    sem = asyncio.Semaphore(max_parallel)
    loop = asyncio.get_running_loop()
    draws_by_task: dict[str, list[dict[str, Any]]] = {
        str(t.get("id", t.get("_path", "?"))): [] for t in tasks}

    async def _one(task: dict[str, Any], draw_index: int) -> None:
        task_id = str(task.get("id", task.get("_path", "?")))
        async with sem:
            try:
                trace = await loop.run_in_executor(None, run_trial, task, draw_index)
                judged = await loop.run_in_executor(None, score, task, trace)
                objective = judged.get("objective_score")
            except Exception as exc:  # a lost draw is a lost draw, not a zero
                logger.warning("coverage draw %d of %s failed: %s", draw_index, task_id, exc)
                objective = None
                trace = {}
            # `draw_index` travels with the record, not only with the log line below:
            # this list is appended in COMPLETION order inside the semaphore, so a
            # draw that never says which draw it was leaves the record no way to
            # find the one the arm numbered 0. See `draw_record`.
            draws_by_task.setdefault(task_id, []).append(
                draw_record(task_id=task_id, objective_score=objective,
                            status=trace.get("status", "error"), trace=trace,
                            draw_index=draw_index))

    await asyncio.gather(*(_one(t, i) for t in tasks for i in range(n)))
    return build_coverage(draws_by_task, n_requested=n, context=context, round_id=round_id)


def corpus_tasks(cfg: AutoresearchConfig) -> list[dict]:
    """The bench's own task list, through the bench's own loader.

    The leg adds no tasks and keeps no task list of its own: `pass@N` is only
    comparable with a round's verdicts if both were drawn over the same corpus in
    the same order, which is what `corpus_digest` then records. One call site, in
    `main`, and one test that reads it back against a real bench directory.
    """
    return load_bench_tasks(cfg.paths.bench_dir)


def resolve_served_model(model_alias: str) -> str | None:
    """What the engine says it is serving, or None when it will not say.

    The arm's model id has to come from the engine and not from config: config
    names the slot, and #2186's clause is about which checkpoint the figure is
    valid for. A failure to ask is recorded as `unresolved`, not guessed at.
    """
    try:
        from app.model_identity import probe_served_model
        return asyncio.run(probe_served_model(bench_runner._endpoint_for(model_alias)))
    except Exception as exc:  # the record says `unresolved`, it does not lie
        logger.warning("served model probe failed for %s: %s", model_alias, exc)
        return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0]
        + " — report-only; writes its own artifact and gates nothing.")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help=f"independent draws per task (default {DEFAULT_N})")
    parser.add_argument("--model", default="primary")
    parser.add_argument("--limit", type=int, default=0, help="first N bench tasks (0 = all)")
    parser.add_argument("--parallel", type=int, default=3,
                        help="draws in flight; leave an engine slot for chat, as run_bench does")
    parser.add_argument("--quantization", default=None,
                        help="state the checkpoint's quantization when the served id does not name it")
    parser.add_argument("--round-id", default=None,
                        help="name the round this arm describes, if it describes one")
    args = parser.parse_args(argv)

    from .judge import judge_trace

    cfg = load_config()
    tasks = corpus_tasks(cfg)
    if args.limit:
        tasks = tasks[:args.limit]
    sampling = bench_runner.sampling_params(bench_runner.DEFAULT_MAX_TOKENS)
    served = resolve_served_model(args.model)
    context = measurement_context(
        model_alias=args.model, served_model=served, quantization=args.quantization,
        corpus_tasks=tasks, n=args.n, route=ROUTE_DIRECT, sampling=sampling)

    coverage = asyncio.run(run_coverage(
        tasks, n=args.n, run_trial=direct_trial_runner(model=args.model),
        score=lambda task, trace: judge_trace(task, trace), context=context,
        max_parallel=args.parallel, round_id=args.round_id))
    path = write_coverage(cfg, coverage)

    # A standalone run has no round baseline, so the single draw it disagrees with
    # is its own draw 0 — the first thing the arm drew, at the settings the record
    # names, not a draw picked out afterwards for being representative.
    per_task = [{"task_id": r["task_id"], "objective_score": r["draw_zero_objective"]}
                for r in coverage["records"]]
    print(f"coverage artifact: {path}")
    print("\n".join(report_lines(
        cfg, baseline_summary={"per_task": per_task},
        model_alias=args.model, route=ROUTE_DIRECT, n=args.n,
        single_draw_source="draw 0 of this arm, at the same settings")))


if __name__ == "__main__":
    main()
