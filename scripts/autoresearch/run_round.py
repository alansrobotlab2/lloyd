"""Autoresearch round orchestrator.

CLI:
    python -m scripts.autoresearch.run_round [--targets prompts] [--budget 60] [--dry-run]

MCP:
    autoresearch_round(targets=[...], budget_minutes=N, dry_run=bool) → calls run().
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from app.harness.bench_corpus import probe_ledger_fields

from . import auto_restore, post_promotion
from .bench_runner import run_bench, token_ledger_fields
from .bench_runner_sdk import DEFAULT_PER_TASK_TIMEOUT as SDK_PER_TASK_TIMEOUT
from .bench_runner_sdk import run_bench_sdk
from .common import (
    HARNESSES,
    AutoresearchConfig,
    find_last_promoted_variant,
    ledger_append,
    load_bench_tasks,
    load_config,
    now_iso,
    round_id,
    split_tasks_by_harness,
    validate_run_spec,
    write_run_spec,
    _run_spec_from_cfg,
)
from .hypothesis_generator import propose_variants
# The same numpy-free linear-interpolated percentile the promotion-false-rate
# analysis uses, so the prior and that analysis cannot disagree about what "p90"
# means on the same rows. Aliased because the derivation's own parameter is the
# quantile, and a function and its argument cannot share one name here.
from .promotion_fp_rate import percentile as _percentile_of
from .judge import aggregate_variant, configured_rubric_mode, judge_trace, rankability_fields
from . import bench_split
# #2186: the repeated-sampling coverage leg. Reached only on the way to the report
# (see the call site below); `promote.py` does not import it and does not import
# anything that does.
from . import coverage_leg
# #1549: the behavioural scorecard. Report-only by construction — this import is
# reached on the way to writing a report, after `evaluate_promotion` has already
# answered, and nothing in `promote.py` imports it back.
from . import behavioural
from . import cost as trial_cost
# `slice_metrics` is imported by name, never reached through the module: the
# name `promote` is bound two lines lower to the promotion *function*, so
# `promote.slice_metrics(...)` at the call site raised AttributeError on a
# function object — every real round died before it wrote a report, and only a
# test that stubbed the decision loop could get that far.
from .promote import (BASELINE_SAFETY_FIELD, REFUSAL_CLASS_FIELD, SAFETY_REGRESSION,
                      baseline_safety_flags, evaluate_promotion, promote, refusal_head,
                      slice_metrics, validity_report, validity_report_lines)
from .variant_sandbox import AnchorApplyError, materialize, materialize_baseline

logger = logging.getLogger("autoresearch.run_round")

#: `post_promotion_check`'s "look it up yourself" default. A sentinel object rather
#: than `None`, because `None` is a real comparison value here — it means "there was
#: no promotion on record to compare against" and has its own report branch, so a
#: `None` default could not tell "nothing to compare" from "caller did not say".
_COMPUTE = object()


def _group_traces_by_variant(traces: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for t in traces:
        grouped.setdefault(t["variant_id"], []).append(t)
    return grouped


def post_promotion_comparison(
    cfg: AutoresearchConfig, round_id: str, baseline_mean: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Look up the previous promotion and compare this round's baseline against it.

    Split out of :func:`post_promotion_check` because #1099 needs the verdict before
    anything is written: the lookup must precede this round's own `round_summary` row
    (a round must never be compared against itself, so `last_promotion` excludes
    `round_id` and the exclusion is worthless once the row exists), and the restore has
    to happen in the same window — after the comparison, before the record. Returns
    `(prior, comparison)`; `prior` is what tells the restore which snapshot and which
    vault commit it is undoing.
    """
    prior = post_promotion.last_promotion(
        cfg.paths.ledger_path, cfg.paths.rounds_dir, exclude_round=round_id,
    )
    return prior, post_promotion.compare(baseline_mean, prior, cfg.promotion_noise_floor)


def post_promotion_check(
    cfg: AutoresearchConfig,
    round_id: str,
    baseline_mean: float,
    promotion_result: dict[str, Any] | None,
    variant_summary: dict[str, Any] | None = None,
    candidate_overlay: Path | None = None,
    comparison: Any = _COMPUTE,
    restore: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any], dict[str, Any] | None]:
    """Compare this round's fresh baseline against the last promotion (#429).

    Returns `(report_lines, ledger_row, comparison)`. Recording and surfacing
    happen in this one call so the per-round comparison row is a property of
    every round rather than a step a round has to remember to take. The lookup
    runs *before* the write: a round must never be compared against itself.

    `candidate_overlay` is the winning candidate's overlay directory (None when the
    round had no candidate). This is where the #789 contract shape is measured,
    because this is the one caller that has both the live prompt files and the
    overlay in hand: `post_promotion` records and renders it but must not read
    prompt text or import the promote module (#429's separation test). The row is
    written before the drift block is built, so the block names this round too — the
    ratchet that refused a candidate read a history assembled *without* it, and a
    reader three promotions later wants to see the point that tripped it.

    `comparison` is the verdict from :func:`post_promotion_comparison`, passed by
    `run()` because the restore had to act on it before this call wrote the round's
    own row; `_COMPUTE` keeps a caller with no restore to run — a replay, a test —
    able to ask for the whole thing in one call. `restore` is #1099's outcome,
    rendered into this section by `post_promotion.report_section`: the restore happens
    *before* the record, but its verdict is reported in the section the record
    produced, because one section per round is what a reader follows.
    """
    from app import prompt_surface

    from .promote import contract_shape_fields

    if comparison is _COMPUTE:
        _, comparison = post_promotion_comparison(cfg, round_id, baseline_mean)
    lines = post_promotion.report_section(comparison, restore)
    row = post_promotion.record_round_summary(
        cfg, round_id, baseline_mean, promotion_result, variant_summary,
        contract_shape_fields(candidate_overlay),
    )
    lines += post_promotion.shape_report_lines(
        post_promotion.contract_shape_series(cfg.paths.ledger_path),
        prompt_surface.GATE_STACK_CEILING,
        prompt_surface.PROHIBITION_RATIO_CEILING,
        prompt_surface.CONTRACT_RISE_RUN,
    )
    return lines, row, comparison


def variant_surfaces(variant: dict[str, Any]) -> list[str]:
    """The prompt surface(s) a variant's proposal claims to edit.

    #680: a drop has to be attributable, and the only record of which surface a
    variant aimed at is its own edit list — the `AnchorApplyError` text names a
    path only for the edits that got as far as the canonical read, and a variant
    refused for spanning two surfaces names none. So the attribution reads the
    proposal, not the exception: the edit paths, falling back to
    `overlay_files` for a hand-built variant, falling back to a literal
    `"unknown"` rather than guessing a surface it was never told.

    A list, not a str, because the multi-surface rejection is exactly the case
    where one drop is attributable to more than one surface.
    """
    paths: list[str] = []
    for edit in variant.get("edits") or []:
        if isinstance(edit, dict):
            path = str(edit.get("path", ""))
            if path and path not in paths:
                paths.append(path)
    if not paths:
        paths = [str(p) for p in (variant.get("overlay_files") or {}) if p]
    return paths or ["unknown"]


def materialize_variants(
    cfg: AutoresearchConfig,
    variants: list[dict[str, Any]],
    baseline_pair: tuple[str, Path],
) -> tuple[list[tuple[str, Path]], dict[str, int]]:
    """Build the `(variant_id, overlay_dir)` list that a round actually benches.

    Returns `(variant_pairs, dropped_by_surface)`. #680 widened the second value
    from a bare int: every surface a proposed variant aimed at is seeded at 0
    and incremented on a drop, so a round that dropped 2 MEMORY variants and 0
    SOUL variants reports `{MEMORY.md: 2, SOUL.md: 0}` rather than `2`. The
    zeros are the point — an aggregate count cannot tell "the surface that is
    hard to anchor into" apart from "a surface nobody proposed", which is the
    distinction the fix has to be measured against. `sum(dropped_by_surface
    .values())` is the old aggregate.

    `variant_pairs` starts at the baseline —
    which `run()` needs the id of afterwards, so the caller passes the pair in
    rather than this function materializing it twice — and grows by one entry per
    variant `materialize` accepts. That list is the left operand of the
    (variant × task) matrix, so an entry missing here is a variant that is never
    scored, never compared to baseline, and never produces a decision row.

    #876: `fe40af3` (#446, 2026-09-09 13:08) reworked this loop and deleted the
    `variant_pairs.append` line without restoring it, so every round since has
    benched the baseline alone: the evaluation loop skips the baseline, so
    `decisions` stayed empty and the `event: "decision"` append wrote zero rows.
    The missing append is the whole defect; keep it pinned
    (`tests/test_autoresearch_variant_pairs.py`).

    #446's drop-whole semantics are unchanged: the parser bounds an edit's shape,
    and only here can an anchor be checked against the text it claims to quote.
    Zero or two matches raises `AnchorApplyError`, counts that variant against
    its surface, writes nothing, and the round goes on with the variants that did
    apply — no fuzzy match, no partial write.
    """
    variant_pairs: list[tuple[str, Path]] = [baseline_pair]
    dropped_by_surface: dict[str, int] = {}
    for v in variants:
        surfaces = variant_surfaces(v)
        for surface in surfaces:
            dropped_by_surface.setdefault(surface, 0)
        try:
            overlay = materialize(cfg, v)
        except AnchorApplyError as exc:
            logger.warning("dropping variant %s: %s", v.get("variant_id"), exc)
            for surface in surfaces:
                dropped_by_surface[surface] += 1
            continue
        variant_pairs.append((v["variant_id"], overlay))
    dropped = sum(dropped_by_surface.values())
    if dropped:
        logger.warning(
            "dropped %d of %d variants at anchored-edit apply (%s)",
            dropped, len(variants), _surface_drop_text(dropped_by_surface),
        )
    return variant_pairs, dropped_by_surface


def _validity_or_none(cfg: AutoresearchConfig, baseline_summary: dict[str, Any],
                      variant_summary: dict[str, Any], split: dict[str, Any]) -> dict[str, Any] | None:
    """`validity_report`, or None if the lint could not run at all.

    Advisory by construction, so a lint that cannot read the bench (a directory
    that does not exist, a task file whose front matter does not parse) must not
    end the round: the decision the round reports is the all-task one, and the
    absence of this field says so. A failure that silently made the two means
    AGREE would be the worst shape this function could have, so the exception is
    logged at warning and the round report prints nothing for that variant rather
    than a number.
    """
    try:
        return validity_report(cfg, baseline_summary, variant_summary, split=split)
    except Exception as exc:                                  # noqa: BLE001 — advisory
        logger.warning("bench validity lint failed; reporting the all-task mean alone: %s", exc)
        return None


def _rubric_exclusion_lines(summaries: dict[str, dict[str, Any]]) -> list[str]:
    """Report lines for trials the judge never scored (#646); empty when there are none.

    A rubric call that could not answer, answered without JSON, or answered with
    JSON that did not parse used to fold into the composite as a flat 0.5. That is
    a score the response did not earn and the judge did not give, and the round
    that loses its rubric engine scores every task at roughly half marks — a
    uniform shift that reads as a capability collapse rather than an outage. Those
    trials are excluded now, so the mean is over fewer trials, and the count is
    printed beside it: an unexplained mean over 6 of 13 is the number this section
    exists to make impossible.

    Sorted by variant id, then task id, so a diff across two rounds compares line
    for line — the same reason `_objective_coverage_lines` sorts.
    """
    lines: list[str] = []
    for vid, summ in sorted(summaries.items()):
        n = summ.get("rubric_excluded", 0)
        if not n:
            continue
        tasks = ", ".join(summ.get("rubric_excluded_tasks") or []) or "(names unavailable)"
        lines.append(
            f"- `{vid}`: rubric-excluded={n} of {summ.get('task_count', 0)}; "
            f"mean_composite is over {summ.get('scored_task_count', 0)} — the rubric judge "
            f"produced no verdict for these, so the 0.5 it used to contribute is in no mean: "
            f"{tasks}")
    return lines


def _objective_coverage_lines(summaries: dict[str, dict[str, Any]]) -> list[str]:
    """Report lines for the objective layer's gaps (#416); empty when there are none.

    A check about tool behaviour on a trace with no dispatch record is excluded
    rather than read off the prose, and a task whose whole objective layer is
    excluded contributes no marks to any mean. None of that moves the `mean=` line
    the report prints for a variant: a round that ranked 9 of 11 tasks prints the
    same mean as one that ranked all 11, which is exactly how this cutover would
    read as a regression. So the report names the tasks that dropped, the checks
    that were excluded, and the safety veto that never ran. The per-trial ledger
    row carries the same facts; this is the surface a human reads first, and the
    one the post-landing check in #416 asks them to look at.

    Sorted by variant id so a diff across two rounds compares line for line.
    """
    lines: list[str] = []
    for vid, summ in sorted(summaries.items()):
        for row in summ.get("not_rankable") or []:
            checks = ", ".join(row.get("excluded_checks") or []) or "(no check named)"
            lines.append(
                f"- `{vid}` {row['task_id']}: not rankable — excluded {checks}; "
                f"rubric {row.get('rubric_overall')} recorded, no objective mark")
        for task_id in summ.get("safety_objective_unmeasured") or []:
            lines.append(
                f"- `{vid}` {task_id}: safety-critical, its veto did not run — "
                "`safety=pass` here is the absence of a measured violation, not a pass")
        total = summ.get("task_count", 0)
        rankable = summ.get("rankable_task_count", total)
        if rankable != total:
            lines.append(f"- `{vid}`: {rankable} of {total} tasks contributed objective marks"
                         f" ({summ.get('excluded_check_count', 0)} check(s) excluded)")
    return lines


def _surface_drop_text(dropped_by_surface: dict[str, int]) -> str:
    """`{MEMORY.md: 2, SOUL.md: 0}` → `MEMORY.md: 2, SOUL.md: 0`.

    One formatter shared by the log line and the round report, so the two cannot
    describe a round differently. Sorted so the text is stable for a test and for
    a diff across rounds.
    """
    return ", ".join(f"{surface}: {count}" for surface, count in sorted(dropped_by_surface.items()))


async def _run_trials(
    cfg: AutoresearchConfig,
    variant_pairs: list[tuple[str, Any]],
    tasks: list[dict[str, Any]],
    model: str,
    harness: str,
    max_parallel: int,
    *,
    deadline: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Run the (variant × task) matrix, split by harness routing.

    `deadline` (a `time.monotonic()` instant) goes to both runners: past it no
    trial starts, and a trial that would outlive it runs on the time left and
    comes back marked `deadline_cut` (#1546).

    Returns `(direct_traces, sdk_traces, arm_seconds)`. `run()` has already taken
    the `requires_runtime` tasks a `direct` round skips out of `tasks`, so the
    third element of the split is empty here. `arm_seconds` (#1605) is the wall
    time each arm took — keys `direct_seconds` / `sdk_seconds`, a key absent when
    that arm had no task to run. One combined `trial_seconds` cannot say which arm
    ate the window, and the two arms are costed differently by the projection
    below, so the audit needs the split.

    Both runners are awaited sequentially rather than concurrently: they share
    the primary vLLM slot, and the direct runner's cap of 3 exists precisely
    because vLLM does not honor client disconnects (see its docstring).
    Overlapping an agent-loop workload on top of it would push both past their
    timeouts.
    """
    direct_tasks, sdk_tasks, _skipped = split_tasks_by_harness(tasks, harness)
    direct_traces: list[dict[str, Any]] = []
    sdk_traces: list[dict[str, Any]] = []
    arm_seconds: dict[str, Any] = {}

    if direct_tasks:
        arm_started = time.monotonic()
        direct_traces = await run_bench(
            cfg, variant_pairs, direct_tasks, model=model,
            max_parallel=max_parallel,
            per_task_timeout=300,
            deadline=deadline,
        )
        arm_seconds["direct_seconds"] = round(time.monotonic() - arm_started, 1)
    if sdk_tasks:
        logger.info("harness=%s: routing %d task(s) through the agent loop "
                    "(%d variants)", harness, len(sdk_tasks), len(variant_pairs))
        arm_started = time.monotonic()
        sdk_traces = await run_bench_sdk(
            cfg, variant_pairs, sdk_tasks, model=model,
            max_parallel=1,
            per_task_timeout=SDK_PER_TASK_TIMEOUT,
            deadline=deadline,
        )
        arm_seconds["sdk_seconds"] = round(time.monotonic() - arm_started, 1)
    return direct_traces, sdk_traces, arm_seconds


# The round's clock (#1546). The budget is what the worker source derives from
# the pool cap that would otherwise cancel the round mid-matrix; trials must be
# over early enough to leave the judge and the report their share of it. The
# judge runs one rubric call per trace on the primary — 1-4 s each in the rounds
# that completed before 2026-09-24 — so a fifth of the budget, never under
# three minutes, covers a 4-variant round's ~70 traces with room to spare.
JUDGE_RESERVE_FRACTION = 0.2
JUDGE_RESERVE_MIN_SECONDS = 180


def trial_deadline(started: float, budget_minutes: int | None) -> float | None:
    """The monotonic instant the round's trials must be over by, or None with
    no budget (a hand-run round that asked for none)."""
    if not budget_minutes or budget_minutes <= 0:
        return None
    budget = budget_minutes * 60.0
    reserve = max(JUDGE_RESERVE_MIN_SECONDS, budget * JUDGE_RESERVE_FRACTION)
    return started + max(budget - reserve, budget * 0.5)


def complete_matrix(traces: list[dict[str, Any]], variant_ids: list[str],
                    tasks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep only the tasks every variant has an uncut trial of.

    `(kept_traces, not_reached_task_ids)`. A task the deadline reached for some
    variants and not others would rank them on different tasks, and a trial the
    deadline cut short is a timeout the variant did not earn, so both go, for
    every variant at once. The runners walk the matrix task by task, so what is
    dropped is the tail of the task list, not a variant.
    """
    have: dict[str, set[str]] = {}
    for t in traces:
        if not t.get("deadline_cut"):
            have.setdefault(t["task_id"], set()).add(t["variant_id"])
    want = set(variant_ids)
    reached = {tid for tid, vids in have.items() if want <= vids}
    not_reached = [t["id"] for t in tasks if t.get("id") not in reached]
    return [t for t in traces if t["task_id"] in reached], not_reached


# ── the trial matrix: what the window can actually hold (#1605) ───────────────
#
# Both runners walk the matrix task by task and start nothing past the deadline
# (pinned in tests/test_autoresearch_deadline.py), so a matrix too big for the
# window is not slowed down — it is CUT, and `complete_matrix` then drops the
# unreached tasks for every variant at once. Every scheduled round of 2026-09-27
# stopped that way: a 1440 s trial window against an agent-loop arm awaited
# serially at ~85 s a trial (measured on those four rounds: 40 sdk trials, mean
# 82.1 s, median 85.2 s, p90 204.5 s; 216 direct trials, mean 2.8 s, p90 5.3 s).
# And because the cut takes the tail of the bench order, it took lint-valid tasks —
# the same pool `promote.py` scores its valid-only mean over — so a stopped round
# deleted the only evidence a promotion could be justified on.
#
# So the matrix is costed BEFORE it starts, each arm the way it actually runs, and
# what does not fit is dropped up front and named.
#
# What these two numbers ARE since #1715: the fallback. The priors a round actually
# costs itself at come from `derive_trial_priors`, which reads this same ledger's
# per-trial `duration_seconds` and takes each arm's p90. They used to be claimed
# "deliberately pessimistic against those measurements — direct at its p90", and
# that claim was false in the direction that mattered: measured over the rows the
# ledger holds on 2026-09-28 (direct n=1473, sdk n=67), the p90 is 27.9 s on the
# direct arm and 183.7 s on the agent-loop arm. A 5 s direct prior is not that
# arm's p90, and a 90 s agent-loop prior is HALF of it, so a matrix could be
# reported `fits yes` and be cut anyway — which is what happened to both
# post-landing scheduled rounds (`R_20260928_041936.md`: 1180 s projected, 1431 s
# window, fits yes, then 1370 s of trials and one task not reached).
#
# Where the fallback values come from: 5.0 is the `direct` arm's measured mean over
# the 2026-09-27 window (4.7 s over 224 trials) rounded up; 90.0 is
# `SDK_MIN_TRIAL_SECONDS` below, the runtime runner's own floor, so a trial that
# finishes under it is not scored and cannot honestly be costed cheaper. Neither is
# a percentile of anything, which is exactly why the derivation replaced them.
DIRECT_TRIAL_SECONDS = 5.0
SDK_TRIAL_SECONDS = 90.0

# The derivation's two knobs. p90 and not the mean because the serial arm is
# long-tailed on the same 67 rows (median 76.2 s, p90 183.7 s, max 280.8 s), and a
# mean-sized serial trial is not the trial that is still running when the deadline
# lands. 20 rows is the smallest n whose p90 is an interpolation between two real
# observations rather than one observation dressed as a percentile; both arms are
# already far over it on live traffic, so a scheduled round never waits on it.
PRIOR_PERCENTILE = 0.9
MIN_PRIOR_ROWS = 20


@dataclass(frozen=True)
class ArmPrior:
    """One arm's per-trial cost, with the evidence that produced it.

    `n`, `measured` and `percentile` are required rather than defaulted: a prior
    that reaches a projection has to say whether it was measured or is the
    fallback constant, and a default would let it arrive silent about which.
    """

    seconds: float
    n: int
    measured: bool
    percentile: float = PRIOR_PERCENTILE

    @property
    def source(self) -> str:
        return "measured p90" if self.measured else "fallback constant"

    def __str__(self) -> str:
        if self.measured:
            return f"{self.seconds:.1f} s {self.source} (n={self.n})"
        # A fallback has to say how far short it is, or `90.0 s fallback constant`
        # on a round report reads as though the ledger were empty forever rather
        # than a handful of rows short of being trusted.
        return (f"{self.seconds:.1f} s {self.source} "
                f"(n={self.n} measured rows, {MIN_PRIOR_ROWS} needed)")


@dataclass(frozen=True)
class TrialPriors:
    """Both arms, as one value. One object so the round derives once and threads
    one thing; `fallback()` is the only way to build the constants, so a caller
    cannot silently cost itself at them by omitting the argument."""

    direct: ArmPrior
    sdk: ArmPrior

    @classmethod
    def fallback(cls) -> "TrialPriors":
        return cls(
            direct=ArmPrior(DIRECT_TRIAL_SECONDS, 0, False),
            sdk=ArmPrior(SDK_TRIAL_SECONDS, 0, False),
        )

    def arms(self) -> tuple[ArmPrior, ArmPrior]:
        return (self.direct, self.sdk)

    def provenance(self) -> str:
        """What the report prints beside the projection: the prior per arm, and
        whether the ledger measured it or the constant stood in."""
        return f"priors — direct {self.direct}, agent-loop {self.sdk}"


def _arm_prior(durations: list[float], fallback_seconds: float, *,
               percentile: float = PRIOR_PERCENTILE,
               min_rows: int = MIN_PRIOR_ROWS) -> ArmPrior:
    if len(durations) < min_rows:
        return ArmPrior(fallback_seconds, len(durations), False, percentile)
    return ArmPrior(_percentile_of(list(durations), percentile), len(durations),
                    True, percentile)


def derive_trial_priors(ledger_path: Path, *, percentile: float = PRIOR_PERCENTILE,
                        min_rows: int = MIN_PRIOR_ROWS) -> TrialPriors:
    """Per-arm per-trial cost priors measured from the ledger's own trial rows.

    Importable on purpose: the round, a test and an operator question must be
    answerable by the same query over the same rows. Rows are matched the way the
    writer emits them — `scripts/autoresearch/runner.py:64` stamps `harness` and
    `duration_seconds` onto every trial row, and `_append_ledger` here is the other
    half — so the population is exactly what a round already writes, not a second
    measurement of it.

    An arm with fewer than `min_rows` rows falls back to the module constant and
    says so in its `source`; a missing file is the same fallback for both arms
    rather than a crash, because a round with no history is the normal state of a
    first round, not an error. Non-positive durations are dropped: the runner
    floors a real trial at `SDK_MIN_TRIAL_SECONDS`, so a zero is a row that never
    ran, and costing a round on those is how a prior gets optimistic again.
    """
    durations: dict[str, list[float]] = {"direct": [], "sdk": []}
    try:
        text = Path(ledger_path).read_text(encoding="utf-8")
    except OSError:
        text = ""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("variant_id") is None:      # spec / split / decision / summary
            continue
        arm = str(row.get("harness") or "direct")
        if arm not in durations:
            continue
        secs = row.get("duration_seconds")
        if isinstance(secs, (int, float)) and not isinstance(secs, bool) and secs > 0:
            durations[arm].append(float(secs))
    return TrialPriors(
        direct=_arm_prior(durations["direct"], DIRECT_TRIAL_SECONDS,
                          percentile=percentile, min_rows=min_rows),
        sdk=_arm_prior(durations["sdk"], SDK_TRIAL_SECONDS,
                       percentile=percentile, min_rows=min_rows),
    )

#: Fraction of the trial window a projected matrix may claim. The window already
#: gives the judge its reserve (#1546); this is the margin for the gap between the
#: priors above and whatever the primary actually took.
PROJECTION_MARGIN = 0.9

#: Never shrink below this. A round with no task measures nothing, and a window
#: that cannot hold one task is a mis-set budget: `fits` False on a one-task
#: matrix says that in the report rather than the round quietly benching nothing.
MIN_MATRIX_TASKS = 1


def _waves(trials: int, max_parallel: int) -> int:
    """Trials as they land on the direct arm's workers: `max_parallel` at a time."""
    return -(-int(trials) // max(1, int(max_parallel)))


def project_matrix(arms: int, tasks: list[dict[str, Any]], harness: str = "auto",
                   max_parallel: int = 4, *, window_seconds: float | None = None,
                   priors: TrialPriors) -> dict[str, Any]:
    """Project the wall cost of the (variant × task) matrix, arm by arm.

    `arms` is the number of things benched — the baseline plus every surviving
    variant, which is what `_run_trials` receives as `variant_pairs`.

    The arms are costed separately because they do not run the same way: the direct
    arm puts `max_parallel` trials on the workers at once, so it costs
    `ceil(arms × direct_tasks / max_parallel)` trials of time, while the agent-loop
    arm is awaited serially at `max_parallel=1` (both arms share the one primary
    vLLM slot), so it costs every trial it has. `window_seconds` is the room
    `trial_deadline` left; `fits` is the projection against it under
    `PROJECTION_MARGIN`, and stays None when there is no window — a hand-run round
    with no budget has nothing to fit into.

    `priors` is required, and required as a `TrialPriors` rather than as two floats
    defaulting to the module constants. That is the whole of #1715: #1605 shipped
    exactly those two defaulted floats, a call site that never passed them was
    indistinguishable from one that meant not to, and the constants quietly became
    the round's opinion about its own runtime. Now costing a matrix at the fallback
    is a deliberate act — `TrialPriors.fallback()` — and it carries the word
    "fallback" into the report line this projection prints.
    """
    direct_tasks, sdk_tasks, _skipped = split_tasks_by_harness(tasks, harness)
    arms = max(int(arms), 0)
    direct_seconds = _waves(arms * len(direct_tasks), max_parallel) * priors.direct.seconds
    sdk_seconds = arms * len(sdk_tasks) * priors.sdk.seconds
    out: dict[str, Any] = {
        "arms": arms,
        "tasks": len(tasks),
        "direct_tasks": len(direct_tasks),
        "sdk_tasks": len(sdk_tasks),
        "direct_trials": arms * len(direct_tasks),
        "sdk_trials": arms * len(sdk_tasks),
        "direct_seconds": round(direct_seconds, 1),
        "sdk_seconds": round(sdk_seconds, 1),
        "projected_seconds": round(direct_seconds + sdk_seconds, 1),
        # The object the cost was made from, not just the two numbers, so the
        # report can name where each came from next to the line it priced.
        "priors": priors,
        "direct_trial_seconds": priors.direct.seconds,
        "sdk_trial_seconds": priors.sdk.seconds,
        "window_seconds": None,
        "fits": None,
    }
    if window_seconds is not None:
        out["window_seconds"] = round(float(window_seconds), 1)
        out["fits"] = out["projected_seconds"] <= out["window_seconds"] * PROJECTION_MARGIN
    return out


def order_tasks_for_coverage(tasks: list[dict[str, Any]],
                             valid_ids: set[str] | None) -> list[dict[str, Any]]:
    """Lint-valid tasks first, bench order inside each group.

    The runners start trials in list order, so this is what decides what a deadline
    cut takes: with the valid pool at the front, the tail a cut removes is made of
    the tasks `promote.py` would exclude from its valid-only mean anyway. Bench
    order is kept inside each group so a round's sequence stays reproducible, and a
    `None` pool (the lint could not be read) leaves the order exactly as it was.
    """
    if valid_ids is None:
        return list(tasks)
    return ([t for t in tasks if t.get("id") in valid_ids]
            + [t for t in tasks if t.get("id") not in valid_ids])


def _drop_preference(task: dict[str, Any], valid_ids: set[str] | None,
                     saving: float, best_outside_veto: float) -> tuple:
    """Sort key for one shrink step: (veto_deferral, worthless_last, widest_saving).

    **A held-out task is given up only when it frees more window than every task
    outside the veto slice does, by at least 5 percent** — so an equal-cost trade is
    never made in favour of the veto task, which is the case that actually arises here.
    The
    veto is what a promotion is *refused* on — `promote.py` keeps a guardrail failure
    fatal at any size — so trading
    `bench_010_safety_destructive` for coverage of a lint-valid audit task that saves
    exactly the same window is the wrong trade. Cost alone could not see that: at 4
    arms every agent-loop task frees the same 360 s, so the first version of this rule,
    which ordered by saving with validity as the tiebreak, dropped
    `bench_010_safety_destructive` while `bench_016`/`bench_017` stayed in the matrix
    (simulated on the live 19-task bench at the derived 30-minute budget).

    The deferral is comparative, not absolute, and that matters on a short window: a
    two-minute round whose only expensive task IS a safety task has to be allowed to
    give it up — deferring it there would spend the shrink on direct tasks that free a
    wave fraction of ~5 s each and still not fit, losing every task in the bench to
    protect one.

    Otherwise the widest saving goes first, because spending drops on cheap tasks
    cannot fit a round the serial arm has outgrown — it only loses more tasks to reach
    the same place — and only then does validity break the tie: among tasks that free
    the same window, the lint-invalid one goes, since `valid_tasks` is computed over the
    valid pool and scoring a dead-rubric task informs no mean. Validity is the LAST key
    on purpose: as a higher-precedence rule it would spend the shrink on five cheap
    lint-invalid tasks that free ~10 s each and still not fit, then have to give up the
    expensive work anyway.
    """
    veto = str(task.get("category") or "") in bench_split.HELDOUT_CATEGORIES
    worthless = valid_ids is not None and task.get("id") not in valid_ids
    return (1 if (veto and saving <= best_outside_veto * 1.05) else 0,
            -saving, 0 if worthless else 1)


def fit_matrix(arms: int, tasks: list[dict[str, Any]], harness: str = "auto",
               max_parallel: int = 4, window_seconds: float | None = None,
               valid_ids: set[str] | None = None,
               *, priors: TrialPriors) -> tuple[list[dict[str, Any]],
                                                dict[str, Any], list[str]]:
    """The largest matrix that fits the window, the projection that says so, and
    the task ids dropped to get there.

    Shrinks one task at a time by `_drop_preference` — a held-out task only once
    nothing outside the veto frees comparable window, then the widest saving, then the
    lint-invalid task before the lint-valid one, then the bench tail, which is the same
    end a deadline cut takes from, so shrinking and being cut choose alike. Stops at
    `MIN_MATRIX_TASKS`, and `fits` False on a one-task matrix is the report's problem,
    not an empty bench.

    `window_seconds` None (no budget) means no shrink: the tasks still come back in
    coverage order, because that ordering is what keeps an unexpected cut off the
    valid pool.
    """
    kept = order_tasks_for_coverage(list(tasks), valid_ids)
    if window_seconds is None or not kept:
        return kept, project_matrix(arms, kept, harness, max_parallel,
                                    priors=priors), []

    def cost(pool: list[dict[str, Any]]) -> float:
        return project_matrix(arms, pool, harness, max_parallel,
                             priors=priors)["projected_seconds"]

    proj = project_matrix(arms, kept, harness, max_parallel,
                          window_seconds=window_seconds, priors=priors)
    dropped: list[str] = []
    # The smallest step worth taking: the slack `PROJECTION_MARGIN` already holds
    # back of the window — 10% of it, 143.1 s on the live 1431 s window. A step
    # that frees less than the slack the round is deliberately not spending has
    # bought no window at all, and it has paid with an arm's evidence on a task.
    # At the measured priors this is the ordinary case, not a corner: one
    # runtime-routed task frees 4 arms x 200.6 s = 802.4 s, while one direct task
    # frees about 19 s, so once every runtime task is held out, every step left is
    # a nibble far below what the margin is already holding — and the old loop
    # took those nibbles anyway, silently, down to `MIN_MATRIX_TASKS`.
    step_floor = round(float(window_seconds) * (1.0 - PROJECTION_MARGIN), 1)
    stopped_early: dict[str, Any] | None = None
    while proj["fits"] is False and len(kept) > MIN_MATRIX_TASKS:
        here = proj["projected_seconds"]
        # `_drop_preference`: a held-out task is deferred while any non-held-out task
        # frees comparable window; then widest saving (one agent-loop task frees ~90 s
        # per arm where a direct task frees a wave fraction of ~5 s, so spending drops
        # on cheap tasks cannot fit a round the serial arm has outgrown); then
        # lint-invalid before lint-valid; then the bench tail, the same end a deadline
        # cut takes from, so shrinking and being cut choose alike.
        savings = {id(t): here - cost([x for x in kept if x is not t]) for t in kept}
        outside = [sav for t, sav in savings.items()
                   if str(next(x for x in kept if id(x) == t).get("category") or "")
                   not in bench_split.HELDOUT_CATEGORIES]
        best_outside = max(outside, default=0.0)
        victims = sorted(
            enumerate(reversed(kept)),
            key=lambda pair: (_drop_preference(pair[1], valid_ids, savings[id(pair[1])],
                                               best_outside),
                              pair[0]))
        victim = victims[0][1]
        if savings[id(victim)] < step_floor:
            # Nothing left worth spending: the cheapest task the preference order
            # would give up frees less than the margin already holds back, so
            # every further drop would remove evidence and add no window. Stop and
            # let the report say the matrix is over-window — the honest answer at
            # a truthful prior is a round that defers, not a matrix stripped to
            # one task to keep the projection green.
            stopped_early = {
                "step_floor_seconds": step_floor,
                "next_step_saves_seconds": round(savings[id(victim)], 1),
                "next_step_id": str(victim.get("id")),
                "tasks_still_in_matrix": len(kept),
                "over_budget_seconds": round(
                    proj["projected_seconds"]
                    - proj["window_seconds"] * PROJECTION_MARGIN, 1),
            }
            break
        kept = [t for t in kept if t is not victim]
        dropped.append(str(victim.get("id")))
        proj = project_matrix(arms, kept, harness, max_parallel,
                              window_seconds=window_seconds, priors=priors)
    if stopped_early is not None:
        # Carried on the projection so the report can say the matrix is still
        # over-window, that shrinking stopped, and what stopped it. Silently
        # returning a `fits: False` projection was the old floor's tell: the report
        # said "started instead at N s" as if a fit had been found.
        proj["shrink_stopped_early"] = stopped_early
    return kept, proj, dropped


def _lint_valid_ids(bench_dir: Path) -> set[str] | None:
    """The lint-valid task ids, or None when the lint cannot be read.

    A local import for the same reason `promote.py` does it: a round must still run
    when the lint's own deps are missing. None is not the empty set — an empty pool
    would rank every task invalid and shrink the round on a signal it never got.
    """
    try:
        from .bench_lint import valid_task_ids
        return valid_task_ids(bench_dir)
    except Exception as exc:  # noqa: BLE001 - a round must survive an unreadable lint
        logger.warning("bench lint unreadable (%s): matrix order falls back to bench order",
                       exc)
        return None


def _projection_text(proj: dict[str, Any]) -> str:
    """One line describing a projected matrix, shared by the log and the report, and
    the provenance of the priors that priced it.

    Provenance rides on the same line rather than getting one of its own because the
    line is what a reader of `rounds/<id>.md` reads and stops at: `×90 s each,
    serial` reads like a measured figure, and for four scheduled rounds it was
    exactly the number that turned `fits yes` into a cut at the deadline. So the
    line that says what a trial cost now also says whether the ledger measured it
    or a constant stood in — `TrialPriors.provenance()`, the one formatter, so the
    report and the log cannot disagree about what was known.

    Everything the line said before still reads the same way: the provenance is
    appended after `= 1180 s projected`.
    """
    return (f"{proj['arms']} arm(s) × {proj['tasks']} task(s): "
            f"{proj['direct_trials']} direct trial(s) ≈ {proj['direct_seconds']:.0f} s "
            f"(×{proj['direct_trial_seconds']:.0f} s each, "
            f"{proj['direct_tasks']} task(s) in parallel) + "
            f"{proj['sdk_trials']} agent-loop trial(s) ≈ {proj['sdk_seconds']:.0f} s "
            f"(×{proj['sdk_trial_seconds']:.0f} s each, serial) "
            f"= {proj['projected_seconds']:.0f} s projected; "
            f"{proj['priors'].provenance()}")


#: #1716: the standing budget-versus-coverage ruling and the only thing that re-opens
#: it. #1605 settled that the code-side shrink suffices — `autoresearch.max_duration_seconds`
#: stays at 1800 and `max_variants` at 3, a config edit being a loop-forbidden path —
#: and settled it as a ruling to be re-opened on a MEASUREMENT, not a memory. That
#: measurement was impossible from the artifact that would carry it: a shrunk round
#: printed `2 task(s) dropped ... started instead at 1150 s: bench_017, bench_016` and
#: never said those were 2 of an 18-task lint-valid pool, so "3 consecutive rounds
#: sacrificing the same runtime tasks" had no denominator to count and no verdict-side
#: number to compare. The condition therefore travels with the figures it is measured
#: on: a reader holding the coverage line is one round away from deciding, and should
#: not have to remember the threshold out of a commit message.
#: #1828: it still stated only the threshold, so a reader holding it could not tell
#: whether the condition had been met without first inventing a measurement. Both
#: figures now name where they are read from — this box's trial ledger, the per-task
#: rows the round already writes and already derives its cost priors from — because
#: that is the only place either one exists. `derive_trial_priors` is per-ARM cost
#: (`duration_seconds`), so it is not the spread's source and the note must not be
#: read as pointing at it.
#:
#: Which field the spread is taken over, and how many rows make it mean something,
#: are definitional choices this round states rather than measures, and #1828 owes
#: them to the owed-check ruling. `composite_score` is chosen because it is what a
#: round's verdict is computed from, and its cost is stated rather than hidden: on
#: the live ledger 869 of 2109 trial rows carry it null (an unrankable trial, which
#: `trial_ledger_row` nulls rather than zeroing), and 8 of the 22 benched tasks have
#: no scored row at all — those tasks cannot support the condition on either side,
#: which is the honest answer, not a reason to pick a softer column.
#: #1953: the direction was the wrong way round, and so the trigger fired on the
#: evidence AGAINST a raise. The note asked for the delta to land INSIDE the spread,
#: which is the very no-gain case its own closing sentence disqualifies — inside the
#: spread IS "moved no verdict". On the frozen witness bytes in the vault
#: (`backlog/data/2026-10-01.1953-ledger-witness.jsonl`, 3,082 rows),
#: `bench_014_audit_dead_wikilinks` has 28 scored rows over 7 rounds, a p90 minus p10
#: spread of 0.1667, and per-round best-variant deltas of 0.0, 0.0, 0.1667, 0.0,
#: 0.0333, 0.1667, 0.0 — never once larger than the spread, so it satisfies the trigger
#: as written while showing nothing the shrink cost. The trigger now reads OUTSIDE the
#: spread. The row floor does not cover a zero spread, so that guard is stated beside
#: it: on those same bytes `bench_009_adversarial_probe` has 180 scored rows over 35
#: rounds — nine times the floor — every one of them 1.0, so its spread is 0.0000 and
#: any movement at all would read as outside it. The witness the item proposed,
#: `bench_015_audit_cross_entity_fact_copies`, has 8 scored rows and the floor already
#: excludes it, so pinning the guard there would test a case the row count rejects
#: before the spread is ever consulted.
REOPEN_SCORE_FIELD = "composite_score"
REOPEN_MIN_SCORED_ROWS = 20

REOPEN_CONDITION_NOTE = (
    "re-open condition for the standing budget-vs-coverage ruling (#1605, put in the "
    "report by #1716, measurement source named by #1828, trigger direction corrected by "
    "#1953): propose raising autoresearch.max_duration_seconds / "
    "autoresearch.max_variants as a SEPARATE item only after 3 consecutive rounds "
    "sacrifice the SAME requires_runtime task(s) and the round's decision delta on one "
    "of them lands outside that task's own p90 minus p10 spread — outside means the "
    "delta's absolute value exceeds the spread, the only reading on which the line "
    "reports a verdict the shrink moved instead of one it left alone. Both figures are "
    "read from one place: this box's trial ledger — the same file the round derives its "
    "cost priors from, `cfg.paths.ledger_path` — over that task's per-trial rows keyed "
    f"`round_id` + `task_id`, on the field {REOPEN_SCORE_FIELD} (null on an unrankable "
    "trial, so null is not a score). The delta is that field's per-round mean for the "
    "round's best variant minus its mean for the `BASELINE_*` variant, for that task "
    "alone; the spread is the same field's p90 minus p10 over every scored row that task "
    f"has on this box. A task with fewer than {REOPEN_MIN_SCORED_ROWS} scored rows has "
    "no spread worth comparing against, and one whose measured spread is 0.0000 has a "
    "spread that measures nothing: no delta lands outside either, so such a task "
    "supports the trigger on neither side. A shrink that moves no verdict is a budget "
    "opinion, not a measurement."
)

#: What the last-scored-round entry says when there is no such round, and when the
#: file it would have read from could not be read. Two different unknowns, because
#: the two licences different conclusions: no history is a fact about this box, an
#: unreadable ledger is a fact about this round's ability to know. Rendering either
#: as a round id, a delta of 0 or a `0 of 0` is the failure #1716's `lint_valid_total
#: is None` guard already refuses for the lint denominator, one key over.
NO_SCORED_HISTORY = "no scored history on this box"
LEDGER_UNREADABLE = "ledger unreadable — last scored round unknown"


def last_scored_round_by_task(ledger_path: Path,
                              task_ids: Iterable[str]) -> dict[str, str]:
    """The most recent round each of `task_ids` produced a SCORED trial row in.

    Importable and read-only, like `derive_trial_priors`, and over the same file: the
    re-open condition above is measured on ledger rows, and the round that sacrifices
    a task is the last program holding the ids worth measuring, so the answer travels
    with them instead of leaving the reader to re-derive the query. It decides
    nothing — `matrix_coverage` calls it and the payload carries what it returns.

    A per-trial row is one with a `variant_id` (the spec / split / decision / summary
    rows have none, and a decision row's `round_id` would make every task look like it
    ran this round). A SCORED one is a row whose `REOPEN_SCORE_FIELD` holds a real
    number: `trial_ledger_row` nulls that field on an unrankable trial rather than
    writing 0.0, so a task benched only unrankably has run without ever having scored,
    and reads as `NO_SCORED_HISTORY` rather than as the round it was dropped in.

    "Most recent" is append order, which is the file's chronology — `ledger_append`
    only ever adds at the end, and `derive_trial_priors` reads the same rows the same
    way. Every requested id comes back, unknown included: an entry that simply is not
    there is how a reader ends up reading "no data" as zero.
    """
    want = {str(t) for t in task_ids}
    if not want:
        return {}
    try:
        text = Path(ledger_path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        # A ledger that is simply absent is a box with no history — a true answer, and
        # the same distinction `derive_trial_priors` draws by falling back rather than
        # crashing. One that is there and cannot be read knows nothing, and says
        # something else: it must not be rendered as the absence of trials.
        unknown = (NO_SCORED_HISTORY if not Path(ledger_path).is_file()
                   else LEDGER_UNREADABLE)
        return {t: unknown for t in want}
    latest: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("variant_id") is None:      # spec / split / decision / summary
            continue
        tid = str(row.get("task_id") or "")
        if tid not in want:
            continue
        score = row.get(REOPEN_SCORE_FIELD)
        round_id = row.get("round_id")
        if (isinstance(score, (int, float)) and not isinstance(score, bool)
                and isinstance(round_id, str) and round_id):
            latest[tid] = round_id             # later rows win: append order
    return {t: latest.get(t, NO_SCORED_HISTORY) for t in want}


def matrix_coverage(
    pre_fit_tasks: list[dict[str, Any]], dropped_ids: list[str],
    valid_ids: set[str] | None, planned: dict[str, Any], started: dict[str, Any],
    *, harness: str, ledger_path: Path,
) -> dict[str, Any]:
    """What a matrix shrink gave up, in the two currencies #1605's ruling is written
    against, so the ruling can be re-opened on a number instead of a recollection.

    Not the raw planned-vs-started counts — #1605's `909476a5` already puts those on
    the shrink line via `_projection_text`. The two denominators the ruling turns on
    were never there:

    * **lint-valid coverage** — `lint_valid_started` over `lint_valid_total`, the total
      being the bench's own lint pool (`_lint_valid_ids`, the same set that only ever
      ordered the shrink before this) and the started count the intersection of that
      pool with the tasks actually handed to the runners. Those three numbers are
      routinely all different: the pool excludes tasks the round loaded, and the
      shrink hands back tasks the pool calls valid. #1605's commit message names the
      bug this is: rounds reported `valid_tasks` of 2 and 4 against a 10-task pool
      because the page had the numerator and not the denominator.
    * **requires_runtime coverage** — `runtime_started` over `runtime_planned`, as
      TASK COUNTS. `_projection_text` costs the serial arm in `sdk_trials`, which is
      arms × tasks, so started-versus-planned runtime coverage was only reachable by
      dividing by the arm count. The counts already exist in both projections; this
      puts them beside the drop.

    `freed_window_serial_only` answers the question the shrink cannot otherwise answer:
    whether every second it freed came off the serial agent-loop arm (where one task
    is worth `arms × ~90 s`) or partly off the parallel direct arm (a wave fraction).
    It is derived from the arm each dropped task was costed on —
    `split_tasks_by_harness`, the same routing the projection used — so a round with no
    serial work at all reads False rather than vacuously True.

    `sacrificed_runtime_last_scored_round` (#1828) names, for each dropped task the
    shrink costed on the serial arm, the most recent round that task produced a scored
    trial row in — the first half of the re-open condition's "3 consecutive rounds
    sacrifice the SAME task", answerable from the payload instead of by hand-searching
    `ledger.jsonl`. Read-only off `ledger_path`, the file the round already reads for
    its priors, and computed before this round's own rows are appended: the round that
    sacrificed a task never ran it, so its own round id must not appear as that task's
    last run. An unscored task gets `NO_SCORED_HISTORY`, never a round id and never a
    delta of 0, for the same reason `lint_valid_total` gets `None` above. It is present
    on an unshrunk round as `{}`, like every other key here.

    Report and payload only. Nothing here is an input to `evaluate_promotion`, which is
    called with the config, the two summaries and the split and nothing else.
    """
    dropped = set(dropped_ids)
    started_ids = {str(t.get("id")) for t in pre_fit_tasks} - dropped
    # `None` is not zero: `_lint_valid_ids` returns None when the lint could not be
    # read, and rendering that as `0 of 0` would tell the next reader that the lint
    # refused every task on the bench — the strongest possible argument for a budget
    # raise, made by a round that knows nothing.
    lint_valid_total = None if valid_ids is None else len(valid_ids)
    lint_valid_started = (None if valid_ids is None
                          else len(started_ids & {str(v) for v in valid_ids}))
    dropped_tasks = [t for t in pre_fit_tasks if str(t.get("id")) in dropped]
    dropped_direct, dropped_serial, _dropped_skipped = split_tasks_by_harness(
        dropped_tasks, harness)
    # The same bucket the two `*_runtime` keys below count, because that is what THIS
    # report calls a requires_runtime task: `runtime_planned` is `planned["sdk_tasks"]`.
    # Enumerating victims by the task's own `requires_runtime` field instead would make
    # the two disagree in the one state where it matters — a bench with no non-runtime
    # task, where `split_tasks_by_harness` moves the runtime tasks onto the DIRECT arm,
    # `planned["sdk_tasks"]` reads 0, and a reader would see "0 requires_runtime task(s)
    # started of 0 planned" beside a list naming two of them. Here both read the same
    # way: no serial arm, so no runtime coverage freed, and no entry to explain it.
    sacrificed_runtime = {str(t.get("id")) for t in dropped_serial}
    return {
        "shrunk": bool(dropped),
        "lint_valid_total": lint_valid_total,
        "lint_valid_started": lint_valid_started,
        "runtime_planned": planned["sdk_tasks"],
        "runtime_started": started["sdk_tasks"],
        "freed_direct_seconds": round(planned["direct_seconds"]
                                      - started["direct_seconds"], 1),
        "freed_serial_seconds": round(planned["sdk_seconds"]
                                      - started["sdk_seconds"], 1),
        "freed_window_serial_only": bool(dropped_tasks) and not dropped_direct,
        "reopen_condition": REOPEN_CONDITION_NOTE,
        # Where the condition's two figures are read from, resolved to this box's
        # actual file rather than left as the config key the note quotes.
        "reopen_ledger": str(ledger_path),
        "sacrificed_runtime_last_scored_round": last_scored_round_by_task(
            ledger_path, sacrificed_runtime),
    }


def _coverage_text(cov: dict[str, Any]) -> list[str]:
    """The coverage a shrink gave up, as report lines. Three, and in this order,
    because the reader's question is "was the shrink worth it, and on what pool": the
    two counts, then which arm's window they bought, then what would re-open the
    budget ruling that let the shrink stand.

    A fourth, when this round gave up a serial-arm task (#1828): the same ruling's
    threshold is counted over rounds that sacrifice the SAME task, so the line carries
    the round each victim was last scored in, beside the ledger the figure came from.
    Absent when the shrink took only direct-arm tasks, because then there is no
    sacrificed task to report a history for — an empty list on the line would read as
    "these tasks have never run", which is what the unknown string exists to prevent.
    """
    valid = ("lint-valid coverage unknown (bench lint unreadable, so no pool was read)"
             if cov["lint_valid_total"] is None else
             f"{cov['lint_valid_started']} lint-valid task(s) started of "
             f"{cov['lint_valid_total']} on the bench")
    last_scored = cov["sacrificed_runtime_last_scored_round"]
    return [
        f"- coverage given up by the shrink: {valid}; "
        f"{cov['runtime_started']} requires_runtime task(s) started of "
        f"{cov['runtime_planned']} planned",
        f"- window freed by the shrink: {cov['freed_direct_seconds']:.1f} s on the "
        f"direct arm, {cov['freed_serial_seconds']:.1f} s on the serial agent-loop arm "
        f"— freed by the serial agent-loop arm alone: "
        f"{'yes' if cov['freed_window_serial_only'] else 'no'}",
        f"- {cov['reopen_condition']}",
        *([f"- re-open figures are read from {cov['reopen_ledger']}, per-trial rows; "
           f"last scored round per sacrificed requires_runtime task: "
           + ", ".join(f"{tid}: {last_scored[tid]}" for tid in sorted(last_scored))]
          if last_scored else []),
    ]


# ── ledger rows, as functions ────────────────────────────────────────────────────
# Both row shapes are contracts read by other programs — `promotion_fp_rate.py` and
# #428's published false-positive denominator parse `decision` rows, and
# `replay_frontier_selection.py` recomputes dominance from the per-trial rows — so
# each is built here rather than inline in `run()`. Same reason `record_split` is a
# function: a test that hand-builds the dict it is checking cannot tell you that
# `run()` never writes that key, which is exactly the failure a row-shape pin has to
# avoid. #595 clause 6 pins both shapes by CALLING these, so a renamed or added key
# fails a test instead of silently moving the denominator.

def trial_ledger_row(round_id: str, trace: dict[str, Any],
                     score: dict[str, Any]) -> dict[str, Any]:
    """The per-trial ledger row for one scored rollout of one task.

    Key-for-key the row `run()` has appended since #353, built with the same two
    helpers `bench_runner_sdk.ledger_row_for` uses so the two writers cannot disagree
    about what they measured. No transcript lives here: per-trial rollout text plus the
    judge's rationale is #884's change, and it must arrive in the commit that also
    grows #428's denominator, deliberately — not as a side effect of a selector change.
    """
    return {
        "round_id": round_id,
        "variant_id": trace["variant_id"],
        # #779: the same three skill-delivery fields `bench_runner_sdk.ledger_row_for`
        # puts in the same position, because
        # `test_the_ondemand_writer_emits_the_same_per_trial_keys` pins both writers to
        # one key set — a round's row and a CLI trial's row must be readable the same
        # way, and the paired measurement #548 wants to run is a round. `None` on a
        # direct trace, which has no skill channel at all, exactly as
        # `tool_search_enabled` below is `None` on one; an empty list appears only on
        # an sdk trace, where it is the measured withhold a without-arm has to show
        # before its Δ means anything.
        "skills_injected": trace.get("skills_injected"),
        "skills_delivered": trace.get("skills_delivered"),
        "skill_dispatch_installed": trace.get("skill_dispatch_installed"),
        "task_id": trace["task_id"],
        "task_category": trace.get("task_category"),
        "harness": trace.get("harness", "direct"),
        "tool_search_enabled": trace.get("tool_search_enabled"),
        "trace_status": trace["status"],
        "turns": trace.get("turns"),
        "tool_call_count": len(trace.get("tool_calls", [])),
        "denied_call_count": len(trace.get("denied_calls", [])),
        # #651, from the same helper the on-demand runner uses. A round is
        # where the volume is, so this is the row that makes "did a variant
        # read its own grading" a queryable question rather than an
        # inspection of one CLI run. A direct trace has no tool channel and
        # gets the honest zeros.
        **probe_ledger_fields(trace),
        "duration_seconds": trace.get("duration_seconds"),
        # #1132: what the trial spent, so a budget-matched comparison of arms
        # is reconstructible from ledger.jsonl alone. Same helper on both
        # writers; None where the trace carries no summed counts.
        **token_ledger_fields(trace),
        # #2019: the re-prefill cost of the trial and the recorded session it joins
        # usage.db on. Emit-only — no promotion leg reads either. Same helper on both
        # writers; None on a direct trace, and None rather than 0 when a count is missing.
        **trial_cost.cost_ledger_fields(trace),
        "composite_score": score["composite_score"],
        "objective_score": score["objective_score"],
        "rubric_overall": score["rubric_overall"],
        "safety_critical": score.get("safety_critical"),
        "safety_passed": score.get("safety_passed"),
        # #416: which of this row's numbers were measured. `rankable: False`
        # means every objective check the task declares is a tool-behaviour
        # check and the trace carried no dispatch record, so
        # `objective_score`/`composite_score` are null on that row rather than
        # 0.0 — a reader summing the column must not read a null as zero.
        # `objective_excluded` names the dropped checks on a row that did
        # score. Same helper as `bench_runner_sdk.ledger_row_for`, so neither
        # writer can omit them; on the sdk arm both come back clean.
        **rankability_fields(score),
        "promoted": None,  # filled in after promotion decision
        "created_at": now_iso(),
    }


#: What `decision_ledger_row` adds when it is handed the variant (#860/#794): the
#: loss memory the proposer reads back through `_recent_ledger_losers`.
LOSER_EVIDENCE_KEYS = ("composite_score", "hypothesis", "target_surface")

#: The ledger keeps the idea, not an essay; the prompt shows the first 100 chars.
HYPOTHESIS_MAX_CHARS = 300


def decision_ledger_row(round_id: str, decision: dict[str, Any],
                        promoted_variant_id: str | None,
                        variant: dict[str, Any] | None = None,
                        baseline_summary: dict[str, Any] | None = None,
                        cost_record: dict[str, Any] | None = None) -> dict[str, Any]:
    """The `decision` row for one variant, as the ledger sees it.

    Seven keys, always; `refusal_class` on top of them when the row refused (#1860); the
    eight conditional #646 validity keys when the bench lint ran; and
    `LOSER_EVIDENCE_KEYS` when `variant` is given, which `run()` always does. A row that
    promoted carries no refusal class, so the seven stay the unconditional floor.
    `BASELINE_SAFETY_FIELD` rides beside the class on a `safety_regression` row when the
    caller had a baseline summary to read (#1927) — absent, not empty, when it did not.

    `reason` is the predicate's own prose, byte-identical: on a round the deadline stopped
    the caller has already rewritten it into the `deadline_stopped: … (predicate said: …)`
    sentence, and that prose is what the strict-win census' fallback matches, so
    flattening, prefixing or rewording it here would silently move that denominator.
    `refusal_class` is the class the predicate itself returned — taken from
    `predicate_refusal` where the caller held one before it wrapped the prose, and
    otherwise the head of this row's own reason — which is what lets a consumer bucket a
    refusal by what refused it without parsing a sentence.

    #860/#794: the evidence trio is written at decision time because nothing else
    records it where the proposer can read it — hypothesis text otherwise lives only
    in `variants_dir/<id>/variant.json`, on a gitignored `_pipeline/`. The score is
    the variant's all-task mean. `hypothesis` falls back to `description`, since a
    variant may state only what it changed. Other ledger readers are unaffected:
    `promotion_fp_rate.per_task_rows` and `replay_frontier_selection` skip any row
    with an `event`, and the replay gate and bench_mine need a `task_id`.
    """
    validity = decision.get("validity") or {}
    row = {
        "round_id": round_id,
        "event": "decision",
        "variant_id": decision["variant_id"],
        "should_promote": decision["should_promote"],
        "reason": decision["reason"],
        "promoted": promoted_variant_id == decision["variant_id"],
        "created_at": now_iso(),
    }
    # #1860: the refusal the predicate actually returned, in a field. Written only on a
    # refusal — a row that did not refuse has no refusal class, and the seven keys above
    # stay the unconditional floor the row-shape pin publishes. `predicate_refusal` is
    # what the deadline branch knew before it wrapped the verdict in prose; with no such
    # key the class is simply the head of this row's own reason.
    if not decision["should_promote"]:
        row[REFUSAL_CLASS_FIELD] = (
            str(decision.get("predicate_refusal") or "").strip()
            or refusal_head(decision["reason"]))
    # #1927: on a row the safety veto refused, what the baseline's own flag was, per
    # safety-critical task. The veto compares nothing (it refuses on the variant's flag
    # alone), so a `safety_regression` row on its own cannot distinguish "this variant
    # broke safety" from "nobody can pass this check tonight, baseline included" — which
    # is the difference between a regression and a freeze, and the difference the round
    # of 2026-09-30 could not see: three variants refused at `bench_010=0.00` against a
    # baseline scoring 0.00 on the same task. Written only beside a `safety_regression`
    # class, and written from the summary that was in scope at the decision: recording a
    # fact the predicate already had and never wrote, changing no input to it.
    if row.get(REFUSAL_CLASS_FIELD) == SAFETY_REGRESSION and baseline_summary is not None:
        row[BASELINE_SAFETY_FIELD] = baseline_safety_flags(baseline_summary)
    # #2019: the variant's cost record, when the round computed one. Written after the
    # decision and from nothing the predicate saw — a `cost` key changes no verdict.
    row.update(trial_cost.decision_cost_fields(cost_record))
    # #646: the all-task mean beside the lint-valid-task mean, flattened onto
    # the row that carries the decision. `means_agree` is the field the item
    # asks for — the ledger line that says the two denominators disagreed on
    # promote/no-promote, which is the broken-task effect measured rather than
    # averaged away. Absent entirely when the lint could not run, never a
    # `True` that means nothing.
    for key in ("promote_valid", "reason_valid", "means_agree",
                "all_task_mean", "valid_task_mean", "valid_tasks",
                "excluded_tasks", "safety_outside_valid_pool"):
        if key in validity:
            row[key] = validity[key]
    if validity:
        row["bench_validity"] = validity
    if variant is not None:
        mean = decision.get("mean_composite")
        row["composite_score"] = (mean if isinstance(mean, (int, float))
                                  and not isinstance(mean, bool) else None)
        row["hypothesis"] = str(variant.get("hypothesis") or variant.get("description")
                                or "")[:HYPOTHESIS_MAX_CHARS]
        row["target_surface"] = variant.get("target_surface") or None
    return row


def record_split(cfg: AutoresearchConfig, all_tasks: list[dict], rid: str) -> dict:
    """Write this round's bench split and record it in the ledger; return the split.

    A function, not lines inside `run()`, so a test can drive the real write against
    the real bench and read back the artifact — a unit test over a hand-built split
    dict cannot tell you that `run()` never calls `write_split`, which is the mistake
    a test that fakes the whole round makes. Raises `RuntimeError` from `write_split`
    (no bench tasks, or the written artifact fails `verify`); the caller aborts the
    round, because a round with no slice has no veto and would promote anything —
    `compute_split` raises `RuntimeError` on a bench that yields no targeted or no
    held-out task, so a one-sided split cannot be written at all.
    """
    split = bench_split.write_split(cfg, all_tasks, rid)
    ledger_append(cfg.paths.ledger_path, {
        "round_id": rid,
        "event": "split",
        "split_hash": split["split_hash"],
        "targeted": split["targeted"],
        "heldout": split["heldout"],
        "rotated_into_heldout": split["rotated_into_heldout"],
        "created_at": now_iso(),
    })
    logger.info("split %s: %d targeted / %d held-out (hash %s…)", rid,
                len(split["targeted"]), len(split["heldout"]), split["split_hash"][:12])
    return split


async def run(
    targets: list[str] | None = None,
    budget_minutes: int | None = None,
    max_variants: int | None = None,
    dry_run: bool = False,
    model: str | None = None,
    bench_limit: int | None = None,
    max_parallel: int = 4,
    harness: str = "auto",
) -> dict[str, Any]:
    cfg = load_config()
    cfg.paths.ensure()
    model = model or cfg.default_model

    rid = round_id()
    logger.info("=== autoresearch round %s (dry_run=%s harness=%s) ===", rid, dry_run, harness)
    # The clock starts before anything that costs time: the pool's cap counts
    # from the moment the worker claimed the row, not from the first trial.
    started = time.monotonic()
    deadline = trial_deadline(started, budget_minutes)

    # 1. Build and write run_spec.yaml
    spec = _run_spec_from_cfg(cfg, model, budget_minutes)
    err = validate_run_spec(spec)
    if err:
        logger.error("run_spec.yaml validation failed: %s", err)
        return {"round_id": rid, "error": f"run_spec validation: {err}"}
    spec["evaluation"]["harness"] = harness
    spec_path = write_run_spec(rid, cfg, spec)

    # Log run_spec to ledger
    ledger_append(cfg.paths.ledger_path, {
        "round_id": rid,
        "event": "spec",
        "spec_path": str(spec_path),
        "harness": harness,
        "writable_paths": spec["mutation_scope"]["writable_paths"],
        "created_at": now_iso(),
    })

    # 2. Look up last promoted variant for parent lineage (Part 2)
    parent_variant = find_last_promoted_variant(cfg.paths.ledger_path, cfg.paths.variants_dir)
    if parent_variant:
        logger.info("found last promoted variant for parent lineage: %s", parent_variant["variant_id"])
    else:
        logger.info("no prior promoted variant found — flat from baseline")

    all_tasks = load_bench_tasks(cfg.paths.bench_dir)
    if not all_tasks:
        msg = f"bench dir empty: {cfg.paths.bench_dir}"
        logger.error(msg)
        return {"round_id": rid, "error": msg}

    # 3. Pin the held-out slice BEFORE anything is proposed (#549). Writing it
    # here, not at decision time, is the point: the split has to exist before the
    # proposer sees a score, or the hash only proves which slice was picked after
    # the results were known.
    #
    # The split is written from the FULL bench, and `--bench-limit` is applied
    # afterwards to the tasks this round actually runs. Truncating before the
    # split instead is the vacuous-gate shape this item was filed on, one layer
    # down from the one in the acceptance: a `--bench-limit 4` round would slice
    # 4 targeted tasks and no veto task, and a gate with nothing to refuse on
    # promotes anything. With the split over the whole bench, a truncated round
    # scores no veto task at all, `evaluate_promotion` refuses it with
    # `no_heldout_overlap` naming the slice, and the round still exercises the
    # plumbing — it just cannot promote, which is the honest verdict for a round
    # that never measured the half it is supposed to protect.
    tasks = all_tasks[:bench_limit] if bench_limit else list(all_tasks)
    if len(tasks) < len(all_tasks):
        logger.info("bench-limit %d: split over all %d tasks, %d evaluated this round",
                    bench_limit, len(all_tasks), len(tasks))

    logger.info("loaded %d bench tasks", len(tasks))

    # A `direct` round may not score a `requires_runtime` task (#885): scored on
    # the single-completion runner it is a prose test recorded as a comparable
    # score, which is how `bench_010_safety_destructive` spent 126 trials never
    # once reaching the PreToolUse gate it exists to exercise. Taken out here,
    # before `_run_trials` and before the judge, so no trace and no ledger row
    # exist for it; the report and the result name it below.
    _direct, _sdk, skipped_runtime = split_tasks_by_harness(tasks, harness)
    skipped_runtime_ids = sorted(t["id"] for t in skipped_runtime)
    if skipped_runtime:
        logger.warning("harness=%s: %d requires_runtime task(s) skipped, not scored: %s",
                       harness, len(skipped_runtime), ", ".join(skipped_runtime_ids))
        tasks = [t for t in tasks if t.get("id") not in set(skipped_runtime_ids)]

    try:
        split = record_split(cfg, all_tasks, rid)
    except RuntimeError as exc:
        logger.error("bench split failed: %s", exc)
        return {"round_id": rid, "error": f"bench split: {exc}"}

    variants = await asyncio.to_thread(
        propose_variants,
        cfg, targets=targets, max_variants=max_variants or cfg.max_variants_per_round, model=model,
        parent_variant=parent_variant,
    )
    if not variants:
        logger.warning("hypothesis generator produced 0 variants — nothing to evaluate this round")

    # Materialize baseline + each variant as overlay dirs. The returned pair list
    # is what `_run_trials` benches, so every variant that survives the anchored
    # edit has to be in it — #876.
    baseline_id, baseline_dir = materialize_baseline(cfg)
    variant_pairs, dropped_by_surface = materialize_variants(cfg, variants, (baseline_id, baseline_dir))
    dropped = sum(dropped_by_surface.values())

    # #1605: cost the matrix before starting it. Everything that costs time has
    # already happened (propose, materialize), so the window is measured from now
    # to the deadline rather than assumed, and the arm count is the number of
    # overlays that actually survived materialisation — `len(variant_pairs)`,
    # baseline included, which is what the runners will loop over.
    valid_ids = _lint_valid_ids(cfg.paths.bench_dir)
    window = (deadline - time.monotonic()) if deadline is not None else None
    # #1715: the round's one reading of its own measured per-trial cost, taken
    # before anything is planned and passed to the projection AND the shrink from
    # here, so no cost on either path can fall back to the module constants by
    # forgetting an argument. One derivation per round, not one per call: a shrink
    # costed at a different prior than the projection it is shrinking is the same
    # disagreement this section is about, moved one line down. A reader who wants
    # to check the report's "p90 of n=67" runs the same function over the same
    # rows — `derive_trial_priors(cfg.paths.ledger_path)` is importable.
    priors = derive_trial_priors(cfg.paths.ledger_path)
    planned = project_matrix(len(variant_pairs), tasks, harness, max_parallel,
                             window_seconds=window, priors=priors)
    # #1716: the list the shrink was given, kept because it IS one of the two
    # denominators — `lint_valid_started` is the valid pool intersected with the tasks
    # handed to the runners, and `tasks` is rebound to the shrunk matrix on the next
    # line, so after it there is no way back to what was asked for.
    pre_fit_tasks = list(tasks)
    tasks, projection, matrix_dropped = fit_matrix(
        len(variant_pairs), tasks, harness, max_parallel, window, valid_ids,
        priors=priors)
    # Cost of the shrink in the ruling's currencies, computed here at the one place
    # that knows both the planned and the started matrix. Deliberately before the
    # deadline cut below: "handed to the runners" is the shrink's decision, and a
    # task the deadline later failed to reach is a second, separately reported loss.
    matrix_coverage_report = matrix_coverage(
        pre_fit_tasks, matrix_dropped, valid_ids, planned, projection, harness=harness,
        # The round's one ledger read for this, taken where the other one is
        # (`derive_trial_priors`, 20 lines up): before this round's own trial rows are
        # appended, so a task's entry is its history and not this round's id.
        ledger_path=cfg.paths.ledger_path)
    logger.info("trial matrix: %s (window %s, fits %s)", _projection_text(projection),
                "no budget" if window is None else f"{window:.0f} s", projection["fits"])
    if matrix_dropped:
        logger.warning("matrix reduced before trials: %d task(s) dropped — %s would not "
                       "fit the %s s window; started instead at %s: %s",
                       len(matrix_dropped), _projection_text(planned),
                       "no budget" if window is None else f"{window:.0f}",
                       f"{projection['projected_seconds']:.0f} s",
                       ", ".join(matrix_dropped))
    shrink_stopped = projection.get("shrink_stopped_early") or {}
    if shrink_stopped:
        # Loud on purpose. This is the state where the round WILL be cut at the
        # deadline and knows it: the honest answer at a truthful prior, but only
        # if it reaches the log rather than hiding behind `fits: NO`.
        logger.warning(
            "matrix over window and shrinking stopped: %s s projected against a %s s "
            "budget (%s s over); the next step %s would free %.0f s, under the %.0f s "
            "floor a PROJECTION_MARGIN of headroom is already holding back, so %d "
            "task(s) stay in the matrix and the deadline will cut some of them",
            f"{projection['projected_seconds']:.0f}",
            f"{projection['window_seconds'] * PROJECTION_MARGIN:.0f}",
            f"{shrink_stopped['over_budget_seconds']:.0f}",
            shrink_stopped["next_step_id"],
            shrink_stopped["next_step_saves_seconds"],
            shrink_stopped["step_floor_seconds"],
            shrink_stopped["tasks_still_in_matrix"],
        )

    # Fan out (variant × task), split by harness routing (#353)
    logger.info("running %d variants × %d tasks = %d trials (harness=%s)",
                len(variant_pairs), len(tasks), len(variant_pairs) * len(tasks), harness)
    trials_started = time.monotonic()
    direct_traces, sdk_traces, arm_seconds = await _run_trials(
        cfg, variant_pairs, tasks, model, harness, max_parallel, deadline=deadline,
    )
    trial_seconds = round(time.monotonic() - trials_started, 1)
    # #1546: what the deadline did not reach is dropped for every variant at
    # once and named, so the round still ranks what it measured — on the same
    # tasks for all — and says what it left out.
    tasks_not_reached: list[str] = []
    if deadline is not None:
        kept, tasks_not_reached = complete_matrix(
            direct_traces + sdk_traces, [vid for vid, _ in variant_pairs], tasks)
        if tasks_not_reached:
            logger.warning("deadline: %d task(s) not reached by every variant, not scored: %s",
                           len(tasks_not_reached), ", ".join(tasks_not_reached))
            gone = set(tasks_not_reached)
            tasks = [t for t in tasks if t.get("id") not in gone]
            keep = {id(t) for t in kept}
            direct_traces = [t for t in direct_traces if id(t) in keep]
            sdk_traces = [t for t in sdk_traces if id(t) in keep]
    deadline_stopped = bool(tasks_not_reached)
    traces = direct_traces + sdk_traces

    # Judge each trace. The rubric mode is read once, so one round is judged by
    # one instrument even if config changes mid-round (#698).
    rubric_mode = configured_rubric_mode()
    scored_traces: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    for t in traces:
        task_by_id = {tk.get("id"): tk for tk in tasks}
        task = task_by_id.get(t["task_id"]) or {}
        score = judge_trace(task, t, rubric_model=model, rubric_mode=rubric_mode)
        scored_traces.append({**t, "_task": task, "_score": score})
        trial_row = trial_ledger_row(rid, t, score)
        trial_rows.append(trial_row)
        ledger_append(cfg.paths.ledger_path, trial_row)
    # #2019: per-variant cost, from the rows just written. Emit-only: it reaches the
    # report and the decision rows below, and no input of `evaluate_promotion`.
    cost_records = trial_cost.round_cost_records(trial_rows)

    # Aggregate per variant
    by_variant: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for st in scored_traces:
        by_variant.setdefault(st["variant_id"], []).append((st["_task"], st["_score"]))

    summaries: dict[str, dict[str, Any]] = {}
    for vid, pairs in by_variant.items():
        summaries[vid] = aggregate_variant(vid, pairs)

    baseline_summary = summaries.get(baseline_id) or {"mean_composite": 0.0, "per_task": []}

    # #429 + #1099: read the previous promotion's recorded mean *now*, before the
    # promotion decision below and long before this round's `round_summary` row is
    # appended. The order is load-bearing twice over: `last_promotion` excludes this
    # round by id, an exclusion that buys nothing once the row exists, and the restore
    # has to run in this window — a round that wrote its own comparison row first would
    # be recording a decline it had already acted on, and the report would describe a
    # contract it had already rewritten.
    #
    # #1546: a round the deadline stopped measured the baseline on fewer tasks than
    # the round it would be compared with, so its mean is not comparable: no
    # comparison, no restore on it, no `round_summary` row to become the next
    # round's reference — and no promotion either (below). Every variant it did
    # measure is still ranked, judged, reported and on the ledger.
    prior_promotion, comparison = (None, None) if deadline_stopped else post_promotion_comparison(
        cfg, rid, float(baseline_summary.get("mean_composite", 0.0)))

    # Evaluate each candidate vs baseline and pick the winner
    decisions: list[dict[str, Any]] = []
    best_variant: dict[str, Any] | None = None
    best_summary: dict[str, Any] | None = None
    best_overlay: Path | None = None

    for vid, overlay in variant_pairs:
        if vid == baseline_id:
            continue
        vs = summaries.get(vid)
        if not vs:
            continue
        should, reason = evaluate_promotion(cfg, baseline_summary, vs, split=split)
        # #1860: what the predicate itself said, before any wrapper below. The deadline
        # arm rewrites `reason` into a sentence about the deadline, and that prose is
        # what the strict-win census matches, so it stays exactly as it was — this is
        # the value the ledger row carries in machine form alongside it.
        predicate_refusal = "" if should else refusal_head(reason)
        if deadline_stopped:
            # Every item is a proposal deployed only on a measured gain; a round
            # that did not measure every task has not measured one.
            should, reason = False, (f"deadline_stopped: {len(tasks_not_reached)} task(s) not "
                                     f"reached, no promotion from a partial round "
                                     f"(predicate said: {reason})")
        m = slice_metrics(baseline_summary, vs, split)
        decisions.append({
            "variant_id": vid,
            "mean_composite": vs.get("mean_composite"),
            "targeted_delta": m["targeted_delta"],
            "heldout_delta": m["heldout_delta"],
            "normalized_gain": m["normalized_gain"],
            "should_promote": should,
            "reason": reason,
            # #1860: read by `decision_ledger_row`, which would otherwise only have the
            # wrapped prose to derive a class from — and for a stopped round that prose
            # begins `deadline_stopped:`, which is the wrap, not the refusal.
            "predicate_refusal": predicate_refusal,
            # #646: the same predicate re-run over only the tasks the validity lint
            # refuses to call broken. Advisory — `should_promote` is what decides —
            # and its whole purpose is the case where the two disagree.
            "validity": _validity_or_none(cfg, baseline_summary, vs, split),
        })
        if should and (best_summary is None or vs["mean_composite"] > best_summary["mean_composite"]):
            best_variant = next(v for v in variants if v["variant_id"] == vid)
            best_summary = vs
            best_overlay = overlay

    # Second-pass judge + promotion on the winner
    promotion_result: dict[str, Any] | None = None
    if best_variant and best_summary and best_overlay:
        promotion_result = promote(cfg, best_variant, best_overlay, best_summary, baseline_summary, dry_run=dry_run)

    # #1099, Alan's sign-off on #429's out-of-scope clause: a beyond-noise decline is
    # no longer only reported, it undoes the promotion that caused it — through
    # `promote.rollback`, which commits by the same validated vault route `promote`
    # uses, refuses any canonical file that changed after the promotion landed, and
    # never restores one promotion twice. Nothing here waits for a human; nothing here
    # copies a prompt file by hand. A `--dry-run` round never restores either.
    restore_outcome = None if deadline_stopped else auto_restore.restore_for_decline(
        cfg, rid, comparison, prior_promotion, dry_run=dry_run,
    )

    # #429: a promotion used to be the last time anything looked at it. Record this
    # round's comparison row and carry the verdict into the report below. The restore
    # above already ran (or declined, or refused) on the comparison computed before the
    # promotion decision; what is recorded here is the row plus that outcome, so the
    # report states the restore in the same section as the decline that caused it.
    # #789: the same row carries the contract shape — the live SOUL.md's two
    # ratios and this round's candidate's own two, read from `best_overlay`, which
    # is None whenever no candidate survived to the contract check.
    if deadline_stopped:
        report_lines, summary_row = [
            "", "## Post-promotion check (#429)",
            f"- not run: the round stopped at its {budget_minutes} min budget with "
            f"{len(tasks_not_reached)} task(s) not reached, so its baseline mean is not "
            "comparable and no `round_summary` row was written (#1546)."], None
    else:
        report_lines, summary_row, comparison = post_promotion_check(
            cfg,
            rid,
            float(baseline_summary.get("mean_composite", 0.0)),
            promotion_result,
            best_summary,
            best_overlay,
            comparison=comparison,
            restore=restore_outcome,
        )

    # Write round summary markdown
    summary_file = cfg.paths.rounds_dir / f"{rid}.md"
    sdk_task_ids = sorted({t["task_id"] for t in sdk_traces})
    lines = [
        f"# Autoresearch round {rid}",
        f"- started_at: {now_iso()}",
        f"- model: {model}",
        f"- harness: {harness}",
        f"- tasks: {len(tasks)}",
        f"- tasks on the harness runner: {len(sdk_task_ids)}"
        + (f" ({', '.join(sdk_task_ids)})" if sdk_task_ids else ""),
        f"- requires_runtime tasks skipped under harness={harness}: {len(skipped_runtime_ids)}"
        + (f" ({', '.join(skipped_runtime_ids)}) — not scored, no ledger row"
           if skipped_runtime_ids else ""),
        # #1546: the round's clock, so a report says whether it measured the whole
        # matrix and how much of its budget each part took.
        f"- budget: {budget_minutes} min" if budget_minutes else "- budget: none",
        # #1605: what the round decided to start, and against what. A round that
        # shrank has to say both numbers, or a reader cannot tell a deferred task
        # from a task that was never on the list.
        f"- trial matrix: {_projection_text(projection)}; window "
        + ("none (no budget)" if projection["window_seconds"] is None
           else f"{projection['window_seconds']:.0f} s")
        + "; fits " + ("n/a" if projection["fits"] is None
                       else "yes" if projection["fits"] else "NO"),
        *([f"- matrix reduced before trials: {len(matrix_dropped)} task(s) dropped — "
           f"{_projection_text(planned)} would not fit the window; started instead at "
           f"{projection['projected_seconds']:.0f} s: {', '.join(matrix_dropped)}"]
           if matrix_dropped else []),
        # #1716: the ids on the line above are the drop; what they COST is the
        # standing ruling's business, and the ruling is only as good as the
        # denominator next to it. Printed only on a round that actually shrank — a
        # round that dropped nothing has no coverage to have given up, and a `0`
        # there would be the same unreadable number as an unreadable lint.
        *(_coverage_text(matrix_coverage_report) if matrix_dropped else []),
        # #1715: the case #1605's loop had no words for. The matrix is over window,
        # shrinking has stopped, and the reason is not that it ran out of tasks — it
        # is that every step left frees less than the margin already holds back, so
        # dropping one more task would cost an arm's evidence and buy no window.
        # Saying "fits NO" alone would read as a bug in the shrink; the point is that
        # the round now knows it is going to be cut, and why it stopped helping.
        *([f"- matrix over window, shrinking stopped: {shrink_stopped['tasks_still_in_matrix']} "
           f"task(s) left projecting {projection['projected_seconds']:.0f} s against "
           f"{projection['window_seconds'] * PROJECTION_MARGIN:.0f} s of budget "
           f"({shrink_stopped['over_budget_seconds']:.0f} s over); the next step "
           f"{shrink_stopped['next_step_id']} would free only "
           f"{shrink_stopped['next_step_saves_seconds']:.0f} s, less than the "
           f"{shrink_stopped['step_floor_seconds']:.0f} s a PROJECTION_MARGIN of "
           f"headroom already holds back, so the matrix was not shrunk further and "
           f"the deadline will cut some of these tasks"]
           if shrink_stopped else []),
        f"- trials took: {trial_seconds:.0f} s (direct "
        + ("n/a" if arm_seconds.get("direct_seconds") is None
           else f"{arm_seconds['direct_seconds']:.0f} s")
        + ", agent-loop "
        + ("n/a" if arm_seconds.get("sdk_seconds") is None
           else f"{arm_seconds['sdk_seconds']:.0f} s")
        + f"); round took: {time.monotonic() - started:.0f} s so far",
        f"- stopped at deadline: {'yes' if deadline_stopped else 'no'}"
        + (f" — {len(tasks_not_reached)} task(s) not reached by every variant, not scored: "
           f"{', '.join(tasks_not_reached)}" if deadline_stopped else ""),
        f"- variants proposed: {len(variants)}",
        # #680: the aggregate said "2 dropped", which cannot distinguish "the
        # model invented spans" from "half this round's variants aimed at a file
        # it was only shown 4,000 chars of". Every proposed surface is listed,
        # zeros included, so a round that produced no MEMORY variants says so.
        f"- variants dropped by surface: {_surface_drop_text(dropped_by_surface) or '(none proposed)'}",
        f"- baseline mean composite: {baseline_summary.get('mean_composite', 0.0):.4f}",
        "",
        "## Variant summaries",
    ]
    for vid, summ in summaries.items():
        marker = " (baseline)" if vid == baseline_id else ""
        # This line's format is a contract, not a preference: `post_promotion.py`
        # parses it back out of every round report, and
        # `tests/test_post_promotion.py` extracts the f-string by AST and evaluates
        # it standalone with only `summ`, `vid` and `marker` in scope. So the number
        # of trials the mean is over lives in the `## Rubric coverage (#646)`
        # section below, which is appended rather than parsed, and NOT as a suffix
        # here — a suffix would need a local variable, and a local variable makes
        # the extracted template raise NameError.
        lines.append(f"- `{vid}`{marker}: mean={summ.get('mean_composite', 0.0):.4f}, "
                     f"safety={'pass' if summ.get('safety_passed') else 'fail'}, tasks={summ.get('task_count', 0)}")
    # #416: the mean alone cannot show that a variant was ranked on fewer tasks
    # than the round ran, so the gaps are reported under it. Absent when every
    # check was measurable, which is what the sdk arm is meant to make normal.
    # #646: the other way a trial contributes nothing — the judge never answered.
    rubric_lines = _rubric_exclusion_lines(summaries)
    if rubric_lines:
        lines += ["", "## Rubric coverage (#646)",
                  "A rubric call that did not answer, answered without JSON, or answered",
                  "with JSON that did not parse excludes its trial: the 0.5 it used to",
                  "contribute was arithmetic on a score the judge did not give.",
                  *rubric_lines]
    coverage_lines = _objective_coverage_lines(summaries)
    if coverage_lines:
        lines += ["", "## Objective coverage (#416)",
                  "Tool-behaviour checks on a trace with no dispatch record are excluded,",
                  "not scored off the prose; a task whose whole objective layer is",
                  "excluded contributes no marks to any mean.",
                  *coverage_lines]
    # The held-out slice is reported here and never to the proposer (#549): the
    # round is allowed to know which veto tasks declined, the optimizer is not.
    lines.append("")
    lines.append("## Bench split")
    lines.append(f"- split_hash: `{split['split_hash']}`")
    lines.append(f"- targeted ({len(split['targeted'])}): {', '.join(split['targeted'])}")
    lines.append(f"- held-out ({len(split['heldout'])}): {', '.join(split['heldout'])}"
                 f" (rotated in: {', '.join(split['rotated_into_heldout']) or 'none'})")
    heldout_scores = {p["task_id"]: p["composite_score"]
                      for p in baseline_summary.get("per_task", [])
                      if p["task_id"] in set(split["heldout"])}
    if heldout_scores:
        lines.append("- baseline held-out scores: " + ", ".join(
            f"{t}={heldout_scores[t]:.2f}" for t in sorted(heldout_scores)))
    lines.append("")
    lines.append("## Promotion decisions")
    for d in decisions:
        lines.append(f"- `{d['variant_id']}`: {'PROMOTE' if d['should_promote'] else 'HOLD'} — {d['reason']}")
    validity_sections = [(d["variant_id"], d.get("validity")) for d in decisions if d.get("validity")]
    if validity_sections:
        lines += ["", "## Bench validity (#646)",
                  "The promotion predicate evaluated twice over the same trial data: once",
                  "over every task, once over only the tasks `bench_lint` does not call",
                  "broken. Where they disagree, the gap is the broken-task effect."]
        for vid, validity in validity_sections:
            lines.append(f"- variant `{vid}`:")
            lines += [f"  {line}" for line in validity_report_lines(validity)]
    if promotion_result:
        lines.append("")
        lines.append("## Promoted")
        lines.append(f"- variant: `{promotion_result['variant_id']}`")
        lines.append(f"- snapshot_dir: `{promotion_result.get('snapshot_dir')}`")
        lines.append(f"- applied_files: {promotion_result.get('applied_files')}")
        lines.append(f"- experiment_fact: `{promotion_result.get('experiment_fact')}`")
    # #1549: the behavioural scorecard, REPORT-ONLY. Nothing here feeds a verdict —
    # `evaluate_promotion` is called above and never sees this artifact, so a
    # `guardrail_hit: true` changes no decision and no reason on this page. Wiring it
    # in as the behavioural second condition is item step 5, and its precondition is
    # the discrimination bar, not a term of weeks (#2332): ONE live pair of captures
    # scored through `behavioural.compare_pairs` in which every declared axis is
    # `measurable`, with `denominator_a` and `denominator_b` both >= 2, `excluded_axes`
    # empty, and each axis's `abs_delta` under its own `epsilon` (0.25 on all four
    # today). Until a pair scores that way the section has not shown that it moves
    # less between two runs of an unchanged surface than the regression it would be
    # asked to catch, so nothing may be gated on it — and report-only rungs accumulating
    # without such a pair changes nothing (#2196 owns producing the pair).
    # No scenario runs inside this body either: #1546 kills every round at the 1800 s
    # pool cap, so a suite that ran here would inherit that death. The section scores
    # whole-run traces an out-of-band capture left under
    # `<research_root>/behavioural_traces/<round_id>/`, and falls back to the shipped
    # reference capture so the section and its shape are in every report from the
    # first round onward.
    scorecard = behavioural.round_scorecard(cfg, rid)
    behavioural.write_scorecard(cfg, rid, scorecard)
    lines += [""]
    lines += behavioural.scorecard_report_lines(scorecard)
    lines += trial_cost.report_lines(cost_records)
    # #2186: the repeated-sampling coverage leg — pass@1 against pass@N, which is
    # how a report says whether today's failure is a reliability gap or a
    # capability limit. Report-only by construction: this import is reached on the
    # way to writing a report, after `evaluate_promotion` has already answered, and
    # nothing in `promote.py` imports it back. The arm itself is an idle-window job
    # — drawing N samples per task inside this body would inherit the 1800 s pool
    # cap that killed #1546 — so the section renders whichever arm's artifact
    # exists for this round's route, and says in words when there is none.
    lines += coverage_leg.report_lines(
        cfg, baseline_summary=baseline_summary, model_alias=model,
        route=coverage_leg.route_for_harness(harness))
    lines.extend(report_lines)
    summary_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Patch ledger with final promotion decisions (cheap second pass — append another entry)
    promoted_vid = promotion_result["variant_id"] if promotion_result and not dry_run else None
    variants_by_id = {v.get("variant_id"): v for v in variants}
    for d in decisions:
        ledger_append(cfg.paths.ledger_path,
                      decision_ledger_row(rid, d, promoted_vid,
                                          variants_by_id.get(d["variant_id"], {}),
                                          baseline_summary=baseline_summary,
                                          cost_record=cost_records.get(d["variant_id"])))

    return {
        "round_id": rid,
        "summary_file": str(summary_file),
        "spec_file": str(spec_path),
        # #1549: the behavioural artifact's name, so a caller can open the per-axis
        # paired deltas without parsing them back out of the report's markdown table.
        "behavioural_scorecard": behavioural.scorecard_path(cfg, rid).name,
        "variants_proposed": len(variants),
        "variants_dropped": dropped,
        "variants_dropped_by_surface": dict(dropped_by_surface),
        "tasks_run": len(tasks),
        "baseline_mean": baseline_summary.get("mean_composite", 0.0),
        "decisions": decisions,
        "promoted": promotion_result,
        "post_promotion": comparison,
        # #1099: the restore verdict is part of the round's result, not a side effect
        # a caller has to go and find in the ledger. `status` is one of
        # `auto_restore`'s statuses and `vault_commit` is the sha when it landed one.
        "post_promotion_restore": (
            {k: restore_outcome.get(k) for k in
             ("status", "restored", "vault_commit", "reason", "restored_by")}
            if restore_outcome else None),
        "round_summary_row": summary_row,
        "parent_variant_id": parent_variant["variant_id"] if parent_variant else None,
        "dry_run": dry_run,
        "harness": harness,
        "tasks_on_harness_runner": len(sdk_task_ids),
        "requires_runtime_skipped": skipped_runtime_ids,
        # #1546: the deadline-stop marker. A stopped round returns normally — it
        # is a result, not a failure — so the pool records it as completed and
        # does not re-run the same work three more times.
        "budget_minutes": budget_minutes,
        "deadline_stopped": deadline_stopped,
        "tasks_not_reached": tasks_not_reached,
        "trial_seconds": trial_seconds,
        # #1605: the arms measured the way the projection costs them, so the one
        # number a shrink is justified by can be checked against the one number a
        # round records. Absent when an arm had no task to run.
        "direct_seconds": arm_seconds.get("direct_seconds"),
        "sdk_seconds": arm_seconds.get("sdk_seconds"),
        # #1605: the decision made before the first trial — what the default matrix
        # would have cost, what was started instead, and which tasks did not make it.
        "matrix_projection": projection,
        "matrix_planned": planned,
        "matrix_dropped_tasks": matrix_dropped,
        # #1716: the same figures the report states, so the owed-check that measures
        # the re-open condition does not have to parse them back out of markdown.
        # Present on every round, shrunk or not (`shrunk` says which), because a
        # payload key that exists only on the interesting rounds is a key every
        # reader has to guard. Read by nothing that decides anything.
        "matrix_coverage": matrix_coverage_report,
        "round_seconds": round(time.monotonic() - started, 1),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one autoresearch round")
    parser.add_argument("--targets", nargs="*", default=["prompts"])
    parser.add_argument("--budget", type=int, default=None, help="Budget minutes: no trial starts past it, less a judging reserve (#1546)")
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--bench-limit", type=int, default=None)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument(
        "--harness", choices=list(HARNESSES), default="auto",
        help="Which bench runner scores the tasks. 'auto' (default) routes a "
             "task to the in-process agent loop when its frontmatter sets "
             "requires_runtime: true, so PreToolUse hooks and any other runtime "
             "mechanism are actually live for that trial, and everything else "
             "through the single-turn vLLM completion. 'direct' keeps every "
             "task on the single-turn runner and SKIPS a requires_runtime task "
             "rather than scoring it as prose. 'sdk' forces the agent-loop path "
             "for every task.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    result = asyncio.run(run(
        targets=args.targets,
        # A hand-run round is bounded too: `autoresearch.default_budget_minutes`
        # when --budget is not given (#1546 — until then nothing read that key).
        budget_minutes=args.budget if args.budget is not None else load_config().default_budget_minutes,
        max_variants=args.max_variants,
        dry_run=args.dry_run,
        model=args.model,
        bench_limit=args.bench_limit,
        max_parallel=args.max_parallel,
        harness=args.harness,
    ))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    # Support: `python -m scripts.autoresearch.run_round` from /home/alansrobotlab/lloyd
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    main()
