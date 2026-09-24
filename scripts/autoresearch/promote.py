"""Promotion pipeline — snapshot, atomic swap, fact write.

A promotion executes when the targeted slice strictly improves, the held-out
slice does not decline, the variant strictly beats baseline on at least N% of the
targeted tasks (a tie is NOT a win — the tie rule and its purpose are written at
the counting line in `slice_metrics`), and every safety probe passes.
Before swapping, we snapshot the current state
(SOUL.md, MEMORY.md, USER.md) into `_pipeline/research/snapshots/<ts>/`.
After swap, we write the winning experiment as a fact under
`cfg.paths.facts_experiments_dir/<variant_id>/` (configured in config.yaml,
currently `~/lloyd-data/_pipeline/vault-derived/facts/experiments/`) so it's
queryable via the normal memory pipeline.

`rollback(snapshot_ts)` reverses a promotion by restoring files from the
named snapshot. Both directions reach the live vault through
`scripts.automod.vault_round.land()`, so whatever the contract files say at any
moment is validated, committed, and in the automod ledger — nothing lands by a
bare file copy.

Nothing in this module but `evaluate_promotion` decides a promotion.
`validity_report` re-runs that same predicate over only the bench tasks the
validity lint does not call broken and logs the pair beside the decision — both
numbers, and whether they disagree — but it is advisory and its own docstring
says why. Do not read a `promote_valid` as a gate, and do not make it one while
the bench's lint-valid pool is empty of scored tasks, which on the live bench it
is (`tests/test_bench_lint.py::test_the_live_valid_pool_is_empty_once_the_real_judge_scores_it`).
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import bench_split
from .common import DEFAULT_MIN_JUDGED_FRACTION, LLOYD_HOME, AutoresearchConfig, now_iso

logger = logging.getLogger("autoresearch.promote")


def _mean(scores: list[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0

# Canonical targets that can be overwritten by promotion
CANONICAL_PROMPTS = {
    "SOUL.md": LLOYD_HOME.parent / "obsidian" / "lloyd" / "SOUL.md",
    "MEMORY.md": LLOYD_HOME.parent / "obsidian" / "lloyd" / "MEMORY.md",
    "USER.md": LLOYD_HOME.parent / "obsidian" / "lloyd" / "USER.md",
}


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


#: The census key of the strict-win leg. One definition, shared with
#: `replay_frontier_selection.py`, which counts this leg by matching the prefix: the
#: reason text and the counter cannot then drift apart, because a counter that stops
#: matching a renamed reason reports 0 and reads as "the leg stopped firing".
REFUSAL_WIN_FRACTION = "insufficient_win_fraction"


def _per_task(summary: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for p in summary.get("per_task", []) or []:
        tid = p.get("task_id")
        score = p.get("composite_score")
        if tid is not None and isinstance(score, (int, float)):
            out[str(tid)] = float(score)
    return out


def _per_task_flags(summary: dict[str, Any]) -> dict[str, tuple[bool, Any]]:
    """task_id -> (safety_critical, safety_passed), read from the same rows the
    scores come from — `judge.aggregate_variant` stamps both onto every per-task row.

    Needed because the frontier's accept path cannot rest on the summary-level
    `safety_passed` alone. The operator is allowed to switch that leg off —
    `promotion_require_safety_pass` is their knob — and a frontier that read only the
    config flag would let a prompt through whose safety-critical probe had just
    failed. The per-task veto is the last thing standing between a frontier and an
    unsafe land, so it is read here, from the row the score itself came from.
    """
    out: dict[str, tuple[bool, Any]] = {}
    for p in summary.get("per_task", []) or []:
        tid = p.get("task_id")
        if tid is not None:
            out[str(tid)] = (bool(p.get("safety_critical")), p.get("safety_passed"))
    return out


def derive_split(*summaries: dict[str, Any]) -> dict[str, Any]:
    """A category split taken from the summaries themselves.

    `judge.aggregate_variant` already stamps `category` onto every per-task row,
    so the two fixed halves of the split are recoverable from any real summary.
    This is the fallback for a caller that has no written round split — it
    carries the categories, never the per-round rotation.
    """
    cats: dict[str, str] = {}
    for summ in summaries:
        for p in summ.get("per_task", []) or []:
            tid, cat = p.get("task_id"), p.get("category")
            if tid and cat:
                cats.setdefault(str(tid), str(cat))
    return {
        "round_id": None,
        "targeted": sorted(t for t, c in cats.items() if c in bench_split.TARGETED_CATEGORIES),
        "heldout": sorted(t for t, c in cats.items() if c in bench_split.HELDOUT_CATEGORIES),
        "rotated_into_heldout": [],
        "split_hash": None,
        "derived_from": "task_category",
    }


def slice_metrics(
    baseline_summary: dict[str, Any],
    variant_summary: dict[str, Any],
    split: dict[str, Any],
) -> dict[str, Any]:
    """Per-slice numbers the gate decides on. One implementation, shared with
    `replay_promotion_gate.py` so a replay cannot drift from the predicate it is
    replaying — a re-derived arithmetic would be measuring the replay, not the gate.
    """
    base_per = _per_task(baseline_summary)
    var_per = _per_task(variant_summary)
    targeted = [str(t) for t in (split.get("targeted") or [])]
    heldout = [str(t) for t in (split.get("heldout") or [])]
    known = set(targeted) | set(heldout)
    unsplit = sorted((set(base_per) | set(var_per)) - known)

    def pair(ids: list[str]) -> tuple[list[str], float, float]:
        shared = [t for t in ids if t in base_per and t in var_per]
        return shared, _mean([base_per[t] for t in shared]), _mean([var_per[t] for t in shared])

    t_ids, t_base, t_var = pair(targeted)
    h_ids, h_base, h_var = pair(heldout)
    # ── The win leg's tie semantics, in one place, deliberately STRICT ─────────
    # A win is a per-task STRICT increase (`variant > baseline`); a tie is not a
    # win, a decrease is a loss. The leg exists to catch a variant that gains on a
    # few tasks while REGRESSING on others, and strictness is what makes a
    # regression a non-win — that is the whole purpose, so it is decided here
    # rather than left to whichever spelling of a comment is nearest.
    # Counting ties as wins would not weaken that purpose — a regression is still a
    # non-win — but it would empty the leg on this bench:
    # `bench_mine.calibrate_candidate` records four of the eleven live tasks sitting
    # at exactly 0.00, so a variant that moved ONE targeted task and tied every other
    # one would score `compared/compared` = 1.00 and clear the threshold outright.
    # And 0.5 is a measured operating point: `promotion_fp_rate.py` derived its
    # false-positive rate over the live promoted-round corpus under exactly these
    # strict semantics, and #549's strict held-out legs below refuse a tie for the
    # same reason.
    # The ledger shape this refuses, reproduced by a named test
    # (`test_the_strict_tie_rule_is_a_deliberate_decision_on_the_ledger_shape`): a
    # 2026-09-01 variant better on 4 of 11 tasks, tied on 7, WORSE ON NONE, whole-bench
    # mean +0.1227, refused at the 0.36 the ledger recorded for it. A ledger census
    # taken 2026-09-16 found all 552 `insufficient_win_fraction` refusals carrying a
    # positive mean delta, and 24 refused variants that strictly dominated their
    # baseline (better on some tasks, worse on none). Refusing those is the decision;
    # printing the split beside it, instead of only the fraction, is what stops the
    # next reader recomputing 28k ledger rows to discover that `losses=0`.
    wins = sum(1 for t in t_ids if var_per[t] > base_per[t])
    losses = sum(1 for t in t_ids if var_per[t] < base_per[t])
    ties = len(t_ids) - wins - losses

    # ── #595: strict dominance, the frontier's own question ────────────────────
    # A different question from the leg above, and asked over BOTH slices. The win
    # fraction asks how many targeted tasks moved — a fraction, so on a bench whose
    # tasks sit at the floor or the ceiling it has a low ceiling by construction (the
    # comment above). Dominance asks whether anything moved backwards at all:
    # strictly better on at least one compared task, strictly worse on none. The
    # shape it names is the one in the comment above — better on 4 of 11, worse on
    # none — and `replay_frontier_selection.py` is the census of it, so the count
    # lives in that script's output rather than twice in this file.
    # The safety veto belongs to THIS condition, not only to the leg that reads
    # `variant_summary["safety_passed"]`: that leg is the operator's to disable
    # (`promotion_require_safety_pass`), and a frontier that accepted a prompt whose
    # safety-critical probe failed would be a frontier with no veto at all. So a
    # per-task failed veto is decided here, where the frontier is decided.
    var_flags = _per_task_flags(variant_summary)
    all_ids = [t for t in targeted + heldout if t in base_per and t in var_per]
    # A task scored on one side and not the other is not a tie and not a regression —
    # it is an absence, and "regressed on none" is only a verdict when there was
    # something to regress on. A truncated rollout (a variant that scored 4 of 11
    # tasks before the round died) would otherwise dominate by never being measured:
    # exactly the shape #549 refuses as `partial_heldout_coverage` and #416 refuses as
    # not-rankable. So coverage is a precondition of the frontier, not a footnote.
    missing_ids = sorted(t for t in targeted + heldout
                         if (t in base_per) != (t in var_per))
    better = [t for t in all_ids if var_per[t] > base_per[t]]
    worse = [t for t in all_ids if var_per[t] < base_per[t]]
    safety_failed = sorted(t for t in all_ids
                           if var_flags.get(t, (False, True))[0]
                           and var_flags.get(t, (False, True))[1] is False)
    dominates = bool(better) and not worse and not safety_failed and not missing_ids

    return {
        "compared": len(t_ids),
        "wins": wins, "ties": ties, "losses": losses,
        "dominates": dominates, "better_ids": better, "worse_ids": worse,
        "safety_failed_ids": safety_failed, "compared_all": len(all_ids),
        "unsplit": unsplit,
        "targeted_ids": t_ids, "targeted_baseline": t_base, "targeted_variant": t_var,
        "targeted_delta": t_var - t_base,
        "heldout_ids": h_ids, "heldout_baseline": h_base, "heldout_variant": h_var,
        "heldout_delta": h_var - h_base,
        "win_fraction": (wins / len(t_ids)) if t_ids else 0.0,
        # HarnessOpt-Bench's normalized gain: the fraction of the seed's *remaining
        # headroom* the change captured. A raw +0.05 over a 0.50 baseline is 10% of
        # what was available; the same +0.05 over 0.85 is 33%. Printed beside the
        # delta so a decision says how much it bought, not only which way it moved.
        "headroom": 1.0 - t_base,
        "normalized_gain": ((t_var - t_base) / (1.0 - t_base)) if t_base < 1.0 else None,
    }


#: The refusal key of the judged-coverage floor (#698). One definition, so a
#: census of ledger reasons can count it by prefix.
REFUSAL_JUDGED_FLOOR = "insufficient_judged_tasks"


def judged_floor_refusal(cfg: AutoresearchConfig, *summaries: tuple[str, dict[str, Any]]) -> str | None:
    """The floor under #646's exclusion: too few judged trials is no verdict.

    #646 took a trial whose rubric call never answered out of every mean, and
    #698 made that trial's composite None rather than a phantom 0.5 — both right,
    and both shrink the denominators the gate decides on. A variant whose judge
    answered on 3 of 11 tasks can then clear a win fraction or a mean delta on
    the three it has. So a summary on either side of the comparison with fewer
    than `promotion_min_judged_fraction` (8/11 by default, the item's "8 of the
    11") of its rankable trials judged refuses, naming the count.

    The denominator is `rankable_task_count`, not `task_count`: a task #416
    already called not-rankable (no dispatch record on the direct arm — 6 of the
    13 live tasks) is a harness fact, reported on its own, and counting it here
    would refuse every direct-arm round for a reason that is not the judge.
    A summary that carries no counts (a hand-built or pre-#646 one) is not
    judged by this floor.
    """
    frac = float(getattr(cfg, "promotion_min_judged_fraction", DEFAULT_MIN_JUDGED_FRACTION))
    for label, summ in summaries:
        rankable = summ.get("rankable_task_count")
        judged = summ.get("scored_task_count")
        if not isinstance(rankable, int) or not isinstance(judged, int) or rankable <= 0:
            continue
        need = math.ceil(frac * rankable - 1e-9)
        if judged < need:
            excluded = ", ".join(summ.get("rubric_excluded_tasks") or []) or "none named"
            return (f"{REFUSAL_JUDGED_FLOOR} ({label}: {judged} of {rankable} rankable "
                    f"tasks judged, need {need} at {frac:.3f}; unjudged: {excluded})")
    return None


def evaluate_promotion(
    cfg: AutoresearchConfig,
    baseline_summary: dict[str, Any],
    variant_summary: dict[str, Any],
    split: dict[str, Any] | None = None,
    require_full_slice: bool = True,
) -> tuple[bool, str]:
    """Decide whether a variant should be promoted over baseline.

    Two conditions on two slices, per AutoDesign (arXiv:2608.13560): the targeted
    slice must improve AND the held-out slice must not decline. `safety_passed`
    and `contract_refusals()` stay as vetoes on top — neither is a substitute for
    the split, because `safety_critical` is set on exactly one of the 11 tasks and
    the contract guard reads the prompt, not the scores.

    The absolute `promotion_min_composite_delta` is gone. It compared a raw delta
    against a fixed 0.05 while the rounds' own baselines were seeded from
    `composite < 0.5` tasks, so the gate's strictness moved with the baseline mix:
    +0.05 off a 0.50 seed captured 10% of the available headroom, +0.05 off 0.85
    captured 33%. `targeted_delta > 0` asks only for a real gain on the pool the
    variant aimed at; the no-decline half is what stops a gain there from paying
    for a loss anywhere else.

    Returns (should_promote, reason).
    """
    if not variant_summary.get("safety_passed", False) and cfg.promotion_require_safety_pass:
        return False, "safety_regression"

    floor = judged_floor_refusal(cfg, ("variant", variant_summary),
                                 ("baseline", baseline_summary))
    if floor:
        return False, floor

    split = split or derive_split(baseline_summary, variant_summary)
    if not split.get("heldout"):
        # No veto slice means this is the pre-#549 gate wearing new clothes.
        return False, "no_heldout_slice"

    m = slice_metrics(baseline_summary, variant_summary, split)
    if m["unsplit"]:
        # A scored task outside both pools is a task no condition looks at —
        # usually a bench file added between the split write and the run.
        return False, f"unsplit_tasks ({len(m['unsplit'])}: {', '.join(m['unsplit'][:3])})"
    if not m["targeted_ids"]:
        return False, "no_targeted_overlap"
    if not m["heldout_ids"]:
        return False, "no_heldout_overlap"
    if require_full_slice:
        # Every task the split declared has to have been scored. A round truncated
        # by `--bench-limit` normally scores SOME of the veto slice — one task of
        # four — and then "the held-out mean did not decline" is a verdict on a
        # quarter of the veto, the same averaging defect the split exists to
        # remove, moved one layer down. Fail closed and name what went unread.
        for label, pool, scored in (("held-out", split["heldout"], m["heldout_ids"]),
                                    ("targeted", split["targeted"], m["targeted_ids"])):
            missing = sorted(set(pool) - set(scored))
            if missing:
                return False, (
                    f"partial_{'heldout' if label == 'held-out' else 'targeted'}_coverage "
                    f"({len(scored)} of {len(pool)} {label} tasks scored; "
                    f"unscored: {', '.join(missing)})"
                )

    if m["targeted_delta"] <= 0:
        return False, (
            f"targeted_no_gain (targeted {m['targeted_baseline']:.4f} → "
            f"{m['targeted_variant']:.4f}, {m['targeted_delta']:+.4f} on "
            f"{len(m['targeted_ids'])} tasks)"
        )
    # Strict: a tie is a refuse. The item's "held-out must not decline, strict —
    # a tie on held-out is a refuse" is the difference between an unchanged slice
    # and an unmeasured one at 5 tasks, and ties are what a rubric judge with a
    # coarse objective check actually returns.
    if m["heldout_delta"] < 0:
        return False, (
            f"heldout_decline (held-out {m['heldout_baseline']:.4f} → "
            f"{m['heldout_variant']:.4f}, {m['heldout_delta']:+.4f} on "
            f"{len(m['heldout_ids'])} veto tasks)"
        )
    if m["heldout_delta"] == 0:
        return False, (
            f"heldout_tie (held-out flat at {m['heldout_variant']:.4f} across "
            f"{len(m['heldout_ids'])} veto tasks — strict no-decline refuses a tie)"
        )
    if m["dominates"] and m["win_fraction"] < cfg.promotion_min_win_fraction:
        # ── #595's accept path ─────────────────────────────────────────────────
        # Reached only with the safety leg answered, every #549 slice leg answered
        # (targeted gained, veto slice gained), and nothing regressed on any scored
        # task. The win fraction is not the right instrument for that shape: it is a
        # fraction of targeted tasks that moved, and on a bench where most tasks sit
        # at floor or ceiling it cannot rise far, so it refuses improvement that cost
        # nothing. This is deliberately BEFORE the win-fraction refusal and
        # deliberately after every veto — it is an alternative selector, never a
        # lowered threshold, and `min_composite_delta`, `min_bench_win_fraction` and
        # `require_safety_pass` are all still read at their loaded values.
        # It cannot accept a variant that only got lucky on one task: `dominates`
        # requires zero regressions across the union of both pools, so a single loss
        # anywhere sends the decision down the refusal path below.
        gain = m["normalized_gain"]
        gain_str = "n/a" if gain is None else f"{gain:.2%}"
        return True, (
            f"promote (dominance path: accepted — {len(m['better_ids'])} of "
            f"{m['compared_all']} tasks improved, 0 regressed, safety veto intact; "
            f"strict-win leg would have refused at {m['win_fraction']:.2f} < "
            f"{cfg.promotion_min_win_fraction}; win_frac={m['win_fraction']:.2f}, "
            f"targeted_delta={m['targeted_delta']:+.4f}, "
            f"heldout_delta={m['heldout_delta']:+.4f}, normalized_gain={gain_str}, "
            f"headroom={m['headroom']:.4f})"
        )
    if m["win_fraction"] < cfg.promotion_min_win_fraction:
        # The `insufficient_win_fraction (X.XX < Y.YY` prefix is load-bearing: the
        # ledger's 552-row census keys on it. Everything after it is the tie/loss
        # split, so a variant that beat baseline with ZERO regressions — the shape
        # the strict leg refuses by design — is recognisable from the round report or
        # the ledger row alone instead of only from a recomputation of the ledger.
        return False, (
            f"{REFUSAL_WIN_FRACTION} ({m['win_fraction']:.2f} < "
            f"{cfg.promotion_min_win_fraction}; wins={m['wins']} "
            f"ties={m['ties']} losses={m['losses']} of {m['compared']} targeted tasks)"
        )

    gain = m["normalized_gain"]
    gain_str = "n/a" if gain is None else f"{gain:.2%}"
    return True, (
        f"promote (targeted_delta={m['targeted_delta']:+.4f}, "
        f"heldout_delta={m['heldout_delta']:+.4f}, normalized_gain={gain_str}, "
        f"win_frac={m['win_fraction']:.2f})"
    )


def _restricted_summary(summary: dict[str, Any], keep: set[str]) -> dict[str, Any]:
    """A copy of `summary` holding only the tasks in `keep`.

    Only `per_task` and the two means are rebuilt; `safety_passed` is carried
    through unchanged so the valid-pool leg still sees the baseline round's own
    safety verdict rather than a veto the filter quietly deleted.
    """
    per_task = [p for p in (summary.get("per_task") or []) if str(p.get("task_id")) in keep]
    composites = [float(p.get("composite_score", 0.0)) for p in per_task]
    out = dict(summary)
    out["per_task"] = per_task
    out["mean_composite"] = round(sum(composites) / len(composites), 4) if composites else 0.0
    return out


def _restricted_split(split: dict[str, Any], keep: set[str]) -> dict[str, Any]:
    out = dict(split)
    out["targeted"] = [t for t in (split.get("targeted") or []) if str(t) in keep]
    out["heldout"] = [t for t in (split.get("heldout") or []) if str(t) in keep]
    return out


#: Below this many scored tasks the valid-pool leg is not a measurement — one
#: task either moved or it did not, and a mean over it is not a mean.
MIN_VALID_POOL_TASKS = 2


def validity_report(
    cfg: AutoresearchConfig,
    baseline_summary: dict[str, Any],
    variant_summary: dict[str, Any],
    split: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The all-task mean beside the lint-valid-task mean, and whether they agree.

    #646 step 5. Heiner's point is that a bench with a fifth of its tasks broken
    re-ranks models on the broken fifth, and you cannot find those tasks by
    solving the others. The lint (`scripts/autoresearch/bench_lint.py`) names
    which of ours are broken in the three ways that are mechanically checkable —
    a boilerplate reply passes, an ask has no verifier, a layer cannot fail — so
    the same promotion predicate can be evaluated twice: over every task, and over
    only the tasks that survive. Where the two disagree, the disagreement IS the
    20%-broken effect, and it gets recorded rather than averaged away.

    The valid-pool leg is ADVISORY ONLY — nothing here refuses a
    promotion the all-task leg allowed. Making the valid-only mean authoritative
    is #646's own deferred step, which waits on a human deciding per task whether
    to tighten a check or retire the task; until then the bench's lint-valid pool
    is small (4 of 13 as measured) and a veto from it would be a veto from four
    numbers. What this function does now is put both numbers, and the agreement
    between them, in the log, the round report and the ledger row.

    Read `reason_valid: valid_pool_too_small` carefully in a live round, because
    its cause is not the one its wording suggests. The pool is small after the
    scored filter, not after the bench: the 4 tasks the lint does not invalidate
    are exactly the ones whose only checks are tool-behaviour checks that no
    configured arm measures, so #416 has already dropped their trials and they
    carry no `per_task` row to restrict. The two exclusions are complementary
    halves of one defect — the lint invalidates the tasks the harness CAN read
    (keyword presence boilerplate passes), and the harness cannot read the tasks
    the lint leaves alone — and their intersection is empty, which is a sharper
    statement of the item's premise than any count of broken tasks: the mean that
    gates a promotion is currently an average over layers that a lazy reply
    satisfies. Reporting `None` here is the correct output; loosening `valid`
    until a number appears would delete the finding.
    """
    from .bench_lint import valid_task_ids  # local import: keeps promote importable without the lint's deps

    split = split or derive_split(baseline_summary, variant_summary)
    scored = {str(p.get("task_id")) for p in (baseline_summary.get("per_task") or [])} | {
        str(p.get("task_id")) for p in (variant_summary.get("per_task") or [])
    }
    valid = valid_task_ids(cfg.paths.bench_dir)
    pool = sorted(scored & valid)
    missing = sorted(scored - valid)

    should_all, reason_all = evaluate_promotion(cfg, baseline_summary, variant_summary, split=split)
    report: dict[str, Any] = {
        "bench_dir": str(cfg.paths.bench_dir),
        "scored_tasks": len(scored),
        "valid_tasks": pool,
        "excluded_tasks": missing,
        "all_task_mean": {
            "baseline": round(float(baseline_summary.get("mean_composite", 0.0)), 4),
            "variant": round(float(variant_summary.get("mean_composite", 0.0)), 4),
        },
        "promote_all": bool(should_all),
        "reason_all": reason_all,
        "authoritative": False,
    }
    report["all_task_mean"]["delta"] = round(
        report["all_task_mean"]["variant"] - report["all_task_mean"]["baseline"], 4)

    if len(pool) < MIN_VALID_POOL_TASKS:
        report.update({
            "valid_task_mean": None,
            "promote_valid": None,
            "reason_valid": f"valid_pool_too_small ({len(pool)} scored lint-valid tasks, need {MIN_VALID_POOL_TASKS})",
            "means_agree": None,
        })
    else:
        base_v = _restricted_summary(baseline_summary, set(pool))
        var_v = _restricted_summary(variant_summary, set(pool))
        should_v, reason_v = evaluate_promotion(
            cfg, base_v, var_v, split=_restricted_split(split, set(pool)),
            require_full_slice=False,
        )
        report["valid_task_mean"] = {
            "baseline": round(float(base_v.get("mean_composite", 0.0)), 4),
            "variant": round(float(var_v.get("mean_composite", 0.0)), 4),
            "tasks": len(pool),
        }
        report["valid_task_mean"]["delta"] = round(
            report["valid_task_mean"]["variant"] - report["valid_task_mean"]["baseline"], 4)
        report.update({
            "promote_valid": bool(should_v),
            "reason_valid": reason_v,
            "means_agree": bool(should_v) == bool(should_all),
        })
    # A safety-critical task outside the valid pool means the advisory leg has no
    # veto in it — say so in the row, because "both means say promote" reads very
    # differently once one of them could not have refused on safety grounds.
    safety_scored = [str(p.get("task_id")) for p in (variant_summary.get("per_task") or [])
                     if p.get("safety_critical")]
    report["safety_outside_valid_pool"] = [t for t in safety_scored if t not in valid]

    if report["means_agree"] is None:
        logger.info(
            "bench validity: all-task mean %.4f → %.4f; valid-pool leg not evaluated (%s); excluded: %s",
            report["all_task_mean"]["baseline"], report["all_task_mean"]["variant"],
            report["reason_valid"], ", ".join(missing) or "(none)",
        )
    else:
        logger.info(
            "bench validity: all-task mean %.4f → %.4f (promote=%s) | lint-valid mean %.4f → %.4f "
            "over %d tasks (promote=%s) | agree=%s%s; excluded: %s",
            report["all_task_mean"]["baseline"], report["all_task_mean"]["variant"], report["promote_all"],
            report["valid_task_mean"]["baseline"], report["valid_task_mean"]["variant"],
            report["valid_task_mean"]["tasks"], report["promote_valid"], report["means_agree"],
            (f"; safety veto outside valid pool: {', '.join(report['safety_outside_valid_pool'])}"
             if report["safety_outside_valid_pool"] else ""),
            ", ".join(missing) or "(none)",
        )
    return report


def validity_report_lines(report: dict[str, Any]) -> list[str]:
    """Render `validity_report` for the round report. Both numbers, always both."""
    a = report["all_task_mean"]
    lines = [
        f"- all-task mean: {a['baseline']:.4f} → {a['variant']:.4f} "
        f"({a['delta']:+.4f}) over {report['scored_tasks']} tasks — promote={report['promote_all']}",
    ]
    v = report.get("valid_task_mean")
    if v is None:
        lines.append(f"- lint-valid mean: not evaluated — {report['reason_valid']}")
    else:
        lines.append(
            f"- lint-valid mean: {v['baseline']:.4f} → {v['variant']:.4f} ({v['delta']:+.4f}) "
            f"over {v['tasks']} lint-valid tasks ({', '.join(report['valid_tasks'])}) "
            f"— promote={report['promote_valid']} ({report['reason_valid']})"
        )
    if report.get("means_agree") is None:
        lines.append("- the two means could not be compared — the valid-pool leg was not evaluated")
    elif report["means_agree"]:
        lines.append("- the two means AGREE on promote/no-promote")
    else:
        lines.append(
            "- the two means DISAGREE on promote/no-promote — that gap is the "
            "broken-task effect #646 is measuring, and it is the finding"
        )
    lines.append(
        f"- excluded as lint-invalid ({len(report['excluded_tasks'])}): "
        f"{', '.join(report['excluded_tasks']) or '(none)'}"
    )
    if report.get("safety_outside_valid_pool"):
        lines.append(
            f"- the lint-valid pool holds no safety-critical task: "
            f"{', '.join(report['safety_outside_valid_pool'])} is excluded from it, so the "
            "valid-pool leg cannot refuse on safety grounds"
        )
    lines.append("- the valid-pool mean is advisory; the all-task leg is what promoted (#646)")
    return lines


def snapshot_current_prompts(cfg: AutoresearchConfig) -> Path:
    """Copy current SOUL/MEMORY/USER into a timestamped snapshot dir. Returns the dir."""
    snap_dir = cfg.paths.snapshots_dir / _ts()
    snap_dir.mkdir(parents=True, exist_ok=True)
    for name, src in CANONICAL_PROMPTS.items():
        if src.exists():
            shutil.copy2(src, snap_dir / name)
    saved = sorted(p.name for p in snap_dir.iterdir() if p.is_file())
    # A snapshot is the only rollback point a promotion has, and `mkdir` had
    # been the whole guarantee: a copy that raised still left a directory, so
    # `promote` went on to overwrite the live contract with nothing to restore.
    # On 2026-09-06, writing `test_promote_refuses_when_the_snapshot_cannot_be_
    # written`, 26 promotions were counted with no matching snapshot; the
    # denominator quoted then was never sourced. This guard is what closed that:
    # measured 2026-09-19, every one of the 67 snapshot dirs held a prompt file
    # (#784). It stays because it is the reason that coverage holds, not a
    # response to a defect still open.
    if not any(name in CANONICAL_PROMPTS for name in saved):
        raise RuntimeError(
            f"snapshot {snap_dir} holds no prompt file — refusing to promote "
            f"with no rollback point (expected any of {sorted(CANONICAL_PROMPTS)})"
        )
    (snap_dir / "snapshot.json").write_text(
        json.dumps({"created_at": now_iso(), "files": saved}, indent=2),
        encoding="utf-8",
    )
    logger.info("snapshotted canonical prompts into %s", snap_dir)
    return snap_dir


def apply_overlay(overlay_dir: Path) -> list[str]:
    """Copy variant overlay files onto canonical prompts. Returns list of applied files."""
    applied: list[str] = []
    for name, dest in CANONICAL_PROMPTS.items():
        src = overlay_dir / name
        if src.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            applied.append(name)
    return applied


def _prospective(overlay_dir: Path, name: str) -> str | None:
    """What `name` would contain after this overlay is applied."""
    src = overlay_dir / name
    if src.exists():
        return src.read_text(encoding="utf-8")
    dest = CANONICAL_PROMPTS.get(name)
    if dest and dest.exists():
        return dest.read_text(encoding="utf-8")
    return None


#: The two surfaces a candidate's own shape can be measured on, in the order the
#: ceilings govern them. SOUL.md first: `check_contract` puts both ceilings on that
#: file, so when an overlay carries both files the candidate ratio that means
#: something to the ratchet is SOUL.md's, and a MEMORY.md number recorded beside it
#: would enter a MEMORY ratio into a SOUL series — the cross-entity fragmentation
#: failure in miniature, one measurement filed under the wrong system.
SHAPE_SURFACES: tuple[str, ...] = ("SOUL.md", "MEMORY.md")


def candidate_shape(overlay_dir: Path) -> dict[str, Any]:
    """The shape a candidate's *own* file has, and which file that was.

    `{surface, gate_share, prohibition_ratio}`, all-`None` when the overlay carries
    neither measured surface. Deliberately the overlay's text rather than the
    prospective merged contract: the live ratios answer "what is the contract now",
    these answer "what is this candidate proposing", and the ratchet needs the
    second one or a no-op overlay would inherit the live value as a plateau.

    `surface` is recorded so the history can be filtered by it. A MEMORY.md-only
    candidate is the case #789 names as the blind spot — `check_contract` puts both
    ceilings on SOUL.md, so a MEMORY-only overlay that doubles its own gate stack
    trips nothing. Recording it is the half this item can close; ceiling-ing
    MEMORY.md is a scope call a person has to make.
    """
    try:
        import prompt_surface
    except ImportError:  # pragma: no cover - repo is always importable
        return {"surface": None, "gate_share": None, "prohibition_ratio": None}
    for name in SHAPE_SURFACES:
        src = overlay_dir / name
        if src.exists():
            shape = prompt_surface.contract_shape(
                src.read_text(encoding="utf-8")
            )
            return {
                "surface": name,
                "gate_share": shape["gate_share"],
                "prohibition_ratio": shape["prohibition_ratio"],
            }
    return {"surface": None, "gate_share": None, "prohibition_ratio": None}


def contract_shape_fields(overlay_dir: Path | None) -> dict[str, Any]:
    """The #789 block one round records: the live contract, and this candidate.

    `contract_*` is the SOUL.md the loop is running against right now — the file whose
    two ratios the ceilings police. `candidate_*` is the overlay's own file, per
    `candidate_shape`. A missing measurement is `None`, never `0.0`: a round with no
    candidate and a contract with no gate stack are different facts, and a zero inside
    a series reads as a fall and masks a real climb.

    Here rather than in `post_promotion`, which writes the block to the ledger: reading
    prompt text needs this module's `CANONICAL_PROMPTS`, and #429's separation test
    pins the recorder's import list so it cannot reach this module.
    """
    fields: dict[str, Any] = {
        "contract_surface": None, "contract_gate_share": None,
        "contract_prohibition_ratio": None,
    }
    soul = CANONICAL_PROMPTS.get("SOUL.md")
    if soul is not None:
        try:
            import prompt_surface

            live = prompt_surface.contract_shape(
                soul.read_text(encoding="utf-8", errors="replace")
            )
            fields = {
                "contract_surface": "SOUL.md",
                "contract_gate_share": live["gate_share"],
                "contract_prohibition_ratio": live["prohibition_ratio"],
            }
        except (ImportError, OSError):  # pragma: no cover - always importable here
            pass
    cand = candidate_shape(overlay_dir) if overlay_dir is not None else {}
    fields.update({
        "candidate_surface": cand.get("surface"),
        "candidate_gate_share": cand.get("gate_share"),
        "candidate_prohibition_ratio": cand.get("prohibition_ratio"),
    })
    return fields


def shape_ratchet_refusals(
    shape: dict[str, Any],
    history: list[dict[str, Any]],
    run: int | None = None,
) -> list[str]:
    """Refuse a candidate that ratchets the contract's shape upward, #789.

    The absolute ceilings in `prompt_surface.check_contract` are per-candidate and
    cannot see this class: a climb that stays under 50% gate stack and under 25%
    prohibitions passes every ceiling forever, one promotion at a time, and each
    round's overlay only has to add a few hundred bytes to gain the ratchet. The
    live record is exactly that shape — gate share over the 65 promotion snapshots
    ran 39.0% (08-23) → 23.7% (09-02) → 39.4 → 52.9 → 63.0 → 63.6% (09-04/05), 7
    rises over 28 changed values. By the third consecutive rise the absolute check
    caught the 09-04 case, so this rule earns its keep only on climbs that stay
    *under* the ceiling, which is where the loop has headroom today (live SOUL.md
    measures 45.3% gate share and 19.3% prohibitions against ceilings of 50%/25%).

    Refuses when the candidate's own value is higher than the last `run - 1`
    recorded candidate values **in order** — three rises counting the candidate.
    The refusal fires only while **both** ratios sit under their ceilings: once one
    is past a ceiling, the absolute check has already refused this candidate and
    named its bytes and line counts, and a second refusal on a trend adds nothing a
    reader could act on.

    A metric with too few recorded values to form a run is skipped rather than
    guessed at: the first rounds after this ships have one row each, and a rule
    that fired on one data point would refuse every candidate for a reason nobody
    could check.
    """
    try:
        import prompt_surface
    except ImportError as exc:  # pragma: no cover - repo is always importable
        return [f"prompt_surface unavailable, refusing to promote blind: {exc}"]

    run = run or prompt_surface.CONTRACT_RISE_RUN
    metrics = (
        ("gate_share", "gate stack", prompt_surface.GATE_STACK_CEILING),
        ("prohibition_ratio", "prohibition lines", prompt_surface.PROHIBITION_RATIO_CEILING),
    )
    values = {key: shape.get(key) for key, _, _ in metrics}
    if any(values[key] is None for key, _, _ in metrics):
        return []
    if any(float(values[key]) > ceiling for key, _, ceiling in metrics):
        return []

    # Oldest first by the round's own timestamp, with file position as the tiebreak,
    # so a ledger whose rows were replayed out of order — a restore, a backfill — still
    # describes the series in the order it happened. A row whose `created_at` is missing
    # or unparseable has no comparable timestamp and is dropped rather than defaulted:
    # defaulting to "oldest" or "newest" would let an untrusted field put a value inside
    # the window and forge a climb, or move one out and hide one. Dropping costs the run
    # a value, and a short run refuses nothing — the safe direction. `post_promotion`
    # stamps `created_at` on every row it writes, so in production this set is empty.
    stamped: list[tuple[str, int, dict]] = []
    for index, row in enumerate(history):
        if not isinstance(row, dict):
            continue
        stamp = row.get("created_at")
        if isinstance(stamp, str) and len(stamp) >= 10 and stamp[4:5] == "-":
            stamped.append((stamp, index, row))
    ordered = [row for _, _, row in sorted(stamped, key=lambda item: (item[0], item[1]))]

    errors: list[str] = []
    for key, label, ceiling in metrics:
        # Recorded under the `candidate_` prefix, which is `SHAPE_FIELDS` and therefore
        # also the row's own column name; a bare key is accepted too so a hand-built
        # history in a test is one dict rather than a simulated ledger row. The window is
        # taken over *usable* values, not over rows: a row that is None for this metric
        # must not consume a window slot, or a series recorded 0.10, None, 0.11, 0.12
        # against a candidate of 0.13 — three real rises — would be read as its last two
        # rows, 0.12 then 0.13, and refuse nothing.
        #
        # The whole ordered series, not the last `run - 1` values, and no cheaper
        # window will do: the run is walked back from the end through *changed* values,
        # so a plateau inside the look-back is part of the climb rather than a break in
        # it. A series recorded g-0.03, g-0.02, g-0.02 against a candidate of g is three
        # rises, which is what the 2026-09-04→05 snapshot run actually looked like
        # (52.9 % → 63.0 % → 63.6 % → 63.6 %); taking two rows positionally would see
        # one rise and refuse nothing. Extra history cannot manufacture a run either —
        # `rising_run` stops at the first non-rise, so a fall anywhere in the window
        # ends the walk wherever it stands.
        field = f"candidate_{key}"
        prior = [
            float(r.get(field, r.get(key))) for r in ordered
            if r.get(field, r.get(key)) is not None
        ]
        series = prompt_surface.rising_run([*prior, float(values[key])], run)
        if len(series) < run:
            continue
        errors.append(
            f"{label} has risen across {len(series)} recorded shapes without crossing "
            f"the {ceiling:.0%} ceiling — "
            f"{' → '.join(f'{v:.1%}' for v in series)} — which no per-candidate "
            f"ceiling can see (backlog #789). Refused as a ratchet; the series is in "
            f"the ledger `round_summary` rows."
        )
    return errors


def contract_refusals(overlay_dir: Path, shape_history: list[dict[str, Any]] | None = None) -> list[str]:
    """Why this variant must not be written over the live contract. [] = fine.

    This path is what produced #464 and #465. It writes Lloyd's identity files
    in the live vault on an hourly cadence with no gate, no test, no review and
    no revert, and on 2026-09-08 it promoted a variant whose own hypothesis was
    "add negative constraints to fix the shape benchmarks" — 64% gate stack,
    28% prohibition lines — over a contract that had been trimmed to 48%/19%
    ninety minutes earlier. A search over prompts is legitimate; landing the
    result unchecked is not, and the search cannot be trusted to police itself
    because bench score is exactly what it is optimising.

    Checked against the *prospective* text — the overlay's file where it has
    one, the live file where it does not — so a variant that only rewrites
    MEMORY.md is still judged against the SOUL.md it will sit beside.

    `shape_history` is the recorded candidate shapes from earlier rounds, oldest
    first (see `post_promotion.contract_shape_history`). It defaults to `None`,
    which means "no series to consult" and skips only the cross-round ratchet — the
    absolute ceilings always run. The two callers that pass it (`promote()` reading
    its own ledger, and the tests) are the only place the ratchet can be evaluated
    at all, because it is the one check here that needs a fact about *other*
    rounds; every other check in this function is answerable from the overlay.
    """
    try:
        import prompt_surface
    except ImportError as exc:  # pragma: no cover - repo is always importable
        return [f"prompt_surface unavailable, refusing to promote blind: {exc}"]
    soul = _prospective(overlay_dir, "SOUL.md")
    if soul is None:
        return ["no SOUL.md to check, in the overlay or on disk"]
    # All three loaded surfaces, not the two this call used to pass. USER.md is in
    # `CANONICAL_PROMPTS` above and in `common._canonical_prompt_paths`, so
    # `apply_overlay` has always been able to overwrite it and
    # `hypothesis_generator` has always been shown its tail as a mutation target —
    # the search could write the largest prompt file while the guard that exists
    # because of #464 could not see it (#1010). #1008's shape is the concrete case:
    # an overlay holding only a USER.md that copies SOUL.md reached disk.
    errors = prompt_surface.check_contract(
        soul,
        _prospective(overlay_dir, "MEMORY.md"),
        _prospective(overlay_dir, "USER.md"),
    )
    # The ratchet compares against a series of SOUL.md candidates (`surface="SOUL.md"`
    # at the call site), so it may only be applied to an overlay that carries one. A
    # MEMORY.md-only overlay has no SOUL.md of its own, so `soul` here is the live
    # contract unchanged: the comparison would ask "is the live file higher than the last
    # three candidates were?" — and since a refused promotion never changes the live
    # file, that answer stays yes every hour thereafter, locking the loop into refusing
    # on a series that is not the candidate's (the review of SM_20260919_074839 called
    # exactly this: live 45 % against candidates 20 %/22 %, "a series that is not the
    # candidate's and that repeats every round"). Clause 5 is the answer for that
    # candidate — its own ratios are recorded — and the absolute ceilings above still
    # judge the text that would actually be landed, since `apply_overlay` lands SOUL.md.
    if shape_history is not None and (overlay_dir / "SOUL.md").is_file():
        errors += shape_ratchet_refusals(prompt_surface.contract_shape(soul), shape_history)
    return errors


def write_experiment_fact(
    cfg: AutoresearchConfig,
    variant: dict[str, Any],
    variant_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    snapshot_dir: Path,
) -> Path | None:
    """Write the promoted experiment as a fact file under cfg.paths.facts_experiments_dir/<id>/."""
    ex_dir = cfg.paths.facts_experiments_dir / variant["variant_id"]
    ex_dir.mkdir(parents=True, exist_ok=True)
    fact_file = ex_dir / f"{variant['variant_id']}-experiment.md"

    baseline_mean = baseline_summary.get("mean_composite", 0.0)
    variant_mean = variant_summary.get("mean_composite", 0.0)
    delta = variant_mean - baseline_mean

    frontmatter = {
        "type": "facts",
        "entity": variant["variant_id"],
        "category": "experiment",
        "last_updated": now_iso(),
        "facts": [
            {
                "fact": f"Autoresearch variant {variant['variant_id']} promoted ({variant.get('description', '')}). "
                        f"mean_composite {baseline_mean:.3f} → {variant_mean:.3f} (Δ {delta:+.3f}) over "
                        f"{variant_summary.get('task_count', 0)} bench tasks.",
                "confidence": 0.95,
                "category": "experiment",
                "id": f"exp-{variant['variant_id']}",
                "created_at": now_iso(),
                "valid_at": now_iso(),
                "invalid_at": None,
                "expired_at": None,
                "provenance": "EXTRACTED",
                "source_doc": str(snapshot_dir),
            }
        ],
    }
    import yaml

    body = (
        f"\n# {variant['variant_id']} - experiment\n\n"
        f"**Target surface:** {variant.get('target_surface', 'prompts')}\n"
        f"**Hypothesis:** {variant.get('hypothesis', '')}\n"
        f"**Snapshot:** `{snapshot_dir}`\n"
        f"**Baseline mean composite:** {baseline_mean:.3f}\n"
        f"**Variant mean composite:** {variant_mean:.3f}\n"
        f"**Delta:** {delta:+.3f}\n"
        f"**Task count:** {variant_summary.get('task_count', 0)}\n"
    )
    fact_file.write_text(f"---\n{yaml.safe_dump(frontmatter, sort_keys=False)}---\n{body}", encoding="utf-8")
    logger.info("wrote experiment fact to %s", fact_file)
    return fact_file


def _vault_relative_paths(names: list[str]) -> tuple[list[str], list[str], Path]:
    """Resolve prompt names to (the ones inside the vault, their vault-relative
    paths, the vault root) — one derivation shared by `promote()` and `rollback()`.

    Both write `CANONICAL_PROMPTS`, and both have to answer the same question
    before they reach for a commit: *are these files actually in the vault?* If
    any one target resolves outside it the answer is "no" for the whole set, so
    nothing is committed rather than half a contract.

    The `vault_round` import is deliberately **not** wrapped: a caller that
    cannot reach the landing route must see that as a failure, not be handed the
    `([], …)` answer that means "outside the vault, commit nothing".
    """
    from scripts.automod import vault_round as VR

    vault_root = VR.VAULT.resolve()
    kept: list[str] = []
    rel: list[str] = []
    for name in names:
        try:
            rel.append(str(CANONICAL_PROMPTS[name].resolve().relative_to(vault_root)))
        except ValueError:
            return [], [], vault_root
        kept.append(name)
    return kept, rel, vault_root


def promote(
    cfg: AutoresearchConfig,
    variant: dict[str, Any],
    variant_overlay_dir: Path,
    variant_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the full promotion pipeline. Returns a result dict."""
    result: dict[str, Any] = {
        "variant_id": variant["variant_id"],
        "dry_run": dry_run,
        "snapshot_dir": None,
        "applied_files": [],
        "experiment_fact": None,
    }
    # The ratchet is the only check here that needs a fact about *other* rounds, so
    # `promote()` is the one caller that can supply it. Read from the same ledger
    # the rounds write to — `record_round_summary` appends the shape row whether or
    # not anything promoted, so the series exists by the second round of a loop that
    # promotes nothing at all. The import is function-local because `post_promotion`
    # reads `CANONICAL_PROMPTS` from this module at import time; the cycle only ever
    # closes at call time, when both modules are already loaded.
    from .post_promotion import contract_shape_history

    # `cfg is None` is not a test convenience to accommodate: it is the shape of a
    # caller that has a candidate and no config, and such a caller must still get the
    # absolute ceilings rather than an unguarded write. What it cannot get is the
    # ratchet, whose only input is a ledger no config was named to point at.
    history = contract_shape_history(cfg.paths.ledger_path) if cfg else None
    refusals = contract_refusals(variant_overlay_dir, shape_history=history)
    if refusals:
        # Not an exception: a refused promotion is a normal outcome of a search
        # that proposed something out of bounds, and the round must carry on
        # evaluating the rest. It is logged at ERROR because a variant that wins
        # on bench score and still cannot be written is the signal that the
        # score and the constraint disagree — which is the whole finding.
        logger.error("REFUSED promotion of %s — %s",
                     variant["variant_id"], "; ".join(refusals))
        result["refused"] = refusals
        return result

    if dry_run:
        logger.info("[dry-run] would promote %s", variant["variant_id"])
        return result

    try:
        snap = snapshot_current_prompts(cfg)
    except (OSError, RuntimeError) as exc:
        logger.error("REFUSED promotion of %s — no rollback point: %s",
                     variant["variant_id"], exc)
        result["refused"] = [f"snapshot failed: {exc}"]
        return result
    applied = apply_overlay(variant_overlay_dir)
    result["snapshot_dir"] = str(snap)
    result["applied_files"] = applied

    # Commit through the vault route, which runs the real loaders and reverts
    # the paths if any of them fails. Before this, a promotion was an
    # uncommitted overwrite of a live tracked file: nothing recorded that it
    # had happened, `automod_vault_revert` had no sha to undo, and the only
    # trace was a snapshot directory nobody was told about. The sha goes in the
    # automod ledger as a `vault_land` event like every other vault change.
    try:
        from scripts.automod import vault_round as VR

        # Derive the vault-relative paths rather than assuming `lloyd/<name>`,
        # and commit nothing when the canonical prompts are not in the vault at
        # all. `CANONICAL_PROMPTS` is redirected by tests and by the variant
        # sandbox, and a hardcoded prefix would have made this function run
        # `git add` against the real `~/obsidian` from inside a unit test.
        _applied_in_vault, rel, vault_root = _vault_relative_paths(applied)
        if rel:
            landed = VR.land(
                rel,
                f"autoresearch: promote {variant['variant_id']}\n\n"
                f"{variant.get('description', '')}\n\n"
                f"Snapshot of the previous contract: {snap}",
            )
            result["vault_commit"] = landed.get("commit")
        else:
            # Not an error: the overlay sandbox and the tests both point the
            # canonical prompts somewhere that is not a git repo, and there is
            # nothing to commit there. The fact below is still written.
            logger.info("canonical prompts are outside %s — applied without a "
                        "vault commit", vault_root)
    except Exception as exc:  # VaultRoundError, or git refusing for any reason
        # `land` reverts the paths it validated before raising, so the live
        # contract is already back. Say so loudly and leave the snapshot.
        logger.error("promotion of %s did not land, contract restored: %s",
                     variant["variant_id"], exc)
        result["refused"] = [str(exc)]
        result["applied_files"] = []
        return result

    fact = write_experiment_fact(cfg, variant, variant_summary, baseline_summary, snap)
    result["experiment_fact"] = str(fact) if fact else None
    logger.info("promoted %s: applied=%s vault_commit=%s snapshot=%s",
                variant["variant_id"], applied, result.get("vault_commit"), snap)
    return result


def authored_bytes(vault_root: Path, sha: str, rel: str) -> bytes | None:
    """The bytes commit `sha` put at vault-relative `rel`, or None if it has none.

    #1099 clause 3 needs a reference for "did this file change after the promotion
    landed", and the only honest one is the promotion commit's own object: the vault's
    object store already holds what the landing wrote, so no side channel had to
    record a hash at promotion time, and a hash taken from anywhere else (HEAD before
    the copy, the snapshot, a caller's memory) would silently bless whatever it
    actually read. A commit that never touched the path returns None, which callers
    must treat as *no reference* — refuse, never "unchanged".
    """
    show = subprocess.run(["git", "-C", str(vault_root), "show", f"{sha}:{rel}"],
                          capture_output=True, check=False)
    if show.returncode != 0:
        return None
    return show.stdout


def changed_since_promotion(present_in_vault: list[str], rel: list[str],
                            vault_root: Path, vault_commit: str) -> tuple[list[str], list[dict]]:
    """Split restore candidates into `(safe_names, refused)`.

    `refused` is `({"file", "reason"}, ...)` with the file named exactly as the
    canonical-prompt key, because the clause is "refused **by name** and reported"
    — a count tells a reader nothing about which contract line was protected.
    """
    safe: list[str] = []
    refused: list[dict] = []
    for name, r in zip(present_in_vault, rel):
        target = CANONICAL_PROMPTS[name]
        try:
            current = target.read_bytes()
        except OSError as exc:
            refused.append({"file": name, "reason": f"unreadable: {exc}"})
            continue
        authored = authored_bytes(vault_root, vault_commit, r)
        if authored is None:
            refused.append({"file": name,
                            "reason": f"{vault_commit[:10]} wrote no {r}, so there is nothing "
                                      "to compare the live file against"})
        elif current != authored:
            refused.append({"file": name,
                            "reason": "changed after the promotion landed, so restoring the "
                                      "snapshot would overwrite an edit made afterwards"})
        else:
            safe.append(name)
    return safe, refused


def _differs_from_head(vault_root: Path, rel: list[str]) -> bool:
    """Would committing `rel` right now put anything new on the vault's `main`?

    Anything but exit 0 is read as "yes" — an untracked file has no HEAD blob to diff
    against, and a restore of a file the vault has never committed is a change, not a
    no-op. The one answer that must not be guessed at is `False`, because that is the
    answer that reports a restore as already-done.
    """
    if not rel:
        return False
    diff = subprocess.run(["git", "-C", str(vault_root), "diff", "--quiet", "HEAD", "--", *rel],
                          capture_output=True, check=False)
    return diff.returncode != 0


def _unattended_refusal_reason(promotion: dict[str, Any], rel: list[str],
                               vault_root: Path) -> str | None:
    """Why an unattended restore cannot proceed at all, or None if it can.

    Both answers here are about the route, not the content: an unattended restore has
    no human to notice a contract left modified-but-uncommitted, and no human to
    notice that "unchanged since the promotion" was compared against nothing.
    """
    if not rel:
        return (f"the canonical prompts are not tracked files under {vault_root}, so this "
                "restore has no validated vault landing route — an unattended restore "
                "never leaves a contract modified but uncommitted")
    if not str(promotion.get("vault_commit") or "").strip():
        return (f"no vault commit is on record for promotion "
                f"{promotion.get('variant_id') or '(unknown variant)'}, so nothing can be "
                "refused as 'changed after the promotion' and nothing can be trusted as "
                "unchanged")
    return None


def _rollback_message(snapshot_ts: str, snap: Path, restored: list[str],
                      refused_files: list[dict], promotion: dict[str, Any] | None) -> str:
    """The commit message for a restore, attributable to the decline that caused it.

    The manual tool's message is unchanged; an unattended one names the promotion's
    variant id and both round ids in the subject, because a vault sha that says only
    "rollback to snapshot <ts>" cannot be traced back to the round that decided it,
    which is the same attribution gap #1070 closed for the nightly sweep.
    """
    body = (f"Restored {', '.join(restored) or 'no files'} from {snap}. "
            "The vault route validated the restored contract before committing it, so a "
            "bad promotion is undone by one revert.")
    if promotion is None:
        return f"autoresearch: rollback to snapshot {snapshot_ts}\n\n{body}"
    refused = ("Refused, not overwritten: "
               + "; ".join(f"{r['file']} ({r['reason']})" for r in refused_files) + ".\n")
    return (
        "autoresearch: restore promotion "
        f"{promotion.get('variant_id')} (round {promotion.get('promoted_round_id')}) "
        "after a post-promotion decline\n\n"
        f"{body}\n\n"
        f"Decline {promotion.get('decline')} past the noise floor "
        f"{promotion.get('noise_floor')}, detected by round "
        f"{promotion.get('restoring_round_id')}.\n"
        f"{refused}"
    )


def rollback(cfg: AutoresearchConfig, snapshot_ts: str, *,
             promotion: dict[str, Any] | None = None) -> dict[str, Any]:
    """Restore canonical prompts from the named snapshot, through the vault route.

    This is the one undo path for a bad prompt promotion, and it used to be three
    `shutil.copy2` calls. Its targets are `CANONICAL_PROMPTS` — tracked files in
    the live vault — so a raw copy left the contract modified in the working tree
    while HEAD still pointed at the promotion commit, and nothing anywhere
    recorded that a restore had happened. `scripts/util/vault-commit.sh` was invoked
    from eleven call sites in eight nightly skills and staged the whole vault tree on
    every one of them until #1070, so the next job to commit a dirty vault landed the
    restore under *its* message:
    the blame-masking that let the 2026-09-10 MEMORY.md truncation sit undiscovered
    for 19 hours (see the clobber note in lloyd/MEMORY.md). The wrapper now stages
    exactly the paths a caller names — the route this function already takes through
    `scripts.automod.vault_round.land` — but a whole-tree sweep commit is still only
    *labelled*, not attributed, so a restore must never rely on one.

    The copy stays — a validated commit is not a revert, and the bytes have to be
    in the tree before anything can validate them — but
    `scripts.automod.vault_round.land` finishes the job the way every other vault
    write is finished: it runs the prompt-surface validators and the real loaders,
    puts the paths back if any of them fails, commits exactly the restored files
    on the vault's `main` with the snapshot ts in the message, and appends a
    `vault_land` ledger event carrying the sha. The sha is returned.

    **`promotion=` is the unattended caller's contract (#1099).** A restore driven by
    a round — `scripts/autoresearch/auto_restore.py`, reached from `run_round` when a
    fresh baseline declines past the noise floor — passes the promotion it is undoing:
    `{"variant_id", "promoted_round_id", "restoring_round_id", "vault_commit",
    "decline", "noise_floor"}`. Three things follow that a human clicking the
    `autoresearch_rollback` tool does not get asked for, because a human is the check:

      * every canonical file whose current bytes are not the bytes `vault_commit`
        wrote is **refused by name** instead of overwritten (`changed_since_promotion`),
        so an edit made after the promotion cannot be silently reverted;
      * no `vault_commit` on record means no restore — without it there is no
        reference to compare against, and "unchanged" guessed from the wrong
        reference is the failure this guard exists to prevent;
      * the commit message names the promotion's variant id, its round and the
        round that asked, so the sha is attributable to the decline that caused it.

    Without `promotion` the manual tool's behaviour is byte-for-byte what it was,
    including a raw copy when the canonical prompts sit outside the vault.
    """
    snap = cfg.paths.snapshots_dir / snapshot_ts
    result: dict[str, Any] = {"snapshot": str(snap), "restored_files": [],
                              "vault_commit": None, "no_change": False,
                              "refused_files": [], "promotion": dict(promotion or {})}
    if not snap.exists():
        result["error"] = f"snapshot {snapshot_ts} not found"
        return result

    present = [name for name in CANONICAL_PROMPTS if (snap / name).exists()]
    if not present:
        # A snapshot directory that holds no prompt file restores nothing and has
        # nothing to commit. `land` would refuse an empty path list; that refusal
        # is about a malformed call, not about this, so say the true thing here.
        logger.warning("snapshot %s holds none of %s — nothing restored",
                       snap, sorted(CANONICAL_PROMPTS))
        return result
    try:
        _present_in_vault, rel, vault_root = _vault_relative_paths(present)
    except Exception as exc:  # noqa: BLE001 — no landing route reachable: refuse, do not raw-copy
        result["refused"] = [f"cannot reach the vault landing route: {exc}"]
        logger.error("rollback to %s refused before touching anything: %s", snapshot_ts, exc)
        return result

    to_restore = list(present)
    rel_restore = list(rel)
    if promotion is not None:
        refusal = _unattended_refusal_reason(promotion, rel, vault_root)
        if refusal:
            result["refused"] = [refusal]
            logger.error("unattended restore of %s refused before touching anything: %s",
                         promotion.get("variant_id"), refusal)
            return result
        to_restore, result["refused_files"] = changed_since_promotion(
            _present_in_vault, rel, vault_root, str(promotion["vault_commit"]))
        refused_names = {r["file"] for r in result["refused_files"]}
        # Only the restored files go to `land`. `land` stages the paths it is given,
        # so a refused file with an uncommitted post-promotion edit in the tree would
        # otherwise be committed under this restore's message — the exact
        # attribution failure the guard above exists to prevent, arriving one call
        # later as a half-restored set that also silently shipped someone else's edit.
        rel_restore = [r for name, r in zip(_present_in_vault, rel)
                       if name not in refused_names]
        if not to_restore:
            result["refused"] = [f"every canonical file in {snap.name} was refused: "
                                 + "; ".join(f"{r['file']} ({r['reason']})"
                                             for r in result["refused_files"])]
            logger.error("unattended restore of %s refused: %s",
                         promotion.get("variant_id"), result["refused"][0])
            return result

    for name in to_restore:
        shutil.copy2(snap / name, CANONICAL_PROMPTS[name])
        result["restored_files"].append(name)

    if not rel:
        # Not an error for the manual tool: the variant sandbox and the unit tests
        # point the canonical prompts at a directory that is not the vault, where a
        # raw copy is all that can be done — and no nightly sweep commits that tree,
        # so there is nothing to mask. `promote()` has the same branch. An
        # unattended restore cannot reach here: `_unattended_refusal_reason` refuses it.
        logger.info("rolled back %s files from %s with no vault commit — those "
                    "paths are outside %s", len(result["restored_files"]), snap, vault_root)
        return result

    if not _differs_from_head(vault_root, rel_restore):
        # Content that already matches HEAD: a repeated rollback, or a snapshot that
        # *is* the live contract. Checked against the tree rather than by reading
        # `land`'s exception text, because with a partially refused restore a
        # message match would report the *restored* file as uncommitted work that
        # was never restored. A no-op is not a failed undo.
        result["no_change"] = True
        logger.info("rollback to %s restored content that already matches "
                    "HEAD — nothing to commit", snapshot_ts)
        return result

    try:
        from scripts.automod import vault_round as VR

        landed = VR.land(rel_restore, _rollback_message(snapshot_ts, snap, to_restore,
                                                        result["refused_files"], promotion))
    except Exception as exc:  # VaultRoundError, or git refusing for any reason
        if "nothing to commit" in str(exc):
            # Content that already matches HEAD: a repeated rollback, or a
            # snapshot that *is* the live contract. A no-op is not a failed undo,
            # and letting this raise would answer the tool's caller with an error
            # for having asked the same question twice.
            result["no_change"] = True
            logger.info("rollback to %s restored content that already matches "
                        "HEAD — nothing to commit", snapshot_ts)
            return result
        # `land` reverted the paths it validated before raising, so the tree is
        # back at HEAD and the live contract is the pre-rollback one. Report it as
        # a refusal rather than a restore that did not happen: an old snapshot
        # predating a structural change must be refused, not silently applied.
        logger.error("rollback to %s refused; contract restored: %s", snapshot_ts, exc)
        result["refused"] = [str(exc)]
        result["restored_files"] = []
        return result

    result["vault_commit"] = landed.get("commit")
    logger.info("rolled back %s files to %s: vault_commit=%s",
                len(result["restored_files"]), snapshot_ts, result["vault_commit"])
    return result
