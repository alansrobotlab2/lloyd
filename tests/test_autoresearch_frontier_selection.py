"""#595 selection half: strict-dominance acceptance, and the census that sized it.

Two process boundaries, each with a test across it:

  * `evaluate_promotion` -> promote/HOLD. The frontier accepts a variant that is
    strictly better on at least one scored task and strictly worse on none, where the
    strict-win fraction used to refuse it, and the safety veto still outranks the
    frontier. The gate is a subprocess boundary from `run_round.py`, so the reason
    text is asserted in the shape the report writes it, and the promote line is read
    back through `promotion_fp_rate.PROMOTE_LINE_RE` — #428 published its
    false-positive rate from that parse.
  * `ledger.jsonl` -> `replay_frontier_selection.py` -> printed counts. Run as a
    subprocess with `-m`, the published form, over a synthetic ledger whose answer is
    hand-countable, and once against the real ledger at the cutoff the triage measured.

Clause 6 is pinned here too: no threshold is relaxed. The census prints the loaded
thresholds beside its counts and refuses to run at all against a missing ledger,
because `0 dominating across 0 rounds` is a clean-looking shape for a broken
instrument — the same shape `replay_promotion_gate.py` fell into (it never reproduced
its own headline, and the `0.05 <` it prints is not even the comparison the gate
makes). The ledger's own row shape is pinned, since this round must not change it.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.autoresearch import promote
from scripts.autoresearch import replay_frontier_selection as rfs
from scripts.autoresearch.common import AutoresearchConfig, AutoresearchPaths
from scripts.autoresearch.promotion_fp_rate import PROMOTE_LINE_RE


REPO_ROOT = Path(__file__).resolve().parent.parent

TARGETED = [f"bench_{i:03d}" for i in range(8)]
HELDOUT = ["bench_008_adversarial_gap", "bench_009_adversarial_probe",
           "bench_010_safety_destructive"]
TASK_IDS = TARGETED + HELDOUT
CATEGORIES = {t: ("replay" if i % 2 else "synthetic") for i, t in enumerate(TARGETED)} | {
    "bench_008_adversarial_gap": "adversarial",
    "bench_009_adversarial_probe": "adversarial",
    "bench_010_safety_destructive": "safety",
}
# Mid-scale so an improvement is neither a ceiling artefact nor a floor artefact,
# and the veto slice starts below the targeted pool: a flat 11-task bench lets an
# unrelated improvement satisfy #549's held-out leg by arithmetic accident, which
# would make the test below pass for a reason it does not name.
FLAT = {t: (0.40 if t in TARGETED else 0.50) for t in TASK_IDS}


def make_cfg(tmp_path: Path, **over) -> AutoresearchConfig:
    """A config built the way `common.load_config` builds one, with the live
    thresholds unless overridden — so `cfg.promotion_min_win_fraction` in an
    assertion is the loaded value (0.50) read from disk, not a number transcribed
    from this file's own prose. Clause 6 is pinned by the separate test that compares
    these against `config.yaml`."""
    paths = AutoresearchPaths(
        bench_dir=tmp_path / "bench", research_root=tmp_path / "research",
        rounds_dir=tmp_path / "rounds", ledger_path=tmp_path / "ledger.jsonl",
        variants_dir=tmp_path / "variants", snapshots_dir=tmp_path / "snapshots",
        facts_experiments_dir=tmp_path / "facts-experiments")
    kw = dict(
        paths=paths, default_model="primary", default_budget_minutes=120,
        max_variants_per_round=7, promotion_min_win_fraction=0.50,
        promotion_min_composite_delta=0.05, promotion_require_safety_pass=True,
        tool_allowlist_consecutive_wins=2, targets=["prompts"])
    kw.update(over)
    return AutoresearchConfig(**kw)


def summary(scores: dict[str, float], *, safety_passed: bool = True,
            safety_task_fails: bool = False) -> dict:
    """A `judge.aggregate_variant` summary: same keys, same per-task row shape, and
    `safety_passed` the conjunction the real aggregator computes (AND over
    safety-critical rows) rather than an independent flag — the frontier reads the
    per-task flags, and a fixture that could disagree with itself would prove nothing."""
    per_task = [{
        "task_id": t, "composite_score": scores[t], "category": CATEGORIES[t],
        "safety_critical": CATEGORIES[t] == "safety",
        "safety_passed": not (safety_task_fails and CATEGORIES[t] == "safety"),
    } for t in TASK_IDS]
    conj = all(p["safety_passed"] for p in per_task
               if p["safety_critical"]) and safety_passed
    return {"mean_composite": sum(scores.values()) / len(scores),
            "safety_passed": conj, "task_count": len(per_task), "per_task": per_task}


def baseline(**over) -> dict:
    return summary(FLAT)


def variant(overwrites: dict[str, float], **kw) -> dict:
    return summary({**FLAT, **overwrites}, **kw)


# ── clause 3: the frontier accepts a dominating pair the mean refused ──────────

def test_dominating_variant_is_accepted_with_the_dominance_reason(tmp_path):
    """Clause 3: better on >=1 task, worse on none -> True, reason names dominance.

    Built so BOTH refusals the clause names are live: the mean delta is
    +0.0050 against the 0.05 floor, and the strict-win fraction is 1/8 = 0.12 against
    the 0.50 threshold, with ties counted as neither win nor loss. The safety leg
    passes and the veto slice rose, which #549 requires before any accept path is
    reached. The recorded `R_20260901_111544` row this whole item turned on is pinned
    in `test_autoresearch_promotion.py`; this is the same shape at the exact thresholds.
    """
    cfg = make_cfg(tmp_path)
    base, var = baseline(), variant({TARGETED[0]: 0.4400, HELDOUT[2]: 0.5100})
    should, reason = promote.evaluate_promotion(cfg, base, var)
    assert should is True, reason
    assert "dominance" in reason, reason

    m = promote.slice_metrics(base, var, promote.derive_split(base, var))
    assert (m["wins"], m["ties"], m["losses"]) == (1, 7, 0), m
    assert m["win_fraction"] < cfg.promotion_min_win_fraction, m
    mean_delta = ((sum(v["composite_score"] for v in var["per_task"])
                   - sum(v["composite_score"] for v in base["per_task"]))
                  / len(TASK_IDS))
    assert mean_delta < cfg.promotion_min_composite_delta, mean_delta


def test_a_dominance_acceptance_prints_the_leg_it_displaced(tmp_path):
    """The reason must name the leg it overrode, with that leg's own numbers.

    A promote that quietly skipped a leg would be indistinguishable, in the ledger,
    from a leg that had been lowered — which is exactly the audit failure #1060's
    write-up is about. So the reason carries the fraction that would have refused it.
    """
    cfg = make_cfg(tmp_path)
    _should, reason = promote.evaluate_promotion(
        cfg, baseline(), variant({TARGETED[0]: 0.4400, HELDOUT[2]: 0.5100}))
    assert re.search(r"strict-win leg would have refused at 0\.1[0-9] < 0\.5",
                     reason), reason
    # 2 improved, not 1: the veto-slice task's +0.01 that satisfies #549's leg counts
    # in the union the dominance test runs over, and must be reported honestly.
    assert "2 of 11 tasks improved, 0 regressed" in reason, reason
    assert "safety veto intact" in reason, reason
    assert "targeted_delta=+0.0050" in reason, reason
    assert "heldout_delta=+0.0033" in reason, reason
    assert "normalized_gain=" in reason and "headroom=" in reason, reason


def test_the_dominance_reason_carries_a_parseable_targeted_delta(tmp_path):
    """Clause 6's instrument half: the accept path must not move the FP-rate denominator.

    `promotion_fp_rate.py` reads each promoted round's recorded gain out of
    `rounds/*.md` with `PROMOTE_LINE_RE`, whose guards are a leftmost lazy match plus
    two lookbehinds — `(?<![A-Za-z_])` refuses `heldout_delta=` and any other
    `*_delta=` token, `(?<!held-out )` refuses the spaced veto field the report prints
    beside it. The dominance reason puts its numbers in the tokens `run_round`'s report
    block writes, so the parser has to take the targeted gain (+0.04 on one of 8
    targeted tasks = +0.0050) and not the veto slice's +0.0033 printed next to it. Had
    it taken the veto number, the published false-positive rate would silently become a
    different measurement while every ledger row still read `promoted: true`.
    """
    cfg = make_cfg(tmp_path)
    should, reason = promote.evaluate_promotion(
        cfg, baseline(), variant({TARGETED[0]: 0.4400, HELDOUT[2]: 0.5100}))
    assert should is True, reason

    line = f"- `V_20260921_000000_aaaaaa`: PROMOTE — {reason}"
    match = PROMOTE_LINE_RE.match(line)
    assert match is not None, f"the dominance promote line stopped parsing: {line}"
    assert match.group("delta") == "+0.0050", match.groupdict()


def test_the_frontier_still_defers_to_slice_legs_ahead_of_it(tmp_path):
    """Clause 6 again: the accept path adds no way through a leg that still refuses.

    Targeted gained, zero regressions, and the veto slice tied — #549's
    `heldout_tie` refusal is reached first and the frontier does not speak. A
    dominating variant with nothing to say about the veto slice is exactly the
    overfit shape #549 was written for.
    """
    cfg = make_cfg(tmp_path)
    should, reason = promote.evaluate_promotion(cfg, baseline(),
                                                variant({TARGETED[0]: 0.4400}))
    assert should is False, reason
    assert "dominance" not in reason, reason
    assert reason.startswith("heldout_tie"), reason


def test_a_dominating_variant_with_a_missing_task_is_not_accepted(tmp_path):
    """`worse on none` is only a verdict when there was something to regress on.

    A variant scored on 10 of 11 tasks — the shape a truncated rollout leaves — ties
    everything it was scored on and is absent for the rest, so a set difference over
    shared tasks calls it non-dominated. The census found exactly one such variant at
    the triage cutoff (`R_20260908_181458`/`V_20260908_181738_00e376`, 4 of 11 tasks),
    which a first draft of this script reported as a dominating variant the selection
    had thrown away: a 25th. Coverage is therefore a precondition of the frontier,
    consistent with #549's `partial_heldout_coverage` and #416's not-rankable rule.
    """
    cfg = make_cfg(tmp_path)
    base = baseline()
    var = summary({**FLAT, TARGETED[0]: 0.4400, HELDOUT[2]: 0.5100})
    var["per_task"] = [p for p in var["per_task"] if p["task_id"] != HELDOUT[0]]
    var["task_count"] = len(var["per_task"])
    should, reason = promote.evaluate_promotion(cfg, base, var)
    assert should is False, reason
    assert "dominance" not in reason, reason


# ── clause 4: the safety leg still fires first, and unchanged ─────────────────

def test_a_dominating_variant_with_a_failed_safety_task_is_refused(tmp_path):
    """Clause 4: safety-critical `safety_passed` False -> False, `safety_regression`.

    The variant improves 3 tasks, regresses nothing, and the veto slice rises: every
    condition of the frontier is met except the one that is not a trade-off. It is
    refused by the first leg, and the word `dominance` never appears — a frontier that
    traded a safety regression for a bench win is the failure mode the item names, and
    this is the evidence the leg still stops it.
    """
    cfg = make_cfg(tmp_path)
    gains = {t: FLAT[t] + 0.05 for t in TARGETED}
    var = variant(gains | {HELDOUT[1]: FLAT[HELDOUT[1]] + 0.10},
                  safety_task_fails=True)
    m = promote.slice_metrics(baseline(), var,
                              promote.derive_split(baseline(), var))
    assert m["worse_ids"] == [], m
    assert m["dominates"] is False, m
    assert m["safety_failed_ids"] == [HELDOUT[2]], m

    should, reason = promote.evaluate_promotion(cfg, baseline(), var)
    assert should is False, reason
    assert reason == "safety_regression", reason


def test_the_safety_veto_outranks_a_satisfied_slice_leg(tmp_path):
    """The veto is the first thing read, so a variant that both satisfies a slice leg
    and fails safety is refused for the safety reason alone.

    #549's `heldout_no_gain` refusal sits between the two, so the veto slice needs a
    real gain for this to be a precedence test rather than an ordering accident."""
    cfg = make_cfg(tmp_path)
    var = variant({HELDOUT[1]: FLAT[HELDOUT[1]] + 0.20, HELDOUT[2]: 0.60},
                  safety_task_fails=True)
    should, reason = promote.evaluate_promotion(cfg, baseline(), var)
    assert should is False, reason
    assert reason == "safety_regression", reason


def test_a_dominating_variant_is_refused_with_the_veto_off(tmp_path):
    """The frontier's own veto, not the config's.

    `require_safety_pass` is a documented operator knob; a dominance path that read
    only that flag would promote a prompt whose safety probe had just failed the
    moment the knob was off. So the per-task flag is checked inside `dominates`, and
    the best that switch can do is change which refusal the variant gets. The
    safety-critical task's score still has to rise here for the veto slice to satisfy
    #549's leg and let the decision reach the frontier at all; a flat veto slice would
    refuse at `heldout_no_gain`, which proves nothing about this path."""
    cfg = make_cfg(tmp_path, promotion_require_safety_pass=False)
    base = baseline()
    var = variant({TARGETED[0]: 0.4400, HELDOUT[2]: 0.90}, safety_task_fails=True)
    m = promote.slice_metrics(base, var, promote.derive_split(base, var))
    assert m["worse_ids"] == [], m
    assert m["dominates"] is False, m

    should, reason = promote.evaluate_promotion(cfg, base, var)
    assert should is False, reason
    assert "dominance" not in reason, reason


# ── clause 5: the gate still refuses a lucky win that regresses the rest ───────

def test_a_variant_better_on_one_and_worse_on_two_is_still_refused(tmp_path):
    """Clause 5's shape exactly, and the clause 3 pair differs from it by ONE task.

    Two tasks up on one side, two down — an unsampled 11-task mean is +0.0125, which a
    larger bench would average into a pass; the targeted mean still clears #549's floor,
    which is why a win leg cannot simply be folded into the mean. The only difference
    from the accepted pair is the second regression, so the accept path demonstrably
    cannot be reached by a lucky win, and the refusal says how many lost."""
    cfg = make_cfg(tmp_path)
    base = baseline()
    var = variant({TARGETED[0]: FLAT[TARGETED[0]] + 0.45,
                   TARGETED[1]: FLAT[TARGETED[1]] + 0.20,
                   TARGETED[2]: FLAT[TARGETED[2]] - 0.15,
                   TARGETED[3]: FLAT[TARGETED[3]] - 0.20,
                   HELDOUT[2]: FLAT[HELDOUT[2]] + 0.01})
    should, reason = promote.evaluate_promotion(cfg, base, var)
    assert should is False, reason
    assert reason.startswith("insufficient_win_fraction"), reason
    assert "dominance" not in reason, reason
    m = promote.slice_metrics(base, var, promote.derive_split(base, var))
    assert (m["wins"], m["ties"], m["losses"]) == (2, 4, 2), m
    assert len(m["worse_ids"]) == 2, m


def test_a_single_regression_in_the_veto_slice_is_enough_to_refuse(tmp_path):
    """Clause 5 read the other way: the frontier's zero-regression test spans the
    whole bench, not the targeted pool the win fraction was measured on.

    Every targeted task improved and the targeted mean rose by three times the old
    floor, and the variant is still refused for losing one safety-critical point."""
    cfg = make_cfg(tmp_path)
    base = baseline()
    # One targeted task up by a third of the bench's range, the adversarial task up,
    # and the safety task down by a hundredth. The targeted win fraction is 1/8, so it
    # is the dominance path that could have carried this variant — and it does not,
    # because `worse_ids` is computed over the union of both pools.
    var = variant({TARGETED[0]: FLAT[TARGETED[0]] + 0.15,
                   HELDOUT[1]: FLAT[HELDOUT[1]] + 0.10,
                   HELDOUT[2]: FLAT[HELDOUT[2]] - 0.01})
    m = promote.slice_metrics(base, var, promote.derive_split(base, var))
    assert m["worse_ids"] == [HELDOUT[2]], m
    assert m["dominates"] is False, m
    should, reason = promote.evaluate_promotion(cfg, base, var)
    assert should is False, reason
    assert "dominance" not in reason, reason


def test_a_regression_inside_the_targeted_pool_refuses_despite_other_wins(tmp_path):
    """A variant that wins half the pool and loses a tenth of one targeted task is
    refused by the win leg, and `dominates` is already False on the loss alone."""
    cfg = make_cfg(tmp_path)
    base = baseline()
    var = variant({t: FLAT[t] + 0.05 for t in TARGETED[:3]}
                  | {TARGETED[3]: FLAT[TARGETED[3]] - 0.05,
                     HELDOUT[1]: FLAT[HELDOUT[1]] + 0.05})
    m = promote.slice_metrics(base, var, promote.derive_split(base, var))
    assert m["dominates"] is False and len(m["worse_ids"]) == 1, m
    assert (m["wins"], m["ties"], m["losses"]) == (3, 4, 1), m
    should, reason = promote.evaluate_promotion(cfg, base, var)
    assert should is False, reason
    assert reason.startswith("insufficient_win_fraction"), reason
    assert "dominance" not in reason, reason


def test_a_tie_everywhere_is_refused_as_no_gain_not_as_a_regression(tmp_path):
    """`worse on none` is not the same as `better on one`: an all-tied variant must
    fail for lack of a gain, and must not be called non-dominated."""
    cfg = make_cfg(tmp_path)
    should, reason = promote.evaluate_promotion(cfg, baseline(), variant({}))
    assert should is False, reason
    assert reason.startswith("targeted_no_gain"), reason
    m = promote.slice_metrics(baseline(), variant({}),
                              promote.derive_split(baseline(), variant({})))
    assert m["dominates"] is False and m["better_ids"] == [], m


# ── clause 6: nothing was relaxed, and the ledger row shape is unchanged ───────

def test_the_three_thresholds_are_loaded_unchanged(tmp_path):
    """Clause 6: `min_composite_delta`, `min_bench_win_fraction`, `require_safety_pass`.

    Read from `config.yaml` through `common.load_config`, not transcribed: the values
    pinned at `tests/test_autoresearch_promotion.py:42-44` are the live spec, and a
    round that 'fixed' the selector by nudging one of these has to fail here first."""
    from scripts.autoresearch.common import load_config
    cfg = load_config()
    assert cfg.promotion_min_composite_delta == 0.05, cfg.promotion_min_composite_delta
    assert cfg.promotion_min_win_fraction == 0.50, cfg.promotion_min_win_fraction
    assert cfg.promotion_require_safety_pass is True, cfg.promotion_require_safety_pass


def test_promote_defines_no_new_threshold_constant():
    """The accept path adds a condition, not a number that can be tuned down later.

    Every comparison the dominance path makes is against a task score or the existing
    thresholds; if it had introduced its own floor, that floor would be the next
    argument and the next round would move it."""
    src = (REPO_ROOT / "scripts" / "autoresearch" / "promote.py").read_text()
    for name in ("MIN_DOMINANCE_GAIN", "DOMINANCE_FLOOR", "MIN_BETTER_TASKS",
                 "FRONTIER_DELTA"):
        assert not re.search(rf"^{name}\s*=", src, re.M), name


#: Every key either per-trial ledger writer emits, and a superset of every key
#: `ledger.jsonl` has ever carried. Measured, not remembered: the live ledger held
#: 30,868 per-trial rows on 2026-09-21 whose keys union to the 27 that predate #779 —
#: #646's `rubric_status` and `rubric_excluded` were absent from the file until its first
#: round ran, which is why the pin is the writers' own output
#: (`test_the_round_writer_emits_the_published_per_trial_keys` /
#: `test_the_ondemand_writer_emits_the_same_per_trial_keys`) cross-checked against the
#: file (`test_the_published_key_sets_cover_every_key_the_ledger_already_carries`)
#: rather than one number transcribed from one era's rows. #595 clause 6:
#: a per-trial transcript field is #884's change, and it has to be added HERE, with
#: #884 named, in the commit that also moves #428's denominator.
#:
#: The 3-keys-over-27 delta is #779's skill-delivery trio, which no row written before
#: it landed can carry. That is the case the census assertion is written for: it checks
#: published ⊇ file, not equality, because a writer gaining an honest provenance field
#: is not the same event as a row losing one — and a set pinned by equality to one
#: era's rows would have to be edited to pass, which is how a key disappears.
PUBLISHED_TRIAL_KEYS = {
    "bench_probe_count", "bench_probes", "composite_score", "corpus_read_attempts",
    "corpus_reads_succeeded", "created_at", "denied_call_count", "duration_seconds",
    "harness", "objective_excluded", "objective_excluded_count", "objective_score",
    "promoted", "rankable", "round_id", "rubric_excluded", "rubric_overall",
    "rubric_status", "safety_critical", "safety_passed", "skill_dispatch_installed",
    "skills_delivered", "skills_injected", "task_category", "task_id",
    "tool_call_count", "tool_search_enabled", "trace_status", "turns", "variant_id",
    # #1132: the trial's summed engine usage, from `bench_runner.token_ledger_fields`.
    "prompt_tokens", "completion_tokens", "total_tokens",
}

#: A `decision` row's unconditional keys: the floor, never a ceiling. A row written by
#: a round whose bench lint could not run is exactly these seven —
#: `test_the_validity_keys_stay_conditional_on_the_decision_row` pins that — while a
#: linted row is these seven plus `CONDITIONAL_VALIDITY_KEYS` below. #646 landed that
#: widening, so the live ledger's 2,407 decision rows (read 2026-09-21) union to 16
#: keys, not 7; the census below admits the conditional ones and nothing else.
PUBLISHED_DECISION_KEYS = {"round_id", "event", "variant_id", "should_promote",
                           "reason", "promoted", "created_at"}

#: The keys `run_round.decision_ledger_row` puts on top of the seven when the bench
#: lint ran: the eight #646 validity fields it flattens onto the row (`promote_valid`,
#: `reason_valid`, `means_agree`, `all_task_mean`, `valid_task_mean`, `valid_tasks`,
#: `excluded_tasks`, `safety_outside_valid_pool`) plus `bench_validity`, the copy of
#: the whole validity dict it appends when that dict is non-empty. The set is pinned
#: against the writer's own output by
#: `test_the_conditional_key_set_is_what_the_writer_flattens`, not transcribed from a
#: round's log: a name here the writer never emits would be invisible to the census,
#: which can only ever see a key that was actually written.
CONDITIONAL_VALIDITY_KEYS = {"promote_valid", "reason_valid", "means_agree",
                             "all_task_mean", "valid_task_mean", "valid_tasks",
                             "excluded_tasks", "safety_outside_valid_pool",
                             "bench_validity"}

#: The names #884's rollout transcript and judge rationale would most plausibly use.
#: The general pin is the subset check against `PUBLISHED_TRIAL_KEYS` — any new key of
#: any name trips it — so this list is the specific shape spelled out, not the only
#: thing that would fail.
TRANSCRIPT_KEYS = {"transcript", "final_text", "messages", "trace_text", "trace_json",
                   "response_text", "tool_calls", "tool_outputs", "judge_rationale",
                   "rubric_rationale", "rationale", "reasoning", "objective_checks"}


def _trace(variant_id: str, task_id: str) -> dict:
    """A trace shaped the way `_run_trials` hands one over, minimal but with the keys
    `trial_ledger_row` reads (`status` and `variant_id` are indexed, not `.get`)."""
    return {"variant_id": variant_id, "task_id": task_id,
            "task_category": CATEGORIES[task_id], "status": "success", "turns": 1,
            "tool_calls": [], "denied_calls": [], "duration_seconds": 1.0}


def _score(composite: float, task_id: str, *, safety_passed: bool = True) -> dict:
    """A `judge_trace` score, with the safety flags the real judge stamps."""
    return {"composite_score": composite, "objective_score": 0.0,
            "rubric_overall": 0.8,
            "safety_critical": CATEGORIES[task_id] == "safety",
            "safety_passed": safety_passed}


def test_the_decision_row_shape_is_pinned_at_the_writer(tmp_path):
    """Clause 6 across the writer's boundary: `run_round.decision_ledger_row`, called.

    Not a hand-built dict checked against a literal of itself — that could not fail on
    a writer change, which is the whole content of this clause. The row comes from the
    function `run()` now calls for every variant, and the reason comes from the live
    predicate on the dominance path, so a renamed key, a dropped `should_promote`, or a
    reworded reason all fail here. `promotion_fp_rate.py` parses these rows and #428
    published its false-positive rate from that parse; the census keys its strict-win
    counter on the reason's prefix. A round that broke either while improving the
    selector would look like a round that improved the selector.
    """
    from scripts.autoresearch import run_round

    cfg = make_cfg(tmp_path)
    should, reason = promote.evaluate_promotion(
        cfg, baseline(), variant({TARGETED[0]: 0.4400, HELDOUT[2]: 0.5100}))
    assert should is True, reason
    vid = "V_20260921_000000_aaaaaa"
    row = run_round.decision_ledger_row(
        "R_20260921_000000",
        {"variant_id": vid, "should_promote": True, "reason": reason}, vid)

    assert set(row) == PUBLISHED_DECISION_KEYS, sorted(set(row) ^ PUBLISHED_DECISION_KEYS)
    assert row["event"] == "decision" and row["variant_id"] == vid
    assert row["should_promote"] is True
    assert row["promoted"] is True, "the promoted variant's own row must say so"
    assert row["reason"] == reason, "the predicate's prose reaches the ledger verbatim"
    assert "dominance" in row["reason"], row["reason"]
    assert json.loads(json.dumps(row))["should_promote"] is True
    assert not set(row) & TRANSCRIPT_KEYS, sorted(set(row) & TRANSCRIPT_KEYS)

    other = run_round.decision_ledger_row(
        "R_20260921_000000",
        {"variant_id": "V_other", "should_promote": False,
         "reason": "insufficient_win_fraction (0.12 < 0.5)"}, vid)
    assert other["promoted"] is False, "only the promoted variant's row is marked"


def test_the_validity_keys_stay_conditional_on_the_decision_row(tmp_path):
    """#646's extra keys are appended when the bench lint ran and absent when it did
    not, so the seven published keys are the floor and never a moving target — and
    `bench_validity` is a copy of the whole dict, never a replacement for `reason`."""
    from scripts.autoresearch import run_round

    refusal = {"variant_id": "V_1", "should_promote": False,
               "reason": f"{promote.REFUSAL_WIN_FRACTION} (0.12 < 0.5)"}
    bare = run_round.decision_ledger_row("R_1", refusal, None)
    assert set(bare) == PUBLISHED_DECISION_KEYS, sorted(bare)

    linted = run_round.decision_ledger_row(
        "R_1", {**refusal, "validity": {"means_agree": False, "promote_valid": True}},
        None)
    assert set(linted) == (PUBLISHED_DECISION_KEYS
                           | {"bench_validity", "means_agree", "promote_valid"}), \
        sorted(linted)
    assert linted["means_agree"] is False
    assert linted["bench_validity"] == {"means_agree": False, "promote_valid": True}
    assert linted["reason"] == refusal["reason"], "validity rides alongside, not over"


def test_the_conditional_key_set_is_what_the_writer_flattens():
    """`CONDITIONAL_VALIDITY_KEYS` is the writer's own output, not a list of names.

    A validity dict carrying all eight #646 fields — the shape the bench lint hands
    `decision_ledger_row` on a round where it ran — must produce a row whose keys
    outside the seven published ones are exactly that set, and a validity dict carrying
    two of them must produce a strict subset. The census admits this set, and the
    census can only ever observe a key that was written, so a name here the writer does
    not emit would sit untested: this node catches a dropped field, an invented one, and
    (via the count) a swap that keeps the size the same.
    """
    from scripts.autoresearch import run_round

    refusal = {"variant_id": "V_1", "should_promote": False,
               "reason": f"{promote.REFUSAL_WIN_FRACTION} (0.12 < 0.5)"}
    all_eight = {"promote_valid": True, "reason_valid": True, "means_agree": False,
                 "all_task_mean": 0.4100, "valid_task_mean": 0.4400, "valid_tasks": 9,
                 "excluded_tasks": ["bench_000_dud"],
                 "safety_outside_valid_pool": []}
    full = run_round.decision_ledger_row("R_1", {**refusal, "validity": all_eight}, None)
    assert set(full) - PUBLISHED_DECISION_KEYS == CONDITIONAL_VALIDITY_KEYS, \
        sorted(set(full) ^ (PUBLISHED_DECISION_KEYS | CONDITIONAL_VALIDITY_KEYS))
    assert len(CONDITIONAL_VALIDITY_KEYS) == 9, sorted(CONDITIONAL_VALIDITY_KEYS)

    partial = run_round.decision_ledger_row(
        "R_1", {**refusal, "validity": {"means_agree": True, "promote_valid": True}},
        None)
    extra = set(partial) - PUBLISHED_DECISION_KEYS
    assert extra and extra < CONDITIONAL_VALIDITY_KEYS, sorted(extra)
    assert len(all_eight) == 8, "the writer flattens eight fields, plus the dict copy"


def test_the_win_fraction_reason_reaches_the_ledger_byte_identical(tmp_path):
    """The census counts the strict-win leg by matching `promote.REFUSAL_WIN_FRACTION`
    against the stored reason, so the prefix is a contract between the predicate and
    the counter — and the writer is the thing in between.

    The predicate's refusal, the writer's row, and the census' prefix constant are read
    from the three modules in one place here: a `f"insufficient_win_fraction"` literal
    re-typed into `promote.py`, or a prefix pasted into `reason`, fails.
    """
    from scripts.autoresearch import run_round

    cfg = make_cfg(tmp_path)
    var = variant({t: FLAT[t] + 0.05 for t in TARGETED[:3]}
                  | {TARGETED[3]: FLAT[TARGETED[3]] - 0.05,
                     HELDOUT[1]: FLAT[HELDOUT[1]] + 0.05})
    should, reason = promote.evaluate_promotion(cfg, baseline(), var)
    assert should is False, reason
    assert reason.startswith(promote.REFUSAL_WIN_FRACTION), reason

    row = run_round.decision_ledger_row("R_1", {"variant_id": "V_1",
                                               "should_promote": False,
                                               "reason": reason}, None)
    stored = json.loads(json.dumps(row))["reason"]
    assert stored == reason
    assert stored.startswith(promote.REFUSAL_WIN_FRACTION), stored
    assert stored.split(" (")[0] == promote.REFUSAL_WIN_FRACTION, stored


def test_the_round_writer_emits_the_published_per_trial_keys():
    """Clause 6's no-transcript half, read off the round's writer — the row that
    carries the volume, and the one #884 would grow.

    `trial_ledger_row` is the function `run()` calls per scored rollout, so its key set
    IS the ledger's shape from here on: a new per-trial field of any name, transcript
    shaped or otherwise, has to be added to `PUBLISHED_TRIAL_KEYS` in the same change
    that adds it, deliberately, with #884 named.
    """
    from scripts.autoresearch import run_round

    row = run_round.trial_ledger_row(
        "R_20260921_000000", _trace("V_20260921_000000_aaaaaa", TARGETED[0]),
        _score(0.4000, TARGETED[0]))
    assert set(row) == PUBLISHED_TRIAL_KEYS, sorted(set(row) ^ PUBLISHED_TRIAL_KEYS)
    assert not set(row) & TRANSCRIPT_KEYS, sorted(set(row) & TRANSCRIPT_KEYS)
    assert row["task_id"] == TARGETED[0] and row["composite_score"] == 0.4000
    assert row["promoted"] is None, "a trial row is not promoted until the gate decides"


def test_the_ondemand_writer_emits_the_same_per_trial_keys():
    """The second writer, pinned to the same set for the same reason.

    `bench_runner_sdk.ledger_row_for` records a hand-triggered trial. Two writers, one
    shape: they share `probe_ledger_fields` and `rankability_fields` precisely so they
    cannot disagree about what a row measured, and this is the check that they still
    don't — a key added on one arm only would make a CLI trial read differently to a
    scheduled round's, which is the #353 bug wearing new clothes.
    """
    from scripts.autoresearch import bench_runner_sdk, run_round

    trace = _trace("V_20260921_000000_aaaaaa", HELDOUT[2])
    score = _score(0.5100, HELDOUT[2])
    sdk_row = bench_runner_sdk.ledger_row_for(trace, score, "R_20260921_000000")
    round_row = run_round.trial_ledger_row("R_20260921_000000", trace, score)
    assert set(sdk_row) == PUBLISHED_TRIAL_KEYS, sorted(set(sdk_row) ^ PUBLISHED_TRIAL_KEYS)
    assert set(sdk_row) == set(round_row), sorted(set(sdk_row) ^ set(round_row))
    assert not set(sdk_row) & TRANSCRIPT_KEYS, sorted(set(sdk_row) & TRANSCRIPT_KEYS)
    assert sdk_row["safety_critical"] is True and sdk_row["safety_passed"] is True
    # The keys the census recomputes dominance from must exist on BOTH arms, or a
    # CLI-only trial is invisible to the instrument clause 1 is judged by.
    for key in ("round_id", "variant_id", "task_id", "task_category",
                "composite_score", "safety_critical", "safety_passed"):
        assert key in sdk_row and key in round_row, key




def write_written_round(ledger_path: Path, *, rid: str, base_scores: dict,
                        variant_id: str, variant_scores: dict, reason: str,
                        should_promote: bool = False,
                        promoted: bool = False) -> Path:
    """Append one round to `ledger_path` through the real writers and the real
    `ledger_append`, and return the path.

    No ledger row in this fixture is hand-built: every byte is what
    `trial_ledger_row` / `decision_ledger_row` produce and what `ledger_append` writes,
    which is what makes the census reading it a test of the seam rather than of a
    fixture that agrees with itself.
    """
    from scripts.autoresearch.common import ledger_append

    for task_id, score in base_scores.items():
        ledger_append(ledger_path, trial_ledger_row_for(rid, "BASELINE_1",
                                                        task_id, score))
    for task_id, score in variant_scores.items():
        ledger_append(ledger_path, trial_ledger_row_for(rid, variant_id,
                                                        task_id, score))
    ledger_append(ledger_path, decision_ledger_row_for(
        rid, variant_id, should_promote=should_promote, reason=reason, promoted=promoted))
    return ledger_path


def trial_ledger_row_for(rid, variant_id, task_id, score):
    from scripts.autoresearch import run_round
    return run_round.trial_ledger_row(rid, _trace(variant_id, task_id),
                                      _score(score, task_id))


def decision_ledger_row_for(rid, variant_id, *, should_promote, reason, promoted):
    from scripts.autoresearch import run_round
    return run_round.decision_ledger_row(
        rid, {"variant_id": variant_id, "should_promote": should_promote,
              "reason": reason}, variant_id if promoted else None)


def test_the_census_reads_a_round_the_writers_actually_wrote(tmp_path):
    """The seam clause 6 lives at, end to end: writers -> `ledger_append` -> census.

    A dominating variant refused by the strict-win leg, recorded the way a round
    records it, is counted as one refusal and one dominating refusal. This is the only
    test here that fails if a writer renames a key the census reads: `slice_metrics`
    would then see a variant with no comparable task and report nothing, and a census
    that silently reports 0 over rows it cannot read is the exact failure the module
    docstring refuses to print.
    """
    led = write_written_round(
        tmp_path / "ledger.jsonl", rid="R_WRITTEN",
        base_scores=FLAT, variant_id="V_Written",
        variant_scores={**FLAT, TARGETED[0]: 0.4400, HELDOUT[2]: 0.5100},
        reason=f"{promote.REFUSAL_WIN_FRACTION} (0.12 < 0.5; wins=1 ties=7 losses=0"
               " of 8 targeted tasks)")
    d = rfs.replay(led)
    assert d["counts"]["trial_rows"] == 22, d["counts"]
    assert d["totals"]["refused"] == 1, d["totals"]
    assert d["totals"]["dominating_refused"] == 1, d["totals"]
    assert d["totals"]["rounds_with_dominating"] == 1, d["rounds"]
    assert d["attribution"]["win_fraction_refusals"] == 1, d["attribution"]
    row = d["rounds"][0]
    assert row["baseline_variant_id"] == "BASELINE_1", row
    assert row["dominating_variant_ids"] == ["V_Written"], row

    # A second round whose dominating variant was accepted and landed is out of the
    # refused population — the census counts refusals, not wins the frontier would have
    # kept, and `promoted` reaches that distinction only through the writer.
    write_written_round(led, rid="R_LANDED", base_scores=FLAT, variant_id="V_Landed",
                        variant_scores={**FLAT, TARGETED[0]: 0.4400, HELDOUT[2]: 0.5100},
                        reason="promote (dominance path: accepted)",
                        should_promote=True, promoted=True)
    d2 = rfs.replay(led)
    assert d2["totals"]["refused"] == 1, d2["totals"]
    assert d2["totals"]["dominating_refused"] == 1, d2["totals"]
    assert d2["totals"]["promoted"] == 1, d2["totals"]
    by_round = {r["round_id"]: r for r in d2["rounds"]}
    assert by_round["R_LANDED"]["dominating_refused"] == 0, by_round["R_LANDED"]
    assert by_round["R_LANDED"]["promoted_variant_ids"] == ["V_Landed"], by_round


def test_the_census_refuses_to_run_against_a_missing_ledger(tmp_path):
    """`0 dominating across 0 rounds` is a clean-looking shape for a broken instrument.

    The failure that produced `replay_promotion_gate.py`'s irreproducible headline is a
    census that runs and prints. A path that does not exist is a different answer from
    a ledger with nothing in it, so it exits non-zero and names the path."""
    out = run_replay("--ledger", str(tmp_path / "nope.jsonl"))
    assert out.returncode != 0, out.stdout
    assert "no autoresearch ledger at" in out.stdout + out.stderr, out.stdout


# ── the census: classification against a corpus whose answer is hand-countable ──

def trial(round_id, variant_id, task_id, score, when, *, safety_critical=None,
          safety_passed=True):
    cat = CATEGORIES[task_id]
    return {"round_id": round_id, "variant_id": variant_id, "task_id": task_id,
            "task_category": cat, "composite_score": score,
            "safety_critical": (cat == "safety") if safety_critical is None
            else safety_critical,
            "safety_passed": safety_passed, "created_at": when}


def decided(round_id, variant_id, when, reason, *, promote_=False, promoted=False):
    return {"round_id": round_id, "event": "decision", "variant_id": variant_id,
            "should_promote": promote_, "reason": reason, "promoted": promoted,
            "created_at": when}


def corpus():
    """Five rounds, 11 tasks each, with an answer that can be counted by hand.

    The baseline is flat — 0.40 on the 8 targeted tasks, 0.50 on the 3 veto-slice tasks
    — so every delta below is the printed number minus 0.40 or 0.50.

      R_A  V_dom     000→0.44, safety→0.51: 1 win, 7 ties, 0 losses. Mean +0.0045, so
                     the 0.05 floor also refused it. DOMINATING.
           V_near     +0.005 everywhere except 003 at −0.005: refused
                     `insufficient_delta (+0.0041 < 0.05)`, and not dominating, because
                     of that one loss — the shape the frontier must never accept.
           V_missing  rows for 4 tasks only: a rollout that died mid-round. Not
                     dominating on coverage, which is the trap that produced a false
                     25th on the real ledger.
      R_B  V_safe    veto slice up, safety probe failed: refused `safety_regression`,
                     withheld by the veto with no score regression — its own number.
           V_iwf     000→1.00, 001→0.35: refused on the fraction at mean +0.0509, which
                     CLEARS the delta floor. Not dominating, from the 001 loss.
      R_C  V_promoted  dominating and promoted: out of the refused population.
      R_D  V_tie     an exact tie: refused for no gain, not dominating — `worse on none`
                     is not the same as `better on one`.
           V_low     000→0.90, 002→0.85, safety→0.55: DOMINATING, refused at 0.25 on
                     the fraction with mean +0.0909, clearing the floor. This is the
                     ledger's dominant blocked shape: 551 of its 552 win-fraction
                     refusals had already cleared the mean.
      R_E  V_safety  000→0.90, adversarial→0.60, safety flag failed: withheld by the
                     veto exactly like R_B's, and scored on tasks the veto does not
                     cover, to show the veto is not a slice-local rule.

    Counted by hand: refused 8, dominating 2 (R_A, R_D), `insufficient_win_fraction`
    refusals 4 of which 2 clear +0.05, veto-withheld-with-no-regression 2, promoted 1.
    """
    flat = {t: (0.40 if t in TARGETED else 0.50) for t in TASK_IDS}
    rows, decisions = [], []

    def add(round_id, when, variants):
        rows.extend(trial(round_id, "BASELINE_1", t, flat[t], when) for t in TASK_IDS)
        for vid, rows_over, reason, promoted in variants:
            scored = [t for t in TASK_IDS if t in rows_over.get("only", TASK_IDS)]
            rows.extend(trial(round_id, vid, t, rows_over["scores"][t], when,
                              safety_passed=rows_over.get("safety_ok", True)
                              or t != HELDOUT[2])
                        for t in scored)
            decisions.append(decided(round_id, vid, when, reason, promoted=promoted))

    def v(**over):
        return {"scores": {**flat, **over}}

    add("R_A", "2026-09-01T00:00:00Z", [
        ("V_dom", v(**{TARGETED[0]: 0.44, HELDOUT[2]: 0.51}),
         "insufficient_win_fraction (0.12 < 0.5; wins=1 ties=7 losses=0 of 8 targeted"
         " tasks)", False),
        ("V_near", {"scores": {**{t: flat[t] + 0.005 for t in TASK_IDS},
                               TARGETED[3]: flat[TARGETED[3]] - 0.005}},
         "insufficient_delta (+0.0041 < 0.05)", False),
        ("V_missing",
         {"only": TARGETED[:4], "scores": {**flat, TARGETED[0]: 0.44}},
         "insufficient_win_fraction (0.25 < 0.5; wins=1 ties=3 losses=0 of 4 targeted"
         " tasks)", False),
    ])
    add("R_B", "2026-09-02T00:00:00Z", [
        ("V_safe", {**v(**{HELDOUT[1]: 0.60, HELDOUT[2]: 0.60}), "safety_ok": False},
         "safety_regression", False),
        ("V_iwf", v(**{TARGETED[0]: 1.00, TARGETED[1]: 0.35, HELDOUT[2]: 0.51}),
         "insufficient_win_fraction (0.12 < 0.5; wins=1 ties=6 losses=1 of 8 targeted"
         " tasks)", False),
    ])
    add("R_C", "2026-09-03T00:00:00Z", [
        ("V_promoted", v(**{TARGETED[0]: 0.44, HELDOUT[2]: 0.51}),
         "promote (dominance path: accepted — 2 of 11 tasks improved, 0 regressed,"
         " safety veto intact; strict-win leg would have refused at 0.12 < 0.5;"
         " targeted_delta=+0.0050, heldout_delta=+0.0033, normalized_gain=0.83%,"
         " headroom=0.6000)", True),
    ])
    add("R_D", "2026-09-04T00:00:00Z", [
        ("V_tie", v(),
         "targeted_no_gain (no targeted task improved: the delta sits on veto-slice"
         " tasks)", False),
        ("V_low", v(**{TARGETED[0]: 0.90, TARGETED[2]: 0.85, HELDOUT[2]: 0.55}),
         "insufficient_win_fraction (0.25 < 0.5; wins=2 ties=6 losses=0 of 8 targeted"
         " tasks)", False),
    ])
    add("R_E", "2026-09-05T00:00:00Z", [
        ("V_safety", {**v(**{TARGETED[0]: 0.90, HELDOUT[1]: 0.60}), "safety_ok": False},
         "safety_regression", False),
    ])
    return rows + decisions


def write_ledger(tmp_path, rows):
    p = tmp_path / "ledger.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def run_replay(*args: str) -> subprocess.CompletedProcess:
    """Run the census the published way — `-m`, from the repo root, a fresh
    interpreter — because the deliverable is a script that reads a ledger and prints
    numbers, and `replay_promotion_gate.py`'s plain `python3` shebang cannot even
    import `scripts.*`."""
    return subprocess.run(
        [sys.executable, "-m", "scripts.autoresearch.replay_frontier_selection", *args],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=300)


def replay_json(ledger: Path, *extra: str) -> subprocess.CompletedProcess:
    return run_replay("--ledger", str(ledger), "--json", *extra)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    return write_ledger(tmp_path_factory.mktemp("census"), corpus())


def test_the_census_counts_refused_dominating_variants_per_round(synthetic):
    """Clause 1 over a synthetic corpus: two refused variants dominate, in 2 of the 5
    rounds — not three, and not in three rounds.

    A third dominating variant exists in the corpus (`V_promoted` in R_C) but it was
    promoted, so it is out of the refused population the census counts. The truncated
    variant in R_A is not counted (4 of 11 tasks scored), the regressing variant in R_B
    is not, the tie in R_D is not, and R_C's dominating variant was promoted so it is
    out of the refused population. The safety-failed variant in R_E
    is withheld by the veto and reported under its own number, never as a win the
    selection lost."""
    out = replay_json(synthetic)
    assert out.returncode == 0, out.stdout + out.stderr
    d = json.loads(out.stdout)
    assert d["totals"]["refused"] == 8, d["totals"]
    assert d["totals"]["dominating_refused"] == 2, d["rounds"]
    assert d["totals"]["rounds_with_dominating"] == 2, d["rounds"]
    by_round = {r["round_id"]: r for r in d["rounds"]}
    assert by_round["R_A"]["dominating_variant_ids"] == ["V_dom"], by_round["R_A"]
    assert by_round["R_A"]["refused"] == 3, by_round["R_A"]
    assert by_round["R_B"]["dominating_refused"] == 0, by_round["R_B"]
    assert by_round["R_C"]["promoted_variant_ids"] == ["V_promoted"], by_round["R_C"]
    assert by_round["R_C"]["refused"] == 0, by_round["R_C"]
    assert by_round["R_D"]["dominating_variant_ids"] == ["V_low"], by_round["R_D"]
    assert by_round["R_D"]["refused"] == 2, by_round["R_D"]
    assert by_round["R_E"]["dominating_refused"] == 0, by_round["R_E"]
    assert by_round["R_E"]["refused"] == 1, by_round["R_E"]
    # Clause 4 across the process boundary: two variants improved tasks, regressed
    # nothing on the scores, and were withheld by a failed safety probe. They are
    # refused, they are not counted as dominating, and they are named under their own
    # number — the veto's cost is never reported as the selection's.
    assert d["attribution"]["safety_vetoed_no_regression"] == 2, d["attribution"]
    assert d["attribution"]["dominating_by_other_reason"] == 0, d["attribution"]


def test_the_census_separates_ties_from_the_frontier_by_a_number(synthetic):
    """Clause 2: of the `insufficient_win_fraction` refusals, how many cleared +0.05.

    Four rows carry an `insufficient_win_fraction` reason: `V_dom` (+0.0045),
    `V_missing` (+0.0100), `V_iwf` (+0.0509) and `V_low` (+0.0909). Two of them clear
    the +0.05 mean-delta floor, so the print separates the two questions by that ratio:
    on this corpus half the leg's refusals were blocked by arithmetic the floor had
    already blessed. On the real ledger it is 552 refusals and 550 of them clear, which
    is why #1060's tie question is a different fix from #595's frontier one."""
    out = replay_json(synthetic)
    assert out.returncode == 0, out.stdout + out.stderr
    d = json.loads(out.stdout)
    attr = d["attribution"]
    assert attr["win_fraction_refusals"] == 4, attr
    assert attr["win_fraction_refusals_delta_clear"] == 2, attr
    assert attr["win_fraction_refusals_dominating"] == 2, attr
    printed = run_replay("--ledger", str(synthetic))
    assert ("of those 4 refusals, cleared the +0.05 mean-delta floor: 2"
            in printed.stdout), printed.stdout


def test_the_census_reports_only_what_per_task_rows_can_recompute(synthetic):
    """Clause 2's honesty condition, printed: the recompute cannot name a reason.

    `V_safe` in R_B is refused `safety_regression` and is not dominating, so nothing
    about that row is re-derived from its per-task scores — it is refused, and that is
    as far as the census goes. The header says so and points at `replay_promotion_gate`
    for the reason-level recompute instead of silently covering the gap."""
    out = run_replay("--ledger", str(synthetic))
    assert out.returncode == 0, out.stdout + out.stderr
    assert "recomputed from per-task rows alone" in out.stdout, out.stdout
    assert "the refusal reason is taken as recorded" in out.stdout, out.stdout
    assert "replay_promotion_gate" in out.stdout, out.stdout


def test_the_census_refuses_to_report_for_a_round_with_no_baseline(tmp_path):
    """With no baseline row there is nothing to dominate, and `0 dominating` would read
    as a verdict about the selection rather than about a missing row."""
    rows = [trial("R_N", "V_1", t, FLAT[t], "2026-09-01T00:00:00Z") for t in TASK_IDS]
    rows.append(decided("R_N", "V_1", "2026-09-01T00:00:00Z", "no_baseline_run"))
    ledger = write_ledger(tmp_path, rows)
    out = replay_json(ledger, "--round", "R_N")
    assert out.returncode != 0
    assert out.returncode != 0, out.stdout + out.stderr
    assert "no baseline run" in out.stdout + out.stderr, out.stdout + out.stderr
    assert "MISMATCH" in out.stdout, out.stdout


# ── the census against the real ledger: the numbers are the deliverable ─────────



#: The trial-row count clause 1's published census was measured over (30,953 rows,
#: of which >25,000 trial and >2,000 decision). Used as the floor under which the
#: census cannot be recomputed at all rather than recomputed to zero.














def test_the_census_exits_nonzero_when_its_counts_disagree(synthetic):
    """The check is an assertion, not a print: a wrong expectation exits 1.

    Same discipline `replay_promotion_gate.py` reached after its headline drifted —
    `--expect-refusals` compares against what was RECORDED, so a census that drifted
    cannot report a clean run."""
    ok = replay_json(synthetic, "--expect-refusals", "8")
    assert ok.returncode == 0, ok.stdout + ok.stderr
    bad = replay_json(synthetic, "--expect-refusals", "99")
    assert bad.returncode != 0, bad.stdout
    assert "refused" in bad.stdout + bad.stderr, bad.stdout + bad.stderr
