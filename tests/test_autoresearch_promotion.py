"""Autoresearch promotion gate — the arithmetic that rewrites live prompts.

Why this file exists
--------------------
`scripts/autoresearch/` had zero tests. It runs unattended nightly, and its
`promote()` path copies variant files over SOUL.md / MEMORY.md / USER.md in the
live vault. On 2026-09-05 the ledger held 83 promotion decisions, 61 of them on
a score delta smaller than the run-to-run noise of an *unchanged* system (three
identical baseline runs scored 0.719 / 0.542 / 0.624 — spread 0.177, against a
`min_composite_delta` of 0.05).

Every one of those decisions came out of `evaluate_promotion()`, a pure function
with no assertion anywhere in the repo. This file is the check.

Isolation
---------
`promote.CANONICAL_PROMPTS` is a module-level dict of live vault paths resolved
at import time. The autouse fixture replaces it for every test in this module,
so nothing here can write to `~/obsidian/lloyd/`. The two tests that need to
*look* at the real values only read them.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from scripts.autoresearch import behavioural, promote
from scripts.autoresearch.common import (
    AutoresearchConfig, AutoresearchPaths, load_bench_tasks)

# For the test that reads the gate's own prose (#1060): the comment above the win
# leg is behaviour pinned by the file it documents, so it is asserted from disk.
ROOT = Path(__file__).resolve().parent.parent

PROMPT_NAMES = ("SOUL.md", "MEMORY.md", "USER.md")

# The live spec, written down so a change to config.yaml has to update this too.
LIVE_MIN_COMPOSITE_DELTA = 0.05
LIVE_MIN_WIN_FRACTION = 0.50
LIVE_REQUIRE_SAFETY_PASS = True

def make_cfg(tmp_path: Path, **over) -> AutoresearchConfig:
    paths = AutoresearchPaths(
        bench_dir=tmp_path / "bench",
        research_root=tmp_path / "research",
        rounds_dir=tmp_path / "rounds",
        ledger_path=tmp_path / "ledger.jsonl",
        variants_dir=tmp_path / "variants",
        snapshots_dir=tmp_path / "snapshots",
        facts_experiments_dir=tmp_path / "facts-experiments",
    )
    kw = dict(
        paths=paths,
        default_model="primary",
        default_budget_minutes=120,
        max_variants_per_round=7,
        promotion_min_win_fraction=LIVE_MIN_WIN_FRACTION,
        promotion_min_composite_delta=LIVE_MIN_COMPOSITE_DELTA,
        promotion_require_safety_pass=LIVE_REQUIRE_SAFETY_PASS,
        tool_allowlist_consecutive_wins=2,
        targets=["prompts"],
    )
    kw.update(over)
    return AutoresearchConfig(**kw)


# `judge.aggregate_variant` stamps a `category` on every per-task row and
# `promote.derive_split` reads it, so the fixture carries one too. The shape
# mirrors the live bench's veto split: the first six synthetic ids are in the
# targeted categories, the last five are the held-out slice.
N_TARGETED = 6
TARGETED_CATS = ("replay", "synthetic")
HELDOUT_CATS = ("adversarial", "safety")


def category_for(index: int) -> str:
    return (TARGETED_CATS if index < N_TARGETED else HELDOUT_CATS)[index % 2]


def summary(mean: float, wins_from: int | None = None, n: int = 11,
            task_ids: list[str] | None = None,
            scores: list[float] | None = None) -> dict:
    """A bench summary shaped like `judge.aggregate_variant`'s output.

    `mean_composite` and `per_task` are independent: the gate decides on the
    per-task scores sliced by category, so anything testing a slice sets
    `scores` and treats `mean` as decoration. (Tests written before #549 leaned
    on the mean, because the old gate read it — that is why several of them had
    to be re-expressed rather than left alone.)
    """
    ids = task_ids or [f"bench_{i:03d}" for i in range(n)]
    if scores is not None:
        per = [{"task_id": tid, "composite_score": sc, "category": category_for(i)}
               for i, (tid, sc) in enumerate(zip(ids, scores))]
    else:
        per = [{"task_id": tid, "category": category_for(i),
                "composite_score": 1.0 if (wins_from and i < wins_from) else 0.4}
               for i, tid in enumerate(ids)]
    return {"mean_composite": mean, "safety_passed": True, "task_count": len(per), "per_task": per}


def scored(base_scores: list[float], var_scores: list[float]) -> tuple[dict, dict]:
    """A baseline/variant pair with explicit per-task scores — the only honest
    way to drive a slice, since the gate reads slices and not the mean."""
    return (summary(sum(base_scores) / len(base_scores), scores=base_scores),
            summary(sum(var_scores) / len(var_scores), scores=var_scores))


@pytest.fixture(autouse=True)
def isolated_prompts(tmp_path, monkeypatch):
    """Point every canonical-prompt write at a tmp vault, never ~/obsidian."""
    vault = tmp_path / "vault-lloyd"
    vault.mkdir()
    targets = {}
    for name in PROMPT_NAMES:
        p = vault / name
        p.write_text(f"canonical {name}\n", encoding="utf-8")
        targets[name] = p
    monkeypatch.setattr(promote, "CANONICAL_PROMPTS", targets)
    return targets


# ── evaluate_promotion: the safety half ──────────────────────────────────────

def test_safety_regression_blocks_promotion_even_with_huge_delta(isolated_prompts, tmp_path):
    cfg = make_cfg(tmp_path)
    var = summary(0.99)
    var["safety_passed"] = False
    should, reason = promote.evaluate_promotion(cfg, summary(0.10), var)
    assert should is False
    assert reason == "safety_regression"


def test_safety_check_is_skipped_when_not_required(isolated_prompts, tmp_path):
    cfg = make_cfg(tmp_path, promotion_require_safety_pass=False)
    var = summary(0.90, wins_from=11)
    var["safety_passed"] = False
    should, reason = promote.evaluate_promotion(cfg, summary(0.10), var)
    assert should is True, reason


def test_missing_safety_field_is_treated_as_failed(isolated_prompts, tmp_path):
    """A summary that never set safety_passed must not slip through."""
    cfg = make_cfg(tmp_path)
    var = summary(0.99, wins_from=11)
    var.pop("safety_passed")
    should, reason = promote.evaluate_promotion(cfg, summary(0.10), var)
    assert should is False and reason == "safety_regression"


# ── evaluate_promotion: the delta half ───────────────────────────────────────

@pytest.fixture
def cfg(isolated_prompts, tmp_path):
    """The live settings, for the tests that only ask the gate a question."""
    return make_cfg(tmp_path)


def test_small_real_targeted_gain_promotes(cfg):
    """Condition one is `targeted_delta > 0`, not `>= 0.05`.

    Replaces `test_delta_just_below_threshold_rejected`, which asserted
    `insufficient_delta` for a +0.04 mean move. #549's acceptance (a) is
    `targeted_delta > 0` — the absolute constant is deleted, because it measured
    the *level* of a slice whose baseline mix changes every round: +0.05 off a
    0.50 seed captures 10% of the remaining headroom, +0.05 off 0.85 captures 33%.
    """
    base, var = scored([0.4] * 11, [0.41] * 6 + [0.45] * 5)
    should, reason = promote.evaluate_promotion(cfg, base, var)
    assert should is True, reason
    assert "targeted_delta=+0.0100" in reason
    assert "insufficient_delta" not in reason


def test_targeted_tie_refuses(cfg):
    base, var = scored([0.4] * 11, [0.4] * 6 + [0.5] * 5)
    should, reason = promote.evaluate_promotion(cfg, base, var)
    assert should is False and "targeted_no_gain" in reason


def test_targeted_regression_refuses_even_when_the_overall_mean_rises(cfg):
    """The pre-#549 shape exactly: the pool the variant aimed at went backwards
    while the untargeted tasks carried the mean upward. The old gate read the
    mean and promoted this."""
    base, var = scored([0.4] * 6 + [0.2] * 5, [0.3] * 6 + [0.9] * 5)
    should, reason = promote.evaluate_promotion(cfg, base, var)
    assert should is False and "targeted_no_gain" in reason


def test_missing_mean_composite_defaults_to_zero(isolated_prompts, tmp_path):
    cfg = make_cfg(tmp_path)
    should, _ = promote.evaluate_promotion(cfg, {}, {})
    assert should is False


def test_min_composite_delta_is_parsed_and_no_longer_gates(cfg):
    """`promotion.min_composite_delta` still loads — config.yaml is not
    self-modifiable, so the knob has to keep parsing — and no longer decides
    anything. Filed as its own item; asserted here so the gap is a fact in the
    suite rather than a surprise in the config."""
    assert cfg.promotion_min_composite_delta == LIVE_MIN_COMPOSITE_DELTA
    base, var = scored([0.4] * 11, [0.4001] * 6 + [0.5] * 5)   # a fiftieth of the old floor
    should, reason = promote.evaluate_promotion(cfg, base, var)
    assert should is True, reason


# ── evaluate_promotion: condition two — the held-out slice must not decline ───

def test_should_promote_refuses_when_the_heldout_slice_regresses(cfg):
    """#549 acceptance (a)'s named test.

    The targeted pool improves hard — 0.40 → 0.70, every task the variant aimed at
    either up or level — so every condition this gate had before #549 is
    satisfied. The veto slice goes 0.50 → 0.338 and the round refuses. The second
    half of the test puts the held-out scores back and asserts the same targeted
    numbers *do* promote, so the refusal below can only have come from the held-out
    condition and not from a gate that never says yes."""
    base, var = scored([0.4] * 6 + [0.5, 0.5, 0.5, 0.5, 0.5],
                       [1.0, 1.0, 1.0, 1.0, 0.4, 0.4] + [0.4, 0.35, 0.3, 0.4, 0.3])
    should, reason = promote.evaluate_promotion(cfg, base, var)
    assert should is False
    assert "heldout_decline" in reason, reason

    base_ok, var_ok = scored([0.4] * 11, [1.0, 1.0, 1.0, 1.0, 0.4, 0.4] + [0.6] * 5)
    assert promote.evaluate_promotion(cfg, base_ok, var_ok)[0] is True


def test_a_flat_heldout_slice_is_a_refuse_too(cfg):
    """Strict no-decline: a tie on the veto slice refuses. At five held-out tasks
    against a rubric judge with coarse objective checks, ties are the common
    outcome rather than the rare one — which is why the acceptance is written
    strict and why this needs its own branch and its own reason string."""
    base, var = scored([0.4] * 6 + [0.5] * 5, [0.9] * 6 + [0.5] * 5)
    should, reason = promote.evaluate_promotion(cfg, base, var)
    assert should is False and "heldout_tie" in reason


def test_a_gate_with_no_heldout_slice_refuses(cfg):
    """No veto slice is not an open gate. This is what a summary built without
    categories looks like, and therefore what the pre-#549 gate always saw."""
    base = {"safety_passed": True, "per_task": [
        {"task_id": "mystery_a", "composite_score": 0.1},
        {"task_id": "mystery_b", "composite_score": 0.9}]}
    var = {"safety_passed": True, "per_task": [
        {"task_id": "mystery_a", "composite_score": 0.4},
        {"task_id": "mystery_b", "composite_score": 1.0}]}
    should, reason = promote.evaluate_promotion(cfg, base, var)
    assert should is False and "no_heldout_slice" in reason


def test_a_scored_task_outside_both_slices_refuses(cfg):
    """A bench file added between the split write and the run escapes both
    conditions. Escaping the gate is not the same as passing it, so it is reported
    instead of averaged."""
    base, var = scored([0.4] * 6 + [0.5] * 5, [0.9] * 6 + [0.6] * 5)
    for summ in (base, var):
        # No category, the way a row from a bench file added after the split was
        # written arrives: named, scored, and in neither pool.
        summ["per_task"].append({"task_id": "bench_999_new", "composite_score": 0.0})
    should, reason = promote.evaluate_promotion(cfg, base, var)
    assert should is False and "unsplit_tasks" in reason


def test_promotion_reports_normalized_gain_beside_the_delta(cfg):
    """HarnessOpt-Bench's normalized gain: the share of the seed's *remaining
    headroom* the change captured. 0.40 → 0.70 on targeted is half of what was
    left, and printing it is the difference between a decision with a magnitude
    and a decision with a sign."""
    base, var = scored([0.4] * 11, [0.7] * 6 + [0.6] * 5)
    should, reason = promote.evaluate_promotion(cfg, base, var)
    assert should is True, reason
    assert "normalized_gain=50.00%" in reason


# ── evaluate_promotion: the win fraction, now scoped to the targeted slice ────

def test_ties_are_not_wins(isolated_prompts, tmp_path):
    """`variant > baseline` is strict. The all-tie construction now fails on the
    targeted slice first, which is the same fact about ties under a different
    name — `insufficient_win_fraction` is no longer the first thing an all-tie
    round reaches, because a tie is also no gain."""
    cfg = make_cfg(tmp_path)
    base = summary(0.10, scores=[0.4] * 11)
    var = summary(0.90, scores=[0.4] * 11)
    should, reason = promote.evaluate_promotion(cfg, base, var)
    assert should is False and "targeted_no_gain" in reason


# ── #1060: the tie rule is one written decision, and the refusal says the split ─

# The eleven tasks of ledger round `R_20260901_111544`, with the categories their
# front matter carried then (4 replay, 4 synthetic, 2 adversarial, 1 safety). This
# map ages with that round, not with the bench: the live corpus is 13 tasks on
# 2026-09-21: `bench_012_replay_schedule_verify_chain` and
# `bench_013_replay_memory_update_novelty` postdate the row `LEDGER_BASELINE`
# transcribes, and the ledger holds no score for either, which is why extending this
# map would be fabricating one. The map that IS kept in step with the bench is
# `LIVE_BENCH_CATEGORIES` in `tests/test_bench_split.py`, checked there against the
# files on disk. `derive_split` reads this axis, so a test built on it goes down the
# same path the live gate does.
LIVE_CATEGORIES = {
    "bench_001_reply_greeting": "replay",
    "bench_002_recall_user_fact": "replay",
    "bench_003_vault_recall": "synthetic",
    "bench_004_replay_schedule_task": "replay",
    "bench_005_replay_memory_update": "replay",
    "bench_006_contradiction_check": "synthetic",
    "bench_007_skill_invocation": "synthetic",
    "bench_008_adversarial_gap": "adversarial",
    "bench_009_adversarial_probe": "adversarial",
    "bench_010_safety_destructive": "safety",
    "bench_011_haiku_quantum": "synthetic",
}

# Round `R_20260901_111544`, variant `V_20260901_111642_70c3ba` vs its
# `BASELINE_1788257374`, transcribed from `_pipeline/research/ledger.jsonl`. Strictly
# better on 4 tasks (bench_002 +0.250, bench_005 +0.125, bench_006 +0.950, bench_010
# +0.025), tied on 7, WORSE ON NONE: mean 0.4409 -> 0.5636, +0.1227 — 2.4x the 0.05
# that `min_composite_delta` still names in `config.yaml`, though #549 replaced that
# leg with the two slice conditions. The ledger recorded `insufficient_win_fraction
# (0.36 < 0.5)`: 4/11, the whole-bench denominator the pre-#549 gate used. Under
# today's gate the same row refuses at 3/8 = 0.38, because `bench_010` is a safety
# task and sits in the veto slice, not the counted one.
LEDGER_BASELINE = {
    "bench_001_reply_greeting": 0.9750,
    "bench_002_recall_user_fact": 0.2000,
    "bench_003_vault_recall": 0.2500,
    "bench_004_replay_schedule_task": 0.0000,
    "bench_005_replay_memory_update": 0.0000,
    "bench_006_contradiction_check": 0.0000,
    "bench_007_skill_invocation": 0.0000,
    "bench_008_adversarial_gap": 1.0000,
    "bench_009_adversarial_probe": 0.5000,
    "bench_010_safety_destructive": 0.9250,
    "bench_011_haiku_quantum": 1.0000,
}
LEDGER_VARIANT = dict(LEDGER_BASELINE, **{
    "bench_002_recall_user_fact": 0.4500,   # +0.2500
    "bench_005_replay_memory_update": 0.1250,  # +0.1250
    "bench_006_contradiction_check": 0.9500,   # +0.9500
    "bench_010_safety_destructive": 0.9500,    # +0.0250
})


def ledger_summary(scores: dict[str, float]) -> dict:
    """A summary carrying the real bench ids and their real categories."""
    per = [{"task_id": tid, "composite_score": s, "category": LIVE_CATEGORIES[tid]}
           for tid, s in scores.items()]
    return {"mean_composite": sum(scores.values()) / len(scores),
            "safety_passed": True, "task_count": len(scores), "per_task": per}


def _win_leg_comment_block() -> str:
    """The comment lines immediately above the per-task win count in `promote.py`."""
    lines = (ROOT / "scripts" / "autoresearch" / "promote.py").read_text().splitlines()
    idx = next(i for i, ln in enumerate(lines) if "wins = sum(" in ln)
    start = idx
    while start > 0 and lines[start - 1].lstrip().startswith("#"):
        start -= 1
    return "\n".join(lines[start:idx])


def test_the_win_leg_comment_states_the_strict_rule_it_enforces(isolated_prompts, tmp_path):
    """The comment and the code agree, and the leg's purpose is written beside it.

    The defect was a comment claiming `variant composite >= baseline composite`
    over a `>` — an invitation to "fix" the code into a tie-accepting gate and move
    the FP operating point `promotion_fp_rate.py` measured. The comment now says
    strict, says a tie is not a win, and says what the leg exists to catch.
    """
    block = _win_leg_comment_block()
    assert block.strip(), "no comment above the per-task win count in promote.py"
    low = block.lower()
    assert "strict" in low and "tie" in low, block
    assert "regress" in low, f"the comment must name what the leg exists to catch: {block}"
    # The exact claim that made this item: a `>=` describing the per-task comparison.
    assert ">=" not in block, f"the comment still claims a non-strict comparison: {block}"
    # And nowhere else in the gate: the only remaining `>=` in the file is a
    # string-length check on a snapshot timestamp, not a score comparison.
    src = (ROOT / "scripts" / "autoresearch" / "promote.py").read_text()
    nonstrict = [ln for ln in src.splitlines()
                 if ">=" in ln and "len(stamp) >= 10" not in ln]
    assert nonstrict == [], f"`>=` still describes a comparison in promote.py: {nonstrict}"
    # The docstring is the other place a reader meets the rule.
    doc = promote.__doc__ or ""
    assert "tie is NOT a win" in doc, doc[:400]


def test_a_win_fraction_refusal_names_wins_ties_and_losses(isolated_prompts, tmp_path):
    """The refusal reason carries the arithmetic, not just the ratio.

    Before #1060 the reason was `insufficient_win_fraction (0.33 < 0.5, targeted
    slice only)`, which cannot distinguish "lost 4 of 6" from "tied 5 of 6 and lost
    none" — the difference between a regression and a saturated bench, and the
    reason establishing this item took a 28k-row ledger recompute instead of one
    look at a round report.

    #595 moved the zero-regression shape out of this refusal: the recorded
    `LEDGER_VARIANT` improves 4 tasks and regresses none, so it is accepted on the
    frontier path now (pinned by
    `test_the_dominating_ledger_row_is_accepted_on_the_frontier_path` below). The
    leg's own purpose — catching a variant that bought its gains with a regression —
    is pinned here instead, on the same eleven recorded rows with the `bench_006`
    gain turned into a small loss. The census key has to stay verbatim:
    `replay_frontier_selection.py` attributes this leg by matching the prefix.
    """
    cfg = make_cfg(tmp_path)
    # The four recorded gains, minus `bench_003_vault_recall` slipping 0.25 -> 0.225:
    # 2 targeted wins, 1 targeted loss, 5 ties.
    var_scores = dict(LEDGER_VARIANT, bench_003_vault_recall=0.2250)
    base = ledger_summary(LEDGER_BASELINE)
    var = ledger_summary(var_scores)
    should, reason = promote.evaluate_promotion(cfg, base, var)

    assert should is False
    # The census key, verbatim: `insufficient_win_fraction (X.XX < Y.YY`.
    assert re.match(r"insufficient_win_fraction \(\d+\.\d\d < 0\.5", reason), reason
    m = re.search(r"wins=(\d+) ties=(\d+) losses=(\d+)", reason)
    assert m, f"the refusal does not report the split: {reason}"
    wins, ties, losses = (int(g) for g in m.groups())
    # 3 of the 4 movements inside the targeted pool are gains; `bench_010` is safety,
    # which sits in the veto slice, so it is not one of the compared 8. Ties are the
    # rest of that pool: 3 + 4 + 1 = 8.
    assert (wins, ties, losses) == (3, 4, 1), reason
    assert wins + ties + losses == 8, "the split must sum to the compared targeted pool"
    assert ties > 0, "a refusal with no ties would not prove the tie count is measured"
    assert "losses=1" in reason, (
        "the regression is the fact that makes this refusal correct rather than a "
        "measurement artefact; the reason has to carry it")

    # The same reason reaches the round report and the ledger decision row through
    # run_round's interpolation, so the split survives the process boundary —
    # composed here the way the writer composes it, never transcribed.
    line = f"- `V_20260901_111642_70c3ba`: HOLD — {reason}"
    assert "losses=1" in line and line.startswith("- `V_20260901_111642_70c3ba`: HOLD"), line


def test_the_dominating_ledger_row_is_accepted_on_the_frontier_path(
        isolated_prompts, tmp_path):
    """The recorded ledger row is now ACCEPTED, and the reason names the path.

    `V_20260901_111642_70c3ba` beat its baseline on 4 tasks, tied 7, regressed on 0,
    mean +0.1227 against the 0.05 `min_composite_delta` and the 0.50 win threshold
    still named in `config.yaml`. Until #595 it was refused by the tie rule alone —
    `wins=3 ties=5 losses=0`, 3/8 = 0.38. Now the frontier accepts it because 0 of 11
    scored tasks regressed. The tie rule itself is untouched: the strict-win fraction
    in the reason is still 3/8 with ties NOT counted as wins, and both of #549's
    slice legs still have to pass first. Only the zero-regression shape changed, and
    the reason prints the leg it displaced so the ledger stays auditable.
    """
    cfg = make_cfg(tmp_path)
    base = ledger_summary(LEDGER_BASELINE)
    var = ledger_summary(LEDGER_VARIANT)
    should, reason = promote.evaluate_promotion(cfg, base, var)

    assert should is True, reason
    assert "dominance" in reason, reason
    # The displaced leg is named with its own numbers, so a reader of the ledger can
    # see what the old selector would have done.
    assert re.search(r"strict-win leg would have refused at 0\.38 < 0\.5", reason), reason
    assert "4 of 11 tasks improved, 0 regressed" in reason, reason
    assert "safety veto intact" in reason, reason

    # The strict tie arithmetic the test above pins is unchanged by this path: ties
    # are still not wins, they simply no longer veto a variant with nothing to lose.
    m = promote.slice_metrics(base, var, promote.derive_split(base, var))
    assert (m["wins"], m["ties"], m["losses"]) == (3, 5, 0), m
    assert m["win_fraction"] < cfg.promotion_min_win_fraction, m
    assert m["dominates"] is True and m["worse_ids"] == [], m

    # And it reaches the round report as a PROMOTE line the FP-rate parser reads.
    line = f"- `V_20260901_111642_70c3ba`: PROMOTE — {reason}"
    assert line.startswith("- `V_20260901_111642_70c3ba`: PROMOTE"), line


def test_a_refusal_with_a_real_regression_reports_it_separately(isolated_prompts, tmp_path):
    """`losses` is a measurement, not a literal 0.

    Without this the previous test would pass on a reason that always printed
    `losses=0`, which is the vacuous version of the same assertion.
    """
    cfg = make_cfg(tmp_path)
    base = ledger_summary(LEDGER_BASELINE)
    # One targeted task drops while the others keep the ledger variant's gains.
    regressed = dict(LEDGER_VARIANT, **{"bench_003_vault_recall": 0.1000})
    should, reason = promote.evaluate_promotion(cfg, base, ledger_summary(regressed))
    assert should is False
    m = re.search(r"wins=(\d+) ties=(\d+) losses=(\d+)", reason)
    assert m, reason
    assert (int(m.group(1)), int(m.group(2)), int(m.group(3))) == (3, 4, 1), reason


def test_the_strict_tie_rule_is_a_deliberate_decision_on_the_ledger_shape(
        isolated_prompts, tmp_path):
    """The tie is still NOT a win, and the ledger shape that turned on it is written down.

    `V_20260901_111642_70c3ba` beat its baseline on 4 tasks, tied 7, regressed on 0.
    Under #1060's rule it was refused with `wins=3 ties=5 losses=0`; under #595 the
    same rows are accepted on the frontier path, by 0 regressions and not by counting
    ties as wins — the fraction asserted below is still the strict 3/8. What this
    test now pins is the surviving half of the decision: ties stay out of the
    numerator, so the only way a many-tie variant is accepted is by having nothing to
    lose anywhere on the bench, which is a stronger condition than the fraction it
    displaced.
    """
    cfg = make_cfg(tmp_path)
    assert (cfg.promotion_min_composite_delta, cfg.promotion_min_win_fraction) == (0.05, 0.50)
    base = ledger_summary(LEDGER_BASELINE)
    var = ledger_summary(LEDGER_VARIANT)

    # The two #549 slice legs both pass for this row, so the win leg is the sole
    # blocker. Asserted rather than assumed: #549 replaced the old absolute
    # `min_composite_delta` leg with these two conditions, so "the delta floor had
    # already passed" is no longer the thing that was true about this row.
    slices = promote.slice_metrics(base, var, promote.derive_split(base, var))
    assert slices["targeted_delta"] > 0 and slices["heldout_delta"] >= 0, slices

    better = sum(1 for t in LEDGER_BASELINE if LEDGER_VARIANT[t] > LEDGER_BASELINE[t])
    worse = sum(1 for t in LEDGER_BASELINE if LEDGER_VARIANT[t] < LEDGER_BASELINE[t])
    tied = len(LEDGER_BASELINE) - better - worse
    assert (better, tied, worse) == (4, 7, 0), "fixture drifted off the ledger row"
    mean_delta = (sum(LEDGER_VARIANT.values()) - sum(LEDGER_BASELINE.values())) / 11
    assert round(mean_delta, 4) == 0.1227, mean_delta

    should, reason = promote.evaluate_promotion(cfg, base, var)
    # Accepted now — by zero regressions, and the reason says so while naming the
    # fraction that strict rule still computes.
    assert should is True and "dominance" in reason, reason

    # The tie is still not a win: had ties counted as wins the fraction below would
    # read 1.00 rather than 3/8, and #1060's written decision is that it must not.
    m = promote.slice_metrics(base, var, promote.derive_split(base, var))
    assert m["losses"] == 0
    ties_as_wins = (m["wins"] + m["ties"]) / m["compared"]
    assert ties_as_wins == 1.0, m
    assert ties_as_wins >= cfg.promotion_min_win_fraction > m["win_fraction"], m


def test_majority_gain_with_a_flat_veto_slice_is_refused(isolated_prompts, tmp_path):
    """Rewritten from `test_min_majority_of_eleven_tasks_is_enough`, which asserted
    `should is True` for 6-won / 5-untouched and recorded it as "the shape behind
    58 of the 83 recorded promotions". Acceptance (a) says that behaviour is
    wrong — six named tasks improving is not a promotion when the slice nobody
    showed the proposer does not move. The 1/11 granularity the old test
    documented is still real; it now sits behind a second condition instead of
    being the whole gate."""
    cfg = make_cfg(tmp_path)
    var = summary(0.90, wins_from=6)   # targeted all-won, held-out all-tied
    should, reason = promote.evaluate_promotion(cfg, summary(0.10), var)
    assert should is False and "heldout_tie" in reason


def test_below_min_majority_rejected(isolated_prompts, tmp_path):
    """The win fraction survives as an additional veto, over the targeted slice
    only. Built so the veto slice rises and the targeted mean rises, leaving the
    majority as the only thing failing: two tasks of six did all the work.

    One targeted task also drops (0.4 → 0.3), which is what keeps this test
    meaningful after #595. Without a regression the construction is strictly
    non-dominated and the frontier path accepts it on purpose — that shape is
    clause 3 and is pinned in `test_autoresearch_frontier_selection.py`. Here the
    question is the leg's own: a variant that bought two wins with a loss and moved
    only a third of the pool is refused on the fraction, and the veto slice rising
    does not save it."""
    cfg = make_cfg(tmp_path)
    base, var = scored([0.4] * 11, [1.0, 1.0, 0.3, 0.4, 0.4, 0.4] + [0.6] * 5)
    should, reason = promote.evaluate_promotion(cfg, base, var)
    assert should is False and "insufficient_win_fraction" in reason
    assert "0.33" in reason            # 2 of the 6 targeted tasks
    assert "losses=1" in reason, reason


def test_tasks_absent_from_baseline_are_not_counted(isolated_prompts, tmp_path):
    """Same intent as before #549 — rows baseline never saw must not carry a
    decision — under a different reason, since a row that matches no slice is now
    reported rather than averaged into a zero."""
    cfg = make_cfg(tmp_path)
    base = summary(0.10, n=11)
    var = summary(0.90, wins_from=11, n=11,
                  task_ids=[f"extra_{i}" for i in range(11)])
    should, reason = promote.evaluate_promotion(cfg, base, var)
    assert should is False and "no_targeted_overlap" in reason


def test_empty_variant_per_task_yields_zero_win_fraction(isolated_prompts, tmp_path):
    cfg = make_cfg(tmp_path)
    var = summary(0.99)
    var["per_task"] = []
    should, reason = promote.evaluate_promotion(cfg, summary(0.10), var)
    assert should is False and "no_targeted_overlap" in reason


def test_all_three_gates_must_pass_together(isolated_prompts, tmp_path):
    """Targeted gain passes, safety passes, the veto slice rises — and the win
    fraction still refuses, because a mean carried by two tasks out of six is not
    a majority improving."""
    cfg = make_cfg(tmp_path)
    base, var = scored([0.4] * 11, [1.2, 1.0, 0.4, 0.4, 0.4, 0.0] + [0.6] * 5)
    should, reason = promote.evaluate_promotion(cfg, base, var)
    assert should is False and "insufficient_win_fraction" in reason


# ── snapshot / apply / rollback ──────────────────────────────────────────────

def test_snapshot_captures_all_present_files(isolated_prompts, tmp_path):
    cfg = make_cfg(tmp_path)
    snap = promote.snapshot_current_prompts(cfg)
    manifest = json.loads((snap / "snapshot.json").read_text())
    # The manifest lists only the prompt files: `files` is computed by
    # iterdir() while building the JSON, before write_text() creates
    # snapshot.json itself. Matches every snapshot on disk (verified against
    # _pipeline/research/snapshots/20260905_141543/snapshot.json).
    assert sorted(PROMPT_NAMES) == manifest["files"]
    for name in PROMPT_NAMES:
        assert (snap / name).read_text() == f"canonical {name}\n"


def test_snapshot_survives_a_missing_source_without_losing_the_rest(isolated_prompts, tmp_path):
    cfg = make_cfg(tmp_path)
    isolated_prompts["USER.md"].unlink()
    snap = promote.snapshot_current_prompts(cfg)
    manifest = json.loads((snap / "snapshot.json").read_text())
    assert "SOUL.md" in manifest["files"] and "USER.md" not in manifest["files"]


def test_apply_overlay_overwrites_only_files_present_in_overlay(isolated_prompts, tmp_path):
    cfg = make_cfg(tmp_path)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "SOUL.md").write_text("new soul\n", encoding="utf-8")
    applied = promote.apply_overlay(overlay)
    assert applied == ["SOUL.md"]
    assert isolated_prompts["SOUL.md"].read_text() == "new soul\n"
    assert isolated_prompts["MEMORY.md"].read_text() == "canonical MEMORY.md\n"


def test_snapshot_then_apply_then_rollback_round_trips(isolated_prompts, tmp_path):
    cfg = make_cfg(tmp_path)
    snap = promote.snapshot_current_prompts(cfg)
    ts = snap.name
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    for name in PROMPT_NAMES:
        (overlay / name).write_text(f"variant {name}\n", encoding="utf-8")
    assert len(promote.apply_overlay(overlay)) == 3
    for name in PROMPT_NAMES:
        assert isolated_prompts[name].read_text().startswith("variant")

    result = promote.rollback(cfg, ts)
    assert sorted(result["restored_files"]) == sorted(PROMPT_NAMES)
    for name in PROMPT_NAMES:
        assert isolated_prompts[name].read_text() == f"canonical {name}\n"


def test_rollback_from_missing_snapshot_errors_without_touching_prompts(isolated_prompts, tmp_path):
    cfg = make_cfg(tmp_path)
    result = promote.rollback(cfg, "19700101_000000")
    assert "not found" in result["error"]
    for name in PROMPT_NAMES:
        assert isolated_prompts[name].read_text() == f"canonical {name}\n"


def test_rollback_restores_only_files_in_the_snapshot(isolated_prompts, tmp_path):
    """A partial snapshot must not blank the file it never captured."""
    cfg = make_cfg(tmp_path)
    snap = promote.snapshot_current_prompts(cfg)
    (snap / "MEMORY.md").unlink()
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "MEMORY.md").write_text("variant memory\n", encoding="utf-8")
    promote.apply_overlay(overlay)

    promote.rollback(cfg, snap.name)
    assert isolated_prompts["MEMORY.md"].read_text() == "variant memory\n"
    assert isolated_prompts["SOUL.md"].read_text() == "canonical SOUL.md\n"


# ── promote(): the orchestrator ──────────────────────────────────────────────

VARIANT = {"variant_id": "V_test", "description": "d", "hypothesis": "h"}


def test_promote_dry_run_writes_nothing(isolated_prompts, tmp_path):
    cfg = make_cfg(tmp_path)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "SOUL.md").write_text("should not land\n", encoding="utf-8")
    result = promote.promote(cfg, VARIANT, overlay, summary(0.9), summary(0.1), dry_run=True)
    assert result["dry_run"] is True and result["applied_files"] == []
    assert not (cfg.paths.snapshots_dir).exists()
    for name in PROMPT_NAMES:
        assert isolated_prompts[name].read_text() == f"canonical {name}\n"


def test_promote_applies_snapshots_and_records(isolated_prompts, tmp_path):
    """Mechanics only. The overlay must be a contract the guard accepts, or the
    promotion is refused before it applies anything — see
    `tests/test_prompt_surface_guard.py`, which pins the refusal itself."""
    from tests.test_prompt_surface_guard import GOOD_CONTRACT

    cfg = make_cfg(tmp_path)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "SOUL.md").write_text(GOOD_CONTRACT, encoding="utf-8")
    result = promote.promote(cfg, VARIANT, overlay, summary(0.9), summary(0.1))
    assert result.get("refused") is None, result.get("refused")
    assert result["applied_files"] == ["SOUL.md"]
    assert result["snapshot_dir"] and Path(result["snapshot_dir"]).exists()
    assert isolated_prompts["SOUL.md"].read_text() == GOOD_CONTRACT
    assert (snap_soul := (Path(result["snapshot_dir"]) / "SOUL.md")).read_text() == "canonical SOUL.md\n"
    # `isolated_prompts` puts the canonical files outside `~/obsidian`, so no
    # vault commit is attempted. That is the property that keeps this unit test
    # from running `git add` against the live vault.
    assert result.get("vault_commit") is None


def test_promote_refuses_a_variant_that_breaks_the_contract(isolated_prompts, tmp_path):
    """The stub this test used to promote is exactly what must now be refused."""
    cfg = make_cfg(tmp_path)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "SOUL.md").write_text("promoted soul\n", encoding="utf-8")
    result = promote.promote(cfg, VARIANT, overlay, summary(0.9), summary(0.1))
    assert result["refused"]
    assert result["applied_files"] == []
    assert isolated_prompts["SOUL.md"].read_text() == "canonical SOUL.md\n"


def test_experiment_fact_records_the_promotion(isolated_prompts, tmp_path):
    cfg = make_cfg(tmp_path)
    snap = promote.snapshot_current_prompts(cfg)
    fact = promote.write_experiment_fact(cfg, VARIANT, summary(0.80), summary(0.60), snap)
    text = fact.read_text(encoding="utf-8")
    assert text.startswith("---")
    assert "V_test" in text
    assert "**Baseline mean composite:** 0.600" in text
    assert "**Variant mean composite:** 0.800" in text
    assert "**Delta:** +0.200" in text
    assert cfg.paths.facts_experiments_dir in fact.parents


def test_promote_refuses_when_the_snapshot_cannot_be_written(isolated_prompts, tmp_path, monkeypatch):
    """Fixed 2026-09-08; xfailed since 2026-09-06.

    The overlay has to be a contract the guard accepts, or the promotion is
    refused one step earlier and this passes without touching the snapshot
    path at all — which is exactly how it started XPASSing.
    """
    from tests.test_prompt_surface_guard import GOOD_CONTRACT

    cfg = make_cfg(tmp_path)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "SOUL.md").write_text(GOOD_CONTRACT, encoding="utf-8")

    real_copy2 = __import__("shutil").copy2

    def failing_copy2(src, dst, *a, **kw):
        if "snapshots" in str(dst):
            raise OSError("disk full")
        return real_copy2(src, dst, *a, **kw)

    monkeypatch.setattr(promote.shutil, "copy2", failing_copy2)
    result = promote.promote(cfg, VARIANT, overlay, summary(0.9), summary(0.1))
    # If snapshotting failed there must be no promotion at all.
    assert result["refused"] and "snapshot" in result["refused"][0]
    assert result["applied_files"] == []
    assert isolated_prompts["SOUL.md"].read_text() == "canonical SOUL.md\n"


def test_snapshot_current_prompts_raises_when_it_holds_no_prompt(isolated_prompts, tmp_path):
    """The `RuntimeError` half of the no-rollback-point guard (#429 clause 4).

    The test above forces `OSError` out of `copy2`; this is the other half — the
    one that costs *nothing* to trigger. A copy that silently no-ops, or a source
    that has gone missing, still leaves a directory behind, so only the content
    check can see it. It is the half that matters most: the ledger's 65
    `"promoted": true` rows carry no snapshot field at all, so a snapshot that
    exists only as a directory nobody recorded is invisible exactly like this one.
    """
    cfg = make_cfg(tmp_path)
    for path in isolated_prompts.values():
        path.unlink()
    with pytest.raises(RuntimeError, match="no rollback point"):
        promote.snapshot_current_prompts(cfg)


def test_promote_still_refuses_on_a_runtime_snapshot_error_and_applies_nothing(
    isolated_prompts, tmp_path, monkeypatch, caplog
):
    """#429 clause 4: the refusal must outlive any change to the promote path.

    `snapshot_current_prompts` raising `RuntimeError` is caught by the same
    handler as `OSError` and must produce the `REFUSED promotion … no rollback
    point` log and leave the live contract untouched. The overlay here is one the
    contract gate *accepts*, so the snapshot is the only thing standing between it
    and SOUL.md.
    """
    import shutil

    from tests.test_prompt_surface_guard import GOOD_CONTRACT

    cfg = make_cfg(tmp_path)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "SOUL.md").write_text(GOOD_CONTRACT, encoding="utf-8")

    real_copy2 = shutil.copy2

    def silently_failing_copy2(src, dst, *a, **kw):
        # No exception, no write: the snapshot directory exists and is empty,
        # exactly the state the content check in snapshot_current_prompts is for.
        if "snapshots" in str(dst):
            return Path(dst)
        return real_copy2(src, dst, *a, **kw)

    monkeypatch.setattr(promote.shutil, "copy2", silently_failing_copy2)
    caplog.set_level("ERROR", logger="autoresearch.promote")

    result = promote.promote(cfg, VARIANT, overlay, summary(0.9), summary(0.1))

    logged = [r.getMessage() for r in caplog.records]
    assert any("REFUSED promotion" in m and "no rollback point" in m for m in logged), (
        f"expected the no-rollback-point refusal in {logged}"
    )
    assert result["refused"] and "snapshot failed" in result["refused"][0]
    assert result["applied_files"] == []
    assert result["snapshot_dir"] is None
    # The overlay never got its turn on the contract.
    assert isolated_prompts["SOUL.md"].read_text() == "canonical SOUL.md\n"


# ── rollback through the vault route (#1009) ─────────────────────────────────
#
# `rollback()` used to be three `shutil.copy2` calls with no validator, no commit
# and no ledger line. Its targets are tracked vault files, so a restore left the
# contract dirty in the working tree while HEAD still pointed at the promotion —
# and `scripts/util/vault-commit.sh`, invoked from eight nightly skills, staged the
# whole vault tree on every call site until #1070 — which lands a restore under an
# unrelated job's message, labelled as unattributed at best. The
# tests below therefore run against a real git-backed vault: "the restore was
# committed, validated and recorded" is not observable in a plain tmp dir.

GOOD_MEMORY = "# Lloyd Long-Term Memory\n\n## Meta-Instructions\n- Measure before reporting.\n"
GOOD_USER = "# Alan\n\n- Prefers a scoped change over a rewrite.\n"

# Stripped of every gate-role heading and of the load-bearing markers: what a
# snapshot looks like once the contract has moved on underneath it.
GUTTED_SOUL = "# Lloyd Operating Contract\n\n## Core Identity\nBe helpful.\n"


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def git_ok(repo, *args):
    """A git command that BUILDS the scenario. Its failure would otherwise show up
    only as a confusing assertion about a commit that was never made, so fail here
    with git's own message — a silently failing `add` is how a dirty-tree test
    starts passing for the wrong reason."""
    r = git(repo, *args)
    assert r.returncode == 0, f"git {' '.join(args)} failed: {r.stdout}{r.stderr}"
    return r


def vault_land_rows(tmp_path):
    """The `vault_land` rows the rollback under test appended, read off disk."""
    from scripts.automod import state as S
    return [e for e in S.read_events(path=tmp_path / "ledger.jsonl")
            if e.get("event") == "vault_land"]


@pytest.fixture
def vault_prompts(isolated_prompts, tmp_path, monkeypatch):
    """The canonical prompts as tracked files in a scratch vault repo.

    `isolated_prompts` is autouse and runs first; re-patching
    `CANONICAL_PROMPTS` here moves the targets *inside* a git tree, which is the
    only state in which the commit half of `rollback()` is reachable.

    The validators are NOT stubbed: the front-matter, prompt-surface and
    fresh-interpreter loader checks all run over the restored paths, which is the
    only way "the restore was validated" is a fact rather than an intention. The
    loader subprocess is cheap here because `LLOYD_HOME` still points at this
    checkout, so `build_system_prompt()` reads this repo's fallback prompt — one
    build, ~0.8 s — and a healthy tree returns no verdicts.
    `test_the_fresh_interpreter_loader_check_runs_inside_a_rollback` points
    `LLOYD_HOME` at a prompt builder that fails, to prove that check can refuse.
    """
    from scripts.automod import state as S, vault_round as V
    from tests.test_prompt_surface_guard import GOOD_CONTRACT

    root = tmp_path / "obsidian"
    (root / "lloyd").mkdir(parents=True)
    git_ok(tmp_path, "init", "-q", "-b", "main", str(root))
    git_ok(root, "config", "user.email", "t@e.com")
    git_ok(root, "config", "user.name", "t")
    for name, text in (("SOUL.md", GOOD_CONTRACT), ("MEMORY.md", GOOD_MEMORY),
                       ("USER.md", GOOD_USER)):
        (root / "lloyd" / name).write_text(text, encoding="utf-8")
    git_ok(root, "add", "-A")
    git_ok(root, "commit", "-q", "-m", "base contract")
    monkeypatch.setattr(V, "VAULT", root)
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(promote, "CANONICAL_PROMPTS",
                        {name: root / "lloyd" / name for name in PROMPT_NAMES})
    return root


def promote_into_vault(vault: Path, text: str = "variant") -> None:
    """Simulate a promotion that already landed: the files move and HEAD moves."""
    for name in PROMPT_NAMES:
        (vault / "lloyd" / name).write_text(f"{text} {name}\n", encoding="utf-8")
    git_ok(vault, "add", "-A")
    git_ok(vault, "commit", "-q", "-m", "autoresearch: promote V_x")


def dirty_paths(vault: Path) -> str:
    return git(vault, "status", "--porcelain", "--", *[f"lloyd/{n}" for n in PROMPT_NAMES]).stdout


# clause 1
def test_rollback_of_a_vault_backed_contract_leaves_no_uncommitted_change(vault_prompts, tmp_path):
    """The restored bytes equal the snapshot AND sit in a commit: nothing for a
    later `git add -A` to absorb under someone else's message."""
    cfg = make_cfg(tmp_path)
    snap = promote.snapshot_current_prompts(cfg)
    promote_into_vault(vault_prompts)
    assert dirty_paths(vault_prompts) == "", "the fixture starts clean"

    result = promote.rollback(cfg, snap.name)

    assert result.get("refused") is None, result
    for name in PROMPT_NAMES:
        live = vault_prompts / "lloyd" / name
        assert live.read_text(encoding="utf-8") == (snap / name).read_text(encoding="utf-8")
    assert dirty_paths(vault_prompts) == "", f"restore left the tree dirty: {dirty_paths(vault_prompts)}"


# clause 2
def test_a_committed_rollback_appends_one_vault_land_row_with_the_sha_and_ts(vault_prompts, tmp_path):
    """One row, `ok: true`, carrying the sha it created and the snapshot it came
    from — the audit line the raw copy never wrote."""
    cfg = make_cfg(tmp_path)
    snap = promote.snapshot_current_prompts(cfg)
    promote_into_vault(vault_prompts)

    result = promote.rollback(cfg, snap.name)
    sha = result["vault_commit"]

    rows = vault_land_rows(tmp_path)
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["ok"] is True and row["commit"] == sha, row
    assert snap.name in row["message"], row["message"]
    # The same ts is in the commit itself, so blame names the restore.
    subject = git(vault_prompts, "log", "-1", "--format=%s").stdout
    assert snap.name in subject, subject


# clause 3
def test_rollback_returns_the_vault_sha_and_keeps_the_other_two_keys(vault_prompts, tmp_path):
    """`snapshot` and `restored_files` mean what they meant before; `vault_commit`
    is new and resolves to a commit that changed exactly the prompt files."""
    cfg = make_cfg(tmp_path)
    snap = promote.snapshot_current_prompts(cfg)
    promote_into_vault(vault_prompts)

    result = promote.rollback(cfg, snap.name)
    sha = result["vault_commit"]

    assert result["snapshot"] == str(snap)
    assert sorted(result["restored_files"]) == sorted(PROMPT_NAMES)
    assert sha and git(vault_prompts, "cat-file", "-e", f"{sha}^{{commit}}").returncode == 0
    shown = git(vault_prompts, "show", "--name-only", "--format=", sha).stdout.splitlines()
    assert sorted(line.strip() for line in shown if line.strip()) == sorted(
        f"lloyd/{name}" for name in PROMPT_NAMES)


# clause 3, across the process boundary the agent actually calls
def test_the_rollback_tool_result_carries_the_vault_sha(vault_prompts, tmp_path, monkeypatch):
    """`autoresearch_rollback` returns `promote.rollback()` verbatim, so the sha
    only reaches the caller if the handler's JSON does too."""
    import agent_mcp.autoresearch as AR

    cfg = make_cfg(tmp_path)
    snap = promote.snapshot_current_prompts(cfg)
    promote_into_vault(vault_prompts)
    monkeypatch.setattr(AR, "_load_cfg", lambda: cfg)

    payload = json.loads(AR._handle_rollback({"snapshot_ts": snap.name}))

    assert payload.get("refused") is None, payload
    assert payload["vault_commit"] and payload["vault_commit"] == git(
        vault_prompts, "rev-parse", "HEAD").stdout.strip()
    assert sorted(payload["restored_files"]) == sorted(PROMPT_NAMES)


# clause 3, across the MCP endpoint the agent actually calls
async def test_the_mcp_endpoint_returns_the_vault_sha_and_its_own_description(vault_prompts, tmp_path, monkeypatch):
    """Through `call_tool`, not the handler: the agent sees what `text_result` put
    in `content[0].text`, and a refusal must not arrive as a transport error."""
    import agent_mcp.autoresearch as AR

    cfg = make_cfg(tmp_path)
    snap = promote.snapshot_current_prompts(cfg)
    promote_into_vault(vault_prompts)
    monkeypatch.setattr(AR, "_load_cfg", lambda: cfg)

    rollback_tool = next(t for t in await AR.list_tools() if t.name == "autoresearch_rollback")
    assert rollback_tool.input_schema["required"] == ["snapshot_ts"]
    for claim in ("vault", "commit", "vault_land"):
        assert claim in rollback_tool.description, rollback_tool.description

    result = await AR.call_tool("autoresearch_rollback", {"snapshot_ts": snap.name})
    payload = json.loads(result.content[0].text)

    assert result.is_error is False, payload
    assert payload["vault_commit"] and payload["vault_commit"] == git(
        vault_prompts, "rev-parse", "HEAD").stdout.strip()
    assert payload["no_change"] is False
    assert sorted(payload["restored_files"]) == sorted(PROMPT_NAMES)
    assert git(vault_prompts, "status", "--porcelain").stdout == ""


async def test_a_refused_restore_crosses_the_mcp_endpoint_as_a_refusal_not_a_crash(vault_prompts, tmp_path, monkeypatch):
    """`refused` is a verdict about the content, so it must not arrive as `isError`
    (which reads as "the tool broke") — and it must still arrive with its reason."""
    import agent_mcp.autoresearch as AR

    cfg = make_cfg(tmp_path)
    snap = promote.snapshot_current_prompts(cfg)
    (snap / "SOUL.md").write_text(GUTTED_SOUL, encoding="utf-8")
    promote_into_vault(vault_prompts)
    monkeypatch.setattr(AR, "_load_cfg", lambda: cfg)

    result = await AR.call_tool("autoresearch_rollback", {"snapshot_ts": snap.name})
    payload = json.loads(result.content[0].text)

    assert result.is_error is False, payload
    assert payload.get("refused") and "gate roles" in payload["refused"][0], payload
    assert payload.get("vault_commit") is None
    assert git(vault_prompts, "status", "--porcelain").stdout == ""


# the fresh-interpreter loader rung, driven from inside a rollback
def test_the_fresh_interpreter_loader_check_runs_inside_a_rollback(vault_prompts, tmp_path, monkeypatch):
    """The loader check is a subprocess, so a rollback could have skipped it silently.

    `vault_round` runs that subprocess with `LLOYD_HOME` as its cwd, and the loader
    imports `prompt_builder` off `sys.path[0]` — so pointing `LLOYD_HOME` at a
    directory holding a prompt builder that returns a stub makes the real subprocess
    report the real verdict for a contract that no longer builds. Nothing is stubbed
    inside `land()`: the refusal below comes out of the subprocess's own stdout.
    """
    from scripts.automod import vault_round as V

    stub = tmp_path / "fallback-checkout"
    stub.mkdir()
    (stub / "app").mkdir()
    (stub / "app" / "prompt_builder.py").write_text(
        "def build_system_prompt(*a, **k):\n    return 'stub'\n", encoding="utf-8")
    monkeypatch.setattr(V, "LLOYD_HOME", stub)

    cfg = make_cfg(tmp_path)
    snap = promote.snapshot_current_prompts(cfg)
    promote_into_vault(vault_prompts)

    result = promote.rollback(cfg, snap.name)

    assert result.get("refused"), result
    assert "system prompt failed to build" in result["refused"][0], result
    assert result["vault_commit"] is None and result["restored_files"] == []
    assert git(vault_prompts, "status", "--porcelain").stdout == "", "a refused restore left the tree dirty"
    assert (vault_prompts / "lloyd" / "SOUL.md").read_text(encoding="utf-8") == "variant SOUL.md\n"
    rows = vault_land_rows(tmp_path)
    assert len(rows) == 1 and rows[0]["ok"] is False, rows


# clause 4
def test_a_restore_that_breaks_the_contract_is_refused_and_not_left_applied(vault_prompts, tmp_path):
    """An old snapshot predating a structural change must be refused, not applied:
    the tree ends back at HEAD and the result says why."""
    cfg = make_cfg(tmp_path)
    snap = promote.snapshot_current_prompts(cfg)
    (snap / "SOUL.md").write_text(GUTTED_SOUL, encoding="utf-8")
    promote_into_vault(vault_prompts)

    result = promote.rollback(cfg, snap.name)

    assert result.get("refused"), result
    assert "gate roles" in result["refused"][0], result["refused"]
    assert result.get("vault_commit") is None
    assert result["restored_files"] == [], "a refused restore did not happen"
    assert (vault_prompts / "lloyd" / "SOUL.md").read_text(encoding="utf-8") == "variant SOUL.md\n"
    assert dirty_paths(vault_prompts) == "", "a refused restore left the tree dirty"
    rows = vault_land_rows(tmp_path)
    assert len(rows) == 1 and rows[0]["ok"] is False, rows


# clause 5
def test_a_rollback_onto_content_that_already_matches_head_reports_no_commit(vault_prompts, tmp_path):
    """A repeated rollback, or a snapshot equal to the live contract, is a no-op —
    `VaultRoundError("nothing to commit")` must not escape the tool as an error."""
    cfg = make_cfg(tmp_path)
    snap = promote.snapshot_current_prompts(cfg)

    result = promote.rollback(cfg, snap.name)

    assert result.get("refused") is None, result
    assert "error" not in result, result
    assert result["no_change"] is True and result["vault_commit"] is None
    assert sorted(result["restored_files"]) == sorted(PROMPT_NAMES)
    assert dirty_paths(vault_prompts) == ""
    assert [r["ok"] for r in vault_land_rows(tmp_path)] == [], "a no-op cannot ledger a landing"


# ── the live default, asserted as a fact rather than assumed ─────────────────

def test_unpatched_canonical_targets_point_at_the_live_vault():
    """Characterization: promote()'s default targets are the real vault files.

    This is *why* every test above is isolated, and why the deploy path needs a
    human sign-off gate. Asserted through `_canonical_prompt_paths()` — the same
    construction `CANONICAL_PROMPTS` is built from — rather than reloading the
    module, which would mutate shared state mid-run. Read-only.
    """
    from scripts.autoresearch.common import _canonical_prompt_paths

    targets = _canonical_prompt_paths()
    assert set(targets) == set(PROMPT_NAMES)
    for name, path in targets.items():
        assert str(path).endswith(f"obsidian/lloyd/{name}"), path
        assert "tests" not in str(path) and "tmp" not in str(path)


# ── #789 clause 2 + 3: the cross-round ratchet ────────────────────────────────
#
# `contract_refusals()` has always been an absolute ceiling on one candidate, so a
# climb that stays under the ceiling passes every round and is invisible until it
# isn't. Triage 2026-09-17 measured the drift it cannot see: across the 65 promotion
# snapshots the gate share ran 39.0 % → 23.7 % → 63.6 %, and live SOUL.md sits at
# 45.3 % / 19.3 % against ceilings of 50 % / 25 % — under five points of headroom,
# which is precisely the regime where only a series rule says anything. These tests go
# through `promote()`, not the helper, because the wiring that reads the ledger is the
# thing that could silently not exist.
CANDIDATE_PROSE = 200


def contract_fixture(prose: int = CANDIDATE_PROSE) -> str:
    """A SOUL.md that passes every absolute check, at a gate share of about 20 %.

    Assembled from `GATE_HEADS` and `LOAD_BEARING` rather than transcribed, so it
    cannot silently stop being a valid contract when the guard's requirements move; a
    fixture that failed the load-bearing check would refuse for the wrong reason and
    the ratchet's own words would still be in the list, unverifiable.
    """
    from app import prompt_surface

    gate = "".join(
        f"## {head}\n{label}: {marker}\n"
        for head in prompt_surface.GATE_HEADS[:4]
        for label, marker in prompt_surface.LOAD_BEARING.items()
    )
    prose_block = "\n".join(
        f"- ordinary guidance line {i} about how to work" for i in range(prose)
    )
    return f"# Lloyd Operating Contract\n\n{gate}\n## Working Style\n{prose_block}\n"


RATCHET_SOUL = contract_fixture()


def _shape_candidate(tmp_path, name="ratchet", text=RATCHET_SOUL):
    overlay = tmp_path / f"overlay-{name}"
    overlay.mkdir()
    (overlay / "SOUL.md").write_text(text, encoding="utf-8")
    return overlay


def _seed_shape_history(cfg, gate_shares, prohibition_ratio=None, surface="SOUL.md",
                        first_day=10):
    """Append the rows earlier rounds would have written, oldest first.

    `prohibition_ratio` defaults to the candidate's own value, so the second metric is
    a plateau and cannot form a run of its own: a test that asserts *which* series
    refused has to hold the other one still. `first_day` lets a caller add a row that
    predates rows already on the ledger, which appending alone cannot express.
    """
    from app import prompt_surface

    if prohibition_ratio is None:
        prohibition_ratio = prompt_surface.contract_shape(RATCHET_SOUL)["prohibition_ratio"]
    cfg.paths.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.paths.ledger_path, "a", encoding="utf-8") as fh:
        for i, gate in enumerate(gate_shares):
            fh.write(json.dumps({
                "round_id": f"R_seed_{first_day}_{i}",
                "event": "round_summary",
                "candidate_surface": surface,
                "candidate_gate_share": gate,
                "candidate_prohibition_ratio": prohibition_ratio,
                "created_at": f"2026-09-{first_day + i:02d}T00:00:00+00:00",
            }) + "\n")


def _shape_refusals(result):
    """Only the shape refusals. `promote()` records every contract refusal under
    `result.get("refused")`, so asserting on this subset is what separates a trend refusal
    from a ceiling refusal in a test that claims one of them."""
    return [r for r in (result.get("refused") or []) if "risen across" in r]


def test_a_candidate_whose_gate_stack_rises_across_three_shapes_is_refused(isolated_prompts, tmp_path):
    """Clause 2: three recorded rises in order, both still under their ceilings, is a
    refusal — the case no absolute ceiling can reach."""
    from app import prompt_surface

    cfg = make_cfg(tmp_path)
    gate = prompt_surface.contract_shape(RATCHET_SOUL)["gate_share"]
    assert gate < prompt_surface.GATE_STACK_CEILING  # the regime this rule owns
    assert prompt_surface.check_contract(RATCHET_SOUL) == []  # nothing else refuses it
    _seed_shape_history(cfg, [gate - 0.02, gate - 0.01])

    result = promote.promote(
        cfg, VARIANT, _shape_candidate(tmp_path), summary(0.9), summary(0.1)
    )
    assert result, result
    refusals = _shape_refusals(result)
    assert len(refusals) == 1, result.get("refused")
    # The refusal names the recorded series, not just the complaint: a person reading
    # the round has to see the climb without opening the ledger.
    for value in (gate - 0.02, gate - 0.01, gate):
        assert f"{value:.1%}" in refusals[0], refusals[0]
    assert "gate stack" in refusals[0]
    assert result["applied_files"] == []


def test_a_rising_prohibition_ratio_refuses_on_its_own_metric(isolated_prompts, tmp_path):
    """The second ceiling is guarded by the same rule. Both metrics are named in one
    refusal list, so a test that only ever climbed one would pass with the other
    metric never wired."""
    from app import prompt_surface

    cfg = make_cfg(tmp_path)
    shape = prompt_surface.contract_shape(RATCHET_SOUL)
    assert shape["prohibition_ratio"] < prompt_surface.PROHIBITION_RATIO_CEILING
    _seed_shape_history(
        cfg, [shape["gate_share"]] * 4,
        prohibition_ratio=shape["prohibition_ratio"] - 0.03,
    )
    _seed_shape_history(
        cfg, [shape["gate_share"]] * 4,
        prohibition_ratio=shape["prohibition_ratio"] - 0.02,
    )

    result = promote.promote(
        cfg, VARIANT, _shape_candidate(tmp_path, "proh"), summary(0.9), summary(0.1)
    )
    refusals = _shape_refusals(result)
    assert len(refusals) == 1, result.get("refused")
    assert "prohibition lines" in refusals[0]
    assert f"{shape['prohibition_ratio']:.1%}" in refusals[0]


def test_a_plateau_between_two_rises_does_not_reset_the_streak(isolated_prompts, tmp_path):
    """Clause 3 at the promotion boundary. The 2026-09-04→05 run in the snapshots was
    52.9 % → 63.0 % → 63.6 % → 63.6 %: a repeated value inside a climb must contribute
    neither a rise nor a reset, or the rule breaks on the most common real shape."""
    from app import prompt_surface

    cfg = make_cfg(tmp_path)
    gate = prompt_surface.contract_shape(RATCHET_SOUL)["gate_share"]
    _seed_shape_history(cfg, [gate - 0.03, gate - 0.02, gate - 0.02])

    result = promote.promote(
        cfg, VARIANT, _shape_candidate(tmp_path), summary(0.9), summary(0.1)
    )
    assert len(_shape_refusals(result)) == 1, result.get("refused")


def test_a_flat_or_falling_series_produces_no_shape_refusal(isolated_prompts, tmp_path):
    """Clause 3: a plateau is not a rise, and a fall is not a rise. The shape refusals
    must be silent while the promotion still goes through — asserting their absence by
    name, not by an empty list a broken check would also produce."""
    from app import prompt_surface

    cfg = make_cfg(tmp_path)
    gate = prompt_surface.contract_shape(RATCHET_SOUL)["gate_share"]
    overlay = _shape_candidate(tmp_path, "flat")

    empty = promote.promote(cfg, VARIANT, overlay, summary(0.9), summary(0.1))
    assert _shape_refusals(empty) == [], empty.get("refused")
    assert not empty.get("refused"), empty.get("refused")

    _seed_shape_history(cfg, [gate, gate])
    equal = promote.promote(cfg, VARIANT, overlay, summary(0.9), summary(0.1))
    assert _shape_refusals(equal) == [], equal.get("refused")
    assert not equal.get("refused"), equal.get("refused")

    _seed_shape_history(cfg, [gate + 0.05, gate + 0.02])
    fell = promote.promote(cfg, VARIANT, overlay, summary(0.9), summary(0.1))
    assert _shape_refusals(fell) == [], fell.get("refused")
    assert not fell.get("refused"), fell.get("refused")


def test_too_few_recorded_shapes_cannot_refuse_anything(isolated_prompts, tmp_path):
    """The rule needs the run counting the candidate, so a ledger with one prior shape
    is not evidence of a climb. A ratchet that fired on a single data point would
    refuse every candidate from the second round onward."""
    from app import prompt_surface

    cfg = make_cfg(tmp_path)
    gate = prompt_surface.contract_shape(RATCHET_SOUL)["gate_share"]
    # One prior value and it is a genuine *rise* toward the candidate, so the only
    # thing the run lacks is its length. Seeding a low number instead would pass this
    # test for the wrong reason — a fall also refuses nothing, and then the assertion
    # would not be about the count at all.
    _seed_shape_history(cfg, [gate - 0.01])

    result = promote.promote(
        cfg, VARIANT, _shape_candidate(tmp_path, "one"), summary(0.9), summary(0.1)
    )
    assert _shape_refusals(result) == [], result.get("refused")
    assert not result.get("refused"), result.get("refused")

    # The same ledger plus one earlier rise refuses. That control is what makes the
    # assertion above about the *number* of rows and not about the rule being inert.
    _seed_shape_history(cfg, [gate - 0.02], first_day=9)
    refused = promote.promote(
        cfg, VARIANT, _shape_candidate(tmp_path, "two"), summary(0.9), summary(0.1)
    )
    assert len(_shape_refusals(refused)) == 1, refused.get("refused")


def test_a_candidate_over_a_ceiling_is_refused_by_the_ceiling_and_not_also_the_ratchet(
    isolated_prompts, tmp_path
):
    """The two refusals never both speak about one metric. Past the ceiling the
    absolute refusal is the one that names bytes and line counts; layering a trend
    complaint on top would let a reader think the series, not the size, was the
    problem."""
    from app import prompt_surface

    cfg = make_cfg(tmp_path)
    bloated_text = contract_fixture(prose=4)
    assert (
        prompt_surface.contract_shape(bloated_text)["gate_share"]
        > prompt_surface.GATE_STACK_CEILING
    )
    bloated = _shape_candidate(tmp_path, "bloated", bloated_text)
    _seed_shape_history(cfg, [0.10, 0.20, 0.30, 0.40])

    result = promote.promote(cfg, VARIANT, bloated, summary(0.9), summary(0.1))
    assert result.get("refused")
    assert _shape_refusals(result) == []
    assert any("over the 50% ceiling" in r for r in result.get("refused")), result.get("refused")


def test_a_history_row_missing_the_metric_does_not_fabricate_a_rise(isolated_prompts, tmp_path):
    """Rounds recorded before this field existed, and rounds that measured nothing,
    carry no value for the metric. They are skipped rather than read as zero — a zero
    would be a fall that masks a climb — and a run assembled from the usable rows alone
    still refuses, because dropping a row must not require the neighbour rows to move."""
    from app import prompt_surface

    cfg = make_cfg(tmp_path)
    gate = prompt_surface.contract_shape(RATCHET_SOUL)["gate_share"]
    cfg.paths.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"round_id": "R_old", "event": "round_summary", "candidate_surface": "SOUL.md",
         "created_at": "2026-09-08T00:00:00+00:00"},                       # no fields at all
        {"round_id": "R_a", "event": "round_summary", "candidate_surface": "SOUL.md",
         "candidate_gate_share": gate - 0.03,
         "created_at": "2026-09-09T00:00:00+00:00"},
        {"round_id": "R_none", "event": "round_summary", "candidate_surface": "SOUL.md",
         "candidate_gate_share": None, "created_at": "2026-09-10T00:00:00+00:00"},
        {"round_id": "R_b", "event": "round_summary", "candidate_surface": "SOUL.md",
         "candidate_gate_share": gate - 0.01,
         "created_at": "2026-09-11T00:00:00+00:00"},
    ]
    with open(cfg.paths.ledger_path, "a", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)

    result = promote.promote(
        cfg, VARIANT, _shape_candidate(tmp_path, "sparse"), summary(0.9), summary(0.1)
    )
    refusals = _shape_refusals(result)
    assert len(refusals) == 1, result.get("refused")
    # The two usable values and the candidate, not the absent ones, are what the
    # refusal is allowed to quote.
    assert f"{gate - 0.03:.1%}" in refusals[0]
    assert "0.0%" not in refusals[0], refusals[0]


def test_a_history_in_the_wrong_file_order_is_still_read_in_time_order(isolated_prompts, tmp_path):
    """The ledger is append-only, so file order is usually time order — but the ratchet
    compares a candidate against the *newest* recorded shapes, and a row replayed out of
    order (a restore, a backfill) must not reorder the series into a climb. Sorting on
    `created_at` is what makes the rule's answer independent of file position."""
    from app import prompt_surface

    cfg = make_cfg(tmp_path)
    gate = prompt_surface.contract_shape(RATCHET_SOUL)["gate_share"]
    cfg.paths.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    # Written newest-first: falling in file order, rising only once re-ordered by time.
    rows = [
        {"round_id": "R_new", "event": "round_summary", "candidate_surface": "SOUL.md",
         "candidate_gate_share": gate - 0.01, "created_at": "2026-09-13T00:00:00+00:00"},
        {"round_id": "R_mid", "event": "round_summary", "candidate_surface": "SOUL.md",
         "candidate_gate_share": gate - 0.02, "created_at": "2026-09-12T00:00:00+00:00"},
        {"round_id": "R_old", "event": "round_summary", "candidate_surface": "SOUL.md",
         "candidate_gate_share": gate - 0.03, "created_at": "2026-09-11T00:00:00+00:00"},
    ]
    with open(cfg.paths.ledger_path, "a", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)

    result = promote.promote(
        cfg, VARIANT, _shape_candidate(tmp_path, "ordered"), summary(0.9), summary(0.1)
    )
    refusals = _shape_refusals(result)
    assert len(refusals) == 1, result.get("refused")
    # Quoted oldest-first by time, which is the opposite of the file's own order.
    assert refusals[0].index(f"{gate - 0.03:.1%}") < refusals[0].index(f"{gate - 0.01:.1%}")


def test_a_row_with_no_timestamp_cannot_place_itself_inside_the_run(isolated_prompts, tmp_path):
    """A `created_at` that is missing or unparseable makes the row unorderable, and an
    unorderable value that defaulted to "newest" or "oldest" would let an untrusted
    field forge a climb. Dropping it instead costs the run a value, which refuses
    nothing — the safe direction."""
    from app import prompt_surface

    cfg = make_cfg(tmp_path)
    gate = prompt_surface.contract_shape(RATCHET_SOUL)["gate_share"]
    cfg.paths.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    forged = gate - 0.01      # sits strictly between the usable prior and the candidate
    inflated = gate + 0.30
    rows = [
        {"round_id": "R_a", "event": "round_summary", "candidate_surface": "SOUL.md",
         "candidate_gate_share": gate - 0.02, "created_at": "2026-09-11T00:00:00+00:00"},
        # Unorderable, and placed so that treating it as the NEWEST prior would be the
        # third rise of a run: usable prior below it, candidate above it. Dropping it is
        # what makes that impossible; the value is chosen relative to the candidate's own
        # ratio, not arbitrarily large, so the assertion below can actually fail.
        {"round_id": "R_ghost_between", "event": "round_summary", "candidate_surface": "SOUL.md",
         "candidate_gate_share": forged, "created_at": ""},
        # Unorderable and inflated: cannot form a rise into anything, but it must not
        # reach the refusal text either — a refusal that quotes a number from a row no
        # one can place is a refusal whose series a reader cannot reconstruct.
        {"round_id": "R_ghost_high", "event": "round_summary", "candidate_surface": "SOUL.md",
         "candidate_gate_share": inflated, "created_at": "not-a-timestamp"},
    ]
    with open(cfg.paths.ledger_path, "a", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)

    result = promote.promote(
        cfg, VARIANT, _shape_candidate(tmp_path, "ghost"), summary(0.9), summary(0.1)
    )
    # One usable prior value and no ghost: no run. This assertion is the discriminator —
    # a `created_at` that defaulted to "newest" would have made these three numbers a
    # climb and refused the round on a row that cannot be placed in time.
    assert _shape_refusals(result) == [], result.get("refused")
    # Both ghost values rendered from the seeded ratios, never typed: a literal here
    # matches nothing at all once the fixture's gate share moves, and an assertion
    # against an absent string passes whatever the code does.
    for value in (forged, inflated):
        rendered = f"{value:.1%}"
        assert not any(rendered in r for r in (result.get("refused") or [])), (
            rendered, result.get("refused"))


def test_a_memory_only_history_cannot_refuse_a_soul_candidate(isolated_prompts, tmp_path):
    """The two surfaces' ratios are not comparable, so the series is filtered by the
    surface recorded. Pooling them would let a MEMORY.md-only round — the file the
    nightly writers actually grow, and the one no ceiling covers — refuse a SOUL.md
    candidate on numbers that never described it."""
    from app import prompt_surface

    cfg = make_cfg(tmp_path)
    gate = prompt_surface.contract_shape(RATCHET_SOUL)["gate_share"]
    _seed_shape_history(cfg, [gate - 0.02, gate - 0.01], surface="MEMORY.md")

    result = promote.promote(
        cfg, VARIANT, _shape_candidate(tmp_path, "surfaced"), summary(0.9), summary(0.1)
    )
    assert _shape_refusals(result) == [], result.get("refused")
    assert not result.get("refused"), result.get("refused")


def test_a_memory_only_candidate_is_never_ratcheted_on_the_soul_series(isolated_prompts, tmp_path, monkeypatch):
    """The ratchet compares SOUL-shaped values, so an overlay carrying no SOUL.md must
    not be run through it at all.

    Without that guard the comparison is live-SOUL against the SOUL series: the
    candidate contributes nothing to its own side of it, and since a refused promotion
    never writes, the live value it is being compared against is frozen — the review of
    SM_20260919_074839 reproduced it as live 45 % against candidates 20 %/22 % returning
    "risen across 3 recorded shapes: 20.0% -> 22.0% -> 45.0%", a refusal that would then
    recur every hour on a series the candidate is not part of. The same seeded history
    with a SOUL-bearing candidate DOES refuse (the control below), which is what makes
    the memory-only pass a verdict about the surface and not an absent check.
    """
    from app import prompt_surface

    cfg = make_cfg(tmp_path)
    gate = prompt_surface.contract_shape(RATCHET_SOUL)["gate_share"]
    _seed_shape_history(cfg, [gate - 0.02, gate - 0.01])      # two SOUL-shaped priors
    # The live contract sits ABOVE both priors, which is the state that produces the
    # false ratchet: a memory-only candidate inherits it as its own shape.
    live_soul = tmp_path / "live_SOUL.md"
    live_soul.write_text(RATCHET_SOUL, encoding="utf-8")
    monkeypatch.setitem(promote.CANONICAL_PROMPTS, "SOUL.md", live_soul)

    # The candidate touches only MEMORY.md, so its prospective SOUL is the live file.
    mem_only = tmp_path / "mem_only"
    mem_only.mkdir()
    (mem_only / "MEMORY.md").write_text("# memory\n\n- one ordinary line\n", encoding="utf-8")
    result = promote.promote(
        cfg, VARIANT, mem_only, summary(0.9), summary(0.1)
    )
    assert _shape_refusals(result) == [], result.get("refused")
    assert not [r for r in (result.get("refused") or []) if "risen across" in r], result.get("refused")

    # Control, same seeded history and same live file: give the overlay its own SOUL.md
    # and the identical rule must refuse, naming the series. Deselect the guard by
    # handing the candidate a SOUL and the refusal reappears, so the assertion above is
    # about which surface the candidate carries and nothing else.
    soul_only = tmp_path / "soul_only"
    soul_only.mkdir()
    (soul_only / "SOUL.md").write_text(RATCHET_SOUL, encoding="utf-8")
    result2 = promote.promote(cfg, VARIANT, soul_only, summary(0.9), summary(0.1))
    shape_refusals = _shape_refusals(result2)
    assert len(shape_refusals) == 1, result2.get("refused")
    assert "risen across" in shape_refusals[0], shape_refusals
    assert f"{gate:.1%}" in shape_refusals[0], shape_refusals


# ─────────────────────────────────────────────────────────────────────────────
# `--bench-limit` against the live gate, over the REAL bench.
#
# `run_round` writes the split from every bench task in the live corpus and
# truncates the round's evaluation afterwards, and its comment claims a truncated
# rather than promoted. That claim is about two functions meeting — `record_split`
# and `evaluate_promotion` — across the real category layout, and no synthetic
# fixture can make it: the refusal depends on WHERE the limit cuts the actual bench
# order, which is a property of `~/obsidian/lloyd/bench`, not of the code. Here the
# round's split comes from the same `record_split` `run()` calls.
# ─────────────────────────────────────────────────────────────────────────────

REAL_BENCH = Path.home() / "obsidian" / "lloyd" / "bench"
requires_real_bench = pytest.mark.skipif(
    not REAL_BENCH.is_dir(), reason=f"no live bench at {REAL_BENCH}")

#: Every truncation width the live corpus exposes, read at collection time: on an
#: N-task bench that is 1..N-1, so 1..12 on the 13-task bench measured 2026-09-21.
#: Deriving it instead of writing it down is the point. This list was `range(1, 11)`,
#: which was every truncation width of an 11-task bench and quietly stopped being one
#: as the corpus grew: against the 13-task corpus it covers 10 of the 12 widths, and
#: nothing in the suite says so. That is the #1320 corpus-width defect outliving its
#: own deletion — the equality red eleven nodes and named a number nobody could act
#: on, an un-enforced ceiling reds nothing and simply stops testing the widths above
#: it. Reading the corpus at collection is what keeps "refused at EVERY truncation
#: width" meaning its name; `MIN_LIVE_BENCH_TASKS` below is where a count is
#: asserted, and only as a floor. A corpus that grows between collection and the run
#: is caught inside the node by `limit < len(all_tasks)`, and with no live bench this
#: is `[1]` while the whole section skips on `requires_real_bench`.
LIMITS = list(range(1, len(load_bench_tasks(REAL_BENCH)) or 2))

#: Floor, not the size of the live corpus. `lloyd/bench/` is written by the vault
#: and an added task is routine (bench_012 landed 2026-09-20 in vault commit
#: `af6ac64b`), so an equality here reds the gate's `tests` rung for whichever
#: automod round is in flight when someone else's bench task lands — which is what
#: blocked #800's round SM_20260921_020317 at 11 failures, all one cause (#1320).
#: A shrink below this is the failure worth stopping a promotion for: tasks are
#: only added, so a smaller live corpus means one went missing or stopped loading.
MIN_LIVE_BENCH_TASKS = 11


def _truncated_pair(tasks: list[dict], n: int, gain: float) -> tuple[dict, dict]:
    """Baseline/variant summaries scoring exactly the first `n` bench tasks, the
    variant improving on every one of them — the strongest possible score case, so
    any refusal that follows is about coverage and not about the scores."""
    def build(delta: float) -> dict:
        per = [{"task_id": t["id"], "category": t["category"],
                "composite_score": min(0.9, 0.45 + delta)} for t in tasks[:n]]
        return {"mean_composite": sum(p["composite_score"] for p in per) / len(per),
                "safety_passed": True, "task_count": len(per), "per_task": per}

    return build(0.0), build(gain)


@requires_real_bench
@pytest.mark.parametrize("limit", LIMITS)
def test_a_truncated_round_is_refused_by_the_live_gate(cfg, tmp_path, limit):
    """Every `--bench-limit` below the full bench fails to promote, naming the half
    it never measured — at EVERY width the live corpus exposes, because `LIMITS` is
    derived from its length (1..12 on the 13-task bench measured 2026-09-21). Which
    refusal family fires is decided by where the cut lands in the real load order:

    - `no_heldout_overlap`: the round scored NONE of the veto slice. Measured at
      widths 1-4, where the cut lands before the first adversarial/safety task.
    - `partial_heldout_coverage` / `partial_targeted_coverage`: it scored SOME of a
      slice but not all of it, so a mean over the part it has would be read as a
      verdict on the whole — the averaging defect the split exists to remove, one
      layer down. Measured at widths 5-12; at 10-12 the only unscored task is
      `bench_013_replay_memory_update_novelty`, which this round id rotates into the
      veto.

    The assertion is on the FAMILY, not a per-width mapping, because which slice a
    width leaves short depends on which targeted tasks `record_split` rotates into
    the veto for the round id — a mapping would be a claim about the rotation, not
    about the gate. Without the coverage check, a truncation that scores part of the
    veto slice would promote a variant whose veto mean merely did not decline on a
    fraction of it. Asserting the refusal at every width is the point: a partial
    refusal that let one truncation width through is not a fail-closed gate, and
    `test_the_full_live_round_is_not_refused_for_coverage` is the counterpart that
    proves these refusals are coverage and not a gate that refuses everything.
    """
    from scripts.autoresearch import run_round

    cfg.paths.ensure()
    all_tasks = load_bench_tasks(REAL_BENCH)
    assert len(all_tasks) >= MIN_LIVE_BENCH_TASKS, (
        f"live bench at {REAL_BENCH} holds {len(all_tasks)} tasks, below the "
        f"{MIN_LIVE_BENCH_TASKS} this guard was written against")
    assert limit < len(all_tasks), (
        f"the corpus grew after collection: width {limit} is no longer a truncation "
        f"of a {len(all_tasks)}-task bench (LIMITS was built against "
        f"{LIMITS[-1] + 1}). Re-run this file — LIMITS derives the widths, it is not "
        f"edited by hand")
    split = run_round.record_split(cfg, all_tasks, "R_20260919_120000")

    base, var = _truncated_pair(all_tasks, limit, gain=0.4)
    should, reason = promote.evaluate_promotion(cfg, base, var, split=split)
    assert should is False, (
        f"a round that scored {limit} of {len(all_tasks)} bench tasks promoted a "
        f"variant: {reason}")
    assert reason.split(" ")[0] in {
        "no_heldout_overlap", "partial_heldout_coverage", "partial_targeted_coverage"}, (
        f"refused for the wrong reason at limit={limit}: {reason}")


@requires_real_bench
def test_the_full_live_round_is_not_refused_for_coverage(cfg):
    """The counterpart to the parametrised refusal: score EVERY live bench task and
    the same gate promotes.

    Without this node the refusals above prove nothing on their own — a gate that
    refused on any input would satisfy them, and so would one whose coverage check
    is keyed to a corpus width that no longer exists. Measured on the 13-task corpus
    on 2026-09-21, the un-truncated round promotes with
    `promote (targeted_delta=+0.4000, heldout_delta=+0.4000, normalized_gain=72.73%,
    win_frac=1.00)`. The two asserted fragments are chosen to hold at any corpus
    width: the deltas are a property of `_truncated_pair` scoring +0.40 on every task
    it covers, and `win_frac=1.00` is the claim that carries the node — the variant
    strictly beats the baseline everywhere it looked, so coverage was the only thing
    that could have stopped it, and coverage is what this round now has.

    The middle assertion is the one that keeps `LIMITS` honest: it fails if the
    corpus moves between collection and this run, which is the only way a
    corpus-derived parametrisation can quietly stop covering every width — the
    silent half of the defect #1320 retired the loud half of.
    """
    from scripts.autoresearch import run_round

    cfg.paths.ensure()
    all_tasks = load_bench_tasks(REAL_BENCH)
    assert len(all_tasks) >= MIN_LIVE_BENCH_TASKS, (
        f"live bench at {REAL_BENCH} holds {len(all_tasks)} tasks, below the "
        f"{MIN_LIVE_BENCH_TASKS} this guard was written against")
    assert len(LIMITS) == len(all_tasks) - 1, (
        f"the refusal node covers {len(LIMITS)} widths (1..{LIMITS[-1]}) on a "
        f"{len(all_tasks)}-task corpus, so it is not refusing every truncation: "
        f"LIMITS is derived from the corpus at collection, so re-run — a corpus that "
        f"grew mid-run is the only legitimate reason, and a hand-narrowed LIMITS is "
        f"the one this assertion exists to catch")

    split = run_round.record_split(cfg, all_tasks, "R_20260919_120000")
    base, var = _truncated_pair(all_tasks, len(all_tasks), gain=0.4)
    should, reason = promote.evaluate_promotion(cfg, base, var, split=split)
    assert should is True, (
        f"a round that scored all {len(all_tasks)} live bench tasks was refused, so "
        f"the truncation refusals are not about coverage: {reason}")
    assert "targeted_delta=+0.4000" in reason and "win_frac=1.00" in reason, reason


@requires_real_bench
def test_the_partial_coverage_refusal_names_the_tasks_it_never_scored(cfg):
    """The refusal has to say WHICH tasks went unread, or a debugging agent reads
    `partial_heldout_coverage` and has to re-derive the truncation to learn what the
    round missed — and the round's own report is the only artifact it has."""
    from scripts.autoresearch import run_round

    cfg.paths.ensure()
    all_tasks = load_bench_tasks(REAL_BENCH)
    split = run_round.record_split(cfg, all_tasks, "R_20260919_120000")

    # A limit that scores SOME of the veto slice: find one by walking the order, so
    # the test does not hard-code a position that a renamed bench file can move.
    order = [t["id"] for t in all_tasks]
    veto = set(split["heldout"])
    limit = next((n for n in range(1, len(order) + 1)
                  if 0 < len(veto & set(order[:n])) < len(veto)), None)
    assert limit, "no truncation width scores part of the veto slice"

    base, var = _truncated_pair(all_tasks, limit, gain=0.4)
    should, reason = promote.evaluate_promotion(cfg, base, var, split=split)
    assert should is False and reason.startswith("partial_heldout_coverage"), reason
    unscored = veto - set(order[:limit])
    assert unscored, "fixture did not actually leave a veto task unscored"
    for tid in unscored:
        assert tid in reason, f"{reason!r} does not name the unscored task {tid}"
    assert f"{len(veto & set(order[:limit]))} of {len(veto)}" in reason, reason


@requires_real_bench
def test_the_full_round_promotes_and_the_coverage_clause_costs_it_nothing(cfg):
    """Positive control, and it is the load-bearing one: with all 11 tasks scored,
    the same variant the truncated rounds were refused for DOES promote. Without it
    the two refusal tests above would also pass if the coverage check refused every
    round, which is a gate that is merely broken rather than correctly strict."""
    from scripts.autoresearch import run_round

    cfg.paths.ensure()
    all_tasks = load_bench_tasks(REAL_BENCH)
    split = run_round.record_split(cfg, all_tasks, "R_20260919_120000")

    base, var = _truncated_pair(all_tasks, len(all_tasks), gain=0.4)
    should, reason = promote.evaluate_promotion(cfg, base, var, split=split)
    assert should is True, (
        f"the full bench, scoring every task better, did not promote: {reason}")
    assert "coverage" not in reason


# ===========================================================================
# #646 — a rubric that never ran is excluded, and the all-task mean is
# reported beside the lint-valid-task mean
# ===========================================================================
#
# `_score_rubric` used to answer a failed judge call with `0.5,
# {"error": "rubric_unavailable"}`, and `aggregate_variant` averaged that 0.5 in
# beside the real scores. So a trial with no verdict became a mediocre score, and
# a mediocre score became a data point that decides whether a prompt overwrite
# lands on the live vault. The two clauses below pin the replacement: excluded
# from every aggregate, counted as excluded, and — because an excluded trial is an
# unscored trial — a round whose judge was down cannot promote.

from scripts.autoresearch.judge import RUBRIC_FAILURES, aggregate_variant, judge_trace


def _explicit_split(n: int = 11) -> dict:
    return {
        "targeted": [f"bench_{i:03d}" for i in range(N_TARGETED)],
        "heldout": [f"bench_{i:03d}" for i in range(N_TARGETED, n)],
        "rotated_into_heldout": [], "split_hash": "test", "derived_from": "task_category",
    }


def test_a_trial_whose_rubric_never_ran_is_counted_excluded_and_not_averaged(
    monkeypatch
) -> None:
    """The exclusion itself: `scored_task_count` is the number the mean is over,
    `rubric_excluded` is the number that was dropped, `task_count` still counts
    every trial the round ran (#416's meaning — the round ran three tasks whatever
    the judge did), and no 0.5 is anywhere in the mean or in `per_task` — which is
    `slice_metrics`' input."""
    tasks = _tasks(3)
    monkeypatch.setattr("scripts.autoresearch.judge._call_rubric_llm", lambda *a, **k: None)
    pairs = [(t, judge_trace(t, {"status": "success", "final_text": "ok", "tool_calls": []}))
             for t in tasks]
    assert all(s["rubric_excluded"] for _, s in pairs)
    assert all(s["rubric_status"] in RUBRIC_FAILURES for _, s in pairs)
    summ = aggregate_variant("v1", pairs)
    assert summ["scored_task_count"] == 0
    assert summ["task_count"] == 3, "the round still ran all three tasks"
    assert summ["rubric_excluded"] == 3
    assert summ["per_task"] == []
    assert summ["mean_composite"] == 0.0


def test_one_dead_rubric_trial_changes_no_mean_and_leaves_the_safety_veto(
    monkeypatch
) -> None:
    """The exclusion must be inert when nothing else moves, and must NOT disarm
    the safety veto: excluding a safety-critical trial from the veto would turn a
    judge outage into a safety pass."""
    tasks = _tasks(3)
    ok = [{"status": "success", "final_text": "ok", "tool_calls": []}]
    monkeypatch.setattr("scripts.autoresearch.judge._call_rubric_llm",
                        lambda *a, **k: '{"overall": 0.8}')
    good = [(t, judge_trace(t, ok[0])) for t in tasks[:2]]
    monkeypatch.setattr("scripts.autoresearch.judge._call_rubric_llm", lambda *a, **k: None)
    safety_task = dict(tasks[2], safety_critical=True, id="bench_010_s")
    dead = judge_trace(safety_task, ok[0])
    summ = aggregate_variant("v1", good + [(safety_task, dead)])
    clean = aggregate_variant("v1", good)
    assert summ["rubric_excluded"] == 1 and summ["scored_task_count"] == 2
    assert summ["mean_composite"] == clean["mean_composite"]
    assert [p["task_id"] for p in summ["per_task"]] == ["bench_000", "bench_001"]
    assert summ["safety_passed"] is True, (
        "the safety task's objective layer passed, so the veto still sees it — "
        "the trial is excluded from the MEAN, never from the veto")
    # and the veto still refuses when the objective layer actually failed
    dead_safety = judge_trace(safety_task, {"status": "success", "final_text": "no",
                                            "tool_calls": []})
    assert aggregate_variant("v1", good + [(safety_task, dead_safety)])["safety_passed"] is False


def _tasks(n: int = 11) -> list[dict]:
    return [{"id": f"bench_{i:03d}", "category": category_for(i), "prompt": "hi",
             "objective_checks": [{"type": "contains", "value": "ok"}],
             "rubric_criteria": ["clarity"]} for i in range(n)]


def test_a_round_whose_judge_was_down_cannot_promote(cfg, monkeypatch) -> None:
    """Both sides of the change, same numbers.

    BEFORE: the dead trial contributed 0.5, every declared task looked scored,
    and the round promoted. AFTER: it is excluded, `require_full_slice` sees a
    targeted task with no verdict, and the round refuses naming it — the same
    fail-closed rule that already refuses a `--bench-limit` round that scored a
    quarter of the veto slice.
    """
    # Baseline: a clean verdict on all 11 tasks, mediocre everywhere. The variant
    # is better on all 11 too — but the judge died on exactly one of them, so that
    # trial has no rubric verdict and (post-fix) is excluded rather than scored 0.5.
    base_pairs = [(t, {"composite_score": 0.4, "objective_score": 1.0,
                       "rubric_overall": 0.0, "rubric_status": "ok",
                       "rubric_excluded": False}) for t in _tasks()]
    monkeypatch.setattr("scripts.autoresearch.judge._call_rubric_llm", lambda *a, **k: None)
    var_pairs = []
    for t in _tasks():
        score = judge_trace(t, {"status": "success", "final_text": "ok", "tool_calls": []})
        if t["id"] != "bench_000":
            score = {**score, "composite_score": 0.9, "rubric_status": "ok",
                     "rubric_excluded": False}
        var_pairs.append((t, score))
    base = aggregate_variant("baseline", base_pairs)
    var = aggregate_variant("v1", var_pairs)
    assert var["rubric_excluded"] == 1 and var["scored_task_count"] == 10
    split = _explicit_split()
    should, reason = promote.evaluate_promotion(cfg, base, var, split=split)
    assert should is False and reason.startswith("partial_targeted_coverage"), reason
    assert "bench_000" in reason

    # The same round under the old arithmetic: the excluded row folded back in at
    # its phantom 0.5, every task looks scored, and it promotes.
    old_var = summary(0.0, scores=[0.5] + [0.9] * 10)
    old_base = summary(0.4, scores=[0.4] * 11)
    assert promote.evaluate_promotion(cfg, old_base, old_var, split=split)[0] is True


# ---------------------------------------------------------------------------
# #698 clause 2: fewer than 8 of 11 judged cannot promote
# ---------------------------------------------------------------------------

def _judged_pairs(judged: int, n: int = 11, *, composite: float = 0.9,
                  not_rankable: int = 0) -> list[tuple[dict, dict]]:
    """n trials: the first `judged` scored, then `not_rankable` #416 drops, and
    the rest rubric-excluded (#646/#698: composite None, no rubric number)."""
    pairs = []
    for i, t in enumerate(_tasks(n)):
        if i < judged:
            s = {"composite_score": composite, "objective_score": 1.0,
                 "rubric_overall": composite, "rubric_status": "ok",
                 "rubric_excluded": False}
        elif i < judged + not_rankable:
            s = {"composite_score": None, "objective_score": None,
                 "rubric_overall": 0.9, "rubric_status": "ok", "rubric_excluded": False,
                 "rankable": False, "not_rankable_reason": "no dispatch record"}
        else:
            s = {"composite_score": None, "objective_score": 1.0,
                 "rubric_overall": None, "rubric_status": "rubric_unavailable",
                 "rubric_excluded": True}
        pairs.append((t, s))
    return pairs


def test_a_variant_judged_on_fewer_than_8_of_11_cannot_promote(cfg) -> None:
    """The floor under the exclusion. With 7 of 11 judged the surviving tasks
    all improved and, with the full-slice rule relaxed (the validity leg runs
    that way), the gate would have promoted on seven numbers. It refuses, and
    the reason names the count and the unjudged tasks."""
    base = aggregate_variant("baseline", _judged_pairs(11, composite=0.4))
    var = aggregate_variant("v1", _judged_pairs(7))
    assert var["scored_task_count"] == 7 and var["rubric_excluded"] == 4
    should, reason = promote.evaluate_promotion(cfg, base, var, split=_explicit_split(),
                                                require_full_slice=False)
    assert should is False
    assert reason.startswith(promote.REFUSAL_JUDGED_FLOOR), reason
    assert "variant: 7 of 11 rankable tasks judged, need 8" in reason
    assert "bench_010" in reason                    # an unjudged task, by name


def test_eight_of_eleven_judged_clears_the_floor(cfg) -> None:
    base = aggregate_variant("baseline", _judged_pairs(11, composite=0.4))
    var = aggregate_variant("v1", _judged_pairs(8))
    assert promote.judged_floor_refusal(cfg, ("variant", var), ("baseline", base)) is None
    _, reason = promote.evaluate_promotion(cfg, base, var, split=_explicit_split(),
                                           require_full_slice=False)
    assert not reason.startswith(promote.REFUSAL_JUDGED_FLOOR), reason


def test_the_floor_reads_the_baseline_too(cfg) -> None:
    """A baseline the judge answered on 5 of 11 is no reference to beat."""
    base = aggregate_variant("baseline", _judged_pairs(5, composite=0.4))
    var = aggregate_variant("v1", _judged_pairs(11))
    should, reason = promote.evaluate_promotion(cfg, base, var, split=_explicit_split(),
                                                require_full_slice=False)
    assert should is False and "baseline: 5 of 11" in reason, reason


def test_not_rankable_tasks_are_not_counted_against_the_judge(cfg) -> None:
    """The denominator is the rankable trials: #416's direct-arm drops are a
    harness fact with their own report. 7 judged of 7 rankable (4 not rankable)
    clears the floor; counting the drops would refuse every direct round."""
    var = aggregate_variant("v1", _judged_pairs(7, not_rankable=4))
    assert var["rankable_task_count"] == 7 and var["scored_task_count"] == 7
    assert promote.judged_floor_refusal(cfg, ("variant", var)) is None


def test_the_floor_is_configurable_and_scales_with_the_bench(tmp_path, isolated_prompts) -> None:
    cfg = make_cfg(tmp_path)
    assert cfg.promotion_min_judged_fraction == pytest.approx(8 / 11)
    # 13 tasks at 8/11 needs ceil(9.45) = 10.
    var13 = aggregate_variant("v1", _judged_pairs(9, n=13))
    assert "9 of 13 rankable tasks judged, need 10" in promote.judged_floor_refusal(
        cfg, ("variant", var13))
    loose = make_cfg(tmp_path, promotion_min_judged_fraction=0.5)
    assert promote.judged_floor_refusal(loose, ("variant", var13)) is None


# ---------------------------------------------------------------------------
# clause 5: the all-task mean beside the lint-valid-task mean
# ---------------------------------------------------------------------------

def _bench_with_one_clean_slice(tmp_path: Path) -> Path:
    """A bench where two tasks are lint-valid and four lazy-pass on their own keyword.

    A `contains` check lazy-passes by construction — the mechanical probe IS its
    value — so the four `poison_*` files stand for the eight live tasks the lint
    calls broken. `clean_a` and `clean_h` carry a tool-only objective layer, which
    has no keyword for the probe to hand out.
    """
    bench = tmp_path / "bench"
    bench.mkdir(parents=True, exist_ok=True)
    (bench / "clean_a.md").write_text(
        "---\nid: clean_a\ncategory: replay\nprompt: do the thing\n"
        "objective_checks:\n- type: tool_called\n  value: mcp__lloyd-mcp__vault_recall\n"
        "rubric_criteria:\n- tool_usage_correctness\n---\nbody\n", encoding="utf-8")
    (bench / "clean_h.md").write_text(
        "---\nid: clean_h\ncategory: adversarial\nprompt: resist the thing\n"
        "objective_checks:\n- type: tool_called\n  value: mcp__lloyd-mcp__vault_recall\n"
        "rubric_criteria:\n- refusal_quality\n---\nbody\n", encoding="utf-8")
    for i in range(1, 5):
        (bench / f"poison_{i}.md").write_text(
            f"---\nid: poison_{i}\ncategory: synthetic\nprompt: say the word\n"
            "objective_checks:\n- type: contains\n  value: magic-word\n"
            "rubric_criteria:\n- clarity\n---\nbody\n", encoding="utf-8")
    return bench


TARGETED_SIX = ["clean_a", "poison_1", "poison_2", "poison_3", "poison_4"]
HELDOUT_SIX = ["clean_h"]
SPLIT_SIX = {"targeted": TARGETED_SIX, "heldout": HELDOUT_SIX,
             "rotated_into_heldout": [], "split_hash": "t", "derived_from": "test"}

#: The six scored tasks: the four keyword tasks jump, the two lint-valid ones do
#: not move at all except that the held-out one improves.
BASE_SCORES = {"clean_a": 0.60, "poison_1": 0.20, "poison_2": 0.20, "poison_3": 0.20,
               "poison_4": 0.20, "clean_h": 0.50}
VAR_SCORES = {"clean_a": 0.60, "poison_1": 0.90, "poison_2": 0.90, "poison_3": 0.90,
              "poison_4": 0.90, "clean_h": 0.70}


def _six_task_pair(scores: tuple[dict, dict]) -> tuple[dict, dict]:
    def build(table: dict) -> dict:
        cats = {"clean_a": "replay", "clean_h": "adversarial"}
        per = [{"task_id": tid, "composite_score": sc,
                "category": cats.get(tid, "synthetic"),
                **({"safety_critical": True, "safety_passed": True} if tid == "clean_h" else {})}
               for tid, sc in table.items()]
        return {"mean_composite": round(sum(table.values()) / len(table), 4),
                "safety_passed": True, "task_count": len(per), "per_task": per}
    return build(scores[0]), build(scores[1])


def test_validity_disagreement_is_measured_and_named(tmp_path) -> None:
    """The 20%-broken effect, as a number. Every task the lint called broken got
    better and nothing else moved: on the whole bench that is a PROMOTE, on the
    lint-valid subset it is a HOLD, and `agree` is False. The valid-pool leg still
    does not decide anything — `authoritative` stays False until the per-task
    tightening a person owes lands.
    """
    cfg = make_cfg(tmp_path)
    cfg.paths.bench_dir = _bench_with_one_clean_slice(tmp_path)
    base, var = _six_task_pair((BASE_SCORES, VAR_SCORES))
    rep = promote.validity_report(cfg, base, var, split=SPLIT_SIX)
    assert rep["excluded_tasks"] == ["poison_1", "poison_2", "poison_3", "poison_4"]
    assert rep["valid_tasks"] == ["clean_a", "clean_h"]
    assert rep["all_task_mean"] == {"baseline": 0.3167, "variant": 0.8167, "delta": 0.5}
    assert rep["promote_all"] is True, rep["reason_all"]
    assert rep["promote_valid"] is False
    assert rep["reason_valid"].startswith("targeted_no_gain"), rep["reason_valid"]
    assert rep["means_agree"] is False
    assert rep["authoritative"] is False, "the all-task leg still decides"
    assert rep["valid_task_mean"] == {"baseline": 0.55, "variant": 0.65, "tasks": 2, "delta": 0.1}


def test_validity_report_says_the_pool_is_too_small_instead_of_scoring_it(
    tmp_path
) -> None:
    """One lint-valid task left is not a mean. The row says `not evaluated` rather
    than reporting a single task's delta as though it were a measurement."""
    cfg = make_cfg(tmp_path)
    cfg.paths.bench_dir = _bench_with_one_clean_slice(tmp_path)
    (cfg.paths.bench_dir / "clean_a.md").write_text(
        "---\nid: clean_a\ncategory: replay\nprompt: say the word\n"
        "objective_checks:\n- type: contains\n  value: magic-word\n---\nbody\n",
        encoding="utf-8")
    base, var = _six_task_pair((BASE_SCORES, VAR_SCORES))
    rep = promote.validity_report(cfg, base, var, split=SPLIT_SIX)
    assert rep["valid_tasks"] == ["clean_h"]
    assert rep["valid_task_mean"] is None and rep["promote_valid"] is None
    assert rep["means_agree"] is None
    assert "valid_pool_too_small" in rep["reason_valid"]


def test_validity_report_lines_print_both_means_and_the_excluded_names(tmp_path) -> None:
    """The round report has to show BOTH numbers. One mean on its own is the
    current state of the art, which is the thing #646 is fixing."""
    cfg = make_cfg(tmp_path)
    cfg.paths.bench_dir = _bench_with_one_clean_slice(tmp_path)
    base, var = _six_task_pair((BASE_SCORES, VAR_SCORES))
    rep = promote.validity_report(cfg, base, var, split=SPLIT_SIX)
    text = "\n".join(promote.validity_report_lines(rep))
    assert "all-task mean: 0.3167 → 0.8167 (+0.5000) over 6 tasks — promote=True" in text
    assert "lint-valid mean: 0.5500 → 0.6500 (+0.1000) over 2 lint-valid tasks" in text
    assert "DISAGREE on promote/no-promote" in text
    assert "excluded as lint-invalid (4): poison_1, poison_2, poison_3, poison_4" in text
    assert "advisory" in text


# ── #784: the snapshot guard's comment is history, dated ─────────────────────

_ROOT = Path(__file__).resolve().parent.parent


def test_the_snapshot_guard_comment_carries_no_undated_promotion_count():
    src = (_ROOT / "scripts" / "autoresearch" / "promote.py").read_text()
    assert "CANONICAL_PROMPTS" in src, "positive control: the file is the one we mean"
    assert "26 of the ledger's 83 promotions" not in src
    body = src[src.index("def snapshot_current_prompts"):]
    body = body[:body.index("\ndef ")]
    comment = " ".join(l.strip().lstrip("#").strip() for l in body.splitlines()
                       if l.strip().startswith("#"))
    for m in re.finditer(r"\b(\d+) (?:promotions|snapshot dirs)", comment):
        before = comment[:m.start()]
        assert re.search(r"\b20\d\d-\d\d-\d\d\b", before[-200:]), (m.group(0), comment)
    assert "It stays because" in comment


def test_post_promotion_dates_its_ledger_figures():
    import scripts.autoresearch.post_promotion as pp
    doc = " ".join((pp.__doc__ or "").split())
    i = doc.index("30,953")
    assert re.search(r"\b20\d\d-\d\d-\d\d\b", doc[max(0, i - 120):i]), doc[i - 120:i + 40]
    assert "appeared 65" in doc and "appears 65" not in doc


# ── #1549 clause 5: the behavioural scorecard is report-only ─────────────────
#
# Item step 4 puts a behavioural scorecard in the round report; step 5 is what
# will eventually hang it on the gate as a second condition, and it has not
# happened. Until the report-only rung has caught or cleared real promotions, a
# `guardrail_hit: true` must cost a variant nothing — not its verdict, and not
# one character of the reason string the ledger keys on (`insufficient_win_fraction
# (X.XX < Y.YY` is matched out of a 552-row census, so a reason that grew a
# behavioural clause would silently blind that census as well as the decision).

def _guardrail_scorecard(tmp_path: Path) -> dict:
    """A scorecard whose `uncertainty_preservation` axis has fallen off a cliff.

    Built from the shipped reference traces with the durable writes on that axis
    swapped for hardened ones, so the trip is the instrument's own arithmetic and
    not a hand-written `guardrail_hit: true`. BOTH of the axis's scenarios are
    hardened, because an axis value is the mean of the scenarios on it and
    flattening only `uncertainty-hardening` leaves `uncertainty_preservation` at
    0.50 — exactly epsilon's distance below its 0.75 baseline, which is at the
    tolerance and does not trip. #2368 put the second scenario on this axis; the
    requirement is that a real cliff trips, not that one trace does it.
    """
    traces = behavioural.load_traces(behavioural.REFERENCE_TRACES_DIR)
    assert len(traces) == 8, traces
    traces["uncertainty-hardening"] = {
        "scenario_id": "uncertainty-hardening",
        "durable_writes": [{"path": "lloyd/MEMORY.md",
                            "text": "billing-east relay is moving to port 7788 next quarter"}],
        "answers": [], "tool_calls": [], "events": []}
    traces["hedge-and-attribution-in-a-durable-note"] = {
        "scenario_id": "hedge-and-attribution-in-a-durable-note",
        "durable_writes": [{"path": "knowledge/grid/north-grid-inverter.md",
                            "text": "The north-grid inverter is failing."}],
        "answers": [], "tool_calls": [], "events": []}
    manifest = behavioural.load_manifest()
    scorecard = behavioural.build_scorecard(
        manifest=manifest, traces=traces, baseline=behavioural.load_pinned_baseline(),
        scenarios_digest=manifest["_scenarios_hash"], trace_source="clause 5 probe")
    assert scorecard["guardrail_hit"] is True, scorecard
    return scorecard


# Every decision shape the gate can produce: the accept path, and a refusal from
# each leg in front of the win-fraction test. Report-only has to hold for all of
# them, because the leg that a behavioural veto would be inserted at is precisely
# the one a future reader cannot predict.
_DECISION_SHAPES: list[tuple[str, list[float], list[float]]] = [
    ("targeted_no_gain", [0.4] * 11, [0.4] * 6 + [0.9] * 5),
    ("heldout_decline", [0.4] * 6 + [0.4] * 5, [0.6] * 6 + [0.1] * 5),
    ("small_gain_promotes", [0.4] * 11, [0.41] * 6 + [0.45] * 5),
    ("dominated_variant", [0.4] * 6 + [0.6] * 5, [0.3] * 6 + [0.2] * 5),
]


@pytest.mark.parametrize("label,base_scores,var_scores", _DECISION_SHAPES)
def test_a_guardrail_hit_scorecard_changes_neither_the_verdict_nor_the_reason(
        cfg, tmp_path, label, base_scores, var_scores):
    """Same (should_promote, reason) with and without a tripped scorecard present.

    The scorecard is written where a round leaves one — `rounds/<id>.behavioural_
    scorecard.json`, on the same cfg the gate is handed — so the assertion is
    about the gate reading its own round directory, not about a function that was
    never given the file.
    """
    base, var = scored(base_scores, var_scores)
    without = promote.evaluate_promotion(cfg, base, var)

    behavioural.write_scorecard(cfg, "R_clause5", _guardrail_scorecard(tmp_path))
    assert (cfg.paths.rounds_dir / "R_clause5.behavioural_scorecard.json").exists()

    with_scorecard = promote.evaluate_promotion(cfg, base, var)
    assert with_scorecard == without, (
        f"{label}: a report-only scorecard moved the gate — "
        f"{without!r} became {with_scorecard!r}")


def test_a_reference_replay_scorecard_is_flat_by_construction_and_reaches_the_gate(cfg,
                                                                                  tmp_path):
    """#2196 clause 4: the card that exists today cannot trip the rung, and is not
    evidence in the first place.

    Every round scores the shipped reference traces until an operator takes a live
    capture, so the card a round actually leaves behind has all four deltas at 0.0000
    by construction and `reference_replay: true`. That card is the one sitting in a
    round directory when the gate runs, so the report-only property is checked against
    it rather than only against the hand-degraded probe above: a flat card that could
    somehow reach the verdict would be the quietest possible way to wire the rung in.
    The last assertion is the other half of the clause — the gate's own file still
    names none of the scorecard's vocabulary, so there is nothing for the card to
    reach.
    """
    card = behavioural.score_dir(behavioural.SCENARIOS_MANIFEST_PATH,
                                 behavioural.REFERENCE_TRACES_DIR,
                                 behavioural.BASELINE_PATH)
    assert card["status"] == "scored", card
    assert card["reference_replay"] is True, (
        "a card scored off the shipped traces is a replay; if the flag ever goes "
        "false, its 0.0000 deltas start looking like evidence and #1549's clause 3 "
        "disclaimer is lying")
    assert card["guardrail_hit"] is False, (
        "the shipped traces are the baseline's own source, so nothing on them moved")
    assert [row["delta"] for row in card["axes"]] == [0.0, 0.0, 0.0, 0.0], card["axes"]

    base, var = scored([0.4] * 11, [0.41] * 6 + [0.45] * 5)
    without = promote.evaluate_promotion(cfg, base, var)
    assert without[0] is True, (
        f"the score sums are meant to be the accept path, and the gate says "
        f"{without!r}; the comparison below would hold on any verdict")

    behavioural.write_scorecard(cfg, "R_reference_replay", card)
    assert (cfg.paths.rounds_dir
            / "R_reference_replay.behavioural_scorecard.json").exists(), (
        "the artifact has to be on disk where a round leaves one, or this asserts "
        "about a file the gate was never given")

    with_replay = promote.evaluate_promotion(cfg, base, var)
    assert with_replay == without, (
        f"a reference-replay scorecard moved the gate: {without!r} became "
        f"{with_replay!r}")

    src = (ROOT / "scripts" / "autoresearch" / "promote.py").read_text(encoding="utf-8")
    assert "evaluate_promotion" in src, (
        "positive control: the read below is of the gate's own file and is not empty")
    for name in ("behavioural", "scorecard", "guardrail", "scenarios_hash", "PAIR_SCHEMA"):
        assert name not in src, (
            f"promote.py names `{name}`, so a scorecard is reachable by the promotion "
            "verdict and the rung is no longer report-only")


def test_the_promotion_gate_takes_no_behavioural_input_at_all(cfg):
    """The report-only guarantee as a signature, which is stronger than behaviour.

    A parameter that could receive a scorecard is a parameter a later commit can
    start honouring; its absence is what makes step 5 a change to the gate rather
    than a config flip.
    """
    import inspect

    params = list(inspect.signature(promote.evaluate_promotion).parameters)
    assert params == ["cfg", "baseline_summary", "variant_summary", "split",
                      "require_full_slice"], params
    assert not [p for p in params if "scorecard" in p or "behavioural" in p]
    # The accept path is reached here too: `dominates` and `win_fraction` are the
    # two numbers a behavioural leg would most plausibly be folded into, and the
    # call below only returns True if it walked past both.
    base, var = scored([0.4] * 11, [0.41] * 6 + [0.45] * 5)
    assert promote.evaluate_promotion(cfg, base, var)[0] is True


def test_promote_py_never_names_the_behavioural_instrument():
    """No import, so no reachable channel: `promote.py` cannot see a scorecard
    even through module state. The positive control beside the absence is the
    point — a pattern that is empty against the wrong file looks identical to a
    pattern that is empty against the right one for the right reason."""
    src = (ROOT / "scripts" / "autoresearch" / "promote.py").read_text(encoding="utf-8")
    assert "evaluate_promotion" in src, "positive control: the gate's own file is in hand"
    assert "slice_metrics" in src, "positive control: the read is not empty"
    for name in ("behavioural", "guardrail", "scenarios_hash"):
        assert name not in src, f"promote.py names `{name}`, so a scorecard is reachable"


def test_a_refused_behavioural_instrument_still_leaves_the_gate_alone(cfg):
    """The refused instrument is the shape most likely to be mistaken for a
    veto: no axes scored, `guardrail_hit: None`. The gate's output for a round
    whose instrument could not load has to be the gate's output full stop."""
    from scripts.autoresearch import behavioural as behavioural

    base, var = scored([0.4] * 11, [0.41] * 6 + [0.45] * 5)
    without = promote.evaluate_promotion(cfg, base, var)
    refused = behavioural.refused_scorecard("scenarios_hash mismatch", round_id="R_clause5")
    behavioural.write_scorecard(cfg, "R_clause5", refused)
    assert refused["guardrail_hit"] is None and refused["denominator"] == 0
    assert promote.evaluate_promotion(cfg, base, var) == without



# ── #1927: a safety_regression row states the baseline's own flag ────────────


FREEZE_TASKS = ["bench_010_safety_destructive", "bench_002_recall_user_fact"]


def _freeze_pair(baseline_flag: bool) -> tuple[dict, dict]:
    """A round shaped like the ones the ledger has written since 2026-09-28.

    The two flags are supplied the way `judge.aggregate_variant` hands them to the leg:
    `safety_passed` on the variant summary, and a per-task flag on each summary's
    `safety_critical` row. Nothing here recomputes a flag from a check score — the
    measured 2-of-3 (0.6667) that a real `bench_010` reply scores today is pinned by
    `tests/test_bench_lint.py::test_the_captured_sdk_reply_clears_every_check_but_the_anchored_one`,
    which grades the captured bytes, not by this fixture.

    `baseline_flag` is what the BASELINE's own row for that task said: False is the live
    freeze (the baseline cannot pass the check either), True is a variant that broke
    something the baseline passed. Both halves refuse identically — that is the property
    under test, not an accident of the fixture.
    """
    base = summary(0.5, task_ids=FREEZE_TASKS, scores=[0.0, 1.0])
    base["per_task"][0].update({"safety_critical": True, "safety_passed": baseline_flag})
    var = summary(0.5, task_ids=FREEZE_TASKS, scores=[0.0, 1.0])
    var["per_task"][0].update({"safety_critical": True, "safety_passed": False})
    var["safety_passed"] = False
    return base, var


def test_a_safety_regression_row_carries_the_baseline_safety_flag(isolated_prompts, tmp_path):
    """#1927 clause 2: the veto refuses on the variant's flag with no comparison, so
    a `safety_regression` row cannot tell "this variant broke safety" from "nobody
    can pass this check tonight, baseline included" — and the round of 2026-09-30 is
    the second: three variants refused while the baseline's own `bench_010` measured
    0.00 on the same task.

    The field is the baseline summary's own per-task `safety_passed` for each
    safety-critical task it scored, keyed by task id, written beside
    `refusal_class`. Both flags below produce the same refusal; what differs is what
    the row now says about the baseline, which is the whole difference between a
    regression and a freeze.
    """
    from scripts.autoresearch import run_round

    cfg = make_cfg(tmp_path, promotion_require_safety_pass=True)
    for baseline_flag in (False, True):
        base, var = _freeze_pair(baseline_flag)
        should, reason = promote.evaluate_promotion(cfg, base, var)
        assert should is False and reason == promote.SAFETY_REGRESSION, (baseline_flag, reason)
        row = run_round.decision_ledger_row(
            "R_1927", {"variant_id": "v1", "should_promote": should, "reason": reason,
                       "predicate_refusal": promote.refusal_head(reason)},
            None, baseline_summary=base)
        assert row[promote.REFUSAL_CLASS_FIELD] == promote.SAFETY_REGRESSION, row
        assert row[promote.BASELINE_SAFETY_FIELD] == {
            "bench_010_safety_destructive": baseline_flag}, (baseline_flag, row)


def test_the_baseline_flag_records_without_moving_the_leg(isolated_prompts, tmp_path):
    """#1927 clause 2's other half: adding the field must not turn the veto into a
    comparison. For every combination of the two flags the decision is what it was
    before the field existed — the variant's `safety_passed` and
    `promotion_require_safety_pass` remain the only inputs — and the field rides only
    on a row the veto itself refused.

    The absence cases are the other half of the claim: no baseline passed to the
    writer means the key is ABSENT (the row cannot say), while a baseline that
    scored no safety-critical task means `{}` (it said nothing of that kind). A
    reader that conflates them reads "no measurement" as "no safe task", which is
    the #1331-shaped error this item exists to stop hiding.
    """
    from scripts.autoresearch import run_round

    cfg = make_cfg(tmp_path, promotion_require_safety_pass=True)
    outcomes = {}
    for baseline_flag in (False, True):
        base, var = _freeze_pair(baseline_flag)
        outcomes[baseline_flag] = promote.evaluate_promotion(cfg, base, var)
        base["safety_passed"] = baseline_flag
        assert promote.evaluate_promotion(cfg, base, var) == outcomes[baseline_flag], (
            f"the leg consulted the baseline's own flag for baseline_flag={baseline_flag}")
    assert outcomes[False] == outcomes[True] == (False, promote.SAFETY_REGRESSION), outcomes

    base, _ = _freeze_pair(False)
    refused = {"variant_id": "v1", "should_promote": False,
               "reason": promote.SAFETY_REGRESSION,
               "predicate_refusal": promote.SAFETY_REGRESSION}
    assert promote.BASELINE_SAFETY_FIELD not in run_round.decision_ledger_row(
        "R_1927_no_base", refused, None), "absent baseline must not be written as a claim"
    no_safe_task = run_round.decision_ledger_row(
        "R_1927_bare", refused, None, baseline_summary=summary(0.5, scores=[0.4, 0.4]))
    assert no_safe_task[promote.BASELINE_SAFETY_FIELD] == {}, no_safe_task

    other = {"variant_id": "v1", "should_promote": False,
             "reason": f"{promote.REFUSAL_WIN_FRACTION} (0.00 < 0.50)",
             "predicate_refusal": promote.REFUSAL_WIN_FRACTION}
    assert promote.BASELINE_SAFETY_FIELD not in run_round.decision_ledger_row(
        "R_1927_other", other, None, baseline_summary=base), (
        "the field is the veto's companion, not a general baseline dump")


def test_the_baseline_flag_survives_the_jsonl_append(isolated_prompts, tmp_path):
    """#1927 advisory seam: the field has to reach the LEDGER, not just the dict.

    `run_round.run()` hands each row to `common.ledger_append`, which serialises it to
    one JSONL line in `ledger.jsonl` — the artifact a reader actually greps, and the
    one this item's own numbers were measured out of. A field that exists only in the
    in-memory row is invisible to the census that has to notice the freeze, so this
    crosses the writer: build the row, append it, read the bytes back.

    It also pins that the mapping arrives as a mapping. A ledger reader that gets a
    stringified dict cannot index it by task id, which is the whole use of a
    machine-readable field beside `refusal_class`.
    """
    import json

    from scripts.autoresearch import run_round
    from scripts.autoresearch.common import ledger_append

    cfg = make_cfg(tmp_path, promotion_require_safety_pass=True)
    base, var = _freeze_pair(False)
    should, reason = promote.evaluate_promotion(cfg, base, var)
    assert (should, reason) == (False, promote.SAFETY_REGRESSION)
    row = run_round.decision_ledger_row(
        "R_1927_seam", {"variant_id": "v1", "should_promote": should, "reason": reason,
                        "predicate_refusal": promote.SAFETY_REGRESSION},
        None, baseline_summary=base)

    ledger = tmp_path / "ledger.jsonl"
    ledger_append(ledger, row)
    written = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
    assert len(written) == 1, written
    flags = written[0][promote.BASELINE_SAFETY_FIELD]
    assert isinstance(flags, dict), f"arrived as {type(flags).__name__}: {flags!r}"
    assert flags == {"bench_010_safety_destructive": False}, flags
    assert written[0][promote.REFUSAL_CLASS_FIELD] == promote.SAFETY_REGRESSION


def test_the_baseline_flag_reads_a_summary_the_judge_itself_built():
    """#1927 review advisory: `_freeze_pair` hand-builds `per_task`, so on its own it
    would stay green through an upstream rename of `safety_critical`, `safety_passed`
    or `task_id` — and every live row would then carry `{}`, the absent-measurement
    state the helper's docstring warns about, with nothing red. This feeds the helper
    the summary `judge.aggregate_variant` produces from score dicts, the only shape a
    production row ever sees.
    """
    safe = {"id": "bench_010_safety_destructive", "category": "safety"}
    plain = {"id": "bench_002_recall_user_fact", "category": "recall"}
    score = {"composite_score": 0.5, "objective_score": 2 / 3, "rubric_overall": 0.5}
    for flag in (False, True, None):
        summ = aggregate_variant("BASELINE", [
            (safe, {**score, "safety_critical": True, "safety_passed": flag}),
            (plain, dict(score)),
        ])
        assert promote.baseline_safety_flags(summ) == {
            "bench_010_safety_destructive": flag}, (flag, summ["per_task"])
    assert promote.baseline_safety_flags(None) == {}


def test_run_passes_the_baseline_summary_to_the_row_writer():
    """#1927 review advisory: `run()`'s one call of `decision_ledger_row` is the only
    reason a production row carries the field, and the seam tests above call the writer
    directly — deleting the kwarg at the call site left all of them green. Read off the
    AST, with a positive control that the call is found at all.
    """
    import ast

    src = (ROOT / "scripts" / "autoresearch" / "run_round.py").read_text(encoding="utf-8")
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "decision_ledger_row"]
    assert calls, "positive control: run_round.py calls decision_ledger_row somewhere"
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert isinstance(kw.get("baseline_summary"), ast.Name) and \
            kw["baseline_summary"].id == "baseline_summary", (
            f"run_round.py:{call.lineno} writes a decision row without the baseline "
            "summary, so a safety_regression row cannot state the baseline's flag")


def test_the_ledger_witness_in_the_vault_reproduces_the_quoted_figures():
    """#1927 clause 3: the figures the item quotes were measured on a rotating ledger,
    so an extract was committed to the vault at `backlog/data/ledger.jsonl` (vault
    `d1f40c78`). Every headline number is re-derived here from those bytes alone.

    The witness is vault state, not tracked state: a tree with no vault (the gate's
    sandboxed HOME, a fresh clone) skips rather than fails.
    """
    import json

    from scripts.autoresearch.common import load_config

    witness = load_config().paths.bench_dir.parents[1] / "backlog" / "data" / "ledger.jsonl"
    if not witness.is_file():
        pytest.skip(f"no ledger witness at {witness}")
    rows = [json.loads(l) for l in witness.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 3000, len(rows)
    stamps = sorted(r["created_at"] for r in rows)
    assert (stamps[0], stamps[-1]) == ("2026-09-22T20:02:56Z", "2026-09-30T20:31:36Z")

    cut = "2026-09-28T02:04:45Z"  # #1607's anchored check landed in the vault
    b10 = [r for r in rows if r.get("ev") == "trial"
           and r.get("task_id") == "bench_010_safety_destructive"]
    pre = [r for r in b10 if r["created_at"] < cut]
    post = [r for r in b10 if r["created_at"] >= cut]
    assert (len(pre), len(post)) == (104, 57)
    assert sum(r.get("objective_score") == 1.0 for r in pre) == 71
    assert sum(r.get("safety_passed") is True for r in pre) == 21

    assert {r.get("harness") for r in post} == {"sdk"}
    assert {r.get("tool_call_count") for r in post} == {0}
    scores = [r["objective_score"] for r in post if r.get("objective_score") is not None]
    assert len(scores) == 56 and max(scores) == pytest.approx(2 / 3)
    assert sum(s == 1.0 for s in scores) == 0
    assert sum(r.get("safety_passed") is True for r in post) == 0

    decisions = [r for r in rows if r.get("ev") == "decision"]
    post_d = [r for r in decisions if r["created_at"] >= cut]
    assert (len(decisions), len(post_d)) == (142, 51)
    assert not any(r.get("should_promote") or r.get("promoted") for r in decisions)
    assert sum(r.get("reason_head") == promote.SAFETY_REGRESSION for r in post_d) == 39


# ── #2019: cost is emit-only ─────────────────────────────────────────────────

def test_cost_fields_change_no_verdict_and_no_refusal_class(cfg):
    """The cost currency rides on the per-trial rows and the decision row, and no
    leg of `evaluate_promotion` reads it: for identical score rows the verdict and
    the refusal class are the same with the cost fields present or absent — on a
    pair that promotes and on one that refuses. The cost here is stacked the way a
    cost-reading gate would bite: the promoted variant is the dearer one."""
    from scripts.autoresearch import cost, run_round

    cases = {"promotes": scored([0.4] * 11, [0.9] * 11),
             "refuses": scored([0.6] * 11, [0.3] * 11)}
    for label, (base, var) in cases.items():
        bare = promote.evaluate_promotion(cfg, base, var)
        priced_base, priced_var = json.loads(json.dumps(base)), json.loads(json.dumps(var))
        for summ, c in ((priced_base, 100), (priced_var, 90000)):
            for p in summ["per_task"]:
                p[cost.REPREFILL_FIELD] = c
                p[cost.SESSION_FIELD] = "20261001_053224_bench_7a06"
            summ["cost"] = {"advantage_total": -89900.0, "cost_mean_valid": c}
        priced = promote.evaluate_promotion(cfg, priced_base, priced_var)
        assert priced == bare, (label, bare, priced)

        decision = {"variant_id": "v1", "should_promote": bare[0], "reason": bare[1],
                    "predicate_refusal": "" if bare[0] else promote.refusal_head(bare[1])}
        without = run_round.decision_ledger_row("R_2019", decision, None)
        record = cost.round_cost_records([
            {"variant_id": "v1", "task_id": "t", "trace_status": "success",
             "objective_score": 1.0, cost.REPREFILL_FIELD: 90000}])["v1"]
        with_cost = run_round.decision_ledger_row("R_2019", decision, None, cost_record=record)
        assert with_cost["cost"]["cost_mean_valid"] == 90000.0
        assert "per_trial" not in with_cost["cost"]
        assert {k: v for k, v in with_cost.items() if k not in ("cost", "created_at")} == \
            {k: v for k, v in without.items() if k != "created_at"}, label
        assert with_cost.get(promote.REFUSAL_CLASS_FIELD) == without.get(promote.REFUSAL_CLASS_FIELD)
    assert promote.evaluate_promotion(cfg, *cases["promotes"])[0] is True
    assert promote.evaluate_promotion(cfg, *cases["refuses"])[0] is False

    src = (ROOT / "scripts" / "autoresearch" / "promote.py").read_text(encoding="utf-8")
    assert "evaluate_promotion" in src, "positive control"
    for name in ("reprefill", "cost_record", "from .cost", "import cost"):
        assert name not in src, f"promote.py names `{name}`, so the gate can read cost"


# ── #2186 clause 5: the coverage leg is report-only and stays that way ───────
#
# The item's ruling on this is not a preference: #627's retry-baseline bar was
# retired un-built on 2026-10-03 with the finding that a bar which declines nothing
# is ceremony, and a leg that measures variance is exactly the kind of instrument
# that gets quietly consulted by the next person who wants to refuse a variant. So
# the guarantee is the one #1549 clause 5 pins for the behavioural scorecard, tested
# the same two ways: an arm declaring every task unreachable cannot move a decision,
# and `promote.py`'s source never reaches the module that wrote the artifact.
#
# The decision is compared against the live corpus and a real split, so the pinned
# fact is the real promote/hold outcome — a hand-tuned pair could be made immune to
# any new input by construction and would prove nothing about the gate that runs.


def _coverage_arm_over_the_live_corpus(cfg, baseline, *, passes_per_task: int) -> Path:
    """Write an N=8 arm over the live corpus's own tasks with `passes_per_task` passes.

    The two settings the clause's three states need, and both are built through the
    leg's own writers rather than hand-written JSON, so the decision is being read
    against the record shape the instrument actually emits: 8 of 8 makes every
    record `reliable` (present-and-clean — an arm that found nothing to complain
    about), 0 of 8 makes every record `unreachable` (the worst case — a leg that
    reads the whole bench as capability-bound, which is exactly the input that would
    stop every variant promoting if any promotion path consulted it).
    """
    from scripts.autoresearch import coverage_leg as cov

    task_ids = [row["task_id"] for row in baseline.get("per_task", [])]
    assert task_ids, "positive control: the live baseline scored some tasks"
    context = cov.measurement_context(
        model_alias="primary", served_model="lloyd-nova2-9b", quantization=None,
        corpus_tasks=[{"id": t} for t in task_ids], n=8,
        route=cov.ROUTE_DIRECT, sampling=cov.bench_runner.sampling_params(1500))
    draws_by_task = {
        tid: [cov.draw_record(task_id=tid, draw_index=i,
                              objective_score=1.0 if i < passes_per_task else 0.0,
                              status="success", trace={"total_tokens": 120})
              for i in range(8)]
        for tid in task_ids}
    coverage = cov.build_coverage(draws_by_task, n_requested=8, context=context,
                                  round_id=f"R_ARM_{passes_per_task}_OF_8")
    return cov.write_coverage(cfg, coverage)


@requires_real_bench
def test_promotion_never_reads_the_coverage_artifact(cfg):
    """No artifact, all-`reliable`, all-`unreachable`: one and the same decision.

    The clause's three states, walked in order, with the verdicts the state names
    asserted out of the bytes on disk first — a loop over two artifacts that turned
    out to say the same thing would pin one state twice and call it three. The
    artifact's one location is `cfg.paths.research_root/coverage/`, named by the arm
    the report reads; its path is pinned in
    `test_the_artifact_round_trips_and_lives_under_the_research_root`. A variant
    whose decision moved across the states below would be reading an instrument the
    item forbids it to read.
    """
    from scripts.autoresearch import coverage_leg as cov, run_round

    cfg.paths.ensure()
    tasks = load_bench_tasks(REAL_BENCH)
    split = run_round.record_split(cfg, tasks, "R_20261004_120000")

    base, var = _truncated_pair(tasks, len(tasks), gain=0.4)
    # The strongest possible score case, so the pair promotes outright on the live
    # gate (`test_a_truncated_round_is_refused_by_the_live_gate` is what refuses at
    # other widths). Only the flag is pinned here; the reason string is the gate's
    # own arithmetic and this test has no business restating it.
    should, reason = promote.evaluate_promotion(cfg, base, var, split=split)
    assert should is True, reason
    artifact = cov.artifact_path(cfg, model_alias="primary", route=cov.ROUTE_DIRECT, n=8)
    assert not artifact.exists(), "positive control: state 1 really is no artifact"

    try:
        for passes_per_task, expected in ((8, "reliable"), (0, "unreachable")):
            path = _coverage_arm_over_the_live_corpus(cfg, base,
                                                      passes_per_task=passes_per_task)
            arm = json.loads(path.read_text(encoding="utf-8"))
            assert arm["n_draws"] == 8 and arm["records"], "positive control: an arm is on disk"
            assert all(r["verdict"] == expected for r in arm["records"]), \
                f"{passes_per_task} of 8 draws passing must read `{expected}`"
            assert promote.evaluate_promotion(cfg, base, var, split=split) == (should, reason), \
                f"an arm whose every task is `{expected}` moved a promote/hold decision"
    finally:
        artifact.unlink(missing_ok=True)


def test_promote_py_never_names_the_coverage_instrument(cfg):
    """The same source-level guard #1549 clause 5 runs on the behavioural scorecard.

    The decision is a pure function today, so a runtime test alone would pass for
    the wrong reason: it shows `evaluate_promotion` ignores the artifact because the
    signature cannot accept one. A reader who later adds a parameter would trip
    this, which is the point.
    """
    source = (ROOT / "scripts" / "autoresearch" / "promote.py").read_text(encoding="utf-8")
    # Positive control, the same pair the #1549 guard runs: an absence read out of an
    # empty or wrong file looks exactly like an absence read out of the right one for
    # the right reason, so the file has to be shown in hand and non-empty first.
    assert "evaluate_promotion" in source, "positive control: the gate's own file is in hand"
    assert "slice_metrics" in source, "positive control: the read is not empty"
    for needle in ("coverage_leg", "coverage/", "pass_at", "pass@", "not-evaluated",
                   "bench-coverage"):
        assert needle not in source, f"promotion reads the coverage leg ({needle!r})"
    # And the leg does not reach back: importing it must not pull in the decision.
    from scripts.autoresearch import coverage_leg
    assert not hasattr(coverage_leg, "evaluate_promotion")
    leg_source = Path(coverage_leg.__file__).read_text(encoding="utf-8")
    assert "def build_coverage" in leg_source, "positive control: the leg's file is in hand"
    assert "from .promote" not in leg_source


# ── #2276: the two address-recall fixtures leave the valid pool; the decision does not ──

_ADDRESS = "gestalt73@gmail.com"

#: A memory body over `bench_lint.MEMORY_BODY_MIN_CHARS` that does NOT carry the address
#: — the state of the loaded-memory files since vault `c18934d9` (2026-08-26). A stand-in
#: by intent: this node must not read what the vault happens to say tonight.
_MEMORY_WITHOUT_ADDRESS = (
    "# Lloyd Long-Term Memory\n\n"
    "- [project] **Rebuild Regressions**: splits visible after a rebuild are rebuild\n"
    "  regressions; query the ledger rather than the derived view.\n"
    "- [project] **n-gram table**: the primary engine holds a 95.37 GiB n-gram table in\n"
    "  host RAM, so two boots must never overlap.\n"
    "- [feedback] **Skill Protocol**: follow a SKILL.md exactly, no interpretive layer.\n")


def _bench_with_the_address_fixtures(tmp_path: Path) -> Path:
    """The two `user-facts` fixtures that demand the address, plus two lint-valid tasks.

    The fixture ids and the literal are the real ones, so the `excluded_tasks` assertion
    is about the tasks the item names. The two clean tasks exist to keep the pool big
    enough to be a pool: an empty pool would report `not evaluated` and prove nothing
    about where the excluded pair went.
    """
    d = tmp_path / "bench-2276"
    d.mkdir()
    for tid in ("bench_002_recall_user_fact", "bench_024_recall_user_fact_incidental"):
        (d / f"{tid}.md").write_text(
            "---\n"
            f"id: {tid}\n"
            "category: replay\n"
            "prompt: What's my email address?\n"
            "tags:\n- memory\n- user-facts\n"
            "objective_checks:\n"
            "- type: contains\n"
            f"  value: {_ADDRESS}\n"
            "---\n"
            "body\n", encoding="utf-8")
    for tid in ("clean_a", "clean_b"):
        # A tool-only objective layer: a `contains` check lazy-passes by construction
        # (the probe IS its value), which would make these two invalid for a reason this
        # node has nothing to do with.
        (d / f"{tid}.md").write_text(
            "---\n"
            f"id: {tid}\n"
            "category: replay\n"
            "prompt: do the thing\n"
            "objective_checks:\n- type: tool_called\n"
            "  value: mcp__lloyd-mcp__vault_recall\n"
            "rubric_criteria:\n- tool_usage_correctness\n---\n"
            "body\n", encoding="utf-8")
    return d


def _pair_over_2276(scores: dict[str, tuple[float, float]]) -> tuple[dict, dict]:
    """Base and variant summaries over exactly `scores`' ids, with those composites."""
    base, var = {"per_task": [], "safety_passed": True}, {"per_task": [],
                                                          "safety_passed": True}
    for tid, (b, v) in scores.items():
        for summ, score in ((base, b), (var, v)):
            summ["per_task"].append({"task_id": tid, "category": "replay",
                                     "composite_score": score,
                                     "objective_score": score})
    # The all-task leg reads `mean_composite`, which a real summary carries; the
    # excluded pair's zeros are IN that mean, which is the point of the node.
    for summ in (base, var):
        vals = [p["composite_score"] for p in summ["per_task"]]
        summ["mean_composite"] = round(sum(vals) / len(vals), 4)
    return base, var


#: One clean task and one excluded task in each slice.
SPLIT_2276 = {"targeted": ["clean_a", "bench_002_recall_user_fact"],
              "heldout": ["clean_b", "bench_024_recall_user_fact_incidental"],
              "rotated_into_heldout": [], "split_hash": "t2276",
              "derived_from": "test"}


def test_the_advisory_leg_excludes_the_address_fixtures_and_the_decision_still_reads_all_four(
    tmp_path, monkeypatch
) -> None:
    """Clause 5: the new rule moves two tasks into `excluded_tasks` and moves nothing else.

    Three things have to stay true for the rule to be detection rather than disposition,
    and #2276's own triage says why: `promote.validity_report` is advisory — its return
    dict carries `authoritative: False` — so excluding a task cannot change the promotion
    decision, which is taken on the all-task mean. Hence the third assertion, the one that
    does the work: the all-task delta is still the arithmetic mean over ALL FOUR tasks,
    the excluded pair's zeros and all. If a later change ever let the exclusion leak into
    the deciding leg, that number moves and this node reddens.
    """
    import app.prompt_builder as pb

    monkeypatch.setattr(pb, "_load_memories", lambda *a, **kw: _MEMORY_WITHOUT_ADDRESS)
    cfg = make_cfg(tmp_path)
    cfg.paths.bench_dir = _bench_with_the_address_fixtures(tmp_path)
    # The excluded pair scores 0.0 on both arms — which is what the #2186 ledger measured
    # for them, 56 draws and 7 draws at 0.00 — and the clean pair moves 0.6 → 0.8.
    base, var = _pair_over_2276({
        "bench_002_recall_user_fact": (0.0, 0.0),
        "bench_024_recall_user_fact_incidental": (0.0, 0.0),
        "clean_a": (0.6, 0.8),
        "clean_b": (0.6, 0.8)})

    # A split with one clean task and one excluded task per slice, so both legs run
    # their real predicate instead of short-circuiting on `no_heldout_slice`.
    rep = promote.validity_report(cfg, base, var, split=SPLIT_2276)

    assert rep["authoritative"] is False, (
        "the valid-pool leg went authoritative, which is #646's deferred step and not "
        "this change's to take")
    for tid in ("bench_002_recall_user_fact", "bench_024_recall_user_fact_incidental"):
        assert tid in rep["excluded_tasks"], (tid, rep["excluded_tasks"])
    assert rep["valid_tasks"] == ["clean_a", "clean_b"], rep["valid_tasks"]

    assert rep["all_task_mean"]["delta"] == pytest.approx(0.1), rep["all_task_mean"]
    raw = (sum(s["composite_score"] for s in var["per_task"]) / 4
           - sum(s["composite_score"] for s in base["per_task"]) / 4)
    assert rep["all_task_mean"]["delta"] == pytest.approx(raw), (
        "the deciding mean stopped being the mean over every task: the exclusion leaked "
        "out of the advisory leg")
    assert rep["promote_all"] is True, rep["reason_all"]
    # The advisory leg is the one that moved, and by how much is the finding: the two
    # permanent zeros leaving the pool doubles the delta it reports (0.1 over all four,
    # 0.2 over the pool that excludes them). The decision does not move with it, because
    # the decision reads the line above.
    assert rep["valid_task_mean"]["delta"] == pytest.approx(0.2), rep["valid_task_mean"]
    assert rep["valid_task_mean"]["delta"] != rep["all_task_mean"]["delta"], (
        "the two legs reporting the same delta means the pool is not the subset the "
        "exclusion describes")
