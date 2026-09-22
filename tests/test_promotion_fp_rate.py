"""The autoresearch promotion false-positive measurement — item #428.

Why this file exists
--------------------
``min_bench_win_fraction`` has been ``0.5`` since 2026-05-23 and 65 promotions had
landed, but the false-positive rate had never been measured, so the keep / raise /
revert decision that backlog #352 and #367 existed to make could not be made. The
measurement lives in :mod:`scripts.autoresearch.promotion_fp_rate` and its result is
published in
``~/obsidian/knowledge/evaluation/autoresearch-promotion-fp-rate-at-threshold-0p5.md``.

A published number is only worth as much as the check that it holds. This file is
that check, in two layers:

  * Synthetic fixtures (the bulk of the file) prove the machinery can report a
    **non-zero** false-positive rate and can report zero for the right reason. The
    headline result is ``0/65``; a measurement that returns ``0`` for a real
    regression would be worthless, and "it returned 0" is not by itself evidence
    that the rule was ever applied.
  * The two tests at the bottom re-run the derivation over the frozen ledger window
    and assert the published figures, including every field of the note's
    machine-checked block.

Isolation
---------
Everything above the live section works on ``tmp_path`` fixtures and touches no
live store. The live section reads ``_pipeline/research/`` read-only, anchored to a
declared cutoff round so rounds that accrue after the promotion loop is re-armed
(#506) cannot change the numbers. The anchor is only real because ``measure`` cuts
*every* counted quantity at it — which it did not: three nodes here pinned counts taken
over the whole live file, the worker appended a round six minutes after the last
promotion landed, and because the store is reached through an absolute path a base
worktree reproduced the red, the gate classified every promotion as externally blocked
and refused it (#1193). Nothing in this file asserts a whole-file count any more: the
invariant is pinned on synthetic data by
``test_appending_a_later_round_changes_no_counted_field_at_the_declared_cutoff`` and
against a copy of the live store by
``test_the_published_recipe_survives_a_round_appended_to_a_copy_of_the_live_store``,
and the note's shell reproduce block is re-run and re-checked by
``test_the_note_s_reproduce_block_commands_agree_with_the_numbers_beside_them``. No test in this file carries the ``live_vault``
marker — including the two that open the published note: reading it is the
repo-to-vault seam this file exists to guard, so it runs on the graded gate pass
(see ``test_note_fields_are_reproduced_by_a_fresh_derivation`` for the argument).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.autoresearch import promote
from scripts.autoresearch import promotion_fp_rate as pfr
# `make_cfg` is the promotion-gate fixture factory; importing it rather than
# rebuilding an AutoresearchConfig here keeps the two files' idea of "the live
# spec" in one place. Cross-test-module imports are already how that file gets
# its contract fixture (`from tests.test_prompt_surface_guard import
# GOOD_CONTRACT`, four times, inside test bodies).
from tests.test_autoresearch_promotion import make_cfg

# ── fixtures ─────────────────────────────────────────────────────────────────


def write_round(rounds_dir: Path, rid: str, baseline_mean: float, promoted: dict[str, tuple[float, float]] = None):
    """Write ``rounds/R_<rid>.md`` in the shape ``run_round.py`` produces.

    ``promoted[variant_id] = (mean, delta)`` — the mean lands in the variant
    summary, the delta in the PROMOTE line, because that is what the real report
    carries and what :func:`promotion_fp_rate.recorded_deltas` reads back.
    """
    rounds_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# Research round {rid}", "", "## Summary", f"- baseline mean composite: {baseline_mean:.4f}", ""]
    for vid, (mean, _delta) in (promoted or {}).items():
        lines.append(f"- `{vid}`: mean={mean:.4f}")
    lines += ["", "## Promotion decisions"]
    for vid, (_mean, delta) in (promoted or {}).items():
        sign = "+" if delta >= 0 else ""
        lines.append(f"- `{vid}`: PROMOTE — promote (delta={sign}{delta:.4f}, win_frac=0.60)")
    (rounds_dir / f"{rid}.md").write_text("\n".join(lines) + "\n")


def write_ledger(ledger_path: Path, rounds: dict[str, dict[str, dict[str, float]]], decisions):
    """`rounds[rid][variant_id][task_id] = composite_score`, plus decision rows."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("w") as fh:
        for rid, variants in rounds.items():
            for vid, tasks in variants.items():
                for task_id, score in tasks.items():
                    fh.write(
                        json.dumps(
                            {
                                "round_id": rid,
                                "variant_id": vid,
                                "task_id": task_id,
                                "composite_score": score,
                                "rubric_overall": score,
                                "objective_score": score,
                                "safety_passed": True,
                                "safety_critical": False,
                                "task_category": "bench",
                                "trace_status": "traced",
                                "turns": 1,
                                "tool_call_count": 0,
                                "duration_seconds": 1.0,
                                "created_at": "2026-01-01T00:00:00Z",
                            }
                        )
                        + "\n"
                    )
        for row in decisions:
            fh.write(json.dumps({"created_at": "2026-01-01T00:00:00Z", **row}) + "\n")


def score(rid: str, seed: float) -> dict[str, float]:
    return {f"task_{i}": round(seed + i * 0.01, 4) for i in range(2)}


# ── the post-#549 report line, composed rather than transcribed ──────────────
#
# Eleven tasks in the live bench's proportions and categories: six in the
# targeted categories (replay/synthetic), five in the veto categories
# (adversarial/safety) — the same axis `bench_split` rotates, and the axis
# `promote.derive_split` reads off the per-task `category` field.
SPLIT_TASKS: list[dict[str, str]] = [
    {"id": f"task_{i}", "category": cat}
    for i, cat in enumerate(["replay", "synthetic"] * 3
                            + ["adversarial", "safety", "adversarial", "safety", "adversarial"])
]


def bench_summary(scores: dict[str, float]) -> dict:
    """A bench summary in the shape ``judge.aggregate_variant`` returns."""
    cat = {t["id"]: t["category"] for t in SPLIT_TASKS}
    per = [{"task_id": tid, "composite_score": sc, "category": cat[tid]}
           for tid, sc in scores.items()]
    return {"mean_composite": sum(scores.values()) / len(scores),
            "safety_passed": True, "task_count": len(per), "per_task": per}


def slice_scores(targeted: float, heldout: float) -> dict[str, float]:
    """Every task at the score its slice earned: ``targeted`` for the six tasks
    the proposer could aim at, ``heldout`` for the five it never saw named."""
    return {t["id"]: (targeted if t["category"] in ("replay", "synthetic") else heldout)
            for t in SPLIT_TASKS}


# Which pool a fixture task belongs to — the same category axis `slice_scores` and
# `promote.derive_split` read, spelled once so a test that moves one slice cannot
# quietly move the other.
_TARGETED = {t["id"] for t in SPLIT_TASKS if t["category"] in ("replay", "synthetic")}
_HELDOUT = {t["id"] for t in SPLIT_TASKS if t["category"] not in ("replay", "synthetic")}


def _is_targeted(task_id: str) -> bool:
    return task_id in _TARGETED


def _is_heldout(task_id: str) -> bool:
    return task_id in _HELDOUT


def promote_line(vid: str, cfg, base: dict[str, float], var: dict[str, float]) -> str:
    """The exact ``- \\`vid\\`: PROMOTE — …`` line ``run_round.run()`` writes.

    Built by calling :func:`promote.evaluate_promotion` — the writer that produces
    the reason string the report interpolates — rather than typed out here. A
    transcription is how the previous version of this fixture came to pin a shape
    the writer never emitted (``promote (delta=+0.3833)`` followed by
    ``targeted_delta=…``: two different spellings of the same quantity, in one
    line, from two different eras of the code), and the reader under test then
    matched the *legacy* field and the test passed on a fiction.
    """
    should, reason = promote.evaluate_promotion(cfg, bench_summary(base), bench_summary(var),
                                                split=None)
    assert should, f"fixture is not promotable under the gate: {reason}"
    return f"- `{vid}`: PROMOTE — {reason}"


@pytest.fixture
def world(tmp_path):
    """Twelve rounds: three promoted, one gate-passed-but-unlanded control, the rest null.

    The baseline means wobble by 0.01 through round 09, then collapse to 0.30 for the
    last three. So ``R_..._091012``'s promotion is followed by a 0.30 drop while the
    ordinary rounds around it stay inside 0.01 — the fixture can therefore produce a
    real false positive without widening the noise floor, which is the whole point of
    the floor.
    """
    root = tmp_path / "research"
    rounds_dir, ledger = root / "rounds", root / "ledger.jsonl"
    plan = [
        ("R_20260101_000000", 0.60, None),
        ("R_20260102_000000", 0.61, None),
        ("R_20260103_000000", 0.60, None),
        ("R_20260104_000000", 0.60, ("V_win_a", 0.10)),  # promoted; later rounds keep running hot
        ("R_20260105_000000", 0.61, None),
        ("R_20260106_000000", 0.60, ("V_win_b", 0.08)),  # promoted; the floor is never cleared
        ("R_20260107_000000", 0.61, None),
        ("R_20260107_120000", 0.61, None),  # gate passed, NOT landed: the matched control
        ("R_20260108_000000", 0.61, None),
        ("R_20260109_000000", 0.60, ("V_crash_c", 0.09)),  # promoted, then everything collapses
        ("R_20260110_000000", 0.30, None),
        ("R_20260111_000000", 0.30, None),
        ("R_20260112_000000", 0.30, None),
    ]
    control_round = "R_20260107_120000"
    evals, decisions = {}, []
    for rid, mean, winner in plan:
        write_round(
            rounds_dir,
            rid,
            mean,
            {winner[0]: (mean + winner[1], winner[1])} if winner else None,
        )
        evals[rid] = {"BASELINE_V": score(rid, mean)}
        if rid == control_round:
            evals[rid]["V_dry"] = score(rid, mean + 0.10)
            decisions.append(
                {
                    "round_id": rid,
                    "event": "decision",
                    "variant_id": "V_dry",
                    "should_promote": True,
                    "promoted": False,
                    "reason": "promote (dry run)",
                }
            )
        elif winner:
            evals[rid][winner[0]] = score(rid, mean + winner[1])
            decisions.append(
                {
                    "round_id": rid,
                    "event": "decision",
                    "variant_id": winner[0],
                    "should_promote": True,
                    "promoted": True,
                    "reason": "promote",
                }
            )
        else:
            decisions.append(
                {
                    "round_id": rid,
                    "event": "decision",
                    "variant_id": "V_other",
                    "should_promote": False,
                    "promoted": False,
                    "reason": "insufficient_delta (+0.0100 < 0.05)",
                }
            )
    # A non-decision row that must NOT enter the denominator.
    decisions.append({"round_id": "R_99991231_000000", "event": "round", "promoted": True, "spec_path": "x"})
    write_ledger(ledger, evals, decisions)
    return {"ledger": ledger, "rounds": rounds_dir, "control_round": control_round}


def measure_world(world, **over):
    return pfr.measure(
        ledger_path=world["ledger"],
        rounds_dir=world["rounds"],
        data_cutoff=over.pop("data_cutoff", "R_20260112_000000"),
        **over,
    )


# ── the denominator ──────────────────────────────────────────────────────────


def test_denominator_is_promoted_decision_rounds(world):
    """``event: "decision"`` AND ``promoted`` — nothing else, and de-duplicated."""
    result = measure_world(world)
    assert result["denominator"] == 3
    assert {r["round_id"] for r in result["rounds"]} == {
        "R_20260104_000000",
        "R_20260106_000000",
        "R_20260109_000000",
    }
    # the dry-run round is the control group, not part of the measured set
    assert world["control_round"] not in [r["round_id"] for r in result["rounds"]]
    assert result["unlanded_gate_passes"]["rows"] == 1


def test_a_promoted_flag_on_a_non_decision_row_is_not_counted(world):
    """``R_99991231`` carries ``promoted: true`` on an ``event: "round"`` row."""
    assert "R_99991231_000000" not in [r["round_id"] for r in measure_world(world)["rounds"]]


def test_repeated_promotions_in_one_round_share_one_denominator_row(tmp_path):
    """Two variants promoted in one round is one round in the denominator.

    The ledger records one decision row per variant, so counting rows instead of
    distinct rounds would inflate the denominator — and with it the rate.
    """
    rounds_dir, ledger = tmp_path / "rounds", tmp_path / "ledger.jsonl"
    plan = [
        ("R_0", 0.60, []),
        ("R_1", 0.60, [("V_a", 0.10), ("V_b", 0.12)]),
        ("R_1b", 0.61, []),
        ("R_2", 0.60, []),
    ]
    evals, decisions = {}, []
    for rid, mean, winners in plan:
        write_round(
            rounds_dir,
            rid,
            mean,
            {vid: (mean + d, d) for vid, d in winners} if winners else None,
        )
        evals[rid] = {"BASELINE_V": score(rid, mean)}
        for vid, d in winners:
            evals[rid][vid] = score(rid, mean + d)
            decisions.append(
                {"round_id": rid, "event": "decision", "variant_id": vid, "should_promote": True, "promoted": True}
            )
        if not winners:
            decisions.append(
                {"round_id": rid, "event": "decision", "variant_id": "V_x", "should_promote": False, "promoted": False}
            )
    write_ledger(ledger, evals, decisions)
    result = pfr.measure(ledger_path=ledger, rounds_dir=rounds_dir, data_cutoff="R_2")
    assert [r["round_id"] for r in result["rounds"]] == ["R_1"]
    assert result["denominator"] == 1
    assert result["rounds"][0]["variants"] == ["V_a", "V_b"]
    assert len(result["gain_side"]["per_round"]) == 2  # both variants still reported


def test_the_parsers_agree_when_the_report_carries_the_post_549_line(tmp_path):
    """#549 made the gate's headline delta the TARGETED slice, not the overall mean.

    The line now carries several signed deltas in different senses — the targeted
    slice, the veto slice, the normalized gain, the win fraction — on one line. The
    FP-rate derivation takes the recorded gain from the report text and the
    ledger-derived gain from the ledger's per-task rows, then compares them;
    ``test_the_real_promote_line_records_the_targeted_gain_and_nothing_else`` pins
    the exact form `run_round` emits, so a formatter change that reorders or renames
    the fields trips a test instead of quietly making the instrument compare a veto
    slice against a mean.

    The two round files below are written in that real shape (composed, not
    hand-written — see the helper), because the whole point of this comparison is
    that the report the instrument reads is the report the round wrote.
    """
    rounds_dir, ledger = tmp_path / "rounds", tmp_path / "ledger.jsonl"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_cfg(tmp_path)

    base = slice_scores(0.40, 0.40)                    # every task at 0.40
    won_targeted = slice_scores(0.80, 0.60)            # targeted +0.40, veto +0.20
    uniform = slice_scores(0.80, 0.80)                 # every task +0.40
    won_veto_only = slice_scores(0.40, 0.90)           # targeted flat, veto +0.50

    def report(rid: str, decision_line: str, split_note: bool = False) -> None:
        lines = [f"# Autoresearch Round {rid}", ""]
        if split_note:
            lines += ["- targeted (6): the replay and synthetic tasks",
                      "- held-out (5): the adversarial and safety tasks, never named to the proposer"]
        lines += ["- baseline mean composite: 0.4000", "", "## Promotion decisions", "",
                  decision_line, ""]
        (rounds_dir / f"{rid}.md").write_text("\n".join(lines), encoding="utf-8")

    # A round that moved the targeted slice harder than the veto slice: the gate's
    # recorded gain (+0.4000) and the ledger's overall per-task delta (+0.3091) are
    # different quantities, and that difference is what this comparison exists to see.
    report("R_20261001_000000", promote_line("V_a", cfg, base, won_targeted), split_note=True)
    # A uniform round: every task moved +0.40, so the targeted slice and the whole
    # bench say the same number. This is the ONLY case where the two parsers must
    # agree — uniform movement is exactly when a slice mean equals the overall mean.
    report("R_20261003_000000", promote_line("V_c", cfg, base, uniform), split_note=True)
    # A PROMOTE line whose only signed number is the veto slice. The real writer
    # cannot emit this for an accept (`targeted_delta` leads every accept reason),
    # so it is a robustness probe, not a transcription of a real round: if the
    # reader ever fell through to a veto-side number, the instrument would credit
    # a held-out move as the recorded gain.
    report("R_20261002_000000",
           "- `V_b`: PROMOTE — promote (heldout_delta=+0.5000, win_frac=1.00)")

    write_ledger(ledger, {
        "R_20261001_000000": {"BASELINE_V": base, "V_a": won_targeted},
        "R_20261002_000000": {"BASELINE_V": base, "V_b": won_veto_only},
        "R_20261003_000000": {"BASELINE_V": base, "V_c": uniform},
    }, [
        {"round_id": "R_20261001_000000", "event": "decision", "variant_id": "V_a",
         "should_promote": True, "promoted": True},
        {"round_id": "R_20261002_000000", "event": "decision", "variant_id": "V_b",
         "should_promote": True, "promoted": True},
        {"round_id": "R_20261003_000000", "event": "decision", "variant_id": "V_c",
         "should_promote": True, "promoted": True},
    ])
    recorded = pfr.recorded_deltas(rounds_dir)
    assert recorded == {("R_20261001_000000", "V_a"): 0.4000,
                        ("R_20261003_000000", "V_c"): 0.4000}, (
        "the targeted delta is the recorded gain; the veto delta, the normalized "
        "gain and the win fraction on the same line are decoration, and a line "
        "carrying only a veto number yields no recorded delta at all")

    table = pfr.score_table(pfr.per_task_rows(ledger))
    rounds = {"R_20261001_000000", "R_20261002_000000", "R_20261003_000000"}
    paired, _ = pfr.paired_deltas(table, rounds, {"R_20261001_000000": ["V_a"],
                                                  "R_20261002_000000": ["V_b"],
                                                  "R_20261003_000000": ["V_c"]})
    assert paired[("R_20261003_000000", "V_c")] == recorded[("R_20261003_000000", "V_c")]
    # The two non-uniform rounds are the point: the ledger delta is diluted by the
    # slice the gate deliberately does NOT score as a gain, so reading the wrong
    # column would show a 0.09/0.23 swing rather than the gate's own number.
    assert paired[("R_20261001_000000", "V_a")] == pytest.approx(0.3091, abs=1e-4)
    assert ("R_20261002_000000", "V_b") not in recorded
    assert paired[("R_20261002_000000", "V_b")] == pytest.approx(0.2273, abs=1e-4)


def test_the_real_promote_line_records_the_targeted_gain_and_nothing_else(tmp_path):
    """The exact PROMOTE-line shape `run_round.run()` writes after #549, pinned.

    Kept separate from the derivation test above so a formatter edit shows up as a
    name that says which contract broke.

    The line is *composed* by the writer (:func:`promote.evaluate_promotion` →
    ``f"- \\`vid\\`: PROMOTE — {reason}"``, the same expression as
    ``run_round.run()``'s decision loop) instead of typed by hand, and the three
    properties pinned are the ones the reader depends on: it parses, the captured
    number is the TARGETED delta, and the veto slice / normalized gain / win
    fraction that also ride on the line are not what gets captured.

    The hand-written version of this test was collected by no one — it lacked the
    ``test_`` prefix — and its content was false about the shape it claimed to pin:
    it led with ``promote (delta=+0.3833)``, which is the legacy reason string,
    followed by ``| targeted_delta=+0.4000``, which the writer never emits after a
    pipe. Real post-#549 lines carry ``delta=`` nowhere at all (the writer replaced
    that field with ``targeted_delta=`` in the reason), so the match it asserted
    would have come from the legacy branch while the assertion named the new field.
    """
    cfg = make_cfg(tmp_path)
    base = slice_scores(0.40, 0.40)
    line = promote_line("V_20261008_074540_4e602b", cfg, base, slice_scores(0.80, 0.60))

    match = pfr.PROMOTE_LINE_RE.match(line)
    assert match is not None, f"the real promote line failed to parse: {line}"
    assert match.group("delta") == "+0.4000", line
    # What the writer actually puts on the line. A rename of any of these four is a
    # reader change too, and this says so at the point of failure.
    assert "PROMOTE — promote (targeted_delta=+0.4000, heldout_delta=+0.2000, " in line, line
    assert "normalized_gain=" in line and "win_frac=" in line, line
    assert "delta=+0.4000" in line and line.count("delta=+") == 2, (
        "two signed `delta=` fields ride this line — the targeted slice and the "
        "veto slice — and only the first may be the recorded gain")

    veto_only = "- `V_x`: PROMOTE — promote (heldout_delta=+0.5000, win_frac=1.00)"
    assert pfr.PROMOTE_LINE_RE.match(veto_only) is None, (
        "a line carrying only the veto slice yields no recorded delta; guessing "
        "there is how the instrument would score a refusal as a promotion")

    legacy = "- `V_20260905_000504_f711cb`: PROMOTE — promote (delta=+0.0782, win_frac=0.55)"
    assert pfr.PROMOTE_LINE_RE.match(legacy).group("delta") == "+0.0782", (
        "rounds recorded before the split keep their single whole-bench delta, and "
        "67 of them are still the corpus this instrument measures")


def test_the_tie_split_added_to_a_refusal_is_never_read_as_a_promotion(tmp_path):
    """#1060 put a wins/ties/losses split into the HOLD reason; the reader is unaffected.

    Two halves, one per direction of the boundary. Writer side: the promote branch
    still emits the line this instrument parses — `promote (… win_frac=…)` — and the
    captured number is still the targeted gain, so the FP-rate denominator and the
    recorded deltas of rounds after this change stay comparable with the 65 before
    it. Reader side: the new split rides only on refusal lines, and a refusal line
    carrying `wins=3 ties=5 losses=0` must yield no recorded delta at all — the
    three new counts are unsigned integers, so the one guard that keeps them out is
    the `PROMOTE` literal, which is why a HOLD line is asserted here and not merely
    assumed.
    """
    cfg = make_cfg(tmp_path)
    base = slice_scores(0.40, 0.40)
    # Two targeted tasks up (+0.20 each), the veto slice up by 0.01, and ONE targeted
    # task 0.05 below baseline. That last term is #595's doing: without a regression
    # this construction is now accepted, because 2 wins / 0 losses is exactly the
    # dominating shape this item reclaimed from the tie rule (the recorded ledger row
    # `V_20260901_111642_70c3ba`). The construction stays a refusal, which is the only
    # thing this test is about: that a HOLD line's split is never read as a delta.
    var = {tid: (sc + 0.20 if tid in ("task_0", "task_2") and _is_targeted(tid)
                 else sc - 0.05 if tid == "task_4" and _is_targeted(tid)
                 else sc + 0.01 if _is_heldout(tid) else sc)
           for tid, sc in base.items()}

    should, reason = promote.evaluate_promotion(cfg, bench_summary(base), bench_summary(var),
                                                split=None)
    assert should is False and reason.startswith("insufficient_win_fraction"), reason
    assert "wins=2 ties=3 losses=1" in reason, reason

    hold_line = f"- `V_20261008_074540_4e602b`: HOLD — {reason}"
    assert pfr.PROMOTE_LINE_RE.match(hold_line) is None, (
        f"a refusal with the new split parsed as a promotion: {hold_line}")

    promoted = promote_line("V_20261008_074541_111111", cfg, base,
                            {tid: (sc + 0.20 if _is_targeted(tid) else sc + 0.01)
                             for tid, sc in base.items()})
    match = pfr.PROMOTE_LINE_RE.match(promoted)
    assert match is not None, f"the promote line stopped parsing: {promoted}"
    assert "win_frac=" in promoted, promoted
    assert match.group("delta") == "+0.2000", promoted
    assert promoted.count("wins=") == 0, (
        "the split belongs to refusals; a promote line carrying it would make the "
        "two report shapes diverge for no reader")


def test_a_window_with_no_null_rounds_refuses_to_invent_a_floor(tmp_path):
    """Every round promoted ⇒ no non-promoted rounds to measure noise against.

    A silent fallback to some assumed constant would print an FP rate with no
    noise control, which is the exact thing backlog #428 was filed to stop.
    """
    rounds_dir, ledger = tmp_path / "rounds", tmp_path / "ledger.jsonl"
    for i in range(4):
        write_round(rounds_dir, f"R_{i}", 0.60, {f"V_{i}": (0.70, 0.10)})
    write_ledger(
        ledger,
        {f"R_{i}": {"BASELINE_V": score(f"R_{i}", 0.60), f"V_{i}": score(f"R_{i}", 0.70)} for i in range(4)},
        [
            {"round_id": f"R_{i}", "event": "decision", "variant_id": f"V_{i}", "should_promote": True, "promoted": True}
            for i in range(4)
        ],
    )
    with pytest.raises(ValueError, match="no null population"):
        pfr.measure(ledger_path=ledger, rounds_dir=rounds_dir, data_cutoff="R_3")


# ── the noise floor and the false-positive rule ──────────────────────────────


def test_floor_is_the_upper_percentile_of_the_null_drops(world):
    """The floor comes from the non-promoted rounds, not from a chosen constant."""
    result = measure_world(world)
    null = result["null_population"]
    assert null["n"] == 9  # non-promoted rounds in the window that have later rounds
    assert 0.1 < null["p95"] < 0.2  # driven by the collapse, not by a chosen constant
    assert result["floor"] == null["p95"]


def test_a_promotion_followed_by_a_beyond_noise_drop_is_reported_as_false_positive(world):
    """The headline result is 0/65, so prove the rule can count one.

    ``R_20260109`` promotes, then the next rounds' baselines sit 0.09 lower — an
    order of magnitude past this fixture's null spread. A measurement blind to that
    is not a false-positive measurement at all.
    """
    result = measure_world(world)
    assert result["fp_count"] == 1
    assert result["fp_rounds"] == ["R_20260109_000000"]
    assert result["fp_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert result["fp_rate_fraction"] == "1/3"
    assert result["band"] == "consider raising to 0.575"


def test_a_drop_inside_the_noise_floor_is_not_counted(world):
    """Same fixture, floor pushed above the observed drop: the count goes to zero.

    Pairs with the test above so neither "always 0" nor "always FP" can pass.
    """
    result = measure_world(world, alt_floors={"way_high": 0.5})
    assert result["sensitivities"]["way_high"]["fp_count"] == 0
    assert result["sensitivities"]["way_high"]["band"] == "keep 0.5"


def test_beyond_noise_drop_is_attributed_to_the_round_that_promoted(tmp_path):
    """The FP list names round ids, and only rounds that were promoted."""
    rounds_dir, ledger = tmp_path / "rounds", tmp_path / "ledger.jsonl"
    plan = [("R_%02d" % i, 0.60) for i in range(1, 8)]
    plan[3] = ("R_04", 0.60)  # the promoted round
    for rid, mean in plan:
        write_round(rounds_dir, rid, mean)
    # The post-R_04 collapse has to exist in the ledger, not just the report: the
    # baseline means are read from per-task rows, so a report-only collapse would be
    # a fixture lying about the source of record.
    collapsed_after = {rid: mean - 0.30 for rid, mean in plan[4:]}
    means = {rid: collapsed_after.get(rid, mean) for rid, mean in plan}
    evals = {rid: {"BASELINE_V": score(rid, means[rid])} for rid, _ in plan}
    evals["R_04"]["V_w"] = score("R_04", 0.70)
    decisions = [
        {"round_id": "R_04", "event": "decision", "variant_id": "V_w", "should_promote": True, "promoted": True}
    ] + [
        {"round_id": rid, "event": "decision", "variant_id": "V_x", "should_promote": False, "promoted": False}
        for rid, _ in plan
        if rid != "R_04"
    ]
    write_ledger(ledger, evals, decisions)
    for rid, mean in plan:
        write_round(rounds_dir, rid, means[rid], {"V_w": (0.70, 0.10)} if rid == "R_04" else None)
    result = pfr.measure(ledger_path=ledger, rounds_dir=rounds_dir, data_cutoff="R_07")
    assert result["fp_rounds"] == ["R_04"]
    assert result["rounds"][0]["drop"] == pytest.approx(0.30, abs=1e-3)


def test_collapsed_rounds_never_serve_as_the_control_window(tmp_path):
    """A ``bm == 0`` round is an infrastructure failure, not a dropped score.

    Left in the control window it would turn any promotion into a false positive of
    size ``bm(r)`` — the trap the crude triage proxy walked into.
    """
    rounds_dir, ledger = tmp_path / "rounds", tmp_path / "ledger.jsonl"
    rids = ["R_01", "R_02", "R_03", "R_04", "R_05"]
    means = {"R_01": 0.60, "R_02": 0.00, "R_03": 0.60, "R_04": 0.60, "R_05": 0.60}
    for rid in rids:
        write_round(
            rounds_dir,
            rid,
            means[rid],
            {"V_w": (means[rid] + 0.10, 0.10)} if rid == "R_01" else None,
        )
    # A collapsed round has to collapse in the ledger rows, which is where the
    # baseline mean is read from; score() would give it a 0.01 second task.
    evals = {
        rid: {"BASELINE_V": {f"task_{i}": 0.0 for i in range(2)} if means[rid] == 0.0 else score(rid, means[rid])}
        for rid in rids
    }
    evals["R_01"]["V_w"] = score("R_01", 0.70)
    decisions = [{"round_id": "R_01", "event": "decision", "variant_id": "V_w", "should_promote": True, "promoted": True}] + [
        {"round_id": rid, "event": "decision", "variant_id": "V_x", "should_promote": False, "promoted": False}
        for rid in rids
        if rid != "R_01"
    ]
    write_ledger(ledger, evals, decisions)
    result = pfr.measure(ledger_path=ledger, rounds_dir=rounds_dir, data_cutoff="R_05")
    assert result["rounds_excluded_zero_baseline"] == 1
    record = result["rounds"][0]
    assert record["control_rounds_used"] == 3  # R_03..R_05, not R_02
    assert record["drop"] == pytest.approx(0.0, abs=1e-3)
    assert result["fp_count"] == 0


def test_an_unlanded_gate_pass_is_a_control_only_if_that_round_promoted_nothing(world):
    """Control-ness is computed from the data, and the data can say no.

    In this fixture the gate-passing round landed nothing, so it *is* a clean
    no-landing control — which is exactly why the live-data test below is worth
    writing: on the live ledger the same query returns rounds that all promoted
    something else, and the flag flips to false.
    """
    result = measure_world(world)
    passes = result["unlanded_gate_passes"]
    assert passes["rows"] == 1 and passes["rounds"] == [world["control_round"]]
    assert passes["also_promoted_rounds"] == 0
    assert passes["is_a_no_landing_control"] is True


# ── the frozen window ────────────────────────────────────────────────────────


def test_rounds_after_the_cutoff_cannot_rewrite_the_measurement(world):
    """Re-arming the loop appends rounds; the published window must not move."""
    before = measure_world(world)
    # A later round with both a report and per-task rows: re-arming the loop appends
    # real data, not just a file.
    write_round(world["rounds"], "R_20260201_000000", 0.05)
    rows = [line for line in world["ledger"].read_text().splitlines() if line.strip()]
    rows += [
        json.dumps(
            {
                "round_id": "R_20260201_000000",
                "variant_id": "BASELINE_V",
                "task_id": task_id,
                "composite_score": value,
            }
        )
        for task_id, value in score("R_20260201_000000", 0.05).items()
    ]
    world["ledger"].write_text("\n".join(rows) + "\n")
    after_same_cutoff = measure_world(world)
    assert after_same_cutoff["floor"] == before["floor"]
    assert after_same_cutoff["fp_count"] == before["fp_count"]
    assert after_same_cutoff["denominator"] == before["denominator"]
    # ... while a deliberately later cutoff does see it, so the knob is real.
    later = measure_world(world, data_cutoff="R_20260201_000000")
    assert later["rounds_with_baseline_mean_in_window"] > before["rounds_with_baseline_mean_in_window"]


def test_appending_a_later_round_changes_no_counted_field_at_the_declared_cutoff(world):
    """#1193 — a count pinned over a whole live file is a time bomb; a window count is not.

    Three nodes in this file used to assert ``decision_rows == 2217`` and
    ``report_crosscheck_rounds == 351`` read over the *whole* ``_pipeline/research`` store.
    Those numbers were the window's figures by accident — the store happened to hold
    nothing past the cutoff — and one autoresearch round appending 4 decision rows and one
    report turned both red. Because the store is reached by absolute path the gate's base
    worktree reproduced the red, classified every promotion as externally blocked, and
    refused it. So the regression is not "the number was wrong", it is "the count must not
    be able to move": re-measuring at the same cutoff, after a later round has landed in
    both stores, must reproduce the whole result.
    """
    base = measure_world(world)
    appended = "R_20260301_000000"

    # A real later round: decision rows, a ledger baseline, and a round report carrying
    # that baseline mean — the two things that used to move the pinned counts.
    with world["ledger"].open("a") as fh:
        fh.write(
            json.dumps(
                {
                    "round_id": appended,
                    "variant_id": "BASELINE_V",
                    "task_id": "bench_002_output_shape",
                    "composite_score": 0.40,
                }
            )
            + "\n"
        )
        for record in (
            {
                "round_id": appended,
                "event": "decision",
                "variant_id": "BASELINE_V",
                "should_promote": False,
                "promoted": False,
                "reason": "insufficient_delta",
            },
            {
                "round_id": appended,
                "event": "decision",
                "variant_id": "V_late",
                "should_promote": True,
                "promoted": True,
                "reason": "promote",
            },
        ):
            fh.write(json.dumps(record) + "\n")
    # write_round names the file after the round id verbatim, so the stem and the ledger's
    # round_id match — the two stores intersect on it, which is what the cross-check counts.
    write_round(world["rounds"], appended, 0.40, {"V_late": (0.47, 0.07)})

    after = measure_world(world)
    assert after == base, "a round appended past the cutoff moved a published field"
    assert after["decision_rows"] == base["decision_rows"]
    assert after["report_crosscheck"]["rounds_compared"] == base["report_crosscheck"]["rounds_compared"]
    # Positive control: the stores really did grow and the reader sees it — the appended
    # round is excluded by the window, not by blindness. Un-windowed, it is counted; a
    # whole-file count is exactly the quantity this file must never assert.
    assert len(pfr.decision_rows(world["ledger"])) > len(
        pfr.decision_rows(world["ledger"], base["data_cutoff_round"])
    )
    assert len(pfr.baseline_means(world["rounds"])) > after["report_crosscheck"]["rounds_compared"]
    grown = measure_world(world, data_cutoff=appended)
    assert grown["decision_rows"] > after["decision_rows"]


@pytest.mark.parametrize(
    "round_id,in_window",
    [
        ("R_20260908_181458", True),  # the cutoff round itself is inside the window
        ("R_20260908_165252", True),
        ("R_20260916_174109", False),  # the round that broke the pins
        ("R_99991231_000000", False),
        ("", False),
        (None, False),
    ],
)
def test_in_window_is_lexicographic_and_excludes_rows_with_no_round(round_id, in_window):
    """The scope rule, asserted directly, including the two edges the ledger can grow."""
    assert pfr.in_window(round_id, "R_20260908_181458") is in_window
    assert pfr.in_window(round_id, None) is True  # no declared window excludes nothing


# ── reporting helpers ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "rate,label",
    [
        (0.0, "keep 0.5"),
        (0.199, "keep 0.5"),
        (0.20, "consider raising to 0.575"),
        (0.399, "consider raising to 0.575"),
        (0.40, "revert to 0.6"),
        (1.0, "revert to 0.6"),
    ],
)
def test_band_boundaries_are_backlog_352(rate, label):
    assert pfr.band_for(rate) == label


def test_unavailable_controls_are_recorded_with_their_evidence():
    """The method deviation is auditable, so both keys carry proof text."""
    text = " ".join(pfr.UNAVAILABLE_CONTROLS.values())
    assert set(pfr.UNAVAILABLE_CONTROLS) == {"same_trace_rescore", "same_day_control_round"}
    assert "run_spec.yaml" in text and "judge_trace" in text
    assert "enabled: false" in text and "#506" in text


def test_percentile_interpolates_between_observations():
    assert pfr.percentile([0.0, 1.0], 0.95) == pytest.approx(0.95)
    assert pfr.percentile([0.5], 0.95) == 0.5
    with pytest.raises(ValueError):
        pfr.percentile([], 0.95)


def test_recorded_deltas_are_read_from_the_round_report(world):
    """The gate's own printed delta, transcribed — cross-checked against the ledger."""
    result = measure_world(world)
    per_round = {g["round_id"]: g for g in result["gain_side"]["per_round"]}
    assert per_round["R_20260104_000000"]["recorded_delta"] == pytest.approx(0.10)
    assert per_round["R_20260104_000000"]["paired_delta"] == pytest.approx(0.10, abs=1e-3)


# ── the published command ────────────────────────────────────────────────────
#
# The vault note publishes exactly one way to reproduce the measurement:
#
#     cd ~/lloyd && .venvs/lloyd/bin/python -m scripts.autoresearch.promotion_fp_rate \
#       --alt-floor "std_324=0.1389" --alt-floor "one_sigma_null=0.0649" \
#       --alt-floor "crude_drop_positive=0.0" --alt-floor "gate_delta_floor=0.05"
#
# Everything above this section tests :func:`promotion_fp_rate.measure` as a
# library. Nothing in the tree calls ``main()`` (graph_explain reports zero
# inbound edges), so
# without the two tests below the published surface — the ``-m`` entry, the
# argparse defaults resolving from ``REPO_ROOT``, ``--alt-floor NAME=VALUE``
# parsing and ``--out`` — would be exercised by no test at all: a broken CLI
# would still ship a green suite and a note telling readers a command that
# fails.


def test_main_writes_the_report_the_note_publishes(world, tmp_path, capsys):
    """``main()`` end-to-end on the fixture world: args in, JSON at ``--out`` out.

    Exercises every knob the published command uses — ``--ledger`` /
    ``--rounds-dir`` / ``--data-cutoff``, ``--alt-floor NAME=VALUE`` parsing and
    ``--out`` — and asserts the written report carries the fixture's
    denominator (3) and FP count (1), i.e. the same pipeline the live number
    runs through, with the CLI as the thing under test.
    """
    out = tmp_path / "report.json"
    rc = pfr.main(
        [
            "--ledger", str(world["ledger"]),
            "--rounds-dir", str(world["rounds"]),
            "--data-cutoff", "R_20260112_000000",
            "--alt-floor", "way_high=0.5",
            "--out", str(out),
        ]
    )  # fmt: skip
    assert rc == 0
    report = json.loads(out.read_text())
    assert report["denominator"] == 3
    assert report["fp_count"] == 1
    assert report["fp_rate_fraction"] == "1/3"
    # --alt-floor NAME=VALUE actually parsed into the sensitivity table
    assert report["sensitivities"]["way_high"]["floor"] == 0.5
    assert report["sensitivities"]["way_high"]["fp_count"] == 0
    assert report["sensitivities"]["way_high"]["band"] == "keep 0.5"
    # stdout carries the identical report (the command prints what it writes)
    assert json.loads(capsys.readouterr().out) == report
    # The bare published command passes no --ledger/--rounds-dir, so its
    # defaults — resolved here from REPO_ROOT — are part of the published
    # contract; a move of the store must fail a test, not the reader.
    assert pfr.DEFAULT_LEDGER == pfr.REPO_ROOT / "_pipeline" / "research" / "ledger.jsonl"
    assert pfr.DEFAULT_ROUNDS_DIR == pfr.REPO_ROOT / "_pipeline" / "research" / "rounds"


def test_the_module_runs_as_the_published_python_m_command(world, tmp_path):
    """The note's command verbatim: a fresh interpreter, ``-m``, cwd = repo root.

    The in-process test above proves the function; this one proves the *form* —
    ``python -m scripts.autoresearch.promotion_fp_rate`` — resolves, imports
    and runs, which is the seam between the published note and the checkout.
    """
    out = tmp_path / "sub_report.json"
    proc = subprocess.run(
        [
            sys.executable, "-m", "scripts.autoresearch.promotion_fp_rate",
            "--ledger", str(world["ledger"]),
            "--rounds-dir", str(world["rounds"]),
            "--data-cutoff", "R_20260112_000000",
            "--out", str(out),
        ],  # fmt: skip
        cwd=pfr.REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = json.loads(out.read_text())
    assert report["denominator"] == 3
    assert report["fp_count"] == 1
    assert report["data_cutoff_round"] == "R_20260112_000000"


# ── the published figures ────────────────────────────────────────────────────
#
# Read-only against the production store, frozen by cutoff. If one of these fails,
# either the derivation changed or the note no longer matches it — both are findings,
# and neither is fixed by editing an assertion.


#: The declared right edge of the published window, and the note's
#: ``data_cutoff_round``. Every live figure below is counted over
#: ``round_id <= WINDOW_CUTOFF`` — the store may grow past it freely.
WINDOW_CUTOFF = "R_20260908_181458"

#: The lookahead each non-promoted round needs before it can contribute a null
#: value. Named here because the volume floor below is derived from it, not
#: typed twice: under ``WINDOW + 1`` rounds no round in the window has a full
#: lookahead, so `pfr.measure` cannot form a null population at all and raises
#: rather than returning a figure this file could compare.
WINDOW = 3


#: Backlog #324's independent figure, knowledge/evaluation/autoresearch-baseline-stability.md

#: Exact text of the note's ``noise_floor_source`` field. Compared for equality, not
#: by substring: the claim is about which store the floor is computed from, and a
#: substring check would pass on prose that named the wrong one.


























#: Every field the note publishes that is a count or a rate — the ones a reader is
#: tempted to nudge. ``schema`` and ``noise_floor_source`` are prose promises and are
#: pinned by the whole-block comparison instead.






# ── the note's shell reproduce block ─────────────────────────────────────────
#
# The note publishes a second way to check its own numbers: a bash block under
# "## Denominator — shell-reproducible" that counts the ledger with jq and prints the
# expected figure in a comment beside each command. Until #1193 those commands counted
# the WHOLE live ledger while the comments printed the window's figures — so the block
# the note offers as the check was the one thing in the note that could not be checked,
# and it silently disagreed with the number next to it the moment the worker appended a
# round. No test ran the block, which is how that shipped. These two tests close that:
# one re-runs the block as a reader would and compares every printed count to the
# number printed beside it, the other proves the published derivation is blind to a
# round appended to a copy of the live store.













