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
(#506) cannot change the numbers; the note-reading test is marked ``live_vault``
because it opens a file under ``~/obsidian``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts.autoresearch import promotion_fp_rate as pfr

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


# ── the published figures ────────────────────────────────────────────────────
#
# Read-only against the production store, frozen by cutoff. If one of these fails,
# either the derivation changed or the note no longer matches it — both are findings,
# and neither is fixed by editing an assertion.

LIVE_RESEARCH = Path.home() / "lloyd" / "_pipeline" / "research"
LIVE_LEDGER = LIVE_RESEARCH / "ledger.jsonl"
LIVE_ROUNDS = LIVE_RESEARCH / "rounds"

NOTE_PATH = (
    Path.home()
    / "obsidian"
    / "knowledge"
    / "evaluation"
    / "autoresearch-promotion-fp-rate-at-threshold-0p5.md"
)

#: Backlog #324's independent figure, knowledge/evaluation/autoresearch-baseline-stability.md
NOISE_STD_FROM_ITEM_324 = 0.1389

#: Exact text of the note's ``noise_floor_source`` field. Compared for equality, not
#: by substring: the claim is about which store the floor is computed from, and a
#: substring check would pass on prose that named the wrong one.
NOISE_FLOOR_SOURCE = (
    "ledger.jsonl BASELINE_* per-task composite_score rows, averaged per round; "
    "cross-checked against rounds/R_*.md"
)


def live_measure():
    return pfr.measure(
        ledger_path=LIVE_LEDGER,
        rounds_dir=LIVE_ROUNDS,
        data_cutoff="R_20260908_181458",
        window=3,
        alpha=0.05,
        alt_floors={
            "std_324": NOISE_STD_FROM_ITEM_324,
            "one_sigma_null": 0.0653,
            "crude_drop_positive": 0.0,
            "gate_delta_floor": 0.05,
        },
    )


def note_machine_checked_block() -> dict:
    text = NOTE_PATH.read_text(encoding="utf-8")
    blocks = [b for b in re.findall(r"```yaml\n(.*?)```", text, re.S) if "autoresearch-promotion-fp" in b]
    assert len(blocks) == 1, f"expected exactly one machine-checked block in {NOTE_PATH}"
    import yaml

    return yaml.safe_load(blocks[0])


def test_live_ledger_denominator_is_sixty_five_promoted_rounds():
    """Clause 2 — the note's denominator is what the ledger reports today."""
    result = live_measure()
    assert result["denominator"] == 65
    assert result["decision_rows"] == 2217
    assert result["should_promote_rows"] == 95
    assert len({r["round_id"] for r in result["rounds"]}) == 65  # rows and rounds agree
    assert result["data_cutoff_round"] == "R_20260908_181458"


def test_the_floor_input_is_the_ledger_and_the_round_reports_agree():
    """The floor is computed from the ledger's own per-task rows; the reports are the
    cross-check, and the switch is only auditable if that check is pinned.

    The reports round to 4 decimals, so agreement is to 1e-3, not to 0. Three rounds
    have a ledger baseline and no report at all — reading the reports as the source
    would drop them from the window without saying so.
    """
    check = live_measure()["report_crosscheck"]
    assert check["source_of_record"].startswith("ledger.jsonl")
    assert check["rounds_compared"] == 351
    assert check["rounds_with_ledger_baseline_but_no_report"] == 3
    assert check["mismatches"] == 0
    assert check["max_abs_delta"] <= 1e-4


def test_live_noise_floor_and_false_positive_rate():
    """Clauses 3, 4 and 6 — floor from on-disk data, FP only past it, band named."""
    result = live_measure()
    assert result["floor"] == pytest.approx(0.1289, abs=1e-4)
    assert "BASELINE_* per-task composite_score rows" in result["floor_definition"]
    assert result["null_population"]["n"] == 281
    assert result["null_population"]["std"] == pytest.approx(0.0649, abs=1e-3)
    assert result["fp_count"] == 0
    assert result["fp_rounds"] == []
    assert result["fp_rate_fraction"] == "0/65"
    assert result["band"] == "keep 0.5"
    # Every promoted round was still evaluated against the rule, and the largest
    # observed drop stayed under the floor — this is why the count is zero.
    assert len([r for r in result["rounds"] if r["drop"] is not None]) == 65
    assert result["promoted_drops"]["max"] == pytest.approx(0.0867, abs=1e-3)
    assert result["promoted_drops"]["max"] < result["floor"]
    # The crude "any drop" rule, which is what the item body's 23.3% proxy was,
    # would have called 11 rounds false positives and still landed in keep-0.5.
    crude = result["sensitivities"]["crude_drop_positive"]
    assert crude["fp_rate_fraction"] == "11/65"
    assert crude["band"] == "keep 0.5"
    assert result["sensitivities"]["std_324"]["fp_count"] == 0


def test_live_ledger_has_no_promotion_free_control_round():
    """The rows that look like a control group are runner-ups, not a control.

    ``should_promote: true`` with ``promoted`` false reads like "passed the gate,
    nothing landed" — 30 such rows over 16 rounds — but every one of those 16 rounds
    promoted a *different* variant. Scoring them as a no-landing control would have
    been scoring the promoted group against itself, so this measures the overlap
    rather than asserting it away. If a future round ever passes the gate and lands
    nothing, this flips and a real control becomes available.
    """
    passes = live_measure()["unlanded_gate_passes"]
    assert passes["rows"] == 30
    assert len(passes["rounds"]) == 16
    assert passes["also_promoted_rounds"] == 16
    assert passes["is_a_no_landing_control"] is False


def test_note_fields_are_reproduced_by_a_fresh_derivation():
    """Clauses 1, 3, 5 and 6 — the published note is the measurement, not prose.

    Every field of the note's machine-checked block must equal a fresh run over the
    cutoff the note itself declares; a hand-edited number fails here.

    Deliberately NOT marked ``live_vault``. That marker is for files an autoresearch
    promotion or a nightly job rewrites, and the gate run deselects it — but this note
    is the artefact this repo exists to keep honest, and the boundary between the code
    and the published number is the seam that has to be crossed on the graded run. The
    note is not rewritten by anything except a re-measurement, and the window it
    reports is frozen by cutoff, so the assertion is stable.
    """
    result = live_measure()
    block = note_machine_checked_block()
    assert block["schema"] == "autoresearch-promotion-fp/1"
    assert block["data_cutoff_round"] == result["data_cutoff_round"]
    assert block["window_rounds"] == result["window_rounds"]
    assert block["alpha"] == result["alpha"]
    assert block["denominator"] == result["denominator"] == 65
    assert block["noise_floor"] == result["floor"]
    assert block["noise_floor_source"] == NOISE_FLOOR_SOURCE
    assert block["null_population_n"] == result["null_population"]["n"]
    assert block["null_population_std"] == result["null_population"]["std"]
    assert block["null_population_p95"] == result["floor"]
    assert block["fp_count"] == result["fp_count"]
    assert block["fp_rate"] == result["fp_rate"]
    assert block["fp_rate_fraction"] == result["fp_rate_fraction"]
    assert block["band"] == result["band"]
    assert block["report_crosscheck_max_abs_delta"] == result["report_crosscheck"]["max_abs_delta"]
    assert block["report_crosscheck_mismatches"] == result["report_crosscheck"]["mismatches"]
    assert block["unlanded_gate_pass_rows"] == result["unlanded_gate_passes"]["rows"]
    assert block["unlanded_gate_pass_rounds"] == len(result["unlanded_gate_passes"]["rounds"])
    assert (
        block["unlanded_gate_pass_rounds_also_promoted"]
        == result["unlanded_gate_passes"]["also_promoted_rounds"]
    )
    assert block["expected_false_alarms_at_alpha"] == result["expected_false_alarms_at_alpha"]
    assert block["fp_rounds"] == result["fp_rounds"] == []


def test_note_states_the_method_deviation_and_the_band():
    """Prose-level presence checks for the same note; see the test above for why
    this one also runs on the graded gate pass.
    """
    text = NOTE_PATH.read_text(encoding="utf-8")
    lowered = text.lower()
    # clause 5: both unavailable controls, each with its evidence
    for needle in ("same-trace re-scoring", "same-day control round", "run_spec.yaml", "enabled: false"):
        assert needle in lowered, f"missing method-deviation evidence: {needle!r}"
    # and no phantom matched control: the note must say what the runner-ups actually are
    assert "not promotion-free rounds" in lowered
    assert "is_a_no_landing_control" in lowered
    # clause 6: the bands, and which one the number lands in
    for needle in ("keep 0.5", "0.575", "revert to 0.6"):
        assert needle in text, f"missing #352 band text: {needle!r}"
    # clause 3: #324's figure and its note path, as the independent cross-check
    assert "0.1389" in text and "autoresearch-baseline-stability.md" in text
    # the headline fraction, and the crude proxy it replaced
    assert "0/65" in text and "11/65" in text
    assert "never-touch" in lowered  # the threshold decision is left to a human
