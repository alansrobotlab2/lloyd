"""The regression guard has to compare, and has to know its own noise.

`run_tool_choice_eval.py` writes a baseline per run and prints nothing
comparable. Round SM_20260908_165950 ran it after rewriting 66% of the
operating contract, spent four iterations guessing the file's shape, and landed
quoting `correct_rate 0.95` against no prior at all — while a 0.90 from 09-04
sat on disk. A number with no baseline is not a guard.

The second thing worth pinning, and the reason #691 exists: the first version
decided against `TOLERANCE = 0.051`, which is *smaller* than the movement two
runs of the identical tree produce (measured 2026-09-09, 82 s apart:
`correct_rate` 0.050, `http_tool_first_rate` 0.071; reproduced 2026-09-13 as
`875noise-a`/`875noise-b`). A gate that fires on its own noise teaches the
implementer to argue past it. So these tests pin three things: direction still
works, a same-tree pair passes, and the threshold that decided each delta is
printed and derived from the metric's own denominator — not from a constant.

A consequence of the honest floor is stated here so nobody re-litigates it: a
two-query flip on 20 (0.10) is inside 2σ and is NOT reportable at n=20. The old
test asserted it was a regression, which is the claim #691 retracted. Fixing
that means raising the query count, and `certifiable_floor` is what tells you
how big a movement you can actually certify today.
"""
from __future__ import annotations

import json

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from eval import compare_tool_choice as C


def _run(n_web=14, n_ctl=6, **overall):
    base = {"correct_rate": 0.9, "http_tool_first_rate": 0.86,
            "control_correct_rate": 1.0, "shelled_to_web_rate": 0.0,
            "no_tool_call_rate": 0.0, "errors": 0, "n_queries": 20}
    base.update(overall)
    n = base.pop("n_queries")
    # `records` carries the split the denominators are counted from: which
    # metric is over all 20 and which is over the 14 web / 6 control rows.
    def rec(cat, ok, web):
        return {"category": cat,
                "scoring": {"correct": ok, "is_web_category": web}}
    # 14 web rows (8 public-search + 6 public-fetch) and 6 control rows, the
    # same split eval/tool_choice_queries.yaml has at n=20.
    records = ([rec("public-search", True, True) for _ in range(8)]
               + [rec("public-fetch", True, True) for _ in range(6)]
               + [rec("localhost", True, False) for _ in range(3)]
               + [rec("structured-api", True, False) for _ in range(3)])
    return {"n_queries": n, "summary": {"overall": base}, "records": records}


# The pair #691 measured: two runs of the SAME unmodified tree, 82 s apart.
# This is the reference failure to beat for acceptance clause 2.
NOISE_A = _run(correct_rate=0.900, http_tool_first_rate=0.857,
               control_correct_rate=1.0, shelled_to_web_rate=0.0)
NOISE_B = _run(correct_rate=0.850, http_tool_first_rate=0.786,
               control_correct_rate=1.0, shelled_to_web_rate=0.0)


# ── direction ───────────────────────────────────────────────────────────────

def test_an_unchanged_run_reports_no_regression():
    assert C.compare(NOISE_A, NOISE_A).regressions == []


def test_a_drop_in_correct_rate_is_a_regression():
    r = C.compare(_run(correct_rate=0.50), _run(correct_rate=0.90))
    assert any("correct_rate" in x for x in r.regressions)


def test_a_rise_in_correct_rate_is_not():
    assert C.compare(_run(correct_rate=0.95), _run(correct_rate=0.90)).regressions == []


def test_a_rise_in_a_failure_rate_is_a_regression():
    """`shelled_to_web_rate` going up is the defect the eval exists to catch."""
    r = C.compare(_run(shelled_to_web_rate=0.50), _run(shelled_to_web_rate=0.00))
    assert any("shelled_to_web_rate" in x for x in r.regressions)


def test_a_fall_in_a_failure_rate_is_an_improvement():
    r = C.compare(_run(no_tool_call_rate=0.0), _run(no_tool_call_rate=0.60))
    assert r.regressions == []


def test_an_errored_run_is_a_regression_whatever_its_rates():
    r = C.compare(_run(errors=3, correct_rate=1.0), _run())
    assert any("errored" in x for x in r.regressions)


# ── clause 1: the floor is per-metric, printed, and decides ──────────────────

def test_every_compared_metric_prints_its_floor_beside_the_delta():
    """Clause 1: a verdict with no threshold beside it is unauditable.

    Asserted over all five metrics: the old output could report `REGRESSED`
    against a constant the reader never saw and could not locate on a number.
    """
    r = C.compare(NOISE_B, NOISE_A)
    rows = [ln for ln in r.lines if "floor" in ln]
    assert len(rows) == len(C.METRICS), r.lines
    for metric in C.METRICS:
        row = next(ln for ln in r.lines if ln.strip().startswith(metric))
        assert "floor " in row, row
        # the floor is derived, and says so: a denominator and at least one term
        assert "@" in row.split("floor", 1)[1], row
    assert set(r.floors) == set(C.METRICS)


def test_the_floor_is_wider_for_the_smaller_denominator():
    """A rate over 6 control rows is intrinsically noisier than one over 20.

    This is why one global constant could not have been right for all five
    metrics: `control_correct_rate` at n=6 and `correct_rate` at n=20 cannot
    share a threshold and both be honest about it.
    """
    r = C.compare(NOISE_B, NOISE_A)
    ctl = r.floors["control_correct_rate"]["floor"]
    corr = r.floors["correct_rate"]["floor"]
    assert ctl > corr, (ctl, corr)
    assert r.floors["control_correct_rate"]["n"] == 6
    assert r.floors["correct_rate"]["n"] == 20


def test_the_decision_is_made_against_the_printed_floor():
    """The printed number and the number used must be the same number.

    A table that prints a floor while deciding against something else would
    satisfy clause 1's letter and reproduce #691 exactly.
    """
    r = C.compare(_run(correct_rate=0.50), _run(correct_rate=0.90))
    floor = r.floors["correct_rate"]["floor"]
    assert -0.40 < -floor                      # the delta really is beyond it
    assert any("correct_rate" in x for x in r.regressions)
    assert f"floor {floor:.3f}" in next(
        ln for ln in r.lines if ln.strip().startswith("correct_rate"))


def test_measured_same_tree_spread_widens_a_floor_and_says_so():
    """The observed spread is a floor too, and the label must show both terms."""
    obs = {"correct_rate": 0.30}
    r = C.compare(_run(correct_rate=0.85), _run(correct_rate=0.90), obs)
    assert r.floors["correct_rate"]["floor"] == pytest.approx(0.30)
    assert r.floors["correct_rate"]["src"].count("=") >= 2   # both terms labelled
    assert "measured same-tree spread=0.300" in r.floors["correct_rate"]["src"]
    assert r.regressions == []                 # 0.05 sits inside the measured floor


# ── clause 2: the same tree must pass ───────────────────────────────────────

def test_two_runs_of_the_same_tree_report_no_regressions():
    """Clause 2, on the numbers #691 measured on an unmodified tree."""
    r = C.compare(NOISE_B, NOISE_A)
    assert r.regressions == [], r.regressions
    assert r.failures == []


def test_the_reference_failure_pair_exits_zero(tmp_path, capsys):
    """Same pair through the CLI: exit 0, not exit 1."""
    a = tmp_path / "noise-a-1.json"
    b = tmp_path / "noise-b-2.json"
    a.write_text(json.dumps(NOISE_A), encoding="utf-8")
    b.write_text(json.dumps(NOISE_B), encoding="utf-8")
    assert C.main(["--current", str(b), "--baseline", str(a)]) == 0
    out = capsys.readouterr().out
    assert "no movement beyond the per-metric noise floor" in out
    assert "REGRESSED" not in out


def test_the_real_82_second_apart_pair_on_disk_still_passes():
    """The pair as the engine recorded it, not a reconstruction of it.

    `eval/noise_reference_tool_choice.yaml` holds the actual `875noise-a` and
    `875noise-b` artifacts — two runs of the same unmodified tree 75 s apart on
    2026-09-13, which moved `correct_rate` -0.050 and `http_tool_first_rate`
    -0.071, the same two magnitudes #691 recorded 82 s apart on 2026-09-09. It is
    YAML because `*.json` is ignored repo-wide and `eval/baselines/` is ignored
    outright: a check reading the live files would skip on a fresh clone and
    certify nothing on the machine grading it. If this ever compares as a
    regression, the tolerance has been pulled back under the noise.
    """
    import yaml as _yaml

    doc = _yaml.safe_load((ROOT / "eval" / "noise_reference_tool_choice.yaml").read_text())
    a, b = doc["a"], doc["b"]
    # The magnitudes themselves, so a fixture that drifts cannot quietly stop
    # testing the thing it was captured for.
    assert a["summary"]["overall"]["correct_rate"] == pytest.approx(0.900)
    assert b["summary"]["overall"]["correct_rate"] == pytest.approx(0.850)
    assert a["summary"]["overall"]["http_tool_first_rate"] == pytest.approx(0.857)
    assert b["summary"]["overall"]["http_tool_first_rate"] == pytest.approx(0.786)
    r = C.compare(b, a)
    assert r.regressions == [], r.regressions
    assert r.failures == [], r.failures
    assert r.split_known is True and r.n_queries == 20


def test_the_reference_fixture_agrees_with_itself():
    """The fixture's per-query records must imply its own summary block.

    Guards the one thing that would make the clause-2 test above worthless: a
    fixture whose `overall` numbers were hand-written while its `records` said
    something else. The comparison reads the rates from the summary and the
    denominators from the records, so a disagreement would be a gate deciding on
    a rate no query produced.
    """
    import yaml as _yaml

    doc = _yaml.safe_load((ROOT / "eval" / "noise_reference_tool_choice.yaml").read_text())
    for arm in ("a", "b"):
        recs = doc[arm]["records"]
        web = [r for r in recs if r["scoring"]["is_web_category"]]
        ctl = [r for r in recs if not r["scoring"]["is_web_category"]]
        o = doc[arm]["summary"]["overall"]
        assert o["n_queries"] == len(recs) == 20
        assert len(web) == 14 and len(ctl) == 6
        assert o["correct_rate"] == pytest.approx(
            sum(1 for r in recs if r["scoring"]["correct"]) / len(recs), abs=0.001)
        assert o["http_tool_first_rate"] == pytest.approx(
            sum(1 for r in web if r["scoring"]["used_http_tool"]) / len(web), abs=0.001)
        assert o["control_correct_rate"] == pytest.approx(
            sum(1 for r in ctl if r["scoring"]["correct"]) / len(ctl), abs=0.001)


def test_the_reference_pair_exits_zero_through_main(tmp_path, capsys):
    """The same pair through the CLI, because exit codes are what the gate reads."""
    import yaml as _yaml

    doc = _yaml.safe_load((ROOT / "eval" / "noise_reference_tool_choice.yaml").read_text())
    pa, pb = tmp_path / "ref-a.json", tmp_path / "ref-b.json"
    pa.write_text(json.dumps(doc["a"]), encoding="utf-8")
    pb.write_text(json.dumps(doc["b"]), encoding="utf-8")
    assert C.main(["--current", str(pb), "--baseline", str(pa),
                   "--floor-record", str(ROOT / "eval" / "noise_floor_tool_choice.yaml")]) == 0
    out = capsys.readouterr().out
    assert "no movement beyond the per-metric noise floor" in out


# ── clause 3: what the instrument can certify, and the one-flip rule ────────

def test_one_flipped_query_is_never_a_movement_for_any_metric():
    """Clause 3, per metric, at that metric's own denominator.

    For every metric: a delta of exactly one query's worth (1/n, the largest
    single flip can do) must not be a regression. The old 0.051 sat *at* 1/20,
    so one coin landing differently on a 20-query set was reportable.
    """
    for metric, higher_is_better in C.METRICS.items():
        n = C.denominators(_run()).get(metric)
        assert n, metric
        one = 1.0 / n
        base = 0.9 if higher_is_better else 0.1
        moved = base - one if higher_is_better else base + one
        r = C.compare(_run(**{metric: round(moved, 6)}), _run(**{metric: base}),
                      observed={})
        assert r.regressions == [], (metric, n, r.regressions)
        assert r.failures == [], (metric, n, r.failures)


def test_output_states_the_smallest_movement_it_can_certify(capsys):
    """Clause 3: n-derived floor of certifiable movement, in the output."""
    r = C.compare(NOISE_B, NOISE_A)
    metric, floor = C.certifiable_floor(
        {m: b["floor"] for m, b in r.floors.items()})
    assert floor == min(b["floor"] for b in r.floors.values())
    rep = C._floor_report(r, {}, False)
    joined = "\n".join(rep)
    assert f"smallest movement this comparison can certify: +{floor:.3f}" in joined
    assert f"(floor of `{metric}`)" in joined
    assert "n=20" in joined
    # and it names the one-flip guarantee per metric, with n
    assert "single flipped query is never a movement" in joined
    assert "1/6=" in joined and "1/20=" in joined


def test_a_two_query_flip_is_reported_as_not_certifiable_at_this_n():
    """The honest consequence of clause 3, replacing the retracted old claim.

    0.10 on 20 queries is inside 2σ, so this instrument cannot call it. The
    output has to say the floor out loud rather than let the reader assume the
    threshold was meaningful — raising n is the fix, never lowering the floor.
    """
    r = C.compare(_run(correct_rate=0.80), _run(correct_rate=0.90), observed={})
    assert r.regressions == []
    assert r.floors["correct_rate"]["floor"] > 0.10, r.floors["correct_rate"]


# ── clause 4: a control move is instrument failure, not regression ───────────

def test_a_control_move_is_an_instrument_failure_not_a_regression():
    """The control rows are ones the prompt surface cannot reach."""
    r = C.compare(_run(control_correct_rate=0.5), _run(control_correct_rate=1.0))
    assert any("control_correct_rate" in x for x in r.failures)
    assert not any("control_correct_rate" in x for x in r.regressions)


def test_control_failure_and_regression_are_distinguishable():
    """A round can be regressing AND have a broken instrument; the two lists
    are separate so exit 3 does not erase the regression and vice versa."""
    r = C.compare(_run(control_correct_rate=0.5, correct_rate=0.40), _run())
    assert r.failures and r.regressions


def test_cli_exits_3_for_a_control_move(tmp_path, capsys):
    """Clause 4: its own exit reason, so the gate cannot read it as a regression."""
    a = tmp_path / "before-1.json"
    b = tmp_path / "after-2.json"
    a.write_text(json.dumps(_run(control_correct_rate=1.0)), encoding="utf-8")
    b.write_text(json.dumps(_run(control_correct_rate=0.5)), encoding="utf-8")
    assert C.main(["--current", str(b), "--baseline", str(a)]) == 3
    out = capsys.readouterr().out
    assert "INSTRUMENT FAILURE (exit 3)" in out
    assert "REGRESSED" not in out
    assert "certified nothing" in out          # says what to do: re-measure


def test_a_control_move_inside_the_floor_is_not_an_instrument_failure():
    """0.167 is one flipped control query of 6 — noise, not a broken instrument."""
    r = C.compare(_run(control_correct_rate=0.833), _run(control_correct_rate=1.0))
    assert r.failures == []


def test_no_instrument_failure_means_no_exit_3(tmp_path, capsys):
    a = tmp_path / "before-1.json"
    b = tmp_path / "after-2.json"
    a.write_text(json.dumps(NOISE_A), encoding="utf-8")
    b.write_text(json.dumps(NOISE_B), encoding="utf-8")
    capsys.readouterr()                       # drain the fixture's own prints
    assert C.main(["--current", str(b), "--baseline", str(a)]) == 0
    assert "INSTRUMENT FAILURE" not in capsys.readouterr().out


# ── denominators and the split ───────────────────────────────────────────────

def test_denominators_count_the_web_and_control_split_from_the_records():
    d = C.denominators(_run())
    assert d["correct_rate"] == 20
    assert d["http_tool_first_rate"] == 14
    assert d["control_correct_rate"] == 6
    assert d["__split_known__"] is True


def test_an_artifact_without_records_says_the_split_is_unknown():
    """Otherwise a caller silently prices the control floor over the wrong n."""
    d = C.denominators({"n_queries": 20})
    assert d["control_correct_rate"] == 20
    assert d["__split_known__"] is False


def test_cli_names_the_unknown_split_risk(tmp_path, capsys):
    """A too-narrow control floor makes an instrument failure HARDER to see."""
    a = tmp_path / "old-1.json"
    b = tmp_path / "new-2.json"
    a.write_text(json.dumps({"n_queries": 20,
                             "summary": {"overall": {"correct_rate": 0.9}}}),
                 encoding="utf-8")
    b.write_text(json.dumps({"n_queries": 20,
                             "summary": {"overall": {"correct_rate": 0.9}}}),
                 encoding="utf-8")
    C.main(["--current", str(b), "--baseline", str(a)])
    assert "control floor is too NARROW" in capsys.readouterr().out


# ── arithmetic ───────────────────────────────────────────────────────────────

def test_binomial_sigma_matches_the_closed_form():
    assert C.binomial_sigma(0.9, 20) == pytest.approx(0.0671, abs=1e-3)
    assert C.binomial_sigma(0.5, 20) == pytest.approx(0.1118, abs=1e-3)


def test_a_metric_that_has_never_been_seen_still_costs_variance():
    """p=0 does not mean σ=0 — the estimator is clamped at 1/n and says so.

    `shelled_to_web_rate` at 0.000 is the case that matters: a rate sitting at
    zero looks like the quietest metric on the board, and a floor of 0 there
    would make the FIRST instance of the exact behaviour the eval exists to
    catch read as a certain regression.
    """
    assert C.binomial_sigma(0.0, 20) > 0.0
    # clamped at p_eff = 1/20
    assert C.binomial_sigma(0.0, 20) == pytest.approx(
        (0.05 * 0.95 / 20) ** 0.5, abs=1e-9)
    floor = C.noise_floor("shelled_to_web_rate", 0.0, 0.05, 20, {})[0]
    assert floor > 0.05                        # one first-ever case is not a finding


def test_a_worse_rate_cannot_buy_a_narrower_band():
    """The binomial term uses the WORSE of the two values, so an improved
    metric does not get to certify its own improvement on a tighter threshold."""
    improved = C.noise_floor("correct_rate", 0.70, 0.95, 20, {})[0]
    degraded = C.noise_floor("correct_rate", 0.95, 0.70, 20, {})[0]
    assert improved == degraded


# ── the floor record ─────────────────────────────────────────────────────────

def test_measure_floor_reads_only_same_tree_pairs(tmp_path):
    """`-a`/`-b` with identical recorded config is what makes it the SAME tree."""
    a = tmp_path / "x-a-20260913-120000.json"
    b = tmp_path / "x-b-20260913-120130.json"
    c = tmp_path / "y-only-20260913-120200.json"
    cfg = {"system_prompt_chars": 43000, "model": "qwen"}
    a.write_text(json.dumps({**NOISE_A, "config": cfg}), encoding="utf-8")
    b.write_text(json.dumps({**NOISE_B, "config": cfg}), encoding="utf-8")
    c.write_text(json.dumps({**_run(correct_rate=0.10),
                             "config": {"system_prompt_chars": 1}}), encoding="utf-8")
    rec = C.measure_floor(tmp_path, tmp_path / "floor.yaml")
    assert len(rec["pairs"]) == 1
    assert rec["metrics"]["correct_rate"]["observed_spread"] == pytest.approx(0.05)
    assert rec["metrics"]["http_tool_first_rate"]["observed_spread"] == pytest.approx(0.071)
    assert "y-only" not in json.dumps(rec)            # unpaired: measures nothing


def test_measure_floor_drops_an_errored_run(tmp_path):
    """An errored run measured the harness, not the model; its 'spread' is junk."""
    a = tmp_path / "e-a-20260913-120000.json"
    b = tmp_path / "e-b-20260913-120130.json"
    cfg = {"system_prompt_chars": 1}
    a.write_text(json.dumps({**_run(correct_rate=0.9), "config": cfg}), encoding="utf-8")
    b.write_text(json.dumps({**_run(correct_rate=0.1, errors=7), "config": cfg}),
                 encoding="utf-8")
    rec = C.measure_floor(tmp_path, tmp_path / "floor.yaml")
    assert rec["pairs"] == []


def test_the_committed_floor_record_is_valid_and_nonempty():
    """The record in the repo is what the gate's runs decide against.

    If it were missing, every floor would silently collapse to binomial-only
    and the reader would only learn that from a line in the output — so the
    file's presence and its two #691 numbers are pinned here.
    """
    import pathlib

    p = pathlib.Path(__file__).resolve().parent.parent / "eval" / "noise_floor_tool_choice.yaml"
    assert p.exists(), f"{p} missing — force-add it; *.json/yaml is gitignored"
    rec = C.load_floor_record(p)
    obs = C.observed_spreads(rec)
    assert obs["correct_rate"] == pytest.approx(0.05)
    assert obs["http_tool_first_rate"] == pytest.approx(0.071)
    assert rec["binomial_sigmas"] == C.BINOMIAL_SIGMAS


# ── the CLI contract that already existed ────────────────────────────────────

@pytest.fixture
def baseline_dir(tmp_path, monkeypatch):
    d = tmp_path / "tool-choice"
    d.mkdir(parents=True)
    monkeypatch.setattr(C, "BASELINE_DIR", d)
    return d


def _write(d, name, run, mtime):
    p = d / name
    p.write_text(json.dumps(run), encoding="utf-8")
    import os
    os.utime(p, (mtime, mtime))
    return p


def test_cli_compares_the_two_newest_runs(baseline_dir, capsys):
    _write(baseline_dir, "old.json", _run(correct_rate=0.90), 1000)
    _write(baseline_dir, "new.json", _run(correct_rate=0.40), 2000)
    assert C.main([]) == 1
    assert "REGRESSED" in capsys.readouterr().out


def test_cli_skips_loudly_with_only_one_run(baseline_dir, capsys):
    """Exit 2, not 0: 'nothing to compare' must not read as 'nothing wrong'."""
    _write(baseline_dir, "only.json", _run(), 1000)
    assert C.main([]) == 2
    assert "not a pass" in capsys.readouterr().out


def test_cli_label_selects_the_run_under_test(baseline_dir, capsys):
    _write(baseline_dir, "prior-1.json", _run(correct_rate=0.90), 1000)
    _write(baseline_dir, "mine-2.json", _run(correct_rate=0.95), 2000)
    assert C.main(["--label", "mine"]) == 0
    out = capsys.readouterr().out
    assert "current : mine-2.json" in out and "baseline: prior-1.json" in out


def test_cli_says_so_when_the_label_has_no_prior(baseline_dir, capsys):
    _write(baseline_dir, "mine-1.json", _run(), 1000)
    _write(baseline_dir, "mine-2.json", _run(), 2000)
    assert C.main(["--label", "mine"]) == 2
    assert "no prior" in capsys.readouterr().out


def test_cli_flags_a_differing_query_count(baseline_dir, capsys):
    a = _run(); a["n_queries"] = 20
    b = _run(); b["n_queries"] = 8
    _write(baseline_dir, "old.json", b, 1000)
    _write(baseline_dir, "new.json", a, 2000)
    C.main([])
    assert "not the same test" in capsys.readouterr().out
