"""Tests for the eval instrument index (#2597).

The item's claim is an aggregation, not a new measurement: one dated row over the
instruments that already emit dated artifacts, tiered, weighted, and with absence
excluded from the denominator rather than scored as a pass. Every test below is
hermetic — each instrument's artifact is a `tmp_path` fixture handed in through
`root_overrides`, the series is written into a `tmp_path` stand-in for the data
root, and the clock is pinned to 2026-10-13 so "fresh against a 1-day cadence"
means the same thing on every night the suite runs. Nothing here creates a file
under the tree it is testing, the published series included: the CLI tests launch
`python -m scripts.eval_instrument_index` as a subprocess with `--roots` pointing
at fixture stores and `--out-dir` at `tmp_path`, so no test run appends a row to
the real series — the one row that was in it before this round had been written by
a test that did exactly that.

The report-only rule is proven from three directions, and each of the three
carries a positive control, because an absence test with nothing to catch is
exactly how the previous version of this file passed while never opening the
files a violation would live in: the scanner is shown one real read and one
docstring mention, the enumeration is shown the named files it must reach and
checked against `git ls-files`, and the import probe is run once with the
publisher imported so that its `False` elsewhere is a verdict and not an inability
to see.

The table grew from seven arms to eight with #3541, which added
`gate_to_land_lag` — the gate-pass→promote p50 the #101 job appends to
`gate_to_land_lag.jsonl`. A latency is not a share, so that arm is carried in the
published table with its own unit and `averageable=False`, exactly as
`funnel_census` already is, and it carries weight 0.0. The exclusion is proven by
arithmetic rather than asserted:
`test_the_lag_arm_is_carried_in_the_table_and_excluded_from_both_totals` builds the
row twice over the same fixtures, once with the arm and once without, and requires
both composites to come out identical — so a future edit that let seconds into the
percentage total fails there whatever the entry's own flags say.

The five clause tests are numbered to match the item's acceptance clauses:

1  `test_run_appends_one_dated_row_covering_eight_components_with_every_declared_field`
2  `test_an_absent_instrument_is_recorded_missing_and_enters_neither_numerator_nor_denominator`
   + `test_with_all_but_one_instrument_missing_the_total_refuses_rather_than_reading_100_percent`
3  `test_the_declared_weight_and_equal_weight_totals_differ_under_non_uniform_weights`
   + `test_a_sign_disagreement_between_the_two_totals_publishes_the_weights_decide_headline`
4  `test_tier_two_requires_a_named_consumer_that_compares_rows`
   + `test_the_same_fresh_component_with_the_consumer_removed_scores_tier_one`
5  `test_each_component_path_is_the_path_its_own_writer_declares`
   + `test_the_state_dir_store_and_the_in_tree_store_both_appear_with_real_newest_dates`

Run: /home/alansrobotlab/lloyd/.venvs/lloyd/bin/python -m pytest \
        tests/test_eval_instrument_index.py -q
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts import eval_instrument_index as EII  # noqa: E402
from scripts.maintenance import gate_to_land_lag as GTL  # noqa: E402

#: The pinned clock. Every seeded artifact below is dated 2026-10-12 (one day
#: old) or earlier, so "arriving on cadence" is decided by the code, not by the
#: date the suite happened to run on.
NOW = datetime(2026, 10, 13, 3, 40, tzinfo=timezone.utc)
FRESH = "2026-10-12"

ALL_KEYS = ("retrieval_nightly", "iv_metrics", "automod_scorecard",
            "landed_outcomes", "tool_quality", "uptake_probe", "funnel_census",
            "gate_to_land_lag")


def _instr(key: str):
    return next(i for i in EII.INSTRUMENTS if i.key == key)


def _component(row: dict, key: str) -> dict:
    return next(c for c in row["components"] if c["key"] == key)


# ── artifacts shaped exactly as each writer emits them ────────────────────────
#
# These builders are positive controls for the value pointers: a pointer that
# drifted from the writer's real document shape fails here rather than publishing
# a `no-value` row forever.

def _seed_json(store: Path, name: str, doc: dict) -> Path:
    store.mkdir(parents=True, exist_ok=True)
    (store / name).write_text(json.dumps(doc) + "\n", encoding="utf-8")
    return store / name


def _seed_series(store: Path, name: str, rows: list[dict]) -> Path:
    store.mkdir(parents=True, exist_ok=True)
    (store / name).write_text("".join(json.dumps(r) + "\n" for r in rows),
                              encoding="utf-8")
    return store / name


def _baseline_doc(rate: float, ci: list[float] | None = None, n: int = 74) -> dict:
    overall: dict = {"n_queries": 81, "doc_hit_rate": rate}
    if ci is not None:
        overall["ci95"] = {"doc_hit_rate": {"ci": ci, "n": n, "kind": "wilson"}}
    return {"label": "nightly-20261012", "summary": {"overall": overall}}


def _seed(store: Path, key: str, day: str, value) -> Path:
    """Write one instrument's artifact the way its own writer writes it."""
    if key == "retrieval_nightly":
        compact = day.replace("-", "")
        return _seed_json(store, f"nightly-{compact}-{compact}-060351.json",
                          _baseline_doc(value, [value - 0.06, value + 0.06]))
    if key == "iv_metrics":
        return _seed_series(store, "iv-metrics.jsonl", [{
            "recorded_at": f"{day}T08:00:52+00:00", "dropped_rate": value,
            "llm_calls": 754, "observations": 1420}])
    if key == "automod_scorecard":
        return _seed_series(store, "scorecard.jsonl", [{
            "computed_at": f"{day}T20:00:00+00:00", "since_days": 7.0,
            "acceptance": {"hit_rate": value, "landed": 26, "met": 7,
                           "with_outcome": 14}}])
    if key == "landed_outcomes":
        # `scripts/automod/landed_outcomes.py::rate` publishes a share as a RECORD,
        # never a string — {k, n, rate, ci_low, ci_high} with Wilson bounds over the
        # same n it prints — and `by_author.<arm>.later_undo_7d` stores that dict.
        # `app/tool_quality.py::rate` is the one that emits the "k / n" STRING, and
        # the tool_quality fixture below is where that form is seeded.
        k = int(round(value * 40))
        # Wilson bounds cannot leave [0,1]; this naive +/- has to be clamped, or a
        # fixture at rate 0.0 (the clause-3 weights do use one) publishes -0.05 as
        # the lower bound of a share — a document no writer emits, and one an
        # interval-validating reader would be right to reject.
        undo = {"k": k, "n": 40, "rate": round(value, 4),
                "ci_low": round(max(0.0, value - 0.05), 4),
                "ci_high": round(min(1.0, value + 0.05), 4)}
        assert 0.0 <= undo["ci_low"] <= undo["rate"] <= undo["ci_high"] <= 1.0, undo
        row = {"generated_at": f"{day}T08:00:00+00:00",
               "by_author": {"agent": {"commits": 40, "later_undo_7d": undo}}}
        return _seed_series(store, "landed_outcomes.jsonl", [row])
    if key == "tool_quality":
        # `app/tool_quality.py::rate` refuses to print a bare count: "k / n".
        return _seed_json(store, f"{day}.json", {
            "schema": 1, "date": day,
            "corpus": {"total_calls": 300, "total_empty": int(value * 300)},
            "corpus_rates": {"empty": f"{int(value * 300)} / 300"}})
    if key == "uptake_probe":
        return _seed_json(store, f"uptake-{day}.json", {
            "probe_timestamp": f"{day}T08:01:21+00:00", "item": "552",
            "classifier": {"precision": value, "n_positives": 23, "tp": 13,
                           "fp": 0}})
    if key == "gate_to_land_lag":
        # The recorder's OWN row shape, as `scripts/maintenance/gate_to_land_lag.py`
        # appends it: `p50`/`p90`/`p99` in SECONDS at the top level beside the `n`
        # rounds they were computed over, stamped by `generated_at`. The numbers are
        # the live journal's (2026-10-29 row, 1509 rounds), because a pointer that
        # drifted from the writer's real document has to fail here, not publish a
        # `no-value` arm forever.
        return _seed_series(store, GTL.RECORD_NAME, [{
            "generated_at": f"{day}T08:06:26+00:00", "n": 1509,
            "p50": value, "p90": 147.32102589607243, "p99": 3315.6,
            "min": 0.41, "max": 4500.0, "over_bucket": 118, "over_ceiling": 0,
            "ceiling_s": 4500.0, "bucket_s": 600.0, "ledger_lines": 64147,
            "gate_pass_rows": 24899, "promoted_rows": 1509,
            "rounds_only_in_archives": 1093, "gate_without_promoted": 7,
            "ledger_files": 3, "state_dir": str(GTL.record_path().parent)}])
    if key == "funnel_census":
        return _seed_json(store, f"funnel-{day}.json", {
            "artifact": "funnel-census", "version": 1, "item": "#2507",
            "measured_at": f"{day}T09:00:00+00:00",
            "stages": [{"stage": "produced", "measured": True, "unit": "artifacts",
                        "k": 90, "n": 100, "rate": 0.9, "ci95": [0.82, 0.94]},
                       {"stage": "surfaced", "measured": True, "unit": "rows",
                        "k": 30, "n": 50, "rate": value, "ci95": [0.46, 0.72]}]})
    raise AssertionError(f"no fixture builder for {key}")


def _seed_all(tmp_path: Path, values: dict, *, omit: tuple[str, ...] = (),
              days: dict | None = None) -> dict:
    """Root overrides for the real INSTRUMENTS tuple, one tmp store per key."""
    days = days or {}
    roots: dict[str, Path] = {}
    for key in ALL_KEYS:
        store = tmp_path / "roots" / key
        if key in omit:
            # The declared location exists in the table and holds nothing, which
            # is the real shape of a dead instrument: a store the writer never
            # reached, not a path nobody declared.
            roots[key] = tmp_path / "declared-but-empty" / key
            continue
        _seed(store, key, days.get(key, FRESH), values[key])
        roots[key] = store
    return roots


def _row(tmp_path: Path, values: dict, *, omit: tuple[str, ...] = (),
         instruments=None, days: dict | None = None, prior_rows: int = 0,
         extra_rows: list[dict] | None = None):
    out = tmp_path / "index"
    out.mkdir(parents=True, exist_ok=True)
    for row in (extra_rows or []):
        EII.append_row(row, out)
    assert len(EII.read_series(out)) == len(extra_rows or [])
    assert len(extra_rows or []) == prior_rows, "the fixture and its count agree"
    row = EII.build_row(instruments or EII.INSTRUMENTS, now=NOW,
                        root_overrides=_seed_all(tmp_path, values, omit=omit,
                                                 days=days),
                        out_dir=out, code_root=tmp_path / "code")
    path = EII.append_row(row, out)
    return row, path, out


def _prior_row(declared: float, equal: float, delta_declared: float,
               delta_equal: float, *, date: str = "2026-10-12",
               qualities: dict | None = None) -> dict:
    """A published prior row, already read: the shape `delta_vs_prior` is taken
    against, and the history the flat-headline window is cut from."""
    return {"schema": 1, "date": date, "generated_at": f"{date}T03:40:00+00:00",
            "totals": {"status": "ok", "declared_weights": declared,
                       "equal_weights": equal, "scored": 6, "of": 7},
            "delta_vs_prior": {"declared_weights": delta_declared,
                               "equal_weights": delta_equal},
            "components": [{"key": k,
                            "quality": (declared if qualities is None
                                        else qualities.get(k))} for k in ALL_KEYS]}


GOOD_VALUES = {
    "retrieval_nightly": 0.689,      # up is better
    "iv_metrics": 0.0451,            # down is better -> quality 0.955
    "automod_scorecard": 0.5,        # up
    "landed_outcomes": 0.075,        # down is better -> quality 0.925
    "tool_quality": 0.04,            # down is better -> quality 0.96
    "uptake_probe": 1.0,             # up
    "funnel_census": 0.6,            # share, reported not averaged
    "gate_to_land_lag": 17.687,      # SECONDS, reported not averaged (#3541)
}


#: The one arm published with weight 0.0: `gate_to_land_lag` is carried in the
#: table and excluded from the composite (#3541), so the "every weight is
#: positive" sweep below has to name it rather than silently pass it.
ZERO_WEIGHT_KEYS = ("gate_to_land_lag",)


# ── clause 1: one dated row, >=6 components, every declared field ─────────────

def test_run_appends_one_dated_row_covering_eight_components_with_every_declared_field(tmp_path):
    """Clause 1 (#2597, grown to eight arms by #3541): running the publisher appends
    ONE dated row under the data root covering >=6 components, each carrying its
    instrument path, newest-artifact date, tier, weight, unit and latest value."""
    row, path, out = _row(tmp_path, GOOD_VALUES)

    assert path == out / "instrument-index.jsonl"
    rows = EII.read_series(out)
    assert len(rows) == 1, "one run appends exactly one row"
    assert row["date"] == "2026-10-13"
    assert row["item"] == "#2597"

    components = row["components"]
    assert len(components) == 8, "the table covers eight instruments, not seven"
    assert len(components) >= 6
    assert sorted(c["key"] for c in components) == sorted(ALL_KEYS)
    for c in components:
        assert c["instrument_path"], c["key"]
        assert c["tier"] in (0, 1, 2), c
        if c["key"] in ZERO_WEIGHT_KEYS:
            assert c["weight"] == 0.0, c
        else:
            assert 0.0 < c["weight"] <= 1.0, c
        assert c["unit"], c
        assert c["weight_why"], c
        assert c["status"], c
        # The fresh arms carry a latest value; the seeded date is the one
        # written above, read back off the file name or the row's own timestamp.
        assert c["newest_artifact"] == FRESH, c
        assert c["value"] is not None, c
    assert row["weights_sum_declared"] == 1.0

    # The interval rides along where the source carries one (#696), and the
    # funnel's unit comes out of the census's own stage denominator field.
    retrieval = _component(row, "retrieval_nightly")
    assert retrieval["ci95"] == pytest.approx([0.629, 0.749], abs=1e-3)
    assert retrieval["denominator"] == 74
    assert _component(row, "funnel_census")["unit"] == "rows"

    # #3541: the lag arm reports the recorder's own percentile, in seconds, over
    # the recorder's own `n` — a value the table shows and no total divides.
    lag = _component(row, "gate_to_land_lag")
    assert lag["value"] == pytest.approx(17.687)
    assert lag["unit"] == "seconds"
    assert lag["denominator"] == 1509
    assert lag["polarity"] == "down"
    assert lag["weight"] == 0.0 and lag["contributes"] is False
    assert lag["in_total_denominator"] is False

    # The two totals and the delta against the prior row are in the row itself.
    assert row["totals"]["status"] == "ok"
    assert row["totals"]["declared_weights"] is not None
    assert row["totals"]["equal_weights"] is not None
    assert row["delta_vs_prior"]["declared_weights"] is None, "no prior row yet"
    assert row["history_rows"] == 0


def test_the_lag_arm_reports_its_own_percentile_in_seconds_over_its_own_n(tmp_path):
    """#3541 clause 2: the arm reads `p50` out of the recorder's own newest row and
    reports it in seconds over that row's own `n`. The seeded value is 17.687 and
    the denominator 1509 — both the live journal's 2026-10-29 numbers — so a
    `value_path` or `n_path` that drifted from the recorder's real row shape fails
    here rather than publishing an empty arm.

    The status is `not-a-rate`, and that is the module's honest verdict rather than a
    defect to work around: `read_instrument` refuses any value outside 0..1, which is
    exactly what a latency of 17.687 seconds is. `funnel_census` arrives the same way
    on live data. What the arm must still publish is the number, its unit and the
    round count it was computed over — an excluded component that printed nothing
    would be indistinguishable from a dead one.

    The second half is the part a zero could fake: with the journal absent the arm
    reads `missing` with `value` None. A publisher that defaulted a missing
    percentile to 0.0 would publish a perfect 0-second gate-to-land lag on the day
    the recorder died — and `down` polarity would score that as quality 1.0."""
    row, _, _ = _row(tmp_path, GOOD_VALUES)
    lag = _component(row, "gate_to_land_lag")
    assert lag["status"] == "not-a-rate", (
        "a p50 of 17.687 s is outside 0..1 by definition, so the module's share "
        f"sanity check refuses it by design; got {lag['status']!r}")
    assert lag["value"] == pytest.approx(17.687), (
        "the value is published even though it is not scored")
    assert lag["unit"] == "seconds"
    assert lag["denominator"] == 1509, "the percentile is quoted over the recorder's n"
    assert "may not enter the total" in lag["detail"], lag["detail"]

    absent, _, _ = _row(tmp_path / "absent", GOOD_VALUES, omit=("gate_to_land_lag",))
    missing = _component(absent, "gate_to_land_lag")
    assert missing["status"] == "missing", (
        "a journal that is not there must read as absence, never as a lag of zero")
    assert missing["value"] is None and missing["quality"] is None
    assert missing["tier"] == 0, (
        "0.0 seconds would score a perfect 1.0 under `down` polarity, so absence has "
        "to arrive as absence — and an absent arm cannot be tier 2 either")


def test_the_lag_arm_is_carried_in_the_table_and_excluded_from_both_totals(tmp_path):
    """#3541 clause 3, the ruling made arithmetic: a p50 in seconds is published in
    the table and contributes nothing to either composite.

    The precedent is `funnel_census`, and the mechanism is the one it already uses
    — `weight=0.0` plus `averageable=False` — not a second exclusion. The check is
    a difference, not a reading of the flags: the same seven fixtures are published
    twice, once through the whole eight-arm table and once with the lag arm taken
    out of it, and the two composites must be equal to the digit. A `seconds` value
    that ever reached a weighted total would move that number by roughly 4 points
    (17.687 x 0.05 of weight against a ~0.6 composite), which is why equality here
    is the proof rather than the entry's own `weight=0.0`."""
    with_lag, _, _ = _row(tmp_path, GOOD_VALUES)
    without = tuple(i for i in EII.INSTRUMENTS if i.key != "gate_to_land_lag")
    assert len(without) == len(EII.INSTRUMENTS) - 1, (
        "positive control: the arm is in the table, so the reduced run is a real "
        "subtraction and not a no-op that would make the equality below vacuous")
    seven, _, _ = _row(tmp_path / "without", GOOD_VALUES, instruments=without)

    assert with_lag["totals"]["declared_weights"] == pytest.approx(
        seven["totals"]["declared_weights"])
    assert with_lag["totals"]["equal_weights"] == pytest.approx(
        seven["totals"]["equal_weights"])
    assert with_lag["totals"]["scored"] == seven["totals"]["scored"], (
        "the lag arm is reported, not scored: the denominator of both totals "
        "cannot move when it is added")

    keys = [c["key"] for c in with_lag["components"]]
    assert "gate_to_land_lag" in keys, "the published table still carries the row"
    assert len(keys) == 8
    lag = _component(with_lag, "gate_to_land_lag")
    assert lag["weight"] == 0.0
    assert lag["averageable"] is False
    assert lag["contributes"] is False and lag["in_total_denominator"] is False
    assert with_lag["weights_sum_declared"] == pytest.approx(1.0), (
        "adding the arm cannot have re-scaled the seven weighted arms")
    excluded = {e["key"]: e for e in with_lag["totals"]["excluded"]}
    assert "gate_to_land_lag" in excluded, (
        "the exclusion has to be PRINTED, the same way the funnel's is — a silent "
        "omission is indistinguishable from a component that never arrived")
    assert excluded["gate_to_land_lag"]["weight"] == 0.0, (
        "the excluded block prints the weight sum it would have carried, so a "
        "non-zero weight here would be seconds entering the total")

    # The printed form, not just the JSON: `--print` is the only form a human
    # reads. The arm's own line carries its value and its unit, and it is named in
    # the excluded block — which is what makes "carried but not averaged" legible
    # rather than a reader having to diff the two totals.
    text = EII.render(with_lag)
    lag_line = next((ln for ln in text.splitlines()
                     if ln.startswith("gate_to_land_lag  ")), None)
    assert lag_line is not None, "the printed table omits the lag arm entirely"
    assert "seconds" in lag_line and "17.687" in lag_line and "1509" in lag_line, lag_line
    assert "0.00" in lag_line, f"the weight column must print 0.00: {lag_line}"
    assert "excluded from the total: gate_to_land_lag" in text, (
        "the printed exclusions must name the arm beside the funnel's")

    # And the case where `averageable=False` is the ONLY rail doing the work.
    # 17.687 s is caught by the share sanity check before any flag is consulted, so
    # a lag arm seeded at 17.687 alone would leave the flag untested; a fast day
    # whose p50 is 0.5 s passes that check and reads `measured`. There it must still
    # contribute nothing — which is the flag, exactly as `funnel_census` is pinned.
    fast = dict(GOOD_VALUES, gate_to_land_lag=0.5)
    fast_row, _, _ = _row(tmp_path / "fast", fast)
    fast_lag = _component(fast_row, "gate_to_land_lag")
    assert fast_lag["status"] == "measured", (
        "the fixture for this half must be a lag a share check would let through, "
        f"got {fast_lag['status']!r}")
    assert fast_lag["quality"] == pytest.approx(0.5), (
        "a quality IS computed for it — which is precisely why the exclusion cannot "
        "rest on the value check, and is what `funnel_census` shows too")
    assert fast_lag["contributes"] is False and fast_lag["in_total_denominator"] is False
    fast_without, _, _ = _row(tmp_path / "fast-without", fast, instruments=without)
    assert fast_row["totals"]["declared_weights"] == pytest.approx(
        fast_without["totals"]["declared_weights"])
    assert fast_row["totals"]["equal_weights"] == pytest.approx(
        fast_without["totals"]["equal_weights"])


def test_the_series_default_lives_under_the_data_root():
    """The publisher's own artifact is a store like every other one it reads."""
    from app import paths

    rel = paths.EVAL_INSTRUMENT_INDEX_DIR.relative_to(paths.DATA_ROOT)
    assert rel == Path("eval") / "instrument-index"


def test_a_component_with_a_short_series_reports_insufficient_series_not_flat(tmp_path):
    """`scorecard.jsonl` holds 2 rows and `uptake` 3 today, so a short series may
    not be read as a flat line: the answer is `insufficient-series`, per the
    flat-headline rule's own window of 4 rows."""
    row, _, _ = _row(tmp_path, GOOD_VALUES,
                     extra_rows=[_prior_row(0.6, 0.5, 0.01, 0.01)] * 2,
                     prior_rows=2)
    assert row["history_rows"] == 2
    for key in ("automod_scorecard", "uptake_probe"):
        assert _component(row, key)["flat_check"] == "insufficient-series(3 of 4 rows)"
        assert "vanity" not in _component(row, key)["flat_check"]


def test_a_component_flat_for_four_rows_while_the_composite_moved_is_named(tmp_path):
    """The vanity-metric detector: Druid's punchline is a vanity dashboard flat at
    92% all year while the weekly score fell 83% to 60%, so a component flat across
    the whole 4-row window while the composite moved is named as a candidate. The
    window has to be deep enough first: 3 priors plus this row is 4 values."""
    priors = [_prior_row(declared=0.5, equal=0.5, delta_declared=0.01,
                         delta_equal=0.01, date=f"2026-10-{11 - i:02d}",
                         qualities={k: (0.5 if k == "uptake_probe" else 0.4)
                                    for k in ALL_KEYS})
                for i in range(3)]
    # uptake holds at 0.5 in every prior row and again in this one, so its series
    # is four identical values; every other arm moves.
    values = dict(GOOD_VALUES, uptake_probe=0.5)
    row, _, _ = _row(tmp_path, values, extra_rows=priors, prior_rows=3)

    assert row["history_rows"] == 3
    assert row["delta_vs_prior"]["declared_weights"] not in (None, 0), (
        "the composite has to move for a flat component to be a finding")
    uptake = _component(row, "uptake_probe")
    assert uptake["quality"] == 0.5
    assert uptake["flat_check"].startswith("flat over 4 rows while the composite moved")
    assert "candidate vanity metric" in uptake["flat_check"]
    assert _component(row, "automod_scorecard")["flat_check"] == "moving"


# ── clause 2: absence excludes, never penalises ───────────────────────────────

def test_an_absent_instrument_is_recorded_missing_and_enters_neither_numerator_nor_denominator(tmp_path):
    """Clause 2: a component whose declared artifact path does not exist is
    recorded `missing` and contributes to neither the numerator nor the
    denominator — it can neither add a pass nor dilute one."""
    row, _, _ = _row(tmp_path, GOOD_VALUES, omit=("tool_quality",))

    missing = _component(row, "tool_quality")
    assert missing["status"] == "missing"
    assert missing["tier"] == 0, "an instrument with no artifact is tier 0"
    assert missing["value"] is None and missing["quality"] is None
    assert missing["contributes"] is False
    assert missing["in_total_denominator"] is False
    assert missing["instrument_path"] == str(tmp_path / "declared-but-empty"
                                             / "tool_quality")

    totals = row["totals"]
    assert totals["scored"] == 5, "six measured, but funnel_census is not averageable"
    assert [e["key"] for e in totals["excluded"] if e["status"] == "missing"] == \
        ["tool_quality"]
    # 1.0 minus the missing arm (0.10) and the non-averageable share (0.05):
    # the renormalisation is printed, not implied by a changed denominator.
    assert totals["renormalised_weight_sum"] == pytest.approx(0.85)
    assert not any(e["key"] == "tool_quality" and e["status"] == "measured"
                   for e in totals["excluded"])


def test_an_absent_instrument_cannot_contribute_a_pass_even_at_a_perfect_score(tmp_path):
    """The negative control the clause asks for: an instrument that is absent
    cannot be the reason a total reads 1.0."""
    values = dict(GOOD_VALUES, tool_quality=0.0)   # 0.0 empty-returns = perfect
    row, _, _ = _row(tmp_path, values, omit=("tool_quality",))
    assert _component(row, "tool_quality")["quality"] is None
    assert row["totals"]["declared_weights"] < 1.0


def test_with_all_but_one_instrument_missing_the_total_refuses_rather_than_reading_100_percent(tmp_path):
    """Clause 2's fixture: all but one instrument missing, and the one that
    survives scores 100% — the total must refuse, not publish that as a composite.
    Seven of the eight are omitted (#3541 grew the table), the surviving one being
    `retrieval_nightly`."""
    row, _, _ = _row(tmp_path, {"retrieval_nightly": 1.0},
                     omit=("iv_metrics", "automod_scorecard", "landed_outcomes",
                           "tool_quality", "uptake_probe", "funnel_census",
                           "gate_to_land_lag"))

    totals = row["totals"]
    assert totals["status"] == "refused"
    assert totals["declared_weights"] is None
    assert totals["equal_weights"] is None
    assert totals["scored"] == 1, "one survivor, and it scores a perfect 1.0"
    assert totals["min_scored"] == EII.MIN_SCORED_COMPONENTS
    assert "refuses" in row["headline"]
    assert "1.0" not in json.dumps({"declared": totals["declared_weights"]})
    assert len(row["dead_or_stale"]) == 7
    assert {d["status"] for d in row["dead_or_stale"]} == {"missing"}


def test_a_stale_instrument_is_excluded_and_named_with_its_declared_cadence(tmp_path):
    """A component older than its own declared cadence contributes `stale` and
    leaves the denominator, with the renormalisation visible."""
    row, _, _ = _row(tmp_path, GOOD_VALUES, days={"automod_scorecard": "2026-09-11"})

    stale = _component(row, "automod_scorecard")
    assert stale["status"] == "stale"
    assert stale["age_days"] == 32
    assert stale["tier"] == 1, "rows are not arriving, so nothing is tracked over time"
    assert stale["contributes"] is False and stale["in_total_denominator"] is False
    assert "cadence of 7 d" in stale["detail"]
    assert "automod_scorecard" in [d["key"] for d in row["dead_or_stale"]]


def test_a_hand_run_instrument_is_never_flagged_stale(tmp_path):
    """Several instruments are hand-run by ruling (the uptake probe exits 3 with
    no table below a 0.70 precision floor). A cadence check that flags them is
    worse than no check, so they carry no declared cadence and no stale verdict."""
    row, _, _ = _row(tmp_path, GOOD_VALUES, days={"uptake_probe": "2026-09-20"})

    uptake = _component(row, "uptake_probe")
    assert uptake["status"] == "measured"
    assert uptake["hand_run_by_ruling"] is True
    assert uptake["cadence_days"] is None
    assert uptake["age_days"] == 23
    assert uptake["tier"] == 1, "hand-run means not tracked over time, not broken"
    assert "uptake_probe" not in [d["key"] for d in row["dead_or_stale"]]


def test_an_artifact_whose_pointer_yields_nothing_is_a_named_refusal_not_a_zero(tmp_path):
    """The census's own convention, applied to the index: a stage it cannot
    measure arrives with a name and a denominator, never as a rate of 0."""
    store = tmp_path / "roots" / "funnel_census"
    row_values = dict(GOOD_VALUES)
    roots = _seed_all(tmp_path, row_values)
    _seed_json(store, "funnel-2026-10-12.json", {
        "artifact": "funnel-census", "version": 1,
        "stages": [{"stage": "produced", "measured": True, "unit": "artifacts",
                    "k": 90, "n": 100, "rate": 0.9},
                   {"stage": "surfaced", "measured": False, "refusal": "empty_denominator",
                    "denominator": 0, "detail": "the window held none"}]})
    row = EII.build_row(now=NOW, root_overrides=roots, out_dir=None,
                        code_root=tmp_path / "code")

    census = _component(row, "funnel_census")
    assert census["status"] == "no-value"
    assert census["value"] is None
    assert "named refusal" in census["detail"]
    assert census["contributes"] is False


def test_a_value_that_is_not_a_share_is_reported_and_never_averaged(tmp_path):
    """Unit mismatch is the near-certain defect of a composite over mixed
    writers, so a value outside 0..1 cannot enter the total at all."""
    roots = _seed_all(tmp_path, GOOD_VALUES)
    _seed_json(tmp_path / "roots" / "retrieval_nightly", "nightly-20261012-20261012.json",
               {"summary": {"overall": {"doc_hit_rate": 650.0}}})
    row = EII.build_row(now=NOW, root_overrides=roots, out_dir=None,
                        code_root=tmp_path / "code")

    retrieval = _component(row, "retrieval_nightly")
    assert retrieval["status"] == "not-a-rate"
    assert retrieval["contributes"] is False
    assert "outside 0..1" in retrieval["detail"]


def test_funnel_census_is_carried_in_the_table_and_excluded_from_the_average(tmp_path):
    """`averageable=False` is the declared answer to the unit risk: the census's
    stages divide by artifacts and rows interchangeably, so it is reported."""
    row, _, _ = _row(tmp_path, GOOD_VALUES)
    census = _component(row, "funnel_census")
    assert census["value"] == 0.6 and census["quality"] == 0.6
    assert census["contributes"] is False and census["in_total_denominator"] is False
    excluded = next(e for e in row["totals"]["excluded"] if e["key"] == "funnel_census")
    assert "not averaged" in excluded["reason"]


# ── clause 3: two totals, provably different, and the sign headline ───────────

def test_the_declared_weight_and_equal_weight_totals_differ_under_non_uniform_weights(tmp_path):
    """Clause 3: both totals in one row, and under non-uniform weights the two
    are different numbers — the test is non-vacuous by construction."""
    values = {"retrieval_nightly": 0.2,    # quality 0.20, weight 0.28
              "iv_metrics": 0.90,          # quality 0.10, weight 0.20
              "automod_scorecard": 0.9,    # quality 0.90, weight 0.18
              "landed_outcomes": 0.0,      # quality 1.00, weight 0.14
              "tool_quality": 0.9,         # quality 0.10, weight 0.10
              "uptake_probe": 1.0,         # quality 1.00, weight 0.05
              "funnel_census": 0.6,
              "gate_to_land_lag": 17.687}  # weight 0.0, reported not averaged
    row, _, _ = _row(tmp_path, values, omit=("tool_quality",))
    totals = row["totals"]

    assert totals["status"] == "ok"
    declared, equal = totals["declared_weights"], totals["equal_weights"]
    # (0.2*0.28 + 0.1*0.20 + 0.9*0.18 + 1.0*0.14 + 1.0*0.05) / 0.85 = 0.5035
    assert declared == pytest.approx(0.5035, abs=1e-4)
    assert equal == pytest.approx((0.2 + 0.1 + 0.9 + 1.0 + 1.0) / 5, abs=1e-6)
    assert declared != equal, "a non-uniform weighting must show up in the number"
    assert abs(declared - equal) > 0.1


def test_a_sign_disagreement_between_the_two_totals_publishes_the_weights_decide_headline(tmp_path):
    """Clause 3's deliverable, implemented: when the two weightings disagree about
    the DIRECTION of the move, the headline is the disagreement, not the number."""
    prior = _prior_row(declared=0.60, equal=0.40, delta_declared=0.02,
                       delta_equal=-0.02)
    values = {"retrieval_nightly": 0.9, "iv_metrics": 0.1,       # quality .9, .9
              "automod_scorecard": 0.1, "landed_outcomes": 0.9,  # quality .1, .1
              "tool_quality": 0.9, "uptake_probe": 0.9,           # quality .1, .9
              "funnel_census": 0.6,
              "gate_to_land_lag": 17.687}   # weight 0.0, in neither total
    row, _, _ = _row(tmp_path, values, extra_rows=[prior], prior_rows=1)
    deltas = row["delta_vs_prior"]

    assert deltas["prior_generated_at"] == prior["generated_at"]
    assert deltas["declared_weights"] < 0 < deltas["equal_weights"], deltas
    assert row["headline"].startswith("the weights decide the answer")
    assert "declared-weight total moved down" in row["headline"]
    assert "equal-weight total moved up" in row["headline"]


def test_agreeing_totals_publish_the_number_rather_than_the_disagreement_headline(tmp_path):
    """The other branch of the same rule: agreement publishes the number, and the
    weights are then named decorative rather than the finding being suppressed."""
    prior = _prior_row(declared=0.50, equal=0.40, delta_declared=0.02,
                       delta_equal=0.02)
    row, _, _ = _row(tmp_path, GOOD_VALUES, extra_rows=[prior], prior_rows=1)

    assert "the weights decide the answer" not in row["headline"]
    assert row["headline"].startswith("6 of 8 instruments measured")
    assert row["weight_agreement"]["prior_rows_compared"] == 1
    assert row["weight_agreement"]["sign_disagreements"] == 0
    assert row["weight_agreement"]["decorative"] is True
    assert "decorative" in row["headline"]


def test_the_first_row_has_no_delta_and_says_so_rather_than_inventing_one(tmp_path):
    row, _, _ = _row(tmp_path, GOOD_VALUES)
    assert row["delta_vs_prior"]["declared_weights"] is None
    assert row["delta_vs_prior"]["prior_generated_at"] is None
    assert row["weight_agreement"]["decorative"] is None
    for c in row["components"]:
        assert c["movement"] in ("no-prior-row", "not-scored"), c["key"]


# ── clause 4: tier 2 needs a consumer that compares rows ─────────────────────

def test_tier_two_requires_a_named_consumer_that_compares_rows(tmp_path):
    """Clause 4: tier 2 means tracked and analysed OVER TIME, which the row can
    only claim by naming the reader that compares this artifact against a prior
    row."""
    row, _, _ = _row(tmp_path, GOOD_VALUES)
    retrieval = _component(row, "retrieval_nightly")

    assert retrieval["tier"] == 2
    assert retrieval["consumer"]["script"] == "scripts/eval_trend_stats.py"
    assert retrieval["consumer"]["compares_rows"] is True
    assert "night against the prior one" in retrieval["consumer"]["how"]
    assert _component(row, "iv_metrics")["tier"] == 2, (
        "iv_metrics_record._prior_breaching compares the last row against the "
        "median of the prior rows")


def test_the_same_fresh_component_with_the_consumer_removed_scores_tier_one(tmp_path):
    """The same component, the same fresh artifact, no named consumer: tier 1.
    Without this the tier column would be a report nicety."""
    from dataclasses import replace

    row, _, _ = _row(tmp_path, GOOD_VALUES)
    baseline = _component(row, "retrieval_nightly")
    assert baseline["tier"] == 2

    instr = _instr("retrieval_nightly")
    reading = EII.read_instrument(instr, today=NOW.date(),
                                  root_override=tmp_path / "roots"
                                  / "retrieval_nightly")
    assert reading.status == "measured" and reading.age_days == 1
    assert EII.tier_of(instr, reading) == 2
    assert EII.tier_of(replace(instr, consumer=None), reading) == 1, (
        "drop the consumer and the same fresh artifact is not tracked over time")

    no_compare = replace(instr, consumer=replace(
        instr.consumer, compares_rows=False))
    assert EII.tier_of(no_compare, reading) == 1, (
        "a consumer that only renders the newest row leaves it at tier 1")

    row2, _, _ = _row(tmp_path / "noconsumer", GOOD_VALUES,
                      instruments=[no_compare if i.key == "retrieval_nightly" else i
                                   for i in EII.INSTRUMENTS])
    assert _component(row2, "retrieval_nightly")["tier"] == 1
    assert baseline["weight"] == _component(row2, "retrieval_nightly")["weight"], (
        "the tier moved, the weight did not")


def test_a_component_with_no_artifact_is_tier_zero_and_a_stale_one_is_tier_one(tmp_path):
    from dataclasses import replace

    row, _, _ = _row(tmp_path, GOOD_VALUES, omit=("tool_quality",),
                     days={"automod_scorecard": "2026-09-11"})
    assert _component(row, "tool_quality")["tier"] == 0
    assert _component(row, "automod_scorecard")["tier"] == 1

    reading = EII.read_instrument(_instr("automod_scorecard"), today=NOW.date(),
                                  root_override=tmp_path / "roots"
                                  / "automod_scorecard")
    assert reading.status == "stale"
    assert EII.tier_of(_instr("automod_scorecard"), reading) == 1
    # Even with a row-comparing consumer, rows that are not arriving cannot be
    # "tracked over time".
    armed = replace(_instr("automod_scorecard"), consumer=EII.Consumer(
        "scripts/eval_trend_stats.py", "compares rows", compares_rows=True))
    assert EII.tier_of(armed, reading) == 1


# ── clause 5: the table is built from each writer's own path constant ─────────

def test_each_component_path_is_the_path_its_own_writer_declares():
    """Clause 5: the component set resolves through each instrument writer's own
    constant, so a store outside the data root and one defaulting into the code
    tree are in the table instead of silently skipped.

    #3541's `gate_to_land_lag` is the awkward case and is asserted in both
    directions below: its journal lives in the automod state dir, outside
    `DATA_ROOT`, so there is no `app.paths` constant for it to name, and the only
    way it can honour this rule is through the recorder's own `record_path()`,
    which joins that state dir with `RECORD_NAME`."""
    from app import paths, uptake
    from scripts import funnel_census, iv_metrics_record
    from scripts.automod import landed_outcomes, scorecard, state

    expected = {
        "retrieval_nightly": paths.EVAL_BASELINES_DIR,
        "tool_quality": paths.TOOL_QUALITY_DIR,
        "funnel_census": paths.DATA_ROOT.joinpath(*funnel_census.FUNNEL_ARTIFACT_SUBDIR),
        "iv_metrics": iv_metrics_record.DEFAULT_SERIES_PATH,
        "automod_scorecard": scorecard.scorecard_path(),
        "landed_outcomes": landed_outcomes.landed_outcomes_path(),
        "uptake_probe": uptake.UPTAKE_ARTIFACT_DIR,
        "gate_to_land_lag": GTL.record_path(),
    }
    assert set(expected) == set(ALL_KEYS) == {i.key for i in EII.INSTRUMENTS}
    for key, want in expected.items():
        assert _instr(key).artifact() == want, key

    # The two stores the clause names, specifically.
    uptake_path = uptake.UPTAKE_ARTIFACT_DIR
    assert paths.LLOYD_HOME in uptake_path.parents, "uptake defaults into the code tree"
    assert paths.DATA_ROOT not in uptake_path.parents
    assert scorecard.scorecard_path().parent == state.STATE_DIR
    assert paths.DATA_ROOT not in state.STATE_DIR.parents, (
        "the automod ledger dir is outside the data root")
    # And the four that ARE under the data root, so the table spans both worlds.
    for key in ("retrieval_nightly", "tool_quality", "funnel_census", "iv_metrics"):
        assert paths.DATA_ROOT in expected[key].parents, key

    # #3541: the lag journal joins BOTH names the recorder owns — the state dir and
    # the file name — rather than one re-spelled string, and it is a single dated
    # series like `landed_outcomes`, not a directory of per-day files.
    lag = _instr("gate_to_land_lag")
    assert lag.artifact().parent == state.STATE_DIR
    assert lag.artifact().name == GTL.RECORD_NAME == "gate_to_land_lag.jsonl"
    assert state.STATE_DIR not in paths.DATA_ROOT.parents, (
        "the lag journal is outside the data root, which is why it could not be an "
        "`app.paths` constant and must resolve through the recorder instead")
    assert lag.pattern is None and lag.date_key == "generated_at"
    assert lag.cadence_days == 1 and lag.hand_run_by_ruling is False
    assert lag.unit == "seconds" and lag.polarity == "down"
    assert lag.value_path == "p50" and lag.n_path == "n"
    assert lag.scheduled_by == "autonomy/101-gate-to-land-lag-daily-record.md"


def test_the_lag_component_path_is_not_a_re_typed_literal():
    """#3541 clause 1's negative half: `test_each_component_path_is_the_path_its_own_writer_declares`
    proves the lag arm RESOLVES to the recorder's path, but a re-typed literal would
    satisfy that too — the equality would simply hold. This node closes the gap: the
    recorder's filename must not appear anywhere in the publisher's source, prose
    included, so the only remaining way to reach the file is `record_path()`.

    The positive control is the same probe run on the recorder, which does spell it:
    proof the check is reading a file that contains the string rather than scoring a
    search that could never hit."""
    source = (ROOT / "scripts" / "eval_instrument_index.py").read_text(encoding="utf-8")
    assert GTL.RECORD_NAME not in source, (
        f"the publisher spells `{GTL.RECORD_NAME}` somewhere; the recorder owns that "
        "name and the table must resolve through it")
    recorder = ROOT / "scripts" / "maintenance" / "gate_to_land_lag.py"
    assert GTL.RECORD_NAME in recorder.read_text(encoding="utf-8"), (
        "positive control failed: the recorder does not name its own journal, so the "
        "absence above would prove nothing")


def test_the_state_dir_store_and_the_in_tree_store_both_appear_with_real_newest_dates(tmp_path):
    """Clause 5 measured: the store outside the data root (an automod state dir
    holding `scorecard.jsonl`) and the one inside the code tree (uptake's default)
    are in the row, with the newest date each actually holds."""
    state_dir = tmp_path / "state" / "lloyd-automod"
    code_tree = tmp_path / "lloyd"
    uptake_dir = code_tree / "eval" / "uptake"
    _seed_series(state_dir, "scorecard.jsonl", [
        {"computed_at": "2026-09-11T02:45:18+00:00",
         "acceptance": {"hit_rate": 0.5, "with_outcome": 14}},
        {"computed_at": "2026-10-12T20:03:00+00:00",
         "acceptance": {"hit_rate": 0.62, "with_outcome": 21}}])
    _seed(uptake_dir, "uptake_probe", "2026-09-11", 0.9)
    _seed(uptake_dir, "uptake_probe", "2026-09-20", 1.0)

    roots = _seed_all(tmp_path, GOOD_VALUES)
    roots["automod_scorecard"] = state_dir / "scorecard.jsonl"
    roots["uptake_probe"] = uptake_dir
    out = tmp_path / "index"
    row = EII.build_row(now=NOW, root_overrides=roots, out_dir=out,
                        code_root=code_tree)

    score = _component(row, "automod_scorecard")
    assert score["newest_artifact"] == "2026-10-12", "the newest of the two rows"
    assert score["value"] == pytest.approx(0.62)
    assert score["outside_data_root"] is True
    assert score["instrument_path"] == str(state_dir)

    uptake = _component(row, "uptake_probe")
    assert uptake["newest_artifact"] == "2026-09-20", "the later of the two tables"
    assert uptake["inside_code_tree"] is True
    assert uptake["outside_data_root"] is True
    assert uptake["instrument_path"] == str(uptake_dir)


def test_the_real_in_tree_store_reports_its_real_newest_date():
    """The same read against the one store that is tracked in git, so the clause
    is not only proven against fixtures: `git ls-files eval/uptake` lists tables
    in the checkout, and the index must report the newest of those."""
    from app import uptake

    store = uptake.UPTAKE_ARTIFACT_DIR
    assert store.is_dir(), f"{store} is tracked in git, so it is a directory here"
    named = [d for d in (EII._date_from_name(p.name) for p in store.glob("uptake-*.json"))
             if d is not None]
    assert named, f"no dated uptake tables under {store}"
    reading = EII.read_instrument(_instr("uptake_probe"), today=NOW.date())
    assert reading.artifact_path != ""
    assert reading.newest_artifact == max(named).isoformat()


# ── the report-only rule ──────────────────────────────────────────────────────

#: The three files allowed to name the index: the publisher itself, `app/paths.py`
#: where the store is declared, and this file — whose whole job is to prove the
#: rule, and which cannot prove it without spelling the name. Everything else that
#: reads the published row would be a gate, a promotion rule or a rollback path,
#: which v1 forbids.
DECLARED_IN = frozenset({
    "scripts/eval_instrument_index.py",
    "app/paths.py",
    "tests/test_eval_instrument_index.py",
})

#: Directories that are not Lloyd's own source: version control, throwaway
#: virtualenvs, caches, build output, and the data root as it resolves inside a
#: worktree (`.lloyd-data`). `qmd` is the separate upstream fork — its own
#: repository, gitignored by this one, and not something a round may edit. That a
#: pruned name never hides Lloyd source is asserted, not assumed, by
#: `test_the_pruned_directories_hold_no_lloyd_source`: pruning is the one way this
#: scan could shrink silently, which is precisely how the previous version of it
#: failed.
_NON_SOURCE_DIRS = frozenset({".git", ".venv", ".venvs", ".pytest_cache",
                              ".mypy_cache", ".ruff_cache", "__pycache__",
                              "node_modules", "dist", "build", "qmd",
                              ".lloyd-data"})

#: Named positives of the enumeration, asserted below. This scan used to walk a
#: hand-picked list of directories that omitted the repo root (where `server.py`
#: lives), omitted `agent_mcp/` and `tests/`, and named an `autonomy` directory
#: that does not exist in this tree at all — so it yielded nothing for a root it
#: silently skipped, and a gate wired into `server.py` would have passed a test
#: whose docstring claimed the property held over the whole tree. A property over
#: an open set is enumerated, and the enumeration gets its own control.
SCAN_SENTINELS = ("server.py",
                  "app/paths.py",
                  "app/routers/dashboard.py",
                  "scripts/eval_instrument_index.py",
                  "scripts/automod/gate.py",
                  "scripts/automod/round.py",
                  "scripts/automod/promote.py",
                  "agent_mcp/__init__.py",
                  "tests/test_eval_instrument_index.py")


def _python_files_in_tree() -> list[Path]:
    """Every Lloyd-owned `.py` file under the repository root, sources included."""
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in _NON_SOURCE_DIRS)
        files.extend(Path(dirpath) / name for name in filenames
                     if name.endswith(".py"))
    return sorted(files)


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """The string constants that are docstrings — prose, so not reads."""
    out: set[int] = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                              ast.AsyncFunctionDef))
                and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            out.add(id(body[0].value))
    return out


def _index_reads_in(path: Path) -> list[str]:
    """Every code-level reference to the index in one file.

    AST rather than grep because prose is not a read: an import, a bound name, an
    attribute, or a string that would reach the module at runtime (an `-m`
    argument, an `import_module` target — the way a scheduler's helper could wire
    itself in without tripping an import-based check). Docstrings are excluded for
    the same reason a comment is: `app/paths.py` pointing at this test in its
    comment is not the test failing.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    prose = _docstring_nodes(tree)
    hits: list[str] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""] + [a.name for a in node.names]
        elif isinstance(node, (ast.Name, ast.Attribute)):
            names = [getattr(node, "id", None) or getattr(node, "attr", "")]
        elif isinstance(node, ast.Constant):
            if id(node) in prose or not isinstance(node.value, str):
                continue
            names = [node.value]
        if any("eval_instrument_index" in n or "EVAL_INSTRUMENT_INDEX" in n
               for n in names):
            hits.append(next(n for n in names
                             if "eval_instrument_index" in n
                             or "EVAL_INSTRUMENT_INDEX" in n))
    return hits


def test_the_scanner_names_a_read_and_forgives_prose(tmp_path):
    """Both controls for the scanner, because a whole-tree absence test is worth
    nothing if the scanner matches nothing. A file that imports the publisher, or
    spawns it with `-m`, must be named; a file that only mentions it in prose must
    not be."""
    imported = tmp_path / "future_gate.py"
    imported.write_text("from scripts import eval_instrument_index\n\n"
                        "def decide():\n"
                        "    return eval_instrument_index.main([])\n", encoding="utf-8")
    spawned = tmp_path / "future_runner.py"
    spawned.write_text("import subprocess\n\n"
                       "subprocess.run(['python', '-m', "
                       "'scripts.eval_instrument_index'])\n", encoding="utf-8")
    prose = tmp_path / "note.py"
    prose.write_text('"""See scripts/eval_instrument_index.py (#2597) for the index."""\n\n'
                     "X = 1\n", encoding="utf-8")
    clean = tmp_path / "other.py"
    clean.write_text("X = 1\n", encoding="utf-8")

    assert "eval_instrument_index" in _index_reads_in(imported), (
        "a file that imports the publisher must be named as an offender")
    assert _index_reads_in(spawned) == ["scripts.eval_instrument_index"], (
        "the `-m` spelling is a read too: it is how a gate would run the publisher "
        "without importing it")
    assert _index_reads_in(prose) == [], "a docstring is prose, not a read"
    assert _index_reads_in(clean) == []


def test_the_tree_scan_reaches_the_boot_path_the_gate_and_the_promoter():
    """Positive control for the enumeration itself: the files a v1 offender would
    live in are in the set the scan walks, and the set is the size of a real tree.
    Without this, an empty or truncated walk would satisfy the absence test below
    the same way the old directory list did."""
    files = {p.relative_to(ROOT).as_posix() for p in _python_files_in_tree()}
    for sentinel in SCAN_SENTINELS:
        assert sentinel in files, (
            f"{sentinel} is Lloyd source in this tree, but the scan that claims the "
            f"whole tree does not reach it")
    assert len(files) > 500, (
        f"the enumeration found {len(files)} .py files; this tree has >1300, so the "
        "walk is being truncated somewhere and the absence test is vacuous")


def test_the_pruned_directories_hold_no_lloyd_source():
    """The prune set is the one place this scan can shrink silently, so it is
    checked rather than trusted: git tracks no file under a pruned directory name.
    Anything gitignored cannot be Lloyd's shipped source, and if a pruned name ever
    did hold tracked source the whole-tree claim below would be a lie — which is
    exactly how the previous directory-list version of this scan passed on the files
    it never opened."""
    listing = subprocess.run(["git", "ls-files"], cwd=str(ROOT), capture_output=True,
                             text=True, timeout=120)
    assert listing.returncode == 0, listing.stderr[-400:]
    tracked = [line for line in listing.stdout.splitlines() if line.endswith(".py")]
    assert tracked, "git ls-files found no Python in this worktree: the control is vacuous"
    buried = [p for p in tracked
              if any(part in _NON_SOURCE_DIRS for part in Path(p).parts[:-1])]
    assert buried == [], (
        f"{len(buried)} tracked file(s) sit under a pruned directory, e.g. "
        f"{buried[:3]} — the scan is skipping source")


def test_no_gate_promotion_or_rollback_path_reads_the_index():
    """The v1 boundary, checked over every module in the tree: no gate, no
    promotion rule and no rollback path may read the published row. The text
    prefilter cannot skip an offender — any import, name or string the AST would
    report contains the substring in the file's own bytes — it only avoids
    parsing 1300 files to find three."""
    offenders: list[str] = []
    for path in _python_files_in_tree():
        rel = path.relative_to(ROOT).as_posix()
        if rel in DECLARED_IN:
            continue
        text = path.read_text(encoding="utf-8")
        if "eval_instrument_index" not in text and "EVAL_INSTRUMENT_INDEX" not in text:
            continue
        offenders.extend(f"{rel}: {hit}" for hit in _index_reads_in(path))
    assert offenders == [], f"something outside the publisher reads the index: {offenders}"


#: What the probe prints: whether the publisher ended up in the boot path's module
#: graph. The question is asked of every spelling the module can be registered
#: under, because `python -m scripts.eval_instrument_index` and
#: `import scripts.eval_instrument_index` both key it as
#: `scripts.eval_instrument_index` and neither matches the bare name. The previous
#: probe asked `'eval_instrument_index' in sys.modules`, an exact-key test that
#: printed False even when the publisher WAS imported — the review rung proved it
#: by importing the module in the probe and watching it still print False. A probe
#: that cannot print True is not a probe, so
#: `test_the_import_probe_reports_true_when_the_publisher_is_imported` is its
#: positive control, run through the same helper.
_IMPORT_PROBE = (
    "import sys, importlib;"
    "importlib.import_module('app.paths');"
    "{extra};"
    "importlib.import_module('app.routers.dashboard');"
    "print(any(k == 'scripts.eval_instrument_index'"
    " or k == 'eval_instrument_index'"
    " or k.endswith('.eval_instrument_index')"
    " for k in sys.modules))"
)


def _boot_path_imports_the_publisher(also_import: str = "") -> bool:
    """Ask a fresh interpreter whether importing the backend's request router
    pulls the publisher in with it. With `also_import` the same question is asked
    of a graph that DOES contain the module, which is the only way to learn the
    probe reports True when it should."""
    extra = (f"importlib.import_module({also_import!r})" if also_import else "pass")
    probe = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE.format(extra=extra)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=180)
    assert probe.returncode == 0, f"probe failed: {probe.stderr[-800:]}"
    verdict = probe.stdout.strip().splitlines()[-1]
    assert verdict in ("True", "False"), f"the probe printed {verdict!r}, not a verdict"
    return verdict == "True"


def test_the_publisher_is_not_imported_by_the_boot_path():
    """Import-proof for the same rule: the backend's own module graph must not pull
    the publisher in, so no request path can be affected by it. Meaningful only
    beside its control below."""
    assert _boot_path_imports_the_publisher() is False


def test_the_import_probe_reports_true_when_the_publisher_is_imported():
    """The control that makes the test above able to fail: importing the publisher
    into the same probe makes it report True, so the False above is a verdict and
    not the probe's inability to see."""
    assert _boot_path_imports_the_publisher("scripts.eval_instrument_index") is True


def test_the_publisher_writes_only_its_own_series(tmp_path):
    """`git status --porcelain` clean after a run: the publisher's sole output is
    one appended line in the series, and it creates nothing else."""
    out = tmp_path / "index"
    row = EII.build_row(now=NOW, root_overrides=_seed_all(tmp_path, GOOD_VALUES),
                        out_dir=out, code_root=tmp_path / "code")
    path = EII.append_row(row, out)
    created = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    assert path.is_file()
    assert created.count("index/instrument-index.jsonl") == 1
    assert not list(tmp_path.glob("**/*.lock"))


# ── the CLI seam ──────────────────────────────────────────────────────────────
# Every other test here drives `build_row` / `append_row` in process, which is the
# library but not the surface the daily task will invoke: `python -m
# scripts.eval_instrument_index` from the repo root, whose only report back is a
# process exit code. These tests run it as a subprocess against a `--roots`
# fixture, so they exercise the `-m` packaging AND never touch the live data root.
# The previous version of this test ran in process against the real stores and
# appended to the real series: the one row published before this round exists was
# written by pytest, not by a schedule.

def _roots_file(tmp_path, values: dict, *, omit: tuple[str, ...] = (),
                days: dict | None = None) -> Path:
    """Seed each instrument into its own tmp store and publish the `--roots`
    snapshot that points the publisher at them instead of at the live data root."""
    roots = _seed_all(tmp_path, values, omit=omit, days=days)
    path = tmp_path / "roots-snapshot.json"
    path.write_text(json.dumps({k: str(v) for k, v in roots.items()}, indent=1),
                    encoding="utf-8")
    return path


def _run_publisher(*argv, cwd=ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "scripts.eval_instrument_index", *argv],
        cwd=str(cwd), capture_output=True, text=True, timeout=180)


def test_the_module_cli_appends_one_row_and_exits_zero_on_the_ok_branch(tmp_path):
    """The `-m` seam a scheduler runs, end to end on fixtures: one row appended to
    the series named by `--out-dir`, stamped by `--now`, and exit 0 because the
    totals measured enough to publish. Every earlier test drove `main()` in
    process, so the `__main__` block and the ROOT `sys.path` bootstrap it sits
    behind were never exercised."""
    out = tmp_path / "cli-ok"
    proc = _run_publisher("--roots", str(_roots_file(tmp_path, GOOD_VALUES)),
                          "--out-dir", str(out),
                          "--now", "2026-10-13T03:40:00+00:00", "--print")
    assert proc.returncode == 0, f"exit {proc.returncode}: {proc.stderr[-1500:]}"
    assert "HEADLINE" in proc.stdout, "`--print` renders the table"

    rows = EII.read_series(out)
    assert len(rows) == 1, "one `-m` run appends exactly one row"
    assert rows[0]["totals"]["status"] == "ok", (
        "this fixture is the ok branch — the exit-1 test below is the other one, so "
        "the pair pins a mapping rather than self-consistency")
    assert rows[0]["date"] == "2026-10-13", "`--now` is what stamps the row"
    assert len(rows[0]["components"]) == 8


def test_the_module_cli_exits_1_and_publishes_the_refusal_when_one_arm_survives(tmp_path):
    """The other exit branch on the same seam: seven of the eight instruments
    missing, so the totals refuse, the refusal is still published as the day's
    record, and the exit code is 1 — because `main()` returning 0 whatever the
    totals said is how a scheduler ends up reading an index that measured nothing
    as a green run. Seven, not six, because #3541 grew the table to eight arms and
    `gate_to_land_lag` is one of the omitted ones: it is seeded nowhere here, and
    a fixture that left it reading the live journal would make this run depend on
    a real artifact instead of a snapshot."""
    out = tmp_path / "cli-refused"
    proc = _run_publisher(
        "--roots", str(_roots_file(
            tmp_path, {"retrieval_nightly": 1.0},
            omit=("iv_metrics", "automod_scorecard", "landed_outcomes",
                  "tool_quality", "uptake_probe", "funnel_census",
                  "gate_to_land_lag"))),
        "--out-dir", str(out), "--now", "2026-10-13T03:40:00+00:00", "--print")
    assert proc.returncode == 1, (
        f"an index that refused exited {proc.returncode}: a caller that ignores the "
        "exit code cannot tell a dead series from a published verdict")
    assert "the index refuses" in proc.stdout

    rows = EII.read_series(out)
    assert len(rows) == 1, "a refused run publishes one row, not zero"
    assert rows[0]["totals"]["status"] == "refused"
    assert len(rows[0]["dead_or_stale"]) == 7


def test_no_record_prints_and_creates_nothing_even_on_the_refused_branch(tmp_path):
    """`--no-record` creates no store at all, and the exit code still follows the
    totals rather than the write that was skipped."""
    out = tmp_path / "cli-no-record"
    proc = _run_publisher("--roots", str(_roots_file(tmp_path, GOOD_VALUES)),
                          "--out-dir", str(out),
                          "--now", "2026-10-13T03:40:00+00:00", "--no-record")
    assert proc.returncode == 0, f"exit {proc.returncode}: {proc.stderr[-1500:]}"
    assert "HEADLINE" in proc.stdout
    assert not out.exists(), "--no-record creates no store and appends nothing"

    refused = _run_publisher(
        "--roots", str(_roots_file(
            tmp_path / "second", {"retrieval_nightly": 1.0},
            omit=tuple(k for k in ALL_KEYS if k != "retrieval_nightly"))),
        "--out-dir", str(out), "--no-record")
    assert refused.returncode == 1, (
        "the exit code follows the totals on a printed run too")
    assert not out.exists()


def test_a_roots_snapshot_naming_an_unknown_component_is_refused_not_ignored(tmp_path):
    """`build_row` resolves overrides by key and silently ignores one that matches
    nothing, so a typo in a snapshot would leave that arm reading the live store
    while the run looked fixture-driven — a thing that reads as overridden while it
    is not, which is the failure this index refuses in its own components."""
    out = tmp_path / "cli-bad-roots"
    roots = _roots_file(tmp_path, GOOD_VALUES)
    doc = json.loads(roots.read_text(encoding="utf-8"))
    doc["retrivel_nightly"] = str(tmp_path / "typo")
    roots.write_text(json.dumps(doc), encoding="utf-8")

    proc = _run_publisher("--roots", str(roots), "--out-dir", str(out), "--no-record")
    assert proc.returncode == 2, (
        f"a snapshot naming a component that is not in the table exited "
        f"{proc.returncode}; that is a partial override, not a run")
    assert "retrivel_nightly" in proc.stderr, "the refusal names the bad key"
    assert not out.exists(), "a refused snapshot publishes nothing"


def test_the_default_out_dir_is_the_declared_store_constant(tmp_path, monkeypatch):
    """No `--out-dir` means `app.paths.EVAL_INSTRUMENT_INDEX_DIR` — the constant,
    not a path re-spelled at the call site, which is the same rule the instrument
    table itself is built under (#2597 clause 5)."""
    from app import paths

    target = tmp_path / "declared" / "instrument-index"
    monkeypatch.setattr(paths, "EVAL_INSTRUMENT_INDEX_DIR", target)
    assert not target.exists()
    rc = EII.main(["--now", "2026-10-13T03:40:00+00:00", "--roots",
                   str(_roots_file(tmp_path, GOOD_VALUES))])
    assert rc == 0
    assert len(EII.read_series(target)) == 1, (
        "the default resolved somewhere other than the patched constant")
