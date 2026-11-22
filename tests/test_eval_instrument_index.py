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

It grew to nine with #3886, which registered `behaviour_exhibit` — the daily
exhibit-rate sweep `app/behaviour_exhibit.py` writes to the store now named by
`app.paths.BEHAVIOUR_EXHIBIT_DIR`. It is reported and not averaged for a different
reason than the lag arm: its one scalar is a share POOLED over four behaviours whose
denominators differ by six orders of magnitude, so the caption arm would decide the
composite. What it is here for is the headline: an unregistered store cannot be
called missing, which is how the exhibit sweep ran once on 2026-11-04 and nobody
noticed for three days — the identical failure #2275's tool_quality is only visible
at all because it IS a row here.
`test_a_day_with_no_exhibit_artifact_prints_behaviour_exhibit_missing_or_stale` pins
that line, and `test_the_exhibit_store_is_named_by_app_paths_and_shared_with_its_writer`
pins the one-path half, including the `Path.home() / "lloyd-data"` fallback that used
to sit behind `out_dir()`.

The five clause tests are numbered to match the item's acceptance clauses:

1  `test_run_appends_one_dated_row_covering_nine_components_with_every_declared_field`
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
import textwrap
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

#: The eight arms #4169 leaves in the table. `automod_scorecard` was here until that
#: item retired it: its store's newest row is 2026-09-12 and nothing writes the file,
#: so it published `stale` every day and its weight was renormalised away every day.
ALL_KEYS = ("retrieval_nightly", "iv_metrics",
            "landed_outcomes", "tool_quality", "uptake_probe", "funnel_census",
            "gate_to_land_lag", "behaviour_exhibit")


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
    if key == "behaviour_exhibit":
        # `app/behaviour_exhibit.py` writes ONE record per run into
        # `behaviour-exhibit-<YYYY-MM-DD>.jsonl`: the per-behaviour `rows`, and beside
        # them `worker_pooled` — the single pooled worker scalar this arm reads, as
        # `{k, n, rate, ci_low, ci_high}` (the `landed_outcomes` pair shape, for the same
        # reason: the interval comes from the writer, not from a re-derivation here). The
        # denominator is the live 2026-11-04 artifact's own worker N for
        # `tool-call-caption-present`, 156,286, and the ±0.00145 half-width is that N's
        # own Wilson half-width — a fixture missing a field the writer always emits is
        # where a drifted pointer hides.
        n = 156286
        k = int(round(value * n))
        return _seed_series(store, f"behaviour-exhibit-{day}.jsonl", [{
            "schema": 1, "measured_at": f"{day}T02:24:50+00:00",
            "rows": [{"id": "tool-call-caption-present", "status": "measured",
                      "by_population": {"worker": {
                          "situations_encountered": n, "exhibited": k,
                          "status": "measured"}}}],
            "worker_pooled": {"population": "worker", "k": k, "n": n,
                              "rate": round(k / n, 6),
                              "ci_low": round(max(0.0, value - 0.00145), 6),
                              "ci_high": round(min(1.0, value + 0.00145), 6)}}])
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
    "landed_outcomes": 0.075,        # down is better -> quality 0.925
    "tool_quality": 0.04,            # down is better -> quality 0.96
    "uptake_probe": 1.0,             # up
    "funnel_census": 0.6,            # share, reported not averaged
    "gate_to_land_lag": 17.687,      # SECONDS, reported not averaged (#3541)
    # The live store's own caption-arm worker rate on 2026-11-04 (141,558 / 156,286).
    # A share pooled over the worker population, reported and never averaged (#3886).
    "behaviour_exhibit": 0.9058,
}


#: The two arms published with weight 0.0: `gate_to_land_lag` and `behaviour_exhibit`
#: are carried in the
#: table and excluded from the composite (#3541), so the "every weight is
#: positive" sweep below has to name it rather than silently pass it.
ZERO_WEIGHT_KEYS = ("gate_to_land_lag", "behaviour_exhibit")


# ── clause 1: one dated row, >=6 components, every declared field ─────────────

def test_run_appends_one_dated_row_covering_nine_components_with_every_declared_field(tmp_path):
    """Clause 1 (#2597, grown to eight arms by #3541 and nine by #3886, back to eight
    when #4169 retired `automod_scorecard`): running the
    publisher appends ONE dated row under the data root covering >=6 components, each
    carrying its instrument path, newest-artifact date, tier, weight, unit and latest
    value.

    The count is asserted rather than sampled because an instrument that is absent from
    the published row is invisible: `behaviour_exhibit` earned its place in the table
    precisely because the sweep's store was not in it, so its death could not be
    reported. A table that quietly dropped an arm would keep printing a composite."""
    row, path, out = _row(tmp_path, GOOD_VALUES)

    assert path == out / "instrument-index.jsonl"
    rows = EII.read_series(out)
    assert len(rows) == 1, "one run appends exactly one row"
    assert row["date"] == "2026-10-13"
    assert row["item"] == "#2597"

    components = row["components"]
    assert len(components) == 8, (
        "eight instruments: the nine #3886 shipped, less the `automod_scorecard` arm "
        "#4169 retired — and an arm that fell out for want of an update would leave "
        "this list at seven, which is the failure this count is here to catch")
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
    a difference, not a reading of the flags: the same fixtures are published
    twice, once through the whole table and once with the lag arm taken
    out of it, and the two composites must be equal to the digit. (#3886 grew the
    table to nine arms underneath this test; the comparison is a difference, so it
    still says what it said — and `behaviour_exhibit` is excluded the same way, for
    which `test_the_exhibit_arm_is_excluded_from_both_totals_and_still_printed` is the
    direct form.) A `seconds` value
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
        "adding the arm cannot have re-scaled the six weighted arms")
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
    """`uptake` holds 3 dated tables on this box and `landed_outcomes` is a series a
    daily job has written for under four days of history in a fixture, so a short
    series may not be read as a flat line: the answer is `insufficient-series`, per the
    flat-headline rule's own window of 4 rows."""
    row, _, _ = _row(tmp_path, GOOD_VALUES,
                     extra_rows=[_prior_row(0.6, 0.5, 0.01, 0.01)] * 2,
                     prior_rows=2)
    assert row["history_rows"] == 2
    for key in ("landed_outcomes", "uptake_probe"):
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
    assert _component(row, "landed_outcomes")["flat_check"] == "moving", (
        "every arm but the pinned one moves: its quality is 1 - 0.075, against 0.4 in "
        "each of the three prior rows")


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
    assert totals["scored"] == 4, (
        "four arms in the denominator: retrieval, iv_metrics, landed_outcomes and "
        "uptake_probe — `funnel_census` is measured but not averageable")
    assert [e["key"] for e in totals["excluded"] if e["status"] == "missing"] == \
        ["tool_quality"]
    # 1.0 minus the missing arm (0.12) and the non-averageable share (0.06):
    # the renormalisation is printed, not implied by a changed denominator.
    assert totals["renormalised_weight_sum"] == pytest.approx(0.82)
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
    Seven of the eight are omitted (#3541 grew the table to eight arms, #3886 to nine,
    #4169 retired `automod_scorecard` back to eight), the surviving one being
    `retrieval_nightly`."""
    row, _, _ = _row(tmp_path, {"retrieval_nightly": 1.0},
                     omit=("iv_metrics", "landed_outcomes",
                           "tool_quality", "uptake_probe", "funnel_census",
                           "gate_to_land_lag", "behaviour_exhibit"))

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
    # `tool_quality` carries a declared cadence of 7 days and is not hand-run, so it
    # is the arm this behaviour belongs on now that `automod_scorecard` is retired
    # (#4169) — the test is about the staleness rule, never about that arm.
    row, _, _ = _row(tmp_path, GOOD_VALUES, days={"tool_quality": "2026-09-11"})

    stale = _component(row, "tool_quality")
    assert stale["status"] == "stale"
    assert stale["age_days"] == 32
    assert stale["tier"] == 1, "rows are not arriving, so nothing is tracked over time"
    assert stale["contributes"] is False and stale["in_total_denominator"] is False
    assert "cadence of 7 d" in stale["detail"]
    assert "tool_quality" in [d["key"] for d in row["dead_or_stale"]]


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
    values = {"retrieval_nightly": 0.2,    # quality 0.20, weight 0.35
              "iv_metrics": 0.90,          # quality 0.10, weight 0.24
              "landed_outcomes": 0.0,      # quality 1.00, weight 0.17
              "tool_quality": 0.9,         # quality 0.10, weight 0.12 — omitted below
              "uptake_probe": 1.0,         # quality 1.00, weight 0.06
              "funnel_census": 0.6,
              "gate_to_land_lag": 17.687,  # weight 0.0, reported not averaged
              "behaviour_exhibit": 0.9058}  # weight 0.0 too (#3886): in neither total
    row, _, _ = _row(tmp_path, values, omit=("tool_quality",))
    totals = row["totals"]

    assert totals["status"] == "ok"
    declared, equal = totals["declared_weights"], totals["equal_weights"]
    # The four scored arms' weights sum to 0.82 (0.35 + 0.24 + 0.17 + 0.06), so:
    # (0.2*0.35 + 0.1*0.24 + 1.0*0.17 + 1.0*0.06) / 0.82 = 0.324 / 0.82 = 0.3951
    assert declared == pytest.approx(0.3951, abs=1e-4)
    assert equal == pytest.approx((0.2 + 0.1 + 1.0 + 1.0) / 4, abs=1e-6)
    assert declared != equal, "a non-uniform weighting must show up in the number"
    assert abs(declared - equal) > 0.1


def test_a_sign_disagreement_between_the_two_totals_publishes_the_weights_decide_headline(tmp_path):
    """Clause 3's deliverable, implemented: when the two weightings disagree about
    the DIRECTION of the move, the headline is the disagreement, not the number."""
    # The prior's declared total sits at 0.70 because the scored arms' qualities below
    # weight out to 0.614/0.94 = 0.6532 — down from it — while their unweighted mean is
    # 2.9/5 = 0.58, up from the prior's 0.40. Before #4169 the 0.70 was 0.60, and the
    # retired arm's 0.1-quality read at weight 0.18 did the pulling instead.
    prior = _prior_row(declared=0.70, equal=0.40, delta_declared=0.02,
                       delta_equal=-0.02)
    values = {"retrieval_nightly": 0.9, "iv_metrics": 0.1,       # quality .9, .9
              "landed_outcomes": 0.9,                            # quality .1
              "tool_quality": 0.9, "uptake_probe": 0.9,           # quality .1, .9
              "funnel_census": 0.6,
              "gate_to_land_lag": 17.687,   # weight 0.0, in neither total
              "behaviour_exhibit": 0.9058}  # weight 0.0 as well (#3886)
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
    # `scored` is the count of arms that entered a total, not of arms that measured:
    # seven components report `measured` here and five are scored, because the two
    # shares carried by ruling — `funnel_census` and `behaviour_exhibit` — are published
    # and excluded. The exclusion of the new one is asserted arithmetically in
    # `test_the_exhibit_arm_is_excluded_from_both_totals_and_still_printed`; what the
    # headline proves is that the DENOMINATOR grew with the table, which is the number a
    # store cannot hide from.
    assert row["headline"].startswith("5 of 8 instruments measured")
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

    # `uptake_probe` is the absent one (tier 0) and `tool_quality` the stale one
    # (tier 1): both behaviours belong on arms the table still carries, which since
    # #4169 it does at eight.
    row, _, _ = _row(tmp_path, GOOD_VALUES, omit=("uptake_probe",),
                     days={"tool_quality": "2026-09-11"})
    assert _component(row, "uptake_probe")["tier"] == 0
    assert _component(row, "tool_quality")["tier"] == 1

    reading = EII.read_instrument(_instr("tool_quality"), today=NOW.date(),
                                  root_override=tmp_path / "roots"
                                  / "tool_quality")
    assert reading.status == "stale"
    assert EII.tier_of(_instr("tool_quality"), reading) == 1
    # Even with a row-comparing consumer, rows that are not arriving cannot be
    # "tracked over time".
    armed = replace(_instr("tool_quality"), consumer=EII.Consumer(
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
    from scripts.automod import landed_outcomes, state

    expected = {
        "retrieval_nightly": paths.EVAL_BASELINES_DIR,
        "tool_quality": paths.TOOL_QUALITY_DIR,
        "funnel_census": paths.DATA_ROOT.joinpath(*funnel_census.FUNNEL_ARTIFACT_SUBDIR),
        "iv_metrics": iv_metrics_record.DEFAULT_SERIES_PATH,
        "landed_outcomes": landed_outcomes.landed_outcomes_path(),
        "uptake_probe": uptake.UPTAKE_ARTIFACT_DIR,
        "gate_to_land_lag": GTL.record_path(),
        # #3886: the exhibit store's ONLY name, which since this round is also the one
        # `app.behaviour_exhibit.out_dir()` returns. Before it, no entry could go here —
        # there was no constant to name, and the arm could not have existed either.
        "behaviour_exhibit": paths.BEHAVIOUR_EXHIBIT_DIR,
    }
    assert set(expected) == set(ALL_KEYS) == {i.key for i in EII.INSTRUMENTS}
    for key, want in expected.items():
        assert _instr(key).artifact() == want, key

    # The two stores the clause names, specifically.
    uptake_path = uptake.UPTAKE_ARTIFACT_DIR
    assert paths.LLOYD_HOME in uptake_path.parents, "uptake defaults into the code tree"
    assert paths.DATA_ROOT not in uptake_path.parents
    # `landed_outcomes` is the arm that proves the state-dir case now that
    # `automod_scorecard` is retired (#4169): same dir, same reason it is not an
    # `app.paths` constant.
    assert landed_outcomes.landed_outcomes_path().parent == state.STATE_DIR
    assert paths.DATA_ROOT not in state.STATE_DIR.parents, (
        "the automod ledger dir is outside the data root")
    # And the five that ARE under the data root, so the table spans both worlds.
    # `behaviour_exhibit` joined them in #3886, and it is the one whose whole purpose is
    # living outside the tree: an artifact the sweep wrote into the repo would be a
    # tracked-file stray that `app/live_strays.py` counts, which
    # `tests/test_behaviour_exhibit_sweep.py` pins from the writer's side.
    for key in ("retrieval_nightly", "tool_quality", "funnel_census", "iv_metrics",
                "behaviour_exhibit"):
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
    holding `landed_outcomes.jsonl`) and the one inside the code tree (uptake's default)
    are in the row, with the newest date each actually holds.

    The state-dir store this test seeded was `scorecard.jsonl` until #4169 retired that
    arm; `landed_outcomes` is the arm that still lives in `state.STATE_DIR`, so the case
    — a store `DATA_ROOT` does not reach, resolved through its writer's own constant —
    is proved by an arm the table actually carries."""
    state_dir = tmp_path / "state" / "lloyd-automod"
    code_tree = tmp_path / "lloyd"
    uptake_dir = code_tree / "eval" / "uptake"
    _seed_series(state_dir, "landed_outcomes.jsonl", [
        {"generated_at": "2026-09-11T02:45:18+00:00",
         "by_author": {"agent": {"commits": 40, "later_undo_7d": {
             "k": 20, "n": 40, "rate": 0.5, "ci_low": 0.35, "ci_high": 0.65}}}},
        {"generated_at": "2026-10-12T20:03:00+00:00",
         "by_author": {"agent": {"commits": 50, "later_undo_7d": {
             "k": 31, "n": 50, "rate": 0.62, "ci_low": 0.48, "ci_high": 0.74}}}}])
    _seed(uptake_dir, "uptake_probe", "2026-09-11", 0.9)
    _seed(uptake_dir, "uptake_probe", "2026-09-20", 1.0)

    roots = _seed_all(tmp_path, GOOD_VALUES)
    roots["landed_outcomes"] = state_dir / "landed_outcomes.jsonl"
    roots["uptake_probe"] = uptake_dir
    out = tmp_path / "index"
    row = EII.build_row(now=NOW, root_overrides=roots, out_dir=out,
                        code_root=code_tree)

    landed = _component(row, "landed_outcomes")
    assert landed["newest_artifact"] == "2026-10-12", "the newest of the two rows"
    assert landed["value"] == pytest.approx(0.62)
    assert landed["outside_data_root"] is True
    assert landed["instrument_path"] == str(state_dir)

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

#: The four files allowed to name the index: the publisher itself, `app/paths.py`
#: where the store is declared, this file — whose whole job is to prove the rule, and
#: which cannot prove it without spelling the name — and, since #3979, the doc-claims
#: suite, which imports the arm *declarations* to grade §Watch the machinery's
#: read-back sentences against the code instead of against the doc's memory of it.
#: Everything else that reads the published row would be a gate, a promotion rule or a
#: rollback path, which v1 forbids. The fourth entry is the only widening of that rule
#: and it is narrowed by name, not trusted:
#: `test_the_fourth_declared_reader_touches_only_the_arm_declarations` binds it to the
#: one attribute it was granted, so being allowed to name the module never becomes a
#: route to the row.
DECLARED_IN = frozenset({
    "scripts/eval_instrument_index.py",
    "app/paths.py",
    "tests/test_eval_instrument_index.py",
    "tests/test_autonomy_jobs_doc_claims.py",
})

#: The declaration-only reader, and the single attribute of the publisher it may use.
DECLARED_DECLARATION_READER = "tests/test_autonomy_jobs_doc_claims.py"
GRANTED_SURFACE = frozenset({"INSTRUMENTS"})

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
                  "tests/test_eval_instrument_index.py",
                  # The declared declaration-only reader: the walk has to reach it, so
                  # that a future edit removing it from DECLARED_IN is caught by the scan
                  # rather than silently shrinking the set of files it ever opened.
                  "tests/test_autonomy_jobs_doc_claims.py")


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


def _publisher_surface_used_by(path: Path) -> set[str]:
    """Which names of the publisher one file actually touches.

    Three spellings, because that is how a widening would arrive: a name imported
    straight out of the module is granted outright, a module bound under an alias is
    used through its attributes, and prose about the module is nothing at all. An
    allow-list entry says a file may name the module; what the v1 boundary is about is
    what it then takes from it, so the grant has to be measured on attributes and
    imports rather than on the import statement.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    aliases: set[str] = set()
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            direct = (node.module or "").endswith("eval_instrument_index")
            for alias in node.names:
                if alias.name.endswith("eval_instrument_index"):
                    aliases.add(alias.asname or alias.name)
                elif direct:
                    used.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.endswith("eval_instrument_index"):
                    aliases.add(alias.asname or alias.name)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in aliases):
            used.add(node.attr)
    return used


def test_the_fourth_declared_reader_touches_only_the_arm_declarations(tmp_path):
    """#3979 widened `DECLARED_IN` by one file; this node is what makes that an entry
    rather than a hole in the v1 boundary.

    §Watch the machinery's read-back sentences are graded against the arm tuple now, so
    the doc-claims suite imports the publisher — and an import is a read, which is why
    the whole-tree scan above had to be told the file exists. What it was granted is one
    attribute: `INSTRUMENTS`, the declarations. The published row stays the publisher's
    alone, so the surface that file uses is pinned to `GRANTED_SURFACE`, and a later edit
    that reaches for `read_instrument`, `build_row` or `main` reds here, in the file that
    owns the rule, instead of widening it through the allow-list. The three seeded files
    are the controls: without them an empty return from the helper — the failure mode
    `_index_reads_in` was itself written against — would satisfy the equality as well as a
    correct reading does.
    """
    reader = ROOT / DECLARED_DECLARATION_READER
    assert reader.is_file(), (
        f"{DECLARED_DECLARATION_READER} is declared as a reader but is not in this tree, "
        "which leaves the grant unowned and the boundary wider than it reads")
    used = _publisher_surface_used_by(reader)
    assert used == GRANTED_SURFACE, (
        f"{DECLARED_DECLARATION_READER} may name the publisher for one attribute and now "
        f"uses {sorted(used) or 'nothing'}; anything outside {sorted(GRANTED_SURFACE)} is "
        "the row side of the module, which v1 keeps to the publisher")

    imported_direct = tmp_path / "widened_import.py"
    imported_direct.write_text(
        "from scripts.eval_instrument_index import read_instrument\n\n"
        "def read(i):\n    return read_instrument(i)\n", encoding="utf-8")
    assert _publisher_surface_used_by(imported_direct) == {"read_instrument"}, (
        "a name imported out of the publisher is not reported as a use, so the grant "
        "above could be widened by an import that never shows up")

    via_alias = tmp_path / "widened_attr.py"
    via_alias.write_text(
        "from scripts import eval_instrument_index as eii\n\n"
        "def build():\n    return eii.build_row(eii.INSTRUMENTS)\n", encoding="utf-8")
    assert _publisher_surface_used_by(via_alias) == {"build_row", "INSTRUMENTS"}, (
        "an attribute read through an alias is not reported, which is exactly the spelling "
        "the real reader uses and the one a widening would keep")

    prose_only = tmp_path / "prose_only.py"
    prose_only.write_text(
        '"""The publisher lives in scripts/eval_instrument_index.py."""\n\nX = 1\n',
        encoding="utf-8")
    assert _publisher_surface_used_by(prose_only) == set(), (
        "a file that only names the module in prose is reported as using its surface, so "
        "a comment would redden this node and the next reader starts trusting the "
        "allow-list instead of the surface")


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
    assert len(rows[0]["components"]) == 8, (
        "the subprocess publishes the whole table, arms carried by ruling included — "
        "nine since #3886 added the ninth, eight since #4169 retired the scorecard arm")


def test_the_module_cli_exits_1_and_publishes_the_refusal_when_one_arm_survives(tmp_path):
    """The other exit branch on the same seam: seven of the eight instruments
    missing, so the totals refuse, the refusal is still published as the day's
    record, and the exit code is 1 — because `main()` returning 0 whatever the
    totals said is how a scheduler ends up reading an index that measured nothing
    as a green run. Seven, not six, because #3541 grew the table to eight arms,
    #3886 to nine and #4169 retired `automod_scorecard` back to eight, and
    `gate_to_land_lag` and `behaviour_exhibit` are both among the
    omitted ones: neither is seeded here, and a fixture that left either reading its
    live store would make this run depend on a real artifact instead of a snapshot —
    for the exhibit arm that would mean reading a store whose only artifact is dated
    2026-11-04 and silently deciding the fixture's staleness from it."""
    out = tmp_path / "cli-refused"
    proc = _run_publisher(
        "--roots", str(_roots_file(
            tmp_path, {"retrieval_nightly": 1.0},
            omit=("iv_metrics", "landed_outcomes",
                  "tool_quality", "uptake_probe", "funnel_census",
                  "gate_to_land_lag", "behaviour_exhibit"))),
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


# ── #3886: the exhibit store gets ONE name, and a missed day gets ONE line ─────
#
# The sweep landed 2026-11-04, wrote one artifact, and was never run again. Nothing
# could have noticed: the store was named by no constant in `app.paths` — the writer
# recomputed `DATA_ROOT / "eval" / "behaviour-exhibit"` inside `out_dir()`, behind an
# `except Exception` that fell back to `Path.home() / "lloyd-data"` — and it was keyed
# in no table, so no reader could call it missing. That is #2275's tool-quality fate
# arm for arm, and the only difference is that tool_quality IS a row here, which is the
# sole reason its death is visible (`ls ~/lloyd-data/eval/tool-quality/` says `No such
# file or directory` and the index says `missing or stale: tool_quality`). These four
# tests are the two halves of the fix: one path (clause 1) and one printed line (clause
# 2). The comparison the arm's artifact carries is `app.behaviour_exhibit`'s own job —
# clauses 4 and 5 are pinned in tests/test_behaviour_exhibit_sweep.py.

#: The live store's only artifact, worker cells for `tool-call-caption-present`:
#: 141,558 exhibited of 156,286 situations = 0.9058, Wilson [0.9043, 0.9072].
_EXH_N = 156286
_EXH_K = 141558


def _exhibit_row(bid: str, k: int, n: int) -> dict:
    """One per-behaviour row, in the shape `app.behaviour_exhibit.sweep` emits it — the
    worker cells carry `exhibited` and `situations_encountered`, which is what
    `worker_pooled` sums. A fixture that renamed either would silently pool to zero."""
    return {"id": bid, "status": "measured",
            "by_population": {"worker": {
                "situations_encountered": n, "exhibited": k, "status": "measured"}}}


def _exhibit_artifact(rows: list[dict], *, measured_at: str) -> dict:
    """An artifact built with the WRITER's own pooling, not a hand-typed imitation.

    The reason to call into `app.behaviour_exhibit` here rather than write the dict: the
    index reads `worker_pooled`, so if the writer ever renames that field or changes how
    `k`/`n` are summed, this file fails on the seam instead of publishing a `no-value`
    row that looks like a missed day.
    """
    import app.behaviour_exhibit as bx
    return {"schema": 1, "item": "#3749", "measured_at": measured_at,
            "window_days": 7, "rows": rows, "worker_pooled": bx.worker_pooled(rows)}


def _plant_exhibit(store: Path, doc: dict, *, day: str) -> Path:
    store.mkdir(parents=True, exist_ok=True)
    path = store / f"behaviour-exhibit-{day}.jsonl"
    path.write_text(json.dumps(doc) + "\n", encoding="utf-8")
    return path


def test_the_exhibit_store_has_one_name_shared_by_its_writer_and_its_readers():
    """Clause 1 as a shared name, checked in the one file that can see all three parties.

    `app.paths` declares the store, `app.behaviour_exhibit.out_dir()` returns exactly that
    object, and this index's directory accessor resolves to it too. Before #3886 the
    middle and third legs could not hold: there was no constant, and `out_dir()` derived
    the path itself — so the index and the sweep each had to spell `eval/behaviour-exhibit`
    and could drift apart with nothing to notice.

    The removed fallback is asserted textually, because a behavioural test cannot reach it:
    on a box where `app.paths` imports cleanly the old `except Exception: base =
    Path.home() / "lloyd-data"` never fires, so the suite was green over a branch that
    would have written the daily artifact into a layout this machine does not use.
    """
    import app.paths as p
    import app.behaviour_exhibit as bx

    want = p.DATA_ROOT / "eval" / "behaviour-exhibit"
    assert p.BEHAVIOUR_EXHIBIT_DIR == want, (
        "the store is named under DATA_ROOT or it is not the store the rest of the "
        f"system can name; found {p.BEHAVIOUR_EXHIBIT_DIR}")
    assert bx.out_dir() == want
    assert bx.out_dir(None) == want, "the no-flag case is the scheduled run's case"
    assert bx.out_dir(Path("/tmp/planted/elsewhere.jsonl")) == Path(
        "/tmp/planted/elsewhere.jsonl"), "an explicit --out still wins"
    assert _instr("behaviour_exhibit").artifact() == want
    assert want not in Path.home().parents and str(want).startswith(str(p.DATA_ROOT))

    # The removed fallback, checked on the CODE and not the prose: `out_dir`'s own
    # docstring names `Path.home() / "lloyd-data"` to explain what used to be there, so a
    # plain substring scan of the function text would trip on its own history. The AST's
    # statements after the docstring are the behaviour.
    import ast
    import inspect

    src = inspect.getsource(bx.out_dir)
    fn = ast.parse(textwrap.dedent(src)).body[0]
    code = "\n".join(ast.get_source_segment(textwrap.dedent(src), st) or ""
                     for st in fn.body[1:])
    assert "home" not in code and "except" not in code, (
        "`out_dir()` grew back a fallback: an instrument that cannot name its own store "
        f"must raise, not quietly write somewhere else. Code was:\n{code}")
    assert "BEHAVIOUR_EXHIBIT_DIR" in code, (
        "out_dir() must RETURN the constant, not rejoin the path from DATA_ROOT — the "
        "second spelling is the bug this clause removes")
    assert code.strip().endswith("return Path(BEHAVIOUR_EXHIBIT_DIR)"), code


def test_a_day_with_no_exhibit_artifact_prints_it_in_headline_and_table(tmp_path):
    """Clause 2's observable, in both places a reader could look. See the sibling docstring."""
    others = {k: v for k, v in GOOD_VALUES.items() if k != "behaviour_exhibit"}
    row, _, _ = _row(tmp_path, others, omit=("behaviour_exhibit",))
    assert [d["key"] for d in row["dead_or_stale"]] == ["behaviour_exhibit"], (
        "the exhibit store is the one arm with no artifact, which is what the item "
        "measured on 2026-11-07: `ls ~/lloyd-data/eval/behaviour-exhibit/` -> one file, "
        "dated 2026-11-04")
    assert row["headline"].endswith("missing or stale: behaviour_exhibit"), row["headline"]
    rendered = EII.render(row)
    assert any(line.startswith("behaviour_exhibit") for line in rendered.splitlines()), (
        "the key must appear as a row of the printed table, not only inside the headline")

    # Two days old against a 1-day cadence: present, and still a missed run.
    stale = tmp_path / "stale-store"
    _plant_exhibit(stale, _exhibit_artifact(
        [_exhibit_row("tool-call-caption-present", _EXH_K, _EXH_N)],
        measured_at="2026-10-11T02:24:50+00:00"), day="2026-10-11")
    reading = EII.read_instrument(_instr("behaviour_exhibit"),
                                  today=NOW.date(), root_override=stale)
    assert reading.status == "stale" and reading.age_days == 2, vars(reading)
    assert "cadence of 1 d" in reading.detail, reading.detail
    assert "109-behaviour-exhibit-daily" in reading.detail, (
        "the line that reports a missed day must also name who was supposed to run it — "
        "that pairing is what makes `stale` actionable instead of a warning")


def test_the_exhibit_arm_reads_the_pair_its_writer_publishes(tmp_path):
    """The arm's pointers, proved against the writer's own `worker_pooled`.

    Values are the live 2026-11-04 artifact's worker cells: pooled alone over the caption
    arm the rate is 141,558 / 156,286 = 0.9058 and the interval is the writer's own
    [0.9043, 0.9072], so the index cannot disagree with the file it reports. The pair
    shape (`{value, ci_low, ci_high}`) is `landed_outcomes`' rule for the same reason: a
    re-derived interval silently re-scales to whatever N the reader assumed. The empty
    case is asserted too — a week with no scored situation is NOT a rate of zero, which
    is also why the user population (29 turns against 13,903 on the 11-04 witness, per
    #3749's owed-check ruling of 2026-11-07T10:34) is not pooled here at all.
    """
    import app.behaviour_exhibit as bx

    store = tmp_path / "exhibit-store"
    doc = _exhibit_artifact(
        [_exhibit_row("tool-call-caption-present", _EXH_K, _EXH_N)],
        measured_at="2026-10-12T02:24:50+00:00")
    assert doc["worker_pooled"]["k"] == _EXH_K and doc["worker_pooled"]["n"] == _EXH_N
    assert doc["worker_pooled"]["behaviours_scored"] == 1
    _plant_exhibit(store, doc, day=FRESH)
    reading = EII.read_instrument(_instr("behaviour_exhibit"),
                                  today=NOW.date(), root_override=store)
    assert reading.status == "measured", vars(reading)
    assert reading.value == pytest.approx(_EXH_K / _EXH_N, abs=1e-6), vars(reading)
    assert reading.denominator == _EXH_N
    assert reading.ci95 == pytest.approx(
        [doc["worker_pooled"][k] for k in ("ci_low", "ci_high")], abs=2e-6)
    lo, hi = bx.wilson_interval(_EXH_K, _EXH_N)
    assert reading.ci95[0] == pytest.approx(lo, abs=2e-6), (
        "the published interval IS wilson_interval's, not a rounded rate plus a constant")
    assert (hi - lo) / 2 * 100 == pytest.approx(0.1449, abs=0.005), (
        "the caption arm's own half-width at N=156,286, in pp — the figure #3886 "
        "correction 7 puts at ±0.15pp where the item text said ±1.5pp")

    empty = _exhibit_artifact(
        [{"id": "x", "status": "not scored: detector cannot read the transcript",
          "by_population": {"worker": {"situations_encountered": 0, "exhibited": 0,
                                       "status": "not scored: detector cannot read"}}}],
        measured_at="2026-10-12T02:24:50+00:00")
    assert empty["worker_pooled"]["rate"] is None, "an empty denominator is not a zero"
    _plant_exhibit(store, empty, day=FRESH)
    reading = EII.read_instrument(_instr("behaviour_exhibit"),
                                  today=NOW.date(), root_override=store)
    assert reading.status == "no-value", vars(reading)
    assert reading.value is None, (
        "an empty denominator reads as NO VALUE, never as a rate of 0.0 — a zero would "
        "enter the arm's history as a collapse in exhibit behaviour that did not happen")
    assert reading.status in ("missing", "stale", "no-value"), (
        "and that status is one `build_row` lists as dead or stale, so the day is named")


def test_the_exhibit_arm_is_carried_in_the_table_and_excluded_from_both_totals(tmp_path):
    """Reported, never averaged — proven by arithmetic, per the #3541 precedent.

    Same fixtures, published twice: once over the whole table and once with
    `behaviour_exhibit` removed from it. The two composites must be equal to the digit,
    which is the only thing that makes "reported, not averaged" a property instead of a
    comment. The arm is nevertheless present in the table with its own unit — being
    present is what makes a dead store a printed line.
    """
    with_arm, _, _ = _row(tmp_path / "with", GOOD_VALUES)
    without = [i for i in EII.INSTRUMENTS if i.key != "behaviour_exhibit"]
    assert sum(i.weight for i in without) == pytest.approx(1.0), (
        "the six weighted arms still sum to 1.0 — the new one adds no weight")
    without_arm, _, _ = _row(tmp_path / "without", GOOD_VALUES, instruments=without)
    assert (with_arm["totals"]["declared_weights"]
            == without_arm["totals"]["declared_weights"]), (
        "a share pooled over denominators six orders of magnitude apart may not move "
        "the composite")
    assert (with_arm["totals"]["equal_weights"]
            == without_arm["totals"]["equal_weights"])
    assert len(without_arm["components"]) == 7 and len(with_arm["components"]) == 8
    assert _component(with_arm, "behaviour_exhibit")["contributes"] is False
    assert any(x["key"] == "behaviour_exhibit" for x in with_arm["totals"]["excluded"])
    assert with_arm["totals"]["status"] == without_arm["totals"]["status"] == "ok"


# ── #4169: the automod_scorecard arm is RETIRED, not left to age out ──────────
#
# `~/.local/state/lloyd-automod/scorecard.jsonl` has had no writer since its newest
# row, 2026-09-12: the only code path that appends it is the `--record` flag of
# `python -m scripts.automod.round scorecard`, which no autonomy task file, crontab
# entry, systemd unit or workers.db queue row invokes. So the arm published `stale`
# every day and its 0.18 was renormalised away every day, while the headline kept
# printing two totals over whatever survived. Two ways out existed — name a writer,
# or remove the arm — and the ruling recorded below in the module source is removal.
# The third option the item suggested (`hand_run_by_ruling=True`, as `uptake_probe`
# carries) is the wrong one and is NOT taken here: that flag suppresses the staleness
# test outright, so a 65-day-old 0.7 would have come back into the composite as if it
# were current. These three tests are the retirement made checkable: the key is gone
# from the row, the weights still add up, and the ruling is in the source where the
# next reader of the table will find it.

#: The declared weights BEFORE #4169, spelled out so the redistribution arithmetic
#: below can be checked from this block alone. The retired arm carried 0.18, so the
#: six surviving weighted arms summed to 0.82 and each was scaled by 1/0.82.
_PRE_4169_WEIGHTS = {"retrieval_nightly": 0.28, "iv_metrics": 0.20,
                     "automod_scorecard": 0.18, "landed_outcomes": 0.14,
                     "tool_quality": 0.10, "uptake_probe": 0.05,
                     "funnel_census": 0.05}
#: The sum the six survivors carried, and the arm that absorbs the rounding residue.
_SURVIVOR_WEIGHT_SUM = 0.82
_RESIDUE_ARM = "retrieval_nightly"


def test_the_retired_scorecard_arm_appears_nowhere_in_a_published_row(tmp_path):
    """Clause 1 (#4169): the arm is out of the table, so a published row carries
    eight components and never names `automod_scorecard` at all.

    The absence is checked over the WHOLE serialised row rather than over
    `components` alone, because the state the item was in for 65 days was precisely
    "not scored but still named": the key sat in `totals.excluded` and in
    `dead_or_stale` with a reason string, which is what made the exclusion visible
    but also permanent. A row that still lists it under either heading is an arm
    ageing out, not an arm retired."""
    keys = {i.key for i in EII.INSTRUMENTS}
    assert "automod_scorecard" not in keys
    assert len(EII.INSTRUMENTS) == 8 == len(ALL_KEYS), (
        "eight arms: nine minus the retired one")

    row, _, _ = _row(tmp_path, GOOD_VALUES)
    assert len(row["components"]) == 8, (
        "a published row's components list holds exactly eight entries")
    assert {c["key"] for c in row["components"]} == keys
    blob = json.dumps(row)
    assert "automod_scorecard" not in blob, (
        "the key must name nothing in the row — not a component, not an entry of "
        "`totals.excluded`, not one of `dead_or_stale`")
    assert not any(c["key"] == "automod_scorecard" for c in row["totals"]["excluded"])
    assert not any(d["key"] == "automod_scorecard" for d in row["dead_or_stale"])


def test_the_eight_declared_weights_sum_to_one_with_the_retired_share_redistributed(tmp_path):
    """Clause 2 (#4169): dropping an arm must not leave the table a set of numbers
    that stopped adding up.

    The rule is the item's: the retired 0.18 goes back to the survivors in
    proportion to the weight each already carried (its old share over the 0.82 the
    survivors summed to), each rounded to two decimals, and the residue lands on the
    largest arm. Proportional rather than a flat split because these weights are
    judgements about reach and coverage, and a flat split would silently promote
    `funnel_census` and `uptake_probe` — the two arms the table calls thinnest —
    onto the same footing as the ones that see every prompt and every turn."""
    weights = {i.key: i.weight for i in EII.INSTRUMENTS}
    assert len(weights) == 8
    assert weights[_RESIDUE_ARM] == max(weights.values()), (
        "the residue goes to the largest arm, so the arm that takes it has to be it")

    survivors = {k: v for k, v in _PRE_4169_WEIGHTS.items()
                 if k not in (_RESIDUE_ARM, "automod_scorecard")}
    for key, old in survivors.items():
        want = round(old / _SURVIVOR_WEIGHT_SUM, 2)
        assert weights[key] == pytest.approx(want, abs=1e-9), (
            f"{key} carried {old} of {0.82}, so its share of the redistribution is "
            f"{old / _SURVIVOR_WEIGHT_SUM:.6f} rounded to {want}")
    assert weights[_RESIDUE_ARM] == pytest.approx(
        1.0 - sum(weights[k] for k in survivors), abs=1e-9), (
        "the largest arm takes exactly what the other five roundings leave")
    assert weights["gate_to_land_lag"] == 0.0 and weights["behaviour_exhibit"] == 0.0, (
        "the two arms carried by ruling stay unweighted; a redistribution is not the "
        "place to change that")

    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)
    row, _, _ = _row(tmp_path, GOOD_VALUES)
    assert row["weights_sum_declared"] == 1.0, (
        "published exactly, not approximately: this is the index's own honesty field, "
        "and it is the field that would have printed 0.82 the day the arm went")


def _instrument_table_source(source: str) -> str:
    """The source text of the `INSTRUMENTS` assignment plus the comment block
    attached directly above it — what "beside the table" means, as the file itself
    defines it rather than as a substring the check happens to find."""
    tree = ast.parse(source)
    assign = next(n for n in tree.body
                  if isinstance(n, (ast.Assign, ast.AnnAssign))
                  and getattr(getattr(n, "target", None), "id", None) == "INSTRUMENTS")
    lines = source.splitlines()
    start = assign.lineno - 1
    while start > 0 and (lines[start - 1].lstrip().startswith("#")
                         or not lines[start - 1].strip()):
        start -= 1
    return "\n".join(lines[start:assign.end_lineno])


def test_the_module_source_records_the_retirement_ruling_beside_the_table():
    """Clause 3 (#4169): the ruling lives in the module, next to the thing it
    retired, with the route back in.

    The pairing is the deliverable. A `git log` line is not this: the next reader
    of the table reads the table, and the only reason this arm was still published
    six weeks after its store died is that nothing anywhere said whether it was
    meant to be alive."""
    source = (ROOT / "scripts" / "eval_instrument_index.py").read_text(encoding="utf-8")
    span = _instrument_table_source(source)

    assert "retrieval_nightly" in span, (
        "positive control: the span must cover the table itself, or every absence "
        "below would be a reading of an empty slice")
    assert "automod_scorecard" in span, (
        "the retirement has to be named where the table is declared")
    assert "automod_scorecard" not in {i.key for i in EII.INSTRUMENTS}, (
        "and the key must be named as history, not carried as an arm")
    assert "scorecard.jsonl" in span, "the ruling names the store it is about"
    assert "2026-09-12" in span, "the ruling names the newest row on disk"
    assert "scripts.automod.round scorecard --record" in span, (
        "the ruling names the command a future writer would run")
    assert "scheduled_by" in span, (
        "and the field to set when it comes back, so a re-added arm cannot be "
        "neither-written-nor-retired a second time")


def test_no_test_before_the_retirement_block_selects_the_retired_arm_by_key():
    """Clause 4 (#4169): the fixture machinery must not keep selecting an arm the
    table no longer carries.

    Checked on the syntax tree rather than by substring, because the distinction that
    matters is a selector versus prose: a test that says "the arm #4169 retired" in a
    comment is the record this item asked for, while `_component(row, "automod_"
    "scorecard")` is a test still asserting an arm's behaviour against a fixture and a
    table that no longer has one. So the test is a STRING CONSTANT EQUAL TO THE BARE KEY —
    that is what a dict-literal key, a `key ==` comparison, an `_instr(...)`/`_component(...)`
    argument and an `omit=` tuple member all compile to, and no sentence about the arm
    compiles to it. Every such constant in the file sits in the retirement block below,
    where its only job is to prove the key is gone.
    """
    path = ROOT / "tests" / "test_eval_instrument_index.py"
    source = path.read_text(encoding="utf-8")
    banner = source.index("# ── #4169")
    banner_line = source[:banner].count("\n") + 1

    selected = sorted({(node.lineno, node.value)
                       for node in ast.walk(ast.parse(source))
                       if isinstance(node, ast.Constant)
                       and node.value in {i.key for i in EII.INSTRUMENTS}
                       | {"automod_scorecard"}
                       and node.lineno < banner_line})
    keys_before = {k for _, k in selected}
    assert keys_before == {i.key for i in EII.INSTRUMENTS}, (
        f"keys selected by constant before the retirement block: {sorted(keys_before)} — "
        "it must be exactly the eight arms the table carries, so that no fixture still "
        "selects the retired one and no absence list quietly dropped a live arm either")
    assert "automod_scorecard" not in keys_before

    after = {node.value for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Constant) and node.lineno >= banner_line
             and node.value == "automod_scorecard"}
    assert after == {"automod_scorecard"}, (
        "positive control: the key IS still a constant in the file, below the banner, "
        "so the check above reads a file that mentions it rather than one that lost it")


# ── #4334: the exhibit arm names the window its artifact was actually swept on ───
#
# #3885 (`199d319b`) gave the sweep `--since`/`--until`/`--split-on` AND published which
# mode ran, as `window_mode`, beside the rates in the artifact. #3886 registered the arm
# in this index but wired it to the STORE only — `app.paths.BEHAVIOUR_EXHIBIT_DIR`, the
# date in the filename, `worker_pooled` — so an artifact swept on an absolute content
# window lands in the same dated store under the same `behaviour-exhibit-YYYY-MM-DD.jsonl`
# name and gets published as an ordinary 7-day mtime row. The scheduled job is barred
# from those flags by its own prompt (`autonomy/109-behaviour-exhibit-daily.md`) and
# nothing in code enforces the bar, while `out_dir()` routes a hand run's artifact into
# that same store whatever the mode: the read side is therefore where the reading is
# fixed, and read-side only — a write-time refusal is ruled out by the item on zero
# measured instances (0 `content` artifacts out of the 8 dated ones on this box).
#
# `resolve_window` returns exactly two modes, `"mtime"` (`app/behaviour_exhibit.py:426`)
# and `"content"` (`:430`), and `199d319b` is the commit that introduced the key. So an
# artifact with NO `window_mode` is not an anomaly: it predates the key, and the only
# window that could produce it is the mtime one. It is named as legacy for that reason —
# a reader comparing it against a content-windowed row needs to know which of the two
# clocks admitted its files, and "silently a 7-day row" hides the key's own birthday.
#
# Every fixture below is a `tmp_path` store handed in through `root_overrides`, exactly
# as the four #3886 tests above are: no test here writes to the live dated store.

def _exhibit_swept_on(mode: "str | None", *, day: str = FRESH) -> dict:
    """One dated exhibit artifact in the shape `sweep` publishes for one window MODE.

    Built on `_exhibit_artifact`, so the pooled scalar is still the writer's own. What
    varies is the window block the writer publishes beside it: `window_mode` is
    `Window.as_dict()["mode"]`; `window_days` is the caller's knob ONLY in mtime mode,
    because `--days 7` printed beside a ten-day `--since 2026-10-26 --until 2026-11-05`
    run is a lie the writer stopped telling in `199d319b`; and the endpoints are the
    record that made a run re-derivable after the day it ran on. `mode=None` is the
    pre-#3885 shape — `window_days: 7` and no mode key at all, which is exactly what
    `/home/alansrobotlab/lloyd-data/eval/behaviour-exhibit/behaviour-exhibit-2026-11-04.jsonl`
    still carries.
    """
    doc = _exhibit_artifact(
        [_exhibit_row("tool-call-caption-present", _EXH_K, _EXH_N)],
        measured_at=f"{day}T02:24:50+00:00")
    if mode is None:
        return doc
    doc["window_mode"] = mode
    if mode == "mtime":
        doc["window_days"] = 7
        doc["window_start"], doc["window_end"] = f"{day}T00:00:00+00:00", None
    else:
        doc["window_days"] = None
        doc["window_start"], doc["window_end"] = ("2026-10-26T00:00:00+00:00",
                                                 "2026-11-05T00:00:00+00:00")
    return doc


def _table_line(row: dict, key: str) -> str:
    """One arm's printed line of the rendered table — the surface a person reads."""
    line = next((ln for ln in EII.render(row).splitlines()
                 if ln.startswith(key + " ")), None)
    assert line is not None, f"{key} is not a row of the printed table:\n{EII.render(row)}"
    return line


def test_the_window_rule_names_a_mode_only_when_it_is_not_the_default_one():
    """`_window_note` is the whole rule, so the rule itself is pinned — every branch.

    `resolve_window` (`app/behaviour_exhibit.py:426` and `:430`) returns `"mtime"` or
    `"content"` and nothing else, so the mtime arm is the one that must stay silent: it is
    7 of the 8 dated artifacts on this box and every one the daily job writes, and an
    appended note there is clause 3 failing. The empty-string arm is in the same class as
    an absent key — a field that records no mode — while a value that is PRESENT but is
    not a mode string is named by its repr, never called absent: "absent" about a value
    that is on disk would be the same misreading in the other direction.
    """
    assert EII._window_note("mtime") == "", EII._window_note("mtime")
    assert EII.DEFAULT_WINDOW_MODE == "mtime", (
        "the default is pinned to the mode the sweep's own `Window` class uses")
    note = EII._window_note("content")
    assert "window_mode=content" in note and "--since" in note, note
    assert "not comparable" in note, note
    for nothing in (None, "", "   "):
        absent = EII._window_note(nothing)
        assert "window_mode absent" in absent and "legacy" in absent, (repr(nothing), absent)
        assert "#3885" in absent and "199d319b" in absent, (repr(nothing), absent)
    weird = EII._window_note(7)
    assert "window_mode=7" in weird and "absent" not in weird, weird


def test_a_content_window_exhibit_artifact_names_its_mode_in_the_published_row(tmp_path):
    """Clause 1 (#4334): a `window_mode: "content"` artifact is published as one.

    The reading's NUMBER is untouched — the arm still reports the writer's pooled rate
    over its own denominator — because this is a naming fix, not a rescoring: what the
    clause forbids is the row reading as a plain 7-day window, and what makes it a
    different row is the mode, stated. Both published surfaces are checked: the serialised
    component's own field and the printed table line, because the item's word is
    "published line" and a field nobody renders is the failure #460 already named here.
    """
    store = tmp_path / "content-store"
    doc = _exhibit_swept_on("content")
    assert doc["window_mode"] == "content" and doc["window_days"] is None, (
        "the fixture is the content shape: no `window_days`, because that key is only "
        "true of the mtime window")
    path = _plant_exhibit(store, doc, day=FRESH)

    reading = EII.read_instrument(_instr("behaviour_exhibit"), today=NOW.date(),
                                  root_override=store)
    assert reading.status == "measured", vars(reading)
    assert reading.value == pytest.approx(_EXH_K / _EXH_N, abs=1e-6), vars(reading)
    assert reading.denominator == _EXH_N
    assert "window_mode=content" in reading.detail, reading.detail
    assert "window_mode=content" in reading.window_note, reading.window_note
    assert reading.detail != (f"newest artifact {path.name}, 1 d old"), (
        "the plain-window string this arm used to publish is gone, byte for byte, over a "
        "content artifact — that string is what clause 1 says may not be the reading")

    row = EII.build_row([_instr("behaviour_exhibit")], now=NOW,
                        root_overrides={"behaviour_exhibit": store},
                        out_dir=None, code_root=tmp_path / "code")
    assert _component(row, "behaviour_exhibit")["window_note"] == reading.window_note, (
        "the published row carries what the reader saw")
    line = _table_line(row, "behaviour_exhibit")
    assert "window_mode=content" in line, line
    assert "--since" in line, (
        "the line has to say how such an artifact gets into a dated store, or a reader "
        "who never ran the flags cannot tell a hand sweep from a broken scheduler")

    # The `-m` seam a person actually runs, across the process boundary: the mode has to
    # reach the printed stdout and not merely the in-process dict. `--no-record` so the
    # run appends nothing to any series, and `--roots` so it reads these fixtures and
    # never the live dated store.
    roots = _seed_all(tmp_path / "cli", GOOD_VALUES)
    roots["behaviour_exhibit"] = store
    snap = tmp_path / "cli-roots.json"
    snap.write_text(json.dumps({k: str(v) for k, v in roots.items()}), encoding="utf-8")
    proc = _run_publisher("--now", "2026-10-13T03:40:00+00:00", "--roots", str(snap),
                          "--no-record", "--print")
    assert proc.returncode == 0, proc.stderr[-500:]
    cli_line = next((ln for ln in proc.stdout.splitlines()
                     if ln.startswith("behaviour_exhibit ")), None)
    assert cli_line is not None, proc.stdout
    assert "window_mode=content" in cli_line, cli_line
    assert "window_mode" not in proc.stdout.split("HEADLINE:")[1].splitlines()[0], (
        "the mode belongs to the arm's own line: the headline is the composite, and the "
        "arm carries no weight in it")


def test_a_pre_3885_exhibit_artifact_with_no_window_mode_is_named_legacy(tmp_path):
    """Clause 2 (#4334): no `window_mode` key is named as legacy, never as an ordinary row.

    The distinction the clause is built on: the key arrived with `199d319b`, so an
    artifact that predates it was definitionally an mtime run. Reporting it as an
    ordinary 7-day row hides that it was written before the window was recorded at all;
    reporting it as a NON-mtime anomaly would call a normal historical artifact broken.
    So the note says both halves — the key is absent, and the window it lacks is the
    mtime one.
    """
    store = tmp_path / "legacy-store"
    doc = _exhibit_swept_on(None)
    assert "window_mode" not in doc and doc["window_days"] == 7, (
        "the 2026-11-04 artifact's real shape, re-read from disk: `window_days: 7`, no "
        "`window_mode`, no endpoints")
    path = _plant_exhibit(store, doc, day=FRESH)

    reading = EII.read_instrument(_instr("behaviour_exhibit"), today=NOW.date(),
                                  root_override=store)
    assert reading.status == "measured", vars(reading)
    assert reading.value == pytest.approx(_EXH_K / _EXH_N, abs=1e-6), vars(reading)
    note = reading.window_note
    assert "window_mode absent" in note, note
    assert "legacy" in note and "#3885" in note, note
    assert "199d319b" in note, "the note names the commit that introduced the key"
    assert "mtime" in note, (
        "and it names the window a pre-#3885 artifact did in fact run on, so the reader "
        "is told what the absent key means rather than left to guess")
    assert reading.detail != f"newest artifact {path.name}, 1 d old", (
        "not silently an ordinary window=7d row")

    row = EII.build_row([_instr("behaviour_exhibit")], now=NOW,
                        root_overrides={"behaviour_exhibit": store},
                        out_dir=None, code_root=tmp_path / "code")
    assert "legacy" in _table_line(row, "behaviour_exhibit")


def test_an_mtime_exhibit_store_publishes_byte_identically_to_the_row_before(tmp_path):
    """Clause 3 (#4334): over the live shape — newest artifact `window_mode: mtime` — nothing changes.

    This is the clause that keeps the fix from being a re-labelling of every day of the
    series, so it is pinned at the byte level in three places: the arm's `detail` is
    required to EQUAL the exact string this arm published before the key was read (an
    appended note of any kind fails it), the whole rendered table must not contain the
    word `window_mode` or `legacy` anywhere, and the headline's stale list must not gain
    the arm. The same check was run against the real store before and after the change
    and printed identical output; this test is the hermetic half of that, and it uses no
    file outside `tmp_path`.
    """
    store = tmp_path / "mtime-store"
    doc = _exhibit_swept_on("mtime")
    assert doc["window_mode"] == "mtime" and doc["window_days"] == 7, (
        "7 of the 8 dated artifacts on this box, and every one the daily job writes, "
        "have exactly this shape")
    path = _plant_exhibit(store, doc, day=FRESH)

    reading = EII.read_instrument(_instr("behaviour_exhibit"), today=NOW.date(),
                                  root_override=store)
    assert reading.status == "measured", vars(reading)
    assert reading.window_note == "", vars(reading)
    assert reading.detail == f"newest artifact {path.name}, 1 d old", (
        "byte-for-byte the pre-#4334 string; age 1 day because the fixture is dated "
        "FRESH against the pinned NOW")

    roots = _seed_all(tmp_path, GOOD_VALUES)
    roots["behaviour_exhibit"] = store
    out = tmp_path / "index"
    out.mkdir(parents=True, exist_ok=True)
    row = EII.build_row(EII.INSTRUMENTS, now=NOW, root_overrides=roots, out_dir=out,
                        code_root=tmp_path / "code")
    rendered = EII.render(row)
    assert "window_mode" not in rendered, rendered
    assert "legacy" not in rendered.lower(), rendered
    assert row["headline"].count("behaviour_exhibit") == 0, row["headline"]
    assert row["dead_or_stale"] == [], (
        "a fresh mtime store is not missing and not stale: clause 2 of the item is that "
        "the arm stays out of the headline's dead list exactly as it is today")
    comp = _component(row, "behaviour_exhibit")
    assert comp["status"] == "measured" and comp["window_note"] == "", comp


def test_naming_the_window_leaves_the_arm_named_by_app_paths_at_zero_weight(tmp_path):
    """Clause 4 (#4334): the arm's identity is untouched — one store, no weight, no share.

    #3886 clause 1 is that the store has one name, resolved through the writer's own
    `app.paths.BEHAVIOUR_EXHIBIT_DIR` rather than a path re-spelled here, and #3886's
    ruling is that this arm is carried in the table and excluded from both composites
    (its pooled denominator is caption-arm-dominated). Reading one more field out of the
    artifact may not disturb either: the path pin, `weight == 0.0` and
    `averageable == False` are re-asserted here, and the three pre-existing pins —
    `test_the_exhibit_store_has_one_name_shared_by_its_writer_and_its_readers`,
    `test_the_exhibit_arm_is_carried_in_the_table_and_excluded_from_both_totals` and
    `test_the_eight_declared_weights_sum_to_one_with_the_retired_share_redistributed` —
    pass unchanged. The third half of the clause, that no hunk touches
    `app/behaviour_exhibit.py`, is a property of the diff: `git show --stat` over this
    round's commit names two files, and neither is the writer.
    """
    import app.paths as p

    instr = _instr("behaviour_exhibit")
    assert instr.artifact() == p.BEHAVIOUR_EXHIBIT_DIR, (
        "#3886 clause 1 still holds after the mode read is wired in")
    assert instr.weight == 0.0 and instr.averageable is False
    assert instr.window_mode_path == "window_mode", (
        "the arm declares the pointer the way it declares `value_path`; the reader is "
        "generic over any arm that carries a window, not hard-coded to one key")

    row, _, _ = _row(tmp_path, GOOD_VALUES)
    comp = _component(row, "behaviour_exhibit")
    assert comp["weight"] == 0.0 and comp["averageable"] is False
    assert comp["contributes"] is False and comp["in_total_denominator"] is False
    assert any(x["key"] == "behaviour_exhibit" for x in row["totals"]["excluded"])
