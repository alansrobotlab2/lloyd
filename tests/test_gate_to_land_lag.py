"""The gate→land lag report (#2759): percentiles with their denominator, and the
catch rate that has no ledger row.

Two numbers decide whether a staleness stamp is worth building for the
verdict→action lane, and before #2759 neither was published anywhere: the
gate-pass→promote lag distribution, and how often the landing's state-identity
check has caught the tree moving. Both were computable from `promotions.jsonl`
and its gzip archives all along, so the report adds no instrumentation — it reads
what is already written. What the nodes here pin are the ways a join like this
lies:

  * a percentile without its `n` (an aggregate over three rounds reads like a
    distribution over a thousand);
  * rounds dropped from the join in silence — a `promoted` row with no passing
    `gate` row must be COUNTED in the output, or a thin join reads as a short lag;
  * the 55 % of the real population that lives in `promotions-archive-*.jsonl.gz`,
    so a report reading only the live file describes less than half the corpus;
  * and the one figure the ledger cannot answer: the promote-time identity refuse
    raises before the `try:` that would append an event, so its catch count is
    UNKNOWN, and printing a bare `0` there would be a fabricated clean bill.

The state dir comes from `scripts/automod/state.py` (`LLOYD_AUTOMOD_STATE`), and
one node drives the script as a subprocess with that variable set, because that
hook is the whole reason a test can point the report at a tmp dir at all.
"""
import gzip
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "maintenance" / "gate_to_land_lag.py"


@pytest.fixture(scope="module")
def gtl():
    # scripts/maintenance has no __init__.py — the same loader
    # tests/test_voice_turn_trend.py uses for this directory.
    assert SCRIPT.is_file(), (f"{SCRIPT} is not on disk — the report script moved or "
                              "was renamed, and #2759's four clauses go unpinned")
    spec = importlib.util.spec_from_file_location("gate_to_land_lag", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _gate(round_id, ts, ok=True, rung="review"):
    return {"ts": ts, "created_at": "2026-10-15T00:00:00Z", "event": "gate",
            "round_id": round_id, "rung": rung, "ok": ok, "detail": f"{rung} done"}


def _promoted(round_id, ts, commit="abcd1234"):
    return {"ts": ts, "created_at": "2026-10-15T00:00:00Z", "event": "promoted",
            "round_id": round_id, "commit": commit, "parent": "1111aaaa"}


def _moved(round_id, ts, base="cafe0000"):
    """A preflight `gate` row whose detail carries the catch the item counts."""
    return {"ts": ts, "created_at": "2026-10-15T00:00:00Z", "event": "gate",
            "round_id": round_id, "rung": "preflight", "ok": False,
            "detail": f"live HEAD moved: abc12345 != base {base}, and rebasing onto "
                      f"it conflicts in ['scripts/x.py']"}


def _store(state_dir, live_rows=(), archive_rows=(), raw_lines=(),
           archive_name="promotions-archive-202609.jsonl.gz"):
    """A state dir holding a live ledger and one gzip archive beside it.

    `raw_lines` go into the live file verbatim after the encoded rows: the
    unreadable-line node needs a line that is not JSON, and no dict makes one.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "promotions.jsonl").open("w", encoding="utf-8") as fh:
        for row in live_rows:
            fh.write(json.dumps(row) + "\n")
        for line in raw_lines:
            fh.write(line + "\n")
    with gzip.open(state_dir / archive_name, "wt", encoding="utf-8") as fh:
        for row in archive_rows:
            fh.write(json.dumps(row) + "\n")
    return state_dir


def _report(gtl, state_dir, ceiling=4_500):
    """One report over a synthetic state dir, as (printed text, row dict).

    The ceiling is injected through the same function the script reads it from
    (`scripts.automod.state.landing_cfg`), because clause 1 forbids a literal
    4500 — a test that handed the script a number would be pinning a number the
    script is not allowed to take. `STATE_DIR` is patched too, which is how the
    node keeps the live automod state out of the test's reach.
    """
    from scripts.automod import state as S
    saved, saved_cfg = S.STATE_DIR, S.landing_cfg
    S.STATE_DIR = state_dir
    S.landing_cfg = lambda repo=None: {"idle_hard_max_wait_s": ceiling}
    try:
        row = gtl.measure(state_dir=state_dir)
        return gtl.render(row), row
    finally:
        S.STATE_DIR, S.landing_cfg = saved, saved_cfg


# ── clause 1: the join, across the live file AND the archives ───────────────


def test_the_join_spans_the_live_ledger_and_the_archives_and_prints_its_percentiles(
        gtl, tmp_path):
    """The item's shape, computed over both files at once.

    Five joined rounds: two whose rows are ONLY in the archive (55 % of the real
    population is archived, so an archive-only round that fell out would halve the
    denominator in silence) and three live. Lags 10, 20, 30, 60 and 3194 s — the
    maximum is the real ledger's max, and the small ones straddle the percentile
    interpolation. The percentiles are graded against `numpy.percentile`'s default
    linear method because that is the definition another reader reaches for; an
    off-by-one interpolation is a trend nobody can reproduce. `over_ceiling` is 0
    at the shipped 4500 s for the same reason the real ledger reports 0: the max
    lag sits under the ceiling, and the count that catches a big ceiling is the
    node below.
    """
    archived = [
        _gate("SM_A", 1_000.0), _promoted("SM_A", 1_010.0),            # lag 10 s
        _gate("SM_B", 2_000.0), _promoted("SM_B", 2_020.0),            # lag 20 s
    ]
    live = [
        _gate("SM_C", 3_000.0), _promoted("SM_C", 3_030.0),            # lag 30 s
        _gate("SM_D", 4_000.0), _promoted("SM_D", 4_060.0),            # lag 60 s
        _gate("SM_E", 5_000.0), _promoted("SM_E", 5_000.0 + 3_194.0),  # lag 3194 s
    ]
    _store(tmp_path / "state", live_rows=live, archive_rows=archived)

    text, row = _report(gtl, tmp_path / "state")

    lags = [10.0, 20.0, 30.0, 60.0, 3194.0]
    # Every non-empty line of both files, not the subset the join kept: a reader that
    # reported the rows it used as the rows it read would understate the corpus by
    # whatever the ledger holds besides `gate` and `promoted` — 47,767 lines on this
    # box, of which only 20,100 are either.
    assert row["ledger_lines"] == 10, row
    assert row["n"] == 5, row
    assert row["p50"] == pytest.approx(float(numpy.percentile(lags, 50))), row
    assert row["p90"] == pytest.approx(float(numpy.percentile(lags, 90))), row
    assert row["p99"] == pytest.approx(float(numpy.percentile(lags, 99))), row
    assert row["max"] == pytest.approx(3194.0), row
    assert row["over_bucket"] == 1 and row["bucket_s"] == 600, row
    assert "over 600 s: 1 (n=5)" in text, text
    assert row["over_ceiling"] == 0 and row["ceiling_s"] == 4_500, row
    # Archive-only rounds are IN the population. Drop them and n falls to 3 while
    # the max stays 3194 — n is the only thing that proves they were read.
    assert row["rounds_only_in_archives"] == 2, row
    for label in ("p50", "p90", "p99", "max"):
        assert label in text, text


def test_the_lag_uses_the_newest_passing_gate_row_and_ignores_failed_rungs(gtl, tmp_path):
    """`ok is True`, and the NEWEST such row — not the first, not any rung.

    The round fails a rung at t=100, passes at t=200 and again at t=300, and lands
    at t=400, so the lag is 100 s. Joining the first passing row prints 200 s;
    joining the failed rung prints 300 s. Both inflate the tail, which is the half
    of the distribution anyone acts on.
    """
    _store(tmp_path / "state", live_rows=[
        _gate("SM_X", 100.0, ok=False, rung="tests"),
        _gate("SM_X", 200.0, ok=True, rung="tests"),
        _gate("SM_X", 300.0, ok=True, rung="review"),
        _promoted("SM_X", 400.0)])

    _, row = _report(gtl, tmp_path / "state")

    assert row["n"] == 1 and row["max"] == pytest.approx(100.0), row


def test_the_ceiling_is_read_from_the_shipped_config_not_literal(gtl, tmp_path):
    """`idle_hard_max_wait_s` is config, so the count has to follow it.

    One round, 500 s of lag, ceiling injected at 400 and then at 4500. A script
    holding 4500 as a constant prints `over_ceiling 0` both times, and the day the
    ceiling moves the count silently stops meaning anything. The printed line also
    names the ceiling it graded against, so a reader can tell a large ceiling from
    a small population.
    """
    _store(tmp_path / "state", live_rows=[_gate("SM_C1", 1_000.0),
                                        _promoted("SM_C1", 1_500.0)])

    text_small, row_small = _report(gtl, tmp_path / "state", ceiling=400)
    text_big, row_big = _report(gtl, tmp_path / "state", ceiling=4_500)

    assert row_small["over_ceiling"] == 1 and row_small["ceiling_s"] == 400, row_small
    assert row_big["over_ceiling"] == 0 and row_big["ceiling_s"] == 4_500, row_big
    assert "400" in text_small and "4500" in text_big, (text_small, text_big)


# ── clause 2: denominators, and nothing dropped in silence ─────────────────


def test_every_percentile_carries_its_n_and_the_dropped_join_is_counted_out_loud(
        gtl, tmp_path):
    """`n=3` on every percentile line, and the orphan `promoted` row reported.

    Three joined rounds plus two rows that cannot join: a `promoted` with no
    passing `gate` row (the ladder writes one row per rung, so a killed run can
    leave none with `ok: true` — a real shape, and 0 of them in the ledger today,
    which is exactly why the counter has to exist before the first one appears),
    and a passing `gate` whose round never landed. Neither belongs in the lag
    population; both must be visible. Printing `n=3` while hiding that two more
    rounds were examined is how a thin join reads as a clean one.
    """
    _store(tmp_path / "state", live_rows=[
        _gate("SM_J1", 1_000.0), _promoted("SM_J1", 1_010.0),
        _gate("SM_J2", 2_000.0), _promoted("SM_J2", 2_020.0),
        _gate("SM_J3", 3_000.0), _promoted("SM_J3", 3_040.0),
        _promoted("SM_ORPHAN", 9_000.0),                     # no gate row at all
        _gate("SM_NEVER", 8_000.0, ok=True, rung="review")])  # never promoted

    text, row = _report(gtl, tmp_path / "state")

    assert row["n"] == 3, row
    assert row["promoted_without_passing_gate"] == 1, row
    assert row["gate_without_promoted"] == 1, row
    for line in text.splitlines():
        if any(k in line for k in ("p50", "p90", "p99", "max")):
            assert "n=3" in line, f"percentile printed without its denominator: {line}"
    assert "promoted with no passing gate row: 1" in text, text
    assert "SM_ORPHAN" in text, (
        "the dropped round is not named, so the count cannot be chased down: " + text)


def test_a_line_that_is_not_json_is_counted_and_does_not_end_the_report(gtl, tmp_path):
    """A truncated final line is normal in an appended JSONL, and skipping it in
    silence is the unreadable-rows lie the voice-trend report refuses to tell.
    """
    _store(tmp_path / "state",
           live_rows=[_gate("SM_K", 1_000.0), _promoted("SM_K", 1_050.0)],
           raw_lines=['{"ts": 1, "event": "gate", "round_i'])

    text, row = _report(gtl, tmp_path / "state")

    assert row["n"] == 1, row
    assert row["unreadable_rows"] == 1, row
    assert "unreadable" in text, text


def test_a_negative_lag_is_reported_rather_than_clamped(gtl, tmp_path):
    """The real ledger has one: `SM_20260908_001027` is stamped promoted 2.5 s
    before its newest passing `gate` row, because the promoter writes its row
    before the ladder's last row lands. Clamping it to 0 would invent a lag that
    never happened; dropping it would shrink `n` in silence.
    """
    _store(tmp_path / "state", live_rows=[_gate("SM_NEG", 1_000.0),
                                         _promoted("SM_NEG", 997.5)])

    text, row = _report(gtl, tmp_path / "state")

    assert row["n"] == 1 and row["negative"] == 1, row
    assert row["max"] == pytest.approx(-2.5), row
    assert "negative" in text, text


# ── clause 3: the moved-tree count, and the catch that leaves no row ───────


def test_the_moved_tree_line_counts_preflight_catches_and_says_the_other_is_unknown(
        gtl, tmp_path):
    """`live HEAD moved` is countable because `gate.py` writes it into the ledger;
    the promote-time identity refuse is not, because it raises before the `try:`
    that appends an event. Both facts on ONE line, because the count without the
    absence is precisely the `0` a reader would misread as "the check never fires".
    """
    _store(tmp_path / "state", live_rows=[
        _gate("SM_P", 1_000.0), _promoted("SM_P", 1_020.0),
        _moved("SM_M1", 500.0, base="aaaa0001"),
        _moved("SM_M2", 600.0, base="aaaa0002"),
        {"ts": 700.0, "event": "land_failed", "round_id": "SM_L", "rung": "land",
         "ok": False, "detail": "backend never went idle"}])

    text, row = _report(gtl, tmp_path / "state")

    assert row["live_head_moved_rows"] == 2, row
    lines = [ln for ln in text.splitlines() if "live HEAD moved" in ln]
    assert len(lines) == 1, text
    low = lines[0].lower()
    assert "2" in lines[0], lines[0]
    assert "unknown" in low and "not zero" in low, (
        "the promote-time absence must sit on the same line as the count: " + lines[0])
    assert "promote" in low, lines[0]


def test_a_moved_catch_stored_only_in_the_archive_is_counted_too(gtl, tmp_path):
    """The catch count is a property of the corpus, not of whichever file happens
    still to be uncompressed: this row lives in the September archive only.
    """
    _store(tmp_path / "state",
           live_rows=[_gate("SM_P", 1_000.0), _promoted("SM_P", 1_020.0)],
           archive_rows=[_moved("SM_OLD", 400.0)])

    _, row = _report(gtl, tmp_path / "state")

    assert row["live_head_moved_rows"] == 1, row


# ── clause 4: --record, one row per call, never the ledger ─────────────────


def test_record_appends_exactly_one_row_per_call_and_never_touches_the_ledger(
        gtl, tmp_path, capsys):
    """Two calls, two rows, and the ledger byte-identical.

    The report's whole safety claim is that apart from its own record file it is
    read-only: the file that appends to `promotions.jsonl` is the audit trail of
    what code runs in production, and a report writing there would be forging
    history. Hence the byte comparison rather than a row count — a rewrite that
    shuffled bytes while preserving the line count still fails here.
    """
    state = tmp_path / "state"
    _store(state, live_rows=[_gate("SM_R", 1_000.0), _promoted("SM_R", 1_020.0)])
    before = (state / "promotions.jsonl").read_bytes()

    from scripts.automod import state as S
    saved, saved_cfg = S.STATE_DIR, S.landing_cfg
    S.STATE_DIR = state
    S.landing_cfg = lambda repo=None: {"idle_hard_max_wait_s": 4_500}
    try:
        assert gtl.main(["--state-dir", str(state), "--record"]) == 0
        assert gtl.main(["--state-dir", str(state), "--record"]) == 0
    finally:
        S.STATE_DIR, S.landing_cfg = saved, saved_cfg

    out = capsys.readouterr().out
    rows = [json.loads(ln) for ln in
            (state / "gate_to_land_lag.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2, rows
    assert [r["n"] for r in rows] == [1, 1], rows
    assert "recorded" in out, out
    assert (state / "promotions.jsonl").read_bytes() == before, (
        "the report wrote to the promotions ledger — it is read-only by contract")


def test_the_record_path_is_the_state_dir_the_env_hook_names(tmp_path):
    """`LLOYD_AUTOMOD_STATE` is how `scripts/automod/state.py:44` resolves its
    directory, so it is how the record file has to be placed: a hard-coded
    `~/.local/state/...` would write into the live automod state from a test and
    ignore the one hook the module offers. Run as a subprocess because the
    variable is read at import time, long before a test could set it.
    """
    state = tmp_path / "state"
    _store(state, live_rows=[_gate("SM_ENV", 1_000.0), _promoted("SM_ENV", 1_030.0)])
    ledger_before = (state / "promotions.jsonl").read_bytes()
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--record"],
        capture_output=True, text=True, timeout=180, check=True, cwd=str(ROOT),
        env=dict(os.environ, LLOYD_AUTOMOD_STATE=str(state)))

    written = (state / "gate_to_land_lag.jsonl").read_text(encoding="utf-8")
    assert len(written.strip().splitlines()) == 1, written
    assert str(state) in proc.stdout, proc.stdout
    assert (state / "promotions.jsonl").read_bytes() == ledger_before


def test_the_default_run_prints_and_writes_nothing(gtl, tmp_path, capsys):
    """No `--record`, no file: the daily job opts in, and a report that always
    appended would be a second unbounded store nobody asked for — the next
    retention-sweep finding, self-inflicted.
    """
    state = tmp_path / "state"
    _store(state, live_rows=[_gate("SM_D", 1_000.0), _promoted("SM_D", 1_020.0)])
    from scripts.automod import state as S
    saved, saved_cfg = S.STATE_DIR, S.landing_cfg
    S.STATE_DIR = state
    S.landing_cfg = lambda repo=None: {"idle_hard_max_wait_s": 4_500}
    try:
        assert gtl.main(["--state-dir", str(state)]) == 0
    finally:
        S.STATE_DIR, S.landing_cfg = saved, saved_cfg

    assert not (state / "gate_to_land_lag.jsonl").exists(), (
        "the report wrote a record file that was never asked for")
    assert "p50" in capsys.readouterr().out


def test_a_store_with_no_joined_rounds_reports_n_zero_without_crashing(gtl, tmp_path):
    """An empty population is an answer, not a crash and not a p50 of 0 s: a
    percentile over nothing is a number a reader would act on.
    """
    _store(tmp_path / "state", live_rows=[_gate("SM_ONLY", 1_000.0)])

    text, row = _report(gtl, tmp_path / "state")

    assert row["n"] == 0, row
    assert row["p50"] is None and row["max"] is None, row
    assert "n=0" in text, text
