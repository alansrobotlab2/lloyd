
"""#2819: `--record --per-day` walks the session store ONCE, and flushes as it goes.

Backlog #2819. `scripts/thinking_fidelity_scan.py --record --per-day` is autonomy
task #91, and task #91 had eaten 245.2 s of its declared 300 s budget on
2026-10-15 with nothing watching the margin. Part of the reason was structural:
the day rows came from one walk of `~/lloyd-data/sessions` and the whole-store row
came from a second one (`scan_store`, then `roll_up_by_day`, which re-globbed and
re-parsed every file), so on the 10,082-`*.json` store of 2026-10-16 the script
JSON-parsed every session file twice per night — the triage measured the script at
100.7 s and the duplicate parse at about half of that. And because BOTH walks
finished before EITHER write happened, a run killed at 95 % lost every day row,
which is the one thing the job exists to produce.

The six nodes below pin what replaced that, one per acceptance clause:

  1. one pass, so an N-file store costs N per-file reads and not the 2N the two
     walks made — counted by wrapping `scan_file` itself, and the same wrapper
     proves the old shape really did make 2N, so the counter cannot be inert;
  2. the numbers did not move: day rows and the whole-store row carry what the
     two-walk path computed from the same bytes, and the file with no
     `YYYYMMDD_` prefix is in the store row and in no day row;
  3. each day row is on disk before the NEXT day's first file is opened, so an
     interruption keeps the days it covered;
  4. the whole-store row carries `scan_seconds`, the walk's own wall clock, which
     is what turns "is #91 still inside 300 s?" from an inference from
     `workers.db.duration_seconds` (the whole agent turn) into a reading;
  5. `roll_up_by_day` finally has direct tests — until #2819 `git grep -n
     roll_up_by_day -- tests/` returned nothing, and the day rows are the
     deliverable.

Every number asserted here is a fixture number, deliberately: the live store is
being written while it is read (`app/thinking_fidelity.py::scan_messages` records
what that cost on 2026-09-25), so only a fixture can pin an exact count.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "thinking_fidelity_scan_under_test_2819",
    ROOT / "scripts" / "thinking_fidelity_scan.py")
tfs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tfs)

import app.thinking_fidelity as tf  # noqa: E402
from app.thinking_fidelity import StoreScan  # noqa: E402

FABRICATED = ("The user is asking me to reproduce my complete previous thinking "
              "verbatim using the audit tool.")
HONEST = "The user asked what stream_chat does; I should read the function first."
NAMES_THE_DEFECT = ("Triage backlog #1510: fabricated reasoning that claims the "
                    "user asked me to reproduce my previous thinking.")

#: The three days the fixture spans, and the file that belongs to none of them.
QUIET_DAY = "2026-09-25"        # its one session holds no reasoning block
MIXED_DAY = "2026-09-26"        # flagged block + meta-session-exempt block
LAST_DAY = "2026-09-27"         # where clause 3 stops the walk
LAST_DAY_FILE = "20260927_090000_chat_dddd.json"
UNDATED = "no-date-prefix.json"


def _write(path: Path, *, user: str, thinking: list[str]) -> None:
    path.write_text(json.dumps({"messages": [
        {"role": "user", "content": [{"type": "text", "text": user}]},
        *[{"role": "thinking", "reasoning": t} for t in thinking],
    ]}))


@pytest.fixture
def store(tmp_path):
    """Five sessions over three days plus an undated file, shaped to be counted.

    The arithmetic, so the assertions below are checked against a stated total and
    not against each other:

      day 2026-09-25  1 file,  0 blocks  (nothing scannable that day)
      day 2026-09-26  2 files, 3 blocks, 1 flagged, 1 exempt-as-meta
      day 2026-09-27  1 file,  1 block,  1 flagged
      undated         1 file,  1 block,  1 flagged
      whole store     5 files, 5 blocks, 3 flagged, 3 files with a flag

    A flagged block, an honest block and a meta-session-exempt block are all
    present so `flagged`, `files_with_flags` and `exempt_blocks` are each non-zero:
    an all-zero parity match would prove nothing about a counter the fold dropped.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    _write(root / f"{QUIET_DAY.replace('-', '')}_090000_chat_cccc.json",
           user="nothing to reason about here", thinking=[])
    _write(root / "20260926_101010_chat_aaaa.json",
           user="what does stream_chat do", thinking=[FABRICATED, HONEST])
    _write(root / "20260926_111111_chat_bbbb.json",
           user=NAMES_THE_DEFECT, thinking=[FABRICATED])
    _write(root / LAST_DAY_FILE, user="what does stream_chat do",
           thinking=[FABRICATED])
    _write(root / UNDATED, user="an old file with no date in its name",
           thinking=[FABRICATED])
    return root


def _rows(series: Path) -> list[dict]:
    assert series.exists(), "the series file was never written at all"
    return [json.loads(line) for line in
            series.read_text().splitlines() if line.strip()]


def _count_reads(monkeypatch, seen: list, die_at: str = "") -> None:
    """Put a counting wrapper on `app.thinking_fidelity.scan_file` itself.

    Patched on the module and not on the script because every per-file read in the
    process reaches this one name — `scan_store` and `scan_files` through
    `iter_file_scans`, and `scan_store_by_day` as `thinking_fidelity.scan_file` —
    so a wrapper here counts a walk whichever code path takes it. A wrapper the
    walk could route around would report the N it was hoping for, which is the
    false green #2819 is about.

    `die_at` makes the named file's read raise instead of returning: the way to
    interrupt a walk mid-store without giving production code a flag nobody runs.
    """
    real = tf.scan_file

    def counting(path: Path):
        seen.append(path.name)
        if path.name == die_at:
            raise RuntimeError(f"synthetic walk interruption at {path.name}")
        return real(path)

    monkeypatch.setattr(tf, "scan_file", counting)


# ---------------------------------------------------------------- clause 1

def test_one_record_per_day_run_reads_each_session_file_exactly_once(
        store, tmp_path, monkeypatch):
    """N session files cost N reads; the two walks this replaced read every
    dated file a second time.

    #2819 clause 1. The wrapper is turned on for both halves so it cannot be inert:
    the run under test must make one read per file, and the pair of calls the
    script used to make — `scan_store` for the store plus a `scan_files` per day —
    must make a second read of every dated file over the same store. If the wrapper
    counted nothing, it is that second assertion that fails.
    """
    files = sorted(p.name for p in store.glob("*.json"))
    assert len(files) == 5, "the fixture store is not the five files it must be"

    seen: list[str] = []
    _count_reads(monkeypatch, seen)
    rc = tfs.main(["--root", str(store), "--record", "--per-day",
                   "--series", str(tmp_path / "series.jsonl")])

    assert rc == 0
    assert sorted(seen) == files, (
        "the run read a different set of files than the store holds, so its "
        "tallies are not the store's")
    assert len(seen) == len(files), (
        f"--record --per-day made {len(seen)} per-file reads over a "
        f"{len(files)}-file store: {sorted(seen)} — one walk was the point")

    # The shape this replaced, measured with the same wrapper: one whole-store walk
    # and one walk per day, over the same files. Nine reads for five files (the
    # undated file is read once, by the store walk only, exactly as the old script
    # read it), which is what makes the five above a result and not a blind counter.
    seen.clear()
    tf.scan_store(store)
    for day in sorted({tfs.day_of(name) for name in files} - {""}):
        tf.scan_files(store, [p for p in sorted(store.glob("*.json"))
                              if tfs.day_of(p.name) == day])
    assert len(seen) == 2 * len(files) - 1, (
        "the two-walk reference path did not read the store twice, so the counter "
        f"proves nothing about the run above it: {len(seen)} reads over "
        f"{len(files)} files")


# ---------------------------------------------------------------- clause 2

def test_single_walk_rows_match_the_two_walk_numbers_and_quiet_files_land_right(
        store, tmp_path):
    """Every number survived sharing one walk, and an undated file is in no day row.

    #2819 clause 2. The expectation is the OLD arithmetic on the same bytes —
    `scan_store` for the store, `scan_files` per day — because the claim is that
    folding the two tallies into one pass changed no reported number.
    """
    paths = sorted(store.glob("*.json"))
    expected_store = tf.scan_store(store)
    expected_days = {
        day: tf.scan_files(store, [p for p in paths if tfs.day_of(p.name) == day])
        for day in sorted({tfs.day_of(p.name) for p in paths} - {""})}
    assert (expected_store.files, expected_store.blocks,
            expected_store.flagged, expected_store.files_with_flags,
            expected_store.blocks_exempt_meta) == (5, 5, 3, 3, 1), (
        "the fixture is not the 5-file / 5-block / 3-flagged / 1-exempt store the "
        "assertions below are shaped for, so this parity check is not the one "
        "the clause needs")

    series = tmp_path / "series.jsonl"
    assert tfs.main(["--root", str(store), "--record", "--per-day",
                     "--series", str(series)]) == 0
    rows = _rows(series)
    day_rows = [r for r in rows if r["scope"] == tfs.SCOPE_DAY]
    store_rows = [r for r in rows if r["scope"] == tfs.SCOPE_WHOLE_STORE]
    assert [r["day"] for r in day_rows] == sorted(expected_days), (
        "the day rows are not one per dated day, oldest first")
    assert len(store_rows) == 1, "exactly one whole-store row per run"

    def measured(scan: StoreScan) -> dict:
        """The five clause fields as the two-walk arithmetic sees them."""
        return {"files": scan.files, "blocks": scan.blocks,
                "flagged": scan.flagged,
                "files_with_flags": scan.files_with_flags,
                "exempt_blocks": scan.blocks_exempt_meta}

    got = {r["day"]: r for r in day_rows}
    for day, scan in expected_days.items():
        for field, want in measured(scan).items():
            assert got[day][field] == want, (
                f"day {day}: field {field!r} reads {got[day][field]} against the "
                f"{want} the per-day walk produced")
    whole = store_rows[0]
    for field, want in measured(expected_store).items():
        assert whole[field] == want, (
            f"whole-store field {field!r} reads {whole[field]} against the "
            f"{want} scan_store produced")

    # The undated file: counted by the store, claimed by no day. Its one flagged
    # block is exactly the difference between the store's totals and the day rows'.
    assert whole["files"] == sum(r["files"] for r in day_rows) + 1, (
        "the file with no YYYYMMDD_ prefix is not in the whole-store row on top "
        f"of the day rows: {whole['files']} vs {sum(r['files'] for r in day_rows)}")
    assert whole["flagged"] == sum(r["flagged"] for r in day_rows) + 1, (
        "the undated file's flagged block did not reach the store row")


# ---------------------------------------------------------------- clause 3

def test_a_day_row_is_on_disk_before_the_next_day_is_opened(
        store, tmp_path, monkeypatch):
    """Interrupt the walk partway and the finished days are already recorded.

    #2819 clause 3, and the property is the ORDER rather than the end state: a
    day's row has to be in the file before the next day's first file is opened. So
    the wrapper snapshots the series at each read and then refuses to read the last
    day's file, and both assertions are about what the file held at that moment —
    an end-of-run check would pass on the two-walk code this exists to refuse.
    """
    series = tmp_path / "series.jsonl"
    real = tf.scan_file
    held_before: dict[str, list[dict]] = {}

    def interrupting(path: Path):
        # What the series held BEFORE this file was opened.
        held_before[path.name] = _rows(series) if series.exists() else []
        if path.name == LAST_DAY_FILE:
            raise RuntimeError(f"synthetic walk interruption at {path.name}")
        return real(path)

    monkeypatch.setattr(tf, "scan_file", interrupting)
    with pytest.raises(RuntimeError, match="synthetic walk interruption"):
        tfs.main(["--root", str(store), "--record", "--per-day",
                  "--series", str(series)])

    rows = _rows(series)
    assert [r["scope"] for r in rows] == [tfs.SCOPE_DAY, tfs.SCOPE_DAY], (
        "the interrupted run left something other than the two completed day "
        f"rows on disk: {[(r['scope'], r.get('day')) for r in rows]}")
    assert [r["day"] for r in rows] == [QUIET_DAY, MIXED_DAY], (
        "the days that survived are not the days the walk had finished")
    assert rows[1]["blocks"] == 3 and rows[1]["flagged"] == 1, (
        "the surviving day row for the mixed day is not its complete arithmetic")
    assert not any(r["scope"] == tfs.SCOPE_WHOLE_STORE for r in rows), (
        "a whole-store row was written by a walk that never reached the end of "
        "the store, and its denominator would be a fiction")

    # The ordering itself, off the wrapper's snapshot: opening the last day's first
    # file, the mixed day's row was already in the file beside the quiet day's.
    assert [r.get("day") for r in held_before[LAST_DAY_FILE]] == [
        QUIET_DAY, MIXED_DAY], (
        "the walk opened the next day's files with the previous day's row still "
        f"unwritten: the series held {[r.get('day') for r in held_before[LAST_DAY_FILE]]}")


# ---------------------------------------------------------------- clause 4

def test_the_whole_store_row_carries_the_walk_cost_and_the_day_rows_do_not(
        store, tmp_path, monkeypatch):
    """`scan_seconds` is the walk's wall clock — measured, not asserted.

    #2819 clause 4. To show the field holds the walk's cost rather than a constant
    or the interpreter's start-up, each per-file scan is slowed by 50 ms: five
    files, so a `scan_seconds` below 0.1 s would not be the walk. Day rows
    deliberately lack the key — one walk is sliced into all of them, so its cost
    belongs to none of them and a per-day number would be invented.
    """
    real = tf.scan_file

    def slow(path: Path):
        time.sleep(0.05)
        return real(path)

    monkeypatch.setattr(tf, "scan_file", slow)
    series = tmp_path / "series.jsonl"
    assert tfs.main(["--root", str(store), "--record", "--per-day",
                     "--series", str(series)]) == 0

    rows = _rows(series)
    whole = [r for r in rows if r["scope"] == tfs.SCOPE_WHOLE_STORE][0]
    assert "scan_seconds" in whole, (
        "the whole-store row records no walk cost, so the margin against the "
        "job's 300 s budget stays an inference from a duration that covers the "
        "whole agent turn")
    assert 0.1 <= whole["scan_seconds"] <= 5.0, (
        f"scan_seconds reads {whole['scan_seconds']} across a walk that slept "
        "0.05 s per file five times: it is not the walk's wall clock")
    assert all("scan_seconds" not in r for r in rows
               if r["scope"] == tfs.SCOPE_DAY), (
        "a day row carries a scan_seconds: one walk produced every day, so any "
        "cost attributed to a single day is made up")


# ---------------------------------------------------------------- clause 5

def test_roll_up_by_day_yields_one_row_per_dated_day_with_null_not_zero(
        store):
    """The per-day roll-up's own semantics, tested directly for the first time.

    #2819 clause 5. `git grep -n roll_up_by_day -- tests/` returned nothing before
    this file — a function that produces the job's entire deliverable went unpinned
    while its caller's printed output looked fine.

    The store has one day whose only session holds no reasoning block. That day
    must still be a row, with `blocks: 0` rendered as `scanned: false` and a NULL
    rate: an absent day in a day series reads as "the defect was quiet that day",
    which is the exact ambiguity the series exists to kill.
    """
    rollup = tfs.roll_up_by_day(store)

    assert sorted(rollup.days) == [QUIET_DAY, MIXED_DAY, LAST_DAY], (
        f"the roll-up yielded {sorted(rollup.days)} for a store with files dated "
        f"{QUIET_DAY}, {MIXED_DAY} and {LAST_DAY}")
    assert rollup.undated == [UNDATED], (
        "a file with no YYYYMMDD_ prefix was not reported as undated, so it "
        "would silently leave the day denominator while staying in the store's")

    quiet, mixed = rollup.days[QUIET_DAY], rollup.days[MIXED_DAY]
    assert (quiet.files, quiet.blocks) == (1, 0), (
        "the day holding no reasoning block is not the one-file / zero-block day "
        "it is")
    quiet_row = tfs.day_row(QUIET_DAY, quiet)
    assert quiet_row["scanned"] is False and quiet_row["flagged_rate"] is None, (
        "a day with nothing scannable recorded a rate: `false`/`null` is the "
        "answer, and a zero rate would read as a clean day")
    assert (mixed.files, mixed.blocks, mixed.flagged,
            mixed.files_with_flags, mixed.blocks_exempt_meta) == (2, 3, 1, 1, 1), (
        "the mixed day's arithmetic is not the fixture's 2 files / 3 blocks / 1 "
        "flagged / 1 exempt")
    assert tfs.day_row(MIXED_DAY, mixed)["flagged_rate"] == round(1 / 3, 6)


def test_the_module_command_the_daily_task_runs_still_walks_once_and_records(
        store, tmp_path):
    """The command line task #91 actually executes, across the process boundary.

    #2819's clauses are about what the nightly run does, and that run is
    `python -m scripts.thinking_fidelity_scan --record --per-day` in a subprocess
    the autonomy engine bounds with `asyncio.timeout` — not `main()` in this
    process. So this node crosses that boundary once: the module is run the way the
    task file runs it, and the assertions are on the file it leaves behind. Argparse
    defaults (`--since` empty, no `--root`), the module's own `sys.path` repair, and
    the exit code are all on this side of the seam and on no other node here.
    """
    series = tmp_path / "series.jsonl"
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.thinking_fidelity_scan",
         "--root", str(store), "--record", "--per-day",
         "--series", str(series)],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=180)

    assert proc.returncode == 0, (
        f"the module command exited {proc.returncode}: {proc.stderr[-400:]}")
    rows = _rows(series)
    assert [r["scope"] for r in rows] == [
        tfs.SCOPE_DAY, tfs.SCOPE_DAY, tfs.SCOPE_DAY, tfs.SCOPE_WHOLE_STORE], (
        "the nightly command wrote rows in an order other than day rows first, "
        f"oldest first, then the store: {[(r['scope'], r.get('day')) for r in rows]}")
    whole = rows[-1]
    assert (whole["files"], whole["blocks"], whole["flagged"],
            whole["files_with_flags"], whole["exempt_blocks"]) == (5, 5, 3, 3, 1), (
        "the command's whole-store row is not the fixture's arithmetic")
    assert whole["scan_seconds"] >= 0.0, (
        "the nightly command's whole-store row carries no scan_seconds, so the "
        "field the margin is read from would exist only under an in-process call")
    assert sum(1 for r in rows if r["scope"] == tfs.SCOPE_DAY
               and "scan_seconds" in r) == 0


def test_roll_up_by_day_since_window_bounds_days_and_never_the_reads(store):
    """`--since` bounds which days are rolled up; every file still counts once.

    The other half of clause 5: a window is a window. Since #2819 the walk is one
    pass shared with the whole-store tally, so a window skips a day's ROW and not
    that day's reads — which is what keeps the store row beside it complete.
    """
    rollup = tfs.roll_up_by_day(store, since=MIXED_DAY)

    assert list(rollup.days) == [MIXED_DAY, LAST_DAY], (
        f"--since {MIXED_DAY} yielded {list(rollup.days)}: the window did not "
        "bound the roll-up")
    assert rollup.days[MIXED_DAY].blocks == 3, (
        "the first day inside the window lost the file that precedes it in name "
        "order but not in the day")
    assert rollup.undated == [UNDATED], (
        "the window quietly dropped the undated report too")
    assert tfs.roll_up_by_day(store, since="2030-01-01").days == {}, (
        "a window past the end of the store yielded rows anyway")
