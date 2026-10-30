
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

import gzip
import importlib.util
import json
import os
import shutil
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


# ---- #3590: one walk, both stored forms -----------------------------------------

def _archive_in_place(path: Path) -> Path:
    """Gzip `path` and unlink it, the way `sweep_sessions` does.

    The same three statements as `scripts/groundskeeper/retention-sweep.py::
    sweep_sessions`: write `<name>.json.gz` from the original's bytes, verify it
    round-trips as JSON, then `unlink()` the original. Clause 3 is a claim about
    what one night of the real producer does to a day row, so the fixture has to
    be produced by that producer's own steps — the gzip, the round-trip guard and
    the deletion of the plain file all matter, and leaving the plain file behind
    would be a state the sweep never writes.
    """
    gz_path = path.with_suffix(".json.gz")
    with path.open("rb") as src, gzip.open(gz_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    with gzip.open(gz_path, "rt", encoding="utf-8") as fh:
        json.load(fh)
    path.unlink()
    return gz_path


@pytest.fixture
def mixed_store(tmp_path):
    """Four sessions over two days, in both stored forms, with one overlap.

      20261005_090000_chat_aaaa   BOTH forms. The live file holds 1 block, its
                                  archive 2 — the state a sweep interrupted
                                  between `gzip.open` and `unlink` leaves, and
                                  the only shape here that tells "read both"
                                  apart from "read the live one".
      20261005_100000_chat_bbbb   ARCHIVED ONLY: 2 blocks, 1 flagged. Two of its
                                  day's three blocks are invisible to a `*.json`
                                  walk — which is what #3590 measured on the live
                                  store as 4 files read out of 729.
      20261006_090000_chat_cccc   plain: 1 block, 0 flagged.
      no-date-prefix              ARCHIVED ONLY: 1 block, 1 flagged. In the
                                  whole-store row and in no day row.

      whole store   4 files, 5 blocks, 2 flagged, 2 of them archived
      2026-10-05    2 files, 3 blocks, 1 flagged, 1 archived
      2026-10-06    1 file,  1 block, 0 flagged, 0 archived

    The archived overlap is deliberately NOT in `files_archived`: that field
    counts the form that was READ, and for that session the live file won.
    """
    root = tmp_path / "sessions"
    root.mkdir()
    live = root / "20261005_090000_chat_aaaa.json"
    _write(live, user="what does stream_chat do", thinking=[HONEST])
    (root / "20261005_090000_chat_aaaa.json.gz").write_bytes(gzip.compress(
        json.dumps({"messages": [
            {"role": "user", "content": "what does stream_chat do"},
            {"role": "thinking", "reasoning": HONEST},
            {"role": "thinking", "reasoning": HONEST},
        ]}).encode()))
    _archive_in_place(_write_return(
        root / "20261005_100000_chat_bbbb.json", user="and the router?",
        thinking=[FABRICATED, HONEST]))
    _write(root / "20261006_090000_chat_cccc.json", user="nothing to add",
           thinking=[HONEST])
    _archive_in_place(_write_return(
        root / "no-date-prefix.json",
        user="an old file with no date in its name", thinking=[FABRICATED]))
    return root


def _write_return(path: Path, *, user: str, thinking: list[str]) -> Path:
    """`_write`, but handing the path back so it can be archived in place."""
    _write(path, user=user, thinking=thinking)
    return path


# ---------------------------------------------------------------- clause 2

def test_the_single_walk_enumerates_plain_and_archived_sessions_exactly_once(
        mixed_store, tmp_path, monkeypatch):
    """Both suffixes are in the one enumeration, and each session is in it once.

    #3590 clause 2. Three things have to hold together, and the fixture is shaped
    so each one fails alone: `*.json` alone sees 2 of the 4 sessions (the last
    assertion, and the reason the enumeration wrapper is not inert), reading both
    forms naively makes 5 files and 7 blocks for a store whose live arithmetic is
    4 and 5 (the `files`/`blocks` assertions, and why the overlap's two copies
    hold DIFFERENT block counts), and `scan_store` has to enumerate through the
    same function rather than glob its own way again — the drift of two
    independent `*.json` globs is exactly how #3590 happened.
    """
    expected = ["20261005_090000_chat_aaaa.json",
                "20261005_100000_chat_bbbb.json.gz",
                "20261006_090000_chat_cccc.json",
                "no-date-prefix.json.gz"]
    paths = tf.session_paths(mixed_store)
    assert [p.name for p in paths] == expected, (
        "the walk is not one path per session with the live form winning an "
        f"overlap, sorted by session id: {[p.name for p in paths]}")
    ids = [tf.session_id(p.name) for p in paths]
    assert len(ids) == len(set(ids)) == 4, (
        f"a session id is enumerated more than once: {ids}")

    enumerations: list[list[str]] = []
    real_paths = tf.session_paths

    def counting_paths(root):
        got = real_paths(root)
        enumerations.append([p.name for p in got])
        return got

    monkeypatch.setattr(tf, "session_paths", counting_paths)
    seen: list[str] = []
    _count_reads(monkeypatch, seen)

    series = tmp_path / "series.jsonl"
    assert tfs.main(["--root", str(mixed_store), "--record", "--per-day",
                     "--series", str(series)]) == 0
    assert len(enumerations) == 1, (
        f"the per-day walk enumerated the store {len(enumerations)} times; one "
        "walk is the whole shape of #2819 and #3590 clause 2")
    run_reads = sorted(seen)
    assert len(run_reads) == 4 and run_reads == expected, (
        f"the run made {len(run_reads)} per-file reads over a 4-session store: "
        f"{run_reads} — a session in both forms must cost ONE read")

    seen.clear()
    whole = tf.scan_store(mixed_store)
    assert len(enumerations) == 2, (
        "`scan_store` did not enumerate through `session_paths`, so it is globbing "
        "the store its own way again — the two-glob drift #3590 was filed on")
    assert enumerations[0] == enumerations[1], (
        "the per-day walk and scan_store disagree about which files the store "
        f"holds: {enumerations[0]} vs {enumerations[1]}")
    assert sorted(seen) == expected, (
        "`scan_store` reads a different set of files than the per-day walk: "
        f"{sorted(seen)}")

    rows = _rows(series)
    store_row = rows[-1]
    assert (store_row["files"], store_row["blocks"], store_row["flagged"],
            store_row["files_archived"], store_row["unreadable"]) == (
        4, 5, 2, 2, 0), (
        "the whole-store row is not 4 sessions / 5 blocks / 2 flagged with 2 "
        "archived and nothing unreadable: reading both forms of the overlap would "
        "read 7 blocks, reading plain files alone would read 1 file and miss bbbb "
        "entirely, and a non-zero `unreadable` means a fixture archive is not "
        "really compressed")
    day_0505 = [r for r in rows if r.get("day") == "2026-10-05"][0]
    assert (day_0505["files"], day_0505["blocks"], day_0505["flagged"],
            day_0505["files_archived"]) == (2, 3, 1, 1), (
        "the mixed day is not 2 files / 3 blocks with its overlap resolved to the "
        "live 1-block file")
    assert whole.files_archived == 2, (
        "files_archived counted the archived copy that was never read")

    plain_only = sorted(p.name for p in mixed_store.glob("*.json"))
    assert len(plain_only) == 2, (
        "the fixture is not the mixed store the clause needs: a plain-only walk "
        f"would see {len(plain_only)} of its 4 sessions, so this test could pass "
        "on the old glob")


# ---------------------------------------------------------------- clause 3

def test_archiving_a_session_in_place_moves_no_day_rows_denominator(store,
                                                                    tmp_path):
    """The same bytes, renamed by the sweep: the day row must not notice.

    #3590 clause 3, run twice for real rather than argued: a full
    `--record --per-day` over the fixture store, then `sweep_sessions`' own
    gzip-and-unlink on one 2026-09-26 session, then the same command again. Every
    field of every row is compared except `recorded_at` (the run moved) and
    `files_archived` (the one field the sweep is ALLOWED to move, and the number
    that says so). The literals are the fixture's stated arithmetic, so a
    denominator that dropped the archived file fails here even against a
    self-consistent pair of runs.

    The last two assertions are the counter-proof: the plain-only glob the walk
    used to make does lose that file (5 → 4), so the unchanged row is the fix
    working and not a fixture that never had an archive in it.
    """
    before = tmp_path / "before.jsonl"
    after = tmp_path / "after.jsonl"
    assert tfs.main(["--root", str(store), "--record", "--per-day",
                     "--series", str(before)]) == 0
    plain_count_before = len(list(store.glob("*.json")))

    gz = _archive_in_place(store / "20260926_101010_chat_aaaa.json")
    assert gz.name == "20260926_101010_chat_aaaa.json.gz"
    assert not (store / "20260926_101010_chat_aaaa.json").exists(), (
        "the fixture did not archive the way the sweep does: it unlinks the "
        "original, and leaving it is a state production never reaches")
    assert len(list(store.glob("*.json"))) == plain_count_before - 1, (
        "the plain-only walk the script used to make still sees the same count, "
        "so this fixture has not actually moved a session into the archive and "
        "the parity below would prove nothing")

    assert tfs.main(["--root", str(store), "--record", "--per-day",
                     "--series", str(after)]) == 0

    first = {r.get("day", r["scope"]): r for r in _rows(before)}
    second = {r.get("day", r["scope"]): r for r in _rows(after)}
    assert sorted(first) == sorted(second), (
        "archiving one session changed WHICH rows exist, not just their numbers")
    for key, old in first.items():
        for field in sorted(old):
            if field in ("recorded_at", "files_archived"):
                continue
            assert second[key][field] == old[field], (
                f"{key}: field {field!r} moved from {old[field]} to "
                f"{second[key][field]} when a session was archived in place — the "
                "sweep is not allowed to change a day's denominator")

    mixed = second[MIXED_DAY]
    assert (mixed["files"], mixed["blocks"], mixed["flagged"]) == (2, 3, 1), (
        "the mixed day is not the fixture's 2 files / 3 blocks / 1 flagged once "
        "one of its sessions is an archive")
    assert (first[MIXED_DAY]["files_archived"],
            mixed["files_archived"]) == (0, 1), (
        "the witness did not record the archive it was made to record")
    whole_old, whole_new = first[tfs.SCOPE_WHOLE_STORE], second[tfs.SCOPE_WHOLE_STORE]
    assert (whole_old["files"], whole_new["files"], whole_old["blocks"],
            whole_new["blocks"]) == (5, 5, 5, 5), (
        "the whole-store denominator moved when nothing was deleted, which is the "
        "#3590 symptom measured on the live store as 280,864 → 260,014 blocks")
    assert whole_new["files_archived"] == 1, (
        "the whole-store row does not name the archived session in its own "
        "denominator")
