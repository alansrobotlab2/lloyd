"""#1656: the prevalence number is a series now, with its denominator attached.

Backlog #1510's ruling says the guard on fabricated reasoning should be lifted
"once the marker rate falls", and a rate that falls needs a series to say so.
`scripts/thinking_fidelity_scan.py` could always compute it — `flagged 172 of
40964 reasoning blocks` — and then forgot it, printed to a terminal. That is the
same gap #460 records for the IV metrics: the measurement works, the *series*
does not exist, so every question needing a before/after has no after.

`--record` is the persistence half, and this file pins the one property that
makes the series usable rather than merely long: **a run that flagged nothing
and a run that scanned nothing must not produce the same row.** `app/uptake.py`
has the machine's worst history on exactly that failure, and a prevalence series
whose zero is ambiguous is the one shape that would make the ruling's gate
decidable by accident. So every row carries numerator AND denominator, and the
rate is `null` — not `0.0` — when there was nothing to divide by.

#1770 is the second half of that gap. A whole-store row over ~59,000 blocks cannot
answer "is the marker rate falling this week" — history dominates it — so the
series needed PER-DAY rows and a job that writes them unattended. The nodes below
`test_per_day_reports_one_row_per_calendar_day` pin the day roll-up, the `--since`
window, the scope field that stops a day's rate and the store's rate meaning two
different things inside one file, and the dispatchability of the autonomy task that
runs it. The same zero-vs-nothing rule applies one level down: a day with no reasoning
block is a row with `blocks: 0` and `scanned: false`, never an absent day.

#3590 is the third way this series can lie with no error anywhere: the retention
sweep gzips a session and unlinks the original, and the scan enumerated only
`*.json`. Nothing was deleted and the denominator fell anyway — the live store's
whole-store row went 280,864 → 260,014 blocks across the runs of
2026-10-27…10-30 — while a day whose sessions had mostly been archived reported
`scanned: true, flagged: 0`, which reads as a clean day and is a day that was
never measured. The three nodes at the bottom of this file pin the row-level
half of the fix: an archive scores exactly like the plain file it came from
(`test_an_archived_session_scores_the_same_blocks_and_flags_as_the_plain_file`),
every row and both printed lines name `files_archived`, so a denominator now
carries a witness the sweep cannot move in silence
(`test_every_row_and_both_printed_lines_name_the_archived_share`), and an archive
that cannot be decoded is counted `unreadable` rather than dropped
(`test_an_undecodable_archive_is_counted_unreadable_and_never_dropped`). The
walk-level half — one enumeration of both forms, and archiving a session moving
no denominator — is pinned in `tests/test_thinking_fidelity_scan_single_walk.py`.
"""

from __future__ import annotations

import gzip
import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "thinking_fidelity_scan_under_test",
    ROOT / "scripts" / "thinking_fidelity_scan.py")
tfs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tfs)

from app import thinking_fidelity as tf  # noqa: E402
from app.thinking_fidelity import StoreScan  # noqa: E402

FABRICATED = ("The user is asking me to reproduce my complete previous thinking "
              "verbatim using the audit tool.")
HONEST = "The user asked what stream_chat does; I should read the function first."


def _write(path: Path, *, user: str, thinking: list[str]) -> None:
    path.write_text(json.dumps({"messages": [
        {"role": "user", "content": [{"type": "text", "text": user}]},
        *[{"role": "thinking", "reasoning": t} for t in thinking],
    ]}))


@pytest.fixture
def store(tmp_path):
    """Two sessions: one flagged block in three, one clean — 4 blocks, 2 files.

    The assertions below name those numbers as literals AND compare them against
    `scan_store`'s own tallies, so a failure says which side moved: the literal
    breaking while the comparison holds means the scan's arithmetic changed, and
    a row disagreeing with the scan means the row stopped reporting what the
    scan measured.
    """
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    _write(sessions / "flagged.json", user="what does stream_chat do?",
           thinking=[FABRICATED, HONEST, HONEST])
    _write(sessions / "clean.json", user="and the router?", thinking=[HONEST])
    return sessions


def _rows(series: Path) -> list[dict]:
    return [json.loads(line) for line in
            series.read_text().splitlines() if line.strip()]


def test_record_appends_one_row_with_numerator_and_denominator(store, tmp_path):
    """Clause 3: the row exists, is dated, and names both sides of the rate."""
    series = tmp_path / "series" / "prevalence.jsonl"

    assert tfs.main(["--root", str(store), "--record",
                     "--series", str(series)]) == 0

    scan = tfs.scan_store(store)
    rows = _rows(series)
    assert len(rows) == 1, "one run is one row"
    row = rows[0]
    assert row["flagged"] == scan.flagged == 1
    assert row["blocks"] == scan.blocks == 4
    assert row["files"] == scan.files == 2
    assert row["files_with_flags"] == scan.files_with_flags == 1
    assert row["exempt_blocks"] == scan.blocks_exempt_meta == 0
    assert row["flagged_rate"] == pytest.approx(0.25)
    assert row["scanned"] is True
    assert row["date"] == datetime.now(timezone.utc).date().isoformat()


def test_two_runs_are_two_rows(store, tmp_path):
    """A series, not a snapshot: a second run appends rather than replaces."""
    series = tmp_path / "prevalence.jsonl"
    args = ["--root", str(store), "--record", "--series", str(series)]

    assert tfs.main(args) == 0
    assert tfs.main(args) == 0

    assert len(_rows(series)) == 2


def test_the_default_series_lives_under_the_production_data_root(store, tmp_path,
                                                                  monkeypatch):
    """The gate reads the default path, so the default is the assertion here.

    `production_data_root` is stubbed to a tmp dir rather than left live: the
    suite's `LLOYD_DATA` is already scratch (`tests/conftest.py`), and a test
    that appended to the real file would be the third writer to a file it does
    not own.
    """
    data_root = tmp_path / "data-root"
    monkeypatch.setattr(tfs, "production_data_root", lambda: data_root)

    assert tfs.main(["--root", str(store), "--record"]) == 0

    default = data_root / tfs.PREVALENCE_RELATIVE
    assert default.is_file(), (
        "the series has to be under the production data root, which is where "
        "the nightly jobs and the gate look for a measured series")
    assert len(_rows(default)) == 1


def test_an_empty_store_is_recorded_as_zero_scanned_not_zero_flagged(tmp_path):
    """The clause's whole point: exit 3, `scanned: false`, rate `null`.

    Exit code and row have to agree, because the exit code is what an unattended
    run checks and the row is what a later pass reads. A row that said
    `flagged_rate: 0.0` over an empty store would read as the marker rate having
    fallen to zero, which is the exact false positive the ruling's gate invites.
    """
    empty = tmp_path / "empty"
    empty.mkdir()
    series = tmp_path / "prevalence.jsonl"

    assert tfs.main(["--root", str(empty), "--record",
                     "--series", str(series)]) == 3

    row = _rows(series)[0]
    assert row["flagged"] == 0
    assert row["blocks"] == 0
    assert row["files"] == 0
    assert row["scanned"] is False
    assert row["flagged_rate"] is None, (
        "0.0 over nothing scanned is the ambiguity this row exists to avoid")


def test_a_clean_store_and_an_empty_store_do_not_share_a_row(tmp_path):
    """The pair side by side in one file, still distinguishable by a reader.

    Two stores with `flagged: 0` in their row: one held reasoning and none of it
    was flagged, the other held nothing at all. Only `blocks` and `scanned` tell
    them apart, so both fields are asserted on both rows rather than once.
    """
    clean = tmp_path / "clean"
    clean.mkdir()
    _write(clean / "a.json", user="what does stream_chat do?",
           thinking=[HONEST, HONEST])
    empty = tmp_path / "empty"
    empty.mkdir()
    series = tmp_path / "prevalence.jsonl"

    assert tfs.main(["--root", str(clean), "--record",
                     "--series", str(series)]) == 0
    assert tfs.main(["--root", str(empty), "--record",
                     "--series", str(series)]) == 3

    clean_row, empty_row = _rows(series)
    assert clean_row["flagged"] == empty_row["flagged"] == 0, (
        "the numerator is the same number in both rows, which is the ambiguity")
    assert (clean_row["blocks"], clean_row["scanned"]) == (2, True)
    assert (empty_row["blocks"], empty_row["scanned"]) == (0, False)
    assert clean_row["flagged_rate"] == 0.0
    assert empty_row["flagged_rate"] is None


def test_row_reports_exempt_blocks_on_their_own_field(tmp_path):
    """Exempt blocks stay out of `flagged` and are still counted, not hidden."""
    sessions = tmp_path / "meta"
    sessions.mkdir()
    _write(sessions / "meta.json",
           user="triage #1510: this session is about 'reproduce my complete "
                "previous thinking' traces",
           thinking=[FABRICATED])

    scan = tfs.scan_store(sessions)
    assert (scan.flagged, scan.blocks_exempt_meta) == (0, 1), (
        "the exemption is scan_messages' own; this assertion only pins the "
        "fixture this test needs")
    row = tfs.prevalence_row(scan)
    assert row["flagged"] == 0
    assert row["exempt_blocks"] == 1
    assert row["exempt_files"] == 1
    assert row["blocks"] == 1


def test_prevalence_row_carries_every_field_the_report_prints():
    """No field of the prose row is missing from the machine row.

    Built from a hand-made `StoreScan` so the numbers are literals the assertion
    can name: a row that silently dropped `files_with_flags` would still be
    valid JSON, which is the failure a reader of the series cannot see.
    """
    scan = StoreScan(root=Path("/fixture/sessions"), files=7, unreadable=1,
                     blocks=100, flagged=3, files_with_flags=2,
                     blocks_exempt_meta=5, files_exempt_meta=1,
                     files_archived=4)

    row = tfs.prevalence_row(scan)

    assert row["files"] == 7
    assert row["files_with_flags"] == 2
    assert row["files_archived"] == 4, (
        "#3590: the row stopped carrying the field the prose line prints beside "
        "`files`, so a denominator that half of the sweep had archived would read "
        "the same as one it had not touched")
    assert row["blocks"] == 100
    assert row["flagged"] == 3
    assert row["exempt_blocks"] == 5
    assert row["exempt_files"] == 1
    assert row["unreadable"] == 1
    assert row["flagged_rate"] == pytest.approx(0.03)
    assert row["root"] == "/fixture/sessions"
    assert set(row) >= {"date", "scope", "recorded_at", "root", "files",
                        "files_with_flags", "files_archived", "blocks",
                        "flagged", "exempt_blocks", "exempt_files",
                        "unreadable", "scanned", "flagged_rate"}
    assert row["scope"] == tfs.SCOPE_WHOLE_STORE, (
        "day rows share this file from #1770 on, so the whole-store row has to "
        "say what its denominator covers")


# ---- #1770: the per-day roll-up, and the schedule that writes it -----------------

#: A user turn that trips `names_the_defect`, so its own reasoning is exempt. The
#: per-day series is the one place this matters as a MEASUREMENT rather than a
#: report nicety: the #1656 exclusion moved real blocks out of `flagged`, so the
#: gating question ("is the marker rate falling?") can only be read off a per-day
#: series that already applies it. Live-store exempt counts on the three days the
#: item was triaged over were 0 / 11 / 1 — a fifth of 09-27's fabricated-shape
#: blocks were meta-session, and a roll-up that inherited nothing would call them
#: a rising rate.
META_USER = ("triage #1510: this session is about 'reproduce my complete previous "
             "thinking' traces, so every block below is about the marker words")


def _day_store(where: Path) -> Path:
    """A store on two calendar days: 4 files, 9 blocks, 2 flagged, 2 exempt.

    The split every assertion below names, per `app.thinking_fidelity.scan_file`:

      2026-09-26   2 files, 4 blocks, 1 flagged, 0 exempt
      2026-09-27   2 files, 5 blocks, 1 flagged, 2 exempt

    The meta-session file is what pins WHERE a day's numbers come from. Its two
    fabricated-shape blocks are counted in `blocks` and NOT in `flagged`, which
    only happens if the day's tally goes through `scan_messages`' own exemption; a
    roll-up that scored the marker set itself would say `flagged: 3` for 09-27,
    and one that dropped exempt blocks instead of counting them would say
    `blocks: 3`.
    """
    sessions = where / "days"
    sessions.mkdir(parents=True)
    _write(sessions / "20260926_101010_chat_aaaa.json",
           user="what does stream_chat do?",
           thinking=[FABRICATED, HONEST, HONEST])
    _write(sessions / "20260926_181818_chat_bbbb.json",
           user="and the router?", thinking=[HONEST])
    _write(sessions / "20260927_090909_chat_cccc.json",
           user="what about the router?", thinking=[FABRICATED, HONEST])
    _write(sessions / "20260927_101010_chat_dddd.json", user=META_USER,
           thinking=[FABRICATED, FABRICATED, HONEST])
    return sessions


def _day_rows(series: Path) -> list[dict]:
    return [r for r in _rows(series) if r["scope"] == tfs.SCOPE_DAY]


def test_per_day_reports_one_row_per_calendar_day(tmp_path, capsys):
    """Clause 1: two dated days in the store are exactly two day rows, each one
    the day's own flagged count over the day's own denominator.

    Read off the appended rows rather than off a helper's return value, because
    the thing an operator and the nightly job both use is the recorded row; the
    printed line is asserted in the same breath, since `--per-day` without
    `--record` is the other half of the interface.
    """
    store = _day_store(tmp_path)
    series = tmp_path / "prevalence.jsonl"

    assert tfs.main(["--root", str(store), "--record", "--per-day",
                     "--series", str(series)]) == 0

    days = _day_rows(series)
    assert [r["day"] for r in days] == ["2026-09-26", "2026-09-27"], (
        "one row per YYYYMMDD filename prefix, oldest day first — a merged or "
        "split day means the grouping is not the filename's date")

    d26, d27 = days
    assert (d26["flagged"], d26["blocks"], d26["files"]) == (1, 4, 2)
    assert d26["flagged_rate"] == pytest.approx(0.25)
    assert (d27["flagged"], d27["blocks"], d27["files"]) == (1, 5, 2)
    assert d27["flagged_rate"] == pytest.approx(0.2)
    assert d27["exempt_blocks"] == 2 and d27["flagged"] == 1, (
        "the meta-session blocks are counted and not flagged, which is "
        "scan_messages' exemption running — not a re-derived marker test")

    whole = tfs.scan_store(store)
    assert (sum(r["blocks"] for r in days),
            sum(r["flagged"] for r in days)) == (whole.blocks, whole.flagged), (
        "the days are a partition of the same scan, not a second arithmetic that "
        "happens to agree today")

    out = capsys.readouterr().out
    assert "per-day 2026-09-26: flagged 1 of 4 blocks" in out
    assert "per-day 2026-09-27: flagged 1 of 5 blocks" in out


def test_since_bounds_the_rollup_and_a_zero_block_day_stays_visible(tmp_path,
                                                                    capsys):
    """Clause 2: `--since` cuts the window, and a day inside it with nothing
    scannable is still a row — `blocks: 0` with a `null` rate, not an absence.

    The two halves are opposites and both are asserted: 2026-09-25 must VANISH
    (else `--since` does nothing) and 2026-09-26 must NOT (else a day with no
    reasoning block reads as a day the defect was quiet on, which is the exact
    ambiguity `--record` was built to kill, one level down). The unwindowed run is
    the control that says which of the two flags did the removing.
    """
    sessions = tmp_path / "window"
    sessions.mkdir()
    _write(sessions / "20260925_080000_chat_eeee.json", user="an older day",
           thinking=[FABRICATED, HONEST])
    _write(sessions / "20260926_080000_chat_ffff.json",
           user="a day that recorded no reasoning at all", thinking=[])
    _write(sessions / "20260927_080000_chat_gggg.json", user="a newer day",
           thinking=[HONEST, FABRICATED])

    windowed = tmp_path / "windowed.jsonl"
    assert tfs.main(["--root", str(sessions), "--record", "--per-day",
                     "--since", "2026-09-26", "--series", str(windowed)]) == 0

    days = _day_rows(windowed)
    assert [r["day"] for r in days] == ["2026-09-26", "2026-09-27"], (
        "2026-09-25 is outside the window and gone; 2026-09-26 is inside it with "
        "a zero denominator and must not go with it")
    zero, newer = days
    assert (zero["flagged"], zero["blocks"], zero["files"]) == (0, 0, 1), (
        "the day is present with its own zero, not omitted")
    assert zero["scanned"] is False
    assert zero["flagged_rate"] is None, (
        "0.0 over nothing scanned is the false 'the rate fell to zero' a reader "
        "of the series would take from it")
    assert (newer["flagged"], newer["blocks"]) == (1, 2)

    whole = [r for r in _rows(windowed) if r["scope"] == tfs.SCOPE_WHOLE_STORE]
    assert whole[0]["blocks"] == 4, (
        "--since bounds the ROLL-UP only: the whole-store row beside it still "
        "covers the day the window excluded")

    out = capsys.readouterr().out
    assert "per-day 2026-09-26: flagged 0 of 0 blocks" in out
    assert "2026-09-25" not in out

    unwindowed = tmp_path / "unwindowed.jsonl"
    assert tfs.main(["--root", str(sessions), "--record", "--per-day",
                     "--series", str(unwindowed)]) == 0
    assert [r["day"] for r in _day_rows(unwindowed)] == [
        "2026-09-25", "2026-09-26", "2026-09-27"], (
        "the same store without --since keeps the oldest day, so the window is "
        "what removed it above and not the grouping")


def test_record_with_per_day_appends_day_rows_beside_the_whole_store_row(tmp_path):
    """Clause 3: one run, one file, and every row says what its denominator covers.

    The `blocks` line is the point of the whole change: 4 and 5 are days and 9 is
    the store, in the same file, and `scope` is what stops a reader averaging the
    wrong pair of rows. `scanned` sits beside `flagged` on every row, day or
    store, because a zero numerator means nothing without its denominator.
    """
    store = _day_store(tmp_path)
    series = tmp_path / "series" / "prevalence.jsonl"

    assert tfs.main(["--root", str(store), "--record", "--per-day",
                     "--series", str(series)]) == 0

    rows = _rows(series)
    assert [r["scope"] for r in rows] == [tfs.SCOPE_DAY, tfs.SCOPE_DAY,
                                          tfs.SCOPE_WHOLE_STORE], (
        "the day rows land first, oldest first, then the whole-store row the "
        "flag always wrote — `tail -1` is still the store")
    assert [r["blocks"] for r in rows] == [4, 5, 9]
    for row in rows:
        assert {"scope", "flagged", "blocks", "scanned", "files", "date",
                "recorded_at", "root", "flagged_rate"} <= set(row), (
            f"{row['scope']} row is missing a field another row carries")

    for day_row_out in rows[:2]:
        assert day_row_out["date"] == day_row_out["day"], (
            "a day row's date is the day it measures, so a reader grouping the "
            "series on `date` gets days and not run dates")
    assert rows[2]["date"] == datetime.now(timezone.utc).date().isoformat()
    assert rows[2]["flagged"] == 2 and rows[2]["scanned"] is True


def test_per_day_record_with_a_stubbed_data_root_writes_only_the_series(tmp_path,
                                                                       monkeypatch):
    """Clause 3's write half: the run touches the series and nothing else.

    `production_data_root` is stubbed rather than left live — the suite's
    `LLOYD_DATA` is already scratch, and a test appending to the real series would
    be a third writer on a file the nightly job owns. The `rglob` line is the
    assertion that makes "only" mean something: a run that also wrote a report, a
    lock, or a second copy of the series under another name would still leave
    `default` readable.
    """
    store = _day_store(tmp_path / "store")
    data_root = tmp_path / "data-root"
    monkeypatch.setattr(tfs, "production_data_root", lambda: data_root)

    assert tfs.main(["--root", str(store), "--record", "--per-day"]) == 0

    default = data_root / tfs.PREVALENCE_RELATIVE
    assert default.is_file(), (
        "the scheduled job runs no --series flag, so the default path is the one "
        "the owed checks will look for")
    assert [r["scope"] for r in _rows(default)] == [tfs.SCOPE_DAY, tfs.SCOPE_DAY,
                                                    tfs.SCOPE_WHOLE_STORE]
    assert [p for p in data_root.rglob("*") if p.is_file()] == [default]


#: The two literals clause 4 names, spelled ONCE so the fixture node and the live
#: node cannot drift into testing two different shapes: the first is what the
#: description has to RUN, the second the file it has to grow.
TASK_COMMAND = "scripts.thinking_fidelity_scan --record --per-day"
TASK_SERIES = "_pipeline/metrics/thinking-fidelity-prevalence.jsonl"

#: A fleet-free id, so `_already_ran_this_period` finds no real run record for it.
FIXTURE_TASK_ID = 9991


def _task_file(where: Path, **overrides) -> Path:
    """Write a task file in the shape clause 4 names and return its path.

    Field-for-field the shape #86 ships: `frequency: daily`, the command in the
    front-matter `description`, a `skill_name`, `status: up_next`, and
    `preferred_hours` for the nightly window. `overrides` is how a node breaks one
    field to prove the predicate reads it. JSON is a YAML subset, so `json.dumps`
    renders scalars and `[2, 3]`-style lists in one branch instead of a hand-rolled
    formatter a list can walk straight out of.
    """
    where.mkdir(parents=True, exist_ok=True)
    fields = {
        "type": "autonomy",
        "id": FIXTURE_TASK_ID,
        "name": "Nightly thinking-fidelity prevalence series (fixture)",
        "status": "up_next",
        "frequency": "daily",
        "skill_name": "thinking-fidelity-prevalence-series",
        "preferred_hours": [2, 3],
        "description": (f"Run `.venvs/lloyd/bin/python -m {TASK_COMMAND}` from "
                        f"`~/lloyd`; it appends rows to "
                        f"`~/lloyd-data/{TASK_SERIES}`."),
    }
    fields.update(overrides)
    fm = "\n".join(
        f"{k}: {json.dumps(v)}" for k, v in fields.items() if v is not None)
    path = where / f"{fields['id']}-fixture.md"
    path.write_text(f"---\n{fm}\n---\n\nbody\n", encoding="utf-8")
    return path


def test_the_task_shape_dispatches_and_a_skill_less_copy_never_does(tmp_path,
                                                                    monkeypatch):
    """Clause 4's dispatch half, over a fixture task dir — runs in every gate.

    `@live_vault` is the wrong home for this one: the gate runs `-m "not
    live_vault"` (`pytest.ini:6-12`), so a clause asserted only against the live
    vault has no assertion that runs automatically. This node crosses the real
    scheduler predicate — `app.autonomy._is_task_due` and its explained twin
    `hold_reason`, the two functions `get_due_tasks` gates on — over a task file in
    the shape clause 4 names, and asserts BOTH directions:

      - the shape dispatches: `_is_task_due` True, `hold_reason` None;
      - the same task with `skill_name` emptied does not, forever: False and
        `"no skill"`.

    The second direction is the incident this round was refused for. A skill-less
    task looks entirely healthy on the board — `up_next`, `daily`, a description
    naming the command — while `_is_task_due` (`app/autonomy.py:2187-2199`) returns
    False before it ever reaches the status or the frequency, and the engine's own
    warning says "it will NEVER run until one is set", for a field that bit task
    #79 in June 2026. Asserting the status string cannot catch that; asserting the
    predicate can.

    `_local_hour` is pinned to the task's first preferred hour rather than left on
    the real clock, because the window gate ignores the `now=` argument and reads
    the machine — an unpinned node would be green only if a gate happened to run
    between 02:00 and 03:00 local.
    """
    from app import autonomy

    monkeypatch.setattr(autonomy, "_local_hour", lambda: 2)
    task = autonomy._parse_task_file(_task_file(tmp_path))
    assert task, "the fixture task file did not parse through the real loader"

    assert autonomy._is_task_due(task, [task]) is True, (
        f"the shape clause 4 names does not dispatch: "
        f"hold_reason={autonomy.hold_reason(task, [task])!r}")
    assert autonomy.hold_reason(task, [task]) is None

    skill_less = autonomy._parse_task_file(
        _task_file(tmp_path / "noskill", skill_name=""))
    assert autonomy._is_task_due(skill_less, [skill_less]) is False, (
        "a task with no skill_name must never dispatch — the scheduler warns once "
        "and skips it forever, so a green test here would be a series that "
        "silently never gets written")
    assert autonomy.hold_reason(skill_less, [skill_less]) == "no skill"

    wrong_status = autonomy._parse_task_file(
        _task_file(tmp_path / "draft", status="draft"))
    assert autonomy.hold_reason(wrong_status, [wrong_status]) == "draft", (
        "only up_next dispatches, which is why the live node pins that string "
        "exactly and not a set of plausible statuses")


LIVE_AUTONOMY_DIR = Path.home() / "obsidian" / "autonomy"


@pytest.mark.live_vault
def test_a_daily_autonomy_task_records_the_prevalence_series():
    """Clause 4, against the live vault: the real task file has that dispatchable
    shape. `@live_vault` because the vault is not a tree this round controls
    (`pytest.ini:9-12`); run it from a worktree with

        ~/lloyd/.venvs/lloyd/bin/python -m pytest \\
            tests/test_thinking_fidelity_scan_prevalence_row.py::test_a_daily_autonomy_task_records_the_prevalence_series -m live_vault -q

    The dispatch MECHANICS are pinned by
    `test_the_task_shape_dispatches_and_a_skill_less_copy_never_does`, which runs in
    every gate; what only this node can see is that the file on the board is the
    shape that node proved dispatches. Both read the same two literals from
    `TASK_COMMAND` / `TASK_SERIES` above, so they cannot drift apart.

    Parsed through the real loader (`app.autonomy._parse_task_file`, the call the
    scheduler makes) and found by what its description RUNS rather than by
    filename: the id is whatever the fleet's next free number was, and a test pinned
    to `91-….md` goes red the day another item takes 91. The description is the only
    channel a worker is handed — `_build_task_prompt` renders the `skill_name`
    SKILL.md and the front-matter `description`, never the body — so "command in
    description, not in the body" is load-bearing, not style: a command below the
    fold is a series that never gets written.
    """
    from app import autonomy

    assert LIVE_AUTONOMY_DIR.is_dir(), f"no {LIVE_AUTONOMY_DIR} to read the fleet from"
    hits = []
    for path in sorted(LIVE_AUTONOMY_DIR.glob("[0-9]*-*.md")):
        task = autonomy._parse_task_file(path)
        if task and "thinking_fidelity_scan" in (task.get("description") or ""):
            hits.append((path, task))
    assert len(hits) == 1, (
        f"expected exactly one autonomy task whose description runs "
        f"thinking_fidelity_scan, got {[str(p) for p, _ in hits]}")
    path, task = hits[0]

    desc = " ".join((task.get("description") or "").split())
    assert TASK_COMMAND in desc, (
        f"{path.name}'s description does not run the recording command: {desc[:120]}")
    assert TASK_SERIES in desc, (
        "the description never names the series the job is supposed to grow, so a "
        "worker cannot report the file it wrote")
    assert task.get("frequency") == "daily", (
        "a rate series needs one row per day to answer 'is it falling', and the "
        "#1656 ruling needs 7 days of them")
    assert task.get("status") == "up_next", (
        f"{path.name} is {task.get('status')!r}; _is_task_due dispatches up_next "
        "and nothing else, so any other status is a series that still needs a "
        "human at a terminal")

    skill = str(task.get("skill_name") or "").strip()
    assert skill, (
        f"{path.name} has no skill_name: _is_task_due "
        "(app/autonomy.py:2187-2199) drops it before status or frequency, and "
        "hold_reason calls it 'no skill' — the field that left #79 undispatched "
        "for months")
    assert autonomy._load_skill_content(skill), (
        f"skill_name {skill!r} resolves to no readable SKILL.md, which fails at "
        "dispatch as 'Skill not found' rather than at authoring")
    assert autonomy.hold_reason(task, [task]) != "no skill", (
        "the one hold the authoring side cannot see from the file alone")


# ---- #3590: an archived session is still in the denominator ---------------------

#: The three ways an archive stops being a transcript, each raising a DIFFERENT
#: exception out of the decompressor, measured on this box rather than assumed:
#: a truncated member ends the stream early (EOFError), a valid 10-byte gzip
#: header over bytes that are not a deflate stream fails inside zlib
#: (zlib.error — the one shape that is NOT an OSError), and plain bytes under a
#: `.json.gz` name are not a gzip file at all (BadGzipFile, an OSError). A gzip
#: route that caught only OSError would survive the first and third and die on
#: the second, taking the rest of the store's rows with it.
def _archive_of(raw: bytes) -> bytes:
    """The real archive of `raw`, so a parity test compares a document to itself."""
    return gzip.compress(raw)


def _archive_in_place(path: Path) -> Path:
    """Gzip `path` and unlink it, the way `sweep_sessions` does.

    Same three statements as `scripts/groundskeeper/retention-sweep.py::
    sweep_sessions` — write the `.json.gz` from the original's bytes, verify it
    round-trips as JSON, `unlink()` the original — because the property under
    test is what one night of the real producer does to a day row, and a fixture
    that archived some other way (leaving the plain file standing, say) would be
    describing a state the sweep never writes.
    """
    gz_path = path.with_suffix(".json.gz")
    with path.open("rb") as src, gzip.open(gz_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    with gzip.open(gz_path, "rt", encoding="utf-8") as fh:
        json.load(fh)
    path.unlink()
    return gz_path


def test_an_archived_session_scores_the_same_blocks_and_flags_as_the_plain_file(tmp_path):
    """Clause 1: the stored form is not allowed to change the verdict.

    Two documents, because "the same" has to hold for the exemption too: an
    ordinary session whose block is FLAGGED, and a meta-session whose identical
    blocks are counted and exempted. The plain-side literals are what stop the
    comparison being 0 == 0 — an archive route that returned nothing at all would
    still match a blank document to a blank document.

    `gzip.decompress(gz) == raw` is asserted first so the test cannot pass on a
    file that is plain bytes under a `.gz` name: that would score as unreadable,
    not as equal, and the failure would name the wrong thing.
    """
    store = tmp_path / "sessions"
    store.mkdir()

    ordinary = store / "20260926_101010_chat_aaaa.json"
    _write(ordinary, user="what does stream_chat do?",
           thinking=[FABRICATED, HONEST])
    meta = store / "20260926_111111_chat_bbbb.json"
    _write(meta, user=META_USER, thinking=[FABRICATED, FABRICATED, HONEST])

    for plain in (ordinary, meta):
        raw = plain.read_bytes()
        gz = plain.with_suffix(".json.gz")
        gz.write_bytes(_archive_of(raw))
        assert gz.read_bytes() != raw, (
            f"{gz.name} is not compressed, so this is not the parity the clause "
            "asks about")
        assert gzip.decompress(gz.read_bytes()) == raw, (
            f"{gz.name} does not hold the bytes of {plain.name}, so the two "
            "scans below are not of the same document")

        before, after = tf.scan_file(plain), tf.scan_file(gz)
        assert (after.blocks, after.flagged, after.exempt,
                after.unreadable) == (before.blocks, before.flagged,
                                      before.exempt, before.unreadable), (
            f"{gz.name} scored differently from the identical document stored as "
            f"{plain.name}: archived "
            f"{(after.blocks, after.flagged, after.exempt, after.unreadable)} vs "
            f"plain "
            f"{(before.blocks, before.flagged, before.exempt, before.unreadable)}")
        assert (before.archived, after.archived) == (False, True), (
            f"the form read is wrong for {gz.name}: `archived` is how "
            "files_archived learns which file came off the gzip route, and a "
            "readable archive that reports False would hide the archives the "
            "denominator is actually made of")

    assert (tf.scan_file(ordinary).blocks, tf.scan_file(ordinary).flagged,
            tf.scan_file(ordinary).exempt) == (2, 1, 0), (
        "the plain-side fixture is not the 2-block / 1-flagged session the "
        "parity above is shaped for")
    assert (tf.scan_file(meta).blocks, tf.scan_file(meta).flagged,
            tf.scan_file(meta).exempt) == (3, 0, 2), (
        "the meta-session fixture is not the 3-block / 0-flagged / 2-exempt "
        "session the exemption parity above is shaped for")


def test_every_row_and_both_printed_lines_name_the_archived_share(tmp_path,
                                                                  capsys):
    """Clause 4: the denominator says where it came from, in row and in prose.

    The same four-file, nine-block store as
    `test_per_day_reports_one_row_per_calendar_day`, with one session per day
    archived in place by the real sweep
    shape. `files` and `blocks` are therefore the numbers that test already
    asserts, and the new field is the only thing allowed to move — which is what
    makes this the witness rather than another counter: the sweep archived half
    the store's sessions and every other number stayed where it was.

    The printed lines are asserted as complete strings because the clause is
    that prose and series cannot disagree, and a field present only in the JSONL
    is the disagreement the #1656 docstring already calls out as the bug.
    """
    store = _day_store(tmp_path)
    _archive_in_place(store / "20260926_101010_chat_aaaa.json")
    _archive_in_place(store / "20260927_101010_chat_dddd.json")

    series = tmp_path / "prevalence.jsonl"
    assert tfs.main(["--root", str(store), "--record", "--per-day",
                     "--series", str(series)]) == 0

    rows = _rows(series)
    days = _day_rows(series)
    whole = rows[-1]
    assert [r["files_archived"] for r in days] == [1, 1], (
        "a day's rows do not name the one session of theirs that arrived as an "
        f"archive: {[r['files_archived'] for r in days]}")
    assert whole["files_archived"] == 2 == sum(
        r["files_archived"] for r in days), (
        "the whole-store witness and the day witnesses disagree, so the store row "
        f"says {whole['files_archived']} archived while its own days say "
        f"{sum(r['files_archived'] for r in days)}")
    assert [(r["files"], r["blocks"], r["flagged"]) for r in days] == [
        (2, 4, 1), (2, 5, 1)], (
        "archiving two of four sessions moved a day's denominator, which is the "
        "#3590 failure itself and not a witness problem")
    assert (whole["files"], whole["blocks"], whole["files_archived"]) == (
        4, 9, 2), (
        "the whole-store row is not 4 files / 9 blocks with 2 of them archived")

    out = capsys.readouterr().out
    assert "4 session files, 2 with a flagged block, 2 archived" in out, (
        "the printed whole-store line does not carry the same archived count as "
        "the row, so a human quoting the prose and a job reading the JSONL are "
        "quoting different denominators")
    assert ("per-day 2026-09-26: flagged 1 of 4 blocks (rate 0.2500; "
            "2 session file(s), 1 archived, 0 exempt as meta-session)") in out, (
        "the printed per-day line does not carry the day's own archived count")

    scan = StoreScan(root=Path("/fixture/sessions"), files=9, unreadable=1,
                     blocks=100, flagged=3, files_with_flags=2,
                     blocks_exempt_meta=5, files_exempt_meta=1,
                     files_archived=4)
    assert tfs.prevalence_row(scan)["files_archived"] == 4
    assert tfs.day_row("2026-09-26", scan)["files_archived"] == 4, (
        "the day row kind dropped the field the whole-store kind carries — two "
        "row shapes sharing one file must carry the same denominator fields")
    assert "4 archived" in scan.report(), (
        "StoreScan.report() lost the number every reader of it quotes")


def test_an_undecodable_archive_is_counted_unreadable_and_never_dropped(tmp_path):
    """Clause 5: the gzip route inherits "unreadable is reported, never skipped".

    Three damaged archives, one readable one. The damaged three are the three
    distinct exception shapes listed above, and they sort BEFORE the good file —
    so `blocks == 2` is also the proof that a rotten member did not end the walk
    and cost every later file its blocks, which is the failure
    `app/uptake.py::read_session` names for the same corpus.

    The control at the end is what makes `unreadable` a finding rather than a
    tautology: the same document, archived whole, scans with its two blocks. The
    bytes are unreadable BECAUSE they are damaged, not because the gzip route is
    dead — and a reader who got this wrong would go and "fix" the reader again.
    """
    store = tmp_path / "sessions"
    store.mkdir()
    good_doc = tmp_path / "intact.json"
    _write(good_doc, user="what does stream_chat do?",
           thinking=[FABRICATED, HONEST])
    intact = good_doc.read_bytes()
    whole_archive = _archive_of(intact)

    (store / "a-trunc.json.gz").write_bytes(
        whole_archive[:max(8, len(whole_archive) // 3)])
    (store / "b-junkbody.json.gz").write_bytes(
        b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03" + b"not-a-deflate-stream" * 4)
    (store / "c-plainbytes.json.gz").write_bytes(intact)
    (store / "z-good.json.gz").write_bytes(whole_archive)

    for name in ("a-trunc.json.gz", "b-junkbody.json.gz",
                 "c-plainbytes.json.gz"):
        one = tf.scan_file(store / name)
        assert one.unreadable and one.archived, (
            f"{name}: unreadable={one.unreadable} archived={one.archived} — a "
            "damaged archive has to be reported AND counted as an archive, or "
            "files_archived under-reports on exactly the files whose readability "
            "is in question")
        assert (one.blocks, one.flagged) == (0, 0)
    good = tf.scan_file(store / "z-good.json.gz")
    assert (good.blocks, good.flagged, good.archived,
            good.unreadable) == (2, 1, True, False), (
        "the readable archive is not the 2-block / 1-flagged file the store "
        "assertions below are shaped for")

    scan = tfs.scan_store(store)
    assert (scan.files, scan.files_archived, scan.unreadable) == (4, 4, 3), (
        "damaged archives were dropped from, or not counted as, the denominator: "
        f"files={scan.files} archived={scan.files_archived} "
        f"unreadable={scan.unreadable}")
    assert (scan.blocks, scan.flagged) == (2, 1), (
        "the walk did not reach the good archive after three rotten ones, so a "
        "damaged member cost every later file its blocks")

    series = tmp_path / "prevalence.jsonl"
    assert tfs.main(["--root", str(store), "--record",
                     "--series", str(series)]) == 0
    row = _rows(series)[0]
    assert (row["files"], row["files_archived"], row["unreadable"],
            row["blocks"]) == (4, 4, 3, 2), (
        "the recorded row lost the damaged archives, or the survivor")
    assert row["scanned"] is True, (
        "a store whose archives mostly failed to decode reported nothing "
        "scanned, which is the ambiguity the whole series exists to remove")

    assert tf.scan_file(store / "z-good.json.gz").blocks == 2, (
        "control broken: an intact archive of this very document does not scan, "
        "so the three unreadables above are the reader failing, not the damage")
