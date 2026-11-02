"""The Knowledge Health Report's Reflection Retention section (#1227).

The count it states must be `copy_gaps`' count, and an empty result must say
in words that no cycle is missing a copy — a silent section reads the same as
a check that never ran.
"""
import importlib
import importlib.util
import inspect
import os
import pytest
import re
import stat
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("khr_retention", ROOT / "scripts/memory/knowledge-health-report.py")
khr = importlib.util.module_from_spec(_spec); sys.modules["khr_retention"] = khr; _spec.loader.exec_module(khr)

from scripts.reflection_archive import copy_gaps, pattern_staleness, reports_in  # noqa: E402

#: Reached by attribute, not imported by name, so that a missing #2996 function is
#: one failing node instead of a collection error that takes the six tests above with
#: it — a red that says "this check does not exist" is worth more than a red that
#: says "the file would not load".
ra = importlib.import_module("scripts.reflection_archive")


def _complete_note(body: str) -> str:
    """Bring a fixture completion note over the floor #3596 introduced.

    `pattern_staleness`, `pointer_mtime_lag` and `mtime_witness` now read only a note
    that is at least `ra.REPORT_MIN_BYTES` and does not say `status: in-progress`, and
    every note these fixtures write is 13-40 bytes of the fields under test. Left
    small, the fixtures would not fail — they would go EMPTY, `stale == [s for s in
    stale if …]` would hold vacuously and the lag assertions would compare against a
    check that declined to measure. Detail lines, so the padded body still reads like
    the note a run writes, and `test_every_completion_note_this_file_writes_is_a_report`
    below re-checks the floor rather than trusting it.
    """
    need = ra.REPORT_MIN_BYTES + 1 - len(body.encode())
    return body if need <= 0 else body + "\n" + (
        "Detailed §2d curation record for this cycle.\n" * (need // 42 + 2))


def _stub_note(d: Path, name: str) -> Path:
    """A dated completion note that is NOT a report: #3596's 68-byte shape.

    `run_39_20261030_080034` stopped at `max_turns` (91 turns against the task file's
    `max_turns: 90`) and left `knowledge-write-2026-10-30.md` at 68 bytes still reading
    `status: in-progress`, where the four prior nights are 8,291-13,494 bytes. Written
    directly rather than through `_complete_note`, because staying under the floor is
    the point.
    """
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text("# Knowledge Write " + name[16:-3] + "\n\n"
                 "status: in-progress\n\n## Files written\n", encoding="utf-8")
    y, m, dd = (int(g) for g in re.search(r"(\d{4})-(\d{2})-(\d{2})", name).groups())
    day_ts = datetime(y, m, dd, 1, 0, tzinfo=timezone.utc).timestamp()
    os.utime(p, (day_ts, day_ts))
    return p


_UNSET = object()


def _report(gaps, stale_patterns=None, pointer_lag=_UNSET, witness=_UNSET) -> str:
    """The generated report, with the two #2996 rows defaulted to *unpassed*.

    A sentinel, not `None`: `None` is the *measured-and-absent* state of these rows in
    `generate_report`, so a `None` default would quietly put the section into its
    #1227 not-measured state for every test that does not name them. A test that
    wants a row names it.
    """
    now = datetime.now(timezone.utc)
    kwargs = {}
    if pointer_lag is not _UNSET:
        kwargs["pointer_lag"] = pointer_lag
    if witness is not _UNSET:
        kwargs["mtime_witness"] = witness
    return khr.generate_report({}, khr.compute_relationship_stats([], {}), [], [], now,
                               copy_gaps=gaps, pattern_staleness=stale_patterns, **kwargs)


def _stale_fixture(tmp_path: Path) -> Path:
    """The #2413 state on a fixture: a 10-06 archive under a 10-07 report.

    Built as files, not as a hand-made finding, so what reaches the report is
    `pattern_staleness`'s own verdict — the seam this file exists to cross.
    """
    d = tmp_path / "reflection"
    d.mkdir()
    for name in ("knowledge-write-2026-10-07.md", "tool-patterns-latest.md",
                 "tool-patterns-latest-2026-10-06-0809.md",
                 "conversation-patterns-latest.md",
                 "conversation-patterns-latest-2026-10-07-0812.md"):
        (d / name).write_text("cycle content\n")
    # The note has to clear the floor #3596 set or this fixture measures nothing;
    # `_stale_fixture` is what the #2413 and #2996 nodes below build on.
    (d / "knowledge-write-2026-10-07.md").write_text(_complete_note("cycle content\n"),
                                                     encoding="utf-8")
    return d


def _section(report: str) -> str:
    return report.split("## Reflection Retention", 1)[1].split("\n## ", 1)[0]


def _stated_count(section: str) -> int:
    return int(re.search(r"Reports naming an archive copy that is missing: (\d+)", section).group(1))


def test_section_count_equals_the_functions_count(tmp_path):
    d = tmp_path / "reflection"; d.mkdir()
    (d / "signals-latest.md").write_text(
        "Archive copied before this write: `signals-latest-2026-09-20-0517.md`\n")
    (d / "knowledge-write-2026-09-20.md").write_text(
        "Copies: `tool-patterns-latest-2026-09-20-0938.md`, "
        "`conversation-patterns-latest-2026-09-20-0938.md`\n")
    (d / "tool-patterns-latest-2026-09-20-0938.md").write_text("kept\n")
    gaps = copy_gaps(d, reports_in(d))
    assert len(gaps) == 2
    section = _section(_report(gaps))
    assert _stated_count(section) == len(gaps)
    for gap in gaps:
        assert f"| `{gap.report.name}` | `{gap.copy}` |" in section
    assert "no cycle is missing a copy" not in section


def test_empty_result_says_no_cycle_is_missing_a_copy():
    section = _section(_report([]))
    assert _stated_count(section) == 0
    assert "no cycle is missing a copy" in section


def test_unmeasured_is_not_reported_as_clean():
    section = _section(_report(None))
    assert "Not measured" in section
    assert "no cycle is missing a copy" not in section


# ── #2413: the §2e pattern archive rides the same section ────────────────────


def test_a_stale_pattern_archive_reaches_the_generated_report(tmp_path):
    """Clause 4: a non-empty `pattern_staleness` is readable in the report text.

    The finding has to arrive through `pattern_staleness` over a fixture directory,
    because the whole reason #2413 exists is that the shape was invisible to every
    reader of the live tree: task 39's 2026-10-07 run skipped §2e, named no copy, so
    `copy_gaps` returned 0 findings and the report said the chain was clean. Asserting
    the stem, the stale copy and the report it lags behind — in the section the
    `copy_gaps` rows already live in — is what makes a silently skipped step surface
    nightly instead of only to a 14-day step-conformance replay.
    """
    d = _stale_fixture(tmp_path)
    stale = pattern_staleness(d)
    assert [f.stem for f in stale] == ["tool-patterns-latest"], stale
    section = _section(_report(copy_gaps(d, reports_in(d)), stale))
    assert "Pattern files behind the newest report: 1" in section
    assert "| `tool-patterns-latest.md` | `tool-patterns-latest-2026-10-06-0809.md` " \
           "| `knowledge-write-2026-10-07.md` |" in section
    assert "every §2e archive is current" not in section


def test_a_current_pattern_archive_says_so_without_erasing_the_pointer_rows(tmp_path):
    """The healthy nightly state, stated as a zero, in the same section as the pointer rows.

    The two checks answer different questions and their counts must not be summed or
    read as one verdict: a fixture with a current archive and a pointer naming an
    absent copy is exactly a report that is clean on #2413's question and dirty on
    #1227's. Both numbers print, and the pattern line says zero rather than omitting
    the row — an absent line reads the same as a check that never ran.
    """
    d = _stale_fixture(tmp_path)
    (d / "tool-patterns-latest-2026-10-07-0900.md").write_text("current archive\n")
    (d / "tool-patterns-latest.md").write_text(
        "Archive copied before this write: "
        "`tool-patterns-latest-2026-10-07-0930.md`\n")
    gaps = copy_gaps(d, reports_in(d))
    stale = pattern_staleness(d)
    assert gaps and not stale, (gaps, stale)
    section = _section(_report(gaps, stale))
    assert _stated_count(section) == len(gaps)
    assert "Pattern files behind the newest report: 0" in section
    assert "every §2e archive is current" in section


def test_an_unmeasured_pattern_check_does_not_claim_current_archives():
    """`None` is not a zero: the section must not read as clean when nothing ran.

    The same rail #1227 pinned for the pointer rows, re-pinned for the new ones — a
    reader cannot tell "0 files are behind" from "nobody looked" unless the text says
    which, and the default for every other caller of `generate_report` is `None`.
    """
    section = _section(_report([], None))
    assert "Not measured" in section
    assert "every §2e archive is current" not in section
    assert "Pattern files behind the newest report" not in section


# ── #2996: the pointer's own mtime, which no name stamp can see ───────────────
#
# `_newest_matching` compares the stamp in a file's NAME "and never on mtime", and
# `pattern_staleness` therefore never stats the pointer. #2832 reproduced the
# consequence: a bare `cp` of a stem cleared the check with no pointer write, and
# `knowledge-health-2026-10-18.md:310` printed "Pattern files behind the newest
# report: 0 — every §2e archive is current." in a directory where
# `tool-failure-patterns-latest.md` carried a 2026-10-13 mtime against a newest
# report day of 2026-10-18. These nodes pin the row that reads the mtime, over a
# scratch tree — and never over `PATTERN_STEMS`, which holds only
# `tool-patterns-latest` and `conversation-patterns-latest` and so could not see the
# one real lag on disk.


def _utc_day(d: datetime) -> float:
    """An `os.utime` stamp at 08:00 UTC on `d`'s date."""
    return datetime(d.year, d.month, d.day, 8, 0, tzinfo=timezone.utc).timestamp()


def _mtime_tree(tmp_path: Path, *, pointer_day: datetime, report_day: datetime,
                copy_day: datetime | None = None, stem: str = "tool-patterns",
                note_body: str = "cycle complete, §2e archived\n") -> Path:
    """A reflection directory whose pointer mtime is set deliberately.

    `pointer_day` is the UTC date stamped on `<stem>-latest.md`'s mtime, `report_day`
    the date carried by the newest `knowledge-write-*` note's NAME, and `copy_day` the
    date carried by a dated archive copy's NAME when one is wanted. Name and mtime are
    separate arguments on purpose: the split between them is the whole defect. The
    pointer is named `<stem>-latest.md` because that is the real shape — appending
    `-patterns-latest` to a stem string that already ends in `-latest` matches
    nothing.
    """
    d = tmp_path / "reflection"
    d.mkdir(parents=True)
    note = d / f"knowledge-write-{report_day:%Y-%m-%d}.md"
    note.write_text(_complete_note(note_body), encoding="utf-8")
    os.utime(note, (_utc_day(report_day), _utc_day(report_day)))
    pointer = d / f"{stem}-latest.md"
    pointer.write_text("patterns\n", encoding="utf-8")
    os.utime(pointer, (_utc_day(pointer_day), _utc_day(pointer_day)))
    if copy_day is not None:
        copy = d / f"{stem}-latest-{copy_day:%Y-%m-%d}-0800.md"
        copy.write_text("archive copy\n", encoding="utf-8")
        os.utime(copy, (_utc_day(copy_day), _utc_day(copy_day)))
    return d


def _days(behind: int) -> datetime:
    """Today (or `behind` days ago) on the UTC clock the check reads."""
    return datetime.now(timezone.utc) - timedelta(days=behind)


def _stat_tree(d: Path) -> dict:
    return {p.name: os.stat(p).st_mtime_ns for p in sorted(d.iterdir())}


def test_a_pointer_lag_row_names_its_stem_its_mtime_and_the_note_behind_it(tmp_path):
    """Clause 1: the row's own fields, and that reading them writes nothing.

    Built on `tool-failure-patterns-latest`, the pointer that lagged five days in the
    live tree and which `PATTERN_STEMS` does not contain — so a row set built over
    that constant would return nothing on the tree this item was filed from. The
    before/after stat snapshot is the read-only half of the clause: the check runs
    inside a nightly report that must not disturb the corpus it is measuring.
    """
    today, yesterday = _days(0), _days(1)
    d = _mtime_tree(tmp_path, pointer_day=yesterday, report_day=today,
                    stem="tool-failure-patterns")
    before = _stat_tree(d)
    rows = ra.pointer_mtime_lag(d)
    assert [(r.stem, r.mtime_date, r.report_day) for r in rows] == [
        ("tool-failure-patterns-latest", f"{yesterday:%Y-%m-%d}", f"{today:%Y-%m-%d}")], rows
    assert rows[0].report == f"knowledge-write-{today:%Y-%m-%d}.md"
    assert _stat_tree(d) == before, "the check wrote to or touched the tree it measures"
    doc = ra.pointer_mtime_lag.__doc__ or ""
    assert "#2832" in doc, "the docstring must name the incident it came from"
    assert "not `pattern_staleness`" in doc, doc


def test_a_current_copy_name_does_not_cure_a_stale_pointer_mtime(tmp_path):
    """Clause 2, the false-green control: the name says current, the mtime says not.

    The governed stem has a dated copy whose NAME carries the newest report's day, so
    `pattern_staleness` finds nothing — the exact state that let
    "0 — every §2e archive is current" print over a stale pointer. `pointer_mtime_lag`
    must still return the one row, because the pointer's bytes are a day old.
    """
    today, yesterday = _days(0), _days(1)
    d = _mtime_tree(tmp_path, pointer_day=yesterday, report_day=today, copy_day=today)
    assert pattern_staleness(d) == [], "the control lost its meaning: name stamps are lagging too"
    rows = ra.pointer_mtime_lag(d)
    assert [r.stem for r in rows] == ["tool-patterns-latest"], rows
    assert rows[0].mtime_date == f"{yesterday:%Y-%m-%d}"


def test_a_pointer_written_on_the_newest_report_day_lags_nothing(tmp_path):
    """Clause 3: current mtime, no note, and no directory all read as zero, not as error.

    The no-note and unreadable-directory cases are `_listing`'s: a nightly report that
    raises on a missing reflection tree is worse than one that says nothing, and a
    tree with no dated note has completed no cycle, so no pointer is behind anything.
    """
    today = _days(0)
    assert ra.pointer_mtime_lag(_mtime_tree(tmp_path, pointer_day=today,
                                            report_day=today)) == []
    empty = tmp_path / "no-reports"
    empty.mkdir()
    (empty / "tool-patterns-latest.md").write_text("patterns\n", encoding="utf-8")
    assert ra.pointer_mtime_lag(empty) == []
    assert ra.pointer_mtime_lag(tmp_path / "absent-reflection-tree") == []


def test_the_report_prints_pointer_lag_beside_the_archive_line(tmp_path):
    """Clause 4: both numbers reach the same section, and an unrun check says so.

    The two rows answer different questions and both must be readable together: the
    archive line keeps saying 0 (the dated copy is current by name) while the mtime
    table names the lagging pointer, its mtime day and the note day it lags behind.
    The `None` case is #1227's rail again — an unmeasured pointer check must not leave
    the section implying the pointers are current.
    """
    today, yesterday = _days(0), _days(1)
    d = _mtime_tree(tmp_path, pointer_day=yesterday, report_day=today, copy_day=today)
    section = _section(_report([], pattern_staleness(d),
                               pointer_lag=ra.pointer_mtime_lag(d)))
    assert "Pattern files behind the newest report: 0" in section
    assert "Pattern pointers behind their newest report (by mtime): 1" in section
    assert (f"| `tool-patterns-latest.md` | `{yesterday:%Y-%m-%d}` "
            f"| `{today:%Y-%m-%d}` |") in section
    unmeasured = _section(_report([], [], pointer_lag=None))
    assert "*Not measured: the pattern pointers were not stat'ed" in unmeasured
    assert "Pattern pointers behind their newest report (by mtime): 0" not in unmeasured
    current = _section(_report([], [], pointer_lag=[]))
    assert "Pattern pointers behind their newest report (by mtime): 0" in current


def test_the_mtime_witness_advisory_fires_at_zero_and_clears_at_two(tmp_path):
    """Clause 5: the note's own `mtime=` witness, counted, and worded as a reporting gap.

    Two post-§2e-fix cycles (three counting 2026-10-16) printed no `mtime=` line while
    their pointers were being written, so the absence measures the note and nothing
    about the corpus — the row must never be readable as corpus lag, or it becomes a
    second alarm that a real pointer lag is already saying better from the filesystem.
    """
    today = _days(0)
    silent = _mtime_tree(tmp_path, pointer_day=today, report_day=today)
    w_zero = ra.mtime_witness(silent)
    assert (w_zero.note, w_zero.count) == (f"knowledge-write-{today:%Y-%m-%d}.md", 0), w_zero
    fired = _section(_report([], [], pointer_lag=[], witness=w_zero))
    assert (f"`mtime=` witness lines in `knowledge-write-{today:%Y-%m-%d}.md`: "
            f"0 of 2 expected") in fired
    assert "a REPORTING GAP in the completion note, NOT a lag in the pattern corpus" in fired
    witness_two = ("cycle complete\nmtime=2026-10-19T08:00:01Z tool-patterns-latest.md\n"
                   "mtime=2026-10-19T08:00:04Z conversation-patterns-latest.md\n")
    stamped = _mtime_tree(tmp_path / "two", pointer_day=today, report_day=today,
                          note_body=witness_two)
    w_two = ra.mtime_witness(stamped)
    assert w_two.count == 2, w_two
    cleared = _section(_report([], [], pointer_lag=[], witness=w_two))
    assert "`mtime=` witness lines in " in cleared
    assert "2 of 2 expected" in cleared
    assert "REPORTING GAP" not in cleared
    nowhere = _section(_report([], [], pointer_lag=[], witness=None))
    assert "*Not measured: no dated completion note was readable" in nowhere


# ── #3596: only a COMPLETE completion note may set the §2e bar ─────────────────
#
# `run_39_20261030_080034` reached its 90-turn cap, landed neither of its declared
# sections, and left a 68-byte note reading `status: in-progress`. That note was the
# newest NAME in the directory, so it became the day the archives were compared
# against: `knowledge-health-2026-10-30.md` printed "Pattern pointers behind their
# newest report (by mtime): 3" against yesterday's 1, and "`mtime=` witness lines: 0
# of 2 expected" against the stub — two halves of one cause, a run that wrote nothing
# having stamped the directory with a fresh note.


_STUB_NAME = "knowledge-write-2026-10-30.md"
_COMPLETE_NAME = "knowledge-write-2026-10-28.md"


def _two_current_copies(d: Path) -> None:
    """Both governed stems archived on 10-28, the last complete note's day."""
    for stem in ("tool-patterns", "conversation-patterns"):
        p = d / f"{stem}-latest-2026-10-28-0809.md"
        p.write_text("archive copy\n", encoding="utf-8")
        os.utime(p, (_utc_day(datetime(2026, 10, 28, tzinfo=timezone.utc)),) * 2)


def test_a_stub_completion_note_does_not_move_the_staleness_bar(tmp_path):
    """Clause 2, the 2026-10-30 shape: current archives, a stub dropped on top."""
    d = _stale_fixture(tmp_path)   # report 10-07; tool copy 10-06, conv copy 10-07
    assert [s.stem for s in pattern_staleness(d)] == ["tool-patterns-latest"], (
        "the control stopped measuring: the fixture report is not read as a report")
    _stub_note(d, _STUB_NAME)
    stale = pattern_staleness(d)
    # Graded against the 10-30 stub BOTH stems fall behind, since no copy is dated
    # 10-30 — that doubling is the bug. Against the last complete note the verdict is
    # the control's, unchanged: a stub adds no finding and moves no day.
    assert [s.stem for s in stale] == ["tool-patterns-latest"], (
        f"a stub moved the staleness bar: {[s.stem for s in stale]}")
    assert not getattr(stale, "reason", ""), (
        "a complete note is present, so this is a measurement — not-measured here would "
        "mean the filter threw the baby out with the stub")


def test_a_stub_completion_note_does_not_rename_the_mtime_witness_note(tmp_path):
    """The witness row grades a note's silence; a stub has nothing to be silent about."""
    # The complete note is yesterday's; the stub is dated today, so it is the newer
    # NAME — the exact relation the 10-30 stub had to the 10-29 report.
    yesterday = _days(1)
    d = _mtime_tree(tmp_path, pointer_day=yesterday, report_day=yesterday,
                    note_body="cycle complete\nmtime=2026-10-19T08:00:01Z tool\n"
                              "mtime=2026-10-19T08:00:04Z conversation\n")
    full = ra.mtime_witness(d)
    assert (full.note, full.count) == (f"knowledge-write-{yesterday:%Y-%m-%d}.md", 2), full
    _stub_note(d, _STUB_NAME)
    w = ra.mtime_witness(d)
    assert w.note == full.note, f"the stub was named as the witness note: {w.note}"
    assert w.count == 2, f"reading the stub rewrote the witness count: {w.count}"


def test_a_directory_of_only_stubs_is_not_measured_as_clean(tmp_path):
    """Empty-and-reasoned: still `== []` for every existing reader, and never a zero.

    `test_a_pointer_written_on_the_newest_report_day_lags_nothing` and the report's own
    `elif not …` branches compare against emptiness, so the fix must not turn a stub
    into a finding. What it adds is the reason, carried on the empty list.
    """
    d = tmp_path / "stubs"
    _stub_note(d, _STUB_NAME)
    (d / "tool-patterns-latest.md").write_text("patterns\n", encoding="utf-8")
    stale, lag = pattern_staleness(d), ra.pointer_mtime_lag(d)
    assert stale == [] and lag == [], "a stub was measured as a finding"
    assert getattr(stale, "reason", "") and getattr(lag, "reason", ""), (
        "empty with no reason prints as \"0 — every §2e archive is current\" over a "
        "directory that was never measured")
    assert _STUB_NAME in stale.reason and "in-progress" in stale.reason, stale.reason
    assert ra.mtime_witness(d).note is None, "a stub was named as a witness note"


def test_the_report_says_why_a_stubbed_directory_was_not_measured(tmp_path):
    """The generated section, not just the checks' return values.

    #1227 already had the right instinct for the witness row — `note is None` says
    *not measured* instead of reading a missing note as a silent one. This gives the
    two rows that used to print a confident zero the same shape, and pins that the
    reason is printed: a row that says only "not measured" cannot be acted on.
    """
    d = tmp_path / "rep"
    _stub_note(d, _STUB_NAME)
    section = _section(_report([], pattern_staleness(d),
                               pointer_lag=ra.pointer_mtime_lag(d),
                               witness=ra.mtime_witness(d)))
    assert section.count("Not measured") == 3, section
    assert "every §2e archive is current" not in section
    assert "behind their newest report (by mtime): 0" not in section
    assert _STUB_NAME in section and "in-progress" in section, (
        "the row does not name the note it refused or why")


def test_a_normal_night_still_prints_both_real_rows(tmp_path):
    """The other side of the same rendering: the filter must not blind the report."""
    today, yesterday = _days(0), _days(1)
    d = _mtime_tree(tmp_path / "ok", pointer_day=yesterday, report_day=today,
                    copy_day=today)
    section = _section(_report([], pattern_staleness(d),
                               pointer_lag=ra.pointer_mtime_lag(d),
                               witness=ra.mtime_witness(d)))
    assert "Not measured" not in section, section
    assert "Pattern files behind the newest report: 0" in section
    assert "Pattern pointers behind their newest report (by mtime): 1" in section
    assert "0 of 2 expected" in section


def test_every_completion_note_this_file_writes_is_a_report(tmp_path):
    """The pad in `_complete_note` is load-bearing, so nothing may drift under it.

    A constant no check reads is what the doc-claim rung exists to catch; this one
    runs the real checks over the directories the file's own builders make, so a
    fixture that quietly became a stub shows up as a red node here rather than as a
    vacuously green staleness test.
    """
    d = _stale_fixture(tmp_path)
    note = d / "knowledge-write-2026-10-07.md"
    assert len(note.read_bytes()) >= ra.REPORT_MIN_BYTES, (
        f"{note.name} is {len(note.read_bytes())} bytes, under the "
        f"{ra.REPORT_MIN_BYTES}-byte floor the §2e checks now require")
    assert [s.stem for s in pattern_staleness(d)] == ["tool-patterns-latest"], (
        "the fixture report stopped being read as a report: staleness came back empty")
    today = _days(0)
    m = _mtime_tree(tmp_path / "pad", pointer_day=_days(1), report_day=today)
    assert ra.mtime_witness(m).note == f"knowledge-write-{today:%Y-%m-%d}.md"
    assert [r.stem for r in ra.pointer_mtime_lag(m)] == ["tool-patterns-latest"]
    assert not getattr(pattern_staleness(m), "reason", "")


# ── #3636: a pointer whose stem stopped producing is residue, not lag ───────────────
#
# The live tree gives `pointer_mtime_lag` exactly one finding, and it is
# `tool-failure-patterns-latest` — mtime 2026-10-13 against a 2026-10-31 note, 0 dated
# copies of that stem anywhere in the directory against 24 under `tool-patterns-latest`,
# the name the nightly producer renamed itself to. So §2e's sole signal every night is a
# file nobody owns, which trains the reader past the row #2832 exists to catch. The split
# below is made from the directory listing alone: a stem with a dated `<stem>-<stamp>.md`
# copy inside `RETIRED_COPY_WINDOW_DAYS` of the newest note is producing, one without is
# retired. `PATTERN_STEMS` stays two-wide — #2832 refused to narrow the pointer
# enumeration to it precisely because of this file — and no allowlist of retired stems
# exists, because the next retirement cannot be known from here. Three stems have already
# been retired in this one directory.


def _split_tree(tmp, *, pointer_day: datetime, report_day: datetime,
                stems: dict[str, datetime | None]) -> Path:
    """A reflection directory holding several stale pointers, each with its own copy day.

    `_mtime_tree` writes the two governed stems and cannot express a stem that stopped
    producing, which is the whole variable here. `stems` maps a pointer's name (minus
    `.md`) to the day of its newest dated copy, or None for "no copy of this stem exists".
    Every pointer's mtime is `pointer_day`, so all of them lag; only production differs.
    A copy's own mtime is deliberately NOT what classifies it — the day is read from the
    copy's NAME, because `_newest_matching`'s rule is that a copy restored by hand carries
    an mtime describing the restoring, and `cp`-ing a copy must not forge a producer.
    """
    d = tmp / "refl-split"
    d.mkdir()
    note = d / f"knowledge-write-{report_day:%Y-%m-%d}.md"
    note.write_text(_complete_note("cycle complete, §2e archived\n"), encoding="utf-8")
    os.utime(note, (_utc_day(report_day),) * 2)
    for stem, copy_day in stems.items():
        p = d / f"{stem}.md"
        p.write_text("patterns\n", encoding="utf-8")
        os.utime(p, (_utc_day(pointer_day),) * 2)
        if copy_day is not None:
            c = d / f"{stem}-{copy_day:%Y-%m-%d}-0800.md"
            c.write_text("archived copy\n", encoding="utf-8")
            os.utime(c, (_utc_day(copy_day),) * 2)
    return d


def test_a_stale_pointer_whose_stem_writes_no_copies_is_retired_not_lagging(tmp_path):
    """Clause 1: no lag row for a stem with no dated copy in the window, and the retired
    set names its stem and the day its pointer stopped on.

    `tool-failure-patterns` is the live shape transplanted into a fixture: 18 days behind,
    no dated copy of its stem anywhere. The window is pinned from both sides in the same
    node by `gadget-patterns`, whose newest copy falls exactly on the boundary the
    constant defines, so "within 7 days" is a number the test states rather than one only
    the implementation knows.
    """
    today, eighteen = _days(0), _days(18)
    edge = today - timedelta(days=ra.RETIRED_COPY_WINDOW_DAYS)
    d = _split_tree(tmp_path, pointer_day=eighteen, report_day=today, stems={
        "tool-failure-patterns-latest": None, "gadget-patterns-latest": edge})
    rows = ra.pointer_mtime_lag(d)
    assert [r.stem for r in rows] == ["gadget-patterns-latest"], (
        f"a stem whose newest copy is exactly {ra.RETIRED_COPY_WINDOW_DAYS} days old "
        f"stopped counting as a live producer, so the window is not the span its name "
        f"claims: {[r.stem for r in rows]}")
    assert not any("tool-failure" in r.stem for r in rows), (
        f"a pointer no producer owns is reported as lag again: {[r.stem for r in rows]}")

    retired = list(rows.retired)
    assert [r.stem for r in retired] == ["tool-failure-patterns-latest"], retired
    assert retired[0].mtime_date == f"{eighteen:%Y-%m-%d}", retired
    assert retired[0].newest_copy is None, (
        "a stem with no dated copy at all must not claim one in the row that says so")


def test_the_copy_window_boundary_is_the_constant_and_an_older_copy_is_retired(tmp_path):
    """One day past the window, the same fixture reads as retired.

    The pair this node forms with the boundary case in the node above is what makes
    `RETIRED_COPY_WINDOW_DAYS` a tested number: with only the boundary case, doubling the
    constant would keep both pointers live and nothing would redden.
    """
    today, ten = _days(0), _days(10)
    d = _split_tree(tmp_path, pointer_day=ten, report_day=today, stems={
        "widget-patterns-latest": today - timedelta(days=ra.RETIRED_COPY_WINDOW_DAYS + 1)})
    rows = ra.pointer_mtime_lag(d)
    assert list(rows) == [], (
        f"a stem whose last copy is {ra.RETIRED_COPY_WINDOW_DAYS + 1} days old is still "
        f"reported as a failed cycle: {[r.stem for r in rows]}")
    assert [(r.stem, r.mtime_date) for r in rows.retired] == [
        ("widget-patterns-latest", f"{ten:%Y-%m-%d}")], list(rows.retired)


def test_a_lagging_pointer_whose_stem_has_a_copy_in_the_window_keeps_its_lag_row(
        tmp_path):
    """Clause 2: #2832's case is untouched — same rows, same fields, same type.

    The stale-pointer shape that item caught is a bare `cp` of the dated copy with no
    pointer write: the copy carries the report's day so `pattern_staleness` sees nothing,
    and only the pointer's mtime betrays it. That must keep returning the row it returns
    today, field for field, because §2e's table and every existing reader index it by name
    and a fix that swallowed it would be the worse half of this change.
    """
    today, five = _days(0), _days(5)
    d = _mtime_tree(tmp_path, pointer_day=five, report_day=today, copy_day=today)
    assert pattern_staleness(d) == [], "the #2832 control lost its meaning: names lag too"
    rows = ra.pointer_mtime_lag(d)
    assert [(r.stem, r.mtime_date, r.report, r.report_day) for r in rows] == [
        ("tool-patterns-latest", f"{five:%Y-%m-%d}",
         f"knowledge-write-{today:%Y-%m-%d}.md", f"{today:%Y-%m-%d}")], rows
    assert all(isinstance(r, ra.PointerMtimeLag) for r in rows), type(rows[0])
    assert list(rows.retired) == [], (
        "a pointer whose stem has today's dated copy was classified retired, which would "
        "hide the exact failure #2832 was filed for")


def test_the_live_or_retired_split_is_made_from_the_listing_and_no_stem_list(tmp_path):
    """Clause 3: two stems named nowhere in the code are told apart correctly.

    `widget-patterns` and `gadget-patterns` exist in no tuple, skill or allowlist, so a
    correct split here cannot have come from a lookup. The source assertions are the other
    half of the clause and are not decoration: `PATTERN_STEMS` is the enumeration #2832
    declined to widen, and a round that added the dead stem to it — or read it inside this
    check — would satisfy every behavioural node above while re-narrowing the instrument.
    """
    assert ra.PATTERN_STEMS == ("tool-patterns-latest",
                               "conversation-patterns-latest"), ra.PATTERN_STEMS
    src = inspect.getsource(ra.pointer_mtime_lag) + inspect.getsource(ra._stem_producing)
    assert "PATTERN_STEMS" not in src, (
        "the pointer check now reads the stem tuple #2832 refused to enumerate with")
    consts = [c for c in ra._stem_producing.__code__.co_consts if isinstance(c, str)]
    assert not [c for c in consts if "patterns" in c], (
        f"a stem name is hard-coded into the live/retired test, which is an allowlist: "
        f"{consts}")

    today, ten = _days(0), _days(10)
    d = _split_tree(tmp_path, pointer_day=ten, report_day=today, stems={
        "widget-patterns-latest": None,
        "gadget-patterns-latest": today - timedelta(days=1)})
    rows = ra.pointer_mtime_lag(d)
    assert [r.stem for r in rows] == ["gadget-patterns-latest"], list(rows)
    assert [(r.stem, r.newest_copy) for r in rows.retired] == [
        ("widget-patterns-latest", None)], list(rows.retired)


def test_the_2e_section_prints_zero_lag_and_names_a_retired_pointer_separately(tmp_path):
    """Clause 4: the rendered §2e, both ways round.

    A tree whose only stale pointer is a retired one prints a lag count of 0 and never
    puts the dead file in the lag table — that table is where a reader looks for lag — and
    grows a separate line naming the stem with the day its pointer stopped. The live case
    is asserted in the same node because the fix must not buy its silence by suppressing
    lag: with a current dated copy the row has to reach the table exactly as it does today.
    """
    today, eighteen = _days(0), _days(18)
    dead = _split_tree(tmp_path, pointer_day=eighteen, report_day=today,
                       stems={"tool-failure-patterns-latest": None})
    section = _section(_report([], [], pointer_lag=ra.pointer_mtime_lag(dead)))
    assert "Pattern pointers behind their newest report (by mtime): 0" in section, section
    assert "| `tool-failure-patterns-latest.md`" not in section, (
        "the retired pointer was rendered as lag, where #2832's readers look")
    assert "Retired pattern pointers, not counted above: 1" in section, section
    # The count may be 0 while something IS behind, so the sentence that carries it may
    # not claim the directory is current — that claim is earned only with nothing retired.
    assert "every pointer whose stem is still producing" in section, section
    assert "every `*-patterns-latest.md` in the directory" not in section, section
    assert (f"`tool-failure-patterns-latest.md` — pointer stopped at "
            f"`{eighteen:%Y-%m-%d}`") in section, section
    # The #2832 footnote closes the lag TABLE, so a tree that renders no table must not
    # grow one. It sat inside the retired branch for a revision of this round, at which
    # point a retired-only directory printed prose about a table it did not have.
    assert "`pattern_staleness` above reads name stamps" not in section, (
        "the #2832 footnote printed on a tree with no lag table")

    live = _mtime_tree(tmp_path / "live", pointer_day=eighteen, report_day=today,
                       copy_day=today)
    live_section = _section(_report([], [], pointer_lag=ra.pointer_mtime_lag(live)))
    assert "Pattern pointers behind their newest report (by mtime): 1" in live_section
    assert (f"| `tool-patterns-latest.md` | `{eighteen:%Y-%m-%d}` "
            f"| `{today:%Y-%m-%d}` |") in live_section
    assert "Retired pattern pointers" not in live_section, live_section
    # The footnote is what tells a reader that a current NAME stamp is not a current
    # POINTER: #2832's whole finding. One revision of this round indented those two lines
    # into the retired branch, and the table row above still passed — so this assertion,
    # not the row, is what pins clause 4's "the same table row it prints today".
    assert "`pattern_staleness` above reads name stamps" in live_section, (
        "a live lag lost the #2832 footnote that closes its table")

    clean = _section(_report([], [], pointer_lag=[]))
    assert "Retired pattern pointers" not in clean, (
        "a directory with nothing retired grew a retired-pointer line anyway")
    assert "every `*-patterns-latest.md` in the directory" in clean, (
        "with nothing retired the section lost its unqualified all-current sentence, so "
        "the qualification is being applied to the wrong zero")


def test_the_retired_half_stays_empty_wherever_the_check_declines_to_measure(tmp_path):
    """Clause 5: the new `.retired` half cannot out-vote an unmeasured check.

    `PointerLagRows` is a `list` subclass so every reader of this function as shipped
    keeps working: `len()`, iteration, `not rows` and `== []` must go on meaning the LAG
    rows, and the first case below exercises all four on a measured tree. What is new is
    the second half, and the failure it rules out is asymmetric: a directory whose only
    dated note is #3596's 68-byte `in-progress` stub has no report day, so nothing is
    behind anything, and a retired half populated there would have §2e naming a file as
    residue on the same night it says it could not measure. The dead pointer is in that
    fixture on purpose — 18 days old with no dated copy, the exact shape the retired half
    exists to classify — so this goes red if the classification is ever moved above the
    report-day bar. Read through `khr._retired_of`, the renderer's own reader, because the
    declined case answers with `Unmeasured`, which carries a reason and no `retired` at
    all. The absent-tree case is the same rail from the other side: empty rows, empty
    retired, no invented name.
    """
    today, one = _days(0), _days(1)
    live = _split_tree(tmp_path, pointer_day=one, report_day=today,
                       stems={"tool-patterns-latest": today})
    rows = ra.pointer_mtime_lag(live)
    assert len(rows) == 1 and [r.stem for r in rows] == ["tool-patterns-latest"], list(rows)
    assert rows != [] and bool(rows) and rows[0].mtime_date == f"{one:%Y-%m-%d}", rows
    assert rows.retired == [], (
        f"a stem with a copy inside the window was written off as retired: "
        f"{[(r.stem, r.mtime_date) for r in rows.retired]}")

    d = tmp_path / "stub-only"
    _stub_note(d, _STUB_NAME)
    dead = d / "tool-failure-patterns-latest.md"
    dead.write_text("# Tool Failure Patterns\n\nresidue\n", encoding="utf-8")
    old = _days(18).timestamp()
    os.utime(dead, (old, old))

    declined = ra.pointer_mtime_lag(d)
    assert len(declined) == 0 and list(declined) == [] and declined == [] \
        and not declined, f"an unmeasured check reported lag rows: {list(declined)}"
    assert khr._retired_of(declined) == [], (
        f"the retired half named residue the check never measured against: "
        f"{[(r.stem, r.mtime_date) for r in khr._retired_of(declined)]}")
    assert _STUB_NAME in getattr(declined, "reason", ""), (
        f"the decline named no note, so §2e falls through to its confident zero: "
        f"{getattr(declined, 'reason', '')!r}")
    assert "Pattern pointers behind their newest report (by mtime): 0" not in _section(
        _report([], [], pointer_lag=declined)), (
        "the rendered section printed a zero over a directory that was never measured")

    missing = ra.pointer_mtime_lag(tmp_path / "absent-reflection-tree")
    assert missing == [] and list(missing) == [] and khr._retired_of(missing) == [], (
        "an absent directory grew rows or retired names")


def test_the_deferral_about_the_dead_stem_no_longer_survives_in_the_prose(tmp_path):
    """The comment that parked this decision may not keep parking it.

    `#_POINTER_LATEST`'s comment ended "Whether that file is §2e-governed is a separate
    ruling", and #2996 — the item that shipped the check — is `done`, so no other pass
    would ever revisit it: the deferral is what let one dead file own §2e for weeks. The
    prose is graded as a surface because it is what the next reader of this file reads
    before touching the shape match, so it has to carry both halves of the settled answer:
    the shape stays (that is #2832's finding), and a stem that stopped writing dated
    copies is residue rather than lag.
    """
    src = inspect.getsource(ra)
    head = src[src.index("#: A governed pattern pointer"):]
    block = head[:head.index("_POINTER_LATEST = re.compile")]
    assert "separate ruling" not in block, block
    assert "retired" in block.lower(), block
    assert "#2832" in block, "the reason the shape match stays must stay beside it"
    assert "separate ruling" not in (ra.pointer_mtime_lag.__doc__ or "")
    assert "retired" in (ra.pointer_mtime_lag.__doc__ or "").lower()


# ── #3626: a pointer that is not a regular file has no date of its own ────────
#
# Both governed pointers were symlinks on the live tree from 2026-10-28 08:09 UTC until
# the 2026-10-31 hand repair, and this section printed zero findings for the whole time:
# `(reflection_dir / name).stat()` — the line that measured them — follows the link, so
# the pointer "was written" on the day the archive copy beside it was. The findings below
# are the type, read with `lstat`, and the rule that the followed date is never reported.

def _pointer_as_link(d: Path, *, target_day: datetime, stem: str = "tool-patterns",
                     dangling: bool = False, as_dir: bool = False) -> Path:
    """Replace `<stem>-latest.md` in `d` with a non-regular entry of the same name.

    The replacement is made *after* `_mtime_tree` builds the directory so the link
    overlays the exact path the check enumerates. `target_day` is the mtime of the dated
    archive copy the link names — deliberately a different file's date, because the whole
    defect is that a date read through the link belongs to that copy and not to the
    pointer. `dangling=True` leaves the copy unwritten (the shape a §2e `Write` creates
    when it writes "through" a link whose archive was never made), and `as_dir=True`
    leaves a directory instead, so the type is proven read rather than hard-coded.
    """
    pointer = d / f"{stem}-latest.md"
    pointer.unlink()
    if as_dir:
        pointer.mkdir()
        return pointer
    archive = d / f"{stem}-latest-{target_day:%Y-%m-%d}-0800.md"
    if not dangling:
        archive.write_text("the archive the pointer was replaced by\n", encoding="utf-8")
        os.utime(archive, (_utc_day(target_day), _utc_day(target_day)))
    pointer.symlink_to(archive.name)
    return pointer


def _lstat_tree(d: Path) -> dict:
    """`_stat_tree`'s read-only witness, in the one form that survives a dangling link."""
    return {p.name: os.lstat(p).st_mtime_ns for p in sorted(d.iterdir())}


def test_a_symlinked_pointer_is_a_named_finding_of_its_own_type(tmp_path):
    """Clause 1 of this half: the enumeration names a non-regular pointer it can see.

    The tree is otherwise current — report and archive copy both today — so before #3626
    there was nothing for the check to report at all: the followed mtime read as current,
    the section printed its confident zero, and the pointer had been a link for three
    days. That is the false green, and it is why the finding has to come from `lstat`.
    """
    today = _days(0)
    d = _mtime_tree(tmp_path, pointer_day=today, report_day=today)
    _pointer_as_link(d, target_day=today)
    before = _lstat_tree(d)
    rows = ra.pointer_mtime_lag(d)
    found = rows.non_regular
    assert [(f.stem, f.kind, f.name) for f in found] == [
        ("tool-patterns-latest", "symbolic link", "tool-patterns-latest.md")], found
    assert found[0].target == f"tool-patterns-latest-{today:%Y-%m-%d}-0800.md", (
        "the link's own target is the fact a repairer needs: it names the copy to "
        "`cp -L --remove-destination` back into the pointer")
    assert _lstat_tree(d) == before, "the check is read-only, and `lstat` keeps that true"


def test_a_non_regular_pointer_is_never_given_a_date_it_does_not_have(tmp_path):
    """Clause 1's second half: the followed mtime is not reported, in either direction.

    Here the copy behind the link is a day older than the newest note, so the pre-#3626
    code would have emitted a LAG row — a row whose "Pointer mtime (UTC)" column names a
    date the pointer does not have, in a table the §2e run reads as an instruction about
    its own pointer. The type finding replaces the date row; it does not add to it.
    """
    today, behind = _days(0), _days(1)
    d = _mtime_tree(tmp_path, pointer_day=behind, report_day=today)
    _pointer_as_link(d, target_day=behind)
    rows = ra.pointer_mtime_lag(d)
    assert [f.stem for f in rows.non_regular] == ["tool-patterns-latest"], rows.non_regular
    assert list(rows) == [], (
        f"a symlink's followed mtime was still measured as the pointer's own: {list(rows)}")
    assert rows.retired == [], rows.retired


def test_a_dangling_pointer_link_is_still_a_named_finding(tmp_path):
    """The enumeration may not use anything that resolves the link, not even to exist.

    `Path.is_file()` answers False for a dangling link and would drop the entry, which is
    the one shape with no date to misread and the one a §2e `Write` produces when the
    archive copy it wrote through had never been made. `os.lstat` sees it; the target is
    reported as written, unresolved, because resolving it is exactly what fails.
    """
    today = _days(0)
    d = _mtime_tree(tmp_path, pointer_day=today, report_day=today)
    missing = f"tool-patterns-latest-{today:%Y-%m-%d}-0800.md"
    _pointer_as_link(d, target_day=today, dangling=True)
    rows = ra.pointer_mtime_lag(d)
    assert [(f.stem, f.kind, f.target) for f in rows.non_regular] == [
        ("tool-patterns-latest", "symbolic link", missing)], rows.non_regular
    assert list(rows) == [] and rows.retired == []


def test_a_directory_wearing_a_pointer_name_is_named_by_its_type(tmp_path):
    """The kind is read from the filesystem, so the row cannot say `symbolic link` always.

    A refusal or a report that hard-codes the type it expects is the same blindness one
    level down: it would describe a `mkdir`-shaped pointer as a link and be wrong in the
    sentence that exists to be exact.
    """
    today = _days(0)
    d = _mtime_tree(tmp_path, pointer_day=today, report_day=today)
    _pointer_as_link(d, target_day=today, as_dir=True)
    found = ra.pointer_mtime_lag(d).non_regular
    assert [(f.kind, f.target) for f in found] == [("directory", None)], found


def test_regular_pointers_yield_no_non_regular_finding(tmp_path):
    """The other side of the clause: a healthy directory stays a zero, with no new noise.

    Two current regular pointers, one complete note. If a non-regular finding could fire
    on a regular file, §2e's own self-check would report a defect on every good night and
    the row would be trained away the way the retired pointer trained the reader past the
    lag row (#3636).
    """
    today = _days(0)
    d = _mtime_tree(tmp_path, pointer_day=today, report_day=today)
    other = d / "conversation-patterns-latest.md"
    other.write_text("conversation patterns\n", encoding="utf-8")
    os.utime(other, (_utc_day(today), _utc_day(today)))
    rows = ra.pointer_mtime_lag(d)
    assert rows.non_regular == [], rows.non_regular
    assert list(rows) == [] and rows.retired == []


def test_the_report_names_a_non_regular_pointer_instead_of_claiming_the_directory_is_current(
        tmp_path):
    """Clause 1's last mile: the printed section, because a collected finding nobody
    prints is the same invisibility this item starts from.

    Two sentences are asserted against. The confident zero — "every `*-patterns-latest.md`
    in the directory was written on or after the newest completion note's day" — is the
    line the symlinked pointers bought for three days, and the retired-pointer qualifier
    is not enough either: it still says the pointers that ARE producing were written.
    """
    today = _days(0)
    d = _mtime_tree(tmp_path, pointer_day=today, report_day=today)
    _pointer_as_link(d, target_day=today)
    section = _section(_report([], [], pointer_lag=ra.pointer_mtime_lag(d)))
    assert "Pattern pointers that are NOT regular files: 1" in section, section
    assert "`tool-patterns-latest.md` is a symbolic link" in section, section
    assert "every `*-patterns-latest.md` in the directory was written on or after" not in \
        section, "the section still claims the pointer was written on the copy's date"
    assert "every pointer whose stem is still producing was written on or after" not in \
        section, section


def test_a_stub_only_directory_still_names_a_non_regular_pointer(tmp_path):
    """The type is not a fact about dates, so the completion-note bar does not hide it.

    A directory whose newest note is a 68-byte stub is correctly *not measured* for lag
    (#3596) — and the same directory can hold a pointer that is a link, which is a state
    no note has anything to say about. The section must print both: the decline, and the
    finding. This is the 2026-10-30 shape, where the run that stubbed out also left the
    pointers as links.
    """
    today = _days(0)
    d = _mtime_tree(tmp_path, pointer_day=today, report_day=today)
    _pointer_as_link(d, target_day=today)
    _stub_note(d, f"knowledge-write-{today:%Y-%m-%d}-extra.md")
    stub = d / f"knowledge-write-{today:%Y-%m-%d}.md"
    stub.write_text("# Knowledge Write \n\nstatus: in-progress\n\n## Files written\n",
                    encoding="utf-8")
    rows = ra.pointer_mtime_lag(d)
    assert getattr(rows, "reason", ""), "the stub did not make this a decline after all"
    assert [f.stem for f in rows.non_regular] == ["tool-patterns-latest"], rows.non_regular
    section = _section(_report([], [], pointer_lag=rows))
    assert "Pattern pointers that are NOT regular files: 1" in section, section
    assert "Not measured" in section, section


def test_the_type_is_read_with_lstat_because_stat_follows_the_link(tmp_path):
    """The mechanism as a counterfactual over outcomes, not as a spelling.

    The first version of this node asserted `"lstat(" in inspect.getsource(...)` and that
    `.stat()` and `is_file` were absent from that source. The review rung was right to
    flag it: that pins three identifiers rather than the behaviour, so a refactor that
    renamed a local, reworded the docstring the exclusion had to strip out, or reached the
    same answer through `os.stat(..., follow_symlinks=False)` would go red for a reason
    unrelated to behaviour -- and a `stat()` reached by some other spelling would have
    stayed green while reintroducing the bug. What has to be true is a property of the
    OUTCOME, so this node runs both classifiers over one tree and compares what each finds.

    Two foils, both the pre-#3626 classifier run over a tree rather than described: take
    `_listing`'s names and ask `Path.stat()` whether each is a regular file.

    On a pointer whose target EXISTS that foil classifies the link as regular and finds
    nothing -- the false green the item was written about -- while the shipped enumeration
    names it and gives it no date row.

    The decisive half is the DANGLING pointer. There the foil cannot see the entry at all,
    on both counts: `_listing`'s `p.is_file()` follows the link, finds nothing behind it and
    drops the name, and `Path.stat()` on that name raises. So any implementation that asks
    a following `stat()` what the type is necessarily reports nothing for a dangling pointer,
    whatever it is spelled as, while an `lstat` read names it as a symbolic link. That is an
    outcome the two mechanisms cannot both produce, which is why this node can catch the
    regression without naming `lstat`, `.stat()` or `is_file` in an assertion -- a control
    run for this node found that the exists-target half alone does NOT separate them (a
    foiled classifier hand-fed the shipped finding still passed it), so the dangling half is
    doing the work and is stated as such rather than the whole node being credited with it.
    """
    today = _days(0)
    d = _mtime_tree(tmp_path, pointer_day=today, report_day=today)
    _pointer_as_link(d, target_day=today)

    # Foil one, run: over an existing target the old classifier saw a regular file.
    pre_fix = [n for n in ra._listing(d)
               if ra._POINTER_LATEST.match(n)
               and not stat.S_ISREG((d / n).stat().st_mode)]
    assert pre_fix == [], (
        "the foil found a non-regular pointer, so `Path.stat()` no longer follows the link "
        "and this comparison is no longer the pre-#3626 premise")
    rows = ra.pointer_mtime_lag(d)
    assert [(f.name, f.kind) for f in rows.non_regular] == [
        ("tool-patterns-latest.md", "symbolic link")], rows.non_regular
    assert list(rows) == [], (
        "the named pointer also got a date row, so its followed mtime is being reported")

    # Foil two, run: over a DANGLING pointer the old classifier is blind twice over.
    g = _mtime_tree(tmp_path / "dangling", pointer_day=today, report_day=today)
    _pointer_as_link(g, target_day=today, dangling=True)
    assert "tool-patterns-latest.md" not in ra._listing(g), (
        "the control broke: `_listing` now reports a dangling pointer link, so it is no "
        "longer the pre-#3626 reader and this half proves nothing")
    with pytest.raises(OSError):
        (g / "tool-patterns-latest.md").stat()
    dangling = ra.pointer_mtime_lag(g)
    assert [(f.name, f.kind, f.target) for f in dangling.non_regular] == [
        ("tool-patterns-latest.md", "symbolic link",
         f"tool-patterns-latest-{today:%Y-%m-%d}-0800.md")], dangling.non_regular
    assert list(dangling) == [], (
        "a dangling pointer was given a date row, and the only date it could carry belongs "
        "to a file that does not exist")


def test_the_pointer_producer_set_is_exactly_the_two_stems_the_vault_check_is_scoped_to():
    """Clause 5's boundary as a code fact, so a candidate commit can be graded on it.

    The clause is prose in `steps-2b-2f.md` and its node reads the live vault, which the
    gate's grader cannot see while judging a candidate -- the review rung's advisory named
    exactly that. This node holds the part of the boundary that needs no vault.

    `PATTERN_STEMS` is the producer set, the stems a §2e cycle actually writes, and it is
    exactly the two names the vault-side `find` is scoped to; the two `autonomous-systems-*`
    files are not in it, which is what "outside this contract" means mechanically. The half
    that makes the scoping load-bearing rather than tidy: their names still MATCH
    `_POINTER_LATEST`, the pattern the governed-root check keys on. A wildcard `find` over
    the vault therefore cannot tell a governed pointer from a knowledge note wearing a
    pointer-shaped name by name alone -- the stem set is the only thing that can, and a
    check that skipped it would order the deletion of two files no producer owns.
    """
    assert set(ra.PATTERN_STEMS) == {"tool-patterns-latest", "conversation-patterns-latest"}, (
        "the §2e producer set moved, and skills/nightly-reflection-knowledge-write/"
        "steps-2b-2f.md's vault-side `find` is scoped to exactly these two stems")

    for name in ("autonomous-systems-failure-patterns-latest.md",
                 "autonomous-systems-success-patterns-latest.md"):
        stem = name[: -len("-latest.md")]
        assert stem not in set(ra.PATTERN_STEMS), (
            f"{stem} is now a §2e producer, so the bullet naming it out-of-contract is stale")
        assert ra._POINTER_LATEST.match(name), (
            f"{name} no longer matches the pointer name pattern, so the wildcard `find` "
            "would not sweep it up and the scoping this clause added guards nothing -- "
            "update the bullet instead of leaving it warning of nothing")


# ── the §2e prose these two checks read, pinned from the node that runs the check ──
#
# The §2e prose these two checks read, pinned from the node that runs the check.
#
# Clauses 4 and 5 of #3626 are changes to a skill file, so their evidence is prose and the
# live tree rather than a function. The previous round cited a node in
# `tests/test_skill_reflection_archive.py`, a file its diff did not touch, and the review
# rung returned "no test in this diff pins the prose" -- so every node here is in THIS file,
# beside the `pointer_mtime_lag` code whose behaviour the prose describes. That pairing is
# the point: a §2e run reads the bullet, then calls the tool these checks measure.
#
# Three nodes, and which of them carry `live_vault` is the decision, taken against two
# house rulings that point in opposite directions:
#   - `pytest.ini:14-19` reserves the mark for a node reading live `~/obsidian` STATE,
#     which "no round under test controls" and an hourly promotion "can rewrite between
#     rounds"; `scripts/automod/gate.py:727` runs `-m "not live_vault and not
#     fault_injection"`, so a marked node is deselected on a hard rung and, per
#     `tests/test_automod_doc_claims.py:311-313`, "certifies nothing" about a candidate.
#   - `tests/test_archived_skill_artifacts.py:30-40` rules the other way for a claim that
#     holds for "Any version of the tree" (:32): unmarked by design, because marking it
#     away is how "the checkout was red while every hard rung reported the suite clean"
#     (:14-15).
# A skill's RULE text is the structural kind, and the precedent for parsing it is
# unmarked -- `tests/test_skill_reflection_archive.py:3354`
# (`test_2e_prints_the_archive_names_it_governs_and_states_the_pointer_rule`) reads this
# same file with no mark. So the two prose nodes and the code-fact node below run on the
# hard rung, where the grader can see them; only the node asserting that the deleted stray
# is absent from the live tree is marked, because that is the one claim an unrelated commit
# can falsify.
#
# Every prose assertion goes through `_flat_prose`: the step file is hand-wrapped at ~80
# columns, so a phrase worth asserting is split across a newline in the source, and
# matching it literally would pin the column it happens to break at instead of the rule.


def _flat_prose(text: str) -> str:
    """`text` with every run of whitespace collapsed to one space.

    The step file is hand-wrapped at ~80 columns, so any phrase long enough to matter is
    split across a newline in the source and cannot be matched literally. Collapsing is
    the only way to assert a rule's wording without pinning the column it happens to
    break at — a re-wrap of the file is an edit to its formatting, not to the rule.
    """
    return re.sub(r"\s+", " ", text)


def _kw_step(rel: str = "steps-2b-2f.md") -> str:
    """The live §2b-2f step text, or a raised FileNotFoundError — never an empty string.

    A node reading `""` finds no violation to flag and passes, which for a prose check is
    the whole failure mode: the bullet it is guarding could be deleted and the node would
    still be green. The vault's skills tree is what the nightly run loads.
    """
    p = Path.home() / "obsidian" / "skills" / "nightly-reflection-knowledge-write" / rel
    if not p.is_file():
        raise FileNotFoundError(f"{p} is missing — the §2e step this node reads is gone, "
                                "which is not the same as the rule being satisfied")
    return p.read_text(encoding="utf-8")


def test_the_2e_stray_pointer_bullet_records_the_2026_10_21_mis_root():
    """Clause 4, prose half: the fourth stray is named, dated, and tied to its commit.

    Unmarked, and that is deliberate rather than an oversight: the precedent for reading
    this file is `tests/test_skill_reflection_archive.py:3354`
    (`test_2e_prints_the_archive_names_it_governs_and_states_the_pointer_rule`), which
    parses the same §2e step text and carries no mark, so it runs on the gate's hard rung.
    A skill's RULE is a structural claim about a surface this item changed, and
    `tests/test_archived_skill_artifacts.py:30-40` is explicit that marking that kind of
    node away is how a checkout goes red while every hard rung says clean. What IS marked
    is the node below, which reads the tree STATE.

    The provenance is what a reader cannot re-derive: `7c0fd930` wrote that whole cycle's
    §2e output into `~/obsidian` instead of
    `~/lloyd-data/_pipeline/reflection/`, and the governed directory holds no dated copy
    stamped 2026-10-21 for either governed stem (the listing jumps `-2026-10-20-0807` to
    `-2026-10-22-0804`). That is why this stray is deleted rather than folded back in.
    """
    body = _flat_prose(_kw_step())
    stray = "projects/ai/research/conversation-patterns-latest.md"

    assert stray in body, "§2e no longer names the 2026-10-21 governed-stem stray at all"
    assert "7c0fd930" in body, "the bullet no longer names the commit that mis-rooted it"
    assert "2026-10-21 mis-root" in body, "the bullet no longer dates the mis-root"
    assert "Four governed-stem strays" in body, (
        "the bullet is supposed to record this stray ALONGSIDE the three #2832 lists, "
        "not in place of them — a removed count is a lost precedent")
    for earlier in ("#2832", "#2502"):
        assert earlier in body, f"the bullet no longer cites {earlier}, the prior instance"



@pytest.mark.live_vault
def test_the_2026_10_21_governed_stem_stray_is_out_of_the_live_vault():
    """Clause 4, state half: the file #3626 removed is neither on disk nor tracked.

    Marked, and marked alone: this is the assertion about live vault STATE, which is what
    `pytest.ini:14-19` reserves the mark for -- "no round under test controls" it, and an
    hourly promotion or a nightly job can rewrite it between rounds. Left unmarked it would
    fail the next author for a previous writer's commit, which is the failure
    `tests/test_archived_skill_artifacts.py:14-15` records happening in the other direction.

    Read live rather than inferred from the bullet above, because prose and tree disagree
    in either direction: the bullet can claim a removal that never happened, or the file can
    be gone with no record of why it was ever there. The second is how a later cycle
    re-creates it -- #2832's lesson was that a strayed pointer gets folded back in by the
    next run that finds one -- which is why both halves are needed and neither is enough.
    """
    stray = "projects/ai/research/conversation-patterns-latest.md"
    vault = Path.home() / "obsidian"
    assert not (vault / stray).exists(), (
        f"~/obsidian/{stray} is back on disk — it was #3626's deletion target")
    tracked = [ln for ln in os.popen(
        f"cd {vault} && git ls-files {stray} 2>/dev/null").read().splitlines() if ln.strip()]
    assert tracked == [], f"~/obsidian/{stray} is still tracked by git: {tracked}"


def test_the_2e_vault_check_is_scoped_to_the_two_governed_stems_and_shields_the_other_two():
    """Clause 5: the vault-side check can be satisfied, and cannot order a deletion.

    The item's own wording demanded `find ~/obsidian -name '*-patterns-latest.md'` print
    nothing. That is not satisfiable and obeying it literally deletes two files the same
    item names as out of scope, so the check is the two governed stems with a `-latest*`
    tail — which still catches a governed-stem pointer written anywhere under the vault,
    the defect the check exists for, and additionally catches a dated copy a mis-rooted
    cycle archived into the vault.

    The shield is as load-bearing as the scoping. Both `autonomous-systems-*` files came
    from the same `7c0fd930`, and whether they are another contract's governed pointers
    or knowledge notes with a pointer-shaped name is a ruling no round can make; a §2e
    run that read only "a -latest.md in the vault is a defect" would delete them.
    """
    body = _flat_prose(_kw_step())

    scoped = ("find ~/obsidian -name 'tool-patterns-latest*.md' "
              "-o -name 'conversation-patterns-latest*.md'")
    assert scoped in body, "§2e's vault check is no longer the two governed stems"
    assert "Do **not** run `find ~/obsidian -name '*-patterns-latest.md'` as this check" in body, (
        "the unsatisfiable wildcard check is not explicitly forbidden, so a run that "
        "remembers the item's wording instead of this file will run it")

    for stem in ("autonomous-systems-failure-patterns-latest.md",
                 "autonomous-systems-success-patterns-latest.md"):
        assert stem in body, f"§2e no longer names {stem} as outside this contract"
    assert "never a §2e run's deletion targets" in body, (
        "the two out-of-contract files are no longer shielded from deletion")
    assert "stems absent from `PATTERN_STEMS`" in body, (
        "the reason those stems are out of contract is no longer stated, so a later "
        "reader cannot tell shielded from overlooked")
    assert "names it in its completion note" in body, (
        "the run is left without a what-to-do-instead: name it, do not delete it")
