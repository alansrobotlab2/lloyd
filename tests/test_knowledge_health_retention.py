"""The Knowledge Health Report's Reflection Retention section (#1227).

The count it states must be `copy_gaps`' count, and an empty result must say
in words that no cycle is missing a copy — a silent section reads the same as
a check that never ran.
"""
import importlib
import importlib.util
import os
import re
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
    note.write_text(note_body, encoding="utf-8")
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
