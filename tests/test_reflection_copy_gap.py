"""A report that names its archive copy, and a directory that lacks it (#1227).

Every other check on reflection retention reads skill text, so a run that
skipped the `cp` lost its cycle silently. `copy_gaps` reads the directory, and
takes it as an argument so these tests run on a tmp fixture — unmarked, because
the gate runs `-m "not live_vault"` and a marked test would never be graded.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reflection_archive import (  # noqa: E402
    PATTERN_STEMS, CopyGap, PatternStaleness, copy_gaps, named_copies,
    pattern_staleness, reports_in)

POINTER = "Archive copied before this write: `signals-latest-2026-09-20-0517.md`\n"
KW_BODY = ("Archived `_pipeline/reflection/tool-patterns-latest-2026-09-20-0938.md` and "
           "`conversation-patterns-latest-2026-09-20-0938.md` before the rewrite.\n")


def _dir(tmp_path: Path, *names: str) -> Path:
    d = tmp_path / "reflection"
    d.mkdir(parents=True)
    for n in names:
        (d / n).write_text("old report\n")
    return d


def test_pointer_naming_an_absent_copy_is_one_finding(tmp_path):
    d = _dir(tmp_path)
    report = d / "signals-latest.md"
    assert copy_gaps(d, {report: POINTER}) == [
        CopyGap(report, "signals-latest-2026-09-20-0517.md")]


def test_body_naming_dated_copies_reports_only_the_missing_one(tmp_path):
    d = _dir(tmp_path, "tool-patterns-latest-2026-09-20-0938.md")
    report = d / "knowledge-write-2026-09-20.md"
    gaps = copy_gaps(d, {report: KW_BODY})
    assert gaps == [CopyGap(report, "conversation-patterns-latest-2026-09-20-0938.md")]


def test_date_only_stamp_is_parsed_not_skipped(tmp_path):
    d = _dir(tmp_path)
    report = d / "signals-latest.md"
    body = "Previous cycle kept at `signals-latest-2026-08-31.md`.\n"
    assert named_copies(body) == ["signals-latest-2026-08-31.md"]
    assert copy_gaps(d, {report: body}) == [CopyGap(report, "signals-latest-2026-08-31.md")]


def test_every_named_copy_present_is_no_finding(tmp_path):
    d = _dir(tmp_path, "signals-latest-2026-09-20-0517.md",
             "tool-patterns-latest-2026-09-20-0938.md",
             "conversation-patterns-latest-2026-09-20-0938.md")
    assert copy_gaps(d, {d / "signals-latest.md": POINTER,
                         d / "knowledge-write-2026-09-20.md": KW_BODY}) == []


def test_report_naming_no_copy_is_not_a_gap(tmp_path):
    # A cycle whose job never wrote the pointer (or wrote nothing to archive)
    # is not a gap: the check asks no calendar-day question. The bare live name
    # `tool-patterns-latest.md` is not a dated copy either.
    d = _dir(tmp_path)
    body = ("No archive copy was needed: `_pipeline/reflection/tool-patterns-latest.md` "
            "does not exist on this box.\n")
    assert named_copies(body) == []
    assert copy_gaps(d, {d / "knowledge-write-2026-09-16.md": body}) == []


def test_removing_one_named_copy_yields_exactly_that_finding(tmp_path):
    names = ("signals-latest-2026-09-20-0517.md",
             "tool-patterns-latest-2026-09-20-0938.md",
             "conversation-patterns-latest-2026-09-20-0938.md")
    d = _dir(tmp_path, *names)
    (d / "signals-latest.md").write_text(POINTER)
    (d / "knowledge-write-2026-09-20.md").write_text(KW_BODY)
    assert copy_gaps(d, reports_in(d)) == []
    (d / names[1]).unlink()
    assert copy_gaps(d, reports_in(d)) == [
        CopyGap(d / "knowledge-write-2026-09-20.md", names[1])]


def test_reports_in_skips_the_dated_copies_themselves(tmp_path):
    d = _dir(tmp_path, "signals-latest-2026-09-19-0500.md")
    (d / "signals-latest.md").write_text(POINTER)
    assert set(reports_in(d)) == {d / "signals-latest.md"}


# ── #2413: pattern files whose archive fell behind the cycles ────────────────
#
# The nodes above cover `copy_gaps`, which reads a report's own pointer. These
# cover `pattern_staleness`, which reads only the directory: task 39's run of
# 2026-10-07 skipped §2e wholesale, named no copy, and so was invisible to the
# pointer check — `copy_gaps` returns 0 findings on the live directory while
# `tool-patterns-latest.md` on disk still said "written 2026-10-06".

KW_1007 = "knowledge-write-2026-10-07.md"


def test_a_stale_newest_copy_is_exactly_one_finding_naming_the_stem(tmp_path):
    """Clause 1: the 2026-10-07 shape, reconstructed on a fixture.

    The live directory held `tool-patterns-latest.md` (first line "written
    2026-10-06") whose newest dated copy was `…-2026-10-06-0809.md`, beside a
    `knowledge-write-2026-10-07.md` report. The healthy `conversation-patterns`
    sibling is in the same fixture so "exactly one" means the check is per stem:
    a rule that reported both files, or none, would pass a weaker assertion.
    """
    d = _dir(tmp_path, KW_1007, "tool-patterns-latest.md",
             "tool-patterns-latest-2026-10-06-0809.md",
             "conversation-patterns-latest.md",
             "conversation-patterns-latest-2026-10-07-0812.md")
    assert pattern_staleness(d) == [
        PatternStaleness("tool-patterns-latest",
                         "tool-patterns-latest-2026-10-06-0809.md", KW_1007)]


def test_a_copy_stamped_the_same_day_currents_it_and_the_name_decides(tmp_path):
    """Clause 2: the healthy nightly state, judged by the UTC stamp, not by mtime.

    The stale-named copy is the one with the newer mtime, so a check reading
    modification times would pick it, see 2026-10-05 under the 2026-10-07 report,
    and fire. §2e's whole stamp rule is `date -u` in the filename, so the name is
    the authority and the mtime is the restore that overwrote it.
    """
    d = _dir(tmp_path, KW_1007, "tool-patterns-latest.md",
             "tool-patterns-latest-2026-10-07-2359.md")
    stale = d / "tool-patterns-latest-2026-10-05-0800.md"
    stale.write_text("older content, newer mtime\n")
    now = time.time()
    os.utime(stale, (now, now))
    os.utime(d / "tool-patterns-latest-2026-10-07-2359.md", (now - 86400, now - 86400))
    assert stale.stat().st_mtime > (d / "tool-patterns-latest-2026-10-07-2359.md").stat().st_mtime
    assert pattern_staleness(d) == []


def test_no_knowledge_write_report_means_nothing_is_owed(tmp_path):
    """Clause 3: a first-run or swept tree is silence, not a gap.

    Every governed file and a dated copy are present, so the only thing missing is
    a completed cycle. Attributing the silence to the missing report is why the
    `-latest` files are here: with them absent, this would pass for the wrong
    reason.

    A directory that does not exist at all is the same case one step earlier, and it
    is not hypothetical for this check: `copy_gaps` never lists the tree, so a missing
    reflection directory has never been its problem, while this one lists it — the
    report's own `main()` over an empty tmp tree raised `FileNotFoundError` here on
    the first run of this node. Silence, not a traceback, is the first-run answer.
    """
    d = _dir(tmp_path, "tool-patterns-latest.md", "tool-patterns-latest-2026-10-06-0809.md",
             "conversation-patterns-latest.md",
             "conversation-patterns-latest-2026-10-06-0810.md")
    assert pattern_staleness(d) == []

    absent = tmp_path / "reflection-never-created"
    assert not absent.exists()
    assert pattern_staleness(absent) == []


def test_zero_copies_owe_an_archive_only_after_a_second_cycle(tmp_path):
    """The two shapes this check had to be told apart, in one fixture pair.

    One report and no copy is §2e's first run: `test -f` gates the `cp` because
    there is nothing to archive yet, so a finding here is the false alarm the
    module's docstring warns about. Two reports and no copy is #436's live shape —
    nightly runs from 2026-08-22 to 2026-09-12 with not one dated copy on disk —
    and does fire. And a directory whose only `knowledge-write-*` file is the
    `…-error-…` variant names no completed cycle either: that file exists live
    (`knowledge-write-error-2026-09-24.md`) for a cycle that died before §2e.
    """
    first = _dir(tmp_path / "first", "knowledge-write-2026-10-07.md",
                 "tool-patterns-latest.md", "conversation-patterns-latest.md")
    assert pattern_staleness(first) == []

    second = _dir(tmp_path / "second", "knowledge-write-2026-10-06.md", KW_1007,
                  "tool-patterns-latest.md", "conversation-patterns-latest.md")
    # Emitted in governed-stem order, which is the order the report table shows,
    # not alphabetically: `PATTERN_STEMS` is the contract the rows follow.
    assert [f.stem for f in pattern_staleness(second)] == list(PATTERN_STEMS)
    assert all(f.newest_copy is None for f in pattern_staleness(second))

    errored = _dir(tmp_path / "errored", "knowledge-write-error-2026-09-24.md",
                   "tool-patterns-latest.md", "tool-patterns-latest-2026-09-20-0800.md")
    assert pattern_staleness(errored) == []
