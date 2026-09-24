"""A report that names its archive copy, and a directory that lacks it (#1227).

Every other check on reflection retention reads skill text, so a run that
skipped the `cp` lost its cycle silently. `copy_gaps` reads the directory, and
takes it as an argument so these tests run on a tmp fixture — unmarked, because
the gate runs `-m "not live_vault"` and a marked test would never be graded.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.reflection_archive import CopyGap, copy_gaps, named_copies, reports_in  # noqa: E402

POINTER = "Archive copied before this write: `signals-latest-2026-09-20-0517.md`\n"
KW_BODY = ("Archived `_pipeline/reflection/tool-patterns-latest-2026-09-20-0938.md` and "
           "`conversation-patterns-latest-2026-09-20-0938.md` before the rewrite.\n")


def _dir(tmp_path: Path, *names: str) -> Path:
    d = tmp_path / "reflection"
    d.mkdir()
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
