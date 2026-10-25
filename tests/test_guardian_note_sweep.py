"""The note sweep's second refuter: a measured fraction, not a supervisor state (#3277).

`scripts/maintenance/guardian_note_sweep.py` retracts the daily-note alarm blocks that a
live read refutes, and until this item it had exactly one read — `supervisorctl status`.
That refuter is structurally blind to the `/tmp is running out of inodes` family, not
merely out of window: `STATE_LINE` looks for a `name: STATE` line, and a tmpwatch body
contains no supervisor state string at all, so `refutations()` returns `[]` for every one
of them whatever the supervisor says. Triage measured it: the sweep's own
`count_refutations` returned 0 for `memory/2026-10-06.md`, `2026-10-15.md`,
`2026-10-16.md` and 0 corpus-wide, while seven unsealed blocks sit in those files
(`2026-10-06.md:13`, `2026-10-15.md:21` and `:55`, `2026-10-16.md:13,37,61,89`).

Blindness, not age, is what makes this worth a second refuter. `guardian.py:2764` returns
`sealed-note-out-of-reach` for exactly this shape — the incident is closed in `ALERT.md`
while the note it alarmed in is outside `notify.DAILY_SCAN_DAYS = 3` and stays open — so
every future tmpwatch seal re-accumulates stranded blocks with nothing able to date them.

The witness this family does have is `tmpwatch.measure`, the same `statvfs` reading that
raised the alarm and the same comparison `Tmpwatch.tick` makes when it resolves one:
`fraction < tmpwatch.CLEAR_FRACTION`. Measured 2026-10-25 on this box: 0.2769 against a
0.70 clear line, while the bodies quote 81% and 82%. A refuter whose witness is a fresh
number can therefore say something true about a stale figure — but only when the number
is actually below the line, which is what most of the nodes here exist to refuse.

Two properties are load-bearing, and both came from prior incidents rather than taste:

* the reading is injected through a `measurer=` / `fraction_reader=` seam, because the
  gate runs the suite where `/tmp` cannot be steered — `read_running`'s `runner=` exists
  for the same reason, and the module docstring at `guardian_note_sweep.py:206-215`
  records #2221's two clause-5 nodes coming back red at `cb235020` for exactly this;
* the stamp is a `dated:` and never a `cleared:`, and neither marker word appears in its
  prose, because the corpus instrument for #2221 is `grep -c 'cleared:'` over the dated
  notes (#1988: an instrument must never read its own description).

The process boundary is the note file: the sweep runs as its own process and its output is
read later by a human or a signal pass, so every node judges re-read bytes — text from the
committed fixture, and file bytes for the CLI nodes. `tests/test_guardian_alert_retraction.py
:449-651` owns the supervisorctl half and is unmodified by this diff; `REFUTED_LINE`'s bytes
are unchanged, and the node at the end of this file pins the two phrases that carry its
honesty claim so a reword there cannot pass silently.
"""
from __future__ import annotations

import importlib.util as _ilu
import shutil
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARDIAN_DIR = ROOT / "agent-services" / "guardian"
for _p in (str(ROOT), str(GUARDIAN_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import notify  # noqa: E402
import tmpwatch as TW  # noqa: E402

_SWEEP_PATH = ROOT / "scripts" / "maintenance" / "guardian_note_sweep.py"
_spec = _ilu.spec_from_file_location("guardian_note_sweep_fraction", _SWEEP_PATH)
SWEEP = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(SWEEP)

#: The real bytes of `~/obsidian/memory/2026-10-15.md` as committed at triage: the two
#: heading-level `/tmp is running out of inodes` sections at :21 and :55, `cleared:` zero,
#: `dated:` zero. Committed rather than read live because the gate runs with `$HOME` at the
#: round home, where `~/obsidian` does not fully exist — a node that opened the live note
#: would skip there and pin nothing, which is the "green means nothing here" defect this
#: script exists to fix. It is a copy, so the apply the item owes remains a human act.
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "guardian_note_2026-10-15_pre-sweep.md"

DAY = "2026-10-25"
#: `tmpwatch.measure("/tmp").fraction` as measured on this box on 2026-10-25.
BELOW = 0.2769
#: The 81% the `:21` body quotes. At or over the clear line it is not refuted, and the
#: sweep may never date from a figure it is reading out of the body itself.
ABOVE = 0.81
#: No program names are RUNNING: the supervisor half must contribute nothing to any node
#: in this file, so every stamp counted here can only have come from the fraction witness.
NO_PROGRAMS: frozenset[str] = frozenset()


def _pre() -> str:
    return FIXTURE.read_text(encoding="utf-8")


def _reading(fraction: float) -> TW.Reading:
    """A `tmpwatch.measure` return whose `.fraction` is `fraction`.

    Built through the producer's own dataclass rather than a bare float, because
    `read_fraction`'s seam is `tmpwatch.measure` itself: a stand-in returning something
    `measure` never returns would exercise a shape the real caller cannot produce.
    `Reading.fraction` is the MAX of the inode and byte fractions, so both halves are set
    here — leaving the byte half at zero would not lower the reading, it would raise every
    reading to 1.0.
    """
    inodes, blocks = 1_048_576, 2 ** 30
    return TW.Reading(path=TW.PATH, inodes_total=inodes, inodes_used=int(fraction * inodes),
                      bytes_total=blocks, bytes_used=int(fraction * blocks))


def _tmp_sections(text: str) -> list[str]:
    """The bodies of the `/tmp is running out of inodes` blocks, in document order."""
    return [text[start:end] for title, start, end in SWEEP._guardian_sections(text)
            if TW.ALERT_TITLE in title]


def _stamp_lines(text: str) -> list[str]:
    """The `dated:` lines in the note, in document order.

    Line-anchored rather than a substring count, because clause 2 is about the STAMP and
    the fixture's own prose is free to mention a date: `timestamp: '2026-10-15T04:41:21'`
    in the front matter is a field, not a retraction, and a count of the bare word would
    grade it as one.
    """
    return [ln for ln in text.splitlines()
            if ln.startswith(notify.DAILY_DATED_PREFIX)]


# ── clause 1: dated only below the line, untouched at or over it or on a failed read ──


def test_an_unsealed_tmp_block_is_dated_only_when_the_measured_fraction_is_below_the_line():
    """The witness that exists, used at the honest moment: 0.2769 < 0.70.

    Two candidate blocks in, two stamps out — the count comes from the sweep's own
    `count_refutations`, the same helper the CLI prints, so "0 hits" from the tool and "0
    hits" from this node are one measurement (#2221's rule for the supervisor half, applied
    to the second refuter). With no RUNNING program passed in, nothing here can have come
    from `supervisorctl`.
    """
    pre = _pre()
    assert len(_tmp_sections(pre)) == 2, "the fixture is the two-block note the item names"
    assert SWEEP.count_refutations(pre, NO_PROGRAMS, BELOW) == 2

    new, n, notes = SWEEP.date_note(pre, NO_PROGRAMS, DAY, BELOW)
    assert n == 2 and len(notes) == 2, (n, notes)
    assert len(_stamp_lines(new)) == 2, new[-600:]
    assert all(TW.ALERT_TITLE in t for t in notes), (
        "every reported dating must name the block it answers: " + repr(notes))


@pytest.mark.parametrize("fraction", [TW.CLEAR_FRACTION, TW.CLEAR_FRACTION + 0.01, ABOVE, 1.0],
                         ids=["at-the-line", "just-over", "the-quoted-81-percent", "full"])
def test_a_tmp_block_is_left_untouched_when_the_reading_is_at_or_over_the_clear_line(fraction):
    """The comparison is `tick`'s (`tmpwatch.py:207`), so the line is exclusive.

    `fraction < CLEAR_FRACTION` is the exact test the guardian itself uses to resolve an
    incident; 0.70 therefore refutes nothing. The `the-quoted-81-percent` case is the one
    that matters: the bodies say 81%, so a sweep that read the figure it was refuting from
    the block would date them forever. Here the reading is injected, it is at or over the
    line, and the note's bytes come back unchanged.
    """
    pre = _pre()
    assert SWEEP.count_refutations(pre, NO_PROGRAMS, fraction) == 0

    new, n, notes = SWEEP.date_note(pre, NO_PROGRAMS, DAY, fraction)
    assert new == pre and n == 0 and notes == []


def test_a_fraction_that_cannot_be_read_raises_rather_than_becoming_a_zero():
    """`read_fraction` refuses, in the same shape as `read_running` refuses.

    A `0.0` from a failed `statvfs` would read as "totally clean" and date every block in
    the corpus, which is the empty-table refusal one level up. So an exception out of the
    measurer becomes a RuntimeError naming the tool, and a value that cannot be a usage
    fraction — NaN, a negative, anything above 1.0 — is refused rather than compared. The
    measurer seam passes through `tmpwatch.measure`'s own return type, so the stand-ins
    return the same shape and the node pins the parsing, not the box.
    """
    def _boom(path: str = TW.PATH):
        raise OSError("statvfs: /tmp: Transport endpoint is not connected")

    with pytest.raises(RuntimeError):
        SWEEP.read_fraction(measurer=_boom)

    for junk in (float("nan"), -0.1, 1.5):
        with pytest.raises(RuntimeError):
            SWEEP.read_fraction(measurer=lambda path=TW.PATH, _v=junk:
                                types.SimpleNamespace(fraction=_v))

    assert SWEEP.read_fraction(measurer=lambda path=TW.PATH: _reading(BELOW)) == pytest.approx(BELOW)


def test_a_block_no_tool_can_re_read_is_left_byte_identical():
    """The second refuter widens the sweep to ONE title family, not to every number.

    A disk-usage figure with no measurement function behind it is exactly the block the
    module docstring has always excluded. `fraction` is below the line here, so the only
    thing that can stop a careless implementation is binding the witness to the alert title
    it measures — which is what this node fails on.
    """
    note = ("---\nsegment: memory\n---\n"
            "## Self-mod guardian: Runtime data is written into the code tree\n\n"
            "91% of the volume is in `/home/alansrobotlab/lloyd/.t`.\n"
            "Find the writer, move the data across, and remove the in-tree copy.\n")
    assert SWEEP.count_refutations(note, NO_PROGRAMS, BELOW) == 0
    new, n, notes = SWEEP.date_note(note, NO_PROGRAMS, DAY, BELOW)
    assert new == note and n == 0 and notes == []


# ── clause 2: the stamp names tool, value, threshold and read date, and no marker word ──


def test_the_new_stamp_names_its_tool_its_value_its_threshold_and_the_day_it_was_read():
    """Four names per stamp, and neither marker word anywhere but its own prefix.

    Each half is falsifiable by a different wrong line. A stamp that only said "looks fine
    now" would name none of the four; one that said `cleared:` would be counted by #2221's
    `grep -c 'cleared:'` instrument as a recovery somebody watched, which is #1988's
    instrument-reading-its-own-description defect re-opened. A second `dated:` in the prose
    would make one pass look like two. The check is the same substring the sealing code and
    `already_sealed()` use, so a rewording that introduced either word in ANY case is
    caught, and the literal lowercase forms the instrument greps are asserted beside it.
    """
    new, n, _ = SWEEP.date_note(_pre(), NO_PROGRAMS, DAY, BELOW)
    assert n == 2
    stamps = _stamp_lines(new)
    assert len(stamps) == 2, stamps

    for stamp in stamps:
        assert "tmpwatch" in stamp and "statvfs" in stamp, (
            "a stamp must name the tool that produced it, or a reader has nothing to "
            "re-run: " + stamp)
        assert f"{BELOW:.4f}" in stamp, "the measured value must be on the line: " + stamp
        assert "CLEAR_FRACTION" in stamp and f"{TW.CLEAR_FRACTION:.2f}" in stamp, (
            "the threshold must be on the line, or the value cannot be judged: " + stamp)
        assert DAY in stamp, "the read date must be on the line: " + stamp
        assert stamp.count(notify.DAILY_DATED_PREFIX) == 1, (
            "a second `dated:` in prose makes one pass read as two: " + stamp)
        assert notify.DAILY_CLEARED_PREFIX not in stamp.lower(), stamp
        assert "cleared:" not in stamp and "dated:" not in stamp[len(notify.DAILY_DATED_PREFIX):]

    assert new.count("cleared:") == 0, "the #2221 instrument must stay at zero"


def test_the_fraction_witness_never_writes_the_guardians_observed_recovery_marker():
    """`grep -c 'cleared:'` over a note this refuter dated stays 0 — the whole-file claim.

    Separate from the line-level assertion above because it is the property the corpus
    instrument measures: the note as a whole, front matter and alarm bodies included, must
    hold no `cleared:` after the pass. `notify.DAILY_CLEARED_PREFIX in new.lower()` is the
    same question case-insensitively, so a stamp reworded to `Cleared:` is caught too.
    """
    new, n, _ = SWEEP.date_note(_pre(), NO_PROGRAMS, DAY, BELOW)
    assert n == 2
    assert "cleared:" not in new
    assert notify.DAILY_CLEARED_PREFIX not in new.lower()
    assert new.count(notify.DAILY_DATED_PREFIX) == 2, (
        "one marker per pass: the count is what a later reader counts")


# ── clause 4: one stamp, ever ──


def test_the_fraction_stamp_is_written_once_and_the_second_pass_finds_nothing():
    """Idempotence, because this pass is run by hand over published notes.

    `already_sealed()` is what stops the second write, and the node asserts it through
    `count_refutations` rather than only through `date_note`, because the CLI decides
    whether to open a file with that helper: a `date_note` that wrote nothing while
    `count_refutations` still reported 2 would have `main` rewrite identical bytes on every
    run and print a nonzero count while it did.
    """
    pre = _pre()
    once, n1, _ = SWEEP.date_note(pre, NO_PROGRAMS, DAY, BELOW)
    assert n1 == 2
    assert SWEEP.count_refutations(once, NO_PROGRAMS, BELOW) == 0

    twice, n2, notes2 = SWEEP.date_note(once, NO_PROGRAMS, "2026-10-26", BELOW)
    assert twice == once and n2 == 0 and notes2 == []


# ── clause 5: the real bytes of 2026-10-15.md, both sections, bodies intact ──


def test_the_real_2026_10_15_bytes_gain_a_stamp_under_both_sections_with_both_bodies_intact():
    """The dated note, measured against the bytes the item names.

    The bodies are snapshotted from the fixture's own section slices before the pass and
    required back verbatim after it, so "byte-for-byte intact" is asserted against the
    region the sweep is supposed to append below rather than against a hand-typed string.
    The `/tmp: … inodes used (N%)` lines and the largest-entries listings are named as
    well, because those are the figures a reader would notice having been edited, and each
    of the two blocks has its OWN numbers — 845,813/(81%) with a 298,118 `/tmp/fwg-b` at
    :21, 858,283/(82%) with a 278,045 one at :55 — so a stamp landing in the wrong section
    would still leave one body intact and is caught by the per-section loop instead.
    """
    pre = _pre()
    for marker in ("/tmp: 845,813 of 1,048,576 inodes used (81%)",
                   "    298,118  /tmp/fwg-b",
                   "/tmp: 858,283 of 1,048,576 inodes used (82%)",
                   "    278,045  /tmp/fwg-b",
                   "_(still open on the next check)_"):
        assert marker in pre, f"the fixture is not the note the item quoted: {marker!r}"
    assert "cleared:" not in pre
    assert not _stamp_lines(pre), "the fixture must be the pre-sweep bytes: it already dated"

    bodies_before = _tmp_sections(pre)
    assert len(bodies_before) == 2

    new, n, _ = SWEEP.date_note(pre, NO_PROGRAMS, DAY, BELOW)
    assert n == 2

    for body in bodies_before:
        assert body in new, "an alarm body was rewritten, not answered"
    for body in _tmp_sections(new):
        assert len([ln for ln in body.splitlines()
                    if ln.startswith(notify.DAILY_DATED_PREFIX)]) == 1, (
            "each of the two sections must carry its own stamp: " + body[-400:])

    assert len(SWEEP.HEADING.findall(new)) == len(SWEEP.HEADING.findall(pre)), (
        "the pass invented a section")
    assert "cleared:" not in new


# ── the CLI boundary: the script's own process, judged by the file it leaves ──


def _copy_fixture(tmp_path: Path) -> Path:
    """A copy of the real bytes in a throwaway notes directory, never the live note."""
    note = tmp_path / "2026-10-15.md"
    shutil.copyfile(FIXTURE, note)
    return note


def _run_cli(note_dir: Path, *argv: str, fraction: object = BELOW,
             frac_raises: bool = False) -> int:
    """`main` with both witnesses injected, so a test never steers the box.

    `running_reader` answers the supervisor the same way the module's docstring records for
    `runner=`: the gate may run where no supervisor answers, and a node that shelled out
    could only assert its refusal on some environments. It reports one RUNNING program,
    which refutes nothing in this fixture, so any block the CLI counts here can only have
    come from the fraction witness.
    """
    def _fraction(path):
        if frac_raises:
            raise RuntimeError("cannot read /tmp: OSError: statvfs refused")
        return fraction

    return SWEEP.main(["--notes", str(note_dir), *argv],
                      running_reader=lambda conf: {"lloyd-mc:lloyd-backend"},
                      fraction_reader=_fraction)


def test_the_cli_dates_a_fixture_copy_once_and_writes_it_once(tmp_path, capsys):
    """The whole tool over the real bytes: dry run writes nothing, apply writes once.

    This is the boundary the change actually crosses — a process opening a note, deciding
    from a reading, and rewriting the file — so the assertions are on the file's bytes:
    two stamps after the apply, both `/tmp: … inodes used (N%)` lines still there, no
    `cleared:` anywhere, and a second `--apply` leaving the file byte-identical.
    """
    note = _copy_fixture(tmp_path)
    before = note.read_bytes()

    rc = _run_cli(tmp_path, "--day", DAY)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert note.read_bytes() == before, "a dry run wrote the note"
    assert "refutable blocks: 2" in out, out
    assert f"{BELOW:.4f}" in out and f"{TW.CLEAR_FRACTION:.2f}" in out, (
        "the report must carry the witness it acted on: " + out)

    rc = _run_cli(tmp_path, "--apply", "--day", DAY)
    capsys.readouterr()
    assert rc == 0
    after = note.read_text(encoding="utf-8")
    assert after != _pre()
    assert len(_stamp_lines(after)) == 2, after[-600:]
    assert "/tmp: 845,813 of 1,048,576 inodes used (81%)" in after
    assert "/tmp: 858,283 of 1,048,576 inodes used (82%)" in after
    assert "cleared:" not in after

    rc = _run_cli(tmp_path, "--apply", "--day", DAY)
    capsys.readouterr()
    assert rc == 0
    assert note.read_text(encoding="utf-8") == after, "the second apply rewrote the note"


def test_the_cli_reports_a_reading_at_or_over_the_line_instead_of_dating_the_body(tmp_path,
                                                                                 capsys):
    """Above the line the tool says 0, and the number it says 0 from is the reading.

    The bodies quote 81%. A sweep that dated from the quoted figure would report 2 here;
    this reports 0 and leaves the file alone. The report names the reading it used, because
    a bare "0" from a pass over seven stranded blocks is the same output as a pass that
    found none — the shape of the `read_running` refusal the module already documents.
    """
    note = _copy_fixture(tmp_path)
    before = note.read_bytes()

    rc = _run_cli(tmp_path, "--apply", "--day", DAY, fraction=ABOVE)
    out = capsys.readouterr().out
    assert rc == 0
    assert note.read_bytes() == before, "the note was rewritten on a reading that refutes nothing"
    assert "blocks: 0" in out, out
    assert f"{ABOVE:.4f}" in out, "the report must name the reading, not just the count: " + out


def test_the_cli_refuses_a_block_whose_fraction_cannot_be_read(tmp_path, capsys):
    """A failed witness is exit 2 and zero bytes, never an untouched note reported clean.

    `main`'s documented answer to a witness it could not read: say so on stderr, exit 2,
    write nothing. Exit 0 with "0 blocks" would read as a corpus with nothing stranded,
    which is the failure mode of every counting guard here — a run that could not measure
    and a run that measured nothing look identical unless the exit code says which.
    """
    note = _copy_fixture(tmp_path)
    before = note.read_bytes()

    rc = _run_cli(tmp_path, "--apply", "--day", DAY, frac_raises=True)
    captured = capsys.readouterr()
    assert rc == 2, (rc, captured.err)
    assert note.read_bytes() == before
    assert "tmpwatch" in captured.err.lower() or "fraction" in captured.err.lower(), captured.err
    assert "dated:" not in note.read_text(encoding="utf-8")


# ── clause 3's share of the new file: the two refuters coexist, the old line keeps its words ──


def test_a_note_holding_both_families_gets_one_stamp_per_witness_from_each_refuter(tmp_path):
    """Adding a second refuter may not disturb the first, in a block the first does match.

    The supervisor half's own nodes (`tests/test_guardian_alert_retraction.py`) pass
    unmodified and are clause 3's evidence; this node pins the interaction only they cannot
    see — a note carrying BOTH families, where the dispatcher has to give each block its
    own witness and neither stamp may overwrite the other. It also pins the two phrases of
    `REFUTED_LINE` that carry its honesty claim ("`supervisorctl status` reports … RUNNING"
    as the refutation, and the sentence saying nobody watched the recovery), so a reword of
    the old line cannot quietly drop the weaker-claim wording while its own nodes, which
    assert only `RUNNING` and the program name, stay green.
    """
    down_body = ("lloyd-mc:lloyd-backend: FATAL: can't find command "
                 "'/home/alansrobotlab/lloyd/.venvs/lloyd/bin/python'\n\n"
                 "HEAD is 8328347d and no self-modification is being observed, so this is "
                 "infrastructure rather than a bad change. Not rewriting history — this "
                 "needs a human.\n")
    pre = _pre()
    both = ("---\nsegment: memory\n---\n"
            "## Self-mod guardian: Service down, but no promotion to revert\n\n"
            + down_body
            + "\n"
            + pre[pre.index("## Self-mod guardian:"):])

    new, n, notes = SWEEP.date_note(both, {"lloyd-mc:lloyd-backend"}, DAY, BELOW)
    assert n == 3, notes
    assert down_body in new, "the supervisor half stopped leaving the body alone"

    stamps = _stamp_lines(new)
    assert len(stamps) == 3, stamps
    sup = [s for s in stamps if "supervisorctl" in s]
    tmp = [s for s in stamps if "tmpwatch" in s]
    assert len(sup) == 1 and len(tmp) == 2, stamps
    assert "`supervisorctl status` reports `lloyd-mc:lloyd-backend` RUNNING" in sup[0], sup[0]
    assert "predates the retraction #2221 gives its alert family" in sup[0], sup[0]
    assert "cleared:" not in new
