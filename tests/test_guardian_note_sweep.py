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

Blindness, not age, is what makes this worth a second refuter. `guardian.py:2779` returns
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
  for the same reason, and `read_running`'s docstring at
  `guardian_note_sweep.py:500-506` records #2221's two clause-5 nodes coming back red at
  `cb235020` for exactly this (re-pointed by that paragraph's own words, not by the range
  this line carried: it said `206-215` and named the MODULE docstring, which was already
  wrong at #3502's base `a4473fda` — the passage was at :322 in `read_running` there, and
  `tripwire_stamp`/`read_tripwire` moved it to :501 here);
* the stamp is a `dated:` and never a `cleared:`, and neither marker word appears in its
  prose, because the corpus instrument for #2221 is `grep -c 'cleared:'` over the dated
  notes (#1988: an instrument must never read its own description).

The process boundary is the note file: the sweep runs as its own process and its output is
read later by a human or a signal pass, so every node judges re-read bytes — text from the
committed fixture, and file bytes for the CLI nodes. `tests/test_guardian_alert_retraction.py
:449-651` owns the supervisorctl half and is unmodified by this diff; `REFUTED_LINE`'s bytes
are unchanged, and the node at the end of this file pins the two phrases that carry its
honesty claim so a reword there cannot pass silently.

The last section of this file adds the THIRD refuter (#3502), and it lives here rather than
in a new file because a second file testing "is this block refuted" is the two-definitions
defect this docstring keeps warning about. Its witness is `guardian.tripwire_is_clear` — the
guardian's own condition, shared rather than re-derived — over the two titles
`VAULT_TRIP_ALERT_TITLE` and `DATA_TRIP_ALERT_TITLE`, and its subject is the stranded
tripwire block at `memory/2026-10-20.md:17`, which triage on 2026-10-27 measured at
`count_refutations` = 0 with 0 `dated:` and 0 `cleared:` markers and which a hand-written
correction on 2026-10-26 had already had to close once by hand.
"""
from __future__ import annotations

import difflib
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

import datawatch  # noqa: E402  (DataWatch / ROOT_MARKER / snapshot_gate — the third witness)
import guardian as Gmod  # noqa: E402  (tripwire_is_clear and the two tripwire alert titles)
import notify  # noqa: E402
import tmpwatch as TW  # noqa: E402
import vaultwatch  # noqa: E402  (VaultWatch / measure / sync_gate — the other tripwire watch)

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

    The bodies are snapshotted from the fixture's own section slices before the pass and every
    non-blank line of them required back, in order and byte-identical, after it — so "bodies
    intact" is asserted against the region the sweep appends into rather than against a
    hand-typed string. Required as lines and not as one contiguous slice because since #3993 a
    stamp belongs INSIDE the span, above the `---` rule and the `### Session` digest the span
    swallows, so the slice is no longer contiguous with itself.
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
        # Not `body in new`: #3993 moved the stamp INSIDE the span, above the `---` rule and the
        # `### Session` digest this note's spans swallow, so a stamp now sits between an alarm's
        # last line and the swallowed tail that `body` also contains. What must survive is the
        # alarm's text, and the check is that its non-blank lines still occur in `new` in the
        # same order — byte-identical, none dropped — rather than the slice being contiguous.
        keep = [ln for ln in body.splitlines() if ln.strip()]
        after = iter(new.splitlines())
        missing = [ln for ln in keep if not any(ln == nxt for nxt in after)]
        assert not missing, f"an alarm body was rewritten, not answered: {missing[:3]}"
    for body in _tmp_sections(new):
        stamps = [ln for ln in body.splitlines()
                  if ln.startswith(notify.DAILY_DATED_PREFIX)]
        assert len(stamps) == 1, (
            "each of the two sections must carry its own stamp: " + body[-400:])
        tail = SWEEP.SPAN_TAIL.search(body, body.index("\n") + 1)
        assert tail is None or body.index(stamps[0]) < tail.start(), (
            "on the note the item names as misplaced, the stamp must land above the rule and "
            "digest the span swallowed, not below them: " + body[:120])

    assert len(SWEEP.HEADING.findall(new)) == len(SWEEP.HEADING.findall(pre)), (
        "the pass invented a section")
    assert "cleared:" not in new


# ── the CLI boundary: the script's own process, judged by the file it leaves ──


def _copy_fixture(tmp_path: Path) -> Path:
    """A copy of the real bytes in a throwaway notes directory, never the live note."""
    note = tmp_path / "2026-10-15.md"
    shutil.copyfile(FIXTURE, note)
    return note


def _clean_tripwire() -> dict:
    """Both tripwire families read clear over roots no test wrote, so nothing is dated.

    The nodes below that are about `/tmp` or the supervisor need the third witness to be
    *silent* rather than absent: `--apply` exits 2 when a family reads damaged, and a node
    about the fraction witness that also got a `data tripwire … un-dateable` refusal on
    stderr would pass for the wrong reason. These readings name roots under /test-root, so
    a node that asserted one of them was stamped would be asserting a root that does not
    exist. `tripwire_watches` and `measure` are never called here, which is the point of
    `tripwire_reader=`: a CLI node cannot walk 705,803 files to prove a parser.
    """
    return {fam: SWEEP.TripwireReading(
                family=fam, root=f"/test-root/{fam}", total=1000, clear=True,
                why=f"{fam} measures whole against this watch's baseline",
                halt_path="/test-state/promotions-halted", halted=False)
            for fam in ("vault", "data")}


def _run_cli(note_dir: Path, *argv: str, fraction: object = BELOW,
             frac_raises: bool = False, tripwire: object = "clean",
             trip_raises: bool = False, snapshot: object = None,
             snap_raises: bool = False) -> int:
    """`main` with all three witnesses injected, so a test never steers the box.

    `running_reader` answers the supervisor the same way the module's docstring records for
    `runner=`: the gate may run where no supervisor answers, and a node that shelled out
    could only assert its refusal on some environments. It reports one RUNNING program,
    which refutes nothing in this fixture, so any block the CLI counts here can only have
    come from the fraction, tripwire or snapshot witness. `snapshot_reader` defaults to
    None — no reading — so the nodes written before #3882 see exactly the pass they saw
    before: an absent witness dates nothing and is reported as absent.
    """
    def _fraction(path):
        if frac_raises:
            raise RuntimeError("cannot read /tmp: OSError: statvfs refused")
        return fraction

    def _tripwire():
        if trip_raises:
            raise RuntimeError("tripwire: cannot read the watch state")
        return _clean_tripwire() if tripwire == "clean" else tripwire

    def _snapshot():
        if snap_raises:
            raise OSError("git: blocked by the sandbox")
        return snapshot

    return SWEEP.main(["--notes", str(note_dir), *argv],
                      running_reader=lambda conf: {"lloyd-mc:lloyd-backend"},
                      fraction_reader=_fraction, tripwire_reader=_tripwire,
                      snapshot_reader=_snapshot)


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


# ═══ the third witness: the guardian's own tripwire clear condition (#3502) ═════════
#
# A tripwire block strands differently from the two families above. When the vault or
# data-root tripwire fires while another alert already holds ALERT.md's single slot,
# `Guardian._retract_tripwire_alert` cannot reach the stale section at all — its own
# docstring says a displaced title leaves the process with "no index to the stale section,
# which is the sweep's subject, not this one's" — and the sweep had no witness that could
# read a file count. `memory/2026-10-20.md` is that case: triage on 2026-10-27 measured
# `count_refutations` = 0 against it while its block carried 0 `dated:` and 0 `cleared:`
# markers, and closing the directive took a hand-written nightly correction on 2026-10-26.
#
# So the criterion here is NOT a new one. It is `guardian.tripwire_is_clear`, the function
# the guardian's `_vault_tripwire_is_clear` / `_data_tripwire_is_clear` call, lifted to
# module level for exactly this second caller. Two consequences carry the weight, and both
# are pinned below: the sweep inherits that function's refusal of a watch with no baseline,
# which `vaultwatch.sync_gate` and `datawatch.snapshot_gate` do NOT refuse (both skip the
# comparison when `w.history` is empty and answer True); and the walk happens once per root
# per pass, because `~/lloyd-data` is 705,803 files as measured on 2026-11-02 and a node
# that let the default run would spend its budget walking a tree it does not own.

#: The real bytes of `~/obsidian/memory/2026-10-20.md` as they stood at triage: one
#: unsealed `Self-mod guardian: Lloyd data root damaged — promotions halted` section at :17
#: with 0 `dated:` and 0 `cleared:` markers anywhere in the file. Committed for the same
#: reason as the 10-15 fixture — the gate runs with `$HOME` at the round home, where the
#: live note is not there to open.
FIXTURE_2020 = Path(__file__).resolve().parent / "fixtures" / "guardian_note_2026-10-20_pre-sweep.md"

DATA_TRIP_TITLE = f"{SWEEP.GUARDIAN_TITLE} {Gmod.DATA_TRIP_ALERT_TITLE}"
VAULT_TRIP_TITLE = f"{SWEEP.GUARDIAN_TITLE} {Gmod.VAULT_TRIP_ALERT_TITLE}"
TMP_TITLE = f"{SWEEP.GUARDIAN_TITLE} {TW.ALERT_TITLE}"
STATE_TITLE = f"{SWEEP.GUARDIAN_TITLE} Service down, but no promotion to revert"
ROLLBACK_TITLE = f"{SWEEP.GUARDIAN_TITLE} Rolled back 1a2b3c4d → 5e6f7a8b"

DATA_TRIP_BODY = (
    "Lloyd data root damaged — promotions halted\n\n"
    "top-level folder 'triage-3087' (3347 files) vanished\n\n"
    "Files: 658273 → 654927\n\n"
    "Promotions halted and the worker is paused. To recover: restore the newest snapshot "
    "into the data root, then run `python3 agent-services/guardian/datawatch.py clear`.\n")
VAULT_TRIP_BODY = (
    "Vault mass deletion — sync stopped\n\n"
    "top-level folder 'knowledge' (3300 files) vanished\n\n"
    "Stop `ob sync` until a human restores the tree, then run "
    "`python3 agent-services/guardian/vaultwatch.py clear`.\n")
TMP_BODY = "/tmp: 845,813 of 1,048,576 inodes used (81%)\n"
STATE_BODY = "lloyd-mc:lloyd-backend: FATAL: can't find command\n"
ROLLBACK_BODY = ("The promotion was reverted and the last known good commit is running "
                 "again.\n")


def _trip_note(*pairs) -> str:
    """A daily note holding the sections named, in order, front matter first."""
    body = "".join(f"## {title}\n\n{text}\n\n" for title, text in pairs)
    return "---\ntype: note\ntimestamp: '2026-10-20T13:52:49'\n---\n\n" + body


def _tripwire_all_families() -> str:
    """One block per alarm family the sweep can be asked about, all five unsealed.

    Why the note is this shape: with the fraction witness at or above its clear line and no
    RUNNING program, the tripwire witness is the ONLY one that can act — so a node that
    asserts "the other three titles came back False" is testing the title binding and not
    the emptiness of the note.
    """
    return _trip_note((DATA_TRIP_TITLE, DATA_TRIP_BODY),
                      (VAULT_TRIP_TITLE, VAULT_TRIP_BODY),
                      (TMP_TITLE, TMP_BODY),
                      (STATE_TITLE, STATE_BODY),
                      (ROLLBACK_TITLE, ROLLBACK_BODY))


def _tripwire_home(tmp_path, *, files=40, state="gstate"):
    """Both tripwire watches over two real trees, each with a real baseline recorded.

    The shape `tests/test_guardian_predicates.py:4800` `_tripwire_watch` builds, both
    families at once: `files` files inside one top-level folder so the folder guard
    (`vaultwatch.TOPDIR_MIN_FILES` = 20) stays in play, and the data root carries its
    `datawatch.ROOT_MARKER` because `DataWatch.judge` refuses a root that lost it. 40
    rather than that helper's 1000 because these nodes run the walk more than once and the
    claim under test is the comparison, not the size. Nothing is a stand-in: the watches,
    their baselines, their state files and their markers are the shipped objects, since the
    reads under test are `tripped()`, `measure()` and the watch's own judge.
    """
    gdir = tmp_path / state
    gdir.mkdir(parents=True, exist_ok=True)
    out = {"gdir": gdir}
    watches = {}
    for family in ("vault", "data"):
        root = tmp_path / family
        (root / "notes").mkdir(parents=True, exist_ok=True)
        for i in range(files):
            (root / "notes" / f"n{i}.md").write_text("x\n", encoding="utf-8")
        if family == "data":
            (root / datawatch.ROOT_MARKER).write_text("", encoding="utf-8")
            watch = datawatch.DataWatch(str(root), gdir)
        else:
            watch = vaultwatch.VaultWatch(str(root), gdir)
        why, _ = watch.tick()
        assert why is None and watch.history, f"{family} watch recorded no baseline"
        watches[family] = watch
        out[f"{family}_root"] = root
    out["watches"] = watches
    return out


def _clear_readings(home, *, halted=False, halt_path=None):
    """The sweep's reading of a home whose two roots measure whole: both families clear."""
    return SWEEP.read_tripwire(watches=home["watches"],
                               halt_reader=lambda: (halt_path or "/s/promotions-halted",
                                                    halted))


def _deletions(pre: str, post: str) -> int:
    """Lines the rewrite removed, counted the way `git diff --numstat` counts them.

    Clause 5 of the item is `git -C ~/obsidian diff --numstat` reporting 0 deletions, so the
    test measures the same quantity rather than a looser "the body is still in there":
    `difflib`'s `replace` opcodes carry a deletion in git's eyes too, and an alarm body that
    was reworded rather than deleted is exactly what this instrument exists to catch.
    """
    pre_lines, post_lines = pre.splitlines(), post.splitlines()
    return sum(i2 - i1 for op, i1, i2, _j1, _j2
               in difflib.SequenceMatcher(None, pre_lines, post_lines).get_opcodes()
               if op in ("delete", "replace"))


def test_a_clear_tripwire_reading_dates_both_tripwire_blocks_and_no_other_title():
    """Clause 1: the witness is bound to two titles, not to the fact that a tree is whole.

    Five unsealed guardian sections in, two stamps out. Both tripwire blocks are dateable
    under the injected reading; the `/tmp` block, the supervisor-state block and the
    `Rolled back` notice are not, under the SAME reading. That asymmetry is the whole safety
    property — a witness that read "some root measures whole" as a licence over every
    stranded block would date an inode alarm from a vault file count, which is #3277's
    binding defect widened.
    """
    readings = {"data": SWEEP.TripwireReading(
                    family="data", root="/home/alansrobotlab/lloyd-data", total=705803,
                    clear=True, why="measures whole", halt_path="/s/halted", halted=False),
                "vault": SWEEP.TripwireReading(
                    family="vault", root="/home/alansrobotlab/obsidian", total=9511,
                    clear=True, why="measures whole", halt_path="/s/halted", halted=False)}
    note = _tripwire_all_families()
    bodies = {title: text for title, text in
              ((DATA_TRIP_TITLE, DATA_TRIP_BODY), (VAULT_TRIP_TITLE, VAULT_TRIP_BODY),
               (TMP_TITLE, TMP_BODY), (STATE_TITLE, STATE_BODY),
               (ROLLBACK_TITLE, ROLLBACK_BODY))}

    for title in (DATA_TRIP_TITLE, VAULT_TRIP_TITLE):
        assert SWEEP.block_is_refuted(title, bodies[title], running=NO_PROGRAMS,
                                      fraction=ABOVE, tripwire=readings), title
    for title in (TMP_TITLE, STATE_TITLE, ROLLBACK_TITLE):
        assert SWEEP.block_is_refuted(title, bodies[title], running=NO_PROGRAMS,
                                      fraction=ABOVE, tripwire=readings) is False, (
            f"{title} was dated on a reading of somebody else's root")

    assert SWEEP.count_refutations(note, NO_PROGRAMS, ABOVE, readings) == 2
    new, n, notes = SWEEP.date_note(note, NO_PROGRAMS, DAY, ABOVE, readings)
    assert n == 2, notes
    assert _deletions(note, new) == 0, "an alarm body was rewritten, not answered"
    assert "cleared:" not in new


def test_a_missing_tripwire_reading_dates_no_tripwire_block():
    """The third witness is only silent when absent, never silent-and-clear.

    `tripwire=None` and `tripwire={}` are the two ways a pass reaches a tripwire block with
    no reading — a caller that never built one, and a reader that returned nothing. Both
    must leave the block alone. A `clear=True` default on either would date every tripwire
    block in the vault from the absence of a measurement, which is the same `fraction=None`
    trap the second witness documents one function above.
    """
    note = _trip_note((DATA_TRIP_TITLE, DATA_TRIP_BODY))
    for missing in (None, {}):
        assert SWEEP.count_refutations(note, NO_PROGRAMS, ABOVE, missing) == 0
        assert SWEEP.date_note(note, NO_PROGRAMS, DAY, ABOVE, missing)[1] == 0
        assert SWEEP.date_note(note, NO_PROGRAMS, DAY, ABOVE, missing)[0] == note


def test_the_witness_refuses_a_watch_with_no_baseline_that_both_gates_accept(tmp_path):
    """Clause 2: the sweep asks the guardian, and inherits a refusal the gates do not make.

    `vaultwatch.sync_gate` (`vaultwatch.py:571`) and `datawatch.snapshot_gate`
    (`datawatch.py:1478`) both answer True when the watch holds no baseline — they skip the
    comparison, because "may I sync / may I snapshot" is a question about the marker and the
    tree, not about a history they may not have. The guardian's condition refuses that state
    as "a missing comparison rather than a favourable one". A sweep bound to the gates would
    date `memory/2026-10-20.md` on a box whose guardian state directory had just been wiped:
    no baseline, no evidence, one retraction.
    """
    home = _tripwire_home(tmp_path)
    cold = {"vault": vaultwatch.VaultWatch(str(home["vault_root"]), home["gdir"] / "cold"),
            "data": datawatch.DataWatch(str(home["data_root"]), home["gdir"] / "cold")}
    for watch in cold.values():
        assert not watch.history, "the cold watch has a baseline; the case is not built"

    assert vaultwatch.sync_gate(root=str(home["vault_root"]),
                                state_dir=home["gdir"] / "cold")[0] is True
    assert datawatch.snapshot_gate(root=str(home["data_root"]),
                                   state_dir=home["gdir"] / "cold")[0] is True
    for family, watch in cold.items():
        clear, why = Gmod.tripwire_is_clear(watch)
        assert clear is False, f"{family}: the shared condition accepted a cold watch: {why}"

    readings = SWEEP.read_tripwire(watches=cold)
    assert all(r.clear is False for r in readings.values())
    note = _tripwire_all_families()
    assert SWEEP.count_refutations(note, NO_PROGRAMS, ABOVE, readings) == 0
    assert SWEEP.date_note(note, NO_PROGRAMS, DAY, ABOVE, readings)[0] == note


def test_the_two_guardian_wrappers_and_the_sweep_answer_from_one_function(tmp_path,
                                                                         monkeypatch):
    """Clause 2's other half: one decision, three callers, no forwarding copy to drift.

    A recorder on the module-level function proves the guardian's own two wrappers and the
    sweep's reader all reach it. The bound method the guardian used to hold is gone rather
    than left behind as a shim: a second definition of "the watch says the tree is whole" is
    the failure the sweep's module docstring names, and a shim is how one starts disagreeing
    with the wire that raised the alarm.
    """
    home = _tripwire_home(tmp_path)
    seen = []
    monkeypatch.setattr(Gmod, "tripwire_is_clear",
                        lambda watch, snap=None: (seen.append(str(watch.root))
                                                  or (True, "recorded")))

    guardian = Gmod.Guardian.__new__(Gmod.Guardian)
    guardian.vault, guardian.data = home["watches"]["vault"], home["watches"]["data"]
    assert guardian._vault_tripwire_is_clear() == (True, "recorded")
    assert guardian._data_tripwire_is_clear() == (True, "recorded")
    SWEEP.read_tripwire(watches=home["watches"], measure=lambda root, **_kw: None)
    assert len(seen) == 4, seen                       # 2 wrappers + 2 families, no extra home
    assert not hasattr(Gmod.Guardian, "_tripwire_is_clear"), (
        "the clear test must live in one place, not in a method that forwards to it")


def test_the_stamp_names_the_flag_file_the_halt_state_is_read_from(tmp_path):
    """The halt measurement is `gstate.AutomodState`'s own flag, not a lookalike path.

    The clause asks the witness to name `AutomodState`'s halt flag as a second measurement.
    `is_halted()` is `self.halted.exists()`, so a reader cannot tell a clean "not set" from a
    flag it could not stat without knowing WHICH file it read — which is why
    `promotion_halt` returns the path next to the answer and the stamp prints it. Writing the
    file is the only way to know the path is the one the guardian reads.
    """
    gdir = tmp_path / "automod-state"
    gdir.mkdir()
    path, halted = SWEEP.promotion_halt(gdir=gdir)
    assert path.endswith("promotions-halted"), path
    assert halted is False
    Path(path).write_text("2026-10-20T13:22Z halted\n", encoding="utf-8")
    assert SWEEP.promotion_halt(gdir=gdir) == (path, True)


@pytest.mark.parametrize("broken_read", [
    "still damaged", "root gone", "marker unparseable", "reader raised"])
def test_a_tripwire_reading_that_cannot_be_had_dates_nothing_and_exits_2(tmp_path, capsys,
                                                                         broken_read):
    """Clause 3: four ways the witness has no answer, and none of them touches the bytes.

    The first three are read through the shipped watches, not a mock, because each is a
    different producer failure with the same consequence for a note: the tree still fails
    the watch's own judge; `measure` answers None for a root it cannot stat; the marker
    exists but does not parse, which `tripped()` reports as a set marker rather than an
    absent one. The fourth is the seam itself raising. All four leave the block byte-for-byte
    as found and make the CLI say which family it could not date and exit 2 — the same
    refusal the other two witnesses make, because exit 0 with "0 blocks" from a pass that
    could not measure is indistinguishable from a corpus with nothing stranded.
    """
    home = _tripwire_home(tmp_path)
    readings = _clear_readings(home)
    assert all(r.clear for r in readings.values()), "the fixture read damaged before it broke"

    if broken_read == "still damaged":
        for f in (home["data_root"] / "notes").glob("*.md"):
            f.unlink()
        readings = SWEEP.read_tripwire(watches=home["watches"])
        assert readings["data"].clear is False, readings["data"].why
    elif broken_read == "root gone":
        shutil.rmtree(home["data_root"])
        readings = SWEEP.read_tripwire(watches=home["watches"])
        assert readings["data"].total is None, readings["data"].why
        assert readings["data"].clear is False
    elif broken_read == "marker unparseable":
        (home["gdir"] / "data-tripped.json").write_text("{not json", encoding="utf-8")
        readings = SWEEP.read_tripwire(watches=home["watches"])
        assert readings["data"].clear is False, readings["data"].why

    note = tmp_path / "notes" / "2026-10-20.md"
    note.parent.mkdir()
    shutil.copyfile(FIXTURE_2020, note)
    before = note.read_bytes()

    rc = _run_cli(note.parent, "--apply", "--day", DAY, fraction=ABOVE,
                  tripwire=(None if broken_read == "reader raised" else readings),
                  trip_raises=(broken_read == "reader raised"))
    captured = capsys.readouterr()
    assert rc == 2, (rc, captured.err)
    assert note.read_bytes() == before, f"{broken_read} wrote to the note"
    assert _stamp_lines(note.read_text(encoding="utf-8")) == []
    assert "tripwire" in captured.err.lower(), captured.err
    if broken_read != "reader raised":
        assert "data" in captured.err.lower(), (
            "the refusal must name the family it could not date: " + captured.err)


def test_the_cli_exits_2_when_the_tripwire_witness_cannot_be_read_at_all(tmp_path, capsys):
    """A witness that raises is a refusal, and it stops the whole apply rather than one family.

    `read_tripwire` builds both watches before it reads either, so its failure is not a
    per-family read failure and the report cannot name a root. What it must not do is print
    "0 blocks" and exit 0: that is the shape of a clean corpus, and this pass measured
    nothing about the tripwire family.
    """
    note = tmp_path / "notes" / "2026-10-20.md"
    note.parent.mkdir()
    shutil.copyfile(FIXTURE_2020, note)
    before = note.read_bytes()

    rc = _run_cli(note.parent, "--apply", "--day", DAY, fraction=ABOVE, trip_raises=True)
    captured = capsys.readouterr()
    assert rc == 2, (rc, captured.err)
    assert note.read_bytes() == before
    assert "tripwire" in captured.err.lower(), captured.err
    assert "guardian.tripwire_is_clear" in captured.err, (
        "the refusal must name the witness it could not read: " + captured.err)


def test_the_tripwire_stamp_is_a_measurement_line_and_never_a_clearance(tmp_path):
    """Clause 4: what the stamp may say, in the words it must say them in.

    Every phrase checked is a claim the pass is entitled to make from the reading it took:
    the function that answered, the root it measured, the count it returned, whether
    `AutomodState`'s halt flag is set and at which path, and the read date. The disclaimer is
    checked because the alternative is a stamp that reads as lifting a pause the guardian
    never held — `tripwire_is_clear`'s own True sentence says the tree "measures whole, not
    that the worker pause or the promotion halt was lifted", and a dated: line that dropped
    that half would be a stronger claim than its witness.
    """
    home = _tripwire_home(tmp_path)
    readings = _clear_readings(home, halt_path="/home/x/.state/lloyd-automod/promotions-halted")
    note = _trip_note((DATA_TRIP_TITLE, DATA_TRIP_BODY))

    new, n, notes = SWEEP.date_note(note, NO_PROGRAMS, DAY, ABOVE, readings)
    assert n == 1, notes
    stamps = _stamp_lines(new)
    assert len(stamps) == 1, stamps
    line = stamps[0]
    assert line.startswith(f"{notify.DAILY_DATED_PREFIX} {DAY}: "), line
    total = vaultwatch.measure(str(home["data_root"]),
                               skip_top=home["watches"]["data"].skip_top).total
    assert readings["data"].total == total, "the stamp's count is not the walk's"
    for phrase in ("`guardian.tripwire_is_clear`", str(home["data_root"]),
                   f"{total:,} files",
                   "/home/x/.state/lloyd-automod/promotions-halted", "not set",
                   "does not assert that the worker pause or the promotion halt was lifted",
                   f"notify.DAILY_SCAN_DAYS = {notify.DAILY_SCAN_DAYS}",
                   "datawatch.py status"):
        assert phrase in line, f"the stamp is missing {phrase!r}: {line}"
    assert "cleared:" not in new
    assert DATA_TRIP_BODY in new, "the alarm body was edited rather than answered"
    assert _deletions(note, new) == 0


def test_a_halt_still_set_is_stamped_as_set_rather_than_refused(tmp_path):
    """The halt flag is a second MEASUREMENT in the stamp, never a third precondition.

    Requiring the flag to be absent before dating would make the sweep's clearance depend on
    an owner that has no timeout: `gstate.set_halted` is cleared only by whoever holds it, so
    a rule that waited on it would leave the block a standing instruction forever — the
    exact state this item exists to end. What the pass can say is what it read: the tree
    measures whole AND the flag is set, both dated, with the reader left to act on the pair.
    """
    home = _tripwire_home(tmp_path)
    readings = _clear_readings(home, halted=True)
    assert all(r.halted for r in readings.values())

    new, n, _notes = SWEEP.date_note(_tripwire_all_families(), NO_PROGRAMS, DAY, ABOVE,
                                     readings)
    assert n == 2
    for line in _stamp_lines(new):
        assert "`/s/promotions-halted` SET" in line, line
        assert "does not assert that the worker pause or the promotion halt was lifted" in line


def test_a_stamped_tripwire_block_is_never_stamped_twice(tmp_path):
    """Clause 4's other half: a second pass over a dated note writes nothing at all.

    The seal is `already_sealed`, the same marker-based test both other witnesses use, and
    it holds across a changed reading: the second pass here runs with a DIFFERENT count and
    a halt flag now set. If the seal were keyed on the reading rather than on the note, a
    re-run would append a second stamp contradicting the first, and the note would carry two
    "measurements" of the same day with no way to tell which the writer stood behind.
    """
    home = _tripwire_home(tmp_path)
    readings = _clear_readings(home)
    note = _trip_note((DATA_TRIP_TITLE, DATA_TRIP_BODY), (VAULT_TRIP_TITLE, VAULT_TRIP_BODY))
    once, n, _ = SWEEP.date_note(note, NO_PROGRAMS, DAY, ABOVE, readings)
    assert n == 2

    again = SWEEP.read_tripwire(
        watches=home["watches"],
        halt_reader=lambda: ("/s/promotions-halted", True))
    assert SWEEP.count_refutations(once, NO_PROGRAMS, ABOVE, again) == 0
    new, n2, notes2 = SWEEP.date_note(once, NO_PROGRAMS, DAY, ABOVE, again)
    assert n2 == 0 and new == once, notes2
    assert len(_stamp_lines(new)) == 2


def test_each_root_is_walked_once_per_pass_and_the_count_is_the_write(tmp_path):
    """Clause 5: three blocks, two roots, two walks — and one measurement behind the number.

    The count and the bytes must come from one reading: `count_refutations` is what the CLI
    prints as "refutable blocks" and `date_note` is what writes, so they ask the same
    `block_is_refuted` and the node asserts the printed figure is the number of stamps that
    landed. The walk count is the cost claim: `vaultwatch.measure` over `~/lloyd-data` is a
    705,803-file walk as measured on 2026-11-02, so a pass that measured per block would
    spend three of them on the three tripwire sections in this note.
    """
    home = _tripwire_home(tmp_path)
    walks = []

    def counting_measure(root, *, skip_top=frozenset()):
        walks.append(str(root))
        return vaultwatch.measure(root, skip_top=skip_top)

    readings = SWEEP.read_tripwire(watches=home["watches"], measure=counting_measure)
    assert all(r.clear for r in readings.values()), readings
    assert sorted(walks) == sorted({str(home["vault_root"]), str(home["data_root"])}), walks

    note = _trip_note((DATA_TRIP_TITLE, DATA_TRIP_BODY),
                      (VAULT_TRIP_TITLE, VAULT_TRIP_BODY),
                      (DATA_TRIP_TITLE + " (again)", DATA_TRIP_BODY),
                      (TMP_TITLE, TMP_BODY))
    assert walks == [str(home["vault_root"]), str(home["data_root"])], (
        "reading the blocks walked a root a second time")
    counted = SWEEP.count_refutations(note, NO_PROGRAMS, ABOVE, readings)
    new, n, _notes = SWEEP.date_note(note, NO_PROGRAMS, DAY, ABOVE, readings)
    assert counted == 3 == n, (counted, n)
    assert len(_stamp_lines(new)) == counted
    assert _deletions(note, new) == 0
    assert walks == [str(home["vault_root"]), str(home["data_root"])], (
        "counting and writing each walked the roots again")


def test_the_cli_takes_one_tripwire_reading_for_a_whole_pass(tmp_path, capsys):
    """The seam is per pass, like `running_reader=` and `fraction_reader=`, not per note.

    Two notes, both holding the same stranded block shape. The witness runs once: a
    `tripwire_reader=` called per note would walk both roots twice for a reading that cannot
    change inside a pass, and the CLI node is the only place that claim is checkable.
    """
    for name in ("2026-10-20.md", "2026-10-21.md"):
        shutil.copyfile(FIXTURE_2020, tmp_path / name)
    calls = []

    def reader():
        calls.append(1)
        return {"data": SWEEP.TripwireReading(
                    family="data", root=str(tmp_path / "lloyd-data"), total=705803,
                    clear=True, why="measures whole", halt_path="/s/halted", halted=False)}

    rc = SWEEP.main(["--notes", str(tmp_path), "--apply", "--day", DAY],
                    running_reader=lambda conf: {"lloyd-mc:lloyd-backend"},
                    fraction_reader=lambda path: ABOVE, tripwire_reader=reader)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert len(calls) == 1, calls
    assert "blocks: 2" in out, out
    for name in ("2026-10-20.md", "2026-10-21.md"):
        text = (tmp_path / name).read_text(encoding="utf-8")
        assert len(_stamp_lines(text)) == 1, name
        assert "cleared:" not in text


def test_the_first_real_refutation_is_the_note_the_item_names():
    """Clause 6's shape, over the committed bytes of `memory/2026-10-20.md`.

    The live `--apply` this item owes is owed because it needs a real reading of
    `~/lloyd-data` and a real marker state, neither of which the gate owns. What CAN be
    pinned here is the shape that pass must land in, on the real bytes: one `dated:` line in
    the tripwire section, `grep -c 'cleared:'` still 0, and 0 deleted lines — the alarm body
    stays verbatim, which is what the 2026-10-26 hand-written correction also chose to do.
    """
    pre = FIXTURE_2020.read_text(encoding="utf-8")
    assert "Self-mod guardian: Lloyd data root damaged — promotions halted" in pre
    assert _stamp_lines(pre) == [], "the fixture is post-sweep bytes, not the triage bytes"
    assert pre.count("cleared:") == 0

    readings = {"data": SWEEP.TripwireReading(
                    family="data", root="/home/alansrobotlab/lloyd-data", total=705803,
                    clear=True,
                    why="/home/alansrobotlab/lloyd-data measures 705803 files against this "
                        "watch's baseline and its data tripwire marker is absent",
                    halt_path="/home/x/.state/lloyd-automod/promotions-halted",
                    halted=False)}
    assert SWEEP.count_refutations(pre, NO_PROGRAMS, ABOVE, readings) == 1
    new, n, notes = SWEEP.date_note(pre, NO_PROGRAMS, DAY, ABOVE, readings)
    assert n == 1, notes
    stamps = _stamp_lines(new)
    assert len(stamps) == 1, stamps
    assert new.count("cleared:") == 0
    assert _deletions(pre, new) == 0
    assert SWEEP.count_refutations(new, NO_PROGRAMS, ABOVE, readings) == 0, (
        "the stamped block is still counted: the next --apply would stamp it again")

# ── clause 3 (#3502): nothing this lift moved is left pointing at the wrong line ──
#: A `guardian.py:<N>` citation is a claim about a file, and this lift moved the file:
#: −58 lines between the tripwire wrappers and `def count_kg_rows` (the method body left
#: the class), +15 from there down (`tripwire_is_clear`'s 73 lines arrived, the 58-line body
#: and its 2-line neighbours departed). The refused round shipped two cites it had itself
#: invalidated — `:431` for a function it then moved to :2892, and `:2764` for a line that
#: was blank at its own base — and a third, `scripts/automod/code_rules.py:390`, named a
#: line inside an unrelated dict. So the check is mechanical and tree-wide: EVERY line
#: citation into `agent-services/guardian/guardian.py`, in any tracked file, must land
#: inside some `def` of that file. Its reach is measured rather than assumed, and it is
#: narrower than it looks — two mutations of one real cite set the boundary. Moving
#: `code_rules.py`'s witness cite to `:100`, a header comment outside every definition, FAILS
#: this node with the cited line's own text; moving the same cite to `:191`, a comment inside
#: `rollback_title`'s body, PASSES. So it refuses a cite that escaped every function and class
#: entirely — a module constant, the header prose, past the last line — and does not refuse a
#: cite inside the WRONG definition. None of the three stale cites this round repaired would it
#: have caught: `code_rules.py:390`'s `:2897` sat in `reconcile_stranded_alert`'s
#: `conditions = {`, `code_rules.py:326`'s `:1058` on a blank line inside `drain_logs`, and the
#: sweep's `:2764` at base was a blank line inside `_retract_tripwire_alert`. Those fell to
#: reading each sentence against the file, which is what the review still does; this node is the
#: floor under the first class, never a substitute for that. Two-digit numbers are outside the
#: scan's `[0-9]{3,4}`, and no citation into a 3,100-line file has ever meant line 42.
GUARDIAN_SRC = ROOT / "agent-services" / "guardian" / "guardian.py"


def test_every_line_citation_into_guardian_py_lands_inside_a_definition():
    import re as _re
    import subprocess as _sp
    lines = GUARDIAN_SRC.read_text().split("\n")

    def head_at(n: int) -> str:
        for i in range(n, 0, -1):
            if _re.match(r"^(    def |def )", lines[i - 1]):
                return lines[i - 1].strip()
        return ""

    found = _sp.run(["git", "-C", str(ROOT), "grep", "-nE",
                     r"guardian\.py:[0-9]{3,4}", "--", "*.py", "*.md"],
                    capture_output=True, text=True, check=True).stdout
    checked = 0
    for where in found.splitlines():
        hit = _re.search(r"guardian\.py:(\d{3,4})", where)
        if hit is None:
            continue
        n = int(hit.group(1))
        assert n <= len(lines), (
            f"{where[:120]} cites guardian.py:{n}, past the end of a {len(lines)}-line "
            "file — the cite is not about this file any more")
        assert head_at(n), (
            f"{where[:120]} cites guardian.py:{n}, and that line is inside no definition "
            f"(it reads {lines[n - 1].strip()[:70]!r}). #3502 moved lines here: −58 between "
            "the tripwire wrappers and `def count_kg_rows`, +15 below it. Re-point the cite "
            "by the text of the line it means, then re-run this node.")
        checked += 1
    assert checked >= 40, (
        f"only {checked} citations checked (52 existed on 2026-11-03) — a scan that found "
        "almost nothing is a coverage gap reading itself as a clean result, which is the "
        "failure mode this file exists to refuse")


# ── the FOURTH refuter: the vault snapshot's own age reading (#3882) ──────────
#
# The three sections at `~/obsidian/memory/2026-11-04.md:26,:39,:52` are the fourth
# family and the one no witness reached: `vaultwatch.snapshot_age_state` is both the
# reading `Guardian._vault_snapshot_stream` raises "Vault snapshots are not arriving"
# from and the only thing in the tree that can say whether a snapshot alarm still holds.
# Two things made those blocks unreachable, and the second is why a live `resolve`
# branch could not have done this alone: `notify.DAILY_SCAN_DAYS = 3` puts 11-04 outside
# every scan on the day the item was filed, and the leg had no now-healthy branch at all
# — so a dating, whose witness is a fresh reading, is the only retraction that can reach
# a block this old. Like the tripwire refuter, it DATES rather than clears: an observed
# fresh commit today says the alarm is no longer true, not that a human verified the
# three days between.
#
# The reading is injected for the standing reason in this file's docstring: the gate
# cannot steer a real `git log` of the live off-vault repository to a chosen HEAD age.

SNAP_TITLE = f"{SWEEP.GUARDIAN_TITLE} {Gmod.VAULT_SNAPSHOT_ALERT_TITLE}"

#: The body as `memory/2026-11-04.md` carries it, reduced to the two sentences that
#: matter: the measurement the alarm was built from, and the directive a reader obeys.
SNAP_BODY = (
    "/home/alansrobotlab/.local/state/lloyd-vault-backup/vault.git commit a4521e90 is "
    "17 h 40 m old, past the 3 h limit.\n\n"
    "Restore into a side directory with "
    "`~/lloyd/scripts/backup/restore-vault.sh <rev> <dest>` — never over the live vault.\n")

SNAP_REPO = "/home/alansrobotlab/.local/state/lloyd-vault-backup/vault.git"
SNAP_SHA = "14f47bfd" + "0" * 32


def _snap_reading(fresh=True, owned=True, sha=SNAP_SHA) -> SWEEP.SnapshotReading:
    """A snapshot reading, in the shape `read_snapshot` maps `snapshot_age_state` into."""
    return SWEEP.SnapshotReading(
        repo=SNAP_REPO, sha=sha, fresh=fresh, owned=owned, max_age=3 * 3600.0,
        text=(f"{SNAP_REPO} commit {sha[:8]} is 3 m old" if fresh else
              f"{SNAP_REPO} commit {sha[:8]} is 17 h 40 m old, past the 3 h limit"))


def _snap_note(with_state: bool = True) -> str:
    """One unsealed snapshot block, beside blocks of the three families it must not touch.

    The `name: STATE` block is in the string fixture because the direct-`date_note` nodes
    hand the pass no supervisor at all, so nothing refutes it and it earns no stamp. A CLI
    node asks for the note WITHOUT it (`with_state=False`): `_run_cli` injects a standing
    RUNNING set — the gate may run where no supervisor answers, and a node that shelled out
    could only assert its refusal on some machines — and that set refutes a `name: STATE`
    block on every pass by design, which would make a one-stamp count a count of the helper
    rather than of the witness under test.
    """
    blocks = [(SNAP_TITLE, SNAP_BODY),
              (VAULT_TRIP_TITLE, VAULT_TRIP_BODY),
              (TMP_TITLE, TMP_BODY),
              (ROLLBACK_TITLE, ROLLBACK_BODY)]
    if with_state:
        blocks.insert(3, (STATE_TITLE, STATE_BODY))
    return _trip_note(*blocks)


def test_a_fresh_snapshot_reading_dates_the_block_and_no_other_title():
    """Clause 4, first half: fresh AND owned refutes, and only this title is refuted.

    The title binding is the clause. `snapshot_age_state` measures the vault's off-vault
    backup repository, so a fresh commit there is an answer about
    `VAULT_SNAPSHOT_ALERT_TITLE` and about nothing else; without the binding the sweep
    would date a disk alarm because a git repository somewhere is current.
    """
    text = _snap_note()
    reading = _snap_reading()
    out, n, notes = SWEEP.date_note(text, NO_PROGRAMS, DAY, snapshot=reading)
    assert n == 1, notes
    assert len(_stamp_lines(out)) == 1, out
    assert out.index(notify.DAILY_DATED_PREFIX) < out.index(VAULT_TRIP_TITLE), (
        "the one stamp landed somewhere other than in the snapshot block")
    for other in (VAULT_TRIP_TITLE, TMP_TITLE, STATE_TITLE, ROLLBACK_TITLE):
        start = out.index(other)
        assert out[start:out.index("\n\n", start)].count(notify.DAILY_DATED_PREFIX) == 0, (
            f"the snapshot witness dated a block of another family: {other}")
    assert SWEEP.count_refutations(text, NO_PROGRAMS, snapshot=reading) == 1
    assert SWEEP.count_refutations(text, NO_PROGRAMS) == 0, (
        "the witness must contribute nothing when no reading was handed to the pass")
    for flag in ("fresh", "owned"):
        held = getattr(reading, flag)
        assert held is True, f"the fixture itself must be a refuting reading: {flag}"
    assert SWEEP.snapshot_stamp(reading, DAY) is not None


def test_a_snapshot_stamp_names_the_commit_and_age_and_is_never_a_clearance():
    """Clause 4, second half: exactly one stamp carrying the commit and age it read.

    A block earns ONE `dated:` line — not one per field of the reading, and never a
    `cleared:` line, which is #2221's rule: an observed recovery today does not
    disprove the three days before it.
    """
    text = _snap_note()
    reading = _snap_reading()
    out, n, _notes = SWEEP.date_note(text, NO_PROGRAMS, DAY, snapshot=reading)
    stamps = _stamp_lines(out)
    assert n == 1 and len(stamps) == 1, stamps
    stamp = stamps[0]
    assert stamp.startswith(notify.DAILY_DATED_PREFIX), stamp
    assert SNAP_SHA[:8] in stamp and "3 m" in stamp, (
        f"the stamp must name the commit and age the reading carries: {stamp}")
    assert "snapshot_age_state" in stamp, (
        f"a dating is only checkable if it names its reading: {stamp}")
    assert "vaultwatch.snapshot_age_state" in stamp, (
        f"the re-pointable command is the published watchdog path, not a bare name: {stamp}")
    assert notify.DAILY_CLEARED_PREFIX not in stamp and "clear" not in stamp.lower(), (
        f"an observed recovery is a dating, never a clearance: {stamp}")
    body_end = text.index(SNAP_BODY) + len(SNAP_BODY)
    assert out[:body_end] == text[:body_end], "the alarm's own words were rewritten"
    assert out.endswith(text[text.index(ROLLBACK_TITLE):]), "the blocks below were disturbed"


def test_a_stale_unowned_or_unread_snapshot_reading_dates_nothing():
    """Clause 5: every refusal leaves the note byte-identical, and no `cleared:` appears.

    Four refusals. A stale HEAD is the ALARM this witness exists to refute, not its
    refutation. An unowned repository is somebody's fixture, and #3149's three sections
    on `memory/2026-10-21.md` are what it looks like when that distinction is missing. A
    probe that raises, and a probe that answers with something that is not the documented
    dict, are the two shapes of "the read did not happen" — and a missing `fresh` key
    defaulted to truthy would date every snapshot block in the vault off nothing.
    """
    text = _snap_note()
    for label, reading in (("stale", _snap_reading(fresh=False)),
                           ("unowned", _snap_reading(owned=False))):
        out, n, notes = SWEEP.date_note(text, NO_PROGRAMS, DAY, snapshot=reading)
        assert out == text and n == 0, f"the {label} reading dated a block: {notes}"
        assert notify.DAILY_CLEARED_PREFIX not in out

    for label, gave in (("raised", OSError("git: blocked")),
                        ("not a reading", "fresh"),
                        ("no axes", {"repo": SNAP_REPO}),
                        ("empty", {})):
        def _reader(_gave=gave):
            if isinstance(_gave, Exception):
                raise _gave
            return _gave
        try:
            got = SWEEP.read_snapshot(age_reader=_reader)
        except RuntimeError as exc:
            got = exc
        assert isinstance(got, RuntimeError), f"{label} was accepted: {got!r}"
        assert label in str(got) or "snapshot" in str(got).lower(), str(got)
        assert SWEEP.snapshot_refutes(SNAP_TITLE, None) is False, (
            "no reading can never refute, whatever the title says")
        assert SWEEP.snapshot_stamp(None, DAY) is None, (
            "no witness is never a stamp, not even an empty one")
        assert SWEEP.snapshot_stamp(reading=None, day=DAY) is None
    assert SWEEP.snapshot_refutes(TMP_TITLE, _snap_reading()) is False, (
        "a fresh snapshot reading must never speak for another family")


def test_the_cli_dates_a_snapshot_block_once_and_refuses_with_exit_2(tmp_path, capsys):
    """The CLI across the boundary: one reading per pass, one write, exit 2 on no reading.

    Drives `main` over a real note file rather than a string, because the clause that
    matters for the page Alan reads is the file's bytes: one stamp per block after
    `--apply`, and a pass that could not read the probe must not print "0 blocks" and
    exit 0 — that is the shape of a clean corpus, and this pass measured nothing.
    """
    def label_for(r, raises):
        return ("raise" if raises else
                "fresh" if (r and r.fresh and r.owned) else
                "stale" if r and not r.fresh else "unowned")

    def _cli(reading, raises=False):
        def _snap():
            if raises:
                raise OSError("git log refused by the sandbox")
            return reading
        notes = tmp_path / f"notes-{label_for(reading, raises)}"
        notes.mkdir(exist_ok=True)
        note = notes / "2026-11-04.md"
        note.write_text(_snap_note(with_state=False), encoding="utf-8")
        # The other two witnesses are handed NOTHING, not a healthy reading: an absent
        # reading refutes nothing, so the one stamp this node counts can only have come
        # from the snapshot one. Passing their defaults would date three blocks and the
        # count below would be a count of the fixture.
        rc = _run_cli(notes, "--apply", "--day", DAY, fraction=None, tripwire={},
                      snapshot=reading, snap_raises=raises)
        return rc, note, capsys.readouterr()

    rc, note, cap = _cli(_snap_reading())
    assert rc == 0, cap.err
    text = note.read_text(encoding="utf-8")
    assert len(_stamp_lines(text)) == 1, text
    assert SNAP_SHA[:8] in text and notify.DAILY_CLEARED_PREFIX not in text
    assert "snapshot" in cap.out.lower() and SNAP_SHA[:8] in cap.out, cap.out

    rc2, note2, cap2 = _cli(_snap_reading(), raises=True)
    assert rc2 == 2, (rc2, cap2.err)
    assert note2.read_text(encoding="utf-8") == _snap_note(with_state=False), \
        "a failed read wrote bytes"
    assert "snapshot" in cap2.err.lower(), cap2.err

    rc3, note3, cap3 = _cli(_snap_reading(fresh=False))
    assert rc3 == 2, (rc3, cap3.err)
    assert note3.read_text(encoding="utf-8") == _snap_note(with_state=False), \
        "a stale reading wrote bytes"

    rc4, note4, _cap4 = _cli(_snap_reading(owned=False))
    assert rc4 == 2, rc4
    assert note4.read_text(encoding="utf-8") == _snap_note(with_state=False), \
        "an unowned reading wrote"


def test_read_snapshot_maps_the_producer_s_own_reading_without_re_rendering_it():
    """The seam itself: the stamp quotes the probe's sentence rather than a second one.

    `snapshot_age_state` already prints repository, commit and age in the unit a person
    reads, and is the same function the alarm and the guardian's own `cleared:` line are
    built from. A stamp that recomputed an age string would be a second definition of an
    age able to drift from the first, so the reading carries `text` and the stamp quotes
    it. The one exception the node pins is a probe whose wording grew a marker word:
    that is refused, because #2221's corpus instrument is `grep -c 'cleared:'` and a
    rendered string must not be able to forge one.
    """
    reading = SWEEP.read_snapshot(age_reader=lambda: {
        "repo": SNAP_REPO, "sha": SNAP_SHA, "committed_at": 1, "age": 180.0,
        "max_age": 10800.0, "fresh": True, "owned": True, "beat_age": 180.0,
        "text": f"{SNAP_REPO} commit {SNAP_SHA[:8]} is 3 m old"})
    assert (reading.fresh, reading.owned, reading.sha) == (True, True, SNAP_SHA)
    stamp = SWEEP.snapshot_stamp(reading, DAY)
    assert stamp is not None and f"{SNAP_REPO} commit {SNAP_SHA[:8]} is 3 m old" in stamp, stamp

    forged = SWEEP.read_snapshot(age_reader=lambda: {
        "repo": SNAP_REPO, "sha": SNAP_SHA, "fresh": True, "owned": True,
        "max_age": 10800.0, "text": "the alarm is cleared: everything is fine"})
    assert SWEEP.snapshot_stamp(forged, DAY) is None, (
        "a probe's wording must not be able to forge a retraction marker")
    blank = SWEEP.read_snapshot(age_reader=lambda: {
        "repo": SNAP_REPO, "sha": None, "fresh": True, "owned": True, "max_age": 10800.0,
        "text": ""})
    assert SWEEP.snapshot_stamp(blank, DAY) is None, (
        "a reading with no sentence in it has nothing to quote, so it dates nothing")


# ── #3993: a dating lands inside the block it retracts ─────────────────────────
#
# The stamp is spliced at the refuted section's `end`, and `sections()` sets that to the NEXT `## `
# heading or, for the note's last section, to the end of the FILE. So a block that the human's own
# `---` rule and `### Session` digest sit inside — which is what a note ending in a guardian alarm
# looks like — was dated below both. On `memory/2026-11-04.md` the retraction of the block at
# :54-64 landed at :72, below Alan's digest at :68, so the destructive instruction inside that
# block reads as unretracted to anyone reading the block, and the human's session digest reads as
# followed by an alarm retraction. The rule is now the block's own content, not the section's index:
# insert above the first bare `---` rule or `#`-heading the span swallowed. Measured 2026-11-09
# across the 125 guardian spans of the 2026 daily notes, 37 have such a tail and 21 of those are NOT
# the note's last section — which is why this is a span rule and not an `is-last-section` test. The
# four nodes below pin the new placement plus the three properties that must not move with it: a
# block with no tail keeps the offset it has always had, a second pass is still a byte-identical
# no-op, and no line of the human's note is dropped or reworded by the new position.

TAIL_NOTE = f"""---
type: note
timestamp: '2026-11-04T08:00:00'
---

# 2026-11-04

## Self-mod guardian: {TW.ALERT_TITLE}

/tmp measured 81% of inodes used at 08:00. This needs a human.

## Something the human wrote

A paragraph of the human's own note, mid-file.

## Self-mod guardian: {TW.ALERT_TITLE}

/tmp measured 82% of inodes used at 09:00. This needs a human.

---

### Session 10:32 PST — Auto-captured

Alan's own session digest for the day.
"""

#: The same note with the refuted block followed straight by another `## ` section: its span holds
#: no `---` rule and no `#`-heading, so the insertion offset must stay exactly where it is today.
#: The positive control — without it a fix that inserted the stamp anywhere (start of file, start of
#: the block, one blank line lower) could satisfy the placement node and corrupt every ordinary note.
UNTILED_NOTE = f"""---
type: note
timestamp: '2026-11-04T08:00:00'
---

# 2026-11-04

## Self-mod guardian: {TW.ALERT_TITLE}

/tmp measured 81% of inodes used at 08:00. This needs a human.

## What the human wrote next

A section with no rule and no sub-heading in it.
"""

FRONT_MATTER_RULES = 2          # the note's own `---` pair, which is no block's tail


def _line_numbers(lines: list[str], pred) -> list[int]:
    """1-based line numbers of every line `pred` accepts."""
    return [n for n, line in enumerate(lines, 1) if pred(line)]


def _dated_numbers(lines: list[str]) -> list[int]:
    return _line_numbers(lines, lambda l: l.startswith(notify.DAILY_DATED_PREFIX))


def _one(lines: list[str], pred, what: str) -> int:
    """Exactly one matching line number, or a failure naming what was found instead.

    Every placement assertion needs line numbers, and a `.index` on a fixture that moved by one
    line reports a number as if it were still the finding."""
    hits = _line_numbers(lines, pred)
    assert len(hits) == 1, f"{what}: expected exactly one line, found {hits} in {lines!r}"
    return hits[0]


def test_the_final_block_dating_lands_above_the_rule_and_digest_that_follow_it(tmp_path, capsys):
    """Clause 1, through `--apply` and re-read from disk: under the block's body, above the rule.

    The artifact is the note, so a splice that is right in the returned string and wrong on disk is
    the same defect with one step missing.
    """
    note = tmp_path / "2026-11-04.md"      # `--notes` is tmp_path, as `_copy_fixture` puts it
    note.write_text(TAIL_NOTE, encoding="utf-8")
    rc = _run_cli(tmp_path, "--apply", "--day", DAY, fraction=0.2769)
    out = capsys.readouterr().out
    assert rc == 0 and "dated blocks: 2" in out, out
    lines = note.read_text(encoding="utf-8").splitlines()
    stamps = _dated_numbers(lines)
    assert len(stamps) == 2, f"both refuted blocks are dated, one per block: {lines!r}"
    rules = _line_numbers(lines, lambda l: l.strip() == "---")
    assert len(rules) == FRONT_MATTER_RULES + 1, f"the human's rule plus the front-matter pair: {rules}"
    rule, digest, final_stamp = rules[-1], _one(
        lines, lambda l: l.startswith("### Session"), "the `### Session` digest"), stamps[1]
    assert rule < digest and digest < len(lines), "the fixture: rule, then the human's digest"
    assert final_stamp < rule, (
        f"the dating for the final guardian block sits at {final_stamp}, at or below the note's "
        f"`---` rule at {rule}: before #3993 the last section's insertion offset was the end of the "
        f"FILE, so the retraction landed below the human's own digest at {digest}")
    assert final_stamp > _one(lines, lambda l: l.startswith("/tmp measured 82%"),
                              "the final block's alarm line"), \
        "the stamp moved above the block it retracts instead of below its body"
    assert lines[final_stamp - 1] != lines[rules[-1] - 1], "the stamp replaced the rule"


def test_the_stamp_sits_under_the_body_it_retracts_and_not_one_line_lower():
    """Clause 1's exact neighbourhood: the stamp follows the alarm line, and nothing else moved."""
    out, n, _ = SWEEP.date_note(TAIL_NOTE, running=frozenset(), day=DAY, fraction=0.2769)
    assert n == 2, out
    lines = out.splitlines()
    stamp = _dated_numbers(lines)[1]
    body_end = _one(lines, lambda l: l.startswith("/tmp measured 82%"),
                    "the final block's alarm line")
    assert stamp == body_end + 2, (
        f"the stamp is at {stamp} and the refuted alarm line at {body_end}: the retraction belongs "
        f"directly under the block's own content, separated by the one blank line the note already "
        f"had, and nowhere else")
    assert lines[stamp - 2] == "", "no blank line invented between the body and its retraction"
    assert _line_numbers(lines, lambda l: l.strip() == "---")[-1] == stamp + 2, \
        "the blank line above the human's rule survived and nothing between them changed"


def test_a_block_with_no_rule_or_heading_in_its_span_keeps_the_offset_it_has_always_had():
    """Clause 2, the positive control: the mid-note position is what it always was.

    `UNTILED_NOTE`'s refuted span ends at the next `## ` heading with nothing swallowed, so the
    offset the sweep has always used is still the right one, and the shape that has always come out
    is asserted rather than eyeballed: alarm line, blank, stamp, then the next heading on the very
    next line.
    """
    out, n, _ = SWEEP.date_note(UNTILED_NOTE, running=frozenset(), day=DAY, fraction=0.2769)
    assert n == 1, out
    pre, post = UNTILED_NOTE.splitlines(), out.splitlines()
    stamp = _dated_numbers(post)[0]
    assert stamp == _one(post, lambda l: l.startswith("/tmp measured 81%"),
                         "the alarm line") + 2, post
    assert post[stamp] == "## What the human wrote next", (
        f"line {stamp + 1} is {post[stamp]!r}: with nothing swallowed by the span the stamp must "
        f"still be the last thing in the block, immediately before the next `## ` heading, exactly "
        f"as it was before #3993 — no blank line invented, no position shifted")
    assert _deletions(UNTILED_NOTE, out) == 0
    assert len(post) - len(pre) == 1, "one line added: the stamp, and nothing else"


def test_a_second_pass_over_the_final_block_fixture_is_byte_identical_and_dates_nothing():
    """Clause 3: the new offset keeps the stamp inside the span, so idempotency survives.

    `already_sealed` is a substring test over the block's own span, so a stamp moved UPWARD inside
    that span is still seen by the next pass and a re-run writes nothing. That is what this node
    exists for: a fix that pushed the stamp past the span's `end` would re-date the block every
    pass, and one that pushed it above the block's own heading would trip the cross-note backstop
    and go silent on that note forever.
    """
    once, n1, _ = SWEEP.date_note(TAIL_NOTE, running=frozenset(), day=DAY, fraction=0.2769)
    assert n1 == 2, once
    twice, n2, notes = SWEEP.date_note(once, running=frozenset(), day=DAY, fraction=0.2769)
    assert twice == once and n2 == 0, (
        f"a second pass changed the note again ({n2} block(s) dated, {notes}): the stamp no longer "
        f"falls inside the span `already_sealed` reads, so every pass would add another")
    sealed = [SWEEP.already_sealed(once[start:end])
              for title, start, end in SWEEP._guardian_sections(once)]
    assert sealed and all(sealed), \
        f"a dated block whose own span no longer reads as sealed: {sealed}"


def test_the_dating_is_the_only_change_and_no_note_line_is_dropped_or_reworded():
    """Clause 4 for both fixtures: insert-only, and every original written line survives in order.

    The fix is a splice position, so the whole risk is collateral: a `---` rule or a digest line
    eaten by the arithmetic would be invisible to the placement nodes, which read only the
    neighbourhood of the stamp. Checked over both shapes — the tail note is the one at risk, the
    untilled note is the one that must never have been.

    Blank lines are counted separately from written ones because the unit of a stamp is a LINE: a
    stamp spliced above a swallowed rule needs a blank under it or that rule becomes its setext
    underline, so the note legitimately gains one blank line per stamp that sits above a tail, and
    gains none where the block already ended in a blank run. Blank lines are therefore asserted by
    count, and every NON-blank line of the human's note is asserted to survive in order, byte for
    byte, with every inserted non-blank line a `dated:` line.
    """
    # 2 for the tail note because it refutes TWO blocks, 1 for the untilled note: the count is not
    # what this node is about, it only has to be right on both sides of the diff.
    for text, dated, blanks_added in ((TAIL_NOTE, 2, 1), (UNTILED_NOTE, 1, 0)):
        out, n, _ = SWEEP.date_note(text, running=frozenset(), day=DAY, fraction=0.2769)
        assert n == dated, (n, out)
        pre, post = text.splitlines(), out.splitlines()
        assert _deletions(text, out) == 0, (out, text)
        ops = {(op, i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in
               difflib.SequenceMatcher(None, pre, post).get_opcodes()}
        assert all(op in ("equal", "insert") for op, _, _ in ops), \
            f"the splice rewrote or moved a line of the note rather than inserting: {sorted(ops)}"
        assert [l for l in post if l.strip() and not l.startswith(notify.DAILY_DATED_PREFIX)] \
            == [l for l in pre if l.strip()], \
            "every written line of the note survives, in order, byte-identical"
        assert len(_dated_numbers(post)) == dated, "the only lines added are the stamps"
        assert sum(1 for l in post if not l.strip()) - sum(1 for l in pre if not l.strip()) \
            == blanks_added, (
            f"{blanks_added} blank line(s) may be gained, one per stamp that sits above a swallowed "
            f"rule so that rule stays a thematic break and not the stamp's setext underline")
        for pred, what in ((lambda l: l.strip() == "---", "`---` rule"),
                           (lambda l: l.startswith("### Session"), "`### Session` digest"),
                           (lambda l: l.startswith("## "), "`## ` heading")):
            assert len(_line_numbers(post, pred)) == len(_line_numbers(pre, pred)), what

    # The counterfactual, as arithmetic on the fixture rather than a claim about some other tree.
    # Before #3993 the splice position WAS the span's `end` — `inserts.append((end, …))`, and
    # `sections()` sets a final span's `end` to the end of the file — so the old placement needs no
    # simulation: it is `f_end`, and its line number is what the stamp used to land on. The new one
    # is the line the stamp actually occupies. Asserting the pair is what stops the placement node
    # above passing on a fixture that had stopped containing the defect.
    tail_out, _, _ = SWEEP.date_note(TAIL_NOTE, running=frozenset(), day=DAY, fraction=0.2769)
    _, _, f_end = list(SWEEP._guardian_sections(TAIL_NOTE))[-1]
    assert f_end == len(TAIL_NOTE), (
        "the final guardian span no longer runs to the end of the note, so this fixture is no "
        "longer the shape that misplaced a dating")

    def _note_line(offset: int) -> int:
        return TAIL_NOTE[:offset].count("\n") + 1

    pre_lines = TAIL_NOTE.splitlines()
    old_line = _note_line(f_end)                  # where the stamp sat before #3993, pre-fix lines
    assert old_line > _line_numbers(pre_lines, lambda l: l.startswith("### Session"))[0] \
        > _line_numbers(pre_lines, lambda l: l.strip() == "---")[-1], (
        f"the pre-#3993 offset for this span is line {old_line} of {len(pre_lines)}, which is not "
        f"below the human's `---` and digest: the fixture has stopped containing the defect, so "
        f"the placement node above proves nothing")
    post_lines = tail_out.splitlines()
    assert _dated_numbers(post_lines)[-1] < _line_numbers(
        post_lines, lambda l: l.strip() == "---")[-1], "the stamp sits at or below the `---`"
