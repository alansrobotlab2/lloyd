"""Nightly handoff validation — the gate between analysis and write.

Why this file exists
--------------------
`validate_handoff.py` is what stands between the Knowledge Analysis artifact and
the stage that rewrites USER.md and MEMORY.md. On 2026-08-24 a handoff claimed
the entity graph was "restored to 12,131 relationships" while the same night's
health report said zero; the claim was consumed and had to be walked back. The
validator itself had no tests, so nothing pinned what it actually catches.

Scope note: it validates *structure*, not truth. It will accept a well-formed
artifact full of false claims — tested explicitly below, because believing the
validator covers more than it does is how the 08-24 overclaim got through.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts import validate_handoff as vh

VALID = """---
generated: 2026-09-05
window: 2026-08-30..2026-09-05
---
# Knowledge Handoff — 2026-09-05

## Person: Alan

### Mental Model — Decision Patterns
- prefers scoped fixes over rewrites

### Mental Model — Communication Preferences
- terse, no sycophancy

### Mental Model — Technical Preferences
- Markdown over JSON

### Mental Model — Project Prioritization
- safety gates before features

### MEMORY.md Additions
- **reflection-pipeline health** — clean for 2 cycles

### MEMORY.md Removals
- (none)

### MEMORY.md Relocations
- (none)

### Profile Updates
- (none)

### Missing Files
- (none)

## Vault Propagations
- knowledge/ai/foo.md

## Tool Patterns — Failures
- `rg` missing post-migration

## Tool Patterns — Successes
- worktree isolation

## Conversation Patterns
- burst then trough

## Priority Order
1. safety
2. hygiene
"""


def run(tmp_path, monkeypatch, capsys, text=None, path=None):
    p = path or (tmp_path / "knowledge-handoff-2026-09-05.md")
    if text is not None:
        p.write_text(text, encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["validate_handoff.py", str(p)])
    code = vh.main()
    return code, capsys.readouterr()


def test_valid_handoff_passes(tmp_path, monkeypatch, capsys):
    code, out = run(tmp_path, monkeypatch, capsys, VALID)
    assert code == 0
    assert "[ok]" in out.out


def test_missing_file_fails(tmp_path, monkeypatch, capsys):
    code, out = run(tmp_path, monkeypatch, capsys,
                    path=tmp_path / "absent.md")
    assert code == 1 and "not found" in out.err


def test_no_arguments_is_a_usage_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["validate_handoff.py"])
    assert vh.main() == 2


def test_too_many_arguments_is_a_usage_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["validate_handoff.py", "a", "b"])
    assert vh.main() == 2


def test_missing_frontmatter_is_reported(tmp_path, monkeypatch, capsys):
    code, out = run(tmp_path, monkeypatch, capsys, VALID.split("---\n", 2)[2])
    assert code == 1 and "frontmatter" in out.out


def test_missing_h1_is_reported(tmp_path, monkeypatch, capsys):
    code, out = run(tmp_path, monkeypatch, capsys,
                    VALID.replace("# Knowledge Handoff — 2026-09-05", "# Notes"))
    assert code == 1 and "Knowledge Handoff" in out.out


@pytest.mark.parametrize("section", vh.REQUIRED_H2)
def test_each_required_h2_is_required(tmp_path, monkeypatch, capsys, section):
    broken = VALID.replace(section, "## Something Else Entirely")
    code, out = run(tmp_path, monkeypatch, capsys, broken)
    assert code == 1 and section in out.out


@pytest.mark.parametrize("h3", vh.REQUIRED_PERSON_H3)
def test_each_required_person_h3_is_required(tmp_path, monkeypatch, capsys, h3):
    broken = VALID.replace(h3, "### Renamed Subsection")
    code, out = run(tmp_path, monkeypatch, capsys, broken)
    assert code == 1 and h3 in out.out


def test_no_person_block_at_all_is_reported(tmp_path, monkeypatch, capsys):
    broken = VALID.replace("## Person: Alan", "## Nobody")
    code, out = run(tmp_path, monkeypatch, capsys, broken)
    assert code == 1 and "no `## Person:" in out.out


def test_second_person_is_validated_too(tmp_path, monkeypatch, capsys):
    """A second person added without the required subsections must not slip in
    unvalidated."""
    broken = VALID + "\n## Person: Second\n\n### Mental Model — Decision Patterns\n- x\n"
    code, out = run(tmp_path, monkeypatch, capsys, broken)
    assert code == 1 and "person `Second`" in out.out


def test_frequency_with_trailing_prose_is_rejected(tmp_path, monkeypatch, capsys):
    """The real drift it exists for: `frequency: 4 (tasks ...)` breaks the
    downstream integer parse."""
    broken = VALID.replace("- prefers scoped fixes over rewrites",
                           "frequency: 4 (tasks this week)")
    code, out = run(tmp_path, monkeypatch, capsys, broken)
    assert code == 1 and "non-integer trailing text" in out.out


def test_frequency_as_a_bare_integer_is_accepted(tmp_path, monkeypatch, capsys):
    ok = VALID.replace("- prefers scoped fixes over rewrites", "frequency: 4")
    code, out = run(tmp_path, monkeypatch, capsys, ok)
    assert code == 0, out.out


def test_frequency_that_is_not_an_integer_is_rejected(tmp_path, monkeypatch, capsys):
    broken = VALID.replace("- prefers scoped fixes over rewrites", "frequency: daily")
    code, out = run(tmp_path, monkeypatch, capsys, broken)
    assert code == 1 and "not a bare integer" in out.out


def test_frequency_line_numbers_point_at_the_offending_line(tmp_path, monkeypatch, capsys):
    broken = VALID.replace("- prefers scoped fixes over rewrites", "frequency: 4 (x)")
    code, out = run(tmp_path, monkeypatch, capsys, broken)
    expected = VALID.splitlines().index("- prefers scoped fixes over rewrites") + 1
    assert f"line {expected}:" in out.out


def test_all_issues_are_reported_together_not_just_the_first(tmp_path, monkeypatch, capsys):
    broken = VALID.replace("## Priority Order", "## Nope").replace(
        "### Missing Files", "### Nope2")
    code, out = run(tmp_path, monkeypatch, capsys, broken)
    assert code == 1
    assert "2 validation issue(s)" in out.out


def test_cosmetic_variation_is_tolerated_deliberately(tmp_path, monkeypatch, capsys):
    """Documented forgiveness: the 35B local model drifts on phrasing, so the
    validator checks presence, not prose."""
    ok = VALID.replace("- terse, no sycophancy",
                       "  - Terse!   No sycophancy whatsoever 🎯")
    code, _ = run(tmp_path, monkeypatch, capsys, ok)
    assert code == 0


def test_structurally_valid_but_factually_false_handoff_passes(tmp_path, monkeypatch, capsys):
    """The limit of this gate, stated as a test.

    The 08-24 incident handoff claimed the entity graph was restored to 12,131
    relationships while the same night's health report said zero. Nothing here —
    or in the validator — catches that. Claim-vs-disk verification has to live
    somewhere else; this only stops malformed artifacts from reaching the writer.
    """
    lie = VALID.replace("- (none)\n\n## Vault Propagations",
                       "- **KG RECOVERED to 12,131 relationships**\n\n## Vault Propagations")
    code, out = run(tmp_path, monkeypatch, capsys, lie)
    assert code == 0, "validator accepted an unfalsifiable claim — by design"
    assert "12,131" not in out.out


# ── split_person_blocks ──────────────────────────────────────────────────────

def test_person_blocks_are_split_by_name():
    text = "## Person: Alan\nbody a\n## Person: Bob\nbody b\n"
    blocks = vh.split_person_blocks(text)
    assert [n for n, _ in blocks] == ["Alan", "Bob"]
    assert "body a" in blocks[0][1] and "body b" in blocks[1][1]


def test_a_person_block_ends_at_the_next_h2():
    """Without this, `### Missing Files` from person one would satisfy person
    two's requirement and every multi-person handoff would pass vacuously."""
    text = ("## Person: Alan\n### Mental Model — Decision Patterns\nx\n"
            "## Vault Propagations\n### Mental Model — Decision Patterns\ny\n")
    blocks = vh.split_person_blocks(text)
    assert len(blocks) == 1
    assert "### Mental Model — Decision Patterns" in blocks[0][1]
    assert "y" not in blocks[0][1]


def test_no_person_blocks_yields_an_empty_list():
    assert vh.split_person_blocks("# just a title\n") == []


def test_person_names_are_trimmed():
    blocks = vh.split_person_blocks("## Person:    Padded Name   \nbody\n")
    assert blocks[0][0] == "Padded Name"


def test_h1_is_not_mistaken_for_an_h2_boundary():
    text = "## Person: Alan\n# H1 inside\nstill alan\n"
    blocks = vh.split_person_blocks(text)
    assert "still alan" in blocks[0][1]


def test_required_section_lists_are_not_empty():
    """A typo that emptied either list would make the validator pass anything."""
    assert vh.REQUIRED_H2 and vh.REQUIRED_PERSON_H3
    assert all(h.startswith("## ") for h in vh.REQUIRED_H2)
    assert all(h.startswith("### ") for h in vh.REQUIRED_PERSON_H3)
    assert len(set(vh.REQUIRED_H2)) == len(vh.REQUIRED_H2)
    assert len(set(vh.REQUIRED_PERSON_H3)) == len(vh.REQUIRED_PERSON_H3)


# ── the clock check: `generated_at` may not run ahead of the file (#2610) ────
#
# Everything above this line is structure. `check_structure` reads delimiters and
# headings only, so a hand-typed stamp shipped green:
# `knowledge-handoff-2026-10-11.md` carries `generated_at:
# 2026-10-11T11:41:00+00:00` against an mtime of 11:23:17Z, and the validator
# printed `[ok] … structure valid` and exited 0 on a stamp 1063 seconds in that
# file's own future. Three prose "never hand-type a clock" rules across the
# reflection skills stopped none of the instances, so what follows pins the
# comparison instead: `generated_at > mtime + CLOCK_AHEAD_MAX_SECONDS` (120 s),
# measured against `os.path.getmtime`.
#
# Direction matters. A stamp that LAGS its mtime is the ordinary case — the amend
# in-place pass rewrites a section and keeps the earlier stamp, and the finalizer
# appends an `## Addendum` after the original write. Over the 15 real
# `knowledge-handoff-*.md` files in `_pipeline/reflection/`, 9 lag and one sits
# exactly at its mtime; the rest are ahead (+1063, +937, +430, +26 s beyond the
# tolerance, +3 and 0 s inside it). So the check is one-directional, and the
# fixtures below assert a lagging stamp PASSES. They are all `tmp_path` files and
# never a walk of that corpus, because three of its files fail this check today
# and correctly so.
#
# Absence is a pass as well: the `VALID` fixture at the top of this file carries
# `generated: 2026-09-05` and no `generated_at` at all, and signals reports use
# another key again (`generated:`). Only a stamp that is present AND parseable AND
# ahead is a finding — treating absence as one would redden this whole file.

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate_handoff.py"
TOL = vh.CLOCK_AHEAD_MAX_SECONDS          # the tolerance the message must name
MTIME = datetime(2026, 10, 12, 5, 0, 0, tzinfo=timezone.utc)   # fixture mtime


def _handoff(stamp: str | None) -> str:
    """A structurally valid handoff, optionally stamped.

    The required sections are written out of `vh.REQUIRED_H2` and
    `vh.REQUIRED_PERSON_H3` rather than typed here, so a fixture can never go red
    because the validator's own lists moved — and so a node that fails on a
    missing section, when it is meant to be failing on a clock, is impossible.
    """
    fm = f"generated_at: {stamp}\n" if stamp is not None else ""
    h2 = "\n".join(f"{h}\nfiller\n" for h in vh.REQUIRED_H2)
    h3 = "\n".join(f"{h}\nfiller\n" for h in vh.REQUIRED_PERSON_H3)
    return f"""---
type: note
{fm}---
# Knowledge Handoff — 2026-10-12

## Summary
filler

## Owes Follow-up
- [ ] owed: nothing.

{h2}
## Person: Nightly Knowledge Write

{h3}"""


def _stamp_file(tmp_path: Path, *, stamp: str | None, mtime: datetime,
                name: str = "knowledge-handoff-2026-10-12.md") -> Path:
    """Write a handoff, then set its mtime, so the pair is the test's own."""
    path = tmp_path / name
    path.write_text(_handoff(stamp), encoding="utf-8")
    os.utime(path, (mtime.timestamp(), mtime.timestamp()))
    return path


def _stamp_at(mtime: datetime, *, offset_seconds: int, fmt: str = "offset") -> str:
    gen = mtime + timedelta(seconds=offset_seconds)
    if fmt == "z":
        return gen.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if fmt == "offset":
        return gen.astimezone(timezone.utc).isoformat()            # +00:00
    if fmt == "naive":
        return gen.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    raise AssertionError(f"unknown stamp format {fmt!r}")


def _cli(path: Path) -> tuple[int, str]:
    """Drive the script the way the nightly step does: argv, exit code, stdout."""
    p = subprocess.run([sys.executable, str(SCRIPT), str(path)],
                       capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


# ── clause 1: ahead of the mtime fails; at the mtime passes ─────────────────


def test_a_stamp_600_seconds_ahead_of_its_mtime_exits_1(tmp_path):
    """The defect as a fixture. 600 s is five times the tolerance, and the
    instances that prompted this were +1063/+937/+430 s — every one of which the
    validator printed `[ok]` over. A presence check cannot fail this file: the
    key is present and well-formed, and only its value is dishonest."""
    path = _stamp_file(tmp_path, stamp=_stamp_at(MTIME, offset_seconds=600),
                       mtime=MTIME)
    code, out = _cli(path)
    assert code == 1, f"a stamp 600 s in the file's own future must be a failure: {out}"
    assert "[fail]" in out, f"a clock failure has to be reported as one: {out}"


def test_the_same_document_stamped_at_its_mtime_exits_0(tmp_path):
    """The control above: same bytes, same mtime, one number moved. Delta 0 is
    what an honest `date -u` taken immediately before the write produces."""
    path = _stamp_file(tmp_path, stamp=_stamp_at(MTIME, offset_seconds=0), mtime=MTIME)
    code, out = _cli(path)
    assert code == 0, f"a stamp equal to its mtime is the honest case: {out}"
    assert "[ok]" in out


def test_the_tolerance_is_120_seconds_and_both_sides_of_it_are_pinned(tmp_path):
    """121 s ahead is a failure, 119 s is not — the boundary rather than a round
    number, because a gap is allowed on purpose: the instant has to come from the
    tool call *before* the write (a harness tool call cannot interpolate a shell
    value), so a few seconds between `date -u` and the write is the honest shape,
    and a zero-tolerance check would fail every real artifact."""
    assert TOL == 120
    ahead = _stamp_file(tmp_path, stamp=_stamp_at(MTIME, offset_seconds=TOL + 1),
                        mtime=MTIME, name="ahead.md")
    inside = _stamp_file(tmp_path, stamp=_stamp_at(MTIME, offset_seconds=TOL - 1),
                         mtime=MTIME, name="inside.md")
    assert _cli(ahead)[0] == 1, f"{TOL + 1} s ahead is past the {TOL} s tolerance"
    assert _cli(inside)[0] == 0, f"{TOL - 1} s ahead is inside it"


# ── clause 2: the check never fires backwards ───────────────────────────────


def test_a_stamp_an_hour_behind_its_mtime_exits_0(tmp_path):
    """The ordinary amend-in-place case: a later pass rewrites the file and the
    stamp keeps its earlier instant, so `generated_at - mtime` is negative. Nine
    of the 15 real handoffs sit that way, so firing backwards would retroactively
    fail most of the corpus and the nightly chain would stop shipping handoffs."""
    path = _stamp_file(tmp_path, stamp=_stamp_at(MTIME, offset_seconds=-3600),
                       mtime=MTIME)
    code, out = _cli(path)
    assert code == 0, f"a lagging stamp must stay silent: {out}"


def test_a_stamp_a_day_behind_its_mtime_exits_0(tmp_path):
    """One hour behind is a re-emit; a day behind is a stamp copied forward from
    yesterday, which is a real defect the corpus does contain (a signals report
    measured −82,928 s). Bounding the stale direction is a separate ruling owed
    on #2610, so this node pins that the check does NOT grow one quietly: a
    fixture asserting the absence of a second direction is what keeps the
    one-directional contract honest."""
    path = _stamp_file(tmp_path, stamp=_stamp_at(MTIME, offset_seconds=-86400),
                       mtime=MTIME)
    code, out = _cli(path)
    assert code == 0, f"the stale direction is out of scope for this check: {out}"


# ── clause 3: absent or unparseable stamps add no new failure ───────────────


def test_a_handoff_with_no_generated_at_key_exits_0(tmp_path):
    """`VALID` at the top of this file carries `generated: 2026-09-05` and no
    `generated_at`, and `check_structure` has never required the key. Absence as
    a failure would redden that entire file, so absence is silence."""
    path = _stamp_file(tmp_path, stamp=None, mtime=MTIME)
    code, out = _cli(path)
    assert code == 0, f"a file with no stamp is not a clock finding: {out}"


def test_an_unparseable_generated_at_exits_0(tmp_path):
    """`tonight`, `2026-13-45T99:00` and `null` are not instants. A validator
    that cannot read a stamp must not report that the stamp is wrong: there are
    no two instants to compare, so there is no finding to make."""
    for raw in ("tonight", "2026-13-45T99:00", "null"):
        path = _stamp_file(tmp_path, stamp=raw, mtime=MTIME, name=f"{raw}.md")
        code, out = _cli(path)
        assert code == 0, f"unparseable stamp {raw!r} must not fail the clock: {out}"


# ── clause 4: the offset formats present in the live corpus ─────────────────


def test_a_z_stamp_and_a_plus_offset_stamp_ahead_both_exit_1(tmp_path):
    """The two shapes real handoffs actually use, both from the same scan:
    2026-10-11 is `2026-10-11T11:41:00+00:00`, 2026-10-12 is
    `2026-10-12T05:18:24Z`. Parsing one and not the other would pass whichever
    format the writer happened to emit that day — coverage that reads as green
    forever."""
    for fmt in ("z", "offset"):
        path = _stamp_file(tmp_path, stamp=_stamp_at(MTIME, offset_seconds=600,
                                                     fmt=fmt),
                           mtime=MTIME, name=f"{fmt}.md")
        code, out = _cli(path)
        assert code == 1, f"a 600 s-ahead {fmt} stamp must fail: {out}"


def test_a_z_stamp_and_a_plus_offset_stamp_at_the_mtime_both_exit_0(tmp_path):
    """The same two formats on the passing side, so the node above cannot be
    satisfied by an implementation that fails every stamp it manages to parse."""
    for fmt in ("z", "offset"):
        path = _stamp_file(tmp_path, stamp=_stamp_at(MTIME, offset_seconds=0, fmt=fmt),
                           mtime=MTIME, name=f"{fmt}-ok.md")
        code, out = _cli(path)
        assert code == 0, f"a {fmt} stamp equal to its mtime must pass: {out}"


def test_a_stamp_with_no_offset_is_read_as_utc(tmp_path):
    """A naive instant has no zone to argue with, and this box runs at -0700: read
    as UTC it is 600 s ahead (a failure), read as local time it would land 7 h in
    the past and pass. The interpretation IS the behaviour, so it is pinned from
    both sides instead of asserted in a comment."""
    ahead = _stamp_file(tmp_path, stamp=_stamp_at(MTIME, offset_seconds=600,
                                                  fmt="naive"),
                        mtime=MTIME, name="naive-ahead.md")
    at = _stamp_file(tmp_path, stamp=_stamp_at(MTIME, offset_seconds=0, fmt="naive"),
                     mtime=MTIME, name="naive-at.md")
    assert _cli(ahead)[0] == 1, "a naive stamp is UTC, so 600 s ahead fails"
    assert _cli(at)[0] == 0, "a naive stamp equal to its mtime passes"
    behind = _stamp_file(tmp_path,
                         stamp=_stamp_at(MTIME, offset_seconds=-7 * 3600 + 600,
                                         fmt="naive"),
                         mtime=MTIME, name="naive-as-local-would-be-ahead.md")
    assert _cli(behind)[0] == 0, (
        "this failing means a naive stamp was read as local time: that 600 s gap "
        "is exactly the failure a -0700 box would invent out of an honest stamp")


def test_the_in_process_check_returns_the_same_verdict_as_the_cli(tmp_path):
    """The comparison is callable, not a side effect of printing: the nightly step
    reads structure findings in-process elsewhere in this file, and a node that
    drove only the subprocess would pin the printer and leave the predicate
    free to disagree with it."""
    ahead = _stamp_file(tmp_path, stamp=_stamp_at(MTIME, offset_seconds=600),
                        mtime=MTIME, name="ahead.md")
    ok = _stamp_file(tmp_path, stamp=_stamp_at(MTIME, offset_seconds=0), mtime=MTIME,
                     name="ok.md")
    assert len(vh.check_clock(ahead)) == 1, "one finding for one dishonest stamp"
    assert vh.check_clock(ok) == [], "no finding for an honest one"
    assert vh.check_clock(_stamp_file(tmp_path, stamp=None, mtime=MTIME,
                                      name="none.md")) == []


# ── clause 5: the message names both instants and the delta ─────────────────


def test_the_clock_failure_names_both_instants_and_the_delta(tmp_path):
    """"invalid generated_at" is not actionable: a reader cannot tell 3 s of write
    latency from a 17-minute hand-typed clock, and the magnitude is what made all
    three instances visible in the first place. So the message carries the stamp
    as written, the mtime as UTC, the delta in seconds and the tolerance it was
    measured against."""
    msg = vh.check_clock(_stamp_file(tmp_path, stamp="2026-10-12T05:10:00+00:00",
                                     mtime=MTIME))[0]
    assert "2026-10-12T05:10:00+00:00" in msg, f"the stamp is not quoted: {msg}"
    assert "2026-10-12T05:00:00" in msg, f"the mtime instant is missing: {msg}"
    assert "600" in msg, f"the delta in seconds is missing: {msg}"
    assert "120" in msg, f"the tolerance is not named: {msg}"


def test_the_delta_is_signed_so_the_direction_is_readable(tmp_path):
    """A bare magnitude reads the same whichever way the clock is wrong. The delta
    is signed, and only an ahead stamp reaches this text at all, so the sign is
    what tells a reader the file claims to have been written after it came to
    exist rather than merely being old."""
    msg = vh.check_clock(_stamp_file(tmp_path, stamp="2026-10-12T05:10:00Z",
                                     mtime=MTIME))[0]
    assert "+600" in msg, f"the delta should read as signed +600: {msg}"


def test_a_missing_section_and_a_future_stamp_are_both_reported(tmp_path):
    """The clock finding goes into the same issues list the structure checks
    write to — not printed and returned early. A file with two problems that
    reports one teaches the reader to expect one, and the next pass stops at the
    first line."""
    path = tmp_path / "both.md"
    dropped = vh.REQUIRED_H2[0]
    broken = _handoff("2026-10-12T05:10:00+00:00").replace(f"{dropped}\n", "", 1)
    assert dropped in vh.REQUIRED_H2 and dropped not in broken, (
        f"the fixture did not actually drop {dropped!r}, so it would prove "
        f"nothing about a file carrying two problems")
    path.write_text(broken, encoding="utf-8")
    os.utime(path, (MTIME.timestamp(), MTIME.timestamp()))
    code, out = _cli(path)
    assert code == 1
    assert "missing required section" in out, f"the structure finding was lost: {out}"
    assert "generated_at" in out, f"the clock finding was lost: {out}"
    assert "600" in out, f"the clock finding lost its magnitude: {out}"
