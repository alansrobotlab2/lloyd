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


# ── the producer-status join: a dead run's artifact is not a handoff (#2614) ──
#
# The incident, end to end. `run_42_20261011_050938` (task #42, Knowledge Analysis)
# died of a `ConnectTimeout` at 342.9 s — `status: failed`, `failure_kind: infra` —
# having CREATED `~/lloyd-data/_pipeline/reflection/knowledge-handoff-2026-10-11.md`
# at 05:12:21Z with three pattern sections still at `- (none)`. Its changes
# manifest (`~/lloyd-data/sessions/20261010_220938_autonomy_7248.changes/
# run_42_20261011_050938/index.json`) holds exactly that one `create` entry at
# `"reverted_at": null`, so the skeleton stayed at the canonical path as the newest
# artifact in the window. #39 dispatched 2 h 45 m later and consumed it; its own
# completion note records "modified 2026-10-10 22:12:21 -0700, inside the
# 1,440-minute freshness window. Validation: pass". The cycle concluded there was
# nothing to write, and the real handoff arrived at 11:23:17Z — 3 h 08 m 28 s after
# the write that should have read it, where it sat unread ~21 h.
#
# Every gate that ran answered a question the artifact could answer for itself, and
# a skeleton answers all of them correctly: `validate_handoff.py` is presence-only,
# and the analysis skill says in its own words that "A skeleton of headings holding
# `- (none)` exits 0 (verified against the checked-in validator)". The question none
# of them asked goes to the PRODUCER: which run last wrote this path, and what did
# that run report? A run that dies mid-write leaves nothing suspicious in the file —
# its `status: in-progress` is what a healthy in-flight handoff says too — but it
# does leave a terminal record: `status: failed`, closed AFTER the file's last
# write, with no successful run since. That triple is the corpse, and it is legible
# from `autonomy-runs/<task>/` plus the file's mtime alone.
#
# Two things are deliberately NOT signals, both for stated reasons. `- (none)`
# content: a quiet day legitimately writes it, so emptiness would refuse the
# corpus's ordinary nights (the report names that as the bad fix). And the
# producer-side revert-or-rename of a dead run's own writes: no honest discriminator
# exists at death time (#2614 §Findings, and
# `test_revert_by_run_id.py::test_autonomy_reverts_only_from_the_explicit_entry_point`
# pins the posture), so this join reads the producer's record instead of touching
# the producer's files.

DEATH_START = datetime(2026, 10, 11, 5, 9, 38, tzinfo=timezone.utc)    # run_42_…050938 start
SKELETON = datetime(2026, 10, 11, 5, 12, 21, tzinfo=timezone.utc)      # its one create entry
DEATH = datetime(2026, 10, 11, 5, 15, 21, tzinfo=timezone.utc)         # its completed_at
PREV_SUCCESS = datetime(2026, 10, 9, 5, 22, 33, tzinfo=timezone.utc)   # newest success at the time
RETRY_WRITE = datetime(2026, 10, 11, 11, 23, 17, tzinfo=timezone.utc)  # the real handoff
RETRY = datetime(2026, 10, 11, 11, 24, 1, tzinfo=timezone.utc)         # its completed_at
LATER_DEATH = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)    # a death after it

DEATH_RECORD = ("run_42_20261011_050938", "failed", DEATH)
PREV_RECORD = ("run_42_20261009_051043", "success", PREV_SUCCESS)
RETRY_RECORD = ("run_42_20261011_111549", "success", RETRY)

# The class marker. The consumer has to tell this refusal from a structural or clock
# one because the two get opposite instructions — see clause 5's node below, and
# `skills/nightly-reflection-knowledge-write/SKILL.md` Step 1.
PRODUCER_MARKER = "producing run record"


def _board(root: Path, *, artifact_template: str, task_id: int = 42) -> Path:
    """A task board holding one task that DECLARES that output-artifact template."""
    board = root / "autonomy"
    board.mkdir(parents=True, exist_ok=True)
    (board / f"{task_id}-fixture-producer.md").write_text(
        "---\n"
        f"id: {task_id}\n"
        "name: Fixture producer\n"
        "frequency: daily\n"
        "status: up_next\n"
        f"output_artifact: {artifact_template}\n"
        "---\n\n## Activity log\n", encoding="utf-8")
    return board


def _records(root: Path, *records: tuple, task_id: int = 42) -> Path:
    """`autonomy-runs/<task>/run_*.md`, spelled the way `_write_run_record` spells it."""
    runs = root / str(task_id)
    runs.mkdir(parents=True, exist_ok=True)
    for run_id, status, completed in records:
        (runs / f"{run_id}.md").write_text(
            "---\n"
            f"task_id: {task_id}\n"
            f"run_id: {run_id}\n"
            f"status: {status}\n"
            f"started_at: '{(completed - timedelta(seconds=300)).isoformat()}'\n"
            f"completed_at: '{completed.isoformat()}'\n"
            "---\n\n# Run record\n", encoding="utf-8")
    return runs


def _chain(tmp_path, monkeypatch, *, records: tuple, mtime: datetime = SKELETON,
           board_exists: bool = True,
           name: str = "knowledge-handoff-2026-10-11.md") -> Path:
    """Build the producer board, its run records and the artifact, all under tmp_path.

    Routed through the two roots the runtime already lets a caller redirect —
    `LLOYD_DATA` (`app.data_root.DATA_ROOT_ENV`) and `LLOYD_VAULT_ROOT`
    (`VAULT_ROOT_ENV`) — so the CLI and the in-process call resolve the same pair a
    nightly run resolves, and nothing here is a test-only back door.
    """
    data = tmp_path / "lloyd-data"
    vault = tmp_path / "obsidian"
    out = data / "_pipeline" / "reflection"
    out.mkdir(parents=True, exist_ok=True)
    artifact = out / name
    artifact.write_text(_handoff(None), encoding="utf-8")
    os.utime(artifact, (mtime.timestamp(), mtime.timestamp()))
    if board_exists:
        _board(vault, artifact_template=str(out / "knowledge-handoff-{date}.md"))
    _records(data / "autonomy-runs", *records)
    monkeypatch.setenv("LLOYD_DATA", str(data))
    monkeypatch.setenv("LLOYD_VAULT_ROOT", str(vault))
    return artifact


# ── clause 1: the refusal fires, and names the record that decided it ─────────


def test_the_newest_record_being_a_death_makes_the_validator_exit_1(tmp_path, monkeypatch):
    """The defect as a fixture: structurally perfect, produced by a dead run.

    The artifact is `_handoff(None)` — every required H2 and H3 present, no clock to
    fail on — and the board declares it as #42's `output_artifact`. The only thing
    wrong with the file is who wrote it: `run_42_20261011_050938` is the newest
    record, reads `failed`, closed 178 s AFTER the file was last written, and the
    newest success (`run_42_20261009_051043`, 10-09) predates that write. The
    2026-10-11 validator printed `[ok] … structure valid` on exactly this.
    """
    artifact = _chain(tmp_path, monkeypatch, records=(PREV_RECORD, DEATH_RECORD))
    code, out = _cli(artifact)
    assert code == 1, f"a dead run's skeleton must not validate as a handoff: {out}"
    assert "[fail]" in out, f"a refusal has to read as one: {out}"


def test_the_producer_refusal_names_the_record_path_status_and_both_instants(
        tmp_path, monkeypatch):
    """"stale artifact" is not actionable, and the class has to be nameable.

    The consumer's next action differs completely by refusal class — a missing H2 is
    answered by re-emitting, a producer-status refusal by NOT consuming — so the
    finding carries its class marker, the run record's name, its terminal status
    word, the mtime it weighed and the instant the run closed. The incident was
    diagnosable in exactly those terms afterwards; nothing said so at the time.
    """
    artifact = _chain(tmp_path, monkeypatch, records=(PREV_RECORD, DEATH_RECORD))
    code, out = _cli(artifact)
    assert code == 1
    assert PRODUCER_MARKER in out, f"the refusal does not name its class: {out}"
    assert "run_42_20261011_050938" in out, f"the run record is not named: {out}"
    assert "failed" in out, f"the terminal status is missing: {out}"
    assert "task #42" in out, f"the producing task is not named: {out}"
    assert "2026-10-11T05:12:21Z" in out, f"the artifact mtime is missing: {out}"
    assert "2026-10-11T05:15:21Z" in out, f"the death's instant is missing: {out}"


# ── clause 2: the verdict is a function, not a printed side effect ────────────


def test_the_join_is_callable_in_process_and_shells_out_to_nothing(tmp_path, monkeypatch):
    """`check_producer_status(path)` answers for itself — no subprocess, no shell.

    #2417's `newest_run_record` and `_artifact_candidates` already existed in
    `app.autonomy`, so this is composition over the same two questions, and the
    nightly step has to be able to ask them without parsing a CLI. The global
    `subprocess.run` / `os.system` tripwires are what keep "no subprocess" a tested
    property rather than a description of the current implementation.

    Both artifacts sit on ONE board under two producers (#42 the dead one, #43 a task
    whose newest record is also a death but whose file a success does claim), so the
    silent answer here is the claimant rescue seen in-process — and the proof that a
    second producer's template does not leak onto the first producer's file.
    """

    def tripwire(*args, **kwargs):
        raise AssertionError("the producer-status join must not shell out")

    monkeypatch.setattr(subprocess, "run", tripwire)
    monkeypatch.setattr(os, "system", tripwire)
    data, vault = tmp_path / "lloyd-data", tmp_path / "obsidian"
    out = data / "_pipeline" / "reflection"
    out.mkdir(parents=True)
    corpse = out / "knowledge-handoff-2026-10-11.md"
    corpse.write_text(_handoff(None), encoding="utf-8")
    os.utime(corpse, (SKELETON.timestamp(), SKELETON.timestamp()))
    live = out / "other-handoff-2026-10-11.md"
    live.write_text(_handoff(None), encoding="utf-8")
    os.utime(live, (RETRY_WRITE.timestamp(), RETRY_WRITE.timestamp()))
    _board(vault, artifact_template=str(out / "knowledge-handoff-{date}.md"))
    _board(vault, artifact_template=str(out / "other-handoff-{date}.md"), task_id=43)
    _records(data / "autonomy-runs", PREV_RECORD, DEATH_RECORD)
    _records(data / "autonomy-runs", PREV_RECORD, DEATH_RECORD, RETRY_RECORD,
             ("run_43_20261011_115900", "failed", LATER_DEATH), task_id=43)
    monkeypatch.setenv("LLOYD_DATA", str(data))
    monkeypatch.setenv("LLOYD_VAULT_ROOT", str(vault))

    findings = vh.check_producer_status(corpse)
    assert len(findings) == 1, f"one refusing record, one finding: {findings}"
    assert PRODUCER_MARKER in findings[0], f"the finding lost its class: {findings[0]}"
    assert vh.check_producer_status(live) == [], (
        f"a file a successful run claims must be silent in-process: "
        f"{vh.check_producer_status(live)}")


def test_producing_task_for_resolves_the_task_through_its_declared_template(
        tmp_path, monkeypatch):
    """The reverse map: `{date}` in a declared `output_artifact` is a date wildcard.

    The board stores the template (`knowledge-handoff-{date}.md`) and the artifact on
    disk is one of its expansions — in either date spelling, since the analysis stage
    names its file by UTC while `_artifact_candidates` resolves local first. A matcher
    that only took literal paths would match nothing the chain ever writes.
    """
    artifact = _chain(tmp_path, monkeypatch, records=(DEATH_RECORD,))
    board = tmp_path / "obsidian" / "autonomy"
    task = vh.producing_task_for(artifact, autonomy_dir=board)
    assert task is not None, "the declared template did not resolve to its producer"
    assert str(task["id"]) == "42", f"wrong task answered: {task}"
    other = tmp_path / "elsewhere.md"
    other.write_text(_handoff(None), encoding="utf-8")
    assert vh.producing_task_for(other, autonomy_dir=board) is None, (
        "an artifact no task declares must resolve to no producer")


# ── clause 3: the incident replayed in both directions ───────────────────────


def test_the_replay_refuses_the_10_11_skeleton_as_the_dead_run_s_output(
        tmp_path, monkeypatch):
    """Replay, refusing half: the skeleton restored, the death the newest record.

    Bytes, mtime and records are the incident's: the create entry at
    `2026-10-11T05:12:21Z`, `run_42_20261011_050938.md` newest at
    `completed_at: '2026-10-11T05:15:21.574264+00:00'`, and the newest success on
    10-09. The gate that ran that night exited 0; this one must not.
    """
    artifact = _chain(tmp_path, monkeypatch, records=(PREV_RECORD, DEATH_RECORD),
                      mtime=SKELETON)
    assert _cli(artifact)[0] == 1, "the replay must reproduce the refusal"


def test_the_replay_passes_once_the_retry_s_success_is_the_newest_record(
        tmp_path, monkeypatch):
    """Replay, healthy half — a gate that cannot pass this is not a gate.

    The retry wrote the real handoff at 11:23:17Z and recorded `run_42_20261011_111549`
    `success` at 11:24:01Z. The newest record is a success and that success postdates
    the mtime, so the artifact has a claimant. Refusing here would stop the nightly
    chain every night, which is the failure mode #2610's one-directional clock check
    took care not to create.
    """
    artifact = _chain(tmp_path, monkeypatch,
                      records=(PREV_RECORD, DEATH_RECORD, RETRY_RECORD),
                      mtime=RETRY_WRITE)
    code, out = _cli(artifact)
    assert code == 0, f"the retried handoff is a real handoff: {out}"
    assert "[ok]" in out


def test_a_death_after_a_successful_retry_leaves_the_claimed_artifact_valid(
        tmp_path, monkeypatch):
    """The other healthy direction: the newest record IS a death, and it is silent.

    A later run died at 12:00 having written nothing, but `run_42_20261011_111549`
    completed at 11:24:01Z — after this file's last write at 11:23:17Z, which is what
    a run that wrote it completing looks like. "A run died" is therefore not the
    predicate; "no successful run can claim this file" is. Firing here would refuse
    every healthy night that was unlucky enough to have a later failure.
    """
    artifact = _chain(tmp_path, monkeypatch,
                      records=(PREV_RECORD, DEATH_RECORD, RETRY_RECORD,
                               ("run_42_20261011_115900", "failed", LATER_DEATH)),
                      mtime=RETRY_WRITE)
    code, out = _cli(artifact)
    assert code == 0, f"a fresh artifact survives an unrelated later death: {out}"


def test_an_artifact_written_after_the_death_is_not_that_death_s_output(
        tmp_path, monkeypatch):
    """The bound that keeps the predicate from degrading to "some run died sometime".

    A retry in flight has not written its own record yet, so the newest record is
    still the earlier death — but a file last touched at 11:20:00Z cannot have come
    from a run that closed at 05:15:21Z, and #42's own Step 0/Step 3 self-validation
    runs in precisely this state (its skill: "confirm the skeleton itself validates
    before reading anything else", and a retry after a death is a documented path).
    Refusing there would tell a healthy retry to re-emit real work and report a
    failing validator, so the mtime has to sit at or before the death for that death
    to own the file.
    """
    artifact = _chain(tmp_path, monkeypatch, records=(PREV_RECORD, DEATH_RECORD),
                      mtime=datetime(2026, 10, 11, 11, 20, 0, tzinfo=timezone.utc))
    code, out = _cli(artifact)
    assert code == 0, f"a file newer than the death is not the corpse: {out}"


# ── clause 4: nothing that validates today newly fails ────────────────────────


def test_an_artifact_no_task_declares_validates_exactly_as_today(tmp_path, monkeypatch):
    """No producer, no opinion — even with a death sitting in the records.

    The board exists and #42's records read `failed`, but nothing declares THIS path
    as its output, so there is no producer to interrogate. The corpus's other
    `_pipeline/` artifacts (signals reports, pattern outputs, the legacy `.json`
    handoff) reach the validator on their own skills' instructions, and reddening
    them would be a new failure this item does not ask for.
    """
    artifact = _chain(tmp_path, monkeypatch, records=(DEATH_RECORD,),
                      name="some-other-report.md")
    code, out = _cli(artifact)
    assert code == 0, f"an undeclared artifact must validate as it does today: {out}"


def test_a_producing_task_with_no_run_records_validates_exactly_as_today(
        tmp_path, monkeypatch):
    """A task that has never run has no terminal record, so there is no verdict.

    Same shape as `newest_run_record`'s own "nothing readable" answer,
    `("", None, "")`: absence is not evidence of a failure. A first run's artifact
    would otherwise refuse itself before the task had ever succeeded once.
    """
    artifact = _chain(tmp_path, monkeypatch, records=())
    code, out = _cli(artifact)
    assert code == 0, f"a task that has never run is not a dead run: {out}"


def test_a_run_record_with_no_status_line_is_not_a_verdict(tmp_path, monkeypatch):
    """The unreadable-record half of clause 4, and the rule `newest_run_record` keeps.

    A truncated write leaves front matter with no `status:`, and `_newest_run_record`
    skips exactly those ("an unreadable record answers nothing instead of becoming a
    verdict"). Reading absence as `failed` would let a partial write on the
    scheduler's side stop the nightly chain — so the newest record by instant is here,
    it is unreadable, and the answer must be silence.
    """
    artifact = _chain(tmp_path, monkeypatch, records=(PREV_RECORD, DEATH_RECORD))
    task_runs = tmp_path / "lloyd-data" / "autonomy-runs" / "42"
    newest = task_runs / "run_42_20261011_050938.md"
    assert newest.is_file(), "the fixture's death record is where the node expects it"
    newest.unlink()
    (task_runs / "run_42_truncated.md").write_text(
        "---\ntask_id: 42\nrun_id: run_42_truncated\n"
        f"completed_at: '{(DEATH + timedelta(hours=1)).isoformat()}'\n"
        "---\n\n# Truncated record, newest by instant, no status to read\n",
        encoding="utf-8")
    code, out = _cli(artifact)
    assert code == 0, f"the newest record has no status to report: {out}"


def test_an_unreadable_task_board_validates_exactly_as_today(tmp_path, monkeypatch):
    """`LLOYD_VAULT_ROOT` pointing at nothing is a missing answer, not a refusal.

    `resolve_data_root_for_tree` raises `DataRootMissing` rather than hand back a root
    it cannot prove, and the vault may be unreadable for other reasons too; a
    validator that crashed on the way to the producer question would be worse than one
    that cannot ask it, because Step 1 would read a traceback as a structure failure
    and start editing headings that were never wrong.
    """
    artifact = _chain(tmp_path, monkeypatch, records=(DEATH_RECORD,),
                      board_exists=False)
    monkeypatch.setenv("LLOYD_VAULT_ROOT", str(tmp_path / "no-such-vault"))
    code, out = _cli(artifact)
    assert code == 0, f"an unreadable board must not fail a valid handoff: {out}"


def test_the_structure_findings_and_the_producer_refusal_are_both_reported(
        tmp_path, monkeypatch):
    """One list, not an early return: a dead run's artifact with a missing H2 has TWO
    problems, and reporting one teaches the reader to expect one. Same reasoning as
    #2610's clock check, which is why this goes beside it rather than in front."""
    data = tmp_path / "lloyd-data" / "_pipeline" / "reflection"
    data.mkdir(parents=True)
    artifact = data / "knowledge-handoff-2026-10-11.md"
    artifact.write_text(_handoff(None).replace(f"{vh.REQUIRED_H2[0]}\n", "", 1),
                        encoding="utf-8")
    os.utime(artifact, (SKELETON.timestamp(), SKELETON.timestamp()))
    _board(tmp_path / "obsidian",
           artifact_template=str(data / "knowledge-handoff-{date}.md"))
    _records(tmp_path / "lloyd-data" / "autonomy-runs", DEATH_RECORD)
    monkeypatch.setenv("LLOYD_DATA", str(tmp_path / "lloyd-data"))
    monkeypatch.setenv("LLOYD_VAULT_ROOT", str(tmp_path / "obsidian"))
    code, out = _cli(artifact)
    assert code == 1
    assert "missing required section" in out, f"the structure finding was lost: {out}"
    assert PRODUCER_MARKER in out, f"the producer finding was lost: {out}"


# ── the seam this check has to cross: the two readers of a run record ─────────


def test_the_validator_reads_the_newest_record_ex_as_app_autonomy_does(
        tmp_path, monkeypatch):
    """`validate_handoff.py` cannot import `app.autonomy`, so its reading is pinned to it.

    The nightly chain runs this file with `/usr/bin/python3`, and `app.autonomy`
    imports `agent_mcp._shared`, which needs `mcp` — a venv module. So the join reads
    the same `autonomy-runs/<task>/run_*.md` files here, and this node is what keeps
    that a projection rather than a second opinion: on one fixture tree, the module's
    own `newest_run_record` and this file's reader must agree on the status word, the
    instant and the run id, which is the whole claim the join rests on.
    """
    from app import autonomy

    artifact = _chain(tmp_path, monkeypatch, records=(PREV_RECORD, DEATH_RECORD))
    runs = tmp_path / "lloyd-data" / "autonomy-runs"
    monkeypatch.setattr(autonomy, "AUTONOMY_RUNS_DIR", runs, raising=True)
    status, when, run_id = autonomy.newest_run_record(42)
    assert (status, run_id) == ("failed", "run_42_20261011_050938"), (
        f"the fixture stopped resembling the incident: {status}, {run_id}")
    assert when == DEATH, f"the module read a different instant: {when}"
    findings = vh.check_producer_status(artifact)
    assert len(findings) == 1, (
        f"the two readers disagree: newest_run_record says {status!r} at {when}, "
        f"and check_producer_status answered {findings}")
    assert "run_42_20261011_050938" in findings[0]


# ── clause 5: the only consumer must stop routing this refusal to /dev/null ───


def test_knowledge_write_step_1_sends_a_producer_refusal_to_the_no_artifact_path():
    """SKILL.md:31 used to order the job through EVERY non-zero exit.

    "If it exits non-zero, proceed but flag it clearly in the completion report … Do
    NOT halt" was written for cosmetic 35B drift, and it is kept word-for-word for
    that case. Applied to a producer-status refusal it is what made a code-side check
    worthless: #39 ran the validator on a skeleton and its own note recorded
    "Validation: pass" anyway. So the two classes have to part in the same breath the
    validator is named.
    """
    skill = (Path.home() / "obsidian" / "skills" / "nightly-reflection-knowledge-write"
             / "SKILL.md")
    lines = skill.read_text(encoding="utf-8").splitlines()
    start = next((i for i, line in enumerate(lines)
                  if line.startswith("## Step 1:")), None)
    assert start is not None, "Step 1 of the knowledge-write skill was not found to check"
    end = next((i for i, line in enumerate(lines[start + 1:], start + 1)
                if line.startswith("## ")), len(lines))
    # The whole section, not the heading: Step 1's rule lives on the line AFTER it, so
    # matching on the heading alone would assert about `## Step 1: Load the Handoff
    # Artifact` and nothing else.
    step1 = "\n".join(lines[start:end])
    # The pinned forgiveness, verbatim, still standing for structure and the clock.
    pinned = ('If it exits non-zero, proceed but flag it clearly in the completion '
              'report under "Handoff validation failures". Do NOT halt — the 35B can '
              'produce partially-valid artifacts and we want best-effort processing.')
    assert pinned in step1, (
        "the flag-do-NOT-halt rule for cosmetic drift must survive verbatim: it is "
        "what keeps a partially-valid 35B artifact from stopping the cycle")
    # …and the carve-out the incident cost 21 hours of unread handoff to ask for.
    assert PRODUCER_MARKER in step1, (
        "Step 1 never names the producer-status class, so a code-side refusal is "
        "routed to the same /dev/null as every other one")
    assert step1.index(pinned) < step1.index(PRODUCER_MARKER), (
        "the carve-out has to follow the rule it carves out of, or the reader meets "
        "'proceed' first and stops there")
    tail = step1[step1.index(PRODUCER_MARKER):]
    assert "proceed" not in tail, (
        f"the producer-status refusal is still being told to proceed: {tail[:300]}")
    assert "error-note" in tail, (
        f"the refusal is not routed to the no-handoff path: {tail[:300]}")
    assert "run record" in tail, (
        f"the completion note is not told to name the record: {tail[:300]}")
