"""ALERT.md's `written:` stamp has to name the zone it is in (#1912).

`Notifier._alert_file` overwrites `~/.local/state/lloyd-guardian/ALERT.md` from
the guardian process, and until this item the stamp it wrote there was
`datetime.now().isoformat()` — naive, on a box that runs at -0700. The file is
the artefact, and an unmarked hour in the artefact a human reads during an
incident is not a style problem: this repo already holds the two opposite
readings of that exact field in its own comments.
`scripts/side_effect_traffic_census.py::parse_stamp` documents that "a naive
stamp is therefore read as *local* time" and names `ALERT.md`'s `written:` field
as one of the three places that skew was logged; `app/skill_telemetry.py` reads
a naive stamp as UTC three lines from a comment that names the same field as the
cost of doing so. Both cite the field; they resolve it seven hours apart.

Marking the zone settles it without moving a digit, which is the shape #1808
landed for the spoken channel's log (`speak.py`'s `%z`, pinned by
`test_the_offset_never_moves_the_wall_clock_reading` in
`tests/test_guardian_speak.py`). `datetime.fromtimestamp` and
`datetime.now()` agree on the wall clock, so the digits this file compares are
taken from the real `datetime` class and not from the writer's own chain.

The process boundary under test is ALERT.md itself: `_alert_file` runs in
`lloyd-guardian.service` off the snapshot staged into
`~/.local/state/lloyd-guardian/bin`, and whatever it writes is read later, in a
different process, by a person or by a signal pass. Every node here therefore
re-reads the bytes from disk and parses them with the API a downstream reader
uses (`datetime.fromisoformat`), rather than inspecting anything in memory.
"""

from __future__ import annotations

import datetime
import os
import re
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARDIAN_DIR = ROOT / "agent-services" / "guardian"
for _p in (str(ROOT), str(GUARDIAN_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import notify  # noqa: E402

LEVEL = "error"
TITLE = "Runtime data is being written into the code tree"
BODY = "3 stray paths under `data/` — move `eval/baselines` out (twice)"

_REAL_DATETIME = datetime.datetime
#: An ISO-8601 numeric UTC offset, which is the only thing this change may add.
_OFFSET_SUFFIX = re.compile(r"[+-]\d{2}:\d{2}$")
#: What the stamp's digits must look like when read back: date, time to the
#: microsecond, nothing converted.
_DIGITS = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?$")


def _notifier(state_dir: Path) -> notify.Notifier:
    """A Notifier whose whole world is `state_dir`.

    `external=False` is what keeps a node here out of the room: journal, toast,
    speech, the daily note and the board are all behind that gate
    (`notify.py:191-207`), so the only channels that fire are the ledger and
    ALERT.md, both of which this helper points at the tmp dir.
    """
    return notify.Notifier(ledger=state_dir / "ledger.jsonl",
                           state_dir=state_dir,
                           vault_root=str(state_dir / "vault"),
                           external=False)


def _written_value(state_dir: Path) -> str:
    """The `written:` value of the ALERT.md on disk, read the way a reader reads it.

    The file's own text plus `datetime.fromisoformat` and nothing else — the
    claim under test is what a later process can conclude from these bytes.
    """
    lines = (state_dir / "ALERT.md").read_text(encoding="utf-8").splitlines()
    stamps = [ln for ln in lines if ln.startswith("written: ")]
    assert len(stamps) == 1, f"expected exactly one `written:` line, got {stamps!r}"
    return stamps[0].split("written: ", 1)[1].strip()


def _clock_frozen_at(moment: float):
    """A `datetime` name for `notify.py` whose `now()` is pinned to one instant.

    Subclasses the real class, so any other `datetime` call a later edit reaches
    through that name still behaves; answers `now()` via `fromtimestamp`, the
    real class's other entry point to the same instant, so the expected digits
    below come from `datetime` itself and share no code with the writer's
    `astimezone().isoformat()` chain.
    """
    class _Frozen(_REAL_DATETIME):
        @classmethod
        def now(cls, tz=None):  # noqa: ANN001 - mirrors the real signature
            return _REAL_DATETIME.fromtimestamp(moment, tz)
    return _Frozen


def test_the_alert_stamp_names_its_own_zone(tmp_path):
    """Clause 1: a machine-facing stamp must say which zone it is in.

    The ambiguity is not hypothetical — it is the disagreement sitting in this
    repo's own comments, and the live witness at triage is
    `written: 2026-09-30T07:04:08.021523` on a box where `date +%z` is -0700.
    """
    assert _notifier(tmp_path)._alert_file(LEVEL, TITLE, BODY) is True

    raw = _written_value(tmp_path)
    parsed = datetime.datetime.fromisoformat(raw)
    assert parsed.utcoffset() is not None, (
        f"{raw!r} carries no offset, so every later reader has to guess whether "
        "it is local or UTC — which is the seven-hour disagreement "
        "`scripts/side_effect_traffic_census.py` and `app/skill_telemetry.py` "
        "currently document about this one field")


def test_marking_the_zone_never_moves_the_wall_clock_reading(tmp_path, monkeypatch):
    """Clause 2: marking the zone is not the same act as converting to UTC.

    `datetime.now(timezone.utc).isoformat()` would satisfy clause 1 and silently
    shift every alert by seven hours — and the whole value of ALERT.md is that a
    person lines its stamp against their memory of the evening. So the instant is
    frozen, the digits are compared against the LOCAL reading of that instant,
    and the UTC reading of the same instant is asserted to be DIFFERENT, so the
    comparison is known to distinguish the two.
    """
    moment = time.time()
    frozen = _clock_frozen_at(moment)
    monkeypatch.setattr(notify, "datetime", frozen)
    assert abs(frozen.now() - _REAL_DATETIME.now()) < datetime.timedelta(seconds=2), (
        "the frozen clock does not answer what `datetime.now()` would have, so "
        "the digit comparison below would be grading a fiction")

    want_digits = _REAL_DATETIME.fromtimestamp(moment).isoformat()
    assert _DIGITS.match(want_digits), f"unexpected naive reading {want_digits!r}"

    assert _notifier(tmp_path)._alert_file(LEVEL, TITLE, BODY) is True

    raw = _written_value(tmp_path)
    parsed = datetime.datetime.fromisoformat(raw)
    assert raw.startswith(want_digits) and _DIGITS.match(raw[:len(want_digits)]), (
        f"the stamp reads {raw!r} but the local wall clock at that moment was "
        f"{want_digits!r} — the writer must keep writing local time, down to the "
        "microseconds, and only MARK the zone")
    assert _OFFSET_SUFFIX.search(raw), (
        f"{raw!r} has no trailing +HH:MM/-HH:MM offset suffix")
    want_offset = _REAL_DATETIME.fromtimestamp(
        moment, datetime.timezone.utc).astimezone().utcoffset()
    assert parsed.utcoffset() == want_offset, (
        f"{raw!r} names the wrong offset for this machine (wanted {want_offset})")

    if want_offset:  # on a UTC box the two readings coincide and prove nothing
        utc_digits = _REAL_DATETIME.fromtimestamp(
            moment, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        assert parsed.strftime("%Y-%m-%dT%H:%M:%S") != utc_digits, (
            f"the stamp reads the UTC wall clock ({utc_digits}) on a box whose "
            "offset is not zero: the digits moved")


def test_the_alert_file_return_and_layout_are_unchanged(tmp_path):
    """Clause 3's first half: the write keeps its whole existing contract.

    Byte equality of the entire file, with only the stamp value substituted —
    that pins the `# {title}` heading, the `level:` line, the single `written:`
    line, the blank line, the body verbatim and the trailing newline in one
    assertion, which is what `tests/test_iv_metrics_series.py:1631` and
    `tests/test_guardian_speak.py:838` lean on when they assert ALERT.md exists
    or must NOT exist. A successful call still returns True.
    """
    stamp = _written_value_of_a_write(tmp_path)

    assert (tmp_path / "ALERT.md").read_text(encoding="utf-8") == (
        f"# {TITLE}\n\nlevel: {LEVEL}\nwritten: {stamp}\n\n{BODY}\n"), (
        "the alert file's layout moved; only the `written:` value was allowed to "
        "gain an offset")


def _written_value_of_a_write(state_dir: Path) -> str:
    """Write once, assert the call reported success, hand back the stamp."""
    assert _notifier(state_dir)._alert_file(LEVEL, TITLE, BODY) is True
    return _written_value(state_dir)


def test_the_alert_file_is_still_last_writer_wins(tmp_path):
    """Clause 3's other half: one `write_text`, so the last alert survives.

    `notify.py:200-202`'s comment and `tests/test_guardian_predicates.py:931`
    both describe ALERT.md as last-writer-wins, and #775's triage rests on it.
    This change edits the format string inside that call, so the overwrite
    behaviour has to still be the overwrite behaviour: one heading, one
    `written:` line, the second alert's body.
    """
    assert _notifier(tmp_path)._alert_file(LEVEL, TITLE, BODY) is True
    second = "supervisord is unreachable"
    assert _notifier(tmp_path)._alert_file("critical", "Guardian watchdog", second) is True

    text = (tmp_path / "ALERT.md").read_text(encoding="utf-8")
    assert text.splitlines()[0] == "# Guardian watchdog", (
        f"the second alert did not overwrite the first: {text!r}")
    assert len([ln for ln in text.splitlines()
                if ln.startswith("written: ")]) == 1, f"two stamps in one file: {text!r}"
    assert BODY not in text, f"the superseded alert survived: {text!r}"


def test_an_unwritable_state_dir_still_returns_false_and_raises_nothing(tmp_path):
    """Clause 3's last half: `except Exception: return False` still holds.

    Two ways of making the write fail, because they fail at different places — a
    state dir that is a regular FILE makes `mkdir` raise `FileExistsError`
    whatever user runs the suite, and a directory without write permission makes
    the `write_text` fail. The permission half is skipped rather than passed
    vacuously under root, where mode bits are advisory.
    """
    as_file = tmp_path / "not-a-dir"
    as_file.write_text("in the way", encoding="utf-8")

    assert _notifier(as_file)._alert_file(LEVEL, TITLE, BODY) is False
    assert as_file.read_text(encoding="utf-8") == "in the way", (
        "the blocked write clobbered whatever was in its place")

    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root ignores mode bits, so the permission case proves nothing")
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        assert _notifier(locked)._alert_file(LEVEL, TITLE, BODY) is False
        assert not (locked / "ALERT.md").exists()
    finally:
        locked.chmod(0o700)


# ── clause 4: the comments that described this field ────────────────────────
#: The two modules the item names as this field's readers-by-comment.
_READER_COMMENTS = ("app/skill_telemetry.py", "scripts/side_effect_traffic_census.py")

#: The claims #1912 falsifies, phrased as the comments phrased them. Past tense
#: is allowed on purpose — `was the skew's third member until #1912` is true and
#: stays; what may not survive is a sentence describing a field that now carries
#: an offset as though it did not.
_FALSIFIED_CLAIM = re.compile(
    r"offset[-\s]?less|offsetless|unmarked|without (?:an )?offset|"
    r"carries no offset|has no offset|known cost",
    re.IGNORECASE)
#: A field still listed as a live hazard rather than a settled one.
_STILL_A_HAZARD = re.compile(r"(?:already|still) logged", re.IGNORECASE)


def _windows_around(text: str, needle: str, radius: int = 280) -> list[str]:
    """Every occurrence of `needle`, with the prose around it.

    A window rather than the enclosing line because both comments wrap, and a
    window rather than the whole docstring because the surrounding text is about
    transcript stamps, where "naive" is still exactly right.
    """
    return [text[max(0, m.start() - radius): m.end() + radius]
            for m in re.finditer(re.escape(needle), text)]


def _scan(text: str) -> list[str]:
    """The falsified claims this text makes about `ALERT.md`."""
    out: list[str] = []
    for window in _windows_around(text, "ALERT.md"):
        for pattern, why in ((_FALSIFIED_CLAIM, "described as carrying no offset"),
                             (_STILL_A_HAZARD, "still listed as a live skew hazard")):
            hit = pattern.search(window)
            if hit:
                start = max(0, hit.start() - 70)
                out.append(f"{why}: …{window[start:hit.end() + 70]}…")
    return out


def test_no_reader_comment_still_calls_the_alert_stamp_offset_less():
    """Clause 4: a comment that describes a field wrongly is part of the diff.

    After the change, `app/skill_telemetry.py`'s "the known cost is the
    offset-less `ALERT.md` `written:` field" describes a field that no longer
    exists, and the census's "the same skew this repo already logged for …
    `ALERT.md`'s `written:` field" keeps it in the present-tense list. Both are
    the only documentation of this field's shape anywhere in the tree, so they
    are graded here rather than left to rot beside the fixed writer.
    """
    offenders = []
    for rel in _READER_COMMENTS:
        offenders.extend(f"{rel}: {finding}"
                         for finding in _scan((ROOT / rel).read_text(encoding="utf-8")))
    assert not offenders, (
        "a reader comment still describes ALERT.md's `written:` stamp as the "
        "item's fix left it:\n" + "\n".join(offenders))


def test_that_reader_comment_scan_fires_on_the_bytes_it_replaces():
    """The control clause 4 needs: the scan above is not vacuous.

    Quoted verbatim from HEAD before this round — these are the exact comment
    bodies the change replaces, and a scan that cannot catch them would pass on
    any text at all.
    """
    pre_fix = {
        "app/skill_telemetry.py":
            "        # A naive stamp is read as UTC, the way every other reader on this box\n"
            "        # reads one. The alternative — refusing — would drop rows the writer\n"
            "        # honestly produced; the known cost is the offset-less `ALERT.md`\n"
            "        # `written:` field, which is not this file's format.\n",
        "scripts/side_effect_traffic_census.py":
            "    A naive stamp is therefore read as *local*\n"
            "    time, which is the only reading that agrees with its writer; treating it as\n"
            "    UTC shifts the whole transcript corpus by the box's offset (here 7-8 hours),\n"
            "    which is the same skew this repo already logged for `git log --since`,\n"
            "    `find -newermt` and `ALERT.md`'s `written:` field.\n",
    }
    for rel, old in pre_fix.items():
        assert _scan(old), (
            f"the scan never fires on {rel}'s pre-#1912 comment, so "
            "test_no_reader_comment_still_calls_the_alert_stamp_offset_less "
            "could never fail")


# ── #1967 clause 5: the retraction's stamp obeys the same rule ───────────────
#
# #1967 gives `resolve()` a second write into ALERT.md — a `cleared:` line —
# and a second timestamp is a second chance to write the ambiguous field
# #1912 exists to forbid. The retraction stamp matters MORE than the alarm
# stamp for the item's actual harm: an agent inheriting the file ages the
# retraction to decide whether the alarm is live, and a naive stamp there is
# two readings seven hours apart exactly where the decision is made. The
# `cleared: ` prefix is matched as a literal rather than imported from
# `notify` so these nodes grade the file's format, not a constant the fix
# was free to define however it liked.


def _cleared_value(state_dir: Path) -> str:
    """The `cleared:` value of the ALERT.md on disk, read the way a reader reads it.

    The stamp is the line's FIRST space-delimited token; what follows it is
    the retraction note in prose, which a reader ages separately from the
    instant — so a node that parsed the whole line with `fromisoformat` would
    be asserting a format the writer never promised.
    """
    lines = (state_dir / "ALERT.md").read_text(encoding="utf-8").splitlines()
    cleared = [ln for ln in lines if ln.startswith("cleared: ")]
    assert len(cleared) == 1, f"expected exactly one `cleared:` line, got {cleared!r}"
    return cleared[0].split("cleared: ", 1)[1].split(" ")[0]


def test_the_cleared_stamp_names_its_own_zone(tmp_path):
    """Clause 5: the retraction's timestamp carries a numeric UTC offset.

    Alert through the shipped `_alert_file`, retract through the shipped
    `resolve`, then age the retraction with `datetime.fromisoformat` — the
    same call `app/skill_telemetry.py` makes for a naive stamp and the source
    of the two-readings disagreement #1912 documents. Fails before #1967
    because `resolve()` never wrote this line at all.
    """
    n = _notifier(tmp_path)
    assert n._alert_file(LEVEL, TITLE, BODY) is True
    assert n.resolve(TITLE, "cause removed") is True

    raw = _cleared_value(tmp_path)
    parsed = datetime.datetime.fromisoformat(raw)
    assert parsed.utcoffset() is not None, (
        f"{raw!r} carries no offset, so a reader ageing the retraction — the "
        "act #1967 exists to make safe — inherits the same seven-hour "
        "ambiguity #1912 closed for `written:`")


def test_the_cleared_stamp_keeps_the_local_wall_clock_reading(tmp_path, monkeypatch):
    """Clause 5's other half: marking the zone again moves no digits.

    Same frozen-instant shape as
    `test_marking_the_zone_never_moves_the_wall_clock_reading`: the alarm and
    its retraction are written inside one frozen instant, so the `cleared:`
    stamp must answer to that instant in LOCAL digits with the zone merely
    named — and must equal the `written:` stamp of the same call chain,
    because a retraction that converted to UTC would read as clearing the
    incident seven hours before it was raised on a -0700 box.
    """
    moment = time.time()
    frozen = _clock_frozen_at(moment)
    monkeypatch.setattr(notify, "datetime", frozen)
    n = _notifier(tmp_path)
    assert n._alert_file(LEVEL, TITLE, BODY) is True
    assert n.resolve(TITLE, "cause removed") is True

    raw = _cleared_value(tmp_path)
    want_digits = _REAL_DATETIME.fromtimestamp(moment).isoformat()
    assert raw.startswith(want_digits) and _DIGITS.match(raw[:len(want_digits)]), (
        f"the retraction stamp reads {raw!r} but the local wall clock at that "
        f"moment was {want_digits!r} — write local time and only MARK the zone")
    assert _OFFSET_SUFFIX.search(raw), f"{raw!r} has no trailing ±HH:MM offset"
    assert raw == _written_value(tmp_path), (
        "the two stamps of one frozen instant disagree; both must answer to "
        "the same local reading with the same zone")
