"""ALERT.md's one slot has to say which incident it buried (#3120).

`Notifier._alert_file` writes `~/.local/state/lloyd-guardian/ALERT.md` with one
`write_text`: one incident, last writer wins. Until this item it read nothing
before overwriting — no level comparison (the write path has never had one), no
seal, no tombstone for whatever it covered. The live witness at filing, read
directly off the file at 2026-10-21T17:41Z: the H1 held `Runtime data is being
written into the code tree` at `level: error` with `grep -c '^cleared:'` → 0,
where earlier that same morning the H1 had held `Lloyd data root damaged —
promotions halted` at `level: critical`. The critical alarm was neither sealed
nor retracted — it was DISPLACED by a lower-severity alert, so the promotion
halt it asserted (raised 2026-10-20T21:24:14Z, adjudicated a false positive and
lifted 21:28:35Z) stopped existing on the surface an agent reads first. The
displacement reproduced again before this round opened: H1 `Vault snapshots are
not arriving` at `level: error`, `grep -c '^cleared:'` → 0, the runtime-data
alarm's copy gone.

Displacement is not retraction, and #3106 does not cover it.
`guardian.py::_retract_tripwire_alert` returns as soon as
`alert_file_incident()[0] != title`, and its own docstring concedes that once
another incident has taken the slot it "has no index to the stale section".
Fixing the titles a resolver can reach therefore adds no seal to an incident
that was buried: the two failures are the retraction route and the
displacement route.

What this file pins, and the shape the fix took:

* the trace sits INSIDE ALERT.md's header as one `displaced:` key naming the
  buried title, its `level=`, its OWN `written=` stamp and whether it was `open`
  or `cleared` — so the file a person opens answers "what did this overwrite,
  and was it still open?" without any state file (#3106's daily-note surface
  cannot be retracted either, so prose there is not the answer);
* the outgoing stamp rides inside that key as `written=` and never as a second
  `written:` line, because `tests/test_guardian_alert_stamp.py:204-205` pins
  that the file carries exactly one `written:` line and a reader ages the LIVE
  alarm, not the one it buried;
* the chain of burials is durable in the ledger as one `alert_displaced` row
  per displacement — the in-file key is one generation deep by construction —
  and no new state file exists to bound;
* a write with nothing to bury is byte-identical to the pre-change layout, so
  the hourly re-alert of a standing incident gains no key.

The process boundary under test is the file and the ledger themselves:
`_alert_file` runs in `lloyd-guardian.service` off the snapshot staged into
`~/.local/state/lloyd-guardian/bin`, and what it writes is read later, in a
different process, by a person or by a signal pass. Every node here therefore
judges bytes re-read from disk — parsed with `Notifier._alert_file_split`, the
same parser a reader uses — and JSONL rows re-read from the ledger, never
anything left in memory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUARDIAN_DIR = ROOT / "agent-services" / "guardian"
for _p in (str(ROOT), str(GUARDIAN_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import notify  # noqa: E402

#: The displaced alarm, quoted from the 2026-10-20 incident this item is about.
DAMAGED = "Lloyd data root damaged — promotions halted"
#: The alert that buried it, and the one that buried the runtime-data alarm's
#: copy on 2026-10-22. A lower-severity watchdog, which is the case the item
#: says nobody had ruled: the write path compares no level, so this displaces a
#: `critical` exactly as easily as an `error` did.
SNAPSHOTS = "Vault snapshots are not arriving"
#: #1536's coalescing incident: the hourly check re-alerts the SAME title, which
#: is one incident outliving many checks and must never read as a displacement.
RUNTIME = "Runtime data is being written into the code tree"
BODY = "no snapshot newer than 26 h; `btrfs send` last ran before the last settle"
BODY2 = "the vault filesystem has not gained a snapshot in over a day"
NOTE = ("the halt was adjudicated a false positive and lifted — no promotion is "
        "blocked and the instructions above are stale")


def _notifier(tmp_path: Path) -> notify.Notifier:
    """A Notifier whose whole world is `tmp_path`.

    `external=False` keeps every node out of the room and off the board: the
    journal, toast, speech, daily note and backlog task all sit behind that
    gate, so the only surfaces these nodes write are the ledger and ALERT.md —
    the two the displacement fix touches, and the same shape
    `tests/test_guardian_alert_stamp.py::_notifier` runs in.
    """
    return notify.Notifier(ledger=tmp_path / "ledger.jsonl",
                           state_dir=tmp_path,
                           vault_root=str(tmp_path / "vault"),
                           backend_url="http://127.0.0.1:1",
                           external=False)


def _text(tmp_path: Path) -> str:
    return (tmp_path / "ALERT.md").read_text(encoding="utf-8")


def _lines(tmp_path: Path) -> list[str]:
    return _text(tmp_path).splitlines()


def _written_values(tmp_path: Path) -> list[str]:
    """Every `written:` header line's value, oldest position first.

    Read as a list rather than a single value on purpose: #1912's rule is that a
    reader ages an alarm from ONE stamp, and a displacement that smuggled the
    outgoing stamp out as a second `written:` line would produce two readable
    alarms from one file.
    """
    return [ln.split("written: ", 1)[1].strip()
            for ln in _lines(tmp_path) if ln.startswith("written: ")]


def _header_keys(tmp_path: Path) -> list[str]:
    """The file's header keys as the shipped parser reads them.

    A trace that only a regex in this test file can find is not a trace a
    reader sees, so the key has to survive `Notifier._alert_file_split` — the
    same parse `_alert_file_retract` and `alert_file_incident` use.
    """
    split = notify.Notifier._alert_file_split(_text(tmp_path))
    assert split is not None, f"ALERT.md is no longer the layout: {_text(tmp_path)!r}"
    return list(split[1])


def _displaced(tmp_path: Path) -> str:
    """The file's single `displaced:` header key, or fail.

    Asserted single: a key that accumulated across displacements would be an
    unbounded growth in the file every alert rewrites, and the chain belongs in
    the ledger, which is already archived.
    """
    found = [k for k in _header_keys(tmp_path)
             if k.startswith(f"{notify.ALERT_DISPLACED_PREFIX} ")]
    assert len(found) == 1, (
        f"expected exactly one {notify.ALERT_DISPLACED_PREFIX!r} header key, "
        f"got {found!r} in {_text(tmp_path)!r}")
    return found[0]


def _rows(tmp_path: Path, event: str) -> list[dict]:
    """Ledger rows of one `event` name, re-read off disk as JSON."""
    path = tmp_path / "ledger.jsonl"
    if not path.is_file():
        return []
    return [r for r in (json.loads(ln) for ln in
                        path.read_text(encoding="utf-8").splitlines() if ln.strip())
            if r.get("event") == event]


def _raise(n: notify.Notifier, level: str, title: str, body: str) -> None:
    """Fan out one alarm through the shipped `alert()` and assert it landed.

    `alert()` and not `_alert_file()` where a node can use it, because the seam
    under test is the write path a production tick takes — including that the
    displacement is recorded ABOVE the `external` gate, where the write is.
    """
    assert n.alert(level, title, body)["alert_file"] is True


# ── the trace in the file a reader opens ──────────────────────────────────


def test_a_later_alert_marks_the_unsealed_alarm_it_buried(tmp_path):
    """The buried incident is named, at its level, and marked as still open.

    The breaking input is the production pair from the item: a `critical`
    `Lloyd data root damaged — promotions halted` alarm, then an `error`
    `Vault snapshots are not arriving` alarm. Before the fix ALERT.md held the
    second and said nothing of the first, which is how the 2026-10-20 promotion
    halt vanished while still unsealed. The key must be a header key (a reader's
    parser sees it) and must not be alarm prose (the alarm's imperative text
    must not survive as a live directive, which is #1967's rule).
    """
    n = _notifier(tmp_path)
    _raise(n, "critical", DAMAGED, BODY)
    _raise(n, "error", SNAPSHOTS, BODY2)

    assert _lines(tmp_path)[0] == f"# {SNAPSHOTS}", (
        "the later alert must still take the slot — refusing the overwrite "
        "would keep a stale alarm reading as live, the defect #1967 closed")
    key = _displaced(tmp_path)
    assert DAMAGED in key, f"the buried incident is not named: {key!r}"
    assert "level=critical" in key, (
        f"{key!r} hides the severity of what it buried, so a reader cannot tell "
        "that an `error` alert displaced a `critical` alarm")
    assert key.endswith(" open"), (
        f"{key!r} does not say the buried incident was still open, which is the "
        "difference between a displaced alarm and a displaced tombstone")


def test_the_buried_stamp_rides_inline_and_the_file_keeps_one_written_line(tmp_path):
    """The outgoing stamp is copied byte-for-byte, inline, as `written=`.

    Two halves, both load-bearing. (a) The burial must say WHEN the alarm it
    buried fired, from that alarm's own `written:` value — the reader's question
    about a displaced alarm is the age of the burial, not of the alert that made
    it. (b) It must say so inside the new key: a second `written:` line would
    break `tests/test_guardian_alert_stamp.py:204-205` and leave the file with
    two alarms a reader could age. The live stamp is also asserted to be the NEW
    one, so copying the old value never falsified it.
    """
    n = _notifier(tmp_path)
    _raise(n, "critical", DAMAGED, BODY)
    first = _written_values(tmp_path)
    assert len(first) == 1, f"one alarm, one stamp: {first!r}"
    _raise(n, "error", SNAPSHOTS, BODY2)

    values = _written_values(tmp_path)
    assert len(values) == 1, (
        f"the displaced stamp leaked out as a second `written:` line: {values!r}")
    assert values[0] != first[0], (
        "the surviving alarm kept the displaced alarm's stamp, so a reader would "
        f"age the live incident from {first[0]!r}")
    assert f"written={first[0]}" in _displaced(tmp_path), (
        f"the key does not carry the buried alarm's own stamp: {_displaced(tmp_path)!r}")


def test_displacing_a_sealed_incident_marks_it_cleared(tmp_path):
    """A displaced tombstone still reads as a tombstone, not as a lost incident.

    The second consequence the item records: the same `write_text` also destroys
    an incident that had been cleanly sealed, so ALERT.md lost the four
    `cleared:` lines' worth of information that the runtime-data incident had
    closed. The seal itself is necessarily overwritten — one slot — so what the
    fix owes the reader is the distinction, and `cleared` at the end of the key
    is that distinction: an incoming `resolve()` on the buried title then has
    nothing left to do, and a reader must not conclude an open alarm vanished.
    """
    n = _notifier(tmp_path)
    _raise(n, "critical", DAMAGED, BODY)
    assert n.resolve(DAMAGED, NOTE) is True
    sealed = [k for k in _header_keys(tmp_path)
              if k.startswith(f"{notify.ALERT_CLEARED_PREFIX} ")]
    assert len(sealed) == 1, f"the first incident is not sealed: {sealed!r}"

    _raise(n, "error", SNAPSHOTS, BODY2)

    key = _displaced(tmp_path)
    assert key.endswith(" cleared"), (
        f"{key!r} reports a sealed incident as though it were an open alarm "
        "displaced by the new one")
    assert not [k for k in _header_keys(tmp_path)
                if k.startswith(f"{notify.ALERT_CLEARED_PREFIX} ")], (
        "the new alarm's file claims a cleared incident, which would leave the "
        "hourly all-clear with something to seal that it did not write")

    # The seal's TEXT exists nowhere else — the `alert` row logged when the
    # incident fired carries its own body, not the retraction note — so the
    # overwrite destroying it is a real loss, and the ledger row is what keeps
    # the answer to "how was it closed?" after the tombstone is gone.
    sealed_rows = _rows(tmp_path, "alert_displaced")
    assert len(sealed_rows) == 1, sealed_rows
    assert sealed_rows[0].get("sealed") is True, sealed_rows[0]
    assert NOTE in str(sealed_rows[0].get("cleared", "")), (
        f"the seal's own text did not survive the burial: {sealed_rows[0]!r}")


def test_a_same_title_realert_records_no_displacement(tmp_path):
    """One incident outliving many checks is not a displacement.

    `RUNTIME` is the alert the hourly watchdog re-fans out with `coalesce=True`,
    and `alert()` reaches `_alert_file` on every one of those checks: if the key
    keyed on "the file already exists", every hourly re-alert would rewrite the
    header and the file would read as a chain of displacements that never
    happened. The breaking input is therefore the SAME title twice, into a state
    dir where nothing has been buried yet — the case where a re-alert must
    produce a key is a different one, and
    `test_the_burial_trace_survives_the_burying_incident_s_own_realert` owns it.
    """
    n = _notifier(tmp_path)
    _raise(n, "error", RUNTIME, BODY)
    _raise(n, "error", RUNTIME, BODY2)

    assert _lines(tmp_path)[0] == f"# {RUNTIME}"
    assert [k for k in _header_keys(tmp_path)
            if k.startswith(f"{notify.ALERT_DISPLACED_PREFIX} ")] == [], (
        "a re-alert of the standing incident recorded a displacement of itself")
    assert _rows(tmp_path, "alert_displaced") == [], (
        "the ledger was told a different incident was buried when the same one "
        "was merely re-asserted")


def test_the_burial_trace_survives_the_burying_incident_s_own_realert(tmp_path):
    """The check that buries nobody must still keep the burial it already shows.

    This is the production shape of the 2026-10-21 displacement: ONE standing
    `coalesce=True` incident re-alerting hourly, whose SECOND copy buried the
    unsealed `critical` alarm. So the same-title re-alert — the write that must
    record no new displacement — is also the write most likely to arrive an hour
    after a real burial. Dropping the key there would erase the in-file trace of
    that burial within one check of making it and leave the ledger holding it
    alone, which is the half a reader of ALERT.md cannot reach.

    Three writes: the alarm, the alert that buries it, a re-alert of that alert.
    The carried key must name the first alarm with its OWN stamp unchanged —
    carrying it forward is not re-stamping it — exactly once, while the live
    `written:` line moves on, so the file still ages the live incident (#1912).
    """
    n = _notifier(tmp_path)
    _raise(n, "critical", DAMAGED, BODY)
    buried = _written_values(tmp_path)[0]
    _raise(n, "error", SNAPSHOTS, BODY2)
    carried = _displaced(tmp_path)
    live = _written_values(tmp_path)[0]

    _raise(n, "error", SNAPSHOTS, BODY)

    assert _displaced(tmp_path) == carried, (
        "a re-alert of the standing incident erased the record of the incident "
        "it had buried an hour earlier")
    assert f"written={buried}" in _displaced(tmp_path), (
        "the carried key lost the buried alarm's own stamp")
    assert _written_values(tmp_path)[0] != live, (
        "the live alarm's stamp did not move, so the file is ageing the "
        "displacement instead of the incident in the slot")


# ── the durable half ──────────────────────────────────────────────────────


def test_every_displacement_appends_one_ledger_event_and_no_extra_alert_row(tmp_path):
    """The chain of burials is durable, and is not a re-logged alert.

    The in-file key is one generation deep: the NEXT displacement overwrites it.
    The ledger is where the whole chain survives — `gstate.append_event` fsyncs,
    and `scripts/groundskeeper/retention-sweep.py:113` archives
    `promotions.jsonl` monthly rather than deleting it. The event gets its own
    name because `app/failure_ledger.py:367` builds per-title onset history out
    of `event == "alert"` rows, so writing the buried alarm as another alert
    would invent a firing that never happened. Two `alert` rows for two alerts
    and one `alert_displaced` row is the whole claim.
    """
    n = _notifier(tmp_path)
    _raise(n, "critical", DAMAGED, BODY)
    first = _written_values(tmp_path)[0]
    _raise(n, "error", SNAPSHOTS, BODY2)

    assert len(_rows(tmp_path, "alert")) == 2, (
        "the displacement re-logged an alert, which is the one thing the item "
        "rules out: the alerts were already recorded")
    events = _rows(tmp_path, "alert_displaced")
    assert len(events) == 1, (
        f"the ledger records {len(events)} displacements for one burial: {events!r}")
    row = events[0]
    assert row.get("title") == DAMAGED, row
    assert row.get("level") == "critical", row
    assert row.get("written") == first, (
        f"the row ages the burial from {row.get('written')!r}, not from the "
        f"buried alarm's own stamp {first!r}")
    assert row.get("sealed") is False, row
    assert row.get("displaced_by") == SNAPSHOTS, row


def test_a_displacement_creates_no_new_state_file(tmp_path):
    """The trace costs no new store, so nothing new grows unbounded.

    The item's own constraint: a JSONL journal beside ALERT.md would be a fresh
    unbounded store needing an entry in the groundskeeper sweep. The fix rides
    the two files that already exist — ALERT.md's header and `ledger.jsonl`,
    which that sweep already archives — so the state dir after a displacement is
    exactly those two files. This node fails the moment a third appears without
    a retention entry, which is the shape the growth alarm catches too late.
    """
    n = _notifier(tmp_path)
    _raise(n, "critical", DAMAGED, BODY)
    _raise(n, "error", SNAPSHOTS, BODY2)

    assert sorted(p.name for p in tmp_path.iterdir()) == ["ALERT.md",
                                                          "ledger.jsonl"], (
        "a new state file appeared; the groundskeeper store list in "
        "scripts/groundskeeper/retention-sweep.py is what bounds one")


# ── what the fix must not break ───────────────────────────────────────────


def test_a_write_with_nothing_to_bury_is_byte_identical_to_the_old_layout(tmp_path):
    """No predecessor means the file's bytes do not move at all.

    Byte equality of the whole file with only the stamp substituted, exactly as
    `tests/test_guardian_alert_stamp.py:176-177` asserts it. That node writes
    into an empty state dir and must keep passing untouched: the displacement
    key is conditional, so the single-incident case — the common case, and the
    one `tests/test_iv_metrics_series.py` and `tests/test_guardian_speak.py`
    lean on — is unchanged, including its trailing newline.
    """
    assert _notifier(tmp_path)._alert_file("error", RUNTIME, BODY) is True

    stamp = _written_values(tmp_path)[0]
    assert _text(tmp_path) == (
        f"# {RUNTIME}\n\nlevel: error\nwritten: {stamp}\n\n{BODY}\n"), (
        "the layout of an alert file with nothing to displace moved; only a "
        "displaced header key was allowed to be new")


def test_a_predecessor_written_by_anything_still_yields_one_header_line(tmp_path):
    """A colon-bearing title cannot forge a header key of its own.

    ALERT.md is not only written by this module: `alert_file_incident` exists
    because the file outlives its writer and a person may have edited it. The
    predecessor here is hand-written with a title containing `: ` and a
    `written:` value that is not a stamp — the shape that would let the copied
    title start a new key line if the key were composed by concatenation into
    the wrong place. The check is that the result is still one `displaced:` key,
    still exactly one `written:` line, and still parseable.
    """
    (tmp_path / "ALERT.md").write_text(
        "# Odd title: halted\n\nlevel: critical\nwritten: never\n\nact now\n",
        encoding="utf-8")

    assert _notifier(tmp_path)._alert_file("error", SNAPSHOTS, BODY) is True

    assert _displaced(tmp_path) == (
        "displaced: Odd title: halted level=critical written=never open"), (
        "the copied fields moved out of the one key line")
    assert len(_written_values(tmp_path)) == 1, _written_values(tmp_path)


def test_a_predecessor_missing_its_level_and_stamp_is_still_named(tmp_path):
    """A header with neither `level:` nor `written:` buries with `unknown`.

    `_alert_file_key` answers `unknown` for both fields, and the branch exists
    because a half-written ALERT.md is the realistic predecessor — the write is
    not atomic, and an interrupted one leaves an H1 and part of a header. A
    missing field must not read as NOTHING BURIED: the title alone still says an
    alarm went missing, which is this item's defect. The input is therefore an H1
    plus one unrelated header key and neither of the two the key quotes — the
    other predecessor nodes all supply both, so without this one a change to the
    fallbacks would vary no subject at all.
    """
    (tmp_path / "ALERT.md").write_text(
        "# Half-written alarm\n\nneeds_human: true\n\nact now\n", encoding="utf-8")

    assert _notifier(tmp_path)._alert_file("error", SNAPSHOTS, BODY) is True

    assert _displaced(tmp_path) == (
        "displaced: Half-written alarm level=unknown written=unknown open"), (
        "a missing level or stamp was reported as no burial rather than an "
        "unknown one")
    assert len(_written_values(tmp_path)) == 1, _written_values(tmp_path)


def test_a_seal_longer_than_the_ledger_cap_survives_at_its_head(tmp_path):
    """A long retraction note reaches the row at exactly the cap, head first.

    `_alert_file_displaced` bounds the copy it logs with
    `notify._DISPLACED_NOTE_CHARS`, which is `_ledger`'s own `body[:2000]` — the
    module's existing answer to how much prose a row may carry, not a number
    invented for this branch. Unbounded, one long seal would grow the ledger past
    what an alert row is allowed; cut from the front, the row would lose the
    clause that says HOW the incident closed while keeping the part that says it
    closed at all. The input is a seal longer than the cap, and the row must hold
    exactly the cap's worth with the note's opening clause inside it.
    """
    note = ("the halt was adjudicated a false positive and lifted, so no "
            "promotion is blocked — " + "detail " * 600)
    assert len(note) > notify._DISPLACED_NOTE_CHARS, (
        f"the fixture is only {len(note)} chars and never reaches the cap")
    n = _notifier(tmp_path)
    _raise(n, "critical", DAMAGED, BODY)
    assert n.resolve(DAMAGED, note) is True

    _raise(n, "error", SNAPSHOTS, BODY2)

    rows = _rows(tmp_path, "alert_displaced")
    assert len(rows) == 1, rows
    assert rows[0].get("sealed") is True, rows[0]
    kept = str(rows[0].get("cleared", ""))
    assert len(kept) == notify._DISPLACED_NOTE_CHARS, (
        f"the row kept {len(kept)} chars of a {len(note)}-char seal, not the "
        f"{notify._DISPLACED_NOTE_CHARS} `_DISPLACED_NOTE_CHARS` allows")
    assert note[:40] in kept, (
        "the copy was cut from the front, losing how the incident was closed")


def test_sealing_the_incoming_incident_leaves_the_displaced_trace_readable(tmp_path):
    """The new key is inert to the retraction, and survives being sealed over.

    What this node judges is the WRITE and the seal that follows it, and only
    that: `_alert_file_retract` copies the header keys byte-for-byte, so the
    `displaced:` line has to survive a seal of the incident in the slot, and
    `alert_file_incident` — `#3109`'s read, added for #3019 — has to keep
    answering `(title, sealed)` for a file that carries the new key. Whether the
    retraction still refuses to touch bytes that are NOT this layout is a
    different input and is pinned where it belongs, by
    `tests/test_guardian_alert_retraction.py::a_sealed_tombstone_takes_the_alarm_s_place`;
    nothing here re-pins it, and nothing here claims to.
    """
    n = _notifier(tmp_path)
    _raise(n, "critical", DAMAGED, BODY)
    _raise(n, "error", SNAPSHOTS, BODY2)

    assert n.resolve(SNAPSHOTS, NOTE) is True

    keys = _header_keys(tmp_path)
    assert [k for k in keys if k.startswith(f"{notify.ALERT_CLEARED_PREFIX} ")], (
        f"the incident in the slot was not sealed: {keys!r}")
    assert _displaced(tmp_path).endswith(" open"), (
        "sealing the incoming incident erased the record of what it had buried")
    assert len(_written_values(tmp_path)) == 1
    assert n.alert_file_incident() == (SNAPSHOTS, True), (
        "alert_file_incident can no longer say what the file asserts")


def test_a_file_that_is_not_an_incident_is_replaced_without_a_fabricated_trace(tmp_path):
    """Bytes no reader can call an incident produce no invented key.

    What this node judges is the WRITE and nothing else: `_alert_file_split`
    answers None for bytes that are not this layout, so `_alert_file` has no
    title to name and must invent none — it replaces the file and records no key
    and no ledger row. That the RETRACTION leaves such bytes alone instead is a
    different input, already pinned by
    `tests/test_guardian_alert_retraction.py::a_sealed_tombstone_takes_the_alarm_s_place`;
    this node borrows that fact as a reason for the shape of the write, and does
    not re-prove it. An unreadable predecessor is a reason not to name it, not a
    reason to withhold a live alarm.
    """
    (tmp_path / "ALERT.md").write_text("notes from the outage\n", encoding="utf-8")

    assert _notifier(tmp_path)._alert_file("error", SNAPSHOTS, BODY) is True

    assert [k for k in _header_keys(tmp_path)
            if k.startswith(f"{notify.ALERT_DISPLACED_PREFIX} ")] == []
    assert _rows(tmp_path, "alert_displaced") == []
    assert _lines(tmp_path)[0] == f"# {SNAPSHOTS}"
