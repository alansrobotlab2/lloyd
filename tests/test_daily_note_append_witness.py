"""#1799 — the delayed witness for a confirmed daily-note alert append.

What was broken. `app.autonomy.append_daily_alert_line` confirms its own append by
reading the note back (`landed = entry in path.read_text(...)`, `app/autonomy.py`),
and that verification can only ever be true of the instant it runs. The
2026-09-27 21:14 fleet-watchdog alarm is the shape it cannot see: the write was
confirmed, the writer returned True, ten minutes later the note was rewritten from an
older snapshot (mtime moved to 21:24:12, content byte-identical to `HEAD`), and the
alarm was gone with nobody the wiser (#1736 owed entry 3). No instrument on this box
witnessed it — `~/obsidian/memory/audit/writes.jsonl` records only the `vault_write`
route.

What the fix is, and where the seam is. On a CONFIRMED append the writer adds one
JSONL row `{ts, note_path, line_sha256, entry_prefix}` to
`~/lloyd-data/alerts/daily-note-appends.jsonl`, and a new leg of the health script
re-reads the last 48 h of those rows and reports any line whose sha is no longer in
its note.

The boundary is a process boundary twice over, and neither half of it is importable
from the other:

  * the writer runs inside `lloyd-backend`, the leg runs in a `python3` the operator
    or an agent starts out of a skill — the ledger file is the only thing between
    them. So the round-trip test feeds the leg a ledger the WRITER wrote, and a row
    the leg's own reader cannot parse would never surface as a failure of the writer.
  * the leg lives in `~/obsidian/skills/system-health-check/system_health_check.py`,
    a vault file with no importable module, so this loads it with
    `importlib.util.spec_from_file_location` exactly as
    `tests/test_system_health_check_vault_sync.py` does, and the CLI seam
    (`--component daily_note_appends`) is exercised by running the script.

Deliberate choices the tests pin, because each is a place a later "improvement"
would quietly break the witness:

- **No dedupe.** `~/obsidian/memory/2026-09-27.md` holds 13 same-shaped alert lines
  and the fleet-watchdog line re-fires on a 6 h cooldown, so N re-fires must be N
  rows. A ledger that collapsed them would hide 12 re-fires from the one check that
  exists to catch a loss (#1727 ruled the same about the read-back).
- **The witness cannot change the writer's answer.** The caller is a scheduler tick.
- **`0 rows examined` is never a clean verdict.** The question is askable of an empty
  input, so the denominator travels with the verdict.
- **A loss names a path and a sha and no cause.** The note has three mutators — this
  writer, `app/post_capture.py:467-499`, and inbound Obsidian Sync (#1752/#1753).

Ledger path and note directory are both honoured per call through
`LLOYD_DAILY_NOTE_APPEND_LEDGER` / `LLOYD_DAILY_NOTE_DIR`, so nothing here writes
into the production ledger or the real vault. And the pair is ONE invariant (#2213):
a confirmed append whose note dir is redirected while the ledger is not leaves NO
witness row at the default ledger. On 2026-10-04 exactly that combination —
`LLOYD_DAILY_NOTE_DIR=$(mktemp -d)`, the ledger variable unset — put the live
ledger's only row in from one synthetic probe, and the leg reported `witnessed
(1 rows examined, 0 lost)` off zero production appends. The skip logs exactly one
WARNING naming both variables, and `tests/conftest.py` gives the suite a scratch
default for the ledger, so the combination cannot resolve to the live file twice
over: the writer refuses it, the environment never defaults to it.
"""
import hashlib
import importlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The leg's script lives in the vault, not this repo (same seam as the vault_sync
# tests): default is the real file, `LLOYD_SHC_SCRIPT` points at a candidate copy.
SCRIPT = Path(os.environ.get("LLOYD_SHC_SCRIPT")
              or Path.home() / "obsidian/skills/system-health-check/system_health_check.py")
SKILL = Path(os.environ.get("LLOYD_SHC_SKILL") or (SCRIPT.parent / "SKILL.md"))
LEDGER_ENV = "LLOYD_DAILY_NOTE_APPEND_LEDGER"

# Red, never skipped: this file is the only evidence for four of the five clauses, so
# a skip here would drop all of it at once and still report a green suite — which is
# how the sibling vault_sync test is written too (it asserts on the same seam).
assert SCRIPT.exists(), (
    f"{SCRIPT} is the leg's own file and it is not there, so there is no leg to "
    "test — the writer half would still pass and the delayed check would be untested")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


shc = _load_module("_shc_daily_note", SCRIPT)

from app.data_root import ACCOUNT_HOME, production_data_root  # noqa: E402
import app.autonomy as A  # noqa: E402


def _rows(ledger: Path):
    if not ledger.exists():
        return []
    return [json.loads(line) for line in
            ledger.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture()
def witness(tmp_path, monkeypatch):
    """A note directory, an empty ledger path, and the env that points at both.

    The note file is created by the writer on the first append, so the first append
    is not flattered by a fixture that made the file beforehand.
    """
    notes = tmp_path / "notes"
    notes.mkdir()
    ledger = tmp_path / "alerts" / "daily-note-appends.jsonl"
    monkeypatch.setenv("LLOYD_DAILY_NOTE_DIR", str(notes))
    monkeypatch.setenv(LEDGER_ENV, str(ledger))
    return {"notes": notes, "ledger": ledger}


def _today():
    return datetime.now().strftime("%Y-%m-%d")


def _alert_lines(note: Path):
    """The alert-shaped lines of a note, front matter and headings excluded."""
    import re
    return [ln for ln in note.read_text(encoding="utf-8").splitlines()
            if re.match(r"^- \d\d:\d\d [A-Z]{2,5} — ", ln)]


# ---------------------------------------------------------------- cl. 1: rows ---
def test_every_confirmed_append_leaves_one_row_and_re_fires_are_not_deduped(witness):
    """3 legitimate re-fires on one day are 3 rows, each checking its own line.

    Same body twice inside the same minute produces the SAME line text, which is the
    dedupe trap: a ledger keyed on the line would hold 1 row for 3 appends, and 2 of
    the 3 re-fires would become invisible to the check that exists to notice a loss.
    """
    # Three distinct bodies because that is what a real re-fire looks like: the
    # fleet-watchdog line carries the oldest item id and age, so no two of its lines
    # were ever identical (18:38 and 21:14 on 2026-09-27 differ in both). Identical
    # bodies are not refused either — `append_daily_alert_line` has NO pre-write
    # presence check, because #1727 rules one out — so they each land as a duplicate
    # line and each get a row; that case is pinned on its own below.
    for probe in ("oldest item id=1701", "oldest item id=1712", "oldest item id=1735"):
        assert A.append_daily_alert_line(f"fleet watchdog: stalled, {probe}") is True

    rows = _rows(witness["ledger"])
    assert len(rows) == 3, (
        "three confirmed appends left "
        f"{len(rows)} witness row(s) — the ledger must not dedupe re-fires (#1727)")
    assert len({r["line_sha256"] for r in rows}) == 3, rows

    note = witness["notes"] / f"{_today()}.md"
    lines = _alert_lines(note)
    assert len(lines) == 3, f"the note holds {len(lines)} alert line(s): {lines}"
    by_sha = {hashlib.sha256(line.encode("utf-8")).hexdigest(): line for line in lines}
    for row in rows:
        assert set(row) == {"ts", "note_path", "line_sha256", "entry_prefix"}, row
        assert row["note_path"] == str(note)
        assert row["line_sha256"] in by_sha, "the row must hash a line of its note"
        assert row["entry_prefix"] == by_sha[row["line_sha256"]][:80]
        assert datetime.fromisoformat(row["ts"]).tzinfo is not None, (
            f"ts {row['ts']!r} carries no offset, so a later reader cannot "
            "subtract it from an aware now()")

    # (There is no "a repeat returns False" assertion to make here: the writer has no
    # pre-write presence check at all, so a repeat does not return False — see the
    # next test, which pins what a repeat actually does.)


def test_an_unconfirmed_append_leaves_no_row_at_all(witness, tmp_path, monkeypatch):
    """No row for a line that did not land, and none written before the read-back.

    A row for an append that failed would make the leg report a loss the writer
    itself manufactured, which is the fastest way to teach everyone to ignore it.
    """
    broken = tmp_path / "mem"
    broken.mkdir()
    (broken / f"{_today()}.md").write_text("---\ntype: note\n---\n\n", encoding="utf-8")
    os.chmod(broken / f"{_today()}.md", 0o400)        # unwritable: the append fails
    witness["notes"].rmdir()
    monkeypatch.setenv("LLOYD_DAILY_NOTE_DIR", str(broken))
    try:
        assert A.append_daily_alert_line("this line never lands") is False
    finally:
        os.chmod(broken / f"{_today()}.md", 0o600)
    assert _rows(witness["ledger"]) == []


# ----------------------------------------------------- cl. 2: the writer's answer ---
def test_an_unwritable_ledger_changes_neither_answer_nor_raises(witness, tmp_path,
                                                                monkeypatch):
    """Ledger refused three different ways: True is still True, and nothing raises.

    The witness is an observer of a scheduler tick's log line. If it could raise, or
    flip the return, the fix for a lost alarm would become a new way to lose runs.
    """
    body = "fleet watchdog: scheduler may be stalled"

    # (a) the ledger's parent is a regular file, so mkdir() refuses.
    blocker = tmp_path / "blocked"
    blocker.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv(LEDGER_ENV, str(blocker / "sub" / "daily-note-appends.jsonl"))
    assert A.append_daily_alert_line(body + " (a)") is True

    # (b) the ledger path itself is a directory, so open(..., "a") refuses.
    a_dir = tmp_path / "led_is_a_dir"
    a_dir.mkdir()
    monkeypatch.setenv(LEDGER_ENV, str(a_dir))
    assert A.append_daily_alert_line(body + " (b)") is True

    # (c) the ledger's directory exists but is not writable.
    cold = tmp_path / "cold"
    cold.mkdir(mode=0o500)
    monkeypatch.setenv(LEDGER_ENV, str(cold / "daily-note-appends.jsonl"))
    assert A.append_daily_alert_line(body + " (c)") is True

    # (d) and the False path still answers False with a healthy ledger waiting.
    monkeypatch.setenv(LEDGER_ENV, str(witness["ledger"]))
    broken = tmp_path / "mem"
    broken.mkdir()
    note = broken / f"{_today()}.md"
    note.write_text("---\ntype: note\n---\n\n", encoding="utf-8")
    os.chmod(note, 0o400)
    monkeypatch.setenv("LLOYD_DAILY_NOTE_DIR", str(broken))
    try:
        assert A.append_daily_alert_line("never lands") is False
    finally:
        os.chmod(note, 0o600)


# --------------------------------------------------------- cl. 3: the delayed loss ---
def _confirm_an_append_then_clobber(witness):
    """Append for real, remember the note's pre-append bytes, then restore them.

    This is the 2026-09-27 sequence: a confirmed append that returned True, then a
    write-back of the pre-append content. `git checkout HEAD -- <note>` is the real
    route (that is how the 09-27 note got byte-identical to HEAD); restoring the
    bytes the note held before the append is the same state change, without a vault
    checkout in a unit test.
    """
    note = witness["notes"] / f"{_today()}.md"
    # The pre-append state has to be captured BEFORE the append, and for the first
    # append of the day it is "the file does not exist" — restoring the note as it
    # read AFTER the append would restore the line we are trying to lose.
    before = note.read_text(encoding="utf-8") if note.exists() else None
    assert A.append_daily_alert_line("fleet watchdog: scheduler may be stalled") is True
    landed = note.read_text(encoding="utf-8")
    assert "fleet watchdog" in landed, "the append must land before we lose it"
    assert landed != (before or ""), "the append must have changed the note"
    lost_row = _rows(witness["ledger"])[-1]
    if before is None:
        note.unlink()
    else:
        note.write_text(before, encoding="utf-8")
    assert not note.exists() or "fleet watchdog" not in note.read_text(encoding="utf-8")
    return note, lost_row, before


def test_a_note_restored_after_a_confirmed_append_is_reported_lost(witness):
    """The exact case the return-time read-back cannot see, seen hours later."""
    note, lost_row, pre = _confirm_an_append_then_clobber(witness)

    result = shc.check_daily_note_appends(ledger=witness["ledger"])
    assert result["state"] == shc.DAILY_NOTE_APPENDS_LOST, result
    assert result["healthy"] is False
    assert result["rows_examined"] == 1
    assert result["rows_lost"] == 1
    assert lost_row["line_sha256"] in shc.daily_note_appends_reason(result), (
        "the report must name the row's sha, or the reader has nothing to look up")
    assert str(note) in shc.daily_note_appends_reason(result)
    assert result["lost"][0]["note_path"] == str(note)
    assert result["lost"][0]["line_sha256"] == lost_row["line_sha256"]

    # It fails the box, and the reason travels with the failure.
    healthy, reasons = shc.compute_overall([], [], [], None, ["daily_note_appends"],
                                           daily_note_appends=result)
    assert healthy is False
    assert any("no longer in its note" in r and str(note) in r for r in reasons), reasons


def test_a_note_rewritten_from_an_older_snapshot_is_lost_while_the_note_stands(witness):
    """The state 2026-09-27 was actually found in: note present, readable, line gone.

    `_confirm_an_append_then_clobber` has to `unlink` for the first append of the day
    (there is no pre-append image to restore), so that test reaches the leg through
    its unreadable-note branch. This one pre-creates the note, so restoring the
    pre-append bytes leaves a file the leg CAN read and simply cannot find the line
    in — the sha-mismatch branch, which is the one the incident needs and which
    reports a different `why`.
    """
    note = witness["notes"] / f"{_today()}.md"
    pre = ("---\ntype: note\n---\n\n# today\n\n"
           "- 06:55 PDT — a line that was already here\n")
    note.write_text(pre, encoding="utf-8")
    assert A.append_daily_alert_line("fleet watchdog: scheduler may be stalled") is True
    lost_row = _rows(witness["ledger"])[-1]
    assert note.read_text(encoding="utf-8") != pre, "the append must have changed the note"

    note.write_text(pre, encoding="utf-8")          # the older snapshot comes back

    assert note.exists(), "this branch is about a note that is still on disk"
    result = shc.check_daily_note_appends(ledger=witness["ledger"])
    assert result["state"] == shc.DAILY_NOTE_APPENDS_LOST, result
    assert result["rows_examined"] == 1 and result["rows_lost"] == 1, result
    assert result["notes_read"] == 1, "the leg read the note; it was not missing"
    assert result["lost"][0]["why"] == "no line of the note hashes to this sha", (
        f"expected the sha-mismatch branch, not {result['lost'][0]['why']!r}")
    assert result["lost"][0]["line_sha256"] == lost_row["line_sha256"]
    assert str(note) in shc.daily_note_appends_reason(result)


def test_the_leg_reports_no_loss_while_the_line_is_still_in_the_note(witness):
    """Positive control: the same ledger, the line intact, is a pass — not a shrug."""
    assert A.append_daily_alert_line("fleet watchdog: scheduler may be stalled") is True
    result = shc.check_daily_note_appends(ledger=witness["ledger"])
    assert result["state"] == shc.DAILY_NOTE_APPENDS_GREEN, result["detail"]
    assert result["healthy"] is True and result["rows_lost"] == 0
    assert result["rows_examined"] == 1
    assert shc.compute_overall([], [], [], None, ["daily_note_appends"],
                               daily_note_appends=result)[0] is True
    green = shc.daily_note_appends_reason(result)
    assert "no longer in its note" not in green, green
    assert "1 confirmed daily-note alert line" in green, green


def test_a_two_line_alert_body_is_not_reported_lost(witness):
    """A body with a newline lands as several note lines; the row still matches.

    The writer hashes the whole `entry`, so a leg that hashed only the note's first
    line would call a two-line alarm lost while it sat plainly in the note.
    """
    assert A.append_daily_alert_line("first line\nsecond line of the same alert") is True
    result = shc.check_daily_note_appends(ledger=witness["ledger"])
    assert result["state"] == shc.DAILY_NOTE_APPENDS_GREEN, result["detail"]
    assert result["rows_examined"] == 1


# ------------------------------------------------------------ cl. 4: the window ---
def _row_for(note, body, ts):
    return {"ts": ts.isoformat(), "note_path": str(note),
            "line_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "entry_prefix": body[:80]}


def test_identical_repeats_each_get_a_row_even_when_their_sha_is_one_value(witness):
    """Two appends of ONE body are two rows, and one line sha cannot collapse them.

    `append_daily_alert_line` has no pre-write presence check (#1727 rules one out
    precisely so a re-fire stays visible), and the line's only per-fire difference is
    its `HH:MM` stamp — so a same-minute repeat is the same line text twice, and a
    ledger keyed on `line_sha256` would hold 1 row where 2 re-fires happened. That is
    the trap clause 1 names, so it is asserted against the note rather than against a
    count invented here: the rows must equal the appends, and the distinct row shas
    must equal the distinct alert lines the note actually holds — 1 when both appends
    fell in the same minute, 2 when the minute rolled over between them. Which shape
    it saw is in the failure message, so a red run says what it measured.
    """
    assert A.append_daily_alert_line("fleet watchdog: scheduler may be stalled") is True
    assert A.append_daily_alert_line("fleet watchdog: scheduler may be stalled") is True

    note = witness["notes"] / f"{_today()}.md"
    lines = _alert_lines(note)
    assert len(lines) == 2, (
        f"the note holds {len(lines)} alert line(s), not the 2 the writer appended: "
        "a pre-write presence check has appeared")

    rows = _rows(witness["ledger"])
    assert len(rows) == 2, (
        f"{len(rows)} row(s) for 2 confirmed appends — a repeat was deduped")
    assert {r["line_sha256"] for r in rows} == {
        hashlib.sha256(ln.encode()).hexdigest() for ln in lines}, (
        "the row shas are not the note's own line shas")
    assert len({r["line_sha256"] for r in rows}) == len(set(lines)), (
        f"2 rows carry {len({r['line_sha256'] for r in rows})} distinct sha(s) while "
        f"the note holds {len(set(lines))} distinct alert line(s)")


def test_only_rows_inside_the_window_are_examined(witness):
    """A 60 h-old row is out of the 48 h window: not examined, and never a loss.

    The stale row's line is deliberately MISSING from the note. If the window filter
    leaked, this test would report a loss instead of 1 examined.
    """
    note = witness["notes"] / f"{_today()}.md"
    now = datetime.now(timezone.utc)
    fresh_body = "- 07:00 PDT — inside the window"
    stale_body = "- 07:00 PDT — outside the window and gone from the note"
    note.write_text("---\ntype: note\n---\n\n" + fresh_body + "\n", encoding="utf-8")
    witness["ledger"].parent.mkdir(parents=True, exist_ok=True)
    witness["ledger"].write_text(
        "\n".join(json.dumps(r) for r in (
            _row_for(note, fresh_body, now - timedelta(hours=1)),
            _row_for(note, stale_body, now - timedelta(hours=60)))) + "\n",
        encoding="utf-8")

    result = shc.check_daily_note_appends(ledger=witness["ledger"])
    assert result["window_seconds"] == 48 * 3600
    assert result["rows_examined"] == 1, result
    assert result["rows_lost"] == 0
    assert result["state"] == shc.DAILY_NOTE_APPENDS_GREEN


def test_an_empty_or_absent_ledger_is_never_reported_clean(witness):
    """0 rows examined must read as nothing examined, in the dict and on the page."""
    for label, ledger in (("absent", witness["notes"] / "never-written.jsonl"),
                          ("empty", witness["ledger"])):
        if label == "empty":
            # An empty ledger is a real state, not a fixture shortcut: the file
            # exists, holds nothing, and the leg must still refuse to call it clean.
            witness["ledger"].parent.mkdir(parents=True, exist_ok=True)
            witness["ledger"].write_text("", encoding="utf-8")
        result = shc.check_daily_note_appends(ledger=ledger)
        assert result["rows_examined"] == 0, label
        assert result["state"] == shc.DAILY_NOTE_APPENDS_NOTHING, (label, result)
        assert result["state"] != shc.DAILY_NOTE_APPENDS_GREEN, label
        assert result["healthy"] is False, (
            f"{label} ledger answered healthy — an unasked question is not a pass")
        assert "not a clean result" in result["detail"], result["detail"]

        page = shc.format_text([], [], [], None, components=["daily_note_appends"],
                               daily_note_appends=result)
        assert "0 rows examined" in page, page[-500:]
        assert "witnessed" not in page, page[-500:]


def test_rows_the_leg_cannot_parse_are_counted_not_dropped(witness):
    """A ledger that fails its own date filter must not read as 0 rows and a pass.

    The malformed row sits beside a good one: examined counts only what the leg
    actually compared, and the unparseable count is printed next to it.
    """
    note = witness["notes"] / f"{_today()}.md"
    body = "- 07:00 PDT — readable"
    note.write_text("---\ntype: note\n---\n\n" + body + "\n", encoding="utf-8")
    witness["ledger"].parent.mkdir(parents=True, exist_ok=True)
    witness["ledger"].write_text(
        json.dumps(_row_for(note, body, datetime.now(timezone.utc))) + "\n"
        + "{not json\n"
        + json.dumps({"note_path": str(note), "line_sha256": "0" * 64}) + "\n",   # no ts
        encoding="utf-8")

    result = shc.check_daily_note_appends(ledger=witness["ledger"])
    assert result["rows_examined"] == 1, result
    assert result["rows_malformed"] == 2, result
    assert result["state"] == shc.DAILY_NOTE_APPENDS_GREEN
    page = shc.format_text([], [], [], None, components=["daily_note_appends"],
                           daily_note_appends=result)
    assert "2 ledger rows" in page and "NOT examined" in page, page[-400:]


# -------------------------- #2213: the note-dir / ledger pair (cl. 1, 2, 3) ---
def _warning_records(caplog):
    """WARNING-or-worse records this module's logger emitted, rendered.

    #2213's skip has to be ONE line naming both variables: a suppressed witness
    that logs nothing is the silent green the item forbids, and a warning per
    append would bury it under fleet traffic. Filtered to the writer's own
    logger so an unrelated record cannot satisfy the count.
    """
    return [(r.levelname, r.getMessage()) for r in caplog.records
            if r.name == "lloyd-autonomy" and r.levelno >= logging.WARNING]


def test_a_redirected_note_dir_with_the_default_ledger_skips_the_witness_and_warns(
        witness, monkeypatch, caplog):
    """cl. 1 + 2: note dir redirected, ledger not — the line lands, True is True,
    the default ledger's byte size is unchanged, and exactly one WARNING names both
    variables.

    This is the 2026-10-04 shape: `LLOYD_DAILY_NOTE_DIR` at a `mktemp -d`,
    `LLOYD_DAILY_NOTE_APPEND_LEDGER` absent. Before the fix the witness row still
    went to the writer's default ledger — the `witness` fixture points the notes at
    a temp dir and this test deletes the ledger pointer, so the red run shows both
    halves: the ledger grew, and no warning fired.
    """
    from app.paths import DATA_ROOT
    monkeypatch.delenv(LEDGER_ENV, raising=False)
    default = DATA_ROOT / "alerts" / "daily-note-appends.jsonl"
    default.parent.mkdir(parents=True, exist_ok=True)
    default.touch()  # so a leak reads as "grew", not as "no file" — byte-identical
    before = default.stat().st_size
    caplog.set_level(logging.WARNING, logger="lloyd-autonomy")

    assert A.append_daily_alert_line("witness leak probe") is True
    assert len(_alert_lines(witness["notes"] / f"{_today()}.md")) == 1, (
        "the note write itself is not what gets skipped")
    assert default.stat().st_size == before, (
        f"the default ledger moved {default.stat().st_size - before} byte(s) off a "
        f"fixture whose notes live under {witness['notes']} — green evidence no "
        "alarm produced, which is what #2213 forbids")

    warns = _warning_records(caplog)
    assert len(warns) == 1, (
        f"a suppressed witness must log exactly one line, saw {warns!r}")
    level, text = warns[0]
    assert level == "WARNING", level
    assert "LLOYD_DAILY_NOTE_DIR" in text, text
    assert "LLOYD_DAILY_NOTE_APPEND_LEDGER" in text, text


def test_a_redirected_note_dir_with_an_explicit_ledger_witnesses_and_the_leg_reads_it(
        witness, caplog):
    """cl. 3: both variables into one temp dir — exactly one row lands in that ledger,
    its `note_path` is the temp note, and the leg reading that same file counts it
    examined; with the ledger explicit the skip must NOT fire, so no warning either.
    """
    caplog.set_level(logging.WARNING, logger="lloyd-autonomy")
    assert A.append_daily_alert_line("fleet watchdog: scheduler may be stalled") is True

    rows = _rows(witness["ledger"])
    assert len(rows) == 1, f"both vars set must still witness: {len(rows)} row(s)"
    assert rows[0]["note_path"] == str(witness["notes"] / f"{_today()}.md"), rows[0]
    assert _warning_records(caplog) == [], (
        "the skip fired with the ledger explicitly set — it must key on the PAIR, "
        "not on the note dir alone")

    result = shc.check_daily_note_appends(ledger=witness["ledger"])
    assert result["rows_examined"] == 1, result
    assert result["state"] == shc.DAILY_NOTE_APPENDS_GREEN, result["detail"]
    assert result["rows_lost"] == 0


def test_the_items_probe_a_redirected_note_dir_never_reaches_the_default_ledger(tmp_path):
    """The item's own acceptance check, pinned: `LLOYD_DAILY_NOTE_DIR=$(mktemp -d)`
    with the ledger variable unset, in a fresh interpreter, no conftest in sight —
    and `LLOYD_DATA` standing in for the data root so the ledger's byte size is
    measurable the way `wc -c ~/lloyd-data/alerts/...` is on the box.

    Subprocess, because the probe is a process-environment claim: in-process,
    conftest has already set both variables (#2213 cl. 5), and the exact absent
    variable this check depends on cannot be produced by a test node that inherits
    them. The child still logs to stderr through logging's last-resort handler, so
    the WARNING naming both variables is asserted as the probe saw it: exactly one
    line of stderr, carrying both names and the skip's own sentence.
    """
    data = tmp_path / "data"
    data.mkdir()
    ledger = data / "alerts" / "daily-note-appends.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("", encoding="utf-8")
    before = ledger.stat().st_size
    notes = tmp_path / "notes"
    notes.mkdir()

    env = {k: v for k, v in os.environ.items()
           if k not in ("LLOYD_DAILY_NOTE_DIR", LEDGER_ENV, "LLOYD_DATA")}
    env.update({"PYTHONPATH": str(REPO), "LLOYD_DATA": str(data),
                "LLOYD_DAILY_NOTE_DIR": str(notes)})
    proc = subprocess.run(
        [sys.executable, "-c",
         "from app.autonomy import append_daily_alert_line; "
         "print('ok', append_daily_alert_line('witness leak probe'))"],
        capture_output=True, text=True, timeout=180, env=env)

    assert proc.returncode == 0, (proc.stdout[-800:], proc.stderr[-800:])
    assert "ok True" in proc.stdout, proc.stdout
    assert len(_alert_lines(notes / f"{_today()}.md")) == 1, (
        "the probe's note write must still happen; only the witness is skipped")
    assert ledger.stat().st_size == before, (
        f"the probe moved the default ledger {ledger.stat().st_size - before} "
        "byte(s) — the item's check demands byte-identical")
    # ONE line of the child's stderr, not a substring over all of it: the loose
    # form is satisfied by ANY child output naming both variables — a traceback, an
    # unrelated warning that mentions a path — and then this node would be pinning
    # nothing but the byte-identical ledger check above. The skip is exactly one
    # line, and both variable names have to be IN that line, which is what cl. 2
    # asks of a suppressed witness: visible, and legible as this skip.
    warn_lines = [ln for ln in proc.stderr.splitlines()
                  if "LLOYD_DAILY_NOTE_DIR" in ln and LEDGER_ENV in ln]
    assert len(warn_lines) == 1, (
        "the suppressed witness must be exactly ONE stderr line naming both "
        f"variables, saw {len(warn_lines)} in {proc.stderr[-800:]}")
    assert "skipping the daily-note append witness" in warn_lines[0], warn_lines[0]


# --------------------- #2213 cl. 4: the un-override path still witnesses -------
def test_a_confirmed_append_with_no_per_test_override_still_witnesses_to_the_default(
        caplog):
    """cl. 4: with neither variable overridden by the test (both stand at the values
    conftest gives every test — cl. 5 is what makes that ledger a scratch file), a
    confirmed append writes exactly ONE row to `_daily_note_append_ledger()`'s answer.

    This is the witness's reason for existing: it must keep firing on the shape
    production runs, where nothing sets either variable and the writer takes its
    default end to end. A skip keyed on the note dir ALONE would silence it here,
    because conftest always redirects the notes; keying on the pair is what this
    node pins. The row count is a delta, because every other test in the session
    that drives a confirmed append without its own ledger pointer appends to this
    same shared file — which is exactly why the delta is not the proof. The proof
    is the row's own identity: its `line_sha256` hashes a line of the note its
    `note_path` names, the leg's own test of a witness.
    """
    default = A._daily_note_append_ledger()
    assert default == Path(os.environ[LEDGER_ENV]), (
        "with no per-test override the resolver must answer the suite default, "
        f"saw {default} vs {os.environ[LEDGER_ENV]}")
    before = len(_rows(default))
    caplog.set_level(logging.WARNING, logger="lloyd-autonomy")

    assert A.append_daily_alert_line("fleet watchdog: default-path witness") is True

    after = _rows(default)
    assert len(after) == before + 1, (
        f"{len(after) - before} row(s) for one confirmed append at the suite "
        "default — the witness stopped firing on the un-override path")
    mine = after[-1]
    assert "default-path witness" in mine["entry_prefix"], mine
    # A count alone cannot say WHICH append arrived, and this ledger is shared by
    # every test in the session that drives a confirmed append without its own
    # pointer. So the row is pinned by the invariant the leg itself grades: the
    # row's `line_sha256` is the sha256 of a line of the note its `note_path`
    # names, and `entry_prefix` is that line's first
    # `WITNESS_ENTRY_PREFIX_CHARS` characters. A sibling test's row cannot satisfy
    # that, and neither can a witness of a line that never reached the note.
    note = Path(mine["note_path"])
    assert note.parent == Path(os.environ["LLOYD_DAILY_NOTE_DIR"]), (
        f"the row names a note outside the suite's own redirected note dir: {mine}")
    assert note.name == f"{_today()}.md", mine
    assert note.exists(), f"the row names a note that is not on disk: {mine}"
    by_sha = {hashlib.sha256(ln.encode("utf-8")).hexdigest(): ln
              for ln in note.read_text(encoding="utf-8").splitlines()}
    assert mine["line_sha256"] in by_sha, (
        "the row hashes no line of its own note, so the leg could never read this "
        f"append back as witnessed: {mine}")
    assert (mine["entry_prefix"]
            == by_sha[mine["line_sha256"]][:A.WITNESS_ENTRY_PREFIX_CHARS]), mine
    assert _warning_records(caplog) == [], (
        "the skip fired while the ledger had a default — it skips only when the "
        "ledger variable is the one that is absent")


def test_the_resolver_default_stays_under_the_data_root_when_the_note_dir_is_overridden(
        witness, monkeypatch):
    """cl. 4 (second half): the resolver's ANSWER must not move with the note dir.

    `test_the_writer_and_the_leg_default_to_the_same_ledger` below calls the
    resolver and compares its tail against the leg's hardcoded literal to catch a
    data-root cutover. A "fix" that made the resolver consult the note dir would
    turn that drift check into a comparison of a scratch path against `~/lloyd-data/
    ...` — red on the fixture instead of red on a cutover. The skip therefore lives
    in `_witness_daily_note_append`, and this node is the trap notice: with the note
    dir redirected (the `witness` fixture) and the ledger variable deleted, the
    resolver still answers the DATA ROOT's `alerts/daily-note-appends.jsonl`.

    FULL PATH, not the tail — the advisory this round's first review left on the
    node that used to sit here. Asserting only the two components
    `("alerts", "daily-note-appends.jsonl")` is satisfied by a resolver answering
    `<tmp>/alerts/daily-note-appends.jsonl`, which is the very behaviour this node
    exists to forbid: the tail is what a data-root cutover moves, and the PARENT is
    what a note-dir leak moves. So the assertion is equality with
    `DATA_ROOT / "alerts" / "daily-note-appends.jsonl"`, the one expression the
    resolver itself returns, and the fixture's note dir is first pinned to a
    different root so the equality cannot be won by both sides naming one temp dir.
    `DATA_ROOT` and not `production_data_root()` is the anchor because inside an
    automod gate `HOME` is the round's symlink farm and `LLOYD_DATA` points the data
    root at a scratch directory (`app/paths.py`'s header); the production-literal
    half of the drift check stays the seam test's job, on the passwd-derived anchor.
    """
    from app.paths import DATA_ROOT
    monkeypatch.delenv(LEDGER_ENV, raising=False)
    notes = Path(os.environ["LLOYD_DAILY_NOTE_DIR"])   # set by the `witness` fixture
    assert notes != DATA_ROOT and DATA_ROOT not in notes.parents, (
        f"the note dir {notes} IS the data root {DATA_ROOT} or sits under it, so "
        "this node cannot tell a ledger that followed the data root from one that "
        "followed the notes")
    redirected = A._daily_note_append_ledger()   # LLOYD_DAILY_NOTE_DIR set by fixture
    assert redirected == DATA_ROOT / "alerts" / "daily-note-appends.jsonl", (
        f"with the note dir redirected to {notes} and the ledger variable absent, "
        f"the resolver answered {redirected} instead of the data root's own "
        f"{DATA_ROOT / 'alerts' / 'daily-note-appends.jsonl'} — a resolver that "
        "consults the note dir breaks the seam test's drift check on the fixture, "
        "which is why #2213's skip sits in the writer instead")
    assert notes not in redirected.parents, (
        f"the default witness ledger resolves INSIDE the redirected note dir "
        f"{notes}: {redirected}")


# ------------------------ #2213 cl. 5: the suite's own ledger default ----------
def test_the_suite_ledger_default_is_a_scratch_file_not_the_production_ledger():
    """cl. 5: the pytest default for `LLOYD_DAILY_NOTE_APPEND_LEDGER` is a FILE path
    under the scratch directory conftest already builds (the `LLOYD_EGRESS_DB`
    shape), so no in-suite default resolves under the production data root.

    Read off `os.environ` — the state every test that touches neither variable
    actually runs in — and checked against `production_data_root()`, the
    passwd-derived anchor the seam test below uses, so it answers the same inside
    a gate (where `HOME` is a symlink farm) as on the box. The `.jsonl` tail is
    asserted because the tuple at `tests/conftest.py:81` creates DIRECTORIES and
    this store is a file: the directory-shaped version of the same mistake would
    leave every append dying on `IsADirectoryError` — no rows, and a leg reading
    nothing.
    """
    env_default = os.environ.get(LEDGER_ENV, "").strip()
    assert env_default, (
        "conftest must give the suite a default for this variable; unset, every "
        "test that drives a confirmed append resolves the witness to the "
        "production data root — which is the leak #2213 files")
    path = Path(env_default)
    assert path.name.endswith(".jsonl"), f"{path} is a directory-shaped default, not a file"
    assert path.parent.name.startswith(("lloyd-test-state-", "lloyd-test-data-")), (
        f"{path} is not under the scratch directory conftest builds")
    assert production_data_root() not in path.parents, path
    assert path != production_data_root() / "alerts" / "daily-note-appends.jsonl", path


# ------------------- #2213 cl. 6: the witness bytes now have a history ----------
#: The committed extract of the production witness ledger, byte-identical to the
#: vault copy at `backlog/data/daily-note-appends.jsonl` (vault commit `5ff50a5e`).
WITNESS_EXTRACT = (Path(__file__).resolve().parent
                   / "fixtures" / "daily_note_append_witness_2213.jsonl")
#: Where clause 6 asked for those bytes to live, for a reader who can reach it.
VAULT_WITNESS_COPY = Path.home() / "obsidian" / "backlog" / "data" / "daily-note-appends.jsonl"


def test_the_committed_witness_bytes_still_carry_the_quoted_one_fixture_row():
    """cl. 6: the ledger bytes the item quotes are committed, and the quoted report is
    re-derived from THOSE BYTES here — 1 line, 281 bytes, and the one row is the
    fixture-derived one the leg has been reporting green.

    Clause 6's reason is "The witness bytes have no history": the ledger lives at
    `~/lloyd-data/alerts/daily-note-appends.jsonl`, in no git tree, and the single row
    in it is the 2026-10-04 probe that made the leg read
    `witnessed (1 rows examined, 0 lost)` off zero production appends. That copy is in
    the vault (`backlog/data/daily-note-appends.jsonl`, landed under #2213's first
    round), but no node can open it: an automod gate runs with `HOME` at the round's
    symlink farm, where `~/obsidian` does not exist, and a node that opened it would
    skip — `tests/fixtures/.gitignore`'s own header says a skipping node pins nothing.
    So the same bytes are committed here, and every figure the item quotes is read off
    them, exactly as clause 6 asks (`wc -l` of the committed bytes is the figure).

    The line count and byte count are what clause 6 quotes; the rest of the node keeps
    those numbers MEANING something, because `wc -l` of an empty file is 0 and of a
    text file saying "1" is 1. The row must carry the writer's exact key set and an
    80-character `entry_prefix` (`WITNESS_ENTRY_PREFIX_CHARS`), so the extract cannot
    be prose standing in for a witness; and its `note_path` must be a `mktemp -d` path
    outside the vault note dir, because a fixture row is precisely what the item's
    premise is — the day's real alert appends left NO row, which is why this family is
    a witness to a gap and not evidence of health.
    """
    assert WITNESS_EXTRACT.exists(), (
        f"{WITNESS_EXTRACT} is missing: the ledger the item quotes has no history "
        "in this tree, which is what clause 6 exists to fix")
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "--error-unmatch",
         str(WITNESS_EXTRACT.relative_to(REPO))],
        capture_output=True, text=True)
    assert tracked.returncode == 0, (
        f"{WITNESS_EXTRACT.relative_to(REPO)} is on disk but NOT tracked — a witness "
        f"nobody can check out is no history at all: {tracked.stderr.strip()}")

    raw = WITNESS_EXTRACT.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    assert len(lines) == 1, (
        f"clause 6's re-derive is `wc -l` of these bytes and the item quotes 1; "
        f"the committed extract has {len(lines)} line(s)")
    assert len(raw) == 281, (
        f"the committed extract is {len(raw)} bytes, not the 281 the item quotes — "
        "either the ledger gained a row or this extract was cut down to fit")
    assert raw.endswith(b"\n") and b"\n\n" not in raw, (
        "JSONL with a blank line is a row the leg skips, and a byte count quoted "
        "from such a file is not the figure `wc -l` answers")

    row = json.loads(lines[0])
    assert set(row) == {"ts", "note_path", "line_sha256", "entry_prefix"}, row
    assert len(row["entry_prefix"]) == A.WITNESS_ENTRY_PREFIX_CHARS, row
    assert len(row["line_sha256"]) == 64, row

    # The quoted figure is not just a count, it is a COUNT OF A FIXTURE ROW — the
    # item's premise, re-derived rather than repeated.
    note = Path(row["note_path"])
    assert note.parent.name.startswith("tmp"), (
        f"the committed row's note is {note}, not a `mktemp -d` directory — if a real "
        "production append ever lands in this extract, this node stops describing the "
        "2026-10-04 gap and must be re-titled, not quietly re-pointed")
    assert not str(note).startswith(str(Path.home() / "obsidian")), (
        f"{note} is inside the vault's real note tree: this row would be a genuine "
        "witness, and the extract's premise (zero production appends witnessed) "
        "would be false")
    assert row["ts"].startswith("2026-10-04T23:26"), row["ts"]

    # Whenever the vault IS reachable (a human's checkout, never a gate), the two
    # copies of these bytes must not have drifted. Guarded, and the node stands on
    # its own without it: this is a cross-check, not the pin.
    if VAULT_WITNESS_COPY.exists():
        assert VAULT_WITNESS_COPY.read_bytes() == raw, (
            f"the vault copy {VAULT_WITNESS_COPY} and the committed extract "
            f"{WITNESS_EXTRACT.relative_to(REPO)} are two histories of one ledger "
            "that no longer agree")


# ------------------------------------------------- the seam: the two default paths ---
def test_the_writer_and_the_leg_default_to_the_same_ledger(monkeypatch):
    """With no override, both halves must resolve one file, or the leg watches nothing.

    The writer asks `app.paths.DATA_ROOT`; the leg is a vault script with no venv and
    cannot import that module, so it hardcodes `~/lloyd-data/...`. Two independent
    literals on one path is the drift this round's own review named: after a data-root
    cutover the writer keeps writing to the new root, the leg keeps reading the old
    one, and it answers `nothing-examined` — which clause 4 makes it refuse to call
    clean, but it is still a witness that silently stopped witnessing, with a green
    suite. Every other test here sets `LLOYD_DAILY_NOTE_APPEND_LEDGER`, so this one
    deletes it.

    The comparison is deliberately HOME-independent, because inside an automod gate
    `HOME` is the round's symlink farm where `~/lloyd-data` is an empty directory
    (`app/paths.py`'s header): the leg's resolved path is checked by its tail, and its
    HARDCODED literal is checked against `production_data_root()`/`ACCOUNT_HOME`, the
    passwd-derived pair that answers the same in a gate as on the box.
    """
    monkeypatch.delenv(LEDGER_ENV, raising=False)
    tail = ("alerts", "daily-note-appends.jsonl")
    writer_default = A._daily_note_append_ledger()
    leg_default = shc._daily_note_append_ledger()
    assert tuple(writer_default.parts[-2:]) == tail, writer_default
    assert tuple(leg_default.parts[-2:]) == tail, leg_default

    hardcoded = PurePosixPath(shc.DAILY_NOTE_APPEND_LEDGER_DEFAULT)
    assert hardcoded.parts[0] == "~", hardcoded
    root_tail = production_data_root().relative_to(ACCOUNT_HOME).parts
    assert hardcoded.parts[1:1 + len(root_tail)] == root_tail, (
        f"the leg hardcodes {hardcoded}, but the production data root is "
        f"{production_data_root()} under {ACCOUNT_HOME} — a cutover moved one "
        "literal and not the other")
    assert hardcoded.parts[1 + len(root_tail):] == tail, hardcoded


# ------------------------------------------------- cl. 5: the CLI and the prose ---
def _run_cli(args, env_extra=None):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    env.update(env_extra or {})
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True,
                          text=True, timeout=180, env=env)


def test_the_leg_is_selectable_by_name_and_fails_the_exit_status(witness, tmp_path):
    """`--component daily_note_appends` runs the leg and nothing else.

    Run as a subprocess because that is the only way this leg is ever reached: the
    skill tells a person or an agent to run the script. Exit `3` on a lost line is
    the ERROR surface — nothing imports this file.
    """
    assert "daily_note_appends" in shc.COMPONENTS
    note = witness["notes"] / f"{_today()}.md"
    body = "- 07:00 PDT — the alarm that later vanished"
    note.write_text("---\ntype: note\n---\n\n", encoding="utf-8")     # line NOT in note
    witness["ledger"].parent.mkdir(parents=True, exist_ok=True)
    witness["ledger"].write_text(
        json.dumps(_row_for(note, body, datetime.now(timezone.utc))) + "\n",
        encoding="utf-8")

    proc = _run_cli(["--component", "daily_note_appends", "--format", "json"],
                    {LEDGER_ENV: str(witness["ledger"])})
    assert proc.returncode == 3, (proc.returncode, proc.stdout[-800:], proc.stderr[-400:])
    payload = json.loads(proc.stdout)
    assert payload["daily_note_appends"]["state"] == shc.DAILY_NOTE_APPENDS_LOST
    assert payload["daily_note_appends"]["rows_examined"] == 1
    assert str(note) in proc.stdout and "line_sha256" in proc.stdout
    # Selecting one component must not RUN the others: they stay at their
    # not-measured placeholder instead of producing data or a verdict.
    assert payload["components"] == ["daily_note_appends"], payload["components"]
    assert payload["disk"] == {"healthy": None, "volumes": []}, payload["disk"]

    text_run = _run_cli(["--component", "daily_note_appends"],
                        {LEDGER_ENV: str(witness["ledger"])})
    assert "Daily-Note Append Witness" in text_run.stdout, text_run.stdout[-800:]
    assert "1 rows examined in the last 48 h" in text_run.stdout

    empty = _run_cli(["--component", "daily_note_appends", "--format", "json"],
                     {LEDGER_ENV: str(tmp_path / "nothing-here.jsonl")})
    page = json.loads(empty.stdout)["daily_note_appends"]
    assert page["state"] == shc.DAILY_NOTE_APPENDS_NOTHING
    assert page["rows_examined"] == 0
    assert "not a clean result" in page["detail"]


def test_the_skill_prose_names_the_component_and_every_verdict_state():
    """The prose is how the leg gets run at all, so its states have to be written down.

    Asserted as the four state strings plus the denominator rule: a run that prints
    `nothing-examined` and a skill that never says what that means is how an empty
    ledger becomes "clean" in someone else's summary.
    """
    assert SKILL.exists(), f"{SKILL} is where the prose must live"
    text = SKILL.read_text(encoding="utf-8")
    assert "--component daily_note_appends" in text
    for state in (shc.DAILY_NOTE_APPENDS_GREEN, shc.DAILY_NOTE_APPENDS_LOST,
                  shc.DAILY_NOTE_APPENDS_NOTHING, shc.DAILY_NOTE_APPENDS_UNREADABLE):
        assert state in text, f"the skill never names the {state!r} state"
    assert ("0 examined is never a pass" in text) or ("0 rows examined" in text), (
        "the skill must state that a nothing-examined ledger is not a clean result")
    assert shc.DAILY_NOTE_APPENDS_NOTHING == "nothing-examined", (
        "the skill text quotes this state name literally")
