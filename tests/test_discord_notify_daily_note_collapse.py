"""#2694 — the daily-note fallback writes one line per alarm FAMILY per day.

What was broken. `discord_alert`'s drop branch (`app.discord_notify.py`) converts a
permanent config decision — `discord.home_channel` is null with an empty token, by
Alan's standing ruling, and `tests/test_autonomy_failure_alert.py` fails any round
that configures it — into a machine log on the surface a human reads. Counted over
`~/obsidian/memory/2026-10-06..13.md`: **37** `Scheduler alert not delivered` lines
against **30** `### Session` captures, 8 of them on 2026-10-09 and 10 on 2026-10-12.
The fallback out-wrote the human content on the one page that decides whether this
fleet looks healthy.

Why the fix is a FAMILY and not a body. The filing guessed "the bodies are
byte-identical per cause"; triage refuted that, and the refutation is the design
constraint. Those 37 lines are **33 distinct bodies**, because
`workers/fleet_watchdog.py` embeds each fire's readings in the sentence
(`oldest claimable queue item is {N} min old`, `_operator_pause_clause`'s
`held {X} h since <ISO ts>`) and #2692 made the head prefix itself move with what the
queue can prove. An exact-body dedupe would have removed 4 lines of 37 (11%) and
removes about nothing going forward, so the key here is the alarm's cause, declared
per family in `app/discord_notify.py::_ALARM_FAMILIES`, with the readings outside it.

Where the bound lives, and why not one level down. `append_daily_alert_line`
(`app/autonomy.py`) must keep turning two calls carrying one body into two lines and
two witness rows — #1727's ruling, pinned by
`tests/test_daily_note_append_witness.py::test_identical_repeats_each_get_a_row_even_when_their_sha_is_one_value`
whose failure message is literally "a pre-write presence check has appeared" — and
clause 5 of this item is that that node stays green. So the fold sits ABOVE the
writer, in `_survive_the_dropped_alert`, and it only ever appends or stays silent: a
suppressed fire writes nothing at all, which is also why the `daily_note_appends`
witness leg can never call a folded line a lost one (clause 4; its leg-side test is in
that file).

The seams, named because the code graph is blind across all three:

  * **process** — the alarms are posted from `lloyd-agent-worker`
    (`workers/fleet_watchdog.py::_alert`, `workers/sources/scheduled_task.py::_alert`)
    and from `lloyd-backend` (`app/autonomy.py::_notify_alert_channel`), and the pool
    restarts ~12x a day. In-process memory therefore bounds nothing: #2662's five
    byte-identical `185 min old` lines were one finding re-posted across restarts.
    The fold's state is a file, and `test_the_fold_survives_a_process_that_no_longer_exists`
    walks a fire through a SECOND PROCESS reading that file.
  * **the vault** — today's note is `~/obsidian/memory/<date>.md`, owned jointly by
    this writer, `app/post_capture.py` and inbound Obsidian Sync. Testified through
    `LLOYD_DAILY_NOTE_DIR`, same as every other daily-note test.
  * **the witness ledger** — `LLOYD_DAILY_NOTE_APPEND_LEDGER`, because a confirmed
    append leaves a row there and the fold must not manufacture a row for a line that
    does not exist.

One asymmetry to know before reading the tests: `app.discord_notify._fold_unavailable`
turns the fold OFF whenever `LLOYD_DAILY_NOTE_DIR` is redirected without
`LLOYD_DAILY_NOTE_FOLD_STATE` (#2213's pair, seen from the suppressing side — a
fixture's day-and-family record must never get to decide whether a production alarm is
written again). `tests/conftest.py` redirects the notes for every test and nothing
defaults the state variable, so the rest of the suite runs with the fold skipped and
sees today's behaviour, and every node below sets BOTH variables to exercise the live
path. `test_only_a_redirected_note_directory_without_a_redirected_state_disables_it`
pins that switch, in both directions.
"""
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import app.autonomy as A                                    # noqa: E402
from app.discord_notify import (                            # noqa: E402
    DAILY_NOTE_FOLD_STATE_ENV, _alarm_family, _doubled, _family_reading,
    discord_alert)

NOTE_ENV = "LLOYD_DAILY_NOTE_DIR"
LEDGER_ENV = "LLOYD_DAILY_NOTE_APPEND_LEDGER"

# ── the alarm texts, copied out of the daily notes they were measured on ──────
# Verbatim from `grep -h 'Scheduler alert not delivered' ~/obsidian/memory/2026-10-{06..13}.md`,
# minus the `**Scheduler alert not delivered** … instead — ` frame this module adds, so
# the patterns under test are matched against what `workers/fleet_watchdog.py`
# actually builds and not against a paraphrase of it.

STALL_185 = ("autonomy scheduler may be stalled: oldest claimable queue item is "
             "185 min old")
STALL_250 = ("autonomy scheduler may be stalled: oldest claimable queue item is "
             "250 min old")
STALL_400 = ("autonomy scheduler may be stalled: oldest claimable queue item is "
             "400 min old")
STALL_915 = ("autonomy scheduler may be stalled: oldest claimable queue item is "
             "915 min old")
STALL_337_PAUSED = (
    "autonomy scheduler may be stalled: oldest claimable queue item is 337 min old; "
    "the worker pool is PAUSED by operator, held 1.4 h since "
    "2026-10-12T02:53:35.952490+00:00 — claims are held by that pause, not by this "
    "scheduler; resume it to drain the backlog")
STALL_OUTRANKED = (
    "queue behind higher-priority work: the owed-check stream has a claimable row "
    "372 min old — priority starvation: it is outranked by higher-priority work, "
    "4 owed-check rows claimed inside 90 min")
NEXTRUN_2 = (
    "2 up_next task(s) more than one period past their own next_run, which the "
    "due-ness stall alarm cannot see | 1 task(s) suppressed as declared parked in "
    "their own file (`parked:`) and not counted above: #68 (Email & Calendar "
    "Triage) is 21.1h past its next_run — held: outside hours 01-04")
NEXTRUN_2_OTHER_BOARD = (
    "2 up_next task(s) more than one period past their own next_run, which the "
    "due-ness stall alarm cannot see | 2 task(s) past their own next_run by less "
    "than one period, which that bound cannot reach, held behind an upstream that "
    "is itself past its next_run or whose newest run record is not a success: #42 "
    "(Nightly Reflection — Knowledge Analysis) is 0.1h past its next_run")
NEXTRUN_MIXED_STATUSES = (
    "1 draft/up_next task(s) more than one period past their own next_run, which "
    "the due-ness stall alarm cannot see")
NEXTRUN_0 = ("0 up_next task(s) more than one period past their own next_run, which "
             "the due-ness stall alarm cannot see")
NEXTRUN_3 = ("3 up_next task(s) more than one period past their own next_run, which "
             "the due-ness stall alarm cannot see")
# A real message from the same route that NO declared family claims — the shape an
# alarm added after this change has until someone declares it.
UNDECLARED_ALARM = ("2 autonomy task file(s) unparseable and INVISIBLE to the "
                    "scheduler: 0001-nightly.md, 0002-config.md")

_ALERT_LINE = re.compile(r"^- \d{2}:\d{2} [A-Z]{2,5} — ")


def _today() -> str:
    """The day that names today's note file: the writer's zone, not UTC."""
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")


def _fire(message: str) -> None:
    """One alarm, down the production route: `discord_alert` and no transport."""
    asyncio.run(discord_alert(message))


def _note(notes: Path) -> Path:
    return notes / f"{_today()}.md"


def _lines(notes: Path) -> list[str]:
    note = _note(notes)
    if not note.exists():
        return []
    return [ln for ln in note.read_text(encoding="utf-8").splitlines()
            if _ALERT_LINE.match(ln)]


def _state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


_FOLD_CLAUSE = re.compile(r"\[daily-note fold: (.*)\]$")


def _fold_clause_text(line: str) -> str:
    """The `[daily-note fold: …]` clause a note line carries, or "" when it has none.

    #3062 needs to assert what the fold's CLEARING SENTENCE promises, and the bullet
    around it is the producer's own text (`workers/fleet_watchdog.py` builds the
    alarm body). Reading only the clause keeps a test honest about which sentence it
    graded: an alarm re-wording itself cannot make a promise about the next line come
    true.

    The `$` is what makes that guarantee real rather than intended: the fold writes
    its clause as the line's tail, so anchoring is what keeps a suffix added after it
    out of the graded text. A line whose clause is no longer its tail matches nothing,
    and every caller asserts ON the returned text, so the helper fails closed — it
    cannot quietly start grading somebody else's prose as fold wording.
    """
    found = _FOLD_CLAUSE.search(line)
    return found.group(1) if found else ""


@pytest.fixture()
def fold(tmp_path, monkeypatch):
    """A note dir, a witness ledger and a fold state, all redirected as a PAIR.

    All three, every test: `app.autonomy._witness_daily_note_append` refuses to
    witness a redirected note without its ledger (#2213) and
    `app.discord_notify._fold_unavailable` refuses to fold a redirected note without
    its state file, so a fixture that set one and not the others would be testing a
    branch that does not exist in production. `CONFIG` is pinned to the live box's
    shape — no channel, no token — so the drop branch under test is the branch this
    box takes, on any machine that runs the suite.
    """
    notes = tmp_path / "notes"
    notes.mkdir()
    ledger = tmp_path / "alerts" / "daily-note-appends.jsonl"
    state = tmp_path / "alerts" / "daily-note-collapse.json"
    monkeypatch.setenv(NOTE_ENV, str(notes))
    monkeypatch.setenv(LEDGER_ENV, str(ledger))
    monkeypatch.setenv(DAILY_NOTE_FOLD_STATE_ENV, str(state))
    monkeypatch.setattr("app.discord_notify.CONFIG",
                        {"discord": {"home_channel": None, "token": ""}},
                        raising=False)
    return {"notes": notes, "ledger": ledger, "state": state}


# ─────────────────────────────── clause 1: the key ────────────────────────────
def test_two_readings_of_one_cause_are_one_family():
    """Clause 1, the half an exact-body dedupe could not pass.

    `185 min old` and `915 min old` are one finding measured twice, and the #1550
    pause clause and the #2692 outranked prefix are the same finding wearing what the
    queue could prove that minute. Each maps to the one key `queue-age`, which is
    what the measured 5-line identical group and the 29 stall lines in the window
    needed: an exact-body key would have collapsed 4 of the 37 lines and left 33.
    """
    for message in (STALL_185, STALL_250, STALL_915, STALL_337_PAUSED,
                    STALL_OUTRANKED):
        family = _alarm_family(message)
        assert family is not None, f"no family claims {message[:60]!r}"
        assert family.name == "queue-age", (
            f"{message[:60]!r} mapped to {family.name!r}, not the one stall key")

    assert _family_reading(_alarm_family(STALL_185), STALL_185) == 185.0
    assert _family_reading(_alarm_family(STALL_915), STALL_915) == 915.0
    # The pause clause's `held 1.4 h` and the starvation clause's `claimed inside
    # 90 min` are not the reading: the wait is.
    assert _family_reading(_alarm_family(STALL_337_PAUSED),
                           STALL_337_PAUSED) == 337.0
    assert _family_reading(_alarm_family(STALL_OUTRANKED),
                           STALL_OUTRANKED) == 372.0


def test_the_two_alarms_that_share_a_page_are_two_families():
    """Clause 1, the other half: two findings never fold into each other.

    The due-ness alarm and the next_run gap are separately streaked, separately
    cooled down and separately watermarked upstream, so folding one into the other
    would let a late board hide a starving queue. The named pair from the clause maps
    to two keys, and so does every pair of distinct alarms here — including the
    next_run alarm's own shapes, which differ only in which `|` arms the scan
    populated and in the statuses word `_nextrun_alert_message` builds.
    """
    stall = _alarm_family(STALL_185)
    nextrun = _alarm_family(NEXTRUN_2)
    assert stall is not None and nextrun is not None
    assert stall.name == "queue-age"
    assert nextrun.name == "past-next-run"
    assert stall.name != nextrun.name

    for other in (NEXTRUN_2_OTHER_BOARD, NEXTRUN_MIXED_STATUSES, NEXTRUN_0,
                  NEXTRUN_3):
        assert _alarm_family(other).name == "past-next-run", (
            f"{other[:60]!r} is the same alarm with different arms")


def test_an_alarm_no_family_claims_is_never_folded(fold):
    """The safety property the two declared families buy their bound with.

    A third producer — an alarm added next month, or the unparseable-file one above,
    which shares the route and no cause sentence — gets today's behaviour: one note
    line per fire, and the fold's state file never even created. That is a deliberate
    asymmetry: the noise this item bounds is a nuisance, and a silence nothing can
    pierce is the outage every other alarm in this file exists to prevent.
    """
    _fire(UNDECLARED_ALARM)
    _fire(UNDECLARED_ALARM)

    lines = _lines(fold["notes"])
    assert len(lines) == 2, (
        f"an undeclared alarm folded to {len(lines)} line(s); only a declared family "
        "may be folded: " + " | ".join(lines))
    assert not fold["state"].exists(), (
        f"the fold wrote state for an alarm it does not claim: "
        f"{fold['state'].read_text(encoding='utf-8')}")


# ────────────────────── clause 2: the fold writes nothing ────────────────────
def test_a_second_fire_of_a_folded_family_writes_no_new_line(fold, caplog):
    """Clause 2, both halves, at the surface each is observable on.

    Two fires of one family on one day leave ONE bullet — the count in the state file
    is where the second fire went — and the `discord_alert (no channel/token
    configured)` WARNING still reaches the log twice, because #1592's acceptance was
    that the WARNING and the note line both appear, not that one replaces the other.
    Suppressing the line must not quietly suppress the log beside it.
    """
    caplog.set_level(logging.WARNING, logger="lloyd-server")
    _fire(STALL_185)
    _fire(STALL_250)

    lines = _lines(fold["notes"])
    assert len(lines) == 1, (
        f"two fires of one family wrote {len(lines)} note lines: "
        + " | ".join(lines))
    assert "185 min old" in lines[0], (
        f"the line kept is not the family's first reading: {lines[0]!r}")

    logged = [r.getMessage() for r in caplog.records
              if r.name == "lloyd-server" and r.levelno == logging.WARNING]
    drop_warnings = [m for m in logged
                     if m.startswith("discord_alert (no channel/token configured)")]
    assert len(drop_warnings) == 2, (
        f"the drop WARNING reached the log {len(drop_warnings)} time(s) for 2 fires "
        f"— the fold may cost a note line, never the log line: {logged}")
    assert any("daily-note fold" in m for m in logged), (
        f"the suppressed fire left no trace naming the fold: {logged}")


def test_a_folded_fire_leaves_every_line_already_in_the_note_untouched(fold):
    """Clause 4's append-or-silent, at the note itself.

    The reason a collapse may not edit its own earlier line is the witness ledger:
    every confirmed append writes a row carrying that line's sha
    (`app/autonomy.py::_witness_daily_note_append`), and
    `~/obsidian/skills/system-health-check/system_health_check.py` books `line-lost`
    / exit 3 for a row whose sha matches no line of its note — so rewriting the first
    line to add `x3` would manufacture a false "a confirmed alert line vanished"
    outage every day it ran. Asserted as bytes: what the note held before the folded
    fire is still a byte-for-byte prefix of it, and the first line is identical.
    """
    _fire(STALL_185)
    before = _note(fold["notes"]).read_bytes()
    before_line = _lines(fold["notes"])[0]

    _fire(STALL_250)                      # folded: 250 < 2 x 185
    after = _note(fold["notes"]).read_bytes()
    assert after.startswith(before), (
        "the note stopped containing what it contained before the folded fire — a "
        "fold may only ever append or stay silent")
    assert len(_lines(fold["notes"])) == 1

    _fire(STALL_400)                      # 400 >= 2 x 185: re-emits
    later_lines = _lines(fold["notes"])
    assert len(later_lines) == 2, (
        f"the doubling fire wrote {len(later_lines)} line(s), expected a second one")
    assert later_lines[0] == before_line, (
        "the line already witnessed changed when a later fire needed a page: "
        f"{before_line!r} -> {later_lines[0]!r}")


# ─────────────── clause 3: the count, the reading, the doubling ──────────────
def test_the_next_line_names_the_fires_it_folded_and_the_newest_reading(fold):
    """Clause 3: the count the fold destroys comes back on the next line out.

    Three fires of the stall family at 185, 250 and 300 min: one line, two fires
    folded. The fourth at 915 min is >= 2x the 185 that is on the page, so it writes,
    and the line that writes has to carry what the page cannot: that two of its
    family were folded into it, and that the reading now is 915 min old.

    The clearing condition travels on the FIRST line too, not just the folded one.
    A single `stalled` bullet under a page of session captures reads as a finding
    someone looked at and closed, which is the exact silence #1121 was filed about;
    the line therefore says it is at most one a day and names what brings the next.

    What it names has to be BOTH triggers (#3062). The line used to promise only the
    doubling, which is a within-day gate: `_day_records` replaces the whole per-day
    record set on a day mismatch, so the first fire of a new calendar day finds no
    `last_emitted` baseline and `_doubled` returns False on None. Doubling therefore
    cannot be what brings that line — the reset is. `~/obsidian/memory/2026-10-18.md`
    :12 and `2026-10-19.md`:12 both open the `past-next-run` family at the identical
    reading `3 task(s) past their next_run` under the doubling promise, and the line
    came anyway. The assertions below read the clause for the trigger rather than the
    word "doubles", and do it for both shapes the fold prints: the first-fire line and
    the `N earlier fires … newest reading X` line.
    """
    _fire(STALL_185)
    first = _lines(fold["notes"])[0]
    first_tail = _fold_clause_text(first)
    assert "daily-note fold" in first, (
        f"the line does not say it is a bounded one: {first!r}")
    assert "at most one line a day" in first, (
        f"the line does not bound itself: {first!r}")
    assert "next calendar day" in first_tail, (
        "the first-fire line names doubling as the only thing that will bring the "
        f"next line, and the day reset removes that gate: {first!r}")
    assert "doubles" in first_tail, (
        "the within-day doubling gate has to stay named on the line it gates: "
        f"{first!r}")
    assert "185 min old" in first

    _fire(STALL_250)
    _fire(STALL_185.replace("185 min", "300 min"))
    assert len(_lines(fold["notes"])) == 1, "a non-doubling fire must not write"

    _fire(STALL_915)
    lines = _lines(fold["notes"])
    assert len(lines) == 2, (
        f"915 min old is >= 2x the 185 on the page and wrote {len(lines)} line(s)")
    second = lines[1]
    assert "2 earlier fires of this alarm today folded into this line" in second, (
        f"the re-emit does not name the two fires it is carrying: {second!r}")
    assert "newest reading 915 min old" in second, (
        f"the re-emit does not name its own reading: {second!r}")
    assert "next calendar day" in _fold_clause_text(second), (
        "the folded-fires line reaches the same clearing sentence as the first one, "
        f"so it must name the day reset too: {second!r}")
    assert "doubles" in _fold_clause_text(second), (
        f"the folded-fires line dropped the doubling gate it still has: {second!r}")
    assert "915 min old" in second
    assert "185 min" not in second.split("daily-note fold")[1], (
        f"the folded count must reset once carried, not accumulate the stale "
        f"baseline: {second!r}")


def test_a_reading_that_has_not_doubled_stays_folded(fold):
    """The chronic case, measured rather than invented.

    185 min on 2026-10-11 16:17 and 196 min on 2026-10-14 02:05 are the endpoints of
    the 22-post run `workers/fleet_watchdog.py::_outranked_clause` counts on
    `server.err*` — the same queue, going nowhere, reported 22 times. 196 is not twice
    185, so the second post stays on the log and off the page. `_doubled` is asserted
    directly beside the walk so a reader can see the threshold rather than infer it
    from which of two fires happened to appear.
    """
    assert _doubled(196.0, 185.0) is False
    assert _doubled(370.0, 185.0) is True
    assert _doubled(185.0, 185.0) is False, "an identical reading never escalates"
    assert _doubled(None, 185.0) is False, "a fire with no reading has no tripwire"
    assert _doubled(200.0, None) is False, (
        "a family that first posted no reading cannot double against nothing")

    _fire(STALL_185)
    _fire(STALL_185.replace("185 min", "196 min"))
    lines = _lines(fold["notes"])
    assert len(lines) == 1, (
        f"196 min old re-wrote the page beside its own 185 min old: {lines}")


def test_zero_doubles_only_above_zero(fold):
    """The one refinement the doubling rule needs, and the case it is for.

    `>= 2x` alone is a lie at zero: 2 x 0 is 0, so a next_run board reporting nothing
    late would re-emit at any reading, including another 0, and the fold would never
    hold for the family whose steady state IS zero. So the rule is strictly greater
    AND at least double — identical for a positive baseline, and at zero it becomes
    "0 stays folded, anything above 0 speaks". Asserted as a walk, because the
    sentence a reader trusts is the one on the note line: 0, 0, then 3 late tasks is
    two lines, and the second names the one fire it folded.
    """
    assert _doubled(0.0, 0.0) is False
    assert _doubled(3.0, 0.0) is True

    _fire(NEXTRUN_0)
    _fire(NEXTRUN_0)
    _fire(NEXTRUN_3)
    lines = _lines(fold["notes"])
    assert len(lines) == 2, (
        f"0, 0, 3 late tasks wrote {len(lines)} line(s), expected the 3 to speak: "
        + " | ".join(lines))
    assert "1 earlier fire of this alarm today folded into this line" in lines[1], (
        f"the singular of one folded fire is what the line must say: {lines[1]!r}")
    assert "newest reading 3 task(s) past their next_run" in lines[1], (
        f"{lines[1]!r}")


def test_the_fold_resets_at_the_next_calendar_day(fold):
    """Clause 3's reset, replayed at the day the state actually carries.

    The reset is a whole-day replace on read (`_day_records`), so it needs no process
    alive at midnight — which matters on a pool that restarts twelve times a day. The
    day is written in the note's own zone, because the note file that names is
    `memory/%Y-%m-%d.md` in America/Los_Angeles: a UTC day boundary would fold a
    23:50 PDT fire against a note that had already rolled over.

    Yesterday's row is rewritten into the state file rather than the clock being
    moved, so what is under test is the comparison the fold makes and not a fake
    clock every later reader has to remember.

    Since #3062 the node also READS the line the reset produced. The same reading
    fires on both days, so nothing but the reset can have written the second one —
    and that line used to end "the next line comes when that reading doubles", a
    promise about a gate `_doubled` cannot evaluate on the first fire of a day
    because `_day_records` has just dropped the day that held its baseline. A wording
    that names only the doubling now fails here, which is the half the old assertion
    on the word "doubles" could not fail on.
    """
    _fire(STALL_185)
    state = _state(fold["state"])
    assert state["day"] == _today(), state
    assert state["families"]["queue-age"]["emitted"] is True
    assert state["families"]["queue-age"]["last_reading"] == 185.0, state

    yesterday = (datetime.now(ZoneInfo("America/Los_Angeles"))
                 - timedelta(days=1)).strftime("%Y-%m-%d")
    state["day"] = yesterday
    state["families"]["queue-age"]["suppressed"] = 3
    fold["state"].write_text(json.dumps(state), encoding="utf-8")

    _fire(STALL_185)
    lines = _lines(fold["notes"])
    assert len(lines) == 2, (
        "a state row from yesterday suppressed today's alarm: " + " | ".join(lines))
    assert "earlier fires" not in lines[1], (
        f"yesterday's 3 folded fires rode into today's line: {lines[1]!r}")
    assert _state(fold["state"])["day"] == _today(), (
        "the write did not re-stamp the state to the day it is counting")

    # The two lines carry one reading, so the reset and not the tripwire is what
    # brought the second: 185 min old on both days, and `_doubled` cannot say True
    # for a reading against its own value.
    assert "185 min old" in lines[0] and "185 min old" in lines[1], (
        "the day-N+1 fire is not the reading yesterday's line carried, so this node "
        f"is no longer replaying a pinned reading: {lines[1]!r}")
    # Scene-setting, not a second check of the reset: the doubling gate itself is
    # graded by `test_a_reading_that_has_not_doubled_stays_folded`. What this guard
    # protects is the inference in the assertion under it — that nothing but the day
    # reset can have written today's line.
    assert _doubled(185.0, 185.0) is False, (
        "an identical reading is not a doubling, so only the day reset can have "
        "written today's line")
    assert "next calendar day" in _fold_clause_text(lines[1]), (
        "today's first line was produced by the day reset, so that is the trigger it "
        f"has to name; naming only the doubling promises a gate that does not reach "
        f"across the boundary: {lines[1]!r}")


# ───────────────────────── the seams, walked for real ────────────────────────
def test_the_fold_survives_a_process_that_no_longer_exists(fold):
    """The process seam, which is the reason the state is a file at all.

    `_state`-style in-memory pacing is what #2662 was: the pool restarts ~12x a day,
    a fresh process satisfies "never alerted" immediately, and 2026-10-13 produced
    five byte-identical `185 min old` note lines from one finding. This walks a
    second, unrelated `python` process — a child that imports the module from
    scratch, so no dict of this one's could be involved — firing the same family
    against the same state file, and the second fire still does not reach the page.
    """
    _fire(STALL_185)
    assert len(_lines(fold["notes"])) == 1

    child = (
        "import asyncio, sys; sys.path.insert(0, %r)\n"
        "import app.discord_notify as dn\n"
        "dn.CONFIG = {'discord': {'home_channel': None, 'token': ''}}\n"
        "asyncio.run(dn.discord_alert(%r))\n" % (str(REPO), STALL_250))
    env = dict(os.environ)
    env.update({NOTE_ENV: str(fold["notes"]),
                LEDGER_ENV: str(fold["ledger"]),
                DAILY_NOTE_FOLD_STATE_ENV: str(fold["state"])})
    done = subprocess.run([sys.executable, "-c", child], env=env, cwd=str(REPO),
                          capture_output=True, text=True, timeout=180)
    assert done.returncode == 0, done.stderr[-2000:]

    lines = _lines(fold["notes"])
    assert len(lines) == 1, (
        "a second process wrote its own line for a family this process had already "
        "put on the page, so the fold was living in memory: " + " | ".join(lines))
    assert _state(fold["state"])["families"]["queue-age"]["suppressed"] == 1, (
        "the folded fire was not counted, so the next line out will not say it "
        "happened: " + fold["state"].read_text(encoding="utf-8"))


#: A day that is not today, stated as a constant rather than computed. `_day_records`
#: compares one string, so any other day is the reset; a date this old can never
#: collide with a clock that rolls while the node is running, which a `now - 1 day`
#: computed in the parent and re-read by the child could.
_AN_OLD_DAY = "2000-01-01"


def test_a_restarted_process_on_a_new_day_names_the_reset_that_fired(fold):
    """Clause 1 and clause 2 together, across both boundaries at once.

    The two seam nodes above each cross one: this file's day-reset node rewrites the
    state file inside one process, and the process node crosses the process inside one
    day. The production event the re-worded sentence describes is their intersection —
    the pool restarts ~12x a day, so the worker that emits the 03:00 PDT alarm is very
    often a fresh interpreter reading a state file an earlier process stamped with
    yesterday's date, holding a `last_reading` it is about to discard.

    That is the case the old wording got wrong where it actually happens: the child has
    no baseline of any kind — not because a reset dropped it in this process, but
    because this process never had one — so the line it writes can only have been
    brought by the day change, and pre-diff it promised a doubling gate. The clause the
    child itself wrote is what is graded here, which no in-process node can produce: a
    clause built by a process that inherited nothing from the one that wrote line 0.
    """
    _fire(STALL_185)
    assert len(_lines(fold["notes"])) == 1

    state = _state(fold["state"])
    assert state["day"] == _today(), state
    state["day"] = _AN_OLD_DAY                      # the file an earlier process left
    state["families"]["queue-age"]["suppressed"] = 3
    fold["state"].write_text(json.dumps(state), encoding="utf-8")

    child = (
        "import asyncio, sys; sys.path.insert(0, %r)\n"
        "import app.discord_notify as dn\n"
        "dn.CONFIG = {'discord': {'home_channel': None, 'token': ''}}\n"
        "asyncio.run(dn.discord_alert(%r))\n" % (str(REPO), STALL_185))
    env = dict(os.environ)
    env.update({NOTE_ENV: str(fold["notes"]),
                LEDGER_ENV: str(fold["ledger"]),
                DAILY_NOTE_FOLD_STATE_ENV: str(fold["state"])})
    done = subprocess.run([sys.executable, "-c", child], env=env, cwd=str(REPO),
                          capture_output=True, text=True, timeout=180)
    assert done.returncode == 0, done.stderr[-2000:]

    lines = _lines(fold["notes"])
    assert len(lines) == 2, (
        "the day change did not re-open the family in a process that had never fired "
        "it, so the reset is not what the two nodes above proved: "
        + " | ".join(lines))
    assert "185 min old" in lines[0] and "185 min old" in lines[1], (
        "the child's line is not the reading the older day's line carried: "
        f"{lines[1]!r}")
    assert "earlier fires" not in lines[1], (
        f"an older day's 3 folded fires rode into today's line: {lines[1]!r}")
    assert "next calendar day" in _fold_clause_text(lines[1]), (
        "the child wrote the line the reset produced, on a reading that cannot have "
        "doubled from anything it knows, so naming only the doubling would promise a "
        f"gate its own process could not evaluate: {lines[1]!r}")
    assert "doubles" in _fold_clause_text(lines[1]), (
        "naming the reset must not drop the within-day gate: " + lines[1])
    assert _state(fold["state"])["day"] == _today(), (
        "the child did not re-stamp the state to the day it is counting")


def test_only_a_redirected_note_directory_without_a_redirected_state_disables_it(
        fold, monkeypatch, caplog):
    """#2213's pair, from the suppressing side.

    Redirected notes with the state variable unset is a fixture, and its day-and-family
    record landing at `~/lloyd-data/alerts/daily-note-collapse.json` would decide
    whether tomorrow's real stall alarm is written at all. So the fold refuses the
    combination and FAILS OPEN — the alarm is written, and the WARNING names both
    variables rather than letting the skip read as a green.
    """
    caplog.set_level(logging.WARNING, logger="lloyd-server")
    monkeypatch.delenv(DAILY_NOTE_FOLD_STATE_ENV)
    _fire(STALL_185)
    _fire(STALL_250)

    lines = _lines(fold["notes"])
    assert len(lines) == 2, (
        f"with the pair broken the fold still suppressed, leaving {len(lines)} "
        "line(s); refusing to fold must fail toward writing")
    assert not fold["state"].exists(), (
        "the fold wrote state to a path the fixture never claimed")
    logged = [r.getMessage() for r in caplog.records
              if r.levelno == logging.WARNING and DAILY_NOTE_FOLD_STATE_ENV in
              r.getMessage()]
    assert logged, (
        "the skipped fold left no warning naming the state variable, so the skip "
        "reads as a clean run")


def test_a_corrupt_state_file_costs_a_line_and_never_a_silence(fold):
    """A torn or hand-edited state file is read as "nothing emitted today".

    `{}` is the fail-open answer (`_read_fold_state`), and the next confirmed emit
    overwrites the file, so the worst a corrupt state does is duplicate one line —
    which is exactly the behaviour before this change. The alternative, refusing to
    write when state cannot be read, is an alarm that reaches no one because a JSON
    brace went missing.
    """
    _fire(STALL_185)
    fold["state"].write_text("{ this is not json", encoding="utf-8")

    _fire(STALL_915)
    lines = _lines(fold["notes"])
    assert len(lines) == 2, (
        "an unreadable state file suppressed the alarm instead of repeating it: "
        + " | ".join(lines))
    assert _state(fold["state"])["families"]["queue-age"]["emitted"] is True, (
        "the recoverable state was not rewritten by the emit that survived")


def test_a_refused_note_line_does_not_spend_the_day(fold, monkeypatch):
    """The emit is recorded only when the note confirms it (#1736's meaning).

    `append_daily_alert_line` returns True only for a line it read back in the file,
    so a False is a line that does not exist. Recording the emit on a False would
    mark the family as on the page and fold every later fire against a line nobody
    can read — the 2026-09-27 21:14 silence, bought back by the fold. So both fires
    here reach the writer, and the state stays empty.
    """
    calls: list[str] = []

    def refuse(body: str) -> bool:
        calls.append(body)
        return False

    monkeypatch.setattr(A, "append_daily_alert_line", refuse)
    _fire(STALL_185)
    _fire(STALL_250)

    assert len(calls) == 2, (
        f"a refused write spent the family's one line, so only {len(calls)} of 2 "
        "fires ever reached the writer")
    assert not fold["state"].exists() or "emitted" not in fold["state"].read_text(), (
        "the fold recorded a line on the page that the note refused: "
        + (fold["state"].read_text(encoding="utf-8")
           if fold["state"].exists() else "<no state>"))
