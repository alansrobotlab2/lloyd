"""Discord webhook helper for autonomy task-completion notifications.

`discord_alert` is the terminal route for several independent callers — the
scheduler's five alarms in `workers/sources/scheduled_task.py::_alert` (model-server
outage, unparseable task files, their recovery, the due-ness stall and the next_run
stall), `autonomy._record_failure`'s infra-ceiling crossing, and the retry-budget
disable — and an alarm about "autonomy scheduler may be stalled" could burn four
nightly cycles while every trace of it sat in a rotating log file.

This module owns the one decision those callers should not each re-derive: whether
the Discord transport is configured at all and, since #1592, where an alarm goes when
it is not, and since #2694 how often a day may see it there — the daily-note fold
below, which is what keeps one finding with a moving reading from filling the page a
person reads to decide whether the fleet is healthy. The config object is the boot-merged one (`config.yaml` +
`data/tool_overrides.yaml` + the process env, merged once at import), read at CALL
time from inside the function rather than copied into a module-level constant: a
token that is unset here can exist later without a code change, and a second, lazier
source of truth would go stale on that change.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import CONFIG
from app.silent_sentinel import is_silent_response


logger = logging.getLogger("lloyd-server")


def _discord_token() -> str:
    """Read DISCORD_BOT_TOKEN from config, expanding ${VAR} placeholders."""
    raw = CONFIG.get("discord", {}).get("token", "")
    if not isinstance(raw, str):
        return ""
    return re.sub(r"\$\{([^}]+)\}", lambda m: os.environ.get(m.group(1), ""), raw)


def _missing_transport_halves() -> list[str]:
    """Which of the two things a Discord post needs are absent right now.

    Named separately rather than as one "not configured", because #1592 puts this
    list on a surface a person reads and they are two different repairs: a channel is
    an id to fill in, a token is a credential the box currently declines to hold.
    """
    missing = []
    if not CONFIG.get("discord", {}).get("home_channel"):
        missing.append("discord.home_channel is unset")
    if not _discord_token():
        missing.append("the bot token is empty")
    return missing


async def _discord_notify_task_complete(task_id: int, task_name: str, response_preview: str) -> None:
    """Post an autonomy task-completion embed to the Discord home channel."""
    home_channel = CONFIG.get("discord", {}).get("home_channel")
    token = _discord_token()
    if not home_channel or not token:
        return
    if not response_preview or is_silent_response(response_preview):
        return
    embed = {
        "title": f"Task Complete: {task_name}",
        "description": response_preview[:2000],
        "color": 5763719,  # green
        "timestamp": datetime.utcnow().isoformat(),
        "footer": {"text": f"Task #{task_id}"},
    }
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"https://discord.com/api/v10/channels/{home_channel}/messages",
                headers={"Authorization": f"Bot {token}", "Content-Type": "application/json"},
                json={"embeds": [embed]},
            )
    except Exception as e:
        logger.warning("Discord notify failed: %s", e)


# ---------------------------------------------------------------------------
# The daily-note fold (#2694)
# ---------------------------------------------------------------------------
#
# `append_daily_alert_line` deliberately writes one line per call (#1727: a presence
# test used as a dedupe key dropped 12 legitimate re-fires out of the 13 same-shaped
# lines on `~/obsidian/memory/2026-09-27.md`), and `tests/test_daily_note_append_witness.py
# ::test_identical_repeats_each_get_a_row_even_when_their_sha_is_one_value` pins that
# with the failure message "a pre-write presence check has appeared". So the bound
# cannot live in the writer, and it must not live below it either: the fallback here
# is the one place that knows a note line is an ALARM AND that it is a repeat of an
# alarm already on the page. Measured over `~/obsidian/memory/2026-10-06..13.md`: 37
# dropped-alert lines against 30 `### Session` captures — the fallback out-wrote the
# human content on the surface a person reads, 8 lines on 2026-10-09 and 10 on
# 2026-10-12, with 29 of the 37 from one alarm family and 8 from the other.
#
# What is folded is therefore an alarm FAMILY, never a byte-identical body: those 37
# lines are 33 distinct bodies, because `workers/fleet_watchdog.py` embeds the
# per-fire reading in the sentence (`oldest claimable queue item is {N} min old`,
# `held {X} h since <ISO ts>`), so an exact-body dedupe would have removed 4 of 37.
# A family is a declared cause, and the number that has to DOUBLE to break the fold
# is declared with it — the fold may never become a silence that nothing can pierce.
# An alarm with no declared family is never folded at all: one more producer than the
# two measured here is today's behaviour (a line per fire), which is the failure mode
# this box can survive, beside the one it cannot (a silent note).

#: Env override for the fold's state file, honoured at every call. Third member of
#: the pair `app.autonomy` keeps for the note writer and its witness ledger
#: (`LLOYD_DAILY_NOTE_DIR` / `LLOYD_DAILY_NOTE_APPEND_LEDGER`), and it carries
#: #2213's reason in its sharper half: what this file decides is whether an alarm is
#: WRITTEN AT ALL, so a fixture that redirected the notes and not the state would be
#: suppressing production alarms off a synthetic day-and-family record. The fold
#: refuses that combination outright (`_fold_unavailable`) rather than guess.
DAILY_NOTE_FOLD_STATE_ENV = "LLOYD_DAILY_NOTE_FOLD_STATE"

#: The day the fold counts against is the day that names the note file.
#: `append_daily_alert_line` opens `memory/%Y-%m-%d.md` in America/Los_Angeles, so a
#: fold keyed on UTC would fold a 23:50 PDT fire against a note that had already
#: rolled over, and split two fires of one hour across two days.
NOTE_DAY_TZ = "America/Los_Angeles"


@dataclass(frozen=True)
class _AlarmFamily:
    """One alarm the fold may collapse, and the reading that ends the collapse.

    `name` is the state key and the word a log line names it by. `match` is applied
    with `re.match` to the message, so it is anchored at the cause sentence and every
    per-fire reading that follows it is simply not part of the key — which is the
    whole point: `is 185 min old` and `is 915 min old` are one finding, and the two
    alarms that share a page are two findings.
    """

    name: str
    match: "re.Pattern[str]"
    reading: "re.Pattern[str]"
    unit: str


_ALARM_FAMILIES: tuple[_AlarmFamily, ...] = (
    _AlarmFamily(
        name="queue-age",
        # Both prefixes the due-ness alarm can carry. `workers/fleet_watchdog.py`
        # picks one from whether the queue can PROVE the pool is claiming
        # (`_outranked_clause`, #2692), and they are one alarm: one streak, one 6 h
        # cooldown, one watermark row (`_STALL_WM_KEY`). Splitting them into two
        # families would let one finding post twice a day by alternating prefixes.
        match=re.compile(r"^(?:autonomy scheduler may be stalled"
                         r"|queue behind higher-priority work):"),
        # The wait the alarm exists to report. Deliberately not the #1550 pause
        # clause's `held {X} h` nor #2692's `claimed inside {N} min`: a hold someone
        # is choosing to hold, and a claim count beside the wait, are not the reading
        # whose doubling says this queue got worse.
        reading=re.compile(r"(\d+) min old"),
        unit="min old"),
    _AlarmFamily(
        name="past-next-run",
        # `_nextrun_alert_message` builds its head from the flagged rows' own
        # statuses (`scope` — `up_next task(s)`, and `draft/up_next task(s)` on a
        # mixed board), so the status word is matched, never pinned. The `|` arms
        # after it come and go with what the scan could see, which is exactly the
        # per-fire variation that must not define a family.
        match=re.compile(r"^\d+ [^:;|]*task\(s\) more than one period past "
                         r"their own next_run"),
        reading=re.compile(r"^(\d+) [^:;|]*task\(s\) more than one period past"),
        unit="task(s) past their next_run"),
)


def _alarm_family(message: str) -> Optional[_AlarmFamily]:
    """The declared family of this alarm, or None when nothing may fold it."""
    for family in _ALARM_FAMILIES:
        if family.match.match(message):
            return family
    return None


def _family_reading(family: _AlarmFamily, message: str) -> Optional[float]:
    """This fire's number for `family`, or None when the sentence has none.

    None is not zero. A family fire with no reading can still be folded, but it has
    no tripwire, so the line that folded it says the only thing that will bring the
    next one: the next calendar day.
    """
    found = family.reading.search(message)
    return float(found.group(1)) if found else None


def _note_now() -> datetime:
    """Now, in the zone that names today's note file.

    One snapshot per fire, and its day is then passed to both halves of the fold: a
    fire that straddled midnight between `_fold_fire` and `_fold_emitted` would
    otherwise suppress against yesterday's record and record the emit under today,
    spending a line the page never got.
    """
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo(NOTE_DAY_TZ))


def _note_day(now: datetime) -> str:
    """The calendar day whose note file this fire belongs to."""
    from zoneinfo import ZoneInfo
    return now.astimezone(ZoneInfo(NOTE_DAY_TZ)).strftime("%Y-%m-%d")


def _fold_state_path() -> Path:
    """`DATA_ROOT/alerts/daily-note-collapse.json`, resolved per call.

    Beside the witness ledger and for the same reason: it is an observation about a
    vault file, and an observation stored inside the tree it observes dies with it.
    Deliberately NOT in the queue's watermark table, which is where the producers'
    own cooldowns live (`workers/fleet_watchdog.py::_record_alert_instant`): those
    pace the ALERT, and a row here paces a different surface, written by a different
    process, and must not be reachable by a watermark reset.
    """
    raw = os.environ.get(DAILY_NOTE_FOLD_STATE_ENV, "").strip()
    if raw:
        return Path(os.path.expanduser(raw))
    from app.paths import DATA_ROOT
    return DATA_ROOT / "alerts" / "daily-note-collapse.json"


def _fold_unavailable() -> Optional[str]:
    """Why the fold must not run right now, or None when it may.

    The #2213 pair, seen from the suppressing side. `LLOYD_DAILY_NOTE_DIR` set with
    `LLOYD_DAILY_NOTE_FOLD_STATE` unset is a fixture — production sets neither — and
    its state row landing at the live path would fold tomorrow's real stall alarm
    against a note the fixture wrote. Skipping is fail-OPEN: the alarm is written, as
    it was before this file existed, and the WARNING says which two variables are
    mismatched. It is loud rather than silent for the reason `_witness_daily_note_append`
    gives: a suppressed fold is not a green.

    It is also the reason the rest of the suite is untouched by this change: `tests/
    conftest.py` redirects the notes for every test and no test defaults this variable,
    so only a test that sets both is exercising a fold. That is a deliberate
    asymmetry, and `tests/test_discord_notify_daily_note_collapse.py` is where the
    live-path behaviour is pinned instead.
    """
    if (os.environ.get("LLOYD_DAILY_NOTE_DIR")
            and not os.environ.get(DAILY_NOTE_FOLD_STATE_ENV, "").strip()):
        return (f"LLOYD_DAILY_NOTE_DIR is set to "
                f"{os.environ.get('LLOYD_DAILY_NOTE_DIR')!r} while "
                f"{DAILY_NOTE_FOLD_STATE_ENV} is not, so the fold's state would "
                f"resolve to the live file {_fold_state_path()} and a fixture's "
                f"day-and-family record could suppress a production alarm")
    return None


def _read_fold_state(path: Path) -> dict:
    """The fold's state, or {} for absent/corrupt/unreadable — never a raise.

    `{}` means "nothing has been emitted today", which fails toward writing the
    alarm. A file that does not parse is overwritten by the next recorded emit, so a
    torn write costs one duplicated line and not a silenced week.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def _write_fold_state(path: Path, state: dict) -> None:
    """Replace the state file atomically, and say so when that failed.

    `os.replace` because the reader is a different process (`lloyd-agent-worker`
    suppresses, `lloyd-backend` may be the next to read) and a half-written file
    there would read as corrupt, which costs a line. A lost update when two processes
    fold at the same instant is possible and accepted: the worst it does is one
    duplicate note line, which is precisely the behaviour before this change.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
        tmp.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001 — a fold is never worth the alarm
        logger.warning("could not write the daily-note fold state to %s: %s — the "
                       "fold may repeat a line today rather than lose one", path, exc)


def _day_records(state: dict, day: str) -> dict:
    """This day's family records, dropping any older day's on the way in.

    The reset is a whole-day replace, so the file holds one day and cannot grow: a
    day's worth of families, at most, whatever the month. Doing it on read rather than
    on a midnight timer is what makes the reset need no process to be alive at
    midnight — a pool that restarts 12 times a day is not a scheduler anyone should
    add.
    """
    if state.get("day") == day and isinstance(state.get("families"), dict):
        return state["families"]
    return {}


def _doubled(reading: Optional[float], last_emitted: Optional[float]) -> bool:
    """Has this family's reading doubled since the line last on the page?

    The tripwire clause asks for ">= 2x the last emitted one", which needs one
    refinement to be usable at zero: 0 x 2 is 0, so a baseline of 0 would re-emit on
    any reading at all, including another 0, and the fold would never hold for a
    family that first posted nothing late. So the rule is strictly greater AND at
    least double — identical for any positive baseline, and the zero case becomes
    "0 stays folded, anything above 0 speaks", which is the sentence worth reading.
    """
    if reading is None or last_emitted is None:
        return False
    return reading > last_emitted and reading >= 2.0 * last_emitted


def _fold_clause(family: _AlarmFamily, suppressed: int,
                 reading: Optional[float]) -> str:
    """The tail on an emitted line: why it is the only one, and what ends it.

    Both halves are required and neither is decoration. The count is the one piece of
    information the fold destroys — with the repeats gone the page cannot say how
    many times the fleet screamed — and the clearing condition is what stops the
    single line from reading as a decision that the alarm was dismissed (#1121's
    silence was exactly that reading, and #1661's parked-only log line exists for the
    same reason).

    A reading-bearing line names TWO triggers, because it has two (#3062). Doubling
    is the within-day gate `_doubled` implements; the next calendar day is the gate
    `_day_records` implements, and it is the only one that can fire on a day's first
    alarm — the reset drops the older day's records wholesale, so that fire has no
    `last_emitted` to double against and `_doubled` answers False for a None baseline.
    Promising only the doubling therefore described a gate that does not exist across
    a day boundary, and the vault shows the line doing exactly that with a reading
    that had not moved: `memory/2026-10-18.md`:12 and `memory/2026-10-19.md`:12 both
    open the `past-next-run` family at `3 task(s) past their next_run` under the
    doubling promise, and the line came anyway. The same `clear` is the tail of both
    shapes below, so both are corrected by the one sentence.
    """
    shown = f"{reading:g} {family.unit}" if reading is not None else None
    if suppressed:
        fires = f"{suppressed} earlier fire" + ("s" if suppressed != 1 else "")
        head = f"[daily-note fold: {fires} of this alarm today folded into this line"
        beside = f"; newest reading {shown}" if shown else ""
    else:
        head = "[daily-note fold: this alarm writes at most one line a day here"
        beside = f"; its reading is {shown}" if shown else ""
    clear = ("; the next line comes when that reading doubles, or on the next "
             "calendar day]" if shown
             else "; the next line comes on the next calendar day]")
    return f" {head}{beside}{clear}"


def _fold_fire(family: _AlarmFamily, day: str,
               reading: Optional[float]) -> Optional[str]:
    """Decide one fire of a declared family. None means: write nothing.

    A returned string is the clause to put on the line, and the caller MUST report
    back by `_fold_emitted` only if the note confirmed it — that ordering is what
    keeps a refused write from spending the day's one line for this family. A
    suppression is recorded immediately, because it wrote nothing and so has nothing
    to confirm.
    """
    path = _fold_state_path()
    records = _day_records(_read_fold_state(path), day)
    record = records.get(family.name)
    if not isinstance(record, dict) or not record.get("emitted"):
        # Nothing of this family is on the page today: this is the line.
        return _fold_clause(family, 0, reading)
    # The anchor is the emitted line, not the reading. A family that posted without
    # one (the overdue-only shape of the queue-age alarm names task ids and no wait)
    # has no tripwire to break, so it stays folded until the day rolls — which is
    # what the clause on its own line says, in the same words.
    suppressed = int(record.get("suppressed") or 0)
    last = record.get("last_reading")
    if isinstance(last, (int, float)) and _doubled(reading, float(last)):
        return _fold_clause(family, suppressed, reading)
    record["suppressed"] = suppressed + 1
    records[family.name] = record
    _write_fold_state(path, {"day": day, "families": records})
    logger.warning("daily-note fold: %s fired again on %s and was folded into the "
                   "line already in today's note (%s folded so far; reading %s, the "
                   "line returns when it doubles from %s)", family.name, day,
                   record["suppressed"], reading, last)
    return None


def _fold_emitted(family: _AlarmFamily, day: str,
                  reading: Optional[float]) -> None:
    """Record that a line for `family` is on today's page, and reset its count.

    Called only after `append_daily_alert_line` returned True. Re-reads the file
    rather than writing the dict `_fold_fire` read, because the two calls are
    separated by a vault write and another process may have folded in between; the
    only number this half owns is its own family's.
    """
    path = _fold_state_path()
    records = _day_records(_read_fold_state(path), day)
    records[family.name] = {"emitted": True, "last_reading": reading,
                            "suppressed": 0,
                            "at": datetime.now().astimezone().isoformat()}
    _write_fold_state(path, {"day": day, "families": records})


def _survive_the_dropped_alert(message: str, title: str) -> None:
    """Put an alarm Discord refused onto today's daily note instead.

    The drop branch used to be a `logger.warning` and a `return`, which on this box
    is not a degraded path but the only one: `config.yaml` carries
    `home_channel: null` and an empty token, there is no `data/tool_overrides.yaml`,
    and `tests/test_autonomy_failure_alert.py` reads that null off the disk and fails
    a round that configures it, because leaving it null is Alan's decision to make and
    not a code change's to make for him. So an alarm here reached a rotating log and no
    person — the 2026-09-27 alarm that #68 was 258 h past its own `next_run` is the
    instance on record, and it is the same class of silence that cost #42 four nightly
    cycles.

    The destination is the daily note, the machine-owned surface `_append_fast_failure_alert`
    already writes alarms to for #1209 and which demonstrably lands on this box. The
    warning beside the write stays on the log: the acceptance for #1592 is that the
    WARNING keeps appearing AND a matching note line appears, not that one replaces the
    other. And nothing raised: the callers are a scheduler tick and a run's failure
    path, and the destination being unavailable is never a reason to change what the
    run or the next tick does.

    `app.autonomy` is imported inside the function for the same reason `httpx` is:
    this module is imported from the MCP server and the backend boot path, where a
    cycle through the autonomy module at import time would be a boot-time failure. If
    that import is ever the thing that stops alerts reaching the note, it surfaces as
    the warning below — which is the same shape as the failure it replaces, and is
    stated here rather than hidden so the next reader sees the trade.

    Since #2694 the write is bounded, and it is bounded HERE rather than in the
    writer. The unbounded version made this box's most reliable transport the noisiest
    thing on the surface a person reads — 37 dropped-alert lines against 30 `###
    Session` captures over `~/obsidian/memory/2026-10-06..13.md`, 8 on 2026-10-09 and
    10 on 2026-10-12 — and the reason is structural, not a runaway producer: the
    producers' own cooldowns are per-alarm instants (`_STALL_WM_KEY`,
    `_NEXTRUN_WM_KEY`), while a day of pool restarts, an operator pause held 18.8 h,
    and an alarm whose head prefix moves with what the queue can prove (#2692) each
    put another line on the page for one finding. So one line per alarm FAMILY per
    day, later fires of that family counted and carried on the next line out, and the
    count and its clearing condition named on the line itself.
    `app.autonomy.append_daily_alert_line` still turns two calls carrying one body
    into two lines and two witness rows — #1727's ruling, pinned by
    `test_identical_repeats_each_get_a_row_even_when_their_sha_is_one_value` — and
    nothing here edits a line already written: a suppressed fire writes NOTHING, so
    the `daily_note_appends` witness leg never sees a line it has to call lost. Every
    failure path in the fold points the same way — no state file, corrupt state file,
    no declared family, an unwritable state file, a refused note — and the alarm is
    written anyway, because the noise this bounds is a nuisance and the silence it can
    buy is the outage this function exists to prevent.
    """
    missing = _missing_transport_halves()
    body = (f"**Scheduler alert not delivered** ({'; '.join(missing)}), so it is "
            f"written here instead — {title}: {message[:1000]}")
    try:
        from app.autonomy import append_daily_alert_line

        family = None
        unavailable = _fold_unavailable()
        if unavailable:
            logger.warning("discord_alert: not folding this alarm into the one "
                           "already on today's note: %s", unavailable)
        else:
            family = _alarm_family(message)
            if family is None:
                logger.debug("discord_alert: no declared family for %r, so nothing "
                             "folds it", message[:80])
        if family is not None:
            day = _note_day(_note_now())
            reading = _family_reading(family, message)
            clause = _fold_fire(family, day, reading)
            if clause is None:
                # Folded. Nothing is written, so nothing is witnessed (#2694 clause
                # 4) — the ledger only ever sees lines that exist — and the
                # `discord_alert (no channel/token configured)` WARNING the caller
                # just logged, plus the fold's own line beside it, are where this
                # fire is recorded instead.
                return
            body += clause
            appended = append_daily_alert_line(body)
            if appended:
                _fold_emitted(family, day, reading)
        else:
            appended = append_daily_alert_line(body)
        if not appended:
            logger.warning("discord_alert: the daily note refused the alert too; "
                           "it exists only in this log line")
    except Exception as e:  # noqa: BLE001 — never the caller's problem
        logger.warning("discord_alert: the daily-note fallback failed: %s", e)


async def discord_alert(message: str, title: str = "⚠️ Autonomy health alert") -> None:
    """Post a plain operational alert (e.g. scheduler stall, unparseable tasks).

    When the transport is unconfigured the alert goes to today's daily note instead of
    evaporating into the log — see `_survive_the_dropped_alert` for why that branch is
    this box's only branch, and for what the note line has to carry.
    """
    home_channel = CONFIG.get("discord", {}).get("home_channel")
    token = _discord_token()
    if not home_channel or not token:
        logger.warning("discord_alert (no channel/token configured): %s", message)
        _survive_the_dropped_alert(message, title)
        return
    embed = {
        "title": title,
        "description": message[:2000],
        "color": 15548997,  # red
        "timestamp": datetime.utcnow().isoformat(),
    }
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"https://discord.com/api/v10/channels/{home_channel}/messages",
                headers={"Authorization": f"Bot {token}", "Content-Type": "application/json"},
                json={"embeds": [embed]},
            )
    except Exception as e:
        logger.warning("Discord alert failed: %s", e)
