"""Fleet surveillance for the autonomy scheduler — the alarms, not the dispatch.

Three alarms and one scan live here, and none of them is about enqueuing:

- the grossly-overdue / starving-queue stall alarm (#24/#48/#75, #1550), with the
  operator-pause clause that says when a held pool, not a stuck scheduler, owns a
  backlog;
- the second stall assertion (#421), keyed on each task's own `next_run`, on its own
  streak and its own 24 h cooldown;
- the unparseable-task-file scan (#939), on its own 30 min cadence, which is the only
  thing that notices a task file the scheduler silently drops.

They used to sit inside `workers/sources/scheduled_task.py:enqueue_if_due`, and that
cost more than length: `workers/pool.py` skips that call outright for a source whose
config block says `enabled: false`, so a dispatch-side switch disarmed every alarm
here and raised no alert about the disarmament — an alarm cannot report its own
silencing (#1682). They were also paced by the dispatch tick and paid its watermark
round-trip, while their own cooldowns are 6 h, 24 h and 30 min.

The seat is `WorkerPool._watch_fleet`, beside `_maybe_sweep_poisoned` and
`_probe_services`, whose docstring is the precedent: a reader that has to run when
the fleet is not running cannot be a work source, because a work source is starved
exactly when the fleet is backed up.

What stayed on the dispatch side, and why:

- `_vllm_healthy` and the gate that acts on it, plus `_note_vllm_outage` and the
  outage accounting it drives. #938's invariant is that an outage pauses DISPATCH and
  not WATCHING, and the gate is dispatch: `enqueue_if_due` probes it, accounts for the
  tick, and returns before enqueuing. If the gate moved here the invariant would hold
  only while nothing sat behind it, which is how #938 broke the first time; #1683 owns
  the outage alarm's threshold, so it stays put for that round too.
- `get_due_tasks`, the per-task model-server skip, and the enqueue itself.

The pacing came with the functions: the 5-tick streaks, the 6 h and 24 h cooldowns and
the 30 min scan are the constants below, unchanged, and the pool's own tick (60 s)
paces them exactly as the source's tick did. The one value read from the source's
config block is `max_duration_seconds` — never its `enabled` key, which is the whole
point of the move.
"""

from __future__ import annotations

import asyncio
import logging

from workers.queue import WorkQueue

logger = logging.getLogger("lloyd-workers.fleet_watchdog")

#: The queue `source` id of the dispatch path these alarms watch, and the
#: sources-config block whose `max_duration_seconds` the starving clause weighs the
#: backlog against. The same string as `scheduled_task.NAME`, spelled out here rather
#: than imported: the watchdog must not have to import the dispatcher to ask it what it
#: is called, and a key of the watchdog's own in `config.yaml` would be a config change
#: outside what a round of this size may land.
NAME = "scheduled-task"
#: The fallback the dispatch tick applied before the split (`src_cfg.get(
#: "max_duration_seconds", 1800)`), kept rather than trusted from the live config so
#: a missing block reads as the old default instead of as zero.
DEFAULT_MAX_DURATION_SECONDS = 1800

# Stall alarm: a task overdue by more than this multiple of its interval (with its
# dependency met) should have run; if any are for N consecutive checks, alert once.
_STALL_INTERVAL_MULT = 2.5
_STALL_ALARM_TICKS = 5
# The alarm fired 100 times in 6 days because it could not tell "dispatch is
# broken" from "there is no capacity": #24/#48/#68/#75 are permanently overdue
# when demand exceeds the 2 shared slots. Re-alerting at most this often keeps
# a genuine stall visible instead of drowned.
_STALL_ALERT_INTERVAL_SECONDS = 6 * 3600
# The second, quieter stall assertion (#421). The alarm above is keyed on
# due-ness and `last_run` and its three exclusions are load-bearing for noise —
# they are also exactly the shapes that sat silent for 41 h and 60 h on
# 2026-09-07/08, so this one is keyed on each task's own `next_run` and reads
# none of those three inputs. Quieter by construction: the key is a whole
# period (not 2.5x a partial one) and the cooldown is a day, not 6 hours. Its
# streak is a separate counter so neither alarm can reset the other's.
_STALL_NEXTRUN_TICKS = 5
_STALL_NEXTRUN_ALERT_INTERVAL_SECONDS = 24 * 3600
#: Where the `next_run` alarm's last-reached-a-person instant is persisted: the
#: queue's own watermark table, under this source, in the same row pattern the
#: operator pause uses (`workers/pool.py:PAUSE_WM_SOURCE`). See
#: `_last_nextrun_alert` for why the instant is the VALUE rather than the row's
#: `updated_at`.
_NEXTRUN_WM_KEY = "nextrun_alerted_at"

# How often to re-scan the task files for ones the scheduler cannot parse (#939).
# This used to be a one-shot per process: the flag was set on the first tick and
# every later tick inherited that first tick's answer, so a file that turned
# unparseable while the scheduler was up — a half-written file from a crashed
# vault sync, a bulk edit cut off before its closing `---` fence — stopped being
# dispatched and produced no alert until the next backend restart, which a round
# under observation can defer for a day. Re-scanning is the cheap half; neither
# stall alarm can cover this class, because both iterate sets built from files
# that ALREADY parse, so a file parsing to `None` is missing from their input as
# well as from dispatch. 30 min costs one directory scan per interval at the
# pool's 60 s tick and sits inside the one-hour ceiling #939 sets.
_UNPARSEABLE_SCAN_SECONDS = 30 * 60
# Surveillance state, module-global for the reason the source had it: the streaks and
# cadence slots have to survive from one scheduler tick to the next, and
# `tests/test_autonomy_scheduler.py` restores this dict's CONTENTS around every test
# that reads it so no test inherits another's streak (#939, #1682). The outage's three
# keys stayed in `scheduled_task._state` with the alarm that writes them.
_state = {
          # `_unparseable_scan_at` is the instant the parse scan last ran (None
          # until the first tick takes its slot) and `_unparseable_alerted` the
          # file names it has already raised. Together they make the scan a
          # cadence with transition-only alerting, so neither an unchanged bad
          # set nor a healthy one sends a message every half hour.
          "unparseable_scan_at": None, "unparseable_alerted": set(),
          "stall_streak": 0, "stall_alerted_at": None,
          # `nextrun_alerted_at` is the instant the `next_run` alarm last REACHED A
          # PERSON, and is written only on that route. A scan that found nothing but
          # declared parks never reaches one, so it writes the key beside it instead —
          # its own log cadence, so the record repeats at the alarm's interval without
          # spending the alert cooldown that a real stall will need (#1661). Of these
          # keys only this one is mirrored into the queue (`_NEXTRUN_WM_KEY`), because
          # it is the only one whose promise spans a restart: "once per 24 h" outlives
          # the process that posted the message, while a streak and a log slot are
          # this tick's business (#1735).
          "nextrun_streak": 0, "nextrun_alerted_at": None,
          "nextrun_parked_logged_at": None}


def _source_cfg() -> dict:
    """The `scheduled-task` block of the sources config, re-read every tick.

    The import is function-local because it is the cycle that matters, not the taste:
    `workers.sources` imports `workers.pool`, and `workers.pool` reaches this module
    from `_scheduler_loop`. Re-reading is also what makes `/api/workers/enable` and the
    Tools page take effect without a restart — and the one field taken out of the
    result is `max_duration_seconds`, never `enabled`.
    """
    from workers.sources import get_sources_config
    return dict(get_sources_config().get(NAME) or {})


def _max_duration_seconds() -> int:
    """What the starving clause weighs the oldest claimable item against.

    Read here instead of handed in as `src_cfg`, so the watchdog needs nothing from the
    dispatch path, and falls back to the same 1800 s the dispatch tick fell back to
    rather than to zero — a zero cap would call every queued item starving.
    """
    try:
        return int(_source_cfg().get("max_duration_seconds")
                   or DEFAULT_MAX_DURATION_SECONDS)
    except (TypeError, ValueError):
        return DEFAULT_MAX_DURATION_SECONDS


def _unparseable_task_files() -> list[str]:
    """Task files the scheduler cannot parse (invisible to dispatch).

    Reads `autonomy.AUTONOMY_DIR`, the one directory constant the scans beside it
    already use (`recover_stuck_tasks` at app/autonomy.py:96, `_find_task_file` at
    :161), rather than spelling `Path.home() / "obsidian" / "autonomy"` a second
    time: the value is the same in production — that is how the constant is
    defined, at app/autonomy.py:77 — and the indirection is what lets a test point the
    real scan at a scratch fleet instead of the live board it would alert about."""
    import re
    from app import autonomy
    d = autonomy.AUTONOMY_DIR
    bad = []
    for p in d.glob("*.md"):
        if not re.match(r"\d+-", p.name):
            continue
        if autonomy._parse_task_file(p) is None:
            bad.append(p.name)
    return bad


def _active_task_ids(queue: WorkQueue) -> set:
    """Task ids with a live queue row (queued/claimed/running)."""
    try:
        items = queue.list_items(source=NAME, limit=500)
    except Exception:
        return set()
    return {str(i.payload.get("task_id")) for i in items
            if i.state in ("queued", "claimed", "running")
            and i.payload.get("task_id") is not None}


def _grossly_overdue(queue: WorkQueue) -> list:
    """Tasks that are DUE RIGHT NOW yet overdue by > _STALL_INTERVAL_MULT *
    interval AND have no queue row waiting for them.

    The queue-row exclusion is the whole point: a task sitting in the queue
    behind a saturated pool is starved for capacity, not stalled, and flagging
    it forever is what turned this alarm into noise. A task that is due, this
    stale, and NOT in the queue means the dispatch path itself is broken (the
    2026-05-28 silent stall this was built for)."""
    from app import autonomy
    # One resolution set and one instant, the same pair `get_due_tasks` uses
    # (#870): this used to hand `_is_task_due` the runnable set to resolve
    # `depends_on` with, so a `paused` upstream was invisible here too and the
    # alarm's idea of "due" was not dispatch's idea of due.
    resolution = autonomy.dependency_resolution_set()
    active = _active_task_ids(queue)
    now = autonomy._utcnow()
    overdue = []
    for t in autonomy._all_runnable_tasks(resolution):
        if not autonomy._is_task_due(t, resolution, now=now):
            continue
        if str(t.get("id")) in active:
            continue
        interval = autonomy._frequency_interval_seconds(t)
        last = autonomy._parse_iso(t.get("last_run"))
        if not interval or not last:
            continue
        if (now - last).total_seconds() > _STALL_INTERVAL_MULT * interval:
            overdue.append(t.get("id"))
    return overdue


def _row_vs_records(task: dict) -> dict:
    """What a task's own run records claim, set beside what its row's stamps claim.

    The watchdog's "did it actually run" half, asked of the records (#2342). This is
    the file's only read of them, and it is a recent one: before this function
    existed the scan reasoned from the row alone — `hours` from
    `autonomy.next_run_gap`, `hold` from `autonomy.hold_reason`, `queued` from live
    queue rows — and a search of this file for the run-record tree, its database or
    a run listing answered with one hit, prose in `_last_nextrun_alert`'s docstring
    warning against raw SQL. So `last_run`, a field in a vault file another writer
    may never have advanced, was the only account this alarm had, and a row that lost
    a completion produced a cause claim about dispatch that the run's own record on
    disk contradicted: `run_79_20261006_051105.md` finished at
    `2026-10-06T05:17:08+00:00` while the alert at 22:00:28 that evening read a
    `next_run: 2026-10-06T05:00:00` as 24 h of silence.

    Four fields, and what each is for:

    * `record_at` / `record_run` — the newest `success` record's finished instant and
      its id, quoted into the alert so a reader can open the file. `''` when the
      records answer nothing, which covers "no directory", "unreadable front matter"
      and "no success" alike: `app.autonomy.newest_successful_run` cannot tell those
      apart and neither can a sentence, because inventing a claim about records this
      reader did not read is the same defect pointed the other way. Such a task keeps
      the wording it has always had, and still alerts.
    * `row_last` — the row's own `last_run`, echoed so the alert can show BOTH sides
      of a disagreement instead of asserting one.
    * `lost_completion` — a success the row never recorded. Strictly newer than
      `last_run`, because `last_run` tracks the last SUCCESS (#86): a record from
      before the row's own stamp is the two accounts agreeing.

    The shape this function cannot describe — a row with no `next_run` at all, which
    `past_next_run` can never flag because the bound requires
    `hours_past_next is not None` — is admitted by the scan itself and shows up here
    as `row_last: ''` beside a real `record_at`. That is #96, unalerted on both
    alarms since before this file existed while its 2026-10-05 success record sat on
    disk.

    Cost, measured on the live board the evening this landed rather than estimated:
    38 rows in the resolution set, 2 shortlisted for a record walk (#68, 499 h past
    its `next_run`, and #96 with no stamp at all), so 2 walks of one small directory
    per tick. That is not a new kind of cost here: `_is_task_due` and `hold_reason`
    each walk one of these same directories for any task everything else called due
    (`app/autonomy.py:2282`, `:2407`, `:2419`), on the scheduler's own 60 s tick.
    """
    from app import autonomy
    at, run_id = autonomy.newest_successful_run(task.get("id"))
    row_last = autonomy._parse_iso(task.get("last_run"))
    return {
        "record_at": at.isoformat() if at else "",
        "record_run": run_id,
        "row_last": row_last.isoformat() if row_last else "",
        "lost_completion": bool(at is not None
                                and (row_last is None or at > row_last)),
    }


def _next_run_stalled(queue: WorkQueue) -> list[dict]:
    """Tasks of ANY status sitting more than one period past their OWN `next_run`.

    The companion to `_grossly_overdue`, and deliberately not a variant of it:
    the only task state it reads is `next_run` (plus `frequency` to know what a
    period is), so the three filters that make the noisy alarm survivable — skip
    not-due, skip queue-waiting, skip never-run — cannot hide a stall here. Those
    three are the shapes #421 was filed for: #51 41 h and #40 60 h overdue,
    `failure_count: 0`, no run record, because an upstream's run was never
    recorded and that makes the dependent not-due forever.

    It used to read `status` as well, and that is the one thing #1121 had to
    change: line 174 of this file was `if status != "up_next": continue`, which
    dropped `draft`, `paused` and `failed` out of the only mechanism built to
    catch a stall. An unattributed `up_next -> draft` flip is exactly how #68
    (frequency `every-15min`, the fleet's highest-volume task) stopped, and the
    alarm shipped for that purpose reported zero stalls while #68 sat ~50 cycles
    past its own `next_run`. The status is not a reason to skip the scan any
    more; it is the REASON, carried in `status` and in `hold` — `hold_reason`
    returns the status string itself for any non-`up_next` task, so widening the
    scan needed no new reason code.

    Each entry therefore carries the reason dispatch itself would give —
    `hold_reason`, the same function the dispatch loop's "Holding #" line uses —
    so the alert says WHY it has not run rather than only THAT it has not. A task
    with no hold reason was skipped below this loop: a per-task model-server
    health check, or the model-server gate. Since #938 that gate sits BELOW this
    scan rather than above it, so an `up_next` task past its own period with no
    queue row is flagged while the engine is down too, and reads as a task
    nothing holds: dispatch skipped it because the model server was
    unreachable, which is the outage alert's own sentence, and the two arrive
    together.

    The bound is unchanged and strict (> one interval), and it comes from
    `autonomy.next_run_gap`, the same predicate `compute_health` reads, so the
    alert and the fleet report cannot call the same board two different things.
    Since #2342 the predicate still decides which stamped rows are listed, and the
    one row it structurally cannot see — no `next_run` key — is admitted beside it
    and then judged by its run records, because widening the shared bound is what
    #421 was filed against. So each entry also carries what the RECORDS say, from
    `_row_vs_records`: `record_at`, `record_run`, `row_last` and `lost_completion`.
    `hours` is None exactly on the admitted-without-a-stamp shape,
    and the return drops such an entry when its records are empty too — a row with
    no stamps and no record has never run, and is the noise this alarm's budget is
    spent avoiding.

    The instant is `autonomy._utcnow()`, like the alarm beside it, so one clock
    pin moves both.

    Each entry carries `parked`: the declaration from that task's OWN file, via
    `autonomy.parked_declaration`, or '' when it says nothing. The scan does not
    filter on it, and that is deliberate twice over. First, `_nextrun_alert_message`
    is the surface a person reads, so that is where a declared park is taken out and
    counted — see its docstring for why the count has to survive; second, the list
    here is what a reader of the mechanism, not of the alert, consults, and a task
    silently missing from it would be indistinguishable from one the gap predicate
    never looked at.

    The declaration is asked of the file and never inferred from `status`, because a
    status is precisely what #1121 proved can flip by accident: #68, parked on
    purpose by Alan on 2026-09-17 with the ruling that nobody restore it or re-file
    an item to re-enable it, sits `draft` exactly like an accidental flip does. A
    suppression derived from `draft` would have undone that widening in one line, a
    few above the test that pins it."""
    from app import autonomy
    now = autonomy._utcnow()
    resolution = autonomy.dependency_resolution_set()
    active = _active_task_ids(queue)
    stalled = []
    for t in resolution:
        gap = autonomy.next_run_gap(t, now=now)
        # Two doors. The first is the predicate, untouched by #2342. The second is the
        # row with NO `next_run` key at all: such a row fails `past_next_run` on its
        # own `hours_past_next is not None` guard, so #96 was invisible HERE exactly as
        # it is to the due-ness alarm, and the only thing that knew it had run is its
        # own record. `gap` is not consulted for it — there is nothing to measure.
        # `next_run_gap` itself is unchanged (clause 5): this is a reporter noticing a
        # shape the predicate is built not to see, not a widening of the bound.
        #
        # Cost stays bounded because the door is keyed on a missing field, not on a
        # slow one: `grep -L '^next_run:' ~/obsidian/autonomy/[0-9]*-*.md` names #96
        # alone fleet-wide, so the record walk below runs for the flagged rows plus
        # that one, not for all 38. Whether the unstamped row is reportable is then
        # answered by the records themselves, in the filter on the return.
        if not (gap["past_next_run"]
                or autonomy._parse_iso(t.get("next_run")) is None):
            continue
        stalled.append({
            "id": t.get("id"),
            "name": t.get("name"),
            "status": str(t.get("status", "")).strip(),
            "hours": gap["hours_past_next_run"],
            "gap_ratio": gap["gap_ratio"],
            "hold": autonomy.hold_reason(t, resolution, now=now),
            "queued": str(t.get("id")) in active,
            "parked": autonomy.parked_declaration(t),
            # Read AFTER the shortlist, so the record walk costs what it claims to
            # cost in `_row_vs_records`: the flagged rows, 0 or 1 on a healthy board,
            # not all 38 of them every tick.
            **_row_vs_records(t),
        })
    return [e for e in stalled
            if e["hours"] is not None or e["record_at"]]


def _parked_note(entry: dict) -> str:
    """The declaration carried on a flagged entry, '' if its file declares nothing."""
    return str(entry.get("parked") or "").strip()


def _nextrun_alert_message(stalled: list[dict]) -> str:
    """Alert text for the scan above, describing the statuses it actually found.

    The count line used to read `N up_next task(s)` unconditionally, which was
    only true while the scan filtered on that status. Now it names the statuses
    in the flagged set: an alert that misdescribes its own contents is the
    failure this item is about, because a reader who knows #68 is `draft` and
    reads "up_next task(s)" concludes the message is about some other task and
    drops it. When every flagged task is `up_next` — late against its own
    `next_run` and held by nothing — the wording is exactly what it was before
    the widening, so a reader of the old alert reads the same sentence.

    A task whose file declares it parked is then taken OUT of that list, and the
    asymmetry is why the count must ride along rather than the entry quietly
    disappearing: suppressing a declared park removes a named task, so silence after
    the suppression is indistinguishable from the silence #1121 was filed about. Zero
    used to mean nothing was late; it can now also mean every late task talked itself
    out of the alert. A count is the cheapest marker that keeps "nothing is stalled"
    and "everything stalled was parked" two different sentences, so it is appended
    even when nothing un-suppressed is left — and the caller alerts on the scan's
    list, not on this string's arithmetic, which is what lets that case be said at
    all. The names behind the count stay in the task files: a park is a decision
    somebody wrote down, and the reader needs proof the scan looked, not a rerun of
    a ruling they made.

    The head carries a second count since #2342, for the entries the predicate cannot
    measure at all (no `next_run` key), and the first count covers only the rows it
    did measure. That split is the same rule as the status one two paragraphs up: a
    line that says "N tasks more than one period past their own next_run" may only
    count tasks that are. On a board where every flagged row has its stamps — the
    shape every test of the legacy alarm seeds — the string is byte-identical."""
    suppressed = [e for e in stalled if _parked_note(e)]
    late = [e for e in stalled if not _parked_note(e)]
    # The head sentence claims "more than one period past their own next_run", and a
    # row with no `next_run` is not that — it is the row with nothing to measure
    # against. Counting it in that clause would reproduce, one level up, the exact
    # defect #1121 was filed about: an alert that misdescribes its own contents. So
    # the count covers the measured entries and the unstamped ones get their own
    # clause, which also keeps the legacy string byte-identical on a board where
    # every flagged row has its stamps.
    measured = [e for e in late if e.get("hours") is not None]
    unstamped = [e for e in late if e.get("hours") is None]
    statuses = sorted({str(e.get("status") or "unknown") for e in late})
    # `up_next task(s)` alone reproduces the pre-widening string byte for byte,
    # so anything that matches on the old alert text keeps matching.
    scope = f"{'/'.join(statuses)} task(s)" if late else "task(s)"
    lines = "; ".join(_stall_line(e) for e in late[:15])
    head = (f"{len(measured)} {scope} more than one period past "
            f"their own next_run, which the due-ness stall alarm cannot see")
    if unstamped:
        head += (f" | {len(unstamped)} task(s) whose row carries no next_run stamp "
                 f"at all, which that predicate cannot see either")
    tail = (f" | {len(suppressed)} task(s) suppressed as declared parked in "
            f"their own file (`parked:`) and not counted above"
            if suppressed else "")
    return f"{head}: {lines}{tail}" if lines else f"{head}{tail}"


def _stall_line(entry: dict) -> str:
    """One flagged task, in the sentence the alert actually shows.

    Three shapes, chosen by what the scan could prove rather than by what it wishes.
    The middle one is the whole point of #2342: the row being late AND a run record
    having completed after the row's own `last_run` is a BOOKKEEPING miss, and
    reporting that as a dispatch failure costs the same attention as a real stall
    while teaching the reader to ignore this alarm.

    `hours` is None on the unstamped shape only, so the two measured branches never
    guard their `:.1f`; a caller that handed an unstamped entry with an elapsed
    figure attached would raise here rather than print `Noneh`.
    """
    if entry.get("hours") is None:
        missing = ("no next_run and no last_run" if not entry.get("row_last")
                   else "no next_run")
        said = (f"run record {entry['record_run']} says it succeeded at "
                f"{entry['record_at']}" if entry.get("record_at")
                else "its run records answer nothing")
        return (
            f"#{entry['id']} ({entry['name']}) has {missing} on its row while {said} "
            f"— a row with no stamps can never satisfy the past_next_run predicate, "
            f"so no other alarm can see it")
    if entry.get("lost_completion"):
        return (
            f"#{entry['id']} ({entry['name']}) row bookkeeping lost a completion: "
            f"run record {entry.get('record_run') or '?'} succeeded at "
            f"{entry.get('record_at')}, after the row's last_run "
            f"{entry.get('row_last') or '(absent)'}, and the row is still "
            f"{entry['hours']:.1f}h past its next_run — that elapsed figure is "
            f"measured from a stamp the task's own records contradict")
    return (f"#{entry['id']} ({entry['name']}) is {entry['hours']:.1f}h past its "
            f"next_run — {_hold_note(entry)}")


def _hold_note(entry: dict) -> str:
    """Why a flagged task has not dispatched, in dispatch's own words.

    Reached only for an entry the run records do NOT contradict. Since #2342 the
    caller asks `_row_vs_records` first, because this function reasons from the row
    alone, and a row that lost a completion turns 'dispatch did not enqueue it' into
    a claim the run record on disk refutes — the sentence that fired at 2026-10-06
    22:00:28 about #79, whose run had finished at 05:17:08 that same morning.
    """
    if entry["hold"]:
        return f"held: {entry['hold']}"
    if entry["queued"]:
        return "nothing holds it; a queue row is already waiting for capacity"
    return "nothing holds it; dispatch did not enqueue it"


def _queue_starving(queue: WorkQueue, max_duration: int) -> float:
    """Age in seconds of the oldest claimable queued item, if it is older than
    3x max_duration. Catches the opposite failure: dispatch fine, workers dead."""
    import datetime as _dt
    try:
        items = queue.list_items(source=NAME, limit=500)
    except Exception:
        return 0.0
    now = _dt.datetime.now(_dt.timezone.utc)
    oldest = 0.0
    for i in items:
        if i.state != "queued":
            continue
        if i.not_before:
            nb = _parse_iso_safe(i.not_before)
            if nb and nb > now:
                continue
        enq = _parse_iso_safe(i.enqueued_at)
        if not enq:
            continue
        oldest = max(oldest, (now - enq).total_seconds())
    return oldest if oldest > 3 * max_duration else 0.0


def _operator_pause_clause(queue: WorkQueue) -> str:
    """A clause naming an engaged operator pause and how long it has been held.

    Empty string when no operator pause is held, which is the overwhelmingly
    common case and must leave the alert byte-identical to what it said before
    #1550 — `agent-services/guardian/detect.py` greps the surrounding sentence, and
    an accusation that leaked into the unpaused case would both break that watch and
    blame a person who touched nothing.

    Why this exists: an operator pause is durable and has no TTL, so a hold taken
    by hand in Mission Control stops claims for as long as nobody resumes it. On
    2026-09-24 one lasted 16.5 h straight through the 22:00-04:00 nightly window —
    queue row 510 enqueued 05:00:19Z and was claimed at 18:44:52.78Z, one second
    after the resume line at 18:44:51.495 — and every alert in that stretch read
    `autonomy scheduler may be stalled: oldest claimable queue item is 637 min old`
    (then 791 min). The scheduler was healthy. The pause was the entire cause, and
    nothing said so.

    The verdict is read from the persisted `(_pool, operator_paused)` watermark
    through `workers.pool.operator_pause_state`, the same read `WorkerPool` uses to
    re-engage a hold after a restart. It cannot use `get_pool()`: this tick runs in
    the backend process, which after a restart has a pool whose in-memory flags came
    from that row anyway — and during an outage where no pool started at all there
    is no object to ask. The row is the durable fact; memory is a copy of it.
    """
    try:
        from workers.pool import operator_pause_state
        state = operator_pause_state(queue)
    except Exception:
        logger.exception("stall alarm: could not read the operator pause; "
                         "alarming without it")
        return ""
    if not state["paused"]:
        return ""
    import datetime as _dt
    held_h = 0.0
    since = state["since"]
    if since:
        taken = _parse_iso_safe(since)
        if taken is not None:
            now = _dt.datetime.now(_dt.timezone.utc)
            held_h = max(0.0, (now - taken).total_seconds() / 3600.0)
    return (f"the worker pool is PAUSED by operator, held {held_h:.1f} h since "
            f"{since} — claims are held by that pause, not by this scheduler; "
            f"resume it to drain the backlog")


def _parse_iso_safe(value):
    from app import autonomy
    return autonomy._parse_iso(value)


def _last_nextrun_alert(queue: WorkQueue):
    """When the `next_run` alarm last reached a person, as the FLEET knows it.

    The later of what this process remembers and what the queue's watermark row
    says, because the alarm's 24 h cooldown is a fact about the fleet and `_state`
    is a fact about one process's uptime. That is #1735: `nextrun_alerted_at` lived
    only in the module-global dict, so a new process satisfied "never alerted"
    immediately, reached its own `_STALL_NEXTRUN_TICKS` streak on the same late
    board, and re-posted — `~/lloyd-data/logs/server.err` has pool starts at
    21:10:13 and 22:16:18 on 2026-09-27 with next_run alerts at 21:14:20 and
    22:20:28, 4 min 07 s and 4 min 10 s after each boot, which is exactly five of the
    pool's 60 s scheduler ticks.

    Why the LATER of two reads rather than the row alone: `_state` keeps the key as
    this process's own cache, so the two can disagree only by one being behind, and
    the later one is the sentence "we already told someone". Reading the row alone
    would let a process that had just alerted, on a queue it cannot read, hand the
    day's silence back and re-post; reading `_state` alone is the defect.

    Why the instant is the row's VALUE and not its `updated_at`, which is where
    `workers/pool.operator_pause_state` reads its clock: that row's value is a
    boolean, so `updated_at` is the only place the instant can go, and #1550 put it
    there so the alert and the Mission Control panel quote one accessor. Here the
    instant IS the payload — `wm_set` writes value and stamp in the same statement,
    so they cannot disagree — and storing it as the value is what keeps the row
    aged through the public API. A row whose `updated_at` is the only clock can only
    be made to look a day old with raw SQL against `workers.db`, which is precisely
    the workaround `wm_updated_at`'s own docstring names as the reason it exists.

    An unreadable or unparseable row is read as no durable knowledge, loudly, and
    the in-process answer stands: this runs inside the scheduler loop, and a
    watermark this tick cannot read is not a reason to stop watching the fleet.
    """
    in_process = _state.get("nextrun_alerted_at")
    try:
        raw = queue.wm_get(NAME, _NEXTRUN_WM_KEY)
    except Exception:
        logger.exception("nextrun stall alarm: could not read the persisted alert "
                         "instant; falling back to this process's own record")
        return in_process
    persisted = _parse_iso_safe(raw)
    if raw and persisted is None:
        logger.warning("nextrun stall alarm: the persisted alert instant %r under "
                       "(%s, %s) does not parse; ignoring it",
                       raw, NAME, _NEXTRUN_WM_KEY)
    if persisted is None or (in_process is not None and in_process > persisted):
        return in_process
    return persisted


def _record_nextrun_alert(queue: WorkQueue, when) -> None:
    """Say that the `next_run` alarm reached a person, here and durably.

    Both halves of one fact, written on the one route that reaches a person — a
    scan that found nothing but declared parks never gets here (#1661), so a park
    cannot spend the cooldown. The write is guarded and the alert still goes out if
    it fails: an alarm that could not record itself is still an alarm, and the
    failure it produces is the old one — a later process may re-post — rather than a
    silenced one.
    """
    _state["nextrun_alerted_at"] = when
    try:
        queue.wm_set(NAME, _NEXTRUN_WM_KEY, when.isoformat())
    except Exception:
        logger.exception("nextrun stall alarm: alert posted but its instant could "
                         "not be persisted; a restart inside the cooldown may "
                         "re-post it")


async def _alert(message: str) -> None:
    try:
        from app.discord_notify import discord_alert
        await discord_alert(message)
    except Exception as e:
        logger.error("alert dispatch failed: %s", e)

async def _scan_unparseable_task_files(loop) -> None:
    """Scan the task files and alert on a CHANGE in the unparseable set (#939).

    The alert is keyed on the transition, not on the scan: `_state
    ["unparseable_alerted"]` holds the names already raised, so a file that stays
    broken is named once and not again, and a file that heals gets a recovery
    message and leaves the set — which is what lets a later re-corruption be heard
    at all. A per-scan alert would be one message every `_UNPARSEABLE_SCAN_SECONDS`
    for as long as the file is broken, which is the noise that made the stall alarm
    beside it unreadable (see `_STALL_ALERT_INTERVAL_SECONDS`).

    The recovery half is not decoration. Without it the alert channel records the
    injury and never its repair, and the only surviving account of a resolved
    incident is an instruction to act on it.

    Runs on the caller's event loop with the directory walk on an executor thread,
    like every other scan here: it parses every task file, and the pool's scheduler
    loop is the same loop that dispatches.
    """
    bad = set(await loop.run_in_executor(None, _unparseable_task_files))
    alerted = set(_state.get("unparseable_alerted") or ())
    _state["unparseable_alerted"] = bad
    newly_bad = bad - alerted
    if newly_bad:
        still = alerted & bad
        msg = (f"{len(newly_bad)} autonomy task file(s) unparseable and INVISIBLE "
               f"to the scheduler: {', '.join(sorted(newly_bad))}"
               + (f" (already alerted, still unparseable: "
                  f"{', '.join(sorted(still))})" if still else ""))
        logger.error("%s", msg)
        await _alert(msg)
    recovered = alerted - bad
    if recovered:
        msg = (f"{len(recovered)} autonomy task file(s) parseable again and back "
               f"on the scheduler: {', '.join(sorted(recovered))}")
        logger.warning("%s", msg)
        await _alert(msg)


async def tick(queue: WorkQueue) -> None:
    """One pass of fleet surveillance, called from the pool's scheduler loop.

    The body is the surveillance the dispatch tick used to run, in the order it ran it,
    with the dispatch half left behind: the vLLM probe and gate, `recover_stuck_tasks`
    (which resets claims so dispatch can resume them) and `get_due_tasks` all stay in
    `scheduled_task.enqueue_if_due`. It never probes the model server: watching needs
    nothing from it, which is exactly what #938 was about.

    Raises are the seat's business — `WorkerPool._watch_fleet` catches, because an
    alarm's own failure must not take the loop that enqueues everything down with it.
    Every detector here is read-only.
    """
    from app import autonomy

    loop = asyncio.get_event_loop()


    # Validation of the task files against the scheduler's own parser, on a
    # cadence rather than once per process: any file the scheduler cannot parse is
    # silently dropped (the 2026-05-28 stall), and neither stall alarm can see it.
    # The slot is claimed BEFORE the scan so a scan that raises — an unreadable
    # directory, a mount mid-flip — costs the next interval, not one executor call
    # per tick. It needs no model server, which since #1682 is a fact about this
    # module rather than a note about sitting above the vLLM gate.
    scan_at = _state.get("unparseable_scan_at")
    scan_now = autonomy._utcnow()
    if (scan_at is None
            or (scan_now - scan_at).total_seconds() >= _UNPARSEABLE_SCAN_SECONDS):
        _state["unparseable_scan_at"] = scan_now
        await _scan_unparseable_task_files(loop)

    # Stall alarm — due, grossly overdue, and NOT waiting in the queue.
    import datetime as _dt
    max_dur = _max_duration_seconds()
    overdue = await loop.run_in_executor(None, _grossly_overdue, queue)
    starving = await loop.run_in_executor(None, _queue_starving, queue, max_dur)
    if overdue or starving:
        _state["stall_streak"] += 1
        now = _dt.datetime.now(_dt.timezone.utc)
        last_alert = _state.get("stall_alerted_at")
        due_for_alert = (last_alert is None or
                         (now - last_alert).total_seconds() >= _STALL_ALERT_INTERVAL_SECONDS)
        if _state["stall_streak"] >= _STALL_ALARM_TICKS and due_for_alert:
            _state["stall_alerted_at"] = now
            parts = []
            if overdue:
                parts.append(f"{len(overdue)} task(s) due and overdue "
                             f">{_STALL_INTERVAL_MULT}x their interval with no queue row "
                             f"(ids: {overdue[:15]})")
            if starving:
                parts.append(f"oldest claimable queue item is {starving/60:.0f} min old")
            # #1550: attribute a starving queue to a held pool before attributing it
            # to anything. Computed on the alert path only — one indexed read of the
            # watermark row, not a per-tick cost — and empty unless an operator pause
            # is actually persisted, so the unpaused message is untouched.
            pause_clause = _operator_pause_clause(queue)
            if pause_clause:
                parts.append(pause_clause)
            msg = "autonomy scheduler may be stalled: " + "; ".join(parts)
            logger.error("%s", msg)
            await _alert(msg)
    else:
        _state["stall_streak"] = 0

    # Second stall assertion (#421), keyed on each task's own `next_run` so it
    # sees the three shapes the alarm above is built to exclude. Separate
    # streak, separate cooldown, separate message: nothing about the noisy
    # alarm's exclusions or text moved.
    #
    # The streak below stays on the scan's own list, declared parks included, so a
    # board whose only late tasks are parked is still a board the scan CONFIRMED late.
    # Filtering the scan would make it indistinguishable from a healthy one, which is
    # the silence #1121 was filed about. What is suppressed here is the ROUTE, not the
    # finding (#1661): a board that flags nothing but tasks which declare their own
    # park has nothing a person has not already written down, and on this fleet that is
    # the steady state — #68 is parked by a standing ruling — so posting it was one
    # daily-note line per restart on the one surface the fallback exists to reach.
    # The finding is logged instead, carrying the same message, so "nothing is late"
    # and "everything late was parked" remain two different sentences in the log as
    # they are in the alert.
    stalled = await loop.run_in_executor(None, _next_run_stalled, queue)
    _state["nextrun_streak"] = (
        (_state.get("nextrun_streak", 0) + 1) if stalled else 0)
    if stalled and _state["nextrun_streak"] >= _STALL_NEXTRUN_TICKS:
        nr_now = _dt.datetime.now(_dt.timezone.utc)
        # One test decides both branches, and it is the test the message applies:
        # `_parked_note`, so a task cannot be counted as suppressed by the string and
        # still counted as late by the trigger.
        if any(not _parked_note(e) for e in stalled):
            last_nextrun = _last_nextrun_alert(queue)
            if (last_nextrun is None
                    or (nr_now - last_nextrun).total_seconds()
                    >= _STALL_NEXTRUN_ALERT_INTERVAL_SECONDS):
                # Recorded before the message is dispatched, so a failed send still
                # spends the cooldown rather than handing the next process a fresh
                # day (#1704's ordering), and recorded durably as well as here, so
                # the next PROCESS is included in that same sentence (#1735).
                _record_nextrun_alert(queue, nr_now)
                msg = _nextrun_alert_message(stalled)
                logger.error("%s", msg)
                await _alert(msg)
        else:
            # Nothing late is unaccounted for, so no alert is posted and the ALERT
            # cooldown is deliberately not consumed: spending a day's silence on a
            # park would mute the first genuinely late task for 24 h, which is the
            # silence this file exists to prevent. The log record has its own slot at
            # the same interval, so the scan says it looked without becoming a line
            # every 60 s for as long as the park stands.
            last_parked_log = _state.get("nextrun_parked_logged_at")
            if (last_parked_log is None
                    or (nr_now - last_parked_log).total_seconds()
                    >= _STALL_NEXTRUN_ALERT_INTERVAL_SECONDS):
                _state["nextrun_parked_logged_at"] = nr_now
                logger.warning(
                    "nextrun stall alarm not posted, declared parks only: %s",
                    _nextrun_alert_message(stalled))
