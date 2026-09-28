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
          "nextrun_streak": 0, "nextrun_alerted_at": None}


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
        if not gap["past_next_run"]:
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
        })
    return stalled


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
    a ruling they made."""
    suppressed = [e for e in stalled if _parked_note(e)]
    late = [e for e in stalled if not _parked_note(e)]
    statuses = sorted({str(e.get("status") or "unknown") for e in late})
    # `up_next task(s)` alone reproduces the pre-widening string byte for byte,
    # so anything that matches on the old alert text keeps matching.
    scope = f"{'/'.join(statuses)} task(s)" if late else "task(s)"
    lines = "; ".join(
        f"#{e['id']} ({e['name']}) is {e['hours']:.1f}h past its next_run"
        f" — {_hold_note(e)}"
        for e in late[:15])
    head = (f"{len(late)} {scope} more than one period past "
            f"their own next_run, which the due-ness stall alarm cannot see")
    tail = (f" | {len(suppressed)} task(s) suppressed as declared parked in "
            f"their own file (`parked:`) and not counted above"
            if suppressed else "")
    return f"{head}: {lines}{tail}" if lines else f"{head}{tail}"


def _hold_note(entry: dict) -> str:
    """Why a flagged task has not dispatched, in dispatch's own words."""
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
    # The trigger below stays on the scan's own list, declared parks included, and
    # that is what lets `_nextrun_alert_message` suppress them at all: filtering here
    # would make a board whose only late tasks are parked indistinguishable from a
    # healthy one, which is the silence #1121 was filed about. The message says
    # "0 …| 1 suppressed" for that board instead of saying nothing.
    stalled = await loop.run_in_executor(None, _next_run_stalled, queue)
    _state["nextrun_streak"] = (
        (_state.get("nextrun_streak", 0) + 1) if stalled else 0)
    if stalled and _state["nextrun_streak"] >= _STALL_NEXTRUN_TICKS:
        last_nextrun = _state.get("nextrun_alerted_at")
        nr_now = _dt.datetime.now(_dt.timezone.utc)
        if (last_nextrun is None
                or (nr_now - last_nextrun).total_seconds()
                >= _STALL_NEXTRUN_ALERT_INTERVAL_SECONDS):
            _state["nextrun_alerted_at"] = nr_now
            msg = _nextrun_alert_message(stalled)
            logger.error("%s", msg)
            await _alert(msg)
