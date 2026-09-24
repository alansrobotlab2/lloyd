"""scheduled-task source — wraps the existing autonomy task files.

Reads ~/obsidian/autonomy/*.md each tick, evaluates `due-ness` per task
(frequency / runs_per_day / preferred_hours / depends_on), and enqueues the
highest-priority due tasks. Execution delegates to autonomy.run_task() which
still writes the per-task markdown run records under autonomy-runs/{task_id}/.
"""

from __future__ import annotations

import asyncio
import logging
import urllib.request
from typing import Any

from workers.queue import WorkQueue, QueueItem

logger = logging.getLogger("lloyd-workers.scheduled_task")

NAME = "scheduled-task"
DEFAULT_PRIORITY = 30

_PRIORITY_MAP = {
    "critical": 10,
    "high": 20,
    "medium": 30,
    "low": 50,
    "background": 70,
}

# vLLM health gate: skip enqueuing when the model server is unreachable so a wedge
# doesn't turn every due task into a ConnectError flood.
_VLLM_HEALTH_URL = "http://127.0.0.1:8096/health"


def _model_health_url(model: str) -> str:
    """Health endpoint for the model a task actually runs on.

    Tasks can pin `model: secondary` in their frontmatter (autonomy.run_task
    resolves it through config.models.<name>.env.ANTHROPIC_BASE_URL). Probing
    only the primary would let a dead secondary turn every one of its tasks
    into a ConnectError flood — the exact failure the primary gate prevents.
    """
    model = str(model or "").strip()
    try:
        from app.config import resolve_model_alias
        model = resolve_model_alias(model)
    except Exception:
        pass
    if not model or model in ("primary", "null", "none"):
        return _VLLM_HEALTH_URL
    try:
        import autonomy
        base = autonomy._get_model_env(model).get("ANTHROPIC_BASE_URL")
        if base:
            return base.rstrip("/") + "/health"
    except Exception:
        pass
    return _VLLM_HEALTH_URL
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
# How long the model server can stay down before the outage itself is an alert
# (not one `logger.warning` line deduped for the whole outage). The gate in
# `enqueue_if_due` pauses dispatch, and until #938 that one line was the only
# thing an outage produced: `agent-services/guardian/policy.py` watches only
# `lloyd-backend`/`lloyd-mcp`, never the model server, and the vault tasks that
# do probe it are autonomy tasks dispatched through this gate, so they die with
# it. At `tick_interval: 60` a 3-day outage was ~4300 ticks and one log line.
# 45 min is long enough to sit past a vLLM restart (the n-gram table alone takes
# minutes to load) and short enough that a person hears about it within an hour.
_VLLM_DOWN_ALERT_SECONDS = 45 * 60
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
_state = {"vllm_down_logged": False,
          # `_unparseable_scan_at` is the instant the parse scan last ran (None
          # until the first tick takes its slot) and `_unparseable_alerted` the
          # file names it has already raised. Together they make the scan a
          # cadence with transition-only alerting, so neither an unchanged bad
          # set nor a healthy one sends a message every half hour.
          "unparseable_scan_at": None, "unparseable_alerted": set(),
          "stall_streak": 0, "stall_alerted_at": None,
          "nextrun_streak": 0, "nextrun_alerted_at": None,
          # Outage accounting: `_vllm_down_since` is the instant the current
          # uninterrupted run of unhealthy ticks began, and `_vllm_down_alerted`
          # says whether THIS outage has already raised its alert. Both reset on
          # the first healthy tick, so the next outage gets one alert of its own
          # rather than inheriting the last one's silence.
          "vllm_down_since": None, "vllm_down_alerted": False}


def _vllm_healthy(timeout: float = 4.0, url: str | None = None) -> bool:
    try:
        with urllib.request.urlopen(url or _VLLM_HEALTH_URL, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def _unparseable_task_files() -> list[str]:
    """Task files the scheduler cannot parse (invisible to dispatch).

    Reads `autonomy.AUTONOMY_DIR`, the one directory constant the scans beside it
    already use (`recover_stuck_tasks` at autonomy.py:96, `_find_task_file` at
    :161), rather than spelling `Path.home() / "obsidian" / "autonomy"` a second
    time: the value is the same in production — that is how the constant is
    defined, at autonomy.py:77 — and the indirection is what lets a test point the
    real scan at a scratch fleet instead of the live board it would alert about."""
    import re
    import autonomy
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
    import autonomy
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
    pin moves both."""
    import autonomy
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
        })
    return stalled


def _nextrun_alert_message(stalled: list[dict]) -> str:
    """Alert text for the scan above, describing the statuses it actually found.

    The count line used to read `N up_next task(s)` unconditionally, which was
    only true while the scan filtered on that status. Now it names the statuses
    in the flagged set: an alert that misdescribes its own contents is the
    failure this item is about, because a reader who knows #68 is `draft` and
    reads "up_next task(s)" concludes the message is about some other task and
    drops it. When every flagged task is `up_next` — late against its own
    `next_run` and held by nothing — the wording is exactly what it was before
    the widening, so a reader of the old alert reads the same sentence."""
    statuses = sorted({str(e.get("status") or "unknown") for e in stalled})
    # `up_next task(s)` alone reproduces the pre-widening string byte for byte,
    # so anything that matches on the old alert text keeps matching.
    scope = f"{'/'.join(statuses)} task(s)"
    lines = "; ".join(
        f"#{e['id']} ({e['name']}) is {e['hours']:.1f}h past its next_run"
        f" — {_hold_note(e)}"
        for e in stalled[:15])
    return (f"{len(stalled)} {scope} more than one period past "
            f"their own next_run, which the due-ness stall alarm cannot "
            f"see: {lines}")


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


def _parse_iso_safe(value):
    import autonomy
    return autonomy._parse_iso(value)


async def _alert(message: str) -> None:
    try:
        from app.discord_notify import discord_alert
        await discord_alert(message)
    except Exception as e:
        logger.error("alert dispatch failed: %s", e)


async def _note_vllm_outage() -> None:
    """Account for one unhealthy tick: log the transition, and alert once if the
    outage has run past `_VLLM_DOWN_ALERT_SECONDS`.

    Called on every unhealthy tick and on nothing else, so the elapsed time it
    measures is the length of ONE uninterrupted run of them — the healthy tick
    clears `_vllm_down_since`, which is what makes a later outage alert again
    instead of inheriting this one's `vllm_down_alerted`.

    The alert is exactly one per outage and the log line stays transition-only:
    at `tick_interval: 60` an alert-per-tick would be 1440 a day, and the
    single deduped `logger.warning` this replaced was 0 a day where a person
    reads it — `logger` output goes to the unit log, `_alert` goes to discord,
    and no other watcher covers the model server (#938)."""
    import datetime as _dt
    now = _dt.datetime.now(_dt.timezone.utc)
    if not _state["vllm_down_logged"]:
        logger.warning("scheduled-task: vLLM unhealthy — pausing enqueue until it recovers")
        _state["vllm_down_logged"] = True
        _state["vllm_down_since"] = now
        _state["vllm_down_alerted"] = False
    if _state.get("vllm_down_alerted"):
        return
    since = _state.get("vllm_down_since") or now
    down_for = (now - since).total_seconds()
    if down_for >= _VLLM_DOWN_ALERT_SECONDS:
        _state["vllm_down_alerted"] = True
        msg = (f"primary model server {_VLLM_HEALTH_URL} has been unreachable for "
               f"{down_for / 60:.0f} min (since {since.isoformat()}): autonomy "
               f"dispatch is PAUSED until it recovers, so every stall alert from "
               f"here on is describing a paused fleet, not a broken one")
        logger.error("%s", msg)
        await _alert(msg)


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


async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> None:
    import autonomy
    from autonomy import get_due_tasks

    loop = asyncio.get_event_loop()

    # Validation of the task files against the scheduler's own parser, on a
    # cadence rather than once per process: any file the scheduler cannot parse is
    # silently dropped (the 2026-05-28 stall), and neither stall alarm can see it.
    # The slot is claimed BEFORE the scan so a scan that raises — an unreadable
    # directory, a mount mid-flip — costs the next interval, not one executor call
    # per tick. It needs no model server, so it sits above the vLLM gate below
    # along with `recover_stuck_tasks`, and a fleet pinned to a dead engine still
    # gets its task files checked.
    scan_at = _state.get("unparseable_scan_at")
    scan_now = autonomy._utcnow()
    if (scan_at is None
            or (scan_now - scan_at).total_seconds() >= _UNPARSEABLE_SCAN_SECONDS):
        _state["unparseable_scan_at"] = scan_now
        await _scan_unparseable_task_files(loop)

    # Reset tasks stuck in_progress past their timeout (e.g. worker died
    # mid-run). Cheap relative to the get_due_tasks parse below. (`autonomy` is
    # already bound above, where the scan reads its clock.)
    recovered = await loop.run_in_executor(None, autonomy.recover_stuck_tasks)
    if recovered:
        logger.warning("Recovered %d stuck task(s): %s", len(recovered), recovered)

    # vLLM health gate — pause DISPATCH when the model server is down. It yields
    # a verdict here and acts on it below, after both stall detectors have run:
    # they read only task files and the queue, so neither needs the model server,
    # exactly like the startup scan and `recover_stuck_tasks` above. Returning at
    # this line is what made a multi-day outage invisible — it silenced dispatch
    # AND the detectors together, and left one deduped log line behind (#938).
    # The detectors' own exclusions are untouched, so an outage does not turn
    # every paused or queue-waiting task into noise: only a task that dispatch
    # would have enqueued and did not is flagged.
    vllm_ok = await loop.run_in_executor(None, _vllm_healthy)

    # Stall alarm — due, grossly overdue, and NOT waiting in the queue.
    import datetime as _dt
    max_dur = int(src_cfg.get("max_duration_seconds", 1800))
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
            msg = "autonomy scheduler may be stalled: " + "; ".join(parts)
            logger.error("%s", msg)
            await _alert(msg)
    else:
        _state["stall_streak"] = 0

    # Second stall assertion (#421), keyed on each task's own `next_run` so it
    # sees the three shapes the alarm above is built to exclude. Separate
    # streak, separate cooldown, separate message: nothing about the noisy
    # alarm's exclusions or text moved.
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

    # The gate pauses dispatch here, once the watching above has run: the outage
    # stops work, it does not stop watching. Returning before `get_due_tasks`
    # preserves the gate's own purpose — a fleet pinned to a dead model server is
    # skipped, not enqueued into a ConnectError retry loop (see the per-task
    # `_model_health_url` check below, which is the same rule one level down).
    if not vllm_ok:
        await _note_vllm_outage()
        return
    if _state["vllm_down_logged"]:
        logger.info("scheduled-task: vLLM healthy again — resuming enqueue")
        _state["vllm_down_logged"] = False
        _state["vllm_down_since"] = None
        _state["vllm_down_alerted"] = False

    due = await loop.run_in_executor(None, get_due_tasks)

    healthy_urls: dict[str, bool] = {}
    for task in due:
        task_id = task.get("id")
        if task_id is None:
            continue
        # A task pinned to a model whose server is down should be skipped, not
        # enqueued into a retry loop.
        url = _model_health_url(task.get("model"))
        if url not in healthy_urls:
            healthy_urls[url] = await loop.run_in_executor(None, _vllm_healthy, 4.0, url)
        if not healthy_urls[url]:
            logger.warning("Skipping #%s (%s): model server %s is unhealthy",
                           task_id, task.get("name"), url)
            continue
        dedup_key = f"scheduled-task:{task_id}"
        prio_str = str(task.get("priority", "medium")).lower()
        priority = _PRIORITY_MAP.get(prio_str, DEFAULT_PRIORITY)
        new_id = queue.enqueue(
            source=NAME,
            kind="run",
            payload={"task_id": task_id, "name": task.get("name")},
            priority=priority,
            dedup_key=dedup_key,
        )
        if new_id is not None:
            logger.info("Enqueued scheduled-task #%s (%s) id=%d prio=%d",
                        task_id, task.get("name"), new_id, priority)


async def execute(item: QueueItem) -> dict[str, Any]:
    from autonomy import run_task, run_trigger
    from app.discord_notify import _discord_notify_task_complete
    from autonomy import _find_task_file, _parse_task_file

    task_id = item.payload.get("task_id")
    if task_id is None:
        raise RuntimeError("scheduled-task item has no task_id in payload")

    # The enqueue-side health gate can't help items already in the queue when
    # vLLM wedges — without this wait they burn all attempts in ~90s of
    # ConnectErrors. Poll briefly for recovery before spending an attempt.
    loop = asyncio.get_event_loop()
    from autonomy import _find_task_file as _ftf, _parse_task_file as _ptf
    _t = _ptf(_ftf(task_id)) if _ftf(task_id) else None
    health_url = _model_health_url((_t or {}).get("model"))
    for _ in range(18):  # up to 90s
        if await loop.run_in_executor(None, _vllm_healthy, 4.0, health_url):
            break
        await asyncio.sleep(5)
    else:
        raise RuntimeError(f"model server {health_url} unhealthy — deferring task")

    # Pass the pool's cap so run_task can keep its own timeout strictly under it
    # and always win the race (otherwise the pool cancels it and no run record
    # is written at all).
    try:
        from workers.sources import get_sources_config
        max_dur = int(get_sources_config().get(NAME, {}).get("max_duration_seconds", 1800))
    except Exception:
        max_dur = 1800
    with run_trigger("scheduler"):
        result = await run_task(int(task_id), max_duration=max_dur)

    preview = (result.get("response_preview") or "")
    if result.get("success") and preview and "[SILENT]" not in preview:
        try:
            path = _find_task_file(task_id)
            if path:
                task = _parse_task_file(path)
                if task and task.get("notify_on_complete", True):
                    await _discord_notify_task_complete(
                        task_id, task.get("name", "Autonomy Task"), preview
                    )
        except Exception as e:
            logger.warning("Discord notify error for task #%s: %s", task_id, e)

    # Report task-level failures IN-BAND rather than raising. Raising sent the
    # item back through the queue's retry path, so a single timeout became up to
    # max_attempts full re-runs (3 x 600s on #36) before the scheduler's own
    # cooldown was ever consulted. run_task has already written the run record,
    # bumped failure_count and set the cooldown.
    status = result.get("status") or ("success" if result.get("success") else "failed")
    out = {
        "status": status,
        "summary": (result.get("response_preview") or result.get("error") or "")[:500],
        "task_id": str(task_id),
        "artifact_path": f"autonomy-runs/{task_id}/{result.get('run_id')}.md"
                         if result.get("run_id") else "",
        "response": result.get("response_preview") or "",
        "meta": result.get("meta") or {},
    }
    # #945 — this dict is a hand-written whitelist, and `claims` was not on it,
    # which severed the pilot's evidence bundle one hop upstream of the verifier:
    # `autonomy.run_task` sets `result["claims"]` for `EVIDENCE_PILOT_TASK_IDS`,
    # `pool.normalize_result` carries the key through, and `pool` verifies and
    # writes `claims_json` when it is present — but every autonomy run reached
    # the pool with no `claims` at all, so 4,922 scheduled-task rows carry an
    # empty bundle and `autonomy_health` has reported `runs_with_bundle: 0`
    # since the pilot shipped.
    #
    # Presence is the copy, not `out["claims"] = result.get("claims")`: the key
    # *being there* is the pool's pilot-scope switch (`normalize_result` maps an
    # absent key to `claims is None` → no bundle, and an empty list to a bundle
    # recording "the model emitted nothing" as a visible gap). Copying the key
    # unconditionally would not change behaviour today, but it would make the
    # adapter — not the model's output — decide which runs get scoped in.
    if "claims" in result:
        out["claims"] = result["claims"]
    return out
