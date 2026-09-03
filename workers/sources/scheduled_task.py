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
_state = {"vllm_down_logged": False, "startup_checked": False,
          "stall_streak": 0, "stall_alerted_at": None}


def _vllm_healthy(timeout: float = 4.0, url: str | None = None) -> bool:
    try:
        with urllib.request.urlopen(url or _VLLM_HEALTH_URL, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def _unparseable_task_files() -> list[str]:
    """Task files the scheduler cannot parse (invisible to dispatch)."""
    import re
    from pathlib import Path
    import autonomy
    d = Path.home() / "obsidian" / "autonomy"
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
    import datetime as _dt
    import autonomy
    all_tasks = autonomy._all_runnable_tasks()
    active = _active_task_ids(queue)
    now = _dt.datetime.now(_dt.timezone.utc)
    overdue = []
    for t in all_tasks:
        if not autonomy._is_task_due(t, all_tasks):
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


async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> None:
    from autonomy import get_due_tasks

    loop = asyncio.get_event_loop()

    # One-time startup validation: loudly surface any task file the scheduler
    # cannot parse (would otherwise be silently dropped — the 2026-05-28 stall).
    if not _state["startup_checked"]:
        _state["startup_checked"] = True
        bad = await loop.run_in_executor(None, _unparseable_task_files)
        if bad:
            msg = (f"{len(bad)} autonomy task file(s) unparseable and INVISIBLE to "
                   f"the scheduler: {', '.join(bad)}")
            logger.error("STARTUP: %s", msg)
            await _alert(msg)

    # Reset tasks stuck in_progress past their timeout (e.g. worker died
    # mid-run). Cheap relative to the get_due_tasks parse below.
    import autonomy
    recovered = await loop.run_in_executor(None, autonomy.recover_stuck_tasks)
    if recovered:
        logger.warning("Recovered %d stuck task(s): %s", len(recovered), recovered)

    # vLLM health gate — skip this tick entirely if the model server is down.
    if not await loop.run_in_executor(None, _vllm_healthy):
        if not _state["vllm_down_logged"]:
            logger.warning("scheduled-task: vLLM unhealthy — pausing enqueue until it recovers")
            _state["vllm_down_logged"] = True
        return
    if _state["vllm_down_logged"]:
        logger.info("scheduled-task: vLLM healthy again — resuming enqueue")
        _state["vllm_down_logged"] = False

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
    from autonomy import run_task
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
    return {
        "status": status,
        "summary": (result.get("response_preview") or result.get("error") or "")[:500],
        "task_id": str(task_id),
        "artifact_path": f"autonomy-runs/{task_id}/{result.get('run_id')}.md"
                         if result.get("run_id") else "",
        "response": result.get("response_preview") or "",
        "meta": result.get("meta") or {},
    }
