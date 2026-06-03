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
# Stall alarm: a task overdue by more than this multiple of its interval (with its
# dependency met) should have run; if any are for N consecutive checks, alert once.
_STALL_INTERVAL_MULT = 2.5
_STALL_ALARM_TICKS = 5
_state = {"vllm_down_logged": False, "startup_checked": False,
          "stall_streak": 0, "stall_alerted": False}


def _vllm_healthy(timeout: float = 4.0) -> bool:
    try:
        with urllib.request.urlopen(_VLLM_HEALTH_URL, timeout=timeout) as r:
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


def _grossly_overdue() -> list[int]:
    """Tasks that are DUE RIGHT NOW (pass every gate incl. preferred-hours and
    deps) yet are overdue by > _STALL_INTERVAL_MULT * interval. A task that is
    due and this stale but still isn't being dispatched means the dispatch path
    is broken (the 2026-05-28 silent stall). Nightly tasks merely waiting for
    their window are NOT due now, so they don't false-positive here."""
    import datetime as _dt
    import autonomy
    all_tasks = autonomy._all_runnable_tasks()
    now = _dt.datetime.now(_dt.timezone.utc)
    overdue = []
    for t in all_tasks:
        if not autonomy._is_task_due(t, all_tasks):
            continue
        interval = autonomy._frequency_interval_seconds(t)
        last = autonomy._parse_iso(t.get("last_run"))
        if not interval or not last:
            continue
        if (now - last).total_seconds() > _STALL_INTERVAL_MULT * interval:
            overdue.append(t.get("id"))
    return overdue


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

    # vLLM health gate — skip this tick entirely if the model server is down.
    if not await loop.run_in_executor(None, _vllm_healthy):
        if not _state["vllm_down_logged"]:
            logger.warning("scheduled-task: vLLM unhealthy — pausing enqueue until it recovers")
            _state["vllm_down_logged"] = True
        return
    if _state["vllm_down_logged"]:
        logger.info("scheduled-task: vLLM healthy again — resuming enqueue")
        _state["vllm_down_logged"] = False

    # Stall alarm — tasks grossly overdue with deps met should have run.
    overdue = await loop.run_in_executor(None, _grossly_overdue)
    if overdue:
        _state["stall_streak"] += 1
        if _state["stall_streak"] >= _STALL_ALARM_TICKS and not _state["stall_alerted"]:
            _state["stall_alerted"] = True
            msg = (f"autonomy scheduler may be stalled: {len(overdue)} task(s) overdue "
                   f">{_STALL_INTERVAL_MULT}x their interval with deps met "
                   f"(ids: {overdue[:15]})")
            logger.error("%s", msg)
            await _alert(msg)
    else:
        _state["stall_streak"] = 0
        _state["stall_alerted"] = False

    due = await loop.run_in_executor(None, get_due_tasks)

    for task in due:
        task_id = task.get("id")
        if task_id is None:
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

    result = await run_task(int(task_id))

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

    if not result.get("success"):
        raise RuntimeError(result.get("error") or "task failed")

    return {
        "summary": (result.get("response_preview") or "")[:500],
        "task_id": str(task_id),
        "artifact_path": f"autonomy-runs/{task_id}/{result.get('run_id')}.md",
        "response": result.get("response_preview") or "",
    }
