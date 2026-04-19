"""scheduled-task source — wraps the existing autonomy task files.

Reads ~/obsidian/autonomy/*.md each tick, evaluates `due-ness` per task
(frequency / runs_per_day / preferred_hours / depends_on), and enqueues the
highest-priority due tasks. Execution delegates to autonomy.run_task() which
still writes the per-task markdown run records under autonomy-runs/{task_id}/.
"""

from __future__ import annotations

import asyncio
import logging
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


async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> None:
    from autonomy import get_due_tasks

    loop = asyncio.get_event_loop()
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

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_task, int(task_id))

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
