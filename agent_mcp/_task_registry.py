"""Background task registry — process-shared.

Holds the active and recently-completed background bash tasks spawned via
`Bash(run_in_background=true)`. The harness's between-turn drain pulls
completion records via `drain_completed_for_session(session_id)` and
materializes them as `<task_notification>` user messages so the model
sees them on the next iteration.

Lifetime: process-scoped. The lifecycle.shutdown_cleanup terminator
walks `list_active()` and kills outstanding subprocesses so a backend
restart doesn't leak. Output files at ``~/lloyd/_pipeline/tasks/<id>.log``
are NOT auto-evicted — operators can prune them out-of-band.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("lloyd-task-registry")

# Where bg-task output logs live. Created on first use.
TASKS_DIR = Path(os.path.expanduser("~/lloyd/_pipeline/tasks"))


# Session correlation. Set by the harness wrapper before each MCP
# tool dispatch (see app.harness.loop._dispatch_one_tool_call) so the
# Bash tool can stamp spawned tasks with the originating session.
current_session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_session_id", default=""
)


@dataclass
class TaskRecord:
    task_id: str
    session_id: str
    command: str
    description: str
    output_path: Path
    process: asyncio.subprocess.Process
    log_fd: int  # parent's copy of the output-file fd; closed after spawn
    started_at: float
    finished_at: float | None = None
    exit_code: int | None = None
    status: str = "running"  # running | completed | failed | killed
    notified: bool = False  # has the completion been drained yet?
    waiter_task: asyncio.Task | None = field(default=None, repr=False)


# task_id -> record. Module-level singleton.
_records: dict[str, TaskRecord] = {}
# Per-session FIFO of completed-but-not-yet-drained records.
_pending_by_session: dict[str, list[TaskRecord]] = {}
_lock = asyncio.Lock()


def new_task_id() -> str:
    """Stable, sortable, human-readable task id."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    return f"bg-{ts}-{secrets.token_hex(3)}"


async def register(
    *,
    session_id: str,
    command: str,
    description: str,
) -> tuple[TaskRecord, int]:
    """Reserve a task slot and open the output log file.

    Returns ``(record, log_fd)`` where ``log_fd`` is an os-level fd opened
    in append mode, ready to be passed as ``stdout=`` / ``stderr=`` to
    ``asyncio.create_subprocess_exec``. The caller is responsible for
    spawning the subprocess and assigning ``record.process`` + starting
    the waiter task via ``start_waiter()``.

    The output file is created empty; subsequent writes by the spawned
    process append in real time so other handlers (the model's Read on
    the path) see the most recent output.
    """
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    task_id = new_task_id()
    output_path = TASKS_DIR / f"{task_id}.log"
    # O_APPEND so concurrent writes (if anything else ever touches the
    # file) don't overwrite each other. O_CREAT|O_WRONLY for spawn.
    log_fd = os.open(output_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)

    record = TaskRecord(
        task_id=task_id,
        session_id=session_id,
        command=command,
        description=description or command,
        output_path=output_path,
        process=None,  # type: ignore[arg-type]
        log_fd=log_fd,
        started_at=time.time(),
    )
    async with _lock:
        _records[task_id] = record
    return record, log_fd


def attach_process(record: TaskRecord, proc: asyncio.subprocess.Process) -> None:
    """Wire a spawned subprocess to its registry record."""
    record.process = proc


def start_waiter(record: TaskRecord) -> None:
    """Kick off the background coroutine that awaits completion and enqueues
    a pending notification onto the session's drain queue.
    """
    record.waiter_task = asyncio.create_task(
        _await_and_complete(record), name=f"bg-task-waiter-{record.task_id}"
    )


async def _await_and_complete(record: TaskRecord) -> None:
    try:
        rc = await record.process.wait()
    except asyncio.CancelledError:
        # The process is still running but the waiter was cancelled
        # (probably backend shutdown). Let kill_all() handle termination;
        # don't enqueue a notification here.
        return
    except Exception as exc:
        logger.warning("waiter for %s raised: %s", record.task_id, exc)
        rc = -1
    finally:
        # Close the parent's copy of the output fd. The child still owns
        # its inherited copy until exit, so output is preserved.
        try:
            os.close(record.log_fd)
        except OSError:
            pass

    record.finished_at = time.time()
    record.exit_code = rc
    if record.status == "killed":
        # already set by terminate(); leave as-is
        pass
    elif rc == 0:
        record.status = "completed"
    else:
        record.status = "failed"

    async with _lock:
        _pending_by_session.setdefault(record.session_id, []).append(record)
    logger.info(
        "bg task %s status=%s rc=%s session=%s",
        record.task_id, record.status, rc, record.session_id,
    )


def get(task_id: str) -> TaskRecord | None:
    return _records.get(task_id)


def list_active() -> list[TaskRecord]:
    """Snapshot of records whose subprocess is still running."""
    return [r for r in _records.values() if r.status == "running"]


async def drain_completed_for_session(session_id: str) -> list[TaskRecord]:
    """Pop all pending completion records for a session.

    Marks each as `notified=True` so re-draining (e.g. on session
    refresh) doesn't double-emit. Records remain accessible via
    ``get(task_id)`` so the model's later ``Read`` on the output path
    keeps working.
    """
    async with _lock:
        pending = _pending_by_session.pop(session_id, [])
    for r in pending:
        r.notified = True
    return pending


async def terminate_all() -> None:
    """Kill every running background task. Called from lifespan shutdown.

    Closes the parent's log fd, sends SIGTERM, and gives each process a
    short window to exit before SIGKILL. Best-effort.
    """
    active = list_active()
    if not active:
        return
    logger.info("shutdown: terminating %d background task(s)", len(active))
    for r in active:
        r.status = "killed"
        try:
            r.process.terminate()
        except (ProcessLookupError, AttributeError):
            pass
    # Give each up to 2s to exit, then force-kill.
    deadline = time.time() + 2.0
    for r in active:
        remaining = max(0.0, deadline - time.time())
        try:
            await asyncio.wait_for(r.process.wait(), timeout=remaining)
        except (asyncio.TimeoutError, AttributeError):
            try:
                r.process.kill()
            except (ProcessLookupError, AttributeError):
                pass
        except Exception:
            pass
    # Cancel any waiter tasks that are still hanging.
    for r in active:
        if r.waiter_task and not r.waiter_task.done():
            r.waiter_task.cancel()


def format_notification(record: TaskRecord) -> str:
    """XML wrapper the model sees as a user message between turns."""
    elapsed_s = (record.finished_at or time.time()) - record.started_at
    summary = f'Background command "{record.description}" '
    if record.status == "completed":
        summary += f"completed (exit code {record.exit_code})"
    elif record.status == "failed":
        summary += f"failed (exit code {record.exit_code})"
    elif record.status == "killed":
        summary += "was terminated"
    else:
        summary += f"finished with status {record.status}"

    return (
        f"<task_notification>\n"
        f"<task_id>{record.task_id}</task_id>\n"
        f"<status>{record.status}</status>\n"
        f"<exit_code>{record.exit_code if record.exit_code is not None else ''}</exit_code>\n"
        f"<output_file>{record.output_path}</output_file>\n"
        f"<elapsed_seconds>{elapsed_s:.1f}</elapsed_seconds>\n"
        f"<summary>{summary}</summary>\n"
        f"</task_notification>"
    )
