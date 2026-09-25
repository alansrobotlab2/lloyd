"""Background task registry — process-shared.

Holds the active and recently-completed background bash tasks spawned via
`Bash(run_in_background=true)`. The harness's between-turn drain pulls
completion records via `drain_completed_for_session(session_id)` and
materializes them as `<task_notification>` user messages so the model
sees them on the next iteration.

Lifetime: process-scoped. The lifecycle.shutdown_cleanup terminator
walks `list_active()` and kills outstanding subprocesses so a backend
restart doesn't leak. Output files at ``~/lloyd-data/_pipeline/tasks/<id>.log``
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
from app.paths import TASKS_DIR  # anchored to DATA_ROOT


# Session correlation. Set by the harness wrapper before each MCP
# tool dispatch (see app.harness.loop._execute_tool_call) so the
# Bash tool can stamp spawned tasks with the originating session.
current_session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_session_id", default=""
)

# The model's own one-line caption for the in-flight tool call, lifted out
# of the request's `_meta` by ``main.call_tool``. Lives beside the session
# id because it has the same lifetime and the same source: both are
# per-dispatch context the harness knows and a tool handler cannot derive.
#
# Two handlers read it, and both open a row a human reads later — the
# background-task list (``Bash(run_in_background=true)``) and the subagent
# list (``Task``). Each used to ask the model for that label itself, with
# an argument whose wording restated the caption the harness had already
# asked for; the model answered once, in whichever field it reached first,
# and on 2026-09-07 that stopped being the right one. One question, one
# field, and the answer arrives here.
current_call_summary: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_call_summary", default=""
)

# The chat turn and the individual tool call this dispatch belongs to, lifted
# out of `_meta` by `main.call_tool` beside the session id. Read by the change
# ledger, which records a file write against (session, turn) so a chat can say
# what a turn changed and offer to undo it. Both are "" for a worker turn or a
# direct `run_query` caller, and an empty turn id turns the ledger off.
current_turn_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_turn_id", default=""
)
current_call_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_call_id", default=""
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


@dataclass
class DiagnosticsRecord:
    """An out-of-band diagnostics result waiting to reach a session.

    Deliberately NOT a TaskRecord. That dataclass carries a `process` and a
    `log_fd`, and `format_notification`, `_task_row`, `list_active` and
    `terminate_all` are all specific to a background *bash* child. A
    diagnostics run has no subprocess to kill, no output file for the model
    to Read, and no place on the dashboard's background-task rows — it rides
    the same per-session drain queue and nothing else.

    It never enters `_records`, so `/state`'s task rows are untouched.

    `elsewhere_*` is the second group #694 added: new errors in files no
    session in that run edited — the caller of an edited component. They ride
    the same record rather than a second one because the two groups are one
    answer to one check, and separate because only one of them is the
    recipient's own doing.
    """

    session_id: str
    kind: str = "typescript"
    files: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    elsewhere_files: list[str] = field(default_factory=list)
    elsewhere_lines: list[str] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    status: str = "ok"                  # ok | failed | timeout
    detail: str = ""
    notified: bool = False


# task_id -> record. Module-level singleton.
_records: dict[str, TaskRecord] = {}
# Per-session FIFO of completed-but-not-yet-drained records. Holds both
# TaskRecord and DiagnosticsRecord; the drain tool branches on the type.
_pending_by_session: dict[str, list[Any]] = {}
_lock = asyncio.Lock()

# #929: the parent a CLOSED subagent's completions belong to, keyed by the
# child's `task:*` session id.
#
# A `task:*` queue key has no reader once its run is over — nothing drains a
# subagent's queue after `Task` returns — so a completion that lands afterwards
# sat here for the life of this long-lived process, promised to a model that had
# already finished. The entry is recorded when a child closes (in the same call
# that hands over whatever was already queued, `hand_off_to_parent`) and
# consulted by `_enqueue`, so a late arrival is written under the parent's key
# from the start: the key the parent's chat-turn drain pops.
# `_subagent_registry.parent_scope` is the authority on who that parent is; this
# map is only the part that has to outlive the run. Bounded, because a child
# session id is reused by nobody except a resumed run, which clears its own
# route (`unhand_off`) before it starts; an entry only needs to survive long
# enough for the child's subprocesses to exit.
_close_handoff: dict[str, str] = {}
_MAX_CLOSE_HANDOFF = 200


def unhand_off(child_session_id: str) -> None:
    """Drop the parent route for a child session that is running again. (#929)

    A resumed `Task` reuses the child's `task:*` session id, and while that run
    is live it IS the reader of the key. Leaving the route in place would send
    the resumed child's completions to the parent's queue instead of to the
    child that was told to expect them, so the route exists only while no run
    names the key. Records already handed off stay with the parent — they belong
    to the run that closed.
    """
    _close_handoff.pop(child_session_id, None)


def hand_off_to_parent(child_session_id: str, parent_session_id: str) -> list[Any]:
    """Move a closed child's queued completions onto the parent's queue.

    Returns the records moved. `notified` is left exactly as found — the
    parent's own chat-turn drain is what emits them, so they must stay
    unclaimed. Idempotent for the same pair.

    Deliberately synchronous: it mutates the same dict `_enqueue` writes under
    `_lock`, and a call that never suspends cannot interleave inside that
    critical section on a single-threaded loop.
    """
    _close_handoff[child_session_id] = parent_session_id
    while len(_close_handoff) > _MAX_CLOSE_HANDOFF:
        # Oldest first, skipping the key just written: a child that closed
        # longest ago has had the most time for its subprocesses to exit, so its
        # entry is the least likely still to be consulted. Dropping one only
        # reverts that session to the pre-#929 shape; it cannot misroute a
        # record, because the map is consulted only for the key it holds.
        for stale in list(_close_handoff):
            if stale != child_session_id:
                _close_handoff.pop(stale, None)
                break
        else:
            break
    moved: list[Any] = []
    for record in _pending_by_session.pop(child_session_id, []):
        record.session_id = parent_session_id
        _pending_by_session.setdefault(parent_session_id, []).append(record)
        moved.append(record)
    return moved


async def enqueue_diagnostics(record: DiagnosticsRecord) -> None:
    """Queue a diagnostics result for its session's next drain.

    Not re-keyed by `_close_handoff`: the tsc runner already routes a subagent's
    result through `_subagent_registry.parent_scope` before it builds the record
    (`_tsc_runner._notify_target`), so this queue has had a reader for the child
    case since then. #929 covers the producer that did not.
    """
    async with _lock:
        _pending_by_session.setdefault(record.session_id, []).append(record)


OTHER_FILE_NOTE = ("files no session in this run edited; may be another "
                   "session's work")


def format_diagnostics_notification(record: DiagnosticsRecord) -> str:
    """XML the model sees as a user message on a later iteration.

    Two groups, told apart by their elements: `<errors>` is the files this
    session edited, `<errors_outside_your_edits>` is the rest of what the run
    found (#694). One payload, because the two are one answer to one check —
    but never one list, because only one of them is the recipient's own doing.
    """
    elapsed = max(0.0, record.finished_at - record.started_at)
    files = ", ".join(record.files) or "(none)"
    elsewhere = record.elsewhere_lines if record.status == "ok" else []
    if record.status == "ok":
        summary = (f"tsc: {len(record.lines)} new error(s) in file(s) you "
                   f"edited this session")
        if elsewhere:
            summary += (f", plus {len(elsewhere)} new error(s) in "
                        f"file(s) you did not edit")
        summary += f" ({elapsed:.1f}s)"
    else:
        # Say so rather than staying silent: a model that was told a check
        # was queued and never hears back waits for it.
        summary = f"tsc check {record.status}: {record.detail}"[:300]
    body = "\n".join(record.lines)
    out = (f"<diagnostics_notification>\n"
           f"<kind>{record.kind}</kind>\n"
           f"<files>{files}</files>\n"
           f"<summary>{summary}</summary>\n"
           f"<errors>\n{body}\n</errors>\n")
    if elsewhere:
        out += ('<errors_outside_your_edits files="'
                + ", ".join(record.elsewhere_files)
                + '" note="' + OTHER_FILE_NOTE + '">\n'
                + "\n".join(elsewhere) + "\n"
                "</errors_outside_your_edits>\n")
    return out + "</diagnostics_notification>"


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

    # #929: the child that started this may have finished since. Its `task:*`
    # queue key has no reader then, so the completion is written under the
    # parent the close hand-off named — the key the parent's chat-turn drain
    # pops. Decided here rather than only swept at close because a subprocess
    # routinely exits after the run that spawned it returned, and decided under
    # the lock so a hand-off running alongside cannot leave this record on a key
    # the hand-off has just abandoned.
    async with _lock:
        session_id = _close_handoff.get(record.session_id, record.session_id)
        record.session_id = session_id
        _pending_by_session.setdefault(session_id, []).append(record)
    logger.info(
        "bg task %s status=%s rc=%s session=%s",
        record.task_id, record.status, rc, session_id,
    )


def get(task_id: str) -> TaskRecord | None:
    return _records.get(task_id)


def list_active() -> list[TaskRecord]:
    """Snapshot of records whose subprocess is still running."""
    return [r for r in _records.values() if r.status == "running"]


def list_recent(limit: int = 10) -> list[TaskRecord]:
    """Finished records, most recently finished first.

    A background task used to leave the dashboard the instant it exited —
    `list_active` filters on `status == "running"`, and nothing else
    rendered the rest. One that died three seconds in was indistinguishable
    from one that never started, which is the opposite of what a background
    task most needs to report.

    Bounded by `limit` rather than by eviction: `_records` is deliberately
    kept whole so a later `get(task_id)` can still resolve an output path
    for the model to Read.
    """
    done = [r for r in _records.values() if r.status != "running"]
    done.sort(key=lambda r: r.finished_at or 0.0, reverse=True)
    return done[:limit]


async def drain_completed_for_session(session_id: str) -> list[Any]:
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
