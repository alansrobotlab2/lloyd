"""Backend lifecycle handlers — startup orphan-reaper and graceful shutdown.

The `claude_agent_sdk` spawns a bundled `claude` CLI subprocess per turn via
`query()`. Under normal flow the SDK's `finally: await query.close()` tears it
down. But if the backend is SIGKILL'd or crashes mid-turn, the subprocess is
reparented to PID 1 / the container shim and keeps spinning in an epoll loop
on dead pipes, burning a full core. Three of these accumulated over two days
and pegged the host before we noticed.

Two safety nets:
  * startup reaper — on boot, kill any bundled-claude process that isn't a
    descendant of this backend (i.e., leaked from a prior run).
  * shutdown handler — on SIGTERM, cancel in-flight turns, give the SDK a
    moment to close its own subprocess, then hard-kill any child `claude`
    still alive before uvicorn returns.
"""

import asyncio
import logging
import os

import psutil

from app.sessions_io import _session_queues

logger = logging.getLogger("lloyd-server.lifecycle")

# Distinctive substring of the lloyd-owned SDK CLI path — avoids killing
# unrelated `claude` processes (e.g. the user's VSCode extension).
_SDK_CLI_MARKER = ".venvs/lloyd/lib/python3.12/site-packages/claude_agent_sdk/_bundled/claude"


def _is_lloyd_sdk_claude(proc: psutil.Process) -> bool:
    try:
        if "claude" not in proc.name():
            return False
        cmdline = proc.cmdline()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    return any(_SDK_CLI_MARKER in arg for arg in cmdline)


def reap_orphaned_sdk_subprocesses() -> None:
    """Kill any bundled-claude process that isn't a descendant of us."""
    my_pid = os.getpid()
    try:
        me = psutil.Process(my_pid)
        my_descendants = {p.pid for p in me.children(recursive=True)}
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        my_descendants = set()

    killed = []
    for proc in psutil.process_iter(attrs=["pid"]):
        pid = proc.info["pid"]
        if pid == my_pid or pid in my_descendants:
            continue
        if not _is_lloyd_sdk_claude(proc):
            continue
        try:
            proc.kill()
            killed.append(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning(f"Could not kill orphaned SDK subprocess {pid}: {e}")

    if killed:
        logger.warning(f"Reaped {len(killed)} orphaned SDK subprocess(es): {killed}")
    else:
        logger.info("No orphaned SDK subprocesses found at startup")


async def shutdown_cleanup(grace_seconds: float = 2.0) -> None:
    """Cancel in-flight turns, wait briefly for clean SDK teardown, then hard-kill stragglers."""
    # 1. Signal cancel on every active turn so `_run_turn` loops break out of
    #    their `async for message in query(...)` and let the SDK's finally
    #    block run `query.close()`.
    consumer_tasks = []
    for session_id, q in list(_session_queues.items()):
        try:
            q.cancel_event.set()
        except Exception:
            pass
        if q.consumer_task and not q.consumer_task.done():
            consumer_tasks.append(q.consumer_task)

    if consumer_tasks:
        logger.info(f"Shutdown: waiting up to {grace_seconds}s for {len(consumer_tasks)} consumer task(s)")
        for t in consumer_tasks:
            t.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(*consumer_tasks, return_exceptions=True),
                timeout=grace_seconds,
            )
        except asyncio.TimeoutError:
            logger.warning("Shutdown: consumer tasks did not finish within grace period")

    # 2. Safety net — anything still alive gets SIGKILL'd so it can't become
    #    an orphan. psutil.children() includes descendants via recursive=True.
    try:
        me = psutil.Process(os.getpid())
        stragglers = [c for c in me.children(recursive=True) if _is_lloyd_sdk_claude(c)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        stragglers = []

    for child in stragglers:
        try:
            child.kill()
            logger.warning(f"Shutdown: hard-killed straggler SDK subprocess {child.pid}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
