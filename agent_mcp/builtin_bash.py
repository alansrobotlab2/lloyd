#!/usr/bin/env python3
"""Lloyd MCP Server: Bash built-in.

Recreates Claude Code's Bash tool. Merged stdout/stderr, configurable
timeout (default 120s, hard cap 600s), 30000-char output truncation —
matching the SDK's behavior so SOUL.md prompts and persisted sessions
keep working unchanged.

IMPORTANT: this server does NOT enforce inner_voice.pretooluse_deny
rules. Those run UPSTREAM in the harness's PreToolUse hook (see
app/inner_voice/heuristics.py wired through app/harness/loop.py). If
you ever invoke this server outside the harness — e.g. by spawning it
directly via mcp-cli — denial does not apply. That's by design: the
deny ruleset is bound to a Lloyd session_id and lives at the agent
layer, not the tool layer.

Background mode (``run_in_background=true``) spawns the command with
stdout/stderr redirected to a file under ``~/lloyd/_pipeline/tasks/``,
returns the task id and output path immediately, and lets the harness
loop drain a completion notification on a later turn. See
``agent_mcp/_task_registry.py`` for the registry and the harness drain
hook in ``app/harness/loop.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
from typing import Any

from mcp.server import Server
from mcp.types import TextContent, Tool

from agent_mcp import _task_registry
from agent_mcp._shared import get_bound_session

logger = logging.getLogger("lloyd-builtin-bash")

app = Server("lloyd-builtin-bash")

DEFAULT_TIMEOUT_MS = 120_000
MAX_TIMEOUT_MS = 600_000
OUTPUT_TRUNCATE_CHARS = 30_000


def _kill_proc_tree(proc: asyncio.subprocess.Process) -> None:
    """Send SIGTERM (then SIGKILL after a short grace period) to the
    process group of `proc`. Requires the proc to have been spawned with
    start_new_session=True so it's the leader of its own pgid.
    """
    if proc.returncode is not None:
        return
    pid = proc.pid
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        try:
            proc.terminate()
        except ProcessLookupError:
            return

    async def _coup_de_grace() -> None:
        await asyncio.sleep(2.0)
        if proc.returncode is None:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    try:
        asyncio.get_running_loop().create_task(_coup_de_grace())
    except RuntimeError:
        pass


async def _bash(args: dict[str, Any]) -> str:
    command = args.get("command", "")
    if not command or not isinstance(command, str):
        return json.dumps({"error": "command (string) is required"})

    timeout_ms = int(args.get("timeout", DEFAULT_TIMEOUT_MS) or DEFAULT_TIMEOUT_MS)
    if timeout_ms <= 0:
        timeout_ms = DEFAULT_TIMEOUT_MS
    timeout_ms = min(timeout_ms, MAX_TIMEOUT_MS)
    timeout_s = timeout_ms / 1000.0

    if args.get("run_in_background"):
        return await _spawn_background(command, args.get("description", ""))

    try:
        # start_new_session=True puts the shell (and everything it
        # spawns — pipes, valgrind, child binaries) into its own process
        # group. That lets us SIGTERM/SIGKILL the whole tree on cancel
        # or timeout via os.killpg. Without this, killing only the shell
        # leaves grandchildren (e.g. valgrind under `timeout`) running.
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=os.getcwd(),
            start_new_session=True,
        )
    except Exception as exc:
        return json.dumps({"error": f"failed to spawn shell: {exc}"})

    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        _kill_proc_tree(proc)
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
        return json.dumps({
            "error": f"command timed out after {timeout_ms}ms",
            "command": command,
        })
    except asyncio.CancelledError:
        # Harness cancelled this dispatch (Stop button → cancel_event →
        # call_tool task cancelled). Kill the whole process group so
        # long-running children (valgrind, etc.) don't outlive the turn.
        _kill_proc_tree(proc)
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
        raise

    output = stdout.decode("utf-8", errors="replace") if stdout else ""
    truncated = False
    if len(output) > OUTPUT_TRUNCATE_CHARS:
        output = output[:OUTPUT_TRUNCATE_CHARS] + "\n[... output truncated ...]"
        truncated = True

    rc = proc.returncode if proc.returncode is not None else -1

    if rc != 0:
        # Surface non-zero exit + output. Don't wrap in JSON — the SDK
        # Bash returned the raw mixed stream so the model can read
        # error messages directly.
        suffix = f"\n[exit code: {rc}]"
        if truncated:
            suffix = f"\n[truncated]" + suffix
        return (output or "(no output)") + suffix

    if truncated:
        return output  # truncation marker is already inside `output`
    return output if output else "(no output)"


async def _spawn_background(command: str, description: str) -> str:
    """Spawn the command detached, log to disk, return task descriptor.

    The subprocess inherits its own stdout/stderr (the open file fd we
    pass via ``create_subprocess_exec``), so output flushes to disk in
    real time without the harness in the loop. A waiter task watches
    `proc.wait()` and pushes a completion record onto the registry; the
    harness drains it on the next turn boundary.
    """
    session_id = get_bound_session()
    record, log_fd = await _task_registry.register(
        session_id=session_id, command=command, description=description,
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", command,
            stdout=log_fd,
            stderr=asyncio.subprocess.STDOUT,
            cwd=os.getcwd(),
            close_fds=True,
        )
    except Exception as exc:
        try:
            os.close(log_fd)
        except OSError:
            pass
        return json.dumps({"error": f"failed to spawn shell: {exc}"})

    _task_registry.attach_process(record, proc)
    _task_registry.start_waiter(record)

    return json.dumps({
        "task_id": record.task_id,
        "output_file": str(record.output_path),
        "started_at": record.started_at,
        "session_id": session_id,
        "note": (
            f"Command running in background. Read {record.output_path} "
            "to inspect progress. A <task_notification> will appear on a "
            "later turn when it completes."
        ),
    })


async def _bg_task_drain(args: dict[str, Any]) -> str:
    """Internal harness helper: pop pending background-task completions
    for the calling session and return them as JSON.

    Hidden from the LLM (filtered by the leading-underscore rule in
    ``app.harness.tool_schema.build_tool_list``); only the harness's
    between-turn drain hook calls this.
    """
    session_id = get_bound_session()
    records = await _task_registry.drain_completed_for_session(session_id)
    return json.dumps({
        "notifications": [
            {
                "task_id": r.task_id,
                "status": r.status,
                "exit_code": r.exit_code,
                "output_file": str(r.output_path),
                "xml": _task_registry.format_notification(r),
            }
            for r in records
        ]
    })


@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="Bash",
            description=(
                "Execute a bash command. Merged stdout/stderr, default "
                "timeout 120000ms (max 600000ms), output truncated at 30000 "
                "characters. Inner voice safety rules apply upstream."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to run"},
                    "description": {
                        "type": "string",
                        "description": "Short human-readable description (informational only)",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Max ms before kill (default 120000, max 600000)",
                    },
                    "run_in_background": {
                        "type": "boolean",
                        "description": (
                            "If true, spawn the command detached and return "
                            "immediately with a task_id and output_file path. "
                            "Read the output_file to inspect progress; a "
                            "<task_notification> message will appear on a later "
                            "turn when the command exits."
                        ),
                    },
                },
                "required": ["command"],
            },
        ),
        Tool(
            name="_BackgroundTaskDrain",
            description=(
                "Internal: pop pending background-task completion records "
                "for the current session. Hidden from the model; called by "
                "the harness drain hook between turns."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "Bash":
        text = await _bash(arguments)
    elif name == "_BackgroundTaskDrain":
        text = await _bg_task_drain(arguments)
    else:
        text = json.dumps({"error": f"Unknown tool: {name}"})
    return [TextContent(type="text", text=text)]
