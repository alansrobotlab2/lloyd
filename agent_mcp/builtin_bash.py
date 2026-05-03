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

Backgrounded shells (`run_in_background=true`) are not yet supported —
return an error so the model knows to retry without that flag. The
SDK's BashOutput / KillShell pair would be added alongside if a real
workflow needs them.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from mcp.server import Server
from mcp.types import TextContent, Tool

logger = logging.getLogger("lloyd-builtin-bash")

app = Server("lloyd-builtin-bash")

DEFAULT_TIMEOUT_MS = 120_000
MAX_TIMEOUT_MS = 600_000
OUTPUT_TRUNCATE_CHARS = 30_000


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
        return json.dumps({
            "error": "run_in_background is not supported by this Bash server. "
                     "Re-issue without that flag, or wrap your command in nohup/disown."
        })

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=os.getcwd(),
        )
    except Exception as exc:
        return json.dumps({"error": f"failed to spawn shell: {exc}"})

    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
        return json.dumps({
            "error": f"command timed out after {timeout_ms}ms",
            "command": command,
        })

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
                        "description": "Not supported — must be false or omitted",
                    },
                },
                "required": ["command"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "Bash":
        text = await _bash(arguments)
    else:
        text = json.dumps({"error": f"Unknown tool: {name}"})
    return [TextContent(type="text", text=text)]
