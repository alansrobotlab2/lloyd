#!/usr/bin/env python3
"""Lloyd MCP Server: filesystem built-ins (Read, Write, Edit, Grep, Glob).

Recreates the Claude Code built-in tools we lost when we ripped the SDK
out. Names and input schemas mirror the SDK contracts so persisted
session JSON, SOUL.md prompts, and inner_voice.pretooluse_deny rules all
keep working unchanged. Output formatting (e.g. cat -n line prefix on
Read) matches too.

Mounted into the unified Server("lloyd") via agent_mcp/main.py MODULES.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import os
from pathlib import Path

from mcp.server import Server
from mcp.types import TextContent, Tool

logger = logging.getLogger("lloyd-builtin-fs")

app = Server("lloyd-builtin-fs")

DEFAULT_READ_LINES = 2000
LINE_PREFIX_FMT = "%6d\t%s"
GREP_BIN = "rg"


def _expand(path: str) -> str:
    """Expand ~ and $VARS so callers can pass `~/obsidian` etc.

    Subprocesses don't expand `~`, and the absolute-path checks below
    reject tilde paths outright. Models reach for `~/...` constantly,
    so normalize at the boundary.
    """
    if not path:
        return path
    return os.path.expanduser(os.path.expandvars(path))


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def _read(args: dict) -> str:
    file_path = _expand(args.get("file_path", ""))
    if not file_path:
        return json.dumps({"error": "file_path is required"})
    if not os.path.isabs(file_path):
        return json.dumps({"error": f"file_path must be absolute, got {file_path!r}"})
    p = Path(file_path)
    if not p.exists():
        return json.dumps({"error": f"file does not exist: {file_path}"})
    if p.is_dir():
        return json.dumps({"error": f"path is a directory, not a file: {file_path}"})

    # offset is 1-based per the schema (matches the line numbers shown in
    # the output). Default 0 means "start at line 1". Coerce 0 → 1 so the
    # math below is uniform.
    offset = max(1, int(args.get("offset", 0) or 1))
    limit = int(args.get("limit", DEFAULT_READ_LINES) or DEFAULT_READ_LINES)
    if limit <= 0:
        limit = DEFAULT_READ_LINES

    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as exc:
        return json.dumps({"error": f"failed to read {file_path}: {exc}"})

    if not lines:
        return "<system-reminder>File exists but is empty.</system-reminder>"

    start_idx = offset - 1
    end = min(len(lines), start_idx + limit)
    selected = lines[start_idx:end]
    out_lines = []
    for i, raw in enumerate(selected, start=offset):
        # Match the SDK's behavior: strip the trailing newline before
        # joining so the prefix lines up cleanly.
        text = raw.rstrip("\n")
        out_lines.append(LINE_PREFIX_FMT % (i, text))
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def _write(args: dict) -> str:
    file_path = _expand(args.get("file_path", ""))
    content = args.get("content", "")
    if not file_path:
        return json.dumps({"error": "file_path is required"})
    if not os.path.isabs(file_path):
        return json.dumps({"error": f"file_path must be absolute, got {file_path!r}"})
    p = Path(file_path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except OSError as exc:
        return json.dumps({"error": f"failed to write {file_path}: {exc}"})
    return f"File written: {file_path} ({len(content)} chars)"


# ---------------------------------------------------------------------------
# Edit
# ---------------------------------------------------------------------------


def _edit(args: dict) -> str:
    file_path = _expand(args.get("file_path", ""))
    old_string = args.get("old_string", "")
    new_string = args.get("new_string", "")
    replace_all = bool(args.get("replace_all", False))

    if not file_path:
        return json.dumps({"error": "file_path is required"})
    if not os.path.isabs(file_path):
        return json.dumps({"error": f"file_path must be absolute, got {file_path!r}"})
    if old_string == new_string:
        return json.dumps({"error": "old_string and new_string are identical"})
    p = Path(file_path)
    if not p.exists():
        return json.dumps({"error": f"file does not exist: {file_path}"})

    try:
        original = p.read_text(encoding="utf-8")
    except OSError as exc:
        return json.dumps({"error": f"failed to read {file_path}: {exc}"})

    count = original.count(old_string)
    if count == 0:
        return json.dumps({"error": "old_string not found in file (must match exactly)"})
    if count > 1 and not replace_all:
        return json.dumps({
            "error": f"old_string occurs {count} times — pass replace_all=True or expand the old_string for uniqueness"
        })

    if replace_all:
        updated = original.replace(old_string, new_string)
        replaced = count
    else:
        updated = original.replace(old_string, new_string, 1)
        replaced = 1

    try:
        p.write_text(updated, encoding="utf-8")
    except OSError as exc:
        return json.dumps({"error": f"failed to write {file_path}: {exc}"})
    return f"Edited {file_path} ({replaced} replacement{'s' if replaced != 1 else ''})"


# ---------------------------------------------------------------------------
# Grep — wraps ripgrep
# ---------------------------------------------------------------------------


async def _grep(args: dict) -> str:
    pattern = args.get("pattern", "")
    if not pattern:
        return json.dumps({"error": "pattern is required"})
    path = _expand(args.get("path", "")) or os.getcwd()
    output_mode = args.get("output_mode", "files_with_matches")
    head_limit = int(args.get("head_limit", 0) or 0)
    multiline = bool(args.get("multiline", False))
    case_insensitive = bool(args.get("-i", False))
    show_line_numbers = bool(args.get("-n", False))
    after = int(args.get("-A", 0) or 0)
    before = int(args.get("-B", 0) or 0)
    context = int(args.get("-C", 0) or 0)
    glob = args.get("glob", "")
    file_type = args.get("type", "")

    cmd: list[str] = [GREP_BIN]
    if output_mode == "files_with_matches":
        cmd.append("-l")
    elif output_mode == "count":
        cmd.append("-c")
    # else "content" mode — ripgrep default
    if case_insensitive:
        cmd.append("-i")
    if show_line_numbers and output_mode == "content":
        cmd.append("-n")
    if multiline:
        cmd.extend(["-U", "--multiline-dotall"])
    if after:
        cmd.extend(["-A", str(after)])
    if before:
        cmd.extend(["-B", str(before)])
    if context:
        cmd.extend(["-C", str(context)])
    if glob:
        cmd.extend(["--glob", glob])
    if file_type:
        cmd.extend(["--type", file_type])
    cmd.extend(["-e", pattern, path])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
    except FileNotFoundError:
        return json.dumps({"error": f"{GREP_BIN} (ripgrep) not found on PATH"})
    except Exception as exc:
        return json.dumps({"error": f"grep failed: {exc}"})

    # ripgrep exit codes: 0 = matches, 1 = no matches, 2 = error
    if proc.returncode == 1:
        return "(no matches)"
    if proc.returncode not in (0, 1):
        err = stderr.decode("utf-8", errors="replace").strip()
        return json.dumps({"error": f"ripgrep exited {proc.returncode}: {err}"})

    out = stdout.decode("utf-8", errors="replace")
    if head_limit > 0:
        out = "\n".join(out.splitlines()[:head_limit])
    # Note: results above ~50K chars get spilled to disk by the harness
    # (app.harness.tool_result_spill). We deliberately do NOT truncate
    # here — the spill mechanism preserves the full output on disk so the
    # model can Read it back if it needs more than the preview.
    return out or "(no matches)"


# ---------------------------------------------------------------------------
# Glob — pathlib glob with mtime-desc sort
# ---------------------------------------------------------------------------


def _glob(args: dict) -> str:
    pattern = args.get("pattern", "")
    if not pattern:
        return json.dumps({"error": "pattern is required"})
    base = _expand(args.get("path", "")) or os.getcwd()
    if not os.path.isabs(base):
        return json.dumps({"error": f"path must be absolute, got {base!r}"})
    base_p = Path(base)
    if not base_p.exists():
        return json.dumps({"error": f"path does not exist: {base}"})

    try:
        matches = list(base_p.glob(pattern))
    except Exception as exc:
        return json.dumps({"error": f"glob failed: {exc}"})

    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    matches.sort(key=_mtime, reverse=True)
    if not matches:
        return "(no matches)"
    return "\n".join(str(p) for p in matches)


# ---------------------------------------------------------------------------
# MCP registration
# ---------------------------------------------------------------------------


@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="Read",
            description=(
                "Read a file from the local filesystem. Returns content with "
                "1-indexed `cat -n`-style line number prefix. Default 2000 "
                "lines per call; pass `offset` and `limit` for larger files."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the file"},
                    "offset": {"type": "integer", "description": "1-based start line (optional)"},
                    "limit": {"type": "integer", "description": "Max lines to read (default 2000)"},
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="Write",
            description=(
                "Write a file to the local filesystem. Overwrites if the file "
                "exists; creates parent directories as needed."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path"},
                    "content": {"type": "string", "description": "File contents"},
                },
                "required": ["file_path", "content"],
            },
        ),
        Tool(
            name="Edit",
            description=(
                "Replace exact-match text in a file. Errors if `old_string` is "
                "missing or appears multiple times (unless `replace_all=true`)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean", "description": "Replace all occurrences (default false)"},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        ),
        Tool(
            name="Grep",
            description=(
                "Search file content with ripgrep. Modes: files_with_matches "
                "(default, list paths), content (lines), count. Supports "
                "case-insensitive (-i), line numbers (-n), context (-A/-B/-C), "
                "glob/type filters, multiline."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern"},
                    "path": {"type": "string", "description": "Search root (default cwd)"},
                    "glob": {"type": "string", "description": "Glob filter, e.g. '*.py'"},
                    "type": {"type": "string", "description": "ripgrep type, e.g. 'py'"},
                    "output_mode": {
                        "type": "string",
                        "enum": ["files_with_matches", "content", "count"],
                    },
                    "-i": {"type": "boolean"},
                    "-n": {"type": "boolean"},
                    "-A": {"type": "integer"},
                    "-B": {"type": "integer"},
                    "-C": {"type": "integer"},
                    "multiline": {"type": "boolean"},
                    "head_limit": {"type": "integer", "description": "Max output lines"},
                },
                "required": ["pattern"],
            },
        ),
        Tool(
            name="Glob",
            description=(
                "Find files by glob pattern. Returns absolute paths sorted by "
                "modification time, newest first."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "e.g. '**/*.py'"},
                    "path": {"type": "string", "description": "Search root (default cwd)"},
                },
                "required": ["pattern"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "Read":
        text = _read(arguments)
    elif name == "Write":
        text = _write(arguments)
    elif name == "Edit":
        text = _edit(arguments)
    elif name == "Grep":
        text = await _grep(arguments)
    elif name == "Glob":
        text = _glob(arguments)
    else:
        text = json.dumps({"error": f"Unknown tool: {name}"})
    return [TextContent(type="text", text=text)]
