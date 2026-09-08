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
import json
import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from mcp.types import Tool

from agent_mcp._shared import get_bound_session, text_result

logger = logging.getLogger("lloyd-builtin-fs")

DEFAULT_READ_LINES = 2000
LINE_PREFIX_FMT = "%6d\t%s"
GREP_BIN = "rg"


# ---------------------------------------------------------------------------
# Read tracking — what this session has actually seen of each file
# ---------------------------------------------------------------------------
#
# `Edit` was exact-match against whatever is on disk right now, with no notion
# of whether the model had ever looked at the file. Two failures follow from
# that, and the second is the expensive one:
#
#   * an edit written from memory that happens to match, which is a guess that
#     got lucky;
#   * an edit against content something else rewrote after the Read. The
#     old_string still matches, so the edit *succeeds* — and silently reverts
#     whatever the other writer did. Nothing anywhere reports it.
#
# Keyed by `os.path.realpath` so a symlink and its target are one file, and
# bounded twice: this process serves every session on the box and lives for
# days.
_READ_SESSIONS_MAX = 256
_READ_PATHS_MAX = 2000

# session_id -> realpath -> (mtime_ns, size)
_read_records: "OrderedDict[str, OrderedDict[str, tuple[int, int]]]" = OrderedDict()

# The sync handlers below run on worker threads (`asyncio.to_thread` in
# call_tool), so two sessions really do mutate this concurrently. Never held
# across anything slower than a dict operation.
_state_lock = threading.Lock()


def _stat_key(path: str) -> tuple[int, int] | None:
    """(mtime_ns, size) for `path`, or None if it cannot be stat'd."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _record_seen(session_id: str, real: str, key: tuple[int, int] | None) -> None:
    if not session_id or not real or key is None:
        return
    with _state_lock:
        paths = _read_records.get(session_id)
        if paths is None:
            paths = _read_records[session_id] = OrderedDict()
        _read_records.move_to_end(session_id)
        paths[real] = key
        paths.move_to_end(real)
        while len(paths) > _READ_PATHS_MAX:
            paths.popitem(last=False)
        while len(_read_records) > _READ_SESSIONS_MAX:
            _read_records.popitem(last=False)


def _seen(session_id: str, real: str) -> tuple[int, int] | None:
    with _state_lock:
        paths = _read_records.get(session_id)
        if not paths:
            return None
        rec = paths.get(real)
        if rec is not None:
            paths.move_to_end(real)
            _read_records.move_to_end(session_id)
        return rec


def reset_read_records() -> None:
    """Drop all read tracking. Tests only."""
    with _state_lock:
        _read_records.clear()


def _gates_enabled() -> bool:
    """Read at call time so a test can monkeypatch.setitem the config."""
    try:
        from app.config import CONFIG
        gates = (CONFIG.get("harness") or {}).get("edit_gates") or {}
        return bool(gates.get("enabled", True))
    except Exception:
        return True


@dataclass
class _Mutation:
    """One in-flight Write or Edit, built on the loop and filled by the thread.

    The handler runs in a worker thread and returns a string; everything the
    loop needs afterwards (what was written, where, whether it succeeded)
    has nowhere else to travel. Later stages — post-edit diagnostics, the
    change ledger — read the same record rather than re-stat'ing the file
    and racing whoever wrote next.
    """

    kind: str                                   # "edit" | "write"
    session_id: str = ""
    gate_on: bool = False
    path: str = ""                              # as the caller gave it, expanded
    real: str = ""                              # os.path.realpath of the above
    existed: bool = False
    pre_bytes: bytes | None = None
    post_text: str = ""
    post_stat: tuple[int, int] | None = None
    ok: bool = False


def _gate_check(mut: _Mutation) -> str | None:
    """The refusal, as a JSON error string, or None to proceed.

    Skipped entirely when no session is bound: unit tests and legacy callers
    dispatch straight into these handlers with no aggregator context, and a
    gate that fired there would fail `tests/test_mcp_layer.py` rather than
    protect anything.
    """
    if not mut.gate_on:
        return None
    if mut.kind == "write" and not mut.existed:
        # Creating a file — including through a dangling symlink — needs no
        # prior Read. There is nothing to clobber.
        return None

    rec = _seen(mut.session_id, mut.real)
    if rec is None:
        if mut.kind == "edit":
            return json.dumps({"error": (
                f"Edit refused: {mut.path} has not been Read in this session. "
                f"Read it first (a partial Read with offset/limit is enough), "
                f"then retry the Edit."
            )})
        return json.dumps({"error": (
            f"Write refused: {mut.path} already exists and has not been Read "
            f"in this session. Read it first, then Write it."
        )})

    current = _stat_key(mut.real)
    if current is not None and current != rec:
        if mut.kind == "edit":
            return json.dumps({"error": (
                f"Edit refused: {mut.path} changed on disk since you last Read "
                f"it (size or mtime differ). Read it again, then retry with the "
                f"current content."
            )})
        return json.dumps({"error": (
            f"Write refused: {mut.path} changed on disk since you last Read it. "
            f"Read it again, then retry."
        )})
    return None


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


def _read(args: dict, session_id: str = "") -> str:
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

    # Stat BEFORE opening. A write that lands between here and the read makes
    # the recorded key older than the bytes we return, so a later Edit is
    # refused as stale — which is the safe direction. Stat'ing afterwards
    # would record the writer's own key and let that Edit through.
    real = os.path.realpath(file_path)
    seen_key = _stat_key(real)

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

    # A partial Read (offset/limit) records normally: the contract is "you
    # looked at this file at this version", not "you read all of it".
    _record_seen(session_id, real, seen_key)

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


def _write(args: dict, mut: _Mutation | None = None) -> str:
    mut = mut if mut is not None else _Mutation(kind="write")
    file_path = _expand(args.get("file_path", ""))
    content = args.get("content", "")
    if not file_path:
        return json.dumps({"error": "file_path is required"})
    if not os.path.isabs(file_path):
        return json.dumps({"error": f"file_path must be absolute, got {file_path!r}"})
    p = Path(file_path)
    mut.path = file_path
    mut.real = os.path.realpath(file_path)
    # `exists()` follows symlinks, so a dangling symlink counts as absent and
    # writing through it is a create.
    mut.existed = p.exists()

    refusal = _gate_check(mut)
    if refusal is not None:
        return refusal

    if mut.existed:
        try:
            mut.pre_bytes = p.read_bytes()
        except OSError:
            mut.pre_bytes = None

    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except OSError as exc:
        return json.dumps({"error": f"failed to write {file_path}: {exc}"})

    mut.post_text = content
    mut.post_stat = _stat_key(mut.real)
    mut.ok = True
    return f"File written: {file_path} ({len(content)} chars)"


# ---------------------------------------------------------------------------
# Edit
# ---------------------------------------------------------------------------


def _edit(args: dict, mut: _Mutation | None = None) -> str:
    mut = mut if mut is not None else _Mutation(kind="edit")
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

    mut.path = file_path
    mut.real = os.path.realpath(file_path)
    mut.existed = True

    refusal = _gate_check(mut)
    if refusal is not None:
        return refusal

    try:
        pre_bytes = p.read_bytes()
    except OSError as exc:
        return json.dumps({"error": f"failed to read {file_path}: {exc}"})
    try:
        original = pre_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # Previously this escaped `read_text` as a bare UnicodeDecodeError and
        # left the aggregator, arriving at the model as an MCP exception with
        # no path in it. A binary file is a normal mistake and deserves a
        # normal error.
        return json.dumps({"error": (
            f"Edit refused: {file_path} is not valid UTF-8 text; Edit only "
            f"handles text files."
        )})

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

    mut.pre_bytes = pre_bytes

    try:
        p.write_text(updated, encoding="utf-8")
    except OSError as exc:
        return json.dumps({"error": f"failed to write {file_path}: {exc}"})

    mut.post_text = updated
    mut.post_stat = _stat_key(mut.real)
    mut.ok = True
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
                "Write a file to the local filesystem. Creates parent "
                "directories as needed. Creating a new file needs nothing "
                "first; OVERWRITING an existing one requires that you have "
                "Read it in this session and that it has not changed since — "
                "otherwise the write is refused and tells you to Read it. Use "
                "Edit for changing part of a file."
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
                "Replace exact-match text in a file. You must Read the file in "
                "this session first — a partial Read with offset/limit counts — "
                "and the file must not have changed since; the edit is refused "
                "otherwise, which is what stops an edit written from memory "
                "from silently reverting somebody else's write. Errors if "
                "`old_string` is missing or appears multiple times (unless "
                "`replace_all=true`). Your own Write or Edit refreshes the "
                "record, so consecutive edits to one file need only one Read."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to modify",
                    },
                    "old_string": {
                        "type": "string",
                        "description": (
                            "Exact text to find, including indentation and "
                            "surrounding lines needed to make it unique in "
                            "the file. Must match byte for byte."
                        ),
                    },
                    "new_string": {
                        "type": "string",
                        "description": (
                            "Replacement text. Must differ from old_string; "
                            "pass an empty string to delete the matched text."
                        ),
                    },
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
                        "description": (
                            "files_with_matches (default) lists matching file "
                            "paths, content shows matching lines, count shows "
                            "per-file match counts."
                        ),
                    },
                    "-i": {"type": "boolean", "description": "Case-insensitive match"},
                    "-n": {
                        "type": "boolean",
                        "description": "Prefix each match with its line number (content mode)",
                    },
                    "-A": {
                        "type": "integer",
                        "description": "Lines of trailing context after each match (content mode)",
                    },
                    "-B": {
                        "type": "integer",
                        "description": "Lines of leading context before each match (content mode)",
                    },
                    "-C": {
                        "type": "integer",
                        "description": "Lines of context on both sides of each match (content mode)",
                    },
                    "multiline": {
                        "type": "boolean",
                        "description": "Let the pattern match across line boundaries (. matches newline)",
                    },
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


def _append_diagnostics(text: str, mut: _Mutation) -> str:
    """Append the post-edit `<diagnostics>` block to a successful result.

    Appended only to a success string. `_shared.text_result` sets `isError`
    by sniffing a leading JSON object with an "error" key, so appending to an
    error payload would both break the JSON and flip the flag — a lint
    finding would start reading as a failed edit.

    Never raises: the model would rather have an edit with no diagnostics
    than an edit that failed because the linter did.
    """
    try:
        from agent_mcp import _edit_diagnostics as diag
        cfg = diag.config()
        if not cfg.get("python", True):
            return text
        block = diag.python_block(mut.path, mut.pre_bytes, mut.post_text,
                                  int(cfg.get("max_lines", diag.DEFAULT_MAX_LINES)))
        if block:
            return f"{text}\n\n{block}"
    except Exception:
        logger.warning("edit diagnostics failed for %s", mut.path, exc_info=True)
    return text


async def call_tool(name: str, arguments: dict):
    # Sync handlers run in a worker thread — this loop also serves SSE chat,
    # voice, and the inner-voice observer; a slow disk read must not stall it.
    session_id = get_bound_session()
    if name == "Read":
        text = await asyncio.to_thread(_read, arguments, session_id)
    elif name in ("Write", "Edit"):
        mut = _Mutation(
            kind="write" if name == "Write" else "edit",
            session_id=session_id,
            # No bound session means no gate: unit tests and legacy callers
            # dispatch straight in, and refusing them protects nothing.
            gate_on=bool(session_id) and _gates_enabled(),
        )
        handler = _write if name == "Write" else _edit
        text = await asyncio.to_thread(handler, arguments, mut)
        if mut.ok:
            # The writer updates the record, so an Edit immediately after a
            # Write or Edit by the same session is allowed without re-Reading.
            _record_seen(mut.session_id, mut.real, mut.post_stat)
            text = _append_diagnostics(text, mut)
    elif name == "Grep":
        text = await _grep(arguments)
    elif name == "Glob":
        text = await asyncio.to_thread(_glob, arguments)
    else:
        text = json.dumps({"error": f"Unknown tool: {name}"})
    return text_result(text)
