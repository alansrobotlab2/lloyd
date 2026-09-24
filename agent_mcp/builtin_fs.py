#!/usr/bin/env python3
"""Lloyd MCP Server: filesystem built-ins (Read, Write, Edit, Grep, Glob).

Recreates the Claude Code built-in tools we lost when we ripped the SDK
out. Names and input schemas mirror the SDK contracts so persisted
session JSON, the persona prompt surfaces, and inner_voice.pretooluse_deny
rules all keep working unchanged. Output formatting (e.g. cat -n line
prefix on Read) matches too.

Two refusals stand in front of every write here, in this order: the write
deny-set (`app.harness.protected_paths.write_deny_reason` — a rule about
*where* a write may land, no session required), then the Read-before-write
clobber gate below (a rule about *who looked*). They are separate on purpose:
the config switch that disables the clobber gate must not disable the other.

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

from agent_mcp._shared import (
    ErrorCode,
    _fit_not_found,
    _near_match_hint,
    get_bound_session,
    text_result,
)
from app.atomic_io import commit_lock, write_text_durable
# One byte-ceiling definition for every lane that can rewrite the two loaded
# memory files — this one, `memory_add`/`memory_replace`, `vault_write`, and the
# vault-round validator — with the number and the wording owned by `prompt_surface`.
# Stdlib-only, so it costs this module nothing to import (#1010).
from app.memory_ceiling import memory_write_error

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


def _current_call_id() -> str:
    try:
        from agent_mcp import _task_registry
        return _task_registry.current_call_id.get()
    except Exception:
        return ""


def _ledger_scope(session_id: str):
    """(session, turn) this mutation belongs to, or None for no ledger."""
    try:
        from agent_mcp import _change_ledger, _task_registry
        if not _change_ledger.enabled():
            return None
        return _change_ledger.scope(session_id, _task_registry.current_turn_id.get())
    except Exception:
        logger.warning("change ledger: scope lookup failed", exc_info=True)
        return None


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
    # Change-ledger scope and entry, resolved on the loop side and carried
    # through the thread so the snapshot is taken from the same bytes the
    # edit read — re-reading the file here would race whoever writes next.
    scope: tuple[str, str] | None = None
    call_id: str = ""
    ledger_entry: object | None = None


def _ledger_begin(mut: _Mutation, *, op: str) -> None:
    """Open this turn's entry and snapshot the pre-image, before the write.

    Failures are logged and swallowed. An edit that fails because the undo
    bookkeeping failed is strictly worse than an edit with no undo.
    """
    if mut.scope is None:
        return
    try:
        from agent_mcp import _change_ledger
        entry = _change_ledger.begin(
            mut.scope, real=mut.real, path=mut.path, op=op,
            call_id=mut.call_id,
            # Set only when a subagent made the change, so a footer can say
            # which of a turn's writes came from a Task rather than the turn.
            via_session=mut.session_id if mut.session_id.startswith("task:") else "",
        )
        _change_ledger.snapshot_pre(mut.scope, entry, mut.pre_bytes)
        mut.ledger_entry = entry
    except Exception:
        logger.warning("change ledger: begin failed for %s", mut.path, exc_info=True)


def _ledger_commit(mut: _Mutation) -> None:
    if mut.scope is None or mut.ledger_entry is None:
        return
    try:
        from agent_mcp import _change_ledger
        post = _change_ledger.sha256_bytes(mut.post_text.encode("utf-8", "replace"))
        _change_ledger.commit(mut.scope, mut.ledger_entry, post)
    except Exception:
        logger.warning("change ledger: commit failed for %s", mut.path, exc_info=True)


def _protected_path_refusal(mut: _Mutation) -> str | None:
    """The deny-set refusal for this mutation's target, or None to proceed.

    Checked before anything else in `_gate_check`, and outside the `gate_on`
    switch: this is a rule about *where* a write may land, not about whether a
    session looked at the file first, so neither an unbound caller nor the
    config switch that disables the clobber gate may open it. It runs ahead of
    the create early-return for the same reason — otherwise removing a denied
    file and writing it back is a route.

    Fails closed. A deny-set that cannot load is not a deny-set that passed,
    and the alternative — every write silently proceeding while the checker is
    broken — is the exact hole #1049 exists to close. The failure is loud in
    the result and in the log.
    """
    try:
        from app.harness.protected_paths import write_deny_reason
        label = write_deny_reason(mut.real or mut.path)
    except Exception:  # noqa: BLE001 — an unloadable checker must not read as a pass
        logger.exception("protected-path check unavailable for %s", mut.path)
        return json.dumps({
            "error": (f"{'Write' if mut.kind == 'write' else 'Edit'} refused: "
                      f"the protected-path check could not run, so {mut.path} "
                      f"is not being written. Report this rather than "
                      f"working around it."),
            "code": ErrorCode.PROTECTED_PATH,
        })
    if label is None:
        return None
    return json.dumps({
        "error": (
            f"{'Write' if mut.kind == 'write' else 'Edit'} refused: {mut.path} "
            f"is protected ({label}). This lane refuses it for every session. "
            f"Land the change through the route that validates it — "
            f"`vault_write` or `automod_vault_land` for vault and prompt "
            f"surfaces, an automod round for code — or ask Alan."
        ),
        "code": ErrorCode.PROTECTED_PATH,
    })


def _gate_check(mut: _Mutation) -> str | None:
    """The refusal, as a JSON error string, or None to proceed.

    The write deny-set runs first and unconditionally (see
    `_protected_path_refusal`). What follows is skipped entirely when no
    session is bound: unit tests and legacy callers dispatch straight into
    these handlers with no aggregator context, and a gate that fired there
    would fail `tests/test_mcp_layer.py` rather than protect anything.
    """
    refusal = _protected_path_refusal(mut)
    if refusal is not None:
        return refusal
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


def _write_target(mut: "_Mutation", p: Path) -> Path:
    """Where an atomic replace of `p` actually has to land.

    `os.replace` renames over a *name*, so given a symlink it swaps the link
    itself for a regular file and orphans whatever it pointed at. The
    truncate-in-place write this replaced followed the link instead, and for a
    dangling link that meant the write created the link's target
    (`test_write_through_a_dangling_symlink_is_a_create`). Renaming onto the
    resolved path keeps both behaviours: the link stays a link, and a dangling
    one still materialises its target. `mut.real` is that resolution, computed
    once before the lock was taken.
    """
    return Path(mut.real) if os.path.islink(mut.path) else p


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

    # Pre-image read, ledger snapshot and the replace are one critical section.
    # Read outside the lock and this is the lost update the item is about: the
    # nightly knowledge-write job edits MEMORY.md through this lane while a chat
    # turn's memory_add appends to it, and whichever read second overwrites the
    # other. `p.write_text` also opened O_TRUNC, so a `memory_read` in that
    # window returned an empty file — every writer in the repo shares one commit
    # path here for both reasons. The gate above stays outside: it is a refusal,
    # not part of the commit, and holding the lock across it would put policy
    # latency inside everyone else's wait.
    try:
        with commit_lock(mut.real):
            if mut.existed:
                try:
                    mut.pre_bytes = p.read_bytes()
                except OSError:
                    mut.pre_bytes = None

            # #1010/#507: `lloyd/MEMORY.md` and `lloyd/USER.md` load into every
            # user-platform system prompt and this is the lane the nightly
            # knowledge-write job uses — the one that grew USER.md 48,068 B →
            # 95,302 B in five days with nothing able to refuse it, because
            # `memory_add` was the only guarded writer. A refusal returns before
            # `_ledger_begin`, so a write that never happened enters no undo record.
            ceiling_msg = memory_write_error(p, content)
            if ceiling_msg:
                return json.dumps({"error": ceiling_msg})

            _ledger_begin(mut, op="write" if mut.existed else "create")

            try:
                dest = _write_target(mut, p)
                dest.parent.mkdir(parents=True, exist_ok=True)
                write_text_durable(dest, content)
            except OSError as exc:
                return json.dumps({"error": f"failed to write {file_path}: {exc}"})

            mut.post_text = content
            mut.post_stat = _stat_key(mut.real)
            mut.ok = True
            _ledger_commit(mut)
    except TimeoutError as exc:
        return json.dumps({"error": f"failed to write {file_path}: {exc}",
                           "code": "LOCK_TIMEOUT"})
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

    # Same critical section as `_write`: read, apply, replace, under one lock on
    # the resolved path. This is the lane the nightly knowledge-write job runs on
    # ("Use `Edit` for targeted changes"), so an Edit of lloyd/MEMORY.md used to
    # race a chat turn's memory_add on the same file — 4 writers x 25 appends lost
    # 82 of 100 at base. The old_string check below is inside the lock for the same
    # reason the read is: matching against bytes this lane does not hold is what
    # turns "the edit applied" into "the edit reverted someone else".
    try:
        with commit_lock(mut.real):
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
                # `original` is already held under the commit lock, so naming the
                # nearest candidate line costs one string scan and no second read
                # of a file the caller is about to re-Read anyway.
                return json.dumps({"error": _fit_not_found(
                    "old_string not found in file (must match exactly)",
                    _near_match_hint(original, old_string))})
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
            # Same ceiling as `Write`, priced on `updated` — the exact bytes about
            # to land — and allowed to shrink an over-ceiling file, which is how a
            # trim through this same lane stays possible after a refusal.
            ceiling_msg = memory_write_error(p, updated)
            if ceiling_msg:
                return json.dumps({"error": ceiling_msg})
            _ledger_begin(mut, op="edit")

            try:
                write_text_durable(_write_target(mut, p), updated)
            except OSError as exc:
                return json.dumps({"error": f"failed to write {file_path}: {exc}"})

            mut.post_text = updated
            mut.post_stat = _stat_key(mut.real)
            mut.ok = True
            _ledger_commit(mut)
    except TimeoutError as exc:
        return json.dumps({"error": f"failed to edit {file_path}: {exc}",
                           "code": "LOCK_TIMEOUT"})
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
                "Use when you know the exact path and want the contents; to locate a file use Glob or Grep first.\n\n"
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
                "Use to create a file or replace one wholesale; to change part of a file you have Read, use Edit instead.\n\n"
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
                "Use to change part of a file you have Read in this session; to replace the whole file use Write instead.\n\n"
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
                "Use to search file contents by pattern; for file names use Glob, for a known path use Read.\n\n"
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
                "Use to find files by name pattern; to search inside files use Grep instead.\n\n"
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
    """Append the post-edit blocks to a successful result.

    Two of them, from two stores. `<diagnostics>` is pyflakes on the pre- and
    post-image of *this file*; `<blast_radius>` is the code graph's inbound
    callers for any module-level interface the edit changed, which is the
    part a per-file linter structurally cannot see. Both are advisory.

    Appended only to a success string. `_shared.text_result` sets `isError`
    by sniffing a leading JSON object with an "error" key, so appending to an
    error payload would both break the JSON and flip the flag — a lint
    finding would start reading as a failed edit.

    `call_tool` runs this in a worker thread, never on the event loop
    (#726): pyflakes on two images of up to `MAX_SOURCE_BYTES` each is
    unbudgeted, and the loop also serves SSE chat, voice and the observer.
    The graph half keeps its own `RAIL_BUDGET_S` join inside
    `_edit_diagnostics`, which bounds what an edit waits for it.

    Never raises: the model would rather have an edit with no diagnostics
    than an edit that failed because the linter did.
    """
    try:
        from agent_mcp import _edit_diagnostics as diag
        from agent_mcp import code_graph
        cfg = diag.config()
        if cfg.get("python", True):
            block = diag.python_block(
                mut.path, mut.pre_bytes, mut.post_text,
                int(cfg.get("max_lines", diag.DEFAULT_MAX_LINES)))
            if block:
                text = f"{text}\n\n{block}"
        if cfg.get("blast_radius", True):
            # The live checkout is only a *fallback* root: an automod
            # worktree builds its own graph with `graph_refresh`, and that is
            # the one describing the tree actually being edited.
            blocks = diag.callers_block(
                mut.path, mut.pre_bytes, mut.post_text,
                real=mut.real, fallback_root=str(code_graph.resolve_root(None)))
            if blocks:
                text = f"{text}\n\n{blocks}"
    except Exception:
        logger.warning("edit diagnostics failed for %s", mut.path, exc_info=True)
    return text


async def _append_tsc_hint(text: str, mut: _Mutation) -> str:
    """Queue a debounced whole-project tsc and say so.

    Saying so is not decoration. tsc is whole-project (`include: ["src"]`)
    and ~5 s, so it cannot ride back on the edit — and a model told nothing
    assumes nothing is coming and either re-checks by hand or moves on.
    """
    try:
        from agent_mcp import _tsc_runner
        hint = await _tsc_runner.note_edit(mut.session_id, mut.real)
        if hint:
            return f"{text}\n{hint}"
    except Exception:
        logger.warning("tsc check could not be queued for %s", mut.path, exc_info=True)
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
            scope=_ledger_scope(session_id),
            call_id=_current_call_id(),
        )
        handler = _write if name == "Write" else _edit
        text = await asyncio.to_thread(handler, arguments, mut)
        if mut.ok:
            # The writer updates the record, so an Edit immediately after a
            # Write or Edit by the same session is allowed without re-Reading.
            _record_seen(mut.session_id, mut.real, mut.post_stat)
            # Off the loop, like the handler: pyflakes on a large file must
            # not stall every other stream this process serves (#726).
            text = await asyncio.to_thread(_append_diagnostics, text, mut)
            text = await _append_tsc_hint(text, mut)
    elif name == "Grep":
        text = await _grep(arguments)
    elif name == "Glob":
        text = await asyncio.to_thread(_glob, arguments)
    else:
        text = json.dumps({"error": f"Unknown tool: {name}"})
    return text_result(text)
