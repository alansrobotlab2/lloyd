"""IDE tab data endpoints — directory listing, file read, file write.

GET  /api/ide/list?path=<abs>   — entries in a directory (dirs-first, name-sort)
GET  /api/ide/file?path=<abs>   — read text file (cap 5MB; binary detection)
POST /api/ide/file              — save file with optional mtime conflict guard

All paths must be absolute. ~ and $VARS are expanded at the boundary so
the model can pass `~/lloyd` etc. and not have it rejected. Obvious traps
(/proc, /sys, /dev) are blocked outright — they're not "code you'd open
in an editor", and walking them can hang the listener.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("lloyd-server.ide")

router = APIRouter()

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_LIST_ENTRIES = 5000
BINARY_SNIFF_BYTES = 8192
_BLOCKED_PREFIXES = ("/proc", "/sys", "/dev", "/run")


def _expand(path: str) -> str:
    if not path:
        return path
    return os.path.expanduser(os.path.expandvars(path))


def _validate_abs(path: str) -> str:
    """Expand, normalize, and absolute-check a path. Raise HTTPException on bad input."""
    if not isinstance(path, str) or not path.strip():
        raise HTTPException(status_code=400, detail="path is required")
    expanded = _expand(path)
    if not os.path.isabs(expanded):
        raise HTTPException(status_code=400, detail=f"path must be absolute, got {expanded!r}")
    norm = os.path.normpath(expanded)
    for prefix in _BLOCKED_PREFIXES:
        if norm == prefix or norm.startswith(prefix + "/"):
            raise HTTPException(status_code=400, detail=f"path is in a blocked filesystem: {prefix}")
    return norm


def _is_binary(sample: bytes) -> bool:
    if b"\x00" in sample:
        return True
    # Roughly: if more than 30% of the sample is outside the common
    # printable + whitespace range, treat it as binary.
    if not sample:
        return False
    text_chars = bytes(range(32, 127)) + b"\n\r\t\b\f\v"
    nontext = sum(1 for b in sample if b not in text_chars)
    return nontext / len(sample) > 0.30


# ── GET /api/ide/list ───────────────────────────────────────────────────

@router.get("/api/ide/list")
async def get_ide_list(path: str):
    abs_path = _validate_abs(path)
    p = Path(abs_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"path does not exist: {abs_path}")
    if not p.is_dir():
        raise HTTPException(status_code=400, detail=f"path is not a directory: {abs_path}")

    entries: list[dict[str, Any]] = []
    try:
        with os.scandir(p) as it:
            for de in it:
                if len(entries) >= MAX_LIST_ENTRIES:
                    break
                try:
                    st = de.stat(follow_symlinks=False)
                    is_dir = de.is_dir(follow_symlinks=False)
                except OSError:
                    continue
                entries.append({
                    "name": de.name,
                    "isDir": is_dir,
                    "size": st.st_size if not is_dir else 0,
                    "mtime": st.st_mtime,
                })
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"permission denied: {abs_path}")

    # dirs first, then case-insensitive name sort
    entries.sort(key=lambda e: (0 if e["isDir"] else 1, e["name"].lower()))
    return JSONResponse({
        "path": abs_path,
        "entries": entries,
        "truncated": len(entries) >= MAX_LIST_ENTRIES,
    })


# ── GET /api/ide/file ───────────────────────────────────────────────────

@router.get("/api/ide/file")
async def get_ide_file(path: str):
    abs_path = _validate_abs(path)
    p = Path(abs_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"file does not exist: {abs_path}")
    if p.is_dir():
        raise HTTPException(status_code=400, detail=f"path is a directory: {abs_path}")

    try:
        st = p.stat()
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"stat failed: {e}")

    if st.st_size > MAX_FILE_BYTES:
        return JSONResponse({
            "path": abs_path,
            "size": st.st_size,
            "mtime": st.st_mtime,
            "binary": False,
            "too_large": True,
            "content": None,
        })

    try:
        with p.open("rb") as f:
            sample = f.read(BINARY_SNIFF_BYTES)
            rest = f.read()
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"permission denied: {abs_path}")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"read failed: {e}")

    if _is_binary(sample):
        return JSONResponse({
            "path": abs_path,
            "size": st.st_size,
            "mtime": st.st_mtime,
            "binary": True,
            "too_large": False,
            "content": None,
        })

    raw = sample + rest
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        content = raw.decode("utf-8", errors="replace")

    return JSONResponse({
        "path": abs_path,
        "size": st.st_size,
        "mtime": st.st_mtime,
        "binary": False,
        "too_large": False,
        "content": content,
    })


# ── POST /api/ide/file ──────────────────────────────────────────────────

@router.post("/api/ide/file")
async def post_ide_file(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be an object")

    path = body.get("path")
    content = body.get("content")
    expected_mtime = body.get("expected_mtime")

    abs_path = _validate_abs(path if isinstance(path, str) else "")
    if not isinstance(content, str):
        raise HTTPException(status_code=400, detail="content must be a string")
    p = Path(abs_path)

    if expected_mtime is not None:
        if not isinstance(expected_mtime, (int, float)):
            raise HTTPException(status_code=400, detail="expected_mtime must be a number")
        if p.exists():
            try:
                actual = p.stat().st_mtime
            except OSError:
                actual = None
            if actual is not None and abs(actual - float(expected_mtime)) > 0.0001:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "file changed externally — refusing to overwrite. "
                        f"expected mtime {expected_mtime}, got {actual}"
                    ),
                )

    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"permission denied: {abs_path}")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"write failed: {e}")

    st = p.stat()
    return JSONResponse({"path": abs_path, "size": st.st_size, "mtime": st.st_mtime})


# ── File tree CRUD ──────────────────────────────────────────────────────

_NOISE_DIRS = {
    "node_modules", ".git", ".venv", ".venvs", "venv",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "dist", "build", ".next", ".turbo", ".cache", "target",
}


@router.post("/api/ide/create")
async def post_ide_create(request: Request):
    """Create an empty file or directory.

    Body: {path: abs, type: "file" | "dir"}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be an object")
    abs_path = _validate_abs(body.get("path") or "")
    kind = body.get("type")
    if kind not in ("file", "dir"):
        raise HTTPException(status_code=400, detail='type must be "file" or "dir"')
    p = Path(abs_path)
    if p.exists():
        raise HTTPException(status_code=409, detail=f"already exists: {abs_path}")
    try:
        if kind == "file":
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("", encoding="utf-8")
        else:
            p.mkdir(parents=True, exist_ok=False)
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"permission denied: {abs_path}")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"create failed: {e}")
    return JSONResponse({"path": abs_path, "type": kind})


@router.post("/api/ide/rename")
async def post_ide_rename(request: Request):
    """Rename / move a file or directory.

    Body: {from: abs, to: abs}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be an object")
    src = _validate_abs(body.get("from") or "")
    dst = _validate_abs(body.get("to") or "")
    if not Path(src).exists():
        raise HTTPException(status_code=404, detail=f"source does not exist: {src}")
    if Path(dst).exists():
        raise HTTPException(status_code=409, detail=f"destination exists: {dst}")
    try:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        os.rename(src, dst)
    except PermissionError:
        raise HTTPException(status_code=403, detail="permission denied")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"rename failed: {e}")
    return JSONResponse({"from": src, "to": dst})


@router.post("/api/ide/delete")
async def post_ide_delete(request: Request):
    """Delete a file or directory (recursive for directories).

    Body: {path: abs}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be an object")
    abs_path = _validate_abs(body.get("path") or "")
    p = Path(abs_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"path does not exist: {abs_path}")
    try:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"permission denied: {abs_path}")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"delete failed: {e}")
    return JSONResponse({"path": abs_path, "deleted": True})


# ── Recursive file listing for Quick File Open ──────────────────────────

@router.get("/api/ide/glob")
async def get_ide_glob(root: str, limit: int = 4000):
    """Walk `root` recursively, return files only (not dirs).

    Skips noisy directories (node_modules, .git, ...) at the path level.
    Caps total at `limit` to keep payload bounded on huge repos.
    """
    abs_root = _validate_abs(root)
    rp = Path(abs_root)
    if not rp.is_dir():
        raise HTTPException(status_code=400, detail=f"root is not a directory: {abs_root}")

    def _iter():
        for dirpath, dirnames, filenames in os.walk(abs_root):
            # in-place filter dirnames so os.walk doesn't descend into noise
            dirnames[:] = [d for d in dirnames if d not in _NOISE_DIRS and not d.startswith(".")]
            for name in filenames:
                # Skip dotfiles by default — match the tree's behavior
                if name.startswith("."):
                    continue
                yield os.path.join(dirpath, name)

    files: list[str] = []
    truncated = False
    for fp in _iter():
        if len(files) >= limit:
            truncated = True
            break
        files.append(fp)
    files.sort()
    return JSONResponse({"root": abs_root, "files": files, "truncated": truncated})


# ── Git diff for gutter markers ─────────────────────────────────────────

def _find_git_root(path: str) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    cur = p if p.is_dir() else p.parent
    for candidate in [cur, *cur.parents]:
        if (candidate / ".git").exists():
            return str(candidate)
    return None


@router.get("/api/ide/git/diff")
async def get_ide_git_diff(path: str):
    """Return git diff hunks for `path` vs HEAD.

    Output: {path, hunks: [{type: "add"|"modify"|"delete", new_start, new_count}]}
    `new_start` is 1-based; `new_count` is 0 for pure deletions (we mark
    the line *above* the deletion with a delete marker).
    """
    abs_path = _validate_abs(path)
    repo = _find_git_root(abs_path)
    if not repo:
        return JSONResponse({"path": abs_path, "hunks": []})

    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "diff", "--no-color", "--unified=0", "HEAD", "--", abs_path,
            cwd=repo,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
    except FileNotFoundError:
        return JSONResponse({"path": abs_path, "hunks": []})
    except Exception as e:
        logger.debug("git diff failed: %s", e)
        return JSONResponse({"path": abs_path, "hunks": []})

    hunks = _parse_unified_diff(stdout.decode("utf-8", errors="replace"))
    return JSONResponse({"path": abs_path, "hunks": hunks})


def _parse_unified_diff(diff_text: str) -> list[dict]:
    """Extract hunk headers from `git diff --unified=0` output.

    Header line shape:  @@ -<old_start>,<old_count> +<new_start>,<new_count> @@
    Counts default to 1 when omitted.
    """
    out: list[dict] = []
    for line in diff_text.splitlines():
        if not line.startswith("@@"):
            continue
        # Slice out the part between the two @@ markers.
        try:
            mid = line.split("@@", 2)[1].strip()
            # mid looks like "-12,3 +14,4"
            old_part, new_part = mid.split(" ")
            old_part = old_part.lstrip("-")
            new_part = new_part.lstrip("+")
            old_start, _, old_count = old_part.partition(",")
            new_start, _, new_count = new_part.partition(",")
            old_count = int(old_count) if old_count else 1
            new_count = int(new_count) if new_count else 1
            new_start_i = int(new_start)
            if new_count == 0:
                kind = "delete"
            elif old_count == 0:
                kind = "add"
            else:
                kind = "modify"
            out.append({
                "type": kind,
                "new_start": new_start_i,
                "new_count": new_count,
            })
        except Exception:
            continue
    return out


# ── AI hover, actions, and inline completions ───────────────────────────

# All three reach into Lloyd's chat surface via the harness directly so
# we don't depend on session state. Each takes a small code chunk and
# returns markdown (hover) or text (actions, completions).

# Lazy-imported on first call; module imports here would force loading
# of the harness too early.
_chat_call = None


async def _ask_lloyd(prompt: str, max_tokens: int = 800) -> str:
    """Single-turn primary-model query with no session, max 1 iteration.

    Used by the IDE AI features. We rely on max_turns=1 to prevent any
    tool-dispatch loop; we collect only text_delta events so any tool
    calls the model emits are ignored. Reasoning blocks come through as
    thinking_delta and are also dropped.
    """
    global _chat_call
    if _chat_call is None:
        from app.harness import run_query, RunOptions
        _chat_call = (run_query, RunOptions)
    run_query, RunOptions = _chat_call

    options = RunOptions(
        model="primary",
        max_turns=1,
        permission_mode="bypassPermissions",
        extra_body={"max_tokens": max_tokens},
    )
    pieces: list[str] = []
    messages = [{"role": "user", "content": prompt}]
    try:
        async for evt in run_query(messages, options):
            if evt.get("type") == "text_delta":
                t = evt.get("text") or ""
                if t:
                    pieces.append(t)
            elif evt.get("type") == "result":
                break
    except Exception as e:
        logger.warning("ide AI query failed: %s", e)
        return f"_(query failed: {e})_"
    return "".join(pieces).strip()


@router.post("/api/ide/ai/hover")
async def post_ide_ai_hover(request: Request):
    """Ask Lloyd to explain a symbol in context.

    Body: {path, code, symbol, line, language?}
    Returns: {markdown: str}
    """
    body = await _read_json_obj(request)
    path = body.get("path") or ""
    code = body.get("code") or ""
    symbol = body.get("symbol") or ""
    line = int(body.get("line") or 0)
    language = body.get("language") or "code"

    if not symbol:
        return JSONResponse({"markdown": ""})

    prompt = (
        f"Explain the symbol `{symbol}` at line {line} in this {language} file "
        f"(`{os.path.basename(path)}`). Be concise — three or four lines. "
        f"If it's a function, mention what it returns. If it's a variable, what it stores. "
        f"Don't quote the entire function back; assume the reader can already see it.\n\n"
        f"```{language}\n{code[:4000]}\n```"
    )
    md = await _ask_lloyd(prompt, max_tokens=400)
    return JSONResponse({"markdown": md})


@router.post("/api/ide/ai/action")
async def post_ide_ai_action(request: Request):
    """Run a named code action (explain / docstring / types / modernize).

    Body: {path, code, range_code, action, language?}
    Returns: {result: str}      — for explain/etc this is markdown
             {edit:   str}      — for transformative actions, the new code
    """
    body = await _read_json_obj(request)
    code = body.get("code") or ""
    range_code = body.get("range_code") or code
    action = body.get("action") or ""
    language = body.get("language") or "code"
    path = body.get("path") or ""

    if action == "explain":
        prompt = (
            f"Explain what this {language} code does, in 3-5 sentences. "
            f"File: `{os.path.basename(path)}`.\n\n"
            f"```{language}\n{range_code[:6000]}\n```"
        )
        md = await _ask_lloyd(prompt, max_tokens=500)
        return JSONResponse({"result": md})

    if action == "docstring":
        prompt = (
            f"Write a concise docstring for the following {language} function or class. "
            f"Return ONLY the docstring text (with surrounding triple-quotes if appropriate "
            f"for the language), no explanation, no code fence. Match the file's existing "
            f"docstring style if you can infer one.\n\n"
            f"```{language}\n{range_code[:4000]}\n```"
        )
        text = await _ask_lloyd(prompt, max_tokens=400)
        return JSONResponse({"edit": text})

    if action == "type_hints":
        prompt = (
            f"Add Python type hints to the following code. Return ONLY the modified "
            f"code, no commentary, no code fence. Preserve all existing logic, comments, "
            f"and formatting.\n\n"
            f"```{language}\n{range_code[:6000]}\n```"
        )
        text = await _ask_lloyd(prompt, max_tokens=1500)
        return JSONResponse({"edit": _strip_codefence(text)})

    if action == "modernize":
        prompt = (
            f"Rewrite this {language} code using current idioms (latest stable runtime). "
            f"Return ONLY the rewritten code, no commentary, no code fence.\n\n"
            f"```{language}\n{range_code[:6000]}\n```"
        )
        text = await _ask_lloyd(prompt, max_tokens=1500)
        return JSONResponse({"edit": _strip_codefence(text)})

    raise HTTPException(status_code=400, detail=f"unknown action: {action}")


@router.post("/api/ide/ai/complete")
async def post_ide_ai_complete(request: Request):
    """Streaming inline completion at the cursor.

    Body: {prefix, suffix, language?, max_tokens?}
    Streams text chunks as plain text/event-stream.
    """
    body = await _read_json_obj(request)
    prefix = body.get("prefix") or ""
    suffix = body.get("suffix") or ""
    language = body.get("language") or "code"

    # Heuristic: don't complete inside string/comment heavy contexts where
    # the answer is rarely useful. Cheap pre-filter; the model can also
    # decline to suggest anything.
    short_tail = prefix.strip().split("\n")[-1] if prefix else ""
    if short_tail.startswith("#") or short_tail.startswith("//"):
        return StreamingResponse(_empty_stream(), media_type="text/plain")

    prompt = (
        f"You are an inline code completion engine. Given the prefix and suffix of a "
        f"{language} file, produce the most likely continuation at the cursor. Output "
        f"ONLY the raw text to insert — no explanation, no code fence, no echoing of the "
        f"prefix. Stop after at most one logical statement, function definition, or "
        f"3-5 lines, whichever is shorter. If nothing useful would help, output nothing.\n\n"
        f"--- PREFIX ---\n{prefix[-3000:]}\n--- CURSOR ---\n{suffix[:1000]}\n--- SUFFIX END ---"
    )

    async def gen():
        from app.harness import run_query, RunOptions
        options = RunOptions(
            model="primary",
            max_turns=1,
            permission_mode="bypassPermissions",
            extra_body={"max_tokens": 200},
        )
        try:
            async for evt in run_query([{"role": "user", "content": prompt}], options):
                if evt.get("type") == "text_delta":
                    t = evt.get("text") or ""
                    if t:
                        yield t
                elif evt.get("type") == "result":
                    break
        except Exception as e:
            logger.warning("ide AI complete failed: %s", e)

    return StreamingResponse(gen(), media_type="text/plain")


# ── Helpers ─────────────────────────────────────────────────────────────

async def _read_json_obj(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json body")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be an object")
    return body


def _strip_codefence(text: str) -> str:
    """Models sometimes wrap output in ```lang ... ``` even when told not to."""
    s = text.strip()
    if s.startswith("```"):
        # drop the opening fence (with or without language tag)
        nl = s.find("\n")
        if nl >= 0:
            s = s[nl + 1:]
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


async def _empty_stream():
    if False:
        yield ""
    return
