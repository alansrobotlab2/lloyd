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

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

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
