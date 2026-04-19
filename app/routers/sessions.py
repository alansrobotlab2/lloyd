"""Session-list / transcript / active-proc / cancel endpoints."""

import json
import os
import time

import psutil
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.paths import SESSIONS_DIR
from app.sessions_io import is_session_active, get_cancel_event


router = APIRouter()


@router.get("/api/sessions")
async def list_sessions():
    sessions = []
    for sf in sorted(SESSIONS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            data = json.loads(sf.read_text())
            if data.get("platform") == "autonomy":
                continue

            mtime = sf.stat().st_mtime
            delta = time.time() - mtime
            if delta < 60:
                relative_time = "just now"
            elif delta < 3600:
                relative_time = f"{int(delta / 60)}m ago"
            elif delta < 86400:
                relative_time = f"{int(delta / 3600)}h ago"
            else:
                relative_time = f"{int(delta / 86400)}d ago"

            sessions.append({
                "id": data.get("session_id", sf.stem),
                "session_key": data.get("session_id", sf.stem),
                "preview": data.get("preview", ""),
                "last_active": relative_time,
                "platform": data.get("platform", "mission-control"),
                "model": data.get("model", ""),
            })
        except Exception:
            continue

    return JSONResponse({"sessions": sessions[:50], "count": len(sessions)})


@router.get("/api/messages/{session_id}")
async def get_messages(session_id: str):
    """Load messages for a session from stored session metadata."""
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    data = json.loads(meta_path.read_text())
    return JSONResponse({
        "session_key": session_id,
        "model": data.get("model", ""),
        "messages": data.get("messages", []),
    })


def _find_active_claude_procs() -> list[dict]:
    """Find live claude SDK subprocesses spawned by this server process."""
    server_pid = os.getpid()
    results = []
    try:
        parent = psutil.Process(server_pid)
        children = parent.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return results

    sdk_to_session: dict[str, str] = {}
    for sf in sorted(SESSIONS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)[:200]:
        try:
            data = json.loads(sf.read_text())
            sdk_id = data.get("sdk_session_id")
            if sdk_id:
                sdk_to_session[sdk_id] = data.get("session_id", sf.stem)
        except Exception:
            continue

    for child in children:
        try:
            if "claude" not in child.name():
                continue
            cmdline = child.cmdline()
            resume_id = None
            model = None
            if "--resume" in cmdline:
                idx = cmdline.index("--resume")
                if idx + 1 < len(cmdline):
                    resume_id = cmdline[idx + 1]
            if "--model" in cmdline:
                idx = cmdline.index("--model")
                if idx + 1 < len(cmdline):
                    model = cmdline[idx + 1]

            session_id = sdk_to_session.get(resume_id) if resume_id else None
            preview = ""
            created_at = None
            if session_id:
                try:
                    sf_data = json.loads((SESSIONS_DIR / f"{session_id}.json").read_text())
                    preview = sf_data.get("preview", "")
                    created_at = sf_data.get("created_at")
                except Exception:
                    pass

            results.append({
                "pid": child.pid,
                "sdk_session_id": resume_id,
                "session_id": session_id,
                "model": model,
                "preview": preview,
                "created_at": created_at,
                "streaming": is_session_active(session_id) if session_id else False,
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return results


@router.get("/api/sessions/active-procs")
async def get_active_procs():
    """List active SDK subprocess sessions — includes ones whose SSE connection has dropped."""
    return JSONResponse({"procs": _find_active_claude_procs()})


@router.post("/api/sessions/{session_id}/kill-proc")
async def kill_session_proc(session_id: str):
    """Kill the SDK subprocess for a session (hard kill, bypasses cancel signal)."""
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    data = json.loads(meta_path.read_text())
    sdk_session_id = data.get("sdk_session_id")

    cancel_event = get_cancel_event(session_id)
    if cancel_event:
        cancel_event.set()

    server_pid = os.getpid()
    killed = False
    try:
        parent = psutil.Process(server_pid)
        for child in parent.children(recursive=True):
            try:
                if "claude" not in child.name():
                    continue
                cmdline = child.cmdline()
                if sdk_session_id and "--resume" in cmdline:
                    idx = cmdline.index("--resume")
                    if idx + 1 < len(cmdline) and cmdline[idx + 1] == sdk_session_id:
                        child.kill()
                        killed = True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    return JSONResponse({"killed": killed, "session_id": session_id})


@router.get("/api/sessions/{session_id}/status")
async def get_session_status(session_id: str):
    """Check whether a session has an active or queued turn."""
    return JSONResponse({"streaming": is_session_active(session_id)})


@router.post("/api/sessions/{session_id}/cancel")
async def cancel_session(session_id: str):
    """Request cancellation of the currently-running turn."""
    cancel_event = get_cancel_event(session_id)
    if cancel_event is None:
        return JSONResponse({"cancelled": False, "detail": "Session is not streaming"})
    cancel_event.set()
    return JSONResponse({"cancelled": True})


@router.post("/api/sessions/clear")
async def clear_session(request: Request):
    data = await request.json()
    session_id = data.get("session_id", "")
    if session_id:
        meta_path = SESSIONS_DIR / f"{session_id}.json"
        if meta_path.exists():
            meta_path.unlink()
    return JSONResponse({"success": True})
