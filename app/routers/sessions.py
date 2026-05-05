"""Session-list / transcript / active-proc / cancel / inject endpoints."""

import json
import os
import time

import psutil
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.paths import SESSIONS_DIR
from app.sessions_io import (
    is_session_active,
    get_cancel_event,
    get_queue_state,
    drain_pending,
    get_active_session_id,
    enqueue_ambient_prefetch,
    peek_ambient_prefetch,
    set_ambient_decision,
    get_current_turn,
    AmbientPrefetchEntry,
)


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
                # Inner Voice (#345): surface experiment tag + opt-in flag
                # for the Inner Voice tab's session list.
                "experiment_id": data.get("experiment_id"),
                "inner_voice": bool(data.get("inner_voice", False)),
            })
        except Exception:
            continue

    return JSONResponse({"sessions": sessions[:50], "count": len(sessions)})


@router.get("/api/sessions/{session_id}/todos")
async def get_session_todos(session_id: str):
    """Return the session's TodoWrite checklist (empty list if unset).

    The TodoWrite MCP tool persists the array under the ``todos`` key in
    the session JSON. The frontend re-fetches this endpoint after every
    TodoWrite tool result; no SSE event needed.
    """
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    data = json.loads(meta_path.read_text())
    return JSONResponse({"todos": data.get("todos", [])})


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
        # Inner Voice: surface experiment tag + opt-in flag +
        # user-turn evaluation flag for the Inner Voice chat tab.
        "experiment_id": data.get("experiment_id"),
        "inner_voice": bool(data.get("inner_voice", False)),
        "inner_voice_evaluate_user_turns": bool(
            data.get("inner_voice_evaluate_user_turns", False)
        ),
    })


@router.patch("/api/sessions/{session_id}")
async def patch_session(session_id: str, request: Request):
    """Patch session metadata. Stage 0 (#345): supports `experiment_id`
    (string or null) and `inner_voice` (bool) for Inner Voice opt-in.

    Future fields can be added here without breaking the wire shape:
    callers send a partial dict; only known keys are applied.
    """
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")

    # Whitelist of patchable keys. Anything else is silently ignored —
    # keeps the endpoint forward-compatible with frontend builds that
    # may try to send unknown fields.
    allowed_keys = {"experiment_id", "inner_voice", "inner_voice_evaluate_user_turns"}
    patch: dict = {}
    for k, v in body.items():
        if k not in allowed_keys:
            continue
        if k == "experiment_id":
            if v is not None and not isinstance(v, str):
                raise HTTPException(status_code=400, detail="experiment_id must be string or null")
            patch[k] = v
        elif k in ("inner_voice", "inner_voice_evaluate_user_turns"):
            if not isinstance(v, bool):
                raise HTTPException(status_code=400, detail=f"{k} must be bool")
            patch[k] = v

    if not patch:
        return JSONResponse({"session_key": session_id, "patched": {}, "noop": True})

    # Use mutate_session for atomic read-modify-write.
    from app.sessions_io import mutate_session
    def _apply(data):
        for k, v in patch.items():
            data[k] = v
    ok = await mutate_session(session_id, _apply)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")

    # Re-read to return the canonical state.
    data = json.loads(meta_path.read_text())
    return JSONResponse({
        "session_key": session_id,
        "patched": patch,
        "experiment_id": data.get("experiment_id"),
        "inner_voice": bool(data.get("inner_voice", False)),
        "inner_voice_evaluate_user_turns": bool(
            data.get("inner_voice_evaluate_user_turns", False)
        ),
    })


# ---------------------------------------------------------------------------
# Stub session creation — Inner Voice tab needs to pre-create sessions with
# `inner_voice: true` BEFORE the first turn so the critic fires on turn 1.
# Regular Chat sessions are still created lazily via post_message_stream;
# this endpoint is for callers that need flags set ahead of time.
# ---------------------------------------------------------------------------


@router.post("/api/sessions/create")
async def create_session(request: Request):
    """Pre-create a stub session JSON with optional Inner Voice flags.

    Body (all optional):
      {
        "model": "primary" | "haiku" | ...           — default model alias
        "platform": "mission-control" | "autonomy" | "inner_voice"
        "inner_voice": true,                          — opt into the critic
        "inner_voice_evaluate_user_turns": true,      — fire on chat turns
        "experiment_id": "stage5-bench-001"           — A/B tag
      }

    Returns:
      {
        "session_key": "20260501_213044_iva3b1",
        "session_id":  "20260501_213044_iva3b1",
        "model":       "primary",
        "inner_voice": true,
        "inner_voice_evaluate_user_turns": true,
        "experiment_id": null,
      }

    The session JSON file is created with empty messages list; the next
    `post_message_stream` call will use the existing session_id and
    append turns to it.
    """
    try:
        body = await request.json() if request.headers.get("content-length") else {}
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}

    # Generate the session_id in the same shape the chat path uses:
    # YYYYMMDD_HHMMSS_<6 hex>. The "iv" prefix on the suffix makes Inner
    # Voice sessions visually distinguishable in `ls sessions/`.
    import datetime as _dt
    import secrets
    ts = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    suffix = "iv" + secrets.token_hex(2)
    session_id = f"{ts}_{suffix}"
    meta_path = SESSIONS_DIR / f"{session_id}.json"

    # Validate optional fields.
    model = body.get("model")
    if model is not None and not isinstance(model, str):
        raise HTTPException(status_code=400, detail="model must be a string")
    platform = body.get("platform") or "mission-control"
    if not isinstance(platform, str):
        raise HTTPException(status_code=400, detail="platform must be a string")

    iv = bool(body.get("inner_voice", False))
    iv_user = bool(body.get("inner_voice_evaluate_user_turns", False))
    if iv_user and not iv:
        # User-turn evaluation requires the master flag too — surface a
        # clear error rather than silently dropping.
        raise HTTPException(
            status_code=400,
            detail=(
                "inner_voice_evaluate_user_turns requires inner_voice=true; "
                "set both or neither"
            ),
        )

    experiment_id = body.get("experiment_id")
    if experiment_id is not None and not isinstance(experiment_id, str):
        raise HTTPException(status_code=400, detail="experiment_id must be string or null")

    # Build the stub. Schema mirrors the lazy-create path in messages.py
    # (session_id, model, created_at, last_active, preview, message_count,
    # messages, platform) plus the optional Inner Voice flags.
    now_iso = _dt.datetime.utcnow().isoformat() + "Z"
    stub = {
        "session_id": session_id,
        "model": model or "",
        "created_at": now_iso,
        "last_active": now_iso,
        "preview": "",
        "message_count": 0,
        "messages": [],
        "platform": platform,
    }
    if iv:
        stub["inner_voice"] = True
    if iv_user:
        stub["inner_voice_evaluate_user_turns"] = True
    if experiment_id is not None:
        stub["experiment_id"] = experiment_id

    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    if meta_path.exists():
        # Astronomically unlikely with the timestamp+token shape, but
        # surface rather than overwrite if it ever fires.
        raise HTTPException(status_code=409, detail="session_id collision; retry")
    meta_path.write_text(json.dumps(stub, indent=2))

    return JSONResponse({
        "session_key": session_id,
        "session_id": session_id,
        "model": stub["model"],
        "platform": stub["platform"],
        "inner_voice": iv,
        "inner_voice_evaluate_user_turns": iv_user,
        "experiment_id": experiment_id,
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
async def cancel_session(session_id: str, request: Request):
    """Request cancellation of the currently-running turn.

    Query params:
        drain_pending: if "true" (or "1"), also clear queued ambient
            turns. User turns are never silently dropped by this call.
    """
    cancel_event = get_cancel_event(session_id)
    drain = request.query_params.get("drain_pending", "").lower() in ("true", "1", "yes")

    drained = 0
    if drain:
        drained = await drain_pending(session_id, source="ambient")

    if cancel_event is None:
        payload = {"cancelled": False, "drained": drained}
        if drained == 0:
            payload["detail"] = "Session is not streaming"
        return JSONResponse(payload)

    cancel_event.set()
    return JSONResponse({"cancelled": True, "drained": drained})


@router.get("/api/sessions/{session_id}/queue")
async def get_session_queue(session_id: str):
    """Return a snapshot of the session's turn queue."""
    return JSONResponse(get_queue_state(session_id))


@router.post("/api/sessions/{session_id}/inject")
async def inject_ambient_turn(session_id: str, request: Request):
    """Enqueue an ambient (background-producer) turn on this session.

    Body: {
      "text": "...",                    # required
      "dedup_key": "..." (optional),
      "priority": "notable"|"urgent",   # default "notable"
      "source": "autonomy:task-42",     # for prompt envelope + logging
      "summary": "..."                  # short label, optional
    }

    Response: { "turn_id": "...", "source": "ambient", "preempted": false,
                "dropped": [...], "deduped": false, "queue": {...} }

    Producers (autonomy, `session_inject_context` MCP tool) call this to
    hand the agent context that should be processed as a real turn.
    `notable` = "mention if worth it", `urgent` = "surface now". Both fire
    a full SDK call — use `/inject-prefetch` for cheap passive signals.

    When `dedup_key` is supplied, any queued ambient with the same key
    is dropped (newest wins) — safe for producers that may re-fire the
    same context while the previous version is still queued.
    """
    data = await request.json()
    text = (data.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    dedup_key = (data.get("dedup_key") or "").strip() or None
    priority = (data.get("priority") or "notable").strip().lower()
    if priority not in ("notable", "urgent"):
        priority = "notable"
    producer_source = (data.get("source") or "producer").strip()
    summary = (data.get("summary") or "").strip()

    # Imported here to avoid an import cycle at module load time.
    from app.routers.messages import build_ambient_turn, enqueue_ambient

    turn = await build_ambient_turn(
        session_id, text,
        dedup_key=dedup_key, priority=priority,
        source=producer_source, summary=summary,
    )
    result = await enqueue_ambient(session_id, turn)
    result["queue"] = get_queue_state(session_id)
    result["priority"] = priority
    return JSONResponse(result)


@router.post("/api/sessions/{session_id}/inject-prefetch")
async def inject_ambient_prefetch(session_id: str, request: Request):
    """Push an ambient signal into the session's prefetch queue (no SDK turn).

    Mechanism 1 for #295: drained by `prefetch_context()` on the user's
    NEXT turn and appended to the <context> block. Cheap and passive —
    zero SDK cost, zero transcript noise until the user naturally engages.

    Body: {
      "source": "autonomy:task-42",      # required
      "summary": "3 new emails",          # required — one-liner
      "content": "...",                   # optional fuller body
      "dedup_key": "autonomy-42",         # default = source
      "ttl_seconds": 3600                  # default 1h
    }

    Response: { "queued": 1, "queue_depth": N, "dropped": [...], "deduped": bool }

    If the target session doesn't exist (e.g. producer called with a
    stale session_id), returns 404. Producers should consult
    `GET /api/sessions/active` first or pass `session_id=""` to let the
    server resolve to the most recent user session.
    """
    import time as _time

    meta_path = SESSIONS_DIR / f"{session_id}.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    data = await request.json()
    source = (data.get("source") or "").strip()
    summary = (data.get("summary") or "").strip()
    if not source:
        raise HTTPException(status_code=400, detail="source is required")
    if not summary:
        raise HTTPException(status_code=400, detail="summary is required")

    content = (data.get("content") or "").strip()
    dedup_key = (data.get("dedup_key") or source).strip()
    ttl_seconds = int(data.get("ttl_seconds") or 3600)
    now = _time.time()

    entry = AmbientPrefetchEntry(
        source=source,
        summary=summary,
        content=content,
        dedup_key=dedup_key,
        expires_at=now + ttl_seconds if ttl_seconds > 0 else 0.0,
        enqueued_at=now,
    )
    result = enqueue_ambient_prefetch(session_id, entry)
    return JSONResponse(result)


@router.get("/api/sessions/active")
async def get_active_session_endpoint():
    """Return the current "active" session ID for ambient producers.

    Resolution order (see `app.sessions_io.get_active_session_id`):
      1. Last session that received a user turn in-memory
      2. Most-recent `platform: mission-control` session by mtime (≤24h)
      3. None

    Explicitly excludes `platform: autonomy` sessions so an autonomy task
    never gets its own session back. Returns `{"session_id": null}` if
    no user session qualifies — producers should treat null as "user
    has no active session, skip injection."
    """
    sid = get_active_session_id()
    return JSONResponse({"session_id": sid})


@router.get("/api/sessions/{session_id}/prefetch-queue")
async def get_prefetch_queue(session_id: str):
    """Debug endpoint — peek at a session's ambient prefetch queue without draining."""
    entries = peek_ambient_prefetch(session_id)
    return JSONResponse({
        "depth": len(entries),
        "entries": [
            {
                "source": e.source,
                "summary": e.summary,
                "dedup_key": e.dedup_key,
                "expires_at": e.expires_at,
                "enqueued_at": e.enqueued_at,
            }
            for e in entries
        ],
    })


@router.post("/api/sessions/{session_id}/ambient-decide")
async def post_ambient_decide(session_id: str, request: Request):
    """Record an ambient-turn decision. Called by the `ambient_decide` MCP tool.

    Only valid when the session's current turn is `source: ambient`.
    `surface=false` also cancels the running turn so the agent stops
    generating further output — the server's `_run_turn` finally clause
    handles persistence rewriting (muted breadcrumb vs normal response).

    Body: {
      "surface": bool,         # required
      "message": str = "",     # if surface=true, the agent's intended reply
                               # (informational — the agent's actual output
                               # is what lands in the transcript)
      "reasoning": str = ""    # why silent; shown in the breadcrumb
    }
    """
    data = await request.json()
    current = get_current_turn(session_id)
    if current is None or current.source != "ambient":
        raise HTTPException(
            status_code=400,
            detail="ambient_decide is only valid during an ambient turn",
        )

    surface = bool(data.get("surface"))
    message = (data.get("message") or "").strip()
    reasoning = (data.get("reasoning") or "").strip()

    set_ambient_decision(session_id, {
        "surface": surface,
        "message": message,
        "reasoning": reasoning,
    })

    # Silent decision → cancel the running turn so the agent doesn't waste
    # further output tokens. The finally clause in `_run_turn` consults
    # the decision state and writes the correct breadcrumb either way.
    if not surface:
        ev = get_cancel_event(session_id)
        if ev is not None:
            ev.set()

    return JSONResponse({
        "decision_recorded": True,
        "surface": surface,
        "turn_id": current.turn_id,
    })


@router.post("/api/sessions/clear")
async def clear_session(request: Request):
    data = await request.json()
    session_id = data.get("session_id", "")
    if session_id:
        meta_path = SESSIONS_DIR / f"{session_id}.json"
        if meta_path.exists():
            meta_path.unlink()
    return JSONResponse({"success": True})
