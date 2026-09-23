"""Desktop tab — what Lloyd sees of the desktop, and the lease that lets him act.

The Browser tab's pattern (``app/routers/browser.py``) for the real desktop:
the aggregator's ``desktop_capture`` / ``desktop_act`` POST a frame after each
capture, and every open Desktop tab streams it. The tab also carries the one
control nothing else may touch — the lease (``app/desktop_lease.py``).

POST /api/desktop/state  — the aggregator publishes one frame.
GET  /api/desktop/state  — SSE: current frame + lease, then each push.
GET  /api/desktop/frame  — the latest frame as JSON.
GET  /api/desktop/lease  — who holds the seat.
POST /api/desktop/lease  — ``{op: grant, minutes}`` / ``{op: revoke}``. Human
                           only: ``safety.check_bash_command`` and the
                           aggregator refuse any tool call that names this
                           route or the lease file.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app import desktop_lease

logger = logging.getLogger("lloyd-server")

router = APIRouter()

_latest: dict | None = None
_subscribers: set[asyncio.Queue] = set()


def _publish(event: str, data: dict) -> None:
    dead = []
    for q in list(_subscribers):
        try:
            q.put_nowait((event, data))
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        _subscribers.discard(q)


def _lease_view() -> dict:
    d = desktop_lease.read()
    return {k: d.get(k) for k in ("holder", "agent_holds", "remaining_s", "epoch",
                                  "reason", "granted_by", "session_id", "expired",
                                  "since", "expires_at")}


def _toast(msg: str) -> None:
    try:
        subprocess.Popen(["hyprctl", "notify", "1", "4000", "rgb(7aa2f7)", msg],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


@router.post("/api/desktop/state")
async def post_desktop_state(frame: dict):
    global _latest
    _latest = frame
    _publish("state", frame)
    return JSONResponse({"ok": True, "subscribers": len(_subscribers)})


@router.get("/api/desktop/frame")
async def get_desktop_frame():
    if _latest is None:
        return JSONResponse({"active": False, "lease": _lease_view()})
    return JSONResponse({"active": True, **_latest, "lease": _lease_view()})


@router.get("/api/desktop/lease")
async def get_desktop_lease():
    return JSONResponse(_lease_view())


@router.post("/api/desktop/lease")
async def post_desktop_lease(body: dict):
    op = (body or {}).get("op")
    if op == "grant":
        try:
            minutes = float((body or {}).get("minutes") or 30)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="minutes must be a number")
        desktop_lease.grant(minutes, by="mission-control",
                            session_id=str((body or {}).get("session_id") or ""))
        _toast(f"Lloyd may use the desktop for {int(minutes)} min — move the mouse to take it back")
    elif op == "revoke":
        desktop_lease.revoke("revoked from Mission Control")
        _toast("Lloyd's desktop lease revoked")
    else:
        raise HTTPException(status_code=400, detail="op must be grant or revoke")
    view = _lease_view()
    _publish("lease", view)
    return JSONResponse(view)


def latest_frame_summary() -> dict:
    """What the Desktop tab shows, small enough for a prompt — never the image."""
    lease = _lease_view()
    if _latest is None:
        return {"active": False, "lease": lease["holder"]}
    w = _latest.get("window") or {}
    return {
        "active": True,
        "window": w.get("title"),
        "class": w.get("class"),
        "ts": _latest.get("ts"),
        "driven_by": _latest.get("tool"),
        "elements": len(_latest.get("elements") or []),
        "lease": lease["holder"],
        "lease_remaining_s": lease.get("remaining_s"),
    }


async def _sse(request: Request):
    q: asyncio.Queue = asyncio.Queue(maxsize=16)
    _subscribers.add(q)
    try:
        yield f"event: lease\ndata: {json.dumps(_lease_view())}\n\n"
        while True:
            if await request.is_disconnected():
                break
            try:
                event, data = await asyncio.wait_for(q.get(), timeout=10.0)
            except asyncio.TimeoutError:
                # The lease runs out and trips without a push; refresh it on
                # the heartbeat so the toggle never shows a grant that is over.
                yield f"event: lease\ndata: {json.dumps(_lease_view())}\n\n"
                continue
            yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
    except asyncio.CancelledError:
        raise
    finally:
        _subscribers.discard(q)


@router.get("/api/desktop/state")
async def get_desktop_state(request: Request):
    return StreamingResponse(_sse(request), media_type="text/event-stream")
