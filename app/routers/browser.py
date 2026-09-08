"""Browser tab — SSE mirror of the agent's live browser session (#278).

The agent drives a headless Chromium through the browser_* MCP tools. This
router is what makes that session visible in Mission Control without a second
window: the MCP server POSTs a frame after each tool call, and every open
Browser tab streams it.

POST /api/browser/state — MCP server publishes one frame (screenshot + annotated a11y snapshot).
GET  /api/browser/state — SSE: the frame that was current when you subscribed, then each push.
GET  /api/browser/frame — the same frame as plain JSON, for a client that doesn't want a stream.

Deliberately NOT on the /api/mc/events bus: that channel fans out to every
page in Mission Control and carries commands, so adding ~20 KB of JPEG to
each of its events would slow every tab down to feed one. The frame is
transient by design — a client that subscribes late gets the last frame it
arrived after, which is the whole point of a live view.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger("lloyd-server")

router = APIRouter()

_latest: dict | None = None
_subscribers: set[asyncio.Queue] = set()


def subscribe() -> asyncio.Queue:
    """Register a frame subscriber; caller must call unsubscribe()."""
    q: asyncio.Queue = asyncio.Queue(maxsize=32)
    _subscribers.add(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    _subscribers.discard(q)


def _publish(frame: dict) -> None:
    """Keep the newest frame and fan it out, dropping dead subscribers.

    The queue is deliberately depth-32 and overwrite-on-full: a viewer on a
    slow link gets the newest frame, not a backlog of stale screenshots.
    """
    global _latest
    _latest = frame
    dead: list[asyncio.Queue] = []
    for q in list(_subscribers):
        try:
            q.put_nowait(frame)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        _subscribers.discard(q)
        logger.debug("browser: dropped stalled SSE subscriber")


@router.post("/api/browser/state")
async def post_browser_state(frame: dict):
    _publish(frame)
    return JSONResponse({"ok": True, "subscribers": len(_subscribers)})


@router.get("/api/browser/frame")
async def get_browser_frame():
    if _latest is None:
        return JSONResponse({"active": False})
    return JSONResponse({"active": True, **_latest})


async def _state_sse(request: Request):
    q = subscribe()
    try:
        # Initial hello so the client knows the channel is live.
        yield f"event: hello\ndata: {json.dumps({'ok': True})}\n\n"
        while True:
            if await request.is_disconnected():
                break
            try:
                frame = await asyncio.wait_for(q.get(), timeout=15.0)
            except asyncio.TimeoutError:
                # Heartbeat keeps the connection from being reaped by
                # intermediaries that close idle SSE sockets.
                yield ": ping\n\n"
                continue
            yield f"event: state\ndata: {json.dumps(frame)}\n\n"
    except asyncio.CancelledError:
        raise
    finally:
        unsubscribe(q)


@router.get("/api/browser/state")
async def get_browser_state(request: Request):
    return StreamingResponse(_state_sse(request), media_type="text/event-stream")
