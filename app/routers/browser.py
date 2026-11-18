"""Browser tab — SSE mirror of the agent's live browser session (#278).

The agent drives a headless Chromium through the browser_* MCP tools. This
router is what makes that session visible in Mission Control without a second
window: the MCP server POSTs a frame after each tool call, and every open
Browser tab streams it.

POST /api/browser/state — MCP server publishes one frame (screenshot + annotated a11y snapshot).
GET  /api/browser/state — SSE: the frame that was current when you subscribed, then each push.
GET  /api/browser/frame — the same frame as plain JSON, for a client that doesn't want a stream.
POST /api/browser/navigate — the tab's URL bar, proxied to the aggregator (see below).
POST /api/browser/input — the tab's pointer and keyboard, proxied the same way.

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

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.aggregator_config import auth_headers_for
from app.aggregator_config import route as aggregator_route

logger = logging.getLogger("lloyd-server")

router = APIRouter()

_latest: dict | None = None
_subscribers: set[asyncio.Queue] = set()

# Playwright runs in the lloyd-mcp process, so a control action has to cross
# that seam. Derived from `services.lloyd_mcp` through the aggregator route
# registry (`app.aggregator_config`), which is also where the request credential
# of #1053 comes from: the URL and the header travel together, so a proxy cannot
# be updated to send one and forget the other.
_MCP_NAVIGATE_URL = aggregator_route("browser_navigate")
_MCP_INPUT_URL = aggregator_route("browser_input")

# A navigation is a page load (up to 30s in the tool) plus a screenshot and
# an aria snapshot, and the viewer is watching a spinner for all of it.
_NAVIGATE_TIMEOUT_S = 45.0
# An input event is a click or a few keystrokes; the aggregator answers once it
# is delivered and settles the frame afterwards. Short, because the tab queues
# events behind each other and one stuck request would hold every keystroke.
_INPUT_TIMEOUT_S = 15.0


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
    live = bool(frame.get("live"))
    # A live frame is pixels only (the aggregator's screencast, ~8 a second).
    # What is KEPT is the merge, so a late subscriber, `/frame` and the prompt
    # summary still have the tree, the title and who drove it; what is FANNED
    # OUT is the light frame, and the tab does the same merge.
    _latest = {**_latest, **frame} if live and _latest is not None else frame
    dead: list[asyncio.Queue] = []
    for q in list(_subscribers):
        try:
            q.put_nowait(frame)
        except asyncio.QueueFull:
            if live:
                # A slow viewer skips pixels; it is not dropped for them. The
                # full frames it has queued stay queued.
                continue
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


def latest_frame_summary() -> dict:
    """What the browser is showing, small enough to put in a prompt.

    `mc_navigate(tab="browser")` reads this. It must never carry
    `screenshot_b64` or `snapshot` — the frame is ~20 KB of base64 plus up
    to 8 KB of accessibility tree, and this goes into the model's context as
    a one-line "here is what you just put on their screen".
    """
    if _latest is None:
        return {"active": False}
    return {
        "active": True,
        "url": _latest.get("url"),
        "title": _latest.get("title"),
        "ts": _latest.get("ts"),
        "driven_by": _latest.get("tool"),
    }


@router.post("/api/browser/navigate")
async def post_browser_navigate(body: dict):
    """The Browser tab's URL bar.

    Deliberately not an MCP call. The user typing a URL is not the agent
    using a tool, and dispatching it as one would write a `browser_navigate`
    into the transcript that the model never made.
    """
    url = body.get("url") if isinstance(body, dict) else None
    if not isinstance(url, str) or not url.strip():
        raise HTTPException(status_code=400, detail="url is required")

    try:
        async with httpx.AsyncClient(timeout=_NAVIGATE_TIMEOUT_S) as client:
            r = await client.post(_MCP_NAVIGATE_URL, json={"url": url.strip()},
                                  headers=auth_headers_for(_MCP_NAVIGATE_URL))
    except Exception as e:
        # The aggregator being down is the single most likely failure here
        # and the viewer needs to be told which half is broken.
        logger.warning("browser/navigate: aggregator unreachable: %s", e)
        return JSONResponse({"error": f"browser service unreachable: {e}"})

    try:
        data = r.json()
    except Exception:
        data = {"error": r.text or f"navigate failed (HTTP {r.status_code})"}
    if r.status_code >= 400 and "error" not in data:
        data = {"error": f"navigate failed (HTTP {r.status_code})"}
    return JSONResponse(data)


@router.post("/api/browser/input")
async def post_browser_input(body: dict):
    """The Browser tab's pointer and keyboard: one click, text, key or wheel.

    Proxied for the same reason the URL bar is, and with one more rule: the
    body of a `type` event can be a password, so nothing here logs the body or
    an error derived from it. The aggregator validates the event; this only
    checks it is an object.
    """
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="body must be an object")
    try:
        async with httpx.AsyncClient(timeout=_INPUT_TIMEOUT_S) as client:
            r = await client.post(_MCP_INPUT_URL, json=body,
                                  headers=auth_headers_for(_MCP_INPUT_URL))
    except Exception as e:
        logger.warning("browser/input: aggregator unreachable: %s", type(e).__name__)
        return JSONResponse({"error": f"browser service unreachable ({type(e).__name__})"})
    try:
        data = r.json()
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    if r.status_code >= 400 or not data:
        data = {"error": data.get("error") or f"input failed (HTTP {r.status_code})"}
    return JSONResponse(data)


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
