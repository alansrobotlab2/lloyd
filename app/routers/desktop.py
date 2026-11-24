"""Desktop tab — what Lloyd sees of the desktop, and the lease that lets him act.

The Browser tab's pattern (``app/routers/browser.py``) for the real desktop:
the aggregator's ``desktop_capture`` / ``desktop_act`` POST a frame after each
capture, and every open Desktop tab streams it. The tab also carries the one
control nothing else may touch — the lease (``app/desktop_lease.py``).

POST /api/desktop/state  — the aggregator publishes one frame. The publisher is
                           the loopback peer only: a peer on
                           ``server.trusted_networks`` must not be able to show
                           this tab a frame nobody captured.
GET  /api/desktop/state  — SSE: current frame + lease, then each push.
GET  /api/desktop/frame  — the latest frame as JSON, while it is still fresh.

A retained frame is Alan's whole screen — the JPEG plus the element names on
it. The access boundary in front of these reads is the peer address
``server.ApiPeerGate`` decides on: loopback, or a network in
``server.trusted_networks``. No read path here carries a device identity, and
that is a ruling rather than an omission — #683 decided "no per-device token
for now" on 2026-09-27, by which point mutual TLS had already been dropped
from the Vite dev server on 2026-06-14 — so there is no stricter gate these
reads are waiting on. What bounds the exposure is what is built: the
loopback-only publisher above, and every read of ``_latest`` going through
:func:`_frame_is_fresh` — past ``desktop.frame_ttl_seconds`` (default 120 s)
the frame answers as if nobody had captured, here and in
``latest_frame_summary``, which is what ``mc_navigate(tab="desktop")`` puts in
the model's context (#1418). A tab that is already open keeps the last frame it
was pushed until it reloads; that is the decided display behaviour, not a
render gap.
GET  /api/desktop/lease  — who holds the seat, on the same peer-address terms as
                           the reads above.
POST /api/desktop/lease  — ``{op: grant, minutes}`` / ``{op: revoke}``. Human
                           only: ``safety.check_bash_command`` and the
                           aggregator refuse any tool call that names this
                           route or the lease file.
                           A grant is accepted from the seat only:
                           ``_peer_is_loopback`` must hold of the peer, the same
                           requirement ``post_desktop_state`` puts on the frame
                           publish, because lending the keyboard and mouse is a
                           decision made where the person is sitting (#4533).
                           Being admitted by ``ApiPeerGate`` is not being asked:
                           every network in ``server.trusted_networks`` passes
                           that rule, which on this box means the whole tailnet,
                           and until #4533 that was enough to lend oneself
                           Alan's desktop for ``MAX_TTL_MINUTES``. The Vite dev
                           server runs on this machine, so the Desktop tab
                           arrives through its ``xfwd`` rewrite as a loopback
                           peer and still grants; a tab on a tailnet device
                           arrives as its own address and is refused, which is
                           the intent, not a regression. ``op: revoke`` stays
                           open to any peer the gate trusts, so the kill switch
                           keeps working from a phone across the house. A grant
                           also ignores ``session_id`` in the body: a caller
                           this route cannot identify must not get to choose
                           which session the lease it requested covers.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import math
import subprocess
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app import desktop_lease

logger = logging.getLogger("lloyd-server")

router = APIRouter()

_latest: dict | None = None
_subscribers: set[asyncio.Queue] = set()

# How long one retained capture stays readable. A capture belongs to the turn
# that took it, so the default is on the order of a turn; a
# ``desktop.frame_ttl_seconds`` in CONFIG overrides it, and an unusable
# override falls back to this rather than to serving the screen forever.
DEFAULT_FRAME_TTL_SECONDS = 120.0


def _frame_ttl_seconds() -> float:
    from app.config import CONFIG

    section = CONFIG.get("desktop") or {}
    try:
        ttl = float(section.get("frame_ttl_seconds", DEFAULT_FRAME_TTL_SECONDS))
    except (TypeError, ValueError):
        return DEFAULT_FRAME_TTL_SECONDS
    return ttl if ttl > 0 else DEFAULT_FRAME_TTL_SECONDS


def _frame_is_fresh(frame: dict | None, *, now: float | None = None) -> bool:
    """Would this frame still be served as the desktop's current state?

    False for a frame carrying no ``ts`` or an unparseable one: the check fails
    closed, so a missing stamp is never evidence that the screen is current.
    """
    if not isinstance(frame, dict):
        return False
    try:
        ts = float(frame.get("ts"))
    except (TypeError, ValueError):
        return False
    if not math.isfinite(ts):
        return False
    return ((time.time() if now is None else now) - ts) <= _frame_ttl_seconds()


def _peer_is_loopback(request: Request) -> bool:
    """Did this request arrive on this machine's own loopback interface?

    Two writes ask it, both because a peer the network admits is not a person at
    the keyboard. The one legitimate publisher of a frame is the aggregator,
    which posts to ``service_url("backend")`` — measured
    ``http://127.0.0.1:8080`` — and anything else is a peer that never took the
    capture. The one legitimate grantor of a desktop lease is a human sitting at
    this box, which is why ``post_desktop_lease`` started asking the same
    question (#4533). An absent or unparseable client is refused too.
    """
    host = (request.client.host if request.client else "") or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


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
async def post_desktop_state(frame: dict, request: Request):
    global _latest
    if not _peer_is_loopback(request):
        # Refused before anything is stored or published: a peer that never
        # took the capture must not be able to put a frame on the Desktop tab
        # or push one down every open stream (#1418).
        logger.warning("desktop: refused a frame from peer %s",
                       request.client.host if request.client else None)
        raise HTTPException(status_code=403,
                            detail="the desktop frame is published by the local "
                                   "aggregator only")
    _latest = frame
    _publish("state", frame)
    return JSONResponse({"ok": True, "subscribers": len(_subscribers)})


@router.get("/api/desktop/frame")
async def get_desktop_frame():
    if not _frame_is_fresh(_latest):
        # Stale reads answer exactly as "nobody has captured since the backend
        # started": no image, no element names, no summary. The retained frame
        # is the screen, and this route's only gate is the peer address.
        return JSONResponse({"active": False, "lease": _lease_view()})
    return JSONResponse({"active": True, **_latest, "lease": _lease_view()})


@router.get("/api/desktop/lease")
async def get_desktop_lease():
    return JSONResponse(_lease_view())


@router.post("/api/desktop/lease")
async def post_desktop_lease(body: dict, request: Request):
    op = (body or {}).get("op")
    if op == "grant":
        if not _peer_is_loopback(request):
            # Refused before the write, the toast and the publish. ApiPeerGate
            # already let this peer in — every network in
            # server.trusted_networks, the whole tailnet by default — and its
            # address rule is not Alan's consent: a stranger on that network
            # could lend itself his keyboard and mouse for MAX_TTL_MINUTES
            # without him, which is what #4533 closed. The Desktop tab on this
            # machine still arrives on loopback, so the seat is unaffected.
            logger.warning("desktop: refused a lease grant from peer %s",
                           request.client.host if request.client else None)
            raise HTTPException(status_code=403,
                                detail="the desktop lease can only be granted "
                                       "from this machine")
        try:
            minutes = float((body or {}).get("minutes") or 30)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="minutes must be a number")
        # session_id is deliberately not read from the body: a pin names which
        # session the lease covers, and a caller this route cannot identify must
        # not choose it. desktop_lease.grant keeps the parameter for in-process
        # grants, which sit behind the tool layer's owner check.
        desktop_lease.grant(minutes, by="mission-control")
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
    """What the Desktop tab shows, small enough for a prompt — never the image.

    Expires on the same clock as the frame route: this goes into the model's
    context through ``mc_navigate``, which is not gated the way ``desktop_*``
    is, so honouring the TTL in one read path and not the other would leave the
    last window title readable after the frame itself is gone (#1418).
    """
    lease = _lease_view()
    if not _frame_is_fresh(_latest):
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
