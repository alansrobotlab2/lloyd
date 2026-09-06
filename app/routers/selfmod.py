"""Self-modification control surface: the drain flag and status.

The **drain flag** closes the one race the idle gate cannot: a user turn can
arrive between the promoter's last quiet poll and its `stopProcess` call.
Losing that turn costs the user their message — `shutdown_cleanup` cancels
in-flight turns with a 2s grace — and under auto-landing that would happen
regularly. While drain is set, `POST /api/message/stream` refuses new turns
with a 503 telling the caller to retry.

Two properties make the flag safe rather than a new way to wedge the backend:

  * **The TTL is mandatory and absolute.** A promoter that dies after setting
    the flag cannot leave the backend refusing turns — the lease expires on
    its own. This is the whole reason to prefer a TTL over a persistent mode.
  * **It is in-memory**, so the landing restart clears it for free. The
    promoter only needs to clear it explicitly on its abort path.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()
logger = logging.getLogger("lloyd-selfmod")

# Absolute epoch after which the drain lapses. 0 = not draining.
_drain_until: float = 0.0
_DRAIN_MAX_TTL = 600.0


def drain_active() -> bool:
    return time.time() < _drain_until


def drain_remaining() -> float:
    return max(0.0, _drain_until - time.time())


def set_drain(on: bool, ttl_s: float = 180.0) -> float:
    global _drain_until
    if not on:
        _drain_until = 0.0
    else:
        _drain_until = time.time() + max(1.0, min(float(ttl_s), _DRAIN_MAX_TTL))
    return _drain_until


@router.post("/api/selfmod/drain")
async def post_drain(request: Request):
    data = await request.json()
    on = bool(data.get("on", True))
    ttl = float(data.get("ttl_s", 180.0))
    set_drain(on, ttl)
    logger.info("selfmod drain %s (ttl=%.0fs)", "ON" if on else "OFF", drain_remaining())
    return JSONResponse({"draining": drain_active(), "remaining_s": round(drain_remaining(), 1)})


@router.get("/api/selfmod/drain")
async def get_drain():
    return JSONResponse({"draining": drain_active(), "remaining_s": round(drain_remaining(), 1)})


@router.get("/api/selfmod/status")
async def get_status(limit: int = 25):
    """State + ledger tail, for the Mission Control banner."""
    try:
        from scripts.selfmod import state as S
    except Exception as exc:
        return JSONResponse({"available": False, "error": str(exc)[:200]}, status_code=200)

    events = S.read_events(limit=limit)
    acked = {e.get("commit") for e in events if e.get("event") == "ack"}
    unacked = [
        e for e in events
        if e.get("event") in ("rollback_succeeded", "rollback_failed", "quarantined")
        and e.get("commit") not in acked
    ]

    guardian = None
    try:
        import json as _json
        from pathlib import Path
        hb = Path.home() / ".local" / "state" / "lloyd-guardian" / "heartbeat.json"
        raw = _json.loads(hb.read_text(encoding="utf-8"))
        raw["age_s"] = round(time.time() - float(raw.get("ts", 0)), 1)
        raw["stale"] = raw["age_s"] > 60
        guardian = raw
    except Exception:
        guardian = None

    return JSONResponse({
        "available": True,
        "last_known_good": S.read_lkg(),
        "current": S.read_current(),
        "halted": S.is_halted(),
        "broken": S.is_broken(),
        "pause_remaining_s": round(S.pause_remaining(), 1),
        "draining": drain_active(),
        "guardian": guardian,
        "unacknowledged": unacked,
        "events": events,
    })


@router.post("/api/selfmod/ack")
async def post_ack(request: Request):
    data = await request.json()
    commit = str(data.get("commit") or "")
    from scripts.selfmod import state as S
    S.append_event({"event": "ack", "commit": commit, "by": "user"})
    return JSONResponse({"acked": commit})
