"""Inner Voice (#345) REST endpoints.

Stage 0 shipped the event-log endpoint live and the others as
stable-shape stubs. Stage 2 wires Brain 2, so `/critiques` now serves
real rows and `/state` reports observation state from session metadata
+ recent critiques.
"""

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

import usage_store
from app import event_log
from app.config import CONFIG
from app.paths import SESSIONS_DIR

logger = logging.getLogger("lloyd-server")

router = APIRouter(prefix="/api/inner_voice", tags=["inner_voice"])


# ---------------------------------------------------------------------------
# Critiques — Stage 2 live (Brain 2 fires + persists from messages.py)
# ---------------------------------------------------------------------------

@router.get("/critiques")
async def list_critiques(
    session_id: str | None = Query(default=None),
    turn_id: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict[str, Any]:
    """List Inner Voice critiques, newest first.

    Filters by session_id and/or turn_id. Empty list if no rows match;
    endpoint shape is stable across all stages so the frontend can poll
    without knowing which stage is live.
    """
    try:
        rows = usage_store.list_inner_voice_critiques(
            session_id=session_id, turn_id=turn_id, limit=limit,
        )
        return {"critiques": rows, "count": len(rows)}
    except Exception as e:
        logger.warning(f"list_critiques failed: {e}")
        return {"critiques": [], "count": 0, "error": str(e)}


# ---------------------------------------------------------------------------
# Interventions — populated in Stage 2+
# ---------------------------------------------------------------------------

@router.get("/interventions")
async def list_interventions(
    session_id: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict[str, Any]:
    """List Inner Voice interventions. Newest first."""
    try:
        rows = usage_store.list_inner_voice_interventions(
            session_id=session_id, limit=limit,
        )
        return {"interventions": rows, "count": len(rows)}
    except Exception as e:
        logger.warning(f"list_interventions failed: {e}")
        return {"interventions": [], "count": 0, "error": str(e)}


# ---------------------------------------------------------------------------
# Observation state — what is Brain 2 doing right now?
# ---------------------------------------------------------------------------

@router.get("/state")
async def get_state(
    session_id: str | None = Query(default=None),
) -> dict[str, Any]:
    """Current Inner Voice state for a session.

    Stage 4 surface:
      * `inner_voice_enabled` — opt-in flag from session JSON.
      * `state` — `idle | observing | critiquing | intervening` (last
        derived from most-recent critique recency).
      * `active_ensemble` — config default; the per-turn ensemble can
        differ (safety_critical / research_writing / code_writing). The
        per-turn name lands in `inner_voice.ensemble_selected` events.
      * `configured_personas` — full set the default ensemble would fire.
      * `nudge_count` — interventions of kind `continue` or `steer`
        (continue = Stage 4 consensus veto; steer = Stage 3 placeholder).
      * `consecutive_vetoes` — current three-strike streak from the
        consensus_termination module's in-process state.
      * `escalations_count` — interventions of kind `escalate`.
      * `max_nudges_per_session` / `veto_severity_threshold` /
        `hard_max_turns` — config values the UI can render alongside.
    """
    iv_cfg = CONFIG.get("inner_voice") or {}
    ensemble_cfg = iv_cfg.get("ensemble") or {}
    default_name = ensemble_cfg.get("default") or "autonomy_default"
    sets = ensemble_cfg.get("sets") or {}
    configured_personas = list(sets.get(default_name) or [])

    inner_voice_enabled = False
    if session_id:
        meta_path = SESSIONS_DIR / f"{session_id}.json"
        if meta_path.exists():
            try:
                data = json.loads(meta_path.read_text())
                inner_voice_enabled = bool(data.get("inner_voice", False))
            except Exception:
                pass

    state = "idle"
    last_critique_at: str | None = None
    if session_id and inner_voice_enabled:
        try:
            recent = usage_store.list_inner_voice_critiques(
                session_id=session_id, limit=1,
            )
            if recent:
                last_critique_at = recent[0].get("created_at")
                state = "observing"
        except Exception:
            pass

    nudges = 0
    escalations = 0
    if session_id:
        try:
            interventions = usage_store.list_inner_voice_interventions(
                session_id=session_id, limit=500,
            )
            for r in interventions:
                k = r.get("kind")
                if k in ("continue", "steer"):
                    nudges += 1
                elif k == "escalate":
                    escalations += 1
        except Exception:
            pass

    consecutive_vetoes = 0
    if session_id:
        try:
            from app.inner_voice import consensus_termination as _ct
            consecutive_vetoes = _ct.get_consecutive_veto_count(session_id)
        except Exception:
            pass

    dis_cfg = iv_cfg.get("disagreement") or {}
    ct_cfg = iv_cfg.get("consensus_termination") or {}
    return {
        "session_id": session_id,
        "inner_voice_enabled": inner_voice_enabled,
        "state": state,
        "active_ensemble": default_name,
        "personas": configured_personas,           # Stage 4: actual fire set
        "configured_personas": configured_personas,
        "nudge_count": nudges,
        "consecutive_vetoes": consecutive_vetoes,
        "escalations_count": escalations,
        "max_nudges_per_session": int(
            (iv_cfg.get("throughput") or {}).get("max_nudges_per_session", 2)
        ),
        "veto_severity_threshold": float(
            dis_cfg.get("veto_severity_threshold", 0.85)
        ),
        "hard_max_turns": int(ct_cfg.get("hard_max_turns", 60)),
        "last_critique_at": last_critique_at,
        "stage": "4",
    }


# ---------------------------------------------------------------------------
# Sessions opted into Inner Voice — Stage 0 returns empty list.
# Real semantics: a session "is Inner Voice" iff its session JSON has
# `inner_voice: true` in metadata. Wire-up lands when the Inner Voice
# tab can actually create sessions (Stage 2).
# ---------------------------------------------------------------------------

@router.get("/sessions")
async def list_inner_voice_sessions(
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict[str, Any]:
    """List sessions opted into Inner Voice (sessions with `inner_voice: true`
    in their JSON metadata). Sorted newest first by mtime. Stage 2 lights
    this up so the Inner Voice tab has its own session list.
    """
    out: list[dict[str, Any]] = []
    if not SESSIONS_DIR.exists():
        return {"sessions": [], "count": 0}
    try:
        files = sorted(
            SESSIONS_DIR.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except Exception as e:
        logger.warning("list_inner_voice_sessions glob failed: %s", e)
        return {"sessions": [], "count": 0, "error": str(e)}

    for path in files:
        if len(out) >= limit:
            break
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not data.get("inner_voice"):
            continue
        out.append({
            "session_id": path.stem,
            "experiment_id": data.get("experiment_id"),
            "title": data.get("title") or data.get("topic") or "",
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "message_count": len(data.get("messages") or []),
        })
    return {"sessions": out, "count": len(out)}


# ---------------------------------------------------------------------------
# Event log — populated from Stage 0 onward (brain1.* on every session).
# This is the only endpoint with real data in Stage 0.
# ---------------------------------------------------------------------------

@router.get("/event_log")
async def get_event_log(
    session_id: str = Query(...),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=1000),
    expand_blobs: bool = Query(default=False),
) -> dict[str, Any]:
    """Paginated read of a session's event log.

    `expand_blobs=true` resolves `{"$blob": "<sha>"}` references back to
    inline strings — useful for the click-to-detail panel on annotation
    cards. Costs extra disk reads, off by default.
    """
    try:
        events = event_log.read_events(
            session_id, offset=offset, limit=limit, expand_blobs=expand_blobs,
        )
        total = event_log.count_events(session_id)
        return {
            "session_id": session_id,
            "events": events,
            "offset": offset,
            "limit": limit,
            "returned": len(events),
            "total": total,
        }
    except Exception as e:
        logger.warning(f"get_event_log failed for {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/event_log/blob/{sha}")
async def get_event_log_blob(sha: str) -> dict[str, Any]:
    """Resolve a single blob hash to its content. Returns 404 on miss."""
    if not sha or len(sha) != 64 or not all(c in "0123456789abcdef" for c in sha):
        raise HTTPException(status_code=400, detail="invalid sha256")
    content = event_log.read_blob(sha)
    if content is None:
        raise HTTPException(status_code=404, detail="blob not found")
    return {"sha": sha, "content": content, "size": len(content)}
