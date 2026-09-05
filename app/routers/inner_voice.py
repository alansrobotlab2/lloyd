"""Inner Voice REST endpoints (thin observer model).

Replaces the old per-stage critique/intervention/grading endpoints with a
single observation timeline. The observer writes one row per significant
event into `inner_voice_observations`; the frontend reads it back via
these endpoints.
"""

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

import usage_store
from app import event_log
from app.paths import SESSIONS_DIR

logger = logging.getLogger("lloyd-server")

router = APIRouter(prefix="/api/inner_voice", tags=["inner_voice"])


# ---------------------------------------------------------------------------
# Observations — replaces critiques + interventions
# ---------------------------------------------------------------------------

@router.get("/observations")
async def list_observations(
    session_id: str | None = Query(default=None),
    turn_id: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict[str, Any]:
    """Observer observations, newest first.

    Each row: one decision the observer made, plus the trigger event, the
    reason and any content (injected text, ambient body, clarify question).

    `action` is one of the five levers (noop | inject | cancel | ambient |
    clarify), `acknowledge_complete`, or a `noop_*` label recording what the
    observer intended before a guard downgraded it. `deny_tool` and `allow`
    appear only in rows written before v4, when Inner Voice could still gate
    tool dispatch.
    """
    try:
        rows = usage_store.list_inner_voice_observations(
            session_id=session_id, turn_id=turn_id, limit=limit,
        )
        return {"observations": rows, "count": len(rows)}
    except Exception as e:
        logger.warning(f"list_observations failed: {e}")
        return {"observations": [], "count": 0, "error": str(e)}


# ---------------------------------------------------------------------------
# State — current observer status for one session
# ---------------------------------------------------------------------------

@router.get("/state")
async def get_state(
    session_id: str | None = Query(default=None),
) -> dict[str, Any]:
    """Current observer state for a session.

    Returns:
      * `inner_voice_enabled` — opt-in flag from session JSON.
      * `evaluate_user_turns` — opt-in for firing on user-typed turns.
      * `observations_count_by_action` — {action: count} across the session.
      * `last_observation_at` — ISO timestamp of the most recent row.
    """
    inner_voice_enabled = False
    evaluate_user_turns = False
    if session_id:
        meta_path = SESSIONS_DIR / f"{session_id}.json"
        if meta_path.exists():
            try:
                data = json.loads(meta_path.read_text())
                inner_voice_enabled = bool(data.get("inner_voice", False))
                evaluate_user_turns = bool(
                    data.get("inner_voice_evaluate_user_turns", False)
                )
            except Exception:
                pass

    counts: dict[str, int] = {}
    last_at: str | None = None
    latest_goal_card: dict[str, Any] | None = None
    latest_user_request: str | None = None
    latest_turn_id: str | None = None
    if session_id and inner_voice_enabled:
        try:
            counts = usage_store.count_inner_voice_observations_by_action(
                session_id=session_id,
            )
            recent = usage_store.list_inner_voice_observations(
                session_id=session_id, limit=1,
            )
            if recent:
                last_at = recent[0].get("created_at")
        except Exception as e:
            logger.warning(f"get_state aggregate failed: {e}")
        # Pull the most recent goal_card extraction from the event log so the
        # UI can render the user's current ask + IV's captured intent. Read in
        # bounded chunks from the tail; goal_card_extracted is rare (1-3 per
        # turn) so the most recent one is usually within the last 200 events.
        try:
            total = event_log.count_events(session_id)
            chunk = 200
            offset = max(0, total - chunk)
            scanned = 0
            while offset >= 0 and scanned < 2000:
                events = event_log.read_events(
                    session_id, offset=offset, limit=chunk,
                )
                # Walk backwards through the chunk so we find the latest first.
                for ev in reversed(events):
                    if ev.get("event") == "inner_voice.goal_card_extracted":
                        d = ev.get("data") or {}
                        latest_goal_card = d.get("goal_card") or {}
                        latest_user_request = d.get("user_request") or None
                        latest_turn_id = ev.get("turn_id")
                        break
                if latest_goal_card is not None or offset == 0:
                    break
                scanned += chunk
                offset = max(0, offset - chunk)
        except Exception as e:
            logger.warning(f"get_state goal_card lookup failed: {e}")

    return {
        "session_id": session_id,
        "inner_voice_enabled": inner_voice_enabled,
        "evaluate_user_turns": evaluate_user_turns,
        "observations_count_by_action": counts,
        "last_observation_at": last_at,
        "latest_goal_card": latest_goal_card,
        "latest_user_request": latest_user_request,
        "latest_turn_id": latest_turn_id,
    }


# ---------------------------------------------------------------------------
# Sessions opted into Inner Voice
# ---------------------------------------------------------------------------

@router.get("/sessions")
async def list_inner_voice_sessions(
    limit: int = Query(default=200, ge=1, le=1000),
) -> dict[str, Any]:
    """List sessions opted into Inner Voice (`inner_voice: true` in JSON).

    Sorted newest first by mtime.
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
            "evaluate_user_turns": bool(
                data.get("inner_voice_evaluate_user_turns", False)
            ),
        })
    return {"sessions": out, "count": len(out)}


# ---------------------------------------------------------------------------
# Event log — kept verbatim, frontend uses it for raw inspection
# ---------------------------------------------------------------------------

@router.get("/event_log")
async def get_event_log(
    session_id: str = Query(...),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=1000),
    expand_blobs: bool = Query(default=False),
) -> dict[str, Any]:
    """Paginated read of a session's event log."""
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
