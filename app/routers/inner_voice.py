"""Inner Voice (#345) REST endpoints.

Stage 0 shipped the event-log endpoint live and the others as
stable-shape stubs. Stage 2 wires Brain 2, so `/critiques` now serves
real rows and `/state` reports observation state from session metadata
+ recent critiques.

Stage 5 adds:
  * `/api/inner_voice/grading_summary` — addressed_rate per persona,
    graded_rate over time, false-positive proxy. Used by the meta-review
    notebook AND by the frontend Inner Voice tab to show "are our
    interventions actually working?"
  * `/state` returns ``stage='5'`` plus a `grading_progress` block
    summarizing recent coverage so the UI pill can render it.
"""

import json
import logging
from datetime import datetime, timedelta
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
    grading_cfg = iv_cfg.get("grading") or {}

    # Stage 5: per-session grading progress. Counts interventions that
    # are still ungraded (`outcome_turn_id IS NULL`) and the count graded
    # since the session started. The session-scoped query is cheap
    # because `inner_voice_interventions` is small per session (max
    # `max_nudges_per_session + 1`-ish rows in the steady state).
    grading_progress: dict[str, Any] = {
        "enabled": bool(grading_cfg.get("enabled", True)),
        "graded": 0,
        "ungraded": 0,
        "addressed_true": 0,
        "addressed_false": 0,
        "addressed_null": 0,
    }
    if session_id:
        try:
            iv_rows = usage_store.list_inner_voice_interventions(
                session_id=session_id, limit=500,
            )
            for r in iv_rows:
                if r.get("outcome_turn_id") is None:
                    grading_progress["ungraded"] += 1
                else:
                    grading_progress["graded"] += 1
                    addr = r.get("outcome_addressed")
                    if addr is True or addr == 1:
                        grading_progress["addressed_true"] += 1
                    elif addr is False or addr == 0:
                        grading_progress["addressed_false"] += 1
                    else:
                        grading_progress["addressed_null"] += 1
        except Exception:
            pass

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
        "grading_progress": grading_progress,        # Stage 5
        "stage": "5",
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


# ---------------------------------------------------------------------------
# Stage 5: grading-pass aggregate metrics
# ---------------------------------------------------------------------------


@router.get("/grading_summary")
async def grading_summary(
    session_id: str | None = Query(default=None),
    hours: float = Query(default=24, ge=0.1, le=24 * 90),
) -> dict[str, Any]:
    """Return Stage 5 grading-pass coverage + outcome distribution.

    Window: ``hours`` (default 24, max 90 days).

    Output:
        {
          "session_id": <str|null>,
          "window_hours": <float>,
          "since_iso": <ISO>,
          "total_interventions":  <int>,        # in window
          "graded":               <int>,        # outcome_addressed not NULL OR graded_at not NULL
          "graded_rate":          <float>,      # graded / total (0.0 if total==0)
          "addressed_true":       <int>,
          "addressed_false":      <int>,
          "addressed_null":       <int>,        # ambiguous (graded but verdict == null)
          "addressed_rate":       <float>,      # true / (true+false), 0.0 if denom 0
          "by_persona": {
            "<persona>": {
                "total":           <int>,
                "graded":          <int>,
                "addressed_true":  <int>,
                "addressed_false": <int>,
                "addressed_null":  <int>,
                "addressed_rate":  <float>,
            }, ...
          },
        }

    The ``by_persona`` block groups interventions by the persona that
    fired the triggering critique. Interventions without a known
    triggering persona (escalations from hard_max_turns, or rows missing
    the FK) bucket under ``"(unknown)"``.
    """
    try:
        since_dt = datetime.utcnow() - timedelta(hours=hours)
        since_iso = since_dt.strftime("%Y-%m-%dT%H:%M:%S")

        # Pull a generous window. The schema has no FK to persona, so we
        # join via triggered_by_critique_id → inner_voice_critiques.persona
        # in Python (saves a SQL JOIN that would also need parametric
        # session/window filtering twice).
        ivs = usage_store.list_inner_voice_interventions(
            session_id=session_id, limit=2000,
        )
        crits = usage_store.list_inner_voice_critiques(
            session_id=session_id, limit=5000,
        )
        crit_persona_by_id: dict[int, str] = {}
        for c in crits:
            cid = c.get("id")
            if cid is not None:
                crit_persona_by_id[int(cid)] = (c.get("persona") or "(unknown)")

        # Window filter on created_at (string ISO comparison works for
        # the YYYY-MM-DDTHH:MM:SS shape the schema uses).
        in_window = [r for r in ivs if (r.get("created_at") or "") >= since_iso]

        total = len(in_window)
        graded = 0
        addressed_true = 0
        addressed_false = 0
        addressed_null = 0
        by_persona: dict[str, dict[str, int]] = {}

        def _bucket(p: str) -> dict[str, int]:
            return by_persona.setdefault(
                p,
                {
                    "total": 0,
                    "graded": 0,
                    "addressed_true": 0,
                    "addressed_false": 0,
                    "addressed_null": 0,
                },
            )

        for r in in_window:
            persona = "(unknown)"
            tcid = r.get("triggered_by_critique_id")
            if tcid is not None:
                persona = crit_persona_by_id.get(int(tcid), "(unknown)")
            bucket = _bucket(persona)
            bucket["total"] += 1

            outcome_set = (
                r.get("outcome_turn_id") is not None
                or r.get("graded_at") is not None
            )
            if outcome_set:
                graded += 1
                bucket["graded"] += 1
                addr = r.get("outcome_addressed")
                if addr is True or addr == 1:
                    addressed_true += 1
                    bucket["addressed_true"] += 1
                elif addr is False or addr == 0:
                    addressed_false += 1
                    bucket["addressed_false"] += 1
                else:
                    addressed_null += 1
                    bucket["addressed_null"] += 1

        # Compute rates
        graded_rate = (graded / total) if total > 0 else 0.0
        denom = addressed_true + addressed_false
        addressed_rate = (addressed_true / denom) if denom > 0 else 0.0

        for p, b in by_persona.items():
            denom_p = b["addressed_true"] + b["addressed_false"]
            b["addressed_rate"] = (
                (b["addressed_true"] / denom_p) if denom_p > 0 else 0.0
            )

        return {
            "session_id": session_id,
            "window_hours": hours,
            "since_iso": since_iso,
            "total_interventions": total,
            "graded": graded,
            "graded_rate": graded_rate,
            "addressed_true": addressed_true,
            "addressed_false": addressed_false,
            "addressed_null": addressed_null,
            "addressed_rate": addressed_rate,
            "by_persona": by_persona,
        }
    except Exception as e:
        logger.warning(f"grading_summary failed: {e}")
        return {
            "error": str(e),
            "session_id": session_id,
            "window_hours": hours,
        }
