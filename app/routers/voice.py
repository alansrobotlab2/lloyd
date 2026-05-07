"""Voice API — narrow surface for the LiveKit RTC pipeline.

Two endpoints:
  POST /api/voice/inject     — accept a transcript (from the LiveKit
                               agent worker) and enqueue it as a real
                               user turn on the named chat session.
  POST /api/livekit/token    — mint a room-scoped JWT for browser
                               clients connecting to the self-hosted
                               LiveKit server.

Phase 8b retired the legacy voice_mode daemon path (the :8092 HTTP
proxy, _VOICE_ACTIVE_SESSION override, _TTS_ENABLED flag, speak_text /
speak_voice_summary helpers, /api/voice/{status,toggle,say,active-
session,tts-status,tts-toggle,config}). The LiveKit worker now calls
/api/voice/inject directly with explicit session_key, and TTS
playback is published as a LiveKit audio track instead of POSTed to
the daemon.
"""

import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import (
    CONFIG,
    _get_model_env,
    _model_base_url,
    _resolve_model_name,
)
from app.harness import RunOptions
from app.paths import SESSIONS_DIR
from app.sessions_io import (
    SessionTurn,
    _save_session_meta,
    enqueue_turn,
    set_last_user_session,
)
from app.mcp_discovery import _get_mcp_servers, _get_disallowed_tools, _get_tool_search_kwargs
from prompt_builder import build_system_prompt
from prefetch import prefetch_context


router = APIRouter()
logger = logging.getLogger("lloyd-server")


# ── /api/voice/inject ─────────────────────────────────────────────────────

@router.post("/api/voice/inject")
async def voice_inject(request: Request):
    """Accept a transcript from the LiveKit agent worker and enqueue it as a
    real user turn on the named chat session.

    Routing: the transcript becomes `source="user"` in the session queue, so
    it streams through `_run_turn` the same way a typed message does —
    persists the user message to the session JSON, broadcasts SSE events to
    any open chat UI, and produces a normal harness reply.

    Payload: {text: str, session_key?: str, speaker?: str}
    Response: {success: bool, session_id: str, turn_id: str}

    The worker derives session_key from the LiveKit room name (room name
    is `lloyd-${session_id}`), so the payload session_key is always
    explicit and authoritative. Anything missing falls back to the legacy
    "voice-main" catch-all session.
    """
    import json

    data = await request.json()
    text = (data.get("text") or "").strip()
    speaker = (data.get("speaker") or "").strip()
    payload_session = data.get("session_key") or data.get("session_id")
    session_id = payload_session or "voice-main"

    if not text:
        raise HTTPException(status_code=400, detail="Message text required")

    prompt_text = (
        f"[{speaker}]: {text}"
        if speaker and speaker.lower() not in ("", "unknown")
        else text
    )

    model = ""
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    existing: dict = {}
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text())
            model = existing.get("model", "") or ""
        except Exception:
            pass

    if not model:
        model = CONFIG.get("model", {}).get("default", "")
    model = _resolve_model_name(model)
    model_env = _get_model_env(model)

    voice_plan = existing.get("plan") or {}
    voice_plan_mode = bool(voice_plan.get("plan_mode"))
    system_prompt = build_system_prompt(
        todos=existing.get("todos") or [], plan=voice_plan,
    )

    _voice_session_id = session_id

    def _voice_refresh_disallowed() -> list[str]:
        from app.paths import SESSIONS_DIR as _SD
        try:
            d = json.loads((_SD / f"{_voice_session_id}.json").read_text())
            live = bool((d.get("plan") or {}).get("plan_mode"))
        except Exception:
            live = False
        return _get_disallowed_tools(plan_mode=live)
    prefetched_text = prefetch_context(prompt_text, session_id=session_id)

    options = RunOptions(
        model=model,
        base_url=model_env.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:8096"),
        system_prompt=system_prompt,
        max_turns=CONFIG.get("agent", {}).get("max_turns", 60),
        permission_mode=CONFIG.get("agent", {}).get(
            "permission_mode", "bypassPermissions"
        ),
        mcp_servers=_get_mcp_servers(),
        disallowed_tools=_get_disallowed_tools(plan_mode=voice_plan_mode),
        disallowed_tools_refresh=_voice_refresh_disallowed,
        env=model_env,
        session_id=session_id,
        priority=0,
        **_get_tool_search_kwargs(),
    )

    await _save_session_meta(session_id, model, preview=prompt_text)

    turn = SessionTurn(
        turn_id=uuid.uuid4().hex[:12],
        source="user",
        payload={
            "text": prompt_text,
            "prefetched_text": prefetched_text,
            "model": model,
            "options": options,
            "meta_path": meta_path,
        },
        enqueued_at=datetime.now(),
    )

    # Lazy import to avoid the messages<->voice import cycle at load time.
    from app.routers.messages import _session_consumer

    try:
        await enqueue_turn(
            session_id,
            turn,
            consumer_factory=lambda: _session_consumer(session_id),
        )
    except Exception as e:
        logger.error(f"Voice inject enqueue failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    set_last_user_session(session_id)

    return JSONResponse(
        {"success": True, "session_id": session_id, "turn_id": turn.turn_id}
    )


# ── /api/voice/summarize ──────────────────────────────────────────────────

@router.post("/api/voice/summarize")
async def voice_summarize(request: Request):
    """Rewrite a primary-model response into a short spoken summary via the
    secondary model. The LiveKit agent worker calls this before TTS so what
    Lloyd says aloud is a tight conversational summary, not the full primary
    response (which is often long, contains code blocks, tool-call references,
    etc. — text that doesn't TTS gracefully).

    Body: {"text": str}
    Response: {"summary": str | null, "used_summary": bool}

    `used_summary` is false (and `summary` echoes the input) when the secondary
    call fails or returns empty — caller should still TTS the text but knows
    it's the raw primary, not the spoken rewrite. Caller does not need to
    branch on this; the convenience is just for logging/telemetry.
    """
    data = await request.json() if (await request.body()) else {}
    text = (data.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    # _sync_secondary_voice_summary uses urllib (sync). Run in executor so we
    # don't block the FastAPI event loop on the secondary's response time
    # (typically 0.5–2s).
    import asyncio as _asyncio
    from app.secondary_models import _sync_secondary_voice_summary

    loop = _asyncio.get_running_loop()
    summary = await loop.run_in_executor(None, _sync_secondary_voice_summary, text)
    if summary:
        return JSONResponse({"summary": summary, "used_summary": True})
    return JSONResponse({"summary": text, "used_summary": False})


# ── /api/livekit/token ────────────────────────────────────────────────────

@router.post("/api/livekit/token")
async def livekit_mint_token(request: Request):
    """Mint a LiveKit JWT scoped to a single room.

    Request body:
      {"session_id": "<id>"}     # session/room key — caller's chat session id
      {"identity": "<id>"}       # optional; defaults to a fresh uuid

    Response:
      {"url": "ws://…", "token": "<jwt>", "room": "<room name>", "identity": "<id>"}
    """
    lk_cfg = (CONFIG.get("livekit") or {}) if isinstance(CONFIG, dict) else {}
    if not lk_cfg or not lk_cfg.get("api_key") or not lk_cfg.get("api_secret"):
        raise HTTPException(status_code=503, detail="LiveKit not configured (config.yaml: livekit.api_key/api_secret missing)")

    body = await request.json() if (await request.body()) else {}
    session_id = (body.get("session_id") or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    identity = (body.get("identity") or f"user-{uuid.uuid4().hex[:8]}").strip()

    room_prefix = lk_cfg.get("room_prefix", "lloyd-")
    room_name = f"{room_prefix}{session_id}"

    # Late import — avoid pulling livekit-api into module load if the feature
    # is unused. AccessToken's API is sync.
    try:
        from livekit import api as lkapi
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"livekit-api not installed: {e}")

    grants = lkapi.VideoGrants(
        room=room_name,
        room_join=True,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
    )
    token = (
        lkapi.AccessToken(lk_cfg["api_key"], lk_cfg["api_secret"])
        .with_identity(identity)
        .with_name(identity)
        .with_grants(grants)
        .to_jwt()
    )
    return JSONResponse({
        "url": lk_cfg.get("url", "ws://127.0.0.1:7880"),
        "token": token,
        "room": room_name,
        "identity": identity,
    })


# ── /api/voice/speakers ───────────────────────────────────────────────────

def _get_speaker_identifier():
    """Lazily build a SpeakerIdentifier sharing config + profiles_dir with
    the LiveKit worker. The class lives in agent-services/ so we add that
    to sys.path on first call."""
    import os
    import sys
    agent_services = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "agent-services",
    )
    if agent_services not in sys.path:
        sys.path.insert(0, agent_services)

    from speaker_id import SpeakerIdentifier  # noqa: E402

    vp_cfg = ((CONFIG.get("livekit") or {}).get("voiceprint") or {})
    if not vp_cfg.get("enabled", True):
        raise HTTPException(
            status_code=503,
            detail="voiceprint matching is disabled in config (livekit.voiceprint.enabled)",
        )
    return SpeakerIdentifier(
        profiles_dir=vp_cfg.get("profiles_dir", "~/lloyd/voice_profiles"),
        threshold=float(vp_cfg.get("profile_threshold", 0.75)),
        unknown_label=str(vp_cfg.get("unknown_label", "Unknown")),
        device=str(vp_cfg.get("device", "cpu")),
    )


@router.get("/api/voice/speakers")
async def voice_speakers_list():
    """List enrolled voice profiles."""
    try:
        sid = _get_speaker_identifier()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"speaker module unavailable: {e}")
    return JSONResponse({"profiles": sid.list_profiles()})


@router.post("/api/voice/speakers/enroll")
async def voice_speakers_enroll(request: Request):
    """Enroll a voice profile from a wav blob.

    Multipart form fields:
      name: str    — profile name (alphanumeric/-/_)
      audio: file  — wav file (any sample rate; resemblyzer resamples)

    Embeds the audio with resemblyzer and saves <profiles_dir>/<name>.npy.
    """
    form = await request.form()
    name = (form.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    audio_file = form.get("audio")
    if audio_file is None:
        raise HTTPException(status_code=400, detail="audio file is required")
    audio_bytes = await audio_file.read()

    import io
    import wave
    import numpy as np
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as w:
            sample_rate = w.getframerate()
            num_channels = w.getnchannels()
            sampwidth = w.getsampwidth()
            n_frames = w.getnframes()
            raw = w.readframes(n_frames)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"could not parse wav: {e}")
    if sampwidth != 2:
        raise HTTPException(status_code=400, detail=f"need 16-bit PCM, got {sampwidth*8}-bit")
    samples = np.frombuffer(raw, dtype=np.int16)
    if num_channels > 1:
        samples = samples.reshape(-1, num_channels).mean(axis=1).astype(np.int16)
    duration_s = len(samples) / max(1, sample_rate)
    if duration_s < 1.0:
        raise HTTPException(status_code=400, detail=f"audio too short ({duration_s:.1f}s); need >= 1s")

    sid = _get_speaker_identifier()
    import asyncio as _asyncio
    loop = _asyncio.get_running_loop()
    try:
        path = await loop.run_in_executor(None, sid.enroll, name, samples, sample_rate)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"enroll failed: {e}")
    return JSONResponse({
        "name": name,
        "path": path,
        "duration_s": round(duration_s, 2),
        "sample_rate": sample_rate,
    })


@router.delete("/api/voice/speakers/{name}")
async def voice_speakers_delete(name: str):
    """Delete a voice profile by name."""
    sid = _get_speaker_identifier()
    if not sid.delete_profile(name):
        raise HTTPException(status_code=404, detail=f"profile not found: {name}")
    return JSONResponse({"deleted": name})
