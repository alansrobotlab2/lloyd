"""Voice Mode endpoints — proxy to voice mode daemon on port 8092, plus
full turn execution on transcript injection.
"""

import asyncio
import json
import logging

import httpx
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ResultMessage, TextBlock
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import (
    CONFIG,
    _get_model_env,
    _resolve_effort,
    _resolve_thinking,
    _model_base_url,
    _resolve_model_name,
)
from app.paths import SESSIONS_DIR
from app.sessions_io import _save_session_meta
from app.mcp_discovery import _get_mcp_servers, _get_disallowed_tools
from prompt_builder import build_system_prompt
from prefetch import prefetch_context


router = APIRouter()
logger = logging.getLogger("lloyd-server")

VOICE_API = "http://127.0.0.1:8092"

# Tracks the MC-focused session so voice transcripts route to whatever session
# the user currently has open. Set by the frontend via /api/voice/active-session.
# None = fall back to voice_bridge_config.json's lloyd_session_key (voice-main).
_VOICE_ACTIVE_SESSION: str | None = None


@router.post("/api/voice/active-session")
async def voice_set_active_session(request: Request):
    """Set the session that voice transcripts should route to.

    Payload: {session_id: str | null}  — null clears the override.
    """
    global _VOICE_ACTIVE_SESSION
    try:
        data = await request.json()
    except Exception:
        data = {}
    sid = data.get("session_id")
    _VOICE_ACTIVE_SESSION = sid if isinstance(sid, str) and sid else None
    return JSONResponse({"active_session": _VOICE_ACTIVE_SESSION})


@router.get("/api/voice/active-session")
async def voice_get_active_session():
    return JSONResponse({"active_session": _VOICE_ACTIVE_SESSION})


@router.get("/api/voice/status")
async def voice_status():
    """Proxy to voice mode status."""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{VOICE_API}/v1/status", timeout=5.0)
            return JSONResponse(r.json())
    except Exception:
        return JSONResponse({"state": "OFFLINE", "voice_enabled": False, "last_transcript": ""})


@router.post("/api/voice/toggle")
async def voice_toggle():
    """Proxy to voice mode toggle."""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(f"{VOICE_API}/v1/voice/toggle", json={}, timeout=5.0)
            return JSONResponse(r.json())
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Voice mode not available: {e}")


@router.post("/api/voice/say")
async def voice_say(request: Request):
    """Proxy to voice mode TTS."""
    try:
        body = await request.json()
        async with httpx.AsyncClient() as client:
            r = await client.post(f"{VOICE_API}/v1/say", json=body, timeout=60.0)
            return JSONResponse(r.json())
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Voice mode not available: {e}")


@router.post("/api/voice/inject")
async def voice_inject(request: Request):
    """Accept a transcript from voice_mode.py, run a turn, speak the response.

    Payload: {text, speaker?, session_key?, speak?=true}
    """
    data = await request.json()
    text = (data.get("text") or "").strip()
    speaker = (data.get("speaker") or "").strip()
    # Priority:
    #   1. explicit session_key/session_id in the inject payload
    #   2. the MC-focused session set by the frontend
    #   3. the legacy "voice-main" catch-all session
    session_id = (
        data.get("session_key")
        or data.get("session_id")
        or _VOICE_ACTIVE_SESSION
        or "voice-main"
    )
    speak = bool(data.get("speak", True))

    if not text:
        raise HTTPException(status_code=400, detail="Message text required")

    prompt_text = (
        f"[{speaker}]: {text}"
        if speaker and speaker.lower() not in ("", "unknown")
        else text
    )

    model = CONFIG.get("model", {}).get("default", "")
    model = _resolve_model_name(model)
    model_env = _get_model_env(model)

    meta_path = SESSIONS_DIR / f"{session_id}.json"
    resume_id = None
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text())
            session_model = _resolve_model_name(existing.get("model", ""))
            if _model_base_url(session_model) == _model_base_url(model):
                resume_id = existing.get("sdk_session_id")
        except Exception:
            pass

    system_prompt = build_system_prompt()
    prefetched_text = prefetch_context(prompt_text, session_id=session_id)

    options = ClaudeAgentOptions(
        model=model,
        system_prompt=system_prompt,
        max_turns=CONFIG.get("agent", {}).get("max_turns", 60),
        permission_mode=CONFIG.get("agent", {}).get(
            "permission_mode", "bypassPermissions"
        ),
        mcp_servers=_get_mcp_servers(),
        disallowed_tools=_get_disallowed_tools(),
        env=model_env,
        effort=_resolve_effort(model),
        thinking=_resolve_thinking(model),
        resume=resume_id,
    )

    await _save_session_meta(session_id, model, preview=prompt_text)

    full_response = ""
    try:
        async for message in query(prompt=prefetched_text, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        full_response += block.text
            elif isinstance(message, ResultMessage):
                if hasattr(message, "result") and message.result:
                    full_response = message.result
    except Exception as e:
        logger.error(f"Voice inject error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if speak and full_response.strip():
        async def _speak(text_to_say: str):
            try:
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"{VOICE_API}/v1/say",
                        json={"text": text_to_say},
                        timeout=180.0,
                    )
            except Exception as e:
                logger.warning(f"Voice inject TTS failed: {e}")

        asyncio.create_task(_speak(full_response))

    return JSONResponse(
        {"success": True, "response": full_response, "session_id": session_id}
    )


@router.get("/api/voice/config")
async def voice_config():
    """Proxy to voice mode config."""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{VOICE_API}/v1/config", timeout=5.0)
            return JSONResponse(r.json())
    except Exception:
        return JSONResponse({"error": "Voice mode not available"})


@router.post("/api/voice/config")
async def voice_set_config(request: Request):
    """Proxy to voice mode config update."""
    try:
        body = await request.json()
        async with httpx.AsyncClient() as client:
            r = await client.post(f"{VOICE_API}/v1/config", json=body, timeout=5.0)
            return JSONResponse(r.json())
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Voice mode not available: {e}")
