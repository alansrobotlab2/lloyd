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
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import (
    CONFIG,
    _get_model_env,
    _model_base_url,
    _resolve_model_name,
)
from app.harness import HookRegistry, RunOptions, install_default_safety_hook
from app.paths import SESSIONS_DIR, VOICE_PROFILES_DIR
from app.sessions_io import (
    SessionTurn,
    _save_session_meta,
    enqueue_turn,
    set_last_user_session,
)
from app.mcp_discovery import _get_mcp_servers, _get_disallowed_tools, _get_harness_kwargs
from prompt_builder import build_system_prompt
from prefetch import prefetch_context_async


router = APIRouter()
logger = logging.getLogger("lloyd-server")


#: Prepended to what the model sees for a spoken turn — never to the text the
#: chat shows, which stays exactly what was said. Rides in the prompt's tail
#: the way the 20-turn memory nudge does, so the cached prefix is untouched,
#: and it is recorded as that turn's subliminal context like every other
#: injection.
#:
#: It exists because the reply is now spoken AS IT STREAMS. The old path
#: waited for the whole answer and had the secondary model rewrite it for
#: speech (0.5-2 s, on a single-tenant engine that every agent turn also
#: queues behind); that rewrite cannot run on text that has not been written
#: yet, so the instruction moves to the one place that can act on it before
#: the first word — the model writing the answer.
VOICE_TURN_REMINDER = (
    "<system-reminder>This message was spoken aloud, and your reply is read "
    "out by text-to-speech as you write it. Answer in plain spoken sentences: "
    "no markdown, bullets, headings, tables or code blocks, and never read out "
    "a URL, a file path or a hash. Lead with the answer and keep it to a few "
    "sentences unless asked for more. If the full answer needs code, a table "
    "or a link, put it in your reply after a line containing only `---` and "
    "say that it is in the chat; nothing after that line is spoken. Write "
    "numbers, times and units the way a person says them. Before you start "
    "work that needs tools, say in one short sentence what you are about to "
    "do, and if the work runs to several tool calls, add a brief spoken "
    "update now and then — the listener hears nothing while tools run."
    "</system-reminder>\n\n"
)

#: The typed-turn counterpart, for a session whose voice room is open: the
#: worker speaks the reply of a typed turn too (see `app/voice_tap.py`), but
#: the person typed, so the chat answer keeps its formatting. What changes is
#: the order — a short spoken answer first, detail after the `---` line the
#: clause splitter stops at.
VOICE_ROOM_REMINDER = (
    "<system-reminder>A voice room is open on this chat, so the start of your "
    "reply is also read aloud as you write it. Open with a short answer in "
    "plain spoken sentences; if the full answer needs formatting, code, a "
    "table or a link, put it after a line containing only `---` — nothing "
    "after that line is spoken.</system-reminder>\n\n"
)


def voice_room_prefix(session_id: str) -> str:
    """What a typed turn on `session_id` carries in its prompt tail because a
    voice room is listening: the interruption note, if the listener cut the
    last spoken reply off, and the typed-turn reminder. Empty when no room is
    open — the common case, and then nothing about the turn changes."""
    from app import voice_tap

    if not voice_tap.has_listener(session_id):
        return ""
    out = voice_tap.take_heard_note(session_id) or ""
    if _voice_turn_cfg().get("typed_turns", "stream") == "stream":
        out += VOICE_ROOM_REMINDER
    return out


def _voice_turn_cfg() -> dict:
    return ((CONFIG.get("livekit") or {}).get("voice_turn") or {})


def _voice_extra_body() -> dict:
    """Request-body extras for a spoken turn.

    `thinking: off` sends `enable_thinking: false`, which the Qwen template
    applies to the generation prompt only — the rendered history, and so the
    cached prefix, is unchanged. Everything before the first spoken word is
    dead air, and a reasoning phase is the largest thing that can sit there.
    """
    if str(_voice_turn_cfg().get("thinking", "on")).lower() in ("off", "false", "0"):
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return {}


# ── /api/voice/inject ─────────────────────────────────────────────────────

@router.post("/api/voice/inject")
async def voice_inject(request: Request):
    """Accept a transcript from the LiveKit agent worker and enqueue it as a
    real user turn on the named chat session.

    Routing: the transcript becomes `source="user"` in the session queue, so
    it streams through `_run_turn` the same way a typed message does —
    persists the user message to the session JSON, broadcasts SSE events to
    any open chat UI, and produces a normal harness reply.

    Payload: {text: str, session_key?: str, speaker?: str, stream?: bool}
    Response: {success: bool, session_id: str, turn_id: str}
              — or, with `stream: true`, the turn's own event stream as SSE
              (the same frames `/api/message/stream` sends), headed by a
              `voice_turn` frame carrying the turn id.

    `stream` is what lets the worker speak a reply while it is being written.
    A turn's events queue has exactly one reader, and until this a spoken
    turn had none: the events accumulated unread and the worker found the
    answer by re-fetching the whole transcript every 500 ms, then waited for
    it to finish. The caller that enqueued the turn is the natural owner of
    that reader.

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
    stream = bool(data.get("stream"))

    prompt_text = (
        f"[{speaker}]: {text}"
        if speaker and speaker.lower() not in ("", "unknown")
        else text
    )

    setup = _voice_turn_setup(session_id)
    model, meta_path, options = setup["model"], setup["meta_path"], setup["options"]
    prefetched_text = await prefetch_context_async(
        prompt_text, session_id=session_id, plan_mode=setup["plan_mode"],
    )
    if _voice_turn_cfg().get("reminder", True):
        prefetched_text = VOICE_TURN_REMINDER + prefetched_text
    from app import voice_tap
    heard = voice_tap.take_heard_note(session_id)
    if heard:
        prefetched_text = heard + prefetched_text

    from app.prompt_layout import append_turn_tail
    prefetched_text = append_turn_tail(prefetched_text, setup.get("turn_tail", ""))

    # Is another turn already running (or queued) on this session? The queue
    # is strictly serial, so this turn will wait — minutes, when a typed turn
    # is mid-investigation — and the worker should say so instead of a filler
    # followed by silence (the 2026-09-24 19:17 "Yeah." waited 44 s).
    from app.sessions_io import get_queue_state
    qs = get_queue_state(session_id)
    queued_behind = bool(qs.get("current")) or bool(qs.get("pending_user"))

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

    voice_tap.mark_voice_turn(turn.turn_id)

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

    if stream:
        from app.routers.messages import _turn_sse_generator

        async def _sse():
            head = {"session_id": session_id, "turn_id": turn.turn_id,
                    "queued_behind": queued_behind}
            yield f"event: voice_turn\ndata: {json.dumps(head)}\n\n"
            async for chunk in _turn_sse_generator(turn):
                yield chunk

        return StreamingResponse(_sse(), media_type="text/event-stream")

    return JSONResponse(
        {"success": True, "session_id": session_id, "turn_id": turn.turn_id}
    )


def _voice_turn_setup(session_id: str) -> dict:
    """Model, system prompt and RunOptions for a spoken turn on `session_id`.

    One definition for the turn and for its prewarm, because the prewarm is
    worth something only if it sends the engine the byte-identical prefix the
    turn will send a moment later — two private copies of "how a voice turn is
    set up" would drift, and the first sign would be a prewarm that silently
    warms a prefix nobody uses.
    """
    import json

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
    # P1: the frozen memory snapshot and the state/delta tail, as the chat
    # path builds them. Both no-ops by default.
    from app import memory_snapshot, prompt_layout

    frozen_mem, memory_note = memory_snapshot.frozen_memories(session_id)
    system_prompt = build_system_prompt(
        session_id=session_id,
        todos=existing.get("todos") or [], plan=voice_plan,
        **prompt_layout.mem_kwargs(frozen_mem),
    )
    turn_tail = prompt_layout.turn_tail(
        existing.get("todos") or [], voice_plan, None, memory_note)

    def _voice_refresh_disallowed() -> list[str]:
        from app.paths import SESSIONS_DIR as _SD
        try:
            d = json.loads((_SD / f"{session_id}.json").read_text())
            live = bool((d.get("plan") or {}).get("plan_mode"))
        except Exception:
            live = False
        return _get_disallowed_tools(plan_mode=live)

    # #1136. A spoken turn has the same tool surface a chat turn has — the same
    # MCP servers, hence the same tier-2 senders — and until this line it had no
    # PreToolUse hook of any kind: the options were built with no `hooks`
    # argument, `RunOptions.hooks` defaults to None, and the SessionTurn ran them
    # verbatim. The cross-file finder added in this same round is what saw it; a
    # per-file grep asks "does this file install hooks", and a file that installs
    # nothing is never asked. The registry is built here so the turn and its
    # prewarm (`_prewarm` reuses this dict) arm identically.
    turn_hooks = HookRegistry()
    install_default_safety_hook(turn_hooks)

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
        extra_body=_voice_extra_body(),
        hooks=turn_hooks,
        **_get_harness_kwargs(),
    )
    return {"model": model, "meta_path": meta_path, "options": options,
            "plan_mode": voice_plan_mode, "turn_tail": turn_tail}


# ── /api/voice/listen ─────────────────────────────────────────────────────

@router.get("/api/voice/listen")
async def voice_listen(session_key: str, request: Request):
    """Every turn's events on a session, for its voice room (`app/voice_tap.py`).

    The worker opens this once per room and speaks the replies of turns it did
    not inject itself — typed turns — through the same clause path as spoken
    ones. It is also how the worker learns that a typed message arrived, which
    opens a conversation (a person typing to Lloyd with the room open is
    talking to him). An SSE comment every 15 s keeps idle proxies from closing
    the stream.
    """
    import asyncio
    import json

    from app import voice_tap

    session_id = (session_key or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_key is required")
    q = voice_tap.open_tap(session_id)

    async def _sse():
        try:
            yield ": voice tap open\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    evt = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield f"event: {evt['event']}\ndata: {json.dumps(evt['data'])}\n\n"
        finally:
            voice_tap.close_tap(session_id, q)

    return StreamingResponse(_sse(), media_type="text/event-stream")


# ── /api/voice/interrupted ────────────────────────────────────────────────

@router.post("/api/voice/interrupted")
async def voice_interrupted(request: Request):
    """The listener talked over a reply; record how much of it they heard.

    Body: {session_key, heard: str, unheard_chars?: int}. `heard` is the text
    of the clauses whose playout completed. The next turn on the session —
    spoken or typed — carries it as a note in its prompt tail, which is
    LiveKit Agents' "truncate the chat context at the playout position" in
    Lloyd's terms: the transcript keeps the whole reply, and the model is told
    which part of it was actually said.
    """
    from app import voice_tap

    data = await request.json() if (await request.body()) else {}
    session_id = (data.get("session_key") or data.get("session_id") or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_key is required")
    heard = str(data.get("heard") or "")
    unheard = int(data.get("unheard_chars") or 0)
    voice_tap.record_interruption(session_id, heard, unheard)
    logger.info("voice interrupted on %s: heard %d chars, %d unheard",
                session_id, len(heard), unheard)
    return JSONResponse({"recorded": True})


# ── /api/voice/prewarm ────────────────────────────────────────────────────

#: session_id -> monotonic time of the last prewarm, so a burst of wake fires
#: (or a wake per sentence inside one window) costs one prefill, not several.
_PREWARM_LAST: dict[str, float] = {}
_PREWARM_DEBOUNCE_S = 20.0


@router.post("/api/voice/prewarm")
async def voice_prewarm(request: Request):
    """Prefill a session's prompt while the user is still talking.

    The worker calls this the moment the wake word fires — mid-sentence,
    measured at ~3 s before the finished utterance is injected. A spoken
    turn's prefix is the session's system prompt, tools and history, and after
    a break it is usually not in the engine's cache any more: the worker pool
    runs 100-200k-token turns around the clock. Measured through a real room
    on 2026-09-17: first answer token 4.16 s after inject on a cold session,
    0.45 s on the warm turn after it. This spends that prefill during the
    user's own sentence.

    Body: {session_key}. Answers at once; the prefill runs in the background.
    Fire-and-forget by design — a prewarm that fails costs nothing but the
    head start.
    """
    import asyncio
    import time

    data = await request.json() if (await request.body()) else {}
    session_id = (data.get("session_key") or data.get("session_id") or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_key is required")
    if not _voice_turn_cfg().get("prewarm", True):
        return JSONResponse({"started": False, "reason": "disabled"})
    now = time.monotonic()
    if now - _PREWARM_LAST.get(session_id, 0.0) < _PREWARM_DEBOUNCE_S:
        return JSONResponse({"started": False, "reason": "recent"})
    _PREWARM_LAST[session_id] = now
    asyncio.get_running_loop().create_task(_prewarm(session_id))
    return JSONResponse({"started": True})


async def _prewarm(session_id: str) -> None:
    """One completion of one token over the prefix the next spoken turn sends.

    Built with the turn's own setup and history loader, so the prefix matches
    — up to the user message, which is the only thing that differs and sits
    last. A session over its compaction threshold is skipped: its real turn
    will summarize, which changes the history, and a prewarm must not spend a
    summarization call on a wake that might have been a false alarm.
    """
    import time

    from app.compaction import load_and_compact_session
    from app.harness import run_query
    from app.routers._messages_harness_adapter import _prepare_messages_for_harness

    t0 = time.monotonic()
    try:
        setup = _voice_turn_setup(session_id)
        options, meta_path = setup["options"], setup["meta_path"]
        history: list[dict] = []
        if meta_path.exists():
            comp = await load_and_compact_session(
                meta_path, model=setup["model"], mode_override="truncate")
            if comp.get("truncated"):
                logger.info("voice prewarm %s: skipped (over the compaction "
                            "threshold — the turn will summarize)", session_id)
                return
            history = await _prepare_messages_for_harness(comp["history"])
            if history and history[-1].get("role") == "user":
                history = history[:-1]
        options.extra_body = {**options.extra_body, "max_tokens": 1}
        options.max_turns = 1
        async for _ in run_query(history + [{"role": "user", "content": "."}], options):
            pass
        logger.info("voice prewarm %s: prefilled in %.2fs (%d history messages)",
                    session_id, time.monotonic() - t0, len(history))
    except Exception as e:
        logger.warning("voice prewarm %s failed after %.2fs: %s",
                       session_id, time.monotonic() - t0, e)


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

    from app.secondary_models import (
        _is_trivially_speakable,
        _sync_secondary_voice_summary,
    )

    # Fast path: short, plain-prose replies don't benefit from a rewrite —
    # the secondary's own prompt would just echo them back. Skip the call.
    if _is_trivially_speakable(text):
        logger.info("voice_summarize: skipped secondary (%d chars, plain prose)", len(text))
        return JSONResponse({"summary": text, "used_summary": False})

    # _sync_secondary_voice_summary uses urllib (sync). Run in executor so we
    # don't block the FastAPI event loop on the secondary's response time
    # (typically 0.5–2s).
    import asyncio as _asyncio

    loop = _asyncio.get_running_loop()
    summary = await loop.run_in_executor(None, _sync_secondary_voice_summary, text)
    if summary:
        return JSONResponse({"summary": summary, "used_summary": True})
    return JSONResponse({"summary": text, "used_summary": False})


# ── /api/voice/ww_miss ────────────────────────────────────────────────────

@router.post("/api/voice/ww_miss")
async def voice_ww_miss(request: Request):
    """Flag a wake-word miss for the diagnostic capture rig.

    Forwards to the LiveKit worker's localhost endpoint
    (default http://127.0.0.1:8501/ww_miss). The worker dumps its
    per-room rolling raw-audio ring + recent score records to
    `<data root>/ww_diag/misses/<ts>_<label>.{wav,json}` — the corpus
    `app.ww_diag` resolves, which on this box is
    `~/lloyd-data/ww_diag/misses/…`. Use this when you
    just said "Hey Lloyd" and Lloyd didn't respond — capturing the audio
    that the wake word didn't fire on is the only way to debug it.

    Body (all optional): {label?, room?, identity?}
      label   — short tag written into the filename (e.g. "laptop_kitchen")
      room    — explicit room name; otherwise the most-active room is used
      identity— for record-keeping only

    Curl from anywhere on this host:
      curl -X POST http://localhost:8080/api/voice/ww_miss \\
           -H 'content-type: application/json' \\
           -d '{"label":"laptop_kitchen"}'
    """
    body = await request.json() if (await request.body()) else {}
    diag_cfg = (
        ((CONFIG.get("livekit") or {}).get("acoustic_wake") or {}).get("diag")
    ) or {}
    host = str(diag_cfg.get("host", "127.0.0.1"))
    port = int(diag_cfg.get("port", 8501))
    target = f"http://{host}:{port}/ww_miss"

    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(target, json=body)
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"LiveKit worker not reachable at {target} — is lloyd-agent-worker running?",
        )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"worker proxy failed: {e}")
    if r.status_code >= 400:
        return JSONResponse(r.json() if r.content else {"ok": False}, status_code=r.status_code)
    return JSONResponse(r.json())


# ── /api/voice/ww_label ───────────────────────────────────────────────────

@router.post("/api/voice/ww_label")
async def voice_ww_label(request: Request):
    """Attach a ground-truth label to an existing wake-word capture.

    Use this after listening to `<data root>/ww_diag/utterances/<id>.wav` or
    `<data root>/ww_diag/misses/<ts>_<label>.wav` — the corpus
    `app.ww_diag` resolves, `~/lloyd-data/ww_diag/…` on this box — to confirm
    whether the user
    actually said the wake word in that recording. The replay/analysis
    script in Phase 1c uses these labels to compute true detection rate
    vs false-accept rate per device class.

    Body: {said_wake_word?: bool, addressed?: "yes"|"no"|"backchannel",
           utterance_id?: str, miss_ts?: number, note?: str}
      Provide utterance_id for a VAD-segmented capture, OR miss_ts for a
      ring-buffer-only miss dump (the integer part of the filename). At
      least one of the two is required, and at least one of the labels:
      `said_wake_word` grades the wake word, `addressed` grades the
      conversation gate (was it meant for Lloyd, not, or an acknowledgement
      like "yeah" while he spoke).

    Examples:
      curl -X POST http://localhost:8080/api/voice/ww_label \\
           -H 'content-type: application/json' \\
           -d '{"utterance_id":"14c9cfa902f5","said_wake_word":true,
                "note":"hey lloyd, scored 0.25"}'
      curl -X POST http://localhost:8080/api/voice/ww_label \\
           -H 'content-type: application/json' \\
           -d '{"miss_ts":1778379115,"said_wake_word":true}'
    """
    body = await request.json() if (await request.body()) else {}
    diag_cfg = (
        ((CONFIG.get("livekit") or {}).get("acoustic_wake") or {}).get("diag")
    ) or {}
    host = str(diag_cfg.get("host", "127.0.0.1"))
    port = int(diag_cfg.get("port", 8501))
    target = f"http://{host}:{port}/ww_label"

    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(target, json=body)
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"LiveKit worker not reachable at {target} — is lloyd-agent-worker running?",
        )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"worker proxy failed: {e}")
    if r.status_code >= 400:
        return JSONResponse(r.json() if r.content else {"ok": False}, status_code=r.status_code)
    return JSONResponse(r.json())


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
        "url": _livekit_url_for_request(request, lk_cfg),
        "token": token,
        "room": room_name,
        "identity": identity,
    })


def _livekit_url_for_request(request: Request, lk_cfg: dict) -> str:
    """Build the WebSocket URL the browser should use to reach LiveKit.

    Clients connect through the Vite proxy at /livekit, so the URL is derived
    from the public host the browser used (forwarded by Vite as
    X-Forwarded-Host / X-Forwarded-Proto). The agent worker uses the
    ``livekit.url`` internal URL — that one is NOT exposed to browsers. An
    explicit ``livekit.client_url`` in config still wins for advanced setups.
    """
    explicit = (lk_cfg.get("client_url") or "").strip()
    if explicit:
        return explicit

    fwd_host = (request.headers.get("x-forwarded-host") or "").split(",")[0].strip()
    fwd_proto = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip()
    host = fwd_host or request.headers.get("host") or "localhost:5173"
    proto = fwd_proto or request.url.scheme
    scheme = "wss" if proto == "https" else "ws"
    return f"{scheme}://{host}/livekit"


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
        profiles_dir=vp_cfg.get("profiles_dir", str(VOICE_PROFILES_DIR)),
        threshold=float(vp_cfg.get("profile_threshold", 0.40)),
        unknown_label=str(vp_cfg.get("unknown_label", "Unknown")),
        device=str(vp_cfg.get("device", "cpu")),
        backend=str(vp_cfg.get("backend", "campplus")),
        model_path=vp_cfg.get("model_path") or None,
        num_threads=int(vp_cfg.get("num_threads", 2)),
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
      audio: file  — wav file (any sample rate; resampled to 16 kHz)

    Embeds the audio with the configured speaker encoder
    (`livekit.voiceprint.backend`, CAM++ by default) and saves
    <profiles_dir>/<name>.<backend>.npy.
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
