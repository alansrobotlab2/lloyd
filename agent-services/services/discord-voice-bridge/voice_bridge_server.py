#!/usr/bin/env python3
"""
Discord Voice Bridge Server — FastAPI bridge between the Node.js Discord
bot and the Lloyd voice pipeline.

Receives PCM audio from the Node.js bridge, feeds it into the pipeline's
_DiscordAudioReader queue, and queues TTS responses for the Node bridge
to play back into Discord.

Run:
    cd ~/lloyd/agent-services
    .venv/bin/python services/discord-voice-bridge/voice_bridge_server.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

# Add project root to sys.path so we can import voice_pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import requests as http_requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from voice_pipeline import (
    PipelineCallbacks,
    PipelineRunner,
    State,
    load_config,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("discord-bridge")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "voice_bridge_config.json"

_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)

# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------


class DiscordAudioPayload(BaseModel):
    user_id: str
    username: str
    pcm_base64: str
    sample_rate: int = 16000


class TtsRequest(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


class BridgeState:
    """Shared mutable state between FastAPI endpoints and the pipeline thread."""

    def __init__(self):
        self.pipeline: PipelineRunner | None = None
        self.config: dict = {}
        self.state: str = "INITIALIZING"
        self.initialized: bool = False
        self.transcript_history: deque[dict] = deque(maxlen=50)

        # OpenClaw
        self.openclaw_url: str | None = None
        self.openclaw_token: str | None = None
        self.use_openclaw: bool = False

        # TTS response queue — items are {pcm_base64: str, sample_rate: int}
        self.tts_queue: deque[dict] = deque(maxlen=10)
        self.tts_lock = threading.Lock()

    def enqueue_tts(self, pcm_bytes: bytes, sample_rate: int) -> None:
        with self.tts_lock:
            self.tts_queue.append({
                "pcm_base64": base64.b64encode(pcm_bytes).decode("ascii"),
                "sample_rate": sample_rate,
            })

    def dequeue_tts(self) -> dict | None:
        with self.tts_lock:
            if self.tts_queue:
                return self.tts_queue.popleft()
            return None


bridge = BridgeState()

# ---------------------------------------------------------------------------
# Pipeline callbacks
# ---------------------------------------------------------------------------


class DiscordPipelineCallbacks:
    """Pipeline callbacks that inject transcripts to OpenClaw and queue TTS."""

    def on_state_changed(self, state: State) -> None:
        bridge.state = state.name
        log.info("Pipeline state: %s", state.name)

    def on_init_progress(self, component: str) -> None:
        log.info("Loading: %s", component)

    def on_init_complete(self) -> None:
        bridge.initialized = True
        log.info("Pipeline ready — enabling voice")
        if bridge.pipeline:
            bridge.pipeline.voice_enabled.set()
            bridge.pipeline.start()

    def on_transcript(self, text: str, speaker: str, is_continuity: bool) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        prefix = "[+] " if is_continuity else ""
        log.info("Transcript [%s]: %s%s", speaker, prefix, text)
        bridge.transcript_history.append({
            "timestamp": ts,
            "speaker": speaker,
            "transcript": text,
            "is_continuity": is_continuity,
        })
        # Inject to OpenClaw in background
        threading.Thread(
            target=_inject_to_openclaw, args=(text, speaker), daemon=True
        ).start()

    def on_continuity_status(self, msg: str) -> None:
        log.debug("Continuity: %s", msg)

    def on_error(self, error: str) -> None:
        log.error("Pipeline error: %s", error)


# ---------------------------------------------------------------------------
# OpenClaw injection + TTS
# ---------------------------------------------------------------------------


def _inject_to_openclaw(text: str, speaker: str) -> None:
    """POST transcript to OpenClaw, extract <summary>, call TTS, queue audio."""
    if not bridge.use_openclaw or not bridge.openclaw_url:
        return

    message = f"[{speaker}]: {text}" if speaker and speaker != "Unknown" else text
    payload = {
        "text": message,
        "mode": "now",
        "sessionKey": "agent:main:main",
    }
    headers = {
        "Authorization": f"Bearer {bridge.openclaw_token}",
        "Content-Type": "application/json",
    }

    try:
        resp = http_requests.post(
            bridge.openclaw_url,
            json=payload,
            headers=headers,
            timeout=30,
            verify=False,
        )
        if not resp.ok:
            log.warning("OpenClaw returned %d", resp.status_code)
            return

        body = resp.json()
        content = ""
        if isinstance(body, dict):
            content = (
                body.get("response", "")
                or body.get("text", "")
                or body.get("content", "")
            )
            if not content and "choices" in body:
                choices = body["choices"]
                if choices:
                    content = choices[0].get("message", {}).get("content", "")

        if not content:
            return

        m = _SUMMARY_RE.search(content)
        tts_text = m.group(1).strip() if m else content

        if tts_text:
            _call_tts_and_queue(tts_text)

    except Exception:
        log.exception("OpenClaw injection failed")


def _call_tts_and_queue(text: str) -> None:
    """Call the TTS endpoint and queue the PCM result for the Node bridge."""
    tts_url = bridge.config.get("tts_url", "http://127.0.0.1:8090")
    reference_id = bridge.config.get("tts_reference_id", "cullen")
    tts_sample_rate = bridge.config.get("tts_sample_rate", 24000)

    log.info("Calling TTS for: %s", text[:80])
    try:
        resp = http_requests.post(
            f"{tts_url}/v1/audio/speech",
            json={
                "model": "tts-1",
                "voice": reference_id,
                "input": text,
                "response_format": "pcm",
            },
            timeout=30,
        )
        if resp.ok:
            pcm_bytes = resp.content
            log.info("TTS returned %d bytes PCM", len(pcm_bytes))
            bridge.enqueue_tts(pcm_bytes, tts_sample_rate)
        else:
            log.warning("TTS returned %d", resp.status_code)
    except Exception:
        log.exception("TTS call failed")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Lloyd Discord Voice Bridge Server")


@app.post("/v1/discord_audio")
async def receive_discord_audio(payload: DiscordAudioPayload):
    """Receive PCM audio from the Node.js Discord bridge."""
    pipeline = bridge.pipeline
    if not pipeline or not bridge.initialized:
        return JSONResponse(
            status_code=503,
            content={"error": "pipeline not ready"},
        )

    audio_queue = pipeline.discord_audio_queue
    if audio_queue is None:
        return JSONResponse(
            status_code=503,
            content={"error": "discord audio queue not available"},
        )

    try:
        pcm_bytes = base64.b64decode(payload.pcm_base64)
        pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)

        # Feed the entire buffer into the queue as-is; _DiscordAudioReader
        # handles reassembly into the correct frame size internally.
        audio_queue.put(pcm_int16)

        return {"status": "ok", "samples": len(pcm_int16)}

    except Exception as e:
        log.exception("Failed to process discord audio")
        return JSONResponse(
            status_code=400,
            content={"error": str(e)},
        )


@app.get("/v1/discord_tts_queue")
async def get_tts_queue():
    """Return queued TTS audio for the Node.js bridge to play."""
    item = bridge.dequeue_tts()
    if item is None:
        return JSONResponse(status_code=204, content=None)
    return item


@app.get("/v1/discord_status")
async def get_status():
    """Return current pipeline state."""
    return {
        "state": bridge.state,
        "initialized": bridge.initialized,
        "transcript_count": len(bridge.transcript_history),
        "tts_queue_size": len(bridge.tts_queue),
    }


@app.post("/v1/discord_tts")
async def manual_tts(req: TtsRequest):
    """Manually trigger TTS and queue the audio."""
    text = req.text.strip()
    if not text:
        return JSONResponse(
            status_code=400,
            content={"error": "no text provided"},
        )

    threading.Thread(
        target=_call_tts_and_queue, args=(text,), daemon=True
    ).start()
    return {"status": "queued", "text": text}


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def _start_pipeline() -> None:
    """Load config, create pipeline, init components."""
    log.info("Loading config from %s", CONFIG_PATH)
    config = load_config(str(CONFIG_PATH))

    # Override input mode to discord
    config["input_mode"] = "discord"
    bridge.config = config

    # OpenClaw integration
    bridge.use_openclaw = config.get("use_openclaw", False)
    if bridge.use_openclaw:
        raw_url = config.get("openclaw_url", "")
        gateway_base = raw_url.rsplit("/v1/", 1)[0]
        bridge.openclaw_url = f"{gateway_base}/hooks/wake"
        bridge.openclaw_token = config.get("openclaw_token", "")
        log.info("OpenClaw: %s", bridge.openclaw_url)

    callbacks = DiscordPipelineCallbacks()
    bridge.pipeline = PipelineRunner(config, callbacks)

    # Init components (blocking — runs in thread)
    bridge.pipeline.init_components()


def main() -> None:
    log.info("Starting Discord Voice Bridge Server on port 8096")

    # Start pipeline init in background
    init_thread = threading.Thread(target=_start_pipeline, daemon=True)
    init_thread.start()

    # Run uvicorn
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=8096,
        log_level="info",
    )
    server = uvicorn.Server(config)

    # Handle signals
    def _shutdown(sig, frame):
        log.info("Received signal %s, shutting down", sig)
        if bridge.pipeline:
            bridge.pipeline.stop()
        server.should_exit = True

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    server.run()


if __name__ == "__main__":
    main()
