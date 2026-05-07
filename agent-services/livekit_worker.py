"""Lloyd LiveKit agent worker — STT bridge.

Joins every `${room_prefix}*` room that has a participant, segments their
audio into utterances via energy-based VAD, transcribes each utterance with
faster-whisper, and POSTs the transcript to /api/voice/inject so it lands
as a user turn in the matching chat session.

Phase 4 deliverable. No TTS / agent audio publishing yet — that's 5A.

Run via:
  python agent-services/livekit_worker.py

Or under supervisord (agent-services/supervisor/conf.d/lloyd-agent-worker.conf).

Config — see config.yaml `livekit:` block:
  livekit.url, .api_key, .api_secret, .room_prefix, .agent_identity
  livekit.stt.{model, device, compute_type, language, beam_size}
  livekit.vad.{speech_rms, silence_ms, min_utterance_ms, max_utterance_ms}
"""
from __future__ import annotations

import asyncio
import io
import logging
import os
import signal
import sys
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from livekit import api as lkapi
from livekit import rtc


LOG = logging.getLogger("lloyd-agent-worker")

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"
INJECT_URL = "http://127.0.0.1:8080/api/voice/inject"

POLL_INTERVAL = 2.0           # seconds between RoomService polls
DEFAULT_ROOM_PREFIX = "lloyd-"


def _load_cfg() -> dict:
    with CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f) or {}
    lk = cfg.get("livekit") or {}
    if not lk.get("url") or not lk.get("api_key") or not lk.get("api_secret"):
        raise SystemExit("config.yaml: livekit.{url,api_key,api_secret} are required")
    return cfg


def _http_url(ws_url: str) -> str:
    if ws_url.startswith("ws://"):
        return "http://" + ws_url[len("ws://"):]
    if ws_url.startswith("wss://"):
        return "https://" + ws_url[len("wss://"):]
    return ws_url


# ── STT ──────────────────────────────────────────────────────────────────

class WhisperSTT:
    """Lazy faster-whisper wrapper. Loads on first transcribe call so the
    worker can advertise itself as ready before the model finishes downloading."""

    def __init__(self, stt_cfg: dict) -> None:
        self.cfg = stt_cfg
        self._model = None
        self._lock = asyncio.Lock()

    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._lock:
            if self._model is not None:
                return
            from faster_whisper import WhisperModel
            model_name = self.cfg.get("model", "base.en")
            device = self.cfg.get("device", "cpu")
            compute_type = self.cfg.get("compute_type", "int8")
            LOG.info("loading faster-whisper model=%s device=%s compute=%s",
                     model_name, device, compute_type)
            t0 = time.monotonic()
            # First-run downloads weights to ~/.cache/huggingface; subsequent
            # loads are instant.
            self._model = await asyncio.to_thread(
                WhisperModel, model_name, device=device, compute_type=compute_type
            )
            LOG.info("faster-whisper loaded in %.1fs", time.monotonic() - t0)

    async def transcribe(self, samples_int16: np.ndarray, sample_rate: int) -> str:
        """Transcribe an utterance. Returns empty string for non-speech."""
        await self._ensure_loaded()
        # Wrap PCM into a WAV BytesIO so faster-whisper.decode_audio can
        # resample it to 16kHz mono float32 via PyAV. Avoids hand-rolling
        # a resampler.
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(samples_int16.tobytes())
        buf.seek(0)

        def _run() -> str:
            segments, _info = self._model.transcribe(
                buf,
                language=self.cfg.get("language", "en"),
                beam_size=int(self.cfg.get("beam_size", 1)),
                vad_filter=False,           # we already segmented
                condition_on_previous_text=False,
            )
            return "".join(s.text for s in segments).strip()

        return await asyncio.to_thread(_run)


# ── VAD / utterance segmentation ─────────────────────────────────────────

class UtteranceSegmenter:
    """Energy-based VAD. Feeds it int16 PCM frames and gets back utterance
    chunks once a trailing silence threshold is crossed.

    State machine:
      silence  →[loud frame]→  speaking  →[silence_ms of quiet]→ emit utterance
                              ⤷ (max_utterance_ms hard cap also emits)

    Buffers a small "lead-in" so the first phoneme isn't clipped.
    """

    LEAD_IN_MS = 200

    def __init__(self, vad_cfg: dict, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self.speech_rms = float(vad_cfg.get("speech_rms", 0.012))
        self.silence_ms = int(vad_cfg.get("silence_ms", 700))
        self.min_ms = int(vad_cfg.get("min_utterance_ms", 300))
        self.max_ms = int(vad_cfg.get("max_utterance_ms", 30000))
        self._lead_in_samples = int(self.LEAD_IN_MS / 1000 * sample_rate)
        self._silence_samples = int(self.silence_ms / 1000 * sample_rate)
        self._max_samples = int(self.max_ms / 1000 * sample_rate)
        self._min_samples = int(self.min_ms / 1000 * sample_rate)

        # Rolling buffer of recent silence so we can prepend lead-in.
        self._lead_buf: list[np.ndarray] = []
        self._lead_count = 0
        # Active utterance buffer.
        self._buf: list[np.ndarray] = []
        self._buf_count = 0
        self._silence_run = 0
        self._speaking = False

    def _rms(self, frame: np.ndarray) -> float:
        if frame.size == 0:
            return 0.0
        # frame is int16; normalize by 32768.
        f32 = frame.astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(f32 * f32)))

    def push(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Append a PCM int16 frame; return a complete utterance if one ends.
        Frame must be 1-D int16."""
        rms = self._rms(frame)
        is_speech = rms >= self.speech_rms

        if not self._speaking:
            # Maintain lead-in ring.
            self._lead_buf.append(frame)
            self._lead_count += frame.size
            while self._lead_count - (self._lead_buf[0].size if self._lead_buf else 0) > self._lead_in_samples:
                self._lead_count -= self._lead_buf[0].size
                self._lead_buf.pop(0)

            if is_speech:
                # Flip into "speaking". Move the lead-in buffer into the
                # utterance buffer so we keep the first phoneme.
                self._speaking = True
                self._buf = list(self._lead_buf)
                self._buf_count = self._lead_count
                self._lead_buf = []
                self._lead_count = 0
                self._silence_run = 0
            return None

        # Speaking — keep accumulating.
        self._buf.append(frame)
        self._buf_count += frame.size
        if is_speech:
            self._silence_run = 0
        else:
            self._silence_run += frame.size

        ended = self._silence_run >= self._silence_samples
        capped = self._buf_count >= self._max_samples
        if ended or capped:
            utterance = np.concatenate(self._buf) if self._buf else np.zeros(0, dtype=np.int16)
            self._reset()
            if utterance.size < self._min_samples:
                return None  # too short, drop
            return utterance
        return None

    def _reset(self) -> None:
        self._buf = []
        self._buf_count = 0
        self._silence_run = 0
        self._speaking = False
        self._lead_buf = []
        self._lead_count = 0


# ── Per-room bridge ──────────────────────────────────────────────────────

class RoomBridge:
    """One-room connection: subscribes to remote audio, segments utterances,
    transcribes them, and POSTs each transcript to /api/voice/inject."""

    def __init__(self, room_name: str, lk_cfg: dict, stt: WhisperSTT,
                 vad_cfg: dict, http_client) -> None:
        self.room_name = room_name
        self.lk_cfg = lk_cfg
        self.stt = stt
        self.vad_cfg = vad_cfg
        self.http = http_client
        self.room = rtc.Room()
        self._tasks: list[asyncio.Task] = []
        # Derive session_id from room name: "lloyd-${session_id}" → session_id
        prefix = lk_cfg.get("room_prefix", DEFAULT_ROOM_PREFIX)
        self.session_id = room_name[len(prefix):] if room_name.startswith(prefix) else room_name

    async def connect(self) -> None:
        token = (
            lkapi.AccessToken(self.lk_cfg["api_key"], self.lk_cfg["api_secret"])
            .with_identity(self.lk_cfg.get("agent_identity", "lloyd-agent"))
            .with_name("Lloyd")
            .with_grants(lkapi.VideoGrants(
                room=self.room_name,
                room_join=True,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            ))
            .to_jwt()
        )
        self.room.on("track_subscribed", self._on_track_subscribed)
        self.room.on("participant_disconnected", self._on_participant_disconnected)

        LOG.info("[%s] connecting (session_id=%s)", self.room_name, self.session_id)
        await self.room.connect(self.lk_cfg["url"], token)
        LOG.info("[%s] connected as %s", self.room_name, self.room.local_participant.identity)

    async def disconnect(self) -> None:
        for t in self._tasks:
            t.cancel()
        await self.room.disconnect()
        LOG.info("[%s] disconnected", self.room_name)

    def _on_track_subscribed(self, track, publication, participant) -> None:  # noqa: ARG002
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        LOG.info("[%s] subscribed to audio from %s", self.room_name, participant.identity)
        task = asyncio.create_task(self._consume_audio(track, participant.identity))
        self._tasks.append(task)

    def _on_participant_disconnected(self, participant) -> None:
        LOG.info("[%s] participant left: %s", self.room_name, participant.identity)

    async def _consume_audio(self, track, identity: str) -> None:
        stream = rtc.AudioStream(track)
        segmenter: Optional[UtteranceSegmenter] = None
        try:
            async for evt in stream:
                frame = evt.frame
                samples = np.frombuffer(frame.data, dtype=np.int16)
                if frame.num_channels > 1:
                    samples = samples.reshape(-1, frame.num_channels).mean(axis=1).astype(np.int16)
                if segmenter is None:
                    segmenter = UtteranceSegmenter(self.vad_cfg, frame.sample_rate)
                utterance = segmenter.push(samples)
                if utterance is not None:
                    asyncio.create_task(self._handle_utterance(utterance, frame.sample_rate, identity))
        except asyncio.CancelledError:
            pass
        finally:
            await stream.aclose()

    async def _handle_utterance(self, samples: np.ndarray, sample_rate: int, identity: str) -> None:
        duration_s = samples.size / sample_rate
        t0 = time.monotonic()
        try:
            text = await self.stt.transcribe(samples, sample_rate)
        except Exception as e:
            LOG.warning("[%s] STT failed for %.2fs utterance: %s", self.room_name, duration_s, e)
            return
        latency = time.monotonic() - t0
        if not text:
            LOG.info("[%s] empty transcript for %.2fs utterance from %s (%.1fs whisper)",
                     self.room_name, duration_s, identity, latency)
            return
        LOG.info("[%s] %s → %r  (%.2fs audio, %.1fs whisper)",
                 self.room_name, identity, text, duration_s, latency)
        try:
            r = await self.http.post(
                INJECT_URL,
                json={
                    "text": text,
                    "speaker": identity,
                    "session_key": self.session_id,
                },
                timeout=10.0,
            )
            if r.status_code >= 300:
                LOG.warning("[%s] inject failed %d: %s",
                            self.room_name, r.status_code, r.text[:200])
        except Exception as e:
            LOG.warning("[%s] inject POST failed: %s", self.room_name, e)


# ── Worker manager ───────────────────────────────────────────────────────

class WorkerManager:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.lk_cfg = cfg["livekit"]
        self.room_prefix = self.lk_cfg.get("room_prefix", DEFAULT_ROOM_PREFIX)
        self.bridges: dict[str, RoomBridge] = {}
        self._stopping = asyncio.Event()
        self._http_url = _http_url(self.lk_cfg["url"])
        self.stt = WhisperSTT(self.lk_cfg.get("stt", {}))
        self.vad_cfg = self.lk_cfg.get("vad", {})
        # Lazy-imported to avoid hard dep ordering with httpx.
        import httpx
        self._http = httpx.AsyncClient()

    async def run(self) -> None:
        LOG.info("worker starting; polling %s every %.1fs for rooms with prefix %r",
                 self._http_url, POLL_INTERVAL, self.room_prefix)
        # Eager-load STT so the first utterance doesn't pay the load latency.
        try:
            await self.stt._ensure_loaded()
        except Exception as e:
            LOG.warning("STT eager-load failed (will retry on first utterance): %s", e)
        try:
            while not self._stopping.is_set():
                try:
                    await self._tick()
                except Exception as e:
                    LOG.warning("tick failed: %s", e)
                try:
                    await asyncio.wait_for(self._stopping.wait(), timeout=POLL_INTERVAL)
                except asyncio.TimeoutError:
                    pass
        finally:
            await self._teardown()

    async def _tick(self) -> None:
        async with lkapi.LiveKitAPI(self._http_url, self.lk_cfg["api_key"], self.lk_cfg["api_secret"]) as svc:
            rooms = (await svc.room.list_rooms(lkapi.ListRoomsRequest())).rooms

        active: dict[str, int] = {}
        for r in rooms:
            if not r.name.startswith(self.room_prefix):
                continue
            non_agent = max(0, r.num_participants - (1 if r.name in self.bridges else 0))
            if non_agent > 0:
                active[r.name] = non_agent

        for room_name in active:
            if room_name in self.bridges:
                continue
            bridge = RoomBridge(room_name, self.lk_cfg, self.stt, self.vad_cfg, self._http)
            try:
                await bridge.connect()
                self.bridges[room_name] = bridge
            except Exception as e:
                LOG.warning("[%s] connect failed: %s", room_name, e)

        for room_name in list(self.bridges):
            if room_name not in active:
                bridge = self.bridges.pop(room_name)
                try:
                    await bridge.disconnect()
                except Exception:
                    pass

    async def _teardown(self) -> None:
        for bridge in list(self.bridges.values()):
            try:
                await bridge.disconnect()
            except Exception:
                pass
        self.bridges.clear()
        try:
            await self._http.aclose()
        except Exception:
            pass

    def request_stop(self) -> None:
        self._stopping.set()


async def _amain() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    cfg = _load_cfg()
    manager = WorkerManager(cfg)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, manager.request_stop)
        except NotImplementedError:
            pass

    await manager.run()


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    # Suppress noisy module loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("livekit").setLevel(logging.INFO)
    asyncio.run(_amain())
