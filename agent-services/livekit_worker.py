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


def _looks_repetitive(text: str) -> bool:
    """Heuristic for Whisper hallucinations on near-silence: any 1- or
    2-word phrase repeated 4+ times consecutively. Catches both "Bye.
    Bye. Bye." and "Thank you. Thank you. Thank you. Thank you." styles
    without affecting normal speech."""
    if not text:
        return False
    import re
    words = re.findall(r"[A-Za-z']+", text.lower())
    if len(words) < 4:
        return False
    for ngram in (1, 2):
        if len(words) < ngram * 4:
            continue
        run = 1
        prev = tuple(words[:ngram])
        for i in range(ngram, len(words) - ngram + 1, ngram):
            cur = tuple(words[i:i + ngram])
            if cur == prev:
                run += 1
                if run >= 4:
                    return True
            else:
                run = 1
            prev = cur
    return False


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
                # Tighten hallucination thresholds — defaults are tuned for
                # long-form audio; short utterances need stricter gates or
                # whisper happily produces "Bye. Bye. Bye…" on near-silence.
                no_speech_threshold=0.7,
                log_prob_threshold=-0.8,
                compression_ratio_threshold=2.0,
            )
            return "".join(s.text for s in segments).strip()

        text = await asyncio.to_thread(_run)
        # Catch the residual hallucinations whisper still ships through:
        # "Thank you. Thank you. ... Bye. Bye. Bye." style output where the
        # same short token repeats. Drop the whole transcript if any token
        # repeats ≥ 4 times consecutively in a small window.
        if _looks_repetitive(text):
            return ""
        return text


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
        self.speech_rms = float(vad_cfg.get("speech_rms", 0.025))
        self.silence_ms = int(vad_cfg.get("silence_ms", 500))
        self.min_ms = int(vad_cfg.get("min_utterance_ms", 350))
        self.max_ms = int(vad_cfg.get("max_utterance_ms", 30000))
        self.min_start_frames = int(vad_cfg.get("min_speech_frames_to_start", 3))
        self.min_voiced_ratio = float(vad_cfg.get("min_voiced_ratio", 0.30))
        self._lead_in_samples = int(self.LEAD_IN_MS / 1000 * sample_rate)
        self._silence_samples = int(self.silence_ms / 1000 * sample_rate)
        self._max_samples = int(self.max_ms / 1000 * sample_rate)
        self._min_samples = int(self.min_ms / 1000 * sample_rate)

        # Rolling buffer of recent silence so we can prepend lead-in.
        self._lead_buf: list[np.ndarray] = []
        self._lead_count = 0
        # Pending consecutive-loud-frames counter before we commit to
        # "speaking" — debounces single noise spikes.
        self._pending_speech_frames = 0
        # Active utterance buffer.
        self._buf: list[np.ndarray] = []
        self._buf_count = 0
        self._voiced_samples = 0
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

            # Require N consecutive voiced frames to commit to "speaking".
            # Suppresses single mic clicks / noise spikes.
            if is_speech:
                self._pending_speech_frames += 1
            else:
                self._pending_speech_frames = 0

            if self._pending_speech_frames >= self.min_start_frames:
                self._speaking = True
                self._buf = list(self._lead_buf)
                self._buf_count = self._lead_count
                # Lead-in samples don't count towards voiced_samples — they're
                # the silence we kept around to avoid clipping the first
                # phoneme. Voiced ratio is computed from speaking-state frames.
                self._voiced_samples = 0
                self._lead_buf = []
                self._lead_count = 0
                self._silence_run = 0
                self._pending_speech_frames = 0
            return None

        # Speaking — keep accumulating.
        self._buf.append(frame)
        self._buf_count += frame.size
        if is_speech:
            self._voiced_samples += frame.size
            self._silence_run = 0
        else:
            self._silence_run += frame.size

        ended = self._silence_run >= self._silence_samples
        capped = self._buf_count >= self._max_samples
        if ended or capped:
            utterance = np.concatenate(self._buf) if self._buf else np.zeros(0, dtype=np.int16)
            voiced = self._voiced_samples
            buf_count = self._buf_count
            self._reset()
            if utterance.size < self._min_samples:
                return None
            # Voiced-ratio gate: real speech is mostly voiced; near-silence
            # utterances that briefly tripped the trigger should be dropped.
            # Compute against the speaking-state frames only (excludes
            # lead-in and trailing silence).
            voicing_window = max(1, buf_count - self._lead_in_samples - self._silence_samples)
            voiced_ratio = voiced / voicing_window
            if voiced_ratio < self.min_voiced_ratio:
                return None
            return utterance
        return None

    def _reset(self) -> None:
        self._buf = []
        self._buf_count = 0
        self._voiced_samples = 0
        self._silence_run = 0
        self._speaking = False
        self._lead_buf = []
        self._lead_count = 0
        self._pending_speech_frames = 0


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
        # Strong refs to in-flight utterance handlers. asyncio's create_task
        # docs: "Save a reference … to avoid a task disappearing mid-execution"
        # — without this, the task can be GC'd between the transcribe log
        # line and the inject POST, dropping transcripts on the floor.
        self._utterance_tasks: set[asyncio.Task] = set()
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
        self.room.on("participant_connected", self._on_participant_connected)
        self.room.on("participant_disconnected", self._on_participant_disconnected)

        LOG.info("[%s] connecting (session_id=%s)", self.room_name, self.session_id)
        await self.room.connect(self.lk_cfg["url"], token)
        LOG.info("[%s] connected as %s", self.room_name, self.room.local_participant.identity)

    @property
    def has_remote_participants(self) -> bool:
        """True if any non-agent participant is currently in the room.
        We trust the room's own participant set over LiveKit RoomService
        polling — the room is the source of truth for our connection."""
        try:
            return len(self.room.remote_participants) > 0
        except Exception:
            return False

    async def disconnect(self) -> None:
        for t in self._tasks:
            t.cancel()
        # Let any in-flight utterance handler finish (transcribe + POST) so
        # the last thing the user said before leaving still lands in the
        # session. Cap the wait so we don't hang on a stuck POST.
        if self._utterance_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._utterance_tasks, return_exceptions=True),
                    timeout=8.0,
                )
            except asyncio.TimeoutError:
                LOG.warning("[%s] utterance tasks still running at shutdown — abandoning",
                            self.room_name)
        await self.room.disconnect()
        LOG.info("[%s] disconnected", self.room_name)

    def _on_track_subscribed(self, track, publication, participant) -> None:  # noqa: ARG002
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        LOG.info("[%s] subscribed to audio from %s", self.room_name, participant.identity)
        task = asyncio.create_task(self._consume_audio(track, participant.identity))
        self._tasks.append(task)

    def _on_participant_connected(self, participant) -> None:
        LOG.info("[%s] participant joined: %s", self.room_name, participant.identity)

    def _on_participant_disconnected(self, participant) -> None:
        LOG.info("[%s] participant left: %s", self.room_name, participant.identity)

    def _handle_utterance_done(self, task: asyncio.Task) -> None:
        self._utterance_tasks.discard(task)
        # Surface anything the handler swallowed silently. If we don't pull
        # the exception out, asyncio prints "Task exception was never
        # retrieved" at GC time — which masks the actual failure mode.
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            LOG.warning("[%s] utterance handler raised: %r", self.room_name, exc)

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
                    task = asyncio.create_task(
                        self._handle_utterance(utterance, frame.sample_rate, identity)
                    )
                    self._utterance_tasks.add(task)
                    task.add_done_callback(self._handle_utterance_done)
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
            # No speaker field: the LiveKit identity ("user-XXXXXX") isn't a
            # real speaker name — it's a per-connection token id — and
            # /api/voice/inject prepends "[speaker]: " when it's set. Diarization
            # / speaker recognition arrive in a later phase and will populate
            # this with a real name.
            r = await self.http.post(
                INJECT_URL,
                json={
                    "text": text,
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
        # last time we saw a non-agent participant in each room — used for
        # the idle-grace teardown logic.
        self._last_seen_remote: dict[str, float] = {}
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

    # Rooms we just disconnected from — backoff before considering rejoin
    # to avoid the polling/event race when a participant briefly drops.
    _IDLE_GRACE_SECONDS = 15.0

    async def _tick(self) -> None:
        async with lkapi.LiveKitAPI(self._http_url, self.lk_cfg["api_key"], self.lk_cfg["api_secret"]) as svc:
            rooms = (await svc.room.list_rooms(lkapi.ListRoomsRequest())).rooms

        # Connect to rooms that have non-agent participants and we're not
        # currently in. RoomService participant counts include the agent
        # only after we've joined, so subtract our presence.
        for r in rooms:
            if not r.name.startswith(self.room_prefix):
                continue
            already_in = r.name in self.bridges
            non_agent = max(0, r.num_participants - (1 if already_in else 0))
            if already_in or non_agent <= 0:
                continue
            bridge = RoomBridge(r.name, self.lk_cfg, self.stt, self.vad_cfg, self._http)
            try:
                await bridge.connect()
                self.bridges[r.name] = bridge
                self._last_seen_remote[r.name] = time.monotonic()
            except Exception as e:
                LOG.warning("[%s] connect failed: %s", r.name, e)

        # Trust the room's own participant set for "should I stay or go?".
        # Only tear down a bridge after _IDLE_GRACE_SECONDS of zero remote
        # participants — otherwise a participant momentarily flickering
        # would force an immediate reconnect.
        now = time.monotonic()
        for room_name, bridge in list(self.bridges.items()):
            if bridge.has_remote_participants:
                self._last_seen_remote[room_name] = now
                continue
            idle_for = now - self._last_seen_remote.get(room_name, now)
            if idle_for >= self._IDLE_GRACE_SECONDS:
                LOG.info("[%s] no remote participants for %.0fs — disconnecting",
                         room_name, idle_for)
                self.bridges.pop(room_name, None)
                self._last_seen_remote.pop(room_name, None)
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
