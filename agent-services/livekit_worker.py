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
import json
import logging
import os
import signal
import sys
import time
import uuid
import wave
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import re
import yaml
from livekit import api as lkapi
from livekit import rtc


LOG = logging.getLogger("lloyd-agent-worker")

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"
ENV_PATH = REPO_ROOT / ".env"
INJECT_URL = "http://127.0.0.1:8080/api/voice/inject"
SUMMARIZE_URL = "http://127.0.0.1:8080/api/voice/summarize"

_ENV_VAR_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _load_env_file(path: Path) -> None:
    """Mirror of app.config._load_env_file — keeps this module standalone.

    Reads simple KEY=VALUE lines from .env into os.environ (without
    overriding values already present). Supervisord-set env still wins.
    """
    if not path.exists():
        return
    try:
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                v = v[1:-1]
            os.environ.setdefault(k, v)
    except OSError:
        pass


def _expand_env(value):
    """Recursively expand ${VAR} in strings within a dict/list tree."""
    if isinstance(value, str):
        return _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value

POLL_INTERVAL = 2.0           # seconds between RoomService polls
DEFAULT_ROOM_PREFIX = "lloyd-"


def _build_speaker_id(vp_cfg: dict):
    """Construct a SpeakerIdentifier from `livekit.voiceprint` config, or
    return None when disabled / construction fails. We swallow construction
    failures (resemblyzer missing, profiles_dir unwritable) because voice
    works fine without it — degrading to identity-only matching is the
    sensible fallback."""
    if not vp_cfg.get("enabled", True):
        LOG.info("voiceprint matching disabled in config")
        return None
    try:
        from speaker_id import SpeakerIdentifier
        return SpeakerIdentifier(
            profiles_dir=vp_cfg.get("profiles_dir", "~/lloyd/voice_profiles"),
            threshold=float(vp_cfg.get("profile_threshold", 0.75)),
            unknown_label=str(vp_cfg.get("unknown_label", "Unknown")),
            device=str(vp_cfg.get("device", "cpu")),
        )
    except Exception as e:
        LOG.warning("voiceprint init failed (%s) — falling back to identity matching", e)
        return None


def _load_cfg() -> dict:
    _load_env_file(ENV_PATH)
    with CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _expand_env(cfg)
    lk = cfg.get("livekit") or {}
    if not lk.get("url") or not lk.get("api_key") or not lk.get("api_secret"):
        raise SystemExit("config.yaml: livekit.{url,api_key,api_secret} are required (check .env for LIVEKIT_API_KEY/LIVEKIT_API_SECRET)")
    return cfg


def _http_url(ws_url: str) -> str:
    if ws_url.startswith("ws://"):
        return "http://" + ws_url[len("ws://"):]
    if ws_url.startswith("wss://"):
        return "https://" + ws_url[len("wss://"):]
    return ws_url


# ── TTS ─────────────────────────────────────────────────────────────────

class TTSStreamer:
    """Streams PCM from the Qwen3-TTS HTTP server into a LiveKit AudioSource.

    One instance per RoomBridge. Holds the published track; queues
    utterances; runs them serially so the agent's voice doesn't overlap
    itself when the harness produces multiple replies in quick succession.
    """

    # LiveKit AudioFrame chunks must be a multiple of 10 ms for the SDK
    # to accept them. 100 ms gives a comfortable buffer with ~1 frame of
    # latency added.
    FRAME_MS = 100

    def __init__(
        self,
        tts_cfg: dict,
        room: "rtc.Room",
        on_utterance_end: Optional[Callable[[], None]] = None,
    ) -> None:
        self.cfg = tts_cfg
        self.api_url = (tts_cfg.get("api_url") or "http://127.0.0.1:8090").rstrip("/")
        self.model = tts_cfg.get("model", "qwen3-tts")
        self.voice = tts_cfg.get("voice", "clone:cullen")
        self.speed = float(tts_cfg.get("speed", 1.0))
        self.sample_rate = int(tts_cfg.get("sample_rate", 24000))
        self.room = room
        # Optional callback fired (synchronously, no args) when an utterance
        # finishes draining — RoomBridge wires this to extend the wake-word
        # continuation window from end-of-speak.
        self.on_utterance_end = on_utterance_end
        self.source: Optional["rtc.AudioSource"] = None
        self.track: Optional["rtc.LocalAudioTrack"] = None
        self._publish_lock = asyncio.Lock()
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None
        # Currently-streaming utterance task (set by _drain). interrupt()
        # cancels it; _drain catches the CancelledError and moves on.
        self._current_task: Optional[asyncio.Task] = None
        self._speaking = asyncio.Event()
        # Lazy import — keeps top-of-file clean.
        import httpx
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=120.0))

    async def ensure_published(self) -> None:
        async with self._publish_lock:
            if self.track is not None:
                return
            self.source = rtc.AudioSource(self.sample_rate, 1)
            self.track = rtc.LocalAudioTrack.create_audio_track("lloyd-tts", self.source)
            options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            await self.room.local_participant.publish_track(self.track, options)
            self._worker_task = asyncio.create_task(self._drain())
            LOG.info("TTSStreamer published track 'lloyd-tts' @ %d Hz", self.sample_rate)

    async def speak(self, text: str) -> None:
        """Queue an utterance for synthesis + playback."""
        text = (text or "").strip()
        if not text:
            return
        await self.ensure_published()
        await self._queue.put(text)

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    async def close(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except (asyncio.CancelledError, Exception):
                pass
        try:
            await self._http.aclose()
        except Exception:
            pass

    def interrupt(self) -> int:
        """Cancel the in-flight utterance and drop everything queued.

        Returns the number of queued utterances that were dropped (the
        in-flight one isn't counted). Safe to call when nothing is
        playing — both operations are no-ops in that case.
        """
        dropped = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                dropped += 1
            except asyncio.QueueEmpty:
                break
        if self._current_task is not None and not self._current_task.done():
            self._current_task.cancel()
        # Best-effort: ask the AudioSource to drop any buffered frames.
        # The Python SDK exposes `clear_queue()` in recent versions; older
        # ones don't, in which case ~100ms of trailing audio may still
        # reach the browser before silence resumes.
        if self.source is not None:
            try:
                self.source.clear_queue()
            except Exception:
                pass
        return dropped

    async def _drain(self) -> None:
        while True:
            text = await self._queue.get()
            self._speaking.set()
            self._current_task = asyncio.create_task(self._stream_utterance(text))
            try:
                await self._current_task
            except asyncio.CancelledError:
                LOG.info("TTS interrupted mid-utterance")
            except Exception as e:
                LOG.warning("TTS error for %r: %s", text[:60], e)
            finally:
                self._current_task = None
                self._speaking.clear()
                # Notify the bridge that this utterance finished draining so
                # it can extend the wake-word continuation window. Best-effort:
                # any callback exception is swallowed, the drain loop survives.
                cb = self.on_utterance_end
                if cb is not None:
                    try:
                        cb()
                    except Exception as e:
                        LOG.warning("on_utterance_end callback raised: %s", e)

    async def _stream_utterance(self, text: str) -> None:
        """POST to /v1/audio/speech with stream:true,response_format:pcm and
        push every PCM chunk into the LiveKit AudioSource."""
        if self.source is None:
            return
        # 100ms LiveKit frame size, in samples + bytes.
        samples_per_frame = self.sample_rate * self.FRAME_MS // 1000
        bytes_per_frame = samples_per_frame * 2
        leftover = bytearray()

        t0 = time.monotonic()
        n_pushed = 0
        async with self._http.stream(
            "POST",
            f"{self.api_url}/v1/audio/speech",
            json={
                "model": self.model,
                "input": text,
                "voice": self.voice,
                "response_format": "pcm",
                "stream": True,
                "speed": self.speed,
            },
        ) as resp:
            if resp.status_code >= 300:
                err = await resp.aread()
                LOG.warning("TTS HTTP %d: %s", resp.status_code, err[:200])
                return
            async for chunk in resp.aiter_bytes():
                if not chunk:
                    continue
                leftover.extend(chunk)
                # Push complete 100ms frames; keep any tail for the next loop.
                while len(leftover) >= bytes_per_frame:
                    frame_bytes = bytes(leftover[:bytes_per_frame])
                    del leftover[:bytes_per_frame]
                    await self._push_frame(frame_bytes, samples_per_frame)
                    n_pushed += 1
        # Tail: any final partial frame (zero-padded to a 10ms boundary).
        if leftover:
            ten_ms = self.sample_rate // 100
            ten_ms_bytes = ten_ms * 2
            tail = bytes(leftover)
            pad = (-len(tail)) % ten_ms_bytes
            if pad:
                tail = tail + b"\x00" * pad
            samples = len(tail) // 2
            if samples:
                await self._push_frame(tail, samples)
                n_pushed += 1
        elapsed = time.monotonic() - t0
        LOG.info("TTS spoke %r in %.2fs (%d frames)", text[:60], elapsed, n_pushed)

    async def _push_frame(self, pcm_bytes: bytes, samples_per_channel: int) -> None:
        if self.source is None:
            return
        frame = rtc.AudioFrame(
            data=pcm_bytes,
            sample_rate=self.sample_rate,
            num_channels=1,
            samples_per_channel=samples_per_channel,
        )
        await self.source.capture_frame(frame)


# ── Wake-word gate / continuation ────────────────────────────────────────

class WakeState:
    """Per-room wake-word gate.

    Two states:
      IDLE         — drop any utterance that doesn't begin with a wake-word.
      CONTINUATION — pass utterances through (from the locked participant)
                     without requiring the wake-word, and extend the window
                     on each pass-through.

    Transition into CONTINUATION on a successful wake-word match or after
    Lloyd finishes a TTS utterance. Transition back to IDLE when the
    continuation window expires.

    Speaker identity is enforced two ways during continuation:
      - LiveKit participant identity (always, cheap: one browser tab → one
        identity)
      - voiceprint anchor cosine (when a SpeakerIdentifier is attached;
        embeds the wake-word utterance and rejects follow-ups whose cosine
        falls below `anchor_threshold`)
    The identity check is necessary-but-not-sufficient when voiceprint is
    on; without voiceprint it's the only check.
    """

    def __init__(self, cfg: dict) -> None:
        self.enabled: bool = bool(cfg.get("enabled", True))
        # Lower-cased + sorted longest-first so "hey lloyd" wins over "lloyd"
        # when the user said "hey lloyd what time is it".
        words = list(cfg.get("words") or ["lloyd"])
        self.words: list[str] = sorted(
            (w.strip().lower() for w in words if w and w.strip()),
            key=len,
            reverse=True,
        )
        self.continuation_seconds: float = float(cfg.get("continuation_seconds", 12.0))
        self.skip_inject_if_only_wake_word: bool = bool(
            cfg.get("skip_inject_if_only_wake_word", True)
        )
        # Set by extend(); compared in in_continuation().
        self._until: float = 0.0
        self._locked_identity: Optional[str] = None
        # Voiceprint anchor — the embedding extracted from the wake-word
        # utterance. Stored alongside the speaker name (from the enrolled
        # profile match, if any) so injects can carry [Alan]: prefixes.
        self._anchor_embedding: Optional[np.ndarray] = None
        self._anchor_name: Optional[str] = None

    def in_continuation(self, at: Optional[float] = None) -> bool:
        """True if the continuation window is open at time `at` (monotonic).
        Defaults to now. Caller passes `at` = utterance start when checking
        whether a freshly-VAD'd utterance qualifies for pass-through — a
        long utterance that started in-window but finished processing just
        past expiry should still count."""
        t = at if at is not None else time.monotonic()
        return t < self._until

    @property
    def locked_identity(self) -> Optional[str]:
        return self._locked_identity if self.in_continuation() else None

    def matches_lock(self, identity: str, at: Optional[float] = None) -> bool:
        """True if `identity` is the locked speaker AND the continuation
        window was open at time `at` (default now)."""
        if self._locked_identity != identity:
            return False
        return self.in_continuation(at=at)

    @property
    def has_lock(self) -> bool:
        """Whether a wake-word has been said at any point in this room's
        history. Distinct from `in_continuation()` which expires with the
        continuation window — `has_lock` stays True so TTS-end can re-open
        the window after a long agent reply that happened to outlast it."""
        return self._locked_identity is not None

    @property
    def anchor_embedding(self) -> Optional[np.ndarray]:
        return self._anchor_embedding if self.in_continuation() else None

    @property
    def anchor_name(self) -> Optional[str]:
        return self._anchor_name if self.in_continuation() else None

    def set_anchor(self, embedding: Optional[np.ndarray], name: Optional[str]) -> None:
        """Lock the voiceprint anchor for the new continuation window.
        Pass None to clear (e.g. when voiceprint matching is disabled)."""
        self._anchor_embedding = embedding
        self._anchor_name = name

    def extend(self, identity: Optional[str] = None) -> None:
        """Reset the continuation window. If `identity` is given, lock to
        that identity; otherwise keep the existing lock (used for TTS-end
        extensions where the speaker hasn't changed)."""
        self._until = time.monotonic() + self.continuation_seconds
        if identity is not None:
            self._locked_identity = identity

    def reset(self) -> None:
        self._until = 0.0
        self._locked_identity = None
        self._anchor_embedding = None
        self._anchor_name = None

    def remaining_s(self) -> float:
        return max(0.0, self._until - time.monotonic())


def _strip_wake_word(text: str, words: list[str]) -> Optional[str]:
    """Match a wake-word at the start of `text` and return the remaining
    content (the user's actual request), or None if no wake-word is
    present.

    Whisper aggressively punctuates output. "Hey Lloyd, what time is it?"
    typically arrives as `'Hey, Lloyd, what time is it?'` with an inserted
    comma — so we match against a normalized version of the text where
    runs of non-alpha characters become single spaces, then strip the
    wake-word region from the original (regex with `\\W+` between word
    parts) so the returned tail keeps the user's original capitalization
    and contractions.

    Word-boundary check: 'lloydian' and 'lloyd's' don't match 'lloyd'.

    Examples (words = ['lloyd', 'hey lloyd']):
      'Hey Lloyd.'             → ''                  (bare wake-word)
      'Hey, Lloyd.'            → ''                  (handles inserted comma)
      'Hey Lloyd, what time?'  → 'what time?'
      'Lloyd, set a timer.'    → 'set a timer.'
      'Lloyd's birthday is...' → None                (apostrophe ≠ boundary)
      'Hello world'            → None
    """
    import re
    normalized = re.sub(r"[^a-z']+", " ", text.lower()).strip()
    if not normalized:
        return None
    for w in words:
        if normalized == w or normalized.startswith(w + " "):
            # Build a regex that matches the wake-word in the original text
            # tolerating any punctuation/whitespace between the word parts
            # and trailing the match. Anchored to the start.
            parts = w.split()
            pattern = r"^\W*" + r"\W+".join(re.escape(p) for p in parts) + r"[\s.,!?;:'\"]*"
            tail = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)
            return tail.strip()
    return None


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


# ── Acoustic wake-word (openWakeWord) ────────────────────────────────────

class AcousticWakeWord:
    """openWakeWord wrapper that detects wake-phrases directly on audio
    rather than going through Whisper text. Loads the user's custom-trained
    Hey_Lloyd / Lloyd ONNX models. Loaded lazily so worker startup stays fast.

    openWakeWord runs on 16kHz int16 mono in 80ms (1280-sample) chunks. We
    resample if the room audio is at a different rate (typically 48kHz),
    then sweep `predict()` across the utterance and keep the max score per
    model. If the max score across any model crosses `threshold`, the
    utterance contained a wake-word.

    Per-utterance state reset matters: `predict()` accumulates state across
    chunks within one utterance (that's how openWakeWord builds confidence),
    so we must reset before each utterance to avoid leaking state from the
    previous one.
    """

    def __init__(self, models_dir: str | Path, engine_dir: str | Path, threshold: float = 0.5):
        self.models_dir = Path(models_dir).expanduser()
        self.engine_dir = Path(engine_dir).expanduser()
        self.threshold = float(threshold)
        self._model = None
        self._lock = asyncio.Lock()

    async def _ensure_loaded(self):
        if self._model is not None:
            return
        async with self._lock:
            if self._model is not None:
                return
            from openwakeword.model import Model
            ww_paths = sorted(str(p) for p in self.models_dir.glob("*.onnx"))
            if not ww_paths:
                raise RuntimeError(f"no .onnx models in {self.models_dir}")
            mel = self.engine_dir / "melspectrogram.onnx"
            emb = self.engine_dir / "embedding_model.onnx"
            if not mel.exists() or not emb.exists():
                raise RuntimeError(f"missing engine files: {mel}, {emb}")
            t0 = time.monotonic()
            self._model = await asyncio.to_thread(
                Model,
                wakeword_model_paths=ww_paths,
                melspec_onnx_model_path=str(mel),
                embedding_onnx_model_path=str(emb),
            )
            LOG.info(
                "openWakeWord loaded in %.2fs (%s) threshold=%.2f",
                time.monotonic() - t0,
                ", ".join(self._model.models.keys()),
                self.threshold,
            )

    def _detect_blocking(self, samples_int16: np.ndarray, sample_rate: int) -> tuple[bool, str, float]:
        # Resample to 16kHz if needed. scipy.signal.resample_poly does a
        # high-quality polyphase resample — much better than naive striding.
        if sample_rate != 16000:
            from scipy.signal import resample_poly
            up, down = 16000, sample_rate
            # Reduce by gcd to keep the polyphase tables small.
            from math import gcd
            g = gcd(up, down)
            samples = resample_poly(samples_int16.astype(np.float32), up // g, down // g)
            samples = np.clip(samples, -32768, 32767).astype(np.int16)
        else:
            samples = samples_int16
        # Reset internal state so we don't leak confidence from a prior
        # utterance — each call starts from zero.
        try:
            self._model.reset()
        except Exception:
            pass
        chunk = 1280  # 80ms @ 16kHz
        max_per_model = {n: 0.0 for n in self._model.models.keys()}
        i = 0
        while i + chunk <= len(samples):
            scores = self._model.predict(samples[i:i + chunk])
            for n, s in scores.items():
                if s > max_per_model[n]:
                    max_per_model[n] = float(s)
            i += chunk
        if not max_per_model:
            return False, "", 0.0
        best_name = max(max_per_model, key=max_per_model.get)
        best_score = max_per_model[best_name]
        return best_score >= self.threshold, best_name, best_score

    async def detect(self, samples_int16: np.ndarray, sample_rate: int) -> tuple[bool, str, float]:
        """Returns (matched, model_name, max_score). All three regardless of
        match — caller can log the score for tuning the threshold."""
        try:
            await self._ensure_loaded()
            return await asyncio.to_thread(self._detect_blocking, samples_int16, sample_rate)
        except Exception as e:
            LOG.warning("acoustic wake-word detect failed: %s", e)
            return False, "", 0.0


def _build_acoustic_wake_word(cfg: dict) -> Optional[AcousticWakeWord]:
    """Construct AcousticWakeWord from the `livekit.acoustic_wake` config
    block, or None when disabled / construction fails. Falls back gracefully
    so the worker still runs (text-match path) if openwakeword isn't
    installed or the model files are missing."""
    if not cfg.get("enabled", True):
        LOG.info("acoustic wake-word disabled in config")
        return None
    try:
        return AcousticWakeWord(
            models_dir=cfg.get("models_dir", "agent-services/models/wakeword"),
            engine_dir=cfg.get("engine_dir", "agent-services/models/openwakeword"),
            threshold=float(cfg.get("threshold", 0.5)),
        )
    except Exception as e:
        LOG.warning("acoustic wake-word init failed (%s) — falling back to text-match only", e)
        return None


# ── Wake-word miss capture (diagnostic rig) ──────────────────────────────

class WakeMissCapture:
    """Diagnostic capture rig for tuning the wake-word detector.

    Maintains three things, all under ~/.lloyd/ww_diag/:
      - scores.jsonl: one structured record per utterance handled — includes
        ww score, threshold, fired flag, transcript prefix, audio path. The
        ground-truth log for replay analysis.
      - utterances/<id>.wav: a copy of every utterance the segmenter
        emitted, so we can replay misses through alternative thresholds /
        models. Bounded at MAX_UTTERANCE_FILES; oldest pruned.
      - misses/<ts>_<label>.{wav,json}: explicit miss reports from the
        /ww_miss endpoint. The wav is the per-room rolling raw-audio ring
        (RING_SECONDS of pre-VAD audio at the room's native rate) — this
        is the only way to recover the speech when VAD never even
        segmented it (the silent failure mode).

    Single-threaded by virtue of running entirely inside the asyncio
    event loop: no locks needed.
    """

    DIAG_DIR = Path("~/.lloyd/ww_diag").expanduser()
    UTTERANCES_DIR = DIAG_DIR / "utterances"
    MISSES_DIR = DIAG_DIR / "misses"
    SCORES_PATH = DIAG_DIR / "scores.jsonl"
    LABELS_PATH = DIAG_DIR / "labels.jsonl"

    MAX_UTTERANCE_FILES = 500
    RING_SECONDS = 5.0
    MISS_RECENT_WINDOW_S = 30.0

    def __init__(self) -> None:
        self.UTTERANCES_DIR.mkdir(parents=True, exist_ok=True)
        self.MISSES_DIR.mkdir(parents=True, exist_ok=True)
        # room_name -> {sr: int, buf: deque[np.ndarray int16], total: int}
        self._rings: dict[str, dict] = {}

    # -- raw audio ring (pre-VAD) --

    def push_raw_frame(self, room: str, samples: np.ndarray, sr: int) -> None:
        """Append an int16 mono frame to the per-room rolling buffer.
        Cheap inline numpy work; safe to call from the audio consumer hot
        loop on every frame."""
        ring = self._rings.get(room)
        if ring is None or ring["sr"] != sr:
            ring = {"sr": int(sr), "buf": deque(), "total": 0}
            self._rings[room] = ring
        ring["buf"].append(samples.astype(np.int16, copy=False))
        ring["total"] += len(samples)
        cap = int(self.RING_SECONDS * sr)
        while ring["total"] > cap and len(ring["buf"]) > 1:
            dropped = ring["buf"].popleft()
            ring["total"] -= len(dropped)

    def drop_room(self, room: str) -> None:
        self._rings.pop(room, None)

    # -- per-utterance record + wav --

    def record_utterance(self, *, utterance_id: str, room: str, identity: str,
                         duration_s: float, rms_mean: float, rms_peak: float,
                         voiced_ratio: float, ww_ran: bool, ww_name: str,
                         ww_score: float, ww_threshold: float, ww_fired: bool,
                         in_continuation: bool, stt_text: str,
                         stt_latency_s: float, samples: np.ndarray,
                         sample_rate: int,
                         client_info: Optional[dict] = None) -> None:
        """Save the utterance audio + append a structured record to
        scores.jsonl. Errors are swallowed and logged — diagnostic code
        must never crash the audio pipeline."""
        audio_path: Optional[Path] = None
        try:
            audio_path = self._save_utterance_wav(utterance_id, samples, sample_rate)
        except Exception as e:
            LOG.warning("ww-diag: utterance wav save failed: %s", e)
        rec = {
            "ts": time.time(),
            "utterance_id": utterance_id,
            "room": room,
            "identity": identity,
            "sample_rate": int(sample_rate),
            "duration_s": round(float(duration_s), 3),
            "rms_mean": round(float(rms_mean), 4),
            "rms_peak": round(float(rms_peak), 4),
            "voiced_ratio": round(float(voiced_ratio), 3),
            "ww_ran": bool(ww_ran),
            "ww_name": ww_name or None,
            "ww_score": round(float(ww_score), 4),
            "ww_threshold": round(float(ww_threshold), 4),
            "ww_fired": bool(ww_fired),
            "in_continuation": bool(in_continuation),
            "stt_text": (stt_text or "")[:200],
            "stt_latency_s": round(float(stt_latency_s), 3),
            "audio_path": str(audio_path) if audio_path else None,
            "client_info": client_info,
        }
        try:
            with self.SCORES_PATH.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError as e:
            LOG.warning("ww-diag: scores.jsonl write failed: %s", e)

    def _save_utterance_wav(self, utterance_id: str, samples: np.ndarray,
                             sample_rate: int) -> Path:
        path = self.UTTERANCES_DIR / f"{utterance_id}.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(samples.astype(np.int16, copy=False).tobytes())
        files = sorted(self.UTTERANCES_DIR.glob("*.wav"),
                       key=lambda p: p.stat().st_mtime)
        excess = len(files) - self.MAX_UTTERANCE_FILES
        for old in files[:max(0, excess)]:
            try:
                old.unlink()
            except OSError:
                pass
        return path

    # -- miss reports --

    def dump_miss(self, label: str, room: Optional[str],
                  identity: Optional[str]) -> dict:
        """Snapshot the rolling ring buffer + recent JSONL records to disk.
        If `room` is None or unknown, falls back to the most-active ring."""
        ts = time.time()
        chosen_room, ring = self._pick_ring(room)
        if ring is None:
            raise RuntimeError("no audio captured yet — start a LiveKit room first")
        sr = ring["sr"]
        audio = (
            np.concatenate(list(ring["buf"]))
            if ring["buf"] else np.zeros(0, dtype=np.int16)
        )
        safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", label or "miss")[:40] or "miss"
        stem = f"{int(ts)}_{safe_label}"
        wav_path = self.MISSES_DIR / f"{stem}.wav"
        json_path = self.MISSES_DIR / f"{stem}.json"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sr))
            wf.writeframes(audio.tobytes())
        recent = self._tail_recent(self.MISS_RECENT_WINDOW_S)
        meta = {
            "ts": ts,
            "label": safe_label,
            "room": chosen_room,
            "identity": identity,
            "ring_sample_rate": int(sr),
            "ring_duration_s": round(len(audio) / max(1, sr), 3),
            "ring_samples": int(len(audio)),
            "wav_path": str(wav_path),
            "recent_utterances": recent,
        }
        json_path.write_text(json.dumps(meta, indent=2))
        return {
            "wav": str(wav_path),
            "json": str(json_path),
            "room": chosen_room,
            "ring_duration_s": meta["ring_duration_s"],
            "recent_count": len(recent),
        }

    def _pick_ring(self, room: Optional[str]):
        if not self._rings:
            return None, None
        if room and room in self._rings:
            return room, self._rings[room]
        chosen = max(self._rings.items(), key=lambda kv: kv[1].get("total", 0))
        return chosen[0], chosen[1]

    def _tail_recent(self, window_s: float) -> list[dict]:
        cutoff = time.time() - window_s
        out: list[dict] = []
        if not self.SCORES_PATH.exists():
            return out
        try:
            with self.SCORES_PATH.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if rec.get("ts", 0) >= cutoff:
                        out.append(rec)
        except OSError:
            pass
        return out

    # -- ground-truth labels --

    def record_label(self, *, utterance_id: Optional[str], miss_ts: Optional[float],
                     said_wake_word: bool, note: Optional[str]) -> dict:
        """Append a ground-truth label for an existing utterance or miss
        dump. Returns the resolved target (utterance_id and/or miss path)
        so the caller can confirm it landed on the right row."""
        if not utterance_id and miss_ts is None:
            raise ValueError("either utterance_id or miss_ts is required")
        resolved: dict = {"said_wake_word": bool(said_wake_word)}
        if utterance_id:
            wav = self.UTTERANCES_DIR / f"{utterance_id}.wav"
            resolved["utterance_id"] = utterance_id
            resolved["utterance_wav_exists"] = wav.exists()
        if miss_ts is not None:
            # Filenames are formed as `{int(ts)}_{label}.{wav,json}` — find
            # the closest match by integer ts prefix.
            prefix = str(int(miss_ts))
            matches = sorted(self.MISSES_DIR.glob(f"{prefix}_*.wav"))
            resolved["miss_ts"] = miss_ts
            resolved["miss_matches"] = [str(p) for p in matches]
        rec = {
            "ts": time.time(),
            "utterance_id": utterance_id or None,
            "miss_ts": miss_ts,
            "said_wake_word": bool(said_wake_word),
            "note": (note or "")[:300] or None,
        }
        try:
            with self.LABELS_PATH.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError as e:
            LOG.warning("ww-diag: labels.jsonl write failed: %s", e)
            raise
        return resolved


async def _start_ww_diag_server(capture: WakeMissCapture,
                                 host: str = "127.0.0.1", port: int = 8501):
    """Start a tiny aiohttp server exposing /ww_miss and /healthz on
    localhost. The lloyd backend proxies /api/voice/ww_miss here. Returns
    the AppRunner so callers can clean up on shutdown."""
    from aiohttp import web

    async def healthz(_req):
        return web.json_response({
            "ok": True,
            "rings": {r: {"sr": v["sr"], "samples": v["total"]}
                      for r, v in capture._rings.items()},
        })

    async def ww_miss(req):
        try:
            body = await req.json() if req.body_exists else {}
        except Exception:
            body = {}
        label = (body.get("label") or "miss").strip() or "miss"
        room = (body.get("room") or "").strip() or None
        identity = (body.get("identity") or "").strip() or None
        try:
            result = capture.dump_miss(label, room, identity)
        except Exception as e:
            LOG.warning("ww_miss: dump failed: %s", e)
            return web.json_response({"ok": False, "error": str(e)}, status=400)
        LOG.info(
            "ww_miss captured: label=%r room=%s ring=%.2fs recent=%d wav=%s",
            label, result["room"], result["ring_duration_s"],
            result["recent_count"], result["wav"],
        )
        return web.json_response({"ok": True, **result})

    async def ww_label(req):
        try:
            body = await req.json() if req.body_exists else {}
        except Exception:
            body = {}
        utterance_id = (body.get("utterance_id") or "").strip() or None
        miss_ts_raw = body.get("miss_ts")
        miss_ts: Optional[float]
        try:
            miss_ts = float(miss_ts_raw) if miss_ts_raw is not None else None
        except (TypeError, ValueError):
            return web.json_response(
                {"ok": False, "error": "miss_ts must be a number"}, status=400,
            )
        if "said_wake_word" not in body:
            return web.json_response(
                {"ok": False, "error": "said_wake_word (bool) is required"}, status=400,
            )
        said = bool(body.get("said_wake_word"))
        note = (body.get("note") or "").strip() or None
        try:
            resolved = capture.record_label(
                utterance_id=utterance_id, miss_ts=miss_ts,
                said_wake_word=said, note=note,
            )
        except ValueError as e:
            return web.json_response({"ok": False, "error": str(e)}, status=400)
        except Exception as e:
            LOG.warning("ww_label: record failed: %s", e)
            return web.json_response({"ok": False, "error": str(e)}, status=500)
        LOG.info("ww_label: utt=%s miss_ts=%s said=%s note=%r",
                 utterance_id, miss_ts, said, note)
        return web.json_response({"ok": True, "resolved": resolved})

    app = web.Application()
    app.router.add_get("/healthz", healthz)
    app.router.add_post("/ww_miss", ww_miss)
    app.router.add_post("/ww_label", ww_label)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    try:
        await site.start()
    except OSError as e:
        await runner.cleanup()
        LOG.warning("ww-diag HTTP could not bind %s:%d (%s) — capture rig disabled",
                    host, port, e)
        return None
    LOG.info("ww-diag HTTP listening on http://%s:%d", host, port)
    return runner


# ── STT ──────────────────────────────────────────────────────────────────

def _load_hotwords(path: str | Path) -> Optional[str]:
    """Read names from a markdown hotwords file (with optional YAML
    frontmatter) and return them as a single space-separated string for
    faster-whisper's `hotwords=` parameter, or None on any error.

    File format (matching the user's ~/obsidian/hotwords.md):
      ---
      title: Hotwords
      tags: [...]
      ---
      Lloyd
      Alan
      Lisa
      ...
    """
    p = Path(path).expanduser()
    if not p.exists():
        return None
    try:
        text = p.read_text()
    except Exception as e:
        LOG.warning("hotwords: failed to read %s: %s", p, e)
        return None
    # Strip YAML frontmatter if present.
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end >= 0:
            text = text[end + 4:]
    names = []
    for line in text.splitlines():
        line = line.strip()
        # Skip blanks, comments, and accidental markdown bullets.
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        names.append(line)
    if not names:
        return None
    return " ".join(names)


class WhisperSTT:
    """Lazy faster-whisper wrapper. Loads on first transcribe call so the
    worker can advertise itself as ready before the model finishes downloading."""

    def __init__(self, stt_cfg: dict) -> None:
        self.cfg = stt_cfg
        self._model = None
        self._lock = asyncio.Lock()
        # Optional name biasing. faster-whisper's `hotwords=` parameter
        # nudges the decoder toward the listed tokens — important for our
        # wake-word detection because Whisper-tiny/base regularly mishear
        # "Lloyd" as "Floyd", "Eloid", or "Alloyed" when said quietly.
        hotwords_path = stt_cfg.get("hotwords_file")
        self.hotwords: Optional[str] = (
            _load_hotwords(hotwords_path) if hotwords_path else None
        )
        if self.hotwords:
            LOG.info("STT hotwords loaded: %r", self.hotwords)

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
            kwargs = dict(
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
            if self.hotwords:
                kwargs["hotwords"] = self.hotwords
            segments, _info = self._model.transcribe(buf, **kwargs)
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

    def __init__(self, vad_cfg: dict, sample_rate: int, room_name: str = "") -> None:
        self.sample_rate = sample_rate
        self.room_name = room_name
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
            dur_s = utterance.size / max(1, self.sample_rate)
            if utterance.size < self._min_samples:
                LOG.info(
                    "[%s][diag-drop] reason=too_short dur=%.2fs min=%.2fs voiced_samples=%d",
                    self.room_name or "?", dur_s, self._min_samples / max(1, self.sample_rate), voiced,
                )
                return None
            voicing_window = max(1, buf_count - self._lead_in_samples - self._silence_samples)
            voiced_ratio = voiced / voicing_window
            if voiced_ratio < self.min_voiced_ratio:
                LOG.info(
                    "[%s][diag-drop] reason=low_voiced dur=%.2fs voiced_ratio=%.2f min=%.2f",
                    self.room_name or "?", dur_s, voiced_ratio, self.min_voiced_ratio,
                )
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

    # How often the room polls /api/messages/<session> for new assistant
    # turns to TTS. 500ms matches the page poll cadence; tightening to
    # 250ms gives marginally lower mouth-latency at minor server load.
    SESSION_POLL_INTERVAL = 0.5
    MESSAGES_URL_TEMPLATE = "http://127.0.0.1:8080/api/messages/{session_id}"

    def __init__(self, room_name: str, lk_cfg: dict, stt: WhisperSTT,
                 vad_cfg: dict, http_client, speaker_id=None,
                 acoustic_wake=None, wake_capture: Optional[WakeMissCapture] = None) -> None:
        self.room_name = room_name
        self.lk_cfg = lk_cfg
        self.stt = stt
        self.vad_cfg = vad_cfg
        self.http = http_client
        self.wake_capture = wake_capture
        # identity -> client_info dict (browser UA, mobile flag, audio
        # constraints). Populated by `client_info` data-channel messages
        # from VoiceRoom on connect. Used by ww-diag to A/B browser DSP
        # configurations.
        self._client_meta: dict[str, dict] = {}
        # Optional SpeakerIdentifier (resemblyzer). When None, the wake-word
        # gate falls back to LiveKit-identity-only matching for continuation.
        self.speaker_id = speaker_id
        vp_cfg = lk_cfg.get("voiceprint", {}) or {}
        # Cosine threshold for "is this still the same speaker as the
        # wake-word utterance". Separate from the profile threshold —
        # anchor matching is an easier task than full identification.
        self.anchor_threshold = float(vp_cfg.get("anchor_threshold", 0.65))
        # Optional AcousticWakeWord — parallel detection path that catches
        # wake-words even when Whisper transcribes them as "Eloid" etc.
        self.acoustic_wake = acoustic_wake
        self.room = rtc.Room()
        self._tasks: list[asyncio.Task] = []
        # Strong refs to in-flight utterance handlers. asyncio's create_task
        # docs: "Save a reference … to avoid a task disappearing mid-execution"
        # — without this, the task can be GC'd between the transcribe log
        # line and the inject POST, dropping transcripts on the floor.
        self._utterance_tasks: set[asyncio.Task] = set()
        # Per-participant audio consumer tasks, so we can cancel a specific
        # one when its participant leaves (otherwise stale streams keep
        # producing duplicate transcripts).
        self._audio_tasks: dict[str, asyncio.Task] = {}
        # TTS pipeline (lazy-initialized on connect).
        self._tts_cfg = lk_cfg.get("tts", {}) or {}
        self.tts: Optional[TTSStreamer] = None
        # Wake-word gate. Per-room so multi-room workers don't share state.
        self.wake = WakeState(lk_cfg.get("wake", {}) or {})
        # Track which assistant message ids have already been TTS'd so the
        # session poller doesn't speak the same reply twice. Seeded at
        # connect time with the entire existing history so we only speak
        # newly-arrived turns.
        self._spoken_ids: set[str] = set()
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
        self.room.on("data_received", self._on_data_received)

        LOG.info("[%s] connecting (session_id=%s)", self.room_name, self.session_id)
        await self.room.connect(self.lk_cfg["url"], token)
        LOG.info("[%s] connected as %s", self.room_name, self.room.local_participant.identity)

        # TTS pipeline + session poller — only spin them up after the room
        # connection is alive so the published track has a parent.
        self.tts = TTSStreamer(
            self._tts_cfg,
            self.room,
            on_utterance_end=self._on_tts_utterance_end,
        )
        await self.tts.ensure_published()
        # Seed the spoken-set with all existing assistant ids so we don't
        # re-speak history when the worker reconnects to an existing room.
        await self._seed_spoken_set()
        poll_task = asyncio.create_task(self._poll_session_messages())
        self._tasks.append(poll_task)

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
        if self.tts is not None:
            await self.tts.close()
            self.tts = None
        await self.room.disconnect()
        if self.wake_capture is not None:
            self.wake_capture.drop_room(self.room_name)
        LOG.info("[%s] disconnected", self.room_name)

    def _on_tts_utterance_end(self) -> None:
        """Called by TTSStreamer when an utterance finishes draining. Extends
        the wake-word continuation window so the user has continuation_seconds
        to follow up after Lloyd stops speaking, without re-saying the
        wake-word.

        Uses `has_lock` rather than `locked_identity` so this still fires
        when TTS playback outlasts the continuation window — without that,
        a long agent reply (10s+ TTS, 6s window) would let the window
        expire mid-speech and we'd silently skip the post-TTS extension.
        """
        if not self.wake.enabled:
            return
        if not self.wake.has_lock:
            return
        was_in_continuation = self.wake.in_continuation()
        self.wake.extend()  # keep identity, refresh timer
        LOG.info(
            "[%s] TTS done — continuation %s to %.1fs",
            self.room_name,
            "extended" if was_in_continuation else "reopened",
            self.wake.continuation_seconds,
        )
        self._schedule_wake_state_publish()

    def _schedule_wake_state_publish(self) -> None:
        """Fire-and-forget data-channel publish of the current wake state.
        Called from sync contexts (TTS callback, gate code that may or may
        not be inside an asyncio task). Schedules `_publish_wake_state`
        without awaiting it."""
        try:
            asyncio.create_task(self._publish_wake_state())
        except RuntimeError:
            # No running event loop (e.g. called too early). Best-effort.
            pass

    async def _publish_wake_state(self) -> None:
        """Broadcast the current wake state to all browser participants on
        the LiveKit data channel. Browsers tick down `remaining_s` locally
        and flip back to 'idle' when it reaches 0, so we don't need to
        also publish on window-expiry — the wake state stays in sync as
        long as we publish on every extend/lock."""
        try:
            wake = self.wake
            payload = {
                "type": "wake_state",
                "state": "listening" if wake.in_continuation() else "idle",
                "remaining_s": round(wake.remaining_s(), 2),
                "continuation_s": wake.continuation_seconds,
                "speaker": wake.anchor_name,  # None when no enrolled match
                "ts": time.time(),
            }
            data = json.dumps(payload).encode("utf-8")
            await self.room.local_participant.publish_data(data, reliable=True)
        except Exception as e:
            LOG.debug("[%s] publish_wake_state failed: %s", self.room_name, e)

    async def _seed_spoken_set(self) -> None:
        url = self.MESSAGES_URL_TEMPLATE.format(session_id=self.session_id)
        try:
            r = await self.http.get(url, timeout=5.0)
            if r.status_code != 200:
                return
            for m in (r.json().get("messages") or []):
                if m.get("role") == "assistant":
                    mid = m.get("id")
                    if mid:
                        self._spoken_ids.add(mid)
            LOG.info("[%s] seeded spoken set with %d existing assistant turns",
                     self.room_name, len(self._spoken_ids))
        except Exception as e:
            LOG.warning("[%s] could not seed spoken set: %s", self.room_name, e)

    async def _poll_session_messages(self) -> None:
        """Watch the session JSON for new assistant turns and TTS them.

        Each candidate assistant message goes through the secondary model
        first (POST /api/voice/summarize) so what we speak is a tight
        spoken-form summary, not the raw primary response (which is often
        long and contains code/markdown that doesn't TTS gracefully).
        Falls back to the raw text if the summary call fails.

        Skips: subliminal / tool messages, empty assistant rows (harness
        tool-call frames), anything already spoken (tracked by id).
        """
        url = self.MESSAGES_URL_TEMPLATE.format(session_id=self.session_id)
        skip_empty = bool(self._tts_cfg.get("skip_empty", True))
        try:
            while True:
                try:
                    r = await self.http.get(url, timeout=5.0)
                    if r.status_code == 200:
                        for m in (r.json().get("messages") or []):
                            if m.get("role") != "assistant":
                                continue
                            mid = m.get("id")
                            if not mid or mid in self._spoken_ids:
                                continue
                            text = "".join(
                                c.get("text", "") for c in (m.get("content") or [])
                                if c.get("type") == "text"
                            ).strip()
                            self._spoken_ids.add(mid)
                            if skip_empty and not text:
                                continue
                            if self.tts is not None:
                                spoken = await self._summarize_for_tts(text)
                                await self.tts.speak(spoken)
                except Exception as e:
                    LOG.warning("[%s] session poll failed: %s", self.room_name, e)
                await asyncio.sleep(self.SESSION_POLL_INTERVAL)
        except asyncio.CancelledError:
            return

    async def _summarize_for_tts(self, text: str) -> str:
        """POST text to /api/voice/summarize, return the spoken-form rewrite.
        Falls back to the raw text if the summary call fails or returns
        used_summary=false. Logs which path was taken so the worker output
        makes the routing clear."""
        try:
            r = await self.http.post(SUMMARIZE_URL, json={"text": text}, timeout=20.0)
            if r.status_code != 200:
                LOG.warning("[%s] summarize HTTP %d: %s", self.room_name, r.status_code, r.text[:200])
                return text
            payload = r.json()
            summary = (payload.get("summary") or "").strip()
            used = bool(payload.get("used_summary"))
            if used and summary:
                LOG.info("[%s] summary %d→%d chars", self.room_name, len(text), len(summary))
                return summary
            LOG.info("[%s] summary fell back to raw (%d chars)", self.room_name, len(text))
            return summary or text
        except Exception as e:
            LOG.warning("[%s] summarize failed, speaking raw: %s", self.room_name, e)
            return text

    def _on_track_subscribed(self, track, publication, participant) -> None:  # noqa: ARG002
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        identity = participant.identity
        # Cancel any prior consumer for this identity (e.g. participant
        # rejoining with the same id after a brief disconnect).
        prior = self._audio_tasks.pop(identity, None)
        if prior is not None and not prior.done():
            prior.cancel()
        LOG.info("[%s] subscribed to audio from %s", self.room_name, identity)
        task = asyncio.create_task(self._consume_audio(track, identity))
        self._tasks.append(task)
        self._audio_tasks[identity] = task

    def _on_participant_connected(self, participant) -> None:
        LOG.info("[%s] participant joined: %s", self.room_name, participant.identity)
        # Push the current wake state so the freshly-joined browser doesn't
        # have to wait for the next utterance to learn whether we're idle
        # or already in continuation.
        self._schedule_wake_state_publish()

    def _on_participant_disconnected(self, participant) -> None:
        identity = participant.identity
        LOG.info("[%s] participant left: %s", self.room_name, identity)
        # Cancel the per-participant audio consumer so a stale stream can't
        # keep producing duplicate transcripts after the participant is gone.
        task = self._audio_tasks.pop(identity, None)
        if task is not None and not task.done():
            task.cancel()

    def _on_data_received(self, packet) -> None:
        """Handle JSON control messages from a participant via the LiveKit
        data channel. Understands {"type": "interrupt"} and
        {"type": "client_info", ...}."""
        try:
            payload = packet.data.decode("utf-8")
            msg = json.loads(payload) if payload else {}
        except Exception as e:
            LOG.warning("[%s] bad data packet: %s", self.room_name, e)
            return
        kind = msg.get("type")
        if kind == "interrupt":
            if self.tts is None:
                return
            dropped = self.tts.interrupt()
            LOG.info("[%s] interrupt: dropped %d queued utterance(s)",
                     self.room_name, dropped)
        elif kind == "client_info":
            # Sender identity comes from the LiveKit packet's participant
            # field; fall back to None if the SDK version doesn't expose it.
            sender = getattr(packet, "participant", None)
            identity = getattr(sender, "identity", None) if sender else None
            if not identity:
                # Some SDK versions put it on packet directly.
                identity = getattr(packet, "participant_identity", None)
            if identity:
                info = {k: v for k, v in msg.items() if k != "type"}
                self._client_meta[identity] = info
                LOG.info(
                    "[%s] client_info from %s: mobile=%s raw_audio=%s ns=%s aec=%s agc=%s sr=%s",
                    self.room_name, identity, info.get("isMobile"),
                    info.get("rawAudio"),
                    (info.get("trackSettings") or {}).get("noiseSuppression"),
                    (info.get("trackSettings") or {}).get("echoCancellation"),
                    (info.get("trackSettings") or {}).get("autoGainControl"),
                    (info.get("trackSettings") or {}).get("sampleRate"),
                )
            else:
                LOG.warning("[%s] client_info dropped (no identity on packet)",
                            self.room_name)
        else:
            LOG.info("[%s] data message ignored: %r", self.room_name, kind)

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
                # Diagnostic: feed raw mono frames into the rolling ring
                # buffer so /ww_miss can recover audio even when VAD never
                # segmented it (the silent failure mode).
                if self.wake_capture is not None:
                    try:
                        self.wake_capture.push_raw_frame(self.room_name, samples, frame.sample_rate)
                    except Exception as e:
                        LOG.debug("[%s] ww-diag ring push failed: %s", self.room_name, e)
                if segmenter is None:
                    segmenter = UtteranceSegmenter(self.vad_cfg, frame.sample_rate, self.room_name)
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

    async def _embed_async(self, samples: np.ndarray, sample_rate: int):
        """Run resemblyzer's blocking embed in a thread so the event loop
        stays responsive. Returns (embedding, name, score) on success or
        (None, unknown_label, 0.0) on failure.

        Note: SpeakerIdentifier.identify() itself returns (name, score, emb)
        for legacy compatibility — we re-tuple it here so callers can
        consistently unpack `emb, name, score = ...` (with the embedding
        first because that's what the gate cares about most)."""
        if self.speaker_id is None:
            return None, "Unknown", 0.0
        loop = asyncio.get_running_loop()
        try:
            name, score, emb = await loop.run_in_executor(
                None, self.speaker_id.identify, samples, sample_rate,
            )
            return emb, name, score
        except Exception as e:
            LOG.warning("[%s] voiceprint embed failed: %s", self.room_name, e)
            return None, "Unknown", 0.0

    async def _handle_utterance(self, samples: np.ndarray, sample_rate: int, identity: str) -> None:
        duration_s = samples.size / sample_rate
        utterance_id = uuid.uuid4().hex[:12]

        # Diagnostic stats: mean/peak RMS over the whole utterance and a
        # voiced-ratio computed in 10ms windows (mirrors the VAD's frame
        # view). Cheap O(n) numpy ops on a few hundred ms of audio.
        f32 = samples.astype(np.float32) / 32768.0
        if f32.size:
            rms_mean = float(np.sqrt(np.mean(f32 * f32)))
            rms_peak = float(np.max(np.abs(f32)))
            chunk = max(1, sample_rate // 100)
            n_chunks = f32.size // chunk
            if n_chunks > 0:
                trimmed = f32[: n_chunks * chunk].reshape(n_chunks, chunk)
                chunk_rms = np.sqrt(np.mean(trimmed * trimmed, axis=1))
                voiced_ratio = float(np.mean(chunk_rms >= float(self.vad_cfg.get("speech_rms", 0.025))))
            else:
                voiced_ratio = 0.0
        else:
            rms_mean = rms_peak = voiced_ratio = 0.0

        utterance_start_t = time.monotonic() - duration_s
        wake = self.wake
        in_continuation = wake.enabled and wake.matches_lock(identity, at=utterance_start_t)

        stt_task = asyncio.create_task(self.stt.transcribe(samples, sample_rate))
        t0 = time.monotonic()

        def _record_diag(text: str, latency: float) -> None:
            cap = self.wake_capture
            if cap is None:
                return
            try:
                cap.record_utterance(
                    utterance_id=utterance_id,
                    room=self.room_name,
                    identity=identity,
                    duration_s=duration_s,
                    rms_mean=rms_mean,
                    rms_peak=rms_peak,
                    voiced_ratio=voiced_ratio,
                    ww_ran=ww_ran,
                    ww_name=ww_name,
                    ww_score=ww_score,
                    ww_threshold=(self.acoustic_wake.threshold
                                  if self.acoustic_wake is not None else 0.0),
                    ww_fired=ww_already_fired,
                    in_continuation=in_continuation,
                    stt_text=text,
                    stt_latency_s=latency,
                    samples=samples,
                    sample_rate=sample_rate,
                    client_info=self._client_meta.get(identity),
                )
            except Exception as e:
                LOG.debug("[%s] ww-diag record failed: %s", self.room_name, e)

        # ── IDLE state: openWakeWord first, publish ASAP ────────────
        ww_already_fired = False
        ww_name = ""
        ww_score = 0.0
        ww_ran = False
        if wake.enabled and not in_continuation and self.acoustic_wake is not None:
            ww_match, ww_name, ww_score = await self.acoustic_wake.detect(samples, sample_rate)
            ww_ran = True
            if ww_match:
                # Run embedding + state-publish before waiting for Whisper.
                # This is the latency-critical path — the UI flips to
                # "Listening" off this publish, so cutting Whisper out of
                # the wait saves ~400ms of perceived wake-word latency.
                emb, name, score = await self._embed_async(samples, sample_rate)
                if emb is not None and self.speaker_id is not None:
                    LOG.info("[%s] wake-word speaker: %s (cos=%.2f)",
                             self.room_name, name, score)
                wake.set_anchor(emb, name if name and name != "Unknown" else None)
                wake.extend(identity)
                await self._publish_wake_state()
                ww_already_fired = True
                LOG.info("[%s] wake-word matched (acoustic=%s/%.2f) — UI notified, awaiting STT",
                         self.room_name, ww_name, ww_score)

        # Now collect Whisper's result.
        try:
            text = await stt_task
        except Exception as e:
            latency = time.monotonic() - t0
            LOG.warning(
                "[%s][diag] STT_FAIL dur=%.2fs rms_mean=%.3f rms_peak=%.3f voiced=%.2f ww=%s err=%s",
                self.room_name, duration_s, rms_mean, rms_peak, voiced_ratio,
                f"{ww_name or '-'}:{ww_score:.2f}" if ww_ran else "skipped",
                e,
            )
            _record_diag("", latency)
            return
        latency = time.monotonic() - t0
        LOG.info(
            "[%s][diag] dur=%.2fs rms_mean=%.3f rms_peak=%.3f voiced=%.2f ww=%s stt_lat=%.2fs cont=%s text=%r",
            self.room_name, duration_s, rms_mean, rms_peak, voiced_ratio,
            f"{ww_name or '-'}:{ww_score:.2f}" if ww_ran else "skipped",
            latency, "Y" if in_continuation else "N", (text or "")[:80],
        )
        _record_diag(text or "", latency)
        if not text:
            # Empty transcript. If acoustic wake-word already fired, treat as
            # bare wake-word (window already opened above). Otherwise drop.
            if ww_already_fired:
                LOG.info(
                    "[%s] bare wake-word — opening %.1fs window (empty Whisper)",
                    self.room_name, wake.continuation_seconds,
                )
                return
            LOG.info("[%s] empty transcript for %.2fs utterance from %s (%.1fs whisper)",
                     self.room_name, duration_s, identity, latency)
            return
        LOG.info("[%s] %s → %r  (%.2fs audio, %.1fs whisper)",
                 self.room_name, identity, text, duration_s, latency)

        # ── Gate decisions ──────────────────────────────────────────
        inject_text: Optional[str] = None
        speaker_name: Optional[str] = None  # populated from anchor or fresh ID
        if not wake.enabled:
            inject_text = text
        elif in_continuation:
            # Identity matches the locked participant. If voiceprint is
            # enabled AND the wake-word utterance was identified as a known
            # speaker, also require the embedding to match the anchor —
            # this is what catches "different person, same browser tab".
            #
            # When the wake-word came back as Unknown (no enrolled profile),
            # the anchor is just a noisy 1s embedding; comparing against it
            # rejects real follow-ups without providing meaningful safety.
            # In that case we fall back to identity-only matching.
            anchor = wake.anchor_embedding
            anchor_named = wake.anchor_name is not None
            if anchor is not None and anchor_named:
                emb, _name, _score = await self._embed_async(samples, sample_rate)
                if emb is None:
                    # Embedding failed — degrade to identity-only this turn
                    # rather than dropping a real utterance.
                    LOG.info("[%s] continuation: voiceprint check skipped (embed failed)",
                             self.room_name)
                else:
                    sim = float(np.dot(emb, anchor))
                    if sim < self.anchor_threshold:
                        LOG.info(
                            "[%s] voiceprint anchor mismatch (cos=%.2f < %.2f) — dropped %r",
                            self.room_name, sim, self.anchor_threshold, text[:80],
                        )
                        return
                    LOG.info("[%s] voiceprint anchor match (cos=%.2f)", self.room_name, sim)
            inject_text = text
            speaker_name = wake.anchor_name
            wake.extend(identity)  # extend on each turn the user takes
            await self._publish_wake_state()
            LOG.info("[%s] continuation pass-through (%.1fs left)",
                     self.room_name, wake.remaining_s())
        else:
            # IDLE state: openWakeWord decision was already made above
            # (parallel with Whisper). If it didn't fire, drop. If it did,
            # the anchor + state publish already happened — we just need
            # to decide what (if anything) to inject from the transcript.
            if not ww_already_fired:
                if self.acoustic_wake is None:
                    LOG.warning("[%s] acoustic wake-word not available — dropping %r",
                                self.room_name, text[:80])
                else:
                    LOG.info(
                        "[%s] no wake-word (acoustic=%s/%.2f) in %r — dropped",
                        self.room_name, ww_name or "?", ww_score, text[:80],
                    )
                return
            # Wake-word already detected & state published. Just clean up
            # the transcript for injection.
            speaker_name = wake.anchor_name
            tail = _strip_wake_word(text, wake.words)
            word_count = len([w for w in text.split() if w.strip(".,!?;:'\"")])
            if tail is not None:
                if not tail and wake.skip_inject_if_only_wake_word:
                    LOG.info(
                        "[%s] bare wake-word (%s/%.2f) — opening %.1fs window, no inject",
                        self.room_name, ww_name, ww_score, wake.continuation_seconds,
                    )
                    return
                inject_text = tail or text
            elif word_count <= 2:
                # Short transcript with no wake-word match: probably a
                # mistranscribed bare wake-word (Whisper heard "Eloid"
                # alone). Open the window and wait for the follow-up.
                LOG.info(
                    "[%s] bare wake-word inferred from short transcript %r (%s/%.2f) — opening %.1fs window",
                    self.room_name, text[:40], ww_name, ww_score, wake.continuation_seconds,
                )
                return
            else:
                # Full transcript with mistranscribed wake-word. Inject as-is.
                inject_text = text
            LOG.info(
                "[%s] wake-word injecting %r (acoustic=%s/%.2f)",
                self.room_name, inject_text[:80], ww_name, ww_score,
            )

        if not inject_text:
            return
        payload = {"text": inject_text, "session_key": self.session_id}
        if speaker_name:
            payload["speaker"] = speaker_name
        try:
            r = await self.http.post(INJECT_URL, json=payload, timeout=10.0)
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
        # SpeakerIdentifier is shared across rooms — one VoiceEncoder load,
        # one profiles_dir watcher, single source of truth. None when
        # voiceprint matching is disabled in config.
        self.speaker_id = _build_speaker_id(self.lk_cfg.get("voiceprint", {}) or {})
        # AcousticWakeWord runs in parallel with text-match wake detection.
        # Critical because Whisper is unreliable at transcribing the brief
        # wake-word phrase ("Eloid"/"Floyd"/"Alloyed" mishears). openWakeWord
        # detects directly on the audio. Shared across rooms, lazy-loaded.
        self.acoustic_wake = _build_acoustic_wake_word(
            self.lk_cfg.get("acoustic_wake", {}) or {}
        )
        # Wake-word miss capture rig (Phase A). Always-on by default; gate
        # via livekit.acoustic_wake.diag.{enabled,host,port}. The aiohttp
        # listener is started in run() so the cleanup hook has a runner.
        diag_cfg = (self.lk_cfg.get("acoustic_wake", {}) or {}).get("diag", {}) or {}
        self._diag_enabled = bool(diag_cfg.get("enabled", True))
        self._diag_host = str(diag_cfg.get("host", "127.0.0.1"))
        self._diag_port = int(diag_cfg.get("port", 8501))
        self.wake_capture: Optional[WakeMissCapture] = (
            WakeMissCapture() if self._diag_enabled else None
        )
        self._diag_runner = None
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
        if self.wake_capture is not None:
            self._diag_runner = await _start_ww_diag_server(
                self.wake_capture, host=self._diag_host, port=self._diag_port,
            )
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
            bridge = RoomBridge(
                r.name, self.lk_cfg, self.stt, self.vad_cfg, self._http,
                speaker_id=self.speaker_id,
                acoustic_wake=self.acoustic_wake,
                wake_capture=self.wake_capture,
            )
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
        if self._diag_runner is not None:
            try:
                await self._diag_runner.cleanup()
            except Exception:
                pass
            self._diag_runner = None
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
