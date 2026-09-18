"""Lloyd LiveKit agent worker — the voice round trip.

Joins every `${room_prefix}*` room that has a participant, listens, injects
what was addressed to Lloyd into the matching chat session as a user turn, and
speaks the reply back on a published `lloyd-tts` track.

The hearing half lives in `agent-services/voice/` and is driven from here:

    frames -> voice.runner.HearingThread -> voice.pipeline.HearingPipeline
                                              StreamResampler     (one resample)
                                              ContinuousWakeWord  (continuous)
                                              SileroSegmenter     (speech, not energy)
              -> HearingEvent -> RoomBridge._on_hearing_event
              -> voice.turn.SmartTurn (finished thought?) -> ASR -> gate -> inject

That structure is the 2026-09-17 rework. The old pipeline was inline in this
file and could not be constructed without a LiveKit room, which is part of why
three weeks of it firing the wake word 5 times in 949 utterances went
unnoticed. Everything below the event boundary is now replayable offline —
`scripts/voice/replay.py` is what replays it.

Run via:
  python agent-services/livekit_worker.py

Or under supervisord (agent-services/supervisor/conf.d/lloyd-agent-worker.conf).

Config — see config.yaml `livekit:` block:
  livekit.url, .api_key, .api_secret, .room_prefix, .agent_identity
  livekit.stt.{backend, model_dir, ...}                        — voice/asr.py
  livekit.vad.{threshold, min_silence_ms, speech_pad_ms, ...}  — voice/vad.py
  livekit.acoustic_wake.{threshold, refractory_ms, ...}        — voice/wake.py
  livekit.turn_detection.{enabled, threshold, ...}             — voice/turn.py
"""
from __future__ import annotations

import asyncio
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
import tts_shaping
import yaml
from livekit import api as lkapi
from livekit import rtc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from voice import asr as voice_asr  # noqa: E402
from voice import turn as voice_turn  # noqa: E402
from voice import vad as voice_vad  # noqa: E402
from voice import wake as voice_wake  # noqa: E402
from voice.pipeline import HearingEvent, HearingPipeline  # noqa: E402
from voice.runner import HearingThread  # noqa: E402
from voice.resample import to_int16  # noqa: E402
from voice.speakable import ClauseStream  # noqa: E402


LOG = logging.getLogger("lloyd-agent-worker")

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"
ENV_PATH = REPO_ROOT / ".env"
#: The backend this worker speaks for. Overridable so a second worker can run
#: beside the live one against a canary backend — which is how
#: `scripts/voice/e2e_voice.py` tests a build end to end without touching the
#: live session store (pair it with LLOYD_LIVEKIT_ROOM_PREFIX, or both workers
#: join the same room and answer twice).
BACKEND_URL = os.environ.get("LLOYD_BACKEND_URL", "http://127.0.0.1:8080").rstrip("/")
INJECT_URL = f"{BACKEND_URL}/api/voice/inject"
SUMMARIZE_URL = f"{BACKEND_URL}/api/voice/summarize"
PREWARM_URL = f"{BACKEND_URL}/api/voice/prewarm"

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
    if os.environ.get("LLOYD_LIVEKIT_ROOM_PREFIX"):
        lk["room_prefix"] = os.environ["LLOYD_LIVEKIT_ROOM_PREFIX"]
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
        self.tail_silence_ms = int(tts_cfg.get("tail_silence_ms", 250))
        # Output shaping — restores the presence band the 12 Hz vocoder drops,
        # and applies `speed`, which the server's *streaming* path silently
        # ignores. See agent-services/tts_shaping.py for both measurements.
        shaping = tts_cfg.get("shaping") or {}
        self._shaper = tts_shaping.OutputShaper(
            self.sample_rate,
            speed=self.speed,
            presence_eq=bool(shaping.get("presence_eq", True)),
            shelves=shaping.get("shelves"),
        )
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
        #: Bumped by `interrupt()`. A clause queued before an interrupt carries
        #: the old number and is dropped, and a streaming reply compares it to
        #: know the listener has cut it off.
        self.generation = 0
        #: Characters synthesised since the queue last drained — the length
        #: of what the listener has been told in this reply.
        self.spoken_chars = 0
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
            LOG.info(
                "TTSStreamer published track 'lloyd-tts' @ %d Hz — voice=%s shaping=%s",
                self.sample_rate, self.voice, self._shaper.describe(),
            )

    async def speak(self, text: str) -> None:
        """Queue an utterance for synthesis + playback."""
        text = (text or "").strip()
        if not text:
            return
        await self.ensure_published()
        await self._queue.put((self.generation, text))

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
        self.generation += 1
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
        """Speak queued clauses back to back.

        A streamed reply arrives as many clauses, and the gap between them is
        what the listener hears. So a clause is followed by the playout wait,
        the tail silence and the end-of-utterance callback only when nothing
        else is queued — and even then the wait gives way the moment another
        clause arrives. Waiting for playout between clauses would put a full
        synthesis latency (~250 ms) of dead air after every sentence.
        """
        while True:
            gen, text = await self._queue.get()
            if gen != self.generation:
                continue  # queued before an interrupt
            self._speaking.set()
            self._current_task = asyncio.create_task(self._stream_utterance(text))
            try:
                await self._current_task
                self.spoken_chars += len(text)
            except asyncio.CancelledError:
                LOG.info("TTS interrupted mid-utterance")
            except Exception as e:
                LOG.warning("TTS error for %r: %s", text[:60], e)
            finally:
                self._current_task = None
            if not self._queue.empty():
                continue
            # Nothing queued: close the reply out. Tail silence first, so the
            # last syllable is not the thing `clear_queue()` discards.
            try:
                await self._push_tail_silence()
                if await self._await_playout_or_next():
                    continue  # another clause arrived while this one played
            except Exception as e:
                LOG.debug("TTS close-out failed: %s", e)
            self._speaking.clear()
            self.spoken_chars = 0
            # Notify the bridge that the reply finished playing so it can
            # extend the wake-word continuation window. Best-effort: any
            # callback exception is swallowed, the drain loop survives.
            cb = self.on_utterance_end
            if cb is not None:
                try:
                    cb()
                except Exception as e:
                    LOG.warning("on_utterance_end callback raised: %s", e)

    async def _await_playout_or_next(self) -> bool:
        """Wait for queued audio to play, or for a new clause. True if a
        clause arrived first — the reply is still going."""
        playout = asyncio.create_task(self._await_playout())
        waiter = asyncio.create_task(self._peek_queue())
        try:
            done, _ = await asyncio.wait({playout, waiter},
                                         return_when=asyncio.FIRST_COMPLETED)
            return waiter in done and not self._queue.empty()
        finally:
            for t in (playout, waiter):
                if not t.done():
                    t.cancel()

    async def _peek_queue(self) -> None:
        while self._queue.empty():
            await asyncio.sleep(0.02)

    async def _push_tail_silence(self) -> None:
        if self.tail_silence_ms <= 0 or self.source is None:
            return
        samples = self.sample_rate * self.tail_silence_ms // 1000
        samples -= samples % (self.sample_rate // 100)
        if samples > 0:
            await self._push_frame(bytes(2 * samples), samples)

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
        drained = False
        try:
            async with self._http.stream(
                "POST",
                f"{self.api_url}/v1/audio/speech",
                json={
                    "model": self.model,
                    "input": text,
                    "voice": self.voice,
                    "response_format": "pcm",
                    "stream": True,
                    # Always 1.0 on the wire. The server drops `speed` on its
                    # streaming path, so we apply it here; asking for it in both
                    # places would stretch twice the day the server grows support.
                    "speed": 1.0,
                },
            ) as resp:
                if resp.status_code >= 300:
                    err = await resp.aread()
                    LOG.warning("TTS HTTP %d: %s", resp.status_code, err[:200])
                    return
                async for chunk in resp.aiter_bytes():
                    if not chunk:
                        continue
                    leftover.extend(self._shaper.process(chunk))
                    # Push complete 100ms frames; keep any tail for the next loop.
                    while len(leftover) >= bytes_per_frame:
                        frame_bytes = bytes(leftover[:bytes_per_frame])
                        del leftover[:bytes_per_frame]
                        await self._push_frame(frame_bytes, samples_per_frame)
                        n_pushed += 1
            leftover.extend(self._shaper.flush())
            drained = True
        finally:
            # An interrupt cancels this coroutine mid-stream. The shaper carries
            # filter and overlap state across chunks, so it has to be emptied
            # here or the next utterance opens with the tail of the one the user
            # just talked over.
            if not drained:
                self._shaper.flush()
        # Trailing silence is `_drain`'s job now, and only at the end of a
        # reply: the last frames of an utterance were being cut off, because
        # this coroutine returns once the audio is *queued* and whatever runs
        # next — `interrupt()` calling `source.clear_queue()`, the room going
        # quiet — discards what has not played. Padding between clauses of one
        # reply would just be a pause in the middle of a sentence.
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
        # The presence shelves add ~1.5 dB of peak. Measured output sits at 0.75
        # full scale, so this should stay silent; if it ever fires the shelf
        # gains are too hot for whatever the server is now sending.
        if self._shaper.clipped and self._shaper.total:
            ratio = self._shaper.clipped / self._shaper.total
            if ratio > 0.001:
                LOG.warning(
                    "TTS shaping clipped %.2f%% of samples — lower livekit.tts.shaping gains",
                    ratio * 100,
                )
            self._shaper.clipped = self._shaper.total = 0

    async def _await_playout(self) -> None:
        """Block until queued audio has played, at most its own duration + 2 s."""
        src = self.source
        if src is None:
            return
        try:
            budget = float(getattr(src, "queued_duration", 0.0) or 0.0) + 2.0
            await asyncio.wait_for(src.wait_for_playout(), timeout=budget)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            LOG.warning("TTS playout wait timed out; continuing")
        except Exception as e:
            # Older SDKs may not expose wait_for_playout at all.
            LOG.debug("TTS playout wait unavailable: %s", e)

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
        #: Also wake on a transcript that STARTS with a wake phrase. See the
        #: config comment for the measurement; the short version is that the
        #: acoustic models catch well under half of clean wake phrases, and
        #: transcribing an idle utterance costs ~60 ms since Parakeet.
        self.text_fallback: bool = bool(cfg.get("text_fallback", True))
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
      'Uh, hey Lloyd, stop.'   → 'stop.'             (one leading filler)

    One leading disfluency is allowed, because a transcript is now also a
    wake path (`livekit.wake.text_fallback`) and people say "uh, hey Lloyd".
    Only one: "so I told Lloyd" must not wake anything.
    """
    import re
    normalized = re.sub(r"[^a-z']+", " ", text.lower()).strip()
    if not normalized:
        return None
    filler = ""
    first, _, rest = normalized.partition(" ")
    if first in _WAKE_FILLERS and rest:
        filler, normalized = first, rest
    for w in words:
        if normalized == w or normalized.startswith(w + " "):
            # Build a regex that matches the wake-word in the original text
            # tolerating any punctuation/whitespace between the word parts
            # and trailing the match. Anchored to the start.
            parts = ([filler] if filler else []) + w.split()
            pattern = r"^\W*" + r"\W+".join(re.escape(p) for p in parts) + r"[\s.,!?;:'\"]*"
            tail = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)
            return tail.strip()
    return None


#: Disfluencies allowed ahead of a wake phrase in a transcript.
_WAKE_FILLERS = frozenset({"uh", "um", "er", "erm", "oh", "ah", "so", "well", "and"})


#: Kept as a module-level name because it is what the transcript gate reads,
#: but the definition now lives beside the recognisers that produce the text —
#: two copies of "does this look hallucinated" is how the engines would come to
#: disagree about the same utterance.
_looks_repetitive = voice_asr.looks_repetitive


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


# ── Per-room bridge ──────────────────────────────────────────────────────

class RoomBridge:
    """One-room connection: subscribes to remote audio, segments utterances,
    transcribes them, and POSTs each transcript to /api/voice/inject."""

    # How often the room polls /api/messages/<session> for new assistant
    # turns to TTS. 500ms matches the page poll cadence; tightening to
    # 250ms gives marginally lower mouth-latency at minor server load.
    SESSION_POLL_INTERVAL = 0.5
    MESSAGES_URL_TEMPLATE = BACKEND_URL + "/api/messages/{session_id}"

    def __init__(self, room_name: str, lk_cfg: dict, stt,
                 vad_cfg: dict, http_client, speaker_id=None,
                 wake_factory=None, wake_capture: Optional[WakeMissCapture] = None,
                 smart_turn=None, streaming_asr=None) -> None:
        self.room_name = room_name
        self.lk_cfg = lk_cfg
        self.stt = stt
        self.vad_cfg = vad_cfg
        self.http = http_client
        self.wake_capture = wake_capture
        # Builds one ContinuousWakeWord per audio stream. It has to be per
        # stream: openWakeWord's Model carries a rolling feature window and is
        # not thread-safe, and the old code shared one instance across every
        # room while calling it from a thread pool.
        self.wake_factory = wake_factory
        self.smart_turn = smart_turn
        self.streaming_asr = streaming_asr
        #: identity -> HearingThread. One pipeline per participant.
        self._hearing: dict[str, HearingThread] = {}
        #: Utterances held back because Smart Turn called them unfinished, keyed
        #: by identity. The next utterance from that speaker is appended to them
        #: so a sentence split by a breath arrives as one turn.
        self._held: dict[str, list] = {}
        self._held_deadline: dict[str, float] = {}
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
        turn_cfg = lk_cfg.get("turn_detection", {}) or {}
        #: How long a turn Smart Turn called unfinished waits for its
        #: continuation before being released anyway.
        self._hold_timeout_s = float(turn_cfg.get("hold_timeout_ms", 2200)) / 1000.0
        #: And the ceiling on how much can accumulate that way, so a speaker
        #: who never lands a complete-sounding sentence still gets answered.
        self._hold_max_s = float(turn_cfg.get("hold_max_seconds", 12.0))
        #: Barge-in needs client-side echo cancellation; without it Lloyd's own
        #: voice returns through the mic and interrupts him. VoiceRoom's
        #: half-duplex mute is the belt to this one's braces.
        self._barge_in_enabled = bool(
            (lk_cfg.get("barge_in", {}) or {}).get("enabled", False)
        )
        self._last_partial = ""
        #: Turns whose reply was spoken from their own stream, so the session
        #: poller — which still covers typed and ambient turns — must not say
        #: them again. Bounded: a turn is only at risk while it is recent.
        self._streamed_turns: deque[str] = deque(maxlen=128)
        #: In-flight spoken-reply consumers, cancelled on disconnect.
        self._turn_tasks: set[asyncio.Task] = set()
        vt = lk_cfg.get("voice_turn", {}) or {}
        self._stream_replies = bool(vt.get("stream_replies", True))
        self._max_spoken_chars = int(vt.get("max_spoken_chars", 1500))
        filler = vt.get("filler", {}) or {}
        self._filler_enabled = bool(filler.get("enabled", True))
        self._filler_after_s = float(filler.get("after_seconds", 2.5))
        self._filler_phrases = list(filler.get("phrases") or
                                    ["One moment.", "Let me check.", "Checking now."])
        self._filler_i = 0
        self._prewarm = bool(vt.get("prewarm", True))
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
        self._tasks.append(asyncio.create_task(self._flush_held_loop()))

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
        # A reply still streaming keeps running server-side (the turn does not
        # depend on its reader); only the speaking stops.
        for t in list(self._turn_tasks):
            t.cancel()
        # Stop the hearing threads first. Each one flushes a half-spoken
        # utterance on the way out, so this has to happen before the wait
        # below or that final sentence has nothing left to run on.
        for ht in list(self._hearing.values()):
            try:
                ht.stop()
            except Exception:
                pass
        self._hearing.clear()
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

    def _schedule_partial_publish(self, text: str) -> None:
        """Push a running transcript to the browser.

        Display only. The final transcript is the offline recogniser's, which
        is both more accurate and cheap enough that there is no reason to
        commit a streaming hypothesis.
        """
        if text == self._last_partial:
            return
        self._last_partial = text

        async def _send():
            try:
                await self.room.local_participant.publish_data(
                    json.dumps({"type": "partial_transcript", "text": text,
                                "ts": time.time()}).encode("utf-8"),
                    reliable=False,
                )
            except Exception as e:
                LOG.debug("[%s] partial publish failed: %s", self.room_name, e)

        try:
            asyncio.create_task(_send())
        except RuntimeError:
            pass

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
                            if m.get("turn_id") in self._streamed_turns:
                                # Already said, clause by clause, as it streamed.
                                self._spoken_ids.add(mid)
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

    def _build_hearing(self, identity: str, sample_rate: int) -> HearingThread:
        """One pipeline per participant, on its own thread.

        Everything in it is stateful per stream — the resampler's filter
        history, Silero's recurrent state, openWakeWord's feature window — so
        sharing any of it between participants interleaves two conversations
        inside one model.
        """
        pipeline = HearingPipeline(
            in_rate=sample_rate,
            wake=self.wake_factory.create() if self.wake_factory is not None else None,
            segmenter=voice_vad.build_segmenter(self.vad_cfg),
            streaming=self.streaming_asr,
        )
        ht = HearingThread(
            pipeline,
            on_event=lambda ev: self._on_hearing_event(ev, identity),
            name=f"hearing-{identity[:8]}",
        )
        ht.start()
        LOG.info("[%s] hearing pipeline up for %s @ %d Hz (wake=%s turn=%s asr=%s)",
                 self.room_name, identity, sample_rate,
                 self.wake_factory is not None, self.smart_turn is not None,
                 getattr(self.stt, "name", "?"))
        return ht

    async def _consume_audio(self, track, identity: str) -> None:
        """Pump frames into the participant's hearing thread.

        This coroutine does no signal processing at all any more. It used to
        run the energy VAD inline, which was cheap; the pipeline that replaced
        it is not, and this is the same event loop that paces TTS frames into
        LiveKit on a 100 ms clock.
        """
        stream = rtc.AudioStream(track)
        hearing: Optional[HearingThread] = None
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
                if hearing is None:
                    hearing = self._build_hearing(identity, frame.sample_rate)
                    self._hearing[identity] = hearing
                hearing.push(samples, frame.sample_rate)
        except asyncio.CancelledError:
            pass
        finally:
            if hearing is not None:
                hearing.stop()
                self._hearing.pop(identity, None)
            await stream.aclose()

    # ── Hearing events ───────────────────────────────────────────────────

    def _on_hearing_event(self, ev: HearingEvent, identity: str) -> None:
        """Called on the event loop, from the hearing thread.

        Synchronous and fast: it either publishes a small state update or
        spawns the utterance handler. Anything that waits belongs in a task.
        """
        if ev.kind == "wake":
            self._on_wake(ev, identity)
        elif ev.kind == "speech":
            self._on_speech_start(identity)
        elif ev.kind == "partial":
            self._schedule_partial_publish(ev.text)
        elif ev.kind == "utterance":
            task = asyncio.create_task(self._handle_utterance_event(ev, identity))
            self._utterance_tasks.add(task)
            task.add_done_callback(self._handle_utterance_done)

    def _on_wake(self, ev: HearingEvent, identity: str) -> None:
        """Open the window the instant the word is heard.

        This is the whole latency win of continuous detection. The old worker
        could not reach this point until the VAD had closed the utterance and
        Whisper had returned, so the UI flipped to "Listening" roughly half a
        second after the user stopped speaking. The speaker embedding is
        deliberately *not* awaited here — it is attached when the utterance
        arrives, a few hundred milliseconds later, and making the window wait
        on Resemblyzer would give back most of what was gained.
        """
        if not self.wake.enabled:
            return
        det = ev.detection
        self.wake.extend(identity)
        self._schedule_wake_state_publish()
        if self._prewarm:
            # The user is still mid-sentence; spend that time prefilling the
            # session's prompt (see /api/voice/prewarm).
            try:
                asyncio.create_task(self._post_prewarm())
            except RuntimeError:
                pass
        LOG.info("[%s] wake fired: %s/%.2f — window open (%.1fs)",
                 self.room_name, det.name if det else "?",
                 det.score if det else 0.0, self.wake.continuation_seconds)

    def _on_speech_start(self, identity: str) -> None:
        """Barge-in: the user started talking while Lloyd was.

        Only inside the continuation window and only for the locked speaker —
        otherwise a television in the room silences every reply. Gated on
        `barge_in.enabled` because it needs client-side echo cancellation to be
        safe: without AEC the agent's own voice comes back through the mic and
        interrupts itself on every utterance.
        """
        if not self._barge_in_enabled:
            return
        if self.tts is None or not self.tts.is_speaking:
            return
        if not self.wake.matches_lock(identity):
            return
        dropped = self.tts.interrupt()
        LOG.info("[%s] barge-in from %s — interrupted, dropped %d queued",
                 self.room_name, identity, dropped)

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
        # SpeakerIdentifier takes int16 and divides by 32768 itself. The hearing
        # pipeline hands over float32 in [-1, 1]; passed straight through, that
        # is a 90 dB attenuation ahead of Resemblyzer's silence trimming —
        # invisible while no profile is enrolled, wrong the day one is.
        if samples.dtype != np.int16:
            samples = to_int16(samples)
        loop = asyncio.get_running_loop()
        try:
            name, score, emb = await loop.run_in_executor(
                None, self.speaker_id.identify, samples, sample_rate,
            )
            return emb, name, score
        except Exception as e:
            LOG.warning("[%s] voiceprint embed failed: %s", self.room_name, e)
            return None, "Unknown", 0.0

    # ── Held (unfinished) turns ──────────────────────────────────────────

    def _take_held(self, identity: str, audio: np.ndarray) -> tuple[np.ndarray, float]:
        """Prepend anything held for this speaker, and clear the hold."""
        held = self._held.pop(identity, None)
        self._held_deadline.pop(identity, None)
        if not held:
            return audio, 0.0
        joined = np.concatenate(held + [audio])
        return joined, sum(h.size for h in held) / 16000

    def _hold(self, identity: str, audio: np.ndarray) -> None:
        self._held[identity] = [audio]
        self._held_deadline[identity] = time.monotonic() + self._hold_timeout_s

    async def _flush_held_loop(self) -> None:
        """Release a held turn when the speaker simply stopped.

        Smart Turn says "unfinished" both for a real mid-sentence pause and for
        a trailing-off sentence nobody intends to finish. Without this, the
        second case would sit in the buffer until the next thing anyone said —
        which could be the following morning.
        """
        try:
            while True:
                await asyncio.sleep(0.25)
                now = time.monotonic()
                for identity, deadline in list(self._held_deadline.items()):
                    if now < deadline:
                        continue
                    held = self._held.pop(identity, None)
                    self._held_deadline.pop(identity, None)
                    if not held:
                        continue
                    audio = np.concatenate(held)
                    LOG.info("[%s] releasing %.2fs held turn from %s (timeout)",
                             self.room_name, audio.size / 16000, identity)
                    ev = HearingEvent(
                        "utterance",
                        utterance=voice_vad.Utterance(
                            audio=audio, start_sample=0, end_sample=audio.size,
                            max_prob=1.0, reason="hold_timeout",
                        ),
                    )
                    task = asyncio.create_task(self._handle_utterance_event(ev, identity))
                    self._utterance_tasks.add(task)
                    task.add_done_callback(self._handle_utterance_done)
        except asyncio.CancelledError:
            return

    def _wake_peak(self, identity: str) -> tuple[str, float]:
        """Best wake score seen since the last drop, for the diag line."""
        ht = self._hearing.get(identity)
        if ht is None or ht.pipeline.wake is None:
            return "", 0.0
        return ht.pipeline.wake.take_peak()

    def _record_diag(self, identity: str, audio: np.ndarray, ev: HearingEvent,
                     text: str, latency: float, in_continuation: bool) -> None:
        cap = self.wake_capture
        if cap is None:
            return
        try:
            f32 = np.asarray(audio, dtype=np.float32)
            cap.record_utterance(
                utterance_id=uuid.uuid4().hex[:12],
                room=self.room_name,
                identity=identity,
                duration_s=f32.size / 16000,
                rms_mean=float(np.sqrt(np.mean(f32 * f32))) if f32.size else 0.0,
                rms_peak=float(np.max(np.abs(f32))) if f32.size else 0.0,
                voiced_ratio=float(ev.utterance.max_prob) if ev.utterance else 0.0,
                ww_ran=True,
                ww_name=ev.wake.name if ev.wake else "",
                ww_score=ev.wake.score if ev.wake else 0.0,
                ww_threshold=(self.wake_factory.threshold
                              if self.wake_factory is not None else 0.0),
                ww_fired=ev.wake is not None,
                in_continuation=in_continuation,
                stt_text=text,
                stt_latency_s=latency,
                # Recorded at 16 kHz now: that is the rate every model in the
                # pipeline actually saw, so a replay reproduces the decision
                # rather than approximating it from the room's 48 kHz.
                samples=to_int16(f32),
                sample_rate=16000,
                client_info=self._client_meta.get(identity),
            )
        except Exception as e:
            LOG.debug("[%s] ww-diag record failed: %s", self.room_name, e)

    async def _handle_utterance_event(self, ev: HearingEvent, identity: str) -> None:
        """Decide what a closed utterance means, and inject it if it was for us.

        The wake decision is already made by the time this runs — `ev.wake` is
        the detection that fell inside this utterance's span, or None. That is
        the inversion at the heart of the rework: the old version had to *ask*
        openWakeWord here, after the VAD, which is why it never heard anything
        the VAD had already mangled.
        """
        utt = ev.utterance
        wake = self.wake
        utterance_start_t = time.monotonic() - utt.duration_s
        in_continuation = wake.enabled and wake.matches_lock(identity, at=utterance_start_t)
        woke = ev.wake is not None
        early: Optional[voice_asr.Transcript] = None

        if wake.enabled and not woke and not in_continuation:
            peak_name, peak = self._wake_peak(identity)
            if wake.text_fallback:
                # The second wake path: the acoustic models missed it, but the
                # transcript may still open with "hey Lloyd". Transcribing an
                # idle utterance is ~60 ms with Parakeet, where Whisper was
                # the reason this was never done.
                try:
                    early = await asyncio.to_thread(self.stt.transcribe, utt.audio)
                except Exception as e:
                    LOG.warning("[%s] text-wake STT failed: %s", self.room_name, e)
                if early is not None and _strip_wake_word(early.text, wake.words) is not None:
                    woke = True
                    LOG.info("[%s] wake from transcript %r (acoustic peak %s/%.2f)",
                             self.room_name, early.text[:60], peak_name or "-", peak)
            if not woke:
                LOG.info(
                    "[%s] not addressed to Lloyd — dropped %.2fs "
                    "(vad=%.2f wake_peak=%s/%.2f text=%r)",
                    self.room_name, utt.duration_s, utt.max_prob, peak_name or "-",
                    peak, (early.text[:60] if early else None),
                )
                return

        # ── Is it a finished thought? ───────────────────────────────
        # A 380 ms silence closes an utterance; people pause longer than that
        # mid-sentence. Rather than lengthening the silence for everyone, hold
        # the audio when Smart Turn says the speaker has not finished, and
        # glue it to what comes next.
        audio, held_s = self._take_held(identity, utt.audio)
        if (self.smart_turn is not None and early is None
                and utt.reason != "max_duration"):
            verdict = await asyncio.to_thread(self.smart_turn.predict, audio)
            if not verdict.complete and held_s < self._hold_max_s:
                self._hold(identity, audio)
                LOG.info(
                    "[%s] holding %.2fs — turn looks unfinished (p=%.2f, %.0f ms)",
                    self.room_name, audio.size / 16000, verdict.probability,
                    verdict.elapsed_ms,
                )
                return
            LOG.debug("[%s] turn complete p=%.2f (%.0f ms)",
                      self.room_name, verdict.probability, verdict.elapsed_ms)

        # ── Transcribe ──────────────────────────────────────────────
        t0 = time.monotonic()
        try:
            # A text wake already transcribed this exact audio; nothing was
            # held in front of it, because a held turn implies an open window.
            result = early if early is not None and held_s == 0 else \
                await asyncio.to_thread(self.stt.transcribe, audio)
        except Exception as e:
            LOG.warning("[%s] STT failed on %.2fs: %s",
                        self.room_name, audio.size / 16000, e)
            return
        text = result.text
        latency = time.monotonic() - t0
        duration_s = audio.size / 16000
        LOG.info(
            "[%s][diag] dur=%.2fs vad=%.2f wake=%s asr=%s/%.2fs cont=%s text=%r",
            self.room_name, duration_s, utt.max_prob,
            (f"{ev.wake.name}:{ev.wake.score:.2f}" if ev.wake else "text") if woke else "-",
            result.backend, latency, "Y" if in_continuation else "N",
            (text or "")[:80],
        )
        self._record_diag(identity, audio, ev, text, latency, in_continuation)

        # ── Who said it ─────────────────────────────────────────────
        # Deliberately after the wake window was opened, not before: the
        # window is what the UI reacts to and Resemblyzer costs ~150 ms.
        if woke:
            emb, name, score = await self._embed_async(audio, 16000)
            if emb is not None and self.speaker_id is not None:
                LOG.info("[%s] wake speaker: %s (cos=%.2f)", self.room_name, name, score)
            wake.set_anchor(emb, name if name and name != "Unknown" else None)
            wake.extend(identity)
            await self._publish_wake_state()

        if not text:
            if woke:
                LOG.info("[%s] bare wake word — window open %.1fs, nothing to inject",
                         self.room_name, wake.continuation_seconds)
            else:
                LOG.info("[%s] empty transcript for %.2fs from %s",
                         self.room_name, duration_s, identity)
            return

        # ── Gate decisions ──────────────────────────────────────────
        inject_text: Optional[str] = None
        speaker_name: Optional[str] = None  # populated from anchor or fresh ID
        samples, sample_rate = audio, 16000
        if not wake.enabled:
            inject_text = text
        elif in_continuation and not woke:
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
            # The wake word fired inside this utterance. All that is left is
            # deciding how much of the transcript is the word itself.
            speaker_name = wake.anchor_name
            tail = _strip_wake_word(text, wake.words)
            word_count = len([w for w in text.split() if w.strip(".,!?;:'\"")])
            ww_name = ev.wake.name if ev.wake else "?"
            ww_score = ev.wake.score if ev.wake else 0.0
            if tail is not None:
                if not tail and wake.skip_inject_if_only_wake_word:
                    LOG.info(
                        "[%s] bare wake-word (%s/%.2f) — opening %.1fs window, no inject",
                        self.room_name, ww_name, ww_score, wake.continuation_seconds,
                    )
                    return
                inject_text = tail or text
            elif word_count <= 2:
                # A short transcript the matcher did not recognise, on an
                # utterance the acoustic model *did*. That is a mistranscribed
                # bare wake word, so open the window and wait for the follow-up
                # rather than injecting "Eloid" as a question.
                LOG.info(
                    "[%s] bare wake-word inferred from short transcript %r (%s/%.2f) — opening %.1fs window",
                    self.room_name, text[:40], ww_name, ww_score, wake.continuation_seconds,
                )
                return
            else:
                # A full sentence whose wake word the ASR spelled differently.
                # The acoustic model is the authority here, so inject as-is.
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
        if not (self._stream_replies and self.tts is not None):
            await self._inject_only(payload)
            return
        # The reply is spoken by its own task, which lives for the whole turn
        # — minutes, with tools. The utterance handler returns now, so the next
        # thing the user says is heard while Lloyd is still answering.
        task = asyncio.create_task(self._speak_voice_turn(payload))
        self._turn_tasks.add(task)
        task.add_done_callback(self._turn_tasks.discard)

    async def _post_prewarm(self) -> None:
        try:
            await self.http.post(PREWARM_URL, json={"session_key": self.session_id},
                                 timeout=5.0)
        except Exception as e:
            LOG.debug("[%s] prewarm request failed: %s", self.room_name, e)

    async def _inject_only(self, payload: dict) -> None:
        try:
            r = await self.http.post(INJECT_URL, json=payload, timeout=10.0)
            if r.status_code >= 300:
                LOG.warning("[%s] inject failed %d: %s",
                            self.room_name, r.status_code, r.text[:200])
        except Exception as e:
            LOG.warning("[%s] inject POST failed: %s", self.room_name, e)

    # ── Speaking a reply as it streams ───────────────────────────────────

    async def _speak_voice_turn(self, payload: dict) -> None:
        """Inject the turn and speak its reply clause by clause as it streams.

        The old path waited for the whole answer, then for the secondary model
        to rewrite it (0.5-2 s), then for the 500 ms session poll to notice
        it. Now the first sentence goes to TTS as soon as the model finishes
        writing it, and the rewrite happens at the source: `/api/voice/inject`
        tells the model it is speaking (see VOICE_TURN_REMINDER there).
        """
        import httpx

        tts = self.tts
        gen = tts.generation
        cs = ClauseStream()
        t0 = time.monotonic()
        state = {"said": 0, "capped": False, "filler": False, "first_at": None}
        filler_task: Optional[asyncio.Task] = None

        async def say(clause: str) -> None:
            if state["capped"] or tts.generation != gen:
                return
            if state["said"] + len(clause) > self._max_spoken_chars and state["said"]:
                state["capped"] = True
                await tts.speak("The rest is in the chat.")
                return
            if state["first_at"] is None:
                state["first_at"] = time.monotonic()
                LOG.info("[%s] first clause %.2fs after inject: %r",
                         self.room_name, state["first_at"] - t0, clause[:60])
            state["said"] += len(clause)
            await tts.speak(clause)

        async def filler() -> None:
            # Only when nothing has been said yet, and only once a turn: a
            # filler is for the silence before the answer, not a tic.
            if not self._filler_enabled or state["filler"] or state["said"]:
                return
            state["filler"] = True
            phrase = self._filler_phrases[self._filler_i % len(self._filler_phrases)]
            self._filler_i += 1
            await tts.speak(phrase)

        async def filler_after_delay() -> None:
            await asyncio.sleep(self._filler_after_s)
            await filler()

        try:
            async with self.http.stream(
                "POST", INJECT_URL, json=dict(payload, stream=True),
                timeout=httpx.Timeout(10.0, read=None),
            ) as resp:
                if resp.status_code >= 300:
                    body = await resp.aread()
                    LOG.warning("[%s] inject failed %d: %s",
                                self.room_name, resp.status_code, body[:200])
                    return
                if "text/event-stream" not in resp.headers.get("content-type", ""):
                    # A backend from before streaming: the turn is queued and
                    # the session poller will speak it the old way.
                    await resp.aread()
                    LOG.info("[%s] inject answered without a stream — poller speaks it",
                             self.room_name)
                    return
                filler_task = asyncio.create_task(filler_after_delay())
                async for event, data in _iter_sse(resp):
                    if tts.generation != gen:
                        LOG.info("[%s] reply interrupted — no longer speaking it",
                                 self.room_name)
                        break
                    if event == "voice_turn":
                        # Registered before the first segment can be persisted,
                        # so the poller never races the stream for it.
                        self._streamed_turns.append(data.get("turn_id", ""))
                    elif event == "text_delta":
                        for clause in cs.feed(data.get("text", "")):
                            await say(clause)
                    elif event == "tool_start":
                        # Whatever was written before the tool is a finished
                        # thought ("Let me check the calendar.") — say it now,
                        # and if nothing at all has been said, say something.
                        for clause in cs.flush():
                            await say(clause)
                        await filler()
                    elif event in ("done", "error"):
                        break
                for clause in cs.flush():
                    await say(clause)
                if cs.skipped_code and not cs.stopped and state["said"]:
                    await say("I've put the code in the chat.")
                LOG.info("[%s] voice turn spoken: %d chars in %.1fs%s%s",
                         self.room_name, state["said"], time.monotonic() - t0,
                         " (capped)" if state["capped"] else "",
                         " (filler)" if state["filler"] else "")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            LOG.warning("[%s] voice turn stream failed: %s", self.room_name, e)
        finally:
            if filler_task is not None and not filler_task.done():
                filler_task.cancel()


async def _iter_sse(resp):
    """(event, data) pairs from a server-sent-events response."""
    event, data_lines = "message", []
    async for line in resp.aiter_lines():
        if not line:
            if data_lines:
                try:
                    payload = json.loads("\n".join(data_lines))
                except ValueError:
                    payload = {}
                yield event, payload if isinstance(payload, dict) else {}
            event, data_lines = "message", []
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())


# ── Worker manager ───────────────────────────────────────────────────────

class WorkerManager:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.lk_cfg = cfg["livekit"]
        self.room_prefix = self.lk_cfg.get("room_prefix", DEFAULT_ROOM_PREFIX)
        self.bridges: dict[str, RoomBridge] = {}
        self._stopping = asyncio.Event()
        self._http_url = _http_url(self.lk_cfg["url"])
        stt_cfg = self.lk_cfg.get("stt", {}) or {}
        # The recogniser itself is shared across rooms and stateless per call;
        # only the *streaming* recogniser hands out per-stream objects.
        self.stt = voice_asr.build_recognizer(stt_cfg)
        if isinstance(self.stt, voice_asr.WhisperRecognizer):
            hw = stt_cfg.get("hotwords_file")
            self.stt.hotwords = _load_hotwords(hw) if hw else None
            if self.stt.hotwords:
                LOG.info("STT hotwords loaded: %r", self.stt.hotwords)
        self.streaming_asr = voice_asr.build_streaming_recognizer(stt_cfg)
        self.vad_cfg = self.lk_cfg.get("vad", {}) or {}
        # SpeakerIdentifier is shared across rooms — one VoiceEncoder load,
        # one profiles_dir watcher, single source of truth. None when
        # voiceprint matching is disabled in config.
        self.speaker_id = _build_speaker_id(self.lk_cfg.get("voiceprint", {}) or {})
        # Validates the wake-word files once here, then mints one model per
        # audio stream — openWakeWord's Model is stateful and not thread-safe,
        # and the old code shared a single instance across every room.
        self.wake_factory = voice_wake.build_factory(
            self.lk_cfg.get("acoustic_wake", {}) or {}
        )
        # Smart Turn is stateless per call and genuinely shareable.
        self.smart_turn = voice_turn.build_smart_turn(
            self.lk_cfg.get("turn_detection", {}) or {}
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
            await asyncio.to_thread(self.stt.load)
        except Exception as e:
            LOG.warning("STT eager-load failed (will retry on first utterance): %s", e)
        # The speaker encoder too: loaded lazily it cost 0.8 s on the first wake
        # of every worker lifetime, in the path between transcript and inject.
        if self.speaker_id is not None:
            try:
                await asyncio.to_thread(self.speaker_id._ensure_encoder)
            except Exception as e:
                LOG.warning("speaker encoder eager-load failed: %s", e)
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
                wake_factory=self.wake_factory,
                wake_capture=self.wake_capture,
                smart_turn=self.smart_turn,
                streaming_asr=self.streaming_asr,
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
