"""Output shaping for the cloned TTS voice on the LiveKit path.

Two corrections, both applied to the PCM stream `livekit_worker` receives from
the Qwen3-TTS server at :8090, both stateful across chunks within a single
utterance and reset between utterances.

**PresenceEQ** — the Qwen3-TTS 12 Hz speech tokenizer rolls its output off
above ~1.5 kHz.  Measured against the clone's own reference clip
(`voice_library/profiles/dave_cullen/ref.wav`), the synthesised voice tracks
the real speaker to within 0.5 dB everywhere below 1.5 kHz and then falls
away: -1.8 dB at 2 kHz, -3.9 dB at 4 kHz, -4.4 dB at 8 kHz, -6.2 dB above
9 kHz.  That missing presence band is what makes it sound hollow and far away
— it carries proximity and consonant detail — and two high shelves put it
back.

The deficit belongs to the vocoder, not to the reference clip: a built-in
voice with no cloning at all measures the same way, and re-cutting the
reference from a second source video did not move it.  So the correction goes
on the output and there is nothing to fix in the profile.

Note what is deliberately *not* here: a cut around 300 Hz.  The output/reference
ratio in that band looks wrong if you measure 200-500 Hz against 2-6 kHz, but
the band's absolute shape already matches the reference to 0.1 dB — the ratio
is off only because the top is missing.  Cutting the low-mid would fix the
ratio by making the voice thin instead of hollow.

**WsolaStretch** — the server's streaming path silently drops `speed`:
`generate_voice_clone_streaming` takes no such parameter while the
non-streaming path applies `librosa.effects.time_stretch`.  Voice mode always
streams, so the configured `speed: 0.70` did nothing there and Lloyd spoke
about 1.5x faster than the voice he is cloning.  This restores it client-side
with WSOLA, which is time-domain: a phase vocoder would add exactly the smeared,
hollow quality this module exists to remove.

Both stages are pure numpy/scipy and hold no LiveKit dependency so they can be
tested directly — see `tests/test_tts_output_shaping.py`.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# Fitted 2026-09-06 against dave_cullen_001/002 over three synthesised
# utterances.  Residual error after correction is under 1 dB from 300 Hz to
# 9 kHz; BOX (200-500 Hz over 2-6 kHz) moves from 8.2-11.4 to 4.3-6.1 against
# the reference's 4.0-4.5, and spectral centroid from 601-710 Hz to 765-975 Hz
# against the reference's 788-899 Hz.
DEFAULT_SHELVES: tuple[dict, ...] = (
    {"freq": 1800.0, "gain_db": 2.75, "q": 1.0},
    {"freq": 8000.0, "gain_db": 4.0, "q": 0.7},
)


def high_shelf_sos(freq: float, gain_db: float, q: float, sample_rate: int) -> np.ndarray:
    """One RBJ high-shelf biquad as a second-order-section row.

    Returns the six `[b0, b1, b2, 1, a1, a2]` coefficients `scipy.signal.sosfilt`
    expects.  A shelf whose corner sits at or above Nyquist is returned as a
    pass-through rather than an error, so an over-ambitious config value
    degrades to "no correction up there" instead of taking the voice down.
    """
    nyquist = sample_rate / 2.0
    if not (0.0 < freq < nyquist) or gain_db == 0.0:
        return np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    q = max(float(q), 1e-3)
    amp = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * freq / sample_rate
    cos_w0, sin_w0 = np.cos(w0), np.sin(w0)
    alpha = sin_w0 / (2.0 * q)
    two_sqrt_a_alpha = 2.0 * np.sqrt(amp) * alpha

    b0 = amp * ((amp + 1.0) + (amp - 1.0) * cos_w0 + two_sqrt_a_alpha)
    b1 = -2.0 * amp * ((amp - 1.0) + (amp + 1.0) * cos_w0)
    b2 = amp * ((amp + 1.0) + (amp - 1.0) * cos_w0 - two_sqrt_a_alpha)
    a0 = (amp + 1.0) - (amp - 1.0) * cos_w0 + two_sqrt_a_alpha
    a1 = 2.0 * ((amp - 1.0) - (amp + 1.0) * cos_w0)
    a2 = (amp + 1.0) - (amp - 1.0) * cos_w0 - two_sqrt_a_alpha
    return np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0])


class PresenceEQ:
    """Cascaded high shelves restoring the HF the vocoder drops.

    Filter state carries across `process` calls so chunk boundaries introduce
    no discontinuity; `reset` starts a new utterance from silence.
    """

    def __init__(self, sample_rate: int, shelves: Optional[Sequence[dict]] = None) -> None:
        from scipy.signal import sosfilt  # noqa: F401 — fail fast if scipy is absent

        specs = list(shelves) if shelves is not None else list(DEFAULT_SHELVES)
        rows = [
            high_shelf_sos(
                float(s.get("freq", 0.0)),
                float(s.get("gain_db", 0.0)),
                float(s.get("q", 0.7)),
                sample_rate,
            )
            for s in specs
        ]
        self.sample_rate = int(sample_rate)
        self.sos = np.vstack(rows) if rows else np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
        self._zi: Optional[np.ndarray] = None
        self.reset()

    def reset(self) -> None:
        # Zeros, not `sosfilt_zi`: an utterance begins from silence, and
        # sosfilt_zi's steady-state initial condition would open it with a
        # step transient.
        self._zi = np.zeros((self.sos.shape[0], 2))

    def process(self, samples: np.ndarray) -> np.ndarray:
        from scipy.signal import sosfilt

        if samples.size == 0:
            return samples.astype(np.float32, copy=False)
        out, self._zi = sosfilt(self.sos, samples.astype(np.float64), zi=self._zi)
        return out.astype(np.float32)


class WsolaStretch:
    """Streaming WSOLA time-stretch.

    `rate` follows librosa's convention: <1 slows down (output is longer), >1
    speeds up.  Feed chunks to `process` and call `flush` once at the end of the
    utterance; between them the average rate is held exactly, because the
    analysis pointer advances by a fixed hop and the waveform-similarity search
    only perturbs *where each frame is read from*, never how far the pointer
    moves.
    """

    def __init__(self, rate: float, sample_rate: int) -> None:
        self.rate = float(rate)
        self.sample_rate = int(sample_rate)
        # ~20 ms synthesis hop, 50 % overlap.  The search radius must exceed one
        # pitch period of the voice (192 samples at 125 Hz, 24 kHz) or the
        # similarity search cannot align consecutive frames and WSOLA degrades
        # into plain overlap-add.
        self.hop = max(64, int(round(sample_rate * 0.020)))
        self.frame = self.hop * 2
        self.radius = max(32, int(round(sample_rate * 0.011)))
        self.analysis_hop = max(1, int(round(self.hop * self.rate)))
        # Periodic Hann: at 50 % overlap consecutive windows sum to exactly 1.
        self.window = np.hanning(self.frame + 1)[: self.frame].astype(np.float64)
        self.reset()

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float64)
        self._origin = 0          # global index of self._buf[0]
        self._ideal = 0           # global index the next frame would ideally start at
        self._pending = np.zeros(self.hop, dtype=np.float64)
        self._tail: Optional[np.ndarray] = None  # natural continuation of the last frame
        self._started = False

    # -- internals ---------------------------------------------------------
    def _best_offset(self, local: int) -> int:
        """Index into `self._buf` for the frame nearest `local` that best
        continues the previous frame's waveform."""
        if self._tail is None:
            return local
        lo = max(0, local - self.radius)
        hi = min(len(self._buf) - self.frame, local + self.radius)
        if hi <= lo:
            return max(0, min(local, len(self._buf) - self.frame))
        seg = self._buf[lo : hi + self.hop]
        if len(seg) < self.hop:
            return lo
        corr = np.correlate(seg, self._tail, mode="valid")
        # Normalise by the candidate's own energy, or the search just walks to
        # the loudest place in the window instead of the best-matching one.
        csum = np.concatenate([[0.0], np.cumsum(seg * seg)])
        energy = csum[self.hop :] - csum[: -self.hop]
        n = min(len(corr), len(energy))
        score = corr[:n] / np.sqrt(energy[:n] + 1e-12)
        return lo + int(np.argmax(score))

    def _run(self) -> np.ndarray:
        out: list[np.ndarray] = []
        while True:
            local = self._ideal - self._origin
            # Need radius of slack on the left and a whole frame plus radius on
            # the right before a frame can be placed.
            if local < self.radius or local + self.frame + self.radius > len(self._buf):
                break
            best = self._best_offset(local)
            frame = self._buf[best : best + self.frame] * self.window
            out.append(self._pending + frame[: self.hop])
            self._pending = frame[self.hop :].copy()
            self._tail = self._buf[best + self.hop : best + self.hop + self.hop].copy()
            self._ideal += self.analysis_hop
            # Drop input we can no longer reach, keeping the search slack.
            keep_from = max(0, (self._ideal - self._origin) - self.radius)
            if keep_from > 0:
                self._buf = self._buf[keep_from:]
                self._origin += keep_from
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float64)

    # -- public ------------------------------------------------------------
    def process(self, samples: np.ndarray) -> np.ndarray:
        if samples.size:
            self._buf = np.concatenate([self._buf, samples.astype(np.float64)])
        if not self._started:
            # The first frame has no predecessor to match, so it is read at the
            # pointer itself; give the pointer its left-hand slack up front.
            self._ideal = self._origin + self.radius
            self._started = True
        return self._run().astype(np.float32)

    def flush(self) -> np.ndarray:
        """Emit the tail of the utterance and reset for the next one."""
        # Pad so the loop can consume what is left; the Hann window tapers the
        # final frame to zero, so the utterance ends without a click.
        self._buf = np.concatenate(
            [self._buf, np.zeros(self.frame + 2 * self.radius, dtype=np.float64)]
        )
        out = np.concatenate([self._run(), self._pending])
        self.reset()
        return out.astype(np.float32)


class OutputShaper:
    """Both corrections plus the int16 <-> float plumbing, as one object.

    Feed it the raw PCM bytes the TTS server streams and it returns PCM bytes
    ready to push at LiveKit, buffering whatever the stretcher cannot yet emit.
    Call `flush` once when the response ends; it drains the tail and resets, so
    the same instance serves every utterance.

    `enabled` is False when there is nothing to do (speed 1.0, EQ off), and the
    caller can then pass bytes straight through untouched.
    """

    def __init__(
        self,
        sample_rate: int,
        speed: float = 1.0,
        presence_eq: bool = True,
        shelves: Optional[Sequence[dict]] = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.speed = float(speed)
        self._eq = PresenceEQ(sample_rate, shelves) if presence_eq else None
        # A stretch at exactly 1.0 is not free — it still overlap-adds — so skip
        # it rather than pay for a no-op.
        self._stretch = (
            WsolaStretch(self.speed, sample_rate)
            if abs(self.speed - 1.0) > 1e-3
            else None
        )
        self._odd = bytearray()  # trailing byte of a sample split across chunks
        self.clipped = 0
        self.total = 0

    @property
    def enabled(self) -> bool:
        return self._eq is not None or self._stretch is not None

    def describe(self) -> str:
        parts = []
        if self._stretch is not None:
            parts.append(f"speed {self.speed:g} (WSOLA)")
        if self._eq is not None:
            parts.append(f"presence EQ {self._eq.sos.shape[0]} shelves")
        return ", ".join(parts) if parts else "passthrough"

    def _emit(self, samples: np.ndarray) -> bytes:
        if samples.size == 0:
            return b""
        self.total += samples.size
        clipped = int(np.count_nonzero(np.abs(samples) > 1.0))
        if clipped:
            self.clipped += clipped
            samples = np.clip(samples, -1.0, 1.0)
        return np.rint(samples * 32767.0).astype("<i2").tobytes()

    def process(self, pcm: bytes) -> bytes:
        """Shape one chunk of signed-16-bit little-endian PCM."""
        if not self.enabled:
            return pcm
        self._odd.extend(pcm)
        n = len(self._odd) - (len(self._odd) % 2)
        if n == 0:
            return b""
        samples = np.frombuffer(bytes(self._odd[:n]), dtype="<i2").astype(np.float32) / 32768.0
        del self._odd[:n]
        if self._stretch is not None:
            samples = self._stretch.process(samples)
        if self._eq is not None:
            samples = self._eq.process(samples)
        return self._emit(samples)

    def flush(self) -> bytes:
        """Drain the utterance tail and reset for the next one."""
        if not self.enabled:
            self._odd.clear()
            return b""
        samples = np.zeros(0, dtype=np.float32)
        if self._stretch is not None:
            samples = self._stretch.flush()
        if self._eq is not None:
            samples = self._eq.process(samples)
            self._eq.reset()
        self._odd.clear()
        return self._emit(samples)
