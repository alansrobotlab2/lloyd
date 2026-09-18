"""One resample, shared by everything downstream.

The old worker resampled three times per utterance — openWakeWord did its own
`resample_poly`, faster-whisper's `decode_audio` did another through PyAV, and
Resemblyzer a third. Each was correct and each was a copy, so the cost scaled
with the number of listeners rather than with the audio.

More importantly, all three ran on a *finished* utterance. A wake word that is
detected continuously needs a continuous 16 kHz stream, which means a resampler
that carries state across frame boundaries. `resample_poly` on each 10 ms frame
independently is not that: it zero-pads both ends of every frame, so every
boundary gets a small transient, and at 480 samples per frame those transients
are a significant fraction of the signal.

`StreamResampler` guards **both** edges. `resample_poly` zero-pads each end of
whatever it is handed, so a naive implementation that only prepends history
still emits a contaminated tail — measured at 6.7e-2 peak error against the
one-shot result, which is audible. So the resampler also holds back the last
`_guard` input samples and emits them on the following call, once they have
filter support on the right. The cost is a fixed delay of `_guard / in_rate`
seconds — 2.7 ms at 48 kHz — and the reward is that the output is
sample-for-sample what `resample_poly` would have produced over the whole
stream at once. `test_streaming_matches_one_shot` pins that to 1e-6.
"""
from __future__ import annotations

from math import ceil, gcd

import numpy as np

TARGET_RATE = 16000


def _guard_samples(up: int, down: int) -> int:
    """Input samples of filter support to keep on each side.

    `resample_poly` designs a FIR of ``2 * 10 * max(up, down) + 1`` taps in the
    *upsampled* domain, so its reach in input samples is that over `up`. Doubled
    and floored at 64, because being generous here costs microseconds and being
    stingy costs a boundary artifact on every frame.
    """
    taps = 20 * max(up, down) + 1
    return max(64, int(ceil(taps / up)) * 2)


def to_float32(samples: np.ndarray) -> np.ndarray:
    """int16 PCM -> float32 in [-1, 1). Everything downstream wants float32:
    Silero, sherpa-onnx and the Whisper mel all normalise this way."""
    if samples.dtype == np.float32:
        return samples
    return samples.astype(np.float32) / 32768.0


def to_int16(samples: np.ndarray) -> np.ndarray:
    """float32 [-1, 1) -> int16, clipped. openWakeWord's melspectrogram model
    is trained on int16-valued input and gives nonsense on [-1, 1) floats."""
    if samples.dtype == np.int16:
        return samples
    return np.clip(samples * 32768.0, -32768, 32767).astype(np.int16)


class StreamResampler:
    """Stateful resampler to 16 kHz mono float32.

    Feed it whatever the room sends; it returns the resampled continuation of
    the stream. A rate change (a participant reconnecting on a different
    device) rebuilds the filter and drops the history, which is a discontinuity
    of one frame rather than a crash.
    """

    def __init__(self, in_rate: int, out_rate: int = TARGET_RATE) -> None:
        self.out_rate = int(out_rate)
        self._in_rate = 0
        self._up = 1
        self._down = 1
        self._guard = 0
        #: Emitted input samples kept for left-hand filter support.
        self._history = np.zeros(0, dtype=np.float32)
        #: Input samples received but not yet emitted, awaiting right-hand
        #: support. These are what stops the tail transient escaping.
        self._pending = np.zeros(0, dtype=np.float32)
        self._configure(int(in_rate))

    def _configure(self, in_rate: int) -> None:
        if in_rate == self._in_rate:
            return
        self._in_rate = in_rate
        g = gcd(self.out_rate, in_rate) or 1
        self._up = self.out_rate // g
        self._down = in_rate // g
        # Round the guard up to a whole number of input periods so the output
        # index arithmetic below stays exact instead of drifting by a sample
        # every few seconds.
        guard = _guard_samples(self._up, self._down)
        self._guard = guard + (-guard % self._down)
        self.reset()

    @property
    def in_rate(self) -> int:
        return self._in_rate

    @property
    def delay_samples(self) -> int:
        """Input samples of algorithmic delay this resampler adds."""
        return self._guard if (self._up, self._down) != (1, 1) else 0

    def reset(self) -> None:
        self._history = np.zeros(0, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)

    def push(self, samples: np.ndarray, in_rate: int | None = None) -> np.ndarray:
        """Resample one frame. Returns float32 at `out_rate`, possibly empty."""
        if in_rate is not None and in_rate != self._in_rate:
            self._configure(int(in_rate))
        x = to_float32(np.asarray(samples).reshape(-1))
        if self._up == 1 and self._down == 1:
            return x.astype(np.float32, copy=False)
        if x.size == 0:
            return np.zeros(0, dtype=np.float32)

        from scipy.signal import resample_poly

        hist, pend = self._history, self._pending
        combined = np.concatenate([hist, pend, x])
        # Hold back the trailing guard so no emitted sample was computed
        # against resample_poly's own right-hand zero padding.
        emit_in = combined.size - hist.size - self._guard
        if emit_in <= 0:
            self._pending = combined[hist.size:]
            return np.zeros(0, dtype=np.float32)

        y = resample_poly(combined, self._up, self._down)
        lo = (hist.size * self._up) // self._down
        hi = ((hist.size + emit_in) * self._up) // self._down
        out = y[lo:hi].astype(np.float32, copy=False)

        consumed = hist.size + emit_in
        keep = min(self._guard, consumed)
        keep -= keep % self._down
        self._history = combined[consumed - keep:consumed] if keep else np.zeros(0, dtype=np.float32)
        self._pending = combined[consumed:]
        return out

    def flush(self) -> np.ndarray:
        """Emit the held-back tail. For end-of-stream only — calling this
        mid-stream reintroduces exactly the boundary transient the guard
        exists to prevent."""
        if self._pending.size == 0 or (self._up, self._down) == (1, 1):
            out, self._pending = self._pending, np.zeros(0, dtype=np.float32)
            return out.astype(np.float32, copy=False)

        from scipy.signal import resample_poly

        hist = self._history
        combined = np.concatenate([hist, self._pending])
        y = resample_poly(combined, self._up, self._down)
        lo = (hist.size * self._up) // self._down
        self.reset()
        return y[lo:].astype(np.float32, copy=False)


class FrameChunker:
    """Re-blocks a stream into fixed-size frames.

    Silero wants exactly 512 samples and openWakeWord exactly 1280; LiveKit
    delivers 480. Every consumer needs this and none of them should own it.
    """

    def __init__(self, frame_size: int) -> None:
        self.frame_size = int(frame_size)
        self._buf = np.zeros(0, dtype=np.float32)

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if self._buf.size:
            x = np.concatenate([self._buf, x])
        n = (x.size // self.frame_size) * self.frame_size
        frames = [x[i:i + self.frame_size] for i in range(0, n, self.frame_size)]
        self._buf = x[n:]
        return frames

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
