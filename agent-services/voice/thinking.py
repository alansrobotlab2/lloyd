"""A quiet "still working" sound for the stretches of a turn with nothing to say.

A tool turn can run for a minute. Progress lines (`conversation.caption_to_speech`)
cover it every ~8 s, but between them the room is silent, and silence on a voice
channel reads as "it died". LiveKit Agents plays a `BackgroundAudioPlayer`
"thinking" clip for exactly this state; this is the synthesised equivalent — a
soft two-tone pulse, well below speech level, generated rather than shipped so
there is no asset to lose in a rebuild.

Level matters twice. It must be audible in earbuds without being noticed as a
voice, and it must stay under the browser's half-duplex speaking threshold
(`AGENT_SPEAKING_THRESHOLD` 0.02 in VoiceRoom.tsx) so that, where half-duplex is
still on, the mic is not muted for a tool turn's whole length. Peak 0.02 full
scale under a raised-cosine envelope keeps RMS near 0.008.
"""
from __future__ import annotations

import numpy as np


class ThinkingSound:
    """Endless 16-bit mono PCM, one frame at a time."""

    def __init__(self, sample_rate: int = 24000, peak: float = 0.02,
                 period_s: float = 1.6, pulse_s: float = 0.32,
                 tones: tuple[float, float] = (392.0, 523.25)) -> None:
        self.sample_rate = int(sample_rate)
        n_period = int(period_s * self.sample_rate)
        n_pulse = int(pulse_s * self.sample_rate)
        t = np.arange(n_pulse) / self.sample_rate
        env = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_pulse) / max(1, n_pulse - 1))
        half = n_pulse // 2
        pulse = np.zeros(n_pulse, dtype=np.float64)
        # Two soft notes, the second a fourth above — a question, not an alarm.
        pulse[:half] = np.sin(2 * np.pi * tones[0] * t[:half])
        pulse[half:] = np.sin(2 * np.pi * tones[1] * t[half:])
        pulse *= env * peak
        cycle = np.zeros(n_period, dtype=np.float64)
        cycle[:n_pulse] = pulse
        self._pcm = (np.clip(cycle, -1, 1) * 32767).astype(np.int16).tobytes()
        self._pos = 0

    def reset(self) -> None:
        self._pos = 0

    def next_frame(self, samples: int) -> bytes:
        """The next `samples` of the loop, as int16 little-endian bytes."""
        want = samples * 2
        out = bytearray()
        while len(out) < want:
            take = min(want - len(out), len(self._pcm) - self._pos)
            out += self._pcm[self._pos:self._pos + take]
            self._pos = (self._pos + take) % len(self._pcm)
        return bytes(out)

    @property
    def rms(self) -> float:
        a = np.frombuffer(self._pcm, dtype=np.int16).astype(np.float64) / 32768
        return float(np.sqrt(np.mean(a * a)))
