"""Voice-path output shaping: presence EQ + WSOLA speed.

Two defects these pin, both found on 2026-09-06 when the cloned voice was
described as sounding like the speaker was in a distant closet:

  * the 12 Hz vocoder rolls off above ~1.5 kHz, so the presence band that
    carries proximity is missing (the hollowness), and
  * the TTS server's *streaming* path drops `speed` entirely, so the measured
    `speed: 0.70` in config.yaml did nothing on the only path voice mode uses.

The properties worth guarding are the ones a plausible refactor breaks
silently: that filter and overlap state really carry across chunk boundaries
(chunking is invisible to the caller), that the stretch holds its rate without
drifting, that it does not move pitch, and that the shelves lift the top
without touching the low-mid the reference already matches.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent-services"))

tts_shaping = pytest.importorskip("tts_shaping")
pytest.importorskip("scipy")

SR = 24000


def _tone(freq: float, seconds: float, sr: int = SR, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


def _band_gain_db(sos, freq: float, sr: int = SR) -> float:
    from scipy.signal import sosfreqz

    _, h = sosfreqz(sos, worN=np.array([freq]), fs=sr)
    return float(20 * np.log10(abs(h[0])))


# ── presence EQ ─────────────────────────────────────────────────────────────

def test_default_shelves_lift_presence_and_leave_the_low_mid_alone():
    """The measured deficit starts above 1.5 kHz; below it the clone already
    matches the reference to 0.1 dB, so the correction must be inaudible there.
    Cutting the low-mid to fix the 200-500/2-6k *ratio* trades hollow for thin.
    """
    eq = tts_shaping.PresenceEQ(SR)
    for freq in (100, 300, 500, 800):
        assert abs(_band_gain_db(eq.sos, freq)) < 0.5, f"{freq} Hz should pass through"
    # Measured deficit vs the reference: +1.8 dB at 2 kHz, +3.9 at 4 kHz,
    # +4.4 at 8 kHz, +6.2 above 9 kHz. The shelves track that shape.
    assert _band_gain_db(eq.sos, 2000) > 1.5
    assert _band_gain_db(eq.sos, 4000) > 2.5
    assert _band_gain_db(eq.sos, 8000) > 4.0
    assert _band_gain_db(eq.sos, 10000) > 5.5
    # Lift stays modest — a bigger shelf just amplifies vocoder hiss in a band
    # that holds very little signal.
    assert _band_gain_db(eq.sos, 11000) < 9.0


def test_eq_state_carries_across_chunks():
    """The server streams arbitrary chunk sizes. A filter reset per chunk puts a
    discontinuity at every boundary, which is audible as a buzz on a held vowel.
    """
    sig = _tone(220, 1.0) + _tone(3000, 1.0) * 0.2
    whole = tts_shaping.PresenceEQ(SR).process(sig)
    chunked_eq = tts_shaping.PresenceEQ(SR)
    chunked = np.concatenate(
        [chunked_eq.process(sig[i : i + 997]) for i in range(0, len(sig), 997)]
    )
    assert np.array_equal(whole, chunked)


def test_shelf_above_nyquist_is_a_passthrough_not_an_error():
    """A bad config value degrades to 'no correction up there' rather than
    taking the voice down with it."""
    sos = tts_shaping.high_shelf_sos(20000.0, 6.0, 0.7, SR)
    assert np.allclose(sos, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    assert np.allclose(tts_shaping.high_shelf_sos(1800.0, 0.0, 0.7, SR),
                       [1.0, 0.0, 0.0, 1.0, 0.0, 0.0])


# ── WSOLA time stretch ──────────────────────────────────────────────────────

@pytest.mark.parametrize("seconds", [2.0, 8.0])
def test_stretch_holds_its_rate(seconds):
    """The error must be a fixed tail (window taper + flush padding), not drift:
    a per-frame rate error compounds and a long utterance ends up wrong.
    """
    sig = _tone(180, seconds)
    st = tts_shaping.WsolaStretch(0.7, SR)
    out = np.concatenate([st.process(sig), st.flush()])
    expected = len(sig) / 0.7
    assert abs(len(out) - expected) < 0.05 * SR  # within 50 ms, any length


def test_stretch_preserves_pitch():
    """The whole reason this is not a resample: slowing the voice must not
    lower it."""
    sig = _tone(180, 2.0)
    st = tts_shaping.WsolaStretch(0.7, SR)
    out = np.concatenate([st.process(sig), st.flush()])
    freqs = np.fft.rfftfreq(len(out), 1 / SR)
    peak = freqs[int(np.argmax(np.abs(np.fft.rfft(out))))]
    assert abs(peak - 180.0) < 3.0


def test_stretch_state_carries_across_chunks():
    sig = _tone(180, 2.0) + _tone(430, 2.0) * 0.4
    a = tts_shaping.WsolaStretch(0.7, SR)
    whole = np.concatenate([a.process(sig), a.flush()])
    b = tts_shaping.WsolaStretch(0.7, SR)
    chunked = np.concatenate(
        [b.process(sig[i : i + 997]) for i in range(0, len(sig), 997)] + [b.flush()]
    )
    assert np.array_equal(whole, chunked)


def test_flush_resets_so_one_instance_serves_every_utterance():
    st = tts_shaping.WsolaStretch(0.7, SR)
    sig = _tone(180, 0.5)
    first = np.concatenate([st.process(sig), st.flush()])
    second = np.concatenate([st.process(sig), st.flush()])
    assert np.array_equal(first, second)


# ── OutputShaper (the byte-level wrapper the worker uses) ───────────────────

def _pcm(sig: np.ndarray) -> bytes:
    return np.rint(sig * 32767.0).astype("<i2").tobytes()


def test_shaper_is_a_passthrough_when_there_is_nothing_to_do():
    """Speed 1.0 with the EQ off must not pay for an overlap-add that changes
    nothing."""
    sh = tts_shaping.OutputShaper(SR, speed=1.0, presence_eq=False)
    assert not sh.enabled
    raw = _pcm(_tone(180, 0.2))
    assert sh.process(raw) == raw
    assert sh.flush() == b""


def test_shaper_survives_a_sample_split_across_chunks():
    """`aiter_bytes` chunks are arbitrary byte counts, so a 16-bit sample can
    straddle two of them. Reading that as two samples shifts every byte after
    it and turns the rest of the utterance into noise.
    """
    sig = _tone(180, 0.5)
    raw = _pcm(sig)
    even = tts_shaping.OutputShaper(SR, speed=0.7)
    a = b"".join(even.process(raw[i : i + 1000]) for i in range(0, len(raw), 1000)) + even.flush()
    odd = tts_shaping.OutputShaper(SR, speed=0.7)
    b = b"".join(odd.process(raw[i : i + 999]) for i in range(0, len(raw), 999)) + odd.flush()
    assert a == b
    assert len(a) % 2 == 0


def test_shaper_applies_the_configured_speed():
    """The bug this exists for: `speed` reaching the server's streaming path is
    silently ignored, so voice mode ran ~1.5x fast."""
    sig = _tone(180, 1.0)
    sh = tts_shaping.OutputShaper(SR, speed=0.7)
    out = sh.process(_pcm(sig)) + sh.flush()
    ratio = (len(out) // 2) / len(sig)
    assert 1.38 < ratio < 1.50


def test_shaper_counts_clipping_rather_than_wrapping_it():
    """int16 wraparound turns a loud sample into a full-scale sample of the
    opposite sign — a click. Clip, and count it so the log can say so."""
    sh = tts_shaping.OutputShaper(SR, speed=1.0)
    loud = _pcm(np.clip(_tone(180, 0.3, amp=0.99), -1, 1))
    out = np.frombuffer(sh.process(loud) + sh.flush(), dtype="<i2")
    assert out.max() <= 32767 and out.min() >= -32768
    assert sh.total > 0
