"""The TTS bake-off harness's pure parts: pacing, the first-clause cut, and the
sub-1 kHz-gated presence measurement (scripts/voice/tts_bakeoff.py)."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("tts_bakeoff", ROOT / "scripts/voice/tts_bakeoff.py")
bake = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bake)


def test_llm_tokens_are_lossless_and_bpe_sized():
    for s in bake.SENTENCES:
        toks = bake.llm_tokens(s)
        assert "".join(toks) == s
        assert 2.5 < len(s) / len(toks) < 6


def test_first_clause_cut_matches_the_worker():
    i, clause = bake.first_clause_cut(bake.llm_tokens("Honestly? I'd wait until the morning."))
    assert clause == "Honestly?"
    assert i < 4


def _voice(sr, top_gain):
    t = np.arange(sr * 2) / sr
    low = sum(np.sin(2 * np.pi * f * t) for f in (150, 300, 600, 900))
    high = sum(np.sin(2 * np.pi * f * t) for f in (2000, 3000, 4000, 6000))
    env = (np.sin(2 * np.pi * 3 * t) > 0).astype(float)  # voiced / silent frames
    return ((low + top_gain * high) * env * 0.05).astype(np.float32)


def test_presence_reads_a_missing_top_as_negative_and_ignores_rate():
    ref = bake.band_profile(_voice(44100, 1.0), 44100)
    dull = bake.band_profile(_voice(24000, 0.5), 24000)
    d = bake.presence_delta(dull, ref)
    assert -7.5 < d["mean_1.5-9k"] < -4.5          # 0.5x amplitude = -6 dB
    same = bake.presence_delta(bake.band_profile(_voice(24000, 1.0), 24000), ref)
    assert abs(same["mean_1.5-9k"]) < 1.0


def test_gate_is_on_low_band_energy_not_total():
    # Loud fricative-like frames (energy only above 2 kHz) must not move the
    # result: they carry no sub-1 kHz energy, so the gate drops them. A total-
    # RMS gate would keep them and read the top as several dB hotter.
    sr = 24000
    x = _voice(sr, 1.0)
    t = np.arange(sr) / sr
    ramp = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.1)  # no onset click
    fric = (0.4 * ramp * sum(np.sin(2 * np.pi * f * t) for f in (2600, 3000, 3300))).astype(np.float32)
    a = bake.band_profile(x, sr)["2500-3500"]
    gap = np.zeros(sr // 5, np.float32)
    b = bake.band_profile(np.concatenate([x, gap, fric]), sr)["2500-3500"]
    assert abs(a - b) < 1.0
    ungated = bake.band_profile(np.concatenate([x, gap, fric]), sr, gate_db=300)["2500-3500"]
    assert ungated - a > 5.0
