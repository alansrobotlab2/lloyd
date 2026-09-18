"""The hearing pipeline: resampling, segmentation, wake attachment, endpointing.

These exist because the 2026-09-17 review could not measure the old pipeline at
all — every part of it was inline in `livekit_worker.RoomBridge` and needed a
LiveKit room to construct, so the only evidence that the wake word had fired
five times in 949 utterances was a log file nobody was reading. The properties
pinned here are the ones whose failure is silent:

  * a streaming resampler that emits a boundary transient still *sounds* fine
    and quietly degrades every model downstream;
  * a wake word fed per-utterance still fires sometimes, which is what made the
    original defect look like a tuning problem for three weeks;
  * an utterance that loses its wake detection is simply never answered.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent-services"))

pytest.importorskip("scipy")
voice = pytest.importorskip("voice")

from voice.pipeline import WAKE_ATTACH_GRACE, HearingPipeline  # noqa: E402
from voice.resample import FrameChunker, StreamResampler, to_float32, to_int16  # noqa: E402
from voice.vad import SileroSegmenter, Utterance  # noqa: E402

SR = 16000


# ── resampling ───────────────────────────────────────────────────────────

def test_streaming_matches_one_shot():
    """Frame-by-frame resampling must equal resampling the whole stream.

    The first implementation prepended history but emitted the tail, so every
    10 ms frame carried `resample_poly`'s right-edge zero-padding transient —
    6.7e-2 peak error, i.e. audible, and invisible to anything but this test.
    """
    from scipy.signal import resample_poly

    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.2, 48000 * 2).astype(np.float32)
    one_shot = resample_poly(x, 1, 3)

    r = StreamResampler(48000)
    out = [r.push(x[i:i + 480], 48000) for i in range(0, x.size, 480)]
    out.append(r.flush())
    streamed = np.concatenate(out)

    n = min(one_shot.size, streamed.size)
    assert n >= one_shot.size - 8, "streaming dropped more than the flush tail"
    assert np.max(np.abs(one_shot[:n] - streamed[:n])) < 1e-6


def test_resampler_is_a_passthrough_at_the_target_rate():
    r = StreamResampler(SR)
    x = np.linspace(-1, 1, 512, dtype=np.float32)
    assert np.array_equal(r.push(x, SR), x)
    assert r.delay_samples == 0


def test_resampler_handles_a_rate_change_without_raising():
    r = StreamResampler(48000)
    r.push(np.zeros(480, dtype=np.int16), 48000)
    out = r.push(np.zeros(441, dtype=np.int16), 44100)
    assert r.in_rate == 44100
    assert out.dtype == np.float32


def test_int16_roundtrip_clips_rather_than_wrapping():
    """A wrapped sample is a full-scale sample of the opposite sign — a click."""
    loud = np.array([2.0, -2.0], dtype=np.float32)
    assert to_int16(loud).tolist() == [32767, -32768]
    assert np.allclose(to_float32(np.array([16384], dtype=np.int16)), [0.5])


def test_chunker_reblocks_without_losing_samples():
    c = FrameChunker(512)
    got = []
    for i in range(10):
        got += c.push(np.full(300, i, dtype=np.float32))
    assert all(f.size == 512 for f in got)
    assert len(got) == (10 * 300) // 512


# ── segmentation ─────────────────────────────────────────────────────────

class _ScriptedVad:
    """Stands in for Silero so segmentation is tested against a known script.

    The real model is exercised by the wake tests below; here the point is the
    state machine, and a test that depended on Silero's exact probabilities
    would be pinning Silero rather than this code.
    """

    def __init__(self, probs):
        self._probs = list(probs)
        self._i = 0

    def __call__(self, frame):
        p = self._probs[min(self._i, len(self._probs) - 1)]
        self._i += 1
        return p

    def reset_states(self):
        self._i = 0


def _segmenter(probs, **kw):
    kw.setdefault("min_silence_ms", 96)     # 3 frames
    kw.setdefault("speech_pad_ms", 32)      # 1 frame
    kw.setdefault("min_utterance_ms", 32)
    return SileroSegmenter(model=_ScriptedVad(probs), **kw)


def _feed(seg, n_frames):
    rng = np.random.default_rng(1)
    return seg.feed(rng.normal(0, 0.1, 512 * n_frames).astype(np.float32))


def test_a_span_of_speech_closes_after_the_silence_threshold():
    seg = _segmenter([0.0] * 2 + [0.9] * 5 + [0.0] * 5)
    utts = _feed(seg, 12)
    assert len(utts) == 1
    assert utts[0].reason == "silence"
    assert utts[0].max_prob == pytest.approx(0.9)


def test_a_pause_shorter_than_the_threshold_does_not_split_the_turn():
    """One sentence with a breath in it is one utterance, not two."""
    seg = _segmenter([0.0] * 2 + [0.9] * 3 + [0.0] * 2 + [0.9] * 3 + [0.0] * 5)
    utts = _feed(seg, 15)
    assert len(utts) == 1


def test_quiet_speech_is_not_dropped_for_being_quiet():
    """The defect this whole module replaces.

    The energy VAD kept `speech_rms: 0.025` while real wake-word utterances
    measured 0.013-0.017, so normal speech at a normal distance was discarded
    as silence. Loudness must not enter the decision at all.
    """
    seg = _segmenter([0.0] * 2 + [0.9] * 5 + [0.0] * 5)
    quiet = (np.random.default_rng(2).normal(0, 0.0004, 512 * 12)).astype(np.float32)
    utts = seg.feed(quiet)
    assert len(utts) == 1, "a quiet but voiced span must still segment"


def test_the_preroll_is_kept_so_the_first_phoneme_survives():
    seg = _segmenter([0.0] * 3 + [0.9] * 4 + [0.0] * 5, speech_pad_ms=64)
    utts = _feed(seg, 12)
    assert len(utts) == 1
    # Onset is frame 3; two frames of pad must appear ahead of it.
    assert utts[0].start_sample <= 3 * 512


def test_a_capped_utterance_continues_rather_than_dropping_the_next_word():
    seg = _segmenter([0.9] * 40, max_utterance_ms=64, min_utterance_ms=0)
    utts = _feed(seg, 20)
    assert len(utts) >= 2
    assert all(u.reason == "max_duration" for u in utts)
    assert seg.in_speech, "speech must stay open across the cap"


def test_flush_returns_a_half_spoken_utterance():
    seg = _segmenter([0.0] * 2 + [0.9] * 6)
    assert _feed(seg, 8) == []
    tail = seg.flush()
    assert tail is not None and tail.reason == "flush"
    assert seg.flush() is None


# ── wake attachment ──────────────────────────────────────────────────────

def _utt(start, end):
    return Utterance(audio=np.zeros(end - start, dtype=np.float32),
                     start_sample=start, end_sample=end, max_prob=0.9,
                     reason="silence")


def test_a_wake_inside_an_utterance_is_attached_to_it():
    assert _utt(0, SR).contains(SR // 2)


def test_a_wake_just_after_the_close_is_still_attached():
    """openWakeWord stamps a fire at the end of the frame that crossed the
    threshold, and needs ~1 s of context to get there — so a short "hey Lloyd"
    can be closed by the VAD a frame or two before the score peaks."""
    u = _utt(0, SR)
    assert u.contains(SR + WAKE_ATTACH_GRACE - 1, grace=WAKE_ATTACH_GRACE)
    assert not u.contains(SR + WAKE_ATTACH_GRACE + 1, grace=WAKE_ATTACH_GRACE)


class _FakeWake:
    """Fires once, at a scripted stream position."""

    def __init__(self, fire_at_sample=None):
        self.fire_at = fire_at_sample
        self._cursor = 0
        self.threshold = 0.4

    def feed(self, audio):
        from voice.wake import WakeDetection

        self._cursor += audio.size
        if self.fire_at is not None and self._cursor >= self.fire_at:
            self.fire_at = None
            return [WakeDetection("Lloyd", 0.8, self._cursor, 0.0)]
        return []

    def take_peak(self):
        return ("Lloyd", 0.05)


def test_the_pipeline_reports_a_wake_before_the_utterance_closes():
    """The latency win. The old worker could not emit a wake until the VAD had
    closed and Whisper had returned; the UI flipped to "Listening" roughly half
    a second after the user stopped speaking."""
    pipe = HearingPipeline(SR, wake=_FakeWake(fire_at_sample=512),
                           segmenter=_segmenter([0.0] * 2 + [0.9] * 5 + [0.0] * 5))
    rng = np.random.default_rng(3)
    events = []
    for i in range(12):
        events += pipe.feed(rng.normal(0, 0.1, 512).astype(np.float32), SR)
    kinds = [e.kind for e in events]
    assert kinds.index("wake") < kinds.index("utterance")


def test_an_utterance_carries_the_wake_that_fell_inside_it():
    pipe = HearingPipeline(SR, wake=_FakeWake(fire_at_sample=512 * 4),
                           segmenter=_segmenter([0.0] * 2 + [0.9] * 5 + [0.0] * 5))
    rng = np.random.default_rng(4)
    events = []
    for i in range(12):
        events += pipe.feed(rng.normal(0, 0.1, 512).astype(np.float32), SR)
    utt = [e for e in events if e.kind == "utterance"]
    assert len(utt) == 1 and utt[0].wake is not None


def test_a_stale_wake_does_not_open_a_later_unrelated_sentence():
    """Without expiry, a wake that produced no utterance would attach itself to
    whatever anyone said next — a minute later, across the room."""
    from voice.pipeline import WAKE_MAX_AGE
    from voice.wake import WakeDetection

    pipe = HearingPipeline(SR, wake=None, segmenter=_segmenter([0.9] * 3 + [0.0] * 5))
    pipe._pending.append(WakeDetection("Lloyd", 0.9, -WAKE_MAX_AGE * 2, 0.0))
    rng = np.random.default_rng(5)
    events = []
    for i in range(10):
        events += pipe.feed(rng.normal(0, 0.1, 512).astype(np.float32), SR)
    utt = [e for e in events if e.kind == "utterance"]
    assert utt and utt[0].wake is None


def test_speech_start_is_reported_for_barge_in():
    pipe = HearingPipeline(SR, wake=None,
                           segmenter=_segmenter([0.0] * 2 + [0.9] * 8))
    rng = np.random.default_rng(6)
    events = []
    for i in range(10):
        events += pipe.feed(rng.normal(0, 0.1, 512).astype(np.float32), SR)
    assert any(e.kind == "speech" for e in events)


def test_the_wake_peak_reported_for_a_drop_belongs_to_that_utterance():
    """The rest of an earlier "hey Lloyd" keeps scoring after its fire, inside
    the refractory window. Carried over, it was reported against the next,
    unaddressed sentence as a 0.94 — a diagnostic that pointed at the wrong
    utterance."""
    class _Wake:
        def __init__(self):
            self.peak = 0.94
            self.reads = 0

        def feed(self, audio):
            return []

        def take_peak(self):
            self.reads += 1
            out, self.peak = self.peak, 0.0
            return ("Hey_Lloyd", out)

    w = _Wake()
    pipe = HearingPipeline(SR, wake=w, segmenter=_segmenter([0.0] * 2 + [0.9] * 5))
    rng = np.random.default_rng(7)
    for _ in range(8):
        pipe.feed(rng.normal(0, 0.1, 512).astype(np.float32), SR)
    assert w.reads == 1 and w.peak == 0.0


def test_silero_onnx_matches_the_reference():
    """The VAD calls Silero's ONNX model through onnxruntime directly, because
    the `silero-vad` package requires torchaudio, which pins torch — installing
    it on 2026-09-17 moved `.venvs/lloyd` from torch 2.11.0 (CUDA 13) to 2.9.1
    under every service in that venv. The fixture is the package's own output
    on this clip, saved before it was removed; the numpy wrapper must
    reproduce it, recurrent state and 64-sample context included."""
    import wave

    from voice.vad import MODEL_PATH, SileroOnnx

    if not MODEL_PATH.exists():
        pytest.skip("vendored silero model missing")
    fx = Path(__file__).resolve().parent / "fixtures" / "voice"
    with wave.open(str(fx / "complete_16k.wav")) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = audio.astype(np.float32) / 32768.0
    reference = np.load(fx / "silero_reference_complete_16k.npy")
    model = SileroOnnx()
    ours = np.array([model(audio[i:i + 512])
                     for i in range(0, audio.size - 511, 512)], dtype=np.float32)
    assert ours.shape == reference.shape
    assert np.max(np.abs(ours - reference)) < 1e-5


def test_the_vad_imports_no_torch():
    """Keep it that way: see the test above for what the torch coupling cost."""
    import ast

    src = (Path(__file__).resolve().parent.parent / "agent-services" / "voice" / "vad.py").read_text()
    imported = {n.names[0].name.split(".")[0] for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.Import)}
    imported |= {n.module.split(".")[0] for n in ast.walk(ast.parse(src))
                 if isinstance(n, ast.ImportFrom) and n.module}
    assert not imported & {"torch", "torchaudio", "silero_vad"}
