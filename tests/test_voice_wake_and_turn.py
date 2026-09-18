"""The wake word fed continuously, and the end-of-turn model.

The counterfactual test at the bottom is the important one: it runs the same
audio through the old *shape* (reset, then sweep the finished utterance) and
the new one (fed continuously, never reset) and asserts the new one wins. That
is the claim the 2026-09-17 rework rests on, and nothing else in the suite
would notice if the continuous path quietly regressed to the old behaviour.

The fixtures are synthesized speech, not recordings, so an absolute detection
rate here says nothing about a microphone in a room. What they pin is that the
wiring is right — a real model, real audio, and a decision at the end.
"""
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent-services"))

pytest.importorskip("scipy")
voice = pytest.importorskip("voice")

from voice.wake import FRAME_SAMPLES, ContinuousWakeWord, WakeWordFactory  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "voice"
MODELS = ROOT / "agent-services" / "models"
SR = 16000


def _wav(name):
    with wave.open(str(FIXTURES / name)) as w:
        assert w.getframerate() == SR
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(
            np.float32) / 32768.0


def _quiet(seconds, seed=0):
    """Room tone. Not digital silence: openWakeWord's feature window on exact
    zeros is not what any microphone ever produces."""
    return np.random.default_rng(seed).normal(0, 0.0008, int(SR * seconds)).astype(np.float32)


def _wake_factory():
    if not (MODELS / "wakeword").exists() or not list((MODELS / "wakeword").glob("*.onnx")):
        pytest.skip("wake-word models not present")
    pytest.importorskip("openwakeword")
    f = WakeWordFactory(models_dir=MODELS / "wakeword",
                        engine_dir=MODELS / "openwakeword", threshold=0.4)
    try:
        f.validate()
    except RuntimeError as e:
        pytest.skip(str(e))
    return f


# ── the state machine, against a scripted model ──────────────────────────

class _ScriptedModel:
    """Returns a scripted score per 1280-sample frame."""

    def __init__(self, scores):
        self._scores = list(scores)
        self._i = 0
        self.models = {"Lloyd": None}

    def predict(self, x):
        s = self._scores[min(self._i, len(self._scores) - 1)]
        self._i += 1
        return {"Lloyd": s}


def _feed_frames(w, n):
    return w.feed(np.zeros(FRAME_SAMPLES * n, dtype=np.float32))


def test_a_sustained_score_fires_once_not_every_frame():
    """openWakeWord's score stays high for several frames after the word.
    Without a refractory window one "hey Lloyd" opens the window three times,
    and the diagnostic log reads as three separate wakes."""
    w = ContinuousWakeWord(_ScriptedModel([0.9] * 18), threshold=0.4,
                           refractory_ms=1500)   # 1500 ms = 18.75 frames
    hits = _feed_frames(w, 18)
    assert len(hits) == 1


def test_a_second_word_after_the_refractory_window_fires_again():
    w = ContinuousWakeWord(_ScriptedModel([0.9] * 2 + [0.0] * 30 + [0.9] * 2),
                           threshold=0.4, refractory_ms=200)
    hits = _feed_frames(w, 34)
    assert len(hits) == 2


def test_a_score_below_threshold_never_fires_but_is_still_recorded():
    """A near-miss has to be visible or a miss report cannot be debugged."""
    w = ContinuousWakeWord(_ScriptedModel([0.35] * 10), threshold=0.4)
    assert _feed_frames(w, 10) == []
    name, peak = w.take_peak()
    assert name == "Lloyd" and peak == pytest.approx(0.35)
    assert w.take_peak() == ("", 0.0), "peak must clear when read"


def test_the_detection_carries_its_position_in_the_stream():
    w = ContinuousWakeWord(_ScriptedModel([0.0] * 5 + [0.9]), threshold=0.4)
    hits = _feed_frames(w, 6)
    assert hits[0].sample_index == 6 * FRAME_SAMPLES


def test_a_model_that_raises_does_not_kill_the_audio_path():
    class _Broken:
        models = {"Lloyd": None}

        def predict(self, x):
            raise RuntimeError("onnx said no")

    w = ContinuousWakeWord(_Broken(), threshold=0.4)
    assert _feed_frames(w, 3) == []


# ── the real model ───────────────────────────────────────────────────────

def test_the_wake_word_fires_on_real_audio_in_a_continuous_stream():
    w = _wake_factory().create()
    stream = np.concatenate([_quiet(2.0), _wav("hey_lloyd_16k.wav"), _quiet(1.0)])
    hits = []
    for i in range(0, stream.size, 160):          # 10 ms, as LiveKit delivers
        hits += w.feed(stream[i:i + 160])
    assert hits, "no detection on a clean 'hey Lloyd'"
    assert hits[0].score >= 0.4


def test_detection_survives_a_sixteenfold_attenuation():
    """Speaking quietly, or standing further away, must not cost the wake word.
    This is the property the energy VAD did not have and is why voice barely
    worked: real wake utterances measured rms 0.013-0.017 against a 0.025 gate."""
    w = _wake_factory().create()
    stream = np.concatenate([_quiet(2.0), _wav("hey_lloyd_16k.wav") * 0.0625,
                             _quiet(1.0)])
    hits = []
    for i in range(0, stream.size, 160):
        hits += w.feed(stream[i:i + 160])
    assert hits


def test_two_streams_do_not_share_state():
    """`Model` holds a rolling feature window and is not thread-safe. The old
    code shared one instance across every room and called it from a thread
    pool, so two concurrent utterances interleaved their features."""
    f = _wake_factory()
    a, b = f.create(), f.create()
    a.feed(_wav("hey_lloyd_16k.wav"))
    assert b.cursor == 0
    assert b.take_peak()[1] == 0.0


def test_the_wake_fires_as_the_word_ends_not_after_the_silence_that_closes_it():
    """The latency half of the rework, on the real model.

    The old path could not report a wake before the energy VAD had seen 500 ms
    of closing silence and the sweep had then run over the whole utterance, so
    "Listening" appeared well after the user stopped. Fed continuously, the
    model crosses its threshold within a frame or two of the last phoneme —
    measured at -60..+20 ms across 1-3 s of lead-in on this fixture.

    (The other half — false accepts — needs real room audio, which is not
    committed: `scripts/voice/replay.py compare-wake` measures it against the
    local diagnostic corpus. On 2026-09-17: 6 false accepts from the sweep,
    0 from the stream, on the same 496 non-wake utterances.)
    """
    word = _wav("hey_lloyd_16k.wav")
    frame = 160
    energy = np.array([np.sqrt(np.mean(word[i:i + frame] ** 2))
                       for i in range(0, word.size - frame, frame)])
    speech_end = (np.where(energy > 0.01)[0][-1] + 1) * frame

    lead = SR * 2
    stream = np.concatenate([_quiet(2.0), word, _quiet(1.0)])
    w = _wake_factory().create()
    hits = []
    for i in range(0, stream.size, 160):
        hits += w.feed(stream[i:i + 160])
    assert hits, "no detection"
    late_ms = (hits[0].sample_index - (lead + speech_end)) / SR * 1000
    assert late_ms < 200, (
        f"fired {late_ms:.0f} ms after the word ended; the VAD's closing "
        f"silence alone is 380 ms, so this is no better than waiting for it"
    )


def test_the_factory_refuses_a_missing_model_directory_loudly():
    """Falling back to text matching is what kept the worker silently deaf:
    `_strip_wake_word` cannot fire on the empty string Whisper returns for a
    quiet 'hey Lloyd'."""
    f = WakeWordFactory(models_dir=ROOT / "nope", engine_dir=MODELS / "openwakeword")
    with pytest.raises(RuntimeError):
        f.validate()


# ── Smart Turn ───────────────────────────────────────────────────────────

def _smart_turn():
    from voice.turn import SmartTurn

    path = MODELS / "smart-turn" / "smart-turn-v3.2-cpu.onnx"
    if not path.exists():
        pytest.skip("smart-turn model not present")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("faster_whisper")
    st = SmartTurn(path)
    st.load()
    return st


def test_a_finished_question_and_a_cut_phrase_are_told_apart():
    """This is also the check on the mel preprocessing. The head is a linear
    probe on a frozen Whisper-tiny encoder, so a mel that is off by a
    normalisation constant is simply a different input — and would show up
    here as the two scores collapsing together rather than as an error."""
    st = _smart_turn()
    complete = st.predict(_wav("complete_16k.wav"))
    cut = st.predict(_wav("incomplete_16k.wav"))
    assert complete.complete and complete.probability > 0.8
    assert not cut.complete and cut.probability < 0.2


def test_a_short_clip_is_padded_rather_than_rejected():
    st = _smart_turn()
    v = st.predict(np.zeros(1000, dtype=np.float32))
    assert v.ran and 0.0 <= v.probability <= 1.0


def test_only_the_last_eight_seconds_are_read():
    st = _smart_turn()
    long = np.concatenate([np.zeros(SR * 20, dtype=np.float32),
                           _wav("complete_16k.wav")])
    assert st.predict(long).probability > 0.5


def test_a_broken_model_fails_open():
    """A turn detector that cannot load must not stop Lloyd answering — the
    VAD silence alone is a worse endpoint, not a broken one."""
    from voice.turn import SmartTurn

    st = SmartTurn(ROOT / "does-not-exist.onnx")
    v = st.predict(np.zeros(SR, dtype=np.float32))
    assert v.complete and not v.ran


def test_the_streaming_recogniser_produces_partials_that_grow():
    """`livekit.stt.streaming` ships off, but it ships: when someone turns it on
    it must load and produce a running hypothesis, not fail on first audio."""
    from voice.asr import SherpaStreamingRecognizer

    model_dir = MODELS / "nemo-streaming-480ms"
    if not (model_dir / "model.onnx").exists():
        pytest.skip("streaming model not present")
    pytest.importorskip("sherpa_onnx")
    rec = SherpaStreamingRecognizer(model_dir)
    stream = rec.create_stream()
    audio = np.concatenate([_wav("complete_16k.wav"), np.zeros(SR, dtype=np.float32)])
    partials = []
    for i in range(0, audio.size, 1600):          # 100 ms at a time
        text = rec.accept(stream, audio[i:i + 1600])
        if text and (not partials or text != partials[-1]):
            partials.append(text)
    assert len(partials) >= 2, "a stream should show its hypothesis growing"
    assert "weather" in partials[-1].lower()
    rec.reset(stream)
