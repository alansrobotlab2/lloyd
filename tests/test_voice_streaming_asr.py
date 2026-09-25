"""The streaming recogniser's transducer path and its end-of-utterance token.

`livekit.stt.streaming` gained a second export layout on 2026-09-24: a
transducer (`encoder/decoder/joiner.onnx`), which is what
parakeet_realtime_eou_120m and the Nemotron streaming models ship as, beside
the original CTC `model.onnx`. The EOU model also emits `<EOU>` in its token
stream. Two things must hold whichever model is configured: no control token
ever reaches a caption, and `endpointed` reports the token the model emitted.

The model tests skip when the (untracked) model directory is absent, like
every other test that needs a downloaded voice model.
"""
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent-services"))

from voice import asr  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "voice"
MODELS = ROOT / "agent-services" / "models"
EOU_DIR = MODELS / "parakeet-realtime-eou-120m"
SR = 16000


def _wav(name):
    with wave.open(str(FIXTURES / name)) as w:
        assert w.getframerate() == SR
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(
            np.float32) / 32768.0


def _floor(seconds, seed=0):
    return np.random.default_rng(seed).normal(0, 1e-3, int(seconds * SR)).astype(np.float32)


def test_control_tokens_never_reach_the_text():
    assert asr.strip_control_tokens("what is the weather<EOU>") == "what is the weather"
    assert asr.strip_control_tokens("yeah<EOB> so<EOU> then") == "yeah so then"
    assert asr.strip_control_tokens("") == ""


def test_the_config_block_reaches_the_recogniser(monkeypatch):
    """precision and language are read, and `enabled: false` builds nothing."""
    assert asr.build_streaming_recognizer({"streaming": {"enabled": False}}) is None
    seen = {}
    monkeypatch.setattr(asr.SherpaStreamingRecognizer, "load", lambda self: seen.update(
        dir=self.model_dir, precision=self.precision, language=self.language,
        threads=self.num_threads))
    asr.build_streaming_recognizer({"streaming": {
        "enabled": True, "model_dir": "/x/eou", "precision": "fp32",
        "language": "en", "num_threads": 3}})
    assert seen == {"dir": Path("/x/eou"), "precision": "fp32", "language": "en",
                    "threads": 3}


def test_the_precision_picks_the_file_and_falls_back(tmp_path):
    for f in ("encoder.onnx", "encoder.int8.onnx", "decoder.onnx"):
        (tmp_path / f).write_bytes(b"")
    r = asr.SherpaStreamingRecognizer(tmp_path, precision="fp32")
    assert r._pick("encoder").endswith("/encoder.onnx")
    assert asr.SherpaStreamingRecognizer(tmp_path)._pick("encoder").endswith("encoder.int8.onnx")
    # Only fp32 present: int8 asked, fp32 served rather than failing.
    assert asr.SherpaStreamingRecognizer(tmp_path)._pick("decoder").endswith("/decoder.onnx")
    with pytest.raises(RuntimeError):
        r._pick("joiner")


@pytest.fixture(scope="module")
def eou():
    if not (EOU_DIR / "tokens.txt").exists():
        pytest.skip("parakeet-realtime-eou-120m not present")
    pytest.importorskip("sherpa_onnx")
    rec = asr.SherpaStreamingRecognizer(EOU_DIR, precision="fp32")
    rec.load()
    return rec


def test_the_eou_model_is_a_transducer_with_the_token(eou):
    assert eou.kind == "transducer" and eou.has_eou


def test_eou_partials_grow_and_the_utterance_ends(eou):
    """A finished question, then a second of room floor: the hypothesis grows,
    the end is reported, and the marker is not in the text."""
    stream = eou.create_stream()
    # With the pipeline's pre-roll: without it the stream loses "what is"
    # and the model never calls the remainder a finished question.
    audio = np.concatenate([_floor(0.5), _wav("complete_16k.wav"), _floor(1.2)])
    partials, ended_at = [], None
    for i in range(0, audio.size, 320):           # 20 ms, as the pipeline feeds
        text = eou.accept(stream, audio[i:i + 320])
        if text and (not partials or text != partials[-1]):
            partials.append(text)
        if ended_at is None and eou.endpointed(stream):
            ended_at = i / SR
    assert len(partials) >= 2
    assert partials[-1].lower().startswith("what is the weather")
    assert all("<" not in p for p in partials)
    assert ended_at is not None, "no <EOU> after a finished question and 1.2 s of quiet"
    eou.reset(stream)
    assert not eou.endpointed(stream), "reset must clear the end-of-utterance state"


def test_a_cut_phrase_is_not_ended_while_it_is_still_being_spoken(eou):
    """The unfinished fixture, fed only while it is being spoken: whatever the
    model decides after the speaker stops, it must not have called the end
    before the words were out."""
    stream = eou.create_stream()
    audio = np.concatenate([_floor(0.5), _wav("incomplete_16k.wav")])
    speech_end = int(np.nonzero(np.abs(audio) > 0.02)[0][-1])
    for i in range(0, speech_end - 3200, 320):    # stop 200 ms short of the end
        eou.accept(stream, audio[i:i + 320])
        assert not eou.endpointed(stream), f"<EOU> at {i / SR:.2f}s, mid-phrase"


class _RecordingStream:
    """A partials recogniser that only records what it is handed."""

    def __init__(self):
        self.fed = []

    def create_stream(self):
        return object()

    def accept(self, stream, audio):
        self.fed.append(np.asarray(audio).size)
        return "x"

    def reset(self, stream):
        pass


def test_a_new_partials_stream_is_handed_the_audio_before_the_onset():
    """The stream opens only once the VAD has heard speech, and a streaming
    encoder drops the first ~0.3 s it is given — so the frame that opens it
    must carry the pre-roll, or every caption loses its first word."""
    pytest.importorskip("scipy")
    from voice.pipeline import STREAM_PREROLL_S, HearingPipeline

    rec = _RecordingStream()
    pipe = HearingPipeline(SR, streaming=rec)
    audio = np.concatenate([_floor(1.0), _wav("complete_16k.wav"), _floor(1.0)])
    for i in range(0, audio.size, 320):
        pipe.feed(audio[i:i + 320], SR)
    assert rec.fed, "the pipeline never opened a partials stream"
    assert rec.fed[0] >= int(0.9 * STREAM_PREROLL_S * SR), rec.fed[:3]
    assert max(rec.fed[1:]) <= 320 * 2, "only the opening frame carries the pre-roll"


def test_the_pre_roll_restores_the_first_word(eou):
    """The counterfactual on the real model: the same question fed from its
    first sample loses "what is"; fed with the pipeline's pre-roll it does not."""
    speech = _wav("complete_16k.wav")

    def run(pre):
        stream = eou.create_stream()
        audio = np.concatenate([pre, speech, _floor(1.0)])
        for i in range(0, audio.size, 320):
            eou.accept(stream, audio[i:i + 320])
        return eou.finish(stream).lower()

    assert run(_floor(0.5)).startswith("what is the weather")
    assert not run(np.zeros(0, np.float32)).startswith("what")


def test_words_after_an_eou_are_still_transcribed(eou):
    """sherpa leaves the prediction network as it was at `<EOU>`, and this
    model then emits nothing more for the rest of the stream (a 45-word
    LibriSpeech clip came out "happy"). Two questions in one stream, a pause
    between: both must be in the text, and the end reported."""
    q = _wav("complete_16k.wav")
    audio = np.concatenate([_floor(0.5), q, _floor(1.5), q, _floor(1.2)])
    stream = eou.create_stream()
    for i in range(0, audio.size, 320):
        eou.accept(stream, audio[i:i + 320])
    text = eou.finish(stream).lower()
    assert eou.endpointed(stream)
    assert text.count("weather") == 2, text
