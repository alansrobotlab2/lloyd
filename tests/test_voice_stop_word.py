"""The stop word: a second acoustic detector, armed only while Lloyd speaks.

The state machine is pinned against a scripted model — disarmed reports
nothing, arming over a score that is already high does not replay the user's
last word as an interrupt, a sustained score fires once. The config half pins
that the stop model can never be loaded as a *wake* word (the wake factory
names its models; a stray file in the directory is not one of them). The real
model tests skip when `lloyd_stop.onnx` is absent: it is a runtime asset.
"""
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent-services"))

pytest.importorskip("scipy")
voice = pytest.importorskip("voice")

from voice.wake import (FRAME_SAMPLES, StopDetector, WakeWordFactory,  # noqa: E402
                        build_factory, build_stop_factory)

MODELS = ROOT / "agent-services" / "models"
STOP_MODEL = MODELS / "wakeword" / "stop" / "lloyd_stop.onnx"
FIXTURES = ROOT / "tests" / "fixtures" / "voice"
SR = 16000


class _Scripted:
    def __init__(self, scores):
        self._scores = list(scores)
        self._i = 0
        self.calls = 0
        self.models = {"lloyd_stop": None}

    def predict(self, x):
        self.calls += 1
        s = self._scores[min(self._i, len(self._scores) - 1)]
        self._i += 1
        return {"lloyd_stop": s}


def _frames(d, n):
    return d.feed(np.zeros(FRAME_SAMPLES * n, dtype=np.float32))


def test_a_disarmed_detector_reports_nothing_but_keeps_listening():
    m = _Scripted([0.95] * 10)
    d = StopDetector(m, threshold=0.8)
    assert _frames(d, 10) == []
    assert m.calls == 10, "fed while disarmed, so the window is warm when armed"


def test_arming_then_a_crossing_fires_once():
    d = StopDetector(_Scripted([0.1] * 3 + [0.9] * 12), threshold=0.8)
    d.arm()
    hits = _frames(d, 15)
    assert len(hits) == 1
    assert hits[0].name == "lloyd_stop" and hits[0].score == pytest.approx(0.9)
    assert hits[0].sample_index == 4 * FRAME_SAMPLES


def test_arming_over_a_high_score_does_not_replay_the_users_last_word():
    """The user said "…wait", Lloyd started answering: the score is still above
    threshold when the detector is armed. It must dip before it can fire."""
    d = StopDetector(_Scripted([0.9] * 4 + [0.9] * 4 + [0.1] * 2 + [0.9] * 2),
                     threshold=0.8, refractory_ms=0)
    _frames(d, 4)
    d.arm()
    assert _frames(d, 4) == []
    assert _frames(d, 2) == []
    assert len(_frames(d, 2)) == 1


def test_disarming_stops_reporting_immediately():
    d = StopDetector(_Scripted([0.1, 0.9, 0.1, 0.9]), threshold=0.8, refractory_ms=0)
    d.arm()
    assert len(_frames(d, 2)) == 1
    d.disarm()
    assert _frames(d, 2) == []


def test_the_refractory_window_holds_a_second_crossing():
    d = StopDetector(_Scripted([0.9, 0.1, 0.9, 0.1] + [0.1] * 20 + [0.9]),
                     threshold=0.8, refractory_ms=1000)
    d.arm()
    assert len(_frames(d, 4)) == 1          # the second crossing is 160 ms later
    assert len(_frames(d, 21)) == 1         # ~2 s later, outside the window


def test_feed_when_disarmed_false_skips_the_model():
    m = _Scripted([0.9] * 5)
    d = StopDetector(m, threshold=0.8, feed_when_disarmed=False)
    _frames(d, 5)
    assert m.calls == 0


def test_the_threshold_is_the_detectors_own():
    d = StopDetector(_Scripted([0.75] * 3), threshold=0.8)
    d.arm()
    assert _frames(d, 3) == []
    d2 = StopDetector(_Scripted([0.75] * 3), threshold=0.7)
    d2.arm()
    assert len(_frames(d2, 3)) == 1


# ── config ───────────────────────────────────────────────────────────────

def test_stop_is_off_unless_asked():
    assert build_stop_factory({}) is None
    assert build_stop_factory({"stop": {"enabled": False}}) is None
    assert build_stop_factory({"enabled": False, "stop": {"enabled": True}}) is None


def test_an_enabled_stop_word_with_no_model_is_a_boot_error(tmp_path):
    with pytest.raises(RuntimeError):
        build_stop_factory({"engine_dir": str(MODELS / "openwakeword"),
                            "stop": {"enabled": True, "model": str(tmp_path / "nope.onnx")}})


def test_named_wake_models_ignore_anything_else_in_the_directory(tmp_path):
    for n in ("hey_lloyd.onnx", "Lloyd.onnx", "lloyd_stop.onnx"):
        (tmp_path / n).write_bytes(b"x")
    f = WakeWordFactory(models_dir=tmp_path, engine_dir=MODELS / "openwakeword",
                        models=["hey_lloyd.onnx", "Lloyd.onnx"])
    if not (MODELS / "openwakeword" / "melspectrogram.onnx").exists():
        pytest.skip("openWakeWord engine files not present")
    f.validate()
    assert [Path(p).name for p in f._paths] == ["hey_lloyd.onnx", "Lloyd.onnx"]
    g = WakeWordFactory(models_dir=tmp_path, engine_dir=MODELS / "openwakeword")
    g.validate()
    assert len(g._paths) == 3, "no `models:` keeps the old glob"
    with pytest.raises(RuntimeError, match="not found"):
        WakeWordFactory(models_dir=tmp_path, engine_dir=MODELS / "openwakeword",
                        models=["gone.onnx"]).validate()


def test_config_never_loads_the_stop_model_as_a_wake_word():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())["livekit"]["acoustic_wake"]
    models = cfg.get("models")
    assert models, "acoustic_wake names its models explicitly"
    stop = cfg["stop"]
    assert Path(stop["model"]).name not in {Path(m).name for m in models}
    # On since 2026-09-24, wired: livekit_worker arms it only while Lloyd
    # speaks (tests/test_voice_duplex.py pins the arming and the stop).
    assert stop["enabled"] is True
    # The stop model lives outside the wake directory's top level, so even the
    # old glob (an older worker reading this tree) cannot pick it up.
    assert Path(stop["model"]).parent != Path(cfg["models_dir"])


def test_config_wake_models_exist_when_the_assets_are_present():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())["livekit"]["acoustic_wake"]
    d = ROOT / cfg["models_dir"]
    if not all((d / m).exists() for m in cfg["models"]):
        pytest.skip("wake-word runtime assets not present in this tree")
    cfg = dict(cfg, models_dir=str(d), engine_dir=str(ROOT / cfg["engine_dir"]))
    pytest.importorskip("openwakeword")
    assert build_factory(cfg) is not None


# ── the real model ───────────────────────────────────────────────────────

def _real():
    if not STOP_MODEL.exists():
        pytest.skip("lloyd_stop.onnx not present")
    pytest.importorskip("openwakeword")
    return build_stop_factory({"engine_dir": str(MODELS / "openwakeword"),
                               "stop": {"enabled": True, "model": str(STOP_MODEL),
                                        "threshold": 0.6}})


def _wav(name):
    with wave.open(str(FIXTURES / name)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


def _quiet(seconds, seed=0):
    return np.random.default_rng(seed).normal(0, 0.0008, int(SR * seconds)).astype(np.float32)


def _run(d, audio):
    hits = []
    for i in range(0, audio.size, 160):
        hits += d.feed(audio[i:i + 160])
    return hits


def test_the_real_stop_word_fires_when_armed_and_not_when_disarmed():
    f = _real()
    stream = np.concatenate([_quiet(2.0), _wav("stop_talking_16k.wav"), _quiet(1.0)])
    armed = f.create()
    _run(armed, _quiet(1.0))
    armed.arm()
    assert _run(armed, stream), "no stop on a clean 'stop talking'"
    idle = f.create()
    assert _run(idle, stream) == []


def test_the_real_stop_word_ignores_lloyd_saying_stop_mid_sentence():
    f = _real()
    d = f.create()
    _run(d, _quiet(1.0))
    d.arm()
    assert _run(d, np.concatenate([_wav("lloyd_says_stop_16k.wav"), _quiet(1.0)])) == []
