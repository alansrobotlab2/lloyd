"""The spoken-word fidelity round trip (#1165): corpus, rater, scoring, report.

Everything here is stubbed — no TTS server, no GPU, no Parakeet model. What is
pinned is what the live run cannot check about itself:

  * the corpus covers the spoken surface the item names, with a bucket and
    target tokens per item and a benign control arm;
  * synthesis asks for the clone voice k>=2 times per item, and the rater is
    Parakeet greedy with no hotword list, so it cannot "hear" a name because
    it was told to expect one;
  * rates print with their n, the control arm is scored apart, an empty
    bucket is "no verdict" rather than a pass, and the result is a dated JSON
    under eval/baselines/.

The live measurement (before/after against the stored baseline) is a human
step on the GPU box, and so is turning `livekit.tts.normalise_text` on.
"""
import io
import json
import re
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))

fid = pytest.importorskip("run_tts_fidelity_eval")

HOTWORD_NAMES = ["Alan", "Lisa", "Emilio", "Alfie", "Lloyd", "Stompy", "Gracie", "gr00t"]


@pytest.fixture(scope="module")
def items():
    return fid.load_corpus()


def _texts(items, bucket=None):
    return [it["text"] for it in items if bucket is None or it["bucket"] == bucket]


# ── the corpus ──────────────────────────────────────────────────────────────

def test_the_corpus_is_pinned_with_a_bucket_and_targets_per_item(items):
    assert len(items) >= 45
    assert len({it["id"] for it in items}) == len(items), "duplicate ids"
    for it in items:
        assert it["bucket"] and isinstance(it["targets"], list) and it["targets"], it["id"]
    buckets = {it["bucket"] for it in items}
    for b in ("names", "technical", "voice_source", "email", "phone", "date",
              "ratio", "count", "path", "punctuation", "bullet_reply", "control"):
        assert b in buckets, b


def test_the_corpus_carries_every_named_hard_form(items):
    text = "\n".join(_texts(items))
    for name in HOTWORD_NAMES:
        assert re.search(rf"\b{re.escape(name)}\b", text), name
    for term in ("qmd", "vLLM", "Tailscale", "supervisord", "Dave Cullen",
                 "2026-09-15", "3.1x", "~23.6k", "—"):
        assert term in text, term
    assert re.search(r"\b[\w.]+@[\w-]+\.\w+", text), "no email"
    assert re.search(r"\d{3}-\d{3}-\d{4}", text), "no phone number"
    assert re.search(r"~/\S+/foo\.md", text), "no ~/…/foo.md path"
    bullets = [it for it in items if it["bucket"] == "bullet_reply"]
    assert bullets and all(len(it["text"]) <= 300 and "\n- " in it["text"] for it in bullets)


def test_the_control_arm_is_benign(items):
    controls = _texts(items, "control")
    assert len(controls) >= 15
    assert not any(re.search(r"\d", s) for s in controls)


# ── synthesis and the rater ─────────────────────────────────────────────────

def test_each_item_is_synthesised_k_times_per_arm(items):
    calls = []

    def synth(text):
        calls.append(text)
        return text.encode()

    res = fid.run(items, synth, lambda wav: "", k=2, join=b" ".join)
    per_item = {}
    for s in res["samples"]:
        per_item[(s["id"], s["arm"])] = per_item.get((s["id"], s["arm"]), 0) + 1
    assert set(per_item.values()) == {2}
    assert len(per_item) == len(items) * 2
    assert res["k"] == 2
    with pytest.raises(ValueError):
        fid.run(items, synth, lambda wav: "", k=0)


def test_live_synthesis_asks_for_the_clone_voice(monkeypatch):
    sent = []

    class _R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"\x00\x00" * 240

    def urlopen(req, timeout=None):
        sent.append(json.loads(req.data))
        return _R()

    monkeypatch.setattr(fid.urllib.request, "urlopen", urlopen)
    wav = fid.live_synth({"api_url": "http://tts.invalid"}, shape=False)("Hello there.")
    assert fid.VOICE == "clone:dave_cullen"
    assert sent[0]["voice"] == "clone:dave_cullen"
    assert sent[0]["input"] == "Hello there." and sent[0]["speed"] == 1.0
    with wave.open(io.BytesIO(wav)) as w:
        assert w.getframerate() == 24000 and w.getnframes() == 240


def test_the_rater_is_parakeet_greedy_with_no_hotword_list():
    cfg = fid.RATER_CFG
    assert cfg["backend"] == "parakeet"
    assert cfg["decoding_method"] == "greedy_search"
    assert cfg["parakeet_hotwords"] is False and not cfg.get("hotwords")
    assert "hotwords_file" not in cfg
    rec = fid.build_rater()  # constructs only; load() is what needs the model
    from voice import asr as voice_asr
    assert type(rec) is voice_asr.SherpaOfflineRecognizer
    assert rec.decoding_method == "greedy_search" and rec.hotwords == []


def test_audio_goes_through_asr_evals_decoder(monkeypatch):
    """The live rater reads wav through `asr_eval.decode_audio` and `to_16k`."""
    import numpy as np

    pcm = (np.sin(np.arange(2400) / 5) * 8000).astype("<i2").tobytes()
    joined = fid.join_wavs([fid.pcm_to_wav(pcm, 24000)] * 2, gap_ms=100)
    x, sr = fid.decode_audio({"bytes": joined})
    assert sr == 24000 and x.size == 2 * (2400 + 2400)

    seen = {}

    class _Rec:
        def transcribe(self, audio):
            seen["n"] = audio.size
            return type("T", (), {"text": "ok"})()

    assert fid.live_transcribe(_Rec())(joined) == "ok"
    assert seen["n"] == pytest.approx(x.size * 16000 / 24000, rel=0.02)


# ── scoring ─────────────────────────────────────────────────────────────────

def test_a_target_is_heard_however_the_rater_spaces_or_spells_it():
    hyp = fid.tokens("The QMD daemon crashed on September 15, 2026 at 3.1x load.")
    assert fid.contains_target(hyp, "qmd|q m d")
    assert fid.contains_target(hyp, "fifteenth|fifteen")
    assert fid.contains_target(hyp, "twenty twenty six|two thousand twenty six")
    assert fid.contains_target(hyp, "three point one")
    assert not fid.contains_target(hyp, "tailscale")
    assert fid.contains_target(fid.tokens("q. m. d."), "qmd")


def test_exact_match_ignores_spacing_and_markdown():
    item = {"text": "The qmd daemon is answering again.", "targets": ["qmd"]}
    assert fid.score_sample(item, "The QMD daemon is answering again")["exact"]
    assert not fid.score_sample(item, "The QMV daemon is answering again")["exact"]


def _stub_items():
    return [
        {"id": "t1", "bucket": "technical", "text": "The qmd daemon is up.",
         "targets": ["qmd|q m d"]},
        {"id": "t2", "bucket": "technical", "text": "Tailscale is down.",
         "targets": ["tailscale"]},
        {"id": "c1", "bucket": "control", "text": "The dog is asleep.",
         "targets": ["dog", "asleep"]},
    ]


def _echo_rater(mishear):
    """Transcribes the stub 'audio' (the sent text) back, garbling some words."""
    def transcribe(wav):
        text = wav.decode()
        for a, b in mishear.items():
            text = text.replace(a, b)
        return text
    return transcribe


def test_rates_are_per_bucket_with_n_and_the_control_arm_apart():
    res = fid.run(_stub_items(), lambda t: t.encode(), _echo_rater({"qmd": "cumdee"}),
                  k=2, join=b" ".join)
    raw, fs = res["arms"]["raw"], res["arms"]["for_speech"]
    assert "control" not in raw["buckets"]
    assert raw["control"]["target"] == {"hit": 4, "n": 4, "rate": 1.0}
    assert raw["buckets"]["technical"]["target"] == {"hit": 2, "n": 4, "rate": 0.5}
    assert raw["buckets"]["technical"]["exact"]["n"] == 4
    # The for_speech arm sent "Q M D", which the stub rater did not garble.
    assert fs["buckets"]["technical"]["target"]["hit"] == 4
    assert raw["overall"]["target"]["n"] == 4  # control excluded from overall

    out = fid.report(res)
    assert "[raw]" in out and "[for_speech]" in out
    assert "50.0% (2/4)" in out
    assert "control arm (benign, scored apart)" in out
    assert "100.0% (4/4)" in out


def test_an_empty_bucket_is_no_verdict_never_a_pass():
    res = fid.summarise([], _stub_items(), 2, ["raw"])
    t = res["arms"]["raw"]["buckets"]["technical"]
    assert t["target"] == {"hit": 0, "n": 0, "rate": None}
    out = fid.report(res)
    assert "no verdict (n=0)" in out
    assert "%" not in out


# ── the stored result ───────────────────────────────────────────────────────

def test_the_result_is_a_dated_json_under_eval_baselines(tmp_path):
    import datetime as dt

    assert fid.default_out_dir().parts[-2:] == ("eval", "baselines")
    res = fid.run(_stub_items(), lambda t: t.encode(), _echo_rater({}), k=2,
                  join=b" ".join)
    out_dir = tmp_path / "eval" / "baselines"
    path = fid.write_result(res, out_dir, {"voice": fid.VOICE},
                            today=dt.date(2026, 9, 24))
    assert path == out_dir / "tts_fidelity_2026-09-24.json"
    data = json.loads(path.read_text())
    assert data["date"] == "2026-09-24" and data["voice"] == "clone:dave_cullen"
    assert data["arms"]["raw"]["control"]["target"]["n"] == 4
    assert len(data["samples"]) == 12


def test_main_runs_end_to_end_with_stubbed_synthesis_and_rater(tmp_path, monkeypatch, capsys):
    class _Rec:
        def load(self):
            pass

    monkeypatch.setattr(fid, "build_rater", lambda: _Rec())
    monkeypatch.setattr(fid, "live_synth", lambda cfg, voice, shape: (lambda t: t.encode()))
    monkeypatch.setattr(fid, "live_transcribe", lambda rec: (lambda wav: wav.decode()))
    monkeypatch.setattr(fid, "join_wavs", lambda parts: b" ".join(parts))
    assert fid.main(["--k", "2", "--out-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "control arm" in out and "wrote" in out
    written = list(tmp_path.glob("tts_fidelity_*.json"))
    assert len(written) == 1
    data = json.loads(written[0].read_text())
    assert data["rater"]["parakeet_hotwords"] is False and data["k"] == 2
    # An echoing rater hears every control sentence.
    assert data["arms"]["raw"]["control"]["target"]["rate"] == 1.0
