"""The TTS bake-off harness: pacing, the first-clause cut, the sub-1 kHz-gated
presence measurement, and the HTTP runner's request body and content-type
guard (scripts/voice/tts_bakeoff.py)."""
import argparse
import importlib.util
import json
import threading
from pathlib import Path

import numpy as np
import pytest

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


# ── the HTTP runner: what it asks for, and what it refuses to decode ────────
#
# #2528, found 2026-10-10 against the vLLM-Omni engine production boots (:8093).
# The runner posted `response_format: "pcm", stream: true` but not
# `stream_format: "audio"`, which is the field that makes omni send the s16le
# this loop decodes; without it the answer is `text/event-stream` carrying
# base64 of that same PCM. Measured against :8093 for one utterance: 61921 SSE
# bytes decoded as s16le read rms -3.83 dBFS / peak 0.980 / crest +3.66 dB
# against the real 42240-byte PCM's -19.1 / 0.457 / +12.3, and `audio_s` 1.29 s
# against a true 0.88 s. A brick-walled signal that does not exist, scored as a
# candidate. The vendored server on :8090 ignores the field and answers
# `audio/pcm` either way, so the bug only bit on the engine in use.


def _pcm(seconds: float = 0.1, sr: int = 24000, freq: float = 180.0) -> bytes:
    t = np.arange(int(sr * seconds)) / sr
    return (np.sin(2 * np.pi * freq * t) * 9000).astype("<i2").tobytes()


def _sse(pcm: bytes) -> bytes:
    """One SSE frame of the shape omni sends: the same PCM, base64'd after an
    `event:` line — ASCII first bytes, not s16le."""
    import base64

    return b"event: speech.audio.delta\ndata: " + base64.b64encode(pcm) + b"\n\n"


def _chunks(b: bytes, n: int = 4096) -> list[bytes]:
    return [b[i:i + n] for i in range(0, len(b), n)] or [b""]


class _Response:
    """The `http.client.HTTPResponse` surface `stream` touches: a Content-Type
    header and chunked reads, used as a context manager."""

    def __init__(self, content_type: str, chunks):
        self.headers = {"Content-Type": content_type}
        self._chunks = list(chunks)

    def read1(self, n):
        return self._chunks.pop(0) if self._chunks else b""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Endpoint:
    """A real loopback `/v1/audio/speech`, because the seam is urllib's
    `HTTPResponse` — `headers` there is an `email.message.Message`, not the dict
    the stub below uses, and the loop reads through `read1`. It answers the way
    omni does: `text/event-stream` to any body that does not ask for
    `stream_format: "audio"`, `audio/pcm` to one that does."""

    def __init__(self, pcm: bytes):
        from http.server import BaseHTTPRequestHandler, HTTPServer

        self.bodies: list[dict] = []
        bodies = self.bodies

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n).decode())
                bodies.append(body)
                sse = body.get("stream_format") != "audio"
                payload = _sse(pcm) if sse else pcm
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "text/event-stream; charset=utf-8" if sse else "audio/pcm")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a):
                pass

        self._srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._srv.server_address[1]}"

    def close(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()


def test_the_guard_holds_over_a_real_socket_and_the_pcm_still_decodes():
    """Same two answers as `_Engine`, but through a real HTTP response, so a
    header lookup written against the stub's dict cannot pass this while failing
    against the engine on :8093."""
    pcm = _pcm()
    srv = _Endpoint(pcm)
    try:
        sse_arm = bake.OpenAIRunner(srv.url, "clone:dave_cullen",
                                    extra={"stream_format": "sse"})
        got = []
        with pytest.raises(bake.NonAudioResponse) as ei:
            for c in sse_arm.stream("Hello there."):
                got.append(c)
        assert got == [], "no chunk of an SSE answer may reach the caller"
        assert "text/event-stream" in str(ei.value)

        audio_arm = bake.OpenAIRunner(srv.url, "clone:dave_cullen")
        out = np.concatenate(list(audio_arm.stream("Hello there.")))
        assert np.array_equal(out, _float(pcm))
        assert srv.bodies[-1]["stream_format"] == "audio"
    finally:
        srv.close()


class _Engine:
    """`urllib.request.urlopen` stand-in that answers the way the two TTS
    servers this runner talks to do: omni (:8093) sends `text/event-stream`
    unless the body carries `stream_format: "audio"`, the vendored server
    (:8090) ignores the field and always sends `audio/pcm`."""

    def __init__(self, pcm: bytes, *, omni: bool = True, content_type: str = "audio/pcm"):
        self.pcm, self.omni, self.content_type = pcm, omni, content_type
        self.bodies: list[dict] = []

    def __call__(self, req, timeout=None):
        self.bodies.append(json.loads(req.data.decode()))
        if self.omni and self.bodies[-1].get("stream_format") != "audio":
            return _Response("text/event-stream; charset=utf-8", _chunks(_sse(self.pcm)))
        return _Response(self.content_type, _chunks(self.pcm))


def _runner(monkeypatch, handler) -> bake.OpenAIRunner:
    monkeypatch.setattr(bake.urllib.request, "urlopen", handler)
    return bake.OpenAIRunner("http://tts.test:8093", "clone:dave_cullen")


def _float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, "<i2").astype(np.float32) / 32768.0


def test_the_runner_asks_the_engine_for_raw_bytes_not_server_sent_events(monkeypatch):
    """`response_format: "pcm"` + `stream: true` alone is not a request for
    bytes: on omni it selects `text/event-stream`. `stream_format: "audio"` is
    the field the worker sends (agent-services/livekit_worker.py) and the one
    the bake-off runner was skipped for in 11fedc3d."""
    engine = _Engine(_pcm())
    list(_runner(monkeypatch, engine).stream("Hello there."))
    assert engine.bodies[0]["stream_format"] == "audio"
    assert engine.bodies[0]["response_format"] == "pcm"
    assert engine.bodies[0]["stream"] is True
    assert engine.bodies[0]["voice"] == "clone:dave_cullen"


def test_audio_pcm_answer_decodes_to_the_samples_the_server_sent(monkeypatch):
    pcm = _pcm()
    got = list(_runner(monkeypatch, _Engine(pcm, omni=False)).stream("Hello there."))
    assert np.array_equal(np.concatenate(got), _float(pcm))


def test_a_text_event_stream_answer_is_refused_and_yields_nothing(monkeypatch):
    """The guard cannot be "content-type contains stream": the *broken* answer
    is text/event-stream and the correct one is audio/pcm, so that assertion
    passes exactly the case it exists to catch. The rule is refuse `text/*`,
    and it fires before the first chunk, so no bogus samples reach the caller."""
    pcm = _pcm()
    monkeypatch.setattr(
        bake.urllib.request, "urlopen",
        lambda req, timeout=None: _Response("text/event-stream; charset=utf-8",
                                            _chunks(_sse(pcm))))
    got = []
    with pytest.raises(bake.NonAudioResponse) as ei:
        for c in bake.OpenAIRunner("http://tts.test:8093", "clone:dave_cullen").stream("Hello there."):
            got.append(c)
    assert got == []
    assert "text/event-stream" in str(ei.value)


def test_octet_stream_is_still_decoded_so_the_guard_is_not_an_audio_prefix_check(monkeypatch):
    """Why the guard refuses `text/*` instead of requiring an `audio/` prefix:
    a candidate server that answers `application/octet-stream` is sending real
    bytes, and refusing it would fail a bake-off that is measuring something."""
    pcm = _pcm()
    r = _runner(monkeypatch, _Engine(pcm, omni=False, content_type="application/octet-stream"))
    assert np.array_equal(np.concatenate(list(r.stream("Hello there."))), _float(pcm))


def _synth_args(out: Path, **over) -> argparse.Namespace:
    d = dict(runner="openai", url="http://tts.test:8093", voice="clone:dave_cullen",
             model="qwen3-tts", sample_rate=24000, extra=None, out=str(out),
             name="cand", n=2, warmup=1, live_text=False, tok_rate=45.0,
             note=None, shape=False)
    d.update(over)
    return argparse.Namespace(**d)


def test_synth_against_an_sse_answering_server_writes_no_candidate_wav(tmp_path, monkeypatch):
    """The refusal has to surface out of `synth`, not become a row of numbers:
    `s00.wav` of an SSE decode is scored as a candidate by `score`, which is how
    the wrong measurement of #2528 survived a week."""
    cand = tmp_path / "cand"
    monkeypatch.setattr(
        bake.urllib.request, "urlopen",
        lambda req, timeout=None: _Response("text/event-stream; charset=utf-8",
                                            _chunks(_sse(_pcm()))))
    with pytest.raises(bake.NonAudioResponse):
        bake.synth(_synth_args(tmp_path))
    assert list(cand.rglob("*.wav")) == []
    assert not (cand / "synth.json").exists()


def test_synth_against_the_omni_engine_writes_the_pcm_it_sends(tmp_path, monkeypatch):
    """The same run against an engine that behaves: the body now asks for
    `stream_format: "audio"`, so every wav holds the server's own s16le rather
    than base64 of it, and `audio_s`/`rtf` divide by real audio seconds."""
    pcm = _pcm(0.2)
    engine = _Engine(pcm)
    monkeypatch.setattr(bake.urllib.request, "urlopen", engine)
    bake.synth(_synth_args(tmp_path))
    assert all(b["stream_format"] == "audio" for b in engine.bodies)
    wavs = sorted((tmp_path / "cand").glob("s*.wav"))
    assert len(wavs) == 2
    expected = _float(pcm)
    for w in wavs:
        x, sr = bake.read_wav(w)
        assert sr == 24000
        assert np.max(np.abs(x - expected)) * 32768 <= 1.0


def test_extra_still_wins_on_the_body_and_the_guard_catches_the_wrong_value(monkeypatch):
    """`--extra` is splatted last, so a caller can still set `stream_format` to
    a value an engine does not accept: that stays the escape hatch (a candidate
    needing a different body is a legitimate bake-off), and what keeps it from
    being silently wrong is the content-type guard, not the merge order."""
    engine = _Engine(_pcm())
    monkeypatch.setattr(bake.urllib.request, "urlopen", engine)
    r = bake.OpenAIRunner("http://tts.test:8093", "clone:dave_cullen",
                          extra={"stream_format": "video"})
    with pytest.raises(bake.NonAudioResponse):
        list(r.stream("Hello there."))
    assert engine.bodies[-1]["stream_format"] == "video"
