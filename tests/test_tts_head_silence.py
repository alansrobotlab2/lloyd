"""Head silence of a synthesis: is the first *frame* also the first *audible sample*? (#3366)

`voice/timeline.py` marks `first_played` at the playout of the first frame the worker
pushes, and `eos_to_audio` is that mark minus `speech_end`. A frame of padded silence
therefore reports as audio: LiveKit's measured case was a provider dashboard claiming
210 ms time-to-first-byte while shipping up to 3/4 s of silence behind it, against a
caller's budget of ~1.5 s from last syllable to first audible sound. Whether that holds
for the local Qwen3-TTS clone is a number, not an argument — so it is measured here, by
`scripts/voice/tts_head_silence.py`, and the decision this file pins is the one that
number bought: measured p50 ~20 ms, so no trim in the shaping path and no
`first_audible` mark in `STAGES` (the item's own ~60 ms no-code-change threshold).

Three properties these nodes hold, because each is a way the measurement could silently
lie:

  * **the onset detector.** ms *before* the first full 10 ms window whose RMS exceeds
    the floor (default 0.01 = -40 dBFS), quantised down to the window grid; and
    `None` — never 0.0 — when no window clears, because a zero there reads as
    "instantly audible" for a clip that never became audible at all. A burst shorter
    than a window is not audio, or the floor would be a peak detector and clip the
    soft fricative onsets this whole item turns on.
  * **two columns, not one.** the server's raw PCM and the same bytes after the
    production `OutputShaper` are reported separately, each with its own N, so padding
    inherited from the vocoder stays separable from padding the EQ or the WSOLA
    warm-up created. They are the same bytes on the way in, so a report that prints
    only one of them cannot answer which side to fix.
  * **bytes vs framing.** the request is the worker's own body — without
    `stream_format: "audio"` vLLM-Omni answers `text/event-stream` — and such an
    answer is refused by name. Decoded as s16le, SSE framing is loud noise with a
    plausible head-silence number attached (measured in `scripts/voice/tts_bakeoff.py`:
    61921 SSE bytes read rms -3.83 dBFS against the real PCM's -19.1).
"""
import importlib.util
import io
import json
import math
import contextlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Production's sample rate (`livekit.tts.sample_rate`), so the ms arithmetic in these
#: nodes is the arithmetic the probe does on real responses.
SR = 24000


@pytest.fixture(scope="module")
def ths():
    path = ROOT / "scripts" / "voice" / "tts_head_silence.py"
    # No importorskip: this diff ships the file, so its absence is a rename, and the
    # answer to that is red nodes naming the path rather than silent skips.
    assert path.is_file(), f"{path} is not on disk — the probe script was not added"
    spec = importlib.util.spec_from_file_location("tts_head_silence", path)
    assert spec is not None and spec.loader is not None, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tts_head_silence"] = mod
    spec.loader.exec_module(mod)
    return mod


# ── fixtures for the detector ───────────────────────────────────────────────

def _s16le(x) -> bytes:
    """float samples in [-1, 1] as the signed-16-bit little-endian the server sends."""
    return np.rint(np.asarray(x, dtype=float) * 32767.0).astype("<i2").tobytes()


def _silence(ms: float, sr: int = SR) -> bytes:
    return b"\x00\x00" * int(sr * ms / 1000)


def _tone(rms: float, ms: float, sr: int = SR, freq: float = 1000.0):
    """A 1 kHz sine whose own RMS is `rms` over its whole length."""
    n = int(sr * ms / 1000)
    t = np.arange(n) / sr
    return np.sin(2 * np.pi * freq * t) * (rms * math.sqrt(2.0))


# ── the onset detector ──────────────────────────────────────────────────────

def test_onset_is_the_ms_before_the_first_window_over_the_floor(ths):
    """200 ms of digital silence then a 0.1-RMS tone is 200 ms of head silence — the
    first frame of that clip is silence, which is the whole defect being measured."""
    pcm = _silence(200) + _s16le(_tone(0.1, 500))
    assert ths.onset_ms(pcm, SR) == 200.0


def test_audio_from_sample_zero_is_zero_and_not_none(ths):
    """0.0 is a real answer here and the fastest one on the box; returning `None` for it
    would drop measurable fixtures out of the population and inflate p50."""
    got = ths.onset_ms(_s16le(_tone(0.1, 500)), SR)
    assert got == 0.0
    assert got is not None


def test_silence_only_reports_none_never_an_instant_onset(ths):
    """One second of zeros never clears the floor. Read as 0.0 it would be the single
    best head-silence result in the sample."""
    assert ths.onset_ms(_silence(1000), SR) is None


def test_the_floor_is_the_declared_one(ths):
    """Just under 0.01 RMS (-40 dBFS) is below the floor and just over is above it.
    The floor is what makes a soft fricative onset read as silence, so it is a declared
    number and not a tuning knob these nodes leave undefined."""
    assert ths.onset_ms(_silence(200) + _s16le(_tone(0.009, 500)), SR) is None
    assert ths.onset_ms(_silence(200) + _s16le(_tone(0.011, 500)), SR) == 200.0


def test_a_burst_shorter_than_a_window_is_not_audio(ths):
    """The floor must be *held* for a whole 10 ms window. A 5 ms transient at full
    scale clears it in RMS terms if windows are allowed to be short, and then the
    detector is a peak detector — which is how a trim ends up biting consonants."""
    assert ths.onset_ms(_silence(200) + _s16le(_tone(0.5, 5)), SR) is None


def test_window_width_is_configurable(ths):
    """Same clip at a 20 ms window: the clearing window starts one 20 ms grid step
    earlier than its 200 ms mark, because onset is quantised down to the grid and so
    never over-reports head silence."""
    pcm = _silence(200) + _s16le(_tone(0.1, 500))
    assert ths.onset_ms(pcm, SR, window_ms=20) == 200.0
    assert ths.onset_ms(_silence(30) + _s16le(_tone(0.1, 500)), SR, window_ms=20) == 20.0


# ── two columns: raw server PCM, and the same bytes after the production shaper ──

@pytest.fixture
def cfg():
    """The shape of `livekit.tts` as `load_cfg()` returns it, with the worker's
    production values in it. Built here rather than read from config.yaml so these
    nodes do not move when Alan retunes the voice by ear."""
    return {"api_url": "http://127.0.0.1:8093", "model": "qwen3-tts",
            "voice": "clone:dave_cullen", "sample_rate": SR, "speed": 1.22,
            "presence_eq": True,
            "shelves": [{"freq": 1800, "gain_db": 2.75, "q": 1.0},
                        {"freq": 8000, "gain_db": 4.0, "q": 0.7}],
            "first_clause_chars": 20, "normalise_text": True}


class _Resp:
    def __init__(self, body: bytes, content_type: str):
        self._body, self._ct = body, content_type

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body

    @property
    def headers(self):
        return {"Content-Type": self._ct}


class _Opener:
    """Stands in for `urllib.request.urlopen`: records the request, answers with bytes."""

    def __init__(self, body=b"", content_type="audio/pcm"):
        self.body, self.ct = body, content_type
        self.requests = []

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        return _Resp(self.body, self.ct)


def test_a_row_carries_raw_and_shaped_head_as_separate_fields(ths, cfg):
    """Clause 2: one synthesis, two measurements of the same bytes. `raw_head_ms` is the
    server's PCM; `shaped_head_ms` is what the worker actually pushed after EQ + WSOLA.

    200 ms of silence then a 0.1-RMS tone is raw 200.0, and the same bytes through the
    production `OutputShaper` at the configured 1.22x come out at 160.0: WSOLA starts
    reading at `buf[radius]` (11 ms in) and the stretch divides the rest of the head by
    the speed, so 200/1.22 = 163.9 ms lands in the window below 170. The shaped value is
    pinned exactly, and strictly below the raw one, on purpose: a bound like
    `0.0 <= shaped <= 200.0`, or the pair of columns checked only for presence, is still
    satisfied by a `shape()` that returns its argument — two names over one measurement,
    which is the single failure mode this split exists to catch."""
    opener = _Opener(body=_silence(200) + _s16le(_tone(0.1, 500)))
    items = ths.fixtures(["Sure, the build finished cleanly."], cfg)
    rows = ths.measure(items, cfg, opener=opener)
    assert len(rows) == 2, "one clause row and one whole-reply row per fixture"
    for row in rows:
        assert "raw_head_ms" in row and "shaped_head_ms" in row
        assert row["raw_head_ms"] == 200.0
        shaped = row["shaped_head_ms"]
        assert shaped is not None
        assert shaped == 160.0, "not what the production shaper does to this fixture"
        assert shaped < row["raw_head_ms"], (
            "shaped == raw means shape() was skipped: one column, two names")


def test_shaping_never_invents_leading_silence(ths, cfg):
    """Audio from sample 0 must still be audio from sample 0 after the shaper. This is
    the item's step 2 question — can the EQ or the WSOLA warm-up *create* head silence —
    and the answer has to hold per fixture, not only in the median."""
    opener = _Opener(body=_s16le(_tone(0.1, 500)))
    items = ths.fixtures(["Sure, the build finished cleanly."], cfg)
    for row in ths.measure(items, cfg, opener=opener):
        assert row["raw_head_ms"] == 0.0
        assert row["shaped_head_ms"] == 0.0, (
            "the shaper added head silence that the server did not send")


def test_each_column_gets_its_own_n(ths):
    """A fixture whose shaped audio never clears the floor shrinks the shaped N without
    shrinking the raw N. Counted over one shared N, a p90 over six measurable samples is
    printed as a p90 over ten — which is when a percentile stops being a percentile."""
    rows = [
        {"kind": "reply", "text": "a", "sent": "a",
         "raw_head_ms": 20.0, "shaped_head_ms": 10.0},
        {"kind": "reply", "text": "b", "sent": "b",
         "raw_head_ms": 30.0, "shaped_head_ms": None},
    ]
    report = ths.summarise(rows)
    assert report["reply"]["raw"]["n"] == 2
    assert report["reply"]["shaped"]["n"] == 1
    assert report["reply"]["raw"]["p50"] == pytest.approx(25.0)
    assert report["reply"]["shaped"]["p50"] == 10.0
    assert set(report) == {"reply"}, "a kind with no rows gets no block, not a row of zeros"


# ── the request is the worker's, and a framing answer is refused ────────────

def test_synthesis_posts_the_workers_own_body(ths):
    """`livekit_worker.py:641-656`: same endpoint, same fields. `stream_format: "audio"`
    is what makes vLLM-Omni send the s16le this decodes, and `speed` must be 1.0 on the
    wire because the server's streaming path drops it and the worker stretches instead —
    asking twice would stretch the probe's audio differently from production's."""
    opener = _Opener(body=b"\x00\x00" * 100)
    ths.synthesize("hello there", api_url="http://127.0.0.1:8093/",
                   model="qwen3-tts", voice="clone:dave_cullen", opener=opener)
    assert len(opener.requests) == 1
    req = opener.requests[0]
    assert req.full_url == "http://127.0.0.1:8093/v1/audio/speech"
    assert json.loads(req.data) == {
        "model": "qwen3-tts", "input": "hello there", "voice": "clone:dave_cullen",
        "response_format": "pcm", "stream": True, "stream_format": "audio",
        "speed": 1.0}


def test_the_pinned_body_is_still_the_workers_body(ths):
    """The literal above is the probe's copy of the worker's body, so on its own it stays
    green when the worker changes and the probe silently measures a different request —
    the `stream_format: "audio"` field in particular, which is the only thing standing
    between this script and a `text/event-stream` answer, has no runtime presence at all
    and can vanish from `livekit_worker.py` with every unit test still passing. So read
    the worker's own source: the endpoint it posts to, and the keys and literals of the
    dict it posts, must be the ones `speech_body` builds."""
    worker = (ROOT / "agent-services" / "livekit_worker.py").read_text()
    block = worker[worker.index("self._http.stream("):]
    block = block[:block.index(") as resp:")]
    assert '/v1/audio/speech' in block, "the worker's speech endpoint moved"
    posted = re.findall(r'^\s{20}"([a-z_]+)":', block, re.M)
    assert posted == ["model", "input", "voice", "response_format", "stream",
                      "stream_format", "speed"], (
        f"the worker posts {posted}; speech_body must post the same fields in the same "
        f"order, or the probe is measuring a request production never sends")
    assert '"response_format": "pcm"' in block and '"stream_format": "audio"' in block
    assert '"speed": 1.0,' in block, (
        "the worker stopped pinning speed to 1.0: the server's streaming path drops the "
        "field, so a non-1.0 wire value would stretch this probe's audio differently")
    assert ths.speech_body("t", model="m", voice="v") == {
        "model": "m", "input": "t", "voice": "v", "response_format": "pcm",
        "stream": True, "stream_format": "audio", "speed": 1.0}


def test_a_non_audio_content_type_is_a_named_error_not_samples(ths):
    """Without `stream_format` the engine answers SSE. Those bytes are not silence and
    not speech, they are base64 text, and measured as PCM they would be the loudest,
    cleanest-onset fixtures in the set."""
    opener = _Opener(body=b'data: {"audio": "AAAA"}\n\n', content_type="text/event-stream")
    with pytest.raises(ths.NonAudioResponse) as exc:
        ths.synthesize("hello", api_url="http://127.0.0.1:8093", model="qwen3-tts",
                       voice="clone:dave_cullen", opener=opener)
    assert "text/event-stream" in str(exc.value)


def test_an_octet_stream_answer_is_accepted_as_bytes(ths):
    """The guard refuses `text/*` rather than requiring an `audio/` prefix: a server
    answering application/octet-stream is sending real waveform bytes, and refusing it
    would refuse the correct answer on one engine."""
    opener = _Opener(body=b"\x00\x00" * 100, content_type="application/octet-stream")
    got = ths.synthesize("hello", api_url="http://127.0.0.1:8093", model="m",
                        voice="v", opener=opener)
    assert got == b"\x00\x00" * 100


def test_the_cli_fails_loudly_on_a_framing_answer(ths, tmp_path, monkeypatch):
    """Non-zero exit, the error named on stderr, and no percentiles printed — a run
    that measured framing must not leave a table behind that reads as a result."""
    fixtures = tmp_path / "fixtures.txt"
    fixtures.write_text("Sure, the build finished cleanly.\n")
    monkeypatch.setattr(ths, "OPENER",
                        _Opener(body=b"data: x\n\n", content_type="text/event-stream"))
    err, out = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
        rc = ths.main(["--fixture-file", str(fixtures)])
    assert rc != 0
    assert "text/event-stream" in err.getvalue()
    assert "p50" not in out.getvalue()


def test_the_cli_fails_loudly_when_the_service_is_down(ths, tmp_path, monkeypatch):
    """A refused connection is an exit and a sentence, not a run of 0 fixtures that
    prints empty percentiles and looks like a measured no."""
    import urllib.error

    def refuse(req, timeout=None):
        raise urllib.error.URLError("connection refused")

    fixtures = tmp_path / "fixtures.txt"
    fixtures.write_text("Sure, the build finished cleanly.\n")
    monkeypatch.setattr(ths, "OPENER", refuse)
    err, out = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
        rc = ths.main(["--fixture-file", str(fixtures)])
    assert rc != 0
    assert err.getvalue().strip()
    assert "p50" not in out.getvalue()


# ── --json: the same two columns, as data ───────────────────────────────────

def test_json_rows_and_summary_carry_both_columns(ths, tmp_path, monkeypatch):
    """The artifact is what the number gets quoted from, so both fields have to be in
    it per row, and each population's summary must state its own N per column."""
    pcm = _silence(200) + _s16le(_tone(0.1, 500))
    monkeypatch.setattr(ths, "OPENER", _Opener(body=pcm))
    fixtures = tmp_path / "fixtures.txt"
    fixtures.write_text("Sure, the build finished cleanly.\n"
                        "There are three new emails today.\n")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = ths.main(["--fixture-file", str(fixtures), "--json"])
    assert rc == 0
    payload = json.loads(out.getvalue())
    kinds = set()
    for row in payload["rows"]:
        assert "raw_head_ms" in row and "shaped_head_ms" in row
        kinds.add(row["kind"])
    assert kinds == {"clause", "reply"}
    for kind in kinds:
        block = payload["summary"][kind]
        for col in ("raw", "shaped"):
            assert set(block[col]) >= {"n", "p50", "p90", "p95"}
            assert block[col]["n"] == 2


# ── the timeline docstring must not promise what the mark does not measure ───

def test_timeline_stops_calling_eos_to_audio_first_audible():
    """`timeline.py` described `eos_to_audio` as "end of speech to first audible audio"
    while the mark behind it is the first *frame's* playout. #3366 measures the head at
    ~20 ms instead of changing the mark, so the sentence has to become true-by-wording:
    it names the frame's playout and points at the probe that bounds the difference."""
    src = (ROOT / "agent-services" / "voice" / "timeline.py").read_text()
    assert "first audible" not in src.lower()
    assert "tts_head_silence" in src, "the corrected wording must point at the probe"


def test_no_first_audible_stage_exists(ths):
    """A 13th stage would move the missing-stage-rate denominator and the retention and
    trend tests that #2273's findings already warn about, to move `eos_to_audio` by 0.6%
    of its median. Re-open only if a landed probe's p90 passes ~100 ms."""
    path = ROOT / "agent-services" / "voice" / "timeline.py"
    spec = importlib.util.spec_from_file_location("tl_stages", path)
    assert spec is not None and spec.loader is not None, path
    tl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tl)
    assert "first_audible" not in tl.STAGES
    assert "first_played" in tl.STAGES
