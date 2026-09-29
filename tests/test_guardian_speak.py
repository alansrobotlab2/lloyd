"""The guardian's spoken channel.

Four things are worth pinning here, and each is something that actually went
wrong rather than something that was merely conceivable:

  * **Shaping must engage, and say so when it cannot.** The first cut called
    `OutputShaper.enabled()` on what is a `@property`, so every utterance went
    out unshaped while the code looked correct and the alert still "worked".
    A shaping fallback that reports nothing is indistinguishable from shaping
    that succeeded, which is why `shape` logs its tier.
  * **Suppression must survive across processes.** The daemon and the nag
    oneshot are different processes, so an in-memory dedupe cannot see both.
  * **Nothing may raise, and nothing may block.** Every channel in notify.py
    holds that contract; this one adds a subprocess, which is a new way to
    violate it.
  * **Consuming a response must have a ceiling that is not the response's own
    length.** On 2026-09-28 19:51 a decoder that never reached EOS streamed
    31,467,360 bytes of s16le at 24000 Hz — 655.57 s of scrambled audio for a
    160-character alert — and `play` gave it 670.6 s of speakers to do it in,
    because the playback budget was `len(pcm) / (2 * sr) + 15.0`. `synthesize`
    now reads in chunks under `max_audio_seconds` and a separate
    `synth_wall_clock`, and the budget is capped by the ceiling.

Nothing here plays audio: `conftest` sets LLOYD_VOICE_ALERTS=0 for every test
and the tests that need dispatch to proceed re-enable it with a stubbed
`subprocess.Popen`. The playback tests below stub `subprocess.run` as well, so
no `paplay` is ever spawned; the stream tests stub `urlopen` with an object that
has the same `read(n)`/`close()` contract as `http.client.HTTPResponse`, so no
TTS server is involved either. The failure-path nodes at the end go the other way
round on purpose: their endpoint is a real port that was bound and closed, so the
`ECONNREFUSED` is genuine, and only the player subprocess is stubbed.

That last group is #1806: a synth failure used to leave nothing but a
`voice.log` line, and on 2026-09-28 the refused utterance was the alert
announcing an outage of the same supervised tree that hosts the TTS server.
"""

from __future__ import annotations

import array
import datetime
import json
import os
import re
import socket
import sys
import time
import wave
from pathlib import Path

import pytest

GUARDIAN_DIR = Path(__file__).resolve().parent.parent / "agent-services" / "guardian"
sys.path.insert(0, str(GUARDIAN_DIR))

import speak  # noqa: E402

SHAPING = Path(__file__).resolve().parent.parent / "agent-services" / "tts_shaping.py"


# ── what it says ──────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected_absent", [
    ("Rolled back 1a2b3c4d → 5e6f7a8b", "1a2b3c4d"),
    ("HEAD is already the last known good (9f8e7d6c)", "9f8e7d6c"),
    ("Clear ~/.local/state/lloyd-automod/BROKEN once resolved", "lloyd-automod"),
])
def test_speakable_drops_what_cannot_be_heard(raw, expected_absent):
    """Hashes and paths are in the toast, the ledger and the journal. Read
    aloud they are just noise in the one channel that cannot be skimmed."""
    assert expected_absent not in speak.speakable(raw)


def test_speakable_keeps_digit_runs_that_are_not_hashes():
    """A round id carries a date. `[0-9a-f]{7,40}` matches "20260906" too, so
    without the "must contain a letter" lookahead the id loses its date and
    every long number in a body silently disappears."""
    out = speak.speakable("Promoted SM_20260906_a1b2")
    assert "20260906" in out


def test_speakable_leaves_no_dangling_preposition():
    """Deleting the hash leaves the word that pointed at it: "Rolled back
    1a2b → 5e6f" becomes "Rolled back to", which sounds truncated."""
    out = speak.speakable("Rolled back 1a2b3c4d → 5e6f7a8b")
    assert out == "Rolled back", out


def test_speakable_removes_the_hole_a_hash_leaves():
    assert "()" not in speak.speakable("the last known good (9f8e7d6c) commit")
    assert " ." not in speak.utterance_for("error", "x", "good (9f8e7d6c). Next.")


def test_utterance_is_bounded_and_led_by_severity():
    long_body = "word " * 400
    out = speak.utterance_for("critical", "Rolled back", long_body, max_chars=120)
    assert out.startswith("Critical.")
    assert len(out) <= 121
    assert speak.utterance_for("error", "x").startswith("Guardian alert.")


# ── suppression ───────────────────────────────────────────────────────

def test_suppression_is_shared_across_processes(tmp_path):
    """The record lives on disk, not in memory, because the daemon and the
    15-minute nag oneshot are different processes. A fresh interpreter must
    see what the previous one said."""
    assert speak.should_speak("STILL BROKEN", tmp_path, window=3600) is True
    assert speak.should_speak("STILL BROKEN", tmp_path, window=3600) is False
    assert (tmp_path / speak.SPOKEN_NAME).is_file()

    # A different key is a different sentence and is not suppressed.
    assert speak.should_speak("Rolled back", tmp_path, window=3600) is True


def test_suppression_expires(tmp_path):
    now = time.time()
    assert speak.should_speak("k", tmp_path, window=60, now=now) is True
    assert speak.should_speak("k", tmp_path, window=60, now=now + 59) is False
    assert speak.should_speak("k", tmp_path, window=60, now=now + 61) is True


def test_corrupt_suppression_file_does_not_mute_the_channel(tmp_path):
    """Failing open is the right direction here: a garbled state file should
    cost a repeated sentence, never a missed alert."""
    (tmp_path / speak.SPOKEN_NAME).write_text("{not json at all")
    assert speak.should_speak("k", tmp_path, window=3600) is True


def test_suppression_file_does_not_grow_without_bound(tmp_path):
    now = time.time()
    for i in range(50):
        speak.should_speak(f"key-{i}", tmp_path, window=60, now=now - 100000)
    speak.should_speak("fresh", tmp_path, window=60, now=now)
    seen = json.loads((tmp_path / speak.SPOKEN_NAME).read_text())
    assert list(seen) == ["fresh"], "stale keys were never pruned"


# ── shaping ───────────────────────────────────────────────────────────

def _tone(seconds=1.0, sr=24000):
    import numpy as np
    t = np.arange(int(sr * seconds)) / sr
    x = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 3000 * t)
    return (x * 32767).astype("<i2").tobytes()


@pytest.mark.skipif(not SHAPING.is_file(), reason="tts_shaping.py absent")
def test_shaping_actually_engages(tmp_path):
    """The regression that motivated this file: `enabled` is a property, and
    calling it returned a bool that raised inside the try, so every utterance
    was passed through untouched while looking healthy."""
    pytest.importorskip("numpy")
    cfg = dict(speak.DEFAULTS, shaping_module=str(SHAPING), speed=0.85)
    pcm = _tone()
    out = speak.shape(pcm, cfg, tmp_path)
    assert out != pcm, "shaping did not run"
    # speed 0.85 stretches the utterance; the length is the cheapest proof.
    assert len(out) > len(pcm) * 1.1


@pytest.mark.skipif(not SHAPING.is_file(), reason="tts_shaping.py absent")
def test_shaping_degrades_to_wsola_and_says_so(tmp_path, monkeypatch):
    """scipy powers the presence EQ; numpy alone still fixes the pace. The
    guardian's system interpreter has numpy and not scipy, so this tier is
    the normal one there — and it must be legible in the log, not silent."""
    pytest.importorskip("numpy")
    real_import = __import__

    def no_scipy(name, *a, **kw):
        if name.startswith("scipy"):
            raise ModuleNotFoundError("No module named 'scipy'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", no_scipy)
    cfg = dict(speak.DEFAULTS, shaping_module=str(SHAPING), speed=0.85)
    out = speak.shape(_tone(), cfg, tmp_path)
    monkeypatch.undo()

    assert len(out) > len(_tone()) * 1.1, "pace was lost along with the EQ"
    log = (tmp_path / speak.LOG_NAME).read_text()
    assert "degraded" in log, "a silent downgrade is the bug this guards"


def test_missing_shaping_module_is_survivable(tmp_path):
    cfg = dict(speak.DEFAULTS, shaping_module=str(tmp_path / "nope.py"))
    pcm = _tone(0.1)
    assert speak.shape(pcm, cfg, tmp_path) == pcm
    assert "unavailable" in (tmp_path / speak.LOG_NAME).read_text()


def _AWAKE(cfg, now=None):
    """Quiet hours, forced off, for tests that are not about quiet hours.

    `speak.dispatch` consults `in_quiet_hours(cfg)` with no `now`, so it reads
    the real clock against a 23->07 default window. Three tests below assert
    that a dispatch *happens*, and between 23:00 and 07:00 local they were
    asserting against a system that had correctly decided to stay quiet.

    That is not academic: the unattended automod loop gates overnight, and the
    gate's `tests` rung is a hard rung. Rounds SM_20260908_065238 (23:52) and
    SM_20260908_105946 (04:07) both failed here and aborted; the same code at
    08:13 did not. The tests that *are* about the policy hold the real
    function and pin it themselves at every hour of the window.
    """
    return False


# ── dispatch ──────────────────────────────────────────────────────────

def test_dispatch_is_muted_by_the_env_switch(tmp_path, monkeypatch):
    monkeypatch.setenv("LLOYD_VOICE_ALERTS", "0")
    called = []
    monkeypatch.setattr(speak.subprocess, "Popen", lambda *a, **k: called.append(a))
    assert speak.dispatch("critical", "t", "b", tmp_path, window=1) is False
    assert called == []


def test_dispatch_spawns_detached_and_returns_immediately(tmp_path, monkeypatch):
    """The unit watchdogs the loop at 90s against a 5s tick, so the channel
    must hand off rather than synthesise inline."""
    monkeypatch.setenv("LLOYD_VOICE_ALERTS", "1")
    # Quiet hours are wall-clock, so without this the assertion below
    # depends on what time the suite runs. See _AWAKE.
    monkeypatch.setattr(speak, "in_quiet_hours", _AWAKE)
    seen = {}

    def fake_popen(cmd, **kw):
        seen["cmd"] = cmd
        seen["kw"] = kw
        return object()

    monkeypatch.setattr(speak.subprocess, "Popen", fake_popen)
    t0 = time.time()
    assert speak.dispatch("critical", "Rolled back", "body", tmp_path, window=3600) is True
    assert time.time() - t0 < 1.0, "dispatch blocked the caller"
    assert seen["kw"]["start_new_session"] is True, "child would die with the guardian"
    assert "--text" in seen["cmd"]
    spoken = seen["cmd"][seen["cmd"].index("--text") + 1]
    assert spoken.startswith("Critical.")

    # Second call inside the window is suppressed, so no second process.
    seen.clear()
    assert speak.dispatch("critical", "Rolled back", "body", tmp_path, window=3600) is False
    assert seen == {}


def test_dispatch_never_raises(tmp_path, monkeypatch):
    """Same contract as every other channel in notify.py."""
    monkeypatch.setenv("LLOYD_VOICE_ALERTS", "1")
    # Quiet hours are wall-clock, so without this the assertion below
    # depends on what time the suite runs. See _AWAKE.
    monkeypatch.setattr(speak, "in_quiet_hours", _AWAKE)

    def boom(*a, **k):
        raise OSError("no fork for you")

    monkeypatch.setattr(speak.subprocess, "Popen", boom)
    assert speak.dispatch("critical", "t", "b", tmp_path, window=3600) is False


def test_worker_python_prefers_the_venv(monkeypatch, tmp_path):
    """The detached child is the one place the stdlib-only rule does not
    apply — it runs after every reliable channel and nothing waits on it —
    and the venv is what buys the presence EQ."""
    venv = tmp_path / "python"
    venv.write_text("#!/bin/sh\n")
    venv.chmod(0o755)
    monkeypatch.setattr(speak, "VENV_PYTHON", str(venv))
    assert speak._worker_python() == str(venv)

    monkeypatch.setattr(speak, "VENV_PYTHON", str(tmp_path / "gone"))
    assert speak._worker_python() == sys.executable


# ── config ────────────────────────────────────────────────────────────

def test_config_overlays_defaults_and_tolerates_junk(tmp_path):
    assert speak.load_config(tmp_path)["voice"] == speak.DEFAULTS["voice"]
    (tmp_path / speak.CONFIG_NAME).write_text('{"voice": "clone:other", "speed": 1.0}')
    cfg = speak.load_config(tmp_path)
    assert cfg["voice"] == "clone:other" and cfg["speed"] == 1.0
    assert cfg["sample_rate"] == speak.DEFAULTS["sample_rate"], "lost a default"

    (tmp_path / speak.CONFIG_NAME).write_text("{{{ not json")
    assert speak.load_config(tmp_path)["voice"] == speak.DEFAULTS["voice"]


# ── consuming a runaway stream (#1779) ────────────────────────────────

class _EndlessResponse:
    """A TTS response that never ends, which is what the 2026-09-28 19:51 decoder
    runaway was: bytes, indefinitely, at a rate that never trips a socket timeout.

    Same contract `http.client.HTTPResponse` offers `synthesize` — `read(n)`
    returns what has arrived rather than blocking for `n`, `read()` with no
    argument reads to end-of-stream, `close()` stops the drain, and exiting the
    `with` block closes it. The body is random bytes, extended as it is drained,
    so "it kept the head" is checkable against the object's own first bytes and
    no window shift could pass for it.

    `max_reads` is this file's own backstop: an implementation with no ceiling
    would otherwise stream inside a test forever, so the fake raises after 2000
    reads (16 MiB at the 8192-byte chunk size, a fraction of a second) naming how
    much it got through — the incident reproduced as a failure rather than a hang.
    """

    def __init__(self, chunk_size: int = 8192, delay: float = 0.0,
                 max_reads: int = 2000):
        self.status = 200
        self.closed = False
        self.reads = 0
        self.chunk_size = chunk_size
        self.delay = delay
        self.max_reads = max_reads
        self._pos = 0
        self.body = os.urandom(1 << 20)

    def _read_one(self, k: int) -> bytes:
        self.reads += 1
        if self.reads > self.max_reads:
            raise AssertionError(
                f"still streaming after {self.reads - 1} reads and {self._pos} "
                f"bytes ({self._pos / float(2 * 24000):.1f}s at 24000Hz): no "
                f"ceiling stopped it")
        if self.delay:
            time.sleep(self.delay)
        if self._pos + k > len(self.body):
            self.body += os.urandom(1 << 20)
        out = self.body[self._pos:self._pos + k]
        self._pos += k
        return out

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            # read-to-EOF: blocks until the peer stops. With no ceiling in the
            # reader this is where the eleven minutes went, so it accumulates
            # until the fake's own backstop raises rather than hanging here.
            parts: list[bytes] = []
            while True:
                parts.append(self._read_one(self.chunk_size))
            return b"".join(parts)        # unreachable while the stream has no end
        return self._read_one(min(n, self.chunk_size))

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _voice_cfg(**over) -> dict:
    """DEFAULTS with overrides, the way `load_config` hands them to `synthesize`.

    Named apart from the `_cfg` further down, which is quiet-hours only.
    """
    return dict(speak.DEFAULTS, **over)


def _streaming(monkeypatch, stream):
    monkeypatch.setattr(speak.urllib.request, "urlopen",
                        lambda req, timeout=None: stream)


class _ChunkedResponse:
    """A response that behaves: hands out `body` in pieces, then end-of-stream.

    `read()` with no argument has to drain to the end like the real
    `HTTPResponse.read()`, otherwise a reader that never chunks would "pass" this
    node for the wrong reason — the one below is a guard, not a clause, and a
    guard that fails for a fake's shortcoming teaches nothing about the ceiling.
    """

    def __init__(self, body: bytes, chunk_size: int):
        self.status = 200
        self.closed = False
        self._body = body
        self._chunk = chunk_size
        self._pos = 0

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            rest = self._body[self._pos:]
            self._pos = len(self._body)
            return rest
        k = min(n, self._chunk)
        out = self._body[self._pos:self._pos + k]
        self._pos += len(out)
        return out

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()



def test_an_endless_stream_stops_at_the_duration_ceiling(tmp_path, monkeypatch):
    """Clause 1. `resp.read()` with no argument is what let the 2026-09-28
    runaway reach 655.57 s; the ceiling is `max_audio_seconds` of mono s16le at
    `sample_rate`, and the head is what survives — the decoder says the real
    sentence first and hallucinates after it, so a truncation still speaks."""
    cfg = _voice_cfg(max_audio_seconds=2.0, synth_wall_clock=90.0)
    cap = int(cfg["max_audio_seconds"] * cfg["sample_rate"] * 2)     # 96000
    assert cap == 96_000, "the ceiling arithmetic this test is about"
    stream = _EndlessResponse(chunk_size=8192)
    _streaming(monkeypatch, stream)

    out = speak.synthesize("Guardian alert. Gate rung failed.", cfg, tmp_path)

    assert out is not None
    assert len(out) <= cap, f"{len(out)} bytes is {len(out) / 48000.0}s of audio"
    assert len(out) == cap, "the ceiling was not reached, so it was not tested"
    assert out == stream.body[:cap], "kept something other than the head of the stream"


def test_the_over_run_closes_the_response_and_stops_reading(tmp_path, monkeypatch):
    """Clause 2. A ceiling that only truncates the *copy* while the socket keeps
    draining still holds the worker open on the runaway's clock — `close()` is
    the part that returns the connection the moment the decision is made."""
    cfg = _voice_cfg(max_audio_seconds=2.0, synth_wall_clock=90.0)
    stream = _EndlessResponse(chunk_size=8192)
    _streaming(monkeypatch, stream)

    speak.synthesize("Guardian alert. Gate rung failed.", cfg, tmp_path)

    assert stream.closed is True, "the response was left open on the over-run"
    # 96000 / 8192 = 11.7, so exactly twelve reads reach the ceiling. A thirteenth
    # would mean it kept draining the socket after deciding to stop.
    assert stream.reads == 12, f"read {stream.reads} times; 12 reaches the ceiling"


def test_a_slow_drip_is_stopped_by_the_wall_clock_not_the_socket_timeout(
        tmp_path, monkeypatch):
    """Clause 3. `synth_timeout` bounds one blocking socket operation, so a
    stream that trickles bytes faster than that never trips it — six such stalls
    are in voice.log on their own. `synth_wall_clock` is the total, and it is
    tested before a read is *started*, which is what makes it independent: here
    the per-socket timeout is 30 s and the total 0.30 s, so nothing but the total
    can explain returning in a third of a second."""
    cfg = _voice_cfg(max_audio_seconds=600.0, synth_wall_clock=0.30, synth_timeout=30.0)
    stream = _EndlessResponse(chunk_size=1024, delay=0.01)
    _streaming(monkeypatch, stream)

    started = time.monotonic()
    out = speak.synthesize("Guardian alert. Gate rung failed.", cfg, tmp_path)
    elapsed = time.monotonic() - started

    assert elapsed < cfg["synth_wall_clock"] + 0.15, (
        f"took {elapsed:.2f}s; the 30s per-socket timeout is the only bound that "
        f"could do that")
    assert speak._pcm_seconds(len(out), cfg["sample_rate"]) < 600.0, (
        "the duration ceiling is what stopped it, not the wall clock")
    assert stream.closed is True


def test_an_over_run_names_the_ceiling_and_the_seconds_received(tmp_path, monkeypatch):
    """Clause 4. An alert that quietly arrives three-quarters-short is the same
    anti-pattern `shape()` logs its fallback for: the channel would report itself
    healthy while saying less than it was told to. One line per ceiling, naming
    which fired, the ceiling, and the seconds the stream actually delivered."""
    cfg_dur = _voice_cfg(max_audio_seconds=2.0, synth_wall_clock=90.0)
    stream = _EndlessResponse(chunk_size=8192)
    _streaming(monkeypatch, stream)
    speak.synthesize("Guardian alert. Gate rung failed.", cfg_dur, tmp_path)
    dur_log = (tmp_path / speak.LOG_NAME).read_text()
    # Twelve 8192-byte reads = 98304 bytes = 2.048 s at 24000 Hz mono s16le.
    assert "duration ceiling" in dur_log, dur_log
    assert "max_audio_seconds=2.00s" in dur_log, "the ceiling is not named in seconds"
    assert f"received {98304 / 48000.0:.2f}s of audio" in dur_log, dur_log

    (tmp_path / speak.LOG_NAME).unlink()
    cfg_wall = _voice_cfg(max_audio_seconds=600.0, synth_wall_clock=0.30, synth_timeout=30.0)
    _streaming(monkeypatch, _EndlessResponse(chunk_size=1024, delay=0.01))
    speak.synthesize("Guardian alert. Gate rung failed.", cfg_wall, tmp_path)
    wall_log = (tmp_path / speak.LOG_NAME).read_text()
    assert "wall-clock" in wall_log, wall_log
    assert "synth_wall_clock=0.30s" in wall_log, "the ceiling is not named in seconds"
    assert "received" in wall_log and "s of audio" in wall_log, wall_log


def test_a_stream_that_ends_on_its_own_is_returned_whole_and_unlogged(tmp_path,
                                                                     monkeypatch):
    """The ceiling must bound, not replace: a healthy 8-second utterance — 395520
    bytes, what a 96-character alert measured on the live server — arrives
    complete, and nothing is logged because nothing was overridden."""
    body = os.urandom(395_520)
    stream = _ChunkedResponse(body, chunk_size=65_536)
    _streaming(monkeypatch, stream)

    out = speak.synthesize("Guardian alert. Gate rung failed.", _voice_cfg(), tmp_path)

    assert out == body
    assert not (tmp_path / speak.LOG_NAME).exists(), "a clean synth logged an over-run"


# ── playback budget ───────────────────────────────────────────────────

def _player_timeout(pcm: bytes, cfg: dict, monkeypatch) -> float:
    """Run `play` with the player stubbed and return the timeout it asked for."""
    seen = {}

    class _Proc:
        returncode = 0

    def fake_run(cmd, **kw):
        seen["timeout"] = kw.get("timeout")
        return _Proc()

    monkeypatch.setattr(speak, "_player", lambda path: ["paplay", path])
    monkeypatch.setattr(speak.subprocess, "run", fake_run)
    assert speak.play(pcm, cfg) is True
    return float(seen["timeout"])


def test_play_timeout_is_capped_by_the_ceiling_not_by_the_pcm(monkeypatch):
    """Clause 5. `budget = len(pcm) / (2 * sr) + 15.0` scaled with the one
    quantity a runaway decoder controls: the 31,467,360-byte artefact of
    2026-09-28 computed its own 670.6 s booking that way. The ceiling caps it —
    60 s of audio + 0.25 s of tail + 15 s of slack = 75.25 s — and a pcm twice or
    five times over the ceiling asks for the same thing, while a short alert
    still gets a short timeout rather than a blanket one."""
    cfg = _voice_cfg(sample_rate=24000, max_audio_seconds=60.0, tail_silence_ms=250)
    at = lambda s: b"\x00" * int(s * 24000 * 2)          # mono s16le, `s` seconds

    over_120s = _player_timeout(at(120.0), cfg, monkeypatch)
    over_300s = _player_timeout(at(300.0), cfg, monkeypatch)
    assert over_120s == pytest.approx(60.0 + 0.25 + 15.0), over_120s
    assert over_300s == pytest.approx(60.0 + 0.25 + 15.0), "budget grew with the pcm"
    assert over_300s < 670.6, "the incident's own booking time must not be reachable"

    short = _player_timeout(at(1.0), cfg, monkeypatch)
    assert short == pytest.approx(1.0 + 0.25 + 15.0), short


def test_the_two_ceiling_keys_resolve_from_defaults(tmp_path):
    """Both keys have to survive a worker that never got a synced voice.json —
    `sync-voice-config.py` projects only six keys out of config.yaml, and these
    two are deliberately not among them, like `synth_timeout` and `max_chars`.
    The numbers are the ones the DEFAULTS comment derives: `max_chars: 240` is
    ~20 s of speech at `speed: 1.22`, so 60 s is ~3x the longest real alert and
    over ten times under the 655.57 s it played on 2026-09-28; 90 s is above the
    60 s `synth_timeout`, so a stalled socket still trips *that* bound first
    rather than having it turned into dead code."""
    cfg = speak.load_config(tmp_path)                       # no voice.json at all
    assert cfg["max_audio_seconds"] == 60.0
    assert cfg["synth_wall_clock"] == 90.0
    assert cfg["max_audio_seconds"] <= 655.57 / 10, "the ceiling must stay far under the incident"
    assert cfg["synth_wall_clock"] > cfg["synth_timeout"], (
        "a total below the per-socket timeout retires the per-socket timeout")

    # A voice.json that overrides *other* keys keeps both ceilings, so a synced
    # config cannot silently reopen the gap.
    (tmp_path / speak.CONFIG_NAME).write_text('{"speed": 1.1, "voice": "clone:other"}')
    overlaid = speak.load_config(tmp_path)
    assert overlaid["speed"] == 1.1
    assert overlaid["max_audio_seconds"] == 60.0 and overlaid["synth_wall_clock"] == 90.0


# ── when the synthesiser is the thing that broke (#1806) ──────────────
#
# On 2026-09-28 the guardian tried to say three sentences out loud and got
# `[Errno 111] Connection refused` — 11:50:51 and 11:51:18, the second of them the
# alert *announcing that agent-supervisord was unreachable* — and the only trace
# that no sound came out was a line in `voice.log`, the log of the channel that
# had just failed. The endpoint it posts to is `[program:agent-tts]`, inside the
# same supervised tree whose unreachability was being announced, so the report
# died with the thing it was reporting.
#
# The nodes below share one choice: the endpoint is a port that was bound and
# closed, so the failure is a real `ECONNREFUSED` through a real `urllib`, not a
# stubbed exception. Only the player subprocess is stubbed, because a test that
# makes noise on Alan's speakers is not a test.

_ALERT = "Guardian alert. supervisord was unreachable. Restarted agent"
_OTHER_ALERT = "Guardian alert. Landed: #1750 promotion settled"


def _refused_port() -> int:
    """A port bound and immediately closed, so a connect gets ECONNREFUSED."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _refusing(monkeypatch, **over):
    """`cfg` pointed at a refused port, a silent player, and witnesses.

    `attempts` records every URL the module asked `urlopen` to open; the probe
    forwards to the real function, so the refusal stays genuine and the list still
    proves what the failure path asked of the network. `seen` is what the player
    was handed — the WAV is read back from inside the stub, before `play` unlinks
    it, which is the only moment it exists.
    """
    cfg = _voice_cfg(api_url=f"http://127.0.0.1:{_refused_port()}", **over)
    attempts: list[str] = []
    seen: dict = {"attempts": attempts, "cmd": None}
    real_urlopen = speak.urllib.request.urlopen

    def probe(req, timeout=None, *args, **kw):
        attempts.append(req.full_url)
        return real_urlopen(req, timeout=timeout, *args, **kw)

    class _Proc0:
        returncode = 0

    def quiet_run(cmd, **kw):
        seen["cmd"] = list(cmd)
        seen["timeout"] = kw.get("timeout")
        with wave.open(cmd[-1], "rb") as w:
            seen["nframes"] = w.getnframes()
            seen["sampwidth"] = w.getsampwidth()
            seen["framerate"] = w.getframerate()
            seen["nchannels"] = w.getnchannels()
            seen["frames"] = w.readframes(w.getnframes())
        return _Proc0()

    monkeypatch.setattr(speak.urllib.request, "urlopen", probe)
    monkeypatch.setattr(speak.subprocess, "run", quiet_run)
    monkeypatch.setattr(speak.shutil, "which",
                        lambda exe: "/usr/bin/pw-play" if exe == "pw-play" else None)
    return cfg, seen


def test_a_refused_synth_leaves_a_loss_record_outside_voice_log(tmp_path, monkeypatch):
    """Clause 1. The record must name the utterance that was lost, and it must not
    be a line in the log of the channel that failed: `voice.log` truncates itself
    to empty at `_LOG_CAP` bytes, so a long burst can erase its own evidence.
    `voice-loss.md` is a separate file, created by this path."""
    cfg, _ = _refusing(monkeypatch)

    assert speak.speak_now(_ALERT, cfg, tmp_path) is False

    body = (tmp_path / speak.LOSS_NAME).read_text(encoding="utf-8")
    assert body.splitlines()[0] == speak.LOSS_HEADING
    assert _ALERT in body, "the record must name the alert that was lost"
    assert "occurrences: 1" in body
    assert "Connection refused" in body, "the record must carry the reason"
    log = (tmp_path / speak.LOG_NAME).read_text(encoding="utf-8")
    assert log.count("synth failed") == 1
    assert speak.LOSS_NAME != speak.LOG_NAME


def test_the_loss_record_needs_no_backend_api_to_be_written(tmp_path, monkeypatch):
    """Clause 2. #1798's precedent posts to `/api/backlog/task-create`, but
    `[program:lloyd-backend]` sits in the same supervised tree as
    `[program:agent-tts]`: on 2026-09-28 they were unreachable together, so a
    backlog-bound report would have gone down with the sound. Asserted both ways —
    the record is on disk, and nothing on this path ever opened a URL that was not
    the synth endpoint."""
    cfg, seen = _refusing(monkeypatch)

    assert speak.speak_now(_ALERT, cfg, tmp_path) is False

    assert (tmp_path / speak.LOSS_NAME).is_file()
    assert _ALERT in (tmp_path / speak.LOSS_NAME).read_text(encoding="utf-8")
    assert seen["attempts"] == [f"{cfg['api_url']}/v1/audio/speech"], seen["attempts"]
    assert not [u for u in seen["attempts"] if "/api/" in u], (
        "the loss record must not depend on the backend API, which shares the outage")


def test_a_second_failure_in_the_window_refreshes_the_one_record(tmp_path, monkeypatch):
    """Clause 3. The 2026-09-23 cluster was five refusals between 14:47 and 16:03;
    five records would read as five incidents and hide the one number that matters.
    Inside `LOSS_WINDOW` the same record is refreshed — count to 2, both
    utterances named — and past it the burst is over, so the next failure starts a
    new record instead of counting an outage that ended last week."""
    cfg, _ = _refusing(monkeypatch)

    assert speak.speak_now(_ALERT, cfg, tmp_path) is False
    assert speak.speak_now(_OTHER_ALERT, cfg, tmp_path) is False

    body = (tmp_path / speak.LOSS_NAME).read_text(encoding="utf-8")
    assert body.count(speak.LOSS_HEADING) == 1, "one record per burst, not one per failure"
    assert "occurrences: 2" in body, body
    assert _ALERT in body and _OTHER_ALERT in body
    assert len(list(tmp_path.glob("voice-loss*"))) == 1

    stale = re.sub(r"burst_started: [0-9.]+",
                   f"burst_started: {time.time() - speak.LOSS_WINDOW - 1.0:.3f}", body)
    (tmp_path / speak.LOSS_NAME).write_text(stale, encoding="utf-8")
    assert speak.speak_now(_ALERT, cfg, tmp_path) is False
    after = (tmp_path / speak.LOSS_NAME).read_text(encoding="utf-8")
    assert after.count(speak.LOSS_HEADING) == 1
    assert "occurrences: 1" in after, "a burst past its window starts over"


def test_the_failure_path_returns_within_two_seconds_and_never_raises(
        tmp_path, monkeypatch):
    """Clause 4. This runs in a detached child of a unit carrying
    `WatchdogSec=90` against a 5 s tick, and the chirp puts a player subprocess
    behind it, so a refused endpoint must cost under 2 s and no exception may
    reach `main` — the entry point the worker is spawned with, whose exit code
    nobody reads. Both hostile surfaces are exercised: a raising player, and a
    state dir that is a plain file so every write on the record path fails."""
    cfg, seen = _refusing(monkeypatch)

    started = time.monotonic()
    assert speak.speak_now(_ALERT, cfg, tmp_path) is False
    elapsed = time.monotonic() - started
    assert elapsed < 2.0, f"the failure path took {elapsed:.2f}s"
    assert 0 < seen["timeout"] <= 2.0, (
        f"the chirp asked for {seen['timeout']}s of speakers, which is not a bound")

    def hostile_run(cmd, **kw):
        raise OSError("player exploded")
    monkeypatch.setattr(speak.subprocess, "run", hostile_run)
    assert speak.speak_now(_ALERT, cfg, tmp_path) is False       # nothing raised

    # A state dir the record cannot be written into — a plain file where the
    # directory has to be. Asserted through `speak_now`, not through `main`:
    # `main` turns every exception into exit 1, so it reports the same result for
    # a guarded writer and an unguarded one, and cannot pin the claim at all.
    blocked = tmp_path / "not-a-directory"                       # a file, not a dir
    blocked.write_text("in the way", encoding="utf-8")
    assert speak.speak_now(_ALERT, cfg, blocked / "sub") is False
    assert speak.main(["--state-dir", str(blocked / "sub"), "--text", _ALERT]) == 1


def test_a_refused_synth_still_puts_locally_made_audio_in_the_player(
        tmp_path, monkeypatch):
    """Clause 5. The player survived that outage even though the synthesiser did
    not — `pw-play` is a separate binary — and in-process speech is not available
    here (espeak, espeak-ng, spd-say, festival, flite, pico2wave: all absent), so
    the degraded attempt is a chirp built in this process from `math` and `array`.
    Asserted on what the player was handed: one channel, 16-bit, the configured
    rate, with real signal in it — after exactly one request to the endpoint, the
    one that failed."""
    cfg, seen = _refusing(monkeypatch)

    assert speak.speak_now(_ALERT, cfg, tmp_path) is False

    assert seen["cmd"] is not None, "the player was never invoked"
    assert seen["cmd"][0] == "pw-play", seen["cmd"]
    assert seen["nframes"] > 0 and seen["sampwidth"] == 2
    assert seen["nchannels"] == 1
    assert seen["framerate"] == int(cfg["sample_rate"])
    pcm = array.array("h", seen["frames"])
    assert max(abs(min(pcm)), max(pcm)) > 0, "the fallback audio was all silence"
    heard = len(pcm) / float(cfg["sample_rate"])
    assert 0.5 <= heard <= 2.0, f"the chirp was {heard:.2f}s"
    assert seen["attempts"] == [f"{cfg['api_url']}/v1/audio/speech"], (
        "the fallback must not go back to the endpoint that just refused")


def test_a_synth_that_answers_with_no_audio_is_recorded_and_chirped(
        tmp_path, monkeypatch):
    """The other dead-silence shape: the server answers and hands back nothing.
    From the room that is the same failure, and the 21 `synth failed` lines are
    not all refusals — 19:45:47 on 2026-09-28 was a `TimeoutError` — so the
    handling cannot be keyed on the connection error alone."""
    cfg, seen = _refusing(monkeypatch)

    class _Empty:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n: int = -1) -> bytes:
            return b""

        def close(self) -> None:
            pass

    monkeypatch.setattr(speak.urllib.request, "urlopen",
                        lambda req, timeout=None: _Empty())
    assert speak.speak_now(_ALERT, cfg, tmp_path) is False

    body = (tmp_path / speak.LOSS_NAME).read_text(encoding="utf-8")
    assert "synth returned no audio" in body and _ALERT in body
    assert seen["nframes"] > 0, "the chirp did not reach the player"


# ── wiring into the fan-out ───────────────────────────────────────────

def _notifier(tmp_path, **kw):
    import notify
    (tmp_path / "obsidian" / "memory").mkdir(parents=True, exist_ok=True)
    kw.setdefault("backend_url", "http://127.0.0.1:1")   # nothing listens
    return notify.Notifier(ledger=tmp_path / "promotions.jsonl",
                           state_dir=tmp_path,
                           vault_root=str(tmp_path / "obsidian"), **kw)


def test_voice_is_below_the_external_gate(tmp_path):
    """The drill runs a real guardian against a throwaway repo. A rehearsal
    that announces a rollback out loud is indistinguishable from a production
    incident to anyone in the room — same argument as the vault note."""
    res = _notifier(tmp_path, external=False).alert(
        "critical", "drill rollback", "body", trigger="crash", commit="a" * 40)
    assert "voice" not in res, res
    assert set(res) == {"ledger", "alert_file"}, res


def test_alert_dispatches_voice_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("LLOYD_VOICE_ALERTS", "1")
    # Quiet hours are wall-clock, so without this the assertion below
    # depends on what time the suite runs. See _AWAKE.
    monkeypatch.setattr(speak, "in_quiet_hours", _AWAKE)
    monkeypatch.setattr(speak.subprocess, "Popen", lambda *a, **k: object())
    res = _notifier(tmp_path).alert("warn", "real rollback", "body")
    assert res["voice"] is True


def test_a_broken_voice_channel_cannot_break_the_alert(tmp_path, monkeypatch):
    """No channel may raise — the loop must survive a wrecked speaker."""
    monkeypatch.setenv("LLOYD_VOICE_ALERTS", "1")
    monkeypatch.setattr(speak, "dispatch",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    res = _notifier(tmp_path).alert("warn", "real rollback", "body")
    assert res["voice"] is False
    assert res["ledger"] is True, "an alert was lost to a speaker fault"


def test_announce_records_nothing(tmp_path, monkeypatch):
    """The nag fires every 15 minutes for as long as an incident lasts. If it
    went through `alert` that would be a ledger row and a backlog task every
    15 minutes, burying the task the rollback actually filed."""
    monkeypatch.setenv("LLOYD_VOICE_ALERTS", "1")
    monkeypatch.setattr(speak.subprocess, "Popen", lambda *a, **k: object())
    res = _notifier(tmp_path).announce("STILL BROKEN", "body", level="critical")

    assert set(res) == {"journal", "desktop", "voice"}, res
    assert not (tmp_path / "promotions.jsonl").exists(), "announce wrote to the ledger"
    assert not (tmp_path / "ALERT.md").exists(), "announce clobbered the alert file"
    assert "backlog" not in res


def test_announce_is_suppressible_like_every_external_channel(tmp_path):
    assert _notifier(tmp_path, external=False).announce("x", "y") == {}


# ── the nag ───────────────────────────────────────────────────────────

def _run_nag(tmp_path, broken: str | None):
    import os
    import subprocess as sp
    automod, guardian = tmp_path / "automod", tmp_path / "guardian"
    automod.mkdir(parents=True, exist_ok=True)
    guardian.mkdir(parents=True, exist_ok=True)
    if broken is not None:
        (automod / "BROKEN").write_text(broken)
    env = dict(os.environ,
               LLOYD_AUTOMOD_STATE=str(automod),
               LLOYD_GUARDIAN_STATE=str(guardian),
               LLOYD_VOICE_ALERTS="0",
               DBUS_SESSION_BUS_ADDRESS="")     # no real desktop toast from a test
    return sp.run([sys.executable, str(GUARDIAN_DIR / "nag.py")],
                  capture_output=True, text=True, env=env, timeout=30), automod


def test_nag_is_silent_when_nothing_is_broken(tmp_path):
    proc, _ = _run_nag(tmp_path, None)
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


def test_nag_announces_through_the_shared_fan_out(tmp_path):
    """It used to be an inline notify-send in the unit — a second, private
    definition of "tell the human" that could never gain a channel notify.py
    grew."""
    proc, automod = _run_nag(tmp_path, "2026-09-06 liveness failed\n")
    assert proc.returncode == 0, proc.stderr
    assert "voice=" in proc.stdout and "journal=" in proc.stdout
    assert not (automod / "promotions.jsonl").exists(), "the nag spammed the ledger"


def test_nag_reads_the_marker_file_not_the_incident_directory(tmp_path):
    """`BROKEN` (the marker) and `broken/` (per-incident dirs) live side by
    side in the same state dir and differ only by case."""
    import nag
    d = tmp_path / "s"
    (d / "broken").mkdir(parents=True)
    assert nag.read_broken(d / "BROKEN") is None
    (d / "BROKEN").write_text("  reason here  \n")
    assert nag.read_broken(d / "BROKEN") == "reason here"


# ── promotion announcements ───────────────────────────────────────────

def test_a_successful_promotion_is_announced(monkeypatch):
    """Until now the loop only ever spoke when it *failed* — every
    notify-send in the tree hung off a guardian alert. A system that can
    rewrite itself in the background and is silent when it works has it
    backwards: the successful landings are the ones nobody is watching a
    terminal for."""
    import notify
    from scripts.automod import promote as promote_mod

    seen = {}

    class FakeNotifier:
        def __init__(self, **kw):
            seen["init"] = kw

        def announce(self, title, body="", level="info"):
            seen.update(title=title, body=body, level=level)
            return {"voice": True}

    monkeypatch.setattr(notify, "Notifier", FakeNotifier)
    promote_mod._announce_promoted("SM_20260906_a1b2", "a" * 40,
                                   ["app/x.py", "tests/y.py"],
                                   title="Give deferred tool descriptions trigger conditions")

    assert seen["title"] == "Landed: Give deferred tool descriptions trigger conditions"
    assert "SM_20260906_a1b2" not in seen["title"] + seen["body"], (
        "the round id is bookkeeping; the toast names the work")
    assert "2 files" in seen["body"]
    assert seen["level"] == "info", "a success must not read as an incident"
    # And it must survive being read aloud: the round id keeps its date.
    spoken = speak.utterance_for("info", seen["title"], seen["body"])
    assert "deferred tool descriptions" in spoken
    assert "20260906" not in spoken, "read aloud, a round id is a date one digit at a time"


def test_one_changed_file_is_not_announced_as_1_files(monkeypatch):
    import notify
    from scripts.automod import promote as promote_mod
    seen = {}

    class FakeNotifier:
        def __init__(self, **kw):
            pass

        def announce(self, title, body="", level="info"):
            seen["body"] = body
            return {}

    monkeypatch.setattr(notify, "Notifier", FakeNotifier)
    promote_mod._announce_promoted("SM_x", "a" * 40, ["only.py"])
    assert "1 file changed" in seen["body"], seen


def test_announcing_cannot_fail_a_promotion(monkeypatch):
    """The promotion has already succeeded and is being observed by the time
    this runs. An announcement must never be able to turn that into a
    failure."""
    import notify
    from scripts.automod import promote as promote_mod

    def boom(**kw):
        raise RuntimeError("notifier exploded")

    monkeypatch.setattr(notify, "Notifier", boom)
    promote_mod._announce_promoted("SM_x", "a" * 40, ["a.py"])   # must not raise


# ── quiet hours ───────────────────────────────────────────────────────

def _at(hour: int) -> float:
    """A local-time epoch for `hour` today. Built through mktime so the test
    is correct in any timezone and across DST, which a fixed epoch is not."""
    lt = list(time.localtime())
    lt[3], lt[4], lt[5] = hour, 30, 0
    lt[8] = -1                      # let mktime work out DST
    return time.mktime(time.struct_time(lt))


def _cfg(**qh):
    base = {"enabled": True, "start": 23, "end": 7, "allow_critical": False}
    return dict(speak.DEFAULTS, quiet_hours={**base, **qh})


@pytest.mark.parametrize("hour,quiet", [
    (23, True), (0, True), (3, True), (6, True),    # inside the wrapped window
    (7, False), (12, False), (22, False),           # outside it
])
def test_quiet_window_wraps_midnight(hour, quiet):
    """23->7 is the normal shape, and it is the one a naive `start <= h < end`
    gets exactly backwards — it would be silent all day and loud all night."""
    assert speak.in_quiet_hours(_cfg(), _at(hour)) is quiet


@pytest.mark.parametrize("hour,quiet", [(1, False), (10, True), (13, True), (18, False)])
def test_quiet_window_without_a_wrap(hour, quiet):
    assert speak.in_quiet_hours(_cfg(start=9, end=17), _at(hour)) is quiet


def test_an_empty_window_is_no_window_not_all_day():
    """start == end has two readings and only one of them is survivable."""
    assert speak.in_quiet_hours(_cfg(start=7, end=7), _at(3)) is False
    assert speak.in_quiet_hours(_cfg(start=7, end=7), _at(15)) is False


def test_quiet_hours_off_by_default_config_and_on_garbage():
    assert speak.in_quiet_hours({}, _at(3)) is False
    assert speak.in_quiet_hours(_cfg(enabled=False), _at(3)) is False
    assert speak.in_quiet_hours({"quiet_hours": {"enabled": True}}, _at(3)) is False
    assert speak.in_quiet_hours(_cfg(start="late", end=7), _at(3)) is False


def test_quiet_hours_withhold_speech_but_nothing_else(tmp_path, monkeypatch):
    """Only the sound is dropped. Every recording channel already ran by the
    time dispatch is reached, which is what makes defaulting this on safe."""
    monkeypatch.setenv("LLOYD_VOICE_ALERTS", "1")
    monkeypatch.setattr(speak, "in_quiet_hours", lambda cfg, now=None: True)
    spawned = []
    monkeypatch.setattr(speak.subprocess, "Popen", lambda *a, **k: spawned.append(a))

    assert speak.dispatch("error", "Rolled back", "b", tmp_path, window=3600) is False
    assert spawned == []
    assert "quiet hours" in (tmp_path / speak.LOG_NAME).read_text()


def test_allow_critical_wakes_you_for_a_rollback(tmp_path, monkeypatch):
    monkeypatch.setenv("LLOYD_VOICE_ALERTS", "1")
    monkeypatch.setattr(speak.subprocess, "Popen", lambda *a, **k: object())
    monkeypatch.setattr(speak, "in_quiet_hours", lambda cfg, now=None: True)
    (tmp_path / speak.CONFIG_NAME).write_text(json.dumps(
        {"quiet_hours": {"enabled": True, "start": 23, "end": 7, "allow_critical": True}}))

    assert speak.dispatch("critical", "Rolled back", "b", tmp_path, window=3600) is True
    assert speak.dispatch("error", "Something else", "b", tmp_path, window=3600) is False


def test_a_withheld_utterance_does_not_burn_the_repeat_slot(tmp_path, monkeypatch):
    """The ordering bug this guards: should_speak *records* as it decides, so
    checking it before the clock would spend the hourly slot on an utterance
    nobody heard — and the 08:00 repeat of an 03:00 alert would stay silent
    for the wrong reason."""
    monkeypatch.setenv("LLOYD_VOICE_ALERTS", "1")
    spawned = []
    monkeypatch.setattr(speak.subprocess, "Popen", lambda *a, **k: spawned.append(a))

    quiet = {"v": True}
    monkeypatch.setattr(speak, "in_quiet_hours", lambda cfg, now=None: quiet["v"])
    assert speak.dispatch("error", "Rolled back", "b", tmp_path, window=3600) is False
    assert not (tmp_path / speak.SPOKEN_NAME).exists(), "a silent drop was recorded"

    quiet["v"] = False              # morning
    assert speak.dispatch("error", "Rolled back", "b", tmp_path, window=3600) is True
    assert len(spawned) == 1


def test_partial_override_keeps_its_sibling_defaults(tmp_path):
    """A voice.json that moves only the start hour must not drop `enabled`."""
    (tmp_path / speak.CONFIG_NAME).write_text('{"quiet_hours": {"start": 22}}')
    qh = speak.load_config(tmp_path)["quiet_hours"]
    assert qh == {"enabled": True, "start": 22, "end": 7, "allow_critical": False}


def test_sync_pushes_quiet_hours_from_the_guardian_block_not_livekit():
    """livekit_worker reads livekit.tts. If quiet hours lived there, a voice
    conversation would go mute at 23:00 — an alert policy leaking into the
    thing it must never touch."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "svc", Path(__file__).resolve().parent.parent
        / "agent-services" / "bin" / "sync-voice-config.py")
    svc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(svc)

    tts = {"voice": "clone:dave_cullen", "speed": 0.85}
    assert "quiet_hours" not in svc.build(tts, {})
    out = svc.build(tts, {"quiet_hours": {"enabled": True, "start": 1, "end": 2}})
    assert out["quiet_hours"]["start"] == 1
    assert out["voice"] == "clone:dave_cullen"


# ── the stamp on each log line (#1808) ────────────────────────────────
# Everything above reads the MESSAGE of a log line; none of it read the stamp,
# which is why the stamp could carry no zone for as long as it did. The file's
# clock is the local wall clock — `timedatectl` says America/Los_Angeles,
# PDT/-0700 — while its two sibling writers in the same state dir mark UTC
# explicitly (`gstate.py:26` writes `%...Z`, `memwatch.py:210` writes
# gmtime+`Z`). A reader that assumes UTC for the guardian's own artefacts is
# right for those and hours wrong here.

def _stamp_and_body(tmp_path) -> tuple[str, str]:
    """The first `voice.log` line, split at its first space: stamp, message."""
    text = (tmp_path / speak.LOG_NAME).read_text(encoding="utf-8")
    stamp, _, body = text.splitlines()[0].partition(" ")
    return stamp, body


def test_a_log_line_stamps_an_explicit_utc_offset(tmp_path):
    """#1808 clause 1: a machine-facing log must say which zone it is in.

    The ambiguity is not hypothetical. The two `Connection refused` lines at
    11:50:51 and 11:51:18 on 2026-09-28 — the refused utterance that was itself
    the alert announcing `agent-supervisord` was unreachable — read as UTC sit
    59 minutes BEFORE the outage they belong to. The signals pass read them as
    PDT only because it stopped to check.
    """
    speak._log(tmp_path, "quiet hours — withholding speech")

    stamp, _ = _stamp_and_body(tmp_path)
    parsed = datetime.datetime.fromisoformat(stamp)
    assert parsed.utcoffset() is not None, (
        f"{stamp!r} carries no offset, so every later reader has to guess "
        "whether it is local or UTC")


def test_the_offset_never_moves_the_wall_clock_reading(tmp_path, monkeypatch):
    """#1808 clause 2: marking the zone is not the same act as converting to UTC.

    `datetime.now(timezone.utc)` would satisfy clause 1, keep this file's green
    record, and silently shift every line by seven hours — and the incident this
    item cites is understood only by lining these lines against a human's memory
    of the evening. So the instant is frozen, and the digits are compared against
    the LOCAL reading of that same instant, computed with the real `strftime`
    captured before the patch so the comparison shares no code with the writer.
    """
    moment = time.time()
    frozen = time.localtime(moment)
    real_strftime = time.strftime
    monkeypatch.setattr(time, "strftime",
                        lambda fmt, *a: real_strftime(fmt, *(a or (frozen,))))

    speak._log(tmp_path, "digits must be unchanged")

    stamp, _ = _stamp_and_body(tmp_path)
    parsed = datetime.datetime.fromisoformat(stamp)
    assert parsed.strftime("%Y-%m-%dT%H:%M:%S") == real_strftime(
        "%Y-%m-%dT%H:%M:%S", frozen), (
        f"the stamp reads {stamp!r} but the local wall clock at that moment "
        f"was {real_strftime('%Y-%m-%dT%H:%M:%S', frozen)!r} — the log must "
        "keep writing local time and only MARK the zone")
    assert parsed.utcoffset() == datetime.datetime.fromtimestamp(
        moment, datetime.timezone.utc).astimezone().utcoffset(), (
        f"{stamp!r} names the wrong offset for this machine")


def test_log_swallows_an_unwritable_state_dir_and_returns_none(tmp_path):
    """#1808 clause 3's first half: the alerting path may not raise.

    Two ways of making the write fail, because they fail at different places —
    a state dir that is a regular FILE makes `mkdir` raise `FileExistsError`
    whatever user runs the suite, and a directory without write permission makes
    the `open` fail. The permission half is skipped rather than passed vacuously
    under root, where mode bits are advisory.
    """
    as_file = tmp_path / "not-a-dir"
    as_file.write_text("in the way", encoding="utf-8")

    assert speak._log(as_file, "must not raise") is None
    assert not (tmp_path / speak.LOG_NAME).exists()

    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root ignores mode bits, so the permission case proves nothing")
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        assert speak._log(locked, "must not raise either") is None
        assert not (locked / speak.LOG_NAME).exists()
    finally:
        locked.chmod(0o700)


def test_the_cap_truncates_above_it_before_appending_and_not_below(tmp_path):
    """#1808 clause 3's second half, with its control: the size bound still fires.

    `_LOG_CAP` had no test at all, and the bound is load-bearing enough that
    `#1806` records its downside in prose: the log that documents a failed
    channel is the one channel can fill. Two properties, because one alone is
    satisfiable by an unconditional truncate — above the cap the file is emptied
    BEFORE the new line is appended, and below it nothing is dropped.
    """
    log = tmp_path / speak.LOG_NAME
    log.write_text("x" * (speak._LOG_CAP + 10), encoding="utf-8")

    speak._log(tmp_path, "after the cap")
    text = log.read_text(encoding="utf-8")
    assert "xxxx" not in text, "a log past _LOG_CAP was not truncated"
    assert text.count("\n") == 1, f"expected only the new line, got {text!r}"
    assert text.endswith("after the cap\n"), (
        "the truncate must happen before the append, not after it")

    speak._log(tmp_path, "below the cap")
    text = log.read_text(encoding="utf-8")
    assert text.count("\n") == 2, f"appending below the cap dropped a line: {text!r}"
    assert "after the cap" in text and "below the cap" in text


def test_the_message_is_written_verbatim_after_the_stamp(tmp_path):
    """#1808 clause 4: the stamp is a prefix, and only a prefix.

    Every pre-existing assertion here reads the body — "degraded",
    "unavailable", "duration ceiling", "quiet hours" — and this change edits
    none of them. What they all lean on is that the text after the stamp is the
    text that was handed in, so that is what is asserted, with the characters a
    real alert carries: an em dash, a quoted path, and parentheses.
    """
    msg = "synth failed (TimeoutError) — read timed out on '/tmp/voice.wav' (twice)"

    speak._log(tmp_path, msg)

    stamp, body = _stamp_and_body(tmp_path)
    assert body == msg, f"the body was altered: {body!r}"
    assert (tmp_path / speak.LOG_NAME).read_text(encoding="utf-8").startswith(
        stamp + " "), "the stamp must be the first token on the line"
