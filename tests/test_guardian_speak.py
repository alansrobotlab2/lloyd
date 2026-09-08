"""The guardian's spoken channel.

Three things are worth pinning here, and they are the three that actually
went wrong while this was being built:

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

Nothing here plays audio: `conftest` sets LLOYD_VOICE_ALERTS=0 for every test
and the two tests that need dispatch to proceed re-enable it with a stubbed
`subprocess.Popen`.
"""

from __future__ import annotations

import json
import sys
import time
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
    ("Clear ~/.local/state/lloyd-selfmod/BROKEN once resolved", "lloyd-selfmod"),
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

    That is not academic: the unattended selfmod loop gates overnight, and the
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
    selfmod, guardian = tmp_path / "selfmod", tmp_path / "guardian"
    selfmod.mkdir(parents=True, exist_ok=True)
    guardian.mkdir(parents=True, exist_ok=True)
    if broken is not None:
        (selfmod / "BROKEN").write_text(broken)
    env = dict(os.environ,
               LLOYD_SELFMOD_STATE=str(selfmod),
               LLOYD_GUARDIAN_STATE=str(guardian),
               LLOYD_VOICE_ALERTS="0",
               DBUS_SESSION_BUS_ADDRESS="")     # no real desktop toast from a test
    return sp.run([sys.executable, str(GUARDIAN_DIR / "nag.py")],
                  capture_output=True, text=True, env=env, timeout=30), selfmod


def test_nag_is_silent_when_nothing_is_broken(tmp_path):
    proc, _ = _run_nag(tmp_path, None)
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


def test_nag_announces_through_the_shared_fan_out(tmp_path):
    """It used to be an inline notify-send in the unit — a second, private
    definition of "tell the human" that could never gain a channel notify.py
    grew."""
    proc, selfmod = _run_nag(tmp_path, "2026-09-06 liveness failed\n")
    assert proc.returncode == 0, proc.stderr
    assert "voice=" in proc.stdout and "journal=" in proc.stdout
    assert not (selfmod / "promotions.jsonl").exists(), "the nag spammed the ledger"


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
    from scripts.selfmod import promote as promote_mod

    seen = {}

    class FakeNotifier:
        def __init__(self, **kw):
            seen["init"] = kw

        def announce(self, title, body="", level="info"):
            seen.update(title=title, body=body, level=level)
            return {"voice": True}

    monkeypatch.setattr(notify, "Notifier", FakeNotifier)
    promote_mod._announce_promoted("SM_20260906_a1b2", "a" * 40,
                                   ["app/x.py", "tests/y.py"])

    assert "SM_20260906_a1b2" in seen["title"]
    assert "2 files" in seen["body"]
    assert seen["level"] == "info", "a success must not read as an incident"
    # And it must survive being read aloud: the round id keeps its date.
    assert "20260906" in speak.utterance_for("info", seen["title"], seen["body"])


def test_one_changed_file_is_not_announced_as_1_files(monkeypatch):
    import notify
    from scripts.selfmod import promote as promote_mod
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
    from scripts.selfmod import promote as promote_mod

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
