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
