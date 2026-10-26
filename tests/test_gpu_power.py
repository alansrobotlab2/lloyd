"""The Services tab's GPU power switch (`app/gpu_power.py`) and the two guardian
readers of its marker."""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

import pytest

from app import gpu_power

REPO = Path(__file__).resolve().parents[1]
CONF_D = REPO / "agent-services" / "supervisor" / "conf.d"
sys.path.insert(0, str(REPO / "agent-services" / "guardian"))
import speak  # noqa: E402


def test_every_conf_that_pins_a_gpu_is_switched():
    """A new engine conf that names a card must join `GPU_PROGRAMS`, or the red
    button leaves it running."""
    pins = re.compile(r'CUDA_VISIBLE_DEVICES=|\bGPU="')
    for conf in CONF_D.glob("*.conf"):
        text = conf.read_text(encoding="utf-8")
        env = [line for line in text.splitlines() if line.startswith("environment=")]
        if any(pins.search(line) for line in env):
            name = re.search(r"^\[program:([^\]]+)\]", text, re.M).group(1)
            assert name in gpu_power.GPU_PROGRAMS, f"{conf.name} pins a GPU"
    # Its card is pinned in the launcher, not the conf.
    assert gpu_power.PRIMARY in gpu_power.GPU_PROGRAMS


def test_speak_and_backend_agree_on_the_marker_name():
    assert speak.GPU_OFF_MARKER == gpu_power.MARKER_NAME


class _Pool:
    def __init__(self, paused_by=()):
        self.paused_by = list(paused_by)
        self.calls = []

    def pause(self, paused, owner="operator", holder_kind="person", actor=None):
        self.calls.append(paused)
        self.paused_by = ["operator"] if paused else []


@pytest.fixture
def box(tmp_path, monkeypatch):
    """A fake supervisord, pool and lock, and a guardian dir under tmp."""
    monkeypatch.setenv("LLOYD_GUARDIAN_STATE", str(tmp_path))
    states = {"agent-llm-primary": "RUNNING", "agent-tts": "RUNNING",
              "agent-djev": "STOPPED", "lloyd-backend": "RUNNING"}
    log = []

    def stop(p, wait=False):
        log.append(("stop", p)); states[p] = "STOPPED"; return True, "stopped"

    def start(p, wait=False):
        log.append(("start", p)); states[p] = "RUNNING"; return True, "started"

    monkeypatch.setattr(gpu_power, "_supervisor_all_lenient",
                        lambda: {k: {"statename": v} for k, v in states.items()})
    monkeypatch.setattr(gpu_power, "process_info", lambda p: {"statename": states[p]})
    monkeypatch.setattr(gpu_power, "stop_process", stop)
    monkeypatch.setattr(gpu_power, "start_process", start)
    monkeypatch.setattr(gpu_power, "_take_regression_lock", lambda: None)
    monkeypatch.setattr(gpu_power, "_drop_regression_lock", lambda: None)
    monkeypatch.setattr(gpu_power, "_mem_available_gib", lambda: 400)
    monkeypatch.setattr(gpu_power, "_health_ok", lambda url: True)
    pool = _Pool()
    import workers.pool as wp
    monkeypatch.setattr(wp, "get_pool", lambda: pool)
    return {"states": states, "log": log, "pool": pool, "dir": tmp_path}


def _run(coro):
    asyncio.run(coro)


def test_off_then_on_round_trips_exactly_what_was_up(box):
    assert gpu_power.begin_off() is None
    _run(gpu_power.power_off())
    assert gpu_power.status()["phase"] == "off"
    assert box["pool"].calls == [True]
    stopped = sorted(p for verb, p in box["log"] if verb == "stop")
    assert stopped == ["agent-llm-primary", "agent-tts"]   # djev was already down
    assert box["states"]["lloyd-backend"] == "RUNNING"
    # The guardian and the speaker read the same file.
    assert "agent-tts" in speak.gpu_off_programs(box["dir"])

    assert gpu_power.begin_on() is None
    _run(gpu_power.power_on())
    started = [p for verb, p in box["log"] if verb == "start"]
    assert started == ["agent-llm-primary", "agent-tts"]    # primary first
    assert box["states"]["agent-djev"] == "STOPPED"
    assert box["pool"].calls == [True, False]
    assert gpu_power.status()["phase"] == "on"
    assert speak.gpu_off_programs(box["dir"]) == frozenset()


def test_an_operator_pause_taken_earlier_survives_the_round_trip(box):
    box["pool"].paused_by = ["operator"]
    gpu_power.begin_off(); _run(gpu_power.power_off())
    gpu_power.begin_on(); _run(gpu_power.power_on())
    assert box["pool"].calls == []


def test_low_host_ram_leaves_everything_off(box, monkeypatch):
    from scripts.automod import ram_gate
    monkeypatch.setattr(ram_gate, "PRIMARY_RAM_WAIT_SECONDS", 0.0)
    gpu_power.begin_off(); _run(gpu_power.power_off())
    monkeypatch.setattr(gpu_power, "_mem_available_gib", lambda: 10)
    gpu_power.begin_on(); _run(gpu_power.power_on())
    st = gpu_power.status()
    assert st["phase"] == "off" and "host RAM" in st["error"]
    assert not [p for verb, p in box["log"] if verb == "start"]
    assert box["pool"].calls == [True]


def test_refusals(box):
    assert gpu_power.begin_on() == "GPU services are not off"
    gpu_power.begin_off(); _run(gpu_power.power_off())
    assert gpu_power.begin_off() == "GPU services are already off"


def test_speech_is_withheld_while_off(tmp_path, monkeypatch):
    monkeypatch.setattr(speak, "voice_enabled", lambda: True)
    monkeypatch.setattr(speak, "in_quiet_hours", lambda cfg, now=None: False)
    spawned = []
    monkeypatch.setattr(speak.subprocess, "Popen", lambda *a, **k: spawned.append(a))
    (tmp_path / speak.GPU_OFF_MARKER).write_text(json.dumps({"programs": ["agent-tts"]}))
    assert speak.dispatch("error", "t", "b", tmp_path, window=0.0) is False
    assert not spawned
