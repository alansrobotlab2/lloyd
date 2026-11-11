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
    button leaves it running.

    Reads through `_unswitched`, the one spelling of the rule, which is also what the bite
    node runs over a fixture conf.d — so the rule cannot differ between the rail and the
    node that proves the rail bites.
    """
    unswitched = _unswitched(CONF_D)
    assert not unswitched, f"conf pins a GPU and is not in GPU_PROGRAMS: {unswitched}"
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


def _conf_pins(conf_dir: Path) -> dict[str, str]:
    """{program: the `environment=` line that names its card} for one conf.d.

    The ONE card-pin extraction in this file, which is the point of it: the rail
    `test_every_conf_that_pins_a_gpu_is_switched` calls it over `CONF_D` and the node below
    calls it over the synthetic directory holding the removed tenant's conf. Taking a
    directory is what lets the fixture be scanned by the rule the rail runs rather than by a
    second copy of it — a copy reachable only from the fixture is the failure mode the rail
    would never see, because drift there leaves the rail green while the pin it exists to
    catch goes unswitched. Both call sites reading one body is what makes that drift
    impossible, so no separate equality node is needed.
    """
    pins = re.compile(r'CUDA_VISIBLE_DEVICES=|\bGPU="')
    out: dict[str, str] = {}
    for conf in sorted(conf_dir.glob("*.conf")):
        text = conf.read_text(encoding="utf-8")
        env = [ln for ln in text.splitlines() if ln.startswith("environment=")]
        hit = [ln for ln in env if pins.search(ln)]
        if hit:
            out[re.search(r"^\[program:([^\]]+)\]", text, re.M).group(1)] = hit[0]
    return out


def _unswitched(conf_dir: Path) -> list[str]:
    """Programs in `conf_dir` that pin a card and are missing from `GPU_PROGRAMS`.

    The ONE spelling of the cross-check: the rail calls it over the live conf.d and the bite
    node calls it over a fixture one. It is kept apart from `_conf_pins` because those are two
    judgements — "this conf names a card" and "that program is not switchable" — and only the
    second is the rule whose teeth have to be demonstrated. A fixture that exercises the first
    alone proves the parser runs and says nothing about the comparison, which is exactly the
    gap a #4029-shaped change falls through: the conf went out of `conf.d` and the name stayed
    in the list, and the reverse direction (a conf whose program IS listed) is what tells a
    rule that reports nothing from a rule that reports the right thing.
    """
    return [name for name in _conf_pins(conf_dir) if name not in gpu_power.GPU_PROGRAMS]


def test_gpu_2_has_two_switchable_tenants_and_the_scan_still_bites(tmp_path):
    """#4029 deleted `agent-decider.conf`; the switch list followed, and the cross-check
    that would catch the next one still has teeth.

    Three things, none of which the node above can say on its own:

      * GPU 2's queue is exactly `agent-djev` (armed, `djev.enabled: true`) and
        `agent-llm-secondary` (pinned, disarmed) — two confs on the card, down from three
        when #4029 took `agent-decider.conf` out;
      * every conf-distributed card-pin is in the list on the live tree, which is the
        node above's assertion restated through `_unswitched`, the one spelling of it;
      * and a fixture conf.d holding one program the list has never heard of beside one it
        has is reported as exactly the first. The live tree has no such conf today, so
        without this the check's own teeth are untested — a rule that stopped comparing
        against `GPU_PROGRAMS` would leave the node above vacuously green, which is the
        shape of every guard this repo has had to rewrite.
    """
    pins = _conf_pins(CONF_D)
    gpu2 = sorted(p for p, env in pins.items() if 'GPU="2"' in env)
    assert gpu2 == ["agent-djev", "agent-llm-secondary"], pins
    assert _unswitched(CONF_D) == [], (
        f"confs pin cards that the switch list omits: {_unswitched(CONF_D)}")

    # Two confs, both pinning GPU 2 in the real conf's shape: one whose program is in the
    # switch list, one whose is not. The listed one is what makes the second assertion a
    # verdict rather than a tautology — a rule that reported EVERY pinned program would pass
    # a fixture holding only the unlisted one.
    (tmp_path / "agent-synthetic-engine.conf").write_text(
        '[program:agent-synthetic-engine]\n'
        'environment=HOME="/nonexistent",GPU="2",PORT="8013"\n', encoding="utf-8")
    (tmp_path / "agent-djev.conf").write_text(
        '[program:agent-djev]\n'
        'environment=HOME="/nonexistent",GPU="2",PORT="8010"\n', encoding="utf-8")
    assert sorted(_conf_pins(tmp_path)) == ["agent-djev", "agent-synthetic-engine"], (
        "the parser read neither conf, so every claim below this one is about a corpus of "
        "nothing and the rail above would be green on a conf.d it no longer reads")
    assert _unswitched(tmp_path) == ["agent-synthetic-engine"], (
        "`_unswitched` is the body `test_every_conf_that_pins_a_gpu_is_switched` runs; if it "
        "names nothing here the rail has no teeth, and if it names the listed conf too it has "
        "stopped comparing against `GPU_PROGRAMS` at all")
    assert "agent-decider" not in gpu_power.GPU_PROGRAMS, (
        "the removed unit's name is back in the switch list, which is the half of #4029 that "
        "no conf-side check can see once its conf is gone")
