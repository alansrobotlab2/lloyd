"""The memory-pressure recorder: evidence for the next oomd kill of the stack.

2026-09-15 04:48:34Z: systemd-oomd killed `agent-supervisord.service` (796
processes, 174.6 GiB) and nothing recorded which process had grown. oomd keys
on `app.slice`'s pressure and kills the descendant reclaiming most, so the
recorder watches the unit, the slice and the host, and lists every process
with its cgroup. Fake cgroup and /proc trees throughout; the placement test
drives the real `Guardian.tick`.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent-services" / "guardian"))

import guardian as G  # noqa: E402
import memwatch as MW  # noqa: E402
import policy  # noqa: E402


def _psi(full_avg10: float) -> str:
    return (f"some avg10={full_avg10 + 1:.2f} avg60=1.00 avg300=0.50 total=100\n"
            f"full avg10={full_avg10:.2f} avg60=0.80 avg300=0.40 total=90\n")


def _cgroups(tmp_path: Path, *, unit=0.0, slice_=0.0, host=0.0) -> tuple[Path, Path]:
    unit_dir = tmp_path / "cg" / "app.slice" / "agent-supervisord.service"
    unit_dir.mkdir(parents=True, exist_ok=True)
    (unit_dir / "memory.pressure").write_text(_psi(unit))
    (unit_dir.parent / "memory.pressure").write_text(_psi(slice_))
    (unit_dir / "memory.current").write_text(str(174 * 2**30))
    (unit_dir / "memory.stat").write_text("anon 1048576000\nfile 2097152\nshmem 140737488355328\n")
    host_file = tmp_path / "host-pressure"
    host_file.write_text(_psi(host))
    return unit_dir, host_file


def _set(unit_dir: Path, host_file: Path, *, unit=0.0, slice_=0.0, host=0.0) -> None:
    (unit_dir / "memory.pressure").write_text(_psi(unit))
    (unit_dir.parent / "memory.pressure").write_text(_psi(slice_))
    host_file.write_text(_psi(host))


def test_pressure_parses_the_kernel_format():
    p = MW.parse_pressure("some avg10=87.36 avg60=42.58 avg300=12.09 total=281000000\n"
                          "full avg10=83.30 avg60=40.00 avg300=11.00 total=270000000\n")
    assert p["full"]["avg10"] == 83.30 and p["some"]["avg60"] == 42.58
    assert MW.parse_pressure("garbage\n") == {}


def test_quiet_pressure_writes_nothing(tmp_path):
    unit, host = _cgroups(tmp_path, unit=14.9, slice_=3.0, host=2.0)
    w = MW.MemWatch(tmp_path / "gstate", unit, host)
    assert w.tick(now=1000.0) is None
    assert not (tmp_path / "gstate" / "mem-pressure").exists()


@pytest.mark.parametrize("where", ["unit", "slice_", "host"])
def test_pressure_anywhere_oomd_looks_triggers_a_snapshot(tmp_path, where):
    """oomd's own decision reads `app.slice`: a browser can push the slice past
    the limit and the stack is still the one killed."""
    unit, host = _cgroups(tmp_path, **{where: 40.0})
    w = MW.MemWatch(tmp_path / "gstate", unit, host)
    path = w.tick(now=1000.0)
    assert path is not None and path.exists()
    snap = json.loads(path.read_text())
    assert snap["peak_full_avg10"] == 40.0
    assert set(snap["pressure"]) == {"unit", "slice", "host"}
    assert snap["processes"], "the real /proc has at least this process"
    assert snap["unit_memory_current_kb"] == 174 * 2**20
    assert snap["unit_memory_stat_kb"]["shmem"] == 140737488355328 // 1024
    assert "MemAvailable" in snap["meminfo_kb"]


def test_snapshots_are_rate_limited_and_bounded(tmp_path):
    unit, host = _cgroups(tmp_path, unit=60.0)
    w = MW.MemWatch(tmp_path / "gstate", unit, host, min_interval=15.0, keep=3)
    top = MW.top_processes
    MW.top_processes = lambda: ([], True)
    try:
        assert w.tick(now=1000.0) is not None
        assert w.tick(now=1010.0) is None, "inside the interval"
        written = [w.tick(now=1000.0 + 20 * i) for i in range(1, 6)]
    finally:
        MW.top_processes = top
    assert all(written)
    kept = sorted((tmp_path / "gstate" / "mem-pressure").glob("*.json"))
    assert len(kept) == 3 and kept[-1] == written[-1], "newest kept"
    assert not list((tmp_path / "gstate" / "mem-pressure").glob("*.tmp"))


def test_a_missing_unit_cgroup_is_not_an_error(tmp_path):
    """The unit's cgroup does not exist while it is down — the moment after a kill."""
    _, host = _cgroups(tmp_path, host=30.0)
    w = MW.MemWatch(tmp_path / "gstate", tmp_path / "nope" / "agent-supervisord.service", host)
    path = w.tick(now=1000.0)
    assert path is not None
    snap = json.loads(path.read_text())
    assert snap["pressure"]["unit"] is None and "unit_memory_current_kb" not in snap


def _fake_proc(root: Path, procs: dict[int, dict]) -> Path:
    for pid, p in procs.items():
        d = root / str(pid)
        d.mkdir(parents=True)
        (d / "status").write_text(
            f"Name:\t{p['name']}\n"
            + (f"VmRSS:\t{p['rss']} kB\nRssAnon:\t{p.get('anon', 0)} kB\n"
               f"RssFile:\t0 kB\nRssShmem:\t{p.get('shm', 0)} kB\nVmSwap:\t0 kB\n"
               if p.get("rss") else ""))
        (d / "cmdline").write_bytes(p.get("cmd", p["name"]).encode().replace(b" ", b"\0"))
    (root / "self").mkdir()
    return root


def test_processes_are_listed_by_rss_and_kernel_threads_skipped(tmp_path):
    proc = _fake_proc(tmp_path / "proc", {
        10: {"name": "kthreadd"},
        20: {"name": "VLLM::EngineCor", "rss": 7_000_000, "shm": 95_000_000},
        30: {"name": "chrome", "rss": 9_000_000, "anon": 8_000_000, "cmd": "chrome --type=renderer"},
        40: {"name": "python", "rss": 1_000_000},
    })
    rows, complete = MW.top_processes(limit=2, proc=proc)
    assert complete and [r["pid"] for r in rows] == [30, 20]
    assert rows[0]["anon_kb"] == 8_000_000 and rows[0]["cmdline"] == "chrome --type=renderer"
    assert rows[1]["shmem_kb"] == 95_000_000


def test_the_walk_stops_at_its_budget(tmp_path):
    """Under the pressure this exists for, /proc reads can stall; a watchdog
    that pings every 5 s against a 90 s limit cannot wait on them."""
    proc = _fake_proc(tmp_path / "proc", {i: {"name": "p", "rss": i} for i in range(1, 50)})
    rows, complete = MW.top_processes(proc=proc, budget=-1.0)
    assert rows == [] and complete is False


# ── placement: every state the tick returns from ────────────────────────────

@pytest.fixture
def guardian(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, "LOG_FILES", ())
    args = types.SimpleNamespace(
        repo=str(tmp_path), state=str(tmp_path / "state"),
        guardian_state=str(tmp_path / "gstate"), supervisor_sock="/nonexistent",
        backend_url="http://127.0.0.1:1/health", mcp_url="http://127.0.0.1:2/health",
        programs="lloyd-mc:lloyd-backend", interval=5.0,
    )
    g = G.Guardian(args)
    monkeypatch.setattr(g, "check_vault", lambda: None)
    monkeypatch.setattr(g, "alert", lambda *a, **k: None)
    monkeypatch.setattr(g, "ensure_chronic", lambda: None)
    monkeypatch.setattr(g, "heartbeat", lambda *a, **k: None)
    unit, host = _cgroups(tmp_path, unit=70.0)
    g.mem = MW.MemWatch(g.gdir, unit, host)
    return g


@pytest.mark.parametrize("condition", ["supervisord_unreachable", "paused", "broken", "armed"])
def test_pressure_is_recorded_whatever_state_the_tick_returns_from(guardian, monkeypatch, condition):
    g = guardian
    snap = {"now": 0.0, "supervisord": "unreachable" if condition == "supervisord_unreachable" else "ok",
            "procs": {}, "probes": {}}
    monkeypatch.setattr(g, "collect", lambda: snap)
    monkeypatch.setattr(g.state, "pause_remaining", lambda cap: 600.0 if condition == "paused" else 0.0)
    monkeypatch.setattr(g.state, "is_broken", lambda: condition == "broken")
    monkeypatch.setattr(g, "evaluate_liveness", lambda s: (False, "ok"))
    monkeypatch.setattr(g.state, "current", lambda: None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0, stdout=""))
    monkeypatch.setattr(MW, "top_processes", lambda: ([], True))
    g.tick()
    assert len(list((g.gdir / "mem-pressure").glob("*.json"))) == 1


def test_a_broken_recorder_never_breaks_the_tick(guardian, monkeypatch):
    g = guardian
    monkeypatch.setattr(g.mem, "tick", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    g.check_memory()   # does not raise


def test_selftest_covers_the_recorder():
    src = (ROOT / "agent-services" / "guardian" / "selftest.py").read_text()
    assert "memwatch" in src and "memory-pressure recorder reads PSI" in src


# ── the unit's slice moved; the slice oomd judges did not ──────────────
# Since 2026-09-17 agent-supervisord.service runs in its own lloyd.slice
# (Slice= in agent-services/systemd/agent-supervisord.service), outside the
# app.slice oomd watches. memwatch used to hardcode the unit under app.slice
# and read the "slice" as the unit's parent — after the move that is a path
# that does not exist and a slice nobody kills in.


def _user_manager(tmp_path: Path, unit_slice: str, *, unit=0.0, oomd_slice=0.0,
                  unit_slice_pressure=0.0) -> Path:
    mgr = tmp_path / "user@1000.service"
    unit_dir = mgr / unit_slice / MW.UNIT
    unit_dir.mkdir(parents=True)
    (unit_dir / "memory.pressure").write_text(_psi(unit))
    (unit_dir.parent / "memory.pressure").write_text(_psi(unit_slice_pressure))
    (mgr / "app.slice").mkdir(exist_ok=True)
    (mgr / "app.slice" / "memory.pressure").write_text(_psi(oomd_slice))
    return mgr


def test_the_unit_is_found_in_its_own_slice(tmp_path, monkeypatch):
    mgr = _user_manager(tmp_path, "lloyd.slice")
    monkeypatch.setattr(MW, "_user_manager", lambda: mgr)
    assert MW.unit_cgroup() == mgr / "lloyd.slice" / MW.UNIT


def test_the_unit_is_still_found_where_it_used_to_live(tmp_path, monkeypatch):
    mgr = _user_manager(tmp_path, "app.slice")
    monkeypatch.setattr(MW, "_user_manager", lambda: mgr)
    assert MW.unit_cgroup() == mgr / "app.slice" / MW.UNIT


def test_the_slice_read_is_the_one_oomd_judges_not_the_units_parent(tmp_path, monkeypatch):
    """The desktop's pressure is what kills now; lloyd.slice's is nobody's."""
    mgr = _user_manager(tmp_path, "lloyd.slice", oomd_slice=40.0, unit_slice_pressure=0.0)
    monkeypatch.setattr(MW, "_user_manager", lambda: mgr)
    w = MW.MemWatch(tmp_path / "state", host=tmp_path / "no-host")
    r = w.readings()
    assert r["slice"]["full"]["avg10"] == 40.0


def test_the_unit_is_re_resolved_when_a_restart_moves_it(tmp_path, monkeypatch):
    """The guardian outlives the unit; a restart is what changes its slice."""
    mgr = _user_manager(tmp_path, "app.slice", unit=10.0)
    monkeypatch.setattr(MW, "_user_manager", lambda: mgr)
    w = MW.MemWatch(tmp_path / "state", host=tmp_path / "no-host")
    assert w.readings()["unit"]["full"]["avg10"] == 10.0
    moved = mgr / "lloyd.slice" / MW.UNIT
    moved.mkdir(parents=True)
    (moved / "memory.pressure").write_text(_psi(55.0))
    import shutil
    shutil.rmtree(mgr / "app.slice" / MW.UNIT)
    assert w.readings()["unit"]["full"]["avg10"] == 55.0
