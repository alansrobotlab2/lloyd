"""A `*.running` automod marker is live only when its pid is one of OURS.

`pid_alive` asks whether a pid exists; it never asks what the process *is*, and
pids are recycled. This box runs `kernel.pid_max = 4194304` and the counter wraps,
so a marker left by a killed gate or landing eventually names somebody else's
process and every reader calls that a running round. Measured on 2026-11-08:
`gate_in_progress("SM_20261016_164802")` returned `{'pid': 4694, 'started_iso':
'2026-10-16T17:37:45Z', 'by': 'run_gate'}` for a 23-day-dead round because pid
4694 was alive — and `/proc/4694/cmdline` was
`/usr/lib/thunderbird/thunderbird -contentproc -isForBrowser …`. One hand-run
calibrate launcher held for its entire 700 s on `SM_20261105_154814/land.running`
(pid 367826) the same way, on the `rounds_landing` leg rather than the gate one.

The process behind a recycled pid keeps changing, so the fix is "is this process
one of ours" (`marker_pid_is_live` → `identity_is_automod` on
`/proc/<pid>/cmdline`, `/proc/<pid>/stat` starttime versus the marker's own stamp
when the cmdline cannot be read) and never a match against one known engine.

These tests plant markers naming real, live processes and read them back through
the real `/proc`: the stale control is this pytest process's own pid (a python
running pytest, which is not automod), and the positive controls are real child
processes launched in the two forms production uses — `-m scripts.automod.round
<verb>` and `…/scripts/automod/round.py <verb>`. Only the unreadable-cmdline leg
is forced, because nothing makes `/proc/<pid>/cmdline` unreadable on demand for a
live pid; the timing it decides on is still the child's real kernel starttime.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from scripts.automod import state as S

RID = "SM_20261108_060009"
#: The marker shapes as written today. The keys and the file names are part of
#: the contract: clause 4 of #3926 says the identity check changes no format.
GATE_KEYS = {"pid", "started_at", "started_iso", "head", "by"}
LAND_KEYS = {"pid", "started_at", "started_iso", "by"}
FLUSH_KEYS = {"pid", "started_at", "started_iso", "by"}

#: A fixture package standing in for the real one in the positive-control
#: children: `python -m scripts.automod.waiter` and
#: `python …/scripts/automod/waiter.py <verb>` are the same argv SHAPES the
#: gate/land/flush children have, and the /proc bytes are the child's own.
_WAITER = "import time\ntime.sleep(180)\n"
#: An argv that `identity_is_automod` accepts, used where the point is that
#: identity is NOT the only leg.
_AUTOMOD_ARGV = "\0".join([sys.executable, "-m", "scripts.automod.round", "gate", RID, "\0"])


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    """Marker paths pointed at a tmp tree, plus a fixture `scripts.automod`
    package to launch children in; no live automod state is touched, and every
    child is killed and reaped at teardown so none can linger as a zombie."""
    (tmp_path / "rounds" / RID).mkdir(parents=True)
    monkeypatch.setattr(S, "ROUNDS_DIR", tmp_path / "rounds")
    monkeypatch.setattr(S, "STATE_DIR", tmp_path)
    monkeypatch.setattr(S, "FLUSH_MARKER_PATH", tmp_path / "flush.running")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "scripts" / "automod").mkdir()
    (tmp_path / "scripts" / "automod" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "scripts" / "automod" / "waiter.py").write_text(_WAITER, encoding="utf-8")
    world = {"tmp": tmp_path, "children": []}
    yield world
    for proc in world["children"]:
        proc.kill()
        proc.wait(timeout=10)


def _raw_cmdline(pid: int) -> bytes:
    try:
        return open(f"/proc/{int(pid)}/cmdline", "rb").read()
    except OSError:
        return b""


def _child(env, argv, *, module_root=None):
    """Spawn a real process and wait until `/proc` shows it having `exec`'d the
    requested program, so a later read is a read of that child and not of a fork
    that is still carrying this process's argv (which would name pytest, not
    automod, and read stale for the wrong reason).

    The readiness probe is a raw `/proc` read, not `S.read_proc_cmdline`: two of
    these tests patch that read away, and a helper asking it whether the child is
    up would then skip the very node it is preparing — and a skip inside a helper
    is a green test that ran nothing."""
    env_path = dict(os.environ)
    if module_root is not None:
        env_path["PYTHONPATH"] = str(module_root)
    log = open(env["tmp"] / "child.log", "ab")
    proc = subprocess.Popen([sys.executable, *argv], cwd=str(module_root or env["tmp"]),
                            env=env_path, stdout=log, stderr=subprocess.STDOUT)
    env["children"].append(proc)
    for _ in range(200):
        if b"automod" in _raw_cmdline(proc.pid):
            return proc
        time.sleep(0.02)
    raise AssertionError(f"child pid {proc.pid} never exec'd: {_raw_cmdline(proc.pid)!r}")


def _plant(kind: str, *, pid: int, started_at: float, rid: str = RID) -> dict:
    """Write one marker of the given kind by hand, with the stamp the caller
    chooses, in the exact JSON shape the writers produce."""
    iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_at))
    rec: dict = {"pid": int(pid), "started_at": started_at, "started_iso": iso}
    if kind == "gate":
        rec.update({"head": "0" * 40, "by": "run_gate"})
        path = S.gate_marker_path(rid)
    elif kind == "land":
        rec["by"] = "round.land"
        path = S.land_marker_path(rid)
    elif kind == "flush":
        rec["by"] = "cli"
        path = S.FLUSH_MARKER_PATH
    else:
        raise AssertionError(f"unknown marker kind {kind!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec), encoding="utf-8")
    return rec


def _lock_path(env, rid: str, *, pid=None, owner: str | None = None,
               since: str | None = None):
    """A lock payload in `Lock.acquire`'s shape: `{pid, owner, since}`, where
    `owner` is `land-<round_id>` and `since` is `now_iso()` TEXT, not a float.
    Written into the tmp tree — the live lock file is the audit trail and never a
    test's to touch."""
    path = env["tmp"] / "lock.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pid": int(pid if pid is not None else os.getpid()),
                                "owner": owner or f"land-{rid}",
                                "since": since or S.now_iso()}), encoding="utf-8")
    return path


def _all_readers(env, rid: str = RID, *, pid=None) -> list:
    """The five graded reads, in one call: gate, land, the cross-round land
    sweep, the single global flush marker, and the lock payload. The lock payload
    is (re)stamped to name `pid` so all five describe the same process."""
    lock = _lock_path(env, rid, pid=pid)
    return [S.gate_in_progress(rid), S.land_in_progress(rid), S.rounds_landing(),
            S.flush_in_progress(), S.landing_lock(lock)]


def _marker_bytes(kind: str) -> bytes:
    path = {"gate": S.gate_marker_path(RID), "land": S.land_marker_path(RID),
            "flush": S.FLUSH_MARKER_PATH}[kind]
    return path.read_bytes()


def _unused_pid() -> int:
    """A pid no process can hold: above `kernel.pid_max` on this box, so
    `pid_alive` rejects it on that alone."""
    try:
        pid_max = int(open("/proc/sys/kernel/pid_max").read().strip())
    except OSError:
        pid_max = 32768
    return pid_max + 7


# --- clause 1: a live pid that is not automod is not a running round -------


def test_a_live_foreign_pid_makes_every_reader_say_nothing_is_running(env):
    """A `gate.running` and a `land.running` naming a live, unrelated pid, plus
    the flush marker and the lock payload naming it too, must read stale in all
    five readers: `gate_in_progress` and `land_in_progress` None,
    `rounds_landing()` [], `flush_in_progress` None, `landing_lock()` None.

    The pid is this pytest process's own, and its `/proc` cmdline is a python
    running pytest — the same shape as pid 4694's Thunderbird content process in
    the live witness, and no fake is involved: `pid_alive` is genuinely True for
    it, which is exactly the bug. Before #3926 all five reads said "running", and
    the cross-round one (`rounds_landing`, which scans every round directory and
    gates the worker pool and the promoter) is what froze the whole loop rather
    than one round."""
    me = os.getpid()
    assert S.pid_alive(me), "the planted pid must be genuinely alive to be the bug"
    assert not S.identity_is_automod(S.read_proc_cmdline(me) or ""), (
        "the planted pid must not be an automod process, or this plants nothing")
    _plant("gate", pid=me, started_at=time.time())
    _plant("land", pid=me, started_at=time.time())
    _plant("flush", pid=me, started_at=time.time())

    assert S.gate_in_progress(RID) is None
    assert S.land_in_progress(RID) is None
    assert S.rounds_landing() == []
    assert S.flush_in_progress() is None
    assert S.landing_lock(_lock_path(env, RID, pid=me)) is None


def test_the_stale_reads_stay_stale_across_a_second_pass(env):
    """Two passes, because a stale read leaves the files in place rather than
    repairing them: a reader that only looked right once is not fixed."""
    me = os.getpid()
    _plant("gate", pid=me, started_at=time.time())
    _plant("land", pid=me, started_at=time.time())
    assert _all_readers(env, RID, pid=me)[:2] == [None, None]
    assert _all_readers(env, RID, pid=me)[:2] == [None, None]
    assert _all_readers(env, RID, pid=me)[2] == []


# --- clause 2: positive control, so the guard still passes a real landing ---


def test_a_real_module_form_automod_child_keeps_its_markers_live(env):
    """`python -m scripts.automod.round <verb>` — the shape
    `round.gate_detached`, `round.land_detached` and `round.flush_detached`
    spawn, `[LIVE_ROOT/.venvs/lloyd/bin/python, "-m", "scripts.automod.round",
    <verb>, round_id]`. A live child of that shape must keep its gate, land,
    flush and lock markers live through all five readers, and `rounds_landing()`
    must name its round; otherwise the identity check would declare every real
    landing dead and the loop would gate on top of it."""
    child = _child(env, ["-m", "scripts.automod.waiter", "gate", RID],
                   module_root=env["tmp"])
    assert child.poll() is None
    _plant("gate", pid=child.pid, started_at=time.time())
    _plant("land", pid=child.pid, started_at=time.time())
    _plant("flush", pid=child.pid, started_at=time.time())

    gate = S.gate_in_progress(RID)
    land = S.land_in_progress(RID)
    assert gate is not None and gate["pid"] == child.pid
    assert land is not None and land["pid"] == child.pid
    assert S.rounds_landing() == [RID]
    assert S.flush_in_progress() is not None
    assert S.landing_lock(_lock_path(env, RID, pid=child.pid)) is not None


def test_a_real_file_form_automod_child_keeps_its_markers_live(env):
    """The other invocation form, measured on the live box: a hand-run
    `/usr/bin/python /home/alansrobotlab/lloyd/scripts/automod/round.py gate …`.
    A marker naming a child launched as `…/scripts/automod/waiter.py <verb>` —
    the same path shape — must read live too, so the guard accepts both spellings
    and not only the `-m` one the detached spawners use."""
    child = _child(env, [str(env["tmp"] / "scripts" / "automod" / "waiter.py"),
                         "land", RID])
    assert child.poll() is None
    _plant("land", pid=child.pid, started_at=time.time())
    _plant("gate", pid=child.pid, started_at=time.time())

    assert S.land_in_progress(RID) is not None
    assert S.gate_in_progress(RID) is not None
    assert S.rounds_landing() == [RID]


def test_identity_names_the_production_argv_shapes_and_refuses_the_neighbours():
    """The predicate over the argv lines the loop actually produces, and over the
    look-alikes: `python -m scripts.automod.round gate|land|flush` (the three
    detached children), `python -m scripts.automod …`, and
    `python …/scripts/automod/round.py gate` (the file form) are automod. The
    Thunderbird content process that inherited pid 4694, the vLLM server an
    earlier witness blamed on it, a pytest run, and a `grep scripts.automod` or
    `tail …/scripts/automod/gate.log` — which mention the path without being the
    loop — are not."""
    py = "/home/alansrobotlab/lloyd/.venvs/lloyd/bin/python"
    for verb in ("gate", "land", "flush"):
        assert S.identity_is_automod("\0".join([py, "-m", "scripts.automod.round", verb,
                                                RID, "\0"])), verb
    assert S.identity_is_automod("\0".join([py, "-m", "scripts.automod", "status", "\0"]))
    assert S.identity_is_automod("\0".join(["/usr/bin/python",
                                            "/home/alansrobotlab/lloyd/scripts/automod/round.py",
                                            "gate", RID, "\0"]))
    assert not S.identity_is_automod(
        "/usr/lib/thunderbird/thunderbird\0-contentproc\0-isForBrowser\0\0")
    assert not S.identity_is_automod(
        "\0".join(["/home/alansrobotlab/lloyd/.venvs/vllm/bin/python", "-m",
                   "vllm.entrypoints.openai.api_server", "--port", "8096", "\0"]))
    assert not S.identity_is_automod(
        "\0".join([py, "-m", "pytest", "tests/test_automod_marker_identity.py", "\0"]))
    assert not S.identity_is_automod(
        "\0".join(["grep", "-r", "scripts.automod", "gate.log", "\0"]))
    assert not S.identity_is_automod(
        "\0".join(["tail", "/home/alansrobotlab/.local/state/lloyd-automod/rounds/"
                   "SM_20261108_043338/gate.log", "\0"]))
    assert not S.identity_is_automod("")
    assert not S.identity_is_automod("\0\0\0")


# --- clause 3: unreadable cmdline falls back to starttime ------------------


def test_an_unreadable_cmdline_rejects_a_process_that_began_after_the_marker(env, monkeypatch):
    """When `/proc/<pid>/cmdline` cannot be read for an alive pid, liveness is
    the clock's: a process that began more than `MARKER_START_SLACK_S` (5.0 s)
    after the marker's stamp cannot be the one that wrote it, so it reads
    not-live — in every reader. The child is real and freshly spawned, and the
    marker is stamped 300 s in the past, so the child began 300 s after it: 60
    times the 5 s the fallback allows. Only the cmdline read is forced closed,
    because no test can make a live pid's `/proc/<pid>/cmdline` unreadable; the
    starttime it is judged on is the child's real one."""
    child = _child(env, ["-m", "scripts.automod.waiter", "gate", RID],
                   module_root=env["tmp"])
    assert S.pid_alive(child.pid)
    assert S._proc_start_epoch(child.pid) is not None, "the timing leg needs its input"
    monkeypatch.setattr(S, "read_proc_cmdline", lambda pid: None)
    stale = time.time() - 300.0
    _plant("gate", pid=child.pid, started_at=stale)
    _plant("land", pid=child.pid, started_at=stale)
    _plant("flush", pid=child.pid, started_at=stale)
    lock = _lock_path(env, RID, pid=child.pid,
                      since=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stale)))

    assert S.gate_in_progress(RID) is None
    assert S.land_in_progress(RID) is None
    assert S.rounds_landing() == []
    assert S.flush_in_progress() is None
    assert S.landing_lock(lock) is None


def test_an_unreadable_cmdline_accepts_a_process_that_began_before_the_marker(env, monkeypatch):
    """The same fallback the other way, so it is a check and not a refusal: a
    process that began before the marker was written reads live, and one that
    began inside the 5.0 s slack just after it reads live too — the marker is
    stamped before the child is forked, and `started_at` is wall clock rebased
    through `/proc/uptime`, so the two need that much room. Measured on two real
    children: the first, whose start precedes its stamp by 300 s, and the second,
    forked immediately after its stamp and asserted to have started after it,
    inside the slack."""
    child = _child(env, ["-m", "scripts.automod.waiter", "gate", RID],
                   module_root=env["tmp"])
    assert S.pid_alive(child.pid)
    monkeypatch.setattr(S, "read_proc_cmdline", lambda pid: None)

    _plant("gate", pid=child.pid, started_at=time.time() + 300.0)
    assert S.gate_in_progress(RID) is not None, "began before the stamp: must read live"

    stamp = time.time()
    fresh = _child(env, ["-m", "scripts.automod.waiter", "gate", RID],
                   module_root=env["tmp"])
    assert fresh.pid != child.pid
    started = S._proc_start_epoch(fresh.pid)
    assert started is not None and started > stamp, "this is the slack side, not the before side"
    assert started <= stamp + S.MARKER_START_SLACK_S, "…and still inside the 5 s allowed"
    _plant("gate", pid=fresh.pid, started_at=stamp)
    assert S.gate_in_progress(RID) is not None, "began within the slack: must read live"


def test_the_fallback_still_needs_the_pid_to_be_alive(env, monkeypatch):
    """`marker_pid_is_live` is liveness AND identity, not identity instead of
    liveness. A zombie whose cmdline says `scripts.automod.round gate` — the
    strongest possible identity claim — still reads not-live, because the zombie
    leg of `pid_alive`, which #3926 deliberately did not touch, runs first. A
    reordering that answered identity before liveness fails here, and a reaped
    pid fails too."""
    child = _child(env, ["-m", "scripts.automod.waiter", "gate", RID],
                   module_root=env["tmp"])
    zombie = child.pid
    _plant("gate", pid=zombie, started_at=time.time())
    monkeypatch.setattr(S, "read_proc_cmdline", lambda pid: _AUTOMOD_ARGV)
    assert S.identity_is_automod(S.read_proc_cmdline(zombie) or ""), (
        "the identity leg must be saying automod for this to pin the order")
    child.kill()
    for _ in range(150):
        try:
            if open(f"/proc/{zombie}/stat").read().rsplit(")", 1)[1].split()[0] == "Z":
                break
        except OSError:
            pass
        time.sleep(0.02)
    assert S.pid_alive(zombie) is False, "the zombie leg is the thing under test here"
    assert S.marker_pid_is_live(zombie, time.time()) is False
    assert S.gate_in_progress(RID) is None
    child.wait(timeout=10)
    assert S.marker_pid_is_live(child.pid, time.time()) is False


# --- clause 4: stale markers survive their readers -------------------------


def test_a_stale_marker_survives_every_reader_byte_for_byte(env):
    """None of the five readers unlinks or rewrites what it judged stale: the
    residue stays on disk as the audit trail, with the same bytes, the same JSON
    keys, and the same three file names (`gate.running`, `land.running`,
    `flush.running`). #3926 changes how a marker is READ, not where it lives or
    what it contains — an age bound on the residue is a separate ruling,
    deliberately not taken here."""
    me = os.getpid()
    planted = {kind: _plant(kind, pid=me, started_at=time.time()) for kind in
               ("gate", "land", "flush")}
    bytes_before = {kind: _marker_bytes(kind) for kind in planted}

    assert _all_readers(env, RID, pid=me) == [None, None, [], None, None]

    assert S.gate_marker_path(RID).name == "gate.running"
    assert S.land_marker_path(RID).name == "land.running"
    assert S.FLUSH_MARKER_PATH.name == "flush.running"
    for kind, rec in planted.items():
        after = _marker_bytes(kind)
        assert after == bytes_before[kind], f"{kind} marker changed under a reader"
        assert set(json.loads(after.decode())) == (
            GATE_KEYS if kind == "gate" else LAND_KEYS if kind == "land" else FLUSH_KEYS)
        assert json.loads(after.decode())["pid"] == me == rec["pid"]
    assert (env["tmp"] / "lock.json").exists()


def test_a_dead_pid_marker_reads_stale_and_is_left_on_disk(env):
    """The pre-existing dead-pid behaviour is unchanged and still
    non-destructive: a marker naming a pid that does not exist reads None/[] from
    the five readers and stays on disk. This is the item's own check 4 — "the
    residue markers stay on disk, merely unread as live" — planted for a dead
    pid, since the real residue cannot be planted by a test at all."""
    dead = _unused_pid()
    assert S.pid_alive(dead) is False
    _plant("gate", pid=dead, started_at=time.time())
    _plant("land", pid=dead, started_at=time.time())
    _plant("flush", pid=dead, started_at=time.time())

    assert S.gate_in_progress(RID) is None
    assert S.land_in_progress(RID) is None
    assert S.rounds_landing() == []
    assert S.flush_in_progress() is None
    assert S.gate_marker_path(RID).exists()
    assert S.land_marker_path(RID).exists()
    assert S.FLUSH_MARKER_PATH.exists()


def test_the_marker_writers_still_write_the_same_keys(env):
    """The write side is untouched: `write_gate_marker`, `write_land_marker` and
    `write_flush_marker` produce exactly the keys readers have always read, so a
    marker written by a live child stays byte-compatible with the old format. And
    a marker written that way by a live NON-automod pid — this process — is still
    read stale: the writer's format and the reader's verdict are independent
    facts."""
    me = os.getpid()
    assert set(S.write_gate_marker(RID, pid=me, head="a" * 40, by="run_gate")) == GATE_KEYS
    assert set(S.write_land_marker(RID, pid=me, by="round.land")) == LAND_KEYS
    assert set(S.write_flush_marker(pid=me, by="cli")) == FLUSH_KEYS
    gate = json.loads(S.gate_marker_path(RID).read_text(encoding="utf-8"))
    assert gate["pid"] == me and gate["by"] == "run_gate"
    assert gate["head"] == "a" * 40
    assert S.gate_in_progress(RID) is None
