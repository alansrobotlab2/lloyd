"""The Services tab's GPU power switch: pause the workers and stop every
supervisord program that holds a GPU, then bring exactly those back.

Two halves, one record. `power_off` pauses the pool, writes the marker, and stops
each GPU program that was up; `power_on` starts the ones the marker names, waits
for the primary's /health, removes the marker and lifts the pause — but only a
pause it took itself, so an operator's earlier hold survives the round trip.

**The marker is what keeps the rest of the box from undoing the switch.** It lives
in the guardian's state dir (`gpu_off.json`) because the guardian is the one thing
that would fight it: `agent-tts` is in its `RECOVERABLE_INFRA` and a STOPPED
synthesiser gets restarted within a tick, and every alert it speaks goes through
that synthesiser and would be journaled as a voice loss. With the marker present
the guardian reads the listed programs' STOPPED as intentional
(`guardian.evaluate_liveness`) and withholds speech (`speak.dispatch`). The
backend's boot slot reconcile (`server._sync_llm_slots`) reads it too, so a
backend restart while the GPUs are off does not boot djev back onto GPU 2.

`regression.lock` is held for the whole off period: the regression runner
measures a promotion on the GPU daemons, and with them stopped it would find a
regression that reproduces — and ask for a rollback of a commit that did nothing.
The lock is a flock on this process, so a backend restart drops it and
`reacquire_lock_at_boot` takes it back.

The primary's start waits for host RAM the way `round restart --only
agent-llm-primary` does (`scripts/automod/ram_gate.py`): it holds a ~95 GiB table
in host RAM, and a boot over a stop that has not finished releasing it is the
2026-09-08 oomd kill.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from app.supervisor_client import (
    _supervisor_all_lenient,
    process_info,
    start_process,
    stop_process,
)

log = logging.getLogger("lloyd-server")

#: Every supervisord program that puts a CUDA context on a card, in start order:
#: the primary first (its RAM gate is the slow step), voice last. Kept in sync with
#: `agent-services/supervisor/conf.d/` by `tests/test_gpu_power.py`, which fails on
#: a conf that pins a GPU and is missing here.
GPU_PROGRAMS = (
    "agent-llm-primary",
    "agent-llm-secondary",
    "agent-djev",
    "agent-decider",
    "agent-qmd-daemon",
    "agent-qmd-watcher",
    "agent-tts",
    "agent-tts-omni",
    "lloyd-agent-worker",
)

PRIMARY = "agent-llm-primary"
PRIMARY_HEALTH = "http://127.0.0.1:8096/health"
PRIMARY_HEALTH_BUDGET_S = 1200.0
STOP_BUDGET_S = 150.0          # the primary's stopwaitsecs is 90
MARKER_NAME = "gpu_off.json"   # also named in agent-services/guardian/speak.py
ACTOR = "mission-control:gpu-off"

_UP_STATES = {"RUNNING", "STARTING", "BACKOFF"}
_DOWN_STATES = {"STOPPED", "EXITED", "FATAL"}

# In-process progress; the marker is the durable half.
_busy: str | None = None       # "stopping" | "starting"
_detail: str = ""
_error: str | None = None
_lock = None                   # scripts.automod.state.Lock while off


def guardian_state_dir() -> Path:
    env = os.environ.get("LLOYD_GUARDIAN_STATE")
    return Path(env) if env else Path.home() / ".local" / "state" / "lloyd-guardian"


def marker_path() -> Path:
    return guardian_state_dir() / MARKER_NAME


def read_marker() -> dict | None:
    try:
        data = json.loads(marker_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def is_off() -> bool:
    return marker_path().exists()


def _write_marker(data: dict) -> None:
    path = marker_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def status() -> dict:
    marker = read_marker()
    procs = _supervisor_all_lenient()
    gpu = [{"id": p, "state": (procs.get(p) or {}).get("statename", "UNKNOWN")}
           for p in GPU_PROGRAMS if p in procs]
    if _busy:
        phase = _busy
    else:
        phase = "off" if marker else "on"
    return {
        "phase": phase,
        "detail": _detail,
        "error": _error,
        "since": (marker or {}).get("since"),
        "stopped": (marker or {}).get("stopped", []),
        "programs": gpu,
    }


# ── the lock that keeps the regression runner off dead daemons ──────────────

def _take_regression_lock() -> str | None:
    """Take `regression.lock`; the refusal reason if a runner holds it."""
    global _lock
    if _lock is not None:
        return None
    from scripts.automod import state as S
    try:
        _lock = S.Lock(S.STATE_DIR / "regression.lock", owner=ACTOR).acquire()
        return None
    except S.LockHeld as exc:
        return str(exc)


def _drop_regression_lock() -> None:
    global _lock
    if _lock is not None:
        try:
            _lock.release()
        except Exception:
            pass
        _lock = None


def reacquire_lock_at_boot() -> None:
    """A restarted backend lost the flock; take it back while the GPUs are off."""
    if is_off():
        reason = _take_regression_lock()
        if reason:
            log.warning("gpu-off: marker present but regression.lock is held: %s", reason)


# ── host RAM, the primary's boot gate ───────────────────────────────────────

def _mem_available_gib() -> int:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // (1024 * 1024)
    except OSError:
        pass
    return 0


def _health_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


async def _wait_stopped(programs: list[str], budget: float) -> list[str]:
    """Poll until every program is down; returns the ones that never got there."""
    deadline = time.monotonic() + budget
    left = list(programs)
    while left and time.monotonic() < deadline:
        await asyncio.sleep(2.0)
        still = []
        for p in left:
            try:
                state = (await asyncio.to_thread(process_info, p)).get("statename", "").upper()
            except Exception:
                state = ""
            if state not in _DOWN_STATES:
                still.append(p)
        left = still
    return left


# ── the two transitions ─────────────────────────────────────────────────────

def begin_off() -> str | None:
    """Validate and claim the transition; the refusal reason, or None to proceed."""
    global _busy, _error, _detail
    if _busy:
        return f"already {_busy}"
    if is_off():
        return "GPU services are already off"
    reason = _take_regression_lock()
    if reason:
        return f"a regression check is running ({reason}); try again when it finishes"
    _busy, _error, _detail = "stopping", None, "pausing workers"
    return None


async def power_off() -> None:
    global _busy, _detail, _error
    try:
        from workers.pool import HOLDER_PERSON, get_pool
        pool = get_pool()
        was_paused = bool(pool and "operator" in (pool.paused_by or []))
        if pool and not was_paused:
            pool.pause(True, owner="operator", holder_kind=HOLDER_PERSON, actor=ACTOR)

        procs = await asyncio.to_thread(_supervisor_all_lenient)
        up = [p for p in GPU_PROGRAMS
              if str((procs.get(p) or {}).get("statename", "")).upper() in _UP_STATES]
        # The marker goes down BEFORE the first stop, so the guardian never sees
        # an unexplained STOPPED agent-tts.
        _write_marker({"since": datetime.now(timezone.utc).isoformat(),
                       "by": ACTOR, "stopped": up, "programs": list(GPU_PROGRAMS),
                       "paused_workers": bool(pool and not was_paused)})
        _detail = f"stopping {len(up)} program(s)"
        failed = []
        for p in up:
            ok, msg = await asyncio.to_thread(stop_process, p, False)
            if not ok:
                failed.append(f"{p}: {msg}")
        stuck = await _wait_stopped(up, STOP_BUDGET_S)
        if failed or stuck:
            _error = "; ".join(failed + [f"{p} did not stop" for p in stuck])
        log.info("gpu-off: stopped %s (paused workers: %s)", up, not was_paused)
    except Exception as exc:
        _error = f"{type(exc).__name__}: {exc}"
        log.exception("gpu-off failed")
    finally:
        _busy, _detail = None, ""


def begin_on() -> str | None:
    global _busy, _error, _detail
    if _busy:
        return f"already {_busy}"
    if not is_off():
        return "GPU services are not off"
    _busy, _error, _detail = "starting", None, ""
    return None


async def power_on() -> None:
    global _busy, _detail, _error
    try:
        marker = read_marker() or {}
        wanted = [p for p in GPU_PROGRAMS if p in (marker.get("stopped") or [])]
        if PRIMARY in wanted:
            from scripts.automod import ram_gate
            floor, abort = ram_gate.PRIMARY_RAM_FLOOR_GIB, ram_gate.PRIMARY_RAM_ABORT_GIB
            deadline = time.monotonic() + ram_gate.PRIMARY_RAM_WAIT_SECONDS
            avail = _mem_available_gib()
            while avail < floor and time.monotonic() < deadline:
                _detail = f"waiting for host RAM ({avail}/{floor} GiB free)"
                await asyncio.sleep(5.0)
                avail = _mem_available_gib()
            if avail < abort:
                _error = (f"only {avail} GiB host RAM free; the primary needs {floor} "
                          f"(abort line {abort}) — nothing started, still off")
                return
        failed = []
        for p in wanted:
            _detail = f"starting {p}"
            ok, msg = await asyncio.to_thread(start_process, p, False)
            if not ok:
                failed.append(f"{p}: {msg}")
        if PRIMARY in wanted:
            deadline = time.monotonic() + PRIMARY_HEALTH_BUDGET_S
            t0 = time.monotonic()
            while not await asyncio.to_thread(_health_ok, PRIMARY_HEALTH):
                if time.monotonic() > deadline:
                    failed.append(f"{PRIMARY} /health not answering after "
                                  f"{PRIMARY_HEALTH_BUDGET_S:.0f}s")
                    break
                _detail = f"primary booting ({time.monotonic() - t0:.0f}s)"
                await asyncio.sleep(5.0)
        try:
            marker_path().unlink()
        except FileNotFoundError:
            pass
        _drop_regression_lock()
        if marker.get("paused_workers") and not failed:
            from workers.pool import get_pool
            pool = get_pool()
            if pool:
                pool.pause(False, owner="operator", actor=ACTOR)
        if failed:
            _error = "; ".join(failed) + " — workers left paused"
        log.info("gpu-on: started %s", wanted)
    except Exception as exc:
        _error = f"{type(exc).__name__}: {exc}"
        log.exception("gpu-on failed")
    finally:
        _busy, _detail = None, ""
