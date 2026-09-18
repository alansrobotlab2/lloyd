"""A test run never addresses the machine's automod state — all the way through.

`gate._child_env` points `LLOYD_AUTOMOD_STATE` at a scratch dir so candidate
tests cannot reach the production ledger. From 2026-09-09 to 2026-09-18 that
held for one test file: `test_automod_hardening.isolated_state` tore down with
`delenv` + `importlib.reload(S)`, which re-read an environment it had just
emptied, and every test after it in the session resolved `S.STATE_DIR` to
`~/.local/state/lloyd-automod`. What that cost, the day it was found:

  * `land_failed` rows for the fixture round `SM_L` in the live ledger, written
    by a SIGTERM handler `round.land` installed and never gave back;
  * `test_a_landing_owns_its_marker…` spinning in `round._land_lock` for as
    long as PRODUCTION had a promotion under observation — 822 s in one run,
    and three of that day's gate runs stretched by three to seven minutes.

Each test below fails on the tree that had the defect.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from scripts.automod import round as R
from scripts.automod import state as S

ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_STATE = Path.home() / ".local" / "state" / "lloyd-automod"


def test_this_run_does_not_address_the_production_state_dir():
    """Wherever this file sorts in the session, `S` names a scratch dir.

    It sorts after `test_automod_hardening.py`, so on the tree with the leaking
    fixture a full run fails here; and `conftest` gives a plain `pytest` — the
    way a round's model runs it, no gate environment — a scratch dir too.
    """
    assert os.environ.get("LLOYD_AUTOMOD_STATE"), "conftest sets a default before any import"
    for path in (S.STATE_DIR, S.LEDGER_PATH, S.CURRENT_PATH, S.ROUNDS_DIR):
        assert PRODUCTION_STATE not in (path, *path.parents), (
            f"{path} is production's — a test that forgets one monkeypatch now writes the "
            f"real ledger or reads the real current.json")


def test_a_test_that_used_isolated_state_leaves_the_session_where_the_caller_put_it(tmp_path):
    """The probe that found it, as a child pytest: one `isolated_state` test,
    then a test that reports where `S` points. Same environment the gate uses."""
    probe = tmp_path / "test_zz_probe.py"
    probe.write_text(textwrap.dedent("""
        def test_where_does_state_point():
            from scripts.automod import state as S
            print("PROBE", S.STATE_DIR)
    """))
    caller_state = tmp_path / "caller-state"
    caller_state.mkdir()
    env = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "PYTHONPATH": str(ROOT),
           "LLOYD_AUTOMOD_STATE": str(caller_state),
           "LLOYD_GUARDIAN_STATE": str(tmp_path / "guardian"), "LLOYD_VOICE_ALERTS": "0"}
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider",
         "tests/test_automod_hardening.py::test_the_master_switch_covers_the_cli_not_only_the_tool",
         str(probe)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "2 passed" in out, out[-1500:]
    pointed = [line.split("PROBE", 1)[1].strip() for line in out.splitlines() if "PROBE" in line]
    assert pointed == [str(caller_state)], (
        f"after one `isolated_state` test the session's state dir is {pointed}, not the "
        f"{caller_state} its caller set — every later test is reading somebody else's state")


def test_a_landing_gives_back_the_signal_handlers_it_took(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
    restore = R._die_loudly_on_signal("SM_X")
    try:
        assert all(signal.getsignal(sig) is not before[sig] for sig in before), "installed"
    finally:
        restore()
    assert {sig: signal.getsignal(sig) for sig in before} == before
    # And while it IS installed it still does its job: the ledger hears about
    # the kill, and the exception is what lets `land`'s `finally` run.
    restore = R._die_loudly_on_signal("SM_X")
    try:
        with pytest.raises(R.LandingKilled):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
    finally:
        restore()
    rows = [e for e in S.read_events(path=S.LEDGER_PATH) if e.get("event") == "land_failed"]
    assert len(rows) == 1 and rows[0]["round_id"] == "SM_X" and rows[0]["killed_by_signal"] == 15


def test_the_chamber_recheck_is_paced_not_spun(monkeypatch, tmp_path):
    """`_land_lock` loops while `current.json` reads `observing` after it has
    the lock. Its brake used to be `wait_for_settle` blocking at the top of the
    loop and nothing else, so with that returning at once — a stub, or one day a
    timeout — it re-parsed config.yaml as fast as one core could."""
    monkeypatch.setattr(S, "LOCK_PATH", tmp_path / "lock")
    monkeypatch.setattr(S, "chamber_enabled", lambda repo=None: True)
    monkeypatch.setattr(R, "LAND_LOCK_POLL", 0.05)
    monkeypatch.setattr(R, "LAND_LOCK_MAX_WAIT", 0.5)
    reads = {"n": 0}

    def observing():
        reads["n"] += 1
        return {"state": "observing", "commit": "abc"}
    monkeypatch.setattr(S, "read_current", observing)
    monkeypatch.setattr(R.P, "wait_for_settle", lambda **k: None)
    started = time.time()
    lock = R._land_lock("SM_P")     # past the deadline it lands anyway; `promote` still refuses
    lock.release()
    assert time.time() - started >= 0.4, "it waited its deadline out"
    # Two reads a lap, a lap every LAND_LOCK_POLL: about twenty. Unpaced, the
    # same half second is tens of thousands.
    assert reads["n"] < 60, f"{reads['n']} reads of current.json in half a second is a spin"
