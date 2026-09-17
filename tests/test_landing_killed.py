"""A landing killed from outside is not a verdict on the change (#1179, 2026-09-17).

The turn ran `timeout 120 .venvs/lloyd/bin/python -m scripts.automod.round land
SM_…` in the foreground of its own Bash. The landing waits for the backend to
go idle, and that turn was what kept it busy; `timeout` sent SIGTERM at 120 s,
Python died with no `finally`, the land marker named a dead pid, the reaper
closed the round a second after the turn ended, and with no promotion and no
`land_failed` on the ledger the item read as `spent` — nine green rungs, a
kept branch, and a trip back through triage.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from app.harness.service_control import check_service_control, find_service_control
from scripts.automod import backlog as B, state as S

ROOT = Path(__file__).resolve().parent.parent
WORKER = "20260917_141042_autocode_cc34"
CHAT = "20260917_141042_ab12cd"
INCIDENT = ("cd ~/lloyd && timeout 120 .venvs/lloyd/bin/python -m scripts.automod.round "
            "land SM_20260917_211249 2>&1 | tail -20")


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "LEDGER_PATH", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(S, "ROUNDS_DIR", tmp_path / "rounds")


# ── the guard ──────────────────────────────────────────────────────────────

def test_the_incident_command_is_refused_for_a_worker_and_names_the_tool():
    why = check_service_control(INCIDENT, WORKER)
    assert why and "automod_land" in why and "END YOUR TURN" in why


@pytest.mark.parametrize("command", [
    ".venvs/lloyd/bin/python -m scripts.automod.round land SM_1",
    "bash -c 'python -m scripts.automod.round land SM_1'",
    "python scripts/automod/round.py land SM_1",
])
def test_every_spelling_of_a_foreground_landing_is_caught(command):
    assert find_service_control(command) == "round land"


@pytest.mark.parametrize("command", [
    "python -m scripts.automod.round land SM_1 --dry-run",
    "python -m scripts.automod.round status",
    "python -m scripts.automod.round gate SM_1",
    "grep -n 'round land' CLAUDE.md",
    "tail -5 ~/.local/state/lloyd-automod/rounds/SM_1/land.log",
])
def test_reading_about_a_landing_and_dry_runs_stay_allowed(command):
    assert check_service_control(command, WORKER) is None


def test_a_person_may_still_land_from_the_command_line():
    assert check_service_control(INCIDENT, CHAT) is None


# ── the ledger ─────────────────────────────────────────────────────────────

def _round(item, rid, *, last_rung="drill", ok=True, landed=True):
    S.append_event({"event": "backlog_implement", "item_id": item, "phase": "started"}, path=S.LEDGER_PATH)
    S.append_event({"event": "gate", "round_id": rid, "rung": last_rung, "ok": ok}, path=S.LEDGER_PATH)
    S.append_event({"event": "backlog_implement", "item_id": item, "phase": "finished", "round_id": rid,
                    "stop_reason": "stop", "num_turns": 100,
                    "outcome": {"landed": landed, "acceptance": "met"}}, path=S.LEDGER_PATH)


def test_a_gate_passed_round_that_never_landed_is_external_and_says_how_to_finish():
    _round(1179, "SM_A")
    verdict, why = B.implement_outcomes(S.LEDGER_PATH)[1179]
    assert verdict == "external"
    assert "automod/SM_A" in why and "automod_land" in why and "never `round land` from Bash" in why


def test_a_promotion_or_a_recorded_landing_failure_is_judged_as_before():
    _round(1, "SM_P")
    S.append_event({"event": "promoted", "round_id": "SM_P", "commit": "a" * 40}, path=S.LEDGER_PATH)
    assert B.implement_outcomes(S.LEDGER_PATH)[1][0] == "spent", "landed: a verdict"
    _round(2, "SM_F")
    S.append_event({"event": "land_failed", "round_id": "SM_F", "ok": False,
                    "external_blocker": False, "detail": "its own diff"}, path=S.LEDGER_PATH)
    assert B.implement_outcomes(S.LEDGER_PATH)[2][0] == "spent", "a landing the round itself broke"
    assert "SM_F" not in B.gate_passed_unlanded_rounds(S.LEDGER_PATH)


def test_a_round_whose_gate_never_finished_green_is_not_this_case():
    _round(3, "SM_R", last_rung="review", ok=False)
    assert "SM_R" not in B.gate_passed_unlanded_rounds(S.LEDGER_PATH)
    _round(4, "SM_T", last_rung="tests", ok=True)
    assert "SM_T" not in B.gate_passed_unlanded_rounds(S.LEDGER_PATH), "a pass so far is not a pass"


def test_it_is_capped_on_its_own_count():
    for n in range(B.EXTERNAL_RETRY_CAP + 1):
        _round(5, f"SM_U{n}")
    assert B.implement_outcomes(S.LEDGER_PATH)[5][0] == "spent"


# ── the process ────────────────────────────────────────────────────────────

def test_sigterm_during_the_wait_writes_a_land_failed_and_clears_the_marker(tmp_path):
    """A real child, a real signal: the default handler would skip `finally`."""
    state = tmp_path / "state"
    script = textwrap.dedent(f"""
        import os, sys, time
        sys.path.insert(0, {str(ROOT)!r})
        os.environ["LLOYD_AUTOMOD_STATE"] = {str(state)!r}
        from scripts.automod import round as R, state as S, promote as P
        S.require_enabled = lambda *a, **k: None
        (S.ROUNDS_DIR / "SM_K").mkdir(parents=True, exist_ok=True)
        (S.ROUNDS_DIR / "SM_K" / "gate.json").write_text('{{"ok": true, "base": "x", "rungs": []}}')
        def wait(*a, **k):
            print("WAITING", flush=True)
            time.sleep(60)
        P.wait_for_rounds = wait
        try:
            R.land("SM_K")
        except R.LandingKilled as exc:
            print("KILLED", S.land_in_progress("SM_K"), flush=True)
            sys.exit(3)
    """)
    p = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True,
                         env={**os.environ, "LLOYD_AUTOMOD_STATE": str(state)})
    assert p.stdout.readline().strip() == "WAITING"
    p.send_signal(signal.SIGTERM)
    out = p.stdout.read()
    assert p.wait(timeout=20) == 3, out
    assert "KILLED None" in out, "the marker is gone: finally ran"
    rows = [ln for ln in (state / "promotions.jsonl").read_text().splitlines() if "land_failed" in ln]
    assert rows and '"external_blocker": true' in rows[-1] and '"killed_by_signal": 15' in rows[-1]
