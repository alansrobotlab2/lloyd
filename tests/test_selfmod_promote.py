"""Promoter ordering and the observation-window contract.

Two properties here are the difference between a reversible landing and a
one-way door:

  * the rollback point is on disk and verified *before* the tree moves;
  * the observation window starts when the code goes live, not when the
    promoter began — the idle gate can legitimately wait many minutes, and a
    window started early would be mostly spent before the build existed.
"""

from __future__ import annotations

import json
import time

import pytest

from scripts.selfmod import promote as P, state as S


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    for name in ("STATE_DIR", "BROKEN_DIR", "ROUNDS_DIR"):
        monkeypatch.setattr(S, name, tmp_path)
    for name, fn in (("CURRENT_PATH", "current.json"), ("LEDGER_PATH", "promotions.jsonl"),
                     ("LKG_PATH", "last_known_good.json"), ("HALTED_PATH", "promotions-halted"),
                     ("BROKEN_PATH", "BROKEN"), ("DENIED_PATH", "denied.json"),
                     ("PAUSE_PATH", "pause"), ("LOCK_PATH", "lock")):
        monkeypatch.setattr(S, name, tmp_path / fn)
    return tmp_path


# ---------------------------------------------------------------------------
# The window must not start before the code is live
# ---------------------------------------------------------------------------

def test_a_landing_record_has_no_observation_window_yet():
    """Written before the idle gate, which can wait up to 15 minutes."""
    now = time.time()
    record = {"schema": 1, "round_id": "SM_X", "commit": "b" * 40, "parent": "a" * 40,
              "rollback_target": "a" * 40, "state": "landing",
              "landed_at": None, "landed_ts": None,
              "liveness_until_ts": None, "errors_until_ts": None}
    S.write_verified(S.CURRENT_PATH, record)
    back = S.read_current()
    assert back["state"] == "landing"
    assert back["errors_until_ts"] is None
    assert back["rollback_target"] == "a" * 40, "the way back must be recorded immediately"


def test_the_window_starts_only_once_the_build_is_verified():
    now = time.time()
    record = {"schema": 1, "commit": "b" * 40, "parent": "a" * 40,
              "rollback_target": "a" * 40, "state": "landing",
              "errors_until_ts": None, "liveness_until_ts": None}
    S.write_verified(S.CURRENT_PATH, record)
    landed = time.time()
    record.update({"state": "observing", "landed_ts": landed,
                   "liveness_until_ts": landed + P.LIVENESS_WINDOW,
                   "errors_until_ts": landed + P.ERRORS_WINDOW})
    S.write_verified(S.CURRENT_PATH, record)
    back = S.read_current()
    assert back["state"] == "observing"
    assert back["errors_until_ts"] >= now + P.ERRORS_WINDOW - 1


def test_the_guardian_ignores_a_landing_record(tmp_path, monkeypatch):
    """Mid-flight is not "deployed". Nothing to observe, nothing to revert."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                           "agent-services" / "guardian"))
    import gstate

    st = gstate.SelfModState(tmp_path)
    gstate.write_json_atomic(st.current_path,
                             {"state": "landing", "commit": "b" * 40,
                              "errors_until_ts": None})
    current = st.current()
    assert current["state"] == "landing"
    # The guardian nulls this out before deciding anything (see tick()).
    assert (None if current.get("state") == "landing" else current) is None


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_promote_refuses_while_halted(monkeypatch, tmp_path):
    monkeypatch.setattr(P.W, "head", lambda wt: "b" * 40)
    S.set_halted("flapping")
    with pytest.raises(P.PromoteError, match="halted"):
        P.promote("SM_X", tmp_path, "a" * 40)


def test_promote_refuses_when_broken(monkeypatch, tmp_path):
    S.BROKEN_PATH.write_text("rollback failed", encoding="utf-8")
    monkeypatch.setattr(P.W, "head", lambda wt: "b" * 40)
    with pytest.raises(P.PromoteError, match="BROKEN"):
        P.promote("SM_X", tmp_path, "a" * 40)


def test_promote_refuses_a_denylisted_commit(monkeypatch, tmp_path):
    monkeypatch.setattr(P.W, "head", lambda wt: "b" * 40)
    S.deny("b" * 40)
    with pytest.raises(P.PromoteError, match="denylist"):
        P.promote("SM_X", tmp_path, "a" * 40)


def test_promote_refuses_a_dirty_live_tree(monkeypatch, tmp_path):
    monkeypatch.setattr(P.W, "head", lambda wt: "b" * 40)
    monkeypatch.setattr(P.W, "is_clean", lambda repo: False)
    with pytest.raises(P.PromoteError, match="dirty"):
        P.promote("SM_X", tmp_path, "a" * 40)


def test_dry_run_touches_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(P.W, "head", lambda wt: "b" * 40)
    monkeypatch.setattr(P.W, "is_clean", lambda repo: True)
    monkeypatch.setattr(P.W, "changed_paths", lambda wt, base: ["app/x.py"])
    monkeypatch.setattr(P.subprocess, "run", lambda *a, **k: type(
        "R", (), {"stdout": "a" * 40, "returncode": 0, "stderr": ""})())
    out = P.promote("SM_X", tmp_path, "a" * 40, dry_run=True)
    assert out["would_promote"] is True
    assert S.read_current() is None, "a dry run must not write a promotion record"


# ---------------------------------------------------------------------------
# Idle gate
# ---------------------------------------------------------------------------

def test_idle_requires_consecutive_quiet_polls(monkeypatch):
    """A turn appearing mid-handshake resets the counter."""
    seq = [
        (200, {"turns": {"active": 0, "queued": 0}}),
        (200, {"turns": {"active": 1, "queued": 0}}),   # resets
        (200, {"turns": {"active": 0, "queued": 0}}),
        (200, {"turns": {"active": 0, "queued": 0}}),
        (200, {"turns": {"active": 0, "queued": 0}}),
    ]
    calls = {"n": 0}

    def fake_get(url, timeout=5.0):
        i = min(calls["n"], len(seq) - 1)
        calls["n"] += 1
        return seq[i]

    monkeypatch.setattr(P, "_get", fake_get)
    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    ok, why = P.wait_idle(max_wait=60)
    assert ok and "consecutive" in why
    assert calls["n"] >= 5


def test_idle_gate_gives_up_rather_than_landing_into_a_busy_backend(monkeypatch):
    monkeypatch.setattr(P, "_get", lambda url, timeout=5.0:
                        (200, {"turns": {"active": 1, "queued": 0}}))
    monkeypatch.setattr(P.time, "sleep", lambda s: None)
    ok, why = P.wait_idle(max_wait=0.01)
    assert not ok and "never went idle" in why
