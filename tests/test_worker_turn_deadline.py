"""A worker turn's wall clock, announced to the model.

`agent.max_turns` has been warned about since `_build_state_anchor` landed, but
iterations are not the budget unattended work dies on. `run_prompt_in_session`
bounds a worker turn by wall time, and automod round SM_20260909_054722
committed 757 lines into its worktree at 06:43:36 and was killed at 06:43:50 —
fourteen seconds, one `automod_gate` call, short of the verdict that would have
landed them, with 32 of its 100 iterations still unspent. The iteration anchor
cannot see that clock and never fired.

`autonomy.run_task` got this on 2026-09-08 for the same failure (#80/#78/#24).
This is that mechanism reaching the other caller, through the one definition in
`app.deadline_anchor` — two copies of "how close is the deadline" drift, and the
one that drifts is the one nobody is watching.
"""
from __future__ import annotations

import asyncio

import pytest

from app import deadline_anchor
from app.deadline_anchor import build_deadline_anchor
from app.routers.messages import _build_state_anchor, _turn_deadline


@pytest.fixture()
def clock(monkeypatch):
    state = {"t": 1000.0}
    monkeypatch.setattr(deadline_anchor.time, "monotonic", lambda: state["t"])
    return state


def _fire(anchor, iteration=1):
    return asyncio.run(anchor(iteration))


def test_no_deadline_means_no_anchor():
    """A chat turn has no wall clock and must never be told it has one."""
    assert build_deadline_anchor(0) is None
    assert build_deadline_anchor(-1) is None
    assert build_deadline_anchor(None) is None


def test_silent_while_the_budget_is_healthy(clock):
    a = build_deadline_anchor(1000, what="turn")
    clock["t"] += 600          # 60%
    assert _fire(a) == []


def test_warns_once_at_seventy_percent(clock):
    a = build_deadline_anchor(1000, what="turn")
    clock["t"] += 700
    out = _fire(a)
    assert len(out) == 1 and out[0]["role"] == "user"
    assert "do not open new" in out[0]["content"]
    clock["t"] += 10
    assert _fire(a) == [], "a warning re-sent every iteration is one the model skips"


def test_the_ninety_percent_warning_says_stop_calling_tools(clock):
    a = build_deadline_anchor(1000, what="turn")
    clock["t"] += 910
    out = _fire(a)
    # Both levels: one check crossed both thresholds, which is the behaviour
    # `test_a_late_first_call_still_fires_both_levels` pins. The 90% one is last.
    assert len(out) == 2
    assert "Stop calling tools" in out[-1]["content"]
    assert "90s of this turn's 1000s budget remain" in out[-1]["content"]


def test_remaining_seconds_are_reported_not_elapsed(clock):
    a = build_deadline_anchor(1000, what="turn")
    clock["t"] += 750
    assert "250s of this turn's" in _fire(a)[0]["content"]


def test_a_late_first_call_still_fires_both_levels(clock):
    """The harness calls this per iteration; one long tool call can carry the
    turn past both thresholds before the next check."""
    a = build_deadline_anchor(1000, what="turn")
    clock["t"] += 950
    out = _fire(a)
    assert len(out) == 2, "neither level is skipped by a late first check"


def test_the_state_anchor_carries_the_deadline_alongside_the_iteration_budget(clock, monkeypatch):
    """Both clocks in one closure: #446 had iterations to spare and no time."""
    monkeypatch.setattr("app.routers.messages._load_session_todos", lambda sid: [])
    anchor = _build_state_anchor("s1", max_turns=100, deadline_seconds=1000)
    clock["t"] += 700
    out = asyncio.run(anchor(10))          # 10% of iterations, 70% of the clock
    assert len(out) == 1
    assert "1000s budget" in out[0]["content"], "the iteration anchor cannot see this"


def test_a_turn_with_no_deadline_still_gets_the_iteration_budget(clock, monkeypatch):
    monkeypatch.setattr("app.routers.messages._load_session_todos", lambda sid: [])
    anchor = _build_state_anchor("s1", max_turns=100, deadline_seconds=0)
    clock["t"] += 100_000
    out = asyncio.run(anchor(90))
    # Iteration 90 of 100 crosses both the 75% and 90% iteration thresholds.
    assert len(out) == 2
    assert all("Iteration 90 of 100" in m["content"] for m in out)
    assert not any("budget remain. Finish this run" in m["content"] for m in out), \
        "no wall clock was given; the turn must not be told it has one"


@pytest.mark.parametrize("payload,expect", [
    ({}, 0.0),
    ({"deadline_seconds": 900}, 900.0),
    ({"deadline_seconds": "900"}, 900.0),
    ({"deadline_seconds": None}, 0.0),
    ({"deadline_seconds": "nonsense"}, 0.0),
    ({"deadline_seconds": -5}, 0.0),
    ({"deadline_seconds": 10 ** 9}, 86_400.0),
])
def test_the_payload_deadline_is_clamped(payload, expect):
    """Only a caller that enforces a clock sends one, and a bad value must not
    produce a nonsense sentence."""
    assert _turn_deadline(payload) == expect


def test_the_worker_sends_the_same_clock_it_enforces():
    """The number in the anchor and the number in the `wait_for` have to be the
    same, or the warning fires at the wrong time."""
    import inspect
    from workers.sources import _common
    src = inspect.getsource(_common.run_prompt_in_session)
    assert '"deadline_seconds": float(timeout_seconds)' in src
    assert "asyncio.wait_for(_stream(), timeout=float(timeout_seconds))" in src
