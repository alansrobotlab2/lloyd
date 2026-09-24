"""The budget and todo anchors each have an off switch (#769, clauses 1-2).

They are the injected text that fires most often on a long turn, and until
#769 neither could be turned off to measure what it is worth: the context
anchor had `harness.context_anchor.enabled`, the budget anchor had no switch
at all (only `max_turns <= 0`, which removes the cap itself), and the todo
anchor only the de-facto `todo_anchor_interval_iterations: 0`. The flags are
read at the emission decision, and a key that is absent keeps today's
behaviour — config.yaml carries neither key yet.
"""
from __future__ import annotations

import asyncio

import pytest

from app import deadline_anchor
from app.config import CONFIG
from app.routers import messages as M


TODOS = [
    {"content": "Read the code", "status": "in_progress"},
    {"content": "Write the review", "status": "pending"},
]


def _texts(out) -> str:
    return "\n".join(str(m.get("content") or "") for m in out)


@pytest.fixture
def harness_cfg(monkeypatch):
    """A private copy of `harness` with neither flag present."""
    cfg = {k: v for k, v in (CONFIG.get("harness") or {}).items()
           if k not in ("budget_anchor", "todo_anchor")}
    monkeypatch.setitem(CONFIG, "harness", cfg)
    return cfg


@pytest.fixture
def clock(monkeypatch):
    state = {"t": 1000.0}
    monkeypatch.setattr(deadline_anchor.time, "monotonic", lambda: state["t"])
    return state


def _budget_run(clock, *, max_turns=20, timeout=1000.0):
    """Walk a chat-path anchor past 75%/90% of its cap and 70%/90% of its clock."""
    anchor = M._build_state_anchor("flag-test", max_turns=max_turns,
                                   deadline_seconds=timeout)
    out = []
    for i in range(1, max_turns + 1):
        clock["t"] += timeout / max_turns   # 5% of the wall clock per iteration
        out.extend(asyncio.run(anchor(i)))
    return out


def test_budget_anchors_fire_when_the_key_is_absent(harness_cfg, clock, monkeypatch):
    monkeypatch.setattr(M, "_load_session_todos", lambda sid: [])
    text = _texts(_budget_run(clock))
    assert "Iteration 15 of 20" in text and "Iteration 18 of 20" in text
    assert text.count("s of this turn's 1000s budget remain") == 2


def test_budget_flag_false_silences_both_budget_clocks(harness_cfg, clock, monkeypatch):
    monkeypatch.setattr(M, "_load_session_todos", lambda sid: [])
    harness_cfg["budget_anchor"] = {"enabled": False}
    out = _budget_run(clock)
    assert "<budget>" not in _texts(out), out


def test_budget_flag_false_silences_the_autonomy_task_anchor(harness_cfg, clock):
    import autonomy

    harness_cfg["budget_anchor"] = {"enabled": False}
    anchor = autonomy._build_task_anchor(1000, 20)
    out = []
    for i in range(1, 21):
        clock["t"] += 50
        out.extend(asyncio.run(anchor(i)))
    assert out == []


def test_budget_flag_true_is_the_same_as_absent(harness_cfg, clock, monkeypatch):
    monkeypatch.setattr(M, "_load_session_todos", lambda sid: [])
    harness_cfg["budget_anchor"] = {"enabled": True}
    assert _texts(_budget_run(clock)).count("<budget>") == 4


def _todo_run(harness_cfg, monkeypatch):
    monkeypatch.setattr(M, "_load_session_todos", lambda sid: list(TODOS))
    harness_cfg["todo_anchor_interval_iterations"] = 5
    anchor = M._build_state_anchor("flag-test")
    return [asyncio.run(anchor(i)) for i in range(1, 13)]


def test_todo_anchor_fires_when_the_key_is_absent(harness_cfg, monkeypatch):
    runs = _todo_run(harness_cfg, monkeypatch)
    assert any("<active_todos>" in _texts(out) for out in runs)


def test_todo_flag_false_silences_the_reminder(harness_cfg, monkeypatch):
    harness_cfg["todo_anchor"] = {"enabled": False}
    runs = _todo_run(harness_cfg, monkeypatch)
    assert not any("<active_todos>" in _texts(out) for out in runs)
