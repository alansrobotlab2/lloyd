"""The primary must be able to see its own todo list mid-turn.

Two mechanisms, tested here:

  * `TodoWrite`'s result echoes the stored list back, so the list is in
    context at the moment it changes (agent_mcp/builtin_todo.py).
  * `_build_state_anchor` re-appends it every N iterations while items
    are still open (app/routers/messages.py).

Before both, the list appeared exactly once — in the call that created
it. On 20260905_151355_iv5174 that was iteration 6 of 52; the review was
delivered with all five items still pending, and the user's task panel
showed nothing accomplished.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_mcp import builtin_todo


TODOS = [
    {"content": "Read the code", "status": "in_progress", "activeForm": "Reading"},
    {"content": "Write the review", "status": "pending", "activeForm": "Writing"},
]


# --------------------------------------------------------------- echo


def _write(monkeypatch, todos, session_id="s1"):
    stored: dict = {}

    async def _mutate(sid, apply_fn):
        data: dict = {}
        apply_fn(data)
        stored.update(data)
        return True

    monkeypatch.setattr(builtin_todo, "mutate_session", _mutate)
    monkeypatch.setattr(builtin_todo, "get_bound_session", lambda: session_id)
    out = asyncio.run(builtin_todo._todo_write({"todos": todos}))
    return out, stored


def test_todowrite_result_echoes_the_list(monkeypatch):
    out, stored = _write(monkeypatch, TODOS)
    assert "Read the code" in out
    assert "Write the review" in out
    assert "[in_progress]" in out
    assert "[pending]" in out
    assert stored["todos"] == TODOS


def test_todowrite_result_reports_open_count(monkeypatch):
    out, _ = _write(monkeypatch, TODOS)
    assert "2 of 2 still open" in out

    partly = [dict(TODOS[0], status="completed"), TODOS[1]]
    out2, _ = _write(monkeypatch, partly)
    assert "1 of 2 still open" in out2


def test_all_completed_clears_and_says_so(monkeypatch):
    done = [dict(t, status="completed") for t in TODOS]
    out, stored = _write(monkeypatch, done)
    assert stored["todos"] == []
    assert "cleared" in out.lower()
    # Nothing to echo — the list is gone.
    assert "Read the code" not in out


def test_invalid_payload_still_errors(monkeypatch):
    monkeypatch.setattr(builtin_todo, "get_bound_session", lambda: "s1")
    out = asyncio.run(builtin_todo._todo_write({"todos": "nope"}))
    assert json.loads(out)["error"]


# -------------------------------------------------------------- anchor


def test_anchor_is_silent_before_the_interval(monkeypatch):
    from app.routers import messages as M
    state = {"todos": list(TODOS)}
    monkeypatch.setattr(M, "_load_session_todos", lambda sid: state["todos"])
    monkeypatch.setitem(M.CONFIG["harness"], "todo_anchor_interval_iterations", 10)

    anchor = M._build_state_anchor("sess")
    for i in range(1, 11):
        assert asyncio.run(anchor(i)) == [], f"fired early at iteration {i}"


def test_anchor_fires_once_the_interval_elapses(monkeypatch):
    from app.routers import messages as M
    state = {"todos": list(TODOS)}
    monkeypatch.setattr(M, "_load_session_todos", lambda sid: state["todos"])
    monkeypatch.setitem(M.CONFIG["harness"], "todo_anchor_interval_iterations", 5)

    anchor = M._build_state_anchor("sess")
    for i in range(1, 6):
        asyncio.run(anchor(i))
    out = asyncio.run(anchor(6))
    assert len(out) == 1, out
    body = out[0]["content"]
    assert out[0]["role"] == "user"
    assert "<active_todos>" in body
    assert "Read the code" in body
    assert "TodoWrite" in body


def test_anchor_resets_when_the_list_changes(monkeypatch):
    """A TodoWrite echo already showed the model the list — don't repeat it."""
    from app.routers import messages as M
    state = {"todos": list(TODOS)}
    monkeypatch.setattr(M, "_load_session_todos", lambda sid: state["todos"])
    monkeypatch.setitem(M.CONFIG["harness"], "todo_anchor_interval_iterations", 3)

    anchor = M._build_state_anchor("sess")
    for i in range(1, 4):
        asyncio.run(anchor(i))
    # Model calls TodoWrite at iteration 3 — list changes, counter resets.
    state["todos"] = [dict(TODOS[0], status="completed"), TODOS[1]]
    assert asyncio.run(anchor(3)) == []
    assert asyncio.run(anchor(5)) == []
    assert len(asyncio.run(anchor(6))) == 1


def test_anchor_silent_when_nothing_is_open(monkeypatch):
    from app.routers import messages as M
    state = {"todos": [dict(t, status="completed") for t in TODOS]}
    monkeypatch.setattr(M, "_load_session_todos", lambda sid: state["todos"])
    monkeypatch.setitem(M.CONFIG["harness"], "todo_anchor_interval_iterations", 2)

    anchor = M._build_state_anchor("sess")
    for i in range(1, 12):
        assert asyncio.run(anchor(i)) == [], f"fired at {i} with no open items"


def test_anchor_silent_with_no_todos(monkeypatch):
    from app.routers import messages as M
    monkeypatch.setattr(M, "_load_session_todos", lambda sid: [])
    monkeypatch.setitem(M.CONFIG["harness"], "todo_anchor_interval_iterations", 2)

    anchor = M._build_state_anchor("sess")
    for i in range(1, 12):
        assert asyncio.run(anchor(i)) == []


def test_interval_zero_disables(monkeypatch):
    from app.routers import messages as M
    monkeypatch.setattr(M, "_load_session_todos", lambda sid: list(TODOS))
    monkeypatch.setitem(M.CONFIG["harness"], "todo_anchor_interval_iterations", 0)

    anchor = M._build_state_anchor("sess")
    for i in range(1, 40):
        assert asyncio.run(anchor(i)) == []


def test_anchor_reads_a_real_session_file(monkeypatch, tmp_path):
    """End-to-end through `_load_session_todos`, not a mocked loader.

    The other anchor tests stub the loader, which would hide a mismatch
    between the on-disk session shape and what the anchor expects.
    """
    from app.routers import messages as M

    session_id = "anchor-real"
    (tmp_path / f"{session_id}.json").write_text(json.dumps({
        "session_id": session_id,
        "messages": [],
        "todos": TODOS,
    }))
    monkeypatch.setattr(M, "SESSIONS_DIR", tmp_path)
    monkeypatch.setitem(M.CONFIG["harness"], "todo_anchor_interval_iterations", 3)

    anchor = M._build_state_anchor(session_id)
    for i in range(1, 4):
        assert asyncio.run(anchor(i)) == []
    out = asyncio.run(anchor(4))
    assert len(out) == 1, out
    assert "Read the code" in out[0]["content"]
    assert "Write the review" in out[0]["content"]


def test_anchor_survives_a_missing_session_file(monkeypatch, tmp_path):
    from app.routers import messages as M

    monkeypatch.setattr(M, "SESSIONS_DIR", tmp_path)
    monkeypatch.setitem(M.CONFIG["harness"], "todo_anchor_interval_iterations", 2)

    anchor = M._build_state_anchor("does-not-exist")
    for i in range(1, 10):
        assert asyncio.run(anchor(i)) == []


# ---------------------------------------------------------------------------
# The budget anchor: a turn cut off at max_turns lands nothing
# ---------------------------------------------------------------------------

def test_budget_anchor_fires_once_at_75_and_90_percent(monkeypatch):
    """The #278 implement round died at 101 of 100 with a gated-and-ready
    change it had not landed; the observer's own budget nudge had been
    exhausted by its inject cap. This one is deterministic."""
    from app.routers import messages as M
    monkeypatch.setattr(M, "_load_session_todos", lambda sid: [])
    anchor = M._build_state_anchor("sess", max_turns=100)
    hits = {i: out for i in range(1, 101) if (out := asyncio.run(anchor(i)))}
    assert sorted(hits) == [75, 90]
    assert "Iteration 75 of 100" in hits[75][0]["content"] and "25 iteration" in hits[75][0]["content"]
    assert "autoimplement_land" in hits[90][0]["content"] and hits[90][0]["role"] == "user"


def test_budget_anchor_is_silent_without_a_budget(monkeypatch):
    from app.routers import messages as M
    monkeypatch.setattr(M, "_load_session_todos", lambda sid: [])
    anchor = M._build_state_anchor("sess", max_turns=0)
    assert all(asyncio.run(anchor(i)) == [] for i in range(1, 50))


def test_budget_and_todo_anchors_ride_the_same_closure(monkeypatch):
    from app.routers import messages as M
    monkeypatch.setattr(M, "_load_session_todos", lambda sid: list(TODOS))
    monkeypatch.setitem(M.CONFIG["harness"], "todo_anchor_interval_iterations", 5)
    anchor = M._build_state_anchor("sess", max_turns=20)
    seen = [(i, asyncio.run(anchor(i))) for i in range(1, 21)]
    budget_at = [i for i, out in seen if any("<budget>" in m["content"] for m in out)]
    todo_at = [i for i, out in seen if any("<active_todos>" in m["content"] for m in out)]
    assert budget_at == [15, 18]
    assert todo_at, "the todo anchor still fires"
