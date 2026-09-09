"""The wall-clock budget an autonomy run dies on must be visible to the model.

Why this file exists
--------------------
Two budgets bound an agent turn and only one of them was ever announced. The
chat path warns at 75%/90% of `max_turns`
(`app/routers/messages.py::_build_state_anchor`). An autonomy run is not bounded
by iterations at all — it dies on `asyncio.timeout(timeout_seconds)` — and
nothing told the model that clock existed.

On 2026-09-08 task #80 (OKF Conformance Check) failed three consecutive runs at
its 300s budget. `validate_okf.py`, the entire scripted deliverable, takes 2.4
seconds. All three run records read "(no output before timeout)": the model had
the answer within the first minute and spent the rest investigating, then was
killed without ever being asked to write it down. #78 and #24 failed the same
way in the same window.

A timeout the model cannot see is a deadline it cannot meet, so the fix is to
say it out loud — twice, escalating — through the `state_anchor` hook the loop
already calls at the top of every iteration.
"""
from __future__ import annotations

import pytest

import autonomy
from app import deadline_anchor


async def collect(anchor, iterations: int = 1) -> list[dict]:
    out = []
    for i in range(1, iterations + 1):
        out.extend(await anchor(i))
    return out


def text(messages) -> str:
    return " ".join(m["content"] for m in messages)


@pytest.fixture
def clock(monkeypatch):
    """A monotonic clock the test drives by hand."""
    state = {"t": 1000.0}
    # The clock lives in `app.deadline_anchor` now — one definition, shared
    # with the autoimplement worker turn. `autonomy._build_deadline_anchor` is a
    # thin flavour of it, and these tests still go through that entry point.
    monkeypatch.setattr(deadline_anchor.time, "monotonic", lambda: state["t"])
    return state


def test_no_anchor_without_a_budget():
    """`state_anchor=None` is the harness's own "nothing to say" — a zero or
    negative timeout must not produce a callback that divides by it."""
    assert autonomy._build_deadline_anchor(0) is None
    assert autonomy._build_deadline_anchor(-1) is None
    assert autonomy._build_deadline_anchor(None) is None


@pytest.mark.asyncio
async def test_silent_while_the_budget_is_healthy(clock):
    anchor = autonomy._build_deadline_anchor(300)
    clock["t"] += 200  # 67% — under the first threshold
    assert await collect(anchor, 5) == []


@pytest.mark.asyncio
async def test_warns_once_at_seventy_percent(clock):
    anchor = autonomy._build_deadline_anchor(300)
    clock["t"] += 210  # exactly 70%
    first = await collect(anchor)
    assert len(first) == 1
    assert "budget" in text(first)

    # The level is spent. A warning re-sent every iteration is a warning the
    # model learns to skip, and it re-prefills nothing but noise.
    clock["t"] += 10
    assert await collect(anchor, 5) == []


@pytest.mark.asyncio
async def test_the_final_warning_says_stop_calling_tools(clock):
    """The two levels are not the same message. At 70% the useful instruction
    is "don't start anything new"; at 90% it is "stop and write", because a run
    that spends its last seconds on one more tool call reports nothing at all."""
    anchor = autonomy._build_deadline_anchor(600)
    clock["t"] += 420
    warn = text(await collect(anchor))
    clock["t"] += 120  # 540s = 90%
    final = text(await collect(anchor))

    assert "do not open new" in warn.lower()
    assert "stop calling tools" in final.lower()
    assert "report" in final.lower()


@pytest.mark.asyncio
async def test_a_slow_iteration_still_gets_both_levels(clock):
    """Thresholds are levels crossed, not instants hit. One tool call spanning
    both fractions must not swallow the warning it jumped over — the model gets
    the strongest applicable message on its next iteration either way."""
    anchor = autonomy._build_deadline_anchor(300)
    clock["t"] += 295
    both = await collect(anchor)
    assert len(both) == 2


@pytest.mark.asyncio
async def test_remaining_seconds_are_reported_not_the_elapsed(clock):
    anchor = autonomy._build_deadline_anchor(1000)
    clock["t"] += 700
    assert "300s of this task's 1000s budget remain" in text(await collect(anchor))


@pytest.mark.asyncio
async def test_anchors_are_user_messages(clock):
    """The loop appends these to `chat_messages` verbatim. A `system` role mid
    conversation is not the position-0 system prompt and reads as a tool of the
    model's own; the chat path's budget anchor is a user message for the same
    reason."""
    anchor = autonomy._build_deadline_anchor(100)
    clock["t"] += 95
    assert all(m["role"] == "user" for m in await collect(anchor))


def test_run_task_wires_the_anchor_to_the_resolved_timeout():
    """The anchor must be built from `timeout` — the value `asyncio.timeout`
    actually gets after the pool-cap clamp — not from the frontmatter's
    `timeout_seconds`. Warning at 70% of a budget the run does not have is
    worse than not warning: it arrives after the kill."""
    source = __import__("inspect").getsource(autonomy.run_task)
    assert "state_anchor=_build_deadline_anchor(timeout)" in source
    # `timeout` is the clamped value; `declared_timeout` is not.
    assert "state_anchor=_build_deadline_anchor(declared_timeout)" not in source
