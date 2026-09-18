"""Both budgets an autonomy run dies on must be visible to the model.

Why this file exists
--------------------
Two clocks bound an autonomy run and for a while neither was announced; then
one was fixed and the other was left. The wall clock is
`asyncio.timeout(timeout_seconds)`, and on 2026-09-08 task #80 (OKF Conformance
Check) failed three consecutive runs at its 300s budget — `validate_okf.py`,
the entire scripted deliverable, takes 2.4 seconds, and all three run records
read "(no output before timeout)". #78 and #24 failed the same way in the same
window, so the wall clock got an anchor (#80, 2026-09-08).

The iteration clock is `RunOptions.max_turns`, and it is still killing runs
unannounced: `app/harness/loop.py` breaks with `stop_reason="max_turns"`,
`app/harness/finalizer.py` records no verdict for that death, and the only
anchor wired on this path had nothing to say about iterations. `workers.db`
holds 20 scheduled-task runs that died at `turns=61` against `agent.max_turns: 60`
between 2026-09-04 and 2026-09-18, across nine tasks (#24, #30, #39, #40, #42,
#47, #58, #65, #83). Six of them predate the wall-clock anchor's landing
(`af038eb`, 2026-09-08 20:21 −07:00). Of the 14 after it, every single one
finished its 61 iterations below 70% of its own clamped timeout — shortest
269 s, longest 2059 s against a 2499 s level (#24, 2026-09-10) — so the
deadline anchor below could not have spoken on any of them. That population is
what the second half of this file pins (#1061).

A budget the model cannot see is one it cannot meet, so the fix is the same in
both cases: say it out loud — twice, escalating — through the `state_anchor`
hook the loop already calls at the top of every iteration. One callable, both
clocks.
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
    # with the autocode worker turn. `autonomy._build_deadline_anchor` is a
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


def test_run_task_wires_one_composed_anchor_to_both_resolved_values():
    """`run_task` passes ONE `state_anchor`, built from `timeout` — the value
    `asyncio.timeout` actually gets after the pool-cap clamp — and from the same
    `max_turns` it hands `RunOptions`.

    Warning at 70% of a budget the run does not have is worse than not warning:
    it arrives after the kill. And two anchors on one callable would be two
    chances to drift, so the composition lives behind one call (#1061).
    """
    source = __import__("inspect").getsource(autonomy.run_task)
    assert "state_anchor=_build_task_anchor(timeout, max_turns)" in source
    # `timeout` is the clamped value; `declared_timeout` is not.
    assert "state_anchor=_build_task_anchor(declared_timeout" not in source
    assert "state_anchor=_build_deadline_anchor(declared_timeout" not in source
    # The cap the anchor warns against is the cap the harness enforces: one
    # local feeds both `RunOptions.max_turns` and the anchor.
    assert "max_turns=max_turns," in source


# ── The iteration clock (#1061) ──────────────────────────────────────────────
#
# The wall clock above is only half the budget. `agent.max_turns` is 60 and the
# loop stops a run the iteration after it crosses that, so the runs that die
# this way die at `turns=61` with no verdict recorded at all. Measured on this
# box on 2026-09-18 in `workers.db`: 20 scheduled-task runs carry
# `stop_reason=max_turns, turns=61`, 2026-09-04 → 2026-09-18, across nine tasks
# (#24, #30, #39, #40, #42, #47, #58, #65, #83), the shortest lasting 269 s.
# Task #39 alone has six of them, and it declares `timeout_seconds: 2400` — its
# 70% wall-clock level is 1680 s, and every one of its six deaths landed between
# 269 s and 1114 s. On all six the wired anchor could not have spoken.


@pytest.mark.asyncio
async def test_iteration_warnings_fire_at_seventy_five_and_ninety_percent(clock):
    """Once per level, on a `role: user` `<budget>` message, at 75% and 90% of
    the run's own `max_turns` — the two levels the chat path uses.

    100 is used instead of the live 60 so the threshold maths is readable: 75
    and 90 are the crossings, and they are the assertions, not the cap.
    """
    anchor = autonomy._build_task_anchor(2400, 100)   # #39's timeout, clean cap
    hits: dict[int, list[dict]] = {}
    for i in range(1, 101):
        fired = [m for m in await anchor(i) if "<budget>Iteration" in m["content"]]
        if fired:
            hits[i] = fired
    assert sorted(hits) == [75, 90], "each level fires once, and nowhere else"
    assert all(m["role"] == "user" for level in hits.values() for m in level)
    assert hits[75][0]["content"].startswith(
        "<budget>Iteration 75 of 100: 25 iteration(s) remain before this turn is stopped."
    )
    assert hits[90][0]["content"].startswith(
        "<budget>Iteration 90 of 100: 10 iteration(s) remain before this turn is stopped."
    )


@pytest.mark.asyncio
async def test_one_slow_iteration_jumping_both_levels_emits_both_warnings(clock):
    """Thresholds are levels crossed, not instants hit — the same rule the
    wall-clock half pins. A run that resumes at 96 after a long tool call still
    hears the 75% message it jumped over."""
    anchor = autonomy._build_task_anchor(2400, 100)
    both = [m for m in await anchor(96) if "<budget>Iteration" in m["content"]]
    assert len(both) == 2
    assert "Iteration 96 of 100" in both[0]["content"]
    assert "Iteration 96 of 100" in both[1]["content"]


@pytest.mark.asyncio
async def test_the_composed_anchor_keeps_both_wall_clock_sentences_byte_identical(clock):
    """Adding the iteration clock must not cost the run either wall-clock
    warning, nor reword it: `app.deadline_anchor` is the one definition, and the
    reference instance built from it is the byte-for-byte contract."""
    anchor = autonomy._build_task_anchor(1000, 100)
    reference = deadline_anchor.build_deadline_anchor(1000, what="task")

    clock["t"] += 700                       # exactly 70% of 1000s
    composed = await anchor(10)             # iteration 10 of 100: no iteration warning
    assert composed == await reference(10)
    assert "300s of this task's 1000s budget remain" in text(composed)
    assert "do not open new" in text(composed).lower()

    clock["t"] += 200                       # 900s = 90%
    composed = await anchor(11)
    assert composed == await reference(11)
    assert "100s of this task's 1000s budget remain" in text(composed)
    assert "stop calling tools" in text(composed).lower()


@pytest.mark.asyncio
async def test_a_run_with_no_wall_clock_is_never_told_it_has_one(clock):
    """`timeout <= 0` used to mean `state_anchor=None` — no anchor at all. With
    a turn cap in play that is the wrong answer: the run still dies at
    `max_turns`, and now has no warning either. So the composed builder returns
    an iteration-only anchor, and still no wall-clock sentence."""
    anchor = autonomy._build_task_anchor(0, 100)
    assert anchor is not None, "a turn cap is a budget even with no wall clock"
    clock["t"] += 1_000_000     # a wall clock this run does not have, ignored
    at_75 = await anchor(75)
    assert len(at_75) == 1, at_75
    assert "Iteration 75 of 100" in at_75[0]["content"]
    assert not any("budget remain" in m["content"] for m in at_75)
    assert len(await anchor(90)) == 1, "the second level still fires once"


def test_no_anchor_at_all_without_either_budget():
    """No wall clock and no turn cap means nothing to warn about, and the
    harness treats `state_anchor=None` as its own 'nothing to say'."""
    assert autonomy._build_task_anchor(0, 0) is None
    assert autonomy._build_task_anchor(None, None) is None
    assert autonomy._build_task_anchor(-1, -5) is None


@pytest.mark.asyncio
async def test_the_composed_anchor_survives_the_seam_the_loop_calls_it_through(clock):
    """`app/harness/loop.py` calls the anchor as `await options.state_anchor(num_turns)`
    inside a `try` that swallows anything it raises and logs a warning.

    That swallow is why this seam needs its own test rather than trust in the
    one above: a composed anchor that is not awaitable, or returns something
    other than `{role, content}` dicts, does not crash the run — it produces a
    warning nobody hears and a run record with no verdict, which is precisely
    the failure this item is about, indistinguishable from never wiring anything.
    So build the anchor the way `run_task` wires it, put it through the real
    `RunOptions` field, and call it the way the loop does.
    """
    from app.harness.options import RunOptions

    options = RunOptions(
        model="test-model",
        max_turns=60,
        state_anchor=autonomy._build_task_anchor(2400, 60),
    )
    assert options.state_anchor is not None

    early = await options.state_anchor(44)     # 44/60 = 73%: under the first level
    assert early == []

    fired = await options.state_anchor(45)     # exactly 75% of 60
    assert len(fired) == 1, fired
    assert set(fired[0]) >= {"role", "content"}
    assert fired[0]["role"] == "user"
    assert fired[0]["content"].startswith(
        "<budget>Iteration 45 of 60: 15 iteration(s) remain before this turn is stopped."
    )

    late = await options.state_anchor(54)      # exactly 90% of 60
    assert len(late) == 1 and "Iteration 54 of 60" in late[0]["content"]
