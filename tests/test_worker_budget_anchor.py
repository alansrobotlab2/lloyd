"""The fourth turn path — the direct in-process worker turn — now sees both clocks.

Why this file exists
--------------------
Four turn paths run a model in this process. `autonomy.run_task` warns about
both of its budgets (`tests/test_autonomy_budget_anchor.py`), the chat router
warns about iterations, context and any deadline the caller sent
(`tests/test_context_anchor.py`), and a session-backed worker source goes
through that same router and inherits it. The fourth — `run_prompt_on_primary`
and `run_prompt_with_run_state`, both built by `_worker_run_options` — had
`state_anchor` left at its `None` default, so `app/harness/loop.py` had nothing
to append and the turn's first notice of either budget was the kill.

The population that died not knowing, `workers.db` `runs`, last 30 days at the
time this was filed: `bench-mine` 97 failures, every one
`empty response (stop_reason=max_turns) — nothing written`; `session-distill`
173 of the same plus 21 engine-side `ConnectError` and 2 wall-clock
`TimeoutError`. The iteration clock is therefore the load-bearing half, and
inner voice is off on this path by design, so there is no observer to say it
instead.

What this file pins, and what it deliberately does not
------------------------------------------------------
The anchor being *present on the object the harness is handed*, for both direct
shapes, at the shared builder the two shapes are already pinned to
(`tests/test_workers_sources.py::test_a_worker_turn_reads_the_merged_config_not_the_file`);
the two messages each clock owes, on a hand-driven monotonic clock; and that
appending them leaves position 0 byte-identical, which is what keeps the
prefix cache across iterations.

It does NOT pin the size of either ceiling. bench-mine's `max_turns=8` and
session-distill's `max_turns=15` are #980/#896's decision; this round only
makes the existing cap visible before it bites.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

import app.deadline_anchor
import app.harness.run_state as RS
import workers.sources._common as C
from app.harness import tool_search_cache
from app.harness.loop import run_query
from app.harness.options import RunOptions
from app.harness.run_state import RunState

#: `workers/sources/bench_mine.py:662` and `:683` both pass this, so an
#: 8-iteration bench-mine turn is the shape the iteration anchor is graded on.
BENCH_MINE_TURNS = 8

#: `workers/sources/session_distill.py:204` passes this as
#: `iterations_per_step` to the state-carried shape.
SESSION_DISTILL_TURNS = 15

STATE_SCHEMA = {
    "title": "distill_run_state",
    "type": "object",
    "properties": {"goal": {"type": "string"}},
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Hermetic pieces: no platform state, no engine, a clock the test drives
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_platform_state(monkeypatch, tmp_path):
    """`_worker_run_options` reads the live config, the system-prompt builder
    and the model env. Only the config's per-source wall clock is wanted here,
    so the other two are stubbed exactly as `tests/unit/test_grant_policy.py`
    does — the anchor is not about them."""
    import prompt_builder

    import autonomy
    monkeypatch.setattr(prompt_builder, "build_system_prompt",
                        lambda *a, **k: "WORKER SYSTEM PROMPT")
    monkeypatch.setattr(autonomy, "_get_model_env", lambda *a, **k: {})
    monkeypatch.setenv("LLOYD_GRANT_DB", str(tmp_path / "grants.db"))


@pytest.fixture
def clock(monkeypatch):
    """The monotonic clock `app.deadline_anchor` reads, driven by hand. The
    same fixture as `tests/test_worker_turn_deadline.py`, because the anchor
    built here is that module's builder."""
    state = {"t": 1000.0}
    monkeypatch.setattr(app.deadline_anchor.time, "monotonic", lambda: state["t"])
    return state


async def _fire(anchor, iteration: int = 1) -> list[dict[str, Any]]:
    return await anchor(iteration)


def _text(messages: list[dict]) -> str:
    return " ".join(m["content"] for m in messages)


def _anchor(max_turns: int, source: str):
    opts = C._worker_run_options(max_turns, source=source)
    assert opts.state_anchor is not None, "the builder stopped wiring an anchor"
    return opts.state_anchor


# ---------------------------------------------------------------------------
# Clause 1: the anchor is on the object each shape hands the harness
# ---------------------------------------------------------------------------


def test_the_shared_builder_returns_an_anchor_with_both_clocks():
    """`_worker_run_options` is the one seam — both direct shapes go through it
    and are forbidden by `tests/test_workers_sources.py` from building their own
    `RunOptions`, so an anchor present here cannot be present for one shape and
    absent for the other."""
    opts = C._worker_run_options(BENCH_MINE_TURNS, source="bench-mine")
    assert opts.state_anchor is not None
    assert opts.max_turns == BENCH_MINE_TURNS


def test_the_transcript_shape_hands_run_query_options_that_carry_an_anchor(
        monkeypatch):
    """`run_prompt_on_primary` is the shape bench-mine, session-distill and
    gap-fill run on. Asserted on the object that actually reaches `run_query`,
    not on the builder's return value the caller could still overwrite."""
    seen: list[RunOptions] = []

    async def fake_run_query(messages, options):
        seen.append(options)
        yield {"type": "result", "stop_reason": "stop", "num_turns": 1,
               "usage": {}, "response_text": "the note"}

    async def passthrough(agen, **_kw):
        async for evt in agen:
            yield evt

    monkeypatch.setattr("app.harness.run_query", fake_run_query)
    monkeypatch.setattr("app.run_recorder.record_events", passthrough)
    monkeypatch.setattr("app.sessions_io.create_session", lambda *a, **k: None)

    asyncio.run(C.run_prompt_on_primary("go", BENCH_MINE_TURNS,
                                        source="bench-mine"))

    assert len(seen) == 1
    assert seen[0].state_anchor is not None, \
        "a bench-mine turn still reaches run_query with no budget anchor"


@pytest.mark.asyncio
async def test_the_state_carried_shape_hands_run_state_turn_an_anchor_and_its_own_window(
        monkeypatch, tmp_path, clock):
    """`run_prompt_with_run_state` builds its template through the same
    builder, and its `source` is what decides the wall clock — session-distill's
    600 s pool cap minus `POOL_TIMEOUT_MARGIN_SECONDS` is a 540 s turn, so this
    anchor owes its first warning at 378 s and says 540 s out loud."""
    seen: list[RunOptions] = []

    async def spy_run_state_turn(**kw):
        seen.append(kw["template"])
        return RS.RunStateResult(state=kw["state"], text="the note", done=True)

    monkeypatch.setattr(C, "run_state_turn", spy_run_state_turn)
    state = RunState(job="session-distill", schema=STATE_SCHEMA, run_dir=tmp_path)

    await C.run_prompt_with_run_state(
        "distill this", job="session-distill", source="session-distill",
        state=state, run_dir=tmp_path, iterations_per_step=SESSION_DISTILL_TURNS)

    assert len(seen) == 1
    anchor = seen[0].state_anchor
    assert anchor is not None, \
        "a state-carried worker turn still reaches run_state_turn with no budget anchor"
    clock["t"] += 378
    assert "540s budget remain" in _text(await _fire(anchor))


def test_a_source_with_no_configured_wall_clock_still_gets_the_iteration_anchor():
    """A source with no `max_duration_seconds` entry is bounded by the pool's
    module-level fallback, not by anything `_common` can read, and this path is
    not bounded by `agent.max_turns` either. Inventing a number would warn at 70
    % of a deadline nobody is enforcing — a lie with a figure in it. The cap that
    IS enforced on the turn is `max_turns`, so that half survives alone. (The
    three sources on this path are all configured — bench-mine 1800 s,
    session-distill 600 s, gap-fill 600 s — so this is the shape a new source
    gets until its config lands.)"""
    assert C.turn_timeout_for("source-with-no-config", default=0.0) == 0.0
    anchor = _anchor(BENCH_MINE_TURNS, "source-with-no-config")
    seen = _run_for(anchor, (1, 6, 8))
    assert seen[1] == []
    assert "Iteration 6 of 8" in _text(seen[6])
    assert "budget remain" not in _text(seen[6] + seen[8]), \
        "the wall clock spoke for a source that has no wall clock"


def _run_for(anchor, iterations) -> dict[int, list[dict]]:
    """Fire the anchor once per iteration number and keep each answer."""
    return {i: asyncio.run(_fire(anchor, i)) for i in iterations}


# ---------------------------------------------------------------------------
# Clause 2: the wall clock, on the source's real window
# ---------------------------------------------------------------------------


def test_the_windows_the_anchors_are_built_from_are_the_ones_that_kill_the_turn():
    """`turn_timeout_for` is the pool's `max_duration_seconds` minus
    `POOL_TIMEOUT_MARGIN_SECONDS`, and the pool enforces the larger number
    (`workers/pool.py`) while the turn enforces this one. session-distill is
    configured at 600 s and bench-mine at 1800 s in `config.yaml`, so their
    turns die at 540 s and 1740 s — and 378 s is where session-distill is first
    owed a sentence."""
    assert C.POOL_TIMEOUT_MARGIN_SECONDS == 60
    assert C.turn_timeout_for("session-distill") == 540.0
    assert C.turn_timeout_for("bench-mine") == 1740.0


def test_the_wall_clock_is_silent_while_the_budget_is_healthy(clock):
    anchor = _anchor(SESSION_DISTILL_TURNS, "session-distill")
    clock["t"] += 377            # one second under 70 % of 540 s
    assert _run_for(anchor, (1,))[1] == []


def test_the_wall_clock_warns_once_at_seventy_percent(clock):
    anchor = _anchor(SESSION_DISTILL_TURNS, "session-distill")
    clock["t"] += 378            # 70 % of session-distill's 540 s window
    out = _run_for(anchor, (1,))[1]
    assert len(out) == 1, out
    assert out[0]["role"] == "user"
    assert "162s of this turn's 540s budget remain" in out[0]["content"]
    assert "do not open new" in out[0]["content"]
    clock["t"] += 60             # still under 90 %
    assert _run_for(anchor, (2,))[2] == [], \
        "a warning re-sent every iteration is one the model skips"


def test_the_wall_clock_escalates_once_at_ninety_percent(clock):
    anchor = _anchor(SESSION_DISTILL_TURNS, "session-distill")
    clock["t"] += 486            # 90 % of session-distill's 540 s window
    out = _run_for(anchor, (1,))[1]
    # One iteration spanning both levels emits both, strongest last — the same
    # rule `tests/test_autonomy_budget_anchor.py` pins for the task path.
    assert len(out) == 2, out
    assert "54s of this turn's 540s budget remain" in out[1]["content"]
    assert "Stop calling tools" in out[1]["content"]
    clock["t"] += 10
    assert _run_for(anchor, (2,))[2] == []


# ---------------------------------------------------------------------------
# Clause 3: the iteration clock, on the source's own cap
# ---------------------------------------------------------------------------


def test_the_iteration_clock_warns_at_seven_five_and_ninety_percent_of_the_cap(clock):
    """An 8-iteration bench-mine turn is told at iteration 6 (75 %) and again at
    iteration 8 (90 %), and at no other iteration. Iterations are the clock this
    population actually dies on: 97 bench-mine failures in 30 days, all
    `stop_reason=max_turns`, against 2 wall-clock timeouts from session-distill."""
    seen = _run_for(_anchor(BENCH_MINE_TURNS, "bench-mine"), range(1, 9))
    assert [i for i, msgs in seen.items() if msgs] == [6, 8], \
        "the iteration anchor spoke at the wrong iterations"
    assert "Iteration 6 of 8: 2 iteration(s) remain" in seen[6][0]["content"]
    assert "Iteration 8 of 8: 0 iteration(s) remain" in seen[8][0]["content"]
    assert "gate and land it now" in seen[8][0]["content"], \
        "the 90 % level must tell the turn to wrap up, not just that it is near"


def test_the_iteration_clock_names_the_cap_a_fifteen_turn_source_has(clock):
    """session-distill runs `iterations_per_step=15`, so its first warning is
    owed at iteration 12 (75 % of 15, integer-bounded like the chat path) — not
    at the iteration bench-mine's cap would produce."""
    seen = _run_for(_anchor(SESSION_DISTILL_TURNS, "session-distill"), range(1, 16))
    assert [i for i, msgs in seen.items() if msgs] == [12, 14]
    assert "Iteration 12 of 15: 3 iteration(s) remain" in seen[12][0]["content"]


# ---------------------------------------------------------------------------
# Clause 3, second half: appending the anchor must not cost the prefix cache
# ---------------------------------------------------------------------------


class _FakePool:
    @property
    def discovered(self):
        return [("lloyd-mcp", [{
            "name": "Bash",
            "description": "shell",
            "inputSchema": {"type": "object", "properties": {}},
        }])]

    async def call_tool(self, name: str, args: dict, *, session_id: str = "", **_kw):
        return {"content": f"FAKE_RESULT[{name}]", "is_error": False}


class _Engine:
    """A scripted engine: one canned completion per iteration, and a record of
    the exact message list each request was built from."""

    def __init__(self, turns: int) -> None:
        self.turns = turns
        self.requests: list[list[dict]] = []

    def __call__(self, **kwargs):
        idx = len(self.requests)
        self.requests.append([dict(m) for m in (kwargs.get("messages") or [])])
        final = idx >= self.turns - 1

        async def gen():
            yield {"choices": [{"delta": {"content": f"thought {idx}"}}]}
            if not final:
                yield {"choices": [{"delta": {"tool_calls": [{
                    "index": 0, "id": f"c{idx}", "type": "function",
                    "function": {"name": "Bash",
                                 "arguments": json.dumps({"command": "true"})}},
                ]}}]}
            yield {"choices": [{"delta": {},
                                "finish_reason": "stop" if final else "tool_calls"}]}
            yield {"choices": [], "usage": {"prompt_tokens": 10,
                                            "completion_tokens": 3}}
        return gen()


@pytest.fixture(autouse=True)
def _isolate_the_loop(monkeypatch):
    """Drive the real `run_query` against the builder's own options, with the
    engine and the tool pool stood in for."""
    asyncio.run(tool_search_cache.clear())
    yield
    asyncio.run(tool_search_cache.clear())


async def _drain(messages, options) -> list[dict]:
    return [evt async for evt in run_query(messages, options)]


def test_the_anchor_rides_the_loop_appended_and_position_zero_unchanged(monkeypatch):
    """An 8-iteration bench-mine turn, driven through the real loop.

    The warning must reach the model (it is appended to the message list the
    next request is built from) and position 0 must stay byte-identical across
    every iteration — rewriting the system prompt would re-prefill the whole
    context, which is the reason `RunOptions.state_anchor` appends at all
    (`app/harness/loop.py`).
    """
    async def _pool_async(_options):
        return _FakePool()

    monkeypatch.setattr("app.harness.loop._build_pool", _pool_async)
    engine = _Engine(turns=BENCH_MINE_TURNS)
    monkeypatch.setattr("app.harness.loop.stream_chat", engine)

    options = C._worker_run_options(BENCH_MINE_TURNS, source="bench-mine")
    options.model = "primary"
    options.tool_search_enabled = False
    asyncio.run(_drain([{"role": "user", "content": "mine this run"}], options))

    requests = engine.requests
    assert len(requests) == BENCH_MINE_TURNS, requests
    heads = [msgs[0] for msgs in requests]
    assert all(h["role"] == "system" for h in heads), heads
    assert len({h["content"] for h in heads}) == 1, "position 0 changed mid-turn"
    assert heads[0]["content"] == "WORKER SYSTEM PROMPT"
    for earlier, later in zip(requests, requests[1:]):
        assert later[:len(earlier)] == earlier, "history was rewritten, not appended"
    # Iteration 6 is the first the model can have seen the warning in.
    assert not any("<budget>Iteration" in str(m.get("content") or "")
                   for m in requests[4]), "warned before iteration 6"
    assert any("<budget>Iteration 6 of 8" in str(m.get("content") or "")
               for m in requests[5])
