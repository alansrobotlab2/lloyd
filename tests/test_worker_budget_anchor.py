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
    from app import prompt_builder

    from app import autonomy
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
    # A prefix, not the whole thing: `_worker_run_options` appends the turn's
    # deny-list block to the system prompt (#1066), and what this test is about
    # is that position 0 is written once, before iteration 1, and never rewritten
    # — which the line above still asserts on the full string.
    assert heads[0]["content"].startswith("WORKER SYSTEM PROMPT"), heads[0]["content"][:80]
    for earlier, later in zip(requests, requests[1:]):
        assert later[:len(earlier)] == earlier, "history was rewritten, not appended"
    # Iteration 6 is the first the model can have seen the warning in.
    assert not any("<budget>Iteration" in str(m.get("content") or "")
                   for m in requests[4]), "warned before iteration 6"
    assert any("<budget>Iteration 6 of 8" in str(m.get("content") or "")
               for m in requests[5])


#: ── #1554: the scratchpad rides the same seam, and only that seam ────────────
#:
#: Clause 3 of #1554 is a cache claim wearing a memory-system costume: the
#: scratchpad's bytes must never enter the position-0 prefix, because that prefix is
#: built once per turn precisely so it stays KV-cached — 7,648,814 re-prefill tokens
#: across 66 prefix misses in the 24 h the item measured, and 79.6% hit rate since
#: boot, is what it costs when something does move. So the tests below are the byte
#: comparison the budget anchor above already uses, run against a turn that HAS
#: scratchpad notes and one that has none, plus the positive half: the notes must
#: still arrive, appended, on the iterations where the budget is short.
#:
#: Notes are written through the real writer (`app.scratchpad.append`) with the data
#: root repointed, so an injected line demonstrably came from disk rather than from
#: an argument someone remembered to pass.

SCRATCH_SESSION = "worker:session-distill:sc1234"

#: The tag key the harness stamps its own appended messages with. Read through the
#: module the rest of this file imports, not a fresh alias.
SCRATCH_TAG = app.deadline_anchor.ANCHOR_TAG


@pytest.fixture
def scratch_dir(tmp_path, monkeypatch):
    """A data root of our own, so the notes under test are real files on disk."""
    import app.scratchpad as sp
    monkeypatch.setattr(sp, "DATA_ROOT", tmp_path / "data")
    return tmp_path / "data"


def _scratch_notes(session_id: str, *notes: str) -> None:
    import app.scratchpad as sp
    for note in notes:
        sp.append(session_id, note)


def test_the_scratchpad_never_moves_the_cached_position_zero_prefix(
        monkeypatch, scratch_dir):
    """Clause 3: a turn whose scratchpad holds content assembles the same
    prompt-prefix bytes as one whose scratchpad is empty.

    Two real loop runs over the same 15-iteration session-distill cap — one
    session with three notes on disk, one with none. Position 0 must be
    byte-identical between them and stable across all fifteen requests, because
    the prefix is built once precisely so it stays KV-cached (7,648,814
    re-prefill tokens across 66 misses in the 24 h #1554 measured; that is the
    invoice for getting this wrong by accident). The notes still arrive, and only
    as appended messages after the turn is short on iterations — the positive
    half, so an inert injector cannot pass by emitting nothing.
    """
    import app.scratchpad as sp

    async def _pool_async(_options):
        return _FakePool()

    monkeypatch.setattr("app.harness.loop._build_pool", _pool_async)

    def _drive(session_id: str, engine: _Engine) -> list[list[dict]]:
        monkeypatch.setattr("app.harness.loop.stream_chat", engine)
        options = C._worker_run_options(
            SESSION_DISTILL_TURNS, source="session-distill", session_id=session_id)
        options.model = "primary"
        options.tool_search_enabled = False
        asyncio.run(_drain([{"role": "user", "content": "do the distill pass"}],
                           options))
        return engine.requests

    _scratch_notes(SCRATCH_SESSION, "ruled out the 4k window",
                   "hypothesis: kv gate", "next: raise batch size")
    assert sp.read(SCRATCH_SESSION), "the fixture stored nothing, so it proves nothing"

    with_notes = _Engine(turns=SESSION_DISTILL_TURNS)
    req_notes = _drive(SCRATCH_SESSION, with_notes)
    empty = _Engine(turns=SESSION_DISTILL_TURNS)
    req_empty = _drive("worker:session-distill:none11", empty)

    assert len(req_notes) == SESSION_DISTILL_TURNS, len(req_notes)
    heads_notes = [m[0]["content"] for m in req_notes]
    heads_empty = [m[0]["content"] for m in req_empty]
    assert len(set(heads_notes)) == 1, "position 0 moved mid-turn"
    assert heads_notes[0] == heads_empty[0], (
        "a session with scratchpad notes assembles a different cached prefix than "
        "an empty one: the scratchpad is reaching position 0")

    # The notes do reach the model — appended, and only once the cap is in sight.
    def _has_note(msgs: list[dict]) -> bool:
        return any("kv gate" in str(m.get("content")) for m in msgs[1:])

    assert not _has_note(req_notes[0]), "injected on iteration 1, before position 0 " \
        "was even cached — the point of the ceiling is that this costs nothing here"
    late = [i for i, msgs in enumerate(req_notes) if _has_note(msgs)]
    assert late, "the scratchpad never reached the model at all: the affordance is inert"
    assert min(late) >= 7, f"first injection at request index {min(late)} of " \
        f"{SESSION_DISTILL_TURNS}, before the 50 % level (index 7 = iteration 8)"
    for i, msgs in enumerate(req_notes):
        if not _has_note(msgs):
            continue
        positions = [j for j, m in enumerate(msgs)
                     if "kv gate" in str(m.get("content"))]
        assert all(p > 0 for p in positions), f"note at position 0 on request {i}"


def test_the_scratchpad_clock_fires_only_at_its_own_levels(scratch_dir):
    """The other half of clause 3: there IS an append path, it fires at 50 % and
    80 % of the cap, and at no other iteration. A test that only asserted absence
    would leave a feature that never injects green, which is the most common way a
    pinned clause fails to pin anything.

    session-distill's cap is 15 turns, so its scratchpad levels are iteration 8
    (50 %) and iteration 12 (80 %).
    """
    _scratch_notes(SCRATCH_SESSION, "hypothesis: kv gate")
    anchor = C._worker_state_anchor(
        SESSION_DISTILL_TURNS, "session-distill", session_id=SCRATCH_SESSION)

    def scratch_msgs(i: int) -> list[dict]:
        return [m for m in _run_for(anchor, (i,))[i]
                if m.get(SCRATCH_TAG, {}).get("kind") == "scratchpad"]

    assert scratch_msgs(1) == [] and scratch_msgs(7) == [], (
        "a scratchpad was injected before the 50 % level, where the model still has "
        "its own history in context and the tokens would be paid for twice")

    first = scratch_msgs(8)
    assert len(first) == 1 and "hypothesis: kv gate" in first[0]["content"], first
    assert first[0]["role"] == "user", (
        "the injection is not a user-role append: the harness stamps and sweeps only "
        "user-role anchors, so anything else reads to the model as an instruction "
        "from the operator and is never removed")

    assert scratch_msgs(11) == [], "the 50 % level re-fired"
    assert len(scratch_msgs(12)) == 1, "the 80 % level did not fire"
    assert scratch_msgs(15) == [], (
        "both levels re-fired at the cap; the notes are already in context")


def test_a_turn_that_named_no_session_gets_no_scratchpad_clock(scratch_dir):
    """`run_prompt_with_run_state` builds its template before any session exists,
    so it names no id and must get exactly the pre-feature anchor. An empty id that
    resolved to *someone's* scratchpad — or to a directory-wide one — would leak one
    run's notes into another turn while every with-session test stayed green."""
    _scratch_notes(SCRATCH_SESSION, "someone else's notes")
    plain = C._worker_state_anchor(SESSION_DISTILL_TURNS, "session-distill")
    assert plain is not None
    for i in (1, 7, 8, 12, 15):
        kinds = [m.get(SCRATCH_TAG, {}).get("kind")
                 for m in _run_for(plain, (i,))[i]]
        assert "scratchpad" not in kinds, f"iteration {i}: kinds={kinds}"


def test_the_third_clock_does_not_disturb_the_two_that_were_there(scratch_dir, clock):
    """Three clocks ride one hook now, so the two that predate this feature have to
    keep firing on their own numbers: the iteration clock at 75 % of 15 (iteration
    12) and session-distill's wall clock at 378 s of its 540 s window.
    `compose_state_anchors` promises that ordering, and a third clock returning
    early is exactly how that promise breaks."""
    _scratch_notes(SCRATCH_SESSION, "hypothesis: kv gate")
    anchor = C._worker_state_anchor(
        SESSION_DISTILL_TURNS, "session-distill", session_id=SCRATCH_SESSION)

    kinds = [m.get(SCRATCH_TAG, {}).get("kind") for m in _run_for(anchor, (12,))[12]]
    assert "iteration_budget" in kinds, f"kinds={kinds}: the budget clock went quiet"
    assert "scratchpad" in kinds, f"kinds={kinds}: the scratchpad clock never joined"

    clock["t"] += 378                            # 70 % of the 540 s turn window
    # Iteration 13 rather than 12: the other two clocks already spent their level at
    # 12 and each fires once, so anything they return here would be a double.
    late = [m.get(SCRATCH_TAG, {}).get("kind") for m in asyncio.run(anchor(13))]
    assert late == ["deadline_budget"], (
        f"kinds={late}: the wall clock either went quiet behind the scratchpad or "
        f"re-fired a level it had already warned on")
