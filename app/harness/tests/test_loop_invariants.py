"""The loop's standing invariants, as behaviour on the replay seams (P13.0).

Each test drives the real `run_query` through `app/harness/tests/_replay.py`
and asserts on what went over the wire or came back out — never on the text of
the loop. The invariants that already have their own driven test live beside
the code they are about and are named here so this file reads as the index:

  - inject after the batch ....... test_dispatch_split.py::
                                   test_a_hook_inject_during_a_batch_lands_after_the_batch
  - captions in wire order, nudge once
                                   tests/test_caption_ratchet_and_cache.py
  - stdio lock per server ........ test_dispatch_split.py (the `_stdio_pool` tests)
  - read-only batch overlap, sibling failure, close-cancels
                                   test_parallel_dispatch.py
"""
from __future__ import annotations

import asyncio

import pytest

from app.harness import loop as L
from app.harness.errors import ContextOverflowError
from app.harness.options import RunOptions
from app.harness.tests import _replay as R


def _opts(**kw):
    kw.setdefault("max_turns", 6)
    kw.setdefault("tool_search_enabled", False)
    return RunOptions(model="m", **kw)


def _three_iterations():
    return [
        R.Step(reasoning="think one", tool_calls=[R.tool_call("c1", "Read", "a")]),
        R.Step(tool_calls=[R.tool_call("c2", "Bash", "b", command="ls")]),
        R.Step(text="done"),
    ]


# ── position 0 ──────────────────────────────────────────────────────────────

async def test_position_0_is_inserted_once_and_never_changes(monkeypatch):
    """The system prompt goes in at index 0 once per turn and every later
    request is the previous one plus an appended tail — the prefix the engine
    cached is never edited, so no iteration re-prefills it."""
    engine = R.ReplayEngine(_three_iterations())
    R.install(monkeypatch, engine, R.ReplayPool())
    await R.drive(_opts(system_prompt="SYSTEM"))

    assert len(engine.requests) == 3
    for req in engine.requests:
        roles = [m["role"] for m in req.messages]
        assert roles[0] == "system" and roles.count("system") == 1, roles
        assert req.messages[0] == {"role": "system", "content": "SYSTEM"}
    for before, after in zip(engine.requests, engine.requests[1:]):
        n = len(before.messages)
        assert len(after.messages) > n
        assert after.messages[:n] == before.messages, "history was edited, not appended"


async def test_a_caller_supplied_system_message_is_not_doubled(monkeypatch):
    engine = R.ReplayEngine([R.Step(text="done")])
    R.install(monkeypatch, engine, R.ReplayPool())
    await R.drive(_opts(system_prompt="OURS"),
                  [{"role": "system", "content": "THEIRS"},
                   {"role": "user", "content": "go"}])
    sent = engine.requests[0].messages
    assert [m["role"] for m in sent] == ["system", "user"]
    assert sent[0]["content"] == "THEIRS"


# ── wire order ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("parallel", [False, True])
async def test_history_is_in_wire_order_on_both_paths(monkeypatch, parallel):
    pool = R.ReplayPool(delay_by_call_id={"c1": 0.04, "c2": 0.02, "c3": 0.0})
    engine = R.ReplayEngine([
        R.Step(tool_calls=[R.tool_call("c1", "Read", "a"),
                           R.tool_call("c2", "Grep", "b"),
                           R.tool_call("c3", "Glob", "c")]),
        R.Step(text="done"),
    ])
    R.install(monkeypatch, engine, pool)
    await R.drive(_opts(parallel_tool_calls_enabled=parallel))
    sent = engine.requests[1].messages
    assert [m["tool_call_id"] for m in sent if m["role"] == "tool"] == ["c1", "c2", "c3"]
    assert [m["role"] for m in sent] == ["user", "assistant", "tool", "tool", "tool"]


async def test_pre_dispatch_runs_sequentially_in_wire_order_before_any_call(
        monkeypatch):
    """Phase 1 of a concurrent batch stays ordered: the pretool gates (a hook
    deny, ToolSearch's mark_loaded) run one at a time in the order the model
    made the calls, and every one of them finishes before the first MCP call
    starts."""
    from app.harness.hooks import HookRegistry

    pool = R.ReplayPool(delay_by_call_id={"c1": 0.02, "c2": 0.02, "c3": 0.02})
    live = {"n": 0, "peak": 0}
    hooks = HookRegistry()

    async def gate(input_dict, tool_use_id, _ctx):
        live["n"] += 1
        live["peak"] = max(live["peak"], live["n"])
        pool.timeline.append(f"pre:{tool_use_id}")
        await asyncio.sleep(0.01)
        live["n"] -= 1
        return {}
    hooks.add_pre_tool_use(None, gate)

    R.install(monkeypatch, R.ReplayEngine([
        R.Step(tool_calls=[R.tool_call("c1", "Read", "a"),
                           R.tool_call("c2", "Grep", "b"),
                           R.tool_call("c3", "Glob", "c")]),
        R.Step(text="done"),
    ]), pool)
    await R.drive(_opts(parallel_tool_calls_enabled=True, hooks=hooks))

    pres = [t for t in pool.timeline if t.startswith("pre:")]
    assert pres == ["pre:c1", "pre:c2", "pre:c3"]
    assert live["peak"] == 1
    first_start = next(i for i, t in enumerate(pool.timeline) if t.startswith("start:"))
    assert all(pool.timeline.index(p) < first_start for p in pres), pool.timeline
    assert pool.max_inflight == 3, "phase 2 did not overlap"


# ── both reasoning keys ─────────────────────────────────────────────────────

@pytest.mark.parametrize("engine_key", ["reasoning", "reasoning_content"])
async def test_reasoning_goes_back_under_both_keys(monkeypatch, engine_key):
    """Whichever spelling the engine streamed, history carries both: vLLM reads
    `reasoning`, llama.cpp's Qwen3.6 template reads `reasoning_content`, and
    each ignores the other silently."""
    engine = R.ReplayEngine([
        R.Step(reasoning="weighing the options", reasoning_key=engine_key,
               tool_calls=[R.tool_call("c1", "Read", "a")]),
        R.Step(text="done"),
    ])
    R.install(monkeypatch, engine, R.ReplayPool())
    out = await R.drive(_opts(preserve_thinking_iterations=4))

    assert "".join(e["text"] for e in R.of_type(out, "thinking_delta")) \
        == "weighing the options"
    asst = [m for m in engine.requests[1].messages if m["role"] == "assistant"][0]
    assert asst["reasoning"] == asst["reasoning_content"] == "weighing the options"


# ── the finalizer's tools array ─────────────────────────────────────────────

async def test_the_finalizer_sends_the_identical_tools_array(monkeypatch):
    """Qwen renders `tools` inside the system message, so the restating request
    must carry exactly the array the turn's last request carried or the whole
    conversation re-prefills."""
    from app.harness import finalizer

    seen = {}

    async def fake_finalizer(**kw):
        seen.update(kw)
        return {"verdict": "ok"}, "", {}

    monkeypatch.setattr(finalizer, "run_finalizer", fake_finalizer)
    engine = R.ReplayEngine(_three_iterations())
    R.install(monkeypatch, engine, R.ReplayPool())
    out = await R.drive(_opts(final_schema={"type": "object"}))

    last_tools = engine.requests[-1].tools
    assert last_tools, "the fixture advertised no tools"
    assert seen["tools"] == last_tools
    assert seen["chat_messages"][:len(engine.requests[-1].messages)] \
        == engine.requests[-1].messages
    assert R.of_type(out, "result")[0]["structured"] == {"verdict": "ok"}


# ── overflow recoveries ─────────────────────────────────────────────────────

async def test_overflow_recoveries_are_at_most_two(monkeypatch):
    """Two recoveries, then the rejection propagates: an engine that keeps
    refusing cannot spin the loop."""
    overflow = ContextOverflowError("prompt too long", requested_input_tokens=300_000)
    engine = R.ReplayEngine([R.Step(tool_calls=[R.tool_call("c1", "Read", "a")])]
                            + [R.Step(raise_=overflow)] * 5)
    R.install(monkeypatch, engine, R.ReplayPool())
    out: list[dict] = []
    with pytest.raises(ContextOverflowError):
        async for evt in L.run_query([{"role": "user", "content": "go"}], _opts()):
            out.append(evt)
    recoveries = [e for e in R.of_type(out, "stream_raw")
                  if "context_overflow_recovery" in (e.get("error") or "")]
    assert len(recoveries) == 2
    assert len(engine.requests) == 1 + 3      # one good, three refused
