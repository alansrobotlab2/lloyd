"""Concurrent dispatch for read-only batches.

Two properties decide whether this is safe rather than merely faster:

* **Qualification comes from the server's own `readOnlyHint`**, not from a
  list kept in the harness. One writer in the batch makes the whole batch
  sequential — including `Bash`, whatever the command says, because
  classifying shell commands is the guessing game the safety hook
  deliberately refuses to play.
* **History goes back in wire order** even when results land out of order,
  so the replayed conversation matches the assistant message's own
  tool_calls array.
"""

from __future__ import annotations

import asyncio
import inspect

from app.harness import loop as L
from app.harness.options import RunOptions
from app.harness.tool_search import LoadedToolSet


def _tc(name, call_id, args=None):
    return {"id": call_id, "function": {"name": name, "arguments": "{}"},
            "_args_dict": dict(args or {}), "_summary": "s"}


class _Pool:
    """Records dispatch overlap so a test can prove concurrency, not just speed."""

    def __init__(self, read_only=("Read", "Grep", "Glob"), delays=None):
        self.discovered = [("lloyd-mcp", [
            {"name": n, "description": "", "inputSchema": {},
             "annotations": {"readOnlyHint": n in read_only}}
            for n in ("Read", "Grep", "Glob", "Bash", "Edit", "Write")
        ])]
        self.delays = delays or {}
        self.inflight = 0
        self.max_inflight = 0
        self.order: list[str] = []

    async def call_tool(self, name, args, **kw):
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            await asyncio.sleep(self.delays.get(kw.get("call_id") or "", 0))
        finally:
            self.inflight -= 1
        self.order.append(kw.get("call_id") or name)
        return {"content": f"result of {name}", "is_error": False}


# ── qualification ───────────────────────────────────────────────────────────

def test_a_batch_of_read_only_calls_qualifies():
    pool = _Pool()
    calls = [_tc("Read", "c1"), _tc("Grep", "c2")]
    assert L._batch_is_read_only(calls, L._read_only_names(pool))


def test_one_writer_disqualifies_the_whole_batch():
    pool = _Pool()
    ro = L._read_only_names(pool)
    for writer in ("Bash", "Edit", "Write"):
        calls = [_tc("Read", "c1"), _tc(writer, "c2"), _tc("Grep", "c3")]
        assert not L._batch_is_read_only(calls, ro), writer


def test_bash_never_qualifies_however_harmless_the_command():
    """Classifying shell commands is what the safety hook refuses to do."""
    pool = _Pool()
    calls = [_tc("Read", "c1"), _tc("Bash", "c2", {"command": "cat /etc/hostname"})]
    assert not L._batch_is_read_only(calls, L._read_only_names(pool))


def test_a_server_with_no_hints_qualifies_nothing():
    pool = _Pool()
    pool.discovered = [("other", [{"name": "mystery", "description": "",
                                   "inputSchema": {}}])]
    assert L._read_only_names(pool) == set()
    assert not L._batch_is_read_only([_tc("mystery", "c1")], set())


def test_a_parse_error_and_toolsearch_do_not_disqualify():
    pool = _Pool()
    ro = L._read_only_names(pool)
    calls = [_tc("Read", "c1"),
             _tc("whatever", "c2", {"__parse_error__": True}),
             _tc("ToolSearch", "c3")]
    assert L._batch_is_read_only(calls, ro)


def test_read_only_names_comes_from_the_annotations_a_real_pool_discovered():
    """The hint is read off what `MCPPool` discovered from the server, not off
    a list kept in the harness: a tool the server marks read-only qualifies, the
    same name unmarked does not, and nothing is inferred from the name."""
    from app.harness.mcp_pool import MCPPool

    pool = MCPPool({})
    pool._register("srv", [
        {"name": "Read", "inputSchema": {}, "annotations": {"readOnlyHint": True}},
        {"name": "Grep", "inputSchema": {}, "annotations": {"readOnlyHint": False}},
        {"name": "custom_lookup", "inputSchema": {},
         "annotations": {"readOnlyHint": True}},
        {"name": "Glob", "inputSchema": {}},
    ])
    assert L._read_only_names(pool) == {"Read", "custom_lookup"}


# ── the batch runs ──────────────────────────────────────────────────────────

async def _drain(pool, calls, **opts):
    """Run one iteration's dispatch through run_query's real batch code.

    Rather than re-implement the loop, this exercises the pieces it calls in
    the same order, which is what the ordering assertions are about.
    """
    options = RunOptions(model="m", parallel_tool_calls_enabled=True, **opts)
    loaded = LoadedToolSet(enabled=False, catalog=[], loaded=set())
    chat: list[dict] = []
    results = {}
    sem = asyncio.Semaphore(options.parallel_tool_calls_max_concurrency)

    async def run_one(tc):
        async with sem:
            return await L._execute_tool_call(tc=tc, pool=pool, options=options,
                                              session_id="s")

    for tc in calls:
        assert await L._pre_dispatch(tc=tc, options=options, session_id="s",
                                     loaded_set=loaded) is None
    tasks = [asyncio.create_task(run_one(tc)) for tc in calls]
    landed = []
    for fut in asyncio.as_completed(tasks):
        evt = await fut
        landed.append(evt["call_id"])
        results[evt["call_id"]] = evt
    for tc in calls:
        chat.append({"role": "tool", "tool_call_id": tc["id"],
                     "content": results[tc["id"]]["content"]})
    return chat, landed


async def test_calls_actually_overlap():
    pool = _Pool(delays={"c1": 0.05, "c2": 0.05, "c3": 0.05})
    await _drain(pool, [_tc("Read", "c1"), _tc("Grep", "c2"), _tc("Glob", "c3")])
    assert pool.max_inflight >= 2, "the batch ran one at a time"


async def test_concurrency_is_bounded():
    pool = _Pool(delays={f"c{i}": 0.05 for i in range(6)})
    calls = [_tc("Read", f"c{i}") for i in range(6)]
    await _drain(pool, calls, parallel_tool_calls_max_concurrency=2)
    assert pool.max_inflight <= 2


async def test_history_is_wire_order_even_when_results_land_reversed():
    pool = _Pool(delays={"c1": 0.06, "c2": 0.03, "c3": 0.0})
    calls = [_tc("Read", "c1"), _tc("Grep", "c2"), _tc("Glob", "c3")]
    chat, landed = await _drain(pool, calls)
    assert landed[0] == "c3", "the fixture did not actually reorder"
    assert [m["tool_call_id"] for m in chat] == ["c1", "c2", "c3"]


# ── the loop wiring ─────────────────────────────────────────────────────────
#
# Driven through the real `run_query` on the replay seams
# (`app/harness/tests/_replay.py`), never by reading its source.

def test_captions_are_accounted_in_wire_order():
    """The ratchet is about the first miss, not the first to come back."""
    calls = [_tc("Read", "c1"), _tc("Grep", "c2"), _tc("Glob", "c3")]
    calls[0]["_summary"] = "looking"
    calls[1]["_summary"] = ""
    calls[2]["_summary"] = ""
    total, present, nudged, nudge_id = L._account_captions(
        calls, {"Read", "Grep", "Glob"}, 0, 0, False, "s", 1)
    assert (total, present, nudged) == (3, 1, True)
    assert nudge_id == "c2", "the nudge must land on the first miss in wire order"


def test_captions_are_only_counted_for_tools_that_asked_for_one():
    calls = [_tc("Read", "c1"), _tc("session_inject_context", "c2")]
    total, present, _n, _i = L._account_captions(
        calls, {"Read"}, 0, 0, False, "s", 1)
    assert total == 1 and present == 1


def test_a_subagent_reads_the_same_config_keys():
    src = inspect.getsource(__import__("agent_mcp.builtin_task",
                                       fromlist=["_task"])._task)
    assert "parallel_tool_calls_enabled=" in src
    assert "parallel_tool_calls_max_concurrency=" in src


def test_config_maps_the_flag(monkeypatch):
    from app import mcp_discovery as D
    from app.config import CONFIG

    monkeypatch.setitem(CONFIG, "harness",
                        {**CONFIG.get("harness", {}),
                         "parallel_tool_calls": {"enabled": True,
                                                 "max_concurrency": 7}})
    kw = D._get_harness_kwargs()
    assert kw["parallel_tool_calls_enabled"] is True
    assert kw["parallel_tool_calls_max_concurrency"] == 7


def test_the_flag_ships_off():
    import yaml
    from pathlib import Path
    root = Path(L.__file__).resolve().parent.parent.parent
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    assert cfg["harness"]["parallel_tool_calls"]["enabled"] is False
    assert RunOptions(model="m").parallel_tool_calls_enabled is False


# ── end to end through run_query ────────────────────────────────────────────
#
# Everything above tests the pieces. These drive the real loop, because the
# thing that would actually break is the wiring between them.

import json  # noqa: E402


class _E2EPool:
    def __init__(self, read_only=("Read", "graph_explain"), delays=None):
        self._discovered = [("lloyd-mcp", [
            {"name": n, "description": "", "inputSchema": {"type": "object",
                                                           "properties": {}},
             "annotations": {"readOnlyHint": n in read_only}}
            for n in ("Read", "graph_explain", "Bash")
        ])]
        self.delays = delays or {}
        self.inflight = 0
        self.max_inflight = 0

    @property
    def discovered(self):
        return self._discovered

    async def call_tool(self, name, args, **kw):
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            await asyncio.sleep(self.delays.get(name, 0))
        finally:
            self.inflight -= 1
        return {"content": f"RESULT[{name}]", "is_error": False}


def _script(tool_calls):
    """One iteration that makes `tool_calls`, then one that answers."""
    turns = [tool_calls, []]

    def stream(**kwargs):
        idx = stream.n
        stream.n += 1
        return _gen(turns[idx] if idx < len(turns) else [])

    stream.n = 0

    async def _gen(calls):
        for i, tc in enumerate(calls):
            yield {"choices": [{"delta": {"tool_calls": [{
                "index": i, "id": tc["id"], "type": "function",
                "function": {"name": tc["name"],
                             "arguments": json.dumps(tc.get("arguments") or {})},
            }]}}]}
        if not calls:
            yield {"choices": [{"delta": {"content": "done"}}]}
        yield {"choices": [{"delta": {},
                            "finish_reason": "tool_calls" if calls else "stop"}]}
        yield {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    return stream


async def _run(monkeypatch, pool, calls, **opts):
    monkeypatch.setattr("app.harness.loop._build_pool",
                        lambda _o: _ready(pool))
    monkeypatch.setattr("app.harness.loop.stream_chat", _script(calls))
    options = RunOptions(model="m", max_turns=4, tool_search_enabled=False, **opts)
    out = []
    async for evt in L.run_query([{"role": "user", "content": "go"}], options):
        out.append(evt)
    return out


async def _ready(pool):
    return pool


READ_ONLY_BATCH = [
    {"id": "c1", "name": "graph_explain", "arguments": {"symbol": "run_query"}},
    {"id": "c2", "name": "Read", "arguments": {"file_path": "/x"}},
]

MIXED_BATCH = [
    {"id": "c1", "name": "Read", "arguments": {"file_path": "/x"}},
    {"id": "c2", "name": "Bash", "arguments": {"command": "ls"}},
]


def _frames(events_out):
    return [(e["type"], e.get("call_id")) for e in events_out
            if e["type"] in ("tool_call", "tool_result")]


async def test_a_read_only_batch_emits_both_calls_before_the_first_result(monkeypatch):
    pool = _E2EPool(delays={"graph_explain": 0.02, "Read": 0.02})
    out = await _run(monkeypatch, pool, READ_ONLY_BATCH,
                     parallel_tool_calls_enabled=True)
    frames = _frames(out)
    assert frames[0][0] == "tool_call" and frames[1][0] == "tool_call"
    assert frames[2][0] == "tool_result"
    assert pool.max_inflight == 2


async def test_a_mixed_batch_stays_interleaved(monkeypatch):
    pool = _E2EPool(delays={"Read": 0.02, "Bash": 0.02})
    out = await _run(monkeypatch, pool, MIXED_BATCH,
                     parallel_tool_calls_enabled=True)
    assert [t for t, _ in _frames(out)] == [
        "tool_call", "tool_result", "tool_call", "tool_result"]
    assert pool.max_inflight == 1


async def test_the_flag_off_keeps_a_read_only_batch_interleaved(monkeypatch):
    pool = _E2EPool(delays={"graph_explain": 0.02, "Read": 0.02})
    out = await _run(monkeypatch, pool, READ_ONLY_BATCH,
                     parallel_tool_calls_enabled=False)
    assert [t for t, _ in _frames(out)] == [
        "tool_call", "tool_result", "tool_call", "tool_result"]
    assert pool.max_inflight == 1


async def test_a_single_read_only_call_does_not_take_the_batch_path(monkeypatch):
    pool = _E2EPool()
    out = await _run(monkeypatch, pool, [READ_ONLY_BATCH[0]],
                     parallel_tool_calls_enabled=True)
    assert [t for t, _ in _frames(out)] == ["tool_call", "tool_result"]


async def test_an_inject_during_a_parallel_batch_lands_after_the_tool_messages(
        monkeypatch):
    """The shape fix, exercised through the real loop."""
    from app.harness.hooks import HookRegistry

    pool = _E2EPool(delays={"graph_explain": 0.03, "Read": 0.0})
    handle: list[dict] = []
    hooks = HookRegistry()

    async def inject(input_dict, tool_use_id, _ctx):
        if input_dict["tool_name"] == "Read":
            handle.append({"role": "user", "content": "[INNER VOICE] wrap up"})
        return {}
    hooks.add_pre_tool_use(None, inject)

    await _run(monkeypatch, pool, READ_ONLY_BATCH,
               parallel_tool_calls_enabled=True, hooks=hooks,
               chat_messages_handle=handle)

    # The exact sequence, not a spot check. The parallel path appends its
    # tool messages in phase 3, AFTER the batch, so an inject that fired
    # during phase 1 lands *before* them without the reorder — which is the
    # same invalid shape, one direction over.
    roles = [m["role"] for m in handle]
    assert roles == ["user", "assistant", "tool", "tool", "user", "assistant"], roles
    tool_ids = [m["tool_call_id"] for m in handle if m["role"] == "tool"]
    assert tool_ids == ["c1", "c2"], "history must be wire order"


# ── behavioural pins on the replay seams (P13.0) ────────────────────────────
#
# These replaced source-substring checks on `run_query`: the flag and
# batch-size gate, "no TaskGroup", "the finally cancels outstanding tasks",
# "Phase 3 writes wire order" and "the mixed path runs the sequential loop".
# Each is now the behaviour the substring was standing in for.

from app.harness.tests import _replay as R  # noqa: E402


def _opts(**kw):
    kw.setdefault("parallel_tool_calls_enabled", True)
    return RunOptions(model="m", max_turns=4, tool_search_enabled=False, **kw)


async def test_a_batch_with_one_bash_runs_sequentially_and_a_read_only_batch_overlaps(
        monkeypatch):
    slow = {"c1": 0.03, "c2": 0.03, "c3": 0.03}

    # Read-only batch, flag on: all three calls are in flight together.
    pool = R.ReplayPool(delay_by_call_id=slow)
    R.install(monkeypatch, R.ReplayEngine([R.Step(tool_calls=[
        R.tool_call("c1", "Read", "a"), R.tool_call("c2", "Grep", "b"),
        R.tool_call("c3", "Glob", "c")]), R.Step(text="done")]), pool)
    await R.drive(_opts())
    assert pool.max_inflight == 3, pool.timeline

    # One Bash in the same batch: strictly one at a time, in wire order.
    pool = R.ReplayPool(delay_by_call_id=slow)
    R.install(monkeypatch, R.ReplayEngine([R.Step(tool_calls=[
        R.tool_call("c1", "Read", "a"), R.tool_call("c2", "Bash", "b", command="ls"),
        R.tool_call("c3", "Glob", "c")]), R.Step(text="done")]), pool)
    await R.drive(_opts())
    assert pool.max_inflight == 1, pool.timeline
    assert pool.timeline == ["start:c1", "done:c1", "start:c2", "done:c2",
                             "start:c3", "done:c3"]

    # Read-only batch with the flag OFF: sequential too.
    pool = R.ReplayPool(delay_by_call_id=slow)
    R.install(monkeypatch, R.ReplayEngine([R.Step(tool_calls=[
        R.tool_call("c1", "Read", "a"), R.tool_call("c2", "Grep", "b")]),
        R.Step(text="done")]), pool)
    await R.drive(_opts(parallel_tool_calls_enabled=False))
    assert pool.max_inflight == 1, pool.timeline


async def test_one_failing_tool_in_a_batch_does_not_cancel_its_siblings(monkeypatch):
    """A TaskGroup would cancel the slow sibling when the fast one raised; one
    tool failing is a tool_result, not a reason to abandon the batch."""
    pool = R.ReplayPool(answers={"c1": RuntimeError("boom")},
                        delay_by_call_id={"c1": 0.0, "c2": 0.05})
    R.install(monkeypatch, R.ReplayEngine([R.Step(tool_calls=[
        R.tool_call("c1", "Read", "a"), R.tool_call("c2", "Grep", "b")]),
        R.Step(text="done")]), pool)
    out = await R.drive(_opts())

    results = {e["call_id"]: e for e in R.of_type(out, "tool_result")}
    assert results["c1"]["is_error"] and "boom" in results["c1"]["content"]
    assert results["c2"]["is_error"] is False
    assert results["c2"]["content"] == "RESULT[Grep]"
    assert pool.cancelled == [] and pool.completed == ["c1", "c2"]
    assert R.of_type(out, "result")[0]["stop_reason"] == "stop"


async def test_an_exception_escaping_the_executor_is_contained_per_call(monkeypatch):
    """The batch's own guard, below `_execute_tool_call`'s: an exception that
    escapes the executor becomes that call's error result and nothing else."""
    pool = R.ReplayPool(delay_by_call_id={"c2": 0.05})
    real = L._execute_tool_call

    async def flaky(**kw):
        if kw["tc"]["id"] == "c1":
            raise RuntimeError("executor blew up")
        return await real(**kw)

    monkeypatch.setattr(L, "_execute_tool_call", flaky)
    R.install(monkeypatch, R.ReplayEngine([R.Step(tool_calls=[
        R.tool_call("c1", "Read", "a"), R.tool_call("c2", "Grep", "b")]),
        R.Step(text="done")]), pool)
    out = await R.drive(_opts())
    results = {e["call_id"]: e for e in R.of_type(out, "tool_result")}
    assert results["c1"]["is_error"] and "executor blew up" in results["c1"]["content"]
    assert results["c2"]["content"] == "RESULT[Grep]"
    assert pool.cancelled == []


async def test_closing_the_generator_mid_batch_cancels_outstanding_calls(monkeypatch):
    """A Stop click or a disconnect closes the generator; a tool still running
    for a turn nobody is reading must be cancelled, not left to finish."""
    pool = R.ReplayPool(delay_by_call_id={"c1": 0.0, "c2": 5.0, "c3": 5.0})
    R.install(monkeypatch, R.ReplayEngine([R.Step(tool_calls=[
        R.tool_call("c1", "Read", "a"), R.tool_call("c2", "Grep", "b"),
        R.tool_call("c3", "Glob", "c")]), R.Step(text="done")]), pool)

    out = await R.drive(_opts(),
                        on_event=lambda e: e["type"] == "tool_result")
    assert [e["call_id"] for e in R.of_type(out, "tool_result")] == ["c1"]
    for _ in range(5):          # let the cancellations be delivered
        await asyncio.sleep(0)
    assert sorted(pool.cancelled) == ["c2", "c3"], pool.timeline
    assert pool.completed == ["c1"]
    assert pool.inflight == 0


async def test_history_is_in_wire_order_when_the_last_call_lands_first(monkeypatch):
    pool = R.ReplayPool(delay_by_call_id={"c1": 0.06, "c2": 0.03, "c3": 0.0})
    engine = R.ReplayEngine([R.Step(tool_calls=[
        R.tool_call("c1", "Read", "a"), R.tool_call("c2", "Grep", "b"),
        R.tool_call("c3", "Glob", "c")]), R.Step(text="done")])
    R.install(monkeypatch, engine, pool)
    out = await R.drive(_opts())

    assert pool.completed == ["c3", "c2", "c1"], "the fixture did not reorder"
    # Yielded as they land...
    assert [e["call_id"] for e in R.of_type(out, "tool_result")] == ["c3", "c2", "c1"]
    # ...but replayed to the engine in the order the model made them.
    sent = engine.requests[1].messages
    asst = [m for m in sent if m["role"] == "assistant"][-1]
    assert [tc["id"] for tc in asst["tool_calls"]] == ["c1", "c2", "c3"]
    assert [m["tool_call_id"] for m in sent if m["role"] == "tool"] == ["c1", "c2", "c3"]


# ── P8: parallel read-only Task fan-out ─────────────────────────────────────
#
# A batch of fresh `Task` calls to a `parallel_safe` profile overlaps even with
# general parallel dispatch off; anything else in the batch, an unsafe profile,
# or a resume keeps it sequential. Each child still gets the parent's grant
# scope and deny list in `_meta` (D4), exactly as a sequential call would.

_TASK_TOOLS = {"Read": True, "Grep": True, "Bash": False, "Task": False}


def _fanout_opts(**kw):
    kw.setdefault("parallel_tool_calls_enabled", False)
    kw.setdefault("parallel_safe_task_profiles", frozenset({"read-only"}))
    return RunOptions(model="m", max_turns=4, tool_search_enabled=False, **kw)


async def _fanout(monkeypatch, calls, **opts):
    slow = {c["id"]: 0.03 for c in calls}
    pool = R.ReplayPool(_TASK_TOOLS, delay_by_call_id=slow)
    R.install(monkeypatch, R.ReplayEngine([R.Step(tool_calls=calls),
                                           R.Step(text="done")]), pool)
    events_out = await R.drive(_fanout_opts(**opts))
    return pool, events_out


async def test_a_parallel_safe_task_batch_overlaps_with_general_dispatch_off(monkeypatch):
    pool, events_out = await _fanout(monkeypatch, [
        R.tool_call("t1", "Task", "a", prompt="x", subagent_type="read-only"),
        R.tool_call("t2", "Task", "b", prompt="y", subagent_type="read-only"),
        R.tool_call("t3", "Task", "c", prompt="z", subagent_type="read-only"),
    ])
    assert pool.max_inflight == 3, pool.timeline
    # History still in wire order.
    tool_msgs = [e for e in R.of_type(events_out, "tool_result")]
    assert {e["call_id"] for e in tool_msgs} == {"t1", "t2", "t3"}


async def test_the_fanout_is_bounded_by_max_concurrency(monkeypatch):
    pool, _ = await _fanout(monkeypatch, [
        R.tool_call(f"t{i}", "Task", "a", prompt="x", subagent_type="read-only")
        for i in range(4)
    ], parallel_tool_calls_max_concurrency=2)
    assert pool.max_inflight == 2, pool.timeline


async def test_one_unsafe_profile_keeps_the_task_batch_sequential(monkeypatch):
    pool, _ = await _fanout(monkeypatch, [
        R.tool_call("t1", "Task", "a", prompt="x", subagent_type="read-only"),
        R.tool_call("t2", "Task", "b", prompt="y", subagent_type="general-purpose"),
    ])
    assert pool.max_inflight == 1, pool.timeline
    # No subagent_type means the tool's default, general-purpose: not safe.
    pool, _ = await _fanout(monkeypatch, [
        R.tool_call("t1", "Task", "a", prompt="x", subagent_type="read-only"),
        R.tool_call("t2", "Task", "b", prompt="y"),
    ])
    assert pool.max_inflight == 1, pool.timeline


async def test_a_resume_never_overlaps(monkeypatch):
    pool, _ = await _fanout(monkeypatch, [
        R.tool_call("t1", "Task", "a", prompt="x", subagent_type="read-only"),
        R.tool_call("t2", "Task", "b", prompt="more", subagent_type="read-only",
                    task_id="task-abc"),
    ])
    assert pool.max_inflight == 1, pool.timeline


async def test_a_task_mixed_with_a_read_waits_for_the_general_flag(monkeypatch):
    calls = [R.tool_call("t1", "Task", "a", prompt="x", subagent_type="read-only"),
             R.tool_call("r1", "Read", "b", path="/a")]
    pool, _ = await _fanout(monkeypatch, calls)
    assert pool.max_inflight == 1, pool.timeline
    pool, _ = await _fanout(monkeypatch, calls, parallel_tool_calls_enabled=True)
    assert pool.max_inflight == 2, pool.timeline


async def test_no_parallel_safe_profiles_means_no_fanout(monkeypatch):
    pool, _ = await _fanout(monkeypatch, [
        R.tool_call("t1", "Task", "a", prompt="x", subagent_type="read-only"),
        R.tool_call("t2", "Task", "b", prompt="y", subagent_type="read-only"),
    ], parallel_safe_task_profiles=frozenset())
    assert pool.max_inflight == 1, pool.timeline


async def test_every_child_of_a_fanout_inherits_the_grant_scope_and_deny_list(monkeypatch):
    """D4 holds on the parallel path: each Task call carries the same grant
    scope and this iteration's deny list in its dispatch kwargs, which
    `mcp_pool.call_tool` puts in `_meta` and `main.call_tool` binds for the
    child (`builtin_task.current_parent_grant_scope` / `_disallowed`)."""
    pool, _ = await _fanout(monkeypatch, [
        R.tool_call("t1", "Task", "a", prompt="x", subagent_type="read-only"),
        R.tool_call("t2", "Task", "b", prompt="y", subagent_type="read-only"),
    ], grant_scope="worker:deep-research",
        disallowed_tools=["email_send", "grant_create"])
    assert pool.max_inflight == 2, pool.timeline
    tasks = [c for c in pool.calls if c["name"] == "Task"]
    assert len(tasks) == 2
    for c in tasks:
        assert c["grant_scope"] == "worker:deep-research"
        assert set(c["disallowed_tools"]) >= {"email_send", "grant_create"}


def test_parallel_safe_task_qualification():
    safe = frozenset({"read-only"})
    ro = {"Read"}
    fresh = _tc("Task", "t1", {"subagent_type": "read-only", "prompt": "p"})
    resume = _tc("Task", "t2", {"subagent_type": "read-only", "task_id": "x"})
    default = _tc("Task", "t3", {"prompt": "p"})
    assert L._batch_is_read_only([fresh, fresh], ro, safe)
    assert L._is_task_fanout([fresh, fresh], safe)
    assert not L._batch_is_read_only([fresh, resume], ro, safe)
    assert not L._batch_is_read_only([fresh, default], ro, safe)
    assert not L._batch_is_read_only([fresh, fresh], ro)          # no profiles
    assert not L._is_task_fanout([fresh, _tc("Read", "r")], safe)
    assert L._batch_is_read_only([fresh, _tc("Read", "r")], ro, safe)
