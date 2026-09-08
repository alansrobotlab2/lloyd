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

import pytest

from app.harness import events, loop as L
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


def test_read_only_names_comes_from_the_annotations_not_a_local_list():
    src = inspect.getsource(L._read_only_names)
    assert 'ann.get("readOnlyHint")' in src
    assert "pool.discovered" in src


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

def test_the_loop_gates_on_the_flag_and_the_batch_size():
    src = inspect.getsource(L.run_query)
    assert 'getattr(options, "parallel_tool_calls_enabled", False)' in src
    assert "len(tool_calls_committed) > 1" in src
    assert "_batch_is_read_only(tool_calls_committed, _read_only_names(pool))" in src


def test_the_loop_does_not_use_a_taskgroup():
    """A TaskGroup cancels its siblings on the first exception; one tool
    failing is a tool_result, not a reason to abandon the batch."""
    src = inspect.getsource(L.run_query)
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert "TaskGroup" not in code


def test_the_loop_cancels_outstanding_tasks_if_the_generator_closes():
    src = inspect.getsource(L.run_query)
    block = src.split("run_parallel = (")[1]
    assert "finally:" in block and "t.cancel()" in block


def test_the_loop_writes_history_in_wire_order():
    src = inspect.getsource(L.run_query)
    block = src.split("# Phase 3")[1]
    assert "for tc in tool_calls_committed:" in block
    assert 'results.get(tc["id"])' in block


def test_the_mixed_path_still_runs_the_original_sequential_loop():
    src = inspect.getsource(L.run_query)
    assert "if tool_calls_committed_done else tool_calls_committed" in src


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
