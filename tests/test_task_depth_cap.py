"""The Task depth-1 cap is enforced, and its written rationale points at it (#1515).

`architecture/tools.md` §Subagents records WHY nesting stops at one level and
names the enforcement by file and line. This pins both layers so a widening
fails loudly:

- every child's `RunOptions.disallowed_tools` (and, for a parallel-safe
  profile, its per-iteration refresh) carries `Task` — the layer that holds on
  the real path, where a child's tool calls re-enter the aggregator over
  loopback and no contextvar survives;
- `MAX_TASK_DEPTH == 1` and the `_task_depth` contextvar refuses a same-context
  nested call — the backstop.

It also checks that the lines the doc cites still say what the doc claims, so
the citation cannot rot silently the way `judge.py:177` did (#1459).
"""
from __future__ import annotations

import ast
import asyncio
import json
import re
from pathlib import Path

import pytest

from agent_mcp import builtin_task

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "architecture" / "tools.md"


def _child_options(monkeypatch, profile_extra=None):
    captured = {}

    async def _fake_run_query(messages, options):
        captured["options"] = options
        yield {"type": "assistant_message", "text": "done", "thinking": "",
               "tool_calls": []}
        yield {"type": "result", "stop_reason": "stop", "num_turns": 1, "usage": {}}

    import app.harness.loop as loop_mod
    monkeypatch.setattr(loop_mod, "run_query", _fake_run_query)
    profile = {"system_prompt": "", "max_turns": 5, "disallowed_tools": [],
               "model": "primary", "base_url": ""}
    profile.update(profile_extra or {})
    monkeypatch.setattr(builtin_task, "_load_subagent_profile", lambda t: dict(profile))
    # Nothing upstream may already ban Task, or the assertion below would pass
    # without the cap's own append.
    import app.mcp_discovery as disc
    monkeypatch.setattr(disc, "_get_disallowed_tools", lambda: [])
    json.loads(asyncio.run(builtin_task._task({"prompt": "look", "description": "d"})))
    return captured["options"]


def test_the_cap_is_one():
    assert builtin_task.MAX_TASK_DEPTH == 1


def test_a_child_is_never_handed_task(monkeypatch):
    opts = _child_options(monkeypatch)
    assert "Task" in opts.disallowed_tools


def test_a_parallel_safe_childs_refresh_keeps_task_out(monkeypatch):
    # The loop reads `disallowed_tools_refresh` INSTEAD of the static list when
    # one is set, so the refresh must carry the ban too.
    opts = _child_options(monkeypatch, {"parallel_safe": True})
    assert "Task" in opts.disallowed_tools
    assert opts.disallowed_tools_refresh is not None
    assert "Task" in set(opts.disallowed_tools_refresh())


def test_a_same_context_nested_call_is_refused():
    token = builtin_task._task_depth.set(builtin_task.MAX_TASK_DEPTH)
    try:
        out = json.loads(asyncio.run(builtin_task._task({"prompt": "nested"})))
    finally:
        builtin_task._task_depth.reset(token)
    assert "recursion limit" in out["error"]


def test_the_harness_refuses_a_disallowed_task_at_dispatch():
    from app.harness.options import RunOptions
    from app.harness.loop import _pre_dispatch

    tc = {"id": "c1", "name": "Task", "_args_dict": {"prompt": "x"},
          "function": {"name": "Task", "arguments": "{}"}}
    # The disallowed gate sits before anything that reads the loaded set, so
    # none is needed to reach it.
    res = asyncio.run(_pre_dispatch(
        tc=tc, options=RunOptions(model="primary", disallowed_tools=["Task"]),
        session_id="test-depth-cap", loaded_set=None, runtime_disallowed={"Task"}))
    # _pre_dispatch returns a tool_result event for a refused call.
    assert isinstance(res, dict) and res.get("is_error"), res
    assert "disabled" in str(res.get("content", ""))


def test_tool_list_omits_a_disallowed_task():
    from app.harness.tool_schema import build_tool_list

    discovered = [("lloyd-mcp", [
        {"name": "Task", "description": "d", "inputSchema": {"type": "object", "properties": {}}},
        {"name": "Read", "description": "d", "inputSchema": {"type": "object", "properties": {}}},
    ])]
    names = {t["function"]["name"] for t in build_tool_list(discovered, {"Task"})}
    assert names == {"Read"}


# ── the doc's citations still point at the enforcement ────────────────────────


def _section() -> str:
    text = DOC.read_text()
    start = text.index("### Subagents (`Task`)")
    return text[start:text.index("\n## ", start)]


def _body(path: str, func: str | None) -> str:
    """Source of ``func`` in ``path`` (the module when None). Citations are
    pinned by the function that holds them, never by line position: a line
    number in a hot file like loop.py moves with every unrelated round (#1459,
    and #1510 moved this very citation 39 lines within a day), and a test that
    goes red on that fails rounds that did nothing wrong."""
    src = (ROOT / path).read_text()
    if func is None:
        return src
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func:
            return ast.get_source_segment(src, node) or ""
    raise AssertionError(f"{path} no longer defines {func}")


def test_the_rationale_carries_a_because_and_the_task_id_alternative():
    flat = " ".join(_section().split())
    assert re.search(r"depth 1.*\bbecause\b", flat), "the cap is stated without a reason"
    assert "task_id" in flat and "No per-level cost has been measured" in flat


@pytest.mark.parametrize("cite,func,needle", [
    ("agent_mcp/builtin_task.py:37", None, "MAX_TASK_DEPTH = 1"),
    ("app/harness/tool_schema.py:186", "build_tool_list", "if bare in disallowed"),
    ("app/harness/loop.py:2855", "_pre_dispatch", "for d in effective_disallowed"),
    ("agent_mcp/builtin_task.py:366-368", "_task", 'disallowed.append("Task")'),
    ("agent_mcp/builtin_task.py:282-285", "_task", "MAX_TASK_DEPTH"),
])
def test_each_cited_site_still_holds_what_the_doc_says(cite, func, needle):
    assert f"`{cite}`" in _section(), f"the doc no longer cites {cite}"
    path = cite.rsplit(":", 1)[0]
    where = func or "module level"
    assert needle in _body(path, func), (
        f"{needle!r} is no longer in {path} ({where}); the enforcement moved — update tools.md")
