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
    # No line number, unlike its four siblings: `loop.py` is the one file in this
    # list that changes every round, and the digits were what went stale (#3130).
    # The doc cites the file the same way, and `_body()` finds the function by AST,
    # so this key holds the enforcement rather than its address.
    ("app/harness/loop.py", "_pre_dispatch", "for d in effective_disallowed"),
    ("agent_mcp/builtin_task.py:366-368", "_task", 'disallowed.append("Task")'),
    ("agent_mcp/builtin_task.py:282-285", "_task", "MAX_TASK_DEPTH"),
])
def test_each_cited_site_still_holds_what_the_doc_says(cite, func, needle):
    assert f"`{cite}`" in _section(), f"the doc no longer cites {cite}"
    # The suffix is the line or range the doc happens to carry, and only the four
    # cold-file keys have one; a bare `app/harness/loop.py` is its own path. Nothing
    # here reads the digits, which is the whole point of #3130 — the number was the
    # part that rotted while this test stayed green.
    path = cite.rsplit(":", 1)[0]
    where = func or "module level"
    assert needle in _body(path, func), (
        f"{needle!r} is no longer in {path} ({where}); the enforcement moved — update tools.md")

#: A line-number citation of the harness loop, the kind that rots. `loop.py`
#: moves with every unrelated round, which is exactly why `_body()` above
#: resolves the function by AST instead of by line: the number is the part that
#: goes stale, the function is the part that holds the claim. Mirrors
#: `LINE_CITE` in `tests/test_testing_doc_claims.py`, whose comment is the same
#: diagnosis ("conftest.py line numbers are how every cite in this repo's docs
#: went stale").
LOOP_LINE_CITE = re.compile(r"loop\.py:\d+")

#: What §9 quotes as the refusal the dispatch-time layer returns, and the string
#: `_pre_dispatch` actually builds (`content=f"Tool {name!r} is disabled by
#: configuration."`). Checked against the source below rather than against a line,
#: so the sentence and the code it describes cannot drift apart again without a red
#: test. The doc's fragment is the code's tail, which is what the last assertion
#: below pins: a rewording of either has to move both.
DOC_QUOTED_REFUSAL = "disabled by configuration"
DISPATCH_REFUSAL = "is disabled by configuration"


def test_the_dispatch_citation_names_the_function_and_no_line_number():
    """§9 points at the dispatch-time layer by function, never by line (#3130).

    The page explains why nesting stops at one level, and this is the layer that
    holds on the real path, so a wrong pointer here is worse than a missing one:
    §9 carried a `loop.py` line number whose line was
    `cut_by_wall = str(args_dict.get("finish_reason", "")) == "length"` — seventy-
    five lines above the disallow check the sentence meant — and the pin below
    stayed green on it, because its own teeth are needle-in-function.
    """
    section = _section()
    assert LOOP_LINE_CITE.findall(section) == [], (
        "a loop.py line number in §9 is an address the next unrelated loop.py "
        f"round strands: {LOOP_LINE_CITE.findall(section)}")
    assert "`app/harness/loop.py`" in section, (
        "the dispatch-time layer must still name the file it lives in")
    # Flattened, the way the rationale node above reads the section: the page
    # wraps its prose, and a wrap is not part of what the sentence quotes.
    cited_layer = " ".join(section[section.index("`app/harness/loop.py`"):][:130].split())
    assert "_pre_dispatch" in cited_layer, (
        f"`_pre_dispatch` must be named beside the file, not elsewhere: {cited_layer!r}")
    assert DOC_QUOTED_REFUSAL in cited_layer, (
        f"§9 must keep quoting the refusal it claims: {cited_layer!r}")


def test_the_page_stops_narrating_the_stale_citation_as_still_open():
    """The review log at the foot of the page used to assert this bug was live.

    It sits outside the §9 section, so no pin read it — which is how a page can
    carry a fixed defect and keep promising to fix it. After #3130 the clause has
    to say the number is gone, not that the test forbids moving it.
    """
    text = DOC.read_text()
    assert LOOP_LINE_CITE.findall(text) == [], (
        f"the whole page is what a reader greps: {LOOP_LINE_CITE.findall(text)}")
    para = next((blk for blk in text.split("\n\n") if "(#3130)" in blk), "")
    assert para, "the page stopped recording #3130 rather than recording it closed"
    assert "Fixed since" in para, (
        "the block that names #3130 has to assert the citation was repaired — a "
        f"review log left in `Filed:` tense keeps promising this fix: {para[-260:]}")


def test_the_dispatch_layer_claim_is_pinned_by_its_string_not_its_line():
    """The claim §9 quotes has to be found in `_pre_dispatch`, wherever it sits.

    This is the half that de-numbering alone does not buy: the doc quoting a
    string is a claim about code, and the only honest way to pin a claim about
    code that moves is to read the code. If the refusal's wording is ever
    reworded, this fails and §9 gets updated with it — the drift that #3130 was
    filed from is a doc asserting a string the function no longer returns.
    """
    body = " ".join(_body("app/harness/loop.py", "_pre_dispatch").split())
    assert DISPATCH_REFUSAL in body, (
        f"§9 quotes {DOC_QUOTED_REFUSAL!r} as what dispatch returns, and "
        "_pre_dispatch no longer contains it — update the doc's sentence")
    # Read the quoted fragment out of the page rather than restating it: this is
    # the assertion that fails when the two drift, in either direction — the code
    # reworded, or the doc quoting a string dispatch never returns.
    flat = " ".join(_section().split())
    quoted = re.search(r'"([^"]+)"', flat[flat.index("`app/harness/loop.py`"):][:220])
    assert quoted, '§9 no longer quotes a refusal beside the loop.py cite'
    assert quoted.group(1) in body, (
        f"§9 quotes {quoted.group(1)!r} as what dispatch returns, and "
        "_pre_dispatch does not contain it — the citation is stranded again")
    assert "for d in effective_disallowed" in body, (
        "the disallow list §9 describes is assembled in this function; the "
        "parametrised pin above cites this function for it")
