"""A Task subagent that never answered must not look like it did.

`_task` used to accumulate every `text_delta` across every iteration, so
a subagent that burned its whole turn budget mid-investigation returned
its OPENING line as though that were the finished work. On session
20260905_151355_iv5174 a review subagent ran 231s and 19 tool calls and
came back with:

    {"response": "\\n\\nI'll start by loading the review skill and
      getting oriented in the repo.\\n\\n"}

The caller had no way to tell that from a real answer. The primary
noticed only because the text was obviously a preamble, then redid four
minutes of work itself.

The answer is now the TERMINAL iteration's text — the one that stopped
without calling a tool — and a subagent that dispatched tools but never
produced a closing message returns an explicit error.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_mcp import builtin_task


def _events(*, iterations, stop_reason="stop", num_turns=None):
    """Build a run_query event stream from (text, tool_names, thinking) rows."""
    evts = []
    for text, tools, thinking in iterations:
        for name in tools:
            evts.append({"type": "tool_call", "id": "x", "name": name, "input": {}})
        evts.append({
            "type": "assistant_message",
            "text": text,
            "thinking": thinking,
            "tool_calls": [{"function": {"name": n}} for n in tools],
        })
    evts.append({
        "type": "result",
        "stop_reason": stop_reason,
        "num_turns": num_turns if num_turns is not None else len(iterations),
        "usage": {},
    })
    return evts


def _run(monkeypatch, evts, max_turns=20):
    async def _fake_run_query(messages, options):
        for e in evts:
            yield e

    import app.harness.loop as loop_mod
    monkeypatch.setattr(loop_mod, "run_query", _fake_run_query)
    monkeypatch.setattr(
        builtin_task, "_load_subagent_profile",
        lambda t: {"system_prompt": "", "max_turns": max_turns,
                   "disallowed_tools": [], "model": "primary", "base_url": ""},
    )
    out = asyncio.run(builtin_task._task(
        {"prompt": "review the thing", "description": "d"}
    ))
    return json.loads(out)


def test_answer_is_the_terminal_iteration_not_the_preamble(monkeypatch):
    res = _run(monkeypatch, _events(iterations=[
        ("I'll start by getting oriented.", ["Read"], ""),
        ("", ["Read"], ""),
        ("The defect is a race in _apply_lever.", [], ""),
    ]))
    assert res["response"] == "The defect is a race in _apply_lever."
    assert "I'll start by" not in res["response"]


def test_tools_but_no_final_answer_is_an_error(monkeypatch):
    """The exact 20260905_151355_iv5174 shape."""
    res = _run(monkeypatch, _events(
        iterations=[("I'll start by getting oriented.", ["Read"], "")] * 19,
        stop_reason="max_turns", num_turns=21,
    ), max_turns=20)

    assert "error" in res, res
    assert "no final answer" in res["error"]
    assert "NOT completed" in res["error"]
    assert res["stop_reason"] == "max_turns"
    assert res["max_turns"] == 20
    assert len(res["tools_used"]) == 19
    # The preamble is still surfaced, but labelled as partial — never as
    # the response.
    assert "response" not in res
    assert "I'll start by" in res["partial_text"]


def test_budget_exhaustion_is_named_in_the_error(monkeypatch):
    res = _run(monkeypatch, _events(
        iterations=[("", ["Read"], "")] * 5,
        stop_reason="max_turns", num_turns=21,
    ), max_turns=20)
    assert "20-turn budget" in res["error"], res["error"]


def test_answer_lost_to_the_reasoning_channel_is_diagnosed(monkeypatch):
    """Empty content with reasoning present is a distinct failure mode."""
    res = _run(monkeypatch, _events(iterations=[
        ("", ["Read"], ""),
        ("", [], "I have concluded the answer but never wrote it out."),
    ]))
    assert "error" in res, res
    assert "reasoning but no final message" in res["error"]


def test_truncation_is_surfaced_even_with_text(monkeypatch):
    res = _run(monkeypatch, _events(iterations=[
        ("", ["Read"], ""),
        ("Partial findings so far.", [], ""),
    ], stop_reason="max_turns", num_turns=21), max_turns=20)

    assert res["response"] == "Partial findings so far."
    assert res["truncated"] is True
    assert res["stop_reason"] == "max_turns"


def test_clean_run_has_no_truncation_noise(monkeypatch):
    res = _run(monkeypatch, _events(iterations=[
        ("", ["Read"], ""),
        ("All good.", [], ""),
    ]))
    assert res["response"] == "All good."
    assert "truncated" not in res
    assert "stop_reason" not in res
    assert res["tools_used"] == ["Read"]
    assert res["description"] == "d"


def test_no_tools_and_no_text_is_not_reported_as_success(monkeypatch):
    """A subagent that did nothing at all still returns an empty response.

    Without tool calls there is no work to misrepresent, so this stays a
    plain (empty) answer rather than an error — the caller can see it.
    """
    res = _run(monkeypatch, _events(iterations=[("", [], "")]))
    assert res["response"] == ""
    assert "error" not in res


def test_recursion_cap_still_holds(monkeypatch):
    token = builtin_task._task_depth.set(builtin_task.MAX_TASK_DEPTH)
    try:
        out = asyncio.run(builtin_task._task({"prompt": "nested"}))
    finally:
        builtin_task._task_depth.reset(token)
    assert "recursion limit" in json.loads(out)["error"]


def test_a_wrapped_up_budget_run_returns_its_summary_marked_truncated(monkeypatch):
    """P6b end to end: the REAL loop, a child that never stops calling tools.

    At its budget the loop asks once, toollessly, where the work stands; that
    answer is the child's terminal iteration, so it becomes `response` with no
    change in `_task` — and `truncated` still says the budget ran out, because
    `stop_reason` stays `max_turns`.
    """
    from app import mcp_discovery
    from app.harness.tests._replay import (
        ReplayEngine, ReplayPool, Step, install, tool_call,
    )

    url = "http://127.0.0.1:8096"
    engine = ReplayEngine([
        Step(tool_calls=[tool_call("c1", "Read", path="/a")]),
        Step(tool_calls=[tool_call("c2", "Read", path="/b")]),
        Step(text="Read /a and /b; the race is in _apply_lever, not yet fixed."),
    ])
    install(monkeypatch, engine, ReplayPool())
    monkeypatch.setattr(mcp_discovery, "max_turns_wrapup_kwargs", lambda: {
        "max_turns_wrapup": True, "max_turns_wrapup_base_urls": (url,)})
    monkeypatch.setattr(
        builtin_task, "_load_subagent_profile",
        lambda t: {"system_prompt": "", "max_turns": 2,
                   "disallowed_tools": [], "model": "primary", "base_url": url},
    )
    res = json.loads(asyncio.run(builtin_task._task(
        {"prompt": "review the thing", "description": "d"})))

    assert "error" not in res, res
    assert res["response"].startswith("Read /a and /b; the race")
    assert res["truncated"] is True
    assert res["stop_reason"] == "max_turns"
    assert res["tools_used"] == ["Read", "Read"]
    assert engine.requests[-1].kwargs["tool_choice"] == "none"


# ── P8: subagent defaults (review 2026-09-24) ───────────────────────────────

def _real_loop_task(monkeypatch, steps, *, profile=None, args=None, tools=None):
    """Run `_task` through the REAL loop against a scripted engine and pool."""
    from app.harness.tests._replay import ReplayEngine, ReplayPool, install

    engine = ReplayEngine(steps)
    pool = ReplayPool(tools)
    install(monkeypatch, engine, pool)
    prof = {"system_prompt": "", "max_turns": 8, "disallowed_tools": [],
            "model": "primary", "base_url": "http://127.0.0.1:8096"}
    prof.update(profile or {})
    monkeypatch.setattr(builtin_task, "_load_subagent_profile", lambda t: dict(prof))
    res = json.loads(asyncio.run(builtin_task._task(
        {"prompt": "review the thing", **(args or {})})))
    return res, engine, pool


def test_an_empty_profile_prompt_gets_the_default_subagent_prompt(monkeypatch):
    from app.harness.tests._replay import Step

    res, engine, _ = _real_loop_task(monkeypatch, [Step(text="The answer.")])
    assert res["response"] == "The answer."
    system = engine.requests[0].messages[0]
    assert system["role"] == "system"
    assert builtin_task._DEFAULT_SUBAGENT_PROMPT in system["content"]
    assert "returned to that agent verbatim" in builtin_task._DEFAULT_SUBAGENT_PROMPT

    # A profile that names its own prompt keeps it, and gets no default.
    _res, engine, _ = _real_loop_task(
        monkeypatch, [Step(text="ok")], profile={"system_prompt": "PROFILE PROMPT"})
    content = engine.requests[0].messages[0]["content"]
    assert "PROFILE PROMPT" in content
    assert builtin_task._DEFAULT_SUBAGENT_PROMPT not in content


def test_the_iteration_anchor_is_sized_to_the_profiles_budget(monkeypatch):
    """A 4-iteration child hears its cap at 75% (iteration 3), in the chat
    path's wording, appended to the conversation — never in position 0."""
    from app.harness.tests._replay import Step, tool_call

    steps = [Step(tool_calls=[tool_call(f"c{i}", "Read", path=f"/{i}")])
             for i in range(3)] + [Step(text="Done: read three files.")]
    res, engine, _ = _real_loop_task(monkeypatch, steps, profile={"max_turns": 4})
    assert res["response"] == "Done: read three files."
    anchors = [m for r in engine.requests for m in r.messages
               if m.get("role") == "user"
               and "<budget>Iteration" in str(m.get("content"))]
    assert anchors, "the child was never told its budget"
    assert "Iteration 3 of 4" in anchors[0]["content"]
    assert "<budget>" not in engine.requests[-1].messages[0]["content"]


def test_final_schema_round_trips_to_structured(monkeypatch):
    from app.harness import finalizer
    from app.harness.tests._replay import Step

    seen = {}

    async def fake_finalizer(**kw):
        seen["schema"] = kw["schema"]
        return {"verdict": "race"}, "", {}

    monkeypatch.setattr(finalizer, "run_finalizer", fake_finalizer)
    schema = {"type": "object", "properties": {"verdict": {"type": "string"}},
              "required": ["verdict"]}
    res, _engine, _ = _real_loop_task(
        monkeypatch, [Step(text="It is a race.")], args={"final_schema": schema})
    assert seen["schema"] == schema
    assert res["response"] == "It is a race."
    assert res["structured"] == {"verdict": "race"}
    assert res["structured_error"] == ""

    # No schema asked for: no structured keys, and no finalizer call.
    seen.clear()
    res, _engine, _ = _real_loop_task(monkeypatch, [Step(text="plain")])
    assert "structured" not in res and not seen

    # A schema that is not an object schema is refused on the result, not
    # by failing the Task — the prose answer still stands.
    res, _engine, _ = _real_loop_task(
        monkeypatch, [Step(text="plain")], args={"final_schema": {"type": "string"}})
    assert res["response"] == "plain"
    assert res["structured"] is None
    assert "type" in res["structured_error"]
    assert not seen

    # The finalizer is skipped after a budget death, and says so.
    from app.harness.tests._replay import tool_call
    res, _engine, _ = _real_loop_task(
        monkeypatch,
        [Step(tool_calls=[tool_call("c1", "Read")]), Step(text="Partial.")],
        profile={"max_turns": 1}, args={"final_schema": schema})
    assert res["stop_reason"] == "max_turns" and res["structured"] is None
    assert res["structured_error"] == "skipped: stop_reason=max_turns"


def test_final_schema_is_advertised_on_the_task_input_schema():
    tool = asyncio.run(builtin_task.list_tools())[0]
    prop = tool.input_schema["properties"]["final_schema"]
    assert prop["type"] == "object"
    assert "final_schema" not in tool.input_schema["required"]


def test_a_parallel_safe_childs_tool_set_is_read_only(monkeypatch):
    """The fan-out is safe because of the child, not because of its prompt:
    a parallel-safe profile's child is advertised only readOnlyHint tools."""
    from app.harness.tests._replay import Step

    tools = {"Read": True, "Grep": True, "vault_write": False, "Bash": False,
             "email_send": False, "Edit": False, "Write": False}
    _res, engine, _ = _real_loop_task(
        monkeypatch, [Step(text="ok")], tools=tools,
        profile={"parallel_safe": True})
    advertised = {t["function"]["name"] for t in (engine.requests[0].tools or [])}
    assert {"Read", "Grep"} <= advertised
    assert not advertised & {"vault_write", "Bash", "email_send", "Edit", "Write"}, advertised

    # The same profile without the flag keeps its ordinary tool set.
    _res, engine, _ = _real_loop_task(
        monkeypatch, [Step(text="ok")], tools=tools, profile={"parallel_safe": False})
    advertised = {t["function"]["name"] for t in (engine.requests[0].tools or [])}
    assert "vault_write" in advertised
