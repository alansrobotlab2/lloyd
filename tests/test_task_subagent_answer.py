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
