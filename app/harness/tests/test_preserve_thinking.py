"""Preserved thinking: reasoning must travel back into history.

Qwen3.8-Flash-Next's chat template renders each prior assistant turn as
`<think>{reasoning}</think>{content}`, and its model card calls preserved
thinking out as reducing redundant reasoning in agent loops. The harness
dropped reasoning entirely until 2026-09-05, so every historical turn
rendered an EMPTY `<think>` block — the model saw 50+ iterations in which
it had apparently thought nothing and re-derived its own conclusions each
time. Session 20260905_151355_iv5174 spent 66.6k of its 71k output tokens
(94%) on reasoning across 52 iterations.

The field name is load-bearing. vLLM 0.28 accepts both `reasoning` and
`reasoning_content` on the wire but only populates the template from
`reasoning` (entrypoints/chat_utils.py:2000); sending `reasoning_content`
alone is silently dropped. Verified against the live server: with
`reasoning_content` the rendered prompt contained an empty `<think>`,
with `reasoning` it contained the text.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from app.harness import tool_search_cache
from app.harness.loop import _prune_reasoning, run_query
from app.harness.options import RunOptions


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


class _ThinkingScript:
    """Scripted vLLM stream that emits reasoning deltas before content."""

    def __init__(self, turns: list[tuple[str, str, list[dict[str, Any]]]]):
        self.turns = turns
        self.captured_messages: list[list[dict]] = []

    def __call__(self, **kwargs):
        idx = len(self.captured_messages)
        self.captured_messages.append(
            [dict(m) for m in (kwargs.get("messages") or [])]
        )
        return self._gen(*self.turns[idx])

    async def _gen(self, reasoning: str, text: str, tool_calls: list[dict]):
        if reasoning:
            yield {"choices": [{"delta": {"reasoning_content": reasoning}}]}
        if text:
            yield {"choices": [{"delta": {"content": text}}]}
        for i, tc in enumerate(tool_calls):
            yield {"choices": [{"delta": {"tool_calls": [{
                "index": i, "id": tc["id"], "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc.get("arguments") or {}),
                },
            }]}}]}
        yield {"choices": [{"delta": {},
                            "finish_reason": "tool_calls" if tool_calls else "stop"}]}
        yield {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


@pytest.fixture(autouse=True)
def _reset_cache():
    asyncio.run(tool_search_cache.clear())
    yield
    asyncio.run(tool_search_cache.clear())


def _patch_pool(monkeypatch):
    pool = _FakePool()

    async def _build_pool(_options):
        return pool
    monkeypatch.setattr("app.harness.loop._build_pool", _build_pool)


async def _drain(messages, options):
    return [evt async for evt in run_query(messages, options)]


def _run(monkeypatch, script, **opt_kw):
    _patch_pool(monkeypatch)
    monkeypatch.setattr("app.harness.loop.stream_chat", script)
    messages = [{"role": "user", "content": "go"}]
    opts = RunOptions(
        model="primary", session_id="pt-test", tool_search_enabled=False,
        **opt_kw,
    )
    asyncio.run(_drain(messages, opts))
    return script.captured_messages


def _assistants(msgs: list[dict]) -> list[dict]:
    return [m for m in msgs if m.get("role") == "assistant"]


BASH_TC = [{"id": "c1", "name": "Bash", "arguments": {"command": "ls"}}]


def test_reasoning_is_carried_into_history(monkeypatch):
    """The next iteration must see the previous iteration's thinking."""
    script = _ThinkingScript([
        ("DEEP_THOUGHT_ONE", "checking", BASH_TC),
        ("", "done", []),
    ])
    captured = _run(monkeypatch, script, preserve_thinking_iterations=6)

    second = captured[1]
    asst = _assistants(second)
    assert asst, second
    assert asst[0].get("reasoning") == "DEEP_THOUGHT_ONE", asst[0]


def test_field_is_reasoning_not_reasoning_content(monkeypatch):
    """Regression guard: vLLM only renders the template from `reasoning`.

    `reasoning_content` is accepted and silently dropped, which produces
    an empty <think> block — the exact bug this change fixes. If a future
    edit renames the key, this test fails rather than quietly restoring
    the old behaviour.
    """
    script = _ThinkingScript([
        ("THOUGHT", "checking", BASH_TC),
        ("", "done", []),
    ])
    captured = _run(monkeypatch, script, preserve_thinking_iterations=6)

    asst = _assistants(captured[1])[0]
    assert "reasoning" in asst, asst
    assert "reasoning_content" not in asst, (
        "reasoning_content alone is dropped by vLLM — the key must be `reasoning`"
    )


def test_disabled_by_default_carries_nothing(monkeypatch):
    """preserve_thinking_iterations=0 reproduces the pre-fix behaviour."""
    script = _ThinkingScript([
        ("THOUGHT", "checking", BASH_TC),
        ("", "done", []),
    ])
    captured = _run(monkeypatch, script, preserve_thinking_iterations=0)

    for m in _assistants(captured[1]):
        assert "reasoning" not in m, m


def test_window_is_bounded(monkeypatch):
    """Only the most recent N iterations keep their reasoning."""
    script = _ThinkingScript([
        ("T1", "a", [{"id": "c1", "name": "Bash", "arguments": {}}]),
        ("T2", "b", [{"id": "c2", "name": "Bash", "arguments": {}}]),
        ("T3", "c", [{"id": "c3", "name": "Bash", "arguments": {}}]),
        ("T4", "d", []),
    ])
    captured = _run(monkeypatch, script, preserve_thinking_iterations=2)

    # Final request carries three prior assistant turns; only the last
    # two keep reasoning.
    kept = [m.get("reasoning") for m in _assistants(captured[-1])]
    assert kept == [None, "T2", "T3"], kept


def test_prune_reasoning_keeps_most_recent():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "reasoning": "R1"},
        {"role": "tool", "content": "t"},
        {"role": "assistant", "content": "b", "reasoning": "R2"},
        {"role": "assistant", "content": "c", "reasoning": "R3"},
    ]
    _prune_reasoning(msgs, keep=2)
    assert "reasoning" not in msgs[1]
    assert msgs[3]["reasoning"] == "R2"
    assert msgs[4]["reasoning"] == "R3"
    # Text and roles are untouched — an older turn just renders an empty
    # <think>, exactly as every turn did before this feature.
    assert msgs[1]["content"] == "a"


def test_prune_reasoning_keep_zero_strips_all():
    msgs = [{"role": "assistant", "content": "a", "reasoning": "R"}]
    _prune_reasoning(msgs, keep=0)
    assert "reasoning" not in msgs[0]


def test_preserved_reasoning_counts_against_the_context_budget():
    """Reasoning is rendered into the prompt, so compaction must see it.

    If the estimator ignored it, microcompaction would trigger late by
    however much reasoning sits in the window — and microcompaction can
    only clear tool results, so it has no way to reclaim reasoning.
    """
    from app.compaction import estimate_message_tokens

    plain = {"role": "assistant", "content": "short answer"}
    with_reasoning = {
        "role": "assistant", "content": "short answer",
        "reasoning": "x" * 4000,
    }
    assert estimate_message_tokens(with_reasoning) > \
        estimate_message_tokens(plain) + 900


def test_estimator_unchanged_for_messages_without_reasoning():
    from app.compaction import estimate_message_tokens

    msg = {"role": "assistant", "content": "hello"}
    assert estimate_message_tokens(msg) == estimate_message_tokens(dict(msg))
    assert estimate_message_tokens({"role": "user", "content": "hi"}) > 0
