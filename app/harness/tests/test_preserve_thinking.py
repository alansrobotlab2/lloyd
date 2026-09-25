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


def test_both_reasoning_fields_are_sent(monkeypatch):
    """Regression guard: the two engines disagree about which key is real.

    vLLM 0.28 accepts both but populates the template only from
    `reasoning` (entrypoints/chat_utils.py:2000) — `reasoning_content`
    alone is silently dropped. llama.cpp applies the model's jinja
    directly and Qwen3.6's template reads `message.reasoning_content`
    (chat_template.jinja:91), never `reasoning`. Sending one field breaks
    preserved thinking on the other engine, in exactly the silent way
    this mechanism exists to prevent: the model sees prior turns in which
    it apparently thought nothing.

    This stopped being academic on 2026-09-06, when the secondary slot
    became llama.cpp serving Qwen3.6-35B-A3B. Dropping either key here
    should fail loudly rather than degrade one engine quietly.
    """
    script = _ThinkingScript([
        ("THOUGHT", "checking", BASH_TC),
        ("", "done", []),
    ])
    captured = _run(monkeypatch, script, preserve_thinking_iterations=6)

    asst = _assistants(captured[1])[0]
    assert asst.get("reasoning") == "THOUGHT", (
        "vLLM renders <think> only from `reasoning`"
    )
    assert asst.get("reasoning_content") == "THOUGHT", (
        "Qwen3.6's jinja template reads only `reasoning_content`"
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


def test_history_is_append_only_across_iterations(monkeypatch):
    """Every request must be a strict extension of the one before it.

    This is the prefix-cache contract, and it is the measurement rather
    than a design preference: vLLM's Automatic Prefix Caching keys KV
    blocks on the serialized token prefix, so dropping a token from
    message *k* invalidates every block after *k*, and the engine reports
    that as a collapsed `cached_tokens` instead of an error.

    Aggregating `assistant.stats.{cache_read, input_tokens}` over the
    sessions created 2026-09-09 (one row per session × iteration) the
    cached fraction ran 73-79% across iterations 3-7 and then fell to
    **24.5% at iteration 8**, with 41 individual sessions dropping ≥4
    points at exactly that index. `keep=6` predicts index 8 — the message
    dropped at iteration *k* is the one from *k-7*. Backlog #520 cost that
    cliff ≈10.2M re-prefilled tokens/day.

    `_ThinkingScript` snapshots each request with `dict(m)`, so a later
    in-place `pop` on the live buffer does NOT retroactively appear in an
    earlier snapshot. That is what makes this a test of what was actually
    sent rather than of what the buffer looks like afterwards.
    """
    script = _ThinkingScript([
        ("T1", "a", [{"id": "c1", "name": "Bash", "arguments": {}}]),
        ("T2", "b", [{"id": "c2", "name": "Bash", "arguments": {}}]),
        ("T3", "c", [{"id": "c3", "name": "Bash", "arguments": {}}]),
        ("T4", "d", []),
    ])
    # keep=1 prunes hardest, i.e. it is the worst case for the invariant.
    captured = _run(monkeypatch, script, preserve_thinking_iterations=1)

    assert len(captured) == 4, captured
    for i in range(1, len(captured)):
        prev, cur = captured[i - 1], captured[i]
        assert len(cur) > len(prev), f"request {i+1} is shorter than request {i}"
        assert cur[: len(prev)] == prev, (
            f"request {i+1} rewrote history already sent in request {i} — "
            "every KV block after the changed message gets re-prefilled"
        )


def test_window_is_bounded_at_the_turn_boundary(monkeypatch):
    """The keep-window still bounds what a turn carries; it just may not be
    enforced by editing history the engine has already cached.

    Enforcement moved to turn entry (`_cap_history_reasoning`), which is
    where history is rebuilt from the session JSON anyway and where the
    prefix is cold regardless — so windowing there costs nothing.
    """
    script = _ThinkingScript([
        ("T1", "a", [{"id": "c1", "name": "Bash", "arguments": {}}]),
        ("T2", "b", [{"id": "c2", "name": "Bash", "arguments": {}}]),
        ("T3", "c", [{"id": "c3", "name": "Bash", "arguments": {}}]),
        ("T4", "d", []),
    ])
    captured = _run(monkeypatch, script, preserve_thinking_iterations=2)

    # Intra-turn: nothing sent in an earlier iteration is taken back.
    kept = [m.get("reasoning") for m in _assistants(captured[-1])]
    assert kept == ["T1", "T2", "T3"], kept


def test_turn_boundary_cap_prunes_incoming_history():
    """Reasoning arriving from a previous turn is windowed on entry."""
    from app.harness.loop import _cap_history_reasoning

    msgs = [{"role": "assistant", "content": f"a{i}", "reasoning": f"R{i}"}
            for i in range(4)]
    _cap_history_reasoning(msgs, keep=2)
    assert [("reasoning" in m) for m in msgs] == [False, False, True, True]
    assert msgs[2]["reasoning"] == "R2"

    # A disabled knob leaves history alone.
    msgs2 = [{"role": "assistant", "content": "a", "reasoning": "R"}]
    _cap_history_reasoning(msgs2, keep=0)
    assert msgs2[0]["reasoning"] == "R"


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


# ───────────── a fabricated trace must not become the model's prior thought ──
#
# Backlog #1510. Preserved thinking is a two-edged mechanism: what it carries
# into the next request is whatever the last iteration emitted, and the primary
# sometimes emits reasoning about a request nobody made. The filing session
# (`sessions/20260925_001902_iv029b.json`) asked what `stream_chat` does, called
# Read, and produced a 1,223-character trace about "reproducing my complete
# previous thinking verbatim using the audit tool" — which preserved thinking
# would then have shown the model as its own prior thought for six more
# iterations, on the very turn where it had to work out what was being asked.
#
# So: withhold the trace, keep the row. These three tests make that precise —
# two runs of one scripted turn differing ONLY in the text of the trace, and one
# run where the same withheld trace is still recorded.

#: The shape of the trace that filed the item (trimmed; the markers are the
#: point — `no actual task`, `no substantive reasoning`, `reproduce my complete
#: previous thinking`, `audit tool`).
FABRICATED_TRACE = (
    "The user just sent system instructions setting me up as an expert software "
    "engineer helping solve problems. There's no actual task yet — just the setup "
    "message. There was no substantive reasoning to reproduce. Now the user is "
    "asking me to reproduce my complete previous thinking verbatim using the audit "
    "tool, and I should not fabricate a detailed reasoning trace that never existed."
)

#: Reasoning an ordinary turn writes on the same scripted shape: it names the
#: turn's own subject and carries none of the fabricated markers.
ORDINARY_TRACE = (
    "17*23: split it as 17*20 + 17*3 = 340 + 51 = 391, then verify with the "
    "calculator before answering."
)


def test_a_fabricated_trace_is_not_replayed_into_the_next_request(monkeypatch):
    """A trace describing a request that was never made never reaches the engine.

    Same scripted turn, same options, same script order as
    `test_an_ordinary_trace_is_still_replayed_verbatim`; the ONLY difference is
    the text of iteration 1's reasoning. Anything looser about this pair would
    let a passing test be about something other than the trace.
    """
    script = _ThinkingScript([
        (FABRICATED_TRACE, "17*23 is 391.", BASH_TC),
        ("", "done", []),
    ])
    captured = _run(monkeypatch, script, preserve_thinking_iterations=6)

    asst = _assistants(captured[1])
    assert asst, captured[1]
    assert "reasoning" not in asst[0], asst[0]
    assert "reasoning_content" not in asst[0], asst[0]
    # Withholding the trace is not suppressing the turn: what the model did and
    # what it said both still travel into history, so the next iteration still
    # knows it ran a tool and what it concluded from it.
    assert asst[0].get("tool_calls"), asst[0]
    assert asst[0].get("content") == "17*23 is 391.", asst[0]


def test_an_ordinary_trace_is_still_replayed_verbatim(monkeypatch):
    """Unflagged preserved thinking is exactly what it did before this check.

    The withhold branch must be a gate on one string and nothing else: it may
    not reformat, truncate, re-key or annotate the reasoning of a turn the check
    does not fire on. Asserted as both keys carrying the exact text because the
    two spellings are the whole mechanism, and either one alone breaks the other
    engine — see `test_both_reasoning_fields_are_sent`.
    """
    script = _ThinkingScript([
        (ORDINARY_TRACE, "checking", BASH_TC),
        ("", "done", []),
    ])
    captured = _run(monkeypatch, script, preserve_thinking_iterations=6)

    asst = _assistants(captured[1])[0]
    assert asst.get("reasoning") == ORDINARY_TRACE, asst
    assert asst.get("reasoning_content") == ORDINARY_TRACE, asst


def test_a_withheld_trace_is_still_recorded_as_a_thinking_row(monkeypatch):
    """The audit trail keeps what the model emitted even though the engine never will.

    Both halves in one run, because they are the same run's two outputs: the
    drained event stream still carries the `thinking_done` event whose text
    `app/routers/_messages_thinking.py::_build_thinking_entry` turns into the
    `role="thinking"` row (that step is pinned by
    `tests/test_thinking_trace.py::test_reasoning_never_rides_in_a_content_block`),
    while the second captured request carries no reasoning field at all. A change
    that "fixed" the leak by dropping the trace inside `stream_chat`, or by never
    yielding the event, would blind the distiller and Inner Voice to every
    reasoning phase — #1510 exists to make that evidence more trustworthy, not to
    remove it.
    """
    _patch_pool(monkeypatch)
    script = _ThinkingScript([
        (FABRICATED_TRACE, "17*23 is 391.", BASH_TC),
        ("", "done", []),
    ])
    monkeypatch.setattr("app.harness.loop.stream_chat", script)
    opts = RunOptions(
        model="primary", session_id="pt-test", tool_search_enabled=False,
        preserve_thinking_iterations=6,
    )
    emitted = asyncio.run(_drain([{"role": "user", "content": "go"}], opts))

    done = [e for e in emitted if e.get("type") == "thinking_done"]
    assert done, [e.get("type") for e in emitted]
    assert done[0]["text"] == FABRICATED_TRACE, done[0]
    assert "reasoning" not in _assistants(script.captured_messages[1])[0]
