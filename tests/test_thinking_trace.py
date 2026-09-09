"""One row per reasoning phase, on the chat timeline.

The harness has always emitted one `thinking_done` per agent-loop
iteration. The router threw almost all of them away: a single
`accumulated_thinking` buffer held the current phase, each new phase
*replaced* it, and it only reached disk on an iteration that produced both
tool calls and non-empty text. A tool-only iteration — the common shape —
never flushed, so a forty-iteration turn persisted exactly one reasoning
phase, the last one, and the chat could only ever show that.

Each phase is now its own `role="thinking"` entry. Two properties of that
shape carry the design and are pinned here; the third — that no transcript
generated from the logs reproduces the reasoning — is in
`test_thinking_trace_transcripts.py`.
"""

from __future__ import annotations

import asyncio

import pytest

from app.harness import loop as loop_mod
from app.harness.options import RunOptions
from app.routers._messages_thinking import _build_thinking_entry


class _Turn:
    turn_id = "T1"


# --------------------------------------------------------------- shape


def test_reasoning_never_rides_in_a_content_block():
    """The one property every transcript producer depends on.

    They all dispatch on role and read `content` blocks of type "text".
    Reasoning in a content block would leak into every one of them; in a
    sibling key it is invisible to all of them without any new filtering.
    """
    entry = _build_thinking_entry(_Turn(), "weighing the options", 4200, 0, 1, "2026-09-08T10:00:00")

    assert entry["role"] == "thinking"
    assert entry["content"] == []
    assert entry["reasoning"] == "weighing the options"
    assert entry["reasoning_ms"] == 4200
    # Nothing anywhere in the content channel.
    assert "weighing" not in str(entry["content"])


def test_phases_are_distinct_rows_in_order():
    """Ids order phases within a turn and never collide.

    The bug this replaces was one phase overwriting another; ids that
    collided would reintroduce it one layer down.
    """
    rows = [
        _build_thinking_entry(_Turn(), f"phase {i}", 100 * i, i, i + 1, "2026-09-08T10:00:00")
        for i in range(4)
    ]
    assert len({r["id"] for r in rows}) == 4
    assert [r["thinking"]["iteration"] for r in rows] == [1, 2, 3, 4]
    assert [r["reasoning"] for r in rows] == ["phase 0", "phase 1", "phase 2", "phase 3"]


def test_char_count_is_recorded_for_the_collapsed_header():
    entry = _build_thinking_entry(_Turn(), "x" * 3201, 11800, 2, 7, "2026-09-08T10:00:00")
    assert entry["thinking"]["chars"] == 3201
    assert entry["thinking"]["turn_id"] == "T1"


# ------------------------------------------------- never re-enters the prompt


def test_compaction_drops_thinking_rows(tmp_path):
    """A thinking row is display state, not conversation.

    `load_and_compact_session` keeps only the conversation roles, so these
    rows never reach `_prepare_messages_for_harness` and never re-enter the
    prompt. Preserved thinking (`loop._assistant_message_for_history`) is a
    separate, in-flight mechanism and is not affected.

    Driven through the real loader rather than a restatement of its
    filter — a second copy of the rule is exactly how the two would come
    to disagree.
    """
    import json

    from app.compaction import load_and_compact_session

    session = tmp_path / "s1.json"
    session.write_text(json.dumps({"messages": [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        _build_thinking_entry(_Turn(), "secret deliberation", 100, 0, 1, "2026-09-08T10:00:00"),
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
    ]}))

    out = asyncio.run(load_and_compact_session(session))
    history = out["history"]

    assert [m["role"] for m in history] == ["user", "assistant"]
    assert "secret deliberation" not in json.dumps(history)


# ------------------------------------------------ one phase per iteration


def _chunk(**delta):
    finish = delta.pop("finish_reason", None)
    return {"choices": [{"delta": delta, "finish_reason": finish}]}


class _FakePool:
    discovered = [(
        "lloyd-mcp",
        [{
            "name": "Bash",
            "description": "run a command",
            "inputSchema": {"type": "object", "properties": {}},
        }],
    )]

    async def call_tool(self, name, args, **kw):
        return {"content": "ok", "is_error": False}


async def _run(monkeypatch, scripts):
    """Drive the loop over one scripted SSE stream per iteration."""
    remaining = list(scripts)

    async def fake_stream_chat(**_kwargs):
        for chunk in remaining.pop(0):
            yield chunk

    async def fake_build_pool(_options):
        return _FakePool()

    monkeypatch.setattr(loop_mod, "stream_chat", fake_stream_chat)
    monkeypatch.setattr(loop_mod, "_build_pool", fake_build_pool)

    out = []
    async for evt in loop_mod.run_query(
        [{"role": "user", "content": "hi"}],
        RunOptions(model="primary", tool_call_summaries=False),
    ):
        out.append(evt)
    return out


@pytest.mark.asyncio
async def test_a_tool_using_turn_reports_every_phase(monkeypatch):
    """The regression, at the source.

    Iteration 1 reasons and calls a tool without writing a word; iteration
    2 reasons again and answers. Both phases are reported. The router used
    to keep only the second — the first never met the "tool calls AND
    text" condition that flushed the buffer, and was overwritten.
    """
    events = await _run(monkeypatch, [
        [
            _chunk(reasoning="first I should look at the disk"),
            _chunk(tool_calls=[{
                "index": 0, "id": "c1", "type": "function",
                "function": {"name": "Bash", "arguments": "{}"},
            }]),
            _chunk(finish_reason="tool_calls"),
        ],
        [
            _chunk(reasoning="the disk is fine, so I can answer"),
            _chunk(content="All good.", finish_reason="stop"),
        ],
    ])

    phases = [e for e in events if e["type"] == "thinking_done"]
    assert len(phases) == 2, "one reasoning phase per iteration"
    assert phases[0]["text"] == "first I should look at the disk"
    assert phases[1]["text"] == "the disk is fine, so I can answer"
    # The first phase produced no text at all — the exact shape whose
    # thinking the old buffer discarded.
    assistant = [e for e in events if e["type"] == "assistant_message"]
    assert assistant[0]["text"] == ""
    assert assistant[0]["tool_calls"]


# ------------------------------------------------------------- kill switch


def test_config_default_is_on():
    """A display-only change with no path into the model's context."""
    import yaml
    from app.paths import LLOYD_HOME

    cfg = yaml.safe_load((LLOYD_HOME / "config.yaml").read_text())
    assert cfg["harness"]["thinking_trace"]["enabled"] is True
