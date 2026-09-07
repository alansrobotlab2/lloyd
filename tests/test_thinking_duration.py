"""How long the model spent thinking, as reported to the transcript.

The chat's collapsed thinking panel says "12.4s · 3,201 chars", and the
seconds come from the harness rather than from the browser: the panel has
to read the same on reload as it did live, and only the loop is in a
position to time the reasoning phase.

What is being measured is deliberately narrow — the first reasoning chunk
to the last one. The iteration's own clock (`assistant_message.duration_ms`)
is not a substitute: it also covers prefill, which on a 262k-token prompt
is minutes of a "thinking time" the model did not spend thinking, and the
answer generated after reasoning ends.
"""

import asyncio

import pytest

from app.harness import loop as loop_mod
from app.harness.options import RunOptions


class _FakePool:
    """Enough of MCPPool for a turn that never dispatches a tool."""

    discovered = [(
        "lloyd-mcp",
        [{
            "name": "Bash",
            "description": "run a command",
            "inputSchema": {"type": "object", "properties": {}},
        }],
    )]


def _chunk(**delta):
    finish = delta.pop("finish_reason", None)
    return {"choices": [{"delta": delta, "finish_reason": finish}]}


async def _collect(monkeypatch, script):
    """Drive one iteration of the loop over a scripted SSE stream.

    `script` is a list of (delay_before_chunk_seconds, chunk).
    """

    async def fake_stream_chat(**_kwargs):
        for delay, chunk in script:
            if delay:
                await asyncio.sleep(delay)
            yield chunk

    async def fake_build_pool(_options):
        return _FakePool()

    monkeypatch.setattr(loop_mod, "stream_chat", fake_stream_chat)
    monkeypatch.setattr(loop_mod, "_build_pool", fake_build_pool)

    events = []
    async for evt in loop_mod.run_query(
        [{"role": "user", "content": "hi"}],
        RunOptions(model="primary", tool_call_summaries=False),
    ):
        events.append(evt)
    return events


@pytest.mark.asyncio
async def test_thinking_done_reports_the_reasoning_phase(monkeypatch):
    events = await _collect(monkeypatch, [
        (0.0, _chunk(reasoning="weighing ")),
        (0.08, _chunk(reasoning="the options")),
        (0.0, _chunk(content="answer", finish_reason="stop")),
    ])

    done = [e for e in events if e["type"] == "thinking_done"]
    assert len(done) == 1
    assert done[0]["text"] == "weighing the options"
    assert done[0]["duration_ms"] >= 70


@pytest.mark.asyncio
async def test_duration_excludes_the_answer_that_follows(monkeypatch):
    """The gap after the last reasoning chunk belongs to the answer.

    Timing the whole iteration instead would report the model as having
    thought for as long as it took to write, which is the number the
    panel already shows in the turn stats.
    """
    events = await _collect(monkeypatch, [
        (0.0, _chunk(reasoning="short thought")),
        (0.30, _chunk(content="a long answer", finish_reason="stop")),
    ])

    done = [e for e in events if e["type"] == "thinking_done"]
    assert done[0]["duration_ms"] < 200


@pytest.mark.asyncio
async def test_no_thinking_means_no_event(monkeypatch):
    """An unreported duration must stay unreported.

    The panel renders chars alone when the field is missing, which is
    also how every session predating it reads.
    """
    events = await _collect(monkeypatch, [
        (0.0, _chunk(content="straight to it", finish_reason="stop")),
    ])

    assert not [e for e in events if e["type"] == "thinking_done"]
