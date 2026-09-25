"""A broken stream is `stream_error`, and is retried once while nothing was
dispatched (review 2026-09-24, D7).

Before this, a malformed SSE line mid-stream was finalized as
`finish_reason="stop"` with whatever half-parsed tool calls had accumulated —
and those were dispatched — while a stall, a dropped connection or a 5xx raised
out of the turn. Both are cheap to retry while no tool-call delta has arrived:
the prefix is cached and nothing ran. Driven through the real `run_query` on
the replay seams (`_replay.py`); the break is scripted after N chunks.
"""
from __future__ import annotations

from typing import Any

import httpx
import pytest

from app.harness import events as E
from app.harness import loop as L
from app.harness.errors import ParseError, StreamStalledError
from app.harness.options import RunOptions
from app.harness.tests import _replay as R


class _BreakingEngine(R.ReplayEngine):
    """A `ReplayEngine` whose step `i` raises `exc` after `after` chunks."""

    def __init__(self, steps: list[R.Step],
                 breaks: dict[int, tuple[int, BaseException]]):
        super().__init__(steps)
        self.breaks = breaks

    def __call__(self, **kwargs: Any):
        idx = len(self.requests)
        gen = super().__call__(**kwargs)
        if idx not in self.breaks:
            return gen
        after, exc = self.breaks[idx]
        return self._broken(gen, after, exc)

    @staticmethod
    async def _broken(gen, after: int, exc: BaseException):
        n = 0
        async for chunk in gen:
            if n >= after:
                raise exc
            n += 1
            yield chunk
        raise exc


def _opts(**kw) -> RunOptions:
    kw.setdefault("max_turns", 6)
    kw.setdefault("tool_search_enabled", False)
    kw.setdefault("stream_retry_backoff_s", 0.0)
    return RunOptions(model="m", **kw)


def _status_error(code: int) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "http://engine/v1/chat/completions")
    return httpx.HTTPStatusError(f"vLLM returned {code}: x", request=req,
                                 response=httpx.Response(code, request=req))


def _result(out: list[dict]) -> dict:
    [res] = R.of_type(out, "result")
    return res


@pytest.fixture
def logged(monkeypatch):
    rows: list[tuple[str, dict]] = []
    monkeypatch.setattr(L, "log_harness_event",
                        lambda sid, name, data, **kw: rows.append((name, data)))
    return rows


async def test_a_stall_before_any_tool_call_is_retried_once_with_the_same_messages(
        monkeypatch, logged):
    engine = _BreakingEngine(
        [R.Step(text="partial"), R.Step(text="done")],
        {0: (1, StreamStalledError(60.0, lines_seen=1))},
    )
    R.install(monkeypatch, engine, R.ReplayPool())
    out = await R.drive(_opts(system_prompt="S"))

    assert len(engine.requests) == 2
    assert engine.requests[0].messages == engine.requests[1].messages
    [retry] = R.of_type(out, "iteration_retry")
    assert retry["reason"] == "stream_stalled" and retry["attempt"] == 1
    res = _result(out)
    assert res["stop_reason"] == "stop"
    assert res["response_text"] == "done"
    assert res["num_turns"] == 1, "the retried attempt counted against max_turns"
    assert [n for n, _ in logged if n == "harness.stream_retried"] == [
        "harness.stream_retried"]


async def test_a_stall_after_a_tool_call_delta_raises_without_retry(monkeypatch):
    # Chunk 0 is the text, chunk 1 the tool call's name frame.
    engine = _BreakingEngine(
        [R.Step(text="checking", tool_calls=[R.tool_call("c1", "Read", "a")])],
        {0: (2, StreamStalledError(60.0, lines_seen=2))},
    )
    pool = R.ReplayPool()
    R.install(monkeypatch, engine, pool)
    with pytest.raises(StreamStalledError):
        await R.drive(_opts())
    assert len(engine.requests) == 1
    assert pool.calls == []


async def test_a_parse_error_mid_tool_call_ends_the_turn_as_stream_error_and_dispatches_nothing(
        monkeypatch):
    # Name frame plus the first argument fragment, then a malformed line.
    engine = _BreakingEngine(
        [R.Step(text="let me look",
                tool_calls=[R.tool_call("c1", "Bash", "b", command="rm -rf x")])],
        {0: (3, ParseError("malformed SSE chunk", raw='{"choices": [{'))},
    )
    pool = R.ReplayPool()
    R.install(monkeypatch, engine, pool)
    history: list[dict] = []
    out = await R.drive(_opts(chat_messages_handle=history))

    assert len(engine.requests) == 1, "a begun tool call is never re-requested"
    assert pool.calls == []
    assert R.of_type(out, "tool_call") == []
    [raw] = R.of_type(out, "stream_raw")
    assert raw["raw"] == '{"choices": [{'
    [asst] = R.of_type(out, "assistant_message")
    assert asst["tool_calls"] == [] and asst["text"] == "let me look"
    assert asst["finish_reason"] == "stream_error"
    assert _result(out)["stop_reason"] == "stream_error"
    last = history[-1]
    assert last["role"] == "assistant" and not last.get("tool_calls")
    assert last["content"] == "let me look"


async def test_a_parse_error_after_the_finish_frame_keeps_the_completion(monkeypatch):
    """Only a trailing line (the usage chunk) was lost: the completion is
    whole, and its tool call runs as it always did."""
    step = R.Step(tool_calls=[R.tool_call("c1", "Read", "a")])
    # text-less step: name frame, two fragments, finish frame = 4 chunks.
    engine = _BreakingEngine(
        [step, R.Step(text="done")],
        {0: (4, ParseError("malformed SSE chunk", raw="data: {"))},
    )
    pool = R.ReplayPool()
    R.install(monkeypatch, engine, pool)
    out = await R.drive(_opts())
    assert [c["call_id"] for c in pool.calls] == ["c1"]
    assert R.of_type(out, "iteration_retry") == []
    assert _result(out)["stop_reason"] == "stop"


async def test_a_5xx_is_retried_and_a_4xx_is_not(monkeypatch):
    engine = R.ReplayEngine([R.Step(raise_=_status_error(503)), R.Step(text="ok")])
    R.install(monkeypatch, engine, R.ReplayPool())
    out = await R.drive(_opts())
    assert len(engine.requests) == 2
    assert R.of_type(out, "iteration_retry")[0]["reason"] == "http_503"
    assert _result(out)["response_text"] == "ok"

    engine = R.ReplayEngine([R.Step(raise_=_status_error(422)), R.Step(text="ok")])
    R.install(monkeypatch, engine, R.ReplayPool())
    with pytest.raises(httpx.HTTPStatusError):
        await R.drive(_opts())
    assert len(engine.requests) == 1


async def test_a_second_failure_raises_as_before(monkeypatch):
    engine = _BreakingEngine(
        [R.Step(text="a"), R.Step(text="b"), R.Step(text="never")],
        {0: (1, StreamStalledError(60.0, lines_seen=1)),
         1: (1, StreamStalledError(60.0, lines_seen=1))},
    )
    R.install(monkeypatch, engine, R.ReplayPool())
    with pytest.raises(StreamStalledError):
        await R.drive(_opts())
    assert len(engine.requests) == 2


async def test_the_retry_event_reports_discarded_text(monkeypatch):
    """The counts are exactly what the discarded attempt streamed, so a
    consumer that trims by them ends with the retried attempt's text only."""
    engine = _BreakingEngine(
        [R.Step(reasoning="hmm let me", text="Wrong answer"),
         R.Step(reasoning="ok", text="Right")],
        {0: (3, httpx.ReadError("connection reset"))},
    )
    R.install(monkeypatch, engine, R.ReplayPool())
    out = await R.drive(_opts())

    [retry] = R.of_type(out, "iteration_retry")
    assert retry["reason"] == "transport"
    assert retry["discarded_text_chars"] == len("Wrong answer")
    assert retry["discarded_thinking_chars"] == len("hmm let me")
    text = thinking = ""
    for evt in out:
        if evt["type"] == "text_delta":
            text += evt["text"]
        elif evt["type"] == "thinking_delta":
            thinking += evt["text"]
        elif evt["type"] == "iteration_retry":
            text, thinking = E.trim_discarded(text, thinking, evt)
    assert (text, thinking) == ("Right", "ok")
    [asst] = R.of_type(out, "assistant_message")
    assert asst["text"] == "Right"


async def test_a_stream_retry_does_not_re_anchor_the_same_iteration(monkeypatch):
    """D9's head guard covers this retry too: one anchor per iteration."""
    seen: list[int] = []

    async def anchor(iteration: int):
        seen.append(iteration)
        return [{"role": "user", "content": f"<anchor {iteration}>"}]

    engine = _BreakingEngine(
        [R.Step(text="x"), R.Step(tool_calls=[R.tool_call("c1", "Read", "a")]),
         R.Step(text="done")],
        {0: (1, httpx.RemoteProtocolError("peer closed"))},
    )
    R.install(monkeypatch, engine, R.ReplayPool())
    await R.drive(_opts(state_anchor=anchor))
    assert seen == [1, 2]
    anchors = [m for m in engine.requests[-1].messages
               if str(m.get("content", "")).startswith("<anchor")]
    assert len(anchors) == 2


async def test_a_cancelled_turn_is_not_retried(monkeypatch):
    import asyncio

    cancel = asyncio.Event()
    cancel.set()
    engine = _BreakingEngine([R.Step(text="x"), R.Step(text="y")],
                             {0: (1, httpx.ReadError("gone"))})
    R.install(monkeypatch, engine, R.ReplayPool())
    opts = _opts(cancel_event=cancel)
    out = await R.drive(opts)
    # The head sees the cancel before any request goes out.
    assert engine.requests == [] and _result(out)["stop_reason"] == "cancelled"

    # Cancelled while the stream breaks: raised, never re-requested.
    cancel = asyncio.Event()
    engine = _BreakingEngine([R.Step(text="x"), R.Step(text="y")],
                             {0: (1, httpx.ReadError("gone"))})
    orig = engine._broken

    async def _cancel_then_break(gen, after, exc):
        async for chunk in orig(gen, after, exc):
            cancel.set()
            yield chunk
    engine._broken = _cancel_then_break
    R.install(monkeypatch, engine, R.ReplayPool())
    with pytest.raises(httpx.ReadError):
        await R.drive(_opts(cancel_event=cancel))
    assert len(engine.requests) == 1


def test_the_config_reaches_run_options(monkeypatch):
    from app import mcp_discovery as D
    from app.config import CONFIG

    monkeypatch.setitem(CONFIG, "harness", {
        **CONFIG.get("harness", {}),
        "stream_retry": {"max_attempts": 0, "backoff_seconds": 5}})
    kw = D._get_harness_kwargs()
    assert kw["stream_retry_max"] == 0
    assert kw["stream_retry_backoff_s"] == 5.0
    opts = RunOptions(model="m", **kw)
    assert opts.stream_retry_max == 0

    import yaml
    from app.paths import LLOYD_HOME
    tracked = yaml.safe_load((LLOYD_HOME / "config.yaml").read_text())
    assert tracked["harness"]["stream_retry"] == {
        "max_attempts": 1, "backoff_seconds": 2}
    defaults = RunOptions(model="m")
    assert (defaults.stream_retry_max, defaults.stream_retry_backoff_s) == (1, 2.0)


async def test_max_attempts_zero_turns_the_retry_off(monkeypatch):
    engine = _BreakingEngine([R.Step(text="x"), R.Step(text="y")],
                             {0: (1, StreamStalledError(60.0, lines_seen=1))})
    R.install(monkeypatch, engine, R.ReplayPool())
    with pytest.raises(StreamStalledError):
        await R.drive(_opts(stream_retry_max=0))
    assert len(engine.requests) == 1
