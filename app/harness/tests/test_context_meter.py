"""The shared context meter, and the relief ladder it drives.

Three autocode rounds died at the context wall on 2026-09-11 with finished
work sitting uncommitted in a worktree. What every one of them lacked was a
number: the engine reports the prompt size, the harness knew it, and nothing
carried it to the three places that needed it — the relief ladder, the
`<context>` anchor and the Inner Voice observer.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from app.harness import tool_search_cache
from app.harness.context_meter import ContextMeter, context_window_for
from app.harness.loop import _parse_failure_text, _relieve_context, run_query
from app.harness.options import RunOptions


# ---------------------------------------------------------------------------
# the meter itself
# ---------------------------------------------------------------------------

def test_an_unmeasured_meter_reports_nothing_and_blocks_nothing():
    """Iteration 1 of every turn is unmeasured. It must read as healthy.

    Every reader fails open on this: an anchor that fired here would be
    guessing, and relief that ran here would prune a turn that has not been
    shown to need it.
    """
    m = ContextMeter(262_144)
    assert m.measured is False
    assert m.used == 0
    assert m.fraction == 0.0
    assert m.threshold_fraction == 0.0
    # Headroom is the whole window, not zero: `headroom < floor` is the
    # comparison every caller makes, and an unmeasured meter must not
    # satisfy it.
    assert m.headroom == m.window


def test_used_is_the_reported_prompt_plus_what_was_appended_after_it():
    m = ContextMeter(262_144)
    msgs = [{"role": "user", "content": "x" * 4_000}]
    m.observe_usage({"input_tokens": 34_000}, len(msgs))
    m.observe_append(msgs)
    assert m.used == 34_000

    msgs.append({"role": "assistant", "content": "y" * 8_000})
    m.observe_append(msgs)
    assert m.used == 36_000        # +2000 estimated tokens
    assert m.headroom == 262_144 - 36_000


def test_observe_append_recomputes_rather_than_accumulates():
    """Called twice for one append it must not double-count.

    The loop calls it from several places per iteration (after anchors,
    after a drain, after tool results, after a hook inject) and cannot
    always know whether an earlier call already covered the same tail.
    """
    m = ContextMeter(262_144)
    msgs = [{"role": "user", "content": "x" * 4_000}]
    m.observe_usage({"input_tokens": 34_000}, len(msgs))
    msgs.append({"role": "assistant", "content": "y" * 8_000})
    m.observe_append(msgs)
    once = m.used
    m.observe_append(msgs)
    m.observe_append(msgs)
    assert m.used == once


def test_resync_reanchors_after_an_in_place_rewrite():
    """Relief rewrites the list; the reported figure then describes a prompt
    that no longer exists. `resync` keeps the fixed cost (system prompt and
    tool schemas, which the estimator cannot see) and re-estimates the rest.
    """
    m = ContextMeter(262_144)
    msgs = [
        {"role": "system", "content": "s" * 4_000},
        {"role": "tool", "tool_call_id": "c1", "content": "big" * 20_000},
    ]
    m.observe_usage({"input_tokens": 50_000}, len(msgs))
    m.observe_append(msgs)
    before = m.used
    # A relief rung clears the big tool result.
    msgs[1]["content"] = "[cleared]"
    m.resync(msgs)
    assert m.used < before
    # The offset survives: the system prompt and tool schemas did not shrink.
    assert m.used > 30_000


def test_an_over_reporting_estimate_cannot_invent_headroom():
    """`offset` is clamped at 0. Without that a conservative engine report
    plus a fat estimate would subtract into negative fixed cost and leave the
    meter claiming room the turn does not have.
    """
    m = ContextMeter(262_144)
    msgs = [{"role": "user", "content": "x" * 400_000}]   # estimate >> report
    m.observe_usage({"input_tokens": 1_000}, len(msgs))
    m.observe_append(msgs)
    m.resync(msgs)
    assert m.used >= 0
    assert m.headroom <= m.window


def test_threshold_fraction_is_measured_against_the_compaction_wall():
    m = ContextMeter(262_144)
    msgs = [{"role": "user", "content": "x"}]
    m.observe_usage({"input_tokens": m.threshold}, len(msgs))
    m.observe_append(msgs)
    assert m.threshold_fraction == pytest.approx(1.0, abs=0.01)
    assert m.fraction < 1.0          # the wall is below the window


def test_context_window_for_falls_back_rather_than_raising():
    assert context_window_for("definitely-not-a-configured-model") > 0
    assert context_window_for("") > 0


# ---------------------------------------------------------------------------
# the relief ladder
# ---------------------------------------------------------------------------

def _opts(**kw) -> RunOptions:
    base = dict(
        model="primary", session_id="", tool_search_enabled=False,
        preserve_thinking_iterations=6,
    )
    base.update(kw)
    return RunOptions(**base)


def _pressured(msgs: list[dict], window: int = 262_144) -> ContextMeter:
    m = ContextMeter(window)
    m.observe_usage({"input_tokens": int(window * 0.95)}, len(msgs))
    m.observe_append(msgs)
    return m


def test_relief_does_nothing_when_the_meter_is_unmeasured():
    """Fail open. A turn with no usage report has not been shown to need
    anything, and pruning it would pay #520's re-prefill for free.
    """
    msgs = [
        {"role": "assistant", "content": "a", "reasoning": "R" * 5_000},
        {"role": "assistant", "content": "b", "reasoning": "R" * 5_000},
    ] * 5
    before = json.dumps(msgs)
    meter = ContextMeter(262_144)          # never observed
    report = _relieve_context(
        msgs, options=_opts(), meter=meter, reason="test",
    )
    assert json.dumps(msgs) == before
    assert report["freed_tokens"] == 0


def test_relief_prunes_reasoning_only_under_pressure():
    msgs = [
        {"role": "assistant", "content": f"m{i}", "reasoning": "R" * 40_000,
         "reasoning_content": "R" * 40_000}
        for i in range(12)
    ]
    meter = _pressured(msgs)
    _relieve_context(
        msgs, options=_opts(context_relief_reasoning_keep_under_pressure=2),
        meter=meter, reason="test",
    )
    with_reasoning = [m for m in msgs if m.get("reasoning")]
    assert len(with_reasoning) <= 6, len(with_reasoning)
    # Both spellings go together or the bound stops bounding anything: the
    # two engines behind this harness read different keys.
    for m in msgs:
        assert bool(m.get("reasoning")) == bool(m.get("reasoning_content"))


def test_relief_is_disabled_by_its_switch():
    msgs = [
        {"role": "assistant", "content": f"m{i}", "reasoning": "R" * 40_000}
        for i in range(12)
    ]
    before = json.dumps(msgs)
    meter = _pressured(msgs)
    report = _relieve_context(
        msgs, options=_opts(context_relief_enabled=False),
        meter=meter, reason="test",
    )
    assert json.dumps(msgs) == before
    assert report["rungs"] == []


def test_relief_reports_which_rungs_it_ran():
    msgs = [
        {"role": "assistant", "content": f"m{i}", "reasoning": "R" * 40_000}
        for i in range(12)
    ]
    meter = _pressured(msgs)
    report = _relieve_context(msgs, options=_opts(), meter=meter, reason="probe")
    assert report["reason"] == "probe"
    assert "tool_results" in report["rungs"]
    assert report["used_before"] >= report["used_after"]


# ---------------------------------------------------------------------------
# the wall-aware parse failure
# ---------------------------------------------------------------------------

def test_a_parse_failure_with_no_meter_reads_exactly_as_it_always_did():
    """A bare `run_query` caller has no meter, and its text must not change."""
    args = {"__parse_error__": True, "error": "Expecting value", "raw": "{oops",
            "finish_reason": "stop", "raw_len": 5}
    assert _parse_failure_text(args, meter=None) == (
        "Tool call arguments could not be parsed as JSON: Expecting value"
    )


def test_a_completion_cut_by_the_wall_is_told_not_to_re_send():
    """The measured failure: round 875 re-sent the same heredoc three times
    at 3147, then 1760, then 1175 tokens, because both the tool result and
    the vault skill said "retry verbatim". Each retry has less room than the
    attempt before it.
    """
    m = ContextMeter(262_144)
    msgs = [{"role": "user", "content": "x"}]
    m.observe_usage({"input_tokens": 258_000}, len(msgs))
    m.observe_append(msgs)
    args = {"__parse_error__": True, "error": "Unterminated string",
            "raw": "x" * 12_000, "finish_reason": "length", "raw_len": 12_000}
    text = _parse_failure_text(args, meter=m)
    assert "cut off by the context window" in text
    assert "Do not re-send" in text
    assert "did NOT run" in text
    assert "commit" in text.lower()


def test_low_headroom_and_a_big_argument_is_read_as_the_wall_too():
    """`finish_reason` is the direct signal, but the stream does not always
    carry one. A large argument string plus almost no room is the same event.
    """
    m = ContextMeter(262_144)
    msgs = [{"role": "user", "content": "x"}]
    m.observe_usage({"input_tokens": 250_000}, len(msgs))
    m.observe_append(msgs)
    args = {"__parse_error__": True, "error": "Unterminated string",
            "raw": "x" * 40_000, "finish_reason": "", "raw_len": 40_000}
    assert "cut off by the context window" in _parse_failure_text(args, meter=m)


def test_a_genuinely_malformed_call_with_room_left_is_told_to_re_send():
    m = ContextMeter(262_144)
    msgs = [{"role": "user", "content": "x"}]
    m.observe_usage({"input_tokens": 40_000}, len(msgs))
    m.observe_append(msgs)
    args = {"__parse_error__": True, "error": "Expecting ',' delimiter",
            "raw": "{bad}", "finish_reason": "stop", "raw_len": 5}
    text = _parse_failure_text(args, meter=m)
    assert "cut off by the context window" not in text
    assert "Do not re-send" not in text


# ---------------------------------------------------------------------------
# the loop end to end
# ---------------------------------------------------------------------------

class _FakePool:
    @property
    def discovered(self):
        return [("lloyd-mcp", [{
            "name": "Bash", "description": "shell",
            "inputSchema": {"type": "object", "properties": {}},
        }])]

    async def call_tool(self, name, args, *, session_id: str = "", **_kw):
        return {"content": "R" * 200, "is_error": False}


class _Script:
    """Scripted `stream_chat` reporting a prompt size that grows per turn."""

    def __init__(self, turns, prompt_tokens):
        self.turns = turns
        self.prompt_tokens = prompt_tokens
        self.captured: list[list[dict]] = []

    def __call__(self, **kwargs):
        idx = len(self.captured)
        self.captured.append([dict(m) for m in (kwargs.get("messages") or [])])
        return self._gen(idx, *self.turns[idx])

    async def _gen(self, idx, text, tool_calls):
        if text:
            yield {"choices": [{"delta": {"content": text}}]}
        for i, tc in enumerate(tool_calls):
            yield {"choices": [{"delta": {"tool_calls": [{
                "index": i, "id": tc["id"], "type": "function",
                "function": {"name": tc["name"],
                             "arguments": json.dumps(tc.get("arguments") or {})},
            }]}}]}
        yield {"choices": [{"delta": {},
                            "finish_reason": "tool_calls" if tool_calls else "stop"}]}
        yield {"choices": [], "usage": {
            "prompt_tokens": self.prompt_tokens[idx], "completion_tokens": 5}}


@pytest.fixture(autouse=True)
def _reset_cache():
    asyncio.run(tool_search_cache.clear())
    yield
    asyncio.run(tool_search_cache.clear())


def _run(monkeypatch, script, options):
    pool = _FakePool()

    async def _build_pool(_o):
        return pool
    monkeypatch.setattr("app.harness.loop._build_pool", _build_pool)
    monkeypatch.setattr("app.harness.loop.stream_chat", script)

    async def _drain():
        return [e async for e in run_query([{"role": "user", "content": "go"}], options)]
    return asyncio.run(_drain())


TC = [{"id": "c1", "name": "Bash", "arguments": {}}]


def test_the_loop_fills_a_caller_supplied_meter(monkeypatch):
    """The router hands one object to the loop, the anchor and the observer.
    If the loop built its own instead, the other two would read zeros.
    """
    meter = ContextMeter(262_144)
    script = _Script([("a", TC), ("done", [])], [120_000, 150_000])
    opts = RunOptions(
        model="primary", session_id="", tool_search_enabled=False,
        context_meter=meter,
    )
    _run(monkeypatch, script, opts)
    assert meter.measured
    assert meter.used >= 150_000


def test_assistant_message_events_carry_the_context_position(monkeypatch):
    script = _Script([("a", TC), ("done", [])], [120_000, 150_000])
    meter = ContextMeter(262_144)
    opts = RunOptions(
        model="primary", session_id="", tool_search_enabled=False,
        context_meter=meter,
    )
    events = _run(monkeypatch, script, opts)
    asst = [e for e in events if e["type"] == "assistant_message"]
    assert asst and asst[-1]["context"] is not None
    ctx = asst[-1]["context"]
    assert ctx["context_window"] == 262_144
    assert ctx["headroom"] == 262_144 - ctx["input_tokens"]


def _inject_on_terminal(hooks, chat_messages, text="[INNER VOICE] keep going"):
    """Register an OnEvent callback that injects on a silent terminal turn.

    This is what the Inner Voice observer does — it holds the same
    `chat_messages` list the loop does (`chat_messages_handle`) and appends
    to it from inside `fire_on_event`.
    """
    fired: list[int] = []

    async def on_event(evt):
        # Once only — the real observer has an inject cap, and an injector
        # that fires on every terminal iteration never lets the turn end.
        if fired:
            return
        if evt.get("type") == "assistant_message" and not evt.get("tool_calls"):
            fired.append(1)
            chat_messages.append({"role": "user", "content": text})
    hooks.add_on_event(on_event)


def test_a_terminal_inject_with_no_headroom_ends_the_turn(monkeypatch):
    """#875's shape exactly: the observer injects on a silent terminal
    iteration at the wall, the loop continues, and the next completion has
    no room to write in — so the model re-sends a cut-off call and the turn
    dies at a vLLM 400. Ending as `context_exhausted` is the honest outcome;
    `run_prompt_in_session` already treats any non-`stop` as no conclusion.
    """
    from app.harness.hooks import HookRegistry

    shared: list[dict] = []
    hooks = HookRegistry()
    _inject_on_terminal(hooks, shared)

    meter = ContextMeter(262_144)
    # One text-only iteration, reporting a prompt that leaves ~2k of room.
    script = _Script([("done", [])], [260_000])
    opts = RunOptions(
        model="primary", session_id="", tool_search_enabled=False,
        context_meter=meter, hooks=hooks, chat_messages_handle=shared,
        context_relief_terminal_floor_tokens=12_000,
        # Nothing for the ladder to free: the residue is the fixed prefix.
        context_relief_shrink_arguments=False,
    )
    events = _run(monkeypatch, script, opts)

    result = [e for e in events if e["type"] == "result"][-1]
    assert result["stop_reason"] == "context_exhausted", result["stop_reason"]
    # And exactly one request was made: the inject was dropped, not answered.
    assert len(script.captured) == 1


def test_a_terminal_inject_with_room_still_continues_the_loop(monkeypatch):
    """The guard must not break the stall rescue it sits in front of — that
    rescue is what landed #278. With headroom the inject is answered.
    """
    from app.harness.hooks import HookRegistry

    shared: list[dict] = []
    hooks = HookRegistry()
    _inject_on_terminal(hooks, shared)

    meter = ContextMeter(262_144)
    script = _Script([("done", []), ("really done", [])], [40_000, 41_000])
    opts = RunOptions(
        model="primary", session_id="", tool_search_enabled=False,
        context_meter=meter, hooks=hooks, chat_messages_handle=shared,
        context_relief_terminal_floor_tokens=12_000,
    )
    events = _run(monkeypatch, script, opts)

    result = [e for e in events if e["type"] == "result"][-1]
    assert result["stop_reason"] != "context_exhausted"
    # Two requests: the inject was answered on a second iteration.
    assert len(script.captured) == 2
    assert any("[INNER VOICE]" in str(m.get("content") or "")
               for m in script.captured[1])


def test_the_guard_relieves_before_it_gives_up(monkeypatch):
    """Dropping the inject is the last resort, not the first move. When the
    ladder can free enough, the turn continues.
    """
    from app.harness.hooks import HookRegistry

    # A big clearable tool result is what the ladder's first rung eats.
    shared: list[dict] = [{"role": "user", "content": "go"}]
    for i in range(40):
        shared.append({"role": "assistant", "content": "", "tool_calls": [{
            "id": f"c{i}", "type": "function",
            "function": {"name": "Bash", "arguments": "{}"}}]})
        shared.append({"role": "tool", "tool_call_id": f"c{i}",
                       "content": "B" * 40_000})

    hooks = HookRegistry()
    _inject_on_terminal(hooks, shared)

    meter = ContextMeter(262_144)
    script = _Script([("done", []), ("ok", [])], [255_000, 120_000])
    opts = RunOptions(
        model="primary", session_id="", tool_search_enabled=False,
        context_meter=meter, hooks=hooks, chat_messages_handle=shared,
        context_relief_terminal_floor_tokens=12_000,
        intra_turn_microcompact_keep_recent=2,
    )
    events = _run(monkeypatch, script, opts)
    result = [e for e in events if e["type"] == "result"][-1]
    # Relief freed room, so the inject was answered rather than dropped.
    assert result["stop_reason"] != "context_exhausted", result["stop_reason"]
    assert len(script.captured) == 2


def test_relief_target_tracks_the_configured_microcompact_target():
    from app.harness.loop import _relief_target

    m = ContextMeter(262_144)
    opts = _opts(intra_turn_microcompact_target_fraction=0.52)
    assert _relief_target(opts, m) == int(m.threshold * 0.52)
