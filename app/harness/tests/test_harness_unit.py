"""Harness unit tests — no live services.

Covers:
  - Event helper shapes
  - Tool-call delta accumulation across multi-chunk streams
  - Argument-parse-error fallback (qwen3_xml resilience)
  - Schema translation: bare-name vs namespaced
  - Hook registry: matcher routing, deny-first-wins, exception swallow
  - Disallowed-tools enforcement
"""

import asyncio
import json

import pytest

from app.harness import events
from app.harness.errors import ParseError
from app.harness.hooks import HookRegistry
from app.harness.loop import (
    _accumulate_tool_call,
    _commit_tool_calls,
    _has_system,
    _merge_usage,
)
from app.harness.tool_schema import (
    build_tool_list,
    mcp_tool_to_openai,
    resolve_tool_name,
)


# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------


def test_text_delta_shape():
    e = events.text_delta("hi")
    assert e == {"type": "text_delta", "text": "hi"}


def test_tool_call_shape():
    e = events.tool_call(call_id="c1", name="Bash", args_json='{"x":1}', args_dict={"x": 1})
    assert e["type"] == "tool_call"
    assert e["name"] == "Bash"
    assert e["args_dict"] == {"x": 1}


def test_result_shape_defaults():
    e = events.result(stop_reason="stop")
    assert e["usage"] == {} and e["num_turns"] == 0 and e["response_text"] == ""


def test_stream_raw_carries_error():
    e = events.stream_raw("garbage", error="bad json")
    assert e["raw"] == "garbage" and e["error"] == "bad json"


# ---------------------------------------------------------------------------
# Tool-call accumulation
# ---------------------------------------------------------------------------


def test_accumulate_single_call_split_args():
    acc: dict = {}
    _accumulate_tool_call(acc, {"index": 0, "id": "call_1", "type": "function",
                                "function": {"name": "Bash", "arguments": '{"co'}})
    _accumulate_tool_call(acc, {"index": 0, "function": {"arguments": 'mmand": "ls"}'}})
    committed = _commit_tool_calls(acc)
    assert len(committed) == 1
    assert committed[0]["id"] == "call_1"
    assert committed[0]["function"]["name"] == "Bash"
    assert committed[0]["_args_dict"] == {"command": "ls"}


def test_accumulate_two_calls_in_one_turn():
    acc: dict = {}
    _accumulate_tool_call(acc, {"index": 0, "id": "c1", "function": {"name": "Read", "arguments": '{"file_path":"/a"}'}})
    _accumulate_tool_call(acc, {"index": 1, "id": "c2", "function": {"name": "Read", "arguments": '{"file_path":"/b"}'}})
    committed = _commit_tool_calls(acc)
    assert [c["_args_dict"]["file_path"] for c in committed] == ["/a", "/b"]


def test_commit_malformed_args_recovers():
    acc: dict = {}
    _accumulate_tool_call(acc, {"index": 0, "id": "c1", "function": {"name": "Bash", "arguments": '{"command": <oops>'}})
    committed = _commit_tool_calls(acc)
    assert len(committed) == 1
    assert committed[0]["_args_dict"]["__parse_error__"] is True
    assert "raw" in committed[0]["_args_dict"]


def test_commit_synthesizes_id_when_missing():
    acc: dict = {}
    _accumulate_tool_call(acc, {"index": 0, "function": {"name": "Read", "arguments": "{}"}})
    committed = _commit_tool_calls(acc)
    assert committed[0]["id"].startswith("call_")


def test_commit_skips_empty_placeholder():
    acc: dict = {}
    _accumulate_tool_call(acc, {"index": 0, "function": {"name": "", "arguments": ""}})
    assert _commit_tool_calls(acc) == []


def test_commit_rejects_non_object_args():
    acc: dict = {}
    _accumulate_tool_call(acc, {"index": 0, "id": "c1", "function": {"name": "Bash", "arguments": '"a string"'}})
    committed = _commit_tool_calls(acc)
    assert committed[0]["_args_dict"]["__parse_error__"] is True


def test_commit_repairs_qwen3_trailing_brace():
    """vLLM's qwen3_xml tool parser sometimes appends an extra `}` to args.
    The harness now repairs this transparently via raw_decode + trailing-
    junk tolerance, so the dispatched tool call works AND the next-turn
    replay sees clean args.
    """
    acc: dict = {}
    bad = '{"file_path": "/home/alansrobotlab/obsidian/lloyd/SOUL.md"}}'
    _accumulate_tool_call(acc, {"index": 0, "id": "c1", "function": {"name": "Read", "arguments": bad}})
    committed = _commit_tool_calls(acc)
    assert len(committed) == 1
    # Args parsed correctly.
    assert committed[0]["_args_dict"] == {"file_path": "/home/alansrobotlab/obsidian/lloyd/SOUL.md"}
    # And the stored arguments string is sanitized for replay.
    assert committed[0]["function"]["arguments"] == '{"file_path": "/home/alansrobotlab/obsidian/lloyd/SOUL.md"}'


def test_commit_repairs_qwen3_trailing_brace_complex_args():
    """Same repair on a multi-key args string with the trailing-brace bug."""
    acc: dict = {}
    bad = ('{"path": "/home/alansrobotlab/obsidian/knowledge", '
           '"output_mode": "files_with_matches", '
           '"pattern": "lloyd|framework", "head_limit": 30}}')
    _accumulate_tool_call(acc, {"index": 0, "id": "c1", "function": {"name": "Grep", "arguments": bad}})
    committed = _commit_tool_calls(acc)
    assert len(committed) == 1
    assert committed[0]["_args_dict"]["head_limit"] == 30
    assert committed[0]["_args_dict"]["pattern"] == "lloyd|framework"
    # Replay-safe args.
    assert committed[0]["function"]["arguments"].endswith("}")
    assert not committed[0]["function"]["arguments"].endswith("}}")


def test_commit_unrepairable_garbage_still_falls_through():
    """When the raw really can't be salvaged, fall through to the
    __parse_error__ path (existing behavior preserved)."""
    acc: dict = {}
    _accumulate_tool_call(acc, {"index": 0, "id": "c1", "function": {"name": "Bash", "arguments": '{not valid json}'}})
    committed = _commit_tool_calls(acc)
    assert committed[0]["_args_dict"]["__parse_error__"] is True
    assert committed[0]["function"]["arguments"] == "{}"  # sanitized for replay


# ---------------------------------------------------------------------------
# Misc loop helpers
# ---------------------------------------------------------------------------


def test_has_system_true():
    assert _has_system([{"role": "system", "content": "x"}, {"role": "user", "content": "y"}])


def test_has_system_false():
    assert not _has_system([{"role": "user", "content": "y"}])


def test_merge_usage_normalizes_openai_keys():
    """vLLM emits OpenAI-style names; harness normalizes to Anthropic-style."""
    out = _merge_usage(
        {"input_tokens": 10, "output_tokens": 5},
        {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
    )
    assert out == {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28}


def test_merge_usage_ignores_non_int():
    out = _merge_usage({}, {"prompt_tokens": 5, "model_name": "primary"})
    assert out == {"input_tokens": 5}


def test_merge_usage_keeps_reasoning_tokens():
    """P11: the reasoning share arrives nested, like the cache hits did, and
    the int-only loop skipped it. It is kept as its own key, never folded into
    `output_tokens` (it is already inside that count), and the cross-iteration
    fold sums it."""
    from app.harness.loop import _accumulate_iteration_usage

    chunk = {"prompt_tokens": 100, "completion_tokens": 40,
             "completion_tokens_details": {"reasoning_tokens": 30}}
    one = _merge_usage({}, chunk)
    assert one["reasoning_tokens"] == 30
    assert one["output_tokens"] == 40
    total = _accumulate_iteration_usage(_accumulate_iteration_usage({}, one), one)
    assert total["reasoning_tokens"] == 60
    # A details block without the count, or a null one, adds no key.
    assert "reasoning_tokens" not in _merge_usage(
        {}, {"completion_tokens_details": {"reasoning_tokens": None}})
    assert "reasoning_tokens" not in _merge_usage(
        {}, {"completion_tokens_details": None})


def test_tool_result_carries_duration_and_error_class():
    ok = events.tool_result(call_id="c", name="Read", content="x",
                            duration_ms=42, handshake_ms=3)
    assert ok["duration_ms"] == 42 and ok["handshake_ms"] == 3
    assert "error_class" not in ok
    # Unmeasured is absent, never zero: a deny never reached the MCP call.
    denied = events.tool_result(call_id="c", name="Bash", content="no",
                                is_error=True, error_class="denied")
    assert denied["error_class"] == "denied"
    assert "duration_ms" not in denied and "handshake_ms" not in denied
    # A failure with no class named is the tool's own failure.
    bare = events.tool_result(call_id="c", name="Bash", content="boom", is_error=True)
    assert bare["error_class"] == "tool_error"
    assert set(events.TOOL_ERROR_CLASSES) >= {
        "denied", "disabled", "parse_error", "transport", "mcp_error",
        "cancelled", "dispatch_failed", "tool_error"}


def test_every_error_class_the_loop_sets_is_a_known_class():
    import inspect
    import re

    from app.harness import loop

    src = inspect.getsource(loop)
    used = set(re.findall(r'error_class="([a-z_]+)"', src))
    used |= set(re.findall(r'return "([a-z_]+_error)"', src))
    assert used, "the loop names no error classes"
    assert used <= set(events.TOOL_ERROR_CLASSES)


def test_result_stop_reason_literal_matches_what_the_loop_emits():
    """Every `stop_reason = "<x>"` the loop assigns is in the Literal, plus the
    engine finish_reasons a terminal iteration passes through (`stop`,
    `length`, `tool_calls`). `context_exhausted` was emitted for months while
    missing from it."""
    import inspect
    import re
    import typing

    from app.harness import loop

    literal = set(typing.get_args(
        events.NormalizedEvent.__annotations__["stop_reason"]))
    src = inspect.getsource(loop)
    assigned = set(re.findall(r'\bstop_reason = "([a-z_]+)"', src))
    assert "context_exhausted" in assigned
    assert assigned <= literal, assigned - literal
    assert {"stop", "length", "tool_calls", "stream_error", "error"} <= literal


# ---------------------------------------------------------------------------
# Tool schema translation
# ---------------------------------------------------------------------------


def test_mcp_tool_to_openai_uses_bare_name():
    t = mcp_tool_to_openai({"name": "Bash", "description": "shell", "inputSchema": {"type": "object"}})
    assert t["function"]["name"] == "Bash"


def test_mcp_tool_to_openai_does_not_namespace_domain_tools():
    t = mcp_tool_to_openai({"name": "memory_add", "description": "x", "inputSchema": {}})
    assert t["function"]["name"] == "memory_add"


def test_mcp_tool_to_openai_rejects_long_names():
    with pytest.raises(ValueError):
        mcp_tool_to_openai({"name": "b" * 80, "description": "", "inputSchema": {}})


def test_resolve_tool_name_bare():
    assert resolve_tool_name("Bash") == (None, "Bash")


def test_resolve_tool_name_legacy_namespaced():
    # Old persisted sessions still carry mcp__server__tool names.
    assert resolve_tool_name("mcp__lloyd-mcp__memory_add") == ("lloyd-mcp", "memory_add")


def test_resolve_tool_name_falls_through_on_malformed():
    # No __ separator after mcp__ — treat as bare.
    assert resolve_tool_name("mcp__weird") == (None, "mcp__weird")


def test_build_tool_list_filters_disallowed_bare():
    discovered = [
        ("lloyd-mcp", [
            {"name": "Bash", "description": "", "inputSchema": {}},
            {"name": "Read", "description": "", "inputSchema": {}},
        ])
    ]
    out = build_tool_list(discovered, disallowed={"Bash"})
    assert [t["function"]["name"] for t in out] == ["Read"]


def test_build_tool_list_accepts_legacy_namespaced_disallow():
    # Rolled-forward configs may still spell disallowed tools the old way.
    discovered = [
        ("lloyd-mcp", [
            {"name": "memory_add", "description": "", "inputSchema": {}},
        ])
    ]
    out = build_tool_list(discovered, disallowed={"mcp__lloyd-mcp__memory_add"})
    assert out == []


def test_a_cross_server_collision_surfaces_as_tool_discovery_error():
    """D13: a discovery failure, named as one (it used to be a bare
    ValueError), carrying both servers."""
    from app.harness.errors import ToolDiscoveryError

    discovered = [
        ("server-a", [{"name": "shared", "description": "", "inputSchema": {}}]),
        ("server-b", [{"name": "shared", "description": "", "inputSchema": {}}]),
    ]
    with pytest.raises(ToolDiscoveryError, match="collision") as info:
        build_tool_list(discovered, disallowed=set())
    assert info.value.servers == ["server-a", "server-b"]


def test_a_long_tool_name_is_warned_not_silently_dropped(caplog):
    """D13: the over-64-char tool is still left out (the engine would reject
    the whole request), but the log names it; the rest of the catalog ships."""
    long_name = "x" * 65
    discovered = [("lloyd-mcp", [
        {"name": long_name, "description": "", "inputSchema": {}},
        {"name": "Read", "description": "", "inputSchema": {}},
    ])]
    with caplog.at_level("WARNING", logger="app.harness.tool_schema"):
        out = build_tool_list(discovered, disallowed=set())
    assert [t["function"]["name"] for t in out] == ["Read"]
    assert any(long_name in r.getMessage() for r in caplog.records)


def test_run_options_carries_no_dead_knobs():
    """D13: four `RunOptions` fields that nothing read — SDK leftovers
    (`permission_mode`, `env`, `history`) and an A/B the loop never wired
    (`context_relief_send_max_tokens_reservation`) — are gone, and so is the
    config key that fed the last one. A knob that reads as a setting and
    does nothing is the failure this pins against."""
    import dataclasses
    from pathlib import Path

    import yaml

    from app.harness.options import RunOptions

    names = {f.name for f in dataclasses.fields(RunOptions)}
    dead = {"permission_mode", "env", "history",
            "context_relief_send_max_tokens_reservation"}
    assert not names & dead, names & dead
    for knob in dead:
        with pytest.raises(TypeError):
            RunOptions(model="m", **{knob: None})
    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[3] / "config.yaml").read_text())
    relief = (cfg.get("harness") or {}).get("context_relief") or {}
    assert "send_max_tokens_reservation" not in relief


# ---------------------------------------------------------------------------
# HookRegistry
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.new_event_loop().run_until_complete(coro)


def test_pre_tool_use_no_callbacks_passes():
    reg = HookRegistry()
    out = asyncio.run(reg.fire_pre_tool_use(session_id="s", tool_name="Bash", tool_input={}))
    assert out == {}


def test_pre_tool_use_deny_wins():
    reg = HookRegistry()

    async def deny(input_data, _id, _ctx):
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                       "permissionDecision": "deny",
                                       "permissionDecisionReason": "no"}}

    reg.add_pre_tool_use("Bash", deny)
    out = asyncio.run(reg.fire_pre_tool_use(session_id="s", tool_name="Bash", tool_input={"command": "rm"}))
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_pre_tool_use_matcher_skips_other_tools():
    reg = HookRegistry()
    fired = []

    async def cb(input_data, _id, _ctx):
        fired.append(input_data["tool_name"])
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                       "permissionDecision": "deny",
                                       "permissionDecisionReason": "no"}}

    reg.add_pre_tool_use("Bash", cb)
    out = asyncio.run(reg.fire_pre_tool_use(session_id="s", tool_name="Read", tool_input={}))
    assert out == {} and fired == []


def test_pre_tool_use_first_deny_wins_short_circuit():
    reg = HookRegistry()
    fired = []

    async def cb_a(input_data, _id, _ctx):
        fired.append("a")
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                       "permissionDecision": "deny",
                                       "permissionDecisionReason": "first"}}

    async def cb_b(input_data, _id, _ctx):
        fired.append("b")
        return {}

    reg.add_pre_tool_use(None, cb_a)
    reg.add_pre_tool_use(None, cb_b)
    out = asyncio.run(reg.fire_pre_tool_use(session_id="s", tool_name="Bash", tool_input={}))
    assert out["hookSpecificOutput"]["permissionDecisionReason"] == "first"
    assert fired == ["a"]


# D5 (review 2026-09-24): a raising PreToolUse callback is a pass only when it
# was registered fail-open; a gate registered fail-closed denies.

async def _raise_cb(input_data, _id, _ctx):
    raise RuntimeError("boom")


def _capture_hook_events(monkeypatch):
    from app.harness import telemetry
    seen = []
    monkeypatch.setattr(
        telemetry, "log_harness_event",
        lambda sid, event, data, **kw: seen.append((sid, event, data)),
    )
    return seen


def test_a_raising_observer_hook_is_a_pass_by_default(monkeypatch):
    _capture_hook_events(monkeypatch)
    reg = HookRegistry()
    reg.add_pre_tool_use(None, _raise_cb)
    out = asyncio.run(reg.fire_pre_tool_use(session_id="s", tool_name="Bash", tool_input={}))
    assert out == {}


def test_a_raising_gate_hook_denies_and_names_the_error(monkeypatch):
    _capture_hook_events(monkeypatch)
    reg = HookRegistry()
    reg.add_pre_tool_use(None, _raise_cb, fail_closed=True)
    out = asyncio.run(reg.fire_pre_tool_use(session_id="s", tool_name="Bash", tool_input={}))
    hso = out["hookSpecificOutput"]
    assert hso["permissionDecision"] == "deny"
    reason = hso["permissionDecisionReason"]
    assert "_raise_cb" in reason
    assert "RuntimeError: boom" in reason
    assert "must not open" in reason


def test_a_raising_gate_beats_a_deliver_registered_ahead_of_it(monkeypatch):
    _capture_hook_events(monkeypatch)
    reg = HookRegistry()

    async def deliver_cb(input_data, _id, _ctx):
        return {"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "skillDeliver": {"skill": "x", "label": "r", "content": "card"},
        }}

    reg.add_pre_tool_use(None, deliver_cb)
    reg.add_pre_tool_use(None, _raise_cb, fail_closed=True)
    out = asyncio.run(reg.fire_pre_tool_use(session_id="s", tool_name="Bash", tool_input={}))
    hso = out["hookSpecificOutput"]
    assert hso.get("permissionDecision") == "deny"
    assert "skillDeliver" not in hso


def test_a_raised_hook_writes_one_hook_raised_event(monkeypatch):
    seen = _capture_hook_events(monkeypatch)
    reg = HookRegistry()
    reg.add_pre_tool_use(None, _raise_cb)
    reg.add_pre_tool_use(None, _raise_cb, fail_closed=True)
    asyncio.run(reg.fire_pre_tool_use(
        session_id="s1", tool_name="Bash", tool_input={}, tool_use_id="c1"))
    assert [e for _s, e, _d in seen] == ["harness.hook_raised"] * 2
    first, second = seen[0][2], seen[1][2]
    assert first["fail_closed"] is False and second["fail_closed"] is True
    assert first["tool"] == "Bash" and first["tool_use_id"] == "c1"
    assert "RuntimeError: boom" in first["error"]
    assert all(s == "s1" for s, _e, _d in seen)

    seen.clear()
    reg2 = HookRegistry()
    reg2.add_pre_tool_use(None, _raise_cb)
    asyncio.run(reg2.fire_pre_tool_use(session_id="s1", tool_name="Bash", tool_input={}))
    assert len(seen) == 1


def test_post_tool_use_fires_all_observers():
    reg = HookRegistry()
    fired = []

    async def cb_a(input_data, _id, _ctx):
        fired.append("a")
        return {}

    async def cb_b(input_data, _id, _ctx):
        fired.append("b")
        return {}

    reg.add_post_tool_use(cb_a)
    reg.add_post_tool_use(cb_b)
    asyncio.run(reg.fire_post_tool_use(session_id="s", tool_name="Bash", tool_input={}, tool_response="ok"))
    assert fired == ["a", "b"]


def test_post_tool_use_failure_carries_error():
    reg = HookRegistry()
    seen = []

    async def cb(input_data, _id, _ctx):
        seen.append(input_data["error"])
        return {}

    reg.add_post_tool_use_failure(cb)
    asyncio.run(reg.fire_post_tool_use_failure(session_id="s", tool_name="Bash", tool_input={}, error="bad args"))
    assert seen == ["bad args"]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_parse_error_carries_raw():
    exc = ParseError("oops", raw='{"bad')
    assert exc.raw == '{"bad'
