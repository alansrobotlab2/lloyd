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


def test_build_tool_list_raises_on_cross_server_collision():
    discovered = [
        ("server-a", [{"name": "shared", "description": "", "inputSchema": {}}]),
        ("server-b", [{"name": "shared", "description": "", "inputSchema": {}}]),
    ]
    with pytest.raises(ValueError, match="collision"):
        build_tool_list(discovered, disallowed=set())


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
