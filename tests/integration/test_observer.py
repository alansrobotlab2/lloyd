"""Inner Voice observer — unit + behavior tests.

These don't hit a running server. They mock the observer's HTTP call
layer and exercise the parsing, lever dispatch, budget cap, and pretool
deny logic directly.

Run:
  /home/alansrobotlab/lloyd/.venvs/lloyd/bin/python tests/integration/test_observer.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

LLOYD_HOME = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LLOYD_HOME))

from app.harness.hooks import HookRegistry
from app.inner_voice import observer as obs_mod
from app.inner_voice.observer import (
    ObserverDecision,
    ObserverState,
    _apply_lever,
    _bash_command_is_safely_readonly,
    _build_event_user_prompt,
    _fast_path_pretool,
    _fast_path_tool_result,
    _normalize_action,
    _parse_goal_card,
    _parse_observer_json,
    extract_goal_card,
    install_observer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(*, ambient_callback=None, clarify_callback=None, budget=3, goal_card=None) -> ObserverState:
    state = ObserverState(
        session_id="test_sess",
        turn_id="test_turn",
        user_request="What's 2+2?",
        chat_messages_handle=[],
        cancel_event=asyncio.Event(),
        enqueue_ambient_callback=ambient_callback,
        clarify_callback=clarify_callback,
        primary_model="primary",
        intervention_budget=budget,
        goal_card=goal_card,
    )
    return state


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def test_parse_full_object():
    raw = '{"action":"noop","reason":"on track"}'
    p = _parse_observer_json(raw)
    assert p == {"action": "noop", "reason": "on track"}, p
    print("test_parse_full_object: OK")


def test_parse_with_prefill():
    # Model returned just the continuation
    raw = '"inject","reason":"empty","content":"answer the user"}'
    p = _parse_observer_json(raw)
    assert p["action"] == "inject", p
    assert p["content"] == "answer the user", p
    print("test_parse_with_prefill: OK")


def test_parse_embedded():
    raw = 'Here\'s my decision: {"action":"cancel","reason":"loop"} done.'
    p = _parse_observer_json(raw)
    assert p["action"] == "cancel", p
    print("test_parse_embedded: OK")


def test_parse_garbage():
    p = _parse_observer_json("complete nonsense")
    assert p is None, p
    print("test_parse_garbage: OK")


def test_parse_bare_word():
    """Model sometimes returns just `noop` after the prefill — no JSON."""
    p = _parse_observer_json("noop")
    assert p == {"action": "noop", "reason": "bare-word fallback"}, p
    p2 = _parse_observer_json("noop\"}")
    assert p2["action"] == "noop", p2
    print("test_parse_bare_word: OK")


def test_parse_dropped_opening_quote():
    """Local models sometimes honor the prefill but drop the value's opening
    quote — e.g. return `inject","reason":"...","content":"..."}` after
    prefill `{"action":` (which would otherwise yield invalid JSON).
    """
    raw = 'inject","reason":"empty terminal","content":"answer the user"}'
    p = _parse_observer_json(raw)
    assert p is not None, p
    assert p["action"] == "inject", p
    assert p["content"] == "answer the user", p
    print("test_parse_dropped_opening_quote: OK")


# ---------------------------------------------------------------------------
# Action normalization
# ---------------------------------------------------------------------------


def test_normalize_noop_path():
    assert _normalize_action("noop", allow_deny_tool=False) == "noop"
    assert _normalize_action("inject", allow_deny_tool=False) == "inject"
    assert _normalize_action("cancel", allow_deny_tool=False) == "cancel"
    assert _normalize_action("ambient", allow_deny_tool=False) == "ambient"
    # pretool actions out-of-context → noop
    assert _normalize_action("allow", allow_deny_tool=False) == "noop"
    assert _normalize_action("deny_tool", allow_deny_tool=False) == "noop"
    # Garbage → noop
    assert _normalize_action("WAT", allow_deny_tool=False) == "noop"
    assert _normalize_action(None, allow_deny_tool=False) == "noop"
    print("test_normalize_noop_path: OK")


def test_normalize_pretool_path():
    assert _normalize_action("allow", allow_deny_tool=True) == "allow"
    assert _normalize_action("deny_tool", allow_deny_tool=True) == "deny_tool"
    # Common variant
    assert _normalize_action("deny", allow_deny_tool=True) == "deny_tool"
    # Post-event actions in pretool context → fail open (allow)
    assert _normalize_action("inject", allow_deny_tool=True) == "allow"
    assert _normalize_action("noop", allow_deny_tool=True) == "allow"
    # Garbage in pretool context → fail open (allow)
    assert _normalize_action("WAT", allow_deny_tool=True) == "allow"
    print("test_normalize_pretool_path: OK")


# ---------------------------------------------------------------------------
# Lever dispatch
# ---------------------------------------------------------------------------


def run_async(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_inject_appends_user_message():
    """Inject uses role=user (not system) because vLLM rejects mid-stream
    system messages — they must live at position 0."""
    state = _make_state()
    decision = ObserverDecision(action="inject", reason="empty", content="please answer")
    run_async(_apply_lever(state, decision, trigger="assistant_message"))
    assert len(state.chat_messages_handle) == 1
    assert state.chat_messages_handle[0]["role"] == "user", state.chat_messages_handle[0]
    assert "please answer" in state.chat_messages_handle[0]["content"]
    assert "[INNER VOICE]" in state.chat_messages_handle[0]["content"]
    assert state.interventions_used == 1
    print("test_inject_appends_user_message: OK")


def test_cancel_sets_event():
    """Cancel sets cancel_event AND does NOT consume the intervention budget.
    Cancel is the escape hatch — rationing it would prevent recovery from
    'primary keeps ignoring my injects' cases."""
    state = _make_state()
    decision = ObserverDecision(action="cancel", reason="loop")
    run_async(_apply_lever(state, decision, trigger="assistant_message"))
    assert state.cancel_event.is_set()
    assert state.interventions_used == 0, "cancel must not count against budget"
    print("test_cancel_sets_event: OK")


def test_cancel_works_when_budget_exhausted():
    """Cancel must remain available after inject/ambient/clarify budget hits.
    This is the fix for the temporal-knowledge-graph stall: observer was
    blocked from cancelling a runaway loop because budget was exhausted on
    earlier injects."""
    state = _make_state(budget=2)
    # Burn the budget on injects.
    for i in range(2):
        d = ObserverDecision(action="inject", reason=f"r{i}", content=f"msg{i}")
        run_async(_apply_lever(state, d, trigger="assistant_message"))
    assert state.interventions_used == 2
    # Now cancel — must still fire.
    cancel_dec = ObserverDecision(action="cancel", reason="injects ignored, force-stop")
    run_async(_apply_lever(state, cancel_dec, trigger="assistant_message"))
    assert cancel_dec.action == "cancel", cancel_dec
    assert state.cancel_event.is_set()
    print("test_cancel_works_when_budget_exhausted: OK")


def test_ambient_fires_callback():
    captured = {}

    async def amb_cb(content, reason):
        captured["content"] = content
        captured["reason"] = reason

    state = _make_state(ambient_callback=amb_cb)
    decision = ObserverDecision(action="ambient", reason="follow up", content="check on this later")
    run_async(_apply_lever(state, decision, trigger="result"))
    assert captured["content"] == "check on this later"
    assert state.interventions_used == 1
    print("test_ambient_fires_callback: OK")


def test_ambient_with_no_callback_degrades_to_noop():
    state = _make_state(ambient_callback=None)
    decision = ObserverDecision(action="ambient", reason="follow up", content="x")
    run_async(_apply_lever(state, decision, trigger="result"))
    assert decision.action == "noop_no_ambient_channel"
    assert state.interventions_used == 0
    print("test_ambient_with_no_callback_degrades_to_noop: OK")


def test_inject_empty_content_degrades_to_noop():
    state = _make_state()
    decision = ObserverDecision(action="inject", reason="x", content="   ")
    run_async(_apply_lever(state, decision, trigger="assistant_message"))
    assert decision.action == "noop_empty_content"
    assert state.interventions_used == 0
    print("test_inject_empty_content_degrades_to_noop: OK")


def test_inject_on_result_translates_to_ambient_when_callback_present():
    """Result-trigger fires AFTER the harness emits its terminal event,
    so inject can't take effect on this turn. With an ambient callback
    wired, it gets translated to ambient (a follow-up turn). Without one,
    it degrades to noop."""
    captured = {}

    async def amb_cb(content, reason):
        captured["content"] = content
        captured["reason"] = reason

    state = _make_state(ambient_callback=amb_cb)
    decision = ObserverDecision(action="inject", reason="incomplete", content="say more")
    run_async(_apply_lever(state, decision, trigger="result"))
    assert decision.action == "ambient", decision
    assert captured.get("content") == "say more"
    assert state.interventions_used == 1
    print("test_inject_on_result_translates_to_ambient_when_callback_present: OK")


def test_inject_on_result_degrades_when_no_callback():
    state = _make_state(ambient_callback=None)
    decision = ObserverDecision(action="inject", reason="incomplete", content="say more")
    run_async(_apply_lever(state, decision, trigger="result"))
    assert decision.action == "noop_inject_on_result", decision
    assert state.interventions_used == 0
    assert len(state.chat_messages_handle) == 0, "must not append after turn end"
    print("test_inject_on_result_degrades_when_no_callback: OK")


def test_cancel_on_result_degrades_to_noop():
    """Cancel after the turn ended is meaningless — the harness already returned."""
    state = _make_state()
    decision = ObserverDecision(action="cancel", reason="too late")
    run_async(_apply_lever(state, decision, trigger="result"))
    assert decision.action == "noop_cancel_on_result", decision
    assert not state.cancel_event.is_set(), "must not set cancel after turn end"
    assert state.interventions_used == 0
    print("test_cancel_on_result_degrades_to_noop: OK")


def test_budget_exhausted():
    state = _make_state(budget=1)
    d1 = ObserverDecision(action="inject", reason="r1", content="msg1")
    run_async(_apply_lever(state, d1, trigger="assistant_message"))
    assert state.interventions_used == 1
    # Second intervention should be downgraded
    d2 = ObserverDecision(action="inject", reason="r2", content="msg2")
    run_async(_apply_lever(state, d2, trigger="assistant_message"))
    assert d2.action == "noop_budget_exhausted"
    assert state.interventions_used == 1, "budget should not advance"
    assert len(state.chat_messages_handle) == 1, "second inject should not have appended"
    print("test_budget_exhausted: OK")


# ---------------------------------------------------------------------------
# Observer call (with mocked HTTP)
# ---------------------------------------------------------------------------


def test_call_observer_parses_response():
    fake_body = {
        "choices": [
            {"message": {"content": '"inject","reason":"empty","content":"answer the user"}'}}
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }

    async def fake_post(**kwargs):
        return fake_body

    with patch.object(obs_mod, "_post_chat_completion", new=fake_post):
        decision = run_async(
            obs_mod._call_observer(
                user_prompt="hi", allow_deny_tool=False, cfg=obs_mod._observer_cfg(),
            )
        )
    assert decision.action == "inject", decision
    assert decision.content == "answer the user", decision
    assert decision.input_tokens == 100, decision
    assert decision.output_tokens == 20, decision
    print("test_call_observer_parses_response: OK")


def test_call_observer_timeout_falls_back_to_noop():
    async def fake_post(**kwargs):
        raise asyncio.TimeoutError()

    with patch.object(obs_mod, "_post_chat_completion", new=fake_post):
        decision = run_async(
            obs_mod._call_observer(
                user_prompt="hi", allow_deny_tool=False, cfg=obs_mod._observer_cfg(),
            )
        )
    assert decision.action == "noop", decision
    assert decision.error == "timeout", decision
    print("test_call_observer_timeout_falls_back_to_noop: OK")


def test_call_observer_pretool_timeout_fails_open():
    async def fake_post(**kwargs):
        raise asyncio.TimeoutError()

    with patch.object(obs_mod, "_post_chat_completion", new=fake_post):
        decision = run_async(
            obs_mod._call_observer(
                user_prompt="hi",
                allow_deny_tool=True,
                cfg=obs_mod._observer_cfg(),
            )
        )
    assert decision.action == "allow", decision
    assert decision.error == "timeout", decision
    print("test_call_observer_pretool_timeout_fails_open: OK")


# ---------------------------------------------------------------------------
# install_observer end-to-end (mocked HTTP)
# ---------------------------------------------------------------------------


def test_install_observer_pretool_deny():
    """Simulate the harness firing PreToolUse → observer denies a destructive Bash."""
    fake_body = {
        "choices": [
            {"message": {"content": '"deny_tool","reason":"rm -rf on user data","content":"refusing"}'}}
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 15},
    }

    async def fake_post(**kwargs):
        return fake_body

    hooks = HookRegistry()
    chat = []
    cancel = asyncio.Event()
    state = install_observer(
        hooks=hooks,
        session_id="test_sess",
        turn_id="test_turn_pretool",
        user_request="clean some files",
        chat_messages_handle=chat,
        cancel_event=cancel,
        primary_model="primary",
    )
    with patch.object(obs_mod, "_post_chat_completion", new=fake_post):
        out = run_async(
            hooks.fire_pre_tool_use(
                session_id="test_sess",
                tool_name="Bash",
                tool_input={"command": "rm -rf /home/alansrobotlab/lloyd"},
            )
        )
    assert out, out
    hso = out.get("hookSpecificOutput") or {}
    assert hso.get("permissionDecision") == "deny", hso
    assert "refusing" in (hso.get("permissionDecisionReason") or ""), hso
    # PreToolUse denials don't count against budget
    assert state.interventions_used == 0
    print("test_install_observer_pretool_deny: OK")


def test_install_observer_assistant_message_inject():
    """Simulate harness firing OnEvent for an empty assistant_message →
    observer injects a system message into chat history."""
    fake_body = {
        "choices": [
            {"message": {"content": '"inject","reason":"empty terminal","content":"answer the user"}'}}
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 15},
    }

    async def fake_post(**kwargs):
        return fake_body

    hooks = HookRegistry()
    chat = []
    cancel = asyncio.Event()
    state = install_observer(
        hooks=hooks,
        session_id="test_sess",
        turn_id="test_turn_inject",
        user_request="what's 2+2?",
        chat_messages_handle=chat,
        cancel_event=cancel,
        primary_model="primary",
    )
    evt = {
        "type": "assistant_message",
        "text": "",
        "tool_calls": [],
        "iteration": 3,
    }
    with patch.object(obs_mod, "_post_chat_completion", new=fake_post):
        run_async(hooks.fire_on_event(evt))
    assert len(chat) == 1, chat
    assert chat[0]["role"] == "user", chat[0]
    assert "answer the user" in chat[0]["content"]
    assert state.interventions_used == 1
    print("test_install_observer_assistant_message_inject: OK")


# ---------------------------------------------------------------------------
# Goal extraction
# ---------------------------------------------------------------------------


def test_parse_goal_card_full():
    raw = '{"success_criteria":["a","b"],"out_of_scope":["c"],"completion_signals":["d"]}'
    p = _parse_goal_card(raw)
    assert p == {
        "success_criteria": ["a", "b"],
        "out_of_scope": ["c"],
        "completion_signals": ["d"],
    }, p
    print("test_parse_goal_card_full: OK")


def test_parse_goal_card_with_prefill():
    # Model honored prefill; raw is the continuation
    raw = '["a"],"out_of_scope":[],"completion_signals":["d"]}'
    p = _parse_goal_card(raw)
    assert p is not None
    assert p["success_criteria"] == ["a"], p
    print("test_parse_goal_card_with_prefill: OK")


def test_parse_goal_card_garbage():
    p = _parse_goal_card("nonsense")
    assert p is None
    print("test_parse_goal_card_garbage: OK")


def test_extract_goal_card_calls_llm():
    fake_body = {
        "choices": [
            {"message": {"content": '["check disk","check services"],"out_of_scope":[],"completion_signals":["all checks reported"]}'}}
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 30},
    }

    async def fake_post(**kwargs):
        return fake_body

    with patch.object(obs_mod, "_post_chat_completion", new=fake_post):
        gc = run_async(extract_goal_card("Run a full systems check please"))
    assert gc is not None
    assert gc["success_criteria"] == ["check disk", "check services"]
    assert gc["completion_signals"] == ["all checks reported"]
    print("test_extract_goal_card_calls_llm: OK")


def test_extract_goal_card_skips_empty_request():
    gc = run_async(extract_goal_card(""))
    assert gc is None
    gc2 = run_async(extract_goal_card("   "))
    assert gc2 is None
    print("test_extract_goal_card_skips_empty_request: OK")


def test_extract_goal_card_returns_none_on_error():
    async def fake_post(**kwargs):
        raise asyncio.TimeoutError()

    with patch.object(obs_mod, "_post_chat_completion", new=fake_post):
        gc = run_async(extract_goal_card("any request"))
    assert gc is None
    print("test_extract_goal_card_returns_none_on_error: OK")


def test_event_prompt_includes_goal_card():
    state = _make_state(goal_card={
        "success_criteria": ["report disk space"],
        "out_of_scope": [],
        "completion_signals": ["disk reported"],
    })
    prompt = _build_event_user_prompt(state, "iter 1 in progress")
    assert "GOAL CARD:" in prompt
    assert "report disk space" in prompt
    assert "disk reported" in prompt
    print("test_event_prompt_includes_goal_card: OK")


def test_event_prompt_handles_missing_goal_card():
    state = _make_state(goal_card=None)
    prompt = _build_event_user_prompt(state, "iter 1 in progress")
    assert "GOAL CARD:" in prompt
    assert "no actionable goal" in prompt
    print("test_event_prompt_handles_missing_goal_card: OK")


def test_tool_result_summary_detects_spilled_payload():
    """When a tool result is a `<persisted-output>` envelope, the summary
    should explicitly name the size + path so the observer can judge
    progress against the data, not nag about reading more."""
    from app.inner_voice.observer_prompt import build_tool_result_summary
    spilled = (
        "<persisted-output>\n"
        "Output too large (68.3 KB, 69,943 chars). Full output saved to: /tmp/test/spill.json\n\n"
        "Preview (first 2.0 KB):\n"
        '{"entity": "lloyd", "category": "overview"...\n'
        "...\n"
        "Read the full file with the Read tool if you need more than the preview.\n"
    )
    summary = build_tool_result_summary("Read", spilled, is_error=False)
    assert "SPILLED" in summary, summary
    assert "68.3 KB" in summary, summary
    assert "/tmp/test/spill.json" in summary, summary
    assert "lloyd" in summary, "preview opener should be visible"
    print("test_tool_result_summary_detects_spilled_payload: OK")


def test_tool_result_summary_falls_back_for_normal_result():
    """Normal small results use the existing format."""
    from app.inner_voice.observer_prompt import build_tool_result_summary
    summary = build_tool_result_summary("Bash", "hello\nworld\n", is_error=False)
    assert "SPILLED" not in summary
    assert "result" in summary
    assert "hello" in summary
    print("test_tool_result_summary_falls_back_for_normal_result: OK")


def test_tool_result_summary_keeps_error_path_for_errors():
    """Even if an error message contains the spill tag, the error path wins
    (we always want the observer to see ERROR for error results)."""
    from app.inner_voice.observer_prompt import build_tool_result_summary
    summary = build_tool_result_summary("Bash", "Permission denied", is_error=True)
    assert "ERROR" in summary
    print("test_tool_result_summary_keeps_error_path_for_errors: OK")


# ---------------------------------------------------------------------------
# Mid-turn microcompaction (harness ↔ observer shared list contract)
# ---------------------------------------------------------------------------


def test_intra_turn_microcompact_clears_in_place():
    """The harness mutates chat_messages in-place (chat_messages[:] = ...)
    so the observer's shared list reference still sees the cleared
    results. This pins that contract.
    """
    from app.harness.microcompact import microcompact, CLEARED_MARKER
    # Build a chat with 18 compactable tool results (>15 threshold).
    msgs: list[dict] = []
    for i in range(18):
        msgs.append({
            "role": "assistant",
            "tool_calls": [{"id": f"tc{i}", "function": {"name": "Read"}}],
        })
        msgs.append({
            "role": "tool",
            "tool_call_id": f"tc{i}",
            "content": f"file content {i}",
        })
    handle = msgs  # observer would hold this reference
    original_id = id(handle)
    compacted, cleared = microcompact(handle, keep_recent_tools=5, count_threshold=15)
    assert cleared >= 13, cleared  # 18 - 5 kept = 13 cleared
    # Simulate the loop's in-place replace.
    handle[:] = compacted
    # The observer's reference is still valid AND sees the cleared content.
    assert id(handle) == original_id, "in-place replace must preserve list identity"
    cleared_msgs = [m for m in handle if m.get("role") == "tool" and CLEARED_MARKER in str(m.get("content", ""))]
    assert len(cleared_msgs) == cleared
    # Most recent 5 stay inline.
    recent_msgs = [m for m in handle if m.get("role") == "tool" and "file content" in str(m.get("content", ""))]
    assert len(recent_msgs) == 5, recent_msgs
    print("test_intra_turn_microcompact_clears_in_place: OK")


def test_intra_turn_microcompact_skips_under_threshold():
    """Under the threshold, microcompact does nothing (no spill markers)."""
    from app.harness.microcompact import microcompact
    msgs: list[dict] = []
    for i in range(5):  # 5 << 15 threshold
        msgs.append({
            "role": "assistant",
            "tool_calls": [{"id": f"tc{i}", "function": {"name": "Read"}}],
        })
        msgs.append({
            "role": "tool",
            "tool_call_id": f"tc{i}",
            "content": f"file content {i}",
        })
    compacted, cleared = microcompact(msgs, keep_recent_tools=5, count_threshold=15)
    assert cleared == 0
    assert compacted == msgs  # unchanged
    print("test_intra_turn_microcompact_skips_under_threshold: OK")


# ---------------------------------------------------------------------------
# Tiered triggering — fast-path
# ---------------------------------------------------------------------------


def test_fast_path_pretool_allows_read_glob_grep():
    for tool in ("Read", "Glob", "Grep"):
        d = _fast_path_pretool(tool, {"path": "/etc/hosts"})
        assert d is not None
        assert d.action == "allow", (tool, d)
    print("test_fast_path_pretool_allows_read_glob_grep: OK")


def test_fast_path_pretool_allows_safe_bash():
    safe_cmds = [
        "ls -la /tmp",
        "cat /etc/hostname",
        "df -h",
        "echo hello",
        "date -u",
        "pwd",
        "whoami",
        "grep -r foo /tmp",
        "find /tmp -type f",
    ]
    for cmd in safe_cmds:
        d = _fast_path_pretool("Bash", {"command": cmd})
        assert d is not None and d.action == "allow", (cmd, d)
    print("test_fast_path_pretool_allows_safe_bash: OK")


def test_fast_path_pretool_escalates_destructive_bash():
    destructive_cmds = [
        "rm -rf /tmp/foo",
        "mv /etc/passwd /tmp",
        "chmod 777 /etc",
        "sudo systemctl restart x",
        "git reset --hard HEAD~1",
        "git push --force origin main",
        "dd if=/dev/zero of=/tmp/junk",
        "echo hello > /etc/hostname",  # redirect to non-/tmp
        "cat malicious.sh | bash",
    ]
    for cmd in destructive_cmds:
        d = _fast_path_pretool("Bash", {"command": cmd})
        assert d is None, f"should escalate: {cmd!r}"
    print("test_fast_path_pretool_escalates_destructive_bash: OK")


def test_bash_safety_classifier():
    assert _bash_command_is_safely_readonly("ls") is True
    assert _bash_command_is_safely_readonly("ls /tmp") is True
    assert _bash_command_is_safely_readonly("") is True  # empty
    assert _bash_command_is_safely_readonly("# comment only") is True
    assert _bash_command_is_safely_readonly("rm /tmp/x") is False
    assert _bash_command_is_safely_readonly("ls && rm foo") is False  # has rm
    assert _bash_command_is_safely_readonly("python3 script.py") is False
    print("test_bash_safety_classifier: OK")


def test_fast_path_pretool_allows_safe_mcp():
    d = _fast_path_pretool("mcp__lloyd-mcp__fact_get", {"entity": "lloyd"})
    assert d is not None and d.action == "allow", d
    d2 = _fast_path_pretool("mcp__lloyd-mcp__memory_search", {"query": "x"})
    assert d2 is not None and d2.action == "allow", d2
    print("test_fast_path_pretool_allows_safe_mcp: OK")


def test_fast_path_pretool_escalates_unknown_mcp():
    # No safe verb in name → escalate
    d = _fast_path_pretool("mcp__lloyd-mcp__autonomy_write_task", {"title": "x"})
    assert d is None, d
    d2 = _fast_path_pretool("mcp__lloyd-mcp__memory_add", {"text": "x"})
    assert d2 is None, d2
    print("test_fast_path_pretool_escalates_unknown_mcp: OK")


def test_fast_path_tool_result_skips_small_benign():
    d = _fast_path_tool_result("Bash", "hello\n", is_error=False)
    assert d is not None and d.action == "noop", d
    print("test_fast_path_tool_result_skips_small_benign: OK")


def test_fast_path_tool_result_skips_parse_errors():
    d = _fast_path_tool_result(
        "Bash",
        "Tool call arguments could not be parsed as JSON: Extra data: line 1 column 5",
        is_error=True,
    )
    assert d is not None and d.action == "noop", d
    print("test_fast_path_tool_result_skips_parse_errors: OK")


def test_fast_path_tool_result_escalates_real_errors():
    d = _fast_path_tool_result("Bash", "Permission denied\n", is_error=True)
    assert d is None, d
    print("test_fast_path_tool_result_escalates_real_errors: OK")


def test_fast_path_tool_result_escalates_large_results():
    big = "x" * 5000
    d = _fast_path_tool_result("Bash", big, is_error=False)
    assert d is None, "large results should escalate to LLM"
    print("test_fast_path_tool_result_escalates_large_results: OK")


# ---------------------------------------------------------------------------
# Cross-event memory
# ---------------------------------------------------------------------------


def test_event_prompt_includes_prior_decisions():
    state = _make_state()
    state.decisions_this_turn = [
        {"trigger": "assistant_message", "action": "inject", "reason": "stay focused", "related_tool": None},
        {"trigger": "tool_result", "action": "noop", "reason": "primary acknowledged", "related_tool": "Bash"},
    ]
    prompt = _build_event_user_prompt(state, "iter 3 in progress")
    assert "YOUR PRIOR DECISIONS THIS TURN:" in prompt
    assert "stay focused" in prompt
    assert "primary acknowledged" in prompt
    assert "[Bash]" in prompt  # related_tool annotation
    print("test_event_prompt_includes_prior_decisions: OK")


def test_event_prompt_omits_decisions_section_when_empty():
    state = _make_state()
    state.decisions_this_turn = []
    prompt = _build_event_user_prompt(state, "iter 1 in progress")
    assert "YOUR PRIOR DECISIONS" not in prompt
    print("test_event_prompt_omits_decisions_section_when_empty: OK")


# ---------------------------------------------------------------------------
# Clarify lever
# ---------------------------------------------------------------------------


def test_clarify_fires_callback_and_cancels():
    captured = {}

    async def clarify_cb(content, reason):
        captured["content"] = content
        captured["reason"] = reason

    state = _make_state(clarify_callback=clarify_cb)
    decision = ObserverDecision(action="clarify", reason="ambiguous", content="A or B?")
    run_async(_apply_lever(state, decision, trigger="assistant_message"))
    assert captured["content"] == "A or B?"
    assert state.cancel_event.is_set(), "clarify must set cancel_event to pause primary"
    assert state.interventions_used == 1
    print("test_clarify_fires_callback_and_cancels: OK")


def test_clarify_with_no_callback_degrades():
    state = _make_state(clarify_callback=None)
    decision = ObserverDecision(action="clarify", reason="ambiguous", content="A or B?")
    run_async(_apply_lever(state, decision, trigger="assistant_message"))
    assert decision.action == "noop_no_clarify_channel"
    assert not state.cancel_event.is_set()
    assert state.interventions_used == 0
    print("test_clarify_with_no_callback_degrades: OK")


def test_clarify_on_result_degrades():
    """Asking a question after the turn ended is nonsensical."""
    async def cb(c, r):
        pass

    state = _make_state(clarify_callback=cb)
    decision = ObserverDecision(action="clarify", reason="x", content="?")
    run_async(_apply_lever(state, decision, trigger="result"))
    assert decision.action == "noop_clarify_on_result"
    assert not state.cancel_event.is_set()
    assert state.interventions_used == 0
    print("test_clarify_on_result_degrades: OK")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


TESTS = [
    test_parse_full_object,
    test_parse_with_prefill,
    test_parse_embedded,
    test_parse_garbage,
    test_parse_bare_word,
    test_parse_dropped_opening_quote,
    test_normalize_noop_path,
    test_normalize_pretool_path,
    test_inject_appends_user_message,
    test_cancel_sets_event,
    test_cancel_works_when_budget_exhausted,
    test_ambient_fires_callback,
    test_ambient_with_no_callback_degrades_to_noop,
    test_inject_empty_content_degrades_to_noop,
    test_inject_on_result_translates_to_ambient_when_callback_present,
    test_inject_on_result_degrades_when_no_callback,
    test_cancel_on_result_degrades_to_noop,
    test_budget_exhausted,
    test_call_observer_parses_response,
    test_call_observer_timeout_falls_back_to_noop,
    test_call_observer_pretool_timeout_fails_open,
    test_install_observer_pretool_deny,
    test_install_observer_assistant_message_inject,
    # Goal extraction
    test_parse_goal_card_full,
    test_parse_goal_card_with_prefill,
    test_parse_goal_card_garbage,
    test_extract_goal_card_calls_llm,
    test_extract_goal_card_skips_empty_request,
    test_extract_goal_card_returns_none_on_error,
    test_event_prompt_includes_goal_card,
    test_event_prompt_handles_missing_goal_card,
    test_tool_result_summary_detects_spilled_payload,
    test_tool_result_summary_falls_back_for_normal_result,
    test_tool_result_summary_keeps_error_path_for_errors,
    test_intra_turn_microcompact_clears_in_place,
    test_intra_turn_microcompact_skips_under_threshold,
    # Tiered triggering
    test_fast_path_pretool_allows_read_glob_grep,
    test_fast_path_pretool_allows_safe_bash,
    test_fast_path_pretool_escalates_destructive_bash,
    test_bash_safety_classifier,
    test_fast_path_pretool_allows_safe_mcp,
    test_fast_path_pretool_escalates_unknown_mcp,
    test_fast_path_tool_result_skips_small_benign,
    test_fast_path_tool_result_skips_parse_errors,
    test_fast_path_tool_result_escalates_real_errors,
    test_fast_path_tool_result_escalates_large_results,
    # Cross-event memory
    test_event_prompt_includes_prior_decisions,
    test_event_prompt_omits_decisions_section_when_empty,
    # Clarify
    test_clarify_fires_callback_and_cancels,
    test_clarify_with_no_callback_degrades,
    test_clarify_on_result_degrades,
]


def main() -> int:
    failed = 0
    for t in TESTS:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {e!r}")
            failed += 1
    print()
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
