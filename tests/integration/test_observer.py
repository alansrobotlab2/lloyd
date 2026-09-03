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
    _STUB_ANNOUNCE_RE,
    _build_event_user_prompt,
    _extract_tool_call,
    _fast_path_assistant_message,
    _fast_path_pretool,
    _fast_path_tool_result,
    extract_goal_card,
    install_observer,
)


# ---------------------------------------------------------------------------
# Isolation: never write to the production usage.db.
#
# `_persist` calls `record_inner_voice_observation` for real, so before this
# every test run appended rows to ~/lloyd/usage.db under the fake session ids
# below — polluting exactly the table `scripts/iv_grade.py` reads to judge the
# subsystem. Rows are captured in memory instead; assert on them if useful.
# ---------------------------------------------------------------------------

RECORDED: list = []


def _no_db_record(**kwargs):
    RECORDED.append(kwargs)
    return len(RECORDED)


obs_mod.record_inner_voice_observation = _no_db_record


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    *,
    ambient_callback=None,
    clarify_callback=None,
    persist_intervention_callback=None,
    budget=3,
    goal_card=None,
) -> ObserverState:
    state = ObserverState(
        session_id="test_sess",
        turn_id="test_turn",
        user_request="What's 2+2?",
        chat_messages_handle=[],
        cancel_event=asyncio.Event(),
        enqueue_ambient_callback=ambient_callback,
        clarify_callback=clarify_callback,
        persist_intervention_callback=persist_intervention_callback,
        primary_model="primary",
        intervention_budget=budget,
        goal_card=goal_card,
    )
    return state


# ---------------------------------------------------------------------------
# Tool-call extraction (v4)
# ---------------------------------------------------------------------------


def test_extract_tool_call_well_formed():
    body = {
        "choices": [{
            "message": {
                "tool_calls": [{
                    "function": {
                        "name": "inject",
                        "arguments": '{"reason":"loop","content":"Try Grep instead."}',
                    },
                }],
            },
        }],
    }
    extracted = _extract_tool_call(body)
    assert extracted is not None
    name, args = extracted
    assert name == "inject"
    assert args["content"] == "Try Grep instead."
    print("test_extract_tool_call_well_formed: OK")


def test_extract_tool_call_dict_arguments():
    """Some vLLM versions return arguments as a dict, not a JSON string."""
    body = {
        "choices": [{
            "message": {
                "tool_calls": [{"function": {"name": "noop", "arguments": {"reason": "ok"}}}],
            },
        }],
    }
    extracted = _extract_tool_call(body)
    assert extracted == ("noop", {"reason": "ok"})
    print("test_extract_tool_call_dict_arguments: OK")


def test_extract_tool_call_no_tool_calls():
    body = {"choices": [{"message": {"content": "text only"}}]}
    assert _extract_tool_call(body) is None
    print("test_extract_tool_call_no_tool_calls: OK")


def test_extract_tool_call_bad_args_json():
    body = {
        "choices": [{
            "message": {
                "tool_calls": [{"function": {"name": "inject", "arguments": "{not-json}"}}],
            },
        }],
    }
    assert _extract_tool_call(body) is None
    print("test_extract_tool_call_bad_args_json: OK")


def test_extract_tool_call_empty_choices():
    assert _extract_tool_call({}) is None
    assert _extract_tool_call({"choices": []}) is None
    print("test_extract_tool_call_empty_choices: OK")


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


def test_inject_invokes_persist_intervention_callback():
    """When inject fires, the persist callback gets ('inject', content, reason).
    This makes the intervention visible in the session JSON and chat UI."""
    persisted = []

    async def persist_cb(kind, content, reason):
        persisted.append((kind, content, reason))

    state = _make_state(persist_intervention_callback=persist_cb)
    decision = ObserverDecision(action="inject", reason="drift", content="please answer")
    run_async(_apply_lever(state, decision, trigger="assistant_message"))
    assert persisted == [("inject", "please answer", "drift")], persisted
    # Transient chat_messages_handle still got the in-memory copy.
    assert len(state.chat_messages_handle) == 1
    print("test_inject_invokes_persist_intervention_callback: OK")


def test_cancel_invokes_persist_intervention_callback():
    """When cancel fires, the persist callback gets ('cancel', '', reason).
    Gives the user a visible explanation of why the turn stopped."""
    persisted = []

    async def persist_cb(kind, content, reason):
        persisted.append((kind, content, reason))

    state = _make_state(persist_intervention_callback=persist_cb)
    decision = ObserverDecision(action="cancel", reason="stuck in loop")
    run_async(_apply_lever(state, decision, trigger="assistant_message"))
    assert persisted == [("cancel", "", "stuck in loop")], persisted
    assert state.cancel_event.is_set()
    print("test_cancel_invokes_persist_intervention_callback: OK")


def test_inject_persist_failure_does_not_break_lever():
    """If the persist callback raises, the intervention still fires (the
    in-memory chat_messages_handle still gets the inject) — persistence is
    best-effort breadcrumb only."""

    async def broken_persist(kind, content, reason):
        raise RuntimeError("disk full")

    state = _make_state(persist_intervention_callback=broken_persist)
    decision = ObserverDecision(action="inject", reason="x", content="msg")
    run_async(_apply_lever(state, decision, trigger="assistant_message"))
    # Intervention still landed in the live chat buffer.
    assert len(state.chat_messages_handle) == 1
    assert state.interventions_used == 1
    print("test_inject_persist_failure_does_not_break_lever: OK")


def test_cancel_persist_failure_does_not_break_lever():
    """If the persist callback raises on cancel, the cancel still fires
    (cancel_event still gets set)."""

    async def broken_persist(kind, content, reason):
        raise RuntimeError("disk full")

    state = _make_state(persist_intervention_callback=broken_persist)
    decision = ObserverDecision(action="cancel", reason="x")
    run_async(_apply_lever(state, decision, trigger="assistant_message"))
    assert state.cancel_event.is_set()
    print("test_cancel_persist_failure_does_not_break_lever: OK")


def test_ambient_fires_callback():
    captured = {}

    async def amb_cb(content, reason, producer="inner_voice"):
        captured["content"] = content
        captured["reason"] = reason
        captured["producer"] = producer

    state = _make_state(ambient_callback=amb_cb)
    decision = ObserverDecision(action="ambient", reason="follow up", content="check on this later")
    run_async(_apply_lever(state, decision, trigger="result"))
    assert captured["content"] == "check on this later"
    # A discretionary IV ambient is tagged `inner_voice`, which the attach
    # gate refuses to observe — only `inner_voice_goal` is self-observed.
    assert captured["producer"] == "inner_voice", captured
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

    async def amb_cb(content, reason, producer="inner_voice"):
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
# Terminal stall rescue — the "stop stopping" regression
# (text-only "Let me …:" iterations that end the turn with work undone)
# ---------------------------------------------------------------------------


def test_stub_announce_regex_matches_real_stalls():
    """Patterns lifted verbatim from the 20260602_204609_iv1734 stall storm."""
    stalls = [
        "The invalidate didn't fully take — they still have active facts. Let me force-expire them:",
        "Let me check what's actually still active after the previous invalidation batch:",
        "Those are clean. Let me batch-check the rest:",
        "I'll go ahead and remove the noise entities.",
    ]
    completes = [
        "Done. I removed 10 noise session entities; the 4 legitimate facts remain intact.",
        "The root cause is the nightly pipeline: it extracts a session entity per run, which is noise.",
        "Here is the complete list of session-prefixed facts: session_pong5, session-distill.",
    ]
    for s in stalls:
        assert _STUB_ANNOUNCE_RE.search(s.strip()), f"should flag stall: {s!r}"
    for s in completes:
        assert not _STUB_ANNOUNCE_RE.search(s.strip()), f"should NOT flag complete: {s!r}"
    print("test_stub_announce_regex_matches_real_stalls: OK")


def test_fast_path_flags_terminal_stub_announce_as_inject():
    """A text-only 'Let me …:' iteration → deterministic budget-bypassing inject."""
    fp = _fast_path_assistant_message("Let me force-expire them:", [])
    assert fp is not None and fp.action == "inject", fp
    assert fp.bypass_budget is True
    assert fp.content.strip()
    print("test_fast_path_flags_terminal_stub_announce_as_inject: OK")


def test_fast_path_ignores_completed_answer():
    """A delivered answer (no announce) is left to terminate naturally."""
    fp = _fast_path_assistant_message(
        "Done — removed 10 noise entities; legitimate session facts intact.", []
    )
    assert fp is None, fp
    print("test_fast_path_ignores_completed_answer: OK")


def test_fast_path_tool_dispatch_only_still_noops():
    """Existing behavior preserved: text-less tool dispatch → noop, not a stall."""
    fp = _fast_path_assistant_message("", [{"function": {"name": "Bash"}}])
    assert fp is not None and fp.action == "noop", fp
    print("test_fast_path_tool_dispatch_only_still_noops: OK")


def test_stall_rescue_inject_bypasses_budget():
    """A stall-rescue inject must continue the loop even after the discretionary
    budget is spent — otherwise a multi-stall storm ends the turn with work
    undone (the original 'stop stopping' failure)."""
    state = _make_state(budget=1)
    spent = ObserverDecision(action="inject", reason="nudge", content="first")
    run_async(_apply_lever(state, spent, trigger="assistant_message"))
    assert state.interventions_used == 1
    # Budget is now exhausted; a normal inject would be downgraded.
    rescue = ObserverDecision(
        action="inject", reason="stall", content="keep going", bypass_budget=True
    )
    run_async(_apply_lever(state, rescue, trigger="assistant_message"))
    assert rescue.action == "inject", "stall-rescue must not be budget-downgraded"
    assert len(state.chat_messages_handle) == 2, "stall-rescue must append to continue loop"
    print("test_stall_rescue_inject_bypasses_budget: OK")


# ---------------------------------------------------------------------------
# Observer call (with mocked HTTP)
# ---------------------------------------------------------------------------


def _fake_tool_call_body(name: str, args: dict, *, in_tok: int = 100, out_tok: int = 20) -> dict:
    """Build a fake vLLM tool-call response body for tests."""
    import json as _j
    return {
        "choices": [{
            "message": {
                "content": "",
                "tool_calls": [{
                    "id": "call_test",
                    "type": "function",
                    "function": {"name": name, "arguments": _j.dumps(args)},
                }],
            },
        }],
        "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok},
    }


def test_call_observer_parses_tool_call():
    body = _fake_tool_call_body("inject", {
        "reason": "empty terminal", "content": "answer the user",
    })

    async def fake_post(**kwargs):
        return body

    with patch.object(obs_mod, "_post_chat_completion_with_tools", new=fake_post):
        decision = run_async(
            obs_mod._call_observer(user_prompt="hi", cfg=obs_mod._observer_cfg())
        )
    assert decision.action == "inject", decision
    assert decision.content == "answer the user", decision
    assert decision.input_tokens == 100, decision
    assert decision.output_tokens == 20, decision
    print("test_call_observer_parses_tool_call: OK")


def test_call_observer_timeout_falls_back_to_noop():
    """Both timeout flavors fold to noop and are LABELED as timeouts.

    httpx raises TimeoutException, which subclasses httpx.HTTPError — so
    an `except httpx.HTTPError` ahead of the timeout branch swallows every
    deadline and files it as a generic transport failure. That is what v4
    did, and it is why all five deadline hits in the first production
    window recorded `http_error: ` with an empty message.
    """
    import httpx

    for exc in (asyncio.TimeoutError(), httpx.ReadTimeout("deadline")):
        async def fake_post(**kwargs):
            raise exc

        with patch.object(obs_mod, "_post_chat_completion_with_tools", new=fake_post):
            decision = run_async(
                obs_mod._call_observer(user_prompt="hi", cfg=obs_mod._observer_cfg())
            )
        assert decision.action == "noop", decision
        assert decision.error and decision.error.startswith("timeout"), (exc, decision)
    print("test_call_observer_timeout_falls_back_to_noop: OK")


def test_call_observer_no_tool_call_falls_back_to_noop():
    """If vLLM returns no tool_call (parser quirk), we noop with error tag."""
    body = {"choices": [{"message": {"content": "weird text"}}]}

    async def fake_post(**kwargs):
        return body

    with patch.object(obs_mod, "_post_chat_completion_with_tools", new=fake_post):
        decision = run_async(
            obs_mod._call_observer(user_prompt="hi", cfg=obs_mod._observer_cfg())
        )
    assert decision.action == "noop", decision
    assert decision.error == "no_tool_call", decision
    print("test_call_observer_no_tool_call_falls_back_to_noop: OK")


def test_call_observer_unknown_lever_coerces_noop():
    """If vLLM somehow returns a tool name outside LEVER_TOOLS, fall back."""
    body = _fake_tool_call_body("deny_tool", {"reason": "old habits"})

    async def fake_post(**kwargs):
        return body

    with patch.object(obs_mod, "_post_chat_completion_with_tools", new=fake_post):
        decision = run_async(
            obs_mod._call_observer(user_prompt="hi", cfg=obs_mod._observer_cfg())
        )
    assert decision.action == "noop", decision
    assert decision.error == "unknown_lever", decision
    print("test_call_observer_unknown_lever_coerces_noop: OK")


# ---------------------------------------------------------------------------
# install_observer end-to-end (mocked HTTP)
# ---------------------------------------------------------------------------


def test_install_observer_pretool_does_not_block():
    """Pretool never blocks dispatch, and by default costs no LLM call.

    Two separate contracts:

    * v4 removed `deny_tool` — hard safety moved to `app/harness/safety.py`
      and the pretool callback must return `{}` no matter what IV decided.
    * v5 turns the pretool LLM judgment OFF by default. Since the observer
      cannot block the call, an inject here only reaches the primary AFTER
      the tool has already run, by which point `tool_result` sees the same
      call plus its outcome. In the first production window pretool was 45%
      of all observer input tokens for three interventions.

    The observation ROW is still written either way — the prior-decisions
    log and the mark-without-evidence check both read tool activity from it.
    """
    body = _fake_tool_call_body("inject", {
        "reason": "off-task tool",
        "content": "Use Read to inspect the file instead of running migrations.",
    })
    calls = []

    async def fake_post(**kwargs):
        calls.append(kwargs)
        return body

    def _fire(cfg_overrides):
        cfg = obs_mod._observer_cfg()
        cfg.update(cfg_overrides)
        hooks = HookRegistry()
        chat = []
        with patch.object(obs_mod, "_observer_cfg", return_value=cfg):
            install_observer(
                hooks=hooks,
                session_id="test_sess",
                turn_id="test_turn_pretool_v5",
                user_request="inspect framework code",
                chat_messages_handle=chat,
                cancel_event=asyncio.Event(),
                primary_model="primary",
            )
            with patch.object(obs_mod, "_post_chat_completion_with_tools", new=fake_post):
                out = run_async(
                    hooks.fire_pre_tool_use(
                        session_id="test_sess",
                        tool_name="Bash",
                        tool_input={"command": "python migrations/run.py"},
                    )
                )
        return out, chat

    # Default: observation only. No LLM call, nothing appended, no block.
    calls.clear()
    out, chat = _fire({"pretool_llm_enabled": False})
    assert out == {}, out
    assert calls == [], "pretool must not call the LLM when disabled"
    assert chat == [], chat

    # Opted in: the LLM runs and its inject lands, but dispatch is still
    # never blocked. Synchronous so the assertion doesn't race the task.
    calls.clear()
    out, chat = _fire({"pretool_llm_enabled": True, "async_nonterminal": False})
    assert out == {}, "pretool must never return a deny dict"
    assert len(calls) == 1, calls
    assert any("[INNER VOICE]" in (m.get("content") or "") for m in chat), chat
    print("test_install_observer_pretool_does_not_block: OK")


def test_install_observer_assistant_message_inject():
    """Simulate harness firing OnEvent for an empty assistant_message →
    observer injects a user message into chat history."""
    fake_body = _fake_tool_call_body("inject", {
        "reason": "empty terminal", "content": "answer the user",
    }, in_tok=50, out_tok=15)

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
    with patch.object(obs_mod, "_post_chat_completion_with_tools", new=fake_post):
        run_async(hooks.fire_on_event(evt))
    assert len(chat) == 1, chat
    assert chat[0]["role"] == "user", chat[0]
    assert "answer the user" in chat[0]["content"]
    assert state.interventions_used == 1
    print("test_install_observer_assistant_message_inject: OK")


# ---------------------------------------------------------------------------
# Goal extraction
# ---------------------------------------------------------------------------


def test_extract_goal_card_calls_llm():
    fake_body = _fake_tool_call_body("record_goal_card", {
        "success_criteria": ["check disk", "check services"],
        "out_of_scope": [],
        "completion_signals": ["all checks reported"],
    }, in_tok=100, out_tok=30)

    async def fake_post(**kwargs):
        return fake_body

    with patch.object(obs_mod, "_post_chat_completion_with_tools", new=fake_post):
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

    with patch.object(obs_mod, "_post_chat_completion_with_tools", new=fake_post):
        gc = run_async(extract_goal_card("any request"))
    assert gc is None
    print("test_extract_goal_card_returns_none_on_error: OK")


def test_extract_goal_card_returns_none_when_no_tool_call():
    """If vLLM returned no tool_call (parser quirk), goal extraction
    falls back to None and observer runs in lighter-touch mode."""
    body = {"choices": [{"message": {"content": "weird text"}}]}

    async def fake_post(**kwargs):
        return body

    with patch.object(obs_mod, "_post_chat_completion_with_tools", new=fake_post):
        gc = run_async(extract_goal_card("Run a full systems check please"))
    assert gc is None
    print("test_extract_goal_card_returns_none_when_no_tool_call: OK")


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


def test_fast_path_pretool_noops_read_glob_grep():
    for tool in ("Read", "Glob", "Grep"):
        d = _fast_path_pretool(tool, {"path": "/etc/hosts"})
        assert d is not None
        assert d.action == "noop", (tool, d)
    print("test_fast_path_pretool_noops_read_glob_grep: OK")


def test_fast_path_pretool_noops_safe_bash():
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
        # git read subcommands: 221 of 264 pretool LLM calls in the first
        # production window were Bash, and status/log/diff had no entry at
        # all, so each one paid for a full observer round-trip.
        "git status",
        "git log --oneline -5",
        "git diff HEAD",
    ]
    for cmd in safe_cmds:
        d = _fast_path_pretool("Bash", {"command": cmd})
        assert d is not None and d.action == "noop", (cmd, d)
    print("test_fast_path_pretool_noops_safe_bash: OK")


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
        # `find` with an action predicate is not a read.
        "find . -delete",
        "find /tmp -exec rm {} +",
        # wget with no flags writes the fetched file into cwd; curl with
        # an output flag or a mutating method is a write, not a fetch.
        "wget http://example.com/a.sh",
        "curl -o /tmp/f http://example.com",
        "curl -X POST http://example.com -d a=1",
        # git subcommands that change state.
        "git commit -m wip",
        "git checkout main",
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
    assert _bash_command_is_safely_readonly("git status") is True
    assert _bash_command_is_safely_readonly("git rev-parse HEAD") is True
    assert _bash_command_is_safely_readonly("git commit -m x") is False
    assert _bash_command_is_safely_readonly("git") is False  # bare, no subcommand
    assert _bash_command_is_safely_readonly("find . -delete") is False
    assert _bash_command_is_safely_readonly("wget http://x/a") is False
    assert _bash_command_is_safely_readonly("curl -s http://x/api") is True
    assert _bash_command_is_safely_readonly("curl -X DELETE http://x/api") is False
    print("test_bash_safety_classifier: OK")


def test_fast_path_pretool_noops_safe_mcp():
    d = _fast_path_pretool("fact_get", {"entity": "lloyd"})
    assert d is not None and d.action == "noop", d
    d2 = _fast_path_pretool("memory_search", {"query": "x"})
    assert d2 is not None and d2.action == "noop", d2
    print("test_fast_path_pretool_noops_safe_mcp: OK")


def test_fast_path_pretool_escalates_unknown_mcp():
    # No safe verb in name → escalate
    d = _fast_path_pretool("autonomy_write_task", {"title": "x"})
    assert d is None, d
    d2 = _fast_path_pretool("memory_add", {"text": "x"})
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
    # Tool-call extraction (v4)
    test_extract_tool_call_well_formed,
    test_extract_tool_call_dict_arguments,
    test_extract_tool_call_no_tool_calls,
    test_extract_tool_call_bad_args_json,
    test_extract_tool_call_empty_choices,
    # Lever dispatch
    test_inject_appends_user_message,
    test_cancel_sets_event,
    test_cancel_works_when_budget_exhausted,
    test_inject_invokes_persist_intervention_callback,
    test_cancel_invokes_persist_intervention_callback,
    test_inject_persist_failure_does_not_break_lever,
    test_cancel_persist_failure_does_not_break_lever,
    test_ambient_fires_callback,
    test_ambient_with_no_callback_degrades_to_noop,
    test_inject_empty_content_degrades_to_noop,
    test_inject_on_result_translates_to_ambient_when_callback_present,
    test_inject_on_result_degrades_when_no_callback,
    test_cancel_on_result_degrades_to_noop,
    test_budget_exhausted,
    test_stub_announce_regex_matches_real_stalls,
    test_fast_path_flags_terminal_stub_announce_as_inject,
    test_fast_path_ignores_completed_answer,
    test_fast_path_tool_dispatch_only_still_noops,
    test_stall_rescue_inject_bypasses_budget,
    # _call_observer (tools-mode mocked)
    test_call_observer_parses_tool_call,
    test_call_observer_timeout_falls_back_to_noop,
    test_call_observer_no_tool_call_falls_back_to_noop,
    test_call_observer_unknown_lever_coerces_noop,
    # install_observer end-to-end
    test_install_observer_pretool_does_not_block,
    test_install_observer_assistant_message_inject,
    # Goal extraction (tools-mode)
    test_extract_goal_card_calls_llm,
    test_extract_goal_card_skips_empty_request,
    test_extract_goal_card_returns_none_on_error,
    test_extract_goal_card_returns_none_when_no_tool_call,
    test_event_prompt_includes_goal_card,
    test_event_prompt_handles_missing_goal_card,
    test_tool_result_summary_detects_spilled_payload,
    test_tool_result_summary_falls_back_for_normal_result,
    test_tool_result_summary_keeps_error_path_for_errors,
    test_intra_turn_microcompact_clears_in_place,
    test_intra_turn_microcompact_skips_under_threshold,
    # Tiered triggering (fast-path noop)
    test_fast_path_pretool_noops_read_glob_grep,
    test_fast_path_pretool_noops_safe_bash,
    test_fast_path_pretool_escalates_destructive_bash,
    test_bash_safety_classifier,
    test_fast_path_pretool_noops_safe_mcp,
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
