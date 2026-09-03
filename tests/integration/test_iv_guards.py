"""Inner Voice guards + v5 behavior changes — unit tests.

Covers the deterministic judgment lifted out of `observer.py`'s closure
into `app/inner_voice/guards.py`, plus the v5 changes that had no test
before: the /goal attach gate, tool-result sampling, off-critical-path
dispatch, and prompt hot-reload.

Every case here traces to something the first production window either
got wrong or could not see. Run:
  .venvs/lloyd/bin/python -m pytest tests/integration/test_iv_guards.py -q
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

LLOYD_HOME = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LLOYD_HOME))

from app.harness.hooks import HookRegistry
from app.inner_voice import guards
from app.inner_voice import observer as obs_mod
from app.inner_voice import observer_prompt as prompt_mod
from app.inner_voice.observer import (
    ObserverDecision,
    ObserverState,
    _fast_path_assistant_message,
    _fast_path_pretool,
    _fast_path_tool_result,
    _tool_name_tokens,
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


def run_async(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _state(**kw) -> ObserverState:
    base = dict(
        session_id="guard_sess",
        turn_id="guard_turn",
        user_request="do the thing",
        chat_messages_handle=[],
        cancel_event=asyncio.Event(),
        primary_model="primary",
        intervention_budget=3,
    )
    base.update(kw)
    return ObserverState(**base)


# ---------------------------------------------------------------------------
# Stall detection — the false positives that would have looped the primary
# ---------------------------------------------------------------------------


def test_stall_detects_real_stalls():
    stalls = [
        "Now let me check the logs:",
        "Let me read the config file.",
        "I'll examine the server status.",
        "Here is what I found so far:",
        "First, I'll start with the database.",
        "I'm going to run the test suite.",
    ]
    for text in stalls:
        assert guards.is_terminal_stall(text), text
    print("test_stall_detects_real_stalls: OK")


def test_stall_ignores_signoffs_and_speech_acts():
    """Sign-offs and in-sentence speech acts are delivered answers.

    Each string below matched the raw announce regex, so v4 fired a
    stall-rescue inject on it. That inject bypasses the intervention
    budget AND bypasses the consecutive-inject suppressor by design, so a
    primary that habitually closes with "Let me know if you need anything
    else" would be re-prompted every iteration until max_turns.
    """
    # These matched the raw announce regex, i.e. v4 injected on all of them.
    v4_false_positives = [
        "Here is the summary of findings.\n\nLet me know if you need anything else!",
        "All tests pass.\n\nI'll be happy to help with anything else.",
        "I need to note that the config is read-only at boot; edit config.yaml.",
        "Summary:\n- ok\n- fine\n\nI should mention one caveat: the cache is cold.",
        "Deployed.\n\nI'll be available if it regresses.",
    ]
    for text in v4_false_positives:
        assert guards._STUB_ANNOUNCE_RE.search(text.strip()), (
            f"precondition: raw regex should match {text!r}"
        )
        assert not guards.is_terminal_stall(text), text

    # Delivered answers the raw regex never flagged (the announce phrase is
    # mid-sentence, not at the start of the final line). Asserted so a
    # future loosening of the announce pattern can't quietly catch them.
    for text in [
        "Done. Please let me know if you'd like the docs updated too.",
        "The answer is 4.",
        "Fixed in three places. The tests cover each one.",
    ]:
        assert not guards.is_terminal_stall(text), text
    print("test_stall_ignores_signoffs_and_speech_acts: OK")


def test_fast_path_does_not_inject_on_signoff():
    """End to end through the fast path, not just the predicate."""
    d = _fast_path_assistant_message(
        "Here's the report.\n\nLet me know if you need anything else!", [],
    )
    assert d is None or d.action != "inject", d
    d2 = _fast_path_assistant_message("Now let me check the logs:", [])
    assert d2 is not None and d2.action == "inject", d2
    assert d2.bypass_budget is True
    print("test_fast_path_does_not_inject_on_signoff: OK")


# ---------------------------------------------------------------------------
# Consecutive-inject suppression across triggers
# ---------------------------------------------------------------------------


def test_suppressor_spans_all_midwork_triggers():
    """A pretool inject suppresses a following tool_result inject.

    Reproduces turn 2cf39d2c0ead: inject at pretool, inject at
    tool_result, inject at pretool, then cancel — four interventions in
    20 seconds with no model turn between them. v4 only compared
    same-trigger pairs, so each guard saw a clean slate.
    """
    prior = [{"trigger": "pretool", "action": "inject", "reason": "off scope"}]
    assert guards.suppress_consecutive_inject(
        action="inject", prior_decisions=prior, is_terminal=False,
    )
    print("test_suppressor_spans_all_midwork_triggers: OK")


def test_suppressor_never_fires_on_terminal_iteration():
    """On a terminal iteration the inject is the only thing keeping the
    loop alive — suppressing there guarantees work is left undone."""
    prior = [{"trigger": "assistant_message", "action": "inject", "reason": "stall"}]
    assert not guards.suppress_consecutive_inject(
        action="inject", prior_decisions=prior, is_terminal=True,
    )
    print("test_suppressor_never_fires_on_terminal_iteration: OK")


def test_suppressor_clears_after_an_intervening_noop():
    prior = [
        {"trigger": "pretool", "action": "inject", "reason": "off scope"},
        {"trigger": "tool_result", "action": "noop", "reason": "fine"},
    ]
    assert not guards.suppress_consecutive_inject(
        action="inject", prior_decisions=prior, is_terminal=False,
    )
    print("test_suppressor_clears_after_an_intervening_noop: OK")


def test_injects_primary_has_seen_collapses_a_dispatch_batch():
    """Three injects inside one dispatch batch is one nudge, not three.

    This is the count that should drive escalation to cancel. Only an
    `assistant_message` after an inject proves the primary got a turn to
    read it.
    """
    batch = [
        {"trigger": "pretool", "action": "inject"},
        {"trigger": "tool_result", "action": "inject"},
        {"trigger": "pretool", "action": "inject"},
    ]
    assert guards.injects_primary_has_seen(batch) == 0
    read_one = batch + [{"trigger": "assistant_message", "action": "noop"}]
    assert guards.injects_primary_has_seen(read_one) == 1
    print("test_injects_primary_has_seen_collapses_a_dispatch_batch: OK")


# ---------------------------------------------------------------------------
# Cancel-for-completion / budget / result downgrades
# ---------------------------------------------------------------------------


def test_cancel_for_completion_downgrades():
    assert guards.cancel_for_completion_verdict(
        action="cancel", reason="task complete", has_pending_tools=True,
        interventions_used=0,
    ) == "noop_cancel_with_pending_tools"
    assert guards.cancel_for_completion_verdict(
        action="cancel", reason="all success criteria met", has_pending_tools=False,
        interventions_used=0,
    ) == "acknowledge_complete"
    # Escalation from ignored injects is the documented path — allowed.
    assert guards.cancel_for_completion_verdict(
        action="cancel", reason="done", has_pending_tools=False,
        interventions_used=2,
    ) is None
    # A cancel for a real reason is never a completion cancel.
    assert guards.cancel_for_completion_verdict(
        action="cancel", reason="destructive loop", has_pending_tools=False,
        interventions_used=0,
    ) is None
    print("test_cancel_for_completion_downgrades: OK")


def test_budget_exempts_cancel_and_stall_rescue():
    assert guards.budget_exhausted(
        action="inject", bypass_budget=False, interventions_used=3, budget=3,
    )
    assert not guards.budget_exhausted(
        action="cancel", bypass_budget=False, interventions_used=9, budget=3,
    )
    assert not guards.budget_exhausted(
        action="inject", bypass_budget=True, interventions_used=9, budget=3,
    )
    print("test_budget_exempts_cancel_and_stall_rescue: OK")


def test_result_trigger_downgrades():
    assert guards.result_trigger_downgrade(
        action="inject", has_ambient_channel=True, has_content=True,
    )[0] == "ambient"
    assert guards.result_trigger_downgrade(
        action="inject", has_ambient_channel=False, has_content=True,
    )[0] == "noop_inject_on_result"
    assert guards.result_trigger_downgrade(
        action="cancel", has_ambient_channel=True, has_content=False,
    )[0] == "noop_cancel_on_result"
    assert guards.result_trigger_downgrade(
        action="noop", has_ambient_channel=True, has_content=False,
    )[0] == "noop"
    print("test_result_trigger_downgrades: OK")


def test_iteration_pressure():
    p = guards.iteration_pressure(48, 60)
    assert p.critical and abs(p.fraction - 0.8) < 1e-9
    assert not guards.iteration_pressure(10, 60).critical
    assert not guards.iteration_pressure(10, 0).critical  # unknown cap
    print("test_iteration_pressure: OK")


# ---------------------------------------------------------------------------
# The /goal attach gate — the bug that made the loop one-shot
# ---------------------------------------------------------------------------


def _gate(session_flags: dict, turn_source: str, producer: str) -> bool:
    from app.routers import _messages_inner_voice as iv
    tmp = Path(tempfile.mkdtemp(prefix="lloyd_gate_"))
    (tmp / "s1.json").write_text(json.dumps(session_flags))
    with patch.object(iv, "SESSIONS_DIR", tmp):
        return iv._iv_should_fire_on_turn("s1", turn_source, producer)


def test_goal_followup_is_observed_but_plain_iv_ambient_is_not():
    """The distinction the whole /goal loop rests on.

    A discretionary IV ambient must not be observed — the intervention
    budget resets each turn, so the observer would spawn and re-judge its
    own follow-ups without bound. A /goal retry MUST be observed, or
    `evaluate_goal_completion` never runs again, `attempts` never passes
    1, and `max_attempts` is unreachable. The runaway risk is bounded
    there by the attempt cap instead.
    """
    on = {"inner_voice": True, "inner_voice_evaluate_user_turns": True}
    assert _gate(on, "ambient", "inner_voice_goal") is True
    assert _gate(on, "ambient", "inner_voice") is False
    assert _gate(on, "ambient", "autonomy:morning-brief") is True
    assert _gate(on, "user", "") is True
    off = {"inner_voice": True, "inner_voice_evaluate_user_turns": False}
    assert _gate(off, "user", "") is False
    assert _gate(off, "ambient", "inner_voice_goal") is True
    assert _gate({"inner_voice": False}, "ambient", "inner_voice_goal") is False
    print("test_goal_followup_is_observed_but_plain_iv_ambient_is_not: OK")


# ---------------------------------------------------------------------------
# Fast-path classifiers
# ---------------------------------------------------------------------------


def test_tool_name_tokens_split_on_word_boundaries():
    assert _tool_name_tokens("delete_status_check") == {"delete", "status", "check"}
    assert _tool_name_tokens("vaultWriteNote") == {"vault", "write", "note"}
    assert _tool_name_tokens("fact.get") == {"fact", "get"}
    print("test_tool_name_tokens_split_on_word_boundaries: OK")


def test_mutation_verbs_beat_read_verbs_in_tool_names():
    """`delete_status_check` fast-noop'd under substring matching because
    it contains "status" and "check"."""
    assert _fast_path_pretool("delete_status_check", {}) is None
    assert _fast_path_pretool("session_delete", {}) is None
    assert _fast_path_pretool("vault_write_note", {"a": 1}) is None
    d = _fast_path_pretool("fact_get", {"entity": "lloyd"})
    assert d is not None and d.action == "noop"
    d2 = _fast_path_pretool("session_recall", {"q": "x"})
    assert d2 is not None and d2.action == "noop"
    print("test_mutation_verbs_beat_read_verbs_in_tool_names: OK")


def test_tool_result_sampling():
    """Benign results are sampled; errors and spills always escalate."""
    small = "ok"
    # Sampler picks 1 in 5 for LLM judgment (escalate -> None).
    judged = [
        _fast_path_tool_result("Bash", small, False, benign_seen=i, sample_every=5)
        is None
        for i in range(1, 21)
    ]
    assert sum(judged) == 4, judged
    # A real error always escalates.
    assert _fast_path_tool_result(
        "Bash", "boom", True, benign_seen=2, sample_every=5,
    ) is None
    # A parse-error is the primary's to retry.
    pe = _fast_path_tool_result(
        "Bash", "Tool call arguments could not be parsed as JSON: x",
        True, benign_seen=2, sample_every=5,
    )
    assert pe is not None and pe.action == "noop"
    # A very large result still escalates regardless of sampling.
    assert _fast_path_tool_result(
        "Bash", "x" * 30000, False, benign_seen=2, sample_every=5,
        escalate_bytes=20000,
    ) is None
    print("test_tool_result_sampling: OK")


# ---------------------------------------------------------------------------
# Off-critical-path dispatch
# ---------------------------------------------------------------------------


def test_nonterminal_judgment_does_not_block_the_hook():
    """A mid-work assistant_message must not hold up the harness.

    `fire_on_event` is awaited inline by the agent loop, so before v5
    every observer round-trip was time the primary spent idle — a mean of
    12.5s per turn across the first production window.
    """
    started = asyncio.Event()

    async def slow_post(**kwargs):
        started.set()
        await asyncio.sleep(5)
        raise AssertionError("should have been cancelled")

    cfg = obs_mod._observer_cfg()
    cfg.update({"async_nonterminal": True, "fast_path_enabled": False})

    async def scenario():
        hooks = HookRegistry()
        with patch.object(obs_mod, "_observer_cfg", return_value=cfg):
            state = install_observer(
                hooks=hooks, session_id="guard_sess", turn_id="async_turn",
                user_request="x", chat_messages_handle=[],
                cancel_event=asyncio.Event(), primary_model="primary",
            )
        evt = {
            "type": "assistant_message", "text": "working",
            "tool_calls": [{"function": {"name": "Bash"}}], "iteration": 2,
        }
        with patch.object(obs_mod, "_post_chat_completion_with_tools", new=slow_post):
            t0 = time.perf_counter()
            await hooks.fire_on_event(evt)
            elapsed = time.perf_counter() - t0
            await asyncio.wait_for(started.wait(), timeout=2)
            assert elapsed < 1.0, f"hook blocked for {elapsed:.2f}s"
            assert state.pending_tasks, "judgment should be in flight"
            # close_observer must cancel in-flight work rather than leave
            # it writing rows against a turn that has ended.
            obs_mod.close_observer(state)
            await asyncio.sleep(0)
        assert state.closed

    run_async(scenario())
    print("test_nonterminal_judgment_does_not_block_the_hook: OK")


def test_terminal_judgment_stays_synchronous():
    """loop.py decides whether to keep looping by checking whether this
    hook grew chat_messages, so a terminal inject must land before
    `fire_on_event` returns."""
    body = {
        "choices": [{"message": {"tool_calls": [{"function": {
            "name": "inject",
            "arguments": json.dumps({"reason": "stall", "content": "finish the task"}),
        }}]}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    async def fake_post(**kwargs):
        return body

    cfg = obs_mod._observer_cfg()
    cfg.update({"async_nonterminal": True, "fast_path_enabled": False})

    async def scenario():
        hooks = HookRegistry()
        chat: list = []
        with patch.object(obs_mod, "_observer_cfg", return_value=cfg):
            install_observer(
                hooks=hooks, session_id="guard_sess", turn_id="sync_turn",
                user_request="x", chat_messages_handle=chat,
                cancel_event=asyncio.Event(), primary_model="primary",
            )
        evt = {
            "type": "assistant_message", "text": "Let me check that",
            "tool_calls": [], "iteration": 4, "finish_reason": "stop",
        }
        with patch.object(obs_mod, "_post_chat_completion_with_tools", new=fake_post):
            await hooks.fire_on_event(evt)
        assert len(chat) == 1, chat
        assert "[INNER VOICE]" in chat[0]["content"]

    run_async(scenario())
    print("test_terminal_judgment_stays_synchronous: OK")


# ---------------------------------------------------------------------------
# Prompt plumbing
# ---------------------------------------------------------------------------


def test_prompt_hot_reload_picks_up_edits():
    """Editing a vault prompt used to need a backend restart."""
    tmp = Path(tempfile.mkdtemp(prefix="lloyd_prompt_")) / "system_prompt.md"
    tmp.write_text("---\ntitle: t\n---\nVERSION ONE\n")
    with patch.object(prompt_mod, "_SYSTEM_PROMPT_PATH", tmp):
        prompt_mod._PROMPT_CACHE.pop(str(tmp), None)
        assert prompt_mod.get_system_prompt() == "VERSION ONE"
        time.sleep(0.01)
        tmp.write_text("---\ntitle: t\n---\nVERSION TWO\n")
        assert prompt_mod.get_system_prompt() == "VERSION TWO"
    print("test_prompt_hot_reload_picks_up_edits: OK")


def test_goal_card_block_for_primary():
    card = {
        "success_criteria": ["tests pass"],
        "out_of_scope": ["refactoring"],
        "completion_signals": ["green CI"],
    }
    block = prompt_mod.build_goal_card_block_for_primary(card)
    assert "<goal_card>" in block and "tests pass" in block
    assert "refactoring" in block and "green CI" in block
    # Conversational turns produce an empty card and get no block.
    assert prompt_mod.build_goal_card_block_for_primary(
        {"success_criteria": [], "out_of_scope": [], "completion_signals": []},
    ) == ""
    assert prompt_mod.build_goal_card_block_for_primary(None) == ""
    print("test_goal_card_block_for_primary: OK")


def test_event_prompt_carries_cross_turn_memory_and_pressure():
    out = prompt_mod.build_user_prompt_for_event(
        user_request="fix it",
        goal_card=None,
        event_summary="EVENT",
        primary_text_so_far="text",
        interventions_used=0,
        interventions_budget=3,
        prior_turn_interventions=[
            {"trigger": "result", "action": "ambient", "reason": "unfinished"},
        ],
        iteration_pressure_note=prompt_mod.build_iteration_pressure_note(48, 60),
    )
    assert "EARLIER TURNS" in out
    assert "unfinished" in out
    assert "ITERATION PRESSURE" in out
    # No stale v3 lever names in the live prompt.
    assert "deny_tool" not in out
    assert "allow" not in out.split("Call exactly one lever tool")[-1]
    print("test_event_prompt_carries_cross_turn_memory_and_pressure: OK")


def test_observation_rows_record_the_observer_model():
    """The `model` column recorded the PRIMARY's alias, which made every
    row useless for "what served the observer?" — the exact question you
    must answer before pointing it at a smaller model."""
    captured = {}

    def fake_record(**kwargs):
        captured.update(kwargs)
        return 1

    state = _state(observer_model="secondary-7b", primary_model="primary")
    with patch.object(obs_mod, "record_inner_voice_observation", fake_record):
        run_async(obs_mod._persist(
            state, ObserverDecision(action="noop", reason="fine"), "result",
        ))
    assert captured["model"] == "secondary-7b", captured
    print("test_observation_rows_record_the_observer_model: OK")


def test_attach_appends_goal_card_to_the_harness_user_message():
    """The block must land on the message the HARNESS sends.

    `chat_messages_handle` is the harness's in-memory list; the session
    JSON keeps its own copy of the user message and never sees this
    block. Appending to the last user message (rather than the system
    prompt) is deliberate — the system prompt has to stay byte-stable for
    vLLM's prefix cache.
    """
    from app.routers import _messages_inner_voice as iv

    card = {
        "success_criteria": ["run the command"],
        "out_of_scope": [],
        "completion_signals": ["output shown"],
    }
    chat = [{"role": "user", "content": "run echo hello"}]

    class _Opts:
        hooks = None
        model = "primary"
        max_turns = 60

    async def fake_extract(*a, **kw):
        return card

    with patch.object(iv, "_iv_should_fire_on_turn", return_value=True), \
         patch.object(iv, "extract_goal_card", fake_extract), \
         patch.object(iv, "_load_prior_turn_interventions", return_value=[]), \
         patch.object(iv, "install_observer",
                      return_value=_state(intervention_budget=3)) as inst, \
         patch.object(iv._event_log, "log_event", lambda *a, **kw: None):
        run_async(iv.attach_observer_for_turn(
            session_id="guard_sess", turn_id="t1", turn_source="user",
            user_request="run echo hello", options=_Opts(),
            chat_messages_handle=chat, cancel_event=asyncio.Event(),
        ))

    assert "<goal_card>" in chat[-1]["content"], chat[-1]["content"]
    assert "run the command" in chat[-1]["content"]
    assert chat[-1]["content"].startswith("run echo hello"), "original text preserved"
    # max_turns must reach the observer or iteration pressure can never fire.
    assert inst.call_args.kwargs["max_turns"] == 60
    print("test_attach_appends_goal_card_to_the_harness_user_message: OK")


TESTS = [
    test_attach_appends_goal_card_to_the_harness_user_message,
    test_stall_detects_real_stalls,
    test_stall_ignores_signoffs_and_speech_acts,
    test_fast_path_does_not_inject_on_signoff,
    test_suppressor_spans_all_midwork_triggers,
    test_suppressor_never_fires_on_terminal_iteration,
    test_suppressor_clears_after_an_intervening_noop,
    test_injects_primary_has_seen_collapses_a_dispatch_batch,
    test_cancel_for_completion_downgrades,
    test_budget_exempts_cancel_and_stall_rescue,
    test_result_trigger_downgrades,
    test_iteration_pressure,
    test_goal_followup_is_observed_but_plain_iv_ambient_is_not,
    test_tool_name_tokens_split_on_word_boundaries,
    test_mutation_verbs_beat_read_verbs_in_tool_names,
    test_tool_result_sampling,
    test_nonterminal_judgment_does_not_block_the_hook,
    test_terminal_judgment_stays_synchronous,
    test_prompt_hot_reload_picks_up_edits,
    test_goal_card_block_for_primary,
    test_event_prompt_carries_cross_turn_memory_and_pressure,
    test_observation_rows_record_the_observer_model,
]


def main() -> int:
    failed = 0
    for t in TESTS:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {t.__name__}: {e!r}")
            failed += 1
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
