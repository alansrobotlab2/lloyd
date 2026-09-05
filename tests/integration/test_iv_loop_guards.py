"""Guards added after turn 20260905_011748_iv84e4 — the looping code review.

That turn: a primary that ran six reformulations of one search, then chased
two symbols that do not exist, then hit a 120s Bash timeout; an observer that
watched 24 tool calls over six minutes without a single LLM judgment, then
spent its whole intervention budget in 88 seconds, one of those injects
asserting a finding the primary had never reported; and a cancel.

Every test here fails against the code as it stood that night. Run:
  .venvs/lloyd/bin/python -m pytest tests/integration/test_iv_loop_guards.py -q
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

LLOYD_HOME = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LLOYD_HOME))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.inner_voice import guards
from app.inner_voice import observer as obs_mod
from app.inner_voice import observer_prompt as prompt_mod
from app.inner_voice.observer import (
    ObserverDecision,
    _apply_decision_guards,
    _fast_path_assistant_message,
    _fast_path_tool_result,
)
from fixture_iv_loop_turn import REAL_BASH_CALLS


# `_persist` records for real; keep the production usage.db clean.
RECORDED: list = []
obs_mod.record_inner_voice_observation = lambda **kw: (RECORDED.append(kw), len(RECORDED))[1]


def _sigs(commands):
    return [guards.tool_call_signature("Bash", {"command": c}) for c in commands]


# ---------------------------------------------------------------------------
# Repetition guard, replayed against the turn it was built for
# ---------------------------------------------------------------------------


def test_repetition_fires_on_the_real_loop():
    """Replay all 28 Bash calls in order; assert where the guard speaks.

    The guard is fed the same bounded ring the pretool hook maintains, and
    the ring is cleared on a fire exactly as the hook clears it — so this is
    the live firing sequence, not a per-call scoring pass.
    """
    ring: list = []
    fired_at: list[int] = []
    for idx, command in REAL_BASH_CALLS:
        ring.append(guards.tool_call_signature("Bash", {"command": command}))
        if len(ring) > 16:
            del ring[:-16]
        verdict = guards.repetition_verdict(ring)
        if verdict is not None:
            fired_at.append(idx)
            ring = []

    # 62 — the 4th grep of the search-reformulation cluster (messages 53-66).
    # 76 — the second pathological cluster, where the primary hunts
    # `build_subliminal_context` / `_messages_subliminal`, a symbol that is
    # never defined anywhere (see the fixture docstring, messages 72-79). The
    # ambient-path tokenisation used through 2026-09-04 could not see this
    # one: `alansrobotlab` padded every identifier set, so the containment
    # ratio was diluted and the real shared symbol never carried a match.
    # Both fires are inside a region the fixture labels pathological, and the
    # 15 healthy exploration calls before them stay silent — see
    # test_repetition_silent_through_healthy_exploration.
    assert fired_at == [62, 76], (
        f"expected fires at the two loop clusters (62, 76); got {fired_at}"
    )
    print("test_repetition_fires_on_the_real_loop: OK")


def test_repetition_silent_through_healthy_exploration():
    """Messages 4-51 are 15 calls of legitimate widening. Never fire there."""
    healthy = [c for i, c in REAL_BASH_CALLS if i <= 51]
    assert len(healthy) == 15
    ring: list = []
    for sig in _sigs(healthy):
        ring.append(sig)
        assert guards.repetition_verdict(ring) is None, sig.preview
    print("test_repetition_silent_through_healthy_exploration: OK")


def test_repetition_names_the_terms_the_primary_kept_chasing():
    loop = [c for i, c in REAL_BASH_CALLS if 53 <= i <= 64]
    verdict = guards.repetition_verdict(_sigs(loop))
    assert verdict is not None
    # The identifiers the primary rewrote its filter around six times.
    assert "iv_cancel_requested" in verdict.shared_terms
    assert "iv_inject_queue" in verdict.shared_terms
    content = guards.repetition_inject_content(verdict)
    assert "iv_cancel_requested" in content
    # The instruction that breaks the loop: an unchanged result is the answer.
    assert "ANSWER" in content
    print("test_repetition_names_the_terms_the_primary_kept_chasing: OK")


def test_repetition_needs_more_than_one_refinement():
    """Narrowing a search once is normal work, not a loop."""
    pair = _sigs([
        'grep -rn "iv_inject_queue" app/',
        'grep -rn "iv_inject_queue" app/ --include="*.py"',
    ])
    assert guards.repetition_verdict(pair) is None
    print("test_repetition_needs_more_than_one_refinement: OK")


def test_repetition_catches_verbatim_repeats_of_any_tool():
    """Exact re-runs need no similarity heuristic — three identical Reads."""
    sigs = [
        guards.tool_call_signature("Read", {"file_path": "/a/b.py"})
        for _ in range(3)
    ]
    verdict = guards.repetition_verdict(sigs)
    assert verdict is not None and verdict.exact
    assert "the same call" in guards.repetition_inject_content(verdict)
    print("test_repetition_catches_verbatim_repeats_of_any_tool: OK")


def test_repetition_ignores_different_tools():
    sigs = [
        guards.tool_call_signature("Read", {"file_path": "/a/b.py"}),
        guards.tool_call_signature("Grep", {"pattern": "/a/b.py"}),
        guards.tool_call_signature("Bash", {"command": "cat /a/b.py"}),
    ]
    assert guards.repetition_verdict(sigs) is None
    print("test_repetition_ignores_different_tools: OK")


# ---------------------------------------------------------------------------
# Fast-path rows must not count as judgment
# ---------------------------------------------------------------------------


def test_suppressor_sees_past_fast_path_rows():
    """The exact row sequence that defeated the suppressor on 2026-09-04.

    A tool_result inject, then the fast-path noops every iteration emits —
    one `assistant_message`, one `pretool` — then a second inject. Before
    this fix the walk-back stopped on the pretool bookkeeping row and
    reported "no prior inject".
    """
    prior = [
        {"trigger": "tool_result", "action": "inject", "reason": "drifting"},
        {"trigger": "assistant_message", "action": "noop",
         "reason": "fast-path: tool-dispatch-only iteration", "fast_path": True},
        {"trigger": "pretool", "action": "noop",
         "reason": "observation-only: pretool LLM disabled", "fast_path": True},
    ]
    assert guards.suppress_consecutive_inject(
        action="inject", prior_decisions=prior, is_terminal=False,
    )
    print("test_suppressor_sees_past_fast_path_rows: OK")


def test_suppressor_still_clears_on_a_judged_noop():
    """An LLM noop is a real look. It must still clear the suppressor."""
    prior = [
        {"trigger": "tool_result", "action": "inject", "reason": "drifting"},
        {"trigger": "tool_result", "action": "noop",
         "reason": "primary recovered", "fast_path": False},
    ]
    assert not guards.suppress_consecutive_inject(
        action="inject", prior_decisions=prior, is_terminal=False,
    )
    print("test_suppressor_still_clears_on_a_judged_noop: OK")


def test_every_deterministic_noop_is_tagged_fast_path():
    """A missed tag silently re-breaks the suppressor and the prompt window."""
    assert _fast_path_tool_result("Read", "hi", False, benign_seen=3, sample_every=5).fast_path
    assert obs_mod._fast_path_pretool("Read", {"file_path": "/x"}).fast_path
    assert obs_mod._fast_path_pretool("Bash", {"command": "ls -la"}).fast_path
    assert _fast_path_assistant_message("", [{"name": "Bash"}]).fast_path
    # The stall-rescue inject is judgment, not bookkeeping — it must NOT be
    # filtered out of the observer's history.
    stall = _fast_path_assistant_message("Now let me check the logs:", [])
    assert stall.action == "inject" and not stall.fast_path
    print("test_every_deterministic_noop_is_tagged_fast_path: OK")


def test_prior_decisions_block_drops_fast_path_noise():
    """At ~8 bookkeeping rows per iteration, the 8-slot window showed nothing.

    Reproduces the window as it stood at the third inject of the real turn:
    every slot a fast-path noop, and one real inject pushed out.
    """
    decisions = [{"trigger": "tool_result", "action": "inject",
                  "reason": "stuck in a loop", "fast_path": False}]
    for _ in range(4):
        decisions += [
            {"trigger": "assistant_message", "action": "noop",
             "reason": "fast-path: tool-dispatch-only iteration", "fast_path": True},
            {"trigger": "pretool", "action": "noop",
             "reason": "observation-only: pretool LLM disabled", "fast_path": True},
            {"trigger": "tool_result", "action": "noop",
             "reason": "fast-path: benign result (unsampled)", "fast_path": True},
        ]
    block = prompt_mod._format_prior_decisions(decisions)
    assert "stuck in a loop" in block, "the only real decision was crowded out"
    assert "fast-path" not in block
    assert "observation-only" not in block
    print("test_prior_decisions_block_drops_fast_path_noise: OK")


# ---------------------------------------------------------------------------
# Inject pacing
# ---------------------------------------------------------------------------


def test_cooldown_blocks_the_88_second_budget_burn():
    """Inject, three iterations, inject again — the real seq 66 -> 75 gap.

    The suppressor allows this (a judged noop sits between them). At a
    cooldown of 4 it is held, so the budget survives to the point where the
    drift is actually established.
    """
    prior = [
        {"trigger": "tool_result", "action": "inject", "reason": "loop", "fast_path": False},
        {"trigger": "assistant_message", "action": "noop", "fast_path": True},
        {"trigger": "tool_result", "action": "noop", "reason": "recovered", "fast_path": False},
        {"trigger": "assistant_message", "action": "noop", "fast_path": True},
        {"trigger": "assistant_message", "action": "noop", "fast_path": True},
    ]
    assert guards.iterations_since_last_inject(prior) == 3
    assert guards.inject_on_cooldown(prior, cooldown_iterations=4)
    assert not guards.inject_on_cooldown(prior, cooldown_iterations=3)
    print("test_cooldown_blocks_the_88_second_budget_burn: OK")


def test_cooldown_silent_before_any_inject():
    prior = [{"trigger": "assistant_message", "action": "noop", "fast_path": True}]
    assert guards.iterations_since_last_inject(prior) is None
    assert not guards.inject_on_cooldown(prior, cooldown_iterations=4)
    print("test_cooldown_silent_before_any_inject: OK")


def _state(**kw):
    from app.inner_voice.observer import ObserverState
    base = dict(
        session_id="loop_sess",
        turn_id="loop_turn",
        user_request="review the inner voice implementation",
        chat_messages_handle=[],
        cancel_event=asyncio.Event(),
        primary_model="primary",
        intervention_budget=3,
        cfg={"inject_cooldown_iterations": 4},
    )
    base.update(kw)
    return ObserverState(**base)


def _one_inject_ago(iterations: int) -> list[dict]:
    """Decision rows for "one inject, then `iterations` primary iterations".

    Shaped like the real turn: a judged noop lands shortly after the inject
    (seq 69 there — "Primary has received the config data it needed"), which
    is what clears `suppress_consecutive_inject` and lets the question reach
    the cooldown. Without that judged row the suppressor answers first and
    the cooldown is never consulted — which is correct, and is why this
    helper models the harder case.
    """
    rows: list[dict] = [
        {"trigger": "tool_result", "action": "inject", "reason": "loop",
         "fast_path": False},
    ]
    for i in range(iterations):
        rows.append(
            {"trigger": "assistant_message", "action": "noop", "fast_path": True}
        )
        if i == 0:
            rows.append(
                {"trigger": "tool_result", "action": "noop",
                 "reason": "primary acted on the nudge", "fast_path": False}
            )
    return rows


def test_cooldown_downgrades_a_discretionary_inject_end_to_end():
    state = _state()
    state.decisions_this_turn = _one_inject_ago(2)
    d = ObserverDecision(action="inject", reason="still drifting", content="focus")
    _apply_decision_guards(state, d, trigger="tool_result", tool_calls=[])
    assert d.action == "noop_inject_on_cooldown", d.action
    assert "2 primary iterations" in d.reason
    print("test_cooldown_downgrades_a_discretionary_inject_end_to_end: OK")


def test_cooldown_exempts_bypass_budget_and_terminal_injects():
    """Stall rescue and the repetition guard must never be rate-limited:
    they are what keeps a turn from ending or looping, and both are specific
    enough that firing twice is a real signal rather than a nag."""
    state = _state()
    state.decisions_this_turn = _one_inject_ago(1)
    rescue = ObserverDecision(
        action="inject", reason="stall rescue", content="continue", bypass_budget=True,
    )
    _apply_decision_guards(state, rescue, trigger="assistant_message", tool_calls=[])
    assert rescue.action == "inject", rescue.reason

    # A terminal iteration is the last chance to act; never hold it back.
    state2 = _state()
    state2.decisions_this_turn = _one_inject_ago(1)
    terminal = ObserverDecision(action="inject", reason="unmet criteria", content="finish")
    _apply_decision_guards(
        state2, terminal, trigger="assistant_message", tool_calls=[], is_terminal=True,
    )
    assert terminal.action == "inject", terminal.reason
    print("test_cooldown_exempts_bypass_budget_and_terminal_injects: OK")


def test_cooldown_releases_once_the_gap_is_real():
    state = _state()
    state.decisions_this_turn = _one_inject_ago(4)
    d = ObserverDecision(action="inject", reason="drift confirmed", content="refocus")
    _apply_decision_guards(state, d, trigger="tool_result", tool_calls=[])
    assert d.action == "inject", d.reason
    print("test_cooldown_releases_once_the_gap_is_real: OK")


# ---------------------------------------------------------------------------
# Failure payloads inside successful results
# ---------------------------------------------------------------------------


def test_subagent_max_turns_is_never_sampled_away():
    """The real payload: 4 minutes, 28 tool calls, nothing returned.

    300 bytes and is_error False, so both the size rule and the 1-in-5
    sampler skipped it and the observer never learned the Task failed.
    """
    payload = (
        '{"response": "\\n[stopped: max_turns]", "tools_used": ["Read", "Bash", '
        '"Bash", "Grep"], "description": "Review IV observer prompt"}'
    )
    assert guards.looks_like_failure_payload(payload)
    # benign_seen deliberately off the sampling boundary — this must escalate
    # regardless of where the sampler happens to be.
    assert _fast_path_tool_result("Task", payload, False, benign_seen=3, sample_every=5) is None
    print("test_subagent_max_turns_is_never_sampled_away: OK")


def test_bash_timeout_payload_escalates():
    payload = '{"error": "command timed out after 120000ms", "command": "grep -rn ..."}'
    assert guards.looks_like_failure_payload(payload)
    assert _fast_path_tool_result("Bash", payload, False, benign_seen=3, sample_every=5) is None
    print("test_bash_timeout_payload_escalates: OK")


def test_ordinary_results_still_fast_path():
    """The escalation must stay rare — it is unsampled and unconditional."""
    for content in [
        "total 48\ndrwxr-xr-x 1 alan alan 972 Sep 4 18:18 .",
        '{"response": "Here is the review you asked for.", "tools_used": ["Read"]}',
        "def build_tool_result_summary(tool_name, result_preview):",
    ]:
        assert not guards.looks_like_failure_payload(content), content
        d = _fast_path_tool_result("Bash", content, False, benign_seen=3, sample_every=5)
        assert d is not None and d.action == "noop"
    print("test_ordinary_results_still_fast_path: OK")


# ---------------------------------------------------------------------------
# The silent primary
# ---------------------------------------------------------------------------


def test_silent_streak_escalates_past_the_fast_path():
    """33 text-free iterations produced 33 zero-cost noops and no judgment."""
    calls = [{"name": "Bash"}]
    # Below the limit: still fast-pathed, still free.
    d = _fast_path_assistant_message("", calls, silent_streak=9, silent_streak_limit=10)
    assert d is not None and d.action == "noop" and d.fast_path
    # At the limit: escalate to LLM judgment.
    assert _fast_path_assistant_message("", calls, silent_streak=10, silent_streak_limit=10) is None
    # Disabled by config.
    d = _fast_path_assistant_message("", calls, silent_streak=99, silent_streak_limit=0)
    assert d is not None and d.action == "noop"
    print("test_silent_streak_escalates_past_the_fast_path: OK")


def test_silent_streak_block_does_not_presume_guilt():
    """A long quiet run is usually productive work. The prompt must say so,
    or a jumpy observer turns this into a nag every 10 iterations."""
    summary = prompt_mod.build_assistant_message_summary(
        22, "", [{"function": {"name": "Bash"}}], "tool_calls", silent_streak=12,
    )
    assert "SILENT STREAK" in summary and "iteration 12" in summary
    assert "that is common" in summary and "not by itself a problem" in summary
    # Absent unless escalated.
    plain = prompt_mod.build_assistant_message_summary(
        3, "", [{"function": {"name": "Bash"}}], "tool_calls",
    )
    assert "SILENT STREAK" not in plain
    print("test_silent_streak_block_does_not_presume_guilt: OK")


# ---------------------------------------------------------------------------
# The observer must be able to see what the primary ran
# ---------------------------------------------------------------------------


def test_tool_result_summary_carries_the_command():
    """Without this the observer sees a docstring with no context — which is
    how it concluded the primary had "discovered a subliminal injection bug"
    it had never mentioned."""
    docstring = '"""Subliminal-injection capture (#306).\n\nThree ephemeral injection sites...'
    summary = prompt_mod.build_tool_result_summary(
        "Bash", docstring, False,
        call_preview="cd ~/lloyd && head -20 app/routers/_messages_subliminal.py",
    )
    assert "head -20 app/routers/_messages_subliminal.py" in summary
    assert "Primary ran:" in summary
    print("test_tool_result_summary_carries_the_command: OK")


def test_tool_result_summary_unchanged_without_a_command():
    """Callers that cannot recover the args must still get a valid summary."""
    summary = prompt_mod.build_tool_result_summary("Read", "file body", False)
    assert "Primary ran:" not in summary
    assert "Tool Read returned result" in summary
    print("test_tool_result_summary_unchanged_without_a_command: OK")


def test_signature_preview_is_bounded():
    """Commands go into the observer's prompt; an unbounded one is a cost bug."""
    sig = guards.tool_call_signature("Bash", {"command": "echo " + "x" * 5000})
    assert len(sig.preview) <= 160
    print("test_signature_preview_is_bounded: OK")


def test_signature_ignores_key_order_and_description():
    """`description` is model narration, not intent — two calls that differ
    only there are the same call."""
    a = guards.tool_call_signature("Grep", {"pattern": "foo", "path": "app/", "description": "first try"})
    b = guards.tool_call_signature("Grep", {"path": "app/", "pattern": "foo", "description": "second try"})
    assert a.exact == b.exact
    print("test_signature_ignores_key_order_and_description: OK")


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()


# ---------------------------------------------------------------------------
# The silent model swap
# ---------------------------------------------------------------------------


def test_observer_resolves_to_the_primary_endpoint():
    """Inner Voice must run on the primary model, and say so out loud.

    `inner_voice.model` read `secondary` from 2026-05-07 onward and resolved
    to primary the whole time, because `resolve_model_alias` rewrites
    secondary -> primary while `secondary_enabled` is false. When de893d7
    turned that flag on for the autonomy scheduler (2026-09-03), the observer
    moved to Qwen3.5-4B with no config change and no log line. On its first
    day there it intervened on 40% of LLM-judged events against primary's
    1.7%, fabricated a finding, and cancelled a turn.

    This asserts the resolved endpoint, not the config string, so the alias
    indirection cannot hide a swap again.
    """
    from app.config import CONFIG
    base_url, model_name = obs_mod._resolve_endpoint()
    primary_url = (CONFIG["models"]["primary"].get("base_url") or "").rstrip("/")
    assert model_name == "primary", (
        f"observer resolved to {model_name!r}; pin inner_voice.model to primary "
        "or record the iv_grade evidence for moving it"
    )
    assert base_url == primary_url, (base_url, primary_url)
    print("test_observer_resolves_to_the_primary_endpoint: OK")


def test_alias_rewrite_is_logged_once(caplog=None):
    """The rewrite that hid the swap must leave a trace in the log."""
    import logging
    from app import config as config_mod

    config_mod._ALIAS_REWRITES_LOGGED.clear()
    original = config_mod.CONFIG.get("secondary_enabled")
    records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Capture()
    config_mod.logger.addHandler(handler)
    prior_level = config_mod.logger.level
    config_mod.logger.setLevel(logging.INFO)
    try:
        config_mod.CONFIG["secondary_enabled"] = False
        assert config_mod.resolve_model_alias("secondary") == "primary"
        assert config_mod.resolve_model_alias("secondary") == "primary"
        assert len(records) == 1, f"expected one log line, got {records}"
        assert "secondary_enabled is false" in records[0]
    finally:
        config_mod.logger.removeHandler(handler)
        config_mod.logger.setLevel(prior_level)
        config_mod.CONFIG["secondary_enabled"] = original
        config_mod._ALIAS_REWRITES_LOGGED.clear()
    print("test_alias_rewrite_is_logged_once: OK")


def test_new_guards_are_on_by_default():
    """A guard that ships disabled is a guard that does not exist. The
    stalled-progress gate sat behind `stalled_progress: false` through the
    entire incident."""
    cfg = obs_mod._observer_cfg()
    assert cfg["repetition_guard_enabled"] is True
    assert cfg["silent_iterations_before_review"] > 0
    assert cfg["inject_cooldown_iterations"] > 0
    assert obs_mod._todo_stewardship_cfg()["stalled_progress"] is True
    print("test_new_guards_are_on_by_default: OK")


# ---------------------------------------------------------------------------
# End to end: the real turn, replayed through the live hooks
# ---------------------------------------------------------------------------


async def _no_goal_card(*a, **kw):
    """Goal-card extraction is an LLM call at turn start; not under test."""
    return None


def test_real_turn_replayed_through_the_pretool_hook():
    """Drive the 28 Bash calls through `fire_pre_tool_use` and assert the
    observer speaks — with no LLM available at all.

    The point of doing this deterministically is that on the night in
    question the observer's three interventions all came from an LLM working
    from a 300-char result fragment, and one of them was a fabrication. This
    path cannot fabricate: it fires on the arguments or not at all.
    """
    from unittest.mock import patch
    from app.harness.hooks import HookRegistry
    from app.inner_voice.observer import install_observer

    async def no_llm(**kwargs):
        raise AssertionError("the repetition guard must not need an LLM call")

    cfg = obs_mod._observer_cfg()
    cfg.update({"pretool_llm_enabled": False, "fast_path_enabled": True})

    async def scenario():
        chat_messages: list = []
        hooks = HookRegistry()
        with patch.object(obs_mod, "_observer_cfg", return_value=cfg), \
             patch.object(obs_mod, "extract_goal_card", new=_no_goal_card):
            state = install_observer(
                hooks=hooks, session_id="loop_replay", turn_id="loop_replay_turn",
                user_request="architecture and code review of inner voice",
                chat_messages_handle=chat_messages,
                cancel_event=asyncio.Event(), primary_model="primary",
            )
        with patch.object(obs_mod, "_post_chat_completion_with_tools", new=no_llm):
            for _idx, command in REAL_BASH_CALLS:
                await hooks.fire_pre_tool_use(
                    session_id="loop_replay",
                    tool_name="Bash",
                    tool_input={"command": command},
                )
        obs_mod.close_observer(state)
        return chat_messages

    injected = asyncio.new_event_loop().run_until_complete(scenario())

    # One nudge per pathological cluster — the search-reformulation loop and
    # the hunt for a symbol that does not exist. See
    # test_repetition_fires_on_the_real_loop for why there are two.
    assert len(injected) == 2, f"expected two nudges, got {len(injected)}"
    body = str(injected[0].get("content"))
    assert "iv_cancel_requested" in body or "iv_inject_queue" in body
    assert "ANSWER" in body
    assert "_messages_subliminal" in str(injected[1].get("content"))
    print("test_real_turn_replayed_through_the_pretool_hook: OK")


def test_healthy_turn_replayed_produces_no_nudge():
    """The same path over the exploration phase alone must stay silent."""
    from unittest.mock import patch
    from app.harness.hooks import HookRegistry
    from app.inner_voice.observer import install_observer

    cfg = obs_mod._observer_cfg()
    cfg.update({"pretool_llm_enabled": False, "fast_path_enabled": True})

    async def scenario():
        chat_messages: list = []
        hooks = HookRegistry()
        with patch.object(obs_mod, "_observer_cfg", return_value=cfg), \
             patch.object(obs_mod, "extract_goal_card", new=_no_goal_card):
            state = install_observer(
                hooks=hooks, session_id="healthy_replay", turn_id="healthy_turn",
                user_request="architecture and code review of inner voice",
                chat_messages_handle=chat_messages,
                cancel_event=asyncio.Event(), primary_model="primary",
            )
        for idx, command in REAL_BASH_CALLS:
            if idx > 51:
                break
            await hooks.fire_pre_tool_use(
                session_id="healthy_replay", tool_name="Bash",
                tool_input={"command": command},
            )
        obs_mod.close_observer(state)
        return chat_messages

    injected = asyncio.new_event_loop().run_until_complete(scenario())
    assert injected == [], f"guard fired during healthy exploration: {injected}"
    print("test_healthy_turn_replayed_produces_no_nudge: OK")


def test_shared_terms_lead_with_the_rarest():
    """The message must name what is being chased, not the ambient path.

    Live turn 20260905_020747_ivfe5f named "alansrobotlab" — a component of
    the home directory in every command — ahead of the symbol the primary was
    actually looping on.
    """
    cmds = [
        "grep -rn zzq_phantom_handle_v3 /home/alansrobotlab/lloyd --include=*.py",
        "ls -la /home/alansrobotlab/lloyd; ls -d /home/alansrobotlab/lloyd/app",
        'grep -rn "zzq_phantom_handle_v3" /home/alansrobotlab/lloyd; echo EXIT=$?',
        "grep -rn --include='*' \"zzq_phantom_handle_v3\" /home/alansrobotlab/lloyd/app",
    ]
    verdict = guards.repetition_verdict(_sigs(cmds))
    assert verdict is not None
    assert verdict.shared_terms[0] == "zzq_phantom_handle_v3", verdict.shared_terms
    assert guards.repetition_inject_content(verdict).index("zzq_phantom_handle_v3") < 120
    print("test_shared_terms_lead_with_the_rarest: OK")
