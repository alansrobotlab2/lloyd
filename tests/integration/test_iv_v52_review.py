"""Fixes from the 2026-09-05 architecture and code review of Inner Voice.

The review that produced these was run with Inner Voice watching, and the
observer's own behaviour during it is the evidence behind most of them: 19
deterministic repetition injects in one evening, 15 of which named a path
segment or an argument key as "the term the primary keeps chasing"; and on
turn 8f3b7e77de07 ten of those also spent the discretionary budget, so the
one correct judgment of that turn — "hit max_turns with zero review
delivered; all six todos unresolved" — was downgraded to
`noop_budget_exhausted`.

Every test here fails against the code as it stood on 2026-09-04.
Run:
  .venvs/lloyd/bin/python -m pytest tests/integration/test_iv_v52_review.py -q
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

LLOYD_HOME = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LLOYD_HOME))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.inner_voice import guards
from app.inner_voice import observer as obs_mod
from app.inner_voice.observer import (
    ObserverDecision,
    ObserverState,
    _apply_decision_guards,
    _apply_lever,
)

# `_persist` writes for real; keep the production usage.db clean. This is a
# module-level assignment in the other IV test modules too, so whichever
# imported last owns the attribute — never assert against a module-level list.
# Use `_capture_rows()` when a test needs to see what was persisted.
obs_mod.record_inner_voice_observation = lambda **kw: 1


def _capture_rows():
    """Context manager yielding the observation rows written inside it."""
    rows: list = []
    ctx = patch.object(
        obs_mod, "record_inner_voice_observation",
        new=lambda **kw: (rows.append(kw), len(rows))[1],
    )
    return rows, ctx

HOME = str(Path.home())
REPO = f"{HOME}/lloyd"


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _state(**kw) -> ObserverState:
    base = dict(
        session_id="v52_sess",
        turn_id="v52_turn",
        user_request="review the inner voice implementation",
        chat_messages_handle=[],
        cancel_event=asyncio.Event(),
        primary_model="primary",
        intervention_budget=3,
        cfg=obs_mod._observer_cfg(),
    )
    base.update(kw)
    return ObserverState(**base)


def _fire(calls):
    """Replay tool calls through the guard the way the pretool hook does."""
    ring, fired = [], []
    for i, (tool, args) in enumerate(calls, 1):
        ring.append(guards.tool_call_signature(tool, args))
        ring = ring[-16:]
        verdict = guards.repetition_verdict(ring)
        if verdict is not None:
            fired.append((i, verdict))
            ring = []
    return fired


# ---------------------------------------------------------------------------
# Repetition guard — the 19-inject storm
# ---------------------------------------------------------------------------


def test_repetition_ignores_unrelated_reads_edits_and_writes():
    """Path-addressed tools share their directory, not their subject.

    Three Reads of unrelated files under one repo shared `file_path` (the
    argument key) and the username (a component of every absolute path), and
    two shared identifiers is all a near match needed. Live consequence:
    `deterministic: 3 near-identical Read calls for alansrobotlab,
    file_path, architecture`.
    """
    cases = {
        "reads": [
            ("Read", {"file_path": f"{REPO}/server.py"}),
            ("Read", {"file_path": f"{REPO}/config.yaml"}),
            ("Read", {"file_path": f"{REPO}/prompt_builder.py"}),
            ("Read", {"file_path": f"{REPO}/autonomy.py"}),
        ],
        "reads in one package": [
            ("Read", {"file_path": f"{REPO}/app/inner_voice/observer.py"}),
            ("Read", {"file_path": f"{REPO}/app/inner_voice/guards.py"}),
            ("Read", {"file_path": f"{REPO}/app/inner_voice/lever_tools.py"}),
        ],
        "edits": [
            ("Edit", {"file_path": f"{REPO}/app/a.py",
                      "old_string": "def foo():\n    pass",
                      "new_string": "def foo():\n    return 1"}),
            ("Edit", {"file_path": f"{REPO}/app/b.py",
                      "old_string": "x = 1", "new_string": "x = 2"}),
            ("Edit", {"file_path": f"{REPO}/web/src/api.ts",
                      "old_string": "const a", "new_string": "const b"}),
        ],
        "writes": [
            ("Write", {"file_path": f"{REPO}/tests/unit/test_a.py",
                       "content": "import pytest\n"}),
            ("Write", {"file_path": f"{REPO}/tests/unit/test_b.py",
                       "content": "import pytest\n"}),
            ("Write", {"file_path": f"{REPO}/tests/unit/test_c.py",
                       "content": "import pytest\n"}),
        ],
        "todo bookkeeping": [
            ("TodoWrite", {"todos": [{"content": "a", "status": "in_progress"}]}),
            ("TodoWrite", {"todos": [{"content": "a", "status": "completed"},
                                     {"content": "b", "status": "in_progress"}]}),
            ("TodoWrite", {"todos": [{"content": "a", "status": "completed"},
                                     {"content": "b", "status": "completed"}]}),
        ],
    }
    for label, calls in cases.items():
        fired = _fire(calls)
        assert fired == [], f"{label}: guard fired on healthy work — {fired}"


def test_repetition_ignores_a_chunked_read_of_one_file():
    """Reading a long file in windows is one job, not four repeats."""
    calls = [
        ("Read", {"file_path": f"{REPO}/app/inner_voice/observer.py",
                  "offset": off, "limit": 600})
        for off in (1, 600, 1200, 1800)
    ]
    assert _fire(calls) == []


def test_repetition_still_catches_a_verbatim_repeat():
    """Exact re-runs need no similarity heuristic and must still fire."""
    calls = [("Read", {"file_path": f"{REPO}/x.py"})] * 4
    fired = _fire(calls)
    assert fired, "a file read four times verbatim is a loop"
    assert fired[0][1].exact
    body = guards.repetition_inject_content(fired[0][1])
    assert "the same call" in body
    # No shared identifiers survive the ambient strip here, so the message
    # must fall back to naming the call itself rather than reading as
    # "run the same call for the same target".
    assert "ANSWER" in body


def test_repetition_catches_a_single_distinctive_symbol():
    """One specific symbol, chased three ways, is the failure this exists for.

    `min_overlap = 2` was calibrated when the username padded every
    identifier set. With ambient path components stripped, a genuine hunt for
    one symbol shares exactly one identifier — so distinctiveness, not count,
    has to carry the match.
    """
    cmds = [
        f"grep -rn zzq_phantom_handle_v3 {REPO} --include=*.py",
        f"ls -la {REPO}; ls -d {REPO}/app",
        f'grep -rn "zzq_phantom_handle_v3" {REPO}; echo EXIT=$?',
        f"grep -rn --include='*' \"zzq_phantom_handle_v3\" {REPO}/app",
    ]
    verdict = guards.repetition_verdict(
        [guards.tool_call_signature("Bash", {"command": c}) for c in cmds]
    )
    assert verdict is not None
    assert "zzq_phantom_handle_v3" in verdict.shared_terms
    # The ambient path component must not be named as the thing being chased.
    assert not any("alansrobotlab" in t for t in verdict.shared_terms)


def test_identifiers_come_from_values_not_argument_keys():
    sig = guards.tool_call_signature(
        "Edit", {"file_path": "/srv/app/thing.py",
                 "old_string": "alpha", "new_string": "beta"},
    )
    for key in ("file_path", "old_string", "new_string"):
        assert key not in sig.idents, f"{key} is an argument key, not a subject"
    # The full key=value rendering still backs the exact-repeat comparison.
    assert "file_path=" in sig.exact


def test_identifiers_drop_the_home_path():
    sig = guards.tool_call_signature(
        "Bash", {"command": f"grep -rn iv_inject_queue {REPO}/app"},
    )
    user = HOME.rsplit("/", 1)[-1]
    assert user not in sig.idents
    assert "iv_inject_queue" in sig.idents


def test_distinctiveness_separates_symbols_from_module_names():
    assert guards._is_distinctive("iv_inject_queue")
    assert guards._is_distinctive("zzq_phantom_handle_v3")
    assert guards._is_distinctive("build_subliminal_context")
    # Two-part names are modules and argument keys — half the calls in a
    # session mention them in passing.
    assert not guards._is_distinctive("inner_voice")
    assert not guards._is_distinctive("file_path")
    assert not guards._is_distinctive("observer_prompt")


# ---------------------------------------------------------------------------
# Stall regex — announces that are actually delivered statements
# ---------------------------------------------------------------------------


def test_stall_ignores_explained_decisions_and_refusals():
    """An announce verb followed by a reason is a decision, not a stall.

    The stall-rescue inject bypasses both the intervention budget and the
    consecutive-inject suppressor, so a primary that closes this way is
    re-prompted every iteration until max_turns.
    """
    delivered = [
        "The fix is in place and tests pass.\n\n"
        "I'll leave the config as-is since it already works.",
        "I'm going to recommend option B because it is simpler.",
        "I will not change that file because it is generated.",
        "Going to the source, the loop appends the assistant message "
        "after the hook.",
        "Two options:\n1. Patch loop.py\n2. Leave it\n\n"
        "I need to hear which you prefer before proceeding.",
        "Let's go with option B.",
        "I'll leave that decision to you.",
        "Summary of findings:\n- A\n- B\n\n"
        "I should also mention nothing else is affected.",
    ]
    for text in delivered:
        assert guards._STUB_ANNOUNCE_RE.search(text.strip()), (
            f"precondition: the raw announce regex should match {text!r}"
        )
        assert not guards.is_terminal_stall(text), text

    # Delivered answers the raw announce regex never flagged. Asserted so a
    # future loosening of the announce pattern cannot quietly catch them.
    for text in [
        "I won't touch the generated file.",
        "That file is generated, so I did not edit it.",
        "The observer runs on the primary model.",
    ]:
        assert not guards.is_terminal_stall(text), text


def test_stall_still_catches_the_real_thing():
    for text in [
        "Now let me check the logs:",
        "Let me read the config file.",
        "I'll examine the server status.",
        "First, I'll start with the database.",
        "I'm going to run the test suite.",
        "Let me verify that now...",
        "I'll update the docs to match.",
        "Here is what I found so far:",
    ]:
        assert guards.is_terminal_stall(text), text


# ---------------------------------------------------------------------------
# Budget accounting
# ---------------------------------------------------------------------------


def test_deterministic_inject_does_not_spend_the_discretionary_budget():
    """The doc always said bypass_budget injects are not counted; the code
    counted them anyway.

    On turn 8f3b7e77de07 ten false repetition injects exhausted a budget of
    three, and the observer's one correct decision of that turn was recorded
    as `noop_budget_exhausted`.
    """
    state = _state()
    for i in range(4):
        d = ObserverDecision(
            action="inject", reason="repetition", content=f"stop {i}",
            bypass_budget=True,
        )
        _run(_apply_lever(state, d, trigger="pretool", related_tool="Bash"))
        assert d.action == "inject", d.action
    assert state.interventions_used == 0, (
        "deterministic injects must leave the discretionary budget untouched"
    )
    assert state.bypass_interventions_used == 4

    # …and a real judgment still gets its full budget afterwards.
    d = ObserverDecision(action="inject", reason="drift", content="refocus")
    _run(_apply_lever(state, d, trigger="assistant_message"))
    assert d.action == "inject"
    assert state.interventions_used == 1


def test_deterministic_injects_are_capped():
    """Exempt from the budget is not the same as unbounded."""
    cfg = dict(obs_mod._observer_cfg())
    cfg["deterministic_inject_budget"] = 2
    state = _state(cfg=cfg)
    actions = []
    for i in range(4):
        d = ObserverDecision(
            action="inject", reason="repetition", content=f"stop {i}",
            bypass_budget=True,
        )
        _run(_apply_lever(state, d, trigger="pretool", related_tool="Bash"))
        actions.append(d.action)
    assert actions == [
        "inject", "inject",
        "noop_deterministic_budget_exhausted",
        "noop_deterministic_budget_exhausted",
    ], actions
    assert len(state.chat_messages_handle) == 2


# ---------------------------------------------------------------------------
# Guard ordering at the result event
# ---------------------------------------------------------------------------


def test_result_inject_survives_a_cooldown_that_would_block_it():
    """At `result` an inject IS a request for an ambient follow-up.

    Running the guards first let the cooldown or the consecutive-inject
    suppressor downgrade it to a noop, discarding the follow-up entirely —
    on the one trigger whose judgments the review found most valuable.
    """
    queued: list = []

    async def ambient_cb(content, reason, producer="inner_voice"):
        queued.append((content, reason, producer))

    state = _state(enqueue_ambient_callback=ambient_cb)
    # A recent inject, so `inject_on_cooldown` would fire.
    state.decisions_this_turn = [
        {"trigger": "tool_result", "action": "inject", "reason": "drift",
         "fast_path": False},
    ]
    decision = ObserverDecision(
        action="inject", reason="turn ended with the work undone",
        content="Finish the review you started.",
    )
    # Production order: translate the lever, THEN run the guards.
    translated, note = guards.result_trigger_downgrade(
        action=decision.action,
        has_ambient_channel=True,
        has_content=True,
    )
    if translated != decision.action:
        decision.action = translated
        if note:
            decision.reason += f" [{note}]"
    _apply_decision_guards(
        state, decision, trigger="result", tool_calls=[], has_pending_tools=False,
    )
    _run(_apply_lever(state, decision, trigger="result"))

    assert decision.action == "ambient", decision.action
    assert queued, "the follow-up must actually be queued"


# ---------------------------------------------------------------------------
# Cancel guards
# ---------------------------------------------------------------------------


def test_cancel_for_ignored_injects_needs_injects_the_primary_saw():
    """Injects fired inside one dispatch batch are one nudge, not three.

    `injects_primary_has_seen` was written for exactly this and had no
    production call site; the cancel it was meant to prevent fired twice, a
    year apart, with the same reason string.
    """
    state = _state()
    state.interventions_used = 3
    # Three injects, no assistant_message between any of them — the primary
    # never completed an iteration and cannot have ignored anything.
    state.decisions_this_turn = [
        {"trigger": "tool_result", "action": "inject", "reason": "loop",
         "fast_path": False},
        {"trigger": "pretool", "action": "inject", "reason": "loop",
         "fast_path": False},
        {"trigger": "tool_result", "action": "inject", "reason": "loop",
         "fast_path": False},
    ]
    decision = ObserverDecision(
        action="cancel",
        reason="primary exhausted inject budget and is stuck in a loop",
    )
    _apply_decision_guards(
        state, decision, trigger="tool_result", tool_calls=[],
        has_pending_tools=True,
    )
    assert decision.action == "noop_cancel_unread_injects", decision.action
    assert not state.cancel_event.is_set()


def test_cancel_for_ignored_injects_allowed_once_the_primary_has_read_one():
    state = _state()
    state.interventions_used = 2
    state.decisions_this_turn = [
        {"trigger": "tool_result", "action": "inject", "reason": "loop",
         "fast_path": False},
        # A completed primary iteration — the inject reached it.
        {"trigger": "assistant_message", "action": "noop", "reason": "still looping",
         "fast_path": False},
        {"trigger": "tool_result", "action": "inject", "reason": "loop",
         "fast_path": False},
        {"trigger": "assistant_message", "action": "noop", "reason": "still looping",
         "fast_path": False},
    ]
    decision = ObserverDecision(
        action="cancel", reason="primary has ignored two injects on the same theme",
    )
    _apply_decision_guards(
        state, decision, trigger="tool_result", tool_calls=[],
        has_pending_tools=True,
    )
    assert decision.action == "cancel", decision.action


def test_cancel_for_a_destructive_loop_needs_no_injects_behind_it():
    """The unread-inject gate must only touch cancels that CLAIM ignored injects."""
    state = _state()
    state.interventions_used = 1
    state.decisions_this_turn = [
        {"trigger": "pretool", "action": "inject", "reason": "stop", "fast_path": False},
    ]
    decision = ObserverDecision(
        action="cancel", reason="primary is in a destructive loop deleting files",
    )
    _apply_decision_guards(
        state, decision, trigger="pretool", tool_calls=[], has_pending_tools=True,
    )
    assert decision.action == "cancel", decision.action


def test_cancel_for_completion_at_pretool_is_not_an_acknowledgement():
    """A tool is dispatching by definition at `pretool`.

    Deriving `has_pending_tools` from an always-empty `tool_calls` list
    recorded these as `acknowledge_complete`, which the UI renders as "IV
    reviewed and agrees the answer is complete" — on a turn mid-tool-call.
    """
    state = _state()
    decision = ObserverDecision(
        action="cancel", reason="all success criteria met, task complete",
    )
    _apply_decision_guards(
        state, decision, trigger="pretool", tool_calls=[], has_pending_tools=True,
    )
    assert decision.action == "noop_cancel_with_pending_tools", decision.action


# ---------------------------------------------------------------------------
# /goal loop — the attempt counter is the only bound on it
# ---------------------------------------------------------------------------


def test_goal_followup_is_skipped_when_the_attempt_counter_will_not_persist():
    """`attempts` is re-read from the session JSON every turn.

    A `inner_voice_goal` follow-up is deliberately exempt from the
    self-observation refusal, so the turn it queues is itself observed and
    can queue another. If the counter never advances, `max_attempts` is never
    reached and the loop runs without bound — and the mutation failure was
    swallowed.
    """
    queued: list = []

    async def ambient_cb(content, reason, producer="inner_voice"):
        queued.append((content, reason, producer))

    async def failed_persist(*a, **kw):
        return None      # session gone, disk error, malformed JSON

    async def unmet(**kw):
        return obs_mod.GoalCompletionVerdict(
            achieved=False, reason="the file was never written",
        )

    state = _state(enqueue_ambient_callback=ambient_cb)
    state.persistent_goal = {"text": "write the report", "attempts": 1}

    rows, capture = _capture_rows()
    with patch.object(obs_mod, "_persist_goal_state", new=failed_persist), \
         patch.object(obs_mod, "evaluate_goal_completion", new=unmet), capture:
        _run(obs_mod._handle_persistent_goal_at_result(
            state, {"response_text": "I will get to it."},
            prior_decision=ObserverDecision(action="noop"),
        ))

    assert queued == [], "an unbounded retry must not be queued"
    assert [r["action"] for r in rows] == ["noop_goal_attempts_not_persisted"], rows


def test_goal_followup_is_queued_when_the_counter_advances():
    """Control: the loop still loops when bookkeeping works."""
    queued: list = []

    async def ambient_cb(content, reason, producer="inner_voice"):
        queued.append((content, reason, producer))

    async def ok_persist(session_id, *, achieved_at=None, bump_attempts=False):
        return {"text": "write the report", "attempts": 2}

    async def unmet(**kw):
        return obs_mod.GoalCompletionVerdict(
            achieved=False, reason="the file was never written",
        )

    state = _state(enqueue_ambient_callback=ambient_cb)
    state.persistent_goal = {"text": "write the report", "attempts": 1}

    with patch.object(obs_mod, "_persist_goal_state", new=ok_persist), \
         patch.object(obs_mod, "evaluate_goal_completion", new=unmet):
        _run(obs_mod._handle_persistent_goal_at_result(
            state, {"response_text": "I will get to it."},
            prior_decision=ObserverDecision(action="noop"),
        ))

    assert len(queued) == 1, queued
    # The tag is load-bearing: it is what makes the follow-up observed.
    assert queued[0][2] == "inner_voice_goal"


# ---------------------------------------------------------------------------
# Attach gate — one read of the session file, not three
# ---------------------------------------------------------------------------


def test_the_attach_gate_parses_the_session_file_once(tmp_path):
    """`_iv_should_fire_on_turn` ran on the turn's critical path and parsed
    the session JSON three times: once itself, once inside
    `_session_iv_evaluate_user_turns_enabled`, which re-invoked
    `_session_inner_voice_enabled` first. Session files reach megabytes.
    """
    import json as _json
    from app.routers import _messages_inner_voice as iv

    (tmp_path / "s1.json").write_text(_json.dumps({
        "inner_voice": True, "inner_voice_evaluate_user_turns": True,
    }))

    calls = {"n": 0}

    class _CountingJson:
        @staticmethod
        def loads(text):
            calls["n"] += 1
            return _json.loads(text)

    with patch.object(iv, "SESSIONS_DIR", tmp_path), \
         patch.object(iv, "json", _CountingJson):
        assert iv._iv_should_fire_on_turn("s1", "user") is True
    assert calls["n"] == 1, f"parsed the session file {calls['n']} times"


def test_the_attach_gate_still_honours_both_flags(tmp_path):
    import json as _json
    from app.routers import _messages_inner_voice as iv

    (tmp_path / "off.json").write_text(_json.dumps({"inner_voice": False}))
    (tmp_path / "ambient_only.json").write_text(_json.dumps({
        "inner_voice": True, "inner_voice_evaluate_user_turns": False,
    }))
    (tmp_path / "both.json").write_text(_json.dumps({
        "inner_voice": True, "inner_voice_evaluate_user_turns": True,
    }))

    with patch.object(iv, "SESSIONS_DIR", tmp_path):
        assert iv._iv_should_fire_on_turn("off", "user") is False
        assert iv._iv_should_fire_on_turn("off", "ambient") is False
        assert iv._iv_should_fire_on_turn("ambient_only", "user") is False
        assert iv._iv_should_fire_on_turn("ambient_only", "ambient") is True
        assert iv._iv_should_fire_on_turn("both", "user") is True
        assert iv._iv_should_fire_on_turn("missing", "user") is False
        # Discretionary IV ambients stay unobserved; the /goal retry does not.
        assert iv._iv_should_fire_on_turn(
            "both", "ambient", "inner_voice") is False
        assert iv._iv_should_fire_on_turn(
            "both", "ambient", "inner_voice_goal") is True
        # The user-turn flag is meaningless without the master switch.
        assert iv._session_iv_evaluate_user_turns_enabled("off") is False


# ---------------------------------------------------------------------------
# The grader has to be able to see a misfiring guard
# ---------------------------------------------------------------------------


def _grade_module():
    sys.path.insert(0, str(LLOYD_HOME / "scripts"))
    import iv_grade
    return iv_grade


def test_grader_does_not_score_guard_injects_as_precision():
    """19 false injects read as a 1.0 landed rate.

    A guard inject only ever fires mid-turn, so "the loop continued
    afterwards" is true by construction. Scoring them alongside model-judged
    injects meant the precision proxy reported perfect precision on the
    evening the repetition guard misfired 19 times.
    """
    iv_grade = _grade_module()
    rows = [
        {"turn_id": "t1", "sequence_in_turn": 1, "trigger": "pretool",
         "action": "inject", "reason": "deterministic: 3 near-identical Read calls"},
        {"turn_id": "t1", "sequence_in_turn": 2, "trigger": "assistant_message",
         "action": "noop", "reason": "on task"},
        {"turn_id": "t1", "sequence_in_turn": 3, "trigger": "pretool",
         "action": "inject", "reason": "deterministic: 3 near-identical Read calls"},
        {"turn_id": "t1", "sequence_in_turn": 4, "trigger": "assistant_message",
         "action": "noop", "reason": "on task"},
        {"turn_id": "t1", "sequence_in_turn": 5, "trigger": "pretool",
         "action": "inject", "reason": "deterministic: 3 near-identical Read calls"},
        {"turn_id": "t1", "sequence_in_turn": 6, "trigger": "assistant_message",
         "action": "noop", "reason": "on task"},
        # One real, model-judged inject that the turn then ignored.
        {"turn_id": "t1", "sequence_in_turn": 7, "trigger": "tool_result",
         "action": "inject", "reason": "primary is drifting from the request"},
    ]
    got = iv_grade._grade_injects(rows)
    assert got["injects"] == 1, "only model-judged injects are scored"
    assert got["deterministic_injects"] == 3
    assert got["stranded"] == 1
    assert got["landed_rate"] == 0.0, "the real inject was stranded, not perfect"
    assert got["deterministic_worst_turns"][0] == {"turn_id": "t1", "injects": 3}


def test_grader_shows_a_trigger_that_acts_without_spending():
    """`pretool` vanished from the cost table entirely on the storm evening.

    Guard interventions cost no LLM call, so they were absent from every
    per-trigger row — the one trigger doing all the intervening was the one
    trigger the report did not mention.
    """
    iv_grade = _grade_module()
    rows = [
        {"turn_id": "t1", "sequence_in_turn": i, "trigger": "pretool",
         "action": "inject", "reason": "deterministic: near-identical calls",
         "input_tokens": 0, "latency_ms": 0, "cache_read": 0,
         "model": "primary", "error": None}
        for i in range(1, 4)
    ]
    cost = iv_grade._cost(rows)
    assert "pretool" in cost["by_trigger"], cost["by_trigger"]
    assert cost["by_trigger"]["pretool"]["guard"] == 3
    assert cost["by_trigger"]["pretool"]["calls"] == 0


# ---------------------------------------------------------------------------
# Configuration reachable from config.yaml
# ---------------------------------------------------------------------------


def test_review_knobs_have_defaults():
    cfg = obs_mod._observer_cfg()
    assert cfg["deterministic_inject_budget"] >= 1
    # Off-critical-path judgments get a longer deadline than the two
    # synchronous terminal calls: all 7 observer errors on 2026-09-04 were
    # `timeout after 5.0s`, every one on a spawned non-terminal call.
    assert cfg["async_timeout_seconds"] > cfg["timeout_seconds"]
