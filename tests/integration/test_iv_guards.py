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
import datetime
import importlib.util
import json
import shutil
import sqlite3
import subprocess
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


def test_observation_rows_keep_the_local_naive_writer_clock():
    """#835 clause 5: the window fix stays in the QUERY and the prose.

    `scripts/iv_grade.py`'s `--since` now normalises both sides of the comparison with
    `replace(..., 'T', ' ')`, which makes the window a clock comparison. That fix is
    only sufficient while `created_at` keeps the shape the writer gives it, so this
    pins the shape and the clock: a row inserted by the real writer must land within
    seconds of LOCAL `datetime.now()`, carrying no UTC offset.

    The durable half of #835 — stamping the column in UTC — is deliberately NOT taken.
    `usage_store.py:339-342` records that SQLite's UTC-naive `CURRENT_TIMESTAMP` would
    "be mis-parse[d] as local time and shift observations into the future by the local
    TZ offset" in the frontend timeline merge, so moving the writer is a `web/src`
    change and belongs to another item. If this test fails, that merge is already
    putting observations hours ahead of the messages beside them — widen nothing here;
    fix the writer together with the frontend.

    Runs against a scratch `usage.db` (this module's own isolation rule: `_persist`
    and its writer must never append to the table the grader is judging).
    """
    import sqlite3
    from datetime import datetime, timedelta

    import usage_store

    with tempfile.TemporaryDirectory() as td:
        scratch = Path(td) / "usage.db"
        with patch.object(usage_store, "DB_PATH", scratch):
            usage_store.record_inner_voice_observation(
                session_id="writer_clock_sess", turn_id="t1", sequence_in_turn=1,
                trigger="result", action="noop", reason="fixture")
            rows = usage_store.list_inner_voice_observations(
                session_id="writer_clock_sess")
            # The seam clause 5 depends on: the row the WRITER produces has to be the
            # row the GRADER's clause can still select. Both halves are the shipped
            # code — one real insert, one real predicate — not two hand-written strings.
            # Loaded by path: `scripts/` holds modules that shadow stdlib names, and
            # putting it on sys.path to import one file can break unrelated tests.
            import importlib.util

            _gv = importlib.util.spec_from_file_location(
                "iv_grade_writer_clock", LLOYD_HOME / "scripts" / "iv_grade.py")
            iv_grade = importlib.util.module_from_spec(_gv)
            _gv.loader.exec_module(iv_grade)
            conn = sqlite3.connect(str(scratch))
            kept = conn.execute(
                f"SELECT COUNT(*) FROM inner_voice_observations "
                f"WHERE {iv_grade.WINDOW_CLAUSE}",
                ((datetime.now() - timedelta(hours=1))
                 .strftime("%Y-%m-%d %H:%M:%S"),),   # the SPACE form
            ).fetchone()[0]
            conn.close()

    assert len(rows) == 1, rows
    stored = rows[0]["created_at"]
    parsed = datetime.fromisoformat(stored)
    assert parsed.tzinfo is None, (
        f"created_at={stored!r} now carries an offset; the frontend timeline merge "
        "reads this column as local time (usage_store.py:339-342)")
    skew = abs((datetime.now() - parsed).total_seconds())
    assert skew < 120, (
        f"created_at={stored!r} is {skew:.0f}s from local datetime.now(): the writer "
        "has moved off the local clock, which shifts every observation relative to "
        "the messages the frontend timeline puts it next to")
    assert kept == 1, (
        f"a row the live writer just stamped was excluded by a space-form bound one "
        f"hour earlier (iv_grade.WINDOW_CLAUSE={iv_grade.WINDOW_CLAUSE!r}) — the "
        "writer's format and the grader's window have drifted apart")
    print("test_observation_rows_keep_the_local_naive_writer_clock: OK")


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


# ---------------------------------------------------------------------------
# The IV-metrics recorder's sustained-breach path, at the bound Alan ruled on
# #460 (0.05; applied to the code by #1288).
#
# These are the only nodes here that cross a *process* boundary on purpose, and
# the boundary is the point: the nightly is
#   cd ~/lloyd && python3 scripts/iv_grade.py --json --since X | python3 scripts/iv_metrics_record.py --out Y
# run through an agent's Bash tool, so the scheduler can act on nothing but the
# exit status and the prose the run reports. The margin the bound exists for is
# equally specific: it has to fire on a series whose median sits at the degraded
# 2026-09-05..09-11 week and stay silent on the history recorded since.
# `tests/test_iv_metrics_series.py` pins the recorder in general and the bound's
# magnitude; these two pin that margin end-to-end, through the exit code.
# ---------------------------------------------------------------------------

RECORDER_SCRIPTS = ("iv_grade.py", "iv_metrics_record.py")

#: Loaded by path, never by putting `scripts/` on sys.path: that directory holds
#: 40+ modules, several of which shadow stdlib names.
_recorder_spec = importlib.util.spec_from_file_location(
    "iv_metrics_record_for_guards", LLOYD_HOME / "scripts" / "iv_metrics_record.py")
IV_METRICS = importlib.util.module_from_spec(_recorder_spec)
_recorder_spec.loader.exec_module(IV_METRICS)  # type: ignore[attr-defined]

#: Per-local-day dropped rates across the degraded 2026-09-05..09-11 stretch,
#: measured off `usage.db` on 2026-09-20 (221 drops over 4,630 calls). The median
#: of these seven is 0.0526: under the retired 0.10 for every night of that week,
#: over the 0.05 that replaced it. These are the six prior nights; the seventh
#: (the night being recorded) is built into the fixture below at the same rate.
IV_DEGRADED_PRIOR_RATES = [0.0548, 0.118, 0.0341, 0.0856, 0.0137, 0.0086]

#: The `dropped_rate` of every row in `_pipeline/reflection/iv-metrics.jsonl`
#: written 2026-09-12..09-19 — the healthy history the bound must not mistake for
#: a bad week (max 0.0203).
IV_RECORDED_HEALTHY_RATES = [0.0203, 0.017, 0.0111, 0.0, 0.0062, 0.0061, 0.0, 0.0091]

#: What the newest live row (window 2026-09-18..09-20) carries: 1 drop / 67 calls.
IV_RECORDED_HEALTHY_LATEST = 1 / 67


def _iv_repo(root: str) -> Path:
    """A throwaway repo root `root/lloyd` holding copies of both scripts.

    Copied, not symlinked: `iv_grade.py` anchors its database at
    `Path(__file__).resolve().parents[1]`, and a symlink resolves back to the real
    checkout — the run would read the production `usage.db`.
    """
    repo = Path(root) / "lloyd"
    (repo / "scripts").mkdir(parents=True)
    for name in RECORDER_SCRIPTS:
        shutil.copy2(LLOYD_HOME / "scripts" / name, repo / "scripts" / name)
    return repo


def _iv_db(path: Path, *, healthy: int, dropped: int) -> None:
    """`healthy` graded observer calls plus `dropped` deadline misses, local-naive.

    The dropped rows carry `error` set with `action='noop'`, which is how
    `app/inner_voice/observer.py` records a verdict that never landed and what makes
    `iv_grade.py`'s `cost.errors` counter count them as drops. A `noop` with no
    `error` is the healthy majority and must not reach that numerator.
    """
    now = datetime.datetime.now() - datetime.timedelta(hours=1)
    with sqlite3.connect(path) as con:
        con.execute("""CREATE TABLE inner_voice_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL, sequence_in_turn INTEGER NOT NULL,
            trigger TEXT NOT NULL, action TEXT NOT NULL, reason TEXT,
            content TEXT, related_tool TEXT, input_tokens INTEGER,
            output_tokens INTEGER, cache_read INTEGER, cache_create INTEGER,
            latency_ms INTEGER, model TEXT, error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        is_drops = [False] * healthy + [True] * dropped
        for i, is_drop in enumerate(is_drops, start=1):
            con.execute(
                "INSERT INTO inner_voice_observations (id, session_id, turn_id,"
                " sequence_in_turn, trigger, action, reason, content, related_tool,"
                " input_tokens, output_tokens, cache_read, cache_create,"
                " latency_ms, model, error, created_at) VALUES (?,?,?,?,?,?,?,?,?,"
                "?,?,?,?,?,?,?,?)",
                (i, "iv_sess_guard", f"t{i}", i, "assistant_message", "noop", "",
                 None, None, 1000, 5, 0, 0,
                 12000 if is_drop else 400, "eco",
                 "timeout after 12.0s: x" if is_drop else None,
                 now.isoformat(timespec="microseconds")))
        con.commit()


def _iv_series(path: Path, rates) -> None:
    """The prior nights, written straight to the series file.

    Only the newest night's exit code is in question; these are read back through
    `_prior_series`, which takes the rates from `dropped_rate` and nothing else, so
    each line is one night's rate rather than a replay of a whole grader report. Its
    third return — was-the-newest-row-a-breach — re-derives from that row's own
    recorded `threshold`, and a row written by this helper carries none, so a breach
    built here reads as just-started and the recorder tries to announce it. That is
    why `_iv_record` mutes the room.
    """
    with path.open("w", encoding="utf-8") as fh:
        for i, rate in enumerate(rates):
            fh.write(json.dumps({
                "since": (datetime.datetime.now()
                          - datetime.timedelta(days=len(rates) - i)).isoformat(
                              timespec="seconds"),
                "llm_calls": 200, "dropped_verdicts": round(rate * 200),
                "dropped_rate": rate,
            }) + "\n")


def _iv_record(repo: Path, series: Path) -> subprocess.CompletedProcess:
    """The documented pipeline, verbatim, as one bash process substitution-free pipe.

    `HOME` is the fixture root so `cd ~/lloyd` lands in `repo`, and the recorder's
    exit status is the returned status: bash reports the last command in a pipeline,
    which is the recorder — the same thing the nightly's Bash tool sees.

    The two room mutes are load-bearing, not hygiene. The degraded fixture below
    breaches, and a row written by `_iv_series` carries no `threshold`, so
    `_prior_breaching` reads that breach as just-started and the recorder reaches its
    `announce()` — journal and desktop toast, both live channels. Today the fan-out
    also fails a second way (the throwaway repo has no `agent-services/guardian/` to
    import), and a run with only that to protect it is how
    `tests/test_iv_metrics_series.py` put 23 real breach lines into this machine's
    journal on 2026-09-20.
    """
    since = (datetime.datetime.now() - datetime.timedelta(days=2)).strftime(
        "%Y-%m-%dT%H:%M:%S")
    script = (f"cd ~/lloyd && python3 scripts/iv_grade.py --json --since '{since}'"
              f" | python3 scripts/iv_metrics_record.py --out '{series}'")
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=False,
        cwd=str(repo.parent),
        env={"HOME": str(repo.parent), "PATH": "/usr/bin:/bin:/usr/local/bin",
             "LLOYD_JOURNAL_ALERTS": "0", "LLOYD_DESKTOP_ALERTS": "0"})


def test_the_recorder_breaches_a_sustained_00526_median():
    """A week whose median is the degraded stretch's 0.0526 exits 2 — what 0.10 could not.

    The night being recorded dropped 1 verdict over 19 LLM calls (0.0526) on top of
    six prior nights from 2026-09-05..09-11, so the median of the 7-row window is
    0.0526: over the 0.05 bound, on at least 3 nights, which is the sustained
    condition only the recorder can see. `flagged` is true on the night itself —
    `flagged` is the per-night field, `breach` is the exit code, and the two are
    different decisions.
    """
    with tempfile.TemporaryDirectory() as td:
        repo, series = _iv_repo(td), Path(td) / "lloyd" / "iv-metrics.jsonl"
        _iv_db(repo / "usage.db", healthy=18, dropped=1)
        _iv_series(series, IV_DEGRADED_PRIOR_RATES)
        p = _iv_record(repo, series)
        assert p.returncode == 2, (
            f"a 7-row median of 0.0526 must exit 2; got {p.returncode}, "
            f"stdout {p.stdout[:200]!r} stderr {p.stderr[-300:]!r}")
        row = json.loads(series.read_text().splitlines()[-1])
        assert row["breach"] is True and row["flagged"] is True, row
        assert abs(row["dropped_rate"] - 1 / 19) < 1e-3, row
        assert abs(row["threshold"] - IV_METRICS.DEFAULT_THRESHOLD) < 1e-9, row
        # The token the autonomy runner's failure detector actually matches on.
        assert "exit code 2" in p.stdout, p.stdout[:200]
        assert "BREACH" in p.stderr, p.stderr[-200:]
    print("test_the_recorder_breaches_a_sustained_00526_median: OK")


def test_the_recorder_stays_quiet_on_the_recorded_healthy_history():
    """Tightening the bound must not make a clean week look like a bad one.

    Same recorder, same 7-row window, the eight nights actually recorded since
    2026-09-12 (max 0.0203) plus a newest night of 1 drop over 67 calls — the 0.0149
    the live file carries for 2026-09-18..09-20. Window median 0.0076, well under
    0.05: not flagged, not a breach, exit 0. A nightly that trips on healthy history
    is how an alert gets muted, and the runner's detector matches the literal
    "exit code" in the output, so its absence is asserted rather than inferred from
    the status.
    """
    with tempfile.TemporaryDirectory() as td:
        repo, series = _iv_repo(td), Path(td) / "lloyd" / "iv-metrics.jsonl"
        _iv_db(repo / "usage.db", healthy=66, dropped=1)
        _iv_series(series, IV_RECORDED_HEALTHY_RATES)
        p = _iv_record(repo, series)
        assert p.returncode == 0, (
            f"the recorded healthy history must exit 0; got {p.returncode}, "
            f"stdout {p.stdout[:200]!r} stderr {p.stderr[-300:]!r}")
        row = json.loads(series.read_text().splitlines()[-1])
        assert row["breach"] is False and row["flagged"] is False, row
        assert abs(row["dropped_rate"] - IV_RECORDED_HEALTHY_LATEST) < 1e-3, row
        assert "exit code" not in p.stdout, p.stdout[:200]
        assert "BREACH" not in p.stderr, p.stderr[-200:]
    print("test_the_recorder_stays_quiet_on_the_recorded_healthy_history: OK")


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
    test_observation_rows_keep_the_local_naive_writer_clock,
    test_the_recorder_breaches_a_sustained_00526_median,
    test_the_recorder_stays_quiet_on_the_recorded_healthy_history,
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
