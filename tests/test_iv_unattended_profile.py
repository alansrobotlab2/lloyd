"""The observer, on a turn nobody is reading — what survived R5.

Three losses on 2026-09-11, all from the observer treating an autocode round
exactly as it treats a chat:

  * **Round 874** — injected "deliver the final report now" on the invented
    premise "working tree clean". Nobody reads a worker turn's prose; the
    harness finalizer produces the report. The model abandoned a healthy
    round at iteration 38 with 44 minutes left.
  * **16 repetition fires in one day**, 7 on the round id alone. `SM_…`
    survives ambient stripping and is 18 characters, so `_is_distinctive`
    lets it carry a near match by itself — and every call in a round
    mentions it. One turn hit the deterministic cap of 5 as its gate started.
  * **Round 875** — injected on a silent terminal iteration at 241k of
    262,144 tokens, with no room left to answer in.
"""

from __future__ import annotations

import pytest

from app.harness.context_meter import ContextMeter
from app.inner_voice import guards as G
from app.inner_voice import observer as O


# ---------------------------------------------------------------------------
# ambient by pattern
# ---------------------------------------------------------------------------

def _sig(tool: str, args):
    return G.tool_call_signature(tool, args)


def test_the_round_id_is_stripped_from_every_idiom():
    """`_strip_cd_prefix` only ever reached Bash. The round id arrives in an
    MCP argument too — `graph_affected(root=…)`, `automod_gate_wait(round_id=…)`
    — and there it survived, un-ambient, carrying matches on its own.
    """
    for text in (
        "cd /home/alansrobotlab/lloyd-work/SM_20260908_165950/home/lloyd && pytest",
        "root=/home/alansrobotlab/lloyd-work/SM_20260908_165950/home/lloyd",
        "round_id=SM_20260912_035714",
        "W=/home/alansrobotlab/lloyd-work/SM_20260911_190850/home/lloyd; $W/x.py",
    ):
        assert "SM_2026" not in G._strip_ambient(text), text


def test_no_signature_carries_the_round_id_as_an_identifier():
    s = _sig("graph_affected", {"root": "/home/x/lloyd-work/SM_20260908_165950/home/lloyd",
                                "symbol": "run_query"})
    assert not any(t.startswith("sm_2026") for t in s.idents)
    assert not any(t.startswith("sm_2026") for t in s.all_idents)
    # and the real subject survives
    assert "run_query" in s.idents


def test_unrelated_calls_sharing_only_the_round_id_do_not_fire():
    """The exact 2026-09-11 shape: four unrelated MCP calls, one worktree."""
    W = "/home/alansrobotlab/lloyd-work/SM_20260911_190850/home/lloyd"
    sigs = [
        _sig("Bash", {"command": f"cd {W} && git status"}),
        _sig("Bash", {"command": f"cd {W} && pytest tests/test_alpha.py"}),
        _sig("Bash", {"command": f"cd {W} && ruff check app/"}),
        _sig("Bash", {"command": f"cd {W} && git diff --stat"}),
    ]
    assert G.repetition_verdict(sigs) is None


def test_a_bare_call_with_no_identifiers_does_not_un_ambient_everything():
    """One bare `ls` in the ring emptied the intersection, so NOTHING read as
    ambient and whatever every other call shared went back to distinctive.

    A call that mentions nothing is evidence about nothing: it must abstain,
    not vote.
    """
    shared = "my_shared_worktree_token"
    ring = [_sig("Bash", {"command": f"{shared} echo alpha_{i}_marker"})
            for i in range(9)]
    assert shared in G.ubiquitous_identifiers(ring), "precondition"

    ring.insert(3, _sig("Bash", {"command": "ls"}))
    ambient = G.ubiquitous_identifiers(ring)
    assert shared in ambient, (
        "one argument-free call emptied the ambient intersection")


def test_an_ambient_term_cannot_carry_a_match():
    """The consequence the test above protects: with the shared term ambient,
    four otherwise-unrelated calls do not read as repetition.
    """
    shared = "my_shared_worktree_token"
    ring = [_sig("Bash", {"command": f"{shared} echo alpha_{i}_marker"})
            for i in range(9)]
    ambient = G.ubiquitous_identifiers(ring)
    assert G.repetition_verdict(ring[-4:], ambient=ambient) is None


def test_a_genuine_repeat_still_fires():
    """The guard must not be defanged: this is what it exists for."""
    sigs = [_sig("Bash", {"command": "grep -rn iv_inject_queue app/"})
            for _ in range(4)]
    v = G.repetition_verdict(sigs)
    assert v is not None
    assert v.exact


# ---------------------------------------------------------------------------
# polling exemption
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool", sorted(G.REPETITION_EXEMPT_TOOLS))
def test_polling_tools_are_exempt(tool):
    sigs = [_sig(tool, {"round_id": "SM_20260912_035714"}) for _ in range(6)]
    assert G.repetition_verdict(sigs) is None


def test_a_namespaced_polling_tool_is_exempt_too():
    sigs = [_sig("mcp__lloyd-mcp__automod_gate_wait", {"round_id": "X"})
            for _ in range(6)]
    assert G.repetition_verdict(sigs) is None


def test_the_exemption_is_additive_not_a_replacement():
    """Config adds a polling tool; it never un-exempts `automod_gate_wait`."""
    sigs = [_sig("automod_gate_wait", {"round_id": "X"}) for _ in range(6)]
    extra = G.REPETITION_EXEMPT_TOOLS | frozenset({"my_new_poller"})
    assert G.repetition_verdict(sigs, exempt_tools=extra) is None


def test_a_non_exempt_tool_still_fires_under_an_exempt_list():
    sigs = [_sig("Bash", {"command": "pytest tests/test_x.py"}) for _ in range(4)]
    assert G.repetition_verdict(
        sigs, exempt_tools=G.REPETITION_EXEMPT_TOOLS) is not None


# ---------------------------------------------------------------------------
# context pressure
# ---------------------------------------------------------------------------

def _meter(fraction_of_window: float, window: int = 262_144) -> ContextMeter:
    m = ContextMeter(window)
    msgs = [{"role": "user", "content": "x"}]
    m.observe_usage({"input_tokens": int(window * fraction_of_window)}, len(msgs))
    m.observe_append(msgs)
    return m


def test_no_meter_is_never_critical_and_never_exhausted():
    """Fails open: a turn whose context position is unknown is judged
    exactly as it always was.
    """
    cp = G.context_pressure(None)
    assert cp.critical is False
    assert cp.exhausted is False


def test_an_unmeasured_meter_is_never_critical():
    cp = G.context_pressure(ContextMeter(262_144))
    assert cp.critical is False
    assert cp.exhausted is False


def test_a_quiet_turn_is_not_critical():
    assert G.context_pressure(_meter(0.20)).critical is False


def test_a_pressed_turn_is_critical_but_not_exhausted():
    cp = G.context_pressure(_meter(0.75), floor_tokens=12_000)
    assert cp.critical is True
    assert cp.exhausted is False


def test_875s_position_is_exhausted():
    """241k of 262,144 — where the observer injected into a turn that could
    not answer.
    """
    m = ContextMeter(262_144)
    msgs = [{"role": "user", "content": "x"}]
    m.observe_usage({"input_tokens": 255_000}, len(msgs))
    m.observe_append(msgs)
    cp = G.context_pressure(m, floor_tokens=12_000)
    assert cp.exhausted is True


def test_a_broken_meter_fails_open():
    class _Broken:
        measured = True
        window = 262_144

        @property
        def used(self):
            raise RuntimeError("boom")

    cp = G.context_pressure(_Broken())
    assert cp.critical is False and cp.exhausted is False


def test_the_loop_and_the_observer_read_one_floor():
    """Two floors would mean the observer speaking into a turn the loop has
    already decided to end.
    """
    from app.config import CONFIG

    configured = int(((CONFIG.get("harness") or {}).get("context_relief") or {})
                     .get("terminal_floor_tokens", 12_000))
    assert O._context_floor_tokens() == configured


# ---------------------------------------------------------------------------
# the unattended profile, retired (IV plan R5)
#
# The observer's own unattended profile — deterministic terminal words, the
# unattended cancel gate, the PLATFORM note — was deleted on 2026-09-24, while
# the observer was off for every worker. Its words moved to the turn guards,
# which run on every turn (tests/test_turn_guards.py). When the observer came
# back on for autocode and autotriage (2026-09-25) a narrower note and rail came
# back with it (tests/test_iv_worker_observer.py). What stays here is what
# still applies to any observed turn.
# ---------------------------------------------------------------------------

def _state(**kw) -> O.ObserverState:
    import asyncio

    base = dict(
        session_id="s", turn_id="t", user_request="do it",
        chat_messages_handle=[], cancel_event=asyncio.Event(),
        cfg={}, platform="mission-control",
    )
    base.update(kw)
    return O.ObserverState(**base)


def test_the_profile_is_gone_and_its_words_moved_to_the_guards():
    assert not hasattr(O, "_is_unattended")
    assert not hasattr(O, "_platform_note")
    assert G.stall_rescue_content(unattended=True, round_open=True) \
        == G.UNATTENDED_ROUND_OPEN_CONTENT
    assert "automod_gate" in G.UNATTENDED_ROUND_OPEN_CONTENT


def test_a_destructive_cancel_is_still_allowed_with_no_injects():
    """The one cancel that is legitimate with nothing behind it."""
    st = _state()
    d = O.ObserverDecision(
        action="cancel", reason="primary is running rm -rf on the vault")
    O._apply_decision_guards(st, d, trigger="assistant_message",
                             tool_calls=[], is_terminal=False)
    assert d.action == "cancel"


def test_a_cancel_about_intent_is_not_observable():
    st = _state()
    d = O.ObserverDecision(action="cancel", reason="the primary is off track")
    O._apply_decision_guards(st, d, trigger="assistant_message",
                             tool_calls=[], is_terminal=True)
    assert d.action == "noop_cancel_not_observable"


def test_the_context_note_prefers_silence():
    st = _state(context_meter=_meter(0.80),
                cfg={"context_pressure_threshold": 0.5})
    note = O._context_pressure_note(st)
    assert "CONTEXT PRESSURE" in note
    assert "prefer noop" in note


def test_no_context_note_without_a_meter():
    assert O._context_pressure_note(_state(context_meter=None)) == ""
