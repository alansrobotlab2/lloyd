"""The context-overflow recovery ladder must not be gated by #800's intra-turn latch.

Backlog #800 latches ONE call site: the per-iteration relief pass, which fired on
every iteration of a long turn (2,474 firings in 8.78 hours on
`logs/server.err.1` as re-measured 2026-09-20; 1,966 in ~8.2 hours at this item's
2026-09-19 triage) and rewrote cached history each time. The latch is scoped to
`reason="intra_turn"` and to nothing else — and that scoping is load-bearing,
because the recovery below is the last thing between a long round and a dead turn:
three autocode rounds died at the wall on 2026-09-11 (866-a, 869, 875) for want of
it. A latch that reached the overflow path would shut the ladder at the one moment
the turn is CERTAIN to overflow.

Read `_relief_harness.py` first. The three tests below drive the real `run_query`
through a real rejection, so what they assert is the shape of the ladder the loop
itself asks for. The older recovery tests in `app/harness/tests/test_overflow_recovery.py`
use an artificial meter and a scripted ladder, which is the right shape for what they
pin and the wrong shape for these: only this driver can show that the recovery call
was handed NO latch on a turn where the per-iteration pass had already latched.

Why this file is at `tests/` and not beside the module it tests: the review rung's
`met`-node rail (`scripts/automod/review.py::_node_rail`) admits a test file only
under `tests/`, so a test that lives under `app/harness/tests/` cannot carry a
clause however green it is — and the two directories are in ONE pytest run
(`pytest.ini` lists both as testpaths), so relocation changes no test's execution,
only which directory it is read from. See backlog #1322 for the rail itself.
"""
import asyncio

import pytest

from app.harness import tool_search_cache
from app.harness.errors import ContextOverflowError
from app.harness.loop import _relieve_context
from tests._relief_harness import (
    BAND, TC, TRIGGER, WALL, relief_harness, seeded_history)


@pytest.fixture(autouse=True)
def _reset_cache():
    asyncio.run(tool_search_cache.clear())
    yield
    asyncio.run(tool_search_cache.clear())


@pytest.fixture
def harness(monkeypatch):
    """The shared relief driver — see `tests/_relief_harness.py`.

    The driver seeds the meter at an exact token level and records, for each ladder
    call, whether it was handed a latch — which is the only way to show that the
    per-iteration pass latched THIS turn and the recovery still ran.
    """
    return relief_harness(monkeypatch)


# -- the overflow ladder ignores the #800 intra-turn latch (clause 3) -------
#
# #800 latches the INTRA-TURN relief pass: one pass per crossing of the target
# instead of one per iteration, because each pass rewrites cached history and
# costs a re-prefill. The latch is scoped to the per-iteration call and nothing
# else. That scoping is load-bearing: three autocode rounds died at the wall on
# 2026-09-11 (866-a, 869, 875), and the recovery below is what catches that. A
# latch that reached this path would shut the ladder at the one moment the turn
# is certain to overflow.

def test_the_overflow_ladder_runs_whole_after_the_latch_already_closed_a_pass(harness):
    """Clause 3, driven: a turn that already spent its one intra-turn pass still runs
    the WHOLE ladder when a request fails on context length.

    The turn crosses target (latch closes after pass 1), then request 2 is rejected at
    300,000 tokens the way the engine rejects an oversized prompt. The recovery is the
    only thing between this round and a dead turn, so the closed latch must not touch
    it — and it does not: the overflow call is recorded with `latch=None` and all four
    rungs, which is the same shape as the pre-latch behaviour the 2026-09-11 rounds
    (866-a, 869, 875) needed and did not get.
    """
    run = harness([("a", [TC(1)]), ("b", [TC(2)]), ("done", [])],
                  level=BAND, max_turns=5, overflow_times=1, overflow_at=2)

    intra = run.intra_passes()
    assert len(intra) == 1, [(c["used"], c["latched"]) for c in run.intra_calls()]

    over = [c for c in run.ladder_calls() if c["reason"] == "overflow"]
    assert len(over) == 1, run.calls
    assert over[0]["latch"] is None, over[0]        # not gated by the pass count
    assert over[0]["rearm"] is None, over[0]        # no re-arm level was recorded either,
    # because none was consulted: `run_query` hands the overflow rung no latch, so
    # `_relieve_context` took no per-turn bound and was free to descend all four rungs.
    assert any(r.startswith("tool_results:") for r in over[0]["rungs"]), over[0]
    assert any(r.startswith("reasoning") for r in over[0]["rungs"]), over[0]
    assert any(r.startswith("truncate") for r in over[0]["rungs"]), over[0]
    # Started from the size the ENGINE reported (300,000), not from the meter's last
    # quiet reading: the recovery adopts the rejection's number as its anchor, which is
    # why it can reach all four rungs and still be over target afterwards.
    assert over[0]["used"] == WALL, over[0]
    assert over[0]["freed_tokens"] > 0, over[0]
    # The turn still finished — the recovery did its job.
    assert run.script.calls == 3, run.script.calls   # +1 for the re-requested turn


def test_a_latched_turn_still_ends_context_exhausted_when_recovery_cannot_keep_up(harness, caplog):
    """Clause 3's other half: the latch must not swallow the terminal verdict.

    Same turn, but every request after the first is rejected, past
    `max_context_overflow_recoveries` (2). Before the latch this turn also had extra
    intra-turn passes to spend, and it still died — so the latch is not what decides
    this, and after the latch the turn must STILL report the exhausted recoveries.
    That warning is the only surface naming the wall; a round that hits it silently is
    how 2026-09-11's round 866-a came back looking like a model failure.
    """
    caplog.set_level("INFO", logger="lloyd-harness-loop")
    run = harness([("a", [TC(1)]), ("b", [TC(2)]), ("done", [])],
                  level=BAND, max_turns=5, overflow_times=3, overflow_at=2)

    over = [c for c in run.ladder_calls() if c["reason"] == "overflow"]
    assert len(over) == 2, run.calls            # recovery ceiling, unchanged by the latch
    assert all(c["latch"] is None for c in over), over
    # 1 accepted + 3 rejected: two recoveries, and the request re-issued after the
    # second one is the request that raises.
    assert run.script.calls == 4, run.script.calls
    assert isinstance(run.raised, ContextOverflowError), run.raised
    given_up = [r.getMessage() for r in caplog.records if "giving up" in r.getMessage()]
    assert len(given_up) == 1, given_up
    assert "after 2 recovery attempts" in given_up[0], given_up[0]
    # The intra-turn latch still did its one pass before any of this.
    assert len(run.intra_passes()) == 1, run.calls


def test_the_overflow_recovery_call_site_is_the_one_that_must_stay_unlatched(harness):
    """Clause 3's boundary, pinned by what the loop hands the ladder on a driven turn.

    The recovery ladder is the last thing between a round and `context_exhausted`, so
    it must run the whole ladder whether or not relief already ran that turn. A latch
    handed to the `reason="overflow"` call would silently disable the recovery on every
    turn that had already run one intra-turn pass, which is every long round.

    This used to read `run_query`'s source with `ast` (P13.0 replaced it): the driver
    records every `_relieve_context` call the REAL loop makes, with the latch object it
    was handed, so the same properties are asserted as behaviour —

      * every per-iteration (`intra_turn`) call of the turn is handed a latch, and it
        is ONE latch object for the whole turn (a latch minted per iteration would
        re-open on every iteration and gate nothing);
      * the latch re-arms at the level the knobs compute (`TRIGGER`, 0.8 of the
        threshold), not at a literal;
      * the `overflow` call on the same turn is handed no latch at all.
    """
    import inspect

    run = harness([("a", [TC(1)]), ("b", [TC(2)]), ("c", [TC(3)]), ("done", [])],
                  level=BAND, max_turns=6, overflow_times=1, overflow_at=3)

    intra = run.intra_calls()
    assert len(intra) >= 2, [(c["used"], c["latched"]) for c in intra]
    latches = {id(c["latch"]) for c in intra}
    assert None not in [c["latch"] for c in intra], intra
    assert len(latches) == 1, f"one latch per turn, found {len(latches)}"
    assert run.intra_passes() and run.intra_passes()[0]["rearm"] == TRIGGER, (
        run.intra_passes())

    over = run.by_reason("overflow")
    assert len(over) == 1, run.calls
    assert over[0]["latch"] is None, (
        "the context-overflow recovery was handed a latch; a latched recovery runs a "
        "partial ladder and kills a round that could still have been saved")

    # Unbounded means what it says: `_relieve_context` has no per-call rung bound to
    # be passed, so "no latch" is the whole of the bound. If someone adds a `stop_at`
    # -style parameter, "no latch" stops meaning "all four rungs" and this is the
    # assertion that goes red and says so.
    rung_bounds = {"stop_at", "max_rung", "rungs", "until"}
    assert not rung_bounds & set(inspect.signature(_relieve_context).parameters), (
        "a per-call rung bound appeared on _relieve_context; the overflow rung is "
        "unbounded only while no such parameter exists")


def test_the_terminal_inject_ladder_runs_whole_after_the_latch_closed_a_pass(harness):
    """The third wall path — the terminal inject — is unlatched too, and the turn
    that still cannot fit still ends `context_exhausted` there.

    Clause 3 names the overflow reason, and the two tests above pin it. This pins the
    adjacent literal in the same branch: the relief pass a terminal inject runs when
    the injected message would not fit (`reason="terminal_inject"`) is the OTHER
    `stop_reason = "context_exhausted"` in the loop — reached only when relief has
    already run and the inject still does not fit. If the latch were ever widened to
    every call site, a turn that had spent its one intra-turn pass would arrive here
    with the ladder closed and inject a message the model cannot answer, which is the
    2026-09-11 round-875 shape (the loop continued on a full window, the model re-sent
    a cut-off heredoc, vLLM 400'd).

    The history carries 16 small tool results so the per-iteration pre-check
    (`tool_count >= 15`) is met and the turn really does latch, and each result is
    under the rungs' `min_chars` floor of 2,000 so every pass frees ~0: the injected
    message is then genuinely refused rather than rescued, which is what makes the
    `context_exhausted` assertion about the giving-up path and not about how much the
    driver happened to clear. The level sits above `truncation_threshold`, so the
    same branch's pre-request pass runs too — asserted unlatched in the same breath,
    because it is the other literal in that branch.
    """
    msgs = seeded_history(results=16, result_chars=200, reasoning=0)
    run = harness([("a", [TC(1)]), ("done", [])], level=WALL - 6_000,
                  growth_chars=0, max_turns=5, inject_once=True,
                  history=msgs)

    assert run.raised is None, run.raised
    assert run.stop_reason() == "context_exhausted", run.events[-1]
    # The per-iteration pass ran exactly once this turn and freed nothing: after this
    # the latch is closed, and every wall-path call below has to ignore it.
    assert len(run.intra_passes()) == 1, run.intra_calls()
    assert run.intra_passes()[0]["freed_tokens"] == 0, run.intra_passes()[0]
    for reason in ("pre_request", "terminal_inject"):
        got = run.by_reason(reason)
        assert len(got) >= 1, (reason, run.calls)
        assert all(c["latch"] is None for c in got), (reason, got)
        assert all(c["rearm"] is None for c in got), (reason, got)
        # Given nothing to take, so the inject was refused on headroom, not rescued.
        assert all(c["freed_tokens"] == 0 for c in got), (reason, got)
    term = run.by_reason("terminal_inject")[0]
    assert term["used"] > WALL - 6_000, term          # still at the wall after the pass


def test_the_rejected_size_reaches_rung_one(harness, monkeypatch):
    """D6: the overflow recovery's rung 1 budgets from the size the engine
    REJECTED, not from the turn's peak `input_tokens`.

    The recovery feeds the rejection's `requested_input_tokens` into the meter,
    but rung 1 used to read `total_usage["input_tokens"]` instead — the peak of
    requests that SUCCEEDED, which never includes the rejected one. Here the
    engine reports no usage at all, so that peak was 0: rung 1 saw only its own
    estimate, sat under its trigger, and cleared nothing at the one moment the
    turn was certain to be over the wall. Reading the meter, it sees 300,000.
    """
    from app.harness import loop as loop_mod

    real = loop_mod._intra_turn_microcompact
    seen: list[tuple[int, int]] = []

    def _spy(chat_messages, *, meter, **kw):
        used = int(meter.used)
        cleared = real(chat_messages, meter=meter, **kw)
        seen.append((used, cleared))
        return cleared

    monkeypatch.setattr(loop_mod, "_intra_turn_microcompact", _spy)
    run = harness([("a", [TC(1)]), ("b", [TC(2)]), ("done", [])],
                  level=BAND, max_turns=5, overflow_times=1, overflow_at=2)

    over = [c for c in run.ladder_calls() if c["reason"] == "overflow"]
    assert len(over) == 1, run.calls
    at_wall = [(used, cleared) for used, cleared in seen if used >= WALL]
    assert len(at_wall) == 1, seen
    assert at_wall[0][1] > 0, (
        f"rung 1 ran at {at_wall[0][0]} tokens and cleared nothing: {seen}")
