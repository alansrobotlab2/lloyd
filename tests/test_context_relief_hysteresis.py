"""The intra-turn context-relief latch: one pass per crossing, not per iteration.

Backlog #800. What the ladder was gated on, before this file, was a STATE:

    if meter.used > _relief_target(...): _relieve_context(...)

A state is not an edge. A long turn sits above that target for the rest of its life —
the pass cannot get under it, because `target` is 126,086 (60% of the 210,144
threshold) while ~44k of system prompt and tool schemas ride inside the meter's
reading and are not in the message list the rungs are allowed to clear — so once a
turn crossed the target the ladder ran on EVERY remaining iteration. Measured
2026-09-20 over `logs/server.err.1` (file spans 09:32:58 → 18:54:28; the firings
run 10:04:29 → 18:51:02, 8.78 h): 2,474 firings = 282 an hour, 73.9% of which
freed under 2,000 tokens and 87.1% of which fired below 145,699, the re-arm level
implied by the `target 109,274` those same lines print (threshold 182,123 × the
0.8 fraction). The item's triage reading one day earlier was 1,966 firings / 240 an
hour / 97.0% below trigger — the level of the absolute numbers moves with whatever
window the writing process resolved; the band does not. Each firing edited
messages near the front of a ~150k-token prompt — `reasoning` is named in every
one of those 2,474 lines — which is the rewrite #520 was filed against.

These tests pin the replacement: the ladder re-enters only when the reading regrows
to the intra-turn trigger (`truncation_threshold(window) x
intra_turn_microcompact_trigger_fraction` = 168,115 here), which is the number
`_intra_turn_microcompact` already uses for its own trigger — so the latch releases
exactly where the ladder would have fired anyway, and no new knob exists.

Read `_relief_harness.py` first. The tests depend on its central property: the
meter's reading rises because the turn appends real content, and falls only because
a rung removed real content. A harness that held the level flat would show one pass
per turn for the wrong reason and could not tell a latch from a fix.

Why this file is at `tests/` and not beside the module it tests: the review rung's
`met`-node rail (`scripts/automod/review.py::_node_rail`) admits a test file only
under `tests/`, so a test that lives under `app/harness/tests/` cannot carry a
clause however green it is — and the two directories are in ONE pytest run
(`pytest.ini` lists both as testpaths), so relocation changes no test's execution,
only which directory it is read from. `test_overflow_recovery.py` lives here for the
same reason. See backlog #1322 for the rail itself.
"""
import asyncio

import pytest

from app.harness import tool_search_cache
from app.harness.context_meter import ContextMeter
from app.harness.loop import (
    _ReliefLatch, _relieve_context, _relief_rearm_level, _relief_target)
from app.harness.options import RunOptions
from tests._relief_harness import (
    BAND, SMALL_GROWTH_CHARS, TARGET, THRESHOLD, TRIGGER, WINDOW, TC, meter_at,
    relief_harness, seeded_history)

# The harness logs to this name (`app/harness/loop.py`'s own logger); a `logger=`
# filter on the MODULE name would capture nothing at all.
LOOP_LOGGER = "lloyd-harness-loop"
_GROWTH_SMALL = SMALL_GROWTH_CHARS // 4 + 69   # what one iteration adds; see harness


@pytest.fixture(autouse=True)
def _clean_cache():
    asyncio.run(tool_search_cache.clear())
    yield
    asyncio.run(tool_search_cache.clear())


@pytest.fixture
def harness(monkeypatch):
    return relief_harness(monkeypatch)


# ── clause 1: one pass per crossing, not one per iteration ─────────────────────

def test_after_a_relief_pass_the_ladder_stays_closed_below_the_trigger(harness):
    """The bug, in one test. A turn that starts over `target` and stays under
    `trigger` — the interval 87.1% of `server.err.1`'s 2,474 firings re-measured
    2026-09-20 sat in — and grows
    by ~2,069 tokens an iteration. Before the latch the loop asked once per
    iteration and the ladder answered every time, because the gate was a state.

    The growth matters: this is not a row of identical checks frozen at one token
    level. Each iteration appends a real result, the reading climbs, and the latch is
    what stops that climb from re-entering a ladder whose target it can never reach
    again. Small growth is deliberate: the turn cannot reach `trigger` within its own
    lifetime, so the refusals below are load-bearing rather than incidental.
    """
    run = harness([("a", [TC(1)]), ("b", [TC(2)]), ("c", [TC(3)]),
                   ("d", [TC(4)]), ("done", [])],
                  level=BAND, growth_chars=SMALL_GROWTH_CHARS, max_turns=5)

    calls = run.intra_calls()
    passes = run.intra_passes()
    assert len(passes) == 1, [(c["used"], c["latched"]) for c in calls]
    assert passes[0]["used"] == BAND + _GROWTH_SMALL, passes[0]
    assert passes[0]["rungs"], "the one pass must have actually run a rung"
    assert passes[0]["freed_tokens"] > 0, passes[0]
    assert passes[0]["after"] < passes[0]["used"], passes[0]

    refusals = run.intra_refusals()
    assert len(refusals) == len(calls) - 1, [(c["used"], c["latched"]) for c in calls]
    assert refusals, calls
    # Each refused check sits higher than the pass that closed the latch: the turn
    # kept growing while the latch held. That is the latch holding on a live turn.
    assert all(c["used"] > TARGET for c in refusals), [c["used"] for c in refusals]
    assert all(c["used"] < TRIGGER for c in refusals), [c["used"] for c in refusals]
    assert all(c["rungs"] == [] for c in refusals), [c["rungs"] for c in refusals]
    assert [c["used"] for c in refusals] == sorted(
        c["used"] for c in refusals), [c["used"] for c in refusals]
    # The refusals climb by the fake pool's own answer each iteration: the level
    # these assertions sit at is measured off the message list, not set by a mock.
    assert refusals[-1]["used"] > refusals[0]["used"] + _GROWTH_SMALL, refusals


def test_the_latch_releases_once_the_prompt_regrows_to_the_trigger(harness):
    """Clause 2: the latch is a hysteresis, not a one-shot fuse. In the same turn,
    once the reading reaches the trigger again, relief runs a SECOND time.

    Driven directly through `_relieve_context` with one meter and one latch, which is
    the whole state the real per-iteration call site hands in. The token levels are
    chosen rather than grown to, because the number an iteration actually adds is the
    inline tool-result window's own doing (the loop trims what goes back into a
    request, and the real log shows ~2,600 tokens an iteration at this size) — the
    clause is about what the latch does AT a level, and that is what this pins.
    """
    msgs = seeded_history(18)
    meter = meter_at(BAND, msgs)
    opts = RunOptions(model="primary", session_id="", tool_search_enabled=False,
                      preserve_thinking_iterations=6, chat_messages_handle=msgs)
    latch = _ReliefLatch()

    first = _relieve_context(msgs, options=opts, meter=meter, reason="intra_turn",
                             keep_recent=15, tool_count=18, iteration=3, latch=latch)
    assert not first.get("latched"), first
    assert first["passes"] == 1, first
    assert first["used_after"] < TRIGGER, first      # still inside the band

    # Same turn, same meter, prompt grown back to just under the trigger: refused.
    meter.observe_usage({"input_tokens": TRIGGER - 1}, len(msgs))
    mid = _relieve_context(msgs, options=opts, meter=meter, reason="intra_turn",
                           keep_recent=15, tool_count=18, iteration=4, latch=latch)
    assert mid["latched"] is True, mid
    assert mid["passes"] == 1, mid                   # the refusal did not count

    # One token more — the trigger itself, which is `>=`-released: relief runs again.
    meter.observe_usage({"input_tokens": TRIGGER}, len(msgs))
    second = _relieve_context(msgs, options=opts, meter=meter, reason="intra_turn",
                              keep_recent=15, tool_count=18, iteration=5, latch=latch)
    assert not second.get("latched"), second
    assert second["passes"] == 2, second             # a SECOND pass, same turn
    # What a release means: the ladder ran again, and it was counted as a second
    # pass. The reading after the pass is the ladder's own re-derivation of the
    # pruned tree (`meter.resync`), which is legitimately lower than the trigger it
    # was released at — what the clause pins is that it ran and was counted.
    assert not second.get("latched"), second


def test_the_latch_is_closed_only_between_a_pass_and_the_rearm_level():
    latch = _ReliefLatch()
    assert latch.passes == 0 and latch.rearm == 0
    assert not latch.closed(TRIGGER + 5)       # no pass has run: never closed
    assert not latch.closed(0)                # ditto, at the absurd end

    latch.passes = 1
    latch.rearm = TRIGGER
    assert not latch.closed(TRIGGER)          # RELEASED at the level: clause 2's ">="
    assert not latch.closed(TRIGGER + 1)      # released above it
    assert latch.closed(TRIGGER - 1)          # still closed one token below
    assert latch.closed(BAND)
    assert latch.announced is False


def test_the_rearm_level_is_the_intra_turn_trigger_from_the_existing_knob():
    """`truncation_threshold(window) x trigger_fraction`, both read from where the
    loop already reads them — no new config key.

    It is NOT derived from `target`: `target / 0.6 * 0.8` truncates to 168,114 here,
    one token below the real trigger, and a latch that releases one token late is a
    latch whose boundary no other line of code agrees with.
    """
    opts = RunOptions(model="primary", session_id="")
    meter = ContextMeter(WINDOW)
    assert _relief_target(opts, meter) == TARGET
    assert _relief_rearm_level(opts, meter) == TRIGGER == int(THRESHOLD * 0.8)
    assert _relief_rearm_level(opts, meter) != int(TARGET / 0.6 * 0.8)
    assert TRIGGER < THRESHOLD

    bumped = RunOptions(model="primary", session_id="",
                        intra_turn_microcompact_trigger_fraction=0.9)
    assert _relief_rearm_level(bumped, meter) == int(THRESHOLD * 0.9)

    # Fail open: an unreadable threshold means "cannot compute", which the ladder
    # treats as "leave it open", never as "close forever at zero".
    class _NoThreshold:
        measured = True

        @property
        def threshold(self):
            raise RuntimeError("meter is not initialised")

    assert _relief_rearm_level(opts, _NoThreshold()) == 0


def test_the_relief_log_line_names_the_pass_count_and_the_rearm_level(caplog):
    """Clause 4. A latched turn must be distinguishable in `logs/server.err` from one
    that is simply under target, and the pass line must say how many passes this turn
    has run and at what level the next one is allowed.

    A quiet turn logs nothing at all, so without these two lines the two states are
    identical in the log — and the follow-up measurement (the drop from the 282
    events an hour `logs/server.err.1` measured on 2026-09-20) would have no way to tell a latch
    that worked from a day with no long rounds. The refusal line is worded so it
    does NOT match the grep for `loop: context relief (intra_turn)`: a held-off
    pass is not an event.
    """
    caplog.set_level("INFO", logger=LOOP_LOGGER)
    msgs = seeded_history(18)
    meter = meter_at(BAND, msgs)
    opts = RunOptions(model="primary", session_id="", tool_search_enabled=False,
                      preserve_thinking_iterations=6, chat_messages_handle=msgs)
    latch = _ReliefLatch()
    for iteration in (3, 4, 5):        # a third refusal: announcement is once per turn
        _relieve_context(msgs, options=opts, meter=meter, reason="intra_turn",
                         keep_recent=15, tool_count=18, iteration=iteration, latch=latch)

    pass_lines = [r.getMessage() for r in caplog.records
                  if "loop: context relief (intra_turn)" in r.getMessage()]
    assert len(pass_lines) == 1, pass_lines
    assert "pass 1 this turn" in pass_lines[0], pass_lines[0]
    assert f"next pass allowed at >={TRIGGER}" in pass_lines[0], pass_lines[0]

    latched = [r.getMessage() for r in caplog.records
               if "context relief latched" in r.getMessage()]
    assert len(latched) == 1, latched                      # once, not per iteration
    assert "below rearm" in latched[0] and str(TRIGGER) in latched[0], latched[0]


def test_the_release_level_follows_the_trigger_fraction_knob_not_a_literal():
    """Clause 2's derivation, driven. The release level must be
    `truncation_threshold(window) x intra_turn_microcompact_trigger_fraction` — the
    existing 0.8 knob, no new config key — not a number that happens to equal what
    that computes today.

    Asserting 168,115 alone cannot tell a derivation from a literal, so the knob is
    MOVED to 0.9 and the whole test is run at a level that discriminates between the
    two: 176,115 is past the default 0.8 level and short of 0.9's. Held there under
    the moved knob; let through under the default one (the last block, which is the
    half that fails if the level were hard-coded). A second independent number
    invented for the latch fails here and passes every test above.

    The level is re-anchored per step with `meter_at`, which is how the harness seeds
    a real `ContextMeter` at all: a synthetic engine report at the current history
    length. A rising prompt IS the engine reporting a bigger one, so re-anchoring
    upward is the real mechanism, not a mock of it.
    """
    moved = int(THRESHOLD * 0.9)                # 189,129
    assert moved != TRIGGER, "the knob has to move to a level that discriminates"
    between = TRIGGER + 8_000                   # 176,115: past 0.8, short of 0.9
    assert TRIGGER < between < moved, (TRIGGER, between, moved)

    msgs = seeded_history(18)
    opts = RunOptions(model="primary", session_id="", tool_search_enabled=False,
                      preserve_thinking_iterations=6, chat_messages_handle=msgs,
                      intra_turn_microcompact_trigger_fraction=0.9)
    latch = _ReliefLatch()

    first = _relieve_context(msgs, options=opts, meter=meter_at(between, msgs),
                             reason="intra_turn", keep_recent=18, tool_count=18,
                             iteration=1, latch=latch)
    assert first["rearm"] == moved, first       # published the MOVED level

    # Still held at `between`: one pass total. Under the default knob this is the
    # level at which the ladder would already have been let back in.
    held = _relieve_context(msgs, options=opts, meter=meter_at(between, msgs),
                            reason="intra_turn", keep_recent=18, tool_count=18,
                            iteration=2, latch=latch)
    assert held["latched"] is True, held
    assert held["rearm"] == moved, held
    assert latch.passes == 1, latch.passes

    # And it releases above the moved level, where the knob now says.
    released = _relieve_context(msgs, options=opts,
                                meter=meter_at(moved + 4_000, msgs),
                                reason="intra_turn", keep_recent=18, tool_count=18,
                                iteration=3, latch=latch)
    assert "latched" not in released, released
    assert latch.passes == 2, latch.passes

    # The same prompt level, the default knob: released. This is what makes the
    # assertion above about the knob rather than about the number.
    msgs2 = seeded_history(18)
    plain = RunOptions(model="primary", session_id="", tool_search_enabled=False,
                       preserve_thinking_iterations=6, chat_messages_handle=msgs2)
    latch2 = _ReliefLatch()
    _relieve_context(msgs2, options=plain, meter=meter_at(between, msgs2),
                     reason="intra_turn", keep_recent=18, tool_count=18,
                     iteration=1, latch=latch2)
    let_in = _relieve_context(msgs2, options=plain, meter=meter_at(between, msgs2),
                              reason="intra_turn", keep_recent=18, tool_count=18,
                              iteration=2, latch=latch2)
    assert "latched" not in let_in, (
        "at 0.8 the release level is 168,115, so a prompt at 176,115 must run a "
        f"second pass; it was held, which means the level is not knob-derived: {let_in}")
    assert latch2.passes == 2, latch2.passes


def test_a_meter_that_cannot_name_its_wall_does_not_publish_a_release_level(caplog):
    """The fail-open direction, made explicit rather than left to `rearm == 0`.

    `_relief_rearm_level` returns 0 when the meter has no readable `threshold`, and
    `closed()` is `passes > 0 and used < rearm` — with `rearm` at 0 that comparison is
    false for every reading, so the ladder would keep running on every iteration AND
    log `next pass allowed at >=0`, a level the turn is already past. That number is
    the whole content of clause 4's log field, so a degraded meter must not publish
    it: the pass is not counted, and the turn says once why the latch did not close.

    A meter that cannot answer for its window costs the drip it always cost, never a
    relief pass it was entitled to run.
    """
    caplog.set_level("WARNING", logger=LOOP_LOGGER)
    msgs = seeded_history(18)
    inner = meter_at(BAND, msgs)

    class _NoWall:
        """Real `used` and real `resync`, no `threshold`: a meter without a window."""
        measured = True

        def __init__(self, wrapped):
            self._wrapped = wrapped

        @property
        def used(self):
            return self._wrapped.used

        def resync(self, messages):
            self._wrapped.resync(messages)

        def observe_append(self, messages):
            self._wrapped.observe_append(messages)

        def observe_usage(self, *args, **kwargs):
            self._wrapped.observe_usage(*args, **kwargs)

        @property
        def threshold(self):
            raise RuntimeError("no window configured")

    degraded = _NoWall(inner)
    opts = RunOptions(model="primary", session_id="", tool_search_enabled=False,
                      preserve_thinking_iterations=6, chat_messages_handle=msgs)
    latch = _ReliefLatch()

    for iteration in (1, 2, 3):
        report = _relieve_context(msgs, options=opts, meter=degraded,
                                  reason="intra_turn", keep_recent=18,
                                  tool_count=18, iteration=iteration, latch=latch)
        assert report.get("latched") is not True, (iteration, report)
        assert report.get("unlatched") is True, report

    assert latch.passes == 0, "a degraded meter armed the latch on a level of 0"
    assert latch.rearm == 0, latch.rearm

    pass_lines = [r.getMessage() for r in caplog.records
                  if "loop: context relief (intra_turn)" in r.getMessage()]
    assert not any("next pass allowed at" in m for m in pass_lines), pass_lines
    warned = [r.getMessage() for r in caplog.records if "NOT latched" in r.getMessage()]
    assert len(warned) == 1, warned             # once per turn, like the refusal
    assert "no re-arm level" in warned[0], warned[0]


# ── the quiet-turn control the post-landing hourly grep is read against ─────────

def test_a_measured_turn_under_target_runs_no_rung_and_logs_nothing(caplog):
    """The control that makes the human's post-landing measurement readable at all.

    `grep -h "loop: context relief (intra_turn)" logs/server.err*` binned per hour is
    what decides whether the latch worked, so a turn sitting UNDER target must not
    contribute a line — otherwise that count is a sum of two populations and a drop
    proves nothing. It must not run a rung either: the per-iteration call site
    (`run_query`'s `reason="intra_turn"` call) is gated only on tool count, so under
    target the only thing
    between a quiet turn and a rewritten history is the ladder's own `_over()`.

    Pinned by CONTRAST, since a 0-line assertion has three false ways to happen. Same
    history, same measured meter arithmetic, one knob (the level): at `target` minus
    20,000 every check runs zero rungs and the log holds no relief line of either
    kind, and one tier up at `BAND` (over target) a pass runs rungs and logs exactly
    one line. `latch=None` throughout — this is the gate, not the latch.

    The meter must be MEASURED for the quiet half to mean anything: `_over()` returns
    True on an unmeasured meter (`_over()`'s `not meter.measured` branch), so an
    unmeasured turn below target
    does run its rungs. That is the ladder's standing design for "no reading, assume
    pressure", and it is why this drives the meter `meter_at` reports a real level to.
    """
    caplog.set_level("INFO", logger=LOOP_LOGGER)
    opts = RunOptions(model="primary", session_id="", tool_search_enabled=False,
                      preserve_thinking_iterations=6)

    for level, expect_lines in ((TARGET - 20_000, 0), (BAND, 1)):
        caplog.clear()
        msgs = seeded_history(18)
        report = _relieve_context(msgs, options=opts, meter=meter_at(level, msgs),
                                  reason="intra_turn", keep_recent=15,
                                  tool_count=18, iteration=3)
        lines = [r.getMessage() for r in caplog.records
                 if "loop: context relief (intra_turn)" in r.getMessage()
                 or "context relief latched" in r.getMessage()]
        assert len(lines) == expect_lines, (level, lines)
        assert report["freed_tokens"] > 0 if expect_lines else report["freed_tokens"] == 0, \
            (level, report)

# The test above is the CONTROL the post-landing hourly grep is read against
# (`grep -h "loop: context relief (intra_turn)" ~/lloyd/logs/server.err*` binned per
# hour — 2,474 firings / 8.78 h on `logs/server.err.1` as re-measured 2026-09-20).
# It is pinned by contrast rather than by a bare 0-line assertion, so a drop in that
# count cannot come from a broken logger or an unmeasured meter, and the per-iteration
# driver control in this file (`test_after_a_relief_pass_the_ladder_stays_closed_below_
# the_trigger`) turns no rung and logs nothing while latched. Both controls live in
# this file rather than only in the driver file because the gate's `met`-node rail
# reads a clause off a test node, and the clause's clause-text is about the log line.
