"""While a round runs, the pool holds everything else back.

An autocode round is the rarest and most expensive job in the pool: one
worktree, nine gate rungs, a landing that restarts the backend. It is also
the one most damaged by sharing the primary — it re-submits a 100-200k-token
context every iteration, so anything else claiming a slot evicts its prefix.
Measured on 2026-09-11: every autocode session showed cold re-prefills of
0.14-0.85M tokens, with 14 scheduled-task and 15 autotriage runs on the same
engine in one 21-hour window.

The KV gate beside this asks "is the engine full *now*". This asks "is the
one job worth protecting running at all".
"""

from __future__ import annotations

import pytest

from workers import pool as P


class _Src:
    def __init__(self, long_lived=False):
        self.LONG_LIVED = long_lived


REGISTRY = {
    "autocode": _Src(True),
    "autotriage": _Src(True),
    "deep-research": _Src(True),
    "arch-review": _Src(),
    "session-distill": _Src(),
    "youtube-digest": _Src(),
    "backlog-cluster": _Src(),
    "scheduled-task": _Src(),
}


@pytest.fixture
def pool(monkeypatch):
    p = P.WorkerPool.__new__(P.WorkerPool)
    p._in_flight = {}
    p._kv_gate = {"engaged": False, "engaged_since": None, "engagements": 0,
                  "kv_usage": None, "held_sources": []}
    p._round_hold = {"engaged": False, "engaged_since": None, "engagements": 0,
                     "held_sources": []}
    p._primary_hold = {"engaged": False, "engaged_since": None, "engagements": 0,
                       "held_sources": [], "answering": None, "last_detail": None,
                       "last_probe": 0.0, "probing": False}
    # The KV gate is a separate question; keep it out of these assertions.
    monkeypatch.setattr(P, "kv_gate_config", lambda: {"enabled": False})
    # And so is reachability: disabled here means the decider never probes the
    # engine, which is the other reason these tests do not need a fake HTTP.
    monkeypatch.setattr(P, "primary_hold_config", lambda: {"enabled": False})
    # The exempt list is pinned, not read from config.yaml: on 2026-09-15
    # `autotriage` joined it there and two of these tests went red on main.
    # No `exempt_bound` key, which since #2152 means the code default is in
    # force — so the KV reading is pinned too, one line below.
    monkeypatch.setattr(P, "round_hold_config",
                        lambda: {"enabled": True, "exempt": ["scheduled-task"]})
    # A test process samples no engine, so `_gate_reading` would answer from
    # whatever the surrounding suite left in `engine_pressure`'s deque. Pinning
    # it to None makes every assertion below mean what it says: with no reading
    # the bound fails open, and a test that wants pressure calls `_kv`.
    monkeypatch.setattr(P.WorkerPool, "_gate_reading",
                        staticmethod(lambda window: None))
    return p


def _run_round(pool, source="autocode", item_id=1):
    pool._in_flight[item_id] = {"source": source, "kind": "run"}


def test_nothing_is_held_when_no_round_is_running(pool):
    assert pool._round_hold_held(REGISTRY) == []
    assert pool.round_hold_status()["engaged"] is False


def test_a_running_round_holds_the_long_lived_sources(pool):
    _run_round(pool)
    held = pool._round_hold_held(REGISTRY)
    for name in ("autotriage", "deep-research", "arch-review",
                 "session-distill", "youtube-digest", "backlog-cluster"):
        assert name in held, name


def test_the_round_never_holds_itself(pool):
    _run_round(pool)
    assert "autocode" not in pool._round_hold_held(REGISTRY)


def test_scheduled_task_is_exempt(pool):
    """Short, time-sensitive, and a user-visible schedule slipping by an hour
    is its own failure.

    The exemption is unbounded only while the engine gives no reading: #2152
    armed `DEFAULT_ROUND_HOLD_EXEMPT_BOUND`, so above a 0.60 KV median the
    source is bound-held instead — see
    `test_no_key_arms_the_default_bound_above_the_reading`. The fixture pins
    `_gate_reading` to None, which is the fail-open case pinned here.
    """
    _run_round(pool)
    assert "scheduled-task" not in pool._round_hold_held(REGISTRY)


def test_the_exempt_list_is_configurable(pool, monkeypatch):
    monkeypatch.setattr(P, "round_hold_config",
                        lambda: {"enabled": True, "exempt": ["youtube-digest"]})
    _run_round(pool)
    held = pool._round_hold_held(REGISTRY)
    assert "youtube-digest" not in held
    assert "scheduled-task" in held      # no longer exempt under this config


def test_the_hold_releases_when_the_round_leaves_in_flight(pool):
    """No lease and no TTL: `_in_flight` is exact and popped on every exit
    path, including the timeout and exception branches. A lease would be a
    second source of truth about the same fact.
    """
    _run_round(pool)
    assert pool._round_hold_held(REGISTRY)
    assert pool.round_hold_status()["engaged"] is True

    pool._in_flight.clear()
    assert pool._round_hold_held(REGISTRY) == []
    assert pool.round_hold_status()["engaged"] is False


def test_another_source_in_flight_does_not_engage_it(pool):
    _run_round(pool, source="deep-research")
    assert pool._round_hold_held(REGISTRY) == []


def test_the_kill_switch_releases_an_engaged_hold(pool, monkeypatch):
    _run_round(pool)
    assert pool._round_hold_held(REGISTRY)
    monkeypatch.setattr(P, "round_hold_config", lambda: {"enabled": False})
    assert pool._round_hold_held(REGISTRY) == []
    assert pool.round_hold_status()["engaged"] is False


def test_engagements_counts_transitions_not_polls(pool):
    _run_round(pool)
    for _ in range(5):
        pool._round_hold_held(REGISTRY)
    assert pool._round_hold["engagements"] == 1
    pool._in_flight.clear()
    pool._round_hold_held(REGISTRY)
    _run_round(pool, item_id=2)
    pool._round_hold_held(REGISTRY)
    assert pool._round_hold["engagements"] == 2


async def test_claim_holds_is_the_union_of_all_three_gates(pool, monkeypatch):
    """One call site so a third gate has an obvious home, and so the claim
    query takes the union rather than whichever gate ran last.

    The third gate is engaged here too, because the 2026-09-23 incident is
    exactly the moment all three are read together and only one of them can
    answer: the KV gate had no /metrics reading to work from (the engine was not
    serving), the round hold had no round in flight, and the claims went out.
    """
    monkeypatch.setattr(P, "kv_gate_config",
                        lambda: {"enabled": True, "max_kv_usage": 0.1,
                                 "window_seconds": 60})
    monkeypatch.setattr(P.WorkerPool, "_gate_reading",
                        staticmethod(lambda window: 0.99))
    _run_round(pool)
    _hold(monkeypatch, _Engine(False))
    held = set(await pool._claim_holds(REGISTRY))
    # KV gate contributes the long-lived set...
    assert {"autotriage", "deep-research"} <= held
    # ...and the round hold contributes the short ones the KV gate ignores.
    assert "session-distill" in held
    assert "youtube-digest" in held
    # `scheduled-task` IS held here, and the round hold's default bound is what
    # holds it (#2152): the fixture's `round_hold` config has no `exempt_bound`
    # key, the reading is forced to 0.99, and 0.99 > 0.60. This is the seam the
    # default has to reach — `_claim_holds` is the single call site that unions
    # the three gates into the claim query's `NOT IN`, so a bound that only
    # `round_hold_status` could see would change nothing.
    assert "scheduled-task" in held
    # And it is the exemption this assertion used to pin, now the bound's lower
    # half: below 0.60 the source claims beside the round as before.
    _kv(monkeypatch, 0.30)
    assert "scheduled-task" not in set(await pool._claim_holds(REGISTRY))
    # `autocode` IS held here, by the KV gate as well as by reachability, and
    # that is correct and pre-existing: the gates decide what may be CLAIMED, and
    # starting a second round under KV pressure is exactly what should not
    # happen. One round at a time is `round_depth`'s job; this is belt and braces.
    assert "autocode" in held
    # What matters for each change is that it is not the only thing holding it:
    # take the round hold away and `session-distill` is still held by KV, and
    # take the KV gate away and `autocode` is still held by reachability.
    assert "autocode" not in pool._round_hold_held(REGISTRY)
    monkeypatch.setattr(P, "kv_gate_config", lambda: {"enabled": False})
    assert "session-distill" not in pool._kv_gate_held(REGISTRY)
    assert await pool._primary_hold_held(REGISTRY) == ["autocode"]


def test_status_reports_the_hold(pool):
    _run_round(pool)
    pool._round_hold_held(REGISTRY)
    st = pool.round_hold_status()
    assert st["engaged"] is True
    assert st["engaged_since"]
    assert "autotriage" in st["held_sources"]
    assert st["exempt"] == ["scheduled-task"]


def test_held_sources_is_a_copy_not_the_live_list(pool):
    _run_round(pool)
    pool._round_hold_held(REGISTRY)
    st = pool.round_hold_status()
    st["held_sources"].append("mutated")
    assert "mutated" not in pool._round_hold["held_sources"]


# ---------------------------------------------------------------------------
# #1101: an optional bound on the exemption
# ---------------------------------------------------------------------------

EXEMPT = ["scheduled-task", "autotriage"]


def _bound(monkeypatch, **bound):
    monkeypatch.setattr(P, "round_hold_config",
                        lambda: {"enabled": True, "exempt": EXEMPT,
                                 "exempt_bound": bound})


def _kv(monkeypatch, value):
    monkeypatch.setattr(P.WorkerPool, "_gate_reading",
                        staticmethod(lambda window: value))


@pytest.mark.parametrize("bound", [{}, {"sources": []}])
def test_an_empty_bound_in_config_never_holds_an_exempt_source(pool, monkeypatch,
                                                               bound):
    """Clause 2: disarming the bound is a config edit, never a code change.

    This is the assertion `test_no_bound_key_never_holds_an_exempt_source` made
    of an *absent* key until #2152 armed a default; absence now means the
    default, so "off" has to be said by the key being present and empty. The
    engine reads 0.99 and two exempt jobs are already in flight beside the
    round, so only a disarmed bound can keep these sources claimable.
    """
    monkeypatch.setattr(P, "round_hold_config",
                        lambda: {"enabled": True, "exempt": EXEMPT,
                                 "exempt_bound": bound})
    _kv(monkeypatch, 0.99)
    _run_round(pool)
    pool._in_flight[2] = {"source": "scheduled-task", "kind": "run"}
    pool._in_flight[3] = {"source": "autotriage", "kind": "run"}
    held = pool._round_hold_held(REGISTRY)
    assert not set(held) & set(EXEMPT)
    st = pool.round_hold_status()
    assert st["exempt_refused"] == {}
    assert st["bound_held"] == []
    if bound == {}:
        assert st["exempt_bound"] is None      # no bound named at all
    else:
        assert st["exempt_bound"] == {"sources": []}   # a bound covering nobody


def test_inflight_bound_holds_a_covered_exempt_source(pool, monkeypatch):
    """Clause 1, in-flight leg: one exempt co-tenant already beside the round
    and `max_exempt_inflight: 1` refuses the next, only for covered sources."""
    _bound(monkeypatch, sources=["scheduled-task"], max_exempt_inflight=1)
    _run_round(pool)
    assert "scheduled-task" not in pool._round_hold_held(REGISTRY)
    pool._in_flight[2] = {"source": "autotriage", "kind": "run"}
    held = pool._round_hold_held(REGISTRY)
    assert "scheduled-task" in held
    assert "autotriage" not in held        # exempt, but not covered by the bound
    assert "autocode" not in held          # the round never holds itself


def test_kv_bound_holds_on_the_gate_reading(pool, monkeypatch):
    """Clause 1, KV leg: the reading is `_gate_reading`'s median, and a missing
    reading fails open."""
    _bound(monkeypatch, kv_median_above=0.5)
    _run_round(pool)
    _kv(monkeypatch, 0.7)
    assert set(EXEMPT) <= set(pool._round_hold_held(REGISTRY))
    _kv(monkeypatch, None)
    assert not set(pool._round_hold_held(REGISTRY)) & set(EXEMPT)


def test_the_bound_flips_and_clears_without_touching_the_rest(pool, monkeypatch):
    """Clause 3: claimable → held → claimable as the KV crosses and clears; the
    non-exempt held set is the same throughout and the KV gate's own state is
    not written by the round hold's read of it."""
    _bound(monkeypatch, sources=["scheduled-task"], kv_median_above=0.5)
    _run_round(pool)
    kv_before = dict(pool._kv_gate)

    _kv(monkeypatch, 0.2)
    base = set(pool._round_hold_held(REGISTRY))
    assert "scheduled-task" not in base
    _kv(monkeypatch, 0.8)
    crossed = set(pool._round_hold_held(REGISTRY))
    assert crossed == base | {"scheduled-task"}
    _kv(monkeypatch, 0.3)
    cleared = set(pool._round_hold_held(REGISTRY))
    assert cleared == base

    assert pool._kv_gate == kv_before
    assert pool.kv_gate_status()["engaged"] is False
    assert pool.kv_gate_status()["held_sources"] == []


def test_refusals_are_counted_per_source(pool, monkeypatch):
    """Clause 4: `exempt_refused` counts each refusal (a claimable→held turn),
    per source, not every claim poll that repeats it."""
    _bound(monkeypatch, kv_median_above=0.5)
    _run_round(pool)
    _kv(monkeypatch, 0.9)
    for _ in range(5):
        pool._round_hold_held(REGISTRY)
    st = pool.round_hold_status()
    assert st["exempt_refused"] == {"autotriage": 1, "scheduled-task": 1}
    assert st["bound_held"] == ["autotriage", "scheduled-task"]
    assert st["exempt_bound"] == {"kv_median_above": 0.5}

    _kv(monkeypatch, 0.1)
    pool._round_hold_held(REGISTRY)
    _kv(monkeypatch, 0.9)
    pool._round_hold_held(REGISTRY)
    assert pool.round_hold_status()["exempt_refused"] == {
        "autotriage": 2, "scheduled-task": 2}
    # The round leaving clears what is held, and keeps the count.
    pool._in_flight.clear()
    assert pool._round_hold_held(REGISTRY) == []
    st = pool.round_hold_status()
    assert st["bound_held"] == []
    assert st["exempt_refused"]["scheduled-task"] == 2


def test_the_bound_is_idle_without_a_round(pool, monkeypatch):
    _bound(monkeypatch, max_exempt_inflight=0, kv_median_above=0.0)
    _kv(monkeypatch, 0.9)
    assert pool._round_hold_held(REGISTRY) == []
    assert pool.round_hold_status()["exempt_refused"] == {}


# ---------------------------------------------------------------------------
# #2152: the bound is armed by default — the key that armed it was in a file the
# loop that shipped it is not allowed to write, so it shipped inert
# ---------------------------------------------------------------------------

# `workers.round_hold.exempt` as config.yaml carries it on 2026-10-04. The
# default reaches only the first: the other three joined on 09-15, 09-16 and
# 09-18 for reasons of their own, and #1101 named only `scheduled-task`.
PRODUCTION_EXEMPT = ["scheduled-task", "autotriage", "autoresearch",
                     "automod-regression"]

# The same four as claimable sources, so "the default does not cover this one"
# is a thing that could have failed rather than a name absent from the registry.
FULL_EXEMPT_REGISTRY = dict(REGISTRY, autoresearch=_Src(),
                            **{"automod-regression": _Src()})


def _no_bound_key(monkeypatch, exempt=("scheduled-task",)):
    """Config as config.yaml has it: a `round_hold` block with no
    `exempt_bound` key. That absence is the state #2152 is about."""
    monkeypatch.setattr(P, "round_hold_config",
                        lambda: {"enabled": True, "exempt": list(exempt)})


def test_no_key_arms_the_default_bound_above_the_reading(pool, monkeypatch):
    """Clause 1: with no key and a round in flight, a KV median over 0.60 holds
    `scheduled-task`; at or under the reading it claims as before.

    Before #2152 this returned nothing held at 0.61 — the inert state the item
    is about — so the first assertion is the one that proves the default is
    live, and the 0.60 one proves the bound reads `above`, not `at least`.
    """
    _no_bound_key(monkeypatch)
    _run_round(pool)
    _kv(monkeypatch, 0.61)
    assert "scheduled-task" in pool._round_hold_held(REGISTRY)
    _kv(monkeypatch, 0.60)
    assert "scheduled-task" not in pool._round_hold_held(REGISTRY)
    _kv(monkeypatch, 0.61)
    assert "scheduled-task" in pool._round_hold_held(REGISTRY)
    _kv(monkeypatch, 0.20)
    assert "scheduled-task" not in pool._round_hold_held(REGISTRY)


def test_the_default_bound_values_are_the_engine_s_own_thresholds():
    """The default's numbers, pinned so that changing one is a decision someone
    made: 0.60 is `DEFAULT_KV_GATE_MAX` and `scripts/vllm_prefix_miss_window.py`'s
    `--gate` default — the reading that report already calls a round-losing
    engine — and the covered source is the single one #1101 named.
    """
    assert P.DEFAULT_ROUND_HOLD_EXEMPT_BOUND == {
        "sources": ("scheduled-task",), "kv_median_above": 0.60}
    assert P.DEFAULT_KV_GATE_MAX == 0.60


def test_the_default_covers_only_scheduled_task(pool, monkeypatch):
    """Clause 4: on config.yaml's four-source exempt list, a full engine takes
    back `scheduled-task` and nobody else.

    Binding all four would be a scope change no clause asked for, and it would
    starve `autoresearch` and `automod-regression`, exempted because a round is
    in flight nearly 100% of the time and a held source then never runs.
    """
    _no_bound_key(monkeypatch, PRODUCTION_EXEMPT)
    _run_round(pool)
    _kv(monkeypatch, 0.99)
    held = pool._round_hold_held(FULL_EXEMPT_REGISTRY)
    assert "scheduled-task" in held
    for name in ("autotriage", "autoresearch", "automod-regression"):
        assert name not in held, name
    assert pool.round_hold_status()["bound_held"] == ["scheduled-task"]


def test_a_missing_reading_holds_nothing_under_the_default(pool, monkeypatch):
    """Clause 4's fail-open half: no sample is not pressure.

    The rule the KV gate already runs on — a pressure signal that failed closed
    turns a stopped sampler into a stopped pool — now applies to a bound that
    nobody switched on, so it has to be read from status rather than assumed.
    """
    _no_bound_key(monkeypatch)
    _run_round(pool)
    _kv(monkeypatch, None)
    held = pool._round_hold_held(REGISTRY)
    assert "scheduled-task" not in held
    assert "arch-review" in held          # the membership hold is untouched
    st = pool.round_hold_status()
    assert st["bound_held"] == []
    assert st["exempt_refused"] == {}
    assert st["exempt_bound"] == {"sources": ("scheduled-task",),
                                  "kv_median_above": 0.60}


def test_an_explicit_bound_replaces_the_default(pool, monkeypatch):
    """Clause 3: a config value wins outright, in behaviour and in status.

    This bound covers `autotriage` only and at 0.90, so the default's numbers
    show in the difference: `scheduled-task` stays claimable at 0.95, where the
    default would have held it at 0.60, and `autotriage` is held, which the
    default never does.
    """
    _bound(monkeypatch, sources=["autotriage"], kv_median_above=0.9)
    _run_round(pool)
    _kv(monkeypatch, 0.95)
    held = pool._round_hold_held(REGISTRY)
    assert "scheduled-task" not in held
    assert "autotriage" in held
    st = pool.round_hold_status()
    assert st["exempt_bound"] == {"sources": ["autotriage"],
                                  "kv_median_above": 0.9}
    assert st["bound_held"] == ["autotriage"]


def test_the_default_bound_engages_only_for_a_round_or_landing(pool, monkeypatch):
    """Clause 5: the bound rides the hold's own engagement, which is what keeps
    it idle most of the day.

    Round leg: nothing in flight and no landing ⇒ nothing held and nothing
    counted, however full the engine reads. Landing leg: the hold outlives the
    last round's turn to cover the landing it was waiting behind, and the bound
    applies across that window too.
    """
    _no_bound_key(monkeypatch)
    _kv(monkeypatch, 0.99)
    monkeypatch.setattr(P.WorkerPool, "_landing_in_flight", lambda self: False)
    assert pool._round_hold_held(REGISTRY) == []
    assert pool.round_hold_status()["exempt_refused"] == {}

    monkeypatch.setattr(P.WorkerPool, "_landing_in_flight", lambda self: True)
    assert "scheduled-task" in pool._round_hold_held(REGISTRY)
    assert pool.round_hold_status()["bound_held"] == ["scheduled-task"]


def test_the_default_bound_counts_one_refusal_per_turn(pool, monkeypatch):
    """Clause 5's flip/clear half under the default: five polls of one held
    turn are one refusal, and held → claimable → held is two.

    `exempt_refused` is the effect counter the item reads after landing, so it
    has to count under the default exactly as it did under a config bound.
    """
    _no_bound_key(monkeypatch)
    _run_round(pool)
    _kv(monkeypatch, 0.95)
    for _ in range(5):
        pool._round_hold_held(REGISTRY)
    st = pool.round_hold_status()
    assert st["exempt_refused"] == {"scheduled-task": 1}
    assert st["bound_held"] == ["scheduled-task"]

    _kv(monkeypatch, 0.10)
    pool._round_hold_held(REGISTRY)
    _kv(monkeypatch, 0.95)
    pool._round_hold_held(REGISTRY)
    assert pool.round_hold_status()["exempt_refused"] == {"scheduled-task": 2}


# ---------------------------------------------------------------------------
# The third gate: is the primary engine answering at all
# ---------------------------------------------------------------------------

class _Engine:
    """The primary's `/health`, answering whatever the test says it said.

    Callable in the shape `primary_engine_answering` has — `(answering, detail)`
    — and it counts the asks, because the whole reason the gate may exist is
    that a claim does not wait on HTTP: an engine probed once per poll would be
    a gate that made the pool slower in order to make it safer.
    """

    def __init__(self, answering=True, detail="HTTP 200"):
        self.answering = answering
        self.detail = detail
        self.asks = 0
        self.timeouts = []

    def __call__(self, timeout=0.0):
        self.asks += 1
        self.timeouts.append(timeout)
        return (self.answering, self.detail)


def _hold(monkeypatch, engine, **cfg):
    """Turn the gate on against `engine`, with the values production runs on.

    Those values are `workers/pool.py`'s defaults, restated here rather than
    imported: `workers.primary_hold` is read over them only if an operator puts
    it in config.yaml, and it is not there, so pinning them is what proves the
    defaults are the operative numbers and not a config file nobody reads.
    """
    conf = {"enabled": True, "sources": ["autocode"], "probe_seconds": 10,
            "probe_timeout_s": 2.0}
    conf.update(cfg)
    monkeypatch.setattr(P, "primary_hold_config", lambda: conf)
    monkeypatch.setattr(P, "primary_engine_answering", engine)
    return engine


async def test_the_defaults_hold_the_round_with_no_config_block(pool, monkeypatch):
    """No `workers.primary_hold` in config.yaml means the code's defaults run.

    The absence is load-bearing, not cosmetic: the promotion gate denies this
    loop any write to config.yaml (#1430's landing reports adding the block as a
    human path), so a hold that needed a config line to switch it on would be a
    hold that is off — the incident it exists for would replay unchanged. The
    numbers are pinned here so that changing a default is a decision someone
    made, not a line that drifted.
    """
    monkeypatch.setattr(P, "primary_hold_config", lambda: {})
    monkeypatch.setattr(P, "primary_engine_answering", _Engine(False))
    assert await pool._primary_hold_held(REGISTRY) == ["autocode"]
    st = pool.primary_hold_status()
    assert st["enabled"] is True and st["sources"] == ["autocode"]
    assert st["probe_seconds"] == 10.0 and st["probe_timeout_s"] == 2.0


async def test_an_engine_that_is_not_answering_holds_the_round(pool, monkeypatch):
    """Clause 4's decision, on the real registry shape.

    The 2026-09-23 19:35 restart: the backend came back first and claimed
    #1220 three times in 65 s against a primary still loading its 95.37 GiB
    n-gram table. vLLM does not bind its port until the model is resident, so
    every claim was `All connection attempts failed` with `round_id: None`, and
    each was booked as an attempt.
    """
    engine = _hold(monkeypatch, _Engine(False, "no answer (connection refused or timed out)"))
    held = await pool._primary_hold_held(REGISTRY)
    assert held == ["autocode"]
    assert "autocode" in await pool._claim_holds(REGISTRY)
    st = pool.primary_hold_status()
    assert st["engaged"] is True
    assert st["held_sources"] == ["autocode"]
    assert st["last_detail"] == "no answer (connection refused or timed out)"
    assert engine.asks == 1


async def test_an_engine_answering_200_holds_nothing(pool, monkeypatch):
    engine = _hold(monkeypatch, _Engine(True))
    assert await pool._primary_hold_held(REGISTRY) == []
    assert pool.primary_hold_status()["engaged"] is False
    assert engine.asks == 1


async def test_the_gate_is_not_decided_on_every_poll(pool, monkeypatch):
    """Sixty claims a minute from six slots; the engine is asked once.

    The KV gate reads a background sampler for this reason and the round hold
    caches a marker read (`_LANDING_PROBE_SECONDS`). A gate that asked the engine
    directly would put an HTTP round trip in front of every claim — turning an
    outage into a slower outage, on the one event loop that also serves every
    request and streams every turn.
    """
    engine = _hold(monkeypatch, _Engine(False))
    for _ in range(60):
        assert await pool._primary_hold_held(REGISTRY) == ["autocode"]
    assert engine.asks == 1


async def test_the_answer_ages_out_and_the_engine_is_asked_again(pool, monkeypatch):
    """The hold cannot latch on a stale reading: a restarted engine comes back
    on its own, within `probe_seconds`, with nothing to clear by hand."""
    engine = _hold(monkeypatch, _Engine(False))
    assert await pool._primary_hold_held(REGISTRY) == ["autocode"]
    pool._primary_hold["last_probe"] -= 11.0
    assert await pool._primary_hold_held(REGISTRY) == ["autocode"]
    assert engine.asks == 2


async def test_the_hold_releases_when_the_engine_answers_again(pool, monkeypatch):
    """Two lines in the log for an hour of outage, and `engagements` counts the
    transitions, not the polls — the same contract as the two gates beside it."""
    engine = _hold(monkeypatch, _Engine(False))
    await pool._primary_hold_held(REGISTRY)
    assert pool.primary_hold_status()["engaged"] is True
    engine.answering, engine.detail = True, "HTTP 200"
    pool._primary_hold["last_probe"] -= 11.0
    assert await pool._primary_hold_held(REGISTRY) == []
    st = pool.primary_hold_status()
    assert st["engaged"] is False and st["engaged_since"] is None
    assert st["engagements"] == 1, "engaged once, not once per poll"


async def test_a_probe_that_cannot_run_is_not_an_engine_down(pool, monkeypatch):
    """Fails open: a broken check must not become a stalled pool.

    The rule the KV gate states for a missing reading — "a pressure signal that
    failed closed would turn an unreachable /metrics into a stopped worker pool"
    — applies with more force to the gate whose ON state is an unreachable
    engine: if the probe itself could fail closed, any bug in this file would
    stop the loop that fixes bugs.
    """
    def boom(timeout=0.0):
        raise OSError("no socket for you")
    _hold(monkeypatch, boom)
    assert await pool._primary_hold_held(REGISTRY) == []
    assert pool.primary_hold_status()["engaged"] is False


async def test_the_kill_switch_releases_an_engaged_primary_hold(pool, monkeypatch):
    _hold(monkeypatch, _Engine(False))
    await pool._primary_hold_held(REGISTRY)
    assert pool.primary_hold_status()["engaged"] is True
    monkeypatch.setattr(P, "primary_hold_config", lambda: {"enabled": False})
    assert await pool._primary_hold_held(REGISTRY) == []
    assert pool.primary_hold_status()["engaged"] is False


async def test_only_the_named_sources_are_held(pool, monkeypatch):
    """Short jobs still claim against a down engine: they lose a cheap retry,
    not an item's one attempt. The gate is about the job the outage cannot
    repair, which is what `LONG_LIVED` declares."""
    engine = _hold(monkeypatch, _Engine(False), sources=["autocode", "autotriage"])
    held = await pool._primary_hold_held(REGISTRY)
    assert held == ["autocode", "autotriage"]
    assert "scheduled-task" not in held and "youtube-digest" not in held
    # A source named in config but not long-lived in the registry is not held
    # either: `LONG_LIVED` is the declaration that decides, not the name list.
    engine.asks = 0
    monkeypatch.setattr(P, "primary_hold_config",
                        lambda: {"enabled": True, "sources": ["youtube-digest"],
                                 "probe_seconds": 10})
    pool._primary_hold["last_probe"] -= 11.0
    assert await pool._primary_hold_held(REGISTRY) == []


async def test_the_probe_asks_the_engine_and_not_the_backend(pool, monkeypatch):
    """The endpoint is the engine's, and that choice is the whole gate.

    The pool runs INSIDE `lloyd-backend`, so the backend's own `/health` is up
    exactly when the pool is polling — asking it would have answered 200 to every
    one of the 02:37 claims. This pins the URL the probe actually requests,
    through the real request helper, so a future edit that "simplifies" it to the
    local backend turns the gate into a no-op that always passes.
    """
    import scripts.automod.promote as PR
    asked = []

    class _R:
        status = 200

    def fake_get(url, timeout=5.0):
        asked.append(url)
        return 200, _R()

    monkeypatch.setattr(PR, "_get", fake_get)
    monkeypatch.setattr(P, "primary_hold_config",
                        lambda: {"enabled": True, "sources": ["autocode"]})
    assert await pool._primary_hold_held(REGISTRY) == []
    assert asked == ["http://127.0.0.1:8096/health"], asked
    assert "8080" not in asked[0], "the backend is up whenever the pool is polling"
    assert pool.primary_hold_status()["probe_timeout_s"] == 2.0


def test_the_http_layer_s_answer_shape_maps_to_a_hold(monkeypatch):
    """The mapping from what `_get` reports to what the gate decides.

    `promote._get` reports a refused connection and a timeout both as
    `(None, None)` — it swallows every exception — and that tuple is the
    load-bearing input here, because an engine still loading its weights is
    unreachable rather than answering badly. Only the HTTP call is faked, so the
    status-to-verdict mapping under test is the shipped one: `None` must not be
    read as "answered, and it looks fine", which is how a wedged listener would
    turn into 1,800 claims an hour against a primary that never came up.
    """
    import scripts.automod.promote as PR
    from workers.pool import primary_engine_answering

    monkeypatch.setattr(PR, "_get", lambda url, timeout=5.0: (None, None))
    assert primary_engine_answering()[0] is False, "no answer is not an answer"
    monkeypatch.setattr(PR, "_get", lambda url, timeout=5.0: (503, None))
    assert primary_engine_answering()[0] is False, "an answered non-200 is not ready"
    monkeypatch.setattr(PR, "_get", lambda url, timeout=5.0: (200, None))
    assert primary_engine_answering()[0] is True, "vLLM's bare 200 with no body is ready"


async def test_the_skipped_claim_stays_queued_with_its_attempt_intact(
        pool, monkeypatch, tmp_path):
    """The clause, across the one boundary this change crosses: gate → queue.

    `claim_next` is where a row's `attempts` rises and its state leaves `queued`,
    so skipping it must leave BOTH alone. `mark_failed` on a dead engine is the
    other thing this could have been made to do, and it would have written the
    same lie the ledger already carries for #1220 and #654: an item retired as
    spent for a stack that was down.
    """
    from workers.queue import WorkQueue
    import scripts.automod.state as ST
    q = WorkQueue(tmp_path / "workers.db")
    item_id = q.enqueue("autocode", "implement", {"item_id": 1220}, priority=10)
    engine = _hold(monkeypatch, _Engine(False, "no answer (connection refused or timed out)"))
    # The implement ledger, pointed somewhere that is not the live one: the claim
    # below is the thing under test, and `autocode.execute` writes its `started`
    # row on entry, so an empty file here is the assertion, not an assumption.
    monkeypatch.setattr(ST, "LEDGER_PATH", tmp_path / "ledger.jsonl")

    held = await pool._claim_holds(REGISTRY)
    assert "autocode" in held
    assert q.claim_next("worker-0", {}, held) is None, "no claim, so no `started` row"
    assert ST.read_events(path=ST.LEDGER_PATH) == [], "nothing recorded against the item"
    row = q.get(item_id)
    assert row.state == "queued" and int(row.attempts or 0) == 0, "offered again later"

    # And the same row is claimable the moment the engine answers — the item was
    # delayed, not consumed. Nothing is cleared or revived to get here: the row
    # was never taken, so there is nothing to put back.
    engine.answering, engine.detail = True, "HTTP 200"
    pool._primary_hold["last_probe"] -= 11.0
    claimed = q.claim_next("worker-0", {}, await pool._claim_holds(REGISTRY))
    assert claimed is not None and claimed.id == item_id
    assert int(q.get(item_id).attempts or 0) == 1, "the attempt is spent by RUNNING, once"


# ---------------------------------------------------------------------------
# wait_idle — one definition of "the engine is free"
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402

from app import vllm_metrics  # noqa: E402


class _FakeClient:
    """Serves a scripted sequence of `num_requests_running` values."""

    def __init__(self, running_sequence):
        self.seq = list(running_sequence)
        self.calls = 0

    async def get(self, url, timeout=None):
        self.calls += 1
        n = self.seq[min(self.calls - 1, len(self.seq) - 1)]

        class _R:
            text = f"vllm:num_requests_running {n}.0\n"
        return _R()

    async def aclose(self):
        pass


def test_a_momentary_zero_is_not_idle():
    """The whole reason this is not two lines at each call site. The gauge is
    sampled, requests arrive between samples, and a bench that starts on the
    first zero it sees is measuring a shared engine while believing it has
    one to itself.
    """
    client = _FakeClient([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    asyncio.run(vllm_metrics.wait_idle(
        "http://x", quiet_s=0.3, limit_s=5, poll_s=0.05, client=client))
    # It cannot have returned on the first sample: the second was busy.
    assert client.calls >= 5


def test_it_raises_rather_than_measuring_a_busy_engine():
    client = _FakeClient([3])
    with pytest.raises(TimeoutError) as exc:
        asyncio.run(vllm_metrics.wait_idle(
            "http://x", quiet_s=0.2, limit_s=0.6, poll_s=0.05, client=client))
    assert "never went idle" in str(exc.value)


def test_allow_running_lets_a_caller_ignore_its_own_request():
    """An eval measuring its own turn should wait for everyone *else* to
    leave, not for a number it is contributing to.
    """
    client = _FakeClient([1])
    asyncio.run(vllm_metrics.wait_idle(
        "http://x", quiet_s=0.2, limit_s=3, poll_s=0.05,
        allow_running=1, client=client))


def test_an_unreachable_engine_is_not_idle():
    """A scrape that fails reads as busy, not as quiet: 'I could not tell'
    must never green-light a measurement.
    """
    class _Dead:
        async def get(self, url, timeout=None):
            raise OSError("connection refused")

        async def aclose(self):
            pass

    with pytest.raises(TimeoutError):
        asyncio.run(vllm_metrics.wait_idle(
            "http://x", quiet_s=0.2, limit_s=0.5, poll_s=0.05, client=_Dead()))


def test_the_bench_script_uses_the_shared_definition():
    """Two private copies of 'is the engine busy' is how the bench and the
    pool would come to disagree about what idle means.
    """
    src = (P.__file__.rsplit("/workers/", 1)[0]
           + "/agent-services/bin/bench-admission-stall.py")
    text = open(src).read()
    assert "vllm_metrics.wait_idle(" in text
    # and no second loop of its own
    assert "quiet_since = quiet_since or time.monotonic()" not in text


# ── a landing is held for too ───────────────────────────────────────────────
#
# 2026-09-19: #608's landing waited 430 s for a sibling round's turn. The hold
# ended with that turn, `youtube-digest`, `board-steward` and `bench-mine`
# started at 17:00:23-25, the landing paused the pool at 17:00:48 — and spent
# fourteen minutes waiting out jobs that had started seconds before it.

def _landing(monkeypatch, rounds):
    from scripts.automod import state as S
    monkeypatch.setattr(S, "rounds_landing", lambda: list(rounds))


def test_a_running_landing_keeps_the_hold_after_the_last_round_ends(pool, monkeypatch):
    _landing(monkeypatch, ["SM_20260919_160709"])
    held = pool._round_hold_held(REGISTRY)
    assert "youtube-digest" in held and "scheduled-task" not in held
    assert pool.round_hold_status()["engaged"] is True


def test_the_hold_ends_with_the_landing(pool, monkeypatch):
    _landing(monkeypatch, ["SM_20260919_160709"])
    assert pool._round_hold_held(REGISTRY)
    _landing(monkeypatch, [])
    pool._landing_probe = (0.0, True)            # the cached answer has aged out
    assert pool._round_hold_held(REGISTRY) == []
    assert pool.round_hold_status()["engaged"] is False


def test_the_state_dir_is_not_read_on_every_claim(pool, monkeypatch):
    from scripts.automod import state as S
    calls = []
    monkeypatch.setattr(S, "rounds_landing", lambda: calls.append(1) or [])
    for _ in range(50):
        pool._round_hold_held(REGISTRY)
    assert len(calls) == 1


def test_an_unreadable_state_dir_is_not_a_landing(pool, monkeypatch):
    from scripts.automod import state as S

    def boom():
        raise OSError("no")
    monkeypatch.setattr(S, "rounds_landing", boom)
    assert pool._round_hold_held(REGISTRY) == []


def test_only_a_live_marker_counts(tmp_path, monkeypatch):
    from scripts.automod import state as S
    monkeypatch.setattr(S, "ROUNDS_DIR", tmp_path)
    monkeypatch.setattr(S, "pid_alive", lambda pid: int(pid or 0) == 777)
    S.write_land_marker("SM_LIVE", pid=777, by="round.land")
    S.write_land_marker("SM_DEAD", pid=778, by="round.land")
    (tmp_path / "SM_NONE").mkdir()
    assert S.rounds_landing() == ["SM_LIVE"]
