"""Prefix-cache miss accounting (`app/prefix_miss.py`).

The 09-09 stall was 100-200k prompts re-prefilled from cold mid-loop, and the
per-iteration numbers that show it (`input_tokens`, `cache_read`) were on disk
all along. What was missing was a count that means something, and three
traps decide whether it does — each pinned here:

* iterations 1-2 are never counted (the engine reports 0 cached for the
  first two requests of any prefix);
* a turn that reads zero everywhere is *unmeasured* — the pre-5531f21
  signature of nothing parsing `cached_tokens` — not a turn full of misses;
* only a miss counts toward `reprefill_tokens`, or a healthy long round's
  appended tail (5-7k an iteration) would sum past the alert line.

The real numbers below are from sessions that are on disk:
`20260910_003817_autocode_308f` iteration 44 (147,117 in / 0 cached) and 45
(149,739 / 76,800).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app import engine_pressure as ep
from app import prefix_miss as pm


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    ep.reset()
    monkeypatch.setattr(pm, "_last_announce_at", 0.0)
    monkeypatch.setattr(pm, "_cfg", lambda: {})

    async def _no_reading(*a, **k):
        return None

    # Hermetic: nothing here may reach the live engine or the real fan-out.
    monkeypatch.setattr(ep, "scrape_once", _no_reading)
    monkeypatch.setattr(pm, "_announce", lambda *a, **k: {})
    yield
    ep.reset()


def _u(inp: int, cached: int) -> dict:
    return {"input_tokens": inp, "cache_read": cached}


# ── the count ─────────────────────────────────────────────────────────


def test_iterations_one_and_two_never_count():
    t = pm.TurnMissTracker()
    assert t.observe(1, _u(200_000, 0)) == []
    assert t.observe(2, _u(200_000, 0)) == []
    assert t.counted == 0
    assert t.summary() == {"reprefill_tokens": 0, "prefix_misses": 0}


def test_a_warm_long_loop_reads_zero_even_with_its_appended_tail():
    """Sixty iterations each re-prefilling a 6k tail. Summing every tail —
    the plan's first formula — would read 348k here and fire the alert on a
    perfectly healthy round."""
    t = pm.TurnMissTracker()
    for i in range(1, 61):
        inp = 200_000 + 3_000 * i
        t.observe(i, _u(inp, inp - 6_000 if i > 2 else 0))
    assert t.summary() == {"reprefill_tokens": 0, "prefix_misses": 0}


def test_a_cold_readmission_mid_loop_is_one_miss_costing_its_uncached_tokens():
    t = pm.TurnMissTracker()
    t.observe(1, _u(150_000, 0))
    t.observe(2, _u(152_000, 0))
    t.observe(3, _u(154_000, 150_000))
    released = t.observe(4, _u(156_000, 0))
    assert [m.iteration for m in released] == [4]
    t.observe(5, _u(158_000, 153_600))
    assert t.summary() == {"reprefill_tokens": 156_000, "prefix_misses": 1}


def test_the_line_is_half_the_prompt_at_100k_or_more():
    t = pm.TurnMissTracker(cache_seen=True)
    # The real iteration 45: 51% reused — a partial, not a miss.
    assert t.observe(45, _u(149_739, 76_800)) == []
    # Too small to be the stall, however cold.
    assert t.observe(46, _u(99_000, 0)) == []
    # The real iteration 44: fully cold.
    assert [m.uncached for m in t.observe(44, _u(147_117, 0))] == [147_117]


def test_a_turn_that_reads_zero_everywhere_is_unmeasured_not_cold():
    """The pre-5531f21 signature: nothing parsed `cached_tokens`, and every
    iteration of every session read 0."""
    t = pm.TurnMissTracker()
    for i in range(1, 8):
        assert t.observe(i, _u(120_000 + i, 0)) == []
    assert t.measured is False
    assert t.summary() == {"reprefill_tokens": None, "prefix_misses": None}
    assert len(t.pending) == 5


def test_a_zero_before_the_first_nonzero_read_is_confirmed_by_it():
    t = pm.TurnMissTracker()
    t.observe(1, _u(150_000, 0))
    t.observe(2, _u(151_000, 0))
    assert t.observe(3, _u(152_000, 0)) == []              # miss, or the bug
    released = t.observe(4, _u(153_000, 150_000))           # the field is live
    assert [m.iteration for m in released] == [3]
    assert t.summary() == {"reprefill_tokens": 152_000, "prefix_misses": 1}


def test_usage_numbers_accept_both_spellings():
    assert pm.usage_numbers({"prompt_tokens": 5, "prompt_tokens_cached": 3}) == (5, 3)
    assert pm.usage_numbers({"input_tokens": 7, "cache_read": 2}) == (7, 2)
    assert pm.usage_numbers(None) == (0, 0)


def test_the_label_names_a_background_source_or_a_chat():
    assert pm.label_for_session("20260910_120001_autocode_9f2a") == "autocode"
    assert pm.label_for_session("20260910_120001_ab12cd") == "a chat turn"


def test_record_iteration_logs_each_confirmed_miss():
    events: list = []
    t = pm.TurnMissTracker(cache_seen=True)
    released = pm.record_iteration(t, 4, _u(180_000, 0),
                                   log=lambda n, d: events.append((n, d)))
    assert len(released) == 1
    name, data = events[0]
    assert name == "brain1.prefix_miss"
    assert data["uncached_tokens"] == 180_000
    assert data["turn_prefix_misses"] == 1


def test_finish_says_so_when_a_turn_could_not_be_measured():
    events: list = []
    t = pm.TurnMissTracker()
    for i in range(1, 5):
        t.observe(i, _u(130_000, 0))
    summary = pm.finish(t, log=lambda n, d: events.append(n))
    assert summary == {"reprefill_tokens": None, "prefix_misses": None}
    assert events == ["brain1.prefix_miss_unmeasured"]


def test_the_kill_switch_stops_the_accounting(monkeypatch):
    monkeypatch.setattr(pm, "_cfg", lambda: {"enabled": False})
    t = pm.TurnMissTracker(cache_seen=True)
    assert pm.record_iteration(t, 4, _u(180_000, 0)) == []


# ── the bell ──────────────────────────────────────────────────────────


def _missed_turn(uncached: int = 150_000):
    t = pm.TurnMissTracker(label="autocode", cache_seen=True)
    t.observe(5, _u(uncached, 0), duration_ms=15_000, now=1000.0)
    return t, t.confirmed[-1]


def _sample(at: float, running: int) -> None:
    ep.record(ep.Sample(at, at, 0.4, running, 0), window=10_000)


@pytest.fixture
def announced(monkeypatch):
    calls: list = []
    monkeypatch.setattr(pm, "_announce",
                        lambda title, body, voice: calls.append((title, body, voice))
                        or {"journal": True, "desktop": True})
    return calls


def test_a_big_miss_beside_a_neighbour_is_announced_once(announced):
    t, miss = _missed_turn()
    _sample(990.0, 2)                     # during the iteration: it + one other
    assert asyncio.run(pm.maybe_announce(t, miss)) is True
    assert asyncio.run(pm.maybe_announce(t, miss)) is False    # once per turn
    assert len(announced) == 1
    title, body, voice = announced[0]
    assert title == "Prefix-cache miss: autocode"
    assert "150k" in body and "1 other request," in body
    assert voice is False                 # off unless config asks


def test_a_cold_prefill_with_the_engine_otherwise_idle_is_not_news(announced):
    t, miss = _missed_turn()
    _sample(990.0, 1)                     # nobody but itself
    assert asyncio.run(pm.maybe_announce(t, miss)) is False
    assert announced == []


def test_at_the_threshold_is_not_over_it(announced):
    t, miss = _missed_turn(uncached=100_000)
    _sample(990.0, 4)
    assert asyncio.run(pm.maybe_announce(t, miss)) is False


def test_the_cooldown_spans_turns(announced, monkeypatch):
    monkeypatch.setattr(pm, "_cfg", lambda: {"announce_cooldown_seconds": 1800})
    t1, m1 = _missed_turn()
    t2, m2 = _missed_turn()
    _sample(990.0, 3)
    assert asyncio.run(pm.maybe_announce(t1, m1)) is True
    assert asyncio.run(pm.maybe_announce(t2, m2)) is False
    assert len(announced) == 1


def test_no_reading_at_all_means_no_announcement(announced):
    t, miss = _missed_turn()              # no samples, and scrape_once -> None
    assert asyncio.run(pm.maybe_announce(t, miss)) is False


def test_a_miss_no_sample_landed_in_asks_the_engine_after_the_fact(announced, monkeypatch):
    """The miss's own request has finished by then, so whatever is running
    is somebody else."""
    async def _one_running(*a, **k):
        return {"requests_running": 1, "kv_cache_usage": 0.3, "requests_waiting": 0}

    monkeypatch.setattr(ep, "scrape_once", _one_running)
    t, miss = _missed_turn()
    assert asyncio.run(pm.maybe_announce(t, miss)) is True


def test_voice_is_a_config_decision(announced, monkeypatch):
    monkeypatch.setattr(pm, "_cfg", lambda: {"announce_voice": True})
    t, miss = _missed_turn()
    _sample(990.0, 2)
    asyncio.run(pm.maybe_announce(t, miss))
    assert announced[0][2] is True


# ── through a real writer ─────────────────────────────────────────────


def test_the_recorder_counts_a_miss_and_puts_it_on_the_usage_row(announced):
    """End to end through `app/run_recorder.py`: the event log carries the
    miss, the usage row carries the count, the transcript's final stats carry
    both numbers. conftest points all three stores at scratch files."""
    import usage_store
    from app import event_log
    from app.run_recorder import record_events
    from app.sessions_io import SESSIONS_DIR, create_session

    sid = "20260910_120000_autocode_beef"
    create_session(sid, platform="worker", model="primary", title="t",
                   source="autocode")
    events = [
        {"type": "assistant_message", "text": "", "iteration": it,
         "tool_calls": [], "duration_ms": 100,
         "usage": {"input_tokens": inp, "output_tokens": 10, "cache_read": cached}}
        for it, inp, cached in [(1, 150_000, 0), (2, 152_000, 0),
                                (3, 154_000, 150_000), (4, 156_000, 0),
                                (5, 158_000, 153_600)]
    ] + [{"type": "result", "stop_reason": "stop", "num_turns": 5,
          "duration_ms": 900, "response_text": "done",
          "usage": {"input_tokens": 158_000, "output_tokens": 50,
                    "cache_read": 457_600}}]

    async def _drive():
        async def _feed():
            for evt in events:
                yield evt
        async for _ in record_events(_feed(), session_id=sid, turn_id="t1",
                                     model="primary", source="autocode"):
            pass

    asyncio.run(_drive())

    row = usage_store._conn().execute(
        "SELECT reprefill_tokens, prefix_misses FROM usage WHERE session_id=?",
        (sid,)).fetchone()
    assert tuple(row) == (156_000, 1)

    logged = "".join(p.read_text() for p in event_log.EVENT_LOGS_DIR.rglob("*")
                     if p.is_file())
    assert "brain1.prefix_miss" in logged

    data = json.loads((SESSIONS_DIR / f"{sid}.json").read_text())
    final = [m for m in data["messages"] if m.get("role") == "assistant"][-1]
    assert final["stats"]["reprefill_tokens"] == 156_000
    assert final["stats"]["prefix_misses"] == 1
