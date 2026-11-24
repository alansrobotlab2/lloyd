"""The research registry — the guarantees the pipeline rides on.

Every property here replaces something the retired markdown checklist could
not do, and several of them are regressions from the file it replaces:

* `key` is not truncated, because a 50-character slug already collides in the
  real corpus (see `test_a_long_topic_is_not_truncated_into_a_collision`).
* `propose` is one atomic statement, because two processes write this file and
  the read-then-write shape it replaces loses that race.
* Retries live here, not in the work queue, because the pool completes an
  in-band `failed` item and never retries it.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import re
import sqlite3
import subprocess
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app import research_store as R
from app.research_store import ResearchStore, StoreUnavailable, normalize_key, parse_legacy


@pytest.fixture
def s(tmp_path) -> ResearchStore:
    return ResearchStore(tmp_path / "research.db")


@pytest.fixture
def note(tmp_path):
    """Factory for a note that is really on disk.

    `finish` refuses a `written` whose artifact is not a real file of
    `MIN_NOTE_BYTES` (#1276), so any test that settles one now needs a file
    rather than a plausible string. Tests that assert the refusal itself pass
    the path size explicitly.
    """
    def make(name: str = "a-note.md", size: int = R.MIN_NOTE_BYTES) -> Path:
        path = tmp_path / "knowledge" / "research" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
        return path

    return make


# ---------------------------------------------------------------------------
# Schema and lifecycle of the store object
# ---------------------------------------------------------------------------


def test_schema_is_idempotent_and_wal(tmp_path):
    path = tmp_path / "research.db"
    first = ResearchStore(path)
    first.propose("a topic about vLLM prefix caching")
    reopened = ResearchStore(path)
    assert reopened.stats()["total"] == 1
    with reopened._connect() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_an_unreadable_registry_raises_rather_than_reading_empty(tmp_path):
    """An empty read here feeds a generator that would re-propose everything."""
    path = tmp_path / "research.db"
    path.write_bytes(b"this is not a database, just some bytes" * 4)
    with pytest.raises(StoreUnavailable):
        ResearchStore(path).stats()


def test_the_default_store_is_opened_lazily_and_can_be_pointed(tmp_path):
    """A module-level instance would create the live file at import time, which
    under the automod gate means a canary writing into production."""
    assert R._default is None, "importing the module must not open a store"
    configured = R.configure(tmp_path / "elsewhere.db")
    assert configured.path == tmp_path / "elsewhere.db"
    assert R.store() is configured
    R.reset()
    assert R._default is None


# ---------------------------------------------------------------------------
# propose / dedup / similar
# ---------------------------------------------------------------------------


def test_proposing_twice_is_one_topic(s):
    first = s.propose("Speculative decoding for local agent loops", domain="local-llm-serving")
    again = s.propose("speculative decoding for local agent loops!", domain="local-llm-serving")
    assert first["created"] is True
    assert again["created"] is False and again["id"] == first["id"]
    assert s.stats()["total"] == 1


def test_the_key_ignores_case_punctuation_and_legacy_suffixes(s):
    s.propose("MoE expert residency on 2x RTX 3090")
    for variant in (
        "moe expert residency on 2x rtx 3090",
        "MoE expert residency on 2x RTX 3090 — researched 2026-09-07",
        "MoE expert residency on 2x RTX 3090 → knowledge/research/x.md",
        "MoE  expert,residency on 2x RTX 3090.",
    ):
        assert s.propose(variant)["created"] is False, variant
    assert s.stats()["total"] == 1


def test_a_long_topic_is_not_truncated_into_a_collision(s):
    """The retired source keyed on a 50-char slug, and these two really are in
    the corpus: both cut to `multi-agent-task-decomposition-hierarchical-plannin`,
    so the second was silently dropped as a duplicate of the first."""
    a = "Multi-agent task decomposition — hierarchical planning for"
    b = "Multi-agent task decomposition — hierarchical planning for complex robotic workflows"
    assert normalize_key(a) != normalize_key(b)
    assert s.propose(a)["created"] is True
    assert s.propose(b)["created"] is True


def test_an_empty_topic_is_refused(s):
    with pytest.raises(ValueError):
        s.propose("   ")
    with pytest.raises(ValueError):
        s.propose("!!! ???")


def test_similar_finds_a_reworded_neighbour_and_ignores_a_stranger(s):
    s.propose("Real-time SLAM integration with VLA policies — mapping during manipulation")
    s.propose("Qwen3 tokenizer vocabulary and multilingual coverage")
    out = s.propose("Real-time SLAM integration with VLA policies during manipulation tasks")
    topics = [row["topic"] for row in out["similar"]]
    assert any("SLAM" in t for t in topics), topics
    assert not any("tokenizer" in t for t in topics), topics


def test_similar_reports_the_status_so_a_generator_can_act_on_it(s, note):
    first = s.propose("Diffusion policy priors for sim-to-real transfer")
    s.claim(first["id"], by="test")
    s.finish(first["id"], "written", artifact_path=str(note("diffusion.md")))
    out = s.propose("Diffusion policy priors for sim to real transfer of manipulation")
    assert out["similar"] and out["similar"][0]["status"] == "written"


def test_a_topic_is_never_similar_to_itself(s):
    out = s.propose("Speculative decoding kv cache tradeoffs")
    assert out["similar"] == []


def test_the_queue_has_a_depth_cap(s):
    for i in range(R.MAX_QUEUED):
        s.propose(f"distinct research topic number {i} about serving")
    refused = s.propose("one topic too many about robotics perception")
    assert refused["created"] is False and refused["id"] is None
    assert "already queued" in refused["refused"]
    # An existing topic can still be re-proposed at the cap — that is a read.
    assert s.propose("distinct research topic number 0 about serving")["created"] is False


# ---------------------------------------------------------------------------
# next / claim / release / finish
# ---------------------------------------------------------------------------


def test_next_returns_the_most_urgent_then_the_oldest(s):
    dull = s.propose("a low priority topic", priority=70)
    urgent = s.propose("an urgent topic", priority=10)
    middle = s.propose("a middling topic", priority=40)
    assert [t["id"] for t in s.next(3)] == [urgent["id"], middle["id"], dull["id"]]


def test_a_claim_is_exclusive(s):
    topic = s.propose("a topic two workers both want")["id"]
    assert s.claim(topic, by="worker-a", queue_id=1) is not None
    assert s.claim(topic, by="worker-b", queue_id=2) is None, "claimed twice"
    assert s.get(topic)["status"] == "researching"
    assert s.get(topic)["attempts"] == 1


def test_a_claimed_topic_is_not_offered_again(s):
    topic = s.propose("a topic")["id"]
    s.claim(topic, by="worker", queue_id=1)
    assert s.next(5) == []


def test_the_same_queue_item_may_reclaim_its_own_topic(s):
    """The pool can call `execute` twice for one item after a raised failure,
    and the topic is legitimately still `researching` from the first attempt."""
    topic = s.propose("a topic whose first turn raised")["id"]
    s.claim(topic, by="worker", queue_id=7)
    assert s.claim(topic, by="worker", queue_id=7) is not None
    assert s.claim(topic, by="other", queue_id=8) is None
    assert s.get(topic)["attempts"] == 2


def test_release_holds_the_topic_back_for_its_backoff(s):
    topic = s.propose("a topic whose turn came back empty")["id"]
    s.claim(topic, by="worker", queue_id=1)
    s.release(topic, error="empty response", backoff_seconds=3600)

    row = s.get(topic)
    assert row["status"] == "queued" and row["queue_id"] is None
    assert row["attempts"] == 1, "a release must not forget the attempt"
    assert row["last_error"] == "empty response"
    assert s.next(5) == [], "the backoff did not hold it back"


def test_a_released_topic_returns_once_its_backoff_expires(s):
    topic = s.propose("a topic")["id"]
    s.claim(topic, by="worker", queue_id=1)
    s.release(topic, error="transient", backoff_seconds=3600)
    past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    with s._connect() as conn:
        conn.execute("UPDATE topics SET not_before=? WHERE id=?", (past, topic))
    assert [t["id"] for t in s.next(5)] == [topic]


def test_finishing_records_the_artifact_and_the_session(s, note):
    topic = s.propose("a topic that got written up")["id"]
    s.claim(topic, by="worker", queue_id=4)
    path = note("a.md")
    row = s.finish(topic, "written", artifact_path=str(path),
                   note="facts=7 sources=4", session_id="20260908_x")
    assert row["status"] == "written"
    assert row["artifact_path"] == str(path)
    assert row["session_id"] == "20260908_x" and row["finished_at"]
    assert row["queue_id"] == 4, "the link back to the workers.db run is kept"


def test_a_settled_outcome_is_not_rewritten(s, note):
    topic = s.propose("a settled topic")["id"]
    s.claim(topic, by="worker")
    s.finish(topic, "nothing_found", note="three searches, nothing")
    with pytest.raises(ValueError, match="already"):
        # A real note, so this reaches the settled-row guard and not the disk
        # gate in front of it — the two refusals say different things.
        s.finish(topic, "written", artifact_path=str(note("y.md")))


def test_only_a_terminal_status_may_finish(s):
    topic = s.propose("a topic")["id"]
    s.claim(topic, by="worker")
    with pytest.raises(ValueError, match="terminal"):
        s.finish(topic, "researching")


def test_exhaust_says_why_it_gave_up(s):
    topic = s.propose("a topic that never worked")["id"]
    s.claim(topic, by="worker")
    row = s.exhaust(topic, "abandoned after 2 failed attempts: empty response")
    assert row["status"] == "nothing_found"
    assert row["extra"]["reason"] == "exhausted"
    assert "abandoned" in row["outcome_note"]


def test_a_crashed_turn_is_reclaimed(s):
    topic = s.propose("a topic whose backend was killed mid-turn")["id"]
    s.claim(topic, by="worker", queue_id=1)
    assert s.reclaim_stale(3600) == 0, "a fresh claim is not stale"
    old = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat()
    with s._connect() as conn:
        conn.execute("UPDATE topics SET started_at=? WHERE id=?", (old, topic))
    assert s.reclaim_stale(3600) == 1
    assert s.get(topic)["status"] == "queued"


def test_every_transition_leaves_a_trace(s, note):
    topic = s.propose("a topic", proposed_by="65", signal="health-report")["id"]
    s.claim(topic, by="worker", queue_id=1)
    s.release(topic, error="nope", backoff_seconds=1)
    s.claim(topic, by="worker", queue_id=2)
    s.finish(topic, "written", artifact_path=str(note("a.md")))
    with s._connect() as conn:
        events = [r["event"] for r in conn.execute(
            "SELECT event FROM events WHERE topic_id=? ORDER BY id", (topic,)).fetchall()]
    assert events == ["proposed", "claimed", "released", "claimed", "written"]


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def test_listing_filters_and_recent_excludes_the_legacy_import(s, note):
    live = s.propose("a live topic")["id"]
    s.claim(live, by="w")
    s.finish(live, "written", artifact_path=str(note("a.md")))
    with s._connect() as conn:  # stand in for an imported row
        conn.execute(
            "INSERT INTO topics (topic, key, status, proposed_at, finished_at) "
            "VALUES ('legacy', 'legacy', 'archived', ?, ?)",
            (datetime.now(timezone.utc).isoformat(),
             datetime.now(timezone.utc).isoformat()))

    assert [t["topic"] for t in s.list(status="written")] == ["a live topic"]
    assert [t["topic"] for t in s.recent(days=30)] == ["a live topic"], \
        "archived legacy rows are a dedup corpus, not recent work"


def test_stats_counts_todays_real_work_only(s):
    live = s.propose("researched today")["id"]
    s.claim(live, by="w")
    s.finish(live, "nothing_found", note="nothing out there")
    s.propose("still waiting")
    with s._connect() as conn:
        conn.execute(
            "INSERT INTO topics (topic, key, status, proposed_at, finished_at) "
            "VALUES ('legacy', 'legacy', 'archived', ?, ?)",
            (datetime.now(timezone.utc).isoformat(),
             datetime.now(timezone.utc).isoformat()))

    stats = s.stats()
    assert stats["done_today"] == 1, "an archived import must not spend the daily budget"
    assert stats["queued"] == 1
    assert stats["by_status"]["archived"] == 1


def test_stale_queued_topics_are_reported_not_deleted(s):
    old = s.propose("a topic nobody got to")["id"]
    ancient = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
    with s._connect() as conn:
        conn.execute("UPDATE topics SET proposed_at=? WHERE id=?", (ancient, old))
    assert s.stats()["stale_queued"] == 1
    assert s.get(old)["status"] == "queued"


# ---------------------------------------------------------------------------
# Saturation is in the payload (#1277)
# ---------------------------------------------------------------------------


def test_stats_names_the_cap_it_is_measured_against(s):
    """`queued: 40` and `queued: 12` were the same shape to a reader that did
    not already know `MAX_QUEUED` out of band; the generator learned the cap
    by being refused at proposal #4."""
    for i in range(12):
        s.propose(f"distinct research topic number {i} about serving")
    stats = s.stats()
    assert stats["max_queued"] == R.MAX_QUEUED
    assert stats["queued_fraction"] == round(12 / R.MAX_QUEUED, 3)
    assert stats["at_cap"] is False

    for i in range(12, R.MAX_QUEUED):
        s.propose(f"distinct research topic number {i} about serving")
    stats = s.stats()
    assert stats["queued"] == R.MAX_QUEUED
    assert stats["queued_fraction"] == 1.0
    assert stats["at_cap"] is True, "a full queue must say so before propose refuses"


def test_stats_reports_the_heads_wait_well_inside_the_stale_gate(s):
    """An 11-day-old head is inside neither gate this store has. `stale_queued`'s
    60-day bar reports 0 for it, and #4274's `STARVED_AFTER_DAYS` of 14 has not
    been crossed either, so the claim order still owes it nothing and its depth
    is 1 — the FIFO answer, not a promoted one. Both waits are reported beside
    each other precisely so a reader is never left guessing which gate an age
    has crossed: at filing the live head was 50 days old and read
    `stale_queued: 0` all the same, which is the disagreement #4274 is about."""
    assert s.stats()["head_age_days"] is None, "an empty queue has no head"
    head = s.propose("the topic at the head of the queue")["id"]
    s.propose("a newer topic behind it")
    eleven_days = (datetime.now(timezone.utc) - timedelta(days=11)).isoformat()
    with s._connect() as conn:
        conn.execute("UPDATE topics SET proposed_at=? WHERE id=?", (eleven_days, head))
    stats = s.stats()
    assert 10.9 <= stats["head_age_days"] <= 11.1
    assert stats["stale_queued"] == 0
    assert stats["oldest_queued_at"] == eleven_days
    assert stats["head_claim_position"] == 1, (
        "both rows are in the default band, so the older one is genuinely first "
        "and an 11-day wait buys nothing on its own")


# ---------------------------------------------------------------------------
# The drain side: how deep the head sits, and what got claimed ahead of it
# (#4274)
# ---------------------------------------------------------------------------

#: `config.yaml` `sources.deep-research.daily_max`, measured 2026-11-19, and the
#: divisor in the arithmetic behind `STARVED_AFTER_DAYS`. Pinned against the file
#: by `test_the_drain_the_gate_divides_by_is_the_shipped_one` rather than trusted.
DAILY_MAX = 3


def _backdate(s: ResearchStore, topic_id: int, *, days: float) -> str:
    """Stamp a queued topic as proposed `days` ago, and return the stamp.

    `propose` always stamps `proposed_at` with `now`, and both gates read exactly
    that column, so any test about a wait has to move it by hand — the same shape
    `test_stale_queued_topics_are_reported_not_deleted` uses for the 60-day gate.
    """
    stamp = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with s._connect() as conn:
        conn.execute("UPDATE topics SET proposed_at=? WHERE id=?", (stamp, topic_id))
    return stamp


def test_stats_reports_the_heads_depth_in_the_order_next_claims_in(s):
    """#4274 clause 1: `head_claim_position` is the oldest queued topic's 1-based
    place in the order `next()` uses, read out of the same SQL `next()` runs, so
    it cannot describe an ordering the store does not have.

    Eight is the shipped shape. One generator run (`proposed_by:
    20261118_162833_autonomy_3e7a`) proposed seven prio-50 topics at 00:34-00:37
    UTC on 2026-11-19 against a `daily_max` of 3, and the prio-55 row that had
    been waiting since 2026-09-30 stood 8th in the claim order — behind seven
    rows nearly fifty days younger. Ten days old here keeps the head inside
    `STARVED_AFTER_DAYS`, so the node pins the depth the payload could not
    express rather than the fix grading its own promotion.
    """
    head = s.propose("the topic that has waited longest", priority=55)["id"]
    for i in range(7):
        s.propose(f"a topic the generator proposed this morning number {i}",
                  priority=50)
    _backdate(s, head, days=10)

    stats = s.stats()
    assert stats["head_claim_position"] == 8, (
        "seven fresher topics in the default band stand in front of the head")
    assert [t["id"] for t in s.next(8)][-1] == head, (
        "a reported depth must be a depth in the order next() hands out")
    assert stats["head_passed_over"] == 0, (
        "nothing has been researched at all here, so the head has been "
        "outranked but not yet passed over")


def test_the_head_is_first_when_nothing_stands_in_front_of_it(s):
    """#4274 clause 1, the healthy edge: 1 when the oldest row is the one that
    would be claimed next, and no position at all for an empty queue — never 0,
    which would read as "zero rows ahead" and as "not measured" at once."""
    assert s.stats()["head_claim_position"] is None, "an empty queue has no head"
    assert s.stats()["head_passed_over"] == 0

    head = s.propose("urgent and already first in line", priority=10)["id"]
    s.propose("a newer topic in a slower band", priority=70)
    assert s.stats()["head_claim_position"] == 1
    assert s.next(1)[0]["id"] == head, "position 1 must be the row next() offers"
    assert s.stats()["head_passed_over"] == 0, "nothing has been researched yet"


def test_a_head_held_back_by_a_backoff_has_no_claim_position(s):
    """The other and only other cause of a null position, pinned so `null` never
    has to be guessed at: the oldest row is sitting on a `not_before` the claim
    order will not read today (a `release` backoff — seconds to minutes), so it is
    in no order `next()` can use and "how deep is it" has no answer."""
    head = s.propose("a topic released with an hour of backoff")["id"]
    s.claim(head, by="w")
    s.release(head, backoff_seconds=3600)
    s.propose("a newer topic in the same band", priority=50)

    stats = s.stats()
    assert stats["oldest_queued_at"] is not None and stats["queued"] == 2
    assert stats["head_claim_position"] is None, (
        "the row named by `oldest_queued_at` cannot be claimed today")


def test_stats_counts_the_topics_researched_ahead_of_the_head(s):
    """#4274 clause 2: `head_passed_over` is the number of topics that reached a
    PRODUCTIVE status having been proposed AFTER the row `oldest_queued_at` names.
    Derived from registry rows only — no clock arithmetic and no counter kept
    beside the table — because the claim to be evidenced is "this head has been
    passed over", and the only thing that evidence is, is rows that were behind
    it and are now settled.

    Two exclusions the number has to get right or it inflates into noise: a
    `rejected` settle is not a topic that was researched (PRODUCTIVE is
    `written` and `nothing_found`), and a productive topic OLDER than the head is
    not something the head was passed over by.
    """
    before = s.propose("researched before the head was even proposed")["id"]
    _backdate(s, before, days=40)
    s.claim(before, by="w")
    s.finish(before, "nothing_found", note="searched, nothing there")
    head = s.propose("the head of the queue", priority=55)["id"]
    twenty_days = _backdate(s, head, days=20)
    for i in range(3):
        later = s.propose(f"proposed after the head number {i}", priority=50)["id"]
        s.claim(later, by="w")
        s.finish(later, "nothing_found", note="searched, nothing there")
    refused = s.propose("proposed after the head, refused as out of scope")["id"]
    s.claim(refused, by="w")
    s.finish(refused, "rejected", note="out of scope")

    stats = s.stats()
    assert stats["queued"] == 1 and stats["oldest_queued_at"] == twenty_days
    assert stats["head_passed_over"] == 3, (
        "three topics proposed after the head were researched; the `rejected` "
        "settle is not productive and the 40-day row predates the head")
    assert stats["head_age_days"] > 19.9 and stats["stale_queued"] == 0, (
        "20 days is past the starvation gate and short of the stale gate: the "
        "two numbers describe different questions")


def test_a_starved_topic_is_claimed_ahead_of_every_fresher_one(s):
    """#4274 clause 3: at `STARVED_AFTER_DAYS` a queued topic is offered ahead of
    any topic that has waited less, whatever the stored `priority`. This is the
    shipped registry in miniature — a prio-55 row 50 days old, and prio-10/40/50
    rows proposed today that the old `ORDER BY priority ASC` put in front of it."""
    aged = s.propose("the fifty-day head", priority=55)["id"]
    _backdate(s, aged, days=50)
    fresh = {prio: s.propose(f"a fresh topic in band {prio}", priority=prio)["id"]
             for prio in (10, 40, 50)}

    assert [t["id"] for t in s.next(4)] == [aged, fresh[10], fresh[40], fresh[50]], (
        "the starved row goes first, and the fresh rows keep their band order")


def test_the_starved_head_is_first_and_the_payload_says_so(s):
    """#4274's acceptance in one registry: the same cohort that reported
    `head_age_days: 50.9` beside `stale_queued: 0` while the head sat 5th-8th in
    the claim order now reports the head first, and the field that used to be
    absent carries the 1 that proves it.

    `stale_queued` deliberately stays 0 here. It answers a different question —
    is this topic too old to research at all, which is a human's call — and
    `test_stale_queued_topics_are_reported_not_deleted` is its falsifier, so the
    drain fix re-purposes nothing from it.
    """
    head = s.propose("the topic at the head of the line", priority=55)["id"]
    for i in range(7):
        s.propose(f"a topic proposed this morning number {i}", priority=50)
    _backdate(s, head, days=50)

    stats = s.stats()
    assert stats["head_claim_position"] == 1, (
        "the head is what next() offers, so its depth is 1")
    assert s.next(1)[0]["id"] == head
    assert stats["head_age_days"] > 49.9
    assert stats["stale_queued"] == 0


def test_a_topic_one_day_short_of_the_gate_still_waits_behind_its_band(s):
    """The gate is a gate: a 13-day-old prio-55 row has not yet outlived one full
    pass over the cap, so the fresh prio-50 row in front of it stays in front and
    the payload says the head is 2nd rather than promoting it early."""
    head = s.propose("thirteen days old", priority=55)["id"]
    _backdate(s, head, days=13)
    fresh = s.propose("proposed today", priority=50)["id"]

    assert [t["id"] for t in s.next(2)] == [fresh, head]
    assert s.stats()["head_claim_position"] == 2


def test_the_gate_moves_only_the_starved_rows_and_leaves_band_order_intact(s):
    """#4274 clause 4: promotion must not collapse the bands it is meant to
    jump. Because the credit is one constant subtracted from every starved row
    and nothing at all below the gate, three properties hold at once, and this
    node is the one that would reddened by a ramp or by a FIFO-within-the-cohort
    implementation:

    * the two starved rows keep their band order even though the slower band is
      the older of the two (a rule that served the cohort FIFO would put
      `aged_dull` first);
    * the fresh rows come back in band order, 30 before 50 before 70;
    * the two fresh prio-50 rows come back in the order they were proposed.
    """
    aged_dull = s.propose("starved and in a slow band", priority=70)["id"]
    aged_urgent = s.propose("starved and in an urgent band", priority=30)["id"]
    _backdate(s, aged_dull, days=30)
    _backdate(s, aged_urgent, days=20)
    fresh_dull = s.propose("fresh and in a slow band", priority=70)["id"]
    fresh_urgent = s.propose("fresh and in an urgent band", priority=30)["id"]
    first_in_band = s.propose("the first fresh row in the default band",
                              priority=50)["id"]
    second_in_band = s.propose("the second fresh row in the default band",
                               priority=50)["id"]

    assert [t["id"] for t in s.next(6)] == [
        aged_urgent, aged_dull, fresh_urgent, first_in_band, second_in_band,
        fresh_dull]


def test_the_starvation_gate_is_one_full_pass_over_the_cap(s):
    """#4274 clause 5, the arithmetic half: the gate and the credit are
    module-level constants, and the gate is `MAX_QUEUED` divided by the drain —
    40 topics at 3 a day is 13.3 days, so 14 is the first whole day at which a
    still-queued row has waited longer than a full pass over a full registry.
    Move the cap or the drain and one of these three assertions says so."""
    assert R.STARVED_AFTER_DAYS == 14
    assert R.STARVED_AFTER_DAYS * DAILY_MAX >= R.MAX_QUEUED, (
        "the gate must be at least one full pass over the cap")
    assert (R.STARVED_AFTER_DAYS - 1) * DAILY_MAX < R.MAX_QUEUED, (
        "and the first whole day at which it is one: 13 days x 3 is 39, short "
        "of a 40-topic queue")


def test_the_drain_the_gate_divides_by_is_the_shipped_one():
    """`DAILY_MAX` above is a number this file asserts, so it is checked against
    the config the source actually reads: `sources.deep-research.daily_max` in
    `config.yaml`. If someone moves the drain, the gate's arithmetic is stale and
    this says which of the two has to move."""
    config = (Path(__file__).resolve().parents[1] / "config.yaml").read_text()
    block = config.split("\n    deep-research:", 1)
    assert len(block) == 2, "config.yaml no longer carries a deep-research source"
    drain = re.search(r"^\s*daily_max:\s*(\d+)", block[1], re.M)
    assert drain, "the deep-research source block names no daily_max"
    assert int(drain.group(1)) == DAILY_MAX, (
        f"config.yaml drains {drain.group(1)} a day; re-derive "
        f"STARVED_AFTER_DAYS from {R.MAX_QUEUED}/{drain.group(1)}")


def test_the_wait_the_schema_quotes_and_the_order_that_grants_it_are_untouched():
    """#4401 clause 4: the round that reconciled `research_propose`'s `priority`
    description changed prose in `agent_mcp/research.py` and nothing else, and it had
    to. The sentence it shipped quotes three numbers this module owns — the 14-day
    wait, the 1000-point credit that lets the wait outrank any band, and the default
    band of 50 callers are told about — so a change billed as text-only that moved one
    of them would make the new prose false in the opposite direction. Until now the
    credit and the signature default were pinned nowhere:
    `test_the_starvation_gate_is_one_full_pass_over_the_cap` reads the gate's
    arithmetic against the cap and the drain, and no node read these together.

    The `ORDER BY` is pinned as text as well as by the nodes above because clause 4
    names it, and shape and meaning fail differently here. Those behavioural nodes are
    the falsifier if the expression's MEANING moves; this is the falsifier if its SHAPE
    moves — the credit subtracted outside the `CASE`, say, which this module's own
    docstring says would make an unparseable `proposed_at` yield NULL, and NULL sorts
    FIRST ascending, handing a row nobody can read the head of the queue.
    """
    import inspect

    assert R.STARVED_AFTER_DAYS == 14
    assert R.STARVED_PRIORITY_CREDIT == 1000

    sql, params = R._claim_query(R._now())
    assert "ORDER BY priority - CASE WHEN" in sql, (
        f"the credit is no longer inside the CASE, which is what keeps an "
        f"unparseable stamp from buying the head of the queue: {sql}")
    assert "ELSE 0 END ASC, proposed_at ASC, id ASC" in sql, (
        f"the FIFO and tie-break terms that make 'position N' a fact are gone: {sql}")
    assert params[2:] == [R.STARVED_AFTER_DAYS * 86400, R.STARVED_PRIORITY_CREDIT], (
        "the ORDER BY no longer binds the constants its docstring says it uses")

    default = inspect.signature(ResearchStore.propose).parameters["priority"].default
    assert default == 50, (
        "`propose`'s default moved: agent_mcp/research.py::research_propose advertises "
        "it to every caller, and the credit's justification above cites it as the "
        "reason 1000 is large enough")


def test_promotion_below_the_gate_is_a_step_not_a_gradient(s):
    """#4274 clause 5, the bounded half. `workers/queue.py` ages per period; this
    store does not, and the difference is what clause 4 depends on: a row under
    the gate has no credit at all, so no amount of waiting below 14 days can
    smear two distinct bands. A 13-day prio-70 row and a one-day prio-10 row
    therefore come back in band order, oldest-credit notwithstanding."""
    slow = s.propose("thirteen days in a slow band", priority=70)["id"]
    urgent = s.propose("one day old and urgent", priority=10)["id"]
    _backdate(s, slow, days=13)

    assert [t["id"] for t in s.next(2)] == [urgent, slow], (
        "age buys nothing below the gate, so bands cannot collapse among fresh "
        "rows the way a per-period ramp would collapse this registry's 50-vs-55")


def test_an_unreadable_proposed_at_stamp_buys_no_promotion(s):
    """The NULL arm of the credit, written down because the other shape is the
    one that hands the queue to a corrupt row. Inside a `CASE` an unparseable
    stamp falls to `ELSE 0`, so a bad age buys nothing; subtracted directly it
    makes the whole sort key NULL, and NULL sorts FIRST ascending, which would
    put an unreadable row at the head of every claim."""
    broken = s.propose("a row whose stamp cannot be parsed")["id"]
    with s._connect() as conn:
        conn.execute("UPDATE topics SET proposed_at='not-a-timestamp' WHERE id=?",
                     (broken,))
    fresh = s.propose("a fresh urgent topic", priority=10)["id"]

    assert [t["id"] for t in s.next(2)] == [fresh, broken]


# ---------------------------------------------------------------------------
# The dedup corpus is named at zero (#3085)
# ---------------------------------------------------------------------------


def test_stats_reports_the_dedup_corpus_even_when_it_is_empty(s):
    """`by_status` is a bare `GROUP BY`, so a status with no rows emits no key
    at all. That is how the 314-row `archived` dedup corpus going missing read
    for four weeks: not as `archived: 0`, but as a payload that never mentioned
    the corpus (#3046, fixed as a reporting defect by #3085). The corpus is the
    one status whose absence IS the alarm, so it is the one key the payload
    carries unconditionally.

    Only that one key is injected: `test_the_import_archives_everything_and_is_
    idempotent` compares `by_status` by exact equality, and a full zero-skeleton
    for every status would break it.
    """
    stats = s.stats()
    assert stats["by_status"].get("archived", "MISSING") == 0, (
        "an empty dedup corpus must read as 0, not as an absent key")
    assert stats["total"] == 0, "a zero-valued key must not invent a row"

    s.propose("a live topic")
    stats = s.stats()
    assert stats["by_status"]["archived"] == 0, "still named once live rows exist"
    assert stats["by_status"]["queued"] == 1


def test_the_dedup_corpus_count_is_the_real_one_once_rows_land(s):
    """The key is a count, not a flag: once rows exist the `GROUP BY` answer
    must win, and it must not be double-counted into `total`."""
    tid = s.propose("a legacy-shaped topic")["id"]
    with s._connect() as conn:
        conn.execute("UPDATE topics SET status='archived' WHERE id=?", (tid,))

    stats = s.stats()
    assert stats["by_status"] == {"archived": 1}, "exactly one corpus key, no skeleton"
    assert stats["total"] == 1


# ---------------------------------------------------------------------------
# The `written` disk gate (#1276)
# ---------------------------------------------------------------------------


def _researched(s: ResearchStore, topic: str = "a topic a turn is working on",
                **kw) -> int:
    """A topic a live turn holds, which is the only state `finish` acts on."""
    tid = s.propose(topic, **kw)["id"]
    s.claim(tid, by="tester")
    assert s.get(tid)["status"] == "researching"
    return tid


@pytest.mark.parametrize("artifact_path", [None, "", "/nonexistent/x.md"])
def test_written_refuses_a_path_that_is_not_a_note_and_leaves_it_retryable(
        s: ResearchStore, artifact_path):
    """The hole this closes: `research_complete(status="written")` settled a
    topic from the caller's string alone, re-creating the tick-without-evidence
    the registry replaced. The row must stay `researching` so `reclaim_stale`
    returns it to the queue — a refusal that settled it would be a worse bug."""
    tid = _researched(s)
    kwargs = {} if artifact_path is None else {"artifact_path": artifact_path}

    with pytest.raises(ValueError, match="written"):
        s.finish(tid, "written", **kwargs)

    row = s.get(tid)
    assert row["status"] == "researching", "refused, not settled"
    assert row["artifact_path"] is None
    assert row["finished_at"] is None, "stats() must not count a refused settle"
    stats = s.stats()
    assert stats["by_status"].get("written", 0) == 0
    assert stats["last_written"] is None, (
        "the two fields that would have reported a phantom note")

    recovered = s.reclaim_stale(older_than_seconds=-1)
    assert recovered == 1, "the refused topic is the retryable kind"
    assert s.get(tid)["status"] == "queued"


def test_written_requires_the_workers_own_substantiality_bar(
        s: ResearchStore, note):
    """A file that exists but holds only a heading is the 90-of-142 failure.
    `MIN_NOTE_BYTES` is inclusive: a note of exactly the floor settles."""
    thin = note("thin.md", size=R.MIN_NOTE_BYTES - 1)
    fat = note("fat.md", size=R.MIN_NOTE_BYTES)
    assert thin.stat().st_size == 399 and fat.stat().st_size == 400

    thin_tid = _researched(s, "a topic whose note came out a stub")
    with pytest.raises(ValueError, match=str(R.MIN_NOTE_BYTES)):
        s.finish(thin_tid, "written", artifact_path=str(thin))
    assert s.get(thin_tid)["status"] == "researching"

    fat_tid = _researched(s, "a topic whose note came out whole")
    row = s.finish(fat_tid, "written", artifact_path=str(fat))
    assert row["status"] == "written"
    assert row["artifact_path"] == str(fat), "stored verbatim, not normalised"


def test_a_refused_written_does_not_suppress_the_topic(s: ResearchStore):
    """The permanent half of the damage. `propose` keys on the topic line, so a
    phantom `written` row answers every later generator with "already known as
    #N (written)" and the topic is never researched for real. Refusing must keep
    it proposable."""
    tid = _researched(s, "Qwen3 engram embedding table at inference")
    with pytest.raises(ValueError):
        s.finish(tid, "written", artifact_path="/nonexistent/x.md")

    out = s.propose("Qwen3 engram embedding table at inference")
    assert out["created"] is False and out["id"] == tid
    assert out["status"] == "researching", "not 'written': the topic is still owed"


@pytest.mark.parametrize("status", ["nothing_found", "duplicate", "rejected"])
def test_only_written_needs_a_path_at_finish(s: ResearchStore, status):
    """At SETTLE TIME, only `written` has to name a file.

    The other terminal statuses claim no file, and making `finish` demand one
    would turn "searched, found nothing" into an error. That rule is the WRITE
    path's and nothing more: the read side went the other way in #3837, where
    `stats()` judges every row that NAMES a path whatever its status — pinned by
    `test_a_path_bearing_row_is_judged_whatever_its_status`. The two are only
    consistent because of the second assertion below: a status that needs no
    path leaves the column NULL, which is the one shape the read-side scan
    reliably never reaches (`test_a_path_free_row_contributes_nothing_to_missing_artifacts`).
    """
    tid = _researched(s, f"a topic settling as {status}")
    out = s.finish(tid, status, note="no artifact for this outcome")
    assert out["status"] == status
    assert out["artifact_path"] is None


def test_exhaust_still_settles_nothing_found_without_a_path(s: ResearchStore):
    """`exhaust` is the give-up path; the check must not reach through it."""
    tid = _researched(s, "a topic that keeps failing")
    row = s.exhaust(tid, "abandoned after 3 attempts")
    assert row["status"] == "nothing_found"
    assert row["artifact_path"] is None


def test_the_store_and_the_worker_hold_the_same_note_floor():
    """The worker checked this bar first; the store now checks it too, and two
    gates that disagree about 400 bytes would bounce a real note forever."""
    from workers.sources import deep_research as D

    assert D._MIN_NOTE_BYTES == R.MIN_NOTE_BYTES == 400


# ---------------------------------------------------------------------------
# Legacy import
# ---------------------------------------------------------------------------


LEGACY = textwrap.dedent("""\
    # Research Queue

    ## Auto-queued (nightly 2024-05-16)
    - [x] Scalable robot evaluation benchmarks — BridgeData V2,Open X-Embodiment metrics
    - [x] Real-time VLM inference on edge devices — Qwen2.5-VL deployment

    ### Auto-queued (nightly 2026-09-06)
    - [x] Scalable robot evaluation benchmarks — BridgeData V2,Open X-Embodiment metrics
    - [x] A topic with a bare filename → 2026-09-06-a-real-note.md
    - [x] A topic with a vault path → `knowledge/research/2026-09-06-a-real-note.md`
    - [x] A topic in parentheses (researched 2026-09-06 → knowledge/research/2026-09-06-a-real-note.md)
    - [x] A topic naming a note that is gone → knowledge/research/2026-01-01-vanished.md
    - [x] A dated topic with no note — researched 2026-09-06
    - [] An unchecked topic in the old bracket style
    - [ ] An unchecked topic in the new bracket style
    """)


@pytest.fixture
def legacy_file(tmp_path, monkeypatch):
    vault = tmp_path / "obsidian"
    (vault / "knowledge" / "research").mkdir(parents=True)
    (vault / "knowledge" / "research" / "2026-09-06-a-real-note.md").write_text("# note")
    monkeypatch.setattr(R, "VAULT_ROOT", vault)
    path = tmp_path / "research-queue.md"
    path.write_text(LEGACY, encoding="utf-8")
    return path


def test_the_legacy_parse_collapses_duplicates_and_reads_every_suffix_form(legacy_file):
    parsed = parse_legacy(legacy_file)
    assert parsed["parsed"] == 10
    assert parsed["unchecked"] == 2
    assert len(parsed["items"]) == 9, "the repeated benchmark topic must collapse"

    repeated = parsed["items"][normalize_key(
        "Scalable robot evaluation benchmarks — BridgeData V2,Open X-Embodiment metrics")]
    assert repeated["occurrences"] == 2
    assert repeated["proposed_at"] == "2026-09-06", "the plausible date wins"

    resolved = [i for i in parsed["items"].values() if i["artifact_path"]]
    assert len(resolved) == 3, "bare, vault-relative and parenthesised all resolve"
    missing = [i for i in parsed["items"].values()
               if i["artifact_raw"] and not i["artifact_path"]]
    assert len(missing) == 1, "a note that is not on disk is recorded but not linked"


def test_a_fabricated_header_date_is_flagged(legacy_file):
    parsed = parse_legacy(legacy_file)
    assert parsed["suspect_dates"] >= 1
    edge = parsed["items"][normalize_key(
        "Real-time VLM inference on edge devices — Qwen2.5-VL deployment")]
    assert edge["header_date_suspect"] is True, "2024-05-16 predates the generator"


def test_the_import_archives_everything_and_is_idempotent(s, legacy_file):
    first = s.import_legacy(legacy_file)
    assert first["parsed"] == 10 and first["unique"] == 9
    assert first["inserted"] == 9
    assert first["artifact_found"] == 3 and first["artifact_missing"] == 1

    assert s.stats()["by_status"] == {"archived": 9}, "nothing imports as written"
    assert s.next(5) == [], "an import must not put work in front of the worker"

    second = s.import_legacy(legacy_file)
    assert second["inserted"] == 0 and second["existing"] == 9


def test_a_dry_run_writes_nothing(s, legacy_file):
    out = s.import_legacy(legacy_file, dry_run=True)
    assert out["unique"] == 9 and out["inserted"] == 0
    assert s.stats()["total"] == 0


def test_a_dry_run_measures_existing_against_the_store(s, legacy_file):
    """The dry run used to `continue` before the block that increments
    `existing`, so it printed `existing: 0` whether or not the keys were
    already in the database — and the #3085 item text quotes exactly that
    constant as its evidence that the corpus needed restoring. A dry run is
    the only pre-flight an operator has before pointing the import at
    production `research.db`, so `existing` has to be measured, not defaulted
    (#3085). A real run's `existing` comes from the `ON CONFLICT DO NOTHING`
    miss; the dry run now answers the same question by key, writing nothing."""
    empty = s.import_legacy(legacy_file, dry_run=True)
    assert empty["existing"] == 0, "an empty store has nothing to already hold"
    assert empty["inserted"] == 0 and empty["dry_run"] is True
    assert s.stats()["total"] == 0, "the empty-store dry run wrote nothing"

    s.import_legacy(legacy_file)
    out = s.import_legacy(legacy_file, dry_run=True)
    assert out["existing"] == 9, "the keys the store holds must be reported"
    assert out["inserted"] == 0
    assert s.stats()["by_status"] == {"archived": 9}, (
        "a dry run against a loaded store still writes nothing")


def test_the_import_keeps_the_occurrence_count_it_collapsed(s, legacy_file):
    s.import_legacy(legacy_file)
    row = [t for t in s.list(status="archived", limit=50)
           if t["topic"].startswith("Scalable robot")][0]
    assert row["extra"]["legacy_occurrences"] == 2


def test_an_imported_topic_still_dedups_a_new_proposal(s, legacy_file):
    """The whole point of keeping 314 archived rows."""
    s.import_legacy(legacy_file)
    out = s.propose("scalable robot evaluation benchmarks, bridgedata v2, open x-embodiment metrics")
    assert out["created"] is False and out["status"] == "archived"


# ---------------------------------------------------------------------------
# Two processes
# ---------------------------------------------------------------------------


def _propose_many(path: str, shared: int, mine: str) -> None:
    st = ResearchStore(path)
    for i in range(shared):
        st.propose(f"a shared research topic numbered {i}")
    for i in range(shared):
        st.propose(f"a {mine} research topic numbered {i}")


def test_two_processes_proposing_the_same_topics_never_double_insert(tmp_path):
    """The reason `propose` is one statement.

    `WorkQueue.enqueue` still does SELECT-then-INSERT, and across processes
    both sides can see "not there". The loser takes an IntegrityError, which
    inside the nightly generator is a tool error in a task that disables itself
    after three failures.
    """
    path = str(tmp_path / "research.db")
    ResearchStore(path)  # create the schema before the fork
    ctx = mp.get_context("fork")
    procs = [ctx.Process(target=_propose_many, args=(path, 12, who))
             for who in ("alpha", "beta")]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
    assert [p.exitcode for p in procs] == [0, 0]

    st = ResearchStore(path)
    assert st.stats()["total"] == 12 + 12 + 12, "12 shared + 12 each distinct"
    with st._connect() as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def _claim_all(path: str, ids: list[int], out) -> None:
    st = ResearchStore(path)
    for topic_id in ids:
        if st.claim(topic_id, by=f"p{id(out)}") is not None:
            out.append(topic_id)


def test_two_processes_racing_for_topics_each_get_their_own(tmp_path):
    path = str(tmp_path / "research.db")
    st = ResearchStore(path)
    ids = [st.propose(f"a claimable topic numbered {i}")["id"] for i in range(30)]

    ctx = mp.get_context("fork")
    with ctx.Manager() as manager:
        a, b = manager.list(), manager.list()
        procs = [ctx.Process(target=_claim_all, args=(path, ids, a)),
                 ctx.Process(target=_claim_all, args=(path, ids, b))]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=60)
        assert [p.exitcode for p in procs] == [0, 0]
        won_a, won_b = list(a), list(b)

    assert sorted(won_a + won_b) == ids, "every topic claimed exactly once"
    assert not (set(won_a) & set(won_b)), "a topic was claimed by both"


# ---------------------------------------------------------------------------
# The read-side re-verification (#3190)
# ---------------------------------------------------------------------------


def test_stats_names_a_written_row_whose_artifact_vanished(s, note):
    """A `written` row is a claim about disk that used to be checked exactly once.

    `_require_real_note` verifies the note at settle time and nothing looked
    again, so a note deleted afterwards left the row believed forever: `stats()`
    itself served it up as `last_written`. That is #3190's failure one step on —
    its card had no row at all, so not even the write-time check ran — and the
    planted-and-deleted row here is the reproduction the item's acceptance names.
    """
    tid = s.propose("a topic whose note gets deleted after it settles")["id"]
    s.claim(tid, by="w")
    art = note("gone-soon.md")
    s.finish(tid, "written", artifact_path=str(art))

    clean = s.stats()["missing_artifacts"]
    assert clean == {"count": 0, "paths": []}, (
        f"a row whose note is on disk was reported missing: {clean}")

    art.unlink()
    missing = s.stats()["missing_artifacts"]
    assert missing["count"] == 1, missing
    assert missing["paths"] == [str(art)], (
        "the payload has to name the file, or the reader goes looking blind")


def test_missing_artifacts_survives_reopening_the_store(s, note):
    """The check is a read of the row and the disk, not a process-local memory.

    The registry is written by the worker and read by a chat turn in another
    process, so a report that only appeared in the process that settled the topic
    would answer `research_stats` with a clean bill of health.
    """
    tid = s.propose("a topic whose note is deleted, then the store reopened")["id"]
    s.claim(tid, by="w")
    art = note("also-gone.md")
    s.finish(tid, "written", artifact_path=str(art))
    art.unlink()
    assert ResearchStore(s.path).stats()["missing_artifacts"]["count"] == 1


def test_a_path_bearing_archived_row_is_judged_against_disk(s):
    """A row that names a note is judged whatever its status — `archived` included.

    The read path used to skip the legacy corpus by status, on the belief that its
    artifacts were meant to be absent. The corpus disproved that belief: most of
    its 314 rows name no note, but 20 of them (measured 2026-11-06) name notes that
    are sitting on disk, which makes them ordinary rows — and a corpus row that
    lost its file is a finding, not a designed absence. So an `archived` row
    pointing at a deleted note has to be counted here exactly as a `written` one is.
    """
    gone = "/nonexistent/knowledge/research/2024-01-01-legacy-note.md"
    archived = s.propose("a legacy-shaped topic")["id"]
    with s._connect() as conn:
        conn.execute("UPDATE topics SET status='archived', artifact_path=? WHERE id=?",
                     (gone, archived))
    missing = s.stats()["missing_artifacts"]
    assert missing == {"count": 1, "paths": [gone]}, (
        "an archived row that names a path is judged like any other row: "
        + repr(missing))


@pytest.mark.parametrize("status",
                         ["archived", "duplicate", "rejected",
                          "nothing_found", "researching"])
def test_a_path_bearing_row_is_judged_whatever_its_status(s, status):
    """The scan's "whatever its status", pinned one status at a time.

    The node above pins the one status a real writer ever stores a legacy path
    under, which is the population #3837 was filed for; this one pins that the
    decision really is made by the PATH and not by a second list of statuses the
    query happens to accept. Each arm plants a row that names a note which is not
    there and has to be reported. The `researching` arm is the one worth arguing
    for: an in-flight row never names a path by itself — the note path a worker is
    handed travels in its queue payload, and only `finish` and `import_legacy`
    write the column — so a `researching` row holding one was put there
    deliberately, and is a row that lost a note like any other row that lost one.
    Narrowing the scan back to `WHERE status='written'` reddens every arm here and
    none of the path-free arms in the node below, so the two together are what
    tell "judged by path" from "judged by status".
    """
    gone = f"/nonexistent/knowledge/research/{status}-note.md"
    tid = s.propose(f"a row left holding a path at status {status}")["id"]
    if status == "researching":
        s.claim(tid, by="w")
    with s._connect() as conn:
        conn.execute("UPDATE topics SET status=?, artifact_path=? WHERE id=?",
                     (status, gone, tid))
    missing = s.stats()["missing_artifacts"]
    assert missing == {"count": 1, "paths": [gone]}, (
        f"a {status} row naming a path has to be judged: {missing}")


def test_a_path_free_row_contributes_nothing_to_missing_artifacts(s):
    """Naming no path is not the same as losing one — at every status but `written`.

    This is what keeps the legacy corpus out of the report now that the scan reads
    every status instead of one: 294 of its 314 rows store no `artifact_path` at
    all, and the path-free terminal outcomes (`nothing_found` here, settled through
    `finish` so the column is really NULL rather than merely unseeded) claim nothing
    about disk either. Both spellings of "no path" are pinned, NULL as
    `import_legacy` writes it and the empty string a hand edit leaves, because the
    query excludes both and only one of them is what the import ever produces.

    `written` is judged even when its path is empty, deliberately: the status IS the
    claim that a note exists, so an empty path there is the phantom #1276 refuses at
    write time. `test_a_written_row_with_an_empty_or_unusable_path_is_reported`
    pins that side, and it is the asymmetry this node's quiet reading depends on.
    """
    archived_null = s.propose("a legacy topic that named no note")["id"]
    archived_empty = s.propose("a hand-edited legacy topic with an empty path")["id"]
    with s._connect() as conn:
        conn.execute("UPDATE topics SET status='archived', artifact_path=NULL "
                     "WHERE id=?", (archived_null,))
        conn.execute("UPDATE topics SET status='archived', artifact_path='' "
                     "WHERE id=?", (archived_empty,))
    quiet = s.propose("a topic answered with no note")["id"]
    s.claim(quiet, by="w")
    s.finish(quiet, "nothing_found", note="nothing out there")
    assert s.stats()["missing_artifacts"] == {"count": 0, "paths": []}, (
        "a row that never named a note is not a missing artifact")


def test_a_written_row_with_an_empty_or_unusable_path_is_reported(s, tmp_path):
    """The phantom row #1276 refuses at write time, seen from the read side.

    A `written` row can still reach that state from outside `finish` — a hand
    edit, or a row predating the write gate — and a path that exists but names a
    DIRECTORY is the other unusable shape: `is_file()` says no, so it reads as
    missing rather than raising. Both have to be missing, not clean.
    """
    a_directory = str(tmp_path / "knowledge")  # exists, and is not a file
    with s._connect() as conn:
        conn.execute(
            "INSERT INTO topics (topic, key, status, proposed_at, finished_at, "
            "artifact_path) VALUES ('empty', 'empty', 'written', "
            "'2026-10-23T00:00:00+00:00', '2026-10-23T00:00:00+00:00', '')")
        conn.execute(
            "INSERT INTO topics (topic, key, status, proposed_at, finished_at, "
            "artifact_path) VALUES ('dir', 'dir', 'written', "
            "'2026-10-23T00:00:00+00:00', '2026-10-23T00:00:00+00:00', ?)",
            (a_directory,))
    missing = s.stats()["missing_artifacts"]
    assert missing["count"] == 2, missing
    assert missing["paths"] == ["", a_directory], missing
    assert R.artifact_on_disk("") is False, "an empty path is not a note"
    assert R.artifact_on_disk(a_directory) is False, (
        "a directory is not a note either, and must not raise")
    # The third unusable shape cannot be stored at all — SQLite refuses a NUL in
    # any string it is handed — so that posture is pinned on the predicate itself.
    assert R.artifact_on_disk("has" + chr(0) + "nul") is False, (
        "a path no syscall accepts must read as absent, not raise ValueError")


def test_a_truncated_note_is_not_counted_as_missing(s, note):
    """The two disk rules stay two rules: this one asks only "is it there".

    `_require_real_note` demands `MIN_NOTE_BYTES` at settle time; a note that was
    substantial then and was trimmed afterwards is a different finding, and
    reporting it as MISSING would send the reader looking for a file that is
    sitting right there.
    """
    tid = s.propose("a topic whose note is trimmed after it settles")["id"]
    s.claim(tid, by="w")
    art = note("trimmed.md")
    s.finish(tid, "written", artifact_path=str(art))
    art.write_bytes(b"x" * 12)
    assert s.stats()["missing_artifacts"]["count"] == 0, (
        "a small file on disk is present; the size rule belongs to `finish`")
    assert R.artifact_on_disk(str(art)) is True


# ---------------------------------------------------------------------------
# #4272 — a `written` settle makes its own claim durable
#
# Base measurement for this whole block, taken with the shipped read path at the
# round's base commit 9aec6a92:
#
#     cd ~/lloyd && .venvs/lloyd/bin/python -m app.research_store stats
#     → missing_artifacts.count: 11   (11 of 171 `written` rows, 522 topics)
#
# Eleven notes that `finish` had verified against `MIN_NOTE_BYTES` and settled
# `written` were no longer on disk, the newest of them (topic 829, `finished_at`
# 2026-11-19T02:16:11Z) gone within three minutes of settling, and none of the
# eleven ever entered the vault's git history. The row's claim was durable only as
# long as some other process let the file sit there, so the tests below pin the
# copy that now happens inside the verified moment.
# ---------------------------------------------------------------------------


def test_a_written_note_deleted_from_the_working_tree_is_still_recoverable(tmp_path):
    """Clause 1. The note lives in a directory standing in for the vault; the
    registry and its snapshot store live somewhere else entirely, which is the
    point: the copy is not a second name for the same file.

    The registry is written by the worker and read by a chat turn in another
    process, so recovery is exercised through a second store object on the same
    db rather than through the one that settled the row.
    """
    vault = tmp_path / "vault"
    data = tmp_path / "lloyd-data"
    store = ResearchStore(data / "research.db",
                          note_store=data / R.NOTE_SNAPSHOT_DIRNAME)
    art = vault / "knowledge" / "research" / "2026-11-19-a-note.md"
    art.parent.mkdir(parents=True)
    art.write_bytes(b"# A note\n" + b"body line\n" * 60)
    assert art.stat().st_size >= R.MIN_NOTE_BYTES

    tid = store.propose("a topic whose note is deleted minutes after it settles")["id"]
    store.claim(tid, by="w")
    settled = store.finish(tid, "written", artifact_path=str(art))
    assert settled["status"] == "written"

    art.unlink()
    assert store.stats()["missing_artifacts"] == {"count": 1, "paths": [str(art)]}, (
        "the working-tree loss stopped being reported, which would read as a "
        "clean vault rather than as a note that still has a copy")

    recovered = ResearchStore(data / "research.db",
                              note_store=data / R.NOTE_SNAPSHOT_DIRNAME
                              ).recover_note(str(art))
    assert recovered is not None, "the settle left no durable copy of its own note"
    assert recovered.stat().st_size >= R.MIN_NOTE_BYTES, (
        f"the recovered copy is {recovered.stat().st_size} bytes, under the "
        f"{R.MIN_NOTE_BYTES} the settle verified against")
    assert recovered.read_bytes().startswith(b"# A note\n"), (
        "the recovered bytes are not the note that was verified")
    # The store the nightly vault commit pass owns holds nothing: the copy is
    # outside the vault tree it snapshots and reverts.
    assert vault not in recovered.parents

    # Positive control. A `rejected` settle carries an artifact path too, and gets
    # no copy — the snapshot belongs to `written`, which is the only status whose
    # claim is that a note exists. Without this line the assertions above would
    # also pass if the copy were keyed on "any status with a path".
    other = vault / "knowledge" / "research" / "2026-11-19-other.md"
    other.write_bytes(b"y" * R.MIN_NOTE_BYTES)
    tid2 = store.propose("a topic that is rejected rather than written")["id"]
    store.claim(tid2, by="w")
    store.finish(tid2, "rejected", artifact_path=str(other), note="not actionable")
    assert store.recover_note(str(other)) is None, (
        "a rejected settle snapshotted its artifact, so the copy is not keyed to "
        "the `written` claim this exists to protect")
    assert store.stats()["undurable_notes"] == {"count": 0, "ids": []}


def test_the_snapshot_root_is_an_injected_value_and_stages_nothing_in_the_vault(tmp_path):
    """Clause 2, and the trap the item names: a unit test must never run `git add`
    against the live vault (the precedent is recorded at
    `scripts/automod/witness.py:465` and `scripts/autoresearch/promote.py:1432`).

    So the vault here is a real temporary git repo and the note is a file it has
    never tracked — the exact state topic 829's note was in when it vanished. The
    assertions that matter are the two `git status` readings: the note is untracked
    before the settle and STILL untracked after it. The durability step copies; it
    never stages, so no test of this mechanism can put anything into a vault's
    index, live or otherwise.
    """
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(vault), *args], capture_output=True,
                              text=True, check=True).stdout

    vault = tmp_path / "vault-repo"
    vault.mkdir()
    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "test")
    # `knowledge/research/` has to be a TRACKED directory before the note appears,
    # or `git status --porcelain` collapses it to `?? knowledge/research/` and the
    # reading below cannot tell "this note is untracked" from "this whole folder
    # is", which is the distinction the clause turns on.
    (vault / "knowledge" / "research").mkdir(parents=True)
    (vault / "knowledge" / "research" / "2026-11-18-committed.md").write_text("x\n")
    git("add", "knowledge/research/2026-11-18-committed.md")
    git("commit", "-q", "-m", "seed the research dir")

    data = tmp_path / "lloyd-data"
    store = ResearchStore(data / "research.db", note_store=tmp_path / "snapshots")
    art = vault / "knowledge" / "research" / "2026-11-19-uncommitted.md"
    art.write_bytes(b"# Uncommitted\n" + b"line\n" * 80)

    assert git("status", "--porcelain").split()[:2] == ["??", "knowledge/research/2026-11-19-uncommitted.md"], (
        "the fixture's note is not the untracked-and-never-committed file this "
        "reproduces, so nothing below measures the case")
    assert git("log", "--all", "--", str(art.relative_to(vault))) == "", (
        "the note already has history, which the lost notes never had")

    tid = store.propose("a topic whose note is untracked in the vault")["id"]
    store.claim(tid, by="w")
    store.finish(tid, "written", artifact_path=str(art))

    assert git("status", "--porcelain").split()[:2] == ["??", "knowledge/research/2026-11-19-uncommitted.md"], (
        "the durability step staged or committed the note: it must copy to its own "
        "store and leave the vault's index exactly as it found it")
    assert git("log", "--all", "--", str(art.relative_to(vault))) == ""
    assert store.recover_note(str(art)) is not None
    assert (tmp_path / "snapshots").is_dir(), (
        "the copy ignored the injected root and went somewhere else")
    assert not (data / R.NOTE_SNAPSHOT_DIRNAME).exists(), (
        "the injected root was ignored in favour of the derived one")

    # The derived default, which is what production gets: a sibling of the
    # registry, so a registry redirected by `configure()` or `LLOYD_RESEARCH_DB`
    # carries its snapshots with it instead of writing beside the live one.
    assert ResearchStore(data / "research.db").note_store \
        == data / R.NOTE_SNAPSHOT_DIRNAME
    reconfigured = R.configure(data / "research.db",
                               note_store=tmp_path / "other-snaps")
    assert reconfigured.note_store == tmp_path / "other-snaps"
    assert R.store().note_store == tmp_path / "other-snaps", (
        "`store()` returned a cached object still pointing at the old root, so the "
        "reconfiguration would be reported and not applied")


def test_a_failed_snapshot_still_settles_the_row_and_stats_names_it(tmp_path):
    """Clause 3. The failure is made unfakeable: the snapshot root is the path of
    an existing regular FILE, so `mkdir` raises `FileExistsError` — an `OSError`
    the durability step catches — on every attempt, by any user, without relying on
    a mode bit a root-run test would ignore.

    The settle is the stronger claim: a note that cleared `MIN_NOTE_BYTES` and a
    row that must reach its terminal state outrank a copy that could not be
    written. Refusing here would instead leave a research run holding a real note
    and an unsettled row, and `finish` refuses to move a `written` row afterwards,
    so a refused settle would be permanent.
    """
    data = tmp_path / "lloyd-data"
    data.mkdir()
    blocker = data / R.NOTE_SNAPSHOT_DIRNAME
    blocker.write_bytes(b"not a directory")
    store = ResearchStore(data / "research.db", note_store=blocker)

    art = tmp_path / "vault" / "knowledge" / "research" / "note.md"
    art.parent.mkdir(parents=True)
    art.write_bytes(b"z" * R.MIN_NOTE_BYTES)

    tid = store.propose("a topic whose snapshot cannot be written")["id"]
    store.claim(tid, by="w")
    settled = store.finish(tid, "written", artifact_path=str(art))

    assert settled["status"] == "written", (
        "a durability failure refused the settle, which is the one outcome this "
        "clause exists to forbid")
    assert settled["extra"]["note_durable"] is False
    assert settled["extra"]["note_durable_error"], (
        "the row says only that it is not durable, not why")

    stats = store.stats()
    assert stats["undurable_notes"] == {"count": 1, "ids": [tid]}, stats
    # Named separately from the missing-artifact finding: the note IS on disk here,
    # so a reader told it is missing would go looking for a file that is there.
    assert stats["missing_artifacts"] == {"count": 0, "paths": []}

    # Positive control: the same registry, same note, one row whose copy DID land.
    # Without it the count of 1 would also be produced by a step that never
    # succeeds, and `note_durable: False` would be a constant rather than a verdict.
    good = ResearchStore(data / "research.db", note_store=data / "snaps")
    art2 = tmp_path / "vault" / "knowledge" / "research" / "note2.md"
    art2.write_bytes(b"z" * R.MIN_NOTE_BYTES)
    tid2 = good.propose("a topic whose snapshot lands")["id"]
    good.claim(tid2, by="w")
    settled2 = good.finish(tid2, "written", artifact_path=str(art2))
    assert settled2["extra"]["note_durable"] is True
    after = good.stats()
    assert after["undurable_notes"] == {"count": 1, "ids": [tid]}, (
        "the durable row was named too, or the undurable one was not")
    art2.unlink()
    assert good.recover_note(str(art2)) is not None


def test_a_row_that_recorded_no_flag_is_not_counted_as_undurable(tmp_path):
    """The control that makes the count above mean something.

    Every `written` row settled before this mechanism shipped — all 171 of them at
    the round's base — recorded neither verdict, and a `stats()` that read their
    silence as failure would report the whole registry broken the night this
    landed. Three shapes must all stay unnamed: no `extra` at all, an `extra` with
    some other key, and an `extra` that is not parseable JSON.
    """
    store = ResearchStore(tmp_path / "research.db", note_store=tmp_path / "snaps")
    ids = []
    for topic in ("settled before the snapshot existed",
                  "settled with an unrelated extra key",
                  "settled with an unreadable extra"):
        tid = store.propose(topic)["id"]
        store.claim(tid, by="w")
        ids.append(tid)
        with store._connect() as conn:
            conn.execute("UPDATE topics SET status='written', finished_at=?, "
                         "session_id='s' WHERE id=?",
                         (R._now(), tid))
    with store._connect() as conn:
        conn.execute("UPDATE topics SET extra='{\"review_id\": 7}' WHERE id=?", (ids[1],))
        conn.execute("UPDATE topics SET extra='not json' WHERE id=?", (ids[2],))

    stats = store.stats()
    assert stats["undurable_notes"] == {"count": 0, "ids": []}, stats
    assert stats["by_status"]["written"] == 3, (
        "the fixture settled no rows, so the count above would prove nothing")
