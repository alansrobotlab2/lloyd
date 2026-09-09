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
import sqlite3
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app import research_store as R
from app.research_store import ResearchStore, StoreUnavailable, normalize_key, parse_legacy


@pytest.fixture
def s(tmp_path) -> ResearchStore:
    return ResearchStore(tmp_path / "research.db")


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
    under the autoimplement gate means a canary writing into production."""
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


def test_similar_reports_the_status_so_a_generator_can_act_on_it(s):
    first = s.propose("Diffusion policy priors for sim-to-real transfer")
    s.claim(first["id"], by="test")
    s.finish(first["id"], "written", artifact_path="/tmp/x.md")
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


def test_finishing_records_the_artifact_and_the_session(s):
    topic = s.propose("a topic that got written up")["id"]
    s.claim(topic, by="worker", queue_id=4)
    row = s.finish(topic, "written", artifact_path="/home/x/knowledge/research/a.md",
                   note="facts=7 sources=4", session_id="20260908_x")
    assert row["status"] == "written"
    assert row["artifact_path"].endswith("a.md")
    assert row["session_id"] == "20260908_x" and row["finished_at"]
    assert row["queue_id"] == 4, "the link back to the workers.db run is kept"


def test_a_settled_outcome_is_not_rewritten(s):
    topic = s.propose("a settled topic")["id"]
    s.claim(topic, by="worker")
    s.finish(topic, "nothing_found", note="three searches, nothing")
    with pytest.raises(ValueError, match="already"):
        s.finish(topic, "written", artifact_path="/tmp/y.md")


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


def test_every_transition_leaves_a_trace(s):
    topic = s.propose("a topic", proposed_by="65", signal="health-report")["id"]
    s.claim(topic, by="worker", queue_id=1)
    s.release(topic, error="nope", backoff_seconds=1)
    s.claim(topic, by="worker", queue_id=2)
    s.finish(topic, "written", artifact_path="/tmp/a.md")
    with s._connect() as conn:
        events = [r["event"] for r in conn.execute(
            "SELECT event FROM events WHERE topic_id=? ORDER BY id", (topic,)).fetchall()]
    assert events == ["proposed", "claimed", "released", "claimed", "written"]


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def test_listing_filters_and_recent_excludes_the_legacy_import(s):
    live = s.propose("a live topic")["id"]
    s.claim(live, by="w")
    s.finish(live, "written", artifact_path="/tmp/a.md")
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
