"""`deep-research` — the source that replaced a write-only sink.

`domain-research` asked a local model to write a knowledge note from
`vault_recall` alone, with `http_search` sitting unused in its toolbox: 142
notes, 90 of them empty, five citing a URL, none ever promoted. What is pinned
here is the machinery that makes this one's outcomes trustworthy — disk
decides whether something was written, the registry owns the retry, and the
turn cannot reach the tools a fetched web page would want.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app import research_store as R
from workers.queue import QueueItem, WorkQueue
from workers.sources import deep_research as D
from workers.sources import _common as C
from workers.sources._common import DrainActive, TurnTimeout, WORKER_AUTOMOD_BAN


@pytest.fixture
def registry(tmp_path):
    store = R.configure(tmp_path / "research.db")
    yield store
    R.reset()


@pytest.fixture
def queue(tmp_path) -> WorkQueue:
    return WorkQueue(tmp_path / "workers.db")


@pytest.fixture
def notes(tmp_path, monkeypatch) -> Path:
    """A throwaway `knowledge/research` the source writes into."""
    d = tmp_path / "obsidian" / "knowledge" / "research"
    d.mkdir(parents=True)
    monkeypatch.setattr(D, "NOTES_DIR", d)
    monkeypatch.setattr(D, "VAULT_ROOT", tmp_path / "obsidian")
    monkeypatch.setattr(D, "_vault_dirty_paths", lambda: set())
    return d


def _item(payload: dict, item_id: int = 1, **kw) -> QueueItem:
    base = dict(id=item_id, source=D.NAME, kind="topic", priority=70,
                payload=payload, dedup_key=None, state="running", attempts=1,
                enqueued_at="", claimed_at=None, claimed_by=None,
                completed_at=None, error=None)
    base.update(kw)
    return QueueItem(**base)


def _turn(text: str, *, writes: Path | None = None, **kw):
    """Stand in for a session-backed turn.

    `writes` is the model writing its note during the turn. Creating the file
    before `execute` instead would trip the recovery branch, which exists for
    the opposite case — a note left behind by an attempt that died before it
    could record anything.
    """
    async def run(prompt, **kwargs):
        run.seen = kwargs
        run.prompt = prompt
        if writes is not None:
            writes.write_text("# A note\n\n" + "body " * 200, encoding="utf-8")
        return {"text": text, "session_id": "20260908_deep_x",
                "stop_reason": kw.get("stop_reason", "stop"),
                "num_turns": kw.get("num_turns", 12), "errors": []}
    run.seen = {}
    run.prompt = ""
    return run


def _payload(registry, notes, topic="Speculative decoding for local agent loops"):
    row = registry.propose(topic, domain="local-llm-serving")
    path = D.note_path_for(topic)
    return {"topic_id": row["id"], "topic": topic, "domain": "local-llm-serving",
            "artifact_path": str(path), "max_turns": 60}, row["id"], path


# ---------------------------------------------------------------------------
# The RESULT block
# ---------------------------------------------------------------------------


def test_a_well_formed_block_parses():
    parsed = D.parse_result(
        "prose about the topic\n\nRESULT: written\nNOTE: /x/2026-09-08-a.md\n"
        "FACTS: 7\nSOURCES: 4\n")
    assert parsed["result"] == "written"
    assert parsed["note"] == "/x/2026-09-08-a.md"
    assert parsed["facts"] == "7" and parsed["sources"] == "4"


def test_the_last_block_wins_if_the_model_restates_itself():
    """A model that answers, reconsiders and re-answers would otherwise have
    its first verdict paired with its last evidence."""
    parsed = D.parse_result(
        "RESULT: nothing_found\n\nactually, on reflection\n\n"
        "RESULT: written\nNOTE: /x/b.md\n")
    assert parsed["result"] == "written" and parsed["note"] == "/x/b.md"


@pytest.mark.parametrize("verb", ["written", "nothing_found", "duplicate"])
def test_every_declared_outcome_parses(verb):
    assert D.parse_result(f"RESULT: {verb}\n")["result"] == verb


def test_an_invented_outcome_is_rejected():
    assert D.parse_result("RESULT: mostly_done\n") is None


def test_no_block_at_all_is_none():
    assert D.parse_result("I researched the thing and it was interesting.") is None
    assert D.parse_result("") is None


def test_a_quoted_or_bracketed_path_is_unwrapped():
    parsed = D.parse_result("RESULT: written\nNOTE: `/x/2026-09-08-a.md`\n")
    assert parsed["note"] == "/x/2026-09-08-a.md"


# ---------------------------------------------------------------------------
# The note path
# ---------------------------------------------------------------------------


def test_the_source_picks_the_path_not_the_model(notes):
    """The skill used to say "run `date +%F` via bash, never guess it" — a
    workaround for a problem that only existed because the model chose. Past
    runs produced notes misdated by days, some dated in the future."""
    path = D.note_path_for("Qwen3 engram embedding table at inference",
                           today="2026-09-08")
    assert path.parent == notes
    assert path.name == "2026-09-08-qwen3-engram-embedding-table-at-inference.md"


def test_the_slug_survives_punctuation_and_length(notes):
    path = D.note_path_for("MoE / expert residency: PCIe vs compute — 2x RTX 3090!!",
                           today="2026-09-08")
    assert path.name.startswith("2026-09-08-moe-expert-residency-pcie-vs-compute")
    assert path.suffix == ".md" and "/" not in path.name[11:]


# ---------------------------------------------------------------------------
# Enqueueing
# ---------------------------------------------------------------------------


async def test_one_topic_per_tick_with_a_dedup_key(registry, queue, notes):
    for i in range(3):
        registry.propose(f"a distinct research topic numbered {i}")
    await D.enqueue_if_due(queue, {"daily_max": 3})

    items = queue.list_items(source=D.NAME)
    assert len(items) == 1, "the whole backlog must not become one day of GPU"
    assert items[0].dedup_key == f"deep-research:{items[0].payload['topic_id']}"
    assert items[0].payload["artifact_path"].endswith(".md")


async def test_the_same_topic_is_not_enqueued_twice(registry, queue, notes):
    registry.propose("a topic")
    await D.enqueue_if_due(queue, {})
    await D.enqueue_if_due(queue, {})
    assert len(queue.list_items(source=D.NAME)) == 1


async def test_the_daily_budget_stops_the_source(registry, queue, notes):
    for i in range(4):
        row = registry.propose(f"topic number {i}")
        if i < 2:
            registry.claim(row["id"], by="w")
            registry.finish(row["id"], "written", artifact_path=f"/tmp/{i}.md")
    await D.enqueue_if_due(queue, {"daily_max": 2})
    assert queue.list_items(source=D.NAME) == [], "two settled today is the budget"


async def test_an_empty_registry_enqueues_nothing(registry, queue, notes):
    await D.enqueue_if_due(queue, {})
    assert queue.list_items(source=D.NAME) == []


async def test_a_crashed_turn_is_reclaimed_before_the_next_tick(registry, queue, notes):
    from datetime import datetime, timedelta, timezone
    topic = registry.propose("a topic whose backend died mid-turn")["id"]
    registry.claim(topic, by="worker", queue_id=99)
    old = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat()
    with registry._connect() as conn:
        conn.execute("UPDATE topics SET started_at=? WHERE id=?", (old, topic))

    await D.enqueue_if_due(queue, {"max_duration_seconds": 1800})
    assert len(queue.list_items(source=D.NAME)) == 1, "the stuck topic came back"


# ---------------------------------------------------------------------------
# Executing: the happy paths
# ---------------------------------------------------------------------------


async def test_a_written_note_is_recorded_with_its_artifact(registry, queue, notes, monkeypatch):
    payload, topic_id, path = _payload(registry, notes)
    monkeypatch.setattr(D, "run_prompt_in_session", _turn(
        f"RESULT: written\nNOTE: {path}\nFACTS: 7\nSOURCES: 4\n", writes=path))

    out = await D.execute(_item(payload, item_id=42))
    assert out["status"] == "success" and out["meta"]["result"] == "written"
    row = registry.get(topic_id)
    assert row["status"] == "written"
    assert row["artifact_path"] == str(path)
    assert row["session_id"] == "20260908_deep_x"
    assert row["queue_id"] == 42, "the link back to the workers.db run"
    assert "facts=7" in row["outcome_note"]


async def test_nothing_found_is_a_successful_run_with_a_recorded_outcome(
        registry, queue, notes, monkeypatch):
    payload, topic_id, _ = _payload(registry, notes)
    monkeypatch.setattr(D, "run_prompt_in_session", _turn(
        "RESULT: nothing_found\nFACTS: 0\nSOURCES: 3\n"))

    out = await D.execute(_item(payload))
    assert out["status"] == "success", "a real answer is not a failed run"
    assert registry.get(topic_id)["status"] == "nothing_found"
    assert registry.recent(days=1)[0]["status"] == "nothing_found"


async def test_a_duplicate_names_what_covers_it(registry, queue, notes, monkeypatch):
    payload, topic_id, _ = _payload(registry, notes)
    monkeypatch.setattr(D, "run_prompt_in_session", _turn(
        "RESULT: duplicate\nDUPLICATE_OF: knowledge/research/2026-08-20-x.md\n"))

    out = await D.execute(_item(payload))
    assert out["meta"]["result"] == "duplicate"
    assert "2026-08-20-x.md" in registry.get(topic_id)["outcome_note"]


async def test_an_unsupported_duplicate_is_downgraded(registry, queue, notes, monkeypatch):
    """"We already know this" with nothing named is indistinguishable from
    giving up, so it is recorded as giving up and says so."""
    payload, topic_id, _ = _payload(registry, notes)
    monkeypatch.setattr(D, "run_prompt_in_session", _turn("RESULT: duplicate\n"))

    out = await D.execute(_item(payload))
    assert out["meta"]["result"] == "nothing_found"
    assert registry.get(topic_id)["extra"]["downgraded_from"] == "duplicate"


# ---------------------------------------------------------------------------
# Executing: disk is the source of truth
# ---------------------------------------------------------------------------


async def test_a_claim_of_written_with_no_note_on_disk_is_a_failure(
        registry, queue, notes, monkeypatch):
    """The model's claim that it wrote something is a claim."""
    payload, topic_id, path = _payload(registry, notes)
    monkeypatch.setattr(D, "run_prompt_in_session", _turn(
        f"RESULT: written\nNOTE: {path}\nFACTS: 9\n"))

    out = await D.execute(_item(payload))
    assert out["status"] == "failed"
    assert "not on disk" in out["summary"]
    assert registry.get(topic_id)["status"] == "queued", "it will be tried again"


async def test_a_stub_of_a_note_does_not_count_as_written(
        registry, queue, notes, monkeypatch):
    payload, topic_id, path = _payload(registry, notes)
    path.write_text("# A note\n", encoding="utf-8")
    monkeypatch.setattr(D, "run_prompt_in_session", _turn(
        f"RESULT: written\nNOTE: {path}\n"))

    out = await D.execute(_item(payload))
    assert out["status"] == "failed"


async def test_a_note_written_without_the_block_is_still_written(
        registry, queue, notes, monkeypatch):
    """The note is there; the model just did not sign off. Retrying would
    write a second one."""
    payload, topic_id, path = _payload(registry, notes)
    monkeypatch.setattr(D, "run_prompt_in_session",
                        _turn("I have finished the research.", writes=path))

    out = await D.execute(_item(payload))
    assert out["status"] == "success" and out["meta"]["result"] == "written"
    row = registry.get(topic_id)
    assert row["status"] == "written" and row["extra"]["result_block_missing"] is True


async def test_a_note_from_a_dead_attempt_is_recovered_without_a_second_turn(
        registry, queue, notes, monkeypatch):
    payload, topic_id, path = _payload(registry, notes)
    path.write_text("# A note\n\n" + "body " * 200, encoding="utf-8")

    def must_not_run(*a, **k):
        raise AssertionError("spent a turn on a topic already written up")
    monkeypatch.setattr(D, "run_prompt_in_session", must_not_run)

    out = await D.execute(_item(payload))
    assert out["status"] == "success"
    assert registry.get(topic_id)["extra"]["recovered"] is True


# ---------------------------------------------------------------------------
# Executing: failure and retry
# ---------------------------------------------------------------------------


async def test_an_empty_turn_releases_the_topic_for_a_later_attempt(
        registry, queue, notes, monkeypatch):
    payload, topic_id, _ = _payload(registry, notes)
    monkeypatch.setattr(D, "run_prompt_in_session",
                        _turn("", stop_reason="max_turns"))

    out = await D.execute(_item(payload))
    assert out["status"] == "failed"
    row = registry.get(topic_id)
    assert row["status"] == "queued" and row["attempts"] == 1
    assert row["not_before"], "the backoff is the registry's, not the queue's"
    assert registry.next(5) == [], "and it holds"


async def test_the_last_attempt_gives_up_and_says_so(
        registry, queue, notes, monkeypatch):
    """Otherwise it is a cycle: `enqueue_if_due` would offer it every tick."""
    payload, topic_id, _ = _payload(registry, notes)
    with registry._connect() as conn:
        conn.execute("UPDATE topics SET attempts=2 WHERE id=?", (topic_id,))
    monkeypatch.setattr(D, "run_prompt_in_session", _turn(""))

    out = await D.execute(_item(payload))
    assert out["status"] == "failed" and "abandoned" in out["summary"]
    row = registry.get(topic_id)
    assert row["status"] == "nothing_found"
    assert row["extra"]["reason"] == "exhausted"


async def test_a_turn_timeout_counts_as_an_attempt(registry, queue, notes, monkeypatch):
    payload, topic_id, _ = _payload(registry, notes)

    async def times_out(*a, **k):
        raise TurnTimeout("worker turn exceeded 1740s; backend cancel accepted")
    monkeypatch.setattr(D, "run_prompt_in_session", times_out)

    out = await D.execute(_item(payload))
    assert out["status"] == "failed" and out["meta"]["turn_timeout"] is True
    assert registry.get(topic_id)["status"] == "queued"


async def test_a_landing_gives_the_topic_straight_back(registry, queue, notes, monkeypatch):
    payload, topic_id, _ = _payload(registry, notes)

    async def draining(*a, **k):
        raise DrainActive("lloyd is landing a code update")
    monkeypatch.setattr(D, "run_prompt_in_session", draining)

    out = await D.execute(_item(payload))
    assert out["status"] == "skipped", "a landing is not the topic's fault"
    row = registry.get(topic_id)
    assert row["status"] == "queued" and not row["not_before"]


async def test_a_topic_someone_else_took_is_skipped(registry, queue, notes, monkeypatch):
    payload, topic_id, _ = _payload(registry, notes)
    registry.claim(topic_id, by="somebody-else", queue_id=999)

    def must_not_run(*a, **k):
        raise AssertionError("researched a topic another worker holds")
    monkeypatch.setattr(D, "run_prompt_in_session", must_not_run)

    out = await D.execute(_item(payload))
    assert out["status"] == "skipped"


async def test_an_item_with_no_topic_id_fails_cleanly(registry, queue, notes):
    out = await D.execute(_item({"topic": "orphan"}))
    assert out["status"] == "failed" and "topic_id" in out["summary"]


# ---------------------------------------------------------------------------
# The prompt and the tool posture
# ---------------------------------------------------------------------------


async def test_the_turn_cannot_reach_what_a_fetched_page_would_want(
        registry, queue, notes, monkeypatch):
    """The turn reads arbitrary web pages. Before this source, no session-backed
    worker passed a deny list at all — `/api/message/stream` builds its
    disallowed set from config plus the request body, and nothing in it reads
    `platform`, so a worker session got exactly a chat's toolbox."""
    payload, _, path = _payload(registry, notes)
    turn = _turn(f"RESULT: written\nNOTE: {path}\n", writes=path)
    monkeypatch.setattr(D, "run_prompt_in_session", turn)

    await D.execute(_item(payload))
    denied = set(turn.seen["extra_disallowed"])
    for name in WORKER_AUTOMOD_BAN:
        assert name in denied, name
    for name in ("Bash", "Read", "Write", "Edit", "Grep", "Glob", "Task",
                 "http_request", "browser_evaluate", "backlog_write_task",
                 "research_propose", "research_complete"):
        assert name in denied, name


def test_the_tools_the_job_needs_are_not_denied():
    for name in ("http_search", "http_fetch", "browser_navigate", "browser_snapshot",
                 "vault_recall", "vault_search", "vault_write", "fact_add"):
        assert name not in D.DISALLOWED, name


async def test_the_turn_runs_in_a_session_without_the_observer(
        registry, queue, notes, monkeypatch):
    """A transcript to review, without a secondary-model call per iteration."""
    payload, _, path = _payload(registry, notes)
    turn = _turn(f"RESULT: written\nNOTE: {path}\n", writes=path)
    monkeypatch.setattr(D, "run_prompt_in_session", turn)

    await D.execute(_item(payload))
    # The source no longer answers this itself. It passed `inner_voice=False`
    # as a literal until 2026-09-10, which is to say it was not a setting —
    # `workers.sources.deep-research.inner_voice` decides now, and the source
    # deferring is what makes the config real.
    assert "inner_voice" not in turn.seen
    assert C.source_inner_voice(D.NAME) is False
    assert turn.seen["source"] == D.NAME


async def test_the_prompt_gives_the_topic_the_path_and_the_date(
        registry, queue, notes, monkeypatch):
    payload, topic_id, path = _payload(registry, notes)
    turn = _turn(f"RESULT: written\nNOTE: {path}\n", writes=path)
    monkeypatch.setattr(D, "run_prompt_in_session", turn)

    await D.execute(_item(payload))
    prompt = turn.prompt
    assert f"Topic #{topic_id}" in prompt
    assert str(path) in prompt
    assert "do not look it up" in prompt, "the reason Bash is denied"
    assert "Do not read any queue" in prompt
    assert "RESULT: <written|nothing_found|duplicate>" in prompt
    assert "## Steps" in prompt or "Step 1" in prompt, "the skill body is included"


async def test_a_vault_write_outside_knowledge_is_flagged(
        registry, queue, notes, monkeypatch):
    payload, topic_id, path = _payload(registry, notes)
    # Dirty before: pre-existing scheduler churn. Dirty after: that, plus two
    # files the turn touched. Only the difference is the turn's.
    seen = iter([{"autonomy/60-x.md"},
                 {"autonomy/60-x.md", "SOUL.md", "skills/x/SKILL.md",
                  "knowledge/research/2026-09-08-a.md"}])
    monkeypatch.setattr(D, "_vault_dirty_paths", lambda: next(seen))
    monkeypatch.setattr(D, "run_prompt_in_session", _turn(
        f"RESULT: written\nNOTE: {path}\n", writes=path))

    out = await D.execute(_item(payload))
    assert out["meta"]["unexpected_vault_writes"] == ["SOUL.md", "skills/x/SKILL.md"], (
        "a note under knowledge/ is the job, and churn that predates the turn "
        "is not the turn's")
    assert registry.get(topic_id)["extra"]["unexpected_vault_writes"]


async def test_pre_existing_vault_churn_is_not_blamed_on_the_turn(
        registry, queue, notes, monkeypatch):
    """The first real research turn reported twenty "unexpected writes", every
    one of them a scheduler-rewritten task file the turn never touched. The
    vault is never clean, so the check has to be a diff, not a snapshot."""
    payload, topic_id, path = _payload(registry, notes)
    dirty = {"autonomy/60-x.md", "backlog/370-y.md", ".obsidian/workspace.json"}
    monkeypatch.setattr(D, "_vault_dirty_paths", lambda: set(dirty))
    monkeypatch.setattr(D, "run_prompt_in_session", _turn(
        f"RESULT: written\nNOTE: {path}\n", writes=path))

    out = await D.execute(_item(payload))
    assert "unexpected_vault_writes" not in out["meta"]
    assert "unexpected_vault_writes" not in registry.get(topic_id)["extra"]


def test_the_source_yields_to_the_automod_workers():
    """The queue dequeues priority ASC and a round is rarer than a note."""
    from workers.sources import autocode, autotriage
    assert autocode.DEFAULT_PRIORITY < autotriage.DEFAULT_PRIORITY
    assert autotriage.DEFAULT_PRIORITY < D.DEFAULT_PRIORITY
