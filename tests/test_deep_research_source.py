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
import inspect
import json
from pathlib import Path

import pytest

from app import research_store as R
from app.harness import finalizer as F
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
                "num_turns": kw.get("num_turns", 12), "errors": [],
                "structured": kw.get("structured"),
                "structured_error": kw.get("structured_error", "")}
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


def _real_note(tmp_path, name: str) -> Path:
    """A note the store will accept for `written`.

    Since #1276 `finish` refuses a `written` whose artifact is not a real file
    of at least `MIN_NOTE_BYTES`, so a test that settles two topics today to
    exercise the daily budget needs two files, not two made-up paths.
    """
    path = tmp_path / "registry-notes" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * 400)
    return path


def test_the_source_and_the_store_hold_the_same_note_bar():
    """Two gates on one property (`_note_is_real` here, `finish` in the store)
    are only an asset while they agree — the store's refusal would otherwise
    arrive for a note the source already certified, and the source would pre-
    check a file the registry then rejects. The store cannot import this module,
    so the number is written twice and pinned here."""
    assert D._MIN_NOTE_BYTES == R.MIN_NOTE_BYTES == 400


async def test_the_daily_budget_stops_the_source(registry, queue, notes, tmp_path):
    for i in range(4):
        row = registry.propose(f"topic number {i}")
        if i < 2:
            registry.claim(row["id"], by="w")
            registry.finish(row["id"], "written",
                            artifact_path=str(_real_note(tmp_path, f"{i}.md")))
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


async def test_the_turn_is_told_what_it_may_not_call(worker_turn_post):
    """The list above is correct policy; the turn's blindness is the defect.

    `Tool 'Bash' is disabled by configuration.` fired 22 times from 09-09 to
    09-12 — and only from the two sources that pass a list, never from a source
    that passes none — with the model in `20260912_173635_deepresearch_360b`
    retrying the *same* denied tool once more before moving on. So the refusal
    spends a call and a reasoning turn to teach the model something the code
    already knew.

    `D.execute` cannot be the subject of this assertion: every test in this file
    replaces `run_prompt_in_session` with a stub, so anything it does after
    receiving the prompt is invisible here. The real one runs, with this
    source's real `DISALLOWED`, and only its transport is swapped — see
    `worker_turn_post` in `tests/conftest.py`.
    """
    await C.run_prompt_in_session(
        "Topic #1: speculative decoding for local agent loops\n",
        title="probe", source=D.NAME, extra_disallowed=list(D.DISALLOWED))

    text = worker_turn_post[0]["text"]
    _, heading, block = text.partition(C.DENIED_TOOLS_HEADING)
    assert block, "the turn's prompt never names the deny list gating its calls"
    for name in D.DISALLOWED:
        assert name in block, f"{name} gates dispatch and is missing from the prompt"
    # The job's own tools stay out of the block. This is the half a hand-written
    # copy in the skill would have got wrong by construction: the same skill
    # (`deep-dive-research`) is retrieved interactively, where every one of
    # these is legal, so the block is only ever true of the turn that carries it.
    for name in ("http_search", "http_fetch", "vault_recall", "vault_write"):
        assert name not in block, name
    assert text.startswith("Topic #1"), "the topic was replaced, not appended to"


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


# ---------------------------------------------------------------------------
# The structured verdict (#710): the finalizer's object first, the block after
# ---------------------------------------------------------------------------


def _obj(result="written", **kw):
    base = {"result": result, "note": "", "duplicate_of": "", "facts": "3", "sources": "5"}
    base.update(kw)
    return base


def test_the_schema_admits_exactly_the_three_outcomes():
    s = D.RESULT_SCHEMA
    assert s["properties"]["result"]["enum"] == ["written", "nothing_found", "duplicate"]
    assert s["properties"]["result"]["enum"] == list(D._RESULTS)
    assert s["additionalProperties"] is False
    assert set(s["required"]) == set(s["properties"])


async def test_the_turn_is_sent_the_schema(registry, queue, notes, monkeypatch):
    payload, _, path = _payload(registry, notes)
    turn = _turn(f"RESULT: written\nNOTE: {path}\n", writes=path)
    monkeypatch.setattr(D, "run_prompt_in_session", turn)
    await D.execute(_item(payload))
    assert turn.seen["final_schema"] is D.RESULT_SCHEMA
    assert turn.seen["final_schema_prompt"]


async def test_the_kill_switch_rides_in_the_payload(registry, queue, notes, monkeypatch):
    payload, _, path = _payload(registry, notes)
    turn = _turn(f"RESULT: written\nNOTE: {path}\n", writes=path)
    monkeypatch.setattr(D, "run_prompt_in_session", turn)
    await D.execute(_item({**payload, "structured_verdict": False}))
    assert turn.seen["final_schema"] is None


async def test_the_kill_switch_is_read_at_enqueue_time(registry, queue, notes):
    """Like autotriage's: a queued item runs under the config that was live
    when it was enqueued."""
    registry.propose("a topic")
    await D.enqueue_if_due(queue, {"structured_verdict": False})
    assert queue.list_items(source=D.NAME)[0].payload["structured_verdict"] is False


async def test_a_structured_verdict_wins_and_the_regex_is_never_reached(
        registry, queue, notes, monkeypatch):
    """The text says nothing_found, the object says duplicate: the object is
    the record, and parse_result is not consulted at all."""
    payload, topic_id, _ = _payload(registry, notes)
    monkeypatch.setattr(D, "run_prompt_in_session", _turn(
        "RESULT: nothing_found\n",
        structured=_obj("duplicate", duplicate_of="knowledge/research/x.md")))

    def boom(text):
        raise AssertionError("parse_result reached with a structured verdict present")
    monkeypatch.setattr(D, "parse_result", boom)

    out = await D.execute(_item(payload))
    assert out["meta"]["result"] == "duplicate"
    assert out["meta"]["verdict_source"] == "structured"
    assert registry.get(topic_id)["extra"]["verdict_source"] == "structured"


async def test_no_object_falls_back_to_the_block_and_says_so(
        registry, queue, notes, monkeypatch):
    payload, topic_id, _ = _payload(registry, notes)
    seen = []
    real = D.parse_result
    monkeypatch.setattr(D, "parse_result", lambda text: seen.append(text) or real(text))
    monkeypatch.setattr(D, "run_prompt_in_session", _turn(
        "RESULT: nothing_found\nFACTS: 0\n", structured=None,
        structured_error="skipped: stop_reason=max_turns"))

    out = await D.execute(_item(payload))
    assert seen, "the block is the fallback when the finalizer gave nothing"
    assert out["meta"]["result"] == "nothing_found"
    assert out["meta"]["verdict_source"] == "regex"
    assert out["meta"]["structured_error"] == "skipped: stop_reason=max_turns"


async def test_an_out_of_vocabulary_object_is_not_a_verdict(registry, queue, notes, monkeypatch):
    payload, _, _ = _payload(registry, notes)
    monkeypatch.setattr(D, "run_prompt_in_session", _turn(
        "RESULT: nothing_found\n", structured=_obj("probably_written")))
    out = await D.execute(_item(payload))
    assert out["meta"]["verdict_source"] == "regex"
    assert out["meta"]["result"] == "nothing_found"


async def test_a_structured_verdict_never_takes_the_no_block_branch(
        registry, queue, notes, monkeypatch):
    """Note on disk, no RESULT in the text, an object from the finalizer: the
    object is the verdict, so this is not a "no RESULT block" guess."""
    payload, topic_id, path = _payload(registry, notes)
    monkeypatch.setattr(D, "run_prompt_in_session", _turn(
        "I have finished the research.", writes=path,
        structured=_obj("written", note=str(path))))

    out = await D.execute(_item(payload))
    assert out["status"] == "success" and out["meta"]["result"] == "written"
    assert "result_block_missing" not in registry.get(topic_id)["extra"]
    assert "no RESULT block" not in out["summary"]
    assert out["meta"]["verdict_source"] == "structured"


# ---------------------------------------------------------------------------
# The bounded grammar (#1706): the schema path was losing to the fallback, and
# the run record blamed a token budget that was never the cause
# ---------------------------------------------------------------------------

#: The first 200 characters of run_deep-research_20260928_032533_df61c0's
#: finalizer completion, recovered from that run's own `structured_error` (the
#: same literal `app/harness/tests/test_finalizer.py` pins): a valid object
#: through `"duplicate_of": ""`, then `"facts": ` followed by newlines, spaces
#: and `"}: 8,` — never closed, and it ran to the 8192-token cap.
DEGENERATE = ('{"result": "written", "note": "/home/alansrobotlab/obsidian/knowledge/'
              'research/2026-09-28-answer-option-order-sensitivity-in-constrained-'
              'llm-decision-.md", "duplicate_of": "", "facts": \n\n       "}: 8,')

_STRINGS = ("note", "duplicate_of", "facts", "sources")


def _bound(field: str) -> int:
    """The schema's cap on one string field, read off the schema itself so no
    test here restates a number the source does not carry."""
    cap = D.RESULT_SCHEMA["properties"][field].get("maxLength")
    assert isinstance(cap, int) and not isinstance(cap, bool) and cap > 0, (
        f"{field} has no finite maxLength, so the grammar is open there and a "
        "degenerate continuation can run to the completion cap again — which is "
        "the failure that cost 6 of 12 verdicts their schema path")
    return cap


def _comment_above(marker: str) -> str:
    """The `#:` block immediately above `marker` in the source."""
    src = Path(inspect.getsourcefile(D)).read_text(encoding="utf-8")
    lines = src[:src.index(marker)].rstrip("\n").split("\n")
    block = []
    while lines and lines[-1].lstrip().startswith("#"):
        block.append(lines.pop())
    return "\n".join(reversed(block))


def test_every_string_field_of_the_schema_is_bounded():
    """The hole was `facts`: with an open string the guided decoder will accept
    any continuation, and six completions took it all the way to 8192 tokens.
    `result` needs no cap because its enum is one."""
    from app.harness import finalizer as F

    for field in _STRINGS:
        cap = _bound(field)
        assert cap <= 200, f"{field} is capped at {cap}, which is not a bound"
    for count in ("facts", "sources"):
        assert _bound(count) <= 16, f"{count} is a count, not an essay"

    s = D.RESULT_SCHEMA
    assert s["properties"]["result"]["enum"] == list(D._RESULTS)
    assert s["additionalProperties"] is False
    assert set(s["required"]) == set(s["properties"])
    assert F._schema_is_bounded(s), (
        "the finalizer still reads this schema as open-ended, so a truncated "
        "completion there would be blamed on harness.finalizer.max_tokens")


def test_the_note_bound_is_a_vault_note_path_with_room():
    """200 is a measured number, not a round one: `_slug` caps the filename at
    60 and the tree is fixed, so the longest path this source can name is
    ~122 characters (and exactly that is the longest in `runs.artifact_path`).
    A cap below the longest producible path would trade a loud truncation for a
    silent one, so the worst case has to stay inside it."""
    worst = str(D.note_path_for("q" * 400))
    assert len(worst) <= _bound("note"), (
        f"the worst-case note path is {len(worst)} chars and the schema cap "
        "would cut it — the bound is the bug now")
    assert worst.endswith(".md")


def test_the_schema_comment_names_the_bound_and_what_it_makes_impossible():
    """The comment used to read "No `maxLength`: the guided decoder would stop
    mid-sentence at it", which is true of the decoder and was the wrong trade:
    stopping at the cap is the point when the alternative is 8192 tokens of
    whitespace. The replacement has to say both halves or the next reader
    reverts it."""
    comment = _comment_above("RESULT_SCHEMA: dict")
    assert comment, "no comment above RESULT_SCHEMA to grade"
    assert "No `maxLength`" not in comment, (
        "the comment still bans a bound; #1706 put one on every string field")
    assert "maxLength" in comment and "8192" in comment, (
        "the comment must name the cap and the runaway it prevents")


def test_an_object_sitting_at_every_bound_is_still_a_structured_verdict():
    """A cap the model cannot actually reach is decoration; a cap that rejects
    a legitimate verdict is a new bug. `_from_structured` collapses whitespace
    and strips quotes and backticks — none of which may eat a field that is
    exactly at its cap."""
    at = {"result": "written",
          "note": "/" + ("n" * (_bound("note") - 1)),
          "duplicate_of": "d" * _bound("duplicate_of"),
          "facts": "9" * _bound("facts"),
          "sources": "8" * _bound("sources")}
    assert len(at["note"]) == _bound("note")
    assert len(at["facts"]) == _bound("facts") == _bound("sources")

    parsed = D.parse_verdict("RESULT: nothing_found\n", at)
    assert parsed is not None and parsed["source"] == "structured", (
        "a verdict at the bounds fell out of the schema path")
    assert parsed["result"] == "written"
    assert parsed["note"] == at["note"] and parsed["duplicate_of"] == at["duplicate_of"]
    assert parsed["facts"] == at["facts"] and parsed["sources"] == at["sources"]


class _EngineResp:
    """One OpenAI-shaped completion response."""

    def __init__(self, content, finish_reason, completion_tokens):
        self.status_code = 200
        self.text = ""
        self._content = content
        self._finish = finish_reason
        self._tokens = completion_tokens

    def json(self):
        return {"choices": [{"message": {"content": self._content},
                             "finish_reason": self._finish}],
                "usage": {"completion_tokens": self._tokens}}


def _fake_engine(content, *, finish_reason="stop", completion_tokens=180):
    """The engine, minus the network: whatever the finalizer asks, it answers
    with `content`. `app/harness/tests/test_finalizer.py` fakes it the same way."""
    resp = _EngineResp(content, finish_reason, completion_tokens)

    class _Cli:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, headers=None, json=None):
            return resp

    return _Cli


def _turn_through_the_real_finalizer(text, *, writes):
    """A stand-in for the loop that keeps the part that matters.

    `_turn` above hands `structured` to the source directly, which is how every
    test in this file drove the schema path — and so none of them ever ran
    `app.harness.finalizer` against `RESULT_SCHEMA`, which is exactly why 7 of
    the 12 fallbacks to the text parser went unnoticed while this file was
    green. Here the turn's `structured` is whatever the real finalizer returns
    for the schema THIS source passes as `final_schema`. Only the HTTP leg is
    faked.
    """
    async def run(prompt, **kwargs):
        run.seen = kwargs
        run.prompt = prompt
        if writes is not None:
            writes.write_text("# A note\n\n" + "body " * 200, encoding="utf-8")
        structured, error, _usage = await F.run_finalizer(
            base_url="http://engine:8096", model="primary",
            chat_messages=[{"role": "user", "content": prompt}], tools=None,
            schema=kwargs.get("final_schema") or D.RESULT_SCHEMA)
        return {"text": text, "session_id": "20260928_deep_x",
                "stop_reason": "stop", "num_turns": 19, "errors": [],
                "structured": structured, "structured_error": error}
    run.seen = {}
    run.prompt = ""
    return run


async def test_a_run_whose_finalizer_answers_with_the_object_records_structured(
        registry, queue, notes, monkeypatch):
    """The verdict the engine sent back as a valid object has to reach that
    run's `meta_json` as `verdict_source: structured` — the field production
    reads, and the one nothing here used to pin end to end."""
    payload, topic_id, path = _payload(registry, notes)
    monkeypatch.setattr(F.httpx, "AsyncClient", _fake_engine(json.dumps(
        {"result": "written", "note": str(path), "duplicate_of": "",
         "facts": "3", "sources": "5"})))
    monkeypatch.setattr(D, "run_prompt_in_session", _turn_through_the_real_finalizer(
        f"The research is done.\nRESULT: written\nNOTE: {path}\n", writes=path))

    out = await D.execute(_item(payload))
    assert out["meta"]["verdict_source"] == "structured", (
        "the schema path answered and the run does not say so")
    assert registry.get(topic_id)["extra"]["verdict_source"] == "structured"
    assert out["meta"]["structured_error"] == ""
    assert out["meta"]["result"] == "written"


async def test_the_degenerate_completion_still_yields_a_verdict_from_the_block(
        registry, queue, notes, monkeypatch):
    """The captured failure, replayed: the finalizer returns junk, so the
    fallback must still answer, the run must record which path did, and the
    reason it stores must not send the next reader to a budget that is already
    8192 and was never the problem."""
    payload, topic_id, path = _payload(registry, notes)
    monkeypatch.setattr(F.httpx, "AsyncClient", _fake_engine(
        DEGENERATE, finish_reason="length", completion_tokens=8192))
    monkeypatch.setattr(D, "run_prompt_in_session", _turn_through_the_real_finalizer(
        f"The research is done.\nRESULT: written\nNOTE: {path}\n", writes=path))

    out = await D.execute(_item(payload))
    assert out["status"] == "success" and out["meta"]["result"] == "written"
    assert out["meta"]["verdict_source"] == "regex", (
        "the block is the fallback, and the run has to say it was used")
    stored = out["meta"]["structured_error"]
    assert stored, "the reason the schema path failed is not in the meta"
    assert "diverged" in stored, stored
    assert "raise harness.finalizer.max_tokens" not in stored, stored
    assert registry.get(topic_id)["extra"]["structured_error"] == stored


# ───────────────────────── FACTS: reconciled against the store (#1709) ──────
# The turn holds `fact_add` on purpose (`vault_write` and `fact_add` stay: they
# are the job, `DISALLOWED` names neither) and it reads attacker-chosen text, so
# "and record this as a durable fact" reaches the fact store with a real
# finding's authority. The vault half of that has a post-turn control —
# `_unexpected_vault_writes` diffs the vault against a baseline — and the facts
# half had none: the only number anywhere was `FACTS:`, the turn grading its own
# homework. These nodes drive the real `agent_mcp.facts._fact_add` into a real
# store retargeted by `kg_store.configure`, so the reconciled quantity is what
# the store holds, never a call the test staged.

from agent_mcp import facts as FACTS                    # noqa: E402
from agent_mcp._task_registry import current_session_id  # noqa: E402
from app import kg_store                                # noqa: E402
from workers.sources import _common as COMMON           # noqa: E402
import workers.sources.youtube_digest as Y              # noqa: E402

#: The session id the fake turn reports and writes its facts under — the same
#: value the source stores in `extra["session_id"]`.
FACTS_SESSION = "20260928_deep_facts"


@pytest.fixture
def fact_store(tmp_path, monkeypatch):
    """A real fact store and index, both under `tmp_path`.

    `agent_mcp.facts` and `app.retrieval` read `FACTS_ROOT` as a module global,
    the same retarget `tests/test_fact_duplicate_guard.py` uses — a stub would
    only prove the source believes whatever the fake returns.
    """
    from agent_mcp import _shared, retrieval
    root = tmp_path / "facts"
    root.mkdir()
    monkeypatch.setattr(_shared, "FACTS_ROOT", root)
    monkeypatch.setattr(FACTS, "FACTS_ROOT", root)
    monkeypatch.setattr(retrieval, "FACTS_ROOT", root)
    kg_store.configure(tmp_path / "kg.sqlite")
    return root


def _turn_writing_facts(text: str, *, writes: Path, add: list[str],
                        refusals: int = 0, session: str = FACTS_SESSION,
                        structured: dict | None = None,
                        write_gate: str = ""):
    """A turn that writes facts through the real tool, under its own session id.

    `refusals` is how many further calls the store refuses as restatements:
    `_fact_add` answers those `{success: True, skipped: True, duplicate: True}`
    having written nothing, which is why the reconciled quantity is facts
    *written* and not calls made — the triage's own session
    `20260927_185948_deepresearch_4e62.json` carries 14 `fact_add` records
    against `FACTS: 12`, and a call-count detector would flag that honest turn.

    `structured` is the finalizer's object for the same turn, so a node can set
    the block and the object against each other (#1773).

    `write_gate="off"` is for a node writing enough findings that the #1487
    paraphrase gate would judge each one against the file it is filling: a
    NOOP on any single fact would move the store count the node fixes, and that
    verdict is a different feature's test. The verbatim refusal above is
    untouched by it — it lands before the gate.
    """
    calls = {"n": 0}

    def _add(entity: str, category: str, body: str) -> dict:
        calls["n"] += 1
        params = {"entity": entity, "category": category,
                  "fact": body, "confidence": 0.9}
        if write_gate:
            params["write_gate"] = write_gate
        return FACTS._fact_add(params)

    async def run(prompt, **kwargs):
        run.seen = kwargs
        if writes is not None:
            writes.write_text("# A note\n\n" + "body " * 200, encoding="utf-8")
        token = current_session_id.set(session)
        try:
            for i, body in enumerate(add):
                _add("Research", "state", f"{body} (finding {i + 1})")
            for i in range(refusals):
                _add("Research", "state", f"{add[0]} (finding 1)")
        finally:
            current_session_id.reset(token)
        return {"text": text, "session_id": session, "stop_reason": "stop",
                "num_turns": 12, "errors": [], "structured": structured,
                "structured_error": ""}
    run.seen = {}
    run.calls = calls
    return run


def _block(path: Path, facts_claimed: str) -> str:
    return (f"Research done.\nRESULT: written\nNOTE: {path}\n"
            f"FACTS: {facts_claimed}\nSOURCES: 4\n")


async def test_a_claim_that_disagrees_with_the_store_is_recorded_on_the_topic(
        registry, queue, notes, fact_store, monkeypatch):
    """Clause 1: a turn claiming 0 facts that wrote 2 must be visible without
    reading its prose."""
    payload, topic_id, path = _payload(registry, notes)
    turn = _turn_writing_facts(_block(path, "0"), writes=path,
                              add=["vLLM pins the grammar per request",
                                   "an open string field has no ceiling"])
    monkeypatch.setattr(D, "run_prompt_in_session", turn)

    out = await D.execute(_item(payload))
    assert out["status"] == "success"
    assert turn.calls["n"] == 2, "the turn must write through the real tool"
    assert out["meta"]["facts_written"] == 2
    assert out["meta"]["facts_claimed"] == 0
    assert out["meta"]["facts_mismatch"] == {"written": 2, "claimed": 0}
    stored = registry.get(topic_id)["extra"]
    assert stored["facts_mismatch"] == {"written": 2, "claimed": 0}, (
        "the finish record is where the next reader looks")


async def test_an_honest_self_report_sets_no_flag(registry, queue, notes,
                                                 fact_store, monkeypatch):
    """Clause 2: agreement is silence — a detector that fires on every run is
    one nobody reads."""
    payload, topic_id, path = _payload(registry, notes)
    turn = _turn_writing_facts(_block(path, "2"), writes=path,
                              add=["one finding", "another finding"])
    monkeypatch.setattr(D, "run_prompt_in_session", turn)

    out = await D.execute(_item(payload))
    assert turn.calls["n"] == 2
    assert out["meta"]["facts_written"] == 2 and out["meta"]["facts_claimed"] == 2
    assert "facts_mismatch" not in out["meta"], (
        f"agreement flagged: {out['meta']['facts_mismatch']}")
    assert "facts_mismatch" not in registry.get(topic_id)["extra"]


async def test_a_refused_duplicate_is_not_a_fact_written(
        registry, queue, notes, fact_store, monkeypatch):
    """Clause 3: three calls, two facts. The refusal wrote nothing, so a turn
    that reports 2 is honest and must not be flagged."""
    payload, topic_id, path = _payload(registry, notes)
    turn = _turn_writing_facts(_block(path, "2"), writes=path,
                              add=["first finding", "second finding"], refusals=1)
    monkeypatch.setattr(D, "run_prompt_in_session", turn)

    out = await D.execute(_item(payload))
    assert turn.calls["n"] == 3, (
        "the third call must reach the store: it is the refusal being counted")
    assert out["meta"]["facts_written"] == 2
    assert out["meta"]["facts_claimed"] == 2
    assert "facts_mismatch" not in out["meta"], (
        "a self-report that leaves refusals out was flagged as a lie")


async def test_recording_a_mismatch_changes_no_fact_and_no_outcome(
        registry, queue, notes, fact_store, monkeypatch):
    """Clause 4: detector, not revert — the posture `_unexpected_vault_writes`
    keeps. Undoing a write would be worse than reporting it."""
    payload, topic_id, path = _payload(registry, notes)
    turn = _turn_writing_facts(_block(path, "0"), writes=path,
                              add=["a fact from the page", "another one"])
    monkeypatch.setattr(D, "run_prompt_in_session", turn)

    out = await D.execute(_item(payload))
    assert out["meta"]["facts_mismatch"], "the setup must produce a mismatch"
    assert out["status"] == "success" and registry.get(topic_id)["status"] == "written", (
        "a mismatch must not change the topic's outcome")
    rows = kg_store.store().facts_idx.for_session(FACTS_SESSION)
    assert len(rows) == 2, "the facts the turn wrote must still be there"
    for row in rows:
        assert not row["expired_at"] and not row["invalid_at"], (
            f"the detector expired a fact: {row['fact'][:40]}")


def test_the_reconciler_is_shared_and_silent_when_it_cannot_measure():
    """The helper guards every source that keeps `fact_add`, and abstains
    rather than crying wolf: `youtube_digest` holds the tool too (its
    `DISALLOWED` names only the browser and registry writers) and its RESULT
    block claims no fact count at all."""
    assert Y.reconcile_fact_writes is COMMON.reconcile_fact_writes
    assert D.reconcile_fact_writes is COMMON.reconcile_fact_writes
    assert COMMON.reconcile_fact_writes("", "3") == {}, (
        "with no session there is nothing to attribute; inventing a zero-count "
        "mismatch on every legacy turn would bury the real ones")


# ────── A garbled structured count falls back to the block (#1773) ───────────
# The finalizer transcribes the turn's RESULT block into a JSON object under
# guided decode, and the transcription has been measured losing the numbers:
# live topics 14/15/17/19/20 carry `facts=]}`, `facts=sources`,
# `facts=result` and an absent `facts` in the field that should hold a count,
# while the transcripts of topics 19 and 20 state `FACTS: 12` and `FACTS: 15`.
# What that costs is not the finish note — it is #1709's reconciliation:
# `_common` coerces the claim with `int()`, and a claim that will not coerce
# sets `facts_claimed` to null and flags nothing, so 2 of 2 post-#1709 runs
# were blind, including one where claim and store agreed. The upstream cause is
# the decode itself (out of scope here); what these nodes pin is that a lost
# number is rescued from the block it was transcribed from, field by field, and
# that the token it replaced is kept, so `facts_claimed: null` can never again
# read as "the turn claimed nothing" when it means "the object lost it".

#: Twelve distinct findings: a store holding twelve for one session has to be
#: twelve facts, not one fact written twelve times.
_TWELVE_FINDINGS = [
    "speculative decoding needs a draft model sharing the target's vocabulary",
    "n-gram drafting reaches the same win without a second model",
    "the acceptance rate, not the draft length, sets the speedup ceiling",
    "a rejected draft token still costs a verification step",
    "large batch sizes dilute the win because verification is memory bound",
    "KV-cache sharing between draft and target is optional, not required",
    "Medusa heads are a fine-tune, not an extra forward pass per token",
    "the drafter's context window has to match the target's",
    "speculative sampling leaves the target distribution unchanged",
    "tree attention verifies several drafts in one pass",
    "an over-long draft is truncated rather than penalised",
    "the win shrinks on a model already saturating GPU memory",
]


async def test_a_junk_structured_fact_count_is_recovered_from_the_block(
        registry, queue, notes, fact_store, monkeypatch):
    """Clause 1, replaying topic 19: the object's count is the word `facts`,
    the turn's own block says twelve, and the store holds twelve for that
    session. The recorded claim is twelve, nothing is flagged, and the finish
    note reads `facts=12` instead of the garbled token."""
    payload, topic_id, path = _payload(registry, notes)
    turn = _turn_writing_facts(
        _block(path, "12"), writes=path, add=_TWELVE_FINDINGS,
        structured=_obj(facts="facts", sources="sources"), write_gate="off")
    monkeypatch.setattr(D, "run_prompt_in_session", turn)

    out = await D.execute(_item(payload))
    assert out["status"] == "success"
    assert turn.calls["n"] == 12, "the store has to hold the twelve being claimed"
    assert out["meta"]["facts_written"] == 12
    assert out["meta"]["facts_claimed"] == 12, (
        "the block's FACTS: 12 is the claim when the object's is a key name")
    assert out["meta"]["facts_claimed_raw"] == "facts"
    assert "facts_mismatch" not in out["meta"], (
        f"a recovered claim that agrees with the store was flagged: "
        f"{out['meta'].get('facts_mismatch')}")
    row = registry.get(topic_id)
    assert row["extra"]["facts_claimed"] == 12
    assert row["outcome_note"].startswith("facts=12 sources=4; session "), (
        f"the note still carries the garbled token: {row['outcome_note']}")


async def test_a_recovered_count_that_understates_the_store_is_flagged(
        registry, queue, notes, fact_store, monkeypatch):
    """Clause 2: the same garbled object, but the block claims three over a
    store holding five. Recovering the count is what makes the mismatch check
    answer again — before #1773 this shape stored `facts_claimed: null` and
    raised no flag at all, which is how a live run wrote fifteen facts and
    looked reconciled."""
    payload, topic_id, path = _payload(registry, notes)
    turn = _turn_writing_facts(
        _block(path, "3"), writes=path, add=_TWELVE_FINDINGS[:5],
        structured=_obj(facts="facts", sources="sources"), write_gate="off")
    monkeypatch.setattr(D, "run_prompt_in_session", turn)

    out = await D.execute(_item(payload))
    assert out["meta"]["facts_written"] == 5
    assert out["meta"]["facts_claimed"] == 3
    assert out["meta"]["facts_mismatch"] == {"written": 5, "claimed": 3}
    assert registry.get(topic_id)["extra"]["facts_mismatch"] == {
        "written": 5, "claimed": 3}, "the finish record is where the flag lives"


async def test_the_claim_is_null_only_when_neither_path_gives_a_number(
        registry, queue, notes, fact_store, monkeypatch):
    """Clause 3: `facts` in the object and `FACTS: facts` in the block, so no
    path yields an integer. `facts_claimed` is null with the object's token
    beside it, and an unusable claim still sets no mismatch — a detector that
    fires because it could not read would be one nobody reads. The second half
    is the other side of "only": a claim the object got right is left alone and
    carries no raw token."""
    payload, topic_id, path = _payload(registry, notes)
    turn = _turn_writing_facts(
        _block(path, "facts"), writes=path, add=_TWELVE_FINDINGS[:2],
        structured=_obj(facts="facts", sources="sources"), write_gate="off")
    monkeypatch.setattr(D, "run_prompt_in_session", turn)

    out = await D.execute(_item(payload))
    assert out["meta"]["facts_written"] == 2
    assert out["meta"]["facts_claimed"] is None
    assert out["meta"]["facts_claimed_raw"] == "facts", (
        "a null has to say the object was unreadable, not that the turn "
        "claimed nothing")
    assert "facts_mismatch" not in out["meta"]
    assert registry.get(topic_id)["extra"]["facts_claimed_raw"] == "facts"

    payload2, _, path2 = _payload(registry, notes, topic="Draft-verification cost")
    turn2 = _turn_writing_facts(
        _block(path2, "2"), writes=path2, add=_TWELVE_FINDINGS[6:8],
        session="20260928_deep_facts_clean",
        structured=_obj(facts="2"), write_gate="off")
    monkeypatch.setattr(D, "run_prompt_in_session", turn2)
    out2 = await D.execute(_item(payload2))
    assert out2["meta"]["facts_written"] == 2 and out2["meta"]["facts_claimed"] == 2
    assert "facts_claimed_raw" not in out2["meta"], (
        "the raw key means 'the object held no number', so a clean claim "
        "must not carry it")


async def test_only_the_counts_fall_back_and_the_object_still_decides_the_rest(
        registry, queue, notes, fact_store, monkeypatch):
    """Clause 4, first half: the object says `duplicate` with a real
    `duplicate_of`, the block says `written` with a usable count, and the two
    disagree. The outcome, the note and the duplicate pointer are the object's
    — the block is consulted for the count and nothing else, and
    `verdict_source` still says `structured`."""
    payload, topic_id, path = _payload(registry, notes)
    turn = _turn_writing_facts(
        _block(path, "2"), writes=path, add=_TWELVE_FINDINGS[2:4],
        structured=_obj("duplicate", note="duplicate of facts",
                        duplicate_of="knowledge/research/2026-09-20-earlier.md",
                        facts="facts", sources="sources"), write_gate="off")
    monkeypatch.setattr(D, "run_prompt_in_session", turn)

    out = await D.execute(_item(payload))
    assert out["meta"]["result"] == "duplicate", (
        "the block's RESULT line is not a vote on the outcome")
    assert out["meta"]["verdict_source"] == "structured"
    assert out["meta"]["facts_claimed"] == 2, (
        "the count is the one field the block does get to answer")
    assert out["meta"]["facts_claimed_raw"] == "facts"
    row = registry.get(topic_id)
    assert row["status"] == "duplicate"
    assert row["outcome_note"] == (
        "duplicate of knowledge/research/2026-09-20-earlier.md; "
        f"facts=2 sources=4; session {FACTS_SESSION}")


async def test_a_junk_count_with_no_result_block_stays_null_rather_than_raising(
        registry, queue, notes, fact_store, monkeypatch):
    """Clause 4, second half: the finalizer answered with a garbled count and
    the turn left no block to rescue it from. `parse_result` returns None for
    that, the run still settles, and the claim is a null rather than an
    exception out of the reconciler."""
    payload, topic_id, path = _payload(registry, notes)
    turn = _turn_writing_facts(
        "The research is written up; the note is at the path above.",
        writes=path, add=_TWELVE_FINDINGS[4:6],
        structured=_obj("written", facts="facts", sources="sources"),
        write_gate="off")
    monkeypatch.setattr(D, "run_prompt_in_session", turn)

    out = await D.execute(_item(payload))
    assert out["status"] == "success" and out["meta"]["result"] == "written"
    assert out["meta"]["verdict_source"] == "structured"
    assert out["meta"]["facts_written"] == 2
    assert out["meta"]["facts_claimed"] is None
    assert out["meta"]["facts_claimed_raw"] == "facts"
    assert registry.get(topic_id)["status"] == "written"


async def test_a_recovered_claim_still_writes_nothing_and_changes_no_outcome(
        registry, queue, notes, fact_store, monkeypatch):
    """Clause 5: with the recovered claim at three against a store holding
    five the detector is awake again, and it stays a detector. `facts_written`
    is the store's own `count_by_session` and not the claim; the markdown fact
    file and the index still hold exactly the five facts the turn wrote, none
    of them expired or invalidated and no row added by the check; and the topic
    settles as `written` with the flag recorded beside it."""
    payload, topic_id, path = _payload(registry, notes)
    turn = _turn_writing_facts(
        _block(path, "3"), writes=path, add=_TWELVE_FINDINGS[7:12],
        structured=_obj(facts="facts", sources="sources"), write_gate="off")
    monkeypatch.setattr(D, "run_prompt_in_session", turn)

    out = await D.execute(_item(payload))
    assert out["meta"]["facts_mismatch"] == {"written": 5, "claimed": 3}
    assert out["meta"]["facts_written"] == 5, (
        "the store's count, never the turn's claim")
    idx = kg_store.store().facts_idx
    assert idx.count_by_session(FACTS_SESSION) == 5
    assert idx.count() == 5, "the detector wrote a row into the fact store"
    rows = idx.for_session(FACTS_SESSION)
    assert len(rows) == 5
    for row in rows:
        assert not row["expired_at"] and not row["invalid_at"], (
            f"the detector expired a fact: {row['fact'][:40]}")
    files = sorted(fact_store.rglob("*.md"))
    assert len(files) == 1 and "**Fact Count:** 5" in (
        files[0].read_text(encoding="utf-8")), "the markdown store moved"
    assert out["status"] == "success" and out["meta"]["result"] == "written"
    assert registry.get(topic_id)["status"] == "written", (
        "a mismatch must not change the topic's outcome")


def test_only_the_junk_count_falls_back_and_a_clean_object_never_reads_the_block():
    """Per field, and nothing more. With both counts usable the structured
    path does not look at the block at all (the end-to-end pin is
    `test_a_structured_verdict_wins_and_the_regex_is_never_reached`); with one
    unusable, only that one is taken from the block and the object's own number
    for the other field is left standing."""
    turn_text = ("done.\nRESULT: written\nNOTE: /x/a.md\nFACTS: 12\nSOURCES: 4\n")
    parsed = D.parse_verdict(turn_text, _obj(facts="facts"))
    assert parsed["facts"] == "12" and parsed["sources"] == "5"
    assert parsed["facts_raw"] == "facts" and "sources_raw" not in parsed
    assert parsed["result"] == "written" and parsed["source"] == "structured"

    clean = D.parse_verdict(turn_text, _obj())
    assert clean["facts"] == "3" and clean["sources"] == "5"
    assert "facts_raw" not in clean and "sources_raw" not in clean


# ---------------------------------------------------------------------------
# #1875: a non-path-shaped duplicate_of is treated as absent at both use sites
# ---------------------------------------------------------------------------
# The finalizer's guided decode emits key names and punctuation as VALUES when it
# runs out of real text. #1773 rescued the COUNTS from that (`_recover_counts`) and
# deliberately left `duplicate_of` as the finalizer said it, so four live topics
# (#14/#15/#19/#23, all `status=written`) shipped an outcome note reading
# `duplicate of null` / `duplicate of facts”:14,` / `duplicate of facts`, and a
# truthy garbage token would have suppressed the downgrade that exists for a
# duplicate with no usable pointer. Topic 23 is the one with receipts:
# `duplicate of facts; facts=11 sources=8; session 20260929_190734_deepresearch_a3d5`
# — the same row where #1773's rescue fixed the count and left the pointer garbled.


async def test_a_written_topic_with_a_garbage_duplicate_pointer_keeps_a_clean_note(
        registry, queue, notes, monkeypatch):
    """Clause 1: topic 23's shape settles with no `duplicate of` prefix at all.

    `result=written` with `duplicate_of="facts"` — a token, not a pointer — must
    produce the note the counts alone would have produced. The bug did not make the
    topic's status wrong; it made the note say something false about a document that
    does not exist, and the registry is where a human reads the outcome.
    """
    payload, topic_id, path = _payload(registry, notes)
    monkeypatch.setattr(D, "run_prompt_in_session", _turn(
        f"RESULT: written\nNOTE: {path}\nFACTS: 11\nSOURCES: 8\n", writes=path,
        structured=_obj("written", duplicate_of="facts",
                        facts="11", sources="8")))

    out = await D.execute(_item(payload))
    assert out["status"] == "success" and out["meta"]["result"] == "written"
    row = registry.get(topic_id)
    assert row["status"] == "written"
    assert row["outcome_note"] == "facts=11 sources=8; session 20260908_deep_x", (
        "the note is the counts and the session, with nothing prepended")
    assert "duplicate of" not in row["outcome_note"], (
        "a garbage token reached the outcome note again — topic 23's shipped string")


async def test_a_duplicate_with_an_unparseable_pointer_downgrades(registry, queue,
                                                                  notes, monkeypatch):
    """Clause 2: `result=duplicate` + `duplicate_of="],}"` is `nothing_found`.

    The downgrade at `deep_research.py` exists for "we already know this" with no
    evidence, and its own test (`test_an_unsupported_duplicate_is_downgraded`) covers
    the EMPTY pointer. A garbage token is the same evidentiary state wearing a truthy
    value, and before this change it skipped the branch entirely — the topic then
    shipped as a duplicate pointing at nothing, which is the outcome the branch was
    written to prevent.
    """
    payload, topic_id, _path = _payload(registry, notes)
    monkeypatch.setattr(D, "run_prompt_in_session", _turn(
        "RESULT: duplicate\nDUPLICATE_OF: ],}\n",
        structured=_obj("duplicate", duplicate_of="],}")))

    out = await D.execute(_item(payload))
    assert out["meta"]["result"] == "nothing_found", (
        "a token that names no document is not evidence of a duplicate")
    row = registry.get(topic_id)
    assert row["status"] == "nothing_found"
    assert row["extra"]["downgraded_from"] == "duplicate"


async def test_the_rejected_pointer_token_is_kept_verbatim_and_the_object_still_wins(
        registry, queue, notes, monkeypatch):
    """Clause 3: rejection is not deletion, and the verdict stays the object's.

    Three things at once, because they are three ways the same fix could quietly
    lose information: the token survives as `duplicate_of_raw` (it names which decode
    field leaked — the clue the next reader needs, and it is gone if the value is
    dropped); `verdict_source` stays `structured`; and the settled result still comes
    from the object, which here DISAGREES with the text block. The block says
    `nothing_found`; the object says `written`; the row says `written`.
    """
    payload, topic_id, path = _payload(registry, notes)
    monkeypatch.setattr(D, "run_prompt_in_session", _turn(
        f"RESULT: nothing_found\nNOTE: {path}\n", writes=path,
        structured=_obj("written", duplicate_of="facts", facts="4", sources="2")))

    out = await D.execute(_item(payload))
    row = registry.get(topic_id)
    assert row["status"] == "written", (
        "the text block decided the outcome, not the structured object")
    assert row["extra"]["verdict_source"] == "structured"
    assert row["extra"]["duplicate_of_raw"] == "facts", (
        "the rejected token has to be readable by whoever comes next")
    assert "duplicate_of_raw" in out["meta"], "the same token on the worker's own meta"


async def test_a_path_shaped_pointer_still_prefixes_the_note_and_still_counts(
        registry, queue, notes, monkeypatch):
    """Clause 4: the legitimate half is untouched, on BOTH paths.

    `test_only_the_counts_fall_back_and_the_object_still_decides_the_rest` already
    pins the byte-exact prefixed note (`outcome_note ==
    "duplicate of knowledge/research/2026-09-20-earlier.md; facts=2 sources=4; …"`,
    green unchanged by this diff); this node pins the other half, which no existing
    test covers: a `duplicate` result carrying a path-shaped pointer must NOT be
    downgraded. Before the change that held by accident — any
    truthy value skipped the branch; after it, it holds because the shape test says
    the pointer is usable, which is the distinction the whole item turns on.
    """
    payload, topic_id, _path = _payload(registry, notes)
    pointer = "knowledge/research/2026-09-26-discrete-event-simulation.md"
    monkeypatch.setattr(D, "run_prompt_in_session", _turn(
        f"RESULT: duplicate\nDUPLICATE_OF: {pointer}\n",
        structured=_obj("duplicate", duplicate_of=pointer, facts="1", sources="6")))

    out = await D.execute(_item(payload))
    assert out["meta"]["result"] == "duplicate", (
        "a real pointer is exactly what the downgrade asks for and must survive")
    row = registry.get(topic_id)
    assert row["status"] == "duplicate"
    assert "downgraded_from" not in row["extra"]
    assert row["outcome_note"] == (
        f"duplicate of {pointer}; facts=1 sources=6; session 20260908_deep_x")
    assert "duplicate_of_raw" not in row["extra"], (
        "an accepted pointer is not a rejected one; recording it as raw would say "
        "the shape test ran against a value it accepted")


async def test_the_shape_test_rejects_a_token_nobody_has_seen(registry, queue,
                                                             notes, monkeypatch):
    """Clause 5: the gate is a shape rule, so an unseen token fails the same way.

    `facts”:14,` is topic 15's shipped corruption and appears NOWHERE in this diff —
    not in the predicate, not in a test list of bad values. So does `null` (topic 14's)
    and a bare registry id, which the prompt's wording does invite and which stays
    rejected by ruling (recorded on `_pointer_shaped`'s docstring, #1980). If the
    gate were an enumeration, this node would pass for the tokens it names and stay
    blind to the next key the decoder renames — which is how `duplicate_of` became the
    live corruption surface after #1773 closed the counts.

    Asserted first end to end — an unseen token settling a topic with an unprefixed
    note — and then as grammar, over the shipped predicate in both directions: the
    four live corruptions plus values nobody has observed are rejected, and the one
    shape that names a document is accepted. That pair is the whole rule, so neither
    half can drift alone.
    """
    # End to end first: a token this diff never mentions in its own prose has to
    # settle the topic with an unprefixed note. The node fails on behaviour, not
    # on a symbol it happens to be missing.
    payload, topic_id, path = _payload(registry, notes)
    monkeypatch.setattr(D, "run_prompt_in_session", _turn(
        f"RESULT: written\nNOTE: {path}\nFACTS: 6\nSOURCES: 3\n", writes=path,
        structured=_obj("written", duplicate_of="null", facts="6", sources="3")))
    out = await D.execute(_item(payload))
    row = registry.get(topic_id)
    assert out["meta"]["result"] == "written"
    assert row["outcome_note"] == "facts=6 sources=3; session 20260908_deep_x"
    assert row["extra"]["duplicate_of_raw"] == "null"

    # Then the grammar itself, both directions, over the shipped predicate.
    rejected = ["facts", "null", "facts”:14,", "],}", "14", "topic 14",
                "sources", "note", "knowledge", "true", "."]
    accepted = ["knowledge/research/a.md", "a.md", "/x/y",
                "knowledge/research/2026-09-26-discrete-event-simulation.md"]
    assert all(not D._pointer_shaped(v) for v in rejected), (
        [v for v in rejected if D._pointer_shaped(v)])
    assert all(D._pointer_shaped(v) for v in accepted), (
        [v for v in accepted if not D._pointer_shaped(v)])


def test_the_pointer_grammar_ruling_is_recorded_as_settled_not_deferred():
    """#1980: the docstring went on deferring the grammar question to #1875's owed
    list after the ruling was made, so the next reader reopened it."""
    doc = " ".join(D._pointer_shaped.__doc__.split())
    assert "widening the grammar is a ruling" not in doc
    assert "owed " + "entry 3" not in doc
    assert "ever been observed" not in doc, "topic #27 did carry a bare id"
    # The ruling: the whole grammar, and what is rejected by decision.
    assert '`"/" in value or value.endswith(".md")` is the whole predicate' in doc
    assert "rejected by decision" in doc and "bare registry id" in doc
    assert "bare directory" in doc and "no `.md`" in doc
    # The measured case and why it refuses.
    assert 'duplicate_of_raw="15"' in doc and "duplicate of 15" in doc
    assert "false pointer" in doc
    # The standing reopen trigger.
    assert "reopens widening" in doc and "result=duplicate" in doc
    # No surface still defers the question.
    for src in (inspect.getsource(D), Path(__file__).read_text(encoding="utf-8")):
        assert "owed " + "entry 3" not in src
        assert "only if someone " + "rules on it" not in src
    # Prose only: the predicate is what it was.
    assert inspect.getsource(D._pointer_shaped).rstrip().endswith(
        'return "/" in value or value.endswith(".md")')

