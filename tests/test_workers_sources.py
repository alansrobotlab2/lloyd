"""Worker sources — what a turn producing nothing means, and what gets mined.

Three failures this pins, all of which ran in production for weeks while
every run record said `success`:

  * A turn that produced no text had its empty output written to the vault as
    a research note and its input retired. 225 of the 498 notes under
    `pending-research/` have the body `(no response)`.
  * `session-distill` re-mined live chats on every tick — 44 runs against one
    session — and mined Lloyd's own worker transcripts back in as observations
    about the user.
  * A worker turn read `config.yaml` off disk, so it never saw the override
    file that decides which tools are actually switched off.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from pathlib import Path

import pytest

from workers.queue import QueueItem, WorkQueue
from workers.sources import _common as C
from workers.sources import bench_mine as BM
from workers.sources import domain_research as DR
from workers.sources import gap_fill as GF
from workers.sources import session_distill as SD


def _item(payload: dict | None = None, **kw) -> QueueItem:
    base = dict(id=1, source="s", kind="k", priority=50, payload=payload or {},
                dedup_key=None, state="running", attempts=1, enqueued_at="",
                claimed_at=None, claimed_by=None, completed_at=None, error=None)
    base.update(kw)
    return QueueItem(**base)


@pytest.fixture
def q(tmp_path) -> WorkQueue:
    return WorkQueue(tmp_path / "workers.db")


# ---------------------------------------------------------------------------
# TurnResult — an empty answer is a failure, not a short success
# ---------------------------------------------------------------------------


def test_a_turn_with_no_text_is_not_ok():
    assert not C.TurnResult(text="", stop_reason="max_turns").ok
    assert not C.TurnResult(text="   \n ").ok
    assert C.TurnResult(text="something").ok


def test_the_failure_summary_names_why_the_turn_ended():
    """`max_turns` and a wedged engine produce the same empty string.

    Keeping the stop reason is the whole point of returning a `TurnResult`
    rather than a bare `str` — without it the two are indistinguishable, which
    is how a research pipeline came to record 225 empty notes as successes.
    """
    summary = C.TurnResult(text="", stop_reason="max_turns", num_turns=20).failure_summary()
    assert "max_turns" in summary and "20" in summary


@pytest.mark.parametrize("mod,payload", [
    (DR, {"topic": "some topic", "slug": "some-topic"}),
    (GF, {"entity": "E", "text": "a gap", "fact_id": "f1"}),
    (SD, {"session_path": "/tmp/nope.json"}),
    (BM, {"loser_task_id": "bench_1", "composite_score": 0.2}),
])
async def test_an_empty_turn_writes_nothing_and_is_recorded_as_failed(
        mod, payload, monkeypatch, tmp_path):
    """No note, no side effects, and a run record that says what happened."""
    written: list = []
    monkeypatch.setattr(mod, "write_staging_note",
                        lambda **kw: written.append(kw) or tmp_path / "x.md")
    monkeypatch.setattr(
        mod, "run_prompt_on_primary",
        lambda *a, **k: _async(C.TurnResult(text="", stop_reason="max_turns")))
    monkeypatch.setattr(SD, "_mark_done_if_exhausted", lambda item: None, raising=False)

    result = await mod.execute(_item(payload))
    assert result["status"] == "failed", f"{mod.NAME} recorded an empty turn as success"
    assert written == [], f"{mod.NAME} wrote a note for a turn that said nothing"
    assert result["meta"]["empty_response"] is True


async def _async(value):
    return value


async def test_an_empty_research_turn_leaves_the_topic_unticked(monkeypatch, tmp_path):
    """The compounding half of the bug.

    domain-research ticked the queue line to `[x]` in the same breath as
    writing the empty note, so the topic could never be retried. 90 of its 142
    notes are empty, and every one of those topics is now closed on disk.
    """
    queue_file = tmp_path / "research-queue.md"
    queue_file.write_text("- [ ] Direct Preference Optimization\n", encoding="utf-8")
    monkeypatch.setattr(DR, "QUEUE_FILE", queue_file)
    monkeypatch.setattr(
        DR, "run_prompt_on_primary",
        lambda *a, **k: _async(C.TurnResult(text="", stop_reason="stop")))

    await DR.execute(_item({"topic": "Direct Preference Optimization", "slug": "dpo"}))
    assert "- [ ]" in queue_file.read_text(), "an empty turn retired its own topic"


async def test_a_real_research_turn_writes_the_note_and_ticks_the_topic(monkeypatch, tmp_path):
    queue_file = tmp_path / "research-queue.md"
    queue_file.write_text("- [ ] Direct Preference Optimization\n", encoding="utf-8")
    monkeypatch.setattr(DR, "QUEUE_FILE", queue_file)
    monkeypatch.setattr(DR, "write_staging_note", lambda **kw: tmp_path / "note.md")
    monkeypatch.setattr(
        DR, "run_prompt_on_primary",
        lambda *a, **k: _async(C.TurnResult(text="## Summary\nreal\n## Confidence\n0.8: ok",
                                            stop_reason="stop")))

    result = await DR.execute(_item({"topic": "Direct Preference Optimization", "slug": "dpo"}))
    assert result["status"] == "success"
    assert "- [x]" in queue_file.read_text()


def test_confidence_parsing_has_one_definition():
    """It was pasted into four sources with three subtly different bodies."""
    assert C.parse_confidence("## Confidence\n0.85: because") == 0.85
    assert C.parse_confidence("no such section") == 0.5
    assert C.parse_confidence("confidence: 1.9") == 1.0, "must clamp"
    assert C.parse_confidence("Confidence 0") == 0.0
    # The pattern only matches a leading 0 or 1, so a number outside the range
    # is not read as an over-confident score — it is not read at all.
    assert C.parse_confidence("confidence: 4.2") == 0.5
    for mod in (DR, GF, SD):
        assert not hasattr(mod, "_parse_confidence"), \
            f"{mod.NAME} still carries its own copy"


# ---------------------------------------------------------------------------
# session-distill selection
# ---------------------------------------------------------------------------


def _session(dir_: Path, name: str, *, platform: str = "mission-control",
             age_seconds: float = 7200) -> Path:
    p = dir_ / f"{name}.json"
    p.write_text(json.dumps({
        "id": name, "title": "t", "model": "primary", "platform": platform,
        "messages": [{"role": "user", "content": "hello"}],
    }, indent=2), encoding="utf-8")
    old = time.time() - age_seconds
    import os
    os.utime(p, (old, old))
    return p


def test_a_worker_session_is_never_distilled(tmp_path, monkeypatch):
    """Lloyd's own triage transcripts are not observations about the user.

    12 of them had already been mined that way. `NON_USER_PLATFORMS` is the
    one definition of this, and this source did not consult it.
    """
    monkeypatch.setattr(SD, "SESSIONS_DIR", tmp_path)
    _session(tmp_path, "20260908_022133_backlogs_56de", platform="worker")
    _session(tmp_path, "20260907_000000_autonomy", platform="autonomy")
    assert SD._scan(time.time(), set()) == []


def test_a_user_session_that_has_gone_quiet_is_eligible(tmp_path, monkeypatch):
    monkeypatch.setattr(SD, "SESSIONS_DIR", tmp_path)
    p = _session(tmp_path, "20260901_120000_ivabcd")
    assert [x[1] for x in SD._scan(time.time(), set())] == [p]


def test_a_session_still_being_used_is_left_alone(tmp_path, monkeypatch):
    """A chat still being typed into is not a transcript to learn from."""
    monkeypatch.setattr(SD, "SESSIONS_DIR", tmp_path)
    _session(tmp_path, "20260908_020000_ivlive", age_seconds=60)
    assert SD._scan(time.time(), set()) == []


def test_a_session_with_a_running_turn_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(SD, "SESSIONS_DIR", tmp_path)
    _session(tmp_path, "20260901_120000_ivbusy")
    import app.sessions_io as sio
    monkeypatch.setattr(sio, "is_session_active", lambda sid: True)
    assert SD._scan(time.time(), set()) == []


def test_an_already_distilled_session_is_never_offered_again(tmp_path, monkeypatch):
    """The regression: 44 runs against one session, 138 across the worst eight.

    Selection keyed on an `mtime > last_mtime` watermark, and a session's
    mtime advances with every message it receives — so an active chat crossed
    the watermark again on the very next tick, forever. The queue's dedup key
    could not help: `mark_completed` releases it by design.
    """
    monkeypatch.setattr(SD, "SESSIONS_DIR", tmp_path)
    p = _session(tmp_path, "20260830_182633_ive386")
    assert SD._scan(time.time(), set()) == [(p.stat().st_mtime, p)]
    assert SD._scan(time.time(), {p.name}) == []


def test_a_session_skipped_while_busy_is_not_stranded(tmp_path, monkeypatch):
    """The moving watermark's mirror-image bug.

    A session that was active when the scan ran had its mtime left behind by
    the advancing watermark, so once it went quiet it was below the cursor and
    never looked at again. Per-session markers have no cursor to fall behind.
    """
    monkeypatch.setattr(SD, "SESSIONS_DIR", tmp_path)
    busy = _session(tmp_path, "20260901_100000_ivbusy", age_seconds=60)
    _session(tmp_path, "20260901_110000_ivother", age_seconds=7200)

    first = SD._scan(time.time(), set())
    assert busy not in [x[1] for x in first]

    import os
    old = time.time() - 7200
    os.utime(busy, (old, old))
    assert busy in [x[1] for x in SD._scan(time.time(), {"20260901_110000_ivother.json"})]


def test_selection_is_oldest_first(tmp_path, monkeypatch):
    monkeypatch.setattr(SD, "SESSIONS_DIR", tmp_path)
    newer = _session(tmp_path, "b_newer", age_seconds=3600)
    older = _session(tmp_path, "a_older", age_seconds=99999)
    assert [x[1] for x in SD._scan(time.time(), set())] == [older, newer]


def test_the_platform_is_read_from_the_head_of_the_file(tmp_path):
    """Session files run to megabytes; only the prefix is read.

    Every producer writes `platform` within the first few keys. A prefix with
    no platform in it reads as a user session, which is the same default
    `is_user_session` applies to a session that has none.
    """
    p = _session(tmp_path, "s", platform="worker")
    assert SD._session_platform(p) == "worker"

    headless = tmp_path / "headless.json"
    headless.write_text('{"messages": [' + '{"x": "' + "y" * 20000 + '"}]}',
                        encoding="utf-8")
    assert SD._session_platform(headless) == ""

    from app.sessions_io import is_user_session
    assert is_user_session({"platform": ""}) is True


async def test_the_old_cursor_is_migrated_rather_than_dropped(tmp_path, monkeypatch, q):
    """Otherwise the fix is a regression on its own history.

    The `last_mtime` cursor was the only record that a session had been
    considered. Replacing it with per-session markers and simply forgetting it
    makes every session below it eligible again — 143 files on this box, most
    already distilled, re-offered three per tick.
    """
    monkeypatch.setattr(SD, "SESSIONS_DIR", tmp_path)
    old = _session(tmp_path, "20260801_already_done", age_seconds=99999)
    new = _session(tmp_path, "20260906_not_yet", age_seconds=7200)
    q.wm_set(SD.NAME, "last_mtime", repr(old.stat().st_mtime))

    await SD.enqueue_if_due(q, {})

    assert f"done:{old.name}" in q.wm_keys(SD.NAME), "an already-distilled session came back"
    assert "last_mtime" not in q.wm_keys(SD.NAME), \
        "the retired cursor is still there for a session to be stranded behind"
    queued = [i.payload["session_path"] for i in q.list_items(source=SD.NAME)]
    assert queued == [str(new)], "only the unconsidered session should be enqueued"


async def test_the_migration_runs_once_and_is_then_inert(tmp_path, monkeypatch, q):
    monkeypatch.setattr(SD, "SESSIONS_DIR", tmp_path)
    _session(tmp_path, "20260801_done", age_seconds=99999)
    q.wm_set(SD.NAME, "last_mtime", repr(time.time()))

    assert SD._migrate_legacy_cursor(q) == 1
    assert SD._migrate_legacy_cursor(q) == 0


async def test_a_distilled_session_is_marked_done_on_success(tmp_path, monkeypatch, q):
    monkeypatch.setattr(SD, "write_staging_note", lambda **kw: tmp_path / "n.md")
    monkeypatch.setattr(
        SD, "run_prompt_on_primary",
        lambda *a, **k: _async(C.TurnResult(text="## Gaps\n- none", stop_reason="stop")))
    import workers.queue as Q
    monkeypatch.setattr(Q, "_queue_instance", q, raising=False)

    await SD.execute(_item({"session_path": str(tmp_path / "20260901_x.json")}))
    assert "done:20260901_x.json" in q.wm_keys(SD.NAME)


async def test_a_session_the_queue_is_about_to_poison_is_given_up_on(tmp_path, monkeypatch, q):
    """Otherwise the failure is a cycle, not a retry.

    Poisoning releases the `dedup_key`, and the next scan — with no marker to
    go on — offers the same session again.
    """
    monkeypatch.setattr(
        SD, "run_prompt_on_primary",
        lambda *a, **k: _async(C.TurnResult(text="", stop_reason="stop")))
    import workers.queue as Q
    monkeypatch.setattr(Q, "_queue_instance", q, raising=False)

    item = _item({"session_path": str(tmp_path / "doomed.json")}, attempts=3)
    result = await SD.execute(item)
    assert result["status"] == "failed"
    assert "done:doomed.json" in q.wm_keys(SD.NAME)


async def test_a_first_failure_leaves_the_session_retryable(tmp_path, monkeypatch, q):
    monkeypatch.setattr(
        SD, "run_prompt_on_primary",
        lambda *a, **k: _async(C.TurnResult(text="", stop_reason="stop")))
    import workers.queue as Q
    monkeypatch.setattr(Q, "_queue_instance", q, raising=False)

    await SD.execute(_item({"session_path": str(tmp_path / "flaky.json")}, attempts=1))
    assert q.wm_keys(SD.NAME) == []


# ---------------------------------------------------------------------------
# bench-mine
# ---------------------------------------------------------------------------


async def test_an_unchanged_ledger_is_not_reparsed(tmp_path, monkeypatch, q):
    """11 MB of JSON parsing, every two hours, to find nothing.

    The ledger is appended to only by an autoresearch round, which is far
    rarer than this source's tick.
    """
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("", encoding="utf-8")
    monkeypatch.setattr(BM, "LEDGER_PATH", ledger)

    parses = {"n": 0}
    monkeypatch.setattr(BM, "_recent_ledger_losers",
                        lambda *a, **k: parses.__setitem__("n", parses["n"] + 1) or [])

    await BM.enqueue_if_due(q, {})
    await BM.enqueue_if_due(q, {})
    assert parses["n"] == 1, "an unchanged ledger was parsed twice"

    ledger.write_text('{"x": 1}\n', encoding="utf-8")
    import os
    os.utime(ledger, (time.time() + 10, time.time() + 10))
    await BM.enqueue_if_due(q, {})
    assert parses["n"] == 2, "a changed ledger was not re-read"


async def test_a_missing_ledger_is_not_an_error(tmp_path, monkeypatch, q):
    monkeypatch.setattr(BM, "LEDGER_PATH", tmp_path / "nope.jsonl")
    await BM.enqueue_if_due(q, {})


# ---------------------------------------------------------------------------
# Worker turns: config source and budget
# ---------------------------------------------------------------------------


def test_a_worker_turn_reads_the_merged_config_not_the_file():
    """`config.yaml` off disk skips `${VAR}` expansion, the canary overlay,
    and — the one that bites — the `data/tool_overrides.yaml` merge, which is
    the live authority for what is switched off. A tool disabled from the
    Tools page stayed advertised to every worker turn."""
    src = inspect.getsource(C.run_prompt_on_primary)
    code = "\n".join(
        line for line in src.splitlines() if not line.strip().startswith("#")
    ).split('"""')[2]  # drop the docstring too — both discuss the old way
    assert "from app.config import CONFIG" in code
    assert "yaml.safe_load" not in code, "a worker turn is re-reading config.yaml"
    assert "config.yaml" not in code


def test_the_turn_budget_sits_strictly_under_the_pool_cap(monkeypatch):
    """Whoever's timer fires first decides what happens next.

    If the pool's wins, it cancels the HTTP request rather than the turn — and
    the chat path is built to keep running when its client disconnects. The
    turn is then orphaned, the pool records a failure and backs off, and for
    `backlog-selfmod` the retry re-selects the same item, because no verdict
    was written.
    """
    import workers.sources as sources
    monkeypatch.setattr(sources, "get_sources_config",
                        lambda: {"backlog-selfmod": {"max_duration_seconds": 3600}},
                        raising=False)
    budget = C.turn_timeout_for("backlog-selfmod")
    assert budget < 3600
    assert budget == 3600 - C.POOL_TIMEOUT_MARGIN_SECONDS


def test_the_margin_leaves_room_to_cancel_the_turn_and_persist_it():
    """Larger than `autonomy._POOL_TIMEOUT_MARGIN`, because this path has to
    reach the backend over HTTP on its way out."""
    import autonomy
    assert C.POOL_TIMEOUT_MARGIN_SECONDS > autonomy._POOL_TIMEOUT_MARGIN


def test_a_source_with_no_cap_still_gets_a_budget(monkeypatch):
    import workers.sources as sources
    monkeypatch.setattr(sources, "get_sources_config", lambda: {}, raising=False)
    assert C.turn_timeout_for("unknown-source") == 3600.0


def test_a_tiny_cap_does_not_produce_a_negative_budget(monkeypatch):
    import workers.sources as sources
    monkeypatch.setattr(sources, "get_sources_config",
                        lambda: {"s": {"max_duration_seconds": 10}}, raising=False)
    assert C.turn_timeout_for("s") == 60


async def test_an_overrunning_turn_is_cancelled_in_the_backend(monkeypatch, tmp_path):
    """Reporting a timeout without cancelling leaves the turn running.

    That is the orphan: the pool moves on, and a 90-iteration triage carries
    on burning the GPU with nobody waiting for it.
    """
    monkeypatch.setattr(C, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(C, "turn_timeout_for", lambda source: 0.05)

    cancelled: list[str] = []

    async def fake_cancel(backend, session_id):
        cancelled.append(session_id)
        return True

    monkeypatch.setattr(C, "_cancel_session_turn", fake_cancel)

    async def never_finishes(*a, **k):
        await asyncio.sleep(30)

    import httpx
    monkeypatch.setattr(httpx.AsyncClient, "__aenter__", never_finishes)

    with pytest.raises(C.TurnTimeout) as excinfo:
        await C.run_prompt_in_session("go", title="t", source="backlog-selfmod")
    assert cancelled, "the turn was left running in the backend"
    assert cancelled[0] in str(excinfo.value)


def test_a_worker_session_is_marked_as_one(tmp_path, monkeypatch):
    """`sessions_io.NON_USER_PLATFORMS` is what keeps a worker turn from being
    mistaken for the user's session by the morning brief and by
    session-distill alike."""
    monkeypatch.setattr(C, "SESSIONS_DIR", tmp_path)
    sid = C.new_worker_session(title="t", source="backlog-selfmod")
    data = json.loads((tmp_path / f"{sid}.json").read_text())

    from app.sessions_io import is_user_session
    assert data["platform"] == "worker"
    assert not is_user_session(data)
    assert data["inner_voice"] is True
