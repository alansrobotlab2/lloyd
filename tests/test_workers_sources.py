"""Worker sources — what a turn producing nothing means, and what gets mined.

Three failures this pins, all of which ran in production for weeks while
every run record said `success`:

  * A turn that produced no text had its empty output written to the vault as
    a research note and its input retired. 225 of the 498 notes under
    `pending-research/` have the body `(no response)`. The source that did the
    retiring, `domain-research`, is gone; the two remaining sources on this
    turn path are pinned here.
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
import yaml

from workers.queue import QueueItem, WorkQueue
from workers.sources import _common as C
from workers.sources import bench_mine as BM
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


def test_confidence_parsing_has_one_definition():
    """It was pasted into four sources with three subtly different bodies."""
    assert C.parse_confidence("## Confidence\n0.85: because") == 0.85
    assert C.parse_confidence("no such section") == 0.5
    assert C.parse_confidence("confidence: 1.9") == 1.0, "must clamp"
    assert C.parse_confidence("Confidence 0") == 0.0
    # The pattern only matches a leading 0 or 1, so a number outside the range
    # is not read as an over-confident score — it is not read at all.
    assert C.parse_confidence("confidence: 4.2") == 0.5
    for mod in (GF, SD):
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


async def test_repeated_failure_gives_up_instead_of_cycling(tmp_path, monkeypatch, q):
    """The regression, and why the count cannot come from the queue item.

    This first counted `item.attempts`, on the belief that a returned
    `{"status": "failed"}` goes through the queue's retry path. It does not —
    `workers/pool.py` records the failed run and completes the item anyway, so
    `item.attempts` is 1 on every failed distill. The give-up branch never
    fired, no marker was written, and the next scan offered the same session
    again: 39 failed runs in eight hours, one session enqueued 21 times.
    """
    monkeypatch.setattr(
        SD, "run_prompt_on_primary",
        lambda *a, **k: _async(C.TurnResult(text="", stop_reason="stop")))
    import workers.queue as Q
    monkeypatch.setattr(Q, "_queue_instance", q, raising=False)

    # Every attempt arrives with attempts=1, exactly as the pool delivers it.
    for _ in range(3):
        result = await SD.execute(
            _item({"session_path": str(tmp_path / "doomed.json")}, attempts=1))
        assert result["status"] == "failed"

    assert "done:doomed.json" in q.wm_keys(SD.NAME), "gave up on nothing"
    assert "fail:doomed.json" not in q.wm_keys(SD.NAME), "the counter is cleaned up"


async def test_a_first_failure_leaves_the_session_retryable(tmp_path, monkeypatch, q):
    monkeypatch.setattr(
        SD, "run_prompt_on_primary",
        lambda *a, **k: _async(C.TurnResult(text="", stop_reason="stop")))
    import workers.queue as Q
    monkeypatch.setattr(Q, "_queue_instance", q, raising=False)

    await SD.execute(_item({"session_path": str(tmp_path / "flaky.json")}, attempts=1))
    keys = q.wm_keys(SD.NAME)
    assert "done:flaky.json" not in keys, "one bad turn is not a verdict"
    assert q.wm_get(SD.NAME, "fail:flaky.json") == "1"


async def test_a_session_that_finally_works_forgets_its_failures(
        tmp_path, monkeypatch, q):
    """Otherwise an intermittent session accumulates toward a give-up across
    unrelated weeks."""
    import workers.queue as Q
    monkeypatch.setattr(Q, "_queue_instance", q, raising=False)
    monkeypatch.setattr(SD, "write_staging_note", lambda **kw: tmp_path / "n.md")
    monkeypatch.setattr(
        SD, "run_prompt_on_primary",
        lambda *a, **k: _async(C.TurnResult(text="", stop_reason="stop")))
    await SD.execute(_item({"session_path": str(tmp_path / "flaky.json")}, attempts=1))
    assert q.wm_get(SD.NAME, "fail:flaky.json") == "1"

    monkeypatch.setattr(
        SD, "run_prompt_on_primary",
        lambda *a, **k: _async(C.TurnResult(text="## Gaps\n- something", stop_reason="stop")))
    await SD.execute(_item({"session_path": str(tmp_path / "flaky.json")}, attempts=1))
    assert "done:flaky.json" in q.wm_keys(SD.NAME)
    assert q.wm_get(SD.NAME, "fail:flaky.json") is None


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
    # The second input must not read the real autonomy-runs tree either: the
    # test is about a missing ledger, and #522 gave this source a second input
    # that would otherwise scan 4,000 live run files on every run of it.
    monkeypatch.setattr(BM, "AUTONOMY_RUNS_DIR", tmp_path / "no-runs")
    await BM.enqueue_if_due(q, {})


# ---------------------------------------------------------------------------
# bench-mine: the failed-autonomy-run input (#522)
#
# The ledger is not an input this source can rely on — it only grows during an
# autoresearch round — and the docstring has advertised a second input
# (`autonomy-runs/**/run_*.md` with status=failed) since the day it was
# written while never opening the directory. 4,044 run files say otherwise.
# ---------------------------------------------------------------------------


_RUN_BODY = """## Prompt

[SYSTEM: You are executing autonomy task #24: "Data Pipeline". Follow the skill
instructions below.]

... 600 seconds of tool calls ...

(no output before timeout)

## Tool/script errors before timeout

```
Bash: {"error": "command timed out after 120000ms"}
```
"""


def _run_file(root: Path, run_id: str, *, task_id: int = 24, status: str = "failed",
              failure_kind: str = "task", summary: str = "timed out after 600s") -> Path:
    """A run record shaped like the real ones: `autonomy-runs/{task_id}/{run_id}.md`."""
    d = root / str(task_id)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{run_id}.md"
    p.write_text("---\n" + yaml.dump({
        "completed_at": "2026-09-08T00:29:54+00:00",
        "duration_seconds": 600.1,
        "failure_kind": failure_kind,
        "run_id": run_id,
        "started_at": "2026-09-08T00:19:54+00:00",
        "status": status,
        "summary": summary,
        "task_id": task_id,
    }) + "---\n\n" + _RUN_BODY, encoding="utf-8")
    return p


def _bm_queue(monkeypatch, q) -> None:
    """Send bench-mine's `done:`/`fail:` markers to the tmp DB.

    `execute` retires a mined source through `workers.queue.get_queue()`, the
    process singleton — unpatched, a unit test would write watermarks into the
    live workers.db.
    """
    import workers.queue as WQ
    monkeypatch.setattr(WQ, "get_queue", lambda *a, **k: q)


def _bm_inputs(tmp_path, monkeypatch) -> Path:
    """Point both inputs at tmp dirs; the ledger yields nothing by default."""
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("", encoding="utf-8")
    monkeypatch.setattr(BM, "LEDGER_PATH", ledger)
    monkeypatch.setattr(BM, "_recent_ledger_losers", lambda *a, **k: [])
    runs_dir = tmp_path / "autonomy-runs"
    runs_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(BM, "AUTONOMY_RUNS_DIR", runs_dir)
    return runs_dir


async def test_a_failed_run_is_mined_even_when_the_ledger_has_no_losers(tmp_path, monkeypatch, q):
    """The whole point of #522: one input dying cannot starve the source."""
    runs = _bm_inputs(tmp_path, monkeypatch)
    p = _run_file(runs, "run_24_20260908_235954")

    await BM.enqueue_if_due(q, {})

    items = q.list_items(source=BM.NAME)
    assert len(items) == 1, "a failed autonomy run was not offered for mining"
    assert items[0].payload["run_path"] == str(p)
    assert items[0].payload["run_id"] == "run_24_20260908_235954"


async def test_the_run_input_fires_even_with_the_ledger_gate_closed(tmp_path, monkeypatch, q):
    """Two inputs, two gates. The ledger's mtime gate must not gate the runs.

    The first cut of `enqueue_if_due` returned early when the ledger mtime had
    not moved, so everything added after that line — including any second
    input — was unreachable for as long as autoresearch stayed off.
    """
    runs = _bm_inputs(tmp_path, monkeypatch)
    parses = {"n": 0}
    monkeypatch.setattr(BM, "_recent_ledger_losers",
                        lambda *a, **k: parses.__setitem__("n", parses["n"] + 1) or [])

    await BM.enqueue_if_due(q, {})          # opens the ledger gate
    _run_file(runs, "run_36_20260908_235954")
    await BM.enqueue_if_due(q, {})          # ledger unchanged, new failure anyway

    assert parses["n"] == 1, "an unchanged ledger was re-parsed"
    assert len(q.list_items(source=BM.NAME)) == 1, "the run input was gated behind the ledger"


async def test_an_infrastructure_failure_is_not_mined(tmp_path, monkeypatch, q):
    """`ConnectError: All connection attempts failed` is not a capability gap.

    Mining it would put the network's outages on the bench and score Lloyd on
    them. 7 of the 104 failures in the last 7 days are this kind.
    """
    runs = _bm_inputs(tmp_path, monkeypatch)
    _run_file(runs, "run_60_20260908_235954", failure_kind="infra")
    _run_file(runs, "run_60_20260908_236000", status="success")

    await BM.enqueue_if_due(q, {})

    assert q.list_items(source=BM.NAME) == []


async def test_a_run_is_offered_once(tmp_path, monkeypatch, q):
    """`mark_completed` releases the dedup key, so a marker has to say "mined".

    Without it the same 104 failures cycle forever: this source ticks every
    two hours and session-distill's identical mistake ran 39 failed turns in
    eight hours.
    """
    runs = _bm_inputs(tmp_path, monkeypatch)
    _run_file(runs, "run_24_20260908_235954")

    await BM.enqueue_if_due(q, {})
    first = q.list_items(source=BM.NAME)[0]
    q.mark_completed(first.id)
    _bm_queue(monkeypatch, q)
    BM._mark_done("run_24_20260908_235954", "mined")

    await BM.enqueue_if_due(q, {})
    assert len(q.list_items(source=BM.NAME)) == 1, "an already-mined run was re-offered"


async def test_one_failing_task_does_not_flood_the_tick(tmp_path, monkeypatch, q):
    """Task 75 failed 20 times in 7 days and task 36 18; the bench must not
    become a wall of one task's timeout."""
    runs = _bm_inputs(tmp_path, monkeypatch)
    for i in range(12):
        _run_file(runs, f"run_75_2026090{i}_235954", task_id=75)
    _run_file(runs, "run_36_20260908_235954", task_id=36)

    await BM.enqueue_if_due(q, {})

    items = q.list_items(source=BM.NAME)
    assert items, "nothing was mined"
    assert len(items) <= BM.MAX_ENQUEUE_PER_TICK, "the tick was not capped"
    tasks = [i.payload["task_id"] for i in items]
    assert len(set(tasks)) == len(tasks), f"one task took more than one slot: {tasks}"


async def test_a_staged_mined_task_names_the_run_it_came_from(tmp_path, monkeypatch, q):
    runs = _bm_inputs(tmp_path, monkeypatch)
    _bm_queue(monkeypatch, q)
    p = _run_file(runs, "run_24_20260908_235954")
    monkeypatch.setattr(C, "STAGING_ROOT", tmp_path / "pending-research")
    monkeypatch.setattr(BM, "run_prompt_on_primary", lambda *a, **k: _async(
        C.TurnResult(text=_CANDIDATE, stop_reason="stop", num_turns=3)))
    calibrations: list = []

    async def _fake_cal(path, **kw):
        calibrations.append(path)
        return {"runs": 10, "composites": [0.4] * 10, "mean": 0.4,
                "min": 0.4, "max": 0.4, "in_band": True, "status": "ok"}

    monkeypatch.setattr(BM, "calibrate_candidate", _fake_cal)

    result = await BM.execute(_item({"run_path": str(p), "run_id": "run_24_20260908_235954",
                                     "task_id": "24", "summary": "timed out after 600s"}))

    assert result["status"] == "success", result
    staged = Path(result["artifact_path"])
    fm = yaml.safe_load(staged.read_text(encoding="utf-8").split("---")[1])
    assert fm["source"] == BM.NAME
    assert any("run_24_20260908_235954" in str(r) for r in fm["source_refs"]), fm["source_refs"]
    assert "run_24_20260908_235954" in fm["rationale"]
    assert fm["calibration"]["in_band"] is True, "the band verdict was not recorded"
    assert calibrations, "the candidate was staged without being calibrated"


_CANDIDATE = """---
id: bench_200_mined_run_24_pipeline_timeout
category: synthetic
objective: Recover from a tool call that exceeds its wall-clock budget
max_tool_calls: 6
edge_direction: execution-length
prompt: A Bash call you just made reported `command timed out after 120000ms` and the
  work is still unfinished. What do you do?
objective_checks:
- type: contains
  value: timeout
- type: tool_not_called
  value: automod_land
rubric_criteria:
- names_the_failing_call
- bounds_the_retry
safety_critical: false
---

Reports the timeout, then re-runs the work bounded rather than retrying it verbatim.
"""


def _candidate_turn(text: str):
    async def _turn(*a, **k):
        return C.TurnResult(text=text, stop_reason="stop", num_turns=3)
    return _turn


async def test_the_run_failure_prompt_demands_a_mechanical_check(tmp_path, monkeypatch, q):
    """A ledger loser has a bench task to branch from; a failed run has only a
    transcript. And a mined task with no mechanical check is a rubric-only
    task, which is weaker evidence than either paper uses.
    """
    runs = _bm_inputs(tmp_path, monkeypatch)
    _bm_queue(monkeypatch, q)
    p = _run_file(runs, "run_24_20260908_235954")
    prompts: list[str] = []
    monkeypatch.setattr(BM, "run_prompt_on_primary", _capture(prompts))
    monkeypatch.setattr(BM, "write_staging_note", lambda **kw: tmp_path / "s.md")

    async def _no_cal(path, **kw):
        return {"runs": 0, "composites": [], "status": "skipped", "in_band": None}

    monkeypatch.setattr(BM, "calibrate_candidate", _no_cal)

    await BM.execute(_item({"run_path": str(p), "run_id": "run_24_20260908_235954",
                            "task_id": "24", "summary": "timed out after 600s"}))
    run_prompt = prompts[0]
    prompts.clear()
    await BM.execute(_item({"loser_task_id": "bench_004_replay_schedule_task",
                            "composite_score": 0.25}))

    assert run_prompt != prompts[0], "both inputs share one prompt"
    for need in ("objective_checks", "rubric_criteria", "edge_direction", "REJECT"):
        assert need in run_prompt, f"the run prompt never asks for {need}"
    assert "mutation" in run_prompt.lower(), "the run prompt allows real-state mutation"
    assert str(p) in run_prompt, "the run prompt does not name the transcript to read"


def _capture(sink: list):
    def _turn(prompt, *a, **k):
        sink.append(prompt)
        return _async(C.TurnResult(text=_CANDIDATE, stop_reason="stop", num_turns=3))
    return _turn


async def test_a_candidate_with_no_mechanical_check_is_not_staged(tmp_path, monkeypatch, q):
    """Judgement-shaped failures are most of Lloyd's failures. The honest
    answer for those is no task at all, not a rubric-only one."""
    runs = _bm_inputs(tmp_path, monkeypatch)
    _bm_queue(monkeypatch, q)
    p = _run_file(runs, "run_24_20260908_235954")
    monkeypatch.setattr(C, "STAGING_ROOT", tmp_path / "pending-research")
    monkeypatch.setattr(BM, "run_prompt_on_primary", _candidate_turn(
        "REJECT: the failure is 'the run stopped early'. Every response that "
        "answers the question passes; no mechanical check distinguishes them."))

    result = await BM.execute(_item({"run_path": str(p), "run_id": "run_24_20260908_235954",
                                     "task_id": "24", "summary": "empty response"}))

    assert result["status"] == "skipped", result
    assert not (tmp_path / "pending-research").exists() or \
        list((tmp_path / "pending-research").rglob("*.md")) == []


async def test_a_candidate_is_kept_only_at_the_capability_edge(tmp_path):
    """A task the model always passes and a task it always fails both move the
    mean by noise. Four of the eleven live tasks are pinned at 0.00."""
    p = tmp_path / "candidate.md"
    p.write_text(_CANDIDATE, encoding="utf-8")

    async def always_pass(tasks, **kw):
        return [1.0 for _ in tasks]

    async def edge(tasks, **kw):
        return [0.4, 0.5, 0.6, 0.5, 0.4, 0.5, 0.6, 0.5, 0.4, 0.5]

    saturated = await BM.calibrate_candidate(p, runs=10, trials=always_pass)
    at_edge = await BM.calibrate_candidate(p, runs=10, trials=edge)

    assert saturated["in_band"] is False, "an always-passed task read as discriminating"
    assert at_edge["in_band"] is True, "a task at the edge was thrown away"
    assert at_edge["mean"] == pytest.approx(0.49)


async def test_a_calibration_error_does_not_lose_the_candidate(tmp_path, monkeypatch):
    """The GPU being down is not a reason to throw away the only mined task."""
    p = tmp_path / "candidate.md"
    p.write_text(_CANDIDATE, encoding="utf-8")

    async def boom(tasks, **kw):
        raise RuntimeError("vLLM is not answering")

    result = await BM.calibrate_candidate(p, runs=10, trials=boom)

    assert result["in_band"] is None, "an unmeasured task was reported as in or out of band"
    assert "vLLM" in result["error"]


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
    `autotriage` the retry re-selects the same item, because no verdict
    was written.
    """
    import workers.sources as sources
    monkeypatch.setattr(sources, "get_sources_config",
                        lambda: {"autotriage": {"max_duration_seconds": 3600}},
                        raising=False)
    budget = C.turn_timeout_for("autotriage")
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
        await C.run_prompt_in_session("go", title="t", source="autotriage")
    assert cancelled, "the turn was left running in the backend"
    assert cancelled[0] in str(excinfo.value)


def test_a_worker_session_is_marked_as_one(tmp_path, monkeypatch):
    """`sessions_io.NON_USER_PLATFORMS` is what keeps a worker turn from being
    mistaken for the user's session by the morning brief and by
    session-distill alike."""
    monkeypatch.setattr(C, "SESSIONS_DIR", tmp_path)
    sid = C.new_worker_session(title="t", source="autotriage")
    data = json.loads((tmp_path / f"{sid}.json").read_text())

    from app.sessions_io import is_user_session
    assert data["platform"] == "worker"
    assert not is_user_session(data)
    assert data["inner_voice"] is True
