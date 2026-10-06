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
import logging
import time
from pathlib import Path

import pytest
import yaml

from workers.queue import QueueItem, WorkQueue
from workers.sources import _common as C
from workers.sources import bench_mine as BM
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


async def test_the_infra_shaped_digest_turn_asks_for_spacing(monkeypatch, tmp_path):
    """An empty turn whose cause is the engine asks to be re-offered later — and
    still spends nothing from the video's own retry budget (#1714).

    The three sources above finish an empty turn and are done with it: each owns
    a registry (`research.db`, the session's `attempts`) and re-offers from there.
    `youtube-digest`'s infra branch owns NOTHING — it deliberately skips the
    script's `--fail` so `seen.json` does not charge a video for an outage (six
    videos went through it in five seconds each on 2026-09-09 and all six were
    right to be retried) — and with no registry touched, the only spacing left was
    the source's `interval_seconds`: 300 s, forever. That is what `defer_seconds`
    is for, and the no-`--fail` half is asserted here in the same test because a
    bound bought by charging the video would be the regression, not the fix.

    The digest module's own harness is imported rather than re-implemented —
    `tests/test_youtube_digest_source.py` owns `_meta`, `_Script` and `_turn`.
    """
    from workers.sources import youtube_digest as Y
    from tests.test_youtube_digest_source import _Script, _item, _meta, _turn

    monkeypatch.setattr(Y, "BACKLOG_DIR", tmp_path / "backlog")
    monkeypatch.setattr(Y, "_vault_dirty_paths", lambda: set())
    script = _Script({"ok": True, "meta": _meta(tmp_path)})
    monkeypatch.setattr(Y, "_script", script)
    monkeypatch.setattr(Y, "run_prompt_in_session", _turn("", stop_reason=None))

    result = await Y.execute(_item({"channel": "ai-engineer", "video_id": "abc123"}))

    assert result["status"] == "failed" and result["meta"]["infra"] is True
    assert result["meta"]["empty_response"] is True
    assert result["defer_seconds"] == Y.INFRA_DEFER_SECONDS == 900, (
        f"the infra branch deferred {result.get('defer_seconds')!r}s; the bound is "
        "INFRA_DEFER_SECONDS (900 s, the scanner's own RETRY_INTERVAL_SECONDS), "
        "stated in the module and in §Intake")
    assert script.modes() == ["--fetch"], (
        "the branch reached for --fail to get a backoff: seen.json would now hold a "
        "failure_count for a primary outage, which is the thing this branch refuses to do")
    assert not any("--fail" in arg for call in script.calls for arg in call[1:])


@pytest.mark.parametrize("mod,payload", [
    (SD, {"session_path": "/tmp/nope.json"}),
    (BM, {"loser_task_id": "bench_1", "composite_score": 0.2}),
])
async def test_an_ordinary_empty_turn_asks_for_no_spacing(mod, payload, monkeypatch, tmp_path):
    """The deferral is the infra branch's, not a blanket rule for empty turns.

    `session-distill` and `bench-mine` decide their own re-offer in their own
    registry, which is what §Intake's "the retry lives outside the queue" means; a
    `defer_seconds` on their empty turn would park a queue row the source has
    already written off, and the dedup key it now keeps would block the next
    legitimate item of the same kind for the length of the bound.
    """
    monkeypatch.setattr(
        mod, "run_prompt_on_primary",
        lambda *a, **k: _async(C.TurnResult(text="", stop_reason="max_turns")))
    monkeypatch.setattr(mod, "write_staging_note",
                        lambda **kw: tmp_path / "x.md")
    monkeypatch.setattr(SD, "_mark_done_if_exhausted", lambda item: None, raising=False)

    result = await mod.execute(_item(payload))
    assert result["status"] == "failed"
    assert "defer_seconds" not in result, (
        f"{mod.NAME} deferred an empty turn it owns the retry for: the row would sit "
        "unclaimable while the source believes it is done")


async def test_a_state_turn_that_wrote_nothing_does_not_report_a_clean_stop(
        monkeypatch, tmp_path):
    """`stop_reason="stop"` in a failure record is a claim about the model.

    #867: the state driver could report `done=True` with a zero-character reply,
    and `run_prompt_with_run_state` translated that straight into `"stop"`. The
    driver no longer honours a `done` with no deliverable, but this mapping is
    the other half of that rule and is the one a source's own run record is
    written from — so it is pinned here too, against a result built by hand
    rather than through the driver, exactly because the two live in different
    modules and a policy enforced on one side of a module boundary drifts.
    """
    from app.harness.run_state import RunStateResult

    async def fake_turn(**kw):
        return RunStateResult(state=kw["state"], done=True, text="",
                              done_refusals=0)
    monkeypatch.setattr(C, "run_state_turn", fake_turn)

    st = C.RunState(job="session-distill",
                    schema={"title": "s", "type": "object", "properties": {},
                            "additionalProperties": False},
                    max_state_chars=2000, run_dir=tmp_path)
    turn = await C.run_prompt_with_run_state(
        "distill it", job="session-distill", state=st, run_dir=tmp_path)

    assert turn.stop_reason != "stop", \
        "a run that wrote nothing reported that the model finished cleanly"
    assert turn.stop_reason == "max_turns"
    assert not turn.ok
    assert "nothing written" in turn.failure_summary()


async def test_a_state_turn_with_a_deliverable_still_reports_a_clean_stop(
        monkeypatch, tmp_path):
    """The guard must not cost a finished run its stop reason."""
    from app.harness.run_state import RunStateResult

    async def fake_turn(**kw):
        return RunStateResult(state=kw["state"], done=True,
                              text="## Confidence\n0.8: fine")
    monkeypatch.setattr(C, "run_state_turn", fake_turn)

    st = C.RunState(job="session-distill",
                    schema={"title": "s", "type": "object", "properties": {},
                            "additionalProperties": False},
                    max_state_chars=2000, run_dir=tmp_path)
    turn = await C.run_prompt_with_run_state(
        "distill it", job="session-distill", state=st, run_dir=tmp_path)

    assert turn.stop_reason == "stop"
    assert turn.ok


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
    for mod in (SD, BM):
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


def test_the_platform_is_found_in_the_head_or_past_it(tmp_path, monkeypatch):
    """#1271: only the first 8 KB used to be read, and a session whose
    `platform` serialised after its messages array read as a user session —
    7 `worker` and 24 `browser` files a fortnight were mined as human chats.
    The head is still the fast path; a miss reads the whole file's top-level
    key. A file with no platform anywhere still reads as a user session."""
    import os
    monkeypatch.setattr(SD, "SESSIONS_DIR", tmp_path)
    early = _session(tmp_path, "s", platform="worker")
    assert SD._session_platform(early) == "worker"

    late = tmp_path / "20260919_120000_autocode_9f2a.json"
    body = [{"role": "user", "content": "y" * 20000}]
    late.write_text(json.dumps({"id": late.stem, "messages": body,
                                "platform": "worker"}), encoding="utf-8")
    old = time.time() - 7200
    os.utime(late, (old, old))
    assert late.stat().st_size > SD._PLATFORM_PREFIX_BYTES
    assert SD._session_platform(late) == "worker"
    assert SD._eligible(late, late.stat().st_mtime, time.time()) == \
        (False, "not a user session")

    # A nested "platform" inside a message is not the session's.
    nested = tmp_path / "nested.json"
    nested.write_text(json.dumps({"messages": [{"meta": {"platform": "worker"}}],
                                  "platform": "mission-control"}), encoding="utf-8")
    assert SD._session_platform(nested) == "mission-control"


def test_a_chat_with_no_platform_anywhere_is_still_eligible(tmp_path, monkeypatch):
    monkeypatch.setattr(SD, "SESSIONS_DIR", tmp_path)
    p = tmp_path / "20260830_182633_ive386.json"
    p.write_text(json.dumps({"id": p.stem, "messages": [
        {"role": "user", "content": "z" * 20000}]}), encoding="utf-8")
    old = time.time() - 7200
    import os
    os.utime(p, (old, old))
    assert SD._session_platform(p) == ""
    assert SD._eligible(p, p.stat().st_mtime, time.time()) == (True, "")


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
# bench-mine: the ledger-loser selector (#625)
#
# `variant_sandbox.materialize_baseline` mints the baseline's id as
# `BASELINE_<int>`, and that id is what reaches `ledger.jsonl` as
# `variant_id`. The filter that read these rows matched lowercase `baseline` —
# a case-sensitive comparison that selected no baseline row in the ledger's
# entire life, so the ledger half of this source never enqueued anything:
# `workers.db` held 0 queue rows of kind `mine` against 134 of kind `mine-run`
# as of 2026-09-19.
#
# Each row is produced the way the pipeline produces one: keys from
# `bench_runner_sdk.ledger_row_for`, bytes from `common.ledger_append` (the same
# appender `run_round.py` calls for every live row). A rename of
# `trace_status`/`composite_score` in the builder, or a change to how a ledger
# line is serialised, therefore lands on these tests rather than passing them
# by.
#
# A fixture ledger rather than the live file. The live one can show the case
# half: it grows again now that #876 has cleared, so counting the rows whose
# `variant_id.upper()` starts `BASELINE` and whose `composite_score` is under
# 0.6 inside the 7-day window yields rows, while the lowercase filter yields
# none. It cannot show the `trace_status` half — no errored baseline row sits
# inside the window — so only a fixture can pin that a harness failure is not
# mined as a weakness of the task.
# ---------------------------------------------------------------------------

def _ledger_row(variant_id: str, task_id: str, score: float, *,
                created_at: str | None = None,
                trace_status: str = "success",
                round_id: str = "R_fixture") -> dict:
    """One baseline-trial ledger row, built by the pipeline's own row builder.

    The keys come from `bench_runner_sdk.ledger_row_for` — the function that
    stamps `trace_status` and `composite_score` onto every row both runners
    write — rather than from a test-local spelling of them. Typing
    `"trace_status"` by hand here would leave this file passing while the
    writer renamed the key, which is the same one-sided contract that let
    `startswith("baseline")` sit unread against `BASELINE_<int>` for the whole
    life of the ledger.

    `trace_status` defaults to `success` because that is what 4,003 of the
    ledger's 4,080 baseline rows carry; 77 are `error`, and every one of those
    scores under the 0.6 loser line.

    `round_id` is a parameter because the pair `(task_id, round_id)` is the
    identity a `done:` marker is filed under (`_item_key` →
    `ledger:{task_id}:{round_id}`), so a test that wants to mark one round of a
    task and offer another cannot get there through `task_id` alone. It still
    defaults to `R_fixture`, the value `ledger_row_for` was already being handed.
    """
    from scripts.autoresearch.bench_runner_sdk import ledger_row_for

    row = ledger_row_for(
        {"variant_id": variant_id, "task_id": task_id, "status": trace_status},
        {"composite_score": score, "objective_score": score, "rubric_overall": score},
        round_id)
    if created_at:
        # `ledger_row_for` stamps `now_iso()`, which is inside the window; only
        # a row deliberately placed elsewhere needs this override.
        row["created_at"] = created_at
    return row


def _fixture_ledger(tmp_path, monkeypatch, rows: list[dict]) -> Path:
    """Point the source at a temp ledger; keep the run input off-screen.

    `_recent_ledger_losers` is deliberately NOT monkeypatched here — the other
    tests in this file do that, which is why none of them ever called the
    selector and nothing caught the case mismatch.

    Each row goes through the real `ledger_append`, so the bytes the selector
    parses are the bytes the pipeline's own serializer produces rather than a
    test-local `json.dumps` guess at them.
    """
    from scripts.autoresearch.common import ledger_append

    ledger = tmp_path / "ledger.jsonl"
    for row in rows:
        ledger_append(ledger, row)
    monkeypatch.setattr(BM, "LEDGER_PATH", ledger)
    monkeypatch.setattr(BM, "AUTONOMY_RUNS_DIR", tmp_path / "no-runs")
    return ledger


def test_an_uppercase_baseline_loser_is_selected(tmp_path, monkeypatch):
    """The selector must select the id the writer actually writes.

    Before #625 was fixed this returned []: `startswith("baseline")` against
    `BASELINE_123` is false, since `startswith` is case-sensitive.
    """
    _fixture_ledger(tmp_path, monkeypatch,
                    [_ledger_row("BASELINE_123", "bench_004_shape_gate", 0.42)])
    rows = BM._recent_ledger_losers()
    assert [r["task_id"] for r in rows] == ["bench_004_shape_gate"], (
        "a BASELINE_-prefixed row scoring 0.42 inside the window was not selected")


def test_a_variant_row_is_still_not_losers(tmp_path, monkeypatch):
    """Widening the prefix check must not turn it into "every low row".

    A `V_*` candidate scoring 0.10 is a variant that lost to the baseline, not
    a baseline loser; mining from it would mine the candidate's own weakness.
    The ledger holds roughly seven `V_*` rows per baseline row, so leaking them
    would dominate everything this selector returns.
    """
    _fixture_ledger(tmp_path, monkeypatch, [
        _ledger_row("BASELINE_123", "bench_004_shape_gate", 0.42),
        _ledger_row("V_20260916_000001_looser", "bench_010_safety_destructive", 0.10),
    ])
    rows = BM._recent_ledger_losers()
    assert [r["task_id"] for r in rows] == ["bench_004_shape_gate"], (
        "a V_* variant row leaked into the baseline losers")


@pytest.mark.parametrize("trace_status", ["error", "timeout"])
def test_a_baseline_trial_that_did_not_run_is_not_a_loser(
        tmp_path, monkeypatch, trace_status):
    """A low score from a broken trial is a measurement of the harness.

    `bench_runner` emits exactly three trace statuses — success, timeout, error
    (`scripts/autoresearch/bench_runner.py:13`) — and the ledger carries 77
    baseline rows with `trace_status: "error"`, every one of them scoring under
    the 0.6 loser line. #522's own case for mining baseline losers rested on
    those scores being real; the selector has to keep that true, or widening the
    prefix (which previously hid every row, errored included) would start
    handing the miner a task the runner never finished as a weakness of the
    model.
    """
    _fixture_ledger(tmp_path, monkeypatch, [
        _ledger_row("BASELINE_123", "bench_004_shape_gate", 0.42),
        _ledger_row("BASELINE_456", "bench_007_broken_trial", 0.10,
                    trace_status=trace_status),
    ])
    rows = BM._recent_ledger_losers()
    assert [r["task_id"] for r in rows] == ["bench_004_shape_gate"], (
        f"a baseline trial with trace_status {trace_status!r} and composite 0.10 "
        "was mined as a loser, so a harness failure reads as a task weakness")


def test_a_baseline_row_with_no_trace_status_is_not_a_loser(tmp_path, monkeypatch):
    """Absence is not success. A row written before the field existed, or by a
    harness that omitted it, carries no evidence the trial completed, so it
    cannot be the basis for a new bench task."""
    row = _ledger_row("BASELINE_789", "bench_005_no_status", 0.30)
    del row["trace_status"]
    _fixture_ledger(tmp_path, monkeypatch, [
        _ledger_row("BASELINE_123", "bench_004_shape_gate", 0.42), row,
    ])
    rows = BM._recent_ledger_losers()
    assert [r["task_id"] for r in rows] == ["bench_004_shape_gate"], (
        "a baseline row with no trace_status was mined as a loser")


def test_both_spellings_of_the_baseline_prefix_are_accepted(tmp_path, monkeypatch):
    """Neither spelling replaces the other; the comparison ignores case.

    Replacing `"baseline"` with `"BASELINE"` would fix every live row and
    silently drop any lowercase row a future writer or a hand-edited fixture
    carries.
    """
    _fixture_ledger(tmp_path, monkeypatch, [
        _ledger_row("BASELINE_123", "bench_004_shape_gate", 0.42),
        _ledger_row("baseline_1", "bench_002_no_preamble", 0.37),
        _ledger_row("V_20260916_000001_looser", "bench_010_safety_destructive", 0.10),
    ])
    ids = sorted(r["task_id"] for r in BM._recent_ledger_losers())
    assert ids == ["bench_002_no_preamble", "bench_004_shape_gate"], (
        "both baseline spellings must be selected, and the V_* row must not be")


async def test_a_ledger_loser_reaches_the_queue_as_a_mine_item(tmp_path, monkeypatch, q):
    """Selection is worth nothing if it stops short of the queue.

    `KIND_LEDGER` is the kind `execute` routes to `_mine_ledger_losers`'s
    prompt; this source produced zero of them before the selector was fixed.
    """
    _fixture_ledger(tmp_path, monkeypatch,
                    [_ledger_row("BASELINE_123", "bench_004_shape_gate", 0.42)])

    await BM.enqueue_if_due(q, {})

    items = [i for i in q.list_items(source=BM.NAME) if i.kind == BM.KIND_LEDGER]
    assert len(items) == 1, "the selected loser never reached the queue"
    assert items[0].payload["loser_task_id"] == "bench_004_shape_gate"
    assert items[0].payload["composite_score"] == 0.42


# ── the ledger input reads its own `done:` markers (#1711) ────────────
#
# The heading over the marker section of the module says a failure corpus is an
# infinite loop without markers, and the ledger half of the module wrote them and
# never read them: `_stage_and_calibrate` retires a pair on both outcomes
# (`_mark_done(_item_key(item), …)`), `mark_completed` releases the dedup key by
# design, and `_enqueue_ledger_losers` asked the queue nothing. Measured over the
# 7 days to 2026-09-28: 133 completed `mine` items, 127 of them one task
# (`bench_007_skill_invocation`) spread over five rounds at 29, 26, 24, 24 and 24
# — 6.54 wall-hours, every one of them a real primary turn, while
# `watermarks` already held the five `done:ledger:bench_007_…` keys.
#
# The key is the pair, so these tests always name a round as well as a task: a
# filter on task_id alone would strand the other rounds of a task that legitimately
# lost again, and the flood is five rounds of ONE task.
# ─────────────────────────────────────────────────────────────────────


async def test_a_ledger_loser_with_a_done_marker_is_not_enqueued(tmp_path, monkeypatch, q):
    """Clause 1, seeded from outside the module: the marker is a literal string,
    not something this test asked the producer to write, so the two sides cannot
    agree with each other and with nothing else. That is the class of bug that
    cost #625 its whole life — a selector and a writer each self-consistent."""
    _fixture_ledger(tmp_path, monkeypatch,
                    [_ledger_row("BASELINE_123", "bench_x", 0.42, round_id="R_1")])
    q.wm_set(BM.NAME, "done:ledger:bench_x:R_1",
             json.dumps({"why": "mined", "at": "2026-09-28T00:00:00+00:00"}))

    await BM.enqueue_if_due(q, {})

    assert [i for i in q.list_items(source=BM.NAME)
            if i.kind == BM.KIND_LEDGER] == [], \
        "a (task, round) pair already carrying a done: marker was offered again"


async def test_an_unmarked_ledger_loser_is_still_enqueued_with_its_fields(
        tmp_path, monkeypatch, q):
    """The positive control for the test above, and the pair-granularity half:
    one row of this task IS marked, its other round is not, and only the marked
    one may vanish. A filter keyed on task_id alone passes the test above and
    fails this one — which is the difference between closing #1711 and replacing
    a flood with a silence."""
    _fixture_ledger(tmp_path, monkeypatch, [
        _ledger_row("BASELINE_123", "bench_x", 0.42, round_id="R_1"),
        _ledger_row("BASELINE_124", "bench_x", 0.31, round_id="R_2"),
    ])
    q.wm_set(BM.NAME, "done:ledger:bench_x:R_1", json.dumps({"why": "mined"}))

    await BM.enqueue_if_due(q, {})

    items = [i for i in q.list_items(source=BM.NAME) if i.kind == BM.KIND_LEDGER]
    assert [i.payload["round_id"] for i in items] == ["R_2"], \
        "the unmarked round of a partly-marked task did not survive the filter"
    payload = items[0].payload
    assert payload["loser_task_id"] == "bench_x"
    assert payload["composite_score"] == 0.31
    assert payload["max_turns"] == BM.DEFAULT_MAX_TURNS, \
        "the envelope changed while the filter was being added"


async def test_a_ledger_loser_is_offered_once(tmp_path, monkeypatch, q):
    """The ledger analogue of `test_a_run_is_offered_once`, and the seam the
    other three do not cross: the marker here is written by the module's own
    retire path through `_item_key`, so this fails if the enqueue side and the
    retire side ever stop spelling the key the same way.

    `mark_completed` is called for real rather than skipped, because releasing
    the dedup key is the mechanism: without a read of the marker, the next tick
    finds an empty queue slot and fills it. The mtime gate is then opened by
    `utime`, which is what an autoresearch round appending a row does to the
    file — the tick that re-offered these five pairs every two hours was opened
    by a write that had nothing to do with them.
    """
    ledger = _fixture_ledger(tmp_path, monkeypatch,
                             [_ledger_row("BASELINE_123", "bench_x", 0.42,
                                          round_id="R_1")])

    await BM.enqueue_if_due(q, {})
    first = [i for i in q.list_items(source=BM.NAME)
             if i.kind == BM.KIND_LEDGER][0]
    q.mark_completed(first.id)
    _bm_queue(monkeypatch, q)
    BM._mark_done(BM._item_key(first), "mined")

    import os
    os.utime(ledger, (time.time() + 10, time.time() + 10))
    await BM.enqueue_if_due(q, {})

    assert len([i for i in q.list_items(source=BM.NAME)
                if i.kind == BM.KIND_LEDGER]) == 1, \
        "an already-mined ledger pair was re-offered"


async def test_a_tick_whose_losers_are_all_marked_offers_nothing_and_widens_nothing(
        tmp_path, monkeypatch, q):
    """Every candidate marked: the tick must come home empty.

    The second input still has to run (a ledger with nothing left to mine is not
    a reason to stop mining failed runs), and the candidate slice must not be
    widened or re-fetched to find something. Both halves are pinned, because the
    tempting "fix" for the resulting silence is to ask the selector for more rows
    — which is the selection-budget decision #1711 explicitly leaves to the
    owed-check job, not something a filter is allowed to sneak in as a loop.

    `_recent_ledger_losers` is spied on rather than run against a fixture here:
    this test is about how many times the enqueue path asks, and how wide it asks.

    #1774 moved the marker filter from after the slice into the scan, which moved
    this test's weight onto the spy: the enqueue path no longer looks at `done`
    itself, so the spy has to honour the `done` kwarg or it is reporting a slate
    the real scan would never have returned. That is not a weakening — with the
    post-slice filter gone, a spy that ignored `done` would hand back three marked
    rows and the `== []` assertion below would then FAIL, which is exactly the
    regression an empty queue here exists to catch. So the kwarg is asserted as
    received as well as honoured.
    """
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("", encoding="utf-8")
    monkeypatch.setattr(BM, "LEDGER_PATH", ledger)
    monkeypatch.setattr(BM, "AUTONOMY_RUNS_DIR", tmp_path / "no-runs")

    rows = [_ledger_row(f"BASELINE_{n}", "bench_x", 0.1 * n, round_id=f"R_{n}")
            for n in (1, 2, 3)]
    for row in rows:
        q.wm_set(BM.NAME, f"done:ledger:{row['task_id']}:{row['round_id']}",
                 json.dumps({"why": "mined"}))

    # Read the real signature BEFORE the spy stands in for it: the default is the
    # budget being asserted against, and inspecting the spy would compare the spy
    # with itself.
    default_limit = inspect.signature(BM._recent_ledger_losers).parameters["limit"].default

    calls: list[tuple[tuple, dict]] = []

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        # Honour the contract the scan now has: drop every row whose pair key is
        # in the set the caller supplied. Spelled as the literal the watermark
        # holds, not via `BM._ledger_key`, so this spy cannot agree with the
        # producer about a format both invented (#625's one-sided contract).
        done = kwargs.get("done") or set()
        return [r for r in rows
                if f"ledger:{r['task_id']}:{r['round_id']}" not in done]

    monkeypatch.setattr(BM, "_recent_ledger_losers", _spy)

    await BM.enqueue_if_due(q, {})          # raises nothing is the assertion

    assert q.list_items(source=BM.NAME) == [], "a marked-through-the-board tick enqueued anyway"
    assert len(calls) == 1, f"the candidate slice was fetched {len(calls)} times, not once"
    assert "done" in calls[0][1], (
        "the enqueue path did not pass the marker set at all, so nothing between "
        "the queue's watermarks and the slice is filtering anything")

    asked = (calls[0][1].get("limit")
             if "limit" in calls[0][1]
             else (calls[0][0][1] if len(calls[0][0]) > 1 else default_limit))
    assert int(asked) <= int(default_limit), \
        f"the tick asked for {asked} candidates to make up for filtered ones"


# ── the marker filter runs inside the scan, before the slice (#1774) ───
#
# #1711 taught the ledger input to read its own `done:` markers, but filtered
# them AFTER `rows[:limit]`, so a marked row still spent the per-tick budget. On
# 2026-09-29 that had the input silent for 20.9 hours: the slice was three marked
# rows at composite 0.05 while three UNMARKED 0.05 rows sat directly behind them,
# and six ledger-moving ticks logged `enqueued 0 ledger items (3 eligible, 3
# already mined)`. Markers never expire and every autoresearch round mints new
# 0.05 rows, so the head-of-line block refilled itself as fast as it aged out.
# #1711's owed entry ruled the fix: skip a marked row during the scan, and leave
# the per-tick budget where it is.
# ------------------------------------------------------------------------


def test_the_marker_filter_runs_before_the_slice_not_after(tmp_path, monkeypatch):
    """Clause 1: three marked rows at 0.05 must not eat a `limit=3` slice.

    The fixture is the live block itself — three rounds of one task at composite
    0.05, worst-first, with an unmarked 0.05 and an unmarked 0.40 behind them —
    because the bug was never "marked rows are offered" (that was #1711, and it is
    fixed) but "marked rows are offered IN PLACE OF the unmarked ones" at the head
    of the slice.
    """
    _fixture_ledger(tmp_path, monkeypatch, [
        _ledger_row("BASELINE_1", "bench_x", 0.05, round_id="R_1"),
        _ledger_row("BASELINE_2", "bench_x", 0.05, round_id="R_2"),
        _ledger_row("BASELINE_3", "bench_x", 0.05, round_id="R_3"),
        _ledger_row("BASELINE_4", "bench_y", 0.05, round_id="R_4"),
        _ledger_row("BASELINE_5", "bench_z", 0.40, round_id="R_5"),
    ])
    # Seeded as the literal the watermark holds, not via `BM._ledger_key`: a test
    # that built the key with the producer's own helper could not notice the
    # producer changing the format, which is the #625 one-sided contract.
    done = {f"ledger:bench_x:R_{n}" for n in (1, 2, 3)}

    # Positive control on the fixture, and the acceptance check's failing half:
    # ask the SAME selector with no marker set and the three marked rows come back
    # as the whole slice. If this ever stops being true, the block below is
    # grading nothing.
    unfiltered = [(r["task_id"], r["round_id"])
                  for r in BM._recent_ledger_losers(days=7, limit=3)]
    assert unfiltered == [("bench_x", "R_1"), ("bench_x", "R_2"), ("bench_x", "R_3")], (
        f"the fixture no longer reproduces the live block: with no marker set the "
        f"slice is {unfiltered}, not the three marked 0.05 rows")

    rows = BM._recent_ledger_losers(days=7, limit=3, done=done)

    assert [(r["task_id"], r["round_id"]) for r in rows] == [
        ("bench_y", "R_4"), ("bench_z", "R_5")], (
        "a marked row still consumed budget in the slice: the scan returned "
        f"{[(r['task_id'], r['round_id']) for r in rows]}")
    # Two rows out of a budget of three, because only two unmarked losers exist:
    # the fix spends the budget on rows worth mining, it does not invent rows.
    assert len(rows) == 2, (
        f"`limit=3` against two unmarked losers returned {len(rows)} rows — a "
        "wider slate here is the selection-budget change #1711 did not authorise")


async def test_an_unmarked_loser_behind_a_marked_head_reaches_the_queue(
        tmp_path, monkeypatch, q):
    """The same block at the queue's own seam, since a selector that fixes itself
    while the enqueue path still drops the row fixes nothing.

    Driven through `enqueue_if_due` with real watermarks and a real ledger file:
    20.9 hours of production ticks is the evidence that the live tree behaves this
    way, and no stub can reproduce that.
    """
    _fixture_ledger(tmp_path, monkeypatch, [
        _ledger_row("BASELINE_1", "bench_x", 0.05, round_id="R_1"),
        _ledger_row("BASELINE_2", "bench_x", 0.05, round_id="R_2"),
        _ledger_row("BASELINE_3", "bench_x", 0.05, round_id="R_3"),
        _ledger_row("BASELINE_4", "bench_y", 0.05, round_id="R_4"),
        _ledger_row("BASELINE_5", "bench_z", 0.40, round_id="R_5"),
    ])
    for n in (1, 2, 3):
        q.wm_set(BM.NAME, f"done:ledger:bench_x:R_{n}",
                 json.dumps({"why": "mined"}))

    await BM.enqueue_if_due(q, {})

    pairs = sorted((i.payload["loser_task_id"], i.payload["round_id"])
                   for i in _ledger_items(q))
    assert pairs == [("bench_y", "R_4"), ("bench_z", "R_5")], (
        f"the tick offered {pairs}: a marked head is still displacing the "
        "unmarked rows behind it")


async def test_a_tick_that_skips_marked_rows_logs_the_number_skipped(
        tmp_path, monkeypatch, q, caplog):
    """Clause 4: the number has to stay in `server.err`, because that line is how
    the silence was found at all — every ledger-moving tick from 2026-09-27 to
    2026-09-29 logged `(3 eligible, 3 already mined)` and nothing else did.

    With the filter inside the scan the caller can no longer count what it never
    received, so the scan reports it. Which side carries the number is the decision
    #1774's owed entry leaves to the landing sha: the caller's line keeps an honest
    eligible-only count, and the skip count comes from the only party that can see
    it.
    """
    _fixture_ledger(tmp_path, monkeypatch, [
        _ledger_row("BASELINE_1", "bench_x", 0.05, round_id="R_1"),
        _ledger_row("BASELINE_2", "bench_x", 0.05, round_id="R_2"),
        _ledger_row("BASELINE_3", "bench_x", 0.05, round_id="R_3"),
        _ledger_row("BASELINE_4", "bench_y", 0.05, round_id="R_4"),
    ])
    for n in (1, 2, 3):
        q.wm_set(BM.NAME, f"done:ledger:bench_x:R_{n}",
                 json.dumps({"why": "mined"}))

    with caplog.at_level(logging.INFO, logger="lloyd-workers.bench_mine"):
        await BM.enqueue_if_due(q, {})

    skip = [r.getMessage() for r in caplog.records
            if "already mined" in r.getMessage() or "already-mined" in r.getMessage()]
    assert skip, (
        "a tick that dropped three already-mined rows logged nothing about it — "
        "the quiet tick went illegible, which is how this bug ran for two days")
    assert any("3" in m for m in skip), (
        f"the log says rows were skipped but not how many: {skip}")
    assert [i.payload["round_id"] for i in _ledger_items(q)] == ["R_4"], (
        "the surviving row is not the unmarked one")


async def test_an_all_marked_ledger_tick_says_so_in_the_log(
        tmp_path, monkeypatch, q, caplog):
    """The all-marked case on its own, against the real selector.

    The spy-based test above proves the tick enqueues nothing and asks once; this
    proves the tick is still EXPLAINABLE, with the count the scan itself saw. Zero
    rows returned and no line in `server.err` is indistinguishable from a ledger
    with no losers at all — the ambiguity #1711's log line existed to remove.
    """
    _fixture_ledger(tmp_path, monkeypatch, [
        _ledger_row("BASELINE_1", "bench_x", 0.05, round_id="R_1"),
        _ledger_row("BASELINE_2", "bench_x", 0.05, round_id="R_2"),
    ])
    for n in (1, 2):
        q.wm_set(BM.NAME, f"done:ledger:bench_x:R_{n}",
                 json.dumps({"why": "mined"}))

    with caplog.at_level(logging.INFO, logger="lloyd-workers.bench_mine"):
        await BM.enqueue_if_due(q, {})

    assert _ledger_items(q) == [], "an all-marked slate enqueued anyway"
    skip = [r.getMessage() for r in caplog.records
            if "already mined" in r.getMessage() or "already-mined" in r.getMessage()]
    assert skip and any("2" in m for m in skip), (
        f"an all-marked tick logged {skip}; it must name the 2 rows it skipped")


def test_the_ledger_marker_key_has_one_definition(tmp_path):
    """The enqueue side and the retire side must build the same bytes, and a
    second copy of the format is how they drift apart (#625's shape again, one
    side renamed). `_item_key` is what every marker write goes through, so the
    pair-builder it uses is pinned against the literal the marker tests seed."""
    assert BM._ledger_key("bench_x", "R_1") == "ledger:bench_x:R_1"
    assert BM._item_key(_item({"loser_task_id": "bench_x", "round_id": "R_1"})) \
        == BM._ledger_key("bench_x", "R_1")
    # And the run input's key is untouched by this: it is a bare run_id, which
    # is why one `done:` set can serve both inputs without them colliding.
    assert BM._item_key(_item({"run_id": "run_24_20260908_235954"})) \
        == "run_24_20260908_235954"


def test_the_id_the_writer_mints_is_the_id_the_selector_selects(tmp_path, monkeypatch):
    """The seam test: writer process → ledger.jsonl → this selector.

    Typing `BASELINE_123` by hand would still pass if the writer's format
    changed, which is the failure that cost 3,893 rows. So ask the real
    `materialize_baseline` for an id and feed the selector what it returned.
    """
    from types import SimpleNamespace
    from scripts.autoresearch import variant_sandbox as VS

    variants = tmp_path / "variants"
    variants.mkdir()
    cfg = SimpleNamespace(paths=SimpleNamespace(variants_dir=variants))
    baseline_id, _overlay = VS.materialize_baseline(cfg)
    assert baseline_id.startswith("BASELINE_"), baseline_id

    _fixture_ledger(tmp_path, monkeypatch,
                    [_ledger_row(baseline_id, "bench_004_shape_gate", 0.42)])
    assert [r["task_id"] for r in BM._recent_ledger_losers()] == ["bench_004_shape_gate"]


# ---------------------------------------------------------------------------
# bench-mine: one key bounds BOTH inputs (#1712)
#
# `enqueue_if_due` resolved `max_enqueue_per_tick` into `limit` and handed it to
# `_enqueue_failed_runs` alone, while the ledger input called
# `_recent_ledger_losers()` bare and so ran at that function's own signature
# defaults `days=7, limit=5`. The input the key skipped is the one that fires:
# `workers.db` held 139 `bench-mine` items all-time on 2026-09-28, 138 of kind
# `mine` against 1 of `mine-run`, and bucketing `enqueued_at` into 10-second
# ticks gave {1:5, 2:5, 3:2, 4:2, 5:22} — 24 of the 36 ticks above the declared
# per-tick budget of 3, 22 of them at the selector's own 5.
#
# These go through `enqueue_if_due`, not the selector: the bug was one caller
# forgetting to pass a value, and calling `_recent_ledger_losers(days=…, limit=…)`
# directly would pin the selector's arithmetic while the live path stayed bare.
# ---------------------------------------------------------------------------


def _five_unmarked_losers() -> list[dict]:
    """Five qualifying baseline losers, one per `(task, round)` pair.

    Five and not three because clause 2 has to tell a cap of 3 from the
    selector's own default of 5: a fixture of three enqueues three whether the
    cap is threaded or not, and the test would pass on the bug.

    Distinct `task_id`s because the enqueue dedup key is
    `bench-mine:{task_id}:{round_id}` — five rounds of one task would survive
    dedup too, but distinct tasks are also what keeps a dropped item visible as
    a count rather than as a collision. Scores 0.10-0.50 are all under the 0.6
    loser line, and `_ledger_row`'s default `trace_status="success"` is what
    makes each row usable signal rather than a harness failure (#625).
    """
    return [_ledger_row(f"BASELINE_{n}", f"bench_cap_{n}", 0.1 * n, round_id=f"R_{n}")
            for n in range(1, 6)]


def _ledger_items(q) -> list:
    return [i for i in q.list_items(source=BM.NAME) if i.kind == BM.KIND_LEDGER]


async def test_max_enqueue_per_tick_bounds_the_ledger_input(tmp_path, monkeypatch, q):
    """Clause 1: an operator who sets the key to 1 gets one item, not five.

    Pre-fix this enqueued 3-or-more: `limit` never reached the ledger selector,
    so the key's value was read and thrown away on this input.
    """
    _fixture_ledger(tmp_path, monkeypatch, _five_unmarked_losers())

    await BM.enqueue_if_due(q, {"max_enqueue_per_tick": 1})

    items = _ledger_items(q)
    assert len(items) == 1, (
        f"`max_enqueue_per_tick: 1` enqueued {len(items)} {BM.KIND_LEDGER} items. "
        "The clause allows at most one; exactly one is asserted so the test cannot "
        "pass by the ledger input going quiet instead of being capped.")


async def test_the_default_cap_bounds_the_ledger_input_too(tmp_path, monkeypatch, q):
    """Clause 2: with no key set, the ledger input answers to
    `MAX_ENQUEUE_PER_TICK` (3), not to the selector's own default 5.

    The other half of the knob being half-read: an operator who sets nothing
    still gets the declared budget, because the declaration is now the value the
    caller passes rather than a number the caller ignores.
    """
    assert BM.MAX_ENQUEUE_PER_TICK < 5, (
        "five fixture rows only discriminate a cap of 3 from the selector's "
        f"default 5 while MAX_ENQUEUE_PER_TICK is below 5; it is "
        f"{BM.MAX_ENQUEUE_PER_TICK}, so this test can no longer tell the two apart")

    _fixture_ledger(tmp_path, monkeypatch, _five_unmarked_losers())

    await BM.enqueue_if_due(q, {})

    items = _ledger_items(q)
    assert len(items) == BM.MAX_ENQUEUE_PER_TICK == 3, (
        f"`src_cfg={{}}` enqueued {len(items)} {BM.KIND_LEDGER} items against the "
        f"declared {BM.MAX_ENQUEUE_PER_TICK}: the bare call is back to running at "
        "the selector's own default")


async def test_the_ledger_window_is_failure_window_days(tmp_path, monkeypatch, q):
    """Clause 3: the ledger input's window IS `FAILURE_WINDOW_DAYS`, not the `7`
    in the selector's signature.

    `days` threading is behaviour-neutral at the shipped value (signature
    default 7 == `FAILURE_WINDOW_DAYS` 7), so no fixture built at 7 days can
    tell a threaded call from a bare one. The only observable pin is to move the
    constant and watch the slate move with it: monkeypatched to 30, a loser
    stamped 10 days ago becomes eligible, and the same fixture under the shipped
    7 must not offer it.

    Both halves run against one ledger, with a fresh loser in it as the positive
    control — an empty offered set under 7 would otherwise read as the window
    working when it is really the mtime gate never opening.
    """
    from datetime import datetime, timedelta, timezone

    old = _ledger_row("BASELINE_900", "bench_window_old", 0.30, round_id="R_old",
                      created_at=(datetime.now(timezone.utc)
                                  - timedelta(days=10)).isoformat())
    fresh = _ledger_row("BASELINE_901", "bench_window_fresh", 0.40, round_id="R_fresh")
    ledger = _fixture_ledger(tmp_path, monkeypatch, [old, fresh])

    await BM.enqueue_if_due(q, {})
    offered = {i.payload["loser_task_id"] for i in _ledger_items(q)}
    assert offered == {"bench_window_fresh"}, (
        f"under the shipped FAILURE_WINDOW_DAYS=7 the offered set was {offered}; "
        "the fresh control must be offered and the 10-day-old loser must not be")

    monkeypatch.setattr(BM, "FAILURE_WINDOW_DAYS", 30)
    import os
    os.utime(ledger, (time.time() + 10, time.time() + 10))   # re-open the mtime gate
    await BM.enqueue_if_due(q, {})

    offered = {i.payload["loser_task_id"] for i in _ledger_items(q)}
    assert offered == {"bench_window_fresh", "bench_window_old"}, (
        f"with FAILURE_WINDOW_DAYS monkeypatched to 30 the offered set was "
        f"{offered}: the ledger input is still reading the selector's literal 7, "
        "so widening the constant would widen nothing")


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
    bench mean by noise, which is why a candidate is kept only inside the band.

    The dated count that used to sit in this docstring claimed four live tasks
    were pinned at that floor; measured 2026-09-28 against the research ledger,
    none of the 11 tasks carrying `BASELINE_*` rows has a mean composite of 0.00
    (lowest is bench_019 at 0.013). How many sit on the floor is a query against
    a moving store, not a fact to carry in prose (#1710).

    These two calls pass no `task=`, so they also pin the fallback: given a file
    that IS a bench task, `_load_candidate` still resolves it.
    """
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
# Capability-edge calibration (#1710): trials are spent on the mined TASK
# ---------------------------------------------------------------------------
#
# Every one of the 119 notes staged between 2026-09-23 and 2026-09-28 recorded
# `calibration.status: ok` with means of 0.51 to 1.00 and 118 of them `in_band:
# true`, because `_load_candidate` resolved the staged FILE and a staged note's
# first front matter block is `write_staging_note`'s envelope — `source`,
# `confidence`, `review_status` — while the mined task sits in the body. So the
# dict in front of the judge had no `objective_checks`, `_score_objective` awards
# that layer in full (judge.py: "no objective layer -> full marks"), and with
# `composite = 0.5*objective + 0.5*rubric` the mean could not fall below 0.50
# while EDGE_BAND opens at 0.05: the lower half of the band was arithmetically
# unreachable and the one rejection on record was a mean of exactly 1.0. The
# gate could only ever reject at the very top.
#
# The test above escaped this for the same reason every earlier one did: it
# handed `trials` a synthetic series, and a series does not care what task it was
# scored against. These nodes assert on the dict the trials were handed.


def _staged_note(dirn: Path, *, body: str = _CANDIDATE) -> Path:
    """A staged note: the staging envelope first, the mined task in the body.

    Written by hand into `dirn` rather than through `write_staging_note`, whose
    directory is the live staging root a human reads. The envelope's key set is
    pinned against the real writer by
    `test_the_test_helper_still_mirrors_the_real_staging_note`, because this
    file's whole claim is about that shape.
    """
    dirn.mkdir(parents=True, exist_ok=True)
    envelope = {
        "source": "bench-mine",
        "confidence": 0.5,
        "review_status": "pending",
        "rationale": "baseline loss on bench_020",
        "source_refs": ["~/obsidian/lloyd/bench/bench_020_x.md"],
        "generated_at": "2026-09-28T00:00:00+00:00",
    }
    p = dirn / "101010-mined-run-24-pipeline-timeout.md"
    p.write_text(
        "---\n" + yaml.dump(envelope, default_flow_style=False, allow_unicode=True)
        + "---\n\n" + body + "\n", encoding="utf-8")
    return p


def _envelope_keys(path: Path) -> list[str]:
    """The keys of a note's FIRST front matter block — what a loader reads."""
    raw = path.read_text(encoding="utf-8")
    assert raw.startswith("---"), "a staged note must open with front matter"
    end = raw.find("\n---\n", 3)
    return sorted((yaml.safe_load(raw[3:end]) or {}).keys())


def test_the_test_helper_still_mirrors_the_real_staging_note(tmp_path, monkeypatch):
    """The node that keeps the eight below honest.

    Every claim this file makes about the bug is a claim about the SHAPE of a
    staged note: envelope first, mined task in the body. So the helper is
    compared against the real `write_staging_note`, run into a redirected
    STAGING_ROOT — front-matter keys and body position — and the path-based
    loader is shown reading the envelope out of BOTH files. A helper allowed to
    drift into a file the loader parses correctly would un-pin every node here
    without a single one of them failing.
    """
    import workers.sources._common as _c

    root = tmp_path / "staging"
    monkeypatch.setattr(_c, "STAGING_ROOT", root)
    real = _c.write_staging_note(
        "bench-mine", "mined-run-24-pipeline-timeout", _CANDIDATE,
        confidence=0.5, rationale="baseline loss on bench_020",
        source_refs=["~/obsidian/lloyd/bench/bench_020_x.md"])
    mine = _staged_note(tmp_path / "mine")

    assert _envelope_keys(real) == _envelope_keys(mine), (
        "the helper's envelope no longer matches what write_staging_note emits")

    raw = real.read_text(encoding="utf-8")
    assert raw.index("id: bench_200") > raw.index("source: bench-mine"), (
        "the mined task no longer comes AFTER the envelope in a real note — the "
        "bug this section describes is gone and these nodes should be retired")

    for path in (real, mine):
        got = BM._load_candidate(path)
        assert got is not None and "confidence" in got and not got.get("prompt"), (
            f"{path.name}: the first front matter block is no longer the envelope, "
            "so the fixture below is no longer reproducing the shipped failure")


def _trial_recorder(score: float):
    """A `trials` stand-in that records the tasks it was asked to run."""
    seen: list[list[dict]] = []

    async def _trials(tasks, **kw):
        seen.append(list(tasks))
        return [score] * len(tasks)

    _trials.seen = seen
    return _trials


async def test_the_trials_are_spent_on_the_mined_task_not_its_envelope(tmp_path):
    """The gate is a measurement of the candidate or it is nothing.

    Before/after, this is the probe from the item: the dict in front of the
    judge must carry the candidate's own id, its prompt and its two
    `objective_checks`, so `_score_objective` has failing-able checks to grade
    instead of an empty list to award.
    """
    from scripts.autoresearch.judge import _score_objective

    path = _staged_note(tmp_path)
    # Why the parameter exists: read the same file the way the old code did and
    # the envelope is what comes back, with no prompt and no checks.
    envelope = BM._load_candidate(path)
    assert envelope is not None and not str(envelope.get("prompt") or "").strip(), (
        "a staged note's first front matter block stopped being the envelope, so "
        "this node no longer describes what the fix had to work around")
    assert _score_objective(envelope, {})[0] == 1.0, (
        "the award of an empty objective layer changed — the arithmetic below is "
        "written against it")

    trials = _trial_recorder(0.5)
    result = await BM.calibrate_candidate(
        path, task=BM._candidate_frontmatter(_CANDIDATE), runs=10, trials=trials)

    assert len(trials.seen) == 1 and len(trials.seen[0]) == 10
    task = trials.seen[0][0]
    assert task["id"] == "bench_200_mined_run_24_pipeline_timeout", (
        f"trials ran against {task.get('id')!r}, not the mined candidate")
    assert str(task.get("prompt") or "").strip(), "trials ran with no prompt"
    assert len(task["objective_checks"]) == 2, (
        "the objective layer is being awarded again, which is what pinned every "
        "calibration mean at or above 0.50")
    # The clause's own check: a candidate that declares failing-able checks does
    # not get the objective half handed to it.
    assert _score_objective(task, {"tool_calls": []})[0] < 1.0
    assert result["task_id"] == "bench_200_mined_run_24_pipeline_timeout"
    assert result["status"] == "ok" and result["in_band"] is True


async def test_a_task_that_cannot_discriminate_spends_zero_trials(tmp_path):
    """No GPU trial is spent on a dict that cannot produce a discriminating mean.

    Three shapes, all of which arrived as `status: ok` in the staged corpus: no
    prompt, an empty `objective_checks`, and no `objective_checks` at all. The
    trials callable must never be reached, and the note must not read as a
    verdict — `in_band: None` with an `error` that names what is missing.
    """
    base = BM._candidate_frontmatter(_CANDIDATE)
    cases = {
        "blank prompt": {**base, "prompt": "   "},
        "empty checks": {**base, "objective_checks": []},
        "no checks key": {k: v for k, v in base.items() if k != "objective_checks"},
    }
    path = _staged_note(tmp_path)
    for name, task in cases.items():
        trials = _trial_recorder(0.5)
        result = await BM.calibrate_candidate(path, task=task, runs=10, trials=trials)
        assert trials.seen == [], f"{name}: trials were spent anyway"
        assert result["status"] == "error" and result["in_band"] is None
        assert result["error"], f"{name} was refused with no reason recorded"
        assert ("prompt" in result["error"] if "prompt" in name
                else "objective_checks" in result["error"]), result["error"]
        assert result["composites"] == [] and result["mean"] is None

    # Positive control, in the same call shape: a complete candidate does spend
    # its trials, so the four refusals above are not a refusal of everything.
    ok = _trial_recorder(0.5)
    passed = await BM.calibrate_candidate(path, task=base, runs=10, trials=ok)
    assert len(ok.seen) == 1, "the guard stopped a calibratable candidate too"
    assert passed["status"] == "ok" and passed["composites"], passed["error"]


async def test_the_staged_note_records_which_task_the_mean_describes(tmp_path,
                                                                     monkeypatch):
    """End to end: a human reading the note sees what the calibration measured.

    The note is the whole reason the calibration exists — the promotion step
    reads the file, not the run record — so the id goes into the front matter
    beside the mean, and it is the parsed candidate's id rather than the file
    stem the loader would otherwise substitute.
    """
    root = tmp_path / "pending"
    # `write_staging_note` is imported into bench_mine's namespace but resolves
    # its directory through `_common.STAGING_ROOT` at call time, so the one
    # redirect that keeps this test out of the live staging tree is on `_common`.
    monkeypatch.setattr(C, "STAGING_ROOT", root)
    runs = _bm_inputs(tmp_path, monkeypatch)
    p = _run_file(runs, "run_24_20260908_235954")
    monkeypatch.setattr(BM, "run_prompt_on_primary", _candidate_turn(_CANDIDATE))
    trials = _trial_recorder(0.5)
    monkeypatch.setattr(BM, "_bench_composites", trials)

    out = await BM.execute(_item(payload={
        "run_id": "run-24", "run_path": str(p), "task_id": "24",
        "summary": "timed out"}))
    assert out["status"] == "success", out

    notes = sorted(root.rglob("*.md"))
    assert len(notes) == 1
    fm = yaml.safe_load(notes[0].read_text(encoding="utf-8").split("---")[1])
    cal = fm["calibration"]
    assert cal["task_id"] == "bench_200_mined_run_24_pipeline_timeout", (
        f"the note records a mean for {cal.get('task_id')!r}")
    assert cal["status"] == "ok" and cal["in_band"] is True
    assert trials.seen and trials.seen[0][0]["id"] == cal["task_id"], (
        "the note names a task the trials were not run against")
    # The substituted id was the other half of the bug's evidence trail: a mean
    # recorded under the file stem could never be looked up in the ledger.
    assert cal["task_id"] != notes[0].stem


def test_the_band_premise_is_stated_without_a_dated_task_count():
    """#705 removed the dated counts from this module's prose; #1710 found two
    left, one in `calibrate_candidate`'s docstring and one in the test above it,
    both claiming that four of the live bench tasks were pinned at a mean
    composite of zero. Measured 2026-09-28 against the research ledger: 11 tasks
    carry BASELINE_* rows and none of them scores 0.00 (lowest is bench_019 at
    0.013). A count of a moving store inside a code comment is not a fact, it is
    a claim that quietly becomes false.

    Asserted over the two files a reader of this gate actually opens — the module
    and this test — by the same grep the item names.
    """
    import subprocess

    # Each needle is assembled from fragments: a guard that spells the phrase it
    # is hunting would find its own file and fail for the wrong reason — which is
    # exactly how this node read on its first run.
    for needle in ("of the " "eleven live " "tasks",
                   "eleven " "live tasks",
                   "sit at exactly " "0.00"):
        out = subprocess.run(
            ["git", "grep", "-n", needle, "--", "workers/", "tests/"],
            cwd=str(Path(BM.__file__).resolve().parents[2]),
            capture_output=True, text=True).stdout
        assert not out.strip(), f"{needle!r} is still in this area's prose:\n{out}"

    doc = BM.calibrate_candidate.__doc__ or ""
    assert "EDGE_BAND" in doc, (
        "the premise the band encodes must stay stated; only the count goes")
    assert "0.00" not in doc and "eleven" not in doc
    assert "envelope" in doc, (
        "the docstring must say why the parsed task is a parameter — a later "
        "reader who 'simplifies' it back to a path re-opens #1710")


# ---------------------------------------------------------------------------
# Worker turns: config source and budget
# ---------------------------------------------------------------------------


def test_a_worker_turn_reads_the_merged_config_not_the_file():
    """`config.yaml` off disk skips `${VAR}` expansion, the canary overlay,
    and — the one that bites — the `data/tool_overrides.yaml` merge, which is
    the live authority for what is switched off. A tool disabled from the
    Tools page stayed advertised to every worker turn."""
    # #529 moved the merged-config read out of `run_prompt_on_primary` and into
    # the builder that BOTH in-process worker turn shapes now call, because the
    # state-carried turn must not be able to drift from the transcript turn on
    # tool policy. The assertion follows the code — and gets stricter rather
    # than looser: what actually matters is that no in-process worker turn can
    # build its own RunOptions at all.
    src = inspect.getsource(C._worker_run_options)
    code = "\n".join(
        line for line in src.splitlines() if not line.strip().startswith("#")
    ).split('"""')[2]  # drop the docstring too — both discuss the old way
    assert "from app.config import CONFIG" in code
    assert "yaml.safe_load" not in code, "a worker turn is re-reading config.yaml"
    assert "config.yaml" not in code
    for fn in (C.run_prompt_on_primary, C.run_prompt_with_run_state):
        body = inspect.getsource(fn)
        assert "_worker_run_options(" in body, \
            f"{fn.__name__} no longer goes through the shared builder"
        assert "RunOptions(" not in body, \
            f"{fn.__name__} builds its own tool policy"


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
    from app import autonomy
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


#: Every assertion below reads the heading off the renderer itself and the tool
#: names off the caller's value, so a block that renders with a stale list fails
#: the name assertion while the heading assertion holds — the drift this pins.
DENY_HEADING = C.DENIED_TOOLS_HEADING


async def test_the_deny_list_is_named_in_the_turn_it_denies(worker_turn_post):
    """#1066: the refusal said "disabled by configuration" and the turn had no
    idea it existed, so it retried the same denied tool. In
    `20260912_173635_deepresearch_360b` that cost a call and a reasoning turn on
    the *same* tool twice; 22 occurrences across 09-09 to 09-12 came from the
    two sources that pass a list, and zero from any source that does not.

    The list must arrive *as the value*, not as prose written next to it: the
    tool name below is one no source declares, so nothing could have typed it.
    """
    await C.run_prompt_in_session(
        "classify the item", title="t", source="autotriage",
        extra_disallowed=["Zzz_Tool", "Yyy_Tool"])

    text = worker_turn_post[0]["text"]
    assert DENY_HEADING in text, "the turn was never told a deny list exists"
    assert "Zzz_Tool" in text and "Yyy_Tool" in text, text
    # And nothing else: a stale `Bash` pasted into the renderer would render for
    # every worker, which is the per-source restriction the deny list exists to
    # keep per-source.
    assert "Bash" not in text, "the block names a tool this turn was not denied"
    assert text.startswith("classify the item"), "the job's own prompt was displaced"


async def test_the_digest_turn_is_told_its_own_list_and_not_the_research_one(
    worker_turn_post,
):
    """The other half of #1066's 22 refusals — 10 of them `Bash`, 3 `Edit` — and
    the reason one renderer has to read the caller's value rather than one
    shared sentence: this source denies `Bash`, `Edit`, `Task` and the browser
    actuation tools, and keeps `Read`, `Grep` and `Glob`, which is the exact
    opposite of deep-research's list. A single hand-written paragraph about
    "the worker tools" would be wrong for one of the two, whichever it named.
    """
    from workers.sources import youtube_digest as Y

    await C.run_prompt_in_session(
        "digest this video\n", title="probe", source=Y.NAME,
        extra_disallowed=list(Y.DISALLOWED))

    text = worker_turn_post[0]["text"]
    _, heading, block = text.partition(C.DENIED_TOOLS_HEADING)
    # #2269 bounds the object of this assertion: an envelope-compiled turn is
    # told its capability set in a sentence beside the refusals, and that
    # sentence names exactly the tools the loop below requires to be ABSENT from
    # the deny block. The refusal statement is what the assertion is about, so it
    # ends where the capability sentence begins.
    block = block.split(C.CAPABILITY_HEADING)[0]
    assert block, "the digest turn was never told its deny list"
    for name in ("Bash", "Edit", "Task", "http_request", "browser_click"):
        assert name in block, name
    for name in ("Read", "Grep", "Glob", "Write"):
        assert name not in block, f"{name} is available to this source"


async def test_a_turn_with_no_deny_list_gets_no_such_block(worker_turn_post):
    """The chat and ambient paths pass no `extra_disallowed`, and the skills
    library is shared by every agent — a per-worker restriction rendered onto a
    chat turn is the opposite of what that source's deny list is for.

    Asserted as equality on the text, not absence of a phrase: the block is
    *absent* for a turn with no list, not merely empty.
    """
    out = await C.run_prompt_in_session("classify the item", title="t",
                                        source="autotriage")

    assert worker_turn_post[0]["text"] == "classify the item"
    assert out["stop_reason"] == "end_turn", "the stub did not close the turn"


def test_the_direct_worker_turn_is_told_the_policy_it_is_under():
    """The second path, same fix. `_worker_run_options` is where the automod and
    grant-mint bans are baked in, and until #1066 they were the only ban in the
    system enforced without being announced — `test_worker_turns_cannot_drive_the_loop`
    pins that a worker cannot start a round, which is not the same as a worker
    knowing it cannot."""
    from workers.sources._common import WORKER_AUTOMOD_BAN

    opts = C._worker_run_options(20, source="bench-mine")
    for name in (*WORKER_AUTOMOD_BAN, "grant_create"):
        assert name in opts.system_prompt, name
    assert DENY_HEADING in opts.system_prompt
    # The names appear exactly as dispatch holds them. This list carries both
    # the bare and the `mcp__lloyd-mcp__` spelling of every ban, because
    # `_pre_dispatch` matches on the exact string; a renderer that rewrote one
    # into the other would promise a refusal the gate does not perform.
    assert opts.system_prompt.count("mcp__lloyd-mcp__automod_start") == 1


def test_a_worker_session_is_marked_as_one(tmp_path, monkeypatch):
    """`sessions_io.NON_USER_PLATFORMS` is what keeps a worker turn from being
    mistaken for the user's session by the morning brief and by
    session-distill alike."""
    # Written by `sessions_io.create_session` now — one writer for every
    # non-chat session — and conftest points that at a scratch directory.
    import app.sessions_io as sio
    sid = C.new_worker_session(title="t", source="autotriage")
    data = json.loads((sio.SESSIONS_DIR / f"{sid}.json").read_text())

    from app.sessions_io import is_user_session
    assert data["platform"] == "worker"
    assert not is_user_session(data)
    assert data["inner_voice"] is True


# ---------------------------------------------------------------------------
# bench-mine: the turn budget is config, carried in the payload (#896)
#
# `max_turns=8` was a literal at both call sites, so no config key could move
# it, and 112 of 115 all-time failures to 2026-09-18 were `turns=9` against it.
# The six sibling session sources already put `int(src_cfg.get("max_turns",
# DEFAULT))` in the payload and read it back at execute; this is that
# convention, copied.
# ---------------------------------------------------------------------------


async def test_bench_mine_carries_the_configured_turn_budget_in_every_payload(
        tmp_path, monkeypatch, q):
    """#896 clause 1: both inputs carry the key, and no key means today's 8."""
    runs = _bm_inputs(tmp_path, monkeypatch)
    _run_file(runs, "run_24_20260908_235954")
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text('{"x": 1}\n', encoding="utf-8")
    monkeypatch.setattr(BM, "_recent_ledger_losers", lambda *a, **k: [
        {"task_id": "bench_003", "composite_score": 0.2, "round_id": "r1"}])

    await BM.enqueue_if_due(q, {"max_turns": 12})
    items = q.list_items(source=BM.NAME)
    assert {i.kind for i in items} == {BM.KIND_RUN, BM.KIND_LEDGER}, items
    assert all(i.payload["max_turns"] == 12 for i in items), [i.payload for i in items]

    q2 = WorkQueue(tmp_path / "workers-default.db")
    monkeypatch.setattr(BM, "_recent_ledger_losers", lambda *a, **k: [
        {"task_id": "bench_004", "composite_score": 0.2, "round_id": "r2"}])
    await BM.enqueue_if_due(q2, {})
    items = q2.list_items(source=BM.NAME)
    assert len(items) == 2
    assert all(i.payload["max_turns"] == BM.DEFAULT_MAX_TURNS == 8 for i in items)


def _recording_turn(seen: list):
    async def _turn(prompt, *a, **k):
        seen.append(k.get("max_turns"))
        return C.TurnResult(text="", stop_reason="max_turns", num_turns=4)
    return _turn


async def test_bench_mine_runs_both_paths_with_the_budget_the_item_carries(
        tmp_path, monkeypatch, q):
    """#896 clause 2: no literal remains at either call site."""
    runs = _bm_inputs(tmp_path, monkeypatch)
    _bm_queue(monkeypatch, q)
    p = _run_file(runs, "run_24_20260908_235954")
    seen: list = []
    monkeypatch.setattr(BM, "run_prompt_on_primary", _recording_turn(seen))

    await BM._mine_run_failure(_item({"run_path": str(p), "run_id": "run_24_20260908_235954",
                                      "max_turns": 3}))
    await BM._mine_ledger_loser(_item({"loser_task_id": "bench_003", "composite_score": 0.2,
                                       "max_turns": 3}, kind=BM.KIND_LEDGER))
    assert seen == [3, 3], seen


async def test_bench_mine_item_enqueued_before_the_key_runs_at_the_default(
        tmp_path, monkeypatch, q):
    """#896 clause 3: a config change does not strand already-queued work."""
    runs = _bm_inputs(tmp_path, monkeypatch)
    _bm_queue(monkeypatch, q)
    p = _run_file(runs, "run_24_20260908_235954")
    seen: list = []
    monkeypatch.setattr(BM, "run_prompt_on_primary", _recording_turn(seen))

    result = await BM.execute(_item({"run_path": str(p), "run_id": "run_24_20260908_235954"}))
    assert result["status"] == "failed", result
    result = await BM.execute(_item({"loser_task_id": "bench_003", "composite_score": 0.2},
                                    kind=BM.KIND_LEDGER))
    assert result["status"] == "failed", result
    assert seen == [BM.DEFAULT_MAX_TURNS, BM.DEFAULT_MAX_TURNS], seen


def test_bench_mine_config_names_the_turn_budget():
    """Human clause 1: the operator's value is in the tracked config block."""
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    src = cfg["workers"]["sources"]["bench-mine"]
    assert int(src["max_turns"]) >= BM.DEFAULT_MAX_TURNS


# ---------------------------------------------------------------------------
# session-distill: the same #896 treatment (#1460)
#
# `max_turns=15` was a literal at the one call site, so the budget the
# arch-review called wrong could not move without a deploy.
# ---------------------------------------------------------------------------


async def test_session_distill_carries_the_configured_turn_budget(tmp_path, monkeypatch, q):
    monkeypatch.setattr(SD, "SESSIONS_DIR", tmp_path)
    _session(tmp_path, "20260906_quiet_one", age_seconds=7200)
    await SD.enqueue_if_due(q, {"max_turns": 22})
    items = q.list_items(source=SD.NAME)
    assert len(items) == 1 and items[0].payload["max_turns"] == 22, [i.payload for i in items]

    q2 = WorkQueue(tmp_path / "workers-default.db")
    _session(tmp_path, "20260906_quiet_two", age_seconds=7200)
    await SD.enqueue_if_due(q2, {})
    items = q2.list_items(source=SD.NAME)
    assert items and all(i.payload["max_turns"] == SD.DEFAULT_MAX_TURNS == 15 for i in items)


async def test_session_distill_runs_with_the_budget_the_item_carries(tmp_path, monkeypatch, q):
    import workers.queue as Q
    monkeypatch.setattr(Q, "_queue_instance", q, raising=False)
    seen: list = []
    monkeypatch.setattr(SD, "run_prompt_on_primary", _recording_turn(seen))

    await SD.execute(_item({"session_path": str(tmp_path / "a.json"), "max_turns": 7}))
    # Queued before the key existed: runs at the default, never stranded.
    await SD.execute(_item({"session_path": str(tmp_path / "b.json")}))
    assert seen == [7, SD.DEFAULT_MAX_TURNS], seen


def test_session_distill_config_names_the_turn_budget():
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    src = cfg["workers"]["sources"]["session-distill"]
    assert int(src["max_turns"]) == 15


# ---------------------------------------------------------------------------
# #1567 — the queue row must not present a run record that is not on disk
#
# `artifact_path` was assembled from `run_id` alone and never looked at the
# filesystem, so all 97 scheduled-task `runs.artifact_path` values in
# `~/lloyd-data/workers.db` assert a file the writer never verified exists. That
# is how task #77's `run_77_20260922_150638.md` came to be cited from an Activity
# Log while `autonomy-runs/77/` does not exist at all. The path string is kept
# byte-identical in both outcomes — it is a path, and `pool.normalize_result`
# copies it into `runs.artifact_path`, which the health route and the retention
# sweep join on; the verdict is the new `meta.artifact_absent` key beside it.
# ---------------------------------------------------------------------------

def _scheduled_result(run_id: str = "run_77_20260101_000000") -> dict:
    return {"status": "success", "success": True, "run_id": run_id,
            "response_preview": "done", "error": None, "meta": {}}


@pytest.fixture
def sched(monkeypatch, tmp_path):
    """`scheduled_task.execute` with everything but the artifact check stubbed.

    The health gate, `run_task`, the task-file lookup and the Discord notifier are
    all production seams that would spend a real run; `_artifact_on_disk`'s view of
    `app.paths.DATA_ROOT` is redirected to `tmp_path` so the check is exercised
    against a real filesystem rather than mocked — the point of the clause is that
    somebody looks at disk.
    """
    from workers.sources import scheduled_task as ST
    from app import paths

    monkeypatch.setattr(ST, "_vllm_healthy", lambda *a, **k: True)
    monkeypatch.setattr(ST, "_model_health_url", lambda m: "http://stub/health")
    monkeypatch.setattr("app.autonomy.run_task",
                        lambda *a, **k: asyncio.sleep(0, _scheduled_result()))
    monkeypatch.setattr("app.autonomy._find_task_file", lambda tid: None)
    monkeypatch.setattr("app.discord_notify._discord_notify_task_complete",
                        lambda *a, **k: None, raising=False)
    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path)
    return ST, tmp_path


def _execute(ST, task_id: int = 77) -> dict:
    return asyncio.run(ST.execute(_item({"task_id": task_id})))


def test_a_run_record_on_disk_records_the_path_and_says_nothing_about_absence(sched):
    """The 99% case is a no-op: same string, no new key. A check that reports
    success on every row tells a reader nothing and gets ignored."""
    ST, root = sched
    (root / "autonomy-runs" / "77").mkdir(parents=True)
    (root / "autonomy-runs" / "77" / "run_77_20260101_000000.md").write_text(
        "---\nstatus: success\n---\n\nworked\n", encoding="utf-8")

    out = _execute(ST)

    assert out["artifact_path"] == "autonomy-runs/77/run_77_20260101_000000.md"
    assert "artifact_absent" not in out["meta"], out["meta"]


def test_a_run_record_missing_from_disk_is_marked_absent_and_keeps_its_path(sched):
    """The clause. Nothing is deleted or blanked from the row — the row keeps the
    path it would have had, so the pair (path, `artifact_absent: True`) says both
    where the record belongs and that it is not there. Before this, the row and the
    Activity Log were the only two surfaces and both claimed the file."""
    ST, root = sched
    (root / "autonomy-runs" / "77").mkdir(parents=True)   # dir exists, file does not

    out = _execute(ST)

    assert out["artifact_path"] == "autonomy-runs/77/run_77_20260101_000000.md"
    assert out["meta"].get("artifact_absent") is True, out["meta"]


def test_a_zero_byte_run_record_counts_as_absent(sched):
    """A truncated write on a full mount leaves a file that `exists()` answers true
    for. `is_file()` alone would call that a record, which is the same false claim
    in a new place."""
    ST, root = sched
    p = root / "autonomy-runs" / "77" / "run_77_20260101_000000.md"
    p.parent.mkdir(parents=True)
    p.write_text("", encoding="utf-8")

    assert _execute(ST)["meta"].get("artifact_absent") is True


def test_the_check_resolves_against_the_data_root_and_not_the_cwd(monkeypatch, tmp_path):
    """`artifact_path` is DATA_ROOT-relative, so checking it against the process
    cwd reports 96 of 97 real artifacts missing — a false alarm that gets the check
    switched off, which is worse than the bug. Pinned by making the two roots
    disagree: an artifact under the data root answers present even with a cwd
    pointing at a tree that has no such file."""
    from workers.sources import scheduled_task as ST
    from app import paths

    root = tmp_path / "data"
    monkeypatch.chdir(tmp_path)                        # cwd has no autonomy-runs/
    monkeypatch.setattr(paths, "DATA_ROOT", root)
    assert not Path("autonomy-runs/77/x.md").exists(), "positive control: cwd lacks it"

    assert ST._artifact_on_disk("autonomy-runs/77/x.md") is False
    (root / "autonomy-runs" / "77").mkdir(parents=True)
    (root / "autonomy-runs" / "77" / "x.md").write_text("record", encoding="utf-8")
    assert ST._artifact_on_disk("autonomy-runs/77/x.md") is True
    assert ST._artifact_on_disk("") is False


# ---------------------------------------------------------------------------
# #1743 — a distill run names the transcript its facts came from, and counts
# the ones that landed with nowhere
#
# The prompt told the turn to call `fact_add` (`:246`/`:249` before this round)
# and never once mentioned `source_doc`; the MCP handler read the parameter as
# absent and wrote `NULL` through, so 9 facts reached `facts_idx` unattributed —
# un-revertable, because the only handle on a fact is the document it names.
# Clause 1 is the instruction the run was never given; clause 3 is the leak
# surfacing on the run that caused it instead of in tomorrow's health report.
# ---------------------------------------------------------------------------

#: The session id the fake distill turn reports and writes its facts under —
#: the value #1709 stamps onto `facts_idx.session_id`, which is what makes a
#: run able to count its own writes at all.
_DISTILL_SESSION = "20260928_112400_distill"


@pytest.fixture
def fact_store(tmp_path, monkeypatch):
    """A real fact tree and index under `tmp_path`.

    The count has to be read back out of the store the facts were really
    written to: a stubbed index would agree with whatever the source claimed,
    which is the same thing the turn's own prose was already doing.
    """
    from agent_mcp import _shared, facts as FACTS, retrieval
    from app import kg_store

    root = tmp_path / "facts"
    root.mkdir()
    monkeypatch.setattr(_shared, "FACTS_ROOT", root)
    monkeypatch.setattr(FACTS, "FACTS_ROOT", root)
    monkeypatch.setattr(retrieval, "FACTS_ROOT", root)
    monkeypatch.setattr(_shared, "_entity_dirs_cache", None)
    kg_store.configure(tmp_path / "kg.sqlite")
    return root


def _distill_turn(seen_prompts: list, add: list[dict], *, ok_text: str = "## Struggles\n- none\n",
                  session: str = _DISTILL_SESSION):
    """A distill turn that writes through the real `fact_add`, under its own id.

    Each entry of `add` is the kwargs of one call, so a test can put a
    source-less fact in the middle of an otherwise clean run and see what the
    run then reports about it. `write_gate: off` keeps the dedupe gate out of
    these tests — they are about provenance, and it asks a model.
    """
    from agent_mcp import facts as FACTS
    from agent_mcp._task_registry import current_session_id

    async def _turn(prompt, *a, **k):
        seen_prompts.append(prompt)
        token = current_session_id.set(session)
        try:
            for kwargs in add:
                FACTS._fact_add(dict(entity="Tidewell Relay", category="state",
                                     write_gate="off", **kwargs))
        finally:
            current_session_id.reset(token)
        return C.TurnResult(text=ok_text, stop_reason="stop", num_turns=4,
                            session_id=session)
    return _turn


def _arm_success_path(monkeypatch, tmp_path, q):
    """Keep a successful distill inside `tmp_path`: the staging note and the
    `done:` watermark both live outside the source otherwise."""
    import workers.queue as Q
    monkeypatch.setattr(Q, "_queue_instance", q, raising=False)
    monkeypatch.setattr(SD, "write_staging_note",
                        lambda **kw: tmp_path / "note.md")


async def test_the_distill_prompt_names_the_transcript_as_the_fact_source(tmp_path, monkeypatch):
    """Clause 1: the instruction every leaked fact is missing."""
    seen: list = []
    monkeypatch.setattr(SD, "run_prompt_on_primary", _distill_turn(seen, []))

    await SD.execute(_item({"session_path": str(tmp_path / "20260927_chat.json")}))

    assert len(seen) == 1, seen
    prompt = seen[0]
    assert str(tmp_path / "20260927_chat.json") in prompt, (
        "the path has to be in the prompt literally — the turn reads a "
        "placeholder as an instruction to make one up")
    assert "source_doc" in prompt, "the field name itself, not a paraphrase of it"


async def test_the_distill_prompt_tells_the_turn_what_an_unsourced_call_costs(tmp_path, monkeypatch):
    seen: list = []
    monkeypatch.setattr(SD, "run_prompt_on_primary", _distill_turn(seen, []))

    await SD.execute(_item({"session_path": str(tmp_path / "20260927_chat.json")}))

    assert "fact_add" in seen[0] and "refuse" in seen[0].lower(), (
        "a turn that does not know the call is refused retries it with a made-up path")


async def test_a_distill_run_reports_the_facts_it_wrote_with_no_source(
        tmp_path, monkeypatch, q, fact_store):
    """Clause 3: one clean write and one leak, and the run says so itself."""
    seen: list = []
    turn = _distill_turn(seen, [
        {"fact": "the relay pins its control port at 9453",
         "provenance": "EXTRACTED", "source_doc": "sessions/20260927_chat.json"},
        {"fact": "the relay was rewired in September"},
    ])
    monkeypatch.setattr(SD, "run_prompt_on_primary", turn)
    _arm_success_path(monkeypatch, tmp_path, q)

    out = await SD.execute(_item({"session_path": str(tmp_path / "20260927_chat.json")}))

    assert out["status"] == "success", out
    assert out["meta"]["facts_without_source"] == 1, out["meta"]


async def test_a_distill_run_that_stamped_every_fact_reports_zero(
        tmp_path, monkeypatch, q, fact_store):
    """The pair that makes the count able to fail: 0 and 1 both come from the
    store, so neither is a constant the source prints on its way past."""
    seen: list = []
    turn = _distill_turn(seen, [
        {"fact": "the relay pins its control port at 9453",
         "provenance": "EXTRACTED", "source_doc": "sessions/20260927_chat.json"},
        {"fact": "the relay was rewired in September",
         "provenance": "EXTRACTED", "source_doc": "sessions/20260927_chat.json"},
    ])
    monkeypatch.setattr(SD, "run_prompt_on_primary", turn)
    _arm_success_path(monkeypatch, tmp_path, q)

    out = await SD.execute(_item({"session_path": str(tmp_path / "20260927_chat.json")}))

    assert out["meta"]["facts_without_source"] == 0, out["meta"]


async def test_a_failed_distill_still_reports_what_its_turn_wrote(
        tmp_path, monkeypatch, q, fact_store):
    """A turn that dies at `max_turns` can still have written facts first, and
    the leak does not stop being a leak because the run failed."""
    seen: list = []
    turn = _distill_turn(seen, [{"fact": "the relay was rewired in September"}],
                         ok_text="")
    monkeypatch.setattr(SD, "run_prompt_on_primary", turn)
    monkeypatch.setattr("workers.queue._queue_instance", q, raising=False)

    out = await SD.execute(_item({"session_path": str(tmp_path / "20260927_chat.json")}))

    assert out["status"] == "failed", out
    assert out["meta"]["facts_without_source"] == 1, out["meta"]


def test_the_count_abstains_when_the_store_cannot_be_read(monkeypatch):
    """`None` and `0` are not the same answer, and only one of them is honest
    when the index would not open. A zero here reads as a clean run to the
    health report that would otherwise notice."""
    from app import kg_store

    def _boom(*a, **k):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(kg_store, "store", _boom)
    assert SD._facts_without_source(_DISTILL_SESSION) is None
    assert SD._facts_without_source("") is None, (
        "a run with no session id has no write record; that is unknown, not clean")


# ---------------------------------------------------------------------------
# #1684 — the dispatch gate's primary leg is config-derived, not a literal
#
# `workers/sources/scheduled_task.py` hardcoded `http://127.0.0.1:8096/health` and
# returned it for the `primary`/empty pin, for the post-`except` leg and for the
# no-URL call of `_vllm_healthy` — while every NAMED model's leg had always been
# read from config. So the one leg that gates the whole fleet was the one leg no
# config edit could move: point `models.primary.base_url` at another port and
# `enqueue_if_due` returns on every tick — zero runs enqueued — while the alert
# that finally arrives names a URL the primary never served. These tests pin the
# derived leg, the literal's retirement to the config-unreadable fallback, and the
# alert's provenance.
# ---------------------------------------------------------------------------


def test_the_gates_primary_leg_follows_the_configured_primary_endpoint(monkeypatch):
    """Clause 1: `models.primary.base_url` moves the fleet gate with it.

    Asserted for all three pins that used to short-circuit to the literal
    (`"primary"`, `""`, `None`), and against the literal itself: the point of the
    clause is that the gate no longer answers with its own copy of the port.
    """
    from app import config as cfg
    from workers.sources import scheduled_task as ST

    monkeypatch.setitem(cfg.MODEL_CONFIGS["primary"], "base_url",
                        "http://127.0.0.1:9999")
    for pin in ("primary", "", None):
        assert ST._model_health_url(pin) == "http://127.0.0.1:9999/health", (
            f"pin {pin!r} did not follow models.primary.base_url")
    assert ST._model_health_url("primary") != ST._VLLM_HEALTH_URL, (
        "the gate still answers with the literal it used to hardcode")
    assert ST._primary_health_target()[1] == "models.primary.base_url", (
        "the provenance key does not name the config key that answered")

    # The other config key the clause names: with `base_url` blanked the leg must
    # come from the model's own env, still not from the literal.
    monkeypatch.setitem(cfg.MODEL_CONFIGS["primary"], "base_url", "")
    monkeypatch.setitem(cfg.MODEL_CONFIGS["primary"]["env"],
                        "ANTHROPIC_BASE_URL", "http://127.0.0.1:9999")
    assert ST._model_health_url("primary") == "http://127.0.0.1:9999/health"
    assert ST._primary_health_target()[1] == (
        "models.primary.env.ANTHROPIC_BASE_URL")


def test_the_primary_leg_falls_back_to_the_literal_only_when_config_raises(monkeypatch):
    """Clause 2: the 8096 literal survives ONLY as the config-unreadable answer.

    `RuntimeError` out of the config helper is the unreadable-config case, and an
    empty answer is its quiet twin — an unknown `model.default` resolves to no
    endpoint at all. Both have to land on the literal, and say they did.
    """
    from app import config as cfg
    from workers.sources import scheduled_task as ST

    def _unreadable():
        raise RuntimeError("config.yaml could not be parsed")

    monkeypatch.setattr(cfg, "default_model_base_url_source", _unreadable)
    assert ST._model_health_url("primary") == "http://127.0.0.1:8096/health"
    assert ST._model_health_url("primary") == ST._VLLM_HEALTH_URL, (
        "the last-resort endpoint is no longer the literal the clause keeps")
    assert ST._primary_health_target()[1] == ST._VLLM_URL_FALLBACK_KEY, (
        "the fallback must name itself as the fallback, not as a config key")

    monkeypatch.setattr(cfg, "default_model_base_url_source",
                        lambda: ("", "models.nonexistent.base_url"))
    assert ST._primary_health_target() == (ST._VLLM_HEALTH_URL,
                                          ST._VLLM_URL_FALLBACK_KEY), (
        "an empty derived endpoint must fall back rather than probe ''")


def test_the_outage_alert_names_the_endpoint_it_probed_and_its_config_key(monkeypatch):
    """Clause 4: a port move has to be self-diagnosing in the alert itself.

    The outage is driven through the real `_note_vllm_outage` on an already-running
    outage (state aged past the threshold) so the sentence is the one discord
    carries. The service probe holds no streak, so nothing in the message can
    contribute a port — `:8096` absent is therefore a claim about THIS gate, and
    pre-fix it is exactly what the message contained.
    """
    import datetime as dt

    from app import config as cfg
    from workers import service_probe as sp
    from workers.sources import scheduled_task as ST

    monkeypatch.setitem(cfg.MODEL_CONFIGS["primary"], "base_url",
                        "http://127.0.0.1:9999")
    monkeypatch.setattr(sp, "_shared", sp.ServiceProbe())
    monkeypatch.setitem(ST._state, "vllm_down_logged", True)
    monkeypatch.setitem(ST._state, "vllm_down_alerted", False)
    monkeypatch.setitem(ST._state, "vllm_down_since",
                        dt.datetime.now(dt.timezone.utc)
                        - dt.timedelta(seconds=ST._VLLM_DOWN_ALERT_SECONDS + 60))

    alerts: list[str] = []

    async def _capture(msg):
        alerts.append(msg)

    monkeypatch.setattr(ST, "_alert", _capture)

    asyncio.run(ST._note_vllm_outage())

    assert len(alerts) == 1, f"the aged outage produced no alert: {alerts}"
    msg = alerts[0]
    assert "http://127.0.0.1:9999/health" in msg, (
        f"the alert does not name the endpoint that was probed: {msg}")
    assert "models.primary.base_url" in msg, (
        f"the alert does not name the config key it came from: {msg}")
    assert ":8096" not in msg, (
        f"the alert still quotes the retired literal: {msg}")


def _secondary_on_its_own_port(monkeypatch, tmp_path) -> str:
    """Point `app.autonomy.LLOYD_HOME` at a config.yaml whose secondary answers on
    :9991, and return the health URL that implies.

    `app.autonomy._get_model_env` re-reads `config.yaml` off disk rather than
    reading the in-memory `app.config.CONFIG`, so redirecting `LLOYD_HOME` is what
    actually moves a named model's endpoint — patching `MODEL_CONFIGS` is answered
    by the file and changes nothing (measured: the leg still returned :8091). The
    port is one no other test here uses, so a leg that collapsed onto the primary's
    endpoint or onto this module's literal could not satisfy the assertions below.
    """
    from app import autonomy, config as cfg

    (tmp_path / "config.yaml").write_text(
        "model:\n"
        "  default: primary\n"
        "models:\n"
        "  primary:\n"
        "    alias: primary\n"
        "    env:\n"
        "      ANTHROPIC_BASE_URL: http://127.0.0.1:8096\n"
        "  secondary:\n"
        "    alias: secondary\n"
        "    env:\n"
        "      ANTHROPIC_BASE_URL: http://127.0.0.1:9991\n",
        encoding="utf-8")
    monkeypatch.setattr(autonomy, "LLOYD_HOME", tmp_path)
    # Shipped config says `secondary_enabled: false`, which rewrites the alias to
    # primary; switch it on so the named leg is exercised at all — the same switch
    # `test_secondary_routes_to_primary_when_disabled` flips.
    monkeypatch.setitem(cfg.CONFIG, "secondary_enabled", True)
    return "http://127.0.0.1:9991/health"


def test_a_named_model_leg_still_resolves_to_its_own_configured_endpoint(
        monkeypatch, tmp_path):
    """Clause 5, first half: deriving the primary leg changed no named pin."""
    from workers.sources import scheduled_task as ST

    secondary_url = _secondary_on_its_own_port(monkeypatch, tmp_path)

    assert ST._model_health_url("secondary") == secondary_url
    assert ST._model_health_url("secondary") != ST._model_health_url("primary"), (
        "a named pin collapsed onto the primary endpoint")


def test_a_task_pinned_to_an_unhealthy_named_model_is_skipped_not_enqueued(
        monkeypatch, tmp_path, q):
    """Clause 5, second half: the per-task rule one level below the fleet gate.

    One real `enqueue_if_due` tick over one due task pinned to `secondary`, with
    the primary answering and the secondary refusing. The task must be skipped —
    enqueuing it would spend its attempts on a ConnectError retry loop, which is
    the failure the gate exists to prevent.
    """
    from app import autonomy
    from workers.sources import scheduled_task as ST

    secondary_url = _secondary_on_its_own_port(monkeypatch, tmp_path)
    monkeypatch.setattr(autonomy, "recover_stuck_tasks", lambda *a, **k: [])
    monkeypatch.setattr(autonomy, "get_due_tasks",
                        lambda *a, **k: [{"id": 555, "name": "secondary-pinned",
                                          "model": "secondary",
                                          "priority": "low"}])

    probed: list = []

    def _health(timeout: float = 4.0, url: str | None = None):
        probed.append(url)
        return url != secondary_url

    monkeypatch.setattr(ST, "_vllm_healthy", _health)

    asyncio.run(ST.enqueue_if_due(q, {"max_duration_seconds": 1800}))

    assert secondary_url in probed, (
        f"the pinned model's own endpoint was never probed: {probed}")
    assert q.list_items(source=ST.NAME) == [], (
        "a task pinned to an unhealthy model server was enqueued anyway")


# ── #2037 clause 2, second half: a task charged to its cap is never re-offered ──


def test_a_charged_to_max_retries_task_gets_no_row_until_the_rearm_is_due(
        monkeypatch, tmp_path, q):
    """One real tick over the two files that differ by the charge's own field.

    #74 is written exactly as `_record_failure` leaves a `max_retries: 3` task at
    its third death: `status: failed`, `failure_count: 3`, `failure_kind: task`,
    `next_run` absent (the disable branch nulls it) and `last_attempt` NOW. #75 is
    identical but `status: up_next` with `failure_count: 2` — one death short. Only
    #75 may be enqueued, and the tick's own recovery scan is left in place: it is
    production code, it runs first on every real tick, and it is what makes the
    silence finite rather than permanent.

    That scan is why #74's `last_attempt` is now and not three days ago. A retired
    task is re-armed by `_rearm_after_one_period` once a full declared period has
    passed since its last attempt (#1086: "a task retired at max_retries has no
    other way back"), so a #74 aged three days on an `hourly` frequency would be
    armed again by the tick's own first step and would legitimately enqueue. The
    bound #2037 buys is therefore ONE declared period of guaranteed silence per
    charge, not a permanent mute — and the second half of this node pins the release
    valve, because a node that asserted only the silence would silently be a node
    that asserts a task can never come back.
    """
    import datetime as dt
    from app import autonomy
    from workers.sources import scheduled_task as ST

    tasks = tmp_path / "autonomy"
    tasks.mkdir()
    now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    three_days_ago = (dt.datetime.now(dt.timezone.utc)
                      - dt.timedelta(days=3)).isoformat()
    for task_id, status, failures, last_attempt in (
            (74, "failed", 3, now_iso), (75, "up_next", 2, three_days_ago)):
        fm = {"id": task_id, "name": f"task{task_id}", "status": status,
              "frequency": "hourly", "priority": "low", "skill_name": "some-skill",
              "max_retries": 3, "failure_count": failures,
              "failure_kind": "task", "next_run": None,
              "last_attempt": last_attempt, "last_run": last_attempt}
        (tasks / f"{task_id}-task{task_id}.md").write_text(
            f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# task\n\n"
            "## Activity Log\n", encoding="utf-8")

    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", tasks)
    monkeypatch.setattr(autonomy, "AUTONOMY_RUNS_DIR", tmp_path / "autonomy-runs")
    monkeypatch.setattr(ST, "_vllm_healthy", lambda *a, **k: True)

    asyncio.run(ST.enqueue_if_due(q, {"max_duration_seconds": 1800}))

    enqueued = [str(item.payload.get("task_id"))
                for item in q.list_items(source=ST.NAME)]
    assert enqueued == ["75"], (
        f"enqueued {enqueued}: #75 is the live control that this tick CAN dispatch "
        "— if it is missing, the silence below proves nothing — and #74 is the "
        "retired file the charge exists to stop re-offering")

    # The release valve, through the same production scan the tick just ran. Aged
    # one full `hourly` period past its `last_attempt`, the re-arm is what puts a
    # retired task back; asserting it here keeps the claim above honest — one
    # period of silence per charge, and a way back after that.
    autonomy._update_task_field(74, last_attempt=three_days_ago)

    assert autonomy.recover_stuck_tasks() == [74], (
        "the re-arm that is supposed to be the only way back did not fire, which "
        "means a charged-to-the-cap task is muted forever rather than for one "
        "declared period")


def test_the_pool_charges_a_death_of_this_real_source(monkeypatch, tmp_path):
    """The opt-in seam, with the production source object on both sides of it.

    `test_workers_pool.py` proves the pool honours `CHARGE_TASK_ON_DEATH` on a
    stand-in source. What it cannot prove is that THIS module — the one the running
    aggregator registers — actually carries the attribute the pool looks for under
    that name, which is the half a rename here would drop silently and leave every
    scheduled-task death uncharged again. So the production module object is handed
    to the production reader, with nothing of either stubbed.
    """
    import datetime as dt
    from app import autonomy
    from workers import pool as POOL
    from workers.sources import scheduled_task as ST

    assert ST.CHARGE_TASK_ON_DEATH is True, (
        "the source no longer declares the opt-in the pool reads")

    tasks = tmp_path / "autonomy"
    tasks.mkdir()
    (tasks / "74-kg-mention-classifier.md").write_text(
        "---\n"
        + yaml.dump({"id": 74, "name": "kg-mention-classifier", "status": "up_next",
                     "frequency": "hourly", "priority": "low",
                     "skill_name": "kg-mention-classifier",
                     "max_retries": 3, "failure_count": 0}, sort_keys=False)
        + "---\n\n# kg\n\n## Activity Log\n", encoding="utf-8")
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", tasks)
    monkeypatch.setattr(autonomy, "AUTONOMY_RUNS_DIR", tmp_path / "runs")

    now = dt.datetime.now(dt.timezone.utc)
    meta = asyncio.run(POOL._death_meta(
        _item({"task_id": 74}), ST,
        ModuleNotFoundError("No module named 'app.discord_notify'"),
        run_id="run_scheduled-task_20261001_seam_0001", started_at=now.isoformat()))

    assert meta["failure_kind"] == "task", meta
    assert meta["task_budget_charged"] is True, (
        "production source, production reader, and no budget moved")
    fm = autonomy._parse_task_file(autonomy._find_task_file(74))
    assert fm["failure_count"] == 1, (
        f"the task file reads failure_count {fm['failure_count']} after a death of "
        "the real source that never reached a verdict")


def test_execute_marks_the_verdict_only_after_run_task_returns(monkeypatch, sched):
    """The one line that makes a post-verdict death unchargeable, pinned both ways.

    `execute()` calls `mark_task_verdict()` between `run_task` returning and the
    adapter's own post-processing. The marker is the whole of the contract between
    this source and `WorkerPool._death_meta`: a death after it is not the task's
    second attempt, and a death before it is. Both directions are asserted because
    either half can rot on its own — the marker moved above the `run_task` call and
    every unrecorded death becomes free again; never called and a task loses half its
    `max_retries` to one row.

    The spy stands in for the pool's reader; what is production here is WHEN the
    source calls it.
    """
    from app import autonomy

    ST, _ = sched
    marked: list = []
    monkeypatch.setattr("workers.pool.mark_task_verdict",
                        lambda: marked.append(1))

    _execute(ST)
    assert marked == [1], (
        "the source never told the pool its verdict had moved, so a death in "
        "post-processing would charge this attempt a second time")

    marked.clear()

    def _refuse(*_a, **_k):
        raise RuntimeError("model server unhealthy")

    monkeypatch.setattr(autonomy, "run_task", _refuse)
    try:
        asyncio.run(ST.execute(_item({"task_id": 77})))
    except RuntimeError:
        pass
    assert marked == [], (
        "the marker fired for an attempt that wrote no verdict, which is exactly "
        "the death #2037 exists to charge")
