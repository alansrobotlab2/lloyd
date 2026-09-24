"""An autonomy run's acceptance is graded and recorded, and decides nothing (#623).

`run_task` writes `status="success"` as a literal once a turn returns text.
These tests pin the independent reading beside it: a task's declared
`acceptance:` block, graded by `scripts/autoresearch/judge.py`'s objective layer
against the run's dispatch record (`app/run_acceptance.py`), written onto the
run's meta and from there onto `runs.meta_json` — and nothing else changed:
same status, no requeue, one attempt.

The end-to-end tests drive the real `run_task` (harness stubbed to a scripted
event stream), the real `scheduled-task` adapter and the real pool worker loop
into a real `WorkQueue`, because #945 showed a hand-built dict cannot prove a
key survives a function it never passed through.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest
import yaml

import autonomy
from app.run_acceptance import (DispatchTrace, false_completion_rate, grade_run,
                                parse_acceptance)
from workers.pool import WorkerPool, normalize_result
from workers.queue import QueueItem, WorkQueue

CLAIM = "I ran Bash and wrote 12 entries to the handoff file."
BASH_CHECK = {"objective_checks": [{"type": "tool_called", "value": "Bash"}]}


def _events(*, bash: str | None, text: str = CLAIM) -> list[dict]:
    """A one-iteration run: optionally a Bash call (`bash` is its result
    content), then a final text block that claims Bash ran either way."""
    events: list[dict] = []
    if bash is not None:
        events += [{"type": "tool_call", "call_id": "c1", "name": "Bash",
                    "args_json": "{}", "summary": "Writing"},
                   {"type": "tool_result", "call_id": "c1", "name": "Bash",
                    "content": bash}]
    events += [{"type": "text_delta", "text": text},
               {"type": "assistant_message", "text": text, "tool_calls": [],
                "iteration": 1, "finish_reason": "stop"},
               {"type": "result", "stop_reason": "stop", "num_turns": 1, "usage": {}}]
    return events


def _trace(events: list[dict], text: str = CLAIM) -> dict:
    d = DispatchTrace()
    for e in events:
        d.observe(e)
    return d.as_trace(text)


# ── clause 1: the block, and its absence ────────────────────────────────────

def test_a_task_with_no_block_grades_no_acceptance_and_that_is_not_a_pass():
    assert parse_acceptance({"id": 1}) is None
    grade = grade_run({"id": 1}, _trace(_events(bash="ok")))
    assert grade == {"grade": "no_acceptance"}


def test_the_block_accepts_only_the_judges_own_check_types():
    spec = parse_acceptance({"acceptance": {
        "objective_checks": [{"type": "contains", "value": "12 entries"},
                             {"type": "file_exists", "value": "x.md"}],
        "rubric": ["accuracy"]}})
    assert spec["checks"] == [{"type": "contains", "value": "12 entries"}]
    assert spec["rubric"] == ["accuracy"]
    assert len(spec["invalid"]) == 1
    grade = grade_run({"acceptance": {"objective_checks": [
        {"type": "file_exists", "value": "x.md"}]}}, _trace(_events(bash="ok")))
    assert grade["grade"] == "acceptance_invalid"


def test_a_bare_list_is_read_as_the_objective_checks():
    spec = parse_acceptance({"acceptance": [{"type": "regex", "value": r"\d+ entries"}]})
    assert spec["checks"] == [{"type": "regex", "value": r"\d+ entries"}]


# ── clause 2: the dispatch record, not the prose ────────────────────────────

def test_claiming_bash_without_a_dispatch_fails_tool_called():
    grade = grade_run({"acceptance": BASH_CHECK}, _trace(_events(bash=None)))
    assert grade["grade"] == "graded_fail"
    assert grade["checks"] == [{"type": "tool_called", "value": "Bash", "passed": False}]


def test_the_same_text_with_a_real_bash_dispatch_passes():
    grade = grade_run({"acceptance": BASH_CHECK}, _trace(_events(bash="wrote 12")))
    assert grade["grade"] == "graded_pass"
    assert grade["score"] == 1.0


def test_a_bash_call_the_hook_refused_did_not_run():
    denied = _events(bash="Tool call denied: destructive command")
    trace = _trace(denied)
    assert trace["tool_calls"] == []
    assert [d["name"] for d in trace["denied_calls"]] == ["Bash"]
    assert grade_run({"acceptance": BASH_CHECK}, trace)["grade"] == "graded_fail"


def test_a_rubric_is_recorded_as_ungraded_not_scored():
    grade = grade_run({"acceptance": {**BASH_CHECK, "rubric": ["clarity"]}},
                      _trace(_events(bash="ok")))
    assert grade["rubric_ungraded"] == ["clarity"]


# ── clause 6: the switch, and the report ────────────────────────────────────

def test_grader_false_grades_nothing():
    assert grade_run({"grader": False, "acceptance": BASH_CHECK},
                     _trace(_events(bash=None))) is None
    assert grade_run({"grader": "false", "acceptance": BASH_CHECK},
                     _trace(_events(bash=None))) is None


def test_the_report_is_none_not_zero_for_a_source_nobody_graded(tmp_path):
    q = WorkQueue(tmp_path / "w.db")
    rows = [("a", "scheduled-task", {"acceptance_grade": {"grade": "graded_fail"}}),
            ("b", "scheduled-task", {"acceptance_grade": {"grade": "graded_pass"}}),
            ("c", "scheduled-task", {"acceptance_grade": {"grade": "no_acceptance"}}),
            ("d", "research", {})]
    for run_id, source, meta in rows:
        q.record_run(run_id=run_id, queue_id=None, source=source, status="success",
                     started_at="2026-09-24T00:00:00+00:00",
                     completed_at="2026-09-24T00:01:00+00:00",
                     duration_seconds=60, meta_json=json.dumps(meta))
    with sqlite3.connect(tmp_path / "w.db") as conn:
        rate = false_completion_rate(conn, since="2026-09-01")
    assert rate == {"scheduled-task": 0.5, "research": None}


# ── clauses 3, 4, 5: through the real run_task, adapter and pool ────────────

@pytest.fixture
def real_run(monkeypatch, tmp_path):
    """Real `run_task`, real adapter, real pool; only I/O around them stubbed."""
    runs_dir = tmp_path / "autonomy-runs"
    monkeypatch.setattr(autonomy, "AUTONOMY_RUNS_DIR", runs_dir)

    def _stub(task: dict, events: list[dict]):
        monkeypatch.setattr(autonomy, "_find_task_file", lambda tid: tmp_path / "t.md")
        monkeypatch.setattr(autonomy, "_parse_task_file", lambda p: task)
        monkeypatch.setattr(autonomy, "_load_skill_content", lambda s: "SKILL BODY")
        monkeypatch.setattr(autonomy, "_update_task_field", lambda *a, **k: None)
        monkeypatch.setattr(autonomy, "_append_activity_log", lambda *a, **k: None)
        monkeypatch.setattr(autonomy, "_get_model_env", lambda m: {})
        monkeypatch.setattr(autonomy, "_task_inner_voice", lambda t: False)
        monkeypatch.setattr("app.run_recorder.recording_enabled", lambda: False)

        async def _run_query(messages, options):
            for evt in events:
                yield evt

        class Opts:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        import app.discord_notify as discord_notify
        import app.harness as harness_mod
        import app.harness.mcp_pool as mcp_pool
        import workers.sources as sources
        import workers.sources.scheduled_task as scheduled_task

        monkeypatch.setattr(harness_mod, "run_query", _run_query)
        monkeypatch.setattr(harness_mod, "RunOptions", Opts)
        monkeypatch.setattr(mcp_pool, "DEFAULT_LLOYD_MCP_SERVERS", {}, raising=False)
        monkeypatch.setattr("prompt_builder.build_system_prompt", lambda **_kw: "SYS")
        monkeypatch.setattr("app.sessions_io.SESSIONS_DIR", tmp_path / "sessions")
        monkeypatch.setattr(scheduled_task, "_vllm_healthy", lambda *a, **k: True)
        monkeypatch.setattr(sources, "get_sources_config", lambda: {})

        async def _no_notify(*a, **k):
            return None

        monkeypatch.setattr(discord_notify, "_discord_notify_task_complete", _no_notify)

    return _stub, runs_dir


def _task(**extra) -> dict:
    task = {"id": 901, "name": "Handoff writer", "skill_name": "handoff",
            "status": "up_next", "timeout_seconds": 300}
    task.update(extra)
    return task


async def _through_the_pool(q: WorkQueue) -> dict:
    q.enqueue(source="scheduled-task", kind="run", payload={"task_id": 901})
    pool = WorkerPool(q, slots=1)
    pool._running = True
    worker = asyncio.create_task(pool._worker_loop("worker-0"))
    for _ in range(100):
        await asyncio.sleep(0.05)
        if q.list_runs(source="scheduled-task"):
            break
    pool._running = False
    worker.cancel()
    try:
        await worker
    except asyncio.CancelledError:
        pass
    return q.list_runs(source="scheduled-task")[0]


async def test_a_false_success_reaches_runs_as_success_graded_fail(real_run, tmp_path):
    """The item's own verification: a run that reports success while its
    declared check is false is readable as status='success' AND graded_fail
    with one SQL query — and the grade changed nothing about the item."""
    stub, runs_dir = real_run
    stub(_task(acceptance=BASH_CHECK), _events(bash=None))
    q = WorkQueue(tmp_path / "w.db")

    await _through_the_pool(q)

    with sqlite3.connect(tmp_path / "w.db") as conn:
        got = conn.execute(
            "SELECT status, json_extract(meta_json, '$.acceptance_grade.grade') "
            "FROM runs WHERE source = 'scheduled-task'").fetchall()
        items = conn.execute("SELECT state, attempts FROM queue").fetchall()
    assert got == [("success", "graded_fail")]
    # Grade-only: completed once, not requeued, one attempt.
    assert items == [("completed", 1)]
    # The run record a person reads carries the same grade.
    rec = next((runs_dir / "901").glob("run_*.md")).read_text(encoding="utf-8")
    fm = yaml.safe_load(rec.split("---\n", 2)[1])
    assert fm["status"] == "success"
    assert fm["acceptance_grade"]["grade"] == "graded_fail"


async def test_a_real_dispatch_reaches_runs_as_graded_pass(real_run, tmp_path):
    stub, _ = real_run
    stub(_task(acceptance=BASH_CHECK), _events(bash="wrote 12"))
    q = WorkQueue(tmp_path / "w.db")
    row = await _through_the_pool(q)
    assert row["status"] == "success"
    assert json.loads(row["meta_json"])["acceptance_grade"]["grade"] == "graded_pass"


async def test_grader_false_writes_the_row_it_wrote_before(real_run, tmp_path):
    stub, _ = real_run
    stub(_task(acceptance=BASH_CHECK, grader=False), _events(bash=None))
    q = WorkQueue(tmp_path / "w.db")
    row = await _through_the_pool(q)
    assert row["status"] == "success"
    assert "acceptance_grade" not in json.loads(row["meta_json"])


async def test_the_adapter_names_the_grade_so_its_whitelist_cannot_drop_it(monkeypatch):
    """Clause 4: a grade `run_task` returns outside `meta` still reaches the
    pool's meta — the adapter lifts it by name, as #945 had to for `claims`."""
    import app.discord_notify as discord_notify
    import workers.sources as sources
    import workers.sources.scheduled_task as scheduled_task

    async def _run(task_id, max_duration=None):
        return {"success": True, "status": "success", "task_id": task_id,
                "run_id": "run_901_x", "response_preview": "done", "meta": {},
                "acceptance_grade": {"grade": "graded_fail"}}

    async def _no_notify(*a, **k):
        return None

    monkeypatch.setattr(scheduled_task, "_vllm_healthy", lambda *a, **k: True)
    monkeypatch.setattr(autonomy, "_find_task_file", lambda tid: None)
    monkeypatch.setattr(autonomy, "run_task", _run)
    monkeypatch.setattr(sources, "get_sources_config", lambda: {})
    monkeypatch.setattr(discord_notify, "_discord_notify_task_complete", _no_notify)

    item = QueueItem(id=1, source="scheduled-task", kind="run", priority=50,
                     payload={"task_id": 901}, dedup_key=None, state="running",
                     attempts=1, enqueued_at="", claimed_at=None, claimed_by=None,
                     completed_at=None, error=None)
    out = await scheduled_task.execute(item)
    norm = normalize_result(item, out)
    assert norm["status"] == "success"
    assert norm["meta"]["acceptance_grade"] == {"grade": "graded_fail"}
