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

from app import autonomy
from app.run_acceptance import (DispatchTrace, false_completion_rate, grade_run,
                                parse_acceptance, trace_from_transcript)
from scripts.autoresearch import judge
from scripts.step_conformance import steps_from_tools
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
        monkeypatch.setattr("app.prompt_builder.build_system_prompt", lambda **_kw: "SYS")
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
    # Wait for the item to settle, not just for the run row: the pool writes
    # the run before it marks the item, and cancelling in between left it
    # `running` under load (flaky on live main, ~1 in 6).
    for _ in range(100):
        await asyncio.sleep(0.05)
        if q.list_runs(source="scheduled-task") and not q.list_items(
                state="running", source="scheduled-task"):
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


# ── #2706: `artifact_written` — the mandated artifact, off the dispatch record ──
#
# Task 94's 2026-10-13 run never executed task 94. Its first action was an Edit
# of `~/obsidian/memory/mental-models.md`, it wrote neither
# `~/obsidian/autonomy/referential-integrity-latest.md` nor the dated copy, and
# it discharged every check the task declares: `tool_called: Bash` was paid for
# by 49 unrelated Bash calls, and the `regex` check matches a run that only
# recites the summary line. No declared check can see a path, so nothing in-run
# could tell — the missing artifact surfaced only in the T+1 trajectory replay.
# These four tests pin the check that can.

ARTIFACT = "referential-integrity-latest.md"
ARTIFACT_CHECK = {"objective_checks": [{"type": "artifact_written", "value": ARTIFACT}]}

#: The numbers `skills/referential-integrity-ledger/SKILL.md` itself prints as
#: its example, so this is exactly the recital the task's `regex` check cannot
#: tell apart from having done the work.
RECITED = ("referential integrity: run 44 | 200 cites, 12 dangling, 4 exempt, "
           "n/a newly dangling")

#: Task 94's real nightly pair, from its trajectory rows: the resolver writes
#: `-latest.md` through a `--report` flag, and the second command copies it to a
#: dated archive. The second is where the artifact basename appears as a path
#: capture, which is why the check fires on this run and not only on a `Write`.
TASK_94_COMMANDS = [
    'cd ~/lloyd && python3 -m scripts.maintenance.referential_integrity '
    f'--report ~/obsidian/autonomy/{ARTIFACT}; echo "EXIT=$?"',
    'mkdir -p ~/obsidian/autonomy/referential-integrity && '
    f'cp ~/obsidian/autonomy/{ARTIFACT} '
    '~/obsidian/autonomy/referential-integrity/$(date -u +%F).md && '
    'ls -1 ~/obsidian/autonomy/referential-integrity/*.md | wc -l',
    f"grep -m1 'newly dangling' ~/obsidian/autonomy/{ARTIFACT}",
]


def _stream(specs) -> list[dict]:
    """`[(name, args), …]` or `[(name, args, result), …]` as a harness event stream.

    The `tool_call` carries `args_dict` exactly as `app.harness.events.tool_call`
    emits it: the write target rides on the call, never on the result. Results
    are `ok` unless the spec names one.
    """
    events: list[dict] = []
    for i, spec in enumerate(specs, start=1):
        name, args = spec[0], spec[1]
        result = spec[2] if len(spec) > 2 else "ok"
        events += [{"type": "tool_call", "call_id": f"c{i}", "name": name,
                    "args_json": json.dumps(args), "args_dict": args},
                   {"type": "tool_result", "call_id": f"c{i}", "name": name,
                    "content": result}]
    return events


def _trace_stream(specs, text: str = RECITED) -> dict:
    trace = DispatchTrace()
    for e in _stream(specs):
        trace.observe(e)
    return trace.as_trace(text)


def _writes(trace: dict) -> list[str]:
    """Every artifact basename the trace's dispatched calls record, in order."""
    return [b for call in trace["tool_calls"] for b in call["writes"]]


def test_artifact_written_is_kept_and_graded_off_the_dispatch_record():
    """Clause 1: a task may declare it, and the verdict is the record's.

    `parse_acceptance` must keep it as a measurable check rather than file it
    `invalid` — an unlisted type is a task graded `acceptance_invalid` on every
    run, which is the same silence in a different costume. And the check must
    answer off the dispatch record: the recital of the exact summary line the
    task wants, with no write, is the run that fooled task 94's own checks.
    """
    spec = parse_acceptance({"acceptance": ARTIFACT_CHECK})
    assert spec["checks"] == ARTIFACT_CHECK["objective_checks"]
    assert spec["invalid"] == []

    off_task = _trace_stream(
        [("Edit", {"file_path": "/home/alansrobotlab/obsidian/memory/mental-models.md",
                   "old_string": "a", "new_string": "b"})])
    grade = grade_run({"acceptance": ARTIFACT_CHECK}, off_task)
    assert grade["grade"] == "graded_fail"
    assert grade["checks"] == [{"type": "artifact_written", "value": ARTIFACT,
                                "passed": False}]

    written = _trace_stream(
        [("Write", {"file_path": f"/home/alansrobotlab/obsidian/autonomy/{ARTIFACT}"})])
    assert grade_run({"acceptance": ARTIFACT_CHECK}, written)["grade"] == "graded_pass"


def test_forty_unrelated_bash_calls_do_not_discharge_it_and_a_redirect_equals_a_write():
    """Clause 2: the target is per call, so volume is not evidence.

    Forty dispatched `Bash` calls on unrelated commands is the 2026-10-13 shape
    and still fails, while one `Write` of a path and one `Bash >` of that same
    path satisfy the check identically — tool substitution must not move the
    verdict, which is the same rule the replay's step identity runs on.
    """
    busy = _trace_stream([("Bash", {"command": f"grep -rn memory_ceiling app/ | head -{i}"})
                          for i in range(40)],
                         text=RECITED)
    assert len(busy["tool_calls"]) == 40
    assert _writes(busy) == []
    assert grade_run({"acceptance": ARTIFACT_CHECK}, busy)["grade"] == "graded_fail"

    via_write = grade_run({"acceptance": ARTIFACT_CHECK}, _trace_stream(
        [("Write", {"file_path": f"/home/a/obsidian/autonomy/{ARTIFACT}"})]))
    via_bash = grade_run({"acceptance": ARTIFACT_CHECK}, _trace_stream(
        [("Bash", {"command": "python3 -m scripts.maintenance.referential_integrity "
                              f"--report /tmp/body.md > ~/obsidian/autonomy/{ARTIFACT}"})]))
    assert via_write["checks"] == via_bash["checks"] == [
        {"type": "artifact_written", "value": ARTIFACT, "passed": True}]


@pytest.mark.parametrize("command,expected", [
    (f"echo hi > ~/obsidian/autonomy/{ARTIFACT}", [ARTIFACT]),
    (f"touch ~/obsidian/autonomy/{ARTIFACT}", [ARTIFACT]),
    (f"cat /tmp/body.md | tee ~/obsidian/autonomy/{ARTIFACT}", [ARTIFACT]),
    (f"cp ~/obsidian/autonomy/{ARTIFACT} ~/archive/2026-10-14.md", [ARTIFACT]),
    # `cp`/`mv`/`install` contribute their FIRST path token, which for a copy
    # *into* the artifact is the source and not the destination. That asymmetry
    # is the published grammar of `scripts/step_conformance.py` (`WRITER_RE` plus
    # the token loop in `steps_from_tools`), reused rather than re-decided here so
    # the in-run grader and the replay cannot disagree (#2706 clause 4). It is
    # what makes task 94's nightly `cp latest → dated` discharge the artifact.
    ("cp /tmp/staged.md ~/obsidian/autonomy/%s" % ARTIFACT, ["staged.md"]),
    ("mv /tmp/staged.md ~/obsidian/autonomy/%s" % ARTIFACT, ["staged.md"]),
    ("install -m 644 /tmp/staged.md ~/obsidian/autonomy/%s" % ARTIFACT, ["staged.md"]),
    ("python3 -m scripts.maintenance.referential_integrity "
     f"--report ~/obsidian/autonomy/{ARTIFACT}", []),
    # The writer scan keeps going past a token whose basename the sanitizer
    # blanks, and stops at the first usable one. Breaking at the first token
    # containing a `/` instead reads `/tmp/preflight-$(date` as the write, gets
    # nothing usable from it, and the run's real step disappears — the shape
    # 9 of 8,770 corpus runs had when this was re-derived on 2026-10-14.
    ("nightly.sh 2>&1 | tee /tmp/preflight-$(date +%s).txt | head -3; "
     "ls -t /tmp/preflight-*.txt | head -1", ["preflight-*.txt"]),
])
def test_the_write_targets_captured_are_the_replays_own_grammar(command, expected):
    """Clause 2 and 4 together, one Bash command at a time.

    The declared `--report <path>` shape captures nothing: no redirect, no
    writer word, so a run that stops after the resolver is graded_fail today —
    the over-fire question the item owes a week of live traffic to settle.
    """
    assert _writes(_trace_stream([("Bash", {"command": command})])) == expected


def test_a_refused_write_never_discharges_it_and_no_record_is_not_measurable():
    """Clause 3: a refused call stays in `denied_calls` and wrote nothing.

    The rail refusing a write is the harness proving the artifact is *not*
    there, so it cannot also be the evidence that it is. And with no dispatch
    record there is no verdict either way: `artifact_written` sits in
    `TOOL_BEHAVIOUR_CHECKS` beside `tool_called`, so the empty trace records
    `not_measurable` — a run that merely recited the summary line in prose is
    excluded, not credited.
    """
    refused = _trace_stream(
        [("Write", {"file_path": f"/home/a/obsidian/autonomy/{ARTIFACT}"},
          "Tool call denied: protected path")])
    assert refused["tool_calls"] == []
    assert [d["name"] for d in refused["denied_calls"]] == ["Write"]
    assert grade_run({"acceptance": ARTIFACT_CHECK}, refused)["grade"] == "graded_fail"

    prose_only = {"final_text": RECITED, "tool_calls": [], "denied_calls": [],
                  "tool_trace_authoritative": False}
    grade = grade_run({"acceptance": ARTIFACT_CHECK}, prose_only)
    assert grade["grade"] == "not_measurable"
    assert grade["checks"] == [{"type": "artifact_written", "value": ARTIFACT,
                                "passed": None}]
    assert "artifact_written" in judge.TOOL_BEHAVIOUR_CHECKS


def test_the_basename_comparison_is_the_ones_the_replay_uses():
    """Clause 4: one basename grammar, so grader and replay cannot diverge.

    Comparison goes through `scripts/step_conformance.py`'s `normalized_basename`,
    which elides a date stamp so a nightly artifact is one name: a write of
    `conversation-patterns-2026-10-14.md` satisfies
    `artifact_written: conversation-patterns-<date>.md`. The stronger half is the
    second assertion — the same calls fed to the replay's own `steps_from_tools`
    must yield the same basename set, because two extractors is how a check and
    a T+1 report start grading different runs.
    """
    dated = {"objective_checks": [{"type": "artifact_written",
                                   "value": "conversation-patterns-<date>.md"}]}
    written = _trace_stream([("Write", {
        "file_path": "/home/alansrobotlab/obsidian/knowledge/conversation/"
                     "conversation-patterns-2026-10-14.md"})])
    assert grade_run({"acceptance": dated}, written)["grade"] == "graded_pass"

    trace = _trace_stream([("Bash", {"command": c}) for c in TASK_94_COMMANDS])
    assert grade_run({"acceptance": ARTIFACT_CHECK}, trace)["grade"] == "graded_pass"
    mine = sorted(set(_writes(trace)))
    replay = sorted(s[len("write:"):] for s in steps_from_tools(
        [{"name": name, "params_summary": {"command": cmd}, "sequence": i}
         for i, (name, cmd) in enumerate([("Bash", c) for c in TASK_94_COMMANDS])])
        if s.startswith("write:"))
    assert mine == replay == [ARTIFACT]


def _transcript(args_raw: str, name: str = "Bash") -> list[dict]:
    """One tool call and its result, as `app/run_recorder.py` writes a session.

    The seam that matters here: on disk the arguments are the JSON *string* that
    was replayed to the engine, not the `args_dict` the live event carried."""
    return [{"role": "user", "content": "run the sweep"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"call_id": "c1", "type": "function",
                 "function": {"name": name, "arguments": args_raw}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "3"},
            {"role": "assistant", "content": RECITED}]


def test_a_regraded_transcript_reaches_the_same_verdict_as_a_live_run():
    """`scripts/autonomy/regrade_runs.py` grades old runs off their transcript.

    If the rebuilt trace lost the arguments, the check would exist only for runs
    nobody has regraded — and re-grading the 2026-10-13 run is precisely what
    anyone who learns about it will do. Truncated arguments (a recorder's cut,
    or a model's unfinished call) are no write target, not a crash: the trace
    still has to rebuild.
    """
    nightly = grade_run({"acceptance": ARTIFACT_CHECK}, trace_from_transcript(
        _transcript(json.dumps({"command": TASK_94_COMMANDS[1]}))))
    assert nightly["grade"] == "graded_pass"
    unrelated = grade_run({"acceptance": ARTIFACT_CHECK}, trace_from_transcript(
        _transcript(json.dumps({"command": "grep -rn memory_ceiling app/ | head -5"}))))
    assert unrelated["grade"] == "graded_fail"
    cut = '{"command": "cp ~/obsidian/autonomy/' + ARTIFACT + ' /tmp/x'
    assert grade_run({"acceptance": ARTIFACT_CHECK},
                     trace_from_transcript(_transcript(cut)))["grade"] == "graded_fail"
