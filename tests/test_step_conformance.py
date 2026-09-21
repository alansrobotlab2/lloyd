"""Learned per-task step conformance (#673) — the read-only detector's clauses.

A worker run records one terminal status, so a run that silently skipped a mandatory stage
lands in `success`. These tests pin the four behaviours #673 promises on that store:

1. three trajectory rows for one `session_source: autonomy-task:N` publish an ordered
   expected step set, and a run whose ordered trace lacks a step every other run has is
   reported as a deviation that *names the step*;
2. a run recorded `status='success'` is scored rather than skipped, and the deviation carries
   that recorded status, so conformance is never inferred from it;
3. a step found missing while the run is inside the grace window is `pending` and becomes a
   deviation only on a re-check after the window lapses, while a step that merely appears
   later than its learned position raises no deviation;
4. every deviation names the runs its expectation was learned from, and a task with too thin
   a history is reported as having *no baseline*, never as clean.

`scripts/step_conformance.py` is a script, not an importable package (there is no
`scripts/__init__.py`, the same convention `tests/test_guard_vacuity.py` loads by), so the
module is loaded from its path. Every corpus here is a real trajectory JSONL plus a real
sqlite `runs` table read back through the same read-only loaders a nightly replay uses.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "step_conformance.py"
TASK = "autonomy-task:7"
# An artifact written before the step that consumes it. A benign re-order can put the read
# first, but no run may lack the write: the set is what the baseline learns.
REPORT = "/lloyd/_pipeline/reflection/report-latest.md"
SOURCE_FILE = "/lloyd/_pipeline/reflection/source-latest.md"


def _load_module():
    spec = importlib.util.spec_from_file_location("step_conformance", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sc = _load_module()


def _tool(name, *, path=None, command=None, seq=1):
    params = {"summary": f"call {seq}"}
    if path is not None:
        params["file_path"] = path
    if command is not None:
        params["command"] = command
    return {
        "name": name,
        "params_summary": params,
        "result_summary": "OK: 2 chars",
        "is_error": False,
        "error_source": None,
        "exit_code": None,
        "sequence": seq,
    }


def _calls(**_kwargs):
    """The shape every clean run of the fixture task takes: read the input, write the artifact."""
    return [
        _tool("Read", path=SOURCE_FILE, seq=1),
        _tool("Bash", command=f"wc -c {REPORT}", seq=2),
        _tool("Write", path=REPORT, seq=3),
    ]


def _broken(**_kwargs):
    """The same run with the artifact-writing call never made, and nothing else changed."""
    return [
        _tool("Read", path=SOURCE_FILE, seq=1),
        _tool("Bash", command=f"wc -c {REPORT}", seq=2),
    ]


def _shuffled(**_kwargs):
    """A benign variation: same steps, consumed before written. Order must not be the alarm."""
    return [
        _tool("Bash", command=f"wc -c {REPORT}", seq=1),
        _tool("Write", path=REPORT, seq=2),
        _tool("Read", path=SOURCE_FILE, seq=3),
    ]


RUNS_SCHEMA = (
    "CREATE TABLE runs (run_id TEXT PRIMARY KEY, queue_id TEXT, source TEXT, task_id TEXT,"
    " status TEXT, started_at TEXT, completed_at TEXT, duration_seconds REAL, summary TEXT,"
    " artifact_path TEXT, response_json TEXT, meta_json TEXT, claims_json TEXT)"
)


def _write_corpus(root: Path, specs) -> tuple[Path, Path]:
    """Materialise specs as a trajectory store + ledger, exactly the two stores the real
    detector joins. Each spec: run_id, session_key, status, calls, source, started/completed."""
    traj = root / "trajectories"
    traj.mkdir(parents=True, exist_ok=True)
    db = root / "workers.db"
    if db.exists():
        db.unlink()
    conn = sqlite3.connect(db)
    conn.execute(RUNS_SCHEMA)
    rows = []
    for spec in specs:
        session_key = spec["session_key"]
        conn.execute(
            "INSERT INTO runs (run_id, source, task_id, status, started_at, completed_at,"
            " duration_seconds, summary, artifact_path, response_json, meta_json, claims_json)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                spec["run_id"],
                spec.get("ledger_source", "scheduled-task"),
                str(spec.get("task_id", 7)),
                spec["status"],
                spec.get("started", "2026-09-10T05:00:00+00:00"),
                spec.get("completed", "2026-09-10T05:12:00+00:00"),
                720.0,
                "fixture run",
                None,
                None,
                json.dumps({"session_id": session_key, "session_ids": [session_key]}),
                None,
            ),
        )
        calls = spec["calls"]
        rows.append(
            {
                "session_key": session_key,
                "agent_id": "lloyd",
                "session_class": "autonomy",
                "session_source": spec.get("source", TASK),
                "timestamp": spec.get("timestamp", "2026-09-10T05:00:00"),
                "tool_count": len(calls),
                "error_count": 0,
                "has_errors": False,
                "tools": calls,
            }
        )
    conn.commit()
    conn.close()
    (traj / "2026-09-10.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return traj, db


def _corpus(root: Path, specs, **learn_kwargs):
    traj, db = _write_corpus(root, specs)
    traces = sc.load_traces(traj)
    runs = sc.load_runs(db)
    joined, unjoined = sc.join_runs(traces, runs)
    return sc.learn(joined, **learn_kwargs), joined, unjoined


def _specs(count: int, make=_calls, *, run_prefix="run_ok", broken_at: int | None = None):
    specs = []
    for i in range(count):
        faulty = broken_at == i
        specs.append(
            {
                "run_id": f"{run_prefix}_{i}",
                "session_key": f"2026091{i}_05000{i}_autonomy_{run_prefix}{i}",
                "status": "success",
                "calls": (_broken() if faulty else make()),
            }
        )
    return specs


# ---------------------------------------------------------------------------
# load boundary: the two stores the check reads
# ---------------------------------------------------------------------------


def test_load_runs_reads_the_recorded_status_and_session_ids_without_inferring_either(tmp_path):
    """The detector's only link to the ledger is this loader, so the boundary test asserts it
    reports what the row *holds*: a success row comes back success, and the session id is
    taken from `meta_json` exactly as stored, in stored order."""
    traj, db = _write_corpus(
        tmp_path,
        [
            {
                "run_id": "run_a",
                "session_key": "sess_a",
                "status": "success",
                "calls": _calls(),
                "started": "2026-09-10T04:00:00+00:00",
                "completed": "2026-09-10T04:30:00+00:00",
            }
        ],
    )
    runs = sc.load_runs(db)
    assert [r.run_id for r in runs] == ["run_a"]
    assert runs[0].status == "success"
    assert runs[0].task_id == "7"
    assert runs[0].started_at == "2026-09-10T04:00:00+00:00"
    assert runs[0].completed_at == "2026-09-10T04:30:00+00:00"
    assert runs[0].session_ids == ("sess_a",)

    # The same row's status is copied, never derived: rewriting it to `failed` in the ledger
    # changes only the reported status, never whether the steps are checked.
    conn = sqlite3.connect(db)
    conn.execute("UPDATE runs SET status='failed' WHERE run_id='run_a'")
    conn.commit()
    conn.close()
    assert sc.load_runs(db)[0].status == "failed"


def test_load_runs_opens_the_ledger_read_only(tmp_path, monkeypatch):
    """A nightly replay must not be able to write the ledger it is auditing, so this is tested
    through the loader rather than through a handle this file opened itself: the handle the
    loader opens is a `mode=ro` URI, and — the observable half, which a read-write handle could
    never imitate — it refuses to *materialise* a database that is not there."""
    seen = {}
    real_connect = sqlite3.connect

    def spy(*args, **kwargs):
        seen["args"] = [str(a) for a in args]
        seen.update(kwargs)
        return real_connect(*args, **kwargs)

    _traj, db = _write_corpus(tmp_path, _specs(1))
    monkeypatch.setattr(sc.sqlite3, "connect", spy)
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    assert [r.run_id for r in sc.load_runs(db)] == ["run_ok_0"]
    sc.load_runs(db)
    assert seen["args"] == [f"file:{db}?mode=ro"], seen
    assert seen["uri"] is True, seen
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before

    missing = tmp_path / "never-created.db"
    with pytest.raises(sqlite3.Error):
        sc.load_runs(missing)
    assert not missing.exists(), "a read-only reader must not materialise the ledger"


def test_the_fixture_ledger_schema_is_the_schema_the_worker_writes(tmp_path):
    """The join runs on `runs.run_id`, `task_id`, `status`, `started_at`, `completed_at` and
    `meta_json`. `workers/queue.py` defines that table as its DDL *plus* two additive
    migrations, so a fixture that copied only the DDL would be missing the very column the join
    needs — the fixture schema is therefore asserted against the real writer's column list."""
    queue_py = REPO_ROOT / "workers" / "queue.py"
    source = queue_py.read_text(encoding="utf-8")
    block = source[source.index("CREATE TABLE IF NOT EXISTS runs ("):].split(");")[0]
    declared = re.findall(r"^ {2}([a-z_]+)\s+(?:TEXT|INTEGER|REAL)", block, re.M)
    migrated = re.findall(r"ALTER TABLE runs ADD COLUMN ([a-z_]+) TEXT", source)
    expected = set(declared) | set(migrated)
    assert {"run_id", "source", "status", "started_at", "completed_at"} <= expected
    assert {"meta_json", "claims_json"} <= expected, "the additive migrations stopped matching"

    _traj, db = _write_corpus(tmp_path, [])
    have = {row[1] for row in sqlite3.connect(db).execute("PRAGMA table_info(runs)")}
    assert have == expected, f"fixture schema drifted from the writer: {have ^ expected}"
    assert sc.load_runs(db) == [], "the loader reads exactly these columns"


def test_a_row_the_real_extractor_wrote_reaches_the_detector(tmp_path):
    """The seam the whole design rests on: `_pipeline/trajectories/*.jsonl` is written by
    `scripts/extract-trajectories.py`, not by this diff. A row this file writes by hand proves
    the detector is self-consistent and nothing else, so the row under test is the output of the
    extractor's own `parse_session` — including the shape `scrub_params` gives a long shell
    command, which is a truncation this file does not imitate."""
    extractor_py = REPO_ROOT / "scripts" / "extract-trajectories.py"
    spec = importlib.util.spec_from_file_location("extract_trajectories_under_test", extractor_py)
    extract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extract)

    long_command = "python3 - <<'PY'\n# " + ("x" * 900) + "\nPY"
    session = {
        "session_id": "20260920_050000_autonomy_seed47",
        "session_start": "2026-09-20T05:00:00+00:00",
        "platform": "autonomy",
        "source": TASK,
        "messages": [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {
                "name": "Read",
                "arguments": json.dumps(
                    {"file_path": "/lloyd/skills/dream-consolidation/SKILL.md"})}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok", "stats": {"is_error": False}},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "function": {
                "name": "Bash",
                "arguments": json.dumps(
                    {"command": "cd /lloyd && touch /lloyd/agents/lloyd/.consolidate-lock"})}}]},
            {"role": "tool", "tool_call_id": "c2", "content": "ok", "stats": {"is_error": False}},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c3", "function": {
                "name": "Bash", "arguments": json.dumps({"command": long_command})}}]},
            {"role": "tool", "tool_call_id": "c3", "content": "ok", "stats": {"is_error": False}},
        ],
    }
    session_path = tmp_path / "20260920_050000_autonomy_seed47.json"
    session_path.write_text(json.dumps(session), encoding="utf-8")
    row = extract.parse_session(session_path)
    assert row is not None and row["session_source"] == TASK
    assert row["tools"][2]["params_summary"]["command"].endswith("chars]"), (
        "the extractor truncates a long command, and that truncation is the point of this test"
    )

    traj = tmp_path / "trajectories"
    traj.mkdir()
    (traj / "2026-09-20.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    traces = sc.load_traces(traj)
    assert len(traces) == 1
    steps = set(traces[0].steps)
    assert {"tool:Read", "read:SKILL.md", "tool:Bash"} <= steps, sorted(steps)
    assert "write:.consolidate-lock" in steps, (
        f"a lock stamp written by `touch` in a real extracted row must be a step: {sorted(steps)}"
    )
    # The truncated command contributes its tool call and invents no artifact: a step the
    # extractor could not have seen would become an expectation the detector never finds.
    assert sorted(s for s in steps if s.startswith("write:")) == ["write:.consolidate-lock"]


def test_join_pairs_a_trace_with_its_ledger_row_by_session_id(tmp_path):
    """The join is the seam between two stores that share no id: the trajectory's
    `session_key` against the run's `meta_json.session_ids`."""
    traj, db = _write_corpus(tmp_path, _specs(3))
    traces = sc.load_traces(traj)
    runs = sc.load_runs(db)
    joined, unjoined = sc.join_runs(traces, runs)
    assert unjoined == []
    assert len(joined) == 3
    for trace, run in joined:
        assert run is not None
        assert trace.session_key in run.session_ids
        assert trace.task_number == 7


def test_a_trace_with_no_ledger_row_is_still_scored_and_says_so(tmp_path):
    """Coverage is reported, not assumed: an unjoinable trace keeps its session key as its
    id and a `None` recorded status, and is still checked against its task's learned shape."""
    specs = _specs(3)
    traj, db = _write_corpus(tmp_path, specs)
    stray = {
        "session_key": "20260910_050000_autonomy_orphan",
        "agent_id": "lloyd",
        "session_class": "autonomy",
        "session_source": TASK,
        "timestamp": "2026-09-10T06:00:00",
        "tool_count": 2,
        "error_count": 0,
        "has_errors": False,
        "tools": _broken(),
    }
    with (traj / "2026-09-10.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(stray) + "\n")
    joined, unjoined = sc.join_runs(sc.load_traces(traj), sc.load_runs(db))
    assert [t.session_key for t in unjoined] == [stray["session_key"]]
    corpus = sc.learn(joined, min_runs=3)
    pair = next((t, r) for t, r in joined if t.session_key == stray["session_key"])
    report = sc.score_run(pair[0], pair[1], corpus)
    assert report is not None
    assert report.run_id == stray["session_key"]
    assert report.run_status is None
    assert {d.step for d in report.deviations} == {"tool:Write", f"write:{Path(REPORT).name}"}


# ---------------------------------------------------------------------------
# step signature: what one run's trace becomes
# ---------------------------------------------------------------------------


def test_a_bash_redirect_counts_as_the_write_it_performs_not_as_a_different_step():
    """Tool substitution is the largest false-alarm source at this scale: the same stage run
    through `Bash > file` must name the same artifact step as the `Write` tool did."""
    via_tool = sc.steps_from_tools([_tool("Write", path=REPORT, seq=1)])
    via_shell = sc.steps_from_tools(
        [_tool("Bash", command=f"date -u +%FT%TZ > {REPORT}", seq=1)]
    )
    assert f"write:{Path(REPORT).name}" in via_tool
    assert f"write:{Path(REPORT).name}" in via_shell


def test_a_date_stamped_nightly_artifact_is_one_step_across_days():
    """`knowledge-handoff-2026-09-20.md` and `-21.md` are the same stage; without eliding the
    stamp a baseline would need one run per day of history to learn anything."""
    assert (
        sc.normalized_basename("/p/knowledge-handoff-2026-09-20.md")
        == "knowledge-handoff-<date>.md"
    )
    assert sc.normalized_basename("/p/knowledge-handoff-2026-09-21.md") == "knowledge-handoff-<date>.md"


def test_an_error_free_trace_keeps_call_order_and_drops_a_generic_target():
    steps = sc.steps_from_tools(
        [
            _tool("Read", path=SOURCE_FILE, seq=2),
            _tool("Bash", command="ls /lloyd", seq=1),
            _tool("Write", path="/tmp/null", seq=3),
        ]
    )
    assert steps[0] == "tool:Bash"  # sequence order, not list order
    assert "read:source-latest.md" in steps
    assert "tool:Write" in steps  # the tool is still a step...
    assert "write:null" not in steps  # ...but a non-name is not an artifact


# ---------------------------------------------------------------------------
# clause 1 — the published set, and a deviation that names the missing step
# ---------------------------------------------------------------------------


def test_three_runs_of_one_task_publish_an_ordered_expected_step_set(tmp_path):
    """Clause 1, first half. Three `autonomy-task:7` traces are enough to publish: the set of
    steps every run took, ordered by where the trace showed them, with the run ids it was
    learned from beside it."""
    corpus, _joined, _unjoined = _corpus(tmp_path, _specs(3), min_runs=3)
    baseline = corpus.baselines[TASK]
    assert baseline.n_runs == 3
    assert baseline.task_id == 7
    assert set(baseline.expected_steps) == {
        "tool:Read",
        f"read:{Path(SOURCE_FILE).name}",
        "tool:Bash",
        "tool:Write",
        f"write:{Path(REPORT).name}",
    }
    # Ordered, not merely a set: the learned median positions are non-decreasing along it.
    positions = [baseline.positions[s] for s in baseline.expected_steps]
    assert positions == sorted(positions)
    published = json.loads(json.dumps(corpus.to_json()))
    task = next(t for t in published["tasks"] if t["session_source"] == TASK)
    assert task["expected_steps"] == baseline.expected_steps
    assert task["learned_from"] == baseline.run_ids
    assert len(task["learned_from"]) == 3


def test_a_run_whose_trace_lacks_a_step_every_other_run_has_names_that_step(tmp_path):
    """Clause 1, second half: one run in four never made the artifact call, so the deviation
    must name the step it is missing — the tool call and the artifact it writes — rather than
    report a score. Every other run of that task did make it."""
    corpus, joined, _unjoined = _corpus(
        tmp_path, _specs(4, broken_at=3), min_runs=3
    )
    baseline = corpus.baselines[TASK]
    # Leave-one-out means the faulty run is not vouched for by itself: the published set still
    # contains the step it skipped, because the other three runs all took it.
    assert f"write:{Path(REPORT).name}" in baseline.expected_steps
    pair = next((t, r) for t, r in joined if r and r.run_id == "run_ok_3")
    report = sc.score_run(pair[0], pair[1], corpus)
    assert report is not None and report.flagged
    assert {d.step for d in report.deviations} == {
        "tool:Write",
        f"write:{Path(REPORT).name}",
    }
    deviation = next(d for d in report.deviations if d.step == f"write:{Path(REPORT).name}")
    assert deviation.kind == "structural"
    assert deviation.state == "deviation"
    assert deviation.learned_from == ["run_ok_0", "run_ok_1", "run_ok_2"]
    assert deviation.learned_from_n == 3
    assert deviation.baseline_runs == 4


def test_a_reordered_but_complete_trace_raises_no_deviation(tmp_path):
    """The set of steps is the contract, not their interleaving: a run that consumes the
    artifact before writing it is a different *ordering*, which is not a skipped step."""
    corpus, joined, _ = _corpus(
        tmp_path,
        _specs(2) + [{"run_id": "run_shuf", "session_key": "sess_shuf", "status": "success", "calls": _shuffled()}] + _specs(1, run_prefix="run_b"),
        min_runs=3,
    )
    pair = next((t, r) for t, r in joined if r and r.run_id == "run_shuf")
    report = sc.score_run(pair[0], pair[1], corpus)
    assert report is not None
    assert report.deviations == []


def test_baselines_are_per_task_never_one_global_shape(tmp_path):
    """His loudest tuning rule: per-client baselines. Task 9 always calls a tool task 7 never
    does, and learning one shape for both would make every task-7 run look broken."""
    specs = _specs(3) + [
        {
            "run_id": f"run_t9_{i}",
            "session_key": f"sess_t9_{i}",
            "status": "success",
            "task_id": 9,
            "source": "autonomy-task:9",
            "calls": _calls() + [_tool("Grep", command=None, path=None, seq=4)],
        }
        for i in range(3)
    ]
    corpus, joined, _ = _corpus(tmp_path, specs, min_runs=3)
    seven = corpus.baselines[TASK]
    nine = corpus.baselines["autonomy-task:9"]
    assert "tool:Grep" in nine.expected_steps
    assert "tool:Grep" not in seven.expected_steps
    # A task-9-shaped run is scored against task 9: the extra tool is not a missing step for 7.
    pair = next((t, r) for t, r in joined if r and r.run_id == "run_t9_0")
    report = sc.score_run(pair[0], pair[1], corpus)
    assert report is not None and report.deviations == []


# ---------------------------------------------------------------------------
# clause 2 — a recorded `success` is scored, and its status rides along
# ---------------------------------------------------------------------------


def test_a_run_recorded_success_is_scored_and_the_deviation_carries_that_status(tmp_path):
    """Clause 2. `success` is precisely the status this detector exists to doubt, so it is
    neither a filter nor a signal: the run is scored, and every deviation repeats the status
    the ledger holds so the disagreement is visible instead of smoothed over."""
    corpus, joined, _ = _corpus(tmp_path, _specs(4, broken_at=2), min_runs=3)
    pair = next((t, r) for t, r in joined if r and r.run_id == "run_ok_2")
    assert pair[1].status == "success"
    report = sc.score_run(pair[0], pair[1], corpus)
    assert report is not None and report.flagged
    assert report.run_status == "success"
    assert all(d.run_status == "success" for d in report.deviations)
    payload = report.to_json()
    assert payload["run_status"] == "success"
    assert payload["deviations"][0]["run_status"] == "success"


def test_status_never_changes_the_verdict(tmp_path):
    """The same trace against the same baseline scores identically under `success` and under
    `failed` — status is reported, never consulted."""
    corpus, joined, _ = _corpus(tmp_path, _specs(4, broken_at=1), min_runs=3)
    pair = next((t, r) for t, r in joined if r and r.run_id == "run_ok_1")
    as_success = sc.score_run(pair[0], pair[1], corpus)
    flipped = sc.RunRecord(
        run_id=pair[1].run_id,
        task_id=pair[1].task_id,
        status="failed",
        started_at=pair[1].started_at,
        completed_at=pair[1].completed_at,
        session_ids=pair[1].session_ids,
    )
    as_failed = sc.score_run(pair[0], flipped, corpus)
    assert [d.step for d in as_success.deviations] == [d.step for d in as_failed.deviations]
    assert as_failed.run_status == "failed"
    assert as_success.run_status == "success"


# ---------------------------------------------------------------------------
# clause 3 — late versus missing, the grace window
# ---------------------------------------------------------------------------


def test_a_missing_step_is_pending_inside_the_grace_window_and_deviation_after_it(tmp_path):
    """Clause 3, first half — his late-vs-missing problem: a step that has not been emitted
    yet is indistinguishable from one that never will be, so the first pass holds the finding
    as `pending` and only a re-check after the window lapses promotes it. `pending` must never
    be counted as a flagged run, or the alarm fires on every in-flight job."""
    specs = _specs(4, broken_at=0)
    for spec in specs:
        spec["started"] = "2026-09-10T05:00:00+00:00"
        spec["completed"] = "2026-09-10T05:12:00+00:00"
    corpus, joined, _ = _corpus(tmp_path, specs, min_runs=3)
    pair = next((t, r) for t, r in joined if r and r.run_id == "run_ok_0")
    grace = 6 * 3600
    inside = datetime(2026, 9, 10, 5, 40, tzinfo=timezone.utc)  # 28 min after it ended
    outside = datetime(2026, 9, 10, 11, 30, tzinfo=timezone.utc)  # 6 h 18 min after

    first = sc.score_run(pair[0], pair[1], corpus, now=inside, grace_seconds=grace)
    assert first is not None
    assert first.deviations == []
    assert first.flagged is False
    assert {d.step for d in first.pending} == {"tool:Write", f"write:{Path(REPORT).name}"}
    held = next(d for d in first.pending if d.step == f"write:{Path(REPORT).name}")
    assert held.state == "pending"
    assert held.grace_seconds == grace
    assert held.run_completed_at == "2026-09-10T05:12:00+00:00"
    assert json.loads(json.dumps(first.to_json()))["pending"][0]["state"] == "pending"

    again = sc.score_run(pair[0], pair[1], corpus, now=outside, grace_seconds=grace)
    assert again is not None and again.flagged
    promoted = next(d for d in again.deviations if d.step == f"write:{Path(REPORT).name}")
    assert promoted.state == "deviation"
    assert again.pending == []


def test_a_step_that_appears_later_than_its_learned_position_raises_nothing(tmp_path):
    """Clause 3, second half: a step taken out of its learned place is *late*, which is an
    observation the report records but never a deviation — only absence is structural here."""
    late_calls = [
        _tool("Read", path=SOURCE_FILE, seq=1),
        _tool("Bash", command=f"wc -c {REPORT}", seq=2),
        _tool("Read", path="/lloyd/CLAUDE.md", seq=3),
        _tool("Read", path="/lloyd/config.yaml", seq=4),
        _tool("Write", path=REPORT, seq=5),
    ]
    corpus, joined, _ = _corpus(
        tmp_path,
        _specs(3) + [{"run_id": "run_late", "session_key": "sess_late", "status": "success", "calls": late_calls}],
        min_runs=3,
    )
    pair = next((t, r) for t, r in joined if r and r.run_id == "run_late")
    report = sc.score_run(pair[0], pair[1], corpus)
    assert report is not None
    assert report.deviations == []
    assert report.pending == []
    assert report.flagged is False


# ---------------------------------------------------------------------------
# clause 5 (first half) — evidence beside every deviation; no baseline is not a clean run
# ---------------------------------------------------------------------------


def test_every_deviation_carries_the_expectation_it_was_judged_against(tmp_path):
    """Clause 5, first half — the per-run half of it, pinned end-to-end over the fixture
    corpus in `tests/test_step_conformance_replay.py`. An unexplained score is his "doctor
    says your health score is 22" failure: each deviation must carry the task id, the run and
    session ids, the offending step, the whole learned expected set, and the run ids that
    produced it."""
    corpus, joined, _ = _corpus(tmp_path, _specs(4, broken_at=1), min_runs=3)
    pair = next((t, r) for t, r in joined if r and r.run_id == "run_ok_1")
    report = sc.score_run(pair[0], pair[1], corpus)
    baseline = corpus.baselines[TASK]
    payload = json.loads(json.dumps(report.to_json()))
    assert payload["task_id"] == 7
    assert payload["run_id"] == "run_ok_1"
    assert payload["session_key"] == pair[0].session_key
    for deviation in payload["deviations"]:
        assert deviation["step"]
        assert deviation["session_source"] == TASK
        assert deviation["expected_steps"] == baseline.expected_steps
        assert deviation["learned_from"] == ["run_ok_0", "run_ok_2", "run_ok_3"]
        assert deviation["learned_from_n"] == len(deviation["learned_from"])
        assert deviation["baseline_runs"] == 4
        assert deviation["checked_at"]


def test_a_task_with_thin_history_reports_no_baseline_rather_than_a_clean_run(tmp_path):
    """Not one of #673's numbered clauses — the precondition for clause 1. Two runs are not a
    baseline: `score_run` returns None and the
    caller reports *no baseline*. Reporting it as clean is how a detector earns being ignored."""
    corpus, joined, _ = _corpus(tmp_path, _specs(2, broken_at=0), min_runs=3)
    assert TASK not in corpus.baselines
    # What is reported as unscored is the runs the corpus could not learn from, by id.
    assert sorted(corpus.unscored) == ["run_ok_0", "run_ok_1"]
    for trace, run in joined:
        assert sc.score_run(trace, run, corpus) is None
    published = json.loads(json.dumps(corpus.to_json()))
    assert published["tasks"] == []
    assert published["unscored_sessions"] == sorted(corpus.unscored)


def test_replay_counts_a_run_with_no_baseline_as_unscored_not_as_quiet(tmp_path):
    """The denominator a reader needs: the replay separates scored, flagged, pending and
    no-baseline so a silent corpus cannot be read as a healthy one."""
    _traj, db = _write_corpus(tmp_path, _specs(2, broken_at=0))
    result = sc.replay(
        trajectory_dir=_traj,
        db_path=db,
        days=None,
        now=datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc),
        min_runs=3,
    )
    assert result["runs_scored"] == 0
    assert result["tasks_scored"] == 0
    assert result["runs_flagged"] == 0
    assert result["tasks_without_baseline"] == 2


# ---------------------------------------------------------------------------
# the nightly surface: a read-only replay, run as a process
# ---------------------------------------------------------------------------


def test_replay_flags_only_the_broken_run_across_a_fourteen_run_corpus(tmp_path):
    """The nightly path end to end, on data shaped like the real store: 14 runs of one task,
    one of which skipped its artifact call, all recorded `success`."""
    corpus_specs = _specs(14, broken_at=7)
    traj, db = _write_corpus(tmp_path, corpus_specs)
    result = sc.replay(
        trajectory_dir=traj,
        db_path=db,
        days=None,
        now=datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc),
        grace_seconds=6 * 3600,
        min_runs=3,
    )
    assert result["runs_scored"] == 14
    assert result["runs_flagged"] == 1
    assert result["deviations"] == 2
    assert result["runs_pending"] == 0
    assert result["flagged_rate"] == pytest.approx(1 / 14)
    assert result["success_recorded_quiet"] == 13
    assert result["success_recorded_flagged"] == 1
    assert result["baseline_digest"]
    assert result["tasks"] == [TASK]
    published = json.loads(json.dumps(result["published"]))
    task = next(t for t in published["tasks"] if t["session_source"] == TASK)
    assert f"write:{Path(REPORT).name}" in task["expected_steps"]


def test_replay_leaves_both_stores_byte_identical_and_exits_zero(tmp_path):
    """Process boundary: the nightly job is a *read-only* replay over the live trajectory
    store and the live ledger. Run it as a subprocess and prove neither store moved."""
    traj, db = _write_corpus(tmp_path, _specs(6, broken_at=5))
    before = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [db, traj / "2026-09-10.jsonl"]
    }
    out = tmp_path / "report.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "replay",
            "--trajectories",
            str(traj),
            "--db",
            str(db),
            "--min-runs",
            "3",
            "--json",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "runs_flagged=1/6" in proc.stdout
    assert "false-alarm" in proc.stdout  # the rate is printed beside the count, always
    assert json.loads(out.read_text())["runs_flagged"] == 1
    after = {
        p: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [db, traj / "2026-09-10.jsonl"]
    }
    assert after == before


def test_the_learn_command_publishes_a_baseline_file_that_loads(tmp_path):
    """The published artifact is the interface: `learn` writes the ordered per-task step set,
    and the file it writes is readable by a later process without this module's memory."""
    traj, db = _write_corpus(tmp_path, _specs(5))
    out = tmp_path / "step-baseline.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "learn",
            "--trajectories",
            str(traj),
            "--db",
            str(db),
            "--min-runs",
            "3",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    published = json.loads(out.read_text())
    assert published["schema"] == 1
    task = next(t for t in published["tasks"] if t["session_source"] == TASK)
    assert task["task_id"] == 7
    assert f"write:{Path(REPORT).name}" in task["expected_steps"]
    assert task["runs_with_step"][f"write:{Path(REPORT).name}"] == 5
    assert len(task["learned_from"]) == 5


# ---------------------------------------------------------------------------
# the two things this must never become
# ---------------------------------------------------------------------------


def test_the_module_cannot_alert_from_inside_itself():
    """"Do not alert yet" is an acceptance clause, and the cheapest way to break it is an
    import a later edit leans on. The detector's only outputs are stdout and the `--json` path
    an operator names, so the module must import no transport and spawn nothing: read the
    imports off the AST, where a comment cannot hide one."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = {
        "smtplib", "email", "requests", "httpx", "urllib", "socket", "ssl", "asyncio",
        "subprocess", " multiprocessing", "sh", "slack", "pymongo",
    }
    assert imported & forbidden == set(), f"grew a channel: {sorted(imported & forbidden)}"
    assert "sqlite3" in imported, "the ledger reader is expected to be here"


def test_every_ledger_connection_is_forced_read_only():
    """Property, not a name: any `sqlite3.connect` in this file must either be opened through a
    `?mode=ro` URI or be the fixture builder creating a database in a temp directory it owns.
    A third shape — a read-write handle on the live ledger — is what this pins out."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent

    def enclosing(node):
        current = node
        while current is not None:
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return current
            current = parents.get(id(current))
        return None

    sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "connect"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "sqlite3"
    ]
    assert sites, "the ledger reader must exist"
    for site in sites:
        owner = enclosing(site)
        assert owner is not None
        body = ast.get_source_segment(source, owner) or ""
        kwargs = {kw.arg for kw in site.keywords}
        if "uri" in kwargs:
            assert "?mode=ro" in body, f"{owner.name}: a URI handle without the read-only flag"
        else:
            # The fixture builder: its database path is derived from the directory the caller
            # hands it, and it never names the live default. Where that directory comes from is
            # the caller's business; what must not happen is this file opening `~/lloyd/workers.db`.
            assert "DEFAULT_DB" not in body, f"{owner.name} could open the live ledger"
            assert "dest" in body and "runs.db" in body, (
                f"{owner.name}: a writable handle must be scoped to the directory it was given"
            )
    assert len(sites) <= 2, f"a third database handle appeared: {len(sites)}"


def test_the_module_names_the_dependency_gate_as_another_owners_decision():
    """#558 owns whether a task was allowed to run; this file must say so and hold no gate of
    its own — a second implementation of a dependency check is how two answers diverge."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "_is_dependency_met" in source, "the boundary is named, not implied"
    assert "depends_on" not in source, "no dependency logic may live in a trace reader"
