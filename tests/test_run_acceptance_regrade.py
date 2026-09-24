"""Reading an acceptance block back over history, and linting it first (#623).

`app/run_acceptance.py` grades a run live. These pin the two things the pilot
needed around it: `trace_from_transcript` rebuilds from a session transcript
the same trace `DispatchTrace` collects off the live event stream (so a past
run is graded exactly as a live one would be), and `acceptance_problems` — read
by `scripts/autonomy/validate_tasks.py` — names a block that would grade
something other than what it says before the task ever runs.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

import autonomy
from app.run_acceptance import (DispatchTrace, acceptance_problems, grade_run,
                                trace_from_transcript)
from scripts.autonomy import regrade_runs

ROOT = Path(__file__).resolve().parent.parent
LINTER = ROOT / "scripts" / "autonomy" / "validate_tasks.py"
BASH_CHECK = {"objective_checks": [{"type": "tool_called", "value": "Bash"},
                                   {"type": "regex", "value": r"wrote \d+ rows"}]}


def _transcript(*, bash_result: str | None, final: str) -> list[dict]:
    """A session transcript in the shape `app/transcript_entries.py` writes."""
    msgs: list[dict] = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
    if bash_result is not None:
        msgs += [
            {"role": "assistant", "content": [{"type": "text", "text": ""}],
             "tool_calls": [{"id": "c1", "call_id": "c1", "type": "function",
                             "function": {"name": "Bash", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1",
             "content": [{"type": "text", "text": bash_result}]},
        ]
    msgs += [{"role": "thinking", "content": [], "reasoning": "done"},
              {"role": "assistant", "content": [{"type": "text", "text": final}]}]
    return msgs


def _live(*, bash_result: str | None, final: str) -> dict:
    d = DispatchTrace()
    if bash_result is not None:
        d.observe({"type": "tool_call", "call_id": "c1", "name": "Bash"})
        d.observe({"type": "tool_result", "call_id": "c1", "name": "Bash",
                   "content": bash_result})
    return d.as_trace(final)


@pytest.mark.parametrize("bash_result", [None, "wrote 12 rows",
                                         "Tool call denied: destructive command"])
def test_a_transcript_rebuilds_the_trace_the_live_run_collected(bash_result):
    final = "Done: wrote 12 rows to the table."
    assert (trace_from_transcript(_transcript(bash_result=bash_result, final=final))
            == _live(bash_result=bash_result, final=final))


def test_a_regraded_claim_without_a_dispatch_still_fails():
    """The #623 property survives the replay: the text claims Bash, nothing ran."""
    trace = trace_from_transcript(_transcript(bash_result=None,
                                              final="I ran Bash and wrote 12 rows."))
    grade = grade_run({"acceptance": BASH_CHECK}, trace)
    assert grade["grade"] == "graded_fail"
    assert grade["checks"][0] == {"type": "tool_called", "value": "Bash", "passed": False}


def test_the_terminal_text_is_the_last_non_empty_assistant_block():
    msgs = _transcript(bash_result="ok", final="final report")
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": "  "}]})
    assert trace_from_transcript(msgs)["final_text"] == "final report"


@pytest.mark.parametrize("block, needle", [
    ({"objective_checks": [{"type": "regex", "value": "(unclosed"}]}, "does not compile"),
    ({"objective_checks": [{"type": "find_all", "value": "gold_member",
                            "gold_items": ["a"]}]}, "find_all is not supported"),
    ({"objective_checks": [{"type": "file_exists", "value": "x.md"}]}, "not a judge check"),
    ({"objective_checks": [], "rubric": ["accuracy"]}, "no objective_checks"),
])
def test_a_block_that_cannot_grade_what_it_says_is_named(block, needle):
    problems = acceptance_problems({"acceptance": block})
    assert any(needle in p for p in problems), problems


def test_a_sound_block_and_an_absent_one_are_clean():
    assert acceptance_problems({"acceptance": BASH_CHECK}) == []
    assert acceptance_problems({}) == []


def test_an_uncompilable_regex_is_what_the_judge_would_fail_forever():
    """Why the linter names it: the judge swallows `re.error` as a fail."""
    grade = grade_run({"acceptance": {"objective_checks": [
        {"type": "regex", "value": "(unclosed"}]}}, _live(bash_result=None, final="(unclosed"))
    assert grade["grade"] == "graded_fail"


def _task_file(d: Path, name: str, acceptance: str) -> None:
    (d / name).write_text("---\nid: 1\nname: x\nstatus: paused\nfrequency: daily\n"
                          f"{acceptance}---\n\nbody\n")


def test_the_linter_warns_on_a_bad_block_and_strict_fails_it(tmp_path):
    aut = tmp_path / "autonomy"
    aut.mkdir()
    (tmp_path / "config.yaml").write_text("models: {}\n")
    _task_file(aut, "1-x.md", "acceptance:\n  objective_checks:\n"
                              "  - {type: regex, value: '(unclosed'}\n")
    args = [sys.executable, str(LINTER), "--autonomy-dir", str(aut),
            "--skills-dir", str(tmp_path), "--config", str(tmp_path / "config.yaml")]
    out = subprocess.run(args, capture_output=True, text=True, timeout=60)
    assert "1-x.md: acceptance regex '(unclosed' does not compile" in out.stdout
    strict = subprocess.run(args + ["--strict"], capture_output=True, text=True, timeout=60)
    assert strict.returncode == 2

    _task_file(aut, "1-x.md", "acceptance:\n  objective_checks:\n"
                              "  - {type: tool_called, value: Bash}\n")
    clean = subprocess.run(args + ["--strict"], capture_output=True, text=True, timeout=60)
    assert "acceptance" not in clean.stdout
    assert clean.returncode == 0, clean.stdout


def test_regrade_grades_history_from_workers_db_and_sessions(tmp_path, monkeypatch):
    aut = tmp_path / "autonomy"
    aut.mkdir()
    (aut / "7-pilot.md").write_text(
        "---\nid: 7\nname: pilot\nstatus: up_next\nacceptance:\n  objective_checks:\n"
        "  - {type: tool_called, value: Bash}\n  - {type: regex, value: 'wrote \\d+ rows'}\n"
        "---\n\nbody\n")
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", aut)
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    for sid, bash in (("s_ok", "wrote 3 rows"), ("s_claim", None)):
        (sessions / f"{sid}.json").write_text(json.dumps(
            {"messages": _transcript(bash_result=bash, final="Done, wrote 3 rows.")}))
    db = tmp_path / "w.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE runs (run_id TEXT, source TEXT, task_id TEXT, status TEXT,"
                     " started_at TEXT, meta_json TEXT)")
        rows = [("r1", "7", "success", "s_ok"), ("r2", "7", "success", "s_claim"),
                ("r3", "7", "failed", "s_ok"), ("r4", "8", "success", "s_ok")]
        for run_id, tid, status, sid in rows:
            conn.execute("INSERT INTO runs VALUES (?, 'scheduled-task', ?, ?, ?, ?)",
                         (run_id, tid, status, "2026-09-24T00:00:00+00:00",
                          json.dumps({"session_id": sid})))

    graded = regrade_runs.regrade(regrade_runs.success_runs(db, "2026-09-01", {"7"}), sessions)
    assert {r["run_id"]: r["grade"] for r in graded} == {"r1": "graded_pass",
                                                        "r2": "graded_fail"}
    summary = regrade_runs.summarise(graded)
    assert summary["tasks"]["7"]["false_completion_rate"] == 0.5
    assert summary["pooled"]["graded"] == 2


def test_wilson_is_none_over_nothing_and_bounded_otherwise():
    assert regrade_runs.wilson(0, 0) is None
    lo, hi = regrade_runs.wilson(0, 20)
    assert lo == 0.0 and 0.15 < hi < 0.17
