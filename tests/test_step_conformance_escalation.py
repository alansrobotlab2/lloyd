"""#2104 clauses 2, 3, 4 and 5 — the one non-alerting escalation the replay may take.

The detector shipped as a report (`replay` prints and exits 0, and its own NOTE line says
`Nothing here alerts`) while the thing it detects — a run that recorded `success` after
skipping the stage that writes an artifact — is invisible unless somebody reads that print
every morning. The guardian's record says every rollback so far was a false positive, so the
answer is not an alert channel: a deviation naming a missing *written* artifact files a backlog
**draft** for a human to adjudicate, and nothing pages anybody.

The clauses, each pinned by the test that names it:

2. a flagged deviation whose missing step begins `write:` files exactly one draft; a flagged
   run whose missing step begins `tool:` files zero, and one missing a `read:` step files none;
3. the filed draft names the task id, the run id, the offending step and that deviation's
   `learned_from` run ids, and one escalating run never produces a second draft;
4. the escalation is confined — its only write is the draft, it makes no alert call, and the
   replay around it still mutates neither the ledger nor the trajectory store;
5. the autonomy task that runs it nightly names the read-only replay at strict support and no
   alert surface (marked `live_vault`: that clause is about a file in the live vault, which the
   gate's `-m "not live_vault"` run therefore does not judge).

#2477 adds a second gate in front of every clause above, on which *run* qualifies: a draft is
filed only for a run whose recorded terminal status is not a ledger failure. `failed` and
`interrupted` file nothing while staying scored, reported and counted in `runs_flagged`;
`success` files exactly as it always did, and so does a trace with no ledger row, whose
recorded status is `None` — an unknown status is not a recorded failure. Those four edges are
pinned by `test_an_aborted_run_missing_its_artifact_files_no_draft_but_still_reports`,
`test_a_green_run_missing_its_artifact_still_files_under_the_same_name`,
`test_a_flagged_run_with_no_ledger_row_still_files` and
`test_the_aborted_run_keeps_its_report_line_and_its_header_counters`.

The corpora are built with the helpers from `tests/test_step_conformance.py`, so every store
here is a real trajectory JSONL plus a real sqlite `runs` table read through the same read-only
loaders the nightly replay uses, and the filing is exercised both ways: through
`file_escalations` directly, and through the shipped CLI as a subprocess.

Why every corpus here has exactly one faulty run: at strict `support` an expectation needs
`n_runs - 1` of `n_runs`, so the moment two runs of one task miss the same step the step stops
being expected of either of them and nothing flags. Two escalating runs therefore come from two
tasks, which is also how the real board looks.

What the clause-5 fixture comparison compares, and what it cannot (#2118): the committed copy of
the nightly task is witness to the instruction the scheduler dispatches, and the live vault file
it is checked against is one the scheduler itself rewrites on every run — `_update_task_field`
restamps its clock keys and `_append_activity_log` appends a bullet, and the whole front-matter
block comes back through `yaml.dump` re-quoted and re-wrapped. Byte equality therefore went red
the morning after task 93's first daily run, with no field of the job changed. `_witnessed_surface`
is what the two files are compared on instead, `SCHEDULER_OWNED_KEYS` excluded, and the three
nodes after it pin both halves of that: a restamp through the shipped writer is not drift, an edit
to a kept key is, and the gate's own selection still collects the node and runs it to a verdict.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import test_step_conformance as base  # tests/ is on sys.path, same as board_presence


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "step_conformance.py"
AUTONOMY_DIR = Path("~/obsidian/autonomy").expanduser()

# The steps the escalation corpora learn, one of each kind, plus a second call of every kind
# that is never dropped: without it, removing `Write` would also remove `tool:Write`, and a
# test that flags two steps could not say which one the filer acted on.
PLAN = "/lloyd/_pipeline/reflection/plan-latest.md"
NOTES = "/lloyd/_pipeline/reflection/notes-latest.md"
SCRATCH = "/lloyd/_pipeline/reflection/scratch-latest.md"

DROPPABLE = {
    "write": ("Write", "/lloyd/_pipeline/reflection/report-latest.md"),
    "read": ("Read", PLAN),
    "tool": ("TodoWrite", None),
}
MISSING_STEP = {"write": "write:report-latest.md", "read": "read:plan-latest.md",
                "tool": "tool:TodoWrite"}


def _load_module():
    spec = importlib.util.spec_from_file_location("step_conformance", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sc = _load_module()


def _clean_calls():
    """Every clean run of a fixture task: read two files, log the plan, count the report,
    write the report, write a scratch note."""
    return [
        base._tool("Read", path=PLAN, seq=1),
        base._tool("Read", path=NOTES, seq=2),
        base._tool("TodoWrite", seq=3),
        base._tool("Bash", command=f"wc -c {base.REPORT}", seq=4),
        base._tool("Write", path=base.REPORT, seq=5),
        base._tool("Write", path=SCRATCH, seq=6),
    ]


def _drop(missing: str):
    """The same run with exactly one call never made."""
    name, path = DROPPABLE[missing]
    return [
        call for call in _clean_calls()
        if not (call["name"] == name
                and (path is None
                     or call["params_summary"].get("file_path") == path))
    ]


def _specs(count: int, *, faulty_at: int, missing: str | None,
           source: str = base.TASK, prefix: str = "run_ok",
           status: str = "success"):
    """`count` runs of one fixture task, one of which never made one of its calls.

    `status` is the terminal status the *faulty* run is recorded with in the ledger; the other
    runs of its task keep `success`, which is the shape the real board has — a task whose
    baseline ran green and whose one bad run died. It defaults to `success`, the status every
    corpus in this file has always carried, so the nodes above this kwarg were added read
    exactly the rows they read before it. #2477's aborted-run nodes pass `failed` or
    `interrupted`, and `base._write_corpus` inserts whatever the spec says into the `runs.status`
    column verbatim, which is what makes the ledger half of the join real rather than a stubbed
    field on the report."""
    specs = []
    for i in range(count):
        faulty = bool(missing) and i == faulty_at
        calls = _drop(missing) if faulty else _clean_calls()
        specs.append({
            "run_id": f"{prefix}_{i}",
            "session_key": f"2026091{i}_0500{i}0_autonomy_{prefix}{i}",
            "status": status if faulty else "success",
            "source": source,
            "task_id": int(source.rsplit(":", 1)[1]),
            "calls": calls,
        })
    return specs


def _replay(root: Path, specs, **kwargs):
    """Score a corpus the way the nightly job does: the real loaders, the real `replay`."""
    traj, db = base._write_corpus(root, specs)
    return sc.replay(trajectory_dir=traj, db_path=db, **kwargs)


def _flagged(result) -> list[dict]:
    return [r for r in result["reports"] if r["deviations"]]


def _steps(report: dict) -> list[str]:
    return [d["step"] for d in report["deviations"]]


def _drafts(backlog: Path) -> list[Path]:
    return sorted(backlog.glob("*.md"))


# ---------------------------------------------------------------------------
# the input the escalation path is handed
# ---------------------------------------------------------------------------


def test_a_run_that_never_wrote_its_artifact_is_flagged_on_one_missing_write_step(tmp_path):
    """Precondition for clauses 2 and 3, and the control on the fixture: the flagged run names
    `write:<artifact>` and nothing else, carries the run id it was judged on, and carries the
    run ids that expectation was learned from. If this corpus flagged two steps the filing tests
    below could not say which one the filer acted on; if it flagged none they would be grading
    an empty input."""
    result = _replay(tmp_path, _specs(6, faulty_at=5, missing="write"))
    flagged = _flagged(result)
    assert len(flagged) == 1, [r["run_id"] for r in flagged]
    assert _steps(flagged[0]) == ["write:report-latest.md"]
    assert flagged[0]["run_id"] == "run_ok_5"
    assert flagged[0]["deviations"][0]["learned_from"] == [f"run_ok_{i}" for i in range(5)]


# ---------------------------------------------------------------------------
# clause 2 — write: files one draft; tool: and read: file nothing
# ---------------------------------------------------------------------------


def test_a_missing_written_artifact_files_exactly_one_backlog_draft(tmp_path):
    """Clause 2, first half. One flagged run whose missing step is `write:` produces one
    `status: draft` item on disk — not `up_next`, because a deviation on unlabelled traffic is
    a candidate, and promoting it would hand the autocode loop work nobody adjudicated."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    result = _replay(corpus, _specs(6, faulty_at=5, missing="write"))
    filed = sc.file_escalations(result, backlog_dir=backlog)
    assert len(filed) == 1
    drafts = _drafts(backlog)
    assert len(drafts) == 1
    assert str(drafts[0]) == filed[0]["path"]
    front_matter = drafts[0].read_text(encoding="utf-8").split("---")[1]
    assert "status: draft" in front_matter, front_matter
    assert sc.ESCALATION_TAG in front_matter, front_matter


@pytest.mark.parametrize("missing", ["tool", "read"])
def test_a_missing_tool_call_or_read_files_no_draft_at_all(tmp_path, missing):
    """Clause 2, second half — the half that keeps the filing rate readable. A run that never
    called `TodoWrite`, or never read a file every other run read, may have taken an equivalent
    path: those steps name a *tool*, not an artifact. Only a missing written artifact is evidence
    that a stage did not happen, so both of these flag and neither files."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    result = _replay(corpus, _specs(6, faulty_at=5, missing=missing))
    flagged = _flagged(result)
    assert _steps(flagged[0]) == [MISSING_STEP[missing]]
    assert sc.escalation_targets(result) == []
    assert sc.file_escalations(result, backlog_dir=backlog) == []
    assert _drafts(backlog) == []


def test_a_clean_corpus_files_no_draft(tmp_path):
    """The control over the filer itself: six runs that all took the same shape flag nothing, so
    the one draft above cannot be credited to a filer that files whenever it is called."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    result = _replay(corpus, _specs(6, faulty_at=None, missing=None))
    assert _flagged(result) == []
    assert sc.file_escalations(result, backlog_dir=backlog) == []
    assert _drafts(backlog) == []


# ---------------------------------------------------------------------------
# #2477 — the escalation gate reads the run's recorded terminal status
# ---------------------------------------------------------------------------


def _forget_ledger_row(db: Path, run_id: str) -> None:
    """Delete one run's ledger row from a fixture corpus, leaving its trajectory in place.

    That is how a real trace reaches the replay with no ledger row: `--source` filters the
    ledger side of the join only, so `join_runs` hands the scorer `(trace, None)` and the
    report's `run_status` is `None` (`scripts/step_conformance.py`, `join_runs`). Deleting the
    row is what makes clause 3's `None` come through the shipped loaders instead of being
    written by hand into a report dict."""
    conn = sqlite3.connect(db)
    try:
        conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
        conn.commit()
    finally:
        conn.close()


@pytest.mark.parametrize("aborted", ["failed", "interrupted"])
def test_an_aborted_run_missing_its_artifact_files_no_draft_but_still_reports(tmp_path, aborted):
    """#2477 clause 1, both edges of the suppressed set. A run the fleet recorded as `failed` or
    `interrupted` never had the chance to write its artifact, so the absence is a consequence of
    the abort and the ledger row already carries the cause — which is exactly what #2238 (task
    24, 14.85 s, `empty response after 14s`) and #2473 (task 38, 235.93 s, `StreamStalledError:
    stream produced no data for 60s after 564 line(s)`) were adjudicated by hand. The run is
    still scored and still flagged: the suppression sits at the escalation gate, so
    `runs_flagged` and the printed deviation still name it while `escalation_targets` returns
    nothing and the board gets no draft."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    result = _replay(corpus, _specs(6, faulty_at=5, missing="write", status=aborted))
    flagged = _flagged(result)
    assert len(flagged) == 1, [r["run_id"] for r in flagged]
    assert flagged[0]["run_status"] == aborted
    assert _steps(flagged[0]) == ["write:report-latest.md"]
    # Scored, not skipped: the header counters are computed before any escalation exists.
    assert result["runs_flagged"] == 1
    assert result["runs_scored"] == 6
    assert result["success_recorded_runs"] == 5
    assert result["success_recorded_flagged"] == 0
    assert sc.escalation_targets(result) == []
    assert sc.file_escalations(result, backlog_dir=backlog) == []
    assert _drafts(backlog) == []


def test_a_green_run_missing_its_artifact_still_files_under_the_same_name(tmp_path):
    """#2477 clause 2. The class this detector exists to doubt is the run that recorded green
    while its artifact went unwritten, and the suppression must not touch it: one target, one
    draft, `run_status: success` in the body. The exact name is asserted because the name IS the
    dedupe key — `file_escalations` compares it against every item already on the board — so a
    green run's title has to keep the shape #2238 and #2473 were filed with or the next replay
    re-files two drafts beside them."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    result = _replay(corpus, _specs(6, faulty_at=5, missing="write"))
    targets = sc.escalation_targets(result)
    assert len(targets) == 1
    assert targets[0]["run_status"] == "success"
    assert sc.escalation_name(targets[0]) == (
        "step-conformance: task 7 run run_ok_5 never wrote report-latest.md")
    filed = sc.file_escalations(result, backlog_dir=backlog)
    assert len(filed) == 1
    assert filed[0]["name"] == sc.escalation_name(targets[0])
    drafts = _drafts(backlog)
    assert len(drafts) == 1
    text = drafts[0].read_text(encoding="utf-8")
    assert "run_status: success" in text
    assert "missing_step: write:report-latest.md" in text
    assert sc.file_escalations(result, backlog_dir=backlog) == [], "re-filed the same run"


def test_a_flagged_run_with_no_ledger_row_still_files(tmp_path):
    """#2477 clause 3. An unknown status is not a recorded failure. The ledger row is deleted
    from the corpus so the trace reaches the scorer unjoined the way a `--source`-filtered
    replay leaves 1,660 of 2,096 traces, and the report's `run_status` is `None` — suppressed on
    that edge would silently drop every deviation outside the ledger's filtered coverage."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    traj, db = base._write_corpus(corpus, _specs(6, faulty_at=5, missing="write"))
    _forget_ledger_row(db, "run_ok_5")
    result = sc.replay(trajectory_dir=traj, db_path=db)
    assert result["traces_without_ledger_row"] == 1, result["traces_loaded"]
    flagged = _flagged(result)
    assert len(flagged) == 1, [r["run_id"] for r in flagged]
    assert flagged[0]["run_status"] is None
    assert _steps(flagged[0]) == ["write:report-latest.md"]
    assert len(sc.escalation_targets(result)) == 1
    filed = sc.file_escalations(result, backlog_dir=backlog)
    assert len(filed) == 1, filed
    assert len(_drafts(backlog)) == 1
    assert "run_status: None" in _drafts(backlog)[0].read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# clause 3 — the draft carries the evidence, and one run never files twice
# ---------------------------------------------------------------------------


def test_the_filed_draft_names_the_task_run_offending_step_and_learned_from_runs(tmp_path):
    """Clause 3, first half. Every identifier a reader needs to reproduce the judgement is in the
    item body: task id, session source, run id, session key, the offending step id, and the run
    ids the expectation was learned from — the same evidence `--json` prints, in prose."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    result = _replay(corpus, _specs(6, faulty_at=5, missing="write"))
    sc.file_escalations(result, backlog_dir=backlog)
    text = _drafts(backlog)[0].read_text(encoding="utf-8")
    assert "task_id: 7" in text
    assert "session_source: autonomy-task:7" in text
    assert "run_id: run_ok_5" in text
    assert "missing_step: write:report-latest.md" in text
    assert "session_key: 20260915_050050_autonomy_run_ok5" in text
    for run_id in [f"run_ok_{i}" for i in range(5)]:
        assert run_id in text, f"learned_from run {run_id} missing from the draft"


def test_one_escalating_run_never_produces_a_second_draft(tmp_path):
    """Clause 3, second half. The nightly replay re-scores the same history for days: Monday's
    deviation is still Tuesday's. Dedupe is against every name already on the board, so the
    second pass files nothing and allocates no id."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    result = _replay(corpus, _specs(6, faulty_at=5, missing="write"))
    first = sc.file_escalations(result, backlog_dir=backlog)
    second = sc.file_escalations(result, backlog_dir=backlog)
    assert len(first) == 1
    assert second == [], second
    assert len(_drafts(backlog)) == 1, [p.name for p in _drafts(backlog)]


def test_two_escalating_runs_file_two_drafts(tmp_path):
    """Dedupe is per run, not per task or per step: the incident #673 was filed over was a task
    failing its stage twice. At strict support two runs of *one* task missing the same step stop
    making it expected at all, so the two runs come from two tasks — which is also the case where
    collapsing them would hide half the evidence."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    specs = _specs(6, faulty_at=5, missing="write")
    specs += _specs(6, faulty_at=2, missing="write",
                    source="autonomy-task:8", prefix="run_other")
    result = _replay(corpus, specs)
    assert len(_flagged(result)) == 2, [(r["task_id"], _steps(r)) for r in _flagged(result)]
    assert len(sc.file_escalations(result, backlog_dir=backlog)) == 2
    assert len(_drafts(backlog)) == 2


def test_a_dry_run_prints_the_answer_and_writes_nothing(tmp_path):
    """The operator's way to see the filing rate before accepting it — and the control that the
    one draft above came from `new_item`, not from a filer that always writes."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    result = _replay(corpus, _specs(6, faulty_at=5, missing="write"))
    would = sc.file_escalations(result, backlog_dir=backlog, dry_run=True)
    assert len(would) == 1
    assert would[0]["path"] == ""
    assert _drafts(backlog) == []


# ---------------------------------------------------------------------------
# #2944 — the reproducer a draft prints is the command that flagged the run
# ---------------------------------------------------------------------------


def _recent(specs, *, days_ago: int = 2):
    """Stamp a fixture corpus `days_ago` days back from now, so a windowed replay can see it.

    Every other corpus in this file is stamped 2026-09-10, which falls outside a `--days 7`
    window as soon as the calendar passes 2026-09-17 — and a `--days` token that filters every
    run out of the replay flags nothing, so a test of a windowed reproducer has to move its own
    history inside the window it is testing. Two days back is inside `--days 7` and still older
    than `DEFAULT_GRACE_SECONDS` (6 h), so the deviation flags rather than sitting pending. The
    trajectory stamp is machine-local wall clock on purpose: `load_traces` parses a naive
    trajectory timestamp as local (`_parse_ts(..., assume_local=True)`), which is what a real
    session JSONL row is."""
    then = datetime.now(timezone.utc) - timedelta(days=days_ago)
    for spec in specs:
        spec["started"] = then.isoformat(timespec="seconds")
        spec["completed"] = (then + timedelta(minutes=12)).isoformat(timespec="seconds")
        spec["timestamp"] = then.astimezone().strftime("%Y-%m-%dT%H:%M:%S")
    return specs


def _reproducer(text: str) -> str:
    """The one line of a draft a window is allowed to change: the command in the fenced block.

    Matched on the `python3` prefix, not on the module path: the body's opening paragraph names
    `scripts/step_conformance.py replay` in prose too, and a matcher that collected that line as
    well would make "exactly one reproducer" unassertable."""
    lines = [ln for ln in text.splitlines() if "python3 scripts/step_conformance.py replay" in ln]
    assert len(lines) == 1, f"expected exactly one reproducer line, got {lines}"
    return lines[0]


def test_a_draft_filed_by_a_seven_day_run_carries_that_window_as_its_reproducer(tmp_path):
    """#2944 clause 1, across the process boundary the nightly filer actually crosses. Job 93
    runs `replay --days 7 --support 1.0 --escalate-writes` — the command the clause-5
    drift-witness nodes pin — so the draft it files has to send the adjudicator to that same
    window. The body used to print a hardcoded `--days 14`, and a 14-day window is a different
    judgement, not a wider one: over 14 days a step two other runs also skipped leaves the
    expectation at strict support, so the printed command prints nothing about the run it was
    filed for. That dead reproducer is recorded in draft 2705's own adjudication, and 12 of the
    21 drafts filed inside the live 7-day window reproduce zero lines today."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    base._write_corpus(corpus, _recent(_specs(6, faulty_at=5, missing="write")))
    proc = _run_cli(corpus, backlog, "--escalate-writes", "--days", "7", "--support", "1.0")
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert "escalations_filed=1/1" in proc.stdout, proc.stdout[-2000:]
    drafts = _drafts(backlog)
    assert len(drafts) == 1, [p.name for p in drafts]
    assert _reproducer(drafts[0].read_text(encoding="utf-8")) == (
        "python3 scripts/step_conformance.py replay --days 7 --support 1.0"
        " | grep -F 'run_ok_5'"
    )


def test_a_draft_filed_at_a_loosened_support_carries_that_support_not_the_default(tmp_path):
    """#2944 clause 2. `--support` is not decoration: with `--days` it is what makes a step
    expected, so a draft filed at 0.9 is a different claim from one filed at the 1.0 default,
    and an adjudicator who re-runs the default re-runs a different detector. The value the
    invocation passed has to reach the body, not the value `DEFAULT_SUPPORT` happens to hold."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    base._write_corpus(corpus, _recent(_specs(6, faulty_at=5, missing="write")))
    proc = _run_cli(corpus, backlog, "--escalate-writes", "--days", "7", "--support", "0.9")
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert "escalations_filed=1/1" in proc.stdout, proc.stdout[-2000:]
    reproducer = _reproducer(_drafts(backlog)[0].read_text(encoding="utf-8"))
    assert "--support 0.9" in reproducer, reproducer
    assert "--support 1.0" not in reproducer, reproducer


def test_a_draft_filed_over_whole_history_omits_the_days_token_and_keeps_the_support(tmp_path):
    """#2944 clause 3, on the CLI's own default. `--days` is optional
    (`common.add_argument("--days", type=int, default=None)`) and `None` means the whole history,
    so the default invocation has to produce a command carrying no `--days` token at all:
    `--days None` is an unrunnable command, which is the failure mode a naive render of the
    value would ship. `--support` always has a value, so it is still named."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    base._write_corpus(corpus, _specs(6, faulty_at=5, missing="write"))
    proc = _run_cli(corpus, backlog, "--escalate-writes")
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert "escalations_filed=1/1" in proc.stdout, proc.stdout[-2000:]
    reproducer = _reproducer(_drafts(backlog)[0].read_text(encoding="utf-8"))
    assert "--days" not in reproducer, reproducer
    assert "--support 1.0" in reproducer, reproducer
    assert "grep -F 'run_ok_5'" in reproducer, reproducer


@pytest.mark.parametrize("days", [None, 3, 7, 14, 30])
def test_the_draft_never_prints_a_days_window_the_invocation_did_not_ask_for(tmp_path, days):
    """#2944 clause 4, over every window the filer can be handed. The printed token is the
    invoking replay's `days`, or no token when it passed none — so `--days 14` survives in a
    draft only because somebody actually ran `--days 14`. That is why the 14 case asserts the
    token IS printed: the fix carries the value, it does not delete the number. The whole-history
    corpus here is stamped two days back so every one of these windows flags the same run,
    otherwise a zero-window would be indistinguishable from a correct render."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    result = _replay(corpus, _recent(_specs(6, faulty_at=5, missing="write")), days=days)
    assert len(sc.escalation_targets(result)) == 1, (
        f"days={days}: the corpus stopped flagging, so this asserts nothing about the render")
    sc.file_escalations(result, backlog_dir=backlog)
    reproducer = _reproducer(_drafts(backlog)[0].read_text(encoding="utf-8"))
    if days is None:
        assert "--days" not in reproducer, reproducer
    else:
        assert f"--days {days} --support" in reproducer, reproducer
    assert ("--days 14" in reproducer) == (days == 14), reproducer
    assert "--support 1.0" in reproducer, reproducer


def test_the_window_changes_the_reproducer_line_and_nothing_else_in_the_body(tmp_path):
    """#2944 clause 5. Two escalating runs of two tasks, filed at `--days 7 --support 1.0`: still
    exactly one draft per run, both reproducers carry that window, and every field the body
    carried before this change is still there verbatim — task id, session source, run id, session
    key, run status, the offending step, the reason, `written_elsewhere`, and the `learned_from`
    ids. Naming the whole field block is what pins "only the reproducer line changed": a body
    that quietly dropped a field while gaining a flag would pass a test that only looked at the
    command."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    specs = _recent(_specs(6, faulty_at=5, missing="write"))
    specs += _recent(_specs(6, faulty_at=2, missing="write",
                           source="autonomy-task:8", prefix="run_other"))
    result = _replay(corpus, specs, days=7, support=1.0)
    assert len(_flagged(result)) == 2, [(r["task_id"], _steps(r)) for r in _flagged(result)]
    filed = sc.file_escalations(result, backlog_dir=backlog)
    assert len(filed) == 2, filed
    drafts = _drafts(backlog)
    assert len(drafts) == 2, [p.name for p in drafts]
    for draft in drafts:
        assert "--days 7 --support 1.0" in _reproducer(draft.read_text(encoding="utf-8"))
    text = _drafts(backlog)[0].read_text(encoding="utf-8")
    for line in (
        "- task_id: 7",
        "- session_source: autonomy-task:7",
        "- run_id: run_ok_5",
        "- session_key: 20260915_050050_autonomy_run_ok5",
        "- run_status: success",
        "- missing_step: write:report-latest.md",
        f"- reason: {sc.REASON_MISSING}",
        "- written_elsewhere: (none: no same-named write anywhere else in this run)",
        "- learned_from_n: 5 of 6 baseline runs",
        "- learned_from: run_ok_0, run_ok_1, run_ok_2, run_ok_3, run_ok_4",
        "- run_completed_at: ",
    ):
        assert line in text, f"the body lost the field line {line!r}"


# ---------------------------------------------------------------------------
# clause 4 — confined and non-alerting, across the seam the nightly job crosses
# ---------------------------------------------------------------------------


def _hash_tree(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_the_shipped_cli_files_the_draft_and_leaves_both_stores_byte_identical(tmp_path):
    """Clause 4 across the process boundary the nightly job actually crosses: `python3
    scripts/step_conformance.py replay --escalate-writes` as a subprocess, which is how the
    autonomy task invokes it. The draft appears, the escalation count prints beside its
    denominator, and the two stores the replay read — the ledger and the trajectory JSONL — are
    byte-for-byte what they were. Without this the filer is only proven against an in-process
    call no scheduler makes."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    base._write_corpus(corpus, _specs(6, faulty_at=5, missing="write"))
    before = _hash_tree(corpus)
    proc = subprocess.run(
        [sys.executable, str(MODULE_PATH), "replay", "--escalate-writes",
         "--trajectories", str(corpus / "trajectories"), "--db", str(corpus / "workers.db"),
         "--backlog-dir", str(backlog)],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert "escalations_filed=1/1" in proc.stdout, proc.stdout[-2000:]
    assert len(_drafts(backlog)) == 1
    assert "write:report-latest.md" in _drafts(backlog)[0].read_text(encoding="utf-8")
    assert _hash_tree(corpus) == before, "replay mutated the ledger or the trace store"


def test_replay_without_the_flag_files_nothing(tmp_path):
    """Clause 4's other half, and the item's clause 2: read-only is the default. Same corpus,
    same subprocess, no `--escalate-writes` — the report prints, the flagged run is still
    reported, and no draft is filed. A human running the replay to look at something cannot put
    work on the board by looking at it."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    base._write_corpus(corpus, _specs(6, faulty_at=5, missing="write"))
    proc = subprocess.run(
        [sys.executable, str(MODULE_PATH), "replay",
         "--trajectories", str(corpus / "trajectories"), "--db", str(corpus / "workers.db")],
        capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert "runs_flagged=1/" in proc.stdout
    assert "escalations_filed=" not in proc.stdout, "escalated without being asked"
    assert _drafts(backlog) == []


def _mixed_corpus(root: Path):
    """One task whose faulty run recorded `success`, and one whose faulty run recorded `failed`.

    Two tasks because at strict support two runs of one task missing the same step stop making
    that step expected at all (see the module docstring), and the suppression has to be judged
    on a corpus where both a filing run and a suppressed run are present — a corpus with only
    the aborted run could not tell the suppression from a filer that files nothing."""
    specs = _specs(6, faulty_at=5, missing="write")
    specs += _specs(6, faulty_at=2, missing="write", source="autonomy-task:8",
                    prefix="run_aborted", status="failed")
    return base._write_corpus(root, specs)


def _run_cli(corpus: Path, backlog: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(MODULE_PATH), "replay",
         "--trajectories", str(corpus / "trajectories"), "--db", str(corpus / "workers.db"),
         "--backlog-dir", str(backlog), *extra],
        capture_output=True, text=True, timeout=180,
    )


def test_the_aborted_run_keeps_its_report_line_and_its_header_counters(tmp_path):
    """#2477 clause 4, across the same process boundary the nightly filer crosses. On a corpus
    holding one green faulty run and one aborted faulty run of a second task, the read-only
    replay still prints the aborted run's `missing 'write:report-latest.md'` and its header
    still counts it: `runs_flagged=2/12` and `success_recorded_runs=11 flagged_among_them=1`.
    Those two numbers are the whole clause — move the suppression down into scoring and the
    first falls to `1/12`, and the aborted run disappears from the report a human reads. The
    filing run, on the same corpus and through the shipped CLI, is the green one alone:
    `escalations_filed=1/1 escalations_deduped=0` and one draft on disk naming `run_ok_5` and
    never `run_aborted_2`."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    _mixed_corpus(corpus)

    read_only = _run_cli(corpus, backlog)
    assert read_only.returncode == 0, read_only.stderr[-1500:]
    assert "runs_flagged=2/12 (" in read_only.stdout, read_only.stdout[-2000:]
    assert ("success_recorded_runs=11 flagged_among_them=1 quiet_among_them=10"
            in read_only.stdout), read_only.stdout[-2000:]
    assert "run_aborted_2 status=failed" in read_only.stdout, read_only.stdout[-2000:]
    assert read_only.stdout.count("missing 'write:report-latest.md'") == 2, read_only.stdout[-2000:]
    assert "escalations_filed=" not in read_only.stdout, "escalated without being asked"
    assert _drafts(backlog) == []

    filing = _run_cli(corpus, backlog, "--escalate-writes")
    assert filing.returncode == 0, filing.stderr[-1500:]
    assert "escalations_filed=1/1 escalations_deduped=0" in filing.stdout, filing.stdout[-2000:]
    drafts = _drafts(backlog)
    assert len(drafts) == 1, [p.name for p in drafts]
    text = drafts[0].read_text(encoding="utf-8")
    assert "run_ok_5" in text
    assert "run_aborted_2" not in text, "the aborted run filed a draft"


def test_the_escalation_path_makes_no_alert_call_and_writes_no_store():
    """Clause 4's confinement, read off the shipped source rather than asserted in prose, in two
    shapes. (a) The module names no alert surface anywhere: not the injection tool, and not the
    two elevated priorities it would be injected at. (b) Inside `file_escalations` — the one
    function allowed to write — the only write call is the shared backlog writer's own
    `new_item`: no sqlite handle, no statement, no notification symbol. A whole-module grep could
    not say (b), because `synth_fixtures` legitimately `INSERT`s into a temp ledger it owns."""
    source = MODULE_PATH.read_text(encoding="utf-8")
    for forbidden in ("session_inject_context", "notable", "urgent"):
        assert forbidden not in source, f"{forbidden} appears in the shipped detector"

    tree = ast.parse(source)
    fn = next(node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef) and node.name == "file_escalations")
    called = {node.func.id for node in ast.walk(fn)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    called |= {node.func.attr for node in ast.walk(fn)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert "new_item" in called, "the filer must write through the shared backlog writer"
    for forbidden in ("connect", "execute", "executescript", "session_inject_context",
                      "discord_notify", "notify", "urlopen", "requests", "post", "send"):
        assert forbidden not in called, f"file_escalations calls {forbidden}"
    body = ast.get_source_segment(source, fn) or ""
    for statement in ("INSERT ", "UPDATE ", "DELETE "):
        assert statement not in body, f"file_escalations carries the statement {statement!r}"


def test_the_draft_is_the_only_file_the_escalation_creates(tmp_path):
    """Confinement measured rather than read: the whole temp root is hashed before and after a
    filing call, so anything the escalation writes outside the backlog directory — a cache, a
    state file, a copy of its own report — shows up as an unexpected new path."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "escalation"
    result = _replay(corpus, _specs(6, faulty_at=5, missing="write"))
    before = set(_hash_tree(tmp_path))
    sc.file_escalations(result, backlog_dir=backlog)
    added = set(_hash_tree(tmp_path)) - before
    assert len(added) == 1, sorted(added)
    assert added.pop().startswith("escalation/"), "wrote outside the backlog directory"


# ---------------------------------------------------------------------------
# #2502 clause 3 — a disk witness in front of every escalation
#
# The trace is the only evidence the replay had, so an artifact written by a construct the
# parser does not read (`python3 - <<'EOF'`, a helper script) read as a dropped stage, and a
# deviation the disk disproves filed a draft anyway. Both corpora below put a real file at
# the artifact's *learned* path — the path the other runs of that task wrote it to — and put
# its mtime on one side or the other of `[run start − grace, run completed + grace]`:
# the run window defaults to 2026-09-10T05:00Z→05:12Z from `base._write_corpus`, and the
# module's own `DEFAULT_GRACE_SECONDS` is 6 h, so the window is 2026-09-09T23:00Z→
# 2026-09-10T11:12Z. The "after the window" draw is the 2026-10-07 and 2026-10-09 shape:
# the pointer's mtime was *later* than the flagged run, because a later night wrote it,
# which is why the bound has to be two-sided rather than "is the file newer than nothing".
# ---------------------------------------------------------------------------


def _witness_specs(count, *, report_path, swapped_path=None, faulty_at):
    """`count` runs that write the pointer at `report_path`, one of which either makes no
    write call at all (`swapped_path=None`, the 10-07 shape) or writes the same basename at
    `swapped_path` instead (the 10-09 shape)."""
    report_path, swapped_path = Path(report_path), (Path(swapped_path) if swapped_path else None)
    specs = []
    for i in range(count):
        faulty = i == faulty_at
        directory = swapped_path.parent if (faulty and swapped_path) else report_path.parent
        calls = [base._tool("Read", path=str(directory / "handoff-latest.md"), seq=1)]
        if not faulty or swapped_path is not None:
            calls.append(base._tool("Write", path=str(directory / report_path.name), seq=2))
        specs.append({
            "run_id": f"run_w_{i}",
            "session_key": f"2026091{i}_0500{i}0_autonomy_w{i}",
            "status": "success",
            "source": base.TASK,
            "task_id": 7,
            "calls": calls,
        })
    return specs


def _learned_artifact(root: Path, *, mtime: str | None) -> Path:
    """The artifact at its learned path, materialised, with its mtime placed on whichever
    side of the run window the test is about. `mtime=None` leaves the file at this instant."""
    path = root / "_pipeline" / "reflection" / "report-latest.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# report\n", encoding="utf-8")
    if mtime is not None:
        stamp = datetime.fromisoformat(mtime).timestamp()
        os.utime(path, (stamp, stamp))
    return path


def test_an_artifact_updated_at_its_learned_path_inside_the_window_is_reported_not_escalated(
    tmp_path, capsys
):
    """Clause 3, first half: the unparsed-write false alarm. The trace never names the write
    (a helper script performed it), the artifact's own mtime says it moved inside the run's
    window, and the deviation is reported under a reason that says so instead of filing a
    draft. It stays a deviation and stays in `runs_flagged` — the witness is a defence
    against escalation, not a verdict that nothing happened."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    report = _learned_artifact(corpus / "store", mtime="2026-09-10T06:00:00+00:00")
    result = _replay(corpus, _witness_specs(6, report_path=str(report), faulty_at=5))

    flagged = _flagged(result)
    assert [r["run_id"] for r in flagged] == ["run_w_5"]
    # The unparsed run is missing two steps — the `Write` tool call and the artifact write —
    # and only the artifact one is the subject of clause 3.
    deviation = next(d for d in flagged[0]["deviations"] if d["step"] == "write:report-latest.md")
    assert deviation["reason"] == sc.REASON_WITNESSED
    assert deviation["witnessed"] is True
    assert deviation["disk_witness"]["path"] == str(report)
    assert deviation["disk_witness"]["mtime"] == "2026-09-10T06:00:00+00:00"
    assert deviation["disk_witness"]["window_start"] == "2026-09-09T23:00:00+00:00"
    assert deviation["disk_witness"]["window_end"] == "2026-09-10T11:12:00+00:00"
    assert result["deviations_witnessed"] == 1
    # Two deviations for that run — the artifact write and the `Write` tool call it also
    # skipped — and the witness excuses exactly one of them. A witness that deleted the
    # deviation from the report would hide the run from `runs_flagged`, which is the
    # difference between "reported, not filed" and "nothing happened".
    assert result["deviations"] == 2

    assert sc.file_escalations(result, backlog_dir=backlog) == []
    assert _drafts(backlog) == []

    sc.print_replay(result)
    printed = capsys.readouterr().out
    assert "disk_witnessed=1" in printed, printed[-1500:]
    assert f"reason={sc.REASON_WITNESSED}" in printed, printed[-1500:]
    assert str(report) in printed, "the witness names the path it read"


@pytest.mark.parametrize("mtime", ["before", "after", "absent"])
def test_a_stale_or_absent_artifact_at_the_learned_path_still_escalates(tmp_path, mtime):
    """Clause 3, second half: the witness must not become an amnesty. Three stale draws —
    an mtime before the window opened, one after it closed (2026-10-07 and 2026-10-09, where
    a *later* night advanced the pointer), and no file at the learned path at all — and each
    still files exactly one draft, because nothing on disk proves this run wrote anything."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    store = corpus / "store"
    if mtime == "before":
        report = _learned_artifact(store, mtime="2026-09-09T22:59:59+00:00")
    elif mtime == "after":
        report = _learned_artifact(store, mtime="2026-09-10T11:12:01+00:00")
    else:
        report = store / "_pipeline" / "reflection" / "report-latest.md"
        assert not report.exists()

    result = _replay(corpus, _witness_specs(6, report_path=str(report), faulty_at=5))
    deviation = next(d for d in _flagged(result)[0]["deviations"]
                     if d["step"] == "write:report-latest.md")
    assert deviation["disk_witness"] is None, (
        f"{mtime}: a witness was recorded for an mtime outside the run's window"
    )
    assert deviation["witnessed"] is False, f"{mtime}: a stale mtime was read as a write"
    assert deviation["reason"] == sc.REASON_MISSING
    assert result["deviations_witnessed"] == 0

    filed = sc.file_escalations(result, backlog_dir=backlog)
    assert len(filed) == 1, f"{mtime}: the witness suppressed an escalation it must not"
    text = _drafts(backlog)[0].read_text(encoding="utf-8")
    assert "run_w_5" in text
    assert "never wrote report-latest.md" in text


def test_the_2026_10_09_path_swap_reports_missing_and_still_files(tmp_path):
    """Clauses 2 and 3 composed on the shape that opened this item. The run wrote
    `conversation-patterns-latest.md` under `knowledge/conversation/` while the pointer at
    the learned path kept the previous day's mtime: the step is reported missing
    (`written_at_other_path`), the disk witness runs on that deviation too and refuses to
    excuse it, and the draft is filed. Under the pre-#2502 detector this run was silent — it
    is listed in #2502's own `learned_from` as a run that took the step."""
    corpus, backlog = tmp_path / "corpus", tmp_path / "backlog"
    store = corpus / "store"
    report = _learned_artifact(store, mtime="2026-09-09T04:40:05+00:00")
    swapped = corpus / "vault" / "knowledge" / "conversation" / report.name

    result = _replay(corpus, _witness_specs(
        6, report_path=str(report), swapped_path=swapped, faulty_at=5))
    assert [d["step"] for d in _flagged(result)[0]["deviations"]] == ["write:report-latest.md"], (
        "the swap run made every tool call its baseline makes, so the artifact write at the "
        "wrong path is the only thing it is missing"
    )
    deviation = _flagged(result)[0]["deviations"][0]
    assert deviation["reason"] == sc.REASON_OTHER_PATH
    assert deviation["witnessed"] is False
    assert deviation["disk_witness"] is None, (
        "the pointer at the learned path kept 2026-09-09T04:40:05, two days before this run"
    )
    assert deviation["written_elsewhere"] == [str(swapped)]

    filed = sc.file_escalations(result, backlog_dir=backlog)
    assert len(filed) == 1, "the path swap filed nothing"
    assert "knowledge/conversation" in _drafts(backlog)[0].read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# clause 5 — the nightly task definition, in the live vault (marked, not skipped)
# ---------------------------------------------------------------------------


@pytest.mark.live_vault
def test_the_nightly_task_file_runs_the_replay_at_strict_support_with_no_alert_surface():
    """Clause 5. The task that makes the shipped detector run is a vault file landed through
    `automod_vault_land`, so it is read back through the real autonomy parser — a front-matter
    field the loader cannot read is a task that never dispatches, which is the exact failure this
    item is about. Marked `live_vault` because the gate's `-m "not live_vault"` run must not
    judge a file a human or a nightly job can rewrite between rounds."""
    from app.autonomy import _parse_task_file  # the real loader, not a re-implementation

    matches = sorted(AUTONOMY_DIR.glob("*-step-conformance-replay.md"))
    assert len(matches) == 1, [p.name for p in matches]
    path = matches[0]
    task = _parse_task_file(path)
    assert task, f"{path} does not parse as an autonomy task"
    description = str(task.get("description") or "")
    assert "scripts/step_conformance.py replay --days 7 --support 1.0" in description
    assert "--escalate-writes" in description, "the nightly run is the filer's only caller"
    assert "session_inject_context" not in description
    assert str(task.get("status", "")).strip() != "draft", "a draft never dispatches"
    # Both readers default a missing field to True, so the file has to carry the `false`: a
    # completion notice is an alert surface, which is the one thing this task may not name.
    assert task.get("notify_on_complete", True) is False, "the task declares a completion notice"
    skill = str(task.get("skill_name") or "").strip()
    assert skill, "dispatch refuses a task with no skill_name"
    skill_path = (Path("~/obsidian/skills") / skill / "SKILL.md").expanduser()
    assert skill_path.is_file(), f"skill_name names no skill on disk: {skill}"
    skill_text = skill_path.read_text(encoding="utf-8")
    assert "replay --days 7 --support 1.0" in skill_text
    for surface in ("session_inject_context", "notable", "urgent", "discord", "email"):
        assert surface not in skill_text, f"the skill names an alert surface: {surface}"


#: The clause-5 witness, committed in the diff rather than read from the live vault. The
#: autonomy file itself is landed by `automod_vault_land` and so cannot ride in this diff —
#: a gate that judges a file its own diff cannot carry grades an absence — and the
#: `live_vault` node above is deselected by the gate's `-m "not live_vault"`. This copy is
#: what makes the diff carry a witness: the real loader parses it, and the live node is what
#: notices the copy has drifted from the file the scheduler actually reads.
FIXTURE_DIR = REPO_ROOT / "tests/fixtures/step_conformance_replay"
FIXTURE_TASK = FIXTURE_DIR / "nightly-task.md"
FIXTURE_SKILL = FIXTURE_DIR / "SKILL.md"

#: Every front-matter key `_update_task_field` (`app/autonomy.py:174`) writes about a RUN
#: rather than about the job: `status` (`:120`, `:582`, `:4069`, `:4506`), `updated`
#: (`:120`, `:582`, `:4069`, `:4506`), `next_run` (`:582`, and the dict `_record_failure`
#: splats at `:3752`), `last_run` and `last_attempt` (`:3037`, `:4506`), `failure_count`
#: (`:582`, `:3037`, `:4506`), and `infra_failure_count` with `infra_rest_until` (`:4506`).
#: This is the task's clock, not its instruction: complete one run of task 93 and `status`,
#: `updated`, `last_run`, `last_attempt` and `next_run` all move with nobody having edited
#: the job, so a witness that graded them had to be re-committed after every daily run to
#: stay green — which is the red tree #2118 is. `_append_activity_log` (`app/autonomy.py:324`)
#: writes the same fact into the body as one bullet under `ACTIVITY_LOG_HEADING`, so that
#: section is out for the same reason. Dropping `status` costs no coverage clause 5 needs:
#: the copy's own node above asserts the copy declares `up_next`, while the live file's
#: `status` is `in_progress` for the whole length of every run by construction (`:4069`).
SCHEDULER_OWNED_KEYS = frozenset({
    "status", "updated", "next_run", "last_run", "last_attempt",
    "failure_count", "infra_failure_count", "infra_rest_until",
})


def _witnessed_surface(path: Path) -> tuple[dict, str]:
    """The part of a task file the committed copy is a witness for: the front matter minus
    `SCHEDULER_OWNED_KEYS`, and the body up to the Activity Log the scheduler appends to.

    Both sides go through the real loader on purpose. `_update_task_field` rewrites the whole
    block with `yaml.dump`, which re-quotes `'runs_flagged=\\d+/\\d+'` as plain and re-wraps
    the folded `description` at its own width without changing one field's value, and that
    re-styling is the other half of what the byte comparison mistook for drift at 7069eebd.
    """
    from app.autonomy import ACTIVITY_LOG_HEADING, _parse_task_file  # the real loader

    task = _parse_task_file(path)
    assert task is not None, f"the real loader cannot read {path}"
    body = str(task.pop("body", ""))
    cut = body.find(ACTIVITY_LOG_HEADING)
    if cut >= 0:
        body = body[:cut]
    surface = {k: v for k, v in task.items()
               if not str(k).startswith("_") and k not in SCHEDULER_OWNED_KEYS}
    return surface, body.rstrip()


@pytest.fixture(scope="module")
def nightly_task():
    from app.autonomy import _parse_task_file  # the real loader, not a re-implementation

    task = _parse_task_file(FIXTURE_TASK)
    assert task is not None, f"the real loader cannot read {FIXTURE_TASK}"
    return task


def test_the_nightly_task_fixture_runs_the_replay_at_strict_support(nightly_task):
    """Clause 5, graded on a copy the diff itself carries: the command is the read-only
    replay at strict support (support 1.0 is the shipped default, so the task states it
    anyway and this node asserts the number) and the file names no alert surface."""
    task = nightly_task
    # `_parse_task_file` exposes no `command` key, so the command the scheduler runs is the one
    # in the description — which is also what `_build_task_prompt` renders to the model.
    description = str(task.get("description") or "")
    assert "scripts/step_conformance.py replay --days 7 --support 1.0" in description
    assert str(task.get("status", "")).strip() == "up_next", "a draft never dispatches"
    for surface in ("session_inject_context", "notable", "urgent", "email"):
        assert surface not in description, f"the description names an alert surface: {surface}"
    assert task.get("notify_on_complete", True) is False, "the task declares a completion notice"


def test_the_nightly_task_fixture_names_a_skill_that_exists_on_disk(nightly_task):
    """Clause 5's other half: dispatch refuses a task with no `skill_name`, and
    `_build_task_prompt` renders a task whose skill file is missing into a stub, so the
    skill has to exist and be the protocol the report is run from."""
    skill = str(nightly_task.get("skill_name") or "").strip()
    assert skill, "dispatch refuses a task with no skill_name"
    assert FIXTURE_SKILL.is_file(), "the committed skill copy is missing"
    text = FIXTURE_SKILL.read_text(encoding="utf-8")
    assert f"name: {skill}" in text, f"the copy is not the skill the task names: {skill}"
    assert "scripts/step_conformance.py replay --days 7 --support 1.0" in text
    for surface in ("session_inject_context", "notable", "urgent", "discord"):
        assert surface not in text, f"the skill names an alert surface: {surface}"


def test_the_nightly_task_fixture_copy_has_not_drifted_from_the_live_task():
    """The fixture is a copy, so it only witnesses clause 5 while it matches the file the
    scheduler reads — and the match that matters is the dispatched surface
    (`_witnessed_surface`), not the bytes. Red here still means the vault task changed and
    the committed witness is stale; it no longer means the scheduler stamped its own clock
    onto the file since the copy was taken. At base 7069eebd the two files differed in
    nothing but `last_attempt`, `last_run`, `next_run` and `updated` — restamped at
    11:00:22Z by run_93_20261003_110001 — one bullet that run appended under
    `## Activity Log`, and the `yaml.dump` re-quote and re-wrap that same writer applies to
    the rest of the block: no field of the job had moved, and task 93 runs daily, so the
    bytes could only be kept equal by re-committing the copy after every run. Update the
    copy, or the file, not this node."""
    live_task = sorted(AUTONOMY_DIR.glob("*-step-conformance-replay.md"))
    live_skill = (Path("~/obsidian/skills") / "step-conformance-replay" / "SKILL.md").expanduser()
    if not live_task or not live_skill.is_file():
        pytest.skip("the vault task or its skill has not landed yet")
    copy_surface, copy_body = _witnessed_surface(FIXTURE_TASK)
    live_surface, live_body = _witnessed_surface(live_task[0])
    assert (copy_surface, copy_body) == (live_surface, live_body), (
        f"{FIXTURE_TASK} and {live_task[0].name} disagree: front-matter keys "
        f"{sorted(k for k in set(copy_surface) | set(live_surface)
                  if copy_surface.get(k) != live_surface.get(k))}, "
        f"body prose differs: {copy_body != live_body}")
    # The skill copy stays a byte comparison: no scheduler writer touches a SKILL.md, so
    # every byte difference there is somebody editing the protocol the run is driven from.
    assert FIXTURE_SKILL.read_text(encoding="utf-8") == live_skill.read_text(encoding="utf-8"), (
        f"{FIXTURE_SKILL} and {live_skill} disagree")


def test_the_drift_witness_still_bites_on_an_edited_instruction(tmp_path):
    """The other half of narrowing the comparison, and the reason it is not a weakening: a
    real edit to any key the witness keeps still turns this red. The three mutants are edits
    a person or a nightly job actually makes to a task file, one per key the clause-5 nodes
    above assert on — re-pointing `skill_name`, giving the job the completion notice clause 5
    forbids (`notify_on_complete: false` -> `true`), and dropping the `runs_flagged=N/M` line
    the description orders pasted verbatim out of the description. Each edits the
    front-matter block only, and each needle is asserted absent from the body first: the same
    words do occur down there (`--support 1.0` among them), and a file-wide replace would
    report drift off the body prose while the field under test went unwitnessed — which is
    exactly how a witness like this goes quietly blind. What it must NOT report is the
    scheduler's own restamp, which the node below pins through the shipped writer."""
    blank, front_matter, body = FIXTURE_TASK.read_text(encoding="utf-8").split("---\n", 2)
    for key, needle, replacement in (
            ("skill_name", "skill_name: step-conformance-replay", "skill_name: other-skill"),
            ("notify_on_complete", "notify_on_complete: false", "notify_on_complete: true"),
            ("description", "runs_flagged=N/M", "runs_flagged=X/Y")):
        assert needle in front_matter, (
            f"{needle!r} is not in the copy's front matter, so this mutant is not the "
            f"{key} edit it claims")
        assert needle not in body, (
            f"{needle!r} also occurs in the body, so this mutant would not isolate {key}")
        mutant = tmp_path / f"{key}-mutant.md"
        mutant.write_text(f"{blank}---\n"
                          + front_matter.replace(needle, replacement)
                          + f"---\n{body}", encoding="utf-8")
        assert _witnessed_surface(mutant) != _witnessed_surface(FIXTURE_TASK), (
            f"editing {key} in the front matter ({needle!r} -> {replacement!r}) "
            "left the drift witness blind")


def test_the_drift_witness_survives_the_shipped_writer_stamping_its_own_run_fields(
        tmp_path, monkeypatch):
    """#2118's own regression, pinned across the seam instead of in prose.
    `_update_task_field` and `_append_activity_log` are the functions the scheduler runs when
    a task completes, and they are what reddened the tree: they rewrote the live task file
    under the committed copy. Running them over that copy here must leave the witnessed
    surface untouched while the bytes on disk demonstrably change — `yaml.dump` re-quotes and
    re-wraps the block on the way through, which is why comparing bytes could not survive one
    run and comparing parsed fields does. If a future writer starts stamping a key that is
    not in `SCHEDULER_OWNED_KEYS`, this node names it; if `SCHEDULER_OWNED_KEYS` is widened
    past the writer's own keys, the node above goes red."""
    import app.autonomy as autonomy

    stamped = tmp_path / "93-step-conformance-replay.md"
    stamped.write_text(FIXTURE_TASK.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", tmp_path)
    completed = "2026-10-04T11:00:22.509721+00:00"
    # The success path's own kwargs, `app/autonomy.py:4506`.
    autonomy._update_task_field(
        93, status="up_next", last_run=completed, last_attempt=completed,
        updated=completed, failure_count=0, infra_failure_count=0,
        infra_rest_until=None, next_run="2026-10-05T11:00:00+00:00")
    autonomy._append_activity_log(93, "Run run_93_20261004_110001 — success (21s)")
    assert stamped.read_text(encoding="utf-8") != FIXTURE_TASK.read_text(encoding="utf-8"), (
        "the shipped writer left the file alone, so this node never crossed the seam")
    assert _witnessed_surface(stamped) == _witnessed_surface(FIXTURE_TASK), (
        "one run of the task, through the writer that makes it, now reads as drift")


def test_the_gate_selection_collects_the_drift_witness_and_it_reaches_a_verdict():
    """Clause 2 as behaviour, not as a claim about this file: run the drift witness's own
    node id out of process under the tests rung's exact selection — the constant is imported
    from `scripts/automod/gate.py`, not restated, so what this pins is the expression the
    gate runs. Reaching `1 passed` there means the node was COLLECTED, so it is neither
    deleted nor marked `live_vault` (which `-m "not live_vault"` would have deselected), and
    that it RAN TO A VERDICT, so it is neither skipped nor xfailed. All four of those answer
    a red tree by not running the test, which is what #2118 forbids; each one trips exactly
    one assert below."""
    from scripts.automod.gate import TESTS_MARK_EXPR

    node = (f"{Path(__file__).resolve().relative_to(REPO_ROOT)}::"
            f"{test_the_nightly_task_fixture_copy_has_not_drifted_from_the_live_task.__name__}")
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "-m", TESTS_MARK_EXPR, "-rs", "-rx", node],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    out = run.stdout + run.stderr
    assert "deselected" not in out, f"{node} is deselected by {TESTS_MARK_EXPR!r}: {out[-800:]}"
    assert "skipped" not in out.lower(), f"{node} skipped rather than ran: {out[-800:]}"
    assert "xfail" not in out.lower(), f"{node} was xfailed rather than run: {out[-800:]}"
    assert "1 passed" in out, f"{node} did not reach a pass: {out[-1500:]}"
    assert run.returncode == 0, f"pytest exited {run.returncode}: {out[-1500:]}"


# ---------------------------------------------------------------------------
# #2708 — a skip the run declared in its own terminal report is not a dropped stage
#
# Why this section exists: task 39's 2026-10-11 run wrote its completion report and, in that
# report's `Not written, with reasons` block, named `conversation-patterns-latest.md` as
# deliberately not written with the guard output pasted under it — the second of the two options
# `skills/nightly-reflection-knowledge-write/steps-2b-2f.md` offers. The escalation gate read
# neither the block nor the licence, filed #2705 as a dropped stage, and the draft asserted the
# run had produced no evidence of its own choice. The de-escalation below stops the draft; the
# report line stays, because "the trace names no write" remains true even when the run explains
# itself, and it is the replay's job to say what the trace shows.
#
# The fixtures are side by side on purpose: clause 3's corpus is clause 1's with one word changed
# in the report — the run names *other* files and never the missing basename, which is the
# #2502 2026-10-07 shape that must keep filing. A de-escalation that cannot tell those two
# reports apart is an amnesty, not a fix.
# ---------------------------------------------------------------------------

#: The artifact the fixture task publishes every night and the faulty night does not — the shape
#: that fired, task 39's `conversation-patterns-latest.md`.
POINTER_ARTIFACT = "patterns-latest.md"

#: The same task's terminal report, under a basename no per-task path table knows. The detector
#: has to reach it through the baseline's learned artifact steps, which is only checkable if the
#: name means nothing to the code.
RUN_REPORT = "midnight-summary-latest.md"

#: §2e's own heading, as the skill words it and as `tests/test_skill_reflection_archive.py` pins
#: that wording in prose against a future edit of the skill. The detector anchors on this literal
#: so the two surfaces move together or one of these tests goes red.
NOT_WRITTEN_HEADING = "## Not written, with reasons"


def _declared_report(*, names: list[str], written: list[str] = ()) -> str:
    """A terminal report: a not-written block naming `names`, then a section after the block
    naming the files in `written` as written."""
    block = "\n".join(f"- **`{name}`** — not written; its handoff section was empty."
                      for name in names)
    after = "\n".join(f"- `{name}` — written, archive copy taken." for name in written)
    return (f"# Midnight summary\n\n## Files written\n\n- `{RUN_REPORT}`.\n\n"
            f"{NOT_WRITTEN_HEADING}\n\n{block}\n\n## Archive copies\n\n{after}\n")


def _pointer_replay(tmp_path: Path, *, report_text: str | None, stray_text: str | None = None,
                    report_by_every_run: bool = True, faulty_at: int = 5, count: int = 6,
                    report_stamp: str = "inside", pointer_elsewhere: bool = False):
    """Six runs of one task, replayed through the real loaders.

    Every run writes its terminal report — that is the artifact the baseline learns, and the one
    the detector reads to find a declared skip — and every run except the faulty one also
    publishes the pattern pointer, so `write:patterns-latest.md` is expected of the faulty run by
    the same support arithmetic the other corpora here use.

    `report_text` is what is on disk at the report's learned path when the replay runs; with
    `report_by_every_run=False` no run's trace names that write, so the task's baseline holds no
    learned report artifact even though the file sits there. `stray_text` is what is on disk in a
    scratch note only the faulty run writes — a path the baseline never learned, because one run
    in six is not a shape. Returns the replay result, so a test asserts on the same dicts
    `escalation_targets` was handed.

    `report_stamp` is the report file's mtime, the evidence the check reads: `inside` (the
    default) stamps it one minute before the ledger's `completed_at` for these runs, where a
    report the run itself wrote has to fall; `stale` stamps it the previous evening, a report this
    run did not write. The window is not invented here — `_write_corpus` defaults every row to
    `started_at` 2026-09-10T05:00:00+00:00 and `completed_at` 05:12:00+00:00, which is what the
    two fixtures below assert against."""
    root = tmp_path / "corpus"
    reflection = root / "reflection"
    report_dir = reflection / "summaries"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / RUN_REPORT
    pointer = reflection / POINTER_ARTIFACT
    stray = reflection / "scratch-note-latest.md"
    if report_text is not None:
        report.write_text(report_text, encoding="utf-8")
    if stray_text is not None:
        stray.write_text(stray_text, encoding="utf-8")
    specs = []
    for i in range(count):
        faulty = i == faulty_at
        # Every run writes a running notes file, so `tool:Write` is a step no shape here drops:
        # with `report_by_every_run=False` the faulty run would otherwise make no write call at
        # all, and the corpus would flag two steps — leaving a test unable to say which one the
        # escalation gate acted on, which is the property every other corpus in this file has.
        calls = [base._tool("Read", path=str(reflection / "handoff-latest.md"), seq=1),
                 base._tool("Write", path=str(reflection / "notes-latest.md"), seq=2)]
        seq = 3
        if report_by_every_run:
            calls.append(base._tool("Write", path=str(report), seq=seq))
            seq += 1
        if stray_text is not None and faulty:
            calls.append(base._tool("Write", path=str(stray), seq=seq))
            seq += 1
        if not faulty:
            calls.append(base._tool("Write", path=str(pointer), seq=seq))
        elif pointer_elsewhere:
            # The #2502 swap shape on top of a declared skip: this run wrote a same-named file at
            # a directory its task's history never writes the artifact to, and also named the
            # basename in its not-written block. The step stays unsatisfied (wrong directory), so
            # a deviation exists and `same_name_elsewhere` is populated — the one combination the
            # reason chain used to resolve in favour of the draft.
            swapped = root / "vault-derived" / "conversation"
            swapped.mkdir(parents=True, exist_ok=True)
            calls.append(base._tool("Write", path=str(swapped / POINTER_ARTIFACT), seq=seq))
        specs.append({"run_id": f"run_dp_{i}",
                      "session_key": f"2026091{i}_0500{i}0_autonomy_dp{i}",
                      "status": "success", "source": base.TASK,
                      "task_id": int(base.TASK.rsplit(":", 1)[1]), "calls": calls})
    traj, db = base._write_corpus(root, specs)
    # The ledger rows exist only now, so the report's mtime — the evidence the check reads — can
    # only be placed relative to them here. `_write_corpus` stamps every row 2026-09-10T05:00:00
    # to 05:12:00+00:00, so `inside` is 05:11:00 and `stale` the evening before the run started.
    stamp = {"inside": "2026-09-10T05:11:00+00:00",
             "stale": "2026-09-09T22:00:00+00:00"}[report_stamp]
    epoch = datetime.fromisoformat(stamp).timestamp()
    os.utime(report, (epoch, epoch))
    return sc.replay(trajectory_dir=traj, db_path=db)


def test_a_declared_skip_beats_the_swap_shape_it_must_not_reopen(tmp_path, capsys):
    """Clause 1 against #2502, in the one case where the two suppressions can collide. The run
    declared the basename in its not-written block *and* left a same-named file at a directory the
    task's history never writes it to, so the deviation carries `written_elsewhere` and used to be
    reasoned `artifact_written_at_other_path` — an escalable reason — which files exactly the draft
    #2708 is about, with `written_elsewhere: (none)` as far as the reader can see it. The declared
    skip now decides the reason, and the swap survives as printed evidence rather than as a draft:
    de-escalation has to remove the accusation, never the observation."""
    result = _pointer_replay(tmp_path, report_text=_declared_report(names=[POINTER_ARTIFACT]),
                             pointer_elsewhere=True)
    flagged = _flagged(result)
    assert _steps(flagged[0]) == [f"write:{POINTER_ARTIFACT}"], _steps(flagged[0])
    deviation = flagged[0]["deviations"][0]
    assert deviation["reason"] == sc.REASON_DECLARED_SKIP, deviation["reason"]
    assert deviation["declared_in"].endswith(RUN_REPORT), deviation["declared_in"]
    elsewhere = deviation["written_elsewhere"]
    assert elsewhere == [str(tmp_path / "corpus" / "vault-derived" / "conversation"
                          / POINTER_ARTIFACT)], elsewhere
    assert sc.escalation_targets(result) == [], sc.escalation_targets(result)
    sc.print_replay(result)
    printed = capsys.readouterr().out
    assert f"written at another path instead: {elsewhere}" in printed, printed
    assert f"reason={sc.REASON_DECLARED_SKIP}" in printed, printed


def test_a_skip_named_in_the_runs_own_report_is_reasoned_out_and_files_no_draft(tmp_path):
    """Clause 1, and the false alarm #2705 is made of. The run dropped the pointer and named
    exactly that basename inside its report's not-written block, so the deviation carries a reason
    outside `ESCALABLE_REASONS` and the escalation gate has no target for it."""
    result = _pointer_replay(tmp_path, report_text=_declared_report(names=[POINTER_ARTIFACT]))
    flagged = _flagged(result)
    assert _steps(flagged[0]) == [f"write:{POINTER_ARTIFACT}"], _steps(flagged[0])
    reason = flagged[0]["deviations"][0]["reason"]
    assert reason == sc.REASON_DECLARED_SKIP, reason
    assert sc.REASON_DECLARED_SKIP not in sc.ESCALABLE_REASONS
    assert sc.escalation_targets(result) == []
    backlog = tmp_path / "backlog"
    assert sc.file_escalations(result, backlog_dir=backlog) == []
    assert _drafts(backlog) == []


def test_a_declared_skip_is_still_printed_and_still_counted_as_flagged(tmp_path, capsys):
    """Clause 2: de-escalation suppresses the draft, never the observation. The run is still in
    `runs_flagged` and `deviations`, `print_replay` still prints its missing step, and the reason
    on that printed line is the new one — which is what lets a reader tell "declared" from
    "unexplained" without opening the report."""
    result = _pointer_replay(tmp_path, report_text=_declared_report(names=[POINTER_ARTIFACT]))
    assert result["runs_flagged"] == 1, result["deviations"]
    assert result["deviations"] == 1
    assert any(r["run_id"] == f"run_dp_{5}" and r["deviations"] for r in result["reports"])
    sc.print_replay(result)
    printed = capsys.readouterr().out
    assert "run_dp_5" in printed, printed
    assert f"missing 'write:{POINTER_ARTIFACT}'" in printed, printed
    assert f"reason={sc.REASON_DECLARED_SKIP}" in printed, printed


def test_the_2502_shape_a_skip_the_report_never_named_still_files(tmp_path):
    """Clause 3, the regression pair to the test above. Same corpus, same heading, same block —
    the report names two *other* files and never the missing basename, which is what task 39's
    2026-10-07 report did. That run filed, and it still must: naming some skip is not naming this
    one, and #2502 exists precisely because that night's pointer went unpublished with no
    declaration."""
    result = _pointer_replay(tmp_path, report_text=_declared_report(
        names=["tool-patterns-latest.md", "user-profile-latest.md"]))
    flagged = _flagged(result)
    assert _steps(flagged[0]) == [f"write:{POINTER_ARTIFACT}"]
    assert flagged[0]["deviations"][0]["reason"] == sc.REASON_MISSING
    targets = sc.escalation_targets(result)
    assert [t["step"] for t in targets] == [f"write:{POINTER_ARTIFACT}"]
    backlog = tmp_path / "backlog"
    assert len(sc.file_escalations(result, backlog_dir=backlog)) == 1
    assert len(_drafts(backlog)) == 1


@pytest.mark.parametrize("shape", ["named_outside_the_block", "no_learned_report",
                                   "declared_only_in_an_unlearned_file",
                                   "report_predates_the_run"])
def test_nothing_else_de_escalates(tmp_path, shape):
    """Clause 4, the four shapes that must keep filing. A report that mentions the basename only
    outside the not-written block has not declared a skip — §2e's licence is a named block, and
    matching the string anywhere in a report would excuse a run that wrote the file and a run that
    bragged about it alike. A task whose baseline learned no report artifact at all cannot be
    de-escalated by a report, so the old behaviour has to be all that changes. And a declaration
    in a file only the faulty run wrote is not a report: the artifact the detector reads is one the
    *baseline* learned, which is why no per-task path table is needed and why one run in six does
    not qualify. And a report on disk whose mtime is the previous evening is somebody else's
    report: the trace names the write, the file was never touched by this run, and the declaration
    inside it cannot be this run's — the shape the window's `started` bound exists for."""
    silence = _declared_report(names=["tool-patterns-latest.md"])
    if shape == "named_outside_the_block":
        text = _declared_report(names=["tool-patterns-latest.md"], written=[POINTER_ARTIFACT])
        result = _pointer_replay(tmp_path, report_text=text)
    elif shape == "no_learned_report":
        result = _pointer_replay(tmp_path, report_text=_declared_report(names=[POINTER_ARTIFACT]),
                                 report_by_every_run=False)
    elif shape == "report_predates_the_run":
        # The same report text and the same learned write in the trace, an mtime the previous
        # evening. `run_report_paths` still names the path, so the only thing refusing to read it
        # is the window; the identical corpus stamped one minute before `completed_at` de-escalates
        # instead, which is the pair that makes that bound the difference rather than decoration.
        result = _pointer_replay(tmp_path, report_text=_declared_report(names=[POINTER_ARTIFACT]),
                                 report_stamp="stale")
    else:
        # The report itself declares nothing; the declaration is in the note only this run wrote.
        result = _pointer_replay(tmp_path, report_text=silence,
                                 stray_text=_declared_report(names=[POINTER_ARTIFACT]))
    flagged = _flagged(result)
    assert _steps(flagged[0]) == [f"write:{POINTER_ARTIFACT}"], (shape, _steps(flagged[0]))
    assert flagged[0]["deviations"][0]["reason"] == sc.REASON_MISSING, shape
    assert [t["step"] for t in sc.escalation_targets(result)] == [f"write:{POINTER_ARTIFACT}"]
    backlog = tmp_path / "backlog"
    assert len(sc.file_escalations(result, backlog_dir=backlog)) == 1, shape


def test_the_declaration_is_read_only_inside_the_not_written_block():
    """The matcher's own edges, pinned beside the corpus tests above because clause 4's first
    shape depends on them: the block is a heading-bounded region, not a substring search, and the
    name inside it is compared the way every other artifact name in this module is — through
    `normalized_basename`, so a report that names `patterns-2026-10-11.md` declares the skip on
    `patterns-<date>.md`."""
    text = _declared_report(names=[POINTER_ARTIFACT], written=["handoff-latest.md"])
    assert sc.declared_skip_in_report(text, POINTER_ARTIFACT)
    assert not sc.declared_skip_in_report(text, "handoff-latest.md"), \
        "a name after the block closed was read as a declaration"
    assert not sc.declared_skip_in_report(text, "no-such-latest.md")
    dated = ("# Midnight summary\n\n"
             f"{NOT_WRITTEN_HEADING}\n\n- `patterns-2026-10-11.md` — not written.\n")
    assert sc.declared_skip_in_report(dated, "patterns-<date>.md")
    assert not sc.declared_skip_in_report(
        "# Midnight summary\n\n## Files written\n\n- `p-latest.md`\n", "p-latest.md")
