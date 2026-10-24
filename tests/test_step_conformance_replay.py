"""#673 clauses 4 and 5 — the detector is accepted against injected faults, not a vibe.

Clause 4: on a replay of the fixture corpus all three injected faults fire as structural
deviations — the presence/lock stamp never written, a dependent run started while its
upstream artifact did not exist, and the nightly reflection handoff never written — and the
benign control fixture fires none.

Clause 5: every deviation carries the task id, the run/session id, the offending step, the
learned expected set and the run ids it was learned from (no bare score), and the replay
prints the detection count beside the false-alarm count on the benign corpus, so a
positives-only tuning cannot report a pass.

The corpus is built as the two real stores — trajectory JSONL plus a sqlite `runs` ledger with
`meta_json.session_ids` — and read back through the same read-only loaders a nightly replay
uses, so these tests exercise the join and the comparison, not just the comparison. Each fault
re-creates one of Lloyd's own confirmed incidents, so a pass here means the detector catches
the failures this machine actually had.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "step_conformance.py"
# Every fixture run ended on 2026-09-20T05:12Z; a month later the grace window has lapsed for
# all of them, so nothing is left `pending` and the counts below mean what they say.
CHECKED = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
GRACE_SECONDS = 6 * 3600


def _load_module():
    spec = importlib.util.spec_from_file_location("step_conformance_replay", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sc = _load_module()


@pytest.fixture()
def fixtures(tmp_path):
    return sc.synth_fixtures(tmp_path / "fixtures")


@pytest.fixture()
def replayed(fixtures):
    """The fixture corpus through the same read-only replay path the nightly job uses."""
    return sc.replay(
        trajectory_dir=fixtures["trajectory_dir"],
        db_path=fixtures["db"],
        days=None,
        now=CHECKED,
        grace_seconds=GRACE_SECONDS,
        min_runs=3,
    )


def _report(result, run_id):
    return next(r for r in result["reports"] if r["run_id"] == run_id)


def _fault_ids(fixtures):
    return {f["run_id"]: f for f in fixtures["injected"]}


# ---------------------------------------------------------------------------
# clause 4 — the three injected faults, and the benign control
# ---------------------------------------------------------------------------


def test_all_three_injected_faults_fire_as_structural_deviations(fixtures, replayed):
    """Clause 4. Each fault deletes one call from an otherwise clean run and leaves the ledger
    saying `success`; the replay must name the deleted step on each one, structurally."""
    faults = _fault_ids(fixtures)
    assert len(faults) == 3, "three incidents are what the acceptance names"
    for run_id, fault in faults.items():
        report = _report(replayed, run_id)
        assert report["deviations"], f"{fault['incident']} was not caught"
        steps = {d["step"] for d in report["deviations"]}
        assert fault["expected_missing_step"] in steps
        assert all(d["kind"] == "structural" for d in report["deviations"])
        assert report["run_status"] == "success", "the faults were recorded successful"
        assert report["pending"] == []


def test_the_three_faults_are_the_three_incidents_the_acceptance_names():
    """The corpus must stay tied to the incidents, or "3/3 detected" stops meaning anything:
    the labelled fault steps are read back from what the builder writes, not from a copy of
    the expected values kept next to the assertion."""
    named = {
        spec["incident"]: spec["expected_missing_step"] for spec in _labelled_faults()
    }
    assert named == {
        "presence/lock stamp never written (09-03 -> 09-09 gate)": "write:.consolidate-lock",
        "dependent ran before its upstream artifact existed (09-08)": "read:knowledge-handoff-<date>.md",
        "nightly reflection handoff never written (09-01/09-02)": "write:knowledge-handoff-<date>.md",
    }


def _labelled_faults():
    """Build the corpus once in-memory to read its labels back, so the assertion above is
    checked against what the builder actually writes rather than a copy of it."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        return sc.synth_fixtures(tmp)["injected"]


def test_the_benign_control_fixtures_fire_none(fixtures, replayed):
    """Clause 4, other half. The benign corpus is not a copy of the clean run: it includes an
    extra call, a reversed order and the artifact step emitted last, and all of them must stay
    quiet — an order-sensitive detector would flag all three variants."""
    benign = fixtures["benign"]
    assert len(benign) == 18, "6 controls per task, 3 tasks"
    flagged = [r for r in replayed["reports"] if r["run_id"] in set(benign) and r["deviations"]]
    assert flagged == []
    # A read-only replay has no labels: `success_recorded_flagged` counts the 3 injected
    # faults, which are indistinguishable from false alarms without a label — which is why
    # replay prints an upper bound and only `validate` prints a measured rate.
    assert replayed["success_recorded_flagged"] == 3
    assert replayed["success_recorded_quiet"] == 18

    variants = {key: sorted(sc.steps_from_tools([_call(spec) for spec in value(sc.FIXTURE_TASKS["autonomy-task:47"]["clean"])]))
                for key, value in sc.BENIGN_VARIANTS.items()}
    assert variants["plain"] == variants["reordered"], (
        "the reordered control must carry exactly the same steps — if it did not, the control "
        "would be testing something other than order-insensitivity"
    )
    assert "tool:vault_search" in variants["extra"], (
        "the extra control must add a step no clean run makes — an extra call that the clean "
        "shape already contained would test nothing"
    )
    assert len(variants["extra"]) > len(variants["plain"])
    late_calls = sc.BENIGN_VARIANTS["late"](sc.FIXTURE_TASKS["autonomy-task:47"]["clean"])
    assert [c["tool"] for c in late_calls] != [c["tool"] for c in sc.FIXTURE_TASKS["autonomy-task:47"]["clean"]], (
        "the late-span control must really move the last call forward — otherwise it is just "
        "the clean run again and the control proves nothing"
    )
    assert variants["late"] == variants["plain"], (
        "a step emitted out of its learned position is late, not missing: same step set"
    )


def _call(spec):
    params = {"summary": "control step"}
    if "file_path" in spec:
        params["file_path"] = spec["file_path"]
    if "command" in spec:
        params["command"] = spec["command"]
    return {"name": spec["tool"], "params_summary": params, "is_error": False, "sequence": 1}


def test_the_benign_controls_are_scored_not_skipped(fixtures, replayed):
    """The quiet control only means something if it was actually looked at: every labelled
    benign run must appear in the scored set with a baseline behind it."""
    scored = {r["run_id"] for r in replayed["reports"]}
    assert set(fixtures["benign"]) <= scored
    for run_id in fixtures["benign"]:
        report = _report(replayed, run_id)
        assert report["run_status"] == "success"
        assert report["session_source"].startswith("autonomy-task:")


# ---------------------------------------------------------------------------
# clause 5 — evidence on every deviation, and the two numbers printed together
# ---------------------------------------------------------------------------


def test_every_deviation_carries_the_expectation_it_was_judged_against(replayed):
    """Clause 5, first half. An alert that says only '0.91' is his health-score-is-22 failure:
    each deviation must name the task, the run and session, the step, the whole learned set,
    and the run ids that produced it."""
    flagged = [r for r in replayed["reports"] if r["deviations"]]
    assert flagged, "the fixture replay must contain deviations to inspect"
    required = {
        "task_id",
        "session_source",
        "run_id",
        "session_key",
        "run_status",
        "step",
        "kind",
        "state",
        "expected_steps",
        "learned_from",
        "learned_from_n",
        "baseline_runs",
    }
    for report in flagged:
        for deviation in report["deviations"]:
            assert required <= set(deviation), sorted(required - set(deviation))
            assert deviation["task_id"] is not None
            assert deviation["run_id"] and deviation["session_key"]
            assert deviation["step"] in deviation["expected_steps"]
            assert deviation["expected_steps"], "a deviation without an expectation is a score"
            assert deviation["learned_from"], "a deviation must say which runs it learned from"
            assert deviation["run_id"] not in deviation["learned_from"], (
                "leave-one-out: the run that skipped the step cannot vouch for it"
            )
            assert deviation["learned_from_n"] == len(deviation["learned_from"])
            assert deviation["state"] == "deviation"
            assert "score" not in deviation and "confidence" not in deviation


def test_replay_prints_detection_beside_false_alarm(fixtures, capsys):
    """Clause 5, second half. The line has to carry both denominators together, and the
    false-alarm denominator has to be the labelled benign corpus — not zero."""
    result = sc.validate_fixtures(
        trajectory_dir=fixtures["trajectory_dir"],
        db_path=fixtures["db"],
        labels=fixtures,
        now=CHECKED,
        grace_seconds=GRACE_SECONDS,
        min_runs=3,
    )
    line = sc.summary_line(result["counts"])
    print(line)
    printed = capsys.readouterr().out.strip()
    assert re.search(r"detection=3/3 \(100\.0%\)", printed), printed
    assert re.search(r"false_alarms=0/18 \(0\.0%\)", printed), printed
    assert "verdict=PASS" in printed
    assert result["counts"]["benign_scored"] == result["counts"]["benign"] == 18


def test_a_run_of_positives_only_cannot_report_a_pass(fixtures):
    """The hole the clause exists to close: a detector tuned only on positives. Strip the
    benign corpus, or the join to it, and the verdict must be FAIL even at 3/3 detection."""
    counts = {
        "detected": 3,
        "injected": 3,
        "false_alarms": 0,
        "benign": 18,
        "benign_scored": 18,
    }
    assert sc.summary_line(counts).endswith("verdict=PASS")
    for broken in (
        {**counts, "benign": 0, "benign_scored": 0},  # no negatives were labelled at all
        {**counts, "benign_scored": 0},  # labelled, but none of them joined a trace
        {**counts, "false_alarms": 1},  # one control flagged
        {**counts, "detected": 2},  # one fault missed
    ):
        assert sc.summary_line(broken).endswith("verdict=FAIL"), broken


def test_validation_scores_every_benign_control_it_was_given(fixtures):
    """`benign_scored` is the anti-holes field: labels that never joined a trace would make the
    false-alarm rate 0/0 and read as a perfect score."""
    result = sc.validate_fixtures(
        trajectory_dir=fixtures["trajectory_dir"],
        db_path=fixtures["db"],
        labels=fixtures,
        now=CHECKED,
        grace_seconds=GRACE_SECONDS,
        min_runs=3,
    )
    assert result["counts"] == {
        "detected": 3,
        "injected": 3,
        "false_alarms": 0,
        "benign": 18,
        "benign_scored": 18,
    }
    assert result["missed"] == []
    assert result["benign_flagged"] == []


def test_detecting_by_label_alone_is_not_enough(fixtures, monkeypatch):
    """Detection is counted only when the deviation names the step the fault removed. Make the
    detector flag *other* steps and the same corpus must score 0/3 — so the 3/3 figure cannot
    come from a stray flag on a run that happens to be in the labels."""
    real_score_run = sc.score_run

    def blind(trace, run, corpus, **kwargs):
        report = real_score_run(trace, run, corpus, **kwargs)
        if report is not None:
            for deviation in report.deviations:
                deviation.step = "tool:SomethingUnrelated"
        return report

    monkeypatch.setattr(sc, "score_run", blind)
    result = sc.validate_fixtures(
        trajectory_dir=fixtures["trajectory_dir"],
        db_path=fixtures["db"],
        labels=fixtures,
        now=CHECKED,
        grace_seconds=GRACE_SECONDS,
        min_runs=3,
    )
    assert result["counts"]["detected"] == 0
    assert result["counts"]["injected"] == 3
    assert {m["run_id"] for m in result["missed"]} == set(_fault_ids(fixtures))


# ---------------------------------------------------------------------------
# the nightly surface: the two commands a person would run
# ---------------------------------------------------------------------------


def test_the_validate_command_exits_zero_on_a_clean_sweep(fixtures, tmp_path):
    """Process boundary for the accept-or-abort check the round runs: `validate` against the
    committed fixture shape, exit 0 only when every fault is caught and no control is flagged."""
    proc = subprocess.run(
        [sys.executable, str(MODULE_PATH), "validate", "--fixtures", str(tmp_path / "fixtures")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "detection=3/3 (100.0%)" in proc.stdout
    assert "false_alarms=0/18 (0.0%)" in proc.stdout
    assert "verdict=PASS" in proc.stdout


def test_the_validate_command_refuses_a_directory_it_cannot_score(tmp_path):
    """A `--fixtures` directory with no labels cannot produce a false-alarm rate, so it must
    exit non-zero and say what is missing rather than print a detection-only pass."""
    empty = tmp_path / "not-fixtures"
    (empty / "trajectories").mkdir(parents=True)
    proc = subprocess.run(
        [sys.executable, str(MODULE_PATH), "validate", "--fixtures", str(empty)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode != 0
    assert "labels.json" in (proc.stdout + proc.stderr)


def test_replaying_the_same_corpus_twice_publishes_the_same_baseline(fixtures, replayed):
    """Determinism: the published step set and its digest come from the corpus, not from the
    wall clock, so two runs of the nightly job before and after a change are comparable."""
    again = sc.replay(
        trajectory_dir=fixtures["trajectory_dir"],
        db_path=fixtures["db"],
        days=None,
        now=datetime(2026, 9, 22, 3, 0, tzinfo=timezone.utc),
        grace_seconds=GRACE_SECONDS,
        min_runs=3,
    )
    assert again["published"] == replayed["published"]
    assert again["baseline_digest"] == replayed["baseline_digest"]
    assert again["flagged_rate"] == replayed["flagged_rate"]


def test_the_replay_flags_three_of_twenty_one_runs(fixtures, replayed):
    """The one number the nightly note carries, stated against its own denominator: 3 flagged
    runs out of 21 scored (3 faults + 18 controls), which is the shape the field study has to
    re-measure on real days before anything alerts."""
    assert replayed["runs_scored"] == 21
    assert replayed["runs_flagged"] == 3
    # Six deviations from three runs: one skipped stage surfaces as both the tool call that
    # never happened and the artifact step it would have emitted.
    assert replayed["deviations"] == 6
    assert replayed["flagged_rate"] == pytest.approx(3 / 21)
    assert replayed["tasks"] == sorted(sc.FIXTURE_TASKS)
# ── #3246 — a phantom capture cannot become an expectation ────────────────────

PHANTOM_TASK = "autonomy-task:76"
#: The queue-health check four of task 76's five runs make. `REDIRECT_RE` has no quote
#: awareness, so it reads the jq threshold's `>` as an output redirect and the `0` after it as
#: the file that was written: those four runs "wrote `0`".
PHANTOM_JQ = "jq -r '.sources[] | select(.pending_wait_over_bound_count > 0) | .name'"
#: The flagged run — `run_scheduled-task_20261021_172318_b8d329` in the live corpus — differs
#: from its peers only in spelling its jq filter without a threshold. It therefore never
#: "wrote `0`", and the replay escalated it for skipping a stage no run of its task ever ran.
THRESHOLD_FREE_JQ = "jq -r '.sources[] | select(.pending_wait_over_bound_count) | .name'"
#: The artifact every run of the task really does publish, so the task keeps a baseline.
REPORT_WRITE = "printf ok | tee /tmp/qhc_regrade.txt"


def _phantom_corpus(tmp_path, *, flagged_at=2, drop_report_at=None):
    """Five real trajectory rows plus a real worker ledger for one five-run task."""
    import json
    import sqlite3

    root = tmp_path / "phantom"
    root.mkdir(parents=True, exist_ok=True)
    db = tmp_path / "workers.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE runs (run_id TEXT, task_id INTEGER, status TEXT, source TEXT,"
                 " started_at TEXT, completed_at TEXT, meta_json TEXT)")
    for i in range(5):
        jq = THRESHOLD_FREE_JQ if i == flagged_at else PHANTOM_JQ
        calls = [{"tool": "Bash", "command": jq}]
        if i != drop_report_at:
            calls.append({"tool": "Bash", "command": REPORT_WRITE})
        (root / "trajectories-0.jsonl").open("a", encoding="utf-8").write(
            json.dumps(sc._trajectory_row(f"phantom-run-{i}", PHANTOM_TASK, calls,
                                         "2026-09-20T05:00:00")) + "\n")
        conn.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)", (
            f"run_task76_{i}", 76, "success", sc.DEFAULT_SOURCE, "2026-09-20T05:00:00+00:00",
            "2026-09-20T05:12:00+00:00", json.dumps({"session_id": f"phantom-run-{i}"})))
    conn.commit()
    conn.close()
    return root, db


def _phantom_replay(tmp_path, **kwargs):
    root, db = _phantom_corpus(tmp_path, **kwargs)
    return sc.replay(trajectory_dir=root, db_path=db, now=CHECKED,
                     grace_seconds=GRACE_SECONDS, min_runs=5)


def test_a_run_that_omits_only_the_phantom_is_neither_flagged_nor_escalated(tmp_path):
    """Clause 4: the task-76 corpus end to end, where the flag has to disappear at its source.

    `learn` needs `n_runs - 1` of `n_runs` at the default `support`, which is why four runs
    carrying `write:0` made `0` an expectation of all five — the same arithmetic that flagged 1
    run of the real task 76's seven. So the fix cannot live in the escalator: every *structural*
    missing step escalates, and the only sound move is that the phantom never enters a trace, and
    so never enters `expected_steps` at all.

    The zero is asserted three ways because each is satisfiable by a detector that has stopped
    working. The task is still SCORED with its real report step expected, so nothing here passed
    for want of a baseline. And the same corpus with the report write missing from one run still
    flags that run and files the draft — the door the nightly job walks is open and armed, and it
    is only the phantom that no longer walks through it.
    """
    res = _phantom_replay(tmp_path)
    assert res["runs_scored"] == 5, "five joined runs, or the zero below measures an empty corpus"
    expected = res["corpus"].baselines[PHANTOM_TASK].expected_steps
    assert "write:qhc_regrade.txt" in expected, (
        f"the step this task really publishes has to stay expected or the zero is vacuous; "
        f"expected_steps={expected}"
    )
    assert "write:0" not in expected, (
        f"a jq threshold is not an artifact; expected_steps={expected}"
    )
    assert res["deviations"] == 0 and res["runs_flagged"] == 0, (
        f"a run that wrote its report is not a skipped stage: "
        f"{[r['deviations'] for r in res['reports']]}"
    )
    backlog = tmp_path / "backlog"
    assert sc.file_escalations(res, backlog_dir=backlog) == [], (
        "nothing to escalate once the phantom is not a step"
    )
    assert not backlog.exists() or not list(backlog.glob("*.md")), (
        "a draft was filed for a clean run"
    )

    # The witness that the zero above is a verdict and not a blind detector: the same corpus,
    # with the artifact the task genuinely publishes missing from that same run, still flags and
    # still files — naming the report, not a number.
    broken = _phantom_replay(tmp_path / "broken", drop_report_at=2)
    assert broken["runs_flagged"] == 1, "the escalator has to stay armed on a real dropped stage"
    broken_backlog = tmp_path / "broken-backlog"
    filed = sc.file_escalations(broken, backlog_dir=broken_backlog)
    assert len(filed) == 1, "a run that really skipped its report still gets escalated"
    drafts = sorted(p.name for p in broken_backlog.glob("*.md"))
    assert len(drafts) == 1 and "qhc_regrade.txt" in (
        broken_backlog / drafts[0]).read_text(encoding="utf-8"), drafts
