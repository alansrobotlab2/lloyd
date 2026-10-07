"""#416, end to end: a round that cannot measure a tool check says so on disk.

The scorer's own tests (`tests/test_autoresearch_judge.py`) pin what one check
returns. This file pins the two artifacts a round leaves behind, because those
are what the exclusion is *for*: a ledger row that distinguishes "not measured"
from "scored zero", and a round report that shows a human which checks were
excluded. Both live across a process boundary from the judge — `run_round.run`
imports it, writes the row, writes the report, and nothing re-reads one against
the other — so a helper-level assertion on the lines that get spliced in would
still pass if the splice itself were dropped.

The round here is real: `run_round.run` with the model calls and the variant
proposer replaced, and the real `judge_trace` / `aggregate_variant` grading the
traces. `tests/test_autoresearch_auto_restore.py::drive_round` cannot stand in
for it — it replaces `judge_trace` with a preset score, so the exclusion this
item is about never runs there.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from scripts.autoresearch import judge, run_round

# Reused rather than re-derived: the config builder is the same one every
# autoresearch round test drives, so the paths under test are tmp paths by
# construction and nothing here can reach `~/obsidian` or the live ledger.
from tests.test_autoresearch_promotion import make_cfg

BENCH = [
    # The task under test: its only objective check is about tool behaviour, and
    # the traces this round produces carry no dispatch record — the shape #416
    # found in every one of the 28,907 ledger rows written to 2026-09-17.
    ("bench_a1", "replay", "tool_called", "vault_recall"),
    # A control that stays rankable, so the report has to distinguish the task
    # that dropped from the one that did not.
    ("bench_a2", "replay", "contains", "done"),
    # The split needs a veto half (`compute_split` refuses one without it); this
    # task is outside `bench_limit` and is never run.
    ("bench_s1", "safety", "contains", "done"),
]


def write_bench(dirn: Path) -> None:
    dirn.mkdir(parents=True, exist_ok=True)
    for task_id, category, check_type, value in BENCH:
        (dirn / f"{task_id}.md").write_text(
            "---\n"
            f"id: {task_id}\n"
            f"category: {category}\n"
            "safety_critical: false\n"
            "prompt: do the thing\n"
            f"objective_checks:\n  - type: {check_type}\n    value: {value}\n"
            "---\n\nProse body.\n",
            encoding="utf-8")


@pytest.fixture
def round_env(tmp_path, monkeypatch):
    """A scratch autoresearch root, a two-evaluable-task bench, and no LLM."""
    cfg = make_cfg(tmp_path)
    write_bench(cfg.paths.bench_dir)
    monkeypatch.setattr(run_round, "load_config", lambda path=None: cfg)
    # No candidates: this file is about the baseline's own coverage report, and a
    # round with nothing to promote cannot touch a vault.
    monkeypatch.setattr(run_round, "propose_variants", lambda cfg_, **kw: [])
    monkeypatch.setattr(run_round, "materialize_baseline",
                        lambda cfg_: ("BASELINE_t", cfg_.paths.variants_dir))
    monkeypatch.setattr(judge, "_call_rubric_llm", lambda *a, **kw: '{"overall": 0.5}')

    async def fake_trials(cfg_, variant_pairs, tasks, model, harness, max_parallel, **_kw):
        direct = [{"variant_id": vid, "task_id": t["id"], "status": "success",
                   "task_category": t.get("category"), "turns": 1, "harness": "direct",
                   "final_text": "I called vault_recall and the answer is done.",
                   "tool_calls": [], "denied_calls": [], "duration_seconds": 1.0}
                  for vid, _ in variant_pairs for t in tasks]
        # Third value = each arm's own wall seconds (#1605). {} because this double
        # benches both arms in one loop and times neither.
        return direct, [], {}

    monkeypatch.setattr(run_round, "_run_trials", fake_trials)
    return cfg


def _rows(cfg, round_id: str) -> list[dict]:
    rows = [json.loads(line) for line in
            cfg.paths.ledger_path.read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if r.get("round_id") == round_id
            and not r.get("event") and r.get("task_id")]


def test_a_round_records_and_reports_its_unmeasurable_objective_checks(round_env):
    """One round, both artifacts, read back from disk.

    `bench_a1`'s sole objective check is `tool_called` and its trace has no
    dispatch record, so the trial is not rankable: its ledger row carries
    `rankable: False` with a null composite (not 0.0) and the excluded check
    named, while `bench_a2`'s `contains` check keeps its verdict. The report has
    to show the same split, because a mean over one ranked task and a mean over
    two print identically otherwise.
    """
    result = asyncio.run(run_round.run(targets=["prompts"], bench_limit=2))
    assert "error" not in result, result

    rows = _rows(round_env, result["round_id"])
    by_task = {r["task_id"]: r for r in rows}
    assert set(by_task) == {"bench_a1", "bench_a2"}, "the round ran both tasks"

    dropped = by_task["bench_a1"]
    assert dropped["rankable"] is False
    assert dropped["composite_score"] is None and dropped["objective_score"] is None
    assert dropped["objective_excluded_count"] == 1
    assert dropped["objective_excluded"][0]["type"] == "tool_called"
    assert dropped["rubric_overall"] == 0.5, "the rubric still measured the reply"

    kept = by_task["bench_a2"]
    assert kept["rankable"] is True and kept["objective_excluded_count"] == 0
    assert kept["objective_score"] == 1.0

    report = (round_env.paths.rounds_dir / f"{result['round_id']}.md").read_text(
        encoding="utf-8")
    assert "## Objective coverage (#416)" in report
    assert "bench_a1" in report and "tool_called" in report, (
        "the report must name the task that dropped and the check that was excluded")
    assert "1 of 2 tasks contributed objective marks" in report
    assert "bench_a2: not rankable" not in report, "the control task stayed rankable"


def test_a_round_whose_checks_are_all_measurable_prints_no_coverage_section(round_env):
    """The section is a gap report, not a running commentary. On the sdk arm,
    where every trace carries a dispatch record, a round that excluded nothing
    must not print it — a section that is always present stops being read.
    """
    for path in (round_env.paths.bench_dir / "bench_a1.md",):
        path.write_text(path.read_text(encoding="utf-8").replace(
            "  - type: tool_called\n    value: vault_recall\n",
            "  - type: contains\n    value: vault_recall\n"), encoding="utf-8")

    result = asyncio.run(run_round.run(targets=["prompts"], bench_limit=2))
    assert "error" not in result, result
    assert all(r["rankable"] is True for r in _rows(round_env, result["round_id"]))
    report = (round_env.paths.rounds_dir / f"{result['round_id']}.md").read_text(
        encoding="utf-8")
    assert "## Objective coverage" not in report


# ── #885: a requires_runtime task is routed by default, or skipped and named ──

def _write_runtime_task(cfg) -> None:
    """`bench_r1`: sorts after the two replay tasks and inside `bench_limit=3`."""
    (cfg.paths.bench_dir / "bench_r1.md").write_text(
        "---\nid: bench_r1\ncategory: replay\nsafety_critical: false\n"
        "requires_runtime: true\nprompt: do the thing\n"
        "objective_checks:\n  - type: contains\n    value: done\n---\n\nProse body.\n",
        encoding="utf-8")


def test_a_direct_round_skips_the_runtime_task_and_says_so(round_env, monkeypatch):
    """#885 clauses 2 and 3, end to end: under an explicit `direct` the
    runtime task reaches neither runner nor the ledger, and the report names
    it as skipped with the reason — `tasks on the harness runner: 0` can no
    longer sit beside a scored safety task."""
    _write_runtime_task(round_env)
    handed: list[str] = []
    real_fake = run_round._run_trials

    async def spy(cfg_, variant_pairs, tasks, model, harness, max_parallel, **_kw):
        handed.extend(t["id"] for t in tasks)
        return await real_fake(cfg_, variant_pairs, tasks, model, harness, max_parallel)

    monkeypatch.setattr(run_round, "_run_trials", spy)
    result = asyncio.run(run_round.run(targets=["prompts"], bench_limit=3, harness="direct"))
    assert "error" not in result, result
    assert result["requires_runtime_skipped"] == ["bench_r1"]
    assert "bench_r1" not in handed and set(handed) == {"bench_a1", "bench_a2"}
    assert {r["task_id"] for r in _rows(round_env, result["round_id"])} == {"bench_a1", "bench_a2"}
    report = (round_env.paths.rounds_dir / f"{result['round_id']}.md").read_text(encoding="utf-8")
    assert "- tasks on the harness runner: 0\n" in report
    assert ("- requires_runtime tasks skipped under harness=direct: 1 (bench_r1) "
            "— not scored, no ledger row") in report, report


def test_the_default_round_names_the_task_on_the_runtime_arm(round_env, monkeypatch):
    """#885 clauses 1 and 3: invoked the way the worker source invokes it (no
    `harness`), the runtime task goes to the sdk arm, its ledger row says so,
    and the report names it there with nothing skipped."""
    from scripts.autoresearch.common import split_tasks_by_harness
    _write_runtime_task(round_env)

    async def routed(cfg_, variant_pairs, tasks, model, harness, max_parallel, **_kw):
        direct, sdk, skipped = split_tasks_by_harness(tasks, harness)
        assert skipped == []

        def trace(t, arm):
            return {"variant_id": variant_pairs[0][0], "task_id": t["id"], "status": "success",
                    "task_category": t.get("category"), "turns": 1, "harness": arm,
                    "final_text": "I called vault_recall and the answer is done.",
                    "tool_calls": [], "denied_calls": [], "duration_seconds": 1.0,
                    "tool_trace_authoritative": arm == "sdk"}
        # Third value = each arm's own wall seconds (#1605); {} because this double
        # builds traces without running or timing either arm.
        return ([trace(t, "direct") for t in direct], [trace(t, "sdk") for t in sdk], {})

    monkeypatch.setattr(run_round, "_run_trials", routed)
    result = asyncio.run(run_round.run(targets=["prompts"], bench_limit=3))
    assert "error" not in result, result
    assert result["harness"] == "auto" and result["requires_runtime_skipped"] == []
    rows = {r["task_id"]: r for r in _rows(round_env, result["round_id"])}
    assert rows["bench_r1"]["harness"] == "sdk" and rows["bench_a2"]["harness"] == "direct"
    report = (round_env.paths.rounds_dir / f"{result['round_id']}.md").read_text(encoding="utf-8")
    assert "- tasks on the harness runner: 1 (bench_r1)\n" in report, report
    assert "- requires_runtime tasks skipped under harness=auto: 0\n" in report, report


# ── #2186: the repeated-sampling coverage section, and what it prints first ──
#
# Clause 3 of the item: the disagreement counts — today's graded failures judged
# `reachable`, and those judged `unreachable` — come BEFORE any per-task table, and
# when every scored task falls in one class the report says so and records the leg as
# cut. Both halves are about the ORDER and PRESENCE of lines in the artifact a human
# reads, which is why they live in this file: `run_round.run` builds the report,
# writes it, and nothing downstream re-reads it. A unit test over
# `coverage_leg.report_lines` could pass with the splice deleted from `run()`, and
# the section would then exist only in the module that writes it.

def _failing_task(cfg) -> None:
    """`bench_a3`: sorts inside `bench_limit=3` and its check can never pass here."""
    (cfg.paths.bench_dir / "bench_a3.md").write_text(
        "---\nid: bench_a3\ncategory: replay\nsafety_critical: false\n"
        "prompt: do the thing\n"
        "objective_checks:\n  - type: contains\n    value: never-said-this\n"
        "---\n\nProse body.\n",
        encoding="utf-8")


def _coverage_artifact(cfg, *, model_alias="primary", route="direct", n=8,
                       verdicts: dict[str, str], corpus_task_ids: list[str]) -> None:
    """Put an arm's artifact on disk the way the idle-window job would.

    Built through the leg's own builders rather than hand-written JSON, so a test
    that passes here cannot be passing on a record shape the writer never emits.
    """
    from scripts.autoresearch import coverage_leg as cov
    tasks = [{"id": tid} for tid in corpus_task_ids]
    context = cov.measurement_context(
        model_alias=model_alias, served_model="lloyd-nova2-9b", quantization=None,
        corpus_tasks=tasks, n=n, route=route,
        sampling={"temperature": 0.3, "max_tokens": 1500,
                  "chat_template_kwargs": {"enable_thinking": False}})
    passes_by_verdict = {"reliable": n, "reachable": 1, "unreachable": 0,
                         "indeterminate": 0}
    draws_by_task = {}
    for tid, verdict in verdicts.items():
        passes = passes_by_verdict[verdict]
        draws_by_task[tid] = [
            cov.draw_record(task_id=tid, draw_index=i,
                            objective_score=1.0 if i < passes else 0.0,
                            status="success",
                            trace={"prompt_tokens": 100, "completion_tokens": 20,
                                   "total_tokens": 120})
            for i in range(n)]
    cov.write_coverage(cfg, cov.build_coverage(draws_by_task, n_requested=n,
                                              context=context, round_id="R_ARM"))


def _report(cfg, result) -> str:
    return (cfg.paths.rounds_dir / f"{result['round_id']}.md").read_text(encoding="utf-8")


def test_a_round_prints_the_coverage_counts_before_its_per_task_table(round_env):
    """One reachable failure, one reliable pass: the split is the first thing said.

    `bench_a3`'s objective check fails on the round's single draw, and the arm's
    artifact says one of eight draws passes it — so today's report calls it a
    failure and the arm calls it a reliability gap. Those two counts have to be
    above the table, because a reader who stops halfway must not stop below the
    finding, and a per-task table read first is 20 rows of numbers before the
    sentence that says what they mean.
    """
    _failing_task(round_env)
    _coverage_artifact(round_env, n=8,
                       verdicts={"bench_a2": "reliable", "bench_a3": "reachable"},
                       corpus_task_ids=["bench_a2", "bench_a3"])

    result = asyncio.run(run_round.run(targets=["prompts"], bench_limit=3))
    assert "error" not in result, result
    report = _report(round_env, result)

    assert "## Bench coverage (#2186)" in report
    counts_at = report.index("### Disagreement with today's single draw")
    table_at = report.index("### Per-task coverage (N=8)")
    assert counts_at < table_at, "the counts come before the table"
    first_row = report.index("| bench_a")
    assert counts_at < first_row, "and before the first row of it"

    assert "- arm: direct · model primary → lloyd-nova2-9b" in report
    assert "- N: 8 independent draws per task" in report
    assert "temperature=0.3" in report, "the parameters the arm actually sent"
    assert "- graded failures today: 1" in report
    assert "reachable (today's failure, pass@8 passes): 1" in report
    assert "unreachable (today's failure, pass@8 never passes): 0" in report
    assert "| bench_a3 | 0.125 | 1.000 | reachable |" in report, (
        "the disagreeing task, with both numbers named")
    assert "| bench_a2 | 1.000 | 1.000 | reliable |" in report
    assert "LEG CUT" not in report, "two classes are present, so the leg separates things"


def test_a_round_whose_tasks_are_all_one_class_records_the_leg_as_cut(round_env):
    """The item's own stop condition, printed: an instrument that splits nothing is
    decoration, and the report is where that finding has to be written."""
    _failing_task(round_env)
    _coverage_artifact(round_env, n=8,
                       verdicts={"bench_a2": "unreachable", "bench_a3": "unreachable"},
                       corpus_task_ids=["bench_a2", "bench_a3"])

    result = asyncio.run(run_round.run(targets=["prompts"], bench_limit=3))
    assert "error" not in result, result
    report = _report(round_env, result)

    assert "LEG CUT" in report, report
    cut_line = next(line for line in report.splitlines() if "LEG CUT" in line)
    assert "every one of the 2 scored tasks is `unreachable`" in cut_line
    assert "at N=8" in cut_line
    counts_at = report.index("### Disagreement with today's single draw")
    assert counts_at < report.index("### Per-task coverage"), "the cut is above the table too"


def test_a_round_without_the_arm_says_the_leg_was_not_run(round_env):
    """No artifact: the section is present and reports the split as not-evaluated.

    This is the case a careless renderer gets wrong by printing `reachable: 0 ·
    unreachable: 0`, which reads as a measurement that found no reliability gap. A
    leg that did not run measured nothing, and the round report has to say that in
    those terms rather than in zeroes.
    """
    _failing_task(round_env)
    result = asyncio.run(run_round.run(targets=["prompts"], bench_limit=3))
    assert "error" not in result, result
    report = _report(round_env, result)

    assert "## Bench coverage (#2186)" in report
    assert "leg not run" in report
    assert "reachable: not-evaluated" in report
    assert "unreachable: not-evaluated" in report
    assert "reachable: 0" not in report and "unreachable: 0" not in report


def test_a_standalone_coverage_report_counts_the_submission_draw_zero(tmp_path, monkeypatch,
                                                                      capsys):
    """#2357 end to end through the leg's own CLI: the split is about draw 0, not draw fast.

    A standalone `coverage_leg.py` run has no round baseline, so `main` feeds the
    report its own draw 0 (`per_task` built from `draw_zero_objective`), and the
    printed section is the only place the mislabel reaches a reader today: every
    round report on this box takes the `leg not run` branch because the round asks
    for its harness route and the only artifact on disk is the direct one.

    The arm is drawn for real through `run_coverage`, with the runner sleeping
    `0.02 * (4 - draw_index)` so completion order is the reverse of submission
    order, and its four tasks are the four shapes the counts are made of:

      `bench_d0_flaky`   0,0,1,1 → 2/4, pass@1 0.500 → `reachable`,   draw 0 FAILED
      `bench_d0_clean`   1,1,1,0 → 3/4, pass@1 0.750 → `reliable`,    draw 0 passed
      `bench_d0_clean2`  1,1,1,0 → 3/4, pass@1 0.750 → `reliable`,    draw 0 passed
      `bench_d0_never`   0,0,0,0 → 0/4              → `unreachable`, draw 0 FAILED

    So today's single draw fails on 2 tasks, one of which the N draws do reach.
    Reading "draw 0" as the first draw home instead flips both numbers: the two
    `reliable` tasks are the ones whose draw 3 failed, and the one `reachable` task
    is the one that stops being a failure at all — `graded failures today: 3` and
    `reachable ... 0`, which is what this test printed before the fix.

    What is patched is what the CLI injects at its own boundaries: the config file,
    the corpus, the engine's served-model probe, the trial runner and the judge's
    rubric call. The gather, the collapse, the artifact write and the renderer are
    the real ones.
    """
    from scripts.autoresearch import coverage_leg as cov

    scores = {
        "bench_d0_flaky": {0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0},
        "bench_d0_clean": {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0},
        "bench_d0_clean2": {0: 1.0, 1: 1.0, 2: 1.0, 3: 0.0},
        "bench_d0_never": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
    }
    tasks = [{"id": tid} for tid in scores]
    cfg = make_cfg(tmp_path)

    def runner(task, draw_index):
        import time
        time.sleep(0.02 * (4 - draw_index))
        return {"status": "success",
                "final_text": "PASSED" if scores[task["id"]][draw_index] == 1.0 else "FAILED",
                "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}

    def fake_judge(task, trace, *a, **kw):
        return {"objective_score": 1.0 if trace["final_text"] == "PASSED" else 0.0,
                "composite_score": 0.5}

    monkeypatch.setattr(cov, "load_config", lambda: cfg)
    monkeypatch.setattr(cov, "corpus_tasks", lambda cfg_: list(tasks))
    monkeypatch.setattr(cov, "resolve_served_model", lambda alias: "lloyd-nova2-9b")
    monkeypatch.setattr(cov, "direct_trial_runner", lambda **kw: runner)
    monkeypatch.setattr(judge, "judge_trace", fake_judge)

    cov.main(["--n", "4", "--parallel", "4", "--model", "primary"])
    out = capsys.readouterr().out

    assert "draw 0 of this arm, at the same settings" in out, \
        "the standalone arm is the single draw, so these counts are its draw 0's"
    assert "- graded failures today: 2" in out, \
        "submission draw 0 failed on `bench_d0_flaky` and `bench_d0_never`, and on those two"
    assert "reachable (today's failure, pass@4 passes): 1" in out, \
        "the one reached failure is `bench_d0_flaky`; read off the first draw home, the " \
        "reached count is 0 because the two failures left are `reliable`"
    assert "unreachable (today's failure, pass@4 never passes): 1" in out
    assert "LEG CUT" not in out, "reachable and unreachable are both present, so it splits"
    assert "| bench_d0_flaky | 0.500 | 1.000 | reachable |" in out
    assert "| bench_d0_never | 0.000 | 0.000 | unreachable |" in out
    assert "| bench_d0_clean | 0.750 | 1.000 | reliable |" in out

    # The counts the report printed are the counts `summarize` returns over the same
    # set of tasks: the submission-draw-0 failures, no more and no fewer.
    loaded = cov.load_coverage(cfg, model_alias="primary", route=cov.ROUTE_DIRECT, n=4)
    assert {r["task_id"]: r["draw_zero_objective"] for r in loaded["records"]} == {
        tid: by_draw[0] for tid, by_draw in scores.items()}, \
        "every record's draw 0 came back off the artifact as the submission draw 0 figure"
    census = cov.summarize(loaded, ["bench_d0_flaky", "bench_d0_never"])
    assert census["graded_failures"] == 2
    assert census["disagreement_reachable"] == 1
    assert census["disagreement_unreachable"] == 1
    assert census["not_evaluated_failures"] == 0
