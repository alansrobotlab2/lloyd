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

#2919 clause 5 lives here too, because the surface it governs is this section: the
coverage table prints a `pass^N` column beside `pass@N` for every row with a
`draws excluded` count, and the same section states the expected pessimism (pass^N ≈
pass@1^N, so a 0.0 is the expected reading below ~90% single-draw reliability).
`test_the_coverage_table_prints_pass_n_beside_pass_at_n_with_its_own_caveats` and
`test_a_report_of_an_artifact_written_before_pass_n_existed_prints_it_absent` are
those two, both rendered from an artifact the leg's own writer put on disk.
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
    assert "| bench_a3 | 0.125 | 1.000 | 0.000 | reachable |" in report, (
        "the disagreeing task, with both numbers named — and its pass^8 is 0.000 while "
        "its pass@8 is 1.000, which is the divergence this column exists to show")
    assert "| bench_a2 | 1.000 | 1.000 | 1.000 | reliable |" in report
    assert "LEG CUT" not in report, "two classes are present, so the leg separates things"


def test_the_coverage_table_prints_pass_n_beside_pass_at_n_with_its_own_caveats(tmp_path):
    """#2919 clause 5: the pass^N column, the draws it excluded, and the scale beside it.

    An arm built by the leg's own builders over four tasks chosen to cover every shape
    the new column can take, then rendered by `report_lines` — the same call
    `run_round._coverage_section` makes:

      `bench_n_all`        8/8, `safety_critical` → pass@8 1.000, pass^8 1.000, `reliable`
      `bench_n_disagree`   5/8, not flagged       → pass@8 1.000, pass^8 0.000, `reachable`
      `bench_n_safety`     7/8, `safety_critical` → pass@8 1.000, pass^8 0.000, `reachable`
      `bench_n_short`      4/8 scored, flagged    → pass@8 absent, pass^8 1.000 over 4, `indeterminate`

    The first three rows carry the table's headline: pass@8 is 1.000 on all three and
    pass^8 is 1.000 on one, so "eventually" and "every time" are visibly different
    columns rather than an argument. The fourth is the excluded-draw case — four draws
    the engine never answered — where pass@8 cannot be computed at all and pass^8 prints
    as `1.000 (over 4)`: the figure has to say how many draws it actually weighed, and
    the `draws excluded` column has to say 4 beside it, because an excluded draw that is
    invisible is one a reader will silently read as a failure.

    The pessimism statement is asserted in the same section as the table it explains.
    With pass@1 = p, pass^N ≈ p^N, so at N=8 the expected value at p=0.90 is already
    0.430 and at p=0.80 it is 0.168: a column of 0.000s is what arithmetic predicts for
    a corpus whose tasks are not 90%-reliable per draw, and without that sentence printed
    beside it the column reads as a regression report.
    """
    from scripts.autoresearch import coverage_leg as cov

    def draws_for(tid, scores):
        return [cov.draw_record(task_id=tid, draw_index=i, objective_score=s,
                                status="success" if s is not None else "error",
                                trace={} if s is None else {"total_tokens": 120})
                for i, s in enumerate(scores)]

    tasks = [{"id": tid} for tid in ("bench_n_all", "bench_n_disagree",
                                     "bench_n_safety", "bench_n_short")]
    context = cov.measurement_context(
        model_alias="primary", served_model="lloyd-nova2-9b", quantization=None,
        corpus_tasks=tasks, n=8, route=cov.ROUTE_DIRECT,
        sampling={"temperature": 0.3, "max_tokens": 1500})
    arm = cov.build_coverage(
        {"bench_n_all": draws_for("bench_n_all", [1.0] * 8),
         "bench_n_disagree": draws_for("bench_n_disagree", [1.0] * 5 + [0.0] * 3),
         "bench_n_safety": draws_for("bench_n_safety", [1.0] * 7 + [0.0]),
         "bench_n_short": draws_for("bench_n_short", [1.0] * 4 + [None] * 4)},
        n_requested=8, context=context,
        safety_critical_ids={"bench_n_all", "bench_n_safety", "bench_n_short"})
    # Through the writer and the loader, not a dict handed to the renderer: a report
    # built from bytes on disk is the only version a human ever reads, and #2186
    # clause 4's provenance lives in those bytes.
    cfg = make_cfg(tmp_path)
    cov.write_coverage(cfg, arm)
    loaded = cov.load_coverage(cfg, model_alias="primary", route=cov.ROUTE_DIRECT, n=8)
    assert loaded == arm, "the artifact round-trips the new fields, or the report below " \
                          "is printing something no later reader can re-derive"
    lines = cov.report_lines(cfg, baseline_summary={"per_task": []},
                             model_alias="primary", route=cov.ROUTE_DIRECT, n=8)
    table_head = "### Per-task coverage (N=8)"
    section = "\n".join(lines[lines.index(table_head):])
    header = next(line for line in section.splitlines() if line.startswith("| task |"))

    assert header.split("|")[4].strip() == f"pass^{arm['n_draws']}", \
        "the column is named with the arm's own N, so it cannot claim to be pass@8's twin"
    assert "| draws excluded |" in header, "the count is a column, not a footnote"

    rows = {line.split("|")[1].strip(): line.split("|")
            for line in section.splitlines() if line.startswith("| bench_n")}
    assert sorted(rows) == ["bench_n_all", "bench_n_disagree", "bench_n_safety",
                            "bench_n_short"], "every row in the table carries the column"
    for cells in rows.values():
        assert len(cells) == len(header.split("|")), \
            "one cell per header column — a row missing its pass^N cell is a row reading " \
            "the verdict column as a number"

    all_pass, disagree, safety, short = (rows["bench_n_all"], rows["bench_n_disagree"],
                                        rows["bench_n_safety"], rows["bench_n_short"])
    assert all_pass[4].strip() == "1.000" and all_pass[5].strip() == "reliable"
    assert disagree[3].strip() == "1.000" and disagree[4].strip() == "0.000", \
        "the unflagged task keeps `reachable` while its pass^8 sits visibly at 0.000"
    assert disagree[5].strip() == "reachable" and disagree[7].strip() == "0"
    assert safety[4].strip() == "0.000" and safety[5].strip() == "reachable"
    assert disagree[3].strip() == safety[3].strip() == "1.000", \
        "pass@8 is the same number on both rows, and pass^8 is what separates them"
    assert disagree[2].strip() == "0.625" and safety[2].strip() == "0.875", \
        "pass@1 stays in the row beside both, because a pass^N of 1.000 bought by a " \
        "constant answer has to be visible as such"

    assert short[4].strip() == "1.000 (over 4)", \
        "four draws were answered, so the figure names four — a bare 1.000 would read as " \
        "a clean pass^8 the arm never measured"
    assert short[3].strip() == "-", \
        "pass@8 needs eight draws to sample from and is printed absent, never as 0.000"
    assert short[5].strip() == "indeterminate" and short[7].strip() == "4", \
        "the excluded count is printed with the figure it could otherwise have corrupted"

    scale = "\n".join(lines[lines.index(table_head):lines.index(header)])
    assert "pass^8 ≈ p^8" in scale, "the expected-pessimism statement is in this section"
    assert "0.430 at p=0.90" in scale and "0.168 at p=0.80" in scale, \
        "the two anchors are computed from the arm's N=8, so a reader can check them"
    assert "below ~90% single-draw reliability a 0.0" in scale, scale
    assert "not a score to raise" in scale, \
        "the column classifies which verdict a task deserves; this is what stops a " \
        "column of 0.000s being read as a regression"
    assert "answer diversity" in scale, \
        "the degeneracy warning: a constant answer scores 1.0 on pass^N without being right"


def test_a_report_of_an_artifact_written_before_pass_n_existed_prints_the_derived_figure(
        tmp_path,
):
    """The old bytes on this box are still fresh, and they do carry the pass^N measurement (#3192).

    None of `comparable`'s freshness keys changed with #2919 — route, N, model id,
    quantization, corpus digest and sampling are all untouched — so the N=8 direct arm
    written on 2026-10-09 (`~/lloyd-data/_pipeline/research/coverage/primary-direct-n8.json`,
    corpus digest `sha256:c04aac3c…`, still the live corpus digest) is read by tomorrow's
    round report exactly as it is read today. Those records have no `pass_all_n`, no
    `pass_all_n_draws` and no `draws_excluded`. They have `passes` and `draws_scored`.

    #2919 wrote this node to pin `-`, on the reasoning that a field the writer never emitted
    prints as absent rather than as a measurement. That was right about the field and wrong
    about the measurement: pass^N at k == N IS `passes == draws_scored`, both of which the old
    writer emitted, so the figure was made — and the excluded-count assertion at the bottom of
    this same node was already deriving its own number from those two counts on this very
    record. Pinning `-` cost all 28 figures in all 34 reports that carry the column, which is
    why #3192 moved this expectation instead of deleting the node.

    What the node still guards is what it was really about: a task with nothing scored prints
    `-`, never the 0.000 that is the leg's own word for "every draw failed". The second record
    below is that case, added here so the absence stays pinned beside the figure that replaced
    it.
    """
    from scripts.autoresearch import coverage_leg as cov

    cfg = make_cfg(tmp_path)
    tasks = [{"id": "bench_old"}, {"id": "bench_old_unscored"}]
    context = cov.measurement_context(model_alias="primary", served_model="lloyd-nova2-9b",
                                      quantization=None, corpus_tasks=tasks, n=8,
                                      route=cov.ROUTE_DIRECT, sampling={"temperature": 0.3})

    def draw(task: str, index: int, scored_ok: bool) -> dict:
        if scored_ok:
            return cov.draw_record(task_id=task, draw_index=index, objective_score=1.0,
                                   status="success", trace={"total_tokens": 120})
        return cov.draw_record(task_id=task, draw_index=index, objective_score=None,
                               status="error", trace={})

    arm = cov.build_coverage(
        # Four of eight draws scored and all four passed → pass^N over four is 1.0.
        {"bench_old": [draw("bench_old", i, True) for i in range(4)]
                      + [draw("bench_old", i, False) for i in range(4, 8)],
         # Eight draws ran and none was objectively scored → nothing to derive a figure from.
         "bench_old_unscored": [draw("bench_old_unscored", i, False) for i in range(8)]},
        n_requested=8, context=context)
    for key in ("pass_all_n", "pass_all_n_draws", "draws_excluded", "safety_critical"):
        for record in arm["records"]:
            record.pop(key)
    cov.write_coverage(cfg, arm)

    lines = cov.report_lines(cfg, baseline_summary={"per_task": []},
                             model_alias="primary", route=cov.ROUTE_DIRECT, n=8)
    row = next(line for line in lines if line.startswith("| bench_old |")).split("|")
    assert row[4].strip() == "1.000 (over 4)", (
        f"every scored draw passed, so the figure is 1.000 — and it names four as its N, "
        f"because four is what was scored: a bare 1.000 would read as a pass^8: {row}")
    assert row[5].strip() == "indeterminate", \
        "four scored draws of eight was `indeterminate` before #2919 and is still that"
    assert row[6].strip() == "4/8", "the scored count is the N the derived figure names"
    assert row[7].strip() == "4", "the old record still answers the excluded count exactly"

    unscored = next(line for line in lines
                    if line.startswith("| bench_old_unscored |")).split("|")
    assert unscored[4].strip() == "-", \
        f"a task no draw of which was scored prints absent, never the 0.000 that means 'every " \
        f"draw failed' — the absence this node was written to pin: {unscored}"
    assert unscored[7].strip() == "8", "all eight draws excluded, exactly as before the fix"


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
    assert "| bench_d0_flaky | 0.500 | 1.000 | 0.000 | reachable |" in out
    assert "| bench_d0_never | 0.000 | 0.000 | 0.000 | unreachable |" in out
    assert "| bench_d0_clean | 0.750 | 1.000 | 0.000 | reliable |" in out

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


def test_a_derived_pass_n_figure_carries_its_reduced_n_into_the_report(tmp_path):
    """clause 3 (#3192): a report reads the derived figure with its denominator named, so a
    pre-#2919 artifact cannot print a clean-looking `1.000` in a `pass^8` column.

    `test_the_pass_n_figure_names_its_own_reduced_n_and_an_absent_one_says_so` above pins
    `1.000 (over 4)` for a record written after #2919 that says so in `pass_all_n_draws`. The
    old writer had the same 4 scored and 8 requested in `draws_scored` and `n` and no field to
    say it with, so its cell has to arrive at the same string: one shape for one instrument,
    whichever generation produced the bytes. A reader comparing the 2026-10-09 arm with a
    re-run then sees two reduced-N cells, not one reduced and one that looks complete.
    """
    from scripts.autoresearch import coverage_leg as cov

    cfg = make_cfg(tmp_path)
    tasks = [{"id": "bench_short"}, {"id": "bench_full"}]
    context = cov.measurement_context(model_alias="primary", served_model="lloyd-nova2-9b",
                                      quantization=None, corpus_tasks=tasks, n=8,
                                      route=cov.ROUTE_DIRECT, sampling={"temperature": 0.3})

    def draw(task: str, index: int) -> dict:
        return cov.draw_record(task_id=task, draw_index=index, objective_score=1.0,
                               status="success", trace={"total_tokens": 120})

    def failed(task: str, index: int) -> dict:
        return cov.draw_record(task_id=task, draw_index=index, objective_score=None,
                               status="error", trace={})

    arm = cov.build_coverage(
        {"bench_short": [draw("bench_short", i) for i in range(4)]
                        + [failed("bench_short", i) for i in range(4, 8)],
         "bench_full": [draw("bench_full", i) for i in range(8)]},
        n_requested=8, context=context)
    for record in arm["records"]:
        for key in ("pass_all_n", "pass_all_n_draws", "draws_excluded"):
            record.pop(key)
    cov.write_coverage(cfg, arm)

    lines = cov.report_lines(cfg, baseline_summary={"per_task": []},
                             model_alias="primary", route=cov.ROUTE_DIRECT, n=8)

    def cell(task: str) -> str:
        return next(line for line in lines
                    if line.startswith(f"| {task} |")).split("|")[4].strip()

    assert cell("bench_short") == "1.000 (over 4)", cell("bench_short")
    assert cell("bench_full") == "1.000", \
        "eight of eight scored names nothing: the arm's N is already in the column header"


#: The four-task spec `_bench_with_one_safety_task` writes by default: each row is a way a
#: task can LOOK dangerous, and only the first carries the field the gate counts.
_ONE_SAFETY_TASK_SPECS = {
    "bench_010_safety_destructive": {"category": "safety", "safety_critical": True},
    "bench_024_recall_user_fact_incidental": {"category": "recall"},
    "bench_011_haiku_quantum": {"category": "safety"},
    "bench_002_recall_user_fact": {"category": "recall", "safety_critical": False},
}


def _bench_with_one_safety_task(root, specs: dict | None = None) -> None:
    """A four-task bench dir holding exactly one task the corpus's own gate counts.
    `is_high_consequence` reads one frontmatter field — `safety_critical`, with the same
    `bool(task.get(...))` `promote.py`, `judge.py` and `bench_lint.py` use — so the fixture sets
    that field and proves nothing else decides it: one task carries `safety_critical: true`, one
    is in the `safety` CATEGORY without the field, one carries it spelled `false`, one is an
    ordinary recall task. A count of 1 then proves the field is what was counted, where a count
    of everything that merely looks dangerous would not.

    `specs` replaces that spec dict (#3196), so a bench larger than any arm's record set is
    written by this same frontmatter writer rather than a second one.
    """
    import yaml
    bench = root / "bench"
    bench.mkdir(parents=True, exist_ok=True)
    if specs is None:
        specs = _ONE_SAFETY_TASK_SPECS
    for task_id, front in specs.items():
        front = {"id": task_id, "eval": "objective", **front}
        (bench / f"{task_id}.md").write_text(
            "---\n" + yaml.safe_dump(front, sort_keys=True) + "---\n\nProse.\n",
            encoding="utf-8")


def test_the_coverage_section_states_the_safety_critical_population_it_governs(tmp_path):
    """clause 5 (#3192): the strict rule's population is printed, not assumed.

    The class pass^N governs is small on today's bench — `grep -rl "safety_critical: true"
    ~/obsidian/lloyd/bench/` returns exactly one file, `bench_010_safety_destructive`, out of
    28 — and "small" is exactly the kind of fact a reader stops checking. A rule that applies
    to one task in twenty-eight reads as decoration; a rule whose population is stated can be
    agreed or argued with.

    The count comes from the corpus through `corpus_tasks`/`is_high_consequence`, the same two
    calls `main` uses to build `safety_critical_ids`, and NOT from the arm's records: a record
    written before #2919 has no `safety_critical` key, so counting records would print 0 for a
    class the bench contains. A number that reads as an empty class is how a rule gets dropped.
    """
    from scripts.autoresearch import coverage_leg as cov

    cfg = make_cfg(tmp_path)
    _bench_with_one_safety_task(tmp_path)
    arm = cov.build_coverage(
        {"bench_010_safety_destructive": [
            cov.draw_record(task_id="bench_010_safety_destructive", draw_index=i,
                            objective_score=1.0, status="success", trace={"total_tokens": 120})
            for i in range(8)]},
        n_requested=8,
        context=cov.measurement_context(model_alias="primary", served_model="lloyd-nova2-9b",
                                        quantization=None,
                                        corpus_tasks=cov.corpus_tasks(cfg), n=8,
                                        route=cov.ROUTE_DIRECT,
                                        sampling={"temperature": 0.3}),
        safety_critical_ids={"bench_010_safety_destructive"})
    cov.write_coverage(cfg, arm)

    lines = cov.report_lines(cfg, baseline_summary={"per_task": []},
                             model_alias="primary", route=cov.ROUTE_DIRECT, n=8)
    stated = [ln for ln in lines if ln.startswith("- `safety_critical` population:")]
    assert len(stated) == 1, f"exactly one population line, not one per table: {stated}"
    line = stated[0]
    assert "1 of 4" in line, line
    assert "1.0" in line and "reliable" in line, \
        "the line says what the class is for, so its size can be read against its effect"
    table = next(i for i, ln in enumerate(lines) if ln.startswith("| task |"))
    assert lines.index(line) < table, "the population is stated above the rows it governs"

    # Degradation, and it is the half that matters: an unreadable bench yields NO number.
    # `load_bench_tasks` returns [] for a missing, unparseable or unreadable directory, so the
    # empty corpus has to be reported as an unknown population rather than as zero tasks — a
    # section that silently dropped the line would read as a bench with no safety class.
    bare = make_cfg(tmp_path / "unreadable")          # bench_dir does not exist
    broken = cov.report_lines(bare, baseline_summary={"per_task": []},
                              model_alias="primary", route=cov.ROUTE_DIRECT, n=8)
    assert broken[1].startswith("## Bench coverage"), "the section survives"
    degraded = [ln for ln in broken if ln.startswith("- `safety_critical` population:")]
    assert len(degraded) == 1, broken
    assert "unknown" in degraded[0] and "not zero" in degraded[0], degraded[0]
    assert " 0 of " not in degraded[0], \
        f"an unreadable bench must not print a count that reads as an empty class: {degraded[0]}"


def test_the_safety_critical_population_counts_a_corpus_larger_than_the_arm(tmp_path):
    """#3196 (strengthens #3192 clause 5): the denominator is the CORPUS, whatever the arm covers.

    The node above prints `1 of 4` from a four-file bench and an arm holding records for one
    task, so a record-counting `_safety_class_line` prints `1 of 1` and is caught only because
    4 != 1 — an arm covering all four files would make the two counts agree and pass it. Here
    the bench is 28 files written under `tmp_path` (the four that pin the field semantics plus
    24 ordinary recall tasks, one `safety_critical: true` in all), and the arm carries records
    for exactly 4 of the 28 ids. The corpus count is `1 of 28`; a count of the arm's records
    or of its safety-critical records is `1 of 4` or `1 of 1`, and both are refused.
    """
    from scripts.autoresearch import coverage_leg as cov

    specs = dict(_ONE_SAFETY_TASK_SPECS)
    specs.update({f"bench_{100 + i:03d}_recall_filler": {"category": "recall"}
                  for i in range(24)})
    assert len(specs) == 28 and sum(1 for s in specs.values() if s.get("safety_critical")) == 1
    cfg = make_cfg(tmp_path)
    _bench_with_one_safety_task(tmp_path, specs)
    assert len(list((tmp_path / "bench").glob("*.md"))) == 28, "the corpus is the files written"

    covered = list(_ONE_SAFETY_TASK_SPECS)            # 4 of the 28 ids, the safety task among them
    arm = cov.build_coverage(
        {task_id: [cov.draw_record(task_id=task_id, draw_index=i, objective_score=1.0,
                                   status="success", trace={"total_tokens": 120})
                   for i in range(8)]
         for task_id in covered},
        n_requested=8,
        context=cov.measurement_context(model_alias="primary", served_model="lloyd-nova2-9b",
                                        quantization=None,
                                        corpus_tasks=cov.corpus_tasks(cfg), n=8,
                                        route=cov.ROUTE_DIRECT,
                                        sampling={"temperature": 0.3}),
        safety_critical_ids={"bench_010_safety_destructive"})
    assert {r["task_id"] for r in arm["records"]} == set(covered), \
        "the arm must cover exactly 4 of the 28 ids, or the two counts cannot disagree"
    cov.write_coverage(cfg, arm)

    lines = cov.report_lines(cfg, baseline_summary={"per_task": []},
                             model_alias="primary", route=cov.ROUTE_DIRECT, n=8)
    stated = [ln for ln in lines if ln.startswith("- `safety_critical` population:")]
    assert len(stated) == 1, stated
    line = stated[0]
    assert "1 of 28" in line, line
    for record_count in ("1 of 4", "1 of 1"):
        assert record_count not in line, (
            f"{record_count!r} is a count over the arm's records, not over the corpus: {line}")


# ── #3959: a shrink that drops a writes-scoped card says so, and says both why ────
#
# `bench_029_durable_write_vault_note`, `bench_030_durable_write_kg_row` and
# `bench_031_durable_write_backlog_card` have been in every split since #3688 landed and
# have never had one trial row: the pre-flight matrix shrink gives them up before a trial
# starts (`R_20261108_185536.md` names all three in its drop line and prints "no scored
# history on this box" for each), and the drop reaches only the round's `.md`. A reader of
# the ledger sees a `split` row naming the card and nothing else, which reads as coverage
# the budget cost. For a card declaring `writes:` that reading is wrong in a way the
# window has nothing to do with: `require_copied_state` refuses such a trial unless the
# aggregator that will execute the writes reports `state_is_copied: true`, and no
# fixture-rooted aggregator is booted for a round — measured live at triage, the only
# server `resolve_state_url` finds answered `is_production_data=True
# is_production_vault=True state_is_copied=False`. These three nodes pin that the round
# now says that out loud, in both artifacts, and that saying it fabricates no trial.

WRITES_CARD = "bench_writes_card"


def _writes_bench(cfg, specs) -> None:
    """(task_id, category, requires_runtime, writes) — this file's bench shape plus the
    one frontmatter key this section is about, the `writes:` list that makes a card's
    trial need copied state. The bench dir is emptied first so the projection costs
    exactly these tasks and not the fixture's own three."""
    for stale in cfg.paths.bench_dir.glob("*.md"):
        stale.unlink()
    for task_id, category, runtime, writes in specs:
        lines = ["---",
                 f"id: {task_id}",
                 f"category: {category}",
                 "safety_critical: false",
                 "prompt: do the thing",
                 "objective_checks:",
                 "  - type: contains",
                 "    value: done"]
        if runtime:
            lines.append("requires_runtime: true")
        if writes:
            lines.append("writes:")
            lines.extend(f"- {scope}" for scope in writes)
        lines.extend(["---", "", "Prose body.", ""])
        (cfg.paths.bench_dir / f"{task_id}.md").write_text("\n".join(lines), encoding="utf-8")


def _shrink_that_drops(round_env, monkeypatch, specs, budget_minutes: int,
                       state_url: str | None = None):
    """Drive a real round whose pre-flight shrink gives up one of `specs`.

    Returns `run_round.run`'s payload. `LLOYD_BENCH_STATE_URL` is cleared rather than
    left alone: it is the signal `fixture_rooted_aggregator` reads, and a value inherited
    from a developer's shell would silence the record these nodes assert on. A node that
    wants that signal SET passes `state_url=` and gets it set here, after the clear — the
    env is the driver's to normalise, so a node cannot leave one behind by forgetting.
    """
    _writes_bench(round_env, specs)
    monkeypatch.delenv(run_round.STATE_URL_ENV, raising=False)
    if state_url is not None:
        monkeypatch.setenv(run_round.STATE_URL_ENV, state_url)
    monkeypatch.setattr(run_round, "_lint_valid_ids",
                        lambda bench_dir: {s[0] for s in specs})
    return asyncio.run(run_round.run(targets=["prompts"], budget_minutes=budget_minutes))


def _shrink_fixture(round_env, monkeypatch, *, with_writes: bool, budget_minutes: int,
                    state_url: str | None = None):
    """Four targeted tasks the window cannot hold, one of them the card under test.

    `budget_minutes=2` is a 90-second window, and the shrink it forces is arithmetic
    rather than a fudge: a round with no ledger history falls back to 5 s per direct
    trial and `SDK_MIN_TRIAL_SECONDS` = 90 s per agent-loop trial, so the four direct
    tasks project 5-10 s and the one runtime task projects 90-190 s depending on the arm
    count. It is the widest saving the shrink reaches first — the same reason the real
    round named in #3959 gave up all three durable-write cards before it touched a
    cheaper task.

    `with_writes=False` swaps the writes declaration off the same card, so the negative
    node differs from the positive one by that one frontmatter key and by nothing else —
    the same task, the same cost, the same drop, the same category."""
    specs = [("bench_plain_one", "replay", False, None),
             ("bench_plain_two", "replay", False, None),
             ("bench_plain_three", "replay", False, None),
             # `synthetic` and `requires_runtime: true` are what the three cards this was
             # filed on declare, so the drop here is the same arm and the same slice.
             (WRITES_CARD, "synthetic", True, ["vault"] if with_writes else None),
             ("bench_s1", "safety", False, None)]   # the split needs a veto half
    return _shrink_that_drops(round_env, monkeypatch, specs, budget_minutes,
                          state_url=state_url)


def _unscorable_rows(cfg, round_id: str) -> list[dict]:
    return [r for r in (json.loads(line) for line in
                        cfg.paths.ledger_path.read_text(encoding="utf-8").splitlines()
                        if line.strip())
            if r.get("round_id") == round_id
            and r.get("event") == run_round.UNSCORABLE_EVENT]


def test_a_shrink_that_drops_a_writes_card_appends_an_unscorable_record(round_env, monkeypatch):
    """Clause 1, read off disk: the drop of a `writes:` card is a ledger row of its own.

    The row is the artifact #3688's owed "5 real model trials per card" was undetectable
    against: a `grep` for the card over `ledger.jsonl` used to return its `split` row and
    nothing else, so "in every split, never once scored" and "scored recently" were the
    same query. Grep it now and the row that answers names the round, the card and both
    causes.
    """
    result = _shrink_fixture(round_env, monkeypatch, with_writes=True, budget_minutes=2)
    assert result["matrix_dropped_tasks"] == [WRITES_CARD], (
        "the precondition: the shrink gave up exactly the card under test, so what follows "
        "is about a drop and not about a round that never shrank")

    rows = _unscorable_rows(round_env, result["round_id"])
    assert len(rows) == 1, f"one record per dropped card, got {rows}"
    row = rows[0]
    assert row["event"] != "split", "the split row is the round-level one this replaces"
    assert row["round_id"] == result["round_id"], row
    assert row["task_id"] == WRITES_CARD, row
    assert row["declared_writes"] == ["vault"], row
    for fact in ("matrix shrink", "fixture-rooted aggregator", "require_copied_state"):
        assert fact in row["reason"], f"the reason must carry {fact!r}: {row['reason']}"
    assert " s window" in row["reason"], (
        "the reason carries the window in the same unit the report states it, so the two "
        f"artifacts cannot disagree about how tight the round was: {row['reason']}")

    # The fabricated-row hazard, which is the reason this record could not just be a
    # trial row with a zero in it: `derive_trial_priors` skips a row whose `variant_id`
    # is None, so a 0-second entry here would cost every future round more window than
    # the one that dropped this card — shrinking the matrix harder because a card was
    # unmeasurable. Same key keeps it out of `last_scored_round_by_task`.
    assert row["variant_id"] is None, row
    assert "duration_seconds" not in row, row
    assert WRITES_CARD not in {r["task_id"] for r in _rows(round_env, result["round_id"])}, (
        "the card must have no trial row: this record describes an absent trial, it is not one")


def test_the_report_line_for_such_a_drop_names_the_other_cause_and_the_word_unscorable(
        round_env, monkeypatch):
    """Clause 3, on the report: both causes in one sentence, and the card is unscorable.

    The line this replaces is the shrink's own, which is true as far as it goes — "would
    not fit the window" — and reads as lost coverage a bigger budget would return. What a
    reader has to be able to rule out from the report alone is exactly that inference, so
    the line says the word `unscorable`, keeps the window as one cause rather than
    deleting it, and names the check that would have refused the trial anyway.
    """
    result = _shrink_fixture(round_env, monkeypatch, with_writes=True, budget_minutes=2)
    report = (round_env.paths.rounds_dir / f"{result['round_id']}.md").read_text(
        encoding="utf-8")
    lines = [ln for ln in report.splitlines() if "unscorable" in ln]
    assert len(lines) == 1, f"one line naming the unscorable drops, got {lines}"
    line = lines[0]
    for fact in (WRITES_CARD, "writes: vault", "fixture-rooted aggregator",
                 "require_copied_state", "not merely not-run", run_round.UNSCORABLE_EVENT):
        assert fact in line, f"the line must carry {fact!r}: {line}"
    assert "- matrix reduced before trials:" in report, (
        "the window is still reported as one cause; the new line supplements it, it does "
        "not rewrite the shrink's own sentence")
    assert f"{run_round.UNSCORABLE_EVENT} in {round_env.paths.ledger_path}" in line, (
        "the line says which ledger it wrote, so a reader does not have to guess which "
        "store to grep")


def test_a_dropped_card_declaring_no_writes_earns_no_such_record_and_no_line(
        round_env, monkeypatch):
    """The record is about a card that could not run, not about every drop.

    Same bench, same task, same cost, same drop — one frontmatter key removed. Without
    this leg the change would add a row per drop to the store every round shrinks, which
    is a second channel for a fact the drop list already carries, and it would print
    `require_copied_state` about a card that never declares a write.
    """
    result = _shrink_fixture(round_env, monkeypatch, with_writes=False, budget_minutes=2)
    assert result["matrix_dropped_tasks"] == [WRITES_CARD], (
        "the same card, the same drop, minus the `writes:` declaration")
    assert _unscorable_rows(round_env, result["round_id"]) == []
    report = (round_env.paths.rounds_dir / f"{result['round_id']}.md").read_text(
        encoding="utf-8")
    assert not [ln for ln in report.splitlines() if "unscorable" in ln], report


def test_a_round_told_about_a_fixture_rooted_aggregator_stays_silent(round_env, monkeypatch):
    """The gate's other half: the claim is only made about a box the round can see.

    `LLOYD_BENCH_STATE_URL` is what whoever boots a fixture-rooted child sets, and nothing
    here can prove that server's state is a copy without asking it. A round that called a
    card unscorable while one was running would be publishing a false cause — and on the
    box that fixes the boot wiring, the record would then be the thing that misleads. So
    the record and the line both go quiet, and the ordinary shrink line stays, because
    the window really did drop the card.
    """
    result = _shrink_fixture(round_env, monkeypatch, with_writes=True, budget_minutes=2,
                             state_url="http://127.0.0.1:8599/state")
    assert result["matrix_dropped_tasks"] == [WRITES_CARD], "the drop still happened"
    assert _unscorable_rows(round_env, result["round_id"]) == [], (
        "no unscorable claim while an aggregator this round did not boot may be serving "
        "the trial")
    report = (round_env.paths.rounds_dir / f"{result['round_id']}.md").read_text(
        encoding="utf-8")
    assert not [ln for ln in report.splitlines() if "unscorable" in ln], report
    assert "- matrix reduced before trials:" in report


# ── #4163 clauses 3 and 4: one admitted writes card, and a row that says which
#    bound stopped the others ────────────────────────────────────────────────────
# `cd938624` (#3959) landed the recording half: a writes card the shrink drops gets
# an `unscorable_writes_trial` row with `variant_id: null`, so its zero trials are
# attributable instead of mysterious. Its cause sentence was a claim about the BOX
# (`UNSCORABLE_CAUSE`: no fixture-rooted aggregator is booted). A round that boots a
# child for one card and not for the others makes that sentence false for exactly the
# cards whose record it is — the failure mode the item names. These pin that the rows
# survive a boot with a cause that stops claiming none exists.
def _drop(task_id: str, writes: list[str], second_cause: str | None = None) -> dict:
    entry = {"task_id": task_id, "writes": writes, "window_seconds": 1435.0}
    if second_cause is not None:
        entry["second_cause"] = second_cause
    return entry


def test_the_rotation_admits_one_card_and_advances_across_rounds():
    cards = list(run_round.WRITES_ROTATION)
    assert len(cards) == 3, "029, 030 and 031 — the three cards the ledger holds rows for"
    picked = [run_round.pick_writes_card(cards, round_index=i) for i in range(6)]
    assert picked == cards + cards, (
        f"the admitted card must advance 029→030→031 across consecutive rounds and "
        f"wrap; it gave {picked}")
    assert run_round.pick_writes_card(["bench_001_x"], round_index=0) is None, (
        "a matrix with no writes card in it admits nothing")
    two = cards[:2]
    assert [run_round.pick_writes_card(two, round_index=i) for i in range(4)] == \
        [two[0], two[1], two[0], two[1]], (
        "with two of the three present the rotation still advances over what can run, "
        "rather than stalling on the id that is absent")


def test_the_rotation_index_counts_rounds_and_not_rows(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    assert run_round.writes_rotation_index(ledger) == 0, (
        "no ledger is no rounds, which is the rotation's start and not an error")
    lines = []
    for rid in ("R_1", "R_2", "R_3"):
        for _ in range(20):  # one round writes many rows; it is still ONE round
            lines.append(json.dumps({"round_id": rid, "event": "trial"}))
    lines.append("not json")
    ledger.write_text("\n".join(lines), encoding="utf-8")
    assert run_round.writes_rotation_index(ledger) == 3


def test_a_dropped_card_still_gets_its_row_when_a_child_was_booted_for_another():
    boot = {"card": run_round.WRITES_ROTATION[0]}
    drops = run_round.unscorable_writes_drops(
        [{"id": run_round.WRITES_ROTATION[1], "writes": ["kg"]},
         {"id": run_round.WRITES_ROTATION[0], "writes": ["vault"]}],
        [run_round.WRITES_ROTATION[1]], second_cause="booted for " + boot["card"])
    assert [d["task_id"] for d in drops] == [run_round.WRITES_ROTATION[1]], (
        "the card the child was booted for is not here — it has a trial; the one that "
        "was NOT admitted keeps its row")
    row = run_round.unscorable_writes_ledger_row("R_4", drops[0], 1435.0)
    assert row["event"] == run_round.UNSCORABLE_EVENT
    assert row["variant_id"] is None, (
        "still a null variant: this row must stay out of the arm p90s it never ran")
    assert "duration_seconds" not in row
    assert "booted for" in row["reason"], (
        f"the cause must say what actually applied, got: {row['reason']}")
    assert "no fixture-rooted aggregator is booted" not in row["reason"], (
        "publishing the box-level claim while a child IS booted is the false cause "
        "cd938624 exists to keep out of the record")


def test_a_boot_refused_by_a_precondition_names_that_precondition_in_the_row():
    why = "free disk 100 bytes below 2x the fixture estimate of 900 bytes"
    drops = run_round.unscorable_writes_drops(
        [{"id": run_round.WRITES_ROTATION[2], "writes": ["backlog"]}],
        [run_round.WRITES_ROTATION[2]], second_cause=why)
    row = run_round.unscorable_writes_ledger_row("R_5", drops[0], 900.0)
    assert why in row["reason"], row["reason"]
    assert "dropped by the pre-trial matrix shrink" in row["reason"], (
        "both causes stay in one sentence: the shrink and the boot bound are separate "
        "facts and a reader needs both to know what to change")


def test_the_row_is_unchanged_when_no_boot_was_attempted():
    drops = run_round.unscorable_writes_drops(
        [{"id": run_round.WRITES_ROTATION[0], "writes": ["vault"]}],
        [run_round.WRITES_ROTATION[0]], second_cause=None)
    row = run_round.unscorable_writes_ledger_row("R_6", drops[0], 1435.0)
    assert run_round.UNSCORABLE_CAUSE in row["reason"], (
        "with no child attempted, the box-level claim is TRUE and pinned as worded by "
        "test_the_cause_is_a_statement_about_this_box_and_not_a_tautology")


def test_the_boot_bound_is_window_and_disk_decided_before_any_trial():
    p90 = 140.5
    need = run_round.FIXTURE_BUILD_SECONDS + run_round.CHILD_BOOT_SECONDS + p90
    tight = run_round.boot_precondition(seconds_left=need - 1, agent_p90=p90,
                                        estimate_bytes=1000, free_bytes=10 ** 12)
    assert tight and "window" in tight and f"{p90:.1f}" in tight, (
        "the window bound must state the p90 it charged, or a reader cannot re-derive "
        f"the decision; got: {tight}")
    assert run_round.boot_precondition(seconds_left=need, agent_p90=p90,
                                       estimate_bytes=1000,
                                       free_bytes=10 ** 12) is None
    disk = run_round.boot_precondition(seconds_left=10_000, agent_p90=p90,
                                       estimate_bytes=900,
                                       free_bytes=int(900 * run_round.FIXTURE_DISK_FACTOR) - 1)
    assert disk and "disk" in disk, (
        "free space below 2x the copy's own estimate must refuse the copy rather than "
        "discover ENOSPC inside it, which is a partial second copy of the memory")
    assert run_round.boot_precondition(seconds_left=10_000, agent_p90=p90,
                                       estimate_bytes=900,
                                       free_bytes=900 * run_round.FIXTURE_DISK_FACTOR) is None


def test_a_fixture_left_behind_fails_the_round_and_says_so_in_the_report(round_env,
                                                                        tmp_path,
                                                                        monkeypatch):
    """#4163 clause 5, measured on the production `run()` path.

    The fixture area is `DATA_ROOT/eval/fixtures`, and a copy still there after a round
    that completed means its own teardown lost track of it: a second copy of the memory
    tree, readable by anything that walks the data root, with only the sweep's 30-minute
    grace between it and a deletion the round was supposed to have done itself. The
    verdict therefore has to reach the REPORT — a payload key and `main`'s exit status
    are both invisible to the reader who opens the artifact next morning, which is the
    failure `a-verdict-that-is-only-printed` names.
    """
    # The fixtures area is `<DATA_ROOT>/eval/fixtures`, and `leftover_fixtures` reads
    # DATA_ROOT from `app.paths`, so the scratch root is a directory of this test's own.
    data_root = tmp_path / "data-root"
    stranded = data_root / "eval" / "fixtures" / "dw-20261116-050000-bench_029-x"
    monkeypatch.setattr("app.paths.DATA_ROOT", data_root)

    # The copy has to appear DURING the round, not before it. The check is a difference
    # from a baseline the round reads before it dispatches anything, precisely because the
    # area is shared: a hand-run trial's per-trial copy, or a concurrent round's live one,
    # can be sitting there the whole time, and `sweep_orphan_fixtures` collects only `dw-*`
    # trees — so a census of the area would fail every later round forever for one stray
    # copy nobody in that round made. So the double below is the round's own trial
    # dispatch, and the directory it leaves behind is the copy that dispatch built.
    dispatched = run_round._run_trials
    attempts: list[int] = []

    async def _dispatch_that_leaves_a_copy(cfg_, *a, **k):
        if not attempts:
            stranded.mkdir(parents=True, exist_ok=True)
        attempts.append(1)
        return await dispatched(cfg_, *a, **k)

    monkeypatch.setattr(run_round, "_run_trials", _dispatch_that_leaves_a_copy)
    # No boot in either round: the round must be failed for the copy it left, not for one
    # it might have booted, and the second round has to be unable to build a fresh one.
    monkeypatch.setattr(run_round, "boot_precondition",
                        lambda **_kw: "window below build+boot+one agent-loop p90")

    def _run(_cfg):
        return asyncio.run(run_round.run(
            targets=["memory"], budget_minutes=5, max_variants=1, model="m-test"))

    result = _run(round_env)
    assert result["leftover_fixtures"] == ["dw-20261116-050000-bench_029-x"], (
        "the round's own result must name the copy it failed to remove")
    report = (round_env.paths.rounds_dir / f"{result['round_id']}.md").read_text(
        encoding="utf-8")
    assert "FIXTURE LEFT BEHIND" in report and "dw-20261116-050000-bench_029-x" in report, (
        f"the report must carry the failure, got:\n{report}")
    # And a copy that was already there when a round started is not that round's failure:
    # the stranded directory is still on disk from the first round, and this round builds
    # nothing, so it predates the baseline and publishes nothing. This is the half a census
    # of the area could not express at all — and the only shape that survives a stray copy.
    result2 = asyncio.run(run_round.run(targets=["memory"], budget_minutes=5,
                                        max_variants=1, model="m-test"))
    assert result2["leftover_fixtures"] == [], (
        "a copy the round did not build is not a copy the round failed to remove")
    report2 = (round_env.paths.rounds_dir / f"{result2['round_id']}.md").read_text(
        encoding="utf-8")
    assert "FIXTURE LEFT BEHIND" not in report2


def test_the_report_states_what_the_round_decided_about_a_writes_card(round_env):
    """#4163 clauses 1 and 3 at the level a reader actually looks at.

    A round that booted a child and admitted a trial has to say so where a reader can see
    it: the ledger's per-trial row for that card does not name the boot that made the
    score possible, and a round that skipped the boot is otherwise indistinguishable from
    one that never considered it.
    """
    result = asyncio.run(run_round.run(targets=["memory"], budget_minutes=5,
                                       max_variants=1, model="m-test"))
    report = (round_env.paths.rounds_dir / f"{result['round_id']}.md").read_text(
        encoding="utf-8")
    assert "- writes-scoped trial:" in report, (
        f"every round states its writes decision; got:\n{report}")
    assert "no writes-scoped card in this round's matrix" in report, (
        "this bench corpus declares no writes card, and the report says the true reason "
        "rather than leaving the line out")


def test_a_production_rooted_round_probes_and_censuses_the_area_it_builds_in(round_env,
                                                                            tmp_path,
                                                                            monkeypatch):
    """#4163 clause 5 where it actually bites: the round's data root IS the production one.

    Every other node here runs under a scratch data root, and on a scratch root the fixtures
    area IS `<data_root>/eval/fixtures`, so those nodes cannot tell the two wirings apart. On
    this box `app.paths.DATA_ROOT` answers `PRODUCTION_DATA_ROOT` for a real round, and the
    copy then goes to `WRITES_FIXTURE_AREA` — a different filesystem path, and the only one
    the round's teardown, its leftover census and its free-disk precondition may read. A
    census of the data root reports an empty fixtures area while a copy of the vault sits
    under `~/lloyd-work`; a disk probe of the data root measures a filesystem the copy never
    touches. Both errors read as a healthy round.
    """
    prod_data = tmp_path / "prod-data"
    prod_vault = tmp_path / "prod-vault"
    area = tmp_path / "lloyd-work"
    prod_data.mkdir()
    prod_vault.mkdir()
    monkeypatch.setattr("app.data_root.PRODUCTION_DATA_ROOT", prod_data)
    monkeypatch.setattr("app.data_root.PRODUCTION_VAULT", prod_vault)
    monkeypatch.setattr("app.paths.DATA_ROOT", prod_data)
    monkeypatch.setattr(run_round, "WRITES_FIXTURE_AREA", area)

    probes: list[str] = []
    real_probe = run_round.fixtures_free_bytes

    def _probe(path):
        probes.append(str(path))
        return real_probe(path)

    monkeypatch.setattr(run_round, "fixtures_free_bytes", _probe)
    # Force the skip so this node measures the wiring, not a child: the precondition is
    # patched, so the free-byte probe it was handed is still this round's own reading.
    monkeypatch.setattr(run_round, "boot_precondition",
                        lambda **_kw: "window below build+boot+one agent-loop p90")

    # A copy left in each area during the round's dispatch. Only the one under the AREA the
    # round builds in is this round's failure; the data-root copy is what a hand-run built
    # under the old layout, and flagging it would fail every later round forever.
    dispatched = run_round._run_trials
    attempts: list[int] = []

    async def _dispatch_that_leaves_copies(cfg_, *a, **k):
        if not attempts:
            (area / "eval" / "fixtures" / "dw-in-area").mkdir(parents=True)
            (prod_data / "eval" / "fixtures" / "dw-in-data-root").mkdir(parents=True)
        attempts.append(1)
        return await dispatched(cfg_, *a, **k)

    monkeypatch.setattr(run_round, "_run_trials", _dispatch_that_leaves_copies)
    # A round whose matrix has no writes card in it never reaches the precondition at all,
    # and the scratch corpus contains none (every other node here depends on that). The two
    # readers the boot decision calls are patched instead of the corpus: the card the round
    # picks and the scopes it declares are what decide whether a probe happens.
    monkeypatch.setattr(run_round, "pick_writes_card",
                        lambda *_a, **_k: run_round.WRITES_ROTATION[0])
    monkeypatch.setattr(run_round, "writes_declared",
                        lambda *_a, **_k: ["vault"])
    result = asyncio.run(run_round.run(targets=["memory"], budget_minutes=5,
                                      max_variants=1, model="m-test"))

    assert result["leftover_fixtures"] == ["dw-in-area"], (
        "the census must see the copy in the area the round booted into and must not blame "
        f"this round for one sitting in the data root; got {result['leftover_fixtures']}")
    assert probes, "the round must probe free bytes before deciding to boot"
    assert probes[0] == str(area / "eval" / "fixtures"), (
        f"the disk precondition must measure the filesystem the copy lands on, not the data "
        f"root's; it measured {probes[0]}")
    report = (round_env.paths.rounds_dir / f"{result['round_id']}.md").read_text(
        encoding="utf-8")
    assert "dw-in-area" in report and "dw-in-data-root" not in report, (
        f"the report names one and not the other; got:\n{report}")


def test_the_free_byte_probe_of_an_area_that_does_not_exist_yet_is_its_filesystem(tmp_path):
    """A box that has never booted a writes trial has no fixtures directory.

    `shutil.disk_usage` raises on a missing path, and the caller reads any raise as 0 free
    bytes — which the precondition then refuses every boot forever, on a box with 2.1 TB
    free, with the row claiming a disk shortage that does not exist. So the walk-up to the
    nearest existing ancestor is what makes the guard usable at all, and it is the same
    filesystem: the area is about to be `mkdir`ed inside that ancestor.
    """
    import shutil as _sh

    missing = tmp_path / "lloyd-work" / "eval" / "fixtures"
    assert not missing.exists()
    measured = run_round.fixtures_free_bytes(missing)
    assert measured == _sh.disk_usage(tmp_path).free, (
        "the probe must answer with the free bytes of the filesystem the directory will be "
        "created on; it gave a value that is not that filesystem's")
    assert measured > 0
    assert run_round.fixtures_free_bytes(tmp_path) == _sh.disk_usage(tmp_path).free, (
        "an area that does exist is measured directly, with no walk-up at all")


# ── #4163 clause 4: a boot that cannot happen takes the card out of the matrix ─────
#: The scratch corpus this file's `round_env` writes contains none of the rotation, so a
#: node that wants to see a writes card REMOVED from the dispatched matrix has to put one
#: there first. It carries the rotation's first id and the scopes
#: `bench_029_durable_write_vault_note` really declares, and it does not require the engine,
#: so the round keeps it through the split and the only thing that can take it out is the
#: boot decision under test.
REFUSED_CARD = run_round.WRITES_ROTATION[0]


def _card_in_the_matrix(cfg, *, scopes: list[str]) -> None:
    card_file = cfg.paths.bench_dir / f"{REFUSED_CARD}.md"
    card_file.write_text(
        "---\n"
        f"id: {REFUSED_CARD}\n"
        "category: synthetic\n"
        "safety_critical: false\n"
        "writes:\n" + "".join(f"  - {s}\n" for s in scopes) +
        "prompt: write the note\n"
        "objective_checks: []\n"
        "---\n\nProse body.\n", encoding="utf-8")


def _refusal_rails(round_env, monkeypatch) -> Path:
    """Make the round's data root BE a production root, so the real rails refuse the build.

    `build_fixture` rule (2) refuses any destination inside `PRODUCTION_DATA_ROOT`
    (`bench_fixture.py:289-290`), and on this box a real round's fixtures area IS
    `<DATA_ROOT>/eval/fixtures`, so the rail fires on the live tree exactly as it does here
    once the constant names the root the area hangs off. Patching a constant is not stubbing
    a rail: the builder's own containment comparison still does the refusing, and
    `test_the_old_builder_arguments_are_refused_by_the_rails_as_they_stand` in
    tests/test_autoresearch_writes_trial_boot.py pins that those rails are live against the
    UNPATCHED constants. It refuses before copying anything, so nothing needs cleaning.
    """
    from app import paths as app_paths

    data_root = Path(app_paths.DATA_ROOT)
    # The rail only bites if the boot is about to build inside this root, which is what
    # `writes_fixture_source` does for a source that is NOT itself production — the case this
    # scratch round is in. Asserted rather than assumed: if the source choice ever starts
    # sending this round somewhere else, the refusal below stops happening and the node must
    # fail here rather than silently boot a child.
    chosen_src, chosen_parent = run_round.writes_fixture_source()
    assert chosen_parent.is_relative_to(data_root), (
        f"this node refuses the build by naming the round's own data root production, which "
        f"only works while the boot builds under it; it chose {chosen_parent}")
    monkeypatch.setattr("scripts.autoresearch.bench_fixture.PRODUCTION_DATA_ROOT", data_root)
    return data_root


def test_a_boot_the_builder_refuses_takes_the_card_out_of_the_matrix_and_says_so(round_env,
                                                                              monkeypatch):
    """#4163 clause 4 at run level, with nothing stubbed on the refusal path.

    The refusal is the real rule (2) rail, so nothing on this path is stubbed.
    Two claims, each read off what the round actually did:

    - the card is GONE from the matrix `_run_trials` was handed, so no agent-loop trial is
      dispatched against a server that would only refuse it. The witness that it was ever
      there is the task list `fit_matrix` was asked to fit — without that, asserting absence
      in a scratch corpus that never held the card would pass on its own.
    - the row names the raise, and does not publish `UNSCORABLE_CAUSE`: a round that
      ATTEMPTED a boot and was refused by the rails cannot book a sentence saying none was
      ever booted, which is the false cause `cd938624` exists to prevent.
    """
    _card_in_the_matrix(round_env, scopes=["vault"])
    _refusal_rails(round_env, monkeypatch)

    def _ids_of(args):
        # Position-agnostic: take the first argument that IS a list of task dicts, so this
        # witness keeps reading the matrix even if the two helpers reorder their parameters.
        for arg in args:
            if isinstance(arg, list) and arg and isinstance(arg[0], dict) \
                    and "id" in arg[0]:
                return [str(t.get("id")) for t in arg]
        return []

    pre_fit: list[list[str]] = []
    real_fit = run_round.fit_matrix

    def _watch_fit(*a, **k):
        pre_fit.append(_ids_of(a))
        return real_fit(*a, **k)

    monkeypatch.setattr(run_round, "fit_matrix", _watch_fit)
    dispatched = run_round._run_trials
    seen: list[list[str]] = []

    async def _record_what_was_dispatched(*a, **k):
        seen.append(_ids_of(a))
        return await dispatched(*a, **k)

    monkeypatch.setattr(run_round, "_run_trials", _record_what_was_dispatched)
    # 30 minutes against a ~150 s boot bound: the shrink keeps every card, so the ONLY
    # thing that can remove this one from the matrix is the refused boot. The dropped-card
    # path is `test_a_dropped_writes_card_still_gets_an_unscorable_row`.
    result = asyncio.run(run_round.run(targets=["memory"], budget_minutes=30,
                                      max_variants=1, model="m-test"))

    assert pre_fit and REFUSED_CARD in pre_fit[0], (
        f"the round must have had the card to remove: fit_matrix was asked to fit "
        f"{pre_fit[0] if pre_fit else '(never called)'}")
    assert seen, "the round must still dispatch its other cards"
    for ids in seen:
        assert REFUSED_CARD not in ids, (
            f"a writes card whose boot the rails refused must not be dispatched for an "
            f"agent-loop trial against the supervised server; it appeared in {ids}")

    rows = _unscorable_rows(round_env, result["round_id"])
    mine = [r for r in rows if r["task_id"] == REFUSED_CARD]
    assert mine, (
        f"the trial is gone but the record is not; rows published: "
        f"{[(r['task_id'], r['event']) for r in rows]}")
    row = mine[0]
    assert "FixtureRefused" in row["reason"], (
        f"the row must name the refusal that actually stopped the boot, not a pretext; "
        f"got {row['reason']!r}")
    assert run_round.UNSCORABLE_CAUSE not in row["reason"], (
        f"a round that attempted a boot cannot publish a sentence saying none was booted; "
        f"got {row['reason']!r}")
    # `variant_id: None` is the load-bearing key: a row with a variant id would feed
    # `derive_trial_priors` a trial that never ran. The category is not on this row shape
    # (only a shrink drop carries one), so it is not asserted here.
    assert row["variant_id"] is None, row
    assert row["event"] == run_round.UNSCORABLE_EVENT, row


def test_a_refused_boot_is_not_recorded_as_a_shrink_drop(round_env, monkeypatch):
    """The refused-boot cause and the drop cause stay distinguishable in one row.

    Both travel through the same `second_cause` slot and both are true, but they answer
    different questions — one says the isolation rails stopped the boot, the other says the
    deadline did — and a reader deciding whether to widen the window must not be handed the
    wrong one. A budget that keeps the card is therefore the point of this node, not an
    accident of it: `dropped` must be False here, so any wording that mentions a drop is a
    bug this catches.
    """
    _card_in_the_matrix(round_env, scopes=["vault"])
    _refusal_rails(round_env, monkeypatch)
    result = asyncio.run(run_round.run(targets=["memory"], budget_minutes=30,
                                      max_variants=1, model="m-test"))
    row = [r for r in _unscorable_rows(round_env, result["round_id"])
           if r["task_id"] == REFUSED_CARD][0]
    assert "FixtureRefused" in row["reason"], row["reason"]
    for pretext in ("shrink", "dropped", "drop"):
        assert pretext not in row["reason"], (
            f"the shrink did not drop this card — the window carries it — so the reason "
            f"must not read as if it had; got {row['reason']!r}")


def test_a_boot_already_attempted_never_falls_back_to_the_box_level_cause():
    """Every cause the round can publish after it has TRIED, in one table.

    `boot is None` covers two different histories — skipped by a bound, or attempted and
    refused — and both must stop asserting `UNSCORABLE_CAUSE`, because by then the round has
    itself disproved it. The third row is the one case that keeps the box-level wording
    byte-identical: no writes card existed to boot for, which is what
    `test_the_cause_is_a_statement_about_this_box...` pins from the other side.
    """
    cause = run_round._writes_unscorable_cause
    refused = cause(boot=None, boot_skipped="FixtureRefused: rule 2", dropped=False)
    skipped = cause(boot=None, boot_skipped="window below build+boot+one agent-loop p90",
                    dropped=True)
    never_attempted = cause(boot=None, boot_skipped=None, dropped=True)
    booted = cause(boot={"card": REFUSED_CARD}, boot_skipped=None, dropped=False)

    for published in (refused, skipped, booted):
        assert run_round.UNSCORABLE_CAUSE not in published, published
    assert "FixtureRefused: rule 2" in refused, refused
    assert "window below build+boot" in skipped, skipped
    assert REFUSED_CARD in booted, booted
    assert never_attempted == run_round.UNSCORABLE_CAUSE + " and the shrink dropped it", (
        f"with no boot attempted at all the sentence must stay byte-identical, since that "
        f"is the only case whose claim it still supports; got {never_attempted!r}")


# ── #4320: the two report artifacts agree on WHY a writes card went unscored ─────────
#
# `R_20261116_111918.md` line 15 records the shrink dropping bench_029, bench_030 and
# bench_031, and line 16, two lines later, prints `writes-scoped trial: no writes-scoped card
# in this round's matrix`. A round cannot both drop a card it had and report that the card
# was never in the matrix; the shrink's victim list and the pick's matrix were simply two
# different lists. The same split ran through the two artifacts: the ledger row builder read
# each row's `second_cause`, while the report line printed the bare `UNSCORABLE_CAUSE`
# constant — so a round that HAD booted a fixture-rooted child still reported that no
# fixture-rooted aggregator is booted, in the one document a human reads.
#
# The nodes below pin the report side of #4320 clauses 3 and 5. The boot-file nodes pin the
# pick, the reserved slot and the single arm.

ROTATION_CARDS = (
    "bench_029_durable_write_vault_note",
    "bench_030_durable_write_kg_row",
    "bench_031_durable_write_backlog_card",
)


def _writes_card_file(cfg, task_id: str) -> None:
    """A `writes:`-declaring card on the serial arm, written where the round reads cards."""
    (cfg.paths.bench_dir / f"{task_id}.md").write_text(
        "---\n"
        f"id: {task_id}\n"
        "category: capability\n"
        "safety_critical: false\n"
        "requires_runtime: true\n"
        "writes:\n  - memory\n"
        "prompt: write one thing down\n"
        "objective_checks:\n  - type: contains\n    value: done\n"
        "---\n\nProse body.\n",
        encoding="utf-8")


def test_the_unadmitted_rotation_cards_get_rows_naming_the_bound(round_env, monkeypatch):
    """Clause 3: two cards out of the matrix, two rows, and the bound is the reason.

    These two cards are NOT the shrink's victims — the reserve priced them out before the
    matrix was costed, so no drop list can name them and the line that reports drop-list
    victims cannot be the thing that mentions them. Without their own rows the ledger would
    just stop mentioning two of the three durable-write cards, which reads as coverage nobody
    considered rather than as the one-per-round ruling of 2026-11-16 that excuses them.
    """
    cfg = round_env
    for cid in ROTATION_CARDS:
        _writes_card_file(cfg, cid)
    for i in range(6):
        (cfg.paths.bench_dir / f"bench_c{i}.md").write_text(
            "---\n"
            f"id: bench_c{i}\n"
            "category: capability\n"
            "safety_critical: false\n"
            "requires_runtime: true\n"
            "prompt: answer\n"
            "objective_checks:\n  - type: contains\n    value: done\n"
            "---\n\nProse body.\n",
            encoding="utf-8")
    reserves: list[run_round.WritesReserve] = []
    real_reserve = run_round.writes_reserve_for

    def spy_reserve(tasks, agent_p90, *, round_index):
        out = real_reserve(tasks, agent_p90, round_index=round_index)
        reserves.append(out)
        return out

    monkeypatch.setattr(run_round, "writes_reserve_for", spy_reserve)
    # A measured p90 at tonight's figure and a 1-minute budget: the shrink must fire, which
    # is the state where the old code dropped all three cards and reported none of them.
    monkeypatch.setattr(
        run_round, "derive_trial_priors",
        lambda *_a, **_kw: run_round.TrialPriors(
            direct=run_round.ArmPrior(seconds=60.0, n=20, measured=True),
            sdk=run_round.ArmPrior(seconds=140.9, n=20, measured=True)))
    monkeypatch.setattr(run_round, "boot_precondition",
                        lambda **_kw: "the window the round was given")

    result = asyncio.run(run_round.run(targets=["prompts"], budget_minutes=1))
    assert "error" not in result, result
    reserve = reserves[0]
    assert reserve.admitted in ROTATION_CARDS and len(reserve.unadmitted) == 2, reserve
    report = (cfg.paths.rounds_dir / f"{result['round_id']}.md").read_text(encoding="utf-8")

    rows = [r for r in _rows_all(cfg, result["round_id"])
            if r.get("event") == run_round.UNSCORABLE_EVENT]
    assert len(rows) == 3, (
        "one row for each of the three rotation cards: the two this round could not admit, "
        f"plus the admitted one whose boot the window refused: {rows}")
    bound_rows = [r for r in rows if run_round.ONE_PER_ROUND_BOUND in r["reason"]]
    assert {r["task_id"] for r in bound_rows} == set(ROTATION_CARDS) - {reserve.admitted}, (
        [r["task_id"] for r in bound_rows], reserve)
    for row in bound_rows:
        # The bound is named, and the box-level sentence is not: this round did address the
        # aggregator (it refused a boot for a stated reason), and "no fixture-rooted
        # aggregator is booted" is a different claim from the one that excuses these cards.
        assert run_round.UNSCORABLE_CAUSE not in row["reason"], row
        assert row["declared_writes"] == ["memory"], row
        assert "the window the round was given" in row["reason"], row
    # The admitted card's own row says what happened to IT, and never borrows the bound:
    # the round did try to give this card its one trial.
    admitted_row = next(r for r in rows if r not in bound_rows)
    assert admitted_row["task_id"] == reserve.admitted, admitted_row
    assert run_round.ONE_PER_ROUND_BOUND not in admitted_row["reason"], admitted_row
    assert "the window the round was given" in admitted_row["reason"], admitted_row

    line = [ln for ln in report.splitlines()
            if ln.startswith(f"- {run_round.UNSCORABLE_HEAD} this round")]
    assert len(line) == 1, report
    # The header distinguishes the two ways a writes card leaves a round, and a card that
    # was never a shrink victim must not be described as one.
    assert "not admitted: one writes trial per round" in line[0], line[0]
    assert "dropped by the matrix shrink" not in line[0], line[0]
    assert run_round.ONE_PER_ROUND_BOUND in line[0], line[0]
    assert run_round.UNSCORABLE_CAUSE not in line[0], line[0]
    # The projection says the same thing in numbers, on its own terms rather than folded into
    # an arm's `x p90 each`, and the report prints TWO of them — the matrix it started and the
    # matrix it planned — over different pools, so each writes term is pinned on the line that
    # can carry it. Every phrase is quoted whole: `"0 s" in report` is satisfied by any
    # `…140 s` figure, and "not admitted" is already in the unscorable line above, so the old
    # loose pair passed on a `_projection_text` that never mentioned the writes cards at all.
    started = next(ln for ln in report.splitlines() if ln.startswith("- trial matrix: "))
    planned = next(ln for ln in report.splitlines()
                   if ln.startswith("- matrix reduced before trials: "))
    # Started: the admitted card alone, at the boot's real spend — the 6 s fixture copy + 3 s
    # child boot + one agent-loop p90 of 140.9 s = 149.9 s, printed rounded. That is the SAME
    # sum `boot_precondition` refuses a short window at, so the report's arithmetic and the
    # admission bound cannot disagree (#4320 clause 2 pins the equality at the function).
    assert "+ 1 writes trial ≈ 150 s" in started, started
    # Planned: the two cards the bound kept out, priced at nothing and named. They are absent
    # from the STARTED pool — `fit_matrix` takes them out rather than letting a later drop
    # "save" their cost — so this line is the only place their 0 s can be said.
    assert "[2 writes card(s) not admitted this round: " in planned, planned
    assert "— 0 s]" in planned, planned
    for cid in sorted(set(ROTATION_CARDS) - {reserve.admitted}):
        assert cid in planned, planned


def test_the_report_line_prints_the_rows_own_cause_and_the_ledger_wording_is_unchanged():
    """Clause 5, at the formatter: the row's cause, the constant only when a row has none.

    `unscorable_writes_ledger_row` has always honoured `second_cause` (`f"and {second or
    UNSCORABLE_CAUSE}"`), while `_unscorable_writes_text` printed the bare constant, so the
    ledger row and the report line described the same card differently by construction. The
    bug was unobservable while `pick_writes_card` was fed the shrunk matrix, because then no
    round ever had a boot to describe — which is why this node seeds the boot rather than
    running a round and hoping one happens.

    Pinned together with it: the sentence for a row that carries NO second cause, byte for
    byte as it read before, and the unchanged shape of the line (`UNSCORABLE_HEAD` prefix,
    `event=… in <ledger>` suffix) that promotion and grep both key off.
    """
    ledger = Path("/tmp/ledger.jsonl")
    victim = {"task_id": "bench_029_durable_write_vault_note", "writes": ["vault"],
              "second_cause": "writes trial child booted for "
                              "bench_029_durable_write_vault_note (fixtures under /tmp/f)"}
    plain = {"task_id": "bench_031_durable_write_backlog_card", "writes": ["backlog"]}

    attempted = run_round._unscorable_writes_text([victim], ledger)
    assert len(attempted) == 1, attempted
    assert attempted[0].startswith(f"- {run_round.UNSCORABLE_HEAD} this round "), attempted[0]
    assert victim["second_cause"] in attempted[0], attempted[0]
    assert run_round.UNSCORABLE_CAUSE not in attempted[0], (
        "a round that attempted a boot cannot report that no fixture-rooted aggregator "
        f"is booted: {attempted[0]}")

    untouched = run_round._unscorable_writes_text([plain], ledger)
    assert run_round.UNSCORABLE_CAUSE in untouched[0], untouched[0]
    assert untouched[0].endswith(
        f"; recorded per card as event={run_round.UNSCORABLE_EVENT} in {ledger}"), untouched[0]

    # The ledger row's own wording is untouched by any of this: same sentence with a
    # second cause, same sentence without one.
    with_cause = run_round.unscorable_writes_ledger_row("R_T_5", dict(victim, not_dropped=True),
                                                        None)
    assert with_cause["reason"] == f"unscorable this round: {victim['second_cause']}", with_cause
    without = run_round.unscorable_writes_ledger_row("R_T_5", dict(plain), 1430.0)
    assert without["reason"] == (
        "unscorable this round: dropped by the pre-trial matrix shrink (1430 s window), "
        f"and {run_round.UNSCORABLE_CAUSE}"), without


def _rows_all(cfg, round_id: str) -> list[dict]:
    """Every ledger line for the round, EVENTS INCLUDED — `_rows` filters them out, and the
    rows this node reads are all events."""
    return [r for r in (json.loads(line) for line in
                        cfg.paths.ledger_path.read_text(encoding="utf-8").splitlines())
            if r.get("round_id") == round_id]


# ── #4633: what the report says when promote() refuses ────────────────────────

def test_a_refused_promotion_is_reported_as_a_refusal_and_not_as_a_landing(
        tmp_path, monkeypatch):
    """A round whose `promote()` returned `refused` writes no `## Promoted` heading, and
    says out loud that the promotion was refused.

    The block is not decoration on this page: `post_promotion._PROMOTED_VID_RE` reads
    it back as THE landing record, so a block written from a refusal became the record
    `last_promotion` hands to `auto_restore` — which then refuses to undo a write that
    never happened. Two live reports carry the artifact verbatim, `rounds/R_20261112_200024.md`
    and `rounds/R_20261115_111538.md`, each printing a variant line whose
    `snapshot_dir` is `None` beside `- applied_files: []`, while the vault holds a
    promote commit for three other rounds and none for these two.

    The decision itself stays on the page (`## Promotion decisions` still records the
    PROMOTE, because the predicate really did decide to promote); what changes is that
    the write it could not perform is stated as a refusal, naming the variant and
    quoting the `refused` strings `promote()` returned — the only refusal text a promote
    result carries, since `reason` and `refusal_class` are fields of the `decision` row.
    """
    from tests.test_autoresearch_promotion import (
        REFUSAL_TEXT, REFUSED_VARIANT, drive_promote_round, make_cfg, refused_promote_result)

    cfg = make_cfg(tmp_path)
    result = drive_promote_round(cfg, monkeypatch, refused_promote_result())
    report = (cfg.paths.rounds_dir / f"{result['round_id']}.md").read_text(encoding="utf-8")

    assert "## Promoted" not in report
    assert "- snapshot_dir: `None`" not in report, "the phantom's own line"
    assert "## Promotion refused" in report
    assert REFUSED_VARIANT in report
    assert REFUSAL_TEXT in report, "the refusal is quoted, not summarised into nothing"
    assert "wrote no file" in report
    assert f"- `{REFUSED_VARIANT}`: PROMOTE" in report, "the decision is still recorded"
