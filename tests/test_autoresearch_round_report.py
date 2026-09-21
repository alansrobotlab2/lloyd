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

    async def fake_trials(cfg_, variant_pairs, tasks, model, harness, max_parallel):
        direct = [{"variant_id": vid, "task_id": t["id"], "status": "success",
                   "task_category": t.get("category"), "turns": 1, "harness": "direct",
                   "final_text": "I called vault_recall and the answer is done.",
                   "tool_calls": [], "denied_calls": [], "duration_seconds": 1.0}
                  for vid, _ in variant_pairs for t in tasks]
        return direct, []

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
