"""#1546: an autoresearch round ends inside the cap the pool would kill it at.

Every round from 2026-09-24 to 09-26 was cancelled at `max_duration_seconds=1800`
mid-matrix, 3-4 times each, and wrote nothing but its run_spec.yaml: the source
handed it a hard-coded 60-minute budget and the round read no budget at all.
These pin the four links of the fix: the source derives the budget from the cap,
the spec records it, both runners stop starting trials at the deadline, and a
round that stopped still ranks, reports and ledgers what it measured — and never
promotes or restores on a partial measurement.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest

from scripts.autoresearch import bench_runner, bench_runner_sdk, run_round
from scripts.autoresearch.common import _run_spec_from_cfg, validate_run_spec
from tests.test_autoresearch_promotion import make_cfg
from tests.test_autoresearch_round_report import _rows, round_env  # noqa: F401  (fixture)
from workers.queue import QueueItem
from workers.sources import autoresearch as src


# ── the source: budget from the cap ─────────────────────────────────────────

class _Queue:
    def __init__(self):
        self.payloads = []

    def enqueue(self, *, source, kind, payload, priority, dedup_key):
        self.payloads.append(payload)
        return 1


def test_the_budget_is_the_pool_cap_in_minutes_never_the_old_sixty():
    q = _Queue()
    asyncio.run(src.enqueue_if_due(q, {"max_duration_seconds": 1800, "max_variants": 3}))
    assert q.payloads == [{"budget_minutes": 30, "max_variants": 3}]


def test_a_budget_may_be_lowered_but_never_raised_past_the_cap():
    assert src.budget_minutes_for({"max_duration_seconds": 1800}, 60) == 30
    assert src.budget_minutes_for({"max_duration_seconds": 1800, "budget_minutes": 20}) == 20
    assert src.budget_minutes_for({"max_duration_seconds": 1800}, "12") == 12
    assert src.budget_minutes_for({}) == src.POOL_DEFAULT_CAP_SECONDS // 60


def test_a_row_queued_with_the_old_sixty_runs_on_the_current_cap(monkeypatch):
    seen = {}

    async def fake_run(**kw):
        seen.update(kw)
        return {"round_id": "R_x", "decisions": [], "deadline_stopped": True,
                "tasks_not_reached": ["bench_a"]}

    monkeypatch.setattr(src, "_live_src_cfg",
                        lambda: {"max_duration_seconds": 1800, "max_variants": 3})
    monkeypatch.setattr(run_round, "run", fake_run)
    item = QueueItem(id=1, source="autoresearch", kind="round", payload={"budget_minutes": 60},
                     priority=60, dedup_key=None, state="running", attempts=1,
                     enqueued_at="", claimed_at=None, claimed_by=None, completed_at=None,
                     error=None)
    out = asyncio.run(src.execute(item))
    assert seen["budget_minutes"] == 30 and seen["max_variants"] == 3
    assert "stopped at its 30 min budget, 1 task(s) not reached" in out["summary"]


# ── the spec: the budget is on disk ────────────────────────────────────────

def test_the_run_spec_records_the_budget_it_was_handed(tmp_path):
    spec = _run_spec_from_cfg(make_cfg(tmp_path), "primary", 30)
    assert spec["budget"]["budget_minutes"] == 30
    assert validate_run_spec(spec) is None


# ── the runners: nothing starts past the deadline ──────────────────────────

VARIANTS = [("BASE", Path("/nonexistent")), ("V1", Path("/nonexistent"))]
TASKS = [{"id": "t1"}, {"id": "t2"}]


def _direct_trace(task, vid, _odir, _model, timeout):
    return {"variant_id": vid, "task_id": task["id"], "status": "success", "timeout": timeout}


def test_the_direct_runner_starts_nothing_once_the_deadline_is_past(monkeypatch):
    monkeypatch.setattr(bench_runner, "_run_one_sync", _direct_trace)
    traces = asyncio.run(bench_runner.run_bench(None, VARIANTS, TASKS, "m",
                                                deadline=time.monotonic() - 1))
    assert traces == []


def test_the_direct_runner_walks_task_by_task_and_keeps_its_timeout_far_from_the_deadline(monkeypatch):
    order = []

    def rec(task, vid, odir, model, timeout):
        order.append((task["id"], vid))
        return _direct_trace(task, vid, odir, model, timeout)

    monkeypatch.setattr(bench_runner, "_run_one_sync", rec)
    traces = asyncio.run(bench_runner.run_bench(None, VARIANTS, TASKS, "m", max_parallel=1,
                                                per_task_timeout=300,
                                                deadline=time.monotonic() + 3600))
    assert order == [("t1", "BASE"), ("t1", "V1"), ("t2", "BASE"), ("t2", "V1")]
    assert {t["timeout"] for t in traces} == {300} and not any(t.get("deadline_cut") for t in traces)


def test_a_trial_near_the_deadline_runs_on_the_time_left_and_is_marked_cut(monkeypatch):
    deadline = time.monotonic() + 40
    monkeypatch.setattr(bench_runner.time, "monotonic", lambda: deadline)  # the trial ran to the end
    trace = bench_runner.mark_if_cut({"task_id": "t1"}, 39, 300, deadline)
    assert trace["deadline_cut"] is True
    assert bench_runner.deadline_timeout(300, deadline + 39, 20) == 39
    assert bench_runner.deadline_timeout(300, deadline + 10, 20) is None
    assert bench_runner.deadline_timeout(300, None, 20) == 300


def test_the_agent_loop_runner_starts_nothing_once_the_deadline_is_past(monkeypatch):
    async def trial(*a, **kw):
        raise AssertionError("a trial started past the deadline")

    monkeypatch.setattr(bench_runner_sdk, "run_trial", trial)
    traces = asyncio.run(bench_runner_sdk.run_bench_sdk(
        None, VARIANTS, TASKS, "m", deadline=time.monotonic() + 30))
    assert traces == [], "under SDK_MIN_TRIAL_SECONDS left, nothing starts"


def test_the_agent_loop_runner_clamps_a_trial_to_the_time_left(monkeypatch):
    got = []

    async def trial(task, vid, odir, model, *, per_task_timeout, **kw):
        got.append(per_task_timeout)
        return {"variant_id": vid, "task_id": task["id"], "status": "success"}

    monkeypatch.setattr(bench_runner_sdk, "run_trial", trial)
    asyncio.run(bench_runner_sdk.run_bench_sdk(
        None, VARIANTS[:1], TASKS[:1], "m", per_task_timeout=600,
        deadline=time.monotonic() + 200))
    assert len(got) == 1 and 190 <= got[0] <= 200


# ── the round: what the deadline reached is ranked, the rest is named ──────

def test_complete_matrix_drops_a_task_any_variant_missed_or_was_cut_on():
    traces = [
        {"variant_id": "B", "task_id": "t1"}, {"variant_id": "V", "task_id": "t1"},
        {"variant_id": "B", "task_id": "t2"},                                      # V never reached t2
        {"variant_id": "B", "task_id": "t3"}, {"variant_id": "V", "task_id": "t3", "deadline_cut": True},
    ]
    kept, missing = run_round.complete_matrix(traces, ["B", "V"],
                                              [{"id": "t1"}, {"id": "t2"}, {"id": "t3"}])
    assert missing == ["t2", "t3"] and {t["task_id"] for t in kept} == {"t1"}


def test_the_trial_deadline_leaves_the_judge_its_reserve():
    assert run_round.trial_deadline(0.0, None) is None
    assert run_round.trial_deadline(0.0, 30) == 1800 - 360
    assert run_round.trial_deadline(0.0, 10) == 600 - 180


def test_a_stopped_round_reports_and_ledgers_what_it_measured_and_promotes_nothing(round_env, monkeypatch):
    cfg = round_env
    variant = {"variant_id": "V_1", "hypothesis": "h", "surface": "SOUL.md", "edits": []}
    monkeypatch.setattr(run_round, "propose_variants", lambda cfg_, **kw: [variant])
    monkeypatch.setattr(run_round, "materialize_variants",
                        lambda cfg_, variants, base: ([base, ("V_1", cfg_.paths.variants_dir)], {}))

    def refuse(*a, **kw):
        raise AssertionError("a deadline-stopped round must not promote, compare or restore")

    monkeypatch.setattr(run_round, "promote", refuse)
    monkeypatch.setattr(run_round, "post_promotion_comparison", refuse)
    monkeypatch.setattr(run_round.auto_restore, "restore_for_decline", refuse)
    monkeypatch.setattr(run_round, "post_promotion_check", refuse)
    handed = {}

    async def stopped(cfg_, variant_pairs, tasks, model, harness, max_parallel, *, deadline=None):
        handed["deadline"] = deadline
        base = {"status": "success", "turns": 1, "harness": "direct", "tool_calls": [],
                "denied_calls": [], "duration_seconds": 1.0,
                "final_text": "I called vault_recall and the answer is done."}
        # Every variant reached bench_a1; only the baseline reached bench_a2.
        return ([{**base, "variant_id": vid, "task_id": "bench_a1"} for vid, _ in variant_pairs]
                + [{**base, "variant_id": variant_pairs[0][0], "task_id": "bench_a2"}], [])

    monkeypatch.setattr(run_round, "_run_trials", stopped)
    started = time.monotonic()
    result = asyncio.run(run_round.run(targets=["prompts"], bench_limit=2, budget_minutes=30))
    assert "error" not in result, result
    assert handed["deadline"] == pytest.approx(started + 1800 - 360, abs=5)

    assert result["deadline_stopped"] is True and result["tasks_not_reached"] == ["bench_a2"]
    assert result["budget_minutes"] == 30 and result["tasks_run"] == 1
    assert [d["should_promote"] for d in result["decisions"]] == [False]
    assert result["decisions"][0]["reason"].startswith("deadline_stopped: 1 task(s) not reached")
    assert result["promoted"] is None and result["round_summary_row"] is None

    rows = [json.loads(line) for line in cfg.paths.ledger_path.read_text().splitlines()]
    mine = [r for r in rows if r.get("round_id") == result["round_id"]]
    assert any(r.get("event") == "spec" for r in mine), "the spec row is written"
    assert {r["task_id"] for r in _rows(cfg, result["round_id"])} == {"bench_a1"}, \
        "trial rows for what every variant reached, none for the task it dropped"
    assert any(r.get("event") == "decision" or "should_promote" in r for r in mine), \
        "the decision row is written"
    assert not any(r.get("event") == "round_summary" for r in mine)

    report = Path(result["summary_file"]).read_text(encoding="utf-8")
    assert "- budget: 30 min" in report
    assert "- stopped at deadline: yes — 1 task(s) not reached by every variant, not scored: bench_a2" in report
    assert "HOLD — deadline_stopped" in report and "not comparable" in report
    spec = (cfg.paths.rounds_dir / result["round_id"] / "run_spec.yaml").read_text()
    assert "budget_minutes: 30" in spec


def test_a_round_that_finished_in_time_is_untouched_by_the_deadline(round_env):
    result = asyncio.run(run_round.run(targets=["prompts"], bench_limit=2, budget_minutes=30))
    assert "error" not in result, result
    assert result["deadline_stopped"] is False and result["tasks_not_reached"] == []
    assert result["tasks_run"] == 2
    report = Path(result["summary_file"]).read_text(encoding="utf-8")
    assert "- stopped at deadline: no\n" in report
