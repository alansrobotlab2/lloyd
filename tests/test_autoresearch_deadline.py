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
import re
import time
from pathlib import Path

import pytest

from scripts.autoresearch import bench_runner, bench_runner_sdk, promote, run_round
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
        # Every variant reached bench_a1; only the baseline reached bench_a2. The
        # third value is each arm's own wall seconds (#1605), and {} is what a
        # double that ran no arm can honestly report.
        return ([{**base, "variant_id": vid, "task_id": "bench_a1"} for vid, _ in variant_pairs]
                + [{**base, "variant_id": variant_pairs[0][0], "task_id": "bench_a2"}], [], {})

    monkeypatch.setattr(run_round, "_run_trials", stopped)
    started = time.monotonic()
    result = asyncio.run(run_round.run(targets=["prompts"], bench_limit=2, budget_minutes=30))
    assert "error" not in result, result
    assert handed["deadline"] == pytest.approx(started + 1800 - 360, abs=5)

    assert result["deadline_stopped"] is True and result["tasks_not_reached"] == ["bench_a2"]
    assert result["budget_minutes"] == 30 and result["tasks_run"] == 1
    assert [d["should_promote"] for d in result["decisions"]] == [False]
    # #1860 clauses 1 and 4, on the row this round actually appended. The prose is
    # byte-identical to the sentence this branch has always written, because its prefix is
    # what the strict-win census matches; the wrapped predicate now ALSO rides in a field,
    # so a consumer buckets this refusal by what refused it instead of by the deadline that
    # happened to be running. 21 of the 124 decision rows on the live ledger (re-measured
    # 2026-09-29) were mis-bucketed by exactly that reading.
    decision = result["decisions"][0]
    assert decision["reason"] == (
        "deadline_stopped: 1 task(s) not reached, no promotion from a partial round "
        "(predicate said: no_targeted_overlap)"), decision["reason"]
    assert "predicate said:" in decision["reason"]
    assert decision["predicate_refusal"] == "no_targeted_overlap", sorted(decision)
    rows = [json.loads(l) for l in cfg.paths.ledger_path.read_text().splitlines()
            if l.strip()]
    row = [r for r in rows if r.get("event") == "decision" and r.get("variant_id") == "V_1"]
    assert len(row) == 1, rows
    assert row[0]["refusal_class"] == "no_targeted_overlap", row[0]
    assert row[0]["reason"] == decision["reason"], "the row carries the prose verbatim"
    assert row[0]["should_promote"] is False and row[0]["promoted"] is False
    assert result["promoted"] is None and result["post_promotion_restore"] is None
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


# ── #1605: the matrix is sized to the window before the first trial ──────────
#
# #1546 stopped the killing; this is the not-finishing. All four scheduled rounds
# of 2026-09-27 (R_20260927_060057, _081610, _121616, _161709) ran out their full
# 1440 s trial window and came back `deadline_stopped` with 2-3 tasks unreached —
# trial_seconds 1368.8, 1397.6, 1408.0, 1436.9 against that window, every time. And
# because `complete_matrix` drops an unreached task for EVERY variant, the cut also
# deleted those tasks from the pool a promotion is scored on: in R_20260927_161709
# the unreached pair was bench_016 and bench_017, both lint-valid, which is what
# took `valid_tasks` from the 10 the bench actually has down to 2.
#
# So the matrix is now costed before it starts, with the two arms costed the way
# they run (the direct arm over `max_parallel` workers, the agent-loop arm serially
# at `max_parallel=1` because both arms share the one primary vLLM slot). The nodes
# below pin the four links: the projection happens before a trial does; a matrix
# that cannot fit is reduced up front and the report names the reduction against
# the projection; a cut mid-matrix can no longer reach the lint-valid pool; and
# each arm's wall seconds land beside `trial_seconds` so the priors stay auditable.
#
# `_lint_valid_ids` is patched rather than the bench coaxed through the real lint,
# for the reason `tests/test_bench_lint.py` gives for replacing that pair: these
# nodes are about what the round DOES with the validity set, and the lint's verdict
# on a given file is that file's own test's to pin.

def _bench(cfg, specs: list[tuple]) -> None:
    """Replace the fixture bench with `specs`: (task_id, requires_runtime, category).

    One `contains: done` objective check each, so a scored trial needs nothing but
    the `final_text` the doubles hand back. The category matters as much as the
    flag: `bench_split.compute_split` refuses a bench with no held-out category
    (adversarial/safety) as a one-sided split, so every bench here keeps one.
    """
    for stale in cfg.paths.bench_dir.glob("*.md"):
        stale.unlink()
    for task_id, runtime, category in specs:
        (cfg.paths.bench_dir / f"{task_id}.md").write_text(
            "---\n"
            f"id: {task_id}\ncategory: {category}\nsafety_critical: false\n"
            "prompt: do the thing\n"
            f"requires_runtime: {'true' if runtime else 'false'}\n"
            "objective_checks:\n  - type: contains\n    value: done\n"
            "---\n\nProse body.\n", encoding="utf-8")


def _quiet_proposer(monkeypatch, n_variants: int = 0) -> None:
    """`n_variants` candidates survive, so the matrix has 1 + n arms — baseline
    included, which is the count the runners loop over and the count the projection
    must cost. `max_variants: 3` is what the scheduled round runs with."""
    variants = [{"variant_id": f"V_{i}", "hypothesis": f"hypothesis {i}",
                 "surface": "SOUL.md", "edits": []} for i in range(1, n_variants + 1)]

    def fake_materialize(cfg_, _variants, baseline_overlay):
        # The pair list is (variant_id, overlay_dir) per arm, baseline first — the
        # exact shape `run()` hands to `_run_trials` and to the projection.
        return ([baseline_overlay]
                + [(f"V_{i}", cfg_.paths.variants_dir) for i in range(1, n_variants + 1)]), {}

    monkeypatch.setattr(run_round, "propose_variants", lambda cfg_, **kw: list(variants))
    monkeypatch.setattr(run_round, "materialize_baseline",
                        lambda cfg_: ("BASELINE_t", cfg_.paths.variants_dir))
    monkeypatch.setattr(run_round, "materialize_variants", fake_materialize)


def _drive_round(monkeypatch, valid_ids=None, arm_seconds: dict | None = None):
    """Patch the round to bench without an engine, and record what it started.

    Returns the list the trial matrix was handed, in the order it was handed: that
    order is the schedule, since both runners start trials down it and stop at the
    deadline, so it is what decides which tasks a cut removes.
    """
    if valid_ids is not None:
        monkeypatch.setattr(run_round, "_lint_valid_ids", lambda bench_dir: set(valid_ids))
    handed: list[str] = []

    async def spy(cfg_, variant_pairs, tasks, model, harness, max_parallel, **kw):
        handed.extend(t["id"] for t in tasks)
        traces = [{"variant_id": vid, "task_id": t["id"], "status": "success", "turns": 1,
                   "harness": "sdk" if t.get("requires_runtime") else "direct",
                   "tool_calls": [], "denied_calls": [], "duration_seconds": 1.0,
                   "task_category": t.get("category"),
                   "final_text": "I called vault_recall and the answer is done."}
                  for vid, _ in variant_pairs for t in tasks]
        return traces, [], dict(arm_seconds or {})

    monkeypatch.setattr(run_round, "_run_trials", spy)
    return handed


def test_the_round_projects_both_arms_separately_before_the_first_trial(round_env, monkeypatch):
    """#1605 clause 1: the projection is made before a trial starts, and it costs the
    two arms the way they actually run.

    Two direct tasks and one `requires_runtime` task, one arm: the direct arm puts
    its trials on 4 workers at once, so it is one 5 s wave, while the agent-loop arm
    is awaited serially at 90 s a trial — 95 s projected against a ~1440 s window,
    so all three start. The split is the point: one combined number cannot tell a
    round which arm to shrink, and at these priors a runtime task costs about
    eighteen direct ones.
    """
    _bench(round_env, [("bench_a1", False, "replay"), ("bench_a2", False, "synthetic"),
                       ("bench_r1", True, "safety")])
    _quiet_proposer(monkeypatch, 0)
    handed = _drive_round(monkeypatch, valid_ids={"bench_a1", "bench_a2", "bench_r1"})
    started = time.monotonic()
    result = asyncio.run(run_round.run(targets=["prompts"], budget_minutes=30))
    assert "error" not in result, result

    proj = result["matrix_projection"]
    assert (proj["arms"], proj["tasks"]) == (1, 3)
    assert (proj["direct_tasks"], proj["sdk_tasks"]) == (2, 1)
    assert (proj["direct_trials"], proj["sdk_trials"]) == (2, 1)
    assert proj["direct_seconds"] == 5.0, "2 trials over 4 workers is one 5 s wave"
    assert proj["sdk_seconds"] == 90.0, "the runtime task is 90 s of window, serially"
    assert proj["projected_seconds"] == 95.0
    assert 1380 < proj["window_seconds"] <= 1440 and time.monotonic() - started < 60
    assert proj["fits"] is True and result["matrix_dropped_tasks"] == []
    assert sorted(handed) == ["bench_a1", "bench_a2", "bench_r1"], "all three started"


def test_a_window_too_small_for_the_serial_task_starts_the_round_without_it(round_env, monkeypatch):
    """#1605 clauses 1 and 2 in their smallest shape: a 3-minute budget leaves a 90 s
    trial window (budget minus the 180 s `JUDGE_RESERVE_MIN_SECONDS` floor), the
    default matrix costs 95 s of it, and the 90 s margin ceiling is 81 s — so the
    round starts on the two direct tasks and says so, rather than starting all three
    and being cut.

    The runtime task's 90 s is not a guess at how long the model needs: it is
    `SDK_MIN_TRIAL_SECONDS` (`scripts/autoresearch/bench_runner_sdk.py:155`), below
    which that runner starts nothing at all. A task the window cannot give 90 s is a
    task the round cannot run, whatever the trial would have wanted.
    """
    _bench(round_env, [("bench_a1", False, "replay"), ("bench_a2", False, "synthetic"),
                       ("bench_r1", True, "safety")])
    _quiet_proposer(monkeypatch, 0)
    handed = _drive_round(monkeypatch, valid_ids={"bench_a1", "bench_a2", "bench_r1"})
    result = asyncio.run(run_round.run(targets=["prompts"], budget_minutes=3))
    assert "error" not in result, result

    assert 80 < result["matrix_projection"]["window_seconds"] <= 90
    assert result["matrix_dropped_tasks"] == ["bench_r1"]
    assert handed == ["bench_a1", "bench_a2"]
    assert result["matrix_projection"]["projected_seconds"] == 5.0
    assert result["matrix_projection"]["fits"] is True
    assert result["deadline_stopped"] is False and result["tasks_not_reached"] == []
    assert result["tasks_run"] == 2
    report = Path(result["summary_file"]).read_text(encoding="utf-8")
    assert "- matrix reduced before trials: 1 task(s) dropped" in report
    assert "= 95 s projected" in report, "the report quotes what the default matrix cost"
    assert "started instead at 5 s: bench_r1" in report


def test_the_scheduled_shape_reaches_every_task_inside_the_derived_window(round_env, monkeypatch):
    """#1605 clause 2 on the live numbers: the shape the scheduled round runs — the
    19-task bench at its real routing split (14 direct / 5 runtime, from the
    `requires_runtime` flags on bench_010, 014, 015, 016, 017), `max_variants: 3` so
    4 arms, and the derived 30-minute budget — ends with `deadline_stopped` false and
    nothing unreached.

    The arithmetic is why it never finished before. The window is 1440 s (1800 minus
    the 360 s judge reserve), and the margin puts the ceiling at 1296 s of it. Four
    arms over 14 direct tasks is 56 trials on 4 workers = 14 waves of 5 s = 70 s; four
    arms over 5 agent-loop tasks is 20 serial trials at 90 s = 1800 s; 1870 s does not
    fit, so the round gives up agent-loop work — 360 s of window each, being 4 arms at
    90 s — until it does: `bench_016` and `bench_017` go, 1150 s projected, three of the
    five agent-loop tasks still scored, and 1150 s is inside what the measurement says
    the round can spend (216 direct trials at a mean 2.8 s and 40 agent-loop trials at a
    mean 82.1 s across the four stopped rounds — 1140 s of trial work in total).
    Seventeen of the nineteen tasks are scored and NONE is unreached. A cut never gets
    that choice: it scores what it reached and deletes the rest from every mean at once,
    which is what 2026-09-27 measured — R_20260927_161709 came in two tasks short, both
    lint-valid, which is how the day's decisions ended up reporting `valid_tasks` of 2
    and 4 against a 10-task pool.

    The safety and adversarial tasks stay in the matrix all the way through, including
    `bench_010`, whose own saving is exactly as large as the audit tasks':
    `_drop_preference` defers a held-out task while any task outside the veto frees
    comparable window. The validity set here is every task on purpose, so this node is
    about cost and the veto; the lint-invalid-first tiebreak has its own node below.
    """
    # Categories and routing mirror the live bench exactly: 6 replay + 10 synthetic +
    # 2 adversarial + 1 safety, with `requires_runtime` on the safety task and on four
    # synthetic ones — bench_010 plus bench_014..017. The categories are load-bearing:
    # the shrink keeps the veto slice (adversarial/safety, `bench_split
    # .HELDOUT_CATEGORIES`) while any task outside it can still be given up, and those
    # four synthetic audit tasks are the non-veto agent-loop work it trades instead.
    specs = [("bench_p1", False, "replay"), ("bench_p2", False, "replay"),
             ("bench_p3", False, "replay"), ("bench_p4", False, "replay"),
             ("bench_p5", False, "replay"), ("bench_p6", False, "replay"),
             ("bench_s1", False, "synthetic"), ("bench_s2", False, "synthetic"),
             ("bench_s3", False, "synthetic"), ("bench_s4", False, "synthetic"),
             ("bench_s5", False, "synthetic"), ("bench_s6", False, "synthetic"),
             ("bench_014", True, "synthetic"), ("bench_015", True, "synthetic"),
             ("bench_016", True, "synthetic"), ("bench_017", True, "synthetic"),
             ("bench_008", False, "adversarial"), ("bench_009", False, "adversarial"),
             ("bench_010", True, "safety")]
    assert len(specs) == 19
    ids = {t for t, _r, _c in specs}
    _bench(round_env, specs)
    _quiet_proposer(monkeypatch, 3)
    handed = _drive_round(monkeypatch, valid_ids=ids)
    result = asyncio.run(run_round.run(targets=["prompts"], budget_minutes=30))
    assert "error" not in result, result

    planned = result["matrix_planned"]
    assert (planned["arms"], planned["tasks"]) == (4, 19)
    assert (planned["direct_tasks"], planned["sdk_tasks"]) == (14, 5)
    assert (planned["direct_seconds"], planned["sdk_seconds"]) == (70.0, 1800.0)
    assert planned["projected_seconds"] == 1870.0
    assert planned["fits"] is False and planned["window_seconds"] <= 1440

    proj = result["matrix_projection"]
    assert (proj["direct_tasks"], proj["sdk_tasks"]) == (14, 3)
    assert proj["projected_seconds"] == 1150.0
    assert proj["fits"] is True
    assert result["matrix_dropped_tasks"] == ["bench_017", "bench_016"]
    assert sorted(handed) == sorted(ids - {"bench_016", "bench_017"})
    # The veto slice survives the shrink — it is what a promotion is refused on.
    assert {"bench_008", "bench_009", "bench_010"} <= set(handed)
    assert result["deadline_stopped"] is False, "the round the item is about, not happening"
    assert result["tasks_not_reached"] == []
    assert result["tasks_run"] == 17

    report = Path(result["summary_file"]).read_text(encoding="utf-8")
    assert "- matrix reduced before trials: 2 task(s) dropped" in report
    assert "= 1870 s projected" in report, "the report names the projection it refused"
    assert "started instead at 1150 s: bench_017, bench_016" in report
    assert "- stopped at deadline: no" in report


def test_a_deadline_cut_can_no_longer_reach_the_lint_valid_pool(round_env, monkeypatch):
    """#1605 clause 3: the task list is ordered so a cut mid-matrix can only remove
    tasks the promotion would exclude anyway.

    An eleven-task bench shaped like the live one — `requires_runtime` set on
    bench_002 and bench_010 as it is on bench_010 and bench_014-017 — with a
    lint-valid set of nine that leaves bench_001 and bench_003 out, and a trial
    matrix that reached nine tasks and stopped. `complete_matrix` then reports the two
    unreached tasks, and they are exactly the two lint-invalid ones, so `valid_tasks`
    (scored intersected with lint-valid, `promote.py:455-459`) still holds all nine.

    The counterfactual below is what makes that a mechanism and not a coincidence: the
    same nine trials against the UNORDERED bench leave bench_010 and bench_011
    unreached, both lint-valid — the tasks the clause exists to spare. The ordering is
    the whole difference, and it costs the round nothing: the same nine tasks are
    scored either way.
    """
    ids = [f"bench_{n:03d}" for n in range(1, 12)]
    runtime = {"bench_002", "bench_010"}
    valid = set(ids) - {"bench_001", "bench_003"}
    _bench(round_env, [(i, i in runtime,
                        "safety" if i == "bench_010" else
                        "adversarial" if i == "bench_011" else
                        "replay" if i in ("bench_001", "bench_002") else "synthetic")
                       for i in ids])
    _quiet_proposer(monkeypatch, 0)
    monkeypatch.setattr(run_round, "_lint_valid_ids", lambda bench_dir: set(valid))

    tasks = run_round.load_bench_tasks(round_env.paths.bench_dir)
    assert [t["id"] for t in tasks] == ids, "bench order is file-name order"
    ordered = run_round.order_tasks_for_coverage(tasks, valid)
    assert [t["id"] for t in ordered[:9]] == sorted(valid), "valid first, bench order inside"
    assert [t["id"] for t in ordered[9:]] == ["bench_001", "bench_003"]
    assert [t["id"] for t in run_round.order_tasks_for_coverage(tasks, None)] == ids, \
        "no lint signal, no reordering"

    def _cut(cut_at: int, pool: list[dict]) -> list[str]:
        """What `complete_matrix` reports when the matrix stopped after `cut_at`
        tasks: the tail, for every variant at once."""
        traces = [{"variant_id": "BASELINE_t", "task_id": t["id"], "status": "success",
                   "turns": 1, "harness": "sdk" if t.get("requires_runtime") else "direct",
                   "tool_calls": [], "denied_calls": [], "duration_seconds": 1.0,
                   "final_text": "done"} for t in pool[:cut_at]]
        _kept, not_reached = run_round.complete_matrix(traces, ["BASELINE_t"], pool)
        return not_reached

    not_reached = _cut(9, ordered)
    assert not_reached == ["bench_001", "bench_003"]
    assert not set(not_reached) & valid, "the cut took nothing a promotion is scored on"
    assert _cut(9, tasks) == ["bench_010", "bench_011"]
    assert set(_cut(9, tasks)) & valid == {"bench_010", "bench_011"}, \
        "unordered, the same cut deletes two valid tasks"

    # And through a whole round, so the numbers the clause is about are the ones a
    # round actually reports when its matrix is cut.
    handed: list[str] = []

    async def cut_trials(cfg_, variant_pairs, tasks_, model, harness, max_parallel, **kw):
        handed.extend(t["id"] for t in tasks_)
        traces = [{"variant_id": vid, "task_id": t["id"], "status": "success", "turns": 1,
                   "harness": "sdk" if t.get("requires_runtime") else "direct",
                   "tool_calls": [], "denied_calls": [], "duration_seconds": 1.0,
                   "task_category": t.get("category"),
                   "final_text": "I called vault_recall and the answer is done."}
                  for vid, _ in variant_pairs for t in tasks_[:len(valid)]]
        return traces, [], {}

    monkeypatch.setattr(run_round, "_run_trials", cut_trials)
    result = asyncio.run(run_round.run(targets=["prompts"], budget_minutes=30))
    assert "error" not in result, result
    assert result["matrix_dropped_tasks"] == [], "the window fit, so nothing was shrunk"
    assert handed == [t["id"] for t in ordered], "the round starts the ordered list"
    assert result["tasks_run"] == 9
    assert result["tasks_not_reached"] == ["bench_001", "bench_003"]
    report = Path(result["summary_file"]).read_text(encoding="utf-8")
    assert "- stopped at deadline: yes" in report, "the cut is reported"
    assert "- matrix reduced before trials" not in report, \
        "nothing was shrunk up front, so nothing claims a shrink"


def test_each_harness_arm_reports_the_seconds_it_took(round_env, monkeypatch):
    """#1605 clause 4: the projection is auditable, so the round records each arm's
    own wall seconds beside the combined `trial_seconds`.

    The two arms are costed differently — parallel waves against a serial queue — so a
    single number cannot be checked against either prior. The keys arrive through the
    real `_run_trials` (pinned by the node below with timed fakes) and are `None`, not
    0.0, for an arm that had no task to run: 0 s would claim a measurement of an arm
    that never started, which is the false-green shape #1654 and #1383 are both about.
    """
    _bench(round_env, [("bench_a1", False, "replay"), ("bench_a2", False, "synthetic"),
                       ("bench_r1", True, "safety")])
    _quiet_proposer(monkeypatch, 0)
    _drive_round(monkeypatch, valid_ids={"bench_a1", "bench_a2", "bench_r1"},
                 arm_seconds={"direct_seconds": 3.2, "sdk_seconds": 61.5})
    result = asyncio.run(run_round.run(targets=["prompts"], budget_minutes=30))
    assert "error" not in result, result
    assert (result["direct_seconds"], result["sdk_seconds"]) == (3.2, 61.5)
    assert result["trial_seconds"] >= 0.0
    report = Path(result["summary_file"]).read_text(encoding="utf-8")
    assert "(direct 3 s, agent-loop 62 s)" in report, report.splitlines()


# The fixture below this section replaces `_run_trials` to keep an engine out of a
# round, so the node that measures the arms has to name the real function: captured at
# import, while the module attribute still is it.
_REAL_RUN_TRIALS = run_round._run_trials


def test_an_arm_that_never_ran_reports_no_seconds_of_its_own(round_env, monkeypatch):
    """#1605 clause 4, at the seam: the seconds are measured by the real `_run_trials`, not
    inherited from a double.

    Both runners are faked with a timed sleep and `_REAL_RUN_TRIALS` — the untouched
    function, since the `round_env` fixture patches `_run_trials` over the module
    attribute — is what gets called, so what is pinned here is the timing and the
    routing and not a double's arithmetic: each arm's key appears only when that arm was
    handed a task, and its value is at least the sleep that arm spent.
    """
    def _wait(seconds):
        async def _fake(cfg_, variant_pairs, tasks, model, **kw):
            await asyncio.sleep(seconds)
            return [{"variant_id": vid, "task_id": t["id"], "status": "success", "turns": 1,
                     "tool_calls": [], "denied_calls": [], "duration_seconds": seconds,
                     "final_text": "I called vault_recall and the answer is done."}
                    for vid, _ in variant_pairs for t in tasks]
        return _fake

    cfg = round_env
    monkeypatch.setattr(run_round, "run_bench", _wait(0.20))
    monkeypatch.setattr(run_round, "run_bench_sdk", _wait(0.05))
    tasks = [{"id": "bench_a1", "prompt": "a", "category": "replay"},
             {"id": "bench_r1", "prompt": "r", "category": "safety",
              "requires_runtime": True}]
    _direct, _sdk, arm_seconds = asyncio.run(_REAL_RUN_TRIALS(
        cfg, [("BASELINE_t", cfg.paths.variants_dir)], tasks,
        "primary", "auto", 4))
    assert arm_seconds["direct_seconds"] >= 0.20
    assert arm_seconds["sdk_seconds"] >= 0.05

    monkeypatch.setattr(run_round, "run_bench", _wait(0.05))
    _direct, _sdk, direct_only = asyncio.run(_REAL_RUN_TRIALS(
        cfg, [("BASELINE_t", cfg.paths.variants_dir)], tasks[:1],
        "primary", "auto", 4))
    assert "direct_seconds" in direct_only and "sdk_seconds" not in direct_only, \
        "an arm with no task reports no time, not zero"


def test_a_safety_task_is_given_up_only_when_nothing_outside_the_veto_can_fit(round_env, monkeypatch):
    """#1605 clause 2's other edge: the veto protection is comparative, so a round
    whose only expensive task IS a safety task gives it up rather than lose its bench.

    A 3-minute budget leaves a 90 s trial window (budget minus the
    `JUDGE_RESERVE_MIN_SECONDS` floor), and the matrix costs 95 s of it: the two direct
    tasks free 5 s each, the safety task 90 s. Deferring the held-out task absolutely
    would spend both drops on the direct tasks, still not fit, and end with a bench of
    one task to protect one task — so `_drop_preference` defers a veto task only while a
    task outside the slice frees comparable window, and here none does. `bench_016`/
    `bench_017` above are the same decision going the other way, where the audit tasks
    did free the same window and so went first.
    """
    _bench(round_env, [("bench_a1", False, "replay"), ("bench_a2", False, "synthetic"),
                       ("bench_r1", True, "safety")])
    _quiet_proposer(monkeypatch, 0)
    handed = _drive_round(monkeypatch, valid_ids={"bench_a1", "bench_a2", "bench_r1"})
    result = asyncio.run(run_round.run(targets=["prompts"], budget_minutes=3))
    assert "error" not in result, result
    assert result["matrix_dropped_tasks"] == ["bench_r1"]
    assert handed == ["bench_a1", "bench_a2"]
    assert result["deadline_stopped"] is False and result["tasks_run"] == 2



# ── the priors a round costs itself on (#1715) ────────────────────────────────
# Everything above took the per-trial cost on trust: `project_matrix` and
# `fit_matrix` defaulted it to the module constants, and on both post-landing
# scheduled rounds that trust printed `fits yes` and the deadline took a task
# anyway — `rounds/R_20260928_041936.md:9-12`, 1180 s projected against a 1431 s
# window, 1370 s of trials spent, 1 task not reached. The priors were 5 s and 90 s;
# the same ledger the round appends to says p90 27.9 s (n=1473) and 183.7 s (n=67).
# So the priors are now derived from those rows, read once per round at one call
# site, named with their provenance beside the line they priced, and a shrink step
# that would buy less window than `PROJECTION_MARGIN` already holds back is refused
# out loud instead of taken all the way down to `MIN_MATRIX_TASKS`.

def _prior_ledger(path: Path, *, direct_secs: float, sdk_secs: float,
                  n_direct: int = 30, n_sdk: int = 25) -> Path:
    """A ledger carrying the per-trial durations the derivation reads.

    `harness` and `duration_seconds` are exactly the two keys a round writes on a
    trial row, and one row per arm per task per variant is what the live file holds,
    so the derivation meets the shape it will meet in production. Durations are
    uniform within an arm because a percentile of identical values is that value,
    which is what lets the asserts below name a number rather than a range.
    """
    rows = []
    for arm, secs, n in (("direct", direct_secs, n_direct), ("sdk", sdk_secs, n_sdk)):
        for i in range(n):
            rows.append({"round_id": "R_priors", "variant_id": f"V_{i}",
                         "task_id": f"bench_{arm}_{i}", "harness": arm,
                         "trace_status": "success", "duration_seconds": secs,
                         "composite_score": 0.5})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


_LIVE = run_round.TrialPriors(
    direct=run_round.ArmPrior(27.9, 1473, True),
    sdk=run_round.ArmPrior(200.6, 67, True))


def _write_tasks(bench: Path, n: int, *, prefix: str = "bench",
                 runtime: bool = False):
    """Write `n` bench task files and hand back the task dicts `load_bench_tasks`
    would read from them. `requires_runtime` is what routes a task to the serial arm
    under `auto`, so a test that wants the matrix split across both arms asks for it
    here rather than hand-building a dict the loader would never produce."""
    bench.mkdir(exist_ok=True)
    out = []
    for i in range(n):
        tid = f"{prefix}{i}"
        (bench / f"{tid}.md").write_text(
            f"---\nid: {tid}\ncategory: audit\n"
            + ("requires_runtime: true\n" if runtime else "")
            + f"---\n\nprompt for {tid}\n", encoding="utf-8")
        out.append({"id": tid, "category": "audit",
                    **({"requires_runtime": True} if runtime else {})})
    return out


def test_the_priors_are_the_p90_of_the_ledgers_rows_and_it_says_which(tmp_path):
    """Clause 1: per-arm p90 with the n that produced it, the constant below the
    precondition, and the report saying which arm got which. `MIN_PRIOR_ROWS` is
    asserted against rather than the literal 20 repeated into the seeds, so moving
    the precondition moves the test with it."""
    ledger = tmp_path / "ledger.jsonl"

    got = run_round.derive_trial_priors(_prior_ledger(
        ledger, direct_secs=27.9, sdk_secs=200.6))
    assert got.sdk.measured is True and got.sdk.n == 25
    assert got.sdk.seconds == pytest.approx(200.6)
    assert got.sdk.percentile == run_round.PRIOR_PERCENTILE
    assert got.direct.measured is True and got.direct.n == 30
    assert got.direct.seconds == pytest.approx(27.9)

    thin = run_round.derive_trial_priors(_prior_ledger(
        ledger, direct_secs=27.9, sdk_secs=200.6, n_direct=run_round.MIN_PRIOR_ROWS,
        n_sdk=run_round.MIN_PRIOR_ROWS - 1))
    assert thin.direct.measured is True, "the precondition is at least 20, not more"
    assert thin.sdk.measured is False
    assert thin.sdk.seconds == run_round.SDK_TRIAL_SECONDS
    assert thin.sdk.n == run_round.MIN_PRIOR_ROWS - 1

    absent = run_round.derive_trial_priors(tmp_path / "nope.jsonl")
    assert (absent.direct.seconds, absent.sdk.seconds) == (
        run_round.DIRECT_TRIAL_SECONDS, run_round.SDK_TRIAL_SECONDS)
    assert absent.direct.measured is False and absent.direct.n == 0

    text = absent.provenance()
    assert "fallback constant" in text and "20 needed" in text, text
    assert "measured p90" not in text
    assert "n=30" in got.provenance() and "fallback" not in got.provenance()


def test_no_cost_on_the_shrink_path_can_fall_back_to_the_module_constants(
        tmp_path, monkeypatch):
    """Clause 2: one derived value in, the same value out of every cost the shrink
    takes. The spy sits on `project_matrix` itself, so a shrink path that rebuilt
    the priors from the constants — or dropped the argument and let a default do it
    — is caught by name rather than inferred from a total that happened to differ."""
    seen: list[run_round.TrialPriors] = []
    real = run_round.project_matrix

    def spy(*args, **kwargs):
        assert "priors" in kwargs, "a projection was costed without priors at all"
        seen.append(kwargs["priors"])
        return real(*args, **kwargs)

    monkeypatch.setattr(run_round, "project_matrix", spy)
    tasks = (_write_tasks(tmp_path, 15) + _write_tasks(tmp_path, 5, prefix="bench_rt", runtime=True))
    kept, proj, dropped = run_round.fit_matrix(4, tasks, "auto", 15,
                                              window_seconds=1431.0, priors=_LIVE)
    assert len(seen) >= 3, f"the shrink took {len(seen) - 1} step(s), too few to mean anything"
    assert all(p is _LIVE for p in seen), "a cost on the shrink path used other priors"
    assert proj["priors"] is _LIVE
    assert proj["direct_trial_seconds"] == 27.9 and proj["sdk_trial_seconds"] == 200.6
    # The number that comes out is the measured one: four runtime tasks go, the
    # fifth stays, 914 s against the same window. At the 90 s constant the identical
    # matrix keeps three runtime tasks and calls itself 1100 s — which is the
    # 2026-09-28 round, and it was cut.
    assert dropped == ["bench_rt4", "bench_rt3", "bench_rt2", "bench_rt1"], dropped
    assert proj["projected_seconds"] == pytest.approx(914.0)
    same_at_constant = run_round.project_matrix(4, tasks, "auto", 15,
                                                window_seconds=1431.0,
                                                priors=run_round.TrialPriors.fallback())
    assert same_at_constant["projected_seconds"] < run_round.project_matrix(
        4, tasks, "auto", 15, window_seconds=1431.0, priors=_LIVE)["projected_seconds"], \
        "the constants cost the identical matrix LESS: they are the optimistic ones"

    src = Path(run_round.__file__).read_text(encoding="utf-8").splitlines()
    calls = [ln.strip() for ln in src
             if "derive_trial_priors(" in ln and not ln.strip().startswith(("#", "def "))]
    assert calls == ["priors = derive_trial_priors(cfg.paths.ledger_path)"], \
        f"the round must read its own cost at exactly one call site, got {calls}"


def test_the_report_line_names_the_prior_per_arm_beside_where_it_came_from(
        round_env, tmp_path, monkeypatch):
    cfg = round_env
    assert cfg.paths.ledger_path == tmp_path / "ledger.jsonl"
    monkeypatch.setattr(run_round, "load_config", lambda: cfg)
    """Clause 3: the projection line and the provenance are one line. `×90 s each,
    serial` on its own reads like a measured figure; four scheduled rounds printed
    it and were cut on the strength of it."""
    _prior_ledger(tmp_path / "ledger.jsonl", direct_secs=27.9, sdk_secs=183.7,
                  n_direct=30, n_sdk=25)
    _quiet_proposer(monkeypatch, 0)
    result = asyncio.run(run_round.run(targets=["prompts"], dry_run=False,
                                       budget_minutes=30))
    body = (cfg.paths.rounds_dir / f"{result['round_id']}.md").read_text()
    line = [ln for ln in body.splitlines() if ln.startswith("- trial matrix:")][0]
    assert "s projected; priors — " in line, line
    assert "direct 27.9 s measured p90 (n=30)" in line, line
    assert "agent-loop 183.7 s measured p90 (n=25)" in line, line
    assert "fallback" not in line, line

    # The fixture's ledger holds the 55 rows the run above read, so the fallback is
    # shown by emptying that file rather than with a second fixture: the point is
    # that a fallback is legible in the same place a measured prior is.
    (tmp_path / "ledger.jsonl").write_text("", encoding="utf-8")
    plain = asyncio.run(run_round.run(targets=["prompts"], dry_run=False,
                                      budget_minutes=30))
    body = (cfg.paths.rounds_dir / f"{plain['round_id']}.md").read_text()
    line = [ln for ln in body.splitlines() if ln.startswith("- trial matrix:")][0]
    assert "direct 5.0 s fallback constant (n=0 measured rows, 20 needed)" in line, line
    assert "agent-loop 90.0 s fallback constant" in line, line


def test_shrinking_stops_when_a_step_buys_less_window_than_the_margin_holds(tmp_path):
    """Clause 4: the floor is on what a step frees, not on how many tasks are left.
    34 direct tasks in a 150 s window at 17 s each projects 170 s against a 135 s
    budget, and every step left frees exactly 17 s — less than the 15 s the margin is
    already declining to spend, so the old loop would have taken 33 of them and
    arrived at `MIN_MATRIX_TASKS` having bought nothing but lost 33 tasks' evidence.
    (The arithmetic is exact here: 17 >= 15 is why the step IS taken at a 180 s
    window, so the window is the thing under test, not the count.)"""
    priors = run_round.TrialPriors(
        direct=run_round.ArmPrior(17.0, 30, True),
        sdk=run_round.ArmPrior(183.7, 25, True))
    kept, proj, dropped = run_round.fit_matrix(4, _write_tasks(tmp_path, 34), "auto", 15,
                                              window_seconds=150.0, priors=priors)
    assert proj["fits"] is False, "the premise: 34 direct tasks do not fit 150 s"
    stopped = proj["shrink_stopped_early"]
    assert stopped["step_floor_seconds"] == pytest.approx(15.0)
    assert stopped["next_step_saves_seconds"] < stopped["step_floor_seconds"], stopped
    assert dropped == ["bench33"], \
        "one step was worth taking (17 s >= the 15 s floor); only the next is not"
    assert len(kept) == 33 > run_round.MIN_MATRIX_TASKS, "stopped by the floor, not at it"
    assert proj["projected_seconds"] > 150.0 * run_round.PROJECTION_MARGIN


def test_the_say_so_line_is_written_when_a_shrink_stops_with_the_matrix_over_window(
        round_env, tmp_path, monkeypatch):
    """Clause 4's report half. `fits NO` alone reads like a bug in the shrink; what
    has to be in the file is that the matrix is over window, that shrinking stopped,
    what stopped it, and that the deadline will therefore take tasks."""
    cfg = round_env
    monkeypatch.setattr(run_round, "load_config", lambda: cfg)
    _quiet_proposer(monkeypatch, 0)
    _prior_ledger(tmp_path / "ledger.jsonl", direct_secs=14.0, sdk_secs=183.7)
    _write_tasks(cfg.paths.bench_dir, 34)
    result = asyncio.run(run_round.run(targets=["prompts"], dry_run=False,
                                       budget_minutes=5))
    body = (cfg.paths.rounds_dir / f"{result['round_id']}.md").read_text()
    line = [ln for ln in body.splitlines() if "matrix over window" in ln]
    assert len(line) == 1, body
    # 34 written + the 3 the `round_env` fixture seeds = the 37-task matrix; 14 s is
    # what one direct task frees at the derived prior, 15 s the margin's own slack.
    assert "shrinking stopped" in line[0] and "37 task(s) left" in line[0], line[0]
    assert "projecting 140 s against 135 s of budget (5 s over)" in line[0], line[0]
    assert "would free only" in line[0] and "headroom already holds back" in line[0]
    assert "the deadline will cut some of these tasks" in line[0], line[0]
    assert result["matrix_dropped_tasks"] == []


def test_a_matrix_that_reports_fits_yes_fits_at_the_priors_it_named(tmp_path, monkeypatch):
    """Clause 5: 20 tasks over 4 arms at the live `auto` routing — 15 parallel, 5
    `requires_runtime` on the serial arm — against the window a 30-minute round
    derives (1440 s, so 1296 s of it is spendable), with the priors derived from a
    ledger carrying live-like durations (direct p90 28.0 s, agent-loop p90 200.6 s).

    Two things have to hold at once, and the first is the one the old code passed
    while a round was being cut: a matrix reported `fits yes` must re-cost under
    window x PROJECTION_MARGIN at the priors the report names. The second is that a
    mid-matrix cut still takes lint-INVALID tasks: `order_tasks_for_coverage` is the
    same function the runners start from, so the tail it leaves at the end is what a
    deadline takes, and that tail must share no task with what the lint calls valid.
    """
    priors = run_round.derive_trial_priors(_prior_ledger(
        tmp_path / "ledger.jsonl", direct_secs=28.0, sdk_secs=200.6))
    bench = tmp_path / "bench"
    bench.mkdir()
    tasks = (_write_tasks(bench, 15) + _write_tasks(bench, 5, prefix="bench_rt", runtime=True))
    window = 1431.0
    kept, proj, dropped = run_round.fit_matrix(4, tasks, "auto", 15,
                                              window_seconds=window, priors=priors)
    assert proj["priors"] is priors
    if proj["fits"]:
        recount = run_round.project_matrix(4, kept, "auto", 15,
                                           window_seconds=window, priors=priors)
        assert recount["projected_seconds"] <= window * run_round.PROJECTION_MARGIN
        assert recount["direct_trial_seconds"] == priors.direct.seconds
        assert recount["sdk_trial_seconds"] == priors.sdk.seconds
    else:
        assert proj["shrink_stopped_early"], \
            "a matrix that stops fitting has to say why shrinking stopped"

    # The lint's own rules are not what is under test here — which end of the order
    # a cut takes is — so the valid pool is stubbed the way the deadline test above
    # stubs it: the first 15 ids valid, the 5 runtime ones not. `valid_task_ids` is
    # the function `run()` calls, so the stub is the same seam production uses.
    valid = {t["id"] for t in tasks[:15]}
    import sys as _sys
    import types
    stub = types.ModuleType("scripts.autoresearch.bench_lint")
    stub.valid_task_ids = lambda _d: set(valid)
    monkeypatch.setitem(_sys.modules, "scripts.autoresearch.bench_lint", stub)
    ordered = run_round.order_tasks_for_coverage(tasks, valid)
    cut = ordered[-5:]                                    # a cut that takes 5
    assert not ({t["id"] for t in cut} & valid), [t["id"] for t in cut]
    assert {t["id"] for t in cut} == {t["id"] for t in tasks[-5:]}


# ── #1716: what a shrink gave up, in the two currencies the ruling is written in ──
#
# The standing budget-versus-coverage ruling says the code-side shrink suffices and
# `autoresearch.max_duration_seconds` stays at 1800 until a measurement re-opens it.
# A round could not produce that measurement: `matrix reduced before trials: 2
# task(s) dropped ... started instead at 1150 s: bench_017, bench_016` names the ids
# but never says that both were lint-valid out of an 18-task valid pool, nor that the
# two tasks came off the serial arm. These nodes pin the two counts, the arm the
# window came from, and the re-open condition, all of it report/payload only.

_SCHEDULED_19: list[tuple[str, bool, str]] = [
    # The live bench's own routing split and categories, same as the 19-task node
    # above: 14 direct / 5 `requires_runtime`, held-out slice = 2 adversarial + 1
    # safety, and the four non-veto agent-loop audit tasks bench_014..017.
    ("bench_p1", False, "replay"), ("bench_p2", False, "replay"),
    ("bench_p3", False, "replay"), ("bench_p4", False, "replay"),
    ("bench_p5", False, "replay"), ("bench_p6", False, "replay"),
    ("bench_s1", False, "synthetic"), ("bench_s2", False, "synthetic"),
    ("bench_s3", False, "synthetic"), ("bench_s4", False, "synthetic"),
    ("bench_s5", False, "synthetic"), ("bench_s6", False, "synthetic"),
    ("bench_014", True, "synthetic"), ("bench_015", True, "synthetic"),
    ("bench_016", True, "synthetic"), ("bench_017", True, "synthetic"),
    ("bench_008", False, "adversarial"), ("bench_009", False, "adversarial"),
    ("bench_010", True, "safety"),
]


def test_a_shrunk_round_reports_the_lint_valid_and_runtime_coverage_it_gave_up(
        round_env, monkeypatch):
    """#1716 clauses 1 and 3, plus the serial-arm state of clause 4, on the shape the
    scheduled round really runs: the 19-task bench, 4 arms, the derived 30-minute
    budget, fallback priors — the node above already shows that matrix dropping
    `bench_017` and `bench_016` to start at 1150 s.

    The lint-valid set is deliberately 18 of the 19 loaded tasks — every id except
    `bench_s1` — so the valid pool is a strict subset of what the round loaded and
    none of the three counts can be read as another. The shrink gives up `bench_016`
    and `bench_017`, both lint-valid, and `bench_s1` survives while being invalid, so
    the coverage line is `16 ... of 18`: not 17 (the tasks started), not 19 (the tasks
    loaded), not 18 (the pool). That triple is the whole of the item — #1605's own
    commit message names the bug as rounds reporting `valid_tasks` of 2 and 4 against a
    10-task pool, and the fix is only real if the denominator is on the same page as
    the numerator.

    The runtime half is the same claim in the other currency: 5 agent-loop tasks were
    planned, 3 started, and it has to be stated as tasks. `_projection_text` prints
    that arm as `sdk_trials` (arms × tasks), so 20 trials against 12 is the same fact
    only after dividing by the arm count — arithmetic a reader of a round report
    should not have to do to learn what the round gave up.
    """
    _bench(round_env, _SCHEDULED_19)
    _quiet_proposer(monkeypatch, 3)
    loaded = {tid for tid, _r, _c in _SCHEDULED_19}
    _drive_round(monkeypatch, valid_ids=loaded - {"bench_s1"})
    result = asyncio.run(run_round.run(targets=["prompts"], budget_minutes=30))
    assert "error" not in result, result
    assert result["matrix_dropped_tasks"] == ["bench_017", "bench_016"], \
        "the precondition: this round shrank, and gave up two lint-valid tasks"

    cov = result["matrix_coverage"]
    assert (cov["lint_valid_started"], cov["lint_valid_total"]) == (16, 18)
    assert (cov["runtime_started"], cov["runtime_planned"]) == (3, 5)
    assert cov["freed_window_serial_only"] is True, \
        "both victims were agent-loop tasks, so every freed second is serial"

    report = Path(result["summary_file"]).read_text(encoding="utf-8")
    assert ("- coverage given up by the shrink: 16 lint-valid task(s) started of "
            "18 on the bench; 3 requires_runtime task(s) started of 5 planned"
            in report), report
    assert "0.0 s on the direct arm, 720.0 s on the serial agent-loop arm" in report
    assert "freed by the serial agent-loop arm alone: yes" in report
    # Clause 4 of the item: the ruling's re-open condition travels with the number.
    # #1953 fixed which side of the number that number has to fall on: the trigger
    # fires on a delta OUTSIDE the spread, so that is what the line must state.
    assert "3 consecutive rounds" in report
    assert "outside that task's own p90 minus p10 spread" in report


def test_an_unreadable_lint_reports_lint_valid_coverage_as_unknown_not_zero(
        round_env, monkeypatch):
    """Clause 2: `_lint_valid_ids` returns None when the lint cannot be read, and
    `None` is not the empty set (`run_round.py:793-806` says so about the shrink
    order). Coverage must inherit that distinction: `0 lint-valid task(s) started of 0`
    would read as a bench whose every task the lint refuses — the strongest possible
    argument for a budget raise — when the truth is that the round knows nothing.

    The runtime half is unaffected by the lint, so it still reads 3 of 5: an unknown
    in one currency does not blank the other one out with it.
    """
    _bench(round_env, _SCHEDULED_19)
    _quiet_proposer(monkeypatch, 3)
    _drive_round(monkeypatch)                       # no valid set stubbed
    monkeypatch.setattr(run_round, "_lint_valid_ids", lambda bench_dir: None)
    result = asyncio.run(run_round.run(targets=["prompts"], budget_minutes=30))
    assert "error" not in result, result
    assert result["matrix_dropped_tasks"] == ["bench_017", "bench_016"]

    cov = result["matrix_coverage"]
    assert cov["lint_valid_total"] is None and cov["lint_valid_started"] is None
    assert (cov["runtime_started"], cov["runtime_planned"]) == (3, 5)

    report = Path(result["summary_file"]).read_text(encoding="utf-8")
    line = [ln for ln in report.splitlines()
            if ln.startswith("- coverage given up by the shrink")][0]
    assert "lint-valid coverage unknown (bench lint unreadable" in line, line
    assert "0 of 0" not in line and " of 0 " not in line, line
    assert "3 requires_runtime task(s) started of 5 planned" in line, line


def test_a_shrink_that_frees_only_direct_window_says_the_serial_arm_is_not_it(
        round_env, tmp_path, monkeypatch):
    """Clause 4's other state, which is the one that makes the flag worth printing: a
    matrix shrunk entirely on the parallel arm.

    The priors are the round's own measurement, not the constants (a measured 200 s
    direct trial, which is what makes a direct task worth more than the step floor),
    and the bench has no `requires_runtime` task at all: 4 arms × 8 direct tasks = 8
    waves of 200 s = 1600 s against a 1440 s window, so 1296 s of it is spendable and
    two tasks go, each freeing one whole wave. 400 s of window, none of it serial.
    `runtime_planned` and `runtime_started` are both 0 here, which is exactly why the
    flag cannot be inferred from the runtime counts and has to be said.
    """
    _prior_ledger(tmp_path / "ledger.jsonl", direct_secs=200.0, sdk_secs=90.0)
    specs = [("bench_d1", False, "replay"), ("bench_d2", False, "replay"),
             ("bench_d3", False, "synthetic"), ("bench_d4", False, "synthetic"),
             ("bench_d5", False, "synthetic"), ("bench_008", False, "adversarial"),
             ("bench_009", False, "adversarial"), ("bench_010", False, "safety")]
    _bench(round_env, specs)
    _quiet_proposer(monkeypatch, 3)
    _drive_round(monkeypatch, valid_ids={t for t, _r, _c in specs})
    result = asyncio.run(run_round.run(targets=["prompts"], budget_minutes=30))
    assert "error" not in result, result

    cov = result["matrix_coverage"]
    assert len(result["matrix_dropped_tasks"]) == 2, result["matrix_dropped_tasks"]
    assert (cov["freed_direct_seconds"], cov["freed_serial_seconds"]) == (400.0, 0.0)
    assert (cov["runtime_planned"], cov["runtime_started"]) == (0, 0)
    assert cov["freed_window_serial_only"] is False

    report = Path(result["summary_file"]).read_text(encoding="utf-8")
    assert "400.0 s on the direct arm, 0.0 s on the serial agent-loop arm" in report
    assert "freed by the serial agent-loop arm alone: no" in report


def test_the_coverage_numbers_reach_no_promotion_decision(round_env, monkeypatch):
    """Clause 5 across the one seam it could travel: `run()` hands
    `promote.evaluate_promotion` the config, the two summaries and the split, and the
    spy is on that call. A coverage figure that leaked into any argument would reach
    the predicate that decides promotions, which the item forbids; a coverage figure
    that leaked into a `decisions` row would reach the ledger row every downstream
    parser reads (`promotion_fp_rate.py`, #428's denominator).

    The assertions are on the call as made, not on a source grep: the numbers are
    computed in the same function that calls the predicate one line later, so only
    the argument list shows whether they crossed.
    """
    seen: list[tuple[tuple, dict]] = []
    real = run_round.evaluate_promotion

    def spy(*args, **kwargs):
        seen.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(run_round, "evaluate_promotion", spy)
    _bench(round_env, _SCHEDULED_19)
    _quiet_proposer(monkeypatch, 3)
    loaded = {tid for tid, _r, _c in _SCHEDULED_19}
    _drive_round(monkeypatch, valid_ids=loaded - {"bench_s1"})
    result = asyncio.run(run_round.run(targets=["prompts"], budget_minutes=30))
    assert "error" not in result, result

    assert len(seen) == 3, f"expected one predicate call per candidate, got {len(seen)}"
    for args, kwargs in seen:
        assert len(args) == 3 and set(kwargs) == {"split"}, \
            "the predicate's own signature is the contract being pinned here"
        blob = json.dumps([list(args), kwargs], sort_keys=True, default=str)
        assert "lint_valid" not in blob, "coverage reached the promotion predicate"
        assert "matrix_coverage" not in blob
        assert "freed_window_serial_only" not in blob
        # #1828 clause 4: the two keys this item adds are figures too, and the last
        # scored round is the one of them a reader could plausibly want in a
        # predicate. The spy is on the call, so this is the same claim the ones above
        # make — nothing about what the round gave up reaches the argument.
        assert "sacrificed_runtime_last_scored_round" not in blob, \
            "the per-task history reached the promotion predicate"
        assert "reopen_ledger" not in blob
        assert "no scored history" not in blob

    assert result["matrix_coverage"]["lint_valid_total"] == 18, \
        "the numbers exist, so the assertions above ruled out a real path"
    for d in result["decisions"]:
        # `predicate_refusal` is #1860's: what `evaluate_promotion` returned before the
        # deadline branch wrapped the prose, which is how the ledger row gets a class that
        # is not the wrap. No coverage figure rides in it — the spy above is what proves
        # that half, this set is what proves nothing else was added.
        assert set(d) == {"variant_id", "mean_composite", "targeted_delta",
                          "heldout_delta", "normalized_gain", "should_promote",
                          "reason", "predicate_refusal", "validity"}, set(d)


# ── #1828: the re-open line has to name where its two figures are read from ──────
#     #1953: ...and the direction it names has to be the verdict-moving one
#
# #1716 put the standing budget-versus-coverage ruling's re-open condition into every
# shrunk round report, but stated only the threshold: 3 consecutive rounds sacrificing
# the same runtime task, with the round's decision delta on one of them landing on that
# task's own measured spread. Neither figure had a source. The only p90 in the tree was
# `derive_trial_priors`'s, which is a per-ARM duration prior — a cost, not a score — so
# the condition could be argued from memory and measured from nothing. These nodes pin
# the source being named (clause 1), the payload entry that answers the "sacrifice the
# SAME task across rounds" half without a hand-search of `ledger.jsonl` (clause 2), and
# the unknown wording owed to a task that has never scored (clause 3).
#
# #1953 then found the direction inverted. Landing ON the spread is the no-gain case,
# and a shrink that moves no verdict is the evidence AGAINST a budget raise, not for
# one; the trigger now fires on a delta landing outside the task's p90 minus p10 spread,
# with a task whose measured spread is 0.0000 excluded beside the scored-row floor. The
# exact phrase #1953 deleted is quoted nowhere in this file — clause 1 is a grep that
# has to come back empty — so the negative assertion in the node below is written
# against the two-word shape that phrase ended in, and is scoped to the rendered line.


def _history_ledger(path: Path, rows: list[tuple[str, str, object]]) -> Path:
    """Seed `path` with the per-trial rows of rounds BEFORE the one under test.

    Each row is `(task_id, round_id, score)`, and a `None` score is the unrankable
    trial — `trial_ledger_row` nulls the field rather than writing 0.0, which is the
    exact state clause 3 is about. `variant_id` is set because that is what makes a row
    a trial row at all: the spec / split / decision / summary rows carry none, and a
    decision row's `round_id` would make every task on the bench look as though it ran
    in that round.

    No `duration_seconds`, on purpose. Those keys are what `derive_trial_priors` reads,
    so a seeded duration would move the arm priors and with them the shrink that the
    nodes below depend on; a row without one is dropped by that derivation and is still
    a complete scored trial row for the query this item is about.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(
        json.dumps({"round_id": rid, "variant_id": f"V_seed_{rid}", "task_id": tid,
                    "harness": "sdk", "trace_status": "success",
                    run_round.REOPEN_SCORE_FIELD: score}) + "\n"
        for tid, rid, score in rows), encoding="utf-8")
    return path


def test_the_reopen_line_names_the_ledger_rows_both_figures_are_read_from(
        round_env, monkeypatch):
    """Clause 1: on a round that shrank, the re-open line states where the delta and
    the spread come from, and the report says which round each victim was last scored
    in.

    The bench, the arms and the budget are the #1716 node's, so the shrink is the same
    one (`bench_017` and `bench_016` go, both serial-arm). What is added is a history:
    `bench_016` scored in two earlier rounds and `bench_017` in one, and the line has to
    credit each with the later of them — "most recent", not "first seen", which is the
    half a reader needs to count three consecutive rounds.

    The assertions are on the report text, not on the constant, because the clause is
    about what a reader of a round report sees: the threshold phrases #1716 pinned stay
    in place (the line is amended, not replaced) and the source now travels with them.
    """
    _history_ledger(round_env.paths.ledger_path, [
        ("bench_016", "R_20260928_041936", 0.4),
        ("bench_016", "R_20260929_082451", 0.6),
        ("bench_017", "R_20260928_041936", 0.5),
    ])
    _bench(round_env, _SCHEDULED_19)
    _quiet_proposer(monkeypatch, 3)
    loaded = {tid for tid, _r, _c in _SCHEDULED_19}
    _drive_round(monkeypatch, valid_ids=loaded - {"bench_s1"})
    result = asyncio.run(run_round.run(targets=["prompts"], budget_minutes=30))
    assert "error" not in result, result
    assert result["matrix_dropped_tasks"] == ["bench_017", "bench_016"], \
        "the precondition: this round shrank and gave up two serial-arm tasks"

    report = Path(result["summary_file"]).read_text(encoding="utf-8")
    line = [ln for ln in report.splitlines()
            if ln.startswith("- re-open condition")][0]
    # The threshold #1716 put there is still there — this is an amendment.
    assert "3 consecutive rounds" in line, line
    # ...and #1953 corrected which side of the spread the delta has to land on.
    assert "outside that task's own p90 minus p10 spread" in line, line
    # ...and it now names the one place each figure exists.
    assert "trial ledger" in line, line
    assert "cfg.paths.ledger_path" in line, line
    assert "`round_id` + `task_id`" in line, line
    assert run_round.REOPEN_SCORE_FIELD in line, line
    assert "p90 minus p10" in line, line
    assert f"{run_round.REOPEN_MIN_SCORED_ROWS} scored rows" in line, line

    cov = result["matrix_coverage"]
    assert cov["reopen_ledger"] == str(round_env.paths.ledger_path)
    assert cov["sacrificed_runtime_last_scored_round"] == {
        "bench_017": "R_20260928_041936",     # the one round it scored in
        "bench_016": "R_20260929_082451",     # the LATER of its two
    }, cov["sacrificed_runtime_last_scored_round"]
    last = cov["sacrificed_runtime_last_scored_round"]
    assert result["round_id"] not in last.values(), \
        ("the round that dropped a task never ran it, so its own id must not be "
         "credited as that task's last run")
    # WHICH tasks the entry covers is the report's own definition of runtime coverage,
    # not a second definition beside it. Every key is a task the bench marks
    # `requires_runtime: true`, and the entry is exactly as wide as the runtime coverage
    # the shrink costed — `runtime_planned` minus `runtime_started`, the two counts on
    # the report line above the re-open note. A reader can check the two keys against
    # each other arithmetically instead of by trusting which one they read first.
    runtime_ids = {md.stem for md in round_env.paths.bench_dir.glob("*.md")
                   if "requires_runtime: true" in md.read_text(encoding="utf-8")}
    assert set(last) <= runtime_ids, f"{sorted(last)} not all runtime: {sorted(runtime_ids)}"
    assert len(last) == cov["runtime_planned"] - cov["runtime_started"], \
        f"{sorted(last)} vs {cov['runtime_planned']} planned - {cov['runtime_started']} started"

    figures = [ln for ln in report.splitlines()
               if ln.startswith("- re-open figures are read from")][0]
    assert str(round_env.paths.ledger_path) in figures, figures
    assert "bench_016: R_20260929_082451" in figures, figures
    assert "bench_017: R_20260928_041936" in figures, figures


def test_the_rendered_reopen_line_names_the_outside_direction_and_the_zero_spread_guard(
        round_env, monkeypatch):
    """#1953 clauses 1 and 2, pinned on the text a shrunk round hands its readers.

    Nothing in this tree computes a p90 minus p10 spread: the re-open condition is a
    rule the owed-check job applies to ledger rows, and what the round owns is the
    sentence telling that job what to look for. So this node pins TEXT — the re-open
    line as it lands in the report file, plus the identical sentence inside the JSON
    payload the owed-check job reads instead of the markdown — and could pin nothing
    else. It is still a node that can fail: reverting the direction, and deleting the
    zero-spread clause, each take it down on their own.

    The direction it replaces asked for the delta to land inside the task's own measured
    spread — the no-gain case the line's own closing sentence disqualifies — so the
    trigger went off on the evidence AGAINST a raise. On the frozen witness bytes the
    note cites, `bench_014_audit_dead_wikilinks` has 28 scored rows across 7 rounds, a
    p90 minus p10 spread of 0.1667, and per-round best-variant deltas peaking at 0.1667:
    never once outside its own spread, yet it satisfied the condition as written. The
    scored-row floor does not cover a zero spread either, so that exclusion has to be on
    the page as well: on those same bytes `bench_009_adversarial_probe` has 180 scored
    rows over 35 rounds — nine times the 20-row floor — every one of them 1.0, so its
    spread is 0.0000 and any movement at all would read as outside it. Both exclusions
    travel with the floor, because a reader applies this line in one pass.

    The phrase this node exists to keep out is never quoted here, so the negative
    assertion runs against the two-word form `p90 spread`: the shape the old line ended
    in, and one the corrected line never writes because it defines the spread as p90
    minus p10. It is scoped to the re-open line, where that shape would come back, and
    says nothing about prose elsewhere in this file.
    """
    _bench(round_env, _SCHEDULED_19)
    _quiet_proposer(monkeypatch, 3)
    loaded = {tid for tid, _r, _c in _SCHEDULED_19}
    _drive_round(monkeypatch, valid_ids=loaded - {"bench_s1"})
    result = asyncio.run(run_round.run(targets=["prompts"], budget_minutes=30))
    assert "error" not in result, result
    assert result["matrix_dropped_tasks"] == ["bench_017", "bench_016"], \
        "the precondition: this round shrank, so its report carries the re-open line"

    report = Path(result["summary_file"]).read_text(encoding="utf-8")
    line = [ln for ln in report.splitlines()
            if ln.startswith("- re-open condition")][0]

    # The seam the owed-check job actually reads: `matrix_coverage` is written into the
    # JSON round artifact, so the corrected sentence has to be in that payload and not
    # only on the markdown page. One sentence, one source — the payload and the report
    # line are the same string, which is what makes pinning the line worth anything.
    payload = result["matrix_coverage"]["reopen_condition"]
    assert payload == line.removeprefix("- "), \
        ("the JSON payload a later job reads carries the same sentence the report "
         "does, not merely a key of the same name")
    assert "outside that task's own p90 minus p10 spread" in payload, payload

    # Clause 1: the direction, and what "outside" is measured against.
    assert "outside that task's own p90 minus p10 spread" in line, line
    assert "the delta's absolute value exceeds the spread" in line, line
    assert "p90 spread" not in line, \
        "the trigger may not name the two-word spread #1953 removed"
    # Clause 2: a task whose measured spread is zero supports the trigger on neither
    # side, and it is stated beside the row floor rather than as a second rule. "Beside"
    # means in the same sentence: a character-distance window would fail a rewording
    # that only moved the wording around, so the line is cut into sentences first. The
    # scored-row count itself is spelled as a digit inside a number, and a sentence
    # boundary needs a period plus a space, so the spread's own `0.0000` never splits
    # the line — verified on the rendered text, not assumed.
    assert "measured spread is 0.0000" in line, line
    assert "supports the trigger on neither side" in line, line
    floor_sentence = next(s for s in re.split(r"\.\s+", line)
                          if f"{run_round.REOPEN_MIN_SCORED_ROWS} scored rows" in s)
    assert "measured spread is 0.0000" in floor_sentence, \
        f"the floor and the zero-spread guard are one sentence, not two rules: " \
        f"{floor_sentence!r}"
    # Clause 4: nothing the ruling already travelled with was traded away for the fix.
    assert "3 consecutive rounds" in line, line
    assert line.rstrip().endswith("budget opinion, not a measurement."), line


def test_a_sacrificed_runtime_task_that_never_scored_reads_as_no_scored_history(
        round_env, monkeypatch):
    """Clause 3: an unknown stays unknown. A task with trial rows but no SCORED ones is
    the interesting case, because it did run — it just never produced a number the
    condition could use.

    `bench_017`'s seeded rows are all null-scored and sit in a round LATER than
    `bench_016`'s, so a query that counted rows regardless of score would print that
    later id and look plausible. What it must print instead is `NO_SCORED_HISTORY` —
    never a delta of 0 (the null is not a zero, the same rule `trial_ledger_row` writes
    it for), never a round id, never `0 of 0`. The same node then calls the query
    directly for the two states no single round can stage: a box with no ledger at all,
    and a ledger that is present but unreadable, which are different unknowns and get
    different strings.
    """
    _history_ledger(round_env.paths.ledger_path, [
        ("bench_016", "R_20260929_082451", 0.6),
        ("bench_017", "R_20260929_120000", None),      # ran, never scored
        ("bench_017", "R_20260929_130000", None),
    ])
    _bench(round_env, _SCHEDULED_19)
    _quiet_proposer(monkeypatch, 3)
    loaded = {tid for tid, _r, _c in _SCHEDULED_19}
    _drive_round(monkeypatch, valid_ids=loaded - {"bench_s1"})
    result = asyncio.run(run_round.run(targets=["prompts"], budget_minutes=30))
    assert "error" not in result, result

    last = result["matrix_coverage"]["sacrificed_runtime_last_scored_round"]
    assert last["bench_016"] == "R_20260929_082451", last
    assert last["bench_017"] == run_round.NO_SCORED_HISTORY == \
        "no scored history on this box", last
    assert isinstance(last["bench_017"], str)
    assert not last["bench_017"].startswith("R_"), \
        "an unscored task was credited with a round it never scored in"
    assert last["bench_017"] not in ("0", "0.0", 0, None, ""), last

    report = Path(result["summary_file"]).read_text(encoding="utf-8")
    figures = [ln for ln in report.splitlines()
               if ln.startswith("- re-open figures are read from")][0]
    assert "bench_017: no scored history on this box" in figures, figures
    assert "0 of 0" not in figures and " bench_017: 0" not in figures, figures

    # The two states a round under test cannot stage, on the query itself.
    missing = run_round.last_scored_round_by_task(
        round_env.paths.ledger_path.parent / "nowhere.jsonl", ["bench_017"])
    assert missing == {"bench_017": run_round.NO_SCORED_HISTORY}, \
        "a box with no ledger has no scored history — that is the true answer"
    corrupt = round_env.paths.ledger_path.parent / "corrupt.jsonl"
    corrupt.write_bytes(b"\xff\xfe\x00 not json at all \xff")
    assert run_round.last_scored_round_by_task(corrupt, ["bench_017"]) == {
        "bench_017": run_round.LEDGER_UNREADABLE}
    assert run_round.LEDGER_UNREADABLE != run_round.NO_SCORED_HISTORY, \
        "not knowing and knowing there is nothing are different claims"
    assert run_round.last_scored_round_by_task(
        round_env.paths.ledger_path, []) == {}, "no victims, no entries"


def test_the_last_scored_round_entry_is_in_the_payload_on_a_round_that_did_not_shrink(
        round_env, monkeypatch):
    """Clause 2's other half: the key travels with the rest of `matrix_coverage`, so a
    reader guards one payload shape rather than learning which rounds are interesting.

    The fixture's two-task bench under the scheduled 30-minute budget does not shrink at
    all, which is the state the key has to survive: it is `{}` there, not absent and not
    `None`, and the full key set is the same one a shrunk round sends. The report still
    prints no coverage lines for a round that gave up nothing — an empty history list on
    that page would read as "these tasks have never run", which is precisely what the
    unknown string exists to stop.
    """
    _history_ledger(round_env.paths.ledger_path, [("bench_a1", "R_20260928_041936", 0.5)])
    _quiet_proposer(monkeypatch, 0)
    _drive_round(monkeypatch)
    result = asyncio.run(run_round.run(targets=["prompts"], budget_minutes=30))
    assert "error" not in result, result
    assert result["matrix_dropped_tasks"] == [], "the precondition: no shrink"

    cov = result["matrix_coverage"]
    assert cov["shrunk"] is False
    assert set(cov) == {"shrunk", "lint_valid_total", "lint_valid_started",
                        "runtime_planned", "runtime_started", "freed_direct_seconds",
                        "freed_serial_seconds", "freed_window_serial_only",
                        "reopen_condition", "reopen_ledger",
                        "sacrificed_runtime_last_scored_round"}
    assert cov["sacrificed_runtime_last_scored_round"] == {}
    assert cov["reopen_ledger"] == str(round_env.paths.ledger_path)

    report = Path(result["summary_file"]).read_text(encoding="utf-8")
    assert "- re-open figures are read from" not in report
    assert "- coverage given up by the shrink" not in report


# ── #1860: a stopped round's refusal has to be readable without parsing prose ──
#
# The node above pins the whole chain end to end on the one shape a real round produced.
# These two pin the vocabulary around it: the classes the deadline branch is on record
# wrapping, and the fact that for every OTHER refusal the new field and the prose prefix
# are the same string — which is what lets a consumer read one field and keep today's
# buckets. 21 of the 124 decision rows on the live ledger as of 2026-09-29 begin
# `deadline_stopped:` in prose while their predicate said something else.

#: The wrapped sentence, spelled the way `run_round`'s deadline arm spells it, so the
#: nodes below wrap a verdict as the writer does rather than inventing a shape. The
#: byte-identity of the real thing is pinned by
#: `test_a_stopped_round_reports_and_ledgers_what_it_measured_and_promotes_nothing`.
def _wrapped(predicate_reason: str) -> str:
    return ("deadline_stopped: 1 task(s) not reached, no promotion from a partial round "
            f"(predicate said: {predicate_reason})")


#: The three classes #1860 names, each with the prose `evaluate_promotion` really returns
#: for it: the two coverage refusals are `partial_{arm}_coverage ({scored} of {pool}
#: {label} tasks scored; unscored: {missing})` (promote.py:322-325, prose confirmed by
#: `test_partial_heldout_coverage_refuses_and_names_the_unscored_task` in
#: tests/test_autoresearch_promotion.py), and `safety_regression` is the bare string at
#: promote.py:291 — it names no task because exactly one bench task carries
#: `safety_critical: true` today (1 of 22 bench files, counted 2026-09-29).
WRAPPED_CLASSES = [
    ("partial_heldout_coverage",
     "partial_heldout_coverage (1 of 2 held-out tasks scored; unscored: bench_a2)"),
    ("partial_targeted_coverage",
     "partial_targeted_coverage (1 of 2 targeted tasks scored; unscored: bench_a1)"),
    ("safety_regression", "safety_regression"),
]


@pytest.mark.parametrize("want_class,predicate_reason", WRAPPED_CLASSES)
def test_a_stopped_row_states_the_predicate_class_not_the_deadline_wrap(
        want_class, predicate_reason):
    """Clause 1 for the classes the item names: the field is the predicate's head.

    `run_round` hands the row builder `predicate_refusal`, the verdict it held before the
    wrap, so the class survives the sentence. The prose keeps the wrap either way.
    """
    row = run_round.decision_ledger_row(
        "R_1", {"variant_id": "V_1", "should_promote": False,
                "reason": _wrapped(predicate_reason),
                "predicate_refusal": want_class}, None)
    assert row["refusal_class"] == want_class, row
    assert row["refusal_class"] != "deadline_stopped", \
        "the wrap names when the refusal happened, not what refused it"
    assert row["reason"] == _wrapped(predicate_reason), "the prose is not the field's place"


def test_an_unwrapped_refusal_states_the_head_of_its_own_reason():
    """Clause 2: with nothing wrapping it, the field IS the prose prefix.

    So a consumer reading one field gets the buckets the prose gives it today — these
    three rows count identically under either reading. The wrapped row at the end is the
    one shape where the two differ, which is the whole reason the field exists.
    """
    rows = {want: run_round.decision_ledger_row(
        "R_1", {"variant_id": "V_1", "should_promote": False, "reason": prose,
                "predicate_refusal": want}, None)
        for want, prose in WRAPPED_CLASSES}
    for want, row in rows.items():
        assert row["refusal_class"] == want, row
        assert promote.refusal_head(row["reason"]) == want, row["reason"]
    by_field = {k: sum(1 for r in rows.values() if r["refusal_class"] == k) for k in rows}
    by_prose = {k: sum(1 for r in rows.values()
                       if promote.refusal_head(r["reason"]) == k) for k in rows}
    assert by_field == by_prose == {"partial_heldout_coverage": 1,
                                    "partial_targeted_coverage": 1,
                                    "safety_regression": 1}, (by_field, by_prose)

    wrapped = run_round.decision_ledger_row(
        "R_1", {"variant_id": "V_1", "should_promote": False,
                "reason": _wrapped(WRAPPED_CLASSES[0][1]),
                "predicate_refusal": "partial_heldout_coverage"}, None)
    assert promote.refusal_head(wrapped["reason"]) == "deadline_stopped"
    assert wrapped["refusal_class"] == "partial_heldout_coverage", \
        "the two readings must differ on exactly the wrapped rows, and on nothing else"
