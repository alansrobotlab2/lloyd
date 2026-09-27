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
