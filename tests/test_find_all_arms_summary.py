"""The #647 arm driver — its summary, and the one seam that needs a fake runner.

Two halves, deliberately unequal in cost. The pure half needs no engine: the spam
delta pairs an `all` reply with its own padded copy, and the judge-independence
count compares the rubric and P x R at one threshold. The seam half does reach
outside the module, so it is tested with a fake `run_bench_sdk` rather than a real
trial: `--max-turns` has to arrive as `run_bench_sdk`'s `max_agent_turns` and be
recorded on the row it produced (#1608 clause 5), and neither end of that is
visible from the summary alone.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("run_find_all_arms",
                                               ROOT / "eval" / "run_find_all_arms.py")
arms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(arms)


def _row(arm, trial, task, objective, rubric, composite, p=None, r=None):
    return {"arm": arm, "trial": trial, "task_id": task, "objective": objective,
            "rubric": rubric, "composite": composite, "precision": p, "recall": r,
            "f1": None, "turns": 3, "rubric_status": "ok"}


def test_spam_is_paired_with_its_own_reply_and_disagreement_is_counted():
    rows = [
        _row("all", 0, "t1", 1.0, 0.9, 0.95, 1.0, 1.0),
        _row("spam", 0, "t1", 0.5, 0.9, 0.70),
        _row("all", 1, "t1", 0.2, 0.8, 0.50, 1.0, 0.2),   # rubric says pass, P x R fail
        _row("spam", 1, "t1", 0.1, 0.8, 0.45),
        _row("one", 0, "t1", 0.1, 0.3, 0.20, 1.0, 0.1),
    ]
    s = arms.summarise(rows)
    assert s["spam"]["n"] == 2
    assert s["spam"]["objective_drop"] == 0.3
    assert s["spam"]["composite_lower"] == 2
    assert s["spam"]["rubric_not_lower"] == 2
    ji = s["judge_independence"]
    assert ji["n"] == 3 and ji["disagree_at_0.5"] == 1
    assert s["cells"]["one/t1"]["recall"] == 0.1


def test_pearson_is_none_when_undefined():
    assert arms._pearson([1.0, 1.0, 1.0], [0.1, 0.5, 0.9]) is None
    assert arms._pearson([0.0, 1.0], [0.0, 1.0]) is None
    assert arms._pearson([0.0, 0.5, 1.0], [0.0, 0.5, 1.0]) == 1.0


# ── the per-task turn budget reaches the runner (#1608 clause 5) ─────────────
#
# `run_bench_sdk` is a separate harness path: it owns the agent loop and its turn
# cap, and until now the driver never named that cap, so every arm silently ran
# under `DEFAULT_MAX_AGENT_TURNS` whatever the arm needed. That is a process seam
# the summary tests above cannot see, so the test below crosses it with a fake
# runner and reads both ends: the kwargs `run_bench_sdk` was handed, and the row
# that landed in `trials.jsonl`. The CLI string is the input, not an args object —
# flag name, type and default are part of what is pinned.

_SCORE = {"composite_score": 0.5, "objective_score": 0.5, "rubric_overall": 0.5,
          "rubric_status": "ok", "precision": 1.0, "recall": 1.0, "f1": 1.0,
          "answer_sets": []}


async def _no_sandbox():
    return None


def _install_fake_harness(monkeypatch, tmp_path):
    """Replace the six things `main_async` reaches outside itself; return the calls."""
    calls = []

    def load_config():
        return SimpleNamespace(default_model="fake-model",
                               paths=SimpleNamespace(bench_dir=tmp_path / "bench"))

    def load_bench_tasks(_bench_dir):
        # The `one` arm rewrites a sentence out of the real prompt and asserts it is
        # there, so the fakes have to carry those sentences.
        tasks = [{"id": tid, "prompt": f"audit {tid}: report each finding once"}
                 for tid in arms.TASKS]
        for tid, (sentence, _rewrite) in arms.ONE_PHRASING.items():
            next(t for t in tasks if t["id"] == tid)["prompt"] = sentence
        return tasks

    async def run_bench_sdk(cfg, variants, tasks, model, **kwargs):
        calls.append({"model": model, "variants": variants,
                      "task_ids": [t["id"] for t in tasks], **kwargs})
        return [{"task_id": t["id"], "session_id": "bench_fake", "status": "completed",
                 "turns": 5, "tool_calls": [], "denied_calls": [],
                 "bench_probe_count": 0, "duration_seconds": 1.0,
                 "final_text": "- one finding"} for t in tasks]

    monkeypatch.setattr(arms, "load_config", load_config)
    monkeypatch.setattr(arms, "load_bench_tasks", load_bench_tasks)
    monkeypatch.setattr(arms, "require_tool_sandbox", _no_sandbox)
    monkeypatch.setattr(arms, "materialize_baseline", lambda cfg: ("BASELINE_V", tmp_path))
    monkeypatch.setattr(arms, "run_bench_sdk", run_bench_sdk)
    monkeypatch.setattr(arms, "judge_trace", lambda task, trace, rubric_model=None: dict(_SCORE))
    return calls


def test_max_turns_reaches_run_bench_sdk_and_every_trial_row(monkeypatch, tmp_path):
    """Clause 5: `--max-turns` becomes `max_agent_turns`, and the row says so."""
    calls = _install_fake_harness(monkeypatch, tmp_path)
    out = tmp_path / "arms"
    # `arms.main` owns the event loop (`asyncio.run(main_async(...))`), so this is a
    # plain synchronous call from the test.
    rc = arms.main(["--out", str(out), "--trials", "1", "--parallel", "1", "--max-turns", "7"])
    assert rc == 0
    # Both arms run: `all` over the four tasks, `one` over the two phrased ones.
    assert len(calls) == 2, [c["task_ids"] for c in calls]
    assert {c["max_agent_turns"] for c in calls} == {7}, "the cap must reach every arm"
    rows = [json.loads(line) for line in (out / "trials.jsonl").open()]
    # 4 `all` + 4 `spam` re-scores of those same replies + 2 `one`. The spam rows
    # carry the budget too: it is the cap that governed the reply they re-score.
    assert len(rows) == 10
    assert {r["turn_budget"] for r in rows} == {7}
    # Beside `turns`, which is what makes a 12-turn reply distinguishable from one
    # the cap cut off.
    assert all("turns" in r and r["turn_budget"] == 7 for r in rows)
    assert {r["arm"] for r in rows} == {"all", "one", "spam"}


def test_the_default_budget_is_the_runners_own(monkeypatch, tmp_path):
    """The default is the runner's constant, imported not copied — and readable alone.

    `--summarise` costs nothing to run and a trial does not, so the default has to
    be inspectable without booting the harness.
    """
    args = arms.build_parser().parse_args(["--out", str(tmp_path)])
    assert args.max_turns == arms.DEFAULT_MAX_AGENT_TURNS == 12
    assert arms.build_parser().parse_args(
        ["--out", str(tmp_path), "--max-turns", "3"]).max_turns == 3
    # A run that passes no --max-turns still records the cap it ran under, so the
    # budget never goes back to being an accident of the harness version.
    calls = _install_fake_harness(monkeypatch, tmp_path)
    arms.main(["--out", str(tmp_path / "d"), "--trials", "1", "--parallel", "1"])
    assert {c["max_agent_turns"] for c in calls} == {arms.DEFAULT_MAX_AGENT_TURNS}
    rows = [json.loads(line) for line in (tmp_path / "d" / "trials.jsonl").open()]
    assert {r["turn_budget"] for r in rows} == {arms.DEFAULT_MAX_AGENT_TURNS}
