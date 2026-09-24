"""The #647 arm summary reads its rows the way the report quotes them.

`eval/run_find_all_arms.py` drives trials on the engine, so only its pure half
is tested: the spam delta pairs an `all` reply with its own padded copy, and the
judge-independence count compares the rubric and P x R at one threshold.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

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
