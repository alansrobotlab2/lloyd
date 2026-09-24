"""The #698 scalar-vs-binary judge comparison driver: its arithmetic, offline.

`eval/autoresearch_judge_compare.py` decides whether the binary judge becomes
the default, so the three numbers it reads that decision off — AUC, the
pairwise flip rate and the within-trace SD — are pinned here on inputs whose
answers are known. No engine is touched: `report` runs over files written here.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def cmp(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "eval"))
    spec = importlib.util.spec_from_file_location(
        "autoresearch_judge_compare", ROOT / "eval" / "autoresearch_judge_compare.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["autoresearch_judge_compare"] = mod
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path)
    monkeypatch.setattr(mod, "TRACES", tmp_path / "traces.jsonl")
    monkeypatch.setattr(mod, "SCORES", tmp_path / "scores.jsonl")
    monkeypatch.setattr(mod, "RESULT", tmp_path / "result.json")
    monkeypatch.setattr(mod, "REFERENCE", tmp_path / "labels.yaml")
    return mod


def test_the_reference_leg_scores_each_judge_against_hand_labels(cmp):
    """Two tasks, one pass and one fail label per condition pair: `binary`
    tracks the labels exactly, `scalar` is inverted on one task — its pooled
    AUC must come out below binary's and the within-task AUCs must say where."""
    labels = {"bench_a": [1, 1, 1, 0, 0, 0], "bench_b": [1, 1, 1, 0, 0, 0]}
    cmp.REFERENCE.write_text("\n".join(f"{k}: {v}" for k, v in labels.items()),
                             encoding="utf-8")
    rows = []
    for t in labels:
        for (c, d), lab in zip(cmp._ORDER, labels[t]):
            rows.append({"task_id": t, "condition": c, "draw": d, "judge": "binary",
                         "k": 0, "score": float(lab)})
            s = float(lab) if t == "bench_a" else 1.0 - lab
            rows.append({"task_id": t, "condition": c, "draw": d, "judge": "scalar",
                         "k": 0, "score": s})
    leg = cmp._reference_leg(cmp._reference_labels(), rows)
    assert leg["mixed_tasks"] == ["bench_a", "bench_b"]
    assert leg["judges"]["binary"]["auc_pooled_single"] == 1.0
    assert leg["judges"]["scalar"]["auc_within_task"] == {"bench_a": 1.0, "bench_b": 0.0}
    assert leg["auc_pooled_single_binary_minus_scalar"]["diff"] == pytest.approx(0.5)


def test_auc_is_the_probability_a_good_pair_is_ordered_right(cmp):
    assert cmp._auc([1.0, 0.9], [0.1, 0.2]) == 1.0
    assert cmp._auc([0.1], [0.9]) == 0.0
    assert cmp._auc([0.5], [0.5]) == 0.5              # a tie is half a win
    assert cmp._auc([], [0.1]) is None


def test_flip_rate_counts_draw_pairs_that_disagree_on_an_ordering(cmp):
    # Two traces, three draws: A>B, A>B, A<B → draw pairs (1,2) agree, (1,3)
    # and (2,3) flip → 2/3.
    assert cmp._flip_rate({"a": [0.9, 0.8, 0.1], "b": [0.5, 0.5, 0.5]}) == pytest.approx(2 / 3)
    assert cmp._flip_rate({"a": [1.0, 1.0], "b": [0.0, 0.0]}) == 0.0
    assert cmp._flip_rate({"a": [1.0, 1.0]}) is None


def test_report_separates_a_stable_discriminating_judge_from_a_noisy_one(cmp):
    """A synthetic corpus where `binary` is perfectly stable and separates good
    from bad, and `scalar` is noisy and does not: the report must say so on
    every axis the switch decision reads."""
    traces, scores = [], []
    for t in ("bench_a", "bench_b", "bench_c"):
        for cond in ("good", "bad"):
            for d in range(2):
                traces.append({"task_id": t, "condition": cond, "draw": d,
                               "final_text": "x", "source": "model"})
                for k in range(3):
                    b = 1.0 if cond == "good" else 0.0
                    s = [0.2, 0.8, 0.5][k]
                    scores.append({"task_id": t, "condition": cond, "draw": d,
                                   "judge": "binary", "k": k, "score": b,
                                   "per": {"a1": cond == "good"}, "evidence_found": [True]})
                    scores.append({"task_id": t, "condition": cond, "draw": d,
                                   "judge": "scalar", "k": k, "score": s,
                                   "per": {"clarity": s}, "evidence_found": None})
    scores.append({"task_id": "bench_a", "condition": "good", "draw": 0,
                   "judge": "scalar", "k": 9, "score": None, "per": {}, "error": "rubric_no_json"})
    cmp.TRACES.write_text("".join(json.dumps(r) + "\n" for r in traces), encoding="utf-8")
    cmp.SCORES.write_text("".join(json.dumps(r) + "\n" for r in scores), encoding="utf-8")
    out = cmp.report()
    b, s = out["judges"]["binary"], out["judges"]["scalar"]
    assert b["auc_single_pooled"] == 1.0 and s["auc_single_pooled"] == 0.5
    assert b["within_trace_sd_mean"] == 0.0 and s["within_trace_sd_mean"] > 0.2
    assert b["tasks_agree_ge_0_9"] == 3 and s["tasks_agree_ge_0_9"] == 0
    assert s["failed"] == 1 and b["failed"] == 0
    assert out["paired_binary_minus_scalar"]["auc_single"]["diff"] == pytest.approx(0.5)
    assert json.loads(cmp.RESULT.read_text())["n_traces"] == 12
