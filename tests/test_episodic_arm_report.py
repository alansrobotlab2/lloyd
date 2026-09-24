"""The committed episodic-arm report (#675): what it must carry, and that the
renderer and summariser produce it from records. The committed artifact is read
from `eval/episodic-arm/`; the synthetic half needs no corpus or daemon."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval.episodic_arm import METRICS, OUT_DIR, render_markdown, summarize

ROOT = Path(__file__).resolve().parents[1]
RESULT = OUT_DIR / "results.json"
REPORT = OUT_DIR / "results.md"


@pytest.fixture(scope="module")
def committed():
    return json.loads(RESULT.read_text())


# ── clause 4: committed, tracked, the six metrics at k = 0, 2, 5 ─────────────

def test_the_report_is_committed_outside_the_ignored_baselines_dir():
    assert OUT_DIR == ROOT / "eval" / "episodic-arm"
    for p in (RESULT, REPORT):
        assert p.is_file()
        ignored = subprocess.run(["git", "check-ignore", "-q", str(p)], cwd=ROOT)
        assert ignored.returncode == 1, f"{p} is gitignored"


def test_every_query_carries_six_metrics_per_k_beside_its_baseline(committed):
    assert committed["config"]["ks"] == [0, 2, 5]
    assert committed["records"]
    for r in committed["records"]:
        for k in ("0", "2", "5"):
            assert set(METRICS) <= set(r["arm"][k])
        assert set(METRICS) <= set(r["baseline"])
    q = committed["baseline"]["nightly_20260909_quoted"]
    assert q["entity_hit_rate"] == 0.50 and q["fact_entity_recall_avg"] == 0.375


def test_zero_gold_count_is_recorded_and_excluded(committed):
    s = committed["summary"]
    zero = [r for r in committed["records"] if r["zero_gold"]]
    assert s["zero_gold_count"] == len(zero)
    assert s["n_eligible"] == len(committed["records"]) - len(zero)
    assert s["arm_k0"]["n"] == s["n_eligible"]


def test_turn_kinds_are_recorded(committed):
    assert committed["config"]["include_kinds"]


# ── clause 5: every baseline entity miss is named and split ──────────────────

def test_every_baseline_miss_is_named_with_both_answers(committed):
    misses = {r["id"] for r in committed["records"] if not r["baseline"]["entity_hit"]}
    split = committed["summary"]["miss_split"]
    assert {q["id"] for q in split["queries"]} == misses
    md = REPORT.read_text()
    for q in split["queries"]:
        assert f"| {q['id']} |" in md
        assert {"arm_surfaced_any_k", "entity_in_corpus"} <= set(q)
    assert split["arm_answers_fraction"] is not None
    assert "Arm surfaced it at any k" in md


# ── the summariser and renderer on synthetic records ─────────────────────────

def _rec(qid, base_hit, arm_hit, zero=False, in_corpus=True):
    arm = {m: 0.0 for m in METRICS}
    arm.update(entity_hit=arm_hit, doc_hit=False, tokens_total=100, tokens_top3=30,
               turns_mean=1.0, clean={m: 0.0 for m in METRICS} | {"entity_hit": arm_hit})
    ctrl = {m: 0.0 for m in METRICS} | {"entity_hit": False, "doc_hit": False,
                                        "tokens_total": 50}
    base = {m: 0.0 for m in METRICS} | {"entity_hit": base_hit, "doc_hit": True,
                                        "fact_entity_recall": 0.0}
    return {"id": qid, "category": "fuzzy", "zero_gold": zero, "latency_ms": 100,
            "corpus_presence": {"entities": ["x"] if in_corpus else [], "docs": [],
                                "any": not zero},
            "arm": {k: dict(arm) for k in ("0", "2", "5")},
            "vault_doc_text_control": {k: dict(ctrl) for k in ("0", "2", "5")},
            "baseline": base}


def test_the_miss_fraction_counts_only_baseline_misses():
    recs = [_rec("a", False, True), _rec("b", False, False, in_corpus=False),
            _rec("c", True, False), _rec("z", False, False, zero=True, in_corpus=False)]
    args = SimpleNamespace(ks=[0, 2, 5], budget=1024, limit=10, models={}, qmd_env={},
                           corpus_source="t", baseline="/x/nightly.json")
    res = summarize(recs, {"ran_at": "t", "summary": {"overall": {}}}, args,
                    {"rows": []}, 3, ["2026-09-01"], ("user", "lloyd"))
    m = res["summary"]["miss_split"]
    assert m["n_miss"] == 3 and m["arm_answers"] == 1
    assert m["arm_answers_fraction"] == round(1 / 3, 3)
    assert m["arm_answers_clean_not_in_control"] == 1
    assert res["summary"]["zero_gold_count"] == 1
    assert res["summary"]["arm_k0"]["n"] == 3
    res["records"] = recs
    md = render_markdown(res)
    for qid in ("a", "b", "z"):
        assert f"| {qid} |" in md
