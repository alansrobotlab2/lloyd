"""#557: the skill-matcher recall eval, ported without its embedding arm.

The labelled set and scorer came from branch automod/SM_20260911_055006
(0c5f61c). Alan's 2026-09-14 ruling kept them and dropped the geometric arm,
so these tests pin three things: the metric definitions (a hand-computed
fixture, so a change to what recall@5 means fails here), the corpus contract
#711 filters, and the arm's absence.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "eval" / "run_skill_match_eval.py"


def _load():
    spec = importlib.util.spec_from_file_location("run_skill_match_eval", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FIXTURE = [
    {"id": "f1", "turn": "t1", "expected_skills": ["a", "b"]},
    {"id": "f2", "turn": "t2", "expected_skills": ["c"]},
    {"id": "f3", "turn": "t3", "expected_skills": ["d"]},
    {"id": "f4", "turn": "t4", "expected_skills": [], "expected_empty": True},
    {"id": "f5", "turn": "t5", "expected_skills": [], "expected_empty": True},
]
RETURNED = {
    "t1": ["a", "x", "y"],                 # 1 of 2 expected
    "t2": [],                              # nothing at all: a no-match
    "t3": ["d", "e", "f", "g", "h", "i"],  # 6 returned; only the top 5 count
    "t4": ["z"],                           # names a skill on an empty turn
    "t5": [],
}


def test_metrics_on_a_hand_computed_fixture():
    ev = _load()
    out = ev.score(FIXTURE, lambda turn: RETURNED[turn])
    assert out["labelled_turns"] == 3
    assert out["expected_empty_turns"] == 2
    assert out["recall@5"] == pytest.approx(0.5)          # (0.5 + 0 + 1) / 3
    assert out["precision@5"] == pytest.approx(0.1333)    # (1/5 + 0 + 1/5) / 3
    assert out["hit@5"] == pytest.approx(0.6667)          # f1, f3
    assert out["no_match_rate"] == pytest.approx(0.3333)  # f2
    assert out["no_match_count"] == 1
    assert out["expected_empty_named_rate"] == pytest.approx(0.5)
    assert out["expected_empty_named_count"] == 1
    assert [t["returned"] for t in out["turns"]][2] == ["d", "e", "f", "g", "h"]


def test_an_empty_population_reports_none_not_zero():
    ev = _load()
    out = ev.score([FIXTURE[3]], lambda turn: [])
    assert out["recall@5"] is None and out["no_match_rate"] is None
    assert out["expected_empty_named_rate"] == 0.0


def test_the_labelled_set_meets_its_contract():
    ev = _load()
    records = ev.load_records()
    assert len(records) >= 40
    assert all(r.get("source") for r in records), "every record carries provenance"
    assert sum(1 for r in records if r.get("expected_empty")) >= 10
    assert len({r["id"] for r in records}) == len(records)


def test_a_stale_label_is_named():
    ev = _load()
    bad = ev.check_labels(FIXTURE, {"a", "b", "c"})
    assert bad == ["f3: expected skill 'd' is not an active skill"]


def test_the_baseline_lives_under_the_data_root():
    from app.paths import EVAL_BASELINES_DIR
    ev = _load()
    assert ev.baseline_path() == EVAL_BASELINES_DIR / "skill_match_baseline.json"
    rep = ev.build_report(FIXTURE, ev.score(FIXTURE, lambda t: RETURNED[t]))
    assert rep["matches_production_defaults"] is True
    assert {"graph_rerank", "rerank_alpha", "graph_top_k", "graph_hops"} <= set(rep)
    assert set(rep["arms"]) == {"lexical"}


def test_the_embedding_arm_did_not_land():
    assert not (ROOT / "agent_mcp" / "skill_cards.py").exists()
    assert not (ROOT / "scripts" / "setup_skill_cards.py").exists()
    imported = set()
    for node in ast.walk(ast.parse(SCRIPT.read_text())):
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported |= {f"{node.module}.{a.name}" for a in node.names}
    assert not [m for m in imported if "skill_cards" in m]


def test_the_scorer_imports_the_production_matcher(monkeypatch):
    """The lexical arm is prefetch's own function, not a copy of it."""
    import prefetch
    ev = _load()
    seen = []
    monkeypatch.setattr(prefetch, "_search_skills",
                        lambda toks: seen.append(toks) or [(1.0, {"name": "k"})])
    assert ev.lexical_top5("restart the backend please") == ["k"]
    assert seen and isinstance(seen[0], set)
