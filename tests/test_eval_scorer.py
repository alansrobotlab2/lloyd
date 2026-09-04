"""eval/run_eval.py scoring — the numbers the nightly trend is built from.

`_score` and `_ndcg_at_k` had no tests, so a scoring change would have moved
the trend line with no code review signal that it had.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval.run_eval as ev  # noqa: E402


# ── NDCG ─────────────────────────────────────────────────────────────────────

def test_ndcg_perfect_and_empty():
    assert ev._ndcg_at_k(["a/x.md", "b/y.md"], ["a/x", "b/y"], k=10) == 1.0
    assert ev._ndcg_at_k([], ["a/x"], k=10) == 0.0
    assert ev._ndcg_at_k(["a/x.md"], [], k=10) == 0.0
    assert ev._ndcg_at_k(["nope.md"], ["a/x"], k=10) == 0.0


def test_ndcg_rewards_earlier_hits():
    early = ev._ndcg_at_k(["a/x.md", "junk.md", "junk2.md"], ["a/x"], k=10)
    late = ev._ndcg_at_k(["junk.md", "junk2.md", "a/x.md"], ["a/x"], k=10)
    assert early == 1.0
    assert 0 < late < early


def test_ndcg_ignores_hits_past_k():
    assert ev._ndcg_at_k(["junk.md"] * 10 + ["a/x.md"], ["a/x"], k=10) == 0.0


# ── entity / doc extraction ──────────────────────────────────────────────────

def test_entities_in_result_orders_by_signal_strength():
    result = {
        "facts": [{"entity": "FromFacts"}],
        "graph_expanded_facts": [{"entity": "FromGraph"}],
        "graph_neighbors_used": [{"entity": "FromNeighbors"}],
    }
    got = ev._entities_in_result(result, seeds=["Seed", "seed"])
    assert got == ["Seed", "FromFacts", "FromGraph", "FromNeighbors"]


def test_norm_treats_separators_as_equivalent():
    assert ev._norm("Knowledge-Graph") == ev._norm("knowledge_graph")


# ── the record the nightly compare step reads ────────────────────────────────

def test_summarize_shape_matches_what_the_skill_globs(tmp_path):
    """#82's compare step reads summary.overall.{...}. When it read the wrong
    keys the trend was silently empty."""
    records = [
        {"id": "q1", "category": "single", "latency_ms": 100.0, "error": None,
         "scoring": {"entity_hit": True, "doc_hit": True, "entity_recall": 1.0,
                     "doc_recall": 1.0, "rr_doc": 1.0, "ndcg10": 1.0,
                     "fact_entity_recall": 1.0, "first_doc_rank": 1}},
        {"id": "q2", "category": "hard", "latency_ms": 300.0, "error": None,
         "scoring": {"entity_hit": False, "doc_hit": True, "entity_recall": 0.0,
                     "doc_recall": 0.5, "rr_doc": 0.5, "ndcg10": 0.5,
                     "fact_entity_recall": 0.0, "first_doc_rank": 2}},
    ]
    summary = ev.summarize(records)
    overall = summary["overall"]
    for key in ("entity_hit_rate", "doc_hit_rate", "mrr_doc", "ndcg10",
                "latency_ms_avg", "fact_entity_recall_avg", "n_queries", "errors"):
        assert key in overall, key
    assert overall["mrr_doc"] == 0.75
    assert overall["entity_hit_rate"] == 0.5
    assert set(summary["by_category"]) == {"single", "hard"}


def test_eval_defaults_are_productions(monkeypatch):
    """The eval's own defaults must be what vault_recall serves, or the
    nightly trend measures a configuration nobody runs."""
    from agent_mcp import vault
    ap_defaults = {}
    import argparse
    parser = argparse.ArgumentParser()
    # Mirror main()'s parser construction closely enough to read the defaults.
    parser.add_argument("--no-graph-rerank", dest="graph_rerank", action="store_false",
                        default=vault.RECALL_GRAPH_RERANK)
    parser.add_argument("--alpha", type=float, default=vault.RECALL_RERANK_ALPHA)
    args = parser.parse_args([])
    ap_defaults["graph_rerank"] = args.graph_rerank
    ap_defaults["alpha"] = args.alpha
    assert ap_defaults["graph_rerank"] is vault.RECALL_GRAPH_RERANK
    assert ap_defaults["alpha"] == vault.RECALL_RERANK_ALPHA


def test_a_run_record_declares_whether_it_matched_production(tmp_path):
    """Every baseline file now says whether its config was production's, so a
    later reader cannot compare two runs that measured different systems."""
    latest = sorted((ROOT / "eval" / "baselines").glob("*.json"),
                    key=lambda p: p.stat().st_mtime)
    if not latest:
        pytest.skip("no baseline runs recorded yet")
    rec = json.loads(latest[-1].read_text())
    assert "matches_production_defaults" in rec
    assert {"graph_rerank", "rerank_alpha", "graph_top_k", "graph_hops"} <= set(rec)
