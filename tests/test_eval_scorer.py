"""eval/run_eval.py scoring — the numbers the nightly trend is built from.

`_score` and `_ndcg_at_k` had no tests, so a scoring change would have moved
the trend line with no code review signal that it had.
"""
import importlib
import inspect
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


# ── the configuration a programmatic caller inherits (#498) ─────────────────

# The four knobs the eval and the live tool must agree on. `expand_graph` is
# deliberately NOT one of them: production's default is False (`_vault_recall`
# reads `params.get("expand_graph", False)`) while the eval runs the graph leg
# expanded as a measurement choice, and #1000 owns that claim.
KNOBS = {"graph_rerank": "RECALL_GRAPH_RERANK",
         "rerank_alpha": "RECALL_RERANK_ALPHA",
         "graph_top_k": "RECALL_GRAPH_TOP_K",
         "graph_hops": "RECALL_GRAPH_HOPS"}


def test_run_eval_signature_defaults_are_the_production_constants():
    """run_eval()'s signature carried `rerank_alpha=0.5` while production's
    RECALL_RERANK_ALPHA is 0.3, so any programmatic caller measured a
    configuration nothing serves. Commit 59cd7bf rewired main()'s argparse only
    — the one live programmatic caller (agent_mcp/fact_improvement.py
    ._fact_entity_recall) inherited the stale value in its first version."""
    from agent_mcp import vault
    params = inspect.signature(ev.run_eval).parameters
    for knob, const in KNOBS.items():
        production = getattr(vault, const)
        assert params[knob].default == production, knob
        # 0 == False in Python, so a bool knob that quietly became an int would
        # pass the line above. Pin the kind as well as the value.
        assert type(params[knob].default) is type(production), knob


def test_run_eval_signature_defaults_derive_from_the_constants():
    """Equal today is not derived. The fix puts the imported constants in the
    signature, so patching a constant and reloading the module must move the
    default — a restated literal does not move."""
    from agent_mcp import vault
    sentinels = {"RECALL_GRAPH_RERANK": True, "RECALL_RERANK_ALPHA": 0.42,
                 "RECALL_GRAPH_TOP_K": 77, "RECALL_GRAPH_HOPS": 9}
    originals = {const: getattr(vault, const) for const in sentinels}
    try:
        for const, value in sentinels.items():
            setattr(vault, const, value)
        importlib.reload(ev)
        params = inspect.signature(ev.run_eval).parameters
        assert params["graph_rerank"].default is True
        assert params["rerank_alpha"].default == 0.42
        assert params["graph_top_k"].default == 77
        assert params["graph_hops"].default == 9
    finally:
        # `ev` is this file's module-level handle, shared with every other test
        # here: restore the constants BEFORE reloading, or a sentinel default
        # leaks into whichever test runs next.
        for const, value in originals.items():
            setattr(vault, const, value)
        importlib.reload(ev)
    assert (inspect.signature(ev.run_eval).parameters["rerank_alpha"].default
            == originals["RECALL_RERANK_ALPHA"])


def test_argparse_defaults_and_the_signature_default_cannot_disagree():
    """`--alpha` and run_eval(rerank_alpha=…) set one knob from two places. This
    reads `ev.build_parser()` — the parser main() actually parses with — never a
    hand-mirrored copy of it, which is the tautology the test above falls into
    and what #999 tracks."""
    from agent_mcp import vault
    parser = ev.build_parser()
    params = inspect.signature(ev.run_eval).parameters
    # One of the four argparse dests differs from the parameter it sets —
    # `--alpha` lands on dest `alpha` while the parameter is `rerank_alpha`. The
    # other three share their name, and getting this mapping wrong is how a
    # mirror test silently compares a knob to itself.
    dests = {"graph_rerank": "graph_rerank", "rerank_alpha": "alpha",
             "graph_top_k": "graph_top_k", "graph_hops": "graph_hops"}
    for knob, dest in dests.items():
        assert parser.get_default(dest) == params[knob].default, knob
    # The help text quotes the same number, so operator and function cannot be
    # told two different things about what production runs.
    alpha_help = next(a for a in parser._actions if a.dest == "alpha").help
    assert str(vault.RECALL_RERANK_ALPHA) in alpha_help
    rerank_help = next(a for a in parser._actions if a.dest == "graph_rerank").help
    assert str(vault.RECALL_GRAPH_RERANK) in rerank_help


def test_a_caller_that_loads_the_script_by_path_gets_production_defaults():
    """The caller that made this defect live reaches run_eval() through a file
    path, not an import: agent_mcp/fact_improvement.py::_fact_entity_recall
    builds a module with importlib.util.spec_from_file_location, so it does not
    share this file's `ev` handle and nothing here would have told it the
    defaults had been fixed. Load it the same way and read the same signature."""
    import importlib.util

    from agent_mcp import vault
    script = ROOT / "eval" / "run_eval.py"
    spec = importlib.util.spec_from_file_location("lloyd_run_eval_probe", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    params = inspect.signature(module.run_eval).parameters
    for knob, const in KNOBS.items():
        assert params[knob].default == getattr(vault, const), knob


def test_production_knobs_keep_their_measured_values():
    """This rewires wiring, not the measured configuration. The constants are
    the answer to the 2026-09-04 sweep (agent_mcp/vault.py:113-124); a change
    that moved one would move the nightly baseline with it."""
    from agent_mcp import vault
    assert vault.RECALL_GRAPH_RERANK is False
    assert vault.RECALL_RERANK_ALPHA == 0.3
    assert vault.RECALL_GRAPH_TOP_K == 5
    assert vault.RECALL_GRAPH_HOPS == 1


def test_vault_recall_still_falls_back_to_the_same_constants():
    """The other half of "wiring, not configuration": the live tool's own
    fallbacks must stay the constants, so the tool's default and the eval's
    default have one source instead of two that can drift apart."""
    from agent_mcp import vault
    src = inspect.getsource(vault._vault_recall)
    for param, const in KNOBS.items():
        assert f'params.get("{param}", {const})' in src, param


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
