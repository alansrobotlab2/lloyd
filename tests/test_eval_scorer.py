"""eval/run_eval.py scoring — the numbers the nightly trend is built from.

`_score` and `_ndcg_at_k` had no tests, so a scoring change would have moved
the trend line with no code review signal that it had.
"""
import ast
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
         "graph_hops": "RECALL_GRAPH_HOPS",
         # Joined by #843: production seeded 10 query entities, the eval scored
         # 5, and nothing in this file's mirror could see it because the seed
         # count was not a knob at all.
         "seed_top_k": "RECALL_SEED_TOP_K"}


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
                 "RECALL_GRAPH_TOP_K": 77, "RECALL_GRAPH_HOPS": 9,
                 # #843: without this row the seed default is pinned for value
                 # only, and a restated literal `10` in the signature passes.
                 "RECALL_SEED_TOP_K": 33}
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
        assert params["seed_top_k"].default == 33, (
            "run_eval's seed default did not move with the constant, so it is a "
            "restated number and not the shared name")
        # The eval's default seeding width must also be what a bare
        # `_vault_recall` serves with the same constant moved, or the eval is
        # measuring a width production no longer uses.
        assert vault.RECALL_SEED_TOP_K == 33
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
             "graph_top_k": "graph_top_k", "graph_hops": "graph_hops",
             "seed_top_k": "seed_top_k"}
    for knob, dest in dests.items():
        assert parser.get_default(dest) == params[knob].default, knob
    seed_help = next(a for a in parser._actions if a.dest == "seed_top_k").help
    assert str(ev.RECALL_SEED_TOP_K) in seed_help, (
        "the operator-facing help and the code default are allowed to name "
        "different seed widths, which is the same misreport in prose")
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
    default have one source instead of two that can drift apart.

    Every knob here but one is read out of `params`, because `params` is the
    `vault_recall` wire shape and those names are in its schema. The seed width is
    the exception, and deliberately so (#843 review): it is keyword-only and is
    resolved against `RECALL_SEED_TOP_K` INSIDE the body, so no client key can
    move it and no definition-time binding can freeze the number. `params.get`
    for this one would be the defect, not the fix."""
    from agent_mcp import vault
    src = inspect.getsource(vault._vault_recall)
    for param, const in KNOBS.items():
        if param == "seed_top_k":
            assert f'params.get("{param}"' not in src, (
                "the seed width became readable from the client's argument dict, "
                "which makes it an undocumented vault_recall parameter")
            assert f"RECALL_SEED_TOP_K if {param} is None" in src, (
                "the seed width stopped resolving against the constant at call "
                "time, so a moved constant would not move production")
            continue
        assert f'params.get("{param}", {const})' in src, param


def _entity_slice_bounds(src: str) -> list[str]:
    """The bound of every slice applied to an `extract_entities_from_query`
    result, as written — via the AST, so a comment mentioning `[:5]` cannot
    trip it and a literal seed count cannot hide in it.

    A slice bound by an int is the bug #843 is about; a bound that is a name is
    the knob's value."""
    tree = ast.parse(src)
    bounds = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript) or not isinstance(node.slice, ast.Slice):
            continue
        called = {n.func.id for n in ast.walk(node.value)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        if not any(name.endswith("extract_entities_from_query") for name in called):
            continue
        upper = ast.get_source_segment(src, node.slice.upper) if node.slice.upper else None
        if upper is None:
            bounds.append("<unbounded>")
        elif upper.lstrip("-").isdigit():
            bounds.append(f"LITERAL {upper}")
        else:
            bounds.append(upper)
    return bounds


EIGHT_ENTITIES = [(f"Ent {i}", 1.0) for i in range(8)]


PERTURBATION_Q8 = {
    # An entity-axis swap whose extractor output is identical in both arms — the
    # shape `label_failures` reads. A variant arm cut narrower than the reference
    # arm reports `seed_moved: True`, i.e. "the swap moved the seed set", which is
    # a truncation artifact and not a result.
    "id": "q8", "axis_changed": "entity", "old_value": "Ent 1", "new_value": "Ent 9",
    "perturbed_query": "eight entity query but about a twin",
    "expected_to_move": ["Ent 9"], "expected_pinned": ["Ent 2"],
}

_Q8_SPEC = {"id": "q8", "query": "eight entity query", "category": "adversarial",
            "expect_entities": ["Ent 7"], "expect_paths": []}


def _recall_stub(calls: list):
    """A `_vault_recall` that records every call as `(params, kwargs)`.

    Two properties are being asserted here rather than a returned value: the seed
    width must reach retrieval and not only the record, and it must reach it as a
    KEYWORD argument. `params` is the `vault_recall` wire shape — `call_tool`
    hands that dict to the same function — so a `seed_top_k` key showing up in it
    would be an undocumented tool parameter (#843 review). Calls are recorded in
    order so a two-arm test can name the reference arm and the variant arm."""
    def fake(params, **kwargs):
        calls.append((dict(params), dict(kwargs)))
        return {"documents": [{"path": "knowledge/a.md"}], "facts": [],
                "graph_expanded_facts": [], "graph_neighbors_used": []}
    return fake


def test_the_record_seeds_at_the_shared_constant(monkeypatch):
    """#843 clause 2. The record's seeds came from a slice of five while
    `_vault_recall` seeded the same extractor at ten, so `entity_hit` — a union
    with these seeds first and largest — called a miss what production served as
    a hit, on 13 of the 20 bench queries. A query that extracts eight entities
    must now yield eight seeds in the record; under the five-wide slice it
    yielded five."""
    calls: list = []
    monkeypatch.setattr(ev, "_extract_entities_from_query", lambda q: list(EIGHT_ENTITIES))
    monkeypatch.setattr(ev, "_vault_recall", _recall_stub(calls))
    recs = ev.run_eval([_Q8_SPEC], limit=5, counterfactual=False)
    assert len(recs[0]["seeds_extracted"]) == 8
    params, kwargs = calls[0]
    assert kwargs.get("seed_top_k") == ev.RECALL_SEED_TOP_K, (
        "the record was scored at a width retrieval did not use")
    assert "seed_top_k" not in params, (
        "the width reached retrieval as a key of the params dict, which IS the "
        "vault_recall wire shape: the tool schema declares no such parameter")
    bounds = _entity_slice_bounds(inspect.getsource(ev.run_eval))
    assert bounds == ["seed_top_k"], (
        f"run_eval()'s seed slice is no longer bounded by the knob: {bounds}")


def test_a_narrower_knob_moves_the_record_and_the_recall_call_together(monkeypatch):
    """The derivation half of clause 2. Eight seeds out of an eight-entity
    fixture is also what a restated literal `[:8]` would produce; equality with a
    count is not derivation. An explicit narrower width must move BOTH the
    record's slice and the width handed to retrieval, together — a record naming a
    width retrieval did not use is #843 moved one line deeper."""
    calls: list = []
    monkeypatch.setattr(ev, "_extract_entities_from_query", lambda q: list(EIGHT_ENTITIES))
    monkeypatch.setattr(ev, "_vault_recall", _recall_stub(calls))
    recs = ev.run_eval([_Q8_SPEC], limit=5, counterfactual=False, seed_top_k=6)
    assert len(recs[0]["seeds_extracted"]) == 6
    assert calls[0][1].get("seed_top_k") == 6, (
        "the record was sliced at 6 while retrieval seeded at the default, so the "
        "seeds written into the artifact are not the seeds recall used")


@pytest.mark.parametrize("width", [None, 3],
                         ids=["production width", "a width that is not production's"])
def test_the_counterfactual_variant_seeds_at_the_same_width(monkeypatch, width):
    """#843 clause 2, the second seed site - reached through `run_eval`, not by
    calling `_attach_counterfactual` with a hand-built dict. The first round's
    review caught that: driving the helper directly asserted a value the test
    itself supplied, and left the plumbing that carries the width from `run_eval`
    to the variant arm with no test on it at all.

    `_attach_counterfactual` sliced the perturbed query's entities at five too, so
    the perturbation half of every artifact was measured on the wrong seed set even
    if only the record's slice had been fixed - and `seed_moved`, the label #537
    rests on, compared two different truncations rather than two queries.

    Run at two widths because the second one is the only arm that can see the
    plumbing. `run_eval`'s parameter and the helper's parameter both default to
    `RECALL_SEED_TOP_K`, so a call site that forgets to forward the width still
    leaves the two arms equal AT THE DEFAULT and every assertion passes on dead
    plumbing. Asked for 3, the variant arm has to answer 3. Mutation-checked:
    dropping `seed_top_k=seed_top_k` from the call site fails this case alone."""
    from eval import counterfactual as cf

    expected = ev.RECALL_SEED_TOP_K if width is None else width
    if width is not None:
        assert expected != ev.RECALL_SEED_TOP_K, (
            "the non-default arm would be indistinguishable from the default one")

    calls: list = []
    monkeypatch.setattr(ev, "_extract_entities_from_query", lambda q: list(EIGHT_ENTITIES))
    monkeypatch.setattr(ev, "_vault_recall", _recall_stub(calls))
    monkeypatch.setattr(cf, "load_records", lambda *a, **k: {"q8": PERTURBATION_Q8})
    ev.run_eval([_Q8_SPEC], limit=5,
                **({} if width is None else {"seed_top_k": width}))

    assert len(calls) == 2, (
        f"expected a reference arm and a variant arm, saw {len(calls)} recall calls")
    (ref_params, ref_kw), (var_params, var_kw) = calls
    assert var_params["query"] == PERTURBATION_Q8["perturbed_query"], (
        "the second recall call was not the variant arm, so which arm carried "
        "which width is unverified")
    assert ref_kw.get("seed_top_k") == expected, (
        f"the reference arm was recalled at {ref_kw.get('seed_top_k')!r}, not the "
        f"{expected} the eval was asked to score at")
    assert var_kw.get("seed_top_k") == expected, (
        "the perturbed arm was not recalled at the width the reference arm was, so "
        "the two arms are two different truncations")
    for params, _kw in calls:
        assert "seed_top_k" not in params, (
            "the width belongs in neither arm's params dict: that dict is the "
            "vault_recall wire shape")
    bounds = _entity_slice_bounds(inspect.getsource(ev._attach_counterfactual))
    assert bounds == ["seed_top_k"], (
        f"the variant seed slice is no longer bounded by the knob's value: {bounds}")
    assert not any(b.startswith("LITERAL") for b in bounds)


def test_the_variant_seed_slice_and_its_label_move_with_the_knob(monkeypatch):
    """What the shared width buys, measured rather than asserted: both arms of an
    entity-axis swap are sliced by the one knob, so `seed_moved` compares two
    queries. Cut the variant arm alone and the same fixture reports the swap as
    having moved the seed set — a truncation artifact wearing the costume of the
    label #537's identity-key premise is read from."""
    from eval import counterfactual as cf

    # The helper's own retrieval call is stubbed too: the assertion is about the
    # seed slice, and letting it reach qmd would make a width test depend on a
    # daemon. `_extract_entities_from_query` returns the same eight entities for
    # both arms, which is the fixture's whole point.
    monkeypatch.setattr(ev, "_extract_entities_from_query", lambda q: list(EIGHT_ENTITIES))
    monkeypatch.setattr(ev, "_vault_recall", _recall_stub([]))

    seeds = [e for e, _ in EIGHT_ENTITIES]
    result = {"documents": [{"path": "knowledge/a.md"}],
              "facts": [{"entity": "Ent 1", "fact_id": "f1"}]}

    rec: dict = {}
    ev._attach_counterfactual(rec, _Q8_SPEC, result, seeds,
                              {"query": _Q8_SPEC["query"]}, PERTURBATION_Q8,
                              seed_top_k=8)
    assert "counterfactual_error" not in rec, rec["counterfactual_error"]
    assert len(rec["counterfactual"]["seeds_variant"]) == 8, (
        "the variant arm was sliced narrower than the arm it is compared with")
    assert rec["counterfactual"]["seed_moved"] is False, (
        "both arms extract the same eight entities, so a seed_moved here came "
        "from the two slices, not from the perturbed query")
    label = cf.label_failures(
        [{"id": "q8", "counterfactual": rec["counterfactual"]}])[0]["label"]
    assert label == "entity_swap_seed_set_unchanged", (
        "a variant arm cut to a different count reports the seed set as moved, "
        "which suppresses the one label #537's premise is read from")

    narrow: dict = {}
    ev._attach_counterfactual(narrow, _Q8_SPEC, result, seeds,
                              {"query": _Q8_SPEC["query"]}, PERTURBATION_Q8,
                              seed_top_k=5)
    assert narrow["counterfactual"]["seed_moved"] is True, (
        "control: with the variant arm cut to five against a reference arm of "
        "eight the label must flip, which is what makes the assertion above a "
        "measurement of the width and not of the fixture")


def test_the_parity_flag_falsifies_on_the_seed_count(monkeypatch):
    """#843 clause 3. `matches_production_defaults` had five terms and none of
    them was the seed count, so an artifact stamped `true` for eight days while
    the eval seeded five against a production that seeded ten. The flag is a
    module-level callable now — comparable without a corpus — and an explicit
    narrower seeding must turn it off."""
    parser = ev.build_parser()
    assert callable(ev.build_run_config)
    default = ev.build_run_config(parser.parse_args([]))
    assert default["seed_top_k"] == ev.RECALL_SEED_TOP_K == 10
    assert default["matches_production_defaults"] is True
    narrower = ev.build_run_config(parser.parse_args(["--seed-top-k", "5"]))
    assert narrower["seed_top_k"] == 5
    assert narrower["matches_production_defaults"] is False, (
        "a seed count that differs from production's must not report fidelity")


def test_the_artifact_records_the_seed_count_it_scored_with():
    """#843 clause 4. A baseline can now be asked how many seeds it scored,
    rather than being assumed to have used whatever the constant says today —
    the assumption is what made the 5-vs-10 split invisible in the file that
    would have had to show it."""
    ns = ev.build_parser().parse_args(["--seed-top-k", "3"])
    cfg = ev.build_run_config(ns)
    assert cfg["seed_top_k"] == 3
    assert "seed_top_k" in cfg and cfg["seed_top_k"] != ev.RECALL_SEED_TOP_K


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
