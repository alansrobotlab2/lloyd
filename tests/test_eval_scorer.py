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


# ── #1548: which leg carried the entity hit ──────────────────────────────────
#
# `_entities_in_result` puts the query's own extracted seeds FIRST, so a gold
# entity the seed extractor happened to name satisfies `entity_hit` with nothing
# retrieval returned. Triage measured that on the live series: every one of the 14
# miss->hit flips between nightly-20260925 and nightly-20260926 (entity_hit_rate
# 0.337 -> 0.500) was carried by `seeds_extracted`, and none by a returned fact
# alone. These tests pin the attribution that makes the two legs separable from the
# artifact itself.

CARIED_BY_FACT = "KG Maintenance Tasks"
CARIED_BY_EXPANDED = "Graph v4 Classifier"
CARIED_BY_NEIGHBOUR = "Alertmanager"
SEED_ONLY_GOLD = "Knowledge Graph"


def _legged_result() -> dict:
    """One retrieval output with an entity on each of the three retrieval legs.

    `entity` is the key `agent_mcp.vault._vault_recall` emits
    (`{**fact, "entity": <resolved>}`), which is the key `_entities_in_result`
    reads, so a leg that is not populated here is absent from the union too.
    """
    return {
        "facts": [{"entity": CARIED_BY_FACT, "text": "t"}],
        "graph_expanded_facts": [{"entity": CARIED_BY_EXPANDED}],
        "graph_neighbors_used": [{"entity": CARIED_BY_NEIGHBOUR, "weight": 1.0}],
        "documents": [{"path": "knowledge/kg.md"}],
    }


def _record(qid: str, spec: dict, result: dict, seeds: list[str]) -> dict:
    """A scored record, shaped like the one `main()` writes."""
    return {"id": qid, "query": spec["query"], "category": spec.get("category", "entity"),
            "seeds_extracted": seeds, "result_summary": {}, "error": None,
            "latency_ms": 10.0, "scoring": ev._score(spec, result, seeds=seeds)}


def test_each_matched_entity_records_which_leg_satisfied_it():
    """#1548 clause 1: the artifact can tell a seed-carried hit from a
    retrieval-carried one, per matched entity, not just per query.

    Four gold entities: one carried only by the query's own seed, and one by each
    of the three retrieval legs. The seed says `["seed"]` with
    `retrieval_satisfied` false and the retrieval legs name themselves. Without
    this the two cases are the same number in the artifact — which is the defect.
    `CARIED_BY_FACT` is ALSO in the seed list, on purpose: an entity retrieval
    returned and the extractor happened to name is carried by retrieval
    (`["seed", "fact"]`), so a fix that subtracted the seed leg out of the
    retrieval legs would report `["seed"]` here and quietly under-credit the
    nights where retrieval agrees with the query.
    """
    spec = {"id": "legs", "query": "how good is the knowledge graph",
            "expect_entities": [SEED_ONLY_GOLD, CARIED_BY_FACT,
                                CARIED_BY_EXPANDED, CARIED_BY_NEIGHBOUR],
            "expect_docs": ["knowledge/kg.md"]}
    sc = ev._score(spec, _legged_result(),
                   seeds=[SEED_ONLY_GOLD, "Graph", CARIED_BY_FACT])
    attrib = sc["entity_legs"]
    assert set(attrib) == {ev._norm(e) for e in spec["expect_entities"]}, attrib
    assert attrib[ev._norm(SEED_ONLY_GOLD)] == {
        "legs": ["seed"], "retrieval_satisfied": False}, attrib
    assert attrib[ev._norm(CARIED_BY_FACT)] == {
        "legs": ["seed", "fact"], "retrieval_satisfied": True}, attrib
    assert attrib[ev._norm(CARIED_BY_EXPANDED)] == {
        "legs": ["graph_expanded_fact"], "retrieval_satisfied": True}, attrib
    assert attrib[ev._norm(CARIED_BY_NEIGHBOUR)] == {
        "legs": ["graph_neighbor"], "retrieval_satisfied": True}, attrib
    # An expected entity that matched nothing is not attributed at all: there is
    # no leg to name, and inventing one would be the guess this field exists to
    # avoid.
    miss = ev._score({"id": "m", "query": "q", "expect_entities": ["Nonexistent Entity"],
                      "expect_docs": []}, _legged_result(), seeds=["Graph"])
    assert miss["entity_legs"] == {}, miss
    assert miss["entity_hit"] is False and miss["entity_hit_retrieval_carried"] is False


def test_summarize_reports_the_retrieval_carried_rate_beside_the_seeded_one():
    """#1548 clause 2: `summary.overall.entity_hit_rate_retrieval_carried` counts
    a query as a hit only when a fact, a graph-expanded fact or a graph neighbour
    carried a matched entity, and the gap to `entity_hit_rate` is exactly the
    seed-carried queries.

    Three records over the SAME retrieval output: one hit the query's seed handed
    it, one hit a returned fact carried, one miss. `entity_hit_rate` cannot tell
    that set from a set of two retrieval-carried hits — the retrieval-carried rate
    can (1/3 vs 2/3), and that is the number the entity leg means.
    """
    result = _legged_result()
    seed_rec = _record("seed-only", {"id": "a", "query": "q", "category": "entity",
                                     "expect_entities": [SEED_ONLY_GOLD],
                                     "expect_docs": []},
                       result, seeds=[SEED_ONLY_GOLD])
    fact_rec = _record("fact-carried", {"id": "b", "query": "q", "category": "entity",
                                        "expect_entities": [CARIED_BY_FACT],
                                        "expect_docs": []},
                       result, seeds=["Unrelated Seed"])
    miss_rec = _record("miss", {"id": "c", "query": "q", "category": "entity",
                                "expect_entities": ["Nonexistent Entity"],
                                "expect_docs": []},
                       result, seeds=["Unrelated Seed"])
    assert seed_rec["scoring"]["entity_hit"] and fact_rec["scoring"]["entity_hit"]
    assert seed_rec["scoring"]["entity_hit_retrieval_carried"] is False
    assert fact_rec["scoring"]["entity_hit_retrieval_carried"] is True

    s = ev.summarize([seed_rec, fact_rec, miss_rec])["overall"]
    n = s["n_queries"]
    assert n == 3
    # `avg()` rounds to 3 dp, as every rate in this artifact does.
    assert s["entity_hit_rate"] == pytest.approx(2 / 3, abs=0.001)
    assert s["entity_hit_rate_retrieval_carried"] == pytest.approx(1 / 3, abs=0.001)
    assert s["entity_hit_rate_retrieval_carried"] <= s["entity_hit_rate"]
    # The gap IS the seed-only query count, on this record set and on any other:
    # multiplying back out is how a reader checks it without re-scoring.
    seed_only = sum(1 for r in (seed_rec, fact_rec, miss_rec)
                    if r["scoring"]["entity_hit"]
                    and not r["scoring"]["entity_hit_retrieval_carried"])
    assert round((s["entity_hit_rate"] - s["entity_hit_rate_retrieval_carried"]) * n) \
        == seed_only == 1
    # Clause 3 of the item: the two legs are reported side by side, so a reader
    # comparing them need not conflate them. `fact_entity_recall_avg`'s
    # denominator is the fact leg and cannot be seed-inflated.
    assert s["fact_entity_recall_avg"] == pytest.approx(1 / 3, abs=0.001)


def test_widening_the_seed_list_cannot_move_the_retrieval_carried_rate():
    """#1548 clause 3: the same retrieval output, scored twice, narrow seeds then
    seeds widened with an extra gold-naming seed.

    `entity_hit_rate` goes UP (0.5 -> 1.0) because the widened list names the gold
    entity, which is exactly the self-fulfilling credit the item is about; the
    retrieval-carried rate does not move, because nothing retrieval returned
    changed. A scorer that deduped seeds into the fact leg, or computed the
    retrieval rate off the seeded union, would show 0.5 -> 1.0 here too.
    """
    result = _legged_result()
    spec_seed_only = {"id": "g", "query": "how good is the knowledge graph",
                      "category": "entity", "expect_entities": [SEED_ONLY_GOLD],
                      "expect_docs": []}
    spec_fact = {"id": "f", "query": "what maintenance tasks does the kg need",
                 "category": "entity", "expect_entities": [CARIED_BY_FACT],
                 "expect_docs": []}
    narrow = [spec_seed_only, spec_fact], ["Graph"]
    wide = [spec_seed_only, spec_fact], ["Graph", SEED_ONLY_GOLD]

    out = {}
    for label, (specs, seeds) in zip(("narrow", "wide"), (narrow, wide)):
        recs = [_record(s["id"], s, result, seeds) for s in specs]
        out[label] = (recs, ev.summarize(recs)["overall"])
    (narrow_recs, narrow_s), (wide_recs, wide_s) = out["narrow"], out["wide"]

    assert narrow_recs[0]["scoring"]["entity_hit"] is False
    assert wide_recs[0]["scoring"]["entity_hit"] is True, (
        "the widened seed must be what flips the seeded hit, or the test proves "
        "nothing about the seed leg")
    assert wide_recs[0]["scoring"]["entity_legs"][ev._norm(SEED_ONLY_GOLD)] == {
        "legs": ["seed"], "retrieval_satisfied": False}
    assert narrow_s["entity_hit_rate"] == pytest.approx(0.5, abs=0.001)
    assert wide_s["entity_hit_rate"] == pytest.approx(1.0, abs=0.001)
    assert (narrow_s["entity_hit_rate_retrieval_carried"]
            == wide_s["entity_hit_rate_retrieval_carried"]
            == pytest.approx(0.5, abs=0.001)), (
        "identical retrieval output, identical retrieval-carried rate — a rate that "
        "moved here is reading the seed list")


def test_the_written_artifact_separates_the_seed_leg_from_the_retrieval_leg(tmp_path,
                                                                            monkeypatch):
    """#1548 clauses 1, 2 and 4, at the process boundary: the bytes on disk.

    `ev.main` runs in-process with the recall and the seed extractor replaced by a
    fixture, and the baseline file it writes is read back. The fixture is the
    defect itself: the extractor names the gold entity `Nightly Retrieval Eval`
    (as `recall_seeds` did for four of the fourteen flipped queries on
    2026-09-26, incl. `nightly-cannot-tell`), while retrieval returns only
    `ObserverState`. In the file the run wrote, that query must say

      scoring.entity_legs     {"nightly retrieval eval": {"legs": ["seed"],
                                                          "retrieval_satisfied": false}}
      scoring.entity_hit / _retrieval_carried   true / false
      result_summary.fact_entities_top10        ["ObserverState"]  (no gold, no seed)
      summary.overall.entity_hit_rate                       1.0
      summary.overall.entity_hit_rate_retrieval_carried     0.0

    Unit-level asserts on `_score` cannot catch the wiring this pins: a
    `result_summary` that keeps calling the seeded union, or a `summarize` whose
    new key reads the wrong field, leaves every unit test green.
    """
    gold, returned = "Nightly Retrieval Eval", "ObserverState"
    monkeypatch.setattr(ev, "_recall_seeds", lambda q, k: [gold])
    monkeypatch.setattr(ev, "_semantic_seed_k", lambda: 1)

    def fake_recall(params, **kw):
        return {"entities": [], "facts": [{"entity": returned, "text": "t",
                                           "source": "memory/x.md"}],
                "documents": [{"path": "memory/daily/2026-09-26.md"}],
                "graph_expanded_facts": [], "graph_neighbors_used": [],
                "fact_read_coverage": {"attempted": 1, "read": 1},
                "graph_expansion": {}}

    def fake_store():
        class _S:
            def stats(self):
                return {"entities_total": 1, "edges_total": 1, "edges_active": 1,
                        "aliases": 0, "facts": 1}
            def resolve(self, text):
                return text
            def active_edges(self):
                return []
            def neighbours(self, *a, **k):
                return {}
        return _S()

    qf = tmp_path / "q.yaml"
    qf.write_text(
        "queries:\n"
        "  - id: seed-carried\n"
        "    query: what did the nightly eval find\n"
        "    category: entity\n"
        f"    expect_entities: [{gold}]\n"
        "    expect_docs: [memory/daily/2026-09-26.md]\n")
    out = tmp_path / "baselines"
    monkeypatch.setattr(ev, "_vault_recall", fake_recall)
    monkeypatch.setattr(ev, "store", fake_store)
    monkeypatch.setattr(ev, "EVAL_BASELINES_DIR", out)
    monkeypatch.setattr(ev, "LLOYD_CODE_ROOT", tmp_path)
    argv = ["--queries", str(qf), "--label", "nightly-legs",
            "--allow-empty-corpus"]
    monkeypatch.setattr(sys, "argv", ["run_eval.py"] + argv)
    assert ev.main() == 0
    written = next(iter(sorted(out.glob("nightly-legs-*.json"))))
    blob = json.loads(written.read_text())
    rec = blob["records"][0]
    assert rec["seeds_extracted"] == [gold], rec["seeds_extracted"]
    assert rec["scoring"]["entity_hit"] is True, "fixture: the seed leg scores it a hit"
    assert rec["scoring"]["entity_hit_retrieval_carried"] is False
    # Keyed by the `_norm`'d gold label — the same spelling `entities_matched`
    # uses, so the two fields index the same entity.
    assert rec["scoring"]["entity_legs"][ev._norm(gold)] == {
        "legs": ["seed"], "retrieval_satisfied": False}, rec["scoring"]["entity_legs"]
    top10 = rec["result_summary"]["fact_entities_top10"]
    assert top10 == [returned], top10
    assert gold not in " ".join(top10), (
        "the field that says what retrieval returned is seeded again, which is the"
        " half of #1548 the item's own proving command tripped over")
    overall = blob["summary"]["overall"]
    assert overall["entity_hit_rate"] == 1.0
    assert overall["entity_hit_rate_retrieval_carried"] == 0.0, (
        "the artifact reports a retrieval gain it did not measure")
    # The rate is registered in CI_METRICS, so it travels with an interval and a
    # denominator like the rate it accompanies, instead of being a bare number.
    assert "entity_hit_rate_retrieval_carried" in ev.CI_METRICS


def test_fact_entities_top10_is_retrieval_only():
    """#1548 clause 4: the artifact's "entities retrieval returned" field must not
    lead with the query's own seeds.

    Before this it was `_entities_in_result(result, seeds)[:10]`, so an entity
    present ONLY in `seeds_extracted` appeared in it, and the field that reads like
    retrieval output was the seed extractor's output — which is how the item's own
    proving command came back reading seeds as answers. The entity on each
    retrieval leg is still there, in leg order, and the seed-only one never is.
    """
    seeds = [SEED_ONLY_GOLD, "Graph", CARIED_BY_FACT]
    got = ev._retrieval_entities(_legged_result())
    assert got == [CARIED_BY_FACT, CARIED_BY_EXPANDED, CARIED_BY_NEIGHBOUR], got
    assert ev._norm(SEED_ONLY_GOLD) not in {ev._norm(e) for e in got}
    # A seed that also names a retrieval entity stays: it IS in the result. The
    # exclusion is of seed-only names, not of seeds.
    assert CARIED_BY_FACT in got
    assert ev._entities_in_result(_legged_result(), seeds=seeds)[:1] == [SEED_ONLY_GOLD], (
        "positive control: the seeded union still leads with the seed, so the two "
        "fields are genuinely different functions and not one renamed")


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


def test_eval_defaults_are_productions():
    """The eval's own CLI defaults must be what vault_recall serves, or the
    nightly trend measures a configuration nobody runs.

    Every value here comes out of `ev.build_parser()` — the parser `main()`
    actually parses (eval/run_eval.py:919) — never a parser this test builds.
    The version #999 replaces constructed its own `argparse.ArgumentParser`,
    handed it the same `vault.RECALL_*` values it then asserted against, and
    stayed green with the real `--alpha` default changed to 0.9; both of its
    assertions were tautologies over its own construction. Reading the real
    parser is what makes this one able to fail.

    Values are compared to the constants, not to literals: naming 0.3 here
    would re-create the stale-number problem, and the measured value of each
    constant is pinned by `test_production_knobs_keep_their_measured_values`.
    """
    from agent_mcp import vault
    # argparse dest -> the constant that default has to be. These are the knobs
    # of KNOBS below in argparse's dest spelling; the one that differs from the
    # `run_eval()` parameter it feeds is `--alpha`, which lands on dest `alpha`
    # while the parameter is `rerank_alpha`.
    consts = {"graph_rerank": "RECALL_GRAPH_RERANK",
              "alpha": "RECALL_RERANK_ALPHA",
              "graph_top_k": "RECALL_GRAPH_TOP_K",
              "graph_hops": "RECALL_GRAPH_HOPS",
              # #843's knob: the eval seeded 5 query entities while production
              # seeded 10, so the seed width is a knob this test also owns.
              "seed_top_k": "RECALL_SEED_TOP_K"}
    args = ev.build_parser().parse_args([])
    for dest, const in consts.items():
        production = getattr(vault, const)
        assert getattr(args, dest) == production, f"--{dest.replace('_', '-')} default != {const}"
        # 0 == False in Python, so a bool knob that quietly became an int would
        # pass the line above. Pin the kind as well as the value.
        assert type(getattr(args, dest)) is type(production), const


def test_eval_defaults_are_productions_cannot_be_satisfied_by_a_hand_mirrored_parser():
    """The guard on the guard. #999's defect was structural — a defaults test
    that read a parser it built itself — and a rewrite that quietly grew a
    mirrored parser back would stay green forever. So this reads the source of
    `test_eval_defaults_are_productions` and requires the two properties the
    clause names: it calls `build_parser()`, and it builds no parser of its own
    (the original did `import argparse` then `argparse.ArgumentParser()`, which
    is what either needle catches)."""
    src = Path(__file__).read_text()
    marker = "def test_eval_defaults_are_productions():"
    assert marker in src, "the test this guard protects is gone or renamed"
    body = src.split(marker, 1)[1].split("\ndef ", 1)[0]
    assert "ev.build_parser()" in body, "no longer reads the parser main() parses"
    assert "ArgumentParser()" not in body, "constructs a parser of its own again"
    assert "import argparse" not in body, "imports argparse to build a parser of its own"


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
    src = inspect.getsource(vault._vault_recall_base)
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
    from app.paths import EVAL_BASELINES_DIR
    latest = sorted(EVAL_BASELINES_DIR.glob("*.json"),
                    key=lambda p: p.stat().st_mtime)
    if not latest:
        pytest.skip("no baseline runs recorded yet")
    rec = json.loads(latest[-1].read_text())
    assert "matches_production_defaults" in rec
    assert {"graph_rerank", "rerank_alpha", "graph_top_k", "graph_hops"} <= set(rec)


# ── #1260 clause 2: the entity leg resolves before it compares ───────────────

def _resolving_store(tmp_path):
    """A real KGStore carrying the alias rows these tests are about.

    Built through `KGStore` + `entities.register` + `aliases.set` rather than a stub
    with a hand-written `resolve`, because the property under test is exactly that
    the scorer defers to the STORE's notion of alias-equivalence. A stub answering
    "yes, those two are the same entity" would have the test assert its own
    assumption; here the scorer is only as alias-aware as a real table makes it.
    """
    import app.kg_store as ks

    db = tmp_path / "kg.sqlite"
    st = ks.KGStore(db)
    for name in ("Autonomy Data Pipeline", "Knowledge Graph", "Task #363",
                 "TGS-RAG", "Backlog System"):
        st.entities.register(name, kind="system")
    st.aliases.set("Autonomy Pipeline", "Autonomy Data Pipeline",
                   kind="semantic", origin="test")
    st.aliases.set("KG", "Knowledge Graph", kind="semantic", origin="test")
    return st


def test_a_retrieved_alias_satisfies_the_gold_canonical(tmp_path, monkeypatch):
    """Gold `Autonomy Data Pipeline` is satisfied by a returned `Autonomy Pipeline`.

    `entity_matches` asked only `exp in got`, so a name that IS the gold entity under
    a different spelling could never satisfy it — and a canonical SHORTER than the
    gold can never be a substring of it, so no amount of better retrieval could close
    those. The 09-18 baseline returned `Task #363` where gold said `Backlog Item #363`
    and scored the pair a miss.
    """
    monkeypatch.setattr(ev, "store", lambda: _resolving_store(tmp_path))
    out = ev._score({"query": "q", "expect_entities": ["Autonomy Data Pipeline"]},
                    {"documents": [], "facts": []}, seeds=["Autonomy Pipeline"])
    assert out["entity_hit"] is True and out["entity_recall"] == 1.0, out
    assert out["entities_matched"] == ["autonomy data pipeline"], (
        "the matched value stays the EXPECTED string, so `entities_matched` remains "
        "comparable with a baseline written before resolution existed: "
        f"{out['entities_matched']}")


def test_a_name_the_alias_table_does_not_map_still_fails_the_gold(tmp_path, monkeypatch):
    """Gold `Knowledge Graph` is NOT satisfied by a returned `Graph`.

    Resolution is not fuzziness. The table maps `KG` onto `Knowledge Graph` and
    pointedly has no row for `Graph`, so a returned `Graph` must stay a miss —
    otherwise the scorer would grade any string sharing a word as the same entity and
    the entity leg would stop measuring anything.
    """
    st = _resolving_store(tmp_path)
    monkeypatch.setattr(ev, "store", lambda: st)
    assert st.resolve("Graph") is None, "the table must stay silent on `Graph`"
    assert st.resolve("KG") == "Knowledge Graph", "and answer on the row it does have"
    miss = ev._score({"query": "q", "expect_entities": ["Knowledge Graph"]},
                     {"documents": [], "facts": []}, seeds=["Graph"])
    assert miss["entity_hit"] is False and miss["entity_recall"] == 0.0, miss
    hit = ev._score({"query": "q", "expect_entities": ["Knowledge Graph"]},
                    {"documents": [], "facts": []}, seeds=["KG"])
    assert hit["entity_hit"] is True, "same scorer, the alias the table actually holds"


def test_resolution_does_not_replace_the_substring_rule(tmp_path, monkeypatch):
    """Containment stays as the additive half, or the corpus loses a target.

    `TGS-RAG Implementation` is no entity row, and `tests/test_eval_corpus_guard.py`
    defends it precisely because containment through the row `#363 TGS-RAG
    Implementation` is what reaches it. An equality-only reading would have demanded
    that expectation be retargeted, which is a substitution for no reason.
    """
    st = _resolving_store(tmp_path)
    monkeypatch.setattr(ev, "store", lambda: st)
    assert st.resolve("TGS-RAG Implementation") is None
    got = ev._score({"query": "q", "expect_entities": ["TGS-RAG Implementation"]},
                    {"documents": [], "facts": []}, seeds=["#363 TGS-RAG Implementation"])
    assert got["entity_hit"] is True, got


def test_the_fact_leg_keeps_its_pre_resolution_definition(tmp_path, monkeypatch):
    """`fact_entity_recall` stays substring-only, deliberately.

    #1164's acceptance is written as `fact_entity_recall_avg >= 0.475`; redefining
    the fact leg here would move the ground under a clause this round does not own.
    So the entity leg resolves and the fact leg does not — an asymmetry with a
    reason, not an oversight, and this test is what keeps it a decision.
    """
    monkeypatch.setattr(ev, "store", lambda: _resolving_store(tmp_path))
    out = ev._score(
        {"query": "q", "expect_entities": ["Knowledge Graph"]},
        {"documents": [], "facts": [{"entity": "KG", "fact": "edges"}]}, seeds=[])
    assert out["entity_hit"] is True, "the entity leg got the resolution"
    assert out["fact_entity_recall"] == 0.0, (
        "the fact leg now resolves too, which moves #1164's acceptance metric: "
        f"{out['fact_entity_recall']}")


def test_an_unreadable_store_leaves_the_comparison_unchanged(tmp_path, monkeypatch):
    """A store that will not open must not turn alias-equivalent names into misses.

    With no resolution the scorer falls back to the substring rule it has always
    used, so a worktree run prints the numbers the pre-#1260 tree printed. The other
    available fallback — resolving every unknown name to one shared bucket — would
    report a storage fault as a retrieval collapse, the shape this file's neighbours
    keep meeting.
    """
    import app.kg_store as ks

    def _refuse():
        raise ks.StoreUnavailable("no kg.sqlite on this path")

    monkeypatch.setattr(ev, "store", _refuse)
    out = ev._score({"query": "q", "expect_entities": ["Autonomy Data Pipeline"]},
                    {"documents": [], "facts": []}, seeds=["Autonomy Pipeline"])
    assert out["entity_hit"] is False, "no store, no resolution — and no false hit"
    kept = ev._score({"query": "q", "expect_entities": ["TGS-RAG Implementation"]},
                     {"documents": [], "facts": []}, seeds=["#363 TGS-RAG Implementation"])
    assert kept["entity_hit"] is True, "containment still works without a store"


# ── #1260 clause 4: the harness reports its own seed-side ceiling ────────────

def _anchor_record(qid, seeds, expects):
    return {"id": qid, "query": qid, "category": "single",
            "seeds_extracted": list(seeds),
            "expected": {"entities": list(expects), "docs": ["x"]},
            "scoring": {"entity_hit": False, "doc_hit": True, "entity_recall": 0.0,
                        "doc_recall": 1.0, "rr_doc": 1.0, "ndcg10": 0.5,
                        "fact_entity_recall": 0.0},
            "latency_ms": 100.0, "error": None}


def test_summarize_reports_the_anchorless_query_count_and_ids():
    """`summary.overall` carries the residue, so the ceiling travels with every
    baseline artifact instead of living in a triage comment.

    Three item families (#569 seed scoring, #633/#634 graph arms, #843 seed width)
    each proposed a knob against a 0.5 entity hit rate that no knob of theirs could
    move, because nothing in the artifact said how many queries had no seed to start
    an entity search from.
    """
    recs = [_anchor_record("anchored", ["Knowledge Graph"], ["Knowledge Graph"]),
            _anchor_record("alias-anchorless", ["Relationship Graph"], ["Knowledge Graph"]),
            _anchor_record("no-gold-entity", ["Index"], [])]
    out = ev.summarize(recs)
    assert out["overall"]["anchorless_query_count"] == 1, out["overall"]
    assert out["overall"]["anchorless_query_ids"] == ["alias-anchorless"], out["overall"]
    # A query with no gold entity is not scorable on the entity leg, so it is not
    # anchorless either; counting it would inflate the residue by the corpus's shape
    # rather than by the extractor's reach.
    assert "no-gold-entity" not in out["overall"]["anchorless_query_ids"]


def test_anchorless_reads_the_seeds_and_not_the_answer():
    """The residue is a property of the seeds, so a good answer cannot hide it.

    Reading the returned entities instead would report the ceiling as zero whenever
    retrieval reached the entity by some other route — which is how this stayed
    invisible: `doc_hit` is 1.0 on all twenty queries while half of them cannot start
    an entity search at all.
    """
    rec = _anchor_record("reached-anyway", ["Index"], ["Knowledge Graph"])
    rec["scoring"] = dict(rec["scoring"], entity_hit=True, entity_recall=1.0)
    assert ev.anchorless_queries([rec]) == ["reached-anyway"]


def test_a_record_with_no_recorded_seeds_is_not_counted_anchorless():
    """Zero recorded seeds and no recorded seeds are different observations.

    A synthetic record, or a baseline written before the field existed, carries
    `None`. Reporting that as anchorless is a verdict from a missing input — the
    count would then measure which artifacts happen to have the key, the same shape
    as every other guard on this box that read its own absent input and reported a
    finding it could not justify.
    """
    rec = _anchor_record("unseeded", ["Index"], ["Knowledge Graph"])
    rec["seeds_extracted"] = None
    assert ev.anchorless_queries([rec]) == []


# ── ci95: the interval beside every scored metric (#696) ─────────────────────

CI_METRICS = ("entity_hit_rate", "doc_hit_rate", "entity_recall_avg",
              "doc_recall_avg", "mrr_doc", "ndcg10", "fact_entity_recall_avg")


def _ci_records(hits, doc_hits, values=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.5,
                                       0.1, 0.3, 0.7, 0.9, 0.2, 0.4, 0.6,
                                       0.8, 0.0, 0.5, 1.0, 0.3, 0.7)):
    """Twenty scored records with named hit patterns and one value vector reused
    across the five non-binary metrics, so a Wilson bound can be checked against
    the same per-query hits the summary was built from."""
    return [
        {"id": f"ci{i}", "category": "single", "latency_ms": 100.0, "error": None,
         "scoring": {"entity_hit": bool(hits[i]), "doc_hit": bool(doc_hits[i]),
                     "entity_recall": values[i], "doc_recall": values[i],
                     "rr_doc": values[i], "ndcg10": values[i],
                     "fact_entity_recall": values[i], "first_doc_rank": 1}}
        for i in range(20)
    ]


def test_summarize_writes_a_ci95_block_covering_all_seven_scored_metrics():
    """Clause 2 of #696. Every number in `overall` is a 20-query point estimate
    and the printed line and the nightly report both read it as a verdict, so
    the interval has to live in the artifact — a summary a reader cannot bound
    is a summary a reader will over-read."""
    hits = [1] * 10 + [0] * 10
    doc_hits = [1] * 17 + [0] * 3
    overall = ev.summarize(_ci_records(hits, doc_hits))["overall"]
    ci = overall["ci95"]
    assert set(CI_METRICS) <= set(ci), set(CI_METRICS) - set(ci)
    for metric in CI_METRICS:
        entry = ci[metric]
        assert isinstance(entry["ci"], list) and len(entry["ci"]) == 2, metric
        assert entry["n"] == 20, metric
    assert ci["params"] == {"confidence": 0.95, "n_resamples": 10000,
                            "seed": ev.evstats.SEED}


def test_the_binary_intervals_come_from_the_same_per_query_hits_as_the_rate():
    """Wilson over k=10 of n=20 is (0.2993, 0.7007) — the interval the item
    quotes, and the reason one flipped query cannot be called a regression."""
    hits = [1] * 10 + [0] * 10
    doc_hits = [1] * 17 + [0] * 3
    overall = ev.summarize(_ci_records(hits, doc_hits))["overall"]
    ci = overall["ci95"]
    assert overall["entity_hit_rate"] == 0.5
    assert ci["entity_hit_rate"]["ci"] == [0.2993, 0.7007]
    assert ci["entity_hit_rate"]["k"] == 10
    # k=17/20 is the item's other worked number: a rate of 0.85 whose honest
    # interval reaches down to 0.64 — the 09-09 vs 09-14 nightly pair that the
    # old report called "doc_hit went 1.00" sits entirely inside this bracket.
    assert overall["doc_hit_rate"] == 0.85
    assert ci["doc_hit_rate"]["ci"] == [0.6396, 0.9476]


def test_the_non_binary_intervals_are_bootstrap_not_wilson():
    """mrr_doc / ndcg10 / the recalls are means of a per-query value, not counts
    of successes, so a binomial interval does not describe them. The check that
    it is a bootstrap: it is computed from the value vector, so a vector with
    spread gets a wide bracket while a constant vector collapses."""
    spread = ev.summarize(_ci_records([1] * 20, [1] * 20))["overall"]["ci95"]
    flat = ev.summarize([
        {"id": f"f{i}", "category": "single", "latency_ms": 10.0, "error": None,
         "scoring": {"entity_hit": True, "doc_hit": True, "entity_recall": 0.5,
                     "doc_recall": 0.5, "rr_doc": 0.5, "ndcg10": 0.5,
                     "fact_entity_recall": 0.5, "first_doc_rank": 1}}
        for i in range(20)])["overall"]["ci95"]
    lo, hi = spread["mrr_doc"]["ci"]
    assert lo < 0.55 < hi, (lo, hi)
    assert spread["ndcg10"]["kind"] == "bootstrap"
    assert flat["mrr_doc"]["ci"] == [0.5, 0.5]      # no spread, no uncertainty
    assert spread["mrr_doc"]["ci"] != flat["mrr_doc"]["ci"]


def test_a_metric_scored_on_no_query_gets_no_interval_not_a_bracket_around_nothing():
    """The zero-denominator failure mode #696 clause 3 names, at the artifact
    layer: with every fact score None the rate is null, and an interval computed
    over an empty vector must be null too rather than the [0.0, 0.0] a naive
    mean-of-nothing would report."""
    records = _ci_records([1] * 20, [1] * 20)
    for r in records:
        r["scoring"]["fact_entity_recall"] = None
    overall = ev.summarize(records)["overall"]
    assert overall["fact_entity_recall_avg"] is None
    entry = overall["ci95"]["fact_entity_recall_avg"]
    assert entry["ci"] == [None, None] and entry["n"] == 0, entry


def test_the_ci95_block_survives_the_baseline_round_trip():
    """NaN is not valid JSON, and a baseline that `json.load` cannot read is not
    a baseline — which is why the empty case serialises as nulls instead. The
    subprocess path that writes a real baseline file is covered end to end in
    `test_eval_ci_reporting.py`; what this pins is the narrower claim that the
    block `summarize` builds survives a JSON round trip with its bounds intact."""
    import json as _json
    # The runner's own harness for this lives in test_eval_corpus_guard.py; here
    # the cheaper equivalent: summarise, dump, reload.
    overall = ev.summarize(_ci_records([1] * 10 + [0] * 10, [1] * 17 + [0] * 3))["overall"]
    blob = _json.dumps({"summary": {"overall": overall}})
    assert "NaN" not in blob and "Infinity" not in blob
    back = _json.loads(blob)["summary"]["overall"]["ci95"]
    assert back["entity_hit_rate"]["ci"] == [0.2993, 0.7007]


# ── #1000: the one knob whose default is deliberately NOT production's ───────

def test_expand_graph_default_is_named_by_one_production_constant():
    """Production's default for graph-expanded recall has a single source.

    Before #1000 it was a bare literal at the read site in `_vault_recall`, with
    no constant for anything to compare against — which is why the eval's
    parity check could substitute a self-comparison and stay undetected for
    eight months. Three surfaces have to agree now, and each is pinned because
    each drifts on its own:

    - `_vault_recall` applies `RECALL_EXPAND_GRAPH` when the key is absent, so
      the constant IS the value rather than a description of it;
    - the `vault_recall` schema states the default through the same constant, so
      what a caller is told cannot drift from what is applied (the schema's
      neighbours already worked this way; this knob was the exception);
    - `eval/run_eval.py` imports it, which is what makes its honesty assertion a
      comparison against production instead of against its own flag.
    """
    import asyncio

    import agent_mcp.vault as vault

    assert vault.RECALL_EXPAND_GRAPH is False, (
        "the eval's parity claim is pinned to this value; if production starts "
        "expanding the graph by default, re-read "
        "tests/test_eval_corpus_guard.py::"
        "test_the_written_baseline_never_claims_graph_parity_it_lacks first")
    src = inspect.getsource(vault._vault_recall_base)
    assert 'params.get("expand_graph", RECALL_EXPAND_GRAPH)' in src, (
        "_vault_recall defaults the knob from a literal again, so the constant "
        "can drift from what production actually serves")
    # Since 2026-09-23 the knob is an eval knob, out of the tool schema, so no
    # client is told a default at all; what production serves is the constant.
    tool = next(t for t in asyncio.run(vault.list_tools())
                if t.name == "vault_recall")
    assert "expand_graph" not in tool.input_schema["properties"]
    assert "expand_graph" in vault.RECALL_EVAL_KNOBS
    assert ev.RECALL_EXPAND_GRAPH is vault.RECALL_EXPAND_GRAPH, (
        "the eval compares against its own copy of the number, which is how "
        "#1000 became invisible in the first place")


def test_matches_production_defaults_never_claims_graph_parity():
    """#1000: no term of the conjunction may compare a CLI flag with itself.

    `and not args.no_graph` closed that expression — a `store_true` flag against
    its own absence, so it could only ever turn the verdict off and could never
    disagree with production, while production defaults the knob to the opposite
    of what the eval runs.

    Pinned in both directions, because either alone is gameable. The term is
    gone and every remaining term is `args.<knob> == RECALL_<KNOB>`; and the
    eval still runs the graph expanded, with the parity claim living in its own
    `expand_graph_matches_production` field. Widening the conjunction instead
    would satisfy the first half by calling every nightly run a non-production
    configuration — a different claim, not an honest one.
    """
    def cfg(argv=None):
        return ev.build_run_config(ev.build_parser().parse_args(list(argv or [])))

    default = cfg([])
    assert default["expand_graph"] is True, (
        "the fix makes the claim honest and must not also flip what the eval "
        "measures: the graph stays expanded by default")
    assert default["expand_graph_matches_production"] is False
    assert default["matches_production_defaults"] is True, (
        "the knob moved out of the conjunction; a default run still matches "
        "production on the six knobs that remain in it")

    production_shaped = cfg(["--no-graph"])
    assert production_shaped["expand_graph"] is False
    assert production_shaped["expand_graph_matches_production"] is True, (
        "the field must be a real comparison, not a constant restatement: it "
        "has to be able to say true when the run does match production")
    assert production_shaped["matches_production_defaults"] is True

    body = inspect.getsource(ev.build_run_config).split("return {", 1)[1]
    assert "args.no_graph" not in body, (
        "a self-comparison is back inside the parity record")
    conjunction = body.split('"matches_production_defaults": (', 1)[1]
    conjunction = conjunction.split("),", 1)[0]
    terms = [ln.strip() for ln in conjunction.splitlines()
             if "==" in ln and not ln.strip().startswith("#")]
    assert len(terms) == 6, f"parity terms changed shape: {terms}"
    for term in terms:
        # Each term is one parsed CLI value against one imported production
        # constant — the only shape that can disagree with production, which is
        # what the self-comparison could not do.
        assert ("args." in term and "RECALL_" in term
                and term.startswith(("args.", "and args."))), term

    # And the call site must not be "fixed" by matching production there: the
    # graph has to stay expanded, which is precisely why the claim moved to its
    # own field instead of into the conjunction.
    assert "expand_graph=not args.no_graph," in inspect.getsource(ev.main)


# ── #1547: the seeding the run scored with, recorded as a sibling field ──────

def test_the_run_config_records_the_semantic_seeding_it_scored_with(monkeypatch):
    """#1547 clauses 1 and 2: a baseline can be asked which seeding defined its
    seeds, and the answer is a SIBLING field, not a seventh parity term.

    #1486 (`dbfde750`, landed 2026-09-25 16:57 PDT) made `seeds_extracted` the
    lexical head UNIONED with up to `k` semantic seeds. The nightly of 09-25
    (seeding off) and the nightly of 09-26 (seeding on) both stamped
    `matches_production_defaults: true` while `entity_hit_rate` moved 0.337 ->
    0.500, `entity_recall_avg` 0.384 -> 0.579 and `anchorless_query_count` 25 ->
    16, with the document leg flat — because the knob lives in
    `retrieval.entity_seeding.semantic` and the conjunction compares six parsed
    args against six `RECALL_*` constants, so a config-sourced knob has neither
    side of a term. A seventh term would compare `semantic_seed_k()` with itself,
    which is #1000's defect: `expand_graph_matches_production` is the precedent
    for a knob config owns, and the sibling is the only satisfiable shape.
    """
    import yaml

    import agent_mcp.retrieval as ret

    def cfg():
        return ev.build_run_config(ev.build_parser().parse_args([]))

    # The plumbing in both directions: the field reports production's accessor
    # and it moves when the accessor moves. A hard-coded `{"enabled": True,
    # "k": 3}` passes the box-config assert below and fails here.
    monkeypatch.setattr(ret, "semantic_seed_k", lambda: 0)
    assert cfg()["semantic_seeding"] == {"enabled": False, "k": 0}, (
        "seeding off must record off, and `k: 0` is what `enabled: false` costs")
    monkeypatch.setattr(ret, "semantic_seed_k", lambda: 2)
    assert cfg()["semantic_seeding"] == {"enabled": True, "k": 2}
    monkeypatch.undo()

    # Unpatched: what THIS box serves, cross-checked against the config FILE
    # rather than against the accessor, so the record cannot be true merely by
    # agreeing with the same function it reads.
    sem = yaml.safe_load((ROOT / "config.yaml").read_text()) \
        ["retrieval"]["entity_seeding"]["semantic"]
    assert (sem.get("enabled"), sem.get("k")) == (True, 3), (
        f"the box's own seeding configuration moved off enabled: true / k: 3 "
        f"(config.yaml: {sem}); update this test's expectation with it")
    assert cfg()["semantic_seeding"] == {"enabled": True, "k": 3}, (
        "the default nightly run must record enabled: true, k: 3 — the run "
        f"recorded {cfg()['semantic_seeding']}")

    # Clause 2, the half that keeps the fix from being a widening.
    assert cfg()["matches_production_defaults"] is True, (
        "recording the seeding must not relabel every nightly a non-production "
        "configuration; that is the `expand_graph` mistake")
    body = inspect.getsource(ev.build_run_config).split("return {", 1)[1]
    assert '"semantic_seeding": _semantic_seeding_record(),' in body, (
        "the record must come from production's accessor, the one definition "
        "`recall_seeds()` slices at")
    conjunction = body.split('"matches_production_defaults": (', 1)[1]
    conjunction = conjunction.split("),", 1)[0]
    terms = [ln.strip() for ln in conjunction.splitlines()
             if "==" in ln and not ln.strip().startswith("#")]
    assert len(terms) == 6, f"the parity conjunction grew a term: {terms}"
    assert "semantic" not in conjunction, (
        "semantic seeding is back inside the conjunction, where it can only be "
        "a knob compared with itself (#1000)")


def test_the_seeding_record_reaches_a_real_artifact_and_a_real_reader(tmp_path):
    """#1547 clause 1, at the real seam: another process writes the bytes, and
    `app.uptake.retrieval_gate` — a different process's reader — bands on them.

    `eval/run_eval.py` runs as a subprocess exactly as the nightly does, into a
    `LLOYD_DATA` root this test owns, on a one-query corpus. The precedent is
    `tests/test_eval_ci_reporting.py::test_a_baseline_the_real_writer_produced_loads_in_the_real_reader`,
    which pins the `ci95` field the same way; a fixture asserting the shape of
    `main()`'s `out` dict would only have proved that a string is in a source
    file. Two claims, both read off the written file:

    1. `semantic_seeding` is a TOP-LEVEL key of the artifact and equals what
       production's accessor reports in this process — the subprocess read the
       same `config.yaml`, so agreement is a cross-process fact and not the same
       function agreeing with itself. It is absent from `summary.overall`,
       because a nested landing would be invisible to the reader, and no
       `run_config` wrapper exists to nest it in.
    2. The real reader bands that real file as its own seeding regime: pointed at
       the directory, the gate's published `shape` names `enabled=True,k=3` (or
       whatever this box's config says), and dropping a keyless pre-#1547 night
       beside it leaves the pool at one night. `nightly-`-prefixed label so the
       file counts as a night at all (#1220).
    """
    import json
    import os
    import subprocess

    from agent_mcp.retrieval import semantic_seeding_record

    label = "nightly-seamprobe1547"
    data = tmp_path / "data"
    queries = tmp_path / "q.yaml"
    queries.write_text("queries:\n"
                       "  - id: seam-probe\n    query: what is lloyd\n"
                       "    category: single\n    expect_entities: [Lloyd]\n"
                       "    expect_docs: [lloyd]\n")
    store, facts = tmp_path / "kg.sqlite", tmp_path / "facts"
    facts.mkdir()
    env = dict(os.environ, PYTHONPATH=str(ROOT), LLOYD_DATA=str(data),
               LLOYD_KG_DB=str(store), LLOYD_FACTS_ROOT=str(facts),
               LLOYD_VOICE_ALERTS="0")
    subprocess.run([sys.executable, "-c",
                    f"from app.kg_store import KGStore; KGStore({str(store)!r}).close()"],
                   cwd=ROOT, env=env, capture_output=True, check=True)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "eval" / "run_eval.py"),
         "--queries", str(queries), "--label", label, "--allow-empty-corpus"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr

    out_dir = data / "eval" / "baselines"
    written = sorted(out_dir.glob(f"*{label}*.json"))
    assert len(written) == 1, written
    blob = json.loads(written[0].read_text())

    rec = semantic_seeding_record()
    assert blob["semantic_seeding"] == rec, (
        f"the subprocess wrote {blob.get('semantic_seeding')!r} and production's "
        f"accessor says {rec!r} in this process: the artifact and the recall "
        "would be reporting different seedings")
    assert set(blob["semantic_seeding"]) == {"enabled", "k"}, (
        "the record must say both things the reader bands by, not one")
    overall = blob["summary"]["overall"]
    assert "semantic_seeding" not in overall and "run_config" not in blob, (
        "the field landed nested, where retrieval_gate never looks")

    import app.uptake as uptake

    gate = uptake.retrieval_gate(baselines_dir=out_dir)
    assert gate["shape"]["semantic_seeding"] == f"enabled={rec['enabled']},k={rec['k']}", (
        gate["shape"])
    assert gate["nights"] == 1, gate
    (out_dir / "nightly-20260925-keyless.json").write_text(json.dumps({
        "label": "nightly-20260925-keyless", "limit": blob["limit"],
        "matches_production_defaults": True,
        "summary": {"overall": {"n_queries": 20, "doc_hit_rate": 0.62,
                                "ndcg10": 0.57}}}))
    assert uptake.retrieval_gate(baselines_dir=out_dir)["nights"] == 1, (
        "a real #1547 artifact banded with a pre-#1547 night")


def test_the_artifact_spread_is_the_only_place_the_seeding_is_written():
    """#1547 clause 1, the shape of the write: the flat landing comes from the
    `**build_run_config(args)` spread inside `main()`'s `out` dict and nowhere
    else, so there is one source for the field the reader bands by.
    """
    out_block = inspect.getsource(ev.main).split("out = {", 1)[1].split("\n    }", 1)[0]
    assert "**build_run_config(args)," in out_block, (
        "the run config is no longer spread into the artifact's top level, so "
        "`semantic_seeding` would not reach a reader")
    assert '"semantic_seeding"' not in out_block, (
        "the field is restated at the call site instead of coming from "
        "build_run_config, which is a second source for the same knob")
    cfg = ev.build_run_config(ev.build_parser().parse_args([]))
    assert isinstance(cfg["semantic_seeding"], dict)
    assert set(cfg["semantic_seeding"]) == {"enabled", "k"}, (
        "the record must say both things the reader bands by, not one")
