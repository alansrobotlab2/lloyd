"""#1164: the self-question retrieval arm's measurement instrument.

Everything is stubbed — no engine, no qmd, no store: the arm's contract is
about how many retrievals it issues, how it fuses them, and where the drafted
text goes, and all three are decidable without a model.
"""
from __future__ import annotations

import pytest

from eval import self_question as sq


def _doc(p):
    return {"path": p}


def _stub_recall(calls):
    def recall(q):
        calls.append(q)
        return {"documents": [_doc(f"{q}/d{i}.md") for i in range(5)],
                "facts": [{"entity": f"E-{q}", "id": f"f{i}"} for i in range(3)]}
    return recall


def test_draft_is_capped_at_three_whatever_the_model_returns():
    ten = "\n".join(f"{i}. question number {i}?" for i in range(10))
    qs = sq.draft_questions("q", drafter=lambda q, c: ten)
    assert len(qs) == sq.MAX_QUESTIONS == 3
    assert qs[0] == "question number 0?"          # list marker stripped


def test_a_ten_question_stub_yields_exactly_three_retrievals():
    calls = []
    recall = _stub_recall(calls)
    ten = [f"question {i}" for i in range(10)]
    drafts = sq.draft_questions("q", drafter=lambda q, c: ten)
    _, lat = sq.expansion_arm(recall("raw"), drafts, recall)
    assert calls == ["raw", "question 0", "question 1", "question 2"]
    assert len(lat) == 3


def test_parse_questions_drops_blanks_and_duplicates():
    assert sq.parse_questions("- What is X?\n\n- what is x?\n* Who owns Y?\n") == [
        "What is X?", "Who owns Y?"]


def test_rrf_with_no_expansion_is_the_raw_list():
    raw = [_doc(f"r{i}") for i in range(6)]
    out = sq.rrf_fuse([raw], key=lambda d: d["path"])
    assert [d["path"] for d in out] == [d["path"] for d in raw]


def test_fusion_is_budget_equal_and_counts_displaced_raw_docs():
    raw = {"documents": [_doc(f"r{i}") for i in range(4)]}
    # Two question lists that agree on two new documents: those outrank the
    # raw tail, which is displaced — and the count says so.
    q = {"documents": [_doc("n1"), _doc("n2"), _doc("r0")]}
    fused = sq.fuse_results(raw, [q, q])
    paths = [d["path"] for d in fused["documents"]]
    assert len(paths) == 4                       # never longer than one recall
    assert paths[0] == "r0"                      # in every list: ranked first
    assert {"n1", "n2"} <= set(paths)
    assert fused["raw_docs_displaced"] == 2


def test_no_raw_hit_is_lost_when_the_fused_budget_holds_them():
    raw = {"documents": [_doc("a"), _doc("b")]}
    q = {"documents": [_doc("c"), _doc("d"), _doc("e")]}
    fused = sq.fuse_results(raw, [q])
    assert {"a", "b"} <= {d["path"] for d in fused["documents"]}
    assert fused["raw_docs_displaced"] == 0


def test_facts_fuse_by_entity_and_id_not_by_position():
    raw = {"facts": [{"entity": "vLLM", "id": "s1"}]}
    q = {"facts": [{"entity": "VLLM", "id": "s1"}, {"entity": "Qwen", "id": "s9"}]}
    fused = sq.fuse_results(raw, [q])
    assert [(f["entity"], f["id"]) for f in fused["facts"]] == [("vLLM", "s1"), ("Qwen", "s9")]


def test_an_errored_draft_recall_degrades_to_the_control():
    raw = {"documents": [_doc("a"), _doc("b")]}

    def boom(q):
        raise RuntimeError("daemon down")

    fused, lat = sq.expansion_arm(raw, ["q1", "q2"], boom)
    assert [d["path"] for d in fused["documents"]] == ["a", "b"]
    assert len(lat) == 2


def _score(spec, result, seeds):
    paths = [d["path"] for d in result.get("documents") or []]
    hit = spec["want"] in paths
    return {"doc_hit": hit, "rr_doc": (1 / (paths.index(spec["want"]) + 1)) if hit else 0.0,
            "ndcg10": 0.0, "doc_recall": 1.0 if hit else 0.0, "entity_hit": False,
            "entity_recall": None, "fact_entity_recall": None}


def test_run_scores_every_arm_on_the_same_query_and_compare_pairs_them():
    queries = [{"id": f"q{i}", "query": f"topic{i}", "want": f"q{i}-answer.md"} for i in range(4)]

    def recall(q):
        # The raw query never finds the answer; a drafted question does.
        if q.startswith("which"):
            qid = q.split()[-1]
            return {"documents": [_doc(f"{qid}-answer.md")]}
        return {"documents": [_doc("noise1.md"), _doc("noise2.md")]}

    recs = sq.run(queries, recall=recall,
                  q_drafter=lambda q, c: f"which doc answers q{q[-1]}",
                  t_drafter=lambda q, c: [],
                  score=_score)
    assert [set(r["arms"]) for r in recs] == [set(sq.ARMS)] * 4
    s = sq.compare(recs, n_resamples=200)
    assert s["arms"]["control"]["doc_hit_rate"] == 0.0
    assert s["arms"]["self_question"]["doc_hit_rate"] == 1.0
    # a tie at rank 1 breaks to the raw list, so the answer lands second
    assert s["arms"]["self_question"]["mrr_doc"] == 0.5
    d = s["paired_vs_control"]["self_question"]["doc_hit_rate"]
    assert d["delta"] == 1.0 and d["wins"] == 4 and d["losses"] == 0
    # topics drafted nothing: the arm IS the control
    assert s["paired_vs_control"]["topics"]["doc_hit_rate"]["delta"] == 0.0
    # the arm's latency is judged in the paired-check context, never the nightly's
    if s["latency_budget"] is not None:
        assert s["latency_budget"]["context"] == "paired_check"


def test_drafted_text_reaches_only_the_run_record(monkeypatch):
    """A full arm pass with every memory/vault write path raising completes."""
    import agent_mcp.facts as facts
    import agent_mcp.session as session
    import agent_mcp.vault as vault
    import app.kg_store as kg

    def refuse(*a, **k):
        raise AssertionError("drafted question text reached a write path")

    for mod, names in ((vault, ("_vault_write",)),
                       (facts, ("_fact_add", "_fact_relate")),
                       (session, ("_memory_add", "_memory_replace")),
                       (kg, ("store",))):
        for n in names:
            monkeypatch.setattr(mod, n, refuse)   # raises if the name moved
    recs = sq.run([{"id": "x", "query": "tell me about #363", "want": "a.md"}],
                  recall=lambda q: {"documents": [_doc("a.md")]},
                  q_drafter=lambda q, c: "What is backlog item 363?",
                  t_drafter=lambda q, c: ["backlog item 363"],
                  score=_score)
    assert recs[0]["arms"]["self_question"]["drafts"] == ["What is backlog item 363?"]


@pytest.mark.parametrize("bad", ["", None])
def test_empty_queries_are_skipped(bad):
    assert sq.run([{"id": "x", "query": bad}], recall=lambda q: {},
                  q_drafter=lambda q, c: "", t_drafter=lambda q, c: [], score=_score) == []
