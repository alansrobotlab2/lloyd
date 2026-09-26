"""#1456: prefetch's topic phrases merged into `vault_recall`, behind a default-off flag.

Pins the flag's default and its reading, that the merge touches the recall tool
path only, and the fusion's two shapes (`facts`: documents untouched, fact lists
fused; `full`: documents fused too), all with the base recall stubbed — no qmd,
no djev, no primary.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_mcp import vault
from app import config

ROOT = Path(__file__).resolve().parents[1]


def _doc(p):
    return {"path": p, "title": p, "snippet": "", "score": 0.5}


def _fact(e, i):
    return {"entity": e, "id": f"{e}-{i}", "fact": f"{e} fact {i}"}


RAW = {"documents": [_doc("a.md"), _doc("b.md"), _doc("c.md")],
       "facts": [_fact("Lloyd", 1), _fact("Lloyd", 2)],
       "query": "q", "n_fact_reads_failed": 0, "fact_read_first_error": None}


@pytest.fixture
def base_calls(monkeypatch):
    calls = []

    def fake_base(params, *, seed_top_k=None, reranker=None, facts_only=False):
        calls.append({"query": params["query"], "facts_only": facts_only,
                      "reranker": reranker})
        if params["query"] == "q":
            return {k: (list(v) if isinstance(v, list) else v) for k, v in RAW.items()}
        t = params["query"]
        return {"documents": [] if facts_only else [_doc(f"{t}.md"), _doc("c.md")],
                "facts": [_fact(t, 1), _fact("Lloyd", 1)],
                "graph_neighbors_used": [{"entity": t, "weight": 1.0}]}

    monkeypatch.setattr(vault, "_vault_recall_base", fake_base)
    return calls


def _mode(monkeypatch, value):
    monkeypatch.setitem(config.CONFIG, "vault_recall",
                        {**(config.CONFIG.get("vault_recall") or {}), "topics_merge": value})


def test_the_flag_defaults_off_in_code_and_the_tracked_config_names_a_mode():
    """Off unless config says otherwise; config.yaml turns on the measured
    "facts" shape only (eval/measurements/recall-topics-merge-2026-09-25.md)."""
    assert vault.RECALL_TOPICS_MERGE_DEFAULT == "off"
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    assert (cfg.get("vault_recall") or {}).get("topics_merge") in ("off", "facts")


def test_no_config_block_means_off(monkeypatch):
    monkeypatch.delitem(config.CONFIG, "vault_recall", raising=False)
    assert vault.recall_topics_merge_mode() == "off"


@pytest.mark.parametrize("value, want", [(None, "off"), ("off", "off"), ("facts", "facts"),
                                         ("FULL", "full"), ("on", "off"), (True, "off")])
def test_the_mode_reads_config_and_anything_unknown_is_off(monkeypatch, value, want):
    _mode(monkeypatch, value)
    assert vault.recall_topics_merge_mode() == want


def test_off_is_the_base_recall_and_never_drafts(monkeypatch, base_calls):
    _mode(monkeypatch, "off")
    monkeypatch.setattr(vault, "_draft_recall_topics",
                        lambda q: pytest.fail("drafted with the merge off"))
    out = vault._vault_recall({"query": "q"})
    assert out == RAW
    assert base_calls == [{"query": "q", "facts_only": False, "reranker": None}]


def test_a_param_cannot_switch_the_merge_on(monkeypatch, base_calls):
    _mode(monkeypatch, "off")
    monkeypatch.setattr(vault, "_draft_recall_topics",
                        lambda q: pytest.fail("a client key reached the merge"))
    vault._vault_recall({"query": "q", "topics_merge": "full"})


def test_the_djev_fallback_recursion_never_drafts(monkeypatch, base_calls):
    _mode(monkeypatch, "full")
    monkeypatch.setattr(vault, "_draft_recall_topics",
                        lambda q: pytest.fail("the fallback drafted a second time"))
    vault._vault_recall({"query": "q"}, reranker="qmd")
    assert base_calls[0]["reranker"] == "qmd"


def test_facts_mode_fuses_fact_lists_and_keeps_the_raw_documents(monkeypatch, base_calls):
    _mode(monkeypatch, "facts")
    monkeypatch.setattr(vault, "_draft_recall_topics", lambda q: ["Alpha", "Beta"])
    out = vault._vault_recall({"query": "q"})
    assert [d["path"] for d in out["documents"]] == ["a.md", "b.md", "c.md"]
    topic_calls = [c for c in base_calls if c["query"] != "q"]
    assert {c["query"] for c in topic_calls} == {"Alpha", "Beta"}
    assert all(c["facts_only"] for c in topic_calls)
    ents = [f["entity"] for f in out["facts"]]
    # Lloyd-1 is in all three lists and wins; the cut is one recall's length (2).
    assert out["facts"][0]["id"] == "Lloyd-1"
    assert len(out["facts"]) == 2 and set(ents) <= {"Lloyd", "Alpha", "Beta"}
    assert out["topics_merge"]["mode"] == "facts"
    assert out["topics_merge"]["topics"] == ["Alpha", "Beta"]
    # The raw recall had no neighbours; fused, the list is still one recall long.
    assert [n["entity"] for n in out["graph_neighbors_used"]] == ["Alpha"]


def test_full_mode_fuses_documents_too(monkeypatch, base_calls):
    _mode(monkeypatch, "full")
    monkeypatch.setattr(vault, "_draft_recall_topics", lambda q: ["Alpha"])
    out = vault._vault_recall({"query": "q"})
    paths = [d["path"] for d in out["documents"]]
    assert len(paths) == 3                       # cut to one recall's length
    assert paths[0] == "c.md"                    # in both lists
    assert not [c for c in base_calls if c["query"] == "Alpha"][0]["facts_only"]


def test_no_topics_or_an_errored_raw_recall_returns_the_raw_result(monkeypatch, base_calls):
    _mode(monkeypatch, "full")
    monkeypatch.setattr(vault, "_draft_recall_topics", lambda q: [])
    assert vault._vault_recall({"query": "q"}) == RAW


def test_the_draft_drops_the_query_itself_and_caps_at_three(monkeypatch):
    import app.secondary_models as sm
    monkeypatch.setattr(sm, "_sync_secondary_focus_extraction",
                        lambda text: ["the query", "A b", "a B", "C d", "E f", "G h"])
    assert vault._draft_recall_topics("the query") == ["A b", "C d", "E f"]


def test_a_failing_drafter_costs_the_merge_not_the_recall(monkeypatch):
    import app.secondary_models as sm

    def boom(text):
        raise RuntimeError("engine down")
    monkeypatch.setattr(sm, "_sync_secondary_focus_extraction", boom)
    assert vault._draft_recall_topics("q") == []


def test_prefetch_does_not_reach_the_merge():
    """The merge is the recall TOOL's; prefetch has its own 300 ms path."""
    src = (ROOT / "prefetch.py").read_text()
    assert "topics_merge" not in src and "_vault_recall(" not in src
