"""`vault_recall` keeps what it found when one leg fails, and hands the model clean rows.

Three properties, each a way relevant information used to be thrown away:

- a qmd outage returned a bare error with `facts: []`, though the fact leg reads
  markdown and never needed qmd;
- a djev miss re-ran the WHOLE recall (seeding, every fact read, the grep leg) to
  re-ask qmd one question in the cross-encoder's shape;
- the model received qmd's raw diff-hunk snippets and the `_`-prefixed bookkeeping
  keys, which cost context and meant nothing to it.
"""
from __future__ import annotations

import asyncio
import json
import urllib.error

import pytest

from agent_mcp import vault
from app import djev, qmd_health

FACT = {"fact": "The guardian performs every rollback", "confidence": 0.9}


@pytest.fixture
def one_entity(monkeypatch):
    reads: list[str] = []

    def fake_get(ent):
        reads.append(ent)
        return {"entity": ent, "facts": [dict(FACT)]}

    monkeypatch.setattr(vault, "extract_entities_from_query", lambda q: [("Guardian", 1.0)])
    monkeypatch.setattr(vault, "semantic_seed_k", lambda: 0)
    monkeypatch.setattr(vault, "get_facts_sync", fake_get)
    return reads


def _outage(monkeypatch):
    def fake_post(payload):
        raise urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))
    monkeypatch.setattr(vault, "_qmd_post", fake_post)


def _recall(**extra):
    return vault._vault_recall({"query": "guardian rollback policy", "grep_code": False,
                                "expand_graph": False, **extra})


def test_an_outage_keeps_the_facts_and_names_the_documents_failure(monkeypatch, one_entity):
    _outage(monkeypatch)
    out = _recall()
    assert "error" not in out, "facts were found; the reply is partial, not a failure"
    assert out["documents"] == []
    assert out["facts"] and out["facts"][0]["fact"] == FACT["fact"]
    assert "not zero matches" in out[vault.RECALL_DOCUMENTS_ERROR_KEY]


def test_an_outage_with_no_facts_is_still_an_error(monkeypatch):
    _outage(monkeypatch)
    monkeypatch.setattr(vault, "extract_entities_from_query", lambda q: [])
    out = _recall(include_facts=False)
    assert "not zero matches" in out["error"]
    assert out[vault.RECALL_DOCUMENTS_ERROR_KEY] == out["error"]


def test_a_djev_miss_re_asks_qmd_only(monkeypatch, one_entity):
    monkeypatch.setattr(vault, "RECALL_RERANKER", "djev")
    monkeypatch.setattr(djev, "enabled", lambda: True)
    monkeypatch.setattr(djev, "rank", lambda *a, **k: None)
    monkeypatch.setattr(qmd_health, "_default_announce", lambda t, b: None)
    qmd_health._reset_for_tests()
    sent: list[dict] = []

    def fake_post(payload):
        sent.append(payload)
        return [{"file": f"qmd://knowledge/guardian-{i}.md", "title": "Guardian",
                 "snippet": "1: body", "score": 0.9 - i / 10} for i in range(3)]

    monkeypatch.setattr(vault, "_qmd_post", fake_post)
    out = _recall()
    assert out["documents"] and out["facts"]
    assert one_entity == ["Guardian"], "the fact leg must not run twice on a djev miss"
    reranks = [p.get("rerank") for p in sent if p.get("fusion") == "global"]
    assert reranks == [False, True]
    assert qmd_health.stats()["djev_fallbacks"] == 1


def test_the_model_gets_clean_snippets_and_no_bookkeeping_keys(monkeypatch):
    monkeypatch.setattr(vault, "RECALL_RERANKER", "qmd")
    monkeypatch.setattr(vault, "extract_entities_from_query", lambda q: [])
    monkeypatch.setattr(vault, "semantic_seed_k", lambda: 0)
    monkeypatch.setattr(vault, "_djev_shadow_rerank", lambda *a, **k: None)
    monkeypatch.setattr(vault, "_qmd_post", lambda payload: [
        {"file": "qmd://memory/2026-11-01.md", "title": "Daily",
         "snippet": "108: @@ -108,4 @@ (107 before, 3 after)\n109: the rollback ran",
         "score": 0.9}])
    res = asyncio.run(vault.call_tool("vault_recall", {"query": "rollback ran",
                                                       "grep_code": False}))
    body = json.loads(res.content[0].text)
    doc = body["documents"][0]
    assert doc["snippet"] == "the rollback ran"
    assert doc["line"] == 108
    assert not [k for k in doc if k.startswith("_")], doc
    # In-process callers (the eval) still see the bookkeeping.
    raw = vault._vault_recall({"query": "rollback ran", "grep_code": False})
    assert "_pre_demote_score" in raw["documents"][0]
