"""vault_search's opt-in recall pipeline and its configurable consolidate default.

`legacy` stays the default and byte-for-byte what vault_search ran before
(tests/test_recall_first_stage.py pins its AND keywords). `recall` is
vault_recall's document leg — OR keywords, djev ordering with the cross-encoder
as fallback — without facts, adopted only once eval/run_vault_search_eval.py says so.
"""
from __future__ import annotations

import pytest

from agent_mcp import vault
from app import djev, qmd_health


def _rows(n=4):
    return [{"file": f"qmd://knowledge/n{i}.md", "title": f"Note {i}",
             "snippet": f"{i + 10}: @@ -{i + 10},2 @@ (1 before, 1 after)\n{i + 11}: body {i}",
             "score": 0.9 - i / 10} for i in range(n)]


@pytest.fixture
def wire(monkeypatch):
    sent: list[dict] = []
    monkeypatch.setattr(vault, "_qmd_post", lambda payload: sent.append(payload) or _rows())
    monkeypatch.setattr(vault, "_grep_lloyd_code", lambda *a, **k: [])
    return sent


def test_the_default_is_legacy_and_unchanged(wire, monkeypatch):
    monkeypatch.setattr(vault, "_vault_search_cfg", lambda: {})
    out = vault._vault_search({"query": "gearbox backlash", "consolidate": False})
    assert out["mode"] == "hybrid"
    assert all("lexMode" not in p for p in wire), "legacy sends no lexMode (AND)"
    assert "title" not in out["results"][0]


def test_recall_pipeline_uses_the_recall_doc_leg_and_djev(wire, monkeypatch):
    monkeypatch.setattr(vault, "_vault_search_cfg", lambda: {"pipeline": "recall"})
    monkeypatch.setattr(vault, "RECALL_RERANKER", "djev")
    monkeypatch.setattr(djev, "enabled", lambda: True)
    qmd_health._reset_for_tests()
    monkeypatch.setattr(djev, "rank", lambda q, cands, **kw: [
        {"index": i, "score": (i + 1) / len(cands)} for i in reversed(range(len(cands)))])
    out = vault._vault_search({"query": "gearbox backlash", "consolidate": False})
    assert out["mode"] == "recall"
    assert wire[0].get("lexMode") == vault.RECALL_LEX_MODE
    first = out["results"][0]
    assert first["path"] == "knowledge/n3.md", "djev's order is kept, not re-sorted by qmd score"
    assert first["title"] == "Note 3" and first["line"] == 13 and first["snippet"] == "body 3"


def test_recall_pipeline_falls_back_to_the_cross_encoder(wire, monkeypatch):
    monkeypatch.setattr(vault, "_vault_search_cfg", lambda: {"pipeline": "recall"})
    monkeypatch.setattr(vault, "RECALL_RERANKER", "djev")
    monkeypatch.setattr(djev, "enabled", lambda: True)
    monkeypatch.setattr(djev, "rank", lambda *a, **k: None)
    monkeypatch.setattr(qmd_health, "_default_announce", lambda t, b: None)
    qmd_health._reset_for_tests()
    out = vault._vault_search({"query": "gearbox backlash", "consolidate": False})
    assert [p.get("rerank") for p in wire] == [False, True]
    assert out["results"]


def test_an_unknown_pipeline_reads_as_legacy(monkeypatch):
    monkeypatch.setattr(vault, "_vault_search_cfg", lambda: {"pipeline": "bogus"})
    assert vault.vault_search_pipeline() == "legacy"


def test_the_consolidate_default_comes_from_config(wire, monkeypatch):
    calls: list = []
    monkeypatch.setattr(vault, "_consolidate_results", lambda q, r: calls.append(1) or "sum")
    monkeypatch.setattr(vault, "_vault_search_cfg", lambda: {"consolidate_default": False})
    vault._vault_search({"query": "gearbox backlash"})
    assert calls == []
    monkeypatch.setattr(vault, "_vault_search_cfg", lambda: {})
    out = vault._vault_search({"query": "gearbox backlash"})
    assert calls == [1] and out["consolidated_summary"] == "sum" and out["results"]


def test_the_shipped_config_is_the_measured_one():
    # 2026-11-10: recall adopted on eval/run_vault_search_eval.py --compare
    # (eval/measurements/vault-search-pipeline-2026-11-10.md); consolidate unmeasured.
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    assert cfg["vault_search"]["pipeline"] == "recall"
    assert cfg["vault_search"]["consolidate_default"] is True
