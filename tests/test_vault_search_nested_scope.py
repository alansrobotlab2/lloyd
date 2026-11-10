"""A `vault_search` scope nested below a collection finds what is in it (#670).

`scope="knowledge/youtube/X"` is a path prefix. It used to be matched whole
against the segment names, missed, and fell back to every collection with a
`max_results`-row ask; the prefix filter then ran over a reply that the rest of
the vault had filled, and the search answered zero while hundreds of in-scope
documents matched. The fake daemon below behaves the way the real one does in
the one respect that matters: it returns only as many rows as it is asked for,
best first.
"""
from __future__ import annotations

import agent_mcp.vault as vault_mod

IN_SCOPE = "knowledge/youtube/X"


def _rows() -> list[dict]:
    # 50 better-scoring rows elsewhere in `knowledge`, then the 3 in scope.
    out = [{"file": f"qmd://knowledge/other/n{i}.md", "score": 0.9 - i * 0.001,
            "snippet": "x"} for i in range(50)]
    out += [{"file": f"qmd://{IN_SCOPE}/v{i}.md", "score": 0.5 - i * 0.01,
             "snippet": "y"} for i in range(3)]
    return out


def _search(monkeypatch, scope: str, max_results: int = 10,
            pipeline: str = "legacy") -> tuple[dict, list]:
    # The nodes below pin the legacy request's ask; the recall pipeline (the
    # deployed one since 2026-11-10) is pinned by its own nodes at the end.
    calls: list = []

    def _fake(query, limit, collections, **kw):
        calls.append((limit, list(collections)))
        return [dict(r) for r in _rows()[:limit]]

    monkeypatch.setattr(vault_mod, "_qmd_daemon_search", _fake)
    monkeypatch.setattr(vault_mod, "_grep_lloyd_code", lambda *a, **k: [])
    return vault_mod._run_vault_search("talk", max_results, 0.0, scope, False,
                                       pipeline=pipeline), calls


def test_a_nested_scope_returns_the_documents_inside_it(monkeypatch):
    got, calls = _search(monkeypatch, IN_SCOPE)
    assert [r["path"] for r in got["results"]] == [
        f"{IN_SCOPE}/v0.md", f"{IN_SCOPE}/v1.md", f"{IN_SCOPE}/v2.md"]
    # Its collection is its first path component, not all eleven.
    assert calls[0][1] == ["knowledge"]
    assert got["collections_searched"] == ["knowledge"]
    # The ask is widened for the prefix filter, within the pool ceiling.
    assert calls[0][0] > 10
    assert calls[0][0] <= vault_mod.QMD_POOL_MAX


def test_the_answer_is_still_trimmed_to_max_results(monkeypatch):
    got, _ = _search(monkeypatch, IN_SCOPE, max_results=2)
    assert len(got["results"]) == 2


def test_a_top_level_scope_keeps_its_own_ask(monkeypatch):
    got, calls = _search(monkeypatch, "knowledge", max_results=10)
    assert calls == [(10, ["knowledge"])]
    assert len(got["results"]) == 10


def test_mixed_scopes_name_each_collection_once(monkeypatch):
    _, calls = _search(monkeypatch, f"{IN_SCOPE}, knowledge/other, memory")
    assert calls[0][1] == ["knowledge", "memory"]


def test_a_scope_outside_the_segments_still_searches_everything(monkeypatch):
    _, calls = _search(monkeypatch, "sessions")
    assert calls[0][1] == vault_mod.VAULT_SEGMENTS


def test_a_scope_outside_the_indexed_segments_is_an_error_not_an_empty_list(monkeypatch):
    """`scope="sessions"` (or `plans`) names no collection. It used to search
    everything and filter the reply to nothing, so the model read "no notes"
    where the truth was "not indexed here"."""
    called: list = []
    monkeypatch.setattr(vault_mod, "_qmd_daemon_search",
                        lambda *a, **k: called.append(1) or [])
    out = vault_mod._vault_search({"query": "talk", "scope": "sessions",
                                   "consolidate": False})
    assert out.get("code") == "INVALID_PARAM", out
    assert "'sessions'" in out["error"] and "knowledge" in out["error"]
    assert out["results"] == []
    assert not called, "an invalid scope must not reach the daemon"


def test_a_mixed_scope_names_only_the_unknown_segment(monkeypatch):
    out = vault_mod._vault_search({"query": "talk", "scope": "knowledge, plans/x",
                                   "consolidate": False})
    assert "'plans'" in out["error"] and "'knowledge'" not in out["error"]


def test_nested_and_multi_segment_scopes_stay_valid():
    assert vault_mod._unknown_scope_segments("knowledge/agents") == []
    assert vault_mod._unknown_scope_segments("people, backlog/") == []
    assert vault_mod._unknown_scope_segments("") == []


# ── The recall pipeline (vault_search.pipeline: recall, 2026-11-10) ──────────
# Its pool is fixed (the doc leg's 32 rows) and ranked, so #670's failure comes
# back unless the nested prefix filters a WIDE reply before the ranker sees it.

def test_recall_pipeline_nested_scope_still_finds_what_is_inside(monkeypatch):
    got, calls = _search(monkeypatch, IN_SCOPE, pipeline="recall")
    assert [r["path"] for r in got["results"]] == [
        f"{IN_SCOPE}/v0.md", f"{IN_SCOPE}/v1.md", f"{IN_SCOPE}/v2.md"]
    assert calls[0][1] == ["knowledge"]
    assert 50 < calls[0][0] <= vault_mod.QMD_POOL_MAX, calls


def test_recall_pipeline_mixed_scope_keeps_its_top_level_rows(monkeypatch):
    got, _ = _search(monkeypatch, f"{IN_SCOPE}, knowledge/other", pipeline="recall")
    paths = [r["path"] for r in got["results"]]
    assert any(p.startswith("knowledge/other/") for p in paths), paths


def test_recall_pipeline_top_level_scope_asks_the_leg_pool(monkeypatch):
    _, calls = _search(monkeypatch, "knowledge", pipeline="recall")
    assert calls[0] == (vault_mod.recall_doc_leg_shape(
        vault_mod.recall_reranker())["limit"], ["knowledge"])

