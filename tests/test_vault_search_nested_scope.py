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


def _search(monkeypatch, scope: str, max_results: int = 10) -> tuple[dict, list]:
    calls: list = []

    def _fake(query, limit, collections, **kw):
        calls.append((limit, list(collections)))
        return [dict(r) for r in _rows()[:limit]]

    monkeypatch.setattr(vault_mod, "_qmd_daemon_search", _fake)
    monkeypatch.setattr(vault_mod, "_grep_lloyd_code", lambda *a, **k: [])
    return vault_mod._run_vault_search("talk", max_results, 0.0, scope, False), calls


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
