"""djev orders the recall's document pool; qmd's cross-encoder is the fallback (#1336).

`agent_mcp/vault.py` carries the measurement. Pinned here: the request the doc
leg sends when djev ranks, the shape of the djev call, the fallback that sends a
recall djev did not answer down the cross-encoder path (counted, never raised),
the kill switch, and that the regression pin warms the same request.
"""
from __future__ import annotations

import pytest

from agent_mcp import vault
from app import djev, qmd_health
from scripts.automod import evalpin


def _rows(n):
    return [{"file": f"qmd://knowledge/doc-{i}.md", "title": f"doc {i}",
             "snippet": f"snippet {i} " * 40, "score": round(1.0 / (i + 1), 4)} for i in range(n)]


@pytest.fixture
def djev_ranks(monkeypatch):
    """The djev ranker on, djev enabled, and the calls to it recorded."""
    monkeypatch.setattr(vault, "RECALL_RERANKER", "djev")
    monkeypatch.setattr(djev, "enabled", lambda: True)
    qmd_health._reset_for_tests()
    calls: list[dict] = []

    def fake_rank(query, candidates, **kw):
        calls.append({"query": query, "n": len(candidates), **kw})
        # Reverse order: an ordering nobody else would produce.
        return [{"index": i, "score": (i + 1) / len(candidates)} for i in reversed(range(len(candidates)))]

    monkeypatch.setattr(djev, "rank", fake_rank)
    return calls


@pytest.fixture
def wire(monkeypatch):
    sent: list[dict] = []

    def fake_post(payload):
        sent.append(dict(payload))
        return _rows(min(int(payload.get("limit", 0)), 32))

    monkeypatch.setattr(vault, "_qmd_post", fake_post)
    monkeypatch.setattr(vault, "_djev_shadow_rerank",
                        lambda *a, **k: pytest.fail("the shadow seam must not run when djev ranks"))
    return sent


def _recall(query="which autonomy tasks maintain the knowledge graph?"):
    return vault._vault_recall({"query": query, "limit": 20, "grep_code": False,
                                "include_facts": False, "expand_graph": False})


def _doc_legs(sent):
    return [b for b in sent if len(b["searches"]) == 2 and b.get("fusion") == "global"]


def test_the_measured_shape_fits_one_canvas():
    shape = vault.recall_doc_leg_shape("djev")
    assert shape == {"limit": 32, "candidateLimit": 20, "rerank": False,
                     "floor": {"autonomy": 2, "architecture": 2, "skills": 2}, "lexMode": "or"}
    # The fused head plus every floor row (two searches each) never passes the
    # pool, and the pool never passes the canvas split.
    worst = shape["candidateLimit"] + 2 * sum(shape["floor"].values())
    assert worst <= shape["limit"] <= djev.CANVAS_CHUNK_QUESTIONS
    assert vault.RECALL_DJEV_CHARS == 160 and vault.RECALL_DJEV_SAMPLES == 1


def test_the_doc_leg_asks_for_a_fused_head_plus_floors_with_the_cross_encoder_off(djev_ranks, wire):
    _recall()
    leg = _doc_legs(wire)[0]
    assert leg["rerank"] is False
    assert leg["candidateLimit"] == 20 and leg["limit"] == 32
    assert leg["collectionFloor"] == {"autonomy": 2, "architecture": 2, "skills": 2}


def test_djev_ranks_the_whole_pool_with_the_measured_call(djev_ranks, wire):
    out = _recall()
    assert len(djev_ranks) == 1
    call = djev_ranks[0]
    assert call["n"] == 32 and call["seam"] == "recall_rank"
    assert call["chars"] == 160 and call["samples"] == 1 and call["max_n"] == 32
    assert call["timeout"] == vault.RECALL_DJEV_TIMEOUT_S
    docs = out["documents"]
    # djev's (reversed) order is what comes back, and scores follow it so a
    # consumer that re-sorts by score keeps it.
    assert docs[0]["path"] == "knowledge/doc-31.md"
    assert [d["score"] for d in docs] == sorted((d["score"] for d in docs), reverse=True)
    assert all("_fusion_score" in d for d in docs)
    assert qmd_health.stats()["djev_ranked"] == 1


def test_a_djev_that_does_not_answer_sends_the_recall_down_the_cross_encoder_path(djev_ranks, wire, monkeypatch):
    monkeypatch.setattr(djev, "rank", lambda *a, **k: None)
    rang: list = []
    monkeypatch.setattr(qmd_health, "_default_announce", lambda t, b: rang.append(t))
    out = _recall()
    legs = _doc_legs(wire)
    assert [leg["rerank"] for leg in legs] == [False, True]
    fallback = legs[1]
    assert fallback["limit"] == fallback["candidateLimit"] == vault.RECALL_GLOBAL_DOC_POOL
    assert fallback["collectionFloor"] == vault.RECALL_COLLECTION_FLOOR
    assert out["documents"], "a fallback still answers"
    stats = qmd_health.stats()
    assert stats["djev_fallbacks"] == 1 and stats["djev_ranked"] == 0
    assert stats["last_djev_fallback_reason"]


def test_a_djev_that_raises_is_a_fallback_not_a_failed_recall(djev_ranks, wire, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("engine gone")
    monkeypatch.setattr(djev, "rank", boom)
    out = _recall()
    assert out["documents"] and qmd_health.stats()["djev_fallbacks"] == 1


def test_the_kill_switch_restores_the_cross_encoder_request(djev_ranks, wire, monkeypatch):
    monkeypatch.setattr(vault, "RECALL_RERANKER", "qmd")
    monkeypatch.setattr(vault, "_djev_shadow_rerank", lambda *a, **k: None)
    _recall()
    leg = _doc_legs(wire)[0]
    assert leg["rerank"] is True and leg["limit"] == leg["candidateLimit"] == 40
    assert leg["collectionFloor"] == {"autonomy": 5, "architecture": 5, "skills": 5}
    assert djev_ranks == []


def test_djev_switched_off_means_the_cross_encoder(djev_ranks, wire, monkeypatch):
    monkeypatch.setattr(djev, "enabled", lambda: False)
    monkeypatch.setattr(vault, "_djev_shadow_rerank", lambda *a, **k: None)
    assert vault.recall_reranker() == "qmd"
    _recall()
    assert _doc_legs(wire)[0]["rerank"] is True and djev_ranks == []


def test_a_stray_param_cannot_choose_the_ranker(djev_ranks, wire):
    vault._vault_recall({"query": "anything", "limit": 20, "grep_code": False, "include_facts": False,
                         "expand_graph": False, "reranker": "qmd"})
    assert _doc_legs(wire)[0]["rerank"] is False and len(djev_ranks) == 1


def test_the_regression_pin_warms_the_request_the_recall_sends(djev_ranks):
    body = evalpin.production_payload("a question")
    assert body["rerank"] is False and body["candidateLimit"] == 20 and body["limit"] == 32
    assert body["collectionFloor"] == {"autonomy": 2, "architecture": 2, "skills": 2}
    assert body["fusion"] == "global"


def test_rank_forwards_the_shape_and_guards_the_canvas(monkeypatch):
    seen = {}

    def fake_ask(state, questions, **kw):
        seen.update(kw, state=state, n=len(questions))
        return None

    monkeypatch.setattr(djev, "ask_sync", fake_ask)
    djev.rank("q", ["x" * 500] * 20, chars=160, samples=1, max_n=32)
    assert seen["samples"] == 1 and seen["n"] == 20
    assert "x" * 161 not in seen["state"] and "x" * 160 in seen["state"]
    with pytest.raises(ValueError):
        djev.rank("q", ["a"] * 33, max_n=32)
    with pytest.raises(ValueError):
        djev.rank("q", ["a"] * 4, max_n=djev.CANVAS_CHUNK_QUESTIONS + 1)
    with pytest.raises(ValueError):
        djev.rank("q", ["a"] * 17)  # the default cap is unchanged for every other caller


def test_fallbacks_are_announced_once_per_cooldown():
    qmd_health._reset_for_tests()
    rang: list = []
    for t in (0, 10, 20):
        qmd_health.note_ranker(False, "down", announce=lambda a, b: rang.append(a), now=1_800_000_000.0 + t)
    import time
    time.sleep(0.05)
    assert qmd_health.stats()["djev_fallbacks"] == 3 and len(rang) == 1
