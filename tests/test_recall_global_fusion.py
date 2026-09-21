"""The recall's default request: global fusion over a small, measured pool.

`agent_mcp/vault.py` carries the eval that moved it (2026-09-19). What is pinned
here is the request itself, because the daemon cannot tell a client that forgot
half of it from one that meant to: `fusion: "global"` without the pool is 240
rows reranked for nothing, and the pool without `fusion` is per-collection
fusion at 40 rows — the worst arm measured (doc_hit 0.80).
"""
from __future__ import annotations

import pytest

from agent_mcp import vault


@pytest.fixture
def wire(monkeypatch):
    sent: list[dict] = []
    monkeypatch.setattr(vault, "_qmd_post", lambda payload: sent.append(dict(payload)) or [])
    return sent


def _recall(query="which autonomy tasks maintain the knowledge graph?"):
    vault._vault_recall({"query": query, "limit": 20, "grep_code": False,
                         "include_facts": False, "expand_graph": False})


def test_the_doc_leg_sends_fusion_pool_and_floor_together(wire):
    _recall()
    leg = [b for b in wire if len(b["searches"]) == 2][0]
    assert leg["fusion"] == "global"
    assert leg["limit"] == leg["candidateLimit"] == vault.RECALL_GLOBAL_DOC_POOL == vault.recall_doc_pool()
    # The measured floor (#1335, 87-query pinned eval): the three small
    # collections global fusion outscores wholesale.
    assert leg["collectionFloor"] == {"autonomy": 5, "architecture": 5, "skills": 5}
    assert leg["collections"] == list(vault.VAULT_SEGMENTS) and leg["rerank"] is True


def test_the_measured_pool_is_not_widened_by_the_small_caller_factor(wire):
    """`QMD_POOL_FACTOR` exists so a limit-5 caller ranks out of 15. Applied to the
    doc leg it turns the measured 40 into 120, three times the rerank for a pool
    nobody evaluated."""
    _recall()
    leg = [b for b in wire if len(b["searches"]) == 2][0]
    assert leg["limit"] == 40 < 40 * vault.QMD_POOL_FACTOR


def test_the_kill_switch_restores_the_old_request_exactly(wire, monkeypatch):
    monkeypatch.setattr(vault, "RECALL_QMD_FUSION", "collection")
    _recall()
    leg = [b for b in wire if len(b["searches"]) == 2][0]
    assert "fusion" not in leg and "collectionFloor" not in leg
    assert leg["limit"] == leg["candidateLimit"] == vault.RECALL_DOC_POOL == vault.QMD_POOL_MAX


def test_a_single_collection_search_has_nothing_to_fuse(wire):
    vault._qmd_daemon_search("anything at all", 10, ["backlog"])
    assert "fusion" not in wire[0] and "collectionFloor" not in wire[0]


def test_a_floor_is_only_sent_for_collections_in_the_request(wire):
    vault._qmd_daemon_search("anything at all", 10, ["memory", "knowledge"])
    assert wire[0]["fusion"] == "global"
    assert "collectionFloor" not in wire[0], "autonomy is not in this request"
    vault._qmd_daemon_search("anything at all", 10, ["memory", "autonomy"])
    assert wire[1]["collectionFloor"] == {"autonomy": 5}
    vault._qmd_daemon_search("anything at all", 10, ["knowledge", "skills", "architecture"])
    assert wire[2]["collectionFloor"] == {"skills": 5, "architecture": 5}


def test_other_callers_keep_their_own_pool_arithmetic(wire):
    """Only the doc leg names an exact pool; `vault_search` and the entity lookup
    still take the factor, capped at the ceiling."""
    vault._qmd_daemon_search("anything at all", 5, list(vault.VAULT_SEGMENTS))
    assert wire[0]["limit"] == 5 * vault.QMD_POOL_FACTOR
    vault._qmd_daemon_search("anything at all", 4 * vault.QMD_POOL_MAX, list(vault.VAULT_SEGMENTS))
    assert wire[1]["limit"] == vault.QMD_POOL_MAX
