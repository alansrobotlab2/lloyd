"""`vault_search.architecture`: which architecture docs recall reaches (2026-11-10).

`vault` is the qmd `architecture` collection (~/obsidian/architecture, April-era
copies and retired docs); `repo` is `lloyd-architecture`, the checkout's own
`architecture/*.md`. One in-place swap so every importer of the lists agrees.
"""
from __future__ import annotations

import pytest

from agent_mcp import vault


@pytest.fixture
def restore():
    before = vault.architecture_collection_mode()
    yield
    vault.apply_architecture_collection(before)


def test_repo_mode_names_the_checkout_collection_everywhere(restore):
    vault.apply_architecture_collection("repo")
    assert "lloyd-architecture" in vault.VAULT_SEGMENTS
    assert "architecture" not in vault.VAULT_SEGMENTS
    assert "lloyd-architecture" in vault.RECALL_DJEV_FLOOR
    assert "lloyd-architecture" in vault.RECALL_COLLECTION_FLOOR
    # the counters keep walking the vault's own directories
    assert "architecture" in vault.VAULT_DIR_SEGMENTS
    assert "lloyd-architecture" not in vault.VAULT_DIR_SEGMENTS


def test_the_swap_is_reversible_and_idempotent(restore):
    for mode in ("repo", "repo", "vault", "vault", "repo"):
        vault.apply_architecture_collection(mode)
    vault.apply_architecture_collection("vault")
    assert vault.VAULT_SEGMENTS.count("architecture") == 1
    assert "lloyd-architecture" not in vault.VAULT_SEGMENTS
    assert set(vault.RECALL_DJEV_FLOOR) == {"autonomy", "architecture", "skills"}


def test_both_mode_asks_for_the_two_and_floors_only_the_vault_copy(restore):
    vault.apply_architecture_collection("both")
    assert "architecture" in vault.VAULT_SEGMENTS
    assert vault.VAULT_SEGMENTS.count("lloyd-architecture") == 1
    assert "architecture" in vault.RECALL_DJEV_FLOOR
    assert "lloyd-architecture" not in vault.RECALL_DJEV_FLOOR
    assert "lloyd-architecture" not in vault.VAULT_DIR_SEGMENTS
    for mode in ("both", "repo", "both", "vault"):
        vault.apply_architecture_collection(mode)
    assert "lloyd-architecture" not in vault.VAULT_SEGMENTS
    assert vault.VAULT_SEGMENTS.count("architecture") == 1


def test_the_deployed_mode_is_both():
    assert vault.architecture_collection_mode() == "both"
    assert "lloyd-architecture" in vault.VAULT_SEGMENTS


def test_vault_read_serves_a_checkout_doc_read_only():
    out = vault._vault_read({"path": "lloyd-architecture/index.md", "num_lines": 3})
    assert "error" not in out, out
    assert out["text"].strip()
    assert "checkout" in out["source"]


def test_vault_read_refuses_to_leave_the_docs_directory():
    out = vault._vault_read({"path": "lloyd-architecture/../config.yaml"})
    assert "error" in out


def test_vault_write_refuses_a_checkout_doc():
    out = vault._vault_write({"path": "lloyd-architecture/index.md", "content": "x"})
    assert "error" in out
    assert "checkout" in str(out)


# The entity-vector index's staleness reaches the recall's result (2026-11-10, item 10).
def test_a_stale_seed_index_is_named_on_the_result(monkeypatch):
    from app import entity_linker
    monkeypatch.setattr(entity_linker, "index_status",
                        lambda: {"stale": True, "age_days": 46.2, "max_age_days": 10})
    assert vault._seed_index_stale() == {
        "age_days": 46.2, "max_age_days": 10,
        "refresh": "scripts/memory/build_entity_vectors.py (job entity-vectors-rebuild)"}
    monkeypatch.setattr(entity_linker, "index_status",
                        lambda: {"stale": False, "age_days": 0.1, "max_age_days": 10})
    assert vault._seed_index_stale() is None
    monkeypatch.setattr(entity_linker, "index_status", lambda: None)
    assert vault._seed_index_stale() is None
