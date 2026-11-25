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


def test_plans_is_a_member_exactly_once_under_every_architecture_mode(restore):
    """#4276 (2026-11-24): `plans` joined the fan-out, and the swap leaves it alone.

    The clause is membership counted, not membership present: this module's own reason
    for existing is that `apply_architecture_collection` rewrites `VAULT_SEGMENTS` in
    place, so a segment appended by a mode transition arrives twice on the next
    `both`, and a segment rebuilt by a literal vanishes under `repo`. The loop is what
    makes `count("plans") == 1` a statement about the swap rather than about the source
    text. `~/obsidian/plans` is a real vault directory, so the counters list carries the
    name too — and it has to stay a separate list, or the swap would start walking the
    checkout's tree.
    """
    assert vault.VAULT_SEGMENTS.count("plans") == 1
    for mode in ("both", "repo", "vault", "both"):
        vault.apply_architecture_collection(mode)
        assert vault.VAULT_SEGMENTS.count("plans") == 1, mode
    assert "plans" in vault.VAULT_DIR_SEGMENTS
    assert vault.VAULT_DIR_SEGMENTS is not vault.VAULT_SEGMENTS


# ── #4392: one export of the mode, so no importer re-maps it ──────────────────
#
# The mapping used to be written twice: here, and as a `== "repo"` ternary in
# `app/prefetch.py`. That second copy honoured one of the three modes, so the
# deployed `both` reached this module's fan-out and never the leg that injects
# `<vault-context>` into every turn — which went on searching the vault's stale
# clone of the architecture docs beside none of the checkout's live ones. These pin
# the single export both sides read, and that the fan-out is built from it.

def test_architecture_collections_names_the_two_corpora_of_each_mode():
    assert vault.architecture_collections("vault") == ["architecture"]
    assert vault.architecture_collections("repo") == ["lloyd-architecture"]
    # `both` in this order: the vault clone holds the slot, the checkout is appended.
    assert vault.architecture_collections("both") == ["architecture", "lloyd-architecture"]
    # An unrecognised value falls back the way `architecture_collection_mode`
    # falls back — to the vault's copy, never to an empty fan-out.
    assert vault.architecture_collections("sideways") == ["architecture"]


def test_architecture_collections_without_a_mode_reads_the_deployed_switch():
    assert vault.architecture_collection_mode() == "both"
    assert vault.architecture_collections() == ["architecture", "lloyd-architecture"]


def test_the_fan_out_names_exactly_what_the_export_names(restore):
    for mode in ("vault", "repo", "both"):
        vault.apply_architecture_collection(mode)
        named = [s for s in vault.VAULT_SEGMENTS
                 if s in ("architecture", "lloyd-architecture")]
        assert named == vault.architecture_collections(mode), mode
    # Two transitions the loop above cannot catch, because its last arm is `both`.
    # Re-applying `both` must not name the checkout's docs a second time, and
    # leaving `both` for a single-name mode must drop the appended one — a fan-out
    # that only ever appended would pass every loop assertion and then ask the
    # daemon for the same collection twice, or keep searching a corpus the switch
    # has stopped naming.
    both_shape = list(vault.VAULT_SEGMENTS)
    vault.apply_architecture_collection("both")
    assert vault.VAULT_SEGMENTS == both_shape
    vault.apply_architecture_collection("vault")
    assert [s for s in vault.VAULT_SEGMENTS
            if s in ("architecture", "lloyd-architecture")] == ["architecture"]
    # The floors follow the fan-out's architecture slot, never the appended name:
    # under `both` the checkout's docs are searched with no floor, which is the
    # shape the #1662 A/B measured (+0.061 mrr_doc over the 79 doc-gold queries).
    vault.apply_architecture_collection("repo")
    assert set(vault.RECALL_DJEV_FLOOR) == {"autonomy", "lloyd-architecture", "skills"}
    assert set(vault.RECALL_COLLECTION_FLOOR) == {"autonomy", "lloyd-architecture", "skills"}
    vault.apply_architecture_collection("both")
    assert set(vault.RECALL_DJEV_FLOOR) == {"autonomy", "architecture", "skills"}
    assert set(vault.RECALL_COLLECTION_FLOOR) == {"autonomy", "architecture", "skills"}
    assert vault.RECALL_DJEV_FLOOR["architecture"] == 2
    assert vault.RECALL_COLLECTION_FLOOR["architecture"] == 5


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
