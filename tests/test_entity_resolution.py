"""Tests for _resolve_entity read/write mode behavior (Task #340 PR 3).

Run: .venvs/lloyd/bin/python -m tests.test_entity_resolution

PR 3 fixes a silent data-corruption bug: previously, fuzzy match ran
regardless of auto_create, so fact_add(entity="Lloyd") could silently
land on a fuzzy-matched neighbour like "lloyd-mc". Reads also wrote to
the alias table as a side effect.

Tests verify:
    - mode is required and keyword-only
    - read mode: exact → alias → fuzzy (in-memory only, no disk writes)
    - write mode: exact → alias → return verbatim (NO fuzzy match)
    - alias table is never written by _resolve_entity in either mode
    - case-insensitive directory matching still works in both modes
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_mcp import _shared  # noqa: E402
from app import kg_store  # noqa: E402


def _isolate_facts_root(tmp_path: Path) -> tuple[Path, Path]:
    """Set up a tempdir as FACTS_ROOT and return (facts_root, aliases_path)."""
    facts_root = tmp_path / "facts"
    facts_root.mkdir()
    aliases_path = facts_root / "entity-aliases.json"
    return facts_root, aliases_path


def _patch_paths(facts_root: Path, aliases_path: Path):
    """Patch all path/cache module attributes for an isolated test.

    `aliases_path` is now only where a test's seed JSON lives; the alias
    lookup itself goes through app.kg_store, pointed at a sibling file.
    """
    kg_store.configure(facts_root.parent / "kg.sqlite")
    return [
        patch.object(_shared, "FACTS_ROOT", facts_root),
        patch.object(_shared, "ALIASES_PATH", aliases_path),
        patch.object(_shared, "_entity_dirs_cache", None),
    ]


def _seed_aliases(mapping: dict) -> None:
    """Put a `{surface: canonical}` map into the configured store.

    `configure`, not `store()`: seeding is provisioning, and since #1236 the
    reader refuses an absent database instead of letting sqlite invent one. The
    two tests that seed without calling `_patch_paths` land on the fresh path
    `conftest._isolate_default_store` points the default at, so this is what
    creates it; where `_patch_paths` already configured a store, `configure`
    just reopens the same file.
    """
    st = kg_store.configure(kg_store._default_path)
    for surface, canonical in mapping.items():
        if surface == canonical:
            st.entities.register(canonical)
        else:
            st.aliases.set(surface, canonical, kind="semantic", origin="test")


# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------

def test_mode_is_required():
    import inspect
    sig = inspect.signature(_shared._resolve_entity)
    mode_param = sig.parameters.get("mode")
    assert mode_param is not None, "mode parameter missing"
    assert mode_param.kind == inspect.Parameter.KEYWORD_ONLY, (
        "mode must be keyword-only"
    )
    assert mode_param.default is inspect.Parameter.empty, (
        "mode must be required (no default) — every callsite must "
        "specify read or write explicitly"
    )


def test_auto_create_removed():
    import inspect
    sig = inspect.signature(_shared._resolve_entity)
    assert "auto_create" not in sig.parameters, (
        "auto_create should be removed; use mode=write instead"
    )


def test_calling_without_mode_raises():
    try:
        _shared._resolve_entity("anything")
    except TypeError as exc:
        assert "mode" in str(exc), f"unexpected TypeError: {exc}"
    else:
        raise AssertionError("expected TypeError when mode is omitted")


# ---------------------------------------------------------------------------
# Exact match — both modes
# ---------------------------------------------------------------------------

def test_exact_match_read_mode():
    with tempfile.TemporaryDirectory() as td:
        facts_root, aliases_path = _isolate_facts_root(Path(td))
        (facts_root / "Lloyd").mkdir()
        with patch.object(_shared, "FACTS_ROOT", facts_root), \
             patch.object(_shared, "ALIASES_PATH", aliases_path), \
             patch.object(_shared, "_entity_dirs_cache", None):
            resolved, is_new = _shared._resolve_entity("Lloyd", mode="read")
            assert resolved == "Lloyd"
            assert is_new is False


def test_exact_match_write_mode():
    with tempfile.TemporaryDirectory() as td:
        facts_root, aliases_path = _isolate_facts_root(Path(td))
        (facts_root / "Lloyd").mkdir()
        with patch.object(_shared, "FACTS_ROOT", facts_root), \
             patch.object(_shared, "ALIASES_PATH", aliases_path), \
             patch.object(_shared, "_entity_dirs_cache", None):
            resolved, is_new = _shared._resolve_entity("Lloyd", mode="write")
            assert resolved == "Lloyd"
            assert is_new is False


def test_case_insensitive_match_both_modes():
    with tempfile.TemporaryDirectory() as td:
        facts_root, aliases_path = _isolate_facts_root(Path(td))
        (facts_root / "Lloyd").mkdir()
        with patch.object(_shared, "FACTS_ROOT", facts_root), \
             patch.object(_shared, "ALIASES_PATH", aliases_path), \
             patch.object(_shared, "_entity_dirs_cache", None):
            for mode in ("read", "write"):
                resolved, is_new = _shared._resolve_entity("LLOYD", mode=mode)
                assert resolved == "Lloyd", f"mode={mode}: got {resolved}"
                assert is_new is False


# ---------------------------------------------------------------------------
# Alias lookup — both modes
# ---------------------------------------------------------------------------

def test_alias_lookup_both_modes():
    with tempfile.TemporaryDirectory() as td:
        facts_root, aliases_path = _isolate_facts_root(Path(td))
        (facts_root / "Lloyd").mkdir()
        _seed_aliases({"llloyd": "Lloyd"})
        with patch.object(_shared, "FACTS_ROOT", facts_root), \
             patch.object(_shared, "ALIASES_PATH", aliases_path), \
             patch.object(_shared, "_entity_dirs_cache", None):
            for mode in ("read", "write"):
                resolved, is_new = _shared._resolve_entity("llloyd", mode=mode)
                assert resolved == "Lloyd", f"mode={mode}: got {resolved}"
                assert is_new is False


def test_alias_is_followed_even_when_the_canonical_has_no_dir_yet():
    """An alias whose canonical has no directory is still authoritative.

    It used to fall through to the caller's input, which meant a merged
    variant was recreated by the next writer: the sweep moved the files,
    the alias said `variant -> canonical`, and fact_add ignored it and made
    `variant/` again. An entity can also legitimately live only in the edge
    graph. Alias rows are now written by a named origin, so they are trusted.
    """
    with tempfile.TemporaryDirectory() as td:
        facts_root, aliases_path = _isolate_facts_root(Path(td))
        (facts_root / "Lloyd").mkdir()
        _seed_aliases({"foo": "Ghost"})
        with patch.object(_shared, "FACTS_ROOT", facts_root), \
             patch.object(_shared, "ALIASES_PATH", aliases_path), \
             patch.object(_shared, "_entity_dirs_cache", None):
            for mode in ("read", "write"):
                resolved, is_new = _shared._resolve_entity("foo", mode=mode)
                assert resolved == "Ghost", mode
                assert is_new is False, mode
            # A name with no alias and no directory still comes back verbatim.
            assert _shared._resolve_entity("bar", mode="write") == ("bar", True)


# ---------------------------------------------------------------------------
# Fuzzy match — read only
# ---------------------------------------------------------------------------

def test_read_mode_fuzzy_match_works():
    """Read mode: fuzzy match resolves to canonical name in memory."""
    with tempfile.TemporaryDirectory() as td:
        facts_root, aliases_path = _isolate_facts_root(Path(td))
        (facts_root / "Lloyd MC").mkdir()
        with patch.object(_shared, "FACTS_ROOT", facts_root), \
             patch.object(_shared, "ALIASES_PATH", aliases_path), \
             patch.object(_shared, "_entity_dirs_cache", None):
            # "lloyd-mc" should fuzzy-match "Lloyd MC" (separator variation)
            resolved, is_new = _shared._resolve_entity("lloyd-mc", mode="read")
            assert resolved == "Lloyd MC", f"expected fuzzy hit, got {resolved!r}"
            assert is_new is False


def test_read_mode_fuzzy_does_not_persist_alias():
    """Read mode must NOT write to the alias table — pure read-side."""
    with tempfile.TemporaryDirectory() as td:
        facts_root, aliases_path = _isolate_facts_root(Path(td))
        (facts_root / "Lloyd MC").mkdir()
        with patch.object(_shared, "FACTS_ROOT", facts_root), \
             patch.object(_shared, "ALIASES_PATH", aliases_path), \
             patch.object(_shared, "_entity_dirs_cache", None):
            assert not aliases_path.exists()
            _shared._resolve_entity("lloyd-mc", mode="read")
            # The alias file should NOT have been created.
            assert not aliases_path.exists(), (
                "read mode wrote to alias table — this is the bug PR 3 "
                "specifically fixes"
            )


def test_write_mode_skips_fuzzy_match():
    """The KEY data-integrity fix: write mode does NOT fuzzy-merge."""
    with tempfile.TemporaryDirectory() as td:
        facts_root, aliases_path = _isolate_facts_root(Path(td))
        (facts_root / "Lloyd MC").mkdir()
        with patch.object(_shared, "FACTS_ROOT", facts_root), \
             patch.object(_shared, "ALIASES_PATH", aliases_path), \
             patch.object(_shared, "_entity_dirs_cache", None):
            # Pre-PR 3, this would have fuzzy-matched to "Lloyd MC".
            # Post-PR 3, write mode returns the verbatim input.
            resolved, is_new = _shared._resolve_entity("lloyd-mc", mode="write")
            assert resolved == "lloyd-mc", (
                f"write mode must return verbatim input, got {resolved!r} "
                f"— the silent fuzzy-merge bug is back"
            )
            assert is_new is True


def test_write_mode_does_not_persist_anything():
    """Write mode: no disk side effects in _resolve_entity itself."""
    with tempfile.TemporaryDirectory() as td:
        facts_root, aliases_path = _isolate_facts_root(Path(td))
        with patch.object(_shared, "FACTS_ROOT", facts_root), \
             patch.object(_shared, "ALIASES_PATH", aliases_path), \
             patch.object(_shared, "_entity_dirs_cache", None):
            assert not aliases_path.exists()
            _shared._resolve_entity("BrandNew", mode="write")
            # _resolve_entity does NOT create the dir or the alias.
            # The caller (e.g. _fact_add) is responsible for creating
            # the dir on first write.
            assert not aliases_path.exists()
            assert not (facts_root / "BrandNew").exists()


# ---------------------------------------------------------------------------
# Empty / edge cases
# ---------------------------------------------------------------------------

def test_empty_string_returns_empty():
    with tempfile.TemporaryDirectory() as td:
        facts_root, aliases_path = _isolate_facts_root(Path(td))
        with patch.object(_shared, "FACTS_ROOT", facts_root), \
             patch.object(_shared, "ALIASES_PATH", aliases_path), \
             patch.object(_shared, "_entity_dirs_cache", None):
            for mode in ("read", "write"):
                resolved, is_new = _shared._resolve_entity("", mode=mode)
                assert resolved == ""
                assert is_new is True


def test_whitespace_stripped():
    with tempfile.TemporaryDirectory() as td:
        facts_root, aliases_path = _isolate_facts_root(Path(td))
        (facts_root / "Lloyd").mkdir()
        with patch.object(_shared, "FACTS_ROOT", facts_root), \
             patch.object(_shared, "ALIASES_PATH", aliases_path), \
             patch.object(_shared, "_entity_dirs_cache", None):
            resolved, is_new = _shared._resolve_entity("  Lloyd  ", mode="read")
            assert resolved == "Lloyd"
            assert is_new is False


def test_unknown_entity_write_mode_returns_verbatim():
    """Critical: write mode returns the literal input for unknown entities,
    so callers (_fact_add) write to that exact name."""
    with tempfile.TemporaryDirectory() as td:
        facts_root, aliases_path = _isolate_facts_root(Path(td))
        with patch.object(_shared, "FACTS_ROOT", facts_root), \
             patch.object(_shared, "ALIASES_PATH", aliases_path), \
             patch.object(_shared, "_entity_dirs_cache", None):
            resolved, is_new = _shared._resolve_entity("MyNewEntity", mode="write")
            assert resolved == "MyNewEntity"
            assert is_new is True


# ---------------------------------------------------------------------------
# Index keys (#957)
# ---------------------------------------------------------------------------

def test_index_keys_follow_declared_aliases_and_never_name_shape():
    """Canonicalising `facts_idx` keys folds declared aliases, never name-shape.

    `Voice-Loop`, `Voice Pipeline` and `voice` are three distinct systems — the
    `06f0e41` guard, after the sweep's suffix tier fused 151 such pairs — and no
    alias row joins them, so the index must keep three keys even though every
    normaliser in the repo calls them one cluster. A declared `vllm` → `vLLM`
    row does fold: with no row the literal key stands, with one the canonical
    replaces it. That asymmetry is the whole contract.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "facts"
        st = kg_store.configure(Path(td) / "kg.sqlite")
        for name in ("Voice-Loop", "Voice Pipeline", "voice", "vLLM"):
            tag = "vllm" if name == "vLLM" else name      # tagged with the variant
            d = root / name
            d.mkdir(parents=True)
            (d / f"{name}-state.md").write_text(
                "---\n"
                f"type: facts\nentity: {tag}\ncategory: state\nfacts:\n"
                f"- id: {name}-001\n  fact: says something about {name}\n"
                "---\n\n# x\n", encoding="utf-8")

        def keys() -> set[str]:
            return {r["entity"] for r in
                    st.conn.execute("SELECT DISTINCT entity FROM facts_idx")}

        st.facts_idx.reindex(root=root)
        assert keys() == {"Voice-Loop", "Voice Pipeline", "voice", "vllm"}

        st.aliases.set("vllm", "vLLM", kind="case", origin="test")
        st.facts_idx.reindex(root=root)
        assert keys() == {"Voice-Loop", "Voice Pipeline", "voice", "vLLM"}
        assert st.facts_idx.count(entity="vLLM", active_only=True) == 1
        assert st.facts_idx.count(entity="vllm", active_only=True) == 0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_TESTS = [
    # index keys (#957)
    test_index_keys_follow_declared_aliases_and_never_name_shape,
    # signature
    test_mode_is_required,
    test_auto_create_removed,
    test_calling_without_mode_raises,
    # exact match
    test_exact_match_read_mode,
    test_exact_match_write_mode,
    test_case_insensitive_match_both_modes,
    # alias
    test_alias_lookup_both_modes,
    test_alias_is_followed_even_when_the_canonical_has_no_dir_yet,
    # fuzzy
    test_read_mode_fuzzy_match_works,
    test_read_mode_fuzzy_does_not_persist_alias,
    test_write_mode_skips_fuzzy_match,
    test_write_mode_does_not_persist_anything,
    # edge cases
    test_empty_string_returns_empty,
    test_whitespace_stripped,
    test_unknown_entity_write_mode_returns_verbatim,
]


def main() -> int:
    passed = 0
    failed: list[tuple[str, str]] = []
    for t in _TESTS:
        try:
            t()
            passed += 1
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            failed.append((t.__name__, repr(e)))
            print(f"  FAIL  {t.__name__}: {e!r}")
    print()
    print(f"{passed}/{len(_TESTS)} passed")
    if failed:
        print()
        print("Failures:")
        for name, err in failed:
            print(f"  {name}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# fact_add's mint gate (#758) keeps write-side attach exact
# ---------------------------------------------------------------------------

def test_fact_add_through_the_mint_gate_still_performs_no_fuzzy_merge(tmp_path):
    """Gating the mint must not reopen #340: a near-miss of an existing entity
    is neither declared nor an exact store name, so it gets its own row."""
    from agent_mcp import facts, retrieval
    from app import entity_naming
    facts_root, aliases_path = _isolate_facts_root(tmp_path)
    (facts_root / "Lloyd MC").mkdir()
    entity_naming.reset_identity_schema_cache()
    patches = _patch_paths(facts_root, aliases_path) + [
        patch.object(facts, "FACTS_ROOT", facts_root),
        patch.object(retrieval, "FACTS_ROOT", facts_root),
    ]
    try:
        for p in patches:
            p.start()
        kg_store.store().entities.register("Lloyd MC", kind="system")
        out = facts._fact_add({"entity": "lloyd-mc", "category": "state",
                               "fact": "serves the dashboard"})
        assert out.get("success") is True, out
        assert out["entity"] == "lloyd-mc"
        assert "resolved_from" not in out
        assert (facts_root / "lloyd-mc" / "lloyd-mc-state.md").exists()
        assert not (facts_root / "Lloyd MC" / "Lloyd MC-state.md").exists()
        assert kg_store.store().entities.exists("lloyd-mc")
    finally:
        for p in reversed(patches):
            p.stop()
        entity_naming.reset_identity_schema_cache()
        kg_store.reset()
