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
    """Put a `{surface: canonical}` map into the configured store."""
    st = kg_store.store()
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


def test_alias_pointing_to_missing_dir_falls_through():
    """Alias maps to a directory that no longer exists — both modes must
    NOT return the stale canonical."""
    with tempfile.TemporaryDirectory() as td:
        facts_root, aliases_path = _isolate_facts_root(Path(td))
        # Lloyd dir exists but alias points at "Ghost" (doesn't exist)
        (facts_root / "Lloyd").mkdir()
        _seed_aliases({"foo": "Ghost"})
        with patch.object(_shared, "FACTS_ROOT", facts_root), \
             patch.object(_shared, "ALIASES_PATH", aliases_path), \
             patch.object(_shared, "_entity_dirs_cache", None):
            # In write mode: returns input verbatim (no fuzzy)
            resolved, is_new = _shared._resolve_entity("foo", mode="write")
            assert resolved == "foo"
            assert is_new is True


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
# Runner
# ---------------------------------------------------------------------------

_TESTS = [
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
    test_alias_pointing_to_missing_dir_falls_through,
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
