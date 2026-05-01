"""Tests for the relationships index cache (Task #340 PR 2).

Run: .venvs/lloyd/bin/python -m tests.test_relationships_cache

Verifies:
    - mtime-based cache hit/miss
    - cross-process detection (mtime change without our save)
    - _save_relationships refreshes cache mtime
    - missing/corrupt files don't poison the cache
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_mcp import facts as memory  # noqa: E402  (post-PR5: cache moved here)


def _reset_cache():
    """Clear the relationships cache between tests."""
    memory._invalidate_relationships_cache()


def _write_index(path: Path, edges: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"edges": edges, "schema_version": 1}, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Cache hit / miss
# ---------------------------------------------------------------------------

def test_load_returns_empty_when_file_missing():
    with tempfile.TemporaryDirectory() as td:
        rel_path = Path(td) / "_relationships.json"  # does not exist
        with patch.object(memory, "RELATIONSHIPS_PATH", rel_path):
            _reset_cache()
            data = memory._load_relationships()
            assert data == {"edges": [], "schema_version": 1}
            # Missing file → cache stays None
            assert memory._relationships_cache is None


def test_load_populates_cache_on_first_call():
    with tempfile.TemporaryDirectory() as td:
        rel_path = Path(td) / "_relationships.json"
        _write_index(rel_path, [{"source": "A", "target": "B", "type": "uses"}])
        with patch.object(memory, "RELATIONSHIPS_PATH", rel_path):
            _reset_cache()
            assert memory._relationships_cache is None
            data = memory._load_relationships()
            assert len(data["edges"]) == 1
            # Cache populated
            assert memory._relationships_cache is not None
            mtime_ns, cached_data = memory._relationships_cache
            assert mtime_ns == rel_path.stat().st_mtime_ns
            assert cached_data is data  # same object reference


def test_load_returns_cached_object_on_second_call():
    """Second call with no mtime change must return the SAME object —
    no JSON re-parse."""
    with tempfile.TemporaryDirectory() as td:
        rel_path = Path(td) / "_relationships.json"
        _write_index(rel_path, [{"source": "A", "target": "B", "type": "uses"}])
        with patch.object(memory, "RELATIONSHIPS_PATH", rel_path):
            _reset_cache()
            first = memory._load_relationships()
            second = memory._load_relationships()
            assert first is second, "expected same object reference (cache hit)"


# ---------------------------------------------------------------------------
# Mtime invalidation (cross-process)
# ---------------------------------------------------------------------------

def test_load_reloads_when_mtime_changes():
    """Simulate an external writer (autonomy classifier) updating the file."""
    with tempfile.TemporaryDirectory() as td:
        rel_path = Path(td) / "_relationships.json"
        _write_index(rel_path, [{"source": "A", "target": "B", "type": "uses"}])
        with patch.object(memory, "RELATIONSHIPS_PATH", rel_path):
            _reset_cache()
            first = memory._load_relationships()
            assert len(first["edges"]) == 1

            # External writer adds an edge and bumps mtime.
            # We rewrite file contents AND advance mtime by 1 second to make
            # the change unambiguous (st_mtime_ns has nanosecond resolution
            # but on some filesystems the actual granularity is coarser).
            _write_index(rel_path, [
                {"source": "A", "target": "B", "type": "uses"},
                {"source": "C", "target": "D", "type": "depends_on"},
            ])
            # Bump mtime forward by 2s to defeat any coarse fs resolution.
            new_atime_mtime = (rel_path.stat().st_atime + 2, rel_path.stat().st_mtime + 2)
            os.utime(rel_path, new_atime_mtime)

            second = memory._load_relationships()
            assert len(second["edges"]) == 2, "expected reload after mtime bump"
            assert first is not second, "expected new object after reload"


# ---------------------------------------------------------------------------
# _save_relationships behavior
# ---------------------------------------------------------------------------

def test_save_refreshes_cache_with_new_mtime():
    with tempfile.TemporaryDirectory() as td:
        rel_path = Path(td) / "_relationships.json"
        _write_index(rel_path, [])
        with patch.object(memory, "RELATIONSHIPS_PATH", rel_path):
            _reset_cache()
            data = memory._load_relationships()
            data["edges"].append({"source": "X", "target": "Y", "type": "uses"})
            memory._save_relationships(data)

            # Cache mtime now matches the freshly-written file
            assert memory._relationships_cache is not None
            cached_mtime, cached_data = memory._relationships_cache
            assert cached_mtime == rel_path.stat().st_mtime_ns
            assert cached_data is data
            # And next load returns the same object — no reparse needed
            again = memory._load_relationships()
            assert again is data


def test_save_then_load_returns_persisted_data():
    with tempfile.TemporaryDirectory() as td:
        rel_path = Path(td) / "_relationships.json"
        _write_index(rel_path, [])
        with patch.object(memory, "RELATIONSHIPS_PATH", rel_path):
            _reset_cache()
            data = memory._load_relationships()
            data["edges"].append({"source": "X", "target": "Y", "type": "uses"})
            memory._save_relationships(data)

            # Force cache miss by clearing
            _reset_cache()
            reloaded = memory._load_relationships()
            assert len(reloaded["edges"]) == 1
            assert reloaded["edges"][0]["source"] == "X"


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def test_corrupt_json_returns_empty_does_not_poison_cache():
    with tempfile.TemporaryDirectory() as td:
        rel_path = Path(td) / "_relationships.json"
        rel_path.write_text("{not valid json", encoding="utf-8")
        with patch.object(memory, "RELATIONSHIPS_PATH", rel_path):
            _reset_cache()
            data = memory._load_relationships()
            assert data == {"edges": [], "schema_version": 1}
            # Cache should NOT hold the empty placeholder — that would mask
            # the real file once it's repaired.
            assert memory._relationships_cache is None

            # Now write valid JSON. Next load should pick it up.
            _write_index(rel_path, [{"source": "A", "target": "B", "type": "uses"}])
            data2 = memory._load_relationships()
            assert len(data2["edges"]) == 1


def test_invalidate_helper_clears_cache():
    with tempfile.TemporaryDirectory() as td:
        rel_path = Path(td) / "_relationships.json"
        _write_index(rel_path, [{"source": "A", "target": "B", "type": "uses"}])
        with patch.object(memory, "RELATIONSHIPS_PATH", rel_path):
            _reset_cache()
            memory._load_relationships()
            assert memory._relationships_cache is not None
            memory._invalidate_relationships_cache()
            assert memory._relationships_cache is None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_TESTS = [
    test_load_returns_empty_when_file_missing,
    test_load_populates_cache_on_first_call,
    test_load_returns_cached_object_on_second_call,
    test_load_reloads_when_mtime_changes,
    test_save_refreshes_cache_with_new_mtime,
    test_save_then_load_returns_persisted_data,
    test_corrupt_json_returns_empty_does_not_poison_cache,
    test_invalidate_helper_clears_cache,
]


def main() -> int:
    passed = 0
    failed: list[tuple[str, str]] = []
    for t in _TESTS:
        try:
            _reset_cache()
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
