"""Every script under scripts/memory/ must import, and the retired write
paths must stay retired.

`semantic-entity-resolution.py` shipped with a missing `hashlib` import on
2026-09-04 and nothing caught it until the task was run by hand: these are
standalone scripts with no test coverage, invoked by an autonomy task whose
failure shows up as a run record nobody reads until the graph is wrong.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "memory"))
sys.path.insert(0, str(ROOT / "scripts" / "memory" / "next-gen-memory"))

MEMORY = ROOT / "scripts" / "memory"

# Scripts that run work on import (none should, but be explicit about it).
SKIP = set()


def _script_paths():
    out = [p for p in sorted(MEMORY.glob("*.py")) if p.name not in SKIP]
    out += [p for p in sorted((MEMORY / "next-gen-memory").glob("*.py"))]
    return out


@pytest.mark.parametrize("path", _script_paths(), ids=lambda p: p.name)
def test_script_imports(path, tmp_path, monkeypatch):
    """Import each script. A NameError or a missing import fails here rather
    than in an autonomy run at 3am."""
    monkeypatch.setenv("LLOYD_KG_DB", str(tmp_path / "kg.sqlite"))
    from app import kg_store
    kg_store.configure(tmp_path / "kg.sqlite")
    try:
        name = "script_" + path.stem.replace("-", "_")
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    finally:
        kg_store.reset()


def _load(path):
    name = "chk_" + path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_semantic_resolution_has_no_apply_path(tmp_path):
    """#67 proposes; only the sweep merges. Its apply bypassed the sweep's
    degraded-graph gate, its ledger, its fact retagging and its revert."""
    from app import kg_store
    kg_store.configure(tmp_path / "kg.sqlite")
    try:
        m = _load(MEMORY / "semantic-entity-resolution.py")
        for gone in ("apply_merge", "apply_alias_only", "save_aliases", "APPLY_LOG"):
            assert not hasattr(m, gone), f"{gone} is back"
        assert hasattr(m, "PROPOSAL_LOG") and hasattr(m, "PROPOSAL_LATEST")
        # the verdict cache the weekly run depends on
        assert m._cache_key("A", "B", "d1", "d2") == m._cache_key("B", "A", "d2", "d1")
        assert m._cache_key("A", "B", "d1", "d2") != m._cache_key("A", "B", "d1", "CHANGED")
    finally:
        kg_store.reset()


def test_v1_classifier_has_no_driver(tmp_path):
    """Running v1 against today's graph would write JSON nothing applies."""
    from app import kg_store
    kg_store.configure(tmp_path / "kg.sqlite")
    try:
        m = _load(MEMORY / "classify-relationships.py")
        assert not hasattr(m, "main"), "v1's main() is back"
        # but the helpers v4 imports must still be there
        for helper in ("_load_fact_snippets", "_resolve_entity_dir", "_load_relationships"):
            assert hasattr(m, helper), helper
    finally:
        kg_store.reset()


def test_no_script_reads_the_retired_json_files():
    """Aliases and edges live in app.kg_store. A script still reading the JSON
    would silently serve a snapshot frozen at the migration."""
    offenders = []
    for path in _script_paths():
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), 1):
            if "_relationships.json" not in line and "entity-aliases.json" not in line:
                continue
            stripped = line.strip()
            # Comments, docstrings and the migration tool may name them.
            if stripped.startswith("#") or path.name == "kg_migrate_to_sqlite.py":
                continue
            if "read_text" in line or "json.load" in line or "write_text" in line:
                offenders.append(f"{path.name}:{line_no}: {stripped}")
    assert not offenders, "scripts still touching the retired JSON:\n" + "\n".join(offenders)


def test_swap_refuses_when_a_fact_was_stated_after_the_export(tmp_path, monkeypatch):
    """The rebuild runs for hours with the system live. A fact stated in a
    chat turn meanwhile lands in the tree `swap` renames to
    facts-quarantine-<ts>, and re-extraction cannot reproduce it."""
    from app import kg_store
    db = tmp_path / "kg.sqlite"
    kg_store.configure(db)
    st = kg_store.store()
    st.facts_idx.reindex(root=tmp_path / "empty", register_entities=False) if False else None
    with st.transaction() as c:
        c.execute(
            "INSERT INTO facts_idx(entity, category, fact_id, text_hash, fact, "
            "created_at, provenance, file_path) VALUES (?,?,?,?,?,?,?,?)",
            ("Lloyd", "state", "stat-001", "h", "stated during the rebuild",
             "2030-01-01T00:00:00+00:00", "STATED", "Lloyd/Lloyd-state.md"))
    kg_store.reset()

    m = _load(MEMORY / "kg_rebuild.py")
    monkeypatch.setattr(m, "VAULT_KG_DB", db)
    state = {"export": {"exported_at": "2020-01-01T00:00:00+00:00"}}
    missed = m._facts_written_since_export(state)
    assert len(missed) == 1 and missed[0]["entity"] == "Lloyd"

    # An export taken after the fact sees nothing outstanding.
    assert m._facts_written_since_export(
        {"export": {"exported_at": "2031-01-01T00:00:00+00:00"}}) == []
    # No export recorded at all -> nothing to compare, no false alarm.
    assert m._facts_written_since_export({}) == []
