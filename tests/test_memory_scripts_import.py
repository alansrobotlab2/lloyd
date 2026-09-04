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


def test_gate_checks_corpus_coverage(tmp_path):
    """Every other gate check is a RATIO, so a rebuild that stopped at 11% of
    the corpus looked exactly as clean as one that finished — 100%
    provenance, 0 duplicates, 0 contamination. Coverage is the check that
    knows the difference."""
    from app import kg_store
    kg_store.configure(tmp_path / "kg.sqlite")
    try:
        m = _load(MEMORY / "kg_rebuild.py")
        assert "corpus_coverage_pct" in m.GATE
        assert m.GATE["corpus_coverage_pct"] >= 95.0
        # The denominator comes from the extractor's own corpus selection, so
        # the two cannot drift apart.
        assert m._corpus_size() > 1000
    finally:
        kg_store.reset()


# ── the rebuild's carry-over import ──────────────────────────────────────────
#
# These 444 facts came from conversations. Re-extraction cannot reproduce
# them, and the tree they currently live in is the one `swap` renames to
# facts-quarantine-<ts>. Everything below exists so a silent drop is
# impossible.

import json as _json
import os as _os
import subprocess as _sp
import sys as _sys

KG_REBUILD = MEMORY / "kg_rebuild.py"


def _carryover(tmp_path, facts, *, aliases=None, edges=None, experiments=True):
    carry = tmp_path / "carryover"
    (carry / "review").mkdir(parents=True)
    (carry / "facts.json").write_text(_json.dumps(facts))
    (carry / "aliases.json").write_text(_json.dumps(aliases or []))
    (carry / "edges.json").write_text(_json.dumps(edges or []))
    if experiments:
        d = carry / "Experiments" / "Exp1"
        d.mkdir(parents=True)
        (d / "Exp1-state.md").write_text(
            "---\ntype: facts\nentity: Exp1\ncategory: state\nfacts:\n"
            "- id: stat-001\n  fact: an autoresearch experiment record\n"
            "  confidence: 1.0\n  provenance: STATED\n"
            "  created_at: '2026-08-01T00:00:00+00:00'\n---\n\n# Exp1 - state\n")
    return carry


def _run_import(tmp_path, carry):
    rebuild = tmp_path / "facts-rebuild"
    rebuild.mkdir(exist_ok=True)
    env = dict(_os.environ,
               LLOYD_FACTS_ROOT=str(rebuild),
               LLOYD_KG_DB=str(tmp_path / "kg-rebuild.sqlite"))
    proc = _sp.run([_sys.executable, str(KG_REBUILD), "_import_worker",
                    "--carryover", str(carry)],
                   cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=300)
    return proc, rebuild


def test_import_carries_facts_aliases_edges_and_experiments(tmp_path):
    carry = _carryover(
        tmp_path,
        [{"entity": "Alan", "category": "preference", "fact": "prefers terse reports",
          "confidence": 0.95, "provenance": "STATED", "source_doc": "sessions/abc.json",
          "valid_at": None}],
        aliases=[{"surface": "alan robotlab", "canonical": "Alan", "kind": "semantic",
                  "origin": "manual", "report_path": None}],
        edges=[{"source": "Alan", "target": "Lloyd", "type": "uses", "confidence": 1.0,
                "provenance": "STATED", "origin": "fact_relate"}])
    proc, rebuild = _run_import(tmp_path, carry)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    stats = _json.loads(proc.stdout.strip().splitlines()[-1])
    assert stats == {"facts": 1, "rejected_junk": 0, "dropped": 0,
                     "aliases": 1, "edges": 1, "experiments": 1}

    from app.kg_store import KGStore
    st = KGStore(tmp_path / "kg-rebuild.sqlite")
    try:
        # the fact landed in the REBUILD tree, with the new ID scheme
        assert (rebuild / "Alan" / "Alan-preference.md").exists()
        row = st.facts_idx.for_entity("Alan")[0]
        assert row["fact"] == "prefers terse reports"
        assert row["fact_id"] == "pref-001"
        assert row["provenance"] == "STATED"
        assert st.aliases.resolve("Alan Robotlab") == "Alan"
        assert st.edges.find_active("Alan", "Lloyd", "uses")
        assert (rebuild / "Experiments" / "Exp1" / "Exp1-state.md").exists()
        assert st.facts_idx.for_entity("Exp1")
    finally:
        st.close()


def test_import_refuses_junk_entities_without_failing(tmp_path):
    """A pipeline-run name is a legitimate refusal, not a lost fact."""
    carry = _carryover(tmp_path, [
        {"entity": "Sweep Run 313", "category": "state", "fact": "junk", "confidence": 0.9,
         "provenance": "STATED", "source_doc": None, "valid_at": None},
        {"entity": "Alan", "category": "state", "fact": "a real one", "confidence": 0.9,
         "provenance": "STATED", "source_doc": None, "valid_at": None},
    ], experiments=False)
    proc, _ = _run_import(tmp_path, carry)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    stats = _json.loads(proc.stdout.strip().splitlines()[-1])
    assert stats["facts"] == 1 and stats["rejected_junk"] == 1 and stats["dropped"] == 0


def test_import_fails_when_a_carried_fact_cannot_land(tmp_path):
    """The whole point. A dropped fact used to be a number in a stats dict and
    the import still exited 0."""
    carry = _carryover(tmp_path, [
        {"entity": "Broken", "category": "state", "fact": "must not vanish",
         "confidence": 0.9, "provenance": "STATED", "source_doc": None, "valid_at": None},
    ], experiments=False)
    rebuild = tmp_path / "facts-rebuild"
    (rebuild / "Broken").mkdir(parents=True)
    # a target file that will not parse -> fact_add quarantines and refuses
    (rebuild / "Broken" / "Broken-state.md").write_text("---\nfacts: [unclosed\nentity: Broken\n")

    proc, _ = _run_import(tmp_path, carry)
    assert proc.returncode == 4, proc.stdout + proc.stderr
    assert "DROPPED" in proc.stdout
    dropped = _json.loads((carry / "dropped-facts.json").read_text())
    assert len(dropped) == 1 and dropped[0]["fact"] == "must not vanish"
    # and re-running after the cause is fixed lands it
    assert list((rebuild / "Broken").glob("*.corrupt-*"))
    proc2, _ = _run_import(tmp_path, carry)
    assert proc2.returncode == 0, proc2.stdout + proc2.stderr


def test_import_worker_refuses_to_run_against_the_live_tree(tmp_path):
    """The guard that stops this writing carried-over facts into the tree
    that is about to be quarantined."""
    carry = _carryover(tmp_path, [], experiments=False)
    live = tmp_path / "facts"
    live.mkdir()
    env = dict(_os.environ, LLOYD_FACTS_ROOT=str(live),
               LLOYD_KG_DB=str(tmp_path / "kg.sqlite"))
    proc = _sp.run([_sys.executable, str(KG_REBUILD), "_import_worker",
                    "--carryover", str(carry)],
                   cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 2
    assert "not the rebuild tree" in proc.stderr
