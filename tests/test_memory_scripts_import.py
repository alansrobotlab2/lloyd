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
    assert stats == {"facts": 1, "already_present": 0, "rejected_junk": 0,
                     "dropped": 0, "aliases": 1, "edges": 1, "edges_dropped": 0,
                     "experiments": 1}

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


# ── swap and rollback ────────────────────────────────────────────────────────
#
# swap renames the live fact tree. A half-completed one is a broken system at
# whatever hour it happens, so both directions are exercised here rather than
# discovered in production.

def _swap_world(tmp_path, m, monkeypatch, *, gate_passed=True):
    """A fake live tree + store and a fake rebuild tree + store, with the
    module's path constants pointed at them."""
    from app.kg_store import KGStore
    derived = tmp_path / "vault-derived"
    live, rebuild = derived / "facts", derived / "facts-rebuild"
    (live / "Old").mkdir(parents=True)
    (live / "Old" / "Old-state.md").write_text(
        "---\ntype: facts\nentity: Old\ncategory: state\nfacts:\n"
        "- id: stat-001\n  fact: the old tree\n  confidence: 0.9\n---\n\n# Old\n")
    (rebuild / "New").mkdir(parents=True)
    (rebuild / "New" / "New-state.md").write_text(
        "---\ntype: facts\nentity: New\ncategory: state\nfacts:\n"
        "- id: stat-001\n  fact: the rebuilt tree\n  confidence: 0.9\n"
        "  created_at: '2026-09-04T00:00:00+00:00'\n  source_doc: knowledge/x.md\n"
        "---\n\n# New\n")

    live_db, rebuild_db = derived / "kg.sqlite", derived / "kg-rebuild.sqlite"
    a = KGStore(live_db); a.entities.register("Old"); a.close()
    b = KGStore(rebuild_db); b.entities.register("New")
    b.edges.add({"source": "New", "target": "Other", "type": "uses"}, origin="test"); b.close()

    monkeypatch.setattr(m, "VAULT_DERIVED_ROOT", derived)
    monkeypatch.setattr(m, "VAULT_FACTS_ROOT", live)
    monkeypatch.setattr(m, "REBUILD_FACTS", rebuild)
    monkeypatch.setattr(m, "VAULT_KG_DB", live_db)
    monkeypatch.setattr(m, "REBUILD_DB", rebuild_db)
    monkeypatch.setattr(m, "STATE_PATH", derived / "rebuild-state.json")
    monkeypatch.setattr(m, "_set_write_enabled", lambda enabled: None)
    m.save_state(gate=gate_passed, export={"exported_at": "2030-01-01T00:00:00+00:00"})
    return live, rebuild, live_db, rebuild_db


class _Args:
    def __init__(self, **kw):
        self.dry_run = False
        self.force = False
        self.__dict__.update(kw)


def test_swap_replaces_the_tree_and_the_store(tmp_path, monkeypatch):
    from app.kg_store import KGStore
    m = _load(KG_REBUILD)
    live, rebuild, live_db, _ = _swap_world(tmp_path, m, monkeypatch)

    assert m.cmd_swap(_Args()) == 0
    # the rebuilt tree is now the live one, the old one is quarantined
    assert (live / "New" / "New-state.md").exists()
    assert not (live / "Old").exists()
    assert not rebuild.exists()
    quarantine = list(tmp_path.glob("vault-derived/facts-quarantine-*"))
    assert len(quarantine) == 1 and (quarantine[0] / "Old" / "Old-state.md").exists()

    st = KGStore(live_db)
    try:
        assert st.entities.exists("New") and not st.entities.exists("Old")
        assert st.edges.count() == 1
        # swap reindexes, so the new tree's facts are queryable immediately
        assert [f["fact"] for f in st.facts_idx.for_entity("New")] == ["the rebuilt tree"]
        assert st.integrity_check() == "ok"
    finally:
        st.close()


def test_rollback_puts_the_old_tree_back(tmp_path, monkeypatch):
    from app.kg_store import KGStore
    m = _load(KG_REBUILD)
    live, _, live_db, _ = _swap_world(tmp_path, m, monkeypatch)
    assert m.cmd_swap(_Args()) == 0
    assert m.cmd_rollback(_Args()) == 0

    assert (live / "Old" / "Old-state.md").exists()
    assert not (live / "New").exists()
    st = KGStore(live_db)
    try:
        assert st.entities.exists("Old") and not st.entities.exists("New")
    finally:
        st.close()


def test_swap_refuses_without_a_passing_gate(tmp_path, monkeypatch):
    m = _load(KG_REBUILD)
    live, rebuild, _, _ = _swap_world(tmp_path, m, monkeypatch, gate_passed=False)
    assert m.cmd_swap(_Args()) == 2
    assert (live / "Old").exists() and rebuild.exists()


def test_swap_refuses_when_a_fact_was_stated_after_the_export(tmp_path, monkeypatch):
    from app.kg_store import KGStore
    m = _load(KG_REBUILD)
    live, rebuild, live_db, _ = _swap_world(tmp_path, m, monkeypatch)
    m.save_state(export={"exported_at": "2020-01-01T00:00:00+00:00"})
    st = KGStore(live_db)
    with st.transaction() as c:
        c.execute("INSERT INTO facts_idx(entity, category, fact_id, text_hash, fact, "
                  "created_at, provenance, file_path) VALUES (?,?,?,?,?,?,?,?)",
                  ("Old", "state", "stat-009", "h", "stated during the rebuild",
                   "2026-09-04T01:00:00+00:00", "STATED", "Old/Old-state.md"))
    st.close()

    assert m.cmd_swap(_Args()) == 3
    assert (live / "Old").exists() and rebuild.exists(), "nothing moved"


def test_swap_dry_run_moves_nothing(tmp_path, monkeypatch):
    m = _load(KG_REBUILD)
    live, rebuild, _, _ = _swap_world(tmp_path, m, monkeypatch)
    assert m.cmd_swap(_Args(dry_run=True)) == 0
    assert (live / "Old").exists() and rebuild.exists()


def test_import_is_idempotent(tmp_path):
    """`fact_add` appends with no duplicate check, so a second `import` used
    to write every carried-over fact again. Both the dropped-fact message and
    `swap`'s refusal tell you to re-run it, so the advice the tool printed was
    the thing that would corrupt the tree it was protecting."""
    facts = [{"entity": "Alan", "category": "preference", "fact": "prefers terse reports",
              "confidence": 0.95, "provenance": "STATED", "source_doc": "sessions/abc.json",
              "valid_at": None}]
    carry = _carryover(tmp_path, facts)

    first, rebuild = _run_import(tmp_path, carry)
    assert first.returncode == 0, first.stdout + first.stderr
    assert _json.loads(first.stdout.strip().splitlines()[-1])["facts"] == 1

    second, _ = _run_import(tmp_path, carry)
    assert second.returncode == 0, second.stdout + second.stderr
    again = _json.loads(second.stdout.strip().splitlines()[-1])
    assert again["facts"] == 0 and again["already_present"] == 1

    from app.kg_store import KGStore
    st = KGStore(tmp_path / "kg-rebuild.sqlite")
    try:
        assert len(st.facts_idx.for_entity("Alan")) == 1
    finally:
        st.close()


# ── #1664: carry-over EDGES, which had no guard, no report and no gate check ──
#
# The facts half of the carry-over was protected three ways and the edges half
# not at all: `swap` checked only facts for staleness, `import` counted a raised
# `ValueError` as a landed edge, and the gate had `carryover_facts` with no
# `carryover_edges`. Every test below runs on temp stores. The live store held 0
# `origin='conversation'` rows at triage (2026-09-27), so no live count can prove
# or disprove any of this.

def _carry_edge(source, target, type_, *, provenance="INFERRED", origin="conversation"):
    return {"source": source, "target": target, "type": type_, "confidence": 1.0,
            "provenance": provenance, "origin": origin}


def test_swap_refuses_when_a_carry_worthy_edge_landed_after_the_export(tmp_path, monkeypatch, capfd):
    """Clause 1 of #1664: the staleness guard at the swap looked only at facts.

    An edge written into the live store after `export` was recorded is an edge
    the export never carried, and `cmd_swap` renames that very file to
    `kg-quarantine-<ts>` and unlinks its `-wal`/`-shm` (:801-806). Task #51 is
    not in `PAUSED_TASKS` and writes through `app.kg_store` directly, so the
    rebuild's own freeze never pauses it: a rebuild that spanned a night used to
    destroy whatever that night landed, with every check reporting a clean run.
    """
    from app.kg_store import KGStore
    m = _load(KG_REBUILD)
    live, rebuild, live_db, rebuild_db = _swap_world(tmp_path, m, monkeypatch)
    m.save_state(export={"exported_at": "2020-01-01T00:00:00+00:00"})
    st = KGStore(live_db)
    st.edges.add({"source": "Alan", "target": "Lloyd", "type": "uses",
                  "provenance": "INFERRED"}, origin="conversation")
    st.edges.add({"source": "Knowledge", "target": "Vault", "type": "mentions",
                  "provenance": "STATED"}, origin="manual")
    # Negative control inside the fixture: re-extraction CAN reproduce an
    # extracted edge, so it is not carry-worthy and must not stop the swap — a
    # guard that fired here would refuse every rebuild that spanned an extract.
    st.edges.add({"source": "Doc", "target": "Entity", "type": "mentions",
                  "provenance": "EXTRACTED"}, origin="extractor")
    st.close()

    missing = m._edges_written_since_export(m.load_state())
    assert {(e["source"], e["target"], e["type"], e["origin"]) for e in missing} == {
        ("Alan", "Lloyd", "uses", "conversation"),
        ("Knowledge", "Vault", "mentions", "manual")}

    assert m.cmd_swap(_Args()) == 3
    err = capfd.readouterr().err
    assert "Alan -[uses]-> Lloyd" in err, err
    assert "Knowledge -[mentions]-> Vault" in err, err
    assert (live / "Old").exists() and rebuild.exists(), "the refusal must move nothing"

    # `--force` swaps anyway, exactly as it does for the facts guard: the operator
    # who re-ran `export` and `import` must still be able to promote.
    assert m.cmd_swap(_Args(force=True)) == 0
    assert (live / "New").exists()
    promoted = KGStore(live_db)
    assert promoted.edges.find_active("Alan", "Lloyd", "uses") is None, (
        "the forced swap was expected to lose the row it was warned about")
    promoted.close()


def test_import_fails_when_a_carried_edge_did_not_land(tmp_path):
    """Clause 2 of #1664: `except ValueError: pass` swallowed every edge drop.

    `edges.add` returns the existing id for a duplicate
    (`app/kg_store.py:709-711`), so the swallowed `ValueError` was never the
    idempotency path — it raises only on a missing endpoint/type (`:703`) or a
    self-loop (`:705`). Each one is a carried edge that is absent from the tree
    about to be promoted, and facts already get a report file, stderr and a
    non-zero exit for exactly that case.
    """
    carry = _carryover(tmp_path, [], experiments=False, edges=[
        _carry_edge("Alan", "Lloyd", "uses"),
        _carry_edge("Solo", "Solo", "uses"),                    # self-loop
        _carry_edge("Lloyd", "Vault", ""),                      # no type
    ])
    proc, _ = _run_import(tmp_path, carry)
    assert proc.returncode == 4, proc.stdout + proc.stderr

    stats = _json.loads(proc.stdout.strip().splitlines()[-1])
    assert stats["edges"] == 1, "the one well-formed edge still had to land"
    assert stats["edges_dropped"] == 2, stats

    assert "DROPPED EDGE Solo -> Solo [uses]" in proc.stderr, proc.stderr
    assert "carried-over edge(s) did not land" in proc.stderr, proc.stderr
    report = carry / "dropped-edges.json"
    assert report.is_file(), "the edge report sits beside dropped-facts.json"
    dropped = _json.loads(report.read_text())
    assert [d["source"] for d in dropped] == ["Solo", "Lloyd"], dropped
    assert "self-loop" in dropped[0]["error"], dropped[0]
    assert "source, target and type" in dropped[1]["error"], dropped[1]


def test_a_carried_edge_that_matched_an_existing_row_counts_as_landed(tmp_path):
    """Clause 2's other half: matching an active row is a landing, not a drop.

    A rebuild is re-run more often than it is run once, so an edge the store
    already holds has to be idempotent the way facts are — otherwise the exit
    this round added would fail every second `import` on a store that is fine.
    """
    edges = [_carry_edge("Alan", "Lloyd", "uses", provenance="STATED",
                         origin="fact_relate")]
    carry = _carryover(tmp_path, [], experiments=False, edges=edges)

    first, _ = _run_import(tmp_path, carry)
    assert first.returncode == 0, first.stdout + first.stderr

    second, out = _run_import(tmp_path, carry)
    assert second.returncode == 0, second.stdout + second.stderr
    stats = _json.loads(second.stdout.strip().splitlines()[-1])
    assert stats["edges"] == 1 and stats["edges_dropped"] == 0, stats
    assert not (carry / "dropped-edges.json").exists()


def _gate_world(tmp_path, m, monkeypatch, *, store_edges, carried_edges):
    """A rebuild tree and carry-over manifest with nothing wrong but the edges.

    `_corpus_size` walks the real corpus and `_hashed_count` reads a
    `_pipeline/` artifact, so both denominators are pinned: neither is what
    these tests are about, and every other check is asserted to still pass so
    the verdict cannot flip for an unrelated reason.
    """
    from app.kg_store import KGStore
    derived = tmp_path / "vault-derived"
    facts = derived / "facts-rebuild"
    (facts / "Alan").mkdir(parents=True)
    (facts / "Alan" / "Alan-state.md").write_text(
        "---\ntype: facts\nentity: Alan\ncategory: state\nfacts:\n"
        "- id: pref-001\n  fact: prefers terse reports\n  confidence: 0.95\n"
        "  provenance: STATED\n  created_at: '2026-09-01T00:00:00+00:00'\n"
        "  source_doc: knowledge/x.md\n---\n\n# Alan - state\n")
    carry = tmp_path / "carryover"
    carry.mkdir()
    (carry / "facts.json").write_text("[]")
    (carry / "aliases.json").write_text("[]")
    (carry / "edges.json").write_text(_json.dumps(carried_edges))

    monkeypatch.setattr(m, "REBUILD_FACTS", facts)
    monkeypatch.setattr(m, "REBUILD_DB", derived / "kg-rebuild.sqlite")
    monkeypatch.setattr(m, "VAULT_DERIVED_ROOT", derived)
    monkeypatch.setattr(m, "STATE_PATH", derived / "rebuild-state.json")
    monkeypatch.setattr(m, "RUN_ROOT", tmp_path / "runs")
    monkeypatch.setattr(m, "_corpus_size", lambda: 1)
    monkeypatch.setattr(m, "_hashed_count", lambda: 1)

    st = KGStore(derived / "kg-rebuild.sqlite")
    st.entities.register("Alan")
    st.entities.register("Lloyd")
    for e in store_edges:
        st.edges.add({"source": e["source"], "target": e["target"],
                      "type": e["type"], "provenance": e["provenance"],
                      "confidence": 1.0}, origin=e["origin"])
    st.facts_idx.reindex(root=facts)
    st.close()
    m.save_state(carryover=str(carry), run_dir=str(tmp_path / "run"))
    return tmp_path / "run" / "gate.json"


def _gate_checks(m, gate_json) -> tuple:
    rc = m.cmd_gate(_Args(min_provenance=100, skip_eval=True))
    gate = _json.loads(gate_json.read_text())
    return rc, gate["checks"]


def test_the_gate_verifies_every_carried_edge_relanded(tmp_path, monkeypatch):
    """Clause 3 of #1664: `carryover_facts` had no counterpart for edges.

    `git grep carryover_edges` was empty repo-wide before this, and `edges.json`
    was read in exactly one place (`import`), which did not report what it could
    not write. A rebuild whose edge carry was cut short therefore passed the
    gate and swapped, promoting a store with no stated edges in it.
    """
    m = _load(KG_REBUILD)
    carried = [_carry_edge("Alan", "Lloyd", "uses"),
               _carry_edge("Lloyd", "Alan", "related-to")]
    # `edges.add` folds the type through `canonical_edge_type` on the way in
    # (`app/kg_store.py:701`), so the carried `related-to` IS the `related_to`
    # row. Matching the manifest's spelling would fail the gate on edges that
    # landed correctly — #1161's fragmentation rebuilt as an unpassable check.
    held = [_carry_edge("Alan", "Lloyd", "uses"),
            _carry_edge("Lloyd", "Alan", "related_to")]
    gate_json = _gate_world(tmp_path, m, monkeypatch, store_edges=held,
                            carried_edges=carried)
    rc, checks = _gate_checks(m, gate_json)
    assert checks["carryover_edges"]["pass"] is True, checks
    assert checks["carryover_edges"]["got"] == "2/2", checks
    assert rc == 0, {k: v for k, v in checks.items() if not v["pass"]}


def test_the_gate_fails_when_a_carried_edge_is_absent_from_the_rebuild_store(
        tmp_path, monkeypatch):
    """Clause 3's other half: a missing carried edge fails the gate itself.

    The store holds one of the two manifest rows, which is what a partial import
    leaves behind — and this asserts every OTHER check still passes, so the
    refusal cannot be credited to something unrelated.
    """
    m = _load(KG_REBUILD)
    carried = [_carry_edge("Alan", "Lloyd", "uses"),
               _carry_edge("Lloyd", "Alan", "related-to")]
    held = [_carry_edge("Alan", "Lloyd", "uses")]
    gate_json = _gate_world(tmp_path, m, monkeypatch, store_edges=held,
                            carried_edges=carried)
    rc, checks = _gate_checks(m, gate_json)
    assert checks["carryover_edges"]["pass"] is False, checks
    assert checks["carryover_edges"]["got"] == "1/2", checks
    assert checks["carryover_edges"]["note"] == "1 missing", checks
    assert rc == 1, checks
    assert all(v["pass"] for k, v in checks.items() if k != "carryover_edges"), (
        "the gate must fail on the edge check alone: " +
        str({k: v for k, v in checks.items() if not v["pass"]}))
