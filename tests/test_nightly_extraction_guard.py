"""#956 — the nightly clean's guard protects what is irreplaceable, and only that.

The 2026-08-22 guard in ``nightly_extraction.py`` named three files as
hand-built graph state a clean must never delete. Two of them are:
``_relationships.json`` and ``entity-registry.json`` cannot be re-derived.
``entity-aliases.json`` could be, from ``kg.sqlite`` (``app.kg_store``), once
the 2026-09 migration made the store the alias source and #474 pinned the
legacy export as a snapshot — yet the guard still called it irreplaceable and
``backup_graph_state`` copied the frozen file into every pre-clean backup
(8 copies, 7 MB, all the 2026-09-03 bytes).

Two things are pinned here, in the order that matters: the protection that
exists because of 2026-08-22 must not regress, and the derivable file is no
longer in it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NGM = ROOT / "scripts" / "memory" / "next-gen-memory"


def _load_nightly():
    # The module resolves its siblings off its own directory when run as a
    # script; loaded by path, the caller injects it (#755).
    if str(NGM) not in sys.path:
        sys.path.insert(0, str(NGM))
    spec = importlib.util.spec_from_file_location("nightly_extraction_guard_under_test",
                                                  NGM / "nightly_extraction.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _extractor(mod):
    # `__init__` builds the model-backed extractors; the clean uses none of
    # them, only the class attributes and `FACTS_DIR`.
    return mod.NightlyExtraction.__new__(mod.NightlyExtraction)


def test_the_guard_names_exactly_the_two_irreplaceable_files():
    mod = _load_nightly()
    assert mod.NightlyExtraction.PROTECTED_NAMES == frozenset({
        "entity-registry.json", "_relationships.json",
    })


def test_a_clean_keeps_the_graph_state_and_removes_the_derivable_alias_export(tmp_path, monkeypatch):
    mod = _load_nightly()
    facts = tmp_path / "facts"
    facts.mkdir()
    for name in ("entity-registry.json", "_relationships.json", "entity-aliases.json",
                 "_relationships.20260822T020000Z.bak.json"):
        (facts / name).write_text("{}", encoding="utf-8")
    (facts / "Lloyd").mkdir()
    (facts / "Lloyd" / "Lloyd-overview.md").write_text("# Lloyd\n", encoding="utf-8")
    pipeline = tmp_path / "_pipeline"
    monkeypatch.setattr(mod, "FACTS_DIR", facts)
    monkeypatch.setattr(mod, "_STATE_PIPELINE", pipeline)

    _extractor(mod).clean_facts_directory()

    kept = sorted(p.name for p in facts.iterdir())
    # The 2026-08-22 protection, and its recovery path, stay exactly as they were.
    assert "entity-registry.json" in kept
    assert "_relationships.json" in kept
    assert "_relationships.20260822T020000Z.bak.json" in kept
    assert "Lloyd" not in kept, "the fact tree itself is what a clean removes"
    # The derivable export is ordinary fact-tree content to the guard now.
    assert "entity-aliases.json" not in kept


def test_the_pre_clean_backup_copies_the_edge_graph_and_not_the_alias_export(tmp_path, monkeypatch):
    mod = _load_nightly()
    facts = tmp_path / "facts"
    facts.mkdir()
    (facts / "_relationships.json").write_text('{"edges": []}', encoding="utf-8")
    (facts / "entity-aliases.json").write_text("{}", encoding="utf-8")
    pipeline = tmp_path / "_pipeline"
    monkeypatch.setattr(mod, "FACTS_DIR", facts)
    monkeypatch.setattr(mod, "_STATE_PIPELINE", pipeline)

    dest = _extractor(mod).backup_graph_state()

    assert dest is not None and dest.parent == pipeline / "backups"
    assert (dest / "_relationships.json").read_text(encoding="utf-8") == '{"edges": []}'
    assert not (dest / "entity-aliases.json").exists()
    assert sorted(p.name for p in pipeline.rglob("entity-aliases.json")) == []
