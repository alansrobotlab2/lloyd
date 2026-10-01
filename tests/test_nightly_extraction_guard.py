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


# ── #1999: the fan-out has no made-up entity to fall back to ─────────────────

class _FakeExtractor:
    def __init__(self, result):
        self.result, self.written = result, []

    def extract_from_document(self, *a, **k):
        return self.result

    def write_fact_file(self, entity, category, data, **kw):
        self.written.append((entity, category, [f["fact"] for f in data["facts"]]))


def _fan_out(tmp_path, monkeypatch, result):
    mod = _load_nightly()
    monkeypatch.setattr(mod, "VAULT", tmp_path)
    x = _extractor(mod)
    x.extractor = _FakeExtractor(result)
    doc = tmp_path / "doc.md"
    doc.write_text("some content", encoding="utf-8")
    return x, x._process_single_file(doc, True, 1, 1)


def test_a_fact_with_no_usable_entity_is_held_back_and_counted_not_filed(tmp_path, monkeypatch,
                                                                         capsys):
    """The document's primary was refused (the result names no entity), one
    fact names its own subject and two name none. The one is filed; the two
    are held back and the run's stdout line says so. Nothing is filed under
    `general`, which is where both used to go."""
    x, out = _fan_out(tmp_path, monkeypatch, {
        "entity": "", "category": "state",
        "facts": [{"fact": "a", "entity": "Lloyd", "category": "state"},
                  {"fact": "b", "entity": "", "category": "state"},
                  {"fact": "c", "category": "state"}]})

    assert x.extractor.written == [("Lloyd", "state", ["a"])]
    assert out[:3] == (1, 1, True)                    # the document is still done
    assert x.held_back_facts == 2
    line = [ln for ln in capsys.readouterr().out.splitlines() if "Processing:" in ln]
    assert len(line) == 1 and "1 entities, 1 facts, 2 held back (no usable entity)" in line[0], line


def test_the_literal_general_is_never_a_filing_entity(tmp_path, monkeypatch, capsys):
    """Neither as the result's primary (an older extractor build, or a model
    that answers the word) nor on a fact: `general` is a category term."""
    x, out = _fan_out(tmp_path, monkeypatch, {
        "entity": "general", "category": "general",
        "facts": [{"fact": "a"}, {"fact": "b", "entity": "general"}]})

    assert x.extractor.written == []
    assert out[:3] == (1, 0, True)
    assert x.held_back_facts == 2
    assert "2 held back" in capsys.readouterr().out


def test_a_fact_without_its_own_entity_still_files_under_a_valid_primary(tmp_path, monkeypatch,
                                                                         capsys):
    """What the fallback was for is kept: a real primary stands in for a fact
    that names no entity, and nothing is counted as held back but what the
    extractor itself refused."""
    x, out = _fan_out(tmp_path, monkeypatch, {
        "entity": "Lloyd", "category": "state", "held_back": 1,
        "facts": [{"fact": "a"}, {"fact": "b", "entity": "vLLM", "category": "usage"}]})

    assert sorted(x.extractor.written) == [("Lloyd", "state", ["a"]), ("vLLM", "usage", ["b"])]
    assert x.held_back_facts == 1
    assert "1 held back" in capsys.readouterr().out
