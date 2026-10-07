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
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from app import kg_store

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


# ── #2350: a clean carries the recorded created_at through the wipe ──────────
#
# `--clean` wipes the entity tree and writes every fact again from its source
# document, so the date a fact was FIRST recorded was destroyed at the wipe and
# re-stamped with the run date. Measured on the live store 2026-10-07: 100,046 of
# ~138,197 fact rows (72%) carry one date, 2026-09-23 — one rebuild wearing the
# whole store's history. `backup_graph_state` copies only `_relationships.json`
# and `memory-graph/`, and the tree is gitignored, so the values existed nowhere
# but the files being deleted: they had to be read out during the clean, and
# these are the tests that hold it in place.
#
# One thing this section deliberately does not exercise: `_refuse_held_facts`
# asks `facts_idx`, whose rows survive a wipe because the clean path re-indexes
# nothing, so re-deriving a fact into a store that still holds its row is refused
# as a duplicate and the fact is lost outright. That is a separate defect with a
# ruling still owed on #2350; the store below is an empty temp one, so the date
# path is measured on its own rather than hidden behind that bug.

RECORDED = "2026-01-05T03:04:05+00:00"


@pytest.fixture
def temp_store(tmp_path):
    """A throwaway knowledge store, so no test here opens the live one."""
    kg_store.configure(tmp_path / "kg.sqlite")
    yield kg_store
    kg_store.reset()


def _fact_file(facts_dir, entity, category, facts) -> Path:
    """One fact file in the shape `write_fact_file` writes."""
    entity_dir = facts_dir / entity
    entity_dir.mkdir(parents=True, exist_ok=True)
    path = entity_dir / f"{entity}-{category}.md"
    frontmatter = {"type": "facts", "entity": entity, "category": category,
                   "facts": facts}
    path.write_text(f"---\n{yaml.dump(frontmatter, sort_keys=False)}---\n\n"
                    f"# {entity} - {category}\n", encoding="utf-8")
    return path


def _rows(path) -> list:
    return yaml.safe_load(path.read_text(encoding="utf-8").split("---")[1])["facts"]


def _cleaner(tmp_path, monkeypatch, facts):
    """A nightly whose clean wipes `facts` and hands the dates to a real
    extractor writing into that same tree.

    `mod.FactExtractor` is the class the loaded nightly module itself bound, so
    the instance under test is the one `clean_facts_directory` talks to and not a
    second copy of the same file.
    """
    mod = _load_nightly()
    monkeypatch.setattr(mod, "FACTS_DIR", facts)
    monkeypatch.setattr(mod, "_STATE_PIPELINE", tmp_path / "_pipeline")
    x = _extractor(mod)
    x.extractor = mod.FactExtractor()
    x.extractor.facts_dir = facts
    return x


def _stamped_this_run(value: str) -> bool:
    """True when `value` is a stamp from this process, not a carried-forward date."""
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return False
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return abs(datetime.now(timezone.utc) - moment) < timedelta(minutes=5)


def test_a_clean_carries_the_recorded_created_at_through_the_wipe(tmp_path, monkeypatch,
                                                                 temp_store):
    """Clause 1: a fact re-derived after a `clean=True` wipe under the same
    identity — same entity, same category, same fact text — comes back with the
    `created_at` it had on disk, not the run's date.

    The claim is planted twice in the one file, with 2026-01-05 first and
    2026-03-03 second, because 708 live files hold one claim twice (#499) and the
    later copy's stamp is not the date the fact was first recorded: the capture
    has to keep the EARLIEST value it finds, not the last one it read.
    """
    facts = tmp_path / "facts"
    facts.mkdir()
    _fact_file(facts, "Lloyd", "state", [
        {"fact": "runs on vLLM", "confidence": 0.9, "created_at": RECORDED},
        {"fact": "runs on vLLM", "confidence": 0.9,
         "created_at": "2026-03-03T00:00:00+00:00"},
    ])
    x = _cleaner(tmp_path, monkeypatch, facts)

    x.clean_facts_directory()

    assert not (facts / "Lloyd").exists(), "the clean must still wipe the entity tree"
    assert x.extractor.created_at_archive, "the clean captured no date to carry"

    # The re-derivation: the same claim, arriving as a new fact object with no
    # date of its own, which is what the extractor hands the writer.
    x.extractor.write_fact_file("Lloyd", "state",
                                {"facts": [{"fact": "runs on vLLM", "confidence": 0.9}]},
                                source_doc="knowledge/lloyd.md", source_hash="abc123")

    rederived = _rows(facts / "Lloyd" / "Lloyd-state.md")
    assert len(rederived) == 1, rederived
    assert rederived[0]["created_at"] == RECORDED, (
        f"the re-derived fact carries {rederived[0]['created_at']}; the pre-clean "
        f"recording date {RECORDED} was captured and then not used"
    )
    assert not _stamped_this_run(rederived[0]["created_at"]), rederived[0]
    assert rederived[0]["source_hash"] == "abc123", "attribution is still the run's"


def test_a_reworded_or_new_fact_is_stamped_with_the_run_date(tmp_path, monkeypatch,
                                                            temp_store):
    """Clause 2: preservation is per fact identity, never per file.

    One file comes back carrying the fact the clean recorded on 2026-01-05, a
    re-wording of it, and a claim that never existed. Only the first keeps the
    old date; the other two are stamped with the run's date, so a rebuild cannot
    make a new claim look five months old just by filing it next to an old one.
    """
    facts = tmp_path / "facts"
    facts.mkdir()
    _fact_file(facts, "Lloyd", "state", [
        {"fact": "runs on vLLM", "confidence": 0.9, "created_at": RECORDED},
    ])
    x = _cleaner(tmp_path, monkeypatch, facts)

    x.clean_facts_directory()
    x.extractor.write_fact_file("Lloyd", "state", {"facts": [
        {"fact": "runs on vLLM", "confidence": 0.9},
        {"fact": "runs on vLLM since the rebuild", "confidence": 0.9},
        {"fact": "has never been recorded anywhere before", "confidence": 0.9},
    ]})

    rows = {r["fact"]: r for r in _rows(facts / "Lloyd" / "Lloyd-state.md")}
    assert set(rows) == {"runs on vLLM", "runs on vLLM since the rebuild",
                         "has never been recorded anywhere before"}, rows
    assert rows["runs on vLLM"]["created_at"] == RECORDED
    for changed in ("runs on vLLM since the rebuild",
                    "has never been recorded anywhere before"):
        assert _stamped_this_run(rows[changed]["created_at"]), (
            f"{changed!r} carries {rows[changed]['created_at']!r} — a fact whose "
            "text changed, or that never existed, must take the run's date"
        )


def test_the_clean_prints_one_line_of_carried_versus_stamped(tmp_path, monkeypatch,
                                                            temp_store, capsys):
    """Clause 3: the run record is the witness. The clean prints one line giving
    how many `created_at` values it carried forward and how many facts had no
    recorded date to carry and will therefore be stamped with the run date.

    Two dated rows and one undated row across one file: 2 carried, 1 to stamp.
    The scanned/rows figures ride on the same line so a reader can tell "2 of 3"
    from "2 of 200,000", which is the difference between a rebuild that preserved
    the store and one that preserved a rounding error.
    """
    facts = tmp_path / "facts"
    facts.mkdir()
    _fact_file(facts, "Lloyd", "state", [
        {"fact": "runs on vLLM", "confidence": 0.9, "created_at": RECORDED},
        {"fact": "has a workspace in Sheffield", "confidence": 0.9,
         "created_at": "2026-02-02T03:04:05+00:00"},
        {"fact": "has no recording date at all", "confidence": 0.9},
    ])
    x = _cleaner(tmp_path, monkeypatch, facts)

    x.clean_facts_directory()

    lines = [ln for ln in capsys.readouterr().out.splitlines()
             if "created_at carry-forward" in ln]
    assert len(lines) == 1, f"expected exactly one witness line, got {lines}"
    line = lines[0]
    assert "2 recorded date(s) carried" in line, line
    assert "1 fact(s) with no recorded date will be stamped with the run date" in line, line
    assert "1 file(s)" in line and "3 fact row(s) read" in line, line
