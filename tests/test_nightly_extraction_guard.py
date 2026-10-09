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
# asks `facts_idx`, whose rows survived a wipe because the clean path touched no
# index, so re-deriving a fact into a store that still held its row was refused as
# a duplicate and the fact lost outright. That was the defect filed from #2350's
# owed entry as #2483, and the section at the foot of this file now measures it —
# the store there is indexed before the clean, as a write would have indexed it.
# The store below stays an empty temp one, so the date path keeps being measured
# on its own and these nodes read exactly the rows they always read.

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


# ── #2483: the clean also retires the index rows of the files it deletes ──────
#
# The section above saves the dates and, on its own, that is all it can save.
# `_refuse_held_facts` asks `facts_idx`, and a clean that touches no index leaves
# every row of the wiped tree standing, so the rebuild that follows writes each
# fact again and is refused by the row of a file that no longer exists: the fact is
# lost outright, and the `created_at` the carry-forward preserved dies with the
# in-memory archive, because a refused write never reaches the stamp site. These
# nodes run the index half against a real store so the two halves are measured
# together — the corpus below is therefore indexed, exactly as a write would have
# indexed it, before the clean is called.
#
# The retire is an UPDATE of `expired_at`, never a DELETE: an expired row stops
# refusing a re-derivation (`find_duplicate` refuses only on a live row) while
# staying in `COUNT(*)`, which is the denominator of #499's duplicate trend.
# Retiring by `reindex(paths=…)` instead would have fixed the refusal by shrinking
# that measure, so the first node below pins WHICH retirement happened, not only
# that the row stopped refusing.

def _write_frontmatter(path, entity, category, facts) -> Path:
    """The front-matter shape `write_fact_file` writes, at an explicit path.

    `_fact_file` derives its location from the entity name, which cannot express a
    file inside a directory the clean protects — and a protected file is the only
    way to prove the retire is scoped to the files that went.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = {"type": "facts", "entity": entity, "category": category, "facts": facts}
    path.write_text(f"---\n{yaml.dump(frontmatter, sort_keys=False)}---\n\n"
                    f"# {entity} - {category}\n", encoding="utf-8")
    return path


def _indexed(paths, root) -> int:
    """Rows in the temp store for `paths`, spelled the way a write would spell them.

    `root` is the facts tree, so `file_path` is stored relative to it — the same
    call `_index_and_link` makes (`update_file(path, root=self.facts_dir)`) and the
    one spelling the retire can match.
    """
    return kg_store.store().facts_idx.reindex(list(paths), root=root)["facts"]


def _live(entity) -> list:
    """Rows the store still considers held: `expired_at` and `invalid_at` both NULL."""
    return kg_store.store().facts_idx.for_entity(entity)


def _every(entity) -> list:
    return kg_store.store().facts_idx.for_entity(entity, include_expired=True)


def test_a_clean_retires_the_index_rows_of_the_files_it_deleted(tmp_path, monkeypatch,
                                                               temp_store):
    """Clause 1: after a clean no LIVE `facts_idx` row names a fact file the clean
    deleted, and every row for an entry the clean kept is still live.

    Three rows over three files: two entity files that a clean removes, and one
    inside `memory-graph/`, a protected directory, which it must not touch. The
    protected file is what keeps this honest — a retire that swept the whole table
    would satisfy the first assertion and fail the second.
    """
    facts = tmp_path / "facts"
    facts.mkdir()
    state = _fact_file(facts, "Lloyd", "state",
                       [{"fact": "runs on vLLM", "confidence": 0.9, "created_at": RECORDED}])
    usage = _fact_file(facts, "Lloyd", "usage",
                       [{"fact": "serves the bench model", "confidence": 0.9,
                         "created_at": RECORDED}])
    kept = _write_frontmatter(facts / "memory-graph" / "graph-index.md", "Lloyd", "state",
                              [{"fact": "an edge the clean cannot re-derive",
                                "confidence": 0.9, "created_at": RECORDED}])
    assert _indexed([state, usage, kept], facts) == 3

    _cleaner(tmp_path, monkeypatch, facts).clean_facts_directory()

    assert not state.exists() and not usage.exists(), "the clean must still wipe the tree"
    assert kept.exists(), "memory-graph/ is protected; this corpus is wrong"
    live = _live("Lloyd")
    assert [r["fact"] for r in live] == ["an edge the clean cannot re-derive"], live
    assert live[0]["file_path"] == "memory-graph/graph-index.md", live
    retired = sorted((r["file_path"], r["expired_at"] is not None, r["invalid_at"])
                     for r in _every("Lloyd") if r["expired_at"] is not None)
    assert retired == [("Lloyd/Lloyd-state.md", True, None),
                       ("Lloyd/Lloyd-usage.md", True, None)], retired


def test_a_clean_lets_the_rebuild_write_an_identical_fact_again(tmp_path, monkeypatch,
                                                               temp_store, capsys):
    """Clause 2: the fact an identical rebuild re-derives after a clean is ACCEPTED.

    The file is re-created at the same path, the write prints no
    `refused N duplicate fact(s)` line, and `write_fact_file` returns the path where
    it used to return `None` — the shape of the whole batch being refused against a
    row of a deleted file, which is #2483's loss and not a warning.
    """
    facts = tmp_path / "facts"
    facts.mkdir()
    path = _fact_file(facts, "Lloyd", "state",
                      [{"fact": "runs on vLLM", "confidence": 0.9, "created_at": RECORDED}])
    assert _indexed([path], facts) == 1
    x = _cleaner(tmp_path, monkeypatch, facts)

    x.clean_facts_directory()
    capsys.readouterr()
    written = x.extractor.write_fact_file(
        "Lloyd", "state", {"facts": [{"fact": "runs on vLLM", "confidence": 0.9}]},
        source_doc="knowledge/lloyd.md", source_hash="abc123")
    printed = capsys.readouterr().out

    assert written == facts / "Lloyd" / "Lloyd-state.md", written
    assert "duplicate fact(s)" not in printed, printed
    assert [r["fact"] for r in _rows(facts / "Lloyd" / "Lloyd-state.md")] == ["runs on vLLM"]
    assert len(_live("Lloyd")) == 1, _every("Lloyd")


def test_the_fact_a_clean_lets_through_keeps_the_date_captured_before_the_wipe(
        tmp_path, monkeypatch, temp_store):
    """Clause 3: #2350's carry-forward still holds once the rows are retired — and
    only once they are, which is why this node is the same corpus as
    `test_a_clean_carries_the_recorded_created_at_through_the_wipe` with an indexed
    store added. Retire the rows and the fact comes back dated 2026-01-05; leave
    them live and the write is refused, so there is no row to date at all.

    Both halves are asserted: the markdown row and the live index row built from it
    carry the pre-clean date, not the run's.
    """
    facts = tmp_path / "facts"
    facts.mkdir()
    path = _fact_file(facts, "Lloyd", "state",
                      [{"fact": "runs on vLLM", "confidence": 0.9, "created_at": RECORDED}])
    assert _indexed([path], facts) == 1
    x = _cleaner(tmp_path, monkeypatch, facts)

    x.clean_facts_directory()
    x.extractor.write_fact_file("Lloyd", "state",
                                {"facts": [{"fact": "runs on vLLM", "confidence": 0.9}]},
                                source_doc="knowledge/lloyd.md", source_hash="abc123")

    rederived = _rows(facts / "Lloyd" / "Lloyd-state.md")
    assert len(rederived) == 1, rederived
    assert rederived[0]["created_at"] == RECORDED, rederived[0]
    assert not _stamped_this_run(rederived[0]["created_at"]), rederived[0]
    live = _live("Lloyd")
    assert len(live) == 1, live
    assert live[0]["created_at"] == RECORDED, live[0]


def test_a_clean_expires_those_rows_and_prints_the_count(tmp_path, monkeypatch,
                                                        temp_store, capsys):
    """Clause 4: the retirement is an expiry, not a deletion, and the clean reports
    it beside the carry-forward witness.

    Expired rows are still rows: `exact_duplicate_stats` documents that its
    `rows` figure — the #499 trend's denominator — counts ones an ingestion wrote,
    expired included, so a clean must leave `COUNT(*)` where it found it. Deleting
    the rows instead (`reindex(paths=…)` on a path that no longer exists does
    exactly that) moves the measure every time anyone rebuilds.

    The corpus is the same three files as clause 1: 3 rows, 2 of them retired. The
    witness line is asserted to sit on a line adjacent to the carry-forward line,
    because "how many rows did this clean retire" is only readable against "how
    many dates did it carry".
    """
    facts = tmp_path / "facts"
    facts.mkdir()
    state = _fact_file(facts, "Lloyd", "state",
                       [{"fact": "runs on vLLM", "confidence": 0.9, "created_at": RECORDED}])
    usage = _fact_file(facts, "Lloyd", "usage",
                       [{"fact": "serves the bench model", "confidence": 0.9,
                         "created_at": RECORDED}])
    kept = _write_frontmatter(facts / "memory-graph" / "graph-index.md", "Lloyd", "state",
                              [{"fact": "an edge the clean cannot re-derive",
                                "confidence": 0.9, "created_at": RECORDED}])
    assert _indexed([state, usage, kept], facts) == 3
    stats = kg_store.store().facts_idx.exact_duplicate_stats()
    x = _cleaner(tmp_path, monkeypatch, facts)

    x.clean_facts_directory()

    out = capsys.readouterr().out
    after = kg_store.store().facts_idx.exact_duplicate_stats()
    assert after["rows"] == stats["rows"] == 3, (stats, after)
    assert kg_store.store().facts_idx.count() == 3, _every("Lloyd")
    assert kg_store.store().facts_idx.count(active_only=True) == 1, _every("Lloyd")
    witness = [ln for ln in out.splitlines() if "facts_idx retired" in ln]
    carry = [ln for ln in out.splitlines() if "created_at carry-forward" in ln]
    assert len(witness) == 1 and len(carry) == 1, out
    assert "2 row(s) expired" in witness[0] and "2 deleted fact file(s)" in witness[0], witness
    lines = out.splitlines()
    assert abs(lines.index(witness[0]) - lines.index(carry[0])) == 1, out


def test_the_retire_reaches_every_file_past_one_statement_of_them(tmp_path, monkeypatch,
                                                                 temp_store, capsys):
    """501 deleted files against one chunk of `expire_files` (500 `file_path` values
    per UPDATE statement, and the live tree is 23,630 files).

    The chunk bound is SQLite's, not this code's preference: a statement may bind
    only so many variables, so the retire loops. A loop whose arithmetic drops the
    tail leaves 1 of 501 files refusing its own re-derivation on a live run, and
    nothing in a 3-file corpus would show it — so this corpus is 501.
    """
    facts = tmp_path / "facts"
    (facts / "Lloyd").mkdir(parents=True)
    paths = [_write_frontmatter(facts / "Lloyd" / f"Lloyd-state-{i:04d}.md", "Lloyd", "state",
                                [{"fact": f"claim number {i}", "confidence": 0.9,
                                  "created_at": RECORDED}])
             for i in range(501)]
    assert _indexed(paths, facts) == 501
    x = _cleaner(tmp_path, monkeypatch, facts)

    x.clean_facts_directory()

    assert _live("Lloyd") == [], [r["file_path"] for r in _live("Lloyd")]
    rows = _every("Lloyd")
    assert len(rows) == 501, len(rows)
    assert all(r["expired_at"] is not None for r in rows)
    assert "501 row(s) expired across 501 deleted fact file(s)" in capsys.readouterr().out
