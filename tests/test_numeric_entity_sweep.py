"""The #1643 expiry sweep: `scripts/memory/expire_numeric_entity_facts.py`.

Two things had to be true before a single `expired_at` could be written here:

* `app/kg_store.py` cannot expire an entity — `entities` has no `expired_at`
  column (`app/kg_store.py:89-97`) and the only `expire()` helper is edges-only
  (`:714`) — so expiry is the fact rows plus the markdown they were indexed from.
* Nothing on the machine did that. `git grep -ln expired_at -- scripts` returned
  12 scripts and none of them expires facts keyed by entity name, which is why
  #743's owed expiry never ran: on the live store 2026-09-28 there were 112
  numeric-shaped entity rows (87 bare digits, 25 `#N`) with 442 unexpired facts
  under them, `expired_at` null on every one.

So the sweep is tested against a store built the way the rest of the graph is —
`kg_store.configure(tmp_path / …)` plus a real fact tree and a real `reindex` —
and never against the live `kg.sqlite`. Running it there is #1643's owed-after-
landing step; a round that swept live data in its own gate could not be re-run.
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import kg_store  # noqa: E402

SPEC = importlib.util.spec_from_file_location(
    "expire_numeric_entity_facts",
    ROOT / "scripts" / "memory" / "expire_numeric_entity_facts.py")
sweep_mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep_mod)

#: The two spellings the live store holds, and the count shape its rows carry:
#: 87 of the 112 numeric rows are bare digits, 25 lead with `#`.
NUMERIC_DIRS = ("1051", "#1052", "294")

#: Real names the same store holds facts under, one of which is one character
#: away from the refused shape. `Task #1053` is the canonical form #743 already
#: refuses at the write side but which the store legitimately holds from before
#: that landing, and `PyTorch 2.7` is the name clause 2 of this item refuses to
#: break — the sweep must read names, not digits.
ORDINARY_DIRS = ("Lloyd", "Task #1053", "PyTorch 2.7")


def _write_fact_file(facts_root: Path, entity: str, category: str, facts: list) -> Path:
    d = facts_root / entity
    d.mkdir(parents=True, exist_ok=True)
    fm = {"type": "facts", "entity": entity, "category": category, "facts": facts}
    path = d / f"{entity}-{category}.md"
    path.write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# {entity}\n",
                    encoding="utf-8")
    return path


@pytest.fixture
def graph(tmp_path):
    """A temp facts tree + a temp store, with the sweep's rows already indexed.

    `reindex(..., root=facts_root)` is what stores a relative `file_path`, and
    `file_path` is how the sweep finds the markdown behind an index row, so an
    index built without `root` would send it hunting in the wrong place.
    """
    facts_root = tmp_path / "facts"
    facts_root.mkdir()
    kg = kg_store.configure(tmp_path / "kg.sqlite")
    yield facts_root, kg
    kg_store.reset()


def _seed(graph):
    """Two numeric entities, one carrying a fact filed under a real name.

    The mixed file is the load-bearing part of the fixture: `facts/1051/` holds
    three facts of its own plus one that the file attributes to `Lloyd`, because
    `_rows_for_file` lets a fact override its entity and the live tree contains
    exactly that shape. A sweep that expired the whole file would look correct on
    a cleaner fixture.
    """
    facts_root, kg = graph
    _write_fact_file(facts_root, "1051", "overview", [
        {"fact": "1051 is a design owner", "id": "n-001"},
        {"fact": "1051 owes a gate rung", "id": "n-002"},
        {"fact": "the note about Lloyd's seeds, filed under the number",
         "id": "n-003", "entity": "Lloyd"},
    ])
    _write_fact_file(facts_root, "1051", "state", [
        {"fact": "1051 was minted by extraction on 2026-09-25", "id": "n-004"},
    ])
    _write_fact_file(facts_root, "#1052", "overview", [
        {"fact": "#1052 is the hash-prefixed spelling", "id": "h-001"},
    ])
    _write_fact_file(facts_root, "Lloyd", "state", [
        {"fact": "Lloyd is the agent", "id": "l-001"},
        {"fact": "Lloyd runs on this box", "id": "l-002"},
    ])
    _write_fact_file(facts_root, "Task #1053", "state", [
        {"fact": "Task #1053 is a real task entity", "id": "t-001"},
    ])
    _write_fact_file(facts_root, "PyTorch 2.7", "state", [
        {"fact": "PyTorch 2.7 is a release", "id": "p-001"},
    ])
    kg.facts_idx.reindex(root=facts_root)
    return facts_root, kg


def _live(kg, entity):
    return kg.facts_idx.for_entity(entity)


def _all_rows(kg, entity):
    return kg.facts_idx.for_entity(entity, include_expired=True)


def _file_facts(path):
    fm = yaml.safe_load(path.read_text(encoding="utf-8").split("---\n", 2)[1])
    return fm["facts"]


# ── clause 3: what it expires, and what it must not touch ────────────────────

def test_the_sweep_expires_every_fact_under_a_numeric_name_and_touches_nothing_else(
        graph, tmp_path):
    """`#?\\d+` as a WHOLE name, both facts and markdown, one direction only.

    The seeded store indexes 4 facts under a numeric name — 3 under `1051`
    (two in `1051-overview.md`, one in `1051-state.md`) and 1 under `#1052` —
    against 6 under real names: `l-001`/`l-002` plus `n-003`, the fact filed in
    the numeric directory that the file itself attributes to `Lloyd`, then one
    each under `Task #1053` and `PyTorch 2.7`. `expired_at` and a reason go on the
    four; those six keep theirs null.
    """
    facts_root, kg = _seed(graph)
    ended = "2026-09-28T12:00:00+00:00"

    dry = sweep_mod.sweep(kg, facts_root, apply=False, ended=ended)
    assert dry["facts_live_before"] == 4, dry
    assert dry["facts_expired"] == 0, "a dry run wrote"
    assert dry["files_touched"] == 0, dry
    assert dry["facts_still_live_after"] == 4, dry
    assert _live(kg, "1051"), "the dry run expired rows in the index"
    for d in NUMERIC_DIRS + ORDINARY_DIRS:
        for p in (facts_root / d).glob("*.md"):
            assert not [f for f in _file_facts(p) if f.get("expired_at")], p

    rep = sweep_mod.sweep(kg, facts_root, apply=True, ended=ended)
    assert rep["facts_expired"] == 4, rep
    assert rep["facts_still_live_after"] == 0, rep
    assert rep["files_touched"] == 3, rep
    assert rep["facts_left_alone_in_scanned_files"] == 1, (
        "the fact a numeric file attributes to `Lloyd` was not counted as left "
        "alone, so the sweep is expiring by file and not by entity name")

    # The index: no live row under a numeric name, and the rows are still there
    # with a date and a reason. Expiry, not deletion — #1025's no-evict ruling.
    assert _live(kg, "1051") == []
    assert _live(kg, "#1052") == []
    expired_rows = _all_rows(kg, "1051") + _all_rows(kg, "#1052")
    assert sorted(r["fact_id"] for r in expired_rows) == ["h-001", "n-001", "n-002",
                                                          "n-004"], expired_rows
    assert all(r["expired_at"] == ended for r in expired_rows), expired_rows

    # The markdown mirrors it, which is the half a store-only update would fake.
    overview = _file_facts(facts_root / "1051" / "1051-overview.md")
    state = _file_facts(facts_root / "1051" / "1051-state.md")
    expired_ids = {f["id"] for f in overview + state if f.get("expired_at")}
    assert expired_ids == {"n-001", "n-002", "n-004"}, sorted(expired_ids)
    for f in overview + state:
        if f["id"] in expired_ids:
            assert f["expired_at"] == ended, f
            assert "#1643" in (f.get("expired_reason") or ""), f
        else:
            assert f["id"] == "n-003", f
            assert not f.get("expired_at"), (
                "the fact this file attributes to `Lloyd` was expired with its "
                f"neighbours: {f}")
    assert all(f.get("expired_at") == ended
               for f in _file_facts(facts_root / "#1052" / "#1052-overview.md"))

    # Nothing else moved. `Task #1053` is one character from the refused shape
    # and `PyTorch 2.7` is a name that merely contains digits.
    # 3 for `Lloyd`, not 2: `n-003` is indexed under `Lloyd` from inside
    # `facts/1051/`, so it is one of the rows this sweep must leave live.
    for entity, want in (("Lloyd", 3), ("Task #1053", 1), ("PyTorch 2.7", 1)):
        assert len(_live(kg, entity)) == want, (
            f"{entity} lost rows to a sweep keyed on `#?\\d+`: {_all_rows(kg, entity)}")
        for p in (facts_root / entity).glob("*.md"):
            assert not [f for f in _file_facts(p) if f.get("expired_at")], p
    assert not [r for r in _all_rows(kg, "Lloyd") if r["expired_at"]], _all_rows(kg, "Lloyd")

    # And the rows themselves are intact: no eviction, no directory removed.
    for name in ("1051", "#1052"):
        assert kg.entities.lookup(name) is not None, (
            f"{name} was evicted; #1025's ruling is to expire, because eviction "
            "makes the id unretrievable")
        assert (facts_root / name).is_dir(), name


def test_a_second_run_finds_nothing_left_and_rewrites_no_file(graph):
    """Idempotence, asserted on the files rather than only on the counts.

    The live run is owed a re-run after a rebuild re-extracts the tree, so a
    second pass must not stamp a fresh `expired_at` on facts that already carry
    one — that would make every later reader's `expired_at` mean "whenever the
    sweep last happened" instead of when the fact stopped being true.
    """
    facts_root, kg = _seed(graph)
    first = sweep_mod.sweep(kg, facts_root, apply=True, ended="2026-09-28T12:00:00+00:00")
    assert first["facts_expired"] == 4, first
    stamped = {p: p.read_text(encoding="utf-8")
               for p in facts_root.rglob("*.md")}

    again = sweep_mod.sweep(kg, facts_root, apply=True, ended="2026-10-01T00:00:00+00:00")
    assert again["facts_expired"] == 0, again
    assert again["facts_live_before"] == 0, again
    assert again["files_touched"] == 0, again
    for p, text in stamped.items():
        assert p.read_text(encoding="utf-8") == text, (
            f"a second run rewrote {p.relative_to(facts_root)}")


# ── clause 4: the report has to distinguish an empty store from a no-op ──────

def test_the_report_names_the_number_it_expired_so_an_empty_store_is_not_a_no_op(
        graph, capsys):
    """Three stores can each print `facts_expired=0`, and only one of them is fine.

    1. no numeric entity row at all — the store is clean;
    2. numeric rows whose facts are already expired — clean, and the sweep is
       re-runnable;
    3. numeric rows with live facts that the sweep failed to expire — a silent
       failure, and the case a count-only report cannot tell from (1).

    So the report carries `numeric_entities`, `facts_live_before`,
    `facts_expired` and `facts_still_live_after` together, measured through
    `kg_store.configure()`'s store and re-measured after the write.
    """
    empty_root = graph[0]
    kg = graph[1]
    empty_root.mkdir(exist_ok=True)

    clean = sweep_mod.sweep(kg, empty_root, apply=True)
    assert (clean["numeric_entities"], clean["facts_live_before"],
            clean["facts_expired"], clean["facts_still_live_after"]) == (0, 0, 0, 0)
    capsys.readouterr()
    sweep_mod.print_report(clean)
    out = capsys.readouterr().out
    assert "numeric_entities=0" in out and "facts_expired=0" in out, out
    assert "nothing to expire" in out, (
        f"a store with no numeric row must say so rather than look like a pass: {out}")

    facts_root, kg = _seed(graph)
    done = sweep_mod.sweep(kg, facts_root, apply=True)
    assert done["facts_expired"] == 4, done
    capsys.readouterr()

    # Store (3): numeric rows with live facts, reported before anything is written.
    facts_root2 = graph[0] / "second"
    facts_root2.mkdir()
    kg_store.reset()
    kg2 = kg_store.configure(facts_root2.parent / "kg2.sqlite")
    _write_fact_file(facts_root2, "294", "state",
                     [{"fact": "294 is a numeric row", "id": "s-001"}])
    kg2.facts_idx.reindex(root=facts_root2)
    before = sweep_mod.sweep(kg2, facts_root2, apply=False)
    assert (before["numeric_entities"], before["facts_live_before"],
            before["facts_expired"]) == (1, 1, 0), before
    sweep_mod.print_report(before)
    out = capsys.readouterr().out
    assert "facts_live=1" in out and "facts_expired=0" in out, out
    assert "facts_to_expire=1" in out and "would expire" in out, (
        f"a dry run must say what it would do, not look like a clean store: {out}")

    after = sweep_mod.sweep(kg2, facts_root2, apply=True)
    assert after["facts_expired"] == 1, after
    assert after["facts_still_live_after"] == 0, after

    # Store (3), the real version of the silent failure: live indexed rows under a
    # numeric name that this run cannot expire, because the file they were indexed
    # from is not on disk — the state a partially-applied rebuild leaves. Nothing
    # is expired, the facts stay live, and the report says the run did nothing
    # rather than reading as a clean store.
    facts_root3 = graph[0] / "third"
    facts_root3.mkdir()
    kg_store.reset()
    kg3 = kg_store.configure(facts_root3.parent / "kg3.sqlite")
    _write_fact_file(facts_root3, "295", "state",
                     [{"fact": "295 is a numeric row", "id": "t-001"}])
    kg3.facts_idx.reindex(root=facts_root3)
    (facts_root3 / "295" / "295-state.md").unlink()
    capsys.readouterr()
    stuck = sweep_mod.sweep(kg3, facts_root3, apply=True)
    assert (stuck["numeric_entities"], stuck["facts_live_before"],
            stuck["facts_expired"], stuck["facts_still_live_after"]) == (1, 1, 0, 1), stuck
    assert stuck["indexed_files_missing_from_disk"] == 1, stuck
    sweep_mod.print_report(stuck)
    out = capsys.readouterr().out
    assert "none expired" in out and "touched nothing" in out, out
    assert "absent from disk" in out and "295-state.md" in out, out


def test_the_sweep_never_writes_outside_the_facts_root(graph, capsys):
    """`file_path` is index data, and index data comes from a rebuild.

    A row whose path escapes the facts root is skipped and reported rather than
    opened: this is the only component in the item that writes to a tree of
    markdown, and a `..` in a stored path must not turn an expiry sweep into a
    write anywhere on the disk.
    """
    facts_root, kg = _seed(graph)
    escapee = facts_root.parent / "elsewhere.md"
    escapee.write_text("---\ntype: facts\nentity: '9999'\ncategory: state\n"
                       "facts:\n- fact: outside the tree\n  id: e-001\n"
                       "---\n\n# 9999\n", encoding="utf-8")
    # Indexed the way anything else here is indexed, then the stored `file_path`
    # is rewritten to the form a rebuild against a different root would leave:
    # relative, and pointing out of the tree.
    kg.facts_idx.reindex([escapee], root=facts_root.parent)
    # `reindex` registers the file's DIRECTORY as the entity row, so the numeric
    # row is registered the way the extractor would have registered it, and then
    # the stored `file_path` is rewritten to the escaping form.
    kg.entities.register("9999", kind="task")
    kg.conn.execute("UPDATE facts_idx SET file_path='../elsewhere.md' "
                    "WHERE fact_id='e-001'")
    kg.conn.commit()

    rep = sweep_mod.sweep(kg, facts_root, apply=True)
    assert "9999" in sweep_mod.numeric_entities(kg), rep
    assert rep["stored_paths_escaping_the_facts_root"] == 1, rep
    assert "../elsewhere.md" in rep["escaped_paths"], rep["escaped_paths"]
    assert not [f for f in _file_facts(escapee) if f.get("expired_at")], (
        "the sweep wrote through a `..` file_path")
    assert escapee.read_text(encoding="utf-8").startswith("---\ntype: facts"), (
        "the escaped file was rewritten at all")
    assert len(_live(kg, "9999")) == 1, "the escaped row was expired as well as skipped"
    sweep_mod.print_report(rep)
    out = capsys.readouterr().out
    assert "escapes the facts root" in out and "../elsewhere.md" in out, out


def test_the_predicate_matches_the_write_side_guard(graph):
    """One shape, two directions: what the guard refuses to mint, this expires.

    `app/entity_naming._TRACKER_ID_RE` is the mint site's pattern and this module
    carries its own copy on purpose — but a copy is only safe while the two agree,
    so the agreement is asserted over the same name list rather than trusted.
    """
    from app.entity_naming import is_backlog_citation_entity

    for name in ("1051", "#1051", "294", "#294", "007"):
        assert sweep_mod.is_numeric_entity_name(name) is True, name
        assert is_backlog_citation_entity(name) is True, name
    for name in ("Lloyd", "Task #1053", "PyTorch 2.7", "1Password", "Backlog System",
                 "Triage of backlog item #338", "v10", "2.7", ""):
        assert sweep_mod.is_numeric_entity_name(name) is False, name


def test_apply_refuses_while_a_rebuild_has_writes_disabled(graph, tmp_path, capsys,
                                                           monkeypatch):
    """`kg_rebuild.py:145` sets `knowledge_graph.write_enabled: false` and extracts
    into a parallel tree that can be renamed `facts-quarantine-<ts>` at any moment;
    `agent_mcp/facts.py::_writes_enabled` refuses writes in that window, and a
    markdown-writing sweep on the same tree has to refuse it too.

    The dry run stays available, because it writes nothing — refusing it would
    only stop a person checking what the rebuild window is about to change.
    """
    facts_root, kg = _seed(graph)
    db = tmp_path / "kg.sqlite"
    monkeypatch.setattr(sweep_mod, "writes_disabled_by_rebuild", lambda: True)

    assert sweep_mod.main(["--apply", "--kg-db", str(db),
                           "--facts-root", str(facts_root)]) == 2
    out = capsys.readouterr().out
    assert "write_enabled = false" in out and "rebuild" in out, out
    assert not [f for f in _file_facts(facts_root / "1051" / "1051-state.md")
                if f.get("expired_at")], "an --apply wrote during the rebuild window"

    monkeypatch.setattr(sweep_mod, "writes_disabled_by_rebuild", lambda: False)
    assert sweep_mod.main(["--apply", "--kg-db", str(db),
                           "--facts-root", str(facts_root)]) == 0
    assert "facts_expired=4" in capsys.readouterr().out
    assert [f for f in _file_facts(facts_root / "1051" / "1051-state.md")
            if f.get("expired_at")], "the write gate never releases"


def test_an_absent_kg_db_is_refused_rather_than_reported_clean(tmp_path, capsys):
    """`configure()` is the provisioning route and CREATES an absent database, so
    a typo in `--kg-db` would otherwise hand this sweep an empty store reporting
    `numeric_entities=0 facts_live=0` — the same reading as a store already clean.
    `store()` refuses the identical mistake on the default path; this is that
    refusal on the flag path.
    """
    missing = tmp_path / "typo" / "kg.sqlite"
    facts_root = tmp_path / "facts"
    facts_root.mkdir()

    assert sweep_mod.main(["--kg-db", str(missing),
                           "--facts-root", str(facts_root)]) == 2
    out = capsys.readouterr().out
    assert "refusing to report a count" in out, out
    assert not missing.exists(), "the refusal still provisioned the store it refused"
