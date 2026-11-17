"""#4227 — `facts_idx` refuses to index a fact file outside the root it indexes against.

Six rows of the 204,999 in the live `facts_idx` (measured through `app.kg_store`
on 2026-11-17) name `/tmp/blastroute2_u9ny_mb6/{A,B,C,D}.md` in `file_path` with
`created_at`, `source_doc` and `provenance` all NULL. They got there because
`_FactsIdx.reindex` and `_rows_for_file` caught the `relative_to` ValueError and
stored `str(path)` verbatim, so a caller that named a file outside the tree it was
indexing against wrote that absolute path into `file_path` without complaint. The
store's own read path hides the damage — those 6 are retired, so `for_entity`
returns `[]` — but `/tmp` is tmpfs on this box, the markdown behind them can never
come back, and a full `reindex()` walks `root` and cannot re-derive a file it does
not contain. The index therefore holds rows no rebuild of that tree can produce.

The rule is the path against **the root passed in**, never against
`VAULT_FACTS_ROOT`: a temp tree has to keep indexing, because
`tests/test_api_contracts.py:893` (`reindex(root=tmp_path / "facts")`), the
migration and repair scripts (`scripts/memory/kg_migrate_to_sqlite.py:108`,
`scripts/memory/repair_fact_ids.py:193`,
`scripts/memory/quarantine_general_facts.py:433`) and the nightly knowledge-health
report all index a tree that is not the live one. It is also the rule the store
already applies to itself elsewhere: `_refuse_stray_create` guards the *store
file*'s path; this one guards a *row's* `file_path`, one layer up, which is the
half `_refuse_stray_create` never saw.
"""
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.kg_store import KGStore  # noqa: E402

#: The live residue's own stamps, copied so a node can ask whether a row is live
#: the same way `for_entity` does (`expired_at IS NULL AND invalid_at IS NULL`).
_INVALID_AT = "2026-10-19T00:00:00"
_EXPIRED_AT = "2026-11-09T06:01:09.594200+00:00"


@pytest.fixture
def db(tmp_path):
    """A store of the module's own, its fact tree named per call.

    Nothing is indexed at fixture time: which root a call is asked for is the
    whole subject here, so each node states it.
    """
    s = KGStore(tmp_path / "kg.sqlite")
    yield s
    s.close()


def _facts_root(tmp_path) -> Path:
    root = tmp_path / "facts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _fact_file(root: Path, entity: str, cat: str, facts: list[dict]) -> Path:
    """A fact file in the shape `parse_fact_file` reads, at `root/<entity>/<entity>-<cat>.md`."""
    d = root / entity
    d.mkdir(parents=True, exist_ok=True)
    fm = {"type": "facts", "entity": entity, "category": cat, "facts": facts}
    p = d / f"{entity}-{cat}.md"
    p.write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# {entity} - {cat}\n",
                 encoding="utf-8")
    return p


def _rows(db) -> list[tuple]:
    """Every `facts_idx` row as stored: (entity, fact_id, file_path, expired_at, invalid_at)."""
    return sorted(tuple(r) for r in db.conn.execute(
        "SELECT entity, fact_id, file_path, expired_at, invalid_at FROM facts_idx"))


def _seed_absolute_row(db, path: Path, entity: str, fact_id: str, *,
                       invalid_at: str | None = None, expired_at: str | None = None) -> None:
    """File a row whose `file_path` is an ABSOLUTE path, the way the pre-#4227 writer left it.

    A direct INSERT is the only way to get that spelling into the table once the
    guard is in — the writer that produced it on the live store is gone, which is
    the point of the change. `created_at`, `source_doc` and `provenance` stay NULL
    here exactly as they do on all 6 live residue rows.
    """
    db.conn.execute(
        "INSERT INTO facts_idx(entity, category, fact_id, text_hash, fact, "
        "invalid_at, expired_at, file_path) VALUES (?,?,?,?,?,?,?,?)",
        (entity, "general", fact_id, f"h-{fact_id}", f"residue row {fact_id}",
         invalid_at, expired_at, str(path)))


def _absolute_paths(db) -> list[str]:
    """Stored `file_path`s that are absolute — the shape this change is about."""
    return sorted(r[2] for r in _rows(db) if Path(r[2]).is_absolute())


# ── clause 1: update_file refuses an out-of-root path, table unchanged ───────

def test_update_file_refuses_a_path_outside_the_root_and_leaves_the_table_unchanged(db, tmp_path):
    """Clause 1: `update_file(p, root=R)` for a `p` outside `R` raises `ValueError`
    naming both, and that call neither writes a row nor deletes one.

    Without the guard the call succeeds: `relative_to(p, R)` raises inside
    `reindex`, the `except ValueError` stores `str(p)`, and two rows appear with
    `/tmp`-shaped spellings the tree walk can never reproduce.

    The seeded row matters as much as the raise. `reindex` deletes by `file_path`
    BEFORE it looks at the file (`DELETE FROM facts_idx WHERE file_path=?`, then
    `if not p.exists(): continue`), so a guard placed after that statement would
    retire a row that happens to carry the refused path's absolute spelling — a
    refusal that silently deletes, through the one store whose convention is
    retire-never-delete. This node is the ordering proof.
    """
    root = _facts_root(tmp_path)
    _fact_file(root, "Tidewell Relay", "state", [{"id": "g-1", "fact": "one live row"}])
    db.facts_idx.reindex(root=root)
    outside = _fact_file(tmp_path / "bench-scratch" / "junkcorpslot", "Junk Corps",
                         "general", [{"id": "x-1", "fact": "written by an ad-hoc trial"},
                                     {"id": "x-2", "fact": "and indexed by absolute path"}])
    _seed_absolute_row(db, outside, "Junk Corps", "old-1", invalid_at=_INVALID_AT)
    before = _rows(db)
    assert [r for r in before if r[2] == str(outside)], "the fixture seeded no colliding row"

    with pytest.raises(ValueError) as caught:
        db.facts_idx.update_file(outside, root=root)

    msg = str(caught.value)
    assert str(outside) in msg, f"the message does not name the refused path: {msg}"
    assert str(root) in msg, f"the message does not name the root it refused against: {msg}"
    assert _rows(db) == before, "the refused call wrote or deleted a row"


# ── clause 2: a path list is refused whole, no absolute spelling is stored ───

def test_reindex_over_a_path_list_refuses_the_out_of_root_path_and_stores_no_absolute_path(
        db, tmp_path):
    """Clause 2: `reindex(paths=[good, outside], root=R)` raises on the out-of-root
    path, and no absolute path outside `R` is ever stored in `file_path`.

    Every path in the list is proven against the root before the transaction
    opens, so the whole call is refused rather than half-applied: the good file is
    not indexed either. That is deliberate — an exception part-way through a batch
    would leave the good rows written and the caller unable to say which paths the
    index reflects, which is the ambiguity this guard exists to remove.
    """
    root = _facts_root(tmp_path)
    good = _fact_file(root, "Tidewell Relay", "state", [{"id": "g-1", "fact": "one live row"}])
    outside = _fact_file(tmp_path / "bench-scratch", "Junk Corps", "general",
                         [{"id": "x-1", "fact": "written outside the tree"}])

    with pytest.raises(ValueError) as caught:
        db.facts_idx.reindex([good, outside], root=root)

    msg = str(caught.value)
    assert str(outside) in msg, f"the message does not name the refused path: {msg}"
    assert str(root) in msg, f"the message does not name the root it refused against: {msg}"
    assert _absolute_paths(db) == [], f"an absolute path reached file_path: {_absolute_paths(db)}"
    assert _rows(db) == [], "the refused call indexed rows anyway"


# ── clause 3: an in-root path is stored exactly as it always was ────────────

def test_a_path_under_the_root_passed_in_keeps_its_tree_relative_spelling(db, tmp_path):
    """Clause 3: a path under the root keeps today's root-relative `file_path`, so the
    temp-root callers keep working — and the guard is proved per-call, not global.

    Nothing here is anywhere near `VAULT_FACTS_ROOT`, which the fixture's store is
    not indexed against either: a guard written against that constant instead of
    the root argument would raise on this call and fail the node. The nested file
    is two directories down on purpose — the spelling is relative to the root
    passed in, not to its depth in some other tree.
    """
    root = _facts_root(tmp_path)
    flat = _fact_file(root, "Tidewell Relay", "state", [{"id": "g-1", "fact": "one live row"}])
    nested = _fact_file(root / "experiments" / "runs", "Tidewell Relay", "usage",
                        [{"id": "u-1", "fact": "a row two directories down"}])

    assert db.facts_idx.update_file(flat, root=root) == 1
    assert db.facts_idx.reindex([nested], root=root)["facts"] == 1

    assert sorted(r[2] for r in _rows(db)) == [
        "Tidewell Relay/Tidewell Relay-state.md",
        "experiments/runs/Tidewell Relay/Tidewell Relay-usage.md",
    ]
    assert _absolute_paths(db) == []
    assert [f["fact_id"] for f in db.facts_idx.for_entity("Tidewell Relay")] == ["g-1", "u-1"]


def test_the_full_tree_rebuild_walks_the_root_and_replaces_the_table_as_before(db, tmp_path):
    """`paths=None` is untouched by the guard: it walks `root`, so every file it
    finds is under `root` and nothing is refused.

    This pins the pre-existing rebuild contract the index rests on — a full rebuild
    replaces the whole table (`DELETE FROM facts_idx` before the walk) and stores
    every path relative to the root it walked. It says nothing about residue: the
    6 live `/tmp` rows are left alone by this change and owned by the retention
    sweep's `_kg_index_residue`, and no new code path deletes them.
    """
    root = _facts_root(tmp_path)
    _fact_file(root, "Tidewell Relay", "state", [{"id": "g-1", "fact": "one live row"},
                                                 {"id": "g-2", "fact": "two live rows"}])
    _fact_file(root, "Loamgate", "general", [{"id": "l-1", "fact": "another entity"}])
    stale = _fact_file(root, "Tidewell Relay", "usage", [{"id": "u-9", "fact": "goes away"}])

    stats = db.facts_idx.reindex(root=root)
    assert stats == {"files": 3, "facts": 4, "entities_registered": 2}, stats
    stale.unlink()
    stats = db.facts_idx.reindex(root=root)

    # The gone file's rows are replaced, not carried: 2 files, 3 rows, and both
    # entities are already registered, so the second walk registers nothing.
    assert stats == {"files": 2, "facts": 3, "entities_registered": 0}, stats
    assert sorted({r[2] for r in _rows(db)}) == [
        "Loamgate/Loamgate-general.md",
        "Tidewell Relay/Tidewell Relay-state.md",
    ]
    assert _absolute_paths(db) == []


# ── clause 4: the retired residue stays retired, and is never deleted ───────

def test_a_refused_call_leaves_the_retired_out_of_root_residue_retired_not_deleted(db, tmp_path):
    """Clause 4: the change prevents NEW out-of-root rows; it deletes no existing ones.

    The 6 live rows are the population shaped here: an absolute `/tmp`-style
    spelling, 4 carrying `invalid_at`, 2 carrying `expired_at`, so 0 are live. The
    live store's copy is verified after landing by the item's own re-check
    (`SELECT COUNT(*) FROM facts_idx WHERE file_path LIKE '/tmp/%'` stays 6); this
    node pins the behaviour that leaves them — a refused `update_file` and a
    refused `reindex` over exactly those paths, twice over, retire nothing, because
    the guard runs ahead of the `DELETE FROM facts_idx WHERE file_path=?`.

    Deleting them is not this change's job either: `expire_files` documents
    retire-never-delete as load-bearing for `exact_duplicate_stats`, and
    `_kg_index_residue` in the retention sweep already selects this population.
    """
    root = _facts_root(tmp_path)
    _fact_file(root, "Tidewell Relay", "state", [{"id": "g-1", "fact": "one live row"}])
    db.facts_idx.reindex(root=root)

    scratch = tmp_path / "bench-scratch" / "junkcorpslot"
    residue = [scratch / "A.md", scratch / "B.md", scratch / "C.md",
               scratch / "D.md", scratch / "E.md", scratch / "F.md"]
    for n, p in enumerate(residue):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("---\ntype: facts\nentity: blast-route\nfacts: []\n---\n",
                     encoding="utf-8")
        _seed_absolute_row(db, p, "blast-route", f"r-{n}",
                           invalid_at=_INVALID_AT if n < 4 else None,
                           expired_at=None if n < 4 else _EXPIRED_AT)
    before = _rows(db)

    for p in residue:                       # every one of them, refused
        with pytest.raises(ValueError):
            db.facts_idx.update_file(p, root=root)
    with pytest.raises(ValueError):
        db.facts_idx.reindex(residue, root=root)

    assert _rows(db) == before, "the refused calls changed the residue population"
    assert len([r for r in _rows(db) if r[0] == "blast-route"]) == 6
    assert db.facts_idx.for_entity("blast-route") == []          # 0 live, as on the live store
    assert len(db.facts_idx.for_entity("blast-route", include_expired=True)) == 6
    assert len([r for r in _rows(db) if r[0] == "Tidewell Relay"]) == 1


# ── the other escaped spelling: a `..` component ─────────────────────────────

def test_a_path_that_escapes_the_root_only_through_a_dotdot_component_is_refused(db, tmp_path):
    """`<root>/../elsewhere/x.md` is under the root as a STRING and outside it as a
    file, so the guard refuses that spelling too.

    `relative_to` compares unresolved parts, so it cheerfully returns
    `../elsewhere/x.md`, and the old fallback stored that verbatim — a
    `file_path` that resolves above the tree the caller named. It is the second of
    the two escaped spellings
    `scripts/memory/quarantine_general_facts.py::files_to_clear` names in its
    docstring ("a `..`, or an absolute path a rebuild left behind") and declines to
    hand back to `reindex`; the writer now refuses to create one. The file really is
    on disk here, so without the check the row would index cleanly under a path no
    walk of that root can reach.
    """
    root = _facts_root(tmp_path)
    outside = _fact_file(tmp_path / "elsewhere", "Junk Corps", "general",
                         [{"id": "x-1", "fact": "written outside the tree"}])
    escaping = Path(str(root) + "/../elsewhere/Junk Corps/Junk Corps-general.md")
    assert escaping.resolve() == outside.resolve(), escaping
    assert escaping.relative_to(root).as_posix().startswith(".."), escaping

    with pytest.raises(ValueError) as caught:
        db.facts_idx.update_file(escaping, root=root)

    msg = str(caught.value)
    assert str(escaping) in msg, f"the message does not name the refused path: {msg}"
    assert str(root) in msg, f"the message does not name the root it refused against: {msg}"
    assert _rows(db) == [], "the escaping path was indexed anyway"
