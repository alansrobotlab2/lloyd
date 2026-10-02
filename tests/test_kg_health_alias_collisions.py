"""kg_health's alias-collision watch (#2074).

Entry 3's ruling in #1234 — a declaration or write-path row beats a sweep
guess, the oldest row wins among case-variants, and `periodic memory capture`
means the skill Periodic Memory Capture and never the pipeline Memory Capture —
had no detector. The live store measured 0 collisions when this was written, so
nothing would tell the next run that a collision had returned, and a ruling that
nothing can check is a ruling that quietly stops holding.

Every count here is seeded through `app.kg_store`'s own alias write path and
read back through `build_snapshot()`, so the script's seam into the store
package is what is under test, not a dict the test built itself.

Live baseline these fixtures are the reduced shape of, measured at HEAD
`85554f68` on 2026-10-02 through `app.kg_store` over `app.paths.VAULT_KG_DB`:
190 alias rows (origins sweep 140, schema 50), 187 distinct `surface_lc`, 47 of
them touched by a schema row, 0 surfaces routing to more than one canonical, 0
resolve/all_lower divergences, and 13,041 `all_lower()` keys of which 12,854
have no alias row behind them. The alias table is rewritten by nightly sweeps,
so those totals drift; the zeros are the point, not the totals.
"""
import collections
import copy
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "kg_health_collisions", ROOT / "scripts" / "memory" / "kg_health.py")
kg_health = importlib.util.module_from_spec(_spec)
sys.modules["kg_health_collisions"] = kg_health
_spec.loader.exec_module(kg_health)

from app import kg_store  # noqa: E402


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A real store plus a one-directory facts root, so `build_snapshot` runs
    end to end over data this test owns (the shape
    `tests/test_kg_health_edge_types.py` uses). `_BASELINE` is the hygiene
    regrowth reference, deliberately a file that does not exist."""
    facts = tmp_path / "facts"
    (facts / "vllm").mkdir(parents=True)
    monkeypatch.setattr(kg_health, "VAULT_FACTS_ROOT", facts)
    s = kg_store.configure(tmp_path / "kg.sqlite")
    s.tmp_baseline = tmp_path / "no-baseline.json"
    yield s
    kg_store.reset()


def _snapshot(s):
    return kg_health.build_snapshot(baseline_path=s.tmp_baseline)


def _collisions(s):
    return _snapshot(s)["aliases"]["collisions"]


def _seed(s, *rows):
    """Write alias rows through the store's only alias write path.

    `surface` is the table's PRIMARY KEY, so two rows differing only in case
    both stand and share one `surface_lc` — exactly the shape entry 3 rules
    on."""
    for surface, canonical, kind, origin in rows:
        s.aliases.set(surface, canonical, kind=kind, origin=origin)


#: The item's own seed, spelled once: a migration row and a later schema row
#: whose only difference is case, naming two different canonicals.
COLLISION = ("x", "A", "case", "migration")
COLLISION_SCHEMA = ("X", "B", "semantic", "schema")


# ── clause 1: the multi-canonical count, snapshot and summary ────────────────

def test_the_snapshot_counts_a_surface_lc_routing_to_two_canonicals(db):
    """Clause 1: the number of `aliases.surface_lc` values whose rows name more
    than one canonical, alongside the total alias-row count in the same
    section."""
    _seed(db, COLLISION, COLLISION_SCHEMA, ("bar", "Baz", "case", "sweep"))

    col = _collisions(db)
    assert col["surfaces_multi_canonical"] == 1, col
    assert col["alias_rows"] == 3, col
    assert col["surfaces"] == 2, col          # 'x' (two rows) and 'bar'


def test_the_summary_prints_the_multi_canonical_count_beside_the_alias_row_total(db, capsys):
    """Clause 1, human half: the printed line carries the count AND the total
    alias-row count, so the count is never a bare number with no denominator —
    the #1535 `51 of 12027 new dirs` defect in a new costume."""
    _seed(db, COLLISION, COLLISION_SCHEMA, ("bar", "Baz", "case", "sweep"))

    kg_health.print_summary(_snapshot(db))
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if "alias collisions " in ln]
    assert len(lines) == 1, out
    first = lines[0]
    assert "surface_lc route to >1 canonical" in first, first
    # The count field itself, not a substring of the line: a printed 10 or 11
    # contains "1" and would pass a substring test.
    assert first.split("surface_lc route")[0].split()[-1] == "1", first
    assert "of 2 distinct surface_lc" in first, first
    assert "3 alias rows total" in first, first


# ── clause 2: resolve/all_lower divergence, including the resolve→None leg ───

def test_the_divergence_count_catches_a_case_variant_the_two_maps_answer_differently(db):
    """Clause 2 over the real store: `resolve('X')` answers the exact-case row
    and `all_lower()['x']` answers the oldest row, so one lowercase surface
    routes two ways and the count says 1, with its denominator printed beside
    it."""
    _seed(db, COLLISION, COLLISION_SCHEMA)

    assert db.aliases.resolve("X") == "B"
    assert db.aliases.all_lower()["x"] == "A"

    col = _collisions(db)
    assert col["resolve_all_lower_divergences"] == 1, col
    assert col["surfaces"] == 1, col


def test_the_divergence_line_prints_its_count_with_the_same_denominators(db, capsys):
    """Clause 2's printed half: the divergence count appears in the human
    summary with the same two denominators the multi-canonical line carries."""
    _seed(db, COLLISION, COLLISION_SCHEMA, ("bar", "Baz", "case", "sweep"))

    kg_health.print_summary(_snapshot(db))
    lines = [ln for ln in capsys.readouterr().out.splitlines()
             if "resolve/all_lower" in ln]
    assert len(lines) == 1, lines
    assert "resolve→None while the lower map names one" in lines[0], lines[0]
    assert lines[0].split("diverge")[0].split()[-1] == "1", lines[0]
    assert "of 2 distinct surface_lc" in lines[0], lines[0]
    assert "3 alias rows total" in lines[0], lines[0]


def test_the_none_leg_counts_a_surface_the_lower_map_names_and_resolve_does_not():
    """Clause 2's second half, on the view pair the clause names.

    Injected views rather than a store, because the leg needs them: `resolve()`
    reads the alias table and the population comes from that same table, so
    `resolve()` can answer `None` under a surface the map still names only when
    the cached hot map and the table are generations apart — the #1234 shape, a
    map still routing a surface whose row is gone. The predicate lives in
    `alias_collisions`, so it is pinned here directly.

    The mirror case is asserted NOT counted: `resolve()` names a canonical the
    map has no key for. An absent key holds no competing canonical, and the
    clause names one direction."""
    one_row = [{"surface": "vllm", "surface_lc": "vllm", "canonical": "vLLM",
                "kind": "case", "origin": "sweep"}]

    # Map names nothing anywhere: neither leg can fire.
    assert kg_health.alias_collisions(one_row, lambda s: None, {})[
        "resolve_all_lower_divergences"] == 0

    # Mirror case: resolve answers, the map has no key for that surface.
    missing_key = {"other": "Unrelated"}
    assert kg_health.alias_collisions(
        one_row, lambda s: "vLLM" if s == "vllm" else None, missing_key)[
        "resolve_all_lower_divergences"] == 0, "an absent key is not a rival canonical"

    # The leg itself: the row is there, resolve answers None, the map names one.
    orphaned = [{"surface": "gone", "surface_lc": "gone", "canonical": "Gone",
                 "kind": "semantic", "origin": "schema"}]
    assert kg_health.alias_collisions(orphaned, lambda s: None, {"gone": "Gone"})[
        "resolve_all_lower_divergences"] == 1
    assert kg_health.alias_collisions(orphaned, lambda s: None, {"gone": "Still Gone"})[
        "schema_resolve_all_lower_divergences"] == 1


# ── clause 3: the schema-restricted counts, any-row reading ──────────────────

def test_a_schema_row_anywhere_on_the_surface_counts_it_in_the_schema_figures(db):
    """Clause 3, reading (ii) — the one the item's own seed proves. The
    collision's schema row is the case-variant only, so filtering rows to
    `origin='schema'` BEFORE grouping would leave one canonical and report 0,
    while the item states 1. Provenance is a property of the surface, not of the
    row that happens to survive."""
    _seed(db, COLLISION, COLLISION_SCHEMA)

    col = _collisions(db)
    assert col["schema_rows"] == 1, col
    assert col["schema_touched_surfaces"] == 1, col
    assert col["schema_surfaces_multi_canonical"] == 1, col
    assert col["schema_resolve_all_lower_divergences"] == 1, col


def test_a_collision_no_schema_row_touched_leaves_the_schema_figures_at_zero(db):
    """The other half of telling a declaration collision from a sweep one: the
    same case-variant collision written entirely by sweep/migration keeps the
    unschema'd counts at 1 and the schema counts at 0, with the schema
    denominators still printed as 0 rather than the section disappearing."""
    _seed(db, ("x", "A", "case", "migration"), ("X", "B", "semantic", "sweep"))

    col = _collisions(db)
    assert (col["surfaces_multi_canonical"],
            col["resolve_all_lower_divergences"]) == (1, 1), col
    assert (col["schema_surfaces_multi_canonical"],
            col["schema_resolve_all_lower_divergences"]) == (0, 0), col
    assert (col["schema_rows"], col["schema_touched_surfaces"]) == (0, 0), col


def test_the_summary_prints_the_schema_counts_with_the_schema_row_total(db, capsys):
    """Clause 3's printed half: the schema line carries its own totals — the
    origin='schema' rows and the surface_lc they touch — beside both restricted
    counts, so a schema collision is distinguishable from a sweep one on the
    page a human reads."""
    _seed(db, COLLISION, COLLISION_SCHEMA, ("bar", "Baz", "case", "sweep"))

    kg_health.print_summary(_snapshot(db))
    lines = [ln for ln in capsys.readouterr().out.splitlines()
             if "alias collisions/schema" in ln]
    assert len(lines) == 1, lines
    line = lines[0]
    # Both restricted counts pinned as fields, not substrings: the multi-canonical
    # figure before "and", the divergent figure between "and" and "divergent".
    assert line.split("and")[0].split()[-1] == "1", line
    assert line.split(" and ")[1].split("divergent")[0].strip() == "1", line
    assert "of 1 surface_lc touched by" in line, line
    assert "1 origin='schema' row " in line, line
    assert "3 alias rows total" in line, line


# ── clause 4: the two fixed points, 1/1/1 and 0/0/0 ──────────────────────────

def test_the_items_seed_yields_one_one_one(db):
    """Clause 4, first half, on the item's literal seed —
    `('x','A','case','migration')` then `('X','B','semantic','schema')` — which
    must give multiple-canonical 1, resolve/all_lower divergence 1 and
    schema-restricted 1."""
    _seed(db, COLLISION, COLLISION_SCHEMA)

    col = _collisions(db)
    assert col["surfaces_multi_canonical"] == 1, col
    assert col["resolve_all_lower_divergences"] == 1, col
    assert col["schema_surfaces_multi_canonical"] == 1, col
    assert col["schema_resolve_all_lower_divergences"] == 1, col


def test_a_store_with_one_canonical_per_surface_yields_zero_zero_zero(db):
    """Clause 4, second half: one canonical per `surface_lc`, one of the two
    surfaces written by a schema row, and every count is 0 — while both
    denominators stay non-zero and visible, because 0-of-3 and 0-of-nothing must
    not print identically."""
    _seed(db, ("bar", "Baz", "case", "sweep"), ("qux", "Quux", "semantic", "schema"))

    col = _collisions(db)
    assert col["surfaces_multi_canonical"] == 0, col
    assert col["resolve_all_lower_divergences"] == 0, col
    assert col["schema_surfaces_multi_canonical"] == 0, col
    assert col["schema_resolve_all_lower_divergences"] == 0, col
    assert (col["alias_rows"], col["surfaces"]) == (2, 2), col
    assert (col["schema_rows"], col["schema_touched_surfaces"]) == (1, 1), col


def test_two_case_variants_naming_the_same_canonical_are_not_a_collision(db):
    """The ruling is about a surface routing two places, not about case variants
    existing: 'x' and 'X' both answering `A` is what entry 3's oldest-row-wins
    rule produces on a healthy store, and it must not trip the watch. The live
    store has case variants in it and prints 0."""
    _seed(db, ("x", "A", "case", "migration"), ("X", "A", "case", "sweep"))

    col = _collisions(db)
    assert col["surfaces_multi_canonical"] == 0, col
    assert col["resolve_all_lower_divergences"] == 0, col


# ── clause 5: building the snapshot leaves both alias maps as it found them ───

def _alias_state(s):
    """Every alias row plus both maps, deep-copied.

    The copy is load-bearing: `all_lower()` hands back the store's cached dict,
    so holding the reference would let an in-place mutation edit the "before"
    state as well, and the test could never see the damage it exists to catch."""
    return {
        "rows": copy.deepcopy(s.aliases.rows()),
        "all": dict(s.aliases.all()),
        "all_lower": dict(s.aliases.all_lower()),
    }


def test_building_a_snapshot_mutates_neither_alias_map(db):
    """Clause 5: the full alias row set, `aliases.all()` and
    `aliases.all_lower()` are equal before and after `build_snapshot()`. This is
    the read a nightly job makes; a write hiding inside the instrumentation
    would rewrite the alias map from the thing that is supposed to only look at
    it."""
    _seed(db, COLLISION, COLLISION_SCHEMA, ("bar", "Baz", "case", "sweep"))
    before = _alias_state(db)

    snap = _snapshot(db)
    assert snap["aliases"]["collisions"]["surfaces_multi_canonical"] == 1, snap
    assert _alias_state(db) == before

    # The comparison is not vacuous: one write through the alias API and the
    # same before/after comparison goes red. That is the only evidence this
    # witness would notice the mutation clause 5 forbids.
    db.aliases.set("zoo", "Zoo Keeper", kind="semantic", origin="schema")
    assert _alias_state(db) != before, "the witness stopped being sensitive to a write"


#: The committed alias extract (clause 6): the full 190-row alias table as read
#: through `app.kg_store` over `app.paths.VAULT_KG_DB` at HEAD `85554f68` on
#: 2026-10-02, one JSON object per row, pinned by its SHA-256 content checksum in
#: `test_the_committed_witness_bytes_have_a_known_digest` — a file hash, not a git
#: object id; the commit that adds these bytes is `d0232a83` on
#: `automod/SM_20261002_191225`. Bytes rather than
#: prose, because the live store is rewritten nightly by the alias sweep and the
#: figures the item quotes must be re-derivable after it moves. Item #2074 clause 6
#: named the vault's witness directory instead, and #2075 put a copy of these
#: bytes there. What bars the path the clause spelled — `backlog/data/kg.sqlite`
#: — is the thing itself: a 116 MB sqlite binary is unreachable, and no 116 MB
#: binary belongs in a git tree either. The extract is text, so
#: the JSONL's location is a choice, and it was made twice on purpose: the
#: durable copy at vault `backlog/data/2026-10-02.2074-kg-alias-witness.jsonl`,
#: and this file, which
#: `test_the_vault_copy_of_the_witness_is_the_same_bytes_the_fixture_holds`
#: holds against it byte for byte.
WITNESS = ROOT / "tests" / "fixtures" / "kg_alias_witness_2074.jsonl"


def _store_views_from(rows):
    """The two views `alias_collisions` compares, rebuilt from alias rows by the
    semantics `app/kg_store.py` documents, so the committed bytes can be run
    through the shipped predicate without a database.

    `all_lower()` takes the OLDEST row per `surface_lc`
    (`app/kg_store.py:991-992`); `resolve()` prefers the exact-case spelling and
    only then falls back to the oldest case-insensitive match
    (`app/kg_store.py:913-919`). Both orderings are oldest-first, so a
    case-variant row written later changes `resolve()` and not the map — which
    is the divergence the watch counts."""
    by_lc = collections.defaultdict(list)
    by_surface = collections.defaultdict(list)
    for r in rows:
        by_lc[r["surface_lc"]].append(r)
        by_surface[r["surface"]].append(r)
    oldest = lambda rs: min(rs, key=lambda r: r["created_at"])["canonical"]

    def resolve(surface):
        if not surface.strip():
            return None
        if by_surface.get(surface):
            return oldest(by_surface[surface])
        match = by_lc.get(surface.lower())
        return oldest(match) if match else None

    return resolve, {lc: oldest(g) for lc, g in by_lc.items()}


def test_the_committed_witness_bytes_reproduce_the_quoted_figures():
    """Clause 6: the quoted report re-derived from committed bytes alone.

    Clause 6 prescribes a sqlite re-derivation over a copy of the live store at
    `backlog/data/kg.sqlite`, and neither half of it works as written. The path:
    a 116 MB sqlite binary is unreachable, and no git tree should carry one
    either, so the bar is the store's size and format and not any write route —
    that directory already holds `.jsonl` witnesses the loop landed itself (vault
    `eb95dce0`, #2046 clause 6), and the JSONL's location is a choice. #2075 made
    the choice twice: these bytes, and a copy at
    `backlog/data/2026-10-02.2074-kg-alias-witness.jsonl` which
    `test_the_vault_copy_of_the_witness_is_the_same_bytes_the_fixture_holds`
    holds against this file. The command: `select count(*) from sqlite_master`
    counts schema objects — measured 3 over an aliases-only extract, with 190
    alias rows underneath it — so it cannot print the 190/50 figures the clause
    says its output is. This node is the deliverable the clause was reaching for:
    the bytes, and the six numbers derived from them.

    The six numbers the item quotes — 190 alias rows, 50 of them
    `origin='schema'`, 0 surfaces routing to more than one canonical, 0 of those
    schema-touched, 187 distinct `surface_lc`, 47 touched by a schema row — come
    out of this file with no live store in sight, and the shipped predicate run
    over the same rows with the store's documented view semantics reports 0/0/0.
    `kg_health`'s own renderer then prints them as the line the nightly job
    will print."""
    rows = [json.loads(line) for line in WITNESS.read_text().splitlines()]
    by_lc = collections.defaultdict(list)
    for r in rows:
        by_lc[r["surface_lc"]].append(r)

    assert len(rows) == 190, len(rows)
    assert sum(1 for r in rows if r["origin"] == "schema") == 50
    assert len(by_lc) == 187, len(by_lc)
    multi = {lc for lc, g in by_lc.items() if len({x["canonical"] for x in g}) > 1}
    assert multi == set(), multi
    assert sum(1 for lc, g in by_lc.items()
               if any(x["origin"] == "schema" for x in g)) == 47

    col = kg_health.alias_collisions(rows, *_store_views_from(rows))
    assert col == {
        "alias_rows": 190, "schema_rows": 50, "surfaces": 187,
        "schema_touched_surfaces": 47, "surfaces_multi_canonical": 0,
        "resolve_all_lower_divergences": 0,
        "schema_surfaces_multi_canonical": 0,
        "schema_resolve_all_lower_divergences": 0,
    }, col

    lines = kg_health.alias_collision_lines(col)
    assert "of 187 distinct surface_lc, 190 alias rows total" in lines[0], lines[0]
    assert "of 47 surface_lc touched by 50 origin='schema' rows" in lines[2], lines[2]


def test_the_committed_witness_bytes_have_a_known_digest():
    """The witness is bytes, not a description of bytes: one edited row and the
    digest moves, so the six figures the item quotes stay attributable to THIS
    extract and not to whatever the live store happens to hold the night someone
    re-runs the report."""
    assert hashlib.sha256(WITNESS.read_bytes()).hexdigest() == (
        "313071833aab7f0ebdaf97e1cd6fca172fa15715952217a5b3c823bbceed2ceb"
    ), f"{WITNESS} changed; re-extract it or update the quoted figures deliberately"


def test_the_witness_plus_one_seeded_case_variant_collision_reports_one():
    """The same predicate over the 190 production rows plus one seeded
    collision: the real corpus neither dilutes nor hides it. A watch that reads
    0 on production and 1 only on a two-row toy has not been shown to work on
    the corpus it will actually run against.

    The seed is aimed at a surface the witness really holds: `the graph` has one
    row, the exact-case spelling `The Graph` → Knowledge Graph, and an alias on a
    fresh `surface_lc` could not collide with anything. Adding a later,
    all-lowercase row under a different canonical is what a second declaration or
    a write path does — `resolve('the graph')` then answers the new canonical
    while `all_lower()['the graph']` still answers the oldest one."""
    rows = [json.loads(line) for line in WITNESS.read_text().splitlines()]
    graph = [r for r in rows if r["surface_lc"] == "the graph"]
    assert [(r["surface"], r["canonical"]) for r in graph] == [("The Graph", "Knowledge Graph")], graph
    rows.append({"surface": "the graph", "surface_lc": "the graph",
                 "canonical": "GraphRAG", "kind": "case", "origin": "schema",
                 "created_at": "2099-01-01T00:00:00+00:00", "report_path": None})

    col = kg_health.alias_collisions(rows, *_store_views_from(rows))
    assert col["surfaces_multi_canonical"] == 1, col
    assert col["resolve_all_lower_divergences"] == 1, col
    assert col["schema_surfaces_multi_canonical"] == 1, col
    assert col["schema_resolve_all_lower_divergences"] == 1, col
    # The seed collided an existing surface; it did not add one.
    assert (col["alias_rows"], col["surfaces"]) == (191, 187), col
    assert (col["schema_rows"], col["schema_touched_surfaces"]) == (51, 47), col


def test_the_summary_renders_a_snapshot_that_predates_the_check_without_fake_zeros(db, capsys):
    """A snapshot dict with no `collisions` key — the JSONs already on disk
    under `_pipeline/metrics/` — prints no collision line. Printing `0 of 0` for
    a run that never measured would put a false green on the very surface this
    item exists to make trustworthy."""
    snap = _snapshot(db)
    del snap["aliases"]["collisions"]

    kg_health.print_summary(snap)
    out = capsys.readouterr().out
    assert "alias collision" not in out, out
    assert "alias coverage" in out, out


# ── #2075: where the witness lives, why it lives there, and what the note publishes ──

#: The reason this witness is text and why it sits in two places, stated once here
#: so the pin below can count it, plus the four phrasings of the excuse that was
#: false — that a witness cannot reach the vault's witness directory because the
#: loop writes it with text. The loop put a `.jsonl` there for #2046 clause 6
#: (vault commit `eb95dce0`), and `scripts/automod/backlog.py` declares that
#: directory `WITNESS_ARTIFACT_DIR` with `jsonl` among `_LINE_WITNESS_EXTS` ("a
#: line count IS the quoted report"). Every string is assembled from fragments so
#: that this module's own source never holds the whole of any of them: the excuses
#: must count ZERO in this file, and the reason must count in the PROSE — a node
#: that quoted either plainly would only be checking itself.
SIZE_REASON = "116 MB sqlite binary is " + "unreachable"
LOCATION_IS_A_CHOICE = "the JSONL's location is " + "a choice"
ABSENT_LOCATION_EXCUSES = (
    "write routes are " + "text-only",
    "vault write routes take " + "text",
    "no binary write " + "route",
    "no binary witness can be " + "committed",
)

#: The vault's copy of the witness and its pointer note, named by path rather than
#: looked up: like `tests/test_action_review_calibration.py`'s `WITNESS`, this is a
#: specific committed artifact in this box's vault, not a vault-shaped directory a
#: fixture could stand in for.
VAULT = Path.home() / "obsidian"
VAULT_WITNESS = VAULT / "backlog" / "data" / "2026-10-02.2074-kg-alias-witness.jsonl"
VAULT_NOTE = VAULT / "backlog" / "data" / "2026-10-02.2074-kg-alias-witness.md"
WITNESS_SHA = "313071833aab7f0ebdaf97e1cd6fca172fa15715952217a5b3c823bbceed2ceb"
#: The six figures in the order the published command prints them: alias rows,
#: `origin='schema'` rows, `surface_lc` routing to more than one canonical, the same
#: restricted to schema-touched surfaces, distinct `surface_lc`, `surface_lc` touched
#: by a schema row.
SIX_FIGURES = "190 50 0 0 187 47"


def _vault_git(*args):
    return subprocess.run(["git", "-C", str(VAULT), *args],
                          capture_output=True, text=True)


def test_the_module_gives_the_size_reason_and_never_the_route_excuse():
    """#2075 clause 3: this module states the real reason, and pins that it does.

    The excuse clause 6 of #2074 was refused for is falsified by the loop's own
    history, so it cannot be re-argued: `git -C ~/obsidian log --oneline --
    backlog/data/2026-10-02.2046-parallel-flaker-witness.jsonl` names `eb95dce0`, a
    `.jsonl` witness landed through `automod_vault_land`. What is true in the old
    sentence is the 116 MB. A comment that keeps repeating the false half teaches
    the next round to skip a route that works, which is how a witness ends up in
    one place when it was reachable in two."""
    src = Path(__file__).resolve().read_text(encoding="utf-8")

    for excuse in ABSENT_LOCATION_EXCUSES:
        assert src.count(excuse) == 0, (
            f"{excuse!r} is back in this module: the limit is the sqlite store's size, "
            "not the route — the loop has landed .jsonl witnesses into backlog/data "
            "itself (vault eb95dce0, #2046 clause 6)")

    # Exactly two copies, both of them prose: the comment over WITNESS and the
    # docstring of the clause 6 node. Those are the two sites that carried the
    # excuse, and each has to carry the reason now — deleting the excuse and
    # leaving silence would let the next reader reconstruct the wrong conclusion.
    assert src.count(SIZE_REASON) == 2, f"{SIZE_REASON!r} x{src.count(SIZE_REASON)}"
    assert src.count(LOCATION_IS_A_CHOICE) == 2, (
        f"{LOCATION_IS_A_CHOICE!r} x{src.count(LOCATION_IS_A_CHOICE)}")


def test_the_vault_copy_of_the_witness_is_the_same_bytes_the_fixture_holds():
    """#2075 clause 1: the witness is reachable at the path the clause's directory
    names, byte for byte, and is committed rather than merely present.

    Three claims, because each is satisfiable alone by accident: the bytes exist
    there; they are the fixture's bytes and not a re-typed lookalike of them; and
    git on the vault's `main` holds them — a working-tree copy that `git log` does
    not name is a local file, not a witness a second reader can re-check."""
    assert VAULT_WITNESS.is_file(), (
        f"{VAULT_WITNESS} is absent, so the pointer note points at nothing")
    blob = VAULT_WITNESS.read_bytes()
    assert blob.count(b"\n") == 190, blob.count(b"\n")
    assert len(blob) == 53_409, len(blob)
    assert hashlib.sha256(blob).hexdigest() == WITNESS_SHA
    assert blob == WITNESS.read_bytes(), (
        "the vault copy and the repo fixture have drifted apart, so the one digest the "
        "note states for both is now two answers to which bytes are the witness")

    assert _vault_git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "main", (
        "the vault is not on main, so 'committed on the vault's main' is not what this "
        "copy can claim")
    rel = str(VAULT_WITNESS.relative_to(VAULT))
    assert _vault_git("log", "--oneline", "-1", "--", rel).stdout.strip(), (
        f"{rel} is not in the vault's history — untracked, so only this box holds it")
    assert _vault_git("diff", "--quiet", "--", rel).returncode == 0, (
        f"{rel} on disk differs from the copy the vault's git holds")


def test_the_note_names_the_vault_copy_as_the_witness_and_states_one_digest_for_both():
    """#2075 clause 2: the pointer note says which copy is durable, gives the same
    checksum to both, and no longer argues the repo copy exists because the vault
    copy could not.

    The digest is read out of the note and compared to the bytes, not written a
    third time here: a second copy of a digest is a second answer to which bytes
    are the witness, and the two drift exactly when it matters."""
    assert VAULT_NOTE.is_file(), VAULT_NOTE
    note = VAULT_NOTE.read_text(encoding="utf-8")

    for excuse in ABSENT_LOCATION_EXCUSES:
        assert note.count(excuse) == 0, (
            f"the note is back to claiming {excuse!r}, which vault eb95dce0 disproves")
    assert SIZE_REASON in note and LOCATION_IS_A_CHOICE in note, (
        "the note has to give the reason in the same words the module pins, or the two "
        "surfaces are telling two versions of why the .sqlite path is empty")
    assert "durable witness" in note, "the note must say which copy is the durable one"
    assert VAULT_WITNESS.name in note and WITNESS.name in note, (
        "both paths have to be named for the two digests to be comparable")

    stated = re.findall(r"\b[0-9a-f]{64}\b", note)
    assert len(stated) >= 2, stated
    assert set(stated) == {WITNESS_SHA}, stated
    assert WITNESS_SHA == hashlib.sha256(WITNESS.read_bytes()).hexdigest()


def test_the_note_published_command_runs_over_the_vault_copy_and_prints_the_six_figures():
    """#2075 clause 4: the re-derivation the note publishes runs over the VAULT
    copy, not only over a repo checkout, and prints the six figures.

    Extracted from the note's own fenced block and executed with the vault root as
    cwd, which is what the note tells its reader to use — the seam under test is
    prose to shell, the one every later run relies on. A note whose command rotted,
    or that still opens only `tests/fixtures/`, fails here instead of in the hands
    of whoever next follows it. The six numbers are the six
    `test_the_committed_witness_bytes_reproduce_the_quoted_figures` asserts."""
    note = VAULT_NOTE.read_text(encoding="utf-8")
    published = [ln.strip()[len('python3 -c "'):].rsplit('"', 1)[0]
                 for ln in note.splitlines() if ln.strip().startswith('python3 -c "')]
    assert published, f"{VAULT_NOTE} publishes no command to run"
    over_vault_copy = [c for c in published if VAULT_WITNESS.name in c]
    assert over_vault_copy, (
        "the published re-derivation still opens only the repo fixture, so a reader "
        f"holding just the vault cannot run it — nothing in the note names {VAULT_WITNESS.name}")
    for cmd in over_vault_copy:
        assert "sqlite_master" not in cmd, (
            "sqlite_master counts schema objects, never alias rows: that is the command "
            "clause 6 named, and why it could not print 190/50")
        run = subprocess.run([sys.executable, "-c", cmd], cwd=str(VAULT),
                             capture_output=True, text=True)
        assert run.returncode == 0, run.stderr
        assert run.stdout.strip() == SIX_FIGURES, run.stdout


# ── #2076: the helper's two views must answer the way the store answers ───────
#
# `_store_views_from` decides, inside this file, what `resolve()` and
# `all_lower()` would answer, and the two witness nodes above run the shipped
# predicate over those rebuilt views. Its docstring says the rebuild follows
# "the semantics `app/kg_store.py` documents"; until this section nothing
# checked that sentence. The node below does: seed a store through the store's
# own alias write path, then require the model and the store to answer the same
# thing for every spelling and every `surface_lc` the seed holds.

#: The equivalence seed, spelled once. Three case variants share one
#: `surface_lc` at three distinct `created_at`s and name two canonicals, so
#: `resolve('Alpha')` answers the exact-case row while `all_lower()['alpha']`
#: answers the oldest row — the divergence the watch counts, inside one group.
#: `beta` / `Beta` is the mixed-origin group (`schema` + `sweep`) on one
#: canonical, and `gamma` is a lone row: the uncontested case, so the node also
#: proves the model matches where nothing is in dispute.
_EQUIV_SEED = (
    # surface, canonical, kind, origin, created_at (oldest first by intent)
    ("alpha", "Alpha Entity", "case", "sweep", "2026-01-01T00:00:00+00:00"),
    ("ALPHA", "Alpha Entity", "case", "migration", "2026-02-01T00:00:00+00:00"),
    ("Alpha", "Bravo Entity", "semantic", "schema", "2026-03-01T00:00:00+00:00"),
    ("beta", "Beta Widget", "case", "schema", "2026-01-05T00:00:00+00:00"),
    ("Beta", "Beta Widget", "case", "sweep", "2026-02-05T00:00:00+00:00"),
    ("gamma", "Gamma Index", "case", "sweep", "2026-01-09T00:00:00+00:00"),
)


def test_the_rebuilt_views_answer_every_surface_the_store_answers(db, monkeypatch):
    """#2076: `_store_views_from` is checked against `app.kg_store`, not asserted.

    Why the seed needs the clock and not just `aliases.set`: the store's only
    alias writer (`set` at `app/kg_store.py:920`) stamps `created_at` from
    `_now()` (`:177`) inside its insert at `:933-937` and takes no timestamp
    argument, and the ordering this node is about IS `created_at`. So the monkey-
    patch hands the write path the seed's stamps in row order and every row still
    goes through the store's own write path — `_seed` is not used here precisely
    because it lets the wall clock decide the order.

    What the store's side of the comparison is: `resolve()` (`:905-919`) answers
    the OLDEST row of the exact-case spelling when one exists and the oldest
    case-insensitive match otherwise; `all_lower()` (`:991-992`) takes the oldest
    row per `surface_lc`. Both are oldest-first, so a later case-variant row moves
    `resolve()` and not the map. The map is read after seeding and after
    `invalidate_caches()`, because it is memoised (`cached("alias_map_lower", …)`
    at `:996`, keyed on `PRAGMA data_version` by `KGStore.cached` at `:316-325`)
    and a read taken before the last write would be compared against a build that
    never saw it.

    The map comparison is one-directional over the seed's own `surface_lc` keys,
    never full-dict equality: `all_lower()` also folds `entities.name_lc → name`
    (`:993-995`), so the store's map is strictly larger than the model's by
    design, and `assert model == store` would fail on entity self-identities
    rather than on anything about aliases.

    The seed is not vacuous, and the node says so twice. The row-order assert
    pins the three `alpha` stamps the tie-break reads, so the node cannot go
    quiet if the clock patch ever stops landing; and the disagreement assert
    names the one seeded spelling where the two views answer differently —
    `resolve('Alpha')` is Bravo Entity, `all_lower()['alpha']` is Alpha Entity —
    which is the shape that makes a changed tie-break flip an answer instead of
    re-printing the same canonical. Measured against the committed witness, that
    shape is exactly what the witness does NOT have: with `oldest` flipped from
    `min` to `max`, every `surface_lc` in the 190-row extract holds one canonical,
    so no answer moves and both witness nodes stay green. This node is the one
    that goes red."""
    stamps = [row[4] for row in _EQUIV_SEED]
    monkeypatch.setattr(kg_store, "_now", lambda: stamps.pop(0))
    for surface, canonical, kind, origin, _when in _EQUIV_SEED:
        db.aliases.set(surface, canonical, kind=kind, origin=origin)
    assert stamps == [], \
        f"the seeded clock was read {len(_EQUIV_SEED) - len(stamps)} of " \
        f"{len(_EQUIV_SEED)} writes; unused stamps {stamps}"

    rows = db.aliases.rows()
    assert len(rows) == len(_EQUIV_SEED), \
        f"`surface` is the PRIMARY KEY (`app/kg_store.py:100-109`) and `set` " \
        f"upserts ON CONFLICT(surface) (`:934`), so a seed that silently " \
        f"replaced a row would shrink this set: {len(rows)} rows"
    by_lc_seed = sorted((r["surface"], r["created_at"], r["canonical"])
                        for r in rows if r["surface_lc"] == "alpha")
    assert by_lc_seed == [
        ("ALPHA", "2026-02-01T00:00:00+00:00", "Alpha Entity"),
        ("Alpha", "2026-03-01T00:00:00+00:00", "Bravo Entity"),
        ("alpha", "2026-01-01T00:00:00+00:00", "Alpha Entity"),
    ], "the tie-break's input order is not what this node reasons about"
    assert {r["origin"] for r in rows if r["surface_lc"] == "beta"} == {"schema", "sweep"}, \
        "the mixed-origin group is not mixed"

    db.invalidate_caches()
    model_resolve, model_lower = _store_views_from(rows)
    store_lower = db.aliases.all_lower()

    for row in rows:
        surface = row["surface"]
        assert model_resolve(surface) == db.aliases.resolve(surface), \
            f"resolve({surface!r}): model {model_resolve(surface)!r} vs store " \
            f"{db.aliases.resolve(surface)!r}"
    # Each spelling's swapcase is a spelling NO row carries, so this leg walks
    # the case-insensitive fallback in both sides rather than the exact-case
    # branch the loop above already covers.
    for row in rows:
        spelled = row["surface"].swapcase()
        assert model_resolve(spelled) == db.aliases.resolve(spelled), \
            f"resolve({spelled!r}) (no row of that spelling): model " \
            f"{model_resolve(spelled)!r} vs store {db.aliases.resolve(spelled)!r}"
    for surface_lc in sorted({r["surface_lc"] for r in rows}):
        assert model_lower[surface_lc] == store_lower[surface_lc], \
            f"all_lower()[{surface_lc!r}]: model {model_lower[surface_lc]!r} vs " \
            f"store {store_lower[surface_lc]!r}"

    disagree = sorted(row["surface"] for row in rows
                      if db.aliases.resolve(row["surface"])
                      != store_lower[row["surface_lc"]])
    assert disagree == ["Alpha"], \
        f"the equivalence must be checked where the two orderings disagree; this " \
        f"seed has {disagree}"
