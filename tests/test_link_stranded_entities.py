"""The stranded-entity linker (#1019): a name that embeds another registered
entity name is mechanical evidence of a relation, so degree zero stops being fate.

#1019's premise, on the 2026-09-18 store: `#363 TGS-RAG Implementation` held 16
active facts with 0 edges active or expired and 0 alias rows, so no graph leg of
retrieval could reach it; 13,589 of 27,552 fact-holding entities sat at degree
zero holding 20.9% of the corpus. Every mechanism that writes an edge reads fact
*prose* (`fact_extractor.py:551`) or a `-relationship.md`
(`seed_relationship_edges.py:205`), and a self-referential entity can have
neither — 94.3% of those stranded facts have `source_doc` NULL, so there is no
document to derive an edge from. The entity's own name is the only evidence left,
and it is enough: `#363 TGS-RAG Implementation` contains `TGS-RAG`.

The fixture is that premise transplanted: a row the probe can name, one stranded
with the same fact count that the boundary rule cannot name (`CameraCfg`), one
that embeds nothing (`UniTacHand`), and the bare/role-noun pair
(`Browser` / `Browser Tool`) that #320 refuses to merge and this rule links.

The four clauses of #1019's acceptance are pinned here in the order the contract
numbers them: the shape of one proposal (clause 1), the degree-zero probe going
quiet on a fixture store (clause 2), the rebuild-durable stamp with its evidence
pointer (clause 3), and the denominators beside the count (clause 4).

The last section is #1833's write gate: the tool's `--apply` is a hand run over a
production store, so it has to refuse while `kg_rebuild.py` holds
`knowledge_graph.write_enabled` false, and every test in that section drives the
real guard with a config the test owns — never by patching the predicate away.
"""
import ast
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "memory" / "link_stranded_entities.py"
sys.path.insert(0, str(ROOT))

from app.kg_store import CARRY_EDGE_ORIGINS, KGStore  # noqa: E402


def _load_linker():
    spec = importlib.util.spec_from_file_location("link_stranded_entities", SCRIPT)
    assert spec and spec.loader, "the linker script is gone"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


linker = _load_linker()

STRANDED = "#363 TGS-RAG Implementation"   # the row #1019 is named for
VICTIM = "CameraCfg"        # stranded, and the rule's named cost: cannot name it
ALONE = "UniTacHand"        # stranded, embeds nothing at all
DUAL = "Obsidian Dataview"  # 16 facts at degree zero on the live store today
TARGET = "Obsidian"
ROLE_NOUN = "Browser Tool"  # the bare/role-noun shape #320 refuses to merge
BARE = "Browser"

FIXTURE_FACTS = {
    DUAL: 16,
    STRANDED: 12,
    VICTIM: 11,
    ALONE: 11,
    # Everything below sits under the probe's 10-fact floor: a target the matcher
    # reaches without becoming a probe hit of its own. The live store has that
    # shape — `TGS-RAG` carries 222 active facts and 101 active edges, so it is
    # neither stranded nor a probe candidate while being what the stranded row
    # should point at.
    "TGS-RAG": 5,       # embedded in STRANDED; `RAG` is nested inside it
    TARGET: 5,          # embedded in DUAL at a space boundary
    ROLE_NOUN: 4,       # embeds BARE at a space boundary
    "RAG": 4,
    BARE: 3,
    "Cfg": 3,           # inside VICTIM at no boundary, so it links nothing
    # A candidate added because the target bound takes one away: the edge that makes
    # `TGS-RAG` an admissible TARGET also lifts it out of the degree-zero candidate
    # pool, so its own proposal to `RAG` cannot survive #1618. `Obsidian Canvas`
    # embeds `Obsidian` and holds the proposal count at four, so the write-path nodes
    # below keep counting four edges and only the target-side story changes.
    "Obsidian Canvas": 11,
}

# `find_proposals` now reads the TARGET's degree as well as its name, so a target
# holding no edge is refused as dead and this fixture would propose nothing at all.
# Three edges, one per target the four proposals use, each to a registered name with
# no facts of its own — a partner holding facts would stop being a degree-zero
# candidate and cost the pass a proposal (the `TGS-RAG` case above).
#
# Each target then clears the bound by a different half of its third condition, which
# is what keeps the bound from being pinned by one shape alone: `TGS-RAG` fan-out 1 /
# 5 facts, `Obsidian` fan-out 2 (`Obsidian Dataview`, `Obsidian Canvas`) / 5 facts,
# `Browser` fan-out 1 (`Browser Tool`) / 3 facts — under `TARGET_FACTS_MIN`, admitted
# only because its fan-out is at or below `TARGET_FAN_OUT_TRUSTED`.
FIXTURE_EDGES = [("TGS-RAG", "Quartz Broker", "related_to"),
                 (TARGET, "Zephyr Scheduler", "related_to"),
                 (BARE, "Cobalt Registry", "related_to")]


def link_fixture_edges(st) -> None:
    """Write `FIXTURE_EDGES` through the store's own writer, not raw SQL."""
    for s, d, ty in FIXTURE_EDGES:
        st.edges.add({"source": s, "target": d, "type": ty,
                      "origin": "manual", "evidence": "fixture scaffolding for #1832"})


def _edge_count(st, name):
    return st._query(
        "SELECT COUNT(*) c FROM edges WHERE (source=? OR target=?) AND expired_at IS NULL",
        (name, name))[0]["c"]


def _shown_number(out: str, label: str) -> int:
    """The count the report prints beside `label`, read from the printed line.

    Takes the first grouped number after the label rather than a fixed column
    offset, so the test pins that a number is there and correct without
    re-implementing the formatter — and cannot mistake the `(of N …)` tail that
    clause 4 hangs on the same line for the count itself.
    """
    line = next(l for l in out.splitlines() if label in l)
    tail = line.split(label)[1]
    assert tail.startswith("  "), f"{label!r} lost its column padding: {line!r}"
    m = re.search(r"[\d,]+", tail)
    assert m, f"no number printed beside {label!r}: {line!r}"
    return int(m.group(0).replace(",", ""))


def build_store(tmp_path) -> KGStore:
    """Registered entity rows plus a `facts_idx` in the shape the live tree
    indexes: one file per entity/category, `text_hash` unique per fact (that
    index is what would reject a second insert of one fact), no expiry, and the
    target-side edges in `FIXTURE_EDGES`.

    Those edges are what the #1618 bound needs in order to exist: with no edge at
    all every target here sits at degree zero, the bound refuses every proposal as a
    dead target, and a fixture that cannot propose anything pins nothing."""
    st = KGStore(tmp_path / "kg.sqlite")
    for name, n_facts in FIXTURE_FACTS.items():
        st.entities.register(name)
        for i in range(n_facts):
            category = "state" if i % 3 else "event"
            st._query(
                "INSERT INTO facts_idx(entity, category, fact_id, text_hash, fact, "
                "confidence, created_at, provenance, file_path) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (name, category, f"fact-{i:03d}",
                 f"h-{name}-{category}-{i}", f"{name} claim number {i}",
                 0.9, "2026-09-18T00:00:00+00:00", "EXTRACTED",
                 f"{name}/{name}-{category}.md"))
    for partner, kind in (("Quartz Broker", "system"), ("Zephyr Scheduler", "system"),
                          ("Cobalt Registry", "system")):
        st.entities.register(partner, kind=kind)
    link_fixture_edges(st)
    return st


# ── clause 1: exactly one edge, to the longest other name the entity embeds ──

@pytest.mark.parametrize("name,winner", [
    # Longest wins: `TGS-RAG` (7) over the `RAG` (3) nested inside it.
    (STRANDED, "TGS-RAG"),
    # The rule's cost, pinned so the trade stays visible: `Cfg` occurs inside
    # `CameraCfg` only inside the word, so `CameraCfg` stays stranded.
    (VICTIM, None),
    # The bare/role-noun shape #320 refuses to merge IS linked: `Browser` sits
    # inside `Browser Tool` at a space boundary.
    (ROLE_NOUN, BARE),
    # Embeds no registered name at all.
    (ALONE, None),
    # A name equal to a registered name modulo case is not a SECOND entity, but
    # the shorter name inside it still is.
    ("tgs-rag", "RAG"),
    # Space-separated names embed too, which a token index would miss.
    ("Camera Cfg Sub", "Cfg Sub"),
])
def test_one_proposal_to_the_longest_other_registered_name(name, winner):
    index = linker.build_target_index(list(FIXTURE_FACTS) + ["Cfg Sub"])
    hit = linker.longest_embedded_name(name, index)
    if winner is None:
        assert hit is None, f"{name} embeds {hit[0]!r}, which it must not"
    else:
        assert hit == (winner, winner.lower())


def test_a_name_occurring_only_inside_a_longer_word_is_not_an_embedding():
    """The boundary rule exists because of what plain substring matching did.

    The first dry run over a copy of the live store matched on plain substring and
    proposed 1,694 edges including `Geometric Deep Learning → METR` and
    `Harrison Chase → HASE`: `metr` is inside `geometric`, `hase` inside `Chase`.
    Those are not relations, and #1019's own verification probe counts only
    boundary-clean embedding, so a wider matcher would write edges its check
    refuses to see. Under the boundary rule that same dry run proposes 938.
    """
    index = linker.build_target_index(
        list(FIXTURE_FACTS) + ["METR", "HASE", "TGS"])
    for name in ("Geometric Deep Learning", "Harrison Chase", "Dragon Boat Festival"):
        assert linker.longest_embedded_name(name, index) is None, name
    # …and the same needle does match where the boundary is real.
    assert linker.longest_embedded_name("TGS Overview", index) == ("TGS", "tgs")


def test_equal_length_ties_break_alphabetically_so_two_runs_agree():
    index = linker.build_target_index(["Gap", "Rag", "Hortense"])
    hit = linker.longest_embedded_name("Rag Gap Thing", index)
    assert hit == ("Gap", "gap"), f"tie must break alphabetically: {hit}"


def test_dry_run_proposes_one_edge_per_stranded_entity_and_writes_nothing(tmp_path, capsys):
    st = build_store(tmp_path)
    assert _edge_count(st, STRANDED) == 0, "fixture must start at degree zero"
    # The count a dry run must NOT move: the fixture's own target scaffolding, which
    # #1832 added so the bound has admissible targets to admit.
    scaffold = st.edges.count(active_only=True)
    assert scaffold == len(FIXTURE_EDGES), (
        f"the fixture's scaffold is what a dry run must leave alone: {scaffold}")
    assert linker.main(["--db", str(st.path), "--sample", "0"]) == 0
    out = capsys.readouterr().out

    pairs = [(p["source"], p["target"]) for p in linker.find_proposals(st)[0]]
    assert len(pairs) == len(set(pairs)) == 4, (
        f"exactly one edge per stranded entity, no duplicates: {pairs}")
    assert sorted(pairs) == sorted([
        (DUAL, TARGET),          # 16 facts at degree zero on the live store too
        (STRANDED, "TGS-RAG"),   # longest wins over the `RAG` nested inside it
        ("Obsidian Canvas", TARGET),   # the stand-in candidate, see below
        (ROLE_NOUN, BARE),       # bare/role noun: linked, not merged
    ]), pairs
    # The proposal this file used to expect from `TGS-RAG` is gone, and that is the
    # point of #1832: `TGS-RAG` embeds `RAG` and holds 5 facts, so before the target
    # bound it was itself a proposed source. The edge that makes it a target worth
    # linking to is the same edge that makes it linked — hence no longer a degree-zero
    # candidate — and `Obsidian Canvas` stands in as a candidate of the same shape so
    # this pass still lands four edges.
    for unnameable in (VICTIM, ALONE):
        assert unnameable not in [s for s, _ in pairs], unnameable
    assert "dry-run" in out, "the default mode must announce itself"
    assert st.edges.count(active_only=True) == scaffold, (
        "--apply was not passed; the only edges in the store may be the fixture's own")
    st.close()


def test_a_fact_label_with_no_entities_row_is_counted_and_not_linked(tmp_path):
    """`facts_idx` carries labels with no `entities` row. Writing an edge onto one
    would make the graph assert a node exists when it does not, so the mass is
    counted and skipped — the denominator is what keeps it visible."""
    st = build_store(tmp_path)
    st._query(
        "INSERT INTO facts_idx(entity, category, fact_id, text_hash, fact, confidence, "
        "created_at, provenance, file_path) VALUES (?,?,?,?,?,?,?,?,?)",
        ("TGS-RAG Implementation", "state", "fact-900", "h-orphan-900",
         "an orphan label with no entity row", 0.9,
         "2026-09-18T00:00:00+00:00", "EXTRACTED",
         "TGS-RAG Implementation/TGS-RAG Implementation-state.md"))
    pairs, dens = linker.find_proposals(st)
    assert "TGS-RAG Implementation" not in [p["source"] for p in pairs]
    assert dens["skipped_no_entity_row"] == 1, dens
    assert dens["embed_a_name"] == (dens["proposed"] + dens["skipped_no_entity_row"]
                                    + dens["refused_generic_target"]
                                    + dens["refused_dead_target"]), (
        "the slices must add up against the denominator they are drawn from")
    st.close()


# ── clause 2: the item's own degree-zero probe, before and after ─────────────

_FACTS_SQL = ("SELECT entity, COUNT(*) c FROM facts_idx "
              "WHERE expired_at IS NULL OR expired_at='' GROUP BY entity")
_EMBED_FMT = "(?<![0-9a-z]){n}(?![0-9a-z])"


def degree_zero_probe(st) -> list[str]:
    """#1019's check, re-read from the store rather than simulated.

    The issue's check enumerates degree-zero entities holding ≥10 active facts;
    the acceptance adds the third condition — its own name embedding another
    registered name — which is the rule this linker exists to close. Raw SQL over
    `facts_idx` and `edges` plus the item's own look-around, written without
    calling the linker's matcher so this check can still disagree with it.
    """
    names = {r["name"].strip().lower() for r in st._query("SELECT name FROM entities")}
    linked = set()
    for r in st._query("SELECT source, target FROM edges WHERE expired_at IS NULL"):
        linked.add(r["source"])
        linked.add(r["target"])
    hits = []
    for r in st._query(_FACTS_SQL):
        label = r["entity"]
        if r["c"] < 10 or label in linked:
            continue
        own = label.strip().lower()
        if any(reg != own and len(reg) >= 3
               and re.search(_EMBED_FMT.format(n=re.escape(reg)), own)
               for reg in names):
            hits.append(label)
    return hits


def test_the_probe_empties_only_after_apply(tmp_path, capsys):
    st = build_store(tmp_path)
    before = sorted(degree_zero_probe(st))
    assert before == sorted([STRANDED, DUAL, "Obsidian Canvas"]), (
        "the fixture must reproduce #1019's premise before the fix: at degree zero, "
        "≥10 active facts, own name embedding another registered name. `CameraCfg` "
        "and `UniTacHand` are stranded with 11 facts each and neither is named, "
        "which is the set the probe's own rule defines. `Obsidian Canvas` is the "
        "third hit #1832 adds: it stands in for the proposal `TGS-RAG` loses when "
        "the edge that makes it an admissible target also makes it linked, so it is "
        f"a candidate in its own right: {before}")

    assert linker.main(["--db", str(st.path), "--apply", "--sample", "0"]) == 0
    capsys.readouterr()

    assert degree_zero_probe(st) == [], "the probe must come back empty after --apply"
    assert _edge_count(st, STRANDED) == 1, "the closed entity carries exactly one active edge"
    assert _edge_count(st, DUAL) == 1
    assert _edge_count(st, VICTIM) == 0, "the rule reaches no edge for `CameraCfg`"
    assert _edge_count(st, ALONE) == 0, "an entity embedding no registered name gets none"
    st.close()


def test_rerunning_is_idempotent(tmp_path, capsys):
    st = build_store(tmp_path)
    linker.main(["--db", str(st.path), "--apply", "--sample", "0"])
    capsys.readouterr()
    first = st.edges.count(active_only=True)
    assert linker.main(["--db", str(st.path), "--apply", "--sample", "0"]) == 0
    second = capsys.readouterr().out
    assert _shown_number(second, "proposed edges") == 0, second
    assert st.edges.count(active_only=True) == first, "a second pass must write nothing"
    assert linker.main(["--db", str(st.path), "--apply", "--sample", "0"]) == 0
    assert st.edges.count(active_only=True) == first
    st.close()


# ── clause 3: a rebuild-durable stamp, and the evidence that names the substring ─

def test_every_written_edge_carries_a_carried_origin_and_its_evidence(tmp_path, capsys):
    st = build_store(tmp_path)
    proposed, _ = linker.find_proposals(st)   # read before anything is written
    assert len(proposed) == 4, "four stranded entities embed a registered name"
    assert {p["origin"] for p in proposed} == {"manual"}, (
        "a tool-written edge is not re-derivable, so it must carry an origin "
        "kg_rebuild carries across a re-derivation")
    assert all(p["evidence"].strip() for p in proposed), "an edge must say why it exists"
    stranded = next(p for p in proposed if p["source"] == STRANDED)
    assert "'tgs-rag'" in stranded["evidence"], "the evidence must name the substring"
    assert STRANDED in stranded["evidence"]

    linker.main(["--db", str(st.path), "--apply", "--sample", "0"])
    capsys.readouterr()
    row = st.edges.find_active(STRANDED, "TGS-RAG", linker.EDGE_TYPE)
    assert row is not None, "the proposed edge is live after --apply"
    assert row["origin"] == "manual" and row["evidence"], row
    assert row["provenance"] == "INFERRED", row["provenance"]
    assert row["provenance"] in _rebuild_carry_provenance(), (
        "the edge is not reproducible from prose, so it must carry a provenance the "
        "rebuild's carry-over filter reads")
    assert "manual" in CARRY_EDGE_ORIGINS, (
        "kg_rebuild.py's carry-over filter reads kg_store.CARRY_EDGE_ORIGINS; if the "
        "origin written here left that tuple, a re-derivation would silently drop the "
        "edge and the entity would be stranded again")
    assert linker.EDGE_ORIGIN == "manual"
    st.close()


def _rebuild_module():
    """`kg_rebuild.py` as a module, so a test can read the values ITS carry-over
    filter reads rather than the ones the writer imported."""
    spec = importlib.util.spec_from_file_location(
        "kg_rebuild_for_link_test", ROOT / "scripts" / "memory" / "kg_rebuild.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rebuild_carry_provenance():
    return _rebuild_module().CARRY_PROVENANCE


def test_the_rebuild_filter_reads_the_origin_this_tool_writes(tmp_path):
    """Clause 3's durability half, graded against the value the filter itself reads.

    The edge row and the `kg_store.CARRY_EDGE_ORIGINS` membership check only prove
    durability while the export filter happens to read that shared constant. This
    loads `kg_rebuild.py` and asks its own name, so the two failure modes the
    claim depends on are caught here rather than at the next rebuild: a filter
    narrowed back to a private literal — `("fact_relate",)` — which would export
    every `fact_relate` edge and silently drop this tool's, and a writer whose
    origin left the carried tuple. Either one strands the entity again with no
    test failing, which is the failure this round exists to prevent.
    """
    rebuild = _rebuild_module()
    assert rebuild.CARRY_EDGE_ORIGINS == CARRY_EDGE_ORIGINS, (
        "kg_rebuild.py must read kg_store.CARRY_EDGE_ORIGINS, not restate the tuple: "
        f"{rebuild.CARRY_EDGE_ORIGINS!r} vs {CARRY_EDGE_ORIGINS!r}")
    assert linker.EDGE_ORIGIN in rebuild.CARRY_EDGE_ORIGINS, (
        "the origin this tool stamps is not in the set the export filter keeps")
    assert linker.EDGE_PROVENANCE in rebuild.CARRY_PROVENANCE, (
        "belt and braces: the filter is an OR over provenance and origin, and the "
        "provenance this tool stamps is carried in its own right")


def test_the_carry_filter_keeps_a_manual_edge_after_a_re_derivation(tmp_path):
    """The clause is about surviving a rebuild, so run the export filter itself."""
    carry = _rebuild_carry_provenance()
    st = build_store(tmp_path)
    linker.main(["--db", str(st.path), "--apply", "--sample", "0"])
    kept = [e for e in st.edges.all(include_expired=False)
            if (e.get("provenance") or "") in carry
            or (e.get("origin") or "") in CARRY_EDGE_ORIGINS]
    assert len(kept) == st.edges.count(active_only=True) == 4 + len(FIXTURE_EDGES), (
        f"the store holds the 4 linker-written edges plus the "
        f"{len(FIXTURE_EDGES)} fixture edges the #1618 bound needs, and every one "
        f"origin is carried across a re-derivation; kept {len(kept)}")
    st.close()


def test_apply_writes_the_edge_with_a_valid_type_and_id(tmp_path, capsys):
    """The process boundary: the tool's payload becomes a real `edges` row the
    store's own readers accept, with an id in the shape every other writer uses."""
    st = build_store(tmp_path)
    linker.main(["--db", str(st.path), "--apply", "--sample", "0"])
    capsys.readouterr()
    row = st.edges.find_active(STRANDED, "TGS-RAG", linker.EDGE_TYPE)
    assert row is not None
    assert {e["type"] for e in st.edges.all()} == {linker.EDGE_TYPE}, (
        "every edge the tool writes is the one type the contract names")
    assert isinstance(row["id"], int) and row["id"] > 0, row["id"]
    assert st.edges.by_id(row["id"])["source"] == STRANDED, "the row is addressable by id"
    assert row["created_at"], "an edge with no timestamp cannot be aged out or audited"
    assert row["confidence"] == linker.EDGE_CONFIDENCE
    st.close()


# ── clause 4: denominators beside the count ─────────────────────────────────

def test_counts_print_beside_the_denominators_they_are_a_slice_of(tmp_path, capsys):
    st = build_store(tmp_path)
    assert linker.main(["--db", str(st.path), "--sample", "0"]) == 0
    out = capsys.readouterr().out

    index = linker.build_target_index(st.entities.all())
    facts = linker.active_fact_counts(st)
    linked = st.edges.degree()
    fact_holding = len(facts)
    degree_zero = sum(1 for k, v in facts.items() if linked.get(k, 0) == 0)
    # An embedding CANDIDATE is a degree-zero fact holder, and the `FIXTURE_EDGES`
    # scaffolding the #1618 bound needs lifts three fixture names above zero:
    # `TGS-RAG`, `Obsidian` and `Browser` are targets now, so no matter how well their
    # names embed somebody else's they are not candidates on their own night.
    embed = sum(1 for k in facts
                if linked.get(k, 0) == 0 and linker.longest_embedded_name(k, index))

    assert _shown_number(out, "fact-holding entities") == fact_holding
    assert _shown_number(out, "degree-zero candidates") == degree_zero
    assert _shown_number(out, "of-which embed a name") == embed
    _proposed, dens = linker.find_proposals(st)
    slice_rows = ("proposed", "refused_generic_target", "refused_dead_target",
                  "skipped_no_entity_row", "skipped_existing_edge")
    assert sum(dens[k] for k in slice_rows) == dens["embed_a_name"], (
        f"the report's rows must partition `embed_a_name`, and they sum to "
        f"{sum(dens[k] for k in slice_rows)} of {dens['embed_a_name']}: {dens}")
    assert _shown_number(out, "proposed edges") == dens["proposed"] == embed, (
        "on this fixture every candidate's target clears the bound, so nothing is "
        "refused and `proposed` still equals `embed_a_name` — a store where one "
        "candidate does NOT clear it is "
        "`test_a_target_refused_by_the_1618_bound_prints_its_own_denominators`, which "
        "pins `proposed` strictly below `embed_a_name`. Every name here embeds another "
        "boundary-cleanly, `Browser Tool` included, and none of it is scoped to the "
        "probe's 10-fact floor")
    # The count is not printed alone: each slice row names the set it is drawn from.
    assert f"(of {fact_holding} fact-holding)" in out, out
    assert f"(of {degree_zero} degree-zero candidates)" in out, out
    assert f"(of {embed} embedding a name)" in out, out
    assert "--apply" in out, "dry-run must name the flag that would change that"
    st.close()


def test_the_floor_moves_the_candidate_slice_not_the_corpus(tmp_path, capsys):
    """`--min-facts` exists for a staged first run, and the report must show it bit.

    An operator raising the floor to link the biggest rows first needs to see the
    candidate slice shrink while the corpus count stands still; a report that
    printed only the proposal count could not tell a raised floor from a clean
    graph — the confusion this clause exists to prevent.
    """
    st = build_store(tmp_path)
    capsys.readouterr()
    linker.main(["--db", str(st.path), "--min-facts", "13", "--sample", "0"])
    floored = capsys.readouterr().out
    # Fixture counts are 16/12/11/11/11/5/5/4/4/3/3, so a floor of 13 leaves
    # `Obsidian Dataview` alone as a candidate, and one proposal with it.
    assert _shown_number(floored, "fact-holding entities") == len(FIXTURE_FACTS), (
        "the corpus denominator does not move with the floor")
    assert _shown_number(floored, "degree-zero candidates") == 1, floored
    assert _shown_number(floored, "of-which embed a name") == 1, floored
    assert _shown_number(floored, "proposed edges") == 1, floored
    st.close()


def test_the_denominators_survive_the_linker_being_pointed_at_an_empty_store(tmp_path, capsys):
    """0 proposals is also the output of a store that failed to load, so the report
    says which of the two it is: no candidate, or candidates embedding nothing."""
    KGStore(tmp_path / "nothing.sqlite").close()
    assert linker.main(["--db", str(tmp_path / "nothing.sqlite"), "--sample", "0"]) == 0
    out = capsys.readouterr().out
    for label, _key, _of, _ofl in linker.report_lines({}):
        assert f"{label} " in out, f"{label!r} is in the table but not the output:\n{out}"
    for label in ("fact-holding entities", "degree-zero candidates",
                  "of-which embed a name", "proposed edges"):
        assert _shown_number(out, label) == 0, out
    assert "nothing stranded" not in out, (
        "the tool prints counts, not a conclusion a reader could misread")


# ── #1833: `--apply` is gated on knowledge_graph.write_enabled ───────────────
#
# Every test here drives the real `linker.writes_disabled_by_rebuild`. Patching
# that predicate out — what `test_numeric_entity_sweep.py:373` does for the
# markdown sweep — would leave the guard itself, and the key name it reads,
# unpinned, so the config is replaced instead: a temp dict in most tests, a temp
# `config.yaml` read back through the real loader in the last one.

SEEDED_A = "Alpha Witness"   # a live edge from ANOTHER writer, see below
SEEDED_B = "Beta Witness"


def _write_flag(monkeypatch, enabled: bool) -> None:
    """Install the config the guard reads, with the flag set to `enabled`.

    `app.config.CONFIG` is the module-level dict `_load_config()` assigns at
    import and that every write gate on this machine reads
    (`agent_mcp/facts.py::_writes_enabled`, the #1573 sweep, now this guard), so
    replacing it is the temp config: the predicate runs for real against a value
    this test owns, and no file under the live tree is touched."""
    import app.config
    monkeypatch.setattr(app.config, "CONFIG",
                        {"knowledge_graph": {"write_enabled": enabled}})


def _seed_one_live_edge(st) -> int:
    """One active edge written by a different origin, and the count it leaves.

    Clause 2's `nothing was written` check needs a non-zero number to hold still:
    on the bare fixture store both the pre-call and the post-call count are 0, and
    `0 == 0` passes a run that wrote four edges. The pair is registered but holds
    no facts, so it is neither a candidate nor an embedding target for anything in
    the fixture — the tests below re-assert the proposal count at 4, which is what
    makes that non-interference a measurement rather than an assumption."""
    st.entities.register(SEEDED_A)
    st.entities.register(SEEDED_B)
    st.edges.add({"source": SEEDED_A, "target": SEEDED_B, "type": "related_to",
                  "evidence": "written by another writer, before the freeze"},
                 origin="fact_relate")
    return st.edges.count(active_only=True)


def _backup_traces(root) -> list[str]:
    """Names of every file the apply path's backup route leaves under `root`.

    `st.backup()` writes `store-backups/kg-stranded-<ts>.sqlite` beside the store,
    so a refusal that landed after it would show up here even with the edge count
    unchanged — which is the half of clause 2 the count cannot see."""
    return sorted(p.name for p in Path(root).rglob("kg-stranded-*"))


def test_apply_refuses_while_the_rebuild_write_flag_is_false(tmp_path, capsys, monkeypatch):
    """Clause 1, with clause 2's silence: non-zero exit, a reason naming the flag
    and the rebuild, after the denominators have printed, and before the backup.

    The denominators come first on purpose. A person holding the pilot in a freeze
    window still has to see how much the gate is holding back — refusing before the
    measurement would make the guard the reason the tool reports nothing at all."""
    st = build_store(tmp_path)
    seeded = _seed_one_live_edge(st)
    assert seeded == len(FIXTURE_EDGES) + 1, (
        f"the freeze check needs exactly one foreign edge on top of the "
        f"{len(FIXTURE_EDGES)} the fixture scaffolds, to be held still")
    _write_flag(monkeypatch, False)
    capsys.readouterr()

    rc = linker.main(["--apply", "--db", str(st.path), "--sample", "0"])
    out = capsys.readouterr().out

    assert rc == 2, f"a frozen --apply must exit non-zero; got {rc}\n{out}"
    assert "write_enabled = false" in out, f"the reason must name the flag:\n{out}"
    assert "rebuild" in out, f"the reason must name who sets the flag:\n{out}"
    assert _shown_number(out, "proposed edges") == 4, (
        f"the denominator block must still print above the refusal:\n{out}")
    assert st.edges.count(active_only=True) == seeded, (
        "the refused --apply wrote an edge into a store under rebuild")
    assert _backup_traces(tmp_path) == [], (
        "the refusal came after st.backup(): a snapshot of a store about to be "
        "replaced is not a restore point")
    st.close()


def test_the_gate_releases_and_apply_writes_when_the_flag_is_true(tmp_path, capsys,
                                                                  monkeypatch):
    """Clause 3: with writes enabled the same call returns 0, writes its four
    proposed edges onto the same store, and still takes the backup the refusal
    skipped — the gate is a freeze check, not a new default for the tool."""
    st = build_store(tmp_path)
    seeded = _seed_one_live_edge(st)
    _write_flag(monkeypatch, True)
    capsys.readouterr()

    rc = linker.main(["--apply", "--db", str(st.path), "--sample", "0"])
    out = capsys.readouterr().out

    assert rc == 0, f"the gate never released: rc={rc}\n{out}"
    assert "REFUSED" not in out, f"the gate is permanently disabling the tool:\n{out}"
    assert "applied 4 edges" in out, out
    assert st.edges.count(active_only=True) == seeded + 4, (
        "the four proposed edges are live beside the seeded one")
    assert st.edges.find_active(STRANDED, "TGS-RAG", linker.EDGE_TYPE) is not None, (
        "the row #1019 is named for is the one that must be linked by a released apply")
    assert len(_backup_traces(tmp_path)) == 1, (
        "a released apply takes the backup clause 2 says a refusal skips")
    st.close()


def test_the_dry_run_stays_available_under_a_freeze(tmp_path, capsys, monkeypatch):
    """Clause 4: the flag being false must not blind anyone. Without `--apply` the
    run still exits 0 and prints its denominators, so the freeze window can be
    inspected while it is closed — which is exactly when the numbers are wanted."""
    st = build_store(tmp_path)
    seeded = _seed_one_live_edge(st)
    _write_flag(monkeypatch, False)
    capsys.readouterr()

    rc = linker.main(["--db", str(st.path), "--sample", "0"])
    out = capsys.readouterr().out

    assert rc == 0, f"a dry run under a freeze must still run: rc={rc}\n{out}"
    assert "dry-run" in out, out
    assert "REFUSED" not in out, f"the refusal is for --apply only:\n{out}"
    assert _shown_number(out, "proposed edges") == 4, out
    assert st.edges.count(active_only=True) == seeded, "a dry run wrote"
    assert _backup_traces(tmp_path) == [], "a dry run backed the store up"
    st.close()


def test_a_frozen_apply_with_nothing_to_propose_still_refuses(tmp_path, capsys,
                                                              monkeypatch):
    """Clause 5: the refusal is decided outside the `if proposed:` branch.

    Before the gate, an `--apply` that proposed nothing printed
    `applied 0 edges (nothing proposed)` and returned 0. That is the print a run
    whose freeze opened mid-pass leaves behind, and it reads exactly like a graph
    with nothing stranded — so the gate has to fire on the empty case too. The
    second half pins that what is refusing is the flag, not the empty store: the
    same store with writes enabled reaches the apply branch and prints the line."""
    empty = tmp_path / "nothing.sqlite"
    KGStore(empty).close()
    _write_flag(monkeypatch, False)
    capsys.readouterr()

    rc = linker.main(["--apply", "--db", str(empty), "--sample", "0"])
    out = capsys.readouterr().out

    assert rc == 2, f"a frozen --apply with zero proposals must refuse too: rc={rc}\n{out}"
    assert "applied 0 edges" not in out, (
        f"the old success print survived on the empty branch:\n{out}")
    assert "REFUSED --apply" in out, out
    assert _backup_traces(tmp_path) == [], out

    _write_flag(monkeypatch, True)
    capsys.readouterr()
    assert linker.main(["--apply", "--db", str(empty), "--sample", "0"]) == 0
    assert "applied 0 edges" in capsys.readouterr().out, (
        "an unfrozen empty store must still report 'nothing proposed', not a refusal")


def test_the_guard_reads_write_enabled_out_of_a_config_file(tmp_path, monkeypatch):
    """The seam the guard sits on: `kg_rebuild.py:145` writes the flag into
    `config.yaml` from its own process, and a linker run started afterwards in a
    different one has to see it. A test that only ever handed the guard a dict
    would still pass if the guard read `write-enable:` or the wrong section,
    because the dict was built from the same wrong guess; here the bytes come from
    a temp file and the reader is the real `app.config._load_config`.
    """
    import app.config

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.delenv("LLOYD_CONFIG_OVERLAY", raising=False)
    monkeypatch.setattr(app.config, "LLOYD_HOME", repo)
    monkeypatch.setattr(app.config, "TOOL_OVERRIDES_PATH", repo / "tool_overrides.yaml")

    def load(text: str) -> None:
        (repo / "config.yaml").write_text(text, encoding="utf-8")
        monkeypatch.setattr(app.config, "CONFIG", app.config._load_config())

    load("knowledge_graph:\n  write_enabled: false\n  fact_write_gate: shadow\n")
    assert linker.writes_disabled_by_rebuild() is True, (
        "the guard does not read `knowledge_graph.write_enabled` out of the file "
        "the rebuild writes — it is guarding nothing")

    load("knowledge_graph:\n  write_enabled: true\n")
    assert linker.writes_disabled_by_rebuild() is False, (
        "the guard does not release when the rebuild restores the flag")

    load("models: {}\n")
    assert linker.writes_disabled_by_rebuild() is False, (
        "a config with no knowledge_graph section is not a rebuild: default is enabled")

    # Failing open is the precedent's behaviour and the only safe direction for a
    # dry-run-by-default tool: an unreadable config must not disable the write path.
    monkeypatch.setattr(app.config, "CONFIG",
                        {"knowledge_graph": "not a mapping"})
    assert linker.writes_disabled_by_rebuild() is False, (
        "a malformed config must fail open, not freeze the tool forever")


# ── #1832 clause 1+2: the #1618 target-side bound ─────────────────────────────
#
# Every store below is built by `build_bound_store`, which takes the fact counts,
# the edges and the extra registered names explicitly, because the bound is a
# statement about three measured properties of a TARGET — fan-out, active degree,
# active facts — and a fixture that shares `FIXTURE_FACTS` cannot hold one value
# per property without the others moving underneath it.

def build_bound_store(tmp_path, *, facts, edges=(), extra_names=(),
                      name="kg.sqlite") -> KGStore:
    """A store with exactly the fan-out / degree / facts a bound case needs.

    `facts` is label → active fact count (registered and indexed); `edges` is
    (source, target) pairs written with the `manual` origin so none of them is
    ever confused with a linker write; `extra_names` registers a name with no
    facts at all, which is how a fan-out partner is built — it must be a
    registered name that embeds the target, and must not itself become a
    candidate, which a fact-holding name would.
    """
    st = KGStore(tmp_path / name)
    for label in (*facts, *extra_names):
        st.entities.register(label)
    for label, n in facts.items():
        for i in range(n):
            st._query(
                "INSERT INTO facts_idx(entity, category, fact_id, text_hash, fact, "
                "confidence, created_at, provenance, file_path) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (label, "event", f"fact-{label}-{i}", f"h-{label}-{i}",
                 f"{label} claim {i}", 0.9, "2026-09-18T00:00:00+00:00",
                 "EXTRACTED", f"{label}/{label}-{i}.md"))
    for src, dst in edges:
        st.edges.add({"source": src, "target": dst, "type": "related_to",
                      "evidence": "fixture scaffolding for #1832"}, origin="manual")
    return st


def _proposed_pairs(st):
    proposals, dens = linker.find_proposals(st, min_facts=1)
    return [(p["source"], p["target"]) for p in proposals], dens


def test_a_dead_or_generic_target_is_refused_and_only_the_good_one_is_proposed(tmp_path):
    """Clause 1: degree 0 or fan-out above `TARGET_FAN_OUT_MAX` refuses the proposal.

    Each of the three targets is built so exactly ONE property distinguishes it,
    because a fixture where the refused target fails all three would pass even if
    the bound only implemented one of them:

      * `Gadget` — fan-out 21 (twenty registered partners plus the candidate),
        8 facts, ONE active edge. It clears the facts bar and is reachable, so
        only the fan-out ceiling can refuse it, and it must be refused as
        GENERIC, not dead.
      * `Cache` — fan-out 1 and 8 active facts, clearing both numeric bars, and
        NO active edge. This is the live `automod` shape (23 facts, fan-out 13,
        zero edges): the degree floor is what refuses it, and it must be refused
        as DEAD, not generic.
      * `Widget Factory` — fan-out 1, 6 facts, one active edge: the same name a
        candidate embeds, reachable and worth linking to, so it is proposed.
    """
    st = build_bound_store(
        tmp_path,
        facts={
            "My Gadget Notebook": 12,      # candidate -> Gadget
            "Session Cache Design": 12,    # candidate -> Cache
            "My Widget Factory Note": 12,  # candidate -> Widget Factory
            "Gadget": 8, "Cache": 8, "Widget Factory": 6,
        },
        edges=[("Registry Node", "Gadget"), ("Registry Node", "Widget Factory")],
        extra_names=[f"Gadget Alpha {i:02d}" for i in range(20)] + ["Registry Node"])

    fan_out = {t: linker.target_fan_out(t, st.entities.all())
               for t in ("Gadget", "Cache", "Widget Factory")}
    assert fan_out == {"Gadget": 21, "Cache": 1, "Widget Factory": 1}, (
        f"the fan-out gadget is what the case is named for: {fan_out}")

    pairs, dens = _proposed_pairs(st)
    assert pairs == [("My Widget Factory Note", "Widget Factory")], (
        f"only the good target may be proposed: {pairs}")
    assert dens["embed_a_name"] == 3, f"three candidates embed a name: {dens}"
    assert dens["refused_generic_target"] == 1, (
        f"`Gadget` is generic by fan-out alone, having an edge and 8 facts: {dens}")
    assert dens["refused_dead_target"] == 1, (
        f"`Cache` is dead by degree alone, clearing both numeric bars: {dens}")
    assert dens["proposed"] == 1, dens
    st.close()


def test_a_low_fan_out_target_is_admitted_below_the_facts_floor_and_a_guard_shape_is_not(
        tmp_path):
    """Clause 2: `facts >= 5 OR fan_out <= 2`, with the live triples reproduced.

    All three targets here already hold an active edge, so nothing below is the
    degree floor's doing — the disjunction is the only rule left running:

      * `Guard`  — fan-out 5, 3 facts, degree 1. Refused: five names wear the
        word and the store has three facts to say about it, which is the shape
        the live store's `Guard` has and the reason the disjunction exists at
        all (fan-out inside the ceiling would otherwise have admitted it).
      * `Scope Creep` — fan-out 1, 4 facts, degree 3. Admitted: under the facts
        floor, but only one other name embeds it, so the name distinguishes.
      * `Hook` — fan-out 11, 6 facts, degree 4. Admitted: inside the ceiling and
        written about enough, which is what stops the ceiling alone from being
        the whole rule.
    """
    st = build_bound_store(
        tmp_path,
        facts={
            "Session Guard Wrapper": 12,   # candidate -> Guard
            "Project Scope Creep Log": 12,  # candidate -> Scope Creep
            "Git Hook Runner": 12,          # candidate -> Hook
            "Guard": 3, "Scope Creep": 4, "Hook": 6,
        },
        edges=[("Registry Node", "Guard"),
               ("Backlog Board", "Scope Creep"), ("Triage Job", "Scope Creep"),
               ("Reviewer Note", "Scope Creep"),
               ("Registry Node", "Hook"), ("Webhook Config", "Hook"),
               ("Post Tool Use", "Hook"), ("Stop Hook", "Hook")],
        extra_names=["Guard Rail 1", "Guard Rail 2", "Guard Rail 3", "Guard Rail 4",
                     "Registry Node", "Backlog Board", "Triage Job", "Reviewer Note",
                     "Webhook Config", "Post Tool Use", "Stop Hook"]
                    + [f"Hook Line {i:02d}" for i in range(9)])

    fan_out = {t: linker.target_fan_out(t, st.entities.all())
               for t in ("Guard", "Scope Creep", "Hook")}
    assert fan_out == {"Guard": 5, "Scope Creep": 1, "Hook": 11}, (
        f"the three live triples, rebuilt: {fan_out}")
    degree = st.edges.degree()
    assert [degree.get(t, 0) for t in ("Guard", "Scope Creep", "Hook")] == [1, 3, 4], degree

    pairs, dens = _proposed_pairs(st)
    assert sorted(t for _, t in pairs) == ["Hook", "Scope Creep"], (
        f"the low-fan-out and well-documented targets are admitted, `Guard` is not: "
        f"{pairs}")
    assert dens["refused_generic_target"] == 1 and dens["refused_dead_target"] == 0, (
        f"`Guard` has an edge, so its refusal must be the generic class: {dens}")
    st.close()


def test_a_target_refused_by_the_1618_bound_prints_its_own_denominators(tmp_path, capsys):
    """Clause 3: one candidate per class, and the report partitions `embed a name`.

    Three candidates, three outcomes, and the arithmetic the item demands: the
    rows after `of-which embed a name` are slices of it, they sum to it, and
    `proposed` is strictly below it by the refused count. That last part is what
    stops a run from printing two reassuring new rows while proposing every
    candidate it always proposed — the pre-change store printed 18 proposed of 18
    embedding, and a version of this tool that counted refusals without acting on
    them would print exactly the same 18.
    """
    st = build_bound_store(
        tmp_path,
        facts={
            "My Gadget Notebook": 12,      # generic: fan-out 21
            "Session Cache Design": 12,    # dead: no edge at all
            "My Widget Factory Note": 12,  # good
            "Gadget": 8, "Cache": 8, "Widget Factory": 6,
        },
        edges=[("Registry Node", "Gadget"), ("Registry Node", "Widget Factory")],
        extra_names=[f"Gadget Alpha {i:02d}" for i in range(20)] + ["Registry Node"])

    assert linker.main(["--db", str(st.path), "--sample", "0", "--min-facts", "1"]) == 0
    out = capsys.readouterr().out

    embed = _shown_number(out, "of-which embed a name")
    proposed = _shown_number(out, "proposed edges")
    generic = _shown_number(out, "refused: generic target")
    dead = _shown_number(out, "refused: dead target")
    assert (embed, proposed, generic, dead) == (3, 1, 1, 1), out
    assert proposed < embed, (
        f"proposed must fall by the refused count; {proposed} of {embed} is the "
        f"pre-change reading, which is what this bound exists to end")
    assert embed - proposed == generic + dead == 2, (embed, proposed, generic, dead)

    _, dens = linker.find_proposals(st, min_facts=1)
    slices = {}
    for label, key, den_key, _ in linker.report_lines(dens):
        if den_key == "embed_a_name":
            slices[label] = dens[key]
    assert sorted(slices) == sorted([
        "proposed edges", "refused: generic target", "refused: dead target",
        "skipped: no entities row", "skipped: edge already live"]), slices
    assert sum(slices.values()) == dens["embed_a_name"], (
        f"the rows after `of-which embed a name` must partition it, and they sum to "
        f"{sum(slices.values())} of {dens['embed_a_name']}: {slices}")

    # The two new rows come out of `report_lines`, which is what makes them print
    # on a store with nothing to refuse as well: `report_lines` builds them from the
    # table, not from the proposals it found.
    empty = build_bound_store(tmp_path, facts={}, name="empty.sqlite")
    capsys.readouterr()
    assert linker.main(["--db", str(empty.path), "--sample", "0"]) == 0
    empty_out = capsys.readouterr().out
    for label in ("refused: generic target", "refused: dead target"):
        assert label in empty_out, f"{label} missing from an empty store's report:\n{empty_out}"
        assert _shown_number(empty_out, label) == 0, empty_out
    st.close()
    empty.close()


def test_the_module_docstring_says_the_bare_role_noun_shape_is_ruled(tmp_path):
    """Clause 4: the page a person reads must not claim the shape is unruled.

    The paragraph used to say "whether linking it is wanted is a scope call no
    test can settle, so it is NOT suppressed here". #1618 settled it on
    2026-09-29, so the sentence is a live falsehood on the surface an operator
    reads before deciding whether to run the tool, and it is pinned here off
    `__doc__` — the loaded module's own text, not a copy of it in this file.
    """
    doc = linker.__doc__
    flat = " ".join(doc.split())

    for stale in ("NOT suppressed", "unruled", "no test can settle"):
        assert stale not in flat, (
            f"the module docstring still tells a reader {stale!r}, which the #1618 "
            f"ruling of 2026-09-29 contradicts")

    assert "1618" in flat, "the docstring must name the ruling that decided it"
    assert "TARGET_FAN_OUT_MAX" in flat and "TARGET_DEGREE_MIN" in flat \
        and "TARGET_FACTS_MIN" in flat and "TARGET_FAN_OUT_TRUSTED" in flat, (
        "the bound must be stated by the constant names, so a renumbered threshold "
        "cannot leave the prose behind")
    assert f"`TARGET_FAN_OUT_MAX` = {linker.TARGET_FAN_OUT_MAX}" in flat, flat[:600]
    assert f"`TARGET_DEGREE_MIN` = {linker.TARGET_DEGREE_MIN}" in flat
    assert f"`TARGET_FACTS_MIN` = {linker.TARGET_FACTS_MIN}" in flat
    assert f"`TARGET_FAN_OUT_TRUSTED` = {linker.TARGET_FAN_OUT_TRUSTED}" in flat
    # The shape is admitted and the CLASS refused — the distinction the ruling made,
    # and the reason there is no word list in this module.
    assert "never by a list of names" in flat, flat[:600]
    # And the function's vocabulary is the report's vocabulary: what it returns is
    # used directly as the denominator key, so a rename on one side and not the
    # other would print a row of zeroes while refusing everything.
    assert linker.target_admissibility(fan_out=1, degree=0, facts=50) \
        == "refused_dead_target"
    assert linker.target_admissibility(fan_out=99, degree=9, facts=50) \
        == "refused_generic_target"
    assert linker.target_admissibility(fan_out=1, degree=1, facts=1) == "admitted"


# ── #2246: the backup's filename is an instant, not a host-local guess ───────
#
# `--apply` snapshots the store to `store-backups/kg-stranded-<stamp>.sqlite`
# before it writes, and until #2246 that stamp came from `datetime.now()` — the
# host's wall clock — with a literal `Z` glued on. This host is
# `America/Los_Angeles`, so every name was seven hours early while claiming UTC.
# The damage is attribution, not cosmetics: the 14:22:26Z apply of 2026-10-05
# produced `kg-stranded-20261005T072226Z.sqlite`, which sits 41 s behind a
# genuinely-UTC nightly run that started at 07:21:45Z and logged verbatim that it
# took no backup, so a reader pairing artefacts by name frames that run for a
# live-store write it did not perform. The same file already formatted an instant
# correctly one clause up (`created_at` is
# `datetime.now(timezone.utc).isoformat()`), which is what made line 479 the
# inconsistency rather than a convention.
#
# What stays out of this section deliberately: ten further naive-`Z` sites in
# other files, enumerated on #2246, and the mislabelled artefact itself, which
# is not renamed because its name disagreeing with its mtime by exactly the host
# offset is the surviving evidence that the bug was real.

_BACKUP_NAME = re.compile(r"kg-stranded-(\d{8}T\d{6}Z)\.sqlite\Z")
STAMP_LAG_BOUND_SECONDS = 5        # the bound clause 1 names


@pytest.fixture
def los_angeles():
    """Run the body with the process timezone forced to a non-UTC zone.

    Forcing the zone is not a nicety: on a box already living in UTC, the naive
    `datetime.now()` and the correct `datetime.now(timezone.utc)` print the same
    digits, so a test that never moves the clock setting passes while the defect
    is live. `time.tzset()` is what makes the change take effect in-process.

    The zone is restored and `tzset()` called again on the way out, so no later
    node in this file inherits a Los Angeles clock.
    """
    previous = os.environ.get("TZ")
    os.environ["TZ"] = "America/Los_Angeles"
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()


def _utc_stamp_claims(name: str) -> datetime:
    """The instant a `kg-stranded-*.sqlite` name asserts, as an aware datetime.

    The `Z` is read as UTC because that is how every later reader reads it, and
    pairing artefacts by name is the whole reason the file is named.
    """
    match = _BACKUP_NAME.fullmatch(name)
    assert match, f"{name!r} is not a kg-stranded-<YYYYMMDDTHHMMSSZ>.sqlite name"
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc)


def test_the_apply_backup_is_named_from_the_utc_clock_under_a_forced_zone(
        tmp_path, capsys, monkeypatch, los_angeles):
    """Clause 1: the backup's name is a UTC instant even when the host's clock is
    not, and the name and the file's own mtime tell the same second.

    The bound is 5 s. The lag that motivated it was 25,200 s — 5,040 times the
    bound — so the node cannot be satisfied by a rounding allowance, and it
    cannot pass vacuously: the zone fixture is what makes the naive form differ
    from the correct one at all.
    """
    st = build_store(tmp_path)
    _write_flag(monkeypatch, True)
    capsys.readouterr()

    before = datetime.now(timezone.utc)
    rc = linker.main(["--apply", "--db", str(st.path), "--sample", "0"])
    after = datetime.now(timezone.utc)
    capsys.readouterr()

    assert rc == 0, (
        f"the write gate did not release, so no backup was ever named: rc={rc}")
    names = _backup_traces(tmp_path)
    assert len(names) == 1, f"a released apply takes exactly one backup: {names}"

    stamp = _utc_stamp_claims(names[0])
    lag = max((before - stamp).total_seconds(),
              (stamp - after).total_seconds(), 0.0)
    assert lag <= STAMP_LAG_BOUND_SECONDS, (
        f"{names[0]} asserts {stamp.isoformat()} but the apply ran between "
        f"{before.isoformat()} and {after.isoformat()} — off by {lag:.1f} s against "
        f"a {STAMP_LAG_BOUND_SECONDS} s bound; a naive host-local stamp put this "
        "name seven hours behind the run that made it")

    backup = tmp_path / "store-backups" / names[0]
    name_vs_mtime = abs(stamp.timestamp() - backup.stat().st_mtime)
    assert name_vs_mtime <= STAMP_LAG_BOUND_SECONDS, (
        f"the name and the artefact's own mtime disagree by {name_vs_mtime:.1f} s, "
        "which is the pairing error #2246 is about: the file is filed under an "
        "instant some other run was awake")

    # The same call one clause earlier, with the zone fixture in place, is what
    # `datetime.now()` would have written; the name on disk must sit exactly one
    # host offset away from it, not on top of it. Stated as the host's own offset
    # rather than a hard-coded 7 h so the node stays true across a DST change.
    host_offset = datetime.now().astimezone().utcoffset()
    naive_name = time.strftime("%Y%m%dT%H%M%SZ")
    naive_claims = datetime.strptime(naive_name, "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc)
    apart = (stamp - naive_claims).total_seconds()
    assert apart == pytest.approx(-host_offset.total_seconds(),
                                  abs=STAMP_LAG_BOUND_SECONDS), (
        f"{names[0]} is {apart:.0f} s from the host-local stamp {naive_name} while "
        f"this host sits {host_offset} from UTC: the name is still the local clock "
        "wearing a Z")
    assert host_offset.total_seconds() != 0.0, (
        "the fixture did not move the zone, so nothing above could fail")


def test_a_real_process_under_the_host_zone_names_the_backup_in_utc(tmp_path):
    """The seam #2246's review named: `--apply` as a SEPARATE process under the
    host's non-UTC zone, which is the only environment where the produced name has
    to be right — autonomy task #24 runs this script as a subprocess, not as an
    import in a patched process.

    The zone is set in the child's own environment, so it is applied by CPython at
    startup rather than by `tzset()` in the test's process. The child reports its
    own naive and UTC stamps before applying, which is what makes this node
    non-vacuous from inside the child: if the two agree, the zone did not take and
    nothing below could fail. `app.config.CONFIG` is replaced in the child's
    module object, so the write gate is read through the same code path a real run
    reads it — `writes_disabled_by_rebuild` does its `from app.config import
    CONFIG` at call time, which is the reason a subprocess can pin this at all.
    """
    st = build_store(tmp_path)
    child = (
        "import json, runpy, sys\n"
        "from datetime import datetime, timezone\n"
        "import app.config\n"
        "app.config.CONFIG = {'knowledge_graph': {'write_enabled': True}}\n"
        "sys.argv = ['link_stranded_entities.py', '--apply', '--db', %r,\n"
        "            '--sample', '0']\n"
        "probe = {'naive': datetime.now().strftime('%%Y%%m%%dT%%H%%M%%SZ'),\n"
        "         'utc': datetime.now(timezone.utc).strftime('%%Y%%m%%dT%%H%%M%%SZ')}\n"
        "sys.stderr.write('PROBE ' + json.dumps(probe) + '\\n')\n"
        "runpy.run_path(%r, run_name='__main__')\n"
    ) % (str(st.path), str(SCRIPT))
    env = {**os.environ, "TZ": "America/Los_Angeles", "PYTHONPATH": str(ROOT)}
    before = datetime.now(timezone.utc)
    proc = subprocess.run([sys.executable, "-c", child], cwd=str(ROOT), env=env,
                          capture_output=True, text=True, timeout=180)
    after = datetime.now(timezone.utc)
    assert proc.returncode == 0, (
        f"the child apply failed: rc={proc.returncode}\n{proc.stderr[-1500:]}")
    probe = json.loads([ln for ln in proc.stderr.splitlines()
                        if ln.startswith("PROBE ")][0][6:])

    def probe_claims(which: str) -> datetime:
        """A probe stamp as the instant it asserts, in the same shape as a name."""
        return _utc_stamp_claims(f"kg-stranded-{probe[which]}.sqlite")

    # How far this host's wall clock lags UTC: on a `-0700` box the local stamp,
    # read as though it were UTC, sits 25,200 s in the past. Measured from the
    # child rather than hard-coded, so the node stays true across a DST change.
    local_lag = (probe_claims("utc") - probe_claims("naive")).total_seconds()
    assert local_lag >= 3600, (
        f"the child's naive and UTC stamps agree ({probe}), so its zone was not the "
        "host's and this node could not have failed for the wrong reason")

    names = _backup_traces(tmp_path)
    assert len(names) == 1, f"a real apply takes exactly one backup: {names}"
    stamp = _utc_stamp_claims(names[0])
    lag = max((before - stamp).total_seconds(),
              (stamp - after).total_seconds(), 0.0)
    assert lag <= STAMP_LAG_BOUND_SECONDS, (
        f"run as its own process the apply named its snapshot {names[0]}, which is "
        f"{lag:.1f} s from the real UTC window "
        f"({before.isoformat()}..{after.isoformat()})")

    from_the_local_one = (stamp - probe_claims("naive")).total_seconds()
    assert from_the_local_one == pytest.approx(
        local_lag, abs=STAMP_LAG_BOUND_SECONDS), (
        f"{names[0]} is {from_the_local_one:.0f} s from the child's own host-local "
        f"stamp {probe['naive']}, while that stamp sits {local_lag:.0f} s behind UTC: "
        "the filename is the local clock wearing a Z")
# ── clause 2: the naive form cannot come back to this file ───────────────────
#
# The shape of this scan is the one #1128 left in
# `tests/test_autonomy_task_write_timestamp.py`: resolve what a `Z`-suffixed
# `strftime` was called on, follow it through the file's own assignments, and
# judge the subject rather than the line number. Matching by name over-approximates
# on purpose, so a shadowed or reused variable is judged by its worst assignment.

_CLOCK_READS = {"datetime.now", "datetime.datetime.now",
                "datetime.utcnow", "datetime.datetime.utcnow"}


def _dotted(node: ast.AST) -> str:
    """`datetime.now` for an attribute chain; `''` for anything else."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return ""
    parts.append(node.id)
    return ".".join(reversed(parts))


def _is_clock_read(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _dotted(node.func) in _CLOCK_READS


def _is_naive_clock_read(node: ast.AST) -> bool:
    """A system-clock read that names no timezone.

    `datetime.now()` with no argument is the host's wall clock. Any
    `datetime.utcnow()` counts whatever its arguments: it hands back a UTC *value*
    carrying no `tzinfo`, which is exactly the thing that reads as UTC once a `Z`
    is appended and misreads the moment it is compared to an aware instant.
    """
    if not _is_clock_read(node):
        return False
    if _dotted(node.func).endswith("utcnow"):
        return True
    return not node.args and not node.keywords


def _is_module_time_strftime(node: ast.AST) -> bool:
    """`time.strftime(...)` — a clock read the `strftime` form hides.

    The module form takes the current time implicitly, so with a `Z` format it is
    the same defect as #2246 wearing different syntax. It is NOT naive when the
    caller passes a `time.gmtime()` struct, which is UTC already.
    """
    return (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute) and node.func.attr == "strftime"
            and _dotted(node.func.value) == "time")


def _utc_struct_passed(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Call) and _dotted(n.func) == "time.gmtime"
               for n in ast.walk(node))


def _assigned_values(tree: ast.AST) -> dict:
    """`name -> every expression assigned to it anywhere in the file`."""
    out: dict[str, list] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out.setdefault(target.id, []).append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                out.setdefault(node.target.id, []).append(node.value)
    return out


def _z_stamps(src: str) -> list:
    """Every `strftime` whose format string ends in `Z`, with its subject resolved.

    `clock_sourced` says a clock read produced the value; `naive` says that read
    named no timezone; `passthrough` says the file was handed the value rather than
    reading the clock itself, which is the one shape the scan must tolerate.
    """
    tree = ast.parse(src)
    assigned = _assigned_values(tree)
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "strftime"):
            continue
        fmt = node.args[0] if node.args else None
        if not (isinstance(fmt, ast.Constant) and isinstance(fmt.value, str)
                and fmt.value.endswith("Z")):
            continue
        receiver = node.func.value
        subjects = [receiver]
        if isinstance(receiver, ast.Name):
            subjects = assigned.get(receiver.id, []) or [receiver]
        # `walk` each subject so a clock read buried in a chain
        # (`datetime.now().astimezone().strftime(...)`) or an expression
        # (`(datetime.now() - x).strftime(...)`) counts, not only one sitting
        # directly under the `strftime`. A chained or computed value is judged by
        # the clock read inside it; #2246 is about the value asserting UTC, and
        # where the naive read sits does not change that.
        def _sourced(subject: ast.AST) -> bool:
            return (any(_is_clock_read(n) for n in ast.walk(subject))
                    or _is_module_time_strftime(node))

        def _naive(subject: ast.AST) -> bool:
            if any(_is_naive_clock_read(n) for n in ast.walk(subject)):
                return True
            return _is_module_time_strftime(node) and not _utc_struct_passed(node)

        sourced = [_sourced(s) for s in subjects]
        found.append({
            "lineno": node.lineno,
            "subjects": subjects,
            "clock_sourced": any(sourced),
            "naive": any(_naive(s) for s in subjects),
            "passthrough": not any(sourced),
        })
    return found


def test_no_z_suffixed_stamp_in_the_linker_comes_from_a_naive_clock():
    """Clause 2: no `Z` stamp in this file may be formatted from a clock read that
    names no timezone, so the shape that produced #2246 cannot return unnoticed.

    The non-vacuity half is the second assertion: the file must keep at least one
    `Z` stamp that IS a clock read. A scan that passes because the format string
    was deleted or the stamp moved to another module is not a guard.
    """
    stamps = _z_stamps(SCRIPT.read_text(encoding="utf-8"))
    offenders = [s["lineno"] for s in stamps if s["naive"]]
    assert offenders == [], (
        f"Z-suffixed stamps formatted from a naive clock read in {SCRIPT}: "
        f"lines {offenders} — the name is the host's wall clock asserting UTC")
    clock_sourced = [s["lineno"] for s in stamps if s["clock_sourced"]]
    assert len(clock_sourced) >= 1, (
        "the apply's backup name is a clock-sourced Z stamp and must stay one; "
        f"found {len(stamps)} Z stamps, none from a clock read "
        f"(lines {[s['lineno'] for s in stamps]})")


def test_the_linker_scan_finds_the_shape_it_claims_to_find():
    """The scan's own falsifiability, in the shape #1128 pinned.

    Without this, the node above could be green because the matcher never
    matched anything — the failure mode this repo has a whole catalogue of. The
    expression that produced #2246, fed to the same code, has to be reported.
    """
    reported = ("ts = datetime.now().strftime('%Y%m%dT%H%M%SZ')\n"
                "path = f'kg-stranded-{ts}.sqlite'\n")
    stamps = _z_stamps(reported)
    assert len(stamps) == 1, "the Z-strftime itself was not found"
    assert stamps[0]["naive"], "the naive local form slipped through the scan"

    correct = ("ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')\n"
               "path = f'kg-stranded-{ts}.sqlite'\n")
    stamps = _z_stamps(correct)
    assert len(stamps) == 1 and stamps[0]["clock_sourced"]
    assert not stamps[0]["naive"], "the correct form was reported as a defect"

    assigned_indirectly = ("import datetime\n"
                           "now = datetime.datetime.now()\n"
                           "ts = now.strftime('%Y%m%dT%H%M%SZ')\n")
    stamps = _z_stamps(assigned_indirectly)
    assert stamps[0]["naive"], "a naive read hidden behind an assignment escaped"

    handed_in = ("def _stamp(received):\n"
                 "    return received.strftime('%Y%m%dT%H%M%SZ')\n")
    stamps = _z_stamps(handed_in)
    assert stamps[0]["passthrough"], "a value the file was handed was reported " \
                                     "as a clock read"

    chained = ("ts = datetime.now().astimezone().strftime('%Y%m%dT%H%M%SZ')\n")
    stamps = _z_stamps(chained)
    assert stamps[0]["naive"], "a naive read hidden inside a call chain escaped"

    chained_correct = ("import datetime\n"
                       "ts = datetime.datetime.now(datetime.timezone.utc)"
                       ".astimezone().strftime('%Y%m%dT%H%M%SZ')\n")
    stamps = _z_stamps(chained_correct)
    assert stamps[0]["clock_sourced"] and not stamps[0]["naive"], \
        "a tz-aware read wearing a chain was reported as a defect"

    module_form = ("import time\n"
                   "ts = time.strftime('%Y%m%dT%H%M%SZ')\n")
    stamps = _z_stamps(module_form)
    assert stamps[0]["naive"], "time.strftime with a Z format, which reads the " \
                              "host clock implicitly, slipped through"

    module_form_utc = ("import time\n"
                       "ts = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())\n")
    stamps = _z_stamps(module_form_utc)
    assert stamps[0]["clock_sourced"] and not stamps[0]["naive"], \
        "time.strftime over a gmtime struct was reported as a defect"


# ─────────────────────────────────────────────────────────────────────────────────
# #3437 — `--apply` needs an attended caller, and the environment is the only
# evidence a caller gets to offer.
#
# Before this section the linker's `--apply` was governed by one check,
# `writes_disabled_by_rebuild()`, which asks the store one question and never looks at
# who is asking. The rule it was supposed to honour — the autonomy-data-pipeline
# skill's SD-stranded-1 executor rule, and the standing debt whose wording is read out
# of `tests/test_owed_check.py` below — lived only as prose, so any caller that thought
# to pass the flag was obeyed. These nodes pin the class the code admits, and the two
# ways the gate could be made decorative if it were written the obvious way:
# `classify_session` answers ATTENDED for an id that names nothing, and a working
# directory under the session scratch root looks exactly like session evidence to a
# fallback that reads the directory name.
# ─────────────────────────────────────────────────────────────────────────────────

#: A live chat id, three parts — the shape `app.sessions_io` calls attended.
ATTENDED_SESSION = "20260922_201704_iv5e28"

#: A live worker id, four parts, date and time first: a background session.
BACKGROUND_SESSION = "20261031_093148_autocode_449b"

#: A subagent's own id. A standalone script carries no parent resolver, so
#: `classify_session` answers UNKNOWN for it — which is the point: an unresolvable
#: subagent pays the lookup and is refused rather than inheriting the attended answer
#: (#1961).
SUBAGENT_SESSION = "task:0f3c9e2a7b"

#: What a refused run must print. The freeze gate above also opens with `REFUSED
#: --apply`, so the marker has to reach into the caller half of the sentence: the two
#: refusals share a prefix precisely because they share a seat, and this file asserts
#: that they stay separable.
REFUSAL = "REFUSED --apply: an attended Lloyd session is required"

#: The session scratch root's directory name, as a literal: the bypass node below must
#: not reach `app.paths`, whose value a guard could re-point out from under the test.
SCRATCH_CWD_ROOT = "session-cwd"

#: The debt this gate enforces, read from the module that carries it, so a rewording
#: there is felt here instead of quietly diverging from the code.
APPLY_DEBT = (ROOT / "tests" / "test_owed_check.py").read_text(encoding="utf-8")


def _env_var() -> str:
    """The variable the gate reads, by the constant the stamper writes it under.

    `agent_mcp/_rpc.py` stamps the calling session's id into every nested Bash child
    through `rpc_policy.ENV_SESSION_ID`, attended and background alike. Naming it
    through that constant is what keeps this file from pinning a string the stamper is
    free to rename.
    """
    from app.harness.rpc_policy import ENV_SESSION_ID
    return ENV_SESSION_ID


@pytest.fixture(autouse=True)
def attended_caller_by_default(monkeypatch):
    """Every node in this file tests the linker's linking behaviour.

    So the twelve existing `linker.main(... "--apply" ...)` call sites — 28 lines of
    this file carry the token, twelve of them are calls, both measured at the base
    commit — stay exactly as they were written: the change under test is who may call,
    not what the linker does. The fixture supplies the attended caller for all of them
    through the one channel the script reads, the Bash tool's session variable, holding
    an id shaped like a live chat session's.

    Autouse and unconditional, the dry-run nodes included: a dry run from an attended
    caller is the ordinary case. The unattended nodes clear it again themselves, which
    is also what proves the gate consults the environment rather than some ambient
    default this fixture happens to leave set.
    """
    monkeypatch.setenv(_env_var(), ATTENDED_SESSION)


def _apply_call_sites(tree) -> list:
    """Every `linker.main([...])` call in a parsed copy of this file passing `--apply`."""
    return [node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "main" and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "linker"
            # `--apply` sits inside the argv list literal the call is handed, so the
            # scan has to go down into the argument rather than only over it.
            and any(isinstance(c, ast.Constant) and c.value == "--apply"
                    for a in node.args for c in ast.walk(a))]


def _pre_existing_apply_sites(text: str, mine=frozenset({
        "test_apply_refuses_unless_the_environment_proves_an_attended_caller",
        "test_the_working_directory_confers_no_session_class",
        "test_an_attended_apply_writes_exactly_what_it_always_wrote",
        "test_a_refused_apply_prints_below_the_denominators_and_above_the_backup",
        "test_the_attended_refusal_and_the_freeze_refusal_name_different_remedies",
    })) -> list:
    """The `--apply` call sites that were in this file before #3437.

    Everything under a node named in `mine` is this round's own probe; what is left is
    the inherited suite, and that is the population clause 3 says nobody edited.
    """
    # One parse, so the two walks speak about the same node objects.
    tree = ast.parse(text)
    owned = set()
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name in mine:
            owned |= {id(n) for n in _apply_call_sites(fn)}
    return [n for n in _apply_call_sites(tree) if id(n) not in owned]


def _deciding_bodies() -> dict:
    """Executable source of the two functions that answer "may this caller write".

    Docstrings stripped, deliberately: both functions spend their prose naming the
    bypass they forbid, so a scan over the whole segment would flag the sentences that
    warn against it. Nothing executable is dropped — `ast.get_source_segment` of each
    body statement below the docstring is the code that runs.
    """
    src = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(src)
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name in ("session_class",
                                                        "unattended_reason"):
            body = n.body[1:] if (n.body and isinstance(n.body[0], ast.Expr)
                                  and isinstance(n.body[0].value, ast.Constant)
                                  and isinstance(n.body[0].value.value, str)) else n.body
            out[n.name] = "\n".join(ast.get_source_segment(src, s) or "" for s in body)
    return out


def test_apply_refuses_unless_the_environment_proves_an_attended_caller(tmp_path, capsys,
                                                                        monkeypatch,
                                                                        caplog):
    """Clause 1: unset, empty, whitespace, BACKGROUND or UNKNOWN all refuse, no edge written.

    The empty case is why this gate had to be written deliberately rather than copied
    from the item's wording: `classify_session` answers ATTENDED for an empty id — by
    design, since no id is a session anyone can attribute privileges to and every
    guard already reads an empty id as "no guard applies". A gate that asked that
    question first would hand the write to any caller that never read the variable.
    #1053 settled the direction for the sibling guard, whose `_attended_by_session_id`
    refuses on an absent id *before* it calls the classifier; that shape is what is
    pinned here, together with the whitespace id the classifier would also hand over.
    """
    from app.harness.service_control import ATTENDED, classify_session

    st = build_store(tmp_path)
    seeded = _seed_one_live_edge(st)   # isolate this gate from the #1833 freeze
    var = _env_var()

    # The hole itself, recorded as a fact this gate must not inherit.
    assert classify_session("") == ATTENDED, \
        "classify_session no longer answers ATTENDED for an empty id, so the " \
        "refusal-before-asking needs re-justifying (#3437)"

    cases = [
        ("unset", lambda: monkeypatch.delenv(var, raising=False), linker.NO_SESSION_ID),
        ("empty string", lambda: monkeypatch.setenv(var, ""), linker.NO_SESSION_ID),
        ("whitespace only", lambda: monkeypatch.setenv(var, "   "), linker.NO_SESSION_ID),
        ("a background worker's id", lambda: monkeypatch.setenv(var, BACKGROUND_SESSION),
         "background"),
        ("a subagent's id", lambda: monkeypatch.setenv(var, SUBAGENT_SESSION), "unknown"),
    ]
    for label, act, expected in cases:
        act()
        assert linker.session_class() == expected, \
            f"{label}: session_class answered {linker.session_class()!r}"
        rc = linker.main(["--db", str(tmp_path / "kg.sqlite"), "--min-facts", "1",
                          "--sample", "0", "--out", str(tmp_path / "p.jsonl"), "--apply"])
        out = capsys.readouterr().out
        assert rc == 2, f"{label}: exit {rc}, out was {out[-500:]!r}"
        assert REFUSAL in out, f"{label}: refused without saying so — {out[-500:]!r}"
        assert "attended" in out.lower(), \
            f"{label}: the message must name the class it needs — {out[-500:]!r}"
        assert var in out or expected != linker.NO_SESSION_ID, \
            f"{label}: an absent id must name the variable it looked for — {out[-400:]!r}"
        assert "write_enabled" not in out, \
            f"{label}: met the freeze refusal instead of the caller's — {out[-400:]!r}"

    assert _backup_traces(tmp_path) == [], \
        "a refused --apply took a backup; refusing and mutating are exclusive"
    assert st.edges.count(active_only=True) == seeded, "a refused --apply wrote edges"

    # A miss against this gate is attributable to it rather than being invisible: #1961
    # made an unresolvable subagent pay the lookup, and the lookup is only useful if it
    # says which guard missed. `guard=` must therefore reach the warning.
    monkeypatch.setenv(var, SUBAGENT_SESSION)
    with caplog.at_level(logging.WARNING, logger="app.harness.service_control"):
        assert linker.session_class() == "unknown"
    assert linker.SESSION_GUARD in caplog.text, \
        f"no miss was attributed to {linker.SESSION_GUARD!r}: the guard name is not " \
        "reaching classify_session, so a systematic miss here reads as a quiet window"


def test_the_working_directory_confers_no_session_class(tmp_path, capsys, monkeypatch):
    """Clause 2: standing in a session's scratch directory proves nothing.

    The item asked for a cwd fallback — read the basename when the environment is
    empty. `app/session_cwd.path_for` names that directory after a *sanitised*
    projection of an id (untrusted characters replaced, truncated at 120), so it is not
    an invertible id to begin with, and any process on this machine may `cd` into one
    of the live directories under the session scratch root before running a command. A
    privilege conferred by a path the caller chooses is the hole, not the fallback, so
    the implementation reads the environment only.
    """
    scratch = tmp_path / SCRATCH_CWD_ROOT / ATTENDED_SESSION   # attended-shaped: 3 parts
    scratch.mkdir(parents=True)
    st = build_store(tmp_path / "elsewhere")                   # the store is not in the cwd
    seeded = _seed_one_live_edge(st)
    _write_flag(monkeypatch, True)
    monkeypatch.delenv(_env_var(), raising=False)
    monkeypatch.chdir(scratch)

    assert linker.session_class() == linker.NO_SESSION_ID, \
        "the working directory answered for the caller"
    rc = linker.main(["--db", str(tmp_path / "elsewhere" / "kg.sqlite"),
                      "--min-facts", "1", "--sample", "0",
                      "--out", str(tmp_path / "p.jsonl"), "--apply"])
    out = capsys.readouterr().out
    assert rc == 2, f"a cwd shaped like an attended session bought a write: exit {rc}"
    assert REFUSAL in out, out[-500:]
    assert st.edges.count(active_only=True) == seeded, \
        "the scratch-directory decoy bought an actual write"
    assert _backup_traces(tmp_path / "elsewhere") == []

    # The same thing from source level, so the behaviour cannot be bought back by a
    # later "fallback" that reads the directory: the two deciding functions touch
    # `os.environ` and nothing that could report where the process is standing.
    deciding = _deciding_bodies()
    assert set(deciding) == {"session_class", "unattended_reason"}, \
        "the deciding functions were renamed or split, and this scan is reading " \
        "whatever now shares their names"
    joined = "\n".join(deciding.values())
    assert "os.environ" in joined, "the caller is no longer read from the environment"
    for tell in ("getcwd", "cwd", SCRATCH_CWD_ROOT, "session_cwd", "chdir", "expanduser",
                 "readlink", "realpath", "os.path"):
        assert tell not in joined, \
            f"the deciding functions now consult {tell!r}: the working directory is " \
            "caller-controlled and confers no session class (#3437)"
    # Positive control, so the scan above is not vacuously true: a fallback spelled the
    # way the item asked for it carries exactly these tells.
    decoy = ('def session_class():\n    import os\n    return os.environ.get("X") '
             'or os.path.basename(os.getcwd())\n')
    for tell in ("getcwd", "cwd", "os.path"):
        assert tell in decoy, f"the decoy lost {tell!r}; the control no longer bites"


def test_an_attended_apply_writes_exactly_what_it_always_wrote(tmp_path, capsys,
                                                               monkeypatch):
    """Clause 3: attended in, nothing about the run moves — and the attended id is the
    fixture's, not something this node or a call site passes.

    The clause the round had to be careful of: the file's existing `--apply` call
    sites drive the linker directly, and a fail-closed gate goes red across every one of
    them. They stay green by one autouse fixture, so this node pins both halves — the
    write still happens exactly as before, and the call sites were not edited to
    accommodate the gate.
    """
    st = build_store(tmp_path)
    seeded = _seed_one_live_edge(st)
    _write_flag(monkeypatch, True)
    monkeypatch.setenv(_env_var(), ATTENDED_SESSION)   # explicit: not an ambient default

    rc = linker.main(["--db", str(tmp_path / "kg.sqlite"), "--min-facts", "1",
                      "--sample", "0", "--out", str(tmp_path / "p.jsonl"), "--apply"])
    out = capsys.readouterr().out
    assert rc == 0, f"an attended apply refused: {out[-700:]!r}"
    assert REFUSAL not in out, out[-700:]
    assert "applied 4 edges" in out, out[-400:]
    assert len(_backup_traces(tmp_path)) == 1, \
        "the attended apply's backup discipline changed"
    assert st.edges.count(active_only=True) == seeded + 4, out[-400:]

    # The pre-existing call sites did not move, and none of them started carrying
    # session evidence of its own. Twelve is the figure measured at the base commit;
    # the clause's 28 is the number of lines here that mention `--apply`, most of them
    # prose. A smaller figure than twelve would mean a site was deleted rather than
    # inherited by the fixture.
    text = Path(__file__).read_text(encoding="utf-8")
    pre = _pre_existing_apply_sites(text)
    assert len(pre) == 12, \
        f"{len(pre)} pre-existing `--apply` call sites, twelve at the base commit: the " \
        "gate was meant to be satisfied by the autouse fixture, not by call-site edits"
    for node in _apply_call_sites(ast.parse(text)):
        call = ast.get_source_segment(text, node) or ""
        assert _env_var() not in call and "session" not in call.lower(), \
            f"a call site started carrying session evidence: {call[:160]}"

    # Exactly one autouse fixture supplies it, so there is one lever, and no second
    # place where an id gets conjured for a node that did not ask for one.
    autouse = {}
    for n in ast.walk(ast.parse(text)):
        if isinstance(n, ast.FunctionDef) and any(
                isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "fixture"
                and any(k.arg == "autouse" and getattr(k.value, "value", None) is True
                        for k in d.keywords)
                for d in n.decorator_list):
            seg = ast.get_source_segment(text, n) or ""
            if "setenv" in seg:
                autouse[n.name] = seg
    assert list(autouse) == ["attended_caller_by_default"], \
        f"autouse fixtures that setenv anything: {list(autouse)}"
    assert "monkeypatch.setenv(_env_var(), ATTENDED_SESSION)" in \
        autouse["attended_caller_by_default"], \
        "the fixture stopped setting exactly the attended id through the stamper's " \
        f"constant `{_env_var()}`: {autouse['attended_caller_by_default'][-260:]!r}"

    # And the rule the gate enforces is the rule its carrier records, not a paraphrase
    # written beside the code.
    assert "attended" in APPLY_DEBT and "--apply" in APPLY_DEBT, \
        "APPLY_DEBT stopped naming an attended --apply: the class this gate admits has " \
        "moved and needs re-deriving from it"


def test_a_refused_apply_prints_below_the_denominators_and_above_the_backup(
        tmp_path, capsys, monkeypatch):
    """Clause 4: the seat is the #1833 gate's seat, and a dry run is never refused.

    The freeze node above pins that ordering for its own gate — denominators after the
    multi-second scan, refusal before `st.backup()`, because a snapshot of a store
    about to be replaced is a restore point and one of a store that will not be touched
    is noise. The attended gate takes the same position so a reader meets one block
    rather than two orders, and a refused apply leaves no `kg-stranded-*` file.

    The second half is the other direction of the same rule: with no session evidence at
    all, the run *without* `--apply` still exits 0 and still prints its denominators. A
    gate that blocked the read would be worked around by running the script from a
    shell with the flag — the behaviour this round exists to stop.
    """
    st = build_store(tmp_path)
    seeded = _seed_one_live_edge(st)
    _write_flag(monkeypatch, True)
    monkeypatch.delenv(_env_var(), raising=False)

    rc = linker.main(["--db", str(tmp_path / "kg.sqlite"), "--min-facts", "1",
                      "--sample", "0", "--out", str(tmp_path / "p.jsonl"), "--apply"])
    out = capsys.readouterr().out
    assert rc == 2, f"exit {rc}, out was {out[-400:]!r}"
    assert "proposed edges" in out, f"the denominators were swallowed: {out[-400:]!r}"
    assert out.index("proposed edges") < out.index(REFUSAL), \
        "the refusal came before the denominators; a refused caller still gets the " \
        "numbers, and the block is scanned top-down (#1833)"
    assert _backup_traces(tmp_path) == [], \
        "the refusal came above st.backup(); #1833's ordering governs this gate too"
    assert st.edges.count(active_only=True) == seeded

    # The same caller, the same store, no `--apply`: exit 0, denominators, no refusal.
    capsys.readouterr()
    rc = linker.main(["--db", str(tmp_path / "kg.sqlite"), "--min-facts", "1",
                      "--sample", "0", "--out", str(tmp_path / "p.jsonl")])
    out = capsys.readouterr().out
    assert rc == 0, f"a dry run was refused for having no session: exit {rc}, " \
                    f"{out[-400:]!r}"
    assert REFUSAL not in out, out[-400:]
    assert _shown_number(out, "proposed edges") == 4, out[-400:]
    assert _backup_traces(tmp_path) == []
    assert st.edges.count(active_only=True) == seeded, "a refused apply wrote edges"


def test_the_attended_refusal_and_the_freeze_refusal_name_different_remedies(
        tmp_path, capsys, monkeypatch):
    """The two refusals stay separable, because their remedies are opposite.

    A freeze says *not now, this store*; an unattended caller says *not you*. Read as
    one message the first invites a retry that cannot succeed and the second invites
    waiting on a flag that was never the problem. So each names only its own cause, and
    when both are true the caller answer comes first — it is a fact about the process
    and costs no config read.
    """
    st = build_store(tmp_path)
    seeded = _seed_one_live_edge(st)
    argv = ["--db", str(tmp_path / "kg.sqlite"), "--min-facts", "1", "--sample", "0",
            "--out", str(tmp_path / "p.jsonl"), "--apply"]

    _write_flag(monkeypatch, True)
    monkeypatch.delenv(_env_var(), raising=False)
    assert linker.main(argv) == 2
    out = capsys.readouterr().out
    assert REFUSAL in out and "write_enabled" not in out and "freeze" not in out.lower(), \
        f"the caller's refusal blamed the store: {out[-400:]!r}"

    _write_flag(monkeypatch, False)
    monkeypatch.setenv(_env_var(), ATTENDED_SESSION)
    assert linker.main(argv) == 2
    out = capsys.readouterr().out
    assert "write_enabled" in out and REFUSAL not in out, \
        f"the store's refusal blamed the caller: {out[-400:]!r}"

    _write_flag(monkeypatch, False)
    monkeypatch.delenv(_env_var(), raising=False)
    assert linker.main(argv) == 2
    out = capsys.readouterr().out
    assert REFUSAL in out and "attended" in out.lower(), \
        f"with both gates shut, the caller answer must come first: {out[-400:]!r}"
    assert _backup_traces(tmp_path) == []
    assert st.edges.count(active_only=True) == seeded, \
        "one of the refusals wrote edges anyway"


# ─────────────────────────────────────────────────────────────────────────────────
# #3768 — the attended class has to be reachable from a real child process.
#
# Every node above reaches `--apply` by `monkeypatch.setenv` in the pytest process
# and calling `linker.main` in the same process. That is the right shape for pinning
# the linker's own gate and the wrong shape for the question #3768 asks: whether any
# caller in production can ever satisfy it. The only writer of the variable was
# `agent_mcp/_rpc.py`'s `bash_env`, which answers `(None, None)` for a turn the loop
# did not stamp, and `harness.rpc.enabled` is false — so before that item, the
# honest environment a Lloyd Bash child gets had no id in it and every real apply
# refused. These nodes drive the script as its own process with the environment a
# real Bash child has, which is the only witness that crosses that boundary.
# ─────────────────────────────────────────────────────────────────────────────────

def _child_env(*, sid: str | None) -> dict:
    """A copy of this process's environment, with the session id set or removed.

    Set-or-removed rather than either alone, because the two runs have to differ in
    exactly one variable: the autouse fixture above has already put an attended id in
    `os.environ`, so an "unset" child built by copying the environment would inherit
    it and every refusal asserted below would be vacuous. And once #3768's own fix is
    live, the pytest process is itself a Bash child whose id was stamped, so
    inheriting would silently decide which class the child believes it is.
    """
    env = {k: v for k, v in os.environ.items() if k != _env_var()}
    if sid is not None:
        env[_env_var()] = sid
    env["PYTHONPATH"] = str(ROOT)
    return env


def _apply_child(db: Path, *, sid: str | None, live_db: Path):
    """Run the script as its own process, and report every store it opened.

    The child wraps `KGStore.__init__` before it runs the script, so the opened paths
    are the child's own record rather than a claim the parent makes about code it
    cannot see into: clause 4's "no test may touch the live store" is then read out of
    the process that could have touched it. The config the write gate reads is
    replaced inside the child (`writes_disabled_by_rebuild` does its
    `from app.config import CONFIG` at call time — the same reason the #2246 node
    above can pin a real process at all), so the apply here is held only by the caller
    class this item is about, with no rebuild freeze in the way. That half is not this
    node's: the freeze is pinned against a freeze-shaped config by
    `test_the_rebuild_freeze_still_refuses_apply`. `main` seats the caller-class check
    first — its own comment gives the reason, that it is a fact about the caller and
    costs no config read — and `writes_disabled_by_rebuild()` second, so this node's
    exit-2 reading is the caller refusal, and neither node has to hold the other's gate
    to cover the pair.
    """
    child = (
        "import runpy, sys\n"
        "import app.config, app.kg_store\n"
        "opened = []\n"
        "_real = app.kg_store.KGStore.__init__\n"
        "def _spy(self, *a, **k):\n"
        "    opened.append(str(a[0] if a else k.get('db_path') or ''))\n"
        "    return _real(self, *a, **k)\n"
        "app.kg_store.KGStore.__init__ = _spy\n"
        "app.config.CONFIG = {'knowledge_graph': {'write_enabled': True}}\n"
        "sys.argv = ['link_stranded_entities.py', '--apply', '--db', %r, '--sample', '0']\n"
        "try:\n"
        "    runpy.run_path(%r, run_name='__main__')\n"
        "finally:\n"
        "    sys.stderr.write('OPENED ' + __import__('json').dumps(\n"
        "        {'opened': opened, 'default_db': %r}) + '\\n')\n"
    ) % (str(db), str(SCRIPT), str(live_db))
    return subprocess.run([sys.executable, "-c", child], cwd=str(ROOT),
                          env=_child_env(sid=sid), capture_output=True, text=True,
                          timeout=300)


def _opened_paths(proc) -> dict:
    """The child's own record: which stores it opened, and which path was the default.

    One JSON object on one `OPENED ` line, so a reader gets the child's list and the
    `--db` default it did not use from the same write, and the parent cannot mistake a
    path in one for a path in the other.
    """
    line = next((ln for ln in proc.stderr.splitlines() if ln.startswith("OPENED ")), "")
    assert line, f"the child reported no store paths:\n{proc.stderr[-800:]}"
    witness = json.loads(line[len("OPENED "):])
    assert set(witness) == {"opened", "default_db"}, witness
    return witness


def test_a_real_apply_writes_only_under_an_attended_id(tmp_path):
    """Clause 4: the same script, the same copy of the store, one variable apart.

    The attended run is the crossing the owed entry says nothing has ever proved:
    `--apply` as its own process, holding the environment a Lloyd Bash child
    actually carries, exiting 0 and writing its four edges into a copy under tmp. The
    refused run is the identical call with the variable removed — exit 2, no edge in
    the copy, and no `kg-stranded-*` snapshot beside it, so a refusal cannot leave
    even the file that a write takes. Neither run may reach the live store, which the
    child's own `KGStore.__init__` record settles, and the live file's own size and
    mtime are re-read afterwards as a second instrument.
    """
    st = build_store(tmp_path)
    seeded = _seed_one_live_edge(st)
    st.close()

    run_dir = tmp_path / "apply-run"
    run_dir.mkdir()
    copy = run_dir / "kg.sqlite"
    copy.write_bytes((tmp_path / "kg.sqlite").read_bytes())
    live_db = Path(linker.VAULT_KG_DB)

    ok = _apply_child(copy, sid=ATTENDED_SESSION, live_db=live_db)
    assert ok.returncode == 0, (
        f"an attended apply from a real process refused: rc={ok.returncode}\n"
        f"{ok.stdout[-900:]}\n{ok.stderr[-900:]}")
    assert "applied 4 edges" in ok.stdout, ok.stdout[-900:]
    assert REFUSAL not in ok.stdout, ok.stdout[-900:]
    names = _backup_traces(tmp_path)
    assert len(names) == 1, f"a real apply takes exactly one backup: {names}"

    after_ok = KGStore(copy)
    try:
        assert after_ok.edges.count(active_only=True) == seeded + 4, \
            "the attended apply's edges did not land in the copy"
        assert after_ok.edges.find_active(STRANDED, "TGS-RAG", linker.EDGE_TYPE) \
            is not None, "the row #1019 is named for is the one that had to link"
    finally:
        after_ok.close()

    # The unset run: same argv, same copy-populated directory, no id in the
    # environment at all. Everything it must not do is what the run above did.
    ref = _apply_child(copy, sid=None, live_db=live_db)
    assert ref.returncode == 2, (
        f"a real apply with no session id in its environment wrote anyway: "
        f"rc={ref.returncode}\n{ref.stdout[-900:]}")
    assert REFUSAL in ref.stdout, ref.stdout[-900:]
    assert _backup_traces(tmp_path) == names, \
        "the refused apply took a backup beside the copy"
    after_ref = KGStore(copy)
    try:
        assert after_ref.edges.count(active_only=True) == seeded + 4, \
            "the refused apply wrote edges into the copy"
    finally:
        after_ref.close()

    # Clause 4's last clause, from each child's own witness: the only store either
    # run constructed was the copy under tmp, and neither one opened the path `--db`
    # defaults to — which in a live run is the production graph. (Under a test tree
    # `app.paths` re-anchors that default into the test root, so what this pins is the
    # property that matters: the run never touches the default, only the path it was
    # handed.) No size-or-mtime check on that store is asserted here: the live system
    # writes its own graph all day, so a moved mtime would not be this node's doing and
    # an unmoved one would not be evidence. A store a process never constructed cannot
    # have read or written, which is why the child reports the list itself.
    for proc, label in ((ok, "attended"), (ref, "unset")):
        witness = _opened_paths(proc)
        assert witness["opened"], f"the {label} run opened no store at all"
        assert witness["default_db"] == str(live_db), \
            f"the {label} child reported the wrong `--db` default"
        for path in witness["opened"]:
            assert Path(path).resolve() == copy.resolve(), \
                f"the {label} run opened {path!r}, not the copy at {copy}"
