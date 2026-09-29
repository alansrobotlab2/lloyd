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
import importlib.util
import re
import sys
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
}


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
    index is what would reject a second insert of one fact), no expiry."""
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
    assert linker.main(["--db", str(st.path), "--sample", "0"]) == 0
    out = capsys.readouterr().out

    pairs = [(p["source"], p["target"]) for p in linker.find_proposals(st)[0]]
    assert len(pairs) == len(set(pairs)) == 4, (
        f"exactly one edge per stranded entity, no duplicates: {pairs}")
    assert sorted(pairs) == sorted([
        (DUAL, TARGET),          # 16 facts at degree zero on the live store too
        (STRANDED, "TGS-RAG"),   # longest wins over the `RAG` nested inside it
        ("TGS-RAG", "RAG"),      # `TGS-RAG` is itself stranded here, and embeds `RAG`
        (ROLE_NOUN, BARE),       # bare/role noun: linked, not merged
    ]), pairs
    for unnameable in (VICTIM, ALONE):
        assert unnameable not in [s for s, _ in pairs], unnameable
    assert "dry-run" in out, "the default mode must announce itself"
    assert st.edges.count(active_only=True) == 0, "--apply was not passed; nothing may be written"
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
    assert dens["embed_a_name"] == dens["proposed"] + dens["skipped_no_entity_row"], (
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
    assert before == sorted([STRANDED, DUAL]), (
        "the fixture must reproduce #1019's premise before the fix: at degree zero, "
        "≥10 active facts, own name embedding another registered name. `CameraCfg` "
        "and `UniTacHand` are stranded with 11 facts each and neither is named, "
        f"which is the set the probe's own rule defines: {before}")

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
    assert len(kept) == st.edges.count(active_only=True) == 4, (
        "the store holds only linker-written edges, and every one must be carried "
        "across a re-derivation")
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
    fact_holding = len(linker.active_fact_counts(st))
    degree_zero = sum(1 for k, v in linker.active_fact_counts(st).items()
                      if st.edges.degree().get(k, 0) == 0)
    embed = sum(1 for k in linker.active_fact_counts(st)
                if linker.longest_embedded_name(k, index))

    assert _shown_number(out, "fact-holding entities") == fact_holding
    assert _shown_number(out, "degree-zero candidates") == degree_zero
    assert _shown_number(out, "of-which embed a name") == embed
    assert _shown_number(out, "proposed edges") == embed, (
        "every embedding candidate is proposed, including `TGS-RAG` and `Browser Tool` "
        "under the probe's 10-fact floor — the rule is not scoped to that floor")
    # The count is not printed alone: each slice row names the set it is drawn from.
    assert f"(of {fact_holding} fact-holding)" in out, out
    assert f"(of {degree_zero} degree-zero candidates)" in out, out
    assert "(of 4 embedding a name)" in out, out
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
    # Fixture counts are 16/12/11/11/5/5/4/4/3/3, so a floor of 13 leaves
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
    assert seeded == 1, "the freeze check needs a live edge to leave untouched"
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
