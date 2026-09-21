"""`scripts/memory/repair_fact_duplicates.py` — the reversible stock repair (#1144).

Five things this file pins, one per acceptance clause:

* dry run by default; ``--apply`` removes byte-identical copies inside one file
  and across one entity's category files, and never across alias folders;
* the survivor is the highest confidence, then the copy carrying
  ``source_doc``/``created_at``, then the earliest;
* every removal lands in a JSONL manifest whose ``--restore`` reproduces the
  pre-repair files byte for byte;
* distinct ``(entity, text_hash)`` pairs are unchanged, no expired or invalid row
  is touched, and the same-entity redundant-row count falls by exactly the
  manifest's row count;
* ``--apply`` refuses without a before/after vault_recall comparison, passes with
  a clean one, and refuses a fall in ``fact_entity_recall_avg`` or any query that
  loses an expected entity.

The fixture is written through the extractor's own body generator, so the bytes
under repair are the bytes the nightly writer produces. A fixture in a format of
this test's own invention would pass while the real tree failed.
"""
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "memory"))
sys.path.insert(0, str(ROOT / "scripts" / "memory" / "next-gen-memory"))

from app import kg_store  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rf = _load("repair_fact_duplicates", "scripts/memory/repair_fact_duplicates.py")
fx = _load("fact_extractor_for_repair", "scripts/memory/next-gen-memory/fact_extractor.py")


def fact(text, **kw):
    """One entry, as the extractor writes it after stamping provenance."""
    e = {"entity": kw.pop("entity", "Lloyd"), "fact": text, "confidence": 0.9,
         "event_date": None, "category": kw.pop("category", "state"),
         "provenance": "EXTRACTED"}
    e.update(kw)
    return e


def write_tree(root: Path, entity: str, category: str, entries: list[dict]) -> Path:
    """Write ``<root>/<entity>/<entity>-<category>.md`` exactly as `write_fact_file` does.

    Same frontmatter dump (``sort_keys=False``, so key order is the entry's own)
    and the same generated body, including the ``**Fact Count:**`` line and the
    per-fact ``### <id>`` sections the repair has to keep consistent.
    """
    d = root / entity
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{entity}-{category}.md"
    fm = {"type": "facts", "entity": entity, "category": category, "facts": entries,
          "last_extracted": "2026-09-10T00:00:00.000000+00:00",
          "last_updated": "2026-09-10T00:00:00.000000+00:00"}
    body = fx.FactExtractor()._generate_markdown_body(entity, category, entries)
    path.write_text("---\n" + yaml.dump(fm, default_flow_style=False, sort_keys=False)
                    + "---\n\n" + body, encoding="utf-8")
    return path


def read_entries(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8").split("---")[1])["facts"] or []


def snapshot(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): p.read_text(encoding="utf-8")
            for p in sorted(root.rglob("*.md"))}


@pytest.fixture
def tree(tmp_path):
    """A fact tree, an index pointed at it, and the CLI pointed at both.

    The store is built rather than stubbed: clause 4 is a claim about row counts
    in ``facts_idx``, and it is the same query ``--apply`` prints, so a fixture
    store and the tool have to read the number the same way.
    """
    facts = tmp_path / "facts"
    facts.mkdir()
    db = tmp_path / "kg.sqlite"
    kg_store.configure(db)
    try:
        yield facts, db
    finally:
        kg_store.reset()


def index(root: Path, db) -> None:
    kg_store.configure(db)
    kg_store.store().facts_idx.reindex(root=root)


def planned_apply(facts: Path, manifest: Path) -> dict:
    """Survey the tree and write what the survey found, bypassing the eval gate.

    The gate has its own tests below against a stubbed comparison; the removal and
    restore semantics must not depend on the eval engine being reachable.
    """
    planned = rf.plan_removals(facts)
    res = rf.apply_removals(planned["plans"], planned["removals_by_file"], manifest,
                            apply=True)
    return {**planned, **res}


def manifest_lines(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ── clause 2: what it removes, and which copy survives ──────────────────────

def test_dry_run_is_the_default_and_changes_nothing(tree):
    """Removing stock is a decision, so the default is a survey. A run that
    removed first and reported after would hand a person a description of the
    thing they meant to be asked about."""
    facts, _db = tree
    write_tree(facts, "Lloyd", "state", [fact("runs on vLLM", id="stat-001")])
    write_tree(facts, "Lloyd", "relationship",
               [fact("runs on vLLM", id="rela-001", category="relationship",
                     confidence=0.7)])
    before = snapshot(facts)
    manifest = facts / "m.jsonl"

    planned = rf.plan_removals(facts)
    res = rf.apply_removals(planned["plans"], planned["removals_by_file"], manifest,
                            apply=False)

    assert res["removed_rows"] == 1, "the dry run reported no removal at all"
    assert snapshot(facts) == before, "the default mode wrote to the fact tree"
    # The stricter claim, asserted whole: a manifest is the record a restore
    # replays, so a survey that leaves one behind hands out a document that
    # describes nothing, and `--restore` of it looks like a no-op success.
    assert not manifest.exists(), \
        "a dry run must not hand out a manifest it never earned"


def test_a_dry_run_never_truncates_a_manifest_it_was_pointed_at(tree):
    """The disjunctive form of the assertion above hid the actual defect: opening
    the manifest in ``"w"`` mode makes a survey *create* the file, and pointing a
    survey at a real manifest truncates the only record that can put those rows
    back. A dry run must not be able to destroy an apply's audit trail."""
    facts, _db = tree
    write_tree(facts, "Lloyd", "state", [fact("runs on vLLM", id="stat-001")])
    write_tree(facts, "Lloyd", "relationship",
               [fact("runs on vLLM", id="rela-001", category="relationship",
                     confidence=0.7)])
    manifest = facts / "m.jsonl"
    prior = planned_apply(facts, manifest)
    assert prior["removed_rows"] == 1
    kept = manifest.read_bytes()
    assert manifest_lines(manifest), "the apply wrote no manifest to lose"

    planned = rf.plan_removals(facts)
    res = rf.apply_removals(planned["plans"], planned["removals_by_file"], manifest,
                            apply=False)

    assert manifest.read_bytes() == kept, \
        "a dry run rewrote the manifest that documents the real repair"
    assert manifest_lines(manifest), "the manifest lost its lines anyway"
    res = rf.restore(manifest, facts_root=facts, apply=True)
    assert res["restored"] == 1 and not res["refused"], \
        "the survey left a manifest that can no longer put the row back"


def test_duplicates_inside_one_file_are_folded(tree):
    """Class (a) of the 2026-09-14 measurement: 676 groups / 684 rows where one
    file holds the same text twice. `_merge_facts` folds those only when the
    document is re-extracted, and the content-hash gate means it usually is not."""
    facts, _db = tree
    path = write_tree(facts, "Lloyd", "state", [
        fact("serves the bench model", id="stat-001", confidence=0.8),
        fact("serves the bench model", id="stat-002", confidence=0.95),
        fact("keeps a vault", id="stat-003"),
    ])

    res = planned_apply(facts, facts / "m.jsonl")
    assert res["same_file_groups"] == 1 and res["cross_file_groups"] == 0

    left = read_entries(path)
    assert [f["fact"] for f in left] == ["serves the bench model", "keeps a vault"]
    assert left[0]["id"] == "stat-002", "the lower-confidence copy survived"


def test_duplicates_across_categories_are_folded(tree):
    """Class (b): 4,986 groups / 5,172 rows — one entity, two category files. This
    is the shape `_merge_facts` structurally cannot see, and 5,114 of the groups
    measured on the live store 2026-09-21 span two different files."""
    facts, _db = tree
    state = write_tree(facts, "Lloyd", "state",
                       [fact("runs on vLLM", id="stat-001", confidence=0.9)])
    rel = write_tree(facts, "Lloyd", "relationship",
                     [fact("runs on vLLM", id="rela-001", category="relationship",
                           confidence=0.9, source_doc="knowledge/x.md")])

    res = planned_apply(facts, facts / "m.jsonl")
    assert res["cross_file_groups"] == 1

    assert [f["id"] for f in read_entries(rel)] == ["rela-001"]
    assert read_entries(state) == [], "the copy without source_doc should have gone"
    assert state.exists(), "the repair removes entries, not whole category files"


def test_alias_folders_are_never_touched(tree):
    """Class (c): 44 groups / 45 rows across folders like `AgentSkills/` and
    `Agent Skills/`. That is entity resolution's to fuse; removing one half of a
    split pair makes the split harder to see, not easier."""
    facts, _db = tree
    a = write_tree(facts, "AgentSkills", "state", [fact("ships skills", id="stat-001")])
    b = write_tree(facts, "Agent Skills", "state", [fact("ships skills", id="stat-001")])

    assert rf.plan_removals(facts)["removals"] == []
    planned_apply(facts, facts / "m.jsonl")
    assert read_entries(a) and read_entries(b), "an alias pair was edited anyway"


def test_survivor_is_the_highest_confidence_copy(tree):
    facts, _db = tree
    keep = write_tree(facts, "Lloyd", "state", [
        fact("tuned at 0.9", id="stat-001", confidence=0.9),
        fact("tuned at 0.9", id="stat-002", confidence=0.95),
        fact("tuned at 0.9", id="stat-003", confidence=0.9),
    ])

    planned = rf.plan_removals(facts)
    assert sorted({o["entry"]["id"] for o in planned["removals"]}) == ["stat-001", "stat-003"]
    planned_apply(facts, facts / "m.jsonl")
    assert [f["id"] for f in read_entries(keep)] == ["stat-002"]


def test_meta_then_created_at_break_a_confidence_tie(tree):
    """The ruling's second and third keys, in that order: the copy carrying
    `source_doc`/`created_at` outlives the one carrying neither, and among dated
    copies the earliest wins — the first holder is the one any outside record
    already names."""
    facts, _db = tree
    with_doc = write_tree(facts, "Lloyd", "state", [
        fact("same claim", id="stat-001", source_doc="knowledge/a.md",
             created_at="2026-09-02T00:00:00+00:00")])
    dated = write_tree(facts, "Lloyd", "event",
                       [fact("same claim", id="even-001", category="event",
                             created_at="2026-09-01T00:00:00+00:00")])
    bare = write_tree(facts, "Lloyd", "goal",
                      [fact("same claim", id="goal-001", category="goal")])

    planned = rf.plan_removals(facts)
    assert {o["rel"] for o in planned["removals"]} == {"Lloyd/Lloyd-event.md",
                                                      "Lloyd/Lloyd-goal.md"}
    planned_apply(facts, facts / "m.jsonl")
    assert [f["id"] for f in read_entries(with_doc)] == ["stat-001"]
    assert read_entries(dated) == [] and read_entries(bare) == []


def test_a_tie_all_the_way_down_still_has_one_survivor(tree):
    """No confidence, no `source_doc`, no date: the plan must still be decidable
    and reproducible. The final tiebreak is the file and the entry's position in
    it, so one tree yields one plan rather than one plan per directory order, and
    a re-run before the apply agrees with the survey that produced it."""
    facts, _db = tree
    path = write_tree(facts, "Twin", "state", [
        fact("no metadata at all", id="stat-002", confidence=None),
        fact("no metadata at all", id="stat-001", confidence=None),
    ])

    first = rf.plan_removals(facts)
    second = rf.plan_removals(facts)
    assert len(first["removals"]) == 1
    assert [(o["seq"], o["entry"]["id"]) for o in first["removals"]] == [(1, "stat-001")], \
        "the later block in the file is the one that goes"
    assert [(o["seq"], o["entry"]["id"]) for o in second["removals"]] == \
        [(o["seq"], o["entry"]["id"]) for o in first["removals"]], "the plan is not reproducible"
    planned_apply(facts, facts / "m.jsonl")
    assert [f["id"] for f in read_entries(path)] == ["stat-002"]


def test_the_body_header_count_follows_the_survivors(tree):
    """The generated body carries `**Fact Count:** N` and one `### <id>` section
    per fact. Leaving either behind would render a file whose prose disagrees with
    its own frontmatter, which is the drift that made the store untrustworthy once
    before."""
    facts, _db = tree
    path = write_tree(facts, "Lloyd", "state", [fact("twin", id="stat-001"),
                                                fact("twin", id="stat-002",
                                                     confidence=0.4)])

    planned_apply(facts, facts / "m.jsonl")

    text = path.read_text(encoding="utf-8")
    assert "**Fact Count:** 1" in text
    assert text.count("### stat-002") == 0
    assert text.count("### stat-001") == 1
    assert text.count("fact: twin") == 1, "the survivor's own entry must stay intact"


# ── clause 3: the manifest and the restore ──────────────────────────────────

def test_manifest_records_path_and_the_full_removed_entry(tree):
    facts, _db = tree
    write_tree(facts, "Lloyd", "state", [fact("x", id="stat-001",
                                              source_doc="knowledge/a.md")])
    losing = fact("x", id="rela-001", category="relationship", provenance="INFERRED",
                  confidence=0.4)
    write_tree(facts, "Lloyd", "relationship", [losing])
    manifest = facts / "m.jsonl"

    planned_apply(facts, manifest)

    assert len(manifest_lines(manifest)) == 1, "one line per removed fact"
    rec = manifest_lines(manifest)[0]
    assert rec["file_path"] == "Lloyd/Lloyd-relationship.md"
    assert rec["entry"] == losing, "the record carries the whole entry, not a summary"
    assert rec["block_lines"], "restore replays source lines, not a re-dump"
    assert rec["text_hash"] and rec["schema"] == rf.SCHEMA


def test_restore_reproduces_the_pre_repair_bytes(tree):
    facts, _db = tree
    state = write_tree(facts, "Lloyd", "state", [
        fact("first claim", id="stat-001"),
        fact("duplicated claim", id="stat-002", confidence=0.8),
        fact("last claim", id="stat-003"),
    ])
    rel = write_tree(facts, "Lloyd", "relationship",
                     [fact("duplicated claim", id="rela-001", category="relationship",
                           confidence=0.7)])
    before = {p.name: p.read_text(encoding="utf-8") for p in (state, rel)}
    manifest = facts / "m.jsonl"

    planned_apply(facts, manifest)
    assert rel.read_text(encoding="utf-8") != before["Lloyd-relationship.md"]
    assert "**Fact Count:** 0" in rel.read_text(encoding="utf-8")

    res = rf.restore(manifest, facts_root=facts, apply=True)
    # Only the relationship file is in the manifest: the `state` copy survived,
    # so that file was never touched and has nothing to put back.
    assert res["restored"] == 1 and not res["refused"], res
    after = {p.name: p.read_text(encoding="utf-8") for p in (state, rel)}
    assert after == before, "restore must be byte-for-byte, not equivalent"


def test_restore_of_two_adjacent_removals_keeps_their_order(tree):
    """Three copies of one text in one file: the two losers are adjacent, so a
    recorded line number would be identical for both and replay in either order.
    The manifest records how many SURVIVORS stand ahead, so replaying in
    descending `seq` rebuilds `stat-002` ahead of `stat-003`."""
    facts, _db = tree
    path = write_tree(facts, "Lloyd", "state", [
        fact("keeper", id="stat-001", confidence=0.99),
        fact("twin", id="stat-002", confidence=0.5),
        fact("twin", id="stat-003", confidence=0.5),
        fact("twin", id="stat-004", confidence=0.9),
    ])
    before = path.read_text(encoding="utf-8")
    manifest = facts / "m.jsonl"

    res = planned_apply(facts, manifest)
    assert len(res["removals"]) == 2
    assert [f["id"] for f in read_entries(path)] == ["stat-001", "stat-004"]

    restore = rf.restore(manifest, facts_root=facts, apply=True)
    assert not restore["refused"], restore
    assert path.read_text(encoding="utf-8") == before


def test_restore_survives_a_folded_scalar_and_unicode(tree):
    """Byte-for-byte is only worth claiming against the format that breaks a
    re-dump: `yaml.dump` folds a long fact across lines and escapes non-ASCII, so
    the manifest has to carry the file's own lines rather than a re-serialisation."""
    facts, _db = tree
    long_text = ("the robot trains in Isaac Lab on a 4090 and evaluates on the bench "
                 "with a mixture of gRPC calls, CUDA kernels and 日本語 tokens, which is "
                 "long enough that the writer folds it across several lines")
    # The FOLDED copy is the one that loses (0.5 against 0.6), so the byte-exact
    # claim is tested against the file whose block the writer wrapped.
    state = write_tree(facts, "Lloyd", "state", [fact(long_text, id="stat-001",
                                                      confidence=0.5)])
    rel = write_tree(facts, "Lloyd", "relationship",
                     [fact(long_text, id="rela-001", category="relationship",
                           confidence=0.6)])
    before = {p.name: p.read_text(encoding="utf-8") for p in (state, rel)}
    assert re.search(r"^ {4}\S", state.read_text(encoding="utf-8"), re.M), \
        "the fixture has to actually be folded or this pins nothing"
    manifest = facts / "m.jsonl"

    planned_apply(facts, manifest)
    assert read_entries(state) == []
    res = rf.restore(manifest, facts_root=facts, apply=True)
    assert not res["refused"], res
    assert {p.name: p.read_text(encoding="utf-8") for p in (state, rel)} == before


def test_restore_refuses_a_tree_it_does_not_belong_to(tree):
    """Splicing a fact into a stranger's file and reporting success is the worst
    thing a restore mode can do. The record names the entity and category it came
    from, so a manifest pointed at another entity's file is refused before a byte
    is written."""
    facts, _db = tree
    write_tree(facts, "Lloyd", "state", [fact("mine", id="stat-001")])
    write_tree(facts, "Lloyd", "relationship",
               [fact("mine", id="rela-001", category="relationship", confidence=0.1)])
    manifest = facts / "m.jsonl"
    planned_apply(facts, manifest)

    other = write_tree(facts, "Other", "state", [fact("someone else", id="stat-001",
                                                     entity="Other")])
    stranger = facts / "stranger.jsonl"
    stranger.write_text("\n".join(
        json.dumps({**json.loads(line), "file_path": "Other/Other-state.md"})
        for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()),
        encoding="utf-8")

    res = rf.restore(stranger, facts_root=facts, apply=True)
    assert res["restored"] == 0 and res["refused"], res
    assert "mine" not in other.read_text(encoding="utf-8"), "the stranger's file was edited"
    assert read_entries(other) == [fact("someone else", entity="Other", id="stat-001")]


def test_a_file_left_with_no_facts_stays_parseable_downstream(tree):
    """Removing a file's only copy must not leave a dangling `facts:` key:
    `entity_merge_disposition._read_fact_file` does `get("facts", [])`, which
    hands back the null a bare `facts:` parses to and then iterates."""
    facts, _db = tree
    emptied = write_tree(facts, "Lloyd", "goal",
                         [fact("only copy", id="goal-001", category="goal", confidence=0.3)])
    write_tree(facts, "Lloyd", "state", [fact("only copy", id="stat-001")])

    planned_apply(facts, facts / "m.jsonl")

    fm = yaml.safe_load(emptied.read_text(encoding="utf-8").split("---")[1])
    assert fm["facts"] == [], "an emptied fact file must parse to a list, not null"


# ── clause 2/3: the shapes the live tree actually holds ────────────────────

def test_a_quoted_multiline_scalar_block_is_repaired_byte_for_byte(tree):
    """A `fact:` value PyYAML has to quote and wrap — what a claim containing
    `: ` becomes under the real writer (`allow_block_flow_scalar=False`, so a
    scalar with a colon inside is quoted and then folded at width 80).

    This is the shape that punishes a block parser which drops the block's first
    line before parsing: what is left is a quoted scalar's tail followed by a
    bare `confidence:` key, which yaml rejects outright ("mapping values are not
    allowed here"), and the entry then looks absent from a file the loader says
    has facts in it.
    """
    facts, _db = tree
    colon_text = ("evidence path: a set of Wikidata (subject, property, object) "
                  "triples describing the reasoning path, generated by templates "
                  "plus 28 AMIE-mined logical rules, which is the point of the "
                  "benchmark and not an incidental detail")
    state = write_tree(facts, "Lloyd", "state",
                       [fact(colon_text, id="stat-001", confidence=0.6)])
    write_tree(facts, "Lloyd", "evaluation",
               [fact(colon_text, id="eval-001", category="evaluation", confidence=0.9)])
    before = state.read_text(encoding="utf-8")
    assert re.search(r"^\s*fact: [" + "'" + "]", before, re.M), "the fixture must be the quoted shape"

    manifest = facts / "m.jsonl"
    planned_apply(facts, manifest)
    assert [f["id"] for f in read_entries(state)] == []

    res = rf.restore(manifest, facts_root=facts, apply=True)
    assert not res["refused"], res
    assert state.read_text(encoding="utf-8") == before


def test_an_unparsable_block_is_skipped_and_never_aborts_the_pass(tree):
    """One file the plan cannot read is a `skipped` line, not a dead run.

    A facts sequence carrying one real entry and one bare scalar (`- just text`)
    parses to one entry for the loader while the line geometry sees two blocks,
    so the two cannot be tied back to each other and the file is left completely
    alone. The pass must go on and repair the real duplicate sitting next to it:
    an unhandled exception on the first odd file would fix nothing and report
    nothing, and the live tree does hold files whose blocks a per-block re-parse
    refuses.
    """
    facts, _db = tree
    odd = facts / "Odd"
    odd.mkdir()
    odd_file = odd / "Odd-state.md"
    odd_file.write_text(
        "---\n"
        "type: facts\n"
        "entity: Odd\n"
        "category: state\n"
        "facts:\n"
        "- entity: Odd\n"
        "  fact: a real entry\n"
        "  id: stat-001\n"
        "- just text, not a mapping at all\n"
        "---\n\n# Odd - State\n", encoding="utf-8")
    good = write_tree(facts, "Lloyd", "state", [fact("alpha", id="stat-001", confidence=0.4),
                                               fact("alpha", id="stat-002", confidence=0.9)])
    # 0.3 so the choice is not a tie: the goal copy must lose on confidence, and
    # a tie would be broken by folder name instead, which is not what is under
    # test here.
    write_tree(facts, "Lloyd", "goal",
               [fact("alpha", id="goal-001", category="goal", confidence=0.3)])
    odd_before = odd_file.read_text(encoding="utf-8")

    planned = planned_apply(facts, facts / "m.jsonl")

    assert planned["skipped"].get("Odd/Odd-state.md"), planned["skipped"]
    assert odd_file.read_text(encoding="utf-8") == odd_before, "a refused file must not move"
    assert [f["id"] for f in read_entries(good)] == ["stat-002"]



def test_distinct_pairs_unchanged_and_rows_fall_by_the_manifest_count(tree):
    facts, db = tree
    write_tree(facts, "Lloyd", "state", [
        fact("alpha", id="stat-001"), fact("alpha", id="stat-002", confidence=0.5),
        fact("beta", id="stat-003"),
    ])
    write_tree(facts, "Lloyd", "relationship",
               [fact("alpha", id="rela-001", category="relationship", confidence=0.5)])
    write_tree(facts, "vLLM", "state", [fact("serves models", id="stat-001")])
    index(facts, db)
    before = rf._duplicate_pairs(db)
    assert (before["active_groups"], before["active_redundant_rows"]) == (1, 2), before
    assert before["distinct_entity_text_pairs"] == 3, before

    manifest = facts / "m.jsonl"
    res = planned_apply(facts, manifest)
    index(facts, db)
    after = rf._duplicate_pairs(db)

    assert res["removed_rows"] == len(manifest_lines(manifest)) == 2, res
    assert (after["active_groups"], after["active_redundant_rows"]) == (0, 0), after
    assert (before["active_redundant_rows"] - after["active_redundant_rows"]
            == len(manifest_lines(manifest)) == res["removed_rows"])
    assert after["distinct_entity_text_pairs"] == before["distinct_entity_text_pairs"] == 3, \
        "a repair that merged distinct texts is not a repair"
    assert (before["same_entity_redundant_rows"] - after["same_entity_redundant_rows"]
            == len(manifest_lines(manifest)))


def test_expired_and_invalid_copies_are_never_touched(tree):
    """Clause 4's `touches no expired or invalid row`, and the reason the repair
    removes rather than expires: `_merge_facts` keeps a retired copy and an active
    copy of one text apart on purpose, and `exact_duplicate_stats()` counts expired
    rows, so expiring would leave this item's own number where it was."""
    facts, db = tree
    a = write_tree(facts, "Lloyd", "state", [
        fact("superseded claim", id="stat-001", expired_at="2026-01-01T00:00:00+00:00"),
        fact("superseded claim", id="stat-002"),
        fact("both retired", id="stat-003", invalid_at="2026-02-01T00:00:00+00:00"),
    ])
    write_tree(facts, "Lloyd", "relationship", [
        fact("superseded claim", id="rela-001", category="relationship",
             invalid_at="2026-03-01T00:00:00+00:00")])
    write_tree(facts, "Lloyd", "goal", [
        fact("both retired", id="goal-001", category="goal",
             expired_at="2026-02-15T00:00:00+00:00")])
    before = snapshot(a.parent)
    index(facts, db)
    retired_sql = ("SELECT COUNT(*) n FROM facts_idx "
                   "WHERE expired_at IS NOT NULL OR invalid_at IS NOT NULL")
    retired_before = kg_store.store()._query(retired_sql)[0]["n"]
    assert retired_before == 4, retired_before

    planned = rf.plan_removals(facts)
    assert planned["mixed_status_groups"] == 1, "a live + retired pair must be refused"
    assert planned["retired_groups"] == 1, "an all-retired group must be refused"
    assert planned["removals"] == [], planned["removals"]

    planned_apply(facts, facts / "m.jsonl")
    assert snapshot(a.parent) == before, "a retired copy was edited"
    index(facts, db)
    assert kg_store.store()._query(retired_sql)[0]["n"] == retired_before


# ── clause 5: --apply is gated on a before/after retrieval comparison ───────

def _stub_recall(avg=0.45, entities=("Lloyd",), errors=0):
    return {"fact_entity_recall_avg": avg, "errors": errors, "corpus": {"facts": 3},
            "facts_root": "stub", "kg_db": "stub",
            "records": [{"query": "who is lloyd", "category": "entity",
                         "fact_entities_matched": list(entities)}]}


@pytest.fixture
def gated(tree, monkeypatch):
    """A tree with one real duplicate, indexed, with the comparison stubbed.

    Stubbed rather than real: the clause is about the gate — refuse without one,
    pass with a clean one, refuse a fall and a lost entity — and the scoring
    itself is `eval/run_eval.py`'s own tested surface.
    """
    facts, db = tree
    write_tree(facts, "Lloyd", "state", [fact("runs on vLLM", id="stat-001")])
    write_tree(facts, "Lloyd", "relationship",
               [fact("runs on vLLM", id="rela-001", category="relationship",
                     confidence=0.3)])
    index(facts, db)
    monkeypatch.setattr(rf, "run_recall", lambda *a, **k: _stub_recall())
    return facts, db


def cli(facts: Path, db: Path, manifest: Path, *extra: str) -> int:
    return rf.main(["--facts-dir", str(facts), "--db", str(db),
                    "--manifest", str(manifest), *extra])


def test_apply_without_a_recall_comparison_is_refused(gated, monkeypatch):
    """The refusal has to be loud and exit non-zero. A silent fallback to "no
    gate" is how a guard dies: the flag survives, the protection stops."""
    facts, db = gated
    before = snapshot(facts)
    started = []
    monkeypatch.setattr(rf, "run_recall",
                        lambda *a, **k: started.append(a) or _stub_recall())

    rc = cli(facts, db, facts / "m.jsonl", "--apply")

    assert rc == 1, "removing rows without a comparison must not exit 0"
    assert started == [], "the refused run must not even start a comparison"
    assert snapshot(facts) == before, "a refusal must not have written"


def test_apply_passes_when_the_stubbed_comparison_is_clean(gated):
    facts, db = gated
    manifest = facts / "m.jsonl"

    rc = cli(facts, db, manifest, "--apply", "--recall-compare")

    assert rc == 0, rc
    assert read_entries(facts / "Lloyd" / "Lloyd-relationship.md") == []
    assert len(manifest_lines(manifest)) == 1


def test_apply_refuses_a_store_that_could_not_show_a_change(gated, tmp_path):
    """A comparison is only evidence if it could have moved. Against a store with
    no duplicate group at all, before and after are identical by construction and
    would certify any edit whatsoever — so the run refuses rather than passes."""
    facts, _db = gated
    empty_db = tmp_path / "empty" / "kg.sqlite"
    empty_db.parent.mkdir()
    kg_store.configure(empty_db)
    kg_store.store().facts_idx.reindex(root=empty_db.parent)

    rc = cli(facts, empty_db, facts / "m.jsonl", "--apply", "--recall-compare")

    assert rc == 1, rc
    assert read_entries(facts / "Lloyd" / "Lloyd-relationship.md"), "it wrote anyway"


def test_apply_refuses_a_fall_in_fact_entity_recall(gated, monkeypatch):
    facts, db = gated
    seq = iter([_stub_recall(avg=0.35), _stub_recall(avg=0.30)])
    monkeypatch.setattr(rf, "run_recall", lambda *a, **k: next(seq))

    rc = cli(facts, db, facts / "m.jsonl", "--apply", "--recall-compare")

    assert rc == 1, "a measured fall in fact_entity_recall_avg must not exit 0"


def test_apply_refuses_a_query_that_loses_an_expected_entity(gated, monkeypatch):
    """The average can hold steady while one query loses its entity — the shape
    #499's 0.35 → 0.30 recall note was about, and why the harness compares
    per-query expected-entity sets and not only the mean."""
    facts, db = gated
    seq = iter([_stub_recall(entities=("Lloyd", "vLLM")), _stub_recall(entities=("Lloyd",))])
    monkeypatch.setattr(rf, "run_recall", lambda *a, **k: next(seq))

    rc = cli(facts, db, facts / "m.jsonl", "--apply", "--recall-compare")

    assert rc == 1, rc


def test_an_unmeasured_arm_is_not_a_pass():
    """A null `fact_entity_recall_avg` is what a run whose fact leg read nothing
    looks like. A gate that read null as "no fall recorded" would certify an empty
    tree — the same failure `run_eval.py`'s over-reach guard already refuses for
    the nightly chain, for the same reason."""
    ok, reasons = rf.gate_verdict(_stub_recall(avg=None), _stub_recall(avg=None))
    assert not ok and any("not measured" in r for r in reasons), reasons
    assert not rf.gate_verdict(_stub_recall(avg=0.4), _stub_recall(avg=None))[0]
    assert not rf.gate_verdict(_stub_recall(avg=0.4), _stub_recall(errors=1))[0]


def test_gate_passes_a_steady_average_with_the_same_entities():
    ok, reasons = rf.gate_verdict(_stub_recall(), _stub_recall())
    assert ok, reasons


def test_run_recall_points_the_eval_at_the_copy_via_the_environment(tmp_path, monkeypatch):
    """The process boundary this gate crosses. `run_eval` resolves its fact root
    and store at import, so pointing it at a copy means a child process with
    LLOYD_FACTS_ROOT/LLOYD_KG_DB in its environment — not a second
    `kg_store.configure()` in here, which would retarget this process's own
    readers, including the caller's before/after measurement."""
    seen = {}

    class Proc:
        returncode = 0
        stdout = json.dumps({"fact_entity_recall_avg": 0.5, "errors": 0,
                             "corpus": {}, "records": []}) + "\n"
        stderr = ""

    def fake_run(cmd, **kw):
        seen.update(cmd=cmd, env=kw["env"], cwd=kw["cwd"])
        return Proc()

    monkeypatch.setattr(rf.subprocess, "run", fake_run)
    harness = tmp_path / "harness"
    (harness / "eval").mkdir(parents=True)
    (harness / "eval" / "run_eval.py").write_text("# stands in for the real harness\n",
                                                  encoding="utf-8")
    out = rf.run_recall(harness, tmp_path / "copy", tmp_path / "k.sqlite", 20)

    assert out["fact_entity_recall_avg"] == 0.5
    assert seen["env"]["LLOYD_FACTS_ROOT"] == str(tmp_path / "copy")
    assert seen["env"]["LLOYD_KG_DB"] == str(tmp_path / "k.sqlite")
    assert seen["cwd"] == str(tmp_path / "harness")
    # Pinned exactly, not by substring: `"eval" in cmd[2] + cmd[3]` was true of any
    # snippet that merely mentioned eval and of any argv order, so it could not
    # tell the harness checkout from the corpus tree — which is the argument the
    # live-run defect sat on. The two roots are separate arguments, and only
    # `test_the_harness_root_is_the_checkout_that_holds_the_eval` and the real
    # subprocess test below can fail.
    assert seen["cmd"][1] == "-c", seen["cmd"][:2]
    assert seen["cmd"][2] == rf._RECALL_SNIPPET, "the child must run this script's snippet"
    assert seen["cmd"][3:] == [str(tmp_path / "harness"), "20"], \
        f"argv must be <harness root>, then <limit>, got {seen['cmd'][3:]}"
    assert seen["cmd"][0] == sys.executable, "the child must be this interpreter"


# ── clause 5, the process boundary itself (unstubbed) ────────────────────────
#
# Everything above this line stubs the comparison, which is right for grading the
# gate's decisions and wrong for grading the boundary: a stubbed `subprocess.run`
# cannot see the two ways the seam breaks — the child resolves the built-in corpus
# instead of the override it was handed, or binds a `run_eval` other than the one
# at `<harness root>/eval/run_eval.py`. Both are caught below by really spawning
# it, with a stand-in for the *scoring* module only.

_STUB_HARNESS = '''"""Stand-in for `eval/run_eval.py`, used only by the seam tests.

The seam is not the scoring — that is run_eval's own tested surface — it is that a
child launched by `run_recall` loads THIS module by path and reads the fact tree
and store named by LLOYD_FACTS_ROOT / LLOYD_KG_DB. So every number here is derived
from what those two paths hold: the mean is a recall over the entities the copy's
store actually has, the per-query match set is that intersection too, and the
corpus report counts files carrying a marker that only the copy has. A child that
resolved the built-in paths would answer with the live corpus's numbers, and a
child that bound a different run_eval would not report this file's path.
"""
import os
import sqlite3
import sys
from pathlib import Path

REPO = "__LLOYD_REPO__"
MARK = "seam marker"


def _resolved_by_app_paths():
    """What the real `app.paths` makes of the two overrides, seen from the child."""
    sys.path.insert(0, REPO)
    from app import paths
    return {"facts_root": str(paths.VAULT_FACTS_ROOT), "kg_db": str(paths.VAULT_KG_DB)}


SEEN = {}          # what the run read, forwarded to the parent as `corpus`


def run_eval(queries, limit=20, expand_graph=True, counterfactual=False, **kw):
    facts_root = Path(os.environ["LLOYD_FACTS_ROOT"])
    db = Path(os.environ["LLOYD_KG_DB"])
    con = sqlite3.connect(str(db))
    try:
        rows = con.execute("SELECT COUNT(*) FROM facts_idx").fetchone()[0]
        held = [r[0] for r in con.execute("SELECT DISTINCT entity FROM facts_idx")]
    finally:
        con.close()
    SEEN.update(rows=rows, held=sorted(held),
                marks=sum(1 for p in sorted(facts_root.rglob("*.md"))
                          if MARK in p.read_text(encoding="utf-8")))
    out = []
    for q in list(queries)[:limit]:
        want = list(q.get("expect_entities") or [])   # the real query file's key
        got = [e for e in want if any(e.lower() == h.lower() for h in held)]
        out.append({"query": q["query"], "category": q.get("category"),
                    "scoring": {"fact_entities_matched": got}})
    SEEN["queries"] = len(out)
    return out


def summarize(records):
    n = len(records) or 1
    recall = sum(len(r["scoring"]["fact_entities_matched"]) for r in records) / (n or 1)
    return {"overall": {"fact_entity_recall_avg": min(recall, 1.0), "errors": 0}}


def _corpus_provenance():
    return {"stub": True, "harness": str(Path(__file__).resolve()),
            "resolved": _resolved_by_app_paths(), **SEEN}
'''

_SEAM_QUERIES = """queries:
  - id: seam-lloyd
    query: who is lloyd
    category: single
    expect_entities: [Lloyd]
  - id: seam-vllm
    query: what serves the bench model
    category: single
    expect_entities: [Lloyd]
"""


def seam_harness(tmp_path: Path) -> Path:
    """A checkout holding `eval/run_eval.py` + the query file the child reads.

    Only the scoring module is stood in; the interpreter, the spawn, the
    environment and the module-loading-by-path are the real ones.
    """
    root = tmp_path / "harness"
    (root / "eval").mkdir(parents=True)
    (root / "eval" / "run_eval.py").write_text(
        _STUB_HARNESS.replace("__LLOYD_REPO__", str(ROOT)), encoding="utf-8")
    (root / "eval" / "vault_recall_queries.yaml").write_text(_SEAM_QUERIES, encoding="utf-8")
    return root


def seam_copy(tmp_path: Path, name: str, entity: str,
              n_facts: int) -> tuple[Path, Path, int]:
    """One fact tree plus its own index; `entity` is what the child can match there."""
    facts, db = tmp_path / f"copy-{name}", tmp_path / f"{name}.sqlite"
    kg_store.configure(db)
    # `entity=` per fact as well as in the folder: the indexer reads a per-fact
    # `entity:` override before the file's tag (app/kg_store.py `_rows_for_file`),
    # so leaving the helper's default here would index the second copy's rows
    # under the first copy's name and the child would answer correctly for the
    # wrong reason.
    entries = [fact(f"seam marker {name} claim {i}", id=f"{name[:4]}-{i:03d}",
                    entity=entity) for i in range(n_facts)]
    write_tree(facts, entity, "state", entries)
    kg_store.store().facts_idx.reindex(root=facts)
    rows = kg_store.store()._query("SELECT COUNT(*) n FROM facts_idx")[0]["n"]
    kg_store.reset()
    return facts, db, rows


def test_run_recall_spawns_a_real_child_that_scores_the_copy_it_is_handed(
        tmp_path, monkeypatch):
    """Clause 5's boundary, crossed for real. `run_eval` resolves its fact root and
    store at import, so the only way to score a copy is a child process carrying
    LLOYD_FACTS_ROOT/LLOYD_KG_DB — and the only proof that carries, at this level,
    is a child that reads them and answers with the copy's own numbers."""
    root = seam_harness(tmp_path)
    facts_a, db_a, rows_a = seam_copy(tmp_path, "alpha", "Lloyd", 3)
    facts_b, db_b, rows_b = seam_copy(tmp_path, "bravo", "Gollum", 7)
    assert (rows_a, rows_b) == (3, 7), "the two copies must be distinguishable"

    out = rf.run_recall(root, facts_a, db_a, 20)

    # Which module the child bound: its own `__file__`, reported from inside the child.
    assert out["corpus"]["harness"] == str((root / "eval" / "run_eval.py").resolve()), \
        "the child loaded a run_eval other than the harness checkout's"
    # The env names are the ones the real resolver honours, not merely present: this
    # is `app.paths` answering from inside the child, not the parent echoing strings.
    assert out["corpus"]["resolved"] == {"facts_root": str(facts_a), "kg_db": str(db_a)}, \
        "app.paths did not resolve the copy from the environment the child got"
    # Both legs are the copy's. The store leg: only the entity that copy's index
    # holds. The file leg: `marks` counts a marker no other tree carries — without
    # it, a child reading the copy's store and the live markdown would score a
    # corpus that half-existed.
    assert out["corpus"]["rows"] == rows_a, out["corpus"]
    assert out["corpus"]["marks"] == 1, "the child read a fact tree that was not the copy"
    assert out["corpus"]["held"] == ["Lloyd"], out["corpus"]
    # The two numbers clause 5's gate reads, each computed by the child from the
    # copy: the mean, and the per-query expected-entity match set.
    assert out["fact_entity_recall_avg"] == pytest.approx(1.0), out
    assert [r["fact_entities_matched"] for r in out["records"]] == [["Lloyd"], ["Lloyd"]], out
    assert out["errors"] == 0 and {"corpus", "records"} <= set(out), out

    other = rf.run_recall(root, facts_b, db_b, 20)
    assert other["corpus"]["held"] == ["Gollum"] and other["corpus"]["marks"] == 1, \
        other["corpus"]
    assert other["fact_entity_recall_avg"] == pytest.approx(0.0), other
    assert [r["fact_entities_matched"] for r in other["records"]] == [[], []], other
    # The pair is what the gate compares, so the comparison now runs on
    # child-produced payloads and not only on the in-process stub. `_entity_losses`
    # prefers the harness's own judgement; pinned to the fallback here so the test
    # does not import the real harness into the test process for a pass/fail that
    # both branches answer the same way.
    monkeypatch.setitem(sys.modules, "eval", types.SimpleNamespace(run_eval=None))
    ok, reasons = rf.gate_verdict(out, other)
    assert not ok and any("lost an expected entity" in r for r in reasons), reasons


def test_a_child_that_dies_is_a_failure_and_not_a_zero_score(tmp_path):
    """The failure half of the boundary. An unreachable eval engine must surface as
    an error naming the child's own output; read as an arm with a null or zero
    average it would either stop the repair with a misleading reason or, worse, be
    argued past."""
    root = seam_harness(tmp_path)
    facts, db, _rows = seam_copy(tmp_path, "charlie", "Lloyd", 2)
    (root / "eval" / "run_eval.py").write_text(
        "def run_eval(*a, **k):\n    raise RuntimeError('eval engine unreachable')\n"
        "def summarize(recs):\n    return {'overall': {}}\n", encoding="utf-8")

    with pytest.raises(RuntimeError) as got:
        rf.run_recall(root, facts, db, 20)

    msg = str(got.value)
    assert "exit" in msg and "eval engine unreachable" in msg, msg


def test_the_child_loads_the_harness_named_in_its_argv(tmp_path):
    """Which checkout's scoring code runs is decided by one argument, so the
    argument is tested by making it decide something. Two harness roots, two
    different answers, one corpus: if the root were derived from the corpus, or
    from `sys.path`, or from whatever `import eval` finds first, one of these two
    readings would be the other's.

    This is the property the live-run defect sat on. `eval/run_eval.py` resolves
    `LLOYD_HOME` from its own `__file__`, so the file that gets loaded also decides
    which tree the harness itself considers home."""
    facts, db, _rows = seam_copy(tmp_path, "alpha", "Lloyd", 3)
    bodies = {"first": "harness-one", "second": "harness-two"}
    for name, marker in bodies.items():
        h = tmp_path / name
        (h / "eval").mkdir(parents=True)
        (h / "eval" / "run_eval.py").write_text(
            "def run_eval(queries, **kw):\n"
            "    return [{'query': q['query'], 'category': q.get('category'),\n"
            "            'scoring': {'fact_entities_matched': []}} for q in queries]\n"
            "def summarize(recs):\n"
            "    return {'overall': {'fact_entity_recall_avg': 0.5, 'errors': 0}}\n"
            "def _corpus_provenance():\n"
            "    return {'marker': " + repr(marker) + "}\n", encoding="utf-8")
        (h / "eval" / "vault_recall_queries.yaml").write_text(_SEAM_QUERIES, encoding="utf-8")

    for name, marker in bodies.items():
        out = rf.run_recall(tmp_path / name, facts, db, 20)
        assert out["corpus"]["marker"] == marker, \
            f"asked for {name}, the child scored with somebody else's harness"
        assert out["corpus"]["harness"] == str(
            (tmp_path / name / "eval" / "run_eval.py").resolve()), out["corpus"]
        # `bound` is the module object's own answer, reported beside the request so a
        # run record shows the child's evidence and not only the argument it was fed.
        assert out["corpus"]["bound"] == out["corpus"]["harness"], out["corpus"]
        assert out["fact_entity_recall_avg"] == pytest.approx(0.5), out


def test_run_recall_refuses_a_root_that_holds_no_eval_harness(tmp_path):
    """A root with no `eval/run_eval.py` is a misrooted call, not a slow run: the
    child would die importing a file that is not there. It matters because the
    pre-approved run is the live tree, whose fact root is the derived corpus — a
    directory that holds no harness at all."""
    with pytest.raises(FileNotFoundError) as got:
        rf.run_recall(tmp_path / "no-harness-here", tmp_path / "copy",
                      tmp_path / "k.sqlite", 20)
    assert "eval/run_eval.py" in str(got.value), got.value


def test_the_harness_root_is_the_checkout_that_holds_the_eval(tree, monkeypatch, tmp_path):
    """The corpus and the harness are two arguments, and only a test on `main` can
    tell them apart. Repairing the live tree is the run Alan pre-approved, and
    there `--facts-dir` *is* the derived corpus: pass that as the harness root and
    the first gate call dies importing `<facts>/eval/run_eval.py`, which does not
    exist — the gate fails on the one path it exists to serve."""
    seen: list[tuple[Path, Path]] = []
    monkeypatch.setattr(rf, "run_recall",
                        lambda root, facts, db, n: seen.append((root, facts)) or _stub_recall())
    live = tmp_path / "vault-derived" / "facts"          # stands in for the default root
    db = tmp_path / "live.sqlite"
    write_tree(live, "Lloyd", "state", [fact("runs on vLLM", id="stat-001")])
    write_tree(live, "Lloyd", "relationship",
               [fact("runs on vLLM", id="rela-001", category="relationship",
                     confidence=0.3)])
    index(live, db)
    monkeypatch.setattr(rf, "VAULT_FACTS_ROOT_DEFAULT", live)
    assert live.resolve() == Path(rf.VAULT_FACTS_ROOT_DEFAULT).resolve()

    rc = cli(live, db, tmp_path / "m.jsonl", "--apply", "--recall-compare")

    assert rc == 0, rc
    assert len(seen) == 2, f"expected a before and an after arm, got {len(seen)}"
    for harness, corpus in seen:
        assert (harness / "eval" / "run_eval.py").is_file(), \
            f"the child was told to load its harness from {harness}, which has none"
        assert corpus == live, "the corpus argument stopped being the tree under repair"


def test_the_snippet_calls_names_the_real_harness_still_exports(tmp_path):
    """The stub above keeps the seam testable without an eval engine, which also
    means it cannot notice `eval/run_eval.py` renaming `summarize` or dropping the
    `expand_graph` kwarg — the stub would still answer. Read the real module's
    signature off its AST (no import: it pulls the retrieval stack) so a rename
    fails here instead of at a live `--apply` part-way through a repair."""
    src = (ROOT / "eval" / "run_eval.py").read_text(encoding="utf-8")
    import ast

    tree = ast.parse(src)
    params: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            params[node.name] = ({a.arg for a in node.args.args}
                                 | {a.arg for a in node.args.kwonlyargs})
    for name in ("run_eval", "summarize", "_corpus_provenance", "count_overreach_regressions"):
        assert name in params, f"eval/run_eval.py no longer defines {name}()"
    assert {"limit", "expand_graph", "counterfactual"} <= params["run_eval"], \
        "run_recall's snippet passes kwargs run_eval no longer accepts"
    assert next(iter(
        a.arg for a in next(n for n in tree.body
                            if isinstance(n, ast.FunctionDef) and n.name == "run_eval"
                            ).args.args)) == "queries"
    # The snippet must still be a caller of that harness, not a second copy of it:
    # two spellings of the scoring is how two sides disagree about an arm's verdict.
    assert "run_eval.py" in rf._RECALL_SNIPPET and "run_eval" in rf._RECALL_SNIPPET
    assert "def summarize" not in rf._RECALL_SNIPPET, "the snippet re-implements the harness"


def test_a_file_whose_bytes_moved_since_the_plan_is_refused_not_cut(tree):
    """The survey runs unlocked, so the bytes it numbered may not be the bytes on
    disk by the time the lock is held — `repair_fact_ids.py` re-reads under the
    lock for the same reason. A moved file is refused rather than cut on line
    numbers that may now name a different fact, and the manifest is fsynced before
    the file moves, so an edit never outruns its own record."""
    facts, _db = tree
    path = write_tree(facts, "Lloyd", "state", [fact("twin", id="stat-001"),
                                                fact("twin", id="stat-002",
                                                     confidence=0.4)])
    planned = rf.plan_removals(facts)
    assert len(planned["removals"]) == 1

    path.write_text(path.read_text(encoding="utf-8").replace(
        "fact: twin", "fact: twin edited by another writer"), encoding="utf-8")
    manifest = facts / "m.jsonl"
    res = rf.apply_removals(planned["plans"], planned["removals_by_file"], manifest,
                            apply=True)

    assert [r["file"] for r in res["refused"]] == ["Lloyd/Lloyd-state.md"], res
    assert res["removed_rows"] == 0
    assert "edited by another writer" in path.read_text(encoding="utf-8")
    assert not manifest_lines(manifest), "a refused file must not appear in the manifest"
