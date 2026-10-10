"""#758 — `fact_add` mints entities through the declared-identity gate.

The MCP write path registered a new entity with no kind, then registered it
a second time from the reindex of its fact directory, also with no kind. That
was where nearly every `kind IS NULL` row in the store came from: 25 of the 33
created between 09-16 and 09-18 had a founding fact file in `_fact_add`'s body
format. The extractor has gone through `gate_entity_name` since #537; this
pins that `fact_add` does too, without the extractor's refusal of an untyped
name — a caller naming a new thing is declaring it.

#2503 is a second defect in the same writer, on the OTHER half of the row: the
record dict it appended carried `fact`, `confidence`, `category`, `id`, the four
timestamps, `provenance` and `source_doc`, and no `entity`. `app/kg_store.py`
indexes the row under `fm.get("entity") or path.parent.name`, so the store knew
the entity and nobody noticed; `scripts/memory/kg_hygiene.py`'s `_tag_of` reads
the PER-RECORD key, and every row this function wrote counted as untagged —
463 of them since 2026-10-02, 457 carrying the `session_id` stamp only this
writer adds. That is the population #2474's rail has been alarming nightly.
The pins below read the MARKDOWN row back, because `facts_idx` cannot show this
gap: it is rebuilt from these files and falls back to the directory name, so a
row with no `entity` key looks identical there.
"""
import asyncio
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_mcp import _shared, facts, retrieval  # noqa: E402
from app import entity_naming, kg_store  # noqa: E402


@pytest.fixture
def tree(tmp_path, monkeypatch):
    root = tmp_path / "facts"
    root.mkdir()
    kg_store.configure(tmp_path / "kg.sqlite")
    # The gate installs the declared keys into the first store it sees in a
    # process; reset so this store is the one that gets them.
    entity_naming.reset_identity_schema_cache()
    monkeypatch.setattr(_shared, "FACTS_ROOT", root)
    monkeypatch.setattr(_shared, "ALIASES_PATH", root / "entity-aliases.json")
    monkeypatch.setattr(_shared, "_entity_dirs_cache", None)
    monkeypatch.setattr(facts, "FACTS_ROOT", root)
    monkeypatch.setattr(retrieval, "FACTS_ROOT", root)
    yield root
    entity_naming.reset_identity_schema_cache()
    kg_store.reset()


def _add(entity, fact="runs on the tailnet", **extra):
    return facts._fact_add({"entity": entity, "category": "state", "fact": fact, **extra})


def _row(name):
    return next((r for r in kg_store.store().entities.rows() if r["name"] == name), None)


def _null_kind_rows():
    return [r["name"] for r in kg_store.store().entities.rows() if r.get("kind") is None]


def test_a_brand_new_name_is_registered_with_a_kind(tree):
    out = _add("Zedlink Relay")
    assert out.get("success") is True, out
    assert (tree / "Zedlink Relay" / "Zedlink Relay-state.md").exists()
    row = _row("Zedlink Relay")
    assert row is not None and row["kind"] is not None
    assert _null_kind_rows() == []


def test_a_directory_the_store_has_not_indexed_yet_gets_a_kind_too(tree):
    # The registry lags the tree: the old path registered this untyped.
    (tree / "Quillgate").mkdir()
    assert _add("Quillgate").get("success") is True
    assert _row("Quillgate")["kind"] is not None
    assert _null_kind_rows() == []


def test_a_declared_entity_type_becomes_the_kind(tree):
    out = _add("Nightjar Sync", entity_type="Pipelines")
    assert out.get("success") is True, out
    assert _row("Nightjar Sync")["kind"] == "pipeline"


def test_an_undeclared_entity_type_is_refused_before_any_write(tree):
    out = _add("Banana Stand", entity_type="banana")
    assert out.get("code") == "INVALID_PARAM", out
    assert not out.get("success")
    assert not (tree / "Banana Stand").exists()
    assert _row("Banana Stand") is None


def test_a_declared_alias_files_under_its_canonical(tree):
    out = _add("The Graph", fact="holds typed edges between entities")
    assert out.get("success") is True, out
    assert out["entity"] == "Knowledge Graph"
    assert out["resolved_from"] == "The Graph"
    assert (tree / "Knowledge Graph" / "Knowledge Graph-state.md").exists()
    assert not (tree / "The Graph").exists()
    assert _row("The Graph") is None
    assert _row("Knowledge Graph")["kind"] == "system"


def test_the_reindex_after_a_write_registers_nothing(tree):
    # The second mint: `update_file` registered the directory name untyped.
    d = tree / "Loose Dir"
    d.mkdir()
    (d / "Loose Dir-state.md").write_text(
        "---\ntype: facts\nentity: Loose Dir\ncategory: state\nfacts: []\n---\n")
    st = kg_store.store()
    st.facts_idx.update_file(d / "Loose Dir-state.md", root=tree, register_entities=False)
    assert _row("Loose Dir") is None


# ---------------------------------------------------------------------------
# #1743 — a machine-derived fact has to name the document it came from
#
# The session-distill prompt told its turn to call `fact_add` and never mentioned
# `source_doc`, the handler wrote the field through as `None`, and 9 rows reached
# `facts_idx` with it NULL. A fact with no source cannot be attributed to the run
# that wrote it or selectively reverted, which is exactly the property
# `kg_hygiene.provenance_coverage` gates a rebuild on at 100%. The refusal is
# scoped to the two provenances that declare *a pass, not a person, produced
# this*: `STATED` is the default and a chat turn states facts with no document
# behind them, so requiring a source there would refuse ordinary conversation
# writes across the whole tool surface.
# ---------------------------------------------------------------------------

def test_an_extracted_fact_with_no_source_doc_is_refused(tree):
    out = _add("Tidewell Relay", provenance="EXTRACTED")
    assert out.get("code") == "MISSING_PARAM", out
    assert not out.get("success")
    assert "source_doc" in out.get("error", ""), (
        "the refusal has to name the field the caller left out")
    assert not (tree / "Tidewell Relay").exists(), (
        "a refusal must not leave a minted entity directory behind")


def test_an_inferred_fact_with_no_source_doc_is_refused(tree):
    out = _add("Tidewell Relay", provenance="INFERRED")
    assert out.get("code") == "MISSING_PARAM", out
    assert not out.get("success")
    assert not (tree / "Tidewell Relay").exists()


def test_the_refusal_reaches_the_turn_through_the_mcp_wrapper(tree):
    """The boundary the leak actually crossed: a distill turn calls this tool over
    MCP, so the refusal has to arrive as an MCP error the turn can read rather
    than as a successful-looking payload whose text happens to mention `error`.
    Testing only the handler would leave the wrapper unexercised."""
    wrapped = asyncio.run(facts.call_tool(
        "fact_add", {"entity": "Tidewell Relay", "category": "state",
                     "fact": "pins its control port at 9453",
                     "provenance": "EXTRACTED"}))
    assert wrapped.is_error is True
    payload = json.loads(wrapped.content[0].text)
    assert payload["code"] == "MISSING_PARAM", payload
    assert "source_doc" in payload["error"], payload
    assert not (tree / "Tidewell Relay").exists()


def test_an_extracted_fact_with_a_blank_source_doc_is_refused(tree):
    # Whitespace is the shape a templated path takes when the value under it is
    # empty, so a field that is present but says nothing cannot satisfy the rule.
    out = _add("Tidewell Relay", provenance="EXTRACTED", source_doc="   ")
    assert out.get("code") == "MISSING_PARAM", out
    assert not (tree / "Tidewell Relay").exists()


def test_an_extracted_fact_with_a_source_doc_still_writes(tree):
    out = _add("Tidewell Relay", provenance="EXTRACTED",
               source_doc="sessions/20260928_005437_distill.json")
    assert out.get("success") is True, out
    rows = kg_store.store().facts_idx.for_entity("Tidewell Relay")
    assert [r["source_doc"] for r in rows] == ["sessions/20260928_005437_distill.json"]


def test_an_unsourced_stated_fact_still_writes(tree):
    """The exemption that keeps this a scoped rule rather than a tool-wide one:
    a chat turn states a fact with no document behind it, and that stays
    writable — default provenance, no `source_doc`, still indexed."""
    out = _add("Tidewell Relay")
    assert out.get("success") is True, out
    rows = kg_store.store().facts_idx.for_entity("Tidewell Relay")
    assert [r["source_doc"] for r in rows] == [None]


# ---------------------------------------------------------------------------
# #2503 — the row names its own entity, not just its directory
#
# `app/kg_store.py::_rows_for_file` indexes a fact under
# `fm.get("entity") or path.parent.name`, and the per-fact override
# `f.get("entity") or entity` sits on top of that — so for eleven months the
# store reported these rows as belonging to their directory and the missing key
# was invisible from the query side. `kg_hygiene._tag_of` is the reader that
# cannot fall back: it returns '' for a missing key as its own answer, because
# "extraction never wrote the field" and "it wrote the field blank" are the two
# shapes of an unattributable record, and that '' is what the #2474 rail counts
# and a merge audit greps. Hence: every assertion here opens the .md file.
# ---------------------------------------------------------------------------

_FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def _markdown_records(path: Path) -> list[dict]:
    """The fact records as the markdown file holds them, parsed the way the
    readers parse them — deliberately NOT `facts_idx.for_entity`, which infers
    the entity from the directory and so cannot show this defect at all."""
    m = _FM_RE.match(path.read_text(encoding="utf-8"))
    assert m, f"{path} carries no YAML front-matter block"
    fm = yaml.safe_load(m.group(1))
    records = fm.get("facts") or []
    assert records, f"{path} holds no fact records"
    return records


def _file(root: Path, entity: str, category: str = "state") -> Path:
    """The fact file `_fact_add` files `entity`'s `category` rows in."""
    return root / entity / f"{entity}-{category}.md"


def test_the_written_row_carries_the_entity_it_was_filed_under(tree):
    """Clause 1: the tag equals the entity name the row was filed under, and it is
    on the markdown row."""
    out = _add("Tidewell Relay", fact="pins its control port at 9453")
    assert out.get("success") is True, out
    records = _markdown_records(_file(tree, "Tidewell Relay"))
    assert len(records) == 1, records
    assert records[0].get("entity") == "Tidewell Relay", records[0]
    # The exact key `kg_hygiene._tag_of` reads, read the way it reads it: a value
    # that is present but blank is as untagged as an absent one.
    assert str(records[0]["entity"]).strip() == "Tidewell Relay"


def test_the_tag_survives_a_second_write_to_the_same_file(tree):
    """The tag is on every row, not only on the first one a fresh file gets.

    `_fact_add` read-modify-writes the whole front matter under a lock, so a row
    appended to an existing file travels back out through the same dump; the live
    corpus's own shape is a file whose every record repeats the entity, and the
    rail counts records, so one tagged row in a hundred does not move it."""
    assert _add("Tidewell Relay", fact="pins its control port at 9453").get("success")
    assert _add("Tidewell Relay",
                fact="drains the queue before a restart").get("success")
    records = _markdown_records(_file(tree, "Tidewell Relay"))
    assert len(records) == 2, records
    assert [r.get("entity") for r in records] == ["Tidewell Relay", "Tidewell Relay"], records


def test_an_alias_declaration_tags_the_row_with_the_canonical_name(tree):
    """The name filed under, not the name asked for.

    `test_a_declared_alias_files_under_its_canonical` shows the DIRECTORY is
    `Knowledge Graph` when the caller says `The Graph`; the rail's own comparison
    is `_same_entity(tag, d.name)`, so a row tagged with the caller's surface
    spelling would read as a fact about some other entity — contamination, the
    worse of the two failures, since `kg_rebuild` refuses a rebuild on it."""
    out = _add("The Graph", fact="holds typed edges between entities")
    assert out.get("success") is True, out
    assert out["entity"] == "Knowledge Graph", out
    records = _markdown_records(_file(tree, "Knowledge Graph"))
    assert [r.get("entity") for r in records] == ["Knowledge Graph"], records


def test_the_index_cannot_be_the_witness(tree):
    """Why every assertion above opens the file instead of querying the store.

    Hand the indexer a row with no `entity` key at all — the bytes `_fact_add`
    wrote until #2503 — and `facts_idx` still files it under the entity, because
    `app/kg_store.py::_rows_for_file` falls back to `path.parent.name`. A test that
    read the index would therefore have passed before this fix and passes after
    it, which is how the writer fed an alarm nobody could trace back to it."""
    assert _add("Tidewell Relay").get("success")
    path = _file(tree, "Tidewell Relay")
    stripped = [{k: v for k, v in r.items() if k != "entity"}
                for r in _markdown_records(path)]
    assert len(stripped) == 1 and all("entity" not in r for r in stripped)
    path.write_text(
        "---\n" + yaml.dump({"type": "facts", "entity": "Tidewell Relay",
                             "category": "state", "facts": stripped},
                            default_flow_style=False, sort_keys=False) + "---\n",
        encoding="utf-8")
    assert "entity" not in _markdown_records(path)[0], "the rewrite did not strip the key"

    st = kg_store.store()
    st.facts_idx.update_file(path, root=tree, register_entities=False)
    rows = st.facts_idx.for_entity("Tidewell Relay", include_expired=True)
    assert len(rows) == 1, (
        "the store did not index the key-less row, so this node's premise is wrong: "
        "it is that fallback which hides the defect, and without it the markdown "
        "reads above would not be the load-bearing assertion")
