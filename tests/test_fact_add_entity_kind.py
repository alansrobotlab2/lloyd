"""#758 — `fact_add` mints entities through the declared-identity gate.

The MCP write path registered a new entity with no kind, then registered it
a second time from the reindex of its fact directory, also with no kind. That
was where nearly every `kind IS NULL` row in the store came from: 25 of the 33
created between 09-16 and 09-18 had a founding fact file in `_fact_add`'s body
format. The extractor has gone through `gate_entity_name` since #537; this
pins that `fact_add` does too, without the extractor's refusal of an untyped
name — a caller naming a new thing is declaring it.
"""
import sys
from pathlib import Path

import pytest

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
