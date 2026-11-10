"""The fact layer's read and read-modify-write paths (2026-11-10 review).

Four defects, each one a way knowledge already on disk stopped reaching the
model or a write reported a success it had not earned:

- `_parse_fact_frontmatter` closed the front matter at the first `---`
  ANYWHERE, so a fact whose text held `---` cut every later fact out of the
  read, and the writers wrote the truncated list back.
- `_reindex_files` swallowed `StoreUnavailable`, so `fact_invalidate` and
  `fact_resolve_apply` reported a clean success over a stale index.
- `fact_get(as_of=...)` compared a YAML-parsed `datetime.date` with a string
  and raised, failing the whole read.
- the rebuild write-freeze covered `fact_add` and `fact_relate` but not the
  two mark writers.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agent_mcp._shared as shared            # noqa: E402
from agent_mcp import facts as facts_mod      # noqa: E402
from agent_mcp import retrieval               # noqa: E402
from app import kg_store                      # noqa: E402
from app.kg_store import StoreUnavailable     # noqa: E402


@pytest.fixture
def tree(tmp_path, monkeypatch):
    facts_root = tmp_path / "facts"
    facts_root.mkdir()
    monkeypatch.setattr(shared, "FACTS_ROOT", facts_root)
    monkeypatch.setattr(retrieval, "FACTS_ROOT", facts_root)
    monkeypatch.setattr(facts_mod, "FACTS_ROOT", facts_root)
    shared._invalidate_entity_dirs_cache()
    retrieval.invalidate_fact_file_cache()
    retrieval._entity_index_cache = None
    st = kg_store.configure(tmp_path / "kg.sqlite")
    yield facts_root, st
    kg_store.reset()
    shared._invalidate_entity_dirs_cache()
    retrieval.invalidate_fact_file_cache()


def _write(root: Path, entity: str, category: str, facts: list[dict],
           raw_valid_at: str | None = None) -> Path:
    d = root / entity
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"fact": f["fact"], "confidence": f.get("confidence", 0.9),
             "category": category, "id": f["id"], "created_at": "2026-10-01T00:00:00+00:00",
             "valid_at": f.get("valid_at"), "expired_at": None, "invalid_at": None,
             "provenance": "STATED", "source_doc": None} for f in facts]
    text = yaml.dump({"type": "facts", "entity": entity, "category": category,
                      "facts": rows}, sort_keys=False)
    if raw_valid_at:
        # An UNQUOTED date, the way 82 live files carry one: YAML reads it
        # back as a `datetime.date`, not a string.
        text = text.replace("valid_at: null", f"valid_at: {raw_valid_at}", 1)
    path = d / f"{entity}-{category}.md"
    path.write_text(f"---\n{text}---\n\n# {entity}\n", encoding="utf-8")
    return path


# ── the fence ────────────────────────────────────────────────────────────────

def test_a_fact_containing_dashes_does_not_cut_the_read():
    text = shared._write_fact_frontmatter({"facts": [
        {"fact": "a --- b"}, {"fact": "line1\n---\nline2"}, {"fact": "later"}]})
    parsed = shared._parse_fact_frontmatter(text + "\n# body\n")
    assert [f["fact"] for f in parsed["facts"]] == ["a --- b", "line1\n---\nline2", "later"]


def test_split_keeps_the_body_and_round_trips_without_growing():
    fm_text = shared._write_fact_frontmatter({"facts": [{"fact": "x"}]})
    content = fm_text + "\n# Entity\n"
    yaml_text, body = shared._split_fact_frontmatter(content)
    assert yaml.safe_load(yaml_text) == {"facts": [{"fact": "x"}]}
    assert body == "\n# Entity\n"
    assert shared._write_fact_frontmatter(yaml.safe_load(yaml_text)) + body == content


@pytest.mark.parametrize("content", ["no front matter", "---\nfoo: bar\nno close",
                                     "----\nfoo: 1\n---\n"])
def test_malformed_front_matter_still_reads_empty(content):
    assert shared._parse_fact_frontmatter(content) == {}


def test_fact_add_after_a_dashed_fact_keeps_every_earlier_fact(tree):
    root, _ = tree
    path = _write(root, "Widget", "state", [
        {"fact": "The cut is between v1 --- v2.", "id": "fact-001"},
        {"fact": "Widget runs on port 9000.", "id": "fact-002"},
    ])
    out = facts_mod._fact_add({"entity": "Widget", "category": "state",
                               "fact": "Widget is written in Rust.", "write_gate": "off"})
    assert out.get("success"), out
    facts = shared._parse_fact_frontmatter(path.read_text())["facts"]
    assert [f["fact"] for f in facts] == [
        "The cut is between v1 --- v2.", "Widget runs on port 9000.",
        "Widget is written in Rust."]


def test_fact_invalidate_after_a_dashed_fact_keeps_the_rest(tree):
    root, _ = tree
    path = _write(root, "Widget", "state", [
        {"fact": "Split at a --- b.", "id": "fact-001"},
        {"fact": "Widget is beta.", "id": "fact-002"},
        {"fact": "Widget ships Fridays.", "id": "fact-003"},
    ])
    out = facts_mod._fact_invalidate({"entity": "Widget", "fact_substring": "beta"})
    assert out["expired_count"] == 1, out
    facts = shared._parse_fact_frontmatter(path.read_text())["facts"]
    assert len(facts) == 3
    assert [bool(f.get("expired_at")) for f in facts] == [False, True, False]
    assert path.read_text().endswith("\n# Widget\n")


# ── the index warning ────────────────────────────────────────────────────────

def test_reindex_failure_is_a_warning_not_a_silent_success(tree, monkeypatch):
    root, st = tree
    _write(root, "Widget", "state", [{"fact": "Widget is beta.", "id": "fact-001"}])

    class _Broken:
        def reindex(self, *a, **k):
            raise StoreUnavailable("disk gone")

    class _St:
        facts_idx = _Broken()

    monkeypatch.setattr(facts_mod, "_store", lambda: _St())
    out = facts_mod._fact_invalidate({"entity": "Widget", "fact_substring": "beta"})
    assert out["success"] and out["expired_count"] == 1
    assert "index not updated" in out["warning"]
    assert "kg reindex" not in out["warning"]


# ── as_of over date-typed fields ─────────────────────────────────────────────

def test_as_of_handles_a_yaml_date(tree):
    root, _ = tree
    _write(root, "Widget", "state", [{"fact": "Widget launched.", "id": "fact-001"}],
           raw_valid_at="2026-03-01")
    fm = shared._parse_fact_frontmatter(
        (root / "Widget" / "Widget-state.md").read_text())
    assert isinstance(fm["facts"][0]["valid_at"], datetime.date)
    before = retrieval.get_facts_sync("Widget", as_of="2026-02-01")
    after = retrieval.get_facts_sync("Widget", as_of="2026-04-01")
    assert before["facts"] == []
    assert [f["fact"] for f in after["facts"]] == ["Widget launched."]


# ── the rebuild freeze ───────────────────────────────────────────────────────

@pytest.mark.parametrize("handler,params", [
    ("_fact_invalidate", {"entity": "Widget", "fact_substring": "beta"}),
    ("_fact_resolve_apply", {"entity": "Widget"}),
])
def test_mark_writers_honour_the_rebuild_freeze(tree, monkeypatch, handler, params):
    root, _ = tree
    path = _write(root, "Widget", "state", [{"fact": "Widget is beta.", "id": "fact-001"}])
    before = path.read_text()
    monkeypatch.setattr(facts_mod, "_writes_enabled", lambda: False)
    out = getattr(facts_mod, handler)(params)
    assert "error" in out and "write_enabled" in out["error"]
    assert path.read_text() == before


def test_fact_add_schema_no_longer_promises_a_default_valid_at():
    import asyncio
    tools = {t.name: t for t in asyncio.run(facts_mod.list_tools())}
    va = tools["fact_add"].input_schema["properties"]["valid_at"]["description"]
    assert "default: today" not in va
    assert "evidence" in tools["fact_relate"].input_schema["properties"]
