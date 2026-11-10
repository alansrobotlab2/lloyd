"""Superseding the facts a re-extracted note's earlier bytes produced.

`knowledge_graph.supersede_on_reextract` ships `off`. These pin the decision
(`plan`, pure) and both armed modes against a temp fact tree and store: `dry_run`
journals and changes nothing; `apply` marks `expired_at` and never deletes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agent_mcp._shared as shared             # noqa: E402
from agent_mcp import facts as facts_mod       # noqa: E402
from agent_mcp import fact_supersede as fs     # noqa: E402
from agent_mcp import retrieval                # noqa: E402
from app import kg_store                       # noqa: E402
from app.kg_store import text_hash             # noqa: E402

DOC = "knowledge/tools/widget.md"
CFG_GLOBS = ("memory/*.md", "**/prs.md")


def _row(fact, h, fid="fact-001", path="Widget/Widget-state.md"):
    return {"fact": fact, "text_hash": text_hash(fact), "source_hash": h,
            "fact_id": fid, "file_path": path, "entity": "Widget"}


def _cfg(mode="apply", ratio=0.5):
    return fs.SupersedeConfig(mode=mode, min_restate_ratio=ratio, exempt_globs=CFG_GLOBS)


# ── plan (pure) ──────────────────────────────────────────────────────────────

def test_old_facts_not_restated_are_expired_and_restated_ones_stay():
    rows = [_row("Widget runs on port 9000.", "old", "fact-001"),
            _row("Widget is beta.", "old", "fact-002"),
            _row("Widget is written in Rust.", "new", "fact-003")]
    p = fs.plan(DOC, "new", ["Widget runs on port 9000.", "Widget is written in Rust."],
                rows, _cfg(), read_whole_note=True)
    assert p.verdict == "expire"
    assert [r["fact_id"] for r in p.expire] == ["fact-002"]
    assert p.kept_restated == 1 and p.old_active == 2


def test_restatement_is_matched_the_way_the_store_dedupes():
    rows = [_row("Widget is beta.", "old")]
    p = fs.plan(DOC, "new", ["  widget IS beta.  "], rows, _cfg(), read_whole_note=True)
    assert p.verdict == "nothing" and p.kept_restated == 1


@pytest.mark.parametrize("doc", ["memory/2026-09-09.md", "knowledge/tools/qmd/prs.md"])
def test_feeds_are_exempt(doc):
    p = fs.plan(doc, "new", ["x"], [_row("old fact", "old")], _cfg(), read_whole_note=True)
    assert p.verdict == "skipped" and "exempt" in p.reason


def test_a_partial_pass_judges_nothing():
    p = fs.plan(DOC, "new", ["x"], [_row("old fact", "old")], _cfg(), read_whole_note=False)
    assert p.verdict == "skipped" and not p.expire


def test_a_thin_pass_judges_nothing():
    rows = [_row(f"old fact {i}", "old", f"fact-{i:03d}") for i in range(10)]
    p = fs.plan(DOC, "new", ["one new fact"], rows, _cfg(ratio=0.5), read_whole_note=True)
    assert p.verdict == "skipped" and "thin" in p.reason


def test_rows_without_an_id_are_left_alone():
    rows = [_row("old a", "old", fid=None), _row("old b", "old", "fact-002")]
    p = fs.plan(DOC, "new", ["n1", "n2"], rows, _cfg(), read_whole_note=True)
    assert [r["fact_id"] for r in p.expire] == ["fact-002"]
    assert "without an id" in p.reason


def test_off_does_nothing_and_reads_nothing(monkeypatch):
    def boom():
        raise AssertionError("store read while off")
    monkeypatch.setattr(kg_store, "store", boom)
    assert fs.supersede_after_reextract(DOC, "new", ["x"], read_whole_note=True,
                                        cfg=_cfg(mode="off")) == {"mode": "off"}


# ── the armed modes, against a temp tree ─────────────────────────────────────

@pytest.fixture
def tree(tmp_path, monkeypatch):
    facts_root = tmp_path / "facts"
    facts_root.mkdir()
    for mod in (shared, retrieval, facts_mod):
        monkeypatch.setattr(mod, "FACTS_ROOT", facts_root)
    shared._invalidate_entity_dirs_cache()
    retrieval.invalidate_fact_file_cache()
    st = kg_store.configure(tmp_path / "kg.sqlite")
    yield facts_root, st, tmp_path
    kg_store.reset()
    shared._invalidate_entity_dirs_cache()
    retrieval.invalidate_fact_file_cache()


def _write_file(root: Path) -> Path:
    d = root / "Widget"
    d.mkdir()
    rows = [
        {"entity": "Widget", "fact": "Widget runs on port 9000.", "id": "fact-001",
         "source_doc": DOC, "source_hash": "old", "expired_at": None, "invalid_at": None},
        {"entity": "Widget", "fact": "Widget is beta.", "id": "fact-002",
         "source_doc": DOC, "source_hash": "old", "expired_at": None, "invalid_at": None},
        {"entity": "Widget", "fact": "Widget is written in Rust.", "id": "fact-003",
         "source_doc": DOC, "source_hash": "new", "expired_at": None, "invalid_at": None},
        {"entity": "Widget", "fact": "Widget has a logo.", "id": "fact-004",
         "source_doc": "knowledge/other.md", "source_hash": "zzz",
         "expired_at": None, "invalid_at": None},
    ]
    fm = {"type": "facts", "entity": "Widget", "category": "state", "facts": rows}
    path = d / "Widget-state.md"
    path.write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# Widget\n", encoding="utf-8")
    return path


def test_dry_run_journals_and_changes_nothing(tree):
    root, st, tmp = tree
    path = _write_file(root)
    st.facts_idx.reindex(root=root)
    before = path.read_text()
    journal = tmp / "j.jsonl"
    out = fs.supersede_after_reextract(
        DOC, "new", ["Widget runs on port 9000.", "Widget is written in Rust."],
        read_whole_note=True, cfg=_cfg(mode="dry_run"), journal_path=journal)
    assert out["would_expire"] == 1 and "expired" not in out
    assert path.read_text() == before
    rows = [json.loads(l) for l in journal.read_text().splitlines()]
    assert [(r["fact_id"], r["old_hash"]) for r in rows] == [("fact-002", "old")]


def test_apply_expires_only_the_superseded_fact(tree):
    root, st, tmp = tree
    path = _write_file(root)
    st.facts_idx.reindex(root=root)
    journal = tmp / "j.jsonl"
    out = fs.supersede_after_reextract(
        DOC, "new", ["Widget runs on port 9000.", "Widget is written in Rust."],
        read_whole_note=True, cfg=_cfg(mode="apply"), journal_path=journal)
    assert out["expired"] == 1, out
    facts = {f["id"]: f for f in shared._parse_fact_frontmatter(path.read_text())["facts"]}
    assert len(facts) == 4                                   # nothing deleted
    assert facts["fact-002"]["expired_at"] and facts["fact-002"]["expired_reason"] == fs.REASON
    assert not any(facts[i].get("expired_at") for i in ("fact-001", "fact-003", "fact-004"))
    active = {r["fact_id"] for r in st.facts_idx.active_for_source(DOC)}
    assert active == {"fact-001", "fact-003"}
    assert json.loads(journal.read_text().splitlines()[0])["applied"] is True


# ── the report ───────────────────────────────────────────────────────────────

def test_report_counts_rows_on_an_old_hash(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts" / "memory"))
    import supersede_report
    from app.atomic_io import hash_bytes
    vault = tmp_path / "vault"
    (vault / "knowledge").mkdir(parents=True)
    (vault / "memory").mkdir()
    (vault / "knowledge" / "a.md").write_bytes(b"now")
    (vault / "memory" / "2026-01-01.md").write_bytes(b"day")
    cur = hash_bytes(b"now")
    rows = [{"source_doc": "knowledge/a.md", "source_hash": cur, "n": 5},
            {"source_doc": "knowledge/a.md", "source_hash": "old", "n": 3},
            {"source_doc": "memory/2026-01-01.md", "source_hash": "old", "n": 7},
            {"source_doc": "knowledge/gone.md", "source_hash": "x", "n": 2}]
    rep = supersede_report.build_report(rows, vault, CFG_GLOBS)
    t = rep["totals"]
    assert t["on_old_hash"] == 3 and t["exempt_on_old_hash"] == 7
    assert t["missing_notes"] == 1 and t["missing_note_rows"] == 2
    assert rep["notes"][0]["source_doc"] == "memory/2026-01-01.md"
