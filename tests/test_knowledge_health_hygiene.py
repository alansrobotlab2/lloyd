"""knowledge-health-report.py — the Hygiene section computed from loaded facts."""
import importlib.util
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("khr", ROOT / "scripts/memory/knowledge-health-report.py")
khr = importlib.util.module_from_spec(_spec); sys.modules["khr"] = khr; _spec.loader.exec_module(khr)


def _facts(root, name, cat, items):
    d = root / name; d.mkdir(parents=True, exist_ok=True)
    fm = {"type": "facts", "entity": name, "category": cat,
          "facts": [{"entity": e, "fact": t, "confidence": 0.9, "category": cat} for e, t in items]}
    p = d / f"{name}-{cat}.md"
    p.write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# {name} - {cat}\n")
    return p


def test_hygiene_from_loaded_entities(tmp_path):
    root = tmp_path / "facts"
    _facts(root, "Intel", "state", [("Intel", "chips"), ("Intel Pipeline System", "scans arxiv")])
    _facts(root, "vLLM", "state", [("vLLM", "serves")])
    old = _facts(root, "vllm", "state", [("vllm", "lowercase twin")])
    _facts(root, "Alfie", "state", [("Alfie", "robot")])
    now = datetime.now(timezone.utc)
    # vLLM is a month old; its lowercase twin was born yesterday
    for f in (root / "vLLM").glob("*.md"):
        os.utime(f, (now.timestamp() - 30 * 86400,) * 2)
    os.utime(old, (now.timestamp() - 86400,) * 2)

    entities = khr.load_entities(root)
    h = khr.compute_hygiene(entities, now)
    assert h["contaminated"] == [("Intel", "Intel Pipeline System", 1)]
    assert h["contaminated_dirs"] == 1 and h["foreign_facts"] == 1
    assert h["near_dup_clusters"] == 1 and h["near_dup_dirs"] == 2
    assert h["near_dup_tiers"] == {"SAFE": 1}
    assert [(n, o) for n, o, _ in h["regrown"]] == [("vllm", "vLLM")]

    report = khr.generate_report(khr.compute_entity_stats(entities),
                                 khr.compute_relationship_stats([], entities), [], [], now, h)
    assert "## Hygiene" in report
    assert "Contaminated entity dirs" in report and "| 1 |" in report
    assert "`Intel` holds 1 fact(s) tagged `Intel Pipeline System`" in report
    assert "`vllm` next to `vLLM` (CASE)" in report


def test_hygiene_section_is_optional(tmp_path):
    now = datetime.now(timezone.utc)
    report = khr.generate_report({}, khr.compute_relationship_stats([], {}), [], [], now)
    assert "## Hygiene" not in report


def test_hygiene_regrowth_uses_fact_created_at(tmp_path):
    root = tmp_path / "facts"
    now = datetime.now(timezone.utc)
    old_iso = (now.replace(microsecond=0) - __import__("datetime").timedelta(days=40)).isoformat()
    new_iso = (now.replace(microsecond=0) - __import__("datetime").timedelta(days=1)).isoformat()
    a = _facts(root, "vLLM", "state", [("vLLM", "serves")])
    b = _facts(root, "vllm", "state", [("vllm", "twin")])
    for p, iso in ((a, old_iso), (b, new_iso)):
        fm = yaml.safe_load(p.read_text().split("---")[1])
        for f in fm["facts"]:
            f["created_at"] = iso
        p.write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\nbody\n")
    # both files were written just now — mtime would call both "new"
    h = khr.compute_hygiene(khr.load_entities(root), now)
    assert h["new_dirs"] == 1
    assert [(n, o) for n, o, _ in h["regrown"]] == [("vllm", "vLLM")]


# ── fact-level exact duplicates (#499 clause 5) ──────────────────────────────

def test_fact_level_duplicate_total_comes_from_facts_idx(tmp_path):
    """`Near-duplicate name clusters` counts entity-name DIRECTORIES
    (`kg_hygiene.near_duplicates` → `_clusters(root)`, on `d.name`), so a report
    that printed only that line had never measured a duplicated FACT. The new
    line is read from `facts_idx.text_hash`, over every row including expired
    ones, and states its denominator."""
    from app import kg_store

    root = tmp_path / "facts"
    _facts(root, "vLLM", "state", [("vLLM", "serves openai api"),
                                   ("vLLM", "serves openai api"),
                                   ("vLLM", "paged attention")])
    _facts(root, "QMD", "state", [("QMD", "indexes the vault")])
    _facts(root, "QMD", "usage", [("QMD", "serves openai api")])   # different entity
    kg_store.configure(tmp_path / "kg.sqlite")
    st = kg_store.store()
    st.facts_idx.reindex(sorted(root.rglob("*.md")), root=root)

    d = khr.fact_duplicate_stats()
    assert d["unavailable"] is False, d
    assert d["rows"] == 5, d                                       # 3 + 1 + 1 indexed rows
    assert d["distinct_texts"] == 3, d                             # 3 different texts
    assert d["duplicate_rows"] == 2, d                             # "serves openai api" lives 3 times
    assert d["groups"] == 1, d
    assert d["same_entity_redundant_rows"] == 1, d                 # the vLLM pair, same entity
    assert d["same_entity_groups"] == 1, d
    assert d["entities_with_exact_dupes"] == 1, d
    kg_store.reset()


def test_fact_level_duplicate_line_renders_distinct_from_name_clusters(tmp_path):
    """The new line is its own row of Summary Stats, so the morning briefing
    sees the trend where every other total already lives — and it is textually
    distinct from the Hygiene section's name-cluster line, which stays put."""
    from app import kg_store

    root = tmp_path / "facts"
    _facts(root, "vLLM", "state", [("vLLM", "serves"), ("vLLM", "serves")])
    kg_store.configure(tmp_path / "kg.sqlite")
    st = kg_store.store()
    st.facts_idx.reindex(sorted(root.rglob("*.md")), root=root)

    now = datetime.now(timezone.utc)
    entities = khr.load_entities(root)
    h = khr.compute_hygiene(entities, now)
    d = khr.fact_duplicate_stats()
    kg_store.reset()

    report = khr.generate_report(khr.compute_entity_stats(entities),
                                 khr.compute_relationship_stats([], {}), [], [], now,
                                 h, fact_dups=d)
    lines = report.splitlines()
    dup_line = [ln for ln in lines if "Exact-duplicate fact rows" in ln]
    assert dup_line == ["| Exact-duplicate fact rows (facts_idx.text_hash) | 1 redundant of 2 rows "
                        "(entities with an exact twin: 1; distinct texts: 1) |"], dup_line
    # The name-cluster line is still there and still says something different.
    cluster_line = [ln for ln in lines if "Near-duplicate name clusters" in ln]
    assert cluster_line and cluster_line != dup_line, (cluster_line, dup_line)


def test_fact_level_duplicate_line_reports_that_it_could_not_measure(tmp_path, monkeypatch):
    """A health line that reads `0` when the store was unreadable is the #499
    failure mode again — a missing denominator has to say so."""
    from app.kg_store import StoreUnavailable

    def boom():
        raise StoreUnavailable("probe: kg.sqlite is a directory")
    monkeypatch.setattr(khr, "_kg_store", boom)

    now = datetime.now(timezone.utc)
    d = khr.fact_duplicate_stats()
    assert d["unavailable"] is True and "rows" not in d, d

    report = khr.generate_report({}, khr.compute_relationship_stats([], {}), [], [], now,
                                 fact_dups=d)
    assert "| Exact-duplicate fact rows (facts_idx.text_hash) | not measured: " in report


def test_the_script_prints_the_fact_level_total(tmp_path):
    """#499 clause 5 is about the *run*, not the helper: the nightly job invokes
    `knowledge-health-report.py` as a script, so the line has to survive
    `main()` — the wiring that drops it between `fact_duplicate_stats()` and the
    file the morning briefing reads is the failure a unit test cannot see."""
    import subprocess
    from app import kg_store

    root = tmp_path / "facts"
    _facts(root, "vLLM", "state", [("vLLM", "serves openai api"),
                                   ("vLLM", "serves openai api")])
    db = tmp_path / "kg.sqlite"
    st = kg_store.configure(db)
    st.facts_idx.reindex(sorted(root.rglob("*.md")), root=root)
    kg_store.reset()

    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/memory/knowledge-health-report.py"),
         "--facts-dir", str(root), "--output-dir", str(out), "--no-alarm-exit"],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "LLOYD_KG_DB": str(db), "LLOYD_FACTS_ROOT": str(root)})
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = next(out.glob("knowledge-health-*.md")).read_text()
    expected = ("| Exact-duplicate fact rows (facts_idx.text_hash) | 1 redundant of 2 rows "
                "(entities with an exact twin: 1; distinct texts: 1) |")
    assert expected in report, report[-1200:]
    # And on stdout, in that line's own form — the job log keeps stdout, which
    # is what a nightly run is actually read from.
    assert ("  Exact-duplicate fact rows (facts_idx.text_hash): 1 redundant of 2 rows "
            "(entities with an exact twin: 1; distinct texts: 1)") in proc.stdout, proc.stdout[-1500:]


# ── #1289: the printed metrics are in the written report ────────────────────

def _hygiene(**over):
    h = {"contaminated_dirs": 0, "foreign_facts": 0, "near_dup_clusters": 0,
         "near_dup_dirs": 0, "near_dup_tiers": {}, "regrown": [], "new_dirs": 0,
         "regrowth_days": 7, "contaminated": []}
    h.update(over)
    return h


def _report(hygiene, **kw):
    now = datetime.now(timezone.utc)
    return khr.generate_report({}, khr.compute_relationship_stats([], {}), [], [], now,
                               hygiene, **kw)


def _row(report, label):
    rows = [l for l in report.splitlines() if l.startswith(f"| {label}")]
    assert len(rows) == 1, report
    return rows[0]


def test_provenance_coverage_row_carries_both_pct_count_and_components():
    pv = {"facts": 318396, "created_at_pct": 35.7, "source_doc_pct": 35.69,
          "both_pct": 35.69}
    row = _row(_report(_hygiene(provenance=pv), duplicate_id_files=0),
               "Provenance coverage")
    assert "35.69% of 318,396 facts" in row
    assert "created_at 35.7%" in row and "source_doc 35.69%" in row


def test_unmeasured_provenance_says_so():
    for pv in ({"error": "StoreUnavailable: locked"}, None):
        h = _hygiene() if pv is None else _hygiene(provenance=pv)
        row = _row(_report(h), "Provenance coverage")
        assert "not measured" in row and "None" not in row, row
    assert "StoreUnavailable" in _row(
        _report(_hygiene(provenance={"error": "StoreUnavailable: locked"})),
        "Provenance coverage")


def test_duplicate_fact_id_row_is_present_at_zero_and_nonzero():
    for n in (0, 4):
        row = _row(_report(_hygiene(), duplicate_id_files=n),
                   "Files with duplicate fact IDs")
        assert row.endswith(f"| {n} |"), row
