"""knowledge-health-report.py — the Hygiene section computed from loaded facts."""
import importlib.util
import os
import subprocess
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


def _kg_hygiene():
    """kg_hygiene by the path `compute_hygiene` imports it under, so the
    baseline seeded here is the one the report reads."""
    sys.path.insert(0, str(ROOT / "scripts" / "memory"))
    import kg_hygiene
    return kg_hygiene


def test_hygiene_from_loaded_entities(tmp_path):
    root = tmp_path / "facts"
    _facts(root, "Intel", "state", [("Intel", "chips"), ("Intel Pipeline System", "scans arxiv")])
    _facts(root, "vLLM", "state", [("vLLM", "serves")])
    _facts(root, "Alfie", "state", [("Alfie", "robot")])
    base = tmp_path / "baseline.json"
    _kg_hygiene().write_baseline(root, base)
    # the lowercase twin appears after the reference, next to its month-old sibling
    _facts(root, "vllm", "state", [("vllm", "lowercase twin")])
    now = datetime.now(timezone.utc)

    entities = khr.load_entities(root)
    h = khr.compute_hygiene(entities, now, baseline_path=base)
    assert h["contaminated"] == [("Intel", "Intel Pipeline System", 1)]
    assert h["contaminated_dirs"] == 1 and h["foreign_facts"] == 1
    assert h["near_dup_clusters"] == 1 and h["near_dup_dirs"] == 2
    assert h["near_dup_tiers"] == {"SAFE": 1}
    assert h["new_dirs"] == 1 and h["dirs_at_baseline"] == 3
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


def test_hygiene_regrowth_is_a_baseline_diff_not_a_created_at_window(tmp_path):
    """Both files were written just now and their `created_at` says so, which is
    the state the 2026-09-23 rebuild left the live tree in: every directory
    dated inside the window, `new_dirs` equal to the store's own size (#1535).
    The reference is the stored directory set, so exactly one directory is
    new here — the twin created after it."""
    root = tmp_path / "facts"
    now = datetime.now(timezone.utc)
    fresh_iso = now.replace(microsecond=0).isoformat()
    a = _facts(root, "vLLM", "state", [("vLLM", "serves")])
    base = tmp_path / "baseline.json"
    _kg_hygiene().write_baseline(root, base)
    b = _facts(root, "vllm", "state", [("vllm", "twin")])
    for p in (a, b):
        fm = yaml.safe_load(p.read_text().split("---")[1])
        for f in fm["facts"]:
            f["created_at"] = fresh_iso                  # rebuilt: every fact re-dated today
        p.write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\nbody\n")

    h = khr.compute_hygiene(khr.load_entities(root), now, baseline_path=base)
    assert h["new_dirs"] == 1 and h["dirs_at_baseline"] == 1
    assert [(n, o) for n, o, _ in h["regrown"]] == [("vllm", "vLLM")]


# ── fact-level exact duplicates (#499 clause 5) ──────────────────────────────

# #1942: the two-row fixtures below carry no `source_doc`, so the same-source
# half of the cell reads zero of their two active rows — printed, never omitted.
SAME_SOURCE_NONE = ("; same-source: 0 of 2 active rows (0.0%) share an entity and a "
                    "source document with an earlier row, in 0 groups: a population at "
                    "risk, not counted as duplicates")


def _sourced(root, name, cat, items):
    """Like `_facts`, with a `source_doc` (and optional `expired_at`) per fact."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    facts = []
    for text, doc, *rest in items:
        f = {"entity": name, "fact": text, "confidence": 0.9, "category": cat,
             "source_doc": doc}
        if rest:
            f["expired_at"] = rest[0]
        facts.append(f)
    fm = {"type": "facts", "entity": name, "category": cat, "facts": facts}
    p = d / f"{name}-{cat}.md"
    p.write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# {name} - {cat}\n")
    return p


def test_the_duplicate_cell_prints_same_source_rows_over_the_active_denominator(tmp_path):
    """#1942: the exact-twin number reads 0 on a store whose rows mostly
    re-assert their own source document in different words. The cell carries
    that second number with the row set it was measured on — active rows, which
    is NOT the `rows` the exact-twin number divides by (one row here is
    expired) — and names it a population at risk, never a duplicate or an error."""
    from app import kg_store

    root = tmp_path / "facts"
    _sourced(root, "vLLM", "state", [("serves the api", "knowledge/a.md"),
                                     ("answers on 8096", "knowledge/a.md"),
                                     ("holds the kv pool", "knowledge/a.md"),
                                     ("was on llama.cpp", "knowledge/old.md", "2026-02-01")])
    _sourced(root, "QMD", "state", [("indexes the vault", "knowledge/c.md")])
    kg_store.configure(tmp_path / "kg.sqlite")
    st = kg_store.store()
    st.facts_idx.reindex(sorted(root.rglob("*.md")), root=root)
    d = khr.fact_duplicate_stats()
    kg_store.reset()

    assert (d["rows"], d["active_rows"]) == (5, 4), d
    assert (d["same_source_paraphrase_groups"], d["same_source_redundant_rows"]) == (1, 2), d
    assert d["same_entity_redundant_rows"] == 0, d          # every text is distinct
    cell = khr._fact_duplicate_cell(d)
    assert "0 redundant of 5 rows" in cell, cell
    assert "2 of 4 active rows (50.0%)" in cell, cell       # the count never stands alone
    assert "population at risk" in cell and "not counted as duplicates" in cell, cell
    same_source = cell.split("same-source:", 1)[1]
    assert "error" not in same_source.lower(), cell

    now = datetime.now(timezone.utc)
    report = khr.generate_report({}, khr.compute_relationship_stats([], {}), [], [], now,
                                 fact_dups=d)
    line = [ln for ln in report.splitlines() if "Exact-duplicate fact rows" in ln]
    assert len(line) == 1 and "2 of 4 active rows" in line[0] \
        and "population at risk" in line[0], line


def test_the_duplicate_cell_reads_an_older_stats_dict_without_the_new_keys():
    """A dict without the #1942 keys (an older store build) renders the cell it
    always did rather than raising: the report degrades, it does not die."""
    cell = khr._fact_duplicate_cell({"rows": 2, "distinct_texts": 1,
                                     "same_entity_redundant_rows": 1,
                                     "entities_with_exact_dupes": 1})
    assert cell == "1 redundant of 2 rows (entities with an exact twin: 1; distinct texts: 1)"

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
                        "(entities with an exact twin: 1; distinct texts: 1)" + SAME_SOURCE_NONE + " |"], dup_line
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
                "(entities with an exact twin: 1; distinct texts: 1)" + SAME_SOURCE_NONE + " |")
    assert expected in report, report[-1200:]
    # And on stdout, in that line's own form — the job log keeps stdout, which
    # is what a nightly run is actually read from.
    assert ("  Exact-duplicate fact rows (facts_idx.text_hash): 1 redundant of 2 rows "
            "(entities with an exact twin: 1; distinct texts: 1)" + SAME_SOURCE_NONE) in proc.stdout, proc.stdout[-1500:]


# ── #1289: the printed metrics are in the written report ────────────────────

def _hygiene(**over):
    h = {"contaminated_dirs": 0, "foreign_facts": 0, "near_dup_clusters": 0,
         "near_dup_dirs": 0, "near_dup_tiers": {}, "regrown": [], "new_dirs": 0,
         "regrowth_days": 7, "contaminated": [],
         # #1535: the phrase kg_hygiene formats, counts AND reference, which is
         # what the regrowth row renders.
         "regrowth_line": "0 of 0 new dirs (baseline 2026-09-26T07:27:00+00:00 "
                          "over 12,027 dirs)",
         "baseline_at": "2026-09-26T07:27:00+00:00", "dirs_at_baseline": 12027}
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


# ── #1535 clause 5: the reference is printed beside the number ────────────────


def test_the_regrowth_row_prints_the_reference_beside_the_number(tmp_path):
    root = tmp_path / "facts"
    _facts(root, "Intel", "state", [("Intel", "chips")])
    _facts(root, "vLLM", "state", [("vLLM", "serves")])
    _facts(root, "Alfie", "state", [("Alfie", "robot")])
    base = tmp_path / "baseline.json"
    _kg_hygiene().write_baseline(root, base)
    _facts(root, "vllm", "state", [("vllm", "twin")])
    h = khr.compute_hygiene(khr.load_entities(root), datetime.now(timezone.utc),
                            baseline_path=base)

    row = _row(_report(h), "Near-duplicate dirs coined")
    assert "1 of 1 new dirs" in row, row
    assert h["baseline_at"] in row, row              # the moment the reference was taken
    assert "over 3 dirs" in row, row                 # and how big it was
    assert "7 days" not in row, row                  # the window bounds nothing


def test_the_regrowth_row_says_not_measured_without_a_baseline(tmp_path):
    root = tmp_path / "facts"
    _facts(root, "Intel", "state", [("Intel", "chips")])
    _facts(root, "vllm", "state", [("vllm", "twin")])
    _facts(root, "Alfie", "state", [("Alfie", "robot")])
    h = khr.compute_hygiene(khr.load_entities(root), datetime.now(timezone.utc),
                            baseline_path=tmp_path / "absent.json")
    assert h["new_dirs"] is None

    row = _row(_report(h), "Near-duplicate dirs coined")
    assert "not measured" in row, row
    assert "absent.json" in row, row                 # says which file is missing
    assert "of 3 new dirs" not in row, row           # never the store's own size


def test_a_hygiene_dict_with_no_reference_cannot_print_a_bare_count():
    """The row used to be composed from the raw keys — `{len(regrown)} of
    {new_dirs} new dirs` — so whatever a caller left in `new_dirs` was printed
    as fact with nothing beside it. 12,027 is the number #1535 was filed on: the
    entire entity store, rendered as `51 of 12027 new dirs` and read as a week's
    growth. Given counts and no reference, the row now prints no number at all."""
    h = _hygiene(new_dirs=12027, near_dup_clusters=51,
                 regrown=[("vllm", "vLLM", "CASE")] * 51)
    del h["regrowth_line"]

    row = _row(_report(h), "Near-duplicate dirs coined")
    assert "not measured" in row, row
    assert "12027" not in row and "12,027" not in row, row
    assert "51" not in row, row


def test_stdout_prints_the_reference_next_to_the_regrowth_count(tmp_path):
    """The stdout recap used to read `regrown in 7d: 1` — the same number with
    even less context than the table row above it. It prints kg_hygiene's
    phrase, which carries the reference or the reason there isn't one."""
    root = tmp_path / "facts"
    _facts(root, "Intel", "state", [("Intel", "chips")])
    env = {**os.environ, "LLOYD_FACTS_ROOT": str(root)}
    script = str(ROOT / "scripts" / "memory" / "knowledge-health-report.py")
    proc = subprocess.run([sys.executable, script, "--output-dir", str(tmp_path),
                           "--facts-dir", str(root), "--no-alarm-exit"],
                          capture_output=True, text=True, timeout=300, env=env, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stderr[-1500:]
    lines = [ln for ln in proc.stdout.splitlines() if "regrowth" in ln]
    assert len(lines) == 1, proc.stdout[-1500:]
    # No baseline exists in the scratch data root this run is given, so the
    # honest rendering is the reason, not a directory count.
    assert "not measured" in lines[0] and "no baseline file at" in lines[0], lines[0]
    assert "regrown in" not in proc.stdout, proc.stdout


# ── #2474: the untagged row and the alarm that goes with it ──────────────────

def _raw_facts(root, name, cat, records):
    """A fact file whose records are written verbatim, so `entity: ''` and a
    missing `entity:` key can be built at all — `_facts` above tags every record
    it is handed, and so no fixture in this file could ever have produced the
    3,531 records #2474 was filed about."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    fm = {"type": "facts", "entity": name, "category": cat, "facts": records}
    p = d / f"{name}-{cat}.md"
    p.write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# {name} - {cat}\n")
    return p


def _blank_tree(root):
    """1 untagged-heavy dir plus one clean dir: 5 records parse, 4 untagged, all
    in 1 dir and 1 file, so the share is 80.0% and 0.5% is far below it. Nothing
    here is contaminated — every tagged record names its own dir."""
    _raw_facts(root, "general", "state", [
        {"entity": "", "fact": "a how-to-fail book"},
        {"entity": "", "fact": "uptake_probe field names"},
        {"entity": "", "fact": "vault redirect routes"},
        {"fact": "the key was never written at all"},
    ])
    _facts(root, "Lloyd", "state", [("Lloyd", "runs the harness")])
    return root


def test_the_hygiene_table_prints_untagged_records_with_their_denominator(tmp_path):
    """Clause: the report shows untagged records as a row of their own.

    Across the seam it was missing at: `kg_hygiene.scan` → `compute_hygiene` →
    the written report. On 2026-10-09 this same corpus printed
    `Contaminated entity dirs | 0` and `Foreign facts | 0` over 3,531 records
    that name no entity, because a blank tag is neither "this entity" nor
    "another entity" and fell out of both sides of that check.
    """
    root = _blank_tree(tmp_path / "facts")
    now = datetime.now(timezone.utc)
    entities = khr.load_entities(root)
    h = khr.compute_hygiene(entities, now)
    assert h["untagged"] == {"facts": 4, "dirs": 1, "files": 1, "records": 5,
                             "share_pct": 80.0, "floor_pct": 0.5,
                             "over_floor": True}
    assert h["contaminated_dirs"] == 0 and h["foreign_facts"] == 0, (
        "the contamination half must be untouched by the records beside it")

    report = khr.generate_report(khr.compute_entity_stats(entities),
                                 khr.compute_relationship_stats([], entities), [], [], now, h)
    row = _row(report, "Untagged fact records")
    for frag in ("4 records in 1 dirs", "1 files", "80.0% of 5 records",
                 "floor 0.5%", "ABOVE FLOOR"):
        assert frag in row, row
    assert "`entity:`" in row, "the row names the field it counts, not a vibe"


def test_the_untagged_row_is_printed_at_zero_and_when_never_measured():
    """A row that appears only when the number is nonzero could not have caught
    this: the defect was invisible partly because nothing in the table had a
    place for it. And a hand-built hygiene dict that never measured untagged must
    read as `not measured`, never as a clean 0 — the same rule the provenance and
    regrowth rows follow (#1289, #1535)."""
    zero = {"facts": 0, "dirs": 0, "files": 0, "records": 100, "share_pct": 0.0,
            "floor_pct": 0.5, "over_floor": False}
    row = _row(_report(_hygiene(untagged=zero)), "Untagged fact records")
    assert "0 records in 0 dirs" in row and "ABOVE FLOOR" not in row, row
    plain = _row(_report(_hygiene()), "Untagged fact records")
    assert "not measured" in plain and "0 records" not in plain, plain


def test_untagged_above_the_floor_makes_the_run_an_alarm(tmp_path):
    """Clause: the exit-2 rail fires on the untagged population.

    The alarm list is what `main()` prints to stderr, alerts on, and turns into
    the exit code the scheduler reads, so the condition is pinned here rather
    than at the subprocess: a run of that script that raises alarms also posts
    them to Discord, which no test should do to a live channel.
    """
    root = _blank_tree(tmp_path / "facts")
    entities = khr.load_entities(root)
    h = khr.compute_hygiene(entities, datetime.now(timezone.utc))
    alarms = khr._alarms({"edges_active": 100}, h, 0, 0)
    assert len(alarms) == 1, alarms
    a = alarms[0]
    for frag in ("4 fact records carry no entity tag", "80.0% of 5 records",
                 "above the 0.5% floor", "contamination rail cannot see them"):
        assert frag in a, a


def test_the_untagged_alarm_is_independent_of_the_contamination_one(tmp_path):
    """The merge alarm keeps its exact meaning and wording: 1 contaminated dir is
    still "a merge went wrong", and a tree with untagged records under the floor
    raises nothing. The two conditions are separate lines of the same list, so
    fixing the blind spot cannot silence the old one or have it stand in."""
    root = tmp_path / "facts"
    _raw_facts(root, "Intel", "state", [
        {"entity": "Intel", "fact": "released the Pro B70 GPU."},
        {"entity": "Intel Pipeline System", "fact": "Scans ArXiv nightly."},
    ])
    entities = khr.load_entities(root)
    h = khr.compute_hygiene(entities, datetime.now(timezone.utc))
    assert h["contaminated_dirs"] == 1
    assert h["untagged"] == {"facts": 0, "dirs": 0, "files": 0, "records": 2,
                             "share_pct": 0.0, "floor_pct": 0.5,
                             "over_floor": False}
    alarms = khr._alarms({"edges_active": 100}, h, 0, 0)
    assert len(alarms) == 1 and "a merge went wrong" in alarms[0], alarms
    assert not any("entity tag" in a for a in alarms), alarms
