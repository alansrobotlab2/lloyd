"""knowledge-health-report.py — the Hygiene section computed from loaded facts."""
import ast
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import time
import tokenize
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
    """The new line is its own row of Summary Stats, so the two jobs that read this
    report as a file — research-queue-generator (#65) and entity-resolution-sweep (#48) —
    see the trend where every other total already lives, and it is textually distinct
    from the Hygiene section's name-cluster line, which stays put.

    The line used to justify itself by a 6 AM briefing that opens nothing; the job was
    retired (#1508, 2026-09-25) and the consumer prose went out of the skill and the task
    file, and this was the last copy of it (#2931)."""
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
    file the two live readers open, research-queue-generator (#65) and
    entity-resolution-sweep (#48), is the failure a unit test cannot see."""
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
    # Floor at the store's own count: usable, and not below half, so the single alarm
    # below is the untagged rail and not #3222's unusable-floor alarm.
    alarms = khr._alarms({"edges_active": 100}, h, 0, 100)
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
    # Same usable floor as the node above, so `len(alarms) == 1` still counts the merge
    # alarm alone and not a floor alarm that arrived with #3222.
    alarms = khr._alarms({"edges_active": 100}, h, 0, 100)
    assert len(alarms) == 1 and "a merge went wrong" in alarms[0], alarms
    assert not any("entity tag" in a for a in alarms), alarms


# ─────────────────────────────────────────────────────────────────────────────
# The report's own repo prose has to name its LIVE readers (#2931)
#
# Vault 4270b5ef removed the dead 6 AM consumer from the shipped skill and from
# `autonomy/60-knowledge-health-report.md`, but three sentences in THIS repo still named
# that job as somebody who opens the report. The retired-skill prose guard could not see
# them because it keys on a quarantined skill's FOLDER SLUG, and those sentences used the
# job's human name instead, which contains no slug. Measured the day this was triaged, the
# guard read 136 paragraphs out of the report script and 36 out of this file and returned 0
# violations on both, so it was never the instrument for this fix and the nodes below are
# what it was blind to. The quarantined skill behind that slug is
# `nightly-morning-briefing`, retired by #1508 on 2026-09-25.
#
# The human name is spelled nowhere in this file on purpose: the nodes below search for it,
# so a sentence carrying it to announce its absence would be the hit. The needle is built by
# joining two words, which is also why the search can be trusted — see the injection node.

PROSE_SURFACES = ("scripts/memory/knowledge-health-report.py",
                  "tests/test_knowledge_health_hygiene.py")
#: Joined, not written: this file searches for the two-word phrase, and a file that states
#: the phrase it searches for cannot be scanned by it.
DEAD_READER_PHRASE = "morning" + " " + "briefing"
LIVE_READERS = ("research-queue-generator", "entity-resolution-sweep")
LIVE_READER_TASKS = ("#65", "#48")
#: The one repo file whose citation of the retired job's name is legitimate: it resolves an
#: entity alias, and the rewording must leave it out of its diff.
ENTITY_SWEEP_FILE = "tests/test_entity_sweep.py"


def _prose_of(text: str) -> str:
    """Every comment line and every string literal in `text`, joined into one string.

    String literals and not only docstrings: the retired claim sat in a TEST docstring, and
    its successor could equally sit in an assert message. Raises on text that does not
    compile, so a file the parse rejected is an error here and never a clean verdict.
    """
    found = [m.group(0) for m in re.finditer(r"#[^\n]*", text)]
    found += [n.value for n in ast.walk(ast.parse(text))
              if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    return "\n".join(found)


def _reader_prose(path: str) -> str:
    """`_prose_of` over a repo file, asserting the read was non-empty: a scan that read
    nothing would report every phrase as absent."""
    prose = _prose_of((ROOT / path).read_text(encoding="utf-8"))
    assert prose.strip(), f"{path}: the prose read came back empty"
    return prose


def test_the_prose_scan_finds_the_dead_reader_name_when_it_is_present():
    """Shown able to fail before its passing is believed.

    A scan that reports the name absent because it read no prose at all — wrong path, empty
    file, a walk over source that never parsed — is indistinguishable from a scan that found
    the name nowhere. So a copy of each real file gets the name injected into both a comment
    and a docstring, and the scan MUST see it there; only then does the real read below mean
    anything.
    """
    for path in PROSE_SURFACES:
        src = (ROOT / path).read_text(encoding="utf-8")
        # Appended at top level rather than spliced in at the first blank line: a splice
        # lands at whatever indentation that blank line sits at, and an indented def inside
        # a function body is a SyntaxError, which the parse below would rightly report as
        # an error instead of the injection this node is asking for.
        docstring = '"' * 3 + DEAD_READER_PHRASE + '"' * 3
        injected = (src + "\n\n# " + DEAD_READER_PHRASE + " reads this\n"
                    "def _injected_2931():\n    " + docstring + "\n")
        assert injected != src, f"{path}: nothing was injected"
        assert DEAD_READER_PHRASE in _prose_of(injected).lower(), (
            f"{path}: the scan cannot see the name even in prose written to contain it, so "
            "its verdict on the real file would mean nothing")


def test_no_comment_or_docstring_in_either_file_names_the_dead_reader():
    """Neither file mentions the retired 6 AM job anywhere — not merely in the three
    sentences #2931 named, because a fourth copy would be caught here rather than by someone
    re-reading the file. The one legitimate survivor in the repo is an entity-alias assertion
    in `tests/test_entity_sweep.py`, which resolves a real entity name and is a different
    thing; this node reads only the two surfaces above."""
    for path in PROSE_SURFACES:
        assert DEAD_READER_PHRASE not in _reader_prose(path).lower(), path


def test_the_row_cap_rationale_names_the_two_jobs_that_actually_read_the_report():
    """The comment above `SECTION_ROW_CAP` justified the cap by a page a human scrolls, in a
    report no human opens on a schedule. It has to name the two live jobs the shipped skill's
    Notes bullet names — research-queue-generator (#65) and entity-resolution-sweep (#48) —
    and keep the cap's own point, which is that the listing is cut off while the count is
    always given in full."""
    lines = (ROOT / "scripts/memory/knowledge-health-report.py").read_text(
        encoding="utf-8").splitlines()
    at = [i for i, v in enumerate(lines) if v.startswith("SECTION_ROW_CAP")]
    assert len(at) == 1, f"SECTION_ROW_CAP is defined {len(at)} times, not once"
    assert lines[at[0]] == "SECTION_ROW_CAP = 50", lines[at[0]]
    start = at[0]
    while start and lines[start - 1].lstrip().startswith("#"):
        start -= 1
    block = "\n".join(lines[start:at[0]])
    assert block.strip(), "the constant lost its rationale comment entirely"
    for want in LIVE_READERS + LIVE_READER_TASKS:
        assert want in block, f"{want} missing from the cap's rationale:\n{block}"
    assert "count" in block and ("cap" in block or "cut off" in block), block
    assert not re.search(r"\d{5,}", block), (
        f"a five-digit row count is back in the rationale, and a snapshot number in a comment "
        f"is a number nobody will re-measure:\n{block}")


def test_the_two_live_readers_are_named_where_a_reader_is_named():
    """The other half of the scan, and the control that keeps the deletion from passing on its
    own: the replacement names have to be findable by the same reading, so a rewording that
    removed the dead reader and named nobody in its place could not clear this file."""
    script = _reader_prose("scripts/memory/knowledge-health-report.py")
    for want in LIVE_READERS + LIVE_READER_TASKS:
        assert want in script, f"{want} is named nowhere in the report script's prose"


def _test_fn(name: str):
    tree = ast.parse((ROOT / "tests/test_knowledge_health_hygiene.py").read_text(
        encoding="utf-8"))
    for fn in tree.body:
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) and fn.name == name:
            return fn
    raise AssertionError(f"{name} is gone from this file")


def test_the_two_rewritten_docstrings_keep_their_own_points():
    """Neither docstring lost the thing it was there to say.

    The first pins that the exact-duplicate line is its own row of Summary Stats, distinct
    from the Hygiene section's name-cluster line, and keeps the live reason the printed line
    matters at all: #65 and #48 read the report as a file. The second keeps that the nightly
    job INVOKES the script as a program, which is why the print has to survive `main()` and
    why that wiring is exactly what a unit test cannot see."""
    doc_a = ast.get_docstring(_test_fn(
        "test_fact_level_duplicate_line_renders_distinct_from_name_clusters"))
    doc_b = ast.get_docstring(_test_fn("test_the_script_prints_the_fact_level_total"))
    assert doc_a and doc_b, "one of the two docstrings is gone"
    assert "own row of Summary Stats" in doc_a and "name-cluster" in doc_a, doc_a
    assert "nightly job invokes" in doc_b and "unit test cannot see" in doc_b, doc_b
    for doc in (doc_a, doc_b):
        assert all(task in doc for task in LIVE_READER_TASKS), doc


def _diff_by_file(diff_text: str):
    """A unified diff as `{path: (deleted base line numbers, added head line numbers)}`.

    Per file, because a flat list of deleted line numbers is meaningless across paths: the
    first version of this helper reported the hygiene file's own docstring lines as deleted
    non-prose from the report script, since both files' deletions shared one list."""
    per = {}
    for chunk in re.split(r"(?m)^diff --git a/\S+ b/", diff_text)[1:]:
        path, _, body = chunk.partition("\n")
        removed, added, b, h = [], [], 0, 0
        for line in body.splitlines():
            head = re.match(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if head:
                b, h = int(head.group(1)), int(head.group(2))
            elif line.startswith(("+++", "---", "\\")):
                continue
            elif line.startswith("-"):
                removed.append(b)
                b += 1
            elif line.startswith("+"):
                added.append(h)
                h += 1
            else:
                b += 1
                h += 1
        per[path] = (removed, added)
    assert per, "the diff carried no file header, so there was nothing to split"
    return per


def _prose_lines(src: str):
    """Line numbers of `src` that are a comment, or sit inside a multi-line string."""
    spans = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            spans.add(tok.start[0])
        elif tok.type == tokenize.STRING and tok.start[0] != tok.end[0]:
            spans.update(range(tok.start[0], tok.end[0] + 1))
    return spans


def _git(*args: str) -> str:
    done = subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True)
    assert done.returncode == 0, f"git {' '.join(args)}: {done.stderr[:200]}"
    return done.stdout


def _phrase_removal_commit(paths):
    """The commit that took the two-word phrase out of `paths`, found by content.

    Graded against a commit located this way rather than against `HEAD^`, because the round's
    own commit is not necessarily HEAD's parent by the time the suite runs again: a rebase or
    a landing merge changes the parent, and a node that then compared against the wrong base
    would either grade an empty diff or grade someone else's change. The pickaxe lists every
    commit whose diff changed the phrase's occurrence count, and the one this node wants is
    the one where the phrase was there before and gone after.
    """
    listed = _git("log", "-S" + DEAD_READER_PHRASE, "--format=%H", "--", *paths)
    for sha in listed.split():
        parent = sha + "^"
        try:
            before = _git("show", f"{parent}:{paths[0]}")
            after = _git("show", f"{sha}:{paths[0]}")
        except AssertionError:
            continue
        if DEAD_READER_PHRASE in _prose_of(before).lower() \
                and DEAD_READER_PHRASE not in _prose_of(after).lower():
            return sha
    return ""


def test_the_rewording_removed_nothing_but_prose_and_touched_no_other_file():
    """Clause 4 as a machine check over the diff rather than a promise in a commit message.

    Every line the rewording DELETED from either file must be a comment line or a line inside
    a multi-line string of the version before it — established by tokenising that version — so
    no assertion, constant or signature can be edited away under cover of a rewording, and
    `SECTION_ROW_CAP` is asserted to be `50` on the live read above. On the added side the
    report script may gain prose only: the change there is comment-only.
    `tests/test_entity_sweep.py` must not appear in that diff at all, because its one hit for
    the phrase is an entity-alias assertion about a real entity name. The node asserts it
    found a rewording commit before it grades one, so a pickaxe that matched nothing cannot
    be reported as a clean diff.
    """
    removal = _phrase_removal_commit(list(PROSE_SURFACES))
    assert removal, (
        "no commit took the two-word phrase out of the two prose surfaces, so there is no "
        "rewording here to grade — either the fix is absent or it moved the phrase somewhere "
        "this node cannot follow")
    base = removal + "^"
    diff = _git("diff", "--unified=0", base, removal, "--", *PROSE_SURFACES,
                ENTITY_SWEEP_FILE)
    # Headers, never a substring over the diff body: this file's own prose cites that
    # filename, so a substring test here would report the citation as a touched path.
    per_file = _diff_by_file(diff)
    touched = set(per_file)
    assert ENTITY_SWEEP_FILE not in touched, (
        "the rewording touched the entity-alias assertion, which names a real entity and is "
        "not this item's prose")
    assert touched and touched <= set(PROSE_SURFACES), sorted(touched - set(PROSE_SURFACES))

    for path in sorted(touched):
        removed, added = per_file[path]
        assert removed, f"{path}: the rewording deletes no line, so that diff is empty"
        spans = _prose_lines(_git("show", f"{base}:{path}"))
        bad = [n for n in removed if n not in spans]
        assert not bad, f"{path}: deleted non-prose line(s) {bad[:6]}"
        if path == "scripts/memory/knowledge-health-report.py":
            head_spans = _prose_lines(_git("show", f"{removal}:{path}"))
            bad2 = [n for n in added if n not in head_spans]
            assert not bad2, (
                "the report script gained non-prose line(s) "
                f"{bad2[:6]}; the change there is comment-only")


# ── #3407: the human-only merge queue is printed, dated, and never an alarm ───
#
# Both members that write merge proposals write them for a person and apply
# nothing: #48's `SUFFIX_SAFE` tier is never passed to `--apply` on schedule, and
# #67 only ever proposes. The count lived only in a per-run record under
# `autonomy-runs/`, on a retention-bounded store nobody reads — so 23 approved
# clusters sat unmentioned, aged 32 days, and were re-presented as fresh.
#
# A cluster is WAITING when a dated plan file holds it with `tier ==
# "SUFFIX_SAFE"` and `status == "SAFE"`. The status filter is load-bearing: on
# the 2026-10-26 tree the tier alone matches 63 clusters, because the 40 the gate
# REFUSED share the tier and differ only in status — and dating from that set
# would date the queue from `python`, first seen 2026-09-22, a refused cluster.
# The approval date exists only in the plan FILENAME: the gate cache
# `semantic-verdicts.jsonl` carries no timestamp field at all.

import datetime as _dt                                                # noqa: E402
from datetime import date as _date                                    # noqa: E402

QUEUE_TODAY = _date(2026, 10, 30)
SUFFIX_OLDEST = "2026-09-24"          # 36 days before QUEUE_TODAY
SUFFIX_AGE = 36
PROPOSAL_OLDEST = "2026-09-23"        # 37 days, and older than any approval
PROPOSAL_AGE = 37
DRY_PLAN = "entity-merges-2026-10-30-20261030T150100Z.jsonl"
# The real apply plan of the same run carries NO marker in its name — only a
# later timestamp and a subset of the rows — which is why newest-of-the-day is
# the wrong file to read.
APPLY_PLAN = "entity-merges-2026-10-30-20261030T150213Z.jsonl"


def _plan_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _sweep_row(canonical, *, tier="SUFFIX_SAFE", status="SAFE"):
    return {"status": status, "norm_key": canonical.lower().replace(" ", ""),
            "canonical": canonical, "tier": tier, "decision": "judged",
            "variants": [[canonical, 1]], "merges": []}


def _proposal_row(canonical, variant, action, first_seen):
    return {"canonical": canonical, "variant": variant, "action": action,
            "verdict": "same", "confidence": 1.0, "cached": False,
            "first_seen_at": f"{first_seen}T02:11:00+00:00",
            "proposed_at": "2026-10-30T02:11:00+00:00",
            "last_seen_at": "2026-10-30T02:11:00+00:00"}


def _queue_store(root, *, proposals=True, extra_latest=False):
    """A memory-graph dir shaped like the live one: two run dates, the older
    holding a cluster that has since dropped out, the newer holding a dry plan
    and the apply plan of the same run."""
    graph = root / "memory-graph"
    graph.mkdir(parents=True, exist_ok=True)
    _plan_rows(graph / "entity-merges-2026-09-24-20260924T180000Z.jsonl", [
        _sweep_row("Alpha"), _sweep_row("Beta"),
        _sweep_row("Gamma"),                       # approved then, gone now
        _sweep_row("Delta", status="AMBIGUOUS"),   # refused, and stays refused
    ])
    _plan_rows(graph / DRY_PLAN, [
        _sweep_row("Alpha"), _sweep_row("Beta"),
        _sweep_row("Epsilon"),                     # approved later than Alpha
        _sweep_row("Delta", status="AMBIGUOUS"),
        _sweep_row("Kappa", tier="PUNCT"),         # applied unattended, not for a human
    ])
    _plan_rows(graph / APPLY_PLAN, [_sweep_row("Kappa", tier="PUNCT")])
    if extra_latest:
        _plan_rows(graph / "entity-merges-latest.jsonl", [_sweep_row("Zeta")])
    if proposals:
        _plan_rows(graph / "semantic-proposals-latest.jsonl", [
            _proposal_row("Gemma", "gemma", "merge", PROPOSAL_OLDEST),
            _proposal_row("Claude", "claude", "merge", "2026-10-21"),
            _proposal_row("Vllm", "vllm", "alias_only", "2026-09-20"),
        ])
    return graph


def _queue(**over):
    q = {"suffix_clusters": 3, "suffix_oldest_date": SUFFIX_OLDEST,
         "suffix_age_days": SUFFIX_AGE, "suffix_refused": 1, "suffix_tier_total": 4,
         "suffix_no_data": None, "merge_proposals": 2, "alias_only": 1,
         "proposals_oldest_date": PROPOSAL_OLDEST, "proposals_age_days": PROPOSAL_AGE,
         "proposals_no_data": None, "plans_scanned": 3,
         "newest_plan_date": "2026-10-30", "measured_on": "2026-10-30"}
    q.update(over)
    return q


def _report_line(report, needle):
    lines = [l for l in report.splitlines() if needle in l]
    assert len(lines) == 1, f"expected exactly one line containing {needle!r}"
    return lines[0]


# -- clause 1: the count is the gate-approved suffix rows, and prints at zero --

def test_the_suffix_count_is_the_safe_rows_of_the_suffix_tier(tmp_path):
    """3 waiting, not the 4 that share the tier and not the PUNCT row either.

    `Delta` is the gate-refused cluster (same tier, status AMBIGUOUS) and `Kappa`
    is a PUNCT cluster #48 applies unattended, so neither is "waiting on a
    person". `Gamma` was approved on 2026-09-24 but is absent from the newest
    run's plan, so it is no longer something a person could apply.
    """
    graph = _queue_store(tmp_path, extra_latest=True)
    q = _kg_hygiene().merge_queue_report(graph, today=QUEUE_TODAY)
    assert q["suffix_clusters"] == 3
    assert q["suffix_refused"] == 1 and q["suffix_tier_total"] == 4
    assert q["suffix_no_data"] is None
    # The `-latest` pointer is not a dated plan: it can be read as the newest
    # document while the run writing it is still going, and `Zeta` is in no
    # dated plan at all.
    assert "zeta" not in str(q)


def test_the_suffix_count_reads_the_dry_plan_not_the_apply_plan_of_the_run(tmp_path):
    """Newest-of-the-day is the apply plan, and it holds no approved suffix row.

    On the live tree 2026-10-30 the newest file is
    `entity-merges-2026-10-30-20261030T150213Z.jsonl`: 109 rows, 0 approved
    suffix clusters, written a minute after the dry plan that holds the 23. A
    newest-of-the-day reading therefore reports a drained queue.
    """
    graph = _queue_store(tmp_path)
    newest = max(graph.glob("entity-merges-*.jsonl"), key=lambda p: p.name)
    assert newest.name == APPLY_PLAN
    assert _jsonl_suffix_safe(newest) == 0
    assert _kg_hygiene().merge_queue_report(graph, today=QUEUE_TODAY)["suffix_clusters"] == 3


def _jsonl_suffix_safe(path):
    return sum(1 for l in path.read_text().splitlines() if l.strip()
               and json.loads(l).get("tier") == "SUFFIX_SAFE"
               and json.loads(l).get("status") == "SAFE")


def test_the_suffix_row_prints_the_count_and_names_the_excluded_refused_set():
    row = _report_line(_report(_hygiene(merge_queue=_queue())),
                       "| Merge proposals left for a human")
    assert "3 gate-approved suffix clusters" in row, row
    assert "1 gate-REFUSED clusters of the 4 sharing the tier are excluded" in row, row


def test_an_empty_suffix_queue_still_prints_a_zero():
    """A drained queue is a real state and must not be a missing row."""
    q = _queue(suffix_clusters=0, suffix_oldest_date=None, suffix_age_days=None,
               suffix_refused=0, suffix_tier_total=0)
    row = _report_line(_report(_hygiene(merge_queue=q)),
                       "| Merge proposals left for a human")
    assert "0 gate-approved suffix clusters" in row, row
    assert "age unknown" in row, row


def test_an_empty_but_readable_queue_is_not_reported_as_no_data(tmp_path):
    graph = tmp_path / "memory-graph"
    graph.mkdir(parents=True)
    _plan_rows(graph / DRY_PLAN, [_sweep_row("Delta", status="AMBIGUOUS")])
    q = _kg_hygiene().merge_queue_report(graph, today=QUEUE_TODAY)
    assert q["suffix_clusters"] == 0
    assert q["suffix_no_data"] is None, "a refused-only plan is readable, not absent"


# -- clause 2: the proposal count is #67's merge rows, alias_only excluded ----

def test_the_proposal_count_is_merge_rows_only(tmp_path):
    graph = _queue_store(tmp_path)
    q = _kg_hygiene().merge_queue_report(graph, today=QUEUE_TODAY)
    assert q["merge_proposals"] == 2, "the alias_only row must not join the count"
    assert q["alias_only"] == 1


def test_the_proposal_row_prints_the_count_and_names_the_excluded_alias_only():
    row = _report_line(_report(_hygiene(merge_queue=_queue())),
                       "| Merge proposals left for a human")
    assert "2 #67 merge proposals" in row, row
    assert "1 `alias_only` proposals excluded" in row, row


# -- clause 3: each number carries the age of its oldest entry ----------------

def test_the_suffix_age_is_the_oldest_dated_plan_holding_a_waiting_cluster(tmp_path):
    """36 days, from the earliest plan holding `Alpha` — not from the newest plan.

    `Gamma`'s 2026-09-24 approval does not count: it is no longer waiting. And
    the date had to come from the filename, because the gate cache that granted
    the approval stores no timestamp.
    """
    q = _kg_hygiene().merge_queue_report(_queue_store(tmp_path), today=QUEUE_TODAY)
    assert q["suffix_oldest_date"] == SUFFIX_OLDEST
    assert q["suffix_age_days"] == SUFFIX_AGE


def test_the_proposal_age_is_the_earliest_first_seen_among_merge_rows(tmp_path):
    """37 days from `first_seen_at`, and the alias_only row older than it is ignored."""
    q = _kg_hygiene().merge_queue_report(_queue_store(tmp_path), today=QUEUE_TODAY)
    assert q["proposals_oldest_date"] == PROPOSAL_OLDEST
    assert q["proposals_age_days"] == PROPOSAL_AGE


def test_an_unparseable_first_seen_loses_its_row_from_the_age_but_not_the_count(tmp_path):
    graph = _queue_store(tmp_path)
    rows = [_proposal_row("Gemma", "gemma", "merge", "not-a-date"),
            _proposal_row("Claude", "claude", "merge", "2026-10-21")]
    _plan_rows(graph / "semantic-proposals-latest.jsonl", rows)
    q = _kg_hygiene().merge_queue_report(graph, today=QUEUE_TODAY)
    assert q["merge_proposals"] == 2
    assert q["proposals_oldest_date"] == "2026-10-21"
    assert q["proposals_age_days"] == 9


def test_the_row_prints_both_ages_beside_their_dates():
    row = _report_line(_report(_hygiene(merge_queue=_queue())),
                       "| Merge proposals left for a human")
    assert f"waiting {SUFFIX_AGE} days (since {SUFFIX_OLDEST})" in row, row
    assert f"waiting {PROPOSAL_AGE} days (since {PROPOSAL_OLDEST})" in row, row


# -- clause 4: reading the queue cannot break the report or trip the alarm ----

def test_an_absent_queue_store_prints_the_row_with_a_no_data_reason(tmp_path):
    q = _kg_hygiene().merge_queue_report(tmp_path / "memory-graph", today=QUEUE_TODAY)
    assert q["suffix_clusters"] == 0 and q["merge_proposals"] == 0
    assert q["suffix_no_data"] == "no dated entity-merges-*.jsonl plan file"
    assert q["proposals_no_data"] == "semantic-proposals-latest.jsonl is absent"
    row = _report_line(_report(_hygiene(merge_queue=q)),
                       "| Merge proposals left for a human")
    assert "no data: no dated entity-merges-*.jsonl plan file" in row, row
    assert "no data: semantic-proposals-latest.jsonl is absent" in row, row


def test_an_empty_plan_and_an_empty_proposal_file_say_which(tmp_path):
    graph = tmp_path / "memory-graph"
    graph.mkdir(parents=True)
    (graph / DRY_PLAN).write_text("", encoding="utf-8")
    (graph / "semantic-proposals-latest.jsonl").write_text("", encoding="utf-8")
    q = _kg_hygiene().merge_queue_report(graph, today=QUEUE_TODAY)
    assert q["suffix_no_data"] == "no readable row in any plan file"
    assert q["proposals_no_data"] == "semantic-proposals-latest.jsonl holds no readable row"


def test_a_torn_plan_line_is_skipped_and_the_section_stands(tmp_path):
    """One corrupt line cannot cost the report its Hygiene section."""
    graph = _queue_store(tmp_path)
    with open(graph / DRY_PLAN, "a", encoding="utf-8") as fh:
        fh.write("{ this line was never closed\n")
    q = _kg_hygiene().merge_queue_report(graph, today=QUEUE_TODAY)
    assert q["suffix_clusters"] == 3 and q["suffix_no_data"] is None
    body = _report(_hygiene(merge_queue=q))
    assert "## Hygiene" in body and "| Provenance coverage" in body


def test_the_merge_queue_is_never_fed_to_the_alarm_channel():
    """A non-empty queue is the DESIGNED state, so it must not exit 2.

    The alarm channel is reserved for counts a job can repair itself: a
    contaminated dir, an untagged share, a duplicate fact ID. `SUFFIX_SAFE` is
    never applied on schedule and #67 never applies at all, so an alarm here
    would fire every night forever on a state nothing can fix, and would teach
    a reader to ignore the channel. A repairable alarm still has to fire beside
    the same queue, which is what the second half of this node pins.
    """
    queue_hygiene = _hygiene(merge_queue=_queue())
    store = {"edges_total": 100, "edges_active": 100}
    assert khr._alarms(store, queue_hygiene, 0, 80) == []
    # Contamination trips its own alarm over the SAME non-empty queue, and the
    # two wordings must not be confused: the repairable alarm's own sentence
    # contains "a merge went wrong", so the filter keys on the queue's terms.
    broken = _hygiene(merge_queue=_queue(), contaminated_dirs=2, foreign_facts=3,
                      contaminated=[("Foo", "Bar", 3)])
    alarms = khr._alarms(store, broken, 0, 80)
    assert len(alarms) == 1 and "2 directories" in alarms[0], alarms
    assert not [a for a in alarms
                if any(t in a.lower() for t in ("suffix", "proposal", "queue"))], alarms


def _run_script(tmp_path, graph, *extra):
    """`main()` as the nightly job runs it, over an ephemeral data root."""
    import subprocess
    from app import kg_store

    root = tmp_path / "facts"
    _facts(root, "vLLM", "state", [("vLLM", "serves openai api")])
    db = tmp_path / "kg.sqlite"
    st = kg_store.configure(db)
    st.facts_idx.reindex(sorted(root.rglob("*.md")), root=root)
    kg_store.reset()

    out = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/memory/knowledge-health-report.py"),
         "--facts-dir", str(root), "--output-dir", str(out),
         "--merge-queue-dir", str(graph), *extra],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, "LLOYD_KG_DB": str(db), "LLOYD_FACTS_ROOT": str(root),
             "LLOYD_DATA": str(tmp_path)})
    dated = next(out.glob("knowledge-health-*.md"))
    run_date = _dt.datetime.strptime(
        dated.name[len("knowledge-health-"):][:10], "%Y-%m-%d").date()
    return proc, dated.read_text(), run_date


def test_a_full_human_queue_leaves_the_nightly_exit_status_at_zero(tmp_path):
    """Clause 4 end to end: the row is in the report the job writes, the number
    is on stdout where an operator reads it, and exit status stays 0 WITHOUT
    `--no-alarm-exit` — the flag is what would hide an alarm, so passing it here
    would pin nothing.

    Ages are read off the report's own run date rather than `QUEUE_TODAY`,
    because this node runs the real `main()`, which dates from the clock: the
    fixture's dates are what the queue holds, the run date is what the script
    measures them against, and a node that hardcoded today's number would rot
    tomorrow.
    """
    proc, report, run_date = _run_script(tmp_path, _queue_store(tmp_path))
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "ALARM" not in proc.stderr, proc.stderr[-1500:]
    row = _report_line(report, "| Merge proposals left for a human")
    suffix_age = (run_date - _dt.date.fromisoformat(SUFFIX_OLDEST)).days
    proposal_age = (run_date - _dt.date.fromisoformat(PROPOSAL_OLDEST)).days
    assert "3 gate-approved suffix clusters" in row
    assert f"waiting {suffix_age} days (since {SUFFIX_OLDEST})" in row, row
    assert f"waiting {proposal_age} days (since {PROPOSAL_OLDEST})" in row, row
    assert "2 #67 merge proposals" in row
    # The same figure on the job's stdout, which is what an operator reads.
    assert ("Merge proposals left for a human: 3 gate-approved suffix clusters"
            in proc.stdout), proc.stdout[-1500:]


# --- #3849: the index measured against the walk it never compared ----------------
#
# Everything above measures the markdown tree. The store on the other side of that
# tree carried 12 fact rows and 43 entity rows the tree cannot show, and the one
# report line that denominates over the index (`Provenance coverage`, printed as
# 100.0% of 191,433) CERTIFIED those rows while every other hygiene metric ignored
# them. These tests are the comparison, in both directions, on one store read.

def _indexed_facts(root: Path, entity: str, category: str, ids: list[str]) -> Path:
    """One fact file written on disk AND indexed, so the two surfaces agree on it.

    The file is the shape `load_entities()` parses, and it reaches `facts_idx` only
    through `update_file` with the same `root` — so an agreement in these fixtures is
    a real agreement between the writer and the reader, and a disagreement is exactly
    the fault being pinned, not an artefact of a hand-made index row.
    """
    d = root / entity
    d.mkdir(parents=True, exist_ok=True)
    body = "".join(
        f'  - id: {i}\n    fact: "fact {i}"\n'
        f"    created_at: '2026-10-01T00:00:00Z'\n"
        f"    source_doc: 'knowledge/x-{i}.md'\n"
        for i in ids)
    p = d / f"{entity}-{category}.md"
    p.write_text("---\ntype: facts\n"
                 f'entity: "{entity}"\ncategory: {category}\n'
                 "source_doc: 'knowledge/x.md'\n"
                 f"facts:\n{body}---\n", encoding="utf-8")
    return p


def _two_entity_tree(tmp_path):
    """A tree the walk parses cleanly: two entity dirs, one fact file each."""
    root = tmp_path / "facts"
    a = _indexed_facts(root, "Alpha", "state", ["st-001", "st-002"])
    b = _indexed_facts(root, "Beta", "skill", ["skl-001"])
    return root, [a, b]


def _walked(root: Path) -> dict:
    """The walk's own output exactly as the report builds it: file -> records."""
    entities = khr.load_entities(root)
    return {entry["file"]: len(entry["facts"])
            for entries in entities.values() for entry in entries}


def _one_deleted_file_one_dirless_entity(tmp_path):
    """Two parsed entity dirs + an indexed file deleted from disk + a registered
    entity that never made a directory. Clause 3's fixture, one build for both tests.
    """
    from app.kg_store import KGStore

    root, files = _two_entity_tree(tmp_path)
    db = KGStore(tmp_path / "kg.sqlite")
    for f in files:
        db.facts_idx.update_file(f, root=root)
        db.entities.register(f.parent.name)
    # An indexed file vanishes. Nothing in the report noticed: `reindex()` would
    # expire the row and the nightly run never calls it.
    gone = _indexed_facts(root, "Gamma", "state", ["st-001"])
    db.facts_idx.update_file(gone, root=root)
    db.entities.register("Gamma")
    gone.unlink()
    # A registry row that never landed a directory. Gamma's DIRECTORY is still there
    # (only the file is unlinked), which is why `unregistered_dirs` stays empty here:
    # Gamma is registered, so a directory the registry lacks is what is absent.
    db.entities.register("Delta Unwritten")
    return root, db


def _unregistered_dir_and_a_short_parsed_file(tmp_path):
    """The two populations every other fixture here pins only at zero.

    - `Epsilon`: an EMPTY directory the registry has never heard of — clause 1's
      REVERSE direction, and the live shape exactly: `Databricks AI` and `Experiments`
      are the two directories with no `entities` row and each holds 0 top-level `.md`.
      Empty, because `facts_idx.update_file` registers the entity it indexes, so any
      indexed directory would be registered by construction and the reverse direction
      could never be non-empty; and an empty one keeps `unwalked_files` at 0, so each
      non-zero field below is the one this fixture set out to make non-zero.
      Without it `unregistered_dirs` is `[]` in every fixture in this file, so an
      implementation that returned `[]` unconditionally, or subtracted the dirs from
      the ghost count, would pass every test.
    - `Zeta`: indexed with three records, then rewritten on disk with two. The index
      still carries three rows for a file the walk now parses at two — the live
      `under_parsed_rows=4`, whose own cause is this report's front-matter reader
      stopping at an interior `---`, reproduced here by the benign route so the test
      pins the COUNT and not that bug.
    """
    from app.kg_store import KGStore

    root = tmp_path / "facts"
    db = KGStore(tmp_path / "kg.sqlite")
    (root / "Epsilon").mkdir(parents=True, exist_ok=True)
    zeta = _indexed_facts(root, "Zeta", "skill", ["sk-001", "sk-002", "sk-003"])
    db.facts_idx.update_file(zeta, root=root)
    db.entities.register("Zeta")      # Epsilon deliberately never registered
    _indexed_facts(root, "Zeta", "skill", ["sk-001", "sk-002"])   # 3 rows, 2 on disk
    return root, db


def test_the_reverse_direction_and_the_short_parse_are_counted_as_themselves(tmp_path):
    """Clause 1's remaining half: the reverse direction comes back as DATA, and the
    surplus living in a file the walk DID open is its own number.

    Every other fixture in this file has `unregistered_dirs == []` and
    `under_parsed_rows == 0`, which is why an implementation that always returned an
    empty list, or that netted directories against ghosts, survived the first review.
    Here one directory has no registry row while zero entities are missing a directory,
    so a netted `entities` would print `-1`, or `0` with the directories unmentioned,
    and a `unregistered_dirs` that was never populated prints nothing.

    `facts` must read 1 from `under_parsed_rows` alone with `unopened_rows` at 0: the
    two halves name different broken things. The 4 of the live 12 that sit in
    short-parsed files are this monitor going blind; the other 8 are the store holding
    rows for files nobody can open — and only the second half falls the count to 8 if
    the front-matter truncation is ever fixed.
    """
    root, db = _unregistered_dir_and_a_short_parsed_file(tmp_path)
    try:
        recon = khr.count_index_ghosts(_walked(root), root, store=db)
        assert recon.get("error") is None, recon
        assert recon["unregistered_dirs"] == ["Epsilon"], (
            f"the directory with no registry row must come back as data: {recon}")
        # Returned beside the ghost count, not folded into it: nothing is missing a
        # directory here, so any subtraction would move this number.
        assert recon["entities"] == 0 and recon["ghost_names"] == [], recon
        assert (recon["under_parsed_rows"], recon["under_parsed_paths"]) == (1, 1), recon
        assert (recon["unopened_rows"], recon["unopened_paths"]) == (0, 0), recon
        assert recon["facts"] == 1, "the surplus is the short parse, and only that"
        assert recon["unwalked_files"] == 0, (
            "Epsilon holds no file for the index to lack; the empty dir is the only "
            "thing about it the registry does not know")
        assert recon["index_short_rows"] == 0, (
            "the walk parses FEWER than the index here; the opposite tilt is a "
            "different tail and must not absorb this one")

        row = _report_line(_report(_hygiene(), index_ghosts=recon),
                           "Index rows with no file")
        assert "**1 fact rows / 0 entity rows**" in row, row
        assert "1 in files the walk parsed short over 1 path(s)" in row, row
        assert "0 in files the walk never opened" in row, row
        assert "1 dir(s) with no registry row: 'Epsilon'" in row, (
            f"the reverse direction must be printed as not-counted, not netted: {row}")
    finally:
        db.close()


def test_the_row_counts_index_rows_the_walk_never_opened_and_dirless_entities(tmp_path):
    """Clause 1: both counts come off ONE walk and ONE store read.

    The store side is built by `facts_idx.update_file` from the very files the walk
    parses, so a zero here means the two surfaces genuinely agree; then one fault per
    population is introduced and each must land in its own count.
    """
    root, db = _one_deleted_file_one_dirless_entity(tmp_path)
    try:
        recon = khr.count_index_ghosts(_walked(root), root, store=db)
        assert recon.get("error") is None, recon
        assert recon["facts"] == 1, f"the deleted file's row must count once: {recon}"
        assert recon["entities"] == 1, f"the dirless entity must count once: {recon}"
        # The attribution, not just a total: which side of the fact fault, which name.
        assert recon["unopened_rows"] == 1 and recon["unopened_paths"] == 1
        assert recon["under_parsed_rows"] == 0 and recon["under_parsed_paths"] == 0
        assert recon["unwalked_files"] == 0, "both surfaces hold the same two files"
        assert recon["ghost_names"] == ["Delta Unwritten"], recon["ghost_names"]
        assert recon["registry_rows"] == 4, "Alpha, Beta, Gamma and Delta are registered"
        # The reverse direction is reported, not dropped: Gamma's directory went with
        # the file, so no directory is left that the registry lacks.
        assert recon["unregistered_dirs"] == [], recon["unregistered_dirs"]
    finally:
        db.close()


def test_the_hygiene_row_prints_zero_and_names_the_failure_instead(tmp_path,
                                                                   monkeypatch):
    """Clause 2: the row is on the page at zero, and an unreadable store prints the
    reason rather than `0`.

    `0` on this row reads as "both surfaces agree" to every reader of the table
    including `_alarms`, so a failed read that printed `0` would be the loudest
    possible false clean. This is `KGStore.store()`'s own #1236 rule, applied to a
    second reader: never hand a caller `0` rows for an absent file.
    """
    from app.kg_store import KGStore, StoreUnavailable

    root, files = _two_entity_tree(tmp_path)
    db = KGStore(tmp_path / "kg.sqlite")
    try:
        for f in files:
            db.facts_idx.update_file(f, root=root)
            db.entities.register(f.parent.name)
        clean = khr.count_index_ghosts(_walked(root), root, store=db)
        assert (clean["facts"], clean["entities"]) == (0, 0), clean
        row = _report_line(_report(_hygiene(), index_ghosts=clean),
                           "Index rows with no file")
        assert "**0 fact rows / 0 entity rows**" in row, row
    finally:
        db.close()

    class _LockedIndex:
        def __init__(self):
            self.file_row_counts = lambda: (_ for _ in ()).throw(
                RuntimeError("database is locked"))

    class _LockedStore:
        facts_idx = _LockedIndex()

        class entities:
            @staticmethod
            def all():
                return []

    broken = khr.count_index_ghosts(_walked(root), root, store=_LockedStore())
    assert broken.get("error") and "database is locked" in broken["error"], broken
    row = _report_line(_report(_hygiene(), index_ghosts=broken),
                       "Index rows with no file")
    assert "not measured" in row and "database is locked" in row, row
    assert "fact rows" not in row, f"a failed read must never print a count: {row}"

    def _absent():
        raise StoreUnavailable("no knowledge-graph database at /x/kg.sqlite")
    monkeypatch.setattr(khr, "_kg_store", _absent)
    missing = khr.count_index_ghosts(_walked(root), root)
    assert "no knowledge-graph database" in missing["error"], missing
    row = _report_line(_report(_hygiene(), index_ghosts=missing),
                       "Index rows with no file")
    assert "not measured" in row and "knowledge-graph database" in row, row
    assert "fact rows" not in row, row
    # And a run that never measured at all (the key absent) still prints the row.
    row = _report_line(_report(_hygiene()), "Index rows with no file")
    assert "not measured" in row, row


def test_one_missing_file_and_one_missing_directory_print_one_and_one(tmp_path):
    """Clause 3: the two populations are counted separately and cannot cancel.

    Exactly the item's fixture — two entity dirs the walk parses, the index holding
    one fact row whose file was deleted, and one entity registered with no directory.
    A single reconciled total would print 0 for one fault of each kind and read as
    clean; a file that vanished and a directory that never existed are different
    faults with different remedies, so the row prints 1 and 1.
    """
    root, db = _one_deleted_file_one_dirless_entity(tmp_path)
    try:
        recon = khr.count_index_ghosts(_walked(root), root, store=db)
        assert (recon["facts"], recon["entities"]) == (1, 1), recon
        row = _report_line(_report(_hygiene(), index_ghosts=recon),
                           "Index rows with no file")
        assert "**1 fact rows / 1 entity rows**" in row, row
        # Neither population may print zero while both hold a fault. Matched on each
        # population's own phrase, because the cell bolds the pair as one run —
        # `"**0**" not in row` could not fire even under a netting implementation.
        assert "0 fact rows" not in row and "/ 0 entity rows" not in row, (
            f"the two faults netted out, which they must never do: {row}")
        # Each half names its own population, so the attributions cannot blur.
        assert "1 in files the walk never opened over 1 path(s)" in row, row
        assert "registry names: 'Delta Unwritten'" in row, row
    finally:
        db.close()


def _ghosts(facts: int, entities: int) -> dict:
    return {"facts": facts, "entities": entities, "error": None,
            "unopened_rows": facts, "unopened_paths": 1 if facts else 0,
            "under_parsed_rows": 0, "under_parsed_paths": 0,
            "index_short_rows": 0, "index_short_paths": 0,
            "unwalked_files": 0, "unregistered_dirs": [],
            "registry_rows": entities, "ghost_names": []}


def test_the_ghost_alarm_fires_only_above_the_named_floor():
    """Clause 4: `GHOST_INDEX_ROWS_ALARM_FLOOR = 60` gates the alarm.

    Today's live readings are 12 fact rows and 43 entity rows, so the nightly exit
    status on the day this lands is exactly what it was before — an alarm that fired
    on the residue would be noise from day one, and a number nobody can act on
    overnight is a footnote with a bell attached. Above the floor the line names WHICH
    population drifted and how many, because an orphaned fact row (a vanished file, or
    a scratch writer that reached the live store) and an orphaned entity row (a
    registration that never landed) are not the same repair.
    """
    assert khr.GHOST_INDEX_ROWS_ALARM_FLOOR == 60
    store = {"edges_active": 53416}

    # Today's tree: silent, both populations, and at the floor exactly.
    assert khr._alarms(store, _hygiene(), 0, 53416, _ghosts(12, 43)) == []
    assert khr._alarms(store, _hygiene(), 0, 53416, _ghosts(60, 60)) == []

    over_facts = khr._alarms(store, _hygiene(), 0, 53416, _ghosts(61, 12))
    assert len(over_facts) == 1, over_facts
    assert "61 facts_idx rows" in over_facts[0], over_facts[0]
    assert "no directory" not in over_facts[0], (
        f"the entity population is at 12, below the floor, so its own line must not "
        f"appear: {over_facts[0]}")

    over_entities = khr._alarms(store, _hygiene(), 0, 53416, _ghosts(12, 61))
    assert len(over_entities) == 1, over_entities
    assert "61 `entities` rows" in over_entities[0], over_entities[0]
    assert "no directory" in over_entities[0], over_entities[0]

    both = khr._alarms(store, _hygiene(), 0, 53416, _ghosts(61, 121))
    assert len(both) == 2, both
    assert any("61 facts_idx rows" in a for a in both), both
    assert any("121 `entities` rows" in a for a in both), both

    # An unmeasured population cannot alarm: a row that prints `not measured` must
    # not also hand the run a clean exit code.
    assert khr._alarms(store, _hygiene(), 0, 53416,
                       {"error": "store unavailable: no such file"}) == []
    assert khr._alarms(store, _hygiene(), 0, 53416, None) == []
    # And the four-argument call every existing test uses is unaffected.
    assert khr._alarms(store, _hygiene(), 0, 53416) == []
