"""kg_hygiene.py — contamination, near-duplicate clusters, regrowth.

Pins the measurements the 2026-09-03 audit computed by hand: 63 directories
held facts about another entity, every one from a suffix merge.
"""
import importlib.util
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "memory"))
import kg_hygiene  # noqa: E402
from app.kg_store import parse_fact_file


def _fact_file(root: Path, dirname: str, category: str, facts: list[tuple[str, str]],
               entity: str | None = None) -> Path:
    d = root / dirname
    d.mkdir(parents=True, exist_ok=True)
    fm = {"type": "facts", "entity": entity or dirname, "category": category,
          "facts": [{"entity": e, "fact": t, "confidence": 0.9, "category": category} for e, t in facts]}
    p = d / f"{dirname}-{category}.md"
    p.write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# {dirname} - {category}\n")
    return p


def _overview(root: Path, dirname: str, entity: str, definition: str) -> Path:
    d = root / dirname
    d.mkdir(parents=True, exist_ok=True)
    fm = {"type": "overview", "entity": entity, "category": "overview", "definition": definition}
    p = d / f"{dirname}-overview.md"
    p.write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# Summary\n\n{definition}\n")
    return p


def _dirs(root: Path, *names: str) -> Path:
    """Entity directories with one fact file each, and the root they live in."""
    for n in names:
        _fact_file(root, n, "state", [(n, f"{n} exists.")])
    return root


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "facts"
    # Intel holds its own facts plus two that belong to the pipeline (the 12:32Z merge)
    _fact_file(root, "Intel", "state", [("Intel", "Intel released the Pro B70 GPU."),
                                         ("Intel Pipeline System", "Scans ArXiv nightly."),
                                         ("Intel Pipeline System", "Scores Hacker News items.")])
    _overview(root, "Intel", "Intel", "Intel is a semiconductor company.")
    # case-only duplicate pair
    _fact_file(root, "vLLM V1 Engine", "state", [("vLLM V1 Engine", "Uses continuous batching.")])
    _fact_file(root, "vLLM V1 engine", "state", [("vLLM V1 engine", "Replaces V0.")])
    # clean
    _fact_file(root, "Alfie", "state", [("Alfie", "Alfie is a humanoid robot.")])
    # ignored bookkeeping
    (root / "_relationships.json").write_text('{"edges": []}')
    return root


def test_contamination_finds_only_the_foreign_facts(tree):
    c = kg_hygiene.contamination(tree)
    assert c["dirs"] == 1
    assert c["foreign_facts"] == 2
    assert c["by_tier"] == {"SUFFIX_SAFE": 1}
    item = c["items"][0]
    assert item["dir"] == "Intel"
    assert list(item["foreign"]) == ["Intel Pipeline System"]
    assert item["foreign"]["Intel Pipeline System"]["files"] == ["Intel-state.md"]


def test_case_variants_are_not_contamination(tree):
    # the overview's entity tag equals the dir name; the vLLM pair differ only by case
    c = kg_hygiene.contamination(tree)
    assert all(it["dir"] == "Intel" for it in c["items"])


def test_near_duplicates_cluster_by_normalised_name(tree):
    n = kg_hygiene.near_duplicates(tree)
    assert n["clusters"] == 1
    assert n["dirs"] == 2
    assert n["by_tier"] == {"SAFE": 1}
    assert sorted(n["samples"][0]) == ["vLLM V1 Engine", "vLLM V1 engine"]


def test_regrowth_counts_only_dirs_created_after_the_baseline(tmp_path):
    """Clause 1: a stored baseline of the entity-dir set, one directory created
    afterwards, and the field reports that ONE directory and names it.

    Before #1535 this counted whatever `created_at` fell inside a 7-day window,
    so the number it returned was a function of the last rebuild's date rather
    than of anything that had been created."""
    root = _dirs(tmp_path / "facts", "Intel", "vLLM V1 Engine", "Alfie")
    base = tmp_path / "baseline.json"
    rec = kg_hygiene.write_baseline(root, base)
    assert rec["dirs"] == 3

    _fact_file(root, "vLLM V1 engine", "state", [("vLLM V1 engine", "Replaces V0.")])
    r = kg_hygiene.regrowth(root, days=7, baseline_path=base)
    assert r["new_dirs"] == 1
    assert r["samples"] == ["vLLM V1 engine"]
    assert r["near_dup_new"] == 1 and r["by_tier"] == {"CASE": 1}
    assert r["dirs_at_baseline"] == 3
    assert r["baseline_at"] == rec["captured_at"]
    assert r["skipped_vanished"] == 0
    assert r["no_baseline_reason"] is None


def test_regrowth_counts_a_non_duplicate_new_dir_in_the_denominator_only(tmp_path):
    """A new entity that is nobody's twin is growth (`new_dirs`) and not
    regrowth (`near_dup_new`). `samples` is the pair list the report renders as
    "`X` next to `Y`", so it names the regrown ones, not every new directory."""
    root = _dirs(tmp_path / "facts", "Intel", "Alfie")
    base = tmp_path / "baseline.json"
    kg_hygiene.write_baseline(root, base)
    _fact_file(root, "Gemma 4", "state", [("Gemma 4", "Open weights.")])
    r = kg_hygiene.regrowth(root, days=7, baseline_path=base)
    assert r["new_dirs"] == 1
    assert r["near_dup_new"] == 0
    assert r["samples"] == [] and r["by_tier"] == {}


def test_regrowth_without_a_baseline_reports_none_not_the_store_size(tmp_path):
    """Clause 2. Every kg_health snapshot taken after the 2026-09-23 rebuild
    printed `new_dirs` == `entities.count` — 11,959 of 11,959 at 02:54Z and
    12,027 of 12,027 at 07:27Z — because the rebuild re-dated every fact and a
    7-day window over those stamps necessarily covers the tree. With no usable
    reference the section abstains; it never returns a directory count."""
    root = _dirs(tmp_path / "facts", "Intel", "vLLM V1 Engine", "vLLM V1 engine", "Alfie")
    r = kg_hygiene.regrowth(root, days=7, baseline_path=tmp_path / "absent.json")
    assert r["new_dirs"] is None
    assert r["near_dup_new"] is None
    assert "absent.json" in r["no_baseline_reason"]
    assert r["dirs_at_baseline"] is None and r["baseline_at"] is None
    assert r["samples"] == [] and r["by_tier"] == {}
    assert r["new_dirs"] != len(kg_hygiene.iter_entity_dirs(root))   # the old shape


def test_regrowth_refuses_a_baseline_taken_over_a_different_tree(tmp_path):
    """`knowledge-health-report.py --facts-dir` measures whichever tree it is
    handed. Diffing it against a reference taken over another one would print a
    confident wrong number, which is the class of thing this section was
    rewritten to stop printing."""
    real = _dirs(tmp_path / "facts", "Intel", "Alfie")
    other = _dirs(tmp_path / "elsewhere", "QMD", "SGLang")
    base = tmp_path / "baseline.json"
    kg_hygiene.write_baseline(other, base)
    r = kg_hygiene.regrowth(real, days=7, baseline_path=base)
    assert r["new_dirs"] is None
    assert "elsewhere" in r["no_baseline_reason"]


def test_regrowth_refuses_a_truncated_baseline(tmp_path):
    """`write_baseline` writes by atomic replace for exactly this reason, but a
    half-written file on disk must still read as "no reference", not as "almost
    no directories existed, so everything is new"."""
    root = _dirs(tmp_path / "facts", "Intel")
    base = tmp_path / "baseline.json"
    base.write_text('{"schema": 1, "dirs": ["Int')
    r = kg_hygiene.regrowth(root, days=7, baseline_path=base)
    assert r["new_dirs"] is None
    assert "unreadable" in r["no_baseline_reason"]


def test_regrowth_survives_a_rebuild_that_reset_every_created_at(tmp_path):
    """The falsifying event is a rebuild, not a date. The 2026-09-23 rebuild
    re-derived every fact's `created_at` and rewrote every file, so all 12,027
    directories were "born in the last 7 days" and the field could not have read
    any other way; a stored name set is not re-datable, so one directory
    created after the baseline is still the only thing reported."""
    root = _dirs(tmp_path / "facts", "vLLM V1 Engine", "Intel", "Alfie")
    base = tmp_path / "baseline.json"
    kg_hygiene.write_baseline(root, base)

    fresh = dt.datetime.now(dt.timezone.utc).isoformat()
    for d in sorted(root.iterdir()):
        for p in d.glob("*.md"):
            fm = yaml.safe_load(p.read_text().split("---")[1])
            for f in fm["facts"]:
                f["created_at"] = fresh
            p.write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\nbody\n")
            os.utime(p, (time.time(),) * 2)          # a rebuild rewrites the files too

    _fact_file(root, "vLLM V1 engine", "state", [("vLLM V1 engine", "Replaces V0.")])
    r = kg_hygiene.regrowth(root, days=7, baseline_path=base)
    assert r["new_dirs"] == 1
    assert r["near_dup_new"] == 1
    assert r["dirs_at_baseline"] == 3


def test_near_dup_new_needs_the_older_name_to_predate_the_baseline(tmp_path):
    """Clause 4. A case variant of a directory that was already there is
    regrowth. Two twins that BOTH appeared after the baseline are two new
    entities and neither is the older one, so the count stays at zero — the
    old window rule called that pair regrowth by comparing their stamps."""
    root = _dirs(tmp_path / "facts", "vLLM V1 Engine")
    base = tmp_path / "baseline.json"
    kg_hygiene.write_baseline(root, base)
    _fact_file(root, "vLLM V1 engine", "state", [("vLLM V1 engine", "Replaces V0.")])
    r = kg_hygiene.regrowth(root, days=7, baseline_path=base)
    assert (r["new_dirs"], r["near_dup_new"]) == (1, 1)
    assert r["samples"] == ["vLLM V1 engine"]
    assert r["by_tier"] == {"CASE": 1}

    both = _dirs(tmp_path / "fresh", "Intel")
    base2 = tmp_path / "baseline2.json"
    kg_hygiene.write_baseline(both, base2)
    _dirs(both, "vLLM V1 Engine", "vLLM V1 engine")
    r2 = kg_hygiene.regrowth(both, days=7, baseline_path=base2)
    assert (r2["new_dirs"], r2["near_dup_new"]) == (2, 0)
    assert r2["samples"] == [] and r2["by_tier"] == {}


def test_describe_always_puts_the_reference_beside_the_number():
    """Clause 5's shared phrasing, including the historic files: every snapshot
    on disk under `_pipeline/metrics/kg-health-*.json` has a regrowth section
    written before the baseline existed (`{"days": 7, "new_dirs": 11959,
    "near_dup_new": 51, ...}`), and a reader of one of those is a reader too.
    It still renders, and says its number has no recorded reference."""
    assert kg_hygiene.describe(
        {"days": 7, "new_dirs": 11959, "near_dup_new": 51,
         "by_tier": {}, "samples": [], "skipped_vanished": 0}) == (
        "51 of 11,959 new dirs (no baseline recorded — this snapshot predates "
        "the entity-dir baseline (#1535))")
    assert kg_hygiene.describe(
        {"new_dirs": 2, "near_dup_new": 1, "baseline_at": "2026-09-26T07:27:00+00:00",
         "dirs_at_baseline": 12027}) == (
        "1 of 2 new dirs (baseline 2026-09-26T07:27:00+00:00 over 12,027 dirs)")
    assert kg_hygiene.describe(
        {"new_dirs": None, "no_baseline_reason": "no baseline file at /tmp/x.json"}) == (
        "not measured: no baseline file at /tmp/x.json")


def test_snapshot_has_all_four_sections(tree):
    s = kg_hygiene.snapshot(tree, days=7)
    assert set(s) >= {"contamination", "untagged", "near_duplicates", "regrowth",
                      "captured_at"}
    assert "items" not in s["contamination"]      # summary only
    assert s["contamination"]["dirs"] == 1


def test_missing_root_is_empty_not_an_error(tmp_path):
    s = kg_hygiene.snapshot(tmp_path / "nope", days=7)
    assert s["contamination"]["dirs"] == 0
    assert s["near_duplicates"]["clusters"] == 0
    # Zero records parsed is the one input where a share has no denominator: it
    # must read as nothing untagged, not as ZeroDivisionError and not as a
    # division that quietly produced 100%.
    assert s["untagged"] == {"facts": 0, "dirs": 0, "files": 0, "records": 0,
                             "share_pct": 0.0,
                             "floor_pct": kg_hygiene.UNTAGGED_ALARM_SHARE,
                             "over_floor": False}


def test_regrowth_survives_a_dir_renamed_mid_pass(tree, tmp_path, monkeypatch):
    """#1404: the entity sweep renamed a dir between the listing and the check,
    and the health report died with FileNotFoundError. A dir that goes missing
    after being listed is skipped out of `new_dirs` and counted, not fatal."""
    base = tmp_path / "baseline.json"
    kg_hygiene.write_baseline(tree, base)
    listed = kg_hygiene.iter_entity_dirs(tree)
    ghost = tree / "World Action Models"
    monkeypatch.setattr(kg_hygiene, "iter_entity_dirs", lambda root: listed + [ghost])
    assert not ghost.exists()
    r = kg_hygiene.regrowth(tree, days=7, baseline_path=base)
    assert r["skipped_vanished"] == 1
    assert r["new_dirs"] == 0
    assert "World Action Models" not in r["samples"]


def test_regrowth_reports_zero_vanished_on_a_quiet_tree(tree, tmp_path):
    base = tmp_path / "baseline.json"
    kg_hygiene.write_baseline(tree, base)
    r = kg_hygiene.regrowth(tree, days=7, baseline_path=base)
    assert r["skipped_vanished"] == 0
    assert r["new_dirs"] == 0                      # nothing was created since


# ── untagged records: the third state (#2474) ────────────────────────────────

def _raw_fact_file(root: Path, dirname: str, category: str, records: list[dict],
                   entity: str | None = None) -> Path:
    """A fact file whose records are written exactly as given.

    `_fact_file` above tags every record it is handed, which is why this defect
    had no fixture: no builder in this file could produce the two shapes the live
    corpus actually holds — `entity: ''` (2,507 records on 2026-10-09) and the
    key absent entirely (1,024 of them).
    """
    d = root / dirname
    d.mkdir(parents=True, exist_ok=True)
    fm = {"type": "facts", "entity": entity or dirname, "category": category,
          "facts": records}
    p = d / f"{dirname}-{category}.md"
    p.write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# {dirname} - {category}\n")
    return p


@pytest.fixture
def blank_tree(tmp_path):
    """The corpus shape #2474 was filed about: untagged records mixed into dirs
    that also hold correct ones, including one dir with a genuine wrong merge.

    10 records parse here. 6 are untagged — 3 blank strings and 1 missing key and
    1 whitespace-only in `general`, 1 blank in `Intel` — spread over 2 dirs and 2
    files, so the share is 60.0%. Contamination is unchanged by all of that: 1
    dir (`Intel`), 1 foreign fact, tier SUFFIX_SAFE.
    """
    root = tmp_path / "facts"
    _raw_fact_file(root, "general", "state", [
        {"entity": "", "fact": "a how-to-fail book"},
        {"entity": "", "fact": "uptake_probe field names"},
        {"entity": "", "fact": "vault redirect routes"},
        {"fact": "the key was never written at all"},
        {"entity": "   ", "fact": "whitespace where a name belongs"},
        {"entity": "general", "fact": "the one record that did get tagged"},
    ])
    _raw_fact_file(root, "Intel", "state", [
        {"entity": "Intel", "fact": "released the Pro B70 GPU."},
        {"entity": "Intel Pipeline System", "fact": "Scans ArXiv nightly."},
        {"entity": "", "fact": "left untagged by the same run"},
    ])
    _fact_file(root, "Alfie", "state", [("Alfie", "Alfie is a humanoid robot.")])
    return root


def test_untagged_records_are_counted_as_their_own_state(blank_tree):
    """Blank, whitespace-only and missing-key tags all count; the number carries
    the denominator it is a share of."""
    u = kg_hygiene.scan(blank_tree)["untagged"]
    assert u["facts"] == 6 and u["dirs"] == 2 and u["files"] == 2
    assert u["records"] == 10, "the denominator is the records this walk parsed"
    assert u["share_pct"] == 60.0
    assert u["floor_pct"] == kg_hygiene.UNTAGGED_ALARM_SHARE
    assert u["over_floor"] is True


def test_an_untagged_record_is_never_foreign(blank_tree):
    """Clause: `foreign_facts_in_dir` hands out only the foreign records.

    Every consumer of that dict reads a key as a wrong merge — `kg_rebuild`
    refuses a rebuild on `contamination()["dirs"]`, and the report's alarm says
    "a merge went wrong" — so a blank tag must not appear in it. `general` holds
    five untagged records and one correct one, and is not contaminated at all.
    """
    assert kg_hygiene.foreign_facts_in_dir(blank_tree / "general") == {}
    intel = kg_hygiene.foreign_facts_in_dir(blank_tree / "Intel")
    assert list(intel) == ["Intel Pipeline System"]
    assert intel["Intel Pipeline System"]["facts"] == 1
    c = kg_hygiene.contamination(blank_tree)
    assert c["dirs"] == 1 and c["foreign_facts"] == 1
    assert c["by_tier"] == {"SUFFIX_SAFE": 1}
    assert "untagged" not in c, ("the foreign-only view must not grow the second "
                                 "tally into the dict whose consumers mean 'merge'")


def test_adding_untagged_records_leaves_contamination_exactly_where_it_was(tmp_path):
    """Both sides of the foreign check stay shut to an untagged record.

    Same tree, measured before and after blank records are appended to a dir that
    already holds a real foreign fact: contamination must be byte-identical while
    the untagged tally grows. The defect was the reverse of this — the same
    records moved nothing at all, in either column.
    """
    root = tmp_path / "facts"
    _raw_fact_file(root, "Intel", "state", [
        {"entity": "Intel", "fact": "released the Pro B70 GPU."},
        {"entity": "Intel Pipeline System", "fact": "Scans ArXiv nightly."},
    ])
    before_c = kg_hygiene.contamination(root)
    before_u = kg_hygiene.scan(root)["untagged"]
    assert before_u == {"facts": 0, "dirs": 0, "files": 0, "records": 2,
                        "share_pct": 0.0,
                        "floor_pct": kg_hygiene.UNTAGGED_ALARM_SHARE,
                        "over_floor": False}

    _raw_fact_file(root, "Intel", "state", [
        {"entity": "Intel", "fact": "released the Pro B70 GPU."},
        {"entity": "Intel Pipeline System", "fact": "Scans ArXiv nightly."},
        {"entity": "", "fact": "blank"},
        {"fact": "no key"},
        {"entity": "\t ", "fact": "whitespace only"},
    ])
    after_c = kg_hygiene.contamination(root)
    after_u = kg_hygiene.scan(root)["untagged"]
    assert after_c == before_c, (before_c, after_c)
    assert after_u["facts"] == 3 and after_u["dirs"] == 1 and after_u["files"] == 1
    assert after_u["records"] == 5, "2 tagged + 3 untagged, all of them counted"
    assert after_u["share_pct"] == 60.0


def test_the_floor_is_a_share_of_records_not_a_count_of_them(tmp_path):
    """`UNTAGGED_ALARM_SHARE` is compared strictly, against a percentage.

    1,000 records in one dir: 5 untagged is exactly 0.5%, which is AT the floor
    and so not over it; 6 is 0.6% and is. An absolute-count floor would lose
    strictness every time extraction grew the corpus, which is the growth being
    watched, and a `>=` here would alarm at the floor rather than above it.
    """
    def share_of(blank: int, tagged: int) -> dict:
        root = tmp_path / f"facts-{blank}"
        recs = [{"entity": "", "fact": f"blank {i}"} for i in range(blank)]
        recs += [{"entity": "Lloyd", "fact": f"tagged {i}"} for i in range(tagged)]
        _raw_fact_file(root, "Lloyd", "state", recs)
        return kg_hygiene.scan(root)["untagged"]

    at = share_of(5, 995)
    assert at["share_pct"] == 0.5 and at["over_floor"] is False
    above = share_of(6, 994)
    assert above["share_pct"] == 0.6 and above["over_floor"] is True


def test_snapshot_reports_untagged_beside_contamination_not_inside_it(blank_tree):
    """kg_health and the knowledge report both read `snapshot()`; the untagged
    tally has to reach them without entering the dict that means "a merge went
    wrong", and the contamination half must keep exactly its old keys."""
    s = kg_hygiene.snapshot(blank_tree, days=7)
    assert s["untagged"]["facts"] == 6 and s["untagged"]["over_floor"] is True
    assert s["contamination"] == {"dirs": 1, "foreign_facts": 1,
                                  "by_tier": {"SUFFIX_SAFE": 1}}


# ── #2503: what the standard writer contributes to that numerator ────────────
#
# `agent_mcp/facts.py::_fact_add` — the `fact_add` MCP tool and, through
# `app/post_capture._write_extracted_facts`, the nightly session extractor — built
# its row dict without an `entity` key: `{fact, confidence, category, id,
# created_at, valid_at, invalid_at, expired_at, provenance, source_doc}`.
# `app/kg_store.py` then indexed the row under `fm.get("entity") or
# path.parent.name`, so the store reported the entity and the gap stayed invisible
# from the query side, while `_tag_of` above read the per-record key and counted
# the record untagged. Measured on the live corpus on 2026-10-10: 3,544 untagged of
# 144,250 records (2.46%, 534 dirs), of which 1,036 have the KEY ABSENT and 463 of
# those were created on or after 2026-10-02 — i.e. written after #1999 had already
# stopped the blank-valued producer. Those 463 are this writer's own output, and
# they are the half of the population a code change can remove. The ~2,509 frozen
# blank-valued rows from the 2026-09-23 re-derivation are not, so the rail is
# expected to stay red after this: that ruling is owed on #2503, and the two nodes
# below deliberately do not make it go quiet.

_FM_BLOCK = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def _untag_every_record(path: Path) -> None:
    """Rewrite a fact file with `entity` deleted from every RECORD.

    The file-level `entity:` key stays: `app/kg_store.py::_rows_for_file` reads
    `fm.get("entity") or path.parent.name`, so with the top key gone the store would
    stop indexing the file at all and the control would be measuring a different
    failure. This is the exact byte shape `_fact_add` emitted before #2503 — every
    record valid YAML, none of them carrying a tag — which is what makes it a
    control rather than a second assertion of the fix.
    """
    text = path.read_text(encoding="utf-8")
    m = _FM_BLOCK.match(text)
    assert m, f"{path} carries no front matter to strip"
    fm = yaml.safe_load(m.group(1))
    records = fm.get("facts") or []
    assert records, f"{path} holds no records to strip"
    fm["facts"] = [{k: v for k, v in r.items() if k != "entity"} for r in records]
    path.write_text(
        "---\n" + yaml.dump(fm, default_flow_style=False, sort_keys=False)
        + "---\n" + text[m.end():], encoding="utf-8")


@pytest.fixture
def writer(tmp_path, monkeypatch):
    """The normal fact-writing path, aimed at a throwaway facts root.

    Returns the live `agent_mcp.facts` module and the root, so a test can call
    `facts._fact_add(...)` and then point `scan_entity_dir` at the directory that
    landed there. Same seams `tests/test_fact_add_entity_kind.py` patches — the
    writer resolves `FACTS_ROOT` through three modules, and the entity-identity
    gate installs its declared keys into the first store it sees.
    """
    from agent_mcp import _shared, facts, retrieval
    from app import entity_naming, kg_store

    root = tmp_path / "facts"
    root.mkdir()
    kg_store.configure(tmp_path / "kg.sqlite")
    entity_naming.reset_identity_schema_cache()
    monkeypatch.setattr(_shared, "FACTS_ROOT", root)
    monkeypatch.setattr(_shared, "ALIASES_PATH", root / "entity-aliases.json")
    monkeypatch.setattr(_shared, "_entity_dirs_cache", None)
    monkeypatch.setattr(facts, "FACTS_ROOT", root)
    monkeypatch.setattr(retrieval, "FACTS_ROOT", root)
    # The paraphrase gate needs the decision engine and has nothing to do with the
    # row's shape; the writer's own behaviour is what is under test here.
    monkeypatch.setenv("LLOYD_FACT_WRITE_GATE", "off")
    try:
        yield facts, root
    finally:
        entity_naming.reset_identity_schema_cache()
        kg_store.reset()


def test_a_file_written_by_the_normal_path_adds_no_untagged_record(writer):
    """#2503 clause 3: the writer's own rows are what the numerator loses.

    Three records, two entities, two categories, all through `_fact_add`: the walk
    sees them, counts none of them untagged, and calls neither directory
    contaminated — a row tagged with its own entity must not read as somebody
    else's fact, which is the worse failure since `kg_rebuild` refuses a rebuild
    on contamination. The control at the end is what makes the zero mean something:
    strip the key out of the file the writer just produced, the exact bytes it
    emitted before #2503, and the same walk has to count all three again. Without
    it this node would pass on a walk that had stopped looking.
    """
    facts, root = writer
    assert facts._fact_add({"entity": "Tidewell Relay", "category": "state",
                            "fact": "pins its control port at 9453"}).get("success")
    assert facts._fact_add({"entity": "Tidewell Relay", "category": "event",
                            "fact": "rolled to 1.4.2 on 2026-10-09"}).get("success")
    assert facts._fact_add({"entity": "Quillgate", "category": "state",
                            "fact": "reindexes nightly at 03:00 local"}).get("success")

    d = root / "Tidewell Relay"
    foreign, untagged, records = kg_hygiene.scan_entity_dir(d)
    assert records == 2, records
    assert untagged == {"facts": 0, "files": []}, untagged
    assert foreign == {}, foreign

    whole = kg_hygiene.scan(root)["untagged"]
    assert whole["facts"] == 0 and whole["records"] == 3, whole

    for name in ("Tidewell Relay", "Quillgate"):
        for f in (root / name).glob("*.md"):
            _untag_every_record(f)
    after = kg_hygiene.scan(root)["untagged"]
    assert after["facts"] == 3, after
    assert after["records"] == 3, "all three still parse; only the tag is gone"
    assert after["over_floor"] is True, (
        "three untagged records of three is 100%, so a rail that calls this quiet "
        "is not measuring what #2474 shipped it to measure")


def test_the_rail_still_counts_a_blank_tag_and_the_floor_is_still_half_a_percent(writer):
    """#2503 clause 4: the fix is a writer fix, not a redefinition of the check.

    A tagged row written by the normal path sits in the tree beside a directory
    holding 6 blank-valued records and 4 correct ones — the frozen 2026-09-23
    shape, unchanged — and the walk must still report 6 untagged of 11 records
    (54.55%), over the floor, while `UNTAGGED_ALARM_SHARE` is still the literal
    0.5 it shipped as in `f7e08d41`. Pinned as a literal rather than against the
    constant, because the constant is what a later round would be tempted to move:
    raising it silences tonight's alarm by redefining the measurement, which is the
    one outcome this item rules out. The population this node counts is exactly the
    one the writer fix cannot touch.
    """
    facts, root = writer
    assert facts._fact_add({"entity": "Tidewell Relay", "category": "state",
                            "fact": "pins its control port at 9453"}).get("success")
    _raw_fact_file(root, "general", "state", [
        {"entity": "", "fact": f"blank {i}"} for i in range(6)
    ] + [
        {"entity": "general", "fact": f"tagged {i}"} for i in range(4)
    ])

    assert kg_hygiene.UNTAGGED_ALARM_SHARE == 0.5, (
        "the floor moved; #2503's ruling is to re-tag or drop the frozen records, "
        "not to raise the number that decides whether they are an alarm")

    u = kg_hygiene.scan(root)["untagged"]
    assert u["facts"] == 6, u
    assert u["records"] == 11, "the writer's own record is in the denominator too"
    assert u["files"] == 1 and u["dirs"] == 1, (
        "the blank directory is the only one counted; the normal path added nothing")
    assert u["share_pct"] == round(600.0 / 11, 2), u
    assert u["floor_pct"] == 0.5
    assert u["over_floor"] is True, (
        "the alarm has to stay red over the frozen population — the expected state "
        "after this fix, per #2503")
    assert kg_hygiene.scan_entity_dir(root / "general")[1]["facts"] == 6


# ── #2676 clause 5: the repair declares nothing the rail does not already count ──

def test_the_repair_agrees_with_the_rail_record_for_record_and_the_floor_holds(
        blank_tree):
    """The one clause that makes the quiet alarm mean something.

    An alarm silenced by narrowing what counts as untagged is worse than a red one:
    the rail would keep reporting a clean corpus while the records that started the
    incident are still there. So this pins both halves in one place — the threshold
    still `UNTAGGED_ALARM_SHARE = 0.5` and the three states #2474 shipped
    (key absent, value blank, value whitespace-only) still all count — and then
    checks the NEW consumer against the rail rather than beside it:
    `retag_untagged_facts` may only ever plan a record this scanner already counts.

    The agreement is per record, over a tree built for the disagreement case: the
    fixture holds all three untagged shapes AND a tagged record in the same files, so
    a predicate that drifted one way or the other shows up as a set difference here
    instead of as a share that happens to land on the same number.
    """
    from scripts.memory import retag_untagged_facts as retag

    assert kg_hygiene.UNTAGGED_ALARM_SHARE == 0.5, (
        "the number moved: #2676's ruling is to repair the records, not to raise "
        "the threshold that decides whether they are an alarm")

    root = blank_tree
    u = kg_hygiene.scan(root=root)["untagged"]
    assert (u["facts"], u["records"], u["floor_pct"], u["over_floor"]) == (
        6, 10, 0.5, True), u

    rail, repair = [], []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.md")):
            _fm, recs = parse_fact_file(f)
            for rec in recs:
                if kg_hygiene._tag_of(rec) == "":
                    rail.append((str(f), rec.get("fact")))
                if retag.record_is_untagged(rec):
                    repair.append((str(f), rec.get("fact")))
    assert rail == repair, (set(rail) ^ set(repair))
    assert len(rail) == 6, rail


# ---------------------------------------------------------------------------
# #4228 clauses 1 and 2 — the waiting SUFFIX queue, listed and packeted
# ---------------------------------------------------------------------------

TODAY = dt.date(2026, 11, 17)

#: The apply form a person runs. Stated as a literal here, on purpose: a packet that
#: silently followed a renamed constant would stop telling the reviewer how to drain
#: the queue, and #4197 (3ead517a) is what made naming the tier the only route.
APPLY_FORM = "--apply --tiers SUFFIX_SAFE"


def _suffix_row(canonical: str, norm_key: str, variants: list[tuple[str, int]],
                status: str = "SAFE", tier: str = "SUFFIX_SAFE",
                decision: str = "SUFFIX_JUDGED — every judge: SAME",
                reason: str = "Both refer to the same thing by two names.") -> dict:
    """One plan row shaped like a real `entity-merges-*.jsonl` sweep record.

    `variants` is `[name, degree]` pairs and `gate` maps a variant to its decision
    plus the per-judge `reason`, because those are the shapes the live plans carry
    (measured on the 2026-11-17 plans: `Hermes` carries `[["Hermes", 68], ["Hermes
    Agent", 26]]` and a reason naming Nous Research). A fixture built on invented
    shapes would pass against a packet that could not read the real files.
    """
    return {"canonical": canonical, "norm_key": norm_key,
            "variants": [list(v) for v in variants], "tier": tier,
            "status": status, "decision": decision,
            "gate": {name: {"decision": "SAME" if status == "SAFE" else "DIFFERENT",
                            "judges": {"j1": {"verdict": ("SAME" if status == "SAFE"
                                                          else "DIFFERENT"),
                                              "reason": reason}}}
                     for name, _ in variants}}


def _queue_graph(tmp_path: Path) -> Path:
    """A graph dir whose plans hold three approved clusters, one refusal and one off-tier.

    Three run dates, so "waiting" and "oldest" are both measurable and cannot be
    confused: `Hermes` is approved on the FIRST date (54 days by TODAY), `Decider
    Service` on the second (43 days), and `Young Pair` only on the newest plan, which
    makes it 2 days old and the one record that must NOT be flagged overdue.
    `Ghost` was approved and has since been REFUSED by the gate on the newest plan,
    and `Ambiguous One` sits in tier SUFFIX_AMBIGUOUS — neither belongs in the packet.
    """
    graph = tmp_path / "memory-graph"
    graph.mkdir()
    hermes = _suffix_row("Hermes", "hermes", [("Hermes", 68), ("Hermes Agent", 26)],
                         reason="Both refer to the same open-source agent framework "
                                "by Nous Research.")
    decider = _suffix_row("Decider Service", "decider service",
                          [("Decider Service", 9), ("Decider", 7)])
    young = _suffix_row("Young Pair", "young pair",
                        [("Young Pair", 4), ("Young Pairing", 3)])
    # Approved on the two early plans and refused on the newest one: this is the
    # shape that makes the REFUSED exclusion sharp. A cluster the gate only ever
    # refused would be excluded by two rules at once and prove neither.
    ghost_approved = _suffix_row("Ghost", "ghost",
                                 [("Ghost", 5), ("Ghost Service", 2)])
    ghost_refused = _suffix_row("Ghost", "ghost",
                                [("Ghost", 5), ("Ghost Service", 2)],
                                status="REFUSED",
                                decision="semantic gate: DIFFERENT",
                                reason="One is a process, the other a persona.")
    ambiguous = _suffix_row("Ambiguous One", "ambiguous one",
                            [("Ambiguous One", 3), ("Ambigous One", 3)],
                            tier="SUFFIX_AMBIGUOUS",
                            decision="SUFFIX_SAFE — semantic gate: review")
    plans = {
        "entity-merges-2026-09-24-20260924T030000Z.jsonl": [hermes, ghost_approved],
        "entity-merges-2026-10-05-20261005T030000Z.jsonl":
            [hermes, decider, ghost_approved],
        "entity-merges-2026-11-15-20261115T030000Z.jsonl":
            [hermes, decider, young, ghost_refused, ambiguous],
        # The pointer, which `_plan_date` deliberately refuses so a run still writing
        # is never read as the newest document. Its contents are inert: were this file
        # read as the newest plan, the packet would be `Hermes` alone.
        "entity-merges-latest.jsonl": [hermes],
    }
    for name, rows in plans.items():
        (graph / name).write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return graph


def test_suffix_review_packet_lists_every_waiting_cluster_oldest_first(tmp_path):
    """Clause 1: the packet is the queue, one record per cluster, SAFE rows only.

    The queue measured 25 clusters with the oldest waiting 54 days as at 2026-11-17,
    and every one of those gate verdicts reads SAME; `merge_queue_report` says how
    many, which is not something a reviewer can act on. This node pins the fields a
    decision needs and the two exclusions that matter: a cluster the gate has since
    REFUSED (the plans held 42 of those as at 2026-11-17) must never reach a human
    told to apply, and a row of another tier must not inflate the list.

    Ages come from the FIRST plan that approved the cluster, not from the newest row
    that repeats it, or every cluster would read as one day old forever.
    """
    graph = _queue_graph(tmp_path)

    records = kg_hygiene.suffix_review_packet(graph, today=TODAY)

    assert [r["canonical"] for r in records] == ["Hermes", "Decider Service",
                                                 "Young Pair"], records
    assert [r["age_days"] for r in records] == [54, 43, 2], records
    assert [r["first_approved"] for r in records] == ["2026-09-24", "2026-10-05",
                                                      "2026-11-15"], records

    hermes = records[0]
    assert hermes["variants"] == ["Hermes", "Hermes Agent"], hermes
    assert hermes["gate_decision"] == "SUFFIX_JUDGED — every judge: SAME", hermes
    assert "Nous Research" in hermes["gate_reasons"]["Hermes Agent"], hermes
    assert hermes["overdue"] is True and records[-1]["overdue"] is False, records

    assert "Ghost" not in [r["canonical"] for r in records], (
        "a cluster the gate refused on the newest plan is in the apply packet")
    assert "Ambiguous One" not in [r["canonical"] for r in records], (
        "a row outside tier SUFFIX_SAFE is in the packet")
    assert all(r["gate_decision"].startswith("SUFFIX_JUDGED") for r in records), records

    # `today=None` is what the nightly passes: the ages must be measured against the
    # real clock, not against a fixture, and the dates must not move.
    live = kg_hygiene.suffix_review_packet(graph)
    assert [r["first_approved"] for r in live] == [r["first_approved"]
                                                   for r in records], live
    assert live[0]["age_days"] == (dt.date.today() - dt.date(2026, 9, 24)).days, live


def test_the_packet_is_a_dated_file_that_states_the_apply_command_and_reads_only_plans(
        tmp_path, monkeypatch):
    """Clause 2: the packet is a file a person can work, and it opens nothing else.

    The call returns the path so the nightly row can name it, and the filename is
    dated so a reviewer can tell which night's queue they are holding.

    "Opens only the plan JSONL" is checked by instrumenting `open` for the whole
    call rather than by reading the code: the store (`kg.sqlite`) and a fact
    directory are planted in the tree FIRST, so an implementation that reached for
    either is caught by name. The rule is not style. Draining this queue means
    writing `vault-derived/kg.sqlite` and moving fact dirs, which is #4009 and
    person-only — a round that was refused on 2026-11-09 for trying
    (`review_premise_unsound: true`). A reporting artifact that could touch the
    store would put a round one import away from the thing the guard exists to stop.

    The packet must also merge nothing: the plan files are asserted byte-identical
    after the call, and no merged fact dir appears.
    """
    graph = _queue_graph(tmp_path)
    store = tmp_path / "kg.sqlite"
    store.write_bytes(b"not a database, a witness")
    facts = tmp_path / "facts" / "Hermes"
    facts.mkdir(parents=True)
    (facts / "hermes-state.md").write_text("---\ntype: facts\n---\n", encoding="utf-8")
    plan_bytes = {p: p.read_bytes() for p in sorted(graph.glob("*.jsonl"))}
    # Only the dated plans are readable: `entity-merges-latest.jsonl` is a pointer
    # `_plan_date` refuses, so the writer must never open it.
    dated_plans = [p for p in plan_bytes if p.name != "entity-merges-latest.jsonl"]
    before = sorted((tmp_path / "facts").iterdir())

    opened: list[str] = []
    real_path_open = Path.open

    def spy(self, *args, **kwargs):
        opened.append(str(self))
        return real_path_open(self, *args, **kwargs)

    # `Path.open`, not `builtins.open`: both the plan reads (`Path.read_text`) and
    # the packet write go through this one route, so spying on `open` alone would
    # watch nothing and pass whatever the writer touched.
    monkeypatch.setattr(Path, "open", spy)
    path = kg_hygiene.write_suffix_review_packet(graph, today=TODAY,
                                                 out_dir=tmp_path / "packets")
    monkeypatch.undo()

    assert path.is_file(), path
    assert path.name == "suffix-review-2026-11-17.md", path
    text = path.read_text(encoding="utf-8")

    assert text.count(APPLY_FORM) == 4, (
        f"expected the apply form once in the preamble and once per cluster (3): "
        f"{text.count(APPLY_FORM)}")
    assert "Hermes" in text and "Decider Service" in text, text
    assert "2026-09-24" in text and "54 days" in text, text
    assert "OVERDUE" in text, text
    assert "Writing this packet applied" in text and "never the store" in text, (
        "the packet does not say it changed nothing, so holding it reads as a done "
        "decision")

    strays = [p for p in opened
              if "kg.sqlite" in p or "/facts/" in p
              or (p.endswith(".jsonl") and "/memory-graph/" not in p)]
    assert not strays, f"the packet writer opened a non-plan file: {strays}"
    assert {p for p in opened if p.endswith(".jsonl")} == {str(p) for p in dated_plans}, (
        f"the writer read plan files it should not have: {opened}")
    assert {p: p.read_bytes() for p in plan_bytes} == plan_bytes, (
        "writing the packet changed a plan file")
    assert sorted((tmp_path / "facts").iterdir()) == before, (
        "writing the packet changed the fact tree")
    assert store.read_bytes() == b"not a database, a witness", "the store was opened"


def test_the_packet_survives_a_sweep_run_that_excluded_the_tier(tmp_path):
    """The queue cannot be a count in one surface and zero in the other.

    A sweep writes several plans on one run date, and the last can be a run that
    excluded the tier: every status is then rewritten to AMBIGUOUS while the keys
    stay put. Measured on the live directory 2026-11-17, which holds three plans at
    16:00:37Z / 16:00:53Z / 16:01:01Z — 24 SAFE rows, 24 SAFE rows, then 0. Reading
    only the final file prints a Hygiene row of 24 clusters beside an empty packet,
    which is the failure this node forbids: the row and the packet are the same 24.
    """
    graph = _queue_graph(tmp_path)
    excluded = _suffix_row("Hermes", "hermes", [("Hermes", 68), ("Hermes Agent", 26)],
                           status="AMBIGUOUS",
                           decision="tier SUFFIX_SAFE excluded by --tiers")
    (graph / "entity-merges-latest-excluded.jsonl").write_text(
        json.dumps(excluded) + "\n", encoding="utf-8")

    records = kg_hygiene.suffix_review_packet(graph, today=TODAY)

    assert [r["canonical"] for r in records] == ["Hermes", "Decider Service",
                                                 "Young Pair"], records
    assert kg_hygiene.suffix_review_summary(graph, today=TODAY)["waiting"] == len(records)


# ---------------------------------------------------------------------------
# #4369 — a cluster an apply report committed is not a cluster waiting
# ---------------------------------------------------------------------------

APPLIED_ON = dt.date(2026, 11, 19)
APPLY_REPORT_NAME = "entity-merges-applied-2026-11-19-20261119T010911Z.json"

#: The 12 pairs exactly as `entity-merges-applied-2026-11-19-20261119T010911Z.json`
#: carries them in `variant_to_canonical` (variant -> canonical). They are §4's APPLY
#: list of `backlog/data/suffix-safe-split-2026-11-09.md`, run at 01:09Z with
#: `--apply --tiers SUFFIX_SAFE`, and every one of the 12 variant directories is gone
#: from `vault-derived/facts/` as a result. Invented names would let a matcher pass on
#: a shape the real reports do not have.
APPLIED_PAIRS = {
    "auto-reply": "auto-reply system",
    "Tester Agent": "Tester",
    "lloyd-tool-mcp.service": "lloyd-tool-mcp",
    "agent-supervisord": "agent-supervisord.service",
    "automod system": "automod",
    "Session Enrichment Pipeline": "Session enrichment",
    "remobi.app": "Remobi",
    "OCI Generative AI": "OCI Generative AI service",
    "Hermes Agent": "Hermes",
    "intelligent adversary": "Intelligent adversary agent",
    "autoresearch system": "Autoresearch",
    "Feature Enablement System": "FeatureEnablement",
}

#: The 12 clusters the 16:00Z plan of the same day still holds as SAFE — the queue a
#: person actually had left. The variant names are stand-ins (`<canonical> System`);
#: what this file pins is the count and the identity of what is dropped.
QUEUE_AFTER_APPLY = ["Reviewer Agent", "Planner Agent", "Anomaly Detection System",
                     "Skill System", "rig-agent", "Verifier Agent", "Tessl Agent",
                     "youtube-content", "Decider", "Workers", "Autolink System",
                     "S2S Authentication"]


def _pair_row(variant: str, canonical: str, **over) -> dict:
    """A plan row carrying the `merges` list a real sweep writes.

    `merges[].variant` is the name the run would consume, which is the key the apply
    report is later written under — the join between a plan and a report.
    """
    row = _suffix_row(canonical, kg_hygiene.sweep().normalize_punct(canonical),
                      [(canonical, 9), (variant, 4)], **over)
    row["merges"] = [{"variant": variant, "canonical": canonical,
                      "tier": "SUFFIX_SAFE", "action": "merge"}]
    return row


def _apply_report(graph: Path, pairs: dict[str, str], *, name: str = APPLY_REPORT_NAME,
                  date: str | None = "2026-11-19",
                  tiers: list[str] | None = None) -> Path:
    """An apply report of `pairs`, shaped like the real one (variant_to_canonical + date)."""
    body = {"tiers_allowed": tiers or ["SUFFIX_SAFE"], "applied_clusters": len(pairs),
            "alias_writes": len(pairs), "report_status": "complete",
            "variant_to_canonical": dict(pairs)}
    if date is not None:
        body["date"] = date
    path = graph / name
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def _queue_after_an_apply(tmp_path: Path) -> Path:
    """The live shape of 2026-11-19: 24 approved suffix clusters in the plans, 12 of
    them committed by the 01:09Z SUFFIX apply.

    Both plans carry all 24 rows — the 09-24 plan so that the survivors still date to
    2026-09-24, which is the half of the acceptance test the old row could not pass:
    the surviving HOLD clusters are the oldest in the queue, so the age RISES when the
    middle of the queue is merged away.
    """
    graph = tmp_path / "memory-graph"
    graph.mkdir()
    rows = ([_pair_row(v, c) for v, c in APPLIED_PAIRS.items()]
            + [_pair_row(f"{c} System", c) for c in QUEUE_AFTER_APPLY])
    for name in ("entity-merges-2026-09-24-20260924T160000Z.jsonl",
                 "entity-merges-2026-11-19-20261119T160035Z.jsonl"):
        (graph / name).write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                                  encoding="utf-8")
    _apply_report(graph, APPLIED_PAIRS)
    return graph


def test_the_packet_drops_every_cluster_an_apply_report_says_is_merged(tmp_path):
    """Clause 1: 12 rows, not 24 — measured on the live store the same way.

    The packet was assembled from the plan files and nothing else, so a merged cluster
    stayed in it for as long as the plan that approved it was on disk. The 13:00Z packet
    of 2026-11-19 held 24 rows and 12 of them named a pair the 01:09Z apply had already
    committed, each still headed `first approved 2026-09-24; waiting 56 days - OVERDUE`
    and each ending `to apply: --apply --tiers SUFFIX_SAFE`: a person working the top
    rows would have been told to re-merge 12 clusters that are merged. Deleting the
    report and getting 24 back is what shows the reports, and not some other rule, are
    what dropped those 12.
    """
    graph = _queue_after_an_apply(tmp_path)

    records = kg_hygiene.suffix_review_packet(graph, today=APPLIED_ON)

    assert len(records) == 12, [r["canonical"] for r in records]
    assert sorted(r["canonical"] for r in records) == sorted(QUEUE_AFTER_APPLY), records
    named = {v for r in records for v in r["variants"]}
    assert not (named & set(APPLIED_PAIRS)), named & set(APPLIED_PAIRS)

    (graph / APPLY_REPORT_NAME).unlink()
    assert len(kg_hygiene.suffix_review_packet(graph, today=APPLIED_ON)) == 24, \
        "without the report the plans still hold 24; the exclusion is the reports"


def test_a_merged_cluster_leaves_the_packet_by_its_variant_name(tmp_path):
    """The join is the name the merge consumed, on the ordinary queue fixture.

    `Hermes` is one of the three waiting clusters of `_queue_graph`; a report naming
    only its variant `Hermes Agent` removes it, and the other two are untouched. The
    fixture's rows carry no `merges` list — the shape of the plans written before that
    field existed — so this also pins that a row with no list of consumed variants is
    matched by any name it carries, canonical included.
    """
    graph = _queue_graph(tmp_path)
    _apply_report(graph, {"Hermes Agent": "Hermes"})

    records = kg_hygiene.suffix_review_packet(graph, today=TODAY)

    assert [r["canonical"] for r in records] == ["Decider Service", "Young Pair"], records


def test_a_cluster_that_only_lends_its_canonical_name_stays_in_the_packet(tmp_path):
    """Over-exclusion is the failure mode on the other side, and it costs a real item.

    `Reviewer Agent` is one of the 12 still waiting. A report that merged some OTHER
    name into it — here `Reviewer Agent` appearing as the DESTINATION of a pair — does
    not make this cluster's own pending merge (`Reviewer Agent System` -> `Reviewer
    Agent`) already done: its directory is live and its variant is still on disk. A row
    that lists the variants it consumes is matched only by those names, or the queue
    would shrink every time one of its names shows up as somebody else's destination.
    """
    graph = _queue_after_an_apply(tmp_path)
    _apply_report(graph, {"Reviewer": "Reviewer Agent"},
                  name="entity-merges-applied-2026-11-19-20261119T020857Z.json")

    records = kg_hygiene.suffix_review_packet(graph, today=APPLIED_ON)

    assert len(records) == 12, [r["canonical"] for r in records]
    assert "Reviewer Agent" in [r["canonical"] for r in records], records


def test_the_suffix_count_is_the_live_queue_after_the_apply(tmp_path):
    """Clause 2: `suffix_clusters` counts what a person can still apply: 12, not 24.

    The row is the one surface that printed a number for this queue, and it printed 25
    on 2026-11-16, 25 on 2026-11-17 and 24 on 2026-11-19 — a −1 for an apply of 12,
    because the newest plan refills with freshly-approved clusters and the survivors are
    the oldest ones. `suffix_applied` is the figure that cannot net away: 12 clusters of
    this tier are committed, on 2026-11-19.
    """
    graph = _queue_after_an_apply(tmp_path)

    q = kg_hygiene.merge_queue_report(graph, today=APPLIED_ON)

    assert q["suffix_clusters"] == 12, q
    assert q["suffix_applied"] == 12 and q["suffix_applied_on"] == "2026-11-19", q
    assert q["suffix_oldest_date"] == "2026-09-24" and q["suffix_age_days"] == 56, q

    (graph / APPLY_REPORT_NAME).unlink()
    before = kg_hygiene.merge_queue_report(graph, today=APPLIED_ON)
    assert (before["suffix_clusters"], before["suffix_applied"],
            before["suffix_applied_on"]) == (24, 0, None), before


def test_a_mechanical_tier_apply_does_not_borrow_the_suffix_applied_number(tmp_path):
    """The count is the tier's own: an applied CASE pair is not a SUFFIX apply.

    #48 applies CASE and PUNCT nightly — `entity-merges-applied-2026-11-19-
    20261119T020857Z.json` (GPT-6 -> GPT 6) and `…T160411Z.json` (Autoresearch Bench ->
    autoresearch-bench) are two such reports on the live tree. Those clusters were never
    in the human queue, so a count that read every report's pairs would print a
    nightly-apply number as though a person's work had been done. `GPT 6` here is a
    gate-REFUSED row of the suffix tier, which is the shape that would be counted by a
    ledger that intersected with the tier instead of with its SAFE rows.
    """
    graph = _queue_after_an_apply(tmp_path)
    refused = _pair_row("GPT-6", "GPT 6", status="AMBIGUOUS",
                        decision="tier SUFFIX_SAFE excluded by --tiers")
    (graph / "entity-merges-2026-11-19-20261119T160153Z.jsonl").write_text(
        json.dumps(refused) + "\n", encoding="utf-8")
    _apply_report(graph, {"GPT-6": "GPT 6"},
                  name="entity-merges-applied-2026-11-19-20261119T020857Z.json",
                  tiers=["CASE"])

    q = kg_hygiene.merge_queue_report(graph, today=APPLIED_ON)

    assert q["suffix_applied"] == 12 and q["suffix_clusters"] == 12, q
    assert q["suffix_refused"] == 1, q
    assert "GPT 6" not in str(kg_hygiene.suffix_apply_ledger(graph))


def test_the_written_packet_states_what_has_already_been_applied(tmp_path):
    """The packet's own header has to say the queue has been worked (#4369 clause 1).

    The 2026-11-19 13:00Z packet announced 24 waiting clusters twelve hours after 12 of
    them were merged and never said so. The header now names the applied count and the
    date of the most recent apply beside the waiting count, and only the 12 rows a
    person can still act on are numbered.
    """
    graph = _queue_after_an_apply(tmp_path)

    path = kg_hygiene.write_suffix_review_packet(graph, today=APPLIED_ON)
    text = path.read_text(encoding="utf-8")

    assert "12 cluster(s) of this tier are ALREADY APPLIED" in text, text.splitlines()[:8]
    assert "most recent apply 2026-11-19" in text, text.splitlines()[:8]
    numbered = [l for l in text.splitlines() if re.match(r"^\d+\. `", l)]
    assert len(numbered) == 12, numbered


def test_an_unreadable_apply_report_never_shrinks_the_queue(tmp_path):
    """A torn report is skipped, and the queue keeps the rows it cannot disprove.

    A report that failed to parse is not evidence that 12 clusters are merged; reading
    it that way would empty the human queue on a truncated write, which is the same
    class of failure as the row that went silent (#3407 clause 4). The same node pins
    the other degradation: a report with no `date` field is still dated by the ISO day
    in its own filename, because that is where the run date always lives.
    """
    graph = _queue_after_an_apply(tmp_path)
    (graph / APPLY_REPORT_NAME).write_text('{"variant_to_canonical": {"auto', encoding="utf-8")

    q = kg_hygiene.merge_queue_report(graph, today=APPLIED_ON)
    assert (q["suffix_clusters"], q["suffix_applied"]) == (24, 0), q
    assert len(kg_hygiene.suffix_review_packet(graph, today=APPLIED_ON)) == 24

    _apply_report(graph, APPLIED_PAIRS, date=None)
    dated = kg_hygiene.merge_queue_report(graph, today=APPLIED_ON)
    assert dated["suffix_applied"] == 12, dated
    assert dated["suffix_applied_on"] == "2026-11-19", dated
