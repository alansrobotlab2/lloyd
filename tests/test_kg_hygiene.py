"""kg_hygiene.py — contamination, near-duplicate clusters, regrowth.

Pins the measurements the 2026-09-03 audit computed by hand: 63 directories
held facts about another entity, every one from a suffix merge.
"""
import importlib.util
import datetime as dt
import os
import sys
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "memory"))
import kg_hygiene  # noqa: E402


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


def test_snapshot_has_all_three_sections(tree):
    s = kg_hygiene.snapshot(tree, days=7)
    assert set(s) >= {"contamination", "near_duplicates", "regrowth", "captured_at"}
    assert "items" not in s["contamination"]      # summary only
    assert s["contamination"]["dirs"] == 1


def test_missing_root_is_empty_not_an_error(tmp_path):
    s = kg_hygiene.snapshot(tmp_path / "nope", days=7)
    assert s["contamination"]["dirs"] == 0
    assert s["near_duplicates"]["clusters"] == 0


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
