"""kg_hygiene.py — contamination, near-duplicate clusters, regrowth.

Pins the measurements the 2026-09-03 audit computed by hand: 63 directories
held facts about another entity, every one from a suffix merge.
"""
import importlib.util
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


def test_regrowth_counts_only_newer_near_duplicates(tree):
    now = time.time()
    old = now - 30 * 86400
    for d in tree.iterdir():
        if d.is_dir():
            for f in d.glob("*.md"):
                os.utime(f, (old, old))
    # the lowercase vLLM dir was born yesterday, next to a month-old sibling
    for f in (tree / "vLLM V1 engine").glob("*.md"):
        os.utime(f, (now - 86400, now - 86400))
    # a brand-new unique entity is not regrowth
    _fact_file(tree, "Gemma 4", "state", [("Gemma 4", "Open weights.")])
    r = kg_hygiene.regrowth(tree, days=7, now=now)
    assert r["new_dirs"] == 2
    assert r["near_dup_new"] == 1
    assert r["samples"] == ["vLLM V1 engine"]
    assert r["by_tier"] == {"CASE": 1}


def test_snapshot_has_all_three_sections(tree):
    s = kg_hygiene.snapshot(tree, days=7)
    assert set(s) >= {"contamination", "near_duplicates", "regrowth", "captured_at"}
    assert "items" not in s["contamination"]      # summary only
    assert s["contamination"]["dirs"] == 1


def test_missing_root_is_empty_not_an_error(tmp_path):
    s = kg_hygiene.snapshot(tmp_path / "nope", days=7)
    assert s["contamination"]["dirs"] == 0
    assert s["near_duplicates"]["clusters"] == 0


def test_regrowth_dates_entities_by_fact_created_at_not_mtime(tree):
    """A bulk revert/merge rewrites every file; mtime then says everything is new."""
    now = time.time()
    # every file was rewritten just now ...
    for d in tree.iterdir():
        if d.is_dir():
            for f in d.glob("*.md"):
                os.utime(f, (now, now))
    # ... but the facts say when the entities really appeared
    def stamp(dirname, when):
        for p in (tree / dirname).glob("*.md"):
            fm = yaml.safe_load(p.read_text().split("---")[1])
            if not fm.get("facts"):          # overview files carry no facts
                continue
            for f in fm["facts"]:
                f["created_at"] = when
            p.write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\nbody\n")
            os.utime(p, (now, now))
    import datetime as dt
    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=40)).isoformat()
    fresh = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)).isoformat()
    stamp("vLLM V1 Engine", old)
    stamp("vLLM V1 engine", fresh)
    stamp("Intel", old); stamp("Alfie", old)
    r = kg_hygiene.regrowth(tree, days=7, now=now)
    assert r["new_dirs"] == 1                  # only the lowercase twin is genuinely new
    assert r["samples"] == ["vLLM V1 engine"]
