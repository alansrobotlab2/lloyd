"""Merging two fact files may not make one ID name two facts.

`assign_ids` fills blanks and deliberately never renumbers an existing ID,
because an ID is a handle `fact_invalidate` and every revert report names.
That is right for appending and wrong for merging, which concatenates two
independently-numbered sequences. On 2026-09-08 the live tree held 130,614
facts whose ID another fact in the same file already had; a tree built by a
fresh extraction held none.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "memory"))

from app.fact_ids import assign_ids, dedupe_ids  # noqa: E402


def _facts(*ids):
    return [{"id": i, "fact": f"fact {n}", "category": "state"} for n, i in enumerate(ids)]


def test_first_holder_keeps_the_id():
    facts = _facts("stat-001", "stat-001", "stat-002")
    moved = dedupe_ids(facts, "state")
    assert moved == 1
    assert facts[0]["id"] == "stat-001", "the ID outside records name must not move"
    assert len({f["id"] for f in facts}) == 3


def test_every_collision_is_renumbered():
    facts = _facts(*(["stat-001"] * 5))
    assert dedupe_ids(facts, "state") == 4
    assert len({f["id"] for f in facts}) == 5


def test_clean_list_is_untouched():
    facts = _facts("stat-001", "stat-002")
    before = [dict(f) for f in facts]
    assert dedupe_ids(facts, "state") == 0
    assert facts == before


def test_facts_without_ids_are_left_to_assign_ids():
    facts = [{"fact": "no id yet", "category": "state"}]
    assert dedupe_ids(facts, "state") == 0
    assert facts[0].get("id") is None
    assign_ids(facts, "state")
    assert facts[0]["id"]


def test_merge_of_two_numbered_files_collides_without_the_fix():
    """The shape the merge actually produces: two files each numbered from 1."""
    a = _facts("stat-001", "stat-002")
    b = _facts("stat-001", "stat-002")
    merged = a + b
    assert len({f["id"] for f in merged}) == 2, "precondition: they collide"
    assert dedupe_ids(merged, "state") == 2
    assert len({f["id"] for f in merged}) == 4


def test_both_merge_paths_dedupe():
    """The sweep merges on apply and the revert merges on undo. Both concatenate."""
    sweep = (ROOT / "scripts" / "memory" / "entity-resolution-sweep.py").read_text()
    body = sweep.split("def _merge_fact_file_into")[1].split("\ndef ")[0]
    assert "dedupe_ids(" in body

    revert = (ROOT / "scripts" / "memory" / "revert-suffix-merges.py").read_text()
    assert revert.count("dedupe_ids(") >= 2, "move_whole and split both merge into a dest"


# --- the repair tool ---------------------------------------------------------

import importlib.util  # noqa: E402

import yaml  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "repair_fact_ids", ROOT / "scripts" / "memory" / "repair_fact_ids.py")
repair_fact_ids = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(repair_fact_ids)


def _write_fact_file(root: Path, entity: str, category: str, facts: list[dict]) -> Path:
    d = root / entity
    d.mkdir(parents=True, exist_ok=True)
    fm = {"type": "facts", "entity": entity, "category": category, "facts": facts}
    p = d / f"{entity}-{category}.md"
    p.write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# {entity} - {category}\n",
                 encoding="utf-8")
    return p


def test_repair_renumbers_and_keeps_every_fact(tmp_path):
    p = _write_fact_file(tmp_path, "RobotArena", "state", [
        {"id": "stat-001", "fact": "alpha", "category": "state"},
        {"id": "stat-001", "fact": "beta", "category": "state"},
        {"id": "stat-002", "fact": "gamma", "category": "state"},
        {"id": "stat-002", "fact": "delta", "category": "state"},
    ])

    res = repair_fact_ids.repair(tmp_path, apply=True)
    assert res["files"] == 1 and res["renumbered"] == 2

    fm = yaml.safe_load(p.read_text().split("---")[1])
    facts = fm["facts"]
    assert [f["fact"] for f in facts] == ["alpha", "beta", "gamma", "delta"], "no fact may be lost"
    assert len({f["id"] for f in facts}) == 4
    assert facts[0]["id"] == "stat-001" and facts[2]["id"] == "stat-002"


def test_repair_dry_run_writes_nothing(tmp_path):
    p = _write_fact_file(tmp_path, "Probe", "state", [
        {"id": "stat-001", "fact": "alpha", "category": "state"},
        {"id": "stat-001", "fact": "beta", "category": "state"},
    ])
    before = p.read_text()

    res = repair_fact_ids.repair(tmp_path, apply=False)

    assert res["renumbered"] == 1, "it still reports what an apply would do"
    assert p.read_text() == before


def test_repair_is_idempotent(tmp_path):
    _write_fact_file(tmp_path, "Probe", "state", [
        {"id": "stat-001", "fact": "alpha", "category": "state"},
        {"id": "stat-001", "fact": "beta", "category": "state"},
    ])
    assert repair_fact_ids.repair(tmp_path, apply=True)["renumbered"] == 1
    assert repair_fact_ids.repair(tmp_path, apply=True)["renumbered"] == 0
