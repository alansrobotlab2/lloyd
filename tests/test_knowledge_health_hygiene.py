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
