"""The architecture docs' source tables name exactly what the pool polls.

#897: `gap-fill` stayed in both docs' rosters as a live source, registered and
polling every 300 s, while it had never once run. The docs read the same for a
source that is idle and one that cannot fire, so the tables are held to the
registry: a retired source has to leave them, and a new one has to arrive —
`board-steward` had been missing from `architecture/workers.md` since it
shipped. Retired sources are §7's business in `workers-jobs.md`, which this
does not read.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

import workers.sources as sources

ROOT = Path(__file__).resolve().parent.parent
ARCH = ROOT / "architecture"

_ROW = re.compile(r"^\| `([a-z][a-z0-9-]*)` \|")


def _table_after(path: Path, heading: str) -> set[str]:
    """Source names in the first table under `heading`."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(heading))
    names: set[str] = set()
    in_table = False
    for ln in lines[start + 1:]:
        if ln.startswith("|"):
            in_table = True
            m = _ROW.match(ln)
            if m:
                names.add(m.group(1))
        elif in_table:
            break
    return names


def _registry() -> set[str]:
    return set(sources.SOURCE_REGISTRY)


def test_gap_fill_is_retired():
    assert "gap-fill" not in _registry()
    assert not (ROOT / "workers" / "sources" / "gap_fill.py").exists()


def test_workers_jobs_roster_is_the_registry():
    assert _table_after(ARCH / "workers-jobs.md", "## 1. The roster") == _registry()


def test_workers_jobs_families_cover_the_registry():
    text = (ARCH / "workers-jobs.md").read_text(encoding="utf-8")
    block = text[text.index("| § | family |"):text.index("## 1. The roster")]
    named = set(re.findall(r"`([a-z][a-z0-9-]*)`", block))
    assert named == _registry()


def test_workers_md_source_table_is_the_registry():
    got = _table_after(ARCH / "workers.md", "| source | prio | what it does |")
    assert got == _registry()


def test_config_configures_only_registered_sources():
    """An unregistered block is inert (the pool iterates the registry), which
    is exactly why it would outlive its source unnoticed."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert set(cfg["workers"]["sources"]) <= _registry()
