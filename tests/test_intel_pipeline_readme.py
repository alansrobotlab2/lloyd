"""#847 (finding 3): `scripts/intel-pipeline/README.md` documented a package that
did not exist — arXiv and Hacker News scanners, a `cd ~/obsidian/agents/...`
invocation into a vault tree deleted on 2026-09-03, and imports of modules
that were never in this checkout. Every `intel_pipeline.<module>` the README
names is resolved against the package on disk, and the dead paths are grepped
for, so the doc cannot describe a tree other than this one again.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = ROOT / "scripts" / "intel-pipeline"
README = PKG_DIR / "README.md"
PACKAGE = PKG_DIR / "intel_pipeline"


def _modules_named(text: str) -> set[str]:
    return set(re.findall(r"intel_pipeline(?:\.[A-Za-z_][A-Za-z0-9_]*)+", text))


def test_every_module_the_readme_names_exists_on_disk():
    named = _modules_named(README.read_text())
    assert named, "the README names no intel_pipeline module at all"
    for dotted in sorted(named):
        rel = Path(*dotted.split(".")[1:])
        assert (PACKAGE / rel).with_suffix(".py").exists() or (PACKAGE / rel).is_dir(), \
            f"README names {dotted}, which is not in {PACKAGE}"


def test_readme_architecture_tree_lists_the_real_scanners():
    text = README.read_text()
    on_disk = {p.stem for p in (PACKAGE / "scanners").glob("*_scanner.py")}
    listed = set(re.findall(r"([a-z_]+_scanner)\.py", text))
    assert listed == on_disk, (listed, on_disk)


def test_readme_points_at_the_live_invocation_not_the_deleted_vault_tree():
    text = README.read_text()
    assert "obsidian/agents" not in text
    assert "obsidian/memory/feeds" not in text
    assert "cd ~/lloyd/scripts/intel-pipeline" in text
    assert "python -m intel_pipeline" in text
