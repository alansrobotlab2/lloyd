"""Every tracked architecture doc says it describes what runs, and says when.

`architecture/index.md` promises that `status: implemented` docs describe what
runs and that anything else is moved to `.archive/`. For a week that sentence was
decorative: four docs carried no `status:`, two said `active` (a word no status
vocabulary in this repo defines) and one said `current` (the arch-review
*verdict* axis, never a front-matter value), and `skills.md` had no date key of
any name (#1104). Nothing walked the front matter, so nothing noticed.

The date keys accepted are the three the docs already use and the arch-review
prompt refreshes (`date`, `updated`, `timestamp`); requiring `date:` alone would
fail docs that the review session itself maintains.

The walk is scoped to tracked, top-level docs: `architecture/.archive/` is
gitignored and holds retired docs whose front matter deliberately stopped
describing anything. It asserts its own set is non-empty so it cannot pass on an
empty list.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "architecture" / "index.md"

STATUSES = ("implemented",)
DATE_KEYS = ("date", "updated", "timestamp")


def _front_matter(path: Path) -> dict:
    lines = path.read_text().splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            data = yaml.safe_load("\n".join(lines[1:i])) or {}
            return data if isinstance(data, dict) else {}
    return {}


def frontmatter_problems(path: Path) -> list[str]:
    """What is wrong with one doc's front matter; empty when it is sound."""
    fm = _front_matter(path)
    problems = []
    status = fm.get("status")
    if status not in STATUSES:
        problems.append(f"{path.name}: status is {status!r}, expected one of {STATUSES}")
    if not any(fm.get(k) for k in DATE_KEYS):
        problems.append(f"{path.name}: no date key ({' | '.join(DATE_KEYS)})")
    return problems


def tracked_docs() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "--", "architecture/*.md"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    return [ROOT / rel for rel in out if rel.count("/") == 1]


def test_the_walked_set_is_the_tracked_top_level_docs():
    docs = tracked_docs()
    assert len(docs) >= 20, f"only {len(docs)} tracked architecture docs found"
    assert INDEX in docs
    assert not any(".archive" in d.parts for d in docs)


def test_every_tracked_doc_is_implemented_and_dated():
    problems = [p for d in tracked_docs() for p in frontmatter_problems(d)]
    assert not problems, "\n".join(problems)


def test_the_index_names_the_same_vocabulary():
    text = " ".join(INDEX.read_text().split())
    for value in STATUSES:
        assert f"`status: {value}`" in text
    for key in DATE_KEYS:
        assert f"`{key}`" in text, f"index.md does not name the {key!r} date key"


GOOD = "---\nsegment: architecture\nstatus: implemented\nupdated: 2026-09-24\n---\n# x\n"


@pytest.mark.parametrize("text,expect_ok", [
    (GOOD, True),
    (GOOD.replace("status: implemented\n", ""), False),
    (GOOD.replace("status: implemented", "status: active"), False),
    (GOOD.replace("updated: 2026-09-24\n", ""), False),
    ("# no front matter at all\n", False),
])
def test_the_validator_fails_a_doc_that_lost_its_status_or_date(tmp_path, text, expect_ok):
    doc = tmp_path / "fixture.md"
    doc.write_text(text)
    assert (frontmatter_problems(doc) == []) is expect_ok
