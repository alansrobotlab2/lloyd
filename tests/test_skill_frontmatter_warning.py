"""#561: a skill whose front matter does not parse loads with an empty
description and no tags — out of retrieval while looking installed. The loader
used to swallow the YAML error; it now logs it once, naming the skill, and the
lint's DEAD detector is pinned on the same fixture.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import agent_mcp.skills as skills_mod

ROOT = Path(__file__).resolve().parent.parent
BROKEN = "---\nname: broken-skill\ndescription: [unclosed\n  tags: :\n---\n# body\n"


def _broken_skill(tmp_path: Path) -> Path:
    d = tmp_path / "broken-skill"
    d.mkdir()
    (d / "SKILL.md").write_text(BROKEN)
    return d


def test_unparseable_front_matter_is_logged_naming_the_skill(tmp_path, caplog):
    d = _broken_skill(tmp_path)
    with caplog.at_level(logging.WARNING, logger="agent_mcp.skills"):
        skill = skills_mod._load_skill(d)
    # Still loads — the fix makes the failure visible, not fatal.
    assert skill is not None and skill["description"] == ""
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "broken-skill" in warnings[0] and "does not parse" in warnings[0]


def test_the_warning_is_once_per_block_not_once_per_walk(tmp_path, caplog):
    d = _broken_skill(tmp_path)
    with caplog.at_level(logging.WARNING, logger="agent_mcp.skills"):
        for _ in range(3):
            skills_mod._load_skill(d)
    assert sum("does not parse" in r.getMessage() for r in caplog.records) == 1


def test_well_formed_front_matter_logs_nothing(tmp_path, caplog):
    d = tmp_path / "fine-skill"
    d.mkdir()
    (d / "SKILL.md").write_text("---\ndescription: ok\ntags: [a]\n---\nbody\n")
    with caplog.at_level(logging.WARNING, logger="agent_mcp.skills"):
        assert skills_mod._load_skill(d)["description"] == "ok"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_skill_lint_flags_the_same_fixture_dead():
    spec = importlib.util.spec_from_file_location(
        "skill_lint_561", ROOT / "scripts" / "skill_lint.py")
    lint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lint)
    fm, _ = skills_mod._parse_frontmatter(BROKEN)
    dead, reasons = lint.check_dead(fm, "mapping values are not allowed here")
    assert dead is True
    assert any(r.startswith("frontmatter:") for r in reasons)
