"""skill-lint counts which live skills an unattended job wrote (#774).

The question used to be answerable only by matching vault commit subjects. The
writer skills now stamp `written_by: {job, date}` and `lint()` counts the stamps
off the front matter, so the number is a lint output rather than a git walk.
Synthetic skills under a temp root, walked by the production walker; no vault.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load():
    # By path, as task #70 runs it (`scripts/` is not a package).
    spec = importlib.util.spec_from_file_location(
        "skill_lint_authorship", ROOT / "scripts" / "skill_lint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LINT = _load()

_SKILLS = {
    "bash-timeouts": "written_by:\n  job: autonomy-58\n  date: '2026-09-24T03:00:00Z'\n",
    "turn-end-todos": "written_by:\n  job: autonomy-58\n  date: '2026-09-24T03:00:00Z'\n",
    "mined": "written_by: {job: mining-57, date: '2026-09-24'}\n",
    "hand-written": "written_by: interactive\n",
    "legacy": "",
    "malformed": "written_by: [autonomy-83]\n",
}


def _records(tmp_path: Path):
    from agent_mcp.skills import iter_active_skills

    for name, stamp in _SKILLS.items():
        d = tmp_path / name
        d.mkdir()
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\n"
            "description: Use this skill when testing authorship counts.\n"
            f"tags: [demo]\nstatus: active\n{stamp}---\n# {name}\n",
            encoding="utf-8")
    return list(iter_active_skills(roots=[tmp_path]))


def test_lint_counts_written_by_without_a_git_walk(tmp_path):
    result = LINT.lint(skill_records=_records(tmp_path))
    authorship = result["authorship"]
    assert authorship["by_job"] == {"autonomy-58": 2, "interactive": 1, "mining-57": 1}
    assert authorship["machine_written"] == 3
    assert authorship["interactive"] == 1
    # No stamp, and a stamp that names no job, are unrecorded — never a job.
    assert authorship["unrecorded"] == ["legacy", "malformed"]


def test_report_prints_the_count_and_keeps_it_out_of_the_verdict(tmp_path):
    result = LINT.lint(skill_records=_records(tmp_path))
    report = LINT.render_report(result)
    assert ("Machine-written live skills: **3** · interactive: **1** · "
            "no `written_by`: **2** of 6.") in report
    assert "| `autonomy-58` | 2 |" in report
    # A measurement, not a finding: an otherwise clean library still reaches
    # the qualified all-zero verdict rather than a findings section.
    assert "## No findings on the checks that can fail" in report
