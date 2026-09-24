"""The Knowledge Health Report's Reflection Retention section (#1227).

The count it states must be `copy_gaps`' count, and an empty result must say
in words that no cycle is missing a copy — a silent section reads the same as
a check that never ran.
"""
import importlib.util
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("khr_retention", ROOT / "scripts/memory/knowledge-health-report.py")
khr = importlib.util.module_from_spec(_spec); sys.modules["khr_retention"] = khr; _spec.loader.exec_module(khr)

from scripts.reflection_archive import copy_gaps, reports_in  # noqa: E402


def _report(gaps) -> str:
    now = datetime.now(timezone.utc)
    return khr.generate_report({}, khr.compute_relationship_stats([], {}), [], [], now,
                               copy_gaps=gaps)


def _section(report: str) -> str:
    return report.split("## Reflection Retention", 1)[1].split("\n## ", 1)[0]


def _stated_count(section: str) -> int:
    return int(re.search(r"Reports naming an archive copy that is missing: (\d+)", section).group(1))


def test_section_count_equals_the_functions_count(tmp_path):
    d = tmp_path / "reflection"; d.mkdir()
    (d / "signals-latest.md").write_text(
        "Archive copied before this write: `signals-latest-2026-09-20-0517.md`\n")
    (d / "knowledge-write-2026-09-20.md").write_text(
        "Copies: `tool-patterns-latest-2026-09-20-0938.md`, "
        "`conversation-patterns-latest-2026-09-20-0938.md`\n")
    (d / "tool-patterns-latest-2026-09-20-0938.md").write_text("kept\n")
    gaps = copy_gaps(d, reports_in(d))
    assert len(gaps) == 2
    section = _section(_report(gaps))
    assert _stated_count(section) == len(gaps)
    for gap in gaps:
        assert f"| `{gap.report.name}` | `{gap.copy}` |" in section
    assert "no cycle is missing a copy" not in section


def test_empty_result_says_no_cycle_is_missing_a_copy():
    section = _section(_report([]))
    assert _stated_count(section) == 0
    assert "no cycle is missing a copy" in section


def test_unmeasured_is_not_reported_as_clean():
    section = _section(_report(None))
    assert "Not measured" in section
    assert "no cycle is missing a copy" not in section
