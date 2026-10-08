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

from scripts.reflection_archive import copy_gaps, pattern_staleness, reports_in  # noqa: E402


def _report(gaps, stale_patterns=None) -> str:
    now = datetime.now(timezone.utc)
    return khr.generate_report({}, khr.compute_relationship_stats([], {}), [], [], now,
                               copy_gaps=gaps, pattern_staleness=stale_patterns)


def _stale_fixture(tmp_path: Path) -> Path:
    """The #2413 state on a fixture: a 10-06 archive under a 10-07 report.

    Built as files, not as a hand-made finding, so what reaches the report is
    `pattern_staleness`'s own verdict — the seam this file exists to cross.
    """
    d = tmp_path / "reflection"
    d.mkdir()
    for name in ("knowledge-write-2026-10-07.md", "tool-patterns-latest.md",
                 "tool-patterns-latest-2026-10-06-0809.md",
                 "conversation-patterns-latest.md",
                 "conversation-patterns-latest-2026-10-07-0812.md"):
        (d / name).write_text("cycle content\n")
    return d


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


# ── #2413: the §2e pattern archive rides the same section ────────────────────


def test_a_stale_pattern_archive_reaches_the_generated_report(tmp_path):
    """Clause 4: a non-empty `pattern_staleness` is readable in the report text.

    The finding has to arrive through `pattern_staleness` over a fixture directory,
    because the whole reason #2413 exists is that the shape was invisible to every
    reader of the live tree: task 39's 2026-10-07 run skipped §2e, named no copy, so
    `copy_gaps` returned 0 findings and the report said the chain was clean. Asserting
    the stem, the stale copy and the report it lags behind — in the section the
    `copy_gaps` rows already live in — is what makes a silently skipped step surface
    nightly instead of only to a 14-day step-conformance replay.
    """
    d = _stale_fixture(tmp_path)
    stale = pattern_staleness(d)
    assert [f.stem for f in stale] == ["tool-patterns-latest"], stale
    section = _section(_report(copy_gaps(d, reports_in(d)), stale))
    assert "Pattern files behind the newest report: 1" in section
    assert "| `tool-patterns-latest.md` | `tool-patterns-latest-2026-10-06-0809.md` " \
           "| `knowledge-write-2026-10-07.md` |" in section
    assert "every §2e archive is current" not in section


def test_a_current_pattern_archive_says_so_without_erasing_the_pointer_rows(tmp_path):
    """The healthy nightly state, stated as a zero, in the same section as the pointer rows.

    The two checks answer different questions and their counts must not be summed or
    read as one verdict: a fixture with a current archive and a pointer naming an
    absent copy is exactly a report that is clean on #2413's question and dirty on
    #1227's. Both numbers print, and the pattern line says zero rather than omitting
    the row — an absent line reads the same as a check that never ran.
    """
    d = _stale_fixture(tmp_path)
    (d / "tool-patterns-latest-2026-10-07-0900.md").write_text("current archive\n")
    (d / "tool-patterns-latest.md").write_text(
        "Archive copied before this write: "
        "`tool-patterns-latest-2026-10-07-0930.md`\n")
    gaps = copy_gaps(d, reports_in(d))
    stale = pattern_staleness(d)
    assert gaps and not stale, (gaps, stale)
    section = _section(_report(gaps, stale))
    assert _stated_count(section) == len(gaps)
    assert "Pattern files behind the newest report: 0" in section
    assert "every §2e archive is current" in section


def test_an_unmeasured_pattern_check_does_not_claim_current_archives():
    """`None` is not a zero: the section must not read as clean when nothing ran.

    The same rail #1227 pinned for the pointer rows, re-pinned for the new ones — a
    reader cannot tell "0 files are behind" from "nobody looked" unless the text says
    which, and the default for every other caller of `generate_report` is `None`.
    """
    section = _section(_report([], None))
    assert "Not measured" in section
    assert "every §2e archive is current" not in section
    assert "Pattern files behind the newest report" not in section
