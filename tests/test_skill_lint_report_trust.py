"""skill-lint's report carries its own answer to "is this 0 trustworthy?" (#903).

Task #70's 2026-09-10 calibration found two categories that cannot flag
anything today — DRIFT passes 53% of the library through an untested length
heuristic and STALE returns before its age check for every skill — and wrote
that qualification into the report by hand. `main` rewrites the report
wholesale, so the next run dropped it and the file ended "All skills pass
lint". These tests pin the qualification to `render_report` itself, on
synthetic results, so no vault is read.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "skill_lint_report_trust", ROOT / "scripts" / "skill_lint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sl = _load()


def _result(**over):
    base = {"generated_at": "2026-09-24T00:00:00", "total": 194,
            "dead": [], "missing_desc": [], "drift": [], "duplicates": [],
            "stale": [], "phantom": [], "missing_script": [], "injection": []}
    base.update(over)
    return base


def _table_rows(report: str) -> dict[str, list[str]]:
    lines = report.splitlines()
    start = lines.index("| category | count | action | is this count trustworthy? |")
    rows = {}
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows[cells[0].split(" (")[0]] = cells
    return rows


def test_every_category_row_carries_a_trust_cell_from_the_table():
    rows = _table_rows(sl.render_report(_result()))
    # Both directions: a new row without an entry, or an entry with no row,
    # is a category whose zero would print unqualified (or a stale claim).
    assert set(rows) == set(sl.CATEGORY_TRUST)
    for category, cells in rows.items():
        assert len(cells) == 4, cells
        assert cells[3] == sl.trust_cell(category)
        verdict, cause = sl.CATEGORY_TRUST[category]
        assert verdict in sl.TRUST_VERDICTS and cause.strip()


def test_drift_and_stale_say_untrustworthy_and_name_their_cause():
    rows = _table_rows(sl.render_report(_result()))
    drift, stale = rows["DRIFT"][3], rows["STALE"][3]
    assert drift.startswith(sl.TRUST_MARK["no"])
    assert "first token" in drift and "never tested" in drift
    assert stale.startswith(sl.TRUST_MARK["no"])
    assert "status: active" in stale and "before the age check" in stale
    assert sl.untrustworthy_categories() == ["DRIFT", "STALE"]


def test_all_zero_result_renders_a_qualified_verdict_not_clean():
    report = sl.render_report(_result())
    assert "## ✅ Clean" not in report
    assert "All skills pass lint" not in report
    verdict = report.split("## No findings on the checks that can fail", 1)[1]
    for category in ("DRIFT", "STALE"):
        assert f"**{category}** — 0 is not trustworthy" in verdict


def test_phantom_only_result_keeps_its_section_and_no_clean_verdict():
    report = sl.render_report(_result(
        phantom=[{"name": "websearch", "tools": ["web_search"]}]))
    assert "## PHANTOM_TOOL — 1 skills naming tools that do not exist" in report
    assert "| `websearch` | `web_search` |" in report
    assert "## ✅ Clean" not in report
    assert "All skills pass lint" not in report
    assert "## No findings on the checks that can fail" not in report


def test_drift_path_two_comment_promises_no_calibration_check():
    source = inspect.getsource(sl.check_description_drift)
    assert "calibration check below" not in source
