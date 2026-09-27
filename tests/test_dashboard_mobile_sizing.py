"""#1685 — dashboard cards must not hold their grid track open on a phone.

Two panels on Mission Control's dashboard rendered 651 px wide inside a 288 px
column at a 320 px viewport. The cause was not a missing media query: a grid
item's `min-width` is `auto`, so the automatic-minimum-size rule floors each
card at its own content, and one unbreakable line (the worker stat row's four
columns) widened the track and every card sharing it.

WHAT THIS FILE PROVES: that the two declarations holding the fix in place are
still on the elements that need them, read out of the component's own source.

WHAT IT DELIBERATELY DOES NOT PROVE: that the page does not overflow. No test
here renders the dashboard. That measurement is
`scripts/maintenance/dashboard_mobile_probe.py`, which drives headless chromium
over the real build at 320/360/390/414 px and exits non-zero on overflow.

An earlier draft of this file tried to measure a fixture built from the same
utility classes in chromium and assert overflow. It was cut: the fixture only
overflowed when its labels were padded out, so the assertion's threshold was
content the test itself chose — a check that reports the number the author
tuned it to report. The probe measures the real page instead.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "web" / "src" / "components" / "pages" / "DashboardPage.tsx"
SOURCE = SRC.read_text(encoding="utf-8")


def panel_root_classes() -> list[str]:
    """The class list on Panel's own root div, read out of the component."""
    body = re.search(r"function Panel\([\s\S]*?\n\}", SOURCE)
    assert body, "Panel() is gone from DashboardPage.tsx; update this test"
    root = re.search(r"cn\(\s*'([^']*)'", body.group(0))
    assert root, "Panel no longer builds its root className with cn('...')"
    return root.group(1).split()


def stat_row_classes() -> list[str]:
    """The worker stat row's grid classes inside WorkersPanel."""
    # Anchored on its first tile: DashboardPage.tsx has four other
    # `grid-cols-2` grids, and a looser match would silently assert about one
    # of those instead.
    row = re.search(
        r'className="(grid[^"]*grid-cols-2[^"]*)">\s*<div>\s*<div[^>]*>In flight',
        SOURCE,
    )
    assert row, "the worker stat row grid was not found; update this test"
    return row.group(1).split()


def test_panel_card_is_not_floored_at_its_content():
    classes = panel_root_classes()
    assert "min-w-0" in classes, (
        f"Panel's root div is {classes}: a grid item's automatic minimum size is "
        "its content, so one unbreakable line in one card widens every card in "
        "the track (#1685)"
    )


def test_worker_stat_row_does_not_claim_four_columns_at_phone_width():
    """Four tiles across a 288 px column is 47 px each, which truncates every
    label — and its max-content is what held the track open."""
    classes = stat_row_classes()
    assert "grid-cols-4" not in classes, f"stat row still claims 4 columns: {classes}"
    assert "sm:grid-cols-4" in classes, f"four columns from sm up was lost: {classes}"
