"""#1685 — dashboard cards must not hold their grid track open on a phone.

Two panels on Mission Control's dashboard rendered 651 px wide inside a 288 px
column at a 320 px viewport. The cause was not a missing media query: a grid
item's `min-width` is `auto`, so the automatic-minimum-size rule floors each
card at its own content, and one unbreakable line (the worker stat row's four
columns) widened the track and every card sharing it.

WHAT THIS FILE PROVES: that the declarations holding these fixes in place are
still on the elements that need them, read out of the component's own source —
#1685's two (Panel's released floor, the worker stat row's column claim) and
#2398's one (TaskLine's note span, which a scheduler-written `blocked` reason of
322 px was flooring the whole `Automation & work` card open).

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


def note_span_classes() -> list[str]:
    """TaskLine's note class list, read out of the component.

    Anchored on the `{note}` interpolation rather than on any class name: the
    class list is the thing under test, and #2398's regression was exactly a
    change to it. `note` is `task.blocked` on a held row (`showReason`), else
    `task.frequency` — both scheduler-written strings, neither of them bounded.
    """
    span = re.search(r'<span className="([^"]*)">\{note\}</span>', SOURCE)
    assert span, "TaskLine's {note} span was not found; update this test"
    return span.group(1).split()


def test_task_line_note_is_not_floored_at_its_own_text():
    """The red tree this pins (#2398, both of its dashboard pins at once).

    `task.blocked` is free text the scheduler writes, so the row's width used to
    be a fact about the fleet's vocabulary rather than about the layout. Measured
    on the live tree 2026-10-08, a held row reading `waiting on #38 (inside its
    36 h stale_bypass window)` put 322 px of unbreakable line into a `flex
    items-center` row whose box is 262 px: the card's content box reached 406 px
    inside its own 235 px track (the desktop pin's `clips its content at 235px`),
    the section reported 407 px inside 288 px at a 320 px viewport, and because
    the note was the row's only unfloored item, `{task.name}` was squeezed to 0
    px wide — the label the row exists to show.
    """
    classes = note_span_classes()
    assert "min-w-0" in classes, (
        f"TaskLine's note span is {classes}: a flex item whose overflow is "
        "VISIBLE takes `min-width: auto`, which resolves to its min-content, so "
        "one long `blocked` reason floors the row, the card and the track (#2398)"
    )
    assert "flex-shrink-0" not in classes, (
        f"TaskLine's note span is {classes}: an item that refuses to shrink "
        "cannot be narrowed by the flex algorithm whatever else is on it"
    )
    assert "whitespace-nowrap" not in classes, (
        f"TaskLine's note span is {classes}: `nowrap` makes the whole reason one "
        "unbreakable line, and below `sm` that line is the phone's only line"
    )
    assert {"whitespace-normal", "break-words"} <= set(classes), (
        f"TaskLine's note span is {classes}: `whitespace-normal break-words` is "
        "the wrapping base that keeps a phone-width row inside its card"
    )
    assert "sm:truncate" in classes, (
        f"TaskLine's note span is {classes}: without the `sm` ellipsis every "
        "desktop row with a reason stops being one line"
    )
