"""Four lists name Mission Control's tabs, and nothing made them agree.

`web/src/components/Sidebar.tsx` renders a tab; three other places decide
what may be done with it, and each was written by hand:

  * ``app.mc_state.VALID_TABS`` — what the frontend is allowed to *report*.
  * ``agent_mcp.mission_control_ui._VALID_TABS`` — what the agent may ask for.
  * ``VALID_TABS`` in ``web/src/hooks/useMcNavigationEvents.ts`` — what the
    frontend will act on when the agent asks.

Drift between them fails in a way nothing reports. On 2026-09-09 `browser`
was in the `Page` union and in none of the three, so a user sitting on the
Browser tab made ``POST /api/mc/state`` return 400 — and ``useMcStateSync``
swallows the failure and records the payload as sent. The mirror kept
serving whichever tab they had been on before, so ``mc_get_state`` answered
confidently and wrongly for as long as they stayed there. `dashboard` was
missing from two of the three, which is the quieter half of the same bug:
the backend has carried a ``_summarize_dashboard`` all along for a tab the
agent was refused and the frontend would have ignored.

The summarizer registry is pinned to the same set because a tab missing
from it is not an error either — ``_summarize_tab`` returns ``{}`` and the
agent is told nothing about where it just sent the user.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import mc_state  # noqa: E402
from app.routers import mc_ui  # noqa: E402
from agent_mcp import mission_control_ui  # noqa: E402

WEB = ROOT / "web" / "src"


def _sidebar_tabs() -> set[str]:
    """The rendered tabs — the source of truth the other three answer to."""
    src = (WEB / "components" / "Sidebar.tsx").read_text(encoding="utf-8")
    m = re.search(r"export type Page\s*=\s*([^\n]+)", src)
    assert m, "no `export type Page` union in Sidebar.tsx"
    return set(re.findall(r"'([a-z_]+)'", m.group(1)))


def _hook_tabs() -> set[str]:
    src = (WEB / "hooks" / "useMcNavigationEvents.ts").read_text(encoding="utf-8")
    m = re.search(r"const VALID_TABS[^=]*=\s*new Set\(\[(.*?)\]\)", src, re.DOTALL)
    assert m, "no VALID_TABS set in useMcNavigationEvents.ts"
    return set(re.findall(r"'([a-z_]+)'", m.group(1)))


def test_backend_mirror_accepts_every_rendered_tab():
    missing = _sidebar_tabs() - mc_state.VALID_TABS
    assert not missing, (
        f"tabs the user can be on that app.mc_state rejects: {sorted(missing)}. "
        "POST /api/mc/state 400s and the mirror keeps reporting the previous tab."
    )


def test_agent_may_navigate_to_every_rendered_tab():
    missing = _sidebar_tabs() - set(mission_control_ui._VALID_TABS)
    assert not missing, (
        f"tabs mc_navigate refuses: {sorted(missing)}"
    )


def test_frontend_acts_on_every_tab_the_agent_may_ask_for():
    missing = set(mission_control_ui._VALID_TABS) - _hook_tabs()
    assert not missing, (
        f"tabs mc_navigate accepts and the frontend silently ignores: "
        f"{sorted(missing)}. The navigate returns 200 and nothing moves."
    )


def test_every_tab_has_a_summarizer():
    missing = mc_state.VALID_TABS - set(mc_ui._SUMMARIZERS)
    assert not missing, (
        f"tabs with no navigate summary: {sorted(missing)}. _summarize_tab "
        "returns {} and the agent learns nothing about where it sent the user. "
        "Register `lambda: {}` if there is genuinely nothing to say."
    )


def test_no_list_carries_a_tab_that_is_not_rendered():
    """The other direction: a tab that exists only in a validator.

    Harmless on its own, but it is how a renamed tab hides — the old name
    keeps validating while nothing renders it.
    """
    rendered = _sidebar_tabs()
    for name, tabs in (
        ("app.mc_state.VALID_TABS", mc_state.VALID_TABS),
        ("mission_control_ui._VALID_TABS", set(mission_control_ui._VALID_TABS)),
        ("useMcNavigationEvents VALID_TABS", _hook_tabs()),
        ("mc_ui._SUMMARIZERS", set(mc_ui._SUMMARIZERS)),
    ):
        stray = tabs - rendered
        assert not stray, f"{name} names tabs Sidebar.tsx does not render: {sorted(stray)}"


def test_browser_tab_is_the_case_that_regressed():
    """Explicit, because the parity tests above pass the day someone deletes
    the Browser tab from all four lists at once."""
    assert "browser" in _sidebar_tabs()
    assert "browser" in mc_state.VALID_TABS
    assert "browser" in mission_control_ui._VALID_TABS
    assert "browser" in _hook_tabs()
    assert "browser" in mc_ui._SUMMARIZERS
