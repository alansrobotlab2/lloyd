"""Six lists name Mission Control's tabs, and nothing made them agree.

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

The sixth list is the one that decides whether the tab *draws*: ``PAGES`` in
``web/src/components/Layout.tsx``, the map ``PAGES[page]`` is looked up in and
the ``{PageComponent && …}`` guard renders. It was pinned by none of the five
until #1274, and its failure is the loudest-looking and quietest-actual of the
set — ``mc_navigate`` accepts the tab, the brief comes back, the frontend
switches ``page``, and the pane stays empty. The checks for it are at the
bottom of this file, behind a declared sticky-page constant rather than an
inline exception, because three pages are legitimately absent from the map.
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
ARCH_DOC = ROOT / "architecture" / "mission-control.md"


def _sidebar_tabs(src: str | None = None) -> set[str]:
    """The rendered tabs — the source of truth the other lists answer to."""
    if src is None:
        src = (WEB / "components" / "Sidebar.tsx").read_text(encoding="utf-8")
    m = re.search(r"export type Page\s*=\s*([^\n]+)", src)
    assert m, "no `export type Page` union in Sidebar.tsx"
    return set(re.findall(r"'([a-z_]+)'", m.group(1)))


def _hook_tabs(src: str | None = None) -> set[str]:
    if src is None:
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


# ── The sixth list: Layout.tsx's PAGES render map (#1274) ───────────────────
#
# `const PAGES: Record<string, React.FC>` (web/src/components/Layout.tsx:79)
# is the map `const PageComponent = PAGES[page]` (:315) reads, and the JSX
# guard `{PageComponent && (…)}` (:557) renders *nothing* when the key is
# missing — which is how a tab could be added to all five lists above, pass
# every assertion in this file, and still show a user an empty pane. That is
# the symptom `test_frontend_acts_on_every_tab_the_agent_may_ask_for` names
# for the hook list — "the navigate returns 200 and nothing moves" — one
# level deeper, and nothing parsed this file before #1274.
#
# `chat`, `ide` and `memory` are absent from the map on purpose: each is
# mounted elsewhere in Layout.tsx under its own `page === '<name>'` guard
# (:391, :528, :549) and kept mounted across tab switches — the IDE and memory
# behind `ideEverActive`/`memoryEverActive` so Monaco and the memory graph do
# not rebuild on every visit, the chat slots because the panel owns a LiveKit
# room per session. The exemption is therefore *declared* below instead of
# derived from the map's contents, so a fourth sticky page is a decision
# someone writes down here, and
# `test_a_declared_sticky_page_is_mounted_outside_the_map` refuses the
# declaration unless Layout.tsx really mounts it.

STICKY_PAGES = frozenset({"chat", "ide", "memory"})


def _render_map_body(src: str | None = None) -> str:
    r"""The text between `const PAGES … = {` and its closing brace."""
    if src is None:
        src = (WEB / "components" / "Layout.tsx").read_text(encoding="utf-8")
    m = re.search(r"const PAGES\s*:\s*Record<[^>]*>\s*=\s*\{(.*?)\n\}",
                  src, re.DOTALL)
    assert m, "no `const PAGES: Record<…> = { … }` render map in Layout.tsx"
    return m.group(1)


def _render_map_tabs(src: str | None = None) -> set[str]:
    r"""The keys of Layout.tsx's PAGES map, parsed from the map body only.

    Parsed, never substring-tested: the body carries the comment
    `// memory is rendered separately below — sticky, like the IDE.`
    (Layout.tsx:81), so `"memory" in <map body>` is true for a tab that has
    no entry in it. `^\s*([a-z_]+):` cannot match that line — it opens with
    `//` after the indent — and
    `test_parsing_the_map_skips_the_comment_that_names_a_sticky_tab` pins
    both halves: the comment is in the body, the tab is not in the keys.
    """
    return set(re.findall(r"^\s*([a-z_]+):", _render_map_body(src), re.MULTILINE))


def test_every_rendered_tab_has_a_render_map_entry():
    """Clause 1: a tab the five lists accept and the map lacks goes red, named.

    This is the assertion that did not exist. Before it, adding a tab to the
    `Page` union, both Python validators, the hook list and the summarizer
    registry — and forgetting Layout.tsx — passed the whole file while
    rendering an empty pane.
    `test_a_tab_the_five_lists_accept_but_the_render_map_lacks_goes_red`
    drives exactly that scenario; this is the check it drives.
    """
    missing = _sidebar_tabs() - _render_map_tabs() - set(STICKY_PAGES)
    print(f"Page union: {len(_sidebar_tabs())}; PAGES keys: {len(_render_map_tabs())}; "
          f"sticky: {len(STICKY_PAGES)}")
    assert not missing, (
        f"tabs the sidebar can switch to that Layout.tsx renders nothing for: "
        f"{sorted(missing)}. `PAGES[page]` is undefined for them, so the "
        "`{PageComponent && …}` guard in Layout.tsx renders an empty pane: "
        "mc_navigate returns 200, the navigate brief comes back, the frontend "
        "switches `page`, and the pane stays empty. Give each one a `PAGES` "
        "entry; only if it is mounted outside the map — sticky, like the IDE "
        "— does it belong in STICKY_PAGES instead."
    )


def test_the_render_map_carries_no_key_outside_the_page_union():
    """The other direction, mirroring `test_no_list_carries_a_tab_…`.

    A key with no tab behind it draws a page nobody can navigate to, and it is
    how a renamed tab leaves its component mounted under the old name.
    """
    stray = _render_map_tabs() - _sidebar_tabs()
    assert not stray, (
        f"Layout.tsx's PAGES map has keys Sidebar.tsx's `Page` union does not "
        f"render: {sorted(stray)}. Either the tab was renamed and the render "
        "entry is stale, or it was dropped from the sidebar and the component "
        "is still mounted for nobody."
    )


def test_the_sticky_exemption_is_exactly_the_declared_trio():
    """Clause 3: the exemption is a constant with a fixed value, not a search.

    Without this the map could be excused for any tab a test author happened
    to notice is missing, and the guard would erode tab by tab. Adding a
    fourth sticky page means editing this line as well as STICKY_PAGES, which
    is the point: it has to be stated twice to be true once.
    """
    assert set(STICKY_PAGES) == {"chat", "ide", "memory"}, (
        f"STICKY_PAGES is {sorted(STICKY_PAGES)}, expected exactly "
        "['chat', 'ide', 'memory']. Those three are absent from PAGES because "
        "they are keep-mounted elsewhere in Layout.tsx; a fourth has to be "
        "mounted the same way (see "
        "test_a_declared_sticky_page_is_mounted_outside_the_map), not just "
        "added to this set."
    )


def test_a_declared_sticky_page_is_mounted_outside_the_map():
    """A declared exemption must be a real mount, not an escape hatch.

    Sticky here means "rendered by a `page === '<name>'` comparison outside
    the map" — that is what `chat`, `ide` and `memory` do at Layout.tsx:391,
    :528 and :549. A name added to STICKY_PAGES with no such mount is a tab
    that renders nothing and a green suite.
    """
    src = (WEB / "components" / "Layout.tsx").read_text(encoding="utf-8")
    for page in sorted(STICKY_PAGES):
        assert f"page === '{page}'" in src, (
            f"'{page}' is declared in STICKY_PAGES but Layout.tsx never "
            f"compares `page === '{page}'`, so nothing mounts it outside the "
            "map and the exemption is hiding a tab that renders an empty "
            "pane. Mount it sticky like the IDE (`{ideEverActive && …}` with "
            "`page === 'ide' ? '' : 'hidden'`), or give it a PAGES entry."
        )


def test_parsing_the_map_skips_the_comment_that_names_a_sticky_tab():
    """Clause 4's trap: the map body *mentions* `memory`; the map has no such key.

    Both halves are asserted because each fails alone. A parser that fell
    back to substring presence would put `memory` in the keys and quietly
    shrink the union this file checks against; a parser that matched nothing
    at all would return an empty set, pass `"memory" not in keys` for free,
    and report the whole map missing — so the known keys are asserted too.
    """
    body = _render_map_body()
    assert "memory" in body, (
        "the comment naming `memory` is gone from the PAGES map body, so this "
        "test's negative half no longer proves anything about the parser — "
        "keep the comment, or move the trap to whatever the body now contains")
    keys = _render_map_tabs()
    assert {"dashboard", "services", "backlog", "browser"} <= keys, (
        f"the parser found only {sorted(keys)}; a map that parses to an empty "
        "or near-empty set makes every parity assertion above pass on the "
        "wrong denominator")
    assert "memory" not in keys, (
        "`memory` was parsed as a PAGES key, but it is only the comment at "
        f"Layout.tsx:81 — the map body is:\n{body.strip()}")


def test_a_tab_the_five_lists_accept_but_the_render_map_lacks_goes_red(
        tmp_path, monkeypatch):
    """The hole this file had, reproduced on a synthetic tree.

    A tab is added to the `Page` union, the hook list and all three Python
    lists — everything five of the original six lists can hold — and to
    nothing in Layout.tsx. That is the change an author makes when they add a
    tab and forget the sixth list, and before #1274 it passed every assertion
    here. So two things have to hold, and both are checked: the five older
    assertions still PASS (proving they cannot see the hole, which is what
    makes the new one non-redundant), and the render-map assertion fails
    naming the tab.

    Every mutation is asserted to have landed before the verdicts are read —
    a synthetic tab that never reached the parsed union would let all six
    assertions pass and report a clean result for a broken scenario.
    """
    fake = "notecards"
    fake_ui = {**mc_ui._SUMMARIZERS, fake: lambda: {}}
    web = tmp_path / "web" / "src"

    sidebar = (WEB / "components" / "Sidebar.tsx").read_text(encoding="utf-8")
    hook = (WEB / "hooks" / "useMcNavigationEvents.ts").read_text(encoding="utf-8")
    layout = (WEB / "components" / "Layout.tsx").read_text(encoding="utf-8")

    sidebar = re.sub(r"(export type Page\s*=\s*)([^\n]+)",
                     lambda m: m.group(0) + f" | '{fake}'", sidebar, count=1)
    assert fake in _sidebar_tabs(sidebar), (
        "the synthetic tab never reached the parsed Page union, so every "
        "verdict below is vacuous")
    hook = re.sub(r"(const VALID_TABS[^=]*=\s*new Set\(\[)(.*?)(\]\))",
                  lambda m: m.group(1) + m.group(2) + f"  '{fake}',\n" + m.group(3),
                  hook, count=1, flags=re.DOTALL)
    assert fake in _hook_tabs(hook), (
        "the synthetic tab never reached the hook's VALID_TABS set, so the "
        "frontend-side assertions below are vacuous")

    for rel, text in (("components/Sidebar.tsx", sidebar),
                      ("hooks/useMcNavigationEvents.ts", hook),
                      # The one file the author of a new tab forgets.
                      ("components/Layout.tsx", layout)):
        dest = web / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")

    me = sys.modules[__name__]
    monkeypatch.setattr(me, "WEB", web)
    monkeypatch.setattr(mc_state, "VALID_TABS", set(mc_state.VALID_TABS) | {fake})
    monkeypatch.setattr(mission_control_ui, "_VALID_TABS",
                        set(mission_control_ui._VALID_TABS) | {fake})
    monkeypatch.setattr(mc_ui, "_SUMMARIZERS", fake_ui)

    assert fake in _sidebar_tabs() and fake in _hook_tabs(), (
        "the patched WEB is not being read, so the scenario is not running "
        "against the synthetic tree at all")
    assert fake not in _render_map_tabs(), (
        "the synthetic tab reached the PAGES map, so the empty-pane scenario "
        "is not being reproduced and the failure expected below proves nothing")

    blind = (
        test_backend_mirror_accepts_every_rendered_tab,
        test_agent_may_navigate_to_every_rendered_tab,
        test_frontend_acts_on_every_tab_the_agent_may_ask_for,
        test_every_tab_has_a_summarizer,
        test_no_list_carries_a_tab_that_is_not_rendered,
    )
    unexpected = {}
    for fn in blind:
        try:
            fn()
        except AssertionError as exc:
            unexpected[fn.__name__] = str(exc).splitlines()[0]
    assert not unexpected, (
        f"the five pre-existing assertions caught it ({sorted(unexpected)}), "
        "so this scenario is no longer the hole #1274 was filed for — either "
        "they were widened (then this node is the duplicate and should be "
        "retired with them) or the synthetic tree is malformed: "
        f"{unexpected}")

    try:
        test_every_rendered_tab_has_a_render_map_entry()
    except AssertionError as exc:
        msg = str(exc)
    else:
        raise AssertionError(
            "a tab in the Page union and every validator, absent from PAGES, "
            "passed test_every_rendered_tab_has_a_render_map_entry: the "
            "render map is unpinned again and the empty pane is back")
    assert fake in msg, f"the failure does not name the tab: {msg}"
    assert "empty pane" in msg, (
        f"the failure does not say what the bug looks like: {msg}")


# The two sentences #1274 falsifies, in the form `architecture/mission-control.md`
# carried them. Matched on whitespace-flattened, lowercased text the way
# `test_dashboard_doc_claims.STALE_ARCH_CLAIMS` matches its own, so a reworded
# regression still lands on it; the value is the former text, quoted so a
# failure names the sentence rather than a keyword.
STALE_RENDER_MAP_CLAIMS = {
    "pinned by none of the five":
        "A sixth list names a tab and is pinned by none of the five",
    "passes every assertion":
        "a tab added to the five lists above but not to `PAGES` passes every "
        "assertion while rendering an empty pane",
}


def test_the_architecture_doc_no_longer_calls_the_render_map_unpinned():
    """Clause 5: the doc may not keep reporting a gap this file closed.

    `architecture/mission-control.md:46-52` described the render map as pinned
    by no test at all and said such a tab "passes every assertion". Both were
    true when #1274 filed them and both are false now, and a reader who
    believes either one stops looking for the guard that exists. Nothing else
    in the suite reads this doc for *tab* claims —
    `test_dashboard_doc_claims.py` reads it for the dashboard's — so this is
    the node that pins it.
    """
    assert ARCH_DOC.is_file(), f"{ARCH_DOC} is not a file"
    text = ARCH_DOC.read_text(encoding="utf-8")
    flat = re.sub(r"\s+", " ", text).lower()
    assert "const pages" in flat, (
        f"{ARCH_DOC} no longer names `const PAGES` at all, so the assertions "
        "below assert over a doc that has stopped describing the render map")
    for claim, former in STALE_RENDER_MAP_CLAIMS.items():
        assert claim not in flat, (
            f"architecture/mission-control.md still states {claim!r} (it read: "
            f"{former!r}). #1274 pins PAGES in this file; a doc that reports a "
            "closed gap as open is the stale copy that keeps it open.")
    # And the paragraph that describes the map has to say what now guards it:
    # deleting the stale sentence would otherwise satisfy the checks above by
    # describing nothing. Scoped to the paragraph, not a character window, so
    # rewrapping the doc cannot move the answer out of reach.
    start = text.index("const PAGES")
    end = text.find("\n\n", start)
    passage = re.sub(r"\s+", " ", text[start:] if end < 0 else text[start:end]).lower()
    assert "test_mc_tab_parity" in passage, (
        "the paragraph describing Layout.tsx's PAGES map no longer names "
        "tests/test_mc_tab_parity.py as what pins it, so a reader has nothing "
        "to follow and the passage re-opens the question #1274 closed")
    assert "sticky" in passage, (
        "the paragraph no longer states why three tabs are exempt from the "
        "map, so the next reader cannot tell the declared exemption from an "
        "oversight and will widen or delete it")
