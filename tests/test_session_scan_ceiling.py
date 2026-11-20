"""Three walkers over one session store, one order and one read budget (#4267).

`app/sessions_io.py::scan_session_heads` is now the only whole-file enumeration of
`SESSIONS_DIR`. Its three users each carried their own copy before this round:

  * `app/routers/dashboard.py::_scan_recent_sessions` — sorted
    `session_head_files()` by mtime and stopped at `_RECENT_KEPT` rows or
    `_RECENT_CEILING = 400` files opened;
  * `app/routers/mc_ui.py::_summarize_chat` — the same sort and its own
    `_CHAT_SCAN_CEILING = 400`;
  * `app/routers/mc_ui.py::_summarize_background` — a raw `SESSIONS_DIR.glob("*.json")`
    (sidecars included) and a BARE `if opened >= 400`, a number with no name.

Three literals on one store, none of them one constant: the copy that had to be
touched every time the directory outgrew its window was whichever one a person
happened to be standing in. `app/routers/sessions.py::_BACKGROUND_PLATFORM_SCAN_CEILING`
= 600 is deliberately NOT one of them and keeps its own value — it budgets 2 KB head
reads of the chat-shaped half only and its loop `continue`s rather than `break`s,
which is the semantics #1064's mis-labelled-exemption fixtures in
`tests/test_session_platform_checks.py` grade; folding it in is a decision of its own
and is recorded on #4267 as owed after landing.

What is pinned here is what the consolidation must preserve and what it buys: one
newest-first ORDER across all three surfaces over a directory whose mtimes do not
follow their names, a sidecar in none of them, a ceiling that bounds READS, and each
walker reaching the budget through the one published value.
"""
import ast
import json
import os
from pathlib import Path

import pytest

from app.harness.microcompact import REDUCTION_SIDECAR_SUFFIX

ROOT = Path(__file__).resolve().parent.parent

#: Newest first. The NAME stamps are scrambled against this order on purpose:
#: `..._160000_...` is the fourth-newest chat and `..._090000_...` the second, so a
#: walker that fell back to name order, or to `glob` order, returns a list that is
#: not a subsequence of this one and fails.
#: (kind, name stamp, slug, `source` if the row carries one)
ROWS = [
    ("sidecar", "170000", "gggggg", None),
    ("chat", "090000", "aaaaaa", None),
    ("background", "130000", "autocode_01", "autocode"),
    ("chat", "140000", "bbbbbb", None),
    ("bench", "070000", "bench_02", "bench"),
    ("chat", "110000", "cccccc", None),
    ("background", "080000", "worker_03", "worker"),
    ("chat", "160000", "dddddd", None),
    ("background", "120000", "autotriage_04", "autonomy-task:9"),
    ("chat", "100000", "eeeeee", None),
    ("chat", "150000", "ffffff", None),
]
PLATFORM = {"chat": "mission-control", "background": "worker",
            "bench": "worker", "sidecar": None}


def _sid(stamp, slug):
    """A session id, as the listings report it — the stem of its file."""
    return f"20261118_{stamp}_{slug}"


def _name(kind, stamp, slug):
    """And its name on disk. A sidecar's suffix already carries the `.json`."""
    base = _sid(stamp, slug)
    return base + REDUCTION_SIDECAR_SUFFIX if kind == "sidecar" else base + ".json"


def _ids(*kinds):
    return [_sid(stamp, slug) for k, stamp, slug, _ in ROWS if k in kinds]


#: The order every walker must agree with. The sidecar is in the directory and in no
#: listing, so it is in no expected sequence either.
NEWEST_FIRST = _ids("chat", "background", "bench")
CHAT_IDS = _ids("chat")
#: Rows the Background brief may list: the bench trial among them is withheld
#: (#2648), which `tests/test_mc_background_brief_parity.py` pins.
BACKGROUND_RUN_IDS = _ids("background")
SIDECAR = _name("sidecar", "170000", "gggggg")
SIDECAR_STEM = _sid("170000", "gggggg")


@pytest.fixture
def store(tmp_path, monkeypatch):
    """The ROWS above written to a directory, mtimes set against their names.

    No `last_active` is written, so the dashboard's own re-sort on `_last_active_ts`
    falls back to mtime and what this file measures is the WALK's order; the
    `last_active`-beats-mtime tie-break is pinned by
    `tests/test_session_platform_checks.py::test_a_mixed_naive_and_aware_directory_lists_in_instant_order`.
    """
    d = tmp_path / "sessions"
    d.mkdir()
    for i, (kind, stamp, slug, source) in enumerate(ROWS):
        path = d / _name(kind, stamp, slug)
        if kind == "sidecar":
            body = {"schema": "microcompact-reduced.v1",
                    "cleared_call_ids": ["toolu_01A"]}
        else:
            body = {"session_id": path.stem, "title": path.stem, "preview": "p",
                    "platform": PLATFORM[kind], "model": "test-model",
                    "messages": []}
            if source:
                body["source"] = source
        path.write_text(json.dumps(body), encoding="utf-8")
        mtime = 1_763_000_000 + (len(ROWS) - i) * 60
        os.utime(path, (mtime, mtime))

    from app.routers import mc_ui
    import app.paths as app_paths
    monkeypatch.setattr(mc_ui, "SESSIONS_DIR", d)
    # The dashboard and the Chat brief import the directory from `app.paths` inside
    # the call, so that is the binding to move.
    monkeypatch.setattr(app_paths, "SESSIONS_DIR", d)
    return d


@pytest.fixture
def reads(monkeypatch):
    """Every `Path.read_text` call in the process, by filename.

    The instrument `tests/test_mc_tab_parity.py` uses: a read budget that is only
    believed and never counted is how three 400s disagreed unnoticed.
    """
    seen: list[str] = []
    real = Path.read_text

    def _counting(self, *a, **kw):
        seen.append(self.name)
        return real(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", _counting)
    return seen


def _in_order(got, want):
    """True if `got` is `want` filtered to its own members — newest-first, no ties."""
    return got == [w for w in want if w in set(got)]


def test_all_three_walkers_return_their_rows_in_one_newest_first_order(store):
    """Clause 3: one walk, so one order — not three orders that happen to agree."""
    from app.routers import dashboard as dash, mc_ui

    chats = [r["id"] for r in mc_ui._summarize_chat()["recent_sessions"]]
    runs = [r["id"] for r in mc_ui._summarize_background()["recent"]]
    recent = [r["session_id"] for r in dash._scan_recent_sessions()]

    for label, got in (("the Chat brief", chats), ("the Background brief", runs),
                       ("the dashboard's recent panel", recent)):
        assert got, f"{label} returned nothing from a store of 10 session heads"
        assert _in_order(got, NEWEST_FIRST), (
            f"{label} returned {got}, which is not newest-first in the store's order "
            f"{NEWEST_FIRST}")
    assert chats == [i for i in NEWEST_FIRST if i in set(CHAT_IDS)][:5]
    assert runs == [i for i in NEWEST_FIRST if i in set(BACKGROUND_RUN_IDS)]
    assert recent == [i for i in NEWEST_FIRST if i in set(CHAT_IDS)]


def test_no_walker_lists_a_reduction_sidecar(store, reads):
    """#3815 inside the shared walk: the sidecar is the NEWEST file in the store, so a
    walker that lost the head filter would list it FIRST — and it would also charge it
    against the budget, which is what the brief's bare `400` used to spend.
    """
    from app.routers import dashboard as dash, mc_ui

    newest = max(store.iterdir(), key=lambda p: p.stat().st_mtime)
    assert newest.name == SIDECAR, f"fixture bug: {newest.name} is the newest file"

    listed = {r["id"] for r in mc_ui._summarize_chat()["recent_sessions"]}
    listed |= {r["id"] for r in mc_ui._summarize_background()["recent"]}
    listed |= {r["session_id"] for r in dash._scan_recent_sessions()}
    assert not [i for i in listed if REDUCTION_SIDECAR_SUFFIX in i], sorted(listed)
    assert SIDECAR_STEM not in listed, (
        "a sidecar reached a listing under its head's stem")
    assert SIDECAR not in reads, f"the shared walk opened the sidecar: {reads}"
    assert mc_ui._summarize_chat()["total_session_count"] == len(NEWEST_FIRST)
    assert mc_ui._summarize_background()["by_source"].get("unknown", 0) == 0


def test_the_walk_opens_no_more_files_than_its_ceiling(store, reads):
    """Clause 3's other half: the ceiling is a PARAMETER of the one walk, and it
    bounds READS — a name it skips costs nothing, so `ceiling=2` means two parses.
    """
    from app import sessions_io

    out = list(sessions_io.scan_session_heads(store, ceiling=2))
    assert [p.stem for p, _ in out] == NEWEST_FIRST[:2], out
    assert len(reads) == 2, reads

    with_skip = list(sessions_io.scan_session_heads(
        store, ceiling=2,
        skip_names=sessions_io.is_background_session_name))
    assert [p.stem for p, _ in with_skip] == CHAT_IDS[:2], with_skip
    assert len(reads) == 4, (
        f"a name the walker skips charged itself against the budget: {reads}")


def test_the_background_brief_takes_the_published_budget(store, reads, monkeypatch):
    """Clause 4, behavioural: lower the ONE published number and this walker stops
    earlier. A walker that spelled its own `400` would read the fixture whole.
    """
    from app.routers import mc_ui
    monkeypatch.setattr(mc_ui, "SESSION_SCAN_CEILING", 2)
    out = mc_ui._summarize_background()
    assert len(reads) == 2, reads
    # The budget is reads, not rows: of those two files one is a conversation, which
    # the brief does not count but did open — as the tab's own pass does too.
    assert out["sessions_total"] == 1, out


def test_the_chat_brief_takes_the_published_budget(store, reads, monkeypatch):
    from app.routers import mc_ui
    monkeypatch.setattr(mc_ui, "SESSION_SCAN_CEILING", 2)
    out = mc_ui._summarize_chat()
    assert len(reads) == 2, reads
    assert [r["id"] for r in out["recent_sessions"]] == CHAT_IDS[:2]


def test_the_dashboard_recent_panel_takes_the_published_budget(
        store, reads, monkeypatch):
    from app import sessions_io
    from app.routers import dashboard as dash
    monkeypatch.setattr(sessions_io, "SESSION_SCAN_CEILING", 2)
    rows = dash._scan_recent_sessions()
    assert len(reads) == 2, reads
    assert [r["session_id"] for r in rows] == CHAT_IDS[:2]


def _tree(rel):
    return ast.parse((ROOT / rel).read_text(encoding="utf-8"))


def _spelled_read_budgets(rel):
    """Every number spelled into CODE as a read budget in one module.

    Two shapes, the two the routers used: a `*_CEILING = <number>` binding, and a
    counter compared against a bare number (`if opened >= 400`).

    This reads the AST rather than grepping the text because both modules now EXPLAIN
    in prose what their old numbers were — `_RECENT_CEILING = 400`, `_CHAT_SCAN_CEILING
    = 400`, `opened >= 400` — and a grep would grade that documentation as a relapse.
    The clause is about what the code spends, so that is what is measured.
    """
    hits = []
    for node in ast.walk(_tree(rel)):
        targets, value = [], None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign):
            targets, value = [node.target], node.value
        for target in targets:
            if (isinstance(target, ast.Name)
                    and target.id.upper().endswith("_CEILING")
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, (int, float))):
                hits.append(f"{target.id} = {value.value}")
        if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
                and node.left.id in {"opened", "files_opened", "reads_opened"}):
            for comp in node.comparators:
                if isinstance(comp, ast.Constant) and isinstance(comp.value, int):
                    hits.append(f"{node.left.id} >= {comp.value}")
    return hits


def _names_used(rel):
    """Identifiers a module mentions in code — bare, or as `sessions_io.X`. The
    dashboard reaches the published value as an attribute of the module it imports,
    the Mission Control router as a `from`-import, and both count as one reference."""
    return {n.id for n in ast.walk(_tree(rel)) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(_tree(rel)) if isinstance(n, ast.Attribute)}


def test_no_router_spells_its_own_read_budget():
    """Clause 4, textual: no numeric read budget left in either router's code, both
    naming the one value `app/sessions_io.py` publishes, and the 600 in `sessions.py`
    still its own number on purpose.
    """
    import app.sessions_io as sessions_io
    from app.routers import sessions as sessions_router

    for rel in ("app/routers/mc_ui.py", "app/routers/dashboard.py"):
        assert _spelled_read_budgets(rel) == [], (
            f"{rel} budgets its own reads again: {_spelled_read_budgets(rel)} — "
            "sessions_io.SESSION_SCAN_CEILING is the one value")
        assert "SESSION_SCAN_CEILING" in _names_used(rel), (
            f"{rel} no longer reaches sessions_io for its read budget")
    assert sessions_io.SESSION_SCAN_CEILING == 400
    assert sessions_router._BACKGROUND_PLATFORM_SCAN_CEILING == 600, (
        "the background listing's head-window ceiling was folded into the whole-file "
        "walk: it budgets reads of the chat-shaped half only and `continue`s rather "
        "than breaking, which #1064's fixtures grade")
