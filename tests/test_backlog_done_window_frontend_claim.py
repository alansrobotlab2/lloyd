"""The board's done window: what `BacklogPage.tsx` is allowed to ask the server for.

Item #1213. Alan's ruling, 2026-09-17: the Done column shows 7 days by default
and a checkbox brings back the rest, with the cutting done by the route so the
payload shrinks with it.

Why this file greps a `.tsx` instead of rendering it: `web/` has no test runner.
`web/package.json` declares no `test` script and neither vitest nor jest, and
`find web/src -name '*.test.*'` is empty. Adding a runner means editing
`package.json`, a build input this loop cannot land, and the item keeps that
decision Alan's. So clause 5 gets the house source-claim pattern — the one
`tests/test_code_graph_doc_claims.py` and `tests/test_automod_doc_claims.py`
use: read the file, assert the claim is present, assert the claim *beside it*
that makes it the right claim rather than a similar-looking one.

What that can and cannot pin. It can pin that the request is built a certain
way, that the checkbox exists and is bound to the state that changes the
request, and that the window is a per-request expression rather than a constant
— those are properties of the text, and the text is the implementation. It
cannot pin a render, so it does not claim to: no assertion here says anything
about what appears on screen, and if a runner ever lands in `web/`, the checkbox
interaction belongs there and this file shrinks.

Each test cites the source node it fails on, so a refactor that moves a claim
renames it or reformats it across lines gets a named failure rather than a
silent green.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "web" / "src" / "components" / "pages" / "BacklogPage.tsx"
API = ROOT / "web" / "src" / "api.ts"
ROUTE = ROOT / "app" / "routers" / "backlog.py"


def _text(path: Path) -> str:
    """The file, or a failure that names the path rather than an OSError."""
    assert path.exists(), f"{path.relative_to(ROOT)} is gone; this test's claims are about it"
    return path.read_text(encoding="utf-8")


SOURCE = _text(PAGE)


def _window_request_call() -> str:
    """The `api.backlogTasks({...})` argument object, verbatim.

    Anchored on the call rather than searched globally: a `done_since` in a
    comment, a dead branch, or a second call that never runs would each satisfy
    a plain substring search, and none of them is the request the page sends.

    Brace-counted, not regexed to the first `})` — the object's own spreads
    contain `{})`, so the nearest closing brace is inside the first line and the
    captured text would stop above the clause being pinned.
    """
    anchor = "api.backlogTasks({"
    start = SOURCE.find(anchor)
    assert start >= 0, (
        "BacklogPage.tsx no longer builds its task request as "
        "`api.backlogTasks({...})`; update this test to read whatever replaced it "
        "rather than deleting the assertion"
    )
    depth, i = 1, start + len(anchor)
    while i < len(SOURCE) and depth:
        if SOURCE[i] == "{":
            depth += 1
        elif SOURCE[i] == "}":
            depth -= 1
        i += 1
    assert depth == 0, "unbalanced braces in the task-request object"
    return SOURCE[start + len(anchor) : i - 1]


def _callback(name: str) -> tuple[str, str]:
    """The body and the dependency list of `const <name> = useCallback(async () => …)`.

    `\n  }, [` as the terminator because that is the only place at this indent
    inside the component: every nested block in these callbacks closes deeper, so
    the first one is the callback's own end.
    """
    m = re.search(rf"const {name} = useCallback\(async \(\) => \{{(.*?)\n  \}}, \[([^\]]*)\]", SOURCE, re.S)
    assert m, f"`{name}` is no longer a useCallback this test can read"
    return m.group(1), m.group(2)


def _callback_body(name: str) -> str:
    return _callback(name)[0]


# ── clause 5: the seven-day default ────────────────────────────────────────


def test_the_window_is_seven_days_and_named_as_a_constant():
    """Alan's ruling is a number, so the number is pinned, not paraphrased.

    A named constant rather than a literal at the call site is the shape the
    item asked for: if the ruling changes — and the item's own calibration says
    7 days is a 41 % cut, so it may — there is one line to change and this test
    is what says it moved.
    """
    m = re.search(r"const DONE_WINDOW_DAYS\s*=\s*(\d+)\s*;", SOURCE)
    assert m, "DONE_WINDOW_DAYS is not a declared constant"
    assert m.group(1) == "7", f"the default window is {m.group(1)} days, not the 7 Alan ruled"
    assert "DONE_WINDOW_DAYS" in _done_since_fn(), (
        "the window constant is declared but the date builder never subtracts it, "
        "so the 7 above and the request in the page are two unrelated facts"
    )


def _done_since_fn() -> str:
    """The body of the function that produces today − window, in YYYY-MM-DD."""
    m = re.search(r"function doneSinceDate\(\)\s*:\s*string\s*\{(.*?)\n\}", SOURCE, re.S)
    assert m, (
        "doneSinceDate() is gone or its signature changed; it is what makes the "
        "window slide, so pin it to whatever replaces it"
    )
    return m.group(1)


def test_the_date_is_local_yyyy_mm_dd_not_a_utc_timestamp_slice():
    """`toISOString().slice(0, 10)` would be one line shorter and wrong here.

    It formats in UTC, and this box runs UTC−7: between 00:00 and 07:00 local it
    names *yesterday*, silently shifting the whole window by a day for the part
    of the day the person looking at the board is most likely to be in it. The
    value must be built from local date parts and zero-padded — the route
    accepts exactly `^\\d{4}-\\d{2}-\\d{2}$` and ignores anything else, so a
    single-digit month is not a slightly-off date, it is no window at all.
    """
    body = _done_since_fn()
    assert "toISOString" not in body, "UTC-formatted date: off by up to a day on a UTC−7 box"
    assert re.search(r"setDate\(\s*d\.getDate\(\)\s*-\s*DONE_WINDOW_DAYS\s*\)", body), (
        "the window is not subtracted in calendar days, so the cut-off is a "
        "wall-clock offset and drifts across a DST boundary"
    )
    assert "getMonth()" in body, "month is not taken from the local calendar"
    assert body.count("padStart(2, \"0\")") >= 2, (
        "month and day must both be zero-padded: the route's regex rejects "
        "'2026-9-1', and a rejected parameter means the window silently vanishes"
    )


def test_the_date_is_computed_per_request_not_once_at_mount():
    """The one clause here that a refactor breaks without changing behaviour.

    The page refetches on `setInterval(loadData, 15_000)` — see the effect
    below, which is not gated on `document.visibilityState` — so a value taken
    once at mount is re-sent forever. On a page left open across midnight that
    is not a stale-by-an-hour bug: the window stops sliding, and after a month
    the board is asking for a month-old week. Hoisting the call to module scope,
    to a `useState` initialiser, or into a `useMemo` with no clock in its deps
    all look like tidy refactors and all reintroduce it.

    So the assertion is about *where the call lives*: inside the callback that
    the interval invokes, and nowhere else.
    """
    # `(?<!function )` because the declaration's own signature — `function
    # doneSinceDate(): string` — contains the call as a substring.
    call_sites = [m.start() for m in re.finditer(r"(?<!function )doneSinceDate\(\)", SOURCE)]
    assert call_sites, "doneSinceDate() is never called, so the page never sends done_since"
    body_start = SOURCE.index("const loadData = useCallback")
    body = _callback_body("loadData")
    in_load = [p for p in call_sites if body_start < p < body_start + len(body) + 60]
    assert in_load, "`done_since`'s date is not computed inside `loadData`"
    # Exactly one call site: a second one is the hoist this clause forbids.
    assert len(call_sites) == 1, (
        f"doneSinceDate() is called {len(call_sites)} times; a second call site is "
        "usually the constant that stops the window sliding"
    )
    assert re.search(r"setInterval\(\s*loadData\s*,\s*15_000\s*\)", SOURCE), (
        "the 15-second poll is gone — if that is deliberate, this test's premise "
        "about a page open across midnight is the thing to renegotiate, not this line"
    )


# ── clause 5: the checkbox, and the two omissions ──────────────────────────


def test_the_request_sends_done_since_only_when_the_window_is_active():
    """The parameter is *conditional*, not always sent with a permissive default.

    The route ignores a bad value but honours a good one, so a request that
    always carried a date could never ask for the full history — the checkbox
    would have nothing to do. The spread form (`...(guard ? {done_since: …} : {})`)
    is the assertion, because the failure mode is a value of `""` or `undefined`
    surviving into the query string, which the route reads as "absent" today and
    which would read as "today" the moment anyone tightened its parsing.
    """
    call = _window_request_call()
    m = re.search(r"\.\.\.\(\s*(\w+)\s*\?\s*\{\s*done_since:\s*doneSinceDate\(\)\s*\}\s*:\s*\{\s*\}\s*\)", call)
    assert m, (
        "done_since is not spread out of the request when its guard is false; "
        f"the request object reads: {call.strip()!r}"
    )
    guard = m.group(1)
    assert guard == "windowActive", f"unexpected guard name {guard!r}"


def test_the_window_guard_is_the_checkbox_and_the_absence_of_a_search():
    """`windowActive` must be false in exactly the two cases that drop the window.

    Second half is the item's reachability trap 2: `?q=` is matched server-side
    against whole bodies, so if the window were applied during a search, a query
    for an item closed last month would return nothing and look like the item
    never existed. A user searching for the thing the window is hiding is the
    common case, not the corner.
    """
    m = re.search(r"const windowActive\s*=\s*([^;]+);", SOURCE)
    assert m, "windowActive is not a declared const"
    expr = re.sub(r"\s+", "", m.group(1))
    assert expr == "!includeAllDone&&!searchParam", (
        f"windowActive reads {expr!r}; it must be false when the box is ticked "
        "and false while a search is running, in either order"
    )
    assert "windowActive" in _window_request_call(), (
        "the guard is computed but the request never consults it"
    )


def test_the_checkbox_exists_and_drives_the_state_the_guard_reads():
    """'include all done' is a real control wired to the guard's own state.

    Three separate ways this goes decorative, each pinned: an input that is not
    a checkbox, a checkbox whose `onChange` writes a different state than the
    one `windowActive` tests, or a controlled input with no `checked` (which
    renders once and never moves again).
    """
    # The page has five other `<label>`s (the modals' field captions), so the
    # block is identified by the state it writes, not by being the first label.
    blocks = [m.group(1) for m in re.finditer(r"<label\b(.*?)</label>", SOURCE, re.S)]
    matching = [b for b in blocks if "includeAllDone" in b]
    assert matching, (
        "no <label> contains `includeAllDone`, so the checkbox is not bound to "
        "the state the request guard reads"
    )
    assert len(matching) == 1, f"{len(matching)} labels write includeAllDone"
    block = matching[0]
    assert 'type="checkbox"' in block, "the done-window control is not a checkbox"
    assert "include all done" in block, "the control's visible text must be 'include all done'"
    assert re.search(r"checked=\{includeAllDone\}", block), "the checkbox is uncontrolled"
    assert re.search(r"onChange=\{\(e\)\s*=>\s*setIncludeAllDone\(e\.target\.checked\)\}", block), (
        "the checkbox does not write `includeAllDone`, the state `windowActive` reads"
    )
    assert re.search(r"disabled=\{!!searchParam\}", block), (
        "the control is left live during a search, when a search has already "
        "dropped the window and ticking it cannot change the request"
    )


def test_ticking_the_box_refetches_instead_of_waiting_for_the_next_poll():
    """`includeAllDone` must be a `loadData` dependency.

    Without it the click changes the guard and nothing else: the board keeps
    showing the windowed set until the next tick of the 15-second interval, which
    reads as a checkbox that does not work.
    """
    _, deps_raw = _callback("loadData")
    deps = re.sub(r"\s+", "", deps_raw)
    assert "includeAllDone" in deps, f"loadData deps are {deps!r}"


# ── the seam: the client's param bag really reaches the route ───────────────


def test_the_api_client_passes_params_through_rather_than_whitelisting_them():
    """`backlogTasks(params?: Record<string, string>)` is why no client change was needed.

    If that signature ever becomes a named-argument list, the page would keep
    building a correct `done_since` and the client would drop it — a filtered
    board that quietly stops filtering, with every assertion above still green.
    """
    src = _text(API)
    m = re.search(r"backlogTasks\(([^)]*)\)", src)
    assert m, "api.backlogTasks is gone"
    assert re.search(r"params\?\s*:\s*Record<string,\s*string>", m.group(1)), (
        f"backlogTasks takes {m.group(1)!r}; a named-argument list would silently "
        "discard done_since before the request is built"
    )
    assert "URLSearchParams" in src, "the param bag is no longer serialised into the query string"


def test_the_route_still_admits_the_parameter_the_page_sends():
    """The claim the two files share: the query name, spelled the same on both sides.

    A one-sided change here is the whole seam failing — the front end asking for
    a window the route no longer takes is a board that shows every closed item
    while every page-level test still passes. The behaviour behind it belongs to
    `tests/test_backlog_route_done_window.py`, which drives the route; this only
    pins that the two files still name the same thing.
    """
    route = _text(ROUTE)
    m = re.search(r"def backlog_tasks\(([^)]*)\)", route)
    assert m, "cannot read the route signature"
    assert "done_since" in m.group(1), (
        f"GET /api/backlog/tasks takes {m.group(1)!r} — no done_since, so the "
        "parameter BacklogPage.tsx sends is ignored by FastAPI"
    )
