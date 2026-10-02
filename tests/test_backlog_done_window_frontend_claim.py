r"""The board's done window: what `BacklogPage.tsx` is allowed to ask the server for.

Item #1213. Alan's ruling, 2026-09-17: the Done column shows 7 days by default
and a checkbox brings back the rest, with the cutting done by the route so the
payload shrinks with it.

Why this file greps a `.tsx` is restated today because the reason it was
written is no longer the reason it is kept. It used to read "`web/` has no test
runner": no `test` script, neither vitest nor jest, no `*.test.*` files. All
four of those claims have been false since `94850c3e` ("web: vitest, and the
gate's frontend rung runs it") — `web/package.json` declares `"test": "vitest
run"`, `web/vitest.config.ts` pins `environment: "node"`, the frontend rung runs
`_vitest_run` (`scripts/automod/gate.py:347`), and ten `*.test.*` files already
shipped under `web/src` at this item's base — `git ls-tree -r --name-only
47580373 web/src | grep -cE '\.test\.(ts|tsx)$'` is 10 — the eleventh being the
node this change adds.

What is still true, and is the whole reason the source-claim pattern — the one
`tests/test_code_graph_doc_claims.py` and `tests/test_automod_doc_claims.py` use
— stays: `web/package.json` pulls in no jsdom and no @testing-library, and that
file is a denied loop path (`scripts/automod/gate.py:1517`), so nothing a round
can land renders a component. Which half of a frontend change is testable is
therefore decided by what each runner can execute, and this file has already
given work to the runner that landed: the pure logic of #2068 — which board the
page stands on — is a vitest node at
`web/src/components/pages/backlogBoardDefault.test.ts`, and what is left here is
what cannot be a pure function: claims about the source that becomes a behaviour
only once the component runs. It can pin that a request is built a certain way,
that a control exists and is bound to the state that changes that request, and
that a value is computed per request rather than hoisted — properties of the
text, and the text is the implementation. It cannot pin a render, and does not
claim to: a mount that counts requests, or a click that moves focus, belongs in
`web/` beside the other test files and is owed there. If `web/` gains a DOM
environment, the interaction assertions below get deleted in favour of ones that
run the page.

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
    below, which since #2068 runs only while the tab is visible, and becoming
    visible is itself a fetch — so a value taken once at mount is re-sent
    forever. Gating the poll changes how often the request is built, not what
    may go into it: the tab that survives to midnight is exactly the one that
    stays open and comes back. On a page left open across midnight that
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


# ── #2068 clauses 2-4: which board, how many requests, which tabs ──────────
#
# Three defects that had never been asserted, so all three survived every prior
# edit to this file. Board facts are the ones `_vitest_run` cannot see and this
# file can cite as measured on the box they were filed from (2026-10-02): the
# route answers (1, "alfie", 7 tasks), (2, "lloyd", 1,997), (3, "personal", 2);
# the cascade's first request — `board_id` absent — was 2,006 rows and
# 1,601,469 bytes; one windowed tasks request is 488,497 bytes.


def _span(start: str, end: str) -> str:
    """The source from `start` through the first `end` after it.

    Both markers are spelled out at each call site rather than derived from
    indentation, because `_callback` above terminates on `\n  }, [` and only
    matches `async () =>` bodies: `resolveBoard` carries a return-type
    annotation and `selectBoard` is not async, so neither is in its grammar, and
    an assertion about a slice that quietly came back empty is worse than no
    assertion.
    """
    assert start in SOURCE, f"{start!r} is not in BacklogPage.tsx — the node this claim is about moved"
    i = SOURCE.index(start)
    assert end in SOURCE[i:], f"{end!r} does not close {start!r}"
    j = SOURCE.index(end, i) + len(end)
    return SOURCE[i:j]


def test_the_default_board_is_resolved_by_name_at_both_defaulting_sites():
    """Clause 2. `boardsData[0].id` reads a positional id: `app/routers/backlog.py`
    numbers boards by enumerating the sorted board *names*, so element 0 is
    whichever board sorts first — here `alfie`, whose 7 items are all
    `status: done` and therefore all outside #1213's seven-day window, so the
    mount rendered an *empty* board — while the 1,997-task `lloyd` board sat at
    index 1.

    The default is `pickDefaultBoard(boardsData, DEFAULT_BOARD_NAME)` now. The
    assertion is on the count of sites: exactly two places in the file fall back
    to a default board — the mount, and the vanished-board re-pick in
    `resolveBoard` — and both of them are the helper.
    """
    assert "setActiveBoard(boardsData[0].id)" not in SOURCE, (
        "the positional default is back: element 0 is the board that sorts "
        "first, not the board the person opening the page means"
    )
    assert SOURCE.count("pickDefaultBoard(boardsData, DEFAULT_BOARD_NAME)") == 2, (
        "both defaulting sites must resolve by name; one has reverted to an "
        "index, which is the defect this clause was filed for"
    )
    assert 'from "./backlogBoardDefault"' in SOURCE, (
        "the preferred name has to come from the module the vitest node pins, "
        "or the page and its test can disagree about the spelling of the board"
    )
    helper = _text(ROOT / "web" / "src" / "components" / "pages" / "backlogBoardDefault.ts")
    assert 'export const DEFAULT_BOARD_NAME = "lloyd"' in helper, (
        "the name the route keys boards by is `lloyd`; anything else matches "
        "nothing and the page falls back to the positional default it just left"
    )


def test_the_tasks_request_is_built_from_a_board_that_is_already_resolved():
    """Clause 3. `loadData` used to read `activeBoard` *and* write it, inside one
    callback that listed `activeBoard` in its own dependencies. A mount therefore
    took the false branch of `activeBoard ? … : {}` and asked for tasks with no
    `board_id` — every board, 1,601,469 bytes — and then its own `setActiveBoard`
    changed the dependency, re-made the callback, re-ran the effect and asked a
    second, this time filtered, time. Both requests fetched the whole board set.

    What a source claim can settle here is the *order*: the board is awaited
    before the task request is assembled, the filter is spread from that awaited
    value, and the two fetches are no longer issued together. The request count
    itself is a render fact, owed to `web/` with the rest of them.
    """
    load = _callback_body("loadData")
    assert "await resolveBoard()" in load, (
        "the board is no longer resolved on the request path, so nothing "
        "guarantees an id exists before the tasks query is built"
    )
    assert load.index("await resolveBoard()") < load.index("api.backlogTasks("), (
        "the tasks request is assembled before the board is known again — that "
        "is the 1,601,469-byte unfiltered fetch"
    )
    assert "...(boardId ? { board_id: String(boardId) } : {})" in _window_request_call(), (
        "the filter must be spread from the resolved id; spreading `activeBoard` "
        "reads state this same pass has not applied yet"
    )
    assert not re.search(r"Promise\.all\(\s*\[", load), (
        "boards and tasks are back in one `Promise.all([...])` call, which is "
        "precisely how the tasks query ended up with nothing to filter on — "
        "matched on the call syntax, because this callback's own comment names "
        "the old shape in prose"
    )
    deps = _callback("loadData")[1]
    assert "activeBoard" not in deps, (
        f"`activeBoard` is back in loadData's dependency list ({deps!r}); with a "
        "board write inside that callback the mount cascades again"
    )


def test_the_callback_that_fetches_tasks_writes_no_board():
    """Clause 3's shape rather than its order: the choice of board moved into
    `resolveBoard`, and the callback that issues `api.backlogTasks` writes no
    board at all. Either half can be undone from one side by an edit that looks
    harmless — a fallback re-added to `loadData`, the resolution pulled back
    into the fetch — so both are pinned, with the writes that remain named."""
    assert "setActiveBoard" not in _callback_body("loadData"), (
        "`loadData` chooses a board again: that write is what changed a "
        "dependency and fired the mount's second request"
    )
    resolve = _span("const resolveBoard = useCallback(", "  }, []);")
    assert "setActiveBoard(resolved)" in resolve, (
        "the default has to reach state from somewhere, and it has to be the "
        "place that resolved it — otherwise the tab highlights one board and "
        "the columns show another"
    )
    select = _span("const selectBoard = useCallback(", "    [loadData],\n  );")
    assert "setActiveBoard(id)" in select and "void loadData()" in select, (
        "selecting a board is now also a request to load it, because nothing "
        "reacts to the board id any more"
    )


def test_every_path_that_changes_the_board_also_loads():
    """The consequence of the restructure, and the way it breaks if half-done:
    no dependency reacts to `activeBoard` now, so a board write that is not
    followed by a load moves the tab and leaves the previous board's rows in the
    columns. That includes the agent's own navigation — `mc_navigate(tab=backlog,
    focus_id=<board id>)`, whose two branches used to write the board and stop.
    """
    assert "onClick={() => setActiveBoard(board.id)}" not in SOURCE, (
        "a board tab that only writes state shows the tab of one board over the "
        "rows of the one that was there"
    )
    assert "onClick={() => selectBoard(board.id)}" in SOURCE
    focus = _span("  useEffect(() => {\n    if (!pendingFocus) return;",
                  "  }, [pendingFocus, tasks, selectBoard]);")
    assert focus.count("selectBoard(asNum)") == 2, (
        "both agent-focus board branches — the known-board hit and the "
        "detail-fetch failure — have to go through the path that loads"
    )
    assert "setActiveBoard(asNum)" not in focus


def test_the_poll_runs_only_while_the_tab_is_visible():
    """Clause 4. `setInterval(loadData, 15_000)` with nothing around it polled a
    hidden tab, and one tick is a boards request plus a tasks request — 488,497
    bytes for the windowed tasks call alone — every fifteen seconds, per open
    tab, for a page nobody is looking at.

    The shape is `DashboardPage.tsx:1263-1280`: a `visibilitychange` listener
    that clears the clock and starts it again, and reads immediately on becoming
    visible because the tab that just appeared is the one whose data is stale.
    Within that block the handler is `:1268-1274`, its `clearInterval` `:1269`,
    the visibility test `:1270`, and the listener's registration
    `:1275`/teardown `:1278`.

    Two things here are stricter than that precedent, and both are asserted
    below because each closes a hole the precedent leaves open. The clock is not
    started at all when the first paint is hidden: a bare listener never fires
    for a tab that stays hidden, so on the page as previously shaped a
    link-clicked background tab polled forever from its mount. And the handler
    clears before it tests visibility, so a hide does not leave the old timer
    running — see `test_the_visibility_handler_clears_before_it_restarts`.
    """
    effect = _span("  useEffect(() => {\n    loadData();", "  }, [loadData]);")
    assert "setInterval(loadData, 15_000)" in effect, (
        "the 15-second interval is the thing this gate exists to stop, so it "
        "still has to be the thing being stopped"
    )
    assert "document.visibilityState" in effect, (
        "the poll is back to running in a hidden tab"
    )
    decide = effect.index("if (document.visibilityState ===")
    assert effect.index("startPolling()", decide) < effect.index(
        'document.addEventListener("visibilitychange"'
    ), (
        "the clock has to be decided before the listener is installed: a tab "
        "that opens hidden and stays hidden must never poll, and a listener "
        "installed first leaves the first `startPolling()` unconditional"
    )
    assert "void loadData()" in effect, (
        "becoming visible has to read at once — `DashboardPage`'s shape — or the "
        "first thing a returning reader sees is up to 15 seconds old"
    )
    assert 'document.removeEventListener("visibilitychange", onVisibility)' in effect, (
        "a listener left attached across an unmount polls a page that no longer "
        "exists, and the next mount adds a second clock beside it"
    )


#: The helper as its caller spells it, so a test that only read the module
#: could not pass while the page imported nothing.
HELPER = ROOT / "web" / "src" / "components" / "pages" / "backlogBoardDefault.ts"

#: The board list as the route answered it on 2026-10-02 — the fixture the
#: vitest node uses too. `app/routers/backlog.py` numbers boards by enumerating
#: sorted board names, which is why `lloyd` is id 2 and not id 1.
LIVE_BOARDS = [{"id": 1, "name": "alfie", "task_count": 7},
               {"id": 2, "name": "lloyd", "task_count": 1997},
               {"id": 3, "name": "personal", "task_count": 2}]

#: #2068 clause 6: the access-log extract, committed on the vault's main.
WITNESS_VAULT_PATH = "backlog/data/2026-10-02.2068-server-log-witness.md"

#: The commit this item was triaged against, and the tree every "at base" count
#: in this file is taken from. Named so a docstring that cites a number can be
#: checked against it.
BASE_SHA = "47580373"


def test_the_default_board_lookup_is_pinned_from_the_python_side_too():
    """Clause 1's behaviour, observed from this suite as well as from vitest.

    `web/src/components/pages/backlogBoardDefault.test.ts` is the runner-native
    home for this and the frontend rung does run it, but it is a file the gate's
    own `test_node_id` check cannot read: the rail's grammar is Python
    `def test_…` and pytest's collector does not see vitest files, so "the clause
    is pinned by a node a run of `tests/` can select" is not satisfiable there.
    Hence the executable proof lives here too — and it is the real module, not a
    restated copy, because node strips the helper's types itself
    (`--experimental-strip-types`), so no build step is involved.

    It does not skip. An earlier version took a conditional skip when node was
    absent or too old to strip types, the review rung read that as an advisory
    against clause 1's only executable proof, and it was right: a skipped
    clause-proof is a green that says nothing, and node is a build requirement of
    `web/` (`web/package.json`'s `test` script is `vitest run`), so its absence is
    a red this repo deserves. The failure mode it was written against (#1954, four
    nodes asserting over an empty denominator) is served the other way by the
    same rule.
    """
    import json
    import shutil
    import subprocess

    # No skip anywhere in this node, which is what answers the advisory on the
    # previous round (`advisory tests/test_backlog_done_window_frontend_claim.py:
    # 509: a new pytest.skip (conditional; the grader judges the condition)`): a
    # conditional skip in a clause's only executable proof is a green that can
    # mean "never ran", and the condition here is node, which this repository
    # cannot build without — `web/package.json`'s `test` script *is* `vitest run`,
    # run through node by the gate's frontend rung. So node being absent or too
    # old to strip types is a red this repo deserves, not a pass.
    node = shutil.which("node")
    assert node is not None, (
        "node is absent from PATH, and web/ is built and tested through node "
        "(`web/package.json`: \"test\": \"vitest run\"), so the helper cannot be "
        "executed and the frontend cannot be built either"
    )
    # `--experimental-strip-types` for node 22; node 23+ strips by default and
    # rejects an unknown flag, so the fallback run is the modern path, not a
    # weaker one — both runs execute the same `pickDefaultBoard` from the same
    # module URL.
    script = (
        "const m = await import(%r);"
        "const LIVE = %s;"
        "console.log(JSON.stringify({"
        "by_name: m.pickDefaultBoard(LIVE, 'lloyd'),"
        "fallback: m.pickDefaultBoard([{id: 4, name: 'zzz'}], 'lloyd'),"
        "empty: m.pickDefaultBoard([], 'lloyd'),"
        "preferred: m.DEFAULT_BOARD_NAME}));"
    ) % (HELPER.as_uri(), json.dumps(LIVE_BOARDS))
    attempt = subprocess.run([node, "--experimental-strip-types", "--input-type=module",
                              "-e", script],
                             capture_output=True, text=True, timeout=60)
    if attempt.returncode != 0 and "bad option" in attempt.stderr:
        attempt = subprocess.run([node, "--input-type=module", "-e", script],
                                 capture_output=True, text=True, timeout=60)
    version = subprocess.run([node, "--version"], capture_output=True,
                             text=True).stdout.strip()
    assert attempt.returncode == 0, (
        f"node {version} could not run the helper: {attempt.stderr[:400]}"
    )
    out = json.loads(attempt.stdout.strip().rsplit("\n", 1)[-1])

    assert out["preferred"] == "lloyd", (
        "the page and its test can disagree about the board's spelling only if "
        "the constant itself moved, which is this assertion's whole job"
    )
    assert out["by_name"] == {"id": 2, "name": "lloyd", "task_count": 1997}, (
        "the lookup answered the wrong board for the live list; an index answer "
        "would be id 1 — the board whose 7 items are all done and outside the "
        "window, which is the empty page this item was filed for"
    )
    assert out["fallback"] == {"id": 4, "name": "zzz"}, (
        "with no name match the caller is documented to get element 0, which is "
        "the right answer on a box that has no `lloyd` board at all"
    )
    assert out["empty"] is None, (
        "an empty list must yield null, not the `undefined` that element 0 of an "
        "empty array is: the page has to tell 'there is no board' from 'I have "
        "not picked one'"
    )


def test_the_mount_cascade_witness_re_derives_to_the_counts_it_quotes():
    """Clause 6: the traffic figures this item quotes had no history, because
    `~/lloyd-data` is not a repo and `server.log` rotates — 50,653 lines and
    4,260,171 bytes when the extract was taken, so a whole-file copy would have
    been mostly unrelated traffic and would have rolled away from the claim. What
    is committed on the vault's main is every line of it that names
    `/api/backlog`, verbatim, each prefixed in the witness with its source line
    number; this node recomputes the quoted counts out of those bytes instead of
    trusting the prose beside them.

    The claim under test: 9 lines, 7 of them `/api/backlog/tasks`, 6 of those 7
    carrying no `board_id`, and the cascade visible as three consecutive source
    lines — `#:31856` with `board_id=1`, `#:31857` the same page asking again
    with no board, `#:31858` the boards fetch. Response *sizes* are deliberately
    not re-derived: uvicorn's access line records method, path and status and
    nothing else, so 1,601,469 B and 488,497 B stay attributed to the route-level
    measurement they came from, exactly as the witness file itself says.
    """
    import subprocess

    # Asserted, not skipped, for the same reason as the node run above: this
    # node is clause 6's whole re-derivation, and a skip here would let the
    # witness vanish while the suite still reported the clause covered. The vault
    # is where every knowledge note and every other witness in this repo lives,
    # so its absence is a broken machine, and an uncommitted witness is exactly
    # the drift clause 6 exists to prevent.
    vault = Path.home() / "obsidian"
    assert (vault / ".git").exists(), (
        f"no vault at {vault} — the committed witness lives at "
        f"{WITNESS_VAULT_PATH} there, so there is nothing to re-derive from"
    )
    committed = subprocess.run(
        ["git", "-C", str(vault), "show", f"HEAD:{WITNESS_VAULT_PATH}"],
        capture_output=True, text=True)
    assert committed.returncode == 0, (
        f"{WITNESS_VAULT_PATH} is not committed on the vault's main, so the "
        f"bytes clause 6 quotes have no history: {committed.stderr[:160]}"
    )
    text = committed.stdout
    assert (vault / WITNESS_VAULT_PATH).read_text(encoding="utf-8") == text, (
        "the witness was edited in the vault's working tree without landing it, "
        "so a reader holding only the repo has different bytes from the ones "
        "that were measured"
    )

    # Sliced on the `#:` markers themselves, not on "the first fenced block".
    # The witness file carries a second fenced block (the live re-check command,
    # one line), and a fence-pairing slip there is a count of 1 reported as
    # though the traffic itself had changed: `assert 1 == 9` was the exact
    # signature a review of this round saw, and it named a line of this file
    # that is not this assert and a file length it does not have. Reading only
    # lines that start with `#:` makes the slice self-identifying, so the count
    # can only be the extract's.
    lines = [l for l in text.splitlines() if l.startswith("#:")]
    assert len(lines) == 9, (
        f"the extract holds {len(lines)} `#:`-marked lines, not the 9 quoted — "
        f"first: {lines[0][:90] if lines else 'none'}"
    )

    markers = [int(m.group(1)) for m in re.finditer(r"^#:(\d+)\s", "\n".join(lines), re.M)]
    assert len(markers) == len(lines), (
        "every extract line has to carry the line number it came from; adjacency "
        "is the finding and the extract alone cannot show it"
    )
    assert markers == sorted(markers), "the extract reorders the log"

    # Counted out of the extract lines, not the whole file: the prose beside the
    # extract quotes request paths too, and a count that reads them would rise
    # when someone wrote a sentence, not when the traffic changed.
    paths = re.findall(r'"GET (/api/backlog[^"]*)"', "\n".join(lines))
    assert len(paths) == len(lines), "an extract line has no request in it"
    tasks = [u for u in paths if u.startswith("/api/backlog/tasks")]
    assert len(tasks) == 7, f"{len(tasks)} tasks requests, not the 7 quoted"
    unfiltered = [u for u in tasks if "board_id=" not in u]
    assert len(unfiltered) == 6, (
        f"{len(unfiltered)} of the {len(tasks)} tasks requests carry no board "
        "filter, not the 6 quoted — the unfiltered request is the whole item"
    )
    assert sum(1 for u in tasks if "board_id=1" in u) == 1, (
        "the cascade is a filtered request beside an unfiltered one; without the "
        "filtered one there is nothing beside anything"
    )

    assert markers[3:6] == [31856, 31857, 31858], (
        f"lines {markers[3:6]} are not the consecutive trio the witness names — "
        "adjacency is what makes this pair a mount rather than two page views"
    )
    assert "board_id=1" in lines[3], lines[3]
    assert "board_id" not in lines[4], (
        f"the line right after the filtered request carries a filter too "
        f"({lines[4]!r}), so it is not the second, unfiltered ask"
    )
    assert "GET /api/backlog/boards" in lines[5], (
        "the boards fetch closing the trio is the evidence the tasks query was "
        "assembled before the board answer existed"
    )


def test_the_visibility_handler_clears_the_clock_before_it_restarts():
    """The half of a visibility gate that a check counting `visibilityState`
    cannot see: an `onVisibility` that calls `startPolling()` without clearing
    first leaves the timer from before the hide running, so hiding the tab does
    not stop the polling and every show/hide cycle stacks another clock on it.
    `DashboardPage.tsx:1269` clears before it tests visibility at `:1270`, and
    this page does the same in both places — inside the handler and in the
    unmount cleanup."""
    effect = _span("  useEffect(() => {\n    loadData();", "  }, [loadData]);")
    handler = _span("const onVisibility = () => {", "    };")
    assert "clearInterval(timer)" in handler and "startPolling()" in handler, (
        "the handler has to both stop the old clock and start the new one"
    )
    assert handler.index("clearInterval(timer)") < handler.index(
        "if (document.visibilityState"), (
        "the clear must run on the hide branch too, which means it precedes the "
        "visibility test rather than sitting inside the visible branch"
    )
    cleanup = effect[effect.index("    return () => {"):]
    assert "clearInterval(timer)" in cleanup, (
        "the timer has to die with the component, or the next mount stacks a "
        "second running clock on top of the one still ticking"
    )


def test_the_columns_are_memoised_on_the_board_rows():
    """The item's fourth claim: `filteredTasks` and `tasksByStatus` rebuilt their
    filter and their per-status buckets on every render, and this component
    re-renders on every keystroke in the search box, every drag and every open
    modal. Behaviour-free, so the gate's `tsc` and `vite build` are the real
    guard and no clause asks for it — this node is here so the claim cannot be
    dropped by a merge that resolves the import line."""
    assert "useMemo" in SOURCE.splitlines()[0], (
        "the import line is where a merge silently drops a hook"
    )
    # Each dependency array is asserted inside its own `useMemo(` call, not as a
    # substring of the file: two bare literals anywhere in a 1,000-line component
    # would satisfy a global `in`, and the memo that actually governs the columns
    # could be unhooked while the node stayed green.
    filtered = _span("const filteredTasks = useMemo(", "\n  );")
    buckets = _span("const tasksByStatus = useMemo(", "\n  );")
    assert "useMemo(" in filtered and "tasks.filter((t) => t.board_id === activeBoard)" in filtered, (
        "`filteredTasks` is rebuilt per render again, or its filter changed shape"
    )
    assert re.search(r"\[tasks,\s*activeBoard\],\s*\);\s*$", filtered), (
        f"`filteredTasks`' deps are not the payload and the board: {filtered[-60:]!r}"
    )
    assert re.search(r"\[filteredTasks\],\s*\);\s*$", buckets), (
        f"the per-status buckets memo does not depend on the rows it buckets: "
        f"{buckets[-60:]!r}"
    )


# ── the seam: the client's param bag really reaches the route ───────────────


#: The claims clause 5 requires this file's docstring to make, and the claims it
#: must never make again. Matched on fragments rather than whole sentences so a
#: rewrap of the prose does not fail the node — but each fragment is one this
#: file's text contains verbatim, so the node is checking this docstring, not any
#: docstring that happens to be friendly.
_DOC_MUST_SAY = (
    "94850c3e",                     # the commit that landed the runner
    "vitest run",                   # what its `test` script is
    'environment: "node"',          # what the runner can and cannot execute
    "backlogBoardDefault.test.ts",  # where the pure logic's node lives now
    "render",                       # what is owed, and where it is owed
)
#: A false claim quoted to be retracted is fine; the same words without one of
#: these are the claim being made again.
_RETRACTION_CUES = ("used to", "have been false", "false since", "no longer")

_DOC_MUST_NOT_SAY = (
    "has no test runner",
    "no `test` script",
    "neither vitest nor jest",
    "is empty",                     # `find web/src -name '*.test.*'` is empty
)


def test_this_files_own_rationale_is_the_true_one():
    """Clause 5's assertion, in the file clause 5 is about.

    Until now clause 5 asked for prose to be rewritten and pinned nothing, so the
    one edit that matters — a merge resolving this docstring back to the version
    that justified grepping a `.tsx` by claiming `web/` has no runner — would keep
    this whole file green while making every claim in it a lie about the tree. The
    repo already has this shape elsewhere (`tests/test_arch_review_source.py:1584`
    asserts `'X' not in mod.__doc__`); it is the cheapest node in the suite and
    the only one that defends the reason all the others are written the way they
    are.

    Both directions are checked against the tree as well as against the docstring,
    so the node cannot be satisfied by a docstring that merely asserts a claim the
    repo no longer supports: the runner is re-found from `web/package.json` and
    `web/vitest.config.ts` here, not quoted from the prose.
    """
    import subprocess

    doc = __doc__ or ""
    assert doc, "this module lost its docstring, which is what clause 5 is about"
    # Wrapped prose must not fail a fragment match: compare on collapsed
    # whitespace, so a rewrap of the rationale is not a clause-5 regression.
    doc = " ".join(doc.split())
    for want in _DOC_MUST_SAY:
        assert want in doc, f"clause 5's replacement rationale has lost {want!r}"
    # The four false claims may survive only as a retraction — the docstring
    # quotes them to say when they stopped being true — so the test is not
    # "absent" but "never asserted": every sentence that contains one must also
    # carry a retraction cue. That is the difference between the sentence clause 5
    # asked for ("It used to read … All four of those claims have been false since
    # 94850c3e") and the merge that quietly deletes the second half of it and
    # leaves the lie standing on its own.
    sentences = re.split(r"(?<=[.?])\s+", doc)
    for ban in _DOC_MUST_NOT_SAY:
        for s in sentences:
            if ban not in s:
                continue
            assert any(cue in s for cue in _RETRACTION_CUES), (
                f"{ban!r} appears in this file's rationale without a retraction "
                f"cue, i.e. asserted: {s[:160]!r}"
            )

    # And the docstring's claims are re-checked against the tree it describes.
    pkg = _text(ROOT / "web" / "package.json")
    assert '"test": "vitest run"' in pkg, (
        "the docstring says the runner exists; if `web/package.json` no longer "
        "agrees, the docstring is the thing that is wrong"
    )
    assert "jsdom" not in pkg and "@testing-library" not in pkg, (
        "the docstring says no render is possible here; a DOM environment "
        "landing in web/ means the interaction claims below should be deleted "
        "in favour of ones that run the page — which is what the docstring "
        "promises, so update both in one change"
    )
    cfg = _text(ROOT / "web" / "vitest.config.ts")
    assert 'environment: "node"' in cfg, "the docstring's scope claim is stale"
    tracked = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", BASE_SHA, "web/src"],
        capture_output=True, text=True, cwd=str(ROOT))
    n_base = len([l for l in tracked.stdout.splitlines()
                  if re.search(r"\.test\.(ts|tsx)$", l)])
    assert n_base == 10, (
        f"{n_base} *.test.* files under web/src at {BASE_SHA}, not the 10 the "
        "docstring cites — the sentence is a measurement, so re-measure it and "
        "change the prose rather than the tree"
    )



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
