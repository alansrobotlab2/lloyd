"""The Runs panel's bench control, pinned from Python — #2648 clause 4.

Clause 4 is about WIRING: a button that reveals the hidden bench rows, hides them again
without a page reload, and cannot change which 150 rows the endpoint returns. The frontend
half of this contract is `web/src/components/pages/BackgroundPage.test.ts`, which EXECUTES
the pieces it can (the reducer, the loader against a stubbed `fetch`, the hook-free
`RunsToolbar`'s own `onClick`). What that file structurally cannot do is reach the JSX
inside `RunsPanel`: `web/vitest.config.ts` pins `environment: "node"`, no DOM package is
installed, and a `.tsx` mount dies at `api.ts`'s module-scope `import.meta.env`.

So this is the `.py` mirror over the same bytes, in the shape
`tests/test_dashboard_label_contract.py` records as the in-repo remedy for #2202: that
file's own header says "a `.test.ts` is not a testpath, so the round's review rung would
read a clause pinned only there as untested", and "being a translation of a test that
already exists is worth nothing on its own" — so this mirror does two things the vitest
file cannot.

It reads FORMATTING-BLIND. The frontend file's wiring assertions are regexes over
`?raw` source, and one earlier round of this work died because a regex straddling a
line-wrap passed on a button that was present in the text and never bound to anything. The
prop table here is extracted by a brace-aware scanner over the element's own opening tag,
so a prop that exists but is not passed, or is passed to an element nothing renders, is a
failure rather than a match.

And it is checked against its own mutations.
`test_the_wiring_check_bites_on_each_of_five_mutations` breaks the chain five ways —
unbind the button, empty the effect's dependency array, zero the count, widen the window on
reveal, derive the count from the rows — and requires each one to redden this file.
A wiring mirror that cannot be broken is decoration.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "web/src/components/pages/BackgroundPage.tsx"
TOOLBAR = ROOT / "web/src/components/pages/RunsToolbar.tsx"
STATE = ROOT / "web/src/components/pages/runsPanelState.ts"
CLIENT = ROOT / "web/src/api.ts"
VITEST = ROOT / "web/src/components/pages/BackgroundPage.test.ts"
BENCH_ROWS = ROOT / "web/src/components/pages/benchRows.ts"

WINDOW = 150


class Broken(Exception):
    """One link of the chain is missing. Carries what was looked for."""


def _need(cond: bool, what: str) -> None:
    if not cond:
        raise Broken(what)


def _code(src: str) -> str:
    """Source with its comments removed.

    The navigation scan has to be over code and not prose: the effect that refetches on a
    toggle carries the sentence "no `location.reload()`" in its own comment, and a scan that
    reads the explanation would report the panel navigating. No `//` in these three files
    opens a string (no URL is built here), so a line-comment strip is safe.
    """
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def _region(src: str, start: str, end: str, label: str) -> str:
    i = src.find(start)
    _need(i >= 0, f"{label}: no {start!r}")
    j = src.find(end, i + len(start))
    return src[i: len(src) if j < 0 else j]


def _jsx_tag(src: str, tag: str, label: str) -> str:
    """The opening tag of `<tag ... />`, brace-aware: a `>` inside `{...}` is not the end.

    This is the whole reason to write the mirror in Python instead of trusting the regexes:
    the check is over the element's prop list, so a prop straddling a line-wrap is read
    exactly as one on a single line, and a prop that is simply absent cannot match.
    """
    i = src.find(f"<{tag}")
    _need(i >= 0, f"{label}: no <{tag} element")
    depth = 0
    k = i + len(tag) + 1
    while k < len(src):
        c = src[k]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == "/" and depth == 0 and src.startswith("/>", k):
            return src[i:k + 2]
        elif c == ">" and depth == 0:
            # A closing `>` for a paired tag — not self-closing, so the control is not a
            # <button>…</button> written this way.
            return src[i:k + 1]
        k += 1
    raise Broken(f"{label}: <{tag} opening tag never closed")


def _prop(tag: str, name: str) -> str | None:
    """The expression of `name={...}` in an opening tag, whitespace-normalised, or None."""
    i = tag.find(f"{name}={{")
    if i < 0:
        return None
    k = i + len(name) + 2
    depth = 1
    while k < len(tag) and depth:
        if tag[k] == "{":
            depth += 1
        elif tag[k] == "}":
            depth -= 1
            if depth == 0:
                break
        k += 1
    return re.sub(r"\s+", " ", tag[i + len(name) + 2:k]).strip()


def _expr(src: str, name: str, label: str) -> str:
    m = re.search(rf"\b{name}\s*=\s*([^;\n]+)", src)
    _need(bool(m), f"{label}: no `{name} =` binding")
    return m.group(1).strip()


def check_chain(panel: str, toolbar: str, state: str, client: str) -> dict:
    """Assert every link from the button to the query string. Raises Broken on the first gap.

    Reads only, returns the facts it verified, so the mutations below can be reported as
    which link each one breaks.
    """
    facts: dict[str, str] = {}

    # --- the panel: state, the refetch trigger, and what it hands the control -------------
    body = _region(panel, "function RunsPanel(", "\nfunction ", "RunsPanel")
    _need("useReducer(panelReduce, initialPanelState)" in body,
          "RunsPanel does not mount runsPanelState's reducer")
    _need("void loadRuns(state.showBench, dispatch)" in body,
          "RunsPanel never calls loadRuns with the class it is showing")
    _need("setInterval(() => { void loadRuns(state.showBench, dispatch) }, POLL_MS)" in body,
          "the 10 s poll does not carry the class either, so a poll would undo a reveal")

    dep_line = re.search(r"useEffect\(\(\) => \{.*?\}, \[([^\]]*)\]\)", body, re.S)
    _need(bool(dep_line), "RunsPanel has no useEffect with a dependency array")
    _need(dep_line.group(1).strip() == "state.showBench",
          f"the refetch trigger depends on {dep_line.group(1).strip()!r}, not on "
          "`state.showBench`, so pressing the control changes nothing on the wire")
    facts["deps"] = dep_line.group(1).strip()

    tag = _jsx_tag(body, "RunsToolbar", "RunsPanel")
    handed = {
        "showBench": _prop(tag, "showBench"),
        "hiddenBench": _prop(tag, "hiddenBench"),
        "shown": _prop(tag, "shown"),
        "onToggle": _prop(tag, "onToggle"),
    }
    _need(handed["showBench"] == "state.showBench",
          "the control is not told which class is showing")
    # The count the reader sees must be the server's, held in state — not a number written
    # at the call site, and not a count of the rows that arrived (clause 3's UI half).
    _need(handed["hiddenBench"] == "state.hiddenBench",
          "the hidden count at the call site is not `state.hiddenBench`, so the number on "
          "screen is no longer the one the endpoint reported")
    _need(handed["shown"] == "state.rows.length",
          "the row-count line is not reading the rows that are on screen")
    _need(handed["onToggle"] == "makeToggleDispatcher(dispatch)",
          "the control's toggle is not bound to the dispatcher for THIS state, so it would "
          "close over a stale one or none")
    facts["handed"] = repr(sorted(handed))

    # No navigation anywhere in the class this control owns. `BackgroundPage.tsx` as a whole
    # does hold a `window.location.reload()` — the Sources tab's manual refresh — which is
    # why the scan is the panel's body and the two modules, not the file.
    for src, label in ((body, "RunsPanel"), (toolbar, "RunsToolbar"), (state, "runsPanelState")):
        _need(not re.search(r"\blocation\b|reload\(|history\.push", _code(src)),
              f"{label} navigates or reloads: the reveal must be a refetch of the route the "
              f"panel already polls")

    # --- the control: the prop reaches an actual clickable element ------------------------
    button = _jsx_tag(toolbar, "button", "RunsToolbar")
    _need(_prop(button, "onClick") == "onToggle",
          "the <button> in RunsToolbar does not bind onClick to the onToggle prop — the "
          "exact failure mode a source-text regex cannot see")
    _need(re.match(r'<button\s+type="button"', button) is not None,
          "the control RunsToolbar renders is not `<button type=\"button\">` — a bare "
          "`<button>` defaults to type=submit, so inside a form it would submit one")
    _need("benchToggleLabel(showBench, hiddenBench)" in toolbar,
          "the button's label is not the shared copy, so pytest and the vitest file would "
          "drift on the wording independently")
    _need("hiddenBenchSuffix(hiddenBench)" in toolbar,
          "the row-count line no longer says what was withheld")

    # --- the state module: one window, one class, the server's count ----------------------
    _need(_expr(state, "RUNS_WINDOW", "runsPanelState") == str(WINDOW),
          f"runsPanelState's window is not {WINDOW}, the width the Runs tab asks for")
    sig = re.search(r"export async function loadRuns\(([^)]*)\)", state)
    _need(bool(sig), "runsPanelState has no exported loadRuns")
    _need("showBench" in sig.group(1) and not re.search(r"\bnumber\b", sig.group(1)),
          f"loadRuns({sig.group(1)}) takes a window from its caller, so revealing rows "
          f"could resize the window")
    _need("api.listBackgroundSessions(RUNS_WINDOW, showBench)" in state,
          "loadRuns does not ask for the fixed window and the current class together")
    _need("return { type: 'toggle' }" in state,
          "toggleEvent does not build a payload-free toggle event, so the click could carry "
          "a window or a row set with it")
    toggle_arm = _region(state, "case 'toggle':", "\nfunction makeToggleDispatcher",
                         "panelReduce's toggle arm")
    returned = re.search(r"return\s*\{([^}]*)\}", toggle_arm)
    _need(bool(returned), "panelReduce's toggle arm returns no object literal")
    keys = [k for k in re.findall(r"(\w+):", returned.group(1))]
    _need(returned.group(1).replace(" ", "").startswith("...state,")
          and keys == ["showBench"] and "showBench: !state.showBench" in returned.group(1),
          f"the toggle arm returns {{{returned.group(1).strip()}}}: the flip must be the "
          f"ONLY change, spread over the previous state, so pressing the control cannot "
          f"blank the rows on screen or clear the count while the new answer is in flight")
    _need(not re.search(r"BENCH_SOURCES|\.source", toggle_arm),
          "the toggle arm is deciding which rows are bench, which is the server's job")
    loaded = _region(state, "case 'loaded':", "case ", "panelReduce's loaded arm")
    _need("hiddenBench: event.out.bench_hidden ?? 0" in loaded,
          "the hidden count is not taken from the response's bench_hidden")
    _need(not re.search(r"hiddenBench:.*\.length", loaded),
          "the hidden count is derived from the rows — the thing clause 3 exists to forbid")
    facts["sig"] = sig.group(1).strip()

    # --- the client: the class is the ONLY thing the request differs by -------------------
    # Not a clamp: `api.ts` sends `limit` verbatim (`limit: String(limit)`), so the property
    # that keeps the two classes on one window lives entirely upstream — the panel passes the
    # constant `RUNS_WINDOW` and `loadRuns` has no window to vary. What the client must never
    # do is touch the window while answering the class, which is what the two assertions on
    # `fn` below pin.
    fn = _region(client, "listBackgroundSessions", "// Display metadata", "api.ts")
    _need("includeBench = false" in fn,
          "the client's class parameter no longer defaults to hidden, so a caller that asks "
          "for nothing gets bench rows")
    _need("new URLSearchParams({ limit: String(limit) })" in fn,
          "the client no longer builds the window out of its own `limit` argument")
    _need("if (includeBench) params.set(\"include_bench\", \"true\")" in fn,
          "the class is no longer the only addition to the query string, so revealing rows "
          "could change which 150 the endpoint sees")
    return facts


def test_the_runs_panel_wires_its_bench_control_end_to_end():
    """Button → dispatcher → reducer → loader → query string, every link read from the source.

    The clause's three claims, each as a link rather than a vibe: the control is bound to
    something that moves the panel (`onClick={onToggle}` on a real `<button>`, and
    `onToggle={makeToggleDispatcher(dispatch)}` at the call site); revealing is a refetch
    and not a navigation (the effect's only dependency is `state.showBench`, and no
    `location`/`reload` appears in the panel body, the toolbar or the state module); and
    the hidden state cannot change which 150 rows come back (`loadRuns` has no window
    parameter, passes `RUNS_WINDOW`, which is 150, and the client clamps at 150) — while
    the count on screen is the endpoint's `bench_hidden`, never a count of the rows.
    """
    check_chain(
        PAGE.read_text(encoding="utf-8"),
        TOOLBAR.read_text(encoding="utf-8"),
        STATE.read_text(encoding="utf-8"),
        CLIENT.read_text(encoding="utf-8"),
    )


def test_the_two_halves_of_the_contract_agree():
    """Pytest and the frontend's own file must pin the same control, or the pair has split.

    The prop table is PARSED out of `BackgroundPage.test.ts` rather than typed a second
    time here, so a prop added on one side and not the other reddens this node instead of
    quietly narrowing one of the two checks.
    """
    ts = VITEST.read_text(encoding="utf-8")
    panel = PAGE.read_text(encoding="utf-8")
    toolbar = TOOLBAR.read_text(encoding="utf-8")

    called = re.search(r"RunsToolbar\(\{([^}]*)\}\)", ts, re.S)
    _need(bool(called), "the vitest file no longer calls RunsToolbar with a props object")
    test_props = set(re.findall(r"\b(showBench|hiddenBench|shown|onToggle)\s*:", called.group(1)))
    panel_props = set(n for n in ("showBench", "hiddenBench", "shown", "onToggle")
                      if _prop(_jsx_tag(_region(panel, "function RunsPanel(", "\nfunction ",
                                                "RunsPanel"), "RunsToolbar", "RunsPanel"), n)
                      is not None)
    assert test_props == panel_props == {"showBench", "hiddenBench", "shown", "onToggle"}, (
        f"the two halves disagree on the control's props: vitest passes {sorted(test_props)}, "
        f"the panel passes {sorted(panel_props)}")

    # The toolbar accepts exactly what both halves pass it.
    accepted = set(re.findall(r"^\s{2}(showBench|hiddenBench|shown|onToggle)\??:",
                              toolbar, re.M))
    assert accepted == test_props, (
        f"`RunsToolbarProps` declares {sorted(accepted)} but the two test halves exercise "
        f"{sorted(test_props)}")

    # The frontend file still asserts BOTH request forms and the same window; if it stopped,
    # the request half of this contract would live only here. The default form is an
    # ABSENCE — the client omits the parameter rather than sending `include_bench=false` —
    # so the third string is the file's own negative assertion, not a URL.
    for need in ("include_bench=true", "not.toContain('include_bench')",
                 "/api/background/sessions?limit=150", "RUNS_WINDOW"):
        assert need in ts, f"the vitest file no longer pins {need!r}"
    assert "makeToggleDispatcher(dispatch)" in ts, (
        "the vitest file no longer presses the control's real onClick, so the click half "
        "of clause 4 would be pinned by nothing executable")


def test_the_wiring_check_bites_on_each_of_five_mutations():
    """Each break in the chain must redden this file; an unbreakable mirror is decoration.

    Five, because those are the five ways this clause can rot without anyone editing the
    test: the button's handler unbound, the effect's dependency emptied, the count zeroed at
    the call site, the window widened on reveal, and the count derived from the rows.
    """
    panel = PAGE.read_text(encoding="utf-8")
    toolbar = TOOLBAR.read_text(encoding="utf-8")
    state = STATE.read_text(encoding="utf-8")
    client = CLIENT.read_text(encoding="utf-8")

    cases = {
        "onClick unbound from the button":
            (lambda t=toolbar: t.replace("onClick={onToggle}", 'disabled')),
        "the effect's dependency array emptied":
            (lambda p=panel: p.replace("}, [state.showBench])", "}, [])")),
        "the count zeroed at the call site":
            (lambda p=panel: p.replace("hiddenBench={state.hiddenBench}", "hiddenBench={0}")),
        "the window widened when bench is revealed":
            (lambda s=state: s.replace(
                "api.listBackgroundSessions(RUNS_WINDOW, showBench)",
                "api.listBackgroundSessions(showBench ? RUNS_WINDOW + 300 : RUNS_WINDOW, "
                "showBench)")),
        "the count derived from the rows instead of the response":
            (lambda s=state: s.replace(
                "hiddenBench: event.out.bench_hidden ?? 0",
                "hiddenBench: (event.out.sessions ?? []).length")),
    }
    # Each case names the file it mutates by the key it lives under, so a mutation that
    # silently stops applying (the string it replaces moves) cannot read as "caught".
    targets = {
        "onClick unbound from the button": ("toolbar", toolbar),
        "the effect's dependency array emptied": ("panel", panel),
        "the count zeroed at the call site": ("panel", panel),
        "the window widened when bench is revealed": ("state", state),
        "the count derived from the rows instead of the response": ("state", state),
    }
    for label, mutate in cases.items():
        which, original = targets[label]
        mutated = mutate()
        assert mutated != original, f"{label}: the mutation matched nothing, so it tested " \
                                    f"nothing"
        broken = {"panel": panel, "toolbar": toolbar, "state": state, "client": client}
        broken[which] = mutated
        try:
            check_chain(broken["panel"], broken["toolbar"], broken["state"], broken["client"])
        except Broken as exc:
            assert str(exc), f"{label}: the mirror caught it with no reason"
        else:
            raise AssertionError(
                f"the mirror did not notice {label!r}: check_chain passed on a tree with "
                f"that link broken, so the assertion for it is vacuous")


def test_the_control_speaks_the_copy_the_test_renders():
    """The label and the suffix the panel shows are the ones the frontend file asserts.

    The panel used to build its own strings; now `benchRows.ts` holds them and the copy is
    asserted in exactly two places, so a wording change that updates only one of them
    reddens something.
    """
    copy = BENCH_ROWS.read_text(encoding="utf-8")
    ts = VITEST.read_text(encoding="utf-8")
    for fn in ("benchToggleLabel", "hiddenBenchSuffix", "emptyStateCopy"):
        assert f"export function {fn}" in copy, f"benchRows.ts no longer exports {fn}"
    for said in ("show 148 bench runs", "hide bench", "148 bench withheld"):
        assert said in ts, (
            f"the frontend file no longer renders {said!r}; the copy in benchRows.ts is "
            f"then asserted by nothing but its own source")
    assert "show ${hiddenBench} bench run" in copy, (
        "the label no longer names the withheld count, so `show bench` would be the same "
        "control clause 3 says is not enough")
