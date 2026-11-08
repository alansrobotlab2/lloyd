"""#3918 clauses 1–4 from Python, over the same bytes the panel ships.

Two files pin one contract, and each does the half the other cannot:

  * `web/src/sidepanel/SidePanelView.test.tsx` is the RENDERING half. It calls
    `focusStateFromMessage` and paints `CheckButton` / `EmptyState` with what came back, so
    the panel's own suite reddens if the shipped mapping or the shipped copy changes. That
    is where a developer editing the panel finds out, and it is what the automod gate's
    `frontend` rung runs (`gate.py::_vitest_run`).
  * THIS file is the same contract as pytest nodes over the same bytes. The review rung's
    node rail accepts a node id only in a test file under one of `pytest.ini`'s testpaths —
    `scripts/automod/testpaths.py::is_test_file` requires a `.py` — so a clause pinned only
    in a `.test.tsx` reads as `test_node_id not in a test file this diff changed` however
    well it was verified. That is the shape recorded in
    `tests/test_dashboard_label_contract.py` for #2202 and in
    `tests/test_background_runs_panel_contract.py` for #2648, and it is why round
    `SM_20261005_002918` lost clauses whose substance its own review text called met.

Being a translation of a test that already exists is worth nothing on its own, so this
mirror checks the two things the vitest file structurally cannot. It checks the mapping for
PURITY — no `chrome.`, no `window`, no state, nothing outside the argument, which a render
assertion cannot see because a mapping with a side effect paints the same pixels. And it
checks the two files against EACH OTHER: the strings the spec asserts with `toContain` are
required to be strings the component still ships, which is the only way this round's
"unchanged copy" claim outlives the round — change one literal in the panel and the drift
node below reddens even if nobody touches a test.

Reads are formatting-blind where formatting could hide the thing: function bodies come from
a paren-then-brace matcher, not a line window, so a `disabled` expression that moved two
lines down is still found, and a mapping that came back out of `focusStateFromMessage` and
into the listener is found ABSENT even though every literal is still in the file. Prose
cannot satisfy any of it: whole-line comments are stripped before every scan, which is the
shape prose takes in both files — a comment trailing code would be scanned as code.

And it is checked against its own mutations. See
`test_the_mirror_bites_on_each_of_six_mutations`: six breaks, each requiring a refusal.
A mirror that cannot be broken is decoration.

Run it: `pytest tests/test_sidepanel_panel_view_contract.py`. No `node_modules`, no vite, no
browser — nothing here renders a pixel. The rendered half lives in the vitest file, and the
real-Chrome half (#3854's leg: these bytes, loaded, with the backend stopped, pressed) lives
in neither.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "web/src/sidepanel/SidePanelApp.tsx"
SPEC = ROOT / "web/src/sidepanel/SidePanelView.test.tsx"
VITEST_CONFIG = ROOT / "web/vitest.config.ts"
PACKAGE_JSON = ROOT / "web/package.json"

#: The three strings the panel shows for a focus push. Clause 2 pins them verbatim.
COPY = ("Check it out, Lloyd", "Ready when you are", "No checkable page here")

#: The modules the spec may import. `react-dom/server` is the render, `vitest` is the
#: harness, `./SidePanelApp` is the subject — anything DOM-shaped is the thing this
#: route exists to avoid (`web/package.json` is a denied build input, so it could not
#: be added and the spec would simply stop collecting).
ALLOWED_IMPORTS = {"vitest", "react-dom/server", "./SidePanelApp"}


class Broken(Exception):
    """One link of the panel-side contract is missing. Carries what was looked for."""


def _need(cond: bool, what: str) -> None:
    if not cond:
        raise Broken(what)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _code(src: str) -> str:
    """Source with its prose removed.

    Whole-line `//` comments and `/* … */` blocks only. Whole-line is the shape prose
    takes in both files here, and the narrowness is deliberate: an unanchored `//` strip
    would also eat the tail of `/disabled=""/` and of `'https://example.com/article'`,
    mangling code that has nothing to do with prose.
    """
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"^[ \t]*//[^\n]*$", "", src, flags=re.M)


def _fn_body(src: str, head: str, label: str) -> str:
    """The body of the function whose definition starts with `head`.

    Paren-matched first, then brace-matched: the props-destructuring brace of
    `function CheckButton({…}: {…})` is NOT where the body starts, and a scan that
    assumed it was would read the parameter list as the whole function and never see the
    `disabled` expression at all.
    """
    i = src.find(head)
    _need(i >= 0, f"{label}: no {head!r}")
    p = src.find("(", i)
    _need(p >= 0, f"{label}: no parameter list after {head!r}")
    depth = 0
    close = -1
    for k in range(p, len(src)):
        if src[k] == "(":
            depth += 1
        elif src[k] == ")":
            depth -= 1
            if depth == 0:
                close = k
                break
    _need(close > 0, f"{label}: unbalanced parentheses after {head!r}")
    j = src.find("{", close)
    _need(j > 0, f"{label}: no body after {head!r}")
    depth = 0
    for k in range(j, len(src)):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[j + 1:k]
    raise Broken(f"{label}: unbalanced braces in the body of {head!r}")


def _check_mapping(app: str, spec: str = "") -> None:
    """Clause 1: the exported mapping, pure, and the panel's only construction site."""
    code = _code(app)
    body = _fn_body(
        code,
        "export function focusStateFromMessage(msg: FocusMessage): PanelState",
        "the shipped panel",
    )
    for expr, meaning in (
        (r"sessionKey:\s*msg\.sessionKey\b", "sessionKey passed through as-is"),
        (r"url:\s*msg\.url\s*\?\?\s*null", "url defaulted with `?? null`"),
        (r"title:\s*msg\.title\s*\?\?\s*null", "title defaulted with `?? null`"),
        (r"canCheck:\s*Boolean\(\s*msg\.canCheck\s*\)", "canCheck read off the message"),
    ):
        _need(re.search(expr, body) is not None,
              f"focusStateFromMessage does not do {meaning}: {expr!r} is not in its body")
    impure = re.search(r"\b(chrome|window|useState|localStorage|Date|Math|fetch)\b", body)
    _need(impure is None,
          "focusStateFromMessage is not pure: it reaches "
          + (f"{impure.group(0)!r} " if impure else "something ")
          + "instead of only its argument")
    calls = code.count("setState(")
    _need(calls == 1,
          f"the panel builds state in more than one place: {calls} `setState(` call(s), "
          "expected the one that calls focusStateFromMessage")
    _need(re.search(r"setState\(\s*focusStateFromMessage\(\s*msg\s*\)\s*\)", code) is not None,
          "the port listener no longer calls `setState(focusStateFromMessage(msg))` — the "
          "mapping has been inlined back into it, which is the #3568 shape this file pins")


def _check_components(app: str, spec: str = "") -> None:
    """Clause 2: the two exports, their `disabled` expression, and the shipped copy."""
    code = _code(app)
    _need("export function CheckButton(" in code, "CheckButton is not exported")
    _need("export function EmptyState(" in code, "EmptyState is not exported")
    button = _fn_body(code, "export function CheckButton(", "CheckButton")
    _need(re.search(r"disabled=\{\s*pending\s*\|\|\s*!canCheck\s*\}", button) is not None,
          "CheckButton's `disabled` is no longer `{pending || !canCheck}` — the flag has "
          "been dropped from the button, which is one half of the #3568 failure")
    empty = _fn_body(code, "export function EmptyState(", "EmptyState")
    _need(re.search(r"canCheck\s*\?\s*\"Ready when you are\"\s*:\s*\"No checkable page here\"",
                    empty) is not None,
          "EmptyState no longer chooses \"Ready when you are\" over \"No checkable page "
          "here\" on `canCheck`")
    _need(re.search(r'const CHECK_LABEL = "Check it out, Lloyd"', code) is not None,
          "`Check it out, Lloyd` is no longer the CHECK_LABEL constant the bundle pin "
          "harvests from this source")
    for string in COPY:
        _need(f'"{string}"' in code, f"{string!r} is not a literal the panel ships")


def _check_spec(app: str, spec: str) -> None:
    """Clause 3: the spec renders only what the mapping returned, and says so in strings
    the panel still ships."""
    code = _code(spec)
    mods = set(re.findall(r"from\s+['\"]([^'\"]+)['\"]", code))
    _need(mods == ALLOWED_IMPORTS,
          f"the spec imports {sorted(mods)}; expected exactly {sorted(ALLOWED_IMPORTS)} — "
          "a DOM package among them means the node-environment route was abandoned, and a "
          "missing './SidePanelApp' means it is no longer testing the shipped panel")
    _need("renderToStaticMarkup(" in code, "the spec no longer renders with renderToStaticMarkup")
    n_it = len(re.findall(r"\bit\(", code))
    _need(n_it >= 5,
          f"the spec has {n_it} `it(` nodes; clause 3 needs at least the five named "
          "assertions, and a spec that lost one has lost an assertion this clause pins")
    body = _fn_body(code, "function panelFor(msg: FocusMessage)", "the spec's renderer")
    _need("focusStateFromMessage(msg)" in body,
          "panelFor no longer calls focusStateFromMessage — the renders are being driven by "
          "a hand-written flag, which is the shape that let clause 3's mutation pass green "
          "in the triage measurement")
    props = [p.strip() for p in re.findall(r"canCheck=\{([^}]*)\}", code)]
    _need(len(props) >= 2,
          f"the spec passes `canCheck` {len(props)} time(s); the button and the empty state "
          "must both be rendered from the mapping")
    _need(all(p == "state.canCheck" for p in props),
          f"the spec renders `canCheck` from {sorted(set(props))}; every render must read "
          "the mapping's output rather than a literal")
    region = code[code.find("FLAG_OMITTED_PUSH"):]
    region = region[: region.find("}") + 1]
    _need("canCheck" not in region,
          "the spec's FLAG_OMITTED_PUSH fixture carries a `canCheck` key — #3568's shape is "
          "an ABSENT field, and `" + "canCheck: false` would pin a different message")
    app_code = _code(app)
    asserted = set(re.findall(r"\.toContain\(\s*['\"]([^'\"]+)['\"]", code))
    for string in COPY:
        _need(string in asserted,
              f"the spec no longer asserts the shipped string {string!r} with toContain")
    for string in asserted:
        if string.startswith("<"):
            continue  # a tag fragment, not copy
        _need(string in app_code,
              f"DRIFT: the spec asserts {string!r}, which the panel no longer ships")


def _include_rx(pattern: str) -> str:
    """A vitest `include` glob as a regex: `**/` crosses directories, `*` stays inside
    one, `{ts,tsx}` is a choice — the three constructs this config's pattern uses."""
    out: list[str] = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "{":
            j = pattern.find("}", i)
            _need(j > i, f"unbalanced `{{` in include pattern {pattern!r}")
            out.append("(?:" + "|".join(re.escape(a) for a in pattern[i + 1:j].split(",")) + ")")
            i = j + 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return "^" + "".join(out) + "$"


def _mutate(what: str, src: str, old: str, new: str, times: int) -> str:
    """Apply a mutation, refusing to pretend a mutation that is not there was applied."""
    found = src.count(old)
    _need(found >= times,
          f"MUTATION STALE: {what} — {old!r} occurs {found} time(s), expected {times}")
    return src.replace(old, new)


#: (what is broken, the file, the exact text, its replacement, how many copies the
#: replacement must cover, the checker that must refuse). Anchored on the shipped text,
#: not on a line number: the mapping line and `INITIAL`'s :25 both read `canCheck: false,`
#: today, so a line-targeted edit would take both, the module would stop collecting, and
#: the run would report a caught mutation that is really a compile error.
MUTATIONS = (
    ("the button stops honouring the flag", APP,
     "disabled={pending || !canCheck}", "disabled={pending}", 1, _check_components),
    ("the mapping hardcodes the flag true", APP,
     "canCheck: Boolean(msg.canCheck),", "canCheck: true,", 1, _check_mapping),
    ("the mapping hardcodes the flag false", APP,
     "canCheck: Boolean(msg.canCheck),", "canCheck: false,", 1, _check_mapping),
    ("the mapping is inlined back into the listener", APP,
     "setState(focusStateFromMessage(msg))",
     "setState({\n          sessionKey: msg.sessionKey,\n          url: msg.url ?? null,\n"
     "          title: msg.title ?? null,\n          canCheck: Boolean(msg.canCheck),\n        })",
     1, _check_mapping),
    ("the refusal copy moved and the spec did not", APP,
     '"No checkable page here"', '"Nothing to check here"', 1, _check_spec),
    ("the spec hand-writes the flag", SPEC,
     "canCheck={state.canCheck}", "canCheck={true}", 2, _check_spec),
)


def test_the_focus_push_mapping_is_exported_pure_and_is_the_only_construction_site():
    """Clause 1. `sessionKey` as-is, `url`/`title` via `?? null`, `canCheck:
    Boolean(msg.canCheck)`, nothing outside the argument read, and one `setState(` in the
    file — the one that calls the mapping."""
    _check_mapping(_read(APP))


def test_the_exported_components_carry_the_shipped_disabled_expression_and_copy():
    """Clause 2. `CheckButton` and `EmptyState` exported, `disabled={pending ||
    !canCheck}`, and all three strings present as literals in the source the panel bundle
    is built from."""
    _check_components(_read(APP))


def test_the_spec_renders_only_what_the_mapping_returned():
    """Clause 3. Both components, both renders, `canCheck={state.canCheck}` out of
    `panelFor` and nothing hand-written; the #3568 fixture omits the field; and every
    string it asserts is a string the panel ships."""
    _check_spec(_read(APP), _read(SPEC))


def test_the_spec_is_collected_by_the_unchanged_config_with_no_dom_package():
    """Clause 5's premise, checked rather than asserted in prose: `web/vitest.config.ts`
    `include` is `src/**/*.test.{ts,tsx}` with `web/` as its root, which reaches
    `src/sidepanel/SidePanelView.test.tsx`, and no DOM renderer exists to need one."""
    cfg = _read(VITEST_CONFIG)
    arr = re.search(r"include:\s*\[([^\]]*)\]", cfg)
    if arr is None:
        raise Broken("web/vitest.config.ts has no `include:` array to read")
    patterns = re.findall(r"['\"]([^'\"]+)['\"]", arr.group(1))
    _need("src/**/*.test.{ts,tsx}" in patterns,
          f"the include patterns are {patterns}; this node's claim that the unchanged "
          "config collects the spec needs that exact pattern")
    rel = f"src/{SPEC.relative_to(ROOT / 'web').as_posix()}"
    _need(any(re.match(_include_rx(p), rel) for p in patterns),
          f"{rel} is not collected by {patterns} — the spec would not run")
    _need(re.search(r"environment:\s*['\"]node['\"]", cfg) is not None,
          "web/vitest.config.ts no longer pins `environment: \"node\"`, which is what makes "
          "renderToStaticMarkup the route instead of a DOM package")
    deps = json.loads(_read(PACKAGE_JSON))
    merged = {**deps.get("dependencies", {}), **deps.get("devDependencies", {})}
    dom = sorted(k for k in merged
                 if k in {"jsdom", "happy-dom"} or k.startswith("@testing-library/"))
    _need(not dom,
          f"a DOM renderer arrived in web/package.json ({dom}); this route exists because "
          "none may be added")


def test_the_scanners_are_not_vacuous_on_the_shipped_bytes():
    """Positive control: the counts the other nodes rely on are actually present, so a
    scan that silently matches nothing cannot pass as a scan that found the contract."""
    app, spec = _read(APP), _read(SPEC)
    code, scode = _code(app), _code(spec)
    _need(code.count("setState(") == 1, f"{code.count('setState(')} setState calls")
    _need("focusStateFromMessage" in code, "the app never names the mapping")
    _need(len(re.findall(r"\bit\(", scode)) >= 5, "fewer than five it() nodes")
    _need(len(re.findall(r"canCheck=\{[^}]*\}", scode)) == 2,
          "the spec does not pass canCheck exactly twice (button + empty state)")
    _need(scode.count("renderToStaticMarkup(") == 2,
          "renderToStaticMarkup is not called exactly twice (the button and the empty state)")
    _need(scode.count("renderToStaticMarkup(")
          == _fn_body(scode, "function panelFor(msg: FocusMessage)", "panelFor"
                     ).count("renderToStaticMarkup("),
          "a render happens outside panelFor, where the mapping is not what drives it")
    _need(_fn_body(scode, "function panelFor(msg: FocusMessage)", "panelFor").count(
        "state.canCheck") == 2, "panelFor does not thread state.canCheck twice")


def test_the_mirror_bites_on_each_of_six_mutations():
    """Every break in MUTATIONS must be refused by the checker that owns its clause.

    A mutation that is not found in the source fails the node too (`_mutate`): a stale
    anchor is not a caught mutation, and the last time this file's contract was checked
    by a text match, an unanchored edit took `INITIAL`'s `canCheck: false` along with the
    mapping's and the module stopped collecting — which reads as red and means nothing.
    """
    app0, spec0 = _read(APP), _read(SPEC)
    for what, path, old, new, times, check in MUTATIONS:
        app, spec = app0, spec0
        if path is APP:
            app = _mutate(what, app0, old, new, times)
        else:
            spec = _mutate(what, spec0, old, new, times)
        try:
            check(app, spec)
        except Broken:
            continue
        raise AssertionError(
            f"the mirror is blind to its own mutation: {what} "
            f"({path.name}) — {check.__name__} accepted the broken bytes")
