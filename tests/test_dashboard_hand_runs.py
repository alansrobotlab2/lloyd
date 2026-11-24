"""#4515: the owed card renders a pending by-hand run, and the pin reddens on the two wrong fixes.

`owed.hand_runs()` is the readout for a decision that is RULED but not EXECUTED. It
reaches the browser and is declared in the client type, but from the commit that
shipped it (`13d24da8`, #3017) until this round **no component rendered it**, so the
dashboard's owed card — the only surface a human reads — never named a pending
carrier or the command an attended session has to run. The data was delivered and
thrown away.

The seam, hop by hop, and which node holds each one:

  1. `scripts/automod/owed.py::hand_runs()` builds the rows, and
     `board_health()` (`scripts/automod/backlog.py`) puts them on the board payload
     as `owed.hand_runs` — pinned by
     `tests/test_owed_check.py::test_board_health_lists_a_pending_hand_run_beside_the_outside_list`.
  2. The route serves that payload (`app/routers/dashboard.py`) and `web/src/api.ts`
     declares `BacklogOwed.hand_runs` against the writer's own `HAND_RUN_FIELDS` —
     pinned by `tests/test_api_contracts.py::test_the_web_client_declares_the_pending_hand_run_row_the_board_emits`.
  3. A component renders it. **This file is that hop, and it is the only one of the
     three that could have gone missing while the suite stayed green:** emission and
     declaration were pinned, rendering was not, which is exactly how an unrendered
     field survives a year.

Why a source-text pin rather than a rendered component. The repo has no harness that
renders a component in a test — `web/package.json` carries `vitest` but no
`@testing-library/*`, no `jsdom`, no `happy-dom` — and the review rung's node rail
accepts a node id only in a `.py` file under a `pytest.ini` testpath
(`scripts/automod/testpaths.py::is_test_file`), so a clause pinned only in a
`.test.ts` is refused however well it was verified. `tests/test_dashboard_label_contract.py`
established the pattern (round `SM_20261005_002918` lost two clauses that way); this
file follows it. It is a reading of the shipped bytes, not a measurement of a pixel.

What the reader does that a grep cannot, and why each half is needed:

  * comments are blanked first, so the prose above the block — which names
    `hand_runs()` and `cmd` — can never answer for the code;
  * the row's fields are searched in the element's VISIBLE text, with every nested
    opening tag blanked, because `key={`hand-run-${h.item_id}`}` already spells
    `h.item_id` in an attribute no reader sees;
  * the cmd's wrapping is looked for on the element that carries the interpolation
    and its ancestors **inside this block only**: the enclosing `BacklogHealthLines`
    div already inherits `font-mono` and `break-words`, so letting the walk leave the
    block would make clause 3 impossible to fail.

Run it: `pytest tests/test_dashboard_hand_runs.py`. No browser, no vite, no
`node_modules`.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAGE = REPO / "web" / "src" / "components" / "pages" / "DashboardPage.tsx"

#: The two declarations clause 3 names, and the pair the `owed_outside` block beside
#: this one already carries (`break-words`; #3141's red tree was one unbreakable
#: token painting past the card). `font-mono` is the monospace face.
MONO_FACE = "font-mono"
WRAP = "break-words"

#: The three fields clause 1 names, spelled as the client type names them
#: (`scripts/automod/owed.py::HAND_RUN_FIELDS` is `item_id, name, set, cmd, since`).
ROW_FIELDS = ("item_id", "set", "cmd")

#: The owed card's pre-existing text, quoted off the shipped bytes. Clause 2 is that
#: an empty queue leaves the card as it was, so these are the lines a rewrite of the
#: summary or of the `outside` rendering would have to touch.
EXISTING_OWED_TEXT = (
    "owed {health.owed.entries} on {health.owed.items} item"
    "{health.owed.items === 1 ? '' : 's'} · due {health.owed.due}",
    '<span className="text-amber-400"> · needs your hands {health.owed.outside.length}</span>',
    "{health.owed?.outside.map((o) => (",
    "<div key={`${o.item_id}-${o.what}`} className=\"break-words text-amber-400/90\">",
    "#{o.item_id}: {o.needs || o.what}",
)


# ── the reader ──────────────────────────────────────────────────────────────

def _mask(src: str) -> str:
    """`src` with every comment replaced by spaces of the SAME length.

    The comment above the block says `owed.hand_runs()` and names `cmd`, so a scanner
    that read comments as code would find the field rendered by prose alone. Offsets
    are preserved rather than removed, so a position in the mask is the same position
    in the real bytes and a mutation can be spliced into them.
    """
    out = list(src)
    i, n = 0, len(src)
    while i < n:
        ch = src[i]
        if ch in "\"'`":
            q, i = ch, i + 1
            while i < n:
                if src[i] == "\\":
                    i += 2
                    continue
                if src[i] == q:
                    i += 1
                    break
                i += 1
            continue
        if ch == "/" and i + 1 < n and src[i + 1] == "/":
            j = src.find("\n", i)
            j = n if j < 0 else j
            out[i:j] = [" "] * (j - i)
            i = j
            continue
        if ch == "/" and i + 1 < n and src[i + 1] == "*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out[i:j] = [" "] * (j - i)
            i = j
            continue
        i += 1
    return "".join(out)


def _enclosing_expr(masked: str, pos: int):
    """`(open, close)` of the JSX expression container `{…}` holding `masked[pos]`.

    Walked by brace depth rather than by line, because the block is an expression
    spanning several lines whose braces nest: `(h) => (…)` and `${h.item_id}`.
    """
    depth = 0
    i = pos
    while i >= 0:
        ch = masked[i]
        if ch == "}":
            depth += 1
        elif ch == "{":
            if depth == 0:
                break
            depth -= 1
        i -= 1
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(masked)):
        if masked[j] == "{":
            depth += 1
        elif masked[j] == "}":
            depth -= 1
            if depth == 0:
                return (i, j)
    return None


def _scan_tag(masked: str, lt: int, bound: int):
    """`(name, attrs, selfclosing, tag_end)` for the JSX tag opening at `masked[lt]`."""
    m = re.match(r"<([A-Za-z][\w.]*)", masked[lt:bound])
    if not m:
        return None
    name = m.group(1)
    i = lt + m.end()
    depth = 0
    while i < bound:
        ch = masked[i]
        if ch in "\"'`":
            q, i = ch, i + 1
            while i < bound:
                if masked[i] == "\\":
                    i += 2
                    continue
                if masked[i] == q:
                    i += 1
                    break
                i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == ">" and depth == 0:
            body = masked[lt + len(name) + 1:i]
            return (name, body, body.rstrip().endswith("/"), i + 1)
        i += 1
    return None


def _elements(masked: str, start: int, end: int):
    """Yield `(name, attrs, tag_lt, content_start, content_end, ancestors)` per element.

    Document order, outermost first, each restricted to `masked[start:end]`. The
    ancestors tuple is the enclosing element stack at the tag's own opening, which is
    what clause 3 walks to find the wrapping the cmd inherits from inside this block.
    """
    stack: list[tuple[str, str, int, tuple]] = []
    i = start
    while i < end:
        lt = masked.find("<", i, end)
        if lt < 0:
            break
        if masked[lt + 1:lt + 2] == "/":
            m = re.match(r"</\s*([A-Za-z][\w.]*)\s*>", masked[lt:end])
            if m and stack and stack[-1][0] == m.group(1):
                name, attrs, tag_lt, cs, anc = stack.pop()
                yield (name, attrs, tag_lt, cs, lt, anc)
                i = lt + m.end()
                continue
            i = lt + 1
            continue
        scanned = _scan_tag(masked, lt, end)
        if scanned is None:
            i = lt + 1
            continue
        name, attrs, selfclosing, tag_end = scanned
        if selfclosing:
            yield (name, attrs, lt, tag_end, tag_end, tuple(stack))
        else:
            stack.append((name, attrs, lt, tag_end, tuple(stack)))
        i = tag_end


def _class_tokens(attrs: str) -> list[str]:
    """Class tokens the element is given, read through a computed `className={cn(…)}`.

    Only quoted literals are harvested: an identifier resolves to something this file
    cannot see, so it is never guessed at.
    """
    m = re.search(r"(?<![\w-])className\s*=\s*\{", attrs)
    if not m:
        lit = re.search(r"(?<![\w-])className\s*=\s*\"([^\"]*)\"", attrs)
        return (lit.group(1) if lit else "").split()
    depth, end = 0, len(attrs)
    for j in range(m.end() - 1, len(attrs)):
        if attrs[j] == "{":
            depth += 1
        elif attrs[j] == "}":
            depth -= 1
            if depth == 0:
                end = j
                break
    return [t for trio in re.findall(r"'([^']*)'|\"([^\"]*)\"|`([^`]*)`", attrs[m.end():end])
            for t in " ".join(trio).split()]


class HandRunRender:
    """What the shipped bytes say the owed card does with `health.owed.hand_runs`."""

    def __init__(self, masked: str, open_at: int, close_at: int, param: str | None):
        self.masked = masked
        self.open_at = open_at
        self.close_at = close_at
        self.expr = masked[open_at:close_at + 1]
        self.param = param
        # `_elements` yields when a tag CLOSES, so the innermost element comes back
        # first; sorting by opening position puts the map's row element at [0].
        self.elements = sorted(_elements(masked, open_at, close_at + 1), key=lambda e: e[2])

    @property
    def row(self):
        """The outermost element the map returns — one per row, which is the line."""
        return self.elements[0] if self.elements else None

    def visible_text(self) -> str:
        """The row's text: its content with every nested opening tag blanked.

        `key={`hand-run-${h.item_id}`}` spells `h.item_id` inside an attribute no
        reader sees, so searching the raw block would let a row that names the id
        nowhere in its text pass clause 1.
        """
        r = self.row
        if r is None:
            return ""
        _, _, _, cs, ce, _ = r
        out = list(self.masked[cs:ce])
        for name, attrs, tag_lt, content_start, _, _ in self.elements:
            if cs <= tag_lt < ce:
                for k in range(tag_lt - cs, min(content_start - cs, len(out))):
                    out[k] = " "
        return "".join(out)

    def cmd_chain(self):
        """`(attrs, …)` for the element carrying the cmd interpolation and its
        ancestors inside this block, innermost first; [] when nothing carries it."""
        if self.param is None:
            return []
        m = re.search(r"\{\s*" + re.escape(self.param) + r"\.cmd\b", self.expr)
        if not m:
            return []
        pos = self.open_at + m.start()
        # Innermost first: the cmd sits in a span inside the row div, and the div would
        # also "contain" it while carrying only half the required pair.
        for name, attrs, tag_lt, cs, ce, anc in reversed(self.elements):
            if cs <= pos < ce:
                # Ancestors are the stack entries enclosing this tag: 5-tuples,
                # `(name, attrs, tag_lt, content_start, whose_ancestors)`.
                return [attrs] + [a for (_, a, _, _, _) in anc]
        return []


def _read(src: str) -> HandRunRender | None:
    """Locate the owed card's hand-run render in the page's bytes, or None.

    None is the pre-#4515 state: the field is read nowhere in the page.
    """
    masked = _mask(src)
    hit = masked.find("hand_runs")
    if hit < 0:
        return None
    span = _enclosing_expr(masked, hit)
    if span is None:
        return None
    open_at, close_at = span
    expr = masked[open_at:close_at + 1]
    m = re.search(r"\.map\s*\(\s*\(?([A-Za-z_$][\w$]*)\)?\s*=>", expr)
    return HandRunRender(masked, open_at, close_at, m.group(1) if m else None)


def _field_clauses(r: HandRunRender) -> list[str]:
    """Clause 1: one line per row, naming item_id, set and cmd in that line's text."""
    bad: list[str] = []
    if ".map" not in r.expr:
        bad.append("the hand-run block is not a map over the rows, so it cannot "
                   "render one line per row of health.owed.hand_runs")
        return bad
    if r.row is None:
        bad.append("the hand-run map returns no element, so there is no line to name")
        return bad
    if r.param is None:
        bad.append("the reader cannot name the map's row parameter, so no field "
                   "interpolation can be attributed to a row")
        return bad
    text = r.visible_text()
    for field in ROW_FIELDS:
        if not re.search(r"\{\s*" + re.escape(r.param) + r"\." + field + r"\b", text):
            bad.append(f"the rendered row does not name the row's {field} in its text")
    return bad


def _empty_clauses(r: HandRunRender) -> list[str]:
    """Clause 2: nothing new is emitted while the queue is empty or the field absent."""
    bad: list[str] = []
    null_safe = (re.search(r"hand_runs\s*\?\.\s*map", r.expr)                 # `{?.map`
                 or re.search(r"hand_runs[^}]*\?\?\s*\[\s*\]", r.expr)        # `?? []`
                 or re.search(r"hand_runs\s*\?\.\s*length", r.expr))          # `{?.length`
    if not null_safe:
        bad.append("an absent hand_runs would throw instead of rendering nothing: "
                   "the map is neither optional-chained, coalesced to [], nor "
                   "guarded on a?.length")
    if "<" in r.expr and ".map" in r.expr and r.expr.index("<") < r.expr.index(".map"):
        bad.append("the block opens an element before the map, so an empty queue "
                   "would still render that element")
    return bad


def _wrap_clauses(r: HandRunRender) -> list[str]:
    """Clause 3: the cmd sits in an element carrying a monospace face and break-words."""
    chain = r.cmd_chain()
    if not chain:
        return ["the cmd is not rendered inside any element of the hand-run block"]
    for attrs in chain:
        tokens = _class_tokens(attrs)
        if MONO_FACE in tokens and WRAP in tokens:
            return []
    return [f"the cmd's element and its in-block ancestors carry no class list with "
            f"both {MONO_FACE} and {WRAP}; their own tokens are "
            f"{[_class_tokens(a) for a in chain]}"]


def _existing_text_clauses(src: str) -> list[str]:
    """Clause 2's other half: the owed card's existing text is byte-for-byte as it was."""
    return [f"the owed card's existing text changed: {line!r} is gone"
            for line in EXISTING_OWED_TEXT if line not in src]


# ── the clauses ─────────────────────────────────────────────────────────────

def test_the_owed_card_renders_one_line_per_pending_hand_run():
    """#4515 clause 1. Each row of `health.owed.hand_runs` renders a line, and that
    line names the row's `item_id`, its `set` and its `cmd` in the text a reader sees.

    Before this round the reader returned None: `grep -rn hand_run web/src
    --include=*.tsx` was empty, and the whole field existed only as `api.ts:2681`'s
    declaration. `test_the_pin_is_red_when_no_component_reads_hand_runs` keeps that
    the state this node refuses.
    """
    r = _read(PAGE.read_text())
    assert r is not None, (
        "no component reads health.owed.hand_runs: the backend emits the rows and "
        "the card never renders them, which is the #4515 defect")
    assert _field_clauses(r) == [], _field_clauses(r)


def test_the_row_named_by_the_reader_is_a_real_row_and_not_a_key():
    """Positive control on the reader: it located ONE map, returning ONE element, whose
    visible text is what the assertions above were run over.

    Without this, a reader that matched nothing would satisfy `test_…` by returning an
    empty violation list — the `test_the_reader_locates_the_seven_rows` precedent in
    `tests/test_dashboard_label_contract.py`.
    """
    r = _read(PAGE.read_text())
    assert r is not None, "the hand-run render is gone from the owed card"
    assert r.expr.count(".map") == 1, f"expected one map, got: {r.expr}"
    assert r.row is not None and r.row[0] == "div", (
        f"the row element is {r.row[0] if r.row else None}, not the div the card's "
        f"other owed lines use: {r.expr}")
    text = r.visible_text()
    for field in ("item_id", "set", "cmd"):
        assert field in text, f"{field} is not in the row's visible text: {text!r}"
    assert "key" not in text, (
        f"the reader failed to blank an opening tag, so attributes leaked into the "
        f"visible text it searches: {text!r}")


def test_an_empty_or_absent_hand_run_queue_renders_nothing_new():
    """#4515 clause 2. With `hand_runs` absent or `[]` the card emits no new element and
    its existing text — `owed N on M items · due D`, `needs your hands N`, the
    `outside` lines — is unchanged.

    An empty array mapped to children contributes no DOM node, so the falsifiable
    halves are the ABSENT case (the access must not throw, which is what
    `_empty_clauses` reads) and the pre-existing lines, quoted off the bytes.
    """
    r = _read(PAGE.read_text())
    assert r is not None, "no hand-run render to reason about"
    assert _empty_clauses(r) == [], _empty_clauses(r)
    assert _existing_text_clauses(PAGE.read_text()) == [], _existing_text_clauses(PAGE.read_text())


def test_the_cmd_is_wrapped_in_a_monospace_face_that_can_break():
    """#4515 clause 3. The `cmd` is one unbreakable, path-heavy token, so the element
    carrying it must itself carry a monospace face and `break-words`, as the
    `owed_outside` block beside it carries `break-words` for the same reason (#3141:
    one such token painted past the card).

    The walk is bounded inside the block on purpose: the enclosing `BacklogHealthLines`
    div already has `font-mono break-words` and is inherited by everything in the
    card, so an outer match would make this clause unfailable.
    """
    r = _read(PAGE.read_text())
    assert r is not None, "no hand-run render to reason about"
    assert _wrap_clauses(r) == [], _wrap_clauses(r)


# ── the pin reddens on the wrong fixes ───────────────────────────────────────

def _splice(src: str, r: HandRunRender, replacement: str) -> str:
    """The page with the whole hand-run expression container replaced."""
    return src[:r.open_at] + replacement + src[r.close_at + 1:]


def test_the_pin_is_red_when_no_component_reads_hand_runs():
    """Clause 4's first red: delete the render, and the pin is the pre-#4515 tree —
    `hand_runs` read nowhere — which `_read` reports as None and
    `test_the_owed_card_renders_one_line_per_pending_hand_run` fails on.
    """
    src = PAGE.read_text()
    r = _read(src)
    assert r is not None, "nothing to mutate: the render is already absent"
    mutated = _splice(src, r, "")
    assert _read(mutated) is None, (
        "the reader still found a hand-run render after the block was deleted, so it "
        "is matching prose or some other field and every clause above is unreadable")
    assert "hand_runs" not in _mask(mutated)


def test_the_pin_is_red_when_the_rendered_row_omits_cmd():
    """Clause 4's second red: drop the cmd from the row's text and keep everything
    else, and the clause-1 node reddens naming the missing field.

    A row that shows `#4503 by hand · Kimchi by Cast AI -> Cast AI` without the
    command is the near-miss this item calls out: the carrier is named but the
    attended session still cannot see what to run.
    """
    src = PAGE.read_text()
    r = _read(src)
    assert r is not None, "nothing to mutate: the render is already absent"
    mutated = _splice(src, r, r.expr.replace("{h.cmd}", "{}"))
    assert "h.cmd" not in _mask(mutated), (
        "the mutation left a cmd interpolation somewhere else in the block, so it "
        "would not catch the wrong fix it exists to catch")
    bad = _field_clauses(_read(mutated))
    assert bad and any("cmd" in b for b in bad), (
        f"omitting the cmd left the clause-1 check green: {bad}")


def test_the_pin_is_red_when_the_cmd_loses_its_wrapping():
    """Clause 4's third red: strip `font-mono` off the cmd's element and clause 3
    reddens. Inheriting the face from the card is exactly what the clause refuses to
    rely on, because the class list is the only thing a later reader can check.
    """
    src = PAGE.read_text()
    r = _read(src)
    assert r is not None, "nothing to mutate: the render is already absent"
    mutated = _splice(src, r, r.expr.replace(f'"{WRAP} {MONO_FACE}"', f'"{WRAP}"'))
    after = _read(mutated)
    assert after is not None, "the mutation deleted the block instead of its class"
    assert MONO_FACE not in after.expr, (
        "font-mono survived the mutation elsewhere in the block, so the wrong fix "
        "this node exists to catch was not actually applied")
    bad = _wrap_clauses(after)
    assert bad, "losing the monospace face left clause 3 green"


def test_the_pin_is_red_when_an_absent_field_would_throw():
    """Clause 4's fourth red: `health.owed.hand_runs.map(…)` — no `?.` — renders the
    same thing on a full queue and on an empty one, so only the null-safety read tells
    the two apart, and clause 2 needs them apart: a snapshot written before #3017
    carries no `hand_runs` key at all.
    """
    src = PAGE.read_text()
    r = _read(src)
    assert r is not None, "nothing to mutate: the render is already absent"
    assert "?.map" in r.expr, "the shipped map is not optional-chained, so the " \
                              "mutation this node exists to run has no target"
    mutated = _splice(src, r, r.expr.replace("?.map", ".map"))
    bad = _empty_clauses(_read(mutated))
    assert bad and any("absent" in b for b in bad), (
        f"a map that throws on an absent hand_runs left clause 2 green: {bad}")


def test_the_pin_is_red_when_the_owed_summary_is_rewritten():
    """Clause 4's fifth red: clause 2 says the existing owed text is unchanged, so a
    rewrite of the summary line reddens it — a pin that cannot notice the card being
    rewritten would pass on any change, which is the failure this file exists to avoid.
    """
    src = PAGE.read_text()
    old = EXISTING_OWED_TEXT[0]
    assert old in src, "the quoted owed summary line is not in the page: the pin is stale"
    mutated = src.replace(old, "owed {health.owed.entries} items", 1)
    bad = _existing_text_clauses(mutated)
    assert bad, "rewriting the owed summary line left clause 2 green"
