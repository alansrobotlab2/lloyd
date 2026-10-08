"""#2202 clauses 1 and 2: the seven row labels that are their row's only text must
carry a wrapping base, and the pin that says so must RED on the two wrong fixes.

Two files pin one contract, and each does the half the other cannot:

  * `web/src/components/pages/DashboardResponsive.test.ts` is the contract the
    frontend's own test suite enforces, which is where a developer editing the page
    finds out, and which the automod gate runs (`gate.py::_vitest_run`).
  * THIS file is the same contract as pytest nodes over the same bytes. The review
    rung's node rail accepts a node id only in a test file under one of
    `pytest.ini`'s testpaths — `scripts/automod/testpaths.py::is_test_file` requires
    a `.py` — so a clause pinned only in a `.test.ts` is refused as
    `test_node_id not in a test file this diff changed` however well it was
    verified. That is how round SM_20261005_002918 lost clauses 1 and 2 while its
    own review text said their substance was met.

`test_the_two_halves_of_the_contract_agree` keeps them from splitting: the span
table is PARSED out of the vitest file here rather than typed a second time, so a
key or a count that moves on one side reddens the other.

What this file does differently, and why it is not only a translation: the vitest
audit decides "is this label wrapped in a link or a button" by INDENTATION and says
so in its own comment. A hand-written wrapper is one level shallower than its
child, so that works, but it is a formatting bet. Here the answer comes from a JSX
element tree, so an `a`/`button` ancestor is named whether it sits three lines up at
a shallower indent or at its child's own indent — and the formatting-blind case is
pinned by a mutation of its own below.

Run it: `pytest tests/test_dashboard_label_contract.py`. No browser, no vite, no
`node_modules`: nothing here renders a pixel. The measured half of #2202 is
`tests/test_dashboard_responsive.py`, and `test_the_reader_locates_the_seven_rows`
is this file's own positive control.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAGE = REPO / "web" / "src" / "components" / "pages" / "DashboardPage.tsx"
CONTRACT = REPO / "web" / "src" / "components" / "pages" / "DashboardResponsive.test.ts"

#: The wrapping base, which must be unconditional, and the ellipsis, which must not.
#: `truncate` is `overflow:hidden; text-overflow:ellipsis; white-space:nowrap` and the
#: nowrap is what turns one long task name into one clipped line; `whitespace-normal`
#: removes it, and `break-words` (`overflow-wrap:break-word`) is what lets an
#: unbreakable run like `scheduled-task:autonomy-self-improvement-pipeline-stage-two`
#: split inside a 232 px column instead of overflowing it. `sm:truncate` restores the
#: ellipsis from 640 px up, which is what keeps the desktop rows one line apiece. A
#: bare `truncate` fails the fix below `sm` while still satisfying any "contains
#: truncate" check, which is why it is forbidden here by name.
WRAPPING_BASE = ("whitespace-normal", "break-words")
VIEWPORT_ELLIPSIS = "sm:truncate"

#: What lets the row reach that wrapping at all, and the half #2298 found missing.
#: Every one of these seven spans is a child of a `flex items-center gap-2` row, and
#: a flex item with visible overflow has `min-width: auto`, which resolves to its
#: MIN-CONTENT size — so the row could never get narrower than the longest
#: unbreakable token in its label. That is also why the defect only exists below
#: `sm`: `sm:truncate` brings `overflow:hidden`, and an item whose overflow is not
#: visible takes an automatic minimum of zero, so from 640 px up the same span was
#: never floored and the desktop dashboard has never shown this.
#: `break-words` cannot help there: `overflow-wrap: break-word` only breaks a word
#: that already does not fit its line box, and per CSS Text it changes no intrinsic
#: size (only `anywhere` and `word-break: break-all` do). Measured on the live tree
#: at 320 px: a `Recent runs` row 262 px wide carried 277 px of content because its
#: summary held `bench_027_recall_user_fact_topic_read:`, so the `ml-auto
#: flex-shrink-0` duration landed with its right edge at x=306 against a section
#: whose content box ends at 304, and those 2 px reddened
#: `test_no_section_overflows_its_box_on_a_phone[320]`. `min-w-0` releases the
#: automatic minimum so the flex algorithm can shrink the label to its share, and
#: only then does `break-words` have a narrow box to break inside. It is the same
#: declaration #1685 put on `Panel` for the grid-item version of this rule, pinned by
#: `tests/test_dashboard_mobile_sizing.py::test_panel_card_is_not_floored_at_its_content`.
SHRINK_RELEASE = "min-w-0"

#: The seven interpolations #1742 named, parsed out of the vitest contract. Keyed by
#: what each span renders, never by its class list: the class list is the thing under
#: test, and a key built from it would move with the regression it must catch.
_SPAN_ROW_RX = re.compile(r"\{\s*key:\s*'([^']*)'\s*,\s*count:\s*(\d+)\s*\}")

#: What may stand between a clipped label and its reader: an intrinsic `a` or
#: `button` ancestor, or a `title` on an intrinsic element at or above the span.
#: Component tags are NOT exempt — `<Section title="Services">` is a React prop that
#: becomes a heading, not a DOM tooltip, and every section on the page carries one.
INTERACTIVE_TAGS = frozenset({"a", "button"})

#: How far up to look before giving up: bounded so the walk stops at the panel that
#: owns the row instead of reaching an unrelated tooltip further out.
MAX_ANCESTOR_HOPS = 12


# ── the reader ──────────────────────────────────────────────────────────────

def _mask(src: str) -> str:
    """`src` with every comment replaced by spaces of the SAME length.

    Comment prose contains things like "below `sm`" and "a < b"; a scanner that read
    those as elements would desynchronise the tree and report a phantom ancestor.
    Offsets are preserved rather than removed, so a position in the mask is the same
    position in the original and a mutation can be spliced into the real bytes.
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


def _scan_tag(masked: str, lt: int):
    """`(name, attrs, selfclosing, end)` for the JSX tag opening at `masked[lt]`, or
    None when the `<` there is not a tag.

    The attribute run is read with brace and quote depth rather than up to the next
    `>`: `onClick={() => {}}` contains a `>` and `title={a > b}` contains two, so a
    `[^>]*` scan would end a tag in the middle of an expression and put every element
    after it in the wrong tree.
    """
    m = re.match(r"<([A-Za-z][\w.]*)", masked[lt:])
    if not m:
        return None
    name = m.group(1)
    i = lt + m.end()
    depth = 0
    n = len(masked)
    while i < n:
        ch = masked[i]
        if ch in "\"'`":
            q, i = ch, i + 1
            while i < n:
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


def _pop(stack, name=None):
    """Close `name` (or the innermost element, for `</>`)."""
    for k in range(len(stack) - 1, -1, -1):
        if name is None or stack[k][0] == name:
            del stack[k:]
            return True
    return False


def _elements(masked: str):
    """Yield `(start, name, attrs, ancestors)` for every element that opens in the
    page, in document order, `ancestors` being the element stack enclosing it."""
    stack: list[tuple[str, str, int]] = []
    i, n = 0, len(masked)
    while i < n:
        lt = masked.find("<", i)
        if lt < 0:
            return
        nxt = masked[lt + 1:lt + 2]
        if nxt == "/":
            if masked[lt:lt + 3] == "</>":
                _pop(stack)
                i = lt + 3
                continue
            m = re.match(r"</\s*([A-Za-z][\w.]*)\s*>", masked[lt:])
            if m:
                _pop(stack, m.group(1))
                i = lt + m.end()
                continue
            i = lt + 1
            continue
        if nxt == ">":
            stack.append((">", "", lt))                   # fragment: <> … </>
            i = lt + 2
            continue
        scanned = _scan_tag(masked, lt)
        if scanned is None:
            i = lt + 1
            continue
        name, attrs, selfclosing, end = scanned
        yield lt, name, attrs, tuple(stack)
        if not selfclosing:
            stack.append((name, attrs, lt))
        i = end


def _attr(attrs: str, name: str) -> str | None:
    """The value of `name="…"` or `name={…}` within a tag's attribute text; None if
    the attribute is absent. The lookbehind keeps `data-title` from answering for
    `title`."""
    m = re.search(r"(?<![\w-])" + re.escape(name) + r"\s*=\s*(\"([^\"]*)\"|\{)", attrs)
    if not m:
        return None
    return m.group(2) if m.group(2) is not None else "{expr}"


def _class_tokens(attrs: str) -> list[str]:
    """Class tokens an element is actually given, reading through a computed className.

    `_attr` answers the literal string `"{expr}"` for any `name={…}` (line 227), so
    every `className={cn(...)}` in the page has been counting as zero class tokens.
    For a LABEL that is right: `test_the_seven_label_spans_carry_a_wrapping_base_and_sm_truncate`
    reads the label's own classes literally, and a label whose wrapping base is
    computed rather than shipped is exactly what that test should redden. For the
    premise rule the question is the opposite one — not "did this class ship
    literally" but "does something beside the label refuse to shrink" — and a
    sibling written `className={cn('ml-auto flex-shrink-0 …', TONE_TEXT[tone])}`
    does refuse to shrink, because `cn` merges its literal arguments and every token
    inside a quoted literal reaches the element. The browser agrees: at a 320 px
    viewport on the live tree, the right-aligned span of a TaskLine row measures
    `cw=42 sw=42` (whole) while the label beside it is squeezed to `cw=0`.

    Only literals are harvested — an identifier such as `TONE_TEXT[tone]` resolves to
    something this file cannot see, so it is never guessed at. That means the harvest
    can only ADD a sibling the source really applies, never excuse a row that has
    none, and `test_premise_scan_reddens_when_no_sibling_holds_whole` is the seed
    that proves it. The row's own classes stay read literally by the caller: `flex`
    on the row is a declaration the contract requires in the text, not merely a
    runtime fact.
    """
    m = re.search(r"(?<![\w-])className\s*=\s*\{", attrs)
    if not m:
        return (_attr(attrs, "className") or "").split()
    depth = 0
    end = len(attrs)
    for j in range(m.end() - 1, len(attrs)):
        if attrs[j] == "{":
            depth += 1
        elif attrs[j] == "}":
            depth -= 1
            if depth == 0:
                end = j
                break
    literals = re.findall(r"'([^']*)'|\"([^\"]*)\"|`([^`]*)`", attrs[m.end():end])
    return [t for group in literals for t in " ".join(group).split()]


def _label_spans(masked: str, key: str):
    """The `<span>` elements whose entire content is `key`.

    "The row's only text" is what makes these seven spans the defect, so the
    definition is structural: an element, named span, whose opening tag is followed
    by nothing but the interpolation and its own close. (`end` is already past the
    tag's `>`, so the content to look for starts at the brace.)
    """
    needle = f"{key}</span>"
    out = []
    for start, name, attrs, ancestors in _elements(masked):
        if name != "span":
            continue
        end = _scan_tag(masked, start)[3]
        if masked[end:end + len(needle) + 6].lstrip().startswith(needle):
            out.append((start, attrs, ancestors, end))
    return out


# ── the audit: one function over a string, so it can be pointed at a broken copy ──

def audit_label_spans(source: str, table: list[tuple[str, int]]) -> list[str]:
    """Every way the spans in `source` fail the contract, as messages.

    A function over a string rather than assertions against the shipped file,
    because a pin nobody has watched FAIL is a pin nobody has tested: each mutation
    node below hands this the page's own bytes with one deliberate defect in them and
    checks that the defect is what comes back.
    """
    masked = _mask(source)
    problems: list[str] = []
    seen = 0
    for key, count in table:
        spans = _label_spans(masked, key)
        if len(spans) != count:
            problems.append(f"{key}: expected {count} label span(s), "
                            f"found {len(spans)}")
            continue
        seen += len(spans)
        for start, attrs, ancestors, _end in spans:
            cls = _attr(attrs, "className") or ""
            classes = cls.split()
            label = f"{key} [{cls}]"
            for need in WRAPPING_BASE:
                if need not in classes:
                    problems.append(f"{label}: no {need}")
            if VIEWPORT_ELLIPSIS not in classes:
                problems.append(f"{label}: no {VIEWPORT_ELLIPSIS}")
            if SHRINK_RELEASE not in classes:
                problems.append(f"{label}: no {SHRINK_RELEASE} — the row cannot "
                                "shrink the label below its longest token")
            if "truncate" in classes:
                problems.append(f"{label}: bare truncate still clips below sm")
            if _attr(attrs, "title") is not None:
                problems.append(f"{label}: title= is a tooltip, not a fix")
            for a_name, a_attrs, _at in list(ancestors)[-MAX_ANCESTOR_HOPS:]:
                if a_name in INTERACTIVE_TAGS:
                    problems.append(f"{label}: wrapped in a <{a_name}>, which is an "
                                    "affordance, not a wrap")
                    break
                if a_name.islower() and _attr(a_attrs, "title") is not None:
                    problems.append(f"{label}: a <{a_name}> ancestor carries title=, "
                                    "which is how a clipped label gets hidden")
                    break
    expected = sum(count for _key, count in table)
    if seen != expected:
        problems.append(f"label spans: expected {expected} in all, found {seen}")
    return problems


def _table() -> list[tuple[str, int]]:
    """The span table from the vitest contract, parsed — never retyped."""
    rows = _SPAN_ROW_RX.findall(CONTRACT.read_text(encoding="utf-8"))
    assert rows, (
        f"no `{{ key: …, count: … }}` rows parse out of {CONTRACT.name}, so this "
        "file has no span table to audit, and an empty audit is a pass")
    return [(k, int(c)) for k, c in rows]


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


# ── building mutants, from located offsets rather than typed literals ────────

def _located(source: str, key: str, index: int = 0):
    """The `index`th label span for `key`, as `(start, attrs, ancestors, end)`."""
    spans = _label_spans(_mask(source), key)
    assert index < len(spans), f"{key}: span {index} not found in the page"
    return spans[index]


def _edit_open_tag(source: str, key: str, fn, index: int = 0) -> str:
    """`source` with one located span's OPENING tag replaced by `fn(tag)`."""
    start, _attrs, _anc, end = _located(source, key, index)
    old = source[start:end]
    new = fn(old)
    assert new != old, f"mutation of {key} changed nothing, so the red below is fake"
    return source[:start] + new + source[end:]


def _wrap_span(source: str, key: str, tag: str, *, same_indent: bool, index: int = 0) -> str:
    """Wrap one located span in the element `tag` (name plus attributes), either one
    indent level out (how a person writes it) or at the span's own indent (how a
    sloppy edit lands)."""
    name = tag.split()[0]
    start, _attrs, _anc, end = _located(source, key, index)
    line_start = source.rfind("\n", 0, start) + 1
    indent = source[line_start:start]
    assert indent.strip() == "", "the located span does not start its own line"
    close = source.find("</span>", end)
    assert close >= 0, f"{key}: no close tag after the located span"
    close_end = source.find("\n", close)
    close_end = len(source) if close_end < 0 else close_end
    outer = indent if same_indent else indent[:-2]
    assert outer != indent or same_indent, f"cannot outdent {indent!r}"
    return (source[:line_start] + f"{outer}<{tag}>\n"
            + source[line_start:close_end]
            + f"\n{outer}</{name}>"
            + source[close_end:])


# ── clause 1: the contract on the shipped page ──────────────────────────────

def test_the_seven_label_spans_carry_a_wrapping_base_and_sm_truncate():
    """The page's row labels meet the contract: each of the seven carries
    `whitespace-normal` and `break-words` unconditionally, `sm:truncate` from the
    small breakpoint up, no bare `truncate`, and no `title`/control standing in for
    the wrap. The assertion is `== []`, so the message is the list of rows that
    failed and why."""
    problems = audit_label_spans(_page(), _table())
    assert problems == [], "dashboard row labels fail the #2202 contract: " + "; ".join(problems)


def test_the_span_table_counts_seven_across_two_files():
    """The denominator, taken from the vitest file's own table: six keys covering
    seven spans, because `{t.name}` renders in two rows (Failed, and Backlog
    Recently-touched). The audit counts per key as well as in total, which is what
    stops an eighth span arriving over a cited key and inheriting the pass.
    """
    table = _table()
    total = sum(count for _key, count in table)
    assert total == 7, f"the contract's span table counts {total}, not 7: {table}"
    assert len(table) == 6, f"expected six keys covering seven spans, got {table}"
    assert [k for k, c in table if c > 1] == ["{t.name}"], (
        f"exactly one key renders in two spans: {table}")
    assert audit_label_spans(_page(), table) == [], (
        "the table and the shipped page disagree, so the denominator is not the one "
        "the audit measured")


def test_the_two_halves_of_the_contract_agree():
    """Drift guard between the two files that pin this contract.

    If the frontend's own test stops requiring one of the three utilities, or stops
    forbidding one of the two substitutions, then pytest is pinning something the
    page no longer promises — and the pair has split. The span table is checked by
    `_table()` parsing this file's rows out of it, so a moved key or count reddens
    `test_the_span_table_counts_seven_across_two_files` instead.
    """
    ts = CONTRACT.read_text(encoding="utf-8")
    for need in (*WRAPPING_BASE, VIEWPORT_ELLIPSIS, SHRINK_RELEASE):
        assert f"classes.includes('{need}')" in ts, (
            f"the vitest contract no longer requires `{need}`, so pytest is stricter "
            f"than the frontend's own test: {CONTRACT.name}")
    for forbid in ("classes.includes('truncate')", "title= is a tooltip", "link/button"):
        assert forbid in ts, (
            f"the vitest contract no longer forbids {forbid!r}: {CONTRACT.name}")


def test_the_reader_locates_the_seven_rows():
    """Positive control on the reader, because every audit above is an empty-list
    assertion and an empty list from a broken reader reads exactly like a pass.

    What is asserted here is the reader's own claim: seven label spans, each a direct
    child of a `div` row. A comment read as a tag, or a close with no matching open,
    desynchronises the element stack and reddens this node rather than silencing the
    audit above — which is the failure mode `_mask` and the brace-depth attribute scan
    in `_scan_tag` exist to prevent.
    """
    masked = _mask(_page())
    parents = {key: [anc[-1][0] for _s, _a, anc, _e in _label_spans(masked, key)]
               for key, _count in _table()}
    assert sum(len(v) for v in parents.values()) == 7, (
        f"the reader located {sum(len(v) for v in parents.values())} label spans, "
        f"not 7: {parents}")
    assert all(parents), f"a cited key has no span at all: {parents}"
    bad = {k: v for k, v in parents.items() if any(p != "div" for p in v)}
    assert not bad, f"these labels are not direct children of a row div: {bad}"


# ── clause 2: the regression and both wrong fixes, each pinned by mutation ───

def test_the_contract_reddens_when_a_span_reverts_to_a_bare_truncate():
    """The regression this item exists because of. #1742's fix was written, review
    refused it, the round was abandoned, and all seven spans went back to one clipped
    line. Reverting one span's classes must come back naming that span.
    """
    reverted = _edit_open_tag(_page(), "{task.name}", lambda tag: tag.replace(
        "whitespace-normal break-words sm:truncate", "truncate"))
    joined = "; ".join(audit_label_spans(reverted, _table()))
    assert "{task.name}" in joined, f"the failure does not name the row: {joined}"
    assert "bare truncate" in joined, (
        f"a bare `truncate` on a row label was not the complaint: {joined}")
    for need in WRAPPING_BASE:
        assert f"no {need}" in joined, (
            f"dropping `{need}` alone went unreported, so the wrapping base is not "
            f"pinned: {joined}")


def test_the_contract_reddens_when_a_span_drops_only_its_viewport_ellipsis():
    """`whitespace-normal break-words` with no `sm:truncate` fixes the phone and
    re-wraps every desktop row — the other half of #1742's acceptance is that `>=sm`
    is untouched. Losing one utility has to be its own red, not a pass by adjacency
    to the two that remain.
    """
    dropped = _edit_open_tag(_page(), "{task.name}",
                             lambda tag: tag.replace(f" {VIEWPORT_ELLIPSIS}", ""))
    joined = "; ".join(audit_label_spans(dropped, _table()))
    assert f"no {VIEWPORT_ELLIPSIS}" in joined, (
        f"`sm:truncate` removed from a row label was not reported: {joined}")


def test_the_contract_reddens_when_a_label_drops_only_its_released_minimum():
    """The #2298 regression, and the one the three #1742 utilities cannot see.

    `whitespace-normal`, `break-words` and `sm:truncate` all stay exactly as shipped
    in this mutant, so a pin that counts those three keeps reporting green while the
    row goes back to being floored at the label's min-content — which is precisely
    how main got red at bca79161: a `Recent runs` row 262 px wide carried 277 px of
    content (summary `bench_027_recall_user_fact_topic_read:`), and the duration its
    `ml-auto flex-shrink-0` span holds whole landed at x=306 where the section's
    content box ends at 304. The complaint must name the row and must be the ONLY
    one, because the other three utilities are untouched here.
    """
    # Token-wise removal, because `min-w-0` is the FIRST class in the shipped
    # string: a `" min-w-0"` replace would match nothing and the mutant would be
    # the shipped file, which is how a mutation node ends up asserting nothing.
    floored = _edit_open_tag(
        _page(), "{task.name}",
        lambda tag: re.sub(
            r'className="([^"]*)"',
            lambda m: 'className="' + " ".join(
                c for c in m.group(1).split() if c != SHRINK_RELEASE) + '"',
            tag, count=1))
    joined = "; ".join(audit_label_spans(floored, _table()))
    assert f"no {SHRINK_RELEASE}" in joined, (
        f"`{SHRINK_RELEASE}` removed from a row label was not reported: {joined}")
    assert "{task.name}" in joined, f"the failure does not name the row: {joined}"
    for untouched in (*WRAPPING_BASE, VIEWPORT_ELLIPSIS):
        assert f"no {untouched}" not in joined, (
            f"dropping only `{SHRINK_RELEASE}` also complained about `{untouched}`, "
            f"so the red is not attributable: {joined}")
    assert "bare truncate" not in joined, (
        f"a label that still wraps was reported as clipped: {joined}")


def _row_premise(src: str) -> dict:
    """Per labelled row: its parent tag, the row's own class tokens, and the class
    tokens of every sibling that refuses to shrink.

    One measurement, read by both the rule and its seed, so the seed cannot pass by
    checking something the rule does not look at. The row's classes go through
    `_attr` (literal only — `flex` must be in the text); a sibling's go through
    `_class_tokens`, because what is asked of a sibling is a runtime fact about what
    the browser is told to hold whole.
    """
    masked = _mask(src)
    els = list(_elements(masked))
    premise = {}
    for key, _count in _table():
        for start, _attrs, ancestors, _end in _label_spans(masked, key):
            assert ancestors, f"{key}: a label span has no parent element at all"
            parent, p_attrs, p_lt = ancestors[-1]
            sibs = [_class_tokens(s_attrs)
                    for _s, _n, s_attrs, _anc in els
                    if _anc and _anc[-1][2] == p_lt and _s != start]
            premise[f"{key}@{start}"] = {
                "parent": parent,
                "row": (_attr(p_attrs, "className") or "").split(),
                "shrinkers": [s for s in sibs if "flex-shrink-0" in s],
            }
    return premise


def test_every_label_sits_in_a_flex_row_beside_a_value_that_stays_whole():
    """The premise that makes `min-w-0` load-bearing rather than decorative.

    `min-w-0` on a block in ordinary flow does nothing at all — `min-width: auto`
    already computes to zero there — so a pin that merely requires the class would
    still pass on a page where the declaration has no job. What gives it one is the
    row around it: a flex container, and a sibling that refuses to shrink (the
    right-aligned duration/count, `flex-shrink-0`), which is the element that gets
    pushed past the card's edge when the label cannot give ground. Read out of the
    element tree, so a row that stops being flex or loses its right-aligned value
    reddens HERE, naming the row, instead of leaving `min-w-0` in place as a class
    that pins an absent mechanism.

    #2398 moved the `TaskLine` row's evidence from its note span to its two
    `cn(...)` spans, which is why the sibling read goes through `_class_tokens`;
    `test_premise_scan_reddens_when_no_sibling_holds_whole` is the seed.
    """
    premise = _row_premise(_page())
    assert len(premise) == 7, f"expected the seven labels, got {len(premise)}"
    not_flex = {k: v["row"] for k, v in premise.items()
                if v["parent"] != "div" or "flex" not in v["row"]}
    assert not not_flex, (
        f"these labels are not in a flex row, so `{SHRINK_RELEASE}` has nothing to "
        f"release there and the rule needs re-deriving, not deleting: {not_flex}")
    no_pusher = {k: v["row"] for k, v in premise.items() if not v["shrinkers"]}
    assert not no_pusher, (
        "these rows have no `flex-shrink-0` sibling to hold its width, so the "
        f"#2298 push-out cannot happen in them and the rule is over-broad: {no_pusher}")


def test_premise_scan_reddens_when_no_sibling_holds_whole():
    """The seed that keeps #2398's `cn(...)` read from becoming a free pass.

    `_class_tokens` was added so the premise rule can see a `TaskLine` row's
    right-aligned span, whose `flex-shrink-0` reaches the element through
    `cn('ml-auto flex-shrink-0 font-mono tabular-nums', TONE_TEXT[tone])`. A read that
    can never fail is not a check, so here both of the row's computed siblings are
    stripped of the token and the SAME measurement must report the row. This is also
    the only pin for #2398's mechanism itself: the shipped page now has NO literal
    `flex-shrink-0` anywhere in `TaskLine`, so a `min-w-0` released on the label and
    the note would be pure decoration if this row quietly lost its whole sibling.

    `count=1` on each strip because both literals are the FIRST occurrence of their
    text in the file, and a seed that does not land asserts nothing — the same trap
    `test_the_contract_reddens_when_a_label_drops_only_its_released_minimum` names for
    `min-w-0`. Both strips are checked before the measurement is read.
    """
    shipped = _page()
    seeded = shipped.replace("cn('ml-auto flex-shrink-0 font-mono tabular-nums',",
                             "cn('ml-auto font-mono tabular-nums',", 1)
    seeded = seeded.replace("cn('h-3 w-3 flex-shrink-0',", "cn('h-3 w-3',", 1)
    assert seeded != shipped, "neither seed landed: the seed is the shipped file"
    for probe in ("cn('ml-auto font-mono tabular-nums',", "cn('h-3 w-3',"):
        assert probe in seeded, f"seed did not land: {probe!r} is not in the mutant"
    # A literal token elsewhere in the file is not the sibling this seed is about,
    # so the precondition is scoped to TaskLine's own row, bounded by the label it
    # holds rather than by a pattern that could match an earlier row: every `<span>`
    # in that row must have lost the token once the two strips land.
    at = seeded.index("{task.name}</span>")
    row = seeded[seeded.rindex('<div className="flex items-center gap-2', 0, at):
                 seeded.index("</div>", at)]
    assert not [c for c in re.findall(r'<span className="([^"]*)"', row)
                if "flex-shrink-0" in c.split()], (
        "a literal `flex-shrink-0` is still on a span inside the seeded row, so this "
        "seed would be testing the shipped page and not the computed siblings")

    premise = _row_premise(seeded)
    bare = {k for k, v in premise.items() if not v["shrinkers"]}
    assert any(k.startswith("{task.name}@") for k in bare), (
        f"stripping `flex-shrink-0` out of both of TaskLine's computed siblings left "
        f"its row looking whole, so the read is a free pass: {sorted(bare)}")


def test_the_contract_reddens_when_a_label_trades_its_wrap_for_a_title():
    """Wrong fix one: keep the label clipped and hang a tooltip on it.

    A `title` is a hover affordance and a phone has no hover — #1742 was refused once
    for exactly this substitution. It is worse than merely useless here: the measured
    instrument's `unreachable` predicate EXEMPTS a label with a `title` on itself or
    an ancestor, so a `title=` added to one of these rows would blind
    `tests/test_dashboard_responsive.py`'s seeded pin along with this one.
    """
    titled = _edit_open_tag(_page(), "{task.name}",
                            lambda tag: tag.replace("<span ", '<span title={task.name} '))
    joined = "; ".join(audit_label_spans(titled, _table()))
    assert "title= is a tooltip, not a fix" in joined, (
        f"a label gained a title= and the audit did not notice: {joined}")


def test_the_contract_reddens_when_a_title_appears_on_the_row_around_a_label():
    """The same exemption one element out: the span stays clean, its row div gains a
    `title`, and that is enough for the clipped text to count as "reachable" to the
    probe while nothing on a phone can invoke it. Found through the element tree, so
    it does not depend on where the attribute was typed.
    """
    start, _attrs, ancestors, _end = _located(_page(), "{task.name}")
    assert ancestors and ancestors[-1][0] == "div", (
        f"the row is not a div any more ({ancestors[-1][0]!r}), so this mutation is "
        "no longer testing what clause 2 describes")
    row_start = ancestors[-1][2]
    row_open = source_open = _page()[row_start:_scan_tag(_mask(_page()), row_start)[3]]
    patched = row_open.replace("<div ", '<div title={task.command} ', 1)
    assert patched != source_open, "could not add a title to the row"
    titled_row = _page()[:row_start] + patched + _page()[len(source_open) + row_start:]
    joined = "; ".join(audit_label_spans(titled_row, _table()))
    assert "carries title=" in joined, (
        f"the row around a label gained a tooltip and nothing was reported: {joined}")


def test_the_contract_reddens_when_a_label_is_wrapped_in_a_button():
    """Wrong fix two: make the row a control so the clipped text is "reachable" by
    tapping through.

    That is a different affordance from every other row on the page, and — like the
    `title` above — the probe's predicate exempts a `button` ancestor, so it would
    turn the measured pin green while the label stayed clipped. Written the way a
    person writes a wrapper: the `<button>` one indent level out.
    """
    wrapped = _wrap_span(_page(), "{task.name}", "button onClick={() => {}}",
                         same_indent=False)
    joined = "; ".join(audit_label_spans(wrapped, _table()))
    assert "wrapped in a <button>" in joined, (
        f"a <button> now contains the label and the audit did not notice: {joined}")


def test_the_wrapper_is_found_even_written_at_its_child_s_indent():
    """The case an indentation heuristic cannot see, pinned.

    The vitest audit proves nesting by indentation and states in its own comment that
    a wrapper written at its child's indent is deliberately not claimed. This audit
    reads the element tree instead, so the same wrapper written FLAT — `<a>` at the
    span's own indent, which is what a quick edit produces — is still a wrapper. That
    is the difference between the two files, and it is the review's finding on round
    SM_20261005_002918 answered in the implementation rather than in prose.
    """
    flat = _wrap_span(_page(), "{r.source}", "a href='#/workers'", same_indent=True)
    assert flat.count("href='#/workers'") == 1, "the mutant was not built as intended"
    joined = "; ".join(audit_label_spans(flat, _table()))
    assert "wrapped in a <a>" in joined, (
        f"a link written at its child's indent hid the label and nothing was "
        f"reported: {joined}")


def test_the_contract_reddens_when_an_eighth_span_appears_over_a_cited_key():
    """The denominator. A new row that interpolates `{src.name}` with the OLD
    clipping class must not inherit the pass the existing spans earned — and an
    eighth span in any shape must not quietly widen the contract.
    """
    start, _attrs, _anc, _end = _located(_page(), "{r.source}")
    line_start = _page().rfind("\n", 0, start) + 1
    indent = _page()[line_start:start]
    extra = (_page()[:line_start]
             + f"{indent}<span className=\"truncate text-muted-foreground\">"
               "{src.name}</span>\n"
             + _page()[line_start:])
    joined = "; ".join(audit_label_spans(extra, _table()))
    assert "{src.name}: expected 1 label span(s), found 2" in joined, (
        f"a second span over a cited key was not counted: {joined}")
