"""#4108: the Background > Sources rollup must say what span it rests on, at PANEL
level, and the pins that say so must RED on the wrong fixes.

Two files pin one contract, and each does the half the other cannot:

  * `web/src/components/pages/BackgroundPage.test.ts` is the contract the frontend's
    own suite enforces — it runs the helper, renders the note with
    `renderToStaticMarkup`, and stubs `fetch`. That is where a developer editing the
    page finds out, and the automod gate runs it (`gate.py::_vitest_run`).
  * THIS file is the same contract as pytest nodes, and the only side that can open
    the ROUTE: vitest stubs `fetch`, so the field names the server actually emits
    have never been compared with the field names the client declares. That
    comparison IS the defect — `app/routers/workers.py` published `oldest_input` and
    `window_clamped_to_hours` (#2127) and `web/src/api.ts` typed them away — and no
    test living in either half alone can see the seam between them.

The review rung's node rail accepts a node id only in a test file under one of
`pytest.ini`'s testpaths — `scripts/automod/testpaths.py::is_test_file` requires a
`.py` — so a clause pinned only in a `.test.ts` is refused as `test_node_id not in a
test file this diff changed` however well it was verified. That is how #2202's round
SM_20261005_002918 lost clauses 1 and 2 while its own review text said their
substance was met, and it is why #4108 does not pin its clauses only in TypeScript.
`test_the_two_halves_of_the_contract_agree` keeps the two files from drifting: the
inputs and the expected phrases are PARSED out of the vitest file and re-derived
here from the helper's rules, so neither side can quietly change the contract.

Run it: `pytest tests/test_sources_panel_window_clamp.py`. No browser, no vite, no
`node_modules`: nothing here renders a pixel. The measured half — the live route at
`days=7` printing no phrase, and a short store printing one — is the post-landing
owed check on #4108, which needs real `workers.db` history and a browser.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
API = REPO / "web" / "src" / "api.ts"
PAGE = REPO / "web" / "src" / "components" / "pages" / "BackgroundPage.tsx"
NOTE = REPO / "web" / "src" / "components" / "pages" / "WorkersWindowNote.tsx"
ROW = REPO / "web" / "src" / "components" / "pages" / "SourceHealthRow.tsx"
CONTRACT = REPO / "web" / "src" / "components" / "pages" / "BackgroundPage.test.ts"
ROUTER = REPO / "app" / "routers" / "workers.py"

#: The pair, by the names the ROUTE emits them (`_health_window_fields`' keys). The
#: client's declared keys have to be these character for character, or the panel
#: reads `undefined` off a response that carried a number.
PAIR = ("oldest_input", "window_clamped_to_hours")

#: The leaf's own prop names, which are NOT the same words: the component is shared
#: furniture and says `clampHours`, where the wire format says
#: `window_clamped_to_hours`. `test_the_panel_threads_…` is what proves the two
#: vocabularies are connected rather than merely similar.
NOTE_PROPS = {"clampHours": "healthWindow.window_clamped_to_hours",
              "oldestInput": "healthWindow.oldest_input",
              "days": "healthWindow.days"}

#: The note element as the panel writes it, and the card map it must sit above.
NOTE_JSX = ('      <WorkersWindowNote\n'
            '        clampHours={healthWindow.window_clamped_to_hours}\n'
            '        oldestInput={healthWindow.oldest_input}\n'
            '        days={healthWindow.days}\n'
            '      />\n')
CARD_MAP = '      {sources.map(s => <SourceCard key={s.name} source={s} onOpen={onOpen} />)}'
CARD_SUMMARY_ROW = ('      <div className="flex flex-wrap gap-x-4 gap-y-1 '
                    'px-3 py-2 text-[10px] text-muted-foreground">')

#: What the route answers on this box, measured for #2127 and again on 2026-11-12:
#: `days=90` clamped to 738.3 h of the 2160 h requested.
LIVE_CLAMP_HOURS = 738.3
OLDEST = '2026-10-13T06:01:10.413329+00:00'


# ── the reader ──────────────────────────────────────────────────────────────

def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _mask(src: str) -> str:
    """`src` with every comment and string literal replaced by spaces of the SAME
    length.

    This tree's prose names the very things a scanner hunts for — "not inside
    SourceCard", `window_clamped_to_hours stays undefined`, a `<WorkersWindowNote`
    quoted inside a JSX comment — so a reader that took those as code would report a
    threading that exists only in a sentence. Offsets are preserved, so a position in
    the mask is the same position in the original and a mutation can be spliced into
    the real bytes.
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


def _balanced(src: str, open_at: int, o: str = "{", c: str = "}") -> int:
    """Index just past the bracket that closes the one at `open_at`."""
    depth = 0
    for k in range(open_at, len(src)):
        if src[k] == o:
            depth += 1
        elif src[k] == c:
            depth -= 1
            if depth == 0:
                return k + 1
    raise AssertionError(f"unbalanced {o}…{c} from offset {open_at}")


def _region(src: str, header: str) -> tuple[int, int]:
    """`(start, end)` over `function <header>` and its brace-matched body.

    Locating a construct by its declaration is what makes the region a fact: a name
    found anywhere in the file proves nothing about WHERE it acts, and where the
    clamp is read from and where the note renders is the whole question here.
    """
    m = re.search(rf"\bfunction\s+{re.escape(header)}\s*\(", src)
    if not m:
        raise AssertionError(f"no `function {header}(` — this reader is stale")
    params_end = _balanced(src, src.index("(", m.end() - 1), "(", ")")
    return m.start(), _balanced(src, src.index("{", params_end))


def _jsx_attrs(src: str, start: int) -> dict[str, str]:
    """The attributes of the JSX element opening at `start`, as `name -> value`.

    Brace-matched, because a prop value is `{healthWindow.window_clamped_to_hours}`
    and a regex that stops at the first space or brace reads the interesting half of
    every expression prop as `{healthWindow`.
    """
    gt = src.index(">", start)
    head = src[src.index(" ", start):gt]
    if head.rstrip().endswith("/"):
        head = head.rstrip()[:-1]
    attrs: dict[str, str] = {}
    i, n = 0, len(head)
    while i < n:
        m = re.compile(r"[A-Za-z_$][\w$]*").match(head, i)
        if not m:
            i += 1
            continue
        name, j = m.group(0), m.end()
        while j < n and head[j].isspace():
            j += 1
        if j < n and head[j] == "=":
            j += 1
            while j < n and head[j].isspace():
                j += 1
            if head.startswith("{", j):
                close = _balanced(head, j)
                attrs[name], i = head[j:close], close
            else:
                k = re.compile(r'"[^"]*"|\'[^\']*|[\w./-]+').match(head, j)
                attrs[name], i = (k.group(0) if k else ""), (k.end() if k else j)
        else:
            attrs[name], i = "", m.end()
    return attrs


def _note_in(body: str) -> tuple[int, dict[str, str]]:
    """`(offset, attrs)` of the one `<WorkersWindowNote …>` inside `body`."""
    masked = _mask(body)
    hits = [m.start() for m in re.finditer(r"<WorkersWindowNote[\s/>]", masked)]
    if len(hits) != 1:
        raise AssertionError(
            f"expected exactly one <WorkersWindowNote …> in the masked source, "
            f"found {len(hits)}")
    return hits[0], _jsx_attrs(masked, hits[0])


def _panel_body() -> str:
    src = _read(PAGE)
    start, end = _region(src, "SourcesPanel")
    return src[start:end]


def _client_type_block() -> str:
    """The `workersHealth` member of the api client, declaration through body."""
    src = _read(API)
    masked = _mask(src)
    m = re.search(r"\bworkersHealth\s*:", masked)
    if not m:
        raise AssertionError("no `workersHealth:` in web/src/api.ts")
    end = masked.find("\n\n", m.start())
    return src[m.start():len(src) if end < 0 else end]


def _contract_pairs() -> list[tuple[tuple, str, bool]]:
    """Every `(args, expected, exact)` pair the vitest contract asserts.

    Parsed from the `.test.ts` rather than retyped, so the two files cannot drift
    apart: a phrase or an input that moves on one side reddens the other.
    """
    src = _read(CONTRACT)
    out = []
    for m in re.finditer(
            r"windowSpanText\(([^)]*)\)\s*\)+\s*\.(toBe|toContain)\(\s*'([^']*)'\s*\)",
            src, re.S):
        args = []
        for raw in m.group(1).split(","):
            tok = raw.strip().rstrip(",").strip()
            if not tok:
                continue
            if tok in ("null", "undefined"):
                args.append(None)
            elif tok.startswith("'"):
                args.append(tok.strip("'"))
            else:
                args.append(float(tok))
        while len(args) < 4:
            args.append(None)
        out.append((tuple(args), m.group(3), m.group(2) == "toBe"))
    return out


def _hours_text(clamp_hours) -> str:
    """The tenth-rounded figure as JS prints it: `String(11.0)` is `"11"`, `257.8`
    keeps its decimal, `0` stays a measurement and not an absence."""
    hours = round(float(clamp_hours) * 10) / 10
    return str(int(hours)) if float(hours).is_integer() else f"{hours:g}"


def _utc_day(stamp) -> str:
    """The UTC date half of an ISO stamp; `''` for one that does not parse."""
    try:
        return dt.datetime.fromisoformat(str(stamp)) \
            .astimezone(dt.timezone.utc).date().isoformat()
    except ValueError:
        return ""


def _phrase(clamp_hours, days, oldest_input, subject=None) -> str:
    """The Python half of the dual implementation — `windowSpanText`'s rules.

    Deliberately written from the helper's stated rules rather than read off its
    bytes: a wording that moves in the TypeScript without the contract moving is a
    contract nobody re-read, and only a second implementation notices.
    """
    if clamp_hours is None:
        return ""
    oldest = _utc_day(oldest_input) if oldest_input else ""
    subject = "Sources rollup" if subject is None else subject
    base = f"{subject} covers {_hours_text(clamp_hours)}h of the {int(days)}d window"
    return f"{base} (oldest input {oldest})" if oldest else base


def _ts_render(clamp_hours, days, oldest_input, subject=None) -> str:
    """The helper's OWN templates, rendered — the other half of the same comparison.

    `_phrase` is blind to a change in `WorkersWindowNote.tsx`: it is a second
    implementation, and a reworded helper leaves it and the `.test.ts` string both
    untouched, agreeing with each other while the shipped code says something else.
    This renders the template literals parsed out of that file, so a rewording or a
    dropped `${dayLabel(days)}` interpolations goes red here even with neither of the
    other two files edited.
    """
    src = _read(NOTE)
    day = re.search(r"function dayLabel\(days: number\): string \{\s*return `([^`]*)`",
                    src, re.S)
    base = re.search(r"const base = `([^`]*)`", src)
    with_oldest = re.search(r"return oldest \? `([^`]*)` : base", src)
    # #4315: the subject is a parameter now, and its DEFAULT is read off the
    # signature, so the Sources wording is still pinned by the helper's own bytes.
    default = re.search(r"subject: string = '([^']*)'", src)
    assert day and base and with_oldest and default, (
        "the helper's phrase is no longer (day label) + (base with hours and window) + "
        "(oldest input): the contract has moved and this reader is stale")
    if clamp_hours is None:
        return ""
    text = base.group(1).replace(
        "${subject}", default.group(1) if subject is None else subject).replace(
        "${hours}", _hours_text(clamp_hours)).replace(
        "${dayLabel(days)}", day.group(1).replace("${days}", str(int(days))))
    oldest = _utc_day(oldest_input) if oldest_input else ""
    return with_oldest.group(1).replace("${base}", text) \
        .replace("${oldest}", oldest).replace("${dayLabel(oldestInput)}", oldest) \
        if oldest else text


# ── clause 1: the route's pair reaches a client that declares it ────────────

def test_the_route_emits_the_pair_the_workers_client_declares():
    """The server's key names and the client's declared keys, in one assertion.

    `_health_window_fields` is the single emitter behind both health routes (#1401 →
    #2127), so its keys ARE the wire format. The panel can only state a span if
    `workersHealth` keeps those two names, and the defect was that it kept neither.
    """
    from app.autonomy import _health_window_fields

    now = dt.datetime(2026, 11, 12, 20, 0, tzinfo=dt.timezone.utc)
    clamped = _health_window_fields(now, days=90, oldest_input=OLDEST, observed=True)
    assert set(clamped) == set(PAIR), clamped
    assert clamped["window_clamped_to_hours"] < 90 * 24
    assert clamped["oldest_input"] == OLDEST
    # A store that DOES cover the window answers with two nulls — today's live
    # `days=7` shape, and the payload the panel has to render silently.
    covering = _health_window_fields(now - dt.timedelta(days=1), days=7,
                                     oldest_input=OLDEST, observed=True)
    assert covering == {"oldest_input": None, "window_clamped_to_hours": None}
    # And a window with no row reaching the tallies is a clamp of ZERO hours, not a
    # missing field: the payload a truthiness test renders as silence.
    unobserved = _health_window_fields(now, days=7, oldest_input=OLDEST, observed=False)
    assert unobserved["window_clamped_to_hours"] == 0.0

    declared = _client_type_block()
    for key, tname in zip(PAIR, ("string", "number")):
        assert re.search(rf"\b{key}\?:\s*{tname}\s*\|\s*null", declared), (
            f"`{key}` is not declared optional AND nullable on `workersHealth` — an "
            "absence has to be a legal value, or the type claims a figure today's "
            "route does not send")
    assert "**window_fields" in _read(ROUTER), (
        "app/routers/workers.py no longer splices `**window_fields` into the health "
        "payload — the client would be declaring fields the server stopped sending")


# ── clause 4: threaded from the response, and at panel level ────────────────

def test_the_panel_threads_the_payloads_own_clamp_into_the_note():
    """Every prop of the note comes from response-held state, read brace-aware.

    A hard-wired `clampHours={undefined}`, a literal `7` for the window, or the note
    rendered below the cards would each keep the phrase and lose the meaning: the
    first is a shape copied from today's payload, the second arithmetic instead of
    measurement, the third a claim attached to the wrong half of the screen.
    """
    body = _panel_body()
    offset, attrs = _note_in(body)
    for name, want in NOTE_PROPS.items():
        got = attrs.get(name)
        assert got is not None, f"<WorkersWindowNote> lost its `{name}` prop"
        assert got.strip("{} ").replace("\n", " ") == want, (
            f"`{name}={got}` is not `{want}` — the note must print what the route "
            "measured, not what the panel assumed")
    # The state it reads is written from the response, which is kept whole rather
    # than reduced to `sources` as it was before #4108.
    masked = _mask(body)
    assert re.search(r"setHealthWindow\(\{[^}]*window_clamped_to_hours:\s*out\."
                     r"window_clamped_to_hours", masked, re.S), (
        "the clamp is not stored from the response (`out.window_clamped_to_hours`)")
    assert re.search(r"oldest_input:\s*out\.oldest_input", masked), (
        "`oldest_input` is not stored from the response")
    assert re.search(r"days:\s*out\.days", masked), (
        "the requested window is not taken from the response's echo either")
    # Panel level: above the card map, inside SourcesPanel's own return.
    assert offset < masked.index("sources.map("), (
        "the note renders below the cards, not at panel level")


def test_the_note_never_appears_inside_a_source_card():
    """The pair is a FLEET bound, so no per-source card may print it.

    `oldest_input` is `MIN(completed_at)` over EVERY source (`app/routers/workers.py`,
    #2127). A card that printed it would attribute one store-wide measurement to one
    source — and lie about any source whose own history starts later than the
    store's.
    """
    src = _read(PAGE)
    start, end = _region(src, "SourceCard")
    card = _mask(src[start:end])
    assert "<WorkersWindowNote" not in card, "the note moved inside SourceCard"
    assert "window_clamped_to_hours" not in card, (
        "SourceCard reads the fleet clamp — a fleet measurement on a per-source row")
    assert "<WorkersWindowNote" not in _mask(_read(ROW)), (
        "the clamp phrase moved into SourceHealthRow, which is one source's row")


# ── clauses 2 and 3: the wording, and silence when there is nothing to say ──

def test_the_helper_is_pure_and_answers_null_rather_than_falsy():
    """The wording lives in an exported pure function, and its silence is a null test.

    `== null` covers both `null` and `undefined` (a backend older than #2127 sends
    neither key) while still printing `0.0`, which `_health_window_fields` emits for
    a window with no row reaching the tallies. A truthiness test folds that payload
    into the same face as the clean bill of health — the report resting on the least
    evidence says nothing about itself.
    """
    src = _read(NOTE)
    assert re.search(r"export function windowSpanText\(", src), (
        "the wording is no longer an exported pure helper — a phrase computed inside "
        "JSX cannot be pinned by an input/output pair in a node test environment")
    assert re.search(r"if \(clampHours == null\) return ''", src), (
        "the null arm is no longer a null test returning the empty string")
    assert "Math.round(clampHours * 10) / 10" in src, (
        "the hours are no longer the payload's own figure rounded to a tenth, which "
        "is the only reading that keeps `11.04` printing as `11h`")


def test_the_two_halves_of_the_contract_agree():
    """The vitest contract's inputs, run through the Python half of the same rules.

    Neither file is taken on trust for what the other asserts: the inputs and the
    expected strings are parsed out of `BackgroundPage.test.ts`, and each is rebuilt
    here. A helper whose wording or rounding changes reddens this node even though
    the vitest file was not touched, and a vitest file that stops pinning the clamp
    reddens it for having nothing to compare.
    """
    pairs = _contract_pairs()
    assert len(pairs) >= 6, (
        f"the vitest contract pins only {len(pairs)} helper cases — a contract this "
        f"thin cannot hold #4108's clauses: {CONTRACT.name}")
    for args, expected, exact in pairs:
        mirror, shipped = _phrase(*args), _ts_render(*args)
        assert mirror == shipped, (
            f"the Python mirror and the helper's own templates disagree on "
            f"windowSpanText{args}: {mirror!r} vs {shipped!r} — one of the two files "
            f"was edited without the other")
        if exact:
            assert shipped == expected, (
                f"windowSpanText{args} renders {shipped!r}, the contract asserts "
                f"{expected!r}")
        else:
            assert expected in shipped, f"windowSpanText{args} lacks {expected!r}"
    # The two readings the item measured, present on both sides rather than one.
    assert any(a[0] == 257.81 and e.startswith("Sources rollup covers 257.8h of the 7d")
               for a, e, _ in pairs), "no clamped-payload case pins the phrase"
    assert any(a[0] is None and e == "" for a, e, _ in pairs), (
        "nothing pins silence for the null-clamp payload today's route returns")


# ── the positive control and the battery ────────────────────────────────────

class Broken(RuntimeError):
    """Raised when a mutation has been applied to the wrong bytes, not to the code."""


def test_the_reader_locates_the_note_the_pair_and_the_contract_nodes():
    """The positive control: every construct this file reasons about is really there.

    Each node above passes on an EMPTY string — a rename leaves a regex that finds
    nothing and asserts nothing — so a rename would silently retire the guard. This
    node cannot pass on absence: it locates the note and exactly its three props,
    both fields in the client type, the splice in the route, and three of the vitest
    nodes by name.
    """
    _, attrs = _note_in(_panel_body())
    assert set(attrs) == set(NOTE_PROPS), attrs
    declared = _client_type_block()
    assert "workersHealth" in declared and all(k in declared for k in PAIR)
    assert "**window_fields" in _read(ROUTER)
    src = _read(CONTRACT)
    for node in ("the client keeps the clamp pair the route sends instead of dropping it",
                 "a payload that covers its window prints no span claim at all",
                 "the panel renders the phrase at panel level, never inside a source card"):
        assert f"it('{node}'" in src, f"the vitest contract no longer defines `{node}`"


def _apply(src: str, needle: str, repl: str) -> str:
    if src.count(needle) != 1:
        raise Broken(f"needle found {src.count(needle)}x, expected exactly 1: "
                     f"{needle[:70]!r}")
    return src.replace(needle, repl)


def _move_note_into_card(s: str) -> str:
    """The plausible wrong fix: the phrase printed as one source's measurement."""
    s = _apply(s, NOTE_JSX, "")
    return _apply(s, CARD_SUMMARY_ROW,
                  '      <WorkersWindowNote clampHours={healthWindow'
                  '.window_clamped_to_hours} oldestInput={healthWindow.oldest_input} '
                  'days={healthWindow.days} />\n' + CARD_SUMMARY_ROW)


def _move_note_below_the_cards(s: str) -> str:
    """Still in `SourcesPanel`, so any "the file contains the element" check passes."""
    s = _apply(s, NOTE_JSX, "")
    return _apply(s, CARD_MAP, CARD_MAP + "\n" + NOTE_JSX)


@pytest.mark.parametrize("path,mutate,expect_fail", [
    # clause 1 — the client drops the pair again: the defect itself.
    (API, lambda s: _apply(s, "    oldest_input?: string | null\n"
                              "    window_clamped_to_hours?: number | null\n", ""),
     test_the_route_emits_the_pair_the_workers_client_declares),
    # clause 1 — required rather than optional/nullable: a field today's payload
    # does not send, and a panel that cannot tell absence from zero.
    (API, lambda s: _apply(s, "    oldest_input?: string | null\n"
                              "    window_clamped_to_hours?: number | null\n"
                              "    sources: WorkerSourceHealth[]",
                           "    oldest_input: string\n"
                           "    window_clamped_to_hours: number\n"
                           "    sources: WorkerSourceHealth[]"),
     test_the_route_emits_the_pair_the_workers_client_declares),
    # clause 1 — the route stops publishing the pair, so the declaration is a
    # promise about a field nobody sends. Caught from the SERVER side, which no
    # frontend test can reach at all.
    (ROUTER, lambda s: _apply(s, "**window_fields, ", ""),
     test_the_route_emits_the_pair_the_workers_client_declares),
    # clause 4 — the panel keeps only `sources`, exactly as it did before #4108.
    (PAGE, lambda s: _apply(s, "        window_clamped_to_hours: out.window_clamped_to_hours,\n"
                               "        oldest_input: out.oldest_input,\n", ""),
     test_the_panel_threads_the_payloads_own_clamp_into_the_note),
    # clause 4 — the prop hard-wired to the shape of today's payload: a silently
    # empty note forever, and every text-only check still sees the threading.
    (PAGE, lambda s: _apply(s, "clampHours={healthWindow.window_clamped_to_hours}",
                            "clampHours={undefined}"),
     test_the_panel_threads_the_payloads_own_clamp_into_the_note),
    # clause 4 — the window re-derived from the request instead of read off the
    # response, so the phrase's second number is arithmetic, not measurement.
    (PAGE, lambda s: _apply(s, "        days: out.days ?? 7,", "        days: 7,"),
     test_the_panel_threads_the_payloads_own_clamp_into_the_note),
    # clause 4 — the note moved below the card map, still inside SourcesPanel.
    (PAGE, _move_note_below_the_cards,
     test_the_panel_threads_the_payloads_own_clamp_into_the_note),
    # clause 4 — the note moved into a per-source card: fleet bound, per-source label.
    (PAGE, _move_note_into_card,
     test_the_note_never_appears_inside_a_source_card),
    # clauses 2/3 — falsy in place of null: `0.0`, the no-data clamp, goes quiet.
    (NOTE, lambda s: _apply(s, "if (clampHours == null) return ''",
                            "if (!clampHours) return ''"),
     test_the_helper_is_pure_and_answers_null_rather_than_falsy),
    # clauses 2/3 — the wording keeps the hours and drops the window they are short of.
    (NOTE, lambda s: _apply(s, " of the ${dayLabel(days)} window", ""),
     test_the_two_halves_of_the_contract_agree),
    # clauses 2/3 — the prose moves, and the two files would split.
    (NOTE, lambda s: _apply(s, "subject: string = 'Sources rollup'",
                            "subject: string = 'Rollup'"),
     test_the_two_halves_of_the_contract_agree),
    # clauses 2/3 — `toFixed` instead of a tenth: `11.04` starts printing `11.0h`.
    # The executed half of this is the vitest node that asserts `11h`; what the
    # pytest side can hold is the rule itself, pinned in the helper's bytes.
    (NOTE, lambda s: _apply(s, "const hours = Math.round(clampHours * 10) / 10",
                            "const hours = Number(clampHours.toFixed(1))"),
     test_the_helper_is_pure_and_answers_null_rather_than_falsy),
])
def test_the_mirror_becomes_red_when_the_pair_stops_reaching_the_panel(
        path, mutate, expect_fail):
    """Each wrong fix above reddens exactly the node named beside it.

    The wrong fixes are the ones that look right: a prop hard-wired to today's
    payload, a truthiness test that covers the null and the zero in one stroke, a
    note that prints the true phrase but attaches it to one source, a client type
    that names both fields without saying they may be absent, and a route that
    quietly stops sending them.
    """
    original = _read(path)
    try:
        path.write_text(mutate(original), encoding="utf-8")
        with pytest.raises((AssertionError, Broken)):
            expect_fail()
    finally:
        path.write_text(original, encoding="utf-8")
    assert _read(path) == original, f"{path.name} was not restored after its mutation"


# ── #4315: the Autonomy fleet strip borrows the note ─────────────────────────

AUTONOMY_PAGE = REPO / "web" / "src" / "components" / "pages" / "AutonomyPage.tsx"


def _autonomy_fleet_block() -> str:
    """The `fleet: { … }` member of `export interface AutonomyHealth`, brace-aware."""
    src = _read(API)
    masked = _mask(src)
    head = masked.index("export interface AutonomyHealth {")
    at = masked.index("fleet:", head)
    open_at = masked.index("{", at)
    return src[open_at:_balanced(masked, open_at)]


def test_the_autonomy_client_declares_the_keys_the_route_emits_into_fleet():
    """#4315 clause 4: `app.autonomy.compute_health` splices `_health_window_fields`
    into its `fleet` block, and the Autonomy client must declare exactly those keys —
    character for character — and the fleet strip must read them by those names. A
    rename on the route moves the emitted set and reddens this node; a rename on the
    client alone leaves a key undeclared and reddens it the same way."""
    from app.autonomy import _health_window_fields

    now = dt.datetime(2026, 11, 19, 7, 16, tzinfo=dt.timezone.utc)
    emitted = set(_health_window_fields(now, days=90, oldest_input=OLDEST, observed=True))
    assert emitted == set(_health_window_fields(now, days=7, oldest_input=None,
                                                observed=True))
    fleet = _autonomy_fleet_block()
    assert emitted == set(PAIR), (
        f"_health_window_fields now emits {sorted(emitted)}, not {sorted(PAIR)}")
    for key, tname in zip(PAIR, ("string", "number")):
        assert re.search(rf"^\s*{key}\?:\s*{tname}\s*\|\s*null;", fleet, re.M), (
            f"AutonomyHealth.fleet does not declare `{key}?: {tname} | null`:\n{fleet}")
    assert re.search(r'"fleet":\s*\{[\s\S]*?\*\*_health_window_fields\(',
                     _read(REPO / "app" / "autonomy.py")), (
        "compute_health no longer splices `_health_window_fields` into `fleet`")
    page = _read(AUTONOMY_PAGE)
    start, end = _region(page, "HealthStrip")
    strip = _mask(page[start:end])
    for key in sorted(emitted):
        assert f"f.{key}" in strip, f"HealthStrip does not read `f.{key}` off the payload"


def test_the_sources_half_is_untouched_by_the_fleet_note():
    """#4315 clause 5: the Background page still renders exactly ONE note, above the
    source cards, with no subject of its own — so its phrase is still the default
    `Sources rollup covers …` — and no source card renders either clamp key."""
    page = _mask(_read(PAGE))
    assert page.count("<WorkersWindowNote") == 1, page.count("<WorkersWindowNote")
    body = _panel_body()
    offset, attrs = _note_in(body)
    assert "subject" not in attrs, attrs
    assert offset < _mask(body).index("sources.map("), "the note left panel level"
    assert _phrase(257.81, 7, OLDEST) == _ts_render(257.81, 7, OLDEST) == (
        "Sources rollup covers 257.8h of the 7d window (oldest input 2026-10-13)")
    start, end = _region(_read(PAGE), "SourceCard")
    card = _mask(_read(PAGE)[start:end])
    for key in PAIR:
        assert key not in card, f"SourceCard renders `{key}`"
    assert not any(k in _mask(_read(ROW)) for k in PAIR), "SourceHealthRow reads the pair"
