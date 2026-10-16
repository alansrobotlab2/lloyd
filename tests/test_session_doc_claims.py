"""#909 clause 5: the drain comments in the session routes are claims, not decor.

`app/routers/sessions.py` carried `# drain all queued turns` above a call that
drained one tier. The comment was true of nothing: `drain_pending` with
`source=None` pops `pending_ambient` only, so the delete route wiped a session
and left any queued user turn in the in-memory queue, where the consumer ran
it for a transcript whose file had just been unlinked. Nobody noticed for
months because the code agreed with the comment's *verb* — it does drain — and
only disagreed with the word *all*.

That is why this file greps the literal string rather than reasoning about the
code: the code is now fixed and stays fixed under test in
`tests/test_session_queue.py`, but a reintroduced comment of that shape is the
artifact that would mislead the next caller, and no behavioural test reads a
comment. Two further assertions hold the prose to the same standard — the
delete handler's docstring has to name both tiers it drains (a docstring that
says "ambient" while calling with `source="all"` is the same lie in a longer
form), and `drain_pending`'s documented `source=None` default has to stay
ambient-only, because `/cancel?drain_pending=true`'s contract — user turns are
never silently dropped — rests on it.

Every absence check here is paired with a positive control, a string known to
be in the same file, and the file's line count is printed: a `grep -c` that
returns 0 because the pattern is malformed against the corpus is a false
negative, and a silently-unread file cannot pass as a clean one.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from app import next_session_notes as nsn_mod
from app import sessions_io
from app.routers import sessions as sess_mod

ROOT = inspect.getsourcefile(sess_mod).rsplit("/app/", 1)[0]
SESSIONS_ROUTE = f"{ROOT}/app/routers/sessions.py"
SESSIONS_IO = inspect.getsourcefile(sessions_io)

# The false comment by its exact wording, so the assertion cannot drift into
# "some comment that looks optimistic". At triage it had exactly one hit
# repo-wide, at `app/routers/sessions.py:767`.
FALSE_DRAIN_COMMENT = "# drain all queued turns"

APP_DIR = Path(ROOT) / "app"
TESTS_DIR = Path(ROOT) / "tests"

# `re.escape` on the call, then the first argument as the anchor: the bare name
# would also match `async def drain_pending(...)`, `from app.sessions_io import
# drain_pending,`, and the unrelated `_drain_pending` closure in
# `app/inner_voice/observer.py`, so a count built on it would not be a count of
# callers.
CALL_PATTERN = re.compile(re.escape("await drain_pending") + r"\((session_id|sid)")


def _scan_call_sites(root) -> list:
    """Every awaited `drain_pending` call under `root`, as (path:line, line).

    The whole-tree equivalent of the grep this item's triage ran, so the
    assertion below is the same measurement rather than a paraphrase of it.
    (Deliberately not spelled as the pattern itself: this file is under `tests/`,
    and a docstring quoting the call verbatim would count itself as a caller.)
    """
    hits = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if CALL_PATTERN.search(line):
                hits.append((f"{path.relative_to(ROOT)}:{number}", line.strip()))
    return hits


def _route_text() -> str:
    with open(SESSIONS_ROUTE, encoding="utf-8") as fh:
        text = fh.read()
    assert len(text.splitlines()) > 500, (
        f"{SESSIONS_ROUTE} reads as only {len(text.splitlines())} lines; an "
        "unread or truncated file must not pass an absence check"
    )
    return text


def test_the_false_drain_comment_is_gone_and_its_file_is_readable():
    text = _route_text()
    # Positive control first, and it has to be a count rather than a presence
    # check. The route module carries exactly two `await drain_pending(` call
    # sites — the delete route's both-tiers drain and the cancel route's
    # ambient-only one — so a module with no drain in it at all fails here
    # instead of passing on the absence assertion alone. A control that survives
    # the removal of the code the test is about is not a control.
    sites = re.findall(r"await drain_pending\(session_id", text)
    assert len(sites) == 2, (
        f"expected 2 drain_pending call sites in {SESSIONS_ROUTE} "
        f"(delete + cancel), found {len(sites)} — the absence check below is "
        "meaningless against a file that drains nothing"
    )
    assert 'drain_pending(session_id, source="ambient")' in text, (
        "positive control failed: the cancel route's explicit ambient drain is "
        "missing, so this file is not the one the clause is about"
    )
    assert FALSE_DRAIN_COMMENT not in text, (
        f"{SESSIONS_ROUTE} carries {FALSE_DRAIN_COMMENT!r} again. `source=None` "
        "is ambient-only; a comment over a one-tier drain is what made "
        "DELETE /api/sessions/{{id}} look like it emptied the queue (#909)"
    )


def test_no_route_drains_with_the_default_and_delete_asks_for_both():
    """The comment is gone; the shape it lied about has to go with it.

    Nothing in the router may drain via the ambient-only default, and the
    delete handler must name the both-tiers sentinel — that is what gives
    `drain_pending`'s `pending_user` pop a production caller.
    """
    text = _route_text()
    calls = re.findall(r"drain_pending\(([^)]*)\)", text)
    assert calls, f"no drain_pending call found in {SESSIONS_ROUTE}"
    assert not re.search(r"drain_pending\([^)]*source=None", text), (
        "a route drains with an explicit source=None again — the default is "
        f"ambient-only ({len(calls)} drain call(s) in this file: {calls})"
    )
    body = inspect.getsource(sess_mod.delete_session)
    assert 'drain_pending(session_id, source="all")' in body, (
        "delete_session no longer asks for both tiers, so a queued user turn "
        "survives the delete and runs to a deleted transcript"
    )


def test_the_pending_user_pop_has_a_production_caller():
    """The item's own acceptance check, pinned: re-run the caller grep.

    Before #909 this grep over `app/` found two callers and neither reached the
    `pending_user` pop, which is what made the branch dead. Now it must find
    exactly two call sites and one of them must be the both-tiers sentinel —
    and the sentinel must sit in the delete handler, not in some second caller
    that nobody routes to.

    The pattern is anchored on the call's first argument (`session_id`/`sid`)
    rather than the bare name, so the `def` line and the unrelated
    `_drain_pending` closure in `app/inner_voice/observer.py` stay out of the
    count. Both denominators are printed: a 0 here is a verdict about a pattern,
    and a pattern that matches nothing because it is malformed against the tree
    looks exactly like a clean result.
    """
    calls = _scan_call_sites(APP_DIR)
    print(f"drain_pending call sites in app/ ({len(calls)}): {calls}")
    assert len(calls) == 2, (
        "expected exactly 2 drain_pending call sites in app/ (the delete route "
        f"and the cancel route); found {len(calls)}: {calls}. The dead-branch "
        "premise of #909 is only closed while that set is exactly these two"
    )
    both = [loc for loc, line in calls if 'source="all"' in line]
    assert len(both) == 1, (
        f"expected exactly 1 both-tiers drain in app/, found {len(both)}: {both}"
    )
    assert "app/routers/sessions.py" in both[0], (
        f"the both-tiers drain is not in the session router: {both[0]}"
    )
    assert 'drain_pending(session_id, source="all")' in dict(calls)[both[0]]
    # Positive control for the same grep over the tests: a scan that came up
    # empty everywhere would pass a `calls == []` reading of the tree, so prove
    # the pattern matches real coverage before trusting what it counts.
    test_calls = _scan_call_sites(TESTS_DIR)
    print(f"drain_pending call sites in tests/ ({len(test_calls)})")
    assert len(test_calls) >= 4, (
        f"only {len(test_calls)} drain_pending call sites under tests/ — the "
        "queue suite's own drains are what this grep is calibrated against"
    )
    assert any('source="all"' in line for _loc, line in test_calls), (
        "no test exercises the sentinel either, so the grep above is counting "
        "a call nothing verifies"
    )


def test_the_delete_docstring_names_both_tiers_it_drains():
    doc = inspect.getdoc(sess_mod.delete_session) or ""
    assert doc, "delete_session lost its docstring"
    # Positive control: the docstring has to be about the drain at all.
    assert "drain" in doc.lower(), f"delete_session's docstring never mentions draining: {doc!r}"
    assert re.search(r"\*\*both\*\*|both\b", doc, re.I), (
        f"delete_session's docstring does not say it drains both tiers: {doc!r}"
    )
    assert re.search(r"ambient", doc, re.I) and re.search(r"\buser\b", doc, re.I), (
        "delete_session's docstring must name both tiers by name — 'ambient' "
        f"and 'user' — not just 'queued turns': {doc!r}"
    )


def test_drain_pending_still_documents_ambient_only_for_none():
    """The default `/cancel?drain_pending=true` depends on: `None` means
    ambient only, and it is still the default argument.
    """
    doc = inspect.getdoc(sessions_io.drain_pending) or ""
    params = inspect.signature(sessions_io.drain_pending).parameters
    assert params["source"].default is None, (
        f"drain_pending's `source` default moved to {params['source'].default!r}; "
        "the ambient-only default is the documented /cancel contract"
    )
    assert re.search(r"source is None[^.]*ambient only", doc, re.S | re.I), (
        f"drain_pending no longer documents None as ambient-only: {doc!r}"
    )
    assert re.search(r"user turns\s+are never silently dropped", doc), (
        "the reason for the ambient-only default left the docstring; it is "
        f"what tells a caller not to copy the delete route: {doc!r}"
    )
    # The both-tiers sentinel has to be documented too — an undocumented
    # `"all"` is the next caller's trap in the other direction.
    assert re.search(r'source="all"[^.]*both tiers', doc, re.S), (
        f'drain_pending does not document the "all" sentinel as both tiers: {doc!r}'
    )
    with open(SESSIONS_IO, encoding="utf-8") as fh:
        assert 'source == "user" or source == "all"' in fh.read(), (
            "the pending_user pop is unreachable again — nothing but the "
            '"all" sentinel reaches it'
        )


ARCH_PAGE = f"{ROOT}/architecture/ambient-context-injection.md"

# The pre-fix paragraph's three false claims, by their exact wording. Each was
# true of the tree the 2026-09-12 review read and is false of this one; the
# page describing them as current is the same trap the code comment was, in a
# longer form.
FALSE_DOC_CLAIMS = (
    "has no caller anywhere",
    "a `# drain all queued turns` comment the code does not honour",
    "so a queued **user** turn survives the wipe",
)


def test_the_architecture_page_describes_the_drain_that_shipped():
    """`architecture/ambient-context-injection.md` is the surface a future
    session reads instead of the code, so the paragraph that walked the reader
    through the unreachable user tier has to describe the sentinel now."""
    with open(ARCH_PAGE, encoding="utf-8") as fh:
        text = fh.read()
    lines = text.splitlines()
    assert len(lines) > 200, (
        f"{ARCH_PAGE} reads as only {len(lines)} lines; an unread or truncated "
        "file must not pass an absence check"
    )
    # Positive control: this page really does document the user tier.
    assert "pending_user" in text, "positive control failed: this is not the queue's page"
    # Wrapped prose: a claim can break across lines anywhere in its middle, so
    # every phrase below is matched against the page with runs of whitespace
    # collapsed. That is a match rule, not a licence — the strings are still
    # whole sentences, and a page with no wrapping matches identically.
    flat = re.sub(r"\s+", " ", text)
    assert 'source="all"' in flat, (
        "the page never names the both-tiers sentinel, so it cannot be "
        "describing the drain that shipped"
    )
    for claim in FALSE_DOC_CLAIMS:
        assert claim not in flat, (
            f"the page still asserts {claim!r} — true before #909 landed, false "
            f"now ({len(FALSE_DOC_CLAIMS)} pre-fix claims checked)"
        )
    # Dated-evidence rule: the superseded claim stays legible only under a date,
    # so a reader knows which tree each sentence describes.
    assert re.search(r"2026-09-20 — \*\*#909 landed\*\*", text), (
        "no dated entry records that the delete route's drain changed; the "
        "corrected paragraph alone does not tell a reader which tree it describes"
    )


# --- #2599: the ambient decision store is the deciding turn's, not the session's

#: The false half of the block comment above `_ambient_decisions`, by its exact
#: wording. The pop sits inside `_run_turn`'s harness `result` branch, and
#: `messages.py`'s Inner-Voice comment names the endings that never reach it:
#: "the user hit Stop, the harness raised, the consumer was cancelled". A
#: comment promising a `finally` clause is what made an orphaned `surface=false`
#: look impossible to the review that read this code before #2599.
FALSE_FINALLY_CLAIM = "finally clause"

#: The arch page's own misattribution. "The turn's own completion path" stated
#: the intent and not the key — and a sentence that sounds like the invariant
#: closes the report instead of opening it.
FALSE_ROW_CLAIM = "popped by the turn's own completion path"


def _comment_above(path: str, anchor: str) -> str:
    """The run of `#` lines immediately above the first line containing `anchor`.

    Scoped to the block rather than the whole file: the absence check below is a
    claim about the comment #2599 refuted, not about the phrase "finally clause"
    anywhere in a 1,000-line module.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    hit = next((i for i, line in enumerate(lines) if anchor in line), None)
    assert hit is not None, f"{Path(path).name} no longer declares {anchor!r}"
    # The block ends a blank line above the declaration, so step back over the
    # padding before collecting `#` lines — a helper that stopped at the blank
    # would return "" and report a present comment as missing.
    start = hit - 1
    while start >= 0 and not lines[start].strip():
        start -= 1
    while start >= 0 and lines[start].lstrip().startswith("#"):
        start -= 1
    block = "\n".join(lines[start + 1:hit])
    assert block, f"no comment block above {anchor!r} in {Path(path).name}"
    assert anchor not in block, (
        f"the block above {anchor!r} reached the declaration itself: {block!r}")
    return block


def test_the_ambient_decision_comment_states_the_turn_scoped_rule():
    """The comment above `_ambient_decisions` has to describe where the pop
    really happens and whose decision a stored one is.

    Both halves matter. The false one — a `finally` clause that never existed —
    is what made the leak read as impossible; the missing one — that only the
    turn which recorded a decision may consume it — is the invariant the code
    now holds, and a reader who is not told it may well break it again.
    """
    block = _comment_above(SESSIONS_IO, "_ambient_decisions: dict")
    assert FALSE_FINALLY_CLAIM not in block, (
        f"the comment promises a pop that does not exist: {block!r}")
    # Positive controls, so this cannot be passed by deleting the comment: it is
    # still about this store, still names the mechanism, and now names both the
    # real pop site and the owning turn.
    assert "ambient_decide" in block, block
    assert "result" in block, f"the comment never names where the pop happens: {block!r}"
    assert "turn_id" in block, f"the comment never says whose it is: {block!r}"


def test_the_arch_page_names_the_deciding_turn_as_the_only_consumer():
    """The page's `_ambient_decisions` row and its consume step are what a future
    session reads instead of the code, so they have to say the decision is
    consumed by the turn that recorded it and by nobody else."""
    with open(ARCH_PAGE, encoding="utf-8") as fh:
        text = fh.read()
    assert len(text.splitlines()) > 200, (
        f"{ARCH_PAGE} reads as only {len(text.splitlines())} lines; an unread or "
        "truncated file must not pass an absence check")
    flat = re.sub(r"\s+", " ", text)
    # Positive controls: still the ambient page, still documenting this store.
    assert "_ambient_decisions" in flat, "positive control failed: not this page"
    assert "ambient_decide" in flat, "positive control failed: not this store"
    assert FALSE_ROW_CLAIM not in flat, (
        "the page still calls the consumer 'the turn's own completion path', "
        "which is the sentence that hid #2599")
    row = re.search(r"^\| `_ambient_decisions` \|.*\|$", text, re.MULTILINE)
    assert row, "the Components table no longer has a row for `_ambient_decisions`"
    assert "turn_id" in row.group(0), row.group(0)
    assert re.search(r"deciding turn", row.group(0)), row.group(0)
    # The walkthrough step shows the widened call, not the session-only pop.
    assert re.search(r"take_ambient_decision\(sid, turn_id\)", flat), (
        "the consume step still pops by session alone")


# --- #2601: the next-session channel's peek must point at a route that exists

#: The peek's false promise, by its exact wording. `peek_next_session_notes` said
#: it existed "for the debug endpoint and for a human checking tomorrow whether a
#: nightly pass delivered anything", while `git grep -n 'next-session' --
#: app/routers/` returned one route — the POST door. A sentence like that is a
#: pointer, and a future session goes looking for the route it promises and finds
#: none: the store's entire public surface was its write door.
FALSE_PEEK_CLAIM = "for the debug endpoint"

#: The route that makes the sentence true, spelled as the router spells it. Named
#: rather than paraphrased so the assertion is about the pointer, not the prose.
READ_DOOR = "GET /api/next-session-notes"


def test_the_peek_names_the_route_that_serves_it():
    """A docstring that points at a route is a claim about the tree, and this one
    pointed at nothing for as long as the channel existed."""
    doc = inspect.getdoc(nsn_mod.peek_next_session_notes) or ""
    assert "note" in doc.lower(), (
        "positive control failed: the peek's docstring no longer describes this "
        "channel, so the absence check below is grading an empty string")
    assert FALSE_PEEK_CLAIM not in doc, (
        f"the docstring still points at a debug endpoint that does not exist: {doc!r}")
    assert READ_DOOR in doc, (
        f"the docstring must name the route that serves it ({READ_DOOR}): {doc!r}")


def test_the_arch_page_lists_the_notes_read_door_beside_the_write_door():
    """The Endpoints table is what a reader consults instead of the router, so the
    read row has to sit beside the write row of the same channel — both doors of
    one store, readable in a single pass of the table.
    """
    with open(ARCH_PAGE, encoding="utf-8") as fh:
        text = fh.read()
    assert len(text.splitlines()) > 200, (
        f"{ARCH_PAGE} reads as only {len(text.splitlines())} lines; an unread or "
        "truncated file must not pass an absence check")
    # Positive controls: still this page, and still the write door it documented.
    post_row = re.search(r"^\| `POST /api/next-session-note` \|.*\|$", text, re.MULTILINE)
    assert post_row, "the Endpoints table no longer has a write row to sit beside"
    get_row = re.search(r"^\| `GET /api/next-session-notes` \|.*\|$", text, re.MULTILINE)
    assert get_row, (
        "the Endpoints table has no read row, so the page still describes a "
        "channel whose only door is the one that writes")
    assert "peek_next_session_notes" in get_row.group(0), (
        f"the read row never names what it serves: {get_row.group(0)}")
    # The table's own vocabulary for this property, the one the prefetch peek row
    # already uses: two peek rows that describe themselves differently are two
    # readers guessing whether either of them is safe to call.
    assert "does not drain" in get_row.group(0), (
        f"the read row never says reading is non-consuming: {get_row.group(0)}")
    post_line = text[:post_row.start()].count("\n")
    get_line = text[:get_row.start()].count("\n")
    assert get_line == post_line + 1, (
        f"the read row has drifted {get_line - post_line:+d} line(s) from the write "
        "row of the same channel; the pair is the point")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))



# ── #2836: the sessions census comments are dated readings, not growth floors ──

#: The two files that describe this directory's shape to whoever tunes it next,
#: resolved through the modules this file already imports rather than a path that
#: could silently name some other checkout.
_CENSUS_MODULES = {"sessions.py": sess_mod, "sessions_io.py": sessions_io}

#: Figures triage found still quoted on 2026-10-16, each measured on 2026-09-10
#: or 2026-09-21. They are not merely old: the quantity they state moved in BOTH
#: directions afterwards (the four-part half of `sessions/` read 10,533 on
#: 2026-09-21, 10,040 on 2026-10-16 morning, 10,083 hours later the same day),
#: and nothing in this codebase explains either movement. A comment quoting one
#: is therefore describing a trend it cannot justify.
_STALE_CENSUS_FIGURES = ("479 of its 643", "3,853", "55 ms", "737 MB")

#: A string that MUST be found in each file. Without it an absence verdict is
#: unfalsifiable — a grep against a file that never resolved also returns nothing,
#: which is this shape's own recorded failure mode (a 0-hit grep needs a control).
_CENSUS_CONTROL = {"sessions.py": "_BACKGROUND_PLATFORM_SCAN_CEILING",
                   "sessions_io.py": "HEAD_PLATFORM_WINDOW"}

#: The readings #2836 published in their place, so the restatement is pinned as
#: well as the deletion. Each token sits whole on one line on purpose: prose that
#: wrapped is not an assertion.
_RESTATED_READINGS = {
    "sessions.py": ("10,169", "10,083", "125 ms", "2026-10-16: median"),
    "sessions_io.py": ("10,083", "1.78 GiB", "byte 160 and a maximum of 215",
                       "370-469"),
}

_STALE_DATE = re.compile(r"2026-09-(?:10|21)\b")
_COMMA_COUNT = re.compile(r"\b\d{1,3},\d{3}\b")
#: The exact mistake #2650's diff made: it wrote `at least 3,854 files and
#: rising` for a directory that shrank ~490 files over the next three days. A
#: count is a reading; the bound is the constant next to it.
_COUNT_AS_FLOOR = re.compile(r"at least [\d,]+ (?:files|sessions)"
                             r"|no fewer than|\band rising\b")


def _census_source(name: str) -> list[str]:
    """The whole file, lines, for one of the two census-carrying sources."""
    path = Path(_CENSUS_MODULES[name].__file__).resolve()
    assert path.name == name and path.is_file(), (
        f"{name} resolved to {path}, which is not this checkout's source — every "
        "absence assertion below would then be reading a file that is not the one "
        "under review")
    return path.read_text(encoding="utf-8").splitlines()


def test_the_sessions_census_comments_quote_no_retired_reading():
    """#2836: no 2026-09-10/09-21 census figure survives in either file, and no
    count is presented as a growth floor.

    `git grep -nE '3,853|55 ms|737 MB|479 of its 643'` over these two files is the
    graded check, and this body is that command — `-E` is not decoration. Plain
    `git grep` is BRE, where a pipe is a LITERAL pipe, so the same pattern without
    it exits 1 on a file containing every string: the check as filed was vacuous,
    and a vacuous check is the failure this file exists to prevent.
    """
    for name, lines in ((n, _census_source(n)) for n in _CENSUS_MODULES):
        print(f"{name}: {len(lines)} lines scanned whole")
        assert _CENSUS_CONTROL[name] in "\n".join(lines), (
            f"{name} does not even contain {_CENSUS_CONTROL[name]!r}, so the "
            "absence assertions below are reading the wrong file")

        for figure in _STALE_CENSUS_FIGURES:
            hits = [i + 1 for i, line in enumerate(lines) if figure in line]
            assert not hits, (
                f"{name}:{hits} still quotes the {figure!r} census figure — the "
                "restatement belongs in this change (#2836 clause 5)")

        stale = [i + 1 for i, line in enumerate(lines)
                 if _COMMA_COUNT.search(line)
                 and any(_STALE_DATE.search(near) for near in lines[max(0, i - 3):i + 4])]
        assert not stale, (
            f"{name}:{stale} put a thousands-scale count next to a 2026-09-10/09-21 "
            "date. Those readings are retired in both directions; a census line "
            "states the date it was read or it is not a census line")

        floors = [i + 1 for i, line in enumerate(lines) if _COUNT_AS_FLOOR.search(line)]
        assert not floors, (
            f"{name}:{floors} presents a measured count as a floor — the shape "
            "`at least 3,854 files and rising`, which the directory contradicted "
            "within three days")


def test_the_restated_sessions_census_readings_are_present_and_dated():
    """The replacement has to exist, not only the deletion.

    Clause 5 is half absence and half replacement: the figures now in these two
    files are 2026-10-16 readings, and pinning them stops a later edit dropping
    the replacement and leaving the comment claiming nothing at all.
    """
    for name, tokens in _RESTATED_READINGS.items():
        text = "\n".join(_census_source(name))
        dated = text.count("2026-10-16")
        assert dated >= 2, (
            f"{name} carries only {dated} `2026-10-16` readings; the census "
            "restatement is supposed to have dated the lines it rewrote")
        for token in tokens:
            assert token in text, (
                f"{name} no longer publishes the 2026-10-16 reading {token!r}")
