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


# ── #3181: a per-day session-mint figure is a reading over a named window ──

#: The two files that tell whoever tunes `sessions/`'s horizons how fast the
#: directory fills, both read BY REPO PATH off the `ROOT` this file already
#: derives — the identical path `tests/test_retention_sweep.py` builds its
#: `spec_from_file_location("retention_sweep", _SCRIPT)` from. Not through
#: `module.__file__` the way `_census_source` above resolves its two: the sweep
#: script cannot be resolved that way at all (a hyphenated filename is not
#: importable by name), and a scan that read one file off a live module and the
#: other off a path would be two rules with one name. This scan reads comments,
#: not behaviour, so the path is the right handle for both — and a path that does
#: not resolve is caught below, before any absence is asserted.
_MINT_SOURCES = {
    "sessions_io.py": Path(ROOT) / "app" / "sessions_io.py",
    "retention-sweep.py": (Path(ROOT) / "scripts" / "groundskeeper"
                           / "retention-sweep.py"),
}

#: A string that MUST be found in each file before any absence is asserted. A
#: scan of a path that never resolved is indistinguishable from a scan of a clean
#: file, and a 0-hit grep with no positive control is this repo's own recorded
#: failure mode — `_CENSUS_CONTROL` above exists for the same reason.
_MINT_CONTROL = {
    "sessions_io.py": "NON_USER_PLATFORMS",
    "retention-sweep.py": "BACKGROUND_SESSION_ARCHIVE_AGE_DAYS",
}

#: The command each figure in those two files comes from, and the window it was
#: run over. It counts SURVIVING files: `sweep_sessions()` gzips a transcript at
#: its horizon instead of deleting it, so a directory count is a count of what is
#: still there. Including `.json.gz` is load-bearing rather than tidy — counted
#: over 2026-10-13 by hand, the day reads 326 live `.json`, 44 archived, 370
#: together, and 370 is the number the comment carries.
_MINT_DERIVATION = "filename date prefix over live .json plus .json.gz"
_MINT_WINDOW = "2026-10-13 to 10-22"
_MINT_RANGE = "333-569"
_MINT_QUALIFIER = "not a count of what was minted"

#: Lines copied from the tree this item was filed on, fed back through the scan by
#: `test_the_mint_claim_rail_sees_the_wording_it_replaced`. The first two are the
#: phrasings this item replaced; the third is the dated claim at
#: `sessions_io.py:154-155` that this item deliberately left alone, here because its
#: figure and its noun sit on different comment lines — which is exactly why the
#: scan joins lines rather than reading them one at a time.
RETIRED_MINT_SHAPES = (
    ["# ~240 background sessions a day against ~14 chats (measured over the week to"],
    ["    ~300 ordinary runs a day and belongs in a newest-first window, while one"],
    ['#: directory to answer "is this one a chat?" — the directory held 370-469',
     "#: four-part files per day over 2026-10-13 to 10-15 against 1-5 chat-shaped"],
)

_ANY_WINDOW = re.compile(r"2026-\d{2}-\d{2}")
_RETIRED_WINDOW = re.compile(r"2026-09-(?:10|21)\b")

#: Framing that turns a reading back into a promise. #2650 wrote
#: `at least 3,854 files and rising` about a directory that had shrunk ~490 files
#: three days later. Mints grow `sessions/` and `sweep_sessions` shrinks it, so a
#: rate measured over any window has neither end pinned — the number is a reading
#: and the constant next to it is the only bound in the file.
_FLOOR_FRAMING = re.compile(
    r"at least|no fewer than|as a minimum|at minimum|minimum of|as low as"
    r"|never under|never below|\bfloor\b|\band rising\b", re.IGNORECASE)

#: The scan exactly as #3181's clause 3 files it, kept as its own constant so a
#: node below can prove the working pattern is a SUPERSET of it rather than a
#: different rule wearing its clothes.
_MINT_CLAIM_CORE = re.compile(
    r"[0-9][0-9,~-]* (?:background )?(?:sessions|runs|chats|transcripts|files)"
    r" (?:a|per) day")

#: Run literally against these two files the core pattern finds ONE of the three
#: per-day mint claims in them. `~240 background sessions a day` matches it;
#: `~300 ordinary runs a day` puts the word `ordinary` into the slot the core
#: wants empty, and the `370-469` claim wraps its noun onto the next comment line,
#: so no single line holds figure and noun together for it to find. A rail blind
#: to two of the three lines it was written for cannot fail on the shape it exists
#: to catch, so the working pattern admits up to three short words between figure
#: and noun and the scan joins wrapped comment lines. That is a widening in one
#: direction only: it does NOT reach the sweep script's other per-day claims —
#: bytes, rows, MB, spoken replies and Bash calls a day at :116, :153, :326,
#: :2291, :2650, :2660, :3510, :3511 — nor `~70 a day` at `sessions_io.py:431`,
#: because none of those nouns is in this set. That is what keeps this item on the
#: three lines it names and off the ten it does not.
_MINT_CLAIM = re.compile(
    r"[0-9][0-9,~-]*(?: [A-Za-z0-9~.,/+%-]{1,14}){0,3} "
    r"(?:background )?(?:sessions|runs|chats|transcripts|files) (?:a|per) day")

_WINDOW_RADIUS = 3


def _mint_source(name: str) -> list[str]:
    """One scanned file, whole, as lines — with the control asserted first.

    Order is the point: control first, then any absence. A path that resolved into
    empty space passes every `not in` check ever written against it, and the only
    thing that says so is a string that had better be there.
    """
    path = _MINT_SOURCES[name]
    assert path.is_file(), (
        f"{name} is scanned at {path}, which is not a file in this checkout — an "
        "unread source reads as a clean one")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert _MINT_CONTROL[name] in "\n".join(lines), (
        f"{name} at {path} does not even contain {_MINT_CONTROL[name]!r}, so this "
        "is not the file the rule was written for")
    assert len(lines) > 200, (
        f"{name} scanned only {len(lines)} lines; a truncated read cannot pass an "
        "absence check")
    return lines


def _mint_join(lines: list[str]) -> tuple:
    """The file as one string with comment markers stripped, plus line offsets.

    Wrapping is the reason this exists rather than a line loop: the largest mint
    claim in the corpus breaks `370-469` away from `four-part files per day` across
    two comment lines, so a per-line scan never sees a claim that is plainly on the
    page. The offsets map a match back to the line a reader has to go fix.
    """
    parts, offsets = [], []
    for line in lines:
        body = line.strip()
        if body.startswith("#:"):
            body = body[2:]
        elif body.startswith("#"):
            body = body[1:]
        offsets.append(len(" ".join(parts)))
        parts.append(body.strip())
    return " ".join(parts), offsets


def _mint_violations(lines: list[str]) -> list[str]:
    """Every per-day mint claim in `lines` that is not a dated reading.

    Clean means: a `2026-MM-DD` within `_WINDOW_RADIUS` lines of the line the
    figure starts on, AND the NEAREST such token is not one of the two retired
    readings, AND no floor framing sits in that same window. Nearest rather than
    any: the line #3181 was filed on failed with the retired `2026-09-10` on the
    very next line and no live date anywhere near it, and a rule that accepted a
    retired window because a newer one happened to be three lines off would
    re-bless that exact shape.
    """
    text, offsets = _mint_join(lines)
    line_of_offset = {off: i for i, off in enumerate(offsets)}
    problems = []
    for match in _MINT_CLAIM.finditer(text):
        first = line_of_offset[max(off for off in offsets if off <= match.start())]
        claim = match.group(0).strip()
        lo = max(0, first - _WINDOW_RADIUS)
        hi = min(len(lines), first + _WINDOW_RADIUS + 1)
        tokens = [(abs(j - first), j, tok)
                  for j in range(lo, hi)
                  for tok in _ANY_WINDOW.findall(lines[j])]
        if not tokens:
            problems.append(
                f"line {first + 1}: {claim!r} states a per-day mint volume with no "
                f"2026-MM-DD within {_WINDOW_RADIUS} lines of it")
            continue
        dist, near_line, token = min(tokens)
        if _RETIRED_WINDOW.search(token):
            problems.append(
                f"line {first + 1}: {claim!r} sits nearest to {token!r} on line "
                f"{near_line + 1} — 2026-09-10 and 2026-09-21 are retired in both "
                "directions by #2836, so a window named by either is not a window")
        framed = _FLOOR_FRAMING.search("\n".join(lines[lo:hi]))
        if framed:
            problems.append(
                f"line {first + 1}: {claim!r} is framed by {framed.group(0)!r}, "
                "which turns a reading over a window back into a bound")
    return problems


def test_no_per_day_mint_claim_sits_outside_a_named_window():
    """#3181 clauses 1-4: every per-day mint-volume claim in either file has to be
    a reading over a stated, non-retired window, or this node fails.

    The graded check for the item is
    `git grep -nE '240 background|14 chats|~300 ordinary'`, which exits 1 once the
    two comments are restated. An absence grep is worth keeping and is not worth
    relying on: it bans the three strings that happened to be wrong, so the next
    `600 sessions a day` added next month walks straight past it and re-creates the
    defect with a number nobody measured. Scanning the shape instead is what makes
    the window a condition of writing the figure down.
    """
    for name in _MINT_SOURCES:
        lines = _mint_source(name)
        problems = _mint_violations(lines)
        hits = sum(1 for _ in _MINT_CLAIM.finditer(_mint_join(lines)[0]))
        print(f"{name}: {len(lines)} lines scanned whole, {hits} per-day mint "
              f"claims found, {len(problems)} of them outside a live window")
        assert not problems, (
            f"{name} states a per-day mint volume that is not a dated reading:\n"
            + "\n".join(problems))


def test_the_mint_figures_are_restated_with_window_derivation_and_qualifier():
    """#3181 clauses 1 and 2: the replacement exists in both files, whole.

    Half of this item is deletion and half is the restatement; pinning only the
    deletion would let a later edit drop the new figure and leave the comment
    asserting nothing at all — which is worse than the wrong figure it replaced,
    because the wrong one at least said where it came from.
    """
    for name in _MINT_SOURCES:
        lines = _mint_source(name)
        text = "\n".join(lines)
        # The derivation and the qualifier are pinned against the same
        # comment-unwrapped text the scan reads, because both are prose that wraps:
        # `_RESTATED_READINGS` above deliberately pins its numerals whole on one
        # line, which is right for `10,169` and wrong for a nine-word clause a
        # docstring breaks across a line end. The figure and its window stay a
        # same-line assertion below, which is the pair that must not travel apart.
        joined = _mint_join(lines)[0]
        assert _MINT_RANGE in text, (
            f"{name} no longer publishes the {_MINT_RANGE!r} four-part-mint range "
            "measured over 2026-10-13 to 10-22 (#3181)")
        assert _MINT_WINDOW in text, (
            f"{name} no longer names the {_MINT_WINDOW!r} window its figure was "
            "read over")
        assert _MINT_DERIVATION in joined, (
            f"{name} publishes a mint figure but no longer carries its derivation "
            f"{_MINT_DERIVATION!r} — a figure with no method is #3181's premise")
        assert _MINT_QUALIFIER in joined, (
            f"{name} lost the surviving-files qualifier {_MINT_QUALIFIER!r}; the "
            "sweep gzips rather than deletes, so a directory count is not a mint "
            "count")
        # The range and its window share a line, so quoting one quotes the other.
        rate_lines = [i + 1 for i, line in enumerate(lines)
                      if _MINT_RANGE in line and "sessions a day" in line
                      and _MINT_WINDOW in line]
        assert rate_lines, (
            f"{name} has no single line holding the range, the words 'sessions a "
            "day' and the window together; a figure that travels without its "
            "window is the artefact this item exists to stop")

    sweep = "\n".join(_mint_source("retention-sweep.py"))
    for retired in ("240 background", "14 chats"):
        assert retired not in sweep, (
            f"retention-sweep.py still carries {retired!r}, the 2026-09-10 reading "
            "#3181 clause 1 restates")
    assert "~300 ordinary" not in "\n".join(_mint_source("sessions_io.py")), (
        "sessions_io.py still carries '~300 ordinary', the undated figure #3181 "
        "clause 2 restates")


def test_the_mint_claim_rail_sees_the_wording_it_replaced():
    """The scan has to be able to fail, and to fail on the wording it replaced.

    An absence check whose pattern quietly stops matching reports a clean tree
    forever, which is #2836's `git grep`-without-`-E` trap in a softer costume. So
    the retired phrasings are fed back through the working pattern, the core
    pattern's own hits are required to be hits here too, and the rule is run
    against synthetic lines it must reject and must accept.
    """
    # Each retired phrasing, verbatim from the tree this item was filed on, is
    # still something the rail can see — and the wrapped one is visible ONLY to
    # the joined scan, which is the reason the scan joins at all.
    for shape in RETIRED_MINT_SHAPES:
        assert _MINT_CLAIM.search(_mint_join(shape)[0]), (
            f"the rail no longer matches {shape[-1].strip()!r}, so a clean run of "
            "the node above would mean nothing")
    wrapped = RETIRED_MINT_SHAPES[2]
    assert not any(_MINT_CLAIM.search(line) for line in wrapped), (
        "the wrapped fixture now matches on a single line, so the join in "
        "_mint_join is no longer load-bearing and the comment above it lies")

    # The working pattern is a superset of the clause's own pattern, checked on
    # the live files rather than asserted in prose.
    for name in _MINT_SOURCES:
        text, _ = _mint_join(_mint_source(name))
        core = {m.group(0) for m in _MINT_CLAIM_CORE.finditer(text)}
        wide = {m.group(0) for m in _MINT_CLAIM.finditer(text)}
        assert core <= wide, (
            f"{name}: the working pattern lost the clause's own hits {sorted(core - wide)}")

    # The rule fires on each defect shape, and only on them.
    undated = ["# the directory takes 900 sessions a day to fill", "", ""]
    one = _mint_violations(undated)
    assert len(one) == 1 and "no 2026-MM-DD" in one[0], (
        f"an undated mint claim read as clean: {one}")

    retired = ["# ~240 background sessions a day against ~14 chats",
               "# measured over the week to 2026-09-10"]
    one = _mint_violations(retired)
    assert len(one) == 1 and "retired" in one[0], (
        f"a claim whose nearest window is the retired reading read as clean: {one}")

    floored = ["# 400 sessions a day over 2026-10-13 to 10-22 (filename date prefix",
               "# over live .json plus .json.gz), at least"]
    one = _mint_violations(floored)
    assert len(one) == 1 and "bound" in one[0], (
        f"a mint claim framed as a floor read as clean: {one}")

    clean = ["#: the directory held 333-569 four-part sessions a day",
             f"#: over {_MINT_WINDOW} ({_MINT_DERIVATION},",
             "#: which is not a count of what was minted)"]
    assert _mint_violations(clean) == [], (
        "the properly dated, derived, non-floored shape was rejected — the rail "
        "is asking for something a restated comment cannot say")
