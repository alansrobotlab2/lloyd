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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
