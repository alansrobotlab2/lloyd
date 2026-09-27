"""The scratchpad #1554 adds: a session's own free-form note file, its ceiling, and
the tool that writes it.

Three processes touch the same bytes. The model's append
(`agent_mcp/builtin_scratchpad.py`) runs in the MCP server process; the injection
(`app.scratchpad.build_scratchpad_anchor`) runs in the backend; the tally the
experiment needs (`summarize`) runs in the pool. They share a file and nothing
else, so calling the module functions in one process proves the file format and
not the seam — `test_the_tool_appends_for_the_bound_session...` is the node that
crosses the real boundary.

Two choices worth stating because they are what lets these tests fail:

* `DATA_ROOT` is patched rather than a resolved path, because `scratchpad_dir()`
  reads it at call time and the guarantee under test is "the file lives under the
  runtime data root" — a claim that needs a data root that could have been
  somewhere else.
* Ceiling assertions read `sp.MAX_INJECT_BYTES` itself, the declared constant,
  never a second copy of the number typed here. If the constant moves, these tests
  move with it and still assert the same thing.

Clause 3 (the cached prefix must not move) is pinned in
`tests/test_worker_budget_anchor.py` and clause 5 (the run record) in
`tests/test_workers_pool.py`, because those clauses are properties of the
machinery each of those files owns.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import agent_mcp.builtin_scratchpad as tool
import app.scratchpad as sp


#: Size of a filler note: big enough that ~6 of them meet the ceiling, small
#: enough that a test can name the third entry by hand.
FILLER = "y" * 700


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """A data root this test owns, so `scratchpad_dir()` resolves inside tmp_path."""
    root = tmp_path / "data"
    monkeypatch.setattr(sp, "DATA_ROOT", root)
    return root


def _fill(session_id: str, n: int) -> None:
    for i in range(n):
        sp.append(session_id, f"entry {i:03d}: {FILLER}")


def _call(args: dict):
    """One tool call through the module a model's call actually lands in."""
    res = asyncio.run(tool.call_tool("Scratchpad", args))
    return res.content[0].text, res.is_error


# ── clause 1: append-and-read, additive, under the data root ─────────────────


def test_appending_twice_leaves_both_notes_in_order_under_the_data_root(data_root):
    """Both halves of clause 1: two writes, both present, in the order written, and
    the file under the runtime data root rather than in the code tree."""
    first = sp.append("worker:bench-mine:aaa111", "tried the 4k window; OOMs at 6k")
    second = sp.append("worker:bench-mine:aaa111", "so the hypothesis is kv pressure")

    text = sp.read("worker:bench-mine:aaa111")
    assert "tried the 4k window" in text and "kv pressure" in text, text
    assert text.index("4k window") < text.index("kv pressure"), (
        "both notes are present but not in the order they were written; a decision "
        "log read out of order argues the opposite case")
    assert (second["writes"], second["bytes"]) == (
        first["writes"] + 1,
        first["bytes"] + len("so the hypothesis is kv pressure")), (
        f"the append reported {second}, which is not one more note of "
        f"{len('so the hypothesis is kv pressure')} bytes than {first}")

    path = sp.scratchpad_path("worker:bench-mine:aaa111")
    assert path.is_file()
    assert sp.scratchpad_dir() == data_root / sp.SCRATCHPAD_DIRNAME
    assert path.is_relative_to(sp.scratchpad_dir()), (
        f"{path} is not inside the scratchpad dir of the data root")
    code_tree = Path(sp.__file__).resolve().parent.parent
    assert not path.is_relative_to(code_tree), (
        f"{path} resolves inside the code tree at {code_tree}, which is the other "
        f"half of the clause: the file has to live under the runtime data root")


def test_an_append_never_replaces_the_file_it_is_writing_into(data_root):
    """`never overwritten` as a filesystem fact rather than a promise in a
    docstring.

    The same inode before and after the second append is the guarantee: a writer
    that built a new file and renamed it into place — the safe-looking way to
    write a file — would pass an ordering test and still lose a concurrent
    writer's note, because a rename carries only what its own author had read.
    """
    sp.append("worker:s1", "note one")
    path = sp.scratchpad_path("worker:s1")
    ino, size = path.stat().st_ino, path.stat().st_size
    sp.append("worker:s1", "note two")
    st = path.stat()
    assert st.st_ino == ino, "the second append replaced the file instead of extending it"
    assert st.st_size > size, "the second append did not grow the file"
    assert sp.stats("worker:s1")["writes"] == 2


def test_reading_an_untouched_session_is_empty_not_an_error(data_root):
    """A run that never wrote has to read as nothing, not as a failure: otherwise
    the first thing the affordance tells a model is that its own tool broke."""
    assert sp.read("worker:no-such-run") == ""
    assert sp.stats("worker:no-such-run") == {"writes": 0, "bytes": 0}


def test_a_note_that_mimics_the_entry_header_is_counted_as_one_write(data_root):
    """`writes` is the number step 1 of #1554 correlates against outcomes, so it has
    to mean "appends". A model that quotes the header format — readable verbatim in
    its own injected digest — would otherwise inflate its own write-rate sample, and
    with it the correlation the item is about."""
    quoted = f'quote of the format: "{sp.ENTRY_HEADER}"'
    sp.append("worker:s1", "one")
    sp.append("worker:s1", quoted)
    st = sp.stats("worker:s1")
    assert st["writes"] == 2, (
        f"one append of quoted text produced {st['writes']} counted writes")
    assert st["bytes"] == len("one") + len(
        quoted.replace("[[scratchpad entry", "[[ scratchpad entry")), (
            "the byte total no longer tracks the content actually stored")
    assert "quote of the format" in sp.read("worker:s1"), (
        "the mimicking note was dropped instead of neutralised")


# ── clause 2: one session, one file ─────────────────────────────────────────


def test_two_runs_of_the_same_source_get_separate_files(data_root):
    """`worker:session-distill:ab12cd` and `worker:session-distill:99ee01` are two
    runs of one source — the shape `new_background_session_id` mints. One shared
    file would hand run 2 run 1's conclusions about the corpus as though run 2 had
    reached them, which is the false reuse this item does not want."""
    sp.append("worker:session-distill:ab12cd", "run one found 3 duplicates")
    sp.append("worker:session-distill:99ee01", "run two is mid rebuild")

    one = sp.read("worker:session-distill:ab12cd")
    two = sp.read("worker:session-distill:99ee01")
    assert "run one" in one and "run two" not in one, one
    assert "run two" in two and "run one" not in two, (
        "the second run's scratchpad carries the first run's note")
    assert sp.stats("worker:session-distill:ab12cd")["writes"] == 1
    assert sp.stats("worker:session-distill:99ee01")["writes"] == 1


@pytest.mark.parametrize("bad", [
    "../worker:victim",
    "worker/../worker:victim",
    "worker:victim/../victim",
    "/tmp/elsewhere",
    "worker\\victim",
    "",
    ".",
    "..",
    "worker:victim\ntail",
])
def test_a_session_id_cannot_address_another_sessions_file(data_root, bad):
    """The second half of clause 2. The way a run would reach another session's
    file — if it could reach one — is by spelling a path into the id, so the
    separator characters are what has to be refused, and refused before anything
    is opened."""
    sp.append("worker:victim", "someone else's decision log")
    with pytest.raises(sp.ScratchpadError):
        sp.append(bad, "injected")
    with pytest.raises(sp.ScratchpadError):
        sp.read(bad)
    assert "injected" not in sp.read("worker:victim")
    written = sorted(p.name for p in sp.scratchpad_dir().iterdir())
    assert written == ["worker:victim.md"], (
        f"a rejected id still put a file in the directory: {written}")


def test_a_suffixed_id_gets_its_own_file_and_not_the_victims(data_root):
    """The non-traversal version of the same clause: ids become filenames by
    appending `.md`, which is injective, so `worker:victim.md` is a different
    session rather than an alias for `worker:victim`. Asserted explicitly because
    the regex alone does not tell a reader that."""
    sp.append("worker:victim", "mine")
    sp.append("worker:victim.md", "not yours")
    victim = sp.read("worker:victim")
    assert "mine" in victim and "not yours" not in victim, (
        "an id spelled with the extension reached into the other session's file")
    assert "not yours" in sp.read("worker:victim.md")
    assert sp.stats("worker:victim")["writes"] == 1


def test_a_real_worker_session_id_with_a_colon_is_accepted(data_root):
    """The population this exists for is colon-shaped: `new_background_session_id`
    (`workers/sources/_common.py`) mints `worker:<source>:<6hex>` for every worker
    turn. A validator that refused colons would reject every real run while every
    traversal test above stayed green."""
    sp.append("worker:bench-mine:9f3a1c", "colon ids are the real shape")
    assert "colon ids" in sp.read("worker:bench-mine:9f3a1c")


# ── clause 4: the injection ceiling ─────────────────────────────────────────


def test_injection_is_bounded_by_the_declared_ceiling(data_root):
    """The bound a caller cannot forget: `digest` applies the declared constant by
    default, so the number has one home. Asserted against that constant, not a
    copy of it."""
    _fill("worker:s1", 40)                       # ~28 KB stored
    body = sp.digest("worker:s1", sp.MAX_INJECT_BYTES)
    assert body, "a session with 40 notes injected nothing"
    assert len(body.encode("utf-8")) <= sp.MAX_INJECT_BYTES, (
        f"digest returned {len(body.encode('utf-8'))} bytes against the declared "
        f"ceiling of {sp.MAX_INJECT_BYTES}")


def test_overflow_drops_the_oldest_entries_and_says_so(data_root):
    """Both halves: the OLDEST content goes — dropping the newest would discard
    "what I am doing next", the one line worth re-injecting — and the count of what
    went is reported, so a truncated history is never read as a complete one."""
    _fill("worker:s1", 40)
    body = sp.digest("worker:s1", sp.MAX_INJECT_BYTES)
    assert "entry 000" not in body and "entry 001" not in body, (
        "the oldest notes survived the ceiling; the drop is coming from the wrong end")
    assert "entry 039" in body, (
        "the newest note was dropped, which is the loss that makes the affordance "
        "useless")
    ent = sp.entries("worker:s1")
    lines = set(body.splitlines())
    kept = sum(1 for _, entry in ent if entry in lines)
    gone = sp.dropped_entries("worker:s1", sp.MAX_INJECT_BYTES)
    assert gone > 0, "the ceiling bit but the injector reports a complete scratchpad"
    assert kept == len(ent) - gone, (
        f"the meta says {gone} of {len(ent)} entries were dropped while the text "
        f"carries {kept}: the model is told the wrong history length")


def test_the_ceiling_holds_for_one_oversized_entry_too(data_root):
    """A bound a single write can exceed is a bound in name. The newest entry's tail
    survives, cut at the ceiling — the end of a note is where the next action is
    stated."""
    big = "z" * (sp.MAX_INJECT_BYTES * 3)
    sp.append("worker:s1", big)
    body = sp.digest("worker:s1", sp.MAX_INJECT_BYTES)
    assert len(body.encode("utf-8")) <= sp.MAX_INJECT_BYTES
    assert body == big[-len(body):], "the kept bytes are not the tail of the note"


def test_an_older_entry_that_only_fits_a_gap_is_dropped_whole(data_root):
    """The kept content is the newest contiguous suffix, not entries spliced around a
    hole. The gap case is live: the small old note and the small new note together
    fit under the ceiling, so a set-based selection would take the old one too — and
    a note that lost its middle is the note that read "ruled out X because Y, so Z"
    and now reads only "so Z"."""
    sp.append("worker:s1", "short old note about why X was ruled out")
    sp.append("worker:s1", "b" * (sp.MAX_INJECT_BYTES - 32))
    sp.append("worker:s1", "newest")
    body = sp.digest("worker:s1", sp.MAX_INJECT_BYTES)
    assert "newest" in body
    assert "short old note" not in body, (
        "an old entry was spliced in beside the big one, so the injection is a "
        "patchwork and not a contiguous tail")


def test_the_anchor_applies_the_ceiling_it_was_built_with_by_default(data_root):
    """The default is the whole safety property at the one call site that matters: if
    the injector had to pass the number, the site that forgot is the one that blows
    the prompt on the iteration it was meant to relieve."""
    _fill("worker:s1", 40)
    anchor = sp.build_scratchpad_anchor("worker:s1", max_turns=10)
    msgs = asyncio.run(anchor(10))          # the last iteration: every level is due
    assert msgs, "a scratchpad full of notes injected nothing at the turn cap"
    for msg in msgs:
        inner = msg["content"].split("<scratchpad>")[1].split("</scratchpad>")[0]
        assert len(inner.strip().encode("utf-8")) <= sp.MAX_INJECT_BYTES, (
            f"an injected body is {len(inner.encode('utf-8'))} bytes against the "
            f"declared ceiling of {sp.MAX_INJECT_BYTES}")


def test_a_nonpositive_ceiling_is_refused_rather_than_ignored(data_root):
    """Zero reads as "inject nothing" to one reader and "inject everything" to
    another — the worst possible ambiguity on the path whose job is to stop the
    prompt growing."""
    sp.append("worker:s1", "anything")
    with pytest.raises(sp.ScratchpadError):
        sp.digest("worker:s1", 0)


# ── clauses 1 and 2 at the seam: the tool the model actually calls ──────────


def test_the_tool_appends_for_the_bound_session_and_refuses_an_unbound_one(
        data_root, monkeypatch):
    """The process boundary this feature crosses. The handler runs in the MCP server
    process and takes its session from what the aggregator bound for the call
    (`META_SESSION_ID`, forwarded by the harness pool): the model has no session
    argument to type, which is what keeps clause 2 true for a caller that would
    otherwise name any id it liked."""
    monkeypatch.setattr(tool, "get_bound_session", lambda: "worker:s1")
    text, err = _call({"action": "append", "content": "hypothesis: kv gate"})
    assert not err, text
    assert "hypothesis: kv gate" in sp.read("worker:s1")
    reported = json.loads(text)
    assert reported["writes"] == 1, text
    assert reported["bytes"] == len("hypothesis: kv gate"), text

    text, err = _call({"action": "read"})
    assert not err and "hypothesis: kv gate" in text, text

    monkeypatch.setattr(tool, "get_bound_session", lambda: "")
    text, err = _call({"action": "append", "content": "should not land"})
    assert err and "should not land" not in sp.read("worker:s1"), text
    assert [p.name for p in sp.scratchpad_dir().iterdir()] == ["worker:s1.md"]


def test_the_tool_refuses_an_empty_append_and_an_unknown_action(data_root, monkeypatch):
    """A model that calls `append` with no text must not be able to conclude it
    recorded something, and `writes` must not move on a call that stored nothing —
    that count is the experiment's independent variable."""
    monkeypatch.setattr(tool, "get_bound_session", lambda: "worker:s1")
    text, err = _call({"action": "append", "content": "   "})
    assert err, text
    assert sp.stats("worker:s1") == {"writes": 0, "bytes": 0}
    text, err = _call({"action": "truncate"})
    assert err, text
    assert sp.stats("worker:s1") == {"writes": 0, "bytes": 0}


def test_the_tool_offers_only_read_and_append_and_is_advertised():
    """Two actions, and the module is listed now that the budget call is made.

    `append`/`read` only: an action that could replace or truncate the file is the
    clobbering clause 1 rules out.

    The other half of this node used to assert the module was NOT in
    `agent_mcp.main.MODULES`, because listing it would have put the internal tool
    catalog past its declared 22,500-token ceiling
    (`tests/test_mcp_layer.py::test_advertised_catalog_stays_under_its_token_ceiling`)
    and that ceiling's own comment recorded re-arming, capping or raising it as
    "all three human calls, recorded on item #1555". Item #1571 made the call by
    the third route — reclaiming description prose, the way `708ed204` closed
    #1555 — so the assertion is inverted and kept: it is the node that goes red if
    anyone un-lists the tool to buy back catalog tokens, which is the one way the
    ceiling must NOT be met.

    Measured on this tree: 22,261 estimated tokens over 104 tools without the
    module (22,433 before #1571 reclaimed 172 from eight tools' descriptions),
    22,429 over 105 with it, against the 22,500 ceiling.
    """
    import agent_mcp.main as M

    tools = asyncio.run(tool.list_tools())
    assert [t.name for t in tools] == ["Scratchpad"]
    enum = tools[0].input_schema["properties"]["action"]["enum"]
    assert enum == ["append", "read"], (
        f"the tool offers {enum}; an action that can replace or truncate the file "
        f"is the clobbering clause 1 rules out")
    assert tool in M.MODULES, (
        "Scratchpad is no longer advertised, so a turn cannot write the notes "
        "#1554's experiment measures. Meeting the catalog ceiling by un-listing a "
        "tool is not allowed — trim a description instead; see "
        "tests/test_mcp_layer.py's ceiling test and item #1571")


def test_a_worker_turn_reaches_the_handler_through_the_aggregator(data_root):
    """The boundary #1571 opened is `agent_mcp.main`, not this module.

    Listing the module is what made the tool callable, and everything between a
    turn's tool call and `call_tool` lives on the other side of that boundary: the
    dispatch table `tools/list` is built from, the `_meta` session binding the
    handler reads its address from (`main.py` sets the contextvar and resets it
    around the call), and the safety and effect gates that sit in front of the
    handler. Calling this module directly proves the file format and none of that,
    so this node calls the aggregator the way the pool does — session id travelling
    as the request's `_meta` — and then checks the bytes landed where
    `app.scratchpad.read`, which runs in the backend process in production, reads
    them. A module listed but unreachable from here, or reachable but writing to a
    path the injection never opens, fails this and nothing else.
    """
    import agent_mcp.main as M

    names = [t.name for t in asyncio.run(M.list_tools())]
    assert "Scratchpad" in names, (
        "the module is in MODULES but the aggregator does not list it")

    res = asyncio.run(M.call_tool(
        "Scratchpad", {"action": "append", "content": "crossed the aggregator seam"},
        meta={M.META_SESSION_ID: "worker:seam"}))
    assert not res.is_error, res.content[0].text
    payload = json.loads(res.content[0].text)
    assert "error" not in payload, payload
    assert payload["writes"] == 1 and payload["bytes"] > 0, payload
    assert "crossed the aggregator seam" in sp.read("worker:seam")

    res = asyncio.run(M.call_tool(
        "Scratchpad", {"action": "read"}, meta={M.META_SESSION_ID: "worker:seam"}))
    assert not res.is_error, res.content[0].text
    assert "crossed the aggregator seam" in json.loads(res.content[0].text)["content"]


def test_the_aggregator_refuses_an_unbound_scratchpad_call(data_root):
    """A `Scratchpad` call with no session in `_meta` never reaches a file.

    The handler refuses an anonymous call rather than inventing a bucket name,
    because an anonymous bucket is the cross-session leak the design exists to
    prevent — but the aggregator gets there first, and this is the check that the
    newly-advertised write tool is actually covered by it: a state-changing tool
    arriving with no session id is denied (`main.py:588`, "sessionless write:
    refused", #1053) rather than treated as "not sandboxed". A module sitting out
    of `MODULES` was never exposed to that gate; listing it put a file-writing tool
    in front of it, so the denial and the empty data root are both asserted here.
    """
    import agent_mcp.main as M

    res = asyncio.run(M.call_tool(
        "Scratchpad", {"action": "append", "content": "orphaned note"}))
    payload = json.loads(res.content[0].text)
    assert res.is_error, res
    assert "no session id" in payload["error"], payload
    assert payload["tool"] == "Scratchpad", payload
    assert not data_root.exists() or not any(data_root.rglob("*")), (
        "a denied call still put bytes somewhere")
