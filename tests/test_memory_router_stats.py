"""``/api/memory/stats`` — the segments it counts, and what a request has to pay to count them.

Two rulings live here.

#3160 settled *which* notes are counted. One hand-maintained list — seven names,
including `agents`, which is a directory in no vault and a collection in no qmd
config — served three promises at once in these routers: the `collections` of a qmd
query, the loop of the stats walk below, and the file count in Mission Control's
navigate brief. Each loop is guarded by `if not seg_dir.is_dir(): continue`, so the
phantom name was skipped without a trace and the five real segments the copy left
out (`architecture`, `lloyd`, `autonomy`, `backlog`, `people`) were missing from the
Memory tab's `docCount` exactly as they were missing from its search results. The
list is now the recall's own, imported rather than restated, so what the tab counts
and what its search box can reach are one set by construction.

#3503 settled what the request costs. Measured over the live vault on 2026-10-27:
21.7s for a cold `/api/memory/stats` on the running backend, 17.6s for the same
build out of process, 17.1s of it `yaml.safe_load` over 8,528 front-matter blocks,
`backlog/` alone 15.3s — against 0.036s for the stat-only signature walk over the
same 11 segments and 8,608 notes. `docCount` and `types` need no note opened, so
they are answered off the signature now; the tag tally, which does, is served from
the last completed build while a worker thread rebuilds it. Every clause of #3503
gets a node below, and the nodes that predate it are kept because the change is
exactly the sort that could put the walk back on the event loop or restate the
segment list.
"""
import asyncio
import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_mcp.vault import VAULT_SEGMENTS
from app.routers import memory as MR

# The two readings #3503 replaces the stale budget note with, in both the module
# and these tests: the cost of the parse, and the cost of the walk that replaced it
# on the request path.
PARSE_SECONDS_LIVE = "21.7s"
PARSE_SECONDS_OOP = "17.6s"
BACKLOG_SECONDS = "15.3s"
WALK_SECONDS = "0.036s"


def _fresh_stats_cache() -> dict:
    """An untouched `_STATS_CACHE`, so no earlier test's tally answers an assertion.

    Four keys: `sig` is the signature last built (or last attempted), `tally` the
    published tag count, `in_flight` the single-rebuild flag clause 4 is about, and
    `task` the strong reference that keeps a fire-and-forget task alive.
    """
    return {"sig": None, "tally": None, "in_flight": False, "task": None}


def _await_tally(tag_count=None, timeout: float = 10.0) -> dict:
    """Block until the background tag build publishes — optionally until THAT build.

    The rebuild is deliberately off the request path (#3503 clause 3), so a test
    that wants the tally waits for the worker, not for a response. Polling the cache
    keeps this a deadline rather than a guessed sleep; `tag_count` is for a test that
    already holds one tally and is waiting for the next one, which a plain
    "is it non-None" check would answer immediately with the stale value.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        tally = MR._STATS_CACHE["tally"]
        if tally is not None and (tag_count is None or tally["tagCount"] == tag_count):
            return tally
        time.sleep(0.02)
    raise AssertionError(
        f"no tally with tagCount={tag_count} within {timeout}s; last published is "
        f"{MR._STATS_CACHE['tally']!r} with in_flight={MR._STATS_CACHE['in_flight']}")


def _await_no_rebuild(timeout: float = 10.0) -> None:
    """Block until `in_flight` is clear. Clause 4's whole subject is this flag."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not MR._STATS_CACHE["in_flight"]:
            return
        time.sleep(0.02)
    raise AssertionError(
        f"`in_flight` was still set {timeout}s after a rebuild finished, so every "
        "later request skips the rebuild and the tag tally is wedged")


@pytest.fixture
def vault(tmp_path, monkeypatch):
    """A tmp vault holding one tagged `.md` per searchable segment.

    `_STATS_CACHE` is module state: without the reset a tally cached by an earlier
    test answers the assertion instead of the file just written — the same reset
    `test_memory_router_frontmatter.py`'s fixture makes, for the same reason.
    """
    root = tmp_path / "obsidian"
    for segment in VAULT_SEGMENTS:
        seg_dir = root / segment
        seg_dir.mkdir(parents=True)
        (seg_dir / "note.md").write_text(
            f"---\ntags: [{segment}-tag]\n---\n\n# {segment} note\n", encoding="utf-8")
    monkeypatch.setattr(MR, "_VAULT", root)
    monkeypatch.setattr(MR, "_STATS_CACHE", _fresh_stats_cache())
    return root


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(MR.router)
    with TestClient(app) as c:
        yield c


# --- #3160: which notes are counted -----------------------------------------


def test_doc_count_counts_backlog_and_architecture_notes(client, vault):
    """#3160 clause 3: `docCount` includes `.md` files under `backlog/` and
    `architecture/`, which the seven-name walk omitted.

    `types` is read as well as the total, because a total can be earned by a
    wrong walk: counting `backlog` twice and `architecture` not at all adds up
    the same while telling a different story about the vault.

    This one still answers on the FIRST request after #3503, which is the point of
    it surviving unchanged: the counts no longer wait for the tag build at all.
    """
    body = client.get("/api/memory/stats").json()
    assert body["types"]["backlog"] == 1, "a backlog note is invisible to the stats walk"
    assert body["types"]["architecture"] == 1, "an architecture note is invisible to it"
    assert body["docCount"] == len(VAULT_SEGMENTS), (
        f"`docCount` is {body['docCount']} over a vault of one note in each of the "
        f"{len(VAULT_SEGMENTS)} searchable segments; the walk's segment set is "
        f"{sorted(body['types'])}")


def test_stats_tags_come_from_the_segments_the_search_reaches(client, vault):
    """The same widened walk is what feeds `tagCount`: a note only `backlog`
    holds has to contribute its tag, or the Memory tab's tag list is a view of a
    subset it does not name.

    Asked for on the poll after the request that started the build, which is the
    shape the tab actually sees since #3503: counts first, tags one refresh later.
    """
    client.get("/api/memory/stats")
    _await_tally()
    body = client.get("/api/memory/stats").json()
    counts = {t["tag"]: t["count"] for t in body["topTags"]}
    assert body["tagCount"] == len(VAULT_SEGMENTS), (
        f"the walk saw tags from {sorted(counts)}, not one per searchable segment")
    for segment in ("backlog", "architecture"):
        assert counts.get(f"{segment}-tag") == 1, (
            f"the tag only a {segment}/ note carries is absent, so {segment} is not "
            "in the walked set")


def test_the_cold_stats_walk_does_not_hold_the_event_loop(monkeypatch, vault):
    """#3160's own cost, pinned — and #3503 moved the cost, so the pin moved too.

    The tag build over the eleven segments is 17.6s on the live vault (2026-10-27:
    8,608 notes, 15.3s of it `backlog/`). `memory_stats` is `async def`, so whatever
    it does inline it does on the event loop and every chat stream waits for it.
    This route now returns without doing the build at all, which means the original
    mechanism — tick a heartbeat across the route's own `await` — would have nothing
    to measure: the route's own span is the 0.036s walk, and five ticks in that
    window is not a bar. So the heartbeat runs across the whole hand-off: the
    request, then the wait for the worker's tally to appear.

    Same 0.3s blocking stand-in, same `ticks >= 5` bar, and that bar still fails the
    regression it was written for. A build that crept back inline would block the
    loop for 0.3s without yielding, so the beat could not reach five ticks; a build
    that crept back into the handler would additionally never publish while the poll
    is running, and the node would say so.
    """
    def slow_build():
        time.sleep(0.3)
        return {"docCount": 0, "tagCount": 7, "types": {}, "topTags": [],
                "lastRefresh": ""}

    monkeypatch.setattr(MR, "_build_stats_payload", slow_build)

    async def run():
        ticks = 0

        async def beat():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        beating = asyncio.create_task(beat())
        try:
            await MR.memory_stats()
            deadline = time.monotonic() + 10
            while MR._STATS_CACHE["tally"] is None:
                if time.monotonic() > deadline:
                    raise AssertionError(
                        "the handed-off tag build never published its tally, so the "
                        "route has nothing stale to serve and nothing off-loop either")
                await asyncio.sleep(0.02)
        finally:
            beating.cancel()
            try:
                await beating
            except asyncio.CancelledError:
                pass
        return ticks

    ticks = asyncio.run(run())
    assert ticks >= 5, (
        f"a heartbeat ticked {ticks} times across a 0.3s stats build, so the walk ran "
        "on the event loop — the real cold walk is 17.6s and every chat stream waits "
        "for it")


def test_the_walked_segment_set_is_the_recall_s():
    """The derivation itself, from the stats side: this route's walk list is
    `agent_mcp.vault.VAULT_SEGMENTS`, not a copy that has to be kept in step by
    hand. `app/routers/memory.py` and `app/routers/mc_ui.py` each carried such a
    copy until #3160."""
    assert list(MR._VAULT_SEGMENTS) == list(VAULT_SEGMENTS), (
        f"`memory.py` walks {list(MR._VAULT_SEGMENTS)} while the recall searches "
        f"{list(VAULT_SEGMENTS)}")


# --- #3503 clause 1: no note is read to answer the counts --------------------


@pytest.fixture
def vault_reads_denied(monkeypatch):
    """Make reading a note under the tmp vault an error, and record every attempt.

    Scoped to paths under `MR._VAULT` on purpose: `Path.read_text` is how pytest,
    Starlette and the import machinery read their own source during a request, and a
    blanket patch would break the harness instead of the route. `seen` doubles as the
    positive control — a guard that never fires proves nothing, so the node that
    asserts it is empty also asserts, through the real builder, that the same guard
    does fire on a genuine note read.
    """
    seen: list = []
    real = Path.read_text

    def guarded(self, *args, **kwargs):
        root = MR._VAULT.parts
        if self.parts[:len(root)] == root:
            seen.append(str(self))
            raise AssertionError(f"a stats request read a note: {self}")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    return seen


def test_the_stats_route_reads_no_note_to_answer_the_counts(client, vault, vault_reads_denied, monkeypatch):
    """#3503 clause 1: `docCount` and `types` come off `_stats_signature()`'s
    per-segment counts, so the request path opens no note file at all.

    `_kick_tag_tally_rebuild` is swapped for a recorder, for two reasons. The kicked
    build DOES read notes — that is its job — and letting it run concurrently would
    race its reads against the `seen == []` assertion that is about the request path
    only. And the recorder is itself the proof that the route still hands the parse
    off rather than quietly dropping the tally.
    """
    kicks: list = []
    monkeypatch.setattr(MR, "_kick_tag_tally_rebuild", lambda sig: kicks.append(sig))

    body = client.get("/api/memory/stats").json()

    assert vault_reads_denied == [], (
        f"the stats request read {len(vault_reads_denied)} note(s): {vault_reads_denied[:3]}. "
        "`docCount`/`types` are supposed to come from the signature walk, which stats "
        "files and never opens one")
    assert body["docCount"] == len(VAULT_SEGMENTS)
    assert body["types"]["backlog"] == 1 and body["types"]["architecture"] == 1
    assert len(kicks) == 1, (
        f"a cold signature queued {len(kicks)} tag rebuilds; clause 3 wants the parse "
        "handed off, clause 4's flag wants exactly one in flight")
    assert len(kicks[0]) == len(VAULT_SEGMENTS)

    # Positive control, through the real builder: the same guard fires on reads the
    # route no longer makes. It swallows the AssertionError the way it swallows any
    # unreadable note, so the evidence is the recorded paths, not a raise — and the
    # count is the builder's own walk of this vault.
    MR._build_stats_payload()
    assert len(vault_reads_denied) == len(VAULT_SEGMENTS), (
        f"the guard recorded {len(vault_reads_denied)} reads from a builder that walks "
        f"one note in each of {len(VAULT_SEGMENTS)} segments, so it was not recording "
        "and the empty assertion above proved nothing")


# --- #3503 clause 2: the signature-derived map equals the walked one ----------


def test_the_signature_derived_map_equals_the_walked_one(client, vault):
    """`types`/`docCount` served off the signature, byte for byte the map the
    note-reading build computes.

    Both are produced in the same test rather than compared to a constant, because
    the property is the agreement: a per-segment count walked by `stat()` and a
    per-segment count obtained by reading each file have to answer the same question
    about the same vault, and the day they don't is the day the Memory tab starts
    reporting a segment it cannot show.
    """
    served = client.get("/api/memory/stats").json()
    walked = MR._build_stats_payload()

    assert served["types"] == walked["types"], (
        f"the signature-derived map {served['types']} disagrees with the walked map "
        f"{walked['types']} over the same vault")
    assert served["docCount"] == walked["docCount"]
    assert served["types"]["backlog"] == 1 and served["types"]["architecture"] == 1
    assert served["docCount"] == len(VAULT_SEGMENTS)


def test_a_segment_directory_that_is_missing_stays_absent_from_types(client, vault):
    """The shape both maps agree on for a segment the vault does not have: ABSENT.

    `_build_stats_payload` has always `continue`d before `types[seg] = count`, so it
    omits the segment; `_stats_signature()` records `(seg, 0, 0.0, False)` for it. A
    `types` map derived from the signature would therefore gain a `0` entry the
    walked map does not emit, and `0` reads as "this segment is empty" when the truth
    is "this segment is not mounted" — the silent-zero shape #3160 is about. The
    seeded fixture creates all eleven directories, which is why this node removes
    one: without it neither map's treatment of a missing segment is tested at all.
    """
    missing = "architecture"
    (vault / missing / "note.md").unlink()
    (vault / missing).rmdir()

    served = client.get("/api/memory/stats").json()
    walked = MR._build_stats_payload()

    assert missing not in walked["types"], "the walked map changed its shape; re-read this clause"
    assert missing not in served["types"], (
        "`types` serves a 0 for a directory that does not exist, which the Memory tab "
        "renders as an empty segment rather than an absent one")
    assert served["docCount"] == walked["docCount"] == len(VAULT_SEGMENTS) - 1


def test_a_note_that_cannot_be_statted_is_counted_as_a_note_that_is_read(client, vault, monkeypatch):
    """The one way the two walks could part, closed: a file `stat()` refuses.

    `_stats_signature()` catches `OSError` around the stat because a note may be
    deleted between the listing and the stat — under a vault that agents write into,
    that is a live window, not a theoretical one. Counting it only after the stat
    succeeds would make the signature report `backlog: 0` for a note the builder
    counts as `backlog: 1`, and the tab's number would move depending on which walk
    lost the race.
    """
    doomed = vault / "backlog" / "note.md"
    real_stat = Path.stat

    def flaky_stat(self, *args, **kwargs):
        if self == doomed:
            raise OSError(2, "No such file or directory", str(self))
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", flaky_stat)

    sig_count = MR._counts_from_signature(MR._stats_signature())[1].get("backlog")
    walked_count = MR._build_stats_payload()["types"].get("backlog")

    assert walked_count == 1, "the builder stopped counting an existing note; the fixture is the suspect"
    assert sig_count == walked_count, (
        f"a note whose stat fails is counted {sig_count} by the signature walk and "
        f"{walked_count} by the read walk")


# --- #3503 clause 3: the tally is served stale while it rebuilds -------------


def test_the_first_request_answers_zeros_and_the_tally_arrives_after_it(client, vault):
    """Zeros on the first build ever — and the counts on that same response are
    already this vault's truth.

    That asymmetry is the design: `docCount` cannot be stale because it is computed
    for this request, while the tag tally has to be 0 until a build lands. A fresh
    backend therefore opens its Memory tab with the right document count and a tag
    count that fills in on the next refresh.
    """
    first = client.get("/api/memory/stats").json()
    assert first["tagCount"] == 0, "a cold process reported a tag count it never built"
    assert first["topTags"] == []
    assert first["docCount"] == len(VAULT_SEGMENTS), (
        "the counts waited for the tag build; clause 1 says they never do")
    assert first["types"]["backlog"] == 1

    _await_tally()
    second = client.get("/api/memory/stats").json()
    assert second["tagCount"] == len(VAULT_SEGMENTS)
    assert second["docCount"] == len(VAULT_SEGMENTS)


def test_a_stale_signature_gets_the_previous_tally_while_the_build_is_parked(client, vault, monkeypatch):
    """A request that finds a stale signature is answered from the last completed
    tally, not from the parse — pinned without a wall-clock threshold.

    The build is parked on an event nothing releases but this test, so a route that
    waited for the parse could not pass an assertion about latency; instead it hangs
    until a watchdog releases the park eight seconds in and then fails the equality
    below with the tally it was not supposed to have. Eight seconds is a watchdog,
    not a threshold: nothing here is graded on how fast the correct path runs.

    The response under test carries the PREVIOUS tally and the CURRENT counts at
    once (11 tags, 12 notes), which is clauses 1 and 3 read together: the counts
    never lag, the tally does.
    """
    enter = threading.Event()
    release = threading.Event()
    real = MR._build_stats_payload

    def parked():
        enter.set()
        release.wait(timeout=30)
        return real()

    client.get("/api/memory/stats")
    _await_tally()
    assert MR._STATS_CACHE["in_flight"] is False, "the first build never finished; fixture broken"

    monkeypatch.setattr(MR, "_build_stats_payload", parked)
    (vault / "backlog" / "extra.md").write_text(
        "---\ntags: [extra-tag]\n---\n\n# Extra\n", encoding="utf-8")
    threading.Timer(8.0, release.set).start()

    body = client.get("/api/memory/stats").json()

    # The park is kicked BY the request above, so `enter` can only be observed after
    # it: if the route had waited for the parse, the line above would still be
    # sitting in `release.wait` until the watchdog, and `body` would carry a tally
    # this response was never allowed to have.
    assert enter.wait(timeout=10), "the rebuild for the new signature never started"
    assert MR._STATS_CACHE["in_flight"] is True, "the flag is clear while a build is parked"
    assert body["tagCount"] == len(VAULT_SEGMENTS), (
        f"a stale signature served tagCount={body['tagCount']}; it must serve the last "
        f"completed tally ({len(VAULT_SEGMENTS)}) and let the rebuild land later")
    assert body["docCount"] == len(VAULT_SEGMENTS) + 1, (
        "the counts went stale with the tally; they come from this request's walk")

    release.set()
    _await_tally(len(VAULT_SEGMENTS) + 1)
    assert client.get("/api/memory/stats").json()["tagCount"] == len(VAULT_SEGMENTS) + 1
    assert MR._STATS_CACHE["in_flight"] is False


# --- #3503 clause 4: a raising build cannot wedge the tally ------------------


def test_a_rebuild_that_raises_clears_the_in_flight_flag_and_is_retried(client, vault, monkeypatch):
    """The flag is released in a `finally`, so one unreadable vault is a bad refresh
    and not a permanently wrong Memory tab.

    Without the `finally` the first build's exception leaves `in_flight` set, every
    later request takes the "already rebuilding" branch, and `tagCount` stays at the
    zeros of a cold process for the life of the server. The node therefore checks the
    flag, then checks the only thing clearing it is for: that a later build actually
    runs and lands.
    """
    calls = {"n": 0}
    real = MR._build_stats_payload

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated unreadable vault")
        return real()

    monkeypatch.setattr(MR, "_build_stats_payload", flaky)

    body = client.get("/api/memory/stats").json()
    assert body["tagCount"] == 0
    _await_no_rebuild()
    assert MR._STATS_CACHE["in_flight"] is False, (
        "a raising rebuild left `in_flight` set, so no tally is ever rebuilt again")
    assert MR._STATS_CACHE["tally"] is None
    assert calls["n"] == 1

    (vault / "backlog" / "later.md").write_text(
        "---\ntags: [late-tag]\n---\n\n# Later\n", encoding="utf-8")
    body = client.get("/api/memory/stats").json()
    assert body["docCount"] == len(VAULT_SEGMENTS) + 1, (
        "the counts stopped moving after a failed build; they are independent of it")
    _await_tally(len(VAULT_SEGMENTS) + 1)
    assert calls["n"] == 2, (
        f"the tally was not retried on the next signature change ({calls['n']} builds)")
    assert client.get("/api/memory/stats").json()["tagCount"] == len(VAULT_SEGMENTS) + 1
    assert MR._STATS_CACHE["in_flight"] is False


# --- #3503 clause 5: the stale budget note is gone from the module -----------


def test_the_module_source_names_neither_stale_note():
    """Neither place in `app/routers/memory.py` still asks the closed question.

    #3503 names two sites carrying the old wording — the block above `_STATS_CACHE`
    and the `_build_stats_payload` docstring — and the second one is easy to miss
    because it restates the first. So this reads the MODULE, not one region: both
    phrases are asserted absent wherever they sit, and the 2026-10-27 readings that
    replaced them are asserted present, which is what stops the node passing on an
    editor who deleted the paragraph and nothing else.
    """
    src = Path(MR.__file__).read_text(encoding="utf-8")

    for stale in ("1.9s", "17.0s", "is itself acceptable", "the ruling #3160 owes"):
        assert stale not in src, (
            f"`{stale}` is still in app/routers/memory.py: the cold-walk budget note "
            "that #3503 is to replace, which a later pass would re-ask")
    for reading in (PARSE_SECONDS_LIVE, PARSE_SECONDS_OOP, BACKLOG_SECONDS, WALK_SECONDS, "2026-10-27"):
        assert reading in src, (
            f"the module does not state the 2026-10-27 reading `{reading}`; the "
            "replacement has to carry the measurements, not just drop the old ones")
