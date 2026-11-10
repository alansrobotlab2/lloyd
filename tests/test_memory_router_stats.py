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
import json
import re
import subprocess
import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_mcp.vault import VAULT_SEGMENTS
# The walk counts the vault's own directories; since 2026-11-10 the recall's
# list can also name `lloyd-architecture`, which lives in the checkout.
from agent_mcp.vault import VAULT_DIR_SEGMENTS
from app.routers import memory as MR

# The two readings #3503 replaces the stale budget note with, in both the module
# and these tests: the cost of the parse, and the cost of the walk that replaced it
# on the request path.
PARSE_SECONDS_LIVE = "21.7s"
PARSE_SECONDS_OOP = "17.6s"
BACKLOG_SECONDS = "15.3s"
WALK_SECONDS = "0.036s"

# #3682's floor, restated here rather than imported from the module: the item's ruling
# is five minutes, so the number the tests backdate past is that ruling, and a module
# that renamed the constant or changed its value fails a node instead of agreeing with
# the test that reads it from it.
FLOOR_SECONDS = 300.0
FLOOR_MINUTES = 5


def _fresh_stats_cache() -> dict:
    """An untouched `_STATS_CACHE`, so no earlier test's tally answers an assertion.

    Five keys: `sig` is the signature last built (or last attempted), `tally` the
    published tag count, `tally_built_at` the monotonic instant the last tally
    COMPLETED (#3682's floor reads it; None until one lands), `in_flight` the
    single-rebuild flag clause 4 is about, and `task` the strong reference that keeps a
    fire-and-forget task alive.

    This dict is a hand-copy of module state that the `vault` fixture installs in
    PLACE of the module's, so a key the module gains and this helper lacks is not a
    harmless omission — the route reads it and raises `KeyError` inside the handler.
    The key set is pinned against the module by
    `test_the_cache_reset_helpers_carry_every_key_the_module_ships`.
    """
    return {"sig": None, "tally": None, "tally_built_at": None, "in_flight": False, "task": None}


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


def _expire_tally_stamp(age_seconds: float = FLOOR_SECONDS + 1.0) -> None:
    """Backdate the tally's completion stamp so #3682's floor has elapsed.

    The stamp is module state that `_rebuild_tag_tally` writes on its success path, so
    backdating it is the same lever the floor itself reads: no injected clock, no
    patched predicate, and the route still does its own arithmetic against
    `time.monotonic()`. Callers that need to know the backdate took assert
    `not MR._tag_tally_is_rate_limited()` afterwards — the alternative, asserting on
    the number passed in here, can only ever agree with itself.

    The default is one second past the floor rather than a huge age on purpose: at
    `age = FLOOR_SECONDS + 1` a floor mistyped LARGER than the five minutes the item
    rules (3,000s, say) leaves the tally rate-limited and the node fails, whereas a
    wildly backdated stamp would clear any floor and never notice the typo.
    """
    MR._STATS_CACHE["tally_built_at"] = time.monotonic() - age_seconds


@pytest.fixture
def vault(tmp_path, monkeypatch):
    """A tmp vault holding one tagged `.md` per searchable segment.

    `_STATS_CACHE` is module state: without the reset a tally cached by an earlier
    test answers the assertion instead of the file just written — the same reset
    `test_memory_router_frontmatter.py`'s fixture makes, for the same reason.
    """
    root = tmp_path / "obsidian"
    for segment in VAULT_DIR_SEGMENTS:
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
    assert body["docCount"] == len(VAULT_DIR_SEGMENTS), (
        f"`docCount` is {body['docCount']} over a vault of one note in each of the "
        f"{len(VAULT_DIR_SEGMENTS)} searchable segments; the walk's segment set is "
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
    assert body["tagCount"] == len(VAULT_DIR_SEGMENTS), (
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
    assert body["docCount"] == len(VAULT_DIR_SEGMENTS)
    assert body["types"]["backlog"] == 1 and body["types"]["architecture"] == 1
    assert len(kicks) == 1, (
        f"a cold signature queued {len(kicks)} tag rebuilds; clause 3 wants the parse "
        "handed off, clause 4's flag wants exactly one in flight")
    assert len(kicks[0]) == len(VAULT_DIR_SEGMENTS)

    # Positive control, through the real builder: the same guard fires on reads the
    # route no longer makes. It swallows the AssertionError the way it swallows any
    # unreadable note, so the evidence is the recorded paths, not a raise — and the
    # count is the builder's own walk of this vault.
    MR._build_stats_payload()
    assert len(vault_reads_denied) == len(VAULT_DIR_SEGMENTS), (
        f"the guard recorded {len(vault_reads_denied)} reads from a builder that walks "
        f"one note in each of {len(VAULT_DIR_SEGMENTS)} segments, so it was not recording "
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
    assert served["docCount"] == len(VAULT_DIR_SEGMENTS)


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
    assert served["docCount"] == walked["docCount"] == len(VAULT_DIR_SEGMENTS) - 1


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
    assert first["docCount"] == len(VAULT_DIR_SEGMENTS), (
        "the counts waited for the tag build; clause 1 says they never do")
    assert first["types"]["backlog"] == 1

    _await_tally()
    second = client.get("/api/memory/stats").json()
    assert second["tagCount"] == len(VAULT_DIR_SEGMENTS)
    assert second["docCount"] == len(VAULT_DIR_SEGMENTS)


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

    #3682 moved this node's premise, not its assertion. It completes a tally and then,
    milliseconds later, needs a rebuild to start — which is exactly what the 5-minute
    floor suppresses — so the completion stamp is backdated past the floor first. That
    is the honest reading of what the node is about: a parked build and a stale
    signature, with the floor already elapsed. The alternative was deleting the node,
    and its assertion (stale signature → previous tally, response not held by the
    parse) is the thing #3503 clause 3 exists to guarantee.
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
    _expire_tally_stamp()
    assert not MR._tag_tally_is_rate_limited(), (
        "the backdated stamp did not put the tally past the floor, so the request below "
        "would be rate-limited rather than parked, and a failure here would read as a "
        "broken rebuild path instead of a fixture that never armed the floor")

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
    assert body["tagCount"] == len(VAULT_DIR_SEGMENTS), (
        f"a stale signature served tagCount={body['tagCount']}; it must serve the last "
        f"completed tally ({len(VAULT_DIR_SEGMENTS)}) and let the rebuild land later")
    assert body["docCount"] == len(VAULT_DIR_SEGMENTS) + 1, (
        "the counts went stale with the tally; they come from this request's walk")

    release.set()
    _await_tally(len(VAULT_DIR_SEGMENTS) + 1)
    assert client.get("/api/memory/stats").json()["tagCount"] == len(VAULT_DIR_SEGMENTS) + 1
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
    assert MR._STATS_CACHE["tally_built_at"] is None, (
        "the failure path wrote a completion stamp, so a build that published nothing "
        "would arm #3682's floor and the retry asserted below would be rate-limited "
        "away — the one way this node could stop kicking and still look green")
    assert calls["n"] == 1

    (vault / "backlog" / "later.md").write_text(
        "---\ntags: [late-tag]\n---\n\n# Later\n", encoding="utf-8")
    body = client.get("/api/memory/stats").json()
    assert body["docCount"] == len(VAULT_DIR_SEGMENTS) + 1, (
        "the counts stopped moving after a failed build; they are independent of it")
    _await_tally(len(VAULT_DIR_SEGMENTS) + 1)
    assert calls["n"] == 2, (
        f"the tally was not retried on the next signature change ({calls['n']} builds)")
    assert client.get("/api/memory/stats").json()["tagCount"] == len(VAULT_DIR_SEGMENTS) + 1
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


# --- #3682: the floor under the rebuild rate ----------------------------------
#
# The hand-off of #3503 bounds a request's LATENCY and never bounded the WORK. The
# kick captured its signature at kick time and stored it on completion, but the vault
# moves during the ~17-20s parse, so every finished tally came back already stale and
# the next request re-kicked it. Measured 2026-11-01: 20s of polling
# `/api/memory/stats` burned 2143 CPU ticks (~1.07 cores pinned) against 72 idle, and a
# GET concurrent with a rebuild took 12.19s against a 0.070s baseline; an in-process
# replay of five sequential requests against a vault that moves during each parse ran
# five kicks and five builds and left `stored_sig_is_current=False` after every one.
# After the floor those same five requests are one build.
#
# Each node below is one clause of #3682, and the burst shape is deliberate: a single
# rate-limited request would also pass a floor keyed on the wrong clock or a floor that
# only ever arms once.


def test_the_floor_is_the_five_minutes_the_item_rules():
    """The floor's value, pinned apart from any behaviour node.

    It lives apart on purpose: an assertion about a constant at the top of a behaviour
    node fails FIRST and hides the behaviour assertion underneath it, so a mutation
    that removes the floor would be reported as "the number is wrong" and the burst
    assertion — the one that says whether the CPU burn is fixed — would never have run.
    """
    assert MR.TAG_TALLY_FLOOR_SECONDS == FLOOR_SECONDS == FLOOR_MINUTES * 60.0, (
        f"the floor is {MR.TAG_TALLY_FLOOR_SECONDS}s, not the {FLOOR_MINUTES} minutes "
        f"({FLOOR_SECONDS}s) #3682 rules")


def test_a_stale_signature_inside_the_floor_serves_the_previous_tally_and_kicks_nothing(client, vault, monkeypatch):
    """#3682 clause 1: five stale-signature requests, one build, and the counts move.

    The vault is written before EVERY request, which is what makes each signature
    genuinely stale and is also what the live box does constantly — 91 note writes
    across 66 distinct 20-second buckets in the 6h before triage. The tally published
    by the first build is therefore always older than the vault by the time the next
    request looks, and the only thing that may stop that request buying another
    17.6s parse is the floor.

    Two counters, and both have to be reachable: the kick recorder is a DELEGATING
    wrapper (it records and then calls the real kick), so if the floor were absent the
    real rebuild would run and `builds` would grow too. A non-delegating recorder would
    make the builder count unreachable — four requests that never reach the builder
    because the test itself disconnected it — and the docstring's claim that builds and
    kicks cross-check each other would be false, which is what the first review of this
    round caught.
    """
    builds: list = []
    real_build = MR._build_stats_payload
    monkeypatch.setattr(MR, "_build_stats_payload",
                        lambda: (builds.append(len(builds) + 1), real_build())[1])

    client.get("/api/memory/stats")      # the first build ever: never rate-limited
    _await_tally()
    assert builds == [1], f"the first-ever request produced {builds}; fixture broken"

    # Recorded AFTER the first build, which has to be a real one: the floor is only
    # meaningful against a tally that actually completed. Delegating, not replacing —
    # see the docstring — so `builds == [1]` below can actually fail.
    kicks: list = []
    real_kick = MR._kick_tag_tally_rebuild
    monkeypatch.setattr(MR, "_kick_tag_tally_rebuild",
                        lambda sig: (kicks.append(sig), real_kick(sig))[1])

    bodies = []
    for i in range(4):
        (vault / "backlog" / f"churn{i}.md").write_text(
            f"---\ntags: [churn-{i}]\n---\n\n# churn {i}\n", encoding="utf-8")
        bodies.append(client.get("/api/memory/stats").json())

    assert kicks == [], (
        f"a burst of 4 stale-signature requests reached the kick site {len(kicks)} "
        "times; the floor has to decide at the decision site, or every request still "
        "queues a parse and the CPU burn this item is about is unchanged")
    assert builds == [1], (
        f"the burst cost {len(builds)} parses of the vault; the item's whole claim is "
        "that one build answers a burst")
    assert MR._STATS_CACHE["in_flight"] is False, (
        "a build is still running after the burst went unanswered, so the tally the "
        "burst served was not the last completed one")
    for i, body in enumerate(bodies, start=1):
        assert body["tagCount"] == len(VAULT_DIR_SEGMENTS), (
            f"request {i} served tagCount={body['tagCount']} instead of the previous "
            f"tally ({len(VAULT_DIR_SEGMENTS)}); rate-limiting the kick must not rate-limit "
            "what is served")
        assert body["docCount"] == len(VAULT_DIR_SEGMENTS) + i, (
            f"request {i} served docCount={body['docCount']} on a vault holding "
            f"{len(VAULT_DIR_SEGMENTS) + i} notes; the counts come off this request's walk, "
            "never off the lagged tally")
    assert bodies[-1]["lastRefresh"], "the served tally lost its completion stamp"


def test_a_stale_signature_past_the_floor_kicks_exactly_one_rebuild_and_the_next_request_serves_it(client, vault, monkeypatch):
    """#3682 clause 2: the floor delays the rebuild, it does not cancel it.

    Two requests after the stamp is backdated: the one that finds the floor elapsed
    starts the rebuild and is still answered from the previous tally, the one right
    behind it must not start a second. Then the tally the parked build publishes is
    what the following request serves — which is the difference between a floor and a
    freeze.
    """
    builds: list = []
    real_build = MR._build_stats_payload
    monkeypatch.setattr(MR, "_build_stats_payload",
                        lambda: (builds.append(len(builds) + 1), real_build())[1])

    client.get("/api/memory/stats")
    _await_tally()
    assert builds == [1], f"the first build did not run; fixture broken ({builds})"

    (vault / "backlog" / "extra.md").write_text(
        "---\ntags: [extra-tag]\n---\n\n# Extra\n", encoding="utf-8")
    _expire_tally_stamp()

    body = client.get("/api/memory/stats").json()
    assert body["tagCount"] == len(VAULT_DIR_SEGMENTS), (
        f"the request that starts the rebuild served tagCount={body['tagCount']}; it "
        "waits for nothing, so it serves the last completed tally")
    assert body["docCount"] == len(VAULT_DIR_SEGMENTS) + 1
    client.get("/api/memory/stats")      # arrives while the rebuild is running or after

    _await_tally(len(VAULT_DIR_SEGMENTS) + 1)
    assert builds == [1, 2], (
        f"a stale signature past the floor produced {builds}: clause 2 wants exactly "
        "one rebuild for it, and a floor that swallowed this kick would leave the tag "
        "count wrong for the life of the process")
    assert client.get("/api/memory/stats").json()["tagCount"] == len(VAULT_DIR_SEGMENTS) + 1, (
        "the rebuild published a tally no request ever served")
    assert MR._STATS_CACHE["in_flight"] is False


def test_the_floor_never_rate_limits_before_a_tally_has_ever_landed(client, vault, monkeypatch):
    """#3682 clause 3: `tally` is None means nothing is rate-limited, ever.

    The floor's key is the completion stamp, so this seeds a stamp with NO tally behind
    it — the state the failure path would leave if the stamp were written there too —
    and asserts the route still treats it as a first build. A floor keyed on the stamp
    alone would park a freshly started backend on `tagCount: 0` for five minutes and
    swallow the retry of a build that raised while nothing had ever published.

    The retry half of the clause is the raising build below: it must leave the stamp
    where it found it, release `in_flight`, and run again on the next signature change
    despite the arming stamp seeded here. (`test_a_rebuild_that_raises_clears_the_in_flight_flag_and_is_retried`
    carries the same rule against a None stamp, which is how it was before #3682.)
    """
    calls = {"n": 0}
    real_build = MR._build_stats_payload

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated unreadable vault")
        return real_build()

    monkeypatch.setattr(MR, "_build_stats_payload", flaky)

    armed_stamp = time.monotonic()
    MR._STATS_CACHE["tally_built_at"] = armed_stamp
    assert MR._STATS_CACHE["tally"] is None, "fixture broken: a tally is already cached"
    assert MR._tag_tally_is_rate_limited() is False, (
        "the floor read a completion stamp with no tally behind it, so a backend that "
        "has never published a tally serves zeros until the floor expires")

    first = client.get("/api/memory/stats").json()
    assert first["tagCount"] == 0, "the first-ever request did not answer zeros"
    assert first["docCount"] == len(VAULT_DIR_SEGMENTS), "the counts waited on the tally"
    _await_no_rebuild()
    assert calls["n"] == 1, f"the first-ever build was not kicked ({calls['n']} calls)"
    assert MR._STATS_CACHE["tally_built_at"] == armed_stamp, (
        "the failure path moved the completion stamp, so a build that published "
        "nothing arms the floor and the retry below stops happening")

    (vault / "backlog" / "later.md").write_text(
        "---\ntags: [late-tag]\n---\n\n# Later\n", encoding="utf-8")
    second = client.get("/api/memory/stats").json()
    assert second["docCount"] == len(VAULT_DIR_SEGMENTS) + 1
    _await_tally(len(VAULT_DIR_SEGMENTS) + 1)
    assert calls["n"] == 2, (
        f"a raising first build was not retried on the next signature change even "
        f"though the stamp seeded here is milliseconds old ({calls['n']} calls): the "
        "floor reached a state with no tally")
    assert client.get("/api/memory/stats").json()["tagCount"] == len(VAULT_DIR_SEGMENTS) + 1
    assert MR._STATS_CACHE["in_flight"] is False


def test_a_rate_limited_response_still_answers_the_counts_without_reading_a_note(client, vault, vault_reads_denied, monkeypatch):
    """#3682 clause 4: the rate-limited response is the same route, not a staler one.

    `docCount` and `types` are served off `_stats_signature()` on this path exactly as
    on a fresh one — the floor only ever declines to QUEUE a parse, and if it were
    implemented by short-circuiting the whole handler it would be caught here by counts
    that stopped moving, and by the note-reading guard, which is the same guard
    #3503 clause 1 uses and is proven live in this node's positive control.
    """
    real_build = MR._build_stats_payload
    # A stand-in builder, so the tally lands without tripping the read guard that is
    # the subject of the assertion below: the guard is about the REQUEST path, and the
    # real builder reads notes by design (the control at the end proves it fires).
    monkeypatch.setattr(MR, "_build_stats_payload", lambda: {
        "docCount": 0, "tagCount": 7, "types": {}, "topTags": [],
        "lastRefresh": "tally-built-before-the-vault-moved"})

    client.get("/api/memory/stats")
    _await_tally()
    assert MR._STATS_CACHE["tally"]["tagCount"] == 7, "the stand-in build never landed"

    # Silenced only now, because the build above has to be a real kick: the floor this
    # node is about applies to a tally that completed, and completes only through
    # `_rebuild_tag_tally` writing its stamp.
    kicks: list = []
    monkeypatch.setattr(MR, "_kick_tag_tally_rebuild", lambda sig: kicks.append(sig))

    (vault / "backlog" / "extra.md").write_text(
        "---\ntags: [extra-tag]\n---\n\n# Extra\n", encoding="utf-8")

    body = client.get("/api/memory/stats").json()
    assert kicks == [], "a rate-limited request still queued a rebuild"
    assert body["tagCount"] == 7 and body["lastRefresh"] == "tally-built-before-the-vault-moved", (
        "the lagged tally is supposed to be what this response serves; something else "
        "answered it")
    assert body["docCount"] == len(VAULT_DIR_SEGMENTS) + 1, (
        f"docCount stayed at {body['docCount']} on a vault holding "
        f"{len(VAULT_DIR_SEGMENTS) + 1} notes: the counts were served from the tally")
    assert body["types"]["backlog"] == 2, (
        f"`types` says backlog={body['types']['backlog']} after a backlog note was "
        "written while the tally sat inside its floor")
    assert vault_reads_denied == [], (
        f"the rate-limited request read {len(vault_reads_denied)} note(s): "
        f"{vault_reads_denied[:3]}. Declining to rebuild is not a licence to serve the "
        "counts from the cached payload")

    real_build()
    assert len(vault_reads_denied) == len(VAULT_DIR_SEGMENTS) + 1, (
        f"the guard recorded {len(vault_reads_denied)} reads from a builder that walks "
        f"{len(VAULT_DIR_SEGMENTS) + 1} notes here (one per segment plus the `extra.md` "
        "this node wrote), so it was not recording and the empty assertion above proved "
        "nothing")


def test_the_cache_reset_helpers_carry_every_key_the_module_ships():
    """The two hand-copied `_STATS_CACHE` resets are complete against the module.

    Both test files REPLACE the module's dict with their own literal, so a key
    `_STATS_CACHE` grows that a literal lacks is not decoration left behind: the route
    reads `tally_built_at` on every stats request, so the reset that lacks it turns
    every node in that file into a `KeyError` from inside the handler. #3682 is exactly
    the change that adds such a key, which is why the check exists now.
    """
    from tests.test_memory_router_frontmatter import _fresh_stats_cache as _frontmatter_reset

    shipped = set(MR._STATS_CACHE)
    assert shipped == {"sig", "tally", "tally_built_at", "in_flight", "task"}, (
        f"`_STATS_CACHE` ships {sorted(shipped)}; update this node and both reset "
        "helpers together, in that order")
    for name, reset in (("test_memory_router_stats.py", _fresh_stats_cache()),
                        ("test_memory_router_frontmatter.py", _frontmatter_reset())):
        assert set(reset) == shipped, (
            f"{name}'s `_fresh_stats_cache` holds {sorted(reset)} but the module ships "
            f"{sorted(shipped)}; the fixture installs its copy in place of the module's, "
            "so the route under test would raise KeyError on the missing key")


# The clause-5 spec lives in TypeScript, where the page's own logic has to be tested.
ROOT = Path(__file__).resolve().parents[1]
REFETCH_SPEC = ROOT / "web" / "src" / "components" / "pages" / "memoryStatsRefresh.test.ts"
REFETCH_HELPER = REFETCH_SPEC.with_name("memoryStatsRefresh.ts")


def test_the_refetch_spec_is_a_spec_this_suite_actually_runs():
    """#3682 clause 5, named from the runner the suite and the review rung use.

    Clause 5's behaviour is TypeScript, and `web/src/components/pages/memoryStatsRefresh.test.ts`
    is where it is actually pinned. That file is invisible to a pytest node id: the
    review rung's own evidence rail decides "is this a test file?" through
    `scripts/automod/testpaths.py::is_test_file`, which can only ever answer about `.py`,
    so a clause whose only node is a `.test.ts` cannot be cited as met by anything that
    reads node ids — which is how this round's first review graded a file it had just run
    13 tests through as `partial` and downgraded it. The precedent for answering that way
    is `tests/test_dashboard_doc_claims.py::test_the_section_error_helpers_have_a_spec_the_suite_actually_runs`:
    this node runs the same binary over the same spec and reports vitest's own test names,
    so the behaviour stays pinned by a node this suite executes and the two runners cannot
    disagree about what `scheduleStatsRefetch` schedules.

    What is asserted HERE is the clause: one re-fetch on an empty tally, none on a landed
    one, and a delay at or beyond the cold build's cost. What stays in the spec is the
    detail (timer identity, unmount, the failed-fetch path). And as in that precedent, an
    absent toolchain is a skip that names the command — `web/node_modules` is untracked and
    `tests/conftest.py` links the parent checkout's in — while an absent SPEC is a failure,
    because then there is nothing to run and green here would be about nothing.
    """
    assert REFETCH_SPEC.is_file(), (
        f"{REFETCH_SPEC} is missing — clause 5's spec is the whole point of this node, so "
        "its absence is a failure and not a skip")
    assert REFETCH_HELPER.is_file(), (
        f"{REFETCH_HELPER} is missing, so the page has no helper to call")

    spec = REFETCH_SPEC.read_text(encoding="utf-8")
    for claim in ("scheduleStatsRefetch(COLD, schedule, apply)",
                  "scheduleStatsRefetch(FRESH, schedule, apply)",
                  "STATS_REFETCH_DELAY_MS"):
        assert claim in spec, (
            f"the spec no longer exercises `{claim}`, so there is nothing here for this "
            "node to run on the clause's behalf")

    web = ROOT / "web"
    vitest = web / "node_modules" / ".bin" / "vitest"
    if not vitest.exists():
        pytest.skip(
            f"{vitest} is not installed — run `npm install` in ~/lloyd/web, then "
            "`cd ~/lloyd/web && npx vitest run src/components/pages/"
            "memoryStatsRefresh.test.ts`. tests/conftest.py links the parent checkout's "
            "node_modules into this tree, and the gate's frontend rung runs the same "
            "spec, so this skip means neither of those had a toolchain to offer.")

    from app.lint_findings import node_env  # the env the gate's rung builds

    rel = str(REFETCH_SPEC.relative_to(web))
    proc = subprocess.run(
        [str(vitest), "run", rel, "--reporter=json", "--silent"],
        cwd=str(web), env=node_env(), capture_output=True, text=True, timeout=300)
    payload, start, end = proc.stdout, proc.stdout.find("{"), proc.stdout.rfind("}")
    assert start >= 0 and end > start, (
        f"vitest produced no JSON report for {rel} (rc={proc.returncode}): "
        f"{(proc.stdout + proc.stderr)[-600:]}")
    report = json.loads(payload[start:end + 1])
    results = [t for f in report.get("testResults", []) for t in f.get("assertionResults", [])]
    names = sorted(t["title"] for t in results)
    print(f"vitest {rel}: {len(names)} tests — {names}")
    assert results, f"vitest reported zero tests for {rel}, so a green here proves nothing"
    failed = sorted(t["title"] for t in results if t.get("status") not in ("passed", "pending"))
    assert proc.returncode == 0 and not failed, (
        f"vitest failed the clause-5 spec: {failed or proc.returncode}")

    # The spec must still pin the two halves of the clause and the delay, so a spec
    # rewritten down to nothing cannot pass by being emptied. Clause 5 is one re-fetch on
    # an empty `lastRefresh`, none on a landed one, and no sooner than the cold build.
    joined = " ".join(names).lower()
    for behaviour in ("exactly one re-fetch", "schedules nothing at all", "cold response"):
        assert behaviour in joined, (
            f"no spec test covers {behaviour!r} any more, so clause 5's "
            f"{behaviour!r} half has nothing pinning it")
    helper = REFETCH_HELPER.read_text(encoding="utf-8")
    delay = re.search(r"STATS_REFETCH_DELAY_MS\s*=\s*([0-9_]+)", helper)
    assert delay, "the helper no longer declares STATS_REFETCH_DELAY_MS as a literal"
    delay_ms = int(delay.group(1).replace("_", ""))
    assert delay_ms >= 21_700, (
        f"the re-fetch fires at {delay_ms}ms, under the 21.7s cold build measured on the "
        "live backend 2026-10-27 — it would draw a second empty tally and spend the only "
        "retry this helper is allowed")
