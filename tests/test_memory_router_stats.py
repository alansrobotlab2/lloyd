"""``/api/memory/stats`` walks the segments the memory search can reach (#3160).

One hand-maintained list — seven names, including `agents`, which is a
directory in no vault and a collection in no qmd config — served three promises
at once in these routers: the `collections` of a qmd query, the loop of the
stats walk below, and the file count in Mission Control's navigate brief. Each
loop is guarded by `if not seg_dir.is_dir(): continue`, so the phantom name was
skipped without a trace and the five real segments the copy left out
(`architecture`, `lloyd`, `autonomy`, `backlog`, `people`) were missing from the
Memory tab's `docCount` exactly as they were missing from its search results.

The list is now the recall's own, imported rather than restated, so what the tab
counts and what its search box can reach are one set by construction. These
tests are the pin on that property from the stats side.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_mcp.vault import VAULT_SEGMENTS
from app.routers import memory as MR


@pytest.fixture
def vault(tmp_path, monkeypatch):
    """A tmp vault holding one tagged `.md` per searchable segment.

    `_STATS_CACHE` is module state: without the reset a payload cached by an
    earlier call answers the assertion instead of the file just written — the
    same reset `test_memory_router_frontmatter.py`'s fixture makes, for the same
    reason.
    """
    root = tmp_path / "obsidian"
    for segment in VAULT_SEGMENTS:
        seg_dir = root / segment
        seg_dir.mkdir(parents=True)
        (seg_dir / "note.md").write_text(
            f"---\ntags: [{segment}-tag]\n---\n\n# {segment} note\n", encoding="utf-8")
    monkeypatch.setattr(MR, "_VAULT", root)
    monkeypatch.setattr(MR, "_STATS_CACHE", {"sig": None, "payload": None})
    return root


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(MR.router)
    with TestClient(app) as c:
        yield c


def test_doc_count_counts_backlog_and_architecture_notes(client, vault):
    """#3160 clause 3: `docCount` includes `.md` files under `backlog/` and
    `architecture/`, which the seven-name walk omitted.

    `types` is read as well as the total, because a total can be earned by a
    wrong walk: counting `backlog` twice and `architecture` not at all adds up
    the same while telling a different story about the vault.
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
    subset it does not name."""
    body = client.get("/api/memory/stats").json()
    counts = {t["tag"]: t["count"] for t in body["topTags"]}
    assert body["tagCount"] == len(VAULT_SEGMENTS), (
        f"the walk saw tags from {sorted(counts)}, not one per searchable segment")
    for segment in ("backlog", "architecture"):
        assert counts.get(f"{segment}-tag") == 1, (
            f"the tag only a {segment}/ note carries is absent, so {segment} is not "
            "in the walked set")


def test_the_cold_stats_walk_does_not_hold_the_event_loop(monkeypatch, vault):
    """#3160's own cost, pinned. Widening this walk to the eleven segments made
    the cold build 17.0s over the live vault (8,157 notes, 107.6 MB read and
    YAML-parsed) where the six directories the seven-name copy found cost 2.4s
    (4,753 notes, 40.3 MB) — both measured 2026-10-22. `memory_stats` is an
    `async def`, so whatever it does inline it does on the event loop: at 17.0s a
    cold cache would stop every chat stream for the length of the walk, so the
    build has to leave the loop.

    Measured rather than grepped for `to_thread`: the builder is swapped for a
    0.3s blocking sleep and a heartbeat coroutine ticks in the same loop while
    the route runs. Ticks during the walk are the loop getting its turn back; a
    build that runs inline never yields, so the count stays at zero.
    """
    import asyncio
    import time

    def slow_build():
        time.sleep(0.3)
        return {"docCount": 0, "tagCount": 0, "types": {}, "topTags": [],
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
            await asyncio.wait_for(MR.memory_stats(), timeout=10)
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
        "on the event loop — the real cold walk is 17.0s and every chat stream waits "
        "for it")


def test_the_walked_segment_set_is_the_recall_s():
    """The derivation itself, from the stats side: this route's walk list is
    `agent_mcp.vault.VAULT_SEGMENTS`, not a copy that has to be kept in step by
    hand. `app/routers/memory.py` and `app/routers/mc_ui.py` each carried such a
    copy until #3160."""
    assert list(MR._VAULT_SEGMENTS) == list(VAULT_SEGMENTS), (
        f"`memory.py` walks {list(MR._VAULT_SEGMENTS)} while the recall searches "
        f"{list(VAULT_SEGMENTS)}")
