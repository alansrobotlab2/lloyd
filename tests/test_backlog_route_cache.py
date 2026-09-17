"""One `/api/backlog/tasks` request parses the corpus at most once, cold, and
not at all when the cache is warm.

Item #1199, causes #1 and #3. The measured shape on the live box 2026-09-16:
reading all 11 MB of `~/obsidian/backlog` costs 0.033 s, and **one** YAML
frontmatter pass over its 1,137 files costs 1.78 s. So the time was parsing,
not I/O — and `/tasks` did it twice (`_backlog_board_map()` at
`app/routers/backlog.py:131`, then its own loop at `:140`) while `/boards` did
it a third time at `:102`. Because the board filter is applied *after* parsing,
server cost was flat and independent of what was asked for: `?board_id=1`, one
task and 2.3 KB of response, still cost 3.65 s.

The fix is a cache keyed on `(path, mtime_ns, size)`, not an in-process memo of
results. The key is the load-bearing part: there are at least three writers to
this directory (`app/routers/backlog.py`, `agent_mcp/backlog.py`,
`agent-services/guardian/notify.py`), so an entry that outlives the file it
came from would serve a stale row to the board *and* back through
`task-update`, where a stale read becomes a stale **write**. Validating stat on
every lookup means a file touched by any writer — any process — is re-parsed on
the route's next pass, which is also why the cache is safe without the watchdog.

`task-update` deliberately reads *through* the cache (`_backlog_parse_fm`
directly): it is about to rewrite the file, so it must not build the new
version out of a snapshot another writer moved.

Everything here is counted through a pass counter wrapped around
`_backlog_parse_fm` — the function that reads a file and parses it — so the
assertions are about real parse work, not about the cache's own bookkeeping.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import pytest
import yaml

from app.routers import backlog as BR


def write_item(d: Path, item_id: int, *, board: str = "lloyd", status: str = "draft",
               body: str = "Do the thing.") -> Path:
    p = d / f"{item_id}-item-{item_id}.md"
    fm = {
        "type": "backlog", "segment": "backlog", "status": status,
        "priority": "low", "board": board, "blocked": False,
        "assigned": False, "position": item_id * 1000,
    }
    p.write_text(
        f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n# Item {item_id}\n\n{body}\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """6 conformant items, a cleared cache, and a counter in front of the parser."""
    d = tmp_path / "backlog"
    d.mkdir()
    for i in range(1, 5):
        write_item(d, i, board="lloyd")
    write_item(d, 5, board="alfie")
    write_item(d, 6, board="alfie", status="done")
    monkeypatch.setattr(BR, "_BACKLOG_DIR", d)
    monkeypatch.setattr(BR, "_FM_CACHE", {})

    calls: list[str] = []
    real = BR._backlog_parse_fm

    def counting_parse(path):
        calls.append(str(path))
        return real(path)

    monkeypatch.setattr(BR, "_backlog_parse_fm", counting_parse)
    return {"dir": d, "calls": calls}


def n_parses(corpus) -> int:
    return len(corpus["calls"])


def files_on_disk(corpus) -> int:
    return len(list(corpus["dir"].glob("*.md")))


def cached_body(corpus, item_id: int) -> str:
    """The body sitting in the cache for one item's file."""
    path = next(corpus["dir"].glob(f"{item_id}-*.md"))
    # entry = (mtime_ns, size, fm, body)
    return BR._FM_CACHE[str(path)][3]


# ── clause 5 ─────────────────────────────────────────────────────────────────

def test_cold_tasks_request_parses_the_corpus_exactly_once(corpus):
    """Was 2 passes here and a 3rd in `/boards`: 3 x 1.78 s of the 4.2 s."""
    json.loads(bytes(BR.backlog_tasks().body))
    assert n_parses(corpus) == files_on_disk(corpus), (
        f"{n_parses(corpus)} parses for {files_on_disk(corpus)} files: the board "
        "map and the task loop are scanning separately again"
    )


def test_warm_tasks_request_parses_nothing(corpus):
    BR.backlog_tasks()
    corpus["calls"].clear()
    rows = json.loads(bytes(BR.backlog_tasks().body))
    assert n_parses(corpus) == 0, f"warm request still parsed {n_parses(corpus)} files"
    assert len(rows) == files_on_disk(corpus), "the warm request returned different rows"


def test_boards_and_tasks_share_one_scan(corpus):
    """/boards then /tasks over the same corpus: the second request is free."""
    BR.backlog_boards()
    boards_pass = n_parses(corpus)
    assert boards_pass == files_on_disk(corpus), "boards itself should be one pass"
    corpus["calls"].clear()
    BR.backlog_tasks()
    assert n_parses(corpus) == 0, "boards left the scan uncached, so tasks re-parsed"


def test_a_filtered_request_scans_once_cold_and_not_at_all_warm(corpus):
    """The `?board_id=1` row in the item: 1 task, 2.3 KB out, 3.65 s — cost was
    flat because the filter ran *after* the parse. That ordering stays (the board
    map has to come from the same scan or it costs another pass), but the scan is
    now one pass cold and nothing warm."""
    board_id = BR._backlog_board_map()["alfie"]  # warms the cache
    corpus["calls"].clear()

    rows = json.loads(bytes(BR.backlog_tasks(board_id=str(board_id)).body))
    assert sorted(r["id"] for r in rows) == [5, 6]
    assert n_parses(corpus) == 0, "a warm filtered request re-parsed the corpus"

    BR._FM_CACHE.clear()
    corpus["calls"].clear()
    BR.backlog_tasks(board_id=str(board_id))
    assert n_parses(corpus) == files_on_disk(corpus), (
        f"{n_parses(corpus)} parses for one cold filtered request over "
        f"{files_on_disk(corpus)} files: it is scanning the corpus more than once"
    )


def test_writing_one_file_re_parses_only_that_file(corpus):
    """The key is (path, mtime_ns, size): a write elsewhere must not invalidate
    the corpus, and a write to a file must not leave its own stale copy behind."""
    BR.backlog_tasks()
    assert n_parses(corpus) == files_on_disk(corpus)
    corpus["calls"].clear()

    target = next(corpus["dir"].glob("3-*.md"))
    text = target.read_text(encoding="utf-8")
    target.write_text(text.replace("Do the thing.", "Do a different thing, now."),
                      encoding="utf-8")
    assert target.stat().st_size != len(text), "fixture must change the size too"

    rows = {r["id"]: r for r in json.loads(bytes(BR.backlog_tasks().body))}
    assert n_parses(corpus) == 1, (
        f"{n_parses(corpus)} files re-parsed after one write; the key is not (path, mtime_ns, size)"
    )
    assert corpus["calls"] == [str(target)]
    assert "different thing" in rows[3]["description_snippet"], (
        "the changed file was re-parsed but the row still shows the old body"
    )


def test_a_new_file_and_a_deleted_file_are_both_seen(corpus):
    BR.backlog_tasks()
    corpus["calls"].clear()

    write_item(corpus["dir"], 7, board="lloyd", body="brand new item")
    rows = {r["id"]: r for r in json.loads(bytes(BR.backlog_tasks().body))}
    assert 7 in rows, "a file created after the cache filled is invisible"
    assert n_parses(corpus) == 1

    gone = next(corpus["dir"].glob("2-*.md"))
    gone.unlink()
    corpus["calls"].clear()
    rows = json.loads(bytes(BR.backlog_tasks().body))
    assert 2 not in {r["id"] for r in rows}
    assert len(rows) == files_on_disk(corpus) == 6
    # A vanished file must not linger in the cache forever.
    assert str(gone) not in BR._FM_CACHE, "the cache grew on every delete"


def test_a_same_size_rewrite_is_re_parsed_because_mtime_ns_is_in_the_key(corpus):
    """A status flip often keeps the byte count, so size alone cannot be the
    key — this proves `mtime_ns` is in it and not decoration next to the size."""
    BR.backlog_tasks()
    corpus["calls"].clear()
    target = next(corpus["dir"].glob("4-*.md"))
    before = target.read_text(encoding="utf-8")
    # "Do the thing." -> "Do the thing!!": 13 bytes in, 15 out is size, so use
    # an equal-length swap instead and let mtime be the only signal.
    assert len("\nDo the thing.\n") == len("\nDo the thinG.\n")
    target.write_text(before.replace("\nDo the thing.\n", "\nDo the thinG.\n"),
                      encoding="utf-8")
    os.utime(target, ns=(target.stat().st_atime_ns, target.stat().st_mtime_ns + 10**9))
    assert target.stat().st_size == len(before)

    BR.backlog_tasks()
    assert n_parses(corpus) == 1, (
        "a same-size rewrite was served from cache: mtime_ns is not in the key"
    )
    assert "Do the thinG." in cached_body(corpus, 4)


def test_the_scan_hands_each_path_to_the_row_builder_exactly_once_per_pass(corpus):
    """The property "at most one parse pass per request" belongs to, stated where
    it is written.

    `_backlog_scan` yields `(path, fm, body)` exactly once per file on disk — so no
    caller, however it filters or re-loops, can end up parsing one file twice in a
    request. Directory order is `glob`'s and is not asserted: the board sorts rows
    itself, and asserting an order the scan never promised would make this test
    fail for the wrong reason the day someone adds a sort.
    """
    out = list(BR._backlog_scan())
    paths = [str(path) for path, _fm, _body in out]
    assert len(paths) == len(set(paths)) == files_on_disk(corpus), (
        f"{len(out)} entries for {files_on_disk(corpus)} files: the scan either "
        "skipped a file or yielded one twice"
    )
    assert all(BR._BACKLOG_PATTERN.match(path.name) for path, _fm, _b in out), (
        "the scan yielded a file the item-id pattern rejects"
    )


# ── the cache under concurrent threadpool requests ───────────────────────────
#
# Both list handlers are plain `def`, so FastAPI runs each request in its own
# threadpool worker and every worker mutates the one module-global `_FM_CACHE`. The
# review rung named the hazard on 2026-09-17: `_backlog_cached_fm` inserted while
# `_prune_fm_cache` iterated the same dict. The symptom is not a stale row — it is
# `RuntimeError: dictionary changed size during iteration` escaping `_backlog_scan`
# into a 500 on the board, raised only when a create, a nightly write and the page's
# 15-second poll overlap. Three tests, covering the three things that have to be
# true: the mutation sites hold the lock (mechanism), the lock really excludes
# (instrument), and the prune under a real hammer raises nowhere while a replica of
# the shipped shape raises under the identical hammer (symptom, with its positive
# control).


class _RecordingLock:
    """A lock stand-in that counts acquisitions.

    It does not exclude — the stress test below uses the production lock — it exists
    so a test can ask "did this mutation happen between acquire and release?", which
    is the property being claimed, rather than whether two threads happened to
    interleave on this run of the machine.
    """

    def __init__(self):
        self.acquisitions = 0

    def __enter__(self):
        self.acquisitions += 1
        return self

    def __exit__(self, *_exc):
        return False


def test_every_cache_mutation_happens_inside_the_lock(tmp_path, monkeypatch):
    """The finding, stated as the property that fixes it.

    Each mutation site is driven through its own production entry point — a scan (the
    insert in `_backlog_cached_fm` plus the sweep in `_prune_fm_cache`), a save and a
    delete (both through `_fm_cache_invalidate`) — and the assertion is that the
    instrumented lock's counter *moved* while that call ran. A mutation that skipped
    the lock leaves the counter where it was, which is the failure being prevented,
    not a number that has to be exceeded.
    """
    d = tmp_path / "backlog"
    d.mkdir()
    for i in (1, 2):
        write_item(d, i, board="lloyd")
    monkeypatch.setattr(BR, "_BACKLOG_DIR", d)
    monkeypatch.setattr(BR, "_FM_CACHE", {})
    rec = _RecordingLock()
    monkeypatch.setattr(BR, "_FM_CACHE_LOCK", rec)

    before = rec.acquisitions
    BR.backlog_tasks()                       # cold scan: two inserts + a sweep
    assert rec.acquisitions > before, "a scan touched the cache without the lock"

    path = d / "1-item-1.md"
    assert str(path) in BR._FM_CACHE, "prerequisite: both entries cached"
    before = rec.acquisitions
    BR._fm_cache_invalidate(path)
    assert rec.acquisitions > before, "_fm_cache_invalidate popped without the lock"
    assert str(path) not in BR._FM_CACHE, "the invalidate did not drop the entry"

    (d / "2-item-2.md").unlink()
    (d / "1-item-1.md").write_text(         # a second write makes the file grow again
        (d / "1-item-1.md").read_text(encoding="utf-8") + "\nmore\n", encoding="utf-8")
    before = rec.acquisitions
    BR.backlog_tasks()
    assert rec.acquisitions > before, "the post-write rescan touched the cache without the lock"


def test_the_cache_lock_actually_excludes():
    """`_FM_CACHE_LOCK` must be a real lock, or the test above pins a ceremony.

    The instrumented lock never blocks, so "ran inside the lock" is only worth
    something if the production object excludes a second holder. It must also not be
    re-entrant: the mutation sites are three statements each, and a re-entrant lock
    would let a nested call believe it was exclusive when it was not.
    """
    assert isinstance(BR._FM_CACHE_LOCK, type(threading.Lock())), (
        f"_FM_CACHE_LOCK is a {type(BR._FM_CACHE_LOCK).__name__}, not a threading.Lock"
    )
    assert BR._FM_CACHE_LOCK.acquire(blocking=False), "could not take an idle lock"
    try:
        assert not BR._FM_CACHE_LOCK.acquire(blocking=False), (
            "_FM_CACHE_LOCK is re-entrant or shared: a nested mutation would not be "
            "excluded, so holding it would not make the prune safe"
        )
    finally:
        BR._FM_CACHE_LOCK.release()


def _legacy_prune(cache: dict, found: list) -> None:
    """The prune this route shipped before #1199's review, kept as the control.

    Iterate the live mapping to pick victims, pop them after, hold nothing. If the
    hammer below cannot make *this* raise, it is too narrow to certify the production
    prune, and a green second half would be reporting a race it never had a chance to
    see.
    """
    if len(cache) <= len(found):
        return
    live = {str(f) for f, _, _ in found}
    for key in [k for k in cache if k not in live]:
        cache.pop(key, None)


def _hammer(cache: dict, lock, prune, seeds: list, *, writers: int = 4,
            trials: int = 3, prunes_per_trial: int = 20) -> list:
    """Run `prune` while `writers` insert into the same mapping. Returns escapes.

    The writers take `lock`, because that is the production invariant: every mutator
    of `_FM_CACHE` holds it (`_backlog_cached_fm`'s insert, `_fm_cache_invalidate`,
    this sweep). The legacy prune does not know the lock exists, which is precisely
    the difference the fix is. `sys.setswitchinterval` is dropped so the GIL hands
    over inside the comprehension; at the default 5 ms the window closes and the test
    passes for the wrong reason.
    """
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    errors: list[BaseException] = []
    stop = threading.Event()

    def writer(tid: int) -> None:
        i = 0
        while not stop.is_set():
            with lock:
                cache[f"/tmp/board/{tid}-{i}.md"] = (0, 0, {}, "")
            i += 1
            if i % 3 == 0:
                with lock:
                    cache.pop(f"/tmp/board/{tid}-{i - 2}.md", None)

    try:
        for _ in range(trials):
            cache.clear()
            for s in seeds:
                cache[s] = (0, 0, {}, "")
            stop.clear()
            threads = [threading.Thread(target=writer, args=(t,)) for t in range(writers)]
            for t in threads:
                t.start()
            try:
                # Each sweep drops the newest 50+k seeds from the live set, which is
                # what a delete looks like to the prune: cache longer than the disk.
                for k in range(prunes_per_trial):
                    prune([(Path(p), {}, "") for p in seeds[: len(seeds) - 50 - k]])
            except BaseException as exc:  # noqa: BLE001 - escapes are the measurement
                errors.append(exc)
            stop.set()
            for t in threads:
                t.join(10)
    finally:
        sys.setswitchinterval(previous)
    return errors


def test_the_shipped_prune_shape_reproduces_the_race_and_the_fixed_one_does_not():
    """The symptom, on both sides of the fix, under one identical hammer.

    1,200 cached entries with 4 inserting writer threads, 3 trials x 20 sweeps.
    Measured on 2026-09-17: the legacy shape raised `dictionary changed size during
    iteration` on every sweep it had to do (60 of 60), the production
    `_prune_fm_cache` raised nothing. The first assertion is a positive control on
    the instrument, not on the code — if it ever fails, widen the hammer before
    trusting the second assertion at all.
    """
    seeds = [f"/tmp/board/seed-{n}.md" for n in range(1200)]
    cache: dict = {}

    control = _hammer(cache, threading.Lock(),
                      lambda found: _legacy_prune(cache, found), seeds)
    assert control, (
        "the legacy prune did NOT raise under this stress, so the hammer cannot "
        "certify the fix — widen it (more entries, more writers) before reading the "
        "second half as a pass"
    )
    assert any("dictionary changed size" in str(e) for e in control), [str(e) for e in control[:2]]

    production = _hammer(
        cache, BR._FM_CACHE_LOCK,
        lambda found: BR._prune_fm_cache(found), seeds)
    assert not production, [str(e) for e in production[:3]]


def test_the_still_shipped_prune_is_the_one_under_test(tmp_path, monkeypatch):
    """The hammer calls `_prune_fm_cache`, so this pins what *it* must keep doing.

    Sweeping in place while holding the lock is the fix; the alternative reading of
    "don't iterate a shared dict" is to rebuild the global onto a new dict, which
    would leave every other holder of the old object reading a stale mapping and would
    make the hammer above measure a dict nothing else can see. So: after a real
    delete, the eviction has to be visible through `BR._FM_CACHE` itself, and the
    object identity has to be the one the module was imported with.
    """
    d = tmp_path / "backlog"
    d.mkdir()
    for i in (1, 2, 3):
        write_item(d, i, board="lloyd")
    monkeypatch.setattr(BR, "_BACKLOG_DIR", d)
    live_cache: dict = {}
    monkeypatch.setattr(BR, "_FM_CACHE", live_cache)

    BR.backlog_tasks()
    assert len(live_cache) == 3, "prerequisite: three entries cached"
    gone = d / "2-item-2.md"
    gone.unlink()
    BR.backlog_tasks()
    assert BR._FM_CACHE is live_cache, (
        "_prune_fm_cache rebound the global instead of sweeping it in place"
    )
    assert str(gone) not in live_cache, sorted(live_cache)
    assert len(live_cache) == 2, sorted(live_cache)


def test_task_update_reads_the_file_fresh_not_from_the_cache(corpus):
    """The stale-read that becomes a stale write: the cache must never be the
    source for a route about to rewrite the file."""
    import asyncio

    BR.backlog_tasks()  # fill the cache with the pre-edit bytes
    corpus["calls"].clear()

    path = corpus["dir"] / "1-item-1.md"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("Do the thing.", "Changed by another writer."),
                    encoding="utf-8")

    async def post():
        class R:
            async def json(self):
                return {"id": 1, "priority": "high"}
        return await BR.backlog_task_update(R())

    asyncio.run(post())
    after = path.read_text(encoding="utf-8")
    assert "Changed by another writer." in after, (
        "task-update rebuilt the file from a cached snapshot and lost the "
        "other writer's edit"
    )
