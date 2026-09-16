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
