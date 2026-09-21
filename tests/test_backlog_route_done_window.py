"""`?done_since=` cuts *done* rows out of the backlog list payload by date.

Item #1213, the front-end half of the board-legibility work #1199 started.
Measured on the live box 2026-09-17: `GET /api/backlog/tasks` returned 1,152
rows, 679 of them `status: done` — 59 % of the payload goes to a column nobody
reads. #1199 fixed what it *costs* to produce those rows (9.2 MB → 920 KB);
this fixes what a person has to scroll. Alan's ruling, same day: the window is
7 days, and the front end asks for it by default.

The three things these tests pin, one per clause, are the three ways a naive
version of this filter is wrong:

* **the date it judges by.** A done row is dated by `completed:`, else
  `updated:`/`updated_at:`, else the file's mtime. `completed:` first because a
  reclosed item's `updated:` is its last touch, not its last completion — so a
  filter built on the row's `updated_at` silently re-dates reopened work. The
  fallbacks are not decoration either: of the 679 done rows on this board today,
  230 have no `completed:` key at all and 23 more carry the literal string
  `None` under it, so 253 of 679 (37 %) can only be dated from something else.
* **which rows it touches.** Only `status: done`. An `up_next` item from March
  is still work, and hiding it would be a data-loss bug wearing a filter.
* **what it does with a bad date.** Nothing. The board polls this route every
  15 seconds, so a 400 or a 500 on one malformed parameter is an outage, not a
  validation message.

`completed: None` is the trap the parse-based check exists for. `None` is not a
YAML null (`null`, `~` and empty are), so those 23 files parse to the *string*
`'None'` — truthy, present, and not a date. `"completed" in fm` accepts every
one of them and hides 23 items; `_fm_date` rejects them and they fall through to
`updated:`.

The last section is why the window can be trusted at all: a row closed through
Mission Control used to have no `completed:` key, because the route wrote
`fm["status"]` and nothing else, so its position in this window was whatever
date last touched it (#1023). Those tests pin the route's writes, through the
same HTTP client these do.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

import pytest
import yaml

from app.routers import backlog as BR


NO_KEY = object()


def write_item(
    d: Path,
    item_id: int,
    *,
    status: str = "draft",
    board: str = "lloyd",
    completed=NO_KEY,
    updated=NO_KEY,
    mtime: datetime | None = None,
) -> Path:
    """One backlog file with its date fields chosen by the caller.

    `NO_KEY` leaves a key out of the front matter entirely, which is a different
    thing from `None` (a key present with a null value) and a third thing from
    the string `"None"` — and all three exist on the real board today, so all
    three have to be expressible here.
    """
    p = d / f"{item_id}-item-{item_id}.md"
    fm: dict = {
        "type": "backlog", "segment": "backlog", "status": status,
        "priority": "medium", "board": board, "blocked": False,
        "assigned": False, "position": item_id * 1000,
    }
    if completed is not NO_KEY:
        fm["completed"] = completed
    if updated is not NO_KEY:
        fm["updated"] = updated
    p.write_text(
        f"---\n{yaml.dump(fm, default_flow_style=False, sort_keys=False)}---\n\n"
        f"# Item {item_id}\n\nBody {item_id}.\n",
        encoding="utf-8",
    )
    if mtime is not None:
        ts = mtime.timestamp()
        os.utime(p, (ts, ts))
    return p


@pytest.fixture
def backlog_dir(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(BR, "_BACKLOG_DIR", d)
    monkeypatch.setattr(BR, "_FM_CACHE", {})
    return d


@pytest.fixture
def client(backlog_dir):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(BR.router)
    with TestClient(app) as c:
        yield c


def ids(rows: list[dict]) -> set[int]:
    return {r["id"] for r in rows}


def get(client, **params) -> tuple[int, list[dict]]:
    """One `GET /api/backlog/tasks`, as the query string would arrive from a browser.

    `urlencode`, not a hand-joined string: the search case sends a value with a
    space in it, and a hand-joined `?q=Body 2` is a different request.
    """
    qs = urlencode(params)
    resp = client.get("/api/backlog/tasks" + (f"?{qs}" if qs else ""))
    return resp.status_code, json.loads(resp.content)


# A cut-off in the middle of the fixture dates, so "before" and "not before"
# are both non-trivially answered by every case below.
CUTOFF = "2026-09-10"
OLD = "2026-03-01T12:00:00"
NEW = "2026-09-15T12:00:00"


# ── clause 1: the parameter filters the payload, its absence does not ─────────

def test_done_since_omits_old_done_rows_and_absence_returns_everything(client, backlog_dir):
    write_item(backlog_dir, 1, status="done", completed=NEW)
    write_item(backlog_dir, 2, status="done", completed=OLD)

    code, unfiltered = get(client)
    code_after, filtered = get(client, done_since=CUTOFF)

    assert code == 200 and code_after == 200
    assert ids(unfiltered) == {1, 2}, "without the parameter nothing is cut"
    assert ids(filtered) == {1}, (
        "a done row completed 2026-03-01 must not survive a 2026-09-10 window"
    )


def test_a_done_row_dated_on_the_cut_off_itself_is_kept(client, backlog_dir):
    """`done_since` is "since", so the boundary date is inside the window."""
    write_item(backlog_dir, 1, status="done", completed="2026-09-10T00:00:00")
    write_item(backlog_dir, 2, status="done", completed="2026-09-09T23:59:59")

    _, rows = get(client, done_since=CUTOFF)

    assert ids(rows) == {1}, "only a date *before* 2026-09-10 falls outside the window"


def test_the_cut_happens_in_the_payload_not_only_in_the_view(client, backlog_dir):
    """The clause is about the response, so the row must be gone from the JSON.

    A view-only filter (the state the board was in before this item: the client
    had no date predicate at all and rendered every row the route returned)
    would keep the same bytes on the wire and fail here.
    """
    for i in range(1, 21):
        write_item(backlog_dir, i, status="done",
                   completed=NEW if i % 2 else OLD)

    _, unfiltered = get(client)
    _, filtered = get(client, done_since=CUTOFF)

    assert len(unfiltered) == 20
    assert len(filtered) == 10
    assert all("2026-03-01" not in json.dumps(r) for r in filtered)


# ── clause 2: which date a done row is judged by ────────────────────────────

def test_completed_wins_over_updated_and_mtime(client, backlog_dir):
    """A reclosed item is judged by when it was closed, not when it was touched.

    `updated:` is recent and `completed:` is old here: judging by the row's
    `updated_at` — the shape `web/src/components/pages/BacklogPage.tsx` already
    receives — would keep this item, which is exactly the wrong answer.
    """
    write_item(backlog_dir, 1, status="done", completed=OLD, updated=NEW)

    _, rows = get(client, done_since=CUTOFF)

    assert ids(rows) == set()


def test_updated_is_the_fallback_when_completed_is_absent(client, backlog_dir):
    """230 of the 679 done rows on the live board have no `completed:` key."""
    write_item(backlog_dir, 1, status="done", completed=None, updated=NEW,
               mtime=datetime(2020, 1, 1))
    write_item(backlog_dir, 2, status="done", updated=OLD,
               mtime=datetime(2026, 9, 15))

    _, rows = get(client, done_since=CUTOFF)

    assert ids(rows) == {1}, (
        "item 1's `completed:` is null, so `updated:` judges it: recent, so kept. "
        "Item 2 has no `completed:` and an old `updated:`, so it is cut — and its "
        "recent mtime must not rescue it, because `updated:` outranks mtime"
    )


def test_a_literal_none_completed_falls_through_to_updated(client, backlog_dir):
    """The 23 `completed: None` files are a string, and must count as absent.

    `None` is not a YAML null token, so `yaml.safe_load` hands back `'None'`. A
    key-presence check accepts it, finds no date, and drops the row; the
    parse-based check falls through to `updated:`, which is recent here.
    """
    write_item(backlog_dir, 1, status="done", completed="None", updated=NEW)

    _, rows = get(client, done_since=CUTOFF)

    assert ids(rows) == {1}, (
        "the literal string 'None' under `completed:` is not a date and must "
        "fall back to `updated:`"
    )


def test_the_mtime_is_the_last_resort_for_a_row_with_no_dates(client, backlog_dir):
    """A file with neither key is dated by its own mtime, in both directions."""
    write_item(backlog_dir, 1, status="done", mtime=datetime(2026, 9, 15, 9, 0))
    write_item(backlog_dir, 2, status="done", mtime=datetime(2026, 3, 1, 9, 0))

    _, rows = get(client, done_since=CUTOFF)

    assert ids(rows) == {1}


def test_an_iso_date_with_an_offset_is_judged_by_its_own_calendar_date(client, backlog_dir):
    """Real front matter carries tz-aware stamps; comparing must not throw.

    `_row_from` emits these rows' `created_at`/`updated_at` as strings, so the
    window is what first parses them back.
    """
    write_item(backlog_dir, 1, status="done", completed="2026-09-16T23:30:00+00:00")
    write_item(backlog_dir, 2, status="done", completed="2026-09-09T23:30:00+00:00")

    _, rows = get(client, done_since=CUTOFF)

    assert ids(rows) == {1}


def test_the_row_completed_boolean_is_still_a_boolean(client, backlog_dir):
    """`completed` on a *row* means "is done"; the window must not take the name.

    The route publishes `"completed": fm.get("status") == "done"`, `BacklogTask`
    in `web/src/api.ts` declares it as `boolean`, and `agent_mcp/backlog.py`
    serialises the front-matter key of the same spelling. A completion *date*
    published under `completed` on a list row would therefore be read as a
    boolean by the board on every done row — truthy, so nothing would break
    visibly, which is exactly why this needs pinning rather than noticing.
    """
    write_item(backlog_dir, 1, status="done", completed=NEW)
    write_item(backlog_dir, 2, status="up_next", completed=OLD)

    _, rows = get(client)

    by_id = {r["id"]: r for r in rows}
    assert by_id[1]["completed"] is True
    assert by_id[2]["completed"] is False


# ── clause 3: only done rows are cut ─────────────────────────────────────────

def test_an_open_row_older_than_the_window_is_still_returned(client, backlog_dir):
    """Every status but `done` survives a window that excludes its dates.

    This is the clause that makes the change a legibility filter instead of a
    way to lose the backlog: an `up_next` item from March is still on the board.
    """
    for i, status in enumerate(["draft", "up_next", "in_progress"], start=1):
        write_item(backlog_dir, i, status=status, completed=OLD, updated=OLD,
                   mtime=datetime(2026, 1, 1))

    _, unfiltered = get(client)
    _, filtered = get(client, done_since=CUTOFF)

    assert ids(unfiltered) == {1, 2, 3}
    assert ids(filtered) == {1, 2, 3}, (
        "`done_since` cuts the Done column and nothing else; open items are "
        "never filtered, whatever their dates say"
    )


def test_an_open_row_that_is_reopened_after_being_closed_does_not_vanish(client, backlog_dir):
    """A row whose `completed:` is stale but whose status is open is open.

    The complement of clause 2's precedence: `completed:` is consulted only for
    rows the filter is allowed to cut.
    """
    write_item(backlog_dir, 1, status="up_next", completed=OLD, updated=OLD)

    _, rows = get(client, done_since=CUTOFF)

    assert ids(rows) == {1}


# ── clause 4: a bad date is not an outage ───────────────────────────────────

@pytest.mark.parametrize("bad", [
    "bogus",
    "",
    "2026-9-1",            # unpadded: not the `YYYY-MM-DD` shape offered
    "2026-02-30",          # right shape, no such day
    "2026-09-10T12:00:00",  # a timestamp, not a date
    "null",
    "🐛",
])
def test_a_malformed_done_since_is_ignored_exactly_as_if_absent(client, backlog_dir, bad):
    """200, and the same rows as no parameter at all.

    The board refetches on `setInterval(loadData, 15_000)` with nothing gating
    it on tab visibility, so a value that only ever came from a URL bar would
    otherwise take the page down once a second for as long as it stayed there.
    """
    write_item(backlog_dir, 1, status="done", completed=NEW)
    write_item(backlog_dir, 2, status="done", completed=OLD)

    code, rows = get(client, done_since=bad)

    assert code == 200, f"`done_since={bad!r}` must not error the route"
    assert ids(rows) == {1, 2}, f"`done_since={bad!r}` must filter nothing"


def test_the_window_and_a_board_filter_compose(client, backlog_dir):
    """`done_since` is one more filter, not a new code path through the others."""
    write_item(backlog_dir, 1, status="done", board="lloyd", completed=NEW)
    write_item(backlog_dir, 2, status="done", board="lloyd", completed=OLD)
    write_item(backlog_dir, 3, status="done", board="other", completed=NEW)
    write_item(backlog_dir, 4, status="up_next", board="lloyd", completed=OLD)

    board_map = BR._board_index()[0]
    lloyd_id = str(board_map["lloyd"])
    _, rows = get(client, board_id=lloyd_id, done_since=CUTOFF)

    assert ids(rows) == {1, 4}


def test_the_window_and_a_search_compose_when_both_are_sent(client, backlog_dir):
    """The route applies both filters when asked for both.

    The *front end* omits `done_since` while a search is active, which is clause
    5 and lives in `tests/test_backlog_done_window_frontend_claim.py`; this pins
    that sending both is still coherent rather than one silently winning.
    """
    write_item(backlog_dir, 1, status="done", completed=NEW)
    write_item(backlog_dir, 2, status="done", completed=OLD)

    _, rows = get(client, q="Body 2", done_since=CUTOFF)

    assert ids(rows) == set(), "an explicit window still applies to search hits"
    _, search_only = get(client, q="Body 2")
    assert ids(search_only) == {2}, "without the window, old done items are findable"


# ── the helpers' own edges, so a future reader cannot widen them by accident ─

def test_the_window_slides_without_restarting_the_process(client, backlog_dir):
    """The cut-off is read per request, so the window moves with the calendar.

    The front end recomputes the date it sends on every fetch for the same
    reason; this pins the server half, where a module-level "today" would freeze
    the window for a backend that stays up for weeks.
    """
    write_item(backlog_dir, 1, status="done", completed=NO_KEY, updated=NO_KEY,
               mtime=datetime.now() - timedelta(days=3))

    _, wide = get(client, done_since=(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"))
    _, narrow = get(client, done_since=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"))

    assert ids(wide) == {1}, "a row touched 3 days ago is inside a 7-day window"
    assert ids(narrow) == set(), "tomorrow's cut-off excludes it — same process, no restart"


# ── #1023: what a status move through this route must write ───────────────────
#
# `_done_date` judges a done row by `completed:` first, so a route that closes an
# item without stamping one leaves that row's window position to whatever next
# touches the file — a priority edit months later re-admits it. 135 items closed
# since 2026-09-01 carry no `completed:`, and this route is a standing generator
# of that set. The fix is one shared recorder for both writers; these pin what it
# writes (clauses 2-4) through the same `client` the window tests use, because
# the POST is the boundary a person's click actually crosses.

_ROOT = Path(__file__).resolve().parents[1]


def post(client, **payload) -> dict:
    """One `POST /api/backlog/task-update`, as the board's modal sends it."""
    resp = client.post("/api/backlog/task-update", json=payload)
    assert resp.status_code == 200, resp.text
    return json.loads(resp.content)


def read_fm(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8").split("---", 2)[1])


def test_a_close_through_the_route_stamps_completed_and_nothing_else_does(
        client, backlog_dir):
    """#1023 clause 2: `completed:` arrives with a close and with no other move.

    The stamp is the shared one — naive UTC with microseconds, the format
    `_apply_status` and `close_landed` already write. It is bounded by two reads
    of UTC taken around the POST rather than by a tolerance in seconds: a budget
    fails a correct change on a loaded suite run, while a bracket is only as wide
    as the request actually took and still rejects a local-time stamp outright,
    which on this box (UTC-7) is seven hours outside it.
    """
    closing = write_item(backlog_dir, 1, status="up_next")
    opening = write_item(backlog_dir, 2, status="draft")

    before = datetime.now(timezone.utc).replace(tzinfo=None)
    post(client, id=1, status="done")
    after = datetime.now(timezone.utc).replace(tzinfo=None)
    post(client, id=2, status="up_next")

    closed = read_fm(closing)
    assert closed["status"] == "done"
    stamped = datetime.fromisoformat(str(closed["completed"]))
    assert before <= stamped <= after, (
        f"`completed:` must be a fresh UTC stamp between {before} and {after}, "
        f"got {closed['completed']!r}"
    )

    moved = read_fm(opening)
    assert moved["status"] == "up_next"
    assert "completed" not in moved, (
        "a move into an open status must not date a completion that did not happen"
    )


def test_a_reopen_through_the_route_leaves_the_earlier_completed_alone(
        client, backlog_dir):
    """A reopen is not an un-close: the earlier `completed:` stays as it was.

    `_done_date` only consults `completed:` on rows whose status is `done`, so a
    stale one on an open row costs nothing — while clearing it would throw away
    the only record of when the item was first closed.
    """
    f = write_item(backlog_dir, 1, status="done", completed=OLD, updated=OLD)

    post(client, id=1, status="up_next")

    fm = read_fm(f)
    assert fm["status"] == "up_next"
    assert str(fm["completed"]) == OLD
    assert any("done → up_next" in str(line) for line in fm["activity_log"])


def test_a_non_status_save_writes_no_activity_line_and_no_completed(
        client, backlog_dir):
    """#1023 clause 3: editing priority, blocked, board or name narrates nothing.

    Every one of these rides through the same handler, so a recorder wired to the
    wrong key would write "up_next → up_next" on every card edit and re-date the
    close each time.
    """
    f = write_item(backlog_dir, 1, status="up_next")

    post(client, id=1, priority="high")
    post(client, id=1, blocked=True)
    post(client, id=1, name="Renamed")
    post(client, id=1, board="other")

    fm = read_fm(f)
    assert "activity_log" not in fm, f"a field edit narrated a move: {fm.get('activity_log')!r}"
    assert "completed" not in fm
    assert fm["priority"] == "high" and fm["blocked"] is True and fm["board"] == "other"


def test_a_modal_save_that_re_posts_an_unchanged_status_records_no_move(
        client, backlog_dir):
    """The modal posts `status` on *every* save, so an unchanged one is not a move.

    `web/src/components/pages/BacklogPage.tsx` builds the update as
    `{ name, status, priority, blocked }` regardless of what was edited, so a
    recorder that logged any posted status would append a `up_next → up_next`
    line — and, on a done card, re-stamp `completed` and pull an item closed
    months ago back into the 7-day window on a title edit.
    """
    f = write_item(backlog_dir, 1, status="up_next")

    post(client, id=1, name="Renamed", status="up_next", priority="high", blocked=False)

    fm = read_fm(f)
    assert fm["status"] == "up_next"
    assert fm["priority"] == "high" and fm["blocked"] is False, "the edits still landed"
    assert "activity_log" not in fm, f"'no move' wrote a line: {fm.get('activity_log')!r}"
    assert "completed" not in fm


def test_the_route_no_longer_assigns_status_itself():
    """#1023 clause 4: `fm["status"] =` is gone from the Mission Control writer.

    Grep-shaped on purpose: the clause is that the assignment does not exist
    anywhere in the file, which no behavioural test can show — but the negative
    needs a control, or it passes for the wrong reason. The pattern that must
    still hit is the recorder's own line, so a check that found nothing because
    the string is spelled differently in this tree fails here instead of passing.
    """
    src = (_ROOT / "app" / "routers" / "backlog.py").read_text(encoding="utf-8")
    assert 'fm["status"]' not in src
    assert "record_status_move(" in src, "the route must record, not merely abstain"

    recorder = (_ROOT / "app" / "backlog_move.py").read_text(encoding="utf-8")
    assert 'fm["status"] =' in recorder, (
        "the assignment moved somewhere else, or the pattern above matches nothing"
    )


def _import_roots(path: Path) -> set[str]:
    """Top-level module names a file imports, read off its AST."""
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_the_shared_recorder_needs_only_the_standard_library():
    """#1023 clause 4: the automod CLI must load it without `mcp` or `httpx`.

    This is the reason `app/backlog_status.py` and `app/backlog_tags.py` live
    where they do: `scripts/automod/backlog.py` is the light module the CLI
    imports, so anything it shares with the backend has to be importable by a
    process that never touches the SDK.
    """
    helper = _ROOT / "app" / "backlog_move.py"
    roots = _import_roots(helper)
    non_stdlib = {root for root in roots if root not in sys.stdlib_module_names}
    assert non_stdlib == {"app"}, (
        f"the recorder may import local modules and stdlib only, imports {sorted(roots)}"
    )
    assert not {"mcp", "httpx", "yaml", "fastapi"} & roots

    # ...and the one local module it leans on is itself stdlib-only, or the
    # `app` allowance above would be a hole rather than a boundary.
    tags = _import_roots(_ROOT / "app" / "backlog_tags.py")
    assert {root for root in tags if root not in sys.stdlib_module_names} == set(), tags


def test_both_writers_call_the_same_record_status_move():
    """One definition, literally: both modules reach the same function object.

    Two copies that agree today are the state this item was filed against — the
    route's copy had drifted until it stamped nothing at all.
    """
    from app.backlog_move import record_status_move as shared
    from app.routers import backlog as route_module
    from scripts.automod import backlog as loop_module

    assert route_module.record_status_move is shared
    assert loop_module.record_status_move is shared
