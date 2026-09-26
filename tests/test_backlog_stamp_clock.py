"""One clock for the backlog store. Item #1517.

The board had two clocks and neither side knew. Four write surfaces stamped
`created`/`updated` with `datetime.now()` — naive *local* time — while the two
that recorded status moves stamped `app/backlog_move.now_stamp()`, naive UTC. On
this box (America/Los_Angeles, PDT, -0700) the two differ by seven hours, so a
file touched forty seconds apart can carry `**2026-09-26T02:48:18**` above
`**2026-09-25T19:49:05.757813**` and say the earlier one happened second. Measured
at triage over `~/obsidian/backlog`: 985 of the 1,249 files with an `activity_log`
hold a stamp more than an hour ahead of their own `stat().st_mtime`.

The stamps stay naive. `app/backlog_move.py:42-48` records why: `_fm_date`
compares calendar dates in the value's own zone, and a mixed population of
offset-bearing and bare stamps is the one thing it cannot judge consistently.
So the fix is not an offset, it is *one* clock — naive UTC from
`app/backlog_move.now_stamp()` on every write surface — plus a documented
cut-off so the readers date the rows the retired writers left behind in the
clock those writers actually used:

* `LOCAL_STAMP_CUTOVER` divides them. A naive stamp below it was written by a
  retired naive-local surface and is read in the machine's local clock; at or
  above it, in UTC. Pre-existing rows are grandfathered, never rewritten.
* On this box a naive-local stamp's numerals always run seven hours *behind* the
  UTC numerals for the same instant, so every legacy row falls below the cut-off
  permanently. The cut-off is therefore set at the instant this change was
  authored, which is necessarily before any row the new code writes.

What each test below pins, in clause order:

1. a `task-update` POST that changes no status — the branch that used to stamp
   `datetime.now()` at `app/routers/backlog.py:663`;
2. a `task-create` POST's `created:` and `updated:` (`:702`);
3. `agent_mcp/backlog.py::add_activity` — the `backlog_write_task` path, the
   highest-traffic writer on the board — both its `**…**` line and `updated:`,
   and `_handle_write`'s own stamp (`:271`) on both its create and update paths;
4. `scripts/automod/backlog.py::new_item` (`:5780`), and a static sweep proving
   no backlog writer calls the local clock any more;
5. the readers: `board_flow`'s 24-hour net (`:5356`, `:5371`) and the route's
   `?done_since=` window both date a row in the clock it was written in, with
   pre-cut-off rows handled by the cut-off rather than mis-dated.

Every assertion here compares a stamp against `datetime.now(timezone.utc)`, so on
a machine already at UTC the two clocks coincide and each test would pass with or
without the fix. The `la_clock` fixture therefore pins the zone to
`America/Los_Angeles` and `_assert_clocks_differ()` is the positive control that
says out loud, in every test, that the two clocks really are hours apart.
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from agent_mcp import backlog as BL
from app import backlog_move as BM
from app.routers import backlog as BR
from scripts.automod import backlog as B

_ROOT = Path(__file__).resolve().parents[1]

#: The five write surfaces #1517 puts on one clock. Each still holds the string
#: its stamps now come from, which is the positive control in
#: `test_no_backlog_writer_calls_the_local_clock`.
WRITE_SURFACES = (
    "app/routers/backlog.py",
    "agent_mcp/backlog.py",
    "scripts/automod/backlog.py",
    "app/backlog_move.py",
)


# ── fixtures and helpers ─────────────────────────────────────────────────────

@pytest.fixture
def la_clock():
    """Run the test with the machine's local zone pinned to America/Los_Angeles.

    The bug is a seven-hour gap between `datetime.now()` and
    `datetime.now(timezone.utc)`; forcing the zone reproduces it on any machine,
    including a CI box at UTC, where it would otherwise be unobservable.
    """
    old = os.environ.get("TZ")
    os.environ["TZ"] = "America/Los_Angeles"
    time.tzset()
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = old
        time.tzset()


@pytest.fixture
def route_dir(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(BR, "_BACKLOG_DIR", d)
    monkeypatch.setattr(BR, "_FM_CACHE", {})
    return d


@pytest.fixture
def client(route_dir):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(BR.router)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mcp_dir(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(BL, "BACKLOG_DIR", d)
    return d


@pytest.fixture
def board(tmp_path):
    d = tmp_path / "backlog"
    d.mkdir()
    return d


def _assert_clocks_differ() -> timedelta:
    """The gap between the two clocks right now, and a refusal to test nothing.

    Without this a UTC machine turns all fifteen assertions below into tautologies:
    `datetime.now()` and `datetime.now(timezone.utc)` are the same numbers there.
    """
    gap = abs(datetime.now(timezone.utc).utcoffset()
              - (datetime.now().astimezone().utcoffset() or timedelta(0)))
    assert gap >= timedelta(hours=2), (
        "these tests pin a naive-local stamp seven hours off UTC; on a UTC-local "
        f"machine (gap {gap}) they would pass with or without the fix"
    )
    return gap


def assert_utc_stamp(stamp, *, what: str) -> datetime:
    """`stamp` is naive (no offset) and UTC — the one clock the store writes.

    Accepts what a real read back from disk gives: PyYAML resolves a bare ISO
    timestamp to a naive `datetime`, and an offset-bearing one to an *aware*
    `datetime`, which this rejects outright because `app/backlog_move.py:42-48`
    rules the mixed population out.
    """
    _assert_clocks_differ()
    if isinstance(stamp, datetime):
        text, parsed = stamp.isoformat(), stamp
    else:
        text = str(stamp).strip()
        assert not re.search(r"(Z|[+-]\d{2}:?\d{2})$", text), (
            f"{what}: {text!r} carries a UTC offset; the store's stamps are naive "
            "(`app/backlog_move.py:42-48`)"
        )
        parsed = datetime.fromisoformat(text)
    assert parsed.tzinfo is None, f"{what}: {text!r} is tz-aware, see above"
    return parsed


def assert_between(stamp, before: datetime, after: datetime, *, what: str) -> datetime:
    """`stamp` falls between two reads of UTC taken around the write.

    Stricter than the clause's "within 2 seconds", and it cannot fail a correct
    change on a loaded suite run the way a fixed budget does: the bracket is only
    as wide as the request took, and a naive-local stamp is seven hours outside it.
    """
    parsed = assert_utc_stamp(stamp, what=what)
    assert before <= parsed <= after, (
        f"{what}: stamp {stamp!r} is not between {before.isoformat()} and "
        f"{after.isoformat()} — {((parsed - before).total_seconds() / 3600):+.2f} h "
        "off UTC now. A naive-local stamp on this box sits ~7 h before this window."
    )
    return parsed


def local_numerals(instant: datetime) -> str:
    """`instant` spelled the way a retired writer spelled it: local, and naive."""
    return instant.astimezone().replace(tzinfo=None).isoformat()


def read_fm(path: Path) -> dict:
    """Read one board file back through the store's own anchored splitter."""
    fm, _ = BL.parse_frontmatter(path.read_text(encoding="utf-8"))
    return fm


def write_item(d: Path, item_id: int, status: str = "draft", **fm) -> Path:
    data = {"type": "backlog", "segment": "backlog", "status": status,
            "priority": "medium", "board": "lloyd", "blocked": False,
            "assigned": False, "position": item_id * 1000, **fm}
    p = d / f"{item_id}-item-{item_id}.md"
    p.write_text(f"---\n{yaml.dump(data, default_flow_style=False, sort_keys=False)}"
                 f"---\n\n# item {item_id}\n\nbody\n", encoding="utf-8")
    return p


def post(client, url: str, **payload) -> dict:
    resp = client.post(url, json=payload)
    assert resp.status_code == 200, resp.text
    return json.loads(resp.content)


def naive_now_calls(path: Path) -> list[int]:
    """Lines calling `datetime.now()` / `datetime.utcnow()` with no argument.

    Parsed, not grepped: a docstring that *describes* the retired call must not
    read as a use of it, and `datetime.now(timezone.utc)` is the clock we want.
    """
    hits: list[int] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call) or node.args or node.keywords:
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr in ("now", "utcnow")):
            continue
        v = fn.value
        if (isinstance(v, ast.Name) and v.id == "datetime") or \
           (isinstance(v, ast.Attribute) and v.attr == "datetime"):
            hits.append(node.lineno)
    return hits


# The instant the cut-off sits at, and a clock a day past it: `CUTOVER + 3 h` is
# the `now` the reader tests judge against, so every "legacy" stamp they
# fabricate is below the cut-off forever rather than only until the wall clock
# advances past it.
C = BM.LOCAL_STAMP_CUTOVER


def legacy_local(now: datetime, hours: int) -> str:
    """A stamp a retired writer left: naive LOCAL numerals for `now - hours`."""
    return (now - timedelta(hours=hours)).astimezone().replace(tzinfo=None).isoformat()


def utc_naive(now: datetime, hours: int) -> str:
    """A stamp the one clock writes: naive UTC numerals for `now - hours`."""
    return (now - timedelta(hours=hours)).replace(tzinfo=None).isoformat()


# ── clause 1: the route's non-status save ────────────────────────────────────

def test_a_priority_save_stamps_updated_in_the_utc_clock(la_clock, client, route_dir):
    """#1517 clause 1, replacing `app/routers/backlog.py:663`.

    The `not status_recorded` branch is what an ordinary card edit takes, and it
    was the one route stamp still on the machine's clock. A `TaskModal` save that
    posts the status *unchanged* lands in the same branch, so it is pinned too.
    """
    f = write_item(route_dir, 1, status="up_next",
                   created="2026-01-01T00:00:00", updated="2026-01-01T00:00:00")
    before = datetime.now(timezone.utc).replace(tzinfo=None)
    post(client, "/api/backlog/task-update", id=1, priority="high")
    after = datetime.now(timezone.utc).replace(tzinfo=None)

    fm = read_fm(f)
    assert fm["priority"] == "high", "the save has to have happened at all"
    assert fm["status"] == "up_next", "and it must not have moved the item"
    assert_between(fm["updated"], before, after, what="route save `updated:`")
    assert "completed" not in fm


def test_a_status_post_that_does_not_move_the_item_stamps_in_the_utc_clock(
        la_clock, client, route_dir):
    """Same clause, other edge into it: `TaskModal` posts the whole form.

    Posting the status the item already has is not a move, so `record_status_move`
    records nothing and the save falls to the same stamp. It has to be the one
    clock too, or every board click re-dates a row in the retired zone.
    """
    f = write_item(route_dir, 2, status="up_next",
                   created="2026-01-01T00:00:00", updated="2026-01-01T00:00:00")
    before = datetime.now(timezone.utc).replace(tzinfo=None)
    post(client, "/api/backlog/task-update", id=2, status="up_next", blocked=True)
    after = datetime.now(timezone.utc).replace(tzinfo=None)

    fm = read_fm(f)
    assert fm["blocked"] is True
    assert fm.get("activity_log") in (None, []), (
        "posting an unchanged status must narrate no move (#1023)"
    )
    assert_between(fm["updated"], before, after,
                   what="route save (unchanged status) `updated:`")


# ── clause 2: the route's create ─────────────────────────────────────────────

def test_a_create_through_the_route_stamps_both_dates_in_the_utc_clock(
        la_clock, client, route_dir):
    """#1517 clause 2, replacing `app/routers/backlog.py:702`.

    One `now` feeds both `created:` and `updated:`, so a create used to put a
    naive-local birth date on an item whose first status move was stamped UTC —
    the pair `board_flow` then compared seven hours apart.
    """
    before = datetime.now(timezone.utc).replace(tzinfo=None)
    out = post(client, "/api/backlog/task-create", name="Clock check",
               description="Body.", board="lloyd", status="draft")
    after = datetime.now(timezone.utc).replace(tzinfo=None)

    f = route_dir / next(p.name for p in route_dir.glob(f"{out['id']}-*.md"))
    fm = read_fm(f)
    assert_between(fm["created"], before, after, what="route create `created:`")
    assert_between(fm["updated"], before, after, what="route create `updated:`")


# ── clause 3: the MCP path, the board's busiest writer ───────────────────────

def test_add_activity_stamps_its_line_and_updated_in_the_utc_clock(la_clock):
    """#1517 clause 3, `agent_mcp/backlog.py::add_activity` (`:96`).

    One stamp feeds both the `**…**` activity line and `updated:`. This is the
    path `backlog_write_task` takes — including the appends that wrote this
    item's own triage log — so it is the surface that produced the split the item
    was filed from.
    """
    before = datetime.now(timezone.utc).replace(tzinfo=None)
    task = BL.add_activity({"id": 7, "filename": "7-x.md"}, "triaged: confirmed")
    after = datetime.now(timezone.utc).replace(tzinfo=None)

    assert_between(task["updated"], before, after, what="add_activity `updated`")
    line = str(task["activity_log"][-1])
    m = re.match(r"^\*\*(.+?)\*\* — ", line)
    assert m, f"activity line lost its stamp: {line!r}"
    assert_between(m.group(1), before, after, what="add_activity's `**…**` line")
    assert str(task["updated"]) in line, (
        "a move and its own log line must not disagree about when they happened"
    )


def test_an_mcp_update_stamps_updated_in_the_utc_clock(la_clock, mcp_dir):
    """#1517 clause 3, `_handle_write`'s own stamp (`:271`) on the update path."""
    write_item(mcp_dir, 11, status="up_next",
               created="2026-01-01T00:00:00", updated="2026-01-01T00:00:00")
    before = datetime.now(timezone.utc).replace(tzinfo=None)
    out = json.loads(BL._handle_write({"task_id": 11, "priority": "high"}))
    after = datetime.now(timezone.utc).replace(tzinfo=None)
    assert out.get("success"), out

    fm = read_fm(mcp_dir / "11-item-11.md")
    assert fm["priority"] == "high"
    assert_between(fm["updated"], before, after, what="`_handle_write` `updated`")


def test_an_mcp_create_stamps_created_in_the_utc_clock(la_clock, mcp_dir):
    """Same clause, create path: `_handle_write` seeds `created` from its `now`."""
    before = datetime.now(timezone.utc).replace(tzinfo=None)
    out = json.loads(BL._handle_write({"name": "Fresh from MCP",
                                       "description": "Body.", "board": "lloyd"}))
    after = datetime.now(timezone.utc).replace(tzinfo=None)
    assert out.get("success") and out.get("created"), out

    files = sorted(mcp_dir.glob(f"{out['task_id']}-*.md"))
    assert len(files) == 1, files
    fm = read_fm(files[0])
    assert_between(fm["created"], before, after, what="`_handle_write` `created`")
    assert_between(fm["updated"], before, after, what="`_handle_write` create `updated`")


# ── clause 4: the automod writer, and the sweep that says "none left" ────────

def test_new_item_stamps_both_dates_in_the_utc_clock(la_clock, board):
    """#1517 clause 4, `scripts/automod/backlog.py::new_item` (stamp at `:5780`).

    The module whose closers had always written UTC created items on the local
    clock, so an item this module filed disagreed with itself the moment the same
    module closed it.
    """
    before = datetime.now(timezone.utc).replace(tzinfo=None)
    item = B.new_item("A fresh automod item", backlog_dir=board)
    after = datetime.now(timezone.utc).replace(tzinfo=None)

    fm = read_fm(item.path)
    assert_between(fm["created"], before, after, what="`new_item` `created`")
    assert_between(fm["updated"], before, after, what="`new_item` `updated`")


@pytest.mark.parametrize("rel", WRITE_SURFACES)
def test_no_backlog_writer_calls_the_local_clock(rel: str):
    """#1517 clause 4's second half: *no* backlog writer emits a naive-local stamp.

    The retired call is gone from all four modules — and the positive control is
    that each one now calls the shared helper, so a zero here cannot come from the
    file having been emptied out from under the check.
    """
    path = _ROOT / rel
    assert naive_now_calls(path) == [], (
        f"{rel} still stamps with the machine's local clock at line(s) "
        f"{naive_now_calls(path)}; every backlog stamp comes from "
        "`app/backlog_move.now_stamp()`"
    )
    assert "now_stamp(" in path.read_text(encoding="utf-8"), (
        f"{rel} never calls the shared helper, so the sweep above proved nothing"
    )


# ── clause 5: the readers date a row in the clock it was written in ──────────

def test_board_flow_counts_creation_in_each_rows_own_clock(la_clock, board):
    """#1517 clause 5, the inflow side (`scripts/automod/backlog.py:5356`).

    Judged against `CUTOVER + 3 h`, so the legacy rows below sit below the
    cut-off on any day this runs:

    * a post-cut-off creation 1 h old — **in**, and out if read as local;
    * the same instant stamped by a retired writer — **in**, read in local time;
    * a retired row 21 h old — **in**, and out if read as UTC;
    * a retired row 30 h old — **out** under either reading, which is what keeps
      the window a window.

    All-local and all-UTC each score 2; the cut-off rule scores 3.
    """
    now = (C + timedelta(hours=3)).replace(tzinfo=timezone.utc)
    write_item(board, 1, "draft", created=utc_naive(now, 1))
    write_item(board, 2, "draft", created=legacy_local(now, 1))
    write_item(board, 3, "draft", created=legacy_local(now, 21))
    write_item(board, 4, "draft", created=legacy_local(now, 30))

    flow = B.board_flow(backlog_dir=board, now=now.timestamp())
    assert flow["24h"]["created"] == 3, flow["24h"]


def test_board_flow_counts_a_close_in_each_rows_own_clock(la_clock, board):
    """#1517 clause 5, the outflow side (`:5371`) and `completed:`'s precedence.

    * a done row whose `updated:` is post-cut-off and 2 h old — **in**;
    * a done row with `completed:` 4 h old — **in** (`completed:` is naive UTC on
      both sides of the cut-off, so it is never re-dated);
    * a done row whose `completed`-less `updated:` is a retired 22 h-old stamp —
      **in**, and out if read as UTC;
    * a done row `completed:` 28 h old but touched 2 h ago — **out**: `completed:`
      wins, so a later edit cannot re-admit a close.

    All-local and all-UTC each score 2; the cut-off rule scores 3.
    """
    now = (C + timedelta(hours=3)).replace(tzinfo=timezone.utc)
    write_item(board, 10, "done", created=utc_naive(now, 40),
               updated=utc_naive(now, 2))
    write_item(board, 11, "done", created=utc_naive(now, 40),
               completed=utc_naive(now, 4), updated=utc_naive(now, 4))
    write_item(board, 12, "done", created=legacy_local(now, 40),
               updated=legacy_local(now, 22))
    write_item(board, 13, "done", created=utc_naive(now, 60),
               completed=utc_naive(now, 28), updated=utc_naive(now, 2))

    flow = B.board_flow(backlog_dir=board, now=now.timestamp())
    assert flow["24h"]["closed"] == 3, flow["24h"]


def test_the_done_window_dates_a_retired_local_row_in_utc(la_clock, client, route_dir):
    """#1517 clause 5, the route's `?done_since=` window.

    `done_since` carries no zone, and the front end computes it from
    `toISOString()`, so a row is judged by its UTC calendar date. A row the
    retired writers stamped has LOCAL numerals, and seven hours is enough to move
    a date: an item done at 06:00 UTC on the cut-over day reads
    `2026-09-25T23:00` and the old code cut it a day early.

    Item 3 is the control that the filter still filters — its `completed:` is a
    real UTC stamp, months old, and must stay cut.
    """
    instant = (C + timedelta(hours=2)).replace(tzinfo=timezone.utc)   # 06:00 UTC
    window = instant.date()
    assert local_numerals(instant).startswith(str(window - timedelta(days=1))), (
        "the case needs the retired local numerals to fall on the previous day"
    )
    write_item(route_dir, 1, "done", updated=local_numerals(instant))
    write_item(route_dir, 2, "done", completed="2026-03-01T00:00:00")
    write_item(route_dir, 3, "up_next", updated=utc_naive(datetime.now(timezone.utc), 1))

    resp = client.get(f"/api/backlog/tasks?done_since={window.isoformat()}")
    assert resp.status_code == 200
    ids = {r["id"] for r in json.loads(resp.content)}
    assert 1 in ids, (
        f"a row whose true close is {instant.isoformat()}Z was cut by "
        f"done_since={window}: its retired local stamp read as a day earlier"
    )
    assert 2 not in ids, "the filter must still cut a genuinely old done row"
    assert 3 in ids, "open rows are never cut by the done window"


def test_the_done_window_reads_its_mtime_rung_in_utc(la_clock, client, route_dir):
    """#1517 clause 5, the last rung of `_done_date` (`app/routers/backlog.py:493`).

    A done row with no usable date is dated by `stat().st_mtime`, which came back
    as a naive LOCAL `datetime` and then got compared against a UTC-derived
    cut-off — the same class of error inside the very function that judges the
    window. Pin the mtime to `instant`, which is 23:00 locally the day before it
    is 06:00 UTC, and require the row to survive the window its instant puts it in.
    """
    instant = (C + timedelta(hours=2)).replace(tzinfo=timezone.utc)   # 06:00 UTC
    assert instant.astimezone().replace(tzinfo=None) < instant.replace(tzinfo=None), (
        "the case needs the local numerals to fall on the previous day"
    )
    f = write_item(route_dir, 4, "done")
    ts = instant.timestamp()
    os.utime(f, (ts, ts))

    resp = client.get(f"/api/backlog/tasks?done_since={instant.date().isoformat()}")
    ids = {r["id"] for r in json.loads(resp.content)}
    assert 4 in ids, (
        f"a done row last touched at {instant.isoformat()}Z was cut by "
        f"done_since={instant.date()}: the mtime rung still answers in local time"
    )


# ── the cut-off itself ───────────────────────────────────────────────────────

def test_the_cut_off_precedes_every_stamp_the_new_code_writes(la_clock):
    """The one invariant the cut-off rule rests on, checked rather than asserted.

    A row stamped by the new code must fall at or above `LOCAL_STAMP_CUTOVER` or
    the readers date it in the retired clock and shift it seven hours into the
    future — the failure the cut-off exists to prevent. A constant moved forward
    by mistake trips this immediately.
    """
    assert C <= datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=2), (
        f"LOCAL_STAMP_CUTOVER is in the future ({C}); every stamp written before "
        "it would be read in the retired local clock"
    )
    assert C == BM.LOCAL_STAMP_CUTOVER.replace(tzinfo=None)
    assert_utc_stamp(BM.now_stamp(), what="`now_stamp()`")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}", BM.now_stamp()), (
        "`now_stamp()` changed shape; `_fm_date` and every board reader parse the "
        "bare ISO form"
    )
