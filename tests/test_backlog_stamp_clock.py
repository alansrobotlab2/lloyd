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


# ── #2183: the comment records the ruling, so nobody re-opens the question ────
#
# The ruling itself is settled and was, twice, before this round: delegated by Alan
# on 2026-09-27 (`backlog/1517-…md:83-86`, "ruled (delegated by Alan) — Grandfather")
# and recorded as `owed_settled` by the owed-check run
# `20261004_053615_owedcheck_a312` on 2026-10-04 (`backlog/1517-…md:131-162`,
# `follow_up: 2183`). What was NOT settled anywhere a reader would look was its
# *text*: `app/backlog_move.py:66` read "backfill vs grandfather is a person's call,
# on #1517", nine lines above the constant it governs — so the one file a future run
# opens before rewriting ~1,000 live board files was still advertising that the
# choice was unmade. The figures this section's prose cites (+7.00 h below the
# cut-off / +0.00 at or above, on 3,717 live `created`/`updated` rows; 1,027 files
# whose `completed:` the pre-fix closers already wrote naive UTC) are that ruling's
# evidence, cited from the `owed_settled` record rather than re-derived here.

MOVE_PY = _ROOT / "app" / "backlog_move.py"

#: The two strings that left the decision looking unmade (#2183 clause 1).
OPEN_QUESTION_STRINGS = ("person's call", "backfill vs grandfather")


def _cutover_comment() -> str:
    """The `#:` block sitting directly above `LOCAL_STAMP_CUTOVER`, read off the file.

    Sliced by adjacency rather than by line number, so editing the comment cannot
    move the subject out from under the check; and folded onto one line with the
    `#:` markers and the wrap removed, because what is being asserted is what the
    comment *says*, not where it happens to break — a phrase spanning a line break
    would otherwise read as absent and the test would fail on the reflow of a
    sentence nobody changed. Asserted fat before anything is looked for inside it:
    an absence check against a block that failed to slice passes for the wrong
    reason, which is the doc-test failure mode #2069 wrote down.
    """
    lines = MOVE_PY.read_text(encoding="utf-8").splitlines()
    at = next(i for i, ln in enumerate(lines)
              if ln.startswith("LOCAL_STAMP_CUTOVER = "))
    start = at
    while start > 0 and lines[start - 1].lstrip().startswith("#"):
        start -= 1
    block = " ".join(" ".join(ln.lstrip("#:").strip()
                              for ln in lines[start:at]).split())
    assert len(block) > 800, (
        f"the comment above `LOCAL_STAMP_CUTOVER` sliced to {len(block)} chars; an "
        "absence check against a near-empty block is a green light, not a guard"
    )
    return block


def test_the_cut_off_comment_offers_nobody_a_backfill_to_propose():
    """#2183 clause 1: neither string that posed the choice as open survives in the
    module at all — not in the cut-off block, not anywhere else in the file.

    The positive control is that the file now says the thing that replaced them.
    Without it, deleting the whole comment would satisfy the absence half by
    deleting the check's subject, which is the way a prose guard goes vacuous.
    """
    text = MOVE_PY.read_text(encoding="utf-8")
    for phrase in OPEN_QUESTION_STRINGS:
        assert phrase not in text, (
            f"app/backlog_move.py still carries {phrase!r}, so a run reading it "
            "believes whether to rewrite ~1,000 board files is still someone's "
            "unsettled question; #1517 ruled it grandfathered on 2026-10-04"
        )
    assert "grandfathered" in text, (
        "the two phrases are gone but so is the ruling that replaced them, so the "
        "absence above proved nothing"
    )


def test_the_cut_off_comment_records_the_ruling_as_settled_and_dated():
    """#2183 clause 2: settled, dated, attributed, and permanent by name.

    Four separate things, each of which the old sentence lacked: what was decided,
    when, on which item, and that the cut-off outlives the migration that could
    have retired it. A comment that merely said "grandfathered" would still read as
    a holding position, because the constant it describes looks exactly like a
    transitional device.
    """
    comment = _cutover_comment()
    assert "grandfathered permanently" in comment, "the ruling is not stated as final"
    assert "never backfilled" in comment, "the ruling's own word is 'never backfill'"
    assert "the ruling on #1517" in comment, (
        "the ruling is not attributed to the item that made it — the constant's own "
        "line mentions #1517 for a different reason, so a bare item number proves "
        "nothing here"
    )
    assert "2026-10-04" in comment, "the ruling carries no date, so it reads as undated"
    assert "permanent discriminator" in comment, "the constant's standing is not named"
    assert "transitional" in comment, (
        "nothing says what the cut-off is NOT, which is the half a future migration "
        "would need to be told"
    )


def test_the_cut_off_comment_names_the_reason_a_backfill_was_refused():
    """#2183 clause 3: the refusal reason is the part that survives contact with a
    future run, because it is what makes the migration *unsafe* rather than merely
    unappealing.

    The store cannot tell, for a row below the cut-off, which of its two writers
    stamped it — the very defect #1517 closed by documenting the cut-off instead of
    recording provenance per row. So a uniform +7 h shift is not a no-op tidy-up: it
    moves the 1,027 pre-cut-off `completed:` values the pre-fix automod closers
    already wrote in naive UTC, which are correct today under either reading.
    """
    comment = _cutover_comment()
    assert "no per-row clock provenance" in comment, (
        "the reason is not stated, so the ruling reads as a preference and a "
        "sufficiently confident run will revisit it"
    )
    assert "1,027" in comment and "completed:" in comment, (
        "the population a backfill would break is not named; the number is the "
        "owed-check ruling's own figure"
    )
    assert "naive UTC" in comment, "the double-moved values' clock is not named"
    assert "double-move" in comment, (
        "the mechanism of the harm is missing, leaving only the conclusion"
    )


# ── #2419: the row serializer dates a dateless file by its inode, in UTC ──────
#
# #1517 put the writers on one clock and moved the `?done_since=` mtime rung onto
# UTC, but it left a second, zone-less epoch call in the same file: when front matter
# carries no `created:`/`updated:`, the serializer fills the row from `stat()`, and
# `datetime.fromtimestamp(ts)` with no `tz=` answers in the machine's local zone while
# every consumer of `created_at`/`updated_at` reads a bare stamp as UTC — the front
# end's `new Date(task.created_at)` and the done window both. Triage measured it on
# the live board (2026-10-08, `TZ=America/Los_Angeles` pinned): item #223, whose front
# matter has no `created:`, was served `created_at = 2026-09-16T14:20:12.891027`
# against an inode instant of 2026-09-16T21:20:12.891027Z — seven hours behind its own
# file — and of 2,357 backlog files 13 have no `created:` and 47 no `updated:`, so the
# fallback has live rows. It is a read path and never writes the file, which is why
# the item is low: what moves is the board's display and anything that orders by
# `created`.

#: The module holding the backlog read path — `_row_from`, both GET routes and the
#: done window all live in it, so one path is the whole read surface.
BACKLOG_READ_PATHS = ("app/routers/backlog.py",)


def fromtimestamp_calls(path: Path) -> tuple[list[int], list[int]]:
    """`(bare, zoned)` line numbers of every `…fromtimestamp(…)` call in `path`.

    "bare" is the zone-less form — no second positional argument and no `tz=`
    keyword — the one that reads an epoch in the machine's local zone. Parsed
    rather than grepped for the reason `naive_now_calls` gives: this module's
    docstrings *describe* the retired call, and a grep would grade the prose.
    """
    bare: list[int] = []
    zoned: list[int] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "fromtimestamp"):
            continue
        if len(node.args) >= 2 or any(kw.arg == "tz" for kw in node.keywords):
            zoned.append(node.lineno)
        else:
            bare.append(node.lineno)
    return bare, zoned


def test_a_row_without_created_is_dated_by_its_ctime_instant(la_clock, client, route_dir):
    """#2419 clause 1: `created_at` is the file's `st_ctime` instant, to the second.

    The file is written with an `updated:` and no `created:`, so the only candidate
    for the field is the inode, and `la_clock` pins the zone rather than inheriting
    it — on this box the two clocks sit seven hours apart, which
    `_assert_clocks_differ()` inside `assert_utc_stamp` says out loud. Read through
    the route, not the helper: `created_at` crosses the HTTP boundary as a string
    (`str(created)` on the row, then JSON), and a test that called `_row_from`
    directly would never see how the value is spelled on the wire.

    Before the fix this fails by exactly the seven hours: the same fixture served
    `2026-09-16T14:20:12` where the inode said `21:20:12Z`.
    """
    f = write_item(route_dir, 40, "draft", updated="2026-01-01T00:00:00")
    assert "created" not in read_fm(f), "the fixture must have no `created:` to fall back on"
    ctime = datetime.fromtimestamp(f.stat().st_ctime, timezone.utc).replace(tzinfo=None)

    rows = {r["id"]: r for r in json.loads(client.get("/api/backlog/tasks").content)}
    detail = json.loads(client.get("/api/backlog/task/40").content)
    for label, row in (("list route", rows[40]), ("detail route", detail)):
        served = assert_utc_stamp(row["created_at"], what=f"{label} `created_at`")
        drift = abs((served - ctime).total_seconds())
        assert drift <= 1, (
            f"{label} dated item 40 at {served.isoformat()} against an inode ctime "
            f"of {ctime.isoformat()} — {drift / 3600:+.2f} h off its own file. A "
            "zone-less `fromtimestamp()` reads the epoch in the machine's zone while "
            "every reader of this field reads it as UTC."
        )


def test_a_row_without_updated_is_dated_by_its_mtime_instant(la_clock, client, route_dir):
    """#2419 clause 2: `updated_at` is the file's `st_mtime` instant, to the second.

    The mtime is pinned with `os.utime` to a fixed post-cut-over instant, so this
    compares against a number the test chose rather than against a clock it read a
    moment ago — and `created:` is present, so only the `updated:` fallback moves.
    """
    instant = (C + timedelta(hours=26)).replace(tzinfo=timezone.utc)
    ts = int(instant.timestamp())
    f = write_item(route_dir, 41, "draft", created="2026-01-01T00:00:00")
    assert "updated" not in read_fm(f), "the fixture must have no `updated:` to fall back on"
    os.utime(f, (ts, ts))

    rows = {r["id"]: r for r in json.loads(client.get("/api/backlog/tasks").content)}
    served = assert_utc_stamp(rows[41]["updated_at"], what="list route `updated_at`")
    drift = abs((served - instant.replace(tzinfo=None)).total_seconds())
    assert drift <= 1, (
        f"item 41 was served updated_at={served.isoformat()} for an mtime of "
        f"{instant.isoformat()}Z — {drift / 3600:+.2f} h off. The done window already "
        "dates the same inode in UTC, so the board and the window disagreed."
    )


def test_a_metadata_derived_stamp_keeps_the_stores_naive_shape(la_clock, client, route_dir):
    """#2419 clause 3: an inode-derived stamp is spelled exactly like a written one.

    Both halves come from one request: item 42 has front matter, item 43 has none, so
    the two rows differ only in where their `created_at` came from. An aware value
    would serialise as `…+00:00`, a shape no front-matter stamp in this store has
    (`app/backlog_move.py:42-48` rules a mixed population out because `_fm_date`
    cannot compare the two), so the fallback has to land on naive numerals — which is
    also what `assert_utc_stamp` refuses an offset over, with the clock-gap control.
    """
    write_item(route_dir, 42, "draft", created="2026-09-01T00:00:00",
               updated="2026-09-01T00:00:00")
    write_item(route_dir, 43, "draft")

    rows = {r["id"]: r for r in json.loads(client.get("/api/backlog/tasks").content)}
    fm_stamp = rows[42]["created_at"]
    meta_stamp = rows[43]["created_at"]
    assert_utc_stamp(meta_stamp, what="metadata-derived `created_at`")
    for label, stamp in (("front-matter", fm_stamp), ("metadata", meta_stamp)):
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?", stamp), (
            f"the {label} stamp {stamp!r} is not the bare ISO form every board reader "
            "parses; the fallback must not introduce a second shape"
        )


@pytest.mark.parametrize("rel", BACKLOG_READ_PATHS)
def test_no_backlog_reader_converts_an_epoch_in_the_local_clock(rel: str):
    """#2419 clause 5, guarded like `test_no_backlog_writer_calls_the_local_clock`.

    No zone-less `fromtimestamp()` survives anywhere in the backlog read path — the
    serializer's two fallbacks were the last of them, and 28 other epoch conversions
    in `app/` and `scripts/` already pass `timezone.utc`. The positive control is the
    same one that test uses: the module must still convert epochs *somewhere*, through
    the shared `_inode_instant`, or a zero here would only prove the calls — or the
    file — had been deleted.
    """
    path = _ROOT / rel
    bare, zoned = fromtimestamp_calls(path)
    text = path.read_text(encoding="utf-8")
    assert bare == [], (
        f"{rel} still converts an epoch in the machine's local zone at line(s) {bare}; "
        "every stat()-derived stamp comes from `_inode_instant`"
    )
    assert zoned, f"{rel} converts no epoch at all, so the sweep above proved nothing"
    assert "_inode_instant(" in text, (
        f"{rel}'s epoch calls do not go through the shared derivation, so the next "
        "reader added here can pick a zone again by hand"
    )
