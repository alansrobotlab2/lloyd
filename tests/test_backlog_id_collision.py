"""One backlog id, two files: both resolvers must refuse, and neither file may be touched.

The incident, measured live at `bd0366ed` and recorded on #3696/#3699: an unattended
session ran `cat >> ~/obsidian/backlog/<hand-guessed-slug>.md`, which CREATED a second
`3691-*.md` on the board with no front matter at all. From that moment every lookup of
id 3691 answered whichever file glob happened to yield first — one `backlog_write_task`
call appended to the phantom, the next overwrote the real item's body with the
phantom's, and no call along the way said *there are two of these*. The board's own
canary went red on the file with no front matter; the corruption of the real item was
silent.

Two resolvers answered that question, each with its own copy of the same
`for f in glob("*.md"): if <id>-prefixed: return f` loop, and both returned the first
match:

* `app.routers.backlog._backlog_find_file` — behind `GET /api/backlog/task/{id}`,
  `POST /api/backlog/task-update` and `POST /api/backlog/task-delete`, whose
  `filepath.unlink()` is how a delete could destroy the wrong item.
* `agent_mcp.backlog.load_task` — which sets the `filename` that `save_task` writes
  back to, so this is the lane `backlog_write_task` actually used. The board route and
  the MCP lane could each have been fixed while the other stayed open; #3701 clause 3
  is the one that says *drive both*, because the prior round verified only the route.

Both loops are gone. Each resolver now asks `app.backlog_write.resolve_backlog_id_file`
— one predicate, so the two cannot drift apart again — and a duplicate id raises
`AmbiguousBacklogId` naming **every** matching path. An error that only counted the pair
could not be acted on: the operator's next move is to work out which file is the phantom.

The other half of what these nodes pin is *what must not happen on the way to the
error*: no file is read as the item, written, or unlinked. A refusal that deleted a file
first would be worse than the bug it fixes.

Scratch board directory throughout (`tmp_path`), because the live board is 3,639 files
and holds zero duplicate ids today — that detector is `#3701`'s own premise check, and a
test that ran against the real tree would pass at the base commit and so pin nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import agent_mcp.backlog as MCP  # noqa: E402
from app import backlog_write as BW  # noqa: E402
from app.routers import backlog as BR  # noqa: E402
from fastapi import HTTPException  # noqa: E402

#: The pair from the incident, under one id. The phantom is exactly what the shell
#: redirect left behind: a body-only file with no front matter.
ID = 3691
REAL = f"{ID}-count-2026-10-29-as-a-tracked-fixture-10-trigger-i.md"
PHANTOM = f"{ID}-hand-guessed-phantom.md"
REAL_TEXT = "---\ntype: note\nsegment: backlog\nstatus: open\n---\n\n# Real item\n\n" \
            "The body the second call overwrote.\n"
PHANTOM_TEXT = "appended by the redirect, never had front matter\n"


class _Req:
    """The board routes take a FastAPI `Request`; this is what the other route tests use."""

    def __init__(self, payload: dict):
        self._payload = payload

    async def json(self) -> dict:
        return self._payload


@pytest.fixture
def colliding_board(tmp_path, monkeypatch):
    """A scratch board holding two files on id 3691, wired into both lanes.

    All three module-level directory names are patched because the two lanes read
    different ones: the route reads `app.routers.backlog._BACKLOG_DIR`, the MCP tool
    reads `agent_mcp.backlog.BACKLOG_DIR`, and `app.routers.backlog` is what
    `task-update` also imports for its own writes.
    """
    d = tmp_path / "backlog"
    d.mkdir()
    (d / REAL).write_text(REAL_TEXT, encoding="utf-8")
    (d / PHANTOM).write_text(PHANTOM_TEXT, encoding="utf-8")
    monkeypatch.setattr(BR, "_BACKLOG_DIR", d)
    monkeypatch.setattr(MCP, "BACKLOG_DIR", d)
    return d


def _both_survive(d: Path):
    assert (d / REAL).exists(), "the real item was destroyed on a refused call"
    assert (d / PHANTOM).exists(), "the phantom was removed by a call that only reports"
    assert (d / REAL).read_text(encoding="utf-8") == REAL_TEXT, \
        "the real item's bytes changed on a refused call — the 3696 overwrite again"
    assert (d / PHANTOM).read_text(encoding="utf-8") == PHANTOM_TEXT, \
        "the phantom's bytes changed on a refused call"


# ── Clause 1: the route-side resolver refuses and names every matching path ─────

def test_two_files_on_one_id_make_the_resolver_refuse_and_name_both(colliding_board):
    """`_backlog_find_file` answers an exception, and returns neither file.

    Measured at `bd0366ed`: this returned `3691-hand-guessed-phantom.md` — the
    phantom, not the real item — because glob order is directory order and the phantom
    sorted out of it. A first-match return is the defect, not an unlucky draw.
    """
    with pytest.raises(BW.AmbiguousBacklogId) as exc:
        BR._backlog_find_file(ID)
    message = str(exc.value)
    assert str(colliding_board / PHANTOM) in message, \
        f"the refusal must name the phantom in full: {message}"
    assert str(colliding_board / REAL) in message, \
        f"the refusal must name the real item in full: {message}"
    _both_survive(colliding_board)


def test_the_refusal_names_every_path_when_the_collision_is_wider_than_the_incident(colliding_board):
    """A third file on the id is named too — the error is not capped at the pair.

    The board's history shows ids re-used after a delete, so three is reachable; an
    error that printed "found 2" and one path would leave a phantom unfound.
    """
    third = colliding_board / f"{ID}-third-spelling.md"
    third.write_text("---\nsegment: backlog\n---\n\n# Third\n", encoding="utf-8")
    with pytest.raises(BW.AmbiguousBacklogId) as exc:
        BR._backlog_find_file(ID)
    message = str(exc.value)
    for name in (PHANTOM, REAL, third.name):
        assert name in message, f"{name} missing from the refusal: {message}"


def test_a_unique_id_still_resolves_after_the_check_lands(colliding_board):
    """The guard is a collision guard, not a shutdown: one match still resolves.

    Without this node, a resolver that raised on *every* lookup would pass clause 1
    while taking the board down.
    """
    only = colliding_board / "3699-only-one-file.md"
    only.write_text("---\nsegment: backlog\n---\n\n# Only\n", encoding="utf-8")
    assert BR._backlog_find_file(3699) == only
    assert BR._backlog_find_file(3698) is None, "an absent id must stay a None, not an error"


# ── Clause 2: the delete route deletes nothing on a colliding id ────────────────

@pytest.mark.asyncio
async def test_task_delete_on_a_colliding_id_deletes_nothing_and_names_both(colliding_board):
    """`POST /api/backlog/task-delete` refuses before it reaches `unlink()`.

    Measured at `bd0366ed`: the route read `filepath = _backlog_find_file(task_id)` and
    called `filepath.unlink()` with no collision check behind it, so a tidy-up call
    deleted whichever file glob yielded — the phantom or the real item, decided by
    directory order.
    """
    with pytest.raises(HTTPException) as exc:
        await BR.backlog_task_delete(_Req({"id": ID}))
    assert exc.value.status_code == 409, \
        f"a duplicate id is a conflict in the board, not a missing item: {exc.value.detail}"
    detail = str(exc.value.detail)
    assert PHANTOM in detail and REAL in detail, \
        f"the 409 must name both files so a person can tell them apart: {detail}"
    _both_survive(colliding_board)


@pytest.mark.asyncio
async def test_task_update_on_a_colliding_id_writes_neither_file(colliding_board):
    """The route that did the damage in the incident refuses too.

    `task-update` is the writer that overwrote the real item's body with the phantom's
    on 2026-11-01; clause 2 is about delete, and this is the sibling that proves the
    fix sits in the resolver rather than in one route.
    """
    with pytest.raises(HTTPException) as exc:
        await BR.backlog_task_update(_Req({"id": ID, "status": "done"}))
    assert exc.value.status_code == 409, str(exc.value.detail)
    _both_survive(colliding_board)


# ── Clause 3: the MCP lane refuses the same collision, driven independently ─────

def test_load_task_the_resolver_the_mcp_lane_actually_uses_raises_on_the_pair(colliding_board):
    """`load_task` is a second resolver, and it now asks the same predicate.

    Driven directly, with no HTTP in sight: `backlog_write_task` never calls
    `_backlog_find_file`, so a round that verified only the route would leave this
    returning the phantom's dict — and `save_task` writes to the `filename` this
    function puts in it, which is how the real body was overwritten.
    """
    with pytest.raises(BW.AmbiguousBacklogId) as exc:
        MCP.load_task(ID)
    message = str(exc.value)
    assert PHANTOM in message and REAL in message, message
    _both_survive(colliding_board)


def test_backlog_write_task_refuses_the_collision_and_writes_neither_file(colliding_board):
    """`backlog_write_task(task_id=3691)` answers the pair and changes no bytes.

    This is the call shape from the incident — an append to an id that has two files.
    The refusal must not be a `None` from a lookup, because every caller reads a
    missing file as "no such item" and would answer *that* and go on to create a third
    file for the id.
    """
    # No await: `_handle_write` is a SYNC handler returning a JSON string
    # (`agent_mcp/backlog.py:333`, `def _handle_write(args: dict) -> str`), while the two
    # board routes above it ARE coroutines. Awaiting a str is a TypeError — and the
    # refusal still fires behind it, which is how this node went red on the first gate
    # while the captured WARNING in its own log line proved the guard had answered.
    result = json.loads(MCP._handle_write(
        {"task_id": ID, "description": "appended finding", "description_mode": "append"}))
    assert result["success"] is False, f"a colliding id must not report success: {result}"
    error = result["error"]
    assert PHANTOM in error and REAL in error, \
        f"the tool error must name both files for the operator to resolve: {error}"
    _both_survive(colliding_board)


def test_backlog_get_task_refuses_rather_than_quoting_a_phantom_as_the_item(colliding_board):
    """Reading is refused for the same reason: an answer from one of the pair is where
    the phantom becomes the item, and then gets written back as it."""
    # Sync for the same reason as the write handler (`agent_mcp/backlog.py:303`).
    result = json.loads(MCP._handle_get({"task_id": ID}))
    assert result["success"] is False, str(result)
    assert PHANTOM in result["error"] and REAL in result["error"], result["error"]
    _both_survive(colliding_board)


# ── The two lanes now share one predicate ───────────────────────────────────────

def test_neither_lane_keeps_its_own_copy_of_the_lookup():
    """The fix is one predicate, not two that were written to agree.

    Both resolvers used to carry their own `glob("*.md")` first-match loop; the drift
    between the board route and the MCP lane is what let the reader-side rule land
    without the writer-side one. Each function's body now delegates to
    `resolve_backlog_id_file`, and neither may glob the board itself.
    """
    import inspect

    for func in (BR._backlog_find_file, MCP.load_task):
        own_body = inspect.getsource(func)
        assert "resolve_backlog_id_file" in own_body, \
            f"{func.__qualname__} no longer asks the shared resolver"
        assert '.glob("*.md")' not in own_body, \
            f"{func.__qualname__} has its own glob lookup back — two predicates again"
