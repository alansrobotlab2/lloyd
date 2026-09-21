"""Moving a backlog item to another board, from the Mission Control modal.

The task modal can now change an item's board. The board it names is the
**name**, not the numeric ``board_id`` the listing also carries, and that is
the whole point of these tests.

``board_id`` is positional: ``_backlog_board_map`` assigns ``idx + 1`` over
``sorted(names)``, and the names are whatever the files on disk happen to say.
So an id is only meaningful relative to the board list it was read from, and
that list changes shape whenever a board appears or vanishes — which is exactly
what a board move can cause, by taking the last item off a board. A browser tab
holding a board list from thirty seconds ago then sends an id that is still
*valid*, for a different board, and the item lands somewhere nobody asked for.
The name cannot drift that way; the vault, ``agent_mcp/backlog.py`` and the
frontmatter all key on it already.

The second half is the silent no-op. ``board_id`` used to resolve through
``id_to_name.get(data["board_id"], fm.get("board", "default"))`` — an
unresolvable id fell back to the board the task was already on and returned
``{"success": True}``. A move the user asked for, reported as done, that did
not happen. It is a 400 now, like an invalid status.

The last section is the other thing this route writes: ``status``. A close or a
repost from the board used to land as a bare ``fm["status"] =`` — no
``activity_log`` line, no reason, no ``completed``, and none of the tag handling
the loop's own writer does — so the one writer a person uses was the one writer
that left no trace (#1023). Those tests pin what the route must now write; the
``completed``/non-status/one-definition clauses are pinned in
``tests/test_backlog_route_done_window.py``.
"""

import json
import re
from pathlib import Path

import pytest
import yaml

from app.routers import backlog as BR
from fastapi import HTTPException


class _Req:
    def __init__(self, payload: dict):
        self._payload = payload

    async def json(self) -> dict:
        return self._payload


def _write(dirpath: Path, task_id: int, *, board: str) -> Path:
    p = dirpath / f"{task_id}-a-task.md"
    p.write_text(
        f"---\ntype: backlog\nsegment: backlog\nstatus: draft\n"
        f"priority: medium\nboard: {board}\nblocked: false\nassigned: false\n"
        f"position: {task_id * 1000}\n---\n\n# A task\n\nBody.\n",
        encoding="utf-8",
    )
    return p


def _fm(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8").split("---", 2)[1])


@pytest.fixture
def board_dir(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(BR, "_BACKLOG_DIR", d)
    return d


# ── The listing carries the name the modal edits ─────────────────────────────

def test_listing_carries_the_board_name_beside_the_id(board_dir):
    """The modal seeds its select from `task.board`, so it has to be there.

    Reverse-mapping the positional id through the board list would reintroduce
    exactly the drift the name exists to avoid.

    Sync, and calls the handler without `await`: the list route became a plain
    `def` so FastAPI runs it in the threadpool instead of on the event loop
    (item #1199 cause 2), which is pinned in tests/test_backlog_route_offload.py.
    """
    _write(board_dir, 1, board="lloyd")
    _write(board_dir, 2, board="alan")
    tasks = {t["id"]: t for t in json.loads(bytes(BR.backlog_tasks().body))}
    assert tasks[1]["board"] == "lloyd"
    assert tasks[2]["board"] == "alan"
    # sorted(["alan", "lloyd"]) -> alan=1, lloyd=2
    assert tasks[1]["board_id"] == 2
    assert tasks[2]["board_id"] == 1


# ── The move itself ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_named_board_moves_the_item(board_dir):
    f = _write(board_dir, 1, board="lloyd")
    _write(board_dir, 2, board="alan")
    resp = await BR.backlog_task_update(_Req({"id": 1, "board": "alan"}))
    assert resp.status_code == 200, resp.body
    assert _fm(f)["board"] == "alan"


@pytest.mark.asyncio
async def test_the_name_survives_a_renumbering_the_id_would_not(board_dir):
    """The drift this is all for, played out.

    A tab reads the board list while `alan`, `lloyd` and `zoo` exist, so
    lloyd is id 2. `zoo`'s only item is then moved away and `zoo` stops
    existing — but the stale tab is not the one that changed anything, and
    `alan`, `lloyd` still puts lloyd at 2, so an id-based move is only wrong
    when the vanished board sorted *before* the target. Use `aaa` for that.
    """
    _write(board_dir, 1, board="aaa")     # id 1
    _write(board_dir, 2, board="lloyd")   # id 3 (aaa, alan, lloyd)
    _write(board_dir, 3, board="alan")    # id 2
    item = _write(board_dir, 4, board="alan")

    # The tab read the list here: aaa=1, alan=2, lloyd=3. Now `aaa` empties.
    await BR.backlog_task_update(_Req({"id": 1, "board": "alan"}))
    assert _fm(board_dir / "1-a-task.md")["board"] == "alan"

    # Boards are now alan=1, lloyd=2. The stale tab's "lloyd" is id 3, which
    # no longer resolves — a 400, not a silent no-op and not a wrong board.
    with pytest.raises(HTTPException) as exc:
        await BR.backlog_task_update(_Req({"id": 4, "board_id": 3}))
    assert exc.value.status_code == 400
    assert _fm(item)["board"] == "alan", "the item must not have moved"

    # The name is immune to the renumbering.
    await BR.backlog_task_update(_Req({"id": 4, "board": "lloyd"}))
    assert _fm(item)["board"] == "lloyd"


@pytest.mark.asyncio
async def test_an_unresolvable_board_id_is_refused_not_ignored(board_dir):
    f = _write(board_dir, 1, board="lloyd")
    for bogus in (99, "lloyd", None, ""):
        with pytest.raises(HTTPException) as exc:
            await BR.backlog_task_update(_Req({"id": 1, "board_id": bogus}))
        assert exc.value.status_code == 400, bogus
    assert _fm(f)["board"] == "lloyd"


@pytest.mark.asyncio
async def test_a_resolvable_board_id_still_works(board_dir):
    """The compatibility path: nothing that sent an id before is broken."""
    f = _write(board_dir, 1, board="lloyd")
    _write(board_dir, 2, board="alan")
    await BR.backlog_task_update(_Req({"id": 1, "board_id": 1}))  # alan
    assert _fm(f)["board"] == "alan"


@pytest.mark.asyncio
async def test_an_empty_board_name_is_refused(board_dir):
    """A blank select must not write `board: ''` and orphan the item."""
    f = _write(board_dir, 1, board="lloyd")
    for bad in ("", "   ", None, 3):
        with pytest.raises(HTTPException) as exc:
            await BR.backlog_task_update(_Req({"id": 1, "board": bad}))
        assert exc.value.status_code == 400, bad
    assert _fm(f)["board"] == "lloyd"


@pytest.mark.asyncio
async def test_the_name_wins_when_both_are_sent(board_dir):
    _write(board_dir, 1, board="alan")
    f = _write(board_dir, 2, board="lloyd")
    await BR.backlog_task_update(_Req({"id": 2, "board": "alan", "board_id": 2}))
    assert _fm(f)["board"] == "alan"


@pytest.mark.asyncio
async def test_a_move_leaves_the_rest_of_the_frontmatter_alone(board_dir):
    """The modal sends the whole form, so a move rides with everything else."""
    f = _write(board_dir, 1, board="lloyd")
    _write(board_dir, 2, board="alan")
    await BR.backlog_task_update(_Req({
        "id": 1, "board": "alan", "name": "Renamed", "status": "up_next",
        "priority": "high", "blocked": True,
    }))
    fm = _fm(f)
    assert fm["board"] == "alan"
    assert fm["status"] == "up_next"
    assert fm["priority"] == "high"
    assert fm["blocked"] is True
    assert fm["type"] == "backlog", "OKF type must survive an update"
    assert "# Renamed" in f.read_text(encoding="utf-8")


# ── Create ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_takes_a_board_name(board_dir):
    _write(board_dir, 1, board="lloyd")
    resp = await BR.backlog_task_create(_Req({"name": "New one", "board": "alan"}))
    assert resp.status_code == 200, resp.body
    created = [f for f in board_dir.glob("*.md") if not f.name.startswith("1-")][0]
    assert _fm(created)["board"] == "alan"


@pytest.mark.asyncio
async def test_create_still_falls_back_when_no_board_is_named(board_dir):
    """Unlike an update, a create with no resolvable board has to land somewhere."""
    resp = await BR.backlog_task_create(_Req({"name": "New one"}))
    assert resp.status_code == 200, resp.body
    created = sorted(board_dir.glob("*.md"))[0]
    assert _fm(created)["board"] == "default"


@pytest.mark.asyncio
async def test_create_ignores_a_blank_board_name(board_dir):
    _write(board_dir, 1, board="lloyd")
    resp = await BR.backlog_task_create(
        _Req({"name": "New one", "board": "   ", "board_id": 1}))
    assert resp.status_code == 200, resp.body
    created = [f for f in board_dir.glob("*.md") if not f.name.startswith("1-")][0]
    assert _fm(created)["board"] == "lloyd", "falls through to the id"


# ── The frontend contract these rest on ──────────────────────────────────────

_PAGE = Path(__file__).resolve().parents[1] / "web/src/components/pages/BacklogPage.tsx"
_API = Path(__file__).resolve().parents[1] / "web/src/api.ts"


def test_the_modal_sends_a_name_and_never_a_board_id():
    src = _PAGE.read_text(encoding="utf-8")
    assert "data.board = board" in src, "create must send the board name"
    assert "updates.board = board" in src, "update must send the board name"
    assert "board_id: defaultBoardId" not in src
    assert re.search(r"\bboard_id\s*[:=]\s*board\b", src) is None, \
        "the modal must not send a positional board id"


def test_the_backlog_writers_raise_on_a_refused_write():
    """`fetch` does not reject on 4xx.

    Without this the 400 above resolves successfully, the modal calls
    `onClose()`, and a refused move looks exactly like a completed one — the
    same silent success the backend change removed, one layer up.
    """
    src = _API.read_text(encoding="utf-8")
    for route in ("task-update", "task-create", "task-delete"):
        # to the method's own closing brace, not the `headers: {...},` one
        block = src.split(f"/backlog/{route}`", 1)[1].split("\n  },", 1)[0]
        assert "if (!r.ok)" in block, f"{route} swallows a non-2xx response"
        assert "throw new Error" in block, f"{route} does not raise"


# ── #1023 clause 1: a status move through the board is attributed ─────────────
#
# `architecture/backlog.md` states as fact that "Every status move is attributed
# in the activity log and carries its reason". For a move made in Mission
# Control that was false: the route assigned `fm["status"]` and nothing else.
# These pin the close itself; the `completed` stamp, the non-status save and the
# one-shared-writer structure are in tests/test_backlog_route_done_window.py.


@pytest.mark.asyncio
async def test_closing_an_item_from_the_board_appends_a_status_move(board_dir):
    """The board's close lands in `activity_log` naming both statuses.

    This is the acceptance check for #1023: a human closing a card must leave
    the same trace the loop leaves, or the audit trail only ever narrates the
    moves the machine made.
    """
    f = _write(board_dir, 1, board="lloyd")
    _write(board_dir, 2, board="alan")
    resp = await BR.backlog_task_update(_Req({"id": 1, "status": "done"}))

    assert resp.status_code == 200, resp.body
    assert json.loads(resp.body)["success"] is True

    fm = _fm(f)
    assert fm["status"] == "done"
    log = fm["activity_log"]
    assert len(log) == 1, f"exactly one line for one move: {log!r}"
    line = str(log[0])
    assert "draft" in line, f"the line must name the status it moved from: {line!r}"
    assert "done" in line, f"the line must name the status it moved to: {line!r}"


@pytest.mark.asyncio
async def test_a_reopen_from_the_board_takes_the_needs_human_tag_off(board_dir):
    """A person reopening a card is the decision `needs-human` was waiting for.

    The loop strips the tag with the reopen (`_apply_status`'s `remove_tags`),
    and its pools then filter on `NEEDS_HUMAN_TAG not in i.tags` — so a human who
    reopened a parked item from the UI put it back on the board while leaving it
    invisible to every triage pool that would ever work it.
    """
    f = board_dir / "1-a-task.md"
    f.write_text(
        "---\ntype: backlog\nsegment: backlog\nstatus: done\n"
        "completed: '2026-03-01T12:00:00'\npriority: medium\nboard: lloyd\n"
        "tags: [needs-human]\n---\n\n# A task\n\nBody.\n",
        encoding="utf-8",
    )
    await BR.backlog_task_update(_Req({"id": 1, "status": "up_next"}))

    fm = _fm(f)
    assert fm["status"] == "up_next"
    assert "needs-human" not in (fm.get("tags") or []), fm.get("tags")
    assert any("done → up_next" in str(line) for line in fm["activity_log"])


@pytest.mark.asyncio
async def test_a_board_close_does_not_invent_an_empty_tags_key(board_dir):
    """Removing a tag is allowed; giving an item `tags: []` is not.

    The rule `close_landed` and `update_frontmatter` already follow: a writer
    that merely moved an item must not add a field it has nothing to say about,
    because `tags: []` is a value every reader now has to distinguish from "no
    tags field at all".
    """
    f = _write(board_dir, 1, board="lloyd")
    await BR.backlog_task_update(_Req({"id": 1, "status": "done"}))
    assert "tags" not in _fm(f)
