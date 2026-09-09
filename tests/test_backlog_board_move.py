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

@pytest.mark.asyncio
async def test_listing_carries_the_board_name_beside_the_id(board_dir):
    """The modal seeds its select from `task.board`, so it has to be there.

    Reverse-mapping the positional id through the board list would reintroduce
    exactly the drift the name exists to avoid.
    """
    _write(board_dir, 1, board="lloyd")
    _write(board_dir, 2, board="alan")
    tasks = {t["id"]: t for t in json.loads(bytes((await BR.backlog_tasks()).body))}
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
