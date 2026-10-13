"""The backlog has a board list, and every writer holds to it.

`app/backlog_boards.py` is the list. Before it, a board was any string a
writer typed: by 2026-09-27 nineteen items sat on eight boards nobody had set
up, ten of them guardian alerts that fell through the create route's
`"default"` fallback, and the four open ones were invisible to the loop, which
reads `board: lloyd` only (`scripts/automod/backlog.py::DEFAULT_BOARDS`).

Each writer is pinned below: the `backlog_write_task` tool (create and move),
Mission Control's create and update routes (by name and by id), and
`scripts/automod/backlog.py::new_item`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from fastapi import HTTPException

from agent_mcp import backlog as BL
from app import backlog_boards as BB
from app.routers import backlog as BR
from scripts.automod import backlog as AB


class _Req:
    def __init__(self, payload: dict):
        self._payload = payload

    async def json(self) -> dict:
        return self._payload


def _write(d: Path, task_id: int, *, board: str | None) -> Path:
    """One task file. ``board=None`` writes no ``board:`` key, which every reader
    here calls the literal board ``default`` — a hand-edited or restored shape
    that ``check_board`` refuses to write."""
    board_line = f"board: {board}\n" if board is not None else ""
    p = d / f"{task_id}-a-task.md"
    p.write_text(
        f"---\ntype: backlog\nsegment: backlog\nstatus: draft\n"
        f"priority: low\n{board_line}blocked: false\nassigned: false\n"
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
    monkeypatch.setattr(BR, "_FM_CACHE", {})
    monkeypatch.setattr(BL, "BACKLOG_DIR", d)
    return d


# ── The list ────────────────────────────────────────────────────────────────

def test_the_list_is_the_three_boards_and_lloyd_is_the_default():
    assert BB.BOARDS == ("lloyd", "personal", "alfie")
    assert BB.DEFAULT_BOARD in BB.BOARDS
    assert AB.DEFAULT_BOARDS == (BB.DEFAULT_BOARD,), \
        "the loop reads the board a boardless create lands on"


@pytest.mark.parametrize("name", ["default", "skills", "Lloyd", "", None, 3])
def test_check_board_refuses_anything_off_the_list(name):
    with pytest.raises(BB.UnknownBoard):
        BB.check_board(name)


def test_check_board_strips_whitespace():
    assert BB.check_board("  alfie ") == "alfie"


# ── backlog_write_task ──────────────────────────────────────────────────────

def test_the_tool_advertises_the_list():
    import asyncio
    tools = asyncio.run(BL.list_tools())
    schema = next(t for t in tools if t.name == "backlog_write_task").input_schema
    assert schema["properties"]["board"]["enum"] == list(BB.BOARDS)


def test_the_tool_refuses_a_create_on_an_unknown_board(board_dir):
    out = json.loads(BL._handle_write({"name": "x", "description": "y", "board": "skills"}))
    assert out["success"] is False and "Unknown board" in out["error"]
    assert list(board_dir.glob("*.md")) == []


def test_the_tool_creates_on_a_listed_board(board_dir):
    out = json.loads(BL._handle_write({"name": "x", "description": "y", "board": "personal"}))
    assert out["success"], out
    assert _fm(next(board_dir.glob("*.md")))["board"] == "personal"


def test_the_tool_refuses_a_move_to_an_unknown_board(board_dir):
    f = _write(board_dir, 1, board="lloyd")
    out = json.loads(BL._handle_write({"task_id": 1, "board": "memory"}))
    assert out["success"] is False
    assert _fm(f)["board"] == "lloyd"


# ── Mission Control's routes ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_create_naming_no_board_lands_on_the_default(board_dir):
    """The guardian's alerts post no board; they used to land on `default`."""
    resp = await BR.backlog_task_create(_Req({"name": "[guardian] something"}))
    assert resp.status_code == 200, resp.body
    assert _fm(next(board_dir.glob("*.md")))["board"] == BB.DEFAULT_BOARD


@pytest.mark.asyncio
async def test_the_create_route_refuses_an_unknown_board(board_dir):
    with pytest.raises(HTTPException) as ei:
        await BR.backlog_task_create(_Req({"name": "x", "board": "Autonomy"}))
    assert ei.value.status_code == 400
    assert list(board_dir.glob("*.md")) == []


@pytest.mark.asyncio
async def test_the_update_route_refuses_an_unknown_board_by_name(board_dir):
    f = _write(board_dir, 1, board="lloyd")
    with pytest.raises(HTTPException) as ei:
        await BR.backlog_task_update(_Req({"id": 1, "board": "vault"}))
    assert ei.value.status_code == 400
    assert _fm(f)["board"] == "lloyd"


@pytest.mark.asyncio
async def test_the_update_route_refuses_a_stray_board_by_id(board_dir):
    """A stray board still on disk has an id; moving onto it is still refused."""
    f = _write(board_dir, 1, board="lloyd")
    _write(board_dir, 2, board="default")
    stray_id = BR._backlog_board_map()["default"]
    with pytest.raises(HTTPException) as ei:
        await BR.backlog_task_update(_Req({"id": 1, "board_id": stray_id}))
    assert ei.value.status_code == 400
    assert _fm(f)["board"] == "lloyd"


def test_the_board_list_shows_every_board_even_empty_and_a_stray(board_dir):
    _write(board_dir, 1, board="lloyd")
    _write(board_dir, 2, board="default")
    boards = {b["name"]: b["tasks_count"]
              for b in json.loads(BR.backlog_boards().body)}
    assert boards == {"lloyd": 1, "personal": 0, "alfie": 0, "default": 1}


# ── #2659: the ids the board route hands out ────────────────────────────────

def test_every_listed_board_carries_a_unique_id_and_filters_by_it(board_dir):
    """A stray board stays reachable: an id of its own, and that id works.

    A configured board's id comes from `BOARDS` alone, so the two strays here —
    one hand-edited item naming `Autonomy`, one file with no `board:` key, which
    every reader calls the literal board `default` — are allocated *above* that
    block rather than sorting into the middle of it. So every board the picker
    lists has a distinct non-zero id, and `?board_id=<that id>` returns exactly
    that board's rows. A stray's id is rank-based among strays, which is safe
    only because a stray is unwritable (`check_board`); the configured ids, the
    ones a write can land on, are not rank-based at all.
    """
    _write(board_dir, 1, board="lloyd")
    _write(board_dir, 2, board="lloyd")
    _write(board_dir, 3, board="Autonomy")
    _write(board_dir, 4, board=None)                    # reads as `default`

    boards = json.loads(bytes(BR.backlog_boards().body))
    ids = [b["id"] for b in boards]
    assert len(ids) == len(set(ids)), f"board ids are not unique: {boards}"
    assert all(i > 0 for i in ids), f"a board was listed with a zero id: {boards}"

    by_name = {b["name"]: b["id"] for b in boards}
    # A stray is allocated above the configured block, so it can never hold the
    # id a client read for `lloyd`, `personal` or `alfie`. Over a corpus of just
    # these four files the strays used to sort first and take ids 1 and 3.
    assert {b: by_name[b] for b in BB.BOARDS} == {"alfie": 1, "lloyd": 2, "personal": 3}
    for stray in ("Autonomy", "default"):
        assert by_name[stray] > len(BB.BOARDS), \
            f"a stray must not hold a configured id: {boards}"

    for name, expected in (("Autonomy", {3}), ("default", {4}), ("lloyd", {1, 2})):
        rows = json.loads(bytes(BR.backlog_tasks(board_id=str(by_name[name])).body))
        assert {r["id"] for r in rows} == expected, \
            f"?board_id={by_name[name]} should return {name}'s rows {expected}"


def test_the_list_filter_still_accepts_a_board_name(board_dir):
    """`?board_id=lloyd` is a name in an id-shaped parameter, and it stays valid.

    `agent-services/guardian/notify.py` sets `BACKLOG_ITEM_BOARD = "lloyd"`
    (line 173) and asks `?board_id=lloyd&q=<title>` for its duplicate-item check
    (line 842), so making the parameter strictly numeric would blind the
    guardian's dedupe — it would read zero rows and file a duplicate alert.
    """
    _write(board_dir, 1, board="lloyd")
    _write(board_dir, 2, board="alfie")
    _write(board_dir, 3, board="Autonomy")

    rows = json.loads(bytes(BR.backlog_tasks(board_id=BB.DEFAULT_BOARD).body))
    assert [r["id"] for r in rows] == [1]
    assert {r["board"] for r in rows} == {"lloyd"}


# ── The loop's own writer ───────────────────────────────────────────────────

def test_new_item_refuses_an_unknown_board(tmp_path):
    with pytest.raises(BB.UnknownBoard):
        AB.new_item("x", board="skills", backlog_dir=tmp_path)
    assert list(tmp_path.glob("*.md")) == []


def test_new_item_defaults_to_the_loops_board(tmp_path):
    item = AB.new_item("x", backlog_dir=tmp_path)
    assert item.board == BB.DEFAULT_BOARD
