"""Two creators never take the same backlog id.

Three lanes create item files, as three processes, and each allocated
`max(id) + 1` with nothing held between the scan and the create. `new_item`'s
`O_EXCL` excluded `<id>-<slug>.md`, so two different names at one id both
succeeded — and autotriage runs two sessions at a time. Two files at one id is
the state `resolve_backlog_id_file` refuses to read (the 2026-11-01 incident on
item 3691). All three now allocate under `app.backlog_write.id_allocation_lock`.
"""

from __future__ import annotations

import json
import re
import threading
from collections import Counter

import pytest

from agent_mcp import backlog as BL
from agent_mcp import backlog_similar as SIM
from app import backlog_write as BW
from scripts.automod import backlog as B


def _ids(d):
    return [int(re.match(r"^(\d+)-", p.name).group(1)) for p in d.glob("*.md")]


def _race(fns):
    start = threading.Barrier(len(fns))
    errors = []

    def run(fn):
        try:
            start.wait(timeout=10)
            fn()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(fn,)) for fn in fns]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors


def test_next_id_is_one_past_the_highest_and_one_on_an_empty_board(tmp_path):
    assert BW.next_backlog_id(tmp_path / "absent") == 1
    d = tmp_path / "backlog"
    d.mkdir()
    assert BW.next_backlog_id(d) == 1
    (d / "7-a.md").write_text("x")
    (d / "12_b.md").write_text("x")
    (d / "notes.md").write_text("x")
    assert BW.next_backlog_id(d) == 13


def test_concurrent_new_items_with_different_names_get_different_ids(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(B, "BACKLOG_DIR", d)
    _race([lambda n=n: B.new_item(f"finding number {n}", "Body.", backlog_dir=d)
           for n in range(12)])
    ids = _ids(d)
    assert len(ids) == 12 and sorted(ids) == list(range(1, 13)), Counter(ids)


def test_the_scan_alone_is_what_used_to_collide(tmp_path):
    """The control: without the lock, a scan held across another create answers
    an id that is already taken."""
    d = tmp_path / "backlog"
    d.mkdir()
    stale = BW.next_backlog_id(d)
    (d / f"{stale}-first.md").write_text("x")
    assert stale == 1 and BW.next_backlog_id(d) == 2


@pytest.fixture
def board(tmp_path, monkeypatch):
    d = tmp_path / "backlog"
    d.mkdir()
    monkeypatch.setattr(BL, "BACKLOG_DIR", d)
    monkeypatch.setattr(SIM, "DEDUPE_LOG", tmp_path / "dedupe.jsonl")
    monkeypatch.setattr(SIM, "dedupe_config", lambda: {**SIM.DEFAULTS, "enabled": False})
    return d


def test_the_mcp_create_allocates_at_the_write_not_at_the_scan(board, monkeypatch):
    """Another creator lands between this call's early scan and its save: the
    item must take the next id, not the one it scanned."""
    real = BL.save_task
    planted = {}

    def dedupe_then_someone_else_creates(args):
        (board / "1-someone-else.md").write_text("---\nstatus: draft\n---\n\n# Someone else\n")
        planted["done"] = True
        return [], None, None

    monkeypatch.setattr(BL, "_dedupe", dedupe_then_someone_else_creates)
    out = json.loads(BL._handle_write({"name": "My finding", "board": "lloyd",
                                       "description": "A finding with a body."}))
    assert planted and out["success"] and out["task_id"] == 2
    assert sorted(p.name for p in board.glob("*.md")) == ["1-someone-else.md", "2-my-finding.md"]
    assert real is BL.save_task
    text = (board / "2-my-finding.md").read_text()
    assert "position: 2000" in text


def test_concurrent_mcp_creates_get_different_ids(board):
    _race([lambda n=n: BL._handle_write({"name": f"Finding {n} on subsystem {n}", "board": "lloyd",
                                         "description": "A finding with a body."})
           for n in range(8)])
    ids = _ids(board)
    assert sorted(ids) == list(range(1, 9)), Counter(ids)


def test_the_loop_and_the_tool_share_one_lock(board, monkeypatch):
    monkeypatch.setattr(B, "BACKLOG_DIR", board)
    fns = []
    for n in range(6):
        fns.append(lambda n=n: B.new_item(f"loop finding {n}", "Body.", backlog_dir=board))
        fns.append(lambda n=n: BL._handle_write({"name": f"Tool finding {n}", "board": "lloyd",
                                                 "description": "A finding with a body."}))
    _race(fns)
    ids = _ids(board)
    assert sorted(ids) == list(range(1, 13)), Counter(ids)
