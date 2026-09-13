"""The suite does not write into the live dedupe log or ask the live daemon.

`conftest._isolate_backlog_dedupe` is the fix; this is the check that it holds
for the exact shape that polluted the log — a create through
`backlog_write_task` with only `BACKLOG_DIR` patched, as
`test_backlog_okf_frontmatter.py` does. The real path is computed here from
`Path.home()` rather than read off the module, because the module's value is
the thing the fixture replaces.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_mcp import backlog as BL
from agent_mcp import backlog_similar as SIM

REAL_LOG = Path.home() / ".local" / "state" / "lloyd-automod" / "dedupe.jsonl"


def _stat(p: Path):
    try:
        st = p.stat()
    except FileNotFoundError:
        return None
    return (st.st_size, st.st_mtime_ns)


def test_a_create_with_only_the_board_patched_leaves_the_real_log_alone(tmp_path, monkeypatch):
    board = tmp_path / "backlog"
    board.mkdir()
    monkeypatch.setattr(BL, "BACKLOG_DIR", board)
    before = _stat(REAL_LOG)
    result = json.loads(BL._handle_write({"name": "A newly written task",
                                          "description": "Body text.", "board": "lloyd"}))
    assert result.get("success"), result
    assert len(list(board.glob("*.md"))) == 1
    assert _stat(REAL_LOG) == before
    assert SIM.DEDUPE_LOG != REAL_LOG


def test_the_semantic_leg_does_not_reach_the_daemon():
    assert SIM.semantic_candidates("x") == []
