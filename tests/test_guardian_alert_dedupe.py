"""The guardian's backlog duplicate guard (#2080), against a real loopback board.

Until 2026-10-29 nothing pinned this seam, and production showed why that mattered:
the hourly data-root shadow alert filed four copies of one incident (#3551, #3558,
#3559, #3566) while an open copy stood on the board. The guard was right whenever it
got an answer; the board read did not get one — `?q=` over a cold parse cache took
18.6 s against a 5 s timeout, and `None` means file anyway. These nodes pin the three
outcomes a filing can have and that each leaves a different record on disk:

  * an OPEN exact-name row on an answering board → nothing is POSTed, the skip is counted;
  * a board that answers with no open match → the item is POSTed, no record is written;
  * a board that cannot be read → the item is still POSTed (silence is the one failure
    the guard must never produce), and the failed read is recorded with its reason.
"""
from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARDIAN_DIR = ROOT / "agent-services" / "guardian"
for _p in (str(ROOT), str(GUARDIAN_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import notify  # noqa: E402

TITLE = "Content-empty databases under the data root are shadowing Lloyd's populated stores"
NAME = f"[guardian] {TITLE}"[:120]


class _Board:
    """A loopback stand-in for the two backlog routes the guard touches."""

    def __init__(self, rows=None, list_status=200):
        self.rows = rows if rows is not None else []
        self.list_status = list_status
        self.gets: list[str] = []
        self.posts: list[dict] = []
        board = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def _send(self, status, payload):
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                board.gets.append(self.path)
                if board.list_status != 200:
                    self._send(board.list_status, {"detail": "boom"})
                else:
                    self._send(200, board.rows)

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                board.posts.append(json.loads(self.rfile.read(n) or b"{}"))
                self._send(200, {"success": True, "id": 9000 + len(board.posts)})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def board_factory():
    made = []

    def make(**kw):
        b = _Board(**kw)
        made.append(b)
        return b

    yield make
    for b in made:
        b.close()


def _notifier(state_dir: Path, url: str) -> notify.Notifier:
    return notify.Notifier(ledger=state_dir / "ledger.jsonl", state_dir=state_dir,
                           vault_root=str(state_dir / "vault"), backend_url=url,
                           external=False)


def _record(state_dir: Path) -> dict:
    path = state_dir / notify.BACKLOG_SKIP_STATE_NAME
    return json.loads(path.read_text()) if path.exists() else {}


def test_an_open_copy_on_an_answering_board_files_nothing_and_counts_the_skip(
        tmp_path, board_factory):
    board = board_factory(rows=[{"id": 3558, "name": NAME, "status": "up_next"}])
    n = _notifier(tmp_path, board.url)

    assert n._backlog_task(TITLE, "body", "", "") is False
    assert n._backlog_task(TITLE, "body", "", "") is False

    assert board.posts == []
    assert len(board.gets) == 2, "every filing must ask the board first"
    rec = _record(tmp_path)
    assert rec["suppressed"][NAME]["count"] == 2
    assert NAME not in rec.get("unread", {})


def test_a_board_with_only_a_closed_copy_files_and_records_nothing(tmp_path, board_factory):
    board = board_factory(rows=[{"id": 3551, "name": NAME, "status": "done"}])
    n = _notifier(tmp_path, board.url)

    assert n._backlog_task(TITLE, "body", "", "") is True

    assert len(board.posts) == 1 and board.posts[0]["name"] == NAME
    assert _record(tmp_path) == {}


def test_an_unreadable_board_still_files_and_records_why(tmp_path, board_factory):
    board = board_factory(list_status=500)
    n = _notifier(tmp_path, board.url)

    assert n._backlog_task(TITLE, "body", "", "") is True

    assert len(board.gets) == 1 and len(board.posts) == 1
    rec = _record(tmp_path)
    assert rec["unread"][NAME]["count"] == 1
    assert "500" in rec["unread"][NAME]["last_reason"]
    assert NAME not in rec.get("suppressed", {})


def test_a_read_that_times_out_is_recorded_as_a_timeout(tmp_path, board_factory, monkeypatch):
    board = board_factory()
    n = _notifier(tmp_path, board.url)

    def slow(*a, **kw):
        if kw.get("timeout") == notify.BACKLOG_READ_TIMEOUT_SECONDS:
            raise TimeoutError("timed out")
        return real(*a, **kw)

    real = notify.urllib.request.urlopen
    monkeypatch.setattr(notify.urllib.request, "urlopen", slow)

    assert n._backlog_task(TITLE, "body", "", "") is True
    assert "TimeoutError" in _record(tmp_path)["unread"][NAME]["last_reason"]


def test_the_read_budget_covers_a_cold_board_parse_inside_the_watchdog():
    # 18.6 s measured cold on 2026-10-29; the unit's WatchdogSec is 90.
    assert 20 <= notify.BACKLOG_READ_TIMEOUT_SECONDS <= 45
