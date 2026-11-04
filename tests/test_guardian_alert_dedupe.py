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

The file also owns the other half of that guard's life cycle (#3757). An OPEN row is
what silences the next filing (`_open_duplicate_on_board`, and `voiceloss.py:123` puts
`draft` in `OPEN_STATUSES`), so an all-clear that seals an incident has to take the row
that alert filed out of the OPEN set, or the board goes on asserting a service the
watchdog measured healthy ten seconds ago — which is #3755, filed 09:18:40.750499Z and
cleared in ALERT.md at 09:18:50Z, still `status: draft` when triage read it at 09:20Z.
`Notifier.resolve` performs that retraction, and the nodes at the foot of this file pin
what it posts: clauses 1, 2 and 3 of #3757.

The process boundary throughout is HTTP: `notify.py` runs inside
`lloyd-guardian.service` and the board answers inside the backend, so both routes a
filing and a retraction touch are served here by one loopback stand-in, and every
assertion is about the bytes that cross it.
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

import guardian  # noqa: E402
import notify  # noqa: E402
import voiceloss  # noqa: E402

TITLE = "Content-empty databases under the data root are shadowing Lloyd's populated stores"
NAME = f"[guardian] {TITLE}"[:120]
#: The service-down family's one fixed title (`guardian.py:93`), and the family #3757 is
#: about: the liveness recovery branch (`guardian.py:2758-2775`) and the boot reconcile
#: `_seal_measured_incident` (`guardian.py:3014`) both retract it through
#: `Notifier.resolve` and through nothing else.
SERVICE_DOWN_TITLE = guardian.SERVICE_DOWN_ALERT_TITLE
SERVICE_DOWN_NAME = f"[guardian] {SERVICE_DOWN_TITLE}"[:120]
#: The two POST routes, spelled as `app/routers/backlog.py` registers them, so the
#: stand-in can answer each with the other's shape and a caller on the wrong route is
#: visible rather than plausible.
CREATE_PATH = "/api/backlog/task-create"
UPDATE_PATH = "/api/backlog/task-update"
#: `level="error"` with no `trigger` and no `needs_human` sits BELOW `alert`'s filing
#: gate (`alert`'s own `if not self.external` branch), so a node that only needs the
#: incident OPEN on ALERT.md
#: does not also POST a create — every POST the retraction nodes see is a close.
LEVEL = "error"
BODY = "lloyd-mc:lloyd-backend: FATAL: can't find command (repeated 24 times)"
#: The retraction note the recovery branch itself builds (`guardian.py:2773`).
NOTE = "liveness read came back healthy: lloyd-mc:lloyd-backend."


class _Board:
    """A loopback stand-in for the two backlog routes a filing and a retraction touch."""

    def __init__(self, rows=None, list_status=200):
        self.rows = rows if rows is not None else []
        self.list_status = list_status
        self.gets: list[str] = []
        self.posts: list[dict] = []
        # The path each POST arrived on, parallel to `posts`. The two routes answer
        # differently and a retraction must not be able to satisfy its clause by
        # hitting the create, so the nodes below name the path, not just the body.
        self.post_paths: list[str] = []
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
                board.post_paths.append(self.path)
                # Each route's own reply shape, from app/routers/backlog.py:
                # `backlog_task_create` answers `{"success": true, "id": N}` and
                # `backlog_task_update` answers
                # `{"success": true, "description_ignored": bool}`. Anything else is
                # a 404, so a caller that posts to a path the guardian does not own
                # cannot read as delivered.
                if self.path == CREATE_PATH:
                    self._send(200, {"success": True, "id": 9000 + len(board.posts)})
                elif self.path == UPDATE_PATH:
                    self._send(200, {"success": True, "description_ignored": False})
                else:
                    self._send(404, {"detail": f"no route {self.path}"})

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


def _resolving_notifier(state_dir: Path, url: str) -> notify.Notifier:
    """A Notifier that fans out for real, so its `resolve` reaches the board.

    `external=True`, because the retraction sits BELOW `resolve`'s external gate
    (the `if not self.external` near its end) and inherits the drill's board-silence from it — which is what
    `tests/test_guardian_alert_retraction.py::test_a_drill_seals_alert_md_and_never_posts_a_close`
    pins. The room channels are muted for every node by `tests/conftest.py`, and
    `state_dir / "vault"` deliberately has no `memory/`, so `alert` here opens the
    incident on ALERT.md and touches no daily note unless a node writes one itself.
    """
    return notify.Notifier(ledger=state_dir / "ledger.jsonl", state_dir=state_dir,
                           vault_root=str(state_dir / "vault"), backend_url=url,
                           external=True)


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


def _closes(board) -> list[dict]:
    """The bodies POSTed to `task-update`, i.e. the closes #3757 is about.

    Path-filtered rather than taken from `posts` outright: `task-create` is the route
    that would file a NEW item, and a retraction that reached it instead would post a
    name rather than an id — a different bug that `posts` alone could not tell from a
    close.
    """
    return [body for path, body in zip(board.post_paths, board.posts)
            if path == UPDATE_PATH]


# ── #3757: the all-clear has to retract the row the alert filed ────────────────
#
# The three nodes below are the same guard seen from the other end. An OPEN exact-name
# row is what `_open_duplicate_on_board` suppresses on, so a row that a cleared incident
# left standing is not merely untidy: it is the reason the NEXT, real outage files
# nothing. #3755 is the instance — ALERT.md's `cleared:` line is stamped
# 2026-11-04T01:18:50.338811-08:00 (09:18:50Z) against an item the same alert filed with
# `created: '2026-11-04T09:18:40.750499'`, `status: draft`, `priority: high`, and
# `curl -w '%{http_code} %{time_total}s' http://localhost:8080/health` answered
# `200 0.366s` from the same pid (1982995) that the alert called unresponsive.


def test_an_all_clear_closes_the_open_row_that_alert_filed_by_id(tmp_path, board_factory):
    """#3757 clause 1: sealing an incident moves the OPEN row off the OPEN set.

    The board is seeded with the exact-name row at `status: draft`, which is open —
    `voiceloss.OPEN_STATUSES = frozenset(PIPELINE_STATUSES) - {"done"}`
    (`voiceloss.py:123`, `PIPELINE_STATUSES` = draft/up_next/in_progress/done at :121) —
    which is the state #3755 was left in. `alert` writes the incident, `resolve` seals
    it, and what must cross the wire is one `task-update` naming that row's id with a
    status outside the OPEN set.
    """
    board = board_factory(rows=[{"id": 3755, "name": SERVICE_DOWN_NAME,
                                 "status": "draft"}])
    n = _resolving_notifier(tmp_path, board.url)
    n.alert(LEVEL, SERVICE_DOWN_TITLE, BODY)
    assert board.posts == [], "nothing here may file a new item"

    n.resolve(SERVICE_DOWN_TITLE, NOTE)

    closes = _closes(board)
    assert len(closes) == 1, f"expected exactly one close, got {closes}"
    assert closes[0]["id"] == 3755, (
        "the close must address the row by the id the board read returned, because "
        f"`task-update` looks a task up by id: {closes[0]}")
    assert closes[0]["status"] not in voiceloss.OPEN_STATUSES, (
        f"`{closes[0]['status']}` is still an open status, so the row would go on "
        "silencing the next filing")


def test_a_clear_with_no_open_row_to_close_posts_nothing(tmp_path, board_factory):
    """#3757 clause 2: an answering board with nothing exact-name and OPEN gets no POST.

    Two board states, because the ordinary case and the already-reconciled case must
    not be confused: an empty match, and a board holding only a `done` copy of the same
    name — which is what a correctly-retracted incident looks like, and re-posting a
    close onto it would write a second status move and a second `activity_log` line into
    an item the guardian already closed.
    """
    for i, rows in enumerate(([], [{"id": 3755, "name": SERVICE_DOWN_NAME,
                                    "status": "done"}])):
        # A state dir per iteration, because ALERT.md is one slot that stays sealed:
        # sharing it would leave the second round with nothing to seal, and "posted
        # nothing" would then be true of a retraction that never ran.
        board = board_factory(rows=rows)
        n = _resolving_notifier(tmp_path / f"s{i}", board.url)
        n.alert(LEVEL, SERVICE_DOWN_TITLE, BODY)

        n.resolve(SERVICE_DOWN_TITLE, NOTE)

        assert board.gets, "the seal happened, so the row lookup is owed a read"
        assert _closes(board) == [], (
            f"a board holding {rows} has no OPEN exact-name row to close")
        assert board.posts == []


def test_a_close_names_only_id_and_status_and_ignores_a_row_that_merely_quotes(
        tmp_path, board_factory):
    """#3757 clause 3: the close carries no body, and only the exact-name row is closed.

    Two separate guarantees ride on one POST. First no `description` key: `GET
    /api/backlog/tasks` returns `description_snippet` and NO `description`
    (`app/routers/backlog.py:421-453`), and `task-update` replaces the whole body and
    drops a shorter one outright (`app/routers/backlog.py:732-733`), so a body derived
    from the list row would silently truncate the item's evidence. Second exactness:
    the `?q=` read also returns rows that quote the alert in their BODY, and #2080's
    rule (reaffirmed for this family by #3229's owed-check, 2026-10-27: "I will not
    loosen #2080's exact-name rule") is that those are not the same incident. So a
    board holding one exact-name OPEN row and one OPEN row that quotes the alert yields
    exactly one POST.
    """
    quoting = {"id": 4001, "name": "[guardian] liveness: 3 consecutive check failures",
               "status": "up_next",
               "description_snippet": f"… while {_quoted_title()} …"}
    board = board_factory(rows=[{"id": 3755, "name": SERVICE_DOWN_NAME,
                                 "status": "draft"},
                                quoting])
    n = _resolving_notifier(tmp_path, board.url)
    n.alert(LEVEL, SERVICE_DOWN_TITLE, BODY)

    n.resolve(SERVICE_DOWN_TITLE, NOTE)

    closes = _closes(board)
    assert len(closes) == 1, (
        "a row that merely quotes the alert is a different incident, and closing it "
        f"would be #2080's matcher loosened by the back door: {closes}")
    assert closes[0]["id"] == 3755 and "description" not in closes[0], (
        f"the close may carry id and status only: {closes[0]}")
    assert set(closes[0]) == {"id", "status"}, (
        f"unexpected keys would be read as an edit, not a close: {closes[0]}")


def _quoted_title() -> str:
    """The alert's own title, as it appears inside ANOTHER incident's body.

    A function rather than a literal so the fixture stays honest: it is the same string
    the guard searches on, so `?q=` legitimately returns the row that quotes it and the
    node is testing the match, not a conveniently-different string.
    """
    return SERVICE_DOWN_TITLE


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
