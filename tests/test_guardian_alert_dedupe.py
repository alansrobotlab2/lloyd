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
    the guard must never produce), and the failed read is recorded with its reason;
  * a CLOSED exact-name row plus a filing THIS guardian posted within
    `BACKLOG_FILING_COOLDOWN_SECONDS` whose bytes match → nothing is POSTed, the skip is
    counted (#4428, the nodes below).

The fourth outcome is the same harm seen one tick later. The three above stop at the row
being OPEN; four incidents (#3227→#3231 at 42m37s, #3551→#3558 at 23m19s, #1346→#1350 at
1h09m, #4239→#4244 at 23m16s) refired the identical alert minutes after the row was
CLOSED, when nothing on the board was open to suppress on.

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

import hashlib
import json
import sys
import threading
from datetime import datetime, timedelta, timezone
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
#: Two firings of the hourly data-root check an hour apart. The measured shadow set is
#: part of the body, so the TITLE is identical and the BYTES are not — which is exactly
#: what #4428's guard has to tell apart: the same finding re-printed, and a finding whose
#: measured paths moved. `#4239` closed at 22:23:51Z and `#4244` filed at 22:47:07Z under
#: one title, so a cooldown keyed on the title alone would have silenced a real change.
SHADOW_A = ("Content-empty databases under the data root. New this hour: "
            "/home/alansrobotlab/lloyd-data/kg.sqlite is 0 bytes")
SHADOW_B = ("Content-empty databases under the data root. New this hour: "
            "/home/alansrobotlab/lloyd-data/research.db is 0 bytes")


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


# ── #4428: a CLOSED row still must not be re-filed with the same bytes ─────────
#
# The three nodes above stop at the row being OPEN, and the corpus shows the harm did not
# stop there: four times a guardian alert filed the identical item again minutes after the
# previous copy CLOSED — #3227→#3231 (42m37s), #3551→#3558 (23m19s), #1346→#1350 (1h09m),
# and the pair that made this item, #4239 (`completed: '2026-11-17T22:23:51.975570'`,
# automod `19717ab3`) → #4244 (`created: '2026-11-17T22:47:07.968284'`, 23m16s later).
# A closed row cannot suppress on its own, and must not: #3757 takes the row out of the
# OPEN set on an all-clear precisely so the next real outage files. So the cooldown is
# keyed on the guardian's OWN record of the filing it made — the exact name, the sha256 of
# the exact `description` bytes it posted, and a `gstate.now_iso()` stamp — and never on a
# close instant, which no route exposes (`_row_from` ships `completed` as a bool;
# `?done_since=` is day-granular) and whose obvious proxy moves on every activity append.


def _filed_record(state_dir: Path) -> dict:
    path = state_dir / notify.BACKLOG_FILING_STATE_NAME
    return json.loads(path.read_text()) if path.exists() else {}


def _note_filing(state_dir: Path, name: str, text: str, *, age_seconds: float) -> None:
    """Write the record `_backlog_task` leaves after a delivered filing, aged by hand.

    The shape is the guardian's own — `filed[<name>]` carrying `hash`, `filed_at` and
    `count`, with `filed_at` a `gstate.now_iso()` UTC second — written here so a node can
    stand a filing a fixed number of seconds in the past instead of waiting for them.

    `hash` is recomputed with `hashlib` rather than with the module's own helper on
    purpose: a node that called that helper would pass whatever it returned, including a
    digest of the wrong input, and every cooldown node below would still be green.
    """
    stamp = (datetime.now(timezone.utc)
             - timedelta(seconds=age_seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {"schema": 1, "board": notify.BACKLOG_ITEM_BOARD,
               "filed": {name: {
                   "hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                   "filed_at": stamp, "count": 1}}}
    path = state_dir / notify.BACKLOG_FILING_STATE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_the_cooldown_constant_is_one_day():
    # Clause 1's named number. Inlined at the call site it stops being a policy a reader
    # can find, and the 24 h figure is a judgement, not a measurement: the four measured
    # close→refiling gaps are 23m16s, 23m19s, 42m37s and 1h09m, while a fifth
    # (#4072→#4091) sits at 25h03m — just outside this window.
    assert notify.BACKLOG_FILING_COOLDOWN_SECONDS == 24 * 3600.0


def test_a_closed_copy_refiles_nothing_when_this_guardian_filed_those_bytes_minutes_ago(
        tmp_path, board_factory):
    """Clause 1: the delivered filing is recorded, and an identical refiring is suppressed.

    The filing happens for real first — an empty board, one POST, the board's own id — and
    only then is the row moved to `done`, which is the state a landed fix or a human close
    leaves it in. Nothing here hand-writes a hash, so the pair
    (bytes this guardian posted, bytes this guardian compared) is the production pair and
    the assertion on the stored digest is about WHAT was digested, not about a helper
    agreeing with itself: it must be sha256 over the exact `description` the POST carried.
    """
    board = board_factory(rows=[])
    n = _notifier(tmp_path, board.url)

    assert n._backlog_task(TITLE, SHADOW_A, "", "") is True
    assert len(board.posts) == 1, "the first firing has to be a real filing"
    posted = board.posts[0]["description"]
    stored = _filed_record(tmp_path)["filed"][NAME]
    assert stored["hash"] == hashlib.sha256(posted.encode("utf-8")).hexdigest(), (
        "the cooldown record is not a digest of the bytes this filing posted, so the "
        "guard below is comparing something else")
    assert stored["filed_at"].endswith("Z"), (
        f"`{stored['filed_at']}` is not the `gstate.now_iso()` UTC shape the age is "
        "computed from; a naive or offset stamp is a second clock in one file")

    board.rows = [{"id": 9001, "name": NAME, "status": "done"}]
    assert n._backlog_task(TITLE, SHADOW_A, "", "") is False

    assert len(board.posts) == 1, "the identical refiling must not reach task-create"
    assert len(board.gets) == 2, "the cooldown is decided AFTER the board is asked"
    rec = _record(tmp_path)
    assert rec["suppressed"][NAME]["count"] == 1
    assert {"first_seen", "last_seen"} <= set(rec["suppressed"][NAME]), (
        "a cooldown suppression has to be countable in the same shape an open-row "
        f"suppression is: {rec['suppressed'][NAME]}")
    assert NAME not in rec.get("unread", {})


def test_a_closed_copy_files_again_when_the_recorded_filing_is_past_the_cooldown(
        tmp_path, board_factory):
    """Clause 2, first half: a filing older than the constant is no reason to stay silent.

    25 h against a 24 h window — the shape #4072→#4091 actually measured (25h03m), which
    is the case the constant is chosen to let through.
    """
    board = board_factory(rows=[{"id": 4091, "name": NAME, "status": "done"}])
    n = _notifier(tmp_path, board.url)
    _note_filing(tmp_path, NAME, SHADOW_A,
                 age_seconds=notify.BACKLOG_FILING_COOLDOWN_SECONDS + 3600)

    assert n._backlog_task(TITLE, SHADOW_A, "", "") is True

    assert len(board.posts) == 1 and board.posts[0]["name"] == NAME
    assert NAME not in _record(tmp_path).get("suppressed", {})


def test_a_closed_copy_files_again_when_this_guardian_never_filed_that_name(
        tmp_path, board_factory):
    """Clause 2, second half — and the half the existing node cannot carry.

    `test_a_board_with_only_a_closed_copy_files_and_records_nothing` files because its
    state dir is empty. Here the filing record EXISTS and is fresh and inside the window,
    and only its key is another alert's name: an item some other family closed, or a row a
    human closed by hand, is not a filing this guardian owes silence to.
    """
    board = board_factory(rows=[{"id": 3558, "name": NAME, "status": "done"}])
    n = _notifier(tmp_path, board.url)
    _note_filing(tmp_path, SERVICE_DOWN_NAME, BODY, age_seconds=60)

    assert n._backlog_task(TITLE, SHADOW_A, "", "") is True

    assert len(board.posts) == 1 and board.posts[0]["name"] == NAME
    assert NAME not in _record(tmp_path).get("suppressed", {})


def test_a_closed_copy_files_at_once_when_the_measured_paths_moved(
        tmp_path, board_factory):
    """Clause 3: the same title with different bytes is a different finding, so it files.

    Both halves of one hour-to-hour pair in a single node, because the case is the
    difference between them: `#4239` and `#4244` share a title and differ by the shadow
    paths each measured, and a cooldown that keyed on the title alone would have silenced
    the second. First a hand-written record for `SHADOW_A` against a firing of `SHADOW_B`;
    then the same thing end to end, filing `SHADOW_A`, closing it, and refiring `SHADOW_B`.
    """
    board = board_factory(rows=[{"id": 4244, "name": NAME, "status": "done"}])
    n = _notifier(tmp_path, board.url)
    _note_filing(tmp_path, NAME, SHADOW_A, age_seconds=3600)

    assert n._backlog_task(TITLE, SHADOW_B, "", "") is True
    assert len(board.posts) == 1

    board.rows = []
    assert n._backlog_task(TITLE, SHADOW_A, "", "") is True
    board.rows = [{"id": 4245, "name": NAME, "status": "done"}]
    assert n._backlog_task(TITLE, SHADOW_B, "", "") is True

    assert len(board.posts) == 3, (
        "a changed body inside the cooldown is a new finding, and suppressing it is the "
        "one failure this guard must not produce")
    assert NAME not in _record(tmp_path).get("suppressed", {})


def test_a_missing_or_unreadable_filing_record_never_suppresses(tmp_path, board_factory):
    """Clause 4's state half: the cooldown can only ever be decided from bytes it can read.

    Four states of one file, each against the same closed row and a fresh record shape: no
    file, bytes that are not JSON, JSON of the wrong shape, and a well-shaped record whose
    stamp cannot be parsed. Every one of them means `file`, for the reason the open-row
    guard already gives: the worst outcome of this guard is a duplicate, and the one it
    must never produce is silence. A state dir per state, because a record written by one
    round would answer the next.
    """
    states = ("absent", "not json at all", "[]", "an unparseable stamp")
    for i, state in enumerate(states):
        state_dir = tmp_path / f"s{i}"
        board = board_factory(rows=[{"id": 3231, "name": NAME, "status": "done"}])
        n = _notifier(state_dir, board.url)
        path = state_dir / notify.BACKLOG_FILING_STATE_NAME
        if state == "absent":
            pass
        elif state == "not json at all":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("not json at all")
        elif state == "[]":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("[]")
        else:
            _note_filing(state_dir, NAME, SHADOW_A, age_seconds=60)
            data = json.loads(path.read_text())
            data["filed"][NAME]["filed_at"] = "the day before yesterday"
            path.write_text(json.dumps(data))

        assert n._backlog_task(TITLE, SHADOW_A, "", "") is True, (
            f"a filing record in state {state!r} silenced a filing")
        assert len(board.posts) == 1, state


def test_an_unimportable_status_vocabulary_still_files(tmp_path, board_factory,
                                                       monkeypatch):
    """Clause 4's vocabulary half, with the cooldown armed rather than idle.

    `sys.modules["voiceloss"] = None` is the interpreter's own way to make
    `import voiceloss` raise. The row is `done` AND a matching filing is recorded, so this
    node fails if either suppression input is consulted before the vocabulary is loaded:
    a guard that decided the cooldown first would suppress here and file nothing.
    """
    board = board_factory(rows=[{"id": 1350, "name": NAME, "status": "done"}])
    n = _notifier(tmp_path, board.url)
    _note_filing(tmp_path, NAME, SHADOW_A, age_seconds=60)
    monkeypatch.setitem(sys.modules, "voiceloss", None)

    assert n._backlog_task(TITLE, SHADOW_A, "", "") is True

    assert len(board.posts) == 1
    assert NAME not in _record(tmp_path).get("suppressed", {})


def test_a_cooldown_suppression_changes_the_backlog_channel_only(tmp_path, board_factory):
    """Clause 5: nothing on this path may widen into silence.

    Two state dirs, one closed row on each board, and ONE difference between them: the
    first has the filing record, the second does not. So every channel value that differs
    is a channel this guard touched, and the assert below names the only one allowed to.
    The alert goes through `alert()` rather than `_backlog_task` because the clause is
    about the fan-out: the ledger row, ALERT.md, the journal, the toast, the voice and the
    daily note all still have to happen for an alert whose item was suppressed.
    """
    filed_text = f"{TITLE}\n\n{SHADOW_A}"
    res = {}
    dirs = {}
    for key, seed in (("suppressed", True), ("filed", False)):
        state_dir = tmp_path / key
        (state_dir / "vault" / "memory").mkdir(parents=True)
        board = board_factory(rows=[{"id": 4244, "name": NAME, "status": "done"}])
        n = _resolving_notifier(state_dir, board.url)
        if seed:
            _note_filing(state_dir, NAME, filed_text, age_seconds=60)
        res[key] = n.alert("error", TITLE, SHADOW_A, needs_human=True)
        dirs[key] = (board, state_dir)

    assert res["suppressed"]["backlog"] is False
    assert res["filed"]["backlog"] is True
    assert {k: v for k, v in res["suppressed"].items() if k != "backlog"} == {
        k: v for k, v in res["filed"].items() if k != "backlog"}, (
        "a cooldown suppression changed a channel other than the backlog filing, and "
        "that is #775's silence arriving through the guard instead of around it")

    suppressed_board, suppressed_dir = dirs["suppressed"]
    filed_board, _ = dirs["filed"]
    assert [p for p, b in zip(suppressed_board.post_paths, suppressed_board.posts)
            if p == CREATE_PATH] == [], "the suppressed alert still filed its item"
    assert len([p for p, b in zip(filed_board.post_paths, filed_board.posts)
                if p == CREATE_PATH]) == 1

    assert TITLE in (suppressed_dir / notify.ALERT_FILE_NAME).read_text()
    assert any('"event": "alert"' in line or '"event":"alert"' in line
               for line in (suppressed_dir / "ledger.jsonl").read_text().splitlines()), (
        "the suppressed alert left no ledger row: the audit trail lost a firing")
    note = (suppressed_dir / "vault" / "memory"
            / f"{datetime.now().strftime('%Y-%m-%d')}.md")
    assert f"{notify.DAILY_SECTION_PREFIX}{TITLE}" in note.read_text(), (
        "the daily note — the surface a human actually reads — lost the suppressed alert")
    assert NAME in _record(suppressed_dir)["suppressed"], (
        "the suppression itself has to be on disk, in backlog-skipped.json")


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
