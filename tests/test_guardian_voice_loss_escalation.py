"""The escalator from `voice-loss.md` to one coalesced backlog item (#1904).

`#1806` wrote the record and nothing read it, so a dead spoken alert left an
artefact in `~/.local/state` where an alarm should have been. This file pins the
reader: a guardian tick that files or refreshes exactly one
`[alerts] voice loss — spoken alerts did not reach the speakers` item on board
`lloyd`, and stops when a person has closed the incident.

Three choices about HOW these are tested, because each is a boundary the change
crosses and a weaker harness would not cross it:

  * **The board is a real HTTP server, not a patched function.** The escalator
    lives in the guardian — its own systemd unit, its own interpreter, a staged
    flat copy of `agent-services/guardian/*.py` — and reaches the board only over
    loopback. So every test below runs a `ThreadingHTTPServer` on a bound port and
    the escalator opens a real socket: the route, the JSON body, the query string
    and a genuine `ECONNREFUSED` are all exercised, and the assertions are on the
    bytes the route was handed. Clause 5 uses a port bound and closed, the same
    trick `test_guardian_speak.py::_refused_port` uses.
  * **The record is written by the writer.** `_record_loss` is called, not
    imitated, so the shape under test is the one production writes — and the
    clock it reads is a stand-in on the `speak` module only, which is what lets
    two bursts land sixty seconds apart without the test sleeping.
  * **The status vocabulary is compared against the real one.** `voiceloss.py`
    re-implements `canonical_status`/`OPEN_STATUSES` because the staged guardian
    cannot import `app/`. A copy that drifts is the whole risk of that
    re-implementation, so one test asserts the two agree over every spelling.

The names carry the clause number they exist for, and the module-level
`_CLAUSES` line is the map the round report cites.
"""

from __future__ import annotations

import datetime
import http.server
import json
import re
import socket
import sys
import threading
import time
import types
import urllib.parse

from pathlib import Path

import pytest

GUARDIAN_DIR = Path(__file__).resolve().parent.parent / "agent-services" / "guardian"
sys.path.insert(0, str(GUARDIAN_DIR))

import gstate  # noqa: E402
import speak  # noqa: E402
import voiceloss  # noqa: E402

from app import backlog_status  # noqa: E402

# Route names spelled as literals, not as `voiceloss._CREATE_PATH`. The clauses
# name the routes; a test that read them back off the module under test would
# follow a rename and still "pass" while posting somewhere the board is not.
CREATE = "/api/backlog/task-create"
UPDATE = "/api/backlog/task-update"
TASKS = "/api/backlog/tasks"

# Clause map: 1 → test_a_first_burst_files_one_draft_item_on_board_lloyd,
# 2 → test_a_later_burst_refreshes_the_one_open_item_and_files_no_second_one,
# 3 → test_a_closed_voice_loss_item_is_neither_refreshed_nor_re_opened,
# 4 → test_an_unchanged_record_files_nothing_on_a_later_tick,
# 5 → test_a_refused_backend_costs_nothing_and_the_child_is_untouched.
# The acceptance check's stamp half (%z, the item's own clause 5) is pinned in
# tests/test_guardian_speak.py::test_the_loss_record_stamps_its_clock_with_an_offset.

_OCCURRENCES = re.compile(r"^Occurrences:[ \t]*(\d+)", re.M)
_LAST_SEEN = re.compile(r"^Last seen:[ \t]*(\S+)", re.M)
_STAMP = "%Y-%m-%dT%H:%M:%S%z"

_LOST = "Guardian alert. supervisord was unreachable. Restarted agent"
_LOST_2 = "Guardian alert. Landed: #1750 promotion settled"
_LOST_3 = "Guardian alert. Worker pool is not running"


# ── harness ───────────────────────────────────────────────────────────
class _Clock:
    """A stand-in for the `time` module with a settable now.

    Rebound on the `speak` module ONLY, so the writer's stamps and its
    in-window test move on command while the escalator, the server and pytest
    keep the real clock. `strftime` accepts the `struct_time` `_record_loss`
    hands it — that call is `time.strftime(fmt, time.localtime(now))`, so the
    outer call must not try to convert a struct again.
    """

    def __init__(self, now: float):
        self.now = float(now)

    def time(self) -> float:
        return self.now

    def localtime(self, t=None):
        return time.localtime(self.now if t is None else t)

    def strftime(self, fmt, t=None):
        return time.strftime(fmt, t if t is not None else self.localtime())


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self._serve(None)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        self._serve(json.loads(raw.decode("utf-8", "replace") or "{}"))

    def _serve(self, payload):
        reply = self.server.board.answer(self.command, self.path, payload)
        body = json.dumps(reply).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence the per-request stderr line
        pass


class _Backend:
    """The board's own three routes, answered from a script, over a real socket.

    Rows are filtered the way `app/routers/backlog.py::backlog_tasks` filters
    them — `?q=` over the name, `board_id` down to one board — so an escalator
    that searched with the wrong needle or the wrong board would see no rows here
    and file a duplicate, which is the failure this stub exists to be able to
    show.
    """

    def __init__(self):
        self.rows: list[dict] = []
        self.requests: list[dict] = []
        self.next_id = 900
        self.update_success = True
        self._srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._srv.board = self
        self.port = self._srv.server_address[1]
        self._thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self._srv.shutdown()
        self._srv.server_close()
        self._thread.join(timeout=5)

    def calls(self, path: str) -> list[dict]:
        return [r for r in self.requests if r["path"] == path]

    def answer(self, method: str, raw_path: str, payload):
        path, _, query = raw_path.partition("?")
        qs = urllib.parse.parse_qs(query)
        self.requests.append({"method": method, "path": path, "query": qs,
                              "payload": payload})
        if path == TASKS:
            needle = (qs.get("q") or [""])[0].lower()
            board = (qs.get("board_id") or [""])[0]
            return [r for r in self.rows
                    if needle in str(r.get("name") or "").lower()
                    and board == str(r.get("board") or "lloyd")]
        if path == CREATE:
            filed = {"success": True, "id": self.next_id}
            self.next_id += 1
            return filed
        if path == UPDATE:
            return {"success": self.update_success, "description_ignored": False}
        return {"success": False, "detail": f"unexpected path {path}"}


@pytest.fixture
def board():
    srv = _Backend()
    try:
        yield srv
    finally:
        srv.stop()


def _refused_port() -> int:
    """A port bound and immediately closed, so a connect gets ECONNREFUSED."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _clock(monkeypatch, now: float) -> _Clock:
    clk = _Clock(now)
    monkeypatch.setattr(speak, "time", clk)
    return clk


def _burst(state_dir: Path, *texts: str) -> None:
    """Append lost utterances with the real writer, one `speak_now` failure each.

    Called rather than hand-rolled: the format under test is then the one
    production writes, including the `%z` stamps the escalator parses.
    """
    for t in texts:
        speak._record_loss(state_dir, t,
                           "ConnectionRefusedError: [Errno 111] Connection refused")


def _escalator(board, state_dir: Path) -> voiceloss.VoiceLossEscalator:
    return voiceloss.VoiceLossEscalator(state_dir, base_url=board.base_url)


def _body(request: dict) -> str:
    return str((request["payload"] or {}).get("description") or "")


def _count(body: str) -> int:
    m = _OCCURRENCES.search(body)
    assert m, f"no `Occurrences: N` line in the posted body:\n{body}"
    return int(m.group(1))


def _last_seen(body: str) -> datetime.datetime:
    m = _LAST_SEEN.search(body)
    assert m, f"no parseable `Last seen:` line in the posted body:\n{body}"
    # Aware, because that is the whole point of #1904's stamp half: a naive stamp
    # here would raise ValueError on this very line.
    return datetime.datetime.strptime(m.group(1), _STAMP)


# ── clause 1 ─────────────────────────────────────────────────────────
def test_a_first_burst_files_one_draft_item_on_board_lloyd(tmp_path, monkeypatch,
                                                           board):
    """Clause 1: no cursor and a record present ⇒ exactly one create, and the
    payload carries the board ruling — `lloyd`, `draft`, `high`, this name.

    `notify.py::_backlog_task` is the route the guardian already has, and it could
    not carry this payload: it hardcodes `up_next`, posts no board, and prefixes
    `[guardian] `. So the four fields below are the point of the new poster, and
    `draft` in particular is what #1893 measured — an alarm filed at any other
    status lands where no pool polls it.
    """
    clk = _clock(monkeypatch, time.time())
    _burst(tmp_path, _LOST, _LOST_2)

    report = _escalator(board, tmp_path).tick(now=clk.now)

    creates = board.calls(CREATE)
    assert report["reason"] == "created", report
    assert len(creates) == 1, f"expected one create, got {len(creates)}"
    assert board.calls(UPDATE) == [], "a first burst files; it has nothing to refresh"
    payload = creates[0]["payload"]
    assert str(payload["name"]).startswith("[alerts] voice loss"), payload
    assert payload["board"] == "lloyd", payload
    assert payload["status"] == "draft", payload
    assert payload["priority"] == "high", payload
    # The search that decided to create rather than refresh asked the same board,
    # with our own needle — otherwise "no open item" is an artefact of a bad query.
    reads = board.calls(TASKS)
    assert reads, "the decision to file must come from the board's list route"
    assert (reads[0]["query"].get("board_id") or [""])[0] == "lloyd", reads[0]
    assert (reads[0]["query"].get("q") or [""])[0] == voiceloss.ITEM_PREFIX, reads[0]
    assert _count(_body(creates[0])) == 2, "the record said two, the board must say two"
    assert _LOST in _body(creates[0]) and _LOST_2 in _body(creates[0]), (
        "the body must name what was lost, which is all the record was ever for")


# ── clause 2 ─────────────────────────────────────────────────────────
def test_a_later_burst_refreshes_the_one_open_item_and_files_no_second_one(
        tmp_path, monkeypatch, board):
    """Clause 2: occurrences advanced and an open row exists ⇒ one update to THAT
    id, `force_body_replace` set, and the count and last-seen both moved.

    The flag is not decoration: `app/routers/backlog.py:614` sets
    `description_ignored` and drops a body shorter than the one on disk, and the
    second burst's quoted utterance list IS allowed to be shorter — so without it
    the route answers 200 while the tally on the board stays where it was, which
    is a silent failure of exactly this clause.
    """
    clk = _clock(monkeypatch, time.time())
    _burst(tmp_path, _LOST, _LOST_2)
    esc = _escalator(board, tmp_path)
    assert esc.tick(now=clk.now)["reason"] == "created"
    created = board.calls(CREATE)[0]["payload"]
    filed_id = board.next_id - 1
    board.rows = [{"id": filed_id, "name": created["name"], "status": "draft",
                   "board": "lloyd"}]

    clk.now += 600.0
    _burst(tmp_path, _LOST_3)
    assert esc.tick(now=clk.now)["reason"] == "refreshed"

    updates = board.calls(UPDATE)
    assert len(updates) == 1, f"expected one update, got {len(updates)}"
    assert updates[0]["payload"]["id"] == filed_id, updates[0]["payload"]
    assert updates[0]["payload"].get("force_body_replace") is True, (
        "without the flag the route drops a shorter body and the count goes stale")
    assert len(board.calls(CREATE)) == 1, "a second burst must not file a second item"

    before, after = _body(board.calls(CREATE)[0]), _body(updates[0])
    assert _count(after) == 3 > _count(before) == 2, (before, after)
    assert _last_seen(after) > _last_seen(before), (
        "the refresh must move last-seen, not merely re-post the old body")


# ── clause 3 ─────────────────────────────────────────────────────────
@pytest.mark.parametrize("status", ["done", "closed", "cancelled", "wontfix"])
def test_a_closed_voice_loss_item_is_neither_refreshed_nor_re_opened(
        tmp_path, monkeypatch, board, status):
    """Clause 3: a row outside the open set is left alone, and a later burst does
    not answer it with a fresh item either.

    The retired spellings are in the parametrised set on purpose: a row reading
    `status: closed` is exactly what a person who has finished with an incident
    leaves behind, and a reader that only knew the four current words would
    refresh it back to life. The second burst is the half that catches an
    implementation which files a NEW item when the old one is closed — that
    overrules whoever closed it, and the record on disk is still the evidence.
    """
    clk = _clock(monkeypatch, time.time())
    _burst(tmp_path, _LOST)
    board.rows = [{"id": 777, "name": voiceloss.ITEM_NAME, "status": status,
                   "board": "lloyd"}]

    esc = _escalator(board, tmp_path)
    first = esc.tick(now=clk.now)

    assert first["reason"] == "closed-suppressed", first
    assert board.calls(CREATE) == [], "a closed incident must not be re-filed"
    assert board.calls(UPDATE) == [], "a closed incident must not be refreshed"

    clk.now += 600.0
    _burst(tmp_path, _LOST_2)
    second = esc.tick(now=clk.now)

    assert second["reason"] == "closed-suppressed", second
    assert board.calls(CREATE) == [], "a later burst must not overrule the close"
    assert board.calls(UPDATE) == [], "a later burst must not re-open the row"
    assert gstate.read_json(esc.cursor_path)["occurrences"] == 2, (
        "the cursor still advances, so an incident a human closed is not read "
        "again on every tick")


# ── clause 4 ─────────────────────────────────────────────────────────
def test_an_unchanged_record_files_nothing_on_a_later_tick(tmp_path, monkeypatch,
                                                           board):
    """Clause 4: the cursor's last-escalated occurrences silence a repeat tick.

    Asserted twice over, because the cursor has two producers: the tick that just
    filed, and a file written by an earlier process or restored from disk. The
    second one is the case a guardian restart is made of — a watch that forgets
    what it filed files the same incident again.
    """
    clk = _clock(monkeypatch, time.time())
    _burst(tmp_path, _LOST)
    esc = _escalator(board, tmp_path)

    assert esc.tick(now=clk.now)["reason"] == "created"
    requests_after_first = len(board.requests)
    board.rows = [{"id": board.next_id - 1, "name": voiceloss.ITEM_NAME,
                   "status": "draft", "board": "lloyd"}]

    again = esc.tick(now=clk.now + 5.0)

    assert again["reason"] == "unchanged", again
    assert len(board.requests) == requests_after_first, (
        "an unchanged record must not even be looked up, let alone posted to")

    gstate.write_json_atomic(esc.cursor_path, {"schema": 1, "occurrences": 1})
    esc2 = _escalator(board, tmp_path)
    third = esc2.tick(now=clk.now + 10.0)

    assert third["reason"] == "unchanged", third
    assert len(board.requests) == requests_after_first, (
        "a cursor written by anything else must hold just as firmly")
    assert esc.cursor_path.name == "voice_loss_cursor.json"
    assert (tmp_path / "voice_loss_cursor.json").is_file(), (
        "the watermark is on disk in the guardian state dir, not in memory")


# ── clause 5 ─────────────────────────────────────────────────────────
def test_a_refused_backend_costs_nothing_and_the_child_is_untouched(
        tmp_path, monkeypatch):
    """Clause 5, the escalation half: the backend refuses the connection ⇒ nothing
    raises, `voice-loss.md` is byte-identical, and the escalation stays owed.

    A refused backend is not an edge case here, it is the expected companion of a
    dead speaker: `[program:lloyd-backend]` shares `agent-supervisord` with
    `[program:agent-tts]`, which is why the record is written with no HTTP in it
    at all. So the record must survive the escalator's failure exactly as it
    survived the failure it records — and the cursor must NOT count what was never
    delivered, or the alarm is silently spent.
    """
    clk = _clock(monkeypatch, time.time())
    _burst(tmp_path, _LOST)
    record = tmp_path / speak.LOSS_NAME
    before = record.read_bytes()
    esc = voiceloss.VoiceLossEscalator(
        tmp_path, base_url=f"http://127.0.0.1:{_refused_port()}")

    report = esc.tick(now=clk.now)          # must not raise

    assert report["reason"] == "unreachable", report
    assert record.read_bytes() == before, "the record is the evidence; it stays put"
    cursor = gstate.read_json(esc.cursor_path)
    assert cursor["occurrences"] == 0, f"an undelivered escalation is still owed: {cursor}"
    assert cursor["next_attempt_ts"] > clk.now, (
        "and it retries, but not every 5 s tick at a route that is not answering")


def test_the_child_that_lost_the_words_makes_no_backend_call(tmp_path, monkeypatch):
    """Clause 5, the other half: `speak_now`'s failure path gains no HTTP call.

    The item's ruling is that the escalation runs from the guardian's loop and not
    from the failing detached child, and the half that has to stay true is that the
    child still records the loss and returns False while asking nothing of
    `/api/`. This is a regression rail on `speak.py`: if a later round moves the
    post into `speak_now` — the obvious-looking place, since that is where the
    failure is — the child starts POSTing to the process that shares the supervisor
    which just died, and this node goes red instead of that becoming an outage
    story again.
    """
    attempts: list[str] = []
    real_urlopen = speak.urllib.request.urlopen

    def probe(req, timeout=None, *args, **kw):
        attempts.append(req.full_url)
        return real_urlopen(req, timeout=timeout, *args, **kw)

    class _NoPlayer:
        returncode = 0

    monkeypatch.setattr(speak.urllib.request, "urlopen", probe)
    monkeypatch.setattr(speak.subprocess, "run", lambda cmd, **kw: _NoPlayer())
    monkeypatch.setattr(speak.shutil, "which", lambda exe: None)
    cfg = dict(speak.DEFAULTS,
               api_url=f"http://127.0.0.1:{_refused_port()}")

    assert speak.speak_now(_LOST, cfg, tmp_path) is False

    assert attempts == [f"{cfg['api_url']}/v1/audio/speech"], attempts
    assert not [u for u in attempts if "/api/" in u], (
        "the child must not reach the backlog route: it dies with the sound")
    body = (tmp_path / speak.LOSS_NAME).read_text(encoding="utf-8")
    assert "occurrences: 1" in body and _LOST in body, body


# ── the seams the change crosses, pinned once ─────────────────────────
def test_the_guardian_and_the_backend_copy_of_the_status_vocabulary_agree():
    """`voiceloss.py` may not import `app.backlog_status`, so it copies it.

    The copy is the risk: a row the guardian thinks is open and the board thinks
    is closed is a closed incident being refreshed, and nothing else in either
    file would say so. Pinned over every spelling that has ever appeared on this
    board, plus the shapes that are not words at all — the same deal
    `gstate.py:73-79` strikes for the halt event names and
    `tests/test_guardian_rollback.py` keeps.
    """
    values = ["draft", "up_next", "in_progress", "done", "closed", "cancelled",
              "wontfix", "Done", " DRAFT ", "review", "archived", "blocked",
              "", " ", None, 7, True]
    assert voiceloss.PIPELINE_STATUSES == backlog_status.PIPELINE_STATUSES
    assert voiceloss.CLOSED_ALIASES == backlog_status.CLOSED_ALIASES
    assert voiceloss.OPEN_STATUSES == backlog_status.OPEN_STATUSES
    for v in values:
        got = voiceloss.canonical_status(v)
        want = backlog_status.canonical_status(v)
        assert got == want, f"{v!r}: guardian says {got!r}, the board says {want!r}"
        assert (got in voiceloss.OPEN_STATUSES) == (want in backlog_status.OPEN_STATUSES), (
            f"{v!r} is open to one reader and closed to the other")


def test_no_guardian_module_reaches_into_the_app_package():
    """The staged guardian is a flat copy, so `app.*` is an ImportError at runtime.

    `guardian-stage.sh:42` is `cp "$SRC"/*.py "$STAGE"/` and the unit runs that
    snapshot with `/usr/bin/python3` — `app/` is not there, and
    `selftest.py --profile staging` does not import every module, so the failure
    would arrive on the first tick in production and stop the watchdog. A
    behavioural test cannot see this boundary at all: it is a property of how the
    directory is shipped, which is why it is checked as text over every file that
    will be copied.
    """
    offenders = []
    for f in sorted(GUARDIAN_DIR.glob("*.py")):
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if re.match(r"\s*(from|import)\s+app(\.|\s)", line):
                offenders.append(f"{f.name}:{n}: {line.strip()}")
    assert offenders == [], "the staged guardian cannot import these:\n" + "\n".join(offenders)


def test_the_guardian_tick_runs_the_escalator_before_it_gives_up(tmp_path,
                                                                 monkeypatch):
    """The item's ruling — "drive it from the guardian loop" — is a placement, so
    it is pinned as one: `tick()` must reach the escalator even when supervisord is
    unreachable, the state a burst during an outage leaves the stack in.

    `voiceloss` is replaced by a spy, so this node is about wiring and ordering
    only; the behaviour is the five clauses above. Everything else `tick()` touches
    is stubbed for the same reason `test_guardian_predicates.py` stubs it: the
    watches act on this machine, and a test of a hook is not entitled to restart
    anything.
    """
    import guardian as G

    seen: list[bool] = []

    class _Spy:
        def tick(self, now=None):
            seen.append(True)
            return {"reason": "no-record"}

    args = types.SimpleNamespace(
        repo=str(tmp_path), state=str(tmp_path / "state"),
        guardian_state=str(tmp_path / "gdir"), supervisor_sock="/nonexistent",
        backend_url="http://127.0.0.1:1/health", mcp_url="http://127.0.0.1:2/health",
        programs="lloyd-mc:lloyd-backend", interval=5.0,
    )
    g = G.Guardian(args)
    g.voiceloss = _Spy()
    monkeypatch.setattr(g, "collect", lambda: {"now": time.time(),
                                               "supervisord": "unreachable",
                                               "supervisord_error": "ECONNREFUSED",
                                               "procs": {}, "probes": {}})
    for name in ("drain_logs", "check_vault", "check_data", "check_memory",
                 "check_tmp"):
        monkeypatch.setattr(g, name, lambda: None)
    monkeypatch.setattr(g, "check_pool", lambda snap: None)

    state = g.tick()

    assert seen == [True], "tick() never reached the escalator"
    assert state == "infra_down", (
        "the tick still returns infra_down — the escalation is added to the loop, "
        "it does not change what the loop decides")


# ── the record's own contract with its new reader ─────────────────────
def test_the_escalator_reads_the_record_the_writer_produced(tmp_path, monkeypatch):
    """The reader and the writer are one file's two ends, so one node reads what
    the other wrote: counts, the two stamps with their offsets, and the named
    utterances.

    This is the boundary #1904 opened — until now nothing on the read side had
    ever parsed `voice-loss.md` — and it is why the stamps gained an explicit
    offset: a file with a reader is a machine-facing payload, and a naive local
    stamp read as UTC is the class that made the 2026-09-28 outage look like it
    began 59 minutes early.
    """
    _clock(monkeypatch, time.time())
    _burst(tmp_path, _LOST, _LOST_2)

    record = speak.read_loss_record(tmp_path)

    assert record["occurrences"] == 2, record
    assert _LOST in record["said"] and _LOST_2 in record["said"], record
    for key in ("first_seen", "last_seen"):
        stamp = record[key]
        assert stamp, f"{key} missing from the record: {record}"
        assert datetime.datetime.strptime(stamp, _STAMP).utcoffset() is not None, (
            f"{key}={stamp!r} carries no offset, so a reader has to guess a zone")


def test_a_record_with_no_count_is_not_a_record(tmp_path):
    """`read_loss_record` answers None for anything without `occurrences`, so a
    half-written file cannot file an alarm with an invented count — and an absent
    file answers the same way, which is the every-tick case.
    """
    assert speak.read_loss_record(tmp_path) is None, "no file is not a record"
    (tmp_path / speak.LOSS_NAME).write_text(
        "# Voice alert lost\n\nlast_seen: 2026-09-30T12:00:00-0700\n",
        encoding="utf-8")
    assert speak.read_loss_record(tmp_path) is None, (
        "a file with no count is not a record either: the alternative is posting "
        "a number nobody observed")

    (tmp_path / speak.LOSS_NAME).write_text(
        f"{speak.LOSS_HEADING}\n\noccurrences: 4\nburst_started: 1761800000.000\n",
        encoding="utf-8")
    assert speak.read_loss_record(tmp_path)["occurrences"] == 4, (
        "a record with no stamps is still a count, and the body says (unstamped)")
