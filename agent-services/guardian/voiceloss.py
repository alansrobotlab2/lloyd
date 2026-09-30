"""Escalate the voice-loss record into one coalesced backlog item (#1904).

`#1806` gave a lost spoken alert a durable record — `speak._record_loss`
coalescing a burst into one `voice-loss.md` — and nothing read it. Two days after
it landed the live file held `occurrences: 2` naming two guardian alerts that never
reached the speakers, and `grep -ril "voice.loss" ~/obsidian/backlog` returned the
writer's own item and nothing else. A record in `~/.local/state` is an artefact;
the class #1806 was opened for is an alarm nobody sees. This is the other half.

The board ruling was already made and shipped, so it is used and not re-litigated:
`app/autonomy.py::_file_daily_note_mismatch` files #1798's dropped-daily-note alarm
as one coalesced item on board `lloyd` at `status: "draft"`, `priority: "high"`
(`_DAILY_NOTE_DROP_BOARD`, and the payload at `:3300-3310`). Same shape here, with
the title this item names.

**Why the guardian's loop hosts it, not the process that failed.** The failing
child is `speak.speak_now`, running detached from a `dispatch` whose whole premise
is that the TTS server is unreachable. `[program:lloyd-backend]` — the process
behind `/api/backlog/task-create` — shares `agent-supervisord` with
`[program:agent-tts]`, so a post from inside that child dies with the sound: the
2026-09-28 11:50Z refused utterance was the alert announcing that very supervisord
was unreachable. The guardian is its own systemd unit (`lloyd-guardian`), so a
record written during an agent-supervisord outage is escalated on the first tick
after the backend answers instead of never. That placement also means the failure
path in `speak.py` gains no HTTP call and stays inside its 2.0 s watchdog contract
(`tests/test_guardian_speak.py::test_the_failure_path_returns_within_two_seconds_and_never_raises`).

**What this module may not do.**

* *Import `app.*`.* `guardian-stage.sh:42` stages this directory as a FLAT COPY
  (`cp "$SRC"/*.py "$STAGE"/`) and the unit runs that snapshot with
  `/usr/bin/python3`, so an `app.backlog_status` import is an `ImportError` in
  production at the first tick and `selftest.py --profile staging` never catches
  it. The open/closed line therefore needs the board's status vocabulary, and
  `canonical_status`/`OPEN_STATUSES` are re-implemented from
  `app/backlog_status.py` below. `tests/test_guardian_voice_loss_escalation.py::
  test_the_guardian_and_the_backend_copy_of_the_status_vocabulary_agree` pins the
  two copies equal over the whole vocabulary — the same deal `gstate.py:73-79`
  strikes for the halt event names.
* *Raise.* Every branch returns a report; the caller logs it. A tick that dies
  here costs the watchdog its other watches.
* *Re-open a closed incident.* `canonical_status` says whether a row is live
  before anything is posted to it, and a row outside `OPEN_STATUSES` is neither
  refreshed nor replaced. A person who closed `[alerts] voice loss` closed that
  incident; the next burst advances the cursor and files nothing, and the record
  on disk stays the evidence.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

from pathlib import Path

import gstate
import speak

#: Where the watermark lives: last-escalated occurrences, in the guardian's own
#: state dir, one small JSON per cursor — the shape `gstate.py:29` `read_json` and
#: `:36` `write_json_atomic` already exist for, and the shape `last_settled.json`
#: and `eval_last.json` are. On disk rather than in a `Guardian` attribute because
#: the escalation must survive a guardian restart: a process that forgets what it
#: already filed files it again.
CURSOR_NAME = "voice_loss_cursor.json"

#: The title, and the prefix that identifies a row as THIS incident. One constant
#: so the create, the search and the reader of the board all spell it the same.
ITEM_PREFIX = "[alerts] voice loss"
ITEM_NAME = f"{ITEM_PREFIX} — spoken alerts did not reach the speakers"

#: Settled on #1798 and shipped by `app/autonomy.py:3119`/`:3303`/`:3308`. Posting
#: no board would land on `DEFAULT_BOARD`, which is `lloyd` today — a coincidence
#: that stopped being true the day someone added a board, and
#: `app/routers/backlog.py:724-730` refuses an unknown one, so the field is live
#: and worth pinning rather than inherited.
ITEM_BOARD = "lloyd"
ITEM_STATUS = "draft"
ITEM_PRIORITY = "high"

_CREATE_PATH = "/api/backlog/task-create"
_UPDATE_PATH = "/api/backlog/task-update"
_LIST_PATH = "/api/backlog/tasks"

#: Loopback bound. `notify.py:562` posts the same route at 5.0 s and this is the
#: same process — the guardian ticks nothing of its own between probes, so it can
#: afford what a worker's event loop cannot (`app/autonomy.py:3120-3129` explains
#: why that caller chose 3.0).
_TIMEOUT_SECONDS = 5.0

#: How long to wait after a failed post before trying again, and the reason it
#: exists: `tick()` runs every `policy.TICK_SECONDS` (5 s), so a route that keeps
#: answering 400 — a rejected board, a name the validator dislikes — would otherwise
#: be POSTed to 12 times a minute forever. 900 s is the same span
#: `policy.ALERT_REPEAT_SECONDS` gives a repeating alert, and at 5 s ticks it costs
#: four attempts an hour against a backend that is down, which is a retry, not a
#: flood. The cursor is NOT advanced on that path, so the escalation is still owed
#: and the next attempt is a real one.
RETRY_SECONDS = 900.0

# ── the board's status vocabulary, re-implemented ─────────────────────
# Copied from `app/backlog_status.py`, which this file cannot import (module
# docstring). The lists and the lopsided mapping are its argument, and they are
# reproduced rather than paraphrased: only words already known to mean "off the
# board" are terminal, and everything else becomes `draft`, because sending a live
# item to `done` is the irreversible direction.
PIPELINE_STATUSES: tuple[str, ...] = ("draft", "up_next", "in_progress", "done")
CLOSED_ALIASES: frozenset[str] = frozenset({"closed", "cancelled", "wontfix"})
OPEN_STATUSES: frozenset[str] = frozenset(PIPELINE_STATUSES) - {"done"}


def canonical_status(value) -> str:
    """Map any frontmatter `status` onto one of `PIPELINE_STATUSES`."""
    s = str(value or "").strip().lower()
    if not s:
        return "draft"
    if s in PIPELINE_STATUSES:
        return s
    if s in CLOSED_ALIASES:
        return "done"
    return "draft"


def item_body(record: dict, *, state_dir: Path) -> str:
    """The whole body for one coalesced item, written from the record.

    The count and the two stamps are on their own lines, `Occurrences: <n>`
    first, because that is the number clause 2 requires to move and a prose
    sentence around it is a thing a later reader cannot parse. The body is posted
    whole with `force_body_replace`, which is what makes "written from the record"
    load-bearing: nothing here is merged with what is on the board, so the board
    can never keep a stale tally that this function did not write.
    """
    said = "".join(f'- "{s}"\n'
                   for s in list(record.get("said") or [])[:speak.LOSS_TEXT_KEEP])
    return (
        "Spoken alerts were dispatched to the voice channel and never reached the "
        f"speakers. `{speak.LOSS_NAME}` in the guardian state dir is the record; "
        "this item is the alarm that the record could not be.\n"
        "\n"
        f"Occurrences: {record['occurrences']}\n"
        f"First seen: {record.get('first_seen') or '(unstamped)'}\n"
        f"Last seen: {record.get('last_seen') or '(unstamped)'}\n"
        f"Burst window: {speak.LOSS_WINDOW:.1f}s\n"
        f"Record: {Path(state_dir) / speak.LOSS_NAME}\n"
        f"Escalated: {gstate.now_iso()}\n"
        "\n"
        "Filed by the guardian's own tick, so a burst during an "
        "`agent-supervisord` outage is escalated once the backend answers. One "
        "item per incident: a later burst refreshes this one rather than adding a "
        "second, and closing it ends the incident.\n"
        "\n"
        "## What did not get said\n"
        f"{said}"
    )


class VoiceLossEscalator:
    """One tick's judgement about `voice-loss.md`: file, refresh, or stay quiet.

    Built once per guardian process (`guardian.py::__init__`) and called from
    `tick()`, above the `infra_down` / `broken` / `paused` early returns for the
    reason `check_pool` gives: a check seated below those returns is a check that
    does not run in the states where things are going wrong — and a dead speaker
    during an outage is exactly that state. It returns a report rather than
    alerting, so the caller owns the log line.
    """

    def __init__(self, state_dir, *, base_url: str,
                 timeout: float = _TIMEOUT_SECONDS,
                 retry_seconds: float = RETRY_SECONDS):
        self.dir = Path(state_dir)
        self.base_url = str(base_url).rstrip("/")
        self.timeout = float(timeout)
        self.retry_seconds = float(retry_seconds)

    # ── the tick ───────────────────────────────────────────────────────
    def tick(self, now: float | None = None) -> dict:
        """Read the record, compare it to the cursor, and act at most once.

        Never raises: the last `except` is a belt for an unexpected shape, and the
        transport layer below already turns a refused connection into `None`.
        """
        now = time.time() if now is None else float(now)
        try:
            record = speak.read_loss_record(self.dir)
            if record is None:
                return self._report("no-record")
            cursor = gstate.read_json(self.cursor_path) or {}
            seen = _occurrences(cursor.get("occurrences"))
            if record["occurrences"] <= seen:
                # Nothing new has been lost since the last time this acted, so the
                # board already carries this burst. This is the branch that keeps a
                # 5 s loop from re-filing one incident 720 times a hour.
                return self._report("unchanged", occurrences=record["occurrences"],
                                    escalated=seen)
            if not self._attempt_is_due(cursor, now):
                return self._report("retry-waiting",
                                    next_attempt_ts=float(cursor.get("next_attempt_ts") or 0))

            # ONE read of the board decides all three outcomes, so the create and
            # the refresh can never be answered by two different views of it.
            rows = self._rows()
            if rows is None:
                # The board could not be read at all. The cursor does not move —
                # the escalation is still owed — and `next_attempt_ts` bounds the
                # retry, because a tick is every 5 s and an unreadable board is
                # not a reason to POST twelve times a minute.
                return self._retry(cursor, now, "unreachable",
                                   occurrences=record["occurrences"])
            open_row = self._pick(rows, open_only=True)
            if open_row is not None:
                return self._refresh(record, cursor, open_row, now)
            closed_row = self._pick(rows, open_only=False)
            if closed_row is not None:
                # A closed incident stays closed: not refreshed, not re-opened,
                # and not answered with a fresh row that overrules the person who
                # closed it. The cursor advances anyway, because the alternative
                # is a board read every tick for an incident someone finished
                # with, and the record on disk is still the evidence.
                self._write_cursor(record, now, action="closed-suppressed",
                                   item_id=_row_id(closed_row))
                return self._report("closed-suppressed",
                                    occurrences=record["occurrences"],
                                    item_id=_row_id(closed_row))
            return self._create(record, cursor, now)
        except Exception as exc:  # noqa: BLE001 — a watch never owns the tick
            return self._report("failed", error=f"{type(exc).__name__}: {exc}")

    # ── the three outcomes ─────────────────────────────────────────────
    def _create(self, record: dict, cursor: dict, now: float) -> dict:
        """File the one coalesced item for this incident."""
        reply = self._post(_CREATE_PATH, {
            "name": ITEM_NAME[:120],
            "description": item_body(record, state_dir=self.dir),
            "board": ITEM_BOARD,
            # NOT `up_next`, which is what `notify.py::_backlog_task` posts for a
            # rollback. `draft` is the only status autotriage's pool filter reads
            # (#1893 measured an alarm filed where nothing polls it), and the
            # ruling on #1798 says the same for alerts.
            "status": ITEM_STATUS,
            "priority": ITEM_PRIORITY,
        })
        if not _succeeded(reply):
            return self._retry(cursor, now, "create-failed",
                               occurrences=record["occurrences"], reply=reply)
        self._write_cursor(record, now, action="created",
                           item_id=_row_id(reply))
        return self._report("created", occurrences=record["occurrences"],
                            item_id=_row_id(reply))

    def _refresh(self, record: dict, cursor: dict, target: dict, now: float) -> dict:
        """Advance the one open row: count and last-seen move, no second item."""
        task_id = target.get("id")
        reply = self._post(_UPDATE_PATH, {
            "id": task_id,
            "description": item_body(record, state_dir=self.dir),
            # The daily-note pilot's fix, for the identical reason
            # (`app/autonomy.py:3325-3335`): the route DROPS a body shorter than
            # the one on disk unless this is set (`app/routers/backlog.py:614`),
            # and a later burst whose quoted utterances are shorter than the last
            # one's IS shorter. Without the flag the route answers 200 while the
            # count on the board stays where it was.
            "force_body_replace": True,
        })
        if not _succeeded(reply):
            return self._retry(cursor, now, "update-failed",
                               occurrences=record["occurrences"],
                               item_id=task_id, reply=reply)
        self._write_cursor(record, now, action="refreshed", item_id=task_id)
        return self._report("refreshed", occurrences=record["occurrences"],
                            item_id=task_id)

    # ── the board, over HTTP ───────────────────────────────────────────
    def _rows(self) -> list | None:
        """Board rows matching our prefix, or None when the board is unreadable.

        Asked of the board's own list route rather than answered by reading the
        backlog directory, because the board is the thing that knows whether an
        item is open — and because this process is not allowed to import the
        reader that would do it in-process (module docstring).

        `None` is the answer that keeps the alarm honest: `tick()` files nothing on
        the strength of a read that failed, because creating on that guess is how a
        duplicate is born and this alarm repeats until it lands. The daily-note
        pilot risks the duplicate instead (`app/autonomy.py:3247-3249`) and can
        afford to, because it is filed once per event rather than retried.
        """
        body = self._request("GET", _LIST_PATH, params={
            "board_id": ITEM_BOARD, "q": ITEM_PREFIX})
        return body if isinstance(body, list) else None

    @staticmethod
    def _pick(rows: list, *, open_only: bool) -> dict | None:
        """The first row of OURS whose open-ness is `open_only`.

        Matched on the name prefix, not on the body — `?q=` searches both, so the
        needle alone would also return an item that merely quotes this one — and
        the open/closed line is `canonical_status`, so a row carrying a retired
        spelling (`closed`, `cancelled`) is read as the closed incident it means
        instead of being refreshed back to life.
        """
        for row in rows:
            if not isinstance(row, dict):
                continue
            if not str(row.get("name") or "").startswith(ITEM_PREFIX):
                continue
            is_open = canonical_status(row.get("status")) in OPEN_STATUSES
            if is_open == open_only:
                return row
        return None

    def _post(self, path: str, payload: dict):
        return self._request("POST", path, payload=payload)

    def _request(self, method: str, path: str, *, payload: dict | None = None,
                 params: dict | None = None):
        """One loopback JSON round-trip. None on any failure; never raises.

        A refused connection is the MOST likely state to coincide with a dead
        speaker, so it is the normal path here and not an exception: it becomes
        None, the cursor does not move, and the next due tick tries again.
        """
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        try:
            req = urllib.request.Request(
                url, data=data, method=method,
                headers={"Content-Type": "application/json"} if data else {})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if not 200 <= resp.status < 300:
                    return None
                raw = resp.read().decode("utf-8", "replace")
            return json.loads(raw) if raw.strip() else None
        except Exception:  # noqa: BLE001 — the alarm outlives its side effect
            return None

    # ── the cursor ─────────────────────────────────────────────────────
    @property
    def cursor_path(self) -> Path:
        return self.dir / CURSOR_NAME

    def _attempt_is_due(self, cursor: dict, now: float) -> bool:
        until = cursor.get("next_attempt_ts")
        try:
            return now >= float(until or 0.0)
        except (TypeError, ValueError):
            return True

    def _retry(self, cursor: dict, now: float, reason: str, **extra) -> dict:
        """Say "not delivered", keep the escalation owed, and bound the retry."""
        gstate.write_json_atomic(self.cursor_path, {
            "schema": 1,
            "occurrences": _occurrences(cursor.get("occurrences")),
            "next_attempt_ts": now + self.retry_seconds,
            "last_attempt": gstate.now_iso(),
            "last_reason": reason,
        })
        return self._report(reason, next_attempt_ts=now + self.retry_seconds, **extra)

    def _write_cursor(self, record: dict, now: float, *, action: str,
                      item_id=None) -> None:
        gstate.write_json_atomic(self.cursor_path, {
            "schema": 1,
            "occurrences": int(record["occurrences"]),
            "action": action,
            "item_id": item_id,
            "last_seen": record.get("last_seen"),
            "escalated_at": gstate.now_iso(),
            "next_attempt_ts": 0.0,
        })

    def _report(self, reason: str, **extra) -> dict:
        out = {"reason": reason}
        out.update(extra)
        return out


# ── small readers with no opinion ─────────────────────────────────────
def _occurrences(value) -> int:
    """A cursor's count as an int, 0 for anything that is not one.

    A cursor written by a half-finished version of this file, or edited by hand,
    must not be able to silence the alarm by reading as a huge number — so a
    non-number is 0, which means "escalate", and not a comparison that throws.
    """
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def _row_id(source) -> int | None:
    """The row id out of a route reply or a board row — `bool` screened out.

    Both carry the key as `id`; neither is trusted to hold an int. The screening
    is `notify.py`'s lesson, restated at its own `_backlog_task`: `True` is an
    `int` subclass, so an id of `true` passes `> 0` and reads as a row that was
    never filed. None here means the cursor records no id, which is honest — a
    pointer invented from a reply that did not carry one is worse than no pointer.
    """
    if not isinstance(source, dict):
        return None
    value = source.get("id")
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _succeeded(reply) -> bool:
    """True only for a reply that says the row was written.

    `app/routers/backlog.py` answers a create with `{"success": true, "id": N}`,
    and `notify.py`'s lesson is that a 2xx alone is not delivery: the route
    defaults a missing `name` to "New Task" and still returns 200, so trusting the
    status line files a task that says nothing and reads as delivered.
    """
    return isinstance(reply, dict) and reply.get("success") is True
