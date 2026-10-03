"""Alert fan-out. Stdlib only, and no channel may raise.

Six channels, ordered so the most reliable goes first. They fail differently
on purpose: the ledger works when the network is down, the journal works when
the filesystem state dir is unreadable, the desktop notification works when
nobody is looking at a browser, the spoken alert works when nobody is looking
at the *screen*, and the vault note works tomorrow morning.

This module is the one place a Lloyd alert becomes visible to a human, which
is why the nag oneshot was folded into it: before, `lloyd-guardian-nag.service`
ran its own inline `notify-send`, so it was structurally incapable of ever
gaining a channel this module grew. Anything that wants to announce something
goes through `alert()`.

Deliberately **not** `app/discord_notify.py`: it is async, it imports
`app.config`, and with `config.yaml`'s token empty — which is the current
production state — it degrades to `logger.warning`, which lands in
`logs/server.err` where nobody reads it. The gap between "alerting configured"
and "alerting works" is exactly what bites during an incident, so this module
reports an unconfigured channel as a visible fact instead of swallowing it.

The channel that actually closes the loop is `backlog_task`: after a rollback,
Lloyd wakes up on known-good code with a work item naming what was reverted,
why, and the `git cherry-pick` that restores his work. That converts a
rollback from punishment into a task, which is the point of the whole design.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

# Sibling module — stdlib only, and staged alongside this file by
# `agent-services/bin/guardian-stage.sh`'s `guardian/*.py` glob. #1887: `_vault_note`
# used to `open(note, "a")` a note that did not exist, which CREATED the day's file
# from a blank lead and a `##` heading with no front matter at all, so the fresh-note
# header lives in `daily_note.py` and every writer of `memory/<date>.md` gets its
# block from that one place.
import daily_note  # noqa: E402

# ── The daily note's incident format (#1536) ──────────────────────────────
#
# `_vault_note`'s own section header, as a constant because the coalescing has to
# FIND the sections it wrote: a format written in one place and re-typed in another
# is how a retraction stops matching the alarm it retracts.
DAILY_SECTION_PREFIX = "## Self-mod guardian: "
# What an OPEN incident's section ends with. Written by `_vault_note` for a
# coalescing alert and removed only by `resolve`, so "is this incident still open"
# is answerable from the note itself — no state file, and a guardian restart ten
# minutes into an incident cannot lose it and start a second section.
DAILY_STILL_OPEN = "_(still open on the next check)_"
# What replaces that marker when the condition clears. A prefix rather than prose so
# a reader scanning the note sees the section's state, not another alarm body.
DAILY_CLEARED_PREFIX = "cleared:"
#: How many dated daily notes `resolve` scans back when it retracts an alarm (#1590). An
#: alert written at 23:05 and cleared at 00:05 lives in TWO files, because `_daily_note` is
#: keyed on the date it is called; scanning today's alone finds nothing to seal and used to
#: report the alarm retracted anyway. Two days is the minimum that can reach a
#: midnight-spanning incident; three leaves an incident that ran two days before its
#: retraction still reachable in the note that alerted. Older than the window is reported,
#: never rewritten — see `resolve`.
DAILY_SCAN_DAYS = 3
#: A daily note's filename stem as `_daily_note` writes it. Only a dated note can hold an
#: open section, so only dated notes are consulted outside the window.
DATED_NOTE_STEM = re.compile(r"\d{4}-\d{2}-\d{2}")

# ── ALERT.md's incident format (#1967) ────────────────────────────────────
#
# The state-dir alarm file is the surface an agent reads FIRST during an
# incident, and until #1967 `resolve()` never touched it: the 2026-09-30
# runtime-data alarm stood un-retracted for over a day — eight `cleared:`
# lines reached the daily note that evening while `ALERT.md` went on ordering
# a reader to remove `~/lloyd/.t`, a directory absent since the writer fix.
# Same coalescing rule as the daily note, one shared spelling each.
#: The single-slot alarm file `alert()` overwrites on EVERY run mode — it is
#: written at `alert()` above the `external` gate, so the retraction lives
#: above that gate too or a drill can raise an alarm it can never close.
ALERT_FILE_NAME = "ALERT.md"
#: Header key that closes an incident inside ALERT.md, named for
#: `DAILY_CLEARED_PREFIX` so the two surfaces answer the same question with
#: the same marker. A reader ages the stamp that follows it — zone-marked per
#: #1912 — to decide whether an inherited alarm is live.
ALERT_CLEARED_PREFIX = "cleared:"
#: What stands between a sealed file's header and the quoted original text:
#: the alarm's imperative instructions survive as a RECORD, not as a live
#: directive, so closing the incident never destroys the evidence of it.
ALERT_CLEARED_BANNER = ("CLEARED — the incident described below is closed. Its "
                        "text is kept quoted for the record: do not act on it.")
#: A header key line of ALERT.md (`level: …`, `written: …`, `cleared: …`). The
#: retraction finds the file's title and its own marker through this and
#: nothing else, so an alert body can never impersonate a header key.
_ALERT_HEADER_KEY = re.compile(r"^[a-z_]+: ")


def _run(cmd: list[str], timeout: float = 5.0) -> bool:
    try:
        subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
        return True
    except Exception:
        return False


def _channel_on(var: str) -> bool:
    """Master mute, checked at dispatch time. Mirrors `speak.voice_enabled`.

    The `external` gate cannot be the only mute, because two channels are
    reachable from *any* process that has a session bus and a journal: the
    desktop toast and the journal line. Any caller that builds a `Notifier`
    with `external` left at its default — which is every in-process test, and
    anything else not named "drill" — fans out to the room.

    On 2026-09-07 every self-mod gate run (each one executes the full suite)
    put `Lloyd guardian: real rollback / body` on the user's screen and wrote
    `STILL BROKEN :: 2026-09-06 liveness failed` to the live journal at
    priority 2. Both were fixture strings from tests that only care whether a
    *different* channel dispatched. A fake critical incident in the live
    journal is the same pollution the `external` gate was added for on
    2026-09-06 — it just arrives through a path that gate never covered.
    """
    return str(os.environ.get(var, "1")).strip().lower() not in (
        "0", "false", "no", "off", "")


NEEDS_HUMAN_MARKER = "needs a human"


def asks_for_a_human(title: str, body: str) -> bool:
    """True when an alert's own message says the guardian cannot act on it.

    A *declaration* in prose, not a severity: two sites end "Not rewriting
    history — this needs a human" at `level="error"`, while the backlog gate used
    to read level and trigger only — so the one channel that produces work a
    human later sees in a queue was the one channel skipped.

    Deliberately matched on `title` + `body` and not on `evidence`: evidence is
    quoted logs and diff text, which can carry the phrase from a message the
    guardian is quoting rather than one it is declaring.

    This is the fallback route, not the primary one. `alert(needs_human=True)` is
    the real flag — "Guardian self-test failed" is arguably the most
    human-requiring alert in the family and never says the words, so prose
    matching alone covers two of three members and breaks on every reword.
    """
    return NEEDS_HUMAN_MARKER in f"{title}\n{body}".lower()


# ── the backlog channel's duplicate guard (#2080) ─────────────────────────────
#: The board's own list route, read before `task-create` is POSTed. The board is
#: the thing that knows whether an item of this title is already open, which is
#: why the question is asked over HTTP rather than answered by re-reading the
#: backlog directory from this process — the same reasoning `voiceloss.py`'s `_rows`
#: gives for its own read of this same route.
BACKLOG_LIST_PATH = "/api/backlog/tasks"
#: The board `_backlog_task` files on. Named in its payload since #1990, and the
#: duplicate read is scoped to the same board, so an item on `personal` that
#: happens to carry a guardian's title can never silence that guardian alert.
BACKLOG_ITEM_BOARD = "lloyd"
#: Where a suppressed filing is recorded, in the guardian's own state dir: one
#: small JSON written by `gstate.write_json_atomic`, the shape `poolwatch.py`'s
#: `pool-silence.json` and `voiceloss.py`'s `voice_loss_cursor.json` are. On disk
#: rather than in a `Notifier` attribute for the reason those two give: the skip
#: has to outlive the process that made it, and be countable by something other
#: than a log line. It records what was skipped and never decides the next one —
#: see `_open_duplicate_on_board`, which is the only suppression input.
BACKLOG_SKIP_STATE_NAME = "backlog-skipped.json"
#: Same loopback bound as the POST it precedes. `voiceloss.py` sets its own timeout
#: off that same number, for the same reason: a loopback that is not answering is
#: the failure mode shared by the alert and the read that precedes it.
BACKLOG_READ_TIMEOUT_SECONDS = 5.0


class Notifier:
    def __init__(self, *, ledger: Path, state_dir: Path, vault_root: str,
                 backend_url: str = "http://127.0.0.1:8080",
                 external: bool = True, voice: bool = True,
                 voice_window: float = 3600.0):
        self.ledger = ledger
        self.state_dir = Path(state_dir)
        self.vault_root = Path(vault_root)
        self.backend_url = backend_url.rstrip("/")
        # When False, only the two channels scoped to this guardian's own state
        # dir fire (ledger, ALERT.md). The drill runs a real guardian against a
        # throwaway repo, and without this its test rollbacks land in the live
        # vault daily note and file real backlog tasks — indistinguishable from
        # production incidents. That actually happened: two drill rollbacks
        # (26574f87, ddd6d1d0) were written to the 2026-09-06 daily note naming
        # commits that exist only in a deleted scratch clone, and reading that
        # note later suggested the audit trail had lost events.
        self.external = external
        # Speech gets its own, much longer repeat window than the toast. The
        # nag fires every 15 minutes for as long as a BROKEN state lasts, and
        # guardian.py's in-memory dedupe cannot see it because it is a separate
        # process — so without this a bad night would say the same sentence out
        # loud ninety-six times. See speak.should_speak, which keeps that
        # record on disk precisely so both producers share it.
        self.voice = voice
        self.voice_window = voice_window

    def alert(self, level: str, title: str, body: str, *, evidence: str = "",
              commit: str = "", trigger: str = "", tag: str = "",
              needs_human: bool = False, coalesce: bool = False) -> dict:
        """Fan out one alert. Returns per-channel success for the heartbeat.

        `coalesce` changes the daily note only, and only for this title. The
        condition it reports can be one incident that outlasts many checks, and the
        daily note is the surface a human reads: the runtime-data stray alert used
        to append a whole section on every hourly check — 21 copies of ONE incident
        in `memory/2026-09-25.md`, each naming a different snapshot of the stray set,
        none ever retracted (#1536). With the flag the incident gets ONE section,
        refreshed in place to the latest finding, and closed by `resolve()`. Every
        other channel still fans out per firing: the ledger is the per-check audit
        trail rather than a reading surface, and the toast and the speech already
        have their own repeat windows.

        `needs_human=True` says the guardian has run out of actions: rollback is
        not the right answer here, and only a person can make it one. It routes
        to the backlog channel exactly like `critical` and `trigger` do, because
        a message addressed to a human has to arrive somewhere a human reads.

        Until #775 the backlog gate read `level == "critical" or trigger`, and all
        three of these sites pass `level="error"` with no trigger. Fifteen
        "Service down, but no promotion to revert" notices were recorded between
        2026-09-06 and 2026-09-14 and not one became a task: each one appended a
        ledger row, overwrote ALERT.md (single write, so only the last survives),
        wrote a journal line, toasted, spoke where quiet hours allowed, and
        appended a section to that day's daily note — the 2026-09-09 note stood at
        167 KB and 341 sections. The prose fallback in `asks_for_a_human` catches
        a future site that forgets the flag; the flag catches the sentence that
        never says it.
        """
        results: dict[str, bool] = {}
        text = f"{title}\n\n{body}".strip()
        if evidence:
            text += f"\n\n--- evidence ---\n{evidence[:4000]}"

        results["ledger"] = self._ledger(level, title, body, evidence, commit, trigger, tag)
        results["alert_file"] = self._alert_file(level, title, text)
        if not self.external:
            return results
        results["journal"] = self._journal(level, f"{title} :: {body}")
        results["desktop"] = self._desktop(level, title, body)
        results["voice"] = self._speak(level, title, body)
        results["vault"] = self._vault_note(title, text, coalesce=coalesce)
        # Four ways into the one channel that becomes work: a terminal state, a
        # rollback's trigger, a site's explicit flag, and the prose that predates
        # the flag. Everything above this line is either ephemeral, overwritten
        # (ALERT.md is write_text — last-writer-wins), or buried in the daily
        # note, so an alert that stops here has no owner.
        if (level == "critical" or trigger or needs_human
                or asks_for_a_human(title, body)):
            results["backlog"] = self._backlog_task(title, text, commit, tag)
        return results

    def announce(self, title: str, body: str = "", level: str = "info") -> dict:
        """Say something without recording an incident.

        `alert` is for events that must survive being missed, so it writes
        ALERT.md, appends to the ledger, and above a threshold files a backlog
        task. Two callers want the *sound* without any of that bookkeeping:

        * a successful promotion, which is not an incident at all — the ledger
          already records it as `promoted`, and a second row saying the same
          thing in different words is how two sources of truth begin;
        * the 15-minute nag, which re-announces an incident that has **already**
          been recorded. Routed through `alert` it would append a ledger row
          and file a fresh backlog task every 15 minutes for as long as the
          BROKEN state lasted, burying the one the rollback filed — the exact
          failure `_backlog_task` was hardened against once already.

        What is left is the three channels a human experiences in the room:
        journal, toast, voice. `level` still steers journal priority and toast
        urgency, so the nag can be loud without being permanent.
        """
        results: dict[str, bool] = {}
        if not self.external:
            return results
        results["journal"] = self._journal(level, f"{title} :: {body}")
        results["desktop"] = self._desktop(level, title, body)
        results["voice"] = self._speak(level, title, body)
        return results

    # ── channels ───────────────────────────────────────────────────────
    def _ledger(self, level, title, body, evidence, commit, trigger, tag) -> bool:
        try:
            import gstate
            gstate.append_event(self.ledger, {
                "event": "alert", "level": level, "title": title,
                "body": body[:2000], "evidence": evidence[:4000],
                "commit": commit, "trigger": trigger, "tag": tag,
            })
            return True
        except Exception:
            return False

    def _journal(self, level: str, message: str) -> bool:
        if not _channel_on("LLOYD_JOURNAL_ALERTS"):
            return False
        prio = {"critical": "2", "error": "3", "warn": "4"}.get(level, "5")
        return _run(["systemd-cat", "-t", "lloyd-guardian", "-p", prio,
                     "--", "echo", message[:4000]])

    def _desktop(self, level: str, title: str, body: str) -> bool:
        if not _channel_on("LLOYD_DESKTOP_ALERTS"):
            return False
        urgency = "critical" if level == "critical" else "normal"
        env_ok = bool(os.environ.get("DBUS_SESSION_BUS_ADDRESS"))
        if not env_ok:
            return False
        return _run(["notify-send", "-u", urgency, f"Lloyd guardian: {title}", body[:400]])

    def _speak(self, level: str, title: str, body: str) -> bool:
        """Say it out loud, in Lloyd's cloned voice. See speak.py.

        Reports *dispatched*, not *heard*: synthesis and playback happen in a
        detached child so this cannot add seconds to a loop the unit watchdogs
        at 90s. Whether the sound actually came out is in voice.log — and when
        it did not, `speak.py` has already written a durable record to
        voice-loss.md and put a locally-generated chirp on the player, because
        the utterance that got `[Errno 111]` at 2026-09-28 11:51:18 was the alert
        announcing an outage of the same supervised tree that hosts the TTS
        server, and a `voice.log` line was the only thing that knew (#1806).

        Below the `external` gate, for the same reason as the vault note and
        the backlog task: the drill runs a real guardian against a throwaway
        repo, and a rehearsal that announces a rollback out loud is
        indistinguishable from a production incident to anyone in the room.
        """
        if not self.voice:
            return False
        try:
            import speak
        except Exception:
            return False
        try:
            return speak.dispatch(level, title, body, self.state_dir,
                                  window=self.voice_window)
        except Exception:
            return False

    def _alert_file(self, level: str, title: str, text: str) -> bool:
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            # `astimezone()` with no argument attaches the machine's zone to the
            # LOCAL reading: the digits, microseconds included, are untouched and
            # only the zone is named (#1912 — the same shape speak.py's `%z` got
            # from #1808). Writing this stamp naive on a box at -0700 left the
            # one artefact a human reads during an outage open to two readings,
            # and this repo's own comments held both of them seven hours apart:
            # `scripts/side_effect_traffic_census.py::parse_stamp` reads a naive
            # stamp as local, `app/skill_telemetry.py` reads one as UTC.
            (self.state_dir / ALERT_FILE_NAME).write_text(
                f"# {title}\n\nlevel: {level}\n"
                f"written: {datetime.now().astimezone().isoformat()}\n\n{text}\n",
                encoding="utf-8",
            )
            return True
        except Exception:
            return False

    def _alert_file_retract(self, title: str, note_text: str) -> bool:
        """Seal the incident ALERT.md is holding, when this title's cause clears (#1967).

        The companion of `_alert_file`, deliberately placed on its caller's side
        of the `external` gate: `alert()` writes that file BEFORE the gate
        returns, so an all-clear that retracted only behind the gate would leave
        every drill and `--no-external-alerts` run holding an alarm it can never
        close — a guard on one of two write surfaces is not a guard (09-22).
        The live witness at filing: the 2026-09-30 runtime-data alarm still
        read as a live directive over a day after `~/lloyd/.t` was gone and the
        writer fix (#1906) had settled, because all eight `cleared:` lines that
        evening went to the daily note and none here.

        Title-scoped because ALERT.md is ONE last-writer-wins slot: the file is
        rewritten only when its H1 names this exact title, so resolving the
        tmpwatch alert can never retract a live runtime-data alarm sharing the
        path. The alarm's text survives QUOTED below a `CLEARED` banner — the
        retraction changes what the instructions ARE (a live order → the record
        of one) without destroying the evidence — and the original header keys,
        `written:` included, are copied byte-for-byte so a reader can still age
        the alarm itself. No file is never created: the hourly all-clear runs
        whether or not anything ever alerted, and conjuring an ALERT.md to
        retract would manufacture an incident in the surface agents read first.
        A file already carrying a `cleared:` header key is left byte-identical,
        which is what keeps the hourly all-clear silent (#1536's contract,
        extended to this surface).

        The return answers only THIS surface: True when the file is in a
        retracted-or-nothing-to-do state after the call, False when it exists
        for this title but could not be read or written — a claim the caller
        folds into `resolve`'s verdict rather than swallowing.
        """
        path = self.state_dir / ALERT_FILE_NAME
        try:
            if not path.is_file():
                return True
            body = path.read_text(encoding="utf-8")
        except OSError:
            return False
        split = self._alert_file_split(body)
        if split is None or split[0] != f"# {title}":
            return True          # another incident's live alarm, or not ours
        h1, keys, alert_text = split
        if any(k.startswith(f"{ALERT_CLEARED_PREFIX} ") or k == ALERT_CLEARED_PREFIX
               for k in keys):
            return True          # already sealed: idempotent silence
        stamp = datetime.now().astimezone().isoformat()
        quoted = "\n".join((f"> {ln}" if ln.strip() else ">")
                           for ln in alert_text.rstrip("\n").split("\n"))
        sealed = (f"{h1}\n\n"
                  + "\n".join(keys)
                  + f"\n{ALERT_CLEARED_PREFIX} {stamp} — {note_text}\n\n"
                  + ALERT_CLEARED_BANNER + "\n\n" + quoted + "\n")
        try:
            path.write_text(sealed, encoding="utf-8")
        except OSError:
            return False
        return True

    @staticmethod
    def _alert_file_split(body: str):
        """(H1 line, header key lines, alert text) of an ALERT.md, or None.

        The layout `_alert_file` writes — `# {title}`, a blank, the `level:`/
        `written:` run plus the `cleared:` line once a retraction has sealed
        it, a blank, the text — is parsed by those header keys and nothing
        else: the text always sits behind a blank line, so alert prose that
        begins with a `key: value`-looking sentence can never extend the
        header. None answers for bytes that are not this layout, and a file
        whose title cannot be read is a file the retraction must not touch.
        """
        lines = body.split("\n")
        if not lines or not lines[0].startswith("# "):
            return None
        i = 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        keys: list[str] = []
        while i < len(lines) and _ALERT_HEADER_KEY.match(lines[i]):
            keys.append(lines[i])
            i += 1
        if not keys:
            return None
        while i < len(lines) and not lines[i].strip():
            i += 1
        return lines[0], keys, "\n".join(lines[i:])

    def _vault_note(self, title: str, text: str, *, coalesce: bool = False) -> bool:
        """Write this alert's section into today's daily note.

        `coalesce=False` — every alert but the runtime-data stray one — appends a
        section per fan-out, byte-identical to the pre-#1536 format. `coalesce=True`
        opens a section on the first finding and REFRESHES that same section on every
        later finding of the same incident, so one incident is one section no matter
        how many checks it outlives.

        When the day's note does not exist yet it is CREATED with the shared
        front-matter header (`daily_note.fresh_header`, #1887) before the section is
        appended — the same block `app/post_capture._append_daily_note` gives a fresh
        note. It used to be created by the `open(note, "a")` below, from a blank lead
        and a `##` heading and no front matter, which made an alert that fired before
        any session capture a conformance violation on arrival:
        `scripts/vault/segment_scan.py` scores a file it cannot parse as missing BOTH
        required keys, and `memory/2026-09-30.md` reached
        `test_scan_exits_0_on_the_live_vault` that way. A note that already exists is
        not touched by this branch at all — its own header stays.
        """
        try:
            note = self._daily_note()
            if note is None:
                return False
            exists = note.is_file()
            if not exists:
                note.parent.mkdir(parents=True, exist_ok=True)
                note.write_text(daily_note.fresh_header(self._stamp(),
                                                        self._today().isoformat()),
                                encoding="utf-8")
            body = note.read_text(encoding="utf-8") if exists else ""
            open_at = self._daily_open_at(body, title) if coalesce else None
            if open_at is not None:
                start, end = open_at
                stale = body[start:end]
                lead = stale[:len(stale) - len(stale.lstrip())]   # keep its blank lead
                body = (body[:start] + lead + text.rstrip()
                        + "\n\n" + DAILY_STILL_OPEN + "\n" + body[end:])
                note.write_text(body, encoding="utf-8")
                return True
            with open(note, "a", encoding="utf-8") as f:
                if coalesce:
                    f.write(f"\n\n## Self-mod guardian: {title}\n\n{text.rstrip()}\n\n"
                            f"{DAILY_STILL_OPEN}\n")
                else:
                    f.write(f"\n\n## Self-mod guardian: {title}\n\n{text}\n")
            return True
        except Exception:
            return False

    def resolve(self, title: str, note_text: str) -> bool:
        """Close an open incident on the daily note. #1536.

        An alarm written to the surface a human reads is only half-written until its
        retraction is on that SAME surface. `memory/2026-09-25.md` holds 21 sections
        of imperative instructions ("Find the writer, move the data across, and
        remove the in-tree copy") for a condition that has since partly resolved, and
        zero lines saying it cleared — so the only surviving record of a finished
        incident is an instruction to do work nobody should do, and the count reads as
        escalating urgency instead.

        This replaces the open marker with one `cleared:` line, so a reader who does
        reach the stale instructions finds the contradiction at the foot of the same
        section rather than in a file they were never sent to. It writes nothing when
        no incident is open for `title`, which is what makes the second, third and
        hundredth all-clear check silent. No other channel is touched: a clear is not
        an alert, so no toast, no speech, no backlog task — and the ledger still holds
        every individual firing either way.

        ALERT.md is sealed first and by the same rules (#1967), ABOVE the
        `external` gate and before any of this: `alert()` writes that file on
        every run mode, so the retraction must run on every run mode too, and
        it is title-scoped because the file is one last-writer-wins slot —
        sealing one incident must never retract a different live one. The
        daily-note surfaces behind the `external` gate stay closed to a drill
        exactly as before; only the state-dir file, which the drill already
        wrote, gains its `cleared:` line.

        The return is a claim about the surfaces it could reach, not about the incident.
        True means: ALERT.md either held this title's alarm and is now sealed, held
        another title's, held nothing, or was already sealed; AND every daily note
        inside the last `DAILY_SCAN_DAYS` that it opened for `title` is now sealed —
        including the note of a day that has since passed, which is
        where an alarm raised before midnight still stands — or that none of them held an
        open section, which is the ordinary hourly all-clear and stays True so the
        idempotent-silence contract holds. False means it could not read or write either
        surface, or that
        an open section for `title` sits in a dated note OLDER than that window: a
        retraction reaches back a fixed number of days and no further, so an alarm beyond
        its reach is not a closed question and must not be reported as one. A note it never
        opened is never counted as cleared — reporting exactly that, `True` over a file
        `_daily_note` had not read because the clock had crossed midnight, was the defect
        (#1590): an alert written 23:05 and all-cleared 00:05 returned success and left the
        alarm's own section claiming it was still open.
        """
        retracted = self._alert_file_retract(title, note_text)
        if not self.external:
            return retracted
        # Checked before sealing, from the filenames: the out-of-window question is "is
        # there an alarm for this title my reach does not cover", and a section sealed a
        # moment ago must not be able to mask one it cannot touch.
        out_of_reach = bool(self._open_days_beyond_window(title))
        wrote = True
        for day in self._scan_days():
            note = self._daily_note_path(day)
            if note is None or not note.is_file():
                continue
            try:
                body = note.read_text(encoding="utf-8")
            except OSError:
                wrote = False
                continue
            sealed = body
            while True:
                open_at = self._daily_open_at(sealed, title)   # newest still-open, if any
                if open_at is None:
                    break
                start, end = open_at
                stale = sealed[start:end].rstrip()
                # The marker is REPLACED, not followed: leaving it standing above the
                # `cleared:` line means the note still claims the incident is open to
                # anyone scanning for that marker, which is the one question a
                # retraction exists to change the answer to.
                if stale.endswith(DAILY_STILL_OPEN):
                    stale = stale[:-len(DAILY_STILL_OPEN)].rstrip()
                sealed = (sealed[:start] + stale
                          + f"\n\n{DAILY_CLEARED_PREFIX} {note_text}\n"
                          + sealed[end:])
            if sealed == body:
                continue
            try:
                note.write_text(sealed, encoding="utf-8")
            except OSError:
                wrote = False
        return False if (out_of_reach or not retracted) else wrote

    # ── daily-note incident plumbing (#1536) ──────────────────────────────
    #
    # Incident state is read back OUT of the note, never held in this process, so a
    # guardian restart mid-incident refreshes the existing section instead of opening
    # a second one: the incident is a fact about the note, not about this runtime.

    def _daily_note(self):
        """Today's note, or None when the vault has no `memory/` to write into."""
        return self._daily_note_path(self._today())

    def _daily_note_path(self, day: date):
        """`<vault>/memory/<day>.md`, or None when the vault has no `memory/`.

        Split out of `_daily_note` because a retraction is no longer keyed on today alone:
        the alarm being retracted may live in yesterday's file, and the only thing that
        distinguishes the two is the date in the name.
        """
        memory = self.vault_root / "memory"
        if not memory.is_dir():
            return None
        return memory / f"{day.strftime('%Y-%m-%d')}.md"

    def _today(self) -> date:
        """The date daily notes are keyed on.

        A method rather than a `datetime.now()` call site so a test can stand on day D+1
        without freezing a clock across a whole module — which is the only way the cross-day
        retraction can be observed at all (#1590).
        """
        return datetime.now().date()

    def _stamp(self) -> datetime:
        """The instant a fresh daily note's `timestamp:` front-matter key records.

        A method beside `_today()` for the same reason (#1590): a test standing on a
        given day must not also be standing on a given second, or the block it
        compares is a measurement of the wall clock. `_today()` answers the FILENAME
        and this answers the KEY, and #1887's whole point is that the two agree with
        what the session-capture writer puts in the same note.
        """
        return datetime.now()

    def _scan_days(self) -> list[date]:
        """The days whose daily notes `resolve` retracts across, newest first."""
        newest = self._today()
        return [newest - timedelta(days=back) for back in range(DAILY_SCAN_DAYS)]

    def _open_days_beyond_window(self, title: str) -> list[date]:
        """Dated notes OUTSIDE the window that still hold an open section for `title`.

        Bounded by the filename, not by a walk of anything: only a `YYYY-MM-DD.md` can be an
        incident's daily note, so this is one `iterdir` of `memory/`. An alarm older than
        `DAILY_SCAN_DAYS` is precisely the case a retraction cannot make, which is why it is
        reported rather than rewritten.
        """
        memory = self.vault_root / "memory"
        if not memory.is_dir():
            return []
        in_window = {day.isoformat() for day in self._scan_days()}
        try:
            stems = sorted(p.stem for p in memory.iterdir()
                           if DATED_NOTE_STEM.fullmatch(p.stem or ""))
        except OSError:
            return []
        beyond = []
        for stem in stems:
            if stem in in_window:
                continue
            try:
                body = (memory / f"{stem}.md").read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if self._daily_open_at(body, title) is not None:
                beyond.append(date.fromisoformat(stem))
        return beyond

    @staticmethod
    def _daily_sections(text: str) -> list[tuple[str, int, int]]:
        """(title, body_start, body_end) per guardian section, in file order.

        The prefix is matched as a literal so a `###` sub-heading inside a body does
        not split its parent, and lines inside a fenced block are skipped so a note
        quoting this very format cannot invent a section out of a code sample.
        """
        out: list[tuple[str, int, int]] = []
        cur: tuple[str, int] | None = None
        offset = 0
        fence = False
        for line in text.splitlines(keepends=True):
            if line.strip().startswith(("```", "~~~")):
                fence = not fence
            elif not fence and line.startswith(DAILY_SECTION_PREFIX):
                if cur is not None:
                    out.append((cur[0], cur[1], offset))
                cur = (line.rstrip("\n")[len(DAILY_SECTION_PREFIX):], offset + len(line))
            offset += len(line)
        if cur is not None:
            out.append((cur[0], cur[1], len(text)))
        return out

    def _daily_open_at(self, text: str, title: str):
        """(body_start, body_end) of `title`'s most recent OPEN section, else None.

        Open means the body still ends with `DAILY_STILL_OPEN`. Newest-match matters:
        after an incident was cleared and a second one opened, the closed one is a
        historical record and the new one is the live incident.
        """
        for sect in reversed(self._daily_sections(text)):
            if sect[0] != title:
                continue
            body = text[sect[1]:sect[2]]
            return (sect[1], sect[2]) if body.rstrip().endswith(DAILY_STILL_OPEN) else None
        return None

    def _backlog_task(self, title: str, text: str, commit: str, tag: str) -> bool:
        """File a backlog item so the revert becomes work, not a mystery.

        The board is asked before it is written to (#2080). Three items share the
        H1 `[guardian] Guardian self-test failed` (#1279, #1280, #1398) and two
        share the automod-skill rollback title (#1346, #1350), every one of them
        filed while an earlier copy of itself was still open, because neither side
        of this seam looked: this method POSTed unconditionally and
        `app/routers/backlog.py::backlog_task_create` computes `max_id + 1` and
        writes. So while an OPEN item on `BACKLOG_ITEM_BOARD` carries the exact
        name this would post, the filing is skipped and recorded under
        `BACKLOG_SKIP_STATE_NAME`; close that item and the next firing files again,
        which is what makes this suppression rather than a mute.

        A read that fails, or answers anything but a list, does NOT suppress — it
        files. That is a deliberate difference from `voiceloss.py`'s `_rows`, which
        files nothing on a failed read: its filing is retried until it lands, and
        this one is not, so a backend outage — precisely when several filing-path
        alerts fire — would take #775's whole channel mute.

        Returns True only when the board accepted a filing. A skip returns False
        for the reason a failed POST does: no task file exists, and
        `results["backlog"]` reports what happened to this alert rather than what
        would have happened had the board been empty. Nothing reads that value
        today (`guardian.py` drops it), which is why the skip is recorded on disk
        instead of in the return.
        """
        item_name = f"[guardian] {title}"[:120]
        if self._open_duplicate_on_board(item_name):
            self._note_backlog_skip(item_name)
            return False
        body = text
        if commit:
            body += (f"\n\nYour work is preserved. To re-apply and investigate:\n"
                     f"    git cherry-pick {commit}\n")
        if tag:
            body += f"The pre-rollback tree is tagged `{tag}`.\n"
        # Field names match app/routers/backlog.py::backlog_task_create, which
        # takes `name`/`description`/`status`, NOT title/body. Sending the wrong
        # keys does not error — the endpoint defaults `name` to "New Task" and
        # returns 2xx, so the alert reads as delivered while filing a task that
        # says nothing. `status` must be one of _VALID_STATUSES.
        #
        # `status` is `draft` because that is the only status anything picks an
        # auto-filed item up from (#1990): autotriage's pool filter reads
        # `TRIAGE_POOL_STATUS` (`draft`) alone, and the implement pool's
        # `ready_confirmed` needs a confirmed triage verdict, which an item
        # filed here has never had. This posted the implement pool's status
        # until then, a dead state for an untriaged item that only the
        # reconciler's push-back rescued. `board` is named rather than inherited
        # from the route's default, as `voiceloss.py` (`ITEM_BOARD`) already
        # does: a default is right only until someone adds a board. Whether the
        # lloyd board is where operational alerts belong at all is a separate
        # ruling (#1798) that this does not settle.
        payload = json.dumps({
            "name": f"[guardian] {title}"[:120],
            "description": body[:6000],
            "board": "lloyd",
            "status": "draft",
            "priority": "high",
        }).encode()
        req = urllib.request.Request(
            f"{self.backend_url}/api/backlog/task-create",
            data=payload, headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if not (200 <= resp.status < 300):
                    return False
                created = json.loads(resp.read().decode("utf-8", "replace") or "{}")
                # Decide from the reply the endpoint actually sends. Since
                # `backlog_task_create` answers `{"success": true, "id": N}`
                # (app/routers/backlog.py:765) and never echoes the `name` it
                # filed, the name this used to read was always empty, so the
                # `if name else True` default was the production path and every
                # 2xx — `{"success": false}`, `{}` — reported a delivered filing
                # that filed nothing. A row id is the only field in that reply
                # that cannot be present unless a task file exists, so it is what
                # the verdict is now made of, along with the success flag.
                # `bool` is screened out because it is an `int` subclass and
                # `True > 0`, and an id of `true` is not a row.
                #
                # What this still cannot see is the drift `7da1e0e4` wrote the
                # name check against: a payload posting `title`/`body` instead of
                # `name`/`description` gets a 200 and an id for the empty task it
                # created, and nothing here will notice. That is a settled
                # decision, not an open one: payload-drift detection declined
                # 2026-09-29 (#1703): the echo variant shipped at 7da1e0e4 and was
                # inert because task-create replies only {success, id}, and a
                # read-back buys the same verdict one request later on the alert
                # path — a best-effort 5 s channel whose dict `guardian.py` drops
                # on the floor, so a read-back would only add a second way to
                # report "not filed" about a task that was filed. The guard is the
                # suite, which already runs on every edit to either side of this
                # seam: `test_the_guardians_captured_body_files_itself_through_the_real_route`
                # replays these bytes through the real route and asserts the task
                # file it wrote — H1, `board`, `status`, `priority`, the alert
                # text — which is the only surface that tells a real filing apart
                # from the `# New Task` with no alert text a drifted payload
                # produces. The H1 and the body are what tell them apart: both
                # land at `draft` since #1990, so status does not.
                # Weaken that written-file assertion, and live detection is owed
                # again.
                row_id = created.get("id")
                return (created.get("success") is True
                        and isinstance(row_id, int)
                        and not isinstance(row_id, bool)
                        and row_id > 0)
        except (urllib.error.URLError, OSError, ValueError):
            return False

    # ── the duplicate guard (#2080) ─────────────────────────────────────────
    def _board_rows(self, name: str):
        """Every board row whose text carries `name`, or None when it is unreadable.

        Asked of the board's own list route over the same loopback bound as the
        POST, with `?q=` as a pre-filter and nothing more: the route matches the
        needle against name, body, tags and board, so what comes back may include an
        item that merely QUOTES a guardian alert, and `_open_duplicate_on_board` is
        where the exact-name test happens. The needle is the name this file would
        have posted and nothing looser, because a row whose name equals it can never
        be filtered out by that needle, while a looser one only widens the scan.

        `None` — never an empty list — is the answer for a refused connection, a
        non-2xx, an unparseable body, or a body that is not a list. The two are
        different facts and must not be conflated: an empty list is a readable board
        with nothing open on it, and `None` is a board nobody can ask. Only `None`
        makes the caller file anyway.
        """
        query = urllib.parse.urlencode({"board_id": BACKLOG_ITEM_BOARD, "q": name})
        req = urllib.request.Request(
            f"{self.backend_url}{BACKLOG_LIST_PATH}?{query}", method="GET")
        try:
            with urllib.request.urlopen(req, timeout=BACKLOG_READ_TIMEOUT_SECONDS) as resp:
                if not (200 <= resp.status < 300):
                    return None
                rows = json.loads(resp.read().decode("utf-8", "replace") or "null")
        except (urllib.error.URLError, OSError, ValueError):
            return None
        return rows if isinstance(rows, list) else None

    def _open_duplicate_on_board(self, name: str) -> bool:
        """True only when the board says an OPEN item already carries this name.

        Exact equality on the row's `name`, not a prefix and no new matcher: the
        route derives that field from the file's H1 (`_row_from` over `_split_body`),
        so the string compared here is the string `backlog_task_create` wrote into
        that H1, which is the string this call would have posted.

        Open-ness is `voiceloss.canonical_status` against `voiceloss.OPEN_STATUSES`,
        the guardian's single copy of the board's status vocabulary — re-implemented
        there from `app/backlog_status.py` because a staged guardian module may not
        import `app.*`, and pinned equal to it by
        `tests/test_guardian_voice_loss_escalation.py::
        test_the_guardian_and_the_backend_copy_of_the_status_vocabulary_agree`.
        Importing it beats writing a second copy under `agent-services/guardian/`,
        and the import failing answers False for the same reason everything else
        undecided does.

        Every inability to decide answers False, and False means the alert files.
        That is the point: the only failure this guard can produce is a duplicate,
        and the only failure it must never produce is silence.
        """
        rows = self._board_rows(name)
        if rows is None:
            return False
        try:
            import voiceloss
        except Exception:
            return False
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("name") or "") != name:
                continue
            if voiceloss.canonical_status(row.get("status")) in voiceloss.OPEN_STATUSES:
                return True
        return False

    def _note_backlog_skip(self, name: str) -> None:
        """Record that a filing for `name` was suppressed, in the guardian's state dir.

        Keyed by the exact item name, so the record answers "how many times has this
        signature been silenced, and when last" without a log grep, and a second
        `Notifier` over the same `state_dir` shows the same count — the property the
        poolwatch watermark buys for the pool-silence alarm and this one buys for a
        repeat that arrives after a restart.

        It is a RECORD and never an input: reading it back here would let a skip
        outlive the item that justified it, and #2080's clause 5 is that closing the
        item must file again. The board is re-read on every filing, which is what
        makes the restart case correct rather than this file.

        Never raises. The skip already happened by the time this runs, and a state
        dir that cannot be written must not cost the ledger row, ALERT.md or the
        daily note that `alert()` fans out around this channel.
        """
        try:
            import gstate
            path = self.state_dir / BACKLOG_SKIP_STATE_NAME
            data = gstate.read_json(path) or {}
            entry = data.get("suppressed")
            entry = entry if isinstance(entry, dict) else {}
            prior = entry.get(name)
            prior = prior if isinstance(prior, dict) else {}
            try:
                count = int(prior.get("count") or 0)
            except (TypeError, ValueError):
                count = 0
            entry[name] = {
                "count": count + 1,
                "first_seen": prior.get("first_seen") or gstate.now_iso(),
                "last_seen": gstate.now_iso(),
                "last_seen_ts": time.time(),
            }
            data.update({
                "schema": 1,
                "suppressed": entry,
                "last_suppressed": name,
                "board": BACKLOG_ITEM_BOARD,
                "list_url": f"{self.backend_url}{BACKLOG_LIST_PATH}",
            })
            gstate.write_json_atomic(path, data)
        except Exception:
            return
