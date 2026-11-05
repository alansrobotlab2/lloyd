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
# The other word a sealed block may carry, and the only other one this module defines
# that CLOSES a section (`DAILY_CROSS_ROOT_PREFIX` below annotates a seal, it never
# makes one). It is NOT a retraction: `cleared:` records that the guardian watched the condition come
# back, `dated:` records only that a later read refutes the state the body quoted. The
# historical sweep of #2221 clause 5 writes it (`scripts/maintenance/guardian_note_sweep.py`)
# because the notes it must fix are older than `DAILY_SCAN_DAYS` above, which is the reach
# `resolve` will ever have — and nobody reading them months later can tell an inferred
# refutation from an observed recovery unless the two words stay different.
DAILY_DATED_PREFIX = "dated:"
#: The line `_vault_note` puts in a NEW coalescing section when the newest `cleared:`
#: line for that title names a root other than the one this check measured (#3628). It
#: is a third kind of line but NOT a third kind of retraction: `cleared:` and `dated:`
#: close a section, this one says WHY the section above it closed. The symptom it names
#: is #1536's own — one incident's whole block re-appended once per hour — because a
#: seal written by a process on a different data root removes the open marker exactly as
#: an honest one does, and the next tick then correctly owes a fresh section. Detection
#: only: whether a cross-root process may seal at all is a ruling #3628 does not take,
#: and no root is recorded here that the alert body and the clearance do not already
#: print.
DAILY_CROSS_ROOT_PREFIX = "cross-root seal:"
#: What `resolve` appends INSIDE a `cleared:` line when the section it is closing measured
#: a root this clearance does not name (#3792). The seal side's half of the constant above,
#: and the reason it is a SUFFIX inside the clearance's own line rather than a line of its
#: own: the line a human reads first is the `cleared:` line, and #3628's label only appears
#: if some later check still has the finding to re-alert — so on the day the finding really
#: did go, an unmarked foreign seal closed a live section and published a certification of
#: a tree nobody is standing in, which is `memory/2026-10-29.md`'s five `cleared:` lines
#: naming five deleted `lloyd-work/SM_*` trees. A different WORD from the prefix above on
#: purpose: that one annotates a new alert, this one qualifies a retraction, and a reader
#: (or a grep) has to be able to tell the two apart. Detection only, as #3628's is: the
#: ruling on whether a cross-root process may seal at all is owed elsewhere, and this
#: names no path that the clearance and the section's own body do not already print.
DAILY_CROSS_ROOT_CLEAR_PREFIX = "cross-root clearance:"
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

#: The line `_vault_note` writes INSTEAD of a second full copy of a section it has
#: already put on today's note (#3793). The head ends inside the alarm's own title
#: because the title is what the fold matches on — see `_fold_owed` — and a line whose
#: subject a reader cannot name is a silence with brackets around it.
#:
#: The shape is the scheduler writer's, from `app/discord_notify.py:346-349`
#: (`[daily-note fold: N earlier fires of this alarm today folded into this line; …]`),
#: which every guardian alarm line in `memory/2026-10-29.md` and `memory/2026-11-04.md`
#: already carries as a promise the guardian's own writer did not keep: 5 sections of
#: one title stood at 11,250 B of that file's 12,809 B (87.8 %), and 3 stood at 5,146 B
#: of 6,872 B (74.9 %).
DAILY_FOLD_HEAD = '[daily-note fold: guardian alarm "'
#: What every fold line promises about the NEXT copy. Both arms hold whatever the
#: incident does next: further firings today fold because the fold reads the day's own
#: note, and a NEW day's note carries no section to fold against, so the first firing
#: after midnight is a full body again. It says nothing about whether the alarm clears —
#: an all-clear that arrives after a fold has nothing left to seal and stays silent
#: (`resolve`'s documented idempotent-silence contract), so a tail that promised a
#: further body "when it clears" would be a sentence the note cannot honour.
DAILY_FOLD_TAIL = ("further firings of this alarm today fold into this line; the next "
                   "full body comes on the next calendar day]")
#: The count clause of a fold line this module wrote, and the only place the number of
#: folded firings is kept. Read back off the note rather than held in process memory,
#: because the 2026-10-29 repeats were written by five DIFFERENT processes — one
#: automod-round worktree per hourly tick — and state that dies with the writer resets
#: a counter a reader can still see on the page.
DAILY_FOLD_FIRES = re.compile(r"(\d+) earlier fires? today folded into this line")
#: The newest ISO-8601 instant inside an incoming alert body. The bodies the fold
#: collapses are stamped with the check's own instant — `guardian.py:474-475` puts
#: `checked %Y-%m-%dT%H:%M:%SZ` in the shadow alert's first sentence, and the vault
#: snapshot alarm's `state["text"]` carries the same shape — and that stamp is exactly
#: the line a fold throws away, so the fold has to carry it forward or the note loses
#: the one fact that says the alarm is still live and when it last spoke.
ISO_STAMP_IN_TEXT = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?")


#: The punctuation a clearance's prose puts after a path: the sentence carries on past
#: it, so a path token has to be read without the character that follows it.
_ROOT_TRAILING_PUNCT = "`*_.,;:!?()[]{}\"'"


def _absolute_roots(line: str) -> list[str]:
    """Absolute path tokens of one line, in the order they appear (#3628).

    A clearance names the root it measured as a bare absolute path — the runtime-data
    one is "no runtime stores inside the code tree of {REPO}, and no database with zero
    data rows shadowing a populated store under {DATA_ROOT}" — and the alert body names
    its own the same way, so comparing the two needs no new state anywhere. Deliberately
    dumb: whitespace-split, strip the punctuation a sentence leaves hanging on a path,
    keep what starts with `/`. A root is a path, and a path with a space in it is not
    something the guardian has ever printed.
    """
    out: list[str] = []
    for token in line.split():
        stripped = token.strip(_ROOT_TRAILING_PUNCT)
        if stripped.startswith("/") and len(stripped) > 1:
            out.append(stripped)
    return out


def _root_covered(root: str, others: list[str]) -> bool:
    """True when `others` already names `root`, or a root of which it is part (#3792).

    Presence, not sole-ness, and containment either way, which is #3628's rule reused:
    a body prints its root AND the files under it, so `/…/lloyd-data/kg.sqlite` is the
    same measurement as `/…/lloyd-data` and an all-clear naming the latter is honest
    about a section that named the former. The trailing slash is not decoration —
    `/home/alansrobotlab/lloyd` does NOT contain `/home/alansrobotlab/lloyd-data`, and a
    bare `startswith` would call the live code tree an ancestor of the live data root and
    silence the marker on the exact box this exists for.
    """
    for other in others:
        if root == other or root.startswith(other + "/") or other.startswith(root + "/"):
            return True
    return False


def _roots_outside(roots: list[str], others: list[str]) -> list[str]:
    """The roots in `roots` no root in `others` covers, with nested paths collapsed.

    Collapsing is what keeps a marker to two roots instead of four: `guardian.py`'s
    shadow body names its data root and then both halves of the shadowed pair under it,
    and only the root is the thing a reader needs beside the other side's. Order and
    duplicates are handled here so both callers stay boring.
    """
    uniq: list[str] = []
    for root in roots:
        if root in uniq or _root_covered(root, others):
            continue
        uniq.append(root)
    return [r for r in uniq
            if not any(o != r and r.startswith(o + "/") for o in uniq)]


def _clearance_measured_roots(cleared_line: str) -> list[str]:
    """The roots a `cleared:` line's own sentence names, minus any #3792 marker.

    `_cross_root_seal_line` asks which root the process that wrote this clearance
    MEASURED, and the marker `resolve` appends names a second root that process explicitly
    did not measure — the root of the section it was closing. Counting it would make a
    marked foreign clearance read as an honest one to the next alert from the live root
    (the marker prints that live root, so `measured_root` would be found "present"), and
    #3628's label would go missing on exactly the day it is owed. So the marker segment is
    cut off before the scan, and the answer stays the clearance's own sentence.
    """
    cut = cleared_line.find(DAILY_CROSS_ROOT_CLEAR_PREFIX)
    if cut >= 0:
        cleared_line = cleared_line[:cut]
    return _absolute_roots(cleared_line)


def _section_measured_roots(section_text: str) -> list[str]:
    """The roots one section's own body prints, minus #3628's label line.

    A section carrying `cross-root seal:` names the root of a THIRD process — the one that
    closed the section above it — and that is not a root this section's check measured.
    Reading it as one would let a label written by the alert side talk the seal side into
    naming a root the body never walked.
    """
    kept = "\n".join(ln for ln in section_text.splitlines()
                     if not ln.strip().startswith(DAILY_CROSS_ROOT_PREFIX))
    return _absolute_roots(kept)


def _foreign_roots(cleared: str | None, measured_root: str) -> list[str]:
    """The roots a clearance names that this check did not measure, in order.

    One predicate for two callers, so the seal label and the fold line can never
    disagree about what "foreign" means: `_cross_root_seal_line` prints the sentence,
    `Notifier._fold_line` names the newest root inside the fold (#3793). Empty means
    there is nothing true to say — no clearance, no root handed over, a clearance that
    names no path, or a clearance that names THIS root (presence, not sole-ness, for
    the reason spelled out in `_cross_root_seal_line`).

    The roots are the clearance's OWN sentence (`_clearance_measured_roots`, #3792), not
    every path token on the line: #3792's `cross-root clearance:` marker prints the root of
    the section it closed, which is a root the clearing process explicitly did not measure,
    and counting it would let a foreign seal read as honest to the fold simply because the
    marker also happens to print `measured_root`.
    """
    if not cleared or not measured_root:
        return []
    foreign: list[str] = []
    for root in _clearance_measured_roots(cleared):
        if root == measured_root or root.startswith(measured_root + "/"):
            return []
        if root not in foreign:
            foreign.append(root)
    return foreign


def _cross_root_line_of(section_text: str) -> str | None:
    """The cross-root label already inside one section's text, if it carries one.

    A refresh rewrites the section from `text` alone, so without carrying the line
    forward the label would vanish on the next check of the same incident and the note
    would stop saying the seal it is describing was foreign. #3628 clause 4's "once per
    section" is once AND kept.
    """
    for line in reversed(section_text.splitlines()):
        if line.strip().startswith(DAILY_CROSS_ROOT_PREFIX):
            return line.strip()
    return None


def _open_marker_span(text: str, body_start: int, window_end: int):
    """(body_start, end-of-marker-line) when `[body_start:window_end]` still is open.

    The LAST line inside the window that is `DAILY_STILL_OPEN` on its own is the
    section's own status line: `_vault_note` emits the body and then the marker, so
    whatever sits after it arrived from another writer, and a body quoting the string
    is not this incident's state. The end returned is just past that line, never the
    window's end, so a caller rewriting the returned range leaves the appended text
    standing where its author put it. None means the window holds no marker line,
    which is what a sealed — or a non-coalescing — section looks like.

    A module function, not a method: `_daily_open_at`'s only dependency stays
    `_daily_sections`, which is the one pair the read-only probes and test shims
    hand to each other.
    """
    offset = body_start
    found = -1
    for line in text[body_start:window_end].splitlines(keepends=True):
        if line.strip() == DAILY_STILL_OPEN:
            found = offset + len(line)
        offset += len(line)
    return (body_start, found) if found >= 0 else None

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
#: Header key naming the incident this file's one slot no longer holds (#3120).
#: `alert()` overwrites a single incident with one `write_text`, so an alert of
#: a different title necessarily buries whatever stood there — and until this
#: key existed it buried it without a trace: the 2026-10-20 `critical`
#: `Lloyd data root damaged — promotions halted` alarm left the file unsealed,
#: under a later `error` alert, taking the halt and its lift with it. The value
#: carries the buried alarm's own `level=` and `written=` INSIDE the key, never
#: as a second `written:` line (one readable stamp per file is
#: `tests/test_guardian_alert_stamp.py`'s rule), and ends `open` or `cleared`:
#: the two burials a reader has to be able to tell apart are an unsealed alarm
#: that went missing and a tombstone whose incident really did close.
ALERT_DISPLACED_PREFIX = "displaced:"
#: How much of a displaced tombstone's `cleared:` text the LEDGER row keeps
#: (#3120). It is `_ledger`'s own bound on an alert body — `body[:2000]` — so a
#: row about a displacement can never be wider than a row about an alert, and the
#: cap is a number this module already enforces rather than one invented for the
#: branch. Nothing about the file depends on it: the seal's text is already gone
#: from ALERT.md whatever its length, so keeping the head of a long retraction
#: note — the clause that says HOW the incident was closed — costs nothing the
#: alert path does not already pay.
_DISPLACED_NOTE_CHARS = 2000
#: A header key line of ALERT.md (`level: …`, `written: …`, `cleared: …`,
#: `displaced: …`). The retraction finds the file's title and its own marker
#: through this and nothing else, so an alert body can never impersonate a
#: header key — and a key this module adds is readable by every existing reader
#: the moment it is written, which is why it is spelled in this shape.
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
#: The route that moves a row's status (`app/routers/backlog.py:689`), posted by the
#: retraction in `resolve` (#3757). `voiceloss.py:96` already writes this route for the
#: voice-loss escalation; this is the same seam walked by the alerting families that
#: file through `_backlog_task`.
BACKLOG_UPDATE_PATH = "/api/backlog/task-update"
#: The only status the route accepts that is NOT open: `task-update` validates against
#: `_VALID_STATUSES = frozenset(PIPELINE_STATUSES)` (`app/routers/backlog.py:62`, i.e.
#: draft/up_next/in_progress/done) and `voiceloss.py:123` counts `draft`, `up_next` and
#: `in_progress` as OPEN, so `done` is the one answer. The consequence is that a guardian
#: all-clear is booked in the item's own `activity_log` as `"set from Mission Control"`
#: (`app/routers/backlog.py:757-759` → `app/backlog_move.py:130`), which is the wrong
#: attribution for a clear. The route takes no caller-supplied reason, and giving it one
#: is a separate ruling (#3757's owed check 2); this does not change the route.
BACKLOG_CLOSED_STATUS = "done"
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
#: How long the duplicate read may take. It was the POST's 5 s until 2026-10-29, and
#: that was too short for the read: `?q=` scans the whole board (3,500 files, 207 MB),
#: and with the backend's parse cache cold — after every landing's restart, or a pass
#: that rewrote many items — the same request measured 18.6 s against 0.3 s warm. The
#: client gave up, uvicorn wrote no access line for a disconnected client, `None`
#: meant file anyway, and one hourly shadow alert filed four copies (#3551, #3558,
#: #3559, #3566) while an open one stood on the board. 30 s covers the cold parse with
#: room and stays well inside the unit's `WatchdogSec=90`; the read runs only on a
#: tick that is already filing.
BACKLOG_READ_TIMEOUT_SECONDS = 30.0


class Notifier:
    def __init__(self, *, ledger: Path, state_dir: Path, vault_root: str,
                 backend_url: str = "http://127.0.0.1:8080",
                 external: bool = True, voice: bool = True,
                 voice_window: float = 3600.0):
        self.ledger = ledger
        self.state_dir = Path(state_dir)
        self.vault_root = Path(vault_root)
        self.backend_url = backend_url.rstrip("/")
        self._board_read_error: str | None = None
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
              needs_human: bool = False, coalesce: bool = False,
              measured_root: str = "") -> dict:
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

        `measured_root` is the absolute data root the CALLING check walked, and only the
        runtime-data check passes it (#3628). `_vault_note` reads it for one purpose: to
        notice that the newest `cleared:` line for this title names a DIFFERENT root,
        which means the section that line sealed was closed by a process that never
        measured this one. Every other caller leaves it empty and gets byte-for-byte the
        note it got before. A caller with no single root to name is not asked to invent
        one; the runtime-data check names `datawatch.DATA_ROOT`, because that is the
        root its own alert body and its own clearance already print.

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
        results["vault"] = self._vault_note(title, text, coalesce=coalesce,
                                            measured_root=measured_root)
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
        """Put this incident into ALERT.md's ONE slot, naming whatever it buries (#3120).

        One file, one incident, one `write_text`, last writer wins: that is
        #775's contract and this method still keeps it — refusing an overwrite
        would leave a stale alarm reading as live, which is the defect #1967
        closed. What it did NOT do before #3120 was anything about what the
        overwrite covered. There is no level comparison anywhere in this write
        path, so an `info` alert buries an open `critical` alarm exactly as the
        `error` watchdog alert did on 2026-10-21: the `critical`
        `Lloyd data root damaged — promotions halted` alarm left the file
        unsealed, un-retracted and unnamed, and with it the halt it asserted
        (raised 2026-10-20T21:24:14Z, adjudicated a false positive and lifted
        21:28:35Z) stopped existing on the surface an agent reads first. The
        retraction route cannot cover this: `_alert_file_retract` is
        title-scoped by #1967's own design and `guardian.py`'s tripwire sweep
        concedes it has no index to a section another incident displaced.

        So the outgoing incident is named BEFORE it is overwritten, in one
        `displaced:` header key carrying its own `level=` and `written=` INLINE
        — a second `written:` line would give the file two alarms a reader could
        age, which `tests/test_guardian_alert_stamp.py` exists to forbid — and
        ending `open` or `cleared`, because an unsealed alarm that went missing
        and a tombstone whose incident really did close are different facts. The
        burial also goes to the ledger as one `alert_displaced` row, which is
        where the whole chain survives (the in-file key is one generation deep by
        construction, and `promotions.jsonl` is archived monthly rather than
        deleted). It is NOT logged as another `alert`:
        `app/failure_ledger.py:367` builds per-title onset history out of those
        rows, and a displacement is not a firing.

        The key is conditional and the ledger row follows the write: a file with
        nothing to bury keeps the byte layout #1912 pinned byte for byte, and a
        write that failed records nothing it cannot stand behind. A predecessor
        no reader can attribute (`_alert_file_split` answers None) yields no
        trace either — inventing a title for bytes that are no one's incident is
        the same fabrication `_alert_file_retract` refuses. A re-alert of the
        incident already in the slot buries nobody and logs nothing, but CARRIES
        the key it is holding: that burial happened one check earlier, and the
        2026-10-21 displacement was itself produced by exactly such a re-alert, so
        dropping the key on a same-title write would erase a real burial's in-file
        trace within an hour of it — see `_alert_file_key`'s three answers, pinned
        by `test_the_burial_trace_survives_the_burying_incident_s_own_realert`.
        """
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            key_line, outgoing = self._alert_file_key(title)
            # `astimezone()` with no argument attaches the machine's zone to the
            # LOCAL reading: the digits, microseconds included, are untouched and
            # only the zone is named (#1912 — the same shape speak.py's `%z` got
            # from #1808). Writing this stamp naive on a box at -0700 left the
            # one artefact a human reads during an outage open to two readings,
            # and this repo's own comments held both of them seven hours apart:
            # `scripts/side_effect_traffic_census.py::parse_stamp` reads a naive
            # stamp as local, `app/skill_telemetry.py` reads one as UTC.
            written = datetime.now().astimezone().isoformat()
            # A header key, so it lands in the block `_alert_file_split` reads
            # before the first blank line — the only block `_alert_file_retract`
            # copies and the only one a reader's parser sees. Below the blank it
            # would be alarm prose again, and the first retraction would quote it.
            displaced = f"\n{key_line}" if key_line else ""
            (self.state_dir / ALERT_FILE_NAME).write_text(
                f"# {title}\n\nlevel: {level}\n"
                f"written: {written}{displaced}\n\n{text}\n",
                encoding="utf-8",
            )
        except Exception:
            return False
        if outgoing:
            self._alert_file_displaced(outgoing, title)
        return True

    def _alert_file_key(self, title: str) -> tuple[str, dict | None]:
        """(the `displaced:` line this write must carry, the incident it buries).

        Three answers, and the difference between the last two is the whole
        reason this is one call and not a boolean:

        `("", None)` — the slot holds nothing: no file, or bytes this format
        cannot name. Nothing is buried and there is no earlier trace to keep.

        `(existing, None)` — the SAME incident is re-alerting, which is the
        ordinary hourly shape of a `coalesce=True` watchdog and buries nobody, so
        nothing is logged. But the key the file already carries is carried
        FORWARD: the burial it records happened one check earlier, and the
        displacement this item is about was itself produced by exactly such a
        re-alert — an unsealed `critical` alarm buried by the second hourly
        `error` copy of one standing incident. Dropping the line whenever the
        incident re-asserts would erase the in-file trace of a real burial within
        one check of it, which is the half the ledger is not for.

        `(new_line, outgoing)` — a DIFFERENT title holds the slot, so this write
        buries it. The key is REWRITTEN to name this burial — the file carries
        one generation, the chain lives in the ledger — and `outgoing` is what the
        caller logs.

        Read through `_alert_file_split`, the parser a reader uses, so a
        predecessor written by another process — or edited by a person — yields
        the same fields; the H1 minus its `"# "` marker is `alert_file_incident`'s
        own extraction, so "same incident?" cannot be answered two ways by the two
        readers. `level` and `written` fall back to `unknown` rather than being
        dropped: a burial with no stamp is still a burial, and a missing field
        would read as "nothing was here".
        """
        try:
            raw = (self.state_dir / ALERT_FILE_NAME).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return "", None
        split = self._alert_file_split(raw)
        if not split:
            return "", None
        carried, level, written, cleared = "", "unknown", "unknown", ""
        for key in split[1]:
            if key.startswith(f"{ALERT_DISPLACED_PREFIX} "):
                carried = key
            elif key.startswith("level: "):
                level = key[len("level: "):].strip() or "unknown"
            elif key.startswith("written: "):
                written = key[len("written: "):].strip() or "unknown"
            elif key.startswith(f"{ALERT_CLEARED_PREFIX} "):
                cleared = key[len(ALERT_CLEARED_PREFIX):].strip()
        held = split[0][len("# "):]
        if held == title:
            return carried, None
        buried = {"title": held, "level": level, "written": written,
                  "cleared": cleared, "sealed": bool(cleared)}
        return (f"{ALERT_DISPLACED_PREFIX} {held} level={level} "
                f"written={written} {'cleared' if cleared else 'open'}", buried)

    def _alert_file_displaced(self, outgoing: dict, incoming_title: str) -> None:
        """Record the burial durably: the in-file key is overwritten by the next one.

        One `alert_displaced` row per displacement, never another `alert` row —
        see `_alert_file`'s reason. `written` is the BURIED alarm's own stamp and
        `created_at`/`ts` are when the burial happened, so the row answers both
        questions a later reader asks: how old was the alarm that vanished, and
        when did it vanish. `cleared` carries the seal's text when the buried
        incident had been closed, because that text lives nowhere else — the
        `alert` row logged when it fired holds its own body, but the retraction
        note is written only into ALERT.md, which is exactly the line this
        overwrite destroys.

        Best-effort in both directions, on the rule `_backlog_task` states for
        this class of helper: never raise from an alert, and never RE-ALERT about
        an alert — a displacement that could not be logged is reported by the
        `displaced:` key in the file, which is the primary trace and needs no
        ledger to be read. The lazy `import gstate` is `_ledger`'s, for `_ledger`'s
        reason: this module's callers reach the ledger through it, and the drill
        stubs that name.
        """
        try:
            entry = {"event": "alert_displaced",
                     "level": outgoing["level"], "title": outgoing["title"],
                     "written": outgoing["written"],
                     "sealed": outgoing["sealed"],
                     "displaced_by": incoming_title}
            if outgoing["sealed"]:
                entry["cleared"] = outgoing["cleared"][:_DISPLACED_NOTE_CHARS]
            import gstate
            gstate.append_event(self.ledger, entry)
        except Exception:
            pass

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

    def alert_file_incident(self) -> tuple[str, bool] | None:
        """`(title, sealed)` for the incident ALERT.md is holding, or None (#3019).

        The read side of the single-slot file `_alert_file` writes and
        `_alert_file_retract` seals. The retraction has always been able to answer
        "is this file still mine to seal" for the title it was handed, and nothing
        could answer the prior question — WHICH incident is this file asserting —
        so a process that did not write it had no way to ask. That question is what
        a restart reconciliation starts from: ALERT.md outlives its writer, and the
        latch every guard retracts on does not.

        Same parse and the same rule as the retraction, so the two cannot disagree
        about what the file says: only a `^[a-z_]+: ` line is a header key, so alarm
        prose that opens with a `key: value`-looking sentence can neither extend the
        header nor pass itself off as a `cleared:` marker.

        Three shapes all answer None — no file, a file that cannot be read, and a
        file that is not this layout — because an incident whose title cannot be
        read is one no caller may act on. The distinction between "no incident" and
        "an unreadable one" is deliberately not made here: both leave a caller with
        nothing it is allowed to seal, and `_alert_file_retract` already treats the
        missing file as success rather than as a problem to report.
        """
        path = self.state_dir / ALERT_FILE_NAME
        try:
            if not path.is_file():
                return None
            body = path.read_text(encoding="utf-8")
        except OSError:
            return None
        split = self._alert_file_split(body)
        if split is None:
            return None
        _, keys, _ = split
        sealed = any(k.startswith(f"{ALERT_CLEARED_PREFIX} ") or k == ALERT_CLEARED_PREFIX
                     for k in keys)
        return split[0][len("# "):], sealed

    def _vault_note(self, title: str, text: str, *, coalesce: bool = False,
                    measured_root: str = "") -> bool:
        """Write this alert's section into today's daily note.

        `coalesce=False` — every alert but the runtime-data stray one — appends a
        section per fan-out, byte-identical to the pre-#1536 format, UNTIL this same
        title has been written to this day's note once already: the second and later
        firings of a title a day has already carried are FOLDED to one line
        (`_fold_owed`, #3793). `coalesce=True` opens a section on the first finding and
        REFRESHES that same section on every later finding of the same incident, so one
        incident is one section no matter how many checks it outlives — and its append
        branch, which an honest seal or a foreign one reaches, folds from the second
        foreign-seal re-paste onward (see `_fold_owed` for why the first one does not).

        The fold is a visible collapse, never a silence: every firing from the second
        onward writes a line naming the title, how many of today's firings it folds,
        and the newest `checked` stamp, and the line updates IN PLACE so one title owns
        one fold line a day however long the incident runs. It is keyed on the heading
        already in the note, not on body bytes, because the 2026-10-29 repeats whose
        bytes the item measured were never byte-identical — the store's byte count and
        the sealing worktree's path both move every hour — so a body hash would have
        folded 0 of the 5 copies on that page.

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
                # A section that already carries the cross-root label keeps it: the
                # refresh rewrites the section from `text` alone, so dropping the line
                # here would un-say, on the second hour of an incident, the one thing
                # that explains why the section above it closed (#3628 clause 4). It is
                # CARRIED and never recomputed — an open section has no new clearance to
                # describe, so a refresh cannot add a label that was not there.
                carry = _cross_root_line_of(stale)
                refreshed = text.rstrip()
                if carry:
                    refreshed += f"\n\n{carry}"
                body = (body[:start] + lead + refreshed
                        + "\n\n" + DAILY_STILL_OPEN + "\n" + body[end:])
                note.write_text(body, encoding="utf-8")
                return True
            # The append branch is where a day's note gains a body, and it is the one
            # place on this path that can see the whole note. Both doors into it re-paste
            # an unchanged alarm: `coalesce=False` never looks for an open section at all
            # (the 2026-11-04 witness — 3 sections, the first two byte-identical), and a
            # coalesced title whose section was sealed is by definition not open (the
            # 2026-10-29 witness — 5 sections, 87.8 % of the file). One predicate covers
            # both, and it reads the note rather than process state, because those five
            # sections came from five separate processes.
            if self._fold_owed(body, title, coalesce=coalesce,
                               measured_root=measured_root):
                return self._write_daily_fold(note, body, title, text, measured_root)
            with open(note, "a", encoding="utf-8") as f:
                if coalesce:
                    # Appending a fresh section is the one moment this writer can see the
                    # seal that made it necessary, so it is the moment the label goes in
                    # (#3628). Written as its own `f.write` and NOT folded into the line
                    # above: with nothing to say the bytes below must be exactly what
                    # this method wrote before the label existed, clause 2's
                    # byte-identical case.
                    seal = self._cross_root_seal_line(body, title, measured_root)
                    f.write(f"\n\n## Self-mod guardian: {title}\n\n{text.rstrip()}\n")
                    if seal:
                        f.write(f"\n{seal}\n")
                    f.write(f"\n{DAILY_STILL_OPEN}\n")
                else:
                    f.write(f"\n\n## Self-mod guardian: {title}\n\n{text}\n")
            return True
        except Exception:
            return False

    def _cross_root_clear_marker(self, section_text: str, note_text: str,
                                 measured_root: str) -> str:
        """The suffix for a `cleared:` line whose section measured a root it does not.

        "" — the old bytes — unless BOTH sides have a root to complain about:

        * the section's body names no absolute path the clearance fails to name. That is
          the honest hourly all-clear by the live guardian, and clause 2 of #3792 is that
          it stay byte-identical: a marker on an honest seal would make the marker mean
          nothing;
        * the clearance has no root of its own to name — neither a `measured_root` from its
          producer nor a path in its own sentence. Then there is nothing true to say about
          where it stood.

        Otherwise the suffix names the root this seal measured and the root the section
        measured, and NOTHING ELSE: the code tree the clearance walks as plainly as its
        data root stays out, which is the half #3628's review advisories never pinned (the
        two nodes it left behind assert both roots are in its label and never what must be
        out of it). Every path here is one the clearance or the section already printed, so
        the marker adds no claim — it only puts the two claims that were already on the
        page next to each other, in the one line a reader reaches first.

        `measured_root` is preferred over sweeping the clearance's path tokens because a
        shipped clearance prints two of them and only one was measured.
        """
        section_roots_all = _section_measured_roots(section_text)
        clearance_roots = _absolute_roots(note_text)
        handed = [measured_root] if measured_root else []
        section_roots = _roots_outside(section_roots_all, clearance_roots + handed)
        if not section_roots:
            return ""
        seal_roots = _roots_outside(handed or clearance_roots, section_roots_all)
        if not seal_roots:
            return ""
        return (
            f" [{DAILY_CROSS_ROOT_CLEAR_PREFIX} "
            f"{self._stamp().strftime('%Y-%m-%dT%H:%M:%SZ')} this all-clear measured "
            f"{' or '.join(seal_roots)}; the section it is closing was measured against "
            f"{' or '.join(section_roots)}. The section is closed, but this line is not a "
            "statement about that second root: the process that wrote it stands in the "
            "first.]")

    def _fold_owed(self, text: str, title: str, *, coalesce: bool,
                   measured_root: str = "") -> bool:
        """Must this firing be a line instead of a body?

        Matched on the HEADING, not the body: `_daily_sections` answers "has today's
        note already carried a section under this title", which is the question the two
        live witnesses actually ask. `memory/2026-11-04.md` holds three sections of one
        title whose first two are byte-identical, and `memory/2026-10-29.md` holds five
        whose bodies are NOT — the store's byte count moves (172,982,272 → 172,990,464 →
        173,170,688) and so does the sealing worktree named in each `cleared:` line — so
        anything keyed on normalised body bytes would have folded 0 of those 5 copies
        while claiming to be a dedupe. A title is the only index a daily note has.

        `coalesce=False` folds from the second firing of a title onward, and that is
        safe because a non-coalescing title has no incident lifecycle to lose: its
        sections carry no `_(still open on the next check)_` marker, so `resolve` never
        seals them and there is no honest all-clear whose "new incident" this could
        hide. #1536 clause 5 pinned the old two-sections-per-firing shape on exactly
        this door (`tests/test_guardian_predicates.py`) in the name of not generalising
        the coalescing fix to every alert; #3793 is the deliberate generalisation, and
        the hazard that clause named — a recurring notice going unread — is answered by
        the line arriving on every firing with the count and the newest stamp on it.

        `coalesce=True` folds only once the note has SAID once that a foreign root
        sealed this title AND that seal still stands — the newest section carries a
        `cross-root seal:` label and the newest `cleared:` line for the title still
        names a root `measured_root` cannot account for. A label is permanent on the
        section that carries it, so keyed on the label alone an honest all-clear
        followed by a genuinely new incident would be folded away and the note would
        carry no open section for a live condition; the clearance test is what ends the
        fold. The firings that get a full body are the two that are
        not repetitions: an honest all-clear followed by a genuinely new incident
        (#3150 clause 4: "coalescing may not turn a title into a mute"), and the first
        re-paste after a seal written by a process that never measured this root
        (#3628 clause 3, whose node is this item's own clause 5 — `test_a_shadow_
        finding_sealed_from_another_root_re_alerts_with_the_label`, which is the shape
        of 2026-10-29 and is why the label keeps its section). From the NEXT foreign
        seal onward the note is re-printing a sentence it has already printed, and that
        is the part that was 87.8 % of the page.
        """
        if not any(sect[0] == title for sect in self._daily_sections(text)):
            return False
        if coalesce:
            newest = [s for s in self._daily_sections(text) if s[0] == title][-1]
            if _cross_root_line_of(text[newest[1]:newest[2]]) is None:
                return False
            # A label is a permanent fact about the section that carries it, so it cannot
            # by itself say what happened to the title NEXT. An honest all-clear that
            # arrives after a labelled section closes the title for real, and the next
            # firing is a genuinely new incident which must get its own body — folding it
            # would leave the note with no open section and a condition nobody printed,
            # the exact unread-alarm failure this item exists to remove. So the fold binds
            # only while the seal still stands: the newest `cleared:` line for this title
            # must still be one this check cannot account for. Honest (or no clearance at
            # all) → the label is superseded → full body. With no `measured_root` handed
            # over nothing can be proven foreign, and an unprovable fold is a silence, so
            # this side stays open.
            if not measured_root:
                return False
            return bool(_foreign_roots(self._newest_cleared_line(text, title),
                                       measured_root))
        return True

    def _daily_fold_span(self, text: str, title: str):
        """(start, end, fires) of this title's fold line in `text`, else None.

        Read back off the note, which is the only state that survives the 2026-10-29
        shape: one process per hourly tick, each of them a different automod-round
        worktree, so an in-memory counter would restart at 1 fifty times a day and the
        number on the page would be a lie. The scan is line-anchored on
        `DAILY_FOLD_HEAD`, so a body that merely mentions a folded alarm — the
        scheduler's `[daily-note fold: …]` line, which sits in these very notes — is not
        mistaken for this writer's: only a line that opens with this module's own head
        and names this title exactly is one.
        """
        offset = 0
        for line in text.splitlines(keepends=True):
            bare = line.rstrip("\n")
            if bare.startswith(DAILY_FOLD_HEAD):
                subject, sep, tail = bare[len(DAILY_FOLD_HEAD):].partition('": ')
                if sep and subject == title:
                    found = DAILY_FOLD_FIRES.match(tail)
                    return offset, offset + len(line), int(found.group(1)) if found else 0
            offset += len(line)
        return None

    def _newest_stamp_in(self, text: str) -> str:
        """The newest ISO-8601 instant the incoming body carries, or this instant.

        `checked <ISO>` is the line the fold discards, so the line has to carry a
        stamp or the note loses the only thing that says the alarm is still live. The
        body's own reading wins over the writer's clock: a check that re-printed a
        finding measured an hour ago should fold under that hour, not under the tick
        that got around to writing it. A body that names no instant at all falls back
        to `_stamp()`, which is what every other dated line on this note is stamped
        with.
        """
        found = ISO_STAMP_IN_TEXT.findall(text)
        if found:
            return max(found)
        return self._stamp().strftime("%Y-%m-%dT%H:%M:%SZ")

    def _fold_line(self, title: str, fires: int, stamp: str, foreign: str) -> str:
        """The one line that stands in for `fires` folded firings of `title`.

        Shape is the scheduler's (`app/discord_notify.py:346-349`), because that is the
        clause every guardian alarm line in the two witness notes already quotes:
        the title, how many of today's fires it folds, the newest reading, and what
        would make the next full body arrive. `foreign` is named only when a
        `cross-root seal:` is what ended the section above — the same fact #3628's label
        exists to say, kept to its root here because the sentence is already on the page
        once and this line is not the place to print it twice.
        """
        plural = "s" if fires != 1 else ""
        clauses = [f'{DAILY_FOLD_HEAD}{title}": {fires} earlier fire{plural} today '
                   "folded into this line instead of a second full body",
                   f"newest check {stamp}"]
        if foreign:
            clauses.append("the section above was sealed by a "
                           f"`{DAILY_CROSS_ROOT_PREFIX}` naming {foreign}")
        clauses.append(DAILY_FOLD_TAIL)
        return "; ".join(clauses)

    def _write_daily_fold(self, note, body: str, title: str, text: str,
                          measured_root: str) -> bool:
        """Write the fold `_fold_owed` called for, and return what `_vault_note` returns.

        One fold line per title per day, not one per firing: an existing line is
        rewritten in place with `fires + 1` and the new stamp, so an incident that runs
        forty hours costs one line and not forty. The bytes before the line and after it
        are untouched, which is the same prefix/suffix discipline the refresh branch
        keeps for the other writers of this note.
        """
        span = self._daily_fold_span(body, title)
        fires = (span[2] if span else 0) + 1
        foreign = _foreign_roots(self._newest_cleared_line(body, title), measured_root)
        line = self._fold_line(title, fires, self._newest_stamp_in(text),
                               foreign[-1] if foreign else "")
        if span is None:
            with open(note, "a", encoding="utf-8") as f:
                f.write(f"\n{line}\n")
            return True
        start, end, _old = span
        note.write_text(body[:start] + line + body[end:], encoding="utf-8")
        return True

    def resolve(self, title: str, note_text: str,
                *, measured_root: str = "") -> bool:
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
        section rather than in a file they were never sent to. The loop below runs
        until `_daily_open_at` answers None, so EVERY open section for `title` in the
        note is sealed, not just the newest one: a day whose note carries several
        copies of one incident — which is what the pre-#3150 openness test produced
        whenever another writer appended after the marker — used to keep its older
        markers standing below the single `cleared:` line, leaving the only surviving
        record of a finished incident an instruction to do work. It writes nothing when
        no incident is open for `title`, which is what makes the second, third and
        hundredth all-clear check silent. What a clear does NOT do is make noise: no
        toast and no speech, and no NEW backlog task, because a clear is not an alert —
        and the ledger still holds every individual firing either way. It does, however,
        reach the backlog row that this title's alert filed (#3757): see
        `_retract_board_item` below the `external` gate, which takes that row out of the
        OPEN set while it still silences the next filing. The sentence this replaces
        read "no other channel is touched… no backlog task", and that is the behaviour
        #3755 was written by: ALERT.md sealed at 09:18:50Z over a `draft` board item the
        same alert created at 09:18:40Z.

        `measured_root` is the runtime-data root THIS process walked, handed over by the
        one check that has a root to name (#3792). It changes nothing about what gets
        sealed and nothing about the return value: its whole effect is that a `cleared:`
        line may say which root it speaks for, in its own text, when that root is not the
        one the section it is closing was measured against. A caller with no root to name
        passes nothing and still gets the marker when its own sentence names a root the
        section's body does not — the check needs no new state, only the two sentences that
        are already on the page.

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
        # Was there anything HERE for this call to seal? Asked before the seal, from the
        # same parse the retraction uses (`alert_file_incident`, #3019), because the
        # board read below is the `?q=` scan #3551 measured at 18.6 s cold on 3,693
        # rows (against the 5 s ceiling then in force; the ceiling is
        # `BACKLOG_READ_TIMEOUT_SECONDS = 30.0` now, so a read that slow now COMPLETES
        # — which is exactly why it must not be scheduled). The ordinary hourly
        # all-clear has nothing to retract and must not be put on a timer that pays it. ALERT.md's slot is one small file in the
        # state dir and it is already read twice by the retraction below; the 30 s cost
        # is a board question, so that is the only thing this flag gates.
        held = self.alert_file_incident()
        alert_seals = bool(held) and held[0] == title and not held[1]
        retracted = self._alert_file_retract(title, note_text)
        if not self.external:
            return retracted
        # Checked before sealing, from the filenames: the out-of-window question is "is
        # there an alarm for this title my reach does not cover", and a section sealed a
        # moment ago must not be able to mask one it cannot touch.
        out_of_reach = bool(self._open_days_beyond_window(title))
        wrote = True
        # Did any daily note actually gain a `cleared:` line on this call? The other
        # half of the seal test above, and the half that carries most real service-down
        # incidents: ALERT.md is ONE last-writer-wins slot, so a second alert displaces
        # the first and the recovery tick finds this title's section open in the note
        # with some other title's alarm in the file.
        sealed_a_note = False
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
                # Asked PER SECTION, from that section's own bytes, because a note whose
                # copies name different roots has one honest seal and one foreign one in
                # the same call (#3150's shape), and because recomputing it here is what
                # lets clause 2's honest case be a byte-identical no-op rather than a
                # special branch.
                marker = self._cross_root_clear_marker(stale, note_text, measured_root)
                sealed = (sealed[:start] + stale
                          + f"\n\n{DAILY_CLEARED_PREFIX} {note_text}{marker}\n"
                          + sealed[end:])
            if sealed == body:
                continue
            try:
                note.write_text(sealed, encoding="utf-8")
            except OSError:
                wrote = False
            else:
                sealed_a_note = True
        # ── the board half of the retraction (#3757) ──────────────────────────
        # Placed here, BELOW the `external` gate and after the seal is known to have
        # happened, so it inherits the drill's board-silence for free and pays the
        # board read only on a tick that actually closed an incident.
        if alert_seals or sealed_a_note:
            self._retract_board_item(title)
        return False if (out_of_reach or not retracted) else wrote

    def _retract_board_item(self, title: str) -> int:
        """Take the board row this title's alert filed out of the OPEN set (#3757).

        Filing and retraction were one story until now, told by one channel.
        `alert` routes a needs-human alert to `_backlog_task`, which posts
        `[guardian] <title>` at `status: draft`/`priority: high` on `lloyd`
        (`_backlog_task`'s payload: `"status": "draft"`, `"priority": "high"`, `"board": "lloyd"`),
        and while a row of that EXACT name stands OPEN the
        next firing is suppressed outright (`_open_duplicate_on_board`) — `draft` is
        in `voiceloss.OPEN_STATUSES` (`voiceloss.py:123`), so a self-clearing incident
        both leaves a `priority: high` item asserting a down service AND silences the
        filing of the next, genuine, outage of the same family. The live witness at
        filing: ALERT.md carries
        `cleared: 2026-11-04T01:18:50.338811-08:00 — liveness read came back healthy:
        lloyd-mc:lloyd-backend.` over the item #3755 that the same alert filed 10
        seconds earlier (`created: '2026-11-04T09:18:40.750499'`, `status: draft`),
        with nothing on the clearance path able to reach it —
        `guardian.py:2758-2775` and the boot reconcile `_seal_measured_incident`
        (`guardian.py:3014`) call `resolve` and nothing else.

        The row is closed by its `id` and nothing else. `GET /api/backlog/tasks`
        carries `description_snippet` and no `description`
        (`app/routers/backlog.py:421-453`), and `task-update` replaces the WHOLE body
        and silently drops a shorter one unless `force_body_replace` is set
        (`app/routers/backlog.py:732-733`), so a body rebuilt from a list row would
        truncate the item's own evidence — which is the opposite of what a retraction
        is for. `voiceloss.py:296` is the only other writer of this route in the
        guardian, and it is the precedent for posting through the route over loopback
        rather than touching the backlog directory from this process.

        Every exact-name OPEN row is closed, not just the newest: #2080's four copies
        (#3551, #3558, #3559, #3566) are the shape this guard exists for, and leaving
        three of four open would keep silencing the next real filing, which is the
        half of the harm this method exists to remove. A row that merely QUOTES this
        alert in its body comes back from the `?q=` read too and is left alone —
        the exact-name rule is #2080's, reaffirmed for this family by #3229's
        owed-check on 2026-10-27 ("I will not loosen #2080's exact-name rule"), and
        this method borrows the test rather than loosening it.

        An unreadable board, an unimportable status vocabulary, or a row with no usable
        `id` all answer 0 and post nothing. That is the fail-closed direction for a
        write: `_open_duplicate_on_board` fails OPEN (files) because its worst case is a
        duplicate, while this one's worst case is closing an item a human still owns,
        and a stale row is the pre-existing condition rather than a new harm. The count
        is not folded into `resolve`'s return, which stays a claim about the two note
        surfaces it seals — the same reason `_backlog_task`'s bool has no reader.
        """
        name = f"[guardian] {title}"[:120]
        try:
            import voiceloss
        except Exception:
            return 0
        self._board_read_error = None
        rows = self._board_rows(name)
        if not rows:                      # None (unreadable) or [] (nothing matching)
            return 0
        closed = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("name") or "") != name:
                continue
            if voiceloss.canonical_status(row.get("status")) not in voiceloss.OPEN_STATUSES:
                continue
            row_id = row.get("id")
            if not isinstance(row_id, int) or isinstance(row_id, bool) or row_id <= 0:
                continue                  # no addressable row: never post `id: null`
            if self._post_board_close(row_id):
                closed += 1
        return closed

    def _post_board_close(self, row_id: int) -> bool:
        """Move one board row out of the OPEN set by id, with no body (#3757).

        `id` and `status` and nothing else: a `description` key would replace the
        item's whole body `app/routers/backlog.py:719-733`, and the list read this
        close is keyed from carries no body to post back. `status` is
        `BACKLOG_CLOSED_STATUS`, the one value outside `voiceloss.OPEN_STATUSES` that
        `_VALID_STATUSES` accepts.
        """
        # Named `close_body` rather than `payload`, for a reason that is not style:
        # `tests/test_guardian_predicates.py::test_the_needs_human_route_posts_the_payload_the_board_reads`
        # cuts `_backlog_task`'s filing block out of THIS file by cutting the source at
        # the first assignment of a json-dumped value to the name `payload`, and taking
        # everything between the filing's own field-name comment and there. One of those
        # written earlier in the file makes that slice empty and the filing node goes red
        # with nothing wrong with the filing — so this method's request bytes get a
        # different name, and this comment deliberately does not spell the pattern out.
        close_body = json.dumps({"id": row_id,
                                 "status": BACKLOG_CLOSED_STATUS}).encode()
        req = urllib.request.Request(
            f"{self.backend_url}{BACKLOG_UPDATE_PATH}",
            data=close_body, headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if not (200 <= resp.status < 300):
                    return False
                reply = json.loads(resp.read().decode("utf-8", "replace") or "{}")
        except (urllib.error.URLError, OSError, ValueError):
            return False
        # The route's own success flag, for the reason `_backlog_task` gives at its
        # create: a 2xx is not evidence the row moved. `backlog_task_update` answers
        # `{"success": true, "description_ignored": …}` (app/routers/backlog.py:811).
        return isinstance(reply, dict) and reply.get("success") is True

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
        """(body_start, marker_end) of `title`'s newest OPEN section, else None.

        Open means `DAILY_STILL_OPEN` still stands on a line of its own inside the
        section's OWN window — the span `_daily_sections` cut for it, which runs to
        the next `## Self-mod guardian:` heading, or to EOF for the last one. It does
        not have to be the last thing IN that window. Other writers reach the same
        daily note — session capture's `### Session … Auto-captured` block, and
        `app/discord_notify.py`'s folded one-line alarm — and an incident whose
        marker such an append walked past is still the same open incident.

        Requiring the marker at the very end of the window is what made a day's note
        re-paste one incident's section four times (#3150, `memory/2026-10-21.md` and
        `memory/2026-10-16.md`): the appended line moved the end, this probe answered
        None, and `_vault_note` took its append branch and pasted the whole body
        again — every check, on a note it had just been told is not empty.

        The end returned is the end of the marker LINE, not of the window, so a
        caller rewriting `text[start:end]` rewrites only the section's own text and
        every byte another writer left after the marker stays in `text[end:]`.

        Newest OPEN match, an older closed section never stopping the walk: after an
        incident was cleared and a second one opened, the closed one is a historical
        record and the newer open one is the live incident; and `resolve` calls this
        in a loop until it answers None, so returning early on a closed newest
        section left every older standing marker in the file unsealed — three
        `_(still open on the next check)_` lines below the `cleared:` line that
        supposedly ended the incident in `memory/2026-10-21.md`.
        """
        for sect in reversed(self._daily_sections(text)):
            if sect[0] != title:
                continue
            span = _open_marker_span(text, sect[1], sect[2])
            if span is not None:
                return span
        return None

    def _newest_cleared_line(self, text: str, title: str) -> str | None:
        """The most recent `cleared:` line belonging to `title`, else None (#3628).

        Section-scoped and title-scoped for the same reason `_daily_open_at` is: a day's
        note carries many incidents' sections under several titles, and #1536's heading is
        the only index it has. Sections are walked newest-first and the first section that
        has a clearance wins, because a clearance in an older section cannot be the newest
        fact about a title that has since alerted again.

        A `dated:` line is deliberately NOT a match. It records that a later read refutes
        the state a body quoted, not that a check watched the condition come back, and what
        this answers is a question about a MEASUREMENT — so only the word a measurement
        writes can carry its root.
        """
        for sect in reversed(self._daily_sections(text)):
            if sect[0] != title:
                continue
            for line in reversed(text[sect[1]:sect[2]].splitlines()):
                if line.startswith(DAILY_CLEARED_PREFIX):
                    return line
        return None

    def _cross_root_seal_line(self, text: str, title: str,
                              measured_root: str) -> str | None:
        """The label for an append whose newest clearance names ANOTHER root (#3628).

        None — silence, and the old bytes — whenever there is nothing true to say:

        * no `measured_root` was handed over. Only the runtime-data check has a root to
          name, and every other alert in the file must keep writing what it wrote before;
        * the title has no prior `cleared:` line. This is the ordinary first incident, and
          the ordinary reopen after an honest one takes the third branch below;
        * that clearance names no absolute path at all. Nothing can be compared, so this
          method claims nothing — which is also why a clearance written without a root
          (the arrivals leg's line, for a title of its own) cannot read as a foreign seal;
        * the measured root IS in it. Presence is the test, not sole-ness: the honest
          runtime-data clearance names the code tree as well as the data root, and a rule
          that demanded one path would label every honest seal on the box.

        Otherwise the line names the roots that clearance carries and this check does not
        share, plus this check's own, and is stamped with `_stamp()` like every other
        timestamp that reaches the note. Detection only: the ruling on whether a
        cross-root process may seal at all is owed elsewhere, and if it comes back
        root-keyed this line becomes a backstop rather than the fix.
        """
        foreign = _foreign_roots(self._newest_cleared_line(text, title), measured_root)
        if not foreign:
            return None
        return (
            f"{DAILY_CROSS_ROOT_PREFIX} "
            f"{self._stamp().strftime('%Y-%m-%dT%H:%M:%SZ')} the most recent "
            f"`{DAILY_CLEARED_PREFIX}` line for this title names "
            f"{' or '.join(foreign)}, which this check did not measure; this check "
            f"measured {measured_root}. A process on that other root closed the section "
            "above, so the finding below is re-printed because a foreign seal ended it, "
            "not because anything here cleared.")

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
        self._board_read_error = None
        if self._open_duplicate_on_board(item_name):
            self._note_backlog_skip(item_name)
            return False
        if self._board_read_error:
            self._note_backlog_skip(item_name, unread=self._board_read_error)
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
        # Why the read failed is kept on the instance for `_backlog_task` to record:
        # an unreadable board and an empty one both end in a filing, and only this
        # tells them apart afterwards (#3559).
        try:
            with urllib.request.urlopen(req, timeout=BACKLOG_READ_TIMEOUT_SECONDS) as resp:
                if not (200 <= resp.status < 300):
                    self._board_read_error = f"HTTP {resp.status}"
                    return None
                rows = json.loads(resp.read().decode("utf-8", "replace") or "null")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            self._board_read_error = f"{type(exc).__name__}: {exc}"[:200]
            return None
        if not isinstance(rows, list):
            self._board_read_error = f"body is {type(rows).__name__}, not a list"
            return None
        return rows

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

    def _note_backlog_skip(self, name: str, *, unread: str | None = None) -> None:
        """Record that a filing for `name` was suppressed, in the guardian's state dir.

        With `unread`, records the opposite: the board could not be read, so the
        filing went ahead without the duplicate check. Kept under its own key
        (`unread`, with the failure's text) in the same file, so a duplicate on the
        board can be traced to a failed read rather than to an empty board.

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
            if unread is not None:
                misses = data.get("unread")
                misses = misses if isinstance(misses, dict) else {}
                prior = misses.get(name)
                prior = prior if isinstance(prior, dict) else {}
                try:
                    count = int(prior.get("count") or 0)
                except (TypeError, ValueError):
                    count = 0
                misses[name] = {"count": count + 1, "last_seen": gstate.now_iso(),
                                "last_reason": unread}
                data.update({"schema": 1, "unread": misses})
                gstate.write_json_atomic(path, data)
                return
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
