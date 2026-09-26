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
import subprocess
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

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
        at 90s. Whether the sound actually came out is in voice.log.

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
            (self.state_dir / "ALERT.md").write_text(
                f"# {title}\n\nlevel: {level}\nwritten: {datetime.now().isoformat()}\n\n{text}\n",
                encoding="utf-8",
            )
            return True
        except Exception:
            return False

    def _vault_note(self, title: str, text: str, *, coalesce: bool = False) -> bool:
        """Write this alert's section into today's daily note.

        `coalesce=False` — every alert but the runtime-data stray one — appends a
        section per fan-out, byte-identical to the pre-#1536 format. `coalesce=True`
        opens a section on the first finding and REFRESHES that same section on every
        later finding of the same incident, so one incident is one section no matter
        how many checks it outlives.
        """
        try:
            note = self._daily_note()
            if note is None:
                return False
            body = note.read_text(encoding="utf-8") if note.exists() else ""
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

        Returns True when the note is in a cleared state for `title` (sealed now, or
        nothing was open), False on failure.
        """
        if not self.external:
            return True
        try:
            note = self._daily_note()
            if note is None or not note.exists():
                return True
            body = note.read_text(encoding="utf-8")
            open_at = self._daily_open_at(body, title)
            if open_at is None:
                return True
            start, end = open_at
            stale = body[start:end].rstrip()
            # The marker is REPLACED, not followed: leaving it standing above the
            # `cleared:` line means the note still claims the incident is open to
            # anyone scanning for that marker, which is the one question a
            # retraction exists to change the answer to.
            if stale.endswith(DAILY_STILL_OPEN):
                stale = stale[:-len(DAILY_STILL_OPEN)].rstrip()
            note.write_text(body[:start]
                            + stale
                            + f"\n\n{DAILY_CLEARED_PREFIX} {note_text}\n"
                            + body[end:], encoding="utf-8")
            return True
        except Exception:
            return False

    # ── daily-note incident plumbing (#1536) ──────────────────────────────
    #
    # Incident state is read back OUT of the note, never held in this process, so a
    # guardian restart mid-incident refreshes the existing section instead of opening
    # a second one: the incident is a fact about the note, not about this runtime.

    def _daily_note(self):
        """Today's note, or None when the vault has no `memory/` to write into."""
        memory = self.vault_root / "memory"
        if not memory.is_dir():
            return None
        return memory / f"{datetime.now().strftime('%Y-%m-%d')}.md"

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
        """File a backlog item so the revert becomes work, not a mystery."""
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
        payload = json.dumps({
            "name": f"[guardian] {title}"[:120],
            "description": body[:6000],
            "status": "up_next",
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
                # A 2xx with a defaulted name means the payload contract drifted.
                name = str(created.get("name") or created.get("task", {}).get("name") or "")
                return "guardian" in name.lower() if name else True
        except (urllib.error.URLError, OSError, ValueError):
            return False
