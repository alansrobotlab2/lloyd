#!/usr/bin/env python3
"""Lloyd self-modification guardian.

Watches the backend and the MCP aggregator and, when a promoted commit breaks
them, restores the last known good tree and restarts the stack. Runs as a
systemd --user unit rather than a supervisord program, for three reasons:

  * `agent-supervisord.service` sets `KillMode=control-group`, so every
    supervisord child dies when that unit restarts or supervisord crashes —
    exactly the scenario a watchdog exists for.
  * Its remediation set includes "restart supervisord". A child cannot restart
    its own supervisor and survive.
  * supervisord parks a program in FATAL after `startretries` and never
    un-parks it. systemd `Restart=always` with `StartLimitIntervalSec=0` never
    gives up. A watchdog that can permanently give up is not a watchdog.

Stdlib only, run from `/usr/bin/python3` (not the venv), from a **pinned
snapshot** outside the repo — so a `uv pip install` that wrecks the venv, or a
SyntaxError Lloyd writes into this file, cannot stop the running guardian.
See `agent-services/bin/guardian-stage.sh`.

Three invariants the code below enforces and the tests assert:

  1. **`HEAD == last_known_good` ⇒ never roll back.** Everything on fire with
     nothing promoted is infrastructure, not a bad change. Rewriting history
     there would destroy a working tree.
  2. **supervisord unreachable is never a code trigger.** It restarts the unit.
  3. **LKG only advances here**, after a promotion survives its full window.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import stat
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import detect            # noqa: E402
import gstate            # noqa: E402
import logtail           # noqa: E402
import notify as notify_mod  # noqa: E402
import policy            # noqa: E402
import poolwatch         # noqa: E402
import probes            # noqa: E402
import rollback as rb    # noqa: E402
import vaultwatch        # noqa: E402
import datawatch         # noqa: E402
import memwatch          # noqa: E402
import tmpwatch          # noqa: E402
import voiceloss          # noqa: E402
from supervisor import SupervisorClient, SupervisordUnreachable  # noqa: E402


#: The head of the reason `evaluate_data_damage` gives when the KG store could
#: not be counted, and the prefix the tick call site matches on to decide that
#: the verdict has to reach the journal even though nothing was damaged. One
#: constant on both sides: the guard and its reader cannot drift apart, which is
#: the same rule that took the path out of `policy.KG_DB` (#1525).
KG_UNREADABLE_MARK = "knowledge graph UNREADABLE"


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [guardian] {msg}", flush=True)


def sd_notify(message: str) -> None:
    """Ping systemd's watchdog. Catches a *hung* guardian, not just a dead one."""
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return
    if addr.startswith("@"):
        addr = "\0" + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(addr)
            s.sendall(message.encode())
    except OSError:
        pass


def rollback_title(current: dict | None, bad: str | None, expected: str) -> str:
    """The rollback alert's headline — what was reverted, in words.

    The promoter writes the item's name onto `current.json` as `title` when it
    lands, because this process is stdlib-only and cannot look it up. Read
    aloud, "Rolled back 1a2b3c4d to 5e6f7a8b" is sixteen letters of noise; the
    hashes stay in the body and the ledger. A record without a title (a
    landing from before the field existed, or a hand-driven round with no
    item) falls back to the hashes rather than to the round id.
    """
    title = str((current or {}).get("title") or "").strip()
    if title:
        return f"Rolled back: {title}"
    return f"Rolled back {(bad or '?')[:8]} → {expected[:8]}"


#: The runtime-data stray alert's title, and therefore the daily-note section
#: heading its incident is coalesced under. One constant because the check that
#: raises it and the check that clears it must agree on the string byte for byte —
#: `notify.resolve()` matches headings literally, and a retitled alarm whose
#: retraction still says the old words leaves the stale instructions standing as the
#: only readable record of the incident (#1536).
RUNTIME_DATA_ALERT_TITLE = "Runtime data is being written into the code tree"

#: The one cause sentence the stray alert may print, and the only reading of its
#: own numbers that earns it: a stray `mtime` inside the window between this
#: check and the previous one. Before #2057 this sentence sat in a constant body
#: printed next to a 0-byte stub whose mtime predated the window, so the alert
#: asserted a mechanism it had not measured — and the file it wrote is read as
#: fact by the next session (`~/.local/state/lloyd-guardian/ALERT.md`).
_WRITER_CLAIM = ("something on this box still resolves a data path off the code "
                 "instead of app.paths.DATA_ROOT")


def _stamp(ts: float) -> str:
    """Offset-bearing local stamp for a measured time.

    Never naive: a local time with no offset in this body is read as UTC by every
    later reader, and these bytes are read by sessions as well as by people
    (#1912, `notify._alert_file`)."""
    return datetime.fromtimestamp(ts).astimezone().isoformat()


def _measure_strays(tree: str, names) -> list[tuple[str, str, float | None]]:
    """`(name, rendered, mtime)` per named stray, one row per name, in order.

    `lstat`, not `stat`, for two reasons, both of them the "the number, not a
    story about the number" half of #2057 clause 1: `stray_in_tree` reports a path
    on `lexists`, so a dangling symlink is a stray that `stat` would answer
    `ENOENT` for and drop, and a symlink's `st_size` is the length of its target
    path — printed bare, 87 bytes of link reads like 87 bytes of data. So the link
    is named as one and its target is printed.

    One name's failure never decides another's: the row list is always as long as
    `names`, and a path that vanished between the check and this call is printed as
    unmeasurable rather than omitted. `mtime` is `None` exactly when the render says
    `unmeasurable`, which is what keeps the writer claim off an unmeasured path
    (#2057 clause 4)."""
    rows: list[tuple[str, str, float | None]] = []
    for name in names:
        path = os.path.join(tree, name)
        try:
            st = os.lstat(path)
        except OSError as exc:
            rows.append((name, f"unmeasurable (lstat: "
                               f"{exc.strerror or exc.__class__.__name__})", None))
            continue
        link = (f", symlink → {os.readlink(path)}"
                if stat.S_ISLNK(st.st_mode) else "")
        rows.append((name,
                     f"{st.st_size} bytes, mtime {_stamp(st.st_mtime)}{link}",
                     st.st_mtime))
    return rows


def _stray_alert_body(tree: str, names, now: float,
                      prev_check: float | None) -> str:
    """The stray alert's whole body, built from nothing but this check's numbers.

    Three sections, each derived from the same `_measure_strays` rows the check
    took (#2057): the named paths with their measured size and mtime; a cause
    sentence only if a measured mtime falls inside the window this check actually
    observed; and the remedy routing, which refuses to offer a retained runtime
    store for deletion or for the exclusion list. The deletion instruction and the
    `KNOWN_GOOD_TOPLEVEL` offer used to be unconditional text, and for a retained
    store both are wrong: the first orders someone to delete a store, the second
    asks a name-list to close a property the check holds over the open set of the
    tree's top level — the standing 2026-09-20 rule.
    """
    rows = _measure_strays(tree, names)
    window = (f"({_stamp(prev_check)} → {_stamp(now)})"
              if prev_check is not None else None)
    written = [name for name, _, mtime in rows
               if prev_check is not None and mtime is not None
               and prev_check < mtime <= now]
    if written:
        cause = (f"Written inside the window this check observed {window}: "
                 f"{', '.join(written)}. That mtime is this check's own "
                 f"measurement, so the cause it carries is measured: "
                 f"{_WRITER_CLAIM} ({policy.DATA_ROOT}).")
    elif prev_check is None:
        cause = ("This check does not identify a writer: it is the first stray "
                 "check since this guardian started, so no window bounds when "
                 "the mtimes above were written.")
    else:
        cause = (f"This check does not identify a writer: no mtime above falls "
                 f"inside the window it observed {window}, so those bytes say "
                 f"nothing about what is writing now.")

    retained = [n for n in names if n in datawatch.RUNTIME_NAMES]
    free = [n for n in names if n not in datawatch.RUNTIME_NAMES]
    paras = ["These exist inside the tree again:\n"
             + "\n".join(f"  {os.path.join(tree, name)} — {rendered}"
                         for name, rendered, _ in rows),
             cause]
    if free:
        # The widened check (#1541) reports anything at the top of the tree git
        # does not track, so a name here may be tooling or a rebuildable cache
        # rather than a writer; the residual of an open-set check is a human
        # deciding which side of the list it is on, and say so where they read it
        # or the cheapest answer to a new `.mypy_cache` is to switch the check off.
        # Named individually and *only* the non-retained ones: a retained path must
        # never appear in the same breath as either offer (#2057 clause 3).
        paras.append(f"Not on the watch list: {', '.join(free)}. If it is tooling "
                     "or a rebuildable cache and not a writer, it belongs in "
                     "KNOWN_GOOD_TOPLEVEL in agent-services/guardian/datawatch.py "
                     "— adding its name there is what stops this alert; deleting "
                     "the directory is not.")
    if retained:
        paras.append(f"Retained ({', '.join(retained)}): already on this check's "
                     "own watch list (`RUNTIME_NAMES`, "
                     "agent-services/guardian/datawatch.py), so neither response "
                     "applies to it: it is not offered for deletion, and it is "
                     "not offered for the exclusion list either — no name-list "
                     "closes a check that enumerates the tree's top level, which "
                     "is an open set. The size and mtime above are where a hunt "
                     "for the writer starts.")
    return "\n\n".join(paras)


class Guardian:
    # Defaults for the three fields #2221 added, declared on the CLASS and not only in
    # `__init__`, because two tests in `test_guardian_rollback.py` build a Guardian with
    # `Guardian.__new__` and assign the subset of attributes their scenario needs — the
    # same reason `_PERSISTED_STATES` below is a class attribute. Declaring them here is
    # the difference between a partial object that answers "nothing is open" and one that
    # raises AttributeError inside `tick()`, which would read as a guardian crash rather
    # than as a test that never set the field. `_down_programs` is a FROZEN set for that
    # reason and every write replaces it with a fresh `set` rather than mutating in place:
    # a mutable class-level default would be shared by every instance in the process.
    _down_programs: frozenset = frozenset()
    _sup_unreachable_open = False
    liveness_fail_streak = 0

    def __init__(self, args):
        self.repo = args.repo
        self.state = gstate.AutomodState(Path(args.state))
        self.gdir = Path(args.guardian_state)
        self.gdir.mkdir(parents=True, exist_ok=True)
        self.sup = SupervisorClient(args.supervisor_sock,
                                    timeout=policy.SUPERVISOR_RPC_TIMEOUT)
        self.backend_url = args.backend_url
        self.mcp_url = args.mcp_url
        self.programs = tuple(p for p in args.programs.split(",") if p)
        self.interval = args.interval
        self.notifier = notify_mod.Notifier(
            ledger=self.state.ledger, state_dir=self.gdir,
            vault_root=policy.VAULT_ROOT,
            backend_url=args.backend_url.rsplit("/health", 1)[0],
            external=not getattr(args, "no_external_alerts", False),
            voice_window=policy.VOICE_REPEAT_SECONDS,
        )
        self._alert_seen: dict[str, float] = {}
        self.cursor = logtail.LogCursor(self.gdir / "logcursors.json")
        self.vault = vaultwatch.VaultWatch(policy.VAULT_ROOT, self.gdir)
        self.data = datawatch.DataWatch(policy.DATA_ROOT, self.gdir)
        self._strays_checked_at = 0.0
        # When the stray check last ran, which is the only window the stray alert
        # is allowed to cite as evidence of a live writer. `None` until it has run
        # twice, and the alert says "first check" rather than guessing (#2057).
        self._strays_prev_check_at: float | None = None
        self._snapshots_checked_at = 0.0
        self.mem = memwatch.MemWatch(self.gdir, memwatch.unit_cgroup(policy.SUPERVISORD_UNIT))
        self.tmp = tmpwatch.TmpWatch()
        # Every alarm the worker fleet has lives inside `WorkerPool._scheduler_loop`
        # (#1682), so the one that notices those alarms being switched off cannot
        # live there too. Same backend, same state dir, this process: no new
        # service, no new endpoint, no new config key. Built once per guardian
        # process, which is what makes its grace streak mean "consecutive ticks".
        self.pool = poolwatch.PoolWatch(
            self.gdir, base_url=self.backend_url.rsplit("/health", 1)[0])
        # A spoken alert that never reached the speakers leaves `voice-loss.md`
        # (#1806), and for two days nothing read it — an artefact in the state dir
        # where an alarm should have been (#1904). Same backend, same state dir,
        # this process, and deliberately NOT the child that failed: the route this
        # posts to is `[program:lloyd-backend]`, which shares `agent-supervisord`
        # with the TTS server that just refused, so a post from inside that child
        # would die with the sound. This unit outlives that outage.
        self.voiceloss = voiceloss.VoiceLossEscalator(
            self.gdir, base_url=self.backend_url.rsplit("/health", 1)[0])

        self.tick_n = 0
        self._tick_events: list[dict] = []
        self._tick_overflow = False
        self.probe_fail: dict[str, int] = {p: 0 for p in self.programs}
        self.probe_timeout: dict[str, int] = {p: 0 for p in self.programs}
        self.probe_http: dict[str, int] = {p: 0 for p in self.programs}
        self.mcp_fatal_streak = 0
        self.start_history: dict[str, list[float]] = {p: [] for p in self.programs}
        self.sup_down_streak = 0
        # #2221. Which programs the CURRENT round of `Service down, but no promotion to
        # revert` has been journalled for, so the retraction on the first healthy tick can
        # NAME them. The daily-note section is coalesced — one section per incident — but a
        # section is opened by the incident's first tick and a later tick of the same
        # incident may find a second program gone, so the set accumulates rather than
        # holding the first name. Emptied by the retraction, which is what makes the second
        # healthy tick write nothing. Held in the process rather than read back out of the
        # note deliberately: a guardian restart mid-incident loses the names, and `resolve`
        # then reports "no open section" for a section it could seal only by inventing a
        # program — the honest failure.
        self._down_programs: set[str] = set()
        # Whether `supervisord was unreachable` has a section open. A boolean, not a set:
        # this family's recovery condition is one fact (the supervisor answered), and the
        # streak cannot serve — `sup_down_streak` is zeroed by the alerting tick, so the
        # tick that would resolve has no memory of the outage.
        self._sup_unreachable_open = False
        # #2221 clause 4: how many consecutive ticks the liveness read has come back
        # down. `FATAL` is a state supervisord PERSISTS across a restart of itself, so the
        # word in an alert body says nothing about when the process stopped; the streak is
        # the only age this loop can state without reading a clock into the note, and it is
        # the age that distinguishes "down since the last tick" from "has been FATAL on the
        # supervisor's say-so for two days".
        self.liveness_fail_streak = 0
        self.quiet_until = 0.0
        self.started_ts = time.time()
        self.last_selftest = 0.0
        self.selftest_ok: bool | None = None
        self._selftest_alerted = 0.0
        self.chronic: set[str] = set()
        self._chronic_built_ts = 0.0
        self.last_alert = ""

    def _beat(self) -> None:
        """Ping systemd's watchdog from inside a long-running step.

        `WATCHDOG=1` is otherwise sent once per loop iteration, and a rollback
        is one iteration: two blocking stops (stopwaitsecs=15 each), a writer
        drain, and up to 90s waiting for the backend to come back. That is
        comfortably past the unit's WatchdogSec=90, so systemd killed the
        guardian partway through the rescue, restarted it, and the resume path
        began the same rollback again — a loop that can never reach BROKEN, in
        precisely the case BROKEN exists to report. The beat stays tied to
        real progress rather than a background thread, so a genuinely *hung*
        guardian is still caught.
        """
        sd_notify("WATCHDOG=1")

    # ── heartbeat ──────────────────────────────────────────────────────
    def heartbeat(self, state: str, extra: dict | None = None) -> None:
        payload = {
            "ts": time.time(), "state": state, "tick": self.tick_n,
            "selftest": self.selftest_ok, "pid": os.getpid(),
            "lkg": (self.state.lkg() or {}).get("commit"),
            "head": rb.head_commit(self.repo),
            "last_alert": self.last_alert,
        }
        if extra:
            payload.update(extra)
        try:
            gstate.write_json_atomic(self.gdir / policy.HEARTBEAT_NAME, payload)
        except Exception:
            pass

    # ── probing ────────────────────────────────────────────────────────
    def _url_for(self, program: str) -> str | None:
        """The endpoint that says whether this program is SERVING, not started.

        `agent-tts` gets one for two reasons #2256 turns on. State cannot be its health
        signal: the process is RUNNING for about four minutes before :8090 answers
        anything at all (`agent-tts.conf:24-26`), so "the supervisor says it is up" and
        "the box can speak" are different claims about it — and a recovery has to be
        confirmed by the second, not the first, which is `#1816`'s rule applied to a new
        program. It is also what lets the tick see a port that stopped answering while
        supervisord had nothing to complain about. The `status` word that counts as
        healthy is per-endpoint — see `probes.ok_statuses_for`.
        """
        if program.endswith("lloyd-backend"):
            return self.backend_url
        if program.endswith("lloyd-mcp"):
            return self.mcp_url
        if program == "agent-tts":
            return policy.TTS_HEALTH_URL
        return None

    def collect(self) -> dict:
        """One immutable snapshot per tick."""
        snap: dict = {"now": time.time(), "supervisord": "ok", "procs": {}, "probes": {}}
        try:
            snap["procs"] = self.sup.all_process_info()
        except SupervisordUnreachable as exc:
            snap["supervisord"] = "unreachable"
            snap["supervisord_error"] = str(exc)[:200]
            snap["backend_health"] = self._backend_health(snap)
            return snap

        for program in self.programs:
            info = snap["procs"].get(program)
            if info:
                start = float(info.get("start") or 0)
                hist = self.start_history.setdefault(program, [])
                if start and (not hist or hist[-1] != start):
                    hist.append(start)
                    del hist[:-10]
            url = self._url_for(program)
            if url:
                snap["probes"][program] = probes.probe(url, policy.PROBE_TIMEOUT_SECONDS)
        snap["backend_health"] = self._backend_health(snap)
        return snap

    def _backend_health(self, snap: dict) -> dict:
        """This tick's verdict for the backend's own `/health`, published for
        whatever downstream has to know whether the backend is answering before it
        concludes anything about what the backend is doing (#1747: a backend that
        is down is not a worker pool that is stopped, and the pool watch that
        cannot tell those apart pages twice per outage).

        Reuses the program probe that already hit this URL rather than asking twice
        a moment apart — the two readings must agree, and `poolwatch` is handed one
        value, not two chances to disagree. Probed directly only when no watched
        program maps to this backend, which is the case a guardian started with
        `--programs` narrowed still has to survive.
        """
        for program, result in snap["probes"].items():
            if result is not None and self._url_for(program) == self.backend_url:
                return result
        return probes.probe(self.backend_url, policy.PROBE_TIMEOUT_SECONDS)

    _PERSISTED_STATES = ("FATAL", "STOPPED", "EXITED", "BACKOFF", "STARTING")
    def _down_program_names(self, reason: str) -> set[str]:
        """Which watched programs a liveness reason string names, as the alert stated them.

        Read off the reason rather than from `detect.process_down` because the reason is
        what the note quotes: `evaluate_liveness` returns the FIRST failure only, so the
        alert body and the retraction must agree on one name per tick, and re-deriving a
        set from the snapshot would let the two disagree. Matching against `self.programs`
        rather than splitting on `":"` is necessary, not tidy: the watched names carry
        colons themselves (`lloyd-mc:lloyd-backend`), so a split yields `lloyd-mc`.

        The consequence of one-name-per-tick is stated plainly: when two programs are down
        and one is earlier in the tuple, only that one is ever named, so the retraction of
        a multi-program incident can name fewer programs than it cleared. That is the alert
        body's limitation, not the retraction's, and a `cleared:` line naming one program of
        two is closer to the truth than the standing instruction that replaces nothing.
        """
        return {name for name in self.programs if reason.startswith(f"{name}: ")}


    def _describe_down(self, reason: str) -> str:
        """Say how long a quoted supervisor state has stood, next to the state itself.

        #2221 clause 2's whole live case is this: `memory/2026-09-29.md` still tells a human
        "lloyd-mc:lloyd-backend: FATAL: can't find command
        '/home/alansrobotlab/lloyd/.venvs/lloyd/bin/python' … this needs a human", while
        `supervisorctl status` has answered `lloyd-mc:lloyd-backend RUNNING` for days and
        the interpreter that file names exists. `FATAL` is what supervisord PERSISTS about a
        process — it is not a probe of the cause, and it says nothing about when the state
        was last true. Streaking the failing liveness reads beside the quote turns the body
        from "the program is FATAL" into "the program has reported FATAL for N consecutive
        reads at a M-second interval", which is a claim a reader can date, and the
        difference between acting on an outage and re-reporting one.

        Only the states that are persisted get the age. `RUNNING` with a failed HTTP probe
        is a live observation made this tick, and dressing that up as a persisted state
        would understate it.
        """
        streak = max(1, self.liveness_fail_streak)
        if any(state in reason for state in self._PERSISTED_STATES):
            return (f"{reason}\n"
                    f"- supervisor state as read on this tick, and it has been read down "
                    f"for {streak} consecutive liveness tick(s) at {self.interval:g}s "
                    f"(a persisted state, not a live probe of its cause)")
        return reason

    def _recoverable_down(self, live_reason: str) -> list:
        """The programs this tick's liveness reason names that the guardian may restart
        itself, rather than report or roll back for.

        Read off the reason with the same helper the retraction uses, because
        `evaluate_liveness` returns the FIRST failure only: the alert body, the retraction
        and this dispatch have to agree on one name per tick, and a second derivation from
        the snapshot is how they would start disagreeing. The one-name-per-tick limit
        `_down_program_names` documents applies here too — a second recoverable program
        down in the same tick waits for the tick after the first one serves again.
        """
        return [p for p in sorted(self._down_program_names(live_reason))
                if p in policy.RECOVERABLE_INFRA]

    def _recover_infra(self, live_reason: str) -> str:
        """Restart the down `RECOVERABLE_INFRA` program(s) and report what happened.

        Two outcomes reach a human. A voice channel that comes back is said out loud as
        `RECOVERED:` — the same shape the code review asked of an unattended recovery as
        everywhere else in this file — and a voice channel that cannot be brought back is
        an error with `needs_human=True`, because at that point the guardian has spent
        its restart budget on the box's only alert route and cannot fix it again.
        """
        out = "recovered"
        for program in self._recoverable_down(live_reason):
            ok, detail = self.recover_service(program, live_reason)
            if ok:
                log(f"{program} recovered: {detail}")
                self.alert("info", f"Recovered: {program}",
                           f"{program} was {live_reason}.\nRestarted; {detail}.",
                           coalesce=True)
                # The incident note this alert opened is sealed by the retraction pass at
                # the top of the next tick — which is now reachable, because the program
                # serves again — rather than here, where the seal would have to assume the
                # restart holds.
            else:
                out = "needs_human"
                log(f"{program} NOT recovered: {detail}")
                self.alert("error", f"Voice alert channel cannot be recovered: {program}",
                           self._describe_down(live_reason) +
                           f"\n\nThe guardian restarted {program} and it did not come back: "
                           f"{detail}\n"
                           f"Spoken alerts are not reaching the room, so this message is "
                           f"here rather than out loud. It needs a human — a restart is all "
                           f"this guardian can do to a synthesiser, and it has done it.",
                           needs_human=True, coalesce=True)
        return out

    def _recent_recoveries(self, program: str) -> int:
        """Recovery restarts of `program` inside the flap window. The same ledger read
        flap protection uses, counting a different event instead of rollbacks."""
        cutoff = time.time() - policy.FLAP_WINDOW_SECONDS
        return sum(
            1 for ev in gstate.read_events(self.state.ledger, limit=gstate.FLAP_SCAN_ROWS)
            if ev.get("event") == "service_recovery"
            and ev.get("program") == program
            and float(ev.get("ts", 0)) >= cutoff
        )

    def recover_service(self, program: str, reason: str) -> tuple[bool, str]:
        """Bring one `RECOVERABLE_INFRA` program back, and confirm it by its endpoint.

        Not `restart_services`: that one stops and starts `RESTART_ORDER`, which is the
        set a rollback moves, and this program is deliberately not in it (see the
        `policy.RECOVERABLE_INFRA` comment). What it does share with that path is the
        rule it ends on — a supervisor state is not health, so a restart counts only
        once :8090 answers (`supervisor.py`'s docstring says the same about the
        backend). Stop-then-start rather than `restartctl` because a FATAL or a wedged
        RUNNING process answers neither, and the box's own 2026-10-05 lesson is that a
        process supervisord has *stopped* stays stopped until someone owns it.

        Bounded by the same numbers flap protection gives the backend: 
        `FLAP_HALT_AFTER` restarts inside `FLAP_WINDOW_SECONDS`, and past that this
        returns False with a reason that names the budget, so the loop never turns a
        four-minute synthesiser boot into a restart storm on a box where the model
        genuinely cannot load. The ledger row is written BEFORE the attempt — same
        reason `recent_rollbacks` counts `rollback_succeeded`: an attempt that hangs is
        an attempt that must still be counted.

        Blocks for up to `HEALTH_WAIT_TTS`. That is a real cost, so the wait is
        announced in `heartbeat.json` first — a reader who finds a seven-minute gap in
        that file should find this program's name in the last row before it — and
        `on_tick=self._beat` is what keeps `WatchdogSec=90` off the guardian's back
        while it waits.
        """
        spent = self._recent_recoveries(program)
        if spent >= policy.FLAP_HALT_AFTER:
            return False, (f"restart budget spent ({spent} of {policy.FLAP_HALT_AFTER} "
                           f"allowed in "
                           f"{policy.FLAP_WINDOW_SECONDS / 3600:.0f}h); not restarting again. "
                           f"The fix is a human's `supervisorctl start {program}` — what the "
                           f"guardian will not do is keep spawning a process that will not "
                           f"serve, nor reach for a rollback, whose route acts on "
                           f"RESTART_ORDER and cannot touch this program.")

        gstate.append_event(self.state.ledger, {
            "event": "service_recovery", "program": program, "cause": reason[:300]})
        log(f"recovering {program} ({reason}) — attempt {spent + 1}/"
            f"{policy.FLAP_HALT_AFTER} in {policy.FLAP_WINDOW_SECONDS / 3600:.0f}h")
        try:
            self.sup.stop(program, wait=True)
        except Exception as exc:                      # already stopped: fine
            log(f"stop {program}: {type(exc).__name__}: {exc}")
        self._beat()
        try:
            started, msg = self.sup.start(program, wait=False)
        except Exception as exc:
            return False, f"supervisord refused to start {program}: {exc}"
        if not started:
            return False, f"supervisord could not start {program}: {msg}"
        log(f"start {program}: {msg} — waiting up to {policy.HEALTH_WAIT_TTS:.0f}s "
            f"for {policy.TTS_HEALTH_URL}")
        self.heartbeat("recovering", {
            "recovering": program,
            "recovering_until": time.time() + policy.HEALTH_WAIT_TTS,
            "recovering_reason": reason[:200]})
        ok, last = probes.wait_healthy(policy.TTS_HEALTH_URL,
                                       policy.HEALTH_WAIT_TTS,
                                       policy.PROBE_TIMEOUT_SECONDS,
                                       on_tick=self._beat)
        if not ok:
            return False, (f"started ({msg}) but {policy.TTS_HEALTH_URL} never answered "
                           f"within {policy.HEALTH_WAIT_TTS:.0f}s "
                           f"(last: {(last or {}).get('kind', 'no probe')})")
        return True, f"{msg}; {policy.TTS_HEALTH_URL} answering"

    def evaluate_liveness(self, snap: dict) -> tuple[bool, str]:
        for program in self.programs:
            info = snap["procs"].get(program)
            result = snap["probes"].get(program)
            is_mcp = program.endswith("lloyd-mcp")
            if result is not None:
                kind = result.get("kind")
                if result["ok"]:
                    self.probe_fail[program] = 0
                    self.probe_timeout[program] = 0
                    self.probe_http[program] = 0
                elif kind == "timeout":
                    # Alive but slow. Counted, but on its own long budget.
                    self.probe_timeout[program] = self.probe_timeout.get(program, 0) + 1
                    self.probe_fail[program] = 0
                    self.probe_http[program] = 0
                elif kind == "http_error":
                    # It ANSWERED. That is not "nothing is listening", and
                    # counting it as one is how a merely degraded aggregator
                    # got three ticks to look like a dead one — the careful
                    # newly-degraded-since-LKG check below sits *after* the
                    # down predicate and so never got a vote. The aggregator's
                    # 503 is judged there instead, and contributes nothing
                    # here; the backend's own 503 (a router that failed to
                    # mount, a startup that never completed) is real, and gets
                    # its own much wider budget.
                    self.probe_fail[program] = 0
                    self.probe_timeout[program] = 0
                    if not is_mcp:
                        self.probe_http[program] = self.probe_http.get(program, 0) + 1
                else:
                    self.probe_fail[program] = self.probe_fail.get(program, 0) + 1
                    self.probe_timeout[program] = 0
                    self.probe_http[program] = 0

            grace = policy.BOOT_GRACE.get(program, policy.DEFAULT_BOOT_GRACE)
            down, reason = detect.process_down(
                info,
                now=snap["now"],
                grace=grace,
                probe_fail_streak=self.probe_fail.get(program, 0),
                probe_threshold=policy.PROBE_FAIL_STREAK,
                probe_timeout_streak=self.probe_timeout.get(program, 0),
                probe_timeout_threshold=policy.PROBE_TIMEOUT_STREAK,
                probe_http_streak=self.probe_http.get(program, 0),
                probe_http_threshold=policy.PROBE_HTTP_ERROR_STREAK,
                start_history=self.start_history.get(program, []),
                crash_loop_starts=policy.CRASH_LOOP_STARTS,
                crash_loop_window=policy.CRASH_LOOP_WINDOW_SECONDS,
                # A STOPPED process while promotions are halted is one WE
                # stopped: flap protection quarantines the backend by design.
                # Reading our own deliberate stop as death produced a "service
                # down" alert every 15 minutes for as long as the quarantine
                # lasted. EXITED is never excused this way.
                #
                # Scoped to the programs this guardian can actually stop —
                # `RESTART_ORDER`, which is what flap protection and a rollback
                # stop and nothing else — because the flag means "we stopped this
                # on purpose" and cannot honestly excuse a program no guardian
                # path ever touches. #2256: `agent-tts` sat STOPPED unowned for
                # 45 minutes, and an unqualified flag would have hidden it for as
                # long as an unrelated backend quarantine happened to last.
                intentional_stop=(self.state.is_halted()
                                  and program in policy.RESTART_ORDER),
            )
            if down:
                return True, f"{program}: {reason}"

            # A degraded aggregator is usually an external-app bridge being
            # absent, not a bad promotion. Only newly-degraded modules count.
            if is_mcp and result and result.get("kind") == "http_error":
                body = result.get("body") or {}
                baseline = ((self.state.lkg() or {}).get("health") or {}).get("mcp_degraded_modules")
                fatal, why = detect.mcp_degraded_is_fatal(body, baseline)
                if fatal:
                    # Confirm across ticks before acting. This verdict fires on
                    # a body reporting zero tools, and an aggregator answering
                    # 500 mid-restart parses to exactly that — so a single bad
                    # response would have reverted code on its own. Every other
                    # detector here requires a streak; this one was reached
                    # only on a 503 before, which hid how sharp it was.
                    self.mcp_fatal_streak += 1
                    if self.mcp_fatal_streak >= policy.MCP_FATAL_STREAK:
                        return True, (f"{program}: {why} "
                                      f"({self.mcp_fatal_streak} consecutive ticks)")
                    log(f"{program}: {why} — {self.mcp_fatal_streak}/"
                        f"{policy.MCP_FATAL_STREAK}, waiting for confirmation")
                else:
                    self.mcp_fatal_streak = 0
            elif is_mcp:
                self.mcp_fatal_streak = 0
        return False, "all watched processes healthy"

    # ── error-rate ─────────────────────────────────────────────────────
    def ensure_chronic(self) -> None:
        # In-process expiry as well as on-disk: the guardian runs for weeks at
        # a time, so a load-once guard would pin the set for the life of the
        # process and make the on-disk TTL unreachable.
        if self._chronic_built_ts and (
                time.time() - self._chronic_built_ts) < policy.CHRONIC_REFRESH_SECONDS:
            return
        cache = self.gdir / "signatures.json"
        cached = gstate.read_json(cache)
        # The cache EXPIRES, and that is not housekeeping. Learned once and
        # kept forever, the chronic set describes the box as it was on the
        # first boot after the state dir was created — so every recurring
        # error that starts happening *later* stays "novel" indefinitely, and
        # the next promotion is reverted for a steady-state failure it had
        # nothing to do with. That is the same shape as the stale log cursor:
        # a detector quietly judging a new commit by old evidence.
        age = time.time() - float((cached or {}).get("built_ts") or 0)
        fresh = bool(cached and isinstance(cached.get("chronic"), list)
                     and age < policy.CHRONIC_REFRESH_SECONDS)
        if fresh:
            self.chronic = set(cached["chronic"])
        else:
            self.chronic = logtail.bootstrap_chronic(
                list(policy.LOG_FILES),
                max_bytes=20 * 1024 * 1024,
                min_distinct_hours=policy.CHRONIC_MIN_DISTINCT_HOURS,
            )
            gstate.write_json_atomic(cache, {
                "chronic": sorted(self.chronic), "built_at": gstate.now_iso(),
                "built_ts": time.time(),
            })
        self._chronic_built_ts = float((cached or {}).get("built_ts") or 0) if fresh else time.time()
        log(f"chronic signature set: {len(self.chronic)} entries (never trigger)"
            f"{'' if fresh else ' — rebuilt'}")

    def drain_logs(self) -> None:
        """Advance the log cursor and buffer this tick's error events.

        Called on EVERY tick, whatever state the guardian is in, and that is
        the whole point. Reading used to happen inside `evaluate_errors`,
        which only runs while a promotion is under observation — so between
        rounds the cursor stood still and the first tick of a new observation
        window read *everything since the last one*.

        On 2026-09-06 that reverted a healthy promotion four seconds after it
        landed, on nine `ConnectError` lines from 11:47–11:56 that morning:
        eight hours stale, produced by an unrelated incident, and attributed
        to a commit that had existed for four seconds. A window that observes
        a commit must only ever see errors that happened while it was open.

        Errors are still *judged* only during an observation window. What
        changed is that the tape always moves.
        """
        events: list[dict] = []
        overflowed = False
        for path in policy.LOG_FILES:
            text, over = self.cursor.read_new(path, policy.LOG_READ_CAP_BYTES)
            overflowed = overflowed or over
            if text:
                events.extend(detect.extract_events(text))
        self.cursor.save()
        self._tick_events = events
        self._tick_overflow = overflowed

    def evaluate_errors(self, current: dict) -> tuple[bool, str]:
        self.ensure_chronic()
        changed = current.get("changed_paths") or []
        events = self._tick_events
        overflowed = self._tick_overflow
        if overflowed:
            return True, "log overflow: >4MiB of stderr in one tick"
        if not events:
            return False, "no new error events"
        return detect.error_spike(
            events,
            chronic=self.chronic,
            changed_paths=changed,
            novel_threshold=policy.NOVEL_SIGNATURE_THRESHOLD,
            fatal_distinct_threshold=policy.NOVEL_FATAL_DISTINCT_THRESHOLD,
            changed_path_threshold=policy.NOVEL_IN_CHANGED_PATH_THRESHOLD,
        )

    def evaluate_data_damage(self, current: dict) -> tuple[bool, str]:
        before_rows = current.get("kg_rows")
        before_files = current.get("vault_files")
        rows_now, kg_read = count_kg_rows(policy.KG_DB)
        files_now = count_vault_files(policy.VAULT_ROOT)
        hit, why = detect.data_damage(before_rows, rows_now, policy.DATA_DROP_FRACTION)
        if hit:
            return True, f"knowledge graph rows {why}"
        hit, why = detect.data_damage(before_files, files_now, policy.DATA_DROP_FRACTION)
        if hit:
            return True, f"vault files {why}"
        # A count that could not be taken is not a store that is intact. The
        # predicate answers "no baseline" both for a missing baseline and for a
        # missing count, and this call site threw the reason away — so until #1525
        # a knowledge graph that could not be opened at all, including one whose
        # data root had moved out from under a restated path, reached the journal
        # as "data intact". The vault tripwire above is still the thing that stops
        # a lost store being shrugged off; this only names which store it could not
        # read, and still rolls back nothing. The branch that NAMED the path goes
        # in too, because the resolver and the fallback agree on the deployed box:
        # `fallback-literal` in this line means the resolver was not loadable,
        # which is a different incident from a root that moved.
        if rows_now is None:
            where = kg_read or "no path given"
            if kg_read:
                where += f" (path from the {policy.KG_DB_SOURCE} branch)"
            return False, (KG_UNREADABLE_MARK
                           + (" (no baseline written yet)" if not before_rows else "")
                           + f": {where}")
        return False, "data intact"

    # ── rollback ───────────────────────────────────────────────────────
    def do_rollback(self, trigger: str, reason: str, *,
                    explicit_target: str | None = None,
                    explicit_commit: str | None = None,
                    explicit_changed: list | None = None,
                    explicit_commits: list | None = None) -> bool:
        current = self.state.current() or {}
        # A request may name a whole batch (the land train's promoter undoing
        # a flush it could not verify); its newest commit is then the blame.
        requested_batch = gstate.batch_commits({"commits": explicit_commits})
        if requested_batch and not gstate._is_sha(explicit_commit):
            explicit_commit = requested_batch[-1]
        # Which commit this request BLAMES, decided once, here, before anything
        # else reads `current.json`. It is a different question from "which
        # promotion is under observation", and running the two together is what
        # discarded two promotions on 2026-09-21: a detached regression check
        # asked for a revert of an older, already-settled commit while a newer
        # promotion sat in `current.json`, the requested commit was read only
        # when that file was absent, and so the observed promotion was the one
        # blamed — HEAD *was* that promotion, the reset route deleted both, and
        # the ledger row named the wrong one (`commit=a802b979` in the request,
        # `commit=1e219da9 route=reset` in the result; then `dbec85aa` →
        # `edc8ec60` an hour later).
        blamed = (explicit_commit if gstate._is_sha(explicit_commit)
                  else current.get("commit"))
        blamed_is_observed = bool(blamed) and blamed == current.get("commit")
        changed = (list(current.get("changed_paths") or []) if blamed_is_observed
                   else list(explicit_changed or []))
        # The batch this rollback takes off `main`, oldest first: the observed
        # record's own `commits` when it is the one blamed, a request's when it
        # names them, else none — and with none, everything below is exactly
        # the single-commit rollback it always was. A request blaming ONE
        # commit of an observed batch (the detached regression check) reverts
        # that commit only, and the batch's window closes unjudged.
        if blamed_is_observed:
            batch = gstate.batch_commits(current)
        else:
            batch = requested_batch
        if batch and batch[-1] != blamed:
            batch = []

        if explicit_target and gstate._is_sha(explicit_target):
            target, source = explicit_target, "explicit rollback request"
        elif blamed_is_observed or not blamed:
            target, source = self.state.rollback_target(current)
        else:
            # The request names a commit that is NOT the promotion under
            # observation. That record's `rollback_target` is the tree as it
            # stood before some *other* promotion, so it is not this rollback's
            # to reach; fall through the ladder as if nothing were observed.
            target, source = self.state.rollback_target(None)
        head = rb.head_commit(self.repo)
        floor = self.state.floor()

        if not target:
            self.escalate("no rollback target", f"{reason}\n\n{source}")
            return False

        # The blamed commit may already be gone — reverted by hand, or by a
        # promoter that failed after its merge and undid itself inline. Its
        # `rollback_target` still points somewhere real, so every check below
        # passes and the guardian would happily rewind the tree a second time,
        # discarding whatever landed since. Absence from history is the tell,
        # and it has to be asked of the commit being BLAMED: asked of the record
        # on disk instead, a stale duplicate request naming an already-reverted
        # commit passes every gate while a live `current.json` vouches for
        # someone else — which is how one rollback could become two.
        if blamed and head and blamed != head and not rb.is_ancestor(
                self.repo, blamed, head):
            self.alert("error", "Promotion is no longer in this history",
                       f"{reason}\n\n{blamed[:8]} is not an ancestor of HEAD "
                       f"({head[:8]}), so it has already been reverted or was never "
                       "landed here. Not rewinding again — that would discard work "
                       "this loop never touched.")
            if blamed_is_observed:
                # The vanished commit is the record on disk, so that record is
                # the stale thing and it goes. Otherwise the window still open
                # belongs to a promotion this request never touched.
                self.state.clear_current()
            return False
        if batch and head:
            # A batch commit already gone from this history is not reverted
            # twice. What is left keeps the batch rule even if it is one
            # commit: the record's target is the OLDEST entry's parent, so a
            # plain reset from HEAD could reach past what is still here.
            batch = [c for c in batch if c == head or rb.is_ancestor(self.repo, c, head)]
        if target == head:
            # Invariant 1. Nothing was promoted; this is infrastructure.
            self.alert("error", "Failure with nothing to revert",
                       f"{reason}\nHEAD is already the last known good ({target[:8]}). "
                       "Not rewriting history — this needs a human.",
                       needs_human=True)
            return False
        if not rb.commit_exists(self.repo, target):
            self.escalate("rollback target missing", f"{target} is not in the object store")
            return False
        if floor and target != floor and not rb.is_ancestor(self.repo, floor, target):
            self.escalate("rollback target below floor",
                          f"{target[:8]} predates the guardian floor {floor[:8]}")
            return False

        stamp = time.strftime("%Y%m%d_%H%M%S")
        gstate.append_event(self.state.ledger, {
            "event": "rollback_started", "trigger": trigger, "reason": reason[:1000],
            "from": head, "to": target, "stamp": stamp,
            **({"commits": batch, "batch": len(batch)} if batch else {}),
        })

        for attempt in range(1, policy.ROLLBACK_MAX_ATTEMPTS + 1):
            try:
                self._rollback_once(target, stamp, trigger, reason, current,
                                    blamed=blamed, changed=changed, batch=batch)
                return True
            except rb.RevertConflict as exc:
                # Deterministic: the same commits against the same tree
                # conflict the same way on every attempt.
                log(f"rollback attempt {attempt} failed, not retrying: {exc}")
                break
            except Exception as exc:
                log(f"rollback attempt {attempt} failed: {exc}")
                if attempt < policy.ROLLBACK_MAX_ATTEMPTS:
                    # Beat through the wait. A bare sleep here is 60s of
                    # silence against a 90s watchdog, immediately after a
                    # failed attempt has already spent most of the budget.
                    waited = 0.0
                    while waited < policy.ROLLBACK_RETRY_SECONDS:
                        self._beat()
                        time.sleep(min(5.0, policy.ROLLBACK_RETRY_SECONDS - waited))
                        waited += 5.0

        gstate.append_event(self.state.ledger, {
            "event": "rollback_failed", "trigger": trigger, "to": target,
        })
        self.escalate("ROLLBACK FAILED", reason)
        return False

    def _rollback_once(self, target: str, stamp: str, trigger: str, reason: str,
                       current: dict | None = None, *,
                       blamed: str | None = None,
                       changed: list | None = None,
                       batch: list | None = None) -> None:
        head_before = rb.head_commit(self.repo)
        batch = list(batch or [])
        current = current if current is not None else (self.state.current() or {})
        if blamed is None:
            blamed = current.get("commit")
        if changed is None:
            changed = list(current.get("changed_paths") or [])
        boot_before = current.get("boot_id")
        observed = current.get("commit")

        # Which ROUTE back, decided by whether HEAD *is* the commit being
        # blamed — never by whether a `current.json` happens to exist.
        # `reset --hard` to the promotion's parent is only correct while HEAD
        # still *is* that promotion. Nightly jobs commit straight to live
        # `main`, and a detached quality check can ask for a commit that
        # SETTLED hours ago, so the tree has often moved on: resetting past
        # that destroys commits nobody asked the guardian to judge, and — when
        # the blame and the observation are different commits — destroys the
        # promotion under observation along with the blamed one, which is what
        # happened twice on 2026-09-21. When HEAD is not the blamed commit,
        # revert exactly that commit and leave the rest standing.
        #
        # A batch (the land train: several merged rounds, one record) resets
        # only when HEAD is its newest commit AND `target..HEAD` is exactly the
        # batch. Otherwise something the loop never promoted sits inside or on
        # top of it, and every batch commit is reverted in place instead.
        if batch:
            surgical = not (head_before == blamed and rb.range_is_exactly(
                self.repo, target, head_before, batch))
        else:
            surgical = bool(blamed and head_before and head_before != blamed)

        # 3. Stop the writers first — see the module docstring.
        for program in reversed(policy.RESTART_ORDER):
            ok, msg = self.sup.stop(program, wait=True)
            log(f"stop {program}: {msg}")
            self._beat()
        holders = rb._drain_writers(self.repo, policy.WRITER_DRAIN_SECONDS,
                                    watch_paths=policy.CLEAN_PATHS + (".git",),
                                    heartbeat=self._beat)
        if holders:
            log(f"warning: pids still holding write fds on the code tree: {holders}")

        # 4-5. Quiesce, then preserve the evidence before destroying it.
        rb._wait_for_index_lock(self.repo, policy.INDEX_LOCK_STALE_SECONDS,
                                heartbeat=self._beat)
        tag = f"guardian-broken-{stamp}"
        evidence = rb.preserve_evidence(self.repo, self.state.broken_dir / stamp, tag)
        log(f"preserved: tag={evidence['tag']} stash={evidence['stash']}")
        self._beat()

        # 6-8. Move the tree, verify it, undo any venv swap.
        if surgical and batch:
            kept = rb.commits_between(self.repo, batch[0], head_before) - len(batch) + 1
            expected = rb.revert_commits(self.repo, batch, reason=reason)
            log(f"reverted {len(batch)} batch commits in place → {expected[:8]}, "
                f"keeping {max(0, kept)} commit(s) the loop did not promote")
        elif surgical:
            kept = rb.commits_between(self.repo, blamed, head_before)
            expected = rb.revert_commit(self.repo, blamed, reason=reason)
            log(f"reverted {blamed[:8]} in place → {expected[:8]}, "
                f"keeping {kept} later commit(s)")
        else:
            rb.restore_tree(self.repo, target, policy.CLEAN_PATHS, policy.PYCACHE_PATHS)
            rb.verify_tree(self.repo, target)
            expected = target
        self._beat()
        # The venv swap belongs to the landing that recorded it. Reverting an
        # older, already-settled commit must not undo a NEWER promotion's venv:
        # that swap is not what this rollback was asked about.
        if current.get("venv_swapped") and observed == blamed:
            failed = rb.swap_venv_back(self.repo)
            log(f"venv reverted, failed clone kept at {failed}")

        # 9-10. Restart and confirm the RUNNING code actually changed.
        for program in policy.RESTART_ORDER:
            ok, msg = self.sup.start(program, wait=False)
            log(f"start {program}: {msg}")
            url = self._url_for(program)
            budget = (policy.HEALTH_WAIT_MCP if program.endswith("lloyd-mcp")
                      else policy.HEALTH_WAIT_BACKEND)
            if url:
                healthy, last = probes.wait_healthy(url, budget, policy.PROBE_TIMEOUT_SECONDS,
                                                    on_tick=self._beat)
                if not healthy:
                    raise rb.RollbackError(
                        f"{program} unhealthy after restart: "
                        f"status={last.get('status')} err={last.get('error')}")

        final = probes.probe(self.backend_url, policy.PROBE_TIMEOUT_SECONDS)
        body = final.get("body") or {}
        if body.get("commit") and body["commit"] != expected:
            raise rb.RollbackError(
                f"backend reports commit {body['commit'][:8]}, expected {expected[:8]} "
                "— the restart did not pick up the reverted code")
        if boot_before and body.get("boot_id") == boot_before:
            raise rb.RollbackError("backend boot_id unchanged — process was never replaced")

        # 11-12. Record, denylist, re-arm quiet.
        # Deny the BLAMED commit, not whatever HEAD happened to be and not the
        # promotion that merely happened to be under observation: on the
        # surgical route HEAD was a later human commit, and denying the
        # observed promotion instead of the blamed one left the change that
        # actually regressed free to be re-derived and re-landed while a change
        # nobody blamed was blocklisted by SHA *and* by content hash. Content
        # hash alongside the SHA, so the same change re-derived under a new SHA
        # is caught too.
        bad = blamed or head_before
        if batch:
            # Every commit the batch took off `main`, each by its own content.
            entries = {str(e.get("commit")): e for e in (current.get("entries") or [])
                       if isinstance(e, dict)} if observed == blamed else {}
            for c in batch:
                paths = (list((entries.get(c) or {}).get("changed_paths") or [])
                         or rb.changed_paths_of(self.repo, c))
                self.state.deny(c, tree_hash=rb.changed_tree_hash(self.repo, c, paths))
        elif bad:
            self.state.deny(bad, tree_hash=rb.changed_tree_hash(self.repo, bad, changed))

        # A promotion that was under observation but NOT blamed now has no
        # window and no verdict. Alan's decision on #1358 (2026-09-23) is to
        # close it unjudged rather than reopen it on the new HEAD: a window
        # re-opened across this rollback's own restart would convict it of the
        # rollback, and every rollback this loop has performed so far has been a
        # false positive. It is not unguarded — the detached regression check
        # still measures it — it is simply not judged by this process, and the
        # row below says so beside the commit that was reverted.
        left_unjudged = observed if (observed and observed != bad) else None

        # LKG must not outlive the change it certifies. `maybe_settle` is the
        # only other writer of this pointer, so without a repoint the rollback
        # that undid or removed the commit LKG names would leave
        # `rollback_target(None)` handing that same dead commit to the NEXT
        # rollback — and on 2026-09-21 dbec85aa settled at 21:37:32Z and was
        # reverted at 21:45:05Z, so the following rollback would have reset
        # `main` back onto it and discarded every commit since. Point it at what
        # this rollback actually restored.
        lkg = self.state.lkg() or {}
        lkg_from = lkg.get("commit")
        repointed = bool(expected and lkg_from) and (
            lkg_from == bad or lkg_from in batch
            or not rb.is_ancestor(self.repo, lkg_from, expected))
        if repointed:
            self.state.set_lkg(expected)

        self.state.clear_current()
        gstate.append_event(self.state.ledger, {
            "event": "rollback_succeeded", "trigger": trigger, "commit": bad,
            "restored": expected, "target": target,
            "route": "revert" if surgical else "reset",
            "left_unjudged": left_unjudged,
            "lkg_repointed_from": lkg_from if repointed else None,
            "head_before": head_before, "tag": tag, "stash": evidence.get("stash"),
            # Only on a batch, so a single-commit row is the row it always was.
            # `state.reverted_commits` counts every sha named here.
            **({"commits": batch, "batch": len(batch)} if batch else {}),
        })
        self.quiet_until = time.time() + policy.POST_ROLLBACK_QUIET_SECONDS
        for program in self.programs:
            self.probe_fail[program] = 0
            self.probe_timeout[program] = 0
            self.start_history[program] = []

        recent = self.state.recent_rollbacks(policy.FLAP_WINDOW_SECONDS)
        extra = ""
        if recent >= policy.FLAP_STOP_AFTER:
            self.state.set_halted(f"{recent} rollbacks in 6h")
            self.sup.stop("lloyd-mc:lloyd-backend", wait=False)
            extra = (f"\n\nQUARANTINE: {recent} rollbacks in 6h. Promotions halted AND the "
                     "backend stopped — something is systematically wrong.")
        elif recent >= policy.FLAP_HALT_AFTER:
            self.state.set_halted(f"{recent} rollbacks in 6h")
            extra = (f"\n\nQUARANTINE: {recent} rollbacks in 6h. Promotions halted; Lloyd keeps "
                     "running on known-good code but cannot land more changes until you clear "
                     f"{self.state.halted}.")

        route = ("Reverted in place, keeping later commits."
                 if surgical else "Reset to the pre-promotion tree.")
        if batch:
            route += (f" A batch of {len(batch)} landings: "
                      + ", ".join(c[:8] for c in batch) + ".")
        # The item name on `current.json` is the OBSERVED promotion's, and a
        # request carries no title, so when this rollback blamed someone else
        # the headline has no right to that name — it falls back to the hashes
        # rather than reading out the title of a change still standing in the
        # tree. (`rollback_title` prefers `title` whenever the record has one.)
        self.alert(
            "critical" if extra else "warn",
            rollback_title(current if observed == blamed else None, bad, expected),
            f"Trigger: {trigger}\n{reason}\n{route}\n"
            f"Reverted {(bad or '?')[:8]} → {expected[:8]}{extra}",
            evidence=json.dumps(evidence, indent=2),
            commit=bad or "", trigger=trigger, tag=tag,
        )

    def escalate(self, title: str, body: str) -> None:
        """Terminal state. Leave the stack stopped rather than half-reverted.

        With no human in the loop, an honestly-dead system is safer than an
        autonomous agent running half-reverted code with a live scheduler.
        """
        self.state.set_broken(f"{title}: {body[:400]}")
        gstate.append_event(self.state.ledger, {"event": "escalated", "title": title,
                                                "body": body[:2000]})
        self.alert("critical", title,
                   body + "\n\nServices left stopped. Once resolved, clear "
                          f"{self.state.broken} with `python -m "
                          "scripts.automod.round recover` — it records who "
                          "cleared it; deleting the file by hand leaves no "
                          "ledger row.")

    def alert(self, level: str, title: str, body: str, **kw) -> None:
        # Repeat-suppression. A persistent condition ticks every 5s, and an
        # un-deduped alert buries the one that matters: on 2026-09-06 five
        # identical "Service down, but no promotion to revert" notices landed
        # in 30 seconds. Same title within the window is logged, not fanned out.
        now = time.time()
        last = self._alert_seen.get(title, 0.0)
        self._alert_seen[title] = now
        if now - last < policy.ALERT_REPEAT_SECONDS:
            log(f"(suppressed repeat) [{level}] {title}")
            self.last_alert = f"{gstate.now_iso()} {level}: {title} (repeat)"
            return
        self.last_alert = f"{gstate.now_iso()} {level}: {title}"
        log(f"ALERT [{level}] {title} :: {body[:200]}")
        try:
            self.notifier.alert(level, title, body, **kw)
        except Exception as exc:
            log(f"notifier failed (continuing): {exc}")

    # ── worker-pool silence ────────────────────────────────────────────
    def check_pool(self, snap: dict) -> None:
        """Alarm when the worker pool is not running while the backend answers
        ok (`poolwatch.py`). This is the alarm the pool cannot host: since #1682
        the dispatch and fleet watches live inside `WorkerPool._scheduler_loop`,
        so the routes that stop that loop — `workers.enabled: false` at boot,
        `POST /api/workers/enable {"enabled": false}` — stop the reporting of the
        problem along with the work.

        Placed with `check_vault`/`check_data`, above the `infra_down`, `broken`
        and `paused` early returns in `tick()`, for the reason their comment
        already gives: a check seated below those returns is a check that does not
        run in the states where things are going wrong.

        Fan-out is the existing one — `self.alert` → notifier → `ALERT.md`, the
        ledger and the daily note. The watermark that bounds the repeat is on disk
        in `gdir`, not in `_alert_seen`, because the silence outlives a guardian
        restart and that dict does not.
        """
        try:
            report = self.pool.tick(snap.get("backend_health"))
        except Exception as exc:  # noqa: BLE001 — same rule as the watches above
            log(f"poolwatch failed (continuing): {exc}")
            return
        alert = report.get("alert")
        if alert:
            self.alert(alert["level"], alert["title"], alert["body"],
                       evidence=alert["evidence"])
        elif report.get("reason") == "running":
            # The condition cleared, so the retraction goes on the surface the
            # alarm used (#1536). `resolve` is idempotent and writes nothing when
            # no section of ours is open, so the healthy tick after the first is
            # silent — and a pool that was never reported cannot be un-reported.
            self.notifier.resolve(
                poolwatch.ALERT_TITLE,
                "the worker pool reports running on the latest check — the "
                "instructions above are stale, nothing further to enable")

    def check_voice_loss(self) -> None:
        """Escalate `voice-loss.md` into one coalesced backlog item (`voiceloss.py`).

        Placed with `check_pool`/`check_tmp`, above the `infra_down`, `broken` and
        `paused` early returns, for the reason their comment already gives and the
        one this item names: the record most often exists because the speaker died
        during an outage, so a check seated below those returns would escalate it
        only when the stack looked healthy — which for this incident is never.

        Fan-out is the board, not `self.alert`: the alert already went out on five
        reliable channels and the sixth is the thing that failed. The watermark is
        the cursor on disk in `gdir`, not an attribute, so a guardian restart does
        not re-file an incident the board already carries. The log line names only
        the outcomes a human would want to see: `no-record` and `unchanged` are
        every healthy tick, and a tick is 5 seconds.
        """
        try:
            report = self.voiceloss.tick()
        except Exception as exc:  # noqa: BLE001 — same rule as the watches above
            log(f"voiceloss failed (continuing): {exc}")
            return
        reason = report.get("reason")
        if reason in ("created", "refreshed", "closed-suppressed"):
            after = (f", follows closed #{report['follows_closed']}"
                     if report.get("follows_closed") is not None else "")
            log(f"voice-loss escalation {reason}: backlog #{report.get('item_id')} "
                f"(occurrences {report.get('occurrences')}{after})")

    # ── memory-pressure evidence ───────────────────────────────────────
    def check_memory(self) -> None:
        """Record who holds the memory while pressure builds toward an oomd
        kill of the stack (`memwatch.py`). Evidence only: it acts on nothing
        and never raises into the tick."""
        try:
            path = self.mem.tick()
        except Exception as exc:  # noqa: BLE001
            log(f"memwatch failed (continuing): {exc}")
            return
        if path:
            log(f"memory pressure snapshot: {path}")

    # ── /tmp headroom ──────────────────────────────────────────────────
    def check_tmp(self) -> None:
        """Alert before /tmp's fixed inode budget runs out (`tmpwatch.py`).

        Both tree deletions happened with /tmp at 100% of its inodes, where
        every mkdir fails while `df -h` reads healthy. Latched in memory: one
        alert per crossing, one on escalation to critical, one clear below 70%.
        Deletes nothing and never raises into the tick."""
        try:
            action, level, body = self.tmp.tick()
        except Exception as exc:  # noqa: BLE001
            log(f"tmpwatch failed (continuing): {exc}")
            return
        if action == "alert":
            log(f"TMP HEADROOM ({level}): {body.splitlines()[0]}")
            self.notifier.alert(level, tmpwatch.ALERT_TITLE, body, coalesce=True)
        elif action == "resolve":
            log(f"tmp headroom recovered: {body}")
            self.notifier.resolve(tmpwatch.ALERT_TITLE, body)

    # ── vault tripwire ─────────────────────────────────────────────────
    def check_vault(self) -> None:
        """Trip on a mass deletion of the vault: stop sync, pause workers,
        halt promotions, keep evidence, alert. Never raises into the tick."""
        try:
            why, snap = self.vault.tick()
        except Exception as exc:  # noqa: BLE001
            log(f"vaultwatch failed (continuing): {exc}")
            return
        if not why:
            return
        log(f"VAULT TRIPWIRE: {why}")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        # The marker goes down FIRST: supervisord autorestarts a program that
        # dies, and `start-obsidian-sync.sh` refuses while the marker exists —
        # so even a sync that comes back before the stop below is gated.
        actions: dict = {"evidence": None}
        try:
            self.vault.trip(why, snap, actions)
        except OSError as exc:
            log(f"could not write vault marker: {exc}")
        actions["sync"] = self._stop_sync()
        actions["workers"] = self._pause_workers()
        try:
            self.state.set_halted(f"vault tripwire: {why}")
            actions["promotions"] = "halted"
        except Exception as exc:  # noqa: BLE001
            actions["promotions"] = f"halt failed: {exc}"
        actions["evidence"] = self._vault_evidence(stamp)
        try:
            marker = self.vault.trip(why, snap, actions)
        except OSError:
            marker = self.vault.marker
        before = self.vault.history[-1].total if self.vault.history else "?"
        after = snap.total if snap else "missing"
        gstate.append_event(self.state.ledger, {"event": "vault_tripwire", "reason": why,
                                                "files_before": before, "files_after": after,
                                                "actions": actions})
        self.alert(
            "critical", "Vault mass deletion — sync stopped",
            f"{why}\n\nFiles: {before} → {after}.\n"
            f"Obsidian Sync: {actions['sync']}\nWorker pool: {actions['workers']}\n"
            f"Promotions: {actions['promotions']}\nEvidence (process list at the "
            f"moment it tripped): {actions['evidence']}\n\n"
            "Restore from a snapshot into a side directory with "
            "`~/lloyd/scripts/backup/restore-vault.sh` (never in place), check it, "
            "swap it in, then clear the tripwire with\n"
            f"  /usr/bin/python3 {Path(vaultwatch.__file__).resolve()} clear\n"
            f"Sync will not start until then. Marker: {marker}")

    # ── data-root tripwire ─────────────────────────────────────────────
    def _runtime_data_incident(self, now: float) -> None:
        """One stray check: raise the alert, or retract it, for this tick.

        Split out of `check_data` so the incident's two edges are testable without
        booting a supervisor, a probe set and a rollback history — `check_data`
        calls this unmodified on the same tick it always did (#1536).

        The body is built here, from an `lstat` taken on this same call, and the
        window it may cite runs from the previous call of *this* method — recorded
        on the way in, so an alert can never print an interval it did not observe
        (#2057 clauses 1 and 4). Read with `getattr` because the tests build a
        `Guardian` with `__new__`; a first check has no previous one and says so
        rather than inventing a bound.

        Three outcomes, not two (#2056). A check either measures the tree or it does
        not, and the failure case used to be folded into the empty-set case: the
        `except` substituted `strays = []`, `elif not strays:` accepted that
        substitution as a measurement, and `resolve` sealed an incident the alert
        branch was re-raising an hour later. `memory/2026-10-02.md` is the record —
        three sections under this one title, two of them ending `cleared:`, against
        `journalctl --user -u lloyd-guardian` ALERT lines for the same title at
        01:48:27, 02:48:29 and 03:48:32 naming `workers.db`, whose inode has birth =
        ctime = mtime 2026-10-02 01:01:36.605 and never moved. The spurious seal is
        also why `coalesce` left three sections for one incident: `_daily_open_at`
        returns an open section only while it still ends in the still-open marker, and
        a clear had already replaced it.

        #2110 is the other half of the same sentence. A clear has always been allowed to
        say only one thing — "nothing further to move" — which is what a tree the
        guardian itself emptied would also say, and what the tree's `workers.db` was told
        when it was found gone with nothing in any journal having moved it. The retraction
        now carries the one fact this method actually holds: did THIS process move
        anything, or not. `moved` is therefore bound above the branch that fills it — the
        retraction is reachable on the SAME tick as a move that emptied the tree (the
        `strays = [...]` filter below and the `elif not strays:` after it are one tick),
        and a local read only in the retraction would be an unbound name on that path.
        """
        # `measured` is a flag rather than a sentinel value in `strays` because the
        # absence of a measurement must not be expressible as either a finding or an
        # all-clear. Set after the call, so an exception anywhere inside the detector
        # leaves it False.
        # Bound here, not inside the branch that fills it, for the reason in the
        # docstring: the retraction at the bottom has to say which of the two states it
        # is in, and a move that empties the tree reaches that retraction on THIS tick.
        moved: list[tuple[str, str]] = []
        measured = False
        try:
            strays = datawatch.stray_in_tree(policy.REPO)
            measured = True
        except Exception as exc:  # noqa: BLE001
            log(f"stray check failed (continuing): {exc}")
        prev_check = getattr(self, "_strays_prev_check_at", None)
        # Recorded on a failed check too: the window is "since the last time this
        # check ran", and the next check may only claim what has happened since then.
        self._strays_prev_check_at = now
        if not measured:
            # Neither edge speaks. Not an alarm, because an alarm asserts "these paths
            # exist" about a tree this check did not read, and not a clearance, which
            # asserts the same thing negatively — #1541 closed the name-list leg of
            # that reasoning on 2026-09-29 and this is the other leg. The section keeps
            # its `_(still open on the next check)_` marker, which is the true state of
            # the record: an open incident whose last measurement failed. The reason is
            # in the log line above.
            return
        if strays and self.data.armed:
            # Residue first: an empty, idle second copy of a store that lives in the
            # data root is moved there and reported as news, not as an incident — on
            # 2026-10-02 one such 0-byte `workers.db` alerted hourly for seven hours
            # and was parked on a human because a file git ignores gives a round no
            # diff to land. Whatever is left alerts exactly as before, and a tree the
            # move emptied falls through to the retraction below.
            try:
                moved.extend(datawatch.quarantine_inert(policy.REPO, strays,
                                                        policy.DATA_ROOT, now))
            except Exception as exc:  # noqa: BLE001
                log(f"stray quarantine failed (continuing): {exc}")
            if moved:
                gone = {name for name, _ in moved}
                strays = [s for s in strays if s not in gone]
                self.notifier.announce(
                    "Moved an empty stray out of the code tree",
                    "\n".join(f"{os.path.join(policy.REPO, name)} → {dest}"
                              for name, dest in moved)
                    + "\n\n0 bytes, idle, and the real store is in "
                    f"{policy.DATA_ROOT}; nothing was deleted.")
        if strays and self.data.armed:
            # `coalesce` is what keeps one incident to one section on the daily
            # note. This check runs every STRAY_CHECK_SECONDS (3600 s) and the
            # condition can outlast many of them, so the hourly cadence used to
            # append a whole new section per check — 21 copies of ONE incident in
            # memory/2026-09-25.md, each naming a different snapshot of the set
            # and none ever retracted. Guardian's own repeat guard cannot help:
            # ALERT_REPEAT_SECONDS is 900 s and 3600 > 900 always, so every
            # finding passed straight through (#1536).
            # Every sentence below is `_stray_alert_body`'s, from the `lstat` this
            # call takes — including the cause, which used to be a remembered
            # mechanism rather than a reading of these numbers, and the two offers,
            # which used to be unconditional and were both wrong for a retained
            # store (#2057).
            self.alert("error", RUNTIME_DATA_ALERT_TITLE,
                       _stray_alert_body(policy.REPO, strays, now, prev_check),
                       coalesce=True)
        elif not strays:
            # The condition cleared, so the retraction goes on the SAME surface
            # the alarm used — the acceptance half of #1536. Not gated on
            # `data.armed` the way the alert is: a guardian that paused mid-incident
            # should still close the section it opened, and `resolve` writes
            # nothing unless one of our sections is actually open. It is
            # idempotent, so the hourly all-clear after the first stays silent.
            #
            # `elif not strays` and not a bare `else`, which is what #2056 clause 2 is
            # about: the alert fires on `strays and self.data.armed`, and a finding
            # held by a disarmed guardian takes neither branch — a paused loop has not
            # closed the incident, it has stopped checking it, so it has earned no
            # statement in either direction. The failed-measurement case above is the
            # third outcome and it returned before reaching here.
            #
            # The line names the root it measured because "on the latest check" is not
            # auditable by the reader who finds it a day later: the two retractions in
            # `memory/2026-10-02.md` each asserted an empty tree and neither said which
            # one, while the path the alert branch was naming an hour later was in the
            # tree both of them claimed to have walked.
            #
            # And it now says which of the two kinds of empty this is (#2110). The
            # sentence used to be a flat all-clear — "nothing further to move" — and that
            # is exactly what a tree the guardian had just emptied also looks like, so the
            # retraction could not distinguish "this loop moved it" from "it is not here
            # and this loop did not do anything". The tree's `workers.db` was found gone with
            # the shipped move path never having run, which is the case the old line
            # described as a clean bill of health.
            if moved:
                cleared = (
                    f"no runtime stores inside the code tree of {policy.REPO} — the "
                    f"guardian moved {len(moved)} inert file"
                    f"{'s' if len(moved) != 1 else ''} on this check "
                    f"({', '.join(name for name, _ in moved)}); the instructions above "
                    "are stale, nothing further to move")
            else:
                cleared = (
                    f"no runtime stores inside the code tree of {policy.REPO} — absent "
                    "with no move recorded by the guardian, so the instructions above "
                    "are stale and nothing here accounts for a path that was named and "
                    "is now gone")
            self.notifier.resolve(RUNTIME_DATA_ALERT_TITLE, cleared)

    def check_data(self) -> None:
        """Trip on a wipe of `~/lloyd-data`: pause workers, halt promotions,
        keep evidence, alert. Hourly, also name any runtime path that came
        back into the code tree, and ask the snapshot directory how old its
        newest snapshot is — the layer the tripwire cannot watch, and the one
        that refuses without leaving a trace anywhere else. Never raises into
        the tick."""
        now = time.time()
        if now - self._strays_checked_at >= policy.STRAY_CHECK_SECONDS:
            self._strays_checked_at = now
            self._runtime_data_incident(now)
        # Is the hourly snapshot still arriving? That layer catches what the
        # tripwire cannot — a root eaten slowly enough never to trip it — and it
        # fails silently: both refusals in `scripts/backup/snapshot-data.sh`
        # `exit 0` by design, the unit is `Type=oneshot` and reports
        # `Result=success` after refusing, and pruning never deletes the newest
        # snapshot, so the entry count keeps looking alive (#1416). Not while the
        # data tripwire is set: refusing then is the intended behaviour and the
        # critical alert for it has already been paged.
        if (now - self._snapshots_checked_at >= policy.SNAPSHOT_CHECK_SECONDS
                and not self.data.tripped()
                and os.path.isfile(os.path.join(policy.DATA_ROOT, datawatch.ROOT_MARKER))):
            self._snapshots_checked_at = now
            try:
                snap_fresh, snap_why = datawatch.snapshot_report(
                    policy.DATA_SNAPSHOTS, policy.SNAPSHOT_MAX_AGE_SECONDS)
            except Exception as exc:  # noqa: BLE001
                log(f"snapshot freshness check failed (continuing): {exc}")
                snap_fresh, snap_why = True, ""
            if not snap_fresh:
                self.alert("error", "Data snapshots are not arriving",
                           f"{snap_why}.\n\nThe hourly read-only snapshot of {policy.DATA_ROOT} is "
                           "layer 2 under the data tripwire, and a stream that stopped is "
                           "invisible from the outside: a refusal exits 0 so the timer never "
                           "flaps, and `Type=oneshot` reports success either way. So the newest "
                           "stamp is the only evidence left that the timer is delivering at all — "
                           "refusing, disabled, masked or a machine that slept through the hour "
                           "all look the same from here. What it said is in the journal:\n"
                           "  journalctl --user -u lloyd-data-snapshot.service -n 40 --no-pager\n"
                           "The script refuses while the data tripwire is set, or when the root "
                           "shrank below its last healthy measurement. `~/lloyd/scripts/backup/"
                           "restore-data.sh` lists the store and prints the newest stamp's age.")
        try:
            why, snap = self.data.tick()
        except Exception as exc:  # noqa: BLE001
            log(f"datawatch failed (continuing): {exc}")
            return
        if not why:
            return
        log(f"DATA TRIPWIRE: {why}")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        actions: dict = {"evidence": None}
        try:
            self.data.trip(why, snap, actions)
        except OSError as exc:
            log(f"could not write data marker: {exc}")
        actions["workers"] = self._pause_workers()
        try:
            self.state.set_halted(f"data tripwire: {why}")
            actions["promotions"] = "halted"
        except Exception as exc:  # noqa: BLE001
            actions["promotions"] = f"halt failed: {exc}"
        actions["evidence"] = self._vault_evidence(stamp)
        try:
            marker = self.data.trip(why, snap, actions)
        except OSError:
            marker = self.data.marker
        before = self.data.history[-1].total if self.data.history else "?"
        after = snap.total if snap else "missing"
        gstate.append_event(self.state.ledger, {"event": "data_tripwire", "reason": why,
                                                "files_before": before, "files_after": after,
                                                "actions": actions})
        self.alert(
            "critical", "Lloyd data root damaged — promotions halted",
            f"{why}\n\nFiles: {before} → {after}.\nWorker pool: {actions['workers']}\n"
            f"Promotions: {actions['promotions']}\nEvidence: {actions['evidence']}\n\n"
            f"Hourly read-only snapshots are in {policy.DATA_SNAPSHOTS}. Restore into "
            "a side directory with `~/lloyd/scripts/backup/restore-data.sh` (never in "
            "place), check it, swap it in with the stack stopped, then clear with\n"
            f"  /usr/bin/python3 {Path(datawatch.__file__).resolve()} clear\n"
            f"Snapshots are refused until then. Marker: {marker}")

    def _stop_sync(self) -> str:
        try:
            ok, detail = self.sup.stop(policy.OBSIDIAN_SYNC_PROGRAM, wait=False)
            if ok:
                return "stopped via supervisord"
            outcome = f"supervisord stop failed ({detail})"
        except Exception as exc:  # noqa: BLE001 — supervisord may be the thing that is down
            outcome = f"supervisord unreachable ({exc})"
        proc = subprocess.run(["pkill", "-f", f"sync --path {policy.VAULT_ROOT}"],
                              capture_output=True, timeout=10, check=False)
        return f"{outcome}; pkill rc={proc.returncode}"

    def _pause_workers(self) -> str:
        import urllib.request
        req = urllib.request.Request(policy.WORKERS_PAUSE_URL, method="POST",
                                     data=b'{"paused": true}',
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return f"paused (HTTP {resp.status})"
        except Exception as exc:  # noqa: BLE001
            return f"pause failed ({exc})"

    def _vault_evidence(self, stamp: str) -> str | None:
        """The process table at the moment of the trip. On 2026-09-10 the
        culprit had exited and left nothing; a listing taken within one tick
        names what was running."""
        out = self.gdir / "vault-incidents" / stamp
        try:
            out.mkdir(parents=True, exist_ok=True)
            ps = subprocess.run(["ps", "-eo", "pid,ppid,etimes,args", "--sort=-etimes"],
                                capture_output=True, text=True, timeout=10, check=False)
            (out / "ps.txt").write_text(ps.stdout, encoding="utf-8")
            cwds = []
            for pid in os.listdir("/proc"):
                if not pid.isdigit():
                    continue
                try:
                    cwd = os.readlink(f"/proc/{pid}/cwd")
                except OSError:
                    continue
                if cwd.startswith(policy.VAULT_ROOT):
                    try:
                        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
                    except OSError:
                        cmd = b""
                    cwds.append(f"{pid}\t{cwd}\t{cmd.decode(errors='replace')[:300]}")
            (out / "cwd-in-vault.txt").write_text("\n".join(cwds) + "\n", encoding="utf-8")
            return str(out)
        except Exception as exc:  # noqa: BLE001
            log(f"vault evidence capture failed: {exc}")
            return None

    # ── settle ─────────────────────────────────────────────────────────
    def maybe_settle(self, current: dict) -> None:
        """Advance LKG once a promotion survives its full observation window."""
        errors_until = float(current.get("errors_until_ts") or 0)
        if not errors_until or time.time() < errors_until:
            return
        if current.get("state") == "landing":
            return
        commit = current.get("commit")
        if not commit:
            return
        mcp = probes.probe(self.mcp_url, policy.PROBE_TIMEOUT_SECONDS)
        degraded = ((mcp.get("body") or {}).get("degraded_modules") or [])

        # Fold in the last quality measurement, if one was taken for exactly
        # this commit. The LKG record has carried an empty `eval` slot since it
        # was designed; the guardian cannot run the eval itself (it is stdlib
        # only, on the system python) so the worker measures and leaves the
        # result here. Reading rather than letting the worker write LKG keeps
        # "the guardian is the only writer of last_known_good.json" true, which
        # is what makes LKG mean *observed healthy in production*.
        eval_baseline = None
        measured = self.state.read_eval_last()
        if measured and measured.get("commit") == commit:
            eval_baseline = measured

        self.state.set_lkg(commit, health={"mcp_degraded_modules": degraded},
                           eval_baseline=eval_baseline)
        # Record what settled BEFORE dropping current.json — a post-landing
        # check running hours later needs the promotion's own parent, and by
        # now the LKG pointer has advanced to the promoted commit itself.
        self.state.write_last_settled({
            "schema": 1,
            "commit": commit,
            "parent": current.get("parent") or current.get("rollback_target"),
            "round_id": current.get("round_id"),
            "changed_paths": current.get("changed_paths") or [],
            **({"commits": gstate.batch_commits(current)}
               if gstate.batch_commits(current) else {}),
            "landed_ts": current.get("landed_ts"),
            "settled_ts": time.time(),
            "settled_at": gstate.now_iso(),
        })
        self.state.clear_current()
        prev = Path(self.repo) / ".venvs" / "lloyd.prev"
        if current.get("venv_swapped") and prev.exists():
            import shutil
            shutil.rmtree(prev, ignore_errors=True)
        batch = gstate.batch_commits(current)
        if batch and batch[-1] == commit:
            # One row per commit, oldest first: everything that joins a
            # landing to its settle (`backlog.close_settled_items`, the
            # promoter's `_settled`) reads one commit per row.
            for c in batch:
                gstate.append_event(self.state.ledger, {
                    "event": "settled", "commit": c, "batch": len(batch),
                    "batch_head": commit})
        else:
            gstate.append_event(self.state.ledger, {"event": "settled", "commit": commit})
        log(f"settled: last known good is now {commit[:8]}"
            + (f" ({len(batch)} landings)" if batch else ""))

    # ── selftest ───────────────────────────────────────────────────────
    def maybe_selftest(self) -> None:
        # A failing selftest is re-asked on the short clock: the heartbeat
        # publishes `selftest_ok`, and a verdict from a boot race must not stand
        # for a day (see policy.SELFTEST_RETRY_SECONDS).
        now = time.time()
        interval = (policy.SELFTEST_RETRY_SECONDS if self.selftest_ok is False
                    else policy.SELFTEST_INTERVAL_SECONDS)
        if now - self.last_selftest < interval:
            return
        self.last_selftest = now
        was = self.selftest_ok
        failures: list = []
        try:
            import selftest
            self.selftest_ok = selftest.run(self, verbose=False, failures=failures)
        except Exception as exc:
            self.selftest_ok = False
            failures.append(("selftest raised", f"{type(exc).__name__}: {exc}"))
            log(f"selftest raised: {exc}")
        # Named here, not only at the page: the detail is what says whether a
        # failure is a boot race or a guardian fault (#1178).
        cause = "\n".join(f"- {n}: {d}" for n, d in failures) or "- (no check reported a detail)"
        if self.selftest_ok:
            if was is False:
                log("selftest: passing again")
            self._selftest_alerted = 0.0
            return
        if now - self.started_ts < policy.SELFTEST_BOOT_GRACE_SECONDS:
            log(f"selftest failed {now - self.started_ts:.0f}s after start "
                f"(boot grace {policy.SELFTEST_BOOT_GRACE_SECONDS:.0f}s); "
                f"retrying in {policy.SELFTEST_RETRY_SECONDS:.0f}s: "
                + "; ".join(f"{n}: {d}" for n, d in failures))
            return
        # One page per failure episode, repeated daily while it lasts — the
        # cadence the old 24 h check gave, now that the check runs every retry.
        if now - self._selftest_alerted < policy.SELFTEST_INTERVAL_SECONDS:
            return
        self._selftest_alerted = now
        self.alert("error", "Guardian self-test failed",
                   "The watchdog can no longer perform one of its own preconditions. "
                   "It is still running but may not be able to act.\n\n"
                   f"Failing check(s):\n{cause}",
                   needs_human=True, evidence=cause)

    # ── main loop ──────────────────────────────────────────────────────
    def tick(self) -> str:
        self.tick_n += 1
        snap = self.collect()
        # Before any early return below. Every `return` in this method that
        # skips this would re-create the staleness bug it exists to prevent.
        self.drain_logs()
        # Same placement rule, for a stronger reason: a vault wipe must be
        # caught while paused, while BROKEN, with supervisord unreachable and
        # with nothing under observation — every state the returns below mean.
        self.check_vault()
        self.check_data()
        # Also above the returns, and for the item's own reason: the state this
        # watch exists for is a supported configuration that reports nothing, and
        # `paused` in particular is a state the pool-silence reading must survive —
        # the vault tripwire pauses workers, and a pool stopped by `workers.enabled`
        # while the guardian is paused would otherwise never be noticed by anyone.
        self.check_pool(snap)
        # And again: pressure building while supervisord is unreachable or the
        # stack is BROKEN is the moment the evidence is for.
        self.check_memory()
        # And /tmp, for the same reason: a full /tmp is a state the stack
        # cannot report from, because nothing in it can create a file.
        self.check_tmp()
        # And the dead speaker's record, for the item's own reason: the burst this
        # escalates is most likely to have happened while the stack was
        # unreachable, so a check below the returns above would file it exactly
        # when it had nothing to file.
        self.check_voice_loss()

        if snap["supervisord"] == "unreachable":
            self.sup_down_streak += 1
            if self.sup_down_streak >= policy.SUPERVISORD_DOWN_STREAK:
                # Invariant 2: never a code trigger.
                log("supervisord unreachable — restarting the unit, NOT rolling back code")
                res = subprocess.run(
                    ["systemctl", "--user", "restart", policy.SUPERVISORD_UNIT],
                    capture_output=True, timeout=60, check=False,
                )
                # The body says what was done; the evidence says what was seen.
                # Until #1178 every row for this title carried `evidence: ""`,
                # so the 2026-09-15 oomd kill of the whole unit read the same
                # as a slow socket. The probe error is the socket's own words
                # and the restart rc says whether the remedy took.
                # getattr, not attribute access: evidence is a nicety, and an
                # alert that raised here would cost the restart's own report.
                stderr = (getattr(res, "stderr", None) or b"")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", "replace")
                stderr = str(stderr).strip()
                evidence = (
                    f"- unreachable for {self.sup_down_streak} consecutive ticks "
                    f"(threshold {policy.SUPERVISORD_DOWN_STREAK}, tick {self.interval:g}s)\n"
                    f"- last probe error: {snap.get('supervisord_error') or '(none recorded)'}\n"
                    f"- systemctl --user restart {policy.SUPERVISORD_UNIT}: rc={getattr(res, 'returncode', '?')}"
                    + (f" stderr={stderr[:300]}" if stderr else "")
                )
                # coalesce=True, and that option IS #2221 clause 3: a section written
                # without it carries no open marker, and `resolve` seals only a body that
                # ends with one, so the 7 `supervisord was unreachable` blocks in the dated
                # notes were un-retractable by construction, not by a forgotten call.
                self.alert("error", "supervisord was unreachable",
                           "Restarted agent-supervisord.service. No code was reverted — "
                           "an unreachable supervisor is infrastructure, not a bad promotion.",
                           evidence=evidence, coalesce=True)
                self._sup_unreachable_open = True
                self.sup_down_streak = 0
            return "infra_down"
        elif self._sup_unreachable_open:
            # The tick the supervisor ANSWERED. Retracted here rather than with the
            # liveness retraction below because this alert's subject is the supervisor
            # socket and nothing else: a retraction written after a service check would
            # leave a reader unable to tell which recovery it named.
            self._sup_unreachable_open = False
            self.notifier.resolve(
                "supervisord was unreachable",
                "agent-supervisord answered again on the next tick.")
        self.sup_down_streak = 0

        if self.state.is_broken():
            return "broken"

        paused = self.state.pause_remaining(policy.PAUSE_MAX_SECONDS)
        live_down, live_reason = self.evaluate_liveness(snap)

        # #2221 clause 4's age. One increment per liveness read, which is the only clock
        # this loop may quote in a note: a wall-clock timestamp in an alert body reads as
        # "the process is down now", while `FATAL` is a state supervisord keeps after its
        # own restart, so the number of consecutive failing reads is what tells a reader
        # whether the state in the body is current or inherited.
        if live_down:
            self.liveness_fail_streak += 1
        else:
            self.liveness_fail_streak = 0

        if self._down_programs and not live_down:
            # The first tick whose liveness read came back not-down, put here at the read
            # itself so nothing between the observation and the retraction can reorder
            # them. Every branch of the `live_down` block below returns, so arriving at this
            # line with the set non-empty is what "the incident that alert was written for
            # has ended" means, and the loop has no other witness of it; without it the seal
            # waits for a LATER incident's recovery, which for `memory/2026-09-29.md`'s
            # FATAL block has meant a standing "this needs a human" instruction ever since.
            # It runs while PAUSED as well, deliberately: a pause suppresses actions on a
            # recovery, not the fact of one, and a note that keeps asserting a down service
            # through a pause is the artefact this clause is about.
            recovered = ", ".join(sorted(self._down_programs))
            self._down_programs = set()
            self.notifier.resolve(
                "Service down, but no promotion to revert",
                f"liveness read came back healthy: {recovered}.")

        if paused > 0:
            if live_down:
                log(f"[paused {paused:.0f}s] would have fired: {live_reason}")
            # Discard, do not merely skip. The promoter holds this pause across
            # its own supervisord restart, so what is in the buffer right now
            # is the deploy's own connection failures — and the observation
            # window for that very deploy opens seconds later. Leaving them
            # buffered would hand the new commit the noise its own landing
            # made. The cursor has already advanced past them.
            self._tick_events = []
            self._tick_overflow = False
            return "paused"

        # A rollback somebody else needs performed. The backend and the
        # aggregator are both inside the blast radius of a rollback — it stops
        # them — so neither can carry one out inline without dying partway
        # through. They write a request; this is the one process that survives
        # the operation, and it already owns evidence preservation, retries,
        # the denylist and flap protection.
        request = self.state.read_rollback_request()
        if request:
            self.state.clear_rollback_request()
            age = time.time() - float(request.get("ts") or 0)
            if age > policy.ROLLBACK_REQUEST_MAX_AGE_SECONDS:
                # Obeying a stale request means acting on state that has moved
                # on since it was written. Say so rather than silently dropping.
                log(f"discarding rollback request {age:.0f}s old")
                self.alert("warn", "Stale rollback request discarded",
                           f"A rollback request written {age:.0f}s ago was not acted on "
                           f"(limit {policy.ROLLBACK_REQUEST_MAX_AGE_SECONDS:.0f}s).\n"
                           f"Trigger: {request.get('trigger')}\n{request.get('reason')}")
            else:
                log(f"rollback requested by pid {request.get('pid')}: "
                    f"{request.get('trigger')}")
                self.do_rollback(request.get("trigger") or "requested",
                                 request.get("reason") or "(no reason given)",
                                 explicit_target=request.get("target"),
                                 explicit_commit=request.get("commit"),
                                 explicit_changed=request.get("changed_paths"),
                                 explicit_commits=request.get("commits"))
                return "rolling_back"

        current = self.state.current()
        # A record still in `landing` means the promoter is mid-flight (it may
        # be waiting on the idle gate). Nothing has been deployed yet, so there
        # is nothing to observe and nothing to revert.
        if current and current.get("state") == "landing":
            current = None

        unrestarted = bool(current) and current.get("restart") is False
        if live_down:
            # #2256: a program a landing cannot have broken gets restarted, not
            # blamed. Ordered ahead of the rollback route below because that route is
            # the WRONG tool for it: `do_rollback("crash", ...)` stops and starts
            # `RESTART_ORDER`, in which `agent-tts` deliberately is not, and it rewrites
            # main to fix a room that went quiet. On 2026-10-05 the synthesiser sat
            # STOPPED with nobody owning it for 45 minutes while three alerts went
            # unspoken, and if a promotion had been under observation the guardian would
            # have reverted it for a TTS stop it did not cause.
            if self._recoverable_down(live_reason):
                return self._recover_infra(live_reason)

            # Rollback is only ever appropriate for a commit the LOOP promoted
            # and is still observing. With no `current.json` the tree moved for
            # some other reason — a human commit, a nightly job — and reverting
            # that would destroy work nobody asked us to judge. Same reasoning
            # as invariant 1 (HEAD == LKG never rolls back), and it is the case
            # that actually bites: HEAD legitimately differs from LKG most of
            # the time.
            # ...and one that replaced a running process. A landing whose
            # files neither service had loaded restarts nothing
            # (`promote.restart_needed`, `restart: false` on the record): the
            # code that just went down is the code that was running before it
            # landed, so reverting the commit cannot be what brings it back.
            if not current or unrestarted:
                log(f"liveness failure with nothing under observation: {live_reason}")
                self.alert("error", "Service down, but no promotion to revert",
                           f"{self._describe_down(live_reason)}\n\nHEAD is "
                           f"{(rb.head_commit(self.repo) or '?')[:8]} and "
                           + ("the promotion under observation restarted no service — "
                              "the running code predates it — "
                              if unrestarted else "no self-modification is being observed, ")
                           + "so this is infrastructure rather than a bad "
                           "change. Not rewriting history — this needs a human.",
                           needs_human=True, coalesce=True)
                # Recorded after the alert call, not before: `Notifier.alert` swallows its
                # own write failures, so an incident whose note was never written must not
                # gain a retraction on a later tick — that would report the recovery of
                # something the note never carried.
                self._down_programs = set(self._down_programs) | \
                    self._down_program_names(live_reason)
                return "down_unobserved"
            log(f"liveness failure: {live_reason}")
            self.do_rollback("crash", live_reason)
            return "rolling_back"

        if not current:
            return "armed"

        if time.time() < self.quiet_until:
            return "quiet"

        errors_until = float(current.get("errors_until_ts") or 0)
        if errors_until and time.time() < errors_until:
            # The error log is the running services' log, and an unrestarted
            # landing changed nothing they run. Data damage is still judged: a
            # script the landing changed can be run by a job inside the window.
            spiked, why = (False, "") if unrestarted else self.evaluate_errors(current)
            if spiked:
                log(f"error-rate failure: {why}")
                self.do_rollback("error_rate", why)
                return "rolling_back"
            damaged, why = self.evaluate_data_damage(current)
            if damaged:
                log(f"data damage: {why}")
                self.do_rollback("data_damage", why)
                return "rolling_back"
            # The unreadable verdict is not damage and must not roll anything
            # back, but it is also not a healthy store, and until here it was
            # silent either way: the reason came back in `why` and this call site
            # read it only under `if damaged:`, which is the shape of the original
            # #1525 complaint — a store the watchdog could not open presented
            # itself as an intact one. `count_kg_rows`'s docstring promises the
            # path reaches the log; this is the line that keeps that promise.
            if why.startswith(KG_UNREADABLE_MARK):
                log(f"data check inconclusive: {why}")
            return "observing"

        self.maybe_settle(current)
        return "armed"

    def run(self) -> int:
        log(f"guardian starting: repo={self.repo} programs={self.programs}")
        sd_notify("READY=1")

        pending = self.state.unfinished_rollback()
        if pending:
            log(f"resuming interrupted rollback to {str(pending.get('to'))[:8]}")
            self.do_rollback("resume", f"resumed after guardian restart: {pending.get('reason','')}")

        state = "armed"
        while True:
            try:
                state = self.tick()
            except Exception as exc:
                log(f"tick error (continuing): {type(exc).__name__}: {exc}")
                state = "error"
            self.heartbeat(state)
            sd_notify("WATCHDOG=1")
            self.maybe_selftest()
            time.sleep(self.interval)


def count_kg_rows(db_path: str) -> tuple[int | None, str]:
    """Total rows across the knowledge-graph store, read-only, and what was read.

    The second element is the store that was opened, or `""` when there was no
    path to open at all. It exists because a bare `None` could not tell an empty
    graph from no graph, and the caller reported either as "data intact" (#1525):
    the path is now in the log line, so a moved root cannot present itself as a
    healthy store.

    Still the guardian's own read-only handle, not `app.kg_store` — the watchdog
    runs system python on a staged snapshot and cannot import it. Unifying this
    counter with the promoter's (`scripts/automod/promote.py::count_kg_rows`),
    which resolve the same file by two different rules, is the follow-up #1525
    records rather than does.
    """
    import sqlite3
    if not db_path:
        return None, ""
    if not Path(db_path).exists():
        return None, str(db_path)
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
        try:
            names = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")]
            return sum(con.execute(f"SELECT count(*) FROM '{n}'").fetchone()[0]
                       for n in names), str(db_path)
        finally:
            con.close()
    except Exception:
        return None, str(db_path)


def count_vault_files(root: str) -> int | None:
    """Note files under the vault, counted the way the vault tripwire counts.

    `.git/**` is excluded (`vaultwatch.SKIP_DIRS`): git packs loose objects by
    the hundred, and a plain `rglob` read two such repacks as data loss —
    2026-09-09 22:01 (#537) and 2026-09-17 08:59 (#1206, 6075 → 5667 with the
    note count unchanged at 5622). `scripts/automod/promote.py` loads this
    same module for the pre-promotion count, so the two sides of the
    comparison share one definition.
    """
    snap = vaultwatch.measure(root)
    return None if snap is None else snap.total


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lloyd self-modification guardian")
    p.add_argument("--repo", default=policy.REPO)
    p.add_argument("--state", default=str(policy.AUTOMOD_STATE))
    p.add_argument("--guardian-state", default=str(policy.GUARDIAN_STATE))
    p.add_argument("--supervisor-sock", default=policy.SUPERVISOR_SOCK)
    p.add_argument("--backend-url", default=policy.BACKEND_HEALTH_URL)
    p.add_argument("--mcp-url", default=policy.MCP_HEALTH_URL)
    p.add_argument("--programs", default=",".join(policy.WATCHED))
    p.add_argument("--interval", type=float, default=policy.TICK_SECONDS)
    p.add_argument("--once", action="store_true", help="single tick, then exit")
    p.add_argument("--no-external-alerts", action="store_true",
                   help="ledger and ALERT.md only — no vault note, desktop "
                        "notification or backlog task. Used by the drill so a "
                        "rehearsal cannot look like a production incident.")
    p.add_argument("--selftest", action="store_true", help="run the self-check and exit")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    g = Guardian(args)
    if args.selftest:
        import selftest
        ok = selftest.run(g, verbose=True)
        return 0 if ok else 1
    if args.once:
        state = g.tick()
        g.heartbeat(state)
        log(f"single tick → {state}")
        return 0
    return g.run()


if __name__ == "__main__":
    sys.exit(main())
