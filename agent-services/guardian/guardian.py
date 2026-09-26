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
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import detect            # noqa: E402
import gstate            # noqa: E402
import logtail           # noqa: E402
import notify as notify_mod  # noqa: E402
import policy            # noqa: E402
import probes            # noqa: E402
import rollback as rb    # noqa: E402
import vaultwatch        # noqa: E402
import datawatch         # noqa: E402
import memwatch          # noqa: E402
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


class Guardian:
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
        self._snapshots_checked_at = 0.0
        self.mem = memwatch.MemWatch(self.gdir, memwatch.unit_cgroup(policy.SUPERVISORD_UNIT))

        self.tick_n = 0
        self._tick_events: list[dict] = []
        self._tick_overflow = False
        self.probe_fail: dict[str, int] = {p: 0 for p in self.programs}
        self.probe_timeout: dict[str, int] = {p: 0 for p in self.programs}
        self.probe_http: dict[str, int] = {p: 0 for p in self.programs}
        self.mcp_fatal_streak = 0
        self.start_history: dict[str, list[float]] = {p: [] for p in self.programs}
        self.sup_down_streak = 0
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
        if program.endswith("lloyd-backend"):
            return self.backend_url
        if program.endswith("lloyd-mcp"):
            return self.mcp_url
        return None

    def collect(self) -> dict:
        """One immutable snapshot per tick."""
        snap: dict = {"now": time.time(), "supervisord": "ok", "procs": {}, "probes": {}}
        try:
            snap["procs"] = self.sup.all_process_info()
        except SupervisordUnreachable as exc:
            snap["supervisord"] = "unreachable"
            snap["supervisord_error"] = str(exc)[:200]
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
        return snap

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
                intentional_stop=self.state.is_halted(),
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
                   body + "\n\nServices left stopped. Clear "
                          f"{self.state.broken} once resolved.")

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
        """
        try:
            strays = datawatch.stray_in_tree(policy.REPO)
        except Exception as exc:  # noqa: BLE001
            log(f"stray check failed (continuing): {exc}")
            strays = []
        if strays and self.data.armed:
            # `coalesce` is what keeps one incident to one section on the daily
            # note. This check runs every STRAY_CHECK_SECONDS (3600 s) and the
            # condition can outlast many of them, so the hourly cadence used to
            # append a whole new section per check — 21 copies of ONE incident in
            # memory/2026-09-25.md, each naming a different snapshot of the set
            # and none ever retracted. Guardian's own repeat guard cannot help:
            # ALERT_REPEAT_SECONDS is 900 s and 3600 > 900 always, so every
            # finding passed straight through (#1536).
            self.alert("error", RUNTIME_DATA_ALERT_TITLE,
                       "These exist inside the tree again:\n  "
                       + "\n  ".join(f"{policy.REPO}/{n}" for n in strays)
                       + f"\n\nSomething still resolves a data path off the code "
                       f"instead of app.paths.DATA_ROOT ({policy.DATA_ROOT}). Find the "
                       "writer, move the data across, and remove the in-tree copy."
                       # The widened check (#1541) reports anything at the top of
                       # the tree git does not track, so one of these names may be
                       # tooling or a rebuildable cache rather than a writer, and
                       # "remove the in-tree copy" is the wrong order for it. The
                       # residual of an open-set check is a human deciding which
                       # side of the list a new name is on; say so where they read
                       # it, or the honest response to a new `.mypy_cache` is to
                       # switch the check off.
                       + "\n\nIf one of these is tooling or a rebuildable cache and "
                       "not a writer, it belongs in KNOWN_GOOD_TOPLEVEL in "
                       "agent-services/guardian/datawatch.py — adding its name there "
                       "is what stops this alert; deleting the directory is not.",
                       coalesce=True)
        elif not strays:
            # The condition cleared, so the retraction goes on the SAME surface
            # the alarm used — the acceptance half of #1536. Not gated on
            # `data.armed` the way the alert is: a guardian that paused mid-incident
            # should still close the section it opened, and `resolve` writes
            # nothing unless one of our sections is actually open. It is
            # idempotent, so the hourly all-clear after the first stays silent.
            self.notifier.resolve(
                RUNTIME_DATA_ALERT_TITLE,
                "no runtime stores inside the code tree on the latest check — the "
                "instructions above are stale, nothing further to move")

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
        # And again: pressure building while supervisord is unreachable or the
        # stack is BROKEN is the moment the evidence is for.
        self.check_memory()

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
                self.alert("error", "supervisord was unreachable",
                           "Restarted agent-supervisord.service. No code was reverted — "
                           "an unreachable supervisor is infrastructure, not a bad promotion.",
                           evidence=evidence)
                self.sup_down_streak = 0
            return "infra_down"
        self.sup_down_streak = 0

        if self.state.is_broken():
            return "broken"

        paused = self.state.pause_remaining(policy.PAUSE_MAX_SECONDS)
        live_down, live_reason = self.evaluate_liveness(snap)
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
                           f"{live_reason}\n\nHEAD is "
                           f"{(rb.head_commit(self.repo) or '?')[:8]} and "
                           + ("the promotion under observation restarted no service — "
                              "the running code predates it — "
                              if unrestarted else "no self-modification is being observed, ")
                           + "so this is infrastructure rather than a bad "
                           "change. Not rewriting history — this needs a human.",
                           needs_human=True)
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
