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
from supervisor import SupervisorClient, SupervisordUnreachable  # noqa: E402


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


class Guardian:
    def __init__(self, args):
        self.repo = args.repo
        self.state = gstate.SelfModState(Path(args.state))
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
        )
        self._alert_seen: dict[str, float] = {}
        self.cursor = logtail.LogCursor(self.gdir / "logcursors.json")

        self.tick_n = 0
        self.probe_fail: dict[str, int] = {p: 0 for p in self.programs}
        self.probe_timeout: dict[str, int] = {p: 0 for p in self.programs}
        self.start_history: dict[str, list[float]] = {p: [] for p in self.programs}
        self.sup_down_streak = 0
        self.quiet_until = 0.0
        self.last_selftest = 0.0
        self.selftest_ok: bool | None = None
        self.chronic: set[str] = set()
        self._chronic_loaded = False
        self.last_alert = ""

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
            if result is not None:
                if result["ok"]:
                    self.probe_fail[program] = 0
                    self.probe_timeout[program] = 0
                elif result.get("kind") == "timeout":
                    # Alive but slow. Counted, but on its own long budget.
                    self.probe_timeout[program] = self.probe_timeout.get(program, 0) + 1
                    self.probe_fail[program] = 0
                else:
                    self.probe_fail[program] = self.probe_fail.get(program, 0) + 1

            grace = policy.BOOT_GRACE.get(program, policy.DEFAULT_BOOT_GRACE)
            down, reason = detect.process_down(
                info,
                now=snap["now"],
                grace=grace,
                probe_fail_streak=self.probe_fail.get(program, 0),
                probe_threshold=policy.PROBE_FAIL_STREAK,
                probe_timeout_streak=self.probe_timeout.get(program, 0),
                probe_timeout_threshold=policy.PROBE_TIMEOUT_STREAK,
                start_history=self.start_history.get(program, []),
                crash_loop_starts=policy.CRASH_LOOP_STARTS,
                crash_loop_window=policy.CRASH_LOOP_WINDOW_SECONDS,
            )
            if down:
                return True, f"{program}: {reason}"

            # A degraded aggregator is usually an external-app bridge being
            # absent, not a bad promotion. Only newly-degraded modules count.
            if program.endswith("lloyd-mcp") and result and result["status"] == 503:
                body = result.get("body") or {}
                baseline = ((self.state.lkg() or {}).get("health") or {}).get("mcp_degraded_modules")
                fatal, why = detect.mcp_degraded_is_fatal(body, baseline)
                if fatal:
                    return True, f"{program}: {why}"
        return False, "all watched processes healthy"

    # ── error-rate ─────────────────────────────────────────────────────
    def ensure_chronic(self) -> None:
        if self._chronic_loaded:
            return
        cache = self.gdir / "signatures.json"
        cached = gstate.read_json(cache)
        if cached and isinstance(cached.get("chronic"), list):
            self.chronic = set(cached["chronic"])
        else:
            self.chronic = logtail.bootstrap_chronic(
                list(policy.LOG_FILES),
                max_bytes=20 * 1024 * 1024,
                min_distinct_hours=policy.CHRONIC_MIN_DISTINCT_HOURS,
            )
            gstate.write_json_atomic(cache, {
                "chronic": sorted(self.chronic), "built_at": gstate.now_iso(),
            })
        self._chronic_loaded = True
        log(f"chronic signature set: {len(self.chronic)} entries (never trigger)")

    def evaluate_errors(self, current: dict) -> tuple[bool, str]:
        self.ensure_chronic()
        changed = current.get("changed_paths") or []
        events: list[dict] = []
        overflowed = False
        for path in policy.LOG_FILES:
            text, over = self.cursor.read_new(path, policy.LOG_READ_CAP_BYTES)
            overflowed = overflowed or over
            if text:
                events.extend(detect.extract_events(text))
        self.cursor.save()
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
        rows_now = count_kg_rows(policy.KG_DB)
        files_now = count_vault_files(policy.VAULT_ROOT)
        hit, why = detect.data_damage(before_rows, rows_now, policy.DATA_DROP_FRACTION)
        if hit:
            return True, f"knowledge graph rows {why}"
        hit, why = detect.data_damage(before_files, files_now, policy.DATA_DROP_FRACTION)
        if hit:
            return True, f"vault files {why}"
        return False, "data intact"

    # ── rollback ───────────────────────────────────────────────────────
    def do_rollback(self, trigger: str, reason: str) -> bool:
        target, source = self.state.rollback_target()
        head = rb.head_commit(self.repo)
        floor = self.state.floor()

        if not target:
            self.escalate("no rollback target", f"{reason}\n\n{source}")
            return False
        if target == head:
            # Invariant 1. Nothing was promoted; this is infrastructure.
            self.alert("error", "Failure with nothing to revert",
                       f"{reason}\nHEAD is already the last known good ({target[:8]}). "
                       "Not rewriting history — this needs a human.")
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
        })

        for attempt in range(1, policy.ROLLBACK_MAX_ATTEMPTS + 1):
            try:
                self._rollback_once(target, stamp, trigger, reason)
                return True
            except Exception as exc:
                log(f"rollback attempt {attempt} failed: {exc}")
                if attempt < policy.ROLLBACK_MAX_ATTEMPTS:
                    time.sleep(policy.ROLLBACK_RETRY_SECONDS)

        gstate.append_event(self.state.ledger, {
            "event": "rollback_failed", "trigger": trigger, "to": target,
        })
        self.escalate("ROLLBACK FAILED", reason)
        return False

    def _rollback_once(self, target: str, stamp: str, trigger: str, reason: str) -> None:
        head_before = rb.head_commit(self.repo)
        current = self.state.current() or {}
        boot_before = current.get("boot_id")

        # 3. Stop the writers first — see the module docstring.
        for program in reversed(policy.RESTART_ORDER):
            ok, msg = self.sup.stop(program, wait=True)
            log(f"stop {program}: {msg}")
        holders = rb._drain_writers(self.repo, policy.WRITER_DRAIN_SECONDS)
        if holders:
            log(f"warning: pids still in the repo after stop: {holders}")

        # 4-5. Quiesce, then preserve the evidence before destroying it.
        rb._wait_for_index_lock(self.repo, policy.INDEX_LOCK_STALE_SECONDS)
        tag = f"guardian-broken-{stamp}"
        evidence = rb.preserve_evidence(self.repo, self.state.broken_dir / stamp, tag)
        log(f"preserved: tag={evidence['tag']} stash={evidence['stash']}")

        # 6-8. Move the tree, verify it, undo any venv swap.
        rb.restore_tree(self.repo, target, policy.CLEAN_PATHS, policy.PYCACHE_PATHS)
        rb.verify_tree(self.repo, target)
        if current.get("venv_swapped"):
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
                healthy, last = probes.wait_healthy(url, budget, policy.PROBE_TIMEOUT_SECONDS)
                if not healthy:
                    raise rb.RollbackError(
                        f"{program} unhealthy after restart: "
                        f"status={last.get('status')} err={last.get('error')}")

        final = probes.probe(self.backend_url, policy.PROBE_TIMEOUT_SECONDS)
        body = final.get("body") or {}
        if body.get("commit") and body["commit"] != target:
            raise rb.RollbackError(
                f"backend reports commit {body['commit'][:8]}, expected {target[:8]} "
                "— the restart did not pick up the reverted code")
        if boot_before and body.get("boot_id") == boot_before:
            raise rb.RollbackError("backend boot_id unchanged — process was never replaced")

        # 11-12. Record, denylist, re-arm quiet.
        if head_before:
            self.state.deny(head_before)
        self.state.clear_current()
        gstate.append_event(self.state.ledger, {
            "event": "rollback_succeeded", "trigger": trigger, "commit": head_before,
            "restored": target, "tag": tag, "stash": evidence.get("stash"),
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

        self.alert(
            "critical" if extra else "warn",
            f"Rolled back {(head_before or '?')[:8]} → {target[:8]}",
            f"Trigger: {trigger}\n{reason}{extra}",
            evidence=json.dumps(evidence, indent=2),
            commit=head_before or "", trigger=trigger, tag=tag,
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
        self.state.set_lkg(commit, health={"mcp_degraded_modules": degraded})
        self.state.clear_current()
        prev = Path(self.repo) / ".venvs" / "lloyd.prev"
        if current.get("venv_swapped") and prev.exists():
            import shutil
            shutil.rmtree(prev, ignore_errors=True)
        gstate.append_event(self.state.ledger, {"event": "settled", "commit": commit})
        log(f"settled: last known good is now {commit[:8]}")

    # ── selftest ───────────────────────────────────────────────────────
    def maybe_selftest(self) -> None:
        if time.time() - self.last_selftest < policy.SELFTEST_INTERVAL_SECONDS:
            return
        self.last_selftest = time.time()
        try:
            import selftest
            self.selftest_ok = selftest.run(self, verbose=False)
        except Exception as exc:
            self.selftest_ok = False
            log(f"selftest raised: {exc}")
        if not self.selftest_ok:
            self.alert("error", "Guardian self-test failed",
                       "The watchdog can no longer perform one of its own preconditions. "
                       "It is still running but may not be able to act.")

    # ── main loop ──────────────────────────────────────────────────────
    def tick(self) -> str:
        self.tick_n += 1
        snap = self.collect()

        if snap["supervisord"] == "unreachable":
            self.sup_down_streak += 1
            if self.sup_down_streak >= policy.SUPERVISORD_DOWN_STREAK:
                # Invariant 2: never a code trigger.
                log("supervisord unreachable — restarting the unit, NOT rolling back code")
                subprocess.run(
                    ["systemctl", "--user", "restart", policy.SUPERVISORD_UNIT],
                    capture_output=True, timeout=60, check=False,
                )
                self.alert("error", "supervisord was unreachable",
                           "Restarted agent-supervisord.service. No code was reverted — "
                           "an unreachable supervisor is infrastructure, not a bad promotion.")
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
            return "paused"

        current = self.state.current()
        # A record still in `landing` means the promoter is mid-flight (it may
        # be waiting on the idle gate). Nothing has been deployed yet, so there
        # is nothing to observe and nothing to revert.
        if current and current.get("state") == "landing":
            current = None

        if live_down:
            # Rollback is only ever appropriate for a commit the LOOP promoted
            # and is still observing. With no `current.json` the tree moved for
            # some other reason — a human commit, a nightly job — and reverting
            # that would destroy work nobody asked us to judge. Same reasoning
            # as invariant 1 (HEAD == LKG never rolls back), and it is the case
            # that actually bites: HEAD legitimately differs from LKG most of
            # the time.
            if not current:
                log(f"liveness failure with nothing under observation: {live_reason}")
                self.alert("error", "Service down, but no promotion to revert",
                           f"{live_reason}\n\nHEAD is "
                           f"{(rb.head_commit(self.repo) or '?')[:8]} and no self-modification "
                           "is being observed, so this is infrastructure rather than a bad "
                           "change. Not rewriting history — this needs a human.")
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
            spiked, why = self.evaluate_errors(current)
            if spiked:
                log(f"error-rate failure: {why}")
                self.do_rollback("error_rate", why)
                return "rolling_back"
            damaged, why = self.evaluate_data_damage(current)
            if damaged:
                log(f"data damage: {why}")
                self.do_rollback("data_damage", why)
                return "rolling_back"
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


def count_kg_rows(db_path: str) -> int | None:
    """Total rows across the knowledge-graph store, read-only."""
    import sqlite3
    if not Path(db_path).exists():
        return None
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
        try:
            names = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")]
            return sum(con.execute(f"SELECT count(*) FROM '{n}'").fetchone()[0] for n in names)
        finally:
            con.close()
    except Exception:
        return None


def count_vault_files(root: str) -> int | None:
    base = Path(root)
    if not base.is_dir():
        return None
    try:
        return sum(1 for p in base.rglob("*") if p.is_file())
    except OSError:
        return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lloyd self-modification guardian")
    p.add_argument("--repo", default=policy.REPO)
    p.add_argument("--state", default=str(policy.SELFMOD_STATE))
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
