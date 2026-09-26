"""Round lifecycle: open a worktree, gate it, land it, clean up.

The round is a *wrapper*, not a code generator. The proposal step is Lloyd
doing ordinary work inside the worktree with his ordinary tools — the same
Edit/Write/Bash he uses everywhere else. This module only guarantees that the
work happens somewhere safe, is judged by the gate that predates it, and is
reversible once landed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from scripts.automod import gate as G, promote as P, spec, state as S, worktree as W

LIVE_ROOT = Path(__file__).resolve().parent.parent.parent


def live_venv_python(live_root: Path | None = None) -> Path:
    """The absolute interpreter a round must run its own verify command with.

    `.venvs/` is gitignored, so it exists only in the live checkout: a round
    worktree under `~/lloyd-work/<round>/home/lloyd` has no `.venvs/` at all
    until the gate's `venv` rung clones one, and that rung is skipped whole on
    a requirements-unchanged diff — after the implement turn in which the
    verify command actually runs. So the relative form the vault has always
    prescribed (`.venvs/lloyd/bin/python`, `CLAUDE.md:12`, 247 files under
    `~/obsidian`) dies inside the round with `No such file or directory`,
    which reads exactly like a failed acceptance check and is not one (#692,
    measured again on the round that found it: exit 127 on the relative form,
    the same pytest invocation through the live interpreter 3 passed).

    The round is therefore told the interpreter it has, in absolute form, in
    both the start response and its run spec. Provisioning a symlinked
    `.venvs` instead was rejected: `promote` decided the venv swap off a
    `.exists()` probe, and through a symlink that probe is true on every
    round — see `swap_candidate_venv`.
    """
    return (live_root or LIVE_ROOT) / ".venvs" / "lloyd" / "bin" / "python"


def _round_id() -> str:
    from scripts.autoresearch.common import round_id
    return round_id().replace("R_", "SM_")


def start(goal: str, *, base: str | None = None, force: bool = False,
          item_id: int | None = None, from_branch: str | None = None,
          opened_by: str = "cli", session_id: str = "") -> dict:
    """Open a round: take the lock, cut a worktree, write the run spec.

    `item_id` binds the round to the backlog item it implements; the gate's
    review rung reads it from `run_spec.yaml` to find the clauses it grades
    against. `from_branch` resumes a round the review rung sent back: the new
    worktree starts from that branch, is rebased onto live HEAD at once so
    the recorded base is the truth, and the old branch is deleted (otherwise
    they accumulate forever). A rebase conflict falls back to a fresh
    worktree and names the paths — the old branch stays for `git diff`.
    """
    S.ensure_dirs()
    if not force:
        S.require_enabled("open a round", LIVE_ROOT)
    if S.is_halted():
        raise RuntimeError(f"promotions are halted: {S.HALTED_PATH.read_text().strip()}")
    if S.is_broken():
        raise RuntimeError(f"guardian is BROKEN: {S.BROKEN_PATH.read_text().strip()}")

    lock = S.Lock(owner="round-start").acquire()
    try:
        # A worktree is cut from HEAD, which is committed state; uncommitted
        # edits in production are not in it and cannot reach it. Refusing here
        # blocked `automod_start` for #448 over one orphaned file while the
        # human was working on something else entirely. Recorded, not refused
        # — the gate and the promoter check the paths that actually matter,
        # which are the ones this round's diff will overlap.
        dirty = W.dirty_paths(LIVE_ROOT)
        rid = _round_id()
        base = base or subprocess.run(
            ["git", "-C", str(LIVE_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip()

        # Prune only: `start` never used the list, and `list_registered` now
        # raises on a failed read, which must not be able to refuse a round
        # opening over a listing.
        W.prune(LIVE_ROOT)
        resumed: dict = {}
        if from_branch and from_branch != f"automod/{rid}" and W.branch_exists(LIVE_ROOT, from_branch):
            wt = W.create_from_branch(rid, from_branch, repo=LIVE_ROOT)
            ok, why, conflicts = W.rebase_onto(wt, base, upstream=_recorded_base(from_branch))
            if ok:
                W.delete_branch(LIVE_ROOT, from_branch)
                resumed = {"from_branch": from_branch, "rebased_onto": base}
            else:
                # Fresh start, old branch kept for `git diff`; the caller is told.
                W.remove(rid, keep_branch=False, repo=LIVE_ROOT)
                wt = W.create(rid, base=base, repo=LIVE_ROOT)
                resumed = {"from_branch": from_branch, "from_branch_conflict": conflicts or [why]}
        elif from_branch:
            resumed = {"from_branch": from_branch, "from_branch_missing": True}
            wt = W.create(rid, base=base, repo=LIVE_ROOT)
        else:
            wt = W.create(rid, base=base, repo=LIVE_ROOT)

        run_spec = {
            "objective": goal,
            "evaluation": {"command": "scripts.automod.gate", "timeout_secs": 3600},
            "budget": {"max_rounds": 1, "max_variants_per_round": 1},
            "mutation_scope": {"writable_paths": list(spec.ALLOWED_GLOBS)},
            "code": {"base_commit": base, "branch": f"automod/{rid}",
                     "worktree": str(wt), "venv_python": str(live_venv_python())},
        }
        if item_id:
            run_spec["item"] = {"id": int(item_id)}
            # A pending amendment from an earlier round was never ratified and
            # is not this round's contract; the implementer reads the clauses
            # next, so restore them now rather than at the first gate.
            try:
                from scripts.automod import backlog as B
                B.orphan_stale_amendments(int(item_id), rid)
            except Exception as exc:  # noqa: BLE001 — bookkeeping, never the round
                # stderr: the CLI prints start()'s result as JSON on stdout.
                import sys
                print(f"[warn] could not orphan stale amendments on #{item_id}: {exc}",
                      file=sys.stderr)
        err = spec.validate_code_run_spec(run_spec)
        if err:
            W.remove(rid)
            raise RuntimeError(f"run spec invalid: {err}")

        out = S.ROUNDS_DIR / rid
        out.mkdir(parents=True, exist_ok=True)
        import yaml
        (out / "run_spec.yaml").write_text(
            yaml.safe_dump(run_spec, sort_keys=False), encoding="utf-8")

        # `opened_by` / `session_id` (2026-09-17): the reaper closes a round a
        # tool opened whose session has gone quiet and that no implement row
        # names; a round a person opened from the CLI is theirs to close.
        # The #1489 ReasoningBank A/B arm of the implement turn opening this
        # round, copied off its `started` row so either row joins the analysis.
        arm: dict = {}
        if item_id:
            try:
                from scripts.automod import reasoning_bank as RB
                arm = RB.arm_for_round(S.read_events(limit=2000), int(item_id))
            except Exception:  # noqa: BLE001 — bookkeeping, never the round
                arm = {}
        S.append_event({"event": "round_start", "round_id": rid, "base": base,
                        "goal": goal[:500], "worktree": str(wt),
                        "opened_by": opened_by, **({"session_id": session_id} if session_id else {}),
                        **({"item_id": int(item_id)} if item_id else {}),
                        **arm,
                        **resumed,
                        **({"live_dirty_paths": dirty[:20]} if dirty else {})})
        out_d = {"round_id": rid, "worktree": str(wt), "base": base,
                 "branch": f"automod/{rid}",
                 # The implementer's verify command has to be runnable from the
                 # cwd it was handed, and the worktree has no `.venvs/` to run
                 # the relative form through (`live_venv_python`).
                 "venv_python": str(live_venv_python()),
                 "run_spec": str(out / "run_spec.yaml"), **resumed}
        if item_id:
            out_d["item_id"] = int(item_id)
        if resumed.get("from_branch_conflict"):
            out_d["note"] = (f"resuming {from_branch} conflicted with live main in "
                             f"{resumed['from_branch_conflict']}; this worktree is fresh "
                             f"and the old branch is kept for `git diff`")
        elif resumed.get("from_branch_missing"):
            out_d["note"] = f"{from_branch} does not exist; this worktree is fresh"
        if dirty:
            out_d["live_dirty_paths"] = dirty[:20]
            out_d["note"] = ("the live tree has uncommitted edits; the gate tolerates them "
                             "unless this round changes the same paths")
        return out_d
    finally:
        lock.release()


def run_gate(round_id: str, *, skip_smoke: bool = False,
             land_on_pass: str | None = None) -> dict:
    """Gate a round; with `land_on_pass`, start its landing the moment it passes.

    `land_on_pass` names who asked (`reaper`). A gate the reaper started has no
    turn behind it to read the pass and call `automod_land`, and waiting for
    the reaper's next look cost up to `interval_seconds` (900 s) of a finished
    change sitting gated and unlanded. The landing it starts is
    `land_detached`, the same spawn every other landing takes, after the same
    checks `autocode._land_if_passed` makes.
    """
    report = _run_gate(round_id, skip_smoke=skip_smoke)
    if land_on_pass and report.get("ok"):
        _land_after_pass(round_id, report, by=land_on_pass)
    return report


def _land_after_pass(round_id: str, report: dict, *, by: str) -> None:
    if S.is_halted() or S.is_broken() or not S.is_enabled(LIVE_ROOT):
        return
    head = W.head(W.worktree_path(round_id)) or ""
    if not head or str(report.get("head") or "") != head:
        return
    started = land_detached(round_id, by=by)
    if started.get("error"):
        print(f"[land-on-pass] {round_id}: landing could not start: {started['error']}")
        return
    item_id = None
    try:
        import yaml
        spec_doc = yaml.safe_load((S.ROUNDS_DIR / round_id / "run_spec.yaml").read_text()) or {}
        item_id = (spec_doc.get("item") or {}).get("id")
    except Exception:  # noqa: BLE001 — the item id is attribution, not the landing
        pass
    # `land_rescued`, like the reaper's own landing: it is what joins a
    # promotion with no `finished` row of its own back to its item, and what
    # keeps `_land_if_passed` from starting a second landing.
    S.append_event({"event": "land_rescued", "round_id": round_id, "item_id": item_id,
                    "head": head, "pid": started["pid"], "verb": "landing", "by": by,
                    "reason": f"the gate {by} started passed at {head[:8]}; landing it at once"})


def _run_gate(round_id: str, *, skip_smoke: bool = False) -> dict:
    wt = W.worktree_path(round_id)
    if not wt.exists():
        raise RuntimeError(f"no worktree for {round_id}")
    spec_path = S.ROUNDS_DIR / round_id / "run_spec.yaml"
    import yaml
    run_spec = yaml.safe_load(spec_path.read_text()) or {}
    base = run_spec["code"]["base_commit"]
    item_id = (run_spec.get("item") or {}).get("id")

    # One gate per round at a time. The second concurrent gate of 2026-09-11
    # was the MCP pool's transport retry, but the CLI can do it too.
    live = S.gate_in_progress(round_id)
    if live and int(live.get("pid") or 0) != os.getpid():
        raise RuntimeError(f"a gate is already running for {round_id} (pid {live['pid']}, "
                           f"started {live.get('started_iso')}); wait for it")
    g = G.Gate(round_id, wt, base, skip_smoke=skip_smoke, item_id=item_id)
    S.write_gate_marker(round_id, pid=os.getpid(), head=W.head(wt) or "", by="run_gate")
    try:
        report = g.run()
        S.write_gate_report(round_id, report.to_dict())
        # Preflight may have rebased the round onto a moved `main`. The spec is
        # where the NEXT gate call reads its base from, and a stale base there
        # makes `changed_paths` sweep the human's commits into the round's diff.
        if report.base != base:
            S.update_run_spec_base(round_id, report.base)
    finally:
        S.clear_gate_marker(round_id)
    return report.to_dict()


def _recorded_base(branch: str) -> str | None:
    """The base the round behind `automod/SM_…` was gated against, when its
    run spec still says and that commit is really in the branch's history.
    Resuming from it replays only that round's commits; see `W.rebase_onto`."""
    import re
    import yaml
    m = re.fullmatch(r"automod/(SM_[0-9A-Za-z_]+)", str(branch or ""))
    if not m:
        return None
    try:
        spec = yaml.safe_load((S.ROUNDS_DIR / m.group(1) / "run_spec.yaml").read_text()) or {}
        base = str((spec.get("code") or {}).get("base_commit") or "")
    except Exception:  # noqa: BLE001 — no spec: the plain rebase, as before
        return None
    if not base:
        return None
    ok = subprocess.run(["git", "-C", str(LIVE_ROOT), "merge-base", "--is-ancestor", base, branch],
                        capture_output=True).returncode == 0
    return base if ok else None


# How long a landing queues behind another one: its idle wait at the hard
# ceiling plus the observation window. Past that something is stuck, and a
# `LockHeld` says so.
LAND_LOCK_MAX_WAIT = 2 * 3600.0
LAND_LOCK_POLL = 10.0


def _land_lock(round_id: str, *, dry_run: bool = False,
               wait_settle: bool = True) -> "S.Lock":
    """The automod lock for a landing, queued for rather than refused.

    With one round at a time a second landing could not exist. With two
    (`workers.sources.autocode.max_inflight`) both can call `automod_land`
    within seconds of each other, and the loser used to die on `LockHeld`
    with a gate-passed round. It waits instead, and waits for the winner's
    observation window too: the chamber check is repeated AFTER the lock is
    taken, because the promotion this landing must not land on top of may
    have been written while it was queued.

    `wait_settle=False` is a land-train merge (`P.landing_is_eager` said []):
    it changes nothing that runs, so the promotion under observation is not
    its to wait out, and it only queues for the lock.
    """
    deadline = time.time() + LAND_LOCK_MAX_WAIT
    chamber = wait_settle and not dry_run and S.chamber_enabled(LIVE_ROOT)
    while True:
        observed = S.read_current() if chamber else None
        if observed and observed.get("state") == "observing":
            # The chamber: this round ran while the last promotion was under
            # observation. Wait for it here, outside the lock and before the
            # pool is paused; `promote` still refuses if it never settled.
            P.wait_for_settle(round_id=round_id, observed=observed)
        try:
            lock = S.Lock(owner=f"land-{round_id}").acquire()
        except S.LockHeld:
            if dry_run or time.time() >= deadline:
                raise
            time.sleep(LAND_LOCK_POLL)
            continue
        now = S.read_current() if chamber else None
        if now and now.get("state") == "observing" and time.time() < deadline:
            lock.release()
            # Paced like the `LockHeld` branch. What bounds this loop is
            # `wait_for_settle` blocking at its top, and a loop whose only
            # brake is a callee is a busy loop the day that callee returns
            # early: with it stubbed, a test spun here at 100% CPU for 822 s,
            # re-parsing config.yaml, until PRODUCTION's promotion settled.
            time.sleep(LAND_LOCK_POLL)
            continue
        return lock


class LandingKilled(RuntimeError):
    """The landing process was signalled before it promoted anything."""


def _die_loudly_on_signal(round_id: str):
    """SIGTERM/SIGHUP while a landing WAITS becomes an external `land_failed`.

    Python's default SIGTERM ends the process with no `finally`: the land
    marker stayed behind naming a dead pid, nothing reached the ledger, and
    `implement_outcomes` read a gate-passed round with no promotion as
    `spent` (#1179, 2026-09-17, killed by `timeout 120` around a foreground
    `round land`). Raised as an exception instead, so `land`'s `finally`
    clears the marker and the ledger says what happened. Only while nothing
    has merged: once `promote` is past the fast-forward its own rollback
    path owns the failure, and this handler is restored to the default there
    by the process simply not being in a wait.

    Returns a callable that puts the previous handlers back, and `land` calls
    it on the way out. The CLI never needed that — its process ends with the
    landing — but `land` is also called in-process (the tests do), and a
    handler that outlives the landing it names speaks for a round that is
    over: on 2026-09-18 a round's model `pkill`ed its own background pytest,
    twice, and each time the handler a finished `land("SM_L")` had left behind
    wrote `land_failed` for that fixture round into the production ledger.
    """
    import signal

    def _handler(signum, _frame):
        S.append_event({"event": "land_failed", "round_id": round_id, "ok": False,
                        "external_blocker": True, "killed_by_signal": int(signum),
                        "detail": (f"the landing process was killed by signal {int(signum)} before it "
                                   f"promoted anything — a foreground `round land` under a Bash "
                                   f"timeout does this; use automod_land")})
        raise LandingKilled(f"landing of {round_id} killed by signal {int(signum)}")
    previous: dict = {}
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            previous[sig] = signal.signal(sig, _handler)
        except (ValueError, OSError):   # not the main thread: nothing to install
            break

    def _restore() -> None:
        for sig, old in previous.items():
            try:
                # `None` is what `signal.signal` returns for a handler that was
                # not installed from Python; the default is the honest stand-in.
                signal.signal(sig, signal.SIG_DFL if old is None else old)
            except (ValueError, OSError):
                pass
    return _restore


def land(round_id: str, *, dry_run: bool = False, force: bool = False) -> dict:
    """Promote a round whose gate passed. Refuses otherwise.

    A real landing owns the round's land marker for its whole life, so the
    implement source's reaper — which runs the moment the turn that called
    `automod_land` ends — cannot abort a round that is waiting for idle or
    chasing a moved `main`. A dry run neither writes nor clears it.
    """
    restore_signals = None
    if not dry_run:
        live = S.land_in_progress(round_id)
        if live and int(live.get("pid") or 0) != os.getpid():
            raise RuntimeError(f"a landing is already running for {round_id} (pid "
                               f"{live['pid']}, started {live.get('started_iso')})")
        S.write_land_marker(round_id, pid=os.getpid(), by="round.land")
        restore_signals = _die_loudly_on_signal(round_id)
    try:
        if not (force or dry_run):
            S.require_enabled("land a round", LIVE_ROOT)
        gate_path = S.ROUNDS_DIR / round_id / "gate.json"
        if not gate_path.exists():
            raise RuntimeError(f"{round_id} has no gate report — run the gate first")
        report = json.loads(gate_path.read_text())
        if not report.get("ok"):
            failed = [r["name"] for r in report.get("rungs", []) if not r["ok"]]
            raise RuntimeError(f"gate did not pass (failed: {failed})")

        wt = W.worktree_path(round_id)
        eager: list = ["dry run"]
        if not dry_run:
            # Outside the lock, or the turn being waited for cannot open its
            # round (see `P.wait_for_rounds`). An external failure: the item
            # keeps its attempt and its branch.
            waited_from = time.time()
            # A landing that restarts nothing kills no turn, so it has no
            # sibling to wait for (`P.restart_needed`; asked again, under the
            # lock, by `promote`). Nor does one the land train will merge and
            # leave to the flush (`P.landing_is_eager` is []): only an eager
            # landing restarts here.
            changed = [str(p) for p in report.get("changed_paths") or []]
            restart, restart_why = P.restart_needed(changed)
            eager = P.landing_is_eager(changed, report)
            waits = restart and bool(eager)
            ok, why = (P.wait_for_rounds(P._idle_budget(None)[1]) if waits
                       else (True, f"not waited for: {restart_why}" if not restart
                             else "not waited for: the restart is deferred to the flush"))
            # On the ledger either way: on 2026-09-18 this wait let nine
            # landings into the drain beside live sibling turns and nothing
            # recorded what it had seen.
            S.append_event({"event": "land_wait_rounds", "round_id": round_id, "ok": ok,
                            "detail": why, "waited_s": round(time.time() - waited_from, 1)})
            if not ok:
                P._land_failed(round_id, why, external=True, waited_rounds=True)
        lock = _land_lock(round_id, dry_run=dry_run,
                          wait_settle=dry_run or bool(eager))
        try:
            result = P.promote(round_id, wt, report["base"],
                               gate_report=report, dry_run=dry_run)
        finally:
            lock.release()
        if not dry_run:
            W.remove(round_id, keep_branch=False, repo=LIVE_ROOT)
    except LandingKilled:
        _clear_dead_landing_record(round_id)
        raise
    finally:
        if not dry_run:
            S.clear_land_marker(round_id, pid=os.getpid())
        if restore_signals is not None:
            restore_signals()
    # After the marker is gone: `flush_due` waits while any landing runs, and
    # this one would otherwise be the landing it waits for.
    if not dry_run and result.get("deferred"):
        result["flush"] = maybe_flush(by=f"land {round_id}")
    return result


def _clear_dead_landing_record(round_id: str) -> None:
    """A landing killed by a signal takes its `landing` record down with it.

    `promote` writes `current.json` in state `landing` before it waits for
    idle, and a SIGTERM in that wait (a host reboot, a `timeout`) left it
    there: on 2026-09-24 the reboot at 13:54Z stranded SM_20260924_132227's
    record, and every later landing — `merge_round`, the flush, `round
    restart` without `--force` — refused behind a landing nothing was running.
    Cleared only when the record is this round's, still `landing`, and its
    commit never reached live `main`; a commit that did merge is the
    promoter's or the guardian's to judge, and its record stays."""
    try:
        current = S.read_current()
        if not current or current.get("round_id") != round_id \
                or current.get("state") != "landing":
            return
        commit = str(current.get("commit") or "")
        head = P._live_head(LIVE_ROOT)
        if commit and head and P._is_ancestor(LIVE_ROOT, commit, head):
            return
        S.clear_current()
        S.append_event({"event": "current_cleared", "round_id": round_id,
                        "commit": commit or None,
                        "reason": "the landing was killed by a signal before its commit "
                                  "reached main; its `landing` record would block every "
                                  "later landing"})
    except Exception:  # noqa: BLE001 — the land_failed row is already written
        pass


def land_detached(round_id: str, *, by: str) -> dict:
    """Start `round land` for a gate-passed round in its own session.

    `{"pid", "log"}` on success, `{"error"}` otherwise. The one definition of
    "start a landing from inside the stack", for its two callers: the
    `automod_land` tool, which runs inside lloyd-mcp, and the implement
    source's reaper, which runs inside the backend. A landing restarts both,
    and both confs set `stopasgroup`, so a landing run in-process is killed
    partway through its own restart (`agent_mcp/automod.py::_land_detached`
    has the long version); `spawn_detached` is what a process-group signal
    cannot reach.

    The marker is written here, with the child's pid, BEFORE this returns: the
    child takes seconds to import and write its own, and the reaper looks at
    an open round the moment its turn is over. Same pid, so the child's write
    replaces it.
    """
    if S.gate_in_progress(round_id):
        return {"error": f"a gate is still running for {round_id} — automod_gate_wait first"}
    if S.land_in_progress(round_id):
        return {"error": f"a landing is already running for {round_id} — end your turn"}
    gate_path = S.ROUNDS_DIR / round_id / "gate.json"
    if not gate_path.exists():
        return {"error": f"{round_id} has no gate report — run automod_gate first"}
    report = json.loads(gate_path.read_text())
    if not report.get("ok"):
        failed = [r["name"] for r in report.get("rungs", []) if not r["ok"]]
        return {"error": f"gate did not pass (failed: {failed})"}
    if not W.worktree_path(round_id).exists():
        return {"error": f"no worktree for {round_id}"}
    log = S.ROUNDS_DIR / round_id / "land.log"
    python = LIVE_ROOT / ".venvs" / "lloyd" / "bin" / "python"
    pid = S.spawn_detached([python, "-m", "scripts.automod.round", "land", round_id],
                           log, cwd=LIVE_ROOT)
    S.write_land_marker(round_id, pid=pid, by=by)
    return {"pid": pid, "log": str(log)}


def gate_detached(round_id: str, *, by: str, skip_smoke: bool = False,
                  land_on_pass: bool = False) -> dict:
    """Start `round gate` for an open round in its own session.

    `{"pid", "log"}` on success, `{"error"}` otherwise. The one spawn, for the
    `automod_gate` tool and for the implement source's reaper, which re-gates
    a round whose only failed rung was a grader that could not be reached
    (`autocode._regate_if_unreviewed`). The marker is written here with the
    child's pid before this returns, for the reason `land_detached` gives.

    `land_on_pass` hands the pass straight to a landing (`run_gate`), for a
    gate no turn is waiting on: the reaper's.
    """
    if not W.worktree_path(round_id).exists():
        return {"error": f"no worktree for {round_id}"}
    if S.gate_in_progress(round_id):
        return {"error": f"a gate is already running for {round_id}"}
    if S.land_in_progress(round_id):
        return {"error": f"a landing is already running for {round_id}"}
    log = S.ROUNDS_DIR / round_id / "gate.log"
    python = LIVE_ROOT / ".venvs" / "lloyd" / "bin" / "python"
    argv = [python, "-m", "scripts.automod.round", "gate", round_id]
    if skip_smoke:
        argv.append("--skip-smoke")
    if land_on_pass:
        argv += ["--land-on-pass", by]
    pid = S.spawn_detached(argv, log, cwd=LIVE_ROOT)
    S.write_gate_marker(round_id, pid=pid, head=W.head(W.worktree_path(round_id)) or "", by=by)
    return {"pid": pid, "log": str(log)}


def flush(*, now: bool = False, by: str = "cli", reason: str = "") -> dict:
    """`round flush [--now]`: restart once for everything the land train holds
    and open its window (`P.flush_pending`). Waits for implement turns and for
    idle like a landing; `--now` does neither and every turn in flight dies
    with the restart and is re-offered."""
    return P.flush_pending(reason or f"flush by {by}", kill_turns=now, by=by)


def flush_detached(*, by: str, now: bool = False) -> dict:
    """Start `round flush` in its own session: the restart it performs kills
    whichever service called it (the implement source runs in the backend).
    The marker is written with the child's pid before this returns, as
    `land_detached` does, so the next look already sees a flush running."""
    live = S.flush_in_progress()
    if live:
        return {"error": f"a flush is already running (pid {live.get('pid')})"}
    log = S.STATE_DIR / "flush.log"
    python = LIVE_ROOT / ".venvs" / "lloyd" / "bin" / "python"
    argv = [python, "-m", "scripts.automod.round", "flush", "--by", str(by)[:120]]
    if now:
        argv.append("--now")
    pid = S.spawn_detached(argv, log, cwd=LIVE_ROOT)
    S.write_flush_marker(pid=pid, by=by)
    return {"pid": pid, "log": str(log)}


def maybe_flush(*, by: str, rounds_in_flight: int | None = None) -> dict | None:
    """Spawn a flush when `P.flush_due` says so. Never raises: a trigger that
    fails leaves the train for the next one."""
    try:
        due, why = P.flush_due(rounds_in_flight)
        if not due:
            return None
        started = flush_detached(by=f"{by}: {why}")
        return {"why": why, **started}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"flush trigger failed: {exc}"[:300]}


def _gate_verdict(round_id: str) -> dict:
    """The round's own `gate.json` verdict, as `{"gate_ok", "gate_head"}` —
    `""` for both when no gate report exists.

    Read here, off disk, and never from the caller: `abort` is the only writer
    of `round_aborted`, and the reason it stores is the caller's narrative,
    which has been wrong about the very gate it names. On 2026-09-15 a round
    was aborted as "Both review attempts spent … the review refused twice on
    clause 4/5 test honesty" twenty seconds after its report said `ok: true`
    with "review: 5 met of 5 clause(s)"; the narrative described the *first*
    gate run. `W.remove` then deleted the round dir, so the artifact that
    disproved it was gone by the time anyone read the row.
    """
    try:
        report = json.loads((S.ROUNDS_DIR / round_id / "gate.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):   # absent, unreadable or truncated: no verdict to carry
        return {"gate_ok": "", "gate_head": ""}
    return {"gate_ok": report.get("ok", ""), "gate_head": report.get("head", "")}


#: A patch over this size is not a lost edit, it is somebody rebuilding the
#: tree; copying it into the audit trail on every close is how the audit trail
#: stops being readable. The paths are recorded whatever the size.
LIVE_DIRTY_PATCH_MAX_BYTES = 2 * 1024 * 1024

#: How many paths ride one ledger event. `start()` has capped its
#: `live_dirty_paths` at 20 since `649193f` made live dirt tolerated; a close
#: row is a row in the same file and gets the same bound.
LIVE_DIRTY_EVENT_LIMIT = 20


def preserve_live_dirt(round_id: str, *, live_root: Path | None = None) -> dict:
    """Copy the live checkout's uncommitted work aside, into the round's state dir.

    An edit made in `~/lloyd` rather than in the round's worktree exists in
    exactly one copy: it is not on `automod/<round_id>` (the branch `abort`
    keeps), not in HEAD, and not in the worktree `W.remove` is about to delete.
    Since `649193f` the loop records such dirt when it *opens* or *judges* a
    round and refuses only on an overlap — which also removed the one moment it
    ever noticed an orphan edit. Measured over `promotions.jsonl` on 2026-09-18:
    45 of 277 `round_start` rows carried `live_dirty_paths`, and 0 of the 250
    `round_aborted`/`round_abandoned` rows carried anything about it. Close is
    the only blind moment, and in the one case observable that day
    (`agent-services/supervisor/conf.d/agent-llm-primary.conf`, dirty on 11 of
    12 consecutive starts) the bytes survived only because a person committed
    them by hand at 16:08Z, after the last row naming the path.

    So a close writes the work down before it closes, under
    `~/.local/state/lloyd-automod/rounds/<round_id>/live-dirty/`, which outlives
    both the worktree (`~/lloyd-work/<round_id>`) and the round:

    - `dirty.patch` — `git diff HEAD`: every tracked modification, staged or
      not, which is the whole of what a working tree can lose;
    - one copy per **untracked `.py`** path, which a diff of HEAD cannot carry.

    `.py` only, because that is the bound the guardian's own preservation uses
    (`agent-services/guardian/rollback.py::preserve_evidence`) for the same
    reason: an unbounded copy of every untracked file sweeps model outputs,
    caches and half-written exports into the audit trail with it. Every other
    untracked path is still *named* in `live_dirty_paths`.

    **No `git stash`**, ever. The live stash stack is one global LIFO list
    shared by every author, and on 2026-09-11 a round implementing an unrelated
    item popped #573's recovered 136-line diff out of it. A file with a round id
    in its path beats a stack entry nobody owns.

    Returns the keys that ride the close event: `live_dirty_paths` (capped at
    `LIVE_DIRTY_EVENT_LIMIT`), `live_dirty_dir`, `live_dirty_patch` and
    `live_dirty_untracked`. A clean tree yields `{"live_dirty_paths": []}` and
    writes nothing at all.
    """
    import shutil
    root = Path(live_root) if live_root else LIVE_ROOT
    paths = W.dirty_paths(root)
    out: dict = {"live_dirty_paths": paths[:LIVE_DIRTY_EVENT_LIMIT]}
    if not paths:
        return out
    dest = S.ROUNDS_DIR / round_id / "live-dirty"
    try:
        dest.mkdir(parents=True, exist_ok=True)
        out["live_dirty_dir"] = str(dest)
        diff = W.git(root, "diff", "HEAD")
        if diff.returncode != 0:
            # A read that failed is not a tree with no edits. Say which of the
            # two it was rather than recording an empty patch for a dirty tree.
            out["live_dirty_error"] = f"git diff HEAD rc={diff.returncode}"[:300]
        elif diff.stdout.strip():
            size = len(diff.stdout.encode("utf-8", "replace"))
            if size > LIVE_DIRTY_PATCH_MAX_BYTES:
                out["live_dirty_patch_skipped"] = (
                    f"{size} bytes over the {LIVE_DIRTY_PATCH_MAX_BYTES}-byte cap")
            else:
                patch = dest / "dirty.patch"
                patch.write_text(diff.stdout, encoding="utf-8")
                out["live_dirty_patch"] = str(patch)
        copied: list[str] = []
        listing = W.git(root, "ls-files", "--others", "--exclude-standard")
        if listing.returncode != 0:
            out["live_dirty_error"] = f"git ls-files rc={listing.returncode}"[:300]
        for rel in listing.stdout.splitlines():
            rel = rel.strip().strip('"')
            if not rel.endswith(".py"):
                continue
            try:
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / rel, target)
                copied.append(rel)
            except OSError as exc:
                # One unreadable file is recorded and the rest still come;
                # dropping the whole preservation over it would be the same
                # loss this function exists to end.
                out["live_dirty_error"] = f"{rel}: {exc}"[:300]
        if copied:
            out["live_dirty_untracked"] = copied
    except OSError as exc:  # noqa: BLE001 — a close must still close
        out["live_dirty_error"] = str(exc)[:300]
    return out


def abort(round_id: str, reason: str = "") -> dict:
    """Close a round, branch kept. `reason` rides the event: seventeen of the
    first seventeen `round_aborted` rows carried nothing but the id.

    The row also carries the gate verdict the round actually had, beside the
    caller's reason, so a narrative that contradicts the artifact is visible
    without opening `gate.json` — which by then is gone (see `_gate_verdict`).

    And it carries the live checkout's uncommitted work, copied aside first
    (see `preserve_live_dirt`). The branch is not where an orphan live edit
    lives, so a close that kept only the branch kept nothing of it.
    """
    verdict = _gate_verdict(round_id)     # before the removal, which deletes the round dir
    preserved = preserve_live_dirt(round_id)   # also before: the tree is nobody's once the worktree is gone
    W.remove(round_id, keep_branch=True, repo=LIVE_ROOT)
    S.append_event({"event": "round_aborted", "round_id": round_id,
                    "reason": " ".join(str(reason or "").split())[:500],
                    **verdict, **preserved})
    return {"aborted": round_id, "branch_kept": f"automod/{round_id}", **verdict, **preserved}


def _unit_drift() -> list[str]:
    """Installed systemd units that differ from the repo's copies.

    The repo copy is not the running definition — systemd reads
    ~/.config/systemd/user. An edit that never got installed is a change that
    looks landed and does nothing, which for a watchdog unit is the worst kind
    of silent no-op. `_apply_service_changes` installs them at landing time;
    this reports drift from any other cause.
    """
    import filecmp
    drift = []
    src_dir = LIVE_ROOT / "agent-services" / "systemd"
    dst_dir = Path.home() / ".config" / "systemd" / "user"
    for src in sorted(src_dir.glob("*.service")) + sorted(src_dir.glob("*.timer")):
        dst = dst_dir / src.name
        if not dst.exists():
            drift.append(f"{src.name}: not installed")
        elif not filecmp.cmp(src, dst, shallow=False):
            drift.append(f"{src.name}: installed copy differs")
    return drift


def _status_live_dirty() -> dict:
    """`status()`'s view of the live checkout's working tree.

    `unit_drift` has always been here and no key has ever described the tracked
    files themselves, which is how an orphan edit in production stayed invisible
    to the one tool that reports on the loop (`automod_status` reads this
    payload; `agent_mcp/automod.py` registers it verbatim). `/health` does look
    at the tree — `app/routers/health.py` runs `git status --porcelain` and
    publishes `git.dirty` as a **boolean**, no paths — so the thing that reports
    dirt reports only that there is some.

    The empty list gets a positive control, because `W.dirty_paths` returns `[]`
    both for a clean tree and for a `git status` that failed (`check=False`);
    the two read the same, and this is the shape that has burned the loop
    repeatedly — a check reading its own missing input. `git status` costs ~5 ms
    here, so the confirming read runs whenever the list is empty.
    """
    paths = W.dirty_paths(LIVE_ROOT)
    out: dict = {"live_dirty_paths": paths}
    if not paths and W.git(LIVE_ROOT, "status", "--porcelain").returncode != 0:
        out["live_dirty_error"] = ("git status could not be read; this empty list is "
                                   "not a clean tree")
    return out


def status() -> dict:
    lkg = S.read_lkg()
    head = subprocess.run(["git", "-C", str(LIVE_ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    return {
        "enabled": S.is_enabled(LIVE_ROOT),
        "head": head,
        "head_is_lkg": bool(lkg and lkg.get("commit") == head),
        "last_known_good": lkg,
        "current": S.read_current(),
        "last_settled": S.read_last_settled(),
        "halted": S.is_halted(),
        "broken": S.is_broken(),
        "pause_remaining_s": round(S.pause_remaining(), 1),
        "rollback_request": S.read_rollback_request(),
        "pending_restart": _status_pending(),
        "unit_drift": _unit_drift(),
        **_status_live_dirty(),
        **_status_worktrees(),
        "recent": S.read_events(limit=15),
    }


def _status_pending() -> dict:
    live = None
    if S.read_pending():
        try:
            from workers.sources import autocode as AC
            live = AC.implement_turns_in_flight()
        except Exception:  # noqa: BLE001 — unreadable reads as "?", never as a gap
            live = None
    try:
        return P.pending_summary(live)
    except Exception as exc:  # noqa: BLE001 — a status is never the thing that fails
        return {"error": str(exc)[:300]}


def _status_worktrees() -> dict:
    """`status()`'s worktree listing, with a failed read made visible.

    `prune_orphans` raises rather than answering `[]` when
    `git worktree list` fails (see `worktree.list_registered`) — a status
    report must not turn that into "no rounds are open". Reporting the failure
    is also what keeps this display consistent with the loop, which declines on
    the same read."""
    try:
        return {"worktrees": W.prune_orphans(LIVE_ROOT)}
    except W.WorktreeListUnavailable as exc:
        return {"worktrees": [], "worktrees_error": str(exc)}


def _served_code_is_head(running: str, head: str) -> tuple[bool, list[str]]:
    """`(the served code equals HEAD, the paths that differ)`.

    `bless` verifies against the RUNNING process because only `/health.commit`
    proves the service — but what that check is *for* is narrower than
    "the shas match". `app/gitinfo.py` and `app/routers/health.py` both state
    it as: only `/health.commit == last_known_good` proves **the service**
    changed. A commit that moves no file the backend imports leaves the served
    code identical, so pinning HEAD there pins exactly what is running.

    That case is now routine rather than exotic. `arch-review` commits one
    `architecture/*.md` per run, up to `daily_max` times a day, so HEAD sits
    documentation-ahead of the served commit most of the time and a
    sha-equality guard makes `bless` unreachable without a restart — a restart
    whose only purpose is to load a markdown file nothing reads at runtime.

    An **allowlist of inert paths**, never a denylist of code: anything this
    does not recognise is code, so a new `.py` can never pass as documentation.
    Fails closed — an unreadable diff (the served commit garbage-collected, git
    unavailable) reports "not equal" and `bless` refuses, which is the safe
    direction for a guard on a rollback target.
    """
    r = subprocess.run(["git", "-C", str(LIVE_ROOT), "diff", "--name-only", running, head],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return False, []
    paths = [p for p in r.stdout.splitlines() if p.strip()]
    code = [p for p in paths if not p.endswith(".md")]
    if code:
        # Since 2026-09-19 a landing that changes no file either service has
        # loaded restarts nothing (`P.restart_needed`), so HEAD routinely sits
        # tests-and-scripts ahead of the served commit as well. Same question,
        # same answer, asked of the running processes; fails closed.
        restart, _ = P.restart_needed(paths)
        if not restart:
            return True, paths
    return not code, (code or paths)


def bless(note: str = "") -> dict:
    """Record the RUNNING commit as last-known-good.

    Needed twice, and both were being done by hand with a note pasted into the
    ledger. First at install: the guardian will not act at all without an LKG,
    and `guardian-stage.sh` declines a snapshot whose selftest cannot read one.
    Second after a manual recovery, where HEAD has been put back by a human and
    the pointer is stranded at whatever last settled — which is precisely how
    one false positive discarded 26 commits.

    Verifies against the RUNNING process, not the working tree. `git rev-parse`
    proves the filesystem; only `/health.commit` proves the service.
    """
    from scripts.automod.promote import _get, BACKEND
    head = subprocess.run(["git", "-C", str(LIVE_ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    if not head:
        raise RuntimeError("cannot read HEAD")
    _, body = _get(f"{BACKEND}/health")
    running = (body or {}).get("commit")
    docs_only = False
    if running and running != head:
        same_code, differing = _served_code_is_head(running, head)
        if not same_code:
            raise RuntimeError(
                f"the running backend reports {running[:8]} but HEAD is {head[:8]}, and "
                f"{len(differing)} non-documentation path(s) differ "
                f"({', '.join(differing[:4])}{' …' if len(differing) > 4 else ''}) — "
                "restart the backend before blessing, or you will pin a commit that "
                "is not the code being served")
        # Documentation-only. The served code IS this commit's code, so HEAD is
        # the right target: rolling back to it restores what is running and
        # keeps the doc, where pinning the older sha would throw the doc away.
        docs_only = True
    if S.read_current():
        raise RuntimeError("a promotion is under observation — let it settle or "
                           "abort it rather than blessing over it")
    waiting = [e for e in S.read_pending() if e.get("restart")]
    if waiting:
        # Merged, not running, never judged: HEAD is not what is served, and
        # blessing it would certify code no window has watched.
        raise RuntimeError(
            f"{len(waiting)} landing(s) are merged but not yet running "
            f"({', '.join(str(e.get('commit'))[:8] for e in waiting[:4])}) — "
            "`round flush` and let the window settle before blessing")
    lkg = S.write_lkg(head)
    detail = f"blessed by hand: {note}" if note else "blessed by hand"
    if docs_only:
        # Recorded, not silent: the gap between the served sha and the blessed
        # one is exactly what a reader of this row would otherwise have to
        # reconstruct, and "it was only docs" is the whole justification.
        detail += f" (documentation-only ahead of served {running[:8]})"
    S.append_event({"event": "settled", "commit": head, "note": detail,
                    "served_commit": running, "docs_only_ahead": docs_only})
    return {"last_known_good": lkg, "verified_running": running,
            "docs_only_ahead": docs_only}


def recover(clear_broken: bool = True, clear_halt: bool = True,
            by: str | None = None) -> dict:
    """Come back from BROKEN: clear the flags and start the stack again.

    The guardian deliberately leaves services STOPPED when it escalates — an
    honestly-dead system beats a half-reverted one. Nothing then started them
    again, and the documented recovery was "clear the flag", which leaves the
    box down. This is the other half.

    `by` is who is clearing, for the ledger: the flap alert tells a human to
    clear the flag, and a name on the halt-clear row is what makes that act
    attributable instead of a file that silently disappeared.
    """
    from app.supervisor_client import start_process
    out: dict = {"cleared": [], "started": []}
    if clear_broken and S.is_broken():
        S.BROKEN_PATH.unlink()
        out["cleared"].append("BROKEN")
    if clear_halt and S.is_halted():
        # `by` names whoever cleared it, because this is the one production
        # route that lifts a halt and the halt-clear ledger row is where that
        # is on the record: the `recovered` row beside it carries no actor and
        # no round_id, and still does. A caller who was told to clear the flag
        # passes their own name (`--by`); with none, the row at least names the
        # call.
        who = (by or "").strip() or f"round recover pid {os.getpid()}"
        S.clear_halted(by=who)
        out["cleared"].append("promotions-halted")
    S.clear_rollback_request()
    for program in ("lloyd-mcp", "lloyd-backend"):
        ok, msg = start_process(program)
        out["started"].append(f"{program}: {msg}")
    S.append_event({"event": "recovered", "cleared": out["cleared"],
                    "started": out["started"]})
    return out


_HEALTH_KEYS = ("open", "draft", "up_next", "self_spawned_open", "implement_pool")


def board_pass() -> dict:
    """autocode's board passes, once, by hand, in housekeeping's order.

    For the moment a rule changes and the board should reflect it now rather
    than at the next 900 s housekeeping tick — the first run after the
    2026-09-14 throughput changes, which expire, unfold and re-triage what had
    accumulated. Each pass honours the same config switch housekeeping reads.
    The reaper is left out: it closes rounds, which is not a board question.
    Board health before and after, and a `board_pass` ledger row.
    """
    from scripts.automod import backlog as B

    def cfg(name: str) -> dict:
        try:
            from app.config import CONFIG
            return dict(((CONFIG.get("workers") or {}).get("sources") or {}).get(name) or {})
        except Exception:  # noqa: BLE001 — every switch defaults on
            return {}
    auto, tri = cfg("autocode"), cfg("autotriage")
    before = B.board_health(S.LEDGER_PATH)
    closed = B.close_settled_items(S.LEDGER_PATH,
                                   close_members=bool(auto.get("close_members_on_settle", True)))
    reopened = B.reopen_reverted_landings(S.LEDGER_PATH)
    unfolded = B.unfold_spent_umbrellas(S.LEDGER_PATH)
    retriaged = B.retriage_spent_items(S.LEDGER_PATH, enabled=bool(auto.get("retriage_spent", True)))
    moved = B.reconcile_statuses(S.LEDGER_PATH, enabled=bool(auto.get("status_pipeline", True)),
                                 retriage_enabled=bool(auto.get("retriage_spent", True)))
    released = B.release_held_confirmations(
        S.LEDGER_PATH, floor=int(tri.get("implement_pool_floor", B.IMPLEMENT_POOL_FLOOR)),
        enabled=bool(tri.get("hold_confirmations", True)))
    expired = B.expire_stale_spawns(S.LEDGER_PATH, enabled=bool(auto.get("expire_spawns", True)))
    after = B.board_health(S.LEDGER_PATH)
    out = {"closed_settled": [r["item_id"] for r in closed if r.get("closed")],
           "reopened_reverted": [r["item_id"] for r in reopened],
           "unfolded": {r["umbrella_id"]: r["released"] for r in unfolded},
           "retriaged": [r["item_id"] for r in retriaged],
           "status_moves": len(moved),
           "released_from_hold": [r["item_id"] for r in released if r.get("moved")],
           "expired": [r["item_id"] for r in expired],
           "before": {k: before.get(k) for k in _HEALTH_KEYS},
           "after": {k: after.get(k) for k in _HEALTH_KEYS}}
    S.append_event({"event": "board_pass", "by": "human",
                    **{k: out[k] for k in ("closed_settled", "reopened_reverted", "retriaged",
                                           "status_moves", "released_from_hold", "expired")},
                    "unfolded": {str(k): v for k, v in out["unfolded"].items()},
                    "draft_before": before.get("draft"), "draft_after": after.get("draft"),
                    "self_spawned_before": before.get("self_spawned_open"),
                    "self_spawned_after": after.get("self_spawned_open")})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Self-modification rounds")
    sub = ap.add_subparsers(dest="cmd", required=True)
    # --force exists because the interlock now covers the CLI as well as the
    # MCP tools. It was only ever checked in the tool wrapper, and the skill's
    # own worked examples drive the CLI — so "the loop ships inert" was true of
    # the surface nobody used and false of the one they did.
    s = sub.add_parser("start"); s.add_argument("goal")
    s.add_argument("--force", action="store_true", help="ignore automod.enabled")
    s.add_argument("--item-id", type=int, default=None,
                   help="backlog item this round implements; the review rung grades against its clauses")
    s.add_argument("--from-branch", default=None,
                   help="resume a branch the review rung sent back (automod/SM_…), rebased onto live main")
    g = sub.add_parser("gate"); g.add_argument("round_id"); g.add_argument("--skip-smoke", action="store_true")
    g.add_argument("--land-on-pass", metavar="BY", default=None,
                   help="start the landing as soon as the gate passes (the reaper's gates)")
    l = sub.add_parser("land"); l.add_argument("round_id"); l.add_argument("--dry-run", action="store_true")
    l.add_argument("--force", action="store_true", help="ignore automod.enabled")
    a = sub.add_parser("abort"); a.add_argument("round_id")
    a.add_argument("--reason", default="", help="recorded on the ledger")
    sub.add_parser("status")
    b = sub.add_parser("bless", help="record the running commit as last-known-good")
    b.add_argument("--note", default="")
    rc = sub.add_parser("recover", help="clear BROKEN/halted and start the stack")
    rc.add_argument("--by", default=None,
                    help="who is clearing, for the halt-clear ledger row (e.g. alan)")
    # A restart by hand looked like a crash to the guardian and fired every
    # alert channel, and the pause-drain-restart procedure was five manual
    # steps in CLAUDE.md. This is that procedure, with the lease the promoter
    # uses so the guardian is blind for exactly the restart.
    r = sub.add_parser("restart", help="pause the pool, drain, restart mcp+backend under the guardian lease")
    r.add_argument("--reason", default="", help="recorded on the ledger")
    r.add_argument("--only", action="append",
                   choices=["lloyd-mcp", "lloyd-backend", P.PRIMARY_PROGRAM],
                   help="restart only this program (repeatable); default both, mcp first. "
                        "agent-llm-primary takes its own leg: host-RAM wait, reread, long boot")
    r.add_argument("--force", action="store_true", help="even while a promotion is under observation")
    r.add_argument("--skip-idle", action="store_true",
                   help="emergency: pause the pool but do not wait for idle; every turn in flight "
                        "dies and is re-offered (for when the idle wait itself is broken)")
    fl = sub.add_parser("flush", help="restart once for every landing the train holds, "
                                      "and open its observation window")
    fl.add_argument("--now", action="store_true",
                    help="do not wait for implement turns or idle; every turn in flight "
                         "dies with the restart and is re-offered")
    fl.add_argument("--by", default="cli", help="who asked, for the ledger")
    fl.add_argument("--reason", default="")
    sc = sub.add_parser("scorecard", help="how the unattended loop is doing, from the ledger")
    sc.add_argument("--since", default="7d")
    sc.add_argument("--json", action="store_true")
    sc.add_argument("--record", action="store_true", help="append the row to scorecard.jsonl")
    sub.add_parser("cluster", help="group the open backlog by what it is about "
                                   "(scripts/automod/cluster.py; its own flags pass through: "
                                   "--write --threshold --no-judge --json …)")
    sub.add_parser("board-pass", help="run autocode's board passes once, now, with board "
                                      "health before and after (the reaper excepted)")
    uo = sub.add_parser("unfold-oversized", help="unfold never-attempted umbrellas whose contract "
                                                 "has at least --min-clauses clauses; members go "
                                                 "back to draft for the sweep to rank")
    uo.add_argument("--min-clauses", type=int, default=8)
    uo.add_argument("--dry-run", action="store_true")
    sub.add_parser("sweep-status", help="how far the backlog sweep has got: unread, ranked, "
                                        "parked, and the last batches")
    bd = sub.add_parser("board-decisions", help="what the queue decisions produced (#904): every "
                                                "promotion into up_next with its deciding event "
                                                "and terminal state, per-day counts, and "
                                                "retirements later reopened")
    bd.add_argument("--days", type=int, default=7)
    bd.add_argument("--summary", action="store_true", help="omit the per-item listings")
    pb = sub.add_parser("priority-backfill", help="write the default priority (low) onto every "
                                                  "backlog item, any board or status, whose "
                                                  "priority is absent, `none` or unknown")
    pb.add_argument("--dry-run", action="store_true")
    pb.add_argument("--reset-open", action="store_true",
                    help="also write low onto every OPEN item whatever it carries, so the "
                         "high tier starts empty; the old value is kept in the activity log")
    # `cluster` hands everything after it to cluster.py's parser. REMAINDER
    # on a subparser does not swallow `--flags`, so the unknowns are collected
    # here instead of refused.
    args, extra = ap.parse_known_args(argv)
    if args.cmd != "cluster" and extra:
        ap.error(f"unrecognized arguments: {' '.join(extra)}")

    if args.cmd == "start":
        print(json.dumps(start(args.goal, force=args.force, item_id=args.item_id,
                               from_branch=args.from_branch), indent=2))
    elif args.cmd == "gate":
        rep = run_gate(args.round_id, skip_smoke=args.skip_smoke,
                       land_on_pass=args.land_on_pass)
        print(json.dumps(rep, indent=2))
        return 0 if rep["ok"] else 1
    elif args.cmd == "land":
        print(json.dumps(land(args.round_id, dry_run=args.dry_run,
                              force=args.force), indent=2))
    elif args.cmd == "abort":
        print(json.dumps(abort(args.round_id, reason=args.reason), indent=2))
    elif args.cmd == "status":
        print(json.dumps(status(), indent=2, default=str))
    elif args.cmd == "bless":
        print(json.dumps(bless(args.note), indent=2, default=str))
    elif args.cmd == "recover":
        # `--by` has to be forwarded or the flag is a lie in `--help`: the whole
        # point of the halt-clear ledger row is to name whoever lifted a freeze,
        # and dispatching `recover()` here dropped the name on the floor and
        # recorded the pid fallback instead (#1365, review round 2).
        print(json.dumps(recover(by=args.by), indent=2, default=str))
    elif args.cmd == "restart":
        programs = tuple(args.only) if args.only else ("lloyd-mcp", "lloyd-backend")
        print(json.dumps(P.restart_stack(programs, reason=args.reason, force=args.force,
                                         skip_idle=args.skip_idle),
                         indent=2, default=str))
    elif args.cmd == "flush":
        print(json.dumps(flush(now=args.now, by=args.by, reason=args.reason),
                         indent=2, default=str))
    elif args.cmd == "scorecard":
        from scripts.automod import scorecard as SC
        row = SC.compute(since_days=SC.parse_since(args.since))
        print(json.dumps(row, indent=2) if args.json else SC.render(row))
        if args.record:
            print(f"\nrecorded → {SC.record(row)}")
    elif args.cmd == "cluster":
        from scripts.automod import cluster as CL
        return CL.main(extra)
    elif args.cmd == "board-pass":
        print(json.dumps(board_pass(), indent=2, default=str))
    elif args.cmd == "unfold-oversized":
        from scripts.automod import backlog as B
        print(json.dumps(B.unfold_oversized_umbrellas(S.LEDGER_PATH, min_clauses=args.min_clauses,
                                                      dry_run=args.dry_run),
                         indent=2, default=str))
    elif args.cmd == "sweep-status":
        print(json.dumps(sweep_status(), indent=2, default=str))
    elif args.cmd == "board-decisions":
        from scripts.automod import backlog as B, board_decisions as BD
        reading = BD.board_decisions(S.LEDGER_PATH, items=B.all_items(None), days=args.days)
        print(json.dumps(BD.summary(reading) if args.summary else reading,
                         indent=2, default=str))
    elif args.cmd == "priority-backfill":
        from scripts.automod import backlog as B
        rows = B.backfill_priority(None, dry_run=args.dry_run, reset_open=args.reset_open)
        if not args.dry_run and rows:
            S.append_event({"event": "priority_backfill", "by": "human", "reset_open": args.reset_open,
                            "written": [r["item_id"] for r in rows if r.get("written")],
                            "skipped": [r["item_id"] for r in rows if not r.get("written")]})
        print(json.dumps({"dry_run": args.dry_run, "count": len(rows), "items": rows},
                         indent=2, default=str))
    return 0


def sweep_status() -> dict:
    """Where the sweep stands: the board's `sweep` block, worth × size over
    the open items, and the last five batches."""
    from scripts.automod import backlog as B
    health = B.board_health(S.LEDGER_PATH)
    items = B.open_items()
    grid: dict[str, int] = {}
    for i in items:
        if i.worth or i.size:
            key = f"{i.worth or '?'}/{i.size or '?'}"
            grid[key] = grid.get(key, 0) + 1
    rows = [d for d in B._ledger_events(S.LEDGER_PATH, "backlog_sweep", require_item=False)]
    recent = [{k: d.get(k) for k in ("batch_id", "verdict", "retired", "duplicates", "kept",
                                    "parked", "unjudged", "num_turns", "created_at")}
              for d in rows[-5:]]
    return {"enabled": B.sweep_enabled(), "sweep": health["sweep"], "draft": health["draft"],
            "up_next": health["up_next"], "worth_by_size": dict(sorted(grid.items())),
            "batches": len(rows), "recent": recent}


if __name__ == "__main__":
    raise SystemExit(main())
