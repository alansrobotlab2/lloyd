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
import subprocess
from pathlib import Path

from scripts.selfmod import gate as G, promote as P, spec, state as S, worktree as W

LIVE_ROOT = Path(__file__).resolve().parent.parent.parent


def _round_id() -> str:
    from scripts.autoresearch.common import round_id
    return round_id().replace("R_", "SM_")


def start(goal: str, *, base: str | None = None, force: bool = False) -> dict:
    """Open a round: take the lock, cut a worktree, write the run spec."""
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
        # blocked `selfmod_start` for #448 over one orphaned file while the
        # human was working on something else entirely. Recorded, not refused
        # — the gate and the promoter check the paths that actually matter,
        # which are the ones this round's diff will overlap.
        dirty = W.dirty_paths(LIVE_ROOT)
        rid = _round_id()
        base = base or subprocess.run(
            ["git", "-C", str(LIVE_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip()

        W.prune_orphans(LIVE_ROOT)
        wt = W.create(rid, base=base, repo=LIVE_ROOT)

        run_spec = {
            "objective": goal,
            "evaluation": {"command": "scripts.selfmod.gate", "timeout_secs": 3600},
            "budget": {"max_rounds": 1, "max_variants_per_round": 1},
            "mutation_scope": {"writable_paths": list(spec.ALLOWED_GLOBS)},
            "code": {"base_commit": base, "branch": f"selfmod/{rid}",
                     "worktree": str(wt)},
        }
        err = spec.validate_code_run_spec(run_spec)
        if err:
            W.remove(rid)
            raise RuntimeError(f"run spec invalid: {err}")

        out = S.ROUNDS_DIR / rid
        out.mkdir(parents=True, exist_ok=True)
        import yaml
        (out / "run_spec.yaml").write_text(
            yaml.safe_dump(run_spec, sort_keys=False), encoding="utf-8")

        S.append_event({"event": "round_start", "round_id": rid, "base": base,
                        "goal": goal[:500], "worktree": str(wt),
                        **({"live_dirty_paths": dirty[:20]} if dirty else {})})
        out_d = {"round_id": rid, "worktree": str(wt), "base": base,
                 "branch": f"selfmod/{rid}",
                 "run_spec": str(out / "run_spec.yaml")}
        if dirty:
            out_d["live_dirty_paths"] = dirty[:20]
            out_d["note"] = ("the live tree has uncommitted edits; the gate tolerates them "
                             "unless this round changes the same paths")
        return out_d
    finally:
        lock.release()


def run_gate(round_id: str, *, skip_smoke: bool = False) -> dict:
    wt = W.worktree_path(round_id)
    if not wt.exists():
        raise RuntimeError(f"no worktree for {round_id}")
    spec_path = S.ROUNDS_DIR / round_id / "run_spec.yaml"
    import yaml
    base = yaml.safe_load(spec_path.read_text())["code"]["base_commit"]

    g = G.Gate(round_id, wt, base, skip_smoke=skip_smoke)
    report = g.run()
    S.write_gate_report(round_id, report.to_dict())
    # Preflight may have rebased the round onto a moved `main`. The spec is
    # where the NEXT gate call reads its base from, and a stale base there
    # makes `changed_paths` sweep the human's commits into the round's diff.
    if report.base != base:
        S.update_run_spec_base(round_id, report.base)
    return report.to_dict()


def land(round_id: str, *, dry_run: bool = False, force: bool = False) -> dict:
    """Promote a round whose gate passed. Refuses otherwise."""
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
    lock = S.Lock(owner=f"land-{round_id}").acquire()
    try:
        result = P.promote(round_id, wt, report["base"],
                           gate_report=report, dry_run=dry_run)
    finally:
        lock.release()
    if not dry_run:
        W.remove(round_id, keep_branch=False, repo=LIVE_ROOT)
    return result


def abort(round_id: str) -> dict:
    W.remove(round_id, keep_branch=True, repo=LIVE_ROOT)
    S.append_event({"event": "round_aborted", "round_id": round_id})
    return {"aborted": round_id, "branch_kept": f"selfmod/{round_id}"}


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
        "unit_drift": _unit_drift(),
        "worktrees": W.prune_orphans(LIVE_ROOT),
        "recent": S.read_events(limit=15),
    }


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
    from scripts.selfmod.promote import _get, BACKEND
    head = subprocess.run(["git", "-C", str(LIVE_ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    if not head:
        raise RuntimeError("cannot read HEAD")
    _, body = _get(f"{BACKEND}/health")
    running = (body or {}).get("commit")
    if running and running != head:
        raise RuntimeError(
            f"the running backend reports {running[:8]} but HEAD is {head[:8]} — "
            "restart the backend before blessing, or you will pin a commit that "
            "is not the code being served")
    if S.read_current():
        raise RuntimeError("a promotion is under observation — let it settle or "
                           "abort it rather than blessing over it")
    lkg = S.write_lkg(head)
    S.append_event({"event": "settled", "commit": head,
                    "note": f"blessed by hand: {note}" if note else "blessed by hand"})
    return {"last_known_good": lkg, "verified_running": running}


def recover(clear_broken: bool = True, clear_halt: bool = True) -> dict:
    """Come back from BROKEN: clear the flags and start the stack again.

    The guardian deliberately leaves services STOPPED when it escalates — an
    honestly-dead system beats a half-reverted one. Nothing then started them
    again, and the documented recovery was "clear the flag", which leaves the
    box down. This is the other half.
    """
    from app.supervisor_client import start_process
    out: dict = {"cleared": [], "started": []}
    if clear_broken and S.is_broken():
        S.BROKEN_PATH.unlink()
        out["cleared"].append("BROKEN")
    if clear_halt and S.is_halted():
        S.clear_halted()
        out["cleared"].append("promotions-halted")
    S.clear_rollback_request()
    for program in ("lloyd-mcp", "lloyd-backend"):
        ok, msg = start_process(program)
        out["started"].append(f"{program}: {msg}")
    S.append_event({"event": "recovered", "cleared": out["cleared"],
                    "started": out["started"]})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Self-modification rounds")
    sub = ap.add_subparsers(dest="cmd", required=True)
    # --force exists because the interlock now covers the CLI as well as the
    # MCP tools. It was only ever checked in the tool wrapper, and the skill's
    # own worked examples drive the CLI — so "the loop ships inert" was true of
    # the surface nobody used and false of the one they did.
    s = sub.add_parser("start"); s.add_argument("goal")
    s.add_argument("--force", action="store_true", help="ignore selfmod.enabled")
    g = sub.add_parser("gate"); g.add_argument("round_id"); g.add_argument("--skip-smoke", action="store_true")
    l = sub.add_parser("land"); l.add_argument("round_id"); l.add_argument("--dry-run", action="store_true")
    l.add_argument("--force", action="store_true", help="ignore selfmod.enabled")
    a = sub.add_parser("abort"); a.add_argument("round_id")
    sub.add_parser("status")
    b = sub.add_parser("bless", help="record the running commit as last-known-good")
    b.add_argument("--note", default="")
    sub.add_parser("recover", help="clear BROKEN/halted and start the stack")
    args = ap.parse_args(argv)

    if args.cmd == "start":
        print(json.dumps(start(args.goal, force=args.force), indent=2))
    elif args.cmd == "gate":
        rep = run_gate(args.round_id, skip_smoke=args.skip_smoke)
        print(json.dumps(rep, indent=2))
        return 0 if rep["ok"] else 1
    elif args.cmd == "land":
        print(json.dumps(land(args.round_id, dry_run=args.dry_run,
                              force=args.force), indent=2))
    elif args.cmd == "abort":
        print(json.dumps(abort(args.round_id), indent=2))
    elif args.cmd == "status":
        print(json.dumps(status(), indent=2, default=str))
    elif args.cmd == "bless":
        print(json.dumps(bless(args.note), indent=2, default=str))
    elif args.cmd == "recover":
        print(json.dumps(recover(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
