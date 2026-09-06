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


def start(goal: str, *, base: str | None = None) -> dict:
    """Open a round: take the lock, cut a worktree, write the run spec."""
    S.ensure_dirs()
    if S.is_halted():
        raise RuntimeError(f"promotions are halted: {S.HALTED_PATH.read_text().strip()}")
    if S.is_broken():
        raise RuntimeError(f"guardian is BROKEN: {S.BROKEN_PATH.read_text().strip()}")

    lock = S.Lock(owner="round-start").acquire()
    try:
        if not W.is_clean(LIVE_ROOT):
            raise RuntimeError("live tree is dirty — commit or stash before opening a round")
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
                        "goal": goal[:500], "worktree": str(wt)})
        return {"round_id": rid, "worktree": str(wt), "base": base,
                "branch": f"selfmod/{rid}",
                "run_spec": str(out / "run_spec.yaml")}
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
    out = S.ROUNDS_DIR / round_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report.to_dict()


def land(round_id: str, *, dry_run: bool = False) -> dict:
    """Promote a round whose gate passed. Refuses otherwise."""
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


def status() -> dict:
    return {
        "last_known_good": S.read_lkg(),
        "current": S.read_current(),
        "halted": S.is_halted(),
        "broken": S.is_broken(),
        "pause_remaining_s": round(S.pause_remaining(), 1),
        "worktrees": W.prune_orphans(LIVE_ROOT),
        "recent": S.read_events(limit=15),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Self-modification rounds")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start"); s.add_argument("goal")
    g = sub.add_parser("gate"); g.add_argument("round_id"); g.add_argument("--skip-smoke", action="store_true")
    l = sub.add_parser("land"); l.add_argument("round_id"); l.add_argument("--dry-run", action="store_true")
    a = sub.add_parser("abort"); a.add_argument("round_id")
    sub.add_parser("status")
    args = ap.parse_args(argv)

    if args.cmd == "start":
        print(json.dumps(start(args.goal), indent=2))
    elif args.cmd == "gate":
        rep = run_gate(args.round_id, skip_smoke=args.skip_smoke)
        print(json.dumps(rep, indent=2))
        return 0 if rep["ok"] else 1
    elif args.cmd == "land":
        print(json.dumps(land(args.round_id, dry_run=args.dry_run), indent=2))
    elif args.cmd == "abort":
        print(json.dumps(abort(args.round_id), indent=2))
    elif args.cmd == "status":
        print(json.dumps(status(), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
