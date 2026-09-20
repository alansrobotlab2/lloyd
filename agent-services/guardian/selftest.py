"""Guardian self-check: can it still *act*, not merely still run?

Used three ways, and they cannot all want the same verdict:

  * `guardian-stage.sh` runs it with `--profile staging` before promoting a new
    snapshot, so a broken guardian is declined at stage time and the previous
    one keeps running;
  * the running guardian runs the default `daily` profile, because a watchdog
    that silently lost socket permissions looks identical to a healthy one
    right up until the day it matters. `agent_mcp/main.py`'s /health docstring
    makes the same argument for the aggregator;
  * a human runs `guardian.py --selftest`, which is also the full profile.

Two profiles because the checks answer two different questions. Five of them
are properties of the *candidate*: can it read its rollback target, read the
repo, write and fsync its state dir, still recognise a vault wipe, still parse
PSI. Three are properties of the *machine at this moment*: whether
supervisord's socket exists, whether the qualified names resolve through it,
whether anything at all answers a health probe.

Staging judges the candidate only. Judging the stack there was backlog #1302:
`agent-supervisord.service` is `Type=simple`, so `lloyd-guardian.service`'s
`After=`/`Wants=` buy ordering and not readiness, the `ExecStartPre` stage runs
2-3 s into the boot, and `/tmp/agent-supervisor.sock` is not there yet.
`guardian-stage.sh` maps any non-zero exit to `REFUSING`, so the first boot
after a guardian fix lands is exactly the boot that declines to stage it —
which happened on both boots of 2026-09-19 (18:56:48, 19:54:11), each followed
an instant later by an error-level page and a HIGH-priority backlog item
(#1279, #1280) that the same cold boot filed against itself.

The alert half is not this file's to fix — `maybe_selftest` judges the full
profile on its first tick with no boot grace, discards the per-check detail,
and that is #1178's. What changed here is only *which checks gate staging*. The
rule the split protects is still the strong one: the snapshot is replaced only
by a copy that has proved it can still perform its own preconditions. A check
the cold boot cannot pass is not a gate on a candidate's health; it is a gate on
the boot's timing.

Stdlib only. Exit 0 = healthy.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

#: Every check, including the three that need the rest of the stack answering.
#: This is what the running guardian executes daily, and what a human gets from
#: `guardian.py --selftest`.
PROFILE_DAILY = "daily"
#: Only the checks that are a property of the candidate, so they can be judged
#: with nothing running. `guardian-stage.sh` gates promotion on this profile.
PROFILE_STAGING = "staging"
PROFILES = (PROFILE_DAILY, PROFILE_STAGING)

#: Said beside a check the chosen profile does not judge, so whoever reads the
#: output can tell "not asked" from "asked and passed".
SKIPPED_NOTE = "stack-dependent; judged by the daily profile"


def _check(name: str, fn, verbose: bool) -> bool:
    try:
        ok, detail = fn()
    except Exception as exc:
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    if verbose:
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def run(g, verbose: bool = True, profile: str = PROFILE_DAILY) -> bool:
    """Judge the guardian's preconditions under one profile; True if all judged checks pass.

    The profile selects *which* checks are asked, never *how* they are judged:
    the same functions, the same thresholds, the same `[ok ]`/`[FAIL]` lines. A
    check the profile excludes prints `[skip]`, is not executed, and cannot move
    the verdict — so `staging` passing is a weaker claim than `daily` passing,
    and the daily run inside the guardian is what still makes the strong one.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown selftest profile {profile!r}, "
                         f"expected one of {', '.join(PROFILES)}")

    import rollback as rb
    from supervisor import SupervisordUnreachable

    checks: list[tuple] = []          # (name, fn, stack_dependent)

    def supervisord():
        try:
            state = g.sup.get_state()
            return True, f"statename={state.get('statename')}"
        except SupervisordUnreachable as exc:
            return False, str(exc)[:120]
    checks.append(("supervisord reachable", supervisord, True))

    def names():
        procs = g.sup.all_process_info()
        missing = [p for p in g.programs if p not in procs]
        return (not missing), (f"resolved {len(g.programs)} names" if not missing
                               else f"unresolvable: {missing}")
    checks.append(("qualified names resolve", names, True))

    def lkg():
        target, source = g.state.rollback_target()
        if not target:
            return False, source
        if not rb.commit_exists(g.repo, target):
            return False, f"{target[:8]} not in the object store"
        return True, f"{target[:8]} via {source}"
    checks.append(("rollback target readable and real", lkg, False))

    def git():
        head = rb.head_commit(g.repo)
        branch = rb.head_branch(g.repo)
        if not head:
            return False, "cannot rev-parse HEAD"
        return True, f"{head[:8]} on {branch}"
    checks.append(("repo readable", git, False))

    def writable():
        g.gdir.mkdir(parents=True, exist_ok=True)
        fd, path = tempfile.mkstemp(dir=g.gdir)
        try:
            os.write(fd, b"selftest")
            os.fsync(fd)
        finally:
            os.close(fd)
            os.unlink(path)
        return True, str(g.gdir)
    checks.append(("state dir writable + fsyncable", writable, False))

    def vault_tripwire():
        # Judged on synthetic snapshots first, so a staged copy that can no
        # longer recognise a wipe is declined even on a healthy vault.
        import vaultwatch as vw
        base = vw.Snapshot(ts=0.0, total=5000, top={"backlog": 1000}, inode=1)
        wiped = vw.Snapshot(ts=5.0, total=12, top={}, inode=1)
        if not vw.evaluate([base], wiped):
            return False, "a 99% drop did not trip"
        if vw.evaluate([base], vw.Snapshot(ts=5.0, total=4990, top={"backlog": 995}, inode=1)):
            return False, "ordinary churn tripped"
        snap = vw.measure(g.vault.root)
        if snap is None:
            return False, f"cannot read {g.vault.root}"
        return True, f"{snap.total} files, tripped={bool(g.vault.tripped())}"
    checks.append(("vault tripwire judges and measures", vault_tripwire, False))

    def memory_pressure():
        # The parse on a fixed sample, then the host's PSI file: a kernel
        # without it leaves the recorder blind, which is worth failing on.
        import memwatch as mw
        p = mw.parse_pressure("some avg10=1.50 avg60=0.20 avg300=0.00 total=9\n"
                              "full avg10=41.00 avg60=3.00 avg300=0.10 total=7\n")
        if (p.get("full") or {}).get("avg10") != 41.0:
            return False, f"pressure parse wrong: {p}"
        host = mw.read_pressure(g.mem.host)
        if host is None:
            return False, f"cannot read {g.mem.host}"
        return True, f"host full avg10={(host.get('full') or {}).get('avg10')}"
    checks.append(("memory-pressure recorder reads PSI", memory_pressure, False))

    def endpoints():
        import probes
        import policy
        b = probes.probe(g.backend_url, policy.PROBE_TIMEOUT_SECONDS)
        m = probes.probe(g.mcp_url, policy.PROBE_TIMEOUT_SECONDS)
        # Reachability, not health: a down service is the guardian's *job*,
        # not a guardian fault. Only a total inability to probe is a failure.
        reachable = (b["status"] is not None) or (m["status"] is not None)
        return reachable, f"backend={b['status']} mcp={m['status']}"
    checks.append(("health endpoints reachable", endpoints, True))

    staging = profile == PROFILE_STAGING
    judged = [(name, fn) for name, fn, stack in checks if not (staging and stack)]
    skipped = [name for name, fn, stack in checks if staging and stack]

    if verbose:
        print(f"guardian selftest [{profile}]:")
    results = [_check(n, f, verbose) for n, f in judged]
    if verbose:
        for name in skipped:
            print(f"  [skip] {name}: {SKIPPED_NOTE}")
    ok = all(results)
    if verbose:
        print(f"  => {'PASS' if ok else 'FAIL'} ({sum(results)}/{len(results)})")
    return ok


def cli(argv: list[str]) -> int:
    """Standalone entry point — what `guardian-stage.sh` execs, and what a human runs.

    Takes `--profile` plus every `guardian.py` flag, so a caller can point the
    run at an absent supervisor socket, unreachable health endpoints and a
    scratch state dir instead of the live ones. That is how
    `tests/test_guardian_selftest.py` puts this through the real boundary —
    argv in, exit code and stdout out — rather than calling `run` and trusting
    the argument plumbing that `guardian-stage.sh` actually depends on.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import guardian as G
    parser = G.build_parser()
    parser.add_argument("--profile", choices=PROFILES, default=PROFILE_DAILY,
                        help="which checks to judge: 'daily' is all eight, "
                             "'staging' is only the five that hold without the "
                             "stack (what guardian-stage.sh gates on)")
    args = parser.parse_args(argv)
    return 0 if run(G.Guardian(args), verbose=True, profile=args.profile) else 1


if __name__ == "__main__":
    sys.exit(cli(sys.argv[1:]))
