"""Guardian self-check: can it still *act*, not merely still run?

Used two ways:
  * `guardian-stage.sh` runs it before promoting a new snapshot, so a broken
    guardian is declined at stage time and the previous one keeps running;
  * the running guardian runs it daily, because a watchdog that silently lost
    socket permissions looks identical to a healthy one right up until the day
    it matters. `agent_mcp/main.py`'s /health docstring makes the same
    argument for the aggregator.

Stdlib only. Exit 0 = healthy.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def _check(name: str, fn, verbose: bool) -> bool:
    try:
        ok, detail = fn()
    except Exception as exc:
        ok, detail = False, f"{type(exc).__name__}: {exc}"
    if verbose:
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def run(g, verbose: bool = True) -> bool:
    import rollback as rb
    from supervisor import SupervisordUnreachable

    checks = []

    def supervisord():
        try:
            state = g.sup.get_state()
            return True, f"statename={state.get('statename')}"
        except SupervisordUnreachable as exc:
            return False, str(exc)[:120]
    checks.append(("supervisord reachable", supervisord))

    def names():
        procs = g.sup.all_process_info()
        missing = [p for p in g.programs if p not in procs]
        return (not missing), (f"resolved {len(g.programs)} names" if not missing
                               else f"unresolvable: {missing}")
    checks.append(("qualified names resolve", names))

    def lkg():
        target, source = g.state.rollback_target()
        if not target:
            return False, source
        if not rb.commit_exists(g.repo, target):
            return False, f"{target[:8]} not in the object store"
        return True, f"{target[:8]} via {source}"
    checks.append(("rollback target readable and real", lkg))

    def git():
        head = rb.head_commit(g.repo)
        branch = rb.head_branch(g.repo)
        if not head:
            return False, "cannot rev-parse HEAD"
        return True, f"{head[:8]} on {branch}"
    checks.append(("repo readable", git))

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
    checks.append(("state dir writable + fsyncable", writable))

    def endpoints():
        import probes
        import policy
        b = probes.probe(g.backend_url, policy.PROBE_TIMEOUT_SECONDS)
        m = probes.probe(g.mcp_url, policy.PROBE_TIMEOUT_SECONDS)
        # Reachability, not health: a down service is the guardian's *job*,
        # not a guardian fault. Only a total inability to probe is a failure.
        reachable = (b["status"] is not None) or (m["status"] is not None)
        return reachable, f"backend={b['status']} mcp={m['status']}"
    checks.append(("health endpoints reachable", endpoints))

    if verbose:
        print("guardian selftest:")
    results = [_check(n, f, verbose) for n, f in checks]
    ok = all(results)
    if verbose:
        print(f"  => {'PASS' if ok else 'FAIL'} ({sum(results)}/{len(results)})")
    return ok


if __name__ == "__main__":
    # Standalone mode for guardian-stage.sh: build a Guardian with defaults.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import guardian as G
    args = G.build_parser().parse_args([])
    sys.exit(0 if run(G.Guardian(args), verbose=True) else 1)
