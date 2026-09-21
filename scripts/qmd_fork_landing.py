#!/usr/bin/env python3
"""Is the qmd fork that production's vec leg serves from actually landed?

`~/lloyd/qmd` is a *separate git clone* — upstream `tobi/qmd`, our branch
`lloyd`, remote `origin` = `alansrobotlab2/qmd` — that the outer repo ignores
out of the gated tree: `.gitignore` carries `/qmd/`, `git ls-files qmd` is
empty, and there is no `.gitmodules`. Production's vector leg nonetheless runs
out of it: `agent-qmd-daemon` (see
`agent-services/supervisor/conf.d/agent-qmd-daemon.conf`) serves
`~/lloyd/qmd/dist/cli/qmd.js` on :8181. So the code that decides every vec-leg
retrieval score lives in a tree no gate rung builds or tests, whose sha appears
in no promotion record, and which no rollback can revert — the guardian's
`git stash push -u` / `reset --hard` / `clean -fd` all leave an ignored path
standing.

Backlog #854 settled the route for now: **a round does not edit the fork; fork
changes are human-landed.** This script is what makes "fork sha X is landed" a
checkable claim instead of a remembered one. It runs the fork's own two steps
and its own git state, and exits non-zero if any of them fails:

    build    node scripts/build.mjs      (the fork's `npm run build`)
    suite    node scripts/test-all.mjs   (the fork's `npm test`)
    git      branch + HEAD sha, clean tree, HEAD not ahead of its upstream

A landed fork is one whose source builds, whose suite passes, and whose commit
is on a branch with an upstream it is not ahead of. Anything else prints a
`FAIL <what>:` line naming the failed step and exits 1.

    python -m scripts.qmd_fork_landing                 # against ~/lloyd/qmd
    python -m scripts.qmd_fork_landing --fork /some/fork

Point it at a copy of the fork, not the live checkout, and make that copy a real
one: `cp -a ~/lloyd/qmd /tmp/qmd-check`. `cp -al` is not safe. The build writes
`dist/` *in place* in the tree it is pointed at, and a hardlink farm shares those
inodes with the served tree, so a build aimed "at a copy" can rewrite the live
artifact. Measured 2026-09-21, twice: the fork copied with `cp -al`, the check run
against the copy, and `~/lloyd/qmd/dist/cli/qmd.js` — never named on the command
line — took a fresh mtime at the exact second each build ran, ending with link
count 1 while the copy's own path moved to a different inode. The live tree was
clean at its pushed HEAD, so what it now holds is a build of the same commit it
came from and `find src -newer dist/cli/qmd.js` is empty; that is the reason no
damage resulted, not a property of the strategy. A running daemon serves whatever
it loaded, so even a rewritten live `dist/` changes nothing until the unit
restarts — which is the other reason this stays a latent hazard rather than an
outage.

One environment substitution can fail the fork's own suite for a reason that is
not a fork defect, and the reader has to know it before filing a verdict:
`test/cli.test.ts` asserts `status` and `doctor` print no line containing
`/home/`, `/Users/` or `/tmp/`, so a `HOME` under one of those prefixes trips
it. The same invocation, run the same day with `HOME` pointed at an empty scratch
directory, printed `step build: ok` and `FAIL suite:` on exactly that assertion
at the fork's pushed HEAD `ea2187e` — a correct verdict (non-zero, naming its
step) whose cause was the environment and not the fork.

Deliberately absent: this is not a gate rung. #854 route (a) — teaching
`scripts/automod/gate.py` to build and test the fork as a second tree, record
the fork sha in the promotion record and give the guardian a revert target — is
a human decision and is NOT implemented here.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_FORK = Path.home() / "lloyd" / "qmd"

# The fork's own two steps, invoked directly instead of through `npm run` /
# `bun run` so that editing the fork's package.json cannot quietly change what
# this check runs. Same files its `build` and `test` scripts name.
STEPS: tuple[tuple[str, str], ...] = (
    ("build", "scripts/build.mjs"),
    ("suite", "scripts/test-all.mjs"),
)

# The fork's suite is a typecheck, vitest, `bun test` and a package smoke; the
# build is a tsc pass plus bundling. Generous, because the alternative to a
# timeout is hanging a human waiting on a verdict.
STEP_TIMEOUT_SECONDS = 1800.0

# How much of a failing step's output to echo. Enough to attribute the failure;
# the fork's suite is verbose and the rest of it is the reader's own scroll.
TAIL_LINES = 25


@dataclass
class Report:
    """One run of the check: the lines to print, and whether it says landed."""

    fork: Path
    lines: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def say(self, line: str) -> None:
        self.lines.append(line)

    def fail(self, what: str, detail: str, output: str = "") -> None:
        """Record a failure. `what` is the step name, so output names it."""
        self.failures.append(what)
        self.say(f"FAIL {what}: {detail}")
        tail = [ln for ln in output.splitlines() if ln.strip()][-TAIL_LINES:]
        if tail:
            self.say(f"--- last {len(tail)} line(s) of {what} output ---")
            self.lines.extend(tail)
            self.say(f"--- end {what} output ---")

    @property
    def ok(self) -> bool:
        return not self.failures


def _bun_dir() -> str | None:
    """The directory holding the `bun` the fork's suite will get, or None.

    A bun the calling environment can already see wins — that is the one node's
    bare-name `spawnSync` finds. Otherwise the per-user install is consulted
    through `Path.home()`, which falls back to the passwd entry when `HOME` is
    unset: the shape of every non-login environment on this box (cron,
    systemd-run, a worker), where a `$HOME`-built path would silently resolve to
    nothing and the suite would die on `spawnSync bun ENOENT`.
    """
    found = shutil.which("bun")
    if found:
        return str(Path(found).parent)
    home_bun = Path.home() / ".bun" / "bin" / "bun"
    return str(home_bun.parent) if home_bun.exists() else None


def _child_env() -> dict[str, str]:
    """Environment for the fork's steps, with `bun` reachable by bare name.

    `qmd/scripts/test-all.mjs:37` runs the Bun suite as
    `run("Bun test suite", "bun", ...)`, a bare-name `spawnSync`. Two ordinary
    environments cannot satisfy it, and neither is the fork's fault: a
    *gate-shaped* one, because `scripts/automod/gate.py:508` hands the test rung
    `PATH=/usr/bin:/bin` on purpose so a developer's `$PATH` cannot turn a red
    suite green; and a *non-login* one, because bun is installed per-user at
    `~/.bun/bin/bun` and reaches `$PATH` only through `.bashrc`/`.profile`.

    So the directory is prepended when it resolves. When nothing resolves the
    suite step is refused by name (`check_steps`) rather than allowed to fail
    after its node half printed green — an ENOENT from mid-suite reads as a
    broken fork, which is a different finding from this one.
    """
    env = dict(os.environ)
    bun_dir = _bun_dir()
    if bun_dir and bun_dir not in env.get("PATH", "").split(os.pathsep):
        env["PATH"] = f"{bun_dir}{os.pathsep}{env.get('PATH', '')}"
    return env


def _run(cmd: list[str], cwd: Path, timeout: float) -> subprocess.CompletedProcess:
    """Run one step, capturing its output so a failure can be quoted back."""
    try:
        return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                              timeout=timeout, env=_child_env())
    except FileNotFoundError:
        # The interpreter itself is unresolvable — a stripped PATH with no node.
        # A named step failure, not a traceback out of main().
        return subprocess.CompletedProcess(
            cmd, returncode=127, stdout="", stderr=f"{cmd[0]}: command not found")
    except subprocess.TimeoutExpired as exc:
        # text=True above, so any partial output is already str.
        partial = exc.stdout if isinstance(exc.stdout, str) else ""
        return subprocess.CompletedProcess(
            cmd, returncode=124,
            stdout=partial + f"\n<timed out after {timeout:.0f}s>", stderr="")


def _git(fork: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(fork), *args], capture_output=True,
                          text=True)


def check_steps(report: Report, fork: Path, *, timeout: float = STEP_TIMEOUT_SECONDS) -> None:
    """Run the fork's build then its suite. Both run even if the build fails,
    so one invocation reports every problem instead of one at a time."""
    node = shutil.which("node")
    if not node:
        report.fail("node", "'node' is not on PATH, so neither fork step can run")
        return
    # Print the interpreters before any step output, so "the fork's suite is
    # red" and "this environment cannot run the fork's suite" are separable
    # findings in the log rather than one red step.
    report.say(f"tool node: {node}")
    bun_dir = _bun_dir()
    report.say(f"tool bun: {bun_dir + '/bun' if bun_dir else 'NOT FOUND'}")
    for step, rel in STEPS:
        script = fork / rel
        if not script.is_file():
            report.fail(step, f"the fork has no {rel} ({script})")
            continue
        if step == "suite" and bun_dir is None:
            # Refused by name rather than run: the suite spawns `bun` by bare
            # name only at test-all.mjs:37, after its node halves print green,
            # so an ENOENT there reads as a broken fork.
            report.fail(step, "the fork's suite spawns `bun` by bare name "
                              "(qmd/scripts/test-all.mjs:37) and none resolves on "
                              "this PATH or under ~/.bun/bin, so the suite cannot "
                              "be judged from this environment")
            continue
        started = time.monotonic()
        done = _run([node, str(script)], cwd=fork, timeout=timeout)
        took = time.monotonic() - started
        if done.returncode != 0:
            report.fail(step, f"'node {rel}' exited {done.returncode} in {took:.1f}s",
                        (done.stdout or "") + (done.stderr or ""))
        else:
            report.say(f"step {step}: ok — node {rel} in {took:.1f}s")


def check_git_state(report: Report, fork: Path) -> None:
    """Print the branch and HEAD sha, and refuse a fork that is dirty or has
    commits its upstream has never seen. That pair is what makes the printed
    sha mean *landed*, not merely *checked out*."""
    if _git(fork, "rev-parse", "--git-dir").returncode != 0:
        report.fail("git", f"{fork} is not a git repository")
        return

    branch = _git(fork, "symbolic-ref", "--short", "-q", "HEAD")
    if branch.returncode != 0:
        report.fail("git", "HEAD is detached, so no branch carries this commit")
        head = _git(fork, "rev-parse", "HEAD").stdout.strip()
        report.say("branch: (detached HEAD)")
        report.say(f"head:   {head or '(unknown)'}")
        return
    name = branch.stdout.strip()

    head = _git(fork, "rev-parse", "HEAD").stdout.strip()
    report.say(f"branch: {name}")
    report.say(f"head:   {head}")

    dirty = _git(fork, "status", "--porcelain").stdout.splitlines()
    if dirty:
        shown = ", ".join(ln.strip() for ln in dirty[:10])
        more = f" (+{len(dirty) - 10} more)" if len(dirty) > 10 else ""
        report.fail("dirty", f"the fork's working tree has {len(dirty)} uncommitted "
                             f"change(s): {shown}{more}")

    upstream = _git(fork, "rev-parse", "--abbrev-ref", "--symbolic-full-name",
                    "@{upstream}")
    if upstream.returncode != 0:
        report.fail("upstream", f"branch {name} has no upstream, so nothing has "
                                f"received {head}")
        return
    remote_branch = upstream.stdout.strip()
    ahead = _git(fork, "rev-list", "--count", f"{remote_branch}..HEAD").stdout.strip()
    behind = _git(fork, "rev-list", "--count", f"HEAD..{remote_branch}").stdout.strip()
    report.say(f"upstream: {remote_branch} — {ahead} ahead, {behind} behind")
    if ahead.isdigit() and int(ahead) > 0:
        report.fail("ahead", f"branch {name} has {ahead} commit(s) origin has never "
                             f"received, so {head} is not landed")


def check(fork: Path, *, timeout: float = STEP_TIMEOUT_SECONDS) -> Report:
    """The whole verdict for one fork path."""
    report = Report(fork=fork)
    report.say(f"fork: {fork}")
    if not fork.is_dir():
        report.fail("fork", f"no fork at {fork}")
        return report
    # Git state before the steps, and the order is not cosmetic: `node
    # scripts/build.mjs` writes `dist/`, which is the directory production serves
    # on :8181. Building a fork whose working tree is dirty would compile
    # uncommitted source into the artifact the running daemon is serving, so the
    # check refuses a dirty or unpushed fork without touching its build at all.
    check_git_state(report, fork)
    if not report.ok:
        # The build would compile this dirty source into the served dist/, and
        # the verdict is already NOT LANDED — nothing is learned by running it.
        report.say("steps: skipped — the fork's git state already refuses it, and "
                   "building would compile this tree into the dist/ production serves")
    else:
        check_steps(report, fork, timeout=timeout)
    if report.ok:
        branch = _git(fork, "symbolic-ref", "--short", "-q", "HEAD").stdout.strip()
        head = _git(fork, "rev-parse", "HEAD").stdout.strip()
        remote = _git(fork, "rev-parse", "--abbrev-ref", "--symbolic-full-name",
                      "@{upstream}").stdout.strip()
        report.say(f"LANDED: branch {branch} @ {head} is clean, built, tested and "
                   f"pushed to {remote}")
    else:
        # Deliberately not `NOT LANDED:` — a caller grepping for the success
        # marker `LANDED:` must not be able to match the failure line too.
        report.say(f"NOT LANDED (failed check(s): {', '.join(report.failures)})")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qmd_fork_landing",
        description="Prove the qmd fork's landed state: its own build, its own "
                    "suite, and its branch/HEAD sha. Non-zero on any failure.")
    parser.add_argument("--fork", type=Path, default=DEFAULT_FORK,
                        help=f"fork checkout to check (default: {DEFAULT_FORK})")
    parser.add_argument("--step-timeout", type=float, default=STEP_TIMEOUT_SECONDS,
                        help=f"seconds per step (default: {STEP_TIMEOUT_SECONDS:.0f})")
    args = parser.parse_args(argv)

    report = check(args.fork.expanduser().resolve(), timeout=args.step_timeout)
    for line in report.lines:
        print(line)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
