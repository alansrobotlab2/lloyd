#!/usr/bin/env python3
"""#2877 — commit the injection-canary corpus the persistence-window fire just wrote.

WHY THE PRODUCER HAS TO BE THE COMMITTER. `lloyd-persistence-window.service` appends
about 30 rows / ~190 KB to the git-tracked
`eval/measurements/injection-canary/rows.jsonl` and writes one dated
`run-<UTC date>-persistence.md` on every Monday-02:10 fire, through its own
`ExecStartPost` (`run_injection_canary.py report --write --force`). Nothing committed
any of it, so every fire leaves the production checkout dirty and the dirt accumulates:
as of 2026-10-17 the whole-checkout `git status --porcelain` is exactly
` M eval/measurements/injection-canary/rows.jsonl` and
`?? eval/measurements/injection-canary/run-2026-10-15-persistence.md` — the 2026-10-15
window, whose commit #2866 closed with as owed and never made. Retention deliberately
will not take the job: `sweep_injection_canary`
(`scripts/groundskeeper/retention-sweep.py:3211`) documents its `apply` as "accepted and
ignored, on purpose… the corpus is git-tracked, so deleting it is a repository write",
and reports counts only. So the unit that writes the corpus is the only thing standing
beside it at the moment the rows land, which is why this is a second `ExecStartPost` and
not a new timer.

WHAT IT COMMITS, AND WHAT IT REFUSES TO COMMIT. One staging command, one directory::

    git -C /home/alansrobotlab/lloyd add -- eval/measurements/injection-canary/

then `git diff --cached --name-only`, and any staged path outside that directory —
somebody else's half-finished `git add` sitting in the live index — stops the step
before it commits anything: it stages nothing back (that would be editing another
writer's index) and leaves the working tree exactly as it found it. The commit it does
make is `git commit --only … -- eval/measurements/injection-canary/`, so the pathspec is
in the commit command too and a path that appeared between the two reads still cannot
ride along. Measured in a throwaway repo: that pair commits the modified tracked file
*and* the newly added dated report, and leaves a dirty file elsewhere in the tree
unmodified and unstaged — the shape `git log -1 --stat` is checked against after the
2026-10-19 fire.

Author identity is left at the repo default: no `-c user.*`, no `GIT_AUTHOR_*`, nothing.
That is what makes the sha-commit audit path read these the way it reads `1cdb06bd`
(`#2397: record the 2026-10-13 and 2026-10-14 persistence-arm windows`), the hand-made
commit this step retires. #668 is the separate, still-open question of distinguishing a
job-authored commit from an accepted one, and this script deliberately does not answer it.

WHY EVERY PATH EXITS 0. systemd marks a `Type=oneshot` unit `Failed` when an
`ExecStartPost` returns non-zero, and the fire that earned this step has already spent
~34 min of held dispatch by the time it runs — so a commit refused for an ordinary
reason (an index lock, a foreign staged path, nothing written) must not read as a failed
window. This is the same reasoning the unit's own comment gives for its `--force`: a
refusal that is not a failure of the window may not change the window's result. A
non-zero exit here would also make the weekly `git status` alarm about this directory
indistinguishable from the window having broken.

WHO MAY DRIVE IT: THE UNIT, BY ITS CGROUP IDENTITY. An agent Bash turn typing `git
commit` in `/home/alansrobotlab/lloyd` is refused by `check_live_git_writes`
(`app/harness/live_git_writes.py:627`, wired at `app/harness/safety.py:518`), which
matches git verbs in the command string. This script's own git calls are its children,
not that string, so `python scripts/maintenance/commit_injection_canary_corpus.py` names
no verb and the lane does not answer it — measured:
`check_live_git_writes("…/python scripts/maintenance/commit_injection_canary_corpus.py",
…) is None`, asserted in `tests/test_safety_gate_discrimination.py`. So the caller check
has to be this script's, and it cannot be `INVOCATION_ID`'s presence. An agent Bash fork
on this box HAS that variable, because the backend it is forked from is itself a systemd
unit and the variable is exported down the whole process tree: measured twice in one turn,
`INVOCATION_ID=862d3b6355d542dea3b564f72aae7174` in both, the same value in both. Presence
would therefore admit exactly the caller the clause is about.

What separates the two callers is which unit the process runs in, and systemd records that
in `/proc/self/cgroup`. Measured on this box: an agent Bash fork is
`0::/user.slice/user-1000.slice/user@1000.service/lloyd.slice/`
`lloyd-bash-ceiling-15230-b609326e.scope` — a `.scope`, naming no service — while the main
process of a live user unit (`agent-supervisord.service`, PID 1695, `systemctl --user show
--value -p MainPID`) is
`0::/user.slice/user-1000.slice/user@1000.service/lloyd.slice/agent-supervisord.service`,
leaf equal to the unit name. systemd runs every `Exec*=` of a unit in that unit's cgroup,
so the 02:10 fire's commit step will carry a `lloyd-persistence-window.service` component
and nothing else on this machine will. The rail is therefore: `INVOCATION_ID` set AND a
cgroup component equal to `UNIT_NAME`. `INVOCATION_ID` stays in the conjunction because the
clause asks for it and because a caller without it cannot be a unit invocation at all.

Not a wall, and stated so a reader does not over-trust it: a turn that bind-mounted a
cgroup or moved its own process into the unit's cgroup could still drive this script, which
is a materially higher bar than setting an environment variable and is the open-set hole
#2780 argues about (`/proc/<pid>/cgroup` is readable but the set of unit names is not
enumerable from a policy file, so a closed set for it wants the same grant mechanism). The
raw verbs stay refused and pinned in `tests/test_safety_gate_discrimination.py` for that
reason, and the real-fire arm of this check is only closed by the 2026-10-19 fire.

`--repo` exists for the tests and for the scratch-worktree check the item asks for:
this script's contract is about ONE checkout, and no test may seed a dirty index in
production. The unit passes no arguments, so the shipped path is the default, and
`tests/test_persistence_window_unit.py` pins that the `ExecStartPost` line carries
nothing after the script path.

Run (the unit's own invocation): .venvs/lloyd/bin/python
    scripts/maintenance/commit_injection_canary_corpus.py
Tests: .venvs/lloyd/bin/python -m pytest tests/test_commit_injection_canary_corpus.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

#: The checkout whose corpus this step exists to commit, and the only one it will
#: commit unless a caller names another with `--repo`. The unit's `ExecStartPost` names
#: this script by absolute path inside it, so systemd hands it the same tree.
LIVE_REPO = "/home/alansrobotlab/lloyd"

#: The corpus directory, as a relative pathspec — the ONE directory this step may
#: stage. `eval/run_injection_canary.py`'s `OUT_DIR` is what writes into it, and
#: `RC.OUT_DIR.name == "injection-canary"` is pinned by `tests/test_persistence_window_unit.py`.
CORPUS_DIR = "eval/measurements/injection-canary"
CORPUS_PATHSPEC = CORPUS_DIR + "/"

#: The unit that fires this step, quoted in the commit message so a reader of
#: `git log` knows which job is speaking and does not look for an interactive author.
UNIT_NAME = "lloyd-persistence-window.service"
ITEM = "2877"

#: Where a process reads which unit it runs in. A parameter to `commit_corpus` so the
#: tests can hand it a witness file — a test cannot put itself in the window unit's cgroup
#: without starting the window — but never an argv flag: an option that took this path
#: would be a switch the very caller being refused could pass.
PROC_CGROUP = "/proc/self/cgroup"

#: Every line this script prints carries this prefix: the step's whole output surface
#: is the journal for one fire, so the reason has to be readable without a second line.
PREFIX = "commit_injection_canary_corpus:"

#: Git gets a timeout because the step runs under a `TimeoutStartSec=4h` unit beside a
#: checkout that an automod landing may be writing to at 02:10; a hung `index.lock` is
#: a reason to commit nothing, never a reason to hold the window open.
GIT_TIMEOUT_S = 120


def _log(detail: str) -> None:
    """One line to stdout, which under a systemd unit is the unit's journal.

    The whitespace collapse is the contract, not cosmetics: git's own failures are
    multi-line, and the one this step meets first — `fatal: not a git repository (or any
    parent up to mount point /)` followed by `Stopping at filesystem boundary
    (GIT_DISCOVERY_ACROSS_FILESYSTEM not set).` — would otherwise put a second journal
    entry beside the step's reason, which is what `journalctl -u
    lloyd-persistence-window.service` is read for after a Monday."""
    print(f"{PREFIX} {' '.join(detail.split())}", flush=True)


def _git(repo: str, *args: str) -> subprocess.CompletedProcess:
    """One git call against `repo`, capturing both streams — never a shell, so the
    pathspec reaches git as one argv token and cannot be re-split."""
    return subprocess.run(["git", "-C", repo, *args],
                          capture_output=True, text=True, timeout=GIT_TIMEOUT_S)


def _staged_paths(repo: str) -> list[str] | None:
    """Every path currently in the index against `HEAD`, or None if git would not say.

    `diff --cached` and not `status --porcelain`: the question is what a commit would
    take, and only the index answers that. Untracked-but-just-added files are in it,
    which is the whole point — the dated report is untracked until the line above.
    """
    proc = _git(repo, "diff", "--cached", "--name-only")
    if proc.returncode != 0:
        return None
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _outside_corpus(path: str) -> bool:
    return not (path == CORPUS_DIR or path.startswith(CORPUS_PATHSPEC))


def is_systemd_invocation(env: dict[str, str] | None = None) -> bool:
    """Whether `INVOCATION_ID` is set — necessary for a unit invocation, NOT sufficient.

    Kept because the clause asks for it and a caller without it cannot be a unit's own
    invocation. It is not the discriminator: an agent Bash fork on this box inherits the
    backend unit's value (measured twice in one turn,
    `INVOCATION_ID=862d3b6355d542dea3b564f72aae7174` both times), so a check that stopped
    here would admit the exact caller it exists to refuse.
    """
    return bool((os.environ if env is None else env).get("INVOCATION_ID"))


def cgroup_components(cgroup_file: str = PROC_CGROUP) -> list[str]:
    """The path components of this process's cgroup, from the v2 `0::<path>` line.

    `/proc/self/cgroup` is one line per hierarchy; v2 has exactly one, of the form
    `0::/user.slice/user-1000.slice/user@1000.service/lloyd.slice/<unit>.service`. An
    unreadable file returns an empty list, which fails closed: no unit identity, no
    commit. Read from the file rather than inferred from `getpid()` because the kernel
    writes it and nothing this process does can change what it says.
    """
    try:
        raw = Path(cgroup_file).read_text(encoding="utf-8")
    except OSError:
        return []
    for line in raw.splitlines():
        _, sep, path = line.partition("::")
        if sep and path.startswith("/"):
            return [part for part in path.split("/") if part]
    return []


def runs_in_window_unit(cgroup_file: str = PROC_CGROUP) -> bool:
    """Whether this process runs in the persistence-window unit's own cgroup.

    systemd places every `Exec*=` process of a unit in that unit's cgroup, so the fire's
    commit step carries a `lloyd-persistence-window.service` component and no other caller
    on this machine does. Measured here: a live user unit's main process is
    `…/lloyd.slice/agent-supervisord.service` (`agent-supervisord.service`, PID 1695),
    while an agent Bash fork is `…/lloyd.slice/lloyd-bash-ceiling-15230-b609326e.scope` —
    a `.scope` that names no service at all.

    Matched on a whole path component, not a substring, so a `lloyd-persistence-window.service.d`
    drop-in directory or a scope that merely contains the name inside a longer string
    cannot satisfy it — a `.d` directory is a real shape on this box and is not an
    invocation of the unit."""
    return UNIT_NAME in cgroup_components(cgroup_file)


def invocation_refusal(env: dict[str, str] | None = None,
                       cgroup_file: str = PROC_CGROUP) -> str | None:
    """Why this caller may not commit, or None if it may.

    The conjunction, each arm's reason named separately, because the journal line is the
    only evidence a Monday-morning reader gets that the step ran and stayed quiet."""
    if not is_systemd_invocation(env):
        return ("no INVOCATION_ID in the environment, so this is not a systemd unit's "
                "invocation")
    if not runs_in_window_unit(cgroup_file):
        seen = "/" + "/".join(cgroup_components(cgroup_file))
        return (f"not running in {UNIT_NAME}'s cgroup — /proc/self/cgroup reads "
                f"{seen!r} — so this caller is not the window's own fire")
    return None


def commit_corpus(repo: str = LIVE_REPO,
                  env: dict[str, str] | None = None,
                  cgroup_file: str = PROC_CGROUP) -> int:
    """Stage the corpus directory, commit it if and only if that is all there is.

    Returns 0 on every path, by contract (see the module docstring): the caller is
    `ExecStartPost`, and a step that refuses must not fail the window that earned it.
    """
    refused = invocation_refusal(env, cgroup_file)
    if refused:
        _log(f"refused: {refused}, and {repo} is not this caller's to commit in "
             f"(#{ITEM}); left the working tree untouched")
        return 0

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    add = _git(repo, "add", "--", CORPUS_PATHSPEC)
    if add.returncode != 0:
        _log(f"refused: `git add -- {CORPUS_PATHSPEC}` exited {add.returncode}: "
             f"{(add.stderr or add.stdout).strip()[:300]}; committed nothing in {repo}")
        return 0

    staged = _staged_paths(repo)
    if staged is None:
        _log(f"refused: `git diff --cached --name-only` failed, so what an index-wide "
             f"commit would take is unknown; committed nothing in {repo}")
        return 0

    foreign = sorted(p for p in staged if _outside_corpus(p))
    if foreign:
        _log(f"refused: {len(foreign)} of {len(staged)} staged path(s) are outside "
             f"{CORPUS_PATHSPEC} (first: {foreign[0]}), so committing would take "
             f"another writer's work; staged nothing back, committed nothing in {repo}")
        return 0

    if not staged:
        _log(f"nothing to commit: `git add -- {CORPUS_PATHSPEC}` staged no path in "
             f"{repo}, so this fire wrote no corpus rows or they were already recorded")
        return 0

    message = (f"#{ITEM}: record the {today} persistence-arm window "
               f"(auto-committed by {UNIT_NAME})")
    committed = _git(repo, "commit", "--only", "-m", message, "--", CORPUS_PATHSPEC)
    if committed.returncode != 0:
        _log(f"refused: `git commit --only -- {CORPUS_PATHSPEC}` exited "
             f"{committed.returncode}: "
             f"{(committed.stderr or committed.stdout).strip()[:300]}; left the "
             f"{len(staged)} staged path(s) in {repo} for the next fire")
        return 0

    sha = _git(repo, "rev-parse", "--short", "HEAD").stdout.strip() or "unknown"
    _log(f"committed {sha} in {repo}: {len(staged)} path(s) under {CORPUS_PATHSPEC}, "
         f"as {message}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=f"Commit the {CORPUS_PATHSPEC} corpus one "
                    f"{UNIT_NAME} fire just wrote (#{ITEM}).")
    ap.add_argument("--repo", default=LIVE_REPO,
                    help=f"checkout to commit in (default: the live one, {LIVE_REPO})")
    ns = ap.parse_args(argv)
    try:
        return commit_corpus(ns.repo)
    except Exception as exc:                                    # noqa: BLE001
        # The last-resort branch of the exit-0 contract: whatever the failure is, a
        # step that cannot commit must still leave the window's result alone. Named,
        # never swallowed — the reason is the only evidence a Monday-morning reader
        # gets that this step ran at all.
        _log(f"refused: {type(exc).__name__}: {exc}; committed nothing")
        return 0


if __name__ == "__main__":
    sys.exit(main())
