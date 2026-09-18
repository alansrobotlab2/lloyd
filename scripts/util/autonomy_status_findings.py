#!/usr/bin/env python3
"""Name autonomy task-status transitions sitting in a staged tree, before commit.

Item #1127. A task file whose `status` moves into `draft` or `paused` stops
dispatching: `_all_runnable_tasks` (`autonomy.py:323`) drops everything outside
`("up_next", "in_progress", "failed")`, and the queue's own source
(`workers/sources/scheduled_task.py:174`) skips anything that is not `up_next`.
Nothing anywhere named such a change before the commit that carried it, so vault
commit `6f657fa9` was able to move #68 (`frequency: every-15min`, the fleet's
highest-volume task) from `up_next` to `draft` inside a nightly pre-flight commit
whose own message certified three *unrelated* prose files as unshrunk. #68 then
ran nothing for ~30 h while looking configured.

This is a REPORT, not a gate. `up_next -> in_progress` is a job claiming its own
task, and blocking that is how a guard gets disabled inside a week — so: print
one line per transition, exit 0, let the commit proceed. Clause 3 of the item is
what keeps it honest: a staged task file whose front matter moved but whose
`status` did not produces NO finding. A permanent alarm is a disabled alarm.

`scripts/util/vault-commit.sh` calls this after staging and before committing, so
what it reports depends on the staged tree alone — never on which job is
committing. A job that commits inline instead of through the wrapper
(nightly-reflection-knowledge-write stages named directories on purpose) runs the
identical check itself:

    python3 ~/lloyd/scripts/util/autonomy_status_findings.py --repo ~/obsidian

An added task file is not a transition — a new task legitimately starts in
`draft` — so files with no HEAD version are skipped. A deleted task file is
named by the commit's own file list, and is a different event than a status flip.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# The two values that stop a job dispatching, so a reader can tell a clobber from
# a claim without opening the file. This is `autonomy.DISPATCH_STOPPING_STATUSES`
# spelled out rather than imported on purpose: `autonomy.py` pulls `yaml`,
# `agent_mcp._shared` and `app.paths` at module level, and this script runs under
# whatever interpreter a shell happens to have (`/usr/bin/python3` on this box has
# no `mcp`, so importing `autonomy` there raises ModuleNotFoundError).
# tests/test_autonomy_status_change_audit.py imports BOTH and fails on drift,
# which is the only way two definitions can stay one fact.
DISPATCH_STOPPING_STATUSES = ("draft", "paused")

_STATUS_RE = re.compile(r"^status:[ \t]*(.*)$")


def _clean_value(raw: str) -> str:
    """Strip whitespace, one layer of quotes, and a trailing YAML comment."""
    value = raw.strip()
    if not value:
        return ""
    value = re.sub(r"[ \t]+#.*$", "", value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def front_status(text: str) -> str:
    """The `status:` value from a markdown file's YAML front matter, or "".

    Read with a regex over the first `---` block rather than a YAML parser: this
    runs on a staged tree that may contain a file whose front matter no parser on
    this box can load, and a check that crashes on damage is a check that reports
    nothing on the exact commit that needed it.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return ""
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            for line in lines[1:i]:
                match = _STATUS_RE.match(line)
                if match:
                    return _clean_value(match.group(1))
            return ""
    return ""


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(("git", "-C", str(repo)) + args,
                          capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        # Exit 128 on `git show HEAD:path` is normal for an added file; callers
        # that mean it pass through "". Anything else raises and becomes
        # CHECK FAILED in the output, never a silent zero.
        raise RuntimeError(f"git {' '.join(args)}: {proc.stderr.strip()[:200]}")
    return proc.stdout


def staged_task_paths(repo: Path) -> list[str]:
    """Task files changed in the index (staged), vault-relative, sorted."""
    out = _git(repo, "diff", "--cached", "--name-only", "-z", "--", "autonomy/")
    return sorted(p for p in out.split("\0") if p.endswith(".md"))


def status_findings(repo: Path) -> tuple[list[str], int]:
    """(findings, number of staged task files examined) for one repo's index.

    The count is returned alongside because "0 findings" over "0 files examined"
    is no verdict — it is the same-looking output as a clean tree, and a check
    that cannot tell the two apart is not a check.
    """
    findings: list[str] = []
    checked = 0
    for rel in staged_task_paths(repo):
        old_text = ""
        new_text = ""
        try:
            old_text = _git(repo, "show", f"HEAD:{rel}")
        except RuntimeError:
            old_text = ""          # added file: no prior status to transition from
        try:
            new_text = _git(repo, "show", f":{rel}")
        except RuntimeError:
            new_text = ""          # deleted file: reported by the commit's file list
        old, new = front_status(old_text), front_status(new_text)
        checked += 1
        if not old or old == new:
            continue               # added file, or front matter moved but status did not
        note = f"{rel}: status {old} -> {new}"
        if new in DISPATCH_STOPPING_STATUSES:
            note += " (dispatch-stopping)"
        findings.append(note)
    return findings, checked


def report(repo: Path) -> int:
    """Print the rung. Always returns 0: this reports, it does not block."""
    try:
        findings, checked = status_findings(repo)
    except Exception as exc:                       # noqa: BLE001 - never block a commit
        print(f"autonomy-status: CHECK FAILED ({exc})", flush=True)
        return 0
    print(f"autonomy-status: {checked} staged autonomy task file(s) checked, "
          f"{len(findings)} status finding(s)", flush=True)
    for finding in findings:
        print(f"autonomy-status FINDING: {finding}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=os.environ.get("VAULT_DIR", str(Path.home() / "obsidian")),
                        help="git repo to inspect (default $VAULT_DIR or ~/obsidian)")
    args = parser.parse_args(argv)
    return report(Path(args.repo).expanduser())


if __name__ == "__main__":
    sys.exit(main())
