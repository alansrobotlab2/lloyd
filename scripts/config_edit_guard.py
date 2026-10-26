#!/usr/bin/env python3
"""Guard a direct edit to `config.yaml` before it reaches `main`.

Background (#3352). Autonomy task #40 (`nightly-reflection-config`) edits
`config.yaml` unattended, on `main`, with no work branch — the skill says so in
its own words, and its only stated validation is `yaml.safe_load`. Nothing else
on that path checks *which* key changed: `config.yaml` is not in
`PROTECTED_WRITE_ROOTS` (`grep -n 'PROTECTED_WRITE_ROOTS'
app/harness/protected_paths.py`), the checkout has no active git hook, and
`automod_vault_land` cannot carry the file (`CONTRACT_PATHS` in
`scripts/automod/vault_round.py` is `SOUL.md`/`MEMORY.md`/`USER.md` only). So an
unattended run could rewrite the engine's own switches — the loop's `enabled`,
the worker pool's `slots`, a model slot, an endpoint or a credential leaf — and
have it become the next restart's behaviour on a YAML-parse pass.

The fence this needs already exists and is already tested: `CONFIG_DENIED_KEYS`
(`grep -n 'CONFIG_DENIED_KEYS' scripts/automod/spec.py`), `CONFIG_DENIED_LEAVES`
and the pure differ `config_value_change` in that file. What it lacked was a
caller outside the automod gate — `check_scope` in the same file was its only
production caller, so the fence only ever judged a *round's* config edit, never
a nightly job's. This module is that caller: it answers, for the on-disk
`config.yaml` against `HEAD`, whether any changed value sits under a denied key
prefix or a denied leaf name, and it refuses naming the offending key path and
the rule that caught it.

The trap in the differ, pinned by `tests/test_config_edit_guard.py`:
`config_value_change` returns `ok=False` for a comment-only edit too, with the
reason `no value changed` and an empty path list. That is *not* a refusal — the
automod lane admits a comment-only edit — so this module maps it to its own
status (`NO_VALUE_CHANGED`, allowed) rather than collapsing it into `DENIED`. A
caller that reads `ok is False` as "denied" would refuse every comment edit.

Two statuses, one boolean, three outcomes:

  * `VALUE_CHANGED`   — allowed. At least one changed path, none denied.
  * `NO_VALUE_CHANGED` — allowed. Nothing but comments/layout moved, or the file
    is byte-identical to `HEAD`. Distinguishable from a refusal by `status`.
  * `DENIED`          — not allowed. `reason` names the offending key path(s)
    and the rule, exactly as the differ worded them.
  * `OFF_MAIN`        — not allowed, commit mode only. `HEAD` is on a branch, so
    committing would strand the change off `main` — the failure
    `skills/nightly-reflection-config/SKILL.md` records as 8+ nightly commits
    stuck on an unmerged `nightly-improvement-*`.

Usage:

  cd ~/lloyd
  .venvs/lloyd/bin/python scripts/config_edit_guard.py           # check, exit 0/2
  .venvs/lloyd/bin/python scripts/config_edit_guard.py --json
  .venvs/lloyd/bin/python scripts/config_edit_guard.py commit -m "nightly-reflection: config 2026-10-26"

`commit` stages `config.yaml` and nothing else: it commits with `config.yaml` as
the only pathspec, so a dirty index belonging to another writer is never swept
into this commit, and it stays on `main` rather than creating a branch. A
refusal leaves the edit in the working tree for the caller to revert
(`git checkout -- config.yaml`); it never commits, and never reverts.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.automod import spec  # noqa: E402  (the fence lives there; #3352 reuses it)

#: The file this guard exists to protect, relative to the repo root.
CONFIG_PATH = "config.yaml"

#: The exact reason `spec.config_value_change` gives when the two documents
#: parse to the same mapping. Keep in step with that function, which is the
#: source of the string; `tests/test_config_edit_guard.py` pins the pair.
NO_VALUE_CHANGED_REASON = "no value changed"

VALUE_CHANGED = "value_changed"
NO_VALUE_CHANGED = "no_value_changed"
DENIED = "denied"
OFF_MAIN = "off_main"

EXIT_ALLOWED = 0
EXIT_DENIED = 2


@dataclass(frozen=True)
class ConfigEditVerdict:
    """What the fence says about the working tree's `config.yaml` vs `HEAD`.

    `allowed` is the only boolean a caller must branch on. `status` is what
    makes a comment-only edit distinguishable from a refusal: both come back
    from `spec.config_value_change` with `ok=False`, so the outcome must be read
    here, not from that tuple's first element.
    """

    allowed: bool
    status: str
    reason: str
    changed_paths: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "status": self.status,
            "reason": self.reason,
            "changed_paths": list(self.changed_paths),
        }


@dataclass(frozen=True)
class CommitOutcome:
    """The result of `commit_config_edit`: the verdict plus whether a commit happened."""

    verdict: ConfigEditVerdict
    committed: bool
    detail: str

    @property
    def allowed(self) -> bool:
        return self.verdict.allowed


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
    )


def head_config(repo_root: Path) -> str | None:
    """`config.yaml` as committed at `HEAD`, or None when it is not there."""
    proc = _git(repo_root, "show", f"HEAD:{CONFIG_PATH}")
    if proc.returncode != 0:
        return None
    return proc.stdout


def current_branch(repo_root: Path) -> str:
    proc = _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    return proc.stdout.strip() if proc.returncode == 0 else ""


def check_config_edit(repo_root: Path | str = REPO_ROOT) -> ConfigEditVerdict:
    """Does the on-disk `config.yaml` change a value the fence denies?

    Compares the working-tree file against the copy committed at `HEAD` of
    `repo_root` through `spec.config_value_change` — the same differ the automod
    gate uses, so the fence is one rule held in one place, not a second key set
    that can drift from it. Refuses when a changed path sits under a
    `CONFIG_DENIED_KEYS` prefix or names a `CONFIG_DENIED_LEAVES` leaf, and the
    reason carries the differ's own wording naming the path and the rule.
    """
    root = Path(repo_root).resolve()
    before = head_config(root)
    if before is None:
        return ConfigEditVerdict(
            False,
            DENIED,
            f"`{CONFIG_PATH}` is not committed at `HEAD` of {root}, so there is "
            f"no committed document to diff against — refusing rather than "
            f"admitting an unverifiable edit",
        )
    on_disk = root / CONFIG_PATH
    if not on_disk.is_file():
        return ConfigEditVerdict(
            False,
            DENIED,
            f"`{CONFIG_PATH}` is not on disk in {root} — removing it is a value "
            f"change this guard does not admit",
        )
    after = on_disk.read_text()
    if before == after:
        return ConfigEditVerdict(
            True, NO_VALUE_CHANGED, "no value changed: identical to HEAD"
        )

    ok, reason, changed_paths = spec.config_value_change(before, after)
    if ok:
        return ConfigEditVerdict(True, VALUE_CHANGED, reason, tuple(changed_paths))
    if reason == NO_VALUE_CHANGED_REASON:
        # Comments and layout only. Admitted — and distinguishable from a
        # refusal by `status`, never by the differ's `ok` flag alone.
        return ConfigEditVerdict(True, NO_VALUE_CHANGED, reason, tuple(changed_paths))
    return ConfigEditVerdict(False, DENIED, reason, tuple(changed_paths))


def commit_config_edit(
    message: str, repo_root: Path | str = REPO_ROOT
) -> CommitOutcome:
    """Commit a `config.yaml` edit through the fence, on `main`, pathspec-only.

    Returns without committing when the fence refuses. `config.yaml` is the only
    pathspec, so whatever else is staged or dirty in the checkout stays the
    business of its own writer.
    """
    root = Path(repo_root).resolve()
    verdict = check_config_edit(root)
    if not verdict.allowed:
        return CommitOutcome(verdict, False, "not committed: the fence refused")

    branch = current_branch(root)
    if branch != "main":
        return CommitOutcome(
            ConfigEditVerdict(
                False,
                OFF_MAIN,
                f"`HEAD` is on branch `{branch}`, not `main` — a commit here "
                f"strands the change off `main`, which is what #3352's rule "
                f"forbids. Get onto `main` first.",
                verdict.changed_paths,
            ),
            False,
            "not committed: HEAD is not on main",
        )

    if _git(root, "diff", "--quiet", "HEAD", "--", CONFIG_PATH).returncode == 0:
        return CommitOutcome(
            verdict, False, f"nothing to commit: `{CONFIG_PATH}` matches HEAD"
        )

    committed = _git(root, "commit", "-m", message, "--", CONFIG_PATH)
    if committed.returncode != 0:
        return CommitOutcome(
            verdict,
            False,
            f"git commit failed: {committed.stderr.strip()[:400] or committed.stdout.strip()[:400]}",
        )
    sha = _git(root, "rev-parse", "HEAD").stdout.strip()
    return CommitOutcome(verdict, True, f"committed {sha[:10]} on main")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="config_edit_guard.py",
        description="Refuse a config.yaml edit that changes a fenced key, before it reaches main.",
    )
    parser.add_argument(
        "--repo",
        default=str(REPO_ROOT),
        help="repo root whose config.yaml is checked (default: this repo)",
    )
    parser.add_argument("--json", action="store_true", help="print the verdict as JSON")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("check", help="check the working tree against HEAD (default)")
    commit = sub.add_parser("commit", help="check, then commit config.yaml if allowed")
    commit.add_argument("-m", "--message", required=True, help="commit message")

    args = parser.parse_args(argv)
    repo = Path(args.repo).resolve()

    if args.command == "commit":
        outcome = commit_config_edit(args.message, repo)
        verdict = outcome.verdict
        detail = outcome.detail
    else:
        verdict = check_config_edit(repo)
        detail = ""

    if args.json:
        payload = verdict.to_dict()
        if args.command == "commit":
            payload["committed"] = outcome.committed
            payload["detail"] = outcome.detail
        print(json.dumps(payload))
    elif verdict.allowed:
        print(f"ALLOWED config.yaml edit [{verdict.status}]: {verdict.reason}")
        if detail:
            print(detail)
    else:
        print(f"REFUSED config.yaml edit [{verdict.status}]: {verdict.reason}", file=sys.stderr)
        if detail:
            print(detail, file=sys.stderr)

    return EXIT_ALLOWED if verdict.allowed else EXIT_DENIED


if __name__ == "__main__":
    raise SystemExit(main())
