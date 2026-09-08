"""Tracked code may not depend on a file that is not tracked.

Why this file exists
--------------------
Three times on 2026-09-08 the self-modification gate's `tests` rung failed —
a hard rung, so every round aborts — for the same underlying reason, and none
of them was a bug in the change under test:

  * `tests/test_tool_overrides.py` asserted `data/tool_overrides.yaml`
    `.exists()`. That file is deliberately untracked and gitignored, `data/`
    has no tracked contents at all, and a worktree is a fresh checkout — so it
    could not pass in *any* round, for *any* diff, from `d11ad8c` onward.
    Three rounds aborted on it in fifteen hours while filing fourteen new
    backlog items and landing nothing.
  * `tests/test_prompt_surface_budget.py` (#377) read the live `~/obsidian`
    vault, which an hourly job rewrites — a tripwire on a hard rung, armed by
    a writer that is not a round at all.
  * and the near-miss that motivated this file: the fix for the second one
    introduced `prompt_surface.py`, imported by tracked tests. Committing the
    importers without `git add`-ing the module would have broken the rung in
    every worktree — the same shape a third time, in the change that was
    cleaning up the second.

The specific defects are fixed where they live. This is the general one: a
worktree contains exactly what is committed, so anything tracked code needs
must be committed too.

**It reads committed content, never the working tree.** That is not a detail.
A guard that inspected files on disk would fail on any author's uncommitted
work-in-progress — including the very refactor that introduces a new module
before adding it — and a test that fails on a dirty tree, on a hard rung, is
the exact class of defect this file exists to prevent. Committed state is also
precisely what the gate judges: the candidate commit, not the desk it was
written on.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True)


def _tracked_python_files() -> list[str]:
    r = _git("ls-files", "-z", "*.py")
    if r.returncode != 0:
        pytest.skip("not a git checkout")
    return [p for p in r.stdout.split("\0") if p]


def _committed(path: str) -> str | None:
    """The file as HEAD has it — not as the author currently has it open."""
    r = _git("show", f"HEAD:{path}")
    return r.stdout if r.returncode == 0 else None


def _local_target(module: str, tracked: set[str]) -> str | None:
    """The repo-relative file `module` resolves to, or None if it is external.

    Longest prefix first, so `scripts.selfmod.backlog` resolves to
    `scripts/selfmod/backlog.py` rather than stopping at a `scripts` package
    that may not exist. Anything that resolves nowhere in the repo is stdlib
    or a site-package and is not this test's business.
    """
    parts = module.split(".")
    for i in range(len(parts), 0, -1):
        stem = "/".join(parts[:i])
        for cand in (f"{stem}.py", f"{stem}/__init__.py"):
            if cand in tracked or (ROOT / cand).exists():
                return cand
    return None


def _imported_modules(source: str, path: str) -> set[str]:
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            # `level > 0` is a relative import: resolved within an already
            # tracked package, so it cannot name an untracked top-level file.
            if node.level == 0 and node.module:
                out.add(node.module)
    return out


def test_no_tracked_module_imports_an_untracked_one():
    """The whole point, in one assertion.

    Committing an importer without its module breaks the suite in every
    worktree while passing on the machine it was written on, because the
    author's copy has the file and the checkout does not.
    """
    tracked = set(_tracked_python_files())
    offenders: list[str] = []

    for path in sorted(tracked):
        source = _committed(path)
        if source is None:
            continue
        for module in sorted(_imported_modules(source, path)):
            target = _local_target(module, tracked)
            if target is not None and target not in tracked:
                offenders.append(f"{path} imports `{module}` -> {target}, which is not tracked")

    assert not offenders, (
        "tracked code depends on files git does not have, so a fresh worktree "
        "cannot run it and the selfmod gate's `tests` rung fails for every "
        "round regardless of its diff:\n  " + "\n  ".join(offenders)
        + "\n\n`git add` them, or the dependency is not real and should go."
    )


def test_the_guard_actually_resolves_local_modules():
    """A guard that silently matched nothing would be indistinguishable from a
    clean repo, and this one's whole job is to notice an absence."""
    tracked = set(_tracked_python_files())
    assert _local_target("prompt_builder", tracked) == "prompt_builder.py"
    assert _local_target("scripts.selfmod.backlog", tracked) == "scripts/selfmod/backlog.py"
    # External packages and stdlib resolve nowhere in the repo.
    assert _local_target("pytest", tracked) is None
    assert _local_target("pathlib", tracked) is None


def test_an_untracked_import_target_is_detected(tmp_path):
    """The failure path, exercised rather than assumed.

    `_local_target` must find a module that exists on disk but is absent from
    the tracked set — that combination *is* the bug, and a resolver that only
    looked at tracked paths would return None and report all-clear.
    """
    tracked = {"tests/importer.py"}
    scratch = ROOT / "_guard_probe_module.py"
    scratch.write_text("x = 1\n", encoding="utf-8")
    try:
        target = _local_target("_guard_probe_module", tracked)
        assert target == "_guard_probe_module.py"
        assert target not in tracked, "this is the shape the guard must catch"
    finally:
        scratch.unlink()


def test_it_reads_head_not_the_working_tree():
    """The property that keeps this guard off the hard-rung tripwire list.

    If it read from disk, every author with an unstaged new module would fail
    the suite — and on the selfmod gate that means every round aborts on
    somebody else's desk state.
    """
    tracked = _tracked_python_files()
    sample = "prompt_builder.py"
    assert sample in tracked
    assert _committed(sample) is not None
    assert _committed("definitely/not/a/path.py") is None
