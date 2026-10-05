"""Reaching the parent checkout's frontend dependencies from a linked worktree (#2233).

A test helper, not a test — the sibling of `dashboard_pins.py`, so `python -m pytest
tests/` collects nothing from here.

What it fixes. `tests/test_dashboard_responsive.py` skips every pin that needs the
served dashboard when `<this tree>/web/node_modules` has no vite, and the only thing
that ever put `node_modules` into a round's worktree was `gate.rung_frontend`
(`scripts/automod/gate.py`, `nm.symlink_to(live_web / "node_modules")`), which returns
early for a diff with no `web/` path. So in every non-frontend round the pins skipped,
and ten skips landed on a suite already at 31-35 against the `PYTEST_MAX_SKIPPED = 40`
ceiling — 41 or more, and the tests rung failed every such round for a reason no diff
could answer (item #2233, rounds `SM_20261005_083944` at 44 skips and
`SM_20261005_091634` at 45, both with the identical `DASHBOARD_PINS_NOT_EXECUTED`
finding, against a ceiling of 40). The live tree HAS the install; the round's tree just
could not see it.

What it does. Resolve the parent checkout — the main working tree whose
`.git/worktrees/<name>` administrative directory this one is — and, when THAT tree has
a working `web/node_modules/.bin/vite`, make this tree's `web/node_modules` a symlink to
it. A symlink, never a copy: the directory is 641 MB on this box and identical by
construction, because `package.json` and the lockfile are paths the gate denies. The
path is gitignored (`.gitignore:13`, `/web/node_modules`), so the link is invisible to
`git status` and never enters a diff.

Why the hook is here and not in the gate. `gate_detached` spawns
`python -m scripts.automod.round gate <rid>` with `cwd=LIVE_ROOT`, and `round.py`
imports `scripts.automod.gate` from the live tree — so a diff that edited only
`_run_suite` would be judged by the gate that predates it, which is how #2214's
ceiling change was refused and ended up as a human promotion. pytest, by contrast, runs
with `cwd=self.worktree`, so the round's OWN `tests/` is what executes the change: a
hook on this side is the one route that can land itself.

What it never does. It does not touch a tree that already has a `web/node_modules`,
real directory or symlink — nothing is created, nothing is removed, nothing is
overwritten. It does not create directories. It does not raise: a tree it cannot link
(no parent checkout, a parent with no vite, a read-only `web/`) is reported as
`UNAVAILABLE` and the pins keep skipping for the reason they have always named, because
the alternative — turning an unavailable dependency into an error — would redden every
box where nobody ran `npm install`, which is the outcome
`tests/dashboard_pins.py`'s docstring exists to argue against.

Call sites, two and both deliberate: `tests/conftest.py` at import, which is where the
gate's parallel invocation reaches it (every xdist worker imports the conftest, so the
race this module is race-safe against is a real one, not a hypothetical), and
`test_dashboard_responsive.py::_vite_binary`, which is where the dependency is
resolved, so the answer is the same whether or not the suite conftest was loaded — the
scratch-tree nodes of `test_dashboard_pin_accounting.py` run the pin file with no
conftest at all.
"""

from __future__ import annotations

import os
from pathlib import Path

#: The link this module creates, and the tree it points at.
VITE_RELPATH = ("web", "node_modules", ".bin", "vite")

#: What `ensure_node_modules_link` reports, and the only three outcomes.
LINKED = "linked"          #: created a symlink to the parent checkout's node_modules
PRESENT = "present"        #: this tree already has a web/node_modules; nothing touched
UNAVAILABLE = "unavailable"  #: no parent checkout, or its vite is not there; nothing created


def _git_file_gitdir(tree: Path) -> Path | None:
    """The `gitdir:` target of `tree/.git`, when `.git` is a FILE.

    A linked worktree's `.git` is a one-line pointer file
    (`gitdir: /repo/.git/worktrees/<name>`), while an ordinary checkout has a `.git`
    DIRECTORY and a submodule's pointer names `.git/modules/<name>`. Requiring the
    `worktrees` segment is what tells those three apart without shelling out to git —
    this runs at conftest import, in every worker, on trees that may not even have a
    git binary on PATH.
    """
    pointer = tree / ".git"
    if not pointer.is_file():
        return None
    try:
        text = pointer.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    if not text.startswith("gitdir:"):
        return None
    gitdir = Path(text.split(":", 1)[1].strip())
    if not gitdir.is_absolute():
        # normpath, not resolve: resolving would follow the very symlinks a round's
        # redirected HOME is made of and could name a different tree than git does.
        gitdir = Path(os.path.normpath(str(tree / gitdir)))
    return gitdir


def main_checkout(tree: Path) -> Path | None:
    """The main working tree this linked worktree belongs to, or None.

    None means "this tree has no parent to inherit from": an ordinary checkout, a
    scratch directory that is not a checkout, a submodule, or a linked worktree whose
    pointer names the tree itself.
    """
    tree = Path(tree)
    gitdir = _git_file_gitdir(tree)
    if gitdir is None or gitdir.parent.name != "worktrees":
        return None
    main = gitdir.parent.parent.parent          # <main>/.git/worktrees/<name>
    try:
        if main.resolve() == tree.resolve():
            return None
    except OSError:
        return None
    return main


def node_modules_of(tree: Path) -> Path:
    """This tree's own `web/node_modules` — a path that may not exist."""
    return Path(tree) / "web" / "node_modules"


def vite_of(tree: Path) -> Path:
    """The vite binary as THIS tree sees it (through the link, once there is one)."""
    return Path(tree).joinpath(*VITE_RELPATH)


def ensure_node_modules_link(tree: Path) -> str:
    """Make this tree's `web/node_modules` reach the parent checkout's, if it can.

    Returns one of `LINKED` / `PRESENT` / `UNAVAILABLE` and never raises. Two workers
    may run this at the same instant — every xdist worker imports the conftest that
    calls it — and the losing one gets `FileExistsError` from `symlink_to`, which is
    not a failure: the link it wanted is the link that now exists, to the same target.
    """
    tree = Path(tree)
    nm = node_modules_of(tree)
    try:
        # `lexists`, not `exists`: a real directory AND a symlink whose target has
        # gone are both an entry this module has no business replacing. Clause 3 of
        # #2233 is that a tree with its own node_modules is left untouched.
        if os.path.lexists(nm):
            return PRESENT
        main = main_checkout(tree)
        if main is None:
            return UNAVAILABLE
        parent_nm = node_modules_of(main)
        # The real binary, not the directory: a symlink whose target vanished
        # promises nothing (the same test `gate._run_suite` applies to its own
        # declaration).
        if not parent_nm.joinpath(".bin", "vite").exists():
            return UNAVAILABLE
        try:
            nm.symlink_to(parent_nm, target_is_directory=True)
        except FileExistsError:
            return PRESENT
        return LINKED
    except OSError:
        # No permission, no `web/` directory, a vanished parent mid-call: a skip
        # somewhere else in the suite is the right outcome; a dead conftest is not.
        return UNAVAILABLE
