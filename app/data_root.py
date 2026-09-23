"""The data-root resolution, readable without a venv.

`app.paths` is the system's one resolver and `architecture/data-home.md` is its
long version. Importing it, though, drags in the package and needs the project
venv — and the jobs that run outside the venv are exactly the ones that need the
root: cron, the guardian's units, `sh -c` children. Each of those re-implemented
the resolution as `${LLOYD_DATA:-~/lloyd-data}`, which is rule 2 with neither the
marker check nor rule 3. `scripts/groundskeeper/retention-sweep.py:59` was the one
that deletes, so its copy turned a sandbox or worktree `--apply` into a sweep of
the LIVE root — invisible to the Bash delete guard (a Python `unlink` is not an
`rm`) and to the tripwire, which needs ~10 % and 200 files gone inside 15 minutes
(~5,600 of the live root's ~56,000) to fire (#1415).

This module is the fix for the copy: the same three rules, in one stdlib-only
file that `app.paths` imports and re-exports and that a script can import with
plain `python3`. Its two constraints are that it imports nothing from `app` and
nothing third-party, and that importing it touches the filesystem only through
`pwd` and an `is_file()` — no `mkdir`, so importing it cannot create a root.

The three rules, first match wins (identical to `app.paths` before the move):

1. **`LLOYD_DATA`** in the environment. The gate, the canary and the test suite
   set it; nothing in production does, on purpose.
2. **The production checkout** (its tree is the passwd home's `lloyd` and is not a
   linked worktree) uses `<passwd home>/lloyd-data`, read off passwd and never off
   `Path.home()`: under a gate's `HOME` that name belongs to the round. The root
   must carry `.lloyd-data-root`; without the marker it raises
   `DataRootMissing` rather than falling back to the code tree, because a silent
   fallback starts a second copy of everything inside the tree — the failure the
   whole data move prevents.
3. **Any other checkout** — the sandbox, a round's worktree, a scratch clone —
   keeps its data inside itself under `.lloyd-data`, so the unsafe direction is
   never what happens by omission.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT_MARKER = ".lloyd-data-root"
#: The one environment variable that overrides the resolution. Also the name
#: `config.yaml`'s `${LLOYD_DATA}/…` keys expand against.
DATA_ROOT_ENV = "LLOYD_DATA"
#: The vault override `agent-services/guardian/vaultwatch.py:46` and
#: `scripts/backup/backup-vault.sh:28` already read; `vault_root()` honours the
#: same name rather than inventing a second knob.
VAULT_ROOT_ENV = "LLOYD_VAULT_ROOT"

try:
    import pwd as _pwd

    ACCOUNT_HOME = Path(_pwd.getpwuid(os.getuid()).pw_dir)
except (ImportError, KeyError):  # pragma: no cover - pwd exists on Linux
    ACCOUNT_HOME = Path.home()


class DataRootMissing(RuntimeError):
    """The production checkout found no marked data root."""


def live_checkout() -> Path:
    """The production code checkout: the passwd home's `lloyd`.

    Off passwd, not `$HOME`, for the reason in the module header — inside an
    automod gate `HOME` is the round's symlink farm, where `~/lloyd` *is* the
    worktree and `~/lloyd-data` is an empty directory.
    """
    return ACCOUNT_HOME / "lloyd"


def production_data_root() -> Path:
    """The live data root, for READERS that mean production on purpose.

    The same path whatever `$HOME` or `LLOYD_DATA` say — the review grader
    reading a round's sessions, the regression check reading the live store.
    Anything that sweeps, gzips or deletes uses `resolve_data_root_for_tree` (or
    `app.paths.DATA_ROOT`) instead, because this function ignores which tree it
    was called from. The readers-only wording was never the whole truth: four
    jobs write through it deliberately — the grader's own session file, the
    nightly extraction's lock, log and backups, the content hasher's index, the
    tool-override sync — and nothing enumerates them, which is the other half of
    #1415 and still owed.
    """
    return ACCOUNT_HOME / "lloyd-data"


PRODUCTION_DATA_ROOT = production_data_root()


def data_root_for_tree(tree: Path) -> Path:
    """Where a non-production checkout keeps its own runtime data."""
    return Path(tree) / ".lloyd-data"


def tree_is_worktree(tree: Path) -> bool:
    """Whether `tree` is a linked git worktree rather than the main checkout.

    `git worktree add` leaves `.git` as a one-line FILE (`gitdir: …`) where the
    main checkout has a DIRECTORY, so one `is_file()` answers it — no `$HOME`
    preference, no live-root computation, nothing to guess wrong.
    """
    return (Path(tree) / ".git").is_file()


def resolve_data_root(*, env: str | None, lloyd_home: Path, is_worktree: bool,
                      live_checkout: Path, production_root: Path) -> Path:
    """The three rules above, as a pure function of what they read.

    Every input is a parameter so a test can put the resolver in a state no
    checkout on this machine is in: an unset env in a tree that is not
    production, a production tree whose root lost its marker. The comparison and
    the marker check are `app.paths`' own, byte for byte — this function moved
    without changing which answer the same inputs get.
    """
    if env:
        return Path(env).expanduser()
    if lloyd_home == live_checkout.resolve() and not is_worktree:
        if not (production_root / DATA_ROOT_MARKER).is_file():
            raise DataRootMissing(
                f"{production_root} has no {DATA_ROOT_MARKER}: the production"
                " checkout keeps its runtime data there and refuses to fall back to"
                " the code tree. Restore it from ~/.lloyd-data-snapshots"
                " (scripts/backup/restore-data.sh), or set LLOYD_DATA explicitly."
            )
        return production_root
    return data_root_for_tree(lloyd_home)


def code_tree(here: Path | None = None) -> Path:
    """The checkout this module is running from.

    `here` is a parameter only so a test can name a tree that is not this one;
    every real caller leaves it unset and gets the module's own location.
    """
    return Path(here if here is not None else Path(__file__).resolve().parent.parent)


def resolve_data_root_for_tree(tree: Path | None = None) -> Path:
    """The whole resolution for one tree, reading the live environment.

    A script's one entry point: it takes the tree it lives in, so it does not
    import `app.paths` (and with it the venv) to find its own data root.
    `DataRootMissing` propagates — a production checkout with an unmarked root is
    a refusal, not a value.
    """
    home = code_tree(tree)
    return resolve_data_root(env=os.environ.get(DATA_ROOT_ENV),
                             lloyd_home=home,
                             is_worktree=tree_is_worktree(home),
                             live_checkout=live_checkout(),
                             production_root=production_data_root())


def vault_root() -> Path:
    """The Obsidian vault, derived exactly as `app.paths.VAULT_ROOT` is.

    `Path.home()` and not passwd: that is what `app.paths:110` does, and a
    second answer here would be the same divergence #1415 is about. It means
    that under a gate's `HOME` this is the round's `~/obsidian`, which
    `scripts/automod/worktree.py::ensure_round_home` *links* into the live vault
    (`HOME_LINK_SKIP` covers `lloyd` and `lloyd-data`, not `obsidian`) — so a
    write here reaches the live vault from a round, and only a symlink stops
    `rmtree`, not `write_text`. That is why the sweep's one vault write is
    overridable: `LLOYD_VAULT_ROOT` lets a round exercise it against a copy.
    """
    return Path(os.environ.get(VAULT_ROOT_ENV) or (Path.home() / "obsidian"))
