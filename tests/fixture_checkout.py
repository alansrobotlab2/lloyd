"""One copy-list for the synthetic checkouts the IV grader tests build.

`scripts/iv_grade.py` is exercised as a real subprocess against a throwaway repo
root, so the fixture has to look like a checkout — `scripts/iv_grade.py` plus
every `app` module its imports reach. Listing those files per test file is how
the same suite broke twice: `paths.py` started importing `app.data_root` in
#1415 and every fixture had to learn it, and #1656 put
`app.inner_voice.session_input` (whose verdict comes from
`app.thinking_fidelity`) behind the grader's session read, which broke a third
time. The chain is therefore named once, here, next to the rule for keeping it
honest:

    every entry must be importable with the stdlib alone, and a fixture that is
    missing one dies with a loud `ModuleNotFoundError` in the subprocess rather
    than silently scoring nothing — so adding an import to the grader fails
    tests until the entry is added, and never passes quietly.

`app/__init__.py` is written empty by the callers, as they already did: the real
one is not part of what a grader fixture is testing.
"""

from __future__ import annotations

import shutil
from pathlib import Path

#: Every `app` module `scripts/iv_grade.py` reaches at import time, in the order
#: a reader checks them: the data-root pair, then the fidelity verdict the
#: session read consults (#1510, #1656).
IV_GRADER_APP_IMPORTS = (
    "app/paths.py",
    "app/data_root.py",
    "app/thinking_fidelity.py",
    "app/inner_voice/__init__.py",
    "app/inner_voice/session_input.py",
)


def copy_app_import_chain(root: Path, repo: Path) -> None:
    """Copy `IV_GRADER_APP_IMPORTS` from the real checkout `root` into `repo`.

    Creates intermediate directories, so a caller only has to have made `repo`
    itself. Copies rather than symlinks for the reason every caller already
    documented: the grader anchors its data root at
    `Path(__file__).resolve().parents[1]`, and a symlink resolves back to the
    live checkout, which would leave the suite reading production data.
    """
    for rel in IV_GRADER_APP_IMPORTS:
        dest = repo / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, dest)
