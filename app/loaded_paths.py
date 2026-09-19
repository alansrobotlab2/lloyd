"""Which files of this tree THIS process has loaded as Python modules.

The automod promoter asks it of the backend and of the aggregator before a
landing (`scripts/automod/promote.py::restart_needed`). A landing restarts both
so that what landed is what runs — and a commit that changes no file either
process has loaded leaves the running code exactly as it was, so the drain,
the wait for every sibling round's turn and the restart buy nothing. On the
night of 2026-09-18 nine of nineteen promotions were that: tests, `eval/`,
docs and scripts that run fresh per invocation.

Asked of the process rather than kept as a list of "cold" directories, because
such a list rots the day something starts importing from one: the backend
imports `scripts/automod/**`, `scripts/autoresearch/**` and `scripts/vault/`
today, and nothing would have told a list about the next one. `sys.modules` is
the fact.

A module imported lazily AFTER the question is asked loads the landed file, so
it needs no restart either; the promoter asks a second time after the merge to
catch an import that raced it. Stdlib only, no `app` imports: the aggregator
serves this too.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def module_files() -> set[str]:
    """Real paths of every file backing a loaded module, `__main__` included."""
    out: set[str] = set()
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if isinstance(f, str) and f:
            try:
                out.add(os.path.realpath(f))
            except OSError:
                continue
    return out


def loaded(paths: list[str], root: Path | None = None) -> list[str]:
    """The members of `paths` (repo-relative) this process has loaded.

    A path need not exist: a file the landing deletes is still a loaded module
    until the process is replaced. Anything that is not a plain relative path
    inside the tree is reported as loaded — the caller's safe direction.
    """
    base = os.path.realpath(str(root or ROOT))
    files = module_files()
    hits: list[str] = []
    for p in paths:
        rel = str(p or "")
        full = os.path.realpath(os.path.join(base, rel))
        if not rel or os.path.isabs(rel) or not full.startswith(base + os.sep) or full in files:
            hits.append(rel)
    return hits


def answer(paths) -> dict:
    """The route body, for both servers."""
    if not isinstance(paths, list) or len(paths) > 5000:
        return {"error": "paths must be a list of at most 5000 repo-relative paths"}
    return {"loaded": loaded([str(p) for p in paths]), "pid": os.getpid(),
            "modules": len(sys.modules)}
