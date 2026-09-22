"""The one rule for a read-only guard whose live data is gone (backlog #1377).

Imported by `test_trajectory_extraction.py` and `test_skill_verdicts.py`; not a
`test_*.py` file, for the reason `tests/_relief_harness.py` gives — pytest does not
collect it, so a broken helper is a collection error in both suites rather than a
mysterious red run.

Why a shared rule
-----------------
The guards in those two files read *data on the machine*, not files in the checkout:
`.gitignore` ignores `/sessions/` and `/_pipeline/`, so neither path is in git and
neither exists in a round's worktree. On 2026-09-22 a fixture teardown deleted the
production tree in 35 seconds (`tests/conftest.py:159-170` states it in code); the
tracked half came back from git by 13:02 PDT and the gitignored half only where a job
regenerated it, so `_pipeline/trajectories` and
`_pipeline/skills/reviews/verdicts.jsonl` stopped existing at all and
`~/lloyd/sessions` fell from the 3,000 files recorded on 2026-09-18 to 58.

Eight guards hard-asserted their presence. They therefore reproduced as red nodes
*at base* in every round's `tests` rung — `base_probe: probed 21 file(s) at base
1842b8cf: 106 already failing`, `external_blocker: true`, from
`~/.local/state/lloyd-automod/rounds/SM_20260922_201206/gate.json` — which blocks
every promotion after the one that noticed, not only that one.

The rule, and the line it must not cross
----------------------------------------
The absence of derived data is a property of the *machine* and gets a named skip whose
reason carries the missing path. Everything else still fails: a path that exists as
the wrong kind, a corpus that is present but empty or indiscriminate, a ledger that
exists but lost rows, a store whose classifier disagrees. A guard must therefore never
test existence itself and skip on the result — that is how a present corpus gets
skipped — it calls this function, which decides which of the two answers applies.
`test_an_absent_live_root_skips_every_guard_that_reads_it_by_name` and the tests
beside it pin both halves per guard.

What this does NOT decide: whether the automod gate's `tests` rung may go green on
named skips while a corpus it treats as required is absent. That is the fork backlog
#1377 step 3 leaves to a person.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def require_live_data(path: Path, what: str, kind: str = "dir") -> None:
    """Skip by name when a live-data root is genuinely absent; assert when it is not.

    `what` names the thing the guard reads (it appears in the skip reason beside the
    path) and `kind` is ``"dir"`` or ``"file"``. Two answers, deliberately:

      * the path does not exist -> ``pytest.skip`` naming it, because no round, no
        checkout and no re-run conjures derived data that was never regenerated;
      * the path exists but is not the kind of thing the guard reads -> a failing
        ``assert``, because that is a statement about the data, not the machine.
    """
    if not path.exists():
        pytest.skip(f"{what} absent at {path}: this machine holds no live copy to "
                    "read, so the guard has nothing it could check")
    if kind == "dir":
        assert path.is_dir(), f"{what} at {path} exists but is not a directory"
    else:
        assert path.is_file(), f"{what} at {path} exists but is not a file"


def require_live_volume(items, floor: int, root: Path, what: str,
                        noun: str = "files") -> None:
    """Skip a guard whose sample is below the size it needs to discriminate.

    Distinct from `require_live_data`: here the root is present and small, which is
    neither a machine with no data nor a full corpus. The reason names BOTH the floor
    and the observed count, because "vacuous" without the two numbers is unattributable
    — the whole reason `assert len(files) > 500, "so this is vacuous"` failed a round
    with no way to tell a 58-file store from a 0-file one.

    `noun` names what is being counted, because not every live store is counted in
    files: the autoresearch guards count *rounds* inside a frozen window, and a reason
    reading "holds 0 files" of a ledger would name the wrong unit. It defaults to
    "files" so the callers pinned before it existed keep their exact wording.
    """
    count = len(items)
    if count < floor:
        pytest.skip(f"{what} at {root} holds {count} {noun}, below the {floor}-{noun[:-1]} "
                    "floor under which this guard cannot discriminate either way")
