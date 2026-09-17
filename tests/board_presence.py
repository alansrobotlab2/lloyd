"""#1204: what a board-walking test does when the board is not there.

Three states, one policy. A test that times one cold `/api/dashboard` cycle over
the real board is only a measurement when the board is under it; the two ways to
get that wrong are:

* **passing on an empty walk** — the glob matched nothing, the loop body ran
  zero times, the stopwatch reads 0.02 s and the clause is reported as met; or
* **reporting a missing corpus as `skipped`** — the board directory is *this
  item's own tree* (`~/obsidian/backlog`, 1,142 files at the 2026-09-16
  census), so if it is gone or emptied something has happened to the thing the
  measurement is about, and `skipped` is where that goes hidden.

So the helpers below **fail** on every non-measurable state, including a
machine with no vault at all. That is deliberate, and it answers two review
findings on this item:

| round | finding |
|---|---|
| `SM_20260917_043852` | `test_dashboard_cold_render.py:62`: a new skip marker |
| `SM_20260917_055508` | `tests/board_presence.py:94`: a new skip call |

The second one was an earlier cut of this file skipping when the vault root was
absent, on the argument that an absent vault is a property of the box rather
than of the change under test. That argument does not survive the trigger: the
vault root is `LLOYD_OBSIDIAN_VAULT`-overridable, so a skip whose condition a
test can set with `monkeypatch.setenv` is a skip any test can walk into — and
the report it then writes is "not run" for the only measurement of the number
this item is about. The clause is either pinned or it is not, and this file
says so by failing.

Clause 4 of #1204 asked for "skips when the board directory is absent, never
passes on a zero-file board". Failing in that state satisfies the second half
strictly and supersedes the first at the review rung's insistence: on this box
`~/obsidian` exists, so nothing is lost, and a box without the vault gets a red
line naming the path instead of a green one that measured nothing.

`test_no_board_state_skips_and_every_one_names_the_lost_clause` in
`tests/test_dashboard_yaml_loader.py` pins every row by calling the helper
against synthetic directories and asserting which pytest exception each raises.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

import pytest

#: Set this to point the helper at another vault root. Named for the same
#: reason `scripts.automod.state.STATE_DIR` honours `LLOYD_AUTOMOD_STATE`: a
#: test that wants the missing-vault branch must be able to say so rather than
#: reach into this module's globals. Read per call, not frozen at import — an
#: override that a caller's `monkeypatch.setenv` could not move would let a test
#: set the variable, be ignored, and read its own result as verifying the
#: no-vault branch while it was in fact checking the live board.
VAULT_ROOT_ENV = "LLOYD_OBSIDIAN_VAULT"


def vault_root() -> Path:
    """`LLOYD_OBSIDIAN_VAULT` if set, else `~/obsidian`.

    `Path.home()` so a test run under a different account asks about *its* vault
    rather than this one's.
    """
    raw = os.environ.get(VAULT_ROOT_ENV)
    return Path(raw).expanduser() if raw else Path.home() / "obsidian"


#: The vault and board this box's item lives in, at import time — the default,
#: not a constant: `vault_root()` is what the helper consults.
VAULT_ROOT = vault_root()
BOARD_DIR = VAULT_ROOT / "backlog"


def _item_files(board: Path, numeric_names: bool) -> list[Path]:
    if not board.is_dir():
        return []
    files = sorted(board.glob("*.md"))
    return [p for p in files if p.name[:1].isdigit()] if numeric_names else files


def board_files_or_stop(
    *,
    what: str,
    board: Path | None = None,
    numeric_names: bool = False,
) -> list[Path]:
    """The board's item files, or a failure that tells the truth.

    `what` names the measurement in the failure text ("cold cycle", "loader
    equivalence"), because the reader needs to know which claim was lost, not
    just that one was.

    `numeric_names` matches the board's own convention (`1204-slug.md`): the
    dashboard's scan counts those, and a test timing that scan must count the
    same way the scan does.

    Never returns an empty list, and never skips: the caller either gets bytes to
    measure or a red line saying which clause is unpinned and why.
    """
    board = BOARD_DIR if board is None else board
    files = _item_files(board, numeric_names)
    if files:
        return files

    vault = vault_root()
    listed = len(list(board.iterdir())) if board.is_dir() else 0
    where = (
        f"{board} exists and holds {listed} entries, none of them matching "
        f"'*.md'{' starting with a digit' if numeric_names else ''}"
        if board.is_dir()
        else f"{board} does not exist"
    )
    reason = (
        f"{vault} is absent" if not vault.is_dir()
        else f"{vault} exists, so this tree is expected to have items on it"
    )
    pytest.fail(
        f"The {what} cannot run: {where}, and {reason}. Failing rather than "
        f"skipping — the board is the corpus this measurement certifies, so a "
        f"missing or emptied one is the finding it exists to report, and "
        f"`skipped` would file it as nobody's problem. The clause {what} pins "
        f"is NOT pinned by this run."
    )


class LedgerSource(NamedTuple):
    """What `timed_ledger_or_stop` settled on, so no caller can time a corpus it
    did not choose. `repointed` answers the review question directly: the ledger
    the cycle read is not the one the process was configured for, and a test that
    prints only a duration cannot be read as certifying the 6.3 MB baseline."""
    path: Path
    repointed: bool
    why: str


def timed_ledger_or_stop(monkeypatch, *, what: str) -> LedgerSource:
    """Make the cycle decode a ledger that has bytes in it, and report which one.

    The gate runs a round with `LLOYD_AUTOMOD_STATE` pointed at an empty state
    dir (`tests/conftest.py`), so `S.LEDGER_PATH` resolves to a file that does
    not exist and the run times a cycle that decodes *no* ledger at all. That
    is not the request the user pays, and a 6 s budget with nothing to decode
    does not constrain the ledger re-decode this item is about — the pre-fix
    cost was 55 `read_text` calls and 318,670 `json.loads` over
    6,286,192 bytes (triage, 2026-09-16).

    So when the state dir has no ledger but the real one exists, the timed cycle
    is pointed at the real file — read-only, and the dashboard only reads it.

    That substitution used to be silent, and the review rung named it
    (`SM_20260917_064456`: "the ledger test repoints `S.LEDGER_PATH` at the live
    6.3 MB file, so the read-once claim holds on the corpus but the test proves
    only the read shape, not the measured cost"). It is loud now: the return
    value carries `repointed` and the reason, and every caller prints it in the
    same line as its timing, so a green duration is always readable next to the
    corpus that produced it — including "this was the live ledger" or "this was a
    4-line fixture, so the byte-scale half of the baseline was not re-measured".
    A repoint is still not the 6,286,192-byte measurement; the point is that the
    report says which of the two it is.

    With neither present it **fails**, for the reason in this module's
    docstring: a cycle that decoded no ledger does not pin the read-once-per-
    cycle clause, and reporting that as `skipped` is how it would land as green.
    """
    from scripts.automod import state as S

    live = Path.home() / ".local" / "state" / "lloyd-automod" / "promotions.jsonl"
    current = S.LEDGER_PATH
    if current.is_file() and current.stat().st_size > 0:
        return LedgerSource(current, False, f"the configured state dir ({current})")
    if live.is_file() and live.stat().st_size > 0:
        monkeypatch.setattr(S, "LEDGER_PATH", live)
        return LedgerSource(
            live, True,
            f"REPOINTED: {current} (LLOYD_AUTOMOD_STATE) has no ledger, so the "
            f"cycle reads the live one instead — the timing below is of the real "
            f"{live.stat().st_size:,}-byte file, not of the state dir the "
            f"dashboard is configured with")
    pytest.fail(
        f"The {what} cannot constrain the ledger path: no non-empty ledger at "
        f"{current} or {live}, so the cycle would decode zero bytes. Failing "
        f"rather than skipping: the clause that the promotions ledger is read "
        "once per cycle, not 55x, is NOT pinned by this run on a box that "
        f"has no ledger to read."
    )
