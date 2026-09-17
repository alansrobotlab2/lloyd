"""#1204: what a board-walking test does when the board is not there.

Two states, and collapsing them is the defect. A test that times one cold
`/api/dashboard` cycle over the real board is only a measurement when the board
is under it; the two ways to get that wrong are:

* **passing on an empty walk** — the glob matched nothing, the loop body ran
  zero times, the stopwatch reads 0.02 s and the clause is reported as met; or
* **skipping on a wiped board** — the board directory is *this item's own tree*
  (`~/obsidian/backlog`, 1,142 files at the 2026-09-16 census), so if the vault
  root is there and the board is not, something has happened to the tree the
  measurement is about. Reporting that as `skipped` hides it; a gate run on such
  a tree would otherwise show two green-not-run timings and a clause nobody
  checked.

So the rule the helpers below implement is three-way, and the distinction is
the vault root:

| the tree | result |
|---|---|
| board dir has item files | return them, measure |
| vault root exists, board empty or missing | **fail** — a moved or emptied board |
| vault root itself absent | **skip**, naming the path — a machine with no vault at all |

The middle row is the one a plain `skipif` gets wrong, and
`test_board_presence.py` (in `tests/test_dashboard_cold_render.py`) pins it by
calling the helper against three synthetic directories and asserting which
pytest exception each one raises.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

#: Set this to point the helper at another vault root. Named for the same
#: reason `scripts.automod.state.STATE_DIR` honours `LLOYD_AUTOMOD_STATE`: a
#: test that wants the no-vault branch must be able to say so rather than reach
#: into this module's globals. Read per call, not frozen at import — an override
#: that a caller's `monkeypatch.setenv` could not move would let a test set the
#: variable, be ignored, and read its own result as verifying the no-vault
#: branch while it was in fact checking the live board.
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
    """The board's item files, or a pytest stop that tells the truth.

    `what` names the measurement in the failure text ("cold cycle", "loader
    equivalence"), because the reader needs to know which claim was lost, not
    just that one was.

    `numeric_names` matches the board's own convention (`1204-slug.md`): the
    dashboard's scan counts those, and a test timing that scan must count the
    same way the scan does.
    """
    board = BOARD_DIR if board is None else board
    files = _item_files(board, numeric_names)
    if files:
        return files

    vault = vault_root()
    if not vault.is_dir():
        pytest.skip(
            f"{vault} does not exist on this machine, so the {what} has "
            f"nothing real to measure. Skipped rather than failed: an absent "
            f"vault is a property of the box, not of the change under test. "
            f"The clause {what} pins is therefore NOT pinned by this run."
        )

    listed = len(list(board.iterdir())) if board.is_dir() else 0
    pytest.fail(
        f"{board} is the board this item measures and {vault} exists, so "
        f"this tree is expected to have items on it — it matched "
        f"{len(_item_files(board, numeric_names))} of {listed} entries under "
        f"'*.md'{' starting with a digit' if numeric_names else ''}. Failing "
        f"instead of skipping: a board that is missing or emptied is exactly "
        f"the state the {what} exists to notice, and `skipped` is how it would "
        f"read as nobody's problem."
    )


def timed_ledger_or_stop(monkeypatch, *, what: str) -> Path:
    """Make the cycle decode a ledger that has bytes in it, and return it.

    The gate runs a round with `LLOYD_AUTOMOD_STATE` pointed at an empty state
    dir (`tests/conftest.py`), so `S.LEDGER_PATH` resolves to a file that does
    not exist and the run times a cycle that decodes *no* ledger at all. That
    is not the request the user pays, and a 6 s budget with nothing to decode
    does not constrain the ledger re-decode this item is about — the pre-fix
    cost was 55 `read_text` calls and 318,670 `json.loads` over
    6,286,192 bytes (triage, 2026-09-16).

    So when the state dir has no ledger but the real one exists, the timed
    cycle is pointed at the real file — read-only, and the dashboard only reads
    it. Returns the path the cycle will actually read, for the caller to print
    its byte count beside the timing.
    """
    from scripts.automod import state as S

    live = Path.home() / ".local" / "state" / "lloyd-automod" / "promotions.jsonl"
    current = S.LEDGER_PATH
    if current.is_file() and current.stat().st_size > 0:
        return current
    if live.is_file() and live.stat().st_size > 0:
        monkeypatch.setattr(S, "LEDGER_PATH", live)
        return live
    pytest.skip(
        f"no non-empty ledger at {current} or {live}, so the {what} cannot "
        f"constrain the ledger decode path — the clause about reading the "
        f"ledger once per cycle is NOT pinned by this run"
    )
