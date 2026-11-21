"""The suite does not file a real backlog item into Alan's vault (#4488).

`scripts/automod/backlog.py` binds `BACKLOG_DIR = vault_root() / "backlog"` at
import, and `vault_root()` is `Path.home() / "obsidian"` unless overridden. Inside
an automod round `$HOME` is the round home, whose `obsidian` is a symlink into the
house vault (`scripts/automod/worktree.py::HOME_LINK_SKIP` skips `lloyd` and
`lloyd-data`, not `obsidian`), so a test that reaches the module's writer with no
`backlog_dir=` creates a real item on the human's board. Item #4487 was exactly
such a file: its title, "A row written without a schedule", is a fixture string
from `tests/test_owed_check.py`, and it has no ledger row for the loop that filed
it. `conftest._isolate_automod_backlog` (the rebind) and
`backlog.refuse_live_board_write` (the refusal) are the fix; the four shapes below
are the check that it holds — the module's own board root, a child pytest run that
reaches the writer with no board dir, a test that names its own board dir, a caller
outside pytest that means the live board, and a pytest run that points the board
back at the house. The pattern is `tests/test_backlog_dedupe_isolation.py`, which
exists for the same failure class on the dedupe log.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.data_root import (PRODUCTION_VAULT, VAULT_ROOT_ENV, VAULT_ROOT_FALLBACK_ENV,
                           is_production_vault)
from scripts.automod import backlog as B

REPO = Path(__file__).resolve().parents[1]
#: The board a leaked write lands on. `PRODUCTION_VAULT` is `ACCOUNT_HOME /
#: "obsidian"` — the account's real home, not `$HOME`, so a round's linked vault
#: and this name the same directory, which is the point.
LIVE_BOARD = PRODUCTION_VAULT / "backlog"
#: The title that leaked (#4487). The child run files an item with it and the
#: parent proves the target vault never saw it.
FIXTURE_TITLE = "A row written without a schedule"
#: An item id no board holds, so `live_blockers` can only answer for it by
#: walking the directory it resolved for itself.
ORPHAN_TARGET = 4488000

_ENV_STRIP = (
    "PYTEST_CURRENT_TEST",       # must come from the child run itself
    VAULT_ROOT_ENV,              # the child's vault comes from its HOME, not an override
    VAULT_ROOT_FALLBACK_ENV,
    "PYTEST_ADDOPTS",
)


def _files(d: Path) -> set[str]:
    """The `.md` names on a board — a listing, never a read of their contents."""
    return {p.name for p in d.glob("*.md")} if d.is_dir() else set()


def _child_env(home: Path | None = None, vault: Path | None = None, **extra) -> dict:
    """Environment for a child process: no inherited test marker, no vault override.

    `PYTEST_CURRENT_TEST` is stripped for the same reason `scripts/automod/vault_guards.py`
    strips it from gate children — it names the *outer* node, so inheriting it would
    make a non-pytest child look like a test and let clause 4 pass for the wrong reason.
    """
    env = {k: v for k, v in os.environ.items() if k not in _ENV_STRIP}
    env["PYTHONPATH"] = str(REPO / "tests")
    if home is not None:
        env["HOME"] = str(home)
    if vault is not None:
        env[VAULT_ROOT_ENV] = str(vault)
    env.update({k: str(v) for k, v in extra.items()})
    return env


def _run_child_pytest(test_path: Path, home: Path, vault: Path) -> subprocess.CompletedProcess:
    """Run one child pytest node against `test_path` with `HOME` at a fake home.

    `-p conftest` is the load-bearing flag: the child file lives in `tmp_path`, so
    pytest would not collect `tests/conftest.py` as a conftest of it, and the
    suite-wide autouse rebind under test would never apply. With `tests/` on
    `PYTHONPATH` the flag loads that exact module as a plugin, which is what
    registers its autouse fixtures. A load failure is not a silent pass: the child's
    own assertion about the board it wrote to goes red with it.
    """
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(test_path), "-p", "conftest",
         "-p", "no:cacheprovider", "-q", "-x"],
        capture_output=True, text=True, cwd=str(REPO), timeout=600,
        env=_child_env(home=home,
                       CHILD_VAULT=vault,
                       CHILD_WOULD_BE=home / "obsidian" / "backlog"))


#: One test, reached by the node below through a child pytest run. It asks for no
#: `backlog_dir=` and no fixture of its own — that is the leak shape.
CHILD_SOURCE = '''
"""One test reached the automod board writer with no board dir of its own."""
import os
from pathlib import Path

from scripts.automod import backlog as B

VAULT = Path(os.environ["CHILD_VAULT"]).resolve()
WOULD_BE = Path(os.environ["CHILD_WOULD_BE"]).resolve()


def test_the_child_files_the_fixture_item_with_no_board_dir():
    # The leak path is real at this HOME: with no override the module default IS
    # the target vault's board, because `vault_root()` is `Path.home()/obsidian`
    # and `obsidian` here is a symlink to the vault below.
    assert WOULD_BE == (VAULT / "backlog").resolve()
    assert os.environ.get("PYTEST_CURRENT_TEST"), "the writer's guard reads this flag"
    item = B.new_item("A row written without a schedule")
    assert item.path.exists(), item.path
    root = item.path.resolve().parent
    assert root != (VAULT / "backlog").resolve(), (
        f"the writer landed on the vault board it was supposed to miss: {root}")
'''


def test_the_board_root_resolves_into_scratch_inside_a_test():
    """Clause 1: inside any test the module's board root is scratch, not the house.

    The scratch root is what the three `root = backlog_dir or BACKLOG_DIR` readers
    resolve — `all_items`, `live_blockers` and `new_item` — so each is exercised
    through the default (`backlog_dir=None`) and must answer for the scratch
    directory: the item `new_item` created is there, `all_items` sees it, and
    `live_blockers` reports the blocker's orphan target, an id the live board does
    not hold. The `inspect` loop is what makes the rebind the mechanism: a reader
    that resolved the vault at call time instead of reading the module global would
    sail through every assertion above while still writing the house.
    """
    root = B.BACKLOG_DIR
    assert root != LIVE_BOARD, root
    assert not is_production_vault(root.parent), (
        f"{root} still resolves through the live vault (its parent is the vault)")

    item = B.new_item("A row the suite files")
    assert item.path.parent == root, item.path
    assert [i.id for i in B.all_items()] == [item.id]

    blocker = B.new_item("A blocker naming an id no board holds",
                         body=f"Blocks #{ORPHAN_TARGET} — PERSON-ONLY: name the target.",
                         tags=("blocker",))
    assert B.live_blockers(root / "no-such-ledger.jsonl") == {blocker.id: ORPHAN_TARGET}

    for fn in (B.all_items, B.live_blockers, B.new_item):
        assert "root = backlog_dir or BACKLOG_DIR" in inspect.getsource(fn), (
            f"{fn.__name__} no longer resolves its board through the module global, "
            "so rebinding `BACKLOG_DIR` no longer covers it")


def test_a_child_pytest_run_with_no_board_dir_leaves_the_target_vault_empty(tmp_path):
    """Clause 2: a child pytest run cannot create a file in the vault its HOME points at.

    The home is a `tmp_path` directory whose `obsidian` is a symlink to a temp
    vault — the round-home shape verbatim, so `vault_root()` resolves straight
    through it to `tmp_path/vault/backlog`. Before the rebind that is where
    `new_item` wrote, and #4487 is the file it wrote.
    """
    vault = tmp_path / "vault"
    (vault / "backlog").mkdir(parents=True)
    home = tmp_path / "home"
    home.mkdir()
    (home / "obsidian").symlink_to(vault)
    child = tmp_path / "test_child_of_the_suite.py"
    child.write_text(CHILD_SOURCE, encoding="utf-8")

    proc = _run_child_pytest(child, home, vault)

    assert proc.returncode == 0, (
        f"the child node exited {proc.returncode} — a child that cannot import the "
        f"suite's conftest is not the case under test:\n{proc.stdout[-2000:]}{proc.stderr[-2000:]}")
    assert _files(vault / "backlog") == set(), (
        f"a pytest run filed {sorted(_files(vault / 'backlog'))} into the vault its "
        "HOME resolves to — the leak #4487 is made of")


def test_a_test_that_names_its_own_board_writes_exactly_there(tmp_path, monkeypatch):
    """Clause 3: a test-local board dir still wins over the suite-wide rebind.

    Both routes a test has: an explicit `backlog_dir=` argument, and its own
    `monkeypatch.setattr` of the global. Each must land on the named directory, and
    the suite's scratch root must stay empty — that is what lets the 25 importer
    files that pass their own `backlog_dir=` keep working unchanged.
    """
    scratch = B.BACKLOG_DIR
    named = tmp_path / "board"
    named.mkdir()

    explicit = B.new_item("An item with an explicit board dir", backlog_dir=named)
    assert explicit.path.parent == named, explicit.path
    assert _files(scratch) == set(), (
        f"an explicit `backlog_dir=` also wrote to the suite root: {sorted(_files(scratch))}")

    monkeypatch.setattr(B, "BACKLOG_DIR", named)
    patched = B.new_item("An item on the test's own patched board")
    assert patched.path.parent == named, patched.path
    assert _files(named) == {explicit.path.name, patched.path.name}


def test_a_caller_outside_pytest_still_writes_the_modules_board_dir(tmp_path):
    """Clause 4: with no `PYTEST_CURRENT_TEST`, the writer writes the module's board.

    Run out of process because the flag is a process fact, and aimed at a temp vault
    through `LLOYD_VAULT_ROOT` so the write is real but nobody's board is. The
    second half of the script is the clause that matters for production: the guard,
    handed the live board with the flag absent, must return — that is
    `scripts/automod/backlog.py`'s red-tree filing, `scripts/automod/owed.py`'s
    follow-ups, `scripts/maintenance/code_rules_census.py` and
    `scripts/step_conformance.py` filing into the house exactly as they do today.
    A guard that refused there is a loop that stops filing its own reports.
    """
    vault = tmp_path / "vault"
    vault.mkdir()
    script = (
        "from app.data_root import PRODUCTION_VAULT\n"
        "from scripts.automod import backlog as B\n"
        "B.refuse_live_board_write(PRODUCTION_VAULT / 'backlog', 'new_item')\n"
        "print(B.new_item('A row filed outside pytest').path)\n"
    )
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, cwd=str(REPO), timeout=300,
                          env=_child_env(vault=vault))
    assert proc.returncode == 0, (
        f"a non-pytest caller was refused or failed (rc={proc.returncode}): "
        f"{proc.stdout[-800:]}{proc.stderr[-800:]}")
    written = sorted(p.name for p in (vault / "backlog").glob("*.md"))
    assert len(written) == 1 and written[0].startswith("1-a-row-filed-outside-pytest"), written
    assert Path(proc.stdout.strip()).is_file(), proc.stdout


def test_a_pytest_run_pointed_at_the_live_board_is_refused_before_writing(monkeypatch):
    """Clause 5: pytest plus the live board as the write target is a refusal, not a file.

    The rebind is a default, so a test can undo it — `test_backlog_dedupe_isolation.py`
    patches `BACKLOG_DIR` itself, and a `live_vault` test that reads the house board
    has to. When such a test then reaches the writer, the write must be refused
    *before* anything is created, naming the directory it refused. The listing is the
    proof of "before": a file born during the call would change it. Both routes in —
    the module global patched back, and an explicit `backlog_dir=` — because the
    writer takes its root from either.
    """
    assert LIVE_BOARD.is_dir(), (
        f"{LIVE_BOARD} is not a directory, so the listing below has no denominator")
    before = _files(LIVE_BOARD)

    with pytest.raises(B.LiveBoardWriteRefused) as refused:
        B.new_item("A row a test tried to file in the house", backlog_dir=LIVE_BOARD)
    assert str(LIVE_BOARD) in str(refused.value), (
        f"the refusal named {refused.value!r}, not the refused directory")

    monkeypatch.setattr(B, "BACKLOG_DIR", LIVE_BOARD)
    with pytest.raises(B.LiveBoardWriteRefused):
        B.new_item("A row through the patched global")

    assert _files(LIVE_BOARD) == before, (
        f"the refused writes still left {sorted(_files(LIVE_BOARD) - before)} on the board")
