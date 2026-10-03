"""A background session's Bash call that leaves an ignored path in the live checkout
is told so on that call (`agent_mcp/_bash_tree_strays.py`).

The incident: 2026-10-02 01:01:36, session `20261002_010103_autonomy_1d90` ran
`cd ~/lloyd && sqlite3 workers.db "select …"`, `sqlite3` created the file, the session
noticed and moved on, and the writer was identified by hand seven hours later.

Every node runs a real shell through `builtin_bash.call_tool` against a real git tree
and reads the result text and the journal file — the two things a model and a person
actually see. The checkout and the journal are redirected to `tmp_path`; nothing here
reads or writes the tree the suite runs from.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent_mcp import _bash_tree_strays, builtin_bash  # noqa: E402
from app import live_strays  # noqa: E402

WORKER = "20261002_010103_autonomy_1d90"   # four parts: an unattended session
CHAT = "20261002_010103_ab12"              # three parts: a person
MARK = "[stray in the code tree]"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=str(repo), check=True,
                   capture_output=True, text=True)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A checkout with the live tree's own rule (`*.db`) and a journal beside it."""
    repo = tmp_path / "live"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / ".gitignore").write_text("*.db\n__pycache__/\n", encoding="utf-8")
    (repo / "tracked.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", ".gitignore", "tracked.py")
    _git(repo, "commit", "-q", "-m", "base")
    journal = tmp_path / "safety" / "tree-strays.jsonl"
    monkeypatch.setattr(_bash_tree_strays, "_live_root", lambda: repo)
    monkeypatch.setattr(_bash_tree_strays, "_journal_path", lambda: journal)
    return repo, journal


def _as(monkeypatch, session_id: str) -> None:
    monkeypatch.setattr(builtin_bash, "get_bound_session", lambda: session_id)


async def _run(command: str, **extra) -> str:
    result = await builtin_bash.call_tool("Bash", {"command": command, **extra})
    return "".join(getattr(part, "text", "") for part in result.content)


def _rows(journal: Path) -> list[dict]:
    if not journal.exists():
        return []
    return [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]


async def test_the_call_that_creates_an_ignored_file_is_told_and_journaled(tree, monkeypatch):
    repo, journal = tree
    _as(monkeypatch, WORKER)
    command = f"cd {repo} && : > workers.db && echo done"
    text = await _run(command)

    assert text.startswith("done"), "the command's own output is first and intact"
    assert MARK in text and "workers.db" in text
    assert str(repo) in text, "the note names the tree it measured"
    rows = _rows(journal)
    assert len(rows) == 1
    assert rows[0]["session"] == WORKER
    assert rows[0]["paths"] == ["workers.db"]
    assert rows[0]["command"] == command


async def test_a_failing_command_is_told_too(tree, monkeypatch):
    """The incident's own shape: the query failed, the file was created anyway."""
    repo, journal = tree
    _as(monkeypatch, WORKER)
    text = await _run(f"cd {repo} && : > workers.db && exit 3")
    assert "[exit code: 3]" in text and MARK in text
    assert len(_rows(journal)) == 1


async def test_a_chat_session_is_not_measured(tree, monkeypatch):
    repo, journal = tree
    _as(monkeypatch, CHAT)
    text = await _run(f"cd {repo} && : > workers.db && echo done")
    assert MARK not in text
    assert _rows(journal) == []


async def test_a_new_source_file_is_not_a_stray(tree, monkeypatch):
    """`??` is a file a job may be about to commit; only an ignored path can never be."""
    repo, journal = tree
    _as(monkeypatch, WORKER)
    text = await _run(f"cd {repo} && echo 'y = 2' > new_module.py && echo done")
    assert MARK not in text
    assert _rows(journal) == []


async def test_a_cache_directory_is_not_a_stray(tree, monkeypatch):
    repo, journal = tree
    _as(monkeypatch, WORKER)
    text = await _run(f"cd {repo} && mkdir -p pkg/__pycache__ && : > pkg/__pycache__/m.pyc "
                      "&& echo done")
    assert MARK not in text
    assert _rows(journal) == []


async def test_a_file_that_was_already_there_is_not_this_calls(tree, monkeypatch):
    repo, journal = tree
    (repo / "usage.db").write_bytes(b"")
    _as(monkeypatch, WORKER)
    text = await _run(f"cd {repo} && ls >/dev/null && echo done")
    assert MARK not in text
    assert _rows(journal) == []


async def test_an_unreadable_tree_costs_the_note_and_nothing_else(tree, tmp_path, monkeypatch):
    repo, journal = tree
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    monkeypatch.setattr(_bash_tree_strays, "_live_root", lambda: not_a_repo)
    _as(monkeypatch, WORKER)
    assert live_strays.ignored(not_a_repo) is None, "the premise: git refuses this root"
    text = await _run(f"cd {not_a_repo} && : > workers.db && echo done")
    assert text.strip() == "done"
    assert _rows(journal) == []


async def test_a_background_command_is_not_measured(tree, monkeypatch):
    """It returns before it has run, so there is no second read to take."""
    _as(monkeypatch, WORKER)
    assert await _bash_tree_strays.before(WORKER, {"command": "x",
                                                   "run_in_background": True}) is None


def test_ignored_reads_ignored_files_only(tree):
    repo, _ = tree
    (repo / "workers.db").write_bytes(b"")
    (repo / "new_module.py").write_text("y = 2\n", encoding="utf-8")
    (repo / "pkg" / "__pycache__").mkdir(parents=True)
    (repo / "pkg" / "__pycache__" / "m.pyc").write_bytes(b"")
    assert live_strays.ignored(repo) == {"workers.db"}


# ── the removal half of the same event class (#2110) ───────────────────

async def test_the_call_that_removes_a_seen_stray_journals_the_removal(tree, monkeypatch):
    """Clause 4: a deletion inside a measured call gets a row of its own.

    The guard diffed appearances only, so the one action this note *tells the session
    to take* — "remove it now" — was the one action that left no trace. `~/lloyd/workers.db`
    vanished with `safety/tree-strays.jsonl` never written, both quarantine directories
    absent, and no instrument able to say which. One JSONL row naming the removed path,
    the session and the command is the whole of what was missing.
    """
    repo, journal = tree
    (repo / "workers.db").write_bytes(b"")     # ignored, and seen before the call
    _as(monkeypatch, WORKER)
    command = f"cd {repo} && rm -- workers.db && echo gone"
    text = await _run(command)

    assert text.startswith("gone"), "the command's own output is first and intact"
    assert MARK not in text, "a removal is not a new stray, so nothing is asserted"
    assert not (repo / "workers.db").exists()
    rows = _rows(journal)
    assert len(rows) == 1, rows
    assert rows[0]["kind"] == "removed", rows[0]
    assert rows[0]["paths"] == ["workers.db"]
    assert rows[0]["session"] == WORKER
    assert rows[0]["command"] == command
    assert rows[0]["root"] == str(repo), "the row names the tree it was measured in"


async def test_a_removal_and_an_appearance_share_the_journal(tree, monkeypatch):
    """The two kinds sit alongside each other, one row each, one shape each.

    The appearance half is the shipped behaviour, so this is the regression pin as
    well: `rm old.db` and `: > new.db` in one call must not lose the appearance note,
    and both rows carry a `kind`, because a journal whose rows are half-typed is a
    journal a later reader has to guess at.
    """
    repo, journal = tree
    (repo / "old.db").write_bytes(b"")
    _as(monkeypatch, WORKER)
    text = await _run(f"cd {repo} && rm -- old.db && : > new.db && echo done")

    assert MARK in text, "the new stray is still noted on the result it appeared in"
    rows = _rows(journal)
    assert [r["kind"] for r in rows] == ["appeared", "removed"], rows
    assert rows[0]["paths"] == ["new.db"] and rows[1]["paths"] == ["old.db"]
    assert all("command" in r and r["session"] == WORKER for r in rows)


async def test_a_path_that_leaves_the_ignore_set_while_still_on_disk_writes_nothing(
        tree, monkeypatch):
    """The `.exists()` filter in `_removed` is load-bearing, and this is the case it exists for.

    `git add -f ignored.db` takes the path out of `live_strays.ignored()` — measured, not
    assumed: the before-snapshot is `['ignored.db']`, the after-snapshot is `[]` — while the
    file sits in the tree the whole time. So the set difference `seen - now` is
    `['ignored.db']`, which is exactly what a naive implementation would journal as a
    deletion of a file that is demonstrably still there. The row is gated on the path
    being absent from disk, which is why this node writes nothing and why the sibling node
    above, where the file really is gone, writes one.

    A permission-error `rm` is the other way to keep a file on disk, and it was the first
    fixture here; it proved unable to pin anything, because a wholly-ignored directory
    (`locked/`) is reported by git as the directory and never as `locked/x.db`, so that
    path was in neither set and the naive difference was empty with or without the filter.
    """
    repo, journal = tree
    (repo / "ignored.db").write_bytes(b"")     # the fixture's .gitignore names this one
    _as(monkeypatch, WORKER)
    before = sorted(_seen(repo))
    text = await _run(f"cd {repo} && git add -f ignored.db && echo tracked")

    assert "tracked" in text
    assert (repo / "ignored.db").exists(), (
        "the fixture needs the file on disk; `git add -f` does not remove it")
    assert before == ["ignored.db"] and sorted(_seen(repo)) == [], (
        f"before={before} after={sorted(_seen(repo))}: this node only discriminates if the "
        "path leaves the ignore set while remaining on disk")
    assert _rows(journal) == [], (
        "a path that left the ignore set but is still on disk was journalled as removed — "
        "that row would be a false deletion record, worse than no record")


def _seen(repo: Path) -> set[str]:
    from app import live_strays
    return live_strays.ignored(repo)
