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
import shutil
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


# ── which instrument could act on the row (#2172) ──────────────────────
#
# Two instruments read this journal over two different path sets: the bracket journals
# every path that entered the checkout's ignored set, nested ones included, while the
# guardian's `datawatch.stray_in_tree` judges the top level minus
# `KNOWN_GOOD_TOPLEVEL` plus the retained `RUNTIME_NAMES`. Nothing on a row said which
# could act, so on 2026-10-03 nightly reflection read three rows for a 0-byte
# `workers.db` that lived 15 seconds and one row for `web/tsconfig.node.tsbuildinfo`,
# filed #2169 at priority high as a silent-clear alerting gap, and the shape it named
# had been closed by `6bf40361` about 23 hours earlier.

_GUARDIAN_SRC = REPO / "agent-services" / "guardian"


def _stage_guardian(repo: Path) -> Path:
    """Put the guardian's modules inside the checkout, the way the stager does.

    `agent-services/bin/guardian-stage.sh:42` is `cp "$SRC"/*.py "$STAGE"/`, so every
    module in that directory reaches the pinned snapshot without a per-file list — which
    is why the reach rule could be added inside `datawatch.py` with no change to the
    stager. The bracket reaches the same directory for the opposite reason: the guardian
    runs from that snapshot under `/usr/bin/python3`, where `app` is not importable, so
    the predicate lives guardian-side and the bracket imports it, never the reverse.
    """
    dest = repo / "agent-services" / "guardian"
    dest.mkdir(parents=True, exist_ok=True)
    for module in sorted(_GUARDIAN_SRC.glob("*.py")):
        shutil.copy2(module, dest / module.name)
    return repo


def test_every_row_names_the_instrument_that_can_act_on_it(tree):
    """Clause 1 (#2172): `actionable_by` is on BOTH row kinds, one of exactly two values.

    On both kinds because the reader's question — "which of these rows is an alert nobody
    actioned?" — is asked of a `removed` row too. Three of the four live rows on 2026-10-03
    were one 0-byte `workers.db` appearing, appearing again, and being `removed` fifteen
    seconds later, against `policy.STRAY_CHECK_SECONDS = 3600.0`; the row that says the
    stray was deleted is the row that most needs to say whether an alert was ever due.

    The label is computed by loading THIS tree's `datawatch.py` — the tree is a real
    checkout and the guardian is staged into it — so what the node pins is the cross-boundary
    read, not a string the fixture agreed to.
    """
    repo, journal = tree
    _stage_guardian(repo)
    _bash_tree_strays._record(WORKER, "cd ~/lloyd && : > workers.db", repo, ["workers.db"])
    _bash_tree_strays._record(WORKER, "rm -- workers.db", repo, ["workers.db"],
                              kind="removed")

    rows = _rows(journal)
    assert [r["kind"] for r in rows] == ["appeared", "removed"], rows
    assert [r["actionable_by"] for r in rows] == ["guardian-strays", "guardian-strays"], rows
    assert sorted(rows[0]) == ["actionable_by", "at", "command", "kind", "paths", "root",
                               "session"], rows[0]
    assert all(r["actionable_by"] in (_bash_tree_strays.GUARDIAN_STRAYS,
                                      _bash_tree_strays.BRACKET_ONLY) for r in rows), (
        "a third value would be a claim about an instrument that does not exist")


def test_a_nested_ignored_path_journals_as_bracket_only(tree):
    """Clause 2 (#2172): the label is `stray_in_tree`'s reach rule, path by path.

    `web/tsconfig.node.tsbuildinfo` is the live row — 105,181 bytes on disk right now,
    ignored by `.gitignore:20` (`*.tsbuildinfo`) — and `web/` is tracked here exactly as
    it is there, so the check names neither the file nor its directory and the row reads
    `bracket-only`. The rest are the same rule's other edges: an unlisted top-level name
    and a path under a `RUNTIME_NAMES` prefix are the two shapes the alert can reach, a
    `KNOWN_GOOD_TOPLEVEL` name and a git-tracked path are the two it cannot. The two-path
    row pins that ONE reachable path makes a row reachable, which is what "at least one"
    in the acceptance means.
    """
    repo, journal = tree
    _stage_guardian(repo)
    (repo / "web").mkdir()
    (repo / "web" / "index.html").write_text("<html>\n", encoding="utf-8")
    _git(repo, "add", "web")                      # tracked, as on the live tree

    def label(*paths: str) -> str:
        _bash_tree_strays._record(WORKER, "npm run typecheck", repo, list(paths))
        return _rows(journal)[-1]["actionable_by"]

    assert label("web/tsconfig.node.tsbuildinfo") == "bracket-only", (
        "the live row: a nested path under a tracked directory is beyond the alert")
    assert label(".vscode/settings.json") == "bracket-only", "KNOWN_GOOD_TOPLEVEL"
    assert label("tracked.py") == "bracket-only", "git tracks it: committed, not stray"
    assert label("probe.db") == "guardian-strays", "an unlisted top-level name"
    assert label("logs/pipeline.log") == "guardian-strays", "a RUNTIME_NAMES nested path"
    assert label("web/tsconfig.node.tsbuildinfo", "workers.db") == "guardian-strays", (
        "one reachable path in the row is enough")


def test_a_tree_with_no_staged_guardian_still_gets_its_row_without_the_label(tree):
    """Clause 4 (#2172), the natural failure: no reach rule, and the row still lands.

    A checkout with no staged guardian is not hypothetical — that is the shape of a tree
    whose snapshot has not been built, and it is every other node in this file: none of
    them stages the guardian, and all of them still get their row, which is why
    `actionable_by` had to be computed in its own nested `try` rather than in the
    existing one. A label that can cost the row would make the journal less complete
    than it was before the label existed, and the row is the record.
    """
    repo, journal = tree
    assert not (repo / "agent-services").exists(), (
        "no guardian staged here, on purpose: this is the missing-rule case")
    _bash_tree_strays._record(WORKER, "cd ~/lloyd && : > workers.db", repo, ["workers.db"])

    rows = _rows(journal)
    assert len(rows) == 1, rows
    assert "actionable_by" not in rows[0], rows[0]
    assert rows[0]["paths"] == ["workers.db"] and rows[0]["kind"] == "appeared"


async def test_a_reach_rule_that_raises_costs_the_label_not_the_note(tree, monkeypatch):
    """Clause 4 (#2172), the raised-call half: nothing escapes the Bash call.

    The import failing is one shape; the predicate raising is the other, and the rule is
    read by exec'ing a file from a tree this process does not control, so either can
    happen while the command itself succeeded. What must not happen is the model losing
    the note — the note is the only thing that ever got a stray deleted in the same turn —
    or the call failing, which would make an instrument about hygiene break the work it is
    measuring. `calls` asserts the label was ATTEMPTED, so this node cannot pass by the
    labelling step being skipped.
    """
    repo, journal = tree
    _as(monkeypatch, WORKER)
    calls: list[tuple[str, tuple[str, ...]]] = []

    def boom(root, paths):
        calls.append((str(root), tuple(paths)))
        raise ImportError("no reach rule for this tree")

    monkeypatch.setattr(_bash_tree_strays, "_actionable_by", boom)
    text = await _run(f"cd {repo} && : > workers.db && echo done")

    assert text.startswith("done"), "the command's own output stands"
    assert MARK in text, "the note stands: a missing label never costs the model the hint"
    assert calls == [(str(repo), ("workers.db",))], "the label was attempted, not skipped"
    rows = _rows(journal)
    assert len(rows) == 1 and "actionable_by" not in rows[0], rows
