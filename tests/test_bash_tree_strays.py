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

import collections
import hashlib
import json
import re
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


# ── the origin row for a stray this call did not create (#2220) ──────────
#
# The layer is wired into Bash alone — `git grep -ln "tree_strays\|live_strays" --
# agent_mcp` returns `_bash_tree_strays.py` and `builtin_bash.py`, and the Write/Edit
# handler `agent_mcp/builtin_fs.py` holds zero occurrences of `stray` — so a file laid by
# `Write` enters the tree unseen, and the FIRST bracket that measures it already holds it
# in its before-snapshot. `appeared()` is a set difference against that snapshot, so the
# origin row for such a path is not missing by accident: it cannot fire. The committed
# witness below is the live journal and shows exactly that: 5 rows, kinds
# `{appeared: 3, removed: 2}`, and the row for `knowledge/` — deleted by session
# `20261004_023233_autocode_8e79` at 2026-10-04T10:42:21+00:00 after a `Write` created it
# at message 361 of that session — has no `appeared` row beside it and never can.
#
# The fix is a `present` row, and its whole risk is the flood it would cause if it keyed
# on the after-set alone: `live_strays.ignored(session_cwd.live_root())` returns 14 paths
# on the real tree right now (`.env`, `qmd/`, `web/tsconfig.node.tsbuildinfo`, eleven
# `agent-services/**` entries), and a branch that journalled "in the after set, never seen
# before" would print fourteen rows on the first measured call of a fresh state, `.env`
# included, indistinguishable from fourteen new incidents. So the decision is made by a
# PERSISTED acknowledgement set seeded silently the first time it is read — a tree's
# standing ignored paths are acknowledged as standing, and only a path that reaches the
# after-set afterwards gets a row. The nodes below pin both halves: that a written stray
# gets exactly one row, and that the silence of the seed is what makes that row mean
# something rather than the branch being broken.

WITNESS = Path(__file__).resolve().parent / "fixtures" / "tree_strays_witness_2220.jsonl"
WITNESS_SHA256 = "28b179aa44380e53391306bef3b9fee0daedc1bccb235ca9e902eb11edb64756"
#: Where clause 6 asks for the same bytes durably. A node may not READ it — the gate runs
#: with HOME at the round home, where `~/obsidian` does not exist, and a node that opened
#: it would skip, which pins nothing — so the repo copy above is what is asserted and this
#: copy is compared whenever it happens to be there.
VAULT_WITNESS = Path.home() / "obsidian" / "backlog" / "data" / "tree-strays.jsonl"


def _ack() -> dict:
    """The persisted acknowledgement state, as written. `_ack_path()` is derived from
    `_journal_path()`, which the `tree` fixture redirects, so the state lands in
    `tmp_path` with the journal and nothing here touches the live safety directory."""
    import json as _json
    path = _bash_tree_strays._ack_path()
    if not path.exists():
        return {}
    return _json.loads(path.read_text(encoding="utf-8"))


def _written_by_another_writer(repo: Path, name: str) -> Path:
    """Lay an ignored file in the tree WITHOUT a Bash call, which is the case at issue.

    A direct `write_bytes` from this process is the same shape as `agent_mcp/builtin_fs.py`
    creating a file for a `Write` tool call: neither is a Bash call, so neither is
    bracketed, and the next bracket sees the file in its before-snapshot. It is not a
    stand-in for the incident — it is the incident's mechanism, minus the model.
    """
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


async def _seed(tree) -> None:
    """One measured call, which is what creates the tree's acknowledgement state."""
    repo, _ = tree
    await _run(f"cd {repo} && echo seeded")


async def test_a_stray_this_call_did_not_create_is_journalled_once_as_present(tree,
                                                                             monkeypatch):
    """Clause 1: the origin row a `Write`-created stray can never get, on the next call.

    The path is laid by `_written_by_another_writer`, not by the command, and the node
    proves before running anything that the before-snapshot already holds it —
    `live_strays.ignored()` returns `['usage.db']` — which is the whole reason
    `appeared(now - seen)` cannot name it. The row must carry the path, the session, the
    root and the command of the call that FOUND it, and the call's own output must stay
    clean: this call did not create the file, so a `[stray in the code tree]` note would
    accuse it, and an accusing note is how an instrument gets its output ignored.
    """
    repo, journal = tree
    _as(monkeypatch, WORKER)
    await _seed(tree)
    _written_by_another_writer(repo, "usage.db")

    assert sorted(_bash_tree_strays._snapshot_sync()[1]) == ["usage.db"], (
        "the premise the item rests on: the stray is in the tree BEFORE this call, so the "
        "before-snapshot holds it and `appeared()` has nothing to report")

    command = f"cd {repo} && ls >/dev/null && echo done"
    text = await _run(command)

    assert text.startswith("done"), "the command's own output is first and intact"
    assert MARK not in text, (
        "this call did not create the path, so nothing may be asserted to it on the result")
    rows = _rows(journal)
    assert len(rows) == 1, f"expected exactly one origin row, got {rows}"
    assert rows[0]["kind"] == "present", rows[0]
    assert rows[0]["paths"] == ["usage.db"], rows[0]
    assert rows[0]["session"] == WORKER, rows[0]
    assert rows[0]["root"] == str(repo), "the row names the tree it was measured in"
    assert rows[0]["command"] == command, "the finding call is the attribution the row has"


async def test_a_present_stray_is_acknowledged_and_not_journalled_again(tree, monkeypatch):
    """Clause 2: one row per tree-path, because the row is an origin, not an alarm.

    `appeared` is a per-call fact and re-fires legitimately when a session recreates a
    stray; a `present` path is the same file sitting there, and a journal that grew a row
    for it on every background call would bury the one row that does name a writer — the
    live journal is 5 rows in total, and its readability is what makes it usable. So the
    path goes into the acknowledgement state as it is journalled, and three further calls
    add nothing. The state file is read here rather than inferred from the row count, so
    the node fails if the silence is the branch never firing rather than the ack working.
    """
    repo, journal = tree
    _as(monkeypatch, WORKER)
    await _seed(tree)
    _written_by_another_writer(repo, "usage.db")

    await _run(f"cd {repo} && echo first")
    for _ in range(3):
        await _run(f"cd {repo} && ls >/dev/null && echo again")

    rows = _rows(journal)
    assert [r["kind"] for r in rows] == ["present"], rows
    assert _ack().get(str(repo), {}).get("paths") == ["usage.db"], (
        "the row was written but the path was never acknowledged, so the next call would "
        "write it again")


async def test_the_silence_comes_from_the_state_file_and_not_from_memory(tree, monkeypatch):
    """The process boundary this change crosses is a file, and this node reads it as a file.

    Every other node in this file runs several calls in ONE process, where a dict held at
    module level would satisfy them all — and the real caller is a different MCP process
    per session, so a module-level cache would work in the suite and be wrong on the box:
    the second background session would see a cache it did not write and stay silent about
    a stray nobody had ever named. So the decision must be a function of the bytes at
    `_ack_path()`, in both directions:

    - erase the journalled path from the state file, and it is a stranger again: the next
      call journals it a second time;
    - add a path to that file that this module never journalled, create the file on disk,
      and the next call stays silent about it.

    Both halves go through the real bracket, not a call into `_after_sync`, because the
    seam is what a later refactor of the bracket has to preserve. The first half erases
    the PATH from an entry that still exists rather than deleting the file: a missing file
    or a missing root entry is clause 3's standing case and seeds silently, journaling
    nothing, so removing the file would assert the opposite of the specified behaviour.
    """
    repo, journal = tree
    _as(monkeypatch, WORKER)
    await _seed(tree)
    _written_by_another_writer(repo, "usage.db")

    await _run(f"cd {repo} && echo first")
    assert [r["kind"] for r in _rows(journal)] == ["present"]

    _bash_tree_strays._write_state({str(repo): {"paths": []}})
    await _run(f"cd {repo} && echo ack-erased")
    assert [r["kind"] for r in _rows(journal)] == ["present", "present"], (
        "the second call stayed silent although the state file on disk no longer holds "
        "this path, so the decision came from something this process remembered")

    # Now a file this module never acknowledged, admitted by a write this process did not
    # make — the shape of a state restored from backup, or of a second tree sharing the
    # file. `usage.db` stays in the set: replacing it wholesale would erase its own
    # acknowledgement, which is the half above, not this one.
    state = _ack()
    state[str(repo)] = {"paths": sorted(set(state[str(repo)]["paths"]) | {"later.db"})}
    _bash_tree_strays._write_state(state)
    _written_by_another_writer(repo, "later.db")
    await _run(f"cd {repo} && echo foreign-ack")
    assert [r["paths"] for r in _rows(journal)] == [["usage.db"], ["usage.db"]], (
        "an acknowledgement this process never wrote did not hold, so the file on disk is "
        "not the only input to the decision and a restored or hand-edited state is ignored")
    assert "later.db" not in journal.read_text(encoding="utf-8"), (
        "a path the state named but no row ever did still got its row, so the ack set is "
        "consulted only for paths this process already journalled")


async def test_the_first_measured_call_seeds_the_acknowledgement_silently(tree, monkeypatch):
    """Clause 3: with no state yet, the standing case stays silent — and the branch works.

    `test_a_file_that_was_already_there_is_not_this_calls` already pins the standing case
    (a file that predates the call: no note, no row). This node pins the same expectation
    against a file that predates the ACKNOWLEDGEMENT STATE, which is the flood rule: on
    the real tree `ignored()` answers 14 paths, and the first measured call after this
    ships must not journal `.env` and thirteen siblings as incidents.

    Silence alone would also be what a broken branch looks like, so the node carries its
    own discrimination: the state file must exist and name the standing path (the seed
    ran), and a path laid AFTER that call must get its row on the call after it. Same
    code, one call apart, one silent and one journalled — which is the only evidence that
    the first silence was a decision.
    """
    repo, journal = tree
    _written_by_another_writer(repo, "usage.db")      # standing, before any bracket ran
    _as(monkeypatch, WORKER)

    text = await _run(f"cd {repo} && ls >/dev/null && echo done")

    assert MARK not in text
    assert _rows(journal) == [], (
        "a path that was already ignored before any bracket ran is part of the tree, not "
        "an incident: 14 such paths exist on the live checkout, `.env` among them")
    assert _ack().get(str(repo), {}).get("paths") == ["usage.db"], (
        "the state was never seeded, so the silence below would be the branch not firing")

    _written_by_another_writer(repo, "second.db")
    await _run(f"cd {repo} && ls >/dev/null && echo done")

    rows = _rows(journal)
    assert [r["kind"] for r in rows] == ["present"], rows
    assert rows[0]["paths"] == ["second.db"], rows[0]


async def test_a_present_row_names_the_instrument_that_can_act_on_it(tree, monkeypatch):
    """Clause 4: the label is the same rule, on the third kind too, computed not guessed.

    `present` is the kind a reader is most likely to act on — it is a stray nobody was
    told about — so a label that drifted on this kind alone would reintroduce #2169, which
    was filed at priority high when nightly reflection read an out-of-reach row as a
    silent-clear alerting gap. Both values are pinned through the real path: the file is
    really ignored (the fixture's `*.db` rule reaches it), really in the after-set, and
    really labelled by the staged guardian's `reachable_by_stray_check`. `web/x.db` is the
    `web/tsconfig.node.tsbuildinfo` shape — nested under a tracked directory, so the alert
    names neither the file nor its directory — and a top-level name it does not explain is
    the shape the alert can reach.
    """
    repo, journal = tree
    (repo / "web").mkdir()
    (repo / "web" / "index.html").write_text("<html>\n", encoding="utf-8")
    _git(repo, "add", "web")                          # tracked, as on the live tree
    _stage_guardian(repo)
    _as(monkeypatch, WORKER)
    await _seed(tree)
    _written_by_another_writer(repo, "web/x.db")      # the row's own path form

    await _run(f"cd {repo} && echo first")
    _written_by_another_writer(repo, "probe.db")
    await _run(f"cd {repo} && echo second")

    rows = _rows(journal)
    assert [r["kind"] for r in rows] == ["present", "present"], rows
    assert rows[0]["paths"] == ["web/x.db"] and rows[0]["actionable_by"] == "bracket-only", (
        "a nested path under a tracked directory is beyond the alert, which is exactly the "
        "live row that was misread as a silent clear (#2169)")
    assert rows[1]["paths"] == ["probe.db"] and rows[1]["actionable_by"] == "guardian-strays"


async def test_a_reach_rule_that_raises_costs_the_label_not_the_present_row(tree,
                                                                           monkeypatch):
    """Clause 4, the raised-call half for the new kind: the row is the record.

    Same rule as the two shipped kinds — `actionable_by` is computed in a nested `try`
    precisely so a tree whose reach rule this process cannot read keeps its row — but it
    has to be pinned per kind, because the easy way to write a new branch is to compute
    the label first and return early when it fails, which would make the newest kind the
    least durable row type in the journal.
    """
    repo, journal = tree
    _as(monkeypatch, WORKER)
    await _seed(tree)
    calls: list[tuple[str, tuple[str, ...]]] = []

    def boom(root, paths):
        calls.append((str(root), tuple(paths)))
        raise ImportError("no reach rule for this tree")

    monkeypatch.setattr(_bash_tree_strays, "_actionable_by", boom)
    _written_by_another_writer(repo, "usage.db")
    text = await _run(f"cd {repo} && echo done")

    assert text.startswith("done") and MARK not in text
    assert calls == [(str(repo), ("usage.db",))], "the label was attempted, not skipped"
    rows = _rows(journal)
    assert len(rows) == 1 and rows[0]["kind"] == "present", rows
    assert "actionable_by" not in rows[0], rows[0]


async def test_a_stray_created_by_this_call_is_journalled_as_appeared_and_never_as_present(
        tree, monkeypatch):
    """Clause 5: the shipped half is untouched, and a stray is not journalled twice.

    A file this call created IS in the after-set and is not yet acknowledged, so the
    naive ordering would give it a `present` row as well — two rows for one event, one of
    them accusing the wrong call. The appearance must keep its note (it is the only thing
    that ever got a stray deleted in the same turn) and must be the only row, on this call
    and on the one after it.
    """
    repo, journal = tree
    _as(monkeypatch, WORKER)
    await _seed(tree)

    text = await _run(f"cd {repo} && : > workers.db && echo done")
    assert MARK in text and "workers.db" in text, "the note stands where it belongs"

    await _run(f"cd {repo} && ls >/dev/null && echo again")

    rows = _rows(journal)
    assert [r["kind"] for r in rows] == ["appeared"], rows
    assert rows[0]["paths"] == ["workers.db"], rows[0]
    assert _ack().get(str(repo), {}).get("paths") == ["workers.db"], (
        "an acknowledged appearance is what stops the next call calling it `present`")


async def test_a_present_stray_that_is_removed_and_laid_again_is_journalled_as_present_again(
        tree, monkeypatch):
    """The acknowledgement is of a path STANDING in the tree, not of a path ever seen.

    Once the file is gone the acknowledgement is false, and leaving it set would make this
    the instrument that can name a writer exactly once per filename for the life of the
    tree — the second `Write` into `knowledge/` would be as silent as the first was. The
    `removed` row is the moment the tree stops holding the path, so it is the moment the
    acknowledgement drops, and the row below is the proof.
    """
    repo, journal = tree
    _as(monkeypatch, WORKER)
    await _seed(tree)
    _written_by_another_writer(repo, "usage.db")
    await _run(f"cd {repo} && echo found")
    await _run(f"cd {repo} && rm -- usage.db && echo gone")
    _written_by_another_writer(repo, "usage.db")      # laid again, by a non-Bash writer
    await _run(f"cd {repo} && echo seen-again")

    rows = _rows(journal)
    assert [r["kind"] for r in rows] == ["present", "removed", "present"], rows
    assert [r["paths"] for r in rows] == [["usage.db"]] * 3, rows


async def test_an_after_read_that_fails_journals_no_present_row(tree, monkeypatch):
    """Fail-open on the new branch: an unreadable after-read may not invent a stray.

    `live_strays.ignored` returns None when git refuses, and `None` is UNKNOWN rather than
    empty — `app/live_strays.py` keeps the two apart by type for exactly this reason, and
    the module's own docstring calls a guard whose missing input reads as a clean answer
    "not a guard". Here the inverse risk applies: subtracting from an unknown set would
    read as "every path in the tree is new", which is the accusation the type exists to
    prevent. So the call journals nothing, including none of the new kind, and the tree
    keeps its acknowledgement state exactly as it was.
    """
    repo, journal = tree
    _as(monkeypatch, WORKER)
    await _seed(tree)
    real = live_strays.ignored
    calls = {"n": 0}

    def flaky(root):
        calls["n"] += 1
        return real(root) if calls["n"] == 1 else None    # before-snapshot fine, after fails

    monkeypatch.setattr(live_strays, "ignored", flaky)
    _written_by_another_writer(repo, "usage.db")
    before_ack = json.dumps(_ack(), sort_keys=True)
    text = await _run(f"cd {repo} && echo done")

    assert text.startswith("done") and MARK not in text
    assert _rows(journal) == [], "an unknown after-read must journal nothing"
    assert json.dumps(_ack(), sort_keys=True) == before_ack, (
        "a failed read must not rewrite the state either: rewriting it from an unknown set "
        "would acknowledge nothing and silently reset every tree-path already known")


def test_the_committed_tree_strays_witness_bytes_still_carry_the_quoted_report():
    """Clause 6: the numbers the item quotes come out of committed bytes, not a live file.

    `~/lloyd-data/safety/tree-strays.jsonl` is a rolling runtime journal: retention sweeps
    it, and the row that proves the gap today is one `rm` and one sweep away tomorrow. So
    the bytes the premise rests on are committed — clause 6's home is
    `backlog/data/tree-strays.jsonl` in the vault, and this copy is byte-identical to it
    (sha256 `28b179aa…4756`) so a node can open them with the gate's HOME pointed at the
    round home. The re-derive is the operation clause 6 names, `wc -l` on the file, and the
    figure it prints is the one the item quotes: 5.

    Which of those 5 rows matter is the rest of the node: one `actionable_by` field on a
    single row (the `grep -c actionable_by` the item quotes), three `appeared` and two
    `removed`, zero `present` — the zero IS the premise — and no `appeared` row naming a
    `knowledge/` path anywhere in the file, which is the origin row the item says can never
    exist. Row 5 is the file's own witness of that: `removed knowledge/`, session
    `20261004_023233_autocode_8e79`, at `2026-10-04T10:42:21+00:00`.
    """
    text = WITNESS.read_text(encoding="utf-8")
    counted = subprocess.run("wc -l < " + str(WITNESS), shell=True, capture_output=True,
                             text=True, check=True).stdout.strip()
    assert counted == "5", f"`wc -l` prints {counted}, not the 5 the item quotes"

    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    assert len(rows) == 5, rows
    assert hashlib.sha256(WITNESS.read_bytes()).hexdigest() == WITNESS_SHA256, (
        "these bytes are the extract; if they moved, every figure below and every comment "
        "quoting them needs re-measuring, not this assertion editing")

    kinds = collections.Counter(r.get("kind") for r in rows)
    assert dict(kinds) == {"appeared": 3, "removed": 2}, (
        f"the file gained or lost a kind: {dict(kinds)} — the zero `present` rows is the "
        "premise, so a `present` here means the extract was replaced by a post-fix journal")
    assert kinds.get("present", 0) == 0

    assert sum(1 for r in rows if "actionable_by" in r) == 1, (
        "`grep -c actionable_by` is quoted as 1: rows 1-4 predate the field's own commit")
    assert rows[-1]["kind"] == "removed" and rows[-1]["paths"] == ["knowledge/"], rows[-1]
    assert rows[-1]["session"] == "20261004_023233_autocode_8e79", rows[-1]
    assert rows[-1]["at"] == "2026-10-04T10:42:21+00:00", rows[-1]
    assert rows[-1]["actionable_by"] == "guardian-strays", rows[-1]
    assert sum(1 for r in rows if r.get("kind") == "appeared"
               and any("knowledge" in p for p in r.get("paths", []))) == 0, (
        "an origin row for `knowledge/` appeared in the extract: the item's claim that the "
        "writer of a Write-created stray cannot be named from this file would be false")

    # The durable copy clause 6 names, asserted as BYTES. Compared, not opened as the
    # source: the gate points HOME at the round home, where `~/obsidian` is not, and a node
    # that READ the vault would skip there and pin nothing at the run that matters — so the
    # repo copy above is what the gate verifies, and this branch is what a local run verifies.
    #
    # It used to accept a whole-line PREFIX of the extract on the reasoning that an
    # append-only journal makes an older extract a clean subset. That was the gap the review
    # rung named: #2172's 4-line copy (`git show 120674bf:backlog/data/tree-strays.jsonl`)
    # satisfies a prefix test while quoting 4 lines and zero `removed knowledge/`, so the
    # durable copy could have stayed stale and still been green. Clause 6's copy is landed
    # (vault `7bee31d5`), so equality is now the requirement, not a stricter one: the two
    # files must be the same bytes, which is also what their shared sha256 says. The prefix
    # branch's second assert — no `"present"` line in the durable copy — is gone because it
    # is now implied: equality with these bytes, whose zero `present` rows are asserted
    # above, leaves nothing for a separate scan to add.
    if VAULT_WITNESS.is_file():
        assert VAULT_WITNESS.read_bytes() == WITNESS.read_bytes(), (
            f"clause 6's durable copy at {VAULT_WITNESS} is not the bytes this node opens, "
            "so the two witnesses quote different numbers about one journal")
    else:
        print(f"#2220: {VAULT_WITNESS} is not readable from this home; the repo copy is what "
              "was asserted above, and the durable copy is compared byte for byte whenever "
              "it is. This branch running is not a pass on the durable copy — it is the "
              "gate's HOME, and the copy's own bytes are what vault_land committed")


# ---------------------------------------------------------------- page and kinds

def test_the_architecture_page_names_all_three_kinds_and_the_state_that_decides_them():
    """The prose a reader acts on is pinned to the code that produces it.

    #2095's failure was this page describing a layer that had never run; #1996's was a
    route naming a path no code wrote. The same shape applies to the kinds: a journal
    row is read by a human or the guardian, and the sentence that explains it is on
    `architecture/data-home.md`. Asserting the page names all three kinds, names the
    state file the `present` decision reads, and says the row carries no note means a
    fourth kind, or a `present` row that suddenly DID attach a note, has to be written
    into the page as well as the code — and the last assert is what makes it one
    measurement rather than two: the kinds the page lists are the kinds the module's
    own calls can produce.
    """
    here = Path(__file__).resolve().parent.parent
    page = (here / "architecture" / "data-home.md").read_text(encoding="utf-8")
    src = (here / "agent_mcp" / "_bash_tree_strays.py").read_text(encoding="utf-8")

    for kind in ("appeared", "removed", "present"):
        assert f"`{kind}`" in page, f"the page stopped naming the `{kind}` kind: {page[:200]}"
    assert _bash_tree_strays.ACK_FILE in page, (
        "the page must name the file the `present` decision reads, or a reader chasing a "
        "missing origin row has nowhere to look")
    assert "did not make it" in page, "the page states the no-note rule and why"

    kinds = set(re.findall(r'kind[^=]*=\s*"(\w+)"', src))
    assert kinds == {"appeared", "removed", "present"}, (
        f"the module can emit {sorted(kinds)} but the page describes three; one of them is "
        "undocumented or dead")
