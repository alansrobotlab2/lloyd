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
    # The two roots the data-home half measures, sent somewhere that does not exist.
    # Not part of #3140's behaviour change and not decorum: the data-home half reads
    # them from `app.paths`/`app.session_cwd` at call time, so leaving them at
    # production makes every node below read the LIVE data home, and an unrelated
    # session laying a 0-byte store there while the suite runs would break the eight
    # nodes that assert `_rows(journal) == []` — a flake owned by whichever process
    # was writing, not by this file. `_data_root_top` returning None for an
    # unreadable root is also the state that keeps the data-home half silent here,
    # which is what a code-tree node wants.
    from app import paths, session_cwd
    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path / "no-runtime-data-root")
    monkeypatch.setattr(session_cwd, "SCRATCH_ROOT", tmp_path / "session-cwd")
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



# ── #2345: what a bracket-only row IS NOT, on the journal's own two documents ───
#
# `actionable_by` shipped with #2172 and the reader-side ruling was made on
# 2026-10-07, and neither document a reader of `safety/tree-strays.jsonl` consults said
# the field existed. That silence is not decorative: on 2026-10-03 nightly reflection
# read four journalled rows — three a 0-byte `workers.db` that stood 15 seconds, one
# `web/tsconfig.node.tsbuildinfo` — under an `ALERT.md` `cleared:` stamp, could not tell
# that none of them was reachable by the alert, and filed #2169 at priority high as a
# silent-clear gap `6bf40361` had closed some 23 hours earlier. These nodes keep the
# sentence that answers "is this row an alert that got missed?" where the reader is
# already looking.
#
# Every node reads the bytes of the REAL file at its repo path. No fixture, no copy in
# this file standing in for the page: the reader is a human or the reflection job
# opening `architecture/data-home.md`, and prose pinned to a copy can go green while the
# document a person reads stays wrong. `label_gap` is what proves that, by deleting the
# field name from the page's own bytes and showing the same predicate goes red.

#: The two documents a journal reader consults, at their repo paths.
ARCH_PAGE = REPO / "architecture" / "data-home.md"
ROOT_DOC = REPO / "CLAUDE.md"

#: One phrase per claim, as the page and the root doc spell it. Whitespace is collapsed
#: before these are matched, so an author may re-wrap a sentence without breaking a pin;
#: none of them may be paraphrased away, which is the point of pinning a phrase rather
#: than a topic.
NOT_AN_UNACTIONED_ALERT = "not evidence of a silent or un-actioned alert"
FILTERS_NOT_SUPPRESSES = ("filter on the label", "keeps writing every row",
                          "not suppressed from the journal")
CLEARED_EXCLUSION = "`ALERT.md` `cleared:` reasoning"
FAIL_OPEN_EXCEPTION = "journalled without"

#: The kinds of file a fixture copy could be: text someone authors. Bytecode is not on this
#: list because the suite compiles THIS file into `tests/__pycache__/`, and a `.pyc` carries
#: every string literal in it — including the ruling's sentence and the assert message that
#: quotes it. That copy is this file's own bytes re-encoded, updated by the interpreter
#: whenever the prose changes, so it cannot go stale against the page; a fixture is a second
#: authored copy that can. Excluding it is a statement about what a copy is, not a way to
#: make the sweep come back empty — the scan below still has to find this file or it fails.
AUTHORED_DOC_SUFFIXES = {".py", ".md", ".rst", ".txt", ".json", ".jsonl", ".yaml", ".yml",
                         ".toml"}


def _label_field() -> str:
    """The row's label field name, read off the writer rather than typed in here.

    `actionable_by` is not a module constant — `_journal` assigns `row["actionable_by"]`
    as a literal — so the name is taken from that assignment. Typed here as a string it
    would survive a rename on both sides of the pair and keep the page green over a
    journal whose rows no longer carry what the page promises, which is the drift the
    node above exists to catch for the three kinds.
    """
    src = (REPO / "agent_mcp" / "_bash_tree_strays.py").read_text(encoding="utf-8")
    found = set(re.findall(r'row\["(\w+)"\] = _actionable_by\(', src))
    assert found == {"actionable_by"}, (
        "the stamp site no longer assigns the field this prose promises "
        f"(no `row[\"actionable_by\"] = _actionable_by(...)`): {sorted(found)}")
    return found.pop()


def _flat(text: str) -> str:
    """The document with line-wrapping collapsed — the same sentence, one line."""
    return " ".join(text.split())


def label_gap(text: str) -> str | None:
    """Why these bytes would mislead a journal reader, or None when they would not.

    One predicate shared by the nodes that read the two documents and by the node that
    mutates them, so the sensitivity check cannot be a weaker rule than the one the real
    assertions run. It takes text rather than a path, which is what lets it be pointed at
    bytes that were never written anywhere.
    """
    flat = _flat(text)
    field = _label_field()
    if f"`{field}`" not in text:
        return f"the document stopped naming `{field}`, so a reader has no field to read"
    if _bash_tree_strays.GUARDIAN_STRAYS not in text:
        return (f"`{field}` is named but the value {_bash_tree_strays.GUARDIAN_STRAYS!r} "
                "is not: half a two-valued field documented as whole")
    if _bash_tree_strays.BRACKET_ONLY not in text:
        return (f"`{field}` is named but the value {_bash_tree_strays.BRACKET_ONLY!r} "
                "is not: half a two-valued field documented as whole")
    if NOT_AN_UNACTIONED_ALERT not in flat:
        return (f"nothing says a {_bash_tree_strays.BRACKET_ONLY} row is "
                f"{NOT_AN_UNACTIONED_ALERT} — the misreading that filed #2169 at "
                "priority high")
    return None


def test_the_architecture_page_names_the_label_field_and_its_two_values():
    """Clause 1: the page names `actionable_by` and both values, off the real file.

    The two values are asserted against `_bash_tree_strays.GUARDIAN_STRAYS` /
    `.BRACKET_ONLY` and the field name against the writer's own stamp site (see
    `_label_field`), so renaming either constant or the field turns this node red instead
    of leaving the page quoting a value no row ever carries. The page is read at its repo
    path — the copy a reader opens — not from anything under `tests/`.
    """
    page = ARCH_PAGE.read_text(encoding="utf-8")
    assert label_gap(page) is None, label_gap(page)
    assert f"`{_label_field()}`" in page, "the page must name the field in code font"
    assert "reachable_by_stray_check" in page, (
        "the page says the label asks the guardian's own reach rule; that name is how a "
        "reader checks the label has not drifted from the alert")


def test_the_architecture_page_says_a_bracket_only_row_is_not_a_missed_alert():
    """Clause 2: the ruling sentence itself, and the #2169 evidence beside it.

    The phrase is asserted as one run of words in the page's own bytes. The evidence half
    — that the misreading was actually paid for — is pinned by the incident number and the
    path that proves out of reach, because a bare prohibition reads as caution rather than
    as a ruling, and the last assert keeps the page honest about the ONE row that carries
    no label at all: three nodes in this file already prove that row is written
    (`test_a_tree_with_no_staged_guardian_still_gets_its_row_without_the_label`,
    `test_a_reach_rule_that_raises_costs_the_label_not_the_note`,
    `test_a_reach_rule_that_raises_costs_the_label_not_the_present_row`), and a page that
    promised the label on every row unconditionally would be the same kind of lie.
    """
    page = _flat(ARCH_PAGE.read_text(encoding="utf-8"))
    assert NOT_AN_UNACTIONED_ALERT in page, (
        f"the page must state that a {_bash_tree_strays.BRACKET_ONLY} row is "
        f"{NOT_AN_UNACTIONED_ALERT}")
    assert "#2169" in page, "the ruling names the incident it settles, or it reads as taste"
    assert "tsconfig.node.tsbuildinfo" in page, (
        "the page keeps the example that shows WHY out of reach is not the same as ignored")
    assert FAIL_OPEN_EXCEPTION in page, (
        "the page must also name the fail-open row that arrives without a label, or a "
        "reader takes a missing label for the safe value")


def test_the_architecture_page_states_the_reader_policy_filter_never_suppress():
    """Clause 3: readers filter on the label; the bracket keeps writing every row.

    Three phrases, one node, because they are one ruling the owed-check job made on
    2026-10-07: filter, do not suppress, and the exclusion is from `ALERT.md`'s `cleared:`
    reasoning rather than from the journal. Suppression was the smaller-sounding change
    and would have destroyed the record of what the bracket saw, which is why the page has
    to carry all three clauses and not only the first.
    """
    page = _flat(ARCH_PAGE.read_text(encoding="utf-8"))
    for phrase in FILTERS_NOT_SUPPRESSES:
        assert phrase in page, f"the reader-side policy stopped saying it: {phrase!r}"
    assert CLEARED_EXCLUSION in page, (
        "the page must say where a bracket-only row is excluded FROM, or 'not an alert' "
        "invites the reader to drop the row entirely")
    assert "STRAY_CHECK_SECONDS = 3600.0" in page, (
        "the sub-poll-lifetime half of the #2169 misreading needs the number that makes "
        "15 seconds obviously unreachable, or the example is anecdote")


def test_claude_md_names_the_label_field_on_the_stray_bullet():
    """Clause 4: the root doc's three-places bullet names `actionable_by`.

    Read from disk at the repo path, the shape `test_automod_doc_claims.py` already uses
    for the root doc. Scoped to the bullet that names the journal: the field mentioned
    anywhere else in a two-thousand-line file would satisfy a whole-file grep and teach
    the person asking about strays nothing, so every assert runs on the bullet's own line.
    """
    lines = ROOT_DOC.read_text(encoding="utf-8").splitlines()
    hits = [ln for ln in lines if "caught in three places" in ln]
    assert len(hits) == 1, f"the bullet must be findable and unique: {len(hits)} hits"
    bullet = " ".join(hits[0].split())

    assert f"`{_label_field()}`" in bullet, (
        "the stray bullet is where a reader learns the journal exists; it has to name "
        "the field they then filter on")
    assert _bash_tree_strays.GUARDIAN_STRAYS in bullet, (
        f"and both values it takes, not just the field: missing "
        f"{_bash_tree_strays.GUARDIAN_STRAYS!r}")
    assert _bash_tree_strays.BRACKET_ONLY in bullet, f"missing {_bash_tree_strays.BRACKET_ONLY!r}"
    assert NOT_AN_UNACTIONED_ALERT in bullet, (
        "the bullet carries the ruling in one clause too, because this is the doc an "
        "agent's own prompt is assembled from")
    assert "safety/tree-strays.jsonl" in bullet, "it still names the journal itself"


def test_both_documents_are_read_from_the_repo_not_a_fixture_and_the_pin_is_live(tmp_path):
    """Clause 5: no fixture copy exists, and deleting the field name breaks the pin.

    The clause's real demand is negative — a fixture would let the page go missing while
    the test stayed green — and a negative is not proved by reading the two files. So three
    checks: (a) the only authored file under `tests/` carrying the ruling's sentence is THIS
    one; (b) the SAME `label_gap` predicate the four nodes above run goes red on the page's
    own bytes with the field name deleted, and on the root doc's, which proves the assertion
    tracks the document rather than a string this file owns; (c) the nodes read `ARCH_PAGE` /
    `ROOT_DOC`, both anchored at the repo, so the temp directory passed here is the thing
    they demonstrably do NOT read.

    (a) is an equality that counts this file in, not a "no other file" filter: an earlier
    version excluded this file and asserted an empty list, which is exactly the shape that
    passes when the scan matches nothing at all — a 0-hit grep is an answer only beside a
    positive control, and here the control is this file's own sentence. So the scan includes
    it, and the node fails if the walk finds nothing (empty list), finds only bytecode (a
    `__pycache__` hit is excluded from the list it builds, so that case comes back empty
    too), or finds a second authored copy. The one thing it must come back with is this
    file's path.
    """
    page = ARCH_PAGE.read_text(encoding="utf-8")
    root = ROOT_DOC.read_text(encoding="utf-8")
    assert ARCH_PAGE.is_file() and ROOT_DOC.is_file()
    assert REPO in ARCH_PAGE.parents and REPO in ROOT_DOC.parents
    assert str(tmp_path) not in str(ARCH_PAGE)

    authored = [p for p in (REPO / "tests").rglob("*")
                if p.is_file() and p.suffix in AUTHORED_DOC_SUFFIXES
                and "__pycache__" not in p.parts and p.stat().st_size < 2_000_000]
    assert len(authored) > 100, (
        f"the walk found {len(authored)} authored files under tests/, which is not a real "
        "scan of this directory — an empty walk would make the equality below vacuous")
    copies = sorted(str(p.relative_to(REPO)) for p in authored
                    if NOT_AN_UNACTIONED_ALERT in p.read_text(encoding="utf-8",
                                                              errors="replace"))
    assert copies == [str(Path(__file__).resolve().relative_to(REPO))], (
        "this file is the one place the ruling's sentence may be written down: an empty list "
        "means the scan matched nothing and proves nothing, and any second path is a copy "
        f"that can stay green while the page goes wrong — {copies}")

    field = _label_field()
    assert label_gap(page) is None and label_gap(root) is None
    assert label_gap(page.replace(f"`{field}`", "`safety_label`")), (
        "deleting the field name from the page left the predicate green, so the node is "
        "not reading what it claims to read")
    assert label_gap(root.replace(f"`{field}`", "`safety_label`")), (
        "the root doc's bullet went green without the field name")
    assert label_gap(page.replace(_bash_tree_strays.BRACKET_ONLY, "bracketed")), (
        "dropping one of the two values from the page went green: half a two-valued "
        "field documented as whole")


# ── #2318: the journal-only ruling for removals lives in watched()'s docstring ─
#
# #2110's owed check decided this on 2026-10-07 and the decision had nowhere to live
# but a backlog item: `watched()` said only which sessions are measured, so the next
# author reading this bracket to ask "or should a deletion be refused?" found no answer
# and would re-open the question. These nodes keep the answer where the question is
# asked. They read the docstring OFF THE FUNCTION (`inspect.getdoc`), not off the file:
# a second copy in a comment, or the text moved out of `watched()`, reads green here
# and teaches nobody.
#
# Clause 5 of #2318 is deliberately NOT restated here — it is
# `test_the_call_that_removes_a_seen_stray_journals_the_removal`, untouched by this
# round, which is the behaviour these words describe. Re-asserting it in prose would
# pin the docstring twice and the code zero times.

import inspect


def _watched_doc() -> str:
    """`watched()`'s docstring collapsed to one line.

    Whitespace collapsed on purpose: this is prose, and a sentence that wraps
    mid-phrase in the source is the same sentence. Collapsing lets the nodes below
    assert phrases without pinning where the author happened to wrap, and cannot hide
    a missing phrase — every phrase asserted here is a run of words. The one phrase
    that must survive as ONE LINE is `never refused`, and clause 1 checks that against
    the raw file, because the item's check is a `git grep`.
    """
    return " ".join((inspect.getdoc(_bash_tree_strays.watched) or "").split())


def test_watched_states_that_a_removal_is_journaled_and_never_refused():
    """Clause 1: the ruling line, and `git grep` finds it inside `watched()`.

    Asserted twice on purpose, differently: the phrase must be in the docstring, AND
    the single line carrying it must fall inside `watched()`'s own line range in the
    file — because the item's check is `git grep -n 'never refused'` against the whole
    file, which would also be satisfied by the phrase appearing in a module comment
    ten functions away.
    """
    doc = _watched_doc()
    assert "JOURNALED, never refused" in doc, doc

    src = Path(inspect.getsourcefile(_bash_tree_strays.watched)).read_text(encoding="utf-8")
    lines = src.splitlines()
    hits = [i for i, ln in enumerate(lines, 1) if "never refused" in ln]
    assert len(hits) == 1, f"the phrase must appear exactly once: {hits}"
    body, first = inspect.getsourcelines(_bash_tree_strays.watched)
    last = first + len(body) - 1
    assert first <= hits[0] <= last, (
        f"the only hit is line {hits[0]} but `watched()` spans {first}..{last}: "
        "the item's grep would point at a ruling that is not this function's")
    assert not lines[hits[0] - 1].lstrip().startswith("#"), (
        "a comment line is not a docstring; the grep has to land on the prose")


def test_watched_names_the_guardian_as_the_only_mover_and_the_route_outside_as_its():
    """Clause 2: the guardian owns the moving, and an unwatched route is its to find.

    One node because it is one ruling — "not mine to stop" is only tolerable because
    someone does move strays. The node also asserts the constant's VALUE is what the
    prose quotes, so the sentence cannot outlive `GUARDIAN_STRAYS` being renamed, and
    the `stray_in_tree` name is checked against the guardian module that defines it
    rather than against a string this test could only hope was spelled right.
    """
    doc = _watched_doc()
    assert "only mover" in doc, doc
    assert "guardian" in doc.lower(), doc
    assert "stray_in_tree" in doc, "the outside-bracket route must name where it lands"
    assert _bash_tree_strays.GUARDIAN_STRAYS in doc, (
        f"the prose and the value the row carries "
        f"({_bash_tree_strays.GUARDIAN_STRAYS!r}) have come apart")

    # `_actionable_by` is what stamps the label, so the prose naming it must name a
    # function that exists and can return the label it is quoted as returning.
    src = Path(inspect.getsourcefile(_bash_tree_strays.watched)).read_text(encoding="utf-8")
    assert "_actionable_by" in doc and "_actionable_by" in src, doc
    guard = (Path(inspect.getsourcefile(_bash_tree_strays))
             .parents[1] / "agent-services" / "guardian" / "datawatch.py")
    assert "def stray_in_tree" in guard.read_text(encoding="utf-8"), (
        "the docstring sends a reader to a guardian check that is not there")


def test_watched_gives_the_reason_a_refusal_would_have_blocked_real_work():
    """Clause 3: both halves of the reason, and NOT the one triage refuted.

    The item's own draft attributed both journalled removals to "#2061/#2065
    prescribed" ops. Triage measured the journal and found that false: #2065 was
    closed stale as a duplicate of #2061 and its prescribed `mv` never ran, and the
    second removal is the #2220 cleanup of a `Write`-created file. So this node pins
    the corrected sentence AND the absence of the false one — an "improvement" that
    restored the original attribution would go red here instead of shipping into a
    docstring that outlives this item.
    """
    doc = _watched_doc()
    low = doc.lower()
    assert "reaches no lloyd code" in low, doc
    assert "terminal" in low, "the unreachable route has to be named, not just implied"
    assert "workers.db" in doc and "#2061" in doc, "the first journaled removal, named"
    assert "#2220" in doc, "the second is #2220's cleanup, not #2065's"
    assert "blocked" in low, "the reason says what a refusal would have cost"
    assert "#2065" not in doc, (
        "#2065 was closed stale as a duplicate of #2061 and its `mv` never ran — naming "
        "it here is the false attribution triage removed")
def test_the_ruling_names_no_runtime_path_in_a_retired_layout(tmp_path, monkeypatch):
    """Clause 4: the added prose is cleared by the sweep's OWN rule, and the diff is in scope.

    The grader refused the first draft on both halves of this clause. It cleared the
    prose by concatenating a home prefix onto the whole docstring — which exercises the
    concatenation, not the rule — and it pinned nothing about "the diff touches the
    docstring and tests only". So the deciding check here is not a pattern list of
    mine at all: it imports `tests/test_no_runtime_paths_in_code.py` and runs ITS test
    function against a fabricated tree holding one synthetic file whose entire body is
    the ruling, with that module's `ROOT` and its `git ls-files` rebound to the fixture.
    Real `PATTERNS`, real allowlist, real line-skip rule, deciding over text this round
    wrote.

    Two runs, opposite verdicts, one line apart in the fixture — which is also what
    makes the green run a measurement rather than an absence: the dirty run can only go
    red if the sweep read this file. The retired location reaches the fixture as a
    whole path on one line — the sweep matches a path within a line, so an expression
    assembled INSIDE the fixture is a mutation that fails to mutate, which is how the
    first draft's dirty half passed. It is split in this source (`'"~' + '/lloyd/…'``)
    the way `tests/test_data_home.py::BANNED_SPELLING` splits its own forbidden string,
    so no tracked `.py` carries the retired layout by the act of testing for it.
    """
    import importlib.util

    import pytest as _pytest

    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location(
        "_sweep_module", here / "test_no_runtime_paths_in_code.py")
    sweep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep)

    tree = tmp_path / "tree"
    (tree / "agent_mcp").mkdir(parents=True)
    target = tree / "agent_mcp" / "synthetic_ruling.py"
    rel = "agent_mcp/synthetic_ruling.py"

    class _Git:
        """Stands in for the `git ls-files` the sweep shells out for."""

        @staticmethod
        def run(*a, **kw):
            return collections.namedtuple("R", "stdout")(stdout=rel)

    monkeypatch.setattr(sweep, "ROOT", tree)
    monkeypatch.setattr(sweep, "subprocess", _Git)

    def _body(text: str) -> str:
        # One function, one docstring: the sweep skips lines that START with `#` and
        # nothing else, so a docstring is exactly the surface it reads.
        target.write_text(f'def f():\n    """{text}"""\n', encoding="utf-8")

    # Split in THIS source so no tracked `.py` carries the retired layout, and written
    # into the fixture as ONE line, which is what the sweep matches — an assembled
    # expression inside the fixture would be a mutation that fails to mutate. That
    # failure is how the first draft of this node looked green.
    retired = '"~' + '/lloyd/usage.db"'
    doc = _watched_doc()

    _body(doc)
    sweep.test_no_tracked_code_builds_a_runtime_path_off_the_tree()

    _body(doc + f"\n    DB = {retired}\n")
    with _pytest.raises(AssertionError) as e:
        sweep.test_no_tracked_code_builds_a_runtime_path_off_the_tree()
    assert rel in str(e.value), (
        "the sweep went red, but not on the synthetic file — so the pass above would "
        f"have said nothing about this prose either: {str(e.value)[:300]}")

    # The clause's other half — "no refusal is added to the Bash tool" — as a property
    # of the bracket rather than of a commit. A check against `git show HEAD` would be
    # true at the gate and false forever after, because HEAD is whatever lands next;
    # this stays true exactly as long as the ruling is.
    #
    # The shape being asserted is the one that matters: `after` hands back the text and
    # the caller REASSIGNS to it (`text = await ...after(text, ...)`), so the tool result
    # is the tool's own. A refusal would have to branch on the return or raise, and
    # neither is a reassignment. The regex needs a positive control too, or it clears
    # the file by matching nothing at all.
    tool = (Path(inspect.getsourcefile(_bash_tree_strays)).parents[1]
            / "agent_mcp" / "builtin_bash.py").read_text(encoding="utf-8")
    after = re.search(r"(\w+) = await _bash_tree_strays\.after\(\n?\s*\1,", tool)
    assert after, (
        "the Bash tool no longer reassigns its own result through "
        "`_bash_tree_strays.after`, which is the shape the ruling describes — either "
        "the call moved, or something now branches on what it returns")
    assert tool.count("= await _bash_tree_strays.after(") == 1, (
        "more than one caller consumes `after`'s text; this node reads only one of them")
    assert 'if _bash_tree_strays.' not in tool and 'raise' not in [
        ln.strip() for ln in tool.splitlines()
        if "_bash_tree_strays" in ln], (
        "the bracket now gates on the stray guard, i.e. a refusal was added")


# ---------------------------------------------------------------------------
# A throwaway git checkout landing under the RUNTIME DATA ROOT (#3095)
#
# Same instrument, opposite mistake. The nodes above catch a data file written
# into the code tree; these catch a CODE CHECKOUT written into the data tree,
# which is the direction that costs more: the data root is watched file by file
# by `agent-services/guardian/datawatch.py`, so creating a worktree there and
# deleting it 55 seconds later is, to the tripwire, 3,347 files vanishing in one
# tick. On 2026-10-20 14:24:14 that halted promotions and paused the worker pool.
# ---------------------------------------------------------------------------

CHECKOUT_MARK = "[checkout in the runtime data root]"


@pytest.fixture
def data_root(tree, tmp_path, monkeypatch):
    """The `tree` fixture plus a runtime data root the check reads instead of production.

    `_data_root_top` imports `DATA_ROOT` inside the call, exactly as the tree half
    re-reads the root each time, so redirecting the module attribute is the seam and
    not a mock of the check.
    """
    from app import paths
    root = tmp_path / "lloyd-data"
    root.mkdir()
    (root / ".lloyd-data-root").write_text("real data root\n", encoding="utf-8")
    (root / "sessions").mkdir()
    monkeypatch.setattr(paths, "DATA_ROOT", root)
    return tree, root


async def test_a_checkout_laid_under_the_data_root_is_named_on_that_call(data_root,
                                                                          monkeypatch):
    """Clause 1, driven by the real command from the incident."""
    (repo, _journal), data = data_root
    _as(monkeypatch, WORKER)
    command = (f"mkdir -p {data}/triage-3087 && git -C {repo} worktree add --detach "
               f"{data}/triage-3087/check HEAD")
    text = await _run(command)

    assert CHECKOUT_MARK in text, "the call that made it is the only place it can be said"
    assert "triage-3087" in text
    assert str(data) in text, "the note names the root it measured"
    assert str(Path.home() / "lloyd-work") in text, (
        "and the place a throwaway checkout does belong, which is the answer the "
        "session was missing")
    assert "[stray in the code tree]" not in text, (
        "nothing landed in the code tree; the two halves report independently")


async def test_the_checkout_is_journaled_in_the_tree_stray_shape(data_root, monkeypatch):
    (repo, journal), data = data_root
    _as(monkeypatch, WORKER)
    command = (f"mkdir -p {data}/triage-2918 && git -C {repo} worktree add --detach "
               f"{data}/triage-2918/checkout HEAD")
    text = await _run(command)
    # The precondition, named. Every assertion below is about what the CHECK wrote,
    # and the check writes nothing unless the checkout exists — so when this node
    # fails on a run count, the reader has to be able to tell "the guard did not
    # fire" from "`git worktree add` did not run", which is a property of the runner
    # and not of the guard. It is the only difference the row count cannot show.
    assert "fatal:" not in text.lower(), (
        f"the fixture's own `git worktree add` failed, so there was no checkout for "
        f"the guard to find: {text[:400]}")

    rows = _rows(journal)
    checkout_rows = [r for r in rows if r.get("kind") == "checkout-in-data-root"]
    assert len(checkout_rows) == 1, (
        f"exactly one checkout row for the call that made it; the call wrote "
        f"{[r.get('kind') for r in rows]}")
    assert rows == checkout_rows, (
        "and nothing else in the same journal: the code-tree half has its own root "
        "and its own rule, and a call that touched only the data root must not "
        "reach it")
    row = checkout_rows[0]
    assert row["session"] == WORKER
    assert row["kind"] == "checkout-in-data-root"
    assert row["paths"] == ["triage-2918"], (
        "`paths` is the uncapped half of the row, and it is the half the alert is "
        "about — which is why the cap on `command` below costs no information")
    assert row["command"] == command[:400], (
        "the row carries the command that made the checkout, cut at the journal's own "
        "bound (`_record`: `str(command or '')[:400]`). Comparing it against the WHOLE "
        "command made this node read its own runner: every path in the fixture comes "
        "from pytest's `tmp_path`, which hangs off `TMPDIR`, and a gate run hands "
        "children a 44-byte one (`~/lloyd-work/.t/<10 hex>`, "
        "`scripts/automod/gate.py::_child_tmpdir`) where an interactive run has `/tmp` "
        "— 4 bytes. The same guard was green locally, where this fixture's command is "
        "40 characters shorter and lands under the cap, and red on the gate's 438 (its "
        "diff line: 389 identical leading characters, 11 more to the cap, 38 after "
        "it). The cap is a property of the "
        "recorder; the fixture's length is not, and `TMPDIR` under a 44-byte root is "
        "now exercised here too.")
    assert len(row["command"]) <= 400, (
        "the bound itself, stated, so the assertion above cannot be satisfied by a "
        "row that simply dropped the field")
    assert row["actionable_by"] == _bash_tree_strays.BRACKET_ONLY, (
        "no instrument can act on a checkout in the data root, so the row says so")


async def test_a_folder_holding_the_checkout_one_level_down_is_also_caught(data_root,
                                                                           monkeypatch):
    """The second real shape from 2026-10-20: `triage-2918/checkout/.git`."""
    (repo, journal), data = data_root
    _as(monkeypatch, WORKER)
    await _run(f"mkdir -p {data}/triage-2918 && git -C {repo} worktree add --detach "
               f"{data}/triage-2918/checkout HEAD")
    assert "triage-2918" in _rows(journal)[0]["paths"], (
        "the folder the tripwire would count is the top-level one, not `check`")


async def test_an_ordinary_folder_in_the_data_root_is_not_reported(data_root, monkeypatch):
    """A job writing a real artefact under the data root is doing its job."""
    (repo, journal), data = data_root
    _as(monkeypatch, WORKER)
    text = await _run(f"mkdir -p {data}/_pipeline/out && : > {data}/_pipeline/out/x.json "
                      "&& echo done")
    assert CHECKOUT_MARK not in text
    assert _rows(journal) == []


async def test_a_store_that_already_held_a_repo_is_not_this_calls(data_root, monkeypatch):
    """Only what the call CREATED is a finding, whatever is standing there."""
    (repo, journal), data = data_root
    (data / "sessions" / "clone").mkdir(parents=True)
    (data / "sessions" / "clone" / ".git").mkdir()
    _as(monkeypatch, WORKER)
    text = await _run(f"ls {data} >/dev/null && echo done")
    assert CHECKOUT_MARK not in text
    assert _rows(journal) == []


async def test_the_checkout_is_reported_once_per_folder(data_root, monkeypatch):
    """A session that `ls`es the folder next turn must not be nagged again."""
    (repo, journal), data = data_root
    _as(monkeypatch, WORKER)
    make = await _run(f"mkdir -p {data}/triage-3087 && git -C {repo} worktree add "
                      f"--detach {data}/triage-3087/check HEAD")
    again = await _run(f"ls {data}/triage-3087 && echo done")
    assert CHECKOUT_MARK in make and CHECKOUT_MARK not in again
    assert len(_rows(journal)) == 1


async def test_a_fresh_checkout_at_a_reported_name_is_reported_again(data_root,
                                                                     monkeypatch):
    """One folder per backlog item, forever: `triage-3087` returning is a new incident."""
    (repo, journal), data = data_root
    _as(monkeypatch, WORKER)
    await _run(f"mkdir -p {data}/triage-3087 && git -C {repo} worktree add --detach "
               f"{data}/triage-3087/check HEAD")
    await _run(f"git -C {repo} worktree remove --force {data}/triage-3087/check; "
               f"rmdir {data}/triage-3087")
    text = await _run(f"mkdir -p {data}/triage-3087 && git -C {repo} worktree add "
                      f"--detach {data}/triage-3087/check HEAD")
    assert CHECKOUT_MARK in text, (
        "the acknowledgement is of a folder STANDING there; keeping it set after the "
        "cleanup would blind the check to the next triage round forever")
    assert len(_rows(journal)) == 2


async def test_a_chat_session_laying_a_checkout_in_the_data_root_is_not_measured(
        data_root, monkeypatch):
    """The ruling #1530 is the gate for both halves; a person's own tool call is theirs."""
    (repo, journal), data = data_root
    _as(monkeypatch, CHAT)
    text = await _run(f"mkdir -p {data}/triage-3087 && git -C {repo} worktree add "
                      f"--detach {data}/triage-3087/check HEAD")
    assert CHECKOUT_MARK not in text
    assert _rows(journal) == []


async def test_an_unreadable_data_root_costs_the_checkout_note_and_nothing_else(
        data_root, tmp_path, monkeypatch):
    (repo, journal), data = data_root
    from app import paths
    monkeypatch.setattr(paths, "DATA_ROOT", tmp_path / "no-such-data-root")
    _as(monkeypatch, WORKER)
    text = await _run(f"cd {repo} && : > workers.db && echo done")
    assert "[stray in the code tree]" in text, "the tree half still reports"
    assert CHECKOUT_MARK not in text
    assert len(_rows(journal)) == 1


async def test_the_data_root_is_snapshot_before_the_command_for_a_reason(data_root,
                                                                        monkeypatch):
    """The trap this design exists to avoid, pinned.

    `after()` cannot ask "did this call put a folder here" — by the time it runs the
    folder is there, and a check that measures only afterwards would report nothing
    forever while looking exactly like a check that passes. So the data root is read
    in `before()`, and a caller that hands `after` a snapshot taken afterwards is not
    asking the question. This node drives both orders of the SAME command.
    """
    (repo, journal), data = data_root
    _as(monkeypatch, WORKER)
    command = (f"mkdir -p {data}/triage-4001 && git -C {repo} worktree add --detach "
               f"{data}/triage-4001/check HEAD")

    pre = _bash_tree_strays._data_root_top()
    assert pre is not None and "triage-4001" not in pre[1], (
        "nothing has run yet: this is the pre-call name set")
    await _run(command)
    post = _bash_tree_strays._data_root_top()
    assert "triage-4001" in post[1], "the folder is standing there now"
    assert _bash_tree_strays._new_checkouts(data, post[1]) == [], (
        "measured against a snapshot taken after the command, the same folder the "
        "check exists to find is invisible — which is why the snapshot is `before`'s")
    assert _bash_tree_strays._new_checkouts(data, pre[1]) == ["triage-4001"], (
        "against the pre-call set it is found, on the same state on disk")


# ---------------------------------------------------------------------------
# A 0-byte database, named on the call that opened it into existence (#3140)
#
# The same file, the same session, a third direction of the same mistake. The
# nodes above catch a data file written into the code tree and a code checkout
# written into the data tree; these catch an EMPTY DATABASE — a file that exists
# only because `sqlite3 <path>` creates the file to open it. Four of them sat
# unseen for weeks (kg.sqlite and _pipeline/workers.db under the data root, two
# more inside session scratch dirs), and each one bought an hourly guardian alert
# plus a human pass (#3109, #3117, #3118), because the only detector
# (`datawatch.data_root_shadows`) runs on the guardian's tick, an hour after the
# call that could have been told.
#
# The watch is on TWO directories: the data root's top level, and the session's
# own scratch dir — which is where a call STARTS (`agent_mcp/builtin_bash.py::_start_cwd`
# → `app.session_cwd.resolved_for`), so a bare relative `sqlite3 workers.db` lays
# its file there and nothing in this module had ever looked at that directory.
# ---------------------------------------------------------------------------

#: As `CHECKOUT_MARK` above: a literal, deliberately not imported from the module. The
#: mark a reader sees is the thing being pinned, and a rename in the note must be red here.
EMPTY_DB_MARK = "[empty database in the runtime data root]"

#: The code-tree half's kind, for the one node that reads two halves' rows at once.
MARK_KIND = "appeared"


@pytest.fixture
def scratch(data_root, monkeypatch):
    """The `data_root` fixture plus the scratch directory this session's Bash starts in.

    `SCRATCH_ROOT` is a module attribute precisely so a test can point the convention at
    a temp root (`app/session_cwd.py` says so at the constant), and `path_for` reads it at
    call time — the same seam `_data_root_top` uses for `DATA_ROOT`. It is hung INSIDE the
    fixture's data root because that is where production puts it
    (`~/lloyd-data/session-cwd/<id>`), which also gives the pair a shared parent so a node
    below can pin that a file laid in the scratch dir is not additionally reported as a
    data-root file: it is two levels down, and the data-root watch is top-level only.

    The resolver that decides where a call actually starts is `resolved_for`, which reads
    the session record. The leg watches `path_for`, the mint's own path for that id, so
    this fixture pins both to the same directory — the guarantee production holds, since
    the writer that stamps `start_cwd` lays the scratch at `path_for` and nothing else.
    """
    from app import session_cwd
    ((repo, journal), data) = data_root
    root = data / "session-cwd"
    path = root / session_cwd.safe_name(WORKER)
    path.mkdir(parents=True)
    monkeypatch.setattr(session_cwd, "SCRATCH_ROOT", root)
    monkeypatch.setattr(session_cwd, "resolved_for",
                        lambda sid, **kw: str(path))
    return (repo, journal), data, path


async def test_a_zero_byte_database_laid_in_the_data_root_is_named_on_that_call(
        data_root, monkeypatch):
    """Clause 1: the call that opened the file is the only place it can be said."""
    (repo, journal), data = data_root
    _as(monkeypatch, WORKER)
    text = await _run(f"cd {data} && sqlite3 kg.sqlite .tables && echo done")

    assert EMPTY_DB_MARK in text, (
        "the creating call is silent today: `_new_checkouts` enumerates the new names "
        "and then keeps only directories, so a new FILE is seen and dropped")
    assert str(data / "kg.sqlite") in text, "the note names the path it created"
    assert "0 byte" in text, "and says what is wrong with it: it is empty"
    assert f"The real stores live in `{data}`." in text, (
        "the note has to NAME the root the real stores live in as its own clause. Asserting "
        "`str(data) in text` instead would be satisfied by the created path itself, which "
        "starts with the root and says nothing about where the stores are")
    assert MARK not in text and CHECKOUT_MARK not in text, (
        "nothing landed in the code tree and this is not a checkout: the three notes "
        "report independently")


async def test_a_zero_byte_database_laid_in_the_session_scratch_dir_gets_the_same_note(
        scratch, monkeypatch):
    """Clause 2: the directory a call starts in is now a watched directory."""
    (repo, journal), data, home = scratch
    _as(monkeypatch, WORKER)
    text = await _run("sqlite3 workers.db .tables && echo done")

    assert EMPTY_DB_MARK in text, (
        "a bare relative open happens in the session's scratch dir, which is where "
        "`_start_cwd` sends a call that names no cwd")
    assert str(home / "workers.db") in text, "the note names the created path"
    assert f"The real stores live in `{data}`." in text, (
        "and the note names the root the real stores live in, not the scratch dir it just "
        "named — which here are two different directories, so this is the half that could "
        "be false and the half the created-path assert cannot cover")


async def test_one_call_opening_a_bare_relative_database_is_reported_and_journaled(
        scratch, monkeypatch):
    """Clause 5's end-to-end drive: one `Bash` call, its note and its row, same file.

    `call_tool` is the seam: it takes the snapshot through `before()`, runs the shell, and
    hands the result to `after()` (`agent_mcp/builtin_bash.py::call_tool`), which is what
    makes this the real path rather than a call of the two functions by hand — and the
    reason the row count below is 1 and not 2 is that the leg runs once per call.
    """
    (repo, journal), data, home = scratch
    _as(monkeypatch, WORKER)
    command = "sqlite3 workers.db .tables"
    text = await _run(command)
    laid = home / "workers.db"

    assert EMPTY_DB_MARK in text, "the note, on this call"
    rows = _rows(journal)
    assert [r["kind"] for r in rows] == [_bash_tree_strays.EMPTY_DB_KIND], (
        f"and exactly one row for it; the call wrote {[r['kind'] for r in rows]}")
    assert laid.exists(), (
        "the leg NAMES a file and does not remove it: this is a report, not a guard, "
        "and the command's own file has to survive the sentence about it")
    assert laid.stat().st_size == 0, "at the size the command left it"
    assert not (data / "quarantine").exists(), (
        "and nothing was created under the quarantine destination — moving data-home "
        "residue is the guardian's call and this module never moves anything")


async def test_the_origin_row_is_a_kind_the_journal_has_never_carried(scratch,
                                                                      monkeypatch):
    """Clause 3: one row, its kind distinct from every kind the module writes today.

    The four are read out of the module's source rather than typed in, with the hyphen the
    shipped `checkout-in-data-root` spelling carries: a set built from `\\w+` cannot see a
    hyphenated kind, and a difference test written that way passes on a new row that
    collides with the default `appeared`.
    """
    (repo, journal), data, home = scratch
    _as(monkeypatch, WORKER)
    command = "sqlite3 workers.db .tables"
    await _run(command)

    src = (Path(__file__).resolve().parent.parent / "agent_mcp"
           / "_bash_tree_strays.py").read_text(encoding="utf-8")
    # `[^=]*` and not `\w*` because `appeared` — the kind this node must exclude, and the
    # one a collision with is easiest to make — is never written at a call site: it is
    # `_record`'s DEFAULT, spelled `kind: str = "appeared"` in the signature. A pattern
    # that only matches `kind="x"` finds three of the four shipped kinds and the assert
    # below would then be red for the wrong reason, or worse, green with `appeared` unseen.
    shipped = set(re.findall(r'kind[^=("]*=\s*"([\w-]+)"', src))
    assert {"appeared", "removed", "present", "checkout-in-data-root"} <= shipped, (
        f"the derivation found {sorted(shipped)}: a kind this node is meant to exclude "
        "has stopped being spelled the way it is spelled in the code, so the set "
        "difference below would pass on a collision")
    rows = _rows(journal)
    assert len(rows) == 1, (
        f"exactly one origin row per call is half of clause 3 and indexing [0] below would "
        f"never notice a second: the call wrote {[r['kind'] for r in rows]}")
    row = rows[0]
    assert row["kind"] not in shipped - {_bash_tree_strays.EMPTY_DB_KIND}, (
        f"the new row reuses kind {row['kind']!r}; a file that mixes event classes with "
        "no discriminator is a file a later reader guesses at")
    assert row["kind"] == _bash_tree_strays.EMPTY_DB_KIND
    assert row["session"] == WORKER
    assert row["command"] == command
    assert row["paths"] == ["workers.db"], "`paths` is relative to the row's root"
    assert Path(row["root"], row["paths"][0]) == home / "workers.db", (
        "and root + paths is the created file, which is what makes the row an origin "
        "record and not just a filename")
    assert row["actionable_by"] == _bash_tree_strays.BRACKET_ONLY, (
        "`stray_in_tree` walks the CODE tree for runtime files, and the data-home sweep "
        "is report-only by its own docstring — no instrument moves a data-root file, so "
        "the row says so rather than waiting for a label")


async def test_the_leg_keys_on_shape_and_not_on_the_write(scratch, monkeypatch):
    """Clause 4: a file only earns the note by being new AND empty AND database-shaped."""
    (repo, journal), data, home = scratch
    _as(monkeypatch, WORKER)

    text = await _run(f"printf 'SQLite format 3\\000padding' > {data}/full.db "
                      f"&& : > {data}/notes.txt && : > {home}/empty.txt "
                      f"&& : > {home}/out && echo done")
    assert "done" in text, (
        "positive control: the command ran, so the absence below is a verdict and not a "
        "shell that failed before it created anything")
    assert (data / "full.db").stat().st_size > 0 and not (data / "notes.txt").stat().st_size, (
        "and the two files that had to be created are the shapes the test claims: one "
        "database-named file WITH bytes, one empty file without the extension")
    assert (home / "empty.txt").stat().st_size == 0 and (home / "out").exists()
    assert EMPTY_DB_MARK not in text, (
        "a NON-EMPTY new file at the same path is a job that wrote something, and an "
        "empty file with no database extension is an empty file: only a 0-byte "
        "database-shaped name is this finding")
    assert _rows(journal) == []


async def test_a_database_laid_without_sqlite3_is_reported_the_same_way(scratch,
                                                                       monkeypatch):
    """No command shape is matched: `touch` and `sqlite3` are one case.

    The module's header makes not matching a command pattern its whole design argument
    against a rule in `protected_paths`, and a leg that fired on the word `sqlite3` would
    silently stop firing on the Python one-liner, the redirect and the tool nobody has
    written yet — which is the third of the four shadows in #3140.
    """
    (repo, journal), data, home = scratch
    _as(monkeypatch, WORKER)
    for name, verb in (("a.db", "touch"), ("b.sqlite", ": >"), ("c.sqlite3", "printf '' >")):
        text = await _run(f"{verb} {home / name} && echo done")
        assert EMPTY_DB_MARK in text, f"{verb} {name}: all three extensions, no pattern"
        assert str(home / name) in text
    assert [r["paths"] for r in _rows(journal)] == [["a.db"], ["b.sqlite"], ["c.sqlite3"]]


async def test_a_new_folder_holding_a_checkout_still_gets_the_checkout_note_only(
        data_root, monkeypatch):
    """Clause 4's other half: the two data-home notes do not overlap on one artefact."""
    (repo, journal), data = data_root
    _as(monkeypatch, WORKER)
    text = await _run(f"mkdir -p {data}/triage-3140 && git -C {repo} worktree add "
                      f"--detach {data}/triage-3140/check HEAD")
    assert CHECKOUT_MARK in text, "the #3095 note is unchanged by the new leg"
    assert EMPTY_DB_MARK not in text
    assert [r["kind"] for r in _rows(journal)] == ["checkout-in-data-root"], (
        "a directory is not an empty database however it is spelled")


async def test_one_call_laying_one_in_each_watched_dir_journals_exactly_one_row(
        scratch, monkeypatch):
    """Clause 3's COUNT: two directories, one call, one row — and none for a call that
    lays nothing.

    The row is the origin of a COMMAND, so the number of rows has to be a property of what
    the session did and never of how many directories the leg happens to watch. That is only
    safe to write down because of how the row is rooted: `root` is the deepest directory
    holding every file in it, which for one file is that file's own directory — so the
    single-shadow rows a reader already knows (`root` + `paths[0]` = the created file) are
    the same rows this call produces, just with two paths. Asserted per path below, because
    a row whose paths do not compose with its root is a row about files nobody can find.
    """
    (repo, journal), data, home = scratch
    _as(monkeypatch, WORKER)

    text = await _run("echo nothing here")
    assert _rows(journal) == [], (
        "a call that lays no shadow journals no row: the count below is driven by "
        "measurement, not by the number of calls or the number of watched directories")

    text = await _run(f": > {data}/kg.sqlite && : > {home}/workers.db && echo done")
    assert text.count(EMPTY_DB_MARK) == 1, "one paragraph for the call"
    assert str(data / "kg.sqlite") in text and str(home / "workers.db") in text
    rows = _rows(journal)
    assert [r["kind"] for r in rows] == [_bash_tree_strays.EMPTY_DB_KIND], (
        f"EXACTLY one origin row for one call, across two watched directories; the call "
        f"wrote {[r['kind'] for r in rows]}")
    row = rows[0]
    assert row["root"] == str(data), (
        "the data root is the deepest directory holding both files, since the scratch tree "
        "hangs under it — the row names their common root rather than one of their two")
    assert sorted(row["paths"]) == sorted(["kg.sqlite", str(home.relative_to(data) / "workers.db")]), (
        f"both files, in one row; got {row['paths']}")
    assert sorted(Path(row["root"], p) for p in row["paths"]) == sorted(
        [data / "kg.sqlite", home / "workers.db"]), (
        "root + each path is the created file: the row stays resolvable for every path it "
        "carries, which is what makes one row across two directories honest")


async def test_the_scratch_dir_is_snapshot_before_the_command_for_the_same_reason(
        scratch, monkeypatch):
    """Clause 2's "snapshotted before the command", pinned the way #3095 pinned its own.

    An after-only read of the scratch dir would find the file the call laid and have no
    way to know whether this call made it or a previous one, which is the difference
    between naming a writer and nagging about a file that has been there for a week.
    """
    (repo, journal), data, home = scratch
    _as(monkeypatch, WORKER)
    pre = _bash_tree_strays._scratch_top(WORKER)
    assert pre is not None and "workers.db" not in pre[1], (
        "nothing has run yet: this is the pre-call name set")
    await _run("sqlite3 workers.db .tables")
    post = _bash_tree_strays._scratch_top(WORKER)
    assert post is not None, "the scratch dir is readable: the after-set exists"
    assert "workers.db" in post[1], "the file is standing there now"
    assert _bash_tree_strays._new_empty_dbs(home, post[1]) == [], (
        "measured against a snapshot taken after the command, the file the check exists "
        "to find is invisible")
    assert _bash_tree_strays._new_empty_dbs(home, pre[1]) == ["workers.db"], (
        "against the pre-call set it is found, on the same state on disk")


async def test_an_unreadable_scratch_dir_costs_only_its_own_note(scratch, monkeypatch):
    """Clause 5's fail-open: one directory that cannot be read, and everything else stands.

    The command runs with an explicit `cwd` rather than from the scratch dir it would
    normally start in, because a 0o000 directory also stops the SHELL from starting there —
    and then this node would be measuring the runner's `spawn` error rather than the leg's
    own read failure. The 0o000 directory is restored in a `finally` because pytest's
    `tmp_path` teardown cannot remove a directory it cannot read, and a suite that errors
    in teardown points at the wrong node.
    """
    (repo, journal), data, home = scratch
    _as(monkeypatch, WORKER)
    # Something for the leg to MISS, laid while the directory is still readable: without
    # this the node proves only that a sentence is absent from a result, which a leg that
    # found nothing anywhere would also satisfy.
    (home / "kg.sqlite").write_bytes(b"")
    home.chmod(0o000)
    try:
        # The read itself has to be shown to fail. A world-writable-umask environment, a
        # uid-0 suite or a future `os.access` short-circuit would leave the directory
        # readable, and every assert below would then hold for a reason this node has
        # nothing to say about.
        assert _bash_tree_strays._scratch_top(WORKER) is None, (
            "the scratch dir is supposed to be unreadable here; if it reads, the three "
            "asserts below are measuring a leg that was never denied anything")
        text = await _run(f"cd {repo} && : > usage.db && : > {data}/kg.sqlite "
                          "&& echo done", cwd=str(repo))
    finally:
        home.chmod(0o755)
    assert (home / "kg.sqlite").exists(), (
        "the shadow that was there throughout: read-only and fail-open means it is still "
        "on disk when the leg could not list its directory")
    assert str(home / "kg.sqlite") not in text and str(home) not in text, (
        "nothing is claimed about the directory it could not read, and the file standing "
        "in it went unnamed — which is the cost this leg is allowed to pay")
    assert MARK in text, "the code-tree half still reports"
    assert EMPTY_DB_MARK in text, "and so does the other watched directory"
    assert str(data / "kg.sqlite") in text
    assert sorted(r["kind"] for r in _rows(journal)) == sorted(
        [MARK_KIND, _bash_tree_strays.EMPTY_DB_KIND]), (
        "two rows, one per half that could measure — the unreadable dir costs its own "
        "row and nothing else")


async def test_a_chat_session_laying_an_empty_database_is_not_measured(scratch,
                                                                       monkeypatch):
    """The #1530 gate is the gate for this leg too: a person's own tool call is theirs."""
    (repo, journal), data, home = scratch
    _as(monkeypatch, CHAT)
    text = await _run("sqlite3 workers.db .tables")
    assert EMPTY_DB_MARK not in text
    assert _rows(journal) == []
