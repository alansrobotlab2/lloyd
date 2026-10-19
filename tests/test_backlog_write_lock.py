"""#3036 — no backlog writer may read-modify-write an item file unlocked.

Three lanes rewrite one item file, as three separate processes, by design:
`agent_mcp.main` answering `backlog_write_task` (`agent_mcp/backlog.py::save_task`),
`server.py` answering the board modal (`app/routers/backlog.py::_write_task_file`),
and the automod loop's own finalizer, status reconciler, note-writer and tag-writer
(`scripts/automod/backlog.py`). Each is a read-modify-write ending in a replace of
the whole file. With nothing between the parse and the replace, two things happen
and neither leaves a trace:

* the later writer's dump carries only the keys its own parse saw, so the earlier
  writer's `activity_log` row and any key only it knew are gone — while the automod
  ledger still records the move that was just lost, which is the board-file-versus-
  ledger disagreement the item names;
* `Path.write_text` opens with `O_TRUNC`, so a reader that hits the file mid-write
  can get a prefix cut inside a quoted scalar, which does not parse, which marks the
  item `_yaml_broken`, which becomes a 409 on every later edit of it (#918).

This file pins the five clauses the triage recorded. Each one **fails against the
pre-change tree at `641cf23a`**, and not merely because `app.backlog_write` does not
exist there to import: the four writer modules at that commit contain no
`commit_lock` call, and both route-side writers end in
`filepath.write_text(f"---\\n{fm_yaml}---\\n\\n{body}")`, so clause 2's AST walk finds
no lock block, clause 1's losing write truncates the file with a stale journal, and
clause 3's forced failure happens *after* the old writer already truncated it.
`tests/test_backlog_write_lock.py::test_the_pre_change_tree_really_had_no_writer_lock`
below is the control that keeps this paragraph honest.

The board is redirected to a tmp directory for every test here — these tests write
item files, and the real board is the live vault (`~/obsidian/backlog`, 1,180 files,
tracked and pushed by `agent-obsidian-sync`).
"""

import ast
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import HTTPException  # noqa: E402

from app import atomic_io  # noqa: E402
from app import backlog_write as BW  # noqa: E402
import agent_mcp.backlog as MCP  # noqa: E402
import app.routers.backlog as BR  # noqa: E402
import scripts.automod.backlog as B  # noqa: E402
import scripts.automod.owed as O  # noqa: E402

BOARD = "lloyd"
#: The text of each writer's row, not the stored line: `update_frontmatter` and
#: `note_item` stamp what they are given (`**<stamp>** — <text>`), so a test looks
#: for its marker *inside* a row. Matching the whole line would pin a stamp format
#: no clause is about.
B_MSG = "loop row: up_next -> in_progress"
#: The aggregator builds its journal as data, so its row is verbatim.
A_ROW = "**2026-10-19T20:00:01** — aggregator row"

_HEAD = """---
type: note
segment: backlog
tags:
- backlog
id: {tid}
name: '#3036 — no backlog writer locks the file or writes atomically'
priority: medium
status: '{status}'
board: lloyd
created: '2026-10-19T19:00:00'
updated: '2026-10-19T19:00:00'
{extra}---

# 3036 — No backlog writer locks the file or writes atomically

The prose of the item. Not this route's to own.
"""


def _item_file(board, tid=3036, *, status="in_progress", extra=""):
    board.mkdir(parents=True, exist_ok=True)
    path = board / f"{tid}-no-backlog-writer-locks-the-file-or-writes-atomically.md"
    path.write_text(_HEAD.format(tid=tid, status=status, extra=extra), encoding="utf-8")
    return path


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """A tmp board, and a lock/temp root the tests may watch, both off-tree.

    `LOCK_DIR`/`SCRATCH_DIR` are the seams `commit_lock` and `atomic_write_text`
    read at call time, so redirecting them here is what lets
    `test_no_lock_or_temp_artefact_lands_inside_the_board_directory` assert on the
    *real* artefacts a write produces rather than on a stub: it can only conclude
    the board stayed clean if the lock and the temp actually went somewhere.
    """
    board = tmp_path / "backlog"
    locks = tmp_path / "locks"
    scratch = locks / "tmp"
    monkeypatch.setattr(atomic_io, "LOCK_DIR", locks)
    monkeypatch.setattr(atomic_io, "SCRATCH_DIR", scratch)
    monkeypatch.setattr(B, "BACKLOG_DIR", board)
    monkeypatch.setattr(MCP, "BACKLOG_DIR", board)
    monkeypatch.setattr(BR, "_BACKLOG_DIR", board)
    with BR._FM_CACHE_LOCK:
        BR._FM_CACHE.clear()
    yield {"board": board, "locks": locks, "scratch": scratch,
           "item": _item_file(board)}
    with BR._FM_CACHE_LOCK:
        BR._FM_CACHE.clear()


def _fm_of(path):
    block = BW.FM.split_frontmatter(path.read_text(encoding="utf-8"))
    return yaml.safe_load(block[0])


def _hold_lock_forever(item_path, ready, release):
    """Thread body: hold the item's commit lock (a separate fd, so a separate
    lock owner — `flock` denies a second fd even inside one process) until the
    test says stop."""
    with atomic_io.commit_lock(item_path, timeout=5) as lock:
        assert lock == atomic_io.lock_file_for(item_path)
        ready.set()
        release.wait(30)
    return None


class _Holder:
    """The lock, held by another lock owner, for the body of a `with`."""

    def __init__(self, item_path):
        self.item_path = item_path
        self.ready = threading.Event()
        self.release = threading.Event()
        self.thread = threading.Thread(target=_hold_lock_forever,
                                       args=(item_path, self.ready, self.release))

    def __enter__(self):
        self.thread.start()
        assert self.ready.wait(10), "the holder thread never took the lock"
        return self

    def __exit__(self, *exc):
        self.release.set()
        self.thread.join(10)
        return False


# ---------------------------------------------------------------- clause 1
# Two updates that overlap in time both land: B commits an activity_log entry
# while A's parse predates it, and afterwards the file has B's row AND A's key.

def test_an_aggregator_write_from_a_stale_parse_keeps_the_loops_activity_row(iso):
    """`backlog_write_task`'s lane holds a parse from before the loop committed.

    The interleaving is the one the item measures: `load_task` at
    `agent_mcp/backlog.py:257` and `save_task` at `:440` bracket the dedupe
    scoring, the ledger appends and the OKF stamping, so the aggregator's dict can
    be as stale as it likes. Before #3036 the aggregator's `write_text` rebuilt the
    front matter from that dict alone, and the loop's row — appended in the window —
    vanished from the file while the ledger still carried it.
    """
    path = iso["item"]
    # A: the aggregator parses the item. Its view of the journal is now stale.
    a_task = MCP.load_task(3036)
    assert a_task is not None and a_task.get("activity_log") is None
    # B: the loop commits a status move, journal row and all.
    assert B.update_frontmatter(path, {"status": "in_progress"},
                                activity=B_MSG) is True
    # A: the aggregator commits, from its stale dict.
    a_task["activity_log"] = [A_ROW]
    a_task["automod_landed"] = "deadbeef"
    assert MCP.save_task(a_task) is True

    fm = _fm_of(path)
    assert any(B_MSG in r for r in fm["activity_log"]), (
        "the loop's row was lost to the aggregator's stale dump")
    assert A_ROW in fm["activity_log"], "the aggregator's own row must still land"
    assert fm["automod_landed"] == "deadbeef"
    assert fm["status"] == "in_progress"
    assert "Not this route's to own." in path.read_text(encoding="utf-8")


def test_the_loop_rewrites_from_a_stale_parse_without_emptying_the_journal(iso):
    """The other direction, through the lane with the most call sites.

    `update_frontmatter` re-reads inside the lock, so a caller cannot hand it a
    stale parse — but `_write_item` and `note_item` are handed a parse taken
    outside the lock by their callers, and before #3036 a `_write_item` whose
    `fm` carried an older journal truncated every row added since.
    """
    path = iso["item"]
    B.update_frontmatter(path, {}, activity=B_MSG)
    # A stale `fm`: parsed before that row existed.
    stale_text = _HEAD.format(tid=3036, status="up_next", extra="")
    stale_fm, stale_body = B._split_frontmatter(stale_text)
    stale_fm["group"] = "autocode"
    B._write_item(path, stale_fm, stale_body)

    fm = _fm_of(path)
    assert len(fm["activity_log"]) == 1 and B_MSG in fm["activity_log"][0], (
        "a writer whose parse predates a journal row must not delete it")
    assert fm["group"] == "autocode"
    assert fm["status"] == "up_next"


def test_two_lanes_writing_one_item_concurrently_both_keep_every_row(iso):
    """The same test without a hand-built schedule: two real writers, one file.

    Each thread appends 20 rows through `note_item` — the store's own append path,
    in two threads of this process, which is the smallest honest stand-in for the
    two live processes named in the item (`agent_mcp.main` and `server.py`), and it
    is a real race: the read and the replace are separated by a YAML dump, so an
    unlocked store loses rows here. Unlocked, this test loses rows; that loss is
    what the clause forbids.
    """
    path = iso["item"]
    n = 20
    errors = []

    def writer(lane):
        try:
            for i in range(n):
                # Zero-padded marker on purpose: `lane A row 1` is a substring of
                # `lane A row 15`, and a marker that matches its neighbour's row is
                # an instrument that reports a loss where there is none.
                assert B.note_item(3036, f"lane-{lane}-row-{i:02d}") is True
        except Exception as exc:  # noqa: BLE001 — reported through `errors`
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(lane,)) for lane in "AB"]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert not errors, errors
    rows = _fm_of(path)["activity_log"]
    lost = [f"lane-{l}-row-{i:02d}" for l in "AB" for i in range(n)
            if sum(1 for r in rows if f"lane-{l}-row-{i:02d}" in r) != 1]
    assert not lost, (
        f"{2 * n} appends to one item left {len(rows)} rows; these markers are "
        f"missing or duplicated: {lost[:6]}")


#: The commit this round started from — `automod_start`'s `base`, which is the tree
#: the item measured and the tree the four clauses did not hold in. Named rather
#: than computed: inside the gate the tests run at the round's own HEAD, so `HEAD`
#: there is the tree that *has* the lock, and `HEAD~1` would only be the base while
#: the branch is one commit long.
PRE_CHANGE = "641cf23a"


def test_the_pre_change_tree_really_had_no_writer_lock():
    """The control for the module docstring: what was broken, read off the
    committed tree rather than asserted from memory.

    Three claims, all of them the item's, all of them about `PRE_CHANGE`: none of
    the three writer modules mentioned `commit_lock`; the two writers outside this
    store were the same bare one-liner; the automod lane had eight. If any of the
    three has since moved, the lost-update shape this file tests has moved with it,
    and this test says so instead of letting the rest of the file pass quietly.
    """
    def _show(rev_path):
        return subprocess.run(["git", "show", rev_path], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout

    loop = _show(f"{PRE_CHANGE}:scripts/automod/backlog.py")
    everything = (_show(f"{PRE_CHANGE}:agent_mcp/backlog.py")
                  + _show(f"{PRE_CHANGE}:app/routers/backlog.py") + loop)
    assert "commit_lock" not in everything, (
        f"{PRE_CHANGE} already locked these writers, so this round's premise is "
        "stale — re-triage #3036 rather than landing a duplicate")
    assert everything.count('write_text(f"---\\n{fm_yaml}---\\n\\n{body}")') == 2, (
        "the two writers outside this store were expected to be the same bare "
        "one-liner at the base; if that moved, the shape tested here has moved")
    assert loop.count(".write_text(") == 8, (
        "the automod lane was measured at eight bare item writes at the base")


# ---------------------------------------------------------------- clause 2
# Every backlog item writer holds a cross-process lock across its read-and-
# replace, and a writer that is renamed fails the guard by name.

#: Every function in the store that may replace an item file. A writer that is
#: renamed, deleted, or has its `with commit_lock(...)` block quietly removed is
#: named in the failure rather than vanishing from coverage — the shape
#: `tests/test_memory_writer_lane.py::_commit_lock_lanes` established for the
#: memory files, over this store's writers instead.
SITES = [
    ("agent_mcp/backlog.py", "save_task"),
    ("app/routers/backlog.py", "_write_task_file"),
    ("scripts/automod/backlog.py", "_write_item"),
    ("scripts/automod/backlog.py", "update_frontmatter"),
    ("scripts/automod/backlog.py", "_apply_status"),
    ("scripts/automod/backlog.py", "note_item"),
    ("scripts/automod/backlog.py", "tag_item"),
    ("scripts/automod/backlog.py", "reopen_reverted_landings"),
    ("scripts/automod/backlog.py", "close_landed"),
    ("scripts/automod/backlog.py", "record_verdict"),
]


def _func_source(rel_path: str, name: str):
    src = (ROOT / rel_path).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node, src
    pytest.fail(f"{rel_path} no longer defines {name}(): a backlog writer was "
                "renamed or removed — update SITES deliberately, never by accident")


def _locked_blocks(fn):
    """The `with commit_lock(...)` blocks in `fn`, including nested ones."""
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.With):
            for item in node.items:
                call = item.context_expr
                if (isinstance(call, ast.Call)
                        and (getattr(call.func, "id", None)
                             or getattr(call.func, "attr", None)) == "commit_lock"):
                    out.append(node)
    return out


@pytest.mark.parametrize("rel_path,func_name", SITES,
                         ids=[f"{p.split('/')[0]}:{n}" for p, n in SITES])
def test_the_backlog_writer_takes_the_commit_lock(rel_path, func_name):
    fn, _ = _func_source(rel_path, func_name)
    blocks = _locked_blocks(fn)
    assert blocks, (
        f"{rel_path}::{func_name} replaces an item file but opens no "
        "`with commit_lock(...)` block, so its read-modify-write is unlocked "
        "again — the lost update #3036 was filed for")


@pytest.mark.parametrize("rel_path,func_name", SITES,
                         ids=[f"{p.split('/')[0]}:{n}" for p, n in SITES])
def test_the_backlog_writer_replaces_only_inside_the_locked_block(rel_path, func_name):
    """The lock has to *cover* the replace: a `commit_lock` opened and released
    two statements before the write is a lock around nothing."""
    fn, src = _func_source(rel_path, func_name)
    blocks = _locked_blocks(fn)
    assert blocks, f"{rel_path}::{func_name} lost its `with commit_lock(...)`"
    writes = [n for n in ast.walk(fn)
              if isinstance(n, ast.Call)
              and ((isinstance(n.func, ast.Attribute)
                    and n.func.attr == "write_item")
                   or (isinstance(n.func, ast.Attribute)
                       and n.func.attr == "write_text"))]
    assert writes, f"{rel_path}::{func_name} no longer writes an item file at all"
    for call in writes:
        inside = any(id(call) in {id(x) for x in ast.walk(block)} for block in blocks)
        assert inside, (
            f"{rel_path}::{func_name} calls "
            f"`{ast.get_source_segment(src, call).splitlines()[0]}` outside its "
            "locked block, so a concurrent writer's row is still clobberable")


def test_no_item_file_writer_outside_the_store_rebuilds_the_file_itself():
    """The adoption pin, replacing the item's original acceptance line.

    The item asked that `git grep -n "flock\\|fcntl" -- agent_mcp/backlog.py` go
    non-empty. That target is wrong once the lock lives in the shared helper: an
    adopter contains no `flock` token, so the grep could be satisfied by a file
    that fixed nothing. What must actually be true is that the two writers outside
    this store stop hand-dumping YAML, so this asserts neither module contains a
    `write_text` call at all.
    """
    offenders = []
    for rel_path in ("agent_mcp/backlog.py", "app/routers/backlog.py"):
        for node in ast.walk(ast.parse((ROOT / rel_path).read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr in {"write_text", "write_bytes"}:
                offenders.append(f"{rel_path}:{node.lineno}")
    assert not offenders, (
        "these writers rebuild an item file by hand instead of going through "
        f"app.backlog_write.write_item: {offenders}")


def test_the_store_itself_has_no_unlocked_replacement_left():
    """Same pin for the lane with the most call sites. The item measured eight
    bare `write_text` calls in `scripts/automod/backlog.py` at the base commit;
    every one of them is now a locked `BW.write_item`."""
    src = (ROOT / "scripts/automod/backlog.py").read_text(encoding="utf-8")
    offenders = [f"scripts/automod/backlog.py:{n.lineno}"
                 for n in ast.walk(ast.parse(src))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr in {"write_text", "write_bytes"}]
    assert not offenders, (
        f"unlocked item writes are back in the automod lane: {offenders}")


# ---------------------------------------------------------------- clause 3
# A write that dies after the replacement bytes are produced leaves the item
# file byte-unchanged, so no reader can ever see a half-written block.

def test_a_write_that_dies_after_the_bytes_are_made_leaves_the_file_unchanged(iso,
                                                                              monkeypatch):
    """The clause's own forcing seam: make the front-matter dump raise mid-update.

    `yaml.dump` is called with the merged front matter and the body already in
    hand, so this is the failure that lands closest to the replace. The old writer
    lost nothing here either — its `write_text` also came after its dump — so what
    this test really pins is the *new* invariant around it: the read, the merge and
    the dump all happen while the caller holds the lock, and a raise at any of
    those points leaves the file byte-identical and the lock free.

    The half of the clause that a raise before the write cannot exercise — a death
    *after* the replacement bytes exist — is the next test, which forces the
    `os.replace` itself to fail.
    """
    path = iso["item"]
    before = path.read_bytes()

    def boom(*a, **k):
        raise RuntimeError("forced front-matter dump failure")

    monkeypatch.setattr(BW.yaml, "dump", boom)
    with pytest.raises(RuntimeError, match="forced front-matter dump failure"):
        B.update_frontmatter(path, {"priority": "high"}, activity="never written")
    assert path.read_bytes() == before, "a failed update changed the item file"
    assert "never written" not in path.read_text(encoding="utf-8")

    monkeypatch.undo()
    assert B.update_frontmatter(path, {"priority": "high"}) is True, (
        "the failed write left the commit lock held, so the next writer cannot in")
    assert _fm_of(path)["priority"] == "high"


def test_a_replace_that_fails_after_the_temp_is_written_leaves_the_file_unchanged(iso,
                                                                                 monkeypatch):
    """The death the clause is really about: the bytes exist, the rename does not
    complete. `os.replace` is the only step that makes new bytes visible to a
    reader, so a writer killed before it — or a temp whose content is invalid —
    cannot be observed at all. `atomic_write_text` writes the temp in
    `SCRATCH_DIR`, off the board, so a stranded temp is also not a file a reader
    of the board can open."""
    path = iso["item"]
    before = path.read_bytes()
    real_replace = os.replace

    def fail_replace(src, dst, *a, **k):
        raise OSError("forced rename failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="forced rename failure"):
        B.update_frontmatter(path, {"priority": "urgent"}, activity="never committed")
    monkeypatch.setattr(os, "replace", real_replace)

    assert path.read_bytes() == before, "a failed rename changed the item file"
    assert not list(iso["board"].glob("*.tmp")), "a temp was written inside the board"
    assert B.update_frontmatter(path, {"priority": "high"}) is True


def test_a_reader_never_observes_an_item_file_whose_front_matter_will_not_parse(iso):
    """The clause as a reader sees it: `load_task`, the board's own scan, and
    `open_items` all parse the file while some other lane is writing it.

    Two writers append 15 journal rows each through two different lanes while this
    test reads the file 40 times between them. Every read must be a complete file
    that parses: `Path.write_text` opens with `O_TRUNC`, so against the pre-change
    writer the same loop catches an empty or prefix-cut file — a prefix ending
    inside a quoted scalar, which the readers mark `_yaml_broken` and the route
    answers with a 409 on every later edit (#918).
    """
    path = iso["item"]
    errors = []
    stop = threading.Event()

    def writer(lane):
        try:
            for i in range(15):
                assert B.update_frontmatter(path, {"status": "in_progress"},
                                            activity=f"lane-{lane}-{i:02d}") is True
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def reader():
        try:
            for _ in range(40):
                if stop.is_set():
                    break
                text = path.read_text(encoding="utf-8")
                block = BW.FM.split_frontmatter(text)
                assert block is not None, "a read saw a file with no closing fence"
                yaml.safe_load(block[0])
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(lane,)) for lane in "AB"]
    threads.append(threading.Thread(target=reader))
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    stop.set()
    assert not errors, errors
    assert len(_fm_of(path)["activity_log"]) == 30


# ---------------------------------------------------------------- clause 4
# No lock or temp artefact is created inside the board directory.

def test_no_lock_or_temp_artefact_lands_inside_the_board_directory(iso):
    """The board is a tracked vault directory.

    `locked_file` — the other lock helper in `app.atomic_io` — puts its lock on a
    sibling `<name>.md.lock` *inside* the tree, and a stray there is a file in
    `git status`, in Obsidian's tree, and pushed by `agent-obsidian-sync`. The
    vault's `.gitignore` has no `*.tmp` rule either, which is the incident
    `tests/test_memory_writer_lane.py::
    test_a_crashed_writer_leaves_no_half_written_temp_inside_the_vault`
    recorded for the memory files. So the store takes its lock from
    `atomic_io.LOCK_DIR` and its temp from `atomic_io.SCRATCH_DIR`, both off-tree.

    The off-tree half is asserted in the same breath: a board directory that is
    empty of artefacts because the writer never locked anything would pass this
    test and fail every other clause in this file.
    """
    board, locks = iso["board"], iso["locks"]
    path = iso["item"]
    item_name = path.name

    a_task = MCP.load_task(3036)
    a_task["status"] = "in_progress"
    assert MCP.save_task(a_task) is True
    assert set(board.iterdir()) == {path}, (
        f"the MCP lane left artefacts in the board: "
        f"{[p.name for p in board.iterdir() if p != path]}")

    fm, body = BR._backlog_parse_fm(path)
    fm["priority"] = "high"
    BR._write_task_file(path, fm, body)
    assert set(board.iterdir()) == {path}, (
        "the board route left artefacts in the board")

    assert B.note_item(3036, "a note") is True
    assert set(board.iterdir()) == {path}, (
        f"the automod lane left artefacts in the board: "
        f"{[p.name for p in board.iterdir() if p != path]}")

    # The lock is real, and it is not in the board.
    expected = locks / (atomic_io.lock_file_for(path).name)
    assert expected.exists(), (
        "no lock file was created for the item, so the writers above were "
        f"serialized by nothing — looked for {expected}")
    assert not list(board.glob("*.lock")) and not list(board.glob("*.tmp")), (
        "a lock or temp file sits inside the tracked board directory")
    assert not list(locks.rglob("*.tmp")), "a writer's temp outlived its commit"
    assert path.name == item_name


# ---------------------------------------------------------------- clause 5
# A writer that cannot take the lock within the lane's wait fails loudly: it
# returns an error the board surfaces, and it leaves the file unchanged.

def test_the_aggregator_lane_reports_a_lock_it_cannot_take(installed_wait, iso, caplog):
    """`backlog_write_task` answers with an error the model can act on.

    `save_task` already had a False-for-two-reasons shape (`_yaml_broken` and a
    refused save), and the tool handler turned both into
    `"Failed to save task"` — a caller could not tell a file it must fix by hand
    from a lane that is momentarily wedged. The reason is in the message now, and
    the refusal is logged, because the caller of an MCP tool cannot see the
    aggregator's stderr.
    """
    path = iso["item"]
    before = path.read_bytes()
    a_task = MCP.load_task(3036)
    a_task["priority"] = "high"
    with _Holder(path), caplog.at_level(logging.WARNING, logger="agent_mcp.backlog"):
        assert MCP.save_task(a_task) is False
    assert path.read_bytes() == before, "a refused write changed the item file"
    warnings = [r.getMessage() for r in caplog.records if "left unchanged" in r.getMessage()]
    assert warnings, (
        "save_task returned False without saying why: the caller cannot tell a "
        "wedged lock from the `_yaml_broken` refusal, and neither can the log")
    assert path.name in warnings[0], warnings[0]


def test_the_board_route_answers_503_rather_than_hanging(installed_wait, iso):
    """A wedged automod finalizer must not stall a board request indefinitely.

    Before #3036 there was no wait at all and no error either: the save simply
    overwrote whatever the finalizer had just written. The board's answer now is a
    503 whose detail names the item and tells the person to retry, with the file
    exactly as it was.
    """
    path = iso["item"]
    before = path.read_bytes()
    fm, body = BR._backlog_parse_fm(path)
    fm["priority"] = "urgent"
    with _Holder(path):
        started = time.monotonic()
        with pytest.raises(HTTPException) as caught:
            BR._write_task_file(path, fm, body)
        waited = time.monotonic() - started
    assert caught.value.status_code == 503
    assert path.name in caught.value.detail
    assert "retry" in caught.value.detail.lower()
    assert waited < 5.0, (
        f"the route waited {waited:.1f}s for a lock its caller cannot see; the "
        "lane's wait is the bound a board request is allowed to take")
    assert path.read_bytes() == before, "a refused board save changed the item file"


def test_the_automod_lane_raises_rather_than_skipping_a_status_move(installed_wait, iso):
    """The loop's lane is the one where a silent skip is the worst answer.

    `backlog_write_task` and the board both have a human or a model on the other
    end of the failure. `_apply_status` does not: it runs in the reconciler, whose
    caller already wrote the move into the promotions ledger. Returning False here
    would reproduce the exact board-versus-ledger disagreement this item was filed
    for, so the lock failure propagates as an exception and the caller's own
    bookkeeping stops with it.
    """
    path = iso["item"]
    before = path.read_bytes()
    with _Holder(path):
        with pytest.raises(TimeoutError) as caught:
            B._apply_status(path, "done", "reconciler move that must not be skipped")
    assert "could not lock" in str(caught.value)
    assert path.read_bytes() == before, "a refused status move changed the item file"


@pytest.fixture
def installed_wait(monkeypatch, caplog):
    """Shorten the lane's lock wait so the timeout paths are testable in seconds.

    `commit_lock(timeout=None)` resolves `DEFAULT_LOCK_WAIT` at call time rather
    than binding it in a signature — deliberately, so this patch is the seam every
    lane actually uses. A writer that hardcoded `timeout=30.0` would wait half a
    minute here and this fixture's bound assertion would catch it.
    """
    monkeypatch.setattr(atomic_io, "DEFAULT_LOCK_WAIT", 0.4)
    yield {"wait": 0.4}


def test_the_owed_check_lane_rebuilt_item_files_by_hand_too():
    """The fourth lane, which the item never named.

    `owed.close_item` closed an item with a fifth bare
    `f"---\\n{yaml.dump(fm, ...)}---\\n{body}"` `write_text` — the same one-liner the
    item quotes for the other two writers, in a module the item's site list does not
    mention. The owed-check job runs on its own schedule while the board and the
    autocode source write the same files, so leaving it out would have fixed three
    lanes and left the fourth able to lose a row exactly as before.

    Asserted by AST rather than by grep so a `write_text` in a docstring or a
    comment — both of which the module has, describing the change — cannot satisfy
    it, and the module object is checked against the path scanned so the pin cannot
    quietly grade a file some other import did not load.
    """
    rel_path = "scripts/automod/owed.py"
    assert Path(O.__file__).name == Path(rel_path).name, (
        f"{rel_path} is not the file the running `{O.__name__}` came from, so "
        "scanning it would grade the wrong tree")
    offenders = [f"{rel_path}:{n.lineno}"
                 for n in ast.walk(ast.parse((ROOT / rel_path).read_text(encoding="utf-8")))
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr in {"write_text", "write_bytes"}]
    assert not offenders, (
        f"the owed-check lane is writing item files by hand again: {offenders}")
    assert O.close_item.__doc__ is None or "write_text" not in O.close_item.__doc__ \
        or "bare `write_text`" in O.close_item.__doc__, (
        "the module's own comment about the bare write no longer matches what the "
        "function does")
