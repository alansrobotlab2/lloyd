"""One ordered commit path per shared memory file: lock, read, atomic replace.

The failure this pins is the one that leaves no trace. Two callers load the same
`lloyd/MEMORY.md`, each appends its own entry, each saves the whole file — and
the second save silently erases the first. Neither writer is malformed; the
missing piece is serialization around the commit. `lloyd/MEMORY.md` and
`lloyd/USER.md` are the worst possible place for it, because they are loaded
into every system prompt: a lost update is experienced as Lloyd being forgetful,
and for the nightly-rewritten paths it is not recoverable from git either.

The same hazard has a second, worse shape that needs no second writer at all.
`Path.write_text` opens with `O_TRUNC`, so a `memory_read` landing between the
truncate and the write hands the model an empty MEMORY.md. Measured at base:
2 empty reads and 2,769 size regressions out of 3,668 concurrent reads while
four writers appended. An atomic replace (`os.replace`) closes that for every
reader, whether or not the reader's writer cooperates with the lock — `flock`
is advisory and only excludes writers that take it.

Six sites own these files, and the triage that opened the item found the lock
must be shared across all six: `agent_mcp/session.py` `_memory_add` /
`_memory_replace` / `_memory_remove`, `agent_mcp/builtin_fs.py` `_write` /
`_edit` (the dominant writer of `lloyd/USER.md` is the nightly knowledge-write
job, which is told to use `Edit` and never touches the `memory_*` tools — so a
lock in the memory tools alone serialises nothing against it), and
`agent_mcp/vault.py` `_vault_write`. They all resolve to the same realpath, so
they must take the same lock: `lock_file_for` keys it on realpath, which is why
the test below asserts the three lanes agree.

Scope of the guarantee, stated rather than assumed: every writer that goes
through the MCP tool layer. A raw script writing the same file, or Obsidian
Sync applying the same note from the other side of the tailnet, is outside what
`flock` can exclude (see the item's follow-up question).
"""

from __future__ import annotations

import ast
import errno
import json
import multiprocessing as mp
import os
import random
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import atomic_io
from app.paths import LLOYD_HOME
from agent_mcp import builtin_fs as FS
from agent_mcp import session as SESSION
from agent_mcp import vault as VAULT

# Read defensively so this module still imports, and every test still RUNS, at
# the commit where it is introduced. A collection error would look like "the
# suite is green" and hide the very losses these tests exist to report.
LOCK_DIR = getattr(atomic_io, "LOCK_DIR", None)
lock_file_for = getattr(atomic_io, "lock_file_for", None)

ENTRY_CHARS = 900          # ≥ 800, per the item: a small file hides the loss
WRITERS = 4
PER_WRITER = 25
BASE_BYTES = 20_000        # the measured shape: ~20 KB, not a toy file
HEADER = "# Writer-lane scratch memory\n"
GO_POLL = 0.002


def _ctx():
    return mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context("spawn")


def _go(go_path: Path) -> None:
    """Block until the parent raises the flag, so the writers really overlap."""
    while not go_path.exists():
        time.sleep(GO_POLL)


def _entry(sentinel: str) -> str:
    return f"{sentinel} " + ("filler text that stands for a real memory entry. " * 30)


def _base_file(path: Path) -> Path:
    lines = [HEADER]
    while len("".join(lines).encode("utf-8")) < BASE_BYTES:
        lines.append(f"- standing entry {len(lines)}: " + "x" * 120 + "\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _spawn_appenders(n_writers: int, per_writer: int, root: Path, go: Path, tag: str):
    procs = []
    for w in range(n_writers):
        p = _ctx().Process(target=_append_worker,
                           args=(root, per_writer, go, f"{tag}-w{w}", root / f"{tag}-w{w}.report"))
        p.start()
        procs.append(p)
    return procs


def _report(path: Path, errors: list[str]) -> None:
    """A worker that reports is evidence; a worker that raises is exit code 1.

    At base the loser of the race does not crash quietly — `Edit` answers
    `old_string not found`, which is the loss talking. Swallowing that into an
    exception would report "a lane died" and hide which side lost what.
    """
    path.write_text(json.dumps({"errors": errors}), encoding="utf-8")


def _reports(root: Path, tag_prefix: str) -> list[str]:
    errors: list[str] = []
    for rep in sorted(root.glob(f"{tag_prefix}*.report")):
        assert rep.stat().st_size, f"{rep.name} was never written — the worker died mid-run"
        errors += json.loads(rep.read_text(encoding="utf-8"))["errors"]
    return errors


def _append_worker(root: Path, per_writer: int, go: Path, tag: str, report: Path) -> None:
    SESSION.MEMORIES_ROOT = Path(root)
    errors: list[str] = []
    _go(go)
    for i in range(per_writer):
        try:
            res = SESSION._memory_add({"file": "MEMORY.md", "entry": _entry(f"{tag}-s{i:03d}")})
            if not res.get("success"):
                errors.append(f"{tag}-s{i:03d}: {res}")
        except Exception as exc:
            errors.append(f"{tag}-s{i:03d}: raised {type(exc).__name__}: {exc}")
    _report(report, errors)


def _collect_sentinels(path: Path, tag: str, expected: int) -> tuple[int, list[str]]:
    text = path.read_text(encoding="utf-8")
    missing = [f"{tag}-w{w}-s{i:03d}" for w in range(WRITERS) for i in range(expected // WRITERS)
               if f"{tag}-w{w}-s{i:03d}" not in text]
    return expected - len(missing), missing


# ── clause 1: four concurrent memory_add writers lose nothing ────────────────

def test_four_concurrent_memory_writers_keep_all_one_hundred_sentinels(tmp_path):
    """4 processes x 25 entries of >=800 chars into a ~20 KB MEMORY.md.

    Calibrated against the triage measurement, not against a guess: two
    writers on a small file does NOT fail (100/100 survived at base), which is
    why the reproduction is 4 writers on a realistic-size file — there it lost
    90-95 % of appends (19/200 and 9/200 survivors).
    """
    root = tmp_path / "memories"
    root.mkdir()
    target = _base_file(root / "MEMORY.md")
    start_size = target.stat().st_size
    assert start_size >= BASE_BYTES, "a scratch file too small to lose anything makes this vacuous"

    go = tmp_path / "go"
    procs = _spawn_appenders(WRITERS, PER_WRITER, root, go, "mu")
    time.sleep(0.05)
    go.touch()
    for p in procs:
        p.join(120)
        assert not p.is_alive(), "a writer hung instead of finishing"
        assert p.exitcode == 0, f"a writer died (exitcode {p.exitcode}) rather than contending"

    kept, missing = _collect_sentinels(target, "mu", WRITERS * PER_WRITER)
    assert kept == WRITERS * PER_WRITER, (
        f"{kept}/{WRITERS * PER_WRITER} sentinels survived four concurrent writers; "
        f"last writer won and erased {len(missing)} appends. First missing: {missing[:3]}"
    )


# ── clause 2: the memory lane and the Write/Edit lane on one path ────────────

def _editing_worker(root: Path, n: int, go: Path, tag: str, report: Path) -> None:
    """Drive builtin_fs._edit — the lane the nightly knowledge-write job uses."""
    errors: list[str] = []
    _go(go)
    path = str(root / "MEMORY.md")
    for i in range(n):
        try:
            res = FS._edit({"file_path": path,
                            "old_string": f"PLACEHOLDER-{tag}-{i:03d}",
                            "new_string": f"EDITED-{tag}-{i:03d}"})
            if res.startswith("{") and "error" in json.loads(res):
                errors.append(f"{tag}-{i:03d}: {res}")
        except Exception as exc:
            errors.append(f"{tag}-{i:03d}: raised {type(exc).__name__}: {exc}")
    _report(report, errors)


def test_memory_add_and_builtin_edit_on_one_path_lose_neither_side(tmp_path):
    """A memory_add issued while Edit writes the same path must not clobber it.

    Both halves have to land: every appended sentinel present, and every edit
    applied. Losing the appends is the lost update; losing the edits is what a
    whole-file memory_add write does to an editor that read before the append.
    """
    root = tmp_path / "memories"
    root.mkdir()
    lines = [HEADER]
    for lane in ("mem", "fs"):
        lines += [f"PLACEHOLDER-{lane}-{i:03d}\n" for i in range(PER_WRITER)]
    while len("".join(lines).encode("utf-8")) < BASE_BYTES:
        lines.append("- filler entry " + "y" * 150 + "\n")
    target = root / "MEMORY.md"
    target.write_text("".join(lines), encoding="utf-8")

    go = tmp_path / "go"
    editor = _ctx().Process(target=_editing_worker,
                            args=(root, PER_WRITER, go, "fs", root / "fs-edit.report"))
    appender = _ctx().Process(target=_append_worker,
                              args=(root, PER_WRITER, go, "mu-w0", root / "mu-add.report"))
    editor.start()
    appender.start()
    time.sleep(0.05)
    go.touch()
    for p in (editor, appender):
        p.join(120)
        assert not p.is_alive(), "a lane hung instead of finishing"
        assert p.exitcode == 0, f"a lane died (exitcode {p.exitcode}) instead of contending"

    text = target.read_text(encoding="utf-8")
    lost_appends = [i for i in range(PER_WRITER) if f"mu-w0-s{i:03d}" not in text]
    lost_edits = [i for i in range(PER_WRITER) if f"EDITED-fs-{i:03d}" not in text]
    assert not _reports(root, "mu-add"), _reports(root, "mu-add")
    assert not _reports(root, "fs-edit"), _reports(root, "fs-edit")
    assert not lost_appends, (
        f"Edit clobbered {len(lost_appends)}/{PER_WRITER} memory_add appends "
        f"(first: mu-w0-s{lost_appends[0]:03d})"
    )
    assert not lost_edits, (
        f"memory_add clobbered {len(lost_edits)}/{PER_WRITER} Edits "
        f"(first: PLACEHOLDER-fs-{lost_edits[0]:03d} still present)"
    )


# ── clause 3a: a reader never sees an empty or torn file ────────────────────

def test_a_reader_never_sees_an_empty_or_truncated_memory_file(tmp_path):
    """Concurrent `memory_read` while four writers append: every read is whole.

    `Path.write_text` opens O_TRUNC, so at base a read in the window returns an
    empty string, and every such read is a prompt that ships with no memory at
    all. Base measurement: 2 empty reads and 2,769 size regressions in 3,668
    reads. Entries only ever append here, so the on-disk size is monotonic and
    a regression is proof of a torn read.
    """
    root = tmp_path / "memories"
    root.mkdir()
    target = _base_file(root / "MEMORY.md")
    go = tmp_path / "go"
    stop = threading.Event()
    bad: list[str] = []
    reads = 0
    last_size = 0

    def reader():
        nonlocal reads, last_size
        _go(go)
        while not stop.is_set():
            try:
                text = target.read_text(encoding="utf-8")
            except Exception as exc:  # a read that raised is already a finding
                bad.append(f"raised {type(exc).__name__}: {exc}")
                continue
            reads += 1
            if not text:
                bad.append("empty")
                continue
            if not text.startswith(HEADER):
                bad.append("missing the header the file was created with")
            if not text.endswith("\n"):
                bad.append("truncated mid-line")
            size = len(text.encode("utf-8"))
            if size < last_size:
                bad.append(f"shrank from {last_size} to {size} bytes")
            last_size = max(last_size, size)

    t = threading.Thread(target=reader)
    t.start()
    procs = _spawn_appenders(WRITERS, PER_WRITER, root, go, "rd")
    go.touch()
    for p in procs:
        p.join(120)
    stop.set()
    t.join(30)

    assert reads >= 50, f"the reader only landed {reads} reads — the test proved nothing"
    assert not bad, (
        f"{len(bad)} of {reads} concurrent reads were torn, e.g. {bad[:3]}"
    )


# ── clause 3b: a writer killed mid-write leaves the previous bytes verbatim ──

def _appending_until_killed(root: Path, tag: str) -> None:
    SESSION.MEMORIES_ROOT = Path(root)
    i = 0
    while True:
        SESSION._memory_add({"file": "MEMORY.md", "entry": _entry(f"{tag}-s{i:04d}")})
        i += 1


def test_a_writer_killed_mid_write_leaves_the_file_byte_whole(tmp_path):
    """SIGKILL the writer at random moments: the file is always a whole prefix.

    "Whole prefix" is the real bar: the file holds the header plus an integral
    number of complete entries. A torn write leaves a half-entry or no bytes at
    all, and on MEMORY.md the next prompt is assembled from that remnant. The
    kill also releases the flock with the descriptor, so a later write must
    still succeed — a lock that wedges on a dead writer is worse than the race.
    """
    root = tmp_path / "memories"
    root.mkdir()
    _base_file(root / "MEMORY.md")
    target = root / "MEMORY.md"
    prefix = "".join(l for l in HEADER)  # header only ever present, complete

    ctx = _ctx()
    for attempt in range(5):
        child = ctx.Process(target=_appending_until_killed, args=(root, "kill"))
        child.start()
        time.sleep(random.uniform(0.05, 0.35))
        os.kill(child.pid, signal.SIGKILL)
        child.join(30)

        text = target.read_text(encoding="utf-8")
        assert text, f"attempt {attempt}: a killed writer left MEMORY.md empty"
        assert text.startswith(prefix) and text.endswith("\n"), (
            f"attempt {attempt}: a killed writer left a torn file "
            f"({len(text)} bytes, trailing {text[-40:]!r})"
        )
        stray = [l for l in text.splitlines()[1:] if not l.startswith(("- ", "kill-s"))]
        assert not stray, f"attempt {attempt}: half-written entry in the file: {stray[:1]}"

        # the lock died with the process: the next writer must get through
        again = SESSION._memory_add({"file": "MEMORY.md", "entry": _entry(f"after-kill-{attempt}")})
        assert again.get("success"), f"attempt {attempt}: the write lock wedged after a kill: {again}"


# ── clause 4: no .lock beside the target, and not inside the vault ───────────

def _scratch_vault(tmp_path, monkeypatch):
    """Point the vault lane and the memory lane at one scratch tree."""
    vault = tmp_path / "vault"
    (vault / "lloyd").mkdir(parents=True)
    (vault / "memory" / "audit").mkdir(parents=True)
    monkeypatch.setattr(VAULT, "VAULT", vault)
    monkeypatch.setattr(VAULT, "AUDIT_LOG_DIR", vault / "memory" / "audit")
    monkeypatch.setattr(VAULT, "AUDIT_LOG_FILE", vault / "memory" / "audit" / "writes.jsonl")
    monkeypatch.setattr(SESSION, "MEMORIES_ROOT", vault / "lloyd")
    # `scratch_dir_for` asks whether the target is inside the *real* vault, so a
    # test that only moves `vault.VAULT` would exercise the sibling-temp branch
    # and report "no temp in the vault" for the wrong reason.
    monkeypatch.setattr("app.paths.VAULT_ROOT", vault)
    monkeypatch.setattr(atomic_io, "SCRATCH_DIR", tmp_path / "locks" / "tmp")
    return vault


def test_no_lock_artefact_appears_beside_a_target_or_under_the_vault(tmp_path, monkeypatch):
    """Lock siblings used to land in the vault; `lloyd/.research-queue.lock` is
    the precedent and it is tracked in git. Locks live outside the tree."""
    vault = _scratch_vault(tmp_path, monkeypatch)
    assert LOCK_DIR is not None, "atomic_io must expose the out-of-tree lock directory"
    assert lock_file_for is not None, "atomic_io must expose the lock key for a path"

    mem = vault / "lloyd" / "MEMORY.md"
    mem.write_text(HEADER, encoding="utf-8")
    assert SESSION._memory_add({"file": "MEMORY.md", "entry": "an entry"})["success"]
    assert SESSION._memory_replace({"file": "MEMORY.md", "old_text": "an entry",
                                    "new_text": "an edited entry"})["success"]
    assert SESSION._memory_remove({"file": "MEMORY.md", "entry": "an edited entry"})["success"]

    other = vault / "knowledge" / "note.md"
    other.parent.mkdir(parents=True, exist_ok=True)
    assert "File written" in FS._write({"file_path": str(other), "content": "# Note\n"})
    assert "Edited" in FS._edit({"file_path": str(other), "old_string": "# Note",
                                 "new_string": "# Edited note"})

    res = VAULT._vault_write({"path": "knowledge/other2.md", "content": "# Two\n"})
    assert res.get("success"), res

    inside = sorted(str(p.relative_to(vault)) for p in vault.rglob("*.lock"))
    assert not inside, f"lock artefacts landed inside the vault tree: {inside}"
    beside = sorted(p.name for p in mem.parent.glob("*.lock"))
    assert not beside, f"lock artefacts landed beside the memory files: {beside}"

    resolved = Path(LOCK_DIR).resolve()
    assert resolved.is_relative_to(LLOYD_HOME.resolve()), (
        f"the lock directory {resolved} does not resolve under {LLOYD_HOME}"
    )
    assert not list(vault.rglob("*.tmp")), (
        f"a half-written temp was left inside the vault: {list(vault.rglob('*.tmp'))}"
    )


def test_a_crashed_writer_leaves_no_half_written_temp_inside_the_vault(tmp_path, monkeypatch):
    """Atomic replace needs a temp, and a temp is a file until the rename.

    `atomic_write_text` wrote `<name>.<pid>.tmp` beside the target, so a writer
    killed between the write and `os.replace` left it inside `~/obsidian`. That
    is the same objection the item raises about lock siblings, and it is worse
    here: the vault's `.gitignore` has no `*.tmp` rule, so a pre-flight commit
    sweeps the stray into the tracked tree, `agent-obsidian-sync` pushes whatever
    it finds, and Obsidian indexes it. Inside the vault the temp goes off-tree.
    """
    vault = _scratch_vault(tmp_path, monkeypatch)
    scratch = tmp_path / "locks" / "tmp"
    monkeypatch.setattr("app.paths.VAULT_ROOT", vault)
    monkeypatch.setattr(atomic_io, "SCRATCH_DIR", scratch)

    target = vault / "lloyd" / "USER.md"
    prior = "the version that was there before\n"
    target.write_text(prior, encoding="utf-8")

    assert atomic_io.scratch_dir_for(target) == scratch, (
        "a target inside the vault must not get a sibling temp"
    )
    assert atomic_io.scratch_dir_for(tmp_path / "outside.md") is None, (
        "a target outside the vault keeps its sibling temp — nothing there "
        "keeps a clean-tree invariant the stray would break"
    )

    # The crash is the gap between the temp write and the rename. Interrupting it
    # deterministically beats racing a SIGKILL against a multi-megabyte write: the
    # filesystem state on the other side of both is identical, and only one of
    # them has a flake rate. The scratch dir is the same one a killed writer would
    # have used, so the stray left here is the stray left there.
    real_replace = os.replace

    def _crash(src, dst, *a, **k):
        if str(src).endswith(".tmp") and str(src).startswith(str(scratch)):
            raise RuntimeError("simulated crash between the temp write and the rename")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(os, "replace", _crash)
    with pytest.raises(RuntimeError):
        atomic_io.write_text_durable(target, "u" * (2 * 1024 * 1024))

    assert target.read_text(encoding="utf-8") == prior, (
        "the interrupted write replaced the file instead of leaving it whole"
    )
    assert not list(vault.rglob("*.tmp")), (
        f"half-written temps left inside the vault: {list(vault.rglob('*.tmp'))}"
    )
    assert list(scratch.glob("*.tmp")), (
        "the stray belongs in the scratch dir — that is the trade: off-tree a "
        "stray is inert, inside the vault it gets committed, synced and indexed"
    )


def test_a_write_that_crosses_filesystems_falls_back_to_a_sibling_temp(tmp_path, monkeypatch):
    """The off-tree temp is a preference, not a requirement.

    `os.replace` cannot rename across filesystems (EXDEV). If the scratch dir and
    the target ever part company — a second mount, a container bind, a
    `LLOYD_LOCK_DIR` pointing elsewhere — the writer must still commit, and still
    atomically, rather than fail every vault write.
    """
    real_replace = os.replace
    seen: list[str] = []

    def _exdev(src, dst, *a, **k):
        seen.append(str(src))
        if str(src).startswith(str(tmp_path / "locks")):
            raise OSError(errno.EXDEV, "cross-device link not permitted", str(src))
        return real_replace(src, dst, *a, **k)

    vault = _scratch_vault(tmp_path, monkeypatch)
    monkeypatch.setattr("app.paths.VAULT_ROOT", vault)
    monkeypatch.setattr(atomic_io, "SCRATCH_DIR", tmp_path / "locks" / "tmp")
    monkeypatch.setattr(os, "replace", _exdev)

    target = vault / "lloyd" / "MEMORY.md"
    target.write_text("first\n", encoding="utf-8")
    atomic_io.write_text_durable(target, "second\n")

    assert target.read_text(encoding="utf-8") == "second\n", (
        "EXDEV should fall back to a sibling temp, not lose the write"
    )
    assert seen[0].startswith(str(tmp_path / "locks")), (
        f"the first attempt should have used the scratch dir: {seen}"
    )
    assert not Path(seen[-1]).is_relative_to(tmp_path / "locks"), (
        f"the retry temp was not a sibling of the target: {seen}"
    )
    assert not list(vault.rglob("*.tmp")), "the fallback temp was left behind"


def test_the_three_lanes_take_the_same_lock_for_one_file(tmp_path, monkeypatch):
    """Shared-state means shared lock: the same note seen three ways is one file.

    `memory_add` spells the path `MEMORIES_ROOT / name`, `vault_write` spells it
    vault-relative, `Write`/`Edit` get an absolute path from the model. Three
    spellings of `lloyd/MEMORY.md` must not pick three locks — that is the whole
    bug with a different name.
    """
    vault = _scratch_vault(tmp_path, monkeypatch)
    assert lock_file_for is not None, "atomic_io must expose the lock key for a path"
    mem = SESSION.MEMORIES_ROOT / "MEMORY.md"
    key_memory = lock_file_for(mem)
    key_vault = lock_file_for(vault / "lloyd" / "MEMORY.md")
    key_fs = lock_file_for(str(vault / "lloyd" / "./MEMORY.md"))
    assert key_memory == key_vault == key_fs, (
        f"the memory, vault and Write/Edit lanes picked different locks for one file: "
        f"{key_memory.name} vs {key_vault.name} vs {key_fs.name}"
    )


# ── clause 5: no bare write_text survives in the six commit sites ────────────

SITES = [
    ("agent_mcp/session.py", "_memory_add"),
    ("agent_mcp/session.py", "_memory_replace"),
    ("agent_mcp/session.py", "_memory_remove"),
    ("agent_mcp/builtin_fs.py", "_write"),
    ("agent_mcp/builtin_fs.py", "_edit"),
    ("agent_mcp/vault.py", "_vault_write"),
]


def _func_source(rel: str, name: str) -> ast.FunctionDef:
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} is gone from {rel}; this guard must be moved, not dropped")


@pytest.mark.parametrize("rel,name", SITES)
def test_commit_site_writes_atomically_under_the_shared_lock(rel, name):
    """One ordered commit path per state boundary, checked by parsing it.

    Grepping a line range rots (every line reference in the originating item
    was two months stale), so this parses the function and looks at the calls
    inside it. `write_text` is what `O_TRUNC` looks like from here.
    """
    fn = _func_source(rel, name)
    bare = [f"{rel}:{n.lineno} {name}()" for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "write_text"]
    assert not bare, f"bare truncate-in-place write, which loses updates and tears reads: {bare}"

    calls = {
        (n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", ""))
        for n in ast.walk(fn) if isinstance(n, ast.Call)
    }
    # `write_text_durable` is this lane's own wrapper over `atomic_write_text`,
    # so accepting it is only honest because the wrapper is pinned below.
    assert {"atomic_write_text", "write_text_durable"} & calls, (
        f"{rel} {name} must replace via write_text_durable/atomic_write_text, "
        f"not in place (found: {sorted(calls)})"
    )
    assert "commit_lock" in calls, f"{rel} {name} must commit under commit_lock()"
    withs = [w for w in ast.walk(fn) if isinstance(w, ast.With)]
    lanes = [w for w in withs
             if isinstance(w.items[0].context_expr, ast.Call)
             and getattr(w.items[0].context_expr.func, "id", "") == "commit_lock"]
    assert lanes, f"{rel} {name} does not hold the lock across its read and replace"
    _assert_reads_inside_the_lock(fn, lanes, rel, name)


def _lock_span(lanes: list[ast.With]) -> tuple[int, int]:
    lo = min(w.items[0].context_expr.lineno for w in lanes)
    hi = max(max(getattr(n, "lineno", 0) for n in ast.walk(w)) for w in lanes)
    return lo, hi


def _assert_reads_inside_the_lock(fn, lanes, rel, name) -> None:
    """The read has to happen inside the lock, not before it.

    `scripts/memory/repair_fact_ids.py:66` records exactly this corollary, and it
    is the half that a reviewer skims: lock the write and a lost update becomes a
    slightly later lost update, because both writers still read the same stale
    bytes. Parsing the line range is how that gets caught instead of agreed to.
    """
    lo, hi = _lock_span(lanes)
    outside = []
    for n in ast.walk(fn):
        if not isinstance(n, ast.Call):
            continue
        attr = getattr(n.func, "attr", "")
        if attr in {"read_text", "read_bytes"} and not (lo <= n.lineno <= hi):
            outside.append(f"{rel}:{n.lineno} {attr}()")
    assert not outside, (
        f"{rel} {name} reads the file outside commit_lock — the read is half the "
        f"critical section: {outside}"
    )


def test_the_durable_writer_is_the_atomic_one():
    """`write_text_durable` earns its place in the six commit sites.

    The guard above accepts it as the atomic replace. If it ever stopped
    delegating — a well-meant simplification back to `Path.write_text` — all six
    sites would silently go back to truncating in place, and every concurrency
    test in this file would still pass, because they exercise the lane through the
    wrapper rather than inspecting it.
    """
    tree = ast.parse((ROOT / "app/atomic_io.py").read_text(encoding="utf-8"))
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    inner = {
        (n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", ""))
        for n in ast.walk(funcs["write_text_durable"]) if isinstance(n, ast.Call)
    }
    assert "atomic_write_text" in inner, sorted(inner)
    assert "write_text" not in inner, sorted(inner)

    # And the thing it delegates to is still rename-based, still off-tree-capable:
    # `open(..., "w")` is O_TRUNC, so a reader between the truncate and the last
    # byte sees a shorter file or an empty one. `os.replace` is what makes a
    # reader see whole bytes on both sides.
    atomic = funcs["atomic_write_text"]
    attrs = {getattr(n.func, "attr", "") for n in ast.walk(atomic) if isinstance(n, ast.Call)}
    assert "replace" in attrs, f"atomic_write_text no longer renames: {sorted(attrs)}"
    assert "write_text" not in attrs, f"atomic_write_text grew a write_text: {sorted(attrs)}"
    # `locked_file`'s sibling lock is still the fact-file path; the memory lane
    # must not quietly become a second user of it and re-pollute the vault. The
    # durable wrapper is also what routes the temp off-tree — it decides by
    # asking whether the target is inside the vault, so the call has to name the
    # path, not a constant.
    wrapper = ast.unparse(funcs["write_text_durable"])
    assert "scratch_dir_for(path)" in wrapper, wrapper
    assert "tmp_dir=" in wrapper, wrapper
