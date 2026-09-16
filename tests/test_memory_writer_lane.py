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
truncate and the write hands the model an empty MEMORY.md. Re-measured at base
on the sizes this file uses — five trials, four writers appending while one
reader loops: 77 to 119 empty reads per trial out of 384 to 542 reads, plus 161
to 370 reads that came back shorter than the one before (an appending file can
only shrink if the read was torn). An atomic replace (`os.replace`) closes that
for every reader, whether or not the reader's writer cooperates with the lock —
`flock` is advisory and only excludes writers that take it.

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
import asyncio
import errno
import fcntl
import json
import multiprocessing as mp
import os
import random
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import atomic_io
from app.paths import LLOYD_HOME, LOGS_DIR
from agent_mcp import _shared as SHARED
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
RACE_BYTES = 200_000       # the same-file race needs milliseconds per commit
HEADER = "# Writer-lane scratch memory\n"
GO_POLL = 0.002


# ── isolation: the pytest parent must not resolve a real root either ─────────
#
# Everything below forks or spawns children, and every child is pinned to a
# scratch tree by the arguments or the environment it is handed. That covers the
# children and leaves the process running this file completely unprotected:
# `agent_mcp.session` binds `MEMORIES_ROOT = Path.home() / "obsidian" / "lloyd"`
# at import, and the tests that call a writer *in the parent* — the
# post-SIGKILL `_memory_add`, the three-lane lock test, the timeout seam —
# therefore write the real MEMORY.md. It is not hypothetical: the post-kill
# `SESSION._memory_add` in `test_a_writer_killed_mid_write_...` appended five
# entries to `~/obsidian/lloyd/MEMORY.md` and the test still passed, because
# every one of its assertions looks at the scratch copy. Those entries are
# loaded into every system prompt, so a test that pollutes them is writing into
# the thing this item is about, in the one place a lost update is experienced as
# Lloyd being wrong.
#
# The trap that makes this easy to get wrong: patching `app.paths` does nothing
# here. `session.MEMORIES_ROOT` is a module-level `Path.home()` literal that
# never consults `app.paths`, and `vault.VAULT` is rebound from
# `agent_mcp._shared`, which took `VAULT_ROOT` from `app.paths` once, at import.
# Each copy has to be named.

LIVE_VAULT = Path.home() / "obsidian"
LIVE_MEMORY_FILES = [LIVE_VAULT / "lloyd" / n for n in ("MEMORY.md", "USER.md")]


def _real(p) -> Path:
    return Path(os.path.realpath(str(p)))


def _assert_off_tree(p, *, what: str) -> Path:
    """Fail now if a scratch root resolves somewhere that is real.

    Checked on the way out of the fixture and again on the way into every spawner
    helper, because the failure this guards is silent in a specific way: a child
    that resolves a live root appends its sentinels into the live file, while the
    test that launched it reads back only the scratch copy it made and reports
    success. An assertion the parent can check is the only signal there is.
    """
    resolved = _real(p)
    logs = _real(LOGS_DIR)
    for label, forbidden in (("the live vault", _real(LIVE_VAULT)),
                             ("the checkout this suite runs from", _real(LLOYD_HOME)),
                             ("the checkout's logs dir (where LOCK_DIR defaults)", logs)):
        assert not resolved.is_relative_to(forbidden), (
            f"{what} resolves to {resolved}, which is inside {label} ({forbidden}): "
            f"a writer would land in real state, not scratch"
        )
    return resolved


def _pin_roots(tmp_path: Path, monkeypatch) -> dict:
    """Pin every root a writer in this process can resolve, to `tmp_path`.

    `builtin_fs` needs nothing: `Write`/`Edit` take an absolute path from the
    caller, so the scratch path *is* the pin. `session` and `vault` resolve a
    root of their own, and those are the globals this pins.
    """
    home = tmp_path / "home"
    vault = home / "obsidian"              # stand-in for ~/obsidian
    memories = vault / "lloyd"             # stand-in for ~/obsidian/lloyd
    audit = vault / "memory" / "audit"
    locks = tmp_path / "locks"
    for d in (memories, audit, locks):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(SESSION, "MEMORIES_ROOT", memories)
    monkeypatch.setattr(VAULT, "VAULT", vault)
    monkeypatch.setattr(VAULT, "AUDIT_LOG_DIR", audit)
    monkeypatch.setattr(VAULT, "AUDIT_LOG_FILE", audit / "writes.jsonl")
    monkeypatch.setattr(SHARED, "VAULT", vault)
    # `app.paths.VAULT_ROOT` is read at call time by `scratch_dir_for`, so unlike
    # the two above this patch is not inert — and without it a target in the
    # scratch vault takes the sibling-temp branch, so the off-tree assertions
    # would pass while proving nothing about the branch they name.
    monkeypatch.setattr("app.paths.VAULT_ROOT", vault)
    # Locks and temps default to `LOGS_DIR/locks`, inside the checkout that runs
    # the suite: moving the vault alone still litters the tree a cleanliness
    # clause is asserting about.
    monkeypatch.setattr(atomic_io, "LOCK_DIR", locks)
    monkeypatch.setattr(atomic_io, "SCRATCH_DIR", locks / "tmp")

    for p in (home, vault, memories, locks):
        _assert_off_tree(p, what="pinned root")
    _PIN_HOME[0] = home
    return {"home": home, "vault": vault, "memories": memories, "audit": audit,
            "locks": locks, "scratch": locks / "tmp"}


@pytest.fixture(autouse=True)
def off_tree(tmp_path, monkeypatch):
    """Apply `_pin_roots` to every test in this module, and hand out the paths."""
    return _pin_roots(tmp_path, monkeypatch)


def _assert_live_memory_files_clean(mark: str) -> None:
    """No sentinel this run invented may appear in the real memory files.

    Read-only, and keyed on a marker rather than on file bytes: the live
    `MEMORY.md` is written by other lanes, so a before/after hash would be a
    flake machine, while "our string is not in it" is exactly the property.
    """
    for live in LIVE_MEMORY_FILES:
        if not live.exists():
            continue
        assert mark not in live.read_text(encoding="utf-8"), (
            f"{live} contains {mark!r}: a test wrote through an unpinned root"
        )


# The scratch home for children, set by `_pin_roots` for the running test. A
# one-slot cell because the fixture rebinds per test while `_child_env` is a
# plain function the tests call directly.
_PIN_HOME: list = [None]


def _child_env(**extra) -> dict:
    """A child environment assembled from what a child needs, not from the shell.

    `env=os.environ.copy()` is the shape that hides a wrong root: the developer's
    shell can export `LLOYD_LOCK_DIR` (or anything else `LLOYD_*`), and
    `app.atomic_io` reads it at import for both `LOCK_DIR` and `SCRATCH_DIR`. The
    child then locks and temps somewhere the test never named, and an assertion
    about the tmp directory below the spawn measures the wrong directory while
    still passing. So start from a named base and drop every `LLOYD_*` a caller
    did not ask for — no inherited value can be load-bearing.

    `HOME` is the scratch home, not the real one, so even a child that forgot to
    assign a root resolves `Path.home() / "obsidian" / "lloyd"` into the tree the
    test owns. The child interpreter is named by absolute path, so its own prefix
    does not depend on any of this.
    """
    assert _PIN_HOME[0] is not None, "_child_env runs only inside the off_tree pin"
    env = {k: os.environ[k] for k in ("PATH", "USER", "LANG", "LC_ALL") if k in os.environ}
    env["HOME"] = str(extra.pop("HOME", _PIN_HOME[0]))
    env["PYTHONPATH"] = str(ROOT)
    assert not [k for k in env if k.startswith("LLOYD_")], "the base env must carry no LLOYD_*"
    env.update(extra)
    return env


def _write_child_script(tmp_path: Path, name: str, body: str) -> Path:
    """Drop a child script into the test's own tmp dir and return its path.

    The preamble is shared so the "every child sets its roots before it calls a
    writer" rule has one place to be true: the roots arrive on argv, and a child
    that fell back to a module default would print the default and be caught by
    the parent's ROOTS assertion rather than discovered in the vault.
    """
    script = tmp_path / name
    script.write_text(_CHILD_PREAMBLE + body, encoding="utf-8")
    return script


_CHILD_PREAMBLE = '''"""One writer, driven out of process, with every root named by argv."""
import os
import sys
from pathlib import Path

ROOT, VAULT_ROOT, MEMORIES_ROOT, LOCK_DIR, TARGET = sys.argv[1:6]
ARGS = sys.argv[6:]
sys.path.insert(0, ROOT)

from app import atomic_io                  # noqa: E402
from app import paths as _paths            # noqa: E402
from agent_mcp import session, vault       # noqa: E402

session.MEMORIES_ROOT = Path(MEMORIES_ROOT)
vault.VAULT = Path(VAULT_ROOT)
vault.AUDIT_LOG_DIR = Path(VAULT_ROOT) / "memory" / "audit"
vault.AUDIT_LOG_FILE = vault.AUDIT_LOG_DIR / "writes.jsonl"
_paths.VAULT_ROOT = Path(VAULT_ROOT)
atomic_io.LOCK_DIR = Path(LOCK_DIR)
atomic_io.SCRATCH_DIR = Path(LOCK_DIR) / "tmp"

# Say what was actually resolved, before the work, so the parent can assert the
# root the child used instead of assuming the one it asked for.
print("ROOTS", str(session.MEMORIES_ROOT), str(vault.VAULT), str(atomic_io.LOCK_DIR),
      flush=True)
'''


def _run_child(tmp_path: Path, name: str, body: str, pin: dict, target,
               *args, extra_env: dict | None = None, timeout: float = 120.0):
    """Run a child writer and return (CompletedProcess, the lines it printed)."""
    script = _write_child_script(tmp_path, name, body)
    env = _child_env(LLOYD_LOCK_DIR=str(pin["locks"]), **(extra_env or {}))
    _assert_off_tree(pin["memories"], what="child's MEMORIES_ROOT")
    _assert_off_tree(pin["vault"], what="child's VAULT")
    _assert_off_tree(pin["locks"], what="child's LOCK_DIR")
    _assert_off_tree(target, what="child's target")
    proc = subprocess.run(
        [sys.executable, str(script), str(ROOT), str(pin["vault"]),
         str(pin["memories"]), str(pin["locks"]), str(target), *args],
        capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=timeout)
    return proc, [ln for ln in proc.stdout.splitlines() if ln]


# ── base numbers, re-measured on this shape ──────────────────────────────────
#
# The lane off — a no-op `commit_lock` plus a bare `Path.write_text`, which is
# what `main` does — at exactly the shape clause 1 uses: 4 forked writers x 25
# appends of ~1.5 KB into a 20 KB file, one busy reader, one tmp dir per trial.
# Five trials, from a throwaway script under /tmp (it patches the two lane
# globals on `agent_mcp.session`, so the forked children inherit the off state):
#
#   survivors   23, 4, 16, 9, 3   of 100 appends   -> 77-97 % lost
#   empty reads 86, 116, 119, 92, 77  out of 391, 542, 456, 384, 411 reads
#   size drops  161, 370, 234, 283, 298
#
# So on this shape a median of about one append in ten survives, and roughly one
# read in four comes back empty. Both are worse than the numbers the item
# carried (19 and 9 survivors per 200 appends; 2 empty reads in 3,668), which
# were measured on a different shape; the figures above are the ones this file's
# sizes reproduce.


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
    _assert_off_tree(root, what=f"{tag} writers' MEMORIES_ROOT")
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
    # The root is the child's first statement and is checked before any write:
    # `memories_root` is a module global the fork inherits, so a child that
    # skipped this line would append into the real MEMORY.md and this test would
    # still read back its scratch copy and pass. A failed assert here is exit
    # code 1, which the parent's `exitcode == 0` check turns into a finding.
    SESSION.MEMORIES_ROOT = Path(root)
    _assert_off_tree(SESSION.MEMORIES_ROOT, what=f"{tag}'s MEMORIES_ROOT")
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

    Base, re-measured on these exact sizes with the lane disabled (a no-op
    `commit_lock` and a bare `Path.write_text`, which is what `main` does):
    3, 4, 9, 16 and 23 of the 100 sentinels survived, in five trials — 77 % to
    97 % of the appends erased, and not one of the losing writers reported a
    failure, because each one's own save was locally correct. Two writers on a
    small file does NOT fail, which is why the reproduction is 4 writers on a
    realistic-size file.
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
    """Drive builtin_fs._edit — the lane the nightly knowledge-write job uses.

    `Write`/`Edit` take an absolute path, so this lane has no memory-root global
    of its own — but it still resolves two things: `app.paths.VAULT_ROOT`, which
    is what decides whether the temp goes off-tree, and the scratch/lock dirs
    behind `atomic_io`. This child relied on inheriting those, so it pins and
    checks them itself rather than trusting the fork.
    """
    from app import paths as app_paths
    errors: list[str] = []
    SESSION.MEMORIES_ROOT = Path(root)
    VAULT.VAULT = Path(root)
    app_paths.VAULT_ROOT = Path(root)
    _assert_off_tree(SESSION.MEMORIES_ROOT, what=f"{tag}'s MEMORIES_ROOT")
    _assert_off_tree(app_paths.VAULT_ROOT, what=f"{tag}'s VAULT_ROOT")
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
    _assert_off_tree(root, what="the two-lane writers' MEMORIES_ROOT")
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
    all. Base, on these sizes: 77-119 empty reads out of 384-542 per trial, and
    161-370 reads that came back shorter than the previous one. Entries only
    ever append here, so the on-disk size is monotonic and a regression is proof
    of a torn read.
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
    _assert_off_tree(SESSION.MEMORIES_ROOT, what=f"{tag}'s MEMORIES_ROOT")
    i = 0
    while True:
        SESSION._memory_add({"file": "MEMORY.md", "entry": _entry(f"{tag}-s{i:04d}")})
        i += 1


def test_a_writer_killed_mid_write_leaves_the_file_byte_whole(tmp_path, off_tree):
    """SIGKILL the writer at random moments: the file is always a whole prefix.

    "Whole prefix" is the real bar: the file holds the header plus an integral
    number of complete entries. A torn write leaves a half-entry or no bytes at
    all, and on MEMORY.md the next prompt is assembled from that remnant. The
    kill also releases the flock with the descriptor, so a later write must
    still succeed — a lock that wedges on a dead writer is worse than the race.

    The scratch root is the pinned `MEMORIES_ROOT`, not a directory of its own:
    that last write is issued by the *parent*, and when it resolved the real
    root this test appended five filler entries to `~/obsidian/lloyd/MEMORY.md`
    on every run while asserting nothing about it.
    """
    root = off_tree["memories"]
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
        again = SESSION._memory_add({"file": "MEMORY.md",
                                     "entry": "- " + _entry(f"after-kill-{attempt}")})
        assert again.get("success"), f"attempt {attempt}: the write lock wedged after a kill: {again}"


# ── clause 4: no .lock beside the target, and not inside the vault ───────────

def _scratch_vault(tmp_path, monkeypatch):
    """Point the vault lane and the memory lane at one scratch tree.

    Delegates to `_pin_roots` so there is exactly one definition of "off-tree" in
    this file: a helper that pinned a subset is how a test ends up asserting
    about a scratch vault while one of its globals still resolves to the real one.
    """
    return _pin_roots(tmp_path, monkeypatch)["vault"]


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
    # The stronger property: running a writer must not dirty the checkout that
    # runs it. Asked of git, in the repo the locks would land in, not asserted of
    # a path pattern — a `.gitignore` rule is exactly the thing that goes missing.
    probe = subprocess.run(
        ["git", "-C", str(LLOYD_HOME), "check-ignore", "-q", "--",
         str(resolved.relative_to(LLOYD_HOME))],
        capture_output=True, text=True, check=False)
    assert probe.returncode == 0, (
        f"{resolved.relative_to(LLOYD_HOME)} is not git-ignored, so every run of "
        "a memory writer leaves an untracked path in the checkout "
        "(git check-ignore said nothing)"
    )
    assert not list(vault.rglob("*.tmp")), (
        f"a half-written temp was left inside the vault: {list(vault.rglob('*.tmp'))}"
    )


def test_the_lane_leaves_no_lock_or_temp_next_to_the_vault_target(tmp_path):
    """Assert the artefact *classes this change owns*, not "anything new".

    The version of this check swept up any new non-`.md` file and called it vault
    pollution, which is a false report twice over: a `.swp` from an editor, a
    `.bak` someone left, or a lock this lane does not even create all failed it,
    and none of them is what the item is about. So name the two classes the lane
    produces — the lock file (`locked_file`'s sibling `<name>.lock`, and the
    `.lloyd-lock` spelling) and the temp it writes before the rename — and then
    plant the unrelated kinds and show the check still passes. A test that failed
    on the planted files would be the finding again, in the other direction.
    """
    target = SESSION.MEMORIES_ROOT / "MEMORY.md"
    target.write_text(HEADER, encoding="utf-8")
    note = VAULT.VAULT / "knowledge" / "note.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text("# Note\n", encoding="utf-8")

    assert SESSION._memory_add({"file": "MEMORY.md", "entry": "one"})["success"]
    assert FS._write({"file_path": str(note), "content": "# Note two\n"}).startswith("File written")
    assert "Edited" in FS._edit({"file_path": str(note), "old_string": "# Note two",
                                 "new_string": "# Note three"})
    assert VAULT._vault_write({"path": "knowledge/note.md",
                               "content": "# Note four\n"}).get("success")

    def _owned_artefacts(where: Path):
        beside = [p for p in where.rglob("*") if p.is_file()]
        locks = [p for p in beside if p.suffix in (".lock", ".lloyd-lock")]
        temps = [p for p in beside if p.name.endswith(".tmp")]
        return locks, temps

    for where in (target.parent, VAULT.VAULT):
        locks, temps = _owned_artefacts(where)
        assert not locks, f"the lane left a lock file under {where}: {[str(p) for p in locks]}"
        assert not temps, f"the lane left a temp under {where}: {[str(p) for p in temps]}"

    # The two classes are asserted, not "no new files": unrelated litter the lane
    # does not own must not read as vault pollution.
    for junk in ("MEMORY.md.swp", "MEMORY.md.bak", "MEMORY.md.orig"):
        (target.parent / junk).write_text("not this lane's artefact\n", encoding="utf-8")
    for where in (target.parent, VAULT.VAULT):
        locks, temps = _owned_artefacts(where)
        assert not locks and not temps, (
            f"the class-specific check tripped on planted editor litter under {where}"
        )

    # And the scratch dir is where the temp went instead, so the assertion above
    # is not passing because nothing was ever written.
    pin_scratch = atomic_io.SCRATCH_DIR
    assert _real(pin_scratch) == _real(tmp_path / "locks" / "tmp"), (
        "the scratch dir is not the pinned one, so the two checks above would "
        "have passed without this lane's temps ever being looked for"
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


def test_a_stale_temp_in_the_scratch_dir_is_collected(tmp_path, monkeypatch):
    """The off-tree dir must not become a permanent graveyard.

    One pid's temp is exactly one filename, so every writer that dies between
    the temp write and the rename leaves a file behind and nothing removes it —
    the directory grows forever and every disk probe counts it. Age-gated, not
    pid-gated: pids get reused, and a temp written seconds ago belongs to a
    write that is still in flight.
    """
    scratch = tmp_path / "locks" / "tmp"
    scratch.mkdir(parents=True)
    monkeypatch.setattr(atomic_io, "SCRATCH_DIR", scratch)

    stale = scratch / f"old.{os.getpid() + 777}.tmp"
    stale.write_text("half a file", encoding="utf-8")
    expired = time.time() - atomic_io.SCRATCH_MAX_AGE_S - 10
    os.utime(stale, (expired, expired))
    live = scratch / f"live.{os.getpid() + 778}.tmp"
    live.write_text("in flight", encoding="utf-8")

    atomic_io._tmp_path(tmp_path / "note.md", scratch)

    assert not stale.exists(), "a stale leftover temp was never collected"
    assert live.exists(), "the sweep deleted a temp that could still be in flight"


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


# ── isolation, asserted ──────────────────────────────────────────────────────

def test_no_test_in_this_module_can_resolve_a_live_root():
    """The autouse pin is doing its job, and here is the proof.

    Every concurrency test below hands a scratch root to a child, so a missing pin
    is invisible in their own assertions: they read back the scratch file. This
    asks the question directly, from inside a test body, while the fixture is
    live — `session.MEMORIES_ROOT` must not be under `~/obsidian`, and a write
    issued from here must land in tmp.
    """
    home_vault = _real(Path.home() / "obsidian")
    assert not _real(SESSION.MEMORIES_ROOT).is_relative_to(home_vault), (
        f"session.MEMORIES_ROOT is {SESSION.MEMORIES_ROOT}: a parent-process writer "
        "in this file would append into the live memory files"
    )
    assert not _real(VAULT.VAULT).is_relative_to(home_vault), str(VAULT.VAULT)
    assert not _real(VAULT.AUDIT_LOG_DIR).is_relative_to(home_vault), str(VAULT.AUDIT_LOG_DIR)
    assert not _real(VAULT.AUDIT_LOG_FILE).is_relative_to(home_vault), str(VAULT.AUDIT_LOG_FILE)
    assert not _real(atomic_io.LOCK_DIR).is_relative_to(_real(LLOYD_HOME)), str(atomic_io.LOCK_DIR)

    res = SESSION._memory_add({"file": "MEMORY.md", "entry": "pinned-parent"})
    assert res.get("success"), res
    assert (SESSION.MEMORIES_ROOT / "MEMORY.md").read_text(encoding="utf-8").strip().endswith(
        "pinned-parent")
    _assert_live_memory_files_clean("pinned-parent")


def test_no_test_asserts_through_an_inert_patch():
    """A patch that cannot change the behaviour is a test that asserts nothing.

    Two ways this happens, and both were live in this file: a module that binds
    its own global at import while the test patches the module it was copied from
    (`app.paths.VAULT_ROOT` vs `session.MEMORIES_ROOT`), and a default argument
    that evaluates a constant once at def time, so `monkeypatch.setattr(mod,
    "DEFAULT_LOCK_WAIT", ...)` changes nothing a later call can see.
    """
    import inspect

    wait = inspect.signature(atomic_io.commit_lock).parameters["timeout"]
    assert wait.default is None, (
        f"commit_lock binds timeout={wait.default!r} as a default value, so patching "
        "DEFAULT_LOCK_WAIT is inert and the timeout test below proves nothing"
    )

    real = Path.home() / "obsidian" / "lloyd"
    assert SESSION.MEMORIES_ROOT != real, "the fixture is not pinning MEMORIES_ROOT"
    assert atomic_io.scratch_dir_for(SESSION.MEMORIES_ROOT / "MEMORY.md") is not None, (
        "app.paths.VAULT_ROOT was not pinned, so the off-tree temp branch is never "
        "entered and the vault-cleanliness tests pass for the wrong reason"
    )

    # And the seam itself: with the constant patched short, an untimed `commit_lock`
    # must wait the patched number, not the shipped 30 s. Held by a second open file
    # description, which is what an flock actually excludes.
    monkey_wait = 0.2
    assert atomic_io.DEFAULT_LOCK_WAIT > 1.0, atomic_io.DEFAULT_LOCK_WAIT
    target = SESSION.MEMORIES_ROOT / "MEMORY.md"
    target.write_text("base\n", encoding="utf-8")
    lock_path = atomic_io.lock_file_for(target)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    monkey_wait_start = time.monotonic()
    try:
        with _patched_wait(monkey_wait):
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                with atomic_io.commit_lock(target):
                    pass
            elapsed = time.monotonic() - started
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    assert elapsed < 1.0, (
        f"patched the lane wait to {monkey_wait}s and it still took {elapsed:.2f}s — "
        "the constant is bound somewhere the call cannot see it"
    )
    assert elapsed >= monkey_wait - 0.05, (elapsed, monkey_wait_start)


class _patched_wait:
    """Patch the lane's wait constant, the seam callers actually use."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._saved = None

    def __enter__(self):
        self._saved = atomic_io.DEFAULT_LOCK_WAIT
        atomic_io.DEFAULT_LOCK_WAIT = self.seconds

    def __exit__(self, *exc):
        atomic_io.DEFAULT_LOCK_WAIT = self._saved
        return False


def test_the_lane_recovers_from_a_full_lock_dir(tmp_path, off_tree, monkeypatch):
    """A lock directory a writer cannot create a lock in must not hang or write.

    "Full" here means unreadable-for-writing: the lock dir is mode 0o500, so the
    open that would create the lock file fails. That is the shape a permissions
    accident, a read-only mount or an exhausted directory takes, and the two wrong
    answers on the other side of it are a hang and a silent write.

    The child's environment is assembled by `_child_env`, not copied from the
    shell: `LLOYD_LOCK_DIR` inherited from a developer's session would move both
    the lock and the scratch dir somewhere this test never named, and the exact
    `tmp_path` assertions below it would be measuring the wrong directory while
    still passing. Nothing load-bearing is inherited either, because the child
    prints what it resolved and the parent asserts the paths it was given.
    """
    target = off_tree["memories"] / "MEMORY.md"
    prior = "the bytes that were there before\n"
    target.write_text(prior, encoding="utf-8")
    os.chmod(off_tree["locks"], 0o500)
    try:
        proc, lines = _run_child(tmp_path, "child_full_lockdir.py", '''
import traceback
try:
    from agent_mcp import session
    res = session._memory_add({"file": Path(TARGET).name, "entry": "sentinel-full-lockdir"})
    print("CODE", res.get("code", "success"), flush=True)
except Exception as exc:
    print("RAISED", type(exc).__name__, flush=True)
print("LOCKDIR", str(atomic_io.LOCK_DIR), flush=True)
print("SCRATCH", str(atomic_io.SCRATCH_DIR), flush=True)
''', off_tree, target, timeout=120)
    finally:
        os.chmod(off_tree["locks"], 0o700)

    assert proc.returncode == 0, f"the child died: {proc.stderr[-800:]}"
    roots = [ln for ln in lines if ln.startswith("ROOTS")][0].split()
    assert _real(roots[1]) == _real(off_tree["memories"]), lines
    assert _real(roots[2]) == _real(off_tree["vault"]), lines
    assert _real(roots[3]) == _real(off_tree["locks"]), (
        f"the child locked somewhere other than the directory under test: {roots[3]}"
    )
    lockdir = [ln for ln in lines if ln.startswith("LOCKDIR")][0].split()[1]
    scratch = [ln for ln in lines if ln.startswith("SCRATCH")][0].split()[1]
    assert _real(lockdir) == _real(off_tree["locks"]), (
        f"reported lock dir {lockdir} is not the directory under test, so every "
        "assertion below it is measuring a directory this test did not name"
    )
    assert _real(scratch).is_relative_to(_real(tmp_path)), scratch

    outcome = [ln for ln in lines if ln.startswith(("CODE", "RAISED"))][0].split()
    assert outcome[0] == "RAISED" and outcome[1] == "PermissionError", (
        f"a lock dir the writer cannot use produced {outcome} — it must surface the "
        "failure, not queue forever and not report success"
    )
    assert target.read_text(encoding="utf-8") == prior, (
        "the write got through anyway: a lock that could not be taken did not stop "
        "the commit, which is a lock that is not a lock"
    )

    # Recovered: the same call, same path, once the directory can hold a lock.
    again = SESSION._memory_add({"file": "MEMORY.md", "entry": "after-lockdir-fixed"})
    assert again.get("success"), again
    text = target.read_text(encoding="utf-8")
    assert text == prior + "after-lockdir-fixed\n", text
    assert "sentinel-full-lockdir" not in text
    _assert_live_memory_files_clean("sentinel-full-lockdir")


# ── the two lanes, as the shipped dispatchers run them ───────────────────────

def test_memory_dispatch_is_inline_and_the_filesystem_lane_is_on_a_thread():
    """Pin the shape the concurrency claim rests on, or correct the claim.

    `agent_mcp/session.py`'s `call_tool` calls the memory handler directly, so a
    `memory_add` runs to completion *on the event loop* — it is a sync function
    doing file I/O, and nothing yields. `agent_mcp/builtin_fs.py`'s `call_tool`
    puts `_write`/`_edit` on a worker thread with `asyncio.to_thread`, explicitly
    so the loop keeps serving SSE chat and voice. That pairing is exactly why the
    two lanes can be inside their respective commits at the same instant: one is
    on the loop, one is on another thread, and neither one's lock-free window is
    visible to the other. If memory dispatch were also on a thread, or if it
    yielded, the interleaving below would not exist as shipped and the tests would
    be pinning a shape nobody runs.
    """
    session_src = (ROOT / "agent_mcp/session.py").read_text(encoding="utf-8")
    fs_src = (ROOT / "agent_mcp/builtin_fs.py").read_text(encoding="utf-8")

    def _call_tool(src: str):
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "call_tool":
                return node
        raise AssertionError("call_tool is gone; this guard must move, not vanish")

    mem = ast.unparse(_call_tool(session_src))
    assert "to_thread" not in mem and "run_in_executor" not in mem, (
        "memory dispatch is no longer inline, so the race below is not the shipped "
        f"shape and the claim about it needs rewriting:\n{mem}"
    )
    assert "handler(arguments)" in mem, mem

    fs = ast.unparse(_call_tool(fs_src))
    assert "asyncio.to_thread(handler, arguments, mut)" in fs, fs
    assert "_write if name ==" in fs and "else _edit" in fs, fs

    # Sync handlers, so the loop is blocked inside the commit rather than yielding
    # at an await: that is what makes the overlap a real overlap.
    tree = ast.parse(session_src)
    for name in ("_memory_add", "_memory_replace", "_memory_remove"):
        defs = [n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name]
        assert defs and isinstance(defs[0], ast.FunctionDef), (
            f"{name} became async, so memory dispatch no longer blocks the loop"
        )

_RACE_ROUNDS = 15


def _race_file(root: Path, rounds: int) -> Path:
    """One file, big enough that a commit costs milliseconds, with N placeholders.

    On a toy file each side finishes before the other starts, so "nothing was
    lost" measures the scheduler instead of the lock. Both halves of the pair
    below use this shape, so neither can be vacuous while the other looks real.
    """
    target = root / "MEMORY.md"
    lines = [HEADER] + [f"PLACEHOLDER-{i:03d}\n" for i in range(rounds)]
    while len("".join(lines).encode("utf-8")) < RACE_BYTES:
        lines.append("- standing entry " + "x" * 200 + "\n")
    target.write_text("".join(lines), encoding="utf-8")
    return target


async def _edit_from_a_thread_memory_add_from_the_loop(target: Path, rounds: int):
    """Run N Edit/append pairs in the shapes the shipped dispatchers use.

    `builtin_fs.call_tool` builds the `_Mutation` on the loop and awaits
    `asyncio.to_thread(handler, arguments, mut)`; `session.call_tool` calls
    `_memory_add(arguments)` inline, so the loop is *inside* the append's commit
    while the worker thread is inside its own. All N edits are dispatched before
    anything is awaited — the default thread pool runs them concurrently, which is
    what several sessions editing while a chat turn appends actually looks like.
    The dispatch entry point is reproduced rather than invoked because
    `call_tool`'s tail runs the edit diagnostics and a code-graph read over the
    file, per iteration, which has nothing to do with which thread writes.
    """
    async def one(i: int):
        mut = FS._Mutation(kind="edit", session_id="",
                           gate_on=bool("") and FS._gates_enabled(), scope=None, call_id="")
        return await asyncio.to_thread(
            FS._edit, {"file_path": str(target),
                       "old_string": f"PLACEHOLDER-{i:03d}",
                       "new_string": f"EDITED-{i:03d}"}, mut)

    async def drive():
        tasks = []
        adds = []
        for i in range(rounds):
            tasks.append(asyncio.ensure_future(one(i)))
            # Inline, on this loop, exactly as session.call_tool runs its handler.
            # Catching here is not softening the assertion: a lane that *raises*
            # instead of returning an error dict is the worse outcome, and the
            # callers below count it as a failure.
            try:
                adds.append(SESSION._memory_add({"file": "MEMORY.md",
                                                 "entry": _entry(f"il-{i:03d}")}))
            except Exception as exc:
                adds.append({"success": False, "raised": f"{type(exc).__name__}: {exc}"})
            await asyncio.sleep(0)      # let the dispatched threads actually run
        texts = await asyncio.gather(*tasks, return_exceptions=True)
        return texts, adds

    return await drive()


def _race_survivors(text: str, rounds: int) -> tuple[int, int]:
    edits = sum(1 for i in range(rounds) if f"EDITED-{i:03d}" in text)
    appends = sum(1 for i in range(rounds) if f"il-{i:03d}" in text)
    return edits, appends


def test_an_edit_on_a_worker_thread_and_a_memory_add_in_the_loop_lose_neither(
        tmp_path, off_tree):
    """The shipped interleaving, with both changes required to survive.

    Edit (worker thread) and memory_add (inline, on the loop) against one file,
    fifteen pairs, dispatched the way the two `call_tool`s dispatch. Both halves
    have to land on every round: a lost append is the lost update this item is
    about, and a lost edit is what a whole-file memory commit does to an editor
    that read before it.

    Edit, not Write, is the lane driven here: a Write is a whole-file overwrite,
    so a Write that commits second legitimately drops the concurrent append and
    "both changes survive" is false for it by design. `call_tool` takes the single
    `asyncio.to_thread` branch for Write and Edit alike, and
    `test_memory_dispatch_is_inline_and_the_filesystem_lane_is_on_a_thread` pins
    that, so the thread-shape claim is pinned for both while the survival claim is
    asserted of the lane that can actually lose one.

    `test_the_same_race_without_the_lock_loses_updates` is its pair: same shape,
    lock removed, and it must lose.
    """
    target = _race_file(off_tree["memories"], _RACE_ROUNDS)
    texts, adds = asyncio.run(
        _edit_from_a_thread_memory_add_from_the_loop(target, _RACE_ROUNDS))

    for i, (text, added) in enumerate(zip(texts, adds)):
        assert not isinstance(text, BaseException), \
            f"round {i}: the threaded Edit raised {text!r}"
        assert not (text.startswith("{") and "error" in json.loads(text)), \
            f"round {i}: the threaded Edit failed: {text}"
        assert added.get("success"), f"round {i}: the inline append failed: {added}"

    edits, appends = _race_survivors(target.read_text(encoding="utf-8"), _RACE_ROUNDS)
    assert edits == _RACE_ROUNDS, (
        f"the inline appends clobbered {_RACE_ROUNDS - edits}/{_RACE_ROUNDS} of the "
        f"threaded Edits ({edits} survived): a memory commit that read before an "
        "Edit committed reverted it"
    )
    assert appends == _RACE_ROUNDS, (
        f"the threaded Edits clobbered {_RACE_ROUNDS - appends}/{_RACE_ROUNDS} of the "
        f"inline appends ({appends} survived)"
    )


def test_the_same_race_without_the_lock_loses_updates(monkeypatch, tmp_path, off_tree):
    """The pair above is a regression test, not a restatement of the fix.

    Identical driver, identical file shape, `commit_lock` replaced on both lanes by
    a no-op context manager. If this passed too, the assertions above would be
    sensitive to something other than the lock.

    Measured on this machine, three runs, against 15/15 and 15/15 with the lock in
    place: edits 7-10 of 15 surviving, appends 11-12 of 15, plus 6-9 threaded Edits
    and 4-6 inline appends that failed outright. The failures are their own finding
    — with no lock, the two threads of this process share one temp name by design
    (`_tmp_path` keys on target + pid), so one writer's `os.replace` moves the file
    out from under the other's and it surfaces as FileNotFoundError. A lost update
    without the lock is sometimes an error and sometimes silence; the silence is
    the dangerous half, and both count as a failure here.
    """
    import contextlib

    @contextlib.contextmanager
    def _no_lock(path, **kw):
        yield Path(path)

    monkeypatch.setattr(SESSION, "commit_lock", _no_lock)
    monkeypatch.setattr(FS, "commit_lock", _no_lock)

    target = _race_file(off_tree["memories"], _RACE_ROUNDS)
    texts, adds = asyncio.run(
        _edit_from_a_thread_memory_add_from_the_loop(target, _RACE_ROUNDS))
    edits, appends = _race_survivors(target.read_text(encoding="utf-8"), _RACE_ROUNDS)
    failed_threads = sum(1 for t in texts
                         if isinstance(t, BaseException)
                         or (t.startswith("{") and "error" in json.loads(t)))
    failed_adds = [a for a in adds if not a.get("success")]
    print(f"\nno-lock survivors: edits {edits}/{_RACE_ROUNDS}, "
          f"appends {appends}/{_RACE_ROUNDS}, "
          f"failed_threads={failed_threads}, failed_adds={len(failed_adds)}"
          f"{'; e.g. ' + str(failed_adds[0].get('raised'))[:160] if failed_adds else ''}")
    assert (edits < _RACE_ROUNDS or appends < _RACE_ROUNDS
            or failed_threads or failed_adds), (
        f"all {2 * _RACE_ROUNDS} changes survived, failed and lost alike, with both "
        "lanes' commit locks removed, so the paired test does not depend on the lock"
    )


# ── the lock-wait seam, through the real callers ─────────────────────────────

def _hold_lock_externally(tmp_path: Path, target: Path, off_tree: dict,
                          seconds: float = 25.0) -> subprocess.Popen:
    """Start a child that takes `target`'s commit lock and keeps it, and wait until it has.

    Out of process because the property is that a *separate* holder makes a caller
    wait, and the way `commit_lock` buys that is an flock on one file: a child that
    is not this process is the only way to ask the question the shipped callers
    actually face. The child prints `HELD` only after `flock` returns, so the
    parent's measurement cannot start early — a sleep is a race, a line on a pipe
    is a synchronisation.
    """
    proc = subprocess.Popen(
        [sys.executable, str(_write_child_script(tmp_path, "child_holds_lock.py", '''
import fcntl, time
lock = atomic_io.lock_file_for(TARGET)
fd = os.open(str(lock), os.O_CREAT | os.O_RDWR, 0o644)
fcntl.flock(fd, fcntl.LOCK_EX)
print("HELD", str(lock), flush=True)
time.sleep(float(ARGS[0]) if ARGS else 25)
''')),
         str(ROOT), str(off_tree["vault"]), str(off_tree["memories"]),
         str(off_tree["locks"]), str(target), str(seconds)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        env=_child_env(LLOYD_LOCK_DIR=str(off_tree["locks"])), cwd=str(ROOT))
    try:
        roots = proc.stdout.readline()      # the preamble's ROOTS line
        held = proc.stdout.readline()
    except BaseException:
        proc.kill()
        raise
    assert held.startswith("HELD"), (
        f"the holder never got the lock (roots={roots!r}, held={held!r}, "
        f"stderr={proc.stderr.read()[-600:]!r})"
    )
    return proc


def _release(proc: subprocess.Popen) -> None:
    """Kill the holder; the flock dies with the descriptor."""
    proc.kill()
    proc.wait(10)


def test_memory_add_reports_lock_timeout_on_the_real_path(tmp_path, off_tree):
    """`_memory_add` against a held lock: LOCK_TIMEOUT, in about the patched wait.

    Drives the real caller, with the wait patched through the lane's module
    constant — the seam a caller has, and the one `commit_lock` resolves at call
    time (pinned by `test_no_test_in_this_module_asserts_through_an_inert_patch`).
    Calling `commit_lock` directly with an explicit `timeout=` proves the primitive
    and leaves this path unexercised, because nothing in production ever passes a
    timeout: it is the constant or nothing. The wall-time assertion is what makes
    the difference visible — the shipped wait is 30 s, so a test that patched
    nothing, or patched something inert, would sit on a half-minute chat turn.
    """
    target = off_tree["memories"] / "MEMORY.md"
    other = "the other writer got there first\n"
    target.write_text(other, encoding="utf-8")

    holder = _hold_lock_externally(tmp_path, target, off_tree)
    try:
        with _patched_wait(0.25):
            started = time.monotonic()
            res = SESSION._memory_add({"file": "MEMORY.md", "entry": "will-not-land"})
            elapsed = time.monotonic() - started
    finally:
        _release(holder)

    assert res.get("code") == "LOCK_TIMEOUT", res
    assert res.get("error"), res
    assert 0.2 <= elapsed < 3.0, (
        f"waited {elapsed:.2f}s for a patched 0.25s wait: the patch is not the seam "
        f"the real call uses (shipped DEFAULT_LOCK_WAIT is "
        f"{atomic_io.DEFAULT_LOCK_WAIT}s)"
    )
    assert "will-not-land" not in target.read_text(encoding="utf-8"), (
        "the timed-out writer replaced the file anyway: a lock timeout that still "
        "writes is a lost update with an error message stapled on"
    )
    # And with the holder gone, the same call goes through — a timeout is a wait,
    # not a broken lock.
    assert SESSION._memory_add({"file": "MEMORY.md",
                                "entry": "landed-after-holder-left"}).get("success")
    assert target.read_text(encoding="utf-8") == other + "landed-after-holder-left\n"
    _assert_live_memory_files_clean("will-not-land")


def test_vault_write_reports_lock_timeout_on_the_real_path(tmp_path, off_tree):
    """The same seam through the vault lane, which has its own error shape.

    `_vault_write` echoes `path` back so a model can retry the right note, and it
    reaches the lock by a different spelling — a vault-relative path — which is
    exactly why the three-lane lock key matters: the holder above took the lock by
    realpath, and a `_vault_write` that keyed differently would sail past it.
    """
    rel = "lloyd/MEMORY.md"
    target = off_tree["vault"] / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    prior = "the version under lock\n"
    target.write_text(prior, encoding="utf-8")

    holder = _hold_lock_externally(tmp_path, target, off_tree)
    try:
        with _patched_wait(0.25):
            started = time.monotonic()
            res = VAULT._vault_write({"path": rel, "content": "will-not-land\n"})
            elapsed = time.monotonic() - started
    finally:
        _release(holder)

    assert res.get("code") == "LOCK_TIMEOUT", res
    assert res.get("path") == rel, res
    assert 0.2 <= elapsed < 3.0, elapsed
    assert target.read_text(encoding="utf-8") == prior, (
        "the timed-out vault write landed anyway"
    )
