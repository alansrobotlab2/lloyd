"""The Bash substrate: protected paths enforced by the kernel, not by a parser (#2109).

Two layers of this file, and both matter:

* **Shape** — `profile_argv` must emit exactly one `--ro-bind <entry> <entry>`
  per protected entry, bind a file as the file and never its parent, skip an
  entry that is absent, and be neither whole-root read-only nor the bench
  profile.
* **Behaviour** — a corpus of protected-tree write and delete shapes is executed
  *for real* under that profile against a scratch `$HOME`, and every one of them
  has to come back refused with the bytes unchanged; writes, `mkdir`s, a
  `git commit` and a loopback TCP connect outside the list have to keep working;
  and a timeout or a cancel through the real `Bash` tool has to leave no member
  of the spawned process group alive.

Nothing here can touch the live vault: `protected_shell_ro_paths()` resolves its
`~/…` templates against `$HOME` on every call, so the `scratch_home` fixture
pointing `$HOME` at a `tmp_path` directory moves every bind into `tmp_path`, and
`test_every_bind_names_a_path_under_the_scratch_home` is the rail that says so.
The real bwrap is used, skipped only if this box cannot build a namespace at all
— a skipped substrate test would otherwise read as a passing one.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_mcp import _path_sandbox as P            # noqa: E402
from agent_mcp import _task_registry as TR          # noqa: E402
from agent_mcp import builtin_bash as B             # noqa: E402
from app.harness import protected_paths as PP       # noqa: E402

_HAS_BWRAP = shutil.which("bwrap") is not None
needs_bwrap = pytest.mark.skipif(not _HAS_BWRAP,
                                 reason="bubblewrap is not installed on this box")
skipif_no_bwrap = pytest.mark.skipif(
    not P.available()[0],
    reason=f"bwrap cannot build a namespace here: {P.available()[1]}")


# ---------------------------------------------------------------------------
# fixture: a scratch $HOME whose protected entries are files under tmp_path
# ---------------------------------------------------------------------------

@pytest.fixture
def scratch_home(tmp_path, monkeypatch):
    """A fake home with the six protected entries present except `~/.openclaw`.

    `~/.openclaw` is left absent on purpose: it is a member of the deny-set that
    does not exist on this box, and clause 1 is that an absent entry is skipped
    rather than breaking the profile build.

    `~/obsidian` is made a git repo with `lloyd/*` committed and `USER.md` then
    dirtied, so `git -C … checkout -- lloyd/USER.md` in the corpus is a command
    with work to do rather than a no-op that would exit 0 without touching
    anything.
    """
    home = tmp_path / "home"
    mem = home / "obsidian" / "lloyd"
    mem.mkdir(parents=True)
    for name in ("SOUL.md", "USER.md", "MEMORY.md"):
        (mem / name).write_text("ORIGINAL\n")
    for rel in ("lloyd/.venvs", "lloyd/agent-services", ".local/share/uv/python",
                "work"):
        (home / rel).mkdir(parents=True)
    (home / "seed.txt").write_text("EVIL\n")

    vault = home / "obsidian"
    _git(vault, "init", "-q")
    _git(vault, "add", "-A")
    _git(vault, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "seed")
    (mem / "USER.md").write_text("DIRTY-BY-A-ROUND\n")

    monkeypatch.setenv("HOME", str(home))
    P.reset_probe_cache()
    yield SimpleNamespace(home=home, mem=mem, vault=vault,
                          seed=home / "seed.txt", work=home / "work")
    P.reset_probe_cache()


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True,
                          check=False)


def _run(command: str, cwd: str) -> tuple[int, str]:
    """Execute `command` under the profile, through `/bin/sh -c` like Bash does."""
    proc = subprocess.run(P.profile_argv(command, cwd), capture_output=True,
                          timeout=90, check=False)
    return proc.returncode, (proc.stdout + proc.stderr).decode(errors="replace")


def _live_procs(marker: str) -> "list[tuple[str, str]]":
    """Every live process whose command line contains `marker`."""
    out = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            cmd = Path(f"/proc/{entry}/cmdline").read_bytes().decode(
                errors="replace").replace("\0", " ").strip()
        except OSError:
            continue        # exited between the listing and the read
        if marker in cmd:
            out.append((entry, cmd))
    return out


# ---------------------------------------------------------------------------
# clause 1 — one bind per entry, a file bound as itself, an absent entry skipped
# ---------------------------------------------------------------------------

def _ro_binds(argv: "list[str]") -> "list[tuple[str, str]]":
    return [(argv[i + 1], argv[i + 2])
            for i, tok in enumerate(argv) if tok == "--ro-bind"]


def test_profile_binds_each_existing_protected_entry_exactly_once(scratch_home):
    entries = PP.protected_shell_ro_paths()
    assert entries, "the scratch home should present the protected entries"
    argv = P.profile_argv("true", str(scratch_home.home))
    binds = _ro_binds(argv)
    assert binds == [(e, e) for e in entries], (
        f"one --ro-bind per entry, source and destination identical; got {binds}")
    assert len(set(dest for _src, dest in binds)) == len(entries), \
        "no entry is bound twice, and no two entries share a destination"


def test_a_file_entry_is_bound_as_the_file_and_never_as_its_parent(scratch_home):
    """A parent-directory bind is the bug: it takes `.consolidate-lock` and any
    sibling with it, and the dream-consolidation lock probe reports that as
    `LOCK_MISSING … proceeding`, so the failure is silent."""
    argv = P.profile_argv("true", str(scratch_home.home))
    dests = {dest for _src, dest in _ro_binds(argv)}
    for name in ("SOUL.md", "USER.md", "MEMORY.md"):
        assert str(scratch_home.mem / name) in dests, f"{name} must be bound itself"
    assert str(scratch_home.mem) not in dests, \
        "the memory directory is never bound: a parent bind would make its " \
        "siblings unwritable"
    assert str(scratch_home.vault) not in dests


def test_an_absent_protected_entry_costs_the_profile_a_bind_only(scratch_home):
    """Clause 1's second half, and deliberately NOT gated on bwrap.

    `~/.openclaw` is a member of the list and absent on this box; a bind needs a
    target, so it has to be dropped. The assertion is the pure argv build, not an
    executed probe, because a profile that cannot build at all must not be able
    to hide that behind a skip: the shape of the argument list is checkable
    without a kernel, and it is the shape that decides whether Bash works.

    The failure mode this pins is the one `_host_socket_paths`' docstring
    records: one un-creatable mount point takes away every Bash call while
    `/state` keeps reporting healthy.
    """
    absent = os.path.expanduser("~/.openclaw")
    if os.path.exists(absent):
        pytest.skip(f"{absent} exists on this box, so no entry of the list is "
                    "absent here and the skip-everything-else path cannot be "
                    "witnessed — the scratch home has no other absent member")
    assert absent not in PP.protected_shell_ro_paths()
    argv = P.profile_argv("true", str(scratch_home.home))
    assert [dest for _src, dest in _ro_binds(argv) if absent in dest] == []
    # …and dropping it costs a bind and nothing else: the profile is still a
    # profile, with its writable root and its command at the end.
    assert argv[:1] == [P.bwrap_path()] and "--bind" in argv
    assert argv[-3:] == ["/bin/sh", "-c", "true"]


@skipif_no_bwrap
def test_the_profile_still_builds_and_runs_with_an_entry_absent(scratch_home):
    """The same fact executed: with `~/.openclaw` absent, a real bwrap spawn
    under the profile still runs the command."""
    rc, out = _run("echo built", str(scratch_home.home))
    assert rc == 0, f"the profile failed to build with an entry absent: {out}"
    assert "built" in out


@skipif_no_bwrap
def test_every_bind_names_a_path_under_the_scratch_home(scratch_home):
    """The rail against this file measuring the live vault instead of a copy."""
    root = os.path.realpath(str(scratch_home.home))
    for src, dest in _ro_binds(P.profile_argv("true", str(scratch_home.home))):
        assert os.path.realpath(src) == src and os.path.realpath(dest) == dest
        assert src.startswith(root + os.sep), src
        assert dest.startswith(root + os.sep), dest


# ---------------------------------------------------------------------------
# clause 2 — the corpus, executed
# ---------------------------------------------------------------------------

# Every vector is a shape that has to be *refused*; the third field is the
# kernel's answer the tool must show. Note the two families: `EROFS`
# ("Read-only file system") for opening a file for writing, and `EBUSY`
# ("Device or resource busy") for unlinking or renaming one that is itself a
# mount point — `#582`'s design note records the file bind refusing
# rm/mv/sed-rename with EBUSY, and a bind of the parent directory is the only
# thing that would make those `EROFS` instead. `EACCES` is the third answer the
# kernel can give; the read-only mount answers first on this box.
CORPUS = [
    ("cat > USER.md",                "cat {seed} > {mem}/USER.md",
     "Read-only file system"),
    ("echo > MEMORY.md",             "echo x > {mem}/MEMORY.md",
     "Read-only file system"),
    ("tee -a MEMORY.md",             "echo x | tee -a {mem}/MEMORY.md",
     "Read-only file system"),
    ("truncate -s 0",                "truncate -s 0 {mem}/MEMORY.md",
     "Read-only file system"),
    ("rm <single file>",             "rm {mem}/USER.md",
     "Device or resource busy"),
    ("find … -delete",               "find {mem} -name USER.md -delete",
     "Device or resource busy"),
    ("ln -sfn",                      "ln -sfn {seed} {mem}/MEMORY.md",
     "Device or resource busy"),
    ("cp over SOUL.md",              "cp {seed} {mem}/SOUL.md",
     "Read-only file system"),
    ("mv onto USER.md",              "mv {seed} {mem}/USER.md",
     "Device or resource busy"),
    ("sed -i in place",              "sed -i s/x/y/ {mem}/MEMORY.md",
     "Device or resource busy"),
    ("python3 open(…, 'w')",
     '''python3 -c "open('{mem}/USER.md','w').write('x')"''',
     "Read-only file system"),
    ("python3 os.remove",
     '''python3 -c "import os;os.remove('{mem}/SOUL.md')"''',
     "Device or resource busy"),
    ("python3 Path.write_text",
     '''python3 -c "from pathlib import Path; '''
     '''Path('{mem}/MEMORY.md').write_text('x')"''',
     "Read-only file system"),
    # perl reports its own rename failure before the kernel's words reach it, so
    # either the syscall's errno or perl's line about it is the witness; and with
    # no perl on the box the refusal would be a 127, so the vector names its
    # dependency and is skipped rather than reddened for the wrong reason.
    ("perl -ni -e",                  "perl -ni -e 'print' {mem}/SOUL.md",
     ("failed to rename", "Operation not permitted", "Read-only file system",
      "Device or resource busy"), "perl"),
    ("git -C … checkout -- lloyd/USER.md",
     "git -C {vault} checkout -- lloyd/USER.md",
     "Device or resource busy"),
]


#: Vector label -> the program it needs, for the ones whose refusal is only
#: meaningful if that program is the one that ran.
def _corpus_vectors(scratch_home):
    """`(label, command, expected_texts, needs_program)` per corpus vector."""
    subs = {"mem": str(scratch_home.mem), "vault": str(scratch_home.vault),
            "seed": str(scratch_home.seed)}
    out = []
    for row in CORPUS:
        label, cmd, expect = row[:3]
        out.append((label, cmd.format(**subs),
                    (expect,) if isinstance(expect, str) else tuple(expect),
                    row[3] if len(row) > 3 else None))
    return out


def test_the_corpus_carries_every_shape_the_item_names(scratch_home):
    """The corpus's own floor: at least twelve vectors, and the shapes the item
    names are each in it. This is a claim about the LIST only — that they are
    refused is `test_every_corpus_vector_is_refused_and_the_bytes_are_unchanged`,
    which executes them, and that the predicate is blind to them is
    `test_the_string_predicate_is_still_blind_to_the_shapes_the_substrate_refuses`.
    Ungated on purpose: a box with no bwrap still pins the floor here."""
    labels = [row[0] for row in _corpus_vectors(scratch_home)]
    # The count the item states, asserted as a number: `>= 12` was true of any
    # list this file could build, and an off-by-one here is the whole clause.
    assert len(labels) == 15, \
        f"{len(labels)} vectors: the item's twelve-plus floor, the three shapes "        f"added for the deny-set files, and the git checkout it rules on "        f"separately. Update deliberately, never to fit: {labels}"
    for named in ("truncate -s 0", "rm <single file>", "find … -delete",
                  "perl -ni -e", "ln -sfn", "cat > USER.md", "tee -a MEMORY.md",
                  "python3 open(…, 'w')",
                  "git -C … checkout -- lloyd/USER.md"):
        assert named in labels, f"{named} is a shape check_bash_command returns " \
                                f"None for today and must be in the corpus"


@skipif_no_bwrap
def test_every_corpus_vector_is_refused_and_the_bytes_are_unchanged(scratch_home):
    before = {name: (scratch_home.mem / name).read_bytes()
              for name in ("SOUL.md", "USER.md", "MEMORY.md")}
    failures = []
    for label, cmd, expects, needs in _corpus_vectors(scratch_home):
        if needs and shutil.which(needs) is None:
            pytest.skip(f"{label} needs {needs!r} installed; without it the "
                        "refusal is a 127 from the shell, not the kernel's")
        rc, out = _run(cmd, str(scratch_home.home))
        if rc == 0:
            failures.append(f"{label}: exited 0 — the write went through")
        elif not any(e in out for e in expects):
            failures.append(f"{label}: rc={rc} but none of {expects} in "
                            f"{out[:160]!r}")
    assert not failures, "vectors the substrate let through:\n" + \
                         "\n".join(failures)
    after = {name: (scratch_home.mem / name).read_bytes()
             for name in ("SOUL.md", "USER.md", "MEMORY.md")}
    assert after == before, \
        f"protected bytes changed under a read-only bind: {before} -> {after}"
    # `git checkout --` had work to do: USER.md is the file the round dirtied
    # after the seed commit, and the kernel — not git's own goodwill — refused it.
    assert b"DIRTY-BY-A-ROUND" in after["USER.md"]


@skipif_no_bwrap
def test_the_two_scheduled_vault_writes_stay_writable(scratch_home):
    """The #2108 pair, pinned as survivors of the file-only bind: dream's
    weekly stamp writes `~/obsidian/lloyd/.consolidate-lock`, and
    historical-knowledge-refresh copies MEMORY.md to a sibling backup. Neither
    is a protected entry, so a round that widened a file bind to its parent
    directory would silently break the nightly — #673 is what that looks like."""
    rc, out = _run(f"echo stamped > {scratch_home.mem}/.consolidate-lock "
                   f"&& cp {scratch_home.mem}/MEMORY.md "
                   f"{scratch_home.mem}/MEMORY.md.backup.20261003 "
                   f"&& cat {scratch_home.mem}/.consolidate-lock",
                   str(scratch_home.home))
    assert rc == 0, f"a scheduled write inside the memory tree was refused: {out}"
    assert (scratch_home.mem / ".consolidate-lock").read_text().strip() == "stamped"
    assert (scratch_home.mem / "MEMORY.md.backup.20261003").read_bytes() == \
        b"ORIGINAL\n"


# ---------------------------------------------------------------------------
# the predicate is unchanged: the substrate is what refuses these
# ---------------------------------------------------------------------------

def test_the_string_predicate_is_still_blind_to_the_shapes_the_substrate_refuses():
    """No new `check_bash_command` regexes are part of this change, and this is
    the proof: these twelve real-path spellings are the residual the 2026-10-03
    triage measured as returning `None` on the hook lane, and they still do.
    They are refused by the mount, so a future run must not "fix" them here —
    and must not quietly add a pattern and call this test obsolete."""
    from app.harness.safety import check_bash_command

    real = os.path.expanduser("~")
    shapes = [
        f"truncate -s 0 {real}/obsidian/lloyd/SOUL.md",
        f"rm {real}/obsidian/lloyd/SOUL.md",
        f"rm {real}/obsidian/lloyd/MEMORY.md",
        f"truncate -s 0 {real}/obsidian/lloyd/USER.md",
        f"find {real}/obsidian -name USER.md -delete",
        "perl -ni -e 'print' " + f"{real}/obsidian/lloyd/SOUL.md",
        f"cat /tmp/x > {real}/obsidian/lloyd/USER.md",
        f"echo x > {real}/obsidian/lloyd/MEMORY.md",
        f"echo x | tee -a {real}/obsidian/lloyd/MEMORY.md",
        f'''python3 -c "open('{real}/obsidian/lloyd/USER.md','w').write('x')"''',
        f"ln -sfn /tmp/evil {real}/obsidian/lloyd/MEMORY.md",
        f"git -C {real}/obsidian checkout -- lloyd/USER.md",
    ]
    still_blind = [cmd for cmd in shapes
                   if check_bash_command(cmd, cwd="/tmp") is not None]
    assert not still_blind, \
        "these shapes are the substrate's job; a predicate that now denies one " \
        f"means the deny-list grew here instead: {still_blind}"


# ---------------------------------------------------------------------------
# clause 3 — still a writable root: not whole-ro, not the bench profile
# ---------------------------------------------------------------------------

def test_the_profile_is_neither_whole_root_read_only_nor_the_bench_profile():
    argv = P.profile_argv("true", "/tmp")
    assert ("--bind" in argv and argv[argv.index("--bind") + 1] == "/"), \
        "the root is bound writable"
    # (source, destination) pairs of every --ro-bind, read out of the argv by
    # position: a `zip(argv, argv[1:])` membership test on a 3-tuple is always
    # True and asserted nothing, which is how this line first shipped.
    assert ("/", "/") not in _ro_binds(argv), \
        "the whole root must not be read-only: that is `_tool_sandbox.bwrap_argv`"
    from agent_mcp import _tool_sandbox as TS
    bench = TS.bwrap_argv("true", "/tmp")
    assert argv[:6] != bench[:6], "this profile is not the bench profile"
    for forbidden in ("--unshare-all", "--tmpfs", "--cap-drop"):
        assert forbidden not in argv, \
            f"{forbidden} is the bench profile's doing: it takes away the " \
            "network and the supervisord/X11 sockets every ordinary turn needs"


@skipif_no_bwrap
def test_writing_mkdir_git_commit_and_loopback_tcp_all_still_work(scratch_home):
    work = scratch_home.work
    checks = {
        "write a file": f"echo hello > {work}/note.md && cat {work}/note.md",
        "mkdir a tree": f"mkdir -p {work}/a/b/c && touch {work}/a/b/c/f",
        "git commit": (f"cd {work} && git init -q && git -c user.email=t@t "
                       f"-c user.name=t add -A && git -c user.email=t@t "
                       f"-c user.name=t commit -qm x && git log --oneline"),
        "loopback TCP connect": f"python3 {work}/tcp_probe.py",
    }
    (work / "tcp_probe.py").write_text(
        "import socket, threading\n"
        "srv = socket.socket(); srv.bind(('127.0.0.1', 0)); srv.listen(1)\n"
        "port = srv.getsockname()[1]\n"
        "def serve():\n"
        "    conn, _ = srv.accept(); conn.sendall(b'pong'); conn.close()\n"
        "threading.Thread(target=serve, daemon=True).start()\n"
        "cli = socket.create_connection(('127.0.0.1', port), timeout=10)\n"
        "assert cli.recv(4) == b'pong'\n"
        "print('tcp-ok')\n")
    broken = []
    for label, cmd in checks.items():
        rc, out = _run(cmd, str(work))
        if rc != 0:
            broken.append(f"{label}: rc={rc} {out[:200]}")
    assert not broken, "the profile refused something a normal turn needs:\n" + \
                       "\n".join(broken)
    assert (work / "note.md").read_text().strip() == "hello"


@skipif_no_bwrap
def test_the_temp_directory_is_the_host_one_not_a_mask(scratch_home):
    """`supervisorctl -c …/supervisord.conf` reaches its server at
    `/tmp/agent-supervisor.sock`, and the bench profile's `--tmpfs /tmp` is what
    makes that impossible. The property that keeps it working is that the
    sandbox shares the host's temp directory, so a file written inside is
    visible outside — asserted here rather than by poking a box-specific socket."""
    marker = f"lloyd-ps-{os.getpid()}-visible"
    rc, out = _run(f'''python3 -c "import tempfile, os; '''
                   f'''p = os.path.join(tempfile.gettempdir(), '{marker}'); '''
                   f'''open(p, 'w').write('here'); print(p)"''',
                   str(scratch_home.home))
    assert rc == 0, out
    host_copy = Path(out.strip().splitlines()[-1])
    assert host_copy.parent == Path(tempfile.gettempdir()), \
        f"the sandbox saw a different temp dir than the host: {host_copy.parent}"
    assert host_copy.read_text() == "here"
    host_copy.unlink()


# ---------------------------------------------------------------------------
# clause 4 — reaping and backgrounding survive the wrapper
# ---------------------------------------------------------------------------

# Process-name markers the reap assertions scan /proc for, derived from this
# pytest process's pid. Fixed numbers were tried and are wrong: an aborted run
# leaves its `sleep` behind (that is the failure under test), and the next run
# then reads the *previous* run's leak as its own — which is how a deliberate
# `--new-session` mutation turned these two nodes red in a later, unrelated
# batch. A pid-derived pair cannot collide with a run that is not alive.
_TICK = 70_000 + (os.getpid() % 4_000) * 2
MARK_A = f"{_TICK}.5"
MARK_B = f"{_TICK + 1}.5"
# Guarded so the test cannot pass without having run inside the substrate: if
# the wrapper were absent the guard exits 9, no timeout fires, and the
# "timed out" assertion below goes red.
_BG_SPAWN = ('test "$LLOYD_PATH_SANDBOX" = enforced || exit 9; '
             f'sleep {MARK_A} & sleep {MARK_B}; wait')


def _survivors(marker: str, seconds: float = 6.0) -> "list[tuple[str, str]]":
    """Poll for the spawned tree to go away, then report whatever is still alive."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        strays = _live_procs(marker)
        if not strays:
            return []
        time.sleep(0.2)
    return _live_procs(marker)


@skipif_no_bwrap
def test_a_foreground_timeout_leaves_no_live_member_of_the_spawned_group(scratch_home):
    # 6000 ms, not 1500: `timeout` under `_SECONDS_CEILING_MS` (5000) is read as
    # SECONDS by the #1994 rescue, so `timeout: 1500` means twenty-five minutes
    # and this test hangs instead of timing out.
    out = asyncio.run(B._bash({"command": _BG_SPAWN, "timeout": 6000}))
    assert "timed out" in out, \
        f"the command did not reach the timeout path, so nothing was reaped: {out[:200]}"
    survivors = _survivors(MARK_A) + _survivors(MARK_B)
    assert not survivors, \
        f"the sandboxed shell's children outlived its process group: {survivors}"


@skipif_no_bwrap
def test_a_cancelled_turn_leaves_no_live_member_of_the_spawned_group(scratch_home):
    async def scenario():
        task = asyncio.ensure_future(
            B._bash({"command": _BG_SPAWN, "timeout": 120_000}))
        await asyncio.sleep(1.5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return "cancelled"
        return f"finished early: {await task}"

    assert asyncio.run(scenario()) == "cancelled"
    survivors = _survivors(MARK_A) + _survivors(MARK_B)
    assert not survivors, \
        f"the sandboxed shell's children outlived the cancelled turn: {survivors}"


@skipif_no_bwrap
def test_run_in_background_still_returns_a_task_id_and_an_output_file(scratch_home,
                                                                      tmp_path,
                                                                      monkeypatch):
    """The id and the file are the whole contract of `run_in_background`, and
    `--die-with-parent` deliberately is *not* in the background profile — a
    background task is meant to outlive the call that returned its id."""
    monkeypatch.setattr(TR, "TASKS_DIR", tmp_path / "tasks")
    # The waiter's notification path is `_task_registry`'s own contract, already
    # covered by tests/test_task_subagent_notification_drain.py; this one is
    # about the spawn.
    monkeypatch.setattr(TR, "start_waiter", lambda record: None)
    sink = scratch_home.work / "bg-sink"
    raw = asyncio.run(B._bash({
        "command": f"echo $LLOYD_PATH_SANDBOX > {sink}",
        "run_in_background": True, "description": "substrate background probe"}))
    payload = json.loads(raw)
    assert payload.get("task_id"), raw
    assert payload.get("output_file"), raw
    assert Path(payload["output_file"]).exists(), payload["output_file"]
    assert "--die-with-parent" not in P.profile_argv("x", background=True)
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline and not sink.exists():
        time.sleep(0.25)
    assert sink.exists(), f"the background command never ran: {raw}"
    assert sink.read_text().strip() == "enforced", \
        "the background path ran without the substrate"


@skipif_no_bwrap
def test_a_foreground_command_reports_the_substrate_in_its_environment(scratch_home):
    out = asyncio.run(B._bash({"command": "echo marker=$LLOYD_PATH_SANDBOX"}))
    assert "marker=enforced" in out, out


# ---------------------------------------------------------------------------
# fail-open reporting, and the fail-open path through the real tool
# ---------------------------------------------------------------------------

def test_bash_still_runs_when_the_substrate_cannot_build(monkeypatch):
    """The open question in the policy sense is not this node's: this is that
    whatever it decides, Bash keeps working. `_path_sandbox.available()` False has
    to reach the spawn, run the command bare, and leave the marker unset — the
    behaviour this tool had before the substrate existed, not a half-built
    sandbox. `/state` then reads `fallback`, which `test_status_reports…` pins."""
    monkeypatch.setattr(P, "available", lambda: (False, "probe: no userns"))
    out = asyncio.run(B._bash({"command": "echo ran=$LLOYD_PATH_SANDBOX$((1+1))"}))
    assert "ran=2" in out, \
        f"the command did not run when the profile could not be built: {out[:200]}"


def test_status_reports_enforcing_or_fallback(monkeypatch):
    real_available = P.available
    st = P.status()
    assert set(st) >= {"enforcing", "fallback", "bwrap", "entries"}
    assert isinstance(st["entries"], list)

    # bwrap present but the namespace cannot be built: the branch a kernel hiccup
    # takes, distinct from the missing-binary one below.
    monkeypatch.setattr(P, "available", lambda: (False, "probe: no userns"))
    P.reset_probe_cache()
    try:
        st = P.status()
        assert st["enforcing"] is False and st["fallback"] is True
        assert "no userns" in (st["error"] or ""), \
            "the reason the substrate is off has to be on /state, not only in a log"
    finally:
        P.reset_probe_cache()

    # No bwrap at all: the probe's own answer is the missing binary, so the real
    # probe goes back in and only the lookup is stubbed.
    monkeypatch.setattr(P, "available", real_available)
    monkeypatch.setattr(P, "bwrap_path", lambda: None)
    P.reset_probe_cache()
    try:
        st = P.status()
        assert st["enforcing"] is False and st["fallback"] is True
        assert "not installed" in (st["error"] or "")
        argv, enforced = P.wrap("true", "/tmp")
        assert enforced is False, "fail-open must not pretend it built a profile"
        assert argv == ["/bin/sh", "-c", "true"], \
            "the fallback spawn is the bare shell this tool used before the " \
            "substrate existed, not a second guess at a sandbox"
    finally:
        P.reset_probe_cache()


def test_the_state_route_carries_the_protected_path_sandbox_key():
    """The boundary #2109 asks the state to be readable at is the HTTP route, not
    the helper: `GET /state` is what a health check and Mission Control read."""
    from agent_mcp import main as M

    resp = asyncio.run(M.state(None))
    payload = json.loads(resp.body)
    assert "protected_path_sandbox" in payload, sorted(payload)
    st = payload["protected_path_sandbox"]
    assert set(st) >= {"enforcing", "fallback", "bwrap"}, st
