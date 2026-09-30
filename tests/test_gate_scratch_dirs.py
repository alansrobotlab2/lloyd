"""The gate's scratch directories: created, never computed; removed, never guessed.

On 2026-09-29 `Gate.rung_static` built its scratch path as
`Path(_run(["mktemp", "-d"]).stdout.strip())`. `/tmp` had no inodes left, so
mktemp failed with an EMPTY stdout, `Path("")` was the gate's working directory
(`~/lloyd`, inherited from the aggregator), and the rung's `finally:
shutil.rmtree(...)` deleted the production tree. The 2026-09-22 deletion fits the
same shape. These tests pin the two rules that make it impossible:

  * a scratch directory is something `tempfile.mkdtemp` CREATED under the round
    dir, or an exception — never a name read off a subprocess;
  * a scratch directory is removed only if it is one of those, and the removal
    refuses the working directory, the home, the live tree, the work root and
    anything that contains them.
"""
from __future__ import annotations

import errno
import re
import subprocess
from pathlib import Path

import pytest

from scripts.automod import gate as G
from scripts.automod import worktree as W

ROOT = Path(__file__).resolve().parent.parent
#: The packages whose Python may run with `~/lloyd` as its working directory.
SCANNED = ("scripts", "app", "agent_mcp", "workers")
#: A quoted `mktemp` argv element. Prose in a docstring is not a subprocess.
MKTEMP_LITERAL = re.compile(r"""["']mktemp["']""")
ROUND = "SM_TEST_SCRATCH"


@pytest.fixture
def work_root(tmp_path, monkeypatch):
    root = tmp_path / "work"
    monkeypatch.setattr(W, "WORK_ROOT", root)
    return root


def _enospc(*_a, **_k):
    raise OSError(errno.ENOSPC, "No space left on device")


def test_no_path_in_the_tree_is_read_off_a_mktemp_subprocess():
    hits: list[str] = []
    for top in SCANNED:
        for py in (ROOT / top).rglob("*.py"):
            try:
                text = py.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for n, line in enumerate(text.splitlines(), 1):
                if MKTEMP_LITERAL.search(line):
                    hits.append(f"{py.relative_to(ROOT)}:{n}: {line.strip()}")
    assert not hits, (
        "a path built from mktemp's stdout is the working directory when mktemp "
        "fails (2026-09-29: it deleted ~/lloyd). Use tempfile.mkdtemp, which "
        "raises instead:\n" + "\n".join(hits))


def test_scratch_dir_is_a_fresh_absolute_directory_under_the_round(work_root):
    path = G._scratch_dir(ROUND, "probe")
    assert path.is_absolute()
    assert path.is_dir() and not any(path.iterdir())
    assert path.parent == work_root / ROUND / Path(*G.SCRATCH_SUBDIR)
    other = G._scratch_dir(ROUND, "probe")
    assert other != path, "two scratch dirs for one purpose must not collide"


def test_scratch_dir_raises_when_it_cannot_create_one(work_root, monkeypatch):
    """The whole incident: a failure to make the directory must be an exception
    `Gate._rung` records as a FAILED rung, never a string a later rmtree eats."""
    monkeypatch.setattr(G.tempfile, "mkdtemp", _enospc)
    with pytest.raises(OSError) as exc:
        G._scratch_dir(ROUND, "probe")
    assert exc.value.errno == errno.ENOSPC


def test_drop_scratch_removes_only_what_scratch_dir_made(work_root):
    path = G._scratch_dir(ROUND, "probe")
    (path / "a.py").write_text("x = 1\n", encoding="utf-8")
    G._drop_scratch(path, ROUND)
    assert not path.exists()
    assert path.parent.is_dir(), "the scratch parent stays; only the child goes"


@pytest.mark.parametrize("target", ["", ".", "..", "relative/path"])
def test_drop_scratch_refuses_relative_and_empty_paths(work_root, tmp_path, monkeypatch, target):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    sentinel = cwd / "server.py"
    sentinel.write_text("# still here\n", encoding="utf-8")
    monkeypatch.chdir(cwd)
    with pytest.raises(RuntimeError, match="refusing"):
        G._drop_scratch(target, ROUND)
    assert sentinel.exists()


def test_drop_scratch_refuses_every_protected_root(work_root, tmp_path, monkeypatch):
    """The roots a wrong target has actually been on this box, and the ones it
    could be next: cwd, home, the live tree, the work root, the round dir, the
    scratch parent itself, and an ordinary directory that is none of those."""
    cwd = tmp_path / "cwd"
    home = tmp_path / "home"
    live = tmp_path / "live"
    elsewhere = tmp_path / "elsewhere"
    for d in (cwd, home, live, elsewhere):
        d.mkdir()
        (d / "keep.txt").write_text("keep\n", encoding="utf-8")
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(G, "LIVE_ROOT", live)
    G._scratch_dir(ROUND, "probe")            # so the round and its parent exist
    parent = work_root / ROUND / Path(*G.SCRATCH_SUBDIR)
    for target in (cwd, home, live, work_root, work_root / ROUND, parent, elsewhere,
                   tmp_path, Path("/")):
        with pytest.raises(RuntimeError, match="refusing"):
            G._drop_scratch(target, ROUND)
    for d in (cwd, home, live, elsewhere):
        assert (d / "keep.txt").exists(), f"{d} lost its contents"


def test_static_rung_fails_closed_when_no_scratch_dir_can_be_made(
        work_root, tmp_path, monkeypatch):
    """`rung_static` with mktemp-style failure injected at the one place a
    directory is made: the rung is recorded FAILED and nothing in the working
    directory or the worktree is touched. This is the 2026-09-29 run, replayed
    with the fix in place."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "server.py").write_text("# production\n", encoding="utf-8")
    monkeypatch.chdir(cwd)
    wt = tmp_path / "worktree"
    (wt / "app").mkdir(parents=True)
    (wt / "app" / "changed.py").write_text("x = 1\n", encoding="utf-8")

    def fake_run(cmd, cwd=None, env=None, timeout=900.0):
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(G, "_run", fake_run)
    monkeypatch.setattr(G, "_pyflakes", lambda python, root, files: set())
    monkeypatch.setattr(G.tempfile, "mkdtemp", _enospc)

    g = G.Gate.__new__(G.Gate)
    g.round_id = ROUND
    g.worktree = wt
    g.base = "0" * 40
    g.live = tmp_path / "live"
    g.python = tmp_path / "python"
    g.home_isolation = "not requested"
    g.report = G.GateReport(round_id=ROUND, base=g.base, head="1" * 40,
                            changed_paths=["app/changed.py"])

    assert g._rung("static", g.rung_static) is False
    rung = g.report.rungs[-1]
    assert rung.name == "static" and rung.ok is False
    assert "No space left on device" in rung.detail
    assert (cwd / "server.py").exists(), "the working directory was touched"
    assert (wt / "app" / "changed.py").exists(), "the worktree was touched"
    assert sorted(p.name for p in cwd.iterdir()) == ["server.py"], (
        "the rung wrote into the working directory")


#: Chromium's singleton socket, relative to TMPDIR: `/.org.chromium.Chromium.`
#: plus six random characters, then `/SingletonSocket`.
CHROMIUM_SOCKET_TAIL = "/.org.chromium.Chromium.abcdef/SingletonSocket"
SUN_PATH_MAX = 107


def test_production_child_tmpdir_leaves_room_for_chromiums_socket():
    """The 2026-09-29 regression: `<round>/gate-state/tmp` was 64 bytes, 45 more
    for Chromium's socket is 109, and every Playwright test's browser aborted.
    Computed against the real work root and a real-length round id."""
    path = G._child_tmp_root() / ("0" * 10)
    assert len(str(path)) <= G.MAX_CHILD_TMPDIR
    assert len(str(path)) + len(CHROMIUM_SOCKET_TAIL) <= SUN_PATH_MAX
    assert G.MAX_CHILD_TMPDIR + len(CHROMIUM_SOCKET_TAIL) <= SUN_PATH_MAX


def test_the_kernel_accepts_a_socket_at_the_longest_allowed_tmpdir():
    """Not arithmetic about the limit — a bind the kernel judges. A directory
    exactly `MAX_CHILD_TMPDIR` bytes long takes Chromium's socket; one as long
    as the 09-29 TMPDIR does not."""
    import socket
    import tempfile
    base = tempfile.mkdtemp(prefix="s", dir="/tmp")
    try:
        def at(length):
            d = Path(base) / ("x" * (length - len(base) - 1))
            sock_path = str(d) + CHROMIUM_SOCKET_TAIL
            Path(sock_path).parent.mkdir(parents=True, exist_ok=True)
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                s.bind(sock_path)
                return True
            except OSError:
                return False
            finally:
                s.close()
        assert at(G.MAX_CHILD_TMPDIR) is True
        assert at(len("/home/alansrobotlab/lloyd-work/SM_20260930_003324/gate-state/tmp")) is False
    finally:
        import shutil
        shutil.rmtree(base, ignore_errors=True)


@pytest.fixture
def short_work_root(monkeypatch):
    import tempfile, shutil
    root = Path(tempfile.mkdtemp(prefix="w", dir="/tmp"))
    monkeypatch.setattr(W, "WORK_ROOT", root)
    yield root
    shutil.rmtree(root, ignore_errors=True)


def test_child_env_gives_a_short_on_disk_tmpdir_and_the_gate_removes_it(short_work_root):
    g = G.Gate.__new__(G.Gate)
    g.round_id = ROUND
    g.worktree = short_work_root / ROUND / "home" / "lloyd"
    g.home_isolation = "not requested"
    env = g._child_env()
    tmp = Path(env["TMPDIR"])
    assert tmp.is_dir() and tmp.parent == short_work_root / G.CHILD_TMP_DIRNAME
    assert len(str(tmp)) <= G.MAX_CHILD_TMPDIR
    assert not str(tmp).startswith("/tmp/") or str(short_work_root).startswith("/tmp/")
    assert G._drop_child_tmpdir(tmp) is True and not tmp.exists()


def test_a_work_root_too_long_for_a_short_tmpdir_keeps_the_system_one(work_root):
    """Pytest's own tmp_path makes the work root far over the bound: the child
    then gets no TMPDIR at all rather than a long one."""
    g = G.Gate.__new__(G.Gate)
    g.round_id = ROUND
    g.worktree = work_root / ROUND / "home" / "lloyd"
    g.home_isolation = "not requested"
    assert "TMPDIR" not in g._child_env()


def test_drop_child_tmpdir_refuses_anything_it_did_not_make(short_work_root, tmp_path):
    made = G._child_tmpdir(ROUND)
    other = short_work_root / G.CHILD_TMP_DIRNAME / "not-a-hash"
    other.mkdir()
    link = short_work_root / G.CHILD_TMP_DIRNAME / "abcdef0123"
    link.symlink_to(tmp_path)
    (tmp_path / "keep").write_text("x")
    for target in (other, link, short_work_root, short_work_root / G.CHILD_TMP_DIRNAME,
                   tmp_path, None, ""):
        assert G._drop_child_tmpdir(target) is False
    assert other.exists() and (tmp_path / "keep").exists() and made.exists()


def test_a_killed_gates_tmpdir_is_pruned_once_its_round_is_gone(short_work_root):
    (short_work_root / "SM_OLD").mkdir()
    old = G._child_tmpdir("SM_OLD")
    (short_work_root / "SM_OLD").rmdir()           # the round was cleaned up
    live_round = short_work_root / "SM_LIVE"
    live_round.mkdir()
    live = G._child_tmpdir("SM_LIVE")              # pruning runs here
    assert not old.exists()
    assert live.exists()
