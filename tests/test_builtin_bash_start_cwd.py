"""#1906: a Bash call that names no `cwd` must not start in the live checkout.

The mechanism: `_resolve_cwd` returns None for a missing argument, the foreground
path passed `cwd=None` to `create_subprocess_shell`, the background path used
`cwd or os.getcwd()`, and the bwrap path passed `cwd` to `bwrap_argv(command, cwd)`,
which falls back to `os.getcwd()` inside. All three inherited the MCP server's own
directory, which supervisord sets to `~/lloyd` — so the review grader for
SM_20260930_031934, handed `TMPDIR=~/lloyd-work/.t/05431072c3`, built its fixture at
the RELATIVE path `.t/05431072c3/r1873` and put nine files on live `main`; the
nightly uptake probe's `eval/uptake/classifier-report.json` arrived the same way. The
guardian noticed hourly and named no writer.

The fix is a precedence resolved ONCE in `_bash`, before the argument forks into the
three spawn paths, plus `app.session_cwd` — the module the three session kinds stamp
through, and the one this file's resolver delegates to rather than duplicating.

Two properties, pulling against each other:

* a session that recorded a start directory runs there, never in the checkout;
* a session that recorded nothing behaves EXACTLY as it does today, and an explicit
  absolute `cwd` still outranks both for every kind of session.

The second is load-bearing: most calls reach Bash with no `cwd`, so changing the
default for unstamped sessions would move every existing session's world.

A node that asserted on the value handed to a faked spawn would pin the argument, not the
behaviour — and the argument already looked correct when the bug shipped. So most of
these nodes spawn for real and read where the child opened. One does not, and the
distinction is stated rather than left to the reader:

* the FOREGROUND nodes spawn a real subprocess and read the `pwd` it writes to a file;
* the SANDBOX node runs a real child through `bwrap` and asserts on its `$PWD`, and
  separately on the single `--chdir` in the argv it was exec'd with;
* `test_the_start_directory_reaches_a_background_spawn` pins the `cwd` handed to
  `_spawn_background` and does NOT read a child's `pwd`, because that spawn is detached
  and its output goes to a log file no test should reap or wait on. It is sound only for
  the reason the fix exists: the precedence resolves once in `_bash`, upstream of the
  fork, so the value this node pins is the same value the foreground node proved inside a
  live process. If the background path ever resolves a directory for itself, this node
  stops being enough.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent_mcp import _tool_sandbox, builtin_bash  # noqa: E402
from app import session_cwd  # noqa: E402

SESSIONS = session_cwd.SESSIONS_DIR


@pytest.fixture(autouse=True)
def _stores_in_tmp(tmp_path, monkeypatch):
    """Sessions and scratch in tmp_path.

    `SESSIONS_DIR` and `SCRATCH_ROOT` are imported into `app.session_cwd` by value, so
    the module attributes are what has to move. A node that missed one would create
    directories under `~/lloyd-data/` from a unit test, which is the class of mistake
    this item is about.
    """
    monkeypatch.setattr(session_cwd, "SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr(session_cwd, "SCRATCH_ROOT", tmp_path / "session-cwd")
    return tmp_path


def _record(session_id: str, **fields) -> Path:
    """A session record, shaped the way `sessions_io.create_session` leaves one."""
    path = session_cwd.SESSIONS_DIR / f"{session_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"session_id": session_id, "platform": "worker",
                                **fields}), encoding="utf-8")
    return path


def _bound(monkeypatch, session_id: str | None) -> None:
    """Bind the call to a session, the way the server binds a real one.

    `get_bound_session` is imported into `builtin_bash`'s own namespace (line 41), so
    that name is the one that moves; patching `agent_mcp._shared.get_bound_session`
    would leave the tool reading the original and the node proving nothing.
    """
    monkeypatch.setattr(builtin_bash, "get_bound_session", lambda: session_id or "")


async def _child_cwd(out: Path) -> str:
    """Run a real `pwd` through the tool, into a file, and report the directory.

    Redirected rather than read from the result because `text_result` prefixes a
    `summary` caption onto the payload; the file holds the child's own bytes.
    """
    payload = await builtin_bash._bash({"command": f"pwd > {out}"})
    assert out.is_file(), payload
    return out.read_text(encoding="utf-8").strip()


# ── clause 1: a session that carries a start directory runs in it ──────

async def test_a_bash_call_with_no_cwd_runs_in_the_sessions_start_dir(
        tmp_path, monkeypatch):
    """`pwd` is inside the recorded directory and is not the live checkout root.

    Spawned for real: the quantity is a child process's working directory, which only
    a child can answer.
    """
    started = tmp_path / "outside" / "SM_TEST"
    _record("s1")
    session_cwd.stamp("s1", started)
    _bound(monkeypatch, "s1")

    where = await _child_cwd(tmp_path / "pwd.txt")
    assert where == str(started), where
    assert Path(where).is_dir()
    assert Path(where) != session_cwd.live_root().resolve(), (
        "the root this fix exists to get out of")
    assert started.is_dir(), "stamping creates the directory it names"


async def test_the_start_directory_reaches_the_sandboxed_command(tmp_path, monkeypatch):
    """The bwrap route gets the same directory, because the precedence resolves once.

    The sandbox takes no `cwd` keyword: `bwrap_argv(command, cwd)` turns it into the
    child's `--chdir` (`agent_mcp/_tool_sandbox.py:171`), so `--chdir` IS the seam — and
    the node asserts on the argv the child was actually exec'd with, not on the argument
    handed to the builder. That difference is not cosmetic: the builder rewrites a start
    under one of its tmpfs directories (`_TMPFS_DIRS`,
    `agent_mcp/_tool_sandbox.py:67`) to `--chdir /`
    (`agent_mcp/_tool_sandbox.py:166-169`), and the bubble mounts those directories
    empty anyway. So a node that pinned the argument could pass on a call whose bubble
    opened at the root of the filesystem, which is what the first version of this node
    did: `pytest`'s `tmp_path` sits under `/tmp`, one of those directories. The scratch
    stamped here lives under `~`, where the real scratch root (`~/lloyd-data`) lives
    too. It is removed in a `finally`, and a run killed before that `finally` costs the
    home directory one dotdir, which the `exists()` check at the top of this node clears on
    the next run — `~` is used because nowhere else under a pytest temporary directory
    survives the builder's tmpfs rewrite to `--chdir /`.
    """
    started = Path.home() / ".lloyd-start-cwd-sandbox-probe"
    if started.exists():
        shutil.rmtree(started)
    try:
        _record("s2")
        session_cwd.stamp("s2", started)
        _bound(monkeypatch, "s2")
        assert started.resolve() == started, "the scratch must be its own physical path"

        seen: list[tuple[str, str | None, list[str]]] = []
        real_argv = _tool_sandbox.bwrap_argv

        def _spy(command: str, cwd):
            argv = real_argv(command, cwd)
            seen.append((command, cwd, argv))
            return argv

        monkeypatch.setattr(_tool_sandbox, "bwrap_argv", _spy)
        monkeypatch.setattr(_tool_sandbox, "sandbox_available", lambda: (True, ""))
        # The flag is a ContextVar and a ContextVar's `get` is read-only, so it is SET
        # rather than replaced — and reset in a `finally`, because a leaked sandbox flag
        # would send every later Bash call in this pytest worker through bwrap.
        token = _tool_sandbox.current_sandboxed.set(True)
        # The child reports where it opened, so the node is not reading only the
        # argument meant to move it: inside a bubble nothing but `--chdir` can change
        # `$PWD`, so a dropped argument or the builder's tmpfs rewrite both surface here.
        probe = (f'test "$PWD" = "{started}" && echo SANDBOX-PWD-OK '
                 f'|| echo "SANDBOX-PWD-NO [$PWD]"')
        try:
            out = await builtin_bash._bash({"command": probe})
        finally:
            _tool_sandbox.current_sandboxed.reset(token)
    finally:
        shutil.rmtree(started, ignore_errors=True)

    assert "SANDBOX-PWD-OK" in out, (
        f"the bubble's own $PWD was not the session's scratch: {out[:300]}")
    assert len(seen) == 1 and seen[0][0] == probe, seen
    assert seen[0][1] == str(started), seen
    argv = seen[0][2]
    chdirs = [i for i, part in enumerate(argv) if part == "--chdir"]
    assert len(chdirs) == 1, f"exactly one --chdir in the spawn argv: {chdirs}"
    assert argv[chdirs[0] + 1] == str(started), (
        f"the bubble was told to open at {argv[chdirs[0] + 1]!r}, not the "
        f"{started} its session record names")
    # Elements, never a substring of a joined argv. `bwrap_argv` ends its vector with
    # `["--", "/bin/bash", "-c", command]` (`agent_mcp/_tool_sandbox.py:170-173`), so the
    # command is one element of what gets exec'd; asserting `"… " + command in
    # " ".join(argv)` would also be true of an argv that had folded the whole vector into
    # a single shell string, which is a different and far more breakable spawn.
    assert argv[-4:] == ["--", "/bin/bash", "-c", probe], (
        f"the bubble execs {argv[-4:]!r}, not the shell-and-command tail the builder "
        "is documented to append")


async def test_the_start_directory_reaches_a_background_spawn(tmp_path, monkeypatch):
    """`run_in_background` inherits it too — one local feeds that call as well.

    A detached child is where a stray matters most: it outlives its turn, so by the
    time datawatch alerts the round is closed and no record names it.
    """
    started = tmp_path / "outside" / "SM_TEST"
    _record("s3")
    session_cwd.stamp("s3", started)
    _bound(monkeypatch, "s3")

    seen: list = []

    async def _fake_spawn(command, description, cwd=None, env=None):
        seen.append(cwd)
        return json.dumps({"task_id": "t1", "status": "running"})

    monkeypatch.setattr(builtin_bash, "_spawn_background", _fake_spawn)
    out = await builtin_bash._bash({"command": "sleep 30", "run_in_background": True,
                                    "description": "a pinned background child"})
    assert json.loads(out)["task_id"] == "t1", out
    assert seen == [str(started)], seen


# ── clause 2: nothing recorded changes nothing; an argument outranks ────

async def test_an_unstamped_session_keeps_todays_default(tmp_path, monkeypatch):
    """A session record with no `start_cwd` behaves exactly as it did before.

    Asserted against the process's own directory — the value `os.getcwd()` and
    `create_subprocess_shell(cwd=None)` both resolve to — because "the server's own
    directory" is the spec for this case, not a fallback being tolerated.
    """
    _record("s4")                      # a record, and no `start_cwd` in it
    _bound(monkeypatch, "s4")
    monkeypatch.chdir(tmp_path)
    assert session_cwd.resolved_for("s4") is None

    where = await _child_cwd(tmp_path / "pwd.txt")
    assert Path(where) == tmp_path, where


async def test_a_call_bound_to_no_session_keeps_todays_default(tmp_path, monkeypatch):
    """No session id means nothing to look up, so nothing moves.

    This is the tool as every interactive chat turn sees it. Moving it would not be a
    safety gain; it would be an unrequested change of working directory for the
    majority of calls.
    """
    _bound(monkeypatch, None)
    monkeypatch.chdir(tmp_path)
    where = await _child_cwd(tmp_path / "pwd.txt")
    assert Path(where) == tmp_path, where


async def test_an_explicit_absolute_cwd_outranks_the_start_directory_for_every_kind(
        tmp_path, monkeypatch):
    """A `cwd` the caller named wins over the one recorded for it, three kinds deep.

    One round turn with its worktree stamped, one worker session with a scratch that
    has since been swept, one session stamped by nothing. All three are the same
    resolver, and the interesting claim is that none of them special-cases the
    argument away — an explicit path is how a session opts back toward the tree, and
    `_resolve_cwd`'s absolute-and-isdir checks are unaffected by any of this.

    Each kind is asserted on all three spawn paths, because the precedence is applied
    once and a leg that stopped honouring it would only show up for the sessions that
    take that path. Foreground is the child's own `pwd`, written where the caller can
    read it; background is the `cwd` given to `_spawn_background`; the sandbox is the
    `cwd` given to `bwrap_argv`, which is the last place the named directory and the
    recorded one are still distinguishable — what bwrap then does with it is pinned by
    `test_the_start_directory_reaches_the_sandboxed_command`.
    """
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    started = tmp_path / "outside" / "SM_TEST"
    outs = tmp_path / "pwd-out"
    outs.mkdir()

    seen_bg: list = []
    seen_sbx: list = []

    async def _fake_bg(command, description, cwd=None, env=None):
        seen_bg.append(cwd)
        return json.dumps({"task_id": "t", "status": "running"})

    def _spy_argv(command, cwd):
        seen_sbx.append((command, cwd))
        return ["/bin/true"]

    for session_id, kind in (("autocode-round", "stamped"),
                             ("worker-turn", "swept"),
                             ("legacy-turn", "unstamped")):
        _record(session_id)
        if kind == "stamped":
            session_cwd.stamp(session_id, started)
        elif kind == "swept":
            session_cwd.stamp(session_id, started)
            shutil.rmtree(started)          # the scratch is gone; the record is not
        _bound(monkeypatch, session_id)
        monkeypatch.setattr(builtin_bash, "_spawn_background", _fake_bg)
        monkeypatch.setattr(_tool_sandbox, "bwrap_argv", _spy_argv)
        monkeypatch.setattr(_tool_sandbox, "sandbox_available", lambda: (True, ""))

        where = outs / kind
        out = await builtin_bash._bash({"command": f"pwd > {where}",
                                        "cwd": str(elsewhere)})
        assert where.is_file(), (kind, out)
        assert where.read_text(encoding="utf-8").strip() == str(elsewhere), (
            f"{kind}: the foreground shell started in {where.read_text()!r}, "
            f"not the {elsewhere} the call named")
        await builtin_bash._bash({"command": "pwd", "cwd": str(elsewhere),
                                  "run_in_background": True, "description": "d"})
        # The flag is a ContextVar whose `get` is read-only, so it is SET here and
        # reset in a `finally`: a leaked flag would send later nodes' Bash to bwrap.
        _tok = _tool_sandbox.current_sandboxed.set(True)
        try:
            await builtin_bash._bash({"command": "true", "cwd": str(elsewhere)})
        finally:
            _tool_sandbox.current_sandboxed.reset(_tok)
        assert seen_bg[-1] == str(elsewhere), (kind, seen_bg)
        assert seen_sbx[-1] == ("true", str(elsewhere)), (kind, seen_sbx)


# ── the resolver's own refusals ────────────────────────────────────────

def test_a_swept_scratch_is_rebuilt_from_the_record_not_inherited_away(tmp_path):
    """A deleted scratch is re-created from the record, not abandoned to the default.

    `resolved_for` is a usable-or-nothing gate for a different reason than it looks: a
    dead path handed to `create_subprocess_shell` fails the call outright, so the
    resolver may not return a directory that cannot be entered — and `None`, the other
    option, means "inherit the server's directory", which for every worker IS the live
    checkout, the one place this change stops writes landing. With the record naming
    its own scratch, re-creating it is the only answer that is both.
    """
    started = tmp_path / "outside" / "SM_TEST"
    _record("s5")
    session_cwd.stamp("s5", started)
    assert started.is_dir()
    assert session_cwd.resolved_for("s5") == str(started)

    shutil.rmtree(started)
    assert session_cwd.resolved_for("s5") == str(started)
    assert started.is_dir(), "reading a session's start directory rebuilds it"


def test_a_record_that_cannot_be_read_is_read_as_no_start_directory(tmp_path):
    """A half-written or unparseable session file is a miss, never a guess.

    The resolver runs on every Bash call, so a reader that raised on the truncated
    JSON a crash mid-`create_session` leaves would take the tool down for that
    session instead of leaving it where it was.
    """
    path = _record("s6")
    assert session_cwd.resolved_for("s6") is None
    path.write_text("{not json", encoding="utf-8")
    assert session_cwd.resolved_for("s6") is None


def test_the_stored_path_is_written_resolved_so_pwd_assertions_agree(tmp_path):
    """`stamp` stores the physical path, so a stored value equals what `pwd` says.

    `pwd` prints the resolved directory; `str(Path(symlink))` does not resolve. On a
    box where the data root is a symlink, a record written in the link's spelling and
    a child reporting the real one make every assertion above read as "the child
    started somewhere else" — the failure shape of the bug, manufactured by the test
    suite. So the node hands `stamp` a symlink and requires the record to name the
    target: a `stamp` that stored its argument verbatim fails here, and would not fail
    a node that compared `.resolve()` to itself.
    """
    target = tmp_path / "scratch-target" / "SM_TEST"
    target.mkdir(parents=True)
    link = tmp_path / "scratch-link"
    link.symlink_to(target, target_is_directory=True)
    assert link.resolve() == target.resolve() and link != target

    _record("s7")
    stored = session_cwd.stamp("s7", link)
    assert stored is not None
    assert stored == target.resolve(), f"stored the link, not the target: {stored}"
    assert session_cwd.read("s7") == str(target.resolve()), session_cwd.read("s7")
    assert session_cwd.read("s7") != str(link), (
        "a record in the link's spelling disagrees with the child's `pwd`")


def test_a_start_directory_inside_the_checkout_is_refused(tmp_path):
    """Stamping the tree is the bug, so it is refused rather than honoured.

    A storer that accepted the checkout would let a caller "fix" the problem by
    stamping `~/lloyd` and reproduce the incident exactly. The refusal is what makes
    the caller that hands in the tree do something other than what it asked.
    """
    live = tmp_path / "live-checkout"
    (live / "sub").mkdir(parents=True)
    elsewhere = tmp_path / "outside"
    elsewhere.mkdir()
    _record("s8")

    original = session_cwd.SESSIONS_DIR / "marker"
    monkey_root = session_cwd.live_root
    session_cwd.live_root = lambda: live
    try:
        assert session_cwd.stamp("s8", live) is None
        assert session_cwd.stamp("s8", live / "sub") is None
        assert "start_cwd" not in json.loads(
            (session_cwd.SESSIONS_DIR / "s8.json").read_text(encoding="utf-8"))
        assert session_cwd.stamp("s8", elsewhere) == elsewhere.resolve()
        assert session_cwd.resolved_for("s8") == str(elsewhere.resolve())
        assert not original.exists()
    finally:
        session_cwd.live_root = monkey_root


def test_stamping_never_drops_a_key_another_writer_owns(tmp_path):
    """`start_cwd` is added to the session record, not written over it.

    The session JSON belongs to `app.sessions_io` and is written from several places;
    a stamper that rewrote the file from its own idea of the shape would delete a
    session's `platform` and change how it is routed.
    """
    _record("s9", platform="worker", inner_voice=False, message_count=3,
            title="nightly thing")
    session_cwd.stamp("s9", tmp_path / "outside" / "SM_TEST")
    meta = json.loads((session_cwd.SESSIONS_DIR / "s9.json").read_text(encoding="utf-8"))
    assert meta["platform"] == "worker" and meta["message_count"] == 3, meta
    assert meta["title"] == "nightly thing", meta
    assert "start_cwd" in meta


def test_a_session_id_that_cannot_be_a_filename_cannot_escape_the_store(tmp_path):
    """The session id reaches a filesystem path, so it is flattened, not trusted.

    The ids the harness mints are already flat (`20260930_170921_autocode_a4e9`), but
    `path_for` is handed a string from a record, and `../../x` reaching
    `SCRATCH_ROOT / name` would be a directory anywhere.
    """
    for evil in ("../../etc/passwd", "a/b/c", ""):
        name = session_cwd.safe_name(evil)
        # What must be true is that the result is ONE flat component, not that it
        # lacks the two characters `.` — `.._.._etc_passwd` still reads as a single
        # file name, which is the whole property.
        assert "/" not in name and chr(92) not in name, (evil, name)
        assert Path(name).name == name, f"{evil!r} sanitised to a path: {name!r}"
        assert session_cwd.path_for(evil).is_relative_to(session_cwd.SCRATCH_ROOT)
