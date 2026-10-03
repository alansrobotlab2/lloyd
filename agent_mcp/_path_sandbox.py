"""Protected paths enforced by the kernel, for every session that is NOT read-only.

The Bash lane has two string guards upstream of a shell — the catastrophic
regex table in `app/harness/safety.py` and `check_bash_write_denied` — and both
are a parser of shell text. A parser is beaten by the next spelling, and the
list of shapes it has no grammar for is measured, not imagined: at `cd201f9a`
`check_bash_command` returned `None` for `truncate -s 0 <SOUL.md>`,
`rm <SOUL.md>`, `find … -delete`, `perl -ni -e`, `ln -sfn <MEMORY.md>`,
`shutil.rmtree('…/obsidian/lloyd')` and every write into
`~/obsidian/lloyd/USER.md` / `MEMORY.md`, which are in no deny-set at all.
Every new verb is therefore a new regex, and a missed one is invisible until it
is used.

So this module enforces the property below the parser instead of beside it:
one `--ro-bind <entry> <entry>` per protected entry, and the kernel answers
`EROFS` to `open(O_WRONLY)`, `EACCES`/`EROFS` to a truncate and `EBUSY` to an
`unlink`/`rename` — for a shape nobody recognised, and for one nobody has
invented yet. The list is `PROTECTED_SHELL_RO_ROOTS`, owned by
`app.harness.protected_paths` and imported here as that one object.

Two things make this profile *different* from the read-only one in
`agent_mcp/_tool_sandbox.py`, which must not be reused here (#582 measured the
reuse first): that profile is `--ro-bind / /` plus `--tmpfs /tmp /run /var/tmp`
plus `--unshare-all --cap-drop ALL`, which removes the network, the supervisord
socket (`/tmp/agent-supervisor.sock`) and the X11/systemd sockets every ordinary
turn uses. The two profiles are keyed to opposite session classes: `_tool_sandbox`
owns `bench_…`/`pt-eval-…`, which may observe but never change this machine, and
this one owns every session that is allowed to change it, changing exactly the
entries on the list.

* **`--bind / /`, not `--ro-bind / /`.** Writes, `mkdir`s, `git commit`s and
  loopback TCP must keep working; `supervisorctl`, `git`, `uv` and CUDA come
  with them.
* **`--dev-bind /dev /dev`, not `--dev /dev` and not nothing.** Measured on this
  box: with no `/dev` argument `git` cannot start at all (`could not open
  '/dev/null': Permission denied`, because devtmpfs belongs to the parent user
  namespace), and `--dev /dev` fixes `/dev/null` but has no `/dev/nvidia*`, so
  `nvidia-smi` breaks. `--dev-bind` re-exposes the host nodes with the host's
  own permissions, which is the status quo: today there is no sandbox at all.
* **One bind per entry, a file bound as the file.** A parent-directory bind of
  `~/obsidian/lloyd/` would make `.consolidate-lock` and a sibling
  `MEMORY.md.backup.<date>` unwritable, and the failure would be silent: the
  dream-consolidation Phase-0 lock probe prints `LOCK_MISSING … proceeding` and
  its 24 h gate then passes every run. Files bind as files, so the two
  scheduled shell writes #2108 named survive — pinned by
  `test_the_scheduled_vault_writes_stay_writable`.
* **An absent entry is skipped, not fatal.** `~/.openclaw` is a deny-set member
  that does not exist on this box, and a bind needs a target: `_host_socket_paths`
  in `_tool_sandbox.py` records one un-creatable mount point breaking *every*
  Bash call while `/state` still read `bwrap: true`.
* **No `--new-session`.** It puts the sandboxed shell in a new session, so
  `os.killpg` in `_kill_proc_tree` reaches only `bwrap` and the command's own
  children survive a timeout as strays — measured, and clause 4 of #2109 is
  exactly "a timeout and a cancel leave no live member of the spawned process
  group". Without it the whole tree inherits the caller's pgid, which is the
  semantics the kill path already relies on.
* **`--die-with-parent` on the foreground path only** — a background task is
  meant to outlive the call that returned its id.
* **It fails OPEN, and says so.** No bwrap, or a bwrap that cannot build a
  namespace here, means the command runs bare with a WARNING and
  `/state` reads `protected_path_sandbox.fallback: true`. Refusing Bash for
  every attended and worker session on a kernel hiccup is an outage of the whole
  agent, and #582's policy question (mirror `_tool_sandbox.refusal` and fail
  closed?) is still open with a person on it.

The shell itself is chosen to match what each path runs today — `/bin/sh` for a
foreground command (`asyncio.create_subprocess_shell` is `/bin/sh -c`), `bash`
for a background one — so this changes the mount table and nothing else.
"""

from __future__ import annotations

import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import threading
import time

from app.harness.protected_paths import (
    PROTECTED_SHELL_RO_ROOTS,
    protected_shell_ro_paths,
)

logger = logging.getLogger("lloyd-path-sandbox")

#: Re-exported so a caller can name the list without importing the deny-set
#: module. Tests assert this IS the object `protected_paths` owns (clause 5).
__all__ = ["PROTECTED_SHELL_RO_ROOTS", "available", "profile_argv", "status",
           "wrap"]

#: Set inside the sandbox so a command can tell where it is running. Not a
#: security property — a command cannot unset it — just a legible marker.
ENV_MARKER = "LLOYD_PATH_SANDBOX"

#: `create_subprocess_shell` runs `/bin/sh -c`; `_spawn_background` execs
#: `bash -c`. The substrate reproduces each so wrapping is not a semantic
#: change riding along with the mount change.
_FOREGROUND_SHELL = "/bin/sh"
_BACKGROUND_SHELL = "bash"

_probe_lock = threading.Lock()
_probe: dict[str, object] = {"ok": None, "at": 0.0, "error": ""}
_PROBE_RETRY_S = 60.0


def bwrap_path() -> str | None:
    return shutil.which("bwrap")


def ro_bind_argv(entries: "list[str] | None" = None) -> list[str]:
    """One `--ro-bind <realpath> <realpath>` per entry that exists on disk.

    Resolved by the caller through `protected_shell_ro_paths()`, which skips an
    entry that is absent — see the module docstring for why one un-bindable
    entry must never reach `bwrap`.
    """
    out: list[str] = []
    for entry in protected_shell_ro_paths() if entries is None else entries:
        out += ["--ro-bind", entry, entry]
    return out


def profile_argv(command: str, cwd: str | None = None, *,
                 background: bool = False) -> list[str]:
    """The bwrap argv that runs `command` with the protected entries read-only.

    Order is load-bearing: `/` first so everything exists, then `/dev` and
    `/proc`, then the read-only binds on top of the writable root.
    """
    bwrap = bwrap_path() or "bwrap"
    start = cwd or os.getcwd()
    if not os.path.isdir(start):
        start = "/"
    argv = [bwrap,
            "--bind", "/", "/",
            "--dev-bind", "/dev", "/dev",
            "--proc", "/proc"]
    argv += ro_bind_argv()
    if not background:
        # A foreground command belongs to the turn that asked for it: if this
        # process dies, it dies. A background task outlives its call by design.
        argv += ["--die-with-parent"]
    argv += ["--setenv", ENV_MARKER, "enforced",
             "--chdir", start, "--",
             _BACKGROUND_SHELL if background else _FOREGROUND_SHELL,
             "-c", command]
    return argv


def wrap(command: str, cwd: str | None = None, *,
         background: bool = False) -> "tuple[list[str], bool]":
    """`(argv, enforced)` — the spawn this caller should perform.

    `enforced` False is the fail-open path: the argv is a plain shell spawn of
    the same command, byte-identical to what this tool did before the substrate
    existed. The caller must report it (`/state` reads `.fallback`); the policy
    of refusing outright is the open question this module does not decide.
    """
    ok, err = available()
    if not ok:
        logger.warning("path sandbox unavailable (%s); running Bash UNSANDBOXED"
                       " — protected-path enforcement is the string guard only",
                       err)
        shell = _BACKGROUND_SHELL if background else "/bin/sh"
        return [shell, "-c", command], False
    return profile_argv(command, cwd, background=background), True


def _probe_command(entries: "list[str] | None" = None) -> str:
    """A command that passes only if the profile is BOTH working and biting.

    Exit 3 = a write that must succeed failed, so bwrap is not really building
    a usable root (the mistake `_tool_sandbox`'s probe guards against from the
    other side). Exit 4 = a protected entry is still writable, so the binds did
    not take — a profile that reported green while binding nothing would
    otherwise be indistinguishable from one that is enforcing.
    """
    probe = os.path.join(tempfile.gettempdir(),
                         f"lloyd-path-sandbox-probe-{os.getpid()}")
    parts = [f'if ! touch {shlex.quote(probe)} 2>/dev/null; then exit 3; fi;',
             f'rm -f {shlex.quote(probe)};']
    for entry in protected_shell_ro_paths() if entries is None else entries:
        parts.append(f'test -w {shlex.quote(entry)} && exit 4;')
    parts.append('exit 0')
    return " ".join(parts)


def available() -> tuple[bool, str]:
    """Whether the profile can be built and actually bites here.

    Cached on success; a failure is re-probed after a minute so a transient
    namespace hiccup does not strand the box in fail-open until restart.
    """
    with _probe_lock:
        now = time.monotonic()
        if _probe["ok"] is True:
            return True, ""
        if _probe["ok"] is False and now - float(_probe["at"]) < _PROBE_RETRY_S:
            return False, str(_probe["error"])
        if not bwrap_path():
            ok, err = False, "bwrap is not installed"
        else:
            try:
                proc = subprocess.run(profile_argv(_probe_command()),
                                      capture_output=True, timeout=20, check=False)
                rc = proc.returncode
                ok = rc == 0
                err = "" if ok else (
                    "the profile blocked a write it should have allowed"
                    if rc == 3 else
                    "a protected entry was still writable inside the profile"
                    if rc == 4 else
                    f"bwrap exited {rc}: "
                    f"{proc.stderr.decode(errors='replace').strip()[:200]}")
            except Exception as exc:  # noqa: BLE001
                ok, err = False, f"bwrap probe failed: {exc}"
        _probe.update(ok=ok, at=now, error=err)
        return ok, err


def reset_probe_cache() -> None:
    """Forget the probe verdict — for a test that moved `HOME`, or a re-check."""
    with _probe_lock:
        _probe.update(ok=None, at=0.0, error="")


def status() -> dict:
    """The `/state` shape. `enforcing` and `fallback` are the two #2109 names;
    `entries` is what this process would bind right now, so a reader can see
    the list the kernel is holding rather than infer it from a constant."""
    ok, err = available()
    return {
        "enforcing": ok,
        "fallback": not ok,
        "bwrap": bool(bwrap_path()),
        "entries": protected_shell_ro_paths(),
        "error": err or None,
    }
