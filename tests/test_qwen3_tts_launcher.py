"""What the TTS launcher hands the process it exec's, across that boundary.

The thing that stops a restarted TTS server from eating the first voice turn
used to live in exactly one place: `agent-tts.conf`'s `environment=` line. The
backend's own default is the opposite — `api/main.py` resolves
`TTS_LAZY_LOAD = _env_bool("TTS_LAZY_LOAD", True)` — so the supervised boot came
up eager while a person running `agent-services/bin/start-qwen3-tts.sh` by hand
got the lazy path. That path is the 2026-09-19 incident: the stack restarted at
19:54, the first voice turn arrived at 20:04:42, audio came out at 20:08:48
(4 min 6 s), and because `TTSStreamer._http` in `agent-services/livekit_worker.py`
is one serial client with `read=120.0`, "One moment." and "Honestly?" were
discarded exactly 120 s apart rather than spoken late.

#1446 moved the default into the launcher. What matters is not that the script
mentions the variable but what the exec'd uvicorn child receives, so every test
here runs the real script and reads the child's own environment: a change that
sets the knob somewhere the `exec` cannot see it, or one that hard-exports over
an operator's value, fails here rather than at 20:08 on the morning after a
restart.

The launcher now carries a second knob of exactly that shape, and the tests read
the child's argv for its flag too: the bind host. #1638 measured `:8090`
answering `/v1/voices` with no credential from 192.168.50.108 — the same
failure mode in reverse, a `--host` written into the `exec` that nobody chose,
binding an unauthenticated service to the LAN. Loopback is the new default for
the same reason `false` is: what a hand-run of this file does is what production
does, and what the app defaults to is not what we want. The two tests that
matter are therefore about the boundary, not the script text — one runs the
launcher with nothing set and reads `--host` out of the argv the uvicorn child
actually got, the other sets a bind host in the environment and reads it back
out of that same argv, because a hard-coded loopback would strand the tailnet
device or Voice Studio bookmark the moment one needs the port.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = ROOT / "agent-services" / "bin" / "start-qwen3-tts.sh"
CONF = ROOT / "agent-services" / "supervisor" / "conf.d" / "agent-tts.conf"

#: The knob. Its two homes are asserted below: the launcher's default and the
#: supervisor conf's override.
KNOB = "TTS_LAZY_LOAD"

# Stands in for the qwen3-tts venv's python. Records the argv the launcher
# exec'd and the knob as the CHILD saw it — the env the script set internally is
# not the claim.
_STUB_PYTHON = """\
#!/usr/bin/env bash
{
  printf 'argv=%s\\n' "$*"
  printf 'TTS_LAZY_LOAD=%s\\n' "${TTS_LAZY_LOAD-<unset>}"
  printf 'PORT=%s\\n' "${PORT-<unset>}"
  printf 'HOST=%s\\n' "${HOST-<unset>}"
} > "$KNOB_DUMP"
"""

# The launcher polls `ss` until :8090 is free and gives up after 30 s. Production
# has `agent-tts` listening on that port, so the real binary would make every
# test below wait 30 s and then exit 1 before the exec. No output from `ss`
# means "nothing holds the port", which is the branch that proceeds.
_STUB_SS = """\
#!/usr/bin/env bash
exit 0
"""


def _run_launcher(tmp_path: Path, **overrides) -> dict[str, str]:
    """Run the repo's launcher for real and return the child's environment.

    The script derives everything from its own location and from `$HOME`
    (`PROJECT_DIR` from `$0`, the venv from `$HOME/lloyd/.venvs/qwen3-tts`), and
    the vendored `services/tts/qwen3-tts/` tree is gitignored — so it is absent
    from a worktree checkout, and `cd "$QWEN3_TTS_DIR"` would abort the script
    under `set -e`. The launcher is therefore copied into a sandbox tree holding
    the two paths it needs, with `$HOME` aimed at a stub venv. Its bytes and its
    mode are the repo's, unchanged.
    """
    sandbox = tmp_path / "sandbox"
    bin_dir = sandbox / "agent-services" / "bin"
    bin_dir.mkdir(parents=True)
    # copy2, not write_text: the mode bits are part of what is under test, and
    # the script is run the way supervisor runs it (see below).
    script = bin_dir / LAUNCHER.name
    shutil.copy2(LAUNCHER, script)
    (sandbox / "agent-services" / "services" / "tts" / "qwen3-tts").mkdir(parents=True)

    venv_bin = sandbox / "home" / "lloyd" / ".venvs" / "qwen3-tts" / "bin"
    venv_bin.mkdir(parents=True)
    stub_python = venv_bin / "python"
    stub_python.write_text(_STUB_PYTHON)
    stub_python.chmod(0o755)

    stub_bin = tmp_path / "stub-bin"
    stub_bin.mkdir()
    (stub_bin / "ss").write_text(_STUB_SS)
    (stub_bin / "ss").chmod(0o755)

    dump = tmp_path / "child.txt"
    # Nothing the backend reads survives from the test runner's own shell —
    # TTS_WARMUP_ON_START is the one this script does not export, so an inherited
    # value would otherwise flow into the dump and read as the launcher's doing.
    # HOST is stripped for the same reason and a sharper one: it is the bind
    # knob, so one inherited value would quietly turn "nothing is set in the
    # environment" into a case where something is, and clause 1 would then pass
    # on the runner's address instead of on the launcher's default. Nothing sets
    # it on this box today — `bash -ic 'echo ${HOST-<unset>}'` prints the unset
    # marker under the login shell, and neither supervisord's environment nor the
    # live agent-tts process's carried it (both read 2026-09-27) — but zsh and
    # ksh do export HOST as the hostname, and the filter is one token, so the
    # test does not have to be right about which shell a future developer ran it
    # from. That uvicorn's response to a hostname is a clean exit rather than a
    # wide bind is the launcher's claim, cited in its own comment.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("TTS_") and k not in ("PORT", "HOST")}
    env.update({
        "HOME": str(sandbox / "home"),
        "PATH": f"{stub_bin}{os.pathsep}{os.environ['PATH']}",
        "KNOB_DUMP": str(dump),
    })
    env.update(overrides)

    # Executed as a bare path, not `bash <script>`: that is how supervisor runs
    # this file (`command=` with no interpreter, unlike the engine launchers),
    # and it is the only form that fails when the exec bit goes missing.
    r = subprocess.run([str(script)], capture_output=True, text=True,
                       env=env, timeout=60)
    assert r.returncode == 0, f"launcher exited {r.returncode}\n{r.stdout}{r.stderr}"
    assert dump.exists(), "the launcher never reached the exec'd child"
    return dict(line.split("=", 1) for line in dump.read_text().splitlines())


def _comments(path: Path) -> str:
    """The script's comment lines, marker stripped, joined for matching."""
    return "\n".join(ln.lstrip("#").strip()
                     for ln in path.read_text().splitlines()
                     if ln.lstrip().startswith("#"))


def _bind_host(argv: str) -> str:
    """The `--host` value in the argv the launcher exec'd."""
    parts = argv.split()
    assert "--host" in parts, f"the child got no --host flag at all: {argv!r}"
    return parts[parts.index("--host") + 1]


def _exec_statement(path: Path) -> str:
    """The launcher's `exec` statement, its backslash continuations joined onto
    one line, and nothing after it.

    Two reasons to scope it this tightly. The file's prose quotes both `0.0.0.0`
    (the app's own default, quoted in order to say why it is wrong) and `--host`
    (the flag name), so a whole-file absence check would be checking the wrong
    thing in both directions. And stopping at the statement rather than running
    to EOF is what keeps that check honest as the file grows: text appended
    below the `exec` today would silently widen the scope of
    `"0.0.0.0" not in exec_stmt` tomorrow, which is pinned by
    test_the_exec_statement_helper_stops_at_the_statement_rather_than_at_eof.
    """
    lines = path.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("exec "))
    parts: list[str] = []
    cursor = start
    while True:
        line = lines[cursor].strip()
        # Drop a trailing continuation, then the space that preceded it, so the
        # join below puts exactly one space where the break was.
        parts.append(line[:-1].rstrip() if line.endswith("\\") else line)
        following = cursor + 1
        if not line.endswith("\\") or following >= len(lines) \
                or not lines[following].strip():
            break
        cursor = following
    return " ".join(parts)


# ── the seam supervisor crosses: it exec's this file by path ─────────
# `agent-tts.conf:2` is `command=/home/alansrobotlab/lloyd/agent-services/bin/…`
# with no `/bin/bash` in front of it, unlike `agent-llm-primary.conf:8`,
# `agent-llm-secondary.conf:2`, `agent-djev.conf:23` and
# `agent-qmd-watcher.conf:2`, which name the interpreter. So for this program
# only, the executable bit is what stands between a restart and a program that
# cannot boot at all — and a round that quietly flips it (the first commit of
# #1446 did, 100755→100644, through an editor rewriting the file) stays green if
# the tests invoke the script through bash.


def test_the_launcher_is_executable_because_supervisor_execs_it_directly():
    mode = LAUNCHER.stat().st_mode
    assert mode & stat.S_IXUSR, f"{LAUNCHER.name} is not executable: {oct(mode)}"
    command = next(ln for ln in CONF.read_text().splitlines()
                   if ln.startswith("command="))
    assert command.endswith(f"/{LAUNCHER.name}"), command
    assert "bash" not in command, \
        "the conf grew an interpreter, so this test's premise needs revisiting"


# ── clause 1: a hand-run boots eagerly ────────────────────────────────


def test_a_run_with_the_knob_absent_reaches_the_child_as_false(tmp_path):
    """The launcher, not the supervisor conf, is what makes this boot eager."""
    child = _run_launcher(tmp_path)
    assert child[KNOB] == "false", "a hand-run still gets the backend's lazy default"
    # The dump is the uvicorn process the launcher exec's, and nothing else:
    # eager loading that never reaches this argv would fix nothing.
    #
    # This line pinned `--host 0.0.0.0` until #1638, and that is the assertion
    # the item's own acceptance clause 1 makes wrong — "The exec'd uvicorn child
    # receives an explicit --host 127.0.0.1 in its argv when nothing is set in
    # the environment … and no --host 0.0.0.0 remains". It was pinning the wide
    # bind as required behaviour, which is how the hole survived four rounds of
    # edits to this file: green tests, 192.168.50.0/24 answering /v1/voices.
    # What the flag is meant to do now is the pair of sections below.
    assert child["argv"] == '-m uvicorn api.main:app --host 127.0.0.1 --port 8090'
    assert child["PORT"] == "8090"


# ── clause 2: an inherited value is passed through, not clobbered ─────
# supervisor expands `environment=` into the child's environment before bash
# runs the script, so whatever conf.d/agent-tts.conf declares is already in
# `$TTS_LAZY_LOAD` by the first line of this file. A plain
# `export TTS_LAZY_LOAD=false` reads identically in a test that sets nothing and
# quietly overwrites that operator.


@pytest.mark.parametrize("declared", ["true", "off"])
def test_a_knob_already_in_the_environment_reaches_the_child_unchanged(
        tmp_path, declared):
    child = _run_launcher(tmp_path, **{KNOB: declared})
    assert child[KNOB] == declared


def test_an_empty_inherited_value_falls_back_to_eager(tmp_path):
    """`${VAR:-default}` treats empty as unset, which is the useful direction:
    `TTS_LAZY_LOAD=""` in the conf would otherwise reach `_env_bool` as an empty
    string, which parses as neither and lands on its lazy default."""
    assert _run_launcher(tmp_path, **{KNOB: ""})[KNOB] == "false"


# ── clause 3: the reason is discoverable from the script ─────────────
# The knob was invisible from the launcher and lived in a conf comment 200 lines
# from the thing a person runs. Two claims have to be in the script's own
# comments, because they are the two that make `false` the default rather than a
# preference.


def test_the_launcher_comments_carry_the_reason_for_the_default():
    comments = _comments(LAUNCHER)
    # (a) the lazy default is the backend's, so the launcher is overriding
    #     something real and not inventing a knob.
    assert re.search(r"backend.s own default is lazy", comments), \
        "nothing says the backend defaults to lazy"
    assert "_env_bool(\"TTS_LAZY_LOAD\", True)" in comments, \
        "the lazy default is not traced to the code that holds it"
    assert "api/main.py" in comments
    # (b) a cold first synthesis is lost, not merely slow — the reason eager
    #     loading is worth ~4 min of unreachable :8090 at every restart.
    assert "TTSStreamer" in comments, "the serial client that times out is unnamed"
    assert "read=120.0" in comments, "the 120 s read timeout is not stated"
    assert re.search(r"DISCARDED, not merely delayed", comments), \
        "the comment does not say the first utterance is discarded, not delayed"
    # The incident is dated, so a later reader can re-check it instead of
    # trusting the numbers.
    assert "2026-09-19" in comments


def test_the_default_is_a_default_and_not_a_hard_export():
    """Structural, because clause 2's failure mode is silent: a hard export
    differs from a default only in the case nobody tests on the day it is
    written."""
    exports = [ln for ln in LAUNCHER.read_text().splitlines()
               if ln.startswith("export TTS_LAZY_LOAD")]
    assert exports == ['export TTS_LAZY_LOAD="${TTS_LAZY_LOAD:-false}"'], exports


# ── clause 1: the default bind reaches the child on loopback ──────────
# The child has to be told, and the two reasons not to rely on a default are
# different defaults. On the app-module route — `python api/main.py` — the module
# resolves `HOST = os.getenv("HOST", "0.0.0.0")` at the untracked
# `agent-services/services/tts/qwen3-tts/api/main.py:72` and hands it to
# `uvicorn.run(host=HOST)` at api/main.py:304-306, so nothing narrows it there.
# On this launcher's route — `-m uvicorn` — the CLI's own `--host` default is
# already 127.0.0.1, but it does not read HOST (its auto_envvar_prefix is
# UVICORN), so leaving the flag out would bind loopback while silently ignoring
# whatever the conf declares: loopback by accident, and a stranded clause 2. The
# flag is therefore pinned in the argv, not just its value. Being told the wrong
# value is worse still — #1638 measured `curl
# http://192.168.50.108:8090/v1/voices` returning 200 with no credential on
# 2026-09-27, from the range `server.py` declines to trust for /api/*, because the
# 1b050d82 gate is middleware in a different process and this is a second uvicorn
# with none.


def test_the_default_bind_reaches_the_child_on_loopback(tmp_path):
    """Nothing set in the environment, so the launcher is the only thing that
    could have chosen loopback here."""
    child = _run_launcher(tmp_path)
    assert _bind_host(child["argv"]) == "127.0.0.1", (
        "the child is still told to answer on every interface: "
        f"--host {_bind_host(child['argv'])}")
    # The flag itself, not merely the value: uvicorn's CLI would default to
    # 127.0.0.1 without it, so this assertion is what keeps a later "the default
    # already does this" simplification from quietly deleting the only carrier of
    # the conf's HOST (the CLI reads UVICORN_HOST, not HOST).
    assert child["argv"].startswith("-m uvicorn api.main:app --host "), child["argv"]
    # The exported env carries it too, for the other entry point — the app module
    # reads HOST itself at api/main.py:72 — so an operator looking at
    # conf.d/agent-tts.conf sees one bind, not two competing ones.
    assert child["HOST"] == "127.0.0.1"


def test_no_wide_bind_survives_in_the_exec_statement():
    """The structural half of "no `--host 0.0.0.0` remains", checked against the
    `exec` rather than the file, and with its positive control asserted first:
    the prose quotes both `0.0.0.0` (the app's default, quoted to say why it is
    wrong) and `--host` (the flag name), so a whole-file absence check would
    read as a pass against text that no longer execs anything."""
    exec_stmt = _exec_statement(LAUNCHER)
    assert '--host "$HOST"' in exec_stmt, \
        f"the exec stopped reading the knob, so the default is decoration: {exec_stmt}"
    assert "0.0.0.0" not in exec_stmt, exec_stmt


def test_the_exec_statement_helper_stops_at_the_statement_rather_than_at_eof(tmp_path):
    """The absence check above is only as narrow as the helper that scopes it.
    This is the case that would otherwise rot it quietly: a line appended after
    the exec's last continuation — a comment, a second command, anything
    quoting `0.0.0.0` — must not end up inside the string being checked.

    Both exec shapes get their own probe, because the helper's two branches are
    the continuation and its absence, and the real launcher exercises only the
    first. The single-line form is the one a future edit collapses the exec into,
    and it is where a trailing-backslash strip left over in the join would show
    up as a phantom argument.
    """
    continued = tmp_path / "continued.sh"
    continued.write_text(
        '#!/usr/bin/env bash\n'
        'cd somewhere\n'
        'exec "$VENV/bin/python" -m uvicorn api.main:app \\\n'
        '    --host "$HOST" \\\n'
        '    --port 8090\n'
        'echo "the app would have used 0.0.0.0 anyway"\n')
    assert _exec_statement(continued) == \
        'exec "$VENV/bin/python" -m uvicorn api.main:app --host "$HOST" --port 8090'

    single_line = tmp_path / "single.sh"
    single_line.write_text(
        '#!/usr/bin/env bash\n'
        'export HOST="${HOST:-127.0.0.1}"\n'
        'exec "$VENV/bin/python" -m uvicorn api.main:app --host "$HOST" --port 8090\n'
        'echo "0.0.0.0"\n')
    assert _exec_statement(single_line) == \
        'exec "$VENV/bin/python" -m uvicorn api.main:app --host "$HOST" --port 8090'


# ── clause 2: a bind host already in the environment passes through ───
# The same shape as TTS_LAZY_LOAD two sections up — supervisor expands
# conf.d/agent-tts.conf's `environment=` into this script's environment before
# bash runs it — but the stakes are inverted: here the narrow value is the one
# that can strand somebody. agent-tts.log on 2026-09-27 carried 4 requests from
# this box's own tailnet address (`/health`, `/openapi.json`, `/v1/voices`,
# `/v1/models`, Swagger/Voice-Studio-shaped) against 638 from 127.0.0.1, and
# those 4 are unattributed, so the override is how a Voice Studio bookmark or a
# tailnet device comes back without a revert: `HOST="0.0.0.0"` on the conf, or
# `HOST="<address to bind>"` for the narrower version of the same thing.


@pytest.mark.parametrize("declared", ["0.0.0.0", "100.105.113.88",
                                      "somewhere.example.invalid"])
def test_a_bind_host_already_in_the_environment_reaches_the_child_unchanged(
        tmp_path, declared):
    """The third case is deliberately not an address — the shape a shell that
    exports HOST for its own reasons, or a typo on the conf line, actually
    arrives in. Clause 2 is pass-through, so this asserts the launcher validates
    nothing and mangles nothing; what uvicorn then does with the value is not
    this script's business, and the launcher's comment cites the code that
    decides it rather than leaving it to guesswork: `sock.bind((self.host,
    self.port))` under `except OSError: sys.exit(1)` at config.py:536-539, plus
    `main.py:614` exiting 3 when the server never started — both read in the
    `.venvs/qwen3-tts` tree that runs this service, so a mistyped or inherited
    HOST fails the boot loudly instead of answering on an interface nobody chose.
    """
    child = _run_launcher(tmp_path, HOST=declared)
    assert _bind_host(child["argv"]) == declared, \
        f"--host clobbered the operator's {declared!r}"
    assert child["HOST"] == declared


def test_an_empty_inherited_bind_fails_closed_to_loopback(tmp_path):
    """`HOST=""` on the conf's `environment=` line is present but meaningless;
    `${VAR:-default}` reading that as unset is the fail-closed direction, and
    uvicorn taking an empty bind string is the other one."""
    assert _bind_host(_run_launcher(tmp_path, HOST="")["argv"]) == "127.0.0.1"


def test_the_bind_default_is_a_default_and_not_a_hard_export():
    """Structural, for the same silent failure mode as the lazy knob's test:
    `export HOST="127.0.0.1"` and `export HOST="${HOST:-127.0.0.1}"` differ only
    in the case nobody thinks to test."""
    exports = [ln for ln in LAUNCHER.read_text().splitlines()
               if ln.startswith("export HOST")]
    assert exports == ['export HOST="${HOST:-127.0.0.1}"'], exports


# ── clause 3: the reason and the check are in the script itself ───────
# A one-token diff (`0.0.0.0` → `127.0.0.1`) is precisely the kind a later
# reader re-widens to make one device work, so the file has to carry both why
# it is narrow and how to tell whether it still is.


def test_the_launcher_comments_carry_the_reason_for_the_loopback_bind():
    comments = _comments(LAUNCHER)
    # (a) why narrow: nothing authenticates, so the interface is the only access
    #     control this listener has.
    assert "no credential of any kind" in comments, \
        "nothing says the service authenticates nothing"
    # \s+, not a space: _comments joins one script line per "\n", so a claim
    # that wraps inside the script must still read as one claim here.
    assert re.search(r"answering unauthenticated on\s+192\.168\.50\.0/24", comments), \
        "the comment does not name the LAN range the wide bind exposed"
    # (b) the wide value is the app's own default, traced to the line that holds
    #     it, so deleting the flag reveals what returns instead of assuming the
    #     app would have bound narrowly anyway.
    assert 'os.getenv("HOST", "0.0.0.0")' in comments, \
        "the app's wide default is not traced to the code that holds it"
    assert "api/main.py" in comments
    # (c) the check, with both edges, so the claim is re-runnable rather than a
    #     promise: the LAN address refuses and loopback still answers.
    assert "ss -ltn" in comments, "the listener check is not written down"
    assert re.search(r"curl[^\n]*192\.168\.50\.108:8090/v1/voices\s+->\s+no connection",
                     comments), \
        "nothing says curling the LAN address must fail to connect"
    assert re.search(r"curl[^\n]*127\.0\.0\.1:8090/v1/voices\s+->\s+200", comments), \
        "nothing says curling loopback must still answer"


def test_the_launcher_comments_name_what_carries_the_bind_and_what_fails_it():
    """Two claims the flag's survival depends on, and neither is clause 3's
    "why loopback" — they are the two that stop a later reader concluding the
    flag is redundant, which is how an override dies: someone simplifies the exec
    because uvicorn's own default is already 127.0.0.1.

    (a) that the flag, not the environment, is the carrier;
    (b) that a value which is not a local address fails the boot rather than
        binding wide, cited to the code that decides it — a file:line a reader can
        open, and opened in the venv that runs the service, not the one that runs
        the backend.
    """
    comments = _comments(LAUNCHER)
    assert "auto_envvar_prefix" in comments, \
        "nothing says uvicorn's CLI reads UVICORN_HOST rather than HOST"
    assert re.search(r"api/main\.py:304", comments), \
        "the app-module entry point that does read HOST is not traced to its line"
    assert "config.py:536-539" in comments, \
        "the bind-failure path is not cited to the OSError/exit in uvicorn's config"
