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
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("TTS_") and k != "PORT"}
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
    assert child["argv"] == "-m uvicorn api.main:app --host 0.0.0.0 --port 8090"
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
