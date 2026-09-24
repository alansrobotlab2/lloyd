"""The restart instruction `scripts/gen-livekit-secrets.sh` prints must work as pasted.

It used to say bare `supervisorctl restart …`. There is no `supervisorctl` on
PATH here and, without `-c`, no way for one to find this stack's socket, so the
operator who just rotated the LiveKit secrets was handed a command that exits 4
and left services on the old values (#1042). CLAUDE.md's Service Management
section is the source for the full form.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "gen-livekit-secrets.sh"
CONF = "agent-services/supervisor/supervisord.conf"
SOCK = Path("/tmp/agent-supervisor.sock")
PROGRAMS = ("lloyd-mc:lloyd-backend", "lloyd-agent-worker", "agent-livekit-server")

#: The commit before #1042; every line that is not a supervisorctl instruction
#: must still read as it did there.
BASE = "2a53554f"


def _lines() -> list[str]:
    return SCRIPT.read_text().splitlines()


def _final_echo_command() -> str:
    echoes = [l for l in _lines() if l.startswith("echo ") and "supervisorctl" in l]
    assert echoes, "the script no longer prints a restart command"
    m = re.fullmatch(r'echo "\s*(.*)"', echoes[-1])
    assert m
    return m.group(1)


def test_every_supervisorctl_instruction_names_the_stack_conf():
    hits = [l for l in _lines() if "supervisorctl" in l]
    assert len(hits) == 2
    for line in hits:
        assert re.search(r"supervisorctl -c \S*" + re.escape(CONF) + r"\b", line), line
    assert not re.search(r"(?<![/\w])supervisorctl restart", SCRIPT.read_text())


@pytest.mark.skipif(not SOCK.exists(), reason=f"{SOCK} absent: no supervisord here")
def test_the_printed_command_runs_verbatim_from_another_cwd(tmp_path):
    cmd = _final_echo_command()
    assert " restart " in cmd
    status = cmd.replace(" restart ", " status ", 1)
    proc = subprocess.run(["bash", "-c", status], cwd=tmp_path,
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for program in PROGRAMS:
        assert program in proc.stdout


def test_only_the_instruction_strings_changed():
    try:
        before = subprocess.run(["git", "show", f"{BASE}:scripts/gen-livekit-secrets.sh"],
                                cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip(f"{BASE} not reachable from this checkout")
    keep = lambda text: [l for l in text.splitlines() if "supervisorctl" not in l]  # noqa: E731
    assert keep(SCRIPT.read_text()) == keep(before)


def test_print_mode_writes_nothing_and_prints_two_lines(tmp_path):
    copy = tmp_path / "scripts" / "gen-livekit-secrets.sh"
    copy.parent.mkdir()
    copy.write_text(SCRIPT.read_text())
    proc = subprocess.run(["bash", str(copy), "--print"], cwd=tmp_path,
                          capture_output=True, text=True, timeout=30,
                          env={**os.environ})
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.splitlines()
    assert len(out) == 2
    assert out[0].startswith("LIVEKIT_API_KEY=") and out[1].startswith("LIVEKIT_API_SECRET=")
    assert not (tmp_path / ".env").exists()
