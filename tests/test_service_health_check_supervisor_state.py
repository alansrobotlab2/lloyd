"""#1040: a crash-looping (BACKOFF) supervisor service must not read healthy.

`supervisorctl status` prints `name  STATENAME  description` per process
(supervisor 4.3.0, `supervisorctl.py:643-655`), and it *exits 0 for BACKOFF*:
`do_status` moves the exit status off SUCCESS only for
`states.STOPPED_STATES` (`supervisorctl.py:696-698`), and `states.py:14-22`
puts BACKOFF in RUNNING_STATES, not STOPPED_STATES. The script decided health
from substring tests over the whole answer plus a
`healthy = result.returncode == 0` fallback, so feeding the real
`check_service` the measured BACKOFF line with `returncode=0` returned
`healthy=True` — a process that is spawned, dies inside `startsecs` and is
being retried printed `[✓] healthy`. The same branch read
`agent-tts  FATAL  can't spawn process: RUNNING helper not found` as
`healthy=True, status="RUNNING"` (the substring hit inside the *description*),
and empty output as `healthy=True, status="unknown"`.

The state is field 2 of each status line, so field 2 is the only field read.
Each test names the acceptance clause it pins.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "service_health_check.py"

# Measured output, quoted verbatim from the item's /tmp supervisord run and
# from triage's replay through the real check_service().
BACKOFF_LINE = (
    "flaky                            BACKOFF   "
    "Exited too quickly (process log may have details)"
)
FATAL_WITH_RUNNING_IN_DESCRIPTION = (
    "agent-tts      FATAL   can't spawn process: RUNNING helper not found"
)
RUNNING_LINE = "lloyd-mc:lloyd-backend      RUNNING   pid 3781637, uptime 1:00:00"

# supervisor's own state names; every one of these except RUNNING is unhealthy.
NOT_RUNNING_STATES = ("STOPPED", "STARTING", "STOPPING", "BACKOFF", "EXITED", "FATAL", "UNKNOWN")

# A SERVICES-shaped entry with no `port`, so the supervisor branch alone
# decides the verdict — the same five of six live entries that are.
SERVICE_DEF = {
    "command": ["supervisorctl", "-c", "/nonexistent/supervisord.conf", "status", "flaky"],
    "category": "lloyd",
}


def _load_module():
    """Load the CLI script as a module.

    It has no package and nothing in the tree imports it — it is invoked as a
    subprocess from Bash — so loading it by path is how a unit test reaches
    `check_service` without inventing an import surface.
    """
    spec = importlib.util.spec_from_file_location("shc_supervisor_state", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shc = _load_module()


class _StubSubprocess:
    """Replaces `shc.subprocess`, so supervisor's answer and exit code are fixed.

    Only the genuinely external answer is stubbed: which state the daemon
    reports with which LSB exit code. The daemon itself cannot be crash-looped
    from a test, and the real-`subprocess` half of that boundary is covered by
    the CLI tests at the bottom of this file.
    """

    TimeoutExpired = subprocess.TimeoutExpired

    def __init__(self, stdout: str, returncode: int = 0, stderr: str = ""):
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr
        self.calls: list[list] = []

    def run(self, command, **_kwargs):
        self.calls.append(list(command))
        return subprocess.CompletedProcess(command, self.returncode, self.stdout, self.stderr)


def _verdict(monkeypatch, stdout: str, returncode: int = 0) -> dict:
    """Run the real `check_service` against a fixed supervisorctl answer."""
    monkeypatch.setattr(shc, "subprocess", _StubSubprocess(stdout, returncode))
    return shc.check_service("flaky", SERVICE_DEF)


# ---------------------------------------------------------------- clause 1
def test_backoff_line_with_a_zero_exit_code_is_not_healthy(monkeypatch):
    """Clause 1: the measured BACKOFF line, with the exit code supervisor really
    gives it (0), is unhealthy and the state survives into `status`."""
    result = _verdict(monkeypatch, BACKOFF_LINE, returncode=0)
    assert result["healthy"] is False, (
        f"a crash-looping process read as healthy: {result}"
    )
    assert "BACKOFF" in result["status"], result


# ---------------------------------------------------------------- clause 2
@pytest.mark.parametrize("returncode", [0, 3])
@pytest.mark.parametrize("state", NOT_RUNNING_STATES)
def test_every_non_running_state_is_unhealthy_whatever_the_exit_code(
    monkeypatch, state, returncode
):
    """Clause 2: the state decides, not the exit code. Each token as field 2 of
    a status line, presented with both the exit code supervisor gives BACKOFF
    (0) and the one it gives STOPPED/EXITED/FATAL/UNKNOWN (3)."""
    line = f"agent-x  {state}   Exited too quickly (process log may have details)"
    result = _verdict(monkeypatch, line, returncode=returncode)
    assert result["healthy"] is False, (
        f"state {state} with returncode={returncode} read as healthy: {result}"
    )
    assert state in result["status"], result


def test_running_state_is_healthy(monkeypatch):
    """Clause 2's other half, so the test above cannot be satisfied by an entry
    that is simply always unhealthy. `status` keeps the shape the script has
    always printed for a running service."""
    result = _verdict(monkeypatch, RUNNING_LINE, returncode=0)
    assert result["healthy"] is True, result
    assert result["status"] == "RUNNING", result


# ---------------------------------------------------------------- clause 3
def test_fatal_whose_description_mentions_running_is_unhealthy(monkeypatch):
    """Clause 3: the state is read from field 2, never matched as a substring
    anywhere in the line. This line was `healthy=True, status="RUNNING"`."""
    result = _verdict(monkeypatch, FATAL_WITH_RUNNING_IN_DESCRIPTION, returncode=0)
    assert result["healthy"] is False, (
        f"a FATAL process whose description contains the word RUNNING read as "
        f"healthy: {result}"
    )
    assert "FATAL" in result["status"], result


# ---------------------------------------------------------------- clause 4
def test_empty_output_is_unhealthy_and_names_the_missing_answer(monkeypatch):
    """Clause 4: an empty answer is not a healthy verdict. It was
    `healthy=True, status="unknown"`."""
    result = _verdict(monkeypatch, "", returncode=0)
    assert result["healthy"] is False, result
    assert result["status"].strip(), f"empty output left an empty status: {result}"
    assert "empty" in result["status"].lower(), result
    assert "unknown" in result["status"].lower(), result


def test_two_line_answer_with_one_backoff_line_is_unhealthy(monkeypatch):
    """Clause 4: a group namespec prints one line per process, so every line
    has to be RUNNING — the first line being RUNNING cannot carry the verdict."""
    answer = f"{RUNNING_LINE}\n{BACKOFF_LINE}"
    result = _verdict(monkeypatch, answer, returncode=0)
    assert result["healthy"] is False, (
        f"one BACKOFF process inside an otherwise RUNNING answer read as healthy: "
        f"{result}"
    )
    assert "BACKOFF" in result["status"], result


def test_two_line_answer_with_every_line_running_is_healthy(monkeypatch):
    """The all-lines-RUNNING positive, so the clause-4 test above cannot be
    satisfied by a parser that just always says unhealthy."""
    answer = f"{RUNNING_LINE}\nlloyd-mc:lloyd-mcp RUNNING pid 3781637, uptime 1:00:00"
    result = _verdict(monkeypatch, answer, returncode=0)
    assert result["healthy"] is True, result
    assert result["status"] == "RUNNING", result


@pytest.mark.parametrize(
    "answer",
    [
        "FAILED",
        f"{RUNNING_LINE}\nnot-a-status-line",
    ],
)
def test_a_line_without_a_state_field_is_unhealthy(monkeypatch, answer):
    """The acceptance check's "any unparsable line" half: fewer than two
    whitespace-separated fields means there is no state to read, so there is no
    healthy verdict — even when the first line said RUNNING."""
    result = _verdict(monkeypatch, answer, returncode=0)
    assert result["healthy"] is False, (
        f"answer with no readable state field read as healthy: {result}"
    )
    assert answer in result["status"], result


# ------------------------------------------- the real process boundary (Bash →
# ------------------------------------------- python → subprocess.run → child)
FAKE_SUPERVISORCTL = (
    "#!{python}\n"
    "import json, os, sys\n"
    "answer = json.loads(os.environ['FAKE_SUPERVISORCTL'])\n"
    "sys.stdout.write(answer['stdout'])\n"
    "sys.stderr.write(answer.get('stderr', ''))\n"
    "sys.exit(answer['returncode'])\n"
)


def _run_cli_with_fake_supervisorctl(tmp_path, stdout: str, returncode: int) -> dict:
    """Run the real script as a real subprocess against a stand-in
    `supervisorctl` that prints `stdout` and exits `returncode`.

    This is the boundary the unit tests above stub: the script's own
    `subprocess.run` call, a real child process, a real exit code, real stdout.
    `--category lloyd` selects the three port-less supervisor entries, so the
    verdict under test is the supervisor branch's alone.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "supervisorctl"
    fake.write_text(FAKE_SUPERVISORCTL.format(python=sys.executable))
    fake.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["FAKE_SUPERVISORCTL"] = json.dumps(
        {"stdout": stdout, "returncode": returncode}
    )
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--category", "lloyd", "--format", "json"],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert proc.returncode == 0, (
        f"the script's own exit status changed (it must not — that is #1029's "
        f"contract): rc={proc.returncode} stderr={proc.stderr[-500:]}"
    )
    return json.loads(proc.stdout)


def test_cli_reports_a_backoff_service_unhealthy_through_a_real_subprocess(tmp_path):
    """Same measured BACKOFF line, but delivered by a real child process with a
    real exit code 0 rather than by a stub: this is what a run of the
    service-health-check skill actually consumes."""
    payload = _run_cli_with_fake_supervisorctl(tmp_path, BACKOFF_LINE, 0)
    entries = payload["services"]
    assert len(entries) == 3, entries
    for entry in entries:
        assert entry["healthy"] is False, (
            f"real supervisorctl output BACKOFF with exit code 0 produced a "
            f"healthy entry: {entry}"
        )
        assert "BACKOFF" in entry["status"], entry
        assert entry["exit_code"] == 0, entry
    assert payload["healthy"] == 0, payload


def test_cli_reports_a_running_service_healthy_through_a_real_subprocess(tmp_path):
    """The positive control across the same boundary: the fake is not simply
    always unhealthy, and the RUNNING path still reports exit_code 0."""
    payload = _run_cli_with_fake_supervisorctl(tmp_path, RUNNING_LINE, 0)
    entries = payload["services"]
    assert len(entries) == 3, entries
    for entry in entries:
        assert entry["healthy"] is True, entry
        assert entry["status"] == "RUNNING", entry
        assert entry["exit_code"] == 0, entry
    assert payload["healthy"] == 3, payload
