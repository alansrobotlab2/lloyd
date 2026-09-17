"""A background turn may not restart or stop an engine or a service.

On 2026-09-17 an autocode turn ran `round restart --only agent-llm-primary`
from Bash to fix a grader that had returned empty output, took the primary
down for five minutes, killed every turn in flight including itself, and
left its freshly reopened round with no owner. Refused now at both Bash
enforcement points, for background sessions only.
"""

from __future__ import annotations

import pytest

from app.harness import safety
from app.harness.service_control import (check_service_control, find_service_control,
                                         is_background_session)

WORKER = "20260916_165238_autocode_1b1a"
AUTONOMY = "20260910_120001_autonomy_9f2a"
CHAT = "20260916_164818_iv2eca"

REFUSED = [
    "cd ~/lloyd && .venvs/lloyd/bin/python -m scripts.automod.round restart --only agent-llm-primary --reason \"grader empty\" 2>&1 | tail -3 &",
    "python -m scripts.automod.round recover",
    "/home/alansrobotlab/.local/share/uv/tools/supervisor/bin/supervisorctl -c /home/alansrobotlab/lloyd/agent-services/supervisor/supervisord.conf restart lloyd-mc:lloyd-backend",
    "supervisorctl stop agent-llm-primary",
    "supervisorctl -c x.conf reread; supervisorctl -c x.conf update",
    "systemctl --user restart lloyd-guardian",
    "systemctl stop agent-supervisord.service",
    "systemctl --user daemon-reload",
    "pkill -f vllm.entrypoints",
    "killall llama-server",
    "bash agent-services/bin/flash-next-run-arm.sh fp8",
    "bash -c 'supervisorctl -c x.conf restart lloyd-mc:lloyd-mcp'",
    "python3 -c \"import subprocess; subprocess.run(['supervisorctl', '-c', 'x.conf', 'restart', 'lloyd-mc:lloyd-backend'])\"",
    "nohup timeout 900 .venvs/lloyd/bin/python -m scripts.automod.round restart --only lloyd-backend > /tmp/r.log 2>&1 &",
]
ALLOWED = [
    "supervisorctl -c x.conf status",
    "/home/alansrobotlab/.local/share/uv/tools/supervisor/bin/supervisorctl -c x.conf tail -f lloyd-mc:lloyd-backend stderr",
    "systemctl --user status lloyd-guardian",
    "systemctl --user list-units | grep lloyd",
    "python -m scripts.automod.round status",
    # `round land` moved to tests/test_landing_killed.py (2026-09-17): a
    # foreground landing waits on the turn that ran it and dies at the Bash
    # timeout. The gate alone stays allowed.
    "python -m scripts.automod.round gate SM_1",
    "pkill -f my_probe_script.py",
    "curl -s localhost:8096/health",
    "grep -n 'supervisorctl restart' CLAUDE.md",
    "echo 'run: round restart --only lloyd-backend' >> notes.md",
    "python3 -c \"print('supervisorctl status')\"",
    "git log --oneline -3 -- scripts/automod/round.py",
]


@pytest.mark.parametrize("command", REFUSED)
def test_a_worker_session_is_refused(command):
    assert find_service_control(command)
    assert check_service_control(command, WORKER)
    assert check_service_control(command, AUTONOMY)
    match = safety.check_bash_command(command, None, session_id=WORKER)
    assert match and match[0].startswith("service control:"), match
    match = safety.check_bash_command(command, None, at_dispatch=True, session_id=WORKER)
    assert match and match[0].startswith("service control:"), "dispatch refuses too"


@pytest.mark.parametrize("command", ALLOWED)
def test_read_only_forms_stay_allowed(command):
    assert check_service_control(command, WORKER) is None
    assert safety.check_bash_command(command, None, session_id=WORKER) is None


@pytest.mark.parametrize("command", REFUSED)
def test_a_chat_session_is_never_refused_here(command):
    assert check_service_control(command, CHAT) is None
    assert check_service_control(command, None) is None
    assert safety.check_bash_command(command, None, session_id=CHAT) is None
    assert safety.check_bash_command(command, None) is None, "no session known: skipped, not refused"


def test_a_subagent_is_classified_by_its_parent():
    parents = {"task:abc": WORKER, "task:def": CHAT, "task:loop": "task:loop"}
    assert is_background_session("task:abc", parent_of=parents.get)
    assert not is_background_session("task:def", parent_of=parents.get)
    assert not is_background_session("task:loop", parent_of=parents.get)
    assert not is_background_session("task:abc"), "no resolver: not refused"
    assert check_service_control("supervisorctl restart x", "task:abc", parent_of=parents.get)
    assert check_service_control("supervisorctl restart x", "task:def", parent_of=parents.get) is None


@pytest.mark.asyncio
async def test_dispatch_refuses_the_incident_command_for_the_worker_session(monkeypatch):
    import agent_mcp.main as M
    cmd = "cd ~/lloyd && .venvs/lloyd/bin/python -m scripts.automod.round restart --only agent-llm-primary --reason x &"
    out = await M.call_tool("Bash", {"command": cmd}, {M.META_SESSION_ID: WORKER})
    content = getattr(out, "content", out)
    text = content[0].text if hasattr(content[0], "text") else str(content[0])
    assert "service control" in text and "background session" in text, text


def test_the_hook_passes_the_session_id(monkeypatch):
    import asyncio
    seen = {}

    def fake(command, cwd=None, *, at_dispatch=False, session_id=None, parent_of=None):
        seen["session_id"] = session_id
        return None
    monkeypatch.setattr(safety, "check_bash_command", fake)
    asyncio.run(safety._safety_pretool_cb(
        {"tool_name": "Bash", "tool_input": {"command": "ls"}, "session_id": WORKER}, None, None))
    assert seen["session_id"] == WORKER
