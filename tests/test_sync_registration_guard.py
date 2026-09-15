"""No tool call may change the Obsidian Sync registration.

On 2026-09-14 Lloyd deleted `~/.config/obsidian-headless/sync/<vaultId>/` twice
from one chat by running the health check's `--vault-sync-round-trip` leg,
whose cleanup `ob sync-unlink`s by vault id. Re-linking takes Alan's E2E
password. These tests pin the refusal at `safety.check_bash_command` — the one
function both the harness hook and the aggregator's dispatch run — and the
allow list beside it, which is as load-bearing: an autotriage turn grepped the
probe's source for `sync-unlink` the same day, and a gate that refuses reading
the code that caused an incident gets routed around.

`HOME` is pointed at a temp tree, so the registration directory is a real
path that exists nowhere near the live one.
"""

from __future__ import annotations

import pytest

import agent_mcp.main as M
from app.harness.safety import check_bash_command
from app.harness.sync_registration import check_sync_registration


@pytest.fixture()
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    (h / ".config" / "obsidian-headless" / "sync" / "566dda12").mkdir(parents=True)
    (h / "obsidian").mkdir()
    monkeypatch.setenv("HOME", str(h))
    return h


# The two commands that did it, verbatim from session 20260914_190323_iv2eca.
INCIDENT = [
    "python3 ~/obsidian/skills/system-health-check/system_health_check.py "
    "--vault-sync-round-trip; echo \"EXIT=$?\"",
    "L=~/lloyd/agent-services/logs\npython3 ~/obsidian/skills/system-health-check/"
    "system_health_check.py --component vault_sync --vault-sync-round-trip; echo \"EXIT=$?\"\n"
    "echo; echo \"=== quota refusal since the 13:25 re-link ===\"; ls -l $L/agent-obsidian-sync.err",
]

REFUSED = INCIDENT + [
    "LLOYD_VAULT_SYNC_ROUND_TRIP=1 python3 ~/obsidian/skills/system-health-check/system_health_check.py",
    "export LLOYD_VAULT_SYNC_ROUND_TRIP=true; python3 system_health_check.py",
    "env LLOYD_VAULT_SYNC_ROUND_TRIP=yes python3 system_health_check.py --format json",
    "ob sync-unlink --path /tmp/scratch",
    "ob sync-setup --vault 566dda12 --path ~/obsidian --device-name x",
    "/usr/bin/ob sync-config --path ~/obsidian --mode pull-only",
    "ob sync --path ~/obsidian",
    "ob logout",
    "ob login --email a@b.c",
    "ob sync-create-remote --name new",
    "ob publish-unlink --path ~/obsidian",
    'OB=$(command -v ob); "$OB" sync-setup --vault x --path ~/obsidian',
    "node /usr/lib/obsidian-headless/cli.js sync-unlink --path /tmp/s",
    "timeout 60 ob sync-unlink --path /tmp/s",
    'bash -c "ob sync-unlink --path /tmp/s"',
    "cd /tmp && bash -c 'cd ~ && ob sync-setup --vault v --path obsidian'",
    "python3 -c \"import subprocess; subprocess.run(['ob', 'sync-unlink', '--path', '/tmp/s'])\"",
    "python3 -c \"import os; os.system('ob sync-setup --vault v --path /tmp/s')\"",
    "python3 - <<'EOF'\nimport os, subprocess\nos.environ['LLOYD_VAULT_SYNC_ROUND_TRIP'] = '1'\n"
    "subprocess.run(['python3', 'system_health_check.py'])\nEOF",
    "rm -rf ~/.config/obsidian-headless/sync/566dda12",
    "rm -rf ~/.config/obsidian-headless",
    "cd ~/.config/obsidian-headless && rm -rf sync/*",
    "mv ~/.config/obsidian-headless/sync /tmp/aside",
    "find ~/.config/obsidian-headless -name '*.db' -delete",
    "cp /tmp/config.json ~/.config/obsidian-headless/sync/566dda12/config.json",
    "echo '{}' > ~/.config/obsidian-headless/sync/566dda12/config.json",
    "sqlite3 ~/.config/obsidian-headless/sync/566dda12/state.db 'delete from files'",
    "sed -i s/a/b/ ~/.config/obsidian-headless/sync/566dda12/config.json",
    "ob sync-config --path ~/obsidian --mode mirror-remote",
    "bash <<'SH'\nob sync-unlink --path /tmp/s\nSH",
    "cat <<'EOF' > ~/.config/obsidian-headless/sync/566dda12/config.json\n{}\nEOF",
]

ALLOWED = [
    # Reading the code and the state is how an incident gets understood.
    'cd ~/obsidian && grep -n "_is_probe_scratch\\|sync-unlink\\|def check_services" '
    "skills/system-health-check/system_health_check.py",
    "grep -rn -- --vault-sync-round-trip ~/obsidian/skills/system-health-check/",
    "grep -rn LLOYD_VAULT_SYNC_ROUND_TRIP=1 ~/obsidian/skills",
    "grep -oE 'sync-unlink.{0,700}' /usr/lib/obsidian-headless/cli.js",
    "ls -la ~/.config/obsidian-headless/sync/",
    "cat ~/.config/obsidian-headless/sync/566dda12/config.json",
    "find ~/.config/obsidian-headless -type f",
    "sqlite3 -readonly ~/.config/obsidian-headless/sync/566dda12/state.db .tables",
    "stat ~/.config/obsidian-headless/sync",
    "sed -n 1,40p ~/.config/obsidian-headless/sync/566dda12/config.json",
    # The read-only ob surface.
    "ob sync-status --path ~/obsidian",
    "ob sync-list-local",
    "ob sync-list-remote",
    "ob login",
    "ob --help",
    "ob sync-unlink --help",
    # The health check without its end-to-end leg, which is the routine run.
    "python3 ~/obsidian/skills/system-health-check/system_health_check.py --component vault_sync",
    "LLOYD_VAULT_SYNC_ROUND_TRIP=0 python3 system_health_check.py",
    "echo 'never pass --vault-sync-round-trip'",
    'printf "%s\\n" "--vault-sync-round-trip is disabled"',
    "git -C ~/obsidian log --oneline -- skills/system-health-check/",
    # `sync-config` with only a path prints the configuration (the incident
    # chat's own read of it, 2026-09-14).
    'OB=/usr/bin/ob\n"$OB" sync-config --path ~/obsidian </dev/null 2>&1 | head -40',
    # Heredoc bodies are data: a commit message and a script editing the skill's
    # prose, both replayed from 2026-09-14 sessions.
    "git commit -q -F - <<'MSG'\n#538: pin it\n\nagent-obsidian-sync is a continuous `ob sync` under\n"
    "autorestart=true, so a client that fails stays RUNNING\nMSG",
    "python3 - <<'EOF'\nfrom pathlib import Path\np = Path('SKILL.md')\n"
    "p.write_text(p.read_text().replace('opt-in', 'never pass --vault-sync-round-trip'))\nEOF",
    "cat <<'EOF'\nob sync-unlink deletes by vault id\nEOF",
    # Neighbours that only look alike.
    "$EDITOR notes.md",
    "rm ~/.config/obsidian/Cache/old.log",
    "python3 -c \"print('ob is short for obsidian')\"",
]


@pytest.mark.parametrize("command", REFUSED)
def test_refused(home, command):
    assert check_sync_registration(command) is not None, command
    # Some spellings (`rm -rf ~/…`) the older regex table refuses first, under
    # its own label; refused is the property.
    assert check_bash_command(command, at_dispatch=True) is not None, command


@pytest.mark.parametrize("command", ALLOWED)
def test_allowed(home, command):
    assert check_sync_registration(command) is None, command
    assert check_bash_command(command, at_dispatch=True) is None, command


def test_the_refusal_names_what_it_protects(home):
    why = check_sync_registration(INCIDENT[0])
    assert "round trip" in why and "registration" in why


class _Recorder:
    def __init__(self):
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, dict(arguments)))
        return [M.TextContent(type="text", text="ran")]


async def test_the_aggregator_refuses_it_for_an_ordinary_chat(home, monkeypatch):
    """The 14:15 call came from a Mission Control chat, not a sandboxed bench
    session and not a path that installed the harness hook — so the refusal has
    to hold at dispatch, for any session id."""
    rec = _Recorder()
    table = dict(getattr(M, "_dispatch", None) or {})
    table["Bash"] = rec
    monkeypatch.setattr(M, "_dispatch", table)
    result = await M.call_tool("Bash", {"command": INCIDENT[1]},
                               {M.META_SESSION_ID: "20260914_190323_iv2eca"})
    text = "\n".join(getattr(b, "text", "") for b in (getattr(result, "content", None) or result))
    assert rec.calls == [], "the command reached the shell"
    assert "Obsidian Sync registration" in text
    ok = await M.call_tool("Bash", {"command": "ob sync-status --path ~/obsidian"},
                           {M.META_SESSION_ID: "20260914_190323_iv2eca"})
    assert rec.calls and rec.calls[0][0] == "Bash"
    assert ok is not None
