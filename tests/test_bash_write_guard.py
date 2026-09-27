"""The Bash lane refuses a write into the deny-set, as the file lane already does.

#1049 shut `Write` and `Edit` on the four entries of `PROTECTED_WRITE_ROOTS` and
left the lane that reaches a shell unasked. Measured at `580e335d`, the commit
this round was opened from, with `.venvs/lloyd/bin/python` calling
`check_bash_command(cmd, cwd=<the lloyd tree>)`: every command in
`DENY_SET_WRITES` below returned None on the harness PreToolUse hook lane *and*
with `at_dispatch=True`, while `agent_mcp/builtin_fs.py:238-239` refused the
identical target with "the OpenClaw credential tree", "the supervisor, guardian
and service units", "the identity file loaded into every system prompt". So
`Read` over `~/obsidian/lloyd/SOUL.md` was refused, and
`mv /tmp/evil ~/obsidian/lloyd/SOUL.md` was one Bash call away from the same
bytes — the item's own first triage line: "the same targets are refused on the
file lane … So `Read`/`Write` over SOUL.md is shut and
`echo x > ~/obsidian/lloyd/SOUL.md` is one Bash call."

Refusing is half of this file. The other half lives in
`tests/test_safety_gate_discrimination.py`, because the obvious mechanism fails
it: `protected_paths.referenced_paths` is over-inclusive by design, so
`cat ~/lloyd/agent-services/supervisord.conf` names a deny-set path (reading the
supervisor conf is routine triage, and was this item's own first step) and
`sed -i 's/a/b/' notes.md` yields a path that exists nowhere. What is refused is
the operand a command *writes*, which is why every command below writes its
target and every command in the other file only names one.

Scratch `$HOME` throughout, as the fs-lane file has it: the deny-set is
home-relative and resolved per call, so the live vault, the live service units
and the live interpreter are never a target here. The absolute spellings below
are the scratch home's own — the same entry the live spelling in the triage
record names, one `HOME` swap away.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_mcp import main as M  # noqa: E402
from app.harness import protected_paths as PP  # noqa: E402
from app.harness.safety import check_bash_command  # noqa: E402

SID = "20260927_143507_chatabc"
ORIGINAL = "ORIGINAL IDENTITY FILE"

#: One existing file per deny-set entry, vault-relative to the scratch home.
#: Same four entries `tests/test_builtin_fs_protected_write.py` writes to.
DENIED = {
    "identity file": "obsidian/lloyd/SOUL.md",
    "credential tree": ".openclaw/config.json",
    "service unit": "lloyd/agent-services/supervisor/conf.d/agent-backend.conf",
    "venv config": "lloyd/.venvs/pyvenv.cfg",
}

#: The eight spellings the triage record measured as PASS on both lanes, plus the
#: mover and in-place-editor forms the same clause names. Each writes its
#: deny-set target: `{home}` is the scratch home and `{repo}` its `lloyd` tree.
DENY_SET_WRITES = {
    "redirect-tilde-credential": "echo x > ~/.openclaw/credentials.json",
    "redirect-absolute-credential": "echo x > {home}/.openclaw/config.json",
    "redirect-printf-absolute-service": "printf x > {repo}/agent-services/supervisord.conf",
    "tee-append-service": "echo x | tee -a ~/lloyd/agent-services/supervisord.conf",
    "cp-destination-venv": "cp /tmp/evil ~/lloyd/.venvs/pyvenv.cfg",
    "install-destination-service": "install -m 644 /tmp/x {repo}/agent-services/y.conf",
    "mv-destination-identity": "mv /tmp/evil ~/obsidian/lloyd/SOUL.md",
    "sed-in-place-service": "sed -i 's/a/b/' ~/lloyd/agent-services/conf/guardian.env",
    "interpreter-open-write": "python3 -c \"open('{home}/.openclaw/x','w').write('x')\"",
    "cd-relative-sed-in-place": "cd ~/lloyd && sed -i 's/x/y/' agent-services/supervisord.conf",
}


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A scratch `$HOME` holding one real file per deny entry, plus the ordinary
    tree a command might legitimately write."""
    h = tmp_path / "home"
    for rel in list(DENIED.values()) + ["lloyd/agent-services/supervisord.conf",
                                       "lloyd/agent-services/conf/guardian.env",
                                       ".openclaw/credentials.json"]:
        p = h / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(ORIGINAL)
    (h / "obsidian" / "lloyd" / "SOUL.md").write_text(ORIGINAL)
    for rel in ("lloyd/app/a.py", "lloyd/app/b.py"):
        p = h / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x\n")
    (h / "obsidian" / "knowledge").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(h))
    return h


def _cmd(template: str, home: Path) -> str:
    return template.format(home=home, repo=home / "lloyd")


def _hook_decision(command: str, cwd: str) -> tuple[str, str]:
    """The harness PreToolUse hook's `(decision, reason)`. The hook also arms the
    outbound content gate, so a bare `deny` would pass on a sibling refusal: the
    reason has to name this check."""
    from app.harness import HookRegistry, install_default_safety_hook
    hooks = HookRegistry()
    install_default_safety_hook(hooks)
    out = asyncio.run(hooks.fire_pre_tool_use(
        session_id=SID, tool_name="Bash",
        tool_input={"command": command, "cwd": cwd}))
    spec = out.get("hookSpecificOutput") or {}
    return (spec.get("permissionDecision"), spec.get("permissionDecisionReason") or "")


# ── clauses 1 and 2: every write shape, on both enforcement lanes ───────────

@pytest.mark.parametrize("key", sorted(DENY_SET_WRITES))
def test_a_write_into_the_deny_set_is_refused_at_dispatch(home, key):
    """The aggregator's own `call_tool` runs exactly this call for every Bash
    dispatch (`agent_mcp/main.py:531`), whatever hooks the caller installed."""
    match = check_bash_command(_cmd(DENY_SET_WRITES[key], home), str(home / "lloyd"),
                              at_dispatch=True, session_id=SID)
    assert match is not None, DENY_SET_WRITES[key]
    assert match[0].startswith("protected write: "), match


@pytest.mark.parametrize("key", sorted(DENY_SET_WRITES))
def test_a_write_into_the_deny_set_is_refused_by_the_hook(home, key):
    command = _cmd(DENY_SET_WRITES[key], home)
    decision, reason = _hook_decision(command, str(home / "lloyd"))
    assert decision == "deny", command
    assert "protected write" in reason, (command, reason)


def test_the_refusal_names_the_shape_and_the_entry_it_belongs_to(home):
    """A block the operator cannot diagnose from the refusal log is a block that
    gets worked around, so the label carries the spelling and the deny-set's own
    reason, not just 'protected path'."""
    match = check_bash_command("echo x > ~/.openclaw/credentials.json",
                              str(home / "lloyd"), at_dispatch=True, session_id=SID)
    assert match is not None
    label = match[0]
    assert "redirect" in label, label
    assert str(home / ".openclaw" / "credentials.json") in label, label
    assert "OpenClaw credential tree" in label, label


#: Each clobbers (or, for the interpreter case, creates) its target when run.
_CLOBBERS = [
    ("echo CLOBBERED > ~/.openclaw/credentials.json", ".openclaw/credentials.json"),
    ("echo CLOBBERED | tee -a ~/lloyd/agent-services/supervisord.conf",
     "lloyd/agent-services/supervisord.conf"),
    ("sed -i 's/ORIGINAL/CLOBBERED/' ~/obsidian/lloyd/SOUL.md", "obsidian/lloyd/SOUL.md"),
    ("python3 -c \"open('{home}/.openclaw/created','w').write('x')\"", ".openclaw/created"),
]


def _run_unguarded(command: str, home: Path, cwd: Path):
    """Run `command` in a real shell with the write check monkeypatched off, so
    the byte assertions below are falsifiable: they name commands that *do* reach
    the target when nothing intercepts them, not commands that write nothing."""
    import subprocess
    env = {**os.environ, "HOME": str(home)}
    return subprocess.run(_cmd(command, home), shell=True, cwd=str(cwd), env=env,
                          capture_output=True, timeout=30)


@pytest.mark.parametrize("template,target", _CLOBBERS)
def test_the_same_commands_reach_the_bytes_with_the_check_removed(home, template, target):
    """The control that makes the assertion below mean something: unguarded, each
    command writes. `check_bash_write_denied` is looked up at call time by
    `check_bash_command`, so patching it here is the only thing standing down."""
    monkeypatched = PP.check_bash_write_denied
    PP.check_bash_write_denied = lambda command, cwd=None: None
    try:
        _run_unguarded(template, home, home / "lloyd")
    finally:
        PP.check_bash_write_denied = monkeypatched
    state = (home / target).read_text() if (home / target).exists() else None
    assert state not in (None, ORIGINAL), (template, state)


@pytest.mark.parametrize("template,target", _CLOBBERS)
def test_a_refused_write_leaves_the_denied_bytes_untouched(home, template, target):
    """Refused, and the bytes are what they were. Only the first assertion can
    fail — the check runs no shell, so the second is a guard against a refusal
    that reports *after* executing, and the node above is what shows these
    commands reach the target when nothing stands between them."""
    assert check_bash_command(_cmd(template, home), str(home / "lloyd"),
                             at_dispatch=True, session_id=SID), template
    state = (home / target).read_text() if (home / target).exists() else None
    assert state in (ORIGINAL, None), (template, state)


# ── clause 3: the write is only visible inside the interpreter's literal ─────

def test_a_write_reached_only_through_a_quoted_literal_is_refused(home):
    """`referenced_paths` is what sees this: the target never appears in the
    outer command's argv, only inside the string `python3 -c` was handed. Its
    falsifiable half is `test_the_same_commands_reach_the_bytes_with_the_check_removed`,
    which runs the interpreter spelling for real and sees the file appear."""
    target = home / ".openclaw" / "x"
    command = f"python3 -c \"open('{target}','w').write('x')\""
    match = check_bash_command(command, str(home / "lloyd"),
                              at_dispatch=True, session_id=SID)
    assert match is not None, command
    assert str(target) in match[0], match
    assert not target.exists()


@pytest.mark.parametrize("command", [
    # pathlib, node and the two-argument mover: the destination is a literal in
    # every case, and the mover's *first* literal is its source.
    "python3 -c \"from pathlib import Path; Path('~/lloyd/agent-services/x').write_text('y')\"",
    "python3 -c \"import shutil; shutil.copy('/tmp/x', '~/lloyd/.venvs/pyvenv.cfg')\"",
    "node -e \"require('fs').writeFileSync('/{home}/.openclaw/creds.json','x')\"",
    "bash -c 'echo x > ~/.openclaw/credentials.json'",
])
def test_a_write_shaped_interpreter_argument_is_refused(home, command):
    cmd = command.format(home=home)
    assert check_bash_command(cmd, str(home / "lloyd"), at_dispatch=True,
                             session_id=SID), cmd


def test_a_read_of_the_same_literal_shape_is_not_refused(home):
    """The pair the clause asks for in one node: `cat` of a deny-set path returns
    None. The interpreter's equivalent — `print(open('<deny-set>').read())` — is
    the same shape with no write mode in it, and is allowed for the same reason."""
    assert check_bash_command("cat ~/lloyd/agent-services/supervisord.conf",
                             str(home / "lloyd")) is None
    read = ("python3 -c \"print(open('"
            + str(home / "lloyd" / "agent-services" / "supervisord.conf")
            + "').read())\"")
    assert check_bash_command(read, str(home / "lloyd"), at_dispatch=True,
                             session_id=SID) is None


# ── the seam: a tool caller gets the refusal, not the shell ─────────────────

async def test_the_refusal_comes_back_over_the_aggregator_with_is_error(home):
    """`agent_mcp.main.call_tool` is the boundary every real Bash call crosses
    and where the command would otherwise reach a shell: the refusal arrives as an
    error result and the bytes are untouched."""
    target = home / "obsidian" / "lloyd" / "SOUL.md"
    await M.list_tools()
    res = await M.call_tool("Bash", {"command": f"echo CLOBBERED > {target}",
                                    "cwd": str(home / "lloyd")},
                           {"lloyd/session_id": SID})
    assert type(res).__name__ == "CallToolResult"
    assert res.is_error is True, res.content[0].text
    assert "protected write" in res.content[0].text, res.content[0].text
    assert target.read_text() == ORIGINAL


def test_the_grant_that_lifts_the_file_lane_lifts_this_lane_too(home):
    """One predicate, both lanes: the Bash check calls the same
    `write_deny_reason` `builtin_fs` calls, so `allow_protected_writes` is a
    permission on the deny-set and not on one tool. The clause's own pair —
    `tee` into `agent-services` passes under a lift naming it, `> ~/.openclaw/x`
    stays refused — is in `tests/test_builtin_fs_protected_write.py`, next to the
    lane that fixture belongs to."""
    command = "echo x | tee -a ~/lloyd/agent-services/supervisord.conf"
    assert check_bash_command(command, str(home / "lloyd"), at_dispatch=True,
                             session_id=SID) is not None
    with PP.allow_protected_writes("test: the nightly service-unit lane",
                                  paths=["~/lloyd/agent-services"]):
        assert check_bash_command(command, str(home / "lloyd"), at_dispatch=True,
                                 session_id=SID) is None
    assert check_bash_command(command, str(home / "lloyd"), at_dispatch=True,
                             session_id=SID) is not None, "the lift cannot outlive its scope"
