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
import json
import os
import shlex
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

#: One existing file per deny-set entry as #1049 shipped it, vault-relative to
#: the scratch home — the four that `tests/test_builtin_fs_protected_write.py`
#: writes to. The fifth (#1741) is not a file standing under its own name but the
#: uv store the venv interpreter links into, so it gets its own fixture,
#: `linked_home`, and the nodes below it: this dict's targets are refused by the
#: set that was here first, and a node that passes at the base commit pins
#: nothing.
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
    # A *shell's* heredoc body is shell code, so #1740's split hands it straight
    # back to the grammar instead of dropping it as data: the write inside it has
    # to stay refused, which is what separates the narrowing from a switch-off.
    "shell-heredoc-body-redirect": "bash <<'SH'\necho x > {home}/.openclaw/credentials.json\nSH\n",
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


#: The store layout every `.venvs/*/pyvenv.cfg` on the live box names, spelled
#: under a scratch home. `.venvs/lloyd/bin/python` is a symlink into it, which is
#: why the `~/lloyd/.venvs` entry never covered a write to that interpreter:
#: realpath-first judges the link by where it points, and the target was outside
#: the set until #1741 put the store in it.
UV_STORE_BIN = ".local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin"

#: The three spellings of one write: the interpreter every lloyd service runs on.
#: The tilde form is the live #1741 probe, `cp /tmp/evil
#: ~/lloyd/.venvs/lloyd/bin/python`, one `HOME` swap away; the second is its own
#: absolute spelling; the third names the store binary the link resolves to, so
#: the entry is pinned on both sides of the link and not just the one a person
#: happens to type. Each writes its target.
INTERPRETER_WRITES = {
    "cp-destination-venv-interpreter": "cp /tmp/evil ~/lloyd/.venvs/lloyd/bin/python",
    "cp-destination-venv-interpreter-absolute": "cp /tmp/evil {repo}/.venvs/lloyd/bin/python",
    "tee-destination-store-binary": "echo clobbered | tee {store}/python3.12",
}

#: What the refusal has to name: the fifth entry's own label (#1741).
STORE_LABEL = "the uv-managed CPython installs the lloyd venvs link to"


@pytest.fixture
def linked_home(tmp_path, monkeypatch):
    """A scratch `$HOME` whose venv interpreter is the symlink it is on the box.

    Not the plain file the fixture above creates: with a real file standing at
    `.venvs/lloyd/bin/python`, realpath keeps it inside `~/lloyd/.venvs`, the old
    four-entry set already refused the write, and every node below would pass at
    the base commit and prove nothing.
    """
    h = tmp_path / "home"
    store = h / UV_STORE_BIN
    store.mkdir(parents=True)
    (store / "python3.12").write_text(ORIGINAL)
    link = h / "lloyd" / ".venvs" / "lloyd" / "bin"
    link.mkdir(parents=True)
    (link / "python").symlink_to(store / "python3.12")
    (h / "lloyd" / ".venvs" / "pyvenv.cfg").write_text(f"home = {store}\n")
    for rel in ("obsidian/lloyd/SOUL.md", ".openclaw/credentials.json",
                "lloyd/agent-services/supervisord.conf"):
        p = h / rel
        p.parent.mkdir(parents=True)
        p.write_text(ORIGINAL)
    monkeypatch.setenv("HOME", str(h))
    return h


@pytest.mark.parametrize("key", sorted(INTERPRETER_WRITES))
def test_a_write_to_the_linked_interpreter_is_refused_at_dispatch(linked_home, key):
    """The dispatch lane, which is the one `agent_mcp.main.call_tool` runs for
    every Bash call. All three keys are refused only because of the fifth entry:
    delete that entry and each of these returns None at the base commit — the
    falsifiable half of this node."""
    cmd = INTERPRETER_WRITES[key].format(home=linked_home, repo=linked_home / "lloyd",
                                         store=linked_home / UV_STORE_BIN)
    match = check_bash_command(cmd, str(linked_home / "lloyd"), at_dispatch=True,
                               session_id=SID)
    assert match is not None, cmd
    assert match[0].startswith("protected write: "), match


@pytest.mark.parametrize("key", sorted(INTERPRETER_WRITES))
def test_a_write_to_the_linked_interpreter_is_refused_by_the_hook(linked_home, key):
    cmd = INTERPRETER_WRITES[key].format(home=linked_home, repo=linked_home / "lloyd",
                                         store=linked_home / UV_STORE_BIN)
    decision, reason = _hook_decision(cmd, str(linked_home / "lloyd"))
    assert decision == "deny", cmd
    assert "protected write" in reason, (cmd, reason)


def test_the_interpreter_refusal_names_the_store_entry(linked_home):
    """The refusal an operator sees must name the entry that fired, and here it is
    the store's, not `~/lloyd/.venvs`'.

    The message quotes the operand as it was typed — `.../lloyd/.venvs/lloyd/bin/python`
    — while the label it carries is the fifth entry's, because realpath-first moved
    the target into the store. That combination is what an operator needs: the
    spelling they typed, and the tree that is actually shut, which is not the one
    the path is standing in. Before #1741 the same call was refused by neither.
    """
    command = "cp /tmp/evil ~/lloyd/.venvs/lloyd/bin/python"
    match = check_bash_command(command, str(linked_home / "lloyd"), at_dispatch=True,
                              session_id=SID)
    assert match is not None, command
    assert STORE_LABEL in match[0], match[0]
    assert "the interpreter the lloyd services run on" not in match[0], match[0]
    assert str(linked_home / "lloyd/.venvs/lloyd/bin/python") in match[0], match[0]


def test_the_store_entry_covers_the_link_only_because_the_link_is_a_link(linked_home):
    """The control that makes the two nodes above mean something: the target is
    refused *through* the symlink, and the same name as an ordinary file under
    `.venvs` is refused by the venv entry instead. Drop the fifth entry and the
    first assertion goes None while the second stays refused — that difference is
    the whole of #1741, and a fixture that made the interpreter a plain file
    would hide it."""
    link = linked_home / "lloyd" / ".venvs" / "lloyd" / "bin" / "python"
    assert link.is_symlink() and link.resolve() == (linked_home / UV_STORE_BIN
                                                   / "python3.12").resolve()
    assert PP.write_deny_reason(str(link)) == STORE_LABEL, "the link, judged by its target"
    assert PP.write_deny_reason(str(linked_home / "lloyd/.venvs/pyvenv.cfg")) \
        == "the interpreter the lloyd services run on", "a real file inside the venv"


def test_a_grant_naming_the_store_lifts_the_bash_write_and_nothing_else(linked_home):
    """The pair on the Bash lane: a lift naming the store opens the interpreter
    write, and leaves `> ~/.openclaw/x` refused, because a narrowed grant is a
    permission on a location and not on the concept of protected paths."""
    interpreter = "cp /tmp/evil ~/lloyd/.venvs/lloyd/bin/python"
    redirect = "echo x > ~/.openclaw/credentials.json"
    for cmd in (interpreter, redirect):
        assert check_bash_command(cmd, str(linked_home / "lloyd"), at_dispatch=True,
                                  session_id=SID) is not None, cmd
    with PP.allow_protected_writes("test: the uv store lane",
                                   paths=["~/.local/share/uv/python"]):
        assert check_bash_command(interpreter, str(linked_home / "lloyd"),
                                  at_dispatch=True, session_id=SID) is None, interpreter
        assert check_bash_command(redirect, str(linked_home / "lloyd"),
                                  at_dispatch=True, session_id=SID) is not None, redirect
    assert check_bash_command(interpreter, str(linked_home / "lloyd"),
                              at_dispatch=True, session_id=SID) is not None, \
        "the lift cannot outlive its scope"


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
    """Run `command` in a real shell, unguarded, so the byte assertions below are
    falsifiable: they name commands that *do* reach the target when nothing
    intercepts them, not commands that write nothing.

    Unguarded because it is a plain `subprocess.run(shell=True)` in a child that
    never imports `protected_paths` and never fires the harness hook — the two
    things that refuse the same string when the harness runs it. Nothing in the
    parent process can stand this one down, which is exactly why it is the
    control: if the command below does not write here, no assertion about a
    refusal protecting those bytes means anything.
    """
    import subprocess
    env = {**os.environ, "HOME": str(home)}
    return subprocess.run(_cmd(command, home), shell=True, cwd=str(cwd), env=env,
                          capture_output=True, timeout=30)


@pytest.mark.parametrize("template,target", _CLOBBERS)
def test_the_same_commands_reach_the_bytes_with_the_check_removed(home, template, target):
    """The control that makes the assertion below mean something: unguarded, each
    command writes.

    It writes because of what `_run_unguarded` is — a shell in a child process
    that never imports `protected_paths`, so neither the in-process predicate nor
    the PreToolUse hook is in front of it. The assignment below is not what stands
    the check down here, and never was: `check_bash_command` imports
    `check_bash_write_denied` at call time, but this node never calls
    `check_bash_command`. The assignment is left in as the belt-and-braces it is,
    and the falsifiable content of this node is entirely in the read-back."""
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


# ── #1740: a payload's write calls are read; its data is not ────────────────
#
# The item's two directions meet in these nodes, and they pull against each other.
# A deny-set path that sits *inside program text* — a case-table entry, a patch
# hunk, a line to grep for — is data the command carries, and up to #1740 it came
# back as a write target: `_tokens` rewrites every "\n" to " ; " and the only thing
# skipped was the heredoc *word*, so a body was visited as its own command segments
# and a `>` anywhere in it named whatever followed as a redirect. The same path in
# a write call the payload actually *performs* is a write, and on `-c` it refused
# while on stdin it returned None — `python3 -` has no operand, and a body read as
# shell contains no shell redirect. So one node below closes the false positives
# and another closes the under-block; either one dropped on its own is the bug
# recurring in the other direction.

#: The write calls `write_targets`' payload arm knows, as templates over one
#: target: `open` with a mode argument, and `pathlib`'s writer. Two because the arm
#: is a set of write-call patterns, not a match on the word `open`.
PAYLOAD_WRITES = (
    "open('{t}','w').write('x')",
    "from pathlib import Path\nPath('{t}').write_text('x')",
)


def _payload_on_stdin(body: str) -> str:
    """A body handed to python the way a patch script is written: `-` as the
    operand and a quoted heredoc word, so the shell expands nothing inside it."""
    return "python3 - <<'PY'\n" + body + "PY\n"


@pytest.mark.parametrize("write_form", PAYLOAD_WRITES)
def test_an_interpreter_write_refuses_on_stdin_as_on_the_flag(home, write_form):
    """#1740 clause 2. A payload's own write is refused whichever way it arrives.

    Measured at `b002b2e9` the `-c` spelling refused and the stdin one returned
    None, so feeding a program on stdin was an under-block that a fix narrowing
    what bodies get parsed as shell would have widened rather than closed. Both
    spellings are asserted on both lanes, and `write_targets` is checked to name
    the target under the interpreter shape — the shape the refusal quotes — so the
    node cannot pass on an unrelated target.
    """
    target = home / ".openclaw" / "created-by-the-command"
    call = write_form.format(t=target)
    on_flag = "python3 -c " + shlex.quote(call)
    on_stdin = _payload_on_stdin(call + "\n")
    # The same write one line after an apostrophe in prose. `_payload_literals`
    # skips a quote with nothing to close it on its own line, so a comment cannot
    # mask the statement below it: an unescape rule that hid this would be the
    # under-block this round was written to close, wearing a fix's clothes.
    after_prose = _payload_on_stdin("# don't mind this line\n" + call + "\n")
    for cmd in (on_flag, on_stdin, after_prose):
        match = check_bash_command(cmd, str(home / "lloyd"), at_dispatch=True,
                                   session_id=SID)
        assert match is not None, cmd
        assert str(target) in match[0], match
        assert not target.exists(), "the check is pre-dispatch, not a wrapper"
        decision, reason = _hook_decision(cmd, str(home / "lloyd"))
        assert decision == "deny", (cmd, decision, reason)
        assert "protected write" in reason, reason


def test_a_deny_set_path_carried_as_data_inside_a_payload_is_not_a_write(home):
    """#1740 clause 3. The fourth false positive, replayed: a `python -c` whose
    payload holds the write only as a case-table string must not refuse.

    The string is built with `json.dumps` rather than typed so its escaping is the
    real thing: one level in, the inner quotes arrive to the guard as `\\"`, which
    is what marks them as text *inside* a literal of the payload rather than as
    statements of it. That distinction is the whole of the fix — a regex over the
    payload cannot see it, and at `b002b2e9` this command returned the target with
    the shape `'a write from an interpreter to'` and refused on both lanes.
    The stdin spelling is asserted alongside it because the payload arm now runs on
    heredoc bodies too, and it must inherit the same data-not-code rule; it did not
    refuse at the base commit only because nothing read the body at all.
    """
    target = home / ".openclaw" / "x"
    entry = "python3 -c \"open('" + str(target) + "','w').write('x')\""
    payload = ("import json\nfor line in [" + json.dumps(entry)
               + "]:\n    print(line)\n")
    commands = ["python3 -c " + shlex.quote(payload), _payload_on_stdin(payload)]
    # The same pair with the quoted write handed to a function that *describes* it,
    # beside a deny-set path as its own argument. Nothing here writes: both paths
    # are arguments. This is the shape that pins the rule as such rather than as a
    # side effect of resolution — read as a call, the quoted `open(` is answered
    # with the literal that follows it in the same statement window, and the node's
    # `write_targets(...) == []` is what turns red.
    described = ("describe(\"open('" + str(target) + "','w').write('x')\", '"
                 + str(home / ".openclaw" / "credentials.json") + "')\n")
    commands += ["python3 -c " + shlex.quote(described), _payload_on_stdin(described)]
    # And the same write mentioned in a comment, which is where most of the
    # deny-set paths in a real patch script actually appear. The apostrophe in
    # `don't` has no closing quote before the path, so the comment's text is what
    # the scan sees, not a call.
    commented = ("# don't reach for open('" + str(target) + "','w') here\n"
                 "print('skipped')\n")
    commands += ["python3 -c " + shlex.quote(commented), _payload_on_stdin(commented)]
    for cmd in commands:
        assert str(target) in cmd and "open(" in cmd and "'w'" in cmd, cmd
        assert PP.write_targets(cmd, str(home / "lloyd")) == [], cmd
        assert check_bash_command(cmd, str(home / "lloyd"), at_dispatch=True,
                                 session_id=SID) is None, cmd
        decision, reason = _hook_decision(cmd, str(home / "lloyd"))
        assert "protected write" not in reason, (cmd, decision, reason)


def test_no_write_target_from_the_reference_traffic_ends_in_a_quote(home):
    """#1740 clause 4. `shlex` hands back a token with its quotes stripped, so a
    path that still ends in one came from text the shell grammar was never meant to
    parse — every false positive the item measured arrived that way.

    Each command is pinned to the targets it is meant to produce, not merely to the
    absence of a quote: two of the four are ordinary writes, so a guard that
    answered nothing for anything would fail here rather than pass. Measured at
    `b002b2e9` the case-table body returned three targets, two of them ending in
    `",` (`…/credentials.json",`) and both in the deny set; here it returns the one
    the command's own redirect names, `/tmp/probe.py`.
    """
    corpus = [
        # The clause-1 shape: a `cat` body whose lines are probe commands.
        "cat > /tmp/probe.py <<'EOF'\n"
        "cases = [\n"
        '    "echo x > ~/.openclaw/credentials.json",\n'
        '    "echo x | tee -a ~/lloyd/agent-services/supervisord.conf",\n'
        "    'sed -i \\'s/a/b/\\' notes.md',\n"
        '    "mv /tmp/evil ~/obsidian/lloyd/SOUL.md",\n'
        "]\n"
        "EOF\n",
        # A patch script of the shape the second reference session ran: the deny-set
        # paths are entries of a table it will visit later, and a `>` sits in prose.
        _payload_on_stdin("TARGETS = [\n    '"
                          + str(home / "lloyd" / "agent-services" / "link.conf")
                          + "',\n    '" + str(home / ".openclaw" / "credentials.json")
                          + "',\n]\nfor p in TARGETS:\n    print('would patch', p, "
                          "'-> done')\n"),
        # Two writes the guard does have to see, so the node cannot pass on a guard
        # that answers nothing for anything.
        f"cp /tmp/a.py {home}/lloyd/scripts/b.py",
        f"sed -i 's/x/y/' {home}/notes.md",
    ]
    expected = [["/tmp/probe.py"], [],
                [str(home / "lloyd" / "scripts" / "b.py")],
                [str(home / "notes.md")]]
    assert sum(len(e) for e in expected) == 3, "the positive control is vacuous"
    for cmd, want in zip(corpus, expected):
        targets = PP.write_targets(cmd, str(home / "lloyd"))
        for path, shape in targets:
            assert not path.endswith(('"', "'")), (path, shape, cmd)
        assert [p for p, _shape in targets] == want, cmd
