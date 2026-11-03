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


def test_every_write_shape_here_is_refused_and_nothing_can_narrow_the_set(linked_home):
    """#3563: the narrowing this node used to pin is gone, so the refusal it
    pinned is what has to hold for all five locations instead.

    Before the delete, a lift naming ONE location opened the interpreter write
    and left `> ~/.openclaw/x` refused — a permission on a location, not on the
    concept. Both halves are now the same half: each of the five deny-set
    locations is refused, and there is no second input to `write_deny_reason`
    that could answer one location and not another. The interpreter symlink is
    the sharpest of the five, because the path named on the command line is
    outside the set and only its realpath is inside it."""
    interpreter = "cp /tmp/evil ~/lloyd/.venvs/lloyd/bin/python"
    redirect = "echo x > ~/.openclaw/credentials.json"
    for cmd in (interpreter, redirect):
        assert check_bash_command(cmd, str(linked_home / "lloyd"), at_dispatch=True,
                                  session_id=SID) is not None, cmd
    assert PP.write_deny_reason(str((linked_home / ".local/share/uv/python").resolve())) \
        == "the uv-managed CPython installs the lloyd venvs link to", \
        "the store the removed narrowing used to name is still in the set"
    lifts = sorted(n for n in dir(PP) if n.endswith("protected_writes"))
    assert not lifts, \
        f"an in-process write lift came back on the module: {lifts}"


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


def test_the_service_unit_write_is_refused_on_the_bash_lane_with_no_lift_behind_it(home):
    """#3563: `tee` into `agent-services` is refused here and on `Write`, and the
    two lanes cannot diverge because they read the same one-input predicate.

    This node used to prove the lanes could be opened *together* by one grant,
    which was the same property read from the wrong side: a shared lift proves
    agreement only while the lift exists. With it deleted, agreement is the only
    state either lane can be in — so the pair this node's docstring pointed at in
    `tests/test_builtin_fs_protected_write.py` is now the pair of refusals there,
    and the `paths=[…]` narrowing this node exercised has no argument left to
    take it."""
    command = "echo x | tee -a ~/lloyd/agent-services/supervisord.conf"
    match = check_bash_command(command, str(home / "lloyd"), at_dispatch=True,
                               session_id=SID)
    assert match is not None, "the service-unit write went through"
    assert match[0].startswith("protected write: "), match
    assert PP.write_deny_reason(str(home / "lloyd/agent-services/supervisord.conf")) \
        == "the supervisor, guardian and service units", \
        "the location the removed narrowing used to name is still in the set"


@pytest.mark.parametrize("root,label", sorted(PP.PROTECTED_WRITE_ROOTS))
def test_a_write_shape_naming_each_of_the_five_roots_is_refused(home, root, label):
    """#3563 clause 3, one root per node: the guard's purpose did not widen with
    the lift removed, because the same five locations are what it covered before.

    The three roots the deleted `paths=[…]` arguments named — `~/lloyd/.venvs`,
    the uv store and `~/lloyd/agent-services` — are the ones this matters for: a
    narrowing used to be able to open exactly one of them, so a refusal here is
    the whole of what that argument used to be tested against. A redirect is
    enough to reach the check; no path has to exist, because the deny-set is
    prefix matching on a resolved path."""
    target = root.replace("~", str(home), 1) + "/x"
    match = check_bash_command(f"echo x > {target}", str(home / "lloyd"),
                               at_dispatch=True, session_id=SID)
    assert match is not None, f"{root} was writable on the Bash lane"
    assert label in match[0], f"the refusal does not name the label for {root}: {match}"


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


# ── #3701 clause 4: a redirect that CREATES a file under the vault tree ───────
#
# The 2026-11-01 incident, from #3696: an unattended session ran
# `cat >> ~/obsidian/backlog/<hand-guessed-slug>.md` and minted a second `3691-*.md`
# with no front matter. Nothing refused it. `write_targets` had already resolved and
# classified that very target — `('…/backlog/probe.md', 'a redirect into')` — and
# `check_bash_write_denied` answered None, because the predicate under it,
# `write_deny_reason`, covers ONE vault file (`~/obsidian/lloyd/SOUL.md`) and no vault
# tree. The tree cannot join `PROTECTED_WRITE_ROOTS`: `builtin_fs` and `vault_write`
# both ask that predicate, and the nightly reflection's sanctioned route for
# `lloyd/USER.md` and `MEMORY.md` is that lane, so covering the tree there takes the
# reflection out every night. The rule therefore goes on the Bash lane, which has no
# sanctioned vault writer at all — and clause 5 in
# `tests/test_protected_paths.py` is the half that stays open.
#
# Every target below is a path that does NOT exist in the scratch home: the guard is
# about creating a file in the vault, so a check that keyed on the file being there
# already would miss the incident exactly.

VAULT_TREE_WRITES = {
    "redirect-creates-board-file-absolute": "cat > {home}/obsidian/backlog/probe.md",
    "redirect-creates-board-file-heredoc": "cat >> ~/obsidian/backlog/probe.md <<'EOF'\n"
                                           "- one correction entry\nEOF\n",
    "redirect-append-tilde-memory": "echo x >> ~/obsidian/memory/corrections.md",
    "redirect-printf-vault-knowledge": "printf x > {home}/obsidian/knowledge/probe.md",
}


@pytest.mark.parametrize("key", sorted(VAULT_TREE_WRITES))
def test_a_redirect_creating_a_file_under_the_vault_tree_is_refused_on_the_bash_lane(home, key):
    """The one-line probe from #3701's acceptance, on every redirect spelling.

    Measured at `bd0366ed`, all four returned None here while `write_targets` named
    the resolved path as `a redirect into`. The refusal must carry the resolved path:
    an operator told only "inside the vault" cannot tell which file a run reached for.
    """
    command = _cmd(VAULT_TREE_WRITES[key], home)
    reason = PP.check_bash_write_denied(command, str(home / "lloyd"))
    assert reason is not None, f"{key}: the vault tree let a shell redirect through"
    assert "redirect" in reason, f"{key}: {reason}"
    assert str(home / "obsidian") in reason, \
        f"{key}: the refusal must name the resolved vault path, got {reason}"
    assert "(#3701)" in reason, f"{key}: {reason}"


def test_the_vault_redirect_refusal_reaches_the_dispatch_lane_with_the_write_label(home):
    """Not only the predicate: the aggregator's own Bash call refuses it.

    `check_bash_command` puts `check_bash_write_denied` fifth in its chain
    (`app/harness/safety.py:521`), and `tests/unit/test_harness_safety.py` pins which
    lane answers each positive, so this pins the label as well as the refusal — the
    vault rule must arrive under `protected write: `, the same label #1049's deny-set
    uses, and not under a new one a deny-log reader would have to learn.
    """
    command = _cmd(VAULT_TREE_WRITES["redirect-creates-board-file-heredoc"], home)
    match = check_bash_command(command, str(home / "lloyd"), at_dispatch=True,
                              session_id=SID)
    assert match is not None, "the heredoc redirect reached the dispatch lane unanswered"
    assert match[0].startswith("protected write: "), match[0]
    assert str(home / "obsidian") in match[0], match[0]


def test_the_vault_tree_is_covered_on_the_bash_lane_without_joining_the_deny_set(home):
    """The deny-set is untouched — the clause that keeps the reflection alive.

    Adding `~/obsidian` to `PROTECTED_WRITE_ROOTS` would refuse every one of these
    commands AND the nightly reflection's sanctioned `vault_write` of
    `lloyd/USER.md`/`MEMORY.md`, which is the outage the module header at
    `protected_paths.py:204` was written to prevent. So the root lives in its own
    constant, and this node is what stops the shortcut.
    """
    templates = [t for t, _label in PP.PROTECTED_WRITE_ROOTS]
    assert "~/obsidian" not in templates, \
        "the vault tree went into the deny-set — the file lane now refuses the reflection"
    assert PP.VAULT_BASH_WRITE_ROOT == "~/obsidian"


def test_a_mover_into_the_vault_tree_is_not_yet_refused_because_that_ruling_is_owed(home):
    """`cp` into the vault still passes, and this node says so on purpose.

    `write_targets` classifies `cp`, `install`, `ln`, `mv`, `rsync` and `tee` with
    shapes of their own, so widening the root over all of them is a one-entry change to
    `VAULT_BASH_WRITE_SHAPES`. #3701 names redirects only and records the widening as
    owed to the owed-check job; a node that asserted the mover refusal today would take
    that ruling by accident and leave no evidence of the decision.
    """
    mover = _cmd("cp /tmp/x.txt {home}/obsidian/backlog/moved-by-cp.md", home)
    assert PP.check_bash_write_denied(mover, str(home / "lloyd")) is None, \
        "a mover into the vault was refused — the owed widening was taken without a ruling"
    assert PP.VAULT_BASH_WRITE_SHAPES == ("a redirect into",), PP.VAULT_BASH_WRITE_SHAPES


# ── #3737: the refusal for a witness-extract target names the lane that works ────
#
# Two refusals on 2026-11-03, both aimed at `backlog/data/`, both answered badly
# because the sentence the session was handed offered only lanes that cannot carry
# witness bytes. Both rows of `~/lloyd-data/safety/denials.jsonl`, copied from that
# journal at this round's base commit:
#
#   * `at: 2026-11-03T15:05:04+00:00`, session `20261103_060506_autocode_1237` —
#     `sed -n '6,11p' $L > backlog/data/2026-11-01.3541-gate-to-land-lag-witness.jsonl`.
#     Answered by hand-emitting the block through `vault_write`'s content argument: the
#     committed artifact reads `wc -l -c` → `6 6371` at vault `f7fa7bca2`, and
#     `~/obsidian/backlog/3686-commit-the-gate-to-land-lag-witness-bytes-for-3541.md:139`
#     criticises exactly that — the bytes "travelled through `vault_write`'s content
#     argument, i.e. through the model's own generation, which is exactly what
#     'byte-identical' cannot be assumed of".
#   * `at: 2026-11-03T17:29:07+00:00`, session `20261103_092822_autocode_3774` —
#     `head -637 ~/lloyd-data/safety/denials.jsonl > backlog/data/2026-10-27.3365-safety-denials.jsonl`.
#     Never answered at all: `ls ~/obsidian/backlog/data/2026-10-27.3365-safety-denials.jsonl`
#     was still `No such file or directory` when this round opened, and is still absent
#     at this round's base.
#
# Both rows carry `guard: "safety"` and `where: "hook"` — the harness PreToolUse hook
# lane, not the aggregator's dispatch-lane check, which journals `where: "dispatch"` and
# fired on two of this journal's nine vault rows and never on a witness target. So the
# lane that must carry the advice is the HOOK's `permissionDecisionReason`, and
# `test_the_witness_staging_advice_reaches_the_session_on_the_hook_lane` is the node that
# stands across it; the dispatch-lane node below is kept because that lane is the
# unconditional backstop and every later session may be refused by either.
#
# The lane that works — stage the bytes OUTSIDE the vault, move them in with `cp` or
# `install -m 644`, then `automod_vault_land` the path — is open: #3701 refused the
# redirect shape alone, and this file's own
# `test_a_mover_into_the_vault_tree_is_not_yet_refused_because_that_ruling_is_owed`
# measures it as open on purpose. It was named nowhere a refused session could read:
# not in `VAULT_BASH_WRITE_LABEL`, not on the board. The nodes below pin the sentence
# for that one directory and pin that it stops there. Widening `VAULT_BASH_WRITE_SHAPES`
# is NOT decided here — the owed #3701 mover ruling stays owed, so the mover node above
# this block passes unmodified, and so does every one of these targets' `cp`/`install`.

#: Redirects into the witness-artifact directory (`scripts/automod/backlog.py`'s
#: `WITNESS_ARTIFACT_DIR`, `backlog/data`), in the three spellings the journal shows.
WITNESS_DIR_WRITES = {
    "redirect-creates-witness-jsonl": "sed -n '6,11p' /tmp/ledger.jsonl > "
                                      "{home}/obsidian/backlog/data/witness.jsonl",
    "redirect-creates-witness-tilde": "echo x > ~/obsidian/backlog/data/probe.jsonl",
    "redirect-append-witness-marker": "echo x >> "
                                      "{home}/obsidian/backlog/data/2026-11-03.3737-marker.md",
}

#: Targets elsewhere in the SAME vault tree. Their refusal must stay the sentence #3701
#: shipped, byte for byte: an extract aimed at the board or a daily note has no staging
#: lane to be told about, and advice naming one would be a claim about a directory the
#: target is not in.
NON_WITNESS_VAULT_WRITES = {
    "board-file": "cat > {home}/obsidian/backlog/probe.md",
    "daily-note": "echo x > {home}/obsidian/memory/2026-11-03.md",
    "knowledge-note": "printf x > {home}/obsidian/knowledge/probe.md",
}

#: The fragments clause 1 demands in a witness refusal. `cp` and `automod_vault_land` are
#: in it because clause 1 names them — but only the three in
#: `WITNESS_STAGING_ONLY_MARKERS` can also be asserted ABSENT elsewhere in the vault:
#: #3701's own sentence already ends "use vault_write or automod_vault_land", so a
#: non-witness refusal that still says exactly that is correct, and an absence check on
#: `automod_vault_land` would fail on the sentence this item does not change.
WITNESS_STAGING_MARKERS = (
    "witness-artifact directory",
    "outside the vault",
    "cp",
    "install -m 644",
    "automod_vault_land",
)

#: The staging wording that belongs to the new sentence ALONE, and so is what clause 2
#: asserts is missing from a refusal for any other vault target.
WITNESS_STAGING_ONLY_MARKERS = (
    "witness-artifact directory",
    "outside the vault",
    "install -m 644",
)


@pytest.mark.parametrize("key", sorted(WITNESS_DIR_WRITES))
def test_a_witness_extract_redirect_refusal_names_the_copy_then_land_convention(home, key):
    """Clause 1: a `backlog/data/` refusal says how byte-exact witness bytes get in.

    The dispatch lane and not only the predicate: `check_bash_command` is the aggregator's
    unconditional backstop (`agent_mcp/main.py`), so a session refused there has to read
    the advice too — the lane the two journal rows actually came through is the hook, and
    `test_the_witness_staging_advice_reaches_the_session_on_the_hook_lane` stands across
    that one. It arrives under `protected write: ` like every other write refusal — the
    deny-log reader keeps one prefix — and keeps `(#3701)`, because the base clause is
    still the reason the redirect is refused; the new sentence only adds where the bytes
    should have been built.
    """
    command = _cmd(WITNESS_DIR_WRITES[key], home)
    match = check_bash_command(command, str(home / "lloyd"), at_dispatch=True,
                               session_id=SID)
    assert match is not None, f"{key}: the witness redirect reached the lane unanswered"
    reason = match[0]
    assert reason.startswith("protected write: "), reason
    assert str(home / "obsidian") in reason, f"{key}: must name the resolved target: {reason}"
    assert "(#3701)" in reason, f"{key}: the base clause was replaced, not extended: {reason}"
    for marker in WITNESS_STAGING_MARKERS:
        assert marker in reason, f"{key}: the refusal does not name {marker!r}: {reason}"


def test_the_witness_staging_advice_reaches_the_session_on_the_hook_lane(home):
    """The lane that ACTUALLY refused the two witness publishes still says the new half.

    Both journal rows for `backlog/data/` carry `where: "hook"` (re-read from
    `~/lloyd-data/safety/denials.jsonl` at this round's base: 15:05:04Z and 17:29:07Z,
    `guard: "safety"`, `session_class: "background"`), so the string that decided what
    those sessions did next was the PreToolUse hook's `permissionDecisionReason` — a
    second boundary between the guard and the model, and the one the first draft of this
    block's own docstring got wrong by crediting the dispatch lane with both refusals. A
    change to a refusal string earns a node on the lane that ships it, so this pins the
    deny decision AND the staging markers here, on the same redirect shape the 15:05:04Z
    row records, and a non-witness target as the contrast that the hook lane is not
    saying it everywhere.
    """
    witness = _cmd(WITNESS_DIR_WRITES["redirect-creates-witness-jsonl"], home)
    decision, reason = _hook_decision(witness, str(home / "lloyd"))
    assert decision == "deny", f"the hook let a witness redirect through: {witness}"
    assert "protected write" in reason, reason
    for marker in WITNESS_STAGING_MARKERS:
        assert marker in reason, f"the hook lane dropped {marker!r}: {reason}"

    board = _cmd(NON_WITNESS_VAULT_WRITES["board-file"], home)
    other_decision, other_reason = _hook_decision(board, str(home / "lloyd"))
    assert other_decision == "deny", f"the board redirect stopped being refused: {board}"
    for marker in WITNESS_STAGING_ONLY_MARKERS:
        assert marker not in other_reason, (
            f"the hook lane leaked staging wording to a board file: {other_reason}")


@pytest.mark.parametrize("key", sorted(NON_WITNESS_VAULT_WRITES))
def test_a_vault_redirect_outside_the_witness_directory_keeps_the_sentence_it_shipped_with(home,
                                                                                            key):
    """Clause 2: the staging wording is scoped to the witness directory and stops there.

    Asserted as byte equality against `PP.VAULT_BASH_WRITE_LABEL`, not as "mentions
    vaulting": `automod_vault_land` is already named by the #3701 sentence for every
    vault target, so the discriminator is the staging markers, and the strongest form of
    "keeps its current text" is that nothing else in the tree gained a character.
    """
    command = _cmd(NON_WITNESS_VAULT_WRITES[key], home)
    target = command.rsplit("> ", 1)[1].strip()
    reason = PP.check_bash_write_denied(command, str(home / "lloyd"))
    assert reason is not None, f"{key}: the vault redirect reached the lane unanswered"
    assert reason == f"a redirect into {target} — {PP.VAULT_BASH_WRITE_LABEL}", reason
    for marker in WITNESS_STAGING_ONLY_MARKERS:
        assert marker not in reason, f"{key}: staging wording leaked to {target}: {reason}"
    assert "use vault_write or automod_vault_land (#3701)" in reason, reason


def test_the_witness_refusal_fires_on_an_extract_that_does_not_exist_yet(home):
    """Clause 3: CREATING a witness file is refused on the predicate that already did it.

    The whole point of the journal rows is that the redirect was the thing that would
    have created the artifact, so an advice clause that only appeared once the file was
    there would describe a case nobody hits. Measured both ways in one node: the target
    and its directory absent, then the same target on disk holding bytes — and the two
    refusals are required to be the SAME string, which is what "no `exists()` test" looks
    like from outside. If anything downstream ever resolves the witness directory only
    when it is present, this node is where it fails.
    """
    target = home / "obsidian" / "backlog" / "data" / "2026-11-03.3737-new.jsonl"
    assert not target.parent.exists(), "the scratch vault must not pre-create the witness dir"
    assert not target.exists()
    reason_missing = PP.check_bash_write_denied(f"echo x > {target}", str(home / "lloyd"))
    assert reason_missing is not None, "a redirect CREATING a witness extract was allowed"
    assert "install -m 644" in reason_missing, reason_missing

    target.parent.mkdir(parents=True)
    target.write_text("committed witness bytes\n")
    reason_present = PP.check_bash_write_denied(f"echo x > {target}", str(home / "lloyd"))
    assert reason_present == reason_missing, (
        "the witness refusal now depends on the target being on disk — "
        f"absent: {reason_missing!r} present: {reason_present!r}")


def test_the_witness_advice_refuses_no_mover_because_the_staging_lane_stays_open(home):
    """Clause 4: telling a session to use `install -m 644` must not close `install -m 644`.

    The sentence names a mover as the sanctioned way in, so the round that added it had
    better not also have widened the refused shapes to cover it — that would make the
    advice a trap and take the owed #3701 ruling in the same commit as a string change.
    Measured here rather than trusted: `cp` and `install -m 644` into the very directory
    the new sentence describes both answer None, and the shape tuple is still the one
    entry `test_a_mover_into_the_vault_tree_is_not_yet_refused_because_that_ruling_is_owed`
    pins from the other side.
    """
    for mover in ("cp /tmp/x.jsonl {home}/obsidian/backlog/data/witness.jsonl",
                  "install -m 644 /tmp/x.jsonl {home}/obsidian/backlog/data/witness.jsonl",
                  "tee {home}/obsidian/backlog/data/witness.jsonl"):
        assert PP.check_bash_write_denied(_cmd(mover, home), str(home / "lloyd")) is None, (
            f"the staging lane named by the refusal was refused: {mover}")
    assert PP.VAULT_BASH_WRITE_SHAPES == ("a redirect into",), PP.VAULT_BASH_WRITE_SHAPES


def test_the_guard_names_the_witness_directory_by_literal_and_agrees_with_the_route(home):
    """The layering the item ruled, and the drift that ruling leaves room for.

    `app` imports `scripts` nowhere — `git grep -nE "^(from|import) scripts" -- 'app/**'`
    returned 0 hits at this round's base — so the guard spells the directory as a literal
    instead of importing `WITNESS_ARTIFACT_DIR` from `scripts/automod/backlog.py:1397`.
    That buys the layering and pays with a second copy of one string, so the copy is
    pinned against the constant it describes, read through the same route that names it.
    """
    assert PP.VAULT_BASH_WRITE_SUBDIR == "backlog/data", PP.VAULT_BASH_WRITE_SUBDIR
    from scripts.automod import backlog as B
    assert PP.VAULT_BASH_WRITE_SUBDIR == B.WITNESS_ARTIFACT_DIR
    target = home / "obsidian" / B.WITNESS_ARTIFACT_DIR / "probe.jsonl"
    reason = PP.check_bash_write_denied(f"echo x > {target}", str(home / "lloyd"))
    assert reason and "witness-artifact directory" in reason, reason
