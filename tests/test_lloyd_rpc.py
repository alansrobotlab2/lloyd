"""P9: `lloyd_rpc`, programmatic tool calling from inside Bash (v1, read-only).

What is pinned here, in the order `architecture/harness.md` (Review 2026-09-24,
P9) states it: the Bash child gets its env from the call's bound `_meta` and only
while `harness.rpc.enabled`; a nested call is refused when the parent's deny list
names it, when it is not read-only, and when it would recurse — above the effect
ledger's claim; the client refuses a call that would outlive the Bash deadline;
every nested call is logged under its parent call id and summed in a trailer; a
nested Read satisfies the read-before-edit gate for its session; a sandboxed
session gets no env and cannot read the credential.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
import os
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "agent-services" / "rpc" / "lloyd_rpc.py"

SID = "20260924_120000_rpctest"


def _client():
    spec = importlib.util.spec_from_file_location("lloyd_rpc_under_test", CLIENT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def rpc_on(monkeypatch, tmp_path):
    """harness.rpc.enabled on, and a private credential file in tmp."""
    from agent_mcp import _rpc, aggregator_auth
    from app.harness import rpc_policy

    monkeypatch.setattr(rpc_policy, "_block", lambda: {"enabled": True})
    token = tmp_path / "aggregator-token"
    token.write_text("rpc-test-credential-0123456789abcdef\n")
    token.chmod(0o600)
    monkeypatch.setenv(aggregator_auth.TOKEN_FILE_ENV, str(token))
    monkeypatch.delenv(aggregator_auth.TOKEN_ENV, raising=False)
    aggregator_auth.reset_for_tests()
    _rpc.reset_for_tests()
    yield token
    _rpc.reset_for_tests()
    aggregator_auth.reset_for_tests()


def _text(result) -> str:
    return "\n".join(getattr(b, "text", "") for b in result.content)


def _is_error(result) -> bool:
    from agent_mcp.main import _result_is_error
    return _result_is_error(result)


def _parent(deny=(), *, session=SID, call_id="call_parent_1", deadline_s=60.0):
    from agent_mcp import _rpc
    from app.harness import rpc_policy

    p = _rpc.Parent(parent_call_id=call_id, session_id=session, turn_id="turn-1",
                    effect_scope="", surface="chat",
                    deny=frozenset(rpc_policy.bash_deny(deny)),
                    deadline=time.time() + deadline_s)
    _rpc._parents[call_id] = p
    return p


def _rpc_meta(parent) -> dict:
    # What the client sends. The session here is deliberately wrong: the
    # server must dispatch as the parent's session, not the script's claim.
    return {"lloyd/session_id": "someone-else",
            "lloyd/rpc_parent_call_id": parent.parent_call_id,
            "lloyd/rpc_depth": 1}


# ── env ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_bash_sets_env_from_bound_meta(rpc_on):
    from agent_mcp import main as M
    from app.harness import rpc_policy

    meta = {"lloyd/session_id": SID, "lloyd/turn_id": "turn-7",
            "lloyd/call_id": "call_bash_7", "lloyd/effect_scope": "item:x:1",
            "lloyd/surface": "chat", "lloyd/rpc_deny": ["Grep", "Bash"]}
    res = await M.call_tool("Bash", {"command": "env | grep '^LLOYD_' | sort",
                                     "timeout": 30_000}, meta)
    out = _text(res)
    env = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    assert env[rpc_policy.ENV_SESSION_ID] == SID
    assert env[rpc_policy.ENV_TURN_ID] == "turn-7"
    assert env[rpc_policy.ENV_PARENT_CALL_ID] == "call_bash_7"
    assert env[rpc_policy.ENV_EFFECT_SCOPE] == "item:x:1"
    assert env[rpc_policy.ENV_SURFACE] == "chat"
    assert env[rpc_policy.ENV_DEPTH] == "1"
    assert env[rpc_policy.ENV_TOKEN_FILE] == str(rpc_on)
    assert env[rpc_policy.ENV_URL].endswith("/mcp")
    deny = json.loads(env[rpc_policy.ENV_DENY])
    assert {"Grep", "Bash", "Task", "ToolSearch", "automod_*"} <= set(deny)
    # The deadline is the Bash timeout less the margin: ~25 s out, not 30.
    left = float(env[rpc_policy.ENV_DEADLINE]) - time.time()
    assert 15 < left <= 30 - rpc_policy.DEADLINE_MARGIN_S + 1
    # No calls were made, so no trailer; and the parent is retired.
    assert "[lloyd_rpc:" not in out
    from agent_mcp import _rpc
    assert _rpc.parent_for("call_bash_7") is None


@pytest.mark.asyncio
async def test_an_unstamped_bash_call_gets_no_rpc_env(rpc_on):
    """The turn decides: a Bash call the loop did not stamp (rpc off for that
    turn) inherits the aggregator's environment exactly as before, even in an
    aggregator whose own config says on."""
    from agent_mcp import main as M

    res = await M.call_tool("Bash", {"command": "env | grep -c '^LLOYD_RPC_' || true"},
                            {"lloyd/session_id": SID, "lloyd/call_id": "c1"})
    assert _text(res).strip().splitlines()[0] == "0"


@pytest.fixture
def rpc_off(monkeypatch):
    """`harness.rpc.enabled` false — production's value, and the condition #3768's
    acceptance names: the session id has to reach the child with rpc still off, so
    the loop never stamps `lloyd/rpc_deny` and `bash_env` returns `(None, None)`."""
    from agent_mcp import _rpc
    from app.harness import rpc_policy

    monkeypatch.setattr(rpc_policy, "_block", lambda: {"enabled": False})
    _rpc.reset_for_tests()
    yield
    _rpc.reset_for_tests()


@pytest.mark.asyncio
async def test_an_unstamped_bash_child_still_gets_its_session_id(rpc_off, monkeypatch):
    """#3768 clauses 1 and 3, foreground branch: the session id travels without rpc.

    The reproduction check the item recorded at triage — an attended session's own
    Bash child running `env | grep -c LLOYD_SESSION_ID` and getting `0` — is this
    node: same call, same unstamped turn, same `harness.rpc.enabled` false. The fix
    is that it now reads `1` with the caller's own id.

    The two other assertions are clause 3's, and they are the ones that fail if the
    merge is written as a replacement: `_rpc.bash_env` returns a full
    `dict(os.environ)` copy when rpc is on and `None` when it is off, and
    `_resource_ceiling.launcher_env` / `_path_sandbox.child_env` both read a `None`
    as "inherit". Handing `create_subprocess_exec` a bare `{LLOYD_SESSION_ID: sid}`
    instead would strip `PATH` — the shell could not run `env` at all, so the
    sentinel and the `PATH` check are what pin the one-key merge rather than a swap.
    """
    from agent_mcp import aggregator_auth
    from agent_mcp import main as M
    from app.harness import rpc_policy

    monkeypatch.setenv("LLOYD_TEST_3768_SENTINEL", "inherited")
    res = await M.call_tool(
        "Bash",
        {"command": "env | sort", "timeout": 30_000},
        {"lloyd/session_id": SID, "lloyd/call_id": "call_unstamped_sid"})
    out = _text(res)
    env = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)

    assert env.get(rpc_policy.ENV_SESSION_ID) == SID, \
        f"the child did not get the caller's session id: {out[:400]!r}"
    assert env.get("LLOYD_TEST_3768_SENTINEL") == "inherited", \
        "the child's environment was replaced, not merged"
    # Equality with this process's PATH, not merely a non-empty PATH: a shell reports a
    # compiled-in default path out of an empty environment, so `env.get("PATH")` alone
    # would read green on a replaced environment — the mistake the review caught on the
    # first attempt. The same argument applies to any variable the shell sets itself.
    assert env.get("PATH") == os.environ.get("PATH"), (
        f"the child's PATH is not the inherited one, so something other than a one-key "
        f"merge reached the spawn: child={env.get('PATH')!r} "
        f"parent={os.environ.get('PATH')!r}")
    # Clause 3's negative half: one key, and only one. No rpc surface leaks in
    # while the feature is off, and no credential path rides along with the id.
    leaked = sorted(k for k in env if k.startswith("LLOYD_RPC_"))
    assert not leaked, f"LLOYD_RPC_* names reached an unstamped child: {leaked}"
    # The one variable whose loss would not be visible in the child's own output: the
    # resource ceiling is applied with `systemd-run`, and #2578 records that the call
    # fails open silently — no ceiling, no error — when `XDG_RUNTIME_DIR` is missing
    # from the environment it is handed. A merge that built a fresh dict from the
    # session id alone would drop it and still leave a working, unbounded shell.
    if "XDG_RUNTIME_DIR" in os.environ:
        assert env.get("XDG_RUNTIME_DIR") == os.environ["XDG_RUNTIME_DIR"], (
            "the merge dropped XDG_RUNTIME_DIR, so every Bash child on this box would "
            f"lose its cgroup ceiling: child={env.get('XDG_RUNTIME_DIR')!r}")
    token_path = str(aggregator_auth.token_path())
    assert token_path not in env.values(), \
        f"the credential file path reached an unstamped child's env: {token_path}"


@pytest.mark.asyncio
async def test_the_id_a_bash_child_carries_classifies_as_attended(rpc_off, tmp_path):
    """The acceptance check itself, read through the consumer rather than `env`.

    #3768's whole purpose is one gate in `scripts/memory/link_stranded_entities.py`:
    it admits `--apply` only when `session_class()` — which reads
    `rpc_policy.ENV_SESSION_ID` and hands the value to `classify_session` — answers
    `attended`. So the check is not that a name appears in the child's environment but
    that the value behind it reaches that classifier as the attended shape. This node
    has the child load the real script and print the real function's answer: the
    exporter and the importer, one process apart, with nothing between them but the
    environment. It prints `attended` only if the round's change is wired end to end;
    before the export existed the probe printed `no-session-id`, which is exactly what
    made every real apply refuse.
    """
    from agent_mcp import main as M

    probe = tmp_path / "linker_class_probe.py"
    probe.write_text(
        "import importlib.util, sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "spec = importlib.util.spec_from_file_location("
        f"'lse', {str(ROOT / 'scripts' / 'memory' / 'link_stranded_entities.py')!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "print('LINKER_CLASS=' + mod.session_class())\n",
        encoding="utf-8")
    res = await M.call_tool(
        "Bash",
        {"command": f"{sys.executable} {probe}", "timeout": 30_000},
        {"lloyd/session_id": SID, "lloyd/call_id": "call_class_probe"})
    out = _text(res)
    # A marker, not a bare line: importing `app.paths` from a gate worktree prints the
    # #3217 anchor banner on stdout ahead of anything the probe says, so the reading is
    # the marked token wherever it lands rather than "whatever line one is".
    assert "LINKER_CLASS=attended" in out, (
        f"a Bash child of session {SID!r} reported {out.strip()!r}, so the linker's "
        f"`--apply` gate is still unreachable from an attended session: {out[:400]!r}")


@pytest.mark.asyncio
async def test_a_background_bash_child_gets_its_session_id_too(rpc_off, tmp_path):
    """#3768 clause 1, the other spawn branch — a real detached child, not a spy.

    `builtin_bash._bash` asks `_rpc.bash_env` for its environment twice — the
    `background=True` call in the `run_in_background` branch and the
    `background=background` call before the foreground spawn — and the item's whole
    failure mode is a fix that reaches only one of them. So this node spawns an actual
    detached process and reads the variable back out of the file it writes, which is
    the only witness a detached child can offer: its tool result is returned before
    the command has run.
    """
    from agent_mcp import main as M
    from app.harness import rpc_policy

    target = tmp_path / "sid.txt"
    res = await M.call_tool(
        "Bash",
        {"command": f'printf "%s" "${{{rpc_policy.ENV_SESSION_ID}}}" > "{target}"',
         "run_in_background": True, "description": "#3768 session-id probe"},
        {"lloyd/session_id": SID, "lloyd/call_id": "call_bg_sid"})
    payload = json.loads(_text(res))
    assert "task_id" in payload, _text(res)

    for _ in range(200):
        if target.exists() and target.read_text():
            break
        await asyncio.sleep(0.05)
    # Read once, into a value, before asserting: a child that never wrote the file must
    # fail with the mismatch below, not with a FileNotFoundError raised out of the
    # assertion's own message.
    seen = target.read_text() if target.exists() else "<never written>"
    assert seen == SID, f"the background child saw {seen!r}, not {SID!r}"


def test_the_session_id_env_helper_names_one_key_for_one_caller():
    """The helper #3768 adds, read at the unit level for each of its three inputs.

    A non-empty id in, one key out; a sandboxed session's id, an empty id and no id
    at all each answer `{}`, which is what makes the merge a no-op rather than a
    blanking. Asserted through the helper and not only through the spawn so a
    regression names the function that broke.
    """
    from agent_mcp import _rpc, _task_registry
    from app.harness import rpc_policy

    tok = _task_registry.current_session_id.set(SID)
    try:
        assert _rpc.session_id_env(sandboxed=False) == {rpc_policy.ENV_SESSION_ID: SID}
        # The belt `bash_env` already carries: the id itself decides, so a lost
        # `sandboxed` flag cannot hand a bench trial an id.
        assert _rpc.session_id_env(sandboxed=True) == {}
    finally:
        _task_registry.current_session_id.reset(tok)

    tok = _task_registry.current_session_id.set("bench_rpc_trial")
    try:
        assert _rpc.session_id_env(sandboxed=False) == {}
        assert _rpc.session_id_env(sandboxed=True) == {}
    finally:
        _task_registry.current_session_id.reset(tok)

    tok = _task_registry.current_session_id.set("")
    try:
        assert _rpc.session_id_env(sandboxed=False) == {}, "an empty id must add nothing"
    finally:
        _task_registry.current_session_id.reset(tok)


def test_bwrap_really_clears_the_session_id_from_a_sandboxed_child():
    """Clause 2 across the boundary it is enforced by: a real `bwrap`, not argv inspection.

    The id is now exported to non-sandboxed children, so the aggregator process's own
    environment can carry it while a bench or eval child is being spawned — and the
    thing that protects that child is not the merge's `sandboxed` flag but bwrap's
    `--unsetenv`. So this spawns: the same `bwrap_argv` the sandbox branch wraps with, an
    environment that does have `LLOYD_SESSION_ID` in it, and the child's own `env` output
    as the reading. A child printing the name means a bench trial could classify itself
    as an attended session and cross a caller gate — which is exactly the direction #3768
    opens, so the check runs against a live spawn rather than against a list of flags.
    """
    from agent_mcp import _tool_sandbox
    from app.harness import rpc_policy

    # The guard is `sandbox_available`'s own probe, not a `bwrap_path()` check: a
    # binary that cannot build a read-only namespace (no user namespaces on this box)
    # fails the spawn for a reason that has nothing to do with the unset.
    buildable, why = _tool_sandbox.sandbox_available()
    if not buildable:
        pytest.skip(f"no usable bubblewrap here ({why}); the unset cannot be spawned")

    argv = _tool_sandbox.bwrap_argv("env", "/tmp")
    env = {**os.environ, rpc_policy.ENV_SESSION_ID: SID}
    proc = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr[-400:]
    leaked = [ln for ln in proc.stdout.splitlines()
              if ln.startswith(f"{rpc_policy.ENV_SESSION_ID}=")]
    assert not leaked, \
        f"bwrap left {rpc_policy.ENV_SESSION_ID} in a sandboxed child's environment: {leaked}"


def test_the_p9_env_paragraph_keeps_the_session_id_out_of_the_stamped_clause():
    """Clause 5: `architecture/harness.md` P9 states the id's condition as the code's.

    Two readings of the same paragraph, both structural rather than a quoted sentence,
    so an honest reword survives and a dishonest one does not. First the order: the
    session id's own sentence, with the condition the round gives it (`non-sandboxed`),
    must come before the `LLOYD_RPC_*` list and before the stamped-turn clause — so the
    paragraph no longer reads as the id arriving only on a stamped turn. Second the
    survival: the stamped-turn clause must still be there, after the rpc names, and the
    paragraph must still say a sandboxed session gets none of it. The condition it
    states is checked against the code that now implements it, in the same node: the
    helper keys on `sandboxed` and on nothing else, so prose and spawn cannot drift apart
    with one of them passing silently.
    """
    from app.harness import rpc_policy

    text = (ROOT / "architecture" / "harness.md").read_text(encoding="utf-8")
    start = text.index("- **The env** (`builtin_bash._bash`")
    bullet = text[start:text.index("\n- **Server-authoritative.", start)]
    sid_at = bullet.index(rpc_policy.ENV_SESSION_ID)
    rpc_at = bullet.index("LLOYD_RPC_")
    stamped_at = bullet.index("only when the turn was stamped")

    assert sid_at < rpc_at < stamped_at, (
        "the P9 env paragraph conditions the session id on the stamped turn again, or "
        "dropped that condition from the rpc names it still belongs to")
    assert "non-sandboxed" in bullet[sid_at:rpc_at], (
        "the session id's sentence no longer states the condition that does apply to it "
        "(#3768: every non-sandboxed child, stamped or not)")
    assert "gets none of" in bullet, "the sandboxed session's exclusion went missing"

    src = (ROOT / "agent_mcp" / "_rpc.py").read_text(encoding="utf-8")
    helper = src[src.index("def session_id_env"):src.index("def bash_env")]
    assert "sandboxed" in helper and "current_rpc_deny" not in helper, (
        "the helper's condition no longer matches the paragraph: it must key on the "
        "sandbox alone, not on the turn being stamped")
    assert "ENV_SESSION_ID" in helper, "the helper no longer exports the documented name"


def test_a_sandboxed_bash_child_gets_no_session_id(rpc_off):
    """#3768 clause 2: the sandbox's guarantee survives the export, two ways.

    First the merge itself: for a sandboxed call `_merge_session_id` returns the
    env it was handed untouched — `None`, which the two spawn paths read as
    "inherit" — so the fix adds nothing to a bench or eval child. Then the pre-existing
    guarantee that this round must not break: `ENV_SESSION_ID` is a member of
    `rpc_policy.ENV_NAMES`, and `_tool_sandbox.bwrap_argv` answers `--unsetenv` for
    every name in it, so a bench child cannot inherit the id from the aggregator
    either. Pulling the name out of `ENV_NAMES` would silently drop that unset,
    which is why it is pinned by name here and not only by the loop above.
    """
    from agent_mcp import _rpc, _task_registry, _tool_sandbox, builtin_bash
    from app.harness import rpc_policy

    assert rpc_policy.ENV_SESSION_ID in rpc_policy.ENV_NAMES, \
        "the sandbox unsets only the names in ENV_NAMES; moving this one out of " \
        "that tuple silently lets a bench child inherit the session id (#3768)"

    tok = _task_registry.current_session_id.set("bench_rpc_trial")
    try:
        assert builtin_bash._merge_session_id(None, sandboxed=True) is None
        assert builtin_bash._merge_session_id(None, sandboxed=False) is None, \
            "a lost sandbox flag bought a bench child a session id"
        assert _rpc.session_id_env(sandboxed=True) == {}
    finally:
        _task_registry.current_session_id.reset(tok)

    # An attended id in the aggregator's own environment is no help to a sandboxed
    # call either: bwrap clears the name before the command runs.
    argv = _tool_sandbox.bwrap_argv("true", "/")
    pairs = list(zip(argv, argv[1:]))
    assert ("--unsetenv", rpc_policy.ENV_SESSION_ID) in pairs, \
        "bwrap stopped unsetting the session id"


def test_sandboxed_session_gets_no_env(rpc_on):
    """No env for a bench/eval session, and inside bwrap the credential is
    neither in the environment nor readable on disk."""
    from agent_mcp import _rpc, _task_registry, _tool_sandbox, aggregator_auth
    from app.harness import rpc_policy

    tok = _task_registry.current_session_id.set("bench_rpc_trial")
    dtok = _rpc.current_rpc_deny.set(("Bash",))
    try:
        assert _rpc.bash_env(sandboxed=True, timeout_s=60, background=False) == (None, None)
        # Even if the flag were lost, the id itself is sandboxed.
        assert _rpc.bash_env(sandboxed=False, timeout_s=60, background=False) == (None, None)
    finally:
        _rpc.current_rpc_deny.reset(dtok)
        _task_registry.current_session_id.reset(tok)
    assert _rpc._parents == {}

    argv = _tool_sandbox.bwrap_argv("true", "/")
    pairs = list(zip(argv, argv[1:]))
    for name in (*rpc_policy.ENV_NAMES, aggregator_auth.TOKEN_ENV,
                 aggregator_auth.TOKEN_FILE_ENV):
        assert ("--unsetenv", name) in pairs, name
    i = argv.index(str(rpc_on))
    assert argv[i - 2:i] == ["--ro-bind", "/dev/null"]


def test_a_credential_file_others_can_read_gets_no_env(rpc_on):
    from agent_mcp import _rpc, _task_registry

    tok = _task_registry.current_session_id.set(SID)
    dtok = _rpc.current_rpc_deny.set(("Bash",))
    try:
        env, parent = _rpc.bash_env(sandboxed=False, timeout_s=60, background=False)
        assert env is not None and parent is not None
        _rpc.finish(parent)
        rpc_on.chmod(0o644)
        assert _rpc.bash_env(sandboxed=False, timeout_s=60, background=False) == (None, None)
    finally:
        _rpc.current_rpc_deny.reset(dtok)
        _task_registry.current_session_id.reset(tok)


# ── admission, in main.call_tool ───────────────────────────────────────────

@pytest.fixture
def no_claims(monkeypatch):
    """Record every effect-ledger claim, so a refusal can be shown to sit above it."""
    from agent_mcp import _tool_effects

    seen: list[str] = []
    real = _tool_effects.claim

    async def spy(name, *a, **kw):
        seen.append(name)
        return await real(name, *a, **kw)

    monkeypatch.setattr(_tool_effects, "claim", spy)
    return seen


@pytest.mark.asyncio
async def test_refuses_parent_denied_tool(rpc_on, no_claims, tmp_path):
    from agent_mcp import main as M

    p = _parent(deny=["Grep"])
    res = await M.call_tool("Grep", {"pattern": "x", "path": str(tmp_path)}, _rpc_meta(p))
    assert _is_error(res)
    assert "Tool call denied: lloyd_rpc" in _text(res)
    assert "Grep" not in no_claims


@pytest.mark.asyncio
async def test_a_script_cannot_shorten_the_deny_list(rpc_on, tmp_path):
    """The client's own copy of the deny list is not what the server reads."""
    from agent_mcp import main as M

    p = _parent(deny=["Grep"])
    meta = dict(_rpc_meta(p), **{"lloyd/rpc_deny": []})
    res = await M.call_tool("Grep", {"pattern": "x", "path": str(tmp_path)}, meta)
    assert _is_error(res) and "lloyd_rpc" in _text(res)


@pytest.mark.asyncio
async def test_refuses_mutating_tool_in_v1(rpc_on, no_claims, tmp_path, monkeypatch):
    from agent_mcp import main as M
    from app.harness import rpc_policy

    # Even with the key flipped: v1 ignores allow_mutating.
    monkeypatch.setattr(rpc_policy, "_block",
                        lambda: {"enabled": True, "allow_mutating": True})
    target = tmp_path / "never.txt"
    p = _parent()
    for tool, args in (("Write", {"file_path": str(target), "content": "x"}),
                       ("vault_write", {"path": "knowledge/x.md", "content": "x"})):
        res = await M.call_tool(tool, args, _rpc_meta(p))
        assert _is_error(res), tool
        assert "read-only tools only" in _text(res), _text(res)
    assert not target.exists()
    assert no_claims == []


@pytest.mark.asyncio
async def test_refuses_bash_task_recursion(rpc_on, no_claims):
    from agent_mcp import main as M

    p = _parent(deny=[])
    for tool, args in (("Bash", {"command": "true"}),
                       ("Task", {"prompt": "x"}),
                       ("automod_status", {}),
                       ("desktop_capture", {})):
        res = await M.call_tool(tool, args, _rpc_meta(p))
        assert _is_error(res), tool
        assert "not available to lloyd_rpc" in _text(res), (tool, _text(res))
    # Depth > 1 is refused on its own too.
    res = await M.call_tool("Read", {"file_path": "/etc/hostname"},
                            dict(_rpc_meta(p), **{"lloyd/rpc_depth": 2}))
    assert _is_error(res) and "depth" in _text(res)
    assert no_claims == []


@pytest.mark.asyncio
async def test_an_unknown_or_expired_parent_is_refused(rpc_on):
    from agent_mcp import main as M

    res = await M.call_tool("Read", {"file_path": "/etc/hostname"},
                            {"lloyd/rpc_parent_call_id": "never-registered"})
    assert _is_error(res) and "unknown or finished parent" in _text(res)
    p = _parent(call_id="expired", deadline_s=-1.0)
    res = await M.call_tool("Read", {"file_path": "/etc/hostname"}, _rpc_meta(p))
    assert _is_error(res) and "deadline" in _text(res)


# ── recording ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_calls_logged_with_parent_id(rpc_on, monkeypatch, tmp_path):
    from agent_mcp import _rpc
    from agent_mcp import main as M
    from app.harness import telemetry

    logged: list[tuple] = []
    monkeypatch.setattr(telemetry, "log_harness_event",
                        lambda sid, ev, data, *, turn_id=None: logged.append(
                            (sid, ev, data, turn_id)))
    f = tmp_path / "a.txt"
    f.write_text("hello rpc\n")
    p = _parent(deny=["Grep"])
    ok = await M.call_tool("Read", {"file_path": str(f)}, _rpc_meta(p))
    assert not _is_error(ok) and "hello rpc" in _text(ok)
    bad = await M.call_tool("Grep", {"pattern": "x"}, _rpc_meta(p))
    assert _is_error(bad)

    rows = [r for r in logged if r[1] == "harness.rpc_call"]
    assert [r[2]["tool"] for r in rows] == ["Read", "Grep"]
    for sid, _ev, data, turn in rows:
        assert sid == SID and turn == "turn-1"
        assert data["parent_call_id"] == p.parent_call_id
        assert len(data["args_digest"]) == 12
    assert rows[0][2]["is_error"] is False
    assert rows[1][2]["is_error"] is True and "refused" in rows[1][2]

    trailer = _rpc.finish(p)
    assert trailer.startswith("[lloyd_rpc: 2 calls — ")
    assert "Grep×1" in trailer and "Read×1" in trailer and "1 error," in trailer
    assert _rpc.parent_for(p.parent_call_id) is None


def test_trailer_shape():
    from app.harness.rpc_policy import trailer

    assert trailer({}, 0, 1.0) == ""
    assert trailer({"Read": 10, "Grep": 2}, 0, 3.14) == \
        "[lloyd_rpc: 12 calls — Read×10 Grep×2, 0 errors, 3.1 s]"


@pytest.mark.asyncio
async def test_an_rpc_read_then_an_edit_passes_gate_check(rpc_on, tmp_path):
    """The nested Read is recorded under the PARENT's session, so the parent's
    own Edit afterwards is not refused as unread — and a different session's
    Edit still is."""
    from agent_mcp import main as M

    f = tmp_path / "gate.py"
    f.write_text("x = 1\n")
    p = _parent()
    res = await M.call_tool("Read", {"file_path": str(f)}, _rpc_meta(p))
    assert not _is_error(res), _text(res)
    edit = {"file_path": str(f), "old_string": "x = 1", "new_string": "x = 2"}
    other = await M.call_tool("Edit", edit, {"lloyd/session_id": "someone-else"})
    assert _is_error(other), _text(other)
    mine = await M.call_tool("Edit", edit, {"lloyd/session_id": SID})
    assert not _is_error(mine), _text(mine)
    assert f.read_text() == "x = 2\n"


# ── the client ─────────────────────────────────────────────────────────────

def test_call_outliving_the_deadline_is_refused_client_side(tmp_path):
    rpc = _client()
    token = tmp_path / "t"
    token.write_text("x" * 32)
    token.chmod(0o600)
    env = {"LLOYD_RPC_URL": "http://127.0.0.1:9/mcp",  # discard port: never reached
           "LLOYD_PARENT_CALL_ID": "p", "LLOYD_SESSION_ID": SID,
           "LLOYD_RPC_TOKEN_FILE": str(token), "LLOYD_RPC_DEPTH": "1",
           "LLOYD_RPC_DENY": "[]", "LLOYD_RPC_DEADLINE": f"{time.time() + 0.3:.3f}"}
    with pytest.raises(rpc.RpcError, match="deadline"):
        rpc.call("Read", file_path="/etc/hostname", env=env)
    # Denied client-side too, before the network.
    env["LLOYD_RPC_DEADLINE"] = f"{time.time() + 60:.3f}"
    env["LLOYD_RPC_DENY"] = json.dumps(["Bash", "automod_*"])
    with pytest.raises(rpc.ToolError, match="denied"):
        rpc.call("automod_status", env=env)
    # And a credential file others can read is refused.
    token.chmod(0o644)
    with pytest.raises(rpc.RpcError, match="0600"):
        rpc.call("Read", file_path="/etc/hostname", env=env)


def test_the_client_outside_a_bash_call_says_so():
    rpc = _client()
    with pytest.raises(rpc.RpcError, match="not inside a Lloyd Bash call"):
        rpc.call("Read", env={})


# ── the loop and the prompt ────────────────────────────────────────────────

@pytest.mark.parametrize("on", [True, False])
def test_the_loop_stamps_rpc_deny_on_bash_calls_only(monkeypatch, on):
    from app.harness import rpc_policy
    from app.harness.options import RunOptions
    from app.harness.tests import _replay as R

    monkeypatch.setattr(rpc_policy, "_block", lambda: {"enabled": on})
    engine = R.ReplayEngine([
        R.Step(tool_calls=[R.tool_call("c1", "Bash", "run", command="true"),
                           R.tool_call("c2", "Read", "read", file_path="/x")]),
        R.Step(text="done"),
    ])
    pool = R.ReplayPool()
    R.install(monkeypatch, engine, pool)
    asyncio.run(R.drive(RunOptions(model="m", max_turns=4, tool_search_enabled=False,
                                   session_id=SID, disallowed_tools=["Glob"])))
    by_name = {c["name"]: c for c in pool.calls}
    assert "rpc_deny" not in by_name["Read"]
    if on:
        deny = by_name["Bash"]["rpc_deny"]
        assert "Glob" in deny and "Task" in deny and "automod_*" in deny
    else:
        assert "rpc_deny" not in by_name["Bash"]


def test_the_prompt_is_bytewise_today_when_off(monkeypatch):
    from app import prompt_builder as PB
    from app.harness import rpc_policy

    monkeypatch.setattr(rpc_policy, "_block", lambda: {})
    off = PB.build_system_prompt(include_skills_index=False, memories_text="")
    assert "lloyd_rpc" not in off
    monkeypatch.setattr(rpc_policy, "_block", lambda: {"enabled": True})
    on = PB.build_system_prompt(include_skills_index=False, memories_text="")
    assert "Programmatic tool calls:" in on and "agent-services/bin/lloyd_rpc" in on
    # The paragraph is inserted; nothing else moves.
    assert on.replace("\n\n" + PB._rpc_hint(), "", 1) == off


def test_eval_tasks_file_parses():
    import yaml

    data = yaml.safe_load((ROOT / "eval" / "rpc_tasks.yaml").read_text())
    tasks = data["tasks"]
    assert len(tasks) == 3
    for t in tasks:
        assert t["id"] and t["prompt"] and t["objective_checks"]


def test_eval_objective_and_decision():
    sys.path.insert(0, str(ROOT))
    from eval import run_rpc_eval as E

    task = {"objective_checks": {"min_recall": 0.9}}
    assert E.objective(task, "#12 and #13", {"#12", "#13"})["pass"]
    assert not E.objective(task, "#12 only; #130", {"#12", "#13"})["pass"]
    assert E.rpc_calls_in(["out\n[lloyd_rpc: 12 calls — Read×10 Grep×2, 1 error, 3.1 s]"]) \
        == {"rpc_calls": 12, "rpc_errors": 1}

    def row(task_id, arm, tokens, ok=True, timeouts=0):
        return {"task_id": task_id, "arm": arm, "prompt_tokens_sum": tokens,
                "num_turns": 3, "wall_s": 1.0, "bash_timeouts": timeouts,
                "objective": {"pass": ok}}
    rows = []
    for t, on_tokens in (("a", 50), ("b", 60), ("c", 100)):
        rows += [row(t, "off", 100), row(t, "off_rep", 100), row(t, "on", on_tokens)]
    verdict = E.decide(rows)
    assert verdict["token_wins"] == 2 and verdict["promote"]
    assert not E.decide(rows + [row("a", "on", 50, timeouts=1)])["promote"]


def test_the_eval_refuses_without_live_bash(monkeypatch, capsys):
    sys.path.insert(0, str(ROOT))
    from eval import run_rpc_eval as E

    monkeypatch.setattr(sys, "argv", ["run_rpc_eval.py", "--out", "/nonexistent"])
    assert E.main() == 2
    assert "live, unsandboxed Bash" in capsys.readouterr().err


# ── end to end: a real shell, the real client, the real aggregator app ─────

@pytest.mark.asyncio
async def test_a_bash_script_reads_through_lloyd_rpc_end_to_end(rpc_on, monkeypatch, tmp_path):
    """The whole path on a loopback socket: Bash spawns with the env, the shell
    runs the CLI and the Python module, the aggregator admits the reads under
    the parent's session and refuses a write, and the Bash result carries the
    trailer."""
    import socket

    import httpx
    import uvicorn

    from agent_mcp import _rpc
    from agent_mcp import main as M

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setattr(_rpc, "_port", port)
    server = uvicorn.Server(uvicorn.Config(M.starlette_app, host="127.0.0.1",
                                           port=port, log_level="error"))
    # The same singleton may already have run under another file's fixture.
    if getattr(M.combined, "_session_manager", None) is not None:
        M.combined._session_manager._has_started = False
    task = asyncio.create_task(server.serve())
    try:
        async with httpx.AsyncClient() as probe:
            for _ in range(400):
                await asyncio.sleep(0.05)
                try:
                    if (await probe.get(f"http://127.0.0.1:{port}/health",
                                        timeout=1.0)).status_code in (200, 503):
                        break
                except Exception:
                    continue
            else:
                pytest.fail("aggregator did not become ready")

        for i in range(3):
            (tmp_path / f"f{i}.txt").write_text(f"line-{i}\n")
        cli = ROOT / "agent-services" / "bin" / "lloyd_rpc"
        lib = ROOT / "agent-services" / "rpc"
        script = (
            f"{cli} call Read '{{\"file_path\": \"{tmp_path}/f0.txt\"}}' && "
            f"python3 -c 'import sys; sys.path.insert(0, \"{lib}\"); import lloyd_rpc; "
            f"out = lloyd_rpc.map(\"Read\", [{{\"file_path\": \"{tmp_path}/f%d.txt\" % i}} "
            f"for i in (1, 2)]); print(\"MAP\", [\"line-\" in o for o in out])' && "
            f"{{ {cli} call Write '{{\"file_path\": \"{tmp_path}/w.txt\", \"content\": \"x\"}}'; "
            f"echo WRITE_RC=$?; }}"
        )
        res = await M.call_tool("Bash", {"command": script, "timeout": 60_000},
                                {"lloyd/session_id": SID, "lloyd/call_id": "call_e2e",
                                 "lloyd/turn_id": "turn-e2e",
                                 "lloyd/rpc_deny": ["Grep"]})
        out = _text(res)
        assert "line-0" in out, out
        assert "MAP [True, True]" in out, out
        assert "WRITE_RC=2" in out and "read-only tools only" in out, out
        assert not (tmp_path / "w.txt").exists()
        assert out.rstrip().splitlines()[-1].startswith(
            "[lloyd_rpc: 4 calls — Read×3 Write×1, 1 error, "), out
        assert _rpc.parent_for("call_e2e") is None
    finally:
        server.should_exit = True
        await task
        # `agent_mcp.main.starlette_app` is a module singleton whose Streamable
        # HTTP session manager refuses a second `.run()`; tests/test_mcp_transport.py
        # boots the same app later in the same process, so hand the manager back
        # in its never-started state (its task group closed with the lifespan).
        mgr = getattr(M.combined, "_session_manager", None)
        if mgr is not None:
            mgr._has_started = False


def test_the_copies_of_the_wire_names_agree():
    """The client is stdlib-only and the pool may not import the policy's
    consumers, so the names are restated; this is what keeps them one set."""
    from agent_mcp import aggregator_auth
    from app.harness import mcp_pool, rpc_policy

    rpc = _client()
    assert mcp_pool.META_RPC_DENY == rpc_policy.META_RPC_DENY
    client_env = {getattr(rpc, n) for n in dir(rpc) if n.startswith("ENV_")}
    assert client_env == set(rpc_policy.ENV_NAMES)
    assert rpc.TOKEN_ENV == aggregator_auth.TOKEN_ENV
    assert rpc.AUTH_HEADER == aggregator_auth.AUTH_HEADER
