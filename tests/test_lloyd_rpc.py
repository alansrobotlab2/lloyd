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
import sys
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
    import prompt_builder as PB
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
