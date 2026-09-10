"""The authority gate has to be on the endpoint, not on the caller.

#534 gates tier-2 and tier-3 tools on a live grant. It is a PreToolUse hook,
so it exists only if whoever built the turn's `HookRegistry` installed it.
Two callers did: `autonomy.run_task` and `workers.sources._common.
_worker_run_options`. `app/routers/messages.py` did not — and every
session-backed worker posts there.

So the four turn paths that read untrusted text and rewrite this repo —
autocode, autotriage, deep-research, youtube-digest — were the ungated ones,
while the two paths that happened to build their own registry were covered
twice. Nothing re-checks at the tool layer (`agent_mcp/main.py`), so the hook
being absent is the gate being absent.

The fix triggers on the SESSION'S OWN PLATFORM, which is the property these
tests pin: a caller cannot forget, because a caller is not asked.
"""

from __future__ import annotations

import json

import pytest

from app.harness import HookRegistry
from app.harness.policy import GRANT_MINT_TOOL, GrantStore, install_policy_hook
from app.routers import messages as M


@pytest.fixture
def sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "SESSIONS_DIR", tmp_path)
    return tmp_path


def _session(sessions, sid, **fields):
    (sessions / f"{sid}.json").write_text(json.dumps(
        {"session_id": sid, "messages": [], **fields}))
    return sid


# ── Which turns are gated ──────────────────────────────────────────────

def test_a_worker_session_is_gated_and_names_its_source(sessions):
    _session(sessions, "s1", platform="worker", source="autotriage")
    assert M._authority_scope_for("s1", {}) == "worker:autotriage"


def test_an_autonomy_session_is_gated(sessions):
    _session(sessions, "s2", platform="autonomy", source="autonomy-task:39")
    assert M._authority_scope_for("s2", {}) == "worker:autonomy-task:39"


def test_a_payload_scope_wins_over_the_derived_one(sessions):
    """The pool knows what it claimed; the session file only knows its source.
    `autonomy-task:39` and `worker:scheduled-task` are the difference between
    a grant made to one nightly job and a grant made to whatever runs on that
    source next."""
    _session(sessions, "s3", platform="worker", source="scheduled-task")
    assert M._authority_scope_for(
        "s3", {"grant_scope": "autonomy-task:39"}) == "autonomy-task:39"


def test_an_attended_chat_turn_is_not_gated(sessions):
    _session(sessions, "s4", platform="mission-control")
    assert M._authority_scope_for("s4", {}) == ""
    # And a session file that does not exist yet is a chat session being
    # created by this very request — gating it would take the UI down.
    assert M._authority_scope_for("never-written", {}) == ""


def test_a_caller_that_names_a_scope_is_gated_whatever_the_platform(sessions):
    """A turn that says whose authority it is borrowing has said what it is."""
    _session(sessions, "s5", platform="mission-control")
    assert M._authority_scope_for("s5", {"grant_scope": "worker:x"}) == "worker:x"


# ── The mint ban ───────────────────────────────────────────────────────

def test_the_ban_is_written_back_into_the_request_body():
    """`_refresh_disallowed_for_session` re-reads `data["extra_disallowed"]`
    on every harness iteration, so a ban that only reached the initial list
    would come off at the first refresh."""
    data: dict = {}
    M._ban_grant_minting(data)
    assert GRANT_MINT_TOOL in data["extra_disallowed"]
    assert f"mcp__lloyd-mcp__{GRANT_MINT_TOOL}" in data["extra_disallowed"]

    # Idempotent, and it keeps what the caller already asked for.
    data = {"extra_disallowed": ["automod_land"]}
    M._ban_grant_minting(data)
    M._ban_grant_minting(data)
    assert data["extra_disallowed"].count(GRANT_MINT_TOOL) == 1
    assert "automod_land" in data["extra_disallowed"]


# ── What the gate actually denies ──────────────────────────────────────

async def _decide(hooks: HookRegistry, tool: str, args: dict | None = None):
    return await hooks.fire_pre_tool_use(
        session_id="s1", tool_name=tool, tool_input=args or {})


def _gated(tmp_path, scope="worker:autotriage"):
    hooks = HookRegistry()
    store = GrantStore(tmp_path / "grants.sqlite")
    store.ensure_schema()
    install_policy_hook(hooks, store=store, scope=scope)
    return hooks, store


@pytest.mark.asyncio
async def test_an_ungranted_tier_two_call_is_denied(tmp_path):
    hooks, _ = _gated(tmp_path)
    out = await _decide(hooks, "email_send", {"to": "a@b.c"})
    decision = out["hookSpecificOutput"]["permissionDecision"]
    assert decision == "deny"


@pytest.mark.asyncio
async def test_a_live_grant_allows_the_same_call(tmp_path):
    hooks, store = _gated(tmp_path)
    import datetime as _dt
    store.mint(scope="worker:autotriage", tool_pattern="email_send",
               quota=1, issued_by="alan",
               expires_at=_dt.datetime.now(_dt.timezone.utc)
                          + _dt.timedelta(minutes=10),
               note="pinned by a test")
    out = await _decide(hooks, "email_send", {"to": "a@b.c"})
    assert out == {}


@pytest.mark.asyncio
async def test_the_gate_denies_nothing_a_worker_does_today(tmp_path):
    """Tier 1 is everything unclassified, and that is deliberate: a gate that
    guesses wrong denies real work and a worker that cannot write its output
    burns a run. So arming this on every session-backed worker turn changes
    nothing about what they already do — it starts gating the
    durable-external surface only."""
    hooks, _ = _gated(tmp_path)
    for tool in ("Bash", "Edit", "Write", "Read", "backlog_write_task",
                 "vault_write", "automod_gate"):
        assert await _decide(hooks, tool, {"command": "ls"}) == {}, tool


@pytest.mark.asyncio
async def test_minting_a_grant_from_inside_a_gated_turn_is_denied(tmp_path):
    """Enforced twice on purpose: the tool is not advertised, and the hook
    denies the name if a local model emits it anyway. A turn subject to a
    gate must not be able to write its way out of it."""
    hooks, _ = _gated(tmp_path)
    out = await _decide(hooks, GRANT_MINT_TOOL, {"scope": "worker:autotriage"})
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.asyncio
async def test_an_unreadable_store_denies_rather_than_opens_the_gate(tmp_path):
    """`fire_pre_tool_use` treats a raising callback as a pass — correct for
    an observer, exactly wrong here. The hook denies instead of raising, so a
    store that cannot be opened closes the gate."""
    hooks = HookRegistry()
    broken = tmp_path / "nope" / "deeper" / "grants.sqlite"
    install_policy_hook(hooks, store=GrantStore(broken), scope="worker:x")
    out = await _decide(hooks, "email_delete", {"id": "1"})
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


# ── The caller carries its scope ───────────────────────────────────────

def test_the_worker_turn_payload_carries_the_pool_bound_scope():
    """`policy.current_scope` is a contextvar bound around the claimed job. It
    is correct in the pool's own task and does not survive the loopback POST,
    so it has to travel in the body."""
    src = (M.__file__.rsplit("/app/", 1)[0] + "/workers/sources/_common.py")
    text = open(src).read()
    assert '"grant_scope": current_scope.get()' in text


def test_every_registry_the_router_builds_arms_the_gate():
    """Three endpoints build a turn's hooks. "The other endpoint is the
    ungated one" is the exact shape of the bug this closes, so the count is
    pinned rather than trusted."""
    text = open(M.__file__).read()
    assert text.count("install_default_safety_hook(iv_hooks)") == 3
    assert text.count("install_policy_hook(iv_hooks, scope=") == 3
