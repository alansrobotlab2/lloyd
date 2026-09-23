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

import datetime as dt
import json

import pytest

from app.harness import HookRegistry, policy
from app.harness.policy import GRANT_MINT_TOOL, GrantStore, install_policy_hook
from app.routers import messages as M

#: A fixed instant, for the cases that inject a clock (`install_policy_hook(…,
#: now=FROZEN)`). Two cases are about the wall clock and are named in their own
#: docstrings: one mints from `dt.datetime.now` (an expired grant must still
#: deny with no clock injected), one patches `policy._now` (the default must be
#: read afresh on every call). Everything else derives from this constant, so it
#: cannot rot into a permanent red the way the hook-path fixtures in
#: `tests/unit/test_grant_policy.py` did on 2026-09-11 (#848/#853; #973 is the
#: seam that made that shape unwriteable).
FROZEN = dt.datetime(2026, 9, 10, 12, 0, tzinfo=dt.timezone.utc)


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


def _gated(tmp_path, scope="worker:autotriage", **hook_kwargs):
    """Build the turn's hooks the way the router does.

    The clock is spelled as `**hook_kwargs` and not as a `now=None` parameter on
    this helper because a helper that always forwarded `now=now` would re-supply
    the keyword on the real-clock calls too — and a default value captured when
    `policy.py` was imported, the wrong shape of this seam, would then be
    indistinguishable from reading the wall clock on every call. Calling
    `install_policy_hook(hooks, store=…, scope=…)` with nothing else is the only
    way to exercise what `app/routers/messages.py` actually gets.
    """
    hooks = HookRegistry()
    store = GrantStore(tmp_path / "grants.sqlite")
    store.ensure_schema()
    install_policy_hook(hooks, store=store, scope=scope, **hook_kwargs)
    return hooks, store


@pytest.mark.asyncio
async def test_an_ungranted_tier_two_call_is_denied(tmp_path):
    hooks, _ = _gated(tmp_path)
    out = await _decide(hooks, "email_send", {"to": "a@b.c"})
    decision = out["hookSpecificOutput"]["permissionDecision"]
    assert decision == "deny"


@pytest.mark.asyncio
async def test_a_live_grant_allows_the_same_call(tmp_path):
    """The positive control, now hermetic: the instant the gate reads is the
    instant the test names, so "live" means live against `FROZEN` and this case
    says the same thing on every date it is run (#973 is the seam; the case
    before it was `expires_at=now()+10min`, which is green until it isn't)."""
    hooks, store = _gated(tmp_path, now=FROZEN)
    store.mint(scope="worker:autotriage", tool_pattern="email_send",
               quota=1, issued_by="alan",
               expires_at=FROZEN + dt.timedelta(minutes=10),
               note="pinned by a test")
    out = await _decide(hooks, "email_send", {"to": "a@b.c"})
    assert out == {}


@pytest.mark.asyncio
async def test_an_expired_grant_denies_when_no_clock_is_injected(tmp_path):
    """`now` is optional and production never passes it, so the default has to
    be the wall clock: a grant that ran out a minute ago denies a tier-2 call.

    Minted from `dt.datetime.now` on purpose, and that is not the habit this
    file otherwise keeps — the frozen `FROZEN` fixtures are behind the wall
    clock, so deriving this one from them would deny for the wrong reason and
    prove nothing about the default. What is under test is precisely that
    omitting the argument still means *today*."""
    hooks, store = _gated(tmp_path)          # no `now=` — production shape
    store.mint(scope="worker:autotriage", tool_pattern="email_send",
               issued_by="alan",
               expires_at=dt.datetime.now(dt.timezone.utc)
                          - dt.timedelta(minutes=1),
               note="ran out a minute ago")
    out = await _decide(hooks, "email_send", {"to": "a@b.c"})
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.asyncio
async def test_the_real_clock_is_asked_again_on_every_call(tmp_path, monkeypatch):
    """Omitting `now` must mean "the wall clock, read at this call" — not a value
    captured when `policy.py` was imported. The hook is installed once and fires
    for the whole turn, so a clock read at install time is a clock that never
    notices an expiry, and a long unattended turn keeps spending a grant that
    died mid-run.

    `now: dt.datetime = _now()` as the default satisfies every other case in
    this file and in `tests/unit/test_grant_policy.py` (those inject a clock, so
    the default never shows) and is caught only here. `_now` is patched rather
    than the calendar waited on: the case is one call after the expiry, and a
    test may not sleep through it."""
    fake = {"at": FROZEN}
    monkeypatch.setattr(policy, "_now", lambda: fake["at"])
    hooks, store = _gated(tmp_path)          # no `now=` — production shape
    store.mint(scope="worker:autotriage", tool_pattern="email_send",
               issued_by="alan",
               expires_at=fake["at"] + dt.timedelta(minutes=10),
               note="live at the patched instant")
    assert await _decide(hooks, "email_send", {"to": "a@b.c"}) == {}
    fake["at"] += dt.timedelta(minutes=20)   # past the expiry, still no `now=`
    out = await _decide(hooks, "email_send", {"to": "a@b.c"})
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


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


def test_the_same_registry_set_also_arms_the_outbound_content_gate():
    """The pin-count convention, applied one guard over (#1136).

    The test above pins three router sites for the Bash floor and the grant
    gate. A fourth guard that only ever got installed on some of those paths is
    the same bug wearing a new name, so the content gate is pinned here in the
    same idiom — and, unlike a per-file count, through a finder that walks every
    dispatch site in the tree, because the hole this closes is a path nobody
    thought to count.
    """
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    text = (repo / "app" / "routers" / "messages.py").read_text(encoding="utf-8")
    assert text.count("install_default_safety_hook(iv_hooks)") == 3
    assert text.count("install_policy_hook(iv_hooks, scope=") == 3

    from app.harness.outbound_content import (
        GATE_ARM_POINTS, find_unarmed_dispatch_paths, stale_gate_arm_points,
    )
    assert find_unarmed_dispatch_paths() == [], find_unarmed_dispatch_paths()
    assert stale_gate_arm_points() == [], stale_gate_arm_points()
    assert len(GATE_ARM_POINTS) == 9, GATE_ARM_POINTS
