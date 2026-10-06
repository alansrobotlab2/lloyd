"""app/harness/policy.py — scope-bound expiring authority grants (#534).

The claim under test is narrow and mechanical: a non-interactive turn
(worker source, autonomy task) may not execute a durable-external or
hard-to-reverse tool unless a human minted a live, in-scope, in-quota
grant for exactly that tool shape beforehand.

Every case here calls the hook callback or `check_grants` directly with a
dict and a temporary SQLite file. There is no model, no system prompt, no
Inner Voice and no L0 block on any path these tests touch — which is the
point: enforcement that needs the prompt to say so is not enforcement.
The prompt-independence of that is itself pinned below, at source level.

One case is deliberately not hermetic: the one under the `#2093` heading reads
the shipped `autonomy/40-*.md` from the real vault, because #2093's claim — that
the file the scheduler reads declares no `grants:` block any more — is a claim
about that file and a fixture cannot evidence it. Its rationale, and why it
fails rather than skips, are at its own docstring.

Run:
  /home/alansrobotlab/lloyd/.venvs/lloyd/bin/python -m pytest tests/unit/test_grant_policy.py
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

LLOYD_HOME = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LLOYD_HOME))

from app.harness import HookRegistry  # noqa: E402
from app.harness import policy  # noqa: E402
from app.harness import safety  # noqa: E402
from app.harness.policy import (  # noqa: E402
    GRANT_MINT_TOOL,
    GrantError,
    GrantStore,
    check_grants,
    effective_tier,
    install_policy_hook,
    tool_tier,
)

# The single clock every fixture in this file is derived from. The pure-policy
# tests pass it as `now=NOW` to `check_grants`; the hook-path tests pass it as
# `now=NOW` to `install_policy_hook`, which forwards it to the same check (#973).
# No fixture here reads the wall clock, which is the point: before the hook took
# a clock, a hook-path fixture had to be minted against the real one while its
# siblings stayed frozen, and one test quietly did neither — every grant it
# minted from `_iso` was already expired by the time the hook read it, so it
# turned red on its own at 2026-09-11T12:00Z and blocked every automod round at
# the `tests` rung for the rest of that day (#848/#853). A parameter cannot be
# forgotten the way that convention was.
NOW = dt.datetime(2026, 9, 10, 12, 0, tzinfo=dt.timezone.utc)


def _iso(delta_hours: float) -> str:
    return (NOW + dt.timedelta(hours=delta_hours)).isoformat()


@pytest.fixture
def store(tmp_path) -> GrantStore:
    s = GrantStore(tmp_path / "workers.db")
    s.ensure_schema()
    return s


def _mint(store, *, scope="autonomy-task:39", tool="email_send",
          predicate="", quota=None, expires=+24.0, issued_by="alan"):
    """Mint a grant whose expiry is always `NOW + expires` hours. There is no
    `expires_at` passthrough: an expiry that came from anywhere else would have
    to be read by a clock that is not `NOW`, and that split is the bug (#973).
    Anything needing a literal expiry calls `store.mint` directly."""
    return store.mint(
        scope=scope, tool_pattern=tool, arg_predicate=predicate,
        quota=quota, issued_by=issued_by,
        expires_at=_iso(expires),
    )


def _check(store, *, scope="autonomy-task:39", tool="email_send",
           tool_input=None) -> policy.Decision:
    return check_grants(store, scope=scope, tool_name=tool,
                        tool_input=tool_input or {}, now=NOW)


def _fire(hooks: HookRegistry, tool: str, tool_input: dict | None = None):
    import asyncio
    return asyncio.new_event_loop().run_until_complete(
        hooks.fire_pre_tool_use(
            session_id="", tool_name=tool, tool_input=tool_input or {},
        )
    )


def _denied(out: dict) -> bool:
    return ((out or {}).get("hookSpecificOutput") or {}) \
        .get("permissionDecision") == "deny"


# ── The discriminating suite ───────────────────────────────────────────────
# One case per failure mode the item names. Each must be a real deny, not a
# pass that happens to look like one because the store was empty anyway —
# which is why every deny case mints a grant first and then breaks exactly
# one property of it.


def test_allowed_call_with_live_grant(store):
    """The one positive control: a live, in-scope grant authorizes and is consumed."""
    g = _mint(store)
    d = _check(store)
    assert d.allowed is True
    assert d.grant_id == g["id"]
    row = store.get(g["id"])
    assert row["consumed"] == 1


def test_expired_grant_denies(store):
    _mint(store, expires=-1.0)  # died an hour ago
    d = _check(store)
    assert d.allowed is False
    assert "expire" in d.reason.lower()


def test_wrong_scope_denies(store):
    """A grant for task #39 must not authorize task #40, same tool."""
    _mint(store, scope="autonomy-task:39")
    d = _check(store, scope="autonomy-task:40")
    assert d.allowed is False
    assert "autonomy-task:40" in d.reason


def test_wrong_tool_denies(store):
    """A grant for email_send must not authorize email_empty_trash."""
    _mint(store, tool="email_send")
    d = _check(store, tool="email_empty_trash")
    assert d.allowed is False
    assert "email_empty_trash" in d.reason


def test_over_quota_denies(store):
    _mint(store, quota=2)
    assert _check(store).allowed is True
    assert _check(store).allowed is True
    d = _check(store)
    assert d.allowed is False
    assert "quota" in d.reason.lower()


def test_worker_self_mint_denied(store):
    """A non-interactive turn may not mint its own authority — the one
    thing that would turn the whole design back into bypassPermissions."""
    d = _check(store, tool=GRANT_MINT_TOOL,
               tool_input={"scope": "autonomy-task:39", "tool": "email_send"})
    assert d.allowed is False
    assert "mint" in d.reason.lower() or "grant" in d.reason.lower()


def test_grant_revoked_mid_run_denies(store):
    """First dispatch consumes, the human revokes, the next dispatch in the
    same run is denied. A run does not hold authority past its revocation."""
    g = _mint(store)
    assert _check(store).allowed is True
    assert store.revoke(g["id"], now=NOW) is True
    d = _check(store)
    assert d.allowed is False


def test_predicate_bounds_the_arguments(store):
    _mint(store, tool="email_update", predicate="len(messageIds)<=2")
    assert _check(store, tool="email_update",
                  tool_input={"messageIds": ["a", "b"]}).allowed is True
    d = _check(store, tool="email_update",
               tool_input={"messageIds": ["a", "b", "c"]})
    assert d.allowed is False
    assert "email_update" in d.reason


def test_tier1_tools_never_reach_the_check(store):
    """vault_write / memory_add are sha-committed and revertable, so a grant
    would be pure friction. They must pass with an empty store."""
    assert tool_tier("vault_write") == 1
    d = _check(store, tool="vault_write", tool_input={"path": "x.md"})
    assert d.allowed is True
    assert d.grant_id is None
    assert store.dispatch_rows() == []


def test_deny_reason_is_the_issuance_ui(store):
    """The deny text must render the exact grant shape that would authorize
    the call — that is what makes issuance a batched renewal rather than a
    live interruption."""
    d = _check(store, tool="email_update", tool_input={"messageIds": ["a"]})
    assert d.allowed is False
    for fragment in ("grant_create", "scope", "email_update", "expires"):
        assert fragment in d.reason, d.reason


# ── Minting rules ──────────────────────────────────────────────────────────


def test_mint_without_expiry_fails_not_infinity(store):
    with pytest.raises(GrantError):
        store.mint(scope="autonomy-task:39", tool_pattern="email_send",
                   issued_by="alan", expires_at=None)
    with pytest.raises(GrantError):
        store.mint(scope="autonomy-task:39", tool_pattern="email_send",
                   issued_by="alan", expires_at="")


def test_mint_without_issuer_fails(store):
    with pytest.raises(GrantError):
        store.mint(scope="autonomy-task:39", tool_pattern="email_send",
                   issued_by="", expires_at=_iso(24))


def test_mint_rejects_unparsable_expiry(store):
    with pytest.raises(GrantError):
        store.mint(scope="autonomy-task:39", tool_pattern="email_send",
                   issued_by="alan", expires_at="next friday")


def test_mint_rejects_injection_shaped_predicate(store):
    """The predicate grammar is a mini-language, not a filter handed to
    `eval`. Anything outside the two accepted shapes is refused at mint."""
    for bad in ("__import__('os').system('rm -rf /')",
                "len(messageIds)<=2 or True",
                "messageIds; DROP TABLE queue;--",
                "len(..len(x))<=2"):
        with pytest.raises(GrantError):
            store.mint(scope="autonomy-task:39", tool_pattern="email_update",
                       arg_predicate=bad, issued_by="alan", expires_at=_iso(24))


def test_worker_scope_cannot_be_minted_by(store):
    """`minted_by` records who minted. A row whose issuer is a worker scope is
    the audit red line acceptance counts, so mint refuses to write one."""
    with pytest.raises(GrantError):
        store.mint(scope="autonomy-task:39", tool_pattern="email_send",
                   issued_by="alan", expires_at=_iso(24),
                   minted_by="worker:scheduled-task")


def test_no_renewal_endpoint_renewal_is_a_new_row(store):
    _mint(store)
    with pytest.raises(AttributeError):
        getattr(store, "renew")  # there is exactly one write path: mint()


@pytest.fixture
def _worker_env(monkeypatch):
    """Make `_worker_run_options` hermetic: it reads the live config, the
    system prompt builder and the model env, none of which the gate is about."""
    from app import prompt_builder
    monkeypatch.setattr(prompt_builder, "build_system_prompt", lambda *a, **k: "sys")
    from app import autonomy
    monkeypatch.setattr(autonomy, "_get_model_env", lambda *a, **k: {})
    return None


# ── Store shape (acceptance: PRAGMA shows a NOT-NULL expires_at) ───────────


def test_grant_table_notnull_expiry_and_partial_index(store):
    with sqlite3.connect(str(store.db_path)) as conn:
        # PRAGMA table_info: (cid, name, type, notnull, dflt, pk)
        cols = {r[1]: r[3] for r in
                conn.execute("PRAGMA table_info(authority_grants)")}
        assert "expires_at" in cols
        assert cols["expires_at"] == 1, "expires_at must be NOT NULL"
        assert cols["issued_by"] == 1, "issued_by must be NOT NULL"
        assert cols["scope"] == 1 and cols["tool_pattern"] == 1
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='authority_grants'")]
        # A partial index on (scope, expires_at) — the live-grant lookup.
        assert any("scope" in n for n in names), names
        sql = conn.execute("SELECT sql FROM sqlite_master WHERE type='index' "
                           "AND name LIKE 'idx_grants%'").fetchone()[0]
        assert "revoked_at IS NULL" in sql, "the index must be partial"


def test_direct_sql_insert_without_expiry_is_rejected_by_the_schema(store):
    """The NOT NULL is the last line, behind the mint validator: a row written
    by hand (or by a future writer that skips validation) cannot be born
    immortal."""
    with sqlite3.connect(str(store.db_path)) as conn, pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO authority_grants (scope, tool_pattern, issued_by,"
            " issued_at) VALUES ('worker:x','email_send','alan','2026-09-10')")


def test_queue_schema_creates_the_grant_table(tmp_path):
    """The store lives in the worker job-queue DB, created by the same
    `_init_db` every other consumer calls — not by a new service."""
    from workers.queue import WorkQueue
    q = WorkQueue(tmp_path / "workers.db")
    with sqlite3.connect(str(q.db_path)) as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert "authority_grants" in tables


def test_an_existing_database_gets_the_table_on_next_init(tmp_path):
    """`~/lloyd/workers.db` already exists on disk, so what matters is not the
    fresh-DB path but that `_init_db` — which every consumer calls on the
    existing file, and which is `CREATE TABLE IF NOT EXISTS` — adds the grant
    table to it rather than requiring a migration step nobody will run."""
    from workers.queue import WorkQueue
    path = tmp_path / "workers.db"
    WorkQueue(path)
    with sqlite3.connect(str(path)) as conn:
        conn.execute("DROP TABLE authority_grants")  # takes its index with it
    WorkQueue(path)  # a second process opening the same existing file
    with sqlite3.connect(str(path)) as conn:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        kept = {r[0] for r in conn.execute("SELECT name FROM sqlite_master "
                                          "WHERE type='table'")}
    assert "authority_grants" in tables
    assert kept == {"queue", "sqlite_sequence", "runs", "watermarks",
                    "authority_grants", "grant_dispatch"}


# ── Hook wiring: the non-interactive dispatch paths ────────────────────────


def test_hook_denies_ungranted_tier2_from_worker_scope(store, monkeypatch):
    monkeypatch.setenv("LLOYD_GRANT_DB", str(store.db_path))
    hooks = HookRegistry()
    install_policy_hook(hooks, store=store, scope="worker:scheduled-task",
                        now=NOW)
    assert _denied(_fire(hooks, "email_send", {"to": "x@y.z"}))


def test_hook_allows_granted_call(store, monkeypatch):
    """The seam's positive half: the hook is installed at an instant inside the
    grant's life and the fixture expiry is derived from the frozen `NOW` like
    every other fixture in this file, so the granted tier-2 call goes through.

    This is the case that could not be written hermetically before #973 — the
    hook read the wall clock, so the same fixture was a grant that expired on
    2026-09-11 and this assertion has been red ever since. It also cannot pass
    if `now` is accepted but not forwarded: the wall clock is past NOW + 24 h,
    which is why the ledger line below names the instant the decision was made
    at rather than merely that the decision was allow."""
    monkeypatch.setenv("LLOYD_GRANT_DB", str(store.db_path))
    g = _mint(store, scope="worker:scheduled-task")      # expires NOW + 24 h
    hooks = HookRegistry()
    install_policy_hook(hooks, store=store, scope="worker:scheduled-task",
                        now=NOW)
    assert not _denied(_fire(hooks, "email_send", {"to": "x@y.z"}))
    assert store.get(g["id"])["consumed"] == 1
    rows = store.dispatch_rows()
    assert [(r["decision"], r["at"]) for r in rows] == [("allow", _iso(0.0))]


def test_hook_denies_granted_call_once_the_injected_instant_is_past_the_expiry(
        store, monkeypatch):
    """The seam's negative half. Same fixture, installed two days later: the
    grant has expired, so the call is denied and the reason names that grant's
    own expiry.

    The ledger line is what makes this a test of the seam rather than a test of
    the calendar. Deny is also what the wall clock would answer here (it is past
    NOW + 24 h too), so a hook that accepted `now` and ignored it would still
    satisfy the deny; the recorded instant cannot be faked that way."""
    monkeypatch.setenv("LLOYD_GRANT_DB", str(store.db_path))
    g = _mint(store, scope="worker:scheduled-task")      # expires NOW + 24 h
    hooks = HookRegistry()
    install_policy_hook(hooks, store=store, scope="worker:scheduled-task",
                        now=NOW + dt.timedelta(hours=48))
    out = _fire(hooks, "email_send", {"to": "x@y.z"})
    assert _denied(out)
    reason = out["hookSpecificOutput"]["permissionDecisionReason"]
    assert "expired at" in reason
    assert store.get(g["id"])["expires_at"] in reason
    assert [(r["decision"], r["at"]) for r in store.dispatch_rows()] == [
        ("deny", _iso(48.0))]


def test_hook_passes_tier1_through(store):
    hooks = HookRegistry()
    install_policy_hook(hooks, store=store, scope="worker:scheduled-task",
                        now=NOW)
    assert not _denied(_fire(hooks, "vault_write", {"path": "a.md"}))


def test_store_failure_fails_closed(store, tmp_path):
    """`HookRegistry.fire_pre_tool_use` treats a raising callback as a pass,
    so a grant gate that crashed would silently open the gate. The callback
    must therefore deny on its own errors rather than raise."""
    broken = GrantStore(tmp_path / "gone" / ".." / "nope.db")
    hooks = HookRegistry()
    install_policy_hook(hooks, store=broken, scope="worker:scheduled-task",
                        now=NOW)
    assert _denied(_fire(hooks, "email_send", {"to": "x@y.z"}))


def test_worker_options_install_the_gate(_worker_env, monkeypatch):
    """The whole item rests on this: `run_prompt_on_primary` turns used to run
    with `hooks=None`, so nothing gated any non-Bash tool on them."""
    from workers.sources._common import _worker_run_options
    opts = _worker_run_options(max_turns=1)
    assert opts.hooks is not None, "worker turns must carry a policy hook"


def test_run_prompt_on_primary_cannot_call_grant_create(_worker_env, monkeypatch,
                                                        store):
    """Acceptance: a `run_prompt_on_primary` turn cannot call `grant_create`.
    Enforced twice — the tool is not advertised, and the hook denies the call
    if a local model emits it anyway."""
    monkeypatch.setenv("LLOYD_GRANT_DB", str(store.db_path))
    from workers.sources._common import _worker_run_options
    opts = _worker_run_options(max_turns=1)
    assert GRANT_MINT_TOOL in opts.disallowed_tools
    assert f"mcp__lloyd-mcp__{GRANT_MINT_TOOL}" in opts.disallowed_tools
    assert _denied(_fire(opts.hooks, GRANT_MINT_TOOL,
                         {"scope": "worker:scheduled-task",
                          "tool": "email_send"}))


def test_worker_options_still_ban_automod(_worker_env):
    """The grant hook is additive; it must not have cost the automod ban."""
    from workers.sources._common import WORKER_AUTOMOD_BAN, _worker_run_options
    opts = _worker_run_options(max_turns=1)
    for name in WORKER_AUTOMOD_BAN:
        assert name in opts.disallowed_tools


def test_autonomy_run_task_installs_the_gate():
    """autonomy.run_task builds its own RunOptions — a second dispatch path —
    so the hook has to be installed there too, not only on worker turns."""
    import inspect

    from app import autonomy
    src = inspect.getsource(autonomy.run_task)
    assert "install_policy_hook" in src, "run_task must gate its own turn"
    assert "grant_create" in src or "GRANT_MINT_TOOL" in src


# ── Autonomy task frontmatter: the second mint path ────────────────────────


def _write_task(aut, task_id, **fm):
    import yaml
    base = {
        "id": task_id, "name": f"task{task_id}",
        "status": "up_next", "frequency": "daily", "priority": "medium",
        "skill_name": str(aut._SKILL_FOR_TESTS), "timeout_seconds": 2,
    }
    base.update(fm)
    body = {k: v for k, v in base.items() if k != "skill_file"}
    (aut.AUTONOMY_DIR / f"{task_id}-task.md").write_text(
        "---\n" + yaml.dump(body, default_flow_style=False) + "---\n\nbody\n",
        encoding="utf-8")


@pytest.fixture
def aut(tmp_path, monkeypatch):
    from app import autonomy
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", tmp_path / "autonomy")
    monkeypatch.setattr(autonomy, "AUTONOMY_RUNS_DIR", tmp_path / "runs")
    autonomy.AUTONOMY_DIR.mkdir()
    skill = tmp_path / "SKILL.md"
    skill.write_text("# test skill\nDo the thing.\n")
    monkeypatch.setattr(autonomy, "_SKILL_FOR_TESTS", str(skill), raising=False)
    return autonomy


def test_malformed_grants_block_fails_task_closed(aut):
    """A task whose `grants:` block cannot be understood must NOT be drained.
    Falling back to "ignore the bad grants and run anyway" is precisely
    fail-open: the malformed block is the human's authorization and ignoring
    it means running with none."""
    _write_task(aut, 1, grants=[{"tool": "email_send"}])  # no expires_at
    assert [t["id"] for t in aut._all_runnable_tasks()] == []
    assert [t["id"] for t in aut.get_due_tasks()] == []


def test_non_list_grants_block_fails_task_closed(aut):
    _write_task(aut, 2, grants="email_send please")
    assert [t["id"] for t in aut._all_runnable_tasks()] == []


def test_valid_grants_block_keeps_task_runnable(aut):
    _write_task(aut, 3, grants=[{
        "tool": "email_send", "predicate": "len(messages)<=3",
        "quota": 5, "expires_at": _iso(24), "issued_by": "alan",
    }])
    ids = [t["id"] for t in aut._all_runnable_tasks()]
    assert ids == [3]


def test_task_grants_are_materialized_into_the_store(aut, store, monkeypatch):
    """The frontmatter block is a declarative mint: at run start it lands as
    ordinary grant rows, so the check path has exactly one place to look."""
    _write_task(aut, 4, grants=[{"tool": "email_send", "expires_at": _iso(24),
                                 "issued_by": "alan"}])
    task = aut._all_runnable_tasks()[0]
    n = policy.sync_task_grants(store, task_id=4, scope="autonomy-task:4",
                                grants=task["grants"], now=NOW)
    assert n == 1
    assert _check(store, scope="autonomy-task:4").allowed is True


def test_materialize_is_idempotent_and_never_renews(aut, store):
    """Re-running a task must not silently extend a grant's life — renewal is
    the human editing the file, and a run that re-issued its own expiry would
    be self-mint wearing frontmatter."""
    grants = [{"tool": "email_send", "expires_at": _iso(24), "issued_by": "alan"}]
    policy.sync_task_grants(store, task_id=5, scope="autonomy-task:5",
                            grants=grants, now=NOW)
    first = store.live(scope="autonomy-task:5", now=NOW)
    n = policy.sync_task_grants(store, task_id=5, scope="autonomy-task:5",
                                grants=grants, now=NOW)
    assert n == 0
    second = store.live(scope="autonomy-task:5", now=NOW)
    assert len(second) == len(first) == 1
    assert second[0]["expires_at"] == first[0]["expires_at"]


# ── #1949: a revocation must outlive the next dispatch of the task that declared it
#
# `run_task` calls `sync_task_grants` on every run (app/autonomy.py:4107), and the
# task file is an ordinary agent-writable markdown file. So the question these four
# cases settle is whether `grant_revoke` is a decision or a delay: if the dedupe
# that asks "is this pair already covered?" reads only `store.live()` — whose
# `revoked_at IS NULL` filter hides the withdrawn row — the next scheduler re-fire
# re-derives the authority from the file that declared it, on the file's own
# unchanged expiry, and the revocation lasted until then and no longer.


def test_a_revoked_unexpired_grant_is_not_re_materialized_by_the_next_run(store):
    """Clause 1: re-running the declaring task must not resurrect its revoked row.

    Same `(scope, tool, predicate)`, same unchanged `grants:` block, expiry still
    ahead of `now` — the only thing that changed between the two calls is that a
    human revoked the row. So the second materialisation returns 0 new grants and
    `check_grants` still denies, citing that revocation and that grant id.
    """
    grants = [{"tool": "email_send", "expires_at": _iso(24), "issued_by": "alan"}]
    assert policy.sync_task_grants(store, task_id=41, scope="autonomy-task:41",
                                   grants=grants, now=NOW) == 1
    gid = store.live(scope="autonomy-task:41", now=NOW)[0]["id"]
    assert store.revoke(gid, now=NOW) is True
    assert store.live(scope="autonomy-task:41", now=NOW) == []
    denied = _check(store, scope="autonomy-task:41")
    assert denied.allowed is False, denied.reason

    n = policy.sync_task_grants(store, task_id=41, scope="autonomy-task:41",
                                grants=grants, now=NOW)
    assert n == 0, (
        f"the next run re-materialised {n} row(s) over a revocation that had not "
        "expired; a withdrawn row still covers its (scope, tool, predicate)")
    after = _check(store, scope="autonomy-task:41")
    assert after.allowed is False, after.reason
    assert "revoked" in after.reason and f"#{gid}" in after.reason, after.reason


def test_a_later_declared_expiry_still_materializes_after_a_revocation(store):
    """Clause 2: renewal stays the human editing the file, revocation included.

    The conservative half of the rule: what a revocation forbids is the *same*
    authority coming back on the *same* terms, so a block whose declared
    `expires_at` is later than the revoked row's is a fresh act of approval and
    must go through and allow. The first assertion below is the boundary that
    makes that meaningful — the unedited block, declaring exactly the expiry the
    revocation killed, still materialises nothing.
    """
    declared = [{"tool": "email_send", "expires_at": _iso(24),
                 "issued_by": "alan"}]
    assert policy.sync_task_grants(store, task_id=42, scope="autonomy-task:42",
                                   grants=declared, now=NOW) == 1
    gid = store.live(scope="autonomy-task:42", now=NOW)[0]["id"]
    assert store.revoke(gid, now=NOW) is True
    assert policy.sync_task_grants(store, task_id=42, scope="autonomy-task:42",
                                   grants=declared, now=NOW) == 0

    edited = [{"tool": "email_send", "expires_at": _iso(48), "issued_by": "alan"}]
    n = policy.sync_task_grants(store, task_id=42, scope="autonomy-task:42",
                                grants=edited, now=NOW)
    assert n == 1, (
        "a human extending `expires_at` past the revoked row's is re-approving "
        "the grant, so the materialisation must go through")
    fresh = store.live(scope="autonomy-task:42", now=NOW)
    assert len(fresh) == 1 and fresh[0]["id"] != gid, fresh
    assert fresh[0]["expires_at"] == _iso(48), fresh
    allowed = _check(store, scope="autonomy-task:42")
    assert allowed.allowed is True, allowed.reason


def test_an_expired_unrevoked_grant_does_not_block_materialization(store):
    """Clause 3: expiry and revocation stay distinguishable in both directions.

    A row that simply ran out is no one's decision, so unlike a revoked row it
    must NOT suppress the re-mint — otherwise this fix would trade a resurrection
    bug for a nightly that can never be re-armed without deleting a row. The
    first materialisation runs at `NOW - 3h` on an expiry that lands at
    `NOW - 2h`, so by `NOW` the row is expired with `revoked_at` never set; the
    fresh block declaring `NOW + 24h` then has to mint and allow.
    """
    ran_out_at = (NOW - dt.timedelta(hours=2)).isoformat()
    stale = [{"tool": "email_send", "expires_at": ran_out_at,
              "issued_by": "alan"}]
    assert policy.sync_task_grants(store, task_id=43, scope="autonomy-task:43",
                                   grants=stale,
                                   now=NOW - dt.timedelta(hours=3)) == 1
    dead = store.live(scope="autonomy-task:43",
                      now=NOW - dt.timedelta(hours=3))[0]
    assert dead["revoked_at"] is None and dead["expires_at"] == ran_out_at, dead
    assert store.live(scope="autonomy-task:43", now=NOW) == []

    fresh = [{"tool": "email_send", "expires_at": _iso(24), "issued_by": "alan"}]
    n = policy.sync_task_grants(store, task_id=43, scope="autonomy-task:43",
                                grants=fresh, now=NOW)
    assert n == 1, (
        "an expired row is not a revocation: it must not block the re-arm")
    allowed = _check(store, scope="autonomy-task:43")
    assert allowed.allowed is True, allowed.reason


def test_a_declined_rematerialization_warns_naming_the_task_and_grant(store,
                                                                      caplog):
    """Clause 4: a task running without its withdrawn authority is said out loud.

    Declining is silent by default, and the silence is the failure mode: the run
    proceeds, its tool call is denied later (or in a week, or never), and nothing
    in the log connects a task that is quietly not doing its job to the decision
    that undid it. One WARNING, naming the task id and the revoked grant id it is
    honouring, from the `lloyd-harness-policy` logger this module defines at
    `policy.py:61` — the one its sibling `[grants] task #…` warning already uses
    on the mint-under-a-standing-shadow path (`policy.py:1410`), so a declined
    re-mint lands in the same stream an operator already reads for grants rather
    than a new one nobody subscribed to.
    """
    import logging

    grants = [{"tool": "email_send", "expires_at": _iso(24), "issued_by": "alan"}]
    policy.sync_task_grants(store, task_id=44, scope="autonomy-task:44",
                            grants=grants, now=NOW)
    gid = store.live(scope="autonomy-task:44", now=NOW)[0]["id"]
    assert store.revoke(gid, now=NOW) is True
    caplog.set_level(logging.WARNING, logger="lloyd-harness-policy")
    caplog.clear()

    assert policy.sync_task_grants(store, task_id=44, scope="autonomy-task:44",
                                   grants=grants, now=NOW) == 0

    warns = [r.getMessage() for r in caplog.records
             if r.name == "lloyd-harness-policy"
             and r.levelno >= logging.WARNING]
    assert len(warns) == 1, f"exactly one warning expected, got: {warns}"
    assert "task #44" in warns[0], warns[0]
    assert f"#{gid}" in warns[0], warns[0]
    assert "revoked" in warns[0], warns[0]


# ── #2023: a revival over a revocation is as loud as the decline beside it ──

def _policy_warnings(caplog) -> list[str]:
    import logging
    return [r.getMessage() for r in caplog.records
            if r.name == "lloyd-harness-policy" and r.levelno >= logging.WARNING]


def test_a_revival_over_a_revocation_warns_naming_both_grants(store, caplog):
    """The strictly-later-expiry mint supersedes a human's `grant_revoke`, and
    until #2023 said nothing: one WARNING naming the task, the revoked grant, its
    `revoked_at`, and the grant just minted. The decline one call earlier keeps
    its own text."""
    import logging

    declared = [{"tool": "email_send", "expires_at": _iso(24), "issued_by": "alan"}]
    policy.sync_task_grants(store, task_id=45, scope="autonomy-task:45",
                            grants=declared, now=NOW)
    old = store.live(scope="autonomy-task:45", now=NOW)[0]["id"]
    assert store.revoke(old, now=NOW) is True
    caplog.set_level(logging.WARNING, logger="lloyd-harness-policy")
    caplog.clear()

    assert policy.sync_task_grants(store, task_id=45, scope="autonomy-task:45",
                                   grants=declared, now=NOW) == 0
    decline = _policy_warnings(caplog)
    assert len(decline) == 1 and "is not materialized" in decline[0], decline
    assert "a revocation is not renewed by re-running the task" in decline[0]
    caplog.clear()

    edited = [{"tool": "email_send", "expires_at": _iso(48), "issued_by": "alan"}]
    assert policy.sync_task_grants(store, task_id=45, scope="autonomy-task:45",
                                   grants=edited, now=NOW) == 1
    new = store.live(scope="autonomy-task:45", now=NOW)[0]["id"]
    assert new != old
    warns = _policy_warnings(caplog)
    assert len(warns) == 1, f"exactly one revival warning expected: {warns}"
    line = warns[0]
    assert "task #45" in line and "over a revocation" in line, line
    assert f"grant #{old} was revoked at {NOW.isoformat()}" in line, line
    assert f"minted grant #{new}" in line, line


def test_a_first_mint_and_an_expired_rearm_say_nothing_about_a_revocation(
        store, caplog):
    """The two negatives. A fresh pair has no prior row; a pair whose row merely
    ran out was never anyone's decision. Both mint 1 and neither warns — a
    nightly that warned on every re-arm would bury the one line that matters."""
    import logging

    caplog.set_level(logging.WARNING, logger="lloyd-harness-policy")
    fresh = [{"tool": "email_send", "expires_at": _iso(24), "issued_by": "alan"}]
    assert policy.sync_task_grants(store, task_id=46, scope="autonomy-task:46",
                                   grants=fresh, now=NOW) == 1
    assert _policy_warnings(caplog) == []

    ran_out = [{"tool": "email_send", "issued_by": "alan",
                "expires_at": (NOW - dt.timedelta(hours=2)).isoformat()}]
    assert policy.sync_task_grants(store, task_id=47, scope="autonomy-task:47",
                                   grants=ran_out,
                                   now=NOW - dt.timedelta(hours=3)) == 1
    caplog.clear()
    assert policy.sync_task_grants(store, task_id=47, scope="autonomy-task:47",
                                   grants=fresh, now=NOW) == 1
    assert [w for w in _policy_warnings(caplog) if "revok" in w] == [], (
        _policy_warnings(caplog))


# ── #2021: the denial for a spent declared grant names the edit that works ──

def _spend(store, *, scope, tool, tool_input):
    for _ in range(2):
        assert check_grants(store, scope=scope, tool_name=tool,
                            tool_input=tool_input, now=NOW).allowed
    denied = check_grants(store, scope=scope, tool_name=tool,
                          tool_input=tool_input, now=NOW)
    assert denied.allowed is False and "over quota (2/2)" in denied.reason, (
        denied.reason)
    return denied.reason


def _mint_line(reason: str, scope: str, tool: str) -> str:
    import re
    expiry = re.search(r"expires_at='([^']+)'\, issued_by", reason).group(1)
    return (f"grant_create(scope='{scope}', tool='{tool}', "
            f"expires_at='{expiry}', issued_by='alan')")


@pytest.mark.parametrize("tool,tool_input", [
    ("email_send", {}),
    (policy.SCHEDULE_STATE_TOOL, {"id": 68, "status": "up_next"}),
])
def test_a_spent_frontmatter_grant_is_denied_with_the_file_edit_that_works(
        store, tool, tool_input):
    """The file already has the `grants:` block — it minted the refusing row —
    so the reason must not send the reader to add one. Still a deny, still
    `over quota (2/2)`: no dispatch decision moves."""
    scope = "autonomy-task:40"
    assert policy.sync_task_grants(
        store, task_id=40, scope=scope, now=NOW,
        grants=[{"tool": tool, "quota": 2, "expires_at": _iso(24),
                 "issued_by": "alan"}]) == 1
    gid = store.live(scope=scope, now=NOW)[0]["id"]

    reason = _spend(store, scope=scope, tool=tool, tool_input=tool_input)

    assert "adds a `grants:` block" not in reason, reason
    assert f"autonomy task #40's own `grants:` block" in reason, reason
    assert "`40-*.md`" in reason and "`expires_at:`" in reason and "`quota:`" in reason
    assert f"grant_revoke(grant_id={gid})" in reason, reason
    assert _mint_line(reason, scope, tool) in reason, (
        "the paste-ready line stays as the one-off route")


def test_a_spent_hand_minted_grant_keeps_its_denial_text_exactly(store):
    """The other half of clause 3: an `interactive-tool` row is described as
    before — paste-ready line, and for the schedule tool the `grants:`-block
    sentence, which is true advice for a file that declares nothing."""
    scope = "autonomy-task:39"
    for tool, tool_input in (("email_send", {}),
                             (policy.SCHEDULE_STATE_TOOL,
                              {"id": 68, "status": "up_next"})):
        store.mint(scope=scope, tool_pattern=tool, quota=2, issued_by="alan",
                   expires_at=NOW + dt.timedelta(hours=24),
                   minted_by="interactive-tool", now=NOW)
        reason = _spend(store, scope=scope, tool=tool, tool_input=tool_input)
        assert reason.endswith(_mint_line(reason, scope, tool)), reason
        assert "grant_revoke" not in reason and "minted from autonomy task" not in reason
        assert ("adds a `grants:` block" in reason) is (
            tool == policy.SCHEDULE_STATE_TOOL), reason


# ── #1949 claim 1: the two ledgers are one store, on committed bytes ───────
#
# The item's literal premise was a restart asymmetry: does `_tool_effects.py`'s
# dedup key survive a `lloyd-mcp`/`lloyd-backend` restart while
# `authority_grants`/`grant_dispatch` do not, or vice versa. The answer is that
# the question has no referent, and the evidence is a table list, not a probe:
# all six names are tables in ONE sqlite file — `~/lloyd-data/workers.db`,
# resolved by `workers.queue.configured_db_path()` on both sides
# (`agent_mcp/_tool_effects.py:158-170`, `policy.default_store()`,
# config.yaml:1536 `db_path: ${LLOYD_DATA}/workers.db`, no `LLOYD_GRANT_DB` or
# `LLOYD_EFFECT_LEDGER_DB` in `agent-services/supervisord.conf`). One file means
# one WAL, so no restart of either service, in either order, can diverge them:
# a spent `tool_effects` row cannot replay as fresh, and a revoked or consumed
# grant row cannot come back with `revoked_at NULL` / `consumed 0`.
#
# The vault artifact below is what makes that claim re-checkable by a reader who
# has no live database. `backlog/data/workers.db` is the committed extract
# #1946 filed for its ROWS; #1949 needs its SCHEMA, so
# `backlog/data/workers-authority-extract.py` now also copies `queue`, `runs`,
# `tool_effects` and `watermarks` as the live file's own DDL with zero rows —
# those four columns hold turn payloads, model output and call results, which
# have no place in a notes repository, and the claim about them is which tables
# share one file, not what is inside them.


QUOTED_INVENTORY = ["authority_grants", "egress_events", "grant_dispatch",
                    "queue", "runs", "tool_effects", "watermarks"]


def _witness_db():
    """The committed extract's bytes, or a skip if this machine has no vault.

    Deliberately not a fixture that copies the file elsewhere: the clause says
    the command must run against the committed extract, and reading it in place,
    read-only, is exactly that. `mode=ro` so a test run can never write the
    artifact it is certifying.
    """
    path = Path.home() / "obsidian" / "backlog" / "data" / "workers.db"
    if not path.exists():
        pytest.skip(f"no vault witness at {path} on this machine")
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _tables(conn) -> list:
    return sorted(r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name <> 'sqlite_sequence' ORDER BY name"))


def test_the_committed_witness_bytes_hold_both_ledgers_in_one_file():
    """One file, seven tables, zero payload rows — read out of committed bytes.

    `sqlite_sequence` is excluded from the inventory throughout: sqlite creates
    it for its AUTOINCREMENT tables, and bookkeeping is not a store. The
    `count(*) from sqlite_master` figure #1946's clause names is larger than
    seven for the same reason plus the autoindexes the copied DDL's UNIQUE and
    PRIMARY KEY constraints bring with them; both figures are in the marker, and
    neither is the claim.
    """
    conn = _witness_db()
    try:
        assert _tables(conn) == QUOTED_INVENTORY, (
            "the committed extract is not the inventory the item quotes, so a "
            "reader with only the vault cannot re-derive the one-store claim")
        for name in QUOTED_INVENTORY:
            ddl = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                (name,)).fetchone()
            assert ddl and ddl[0] and ddl[0].strip(), (
                f"{name} is not a real table in the extract")
        # The two safety columns the item's probe was written to chase.
        effects = {d[1] for d in conn.execute("PRAGMA table_info(tool_effects)")}
        grants = {d[1] for d in conn.execute("PRAGMA table_info(authority_grants)")}
        assert "effect_key" in effects and "status" in effects, effects
        assert {"revoked_at", "consumed", "expires_at"} <= grants, grants
        # Zero payload rows, with a positive control so the count cannot be
        # vacuous: a check whose denominator can be zero is not a check.
        for name in ("queue", "runs", "tool_effects", "watermarks"):
            assert conn.execute(f"SELECT count(*) FROM {name}").fetchone()[0] == 0, (
                f"{name} was committed with rows, which are prompt text in a "
                "notes repository")
        for name in ("authority_grants", "grant_dispatch", "egress_events"):
            assert conn.execute(f"SELECT count(*) FROM {name}").fetchone()[0] > 0, (
                f"{name} is empty, so the zero counts above prove nothing")
    finally:
        conn.close()


def test_the_witness_marker_reports_the_inventory_its_bytes_hold():
    """The marker cannot drift from the bytes it describes."""
    marker_path = (Path.home() / "obsidian" / "backlog" / "data"
                   / "workers-authority.witness.md")
    if not marker_path.exists():
        pytest.skip(f"no marker at {marker_path} on this machine")
    conn = _witness_db()
    try:
        tables, objects = _tables(conn), conn.execute(
            "SELECT count(*) FROM sqlite_master").fetchone()[0]
    finally:
        conn.close()
    text = marker_path.read_text(encoding="utf-8")
    block = text.split("```json")[1].split("```")[0]
    marker = json.loads(block)
    assert marker["table_inventory"] == tables, (
        "the marker names a different inventory than the bytes hold")
    assert marker["sqlite_master_entries"] == objects, (
        "the marker's object count is not what `select count(*) from "
        f"sqlite_master` answers over these bytes ({objects})")
    assert sum(marker["sqlite_master_by_type"].values()) == objects, (
        "the per-type breakdown does not account for every object")
    assert marker["rows_not_copied"] == {k: 0 for k in
                                         ("queue", "runs", "tool_effects",
                                          "watermarks")}, marker["rows_not_copied"]


# A third node belonged here — the same inventory read from the running queue's
# own `~/lloyd-data/workers.db`, which is where "one file, so no restart can
# diverge them" is actually about. It was written, and the promotion gate settled
# it: the gate runs the suite under a round home (SM_20261001_060931, `home/`
# beside the worktree) so `Path.home() / "lloyd-data" / "workers.db"` resolves to
# the sandbox's own store, which holds six of the seven names but no
# `egress_events`, and the node failed with `no longer holds
# ['egress_events']`. A unit test that reaches the real user home is the wrong
# shape here: that home swap exists precisely so a fixture cannot read, or
# delete, the running system — and it is why the two nodes above read the VAULT,
# which the round home symlinks (`home/obsidian` → the real vault, so the gate
# ran them against the landed bytes rather than skipping them). The live
# inventory was measured once, by hand, and is recorded on the item; what stays
# pinned is the committed extract's copy of it — the same seven objects, the live
# file's own DDL, and no dependency on a machine.
# ── Prompt-independence (the acceptance's framing) ─────────────────────────


def test_policy_module_does_not_read_the_prompt_surface():
    """'100% with Inner Voice off and the L0 block stripped' cannot be
    demonstrated by running a model and hoping. It is demonstrated by the
    module never touching those surfaces at all — no prompt builder, no
    session flags, no inner-voice import, no model call."""
    import re

    src = (LLOYD_HOME / "app" / "harness" / "policy.py").read_text()
    for forbidden in ("prompt_builder", "inner_voice", "build_system_prompt",
                      "SOUL.md", "run_query", "heuristic", "vLLM"):
        assert forbidden not in src, f"policy.py must not reference {forbidden}"
    # Word-boundary, because `fullmatch` contains the letters.
    assert not re.search(r"\b(llm|LLM|LLM_HOME)\b", src), "no model reference"


def test_check_grants_is_pure_stdlib():
    import inspect

    import app.harness.policy as mod
    src = inspect.getsource(mod)
    assert "async def check_grants" not in src
    # Only stdlib + the in-house hook/store types, nothing that could reach a
    # model or a prompt. Parsed from `^import X` / `^from X import Y`.
    import re
    top = set(re.findall(r"^(?:from|import)\s+([A-Za-z_][\w.]*)", src, re.M))
    allowed = {
        "asyncio", "contextvars", "dataclasses", "datetime", "json",
        "logging", "re", "sqlite3", "os", "typing", "pathlib", "sys",
        "time", "threading", "uuid", "app", "workers", "__future__",
    }
    for name in top:
        assert name.split(".")[0] in allowed, f"unexpected import in policy.py: {name}"


# ── Dispatch ledger (the join #521/#525 need) ──────────────────────────────


def test_dispatch_is_ledgered_with_grant_id_or_reason(store):
    _mint(store)
    _check(store, tool="vault_write")            # tier 1 → not ledgered
    _check(store)                               # allow → ledgered with grant_id
    _check(store, tool="calendar_delete_event")  # deny → ledgered with reason
    rows = store.dispatch_rows()
    assert len(rows) == 2
    allowed = [r for r in rows if r["decision"] == "allow"]
    denied = [r for r in rows if r["decision"] == "deny"]
    assert len(allowed) == 1 and allowed[0]["grant_id"] is not None
    assert len(denied) == 1 and denied[0]["reason"]


# ── Scheduler state is authority too (#724) ─────────────────────────────────
# `autonomy_write_task` rewrites the six fields the scheduler reads to decide
# whether and when a task runs. Until #724 it sat at tier 1 while its sibling
# `autonomy_delete_task` sat at tier 3, so an unattended turn that could not
# send an email could re-arm a nightly job. The dispatches behind that — a
# backlog-triage turn arming #84, automod implement turns parking and re-arming
# #85, autonomy task #40 arming #68 and #85 while it ran — are counted on the
# item, with the command that reproduces them, because `event_logs/` is a
# rolling window and a figure quoted here would be wrong within days.
#
# What the counts settled on is structural and stable, so it is what the cases
# below encode: the benign writes carried `activity_note` (plus a run `summary`)
# and nothing else, and every armed-and-parked write touched one of the six
# fields. A gate keyed on the target id — the fix this item originally proposed
# — would therefore have blocked the notes and passed the schedule writes, which
# is why the predicate is the field set.

#: A schedule write: one of the six clauses names, on an existing task.
SCHED_CALL = {"id": 68, "status": "up_next"}
#: The run-record habit the skills prescribe: own id, activity note, nothing else.
RUN_RECORD_CALL = {"id": 68, "activity_note": "run completed"}


def test_autonomy_write_task_is_tier2_while_its_read_siblings_stay_tier1():
    """The tier table is the whole gate; `tool_tier` returns 1 for anything
    unlisted, so a schedule-rewriting tool listed nowhere is a tool with no
    check at all. The read siblings must NOT be promoted — a task that cannot
    list the board cannot do its job."""
    assert tool_tier("autonomy_write_task") == 2
    assert tool_tier("autonomy_delete_task") == 3  # unchanged
    for read_only in ("autonomy_tasks", "autonomy_get_task", "autonomy_health"):
        assert tool_tier(read_only) == 1, read_only


def test_unattended_schedule_write_is_denied_without_a_grant(store, monkeypatch):
    """The acceptance's negative half: a grant-scoped turn, no live grant, the
    call denied and the reason naming the scope."""
    from app.harness import HookRegistry

    monkeypatch.setattr(policy, "default_store", lambda: store)
    hooks = HookRegistry()
    policy.install_policy_hook(hooks, scope="autonomy-task:68", now=NOW)
    out = _fire(hooks, "autonomy_write_task", SCHED_CALL)
    assert _denied(out) is True
    reason = out["hookSpecificOutput"]["permissionDecisionReason"]
    assert "autonomy-task:68" in reason
    assert "autonomy_write_task" in reason


def test_deny_reason_names_the_target_and_the_field_that_moved(store):
    """A denial a run cannot act on is a run that retries. The reason has to
    say which task and which dispatch-affecting field, not just the tool."""
    d = _check(store, scope="autonomy-task:68", tool="autonomy_write_task",
               tool_input={"id": 68, "depends_on": 42, "frequency": "every-15min"})
    assert d.allowed is False
    assert "target #68" in d.reason
    assert "depends_on" in d.reason and "frequency" in d.reason
    assert "grants:" in d.reason  # points at the frontmatter route, not only the tool


@pytest.mark.parametrize("field,value", [
    ("status", "up_next"), ("depends_on", 42), ("frequency", "daily"),
    ("scheduled_at", "2026-09-21T01:00:00Z"), ("auto_advance", True),
    ("skill_name", "nightly-vault-maintenance"),
])
def test_each_dispatch_affecting_field_is_gated(store, field, value):
    """Six clauses, six fields. A gate that missed one is a gate with a hole
    the next nightly walks through."""
    d = _check(store, scope="autonomy-task:68", tool="autonomy_write_task",
               tool_input={"id": 68, field: value})
    assert d.allowed is False, f"{field} passed ungated"


def test_a_run_record_note_is_not_a_schedule_write(store):
    """The benign half, and the reason the gate keys on fields rather than on
    target id.

    Every unattended `autonomy_write_task` call that touched no gated field, in
    the dispatches counted on #724, took an id and appended a note — the shape
    `skills/sandbox-permission-errors/SKILL.md` prescribes as the conflict-free
    update ("use `activity_note` only"). So an id-keyed gate ('deny if the
    target is the caller's own task') would have refused those, while every
    armed-and-parked write — which named a *different* task — passed. The
    field-set predicate is what makes the two cases come out the right way round.
    """
    d = _check(store, scope="autonomy-task:68", tool="autonomy_write_task",
               tool_input=RUN_RECORD_CALL)
    assert d.allowed is True


def test_creating_a_task_in_a_dispatchable_state_is_a_schedule_write(store):
    """No `id` means create, and create takes a `status` — so the same hole
    reopens one edit wider if the gate looks only at updates. A new task in
    `up_next` is dispatched exactly like an existing one moved to `up_next`."""
    d = _check(store, scope="worker:autotriage", tool="autonomy_write_task",
               tool_input={"name": "Nightly thing", "status": "up_next"})
    assert d.allowed is False
    assert "target #new" in d.reason


def test_creating_a_task_inert_stays_open(store):
    """The positive control for the case above: a create with no dispatchable
    status is inert — `agent_mcp/autonomy.py` defaults it to `draft`, which
    dispatch does not run — so the pipeline-dispatch route an unattended turn
    uses to hand work to the queue survives."""
    d = _check(store, scope="worker:autotriage", tool="autonomy_write_task",
               tool_input={"name": "Nightly thing", "frequency": "daily"})
    assert d.allowed is True


def test_the_real_noop_is_an_empty_value_the_handler_drops(store):
    """An update whose value is empty writes nothing, so denying it would refuse
    a call that cannot change dispatch.

    The mechanism is `_handle_write`'s update loop in `agent_mcp/autonomy.py`:
    each update field is copied only `if params.get(key)`, so `status: ""` never
    reaches `_update_task_field` and the task file keeps its old value. The tier
    demotion is what stops the gate denying a write the handler is about to
    discard — the same 'empty means no dispatch-affecting change' rule the
    create path uses one branch earlier, where an empty status falls back through
    `params.get("status", "") or "draft"` and dispatch runs only `up_next`.
    """
    assert effective_tier("autonomy_write_task",
                          {"id": 84, "status": "", "scheduled_at": ""}) == 1
    d = _check(store, scope="autonomy-task:84", tool="autonomy_write_task",
               tool_input={"id": 84, "status": ""})
    assert d.allowed is True


def test_a_resent_status_is_gated_because_the_gate_reads_the_call(store):
    """No disk lookup. A worker turn can genuinely mean to park a task, or to
    arm one — the dispatches counted on #724 include an automod implement round
    writing `status: draft` to #85 and a later one writing `up_next` to the same
    task — so the gate decides from the argument set and lets a human's grant
    answer the question, rather than guessing from a task file the caller may not
    own.
    Carrying the current value in to slip a real change past a file comparison
    is the attack that would make such a comparison useless, so a resent status
    is gated too: the check never consults disk, so it cannot be stale, and the
    human's grant is the only thing that opens it."""
    assert effective_tier("autonomy_write_task",
                          {"id": 85, "status": "draft"}) == 2
    d = _check(store, scope="worker:autocode", tool="autonomy_write_task",
               tool_input={"id": 85, "status": "draft"})
    assert d.allowed is False


def test_a_field_that_only_times_the_run_is_still_a_schedule_write(store):
    """`scheduled_at` and `auto_advance` are dispatch-affecting even though they
    leave `status` alone: one re-times a run, the other decides whether a
    completion silently re-arms the task. One of the measured writes touched
    only `scheduled_at`, and that is what moved task #84's next dispatch."""
    assert effective_tier("autonomy_write_task",
                          {"id": 84, "scheduled_at": "2026-09-10T02:00:00Z"}) == 2
    assert effective_tier("autonomy_write_task",
                          {"id": 84, "auto_advance": False}) == 2
    # The nightly chain's order lives only in `preferred_hours`, so a write to
    # it re-orders #38 -> #42 -> #39 -> #40 without touching any status.
    assert effective_tier("autonomy_write_task",
                          {"id": 39, "preferred_hours": [2, 3, 4]}) == 2


def test_a_description_edit_is_not_a_schedule_write(store):
    assert effective_tier("autonomy_write_task",
                          {"id": 68, "description": "reworded"}) == 1


def test_declared_grant_reopens_the_schedule_write(aut, store, monkeypatch):
    """The fix is a gate plus an explicit grant, not a gate alone: the nightly
    #40 → #68/#85 re-arm keeps working because the human wrote it into the task
    file, not because nothing was checking."""
    _write_task(aut, 40, grants=[{"tool": "autonomy_write_task",
                                  "expires_at": _iso(24), "issued_by": "alan"}])
    task = aut._all_runnable_tasks()[0]
    policy.sync_task_grants(store, task_id=40, scope="autonomy-task:40",
                            grants=task["grants"], now=NOW)
    d = _check(store, scope="autonomy-task:40", tool="autonomy_write_task",
               tool_input={"id": 68, "status": "up_next"})
    assert d.allowed is True, d.reason


def test_schedule_denial_is_ledged_on_the_decision_surface(store):
    """Clause 1's last half: the denial has to be countable after the fact,
    because the only proof a promotion like this cost nothing is a window with
    zero new `denied` rows for the nightly chain."""
    _check(store, scope="autonomy-task:68", tool="autonomy_write_task",
           tool_input=SCHED_CALL)
    _check(store, scope="autonomy-task:68", tool="autonomy_write_task",
           tool_input=RUN_RECORD_CALL)
    rows = [r for r in store.dispatch_rows()
            if r["tool"] == "autonomy_write_task"]
    assert len(rows) == 1
    assert rows[0]["decision"] == "deny"
    assert rows[0]["scope"] == "autonomy-task:68"
    assert "target #68" in rows[0]["reason"]


def test_the_gate_lives_in_policy_not_in_the_prompt():
    """Prompt-independence, the same reasoning as the module-level tests above:
    the check is in `check_grants`, which every one of the four call sites
    reaches through the same hook, so no turn talks its way past it and no
    prompt edit turns it off."""
    src = (LLOYD_HOME / "app" / "harness" / "policy.py").read_text()
    assert "SCHEDULE_STATE_TOOL" in src
    assert "changes_schedule_state" in src
    assert "system_prompt" not in src


# ── #2093: the shipped #40 file declares NO re-arm authority, read from the
#      scheduler's own dir ─────────────────────────────────────────────────────
#
# This section is the deliberate reversal of what #724 clause 4 shipped. That
# clause required `40-nightly-reflection-config.md` to carry a `grants:` block
# so the nightly could re-arm #68 and #85 (repo `5b1c62db`, vault `160cbfa3`).
# Both targets are gone — #68 sits at `status: draft` under Alan's 2026-09-17
# do-not-restore ruling, and #85 was retired by vault `bf363373` — and
# `skills/nightly-reflection-config/SKILL.md` has no re-arm step, so #2093
# deleted the block. What is left standing here is the same measurement run in
# the opposite direction: the file parses and keeps its dispatch fields, and the
# schedule write the block used to pay for is now denied and ledgered.


def _real_task_40() -> Path:
    """The real `autonomy/40-*.md`, or a failure that names the lost clause.

    `LLOYD_OBSIDIAN_VAULT` is the override `tests/board_presence.py` honours,
    read per call so a caller's `monkeypatch.setenv` moves it. Failing on an
    absent file — never skipping — is that file's policy, and the reason it
    transfers: #2093's claim is about that exact file, so a missing file is a
    missing measurement, and a skip would report the authority as gone on the
    strength of never having looked.
    """
    import os

    raw = os.environ.get("LLOYD_OBSIDIAN_VAULT")
    vault = Path(raw).expanduser() if raw else Path.home() / "obsidian"
    board = vault / "autonomy"
    hits = sorted(board.glob("40-*.md"))
    if not hits:
        pytest.fail(f"#2093's absence claim is unpinned: no 40-*.md under {board}. "
                    "The item is that the nightly config task declares no "
                    "`grants:` block, so no file is no evidence, not a pass.")
    return hits[0]


def test_the_shipped_task_declares_no_rearm_grant_and_the_write_stays_denied(tmp_path):
    """#2093, against the file the scheduler reads: no authority, so no re-arm.

    The node that shipped here asserted the opposite — that the block in
    `40-nightly-reflection-config.md` materialised into a row and re-opened
    `{"id": 68, "status": "up_next"}`. Removing the block makes that node false,
    so it is rewritten rather than deleted, and each half of the old positive
    assertion is kept as a negative one over the same call: the file still
    parses with the fields the scheduler dispatches on, nothing validates or
    mints from it, and the same schedule write is denied and ledgered as a
    denial naming target #68.

    Why `now=NOW` here when the deleted node passed no clock: that node's point
    was an expiry, and an expiry has to be judged by the clock dispatch uses
    (#534 gives a run no way to extend its own expiry). There is no expiry left
    to judge, so the file-wide rule applies instead — every fixture derives from
    one clock, because a fixture that reads the wall clock turns red on its own
    day and blocks every automod round (#848/#853, #973). The falsification
    survives the choice either way: re-add a block and `sync_task_grants` mints
    a row under the real clock too, since the file's declared expiry is
    2026-12-31.
    """
    from app import autonomy

    path = _real_task_40()
    task = autonomy._parse_task_file(path)
    assert task is not None, f"{path.name} does not parse; the scheduler would not run it"

    # Clause 1: giving up the authority must not cost the task its own dispatch.
    assert task.get("status") == "up_next", (
        f"{path.name} is {task.get('status')!r}, not up_next. #2093 removes an "
        "authority from this task, not the task — a status that moved here is a "
        "different change, and a stopped nightly hides the rest of this node")
    assert task.get("skill_name") == "nightly-reflection-config", (
        f"{path.name} resolves skill_name={task.get('skill_name')!r}; without "
        "that name the run has no SKILL.md to follow, whatever its grants say")

    # The absence itself, stated before anything else can depend on it. Re-adding
    # a block fails here, and the message is the reason it was removed.
    assert "grants" not in task, (
        f"{path.name} declares a `grants:` block again: {task.get('grants')!r}. "
        "#2093 deleted #724 clause 4's block because both things it authorised "
        "are gone — #68 is parked by Alan's 2026-09-17 ruling and #85 was "
        "retired by vault bf363373 — so putting one back is a human decision, "
        "not drift this file can accept quietly")

    # Clause 2: a file declaring nothing yields neither specs nor errors, and
    # materialising it leaves no row behind.
    specs, errors = policy.validate_task_grants(task.get("grants"))
    assert (specs, errors) == ([], []), (
        f"a file with no `grants:` key must validate to no specs and no errors; "
        f"got specs={specs} errors={errors}")

    store = GrantStore(tmp_path / "workers.db")
    store.ensure_schema()
    minted = policy.sync_task_grants(store, task_id=40, scope="autonomy-task:40",
                                     grants=task.get("grants"), now=NOW)
    assert minted == 0, (
        f"the dispatcher put {minted} authority row(s) on the books from a file "
        "that declares none")
    assert store.live(scope="autonomy-task:40") == [], (
        "a row exists for the nightly's scope although its file declares no "
        f"grant: {store.live(scope='autonomy-task:40')}")

    # Clause 3: the write the block used to pay for is denied, and the denial
    # names the task and the field it would have moved.
    d = check_grants(store, scope="autonomy-task:40",
                     tool_name="autonomy_write_task", tool_input=SCHED_CALL,
                     now=NOW)
    assert d.allowed is False, (
        f"an unattended nightly turn can still re-arm the parked #68: {d.reason}")
    assert d.grant_id is None, (
        "denied while consuming a row — then what stopped it was the quota, not "
        "the removed authority")
    assert "target #68" in d.reason, (
        f"denied, but not naming the task the call targets: {d.reason}")

    # The denial has to be countable afterwards: this is the row a human reads
    # to learn a nightly tried to re-arm a parked task, and the ruling on #2093
    # is that such a row is the gate working, not a regression to silence.
    rows = [r for r in store.dispatch_rows() if r["tool"] == "autonomy_write_task"]
    assert len(rows) == 1, f"expected exactly one ledger row, got {rows}"
    assert rows[0]["decision"] == "deny", rows[0]
    assert rows[0]["scope"] == "autonomy-task:40", rows[0]
    assert "target #68" in rows[0]["reason"], rows[0]

    # The control that keeps the deny above from being the tool-wide stop sign:
    # in the SAME store, one minted row re-opens the identical call, so the gate
    # is authority-absent rather than closed. Clause 5 pins the same property at
    # the MCP seam; it is pinned here because this is the node whose deny would
    # otherwise pass on an empty registry.
    granted = store.mint(scope="autonomy-task:40",
                         tool_pattern="autonomy_write_task", arg_predicate="",
                         quota=None, issued_by="alan", expires_at=_iso(24.0))
    reopened = check_grants(store, scope="autonomy-task:40",
                            tool_name="autonomy_write_task",
                            tool_input=SCHED_CALL, now=NOW)
    assert reopened.allowed is True, (
        f"the gate became tool-wide: a live row no longer re-opens the write — "
        f"{reopened.reason}")
    assert reopened.grant_id == granted["id"], reopened

    # ...and the row did not leak: a scope holding nothing is still denied.
    other = check_grants(store, scope="autonomy-task:41",
                         tool_name="autonomy_write_task", tool_input=SCHED_CALL,
                         now=NOW)
    assert other.allowed is False, "a row minted for #40 leaked to another task's scope"


# ── Bash tiered by command shape (#740) ────────────────────────────────────
#
# The silent no-op this section closes: `GrantStore.mint` accepts a
# `tool='Bash'` row, and until #740 a Bash call was answered tier 1 and returned
# before the store was opened — so a human who minted that row got a grant that
# could never be consulted, and `consumed` stayed 0 forever. Clauses 4 and 5 are
# the two halves of that: the deny must fire and name what it matched, and a live
# row must actually be read and consumed.
#
# The tier NUMBER and the shape table are `tests/test_grant_bash_tier.py`; what
# lives here is the authority question — who may run a durable command — and the
# hook that is the only enforcement point on an unattended turn.

DURABLE_BASH = "supervisorctl restart agent-tts"
ORDINARY_BASH = "ls -la"


class _NeverOpenedStore(GrantStore):
    """A real store that fails the test if the tier check reaches it.

    Clause 5 asserts more than an allow: for a tier-1 command and for every
    interactive call the store must not be opened at all, because the gate's cost
    model is that an ordinary command pays no sqlite read. A flag on a fake would
    prove the fake; subclassing the real store proves `check_grants`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.touched = []

    def _connect(self):
        self.touched.append("connect")
        return super()._connect()


def _bash(store, command, *, scope="autonomy-task:39"):
    return _check(store, scope=scope, tool="Bash",
                  tool_input={"command": command})


def test_a_durable_bash_command_from_an_unattended_scope_is_denied(store):
    """Clause 4. The same call that triage proved allowed, four shapes and all."""
    d = _bash(store, DURABLE_BASH)
    assert d.allowed is False
    assert d.grant_id is None


@pytest.mark.parametrize("command", [
    "supervisorctl restart agent-tts",
    "git push origin main",
    "curl -X POST https://api.vendor/v1/deploy",
    'python -m scripts.automod.round restart --only lloyd-backend --reason "x"',
])
def test_the_deny_reason_names_the_matched_shape_and_shows_the_mint_shape(command):
    """Clause 4's informative half, on all four shapes the item names.

    `email_send`'s deny is actionable from the tool name alone. `Bash` is not:
    the same name ran `ls` one call earlier, so whoever mints the grant has to
    see WHICH shape this call matched, or the decision is made blind — and the
    mint line still has to be there, because that line is what makes issuance a
    batched renewal rather than a live interruption (#534's design).
    """
    with tempfile.TemporaryDirectory() as tmp:
        store = GrantStore(Path(tmp) / "workers.db")
        store.ensure_schema()
        d = _bash(store, command)
        assert d.allowed is False, command
        label = safety.match_durable_external(command)
        assert label and label in d.reason, (command, d.reason)
        for fragment in ("grant_create", "scope", "Bash", "expires"):
            assert fragment in d.reason, (command, fragment, d.reason)


def test_the_bash_deny_discloses_the_grant_s_breadth():
    """One live `tool='Bash'` row authorises every durable-shaped command in the
    scope, because `_PREDICATE_RE` bounds lengths and integers and cannot express
    "only this shape". That is Alan's open decision on #740, and until it is made
    the deny text has to say it — a grant minted on the strength of one denied
    `git push` silently covers a service restart in the same scope.
    """
    with tempfile.TemporaryDirectory() as tmp:
        store = GrantStore(Path(tmp) / "workers.db")
        store.ensure_schema()
        d = _bash(store, "git push origin main")
        assert "EVERY" in d.reason and "predicate" in d.reason, d.reason


def test_a_live_bash_grant_is_consulted_and_consumed(store):
    """Clause 5's positive half, and the exact probe the item says to re-run:
    `allowed=True`, a `grant_id` returned, `consumed` incremented.

    Before this change the same call returned `allowed=True` with
    `grant=None, consumed=0` — indistinguishable from having no grant at all,
    which is what made the minted row false assurance rather than a broken
    control."""
    g = _mint(store, tool="Bash")
    d = _bash(store, "git push origin main")
    assert d.allowed is True, d.reason
    assert d.grant_id == g["id"]
    assert store.get(g["id"])["consumed"] == 1
    assert [(r["decision"], r["grant_id"]) for r in store.dispatch_rows()] == [
        ("allow", g["id"])]


def test_one_bash_grant_covers_each_durable_shape_in_its_scope(store):
    """The breadth, pinned as behaviour rather than argued: one row, three
    different shapes, three consumes. A narrower row would need a predicate the
    mini-language does not have — which is why the deny text discloses it."""
    g = _mint(store, tool="Bash")
    for command in ("git push origin main", DURABLE_BASH,
                    "curl -X POST https://api.vendor/v1/deploy"):
        assert _bash(store, command).allowed is True, command
    assert store.get(g["id"])["consumed"] == 3


def test_a_tier1_bash_command_never_reaches_the_store(tmp_path):
    """Clause 5's negative half, measured rather than inferred from `allowed`:
    `ls -la` must not cost a store read, from an unattended scope or any other."""
    store = _NeverOpenedStore(tmp_path / "workers.db")
    store.ensure_schema()
    store.touched.clear()        # schema creation is setup, not the measured call
    for scope in ("autonomy-task:39", "worker:scheduled-task", "interactive"):
        d = _bash(store, ORDINARY_BASH, scope=scope)
        assert d.allowed is True and d.grant_id is None, scope
    assert store.touched == [], "tier-1 Bash opened the grant store"


def test_an_interactive_bash_call_keeps_the_tier1_passthrough(tmp_path):
    """Every Bash call from an interactive scope stays passthrough — with the
    store closed, not merely with an allow.

    A chat turn may restart a service, push a branch and POST to a vendor API
    exactly as it did before #740; the tier exists to gate unattended scopes, and
    making a human's own turn pay a grant check it cannot fail would be friction
    with no authority content."""
    store = _NeverOpenedStore(tmp_path / "workers.db")
    store.ensure_schema()
    store.touched.clear()        # schema creation is setup, not the measured call
    for command in (DURABLE_BASH, "git push origin main",
                    "curl -X POST https://api.vendor/v1/deploy"):
        d = _bash(store, command, scope="interactive")
        assert d.allowed is True and d.grant_id is None, command
    assert store.touched == [], "an interactive Bash call opened the grant store"


def test_hook_denies_a_durable_bash_command_from_a_worker_scope(store, monkeypatch):
    """The seam the fleet actually runs through: `workers/sources/_common.py`
    installs this hook, and the callback's own tier short-circuit is where a
    name-level read used to return before `check_grants` ever saw the command."""
    monkeypatch.setenv("LLOYD_GRANT_DB", str(store.db_path))
    hooks = HookRegistry()
    install_policy_hook(hooks, store=store, scope="worker:scheduled-task", now=NOW)
    out = _fire(hooks, "Bash", {"command": DURABLE_BASH})
    assert _denied(out)
    assert "supervisorctl" in (
        out["hookSpecificOutput"]["permissionDecisionReason"])


def test_hook_allows_a_durable_bash_command_on_a_live_grant(store, monkeypatch):
    """The seam's positive half, at the same instant the hook is installed:
    allowed, and the row consumed — the unattended path can still do real
    durable work when a person said it may."""
    monkeypatch.setenv("LLOYD_GRANT_DB", str(store.db_path))
    g = _mint(store, scope="worker:scheduled-task", tool="Bash")
    hooks = HookRegistry()
    install_policy_hook(hooks, store=store, scope="worker:scheduled-task", now=NOW)
    assert not _denied(_fire(hooks, "Bash", {"command": DURABLE_BASH}))
    assert store.get(g["id"])["consumed"] == 1


def test_hook_passes_an_ordinary_bash_command_through(store, monkeypatch):
    """A worker still runs `ls`: the hook must not become a Bash gate in general.
    Asserted with a store that has no Bash row, so the allow is the tier and not
    a grant."""
    monkeypatch.setenv("LLOYD_GRANT_DB", str(store.db_path))
    hooks = HookRegistry()
    install_policy_hook(hooks, store=store, scope="worker:scheduled-task", now=NOW)
    assert not _denied(_fire(hooks, "Bash", {"command": ORDINARY_BASH}))
    assert store.dispatch_rows() == []


def test_a_minted_bash_grant_is_no_longer_a_silent_no_op(store):
    """The row that used to be inert is now load-bearing, so the guard that made
    it safe to mint one is what a reviewer would want to see: an expired row
    denies. `mint` still will not refuse a `Bash` pattern — refusing it was the
    alternative to making the row live, and making it live is the fix."""
    g = _mint(store, tool="Bash", expires=-1.0)          # expired before NOW
    d = _bash(store, DURABLE_BASH)
    assert d.allowed is False
    assert store.get(g["id"])["consumed"] == 0, "an expired grant was consumed"


def test_a_tier2_bash_grant_does_not_leak_to_another_scope(store):
    """A `Bash` grant is scoped the way every other grant is: task 39's row does
    not authorise task 40's restart."""
    _mint(store, scope="autonomy-task:39", tool="Bash")
    d = _bash(store, DURABLE_BASH, scope="autonomy-task:40")
    assert d.allowed is False


# ── #2313: a grant the task stopped declaring is withdrawn ──────────────────
#
# Everything #1949, #2021 and #2023 above settle is what happens to a row the
# file STILL declares. None of them settled the other direction, and the answer
# that came back from the live store is that nothing revoked it: `sync_task_grants`
# iterated its `specs` and returned, so deleting a `grants:` block withdrew no
# row. #2093 deleted `40-nightly-reflection-config.md`'s block (vault `420a60b5`)
# and `authority_grants` id 1 — scope `autonomy-task:40`, tool
# `autonomy_write_task`, `quota NULL`, expiring 2026-12-31, `minted_by
# frontmatter:40` — stayed live, still answering `check_grants` for
# `{"id": 68, "status": "up_next"}`: the re-arm of the task Alan's 2026-09-17
# ruling keeps parked. The nodes below are that row's shape at unit scale. The
# dispatch seam that reaches them on a real nightly — `run_task`, whose
# `if _specs:` used to skip this call for a task declaring nothing — is pinned in
# `tests/test_autonomy_frontmatter_normalization.py`.

FM40_SCOPE = "autonomy-task:40"
FM40_TOOL = "autonomy_write_task"


def _frontmatter_row(store, *, task=40, scope=None, tool=FM40_TOOL,
                     predicate="", expires=+24.0):
    """A live row of the shape `sync_task_grants` writes for task `task`.

    Built with `store.mint` and the file's own `minted_by` rather than through a
    sync call: a node about withdrawing a row must not also depend on the mint
    path being upright, or a broken mint turns these red for the wrong reason.
    """
    return store.mint(scope=scope or f"autonomy-task:{task}",
                      tool_pattern=tool, arg_predicate=predicate, quota=None,
                      issued_by="alan", expires_at=_iso(expires),
                      minted_by=f"frontmatter:{task}")


def _grant_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.name == "lloyd-harness-policy"
            and r.levelno >= logging.WARNING]


def test_an_undeclared_grants_block_withdraws_the_row_it_minted(store):
    """#2313 clause 1: the file saying nothing takes the authority back.

    `grants: []` and no `grants:` key at all are the same statement — the
    scheduler reads an absent key as `None` and `validate_task_grants` turns both
    into no specs — so both are run here, and both leave the scope with nothing
    live. The denial that follows is the one the item's acceptance check measures
    against the real store: the same call for the same target #68.
    """
    row = _frontmatter_row(store)

    assert policy.sync_task_grants(store, task_id=40, scope=FM40_SCOPE,
                                   grants=[], now=NOW) == 0
    after = store.get(row["id"])
    assert after["revoked_at"] == _iso(0), (
        f"the row a file stopped declaring is still unrevoked: {after}")
    assert store.live(scope=FM40_SCOPE, now=NOW) == [], (
        "the withdrawn row is still live for its own scope")

    # The absence rather than the empty list: what the shipped #40 file actually
    # is, read the way `_parse_task_file` hands it over — no `grants` key.
    assert policy.sync_task_grants(store, task_id=40, scope=FM40_SCOPE,
                                   grants=None, now=NOW) == 0, (
        "a task with no `grants:` key at all still mints or still raises; the "
        "nightly whose block was deleted is exactly this shape")
    assert store.get(row["id"])["revoked_at"] == _iso(0), (
        "a second undeclared sync un-revoked the row")

    d = check_grants(store, scope=FM40_SCOPE, tool_name=FM40_TOOL,
                     tool_input=SCHED_CALL, now=NOW, record=False)
    assert d.allowed is False, (
        f"the withdrawn row still pays for the parked re-arm: {d.reason}")
    assert "target #68" in d.reason, d.reason


def test_the_withdrawal_is_announced_once_naming_the_row_and_its_tool(
        store, caplog):
    """#2313 clause 2: no authority leaves the books in silence.

    #2021 and #2023 each added a warning because a grant that changed state
    quietly read as a grant that did not change; a withdrawal that only showed up
    as a later denial is the same class of defect. One line, naming the grant id,
    the tool and the scope — and a second sync over the same dead row says
    nothing new, because `revoke()` matches `revoked_at IS NULL` and a nightly
    that fires every day must not re-announce one withdrawal 365 times.
    """
    row = _frontmatter_row(store)
    caplog.set_level(logging.WARNING, logger="lloyd-harness-policy")
    caplog.clear()

    policy.sync_task_grants(store, task_id=40, scope=FM40_SCOPE, grants=[],
                            now=NOW)

    warns = _grant_warnings(caplog)
    assert len(warns) == 1, f"{len(warns)} warnings for one withdrawal: {warns}"
    line = warns[0]
    assert f"grant #{row['id']}" in line, line
    assert FM40_TOOL in line, f"the withdrawn row's tool is not named: {line}"
    assert "task #40" in line and FM40_SCOPE in line, line
    assert "no longer declares" in line, line

    caplog.clear()
    policy.sync_task_grants(store, task_id=40, scope=FM40_SCOPE, grants=[],
                            now=NOW)
    assert _grant_warnings(caplog) == [], (
        "the same withdrawal announced twice; a daily task re-runs this sync "
        "every day and the second line is noise that hides the first")


def test_withdrawal_takes_only_this_files_own_row_and_by_id_not_prefix(store):
    """#2313 clause 4: the key is equality with `frontmatter:<task_id>`.

    Three rows a withdrawal must leave standing, in the one store where the
    pass runs:

    * a HUMAN row for the very pair being withdrawn (`grant_create` writes
      `minted_by='human'`) — without the minter filter the empty block would
      revoke a human's grant, which is the one authority this design says a file
      may not touch;
    * a WORKER-minted row, the class `count_worker_minted` exists to hold at
      zero; its `minted_by` is written straight in, because `store.mint` refuses
      a worker identity at mint time (policy's own rule). It is asserted off the
      row itself and not off that counter: `count_worker_minted` selects only
      `minted_by` and then reads `r["issued_by"]` (app/harness/policy.py:686),
      which raises `IndexError` on any store holding a non-worker row — recorded
      on #2313, out of this round's scope;
    * a row in TASK 4's OWN SCOPE whose `minted_by` names task 40 — the shape a
      `startswith(FRONTMATTER_MINTER)` implementation would withdraw, since
      `frontmatter:4` is a prefix of `frontmatter:40`. Task 4's sync withdraws
      only `frontmatter:4`, which is the sibling row minted below it.
    """
    human = _mint(store, scope=FM40_SCOPE, tool=FM40_TOOL)
    worker = store.mint(scope=FM40_SCOPE, tool_pattern="calendar_create",
                        arg_predicate="", quota=None, issued_by="alan",
                        expires_at=_iso(24.0))
    with store._connect() as conn:
        conn.execute("UPDATE authority_grants SET minted_by=? WHERE id=?",
                     ("worker:scheduled-task", worker["id"]))
    neighbour_in_task4_scope = _frontmatter_row(store, task=40,
                                                scope="autonomy-task:4")
    own = _frontmatter_row(store, task=4)
    # The human and worker rows go into TASK 4's own scope, alongside the row
    # that sync does withdraw. In `autonomy-task:40` they would be shielded by
    # `store.live()`'s scope filter no matter what the minter test does, and an
    # assertion that survives every implementation measures nothing (the
    # review's finding on the first attempt at this round).
    human_in_task4_scope = _mint(store, scope="autonomy-task:4",
                                 tool=FM40_TOOL)
    worker_in_task4_scope = store.mint(scope="autonomy-task:4",
                                       tool_pattern="calendar_create",
                                       arg_predicate="", quota=None,
                                       issued_by="alan", expires_at=_iso(24.0))
    with store._connect() as conn:
        conn.execute("UPDATE authority_grants SET minted_by=? WHERE id=?",
                     ("worker:scheduled-task", worker_in_task4_scope["id"]))

    policy.sync_task_grants(store, task_id=4, scope="autonomy-task:4",
                            grants=[], now=NOW)

    assert store.get(own["id"])["revoked_at"] == _iso(0), (
        "task 4's own row survived its own withdrawal — the pass would be a "
        "no-op, and every row-sharing-assertion below it would be unmeasured")
    assert store.get(human_in_task4_scope["id"])["revoked_at"] is None, (
        "a human's grant for the same (scope, tool) was revoked by a task file "
        "editing itself — that is only possible if the pass dropped its "
        "`minted_by` test, which is what this assert is for")
    assert store.get(worker_in_task4_scope["id"])["revoked_at"] is None, (
        "a worker-minted row in the withdrawing scope was revoked by a task "
        "file, so the pass is not testing the minter at all")
    assert store.get(worker_in_task4_scope["id"])["minted_by"] == \
        "worker:scheduled-task", (
        "the row this pass stepped over is not the worker-shaped row the "
        "fixture wrote, so the assertion above compared nothing")
    assert store.get(neighbour_in_task4_scope["id"])["revoked_at"] is None, (
        "task 4 withdrew task 40's row: `frontmatter:4` is a prefix of "
        "`frontmatter:40`, so this is the startswith bug")
    # The same two shapes in task 40's own scope, where nothing is withdrawn by
    # task 4's pass: they are here so the pair above is not the only copy of the
    # guarantee, and so a `scope`-less pass still has to answer for them.
    assert store.get(human["id"])["revoked_at"] is None, (
        "a human row was revoked by a sync over a different scope")
    assert store.get(worker["id"])["revoked_at"] is None, (
        "a worker row in another scope was revoked by task 4's sync")


def test_a_grants_block_the_loader_cannot_read_costs_no_row(store, aut,
                                                            caplog):
    """#2313 clause 5: a parse error must not spend the human's authority.

    The withdrawal is a decision about what the file declares, so it can only
    run on a file that was understood. `sync_task_grants` raises `GrantError` on
    validation errors before its mint loop and before the withdrawal below it,
    and `run_task` refuses the run on the same errors (`app/autonomy.py`,
    "refusing to run ungated") — the fail-closed pair #534 established. The
    hazard this pins is the implementation that withdrew first and validated
    second: `grants: email_send please` would then revoke the row a previous,
    readable block minted, and the task would be both stopped AND disarmed.
    """
    row = _frontmatter_row(store)
    caplog.set_level(logging.WARNING, logger="lloyd-harness-policy")
    caplog.clear()

    bad = [{"tool": FM40_TOOL, "expires_at": _iso(24.0)}]  # no issued_by
    specs, errors = policy.validate_task_grants(bad)
    assert errors and not specs, (specs, errors)
    with pytest.raises(GrantError):
        policy.sync_task_grants(store, task_id=40, scope=FM40_SCOPE,
                                grants=bad, now=NOW)
    assert store.get(row["id"])["revoked_at"] is None, (
        "an unreadable block withdrew the row its readable predecessor minted")
    assert store.live(scope=FM40_SCOPE, now=NOW), (
        "the scope lost its authority to a typo")
    assert _grant_warnings(caplog) == [], (
        "a call that did not run still announced something")

    # The same block at the loader: not runnable, and no sync attempted.
    _write_task(aut, 40, grants="email_send please")
    task = aut._parse_task_file(aut.AUTONOMY_DIR / "40-task.md")
    assert aut._grant_block_errors(task, aut.AUTONOMY_DIR / "40-task.md"), (
        "the loader reported no error for a block `validate_task_grants` "
        "rejects, so the two surfaces disagree about what is readable")
    assert aut._all_runnable_tasks() == [], (
        "a task with an unreadable block is runnable, so run_task would reach "
        "the withdrawal anyway")
    assert store.get(row["id"])["revoked_at"] is None


def test_the_withdrawal_is_visible_to_the_store_the_hook_builds(tmp_path):
    """The hook's read-after-write, across two `GrantStore` objects on one file.

    The sync runs in the scheduler (`app/autonomy.py`, at dispatch); the
    pre-tool-use hook reads authority through a store it builds itself
    (`default_store()`, policy.py:693). Both are named in this round's review as
    the seam that has to hold, and one `store` fixture cannot show it: with a
    single object the read is the same object's write. Two instances over one
    path is the shape the two processes actually have — the store is stateless
    per call apart from `_schema_applied`, so what is being proved here is the
    commit landing in the file, not a cache being warm.
    """
    db = tmp_path / "workers.db"
    writer = GrantStore(db)
    writer.ensure_schema()
    reader = GrantStore(db)          # the hook's own object, opened before
    reader.ensure_schema()

    _frontmatter_row(writer)
    assert check_grants(reader, scope=FM40_SCOPE, tool_name=FM40_TOOL,
                        tool_input=SCHED_CALL, now=NOW, record=False).allowed, (
        "the reader never allowed the call in the first place, so the denial "
        "below would prove nothing")

    policy.sync_task_grants(writer, task_id=40, scope=FM40_SCOPE, grants=[],
                            now=NOW)

    d = check_grants(reader, scope=FM40_SCOPE, tool_name=FM40_TOOL,
                     tool_input=SCHED_CALL, now=NOW, record=False)
    assert d.allowed is False, (
        "the sync revoked the row and the hook's store still pays for the "
        f"call: {d.reason}")
    assert "target #68" in d.reason, d.reason
    assert d.grant_id is None, (
        f"the denial still names a paying row: grant #{d.grant_id}")
