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

Run:
  /home/alansrobotlab/lloyd/.venvs/lloyd/bin/python -m pytest tests/unit/test_grant_policy.py
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import sys
from pathlib import Path

import pytest

LLOYD_HOME = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LLOYD_HOME))

from app.harness import HookRegistry  # noqa: E402
from app.harness import policy  # noqa: E402
from app.harness.policy import (  # noqa: E402
    GRANT_MINT_TOOL,
    GrantError,
    GrantStore,
    check_grants,
    install_policy_hook,
    tool_tier,
)

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
    return store.mint(
        scope=scope, tool_pattern=tool, arg_predicate=predicate,
        quota=quota, issued_by=issued_by, expires_at=_iso(expires),
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
    import prompt_builder
    monkeypatch.setattr(prompt_builder, "build_system_prompt", lambda *a, **k: "sys")
    import autonomy
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
    install_policy_hook(hooks, store=store, scope="worker:scheduled-task")
    assert _denied(_fire(hooks, "email_send", {"to": "x@y.z"}))


def test_hook_allows_granted_call(store, monkeypatch):
    monkeypatch.setenv("LLOYD_GRANT_DB", str(store.db_path))
    _mint(store, scope="worker:scheduled-task")
    hooks = HookRegistry()
    install_policy_hook(hooks, store=store, scope="worker:scheduled-task")
    assert not _denied(_fire(hooks, "email_send", {"to": "x@y.z"}))


def test_hook_passes_tier1_through(store):
    hooks = HookRegistry()
    install_policy_hook(hooks, store=store, scope="worker:scheduled-task")
    assert not _denied(_fire(hooks, "vault_write", {"path": "a.md"}))


def test_store_failure_fails_closed(store, tmp_path):
    """`HookRegistry.fire_pre_tool_use` treats a raising callback as a pass,
    so a grant gate that crashed would silently open the gate. The callback
    must therefore deny on its own errors rather than raise."""
    broken = GrantStore(tmp_path / "gone" / ".." / "nope.db")
    hooks = HookRegistry()
    install_policy_hook(hooks, store=broken, scope="worker:scheduled-task")
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

    import autonomy
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
    import autonomy
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
