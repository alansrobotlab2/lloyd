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

One case is deliberately not hermetic: the last one reads the shipped
`autonomy/40-*.md` from the real vault, because clause 4 of #724 is a claim
about that file and a fixture cannot evidence it. Its rationale, and why it
fails rather than skips, are at its own docstring.

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


# ── Clause 4: the shipped nightly re-arm, read from the scheduler's own dir ──


def _real_task_40() -> Path:
    """The real `autonomy/40-*.md`, or a failure that names the lost clause.

    `LLOYD_OBSIDIAN_VAULT` is the override `tests/board_presence.py` honours,
    read per call so a caller's `monkeypatch.setenv` moves it. Failing on an
    absent file — never skipping — is that file's policy, and the reason it
    transfers: this task's `grants:` block *is* the authorisation the nightly
    re-arm runs on, so no file means no authorisation, not no measurement.
    """
    import os

    raw = os.environ.get("LLOYD_OBSIDIAN_VAULT")
    vault = Path(raw).expanduser() if raw else Path.home() / "obsidian"
    board = vault / "autonomy"
    hits = sorted(board.glob("40-*.md"))
    if not hits:
        pytest.fail(f"clause 4 of #724 is unpinned: no 40-*.md under {board}. "
                    "The nightly #40 → #68/#85 re-arm is meant to be authorised "
                    "by a `grants:` block in that file, so a missing file is a "
                    "missing authorisation.")
    return hits[0]


def test_the_shipped_nightly_rearm_grant_materialises_and_reopens_the_write(tmp_path):
    """The acceptance's third half, run against the file the scheduler reads.

    Every other case in this file parses a fixture, so it stays true whatever
    the shipped task file says. This one cannot: it reads the real
    `40-nightly-reflection-config.md` through the scheduler's own
    `_parse_task_file`, validates the block, materialises it with
    `sync_task_grants`, and then asks the question the clause is about — does a
    `autonomy-task:40` turn holding nothing but that declared block get to move
    #68 to `up_next`?

    No `now=` anywhere below, because dispatch passes no `now=`: the expiry is
    judged by the real clock, so a block nobody renews goes red here on the same
    day it starts denying the nightly, and `#534` deliberately gives the run no
    way to extend it.
    """
    import autonomy

    path = _real_task_40()
    task = autonomy._parse_task_file(path)
    assert task is not None, f"{path.name} does not parse; the scheduler would not run it"
    specs, errors = policy.validate_task_grants(task.get("grants"))
    assert errors == [], f"the shipped grants block is not acceptable: {errors}"
    assert [s["tool"] for s in specs] == ["autonomy_write_task"], (
        "clause 4 names exactly one authority; any other entry is a scope call "
        "a human has to make, not drift this file can accept quietly")
    assert specs[0]["expires_at"] > dt.datetime.now(dt.timezone.utc), (
        f"{path.name}'s grant expired on "
        f"{specs[0]['expires_at'].isoformat()}; renewal is a human editing the "
        "block, and until then the nightly re-arm is denied, as it should be")

    store = GrantStore(tmp_path / "workers.db")
    store.ensure_schema()
    assert policy.sync_task_grants(store, task_id=40, scope="autonomy-task:40",
                                   grants=task["grants"]) >= 1

    d = check_grants(store, scope="autonomy-task:40",
                     tool_name="autonomy_write_task", tool_input=SCHED_CALL)
    assert d.allowed is True, f"the shipped block does not re-open the write: {d.reason}"
    assert d.grant_id is not None, ("allowed without consuming a row means the "
                                    "gate was absent, not satisfied")
    # The control: the allow came from #40's row, not from the tool being open.
    other = check_grants(store, scope="autonomy-task:41",
                         tool_name="autonomy_write_task", tool_input=SCHED_CALL)
    assert other.allowed is False, "a declared grant leaked to another task's scope"
