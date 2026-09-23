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
            " task_id, scope, reason, at FROM egress_events ORDER BY id"))
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
