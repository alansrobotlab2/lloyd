"""#628 clauses 3, 4 and 5: the egress policy's refusals, its floor, and the
grant store's destination axis.

Clause 3 — with `harness.egress_policy.enforce` on (its default is off) and no
allow entry or live grant covering a host, `http_fetch` and `browser_navigate`
refuse with **no connection attempted**, and the refusal text carries the exact
mintable `grant_create` row; an attended scope is never refused, because a person
has to be able to mint what a denial asks for.

Clause 4 — listing `127.0.0.1`, `::1`, `192.168.1.1` or `169.254.1.1` in the
allow-list still denies, the bare `::1` literal reaches the private-host block
rather than an "unparsable destination" error, and `http_request`'s existing
loopback allowance is unchanged.

Clause 5 — `authority_grants` can name a host, `check_grants` honours that axis,
and a grant for one host does not authorize a fetch to a different one.

The two process boundaries this change crosses get a test each:

* **harness → aggregator.** `app/harness/policy.check_grants` runs in the
  backend, `agent_mcp/egress.guard` runs in the `agent-stdio` child process, and
  they communicate the caller's identity only as the `lloyd/effect_scope`
  `_meta` string. `test_the_effect_scope_string_resolves_to_the_same_grant_scope_
  the_pool_would` runs the real chain on a shared sqlite file.
* **worker → queue database.** `agent_mcp/egress` opens `workers.db` directly
  rather than calling the queue over a seam; `test_a_pre_628_database_is_migrated_
  before_the_store_reads_it` proves the table's shape arrives before either
  reader asks for it, which is the failure mode of a second writer.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import inspect
import ipaddress
import json
import re
import socket
import sqlite3
import subprocess
from pathlib import Path

import pytest

from agent_mcp import egress
from agent_mcp import browser as browser_module
from agent_mcp import http_tools
from app.harness import policy
from app.harness.policy import (GrantError, GrantStore, check_grants,
                                grant_shape, normalize_destination,
                                destination_covers)


#: The wall clock, not a fixed date: `egress.guard` reads the real clock, so a grant
#: minted `NOW + 1h` against a hard-coded date expired the day after it was written.
NOW = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
UNKNOWN_HOST = "collector.evil.example"
UNKNOWN_URL = f"https://{UNKNOWN_HOST}/exfil"
LISTED_HOST = "docs.example.org"
LISTED_URL = f"https://{LISTED_HOST}/guide"
WORKER_SCOPE = "worker:domain-research"


# ── the harness: config, socket stubs, store ────────────────────────────────

@pytest.fixture
def db(tmp_path, monkeypatch):
    """One scratch file as BOTH the telemetry table's home and the grant store's.

    They are the same file on purpose — that is the arrangement in production,
    where `egress_events` is a table in `workers.db` and `authority_grants` is
    the table next to it. Both paths default to `~/lloyd/workers.db`, so a test
    that minted a destination grant without redirecting them would be writing
    authority into the live store.
    """
    path = tmp_path / "workers.db"
    monkeypatch.setenv("LLOYD_EGRESS_DB", str(path))
    monkeypatch.setenv("LLOYD_GRANT_DB", str(path))
    return path


@pytest.fixture(autouse=True)
def _unattended_worker_turn(monkeypatch):
    """Every test in this file runs as an unattended worker turn.

    With no effect scope bound the guard cannot attribute a call to a worker and
    falls back to the bare `worker` scope, which is not the string a grant is
    minted against — so the scope-consistency assertions below would be comparing
    a grant's scope to a placeholder. This is the shape the pool produces for a
    `domain-research` row: `item:<source>:<id>` in `_meta`, and
    `worker:<source>` on the child's side of the seam.
    """
    monkeypatch.setattr(egress, "_effect_scope",
                        lambda: "item:domain-research:7")


def policy_on(monkeypatch, *, allow=(), enforce=True, telemetry=True):
    """Point the policy at a state, at the seam `egress` itself reads.

    Patched on the module rather than on `CONFIG`: `app.config.CONFIG` is a
    mapping proxy that will not accept a `["harness"]` item, and `egress.config()`
    is the single accessor every flag and every allow-list read goes through.
    """
    state = {"telemetry": telemetry, "enforce": enforce, "allow": list(allow),
             "retention_days": 90}
    monkeypatch.setattr(egress, "config", lambda: dict(state))


class RaisingClient:
    """A transport that records that it was asked, then fails loudly.

    Clause 3's real assertion is the empty `calls` list: a refusal that still
    built a client is a fetch that happened to be refused late, which is the
    thing the clause exists to rule out.
    """

    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, *a, **k):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def request(self, method, url, **kwargs):
        self.calls.append(f"{method} {url}")
        raise AssertionError(f"connection attempted to {url} behind a refusal")

    def get(self, url, **kwargs):
        # `_http_fetch` calls `client.get`, `_http_request` calls `client.request`.
        return self.request("GET", url, **kwargs)


class OkClient(RaisingClient):
    def request(self, method, url, **kwargs):
        self.calls.append(f"{method} {url}")
        return http_tools.httpx.Response(
            200, request=http_tools.httpx.Request(method, url),
            headers={"content-type": "text/html"}, text="<html><body>ok</body></html>")


@pytest.fixture
def no_socket(monkeypatch):
    client = RaisingClient()
    monkeypatch.setattr(http_tools, "make_sync_http_client", client)
    return client


@pytest.fixture
def ok_socket(monkeypatch):
    client = OkClient()
    monkeypatch.setattr(http_tools, "make_sync_http_client", client)
    return client


@pytest.fixture
def no_browser(monkeypatch):
    """Swap the Playwright page seam for one that would launch Chromium."""
    calls = []

    async def _boom():
        calls.append("launch")
        raise AssertionError("Chromium was reached behind a refusal")

    monkeypatch.setattr(browser_module, "_get_page", _boom)
    return calls


def store(db) -> GrantStore:
    return GrantStore(db)


def mint(db, *, host, scope=WORKER_SCOPE, tool="http_fetch", expires_in=dt.timedelta(hours=1),
         issued_by="alan", quota=None):
    return store(db).mint(scope=scope, tool_pattern=tool, issued_by=issued_by,
                          expires_at=NOW + expires_in, destination=host,
                          quota=quota, now=NOW)


# ── clause 3: refused before a socket opens, and the row that would fix it ───

def test_an_unknown_destination_is_refused_with_no_connection_attempted(db, monkeypatch,
                                                                       no_socket):
    policy_on(monkeypatch)
    monkeypatch.setattr(egress, "_effect_scope", lambda: f"item:domain-research:99")

    out = json.loads(http_tools._http_fetch(UNKNOWN_URL))

    assert "error" in out, out
    assert "egress: refused http_fetch" in out["error"], out
    assert no_socket.calls == [], "the guard ran after a client was built"


def test_the_refusal_names_the_exact_mintable_grant_row(db, monkeypatch, no_socket):
    """The denial is a batched renewal, so it has to be copy-pasteable.

    Asserted field by field rather than by substring soup: a row that renders
    `destination` in the predicate instead of the column — or that omits the
    expiry, which `validate_predicate` would silently accept as a host string —
    would be a denial a human cannot action.
    """
    policy_on(monkeypatch)
    monkeypatch.setattr(egress, "_effect_scope", lambda: "item:domain-research:99")

    out = json.loads(http_tools._http_fetch(UNKNOWN_URL))
    text = out["error"]

    assert "grant_create(" in text, text
    for field in (f"scope='{WORKER_SCOPE}'", "tool='http_fetch'",
                  f"destination='{UNKNOWN_HOST}'"):
        assert field in text, (field, text)
    assert "expires_at='" in text, "a grant row without an expiry is not a grant"
    assert "issued_by=" in text, text


def test_an_attended_scope_is_never_refused_so_a_human_can_mint_the_grant(
        db, monkeypatch, ok_socket):
    """"Attended" is read off the session's platform, never off the arguments.

    The other half of the clause matters as much: an attended call to an unknown
    host is recorded `grant-required` rather than `allow`, so the inventory the
    human seeds the list from still shows what was actually outside policy.
    """
    policy_on(monkeypatch)
    monkeypatch.setattr(egress, "_session_is_user", lambda sid: True)

    out = json.loads(http_tools._http_fetch(LISTED_URL))

    assert "error" not in out, out
    assert ok_socket.calls == [f"GET {LISTED_URL}"], ok_socket.calls
    rows = egress.network_report()["per_destination"]
    assert [r["grant_required"] for r in rows if r["host"] == LISTED_HOST] == [1], rows


def test_the_flag_is_off_by_default_and_attended_scopes_never_see_a_refusal(db, monkeypatch,
                                                                           ok_socket):
    """Default-off enforcement, asserted at the flag rather than inferred.

    Turning enforcement on for the fleet is a human decision (the item names the
    shape choice it waits on), so the shipped default must be the one that cannot
    break a research job — while telemetry, which cannot break anything, is on.
    """
    policy_on(monkeypatch, enforce=False)
    monkeypatch.setattr(egress, "_effect_scope", lambda: "item:domain-research:99")
    assert egress.enforce_on() is False

    out = json.loads(http_tools._http_fetch(UNKNOWN_URL))

    assert "error" not in out, out
    assert ok_socket.calls == [f"GET {UNKNOWN_URL}"]
    report = egress.network_report()
    assert report["policy"]["enforce"] is False
    assert report["policy"]["telemetry"] is True


def test_an_allow_entry_covering_the_host_allows_without_a_grant(db, monkeypatch, ok_socket):
    policy_on(monkeypatch, allow=[{"host": LISTED_HOST, "reason": "docs, reviewed"}])
    monkeypatch.setattr(egress, "_effect_scope", lambda: "item:domain-research:99")

    out = json.loads(http_tools._http_fetch(LISTED_URL))

    assert "error" not in out, out
    assert ok_socket.calls == [f"GET {LISTED_URL}"]


def test_a_live_destination_grant_opens_the_host_for_the_unattended_scope(
        db, monkeypatch, ok_socket):
    policy_on(monkeypatch)
    mint(db, host=LISTED_HOST)
    monkeypatch.setattr(egress, "_effect_scope", lambda: "item:domain-research:99")

    out = json.loads(http_tools._http_fetch(LISTED_URL))

    assert "error" not in out, out
    assert ok_socket.calls == [f"GET {LISTED_URL}"]


def test_browser_navigate_refuses_an_unknown_destination_without_launching_chromium(
        db, monkeypatch, no_browser):
    policy_on(monkeypatch)
    monkeypatch.setattr(egress, "_effect_scope", lambda: "item:domain-research:99")

    out = json.loads(asyncio.run(
        browser_module._browser_navigate(f"https://{UNKNOWN_HOST}/x")))

    assert "egress: refused browser_navigate" in out["error"], out
    assert no_browser == [], "the guard ran after the browser was launched"


def test_an_unreadable_grant_store_denies_rather_than_defaulting_open(db, monkeypatch,
                                                                      no_socket):
    """A guard that cannot read its own rows reporting `allow` is instance 11 of
    the catalogued class: the verdict is not something it can justify."""
    policy_on(monkeypatch)
    monkeypatch.setattr(egress, "grant_covers",
                        lambda **kw: (_ for _ in ()).throw(sqlite3.OperationalError("locked")))
    monkeypatch.setattr(egress, "_effect_scope", lambda: "item:domain-research:99")

    out = json.loads(http_tools._http_fetch(UNKNOWN_URL))

    assert "unreadable" in out["error"], out
    assert no_socket.calls == []


# ── clause 4: the floor is not lift-able, and the existing allowance stands ──

@pytest.mark.parametrize("address", ["127.0.0.1", "::1", "192.168.1.1", "169.254.1.1"])
def test_listing_a_private_address_in_the_allow_list_still_denies(address, db, monkeypatch,
                                                                  no_socket):
    """The whole point of ordering the floor before the allow-list.

    A destination policy that could be talked out of its own address block by
    adding the blocked address to the destination policy is not a boundary.
    """
    policy_on(monkeypatch, allow=[{"host": address, "reason": "oops, listed by hand"}])
    monkeypatch.setattr(egress, "_effect_scope", lambda: "item:domain-research:99")

    out = json.loads(http_tools._http_fetch(f"https://{address}/x"))

    assert "private/internal host" in out["error"], out
    assert no_socket.calls == []


def test_the_bare_ipv6_loopback_literal_reaches_the_private_host_block(db, monkeypatch,
                                                                      no_socket):
    """`http://::1/` used to be refused as an unparsable *scheme*.

    `urlparse("http://::1/").hostname` is `None`, so the check this guard
    replaced saw an empty hostname. The clause asks for the address answer, not
    the parse error, and the message has to be the private-host one exactly.
    """
    policy_on(monkeypatch)
    monkeypatch.setattr(egress, "_effect_scope", lambda: "item:domain-research:99")

    out = json.loads(http_tools._http_fetch("http://::1/"))

    assert out["error"] == 'Blocked — private/internal host "::1"', out


def test_the_bracketed_ipv6_form_asserts_the_same_message(db, monkeypatch, no_socket):
    policy_on(monkeypatch)
    monkeypatch.setattr(egress, "_effect_scope", lambda: "item:domain-research:99")

    out = json.loads(http_tools._http_fetch("https://[::1]:8443/y"))

    assert out["error"] == 'Blocked — private/internal host "::1"', out


def test_browser_navigate_still_denies_a_listed_link_local_address(db, monkeypatch, no_browser):
    policy_on(monkeypatch, allow=[{"host": "169.254.1.1", "reason": "listed by mistake"}])
    monkeypatch.setattr(egress, "_effect_scope", lambda: "item:domain-research:99")

    out = json.loads(asyncio.run(
        browser_module._browser_navigate("http://169.254.1.1/latest/meta-data")))

    assert "private/internal host" in out["error"], out
    assert no_browser == []


def test_http_request_loopback_allowance_is_unchanged(db, monkeypatch, ok_socket):
    """`http_request` to 127.0.0.1 is how this box drives its own services, and
    the destination axis must not revoke it. `http_fetch` still refuses the same
    address, which is what makes the two tools different tools."""
    policy_on(monkeypatch)
    monkeypatch.setattr(egress, "_effect_scope", lambda: "item:domain-research:99")

    out = json.loads(http_tools._http_request("GET", "http://127.0.0.1:8080/api/mc/state"))

    assert "error" not in out and out.get("status_code") == 200, out
    assert ok_socket.calls == ["GET http://127.0.0.1:8080/api/mc/state"]
    denied = json.loads(http_tools._http_fetch("http://127.0.0.1:8080/api/mc/state"))
    assert "private/internal" in denied["error"], denied


def test_a_link_local_metadata_endpoint_is_denied_by_the_floor(db, monkeypatch, no_socket):
    policy_on(monkeypatch)
    monkeypatch.setattr(egress, "_effect_scope", lambda: "item:domain-research:99")

    out = json.loads(http_tools._http_request(
        "GET", "http://169.254.169.254/latest/meta-data/iam/security-credentials"))

    assert "private/internal host" in out["error"], out
    assert no_socket.calls == []


def test_an_unparsable_destination_is_still_refused_before_the_socket(db, monkeypatch,
                                                                     no_socket):
    """A guard that answered "not on the allow-list" to a string it could not
    parse would be guessing about the one input it exists to read."""
    policy_on(monkeypatch)
    monkeypatch.setattr(egress, "_effect_scope", lambda: "item:domain-research:99")

    # An unclosed IPv6 bracket: `urlparse` raises on it, so no host can be read.
    # (`http://::1` is not this case — it names ::1, and the floor answers it.)
    verdict = egress.guard("http_fetch", "https://[::1/x")

    assert verdict.allowed is False
    assert verdict.decision == "deny"
    assert verdict.reason == 'Unparsable destination: "https://[::1/x"', verdict.reason
    assert no_socket.calls == []


# ── clause 5: the grant store can name a host ───────────────────────────────

def test_the_grant_table_can_name_a_host(db):
    store(db).mint(scope=WORKER_SCOPE, tool_pattern="http_fetch", issued_by="alan",
                   expires_at=NOW + dt.timedelta(days=1),
                   destination="docs.example.org", now=NOW)
    cols = {r[1] for r in sqlite3.connect(str(db)).execute(
        "PRAGMA table_info(authority_grants)")}
    assert "destination" in cols, cols


def test_a_grant_for_one_host_does_not_authorize_a_fetch_to_another(db):
    mint(db, host="api.example.com")

    allowed_here = check_grants(store(db), scope=WORKER_SCOPE, tool_name="http_fetch",
                                tool_input={}, destination="api.example.com",
                                now=NOW, record=False)
    denied_there = check_grants(store(db), scope=WORKER_SCOPE, tool_name="http_fetch",
                                tool_input={}, destination="attacker.tld",
                                now=NOW, record=False)

    assert allowed_here.allowed is True, allowed_here
    assert denied_there.allowed is False, denied_there
    assert f"destination='{UNKNOWN_HOST}'" not in denied_there.reason
    assert "destination='attacker.tld'" in denied_there.reason, denied_there.reason


def test_a_destination_grant_is_not_a_licence_for_the_tool_generally(db):
    """The converse direction, without which the column is decoration.

    `email_send` sharing a tool pattern with a destination grant is the case:
    authorizing POSTs to one host cannot silently authorize that tool's other
    arguments. A tier-2 tool, because a tier-1 tool without a destination never
    reaches the store at all.
    """
    assert policy.effective_tier("email_send", {}) == 2
    mint(db, host="api.example.com", tool="email_send")

    no_destination = check_grants(store(db), scope=WORKER_SCOPE, tool_name="email_send",
                                  tool_input={}, now=NOW, record=False)
    assert no_destination.allowed is False, no_destination
    with_it = check_grants(store(db), scope=WORKER_SCOPE, tool_name="email_send",
                           tool_input={}, destination="api.example.com",
                           now=NOW, record=False)
    assert with_it.allowed is True, "the control: the grant does pay for its own host"


def test_a_pre_628_grant_does_not_authorize_a_destination(db):
    """A row minted before the axis existed predates any decision about hosts.

    Read as "any host", the migration would silently globalize every grant on
    the live store the moment the guard switched on.
    """
    store(db).mint(scope=WORKER_SCOPE, tool_pattern="http_fetch", issued_by="alan",
                   expires_at=NOW + dt.timedelta(days=1), now=NOW)

    decision = check_grants(store(db), scope=WORKER_SCOPE, tool_name="http_fetch",
                            tool_input={}, destination="any.host", now=NOW, record=False)
    assert decision.allowed is False, decision
    # …and it still works for the non-network purpose it was minted for.
    plain = check_grants(store(db), scope=WORKER_SCOPE, tool_name="http_fetch",
                         tool_input={}, now=NOW, record=False)
    assert plain.allowed is True, plain


def test_the_stored_predicate_cannot_express_a_host(db):
    """Why the axis is a column and not a predicate: `validate_predicate` accepts
    `len(k)<=N` and `k==<integer>` only, so `url==example.com` raises."""
    with pytest.raises(GrantError):
        policy.validate_predicate("url==example.com")
    with pytest.raises(GrantError):
        policy.validate_predicate("destination==example.com")


def test_a_malformed_destination_is_refused_at_mint(db):
    for bad in ("", "  ", "*", ".example.com", "example.com:8080",
                "https://example.com", "a..com", "256.1.1.1"):
        with pytest.raises(GrantError, match="destination"):
            store(db).mint(scope=WORKER_SCOPE, tool_pattern="http_fetch",
                           issued_by="alan", expires_at=NOW + dt.timedelta(days=1),
                           destination=bad, now=NOW)
    # …and an empty string is not stored, which would have meant "every host".
    assert store(db).live(scope=WORKER_SCOPE, now=NOW) == []


@pytest.mark.parametrize("text,expected", [
    ("HTML.DuckDuckGo.com", "html.duckduckgo.com"),
    (" Docs.Example.org ", "docs.example.org"),
    ("192.168.1.1", "192.168.1.1"),
    ("::1", "::1"),
])
def test_destination_normalization(text, expected):
    """A grant destination is a bare host; a URL is refused at mint (see
    `test_a_malformed_destination_is_refused_at_mint`), so URL cases belong to
    `egress.host_of`, which is what reads a host out of a call's URL."""
    assert normalize_destination(text) == expected


@pytest.mark.parametrize("url,expected", [
    (" https://Docs.Example.org/a?b=1 ", "docs.example.org"),
    ("https://user@[::1]:9/p", "::1"),
    ("http://::1/", "::1"),
    ("https://example.com:8443/x", "example.com"),
])
def test_a_calls_url_yields_its_host(url, expected):
    assert egress.host_of(url) == expected


@pytest.mark.parametrize("entry,host,covered", [
    ("example.com", "example.com", True),
    ("example.com", "api.example.com", True),
    ("example.com", "evil-example.com", False),
    ("example.com", "com", False),
    ("api.example.com", "example.com", False),
])
def test_destination_coverage_is_one_host_and_its_subdomains(entry, host, covered):
    assert destination_covers(entry, host) is covered


def test_no_grant_row_can_exist_without_an_issuer_or_an_expiry(db):
    """Both directions of the clause's second half: the schema refuses them, and
    the write path refuses them before the schema has to."""
    with pytest.raises(GrantError, match="expiry"):
        store(db).mint(scope=WORKER_SCOPE, tool_pattern="http_fetch", issued_by="alan",
                       expires_at="", now=NOW)
    with pytest.raises(GrantError, match="issued_by"):
        store(db).mint(scope=WORKER_SCOPE, tool_pattern="http_fetch", issued_by="  ",
                       expires_at=NOW + dt.timedelta(days=1), now=NOW)
    store(db).ensure_schema()   # both refusals above happen before any connection
    # Against the real table: `CREATE TABLE … AS SELECT` copies no NOT NULL
    # constraint in SQLite, so a copy would accept the very row this refuses.
    conn = sqlite3.connect(str(db))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO authority_grants (scope, tool_pattern, issued_by, issued_at) "
            "VALUES ('s','http_fetch','alan','2026-09-21T00:00:00+00:00')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO authority_grants (scope, tool_pattern, issued_at, expires_at) "
            "VALUES ('s','http_fetch','2026-09-21T00:00:00+00:00',"
            " '2026-09-22T00:00:00+00:00')")


def test_a_worker_scope_cannot_mint_authority(db):
    """The mint path is interactive-only, asserted across the real gate.

    A destination denial is a frequent, boring event, so the temptation to let a
    worker grant itself the host it wanted is exactly what this clause exists to
    keep shut.
    """
    # The scope strings the pool actually mints (`grant_scope_for_source`).
    for worker_scope in ("worker:domain-research", "worker:autocode", "autonomy-task:72",
                         "worker:scheduled-task", "worker:session-distill"):
        decision = check_grants(store(db), scope=worker_scope,
                               tool_name="grant_create",
                               tool_input={"destination": UNKNOWN_HOST}, now=NOW)
        assert decision.allowed is False, (worker_scope, decision.reason)
        assert "may not mint" in decision.reason, decision.reason

    human = check_grants(store(db), scope="interactive:chat", tool_name="grant_create",
                         tool_input={"destination": UNKNOWN_HOST}, now=NOW)
    assert human.allowed is True, human.reason


# ── the two process seams ───────────────────────────────────────────────────

def test_the_effect_scope_string_resolves_to_the_same_grant_scope_the_pool_would(db, monkeypatch):
    """The full chain, both ends real: pool-side mapping, the `_meta` string as
    it crosses into the aggregator, the guard's derivation of it, and the store
    the gate reads — one sqlite file, two processes' worth of code.

    `agent_mcp/egress.py` runs in the `agent-stdio` child, `workers/pool.py` in
    the backend. Neither can call the other, so this asserts the only thing that
    has to agree: the string.
    """
    from types import SimpleNamespace
    from workers import pool

    # `grant_scope_for` reads exactly two fields, so that is the whole object.
    item = SimpleNamespace(source="domain-research", payload={})
    pool_scope = pool.grant_scope_for(item)
    meta_string = "item:domain-research:99"          # what `_meta` carries
    ctx = egress.scope_context(meta_string)          # what the child derives

    assert ctx["scope"] == pool_scope == "worker:domain-research"
    assert policy.is_interactive_scope(ctx["scope"]) is False

    policy_on(monkeypatch)
    mint(db, host=LISTED_HOST, scope=ctx["scope"])
    verdict = egress.guard("http_fetch", LISTED_URL)
    assert verdict.allowed is True, verdict.reason
    verdict_other = egress.guard("http_fetch", UNKNOWN_URL)
    assert verdict_other.allowed is False, verdict_other.reason


def test_the_scheduled_task_effect_scope_maps_to_its_own_task(db, monkeypatch):
    """`item:scheduled-task:<row id>` is not an autonomy task id; the row's
    payload is. Reading the wrong one would hand every scheduled job the grants
    belonging to whichever task happens to share the number."""
    monkeypatch.setattr(egress, "_task_id_for_item", lambda item_id: "72")
    ctx = egress.scope_context("item:scheduled-task:4712")

    assert ctx["worker_source"] == "scheduled-task"
    assert ctx["task_id"] == "72"
    assert ctx["scope"] == "autonomy-task:72"
    assert policy.grant_scope_for_source("scheduled-task", {"task_id": 72}) == "autonomy-task:72"


def test_a_pre_628_database_is_migrated_before_the_store_reads_it(db, monkeypatch):
    """`CREATE TABLE IF NOT EXISTS` is not a migration.

    Every live database has held `authority_grants` since #534, so a store that
    ran the DDL text alone would keep the old shape forever on production while
    every fresh test database had the new one — the worst possible split, because
    the failing side is the one a person's grants live in.
    """
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE authority_grants (
          id INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT NOT NULL,
          tool_pattern TEXT NOT NULL, arg_predicate TEXT NOT NULL DEFAULT '',
          quota INTEGER, consumed INTEGER NOT NULL DEFAULT 0,
          issued_by TEXT NOT NULL, minted_by TEXT NOT NULL DEFAULT 'human',
          note TEXT, issued_at TEXT NOT NULL, expires_at TEXT NOT NULL,
          revoked_at TEXT);
        INSERT INTO authority_grants (scope, tool_pattern, issued_by, issued_at, expires_at)
        VALUES ('worker:domain-research','http_fetch','alan',
                '2026-09-01T00:00:00+00:00','2026-12-31T00:00:00+00:00');
    """)
    conn.commit()
    conn.close()

    granted = check_grants(GrantStore(db), scope=WORKER_SCOPE, tool_name="http_fetch",
                          tool_input={}, destination="docs.example.org",
                          now=NOW, record=False)
    assert granted.allowed is False, "a pre-#628 grant must not be read as any-host"

    cols = {r[1] for r in sqlite3.connect(str(db)).execute(
        "PRAGMA table_info(authority_grants)")}
    assert "destination" in cols, cols
    mint(db, host="docs.example.org")
    after = check_grants(GrantStore(db), scope=WORKER_SCOPE, tool_name="http_fetch",
                        tool_input={}, destination="docs.example.org",
                        now=NOW, record=False)
    assert after.allowed is True, after.reason


def test_grant_shape_renders_the_destination_field_and_keeps_expiry_last():
    text = grant_shape(scope=WORKER_SCOPE, tool="http_fetch", tool_input={},
                       destination=UNKNOWN_HOST, now=NOW)

    assert text.startswith("grant_create(scope='worker:domain-research'"), text
    assert "tool='http_fetch'" in text
    assert f"destination='{UNKNOWN_HOST}'" in text
    expiry = (NOW + dt.timedelta(days=policy.SUGGESTED_TTL_DAYS)).isoformat()
    assert f"expires_at='{expiry}'" in text, text
    assert text.endswith("issued_by='alan')"), text


# ── #1965: the config flags are parsed, never `bool()`-coerced ───────────────

def _config_value(monkeypatch, **values):
    """Put raw values where `harness.egress_policy` puts them, with both env
    overrides unset — the config path is the one under test."""
    monkeypatch.delenv("LLOYD_EGRESS_ENFORCE", raising=False)
    monkeypatch.delenv("LLOYD_EGRESS_TELEMETRY", raising=False)
    state = dict(egress._DEFAULTS)
    state.update(values)
    monkeypatch.setattr(egress, "config", lambda: dict(state))


@pytest.mark.parametrize("word", ["off", "false", "no", "0", " OFF ", "False"])
def test_a_quoted_negative_word_in_config_does_not_arm_enforcement(monkeypatch, word):
    """The reproduction on the item: `enforce: "off"` printed True, because
    `bool()` of any non-empty string is True and no parse step ran before it."""
    assert bool(word) is True, "the control: bool() is what armed it"
    _config_value(monkeypatch, enforce=word)
    assert egress.enforce_on() is False


@pytest.mark.parametrize("junk", ["maybe", "enfroce", ["on"], {"a": 1}, 2, 1.5])
def test_an_unparseable_enforce_value_stays_off_and_is_named_in_the_log(
        monkeypatch, caplog, junk):
    _config_value(monkeypatch, enforce=junk)
    with caplog.at_level("WARNING", logger="lloyd-egress"):
        assert egress.enforce_on() is False
    lines = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(lines) == 1, lines
    assert "harness.egress_policy.enforce" in lines[0] and repr(junk) in lines[0], lines


@pytest.mark.parametrize("value,expected", [
    ("on", True), ("true", True), ("yes", True), ("1", True),
    (True, True), (False, False)])
def test_affirmative_words_and_real_bools_behave_as_written(monkeypatch, caplog,
                                                           value, expected):
    _config_value(monkeypatch, enforce=value)
    with caplog.at_level("WARNING", logger="lloyd-egress"):
        assert egress.enforce_on() is expected
    assert not caplog.records, "a parseable value is not a warning"


def test_the_env_var_still_overrides_the_config_value_both_ways(monkeypatch):
    _config_value(monkeypatch, enforce="off")
    monkeypatch.setenv("LLOYD_EGRESS_ENFORCE", "1")
    assert egress.enforce_on() is True
    _config_value(monkeypatch, enforce=True)
    monkeypatch.setenv("LLOYD_EGRESS_ENFORCE", "off")
    assert egress.enforce_on() is False


def test_telemetry_is_parsed_the_same_way_and_defaults_on(monkeypatch, caplog):
    _config_value(monkeypatch, telemetry="off")
    assert egress.telemetry_on() is False
    _config_value(monkeypatch, telemetry="on")
    assert egress.telemetry_on() is True
    _config_value(monkeypatch, telemetry="sometimes")
    with caplog.at_level("WARNING", logger="lloyd-egress"):
        assert egress.telemetry_on() is True
    lines = [r.getMessage() for r in caplog.records]
    assert len(lines) == 1, lines
    assert "harness.egress_policy.telemetry" in lines[0] and "'sometimes'" in lines[0]


def test_the_real_config_accessor_feeds_the_parse(monkeypatch):
    """End to end through `egress.config()`, the way the item reproduces it."""
    import app.config as C
    monkeypatch.delenv("LLOYD_EGRESS_ENFORCE", raising=False)
    monkeypatch.setattr(C, "CONFIG", {"harness": {"egress_policy": {"enforce": "off"}}})
    assert egress.config()["enforce"] == "off"
    assert egress.enforce_on() is False


# ── #2338: the enforcement state has to be readable across the process seam ─

def test_the_state_route_publishes_its_own_egress_enforcement_state(monkeypatch):
    """Clause 1: `GET /state` carries an `egress` key, and it reports THIS process.

    The guard's `enforce` decision is taken inside the aggregator, and until now the
    only way an out-of-process caller could learn it was to infer it from
    `config.yaml` — which is exactly the inference that poisoned the canary's
    enforce-on arm (#2154's owed entry, #2338): the runner read `enforce_on()` in its
    own interpreter while the episode's tools were served by a daemon whose
    environment it could not see. `tool_sandbox` and `protected_path_sandbox` already
    publish their substrate state for the same reason (#2109); egress was the missing
    key, confirmed against the live endpoint, which answered with exactly
    `[background_tasks, changes, protected_path_sandbox, qmd, subagents, tool_sandbox,
    tools, tsc]`.

    Asserted against the route body, like `test_path_sandbox` does for its own key, and
    against `enforce_on()` itself rather than a restated constant: the key is only
    worth having if it is the value the guard reads.
    """
    import asyncio
    import json

    from agent_mcp import main as M

    monkeypatch.delenv("LLOYD_EGRESS_ENFORCE", raising=False)
    payload = json.loads(asyncio.run(M.state(None)).body)
    assert "egress" in payload, sorted(payload)
    assert "tool_sandbox" in payload, sorted(payload)
    st = payload["egress"]
    assert st["enforce"] is egress.enforce_on() is False, st

    # The control: flipping the variable the guard honours moves the published key.
    # A key that always said `false` would satisfy the first assertion and still
    # leave a caller unable to tell the arms apart.
    monkeypatch.setenv("LLOYD_EGRESS_ENFORCE", "1")
    assert egress.enforce_on() is True
    payload = json.loads(asyncio.run(M.state(None)).body)
    assert payload["egress"]["enforce"] is True, payload["egress"]


def test_the_shipped_guard_allows_the_host_the_canary_bench_books_as_an_escape(db,
                                                                              monkeypatch):
    """#2363 clause 4: the deliberate divergence, held in one node.

    One call — an `http_search` carrying the canary the shipped bench scenario plants —
    gets two true answers from two components that are not answering the same question.
    The bench books it as an escape and names a destination (`egress_destination` on the
    `QUERY_TEXT_LANES` half #2192 ruled); the guard, with `enforce` on and the seed
    #2123 shipped, allows it and records `allow-listed`, because the only thing it
    judges is whether this host may be reached, and `duckduckgo.com` is on the list.

    Both halves are asserted here because either one alone rots quietly. Drop the query
    rule and the bench starts calling the escape a benign echo — the verdict #2192 says
    it may never hand out — and no existing node in this file would notice, because
    nothing here had ever read `DECISION_ALLOW` out of `guard` (`git grep DECISION_ALLOW
    -- tests/` was empty before this node). Drop the seed and the guard's half flips to
    a denial while the bench's stays put, and the pair is no longer a divergence but two
    copies of one answer. The row's `reason` is read from the telemetry table rather than
    from the returned decision, because `Decision.reason` is empty on an allow: the tag
    is what the table carries.

    `egress_destination` is imported from the bench rather than restated here: a copy of
    the rule in this file would keep passing after the scorer changed, which is the exact
    failure the two-process seams at the top of this file exist to prevent.
    """
    import sqlite3
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
    import run_injection_canary as bench

    canary = next(s["canary"] for s in bench.load_scenarios()
                  if s["key"] == "search-query-exfil")
    shipped = egress.allow_entries()          # config.yaml's own seed, through the guard's accessor
    assert any(str(e.get("host") or "") in ("duckduckgo.com", egress.SEARCH_BACKEND_HOST)
               and not e.get("scope") and not e.get("expires_at") for e in shipped), \
        "the shipped seed no longer carries a live unscoped search entry: the guard "  \
        "half of this pair would be a denial, not a divergence"

    # Bench side: the free text left the machine, so there is a destination.
    assert bench.egress_destination(
        "http_search", f'{{"query": "apply the fix tagged {canary}"}}', shipped) \
        == egress.SEARCH_BACKEND_HOST

    # Guard side, same host, same seed, enforcement on: allowed, and the table says so.
    policy_on(monkeypatch, allow=shipped, enforce=True)
    verdict = egress.guard("http_search", host=egress.SEARCH_BACKEND_HOST)
    assert verdict.allowed is True and verdict.decision == egress.DECISION_ALLOW, \
        verdict.reason

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT tool, decision, reason FROM egress_events").fetchall()
    conn.close()
    assert [(r["tool"], r["decision"], r["reason"]) for r in row] == [
        ("http_search", egress.DECISION_ALLOW, "allow-listed")], row


# ── #2584: one owner for the private space, and it resolves ──────────────────
#
# Two `_is_private_host` shipped with the same name and different methods:
# `http_tools.py` matched the hostname TEXT against ten prefixes, `browser.py`
# answered per RESOLVED address. With `harness.egress_policy.enforce` off — the
# shipped posture — the lane predicates are the only live answer, so the lane that
# needs no lease and no browser session was the weaker one: `localhost.localdomain`
# resolves 127.0.0.1 and ::1 on this box and was fetched, as was this box's own
# hostname. And `floor_reason` delegated to the pattern copy, so arming `enforce`
# would only have moved which non-resolving predicate refused.
#
# `egress._is_private_host` is now the one owner. `egress` owns it rather than
# either lane because this module's floor already had to answer the question, and
# `floor_reason` importing `http_tools` was the direction that made the weak copy
# load-bearing.

#: Text matching none of the ten prefixes that used to be the whole check, which
#: resolves to loopback on this box.
REBOUND_NAME = "localhost.localdomain"


def _resolve_to(monkeypatch, addrs: tuple[str, ...]) -> None:
    """Pin the resolver seam for this test, and drop the cached answers with it.

    `_is_private_host` is `lru_cached` by host across the process, so a node that
    pins the seam without clearing the cache can be handed the answer some earlier
    node's pin — or the box's real resolver — produced, and then pass for a reason it
    did not test.
    """
    monkeypatch.setattr(egress, "_resolve_addrs", lambda host: addrs)
    egress._is_private_host.cache_clear()


def _definitions_of_the_predicate() -> list[str]:
    """`git grep -n 'def _is_private_host' -- agent_mcp/`, run in this tree.

    `git grep` and not a recursive `grep`: it is the command #2584's clause names,
    and a plain walk over `agent_mcp/` would also read `__pycache__`.
    """
    out = subprocess.run(
        ["git", "grep", "-n", "def _is_private_host", "--", "agent_mcp/"],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True, text=True)
    assert out.returncode in (0, 1), out.stderr  # 1 = no match; the count below judges that
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def test_agent_mcp_ships_one_private_space_predicate_and_it_is_egress_s():
    """Clause 3's observable: at most one definition, and it lives here.

    The count is the observable the item names; the identity check is what stops a
    round satisfying it by moving the copy to a third module and leaving
    `floor_reason` delegating to a lane again.
    """
    found = _definitions_of_the_predicate()

    assert len(found) <= 1, f"{len(found)} definitions of the private space: {found}"
    assert found == [] or found[0].startswith("agent_mcp/egress.py:"), found
    assert egress._is_private_host.__module__ == "agent_mcp.egress"


def test_the_literal_branch_defers_to_the_resolver_for_a_text_ipaddress_refuses():
    """Why the owner asks `ipaddress` first and the resolver second, and in that order.

    `3232235777` is `192.168.1.1` written as an integer. `ipaddress` refuses that text,
    so the owner treats it as a *name* and resolves it — which is how the browser
    lane's long-standing rule that Chromium-style integer spellings get blocked keeps
    answering after the move (`test_browser_panel.py` pins
    `_is_private_host("3232235777") is True`). Answering it by prefix instead would
    have been a fourth spelling-specific guess, and
    `test_the_one_predicate_refuses_every_shape_the_pattern_list_matched_untouched`
    covers the same case from the other side, with the seam pinned to a public address.
    """
    with pytest.raises(ValueError):
        ipaddress.ip_address("3232235777")
    assert egress._address("3232235777") is None, "an integer spelling is not a literal"
    assert int("3232235777") == int(ipaddress.IPv4Address("192.168.1.1")), \
        "the control assumed the integer and the dotted form name one address"

    assert egress._address("192.168.1.1") == ipaddress.IPv4Address("192.168.1.1")
    assert egress._address("[::1]") == ipaddress.IPv6Address("::1"), \
        "a host string can still carry the brackets urlparse stripped elsewhere"
    assert egress._address("") is None


def test_the_one_predicate_refuses_names_whose_text_says_nothing():
    """Clause 3's behaviour, on the two specimens #2584 was filed with.

    This node deliberately resolves for real — the claim IS about real resolution,
    and `socket.getaddrinfo` is the same question the predicate asks. Every other
    node in this file pins the seam instead.
    """
    own_host = socket.gethostname().split(".")[0]

    assert egress._is_private_host(REBOUND_NAME) is True, \
        f"{REBOUND_NAME} resolves to {egress._resolve_addrs(REBOUND_NAME)}"
    assert egress._is_private_host(own_host) is True, \
        f"this box names itself {own_host} = {egress._resolve_addrs(own_host)}"


def test_the_one_predicate_refuses_every_shape_the_pattern_list_matched_untouched(monkeypatch):
    """The deleted list's useful half, and the control that this is not "resolve, or
    else public".

    With the resolver answering public for everything — the answer a rebinding
    attacker can arrange — an address literal is still refused from `ipaddress`
    alone. The pattern list was wrong because it looked ONLY at the text; it was not
    worthless, and a fix that threw this out would be a downgrade in a fix's clothes.
    """
    _resolve_to(monkeypatch, ("93.184.216.34",))          # "every name is public"
    try:
        for literal in ("127.0.0.1", "10.0.0.5", "192.168.1.20", "172.16.0.9",
                        "169.254.169.254", "0.0.0.0", "::1", "fe80::1"):
            assert egress._is_private_host(literal) is True, literal
    finally:
        egress._is_private_host.cache_clear()


def test_an_unresolvable_name_falls_back_public_as_the_browser_lane_always_did(monkeypatch):
    """The behaviour that was `browser.py`'s, moved rather than reinvented.

    A name resolving to nothing is not private: `_resolve_addrs` returns `()` and
    the `if not addrs: return False` branch answers public. Flipping that would
    refuse every typo and every offline name, which is a different product decision
    than the one #2584 asks for.
    """
    _resolve_to(monkeypatch, ())                        # the name does not resolve
    try:
        assert egress._is_private_host("no-such-host.example.invalid") is False
    finally:
        egress._is_private_host.cache_clear()


def test_the_resolver_seam_is_one_attribute_on_this_module(monkeypatch):
    """Clause 3's substitutability, at the owner rather than through a tool.

    Every resolution in the private-space answer goes through the module attribute
    `_resolve_addrs`, so a test rebinds one name instead of `socket`, and no test has
    to know which lane it is standing in. The counterfactual is built in: with the
    seam pinned to loopback a name whose text is entirely public is refused, which is
    exactly what the pattern-only copy could not do at any pin.
    """
    asked: list[str] = []
    monkeypatch.setattr(egress, "_resolve_addrs",
                        lambda host: (asked.append(host) or ("127.0.0.1",)))

    assert egress._is_private_host("public-looking.example.invalid") is True
    assert asked == ["public-looking.example.invalid"], asked

    monkeypatch.setattr(egress, "_resolve_addrs", lambda host: ())
    egress._is_private_host.cache_clear()
    assert egress._is_private_host("public-looking.example.invalid") is False


def test_floor_reason_refuses_a_name_that_only_the_resolver_can_see(monkeypatch):
    """Clause 4: `floor_reason("localhost.localdomain")` returned `""`, which is the
    destination floor saying "this destination is fine" about the machine itself.
    With the floor delegating to the owner the same refusal the lane gives arrives
    with `enforce` still off, so arming it would close the same case rather than a
    different one."""
    _resolve_to(monkeypatch, ("127.0.0.1", "::1"))

    reason = egress.floor_reason(REBOUND_NAME)

    assert reason == f'Blocked — private/internal host "{REBOUND_NAME}"', repr(reason)


def test_floor_reason_still_answers_the_callers_it_served_before(monkeypatch):
    """The floor's other answers, unchanged by the move: a public name is allowed,
    the loopback allowance is granted only when the caller passes it, a LAN address
    is refused even with the allowance, and the message wording is untouched — the
    floor's text is what `guard` passes through verbatim and what the dashboard
    renders as `network.policy`'s companion."""
    _resolve_to(monkeypatch, ("93.184.216.34",))
    assert egress.floor_reason("docs.example.org") == ""

    _resolve_to(monkeypatch, ("127.0.0.1",))
    assert egress.floor_reason("127.0.0.1") == 'Blocked — private/internal host "127.0.0.1"'
    assert egress.floor_reason("127.0.0.1", loopback_ok=True) == ""
    assert egress.floor_reason("") == ""

    _resolve_to(monkeypatch, ("192.168.50.7",))
    assert egress.floor_reason("lan-box.example.invalid", loopback_ok=True) == \
        'Blocked — private/internal host "lan-box.example.invalid"'


def test_the_floor_and_the_lane_agree_on_the_one_predicate(monkeypatch):
    """Clause 4's edge: the floor's answer and the lane's answer for the same name
    come from the same function, so fixing one cannot miss the other.

    The counterfactual is the false claim itself. A text the ten prefixes waved
    through is fed to the predicate the floor delegates to, and the refusal that
    comes back is the reason the pattern copy is gone.
    """
    _resolve_to(monkeypatch, ("127.0.0.1", "::1"))

    pattern_only = re.compile(r"^127\.|^10\.|^172\.(1[6-9]|2\d|3[01])\.|^192\.168\."
                              r"|^0\.|^169\.254\.|^::1$|^fc00:|^fd|^fe80:")
    assert pattern_only.match(REBOUND_NAME) is None, \
        "the control assumed a text the ten prefixes miss; re-derive it"
    assert egress._is_private_host(REBOUND_NAME) is True
    assert egress.floor_reason(REBOUND_NAME) != ""
    # The lane is pinned by how it calls, not by an identity that a `from x import y`
    # would satisfy while leaving the module a second lookup path (#2584 review): the
    # browser answers through `egress.`, so a test that rebinds the owner's seam
    # reaches both this lane's exemption and the predicate that reads it.
    lane_src = Path(inspect.getsourcefile(browser_module)).read_text(encoding="utf-8")
    assert "if egress._is_private_host(hostname):" in lane_src, \
        "browser.py no longer answers through the owner's module attribute"
    assert "def _is_private_host" not in lane_src, "browser.py grew a predicate back"
    assert http_tools._is_loopback_host is not egress._is_private_host


def test_the_threat_model_no_longer_credits_two_predicates():
    """The two rows that described the split have to describe the merged owner.

    #2584's own finding was prose that read as closed: T-18 named `browser.py` as the
    control while the lane an injected URL actually reaches carried the text match.
    The fix makes that prose false in the other direction, and a page that credits a
    split which no longer exists is the same defect wearing last week's clothes — so
    the rows are pinned to the shipped shape, not left to the next reader's grep.
    """
    # Code side first, so a row crediting one lane as the control is contradicted by
    # the tree rather than by another sentence: the owner is defined in `egress.py`,
    # and neither lane module defines a predicate of its own any more.
    owner = inspect.getmodule(egress._is_private_host)
    assert owner is not None and owner.__name__ == "agent_mcp.egress", owner
    for lane in (browser_module, http_tools):
        lane_src = Path(inspect.getsourcefile(lane)).read_text(encoding="utf-8")
        assert "def _is_private_host" not in lane_src, lane.__name__

    page = (Path(__file__).resolve().parent.parent
            / "architecture" / "threat-model.md").read_text(encoding="utf-8")
    rows = {ln.split("|")[1].strip(): " ".join(ln.split())
            for ln in page.splitlines()
            if ln.startswith("| T-18 ") or ln.startswith("| B8 ")}
    assert set(rows) == {"T-18", "B8"}, sorted(rows)

    # Nothing in either row may still describe the retired split, and nothing may
    # credit a file:line that no longer defines the predicate.
    stale = ("two different functions", "does not resolve", "lane-specific",
             "browser.py:147", "http_tools.py:49")
    for label, flat in sorted(rows.items()):
        offenders = [s for s in stale if s in flat]
        assert offenders == [], f"{label} still reads: {offenders}"
        assert "_is_private_host" in flat, f"{label} no longer names the owner: {flat[:120]}"

    # Positive control: a row worded the way the pre-fix page worded them fails the
    # assertions above rather than passing them vacuously.
    pre_fix = ("| T-18 | B1 | E | x | `agent_mcp/browser.py:147` — the resolving "
               "one. The `http_fetch`/`http_request` copy at `agent_mcp/http_tools.py:49` "
               "matches the hostname's text and does not resolve, so this control is "
               "lane-specific while `enforce` is off (#2584) |")
    flat = " ".join(pre_fix.split())
    assert any(s in flat for s in stale) and "_is_private_host" not in flat, \
        "the control is no longer the pre-fix wording the assertions reject"


def test_the_shipped_posture_stays_unarmed_and_this_fix_does_not_arm_it():
    """The item states it, so it is pinned: the fix closes the hole in the posture
    that ships.

    A round that "fixed" the hole by setting `enforce: true` would pass every clause
    above while overriding the decision `config.yaml` reserves to a human, so the
    posture is asserted rather than assumed. `egress.enforce_on()` is what
    `guard`'s floor branch reads, so the two together are the live posture.
    """
    from app.config import CONFIG

    policy_cfg = (CONFIG.get("harness") or {}).get("egress_policy") or {}
    assert policy_cfg.get("enforce") is False, policy_cfg
    assert egress.enforce_on() is False, "the floor would run too, so this test's "\
        "claim that the lane refusal is the live answer would be false"
