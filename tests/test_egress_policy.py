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
import json
import sqlite3

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
