"""#4044 clauses 2, 3 and 4: what the Bash recording arm puts in a child's environment,
what it writes when that child reaches somewhere, and the one thing it must never do —
change what the child sees, or refuse it.

The premise these nodes hold in place is the one triage re-verified at `1b704b04` and
again on 2026-11-11: `egress.guard` is reached from exactly four tools, and the Bash
spawn path touches the egress module only to *read* `scope_context()` for credentials.
So a Bash child — the tool session-distill uses 721 times in 14 days, in a source that
sits in `capabilities.UNTRUSTED_INGEST_SOURCES` — can contact any host on the box's
network and no row is written. The destination inventory that
`harness.egress_policy.enforce` would be armed from has never counted the widest tool.

Three rules decide how these nodes are written, each against a failure this change could
have shipped with:

* **A real child, at the real spawn site.** Every environment claim is read back out of a
  process spawned by `builtin_bash._bash` — the same coroutine the Bash tool calls — not
  out of `proxy_env_for_turn()`. A node that called the builder directly would pass while
  the spawn site stopped calling it, the residual #3846's own test file names.
* **The flag-off half is measured, not read out of the YAML.** `bash_record: false` in
  config.yaml is a claim about a file; the claim that matters is that the child's
  environment is the one it got yesterday, so the default-state node inspects a real
  child's keys.
* **Nothing here is evidence about clients.** These nodes drive the recorder through
  loopback destinations, from a child that is told to speak to it. Whether curl, pip,
  yt-dlp, Node's native `fetch` or a raw TCP connect hand their traffic to an env-var
  proxy at all is the per-class recall figure that needs real egress from a live session;
  it is recorded as owed on #4044, and a green suite here does not answer it.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import re
import socket
import sqlite3
import sys
import threading
import time

import pytest

from agent_mcp import _bash_egress_proxy as proxy
from agent_mcp import _path_sandbox
from agent_mcp import _tool_sandbox
from agent_mcp import builtin_bash as B
from agent_mcp import egress
from app.harness import policy as _policy

#: The source every node here runs as, and the grant scope it derives to. Spelled out
#: rather than derived from the constant's own text because the arm's whole attribution
#: model is "the scope captured at the spawn site", and `''` is both the unscoped answer
#: and the answer a broken capture produces — a node asserting the empty case would pass
#: on the bug it exists to catch.
SOURCE = "session-distill"
EFFECT_SCOPE = f"item:{SOURCE}:4044"
GRANT_SCOPE = f"worker:{SOURCE}"

#: The body the fixture origin serves. Distinctive so a node cannot mistake the
#: recorder's own error page for the origin's answer.
ORIGIN_BODY = "origin-body-4044\n"


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A workers.db of our own. The arm's flags are per-node, never inherited."""
    path = tmp_path / "workers.db"
    monkeypatch.setenv("LLOYD_EGRESS_DB", str(path))
    monkeypatch.setenv("LLOYD_GRANT_DB", str(path))
    monkeypatch.setattr(egress, "_init_done", set())
    return path


def _flags(monkeypatch, *, bash_record: bool, enforce: bool = False) -> None:
    """The config the guard and the arm read, with one variable per node.

    Replaced wholesale rather than patched key-by-key because `egress.config()` merges
    `_DEFAULTS` under the file's values, and a node that only flipped `bash_record` would
    silently inherit whatever `enforce` the surrounding config carries — including the
    canary arm's `LLOYD_EGRESS_ENFORCE=1`.
    """
    monkeypatch.setattr(egress, "config", lambda: {
        "telemetry": True, "enforce": enforce, "bash_record": bash_record,
        "allow": [], "retention_days": 90})
    # Both overrides, because `telemetry_on()` and `enforce_on()` each read an env
    # override BEFORE the config value, and this box's MCP process really does run with
    # `LLOYD_EGRESS_ENFORCE=1` — the Monday egress-window arm (`36c3607f`, 2026-11-02).
    # Without the two clears, an `enforce=False` node here would silently be an
    # `enforce=True` node, and the `enforce=True` node would prove nothing about the
    # flag it names.
    for name in ("LLOYD_EGRESS_ENFORCE", "LLOYD_EGRESS_BASH_RECORD",
                 "LLOYD_EGRESS_TELEMETRY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _turn(monkeypatch):
    """A worker turn with a real grant scope, and no path sandbox.

    The sandbox patch sends `_bash` down its non-sandboxed branch, which is the branch a
    test can spawn; the sandboxed branch's rule (never proxy a bench or eval child) is
    checked on the contextvar directly, since that branch needs the bwrap substrate. The
    effect scope is set through the contextvar the aggregator itself binds, so
    `scope_context()` resolves the same way it does on a real worker turn.
    """
    monkeypatch.setattr(_path_sandbox, "available",
                        lambda: (False, "test: path sandbox substrate held back"))
    token = _policy.current_effect_scope.set(EFFECT_SCOPE)
    try:
        yield
    finally:
        _policy.current_effect_scope.reset(token)
        proxy.stop_all()


@pytest.fixture
def origin():
    """A loopback HTTP origin serving one body, on an ephemeral port. Stopped on teardown."""
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 — http.server's spelling
            body = ORIGIN_BODY.encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str = "", *args) -> None:    # keep the suite's output readable
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def rows(db, *, via: str | None = None) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        sql = ("SELECT tool, via, destination, host, decision, scope, session_id,"
               " worker_source, task_id, reason FROM egress_events")
        args: tuple = ()
        if via is not None:
            sql += " WHERE via=?"
            args = (via,)
        return list(conn.execute(sql + " ORDER BY id", args))
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def _child(command: str, *, background: bool = False, timeout: int = 60) -> str:
    """Run one real Bash child through the tool's own coroutine."""
    args = {"command": command, "timeout": timeout}
    if background:
        args["run_in_background"] = True
        args["summary"] = "probe the child environment"
    return asyncio.run(B._bash(args))


def _wait_for_rows(db, *, via: str, count: int = 1, seconds: float = 10.0):
    """Poll for the recorder's rows, and fail loudly if they do not arrive.

    The row is written by the accept thread, out of band from the command that caused it,
    so reading the table the instant `_bash` returns races it — and the race's failure
    mode is a red node whose cause is timing, which is how a real blind spot gets
    ignored later. Waiting is the honest shape of an out-of-band instrument.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        got = rows(db, via=via)
        if len(got) >= count:
            return got
        time.sleep(0.05)
    raise AssertionError(
        f"expected >= {count} row(s) with via={via!r} within {seconds}s; saw "
        f"{len(rows(db))} row(s) of any via")


# The children are one-liners run by this session's interpreter — the only python
# guaranteed present — and each reads the recorder's address out of its OWN environment
# rather than out of a fixture. The point is what a child can see and use.
_ENV_DUMP = ("import json,os,sys;"
             "open(sys.argv[1],'w').write(json.dumps({k:v for k,v in os.environ.items()"
             " if 'PROXY' in k.upper()}))")
_ABSOLUTE_PROBE = ("import sys,urllib.request;"
                   "sys.stdout.write(urllib.request.urlopen(sys.argv[1],timeout=25)"
                   ".read().decode())")
_CONNECT_PROBE = (
    "import os,socket,sys;"
    "p=os.environ['HTTPS_PROXY'];h=p.split('//')[1].split(':')[0];"
    "n=int(p.rsplit(':',1)[1]);"
    "s=socket.create_connection((h,n),timeout=25);"
    "s.sendall(b'CONNECT '+sys.argv[1].encode()+b' HTTP/1.1\\r\\n\\r\\n');"
    "sys.stdout.write(s.recv(256).decode('latin-1').split('\\r\\n')[0])")
_RAW_ABSOLUTE_PROBE = (
    "import os,socket,sys;"
    "p=os.environ['HTTP_PROXY'];h=p.split('//')[1].split(':')[0];"
    "n=int(p.rsplit(':',1)[1]);"
    "s=socket.create_connection((h,n),timeout=25);"
    "s.sendall(b'GET '+sys.argv[1].encode()+b' HTTP/1.1\\r\\nHost: x\\r\\n\\r\\n');"
    "sys.stdout.write(s.recv(8192).decode('latin-1'))")


# ── clause 2: the environment the arm hands a child ─────────────────────────

def test_the_shipped_default_hands_the_child_exactly_todays_environment(tmp_path, db):
    """Flag off — the shipped state — and no proxy variable reaches a real child.

    Asserted on a spawned process rather than read out of config.yaml because the flag
    governs every unattended `curl` on the box: #2269 records a Bash narrowing that broke
    the `curl` health checks and was reverted, so "off means off" has to be a measured
    property of the spawn. The filter is on the variable NAME, so a value inherited from
    a developer shell would not hide an injected one — and no node here sets one by hand.
    """
    out = tmp_path / "off.json"
    _child(f'{sys.executable} -c "{_ENV_DUMP}" {out}')

    seen = json.loads(out.read_text(encoding="utf-8"))
    assert seen == {}, (
        f"the shipped default handed the child proxy variables: {seen} — every "
        "unattended curl on the box would now depend on the recorder being up")


def test_the_flag_on_child_gets_the_recorder_in_every_spelling_a_client_reads(
        tmp_path, db, monkeypatch):
    """With `bash_record` on, the foreground spawn site points the child at the loopback
    recorder, and NO_PROXY keeps loopback and RFC1918 off it.

    Both cases of each name are asserted because clients disagree about which they read:
    curl and git follow `http_proxy`, most libraries follow `HTTP_PROXY`, and `ALL_PROXY`
    is what a non-HTTP client reads. `NO_PROXY` is the half that keeps the arm from
    breaking the fleet's own probes — the watchdog's `curl http://127.0.0.1:PORT/health`
    must never depend on this process being alive.
    """
    _flags(monkeypatch, bash_record=True)
    out = tmp_path / "on.json"
    _child(f'{sys.executable} -c "{_ENV_DUMP}" {out}')

    seen = json.loads(out.read_text(encoding="utf-8"))
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy",
                 "https_proxy", "all_proxy"):
        assert name in seen, f"{name} missing from {sorted(seen)}"
        url = seen[name]
        assert url.startswith("http://127.0.0.1:") and url.count(":") == 2, (
            f"{name}={url!r} is not a loopback proxy URL")
        assert int(url.rsplit(":", 1)[1]) > 0, f"{name}={url!r} names no bound port"
    # Loopback, the IPv6 loopback, link-local (the cloud metadata address, and a local
    # probe's) and the three RFC1918 ranges. Named as a subset rather than as "loopback
    # is covered" because 172.16.0.0/12 is the range a LAN-attached service most likely
    # sits in, and it is the one a seed would silently drop.
    assert {"127.0.0.1", "::1", "10.0.0.0/8", "172.16.0.0/12",
            "192.168.0.0/16"} <= set(seen["NO_PROXY"].split(",")), seen["NO_PROXY"]
    assert set(seen["NO_PROXY"].split(",")) == set(proxy.NO_PROXY_SEED), (
        "the child's NO_PROXY is not the shipped seed")
    assert set(seen["no_proxy"].split(",")) == set(seen["NO_PROXY"].split(",")), (
        "the lowercase no_proxy differs from NO_PROXY; urllib reads only the lowercase")


def test_a_background_child_gets_the_recorder_too(tmp_path, db, monkeypatch):
    """Spawn site two. A detached child outlives the turn that spawned it and its output
    lands in a file the model reads back, so it is the least auditable of the three."""
    _flags(monkeypatch, bash_record=True)
    out = tmp_path / "bg.json"
    _child(f'{sys.executable} -c "{_ENV_DUMP}" {out}', background=True)

    deadline = time.monotonic() + 60
    while not out.is_file() and time.monotonic() < deadline:
        time.sleep(0.25)
    assert out.is_file(), "the background child never wrote its environment"
    seen = json.loads(out.read_text(encoding="utf-8"))
    assert "HTTP_PROXY" in seen and "ALL_PROXY" in seen, (
        f"a background child's environment holds no recorder: {sorted(seen)}")


def test_the_sandboxed_branch_never_hands_a_bench_child_the_recorder(monkeypatch):
    """The bwrap spawn site must stay clean, and the rule lives on the contextvar rather
    than on an argument a caller passes.

    `_tool_sandbox.current_sandboxed` is the box's existing fact for "this turn is a bench
    or eval arm", and `_bash` already reads it to choose its branch — so reading the same
    variable here cannot drift from the branch that decides whether bwrap runs, and a
    future fourth spawn site cannot forget to pass a boolean. A proxy hop inside a graded
    run would put this module's latency into a score. The bwrap branch cannot be spawned
    from a test (it needs the substrate), which is why this node checks the decision and
    the node below counts the branch structurally.
    """
    _flags(monkeypatch, bash_record=True)
    assert proxy.proxy_env_for_turn(), "the arm is inert even with the flag on"

    token = _tool_sandbox.current_sandboxed.set(True)
    try:
        assert proxy.proxy_env_for_turn() == {}, (
            "a sandboxed bench/eval turn would receive the proxy variables; a graded "
            "run's latency would then include this module")
    finally:
        _tool_sandbox.current_sandboxed.reset(token)


def test_every_spawn_site_routes_its_environment_through_the_recorder():
    """The bwrap branch is the one a test cannot spawn, so it is counted.

    Same shape as #3846's structural node, for the same reason: three sites, two shapes,
    and deleting one arm has to be a red node rather than a comment that stopped being
    true. `_vault_for_child` is the seam #3846 established as the one builder every Bash
    spawn routes through; #4044 composes its rule inside that builder, so this counts the
    composition once and the three call sites again — the number 3 is the source's spawn
    count, not a preference.
    """
    src = open(B.__file__, encoding="utf-8").read()
    assert re.search(r"def _vault_for_child\(.*?\n    return _egress_proxy_for_child\(",
                     src, re.S), (
        "_vault_for_child no longer composes the egress recorder — a Bash child would "
        "get its credentials filtered and its destinations unrecorded")
    assert len(re.findall(r"(?:env=|env\s*=)\s*_vault_for_child\(", src)) == 3, (
        "the count of spawn sites routing through the single builder is no longer 3; "
        "recount the create_subprocess sites in the source, do not edit this number")


# ── clause 3: what lands in egress_events, under which attribution ──────────

def test_an_absolute_uri_request_through_a_real_child_writes_one_bash_row(
        db, origin, monkeypatch):
    """`GET http://host/path` from a child → one row, the arm's `via`, the turn's scope.

    The child is driven by `urllib`, which honours the environment variables, so the
    request that arrives is the absolute-URI form a real client sends a proxy. The scope
    assertion is what this node exists for: it must be the SPAWNING TURN's
    `scope_context()` string — the same one `_child_grant_scope` keys a credential grant
    to — because the accept thread has no turn of its own, and a row attributed to
    nothing cannot answer "which run touched this host".
    """
    _flags(monkeypatch, bash_record=True)
    # Loopback is in NO_PROXY by design, so this node is the one place that drops it:
    # without that the child would bypass the recorder for a loopback destination and
    # the node would pass by never exercising the arm. The shipped seed is pinned by the
    # environment node above.
    monkeypatch.setattr(proxy, "NO_PROXY_SEED", ("::1",))
    url = f"http://127.0.0.1:{origin.server_address[1]}/probe"

    out = _child(f'{sys.executable} -c "{_ABSOLUTE_PROBE}" {url}')
    assert ORIGIN_BODY in out, f"the child did not get the origin's body: {out[:300]}"

    (row,) = _wait_for_rows(db, via=egress.VIA_BASH_PROXY)
    assert row["via"] == "bash_proxy"
    assert row["tool"] == "bash"
    assert row["destination"] == "http://127.0.0.1", row["destination"]
    assert row["host"] == egress.host_of(url), (
        f"host {row['host']!r} is not what egress.host_of derives for {url!r}")
    assert row["scope"] == GRANT_SCOPE == egress.scope_context()["scope"], (
        "the row does not name the spawning turn's grant scope")
    assert row["worker_source"] == SOURCE


def test_a_bare_connect_from_a_child_writes_one_row_naming_the_host(
        db, origin, monkeypatch):
    """`CONNECT host:port` — the first thing every TLS client sends — gets a row too.

    The child opens the tunnel by hand with no TLS, because the finding is the
    destination and not the session key: the recorder answers the CONNECT and then
    relays opaque bytes, never terminating TLS, so what it can name is `scheme://host`
    exactly as `egress.host_of` produces it for the four tools. This is the arm's only
    view of https traffic, and it is a host-only view by construction.
    """
    _flags(monkeypatch, bash_record=True)
    authority = f"127.0.0.1:{origin.server_address[1]}"

    out = _child(f'{sys.executable} -c "{_CONNECT_PROBE}" {authority}')
    assert "200" in out, f"the recorder did not establish the tunnel: {out[:300]}"

    (row,) = _wait_for_rows(db, via=egress.VIA_BASH_PROXY)
    assert row["host"] == egress.host_of(f"https://{authority}"), row["host"]
    assert row["destination"] == "https://127.0.0.1", row["destination"]
    assert row["decision"] == egress.DECISION_ALLOW
    assert row["scope"] == GRANT_SCOPE


def test_the_report_publishes_the_arm_row_beside_its_own_denominator(
        db, origin, monkeypatch):
    """`network_report()` is the sanctioned read path, so the row has to be visible there.

    The clause says "read back through egress.network_report()" and the reason is not
    ceremony: the dashboard and the arming report both read that function, so an arm whose
    rows appear only under a raw `SELECT` is an arm nobody acting on the inventory would
    ever see.
    """
    _flags(monkeypatch, bash_record=True)
    monkeypatch.setattr(proxy, "NO_PROXY_SEED", ("::1",))
    egress.guard("http_fetch", "https://tool-lane.example/x")
    url = f"http://127.0.0.1:{origin.server_address[1]}/probe"
    _child(f'{sys.executable} -c "{_ABSOLUTE_PROBE}" {url}')
    _wait_for_rows(db, via=egress.VIA_BASH_PROXY)

    report = egress.network_report()
    by_via = {v["via"]: v for v in report["per_via"]}
    assert by_via[egress.VIA_BASH_PROXY]["count"] == 1, report["per_via"]
    assert by_via[egress.VIA_BASH_PROXY]["distinct_hosts"] == 1, report["per_via"]
    assert by_via[egress.VIA_TOOL]["count"] == 1, report["per_via"]
    # The scope, not just the row: the clause asks for the attribution to be readable
    # through this function, and `per_scope` is tool-only by clause 5, so the arm's
    # attribution has a home of its own under `per_via` — otherwise the question "which
    # run's child reached a host that holds a grant" is answerable only by opening the
    # store, which is the thing the sanctioned accessor exists to prevent.
    bash_scopes = {s["scope"]: s["count"]
                   for s in by_via[egress.VIA_BASH_PROXY]["scopes"]}
    assert bash_scopes == {GRANT_SCOPE: 1}, by_via[egress.VIA_BASH_PROXY]["scopes"]
    # The same turn wrote one tool row and one Bash row, so `per_scope` proves which of
    # them it counts: one, the tool lane's. The Bash arm's attribution has a home of its
    # own under `per_via`, because clause 5 keeps the tool figure a tool figure.
    (scope_row,) = report["per_scope"]
    assert scope_row["scope"] == GRANT_SCOPE and scope_row["total"] == 1, \
        report["per_scope"]


# ── clause 4: the arm refuses nothing, and changes nothing ──────────────────

def test_the_recorder_changes_nothing_the_child_can_see(db, origin, monkeypatch):
    """Same command, same URL, armed and disarmed: identical result text.

    The result text is what the tool returns, so this compares the child's stdout AND the
    `[exit code: N]` framing a non-zero exit adds — one assertion over both halves of
    "the run behaved the same". This is the break-detector for the injection, run against
    a loopback origin rather than asserted as a hope: the recorded run goes through an
    extra process and still owes the child the origin's exact bytes and status.
    """
    _flags(monkeypatch, bash_record=False)
    url = f"http://127.0.0.1:{origin.server_address[1]}/probe"
    command = f'{sys.executable} -c "{_ABSOLUTE_PROBE}" {url}'

    unrecorded = _child(command)
    _flags(monkeypatch, bash_record=True)
    monkeypatch.setattr(proxy, "NO_PROXY_SEED", ("::1",))
    recorded = _child(command)

    assert ORIGIN_BODY in recorded, f"the recorded run lost the body: {recorded[:300]!r}"
    assert recorded == unrecorded, (
        "the recorded run differs from the unrecorded one — the proxy changed what the "
        f"child saw:\n  off: {unrecorded[:200]!r}\n  on : {recorded[:200]!r}")


def test_an_unreachable_origin_is_recorded_and_the_child_is_told(db, monkeypatch):
    """A destination that cannot be dialed is still a destination, and the failure stays
    the origin's.

    The row is written before the connect, so "the child tried an unlisted host" survives
    the host being dead — the case that matters for an endpoint taken down between the run
    and the reading. The child gets a 502 out of the recorder rather than a hang or a
    silently truncated stream, because a recorder that swallows a failure makes a failed
    run read as a successful one.
    """
    _flags(monkeypatch, bash_record=True)
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    dead_port = probe.getsockname()[1]
    probe.close()

    out = _child(f'{sys.executable} -c "{_RAW_ABSOLUTE_PROBE}" '
                 f'"http://127.0.0.1:{dead_port}/nope"')

    assert "502" in out, f"the child was not told the origin was unreachable: {out[:300]}"
    (row,) = _wait_for_rows(db, via=egress.VIA_BASH_PROXY)
    assert row["destination"] == "http://127.0.0.1"
    assert row["decision"] == egress.DECISION_ALLOW, (
        "a refused connection became a policy decision — the recorder has no such branch")


@pytest.mark.parametrize("enforce", [False, True], ids=["enforce-off", "enforce-on"])
def test_no_bash_row_can_carry_a_denial_whatever_enforce_says(
        db, origin, monkeypatch, enforce):
    """The recording arm has no denial in it, including in a process where `enforce` is
    true — not hypothetical: the Monday egress-window arm runs the aggregator with
    `LLOYD_EGRESS_ENFORCE=1` (`36c3607f`, 2026-11-02).

    Three things pinned together, because the tempting shortcut is any one of them: the
    child's request is NOT blocked with enforcement armed, the row says `allow`, and no
    row in the table carries `deny`. The first is why the arm is safe to switch on at all
    — a Bash destination turning into a refusal inside the canary's own enforce process
    would take the canary down.
    """
    _flags(monkeypatch, bash_record=True, enforce=enforce)
    monkeypatch.setattr(proxy, "NO_PROXY_SEED", ("::1",))
    url = f"http://127.0.0.1:{origin.server_address[1]}/probe"

    out = _child(f'{sys.executable} -c "{_ABSOLUTE_PROBE}" {url}')

    assert ORIGIN_BODY in out, f"the child was refused with enforce={enforce}: {out[:300]}"
    got = _wait_for_rows(db, via=egress.VIA_BASH_PROXY)
    assert len(got) == 1, "the recorded run wrote a different number of rows"
    assert {r["decision"] for r in got} == {egress.DECISION_ALLOW}
    assert not [r for r in rows(db) if r["decision"] == egress.DECISION_DENY], (
        "a row in the table carries a denial while only the recorder was armed")
