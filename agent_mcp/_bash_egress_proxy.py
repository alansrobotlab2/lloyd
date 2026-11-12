"""#4044 — record the destinations a Bash-spawned process subtree contacts.

The gap this module closes, verified in the tree it was filed against: `egress.guard`
is reached from exactly four tools (`http_search`, `http_fetch`, `http_request`,
`browser_navigate`), while `agent_mcp/builtin_bash.py` touches the egress module at one
place only — `_child_grant_scope`, whose whole job is to read `egress.scope_context()`
so the child gets real credentials or placeholders. No Bash caller reaches `guard`. And
the sandbox that ordinary Bash runs under (`_path_sandbox`) keeps the network on
purpose, so a Bash child holding a granted credential can reach any public host and
that destination is written down nowhere: not in the transcript, and not in
`egress_events`. That is the shape the Zenity talk describes — an install script's
post-install process whose effects "never appear in the agent's own trace" — arriving on
a box whose own destination inventory (`harness.egress_policy.allow`) is seeded from a
table with no Bash arm in it.

**What this is: a recording arm.** It puts a loopback proxy in a Bash child's
environment and writes one `egress_events` row per destination it observes, under
`via='bash_proxy'`. It refuses nothing, ever: the forwarding path never consults the
allow-list, `enforce` is not read here, and `egress.record_bash_destination` has no
branch that computes a denial. Two independent reasons, both load-bearing:

1. The arm exists to *measure* what a Bash child reaches, and an arm that also gates
   would make the measurement and the policy one decision — the exact conflation
   `egress.py`'s own header names as the failure mode ("either you're approving a
   bunch of stuff or you're hoping your auto approval flow is configured properly").
   Arming `harness.egress_policy.enforce` stays #2702/#2787's call.
2. The injection puts a process this box owns onto the path of every unattended
   `curl`, `pip`, `yt-dlp` and `git` the fleet runs. A recorder that can refuse is an
   outage nobody voted for, and #2269 is the precedent: a Bash narrowing that broke
   `curl` health checks and was reverted.

Read "refuses nothing" as exactly what it is: a claim about POLICY, not a promise
that a process standing on the network path cannot fail a request. The hop can break
a child's call for a transport reason and does so by design in one place — a
destination this box cannot reach is answered `502 Bad Gateway` by `_do_connect` and
`_do_request`. What the 502 does not touch is the row: both handlers record BEFORE
they dial, so a destination is in the inventory whether the hop then succeeds or
not, which is the order that keeps a partial network from quietly shrinking the
table. And because the clause is about policy rather than about the hop, the tests
that pin it read the child's own stdout and exit status and not only the rows — the
only reason the bug in `_relay` was catchable at all.

**What this is NOT: a boundary.** A forced env-var proxy is advisory. curl, wget,
requests/httpx, pip, yt-dlp and git-over-https honour `HTTP_PROXY`/`HTTPS_PROXY`;
Node's native `fetch`, Go binaries (including `gh`), and any raw TCP or SSH connect
ignore them completely. So a `bash_proxy` row is proof a client that reads those
variables reached a host; the *absence* of rows is not proof a child reached nothing.
Presenting this as containment is the failure mode, and the per-class recall figure the
item asks for is what turns that caveat into a number. Root-grade alternatives are
unavailable on this box: `--unshare-net` inside bubblewrap cuts connectivity with no
route back without a root-created veth, and nftables/conntrack/auditd/eBPF all need
root, which the harness denies.

Scope, and why it is carried rather than looked up
--------------------------------------------------
Every row names the **spawning turn's** grant scope — the same string
`_child_grant_scope` keys a credential grant to. It cannot be resolved when the request
arrives: the accept loop is a thread of this process with no turn context in it, and a
lookup there would attribute every Bash destination on the box to whatever turn happened
to be in flight when the accept ran. So the scope is captured at the spawn site, at the
one moment the correct answer is actually in scope, and attached to the listener the
child was given. One listener per (scope, session) pair, bound to an ephemeral loopback
port — the port is never a config key, because `host`, `port` and `url` are denied
config leaves (`scripts/automod/spec.py` `CONFIG_DENIED_LEAVES`) and a fixed port would
collide with anything else on the box anyway.

The limit that follows from that design is named rather than hidden: within one
(scope, session) pair the recorder cannot tell *which* child opened a given connection,
because that needs the `/proc`-and-cgroup attribution the item defers. Per-connection
process identity is the second instrument's job, not this one's.
"""

from __future__ import annotations

import contextlib
import errno
import logging
import selectors
import socket
import threading
import urllib.parse
from dataclasses import dataclass

from agent_mcp import egress

logger = logging.getLogger("lloyd-bash-egress")

#: The variables a proxy-aware client reads. `NO_PROXY` is in the set because it is the
#: half that keeps the arm from breaking working paths: without it, a local-service
#: `curl http://127.0.0.1:8080/health` — the fleet watchdog's own probe, and every
#: canary — would be routed to us and depend on our forwarding to work.
PROXY_VARS: tuple[str, ...] = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")

#: `NO_PROXY`'s seed: loopback, the IPv6 loopback, link-local (the cloud metadata
#: address every credential-stealing example names, and also the one a local service
#: probe can land on) and the three RFC1918 ranges. A local or on-LAN service stays
#: directly reachable whether or not this module is healthy, which is the difference
#: between a recording arm and a self-inflicted outage.
NO_PROXY_SEED: tuple[str, ...] = (
    "localhost", "127.0.0.1", "::1", "0.0.0.0",
    "169.254.0.0/16", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
)

#: How many (scope, session) listeners one process may hold. Past this, a new Bash
#: child gets **today's environment** — no proxy variables at all — and coverage of
#: that turn is lost. Losing coverage is the chosen failure: the alternative is
#: evicting a listener a live child is still using, which turns a quiet recording arm
#: into a connection-refused error inside someone else's unattended run.
_MAX_LEASES = 256

#: A relay hops when neither side has sent a byte for this long. Bounded, so a client
#: that connected and went away cannot pin a thread and two sockets forever; long
#: enough that a slow but living download is never cut mid-stream.
_IDLE_TIMEOUT_S = 120.0

#: Cap on the request head we parse off the client. A proxy request line longer than
#: this is not a destination we can name.
_HEAD_MAX = 65536

_CONNECT_TIMEOUT_S = 10.0

_KEY = tuple[str, str]          # (scope, session_id) — the lease's identity


@dataclass(frozen=True)
class Attribution:
    """Who the children behind one listener are, captured at the spawn site.

    `worker_source` and `task_id` ride along so a Bash row is as attributable as a
    tool row. `session_id` is exact — it is part of the listener's key — but within
    one session the module cannot name the *process*, only the turn (see the module
    docstring), which is why it never claims a pid.
    """

    scope: str
    session_id: str
    worker_source: str = ""
    task_id: str = ""


@dataclass
class _Lease:
    key: _KEY
    attribution: Attribution
    sock: socket.socket
    port: int
    requests: int = 0


_lock = threading.Lock()
_leases: dict[_KEY, _Lease] = {}
_by_fd: dict[int, _Lease] = {}
_sel = selectors.DefaultSelector()
_wake_r, _wake_w = socket.socketpair()
_running = False
_loop_thread: threading.Thread | None = None


# ── the public surface the Bash spawn path uses ─────────────────────────────

def proxy_env_for_turn() -> dict[str, str]:
    """The proxy variables for the turn currently running a Bash call, or `{}`.

    `{}` means "this child gets exactly the environment it would have got yesterday",
    and it is returned whenever anything stands in the way: the flag is off (the
    shipped default), the session is a sandboxed bench or eval arm, the scope cannot be
    derived, no port can be bound, or the listener table is full. Never an exception:
    this is called on the spawn path of every Bash command on the box, and a recorder
    that raises has taken the Bash tool down.

    The sandboxed rule lives HERE rather than as a boolean the caller passes, so that a
    future fourth spawn site cannot forget to pass it: `_tool_sandbox.current_sandboxed`
    is the one thing that already means "this turn is a bench or eval arm", and a bench
    arm's environment is part of what is being measured — a proxy hop inside a graded
    run would put this module's latency into a score.
    """
    if not egress.bash_record_on():
        return {}
    try:
        from agent_mcp import _tool_sandbox
        if _tool_sandbox.current_sandboxed.get():
            return {}
    except Exception:  # noqa: BLE001 — no sandbox state is not a sandboxed session
        pass
    try:
        ctx = egress.scope_context()
    except Exception as exc:  # noqa: BLE001 — no scope means no attributable listener
        logger.debug("bash egress: scope unavailable (%s); spawning without a proxy",
                     exc)
        return {}
    session_id = ""
    try:
        session_id = egress._session_id()
    except Exception:  # noqa: BLE001 — session is attribution, not a gate
        pass
    attribution = Attribution(
        scope=str(ctx.get("scope") or ""),
        session_id=session_id,
        worker_source=str(ctx.get("worker_source") or ""),
        task_id=str(ctx.get("task_id") or ""),
    )
    port = _ensure_listener(attribution)
    if port is None:
        return {}
    base = f"http://127.0.0.1:{port}"
    return {
        "HTTP_PROXY": base, "HTTPS_PROXY": base, "ALL_PROXY": base,
        "NO_PROXY": ",".join(NO_PROXY_SEED),
        # The lowercase spellings, because the clients that honour them do not agree on
        # the case: curl and git read `http_proxy`, and some tools read only the
        # lowercase form while others read only the uppercase one. Setting both is the
        # only shape that does not silently choose a side.
        "http_proxy": base, "https_proxy": base, "all_proxy": base,
        "no_proxy": ",".join(NO_PROXY_SEED),
    }


def stop_all() -> None:
    """Close every listener and forget the table. For tests and for teardown.

    Idempotent, and never raises. Children still holding the variables in their
    environment will find the port closed; that is exactly what a child whose turn has
    ended is expected to handle, which is why the flag ships off and a live listener
    outlives its turn rather than being revoked mid-command.
    """
    global _running
    _running = False
    with contextlib.suppress(OSError):
        _wake_w.close()
    with _lock:
        for lease in list(_leases.values()):
            with contextlib.suppress(Exception):
                _sel.unregister(lease.sock)
            with contextlib.suppress(OSError):
                lease.sock.close()
        _leases.clear()
        _by_fd.clear()


# ── the listener ────────────────────────────────────────────────────────────

def _ensure_listener(attribution: Attribution) -> int | None:
    """The loopback port serving this attribution's children, creating it on first use.

    Bound to port 0 and read back — an ephemeral port, chosen by the kernel, never a
    config value. Listeners are keyed by (scope, session) and never evicted, because the
    one thing worse than a missing row is a child that loses the proxy mid-command and
    gets `connection refused` from a client that does not retry.
    """
    key = (attribution.scope, attribution.session_id)
    with _lock:
        existing = _leases.get(key)
        if existing is not None:
            return existing.port
        if len(_leases) >= _MAX_LEASES:
            logger.warning(
                "bash egress: %d listeners already open (cap %d) — spawning this "
                "child WITHOUT the recorder; its destinations will not be recorded",
                len(_leases), _MAX_LEASES)
            return None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 0))
            sock.listen(16)
            sock.setblocking(False)
        except OSError as exc:
            logger.warning("bash egress: cannot bind a loopback listener (%s) — "
                           "spawning without the recorder", exc)
            return None
        port = int(sock.getsockname()[1])
        lease = _Lease(key=key, attribution=attribution, sock=sock, port=port)
        _leases[key] = lease
        _by_fd[sock.fileno()] = lease
        _sel.register(sock, selectors.EVENT_READ, data=lease)
        _start_loop_locked()
        with contextlib.suppress(OSError):
            _wake_w.send(b"x")
        return port


def _start_loop_locked() -> None:
    """Start the accept thread if one is not running. Caller holds `_lock`."""
    global _running, _loop_thread
    if _loop_thread is not None and _loop_thread.is_alive():
        _running = True
        return
    _running = True
    _loop_thread = threading.Thread(target=_accept_loop, name="bash-egress-proxy",
                                    daemon=True)
    _loop_thread.start()


def _accept_loop() -> None:
    """Accept on every registered listener until `stop_all`.

    One thread for the whole process, and per-connection threads below it: the fleet
    opens tens of Bash children, and one thread per idle listener would be a standing
    tax on the aggregator for an arm that ships off.
    """
    _sel.register(_wake_r, selectors.EVENT_READ, data=None)
    while _running:
        try:
            events = _sel.select(timeout=1.0)
        except OSError:
            break
        for key, _mask in events:
            if key.data is None:                     # the wake pipe, or a closed fd
                with contextlib.suppress(OSError):
                    _wake_r.recv(4096)
                continue
            lease: _Lease = key.data
            try:
                conn, _addr = lease.sock.accept()
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EINTR):
                    logger.debug("bash egress: accept failed (%s)", exc)
                continue
            conn.setblocking(True)
            threading.Thread(target=_handle, args=(conn, lease), daemon=True).start()
    with contextlib.suppress(Exception):
        _sel.unregister(_wake_r)


# ── one proxied request ─────────────────────────────────────────────────────

def _handle(conn: socket.socket, lease: _Lease) -> None:
    """Observe one proxied request, record it, and forward it. In this order.

    The order is the point: a destination is recorded even when the origin is
    unreachable, because "the child tried to reach an unlisted host" is the finding, and
    a recorder that only writes rows for destinations that answered cannot see a
    blocked-or-dead endpoint at all.
    """
    try:
        head, rest = _read_head(conn)
        if not head:
            return
        lease.requests += 1
        line, _, headers_blob = head.partition(b"\r\n")
        try:
            fields = line.decode("latin-1", "replace").split(" ")
        except Exception:  # noqa: BLE001 — a head we cannot decode is not a destination
            return
        if len(fields) != 3:
            return                                  # not a request line: close, no row
        method, target = fields[0].upper(), fields[1]
        if method == "CONNECT":
            _do_connect(conn, target, lease)
        else:
            _do_request(conn, fields[0], target, headers_blob, rest, lease)
    except Exception as exc:  # noqa: BLE001 — the recorder never propagates to the child
        logger.debug("bash egress: request handling failed (%s)", exc)
    finally:
        with contextlib.suppress(OSError):
            conn.close()


def _read_head(conn: socket.socket) -> "tuple[bytes, bytes]":
    """Read up to the blank line. Returns `(head_without_terminator, leftover_bytes)`.

    The leftover is everything already buffered past the head — typically a POST body —
    and it is handed to the origin rather than re-read, because bytes taken off a socket
    cannot be un-taken and a body that is not forwarded is a request that was eaten.
    """
    buf = b""
    while b"\r\n\r\n" not in buf and len(buf) < _HEAD_MAX:
        try:
            chunk = conn.recv(8192)
        except OSError:
            break
        if not chunk:
            break
        buf += chunk
    head, sep, rest = buf.partition(b"\r\n\r\n")
    return (head, rest) if sep else (b"", b"")


def _authority_hostport(authority: str) -> "tuple[str, int]":
    """`host:port` from a CONNECT authority, with the default port 443.

    The host comes back through `egress.host_of`, so a bracketed IPv6 literal and a
    userinfo-bearing authority resolve exactly as they do on the tool lane — one host
    vocabulary, two observers. The port is read off the same authority the host came
    from, and a non-numeric tail is the DEFAULT rather than a guess: `host:` and
    `host:http` are both a client saying "the usual port", and a port taken from the
    wrong side of the last colon would dial a different service than the one the child
    asked for.
    """
    text = str(authority or "")
    host = egress.host_of(f"https://{text}")
    head, sep, tail = text.rpartition(":")
    if not sep:
        return host, 443
    return host, (int(tail) if tail.isdigit() else 443)


def _origin_hostport(target: str, headers_blob: bytes) -> "tuple[str, str, int]":
    """`(host, origin_form_path, port)` for an absolute-URI proxied request.

    Falls back to the `Host` header when the client sent origin-form (`GET /path`),
    which curl does for a proxy configured with a path-bearing URL. Without that
    fallback the destination would be unnamed — and an unnamed destination is the row
    this arm exists to prevent.
    """
    parsed = urllib.parse.urlparse(target)
    host = egress.host_of(target)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = target
    if not host:
        for name, value in _headers(headers_blob):
            if name == b"host":
                host = egress.host_of(f"//{value.decode('latin-1', 'replace')}")
                port = _host_header_port(value) or 80
                break
    if host and not parsed.netloc:
        path = f"/{target}" if not target.startswith("/") else target
        port = _host_header_port(_host_header(headers_blob)) or port
    origin_form = path or "/"
    if parsed.scheme and parsed.netloc:
        origin_form = (parsed.path or "/")
        if parsed.query:
            origin_form += f"?{parsed.query}"
    return host, origin_form, port


def _host_header_port(value: bytes) -> int:
    tail = str(value.decode("latin-1", "replace")).rsplit(":", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def _headers(headers_blob: bytes) -> "list[tuple[bytes, bytes]]":
    out = []
    for raw in headers_blob.split(b"\r\n"):
        if not raw:
            continue
        name, _, value = raw.partition(b":")
        out.append((name.strip().lower(), value.strip()))
    return out


def _host_header(headers_blob: bytes) -> bytes:
    for name, value in _headers(headers_blob):
        if name == b"host":
            return value
    return b""


def _record(lease: _Lease, destination: str, host: str) -> None:
    """Write the row, and never let the write decide whether the request proceeds."""
    if not egress.telemetry_on():
        return                      # telemetry off is the whole table being off
    attr = lease.attribution
    try:
        egress.record_bash_destination(
            destination=destination, host=host, scope=attr.scope,
            session_id=attr.session_id, worker_source=attr.worker_source,
            task_id=attr.task_id)
    except Exception as exc:  # noqa: BLE001 — recording is not on the request's critical path
        logger.warning("bash egress: could not record %s (%s)", destination, exc)


def _do_connect(conn: socket.socket, authority: str, lease: _Lease) -> None:
    """A bare `CONNECT host:port`: record the host, tunnel it, relay opaque bytes.

    The success path answers `200 Connection established` and then copies bytes it cannot
    read; an unreachable origin answers `502 Bad Gateway` (`origin is None`, just below).
    The row is written before either, so the destination is inventoried on both paths.
    

    TLS is not terminated — beyond the CONNECT handshake we move bytes we cannot read.
    That is the whole cost model of the arm: it knows the destination, not the payload,
    so a credential in a request body stays private while the host it went to is
    inventoried.
    """
    host, port = _authority_hostport(authority)
    if not host:
        return
    _record(lease, f"https://{host}", host)
    origin = _connect(host, port)
    if origin is None:
        with contextlib.suppress(OSError):
            conn.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n"
                         b"Connection: close\r\n\r\n")
        return
    try:
        conn.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
    except OSError:
        origin.close()
        return
    _relay(conn, origin)
    with contextlib.suppress(OSError):
        origin.close()


def _do_request(conn: socket.socket, method: str, target: str,
                headers_blob: bytes, rest: bytes, lease: _Lease) -> None:
    """An absolute-URI request (`GET http://host/path`): record, rewrite, forward.

    The rewrite is two lines of the request head and nothing else. `Connection: close`
    is set so the origin frames its own response and the relay terminates, and the
    `Proxy-*` headers are dropped because they are addressed to us, not to the origin.
    The request line's target becomes origin-form, which is what an origin server
    accepts; the status line and body that come back are the origin's own bytes,
    forwarded without touching them.

    The one way this path answers for itself is `502 Bad Gateway`, when `_connect` cannot
    reach the origin — after the record, never before it, so an unreachable host is still
    a host in the inventory.
    """
    host, origin_form, port = _origin_hostport(target, headers_blob)
    if not host:
        return
    parsed_scheme = urllib.parse.urlparse(target).scheme or "http"
    _record(lease, f"{parsed_scheme}://{host}", host)
    origin = _connect(host, port)
    if origin is None:
        with contextlib.suppress(OSError):
            conn.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n"
                         b"Connection: close\r\n\r\n")
        return
    kept = [(name, value) for name, value in _headers(headers_blob)
            if not name.startswith(b"proxy-") and name != b"connection"]
    head = (f"{method} {origin_form} HTTP/1.1\r\n"
            + "".join(f"{name.decode('latin-1')}: {value.decode('latin-1')}\r\n"
                      for name, value in kept)
            + "Connection: close\r\n\r\n").encode("latin-1")
    try:
        origin.sendall(head + rest)
    except OSError:
        origin.close()
        return
    _relay(conn, origin)
    with contextlib.suppress(OSError):
        origin.close()


def _connect(host: str, port: int) -> "socket.socket | None":
    try:
        return socket.create_connection((host, port), timeout=_CONNECT_TIMEOUT_S)
    except OSError as exc:
        logger.debug("bash egress: %s:%d unreachable (%s)", host, port, exc)
        return None


def _relay(a: socket.socket, b: socket.socket) -> None:
    """Copy bytes both ways until one side EOFs or the hop goes idle.

    Deliberately naive: no parsing, no buffering policy, no chunk re-encoding. Every
    byte the origin sends is the byte the client gets, which is what makes "the recorded
    run and the unrecorded run return the same stdout and status" a property of this
    function rather than a hope.
    """
    sel = selectors.DefaultSelector()
    try:
        # The socket travels in `key.data`, not `key.fileobj`: the selector library types
        # `fileobj` as "anything with a fileno", so `recv` is not a method it can be asked
        # for. `data` is the field that carries the registered object itself.
        sel.register(a, selectors.EVENT_READ, data=a)
        sel.register(b, selectors.EVENT_READ, data=b)
        open_sockets = 2
        while open_sockets:
            try:
                events = sel.select(_IDLE_TIMEOUT_S)
            except OSError:
                return
            if not events:
                return                      # stalled past the bound; drop the hop
            for key, _mask in events:
                # The socket travels in `key.data`, the field that carries the registered
                # object, rather than `key.fileobj`, which the selector library types as
                # "anything with a fileno" — so `recv` is not a method the type system will
                # let us ask for, and the `# type: ignore` that used to paper over it was
                # this function's only statement that a socket is what comes back.
                # `key.data` is never None on this path: the wake-pipe row is consumed in
                # `_accept_loop`, and both registrations below pass the socket itself.
                sock: socket.socket = key.data
                try:
                    data = sock.recv(65536)
                except OSError:
                    data = b""
                other = b if sock is a else a
                if not data:
                    open_sockets -= 1
                    with contextlib.suppress(Exception):
                        sel.unregister(sock)
                    with contextlib.suppress(OSError):
                        # Half-close the WRITE half of the socket that is STILL OPEN, named
                        # explicitly. An EOF on `sock` says that direction is finished, so
                        # the FIN has to leave through the other one — toward whoever is
                        # still reading. Half-closing `sock` instead cuts the direction that
                        # is still carrying the response body, which is how a recorder ends
                        # up writing a perfect row while handing the child
                        # `RemoteDisconnected`. The tests read the child's stdout and exit
                        # status rather than the rows, which is the only reason that bug is
                        # visible at all.
                        other.shutdown(socket.SHUT_WR)
                    continue
                try:
                    other.sendall(data)
                except OSError:
                    return
    finally:
        with contextlib.suppress(Exception):
            sel.close()
