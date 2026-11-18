"""The browser's connection gate: a loopback forward proxy that asks before it dials.

Until 2026-11-17 the private-host guard was a Playwright route interceptor
(`context.route("**/*", …)`). Any interceptor makes Chromium pause EVERY request
and ship its whole body, base64, over the automation pipe to Playwright's driver
before the request may leave. Measured that day on this box, same Chromium, a
600 MiB upload to a loopback sink:

    chunk     no interceptor    with interceptor
    8 MiB     771 MiB/s         15 MiB/s
    64 MiB    927 MiB/s         1 MiB/s

which is what turned a ten-minute 3.48 GB YouTube upload into a projected 2.5
hours (NIC at ~440 KiB/s, the driver holding 1.9 GB of chunks it had been
handed). An interceptor also switches the HTTP cache off and puts a Python round
trip in front of every subresource.

So the decision moved to where the socket is opened. Chromium is launched with
this proxy as its only route out (Playwright forces loopback through it too),
and for each connection the proxy asks `decide(host, port, url)`:

- a reason  -> `403`, nothing is dialled;
- addresses -> it dials one of THOSE addresses, never the name again, so the
  address that was judged is the address that is reached (the interceptor judged
  a name Chromium then resolved for itself);
- a raise   -> fail open with a warning, the same trade the interceptor made: a
  bug here should cost depth, not every page load. The entry-point checks and
  `_enforce_landing` still stand.

Bodies never cross a tool boundary: HTTPS is a `CONNECT` tunnel spliced byte for
byte, and plain HTTP is forwarded one request per connection (`Connection:
close`), so a kept-alive socket can never carry a second request to a host
nobody judged. A redirect is a new connection and is judged like any other,
which the interceptor never saw.

Runs on its own thread and loop: a relay must not wait on the aggregator's loop,
and the aggregator's loop must not relay a download.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import urllib.parse
from typing import Callable

logger = logging.getLogger("lloyd-browser")

Decide = Callable[[str, int, str], "tuple[str | None, tuple[str, ...]]"]

_HEAD_LIMIT = 64 * 1024
_HEAD_TIMEOUT_S = 30.0
_DIAL_TIMEOUT_S = 15.0
_CHUNK = 256 * 1024
# After the client hangs up, how long the upstream gets to finish talking.
_LINGER_S = 30.0


def _split_host_port(text: str, default_port: int) -> tuple[str, int]:
    """`host:port`, `[v6]:port`, `host` -> (host without brackets, port)."""
    text = text.strip()
    if text.startswith("["):
        host, _, rest = text[1:].partition("]")
        port = rest[1:] if rest.startswith(":") else ""
    elif text.count(":") == 1:
        host, _, port = text.partition(":")
    else:
        host, port = text, ""
    return host, int(port) if port.isdigit() else default_port


def _refusal(status: str, reason: str) -> bytes:
    body = reason.encode("utf-8", "replace")
    return (f"HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode() + body


async def _pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await reader.read(_CHUNK)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionError, OSError, asyncio.IncompleteReadError):
        pass
    try:
        if writer.can_write_eof():
            writer.write_eof()
    except (ConnectionError, OSError, RuntimeError):
        pass


async def _splice(cr, cw, ur, uw) -> None:
    up = asyncio.ensure_future(_pump(cr, uw))
    down = asyncio.ensure_future(_pump(ur, cw))
    try:
        await asyncio.wait({up, down}, return_when=asyncio.FIRST_COMPLETED)
        if not down.done():
            # The client is done sending; let the answer finish, within reason.
            await asyncio.wait({down}, timeout=_LINGER_S)
    finally:
        for task in (up, down):
            if not task.done():
                task.cancel()
        await asyncio.gather(up, down, return_exceptions=True)


class GuardProxy:
    """One listening socket on 127.0.0.1, one `decide` asked per connection."""

    def __init__(self, decide: Decide):
        self._decide = decide
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: asyncio.AbstractServer | None = None
        self._thread: threading.Thread | None = None
        self.port = 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> "GuardProxy":
        """Bind and serve on a daemon thread. Raises if the socket cannot bind."""
        if self._thread is not None:
            return self
        ready = threading.Event()
        failure: list[BaseException] = []

        def run() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                self._server = loop.run_until_complete(
                    asyncio.start_server(self._serve, "127.0.0.1", 0, limit=_HEAD_LIMIT))
                self.port = self._server.sockets[0].getsockname()[1]
                self._loop = loop
            except BaseException as exc:  # noqa: BLE001 — reported to the starter
                failure.append(exc)
                ready.set()
                return
            ready.set()
            try:
                loop.run_forever()
            finally:
                loop.close()

        self._thread = threading.Thread(target=run, name="browser-guard-proxy", daemon=True)
        self._thread.start()
        ready.wait(10)
        if failure or not self.port:
            self._thread = None
            raise RuntimeError(f"browser guard proxy did not start: "
                               f"{failure[0] if failure else 'timed out'}")
        return self

    def stop(self) -> None:
        loop, server, thread = self._loop, self._server, self._thread
        self._loop = self._server = self._thread = None
        if loop is None:
            return

        def _halt() -> None:
            if server is not None:
                server.close()
            for task in asyncio.all_tasks(loop):
                task.cancel()
            loop.call_soon(loop.stop)

        loop.call_soon_threadsafe(_halt)
        if thread is not None:
            thread.join(5)

    # ── one connection ──────────────────────────────────────────────────────

    async def _serve(self, cr: asyncio.StreamReader, cw: asyncio.StreamWriter) -> None:
        uw = None
        try:
            try:
                head = await asyncio.wait_for(cr.readuntil(b"\r\n\r\n"), _HEAD_TIMEOUT_S)
            except (asyncio.IncompleteReadError, asyncio.LimitOverrunError,
                    asyncio.TimeoutError, ConnectionError, OSError):
                return
            line, _, rest = head.partition(b"\r\n")
            try:
                method, target, version = line.decode("latin-1").split(" ", 2)
            except ValueError:
                cw.write(_refusal("400 Bad Request", "malformed request line"))
                return
            tunnel = method.upper() == "CONNECT"
            if tunnel:
                host, port = _split_host_port(target, 443)
                url = f"https://{target}/"
                forward = b""
            else:
                parsed = urllib.parse.urlsplit(target)
                if parsed.scheme != "http" or not parsed.hostname:
                    cw.write(_refusal("400 Bad Request", "this proxy forwards http:// and CONNECT only"))
                    return
                host, port, url = parsed.hostname, parsed.port or 80, target
                path = (parsed.path or "/") + (f"?{parsed.query}" if parsed.query else "")
                kept = [h for h in rest.split(b"\r\n") if h and not h.lower().startswith(
                    (b"proxy-connection:", b"connection:", b"proxy-authorization:"))]
                forward = b"\r\n".join(
                    [f"{method} {path} {version}".encode("latin-1"), *kept,
                     b"Connection: close", b"", b""])

            loop = asyncio.get_running_loop()
            try:
                reason, addrs = await loop.run_in_executor(None, self._decide, host, port, url)
            except Exception as exc:  # noqa: BLE001 — the documented fail-open
                logger.warning("browser: connection guard failed open on %s: %s",
                               url[:120], f"{type(exc).__name__}: {exc}")
                reason, addrs = None, (host,)
            if reason:
                cw.write(_refusal("403 Forbidden", reason))
                return
            if not addrs:
                cw.write(_refusal("502 Bad Gateway", f'"{host}" did not resolve'))
                return

            ur = None
            for addr in addrs:
                try:
                    ur, uw = await asyncio.wait_for(
                        asyncio.open_connection(addr, port, limit=_CHUNK), _DIAL_TIMEOUT_S)
                    break
                except (OSError, asyncio.TimeoutError):
                    continue
            if ur is None or uw is None:
                cw.write(_refusal("502 Bad Gateway", f'could not connect to "{host}"'))
                return
            if tunnel:
                cw.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
                await cw.drain()
            else:
                uw.write(forward)
                await uw.drain()
            await _splice(cr, cw, ur, uw)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — one connection, never the proxy
            logger.debug("browser guard proxy: connection dropped: %s", type(exc).__name__)
        finally:
            for writer in (uw, cw):
                if writer is None:
                    continue
                try:
                    if writer is cw:
                        await asyncio.wait_for(writer.drain(), 5)
                    writer.close()
                except Exception:  # noqa: BLE001
                    pass
