"""The browser's connection gate (`agent_mcp/_browser_proxy.py`).

Replaces the route interceptor on 2026-11-17: an interceptor shipped every
request body over Playwright's pipe (a 3.48 GB upload projected at 2.5 hours).
These drive the proxy over real loopback sockets, with no browser.
"""
import asyncio

import pytest

from agent_mcp._browser_proxy import GuardProxy, _split_host_port
from agent_mcp import browser as browser_module, egress


async def _origin(seen: list):
    async def handle(reader, writer):
        head = await reader.readuntil(b"\r\n\r\n")
        seen.append(head)
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
        await writer.drain()
        writer.close()
    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


async def _ask(proxy: GuardProxy, payload: bytes) -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", proxy.port)
    writer.write(payload)
    await writer.drain()
    data = await asyncio.wait_for(reader.read(), 10)
    writer.close()
    return data


@pytest.fixture
def started():
    made = []

    def make(decide):
        proxy = GuardProxy(decide).start()
        made.append(proxy)
        return proxy
    yield make
    for proxy in made:
        proxy.stop()


async def test_plain_http_is_forwarded_in_origin_form_one_request_per_connection(started):
    seen: list = []
    server, port = await _origin(seen)
    asked = []

    def decide(host, p, url):
        asked.append((host, p, url))
        return None, ("127.0.0.1",)
    proxy = started(decide)
    out = await _ask(proxy, (f"GET http://site.test:{port}/a?b=1 HTTP/1.1\r\nHost: site.test:{port}\r\n"
                             "Proxy-Connection: keep-alive\r\n\r\n").encode())
    server.close()
    assert out.endswith(b"ok")
    assert asked == [("site.test", port, f"http://site.test:{port}/a?b=1")]
    head = seen[0].lower()
    assert head.startswith(b"get /a?b=1 http/1.1\r\n")
    # One request per connection: a kept-alive socket could otherwise carry a
    # second request, for a host nobody judged, to the first host's address.
    assert b"connection: close" in head and b"proxy-connection" not in head


async def test_the_judged_address_is_the_dialled_address(started):
    """`unresolvable.invalid` reaches the origin only because decide named 127.0.0.1."""
    seen: list = []
    server, port = await _origin(seen)
    proxy = started(lambda h, p, u: (None, ("127.0.0.1",)))
    out = await _ask(proxy, f"CONNECT unresolvable.invalid:{port} HTTP/1.1\r\n\r\nGET / HTTP/1.1\r\n\r\n".encode())
    server.close()
    assert out.startswith(b"HTTP/1.1 200 Connection established\r\n\r\n")
    assert out.endswith(b"ok"), "the tunnel must splice bytes both ways"


async def test_a_refusal_dials_nothing(started):
    seen: list = []
    server, port = await _origin(seen)
    proxy = started(lambda h, p, u: ("Blocked — private/internal host", ()))
    for payload in (f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\n\r\n",
                    f"GET http://127.0.0.1:{port}/ HTTP/1.1\r\n\r\n"):
        out = await _ask(proxy, payload.encode())
        assert out.startswith(b"HTTP/1.1 403 "), out[:60]
        assert "Blocked — private/internal host".encode() in out
    server.close()
    assert seen == [], "a refused connection reached the origin"


async def test_a_guard_that_raises_fails_open_and_says_so(started, caplog):
    seen: list = []
    server, port = await _origin(seen)

    def decide(host, p, url):
        raise OSError("resolver exploded")
    proxy = started(decide)
    with caplog.at_level("WARNING", logger="lloyd-browser"):
        out = await _ask(proxy, f"GET http://127.0.0.1:{port}/ HTTP/1.1\r\n\r\n".encode())
    server.close()
    assert out.endswith(b"ok")
    warns = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("failed open" in w and "OSError" in w for w in warns), warns


async def test_an_unresolvable_name_and_a_non_http_target_are_answered_not_hung(started):
    proxy = started(lambda h, p, u: (None, ()))
    assert (await _ask(proxy, b"GET http://nowhere.invalid/ HTTP/1.1\r\n\r\n")).startswith(b"HTTP/1.1 502 ")
    assert (await _ask(proxy, b"GET ftp://x/ HTTP/1.1\r\n\r\n")).startswith(b"HTTP/1.1 400 ")


def test_host_port_spellings():
    assert _split_host_port("example.com:8443", 443) == ("example.com", 8443)
    assert _split_host_port("[::1]:443", 443) == ("::1", 443)
    assert _split_host_port("example.com", 80) == ("example.com", 80)


def test_guard_connect_refuses_the_lan_and_pins_what_it_allows(monkeypatch):
    monkeypatch.setenv("LLOYD_BROWSER_BLOCK_PRIVATE", "1")
    before = browser_module._block_log["seq"]
    reason, addrs = browser_module._guard_connect("192.168.1.1", 443, "https://192.168.1.1:443/")
    assert reason and "192.168.1.1" in reason and addrs == ()
    assert browser_module._block_log["seq"] == before + 1, "a refusal must be recorded"

    monkeypatch.setattr(egress, "_resolve_addrs", lambda h: ("93.184.216.34",))
    assert browser_module._guard_connect("public.test", 443) == (None, ("93.184.216.34",))
    # A name that answers with a LAN address is refused on the address.
    monkeypatch.setattr(egress, "_resolve_addrs", lambda h: ("10.0.0.7",))
    reason, addrs = browser_module._guard_connect("rebind.test", 80)
    assert reason and addrs == ()
    # The machine itself stays reachable: Playwright sends loopback here too.
    assert browser_module._guard_connect("localhost", 8080) == (None, ("127.0.0.1", "::1"))
    monkeypatch.setattr(egress, "_resolve_addrs", lambda h: ("::1",))
    assert browser_module._guard_connect("::1", 8080) == (None, ("::1",))
