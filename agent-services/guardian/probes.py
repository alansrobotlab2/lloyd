"""HTTP health probes. Stdlib urllib only — no httpx, no requests.

A probe returns a verdict rather than raising, because "the endpoint refused
the connection" and "the endpoint returned 503" are both ordinary inputs to
the failure predicate, not exceptional conditions.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request


def probe(url: str, timeout: float) -> dict:
    """GET `url`. Returns {ok, status, body, error, latency_ms}.

    `ok` means HTTP 200 AND, when the body carries a `status` field, that it
    reads "ok". A 503 from either /health is a real negative: both the backend
    and the aggregator use 503 to mean degraded.
    """
    import time as _t
    started = _t.monotonic()
    out = {"ok": False, "status": None, "body": None, "error": None,
           "latency_ms": None, "kind": "ok"}
    try:
        req = urllib.request.Request(url, method="GET")
        # Never route a loopback health check through a proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=timeout) as resp:
            raw = resp.read(65536)
            out["status"] = resp.status
    except urllib.error.HTTPError as e:
        out["status"] = e.code
        try:
            raw = e.read(65536)
        except Exception:
            raw = b""
    except (urllib.error.URLError, socket.timeout, OSError, ValueError) as e:
        out["error"] = str(e)[:200]
        out["latency_ms"] = round((_t.monotonic() - started) * 1000, 2)
        # WHY it failed matters more than THAT it failed. A refused connection
        # means the process is gone. A timeout on a port that is still open
        # means the process is alive and busy — which for this backend is
        # routine: /health shares an event loop with the agent's own work, and
        # an autoresearch round fans out 77 trials through it.
        blob = f"{type(e).__name__}: {e}".lower()
        if "refused" in blob or "no route" in blob or "not known" in blob:
            out["kind"] = "refused"
        elif "timed out" in blob or "timeout" in blob:
            out["kind"] = "timeout"
        else:
            out["kind"] = "error"
        return out

    out["latency_ms"] = round((_t.monotonic() - started) * 1000, 2)
    try:
        out["body"] = json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        out["body"] = None
    body_status = (out["body"] or {}).get("status") if isinstance(out["body"], dict) else None
    out["ok"] = out["status"] == 200 and (body_status in (None, "ok"))
    out["kind"] = "ok" if out["ok"] else "http_error"
    return out


def wait_healthy(url: str, timeout_total: float, probe_timeout: float,
                 interval: float = 1.0) -> tuple[bool, dict]:
    """Poll until `url` is healthy or the budget expires."""
    import time as _t
    deadline = _t.monotonic() + timeout_total
    last: dict = {}
    while _t.monotonic() < deadline:
        last = probe(url, probe_timeout)
        if last["ok"]:
            return True, last
        _t.sleep(interval)
    return False, last
