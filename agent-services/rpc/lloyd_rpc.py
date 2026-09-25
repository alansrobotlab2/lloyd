#!/usr/bin/env python3
"""lloyd_rpc — call Lloyd's read-only tools from inside a Bash command (P9).

Stdlib only, so it runs under whatever python3 the shell finds. It works only
inside a Bash call the aggregator gave the environment to (`harness.rpc.enabled`,
a non-sandboxed session): the aggregator URL, the credential file, the session,
the parent Bash call id, the deny list and the deadline all arrive as
`LLOYD_*` variables (`app/harness/rpc_policy.py` names them).

Command line (`agent-services/bin/lloyd_rpc`):

    lloyd_rpc call Read '{"file_path": "/abs/path"}'
    lloyd_rpc call Grep - < args.json
    lloyd_rpc map Read '[{"file_path": "/a"}, {"file_path": "/b"}]' --concurrency 4

`call` prints the tool's text; exit 1 on a tool error, 2 on a refusal or a
usage error. `map` prints a JSON list of `{"ok": bool, "text": str}` in input
order.

Python:

    import sys; sys.path.insert(0, "<lloyd>/agent-services/rpc"); import lloyd_rpc
    text = lloyd_rpc.call("Grep", pattern="def main", path="/abs/dir")
    texts = lloyd_rpc.map("Read", [{"file_path": p} for p in paths], concurrency=4)

`call` raises `ToolError` for a tool error (the server's refusal included) and
`RpcError` for anything else; `map` returns the exception in place of each
failure rather than losing the rest.

The server is the authority on what may run — this client refuses early only to
save a round trip. It never mints a credential (`aggregator_auth.read_token(
publish=False)` semantics) and refuses a credential file anyone else could read.
"""
from __future__ import annotations

import concurrent.futures
import fnmatch
import itertools
import json
import os
import stat
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Iterable

# Must match app/harness/rpc_policy.py and agent_mcp/aggregator_auth.py.
ENV_URL = "LLOYD_RPC_URL"
ENV_TOKEN_FILE = "LLOYD_RPC_TOKEN_FILE"
ENV_SESSION_ID = "LLOYD_SESSION_ID"
ENV_TURN_ID = "LLOYD_TURN_ID"
ENV_PARENT_CALL_ID = "LLOYD_PARENT_CALL_ID"
ENV_EFFECT_SCOPE = "LLOYD_EFFECT_SCOPE"
ENV_SURFACE = "LLOYD_SURFACE"
ENV_DENY = "LLOYD_RPC_DENY"
ENV_DEADLINE = "LLOYD_RPC_DEADLINE"
ENV_DEPTH = "LLOYD_RPC_DEPTH"
TOKEN_ENV = "LLOYD_MCP_TOKEN"
AUTH_HEADER = "X-Lloyd-Aggregator-Token"

MIN_REMAINING_S = 1.0
MAX_CONCURRENCY = 8
_MIN_TOKEN_CHARS = 16

_ids = itertools.count(1)


class RpcError(Exception):
    """The call did not reach a tool answer: no env, refused client-side,
    transport failure, or a protocol error."""


class ToolError(RpcError):
    """The tool answered with an error (a server refusal is one)."""

    def __init__(self, tool: str, text: str):
        super().__init__(f"{tool}: {text}")
        self.tool = tool
        self.text = text


def _env(env: dict | None = None) -> dict:
    return env if env is not None else os.environ


def _context(env: dict | None = None) -> dict[str, str]:
    e = _env(env)
    url, parent = e.get(ENV_URL, ""), e.get(ENV_PARENT_CALL_ID, "")
    if not url or not parent:
        raise RpcError("lloyd_rpc: not inside a Lloyd Bash call with "
                       "harness.rpc.enabled (no LLOYD_RPC_URL / LLOYD_PARENT_CALL_ID)")
    return {k: e.get(k, "") for k in (ENV_URL, ENV_TOKEN_FILE, ENV_SESSION_ID,
                                      ENV_TURN_ID, ENV_PARENT_CALL_ID,
                                      ENV_EFFECT_SCOPE, ENV_SURFACE, ENV_DENY,
                                      ENV_DEADLINE, ENV_DEPTH)}


def _token(env: dict | None = None) -> str:
    """The aggregator credential: `LLOYD_MCP_TOKEN`, else the token file, which
    must be a regular file of ours with no group/other bits."""
    e = _env(env)
    direct = (e.get(TOKEN_ENV) or "").strip()
    if direct:
        return direct
    path = e.get(ENV_TOKEN_FILE, "")
    if not path:
        raise RpcError("lloyd_rpc: no credential (LLOYD_RPC_TOKEN_FILE unset)")
    try:
        st = os.stat(path)
    except OSError as exc:
        raise RpcError(f"lloyd_rpc: credential file unreadable: {exc}") from None
    if not stat.S_ISREG(st.st_mode) or st.st_mode & 0o077:
        raise RpcError(f"lloyd_rpc: refusing credential file {path}: it must be "
                       "a regular file with mode 0600")
    try:
        with open(path, encoding="utf-8") as fh:
            token = fh.read().strip()
    except OSError as exc:
        raise RpcError(f"lloyd_rpc: credential file unreadable: {exc}") from None
    if len(token) < _MIN_TOKEN_CHARS:
        raise RpcError("lloyd_rpc: credential file holds no credential")
    return token


def _deny(ctx: dict) -> list[str]:
    try:
        value = json.loads(ctx.get(ENV_DENY) or "[]")
    except ValueError:
        return []
    return [str(v) for v in value] if isinstance(value, list) else []


def _denied(tool: str, deny: Iterable[str]) -> bool:
    for pat in deny:
        if pat == tool or (any(c in pat for c in "*?[") and fnmatch.fnmatchcase(tool, pat)):
            return True
    return False


def remaining_seconds(env: dict | None = None, now: float | None = None) -> float:
    """Seconds left before the parent Bash call's rpc deadline (inf if none)."""
    raw = _env(env).get(ENV_DEADLINE, "")
    try:
        deadline = float(raw)
    except (TypeError, ValueError):
        return float("inf")
    return deadline - (time.time() if now is None else now)


def call(tool: str, args: dict | None = None, /, *, timeout: float | None = None,
         env: dict | None = None, **kwargs: Any) -> str:
    """Call one read-only tool; return its text. Keyword arguments are the
    tool's arguments (merged over `args`)."""
    ctx = _context(env)
    arguments = dict(args or {})
    arguments.update(kwargs)
    if _denied(tool, _deny(ctx)):
        raise ToolError(tool, "Tool call denied: lloyd_rpc: not available in this "
                              "turn (client-side, from LLOYD_RPC_DENY)")
    left = remaining_seconds(env)
    if left < MIN_REMAINING_S:
        raise RpcError(f"lloyd_rpc: {tool} would outlive the Bash call's deadline "
                       f"({max(left, 0.0):.1f} s left); not sent")
    budget = left if timeout is None else min(float(timeout), left)

    meta: dict[str, Any] = {
        "lloyd/session_id": ctx[ENV_SESSION_ID],
        "lloyd/rpc_parent_call_id": ctx[ENV_PARENT_CALL_ID],
        "lloyd/rpc_depth": int(ctx[ENV_DEPTH] or 1),
    }
    for key, name in (("lloyd/turn_id", ENV_TURN_ID),
                      ("lloyd/effect_scope", ENV_EFFECT_SCOPE),
                      ("lloyd/surface", ENV_SURFACE)):
        if ctx[name]:
            meta[key] = ctx[name]
    body = json.dumps({
        "jsonrpc": "2.0", "id": next(_ids), "method": "tools/call",
        "params": {"name": tool, "arguments": arguments, "_meta": meta},
    }).encode("utf-8")
    req = urllib.request.Request(ctx[ENV_URL], data=body, method="POST", headers={
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        AUTH_HEADER: _token(env),
    })
    try:
        with urllib.request.urlopen(req, timeout=budget) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:300].decode("utf-8", "replace")
        raise RpcError(f"lloyd_rpc: {tool}: HTTP {exc.code}: {detail}") from None
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise RpcError(f"lloyd_rpc: {tool}: {exc}") from None
    return _parse(tool, raw)


def _parse(tool: str, raw: bytes) -> str:
    text = raw.decode("utf-8", "replace").strip()
    if text.startswith("event:") or text.startswith("data:"):
        text = "\n".join(line[5:].strip() for line in text.splitlines()
                         if line.startswith("data:"))
    try:
        payload = json.loads(text)
    except ValueError:
        raise RpcError(f"lloyd_rpc: {tool}: unreadable response: {text[:200]}") from None
    if "error" in payload:
        err = payload["error"] or {}
        raise RpcError(f"lloyd_rpc: {tool}: {err.get('message') or err}")
    result = payload.get("result") or {}
    out = "\n".join(str(b.get("text", "")) for b in result.get("content") or []
                    if isinstance(b, dict) and b.get("type") == "text")
    if result.get("isError") or result.get("is_error"):
        raise ToolError(tool, out)
    return out


def map(tool: str, args_list: Iterable[dict], *, concurrency: int = 4,  # noqa: A001
        timeout: float | None = None, env: dict | None = None) -> list:
    """`call(tool, args)` for each, `concurrency` at a time, in input order.
    A failure is returned in its place as the exception, never raised."""
    items = [dict(a or {}) for a in args_list]
    workers = max(1, min(int(concurrency or 1), MAX_CONCURRENCY))

    def one(a: dict):
        try:
            return call(tool, a, timeout=timeout, env=env)
        except RpcError as exc:
            return exc

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, items))


def _load_json(arg: str) -> Any:
    text = sys.stdin.read() if arg == "-" else arg
    return json.loads(text) if text.strip() else {}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    usage = ("usage: lloyd_rpc call TOOL [JSON|-]\n"
             "       lloyd_rpc map TOOL JSON_LIST|- [--concurrency N]")
    if len(argv) < 2 or argv[0] not in ("call", "map"):
        print(usage, file=sys.stderr)
        return 2
    verb, tool, rest = argv[0], argv[1], argv[2:]
    concurrency = 4
    if "--concurrency" in rest:
        i = rest.index("--concurrency")
        try:
            concurrency = int(rest[i + 1])
        except (IndexError, ValueError):
            print(usage, file=sys.stderr)
            return 2
        rest = rest[:i] + rest[i + 2:]
    try:
        payload = _load_json(rest[0]) if rest else ({} if verb == "call" else [])
    except ValueError as exc:
        print(f"lloyd_rpc: arguments are not JSON: {exc}", file=sys.stderr)
        return 2
    if verb == "call":
        if not isinstance(payload, dict):
            print("lloyd_rpc: call takes a JSON object", file=sys.stderr)
            return 2
        try:
            print(call(tool, payload))
        except ToolError as exc:
            print(exc.text, file=sys.stderr)
            return 2 if "Tool call denied" in exc.text else 1
        except RpcError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        return 0
    if not isinstance(payload, list):
        print("lloyd_rpc: map takes a JSON list of objects", file=sys.stderr)
        return 2
    results = map(tool, payload, concurrency=concurrency)
    print(json.dumps([{"ok": not isinstance(r, Exception),
                       "text": (r.text if isinstance(r, ToolError) else str(r))}
                      for r in results], ensure_ascii=False))
    return 1 if any(isinstance(r, Exception) for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
