"""Drive one real agent turn through a canary, end to end.

Purpose-built rather than repointing `tests/integration/smoke_observer_e2e.py`:
that script hardcodes the live LLOYD_HOME and backend URL, writes a session
file into the *live* sessions dir, and asserts on Inner Voice observations,
which adds LLM calls and a 30s poll loop the gate does not want. Its
SSE-draining shape is worth copying; its assertions are not.

The sentinel round-trip is the entire point. Asserting merely that "a tool was
called" would pass on a build whose dispatch is broken. Requiring a unique
string to travel user message → harness → MCP dispatch → real `Bash` execution
→ tool result → model → SSE `done` payload exercises the whole chain, over
HTTP, through the router stack, the session queue and the lazily-opened
`MCPPool`.

Runs at vLLM `priority: 1` so a real user's chat preempts the gate.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _post_stream(url: str, payload: dict, timeout: float):
    """Yield parsed SSE events from POST `url`."""
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as resp:
        event_type = None
        for raw in resp:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                blob = line[5:].strip()
                if not blob:
                    continue
                try:
                    data = json.loads(blob)
                except ValueError:
                    data = {"raw": blob}
                yield (event_type or data.get("type") or "message"), data
            elif not line:
                event_type = None


def run(*, backend: str, sessions_dir: Path, timeout: float = 150.0,
        model: str = "primary") -> dict:
    sentinel = f"LLOYD_CANARY_{secrets.token_hex(4)}"
    session_id = f"canary_{int(time.time())}_{secrets.token_hex(3)}"
    sessions_dir = Path(sessions_dir)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / f"{session_id}.json").write_text(json.dumps({
        "id": session_id,
        "title": "selfmod canary smoke",
        "model": model,
        "platform": "canary",
        "inner_voice": False,   # deterministic and cheap
        "messages": [],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }), encoding="utf-8")

    prompt = (f"Run `echo {sentinel}` via the Bash tool and tell me exactly "
              f"what it printed.")
    report: dict = {
        "ok": False, "sentinel": sentinel, "session_id": session_id,
        "tool_called": False, "tool_result_ok": False, "sentinel_in_response": False,
        "errors": [], "events": [],
    }

    started = time.time()
    try:
        for name, data in _post_stream(
            f"{backend}/api/message/stream",
            {"session_id": session_id, "text": prompt, "model": model, "priority": 1},
            timeout,
        ):
            report["events"].append(name)
            if name == "error":
                report["errors"].append(str(data)[:400])
            elif name == "tool_start" and data.get("name") == "Bash":
                report["tool_called"] = True
            elif name == "tool_complete":
                blob = json.dumps(data)
                if sentinel in blob:
                    report["tool_result_ok"] = True
                    if data.get("is_error"):
                        report["errors"].append("Bash tool_complete reported is_error")
            elif name == "done":
                if sentinel in json.dumps(data):
                    report["sentinel_in_response"] = True
                break
            if time.time() - started > timeout:
                report["errors"].append(f"smoke exceeded {timeout}s")
                break
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        report["errors"].append(f"HTTP {e.code}: {body}")
    except Exception as e:
        report["errors"].append(f"{type(e).__name__}: {str(e)[:300]}")

    report["duration_s"] = round(time.time() - started, 1)
    if not report["tool_called"]:
        report["errors"].append("no Bash tool_start event — the model never dispatched a tool")
    if not report["tool_result_ok"]:
        report["errors"].append("sentinel never appeared in a tool result — dispatch or "
                                "execution is broken")
    if not report["sentinel_in_response"]:
        report["errors"].append("sentinel never reached the final response")
    report["ok"] = not report["errors"]
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="One real agent turn against a canary")
    ap.add_argument("--backend", default="http://127.0.0.1:18080")
    ap.add_argument("--sessions-dir", required=True)
    ap.add_argument("--timeout", type=float, default=150.0)
    ap.add_argument("--model", default="primary")
    args = ap.parse_args(argv)

    report = run(backend=args.backend, sessions_dir=Path(args.sessions_dir),
                 timeout=args.timeout, model=args.model)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
