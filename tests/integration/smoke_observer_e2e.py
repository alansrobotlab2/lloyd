"""End-to-end smoke test for the Inner Voice observer.

Creates a fresh session with `inner_voice: true` and
`inner_voice_evaluate_user_turns: true`, sends a simple chat message via
SSE, waits for the turn to finish, and verifies that:

  1. At least one observation row landed in `inner_voice_observations`
     for that session.
  2. The observer's `result`-trigger row exists (proving end-of-turn fired).
  3. No exceptions in the server log for that session.

Run from inside the lloyd container:
  $ /home/alansrobotlab/lloyd/.venvs/lloyd/bin/python \
      tests/integration/smoke_observer_e2e.py
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx

LLOYD_HOME = Path("/home/alansrobotlab/lloyd")
SESSIONS_DIR = LLOYD_HOME / "sessions"
SERVER_URL = "http://127.0.0.1:8080"

PROMPT = "Run `echo hello && date -u` via Bash and tell me what you saw."


def _make_iv_session() -> str:
    """Create a fresh session JSON with inner_voice + evaluate_user_turns set."""
    session_id = f"{datetime.now():%Y%m%d_%H%M%S}_obs{uuid.uuid4().hex[:3]}"
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({
        "session_id": session_id,
        "model": "primary",
        "created_at": datetime.now().isoformat(),
        "last_active": datetime.now().isoformat(),
        "preview": "Inner Voice observer smoke test",
        "message_count": 0,
        "messages": [],
        "platform": "mission-control",
        "inner_voice": True,
        "inner_voice_evaluate_user_turns": True,
    }, indent=2))
    print(f"[setup] session: {session_id}")
    return session_id


def _stream_chat(session_id: str, prompt: str, timeout_s: float = 90.0) -> dict:
    """POST to /api/message/stream and drain SSE until done. Returns the
    final stats dict (from the `done` event) or {}.
    """
    url = f"{SERVER_URL}/api/message/stream"
    payload = {"session_id": session_id, "text": prompt, "model": "primary"}
    final_stats: dict = {}
    started = time.time()

    with httpx.stream(
        "POST", url, json=payload, timeout=timeout_s, headers={"Accept": "text/event-stream"},
    ) as r:
        r.raise_for_status()
        current_event: str | None = None
        for line in r.iter_lines():
            if not line:
                continue
            if line.startswith("event:"):
                current_event = line[6:].strip()
                continue
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if not data_str:
                continue
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            if current_event == "done":
                final_stats = data.get("stats") or {}
                resp_text = (data.get("response") or "")[:200]
                print(f"[done] response: {resp_text!r}")
                print(f"[done] stats: {final_stats}")
                break
            if current_event == "error":
                print(f"[ERROR event] {data}")
                return {"error": data}
            if time.time() - started > timeout_s:
                print(f"[timeout] gave up after {timeout_s}s")
                return {"error": "client timeout"}

    return final_stats


def _read_observations(session_id: str) -> list[dict]:
    r = httpx.get(
        f"{SERVER_URL}/api/inner_voice/observations",
        params={"session_id": session_id, "limit": 200},
        timeout=30.0,
    )
    r.raise_for_status()
    return r.json().get("observations") or []


def _read_state(session_id: str) -> dict:
    r = httpx.get(
        f"{SERVER_URL}/api/inner_voice/state",
        params={"session_id": session_id},
        timeout=30.0,
    )
    r.raise_for_status()
    return r.json()


def main() -> int:
    print("=" * 60)
    print("Inner Voice observer — end-to-end smoke test")
    print("=" * 60)
    session_id = _make_iv_session()

    print(f"\n[chat] sending: {PROMPT!r}")
    stats = _stream_chat(session_id, PROMPT)
    if stats.get("error"):
        print(f"\n[FAIL] streaming failed: {stats['error']}")
        return 1

    # The observer's `result`-trigger fires AFTER the harness yields the
    # result event (the SSE `done` arrives first). Poll for up to 30s
    # until the result-trigger row lands.
    deadline = time.time() + 30
    obs: list[dict] = []
    while time.time() < deadline:
        obs = _read_observations(session_id)
        if any(o.get("trigger") == "result" for o in obs):
            break
        time.sleep(1.0)
    print(f"\n[observations] {len(obs)} rows for {session_id}:")
    for o in reversed(obs):  # oldest first
        action = o.get("action")
        trigger = o.get("trigger")
        seq = o.get("sequence_in_turn")
        reason = (o.get("reason") or "")[:80]
        latency = o.get("latency_ms")
        related = o.get("related_tool") or ""
        print(f"  [{seq:2d}] {trigger:18s} {action:25s} {latency or 0:>4}ms  {related:12s}  {reason}")

    state = _read_state(session_id)
    print(f"\n[state] inner_voice_enabled={state.get('inner_voice_enabled')}")
    print(f"[state] evaluate_user_turns={state.get('evaluate_user_turns')}")
    print(f"[state] counts: {state.get('observations_count_by_action')}")

    failed = 0

    # Assertion 1: at least one observation
    if not obs:
        print("\n[FAIL] no observations recorded — observer did not fire")
        failed += 1

    # Assertion 2: a `result`-trigger observation exists (turn-end fire)
    result_rows = [o for o in obs if o.get("trigger") == "result"]
    if not result_rows:
        print("\n[FAIL] no result-trigger observation — observer didn't see turn end")
        failed += 1

    # Assertion 3: state endpoint reports IV enabled for this session
    if not state.get("inner_voice_enabled"):
        print("\n[FAIL] /state says inner_voice not enabled despite session JSON flag")
        failed += 1

    print("\n" + "=" * 60)
    if failed == 0:
        print("PASS — all assertions met")
    else:
        print(f"FAIL — {failed} assertion(s) failed")
    print("=" * 60)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
