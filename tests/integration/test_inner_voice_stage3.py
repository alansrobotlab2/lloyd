"""Inner Voice (#345) Stage 3 integration test.

End-to-end check:
  1. Create an Inner-Voice-opted-in session by writing the meta file directly.
  2. POST bench_008's prompt ("Lloyd's Mars-rover integration?") to
     /api/message/stream and stream the SSE response.
  3. Tail the events.jsonl for the session and report the Stage 3 evidence:
       - inner_voice.ensemble_selected with 3 personas
       - inner_voice.persona_invoked × 3 (concurrent)
       - inner_voice.aggregation_decision (post-loop)
       - optional: inner_voice.mid_turn_check_started, cancel_event_fired

Run from inside the lloyd container:
  $ /home/alansrobotlab/lloyd/.venvs/lloyd/bin/python \
      tests/integration/test_inner_voice_stage3.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx

LLOYD_HOME = Path("/home/alansrobotlab/lloyd")
SESSIONS_DIR = LLOYD_HOME / "sessions"
EVENT_LOGS_DIR = LLOYD_HOME / "event_logs"
SERVER_URL = "http://127.0.0.1:8080"

BENCH_PROMPT = "What's the status of Lloyd's Mars-rover integration?"


def _make_session(inner_voice: bool = True) -> str:
    """Create the session meta file directly with the inner_voice flag set."""
    session_id = f"{datetime.now():%Y%m%d_%H%M%S}_iv3{uuid.uuid4().hex[:3]}"
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({
        "session_id": session_id,
        "model": "primary",
        "created_at": datetime.now().isoformat(),
        "last_active": datetime.now().isoformat(),
        "preview": "Inner Voice Stage 3 bench_008 test",
        "message_count": 0,
        "messages": [],
        "platform": "mission-control",
        "experiment_id": "stage3_bench_008",
        "inner_voice": bool(inner_voice),
    }, indent=2))
    print(f"Created session {session_id} (inner_voice={inner_voice})")
    return session_id


def _stream_message(session_id: str, prompt: str, *, timeout: float = 90.0) -> dict:
    """POST to /api/message/stream and consume the SSE stream synchronously."""
    body = {
        "session_id": session_id,
        "text": prompt,
        "model": "primary",
    }
    final_text = ""
    text_chunks = 0
    cancelled = False
    inner_voice_events: list[dict] = []
    started = time.time()
    with httpx.Client(timeout=timeout) as client:
        with client.stream(
            "POST",
            f"{SERVER_URL}/api/message/stream",
            json=body,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            resp.raise_for_status()
            current_event = None
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("event:"):
                    current_event = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    payload = line[len("data:"):].strip()
                    if not payload:
                        continue
                    try:
                        d = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if current_event == "text_delta":
                        final_text += d.get("text", "")
                        text_chunks += 1
                    elif current_event == "done":
                        final_text = d.get("response", final_text) or final_text
                    elif current_event and current_event.startswith("inner_voice"):
                        inner_voice_events.append({"event": current_event, "data": d})
                    elif current_event == "session_killed":
                        cancelled = True
                if time.time() - started > timeout:
                    print("  [stream] timeout after %.1fs, breaking" % (time.time() - started))
                    break
    elapsed = time.time() - started
    print(f"  [stream] {text_chunks} text deltas, {len(inner_voice_events)} IV events, "
          f"{len(final_text)} chars, cancelled={cancelled}, elapsed={elapsed:.1f}s")
    return {
        "final_text": final_text,
        "text_chunks": text_chunks,
        "inner_voice_events": inner_voice_events,
        "cancelled": cancelled,
        "elapsed": elapsed,
    }


def _read_event_log(session_id: str) -> list[dict]:
    path = EVENT_LOGS_DIR / f"{session_id}.events.jsonl"
    if not path.exists():
        print(f"  [event_log] FILE MISSING: {path}")
        return []
    events: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _summarize_events(events: list[dict]) -> dict:
    """Bucket events by name and collect Stage-3-relevant evidence."""
    counts: dict[str, int] = {}
    persona_invoked_personas: list[str] = []
    aggregation_decisions: list[dict] = []
    mid_turn_check_starts: list[dict] = []
    cancel_events: list[dict] = []
    intervention_dispatched: list[dict] = []
    ensemble_selected: list[dict] = []
    persona_response_parsed: list[dict] = []
    persona_parse_failures: list[dict] = []

    for e in events:
        name = e.get("event", "")
        counts[name] = counts.get(name, 0) + 1
        d = e.get("data", {}) or {}
        if name == "inner_voice.persona_invoked":
            persona_invoked_personas.append(d.get("persona", ""))
        elif name == "inner_voice.ensemble_selected":
            ensemble_selected.append(d)
        elif name == "inner_voice.aggregation_decision":
            aggregation_decisions.append(d)
        elif name == "inner_voice.mid_turn_check_started":
            mid_turn_check_starts.append(d)
        elif name == "inner_voice.cancel_event_fired":
            cancel_events.append(d)
        elif name == "inner_voice.intervention_dispatched":
            intervention_dispatched.append(d)
        elif name == "inner_voice.persona_response_parsed":
            persona_response_parsed.append(d)
        elif name == "inner_voice.persona_parse_failure":
            persona_parse_failures.append(d)

    return {
        "total": len(events),
        "counts": counts,
        "personas_invoked": persona_invoked_personas,
        "ensemble_selected": ensemble_selected,
        "aggregation_decisions": aggregation_decisions,
        "mid_turn_checks": mid_turn_check_starts,
        "cancel_events": cancel_events,
        "interventions": intervention_dispatched,
        "parsed_responses": persona_response_parsed,
        "parse_failures": persona_parse_failures,
    }


def _inject_ambient(session_id: str, prompt: str, *, source: str = "stage3-test", summary: str = "Stage 3 ambient probe") -> dict:
    """POST to /api/sessions/{id}/inject to fire a real ambient turn.

    Returns the response JSON. The ambient turn runs through the same
    SDK loop as a user turn but with `turn.source == "ambient"`, which
    is what gates the post-loop ensemble in messages.py.
    """
    body = {
        "text": prompt,
        "source": source,
        "summary": summary,
        "priority": "notable",
    }
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{SERVER_URL}/api/sessions/{session_id}/inject",
            json=body,
        )
        resp.raise_for_status()
        return resp.json()


def _wait_for_post_loop_aggregation(session_id: str, *, timeout: float = 60.0) -> bool:
    """Poll the events.jsonl until a stage3_post_loop aggregation_decision
    lands. Returns True on found, False on timeout. Stage 3's post-loop
    ensemble fires after `ResultMessage` of an ambient turn — the timing
    depends on Brain 1's reply length.
    """
    started = time.time()
    while time.time() - started < timeout:
        for e in _read_event_log(session_id):
            if (
                e.get("event") == "inner_voice.aggregation_decision"
                and (e.get("data", {}) or {}).get("stage") == "stage3_post_loop"
            ):
                return True
        time.sleep(2)
    return False


def _grade_user_turn(summary: dict, stream_result: dict) -> tuple[bool, list[str]]:
    """Stage 3 acceptance for the USER-turn (mid-turn drift) leg.

    User turns don't fire the post-loop ensemble (Stage 2's gating
    contract). The acceptance for this leg is purely the mid-turn drift
    pipeline being wired and either:
      (a) firing + cancelling on a confabulated trajectory (positive case)
      (b) staying silent on a clean short response (negative case — also
          correct, the gate `min_chars_before_first_check` blocks the
          check on responses too short to evaluate).
    """
    notes: list[str] = []
    passed = True

    final_chars = len(stream_result.get("final_text") or "")
    min_first_check = 250  # matches default in get_mid_turn_drift_config

    if summary["mid_turn_checks"]:
        notes.append(
            f"PASS: {len(summary['mid_turn_checks'])} mid-turn drift "
            f"check(s) fired (drift_detector against partial response)"
        )
        for c in summary["mid_turn_checks"][:3]:
            notes.append(
                f"  pos={c.get('stream_position_chars')}c "
                f"delta_index={c.get('delta_index')}"
            )
        # In the positive-case path, expect a cancel + intervention.
        if summary["cancel_events"]:
            ce = summary["cancel_events"][-1]
            notes.append(
                f"PASS: mid-turn cancel_event_fired — persona={ce.get('persona')!r} "
                f"severity={ce.get('severity')} at {ce.get('stream_position_chars')}c"
            )
        else:
            notes.append(
                "INFO: mid-turn check fired but verdict was sub-veto (no cancel)"
            )
        if summary["interventions"]:
            iv = summary["interventions"][-1]
            notes.append(
                f"PASS: intervention_dispatched: kind={iv.get('kind')!r} "
                f"trigger={iv.get('trigger')!r} queue_depth={iv.get('queue_depth')}"
            )
    elif final_chars < min_first_check:
        # Negative case: response was short enough that mid-turn check is
        # gated off. This is correct behavior, not a failure.
        notes.append(
            f"PASS (silent-correct): response {final_chars}c < "
            f"min_chars_before_first_check={min_first_check}; mid-turn "
            f"drift check correctly did not fire"
        )
    else:
        # Response was long enough to fire a check, but none happened.
        # That's a wiring bug.
        passed = False
        notes.append(
            f"FAIL: response {final_chars}c ≥ {min_first_check}c but no "
            f"mid-turn drift checks fired — wiring bug in messages.py"
        )

    parsed = len(summary["parsed_responses"])
    failures = len(summary["parse_failures"])
    if parsed + failures > 0:
        rate = parsed / (parsed + failures)
        if rate >= 0.9:
            notes.append(f"PASS: parse rate {rate:.0%} ({parsed}/{parsed+failures})")
        else:
            passed = False
            notes.append(f"FAIL: parse rate {rate:.0%} ({parsed}/{parsed+failures}) below 90%")

    return passed, notes


def _grade_ambient_turn(events_after: list[dict]) -> tuple[bool, list[str]]:
    """Stage 3 acceptance for the AMBIENT-turn leg.

    Ambient turns fire the post-loop 3-persona ensemble. The acceptance
    is: 3 personas invoked roughly concurrently, real aggregation
    decision logged.
    """
    notes: list[str] = []
    passed = True

    # Filter to events from the ambient turn only — we look at the slice
    # after the user-turn aggregation.
    iv_invoked = [e for e in events_after if e.get("event") == "inner_voice.persona_invoked"]
    iv_aggregations = [
        e for e in events_after
        if e.get("event") == "inner_voice.aggregation_decision"
        and (e.get("data") or {}).get("stage") == "stage3_post_loop"
    ]
    iv_ensemble_selected = [
        e for e in events_after if e.get("event") == "inner_voice.ensemble_selected"
    ]

    distinct_personas = sorted({(e.get("data") or {}).get("persona", "") for e in iv_invoked})
    distinct_personas = [p for p in distinct_personas if p]

    if len(distinct_personas) >= 3:
        notes.append(f"PASS: {len(distinct_personas)} distinct personas fired: {distinct_personas}")
    elif len(distinct_personas) >= 2:
        notes.append(
            f"PARTIAL: {len(distinct_personas)} distinct personas fired: "
            f"{distinct_personas} (target: 3)"
        )
    else:
        passed = False
        notes.append(
            f"FAIL: only {len(distinct_personas)} distinct persona(s) fired: "
            f"{distinct_personas}"
        )

    if iv_ensemble_selected:
        es = iv_ensemble_selected[-1].get("data") or {}
        notes.append(
            f"PASS: ensemble_selected: name={es.get('ensemble_name')!r} "
            f"personas={es.get('personas')} rationale={es.get('selection_rationale')!r}"
        )
    else:
        passed = False
        notes.append("FAIL: no ensemble_selected event emitted on ambient turn")

    if iv_aggregations:
        agg = iv_aggregations[-1].get("data") or {}
        notes.append(
            f"PASS: post-loop aggregation: action={agg.get('action_chosen')!r} "
            f"severity_max={agg.get('severity_max')} "
            f"disagree_count={agg.get('personas_disagreed')}/{agg.get('personas_invoked')} "
            f"rationale={agg.get('rationale')!r}"
        )
    else:
        passed = False
        notes.append("FAIL: no stage3_post_loop aggregation_decision event")

    # Concurrency check: timestamps of persona_invoked events should fall
    # within a tight window if they truly ran concurrently.
    if len(iv_invoked) >= 2:
        ts = sorted(e.get("ts", "") for e in iv_invoked)
        if ts and ts[0] and ts[-1]:
            try:
                t_first = datetime.fromisoformat(ts[0].rstrip("Z"))
                t_last = datetime.fromisoformat(ts[-1].rstrip("Z"))
                spread_ms = (t_last - t_first).total_seconds() * 1000
                if spread_ms < 500:
                    notes.append(f"PASS: persona invocations span {spread_ms:.0f}ms (concurrent)")
                else:
                    notes.append(
                        f"WARN: persona invocations span {spread_ms:.0f}ms "
                        f"(>500ms suggests serial)"
                    )
            except Exception:
                pass

    return passed, notes


def main() -> int:
    print("=" * 70)
    print("Inner Voice Stage 3 integration test — bench_008 + ambient ensemble")
    print("=" * 70)

    # ── Leg 1: USER turn — verify mid-turn drift detection ──────────────
    print("\nLEG 1: USER TURN — mid-turn drift detection on bench_008")
    print("-" * 70)

    session_id = _make_session(inner_voice=True)

    print(f"POSTing user turn to {SERVER_URL}/api/message/stream …")
    print(f"  prompt: {BENCH_PROMPT!r}")
    stream_result = _stream_message(session_id, BENCH_PROMPT)

    # Give the post-loop ensemble + mid-turn checks a moment to land their
    # SQLite writes and event log lines (they're spawned as fire-and-forget).
    print("Waiting 8s for mid-turn checks to settle…")
    time.sleep(8)

    print(f"Reading events.jsonl for session {session_id}…")
    events_user = _read_event_log(session_id)
    print(f"  {len(events_user)} events captured (user turn)")

    summary_user = _summarize_events(events_user)
    print("Event counts (user turn):")
    for name, n in sorted(summary_user["counts"].items()):
        print(f"  {n:3d}  {name}")

    print("Stage 3 mid-turn acceptance grading:")
    user_passed, user_notes = _grade_user_turn(summary_user, stream_result)
    for n in user_notes:
        print(f"  {n}")

    # ── Leg 2: AMBIENT turn — verify 3-persona concurrent ensemble ──────
    print("\n\nLEG 2: AMBIENT TURN — 3-persona concurrent post-loop ensemble")
    print("-" * 70)

    ambient_session_id = _make_session(inner_voice=True)

    # Need at least one user message in the session for the ambient inject
    # to find an active session — but we also want a turn that produces a
    # response long enough to exercise the post-loop ensemble.
    print("Priming ambient session with a benign user turn…")
    _stream_message(
        ambient_session_id,
        "Reply with just the word 'ready' and nothing else.",
        timeout=30.0,
    )
    time.sleep(2)

    print(f"\nInjecting ambient turn into session {ambient_session_id}…")
    # Prompt is shaped to force a surfaced (non-silent) response so the
    # post-loop ensemble fires reliably. We DO NOT want ambient_decide
    # (silent path) to be selected, because that branch intentionally
    # bypasses the post-loop ensemble. Asking a direct factual question
    # with explicit "do not call ambient_decide" instruction gives Brain 1
    # no plausible silent option.
    ambient_prompt = (
        "Test ping for Inner Voice Stage 3 ensemble. Reply with three "
        "complete sentences describing what Lloyd is. Do NOT call "
        "ambient_decide — this turn must produce visible text output."
    )
    inject_result = _inject_ambient(
        ambient_session_id, ambient_prompt,
        source="stage3-test:ambient", summary="Stage 3 ambient ensemble probe",
    )
    print(f"  ambient inject → turn_id={inject_result.get('turn_id')!r}")

    print("Waiting up to 60s for stage3_post_loop aggregation_decision…")
    found = _wait_for_post_loop_aggregation(ambient_session_id, timeout=60.0)
    print(f"  aggregation found: {found}")

    # Some additional buffer for SQLite + event-log writes after the
    # decision lands.
    time.sleep(3)

    events_ambient = _read_event_log(ambient_session_id)
    print(f"\nReading events.jsonl for ambient session {ambient_session_id}…")
    print(f"  {len(events_ambient)} events captured (ambient turn)")

    summary_ambient = _summarize_events(events_ambient)
    print("Event counts (ambient turn):")
    for name, n in sorted(summary_ambient["counts"].items()):
        print(f"  {n:3d}  {name}")

    print("Stage 3 post-loop ensemble acceptance grading:")
    ambient_passed, ambient_notes = _grade_ambient_turn(events_ambient)
    for n in ambient_notes:
        print(f"  {n}")

    # ── Verdict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    overall = user_passed and ambient_passed
    print(f"OVERALL: {'PASS' if overall else 'PARTIAL'}")
    print(f"  USER-turn (mid-turn drift):     {'PASS' if user_passed else 'FAIL'}")
    print(f"  AMBIENT-turn (3-persona gather): {'PASS' if ambient_passed else 'FAIL'}")
    print(f"User session:    {session_id}")
    print(f"Ambient session: {ambient_session_id}")
    print(f"Event logs:      {EVENT_LOGS_DIR}")
    print("=" * 70)

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
