"""Inner Voice (#345) Stage 4 integration test.

Runs four end-to-end scenarios against the Brain 2 ensemble + consensus
termination + escape hatch logic:

  1. **Routing** — verify safety_critical / research_writing / code_writing
     are picked correctly given synthetic tool-call histories.
  2. **Veto path (in-process)** — synthetic critique list with severity
     above the veto threshold produces a `vetoed` decision and a
     please-continue ambient.
  3. **Escape hatches (in-process)** — Brain 2 timeout, three-strike,
     max_nudges, hard_max_turns each fire their specific decision branch.
  4. **HTTP `/state` shape** — the Stage 4 endpoint returns the new fields
     (consecutive_vetoes, escalations_count, hard_max_turns, stage='4').

The first three are pure-Python smoke tests against the modules — no
Brain 2 inference call needed. The fourth requires the lloyd-backend
running on port 8080 (CLAUDE.md procedure).

Run from inside the lloyd container:
  $ /home/alansrobotlab/lloyd/.venvs/lloyd/bin/python \\
      tests/integration/test_inner_voice_stage4.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path

import httpx

LLOYD_HOME = Path("/home/alansrobotlab/lloyd")
SESSIONS_DIR = LLOYD_HOME / "sessions"
EVENT_LOGS_DIR = LLOYD_HOME / "event_logs"
SERVER_URL = "http://127.0.0.1:8080"

sys.path.insert(0, str(LLOYD_HOME))

from app.inner_voice import (  # noqa: E402
    consensus_termination as _ct,
    ensemble as _ens,
)
from app.inner_voice.critic import Critique  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check(label: str, cond: bool, detail: str = "") -> bool:
    sym = "OK" if cond else "FAIL"
    print(f"  [{sym}] {label}{(': ' + detail) if detail else ''}")
    return cond


# ---------------------------------------------------------------------------
# 1. Routing
# ---------------------------------------------------------------------------


def test_routing() -> bool:
    print("\n=== 1. Ensemble routing ===")
    cases = [
        ("safety_critical / Bash rm",
         [{"name": "Bash", "input": {"command": "rm -rf /tmp/foo"}}],
         "safety_critical",
         lambda p: "adversarial_red_team" in p),
        ("safety_critical / git push --force",
         [{"name": "Bash", "input": {"command": "git push origin main --force"}}],
         "safety_critical",
         lambda p: "adversarial_red_team" in p),
        ("safety_critical / email_send",
         [{"name": "mcp__lloyd-mcp__email_send", "input": {}}],
         "safety_critical",
         lambda p: "adversarial_red_team" in p),
        ("safety_critical / fact_invalidate",
         [{"name": "mcp__lloyd-mcp__fact_invalidate", "input": {}}],
         "safety_critical",
         lambda p: True),
        ("research_writing / vault_write",
         [{"name": "mcp__lloyd-mcp__vault_write", "input": {}}],
         "research_writing",
         lambda p: "hallucination_flag" in p),
        ("research_writing / 3 vault_search",
         [{"name": "mcp__lloyd-mcp__vault_search"} for _ in range(3)],
         "research_writing",
         lambda p: True),
        ("research_writing / http_fetch",
         [{"name": "mcp__lloyd-mcp__http_fetch", "input": {}}],
         "research_writing",
         lambda p: True),
        ("code_writing / 2 edits",
         [{"name": "Edit"}, {"name": "Write"}],
         "code_writing",
         lambda p: True),
        ("autonomy_default / read-only",
         [{"name": "Read"}],
         "autonomy_default",
         lambda p: "completion_checker" in p),
        ("autonomy_default / no tools",
         [],
         "autonomy_default",
         lambda p: True),
    ]
    ok = True
    for desc, tcs, expect_name, persona_check in cases:
        name, personas, _ = _ens._select_ensemble_for_turn(
            "ambient", "", tool_calls=tcs,
        )
        cond = name == expect_name and persona_check(personas)
        ok &= _check(
            desc, cond,
            f"got name={name!r} personas={personas}",
        )
    return ok


# ---------------------------------------------------------------------------
# 2. Veto path
# ---------------------------------------------------------------------------


async def test_veto_path() -> bool:
    print("\n=== 2. Veto path (in-process) ===")
    sid = f"stage4_unit_veto_{uuid.uuid4().hex[:6]}"
    cs = [
        Critique(persona="completion_checker", disagrees=True, severity=0.9,
                 reason="task asked 3 notes, only 2 vault_write calls"),
        Critique(persona="hallucination_flag", disagrees=False, severity=0.0),
    ]
    decision = await _ct.evaluate(
        session_id=sid,
        turn_id="turn1",
        response_text="All set. SIGNAL:TASK_COMPLETE",
        critiques=cs,
        ensemble_name="research_writing",
    )
    ok = True
    ok &= _check("action == vetoed", decision.action == "vetoed",
                 f"got {decision.action}")
    ok &= _check("please_continue_kwargs present",
                 decision.please_continue_kwargs is not None)
    if decision.please_continue_kwargs:
        ok &= _check("ambient content includes please-continue tag",
                     "<inner-voice kind='please-continue'>" in
                     decision.please_continue_kwargs.get("content", ""))
        ok &= _check("ambient summary names veto severity",
                     "0.90" in decision.please_continue_kwargs.get("summary", ""))
    ok &= _check("nudge_count_after == 1",
                 decision.nudge_count_after == 1,
                 f"got {decision.nudge_count_after}")
    ok &= _check("consecutive_vetoes_after == 1",
                 decision.consecutive_vetoes_after == 1)
    return ok


# ---------------------------------------------------------------------------
# 3. Escape hatches
# ---------------------------------------------------------------------------


async def test_escape_hatches() -> bool:
    print("\n=== 3. Escape hatches (4 hatches) ===")
    ok = True

    # Hatch 1: Brain 2 timeout — every persona errored
    sid = f"stage4_hatch1_{uuid.uuid4().hex[:6]}"
    cs = [Critique(persona="completion_checker", error="timeout after 5s")]
    d = await _ct.evaluate(
        session_id=sid, turn_id="t1", response_text="done. SIGNAL:TASK_COMPLETE",
        critiques=cs, ensemble_name="autonomy_default",
    )
    ok &= _check("hatch 1 / brain2_timeout",
                 d.action == "accepted_brain2_timeout"
                 and d.hatch_fired == "brain2_timeout",
                 f"got action={d.action} hatch={d.hatch_fired}")

    # Hatch 2: three-strike — pre-set streak to 2
    sid = f"stage4_hatch2_{uuid.uuid4().hex[:6]}"
    _ct._session_consecutive_vetoes[sid] = 2
    cs = [Critique(persona="completion_checker", disagrees=True,
                   severity=0.9, reason="still incomplete")]
    d = await _ct.evaluate(
        session_id=sid, turn_id="t1", response_text="SIGNAL:TASK_COMPLETE",
        critiques=cs, ensemble_name="autonomy_default",
    )
    ok &= _check("hatch 2 / three_strike",
                 d.action == "accepted_three_strike"
                 and d.hatch_fired == "three_strike",
                 f"got action={d.action} hatch={d.hatch_fired}")

    # Hatch 3: max_nudges — pre-bump nudge count to cap
    sid = f"stage4_hatch3_{uuid.uuid4().hex[:6]}"
    cap = _ct._max_nudges_per_session()
    _ct._session_nudge_counts[sid] = cap
    cs = [Critique(persona="completion_checker", disagrees=True,
                   severity=0.9, reason="incomplete")]
    d = await _ct.evaluate(
        session_id=sid, turn_id="t1", response_text="SIGNAL:TASK_COMPLETE",
        critiques=cs, ensemble_name="autonomy_default",
    )
    ok &= _check(f"hatch 3 / max_nudges (cap={cap})",
                 d.action == "escalated_max_nudges"
                 and d.hatch_fired == "max_nudges"
                 and d.escalation_kwargs is not None,
                 f"got action={d.action} hatch={d.hatch_fired}")

    # Hatch 4: hard_max_turns
    sid = f"stage4_hatch4_{uuid.uuid4().hex[:6]}"
    cs = [Critique(persona="completion_checker", disagrees=False, severity=0.0)]
    d = await _ct.evaluate(
        session_id=sid, turn_id="t1", response_text="SIGNAL:TASK_COMPLETE",
        critiques=cs, ensemble_name="autonomy_default",
        hard_max_turns_hit=True,
    )
    ok &= _check("hatch 4 / hard_max_turns",
                 d.action == "escalated_max_turns"
                 and d.hatch_fired == "hard_max_turns",
                 f"got action={d.action} hatch={d.hatch_fired}")

    # Sanity: no SIGNAL:TASK_COMPLETE → has_task_complete_signal returns False
    ok &= _check("BLOCKED suppresses TASK_COMPLETE detection",
                 _ct.has_task_complete_signal(
                     "SIGNAL:BLOCKED:foo\nSIGNAL:TASK_COMPLETE"
                 ) is False)
    ok &= _check("plain response ≠ TASK_COMPLETE",
                 _ct.has_task_complete_signal("Just text") is False)
    return ok


# ---------------------------------------------------------------------------
# 4. HTTP /state shape
# ---------------------------------------------------------------------------


def test_state_endpoint() -> bool:
    print("\n=== 4. HTTP /api/inner_voice/state Stage 4 shape ===")
    try:
        r = httpx.get(
            f"{SERVER_URL}/api/inner_voice/state",
            timeout=10.0,
        )
    except httpx.HTTPError as e:
        return _check("backend reachable", False, str(e))
    if r.status_code != 200:
        return _check("HTTP 200", False, f"status={r.status_code}")
    body = r.json()
    ok = True
    ok &= _check("HTTP 200", True)
    for k in (
        "stage", "consecutive_vetoes", "escalations_count",
        "veto_severity_threshold", "hard_max_turns", "personas",
        "configured_personas", "max_nudges_per_session",
    ):
        ok &= _check(f"field present: {k}", k in body, f"got keys={list(body.keys())}")
    ok &= _check("stage == '4'", body.get("stage") == "4",
                 f"got {body.get('stage')!r}")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    routing_ok = test_routing()
    veto_ok = await test_veto_path()
    hatches_ok = await test_escape_hatches()
    state_ok = test_state_endpoint()
    all_ok = routing_ok and veto_ok and hatches_ok and state_ok
    print("\n=== Stage 4 summary ===")
    print(f"  routing       : {'PASS' if routing_ok else 'FAIL'}")
    print(f"  veto path     : {'PASS' if veto_ok else 'FAIL'}")
    print(f"  escape hatches: {'PASS' if hatches_ok else 'FAIL'}")
    print(f"  /state shape  : {'PASS' if state_ok else 'FAIL'}")
    print()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
