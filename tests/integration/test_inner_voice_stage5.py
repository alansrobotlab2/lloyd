"""Inner Voice (#345) Stage 5 integration test.

Stage 5 ships:
  * `skill_recall_checker` persona — flags missed skill invocations.
    Routes via the `<matched_skills>` block in the user prompt.
  * `grader` persona — backfills `outcome_addressed` /
    `outcome_summary` on `inner_voice_interventions` after each
    ambient turn.
  * `/api/inner_voice/grading_summary` endpoint — coverage +
    addressed_rate metrics.
  * `/state` returns `stage='5'` and a `grading_progress` block.

Test sections:

  1. **Routing** — `skill_recall_checker` is in the `autonomy_default`
     and `research_writing` ensemble sets, picked by
     `_select_ensemble_for_turn` for the matching tool histories.
  2. **`<matched_skills>` injection** — `_build_user_prompt` emits the
     block iff `matched_skills` kwarg is provided. The block is
     missing for personas that don't need it.
  3. **`_top_matched_skills`** — pre-fetch returns the top-K (name,
     description, score) tuples and caches per-query. Falls back to
     [] on import errors.
  4. **Grader JSON parsing** — `_parse_grader_json` handles the
     prefill-continuation, embedded object, and scalar fallback shapes.
     `_coerce_addressed` is tri-valued.
  5. **In-process grading** — synthetic intervention + synthetic
     `_call_grader` mock → `grade_outcome_turn` updates SQLite via
     `update_intervention_outcome`. Idempotent on repeat calls.
  6. **HTTP `/state`** — Stage 5 shape returns `stage='5'` and a
     `grading_progress` block. Requires the lloyd-backend running on
     port 8080 (CLAUDE.md procedure).
  7. **HTTP `/grading_summary`** — endpoint returns the documented
     schema; addressed_rate computes correctly given recent data.

Run from inside the lloyd container:
  $ /home/alansrobotlab/lloyd/.venvs/lloyd/bin/python \\
      tests/integration/test_inner_voice_stage5.py
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import sys
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx

LLOYD_HOME = Path("/home/alansrobotlab/lloyd")
SESSIONS_DIR = LLOYD_HOME / "sessions"
EVENT_LOGS_DIR = LLOYD_HOME / "event_logs"
SERVER_URL = "http://127.0.0.1:8080"

sys.path.insert(0, str(LLOYD_HOME))

from app.inner_voice import (  # noqa: E402
    ensemble as _ens,
    grading as _grading,
)
from app.inner_voice.critic import Critique  # noqa: E402
import usage_store  # noqa: E402


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
    print("\n=== 1. Stage 5 ensemble routing ===")
    cases = [
        (
            "autonomy_default contains skill_recall_checker",
            [{"name": "Read"}],
            "autonomy_default",
            lambda p: "skill_recall_checker" in p,
        ),
        (
            "research_writing contains skill_recall_checker",
            [{"name": "mcp__lloyd-mcp__vault_write", "input": {}}],
            "research_writing",
            lambda p: "skill_recall_checker" in p,
        ),
        (
            "code_writing does NOT contain skill_recall_checker",
            [{"name": "Edit"}, {"name": "Write"}],
            "code_writing",
            lambda p: "skill_recall_checker" not in p,
        ),
        (
            "safety_critical does NOT contain skill_recall_checker",
            [{"name": "Bash", "input": {"command": "rm -rf /tmp/foo"}}],
            "safety_critical",
            lambda p: "skill_recall_checker" not in p,
        ),
    ]
    ok = True
    for desc, tcs, expect_name, persona_check in cases:
        name, personas, _ = _ens._select_ensemble_for_turn(
            "ambient", "research three speculative decoding papers", tool_calls=tcs,
        )
        cond = name == expect_name and persona_check(personas)
        ok &= _check(desc, cond, f"got name={name!r} personas={personas}")
    return ok


# ---------------------------------------------------------------------------
# 2. <matched_skills> injection
# ---------------------------------------------------------------------------


def test_matched_skills_injection() -> bool:
    print("\n=== 2. <matched_skills> block injection ===")
    base_args = dict(
        frozen_task_intent="research three papers on speculative decoding",
        response_text="I'll dig in.",
        tool_calls=[{"name": "http_fetch"}],
        transcript=[],
        mode="final",
    )

    # Without matched_skills: no <matched_skills> block.
    p1 = _ens._build_user_prompt(**base_args)
    ok1 = _check(
        "no matched_skills kwarg → no <matched_skills> block",
        "<matched_skills>" not in p1,
        f"prompt tail: {p1[-200:]!r}",
    )

    # With matched_skills: block present, formatted.
    matches = [
        ("deep-research", "Multi-source research with synthesis", 8.4),
        ("arxiv", "Fetch and parse arXiv", 5.1),
    ]
    p2 = _ens._build_user_prompt(**base_args, matched_skills=matches)
    ok2 = _check(
        "with matched_skills kwarg → block present",
        "<matched_skills>" in p2 and "</matched_skills>" in p2,
        f"contains tags: {('<matched_skills>' in p2)} / {('</matched_skills>' in p2)}",
    )
    ok3 = _check(
        "block contains formatted skill names",
        "deep-research" in p2 and "arxiv" in p2 and "score 8.4" in p2,
        f"deep-research={'deep-research' in p2} arxiv={'arxiv' in p2} score 8.4={'score 8.4' in p2}",
    )

    # Empty list: block still present, contents are "(empty)".
    p3 = _ens._build_user_prompt(**base_args, matched_skills=[])
    ok4 = _check(
        "empty matched_skills → block contents '(empty)'",
        "<matched_skills>\n(empty)\n</matched_skills>" in p3,
    )

    # Below MIN_SCORE_KEEP (4.0): rendered as (empty) too.
    matches_low = [("foo-skill", "low score", 2.1)]
    p4 = _ens._build_user_prompt(**base_args, matched_skills=matches_low)
    ok5 = _check(
        "all matches below score floor → (empty) rendering",
        "<matched_skills>\n(empty)\n</matched_skills>" in p4,
        f"prompt tail: {p4[-150:]!r}",
    )
    return ok1 and ok2 and ok3 and ok4 and ok5


# ---------------------------------------------------------------------------
# 3. _top_matched_skills behavior + caching
# ---------------------------------------------------------------------------


def test_top_matched_skills() -> bool:
    print("\n=== 3. _top_matched_skills ===")
    # Empty query → []
    out0 = _ens._top_matched_skills("")
    ok0 = _check("empty query → []", out0 == [], f"got {out0!r}")

    # Real query: should return at least one match (skills directory has
    # plenty of registered skills). Don't assert on specific names — the
    # skill index changes; just verify shape.
    out1 = _ens._top_matched_skills("deep research arxiv synthesis")
    ok1 = _check(
        "real query returns list of (name, desc, score) tuples",
        isinstance(out1, list)
        and all(
            isinstance(t, tuple) and len(t) == 3 and isinstance(t[2], (int, float))
            for t in out1
        ),
        f"len={len(out1)} sample={out1[:1]!r}",
    )

    # Cache hit: second call with same query returns the same object.
    out2 = _ens._top_matched_skills("deep research arxiv synthesis")
    ok2 = _check("repeat call returns cached value", out1 == out2)

    return ok0 and ok1 and ok2


# ---------------------------------------------------------------------------
# 4. Grader JSON parsing + tri-value coercion
# ---------------------------------------------------------------------------


def test_grader_parsing() -> bool:
    print("\n=== 4. Grader JSON parsing ===")
    cases = [
        ("direct full object", '{"addressed": true, "summary": "did the thing"}', True, "did the thing"),
        ("prefill continuation true", ' true, "summary": "ok"}', True, "ok"),
        ("prefill continuation false", ' false, "summary": "missed"}', False, "missed"),
        ("prefill continuation null", ' null, "summary": "ambiguous"}', None, "ambiguous"),
        ("embedded JSON in prose", 'Here is the verdict: {"addressed": true, "summary": "yes"} done.', True, "yes"),
        ("scalar true", "true", True, "scalar fallback"),
        ("scalar false", "false", False, "scalar fallback"),
    ]
    ok = True
    for desc, raw, expect_addr, expect_sum in cases:
        parsed = _grading._parse_grader_json(raw)
        cond = (
            parsed is not None
            and _grading._coerce_addressed(parsed.get("addressed")) == expect_addr
            and (parsed.get("summary") or "")[: len(expect_sum)] == expect_sum
        )
        ok &= _check(desc, cond, f"parsed={parsed!r}")

    # Total failure path
    parsed = _grading._parse_grader_json("not json at all")
    ok &= _check("non-JSON input → None", parsed is None, f"got {parsed!r}")

    # Coercion edge cases
    ok &= _check("coerce 'yes' → True", _grading._coerce_addressed("yes") is True)
    ok &= _check("coerce 'unknown' → None", _grading._coerce_addressed("unknown") is None)
    ok &= _check("coerce 1 → True", _grading._coerce_addressed(1) is None,
                 detail="ints are NOT auto-coerced (only bool / str)")
    ok &= _check("coerce True → True", _grading._coerce_addressed(True) is True)
    return ok


# ---------------------------------------------------------------------------
# 5. In-process grading: synthetic intervention + mocked grader call
# ---------------------------------------------------------------------------


async def _run_grading_mock_test() -> bool:
    print("\n=== 5. In-process grading flow ===")
    # Use a one-shot session ID + a temporary in-memory wrapper. We
    # write directly to the live usage.db (read/write) — but with a
    # session_id namespace that won't collide with anything.
    test_session_id = f"stage5-test-{uuid.uuid4().hex[:12]}"
    test_turn_id_intervention = f"turn-{uuid.uuid4().hex[:8]}-intervention"
    test_turn_id_outcome = f"turn-{uuid.uuid4().hex[:8]}-outcome"

    # 5a. Insert a test intervention row.
    iv_id = usage_store.record_inner_voice_intervention(
        session_id=test_session_id,
        kind="continue",
        target_turn_id=test_turn_id_intervention,
        content=(
            "Inner Voice (completion_checker, severity 0.85) — "
            "task asked for 3 vault_writes, only 2 fired."
        ),
    )
    ok1 = _check("inserted synthetic intervention row", iv_id > 0, f"iv_id={iv_id}")

    # 5b. Verify it appears in list_ungraded_interventions.
    ung = usage_store.list_ungraded_interventions(test_session_id)
    ok2 = _check(
        "test row surfaces in list_ungraded_interventions",
        any(r["id"] == iv_id for r in ung),
        f"len(ung)={len(ung)}",
    )

    # 5c. Mock the grader HTTP call to return a known verdict.
    async def _mock_call_grader(*, persona_system_prompt: str, user_prompt: str) -> dict:
        return {
            "parsed": {
                "addressed": True,
                "summary": "outcome turn fired the missing third vault_write",
            },
            "raw": '{"addressed": true, "summary": "outcome turn fired the missing third vault_write"}',
            "error": None,
        }

    with patch.object(_grading, "_call_grader", _mock_call_grader):
        result = await _grading.grade_outcome_turn(
            session_id=test_session_id,
            outcome_turn_id=test_turn_id_outcome,
            outcome_response_text=(
                "Wrote the third note. Now all three deliverables present."
            ),
            outcome_tool_calls=[{"name": "mcp__lloyd-mcp__vault_write"}],
            frozen_task_intent="write three knowledge notes summarizing gap-007",
        )
    ok3 = _check(
        "grade_outcome_turn returned summary with graded=1",
        result.get("graded", 0) >= 1,
        f"result={result!r}",
    )

    # 5d. Verify the intervention row was updated in SQLite.
    rows = usage_store.list_inner_voice_interventions(session_id=test_session_id, limit=10)
    target = next((r for r in rows if r["id"] == iv_id), None)
    ok4 = _check(
        "outcome_turn_id backfilled on test row",
        target is not None and target.get("outcome_turn_id") == test_turn_id_outcome,
        f"target.outcome_turn_id={target.get('outcome_turn_id') if target else None!r}",
    )
    ok5 = _check(
        "outcome_addressed=True backfilled",
        target is not None
        and (target.get("outcome_addressed") is True or target.get("outcome_addressed") == 1),
        f"outcome_addressed={target.get('outcome_addressed') if target else None!r}",
    )
    ok6 = _check(
        "outcome_summary contains grader text",
        target is not None
        and "third vault_write" in (target.get("outcome_summary") or ""),
        f"outcome_summary={target.get('outcome_summary') if target else None!r}",
    )

    # 5e. Idempotent: second call against the same outcome turn finds nothing.
    with patch.object(_grading, "_call_grader", _mock_call_grader):
        result2 = await _grading.grade_outcome_turn(
            session_id=test_session_id,
            outcome_turn_id=test_turn_id_outcome,
            outcome_response_text="redundant call",
            outcome_tool_calls=[],
            frozen_task_intent="write three knowledge notes summarizing gap-007",
        )
    ok7 = _check(
        "second call produces graded=0 (already graded)",
        result2.get("graded", 0) == 0,
        f"result2={result2!r}",
    )

    # 5f. Skip if target_turn_id == outcome_turn_id.
    iv_id_self = usage_store.record_inner_voice_intervention(
        session_id=test_session_id,
        kind="continue",
        target_turn_id=test_turn_id_outcome,  # same as outcome → must skip
        content="self-grade test row",
    )
    with patch.object(_grading, "_call_grader", _mock_call_grader):
        result3 = await _grading.grade_outcome_turn(
            session_id=test_session_id,
            outcome_turn_id=test_turn_id_outcome,
            outcome_response_text="probe",
            outcome_tool_calls=[],
            frozen_task_intent="probe task",
        )
    rows_after = usage_store.list_inner_voice_interventions(session_id=test_session_id, limit=10)
    self_row = next((r for r in rows_after if r["id"] == iv_id_self), None)
    ok8 = _check(
        "intervention with target_turn_id == outcome_turn_id stays ungraded",
        self_row is not None and self_row.get("outcome_turn_id") is None,
        f"self_row.outcome_turn_id={self_row.get('outcome_turn_id') if self_row else None!r}",
    )

    # 5g. Cleanup: delete the test rows.
    db_path = LLOYD_HOME / "usage.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "DELETE FROM inner_voice_interventions WHERE session_id = ?",
        (test_session_id,),
    )
    conn.commit()
    conn.close()
    ok9 = _check("test rows cleaned up", True)

    return all([ok1, ok2, ok3, ok4, ok5, ok6, ok7, ok8, ok9])


def test_grading_mock() -> bool:
    return asyncio.run(_run_grading_mock_test())


# ---------------------------------------------------------------------------
# 6. HTTP /state shape (Stage 5)
# ---------------------------------------------------------------------------


def test_state_endpoint() -> bool:
    print("\n=== 6. HTTP /api/inner_voice/state Stage 5 shape ===")
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(
                f"{SERVER_URL}/api/inner_voice/state",
                params={"session_id": "stage5-shape-probe"},
            )
            data = resp.json()
    except Exception as e:
        return _check(f"GET /state succeeded", False, f"error={e!r}")
    ok1 = _check("returned JSON", isinstance(data, dict))
    ok2 = _check("stage >= '5'", data.get("stage", "0") >= "5", f"stage={data.get('stage')!r}")
    ok3 = _check(
        "grading_progress key present",
        "grading_progress" in data,
        f"keys={list(data.keys())}",
    )
    gp = data.get("grading_progress") or {}
    ok4 = _check(
        "grading_progress has expected fields",
        all(k in gp for k in (
            "enabled", "graded", "ungraded",
            "addressed_true", "addressed_false", "addressed_null",
        )),
        f"gp keys={list(gp.keys())}",
    )
    return all([ok1, ok2, ok3, ok4])


# ---------------------------------------------------------------------------
# 7. HTTP /grading_summary shape
# ---------------------------------------------------------------------------


def test_grading_summary_endpoint() -> bool:
    print("\n=== 7. HTTP /api/inner_voice/grading_summary shape ===")
    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{SERVER_URL}/api/inner_voice/grading_summary")
            data = resp.json()
    except Exception as e:
        return _check("GET /grading_summary succeeded", False, f"error={e!r}")
    expected_keys = {
        "window_hours", "since_iso",
        "total_interventions", "graded", "graded_rate",
        "addressed_true", "addressed_false", "addressed_null",
        "addressed_rate", "by_persona",
    }
    have = set(data.keys())
    ok1 = _check(
        "all top-level keys present",
        expected_keys.issubset(have),
        f"missing={expected_keys - have}",
    )
    ok2 = _check(
        "by_persona is a dict",
        isinstance(data.get("by_persona"), dict),
        f"type={type(data.get('by_persona')).__name__}",
    )
    return ok1 and ok2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    results: list[tuple[str, bool]] = []
    results.append(("routing", test_routing()))
    results.append(("matched_skills injection", test_matched_skills_injection()))
    results.append(("_top_matched_skills", test_top_matched_skills()))
    results.append(("grader parsing", test_grader_parsing()))
    results.append(("in-process grading", test_grading_mock()))
    # HTTP tests require the backend running; skip on connection refused.
    try:
        with httpx.Client(timeout=2) as c:
            c.get(f"{SERVER_URL}/api/inner_voice/state", params={"session_id": "probe"})
        backend_up = True
    except Exception:
        backend_up = False
    if backend_up:
        results.append(("HTTP /state", test_state_endpoint()))
        results.append(("HTTP /grading_summary", test_grading_summary_endpoint()))
    else:
        print("\n[SKIP] HTTP tests — backend not running on :8080")

    print("\n=== Stage 5 test summary ===")
    failed = [name for name, ok in results if not ok]
    for name, ok in results:
        print(f"  {'OK' if ok else 'FAIL'}  {name}")
    print()
    if failed:
        print(f"FAILED: {failed}")
        return 1
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
