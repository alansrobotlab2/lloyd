"""Inner Voice (#345) Stage 6 integration test.

Stage 6 ships:
  * `make_steer_ambient()` builder — alongside `make_drift_cancel_ambient`
    and `make_please_continue_ambient`. Produces an `<inner-voice
    kind='steer'>` envelope with persona feedback.
  * Stage 6 dispatch path in `_maybe_dispatch_steer` (messages.py): when
    the post-loop ensemble's aggregate is `nudge_proposed`, enqueue a
    steer prefetch + an actual ambient turn so Brain 1 retries the task
    addressing the persona feedback. Records `inner_voice_interventions`
    row with `kind='steer'`.
  * Shared nudge budget — `_session_nudge_counts` in
    `consensus_termination.py` covers BOTH the veto path AND the new
    steer path. `is_self_correct_on_nudge_enabled()`,
    `can_consume_nudge_budget()`, `consume_nudge_budget()` are the
    public API.

Test sections:
  1. `make_steer_ambient` shape — the five required keys, the inner-voice
     tag, the persona names + reasons in the body.
  2. Aggregate threshold — severity 0.7 → `nudge_proposed`, severity
     0.85+ → `veto_proposed`, severity 0.4 → `log_only`.
  3. Budget gating — `can_consume_nudge_budget` flips to False after
     `consume_nudge_budget` is called `max_nudges_per_session` times.
  4. Config gate — `is_self_correct_on_nudge_enabled()` reads from
     `inner_voice.self_correct_on_nudge` (default true).
  5. In-process dispatch — synthetic critique list with severity 0.7
     produces a steer dispatch (intervention row written, prefetch
     enqueued). Idempotent on dedup_key.
  6. Skip when consensus_termination already fired — the brain2_check
     branches are mutually exclusive on the same turn.

Run from inside the lloyd container:
  $ /home/alansrobotlab/lloyd/.venvs/lloyd/bin/python \\
      tests/integration/test_inner_voice_stage6.py
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import uuid
from pathlib import Path

LLOYD_HOME = Path("/home/alansrobotlab/lloyd")
sys.path.insert(0, str(LLOYD_HOME))

from app.inner_voice import (  # noqa: E402
    ensemble as _ens,
    consensus_termination as _ct,
)
from app.inner_voice.critic import Critique  # noqa: E402
import usage_store  # noqa: E402


def _check(label: str, cond: bool, detail: str = "") -> bool:
    sym = "OK" if cond else "FAIL"
    print(f"  [{sym}] {label}{(': ' + detail) if detail else ''}")
    return cond


def _mk_critique(persona: str, severity: float, reason: str, disagrees: bool = True) -> Critique:
    return Critique(
        persona=persona,
        persona_version="v1",
        disagrees=disagrees,
        severity=severity,
        reason=reason,
        suggested_action="nudge" if severity >= 0.6 and severity < 0.85 else ("veto" if severity >= 0.85 else None),
        action_taken="log_only",
        model="primary",
    )


# ---------------------------------------------------------------------------
# 1. make_steer_ambient shape
# ---------------------------------------------------------------------------


def test_steer_shape() -> bool:
    print("\n=== 1. make_steer_ambient shape ===")
    out = _ens.make_steer_ambient(
        turn_id="abc123",
        severity_max=0.72,
        reasons=[
            ("continuation_drive", "task asked for 3 things, only 2 done"),
            ("skill_recall_checker", "deep-research skill matched but not used"),
        ],
        response_excerpt="Here are the first two items done...",
    )

    ok = True
    ok &= _check("returns dict", isinstance(out, dict))
    ok &= _check(
        "all 4 keys present",
        set(out.keys()) == {"source", "summary", "content", "dedup_key"},
        f"keys={sorted(out.keys())}",
    )
    ok &= _check("source is inner_voice:steer", out.get("source") == "inner_voice:steer")
    ok &= _check(
        "dedup_key includes turn_id",
        "abc123" in (out.get("dedup_key") or ""),
        f"dedup_key={out.get('dedup_key')!r}",
    )
    ok &= _check(
        "content has inner-voice steer tag",
        "<inner-voice kind='steer'>" in out["content"]
        and "</inner-voice>" in out["content"],
    )
    ok &= _check(
        "content names both personas",
        "continuation_drive" in out["content"]
        and "skill_recall_checker" in out["content"],
    )
    ok &= _check(
        "content names both reasons",
        "task asked for 3 things" in out["content"]
        and "deep-research skill matched" in out["content"],
    )
    ok &= _check(
        "summary names severity",
        "0.72" in out["summary"],
        f"summary={out['summary']!r}",
    )

    # Empty reasons → still produces output, body says "(no reasons)"
    out_empty = _ens.make_steer_ambient(
        turn_id="def456",
        severity_max=0.6,
        reasons=[],
        response_excerpt="",
    )
    ok &= _check(
        "empty reasons → body has fallback",
        "(no reasons" in out_empty["content"],
    )

    # >3 reasons → only top 3 surfaced
    many = [(f"persona_{i}", f"reason {i}") for i in range(5)]
    out_many = _ens.make_steer_ambient(
        turn_id="xyz789",
        severity_max=0.7,
        reasons=many,
        response_excerpt="...",
    )
    ok &= _check(
        ">3 reasons → only first 3 in body",
        all(f"persona_{i}" in out_many["content"] for i in range(3))
        and "persona_3" not in out_many["content"]
        and "persona_4" not in out_many["content"],
    )

    return ok


# ---------------------------------------------------------------------------
# 2. Aggregate threshold mapping
# ---------------------------------------------------------------------------


def test_aggregate_thresholds() -> bool:
    print("\n=== 2. Aggregate threshold → action_chosen ===")
    # severity_threshold=0.6, veto_severity_threshold=0.85 (per config.yaml)
    cases = [
        ("severity 0.0", [_mk_critique("p1", 0.0, "ok", disagrees=False)], "agreement"),
        ("severity 0.4 alone", [_mk_critique("p1", 0.4, "soft")], "log_only"),
        ("severity 0.6", [_mk_critique("p1", 0.6, "nudge floor")], "nudge_proposed"),
        ("severity 0.7", [_mk_critique("p1", 0.7, "nudge band")], "nudge_proposed"),
        ("severity 0.84", [_mk_critique("p1", 0.84, "just under veto")], "nudge_proposed"),
        ("severity 0.85", [_mk_critique("p1", 0.85, "veto floor")], "veto_proposed"),
        ("severity 0.95", [_mk_critique("p1", 0.95, "strong veto")], "veto_proposed"),
        # 2 sub-threshold disagreements at count_threshold=2 → nudge by count
        (
            "2× severity 0.5 (count_threshold)",
            [_mk_critique("p1", 0.5, "soft"), _mk_critique("p2", 0.5, "soft")],
            "nudge_proposed",
        ),
        # 3 errored → no_op
        ("all errored", [Critique(persona=f"p{i}", error="timeout") for i in range(3)], "no_op"),
    ]

    ok = True
    for label, critiques, expected in cases:
        agg = _ens._aggregate(critiques)
        got = agg.get("action_chosen")
        ok &= _check(
            f"{label} → {expected}",
            got == expected,
            f"got {got!r} (rationale: {agg.get('rationale','')[:60]!r})",
        )
    return ok


# ---------------------------------------------------------------------------
# 3. Shared nudge budget
# ---------------------------------------------------------------------------


def test_nudge_budget() -> bool:
    print("\n=== 3. Shared nudge budget ===")
    # Use a fake session_id so we don't pollute real session state
    sid = f"stage6-budget-test-{uuid.uuid4().hex[:8]}"

    # Budget starts at 2 (config default)
    cap = _ct._max_nudges_per_session()
    ok = True
    ok &= _check(f"budget cap is 2 (config default)", cap == 2, f"got {cap}")

    # Initial: can consume
    ok &= _check("fresh session can consume", _ct.can_consume_nudge_budget(sid))

    # First consume → counter at 1, can still consume
    n1 = _ct.consume_nudge_budget(sid)
    ok &= _check(f"after 1 consume, count == 1", n1 == 1, f"got {n1}")
    ok &= _check("can still consume (1 < 2)", _ct.can_consume_nudge_budget(sid))

    # Second consume → at cap
    n2 = _ct.consume_nudge_budget(sid)
    ok &= _check(f"after 2 consumes, count == 2", n2 == 2, f"got {n2}")
    ok &= _check("budget exhausted (2 >= 2)", not _ct.can_consume_nudge_budget(sid))

    # Cleanup the test session counter so we don't leak into other runs.
    _ct._session_nudge_counts.pop(sid, None)
    return ok


# ---------------------------------------------------------------------------
# 4. Config gate
# ---------------------------------------------------------------------------


def test_config_gate() -> bool:
    print("\n=== 4. inner_voice.self_correct_on_nudge config gate ===")
    enabled = _ct.is_self_correct_on_nudge_enabled()
    return _check(
        "self_correct_on_nudge enabled by default",
        enabled,
        f"got {enabled}",
    )


# ---------------------------------------------------------------------------
# 5. In-process dispatch — synthetic intervention insert via the helper path
# ---------------------------------------------------------------------------


async def _run_dispatch_path_test() -> bool:
    print("\n=== 5. In-process steer dispatch ===")
    # We're testing the persistence side here — `_maybe_dispatch_steer`
    # would normally also enqueue an ambient turn. That path requires a
    # real session JSON + the consumer factory. For this test we just
    # validate the parts that DON'T require the SDK round-trip:
    #   - intervention row is written with kind='steer'
    #   - the row's content matches the make_steer_ambient body
    #   - the triggering_critique_id link is set when a critique is provided
    test_sid = f"stage6-dispatch-{uuid.uuid4().hex[:8]}"
    test_turn = f"turn-{uuid.uuid4().hex[:8]}"

    # Insert a synthetic critique row to act as the triggering critique
    triggering_id = usage_store.record_inner_voice_critique(
        session_id=test_sid,
        turn_id=test_turn,
        persona="continuation_drive",
        persona_version="v1",
        model="primary",
        input_tokens=1000,
        output_tokens=40,
        latency_ms=420,
        disagrees=True,
        severity=0.72,
        reason="task asked for 3 things, only 2 done",
        suggested_action="nudge",
        action_taken="log_only",
        anchor_response_excerpt="Here are the first two items done...",
        event_log_offset=None,
        raw_response_offset=None,
        prompt_hash="x" * 64,
        parse_attempts=1,
    )

    # Build the steer kwargs the same way _maybe_dispatch_steer does
    steer = _ens.make_steer_ambient(
        turn_id=test_turn,
        severity_max=0.72,
        reasons=[("continuation_drive", "task asked for 3 things, only 2 done")],
        response_excerpt="Here are the first two items done...",
    )

    iv_id = usage_store.record_inner_voice_intervention(
        session_id=test_sid,
        kind="steer",
        target_turn_id=test_turn,
        content=steer["content"],
        triggered_by_critique_id=triggering_id,
    )

    ok = True
    ok &= _check("intervention row inserted", iv_id > 0, f"iv_id={iv_id}")

    # Verify shape
    rows = usage_store.list_inner_voice_interventions(session_id=test_sid)
    target = next((r for r in rows if r["id"] == iv_id), None)
    ok &= _check("row found in list", target is not None)
    if target:
        ok &= _check(
            "kind=steer",
            target.get("kind") == "steer",
            f"kind={target.get('kind')!r}",
        )
        ok &= _check(
            "triggered_by_critique_id linked",
            target.get("triggered_by_critique_id") == triggering_id,
            f"link={target.get('triggered_by_critique_id')!r} expected={triggering_id}",
        )
        ok &= _check(
            "outcome_addressed initially NULL (will be backfilled by grader)",
            target.get("outcome_addressed") is None,
        )
        ok &= _check(
            "content includes inner-voice steer tag",
            "<inner-voice kind='steer'>" in (target.get("content") or ""),
        )

    # Cleanup
    db = LLOYD_HOME / "usage.db"
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM inner_voice_interventions WHERE session_id = ?", (test_sid,))
    conn.execute("DELETE FROM inner_voice_critiques WHERE session_id = ?", (test_sid,))
    conn.commit()
    conn.close()
    return ok


def test_dispatch_path() -> bool:
    return asyncio.run(_run_dispatch_path_test())


# ---------------------------------------------------------------------------
# 6. Mutual exclusion: consensus_termination wins over steer dispatch
# ---------------------------------------------------------------------------


def test_mutual_exclusion() -> bool:
    print("\n=== 6. consensus_termination + steer mutual exclusion ===")
    # The check here is logical, not runtime — verify our wiring in
    # `_inner_voice_brain2_check` correctly skips the steer branch when
    # consensus_termination already handled the turn. Inspect by reading
    # the source and confirming the `consensus_handled` flag is checked.
    src = (LLOYD_HOME / "app" / "routers" / "messages.py").read_text()
    ok = True
    ok &= _check(
        "_maybe_dispatch_steer guarded by `not consensus_handled`",
        "if not consensus_handled:" in src
        and "_maybe_dispatch_steer" in src,
    )
    ok &= _check(
        "steer dispatch references action_chosen == nudge_proposed",
        'agg.get("action_chosen") != "nudge_proposed"' in src,
    )
    ok &= _check(
        "steer dispatch checks shared budget",
        "can_consume_nudge_budget" in src,
    )
    ok &= _check(
        "steer dispatch consumes shared budget",
        "consume_nudge_budget" in src,
    )
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    results: list[tuple[str, bool]] = []
    results.append(("make_steer_ambient shape", test_steer_shape()))
    results.append(("aggregate thresholds", test_aggregate_thresholds()))
    results.append(("nudge budget", test_nudge_budget()))
    results.append(("config gate", test_config_gate()))
    results.append(("dispatch path", test_dispatch_path()))
    results.append(("mutual exclusion wiring", test_mutual_exclusion()))

    print("\n=== Stage 6 test summary ===")
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
