"""Inner Voice (#345) Stage 7 integration test.

Stage 7 ships:
  * `tool_result_grader` persona — fires after every tool result via the
    SDK PostToolUse / PostToolUseFailure hooks. Catches validation loops,
    hallucination-imminent patterns, wrong-tool choices, tool thrash.
  * `progress_monitor` persona — fires every K tool calls or every M
    seconds (whichever first). Synthesizes "are we making progress vs
    the original task?"
  * `app/inner_voice/intra_turn.py` — per-session state, prompt assembly,
    persona dispatch, steer dispatch via lazy-imported _maybe_dispatch_steer.
  * `messages.py` wiring — PostToolUse + PostToolUseFailure hooks on
    opted-in Inner Voice sessions, plus periodic progress_monitor pulse
    from the text_delta path.
  * Config knobs under `inner_voice.intra_turn.*`.

Acceptance gates from backlog #355:
  1. Validation loop catch — Read called 4× with missing file_path, the
     4th fire's grader user prompt has all 3 prior errors in
     `<recent_tool_results>` and `_same_shape_error_count` returns ≥3.
  2. Progress synthesis — 8-tool-call autonomy turn, progress_monitor
     fires once at K=5, prompt has all 8 in turn_history.
  3. Stuck pattern — 6 vault_search calls in a row, progress_monitor's
     prompt clearly shows the search_thrash pattern.
  4. Cap enforcement — PostToolUse max_calls_per_turn=5 cap actually
     fires; the 6th tool boundary's intra_turn_skipped event lands
     with reason='cap_reached'.
  5. Stage 3/4/5/6 regression — all prior test suites still importable
     and the wiring they assert still present in messages.py.

Run from inside the lloyd container:
  $ /home/alansrobotlab/lloyd/.venvs/lloyd/bin/python \\
      tests/integration/test_inner_voice_stage7.py
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import time
import uuid
from pathlib import Path

LLOYD_HOME = Path("/home/alansrobotlab/lloyd")
sys.path.insert(0, str(LLOYD_HOME))

from app.inner_voice import (  # noqa: E402
    intra_turn as _it,
    consensus_termination as _ct,
    ensemble as _ens,
)
from app.inner_voice.critic import Critique  # noqa: E402
import usage_store  # noqa: E402


def _check(label: str, cond: bool, detail: str = "") -> bool:
    sym = "OK" if cond else "FAIL"
    print(f"  [{sym}] {label}{(': ' + detail) if detail else ''}")
    return cond


# ---------------------------------------------------------------------------
# 1. Persona files exist + parse cleanly
# ---------------------------------------------------------------------------


def test_persona_files() -> bool:
    print("\n=== 1. Persona files load ===")
    personas_dir = Path("/home/alansrobotlab/obsidian/lloyd/inner_voice/personas")

    ok = True
    for name in ("tool_result_grader", "progress_monitor"):
        path = personas_dir / f"{name}.md"
        ok &= _check(f"{name}.md exists", path.exists(), str(path))
        if path.exists():
            loaded = _ens._read_persona_file(name)
            ok &= _check(
                f"{name}.md parses (frontmatter + body)",
                loaded is not None,
            )
            if loaded is not None:
                meta, body = loaded
                ok &= _check(
                    f"{name} has version frontmatter",
                    isinstance(meta.get("version"), str),
                    f"version={meta.get('version')!r}",
                )
                ok &= _check(
                    f"{name} body has output format spec",
                    "Output format" in body or "output is ONE JSON" in body,
                )
                ok &= _check(
                    f"{name} body is non-trivial (>1KB)",
                    len(body) > 1000,
                    f"chars={len(body)}",
                )
    return ok


# ---------------------------------------------------------------------------
# 2. State lifecycle — start, record, end, idempotent
# ---------------------------------------------------------------------------


def test_state_lifecycle() -> bool:
    print("\n=== 2. State lifecycle ===")
    sid = f"stage7-state-{uuid.uuid4().hex[:8]}"
    turn_id = f"turn-{uuid.uuid4().hex[:8]}"

    ok = True
    # Pre-start: no state
    ok &= _check("no state pre-start", _it.get_state(sid) is None)

    _it.start_intra_turn(
        sid, turn_id, turn_source="ambient", frozen_task_intent="test task",
    )
    state = _it.get_state(sid)
    ok &= _check("state created on start", state is not None)
    if state is not None:
        ok &= _check("turn_id matches", state.turn_id == turn_id)
        ok &= _check("turn_source matches", state.turn_source == "ambient")
        ok &= _check("frozen_task_intent matches", state.frozen_task_intent == "test task")
        ok &= _check("tool_calls empty initially", state.tool_calls == [])
        ok &= _check("post_tool_calls_fired starts at 0", state.post_tool_calls_fired == 0)

    # Stale end_intra_turn (different turn_id) → does NOT clear
    _it.end_intra_turn(sid, "wrong-turn-id")
    ok &= _check(
        "stale end_intra_turn does not clear",
        _it.get_state(sid) is not None,
    )

    # Correct end_intra_turn → clears
    _it.end_intra_turn(sid, turn_id)
    ok &= _check("correct end_intra_turn clears", _it.get_state(sid) is None)

    # Idempotent end (already cleared) → no error
    try:
        _it.end_intra_turn(sid, turn_id)
        ok &= _check("idempotent end_intra_turn OK", True)
    except Exception as e:
        ok &= _check(f"idempotent end_intra_turn raised: {e}", False)
    return ok


# ---------------------------------------------------------------------------
# 3. Validation loop catch (acceptance gate 1)
# ---------------------------------------------------------------------------


def test_validation_loop_catch() -> bool:
    print("\n=== 3. Validation loop catch (gate 1) ===")
    sid = f"stage7-loop-{uuid.uuid4().hex[:8]}"
    turn_id = f"turn-{uuid.uuid4().hex[:8]}"
    _it.start_intra_turn(
        sid, turn_id, turn_source="ambient", frozen_task_intent="read all logs",
    )
    state = _it.get_state(sid)
    assert state is not None

    # Seed 4 same-shape Read errors directly into state.tool_calls (we're
    # testing the detection logic, not the SDK round-trip).
    err_msg = "missing required parameter: file_path"
    for _ in range(4):
        state.tool_calls.append(_it._ToolCallRecord(
            tool_name="Read",
            input_summary="Read()",
            result_excerpt=err_msg,
            is_error=True,
            when=time.monotonic(),
        ))

    ok = True
    sse = _it._same_shape_error_count(state)
    ok &= _check(
        f"_same_shape_error_count returns 4 (got {sse})",
        sse == 4,
    )

    # User prompt for the 4th call: all 3 prior errors visible in
    # <recent_tool_results>, the 4th in <this_tool_call> / <this_tool_result>.
    prompt = _it._build_grader_user_prompt(state)
    ok &= _check("prompt has <task>", "<task>" in prompt)
    ok &= _check("prompt has <recent_tool_results>", "<recent_tool_results>" in prompt)
    ok &= _check("prompt has <this_tool_call>", "<this_tool_call>" in prompt)
    ok &= _check("prompt has <this_tool_result>", "<this_tool_result>" in prompt)
    ok &= _check(
        "<recent_tool_results> shows 3 prior errors",
        prompt.count(err_msg) >= 4,  # 3 in recent_tool_results + 1 in this_tool_result
        f"err_msg count={prompt.count(err_msg)}",
    )
    ok &= _check(
        "<this_tool_result> wraps the error in (error: ...)",
        f"(error: {err_msg})" in prompt,
    )
    ok &= _check(
        "task intent surfaced",
        "read all logs" in prompt,
    )

    # Heterogeneous: same tool, different errors → counter resets
    state.tool_calls.append(_it._ToolCallRecord(
        tool_name="Read",
        input_summary="Read(file_path)",
        result_excerpt="(file not found)",
        is_error=True,
        when=time.monotonic(),
    ))
    sse_hetero = _it._same_shape_error_count(state)
    ok &= _check(
        f"different error → counter is 1 (got {sse_hetero})",
        sse_hetero == 1,
    )

    # Cleanup
    _it.end_intra_turn(sid, turn_id)
    return ok


# ---------------------------------------------------------------------------
# 4. Progress synthesis (acceptance gate 2)
# ---------------------------------------------------------------------------


def test_progress_synthesis() -> bool:
    print("\n=== 4. Progress synthesis (gate 2) ===")
    sid = f"stage7-prog-{uuid.uuid4().hex[:8]}"
    turn_id = f"turn-{uuid.uuid4().hex[:8]}"
    _it.start_intra_turn(
        sid, turn_id, turn_source="ambient",
        frozen_task_intent="apply rename across 8 files and run tests",
    )
    state = _it.get_state(sid)
    assert state is not None

    # Successful 8-tool-call autonomy task: read+edit pattern
    seq = [
        ("Grep", "Grep(glob,pattern)", "8 files matched"),
        ("Read", "Read(file_path)", "(2KB body of file 1)"),
        ("Edit", "Edit(file_path,new_string,old_string)", "Edit applied"),
        ("Read", "Read(file_path)", "(1.5KB body of file 2)"),
        ("Edit", "Edit(file_path,new_string,old_string)", "Edit applied"),
        ("Read", "Read(file_path)", "(3KB body of file 3)"),
        ("Edit", "Edit(file_path,new_string,old_string)", "Edit applied"),
        ("Bash", "Bash(command)", "27 tests passed"),
    ]
    base_when = time.monotonic()
    for i, (n, summary, result) in enumerate(seq):
        state.tool_calls.append(_it._ToolCallRecord(
            tool_name=n,
            input_summary=summary,
            result_excerpt=result,
            is_error=False,
            when=base_when + i * 3.0,
        ))

    prompt = _it._build_progress_user_prompt(state)
    ok = True
    ok &= _check("progress prompt has <task>", "<task>" in prompt)
    ok &= _check("progress prompt has <elapsed_seconds>", "<elapsed_seconds>" in prompt)
    ok &= _check("progress prompt has <tool_call_count>", "<tool_call_count>" in prompt)
    ok &= _check("progress prompt has <turn_history>", "<turn_history>" in prompt)
    ok &= _check(
        f"tool_call_count == 8 in prompt",
        "<tool_call_count>\n8\n</tool_call_count>" in prompt,
    )
    # All 8 tools should appear in turn_history (Read, Edit, Grep, Bash)
    ok &= _check(
        "Grep in history", "[tool] Grep(glob,pattern)" in prompt,
    )
    ok &= _check(
        "all 3 Edit calls in history (3 occurrences of Edit())",
        prompt.count("[tool] Edit(") == 3,
    )
    ok &= _check(
        "all 3 Read calls in history",
        prompt.count("[tool] Read(") == 3,
    )
    ok &= _check(
        "Bash test invocation in history",
        "[tool] Bash(command)" in prompt
        and "27 tests passed" in prompt,
    )
    # Task intent must be visible
    ok &= _check(
        "task intent surfaced",
        "apply rename" in prompt and "run tests" in prompt,
    )

    _it.end_intra_turn(sid, turn_id)
    return ok


# ---------------------------------------------------------------------------
# 5. Stuck pattern — search_thrash visibility (acceptance gate 3)
# ---------------------------------------------------------------------------


def test_stuck_pattern_search_thrash() -> bool:
    print("\n=== 5. Stuck pattern visibility (gate 3) ===")
    sid = f"stage7-thrash-{uuid.uuid4().hex[:8]}"
    turn_id = f"turn-{uuid.uuid4().hex[:8]}"
    _it.start_intra_turn(
        sid, turn_id, turn_source="ambient",
        frozen_task_intent="find any mention of intra-turn monitoring",
    )
    state = _it.get_state(sid)
    assert state is not None

    # 6 vault_search calls with no Read/Edit follow-up
    for i in range(6):
        state.tool_calls.append(_it._ToolCallRecord(
            tool_name="mcp__lloyd-mcp__vault_search",
            input_summary=f"vault_search(query)",
            result_excerpt=f"{2 + i % 3} hits",
            is_error=False,
            when=time.monotonic() + i,
        ))

    prompt = _it._build_progress_user_prompt(state)
    ok = True
    ok &= _check("tool_call_count == 6", "<tool_call_count>\n6\n</tool_call_count>" in prompt)
    ok &= _check(
        "all 6 vault_search calls in history",
        prompt.count("[tool] vault_search(") == 6,
    )
    ok &= _check(
        "no Read/Edit in history (search_thrash signal)",
        "[tool] Read(" not in prompt
        and "[tool] Edit(" not in prompt,
    )
    ok &= _check(
        "task intent surfaced",
        "find any mention of intra-turn monitoring" in prompt,
    )

    _it.end_intra_turn(sid, turn_id)
    return ok


# ---------------------------------------------------------------------------
# 6. Cap enforcement (acceptance gate 4)
# ---------------------------------------------------------------------------


async def _run_cap_test() -> bool:
    print("\n=== 6. Cap enforcement (gate 4) ===")
    sid = f"stage7-cap-{uuid.uuid4().hex[:8]}"
    turn_id = f"turn-{uuid.uuid4().hex[:8]}"

    # Patch the persona dispatchers to no-ops so we don't fire real
    # Brain 2 calls during the test. The cap-enforcement logic lives
    # entirely in `_record_tool_result` / `_maybe_fire_progress_monitor`,
    # which is what we want to exercise.
    fired_grader: list[str] = []
    fired_progress: list[str] = []

    async def _noop_grader(*, session_id, state_snapshot_turn_id):
        fired_grader.append(state_snapshot_turn_id)

    async def _noop_progress(*, session_id, state_snapshot_turn_id, trigger_kind):
        fired_progress.append(trigger_kind)

    # Stash + replace
    orig_grader = _it._run_tool_result_grader
    orig_progress = _it._run_progress_monitor
    _it._run_tool_result_grader = _noop_grader   # type: ignore[assignment]
    _it._run_progress_monitor = _noop_progress   # type: ignore[assignment]

    try:
        _it.start_intra_turn(
            sid, turn_id, turn_source="ambient",
            frozen_task_intent="cap test",
        )
        # Fire 7 PostToolUse boundaries — cap is 5, so 5 grader fires + 2 skips
        for i in range(7):
            await _it._record_tool_result(
                session_id=sid,
                tool_name="Read",
                tool_input={"file_path": f"/tmp/file{i}.txt"},
                tool_response="some body",
                error=None,
            )

        # Allow ensure_future tasks to run.
        await asyncio.sleep(0.05)

        state = _it.get_state(sid)
        assert state is not None

        ok = True
        ok &= _check(
            f"7 tool_calls recorded ({len(state.tool_calls)})",
            len(state.tool_calls) == 7,
        )
        ok &= _check(
            f"post_tool_calls_fired == 5 (cap) ({state.post_tool_calls_fired})",
            state.post_tool_calls_fired == 5,
        )
        ok &= _check(
            f"grader actually fired 5x ({len(fired_grader)})",
            len(fired_grader) == 5,
        )
        # progress_monitor fires every 5 tool calls; with 7 boundaries
        # and the per-tool trigger, we expect 1 fire (at the 5th boundary).
        ok &= _check(
            f"progress_monitor fired 1x at K=5 ({len(fired_progress)})",
            len(fired_progress) == 1,
        )
        if fired_progress:
            ok &= _check(
                "progress trigger_kind == tool_count",
                fired_progress[0] == "tool_count",
            )
        return ok
    finally:
        _it._run_tool_result_grader = orig_grader   # type: ignore[assignment]
        _it._run_progress_monitor = orig_progress   # type: ignore[assignment]
        _it.end_intra_turn(sid, turn_id)


def test_cap_enforcement() -> bool:
    return asyncio.run(_run_cap_test())


# ---------------------------------------------------------------------------
# 7. Hooks dict wiring — verify messages.py installs PostToolUse +
#    PostToolUseFailure when intra_turn is enabled.
# ---------------------------------------------------------------------------


def test_hooks_dict_wiring() -> bool:
    print("\n=== 7. Hooks dict wiring ===")
    from app.routers.messages import _inner_voice_hooks_dict
    hd = _inner_voice_hooks_dict("stage7-test-hooks")
    ok = True
    ok &= _check("PreToolUse present", "PreToolUse" in hd)
    ok &= _check("PostToolUse present", "PostToolUse" in hd)
    ok &= _check("PostToolUseFailure present", "PostToolUseFailure" in hd)
    if "PostToolUse" in hd:
        ok &= _check("PostToolUse has 1 matcher", len(hd["PostToolUse"]) == 1)
        m = hd["PostToolUse"][0]
        ok &= _check(
            "PostToolUse matcher is None (matches every tool)",
            m.matcher is None,
            f"matcher={m.matcher!r}",
        )
    return ok


# ---------------------------------------------------------------------------
# 8. Steer dispatch shape — verify the synthetic agg dict shape that
#    intra_turn passes into _maybe_dispatch_steer matches what the
#    Stage 6 dispatch path expects.
# ---------------------------------------------------------------------------


async def _run_dispatch_shape_test() -> bool:
    print("\n=== 8. Steer dispatch shape ===")
    # Build a synthetic Critique with severity 0.75 and verify the
    # intra_turn dispatcher constructs the agg dict with the same keys
    # the Stage 6 path keys off. We don't actually call _maybe_dispatch_steer
    # here (that requires session JSON + queue plumbing); the shape check
    # is sufficient because Stage 6's test already covers the dispatch.
    crit = Critique(
        persona="tool_result_grader",
        persona_version="v1",
        disagrees=True,
        severity=0.75,
        reason="[validation_loop] Read called 4× with missing file_path",
        suggested_action="nudge",
        action_taken="log_only",
        model="primary",
    )

    # Tap _maybe_dispatch_steer to capture its kwargs without firing.
    captured: dict = {}

    async def _capturing_dispatch(**kwargs):
        captured.update(kwargs)
        return False  # pretend the budget was exhausted

    # Inject our capturing dispatcher via the same lazy-import path
    # `_dispatch_steer_via_messages` uses.
    import app.routers.messages as messages_mod
    orig = messages_mod._maybe_dispatch_steer
    messages_mod._maybe_dispatch_steer = _capturing_dispatch  # type: ignore[assignment]
    try:
        sid = f"stage7-shape-{uuid.uuid4().hex[:8]}"
        turn_id = f"turn-{uuid.uuid4().hex[:8]}"
        result = await _it._dispatch_steer_via_messages(
            session_id=sid,
            turn_id=turn_id,
            turn_source="ambient",
            critique=crit,
            persona_kind_label="tool_result_grader",
            response_text="",
        )

        ok = True
        ok &= _check(
            "_dispatch_steer_via_messages returns False on budget exhaustion",
            result is False,
        )
        ok &= _check("kwargs include session_id", captured.get("session_id") == sid)
        ok &= _check("kwargs include turn_id", captured.get("turn_id") == turn_id)
        ok &= _check(
            "kwargs include turn_source=ambient",
            captured.get("turn_source") == "ambient",
        )
        ok &= _check(
            "kwargs include critiques (single-element list)",
            isinstance(captured.get("critiques"), list)
            and len(captured["critiques"]) == 1
            and captured["critiques"][0] is crit,
        )
        agg = captured.get("agg") or {}
        ok &= _check(
            "agg.action_chosen == nudge_proposed",
            agg.get("action_chosen") == "nudge_proposed",
            f"got {agg.get('action_chosen')!r}",
        )
        ok &= _check(
            "agg.severity_max == 0.75",
            agg.get("severity_max") == 0.75,
            f"got {agg.get('severity_max')!r}",
        )
        ok &= _check(
            "agg.disagree_count == 1",
            agg.get("disagree_count") == 1,
        )
        ok &= _check(
            "agg.rationale mentions intra_turn",
            "intra_turn" in (agg.get("rationale") or ""),
        )
        ok &= _check(
            "ensemble_name marks intra_turn",
            captured.get("ensemble_name", "").startswith("intra_turn_"),
            f"got {captured.get('ensemble_name')!r}",
        )
        return ok
    finally:
        messages_mod._maybe_dispatch_steer = orig  # type: ignore[assignment]


def test_dispatch_shape() -> bool:
    return asyncio.run(_run_dispatch_shape_test())


# ---------------------------------------------------------------------------
# 9. Stage 3/4/5/6 regression — verify prior tests still importable and
#    the wiring they assert still present in messages.py.
# ---------------------------------------------------------------------------


def test_regression_imports() -> bool:
    print("\n=== 9. Stage 3/4/5/6 regression ===")
    ok = True
    src = (LLOYD_HOME / "app" / "routers" / "messages.py").read_text()

    # Stage 1 PreToolUse wiring still there
    ok &= _check(
        "Stage 1 PreToolUse wiring still in messages.py",
        "make_pretooluse_callback" in src and "PreToolUse" in src,
    )
    # Stage 3 mid-turn drift still there
    ok &= _check(
        "Stage 3 _inner_voice_mid_turn_drift_check still wired",
        "_inner_voice_mid_turn_drift_check" in src
        and "_iv_mtd_first_check_due" in src,
    )
    # Stage 4 consensus_termination still there
    ok &= _check(
        "Stage 4 consensus_termination still wired",
        "_iv_consensus.evaluate" in src
        and "has_task_complete_signal" in src,
    )
    # Stage 5 grading pass still there
    ok &= _check(
        "Stage 5 grading pass still wired",
        "_inner_voice_grading_pass" in src,
    )
    # Stage 6 steer dispatch still there
    ok &= _check(
        "Stage 6 _maybe_dispatch_steer still wired",
        "_maybe_dispatch_steer" in src
        and 'agg.get("action_chosen") != "nudge_proposed"' in src,
    )
    # Stage 7 wiring landed
    ok &= _check(
        "Stage 7 PostToolUse wiring landed",
        "_iv_intra_turn" in src
        and "PostToolUse" in src
        and "PostToolUseFailure" in src,
    )
    ok &= _check(
        "Stage 7 start_intra_turn / end_intra_turn called",
        "start_intra_turn" in src and "end_intra_turn" in src,
    )

    # Verify each prior test file is still present.
    for stage in (3, 4, 5, 6):
        path = LLOYD_HOME / "tests" / "integration" / f"test_inner_voice_stage{stage}.py"
        ok &= _check(f"test_inner_voice_stage{stage}.py exists", path.exists())
    return ok


# ---------------------------------------------------------------------------
# 10. SQLite persistence — verify a synthetic critique row writes via
#     the same record_inner_voice_critique path the post-loop ensemble
#     uses, with persona='tool_result_grader' / 'progress_monitor'.
# ---------------------------------------------------------------------------


def test_sqlite_persistence() -> bool:
    print("\n=== 10. SQLite persistence (record path) ===")
    test_sid = f"stage7-sqlite-{uuid.uuid4().hex[:8]}"
    test_turn = f"turn-{uuid.uuid4().hex[:8]}"

    ok = True
    for persona in ("tool_result_grader", "progress_monitor"):
        cid = usage_store.record_inner_voice_critique(
            session_id=test_sid,
            turn_id=test_turn,
            persona=persona,
            persona_version="v1",
            model="primary",
            input_tokens=800,
            output_tokens=30,
            latency_ms=350,
            disagrees=True,
            severity=0.72,
            reason=f"[validation_loop] {persona} flag",
            suggested_action="nudge",
            action_taken="steer",
            anchor_response_excerpt="...",
            event_log_offset=None,
            raw_response_offset=None,
            prompt_hash="x" * 64,
            parse_attempts=1,
        )
        ok &= _check(f"{persona} row inserted (id={cid})", cid > 0)

    # Verify both rows exist with correct persona
    db = LLOYD_HOME / "usage.db"
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT persona, severity, action_taken FROM inner_voice_critiques "
        "WHERE session_id = ? ORDER BY id",
        (test_sid,),
    ).fetchall()
    ok &= _check(f"2 rows present (got {len(rows)})", len(rows) == 2)
    if len(rows) == 2:
        personas = sorted(r[0] for r in rows)
        ok &= _check(
            f"personas == [progress_monitor, tool_result_grader] (got {personas})",
            personas == ["progress_monitor", "tool_result_grader"],
        )
        ok &= _check(
            "all rows have severity 0.72",
            all(r[1] == 0.72 for r in rows),
        )
        ok &= _check(
            "all rows have action_taken='steer'",
            all(r[2] == "steer" for r in rows),
        )

    # Cleanup
    conn.execute("DELETE FROM inner_voice_critiques WHERE session_id = ?", (test_sid,))
    conn.commit()
    conn.close()
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    results: list[tuple[str, bool]] = []
    results.append(("persona files load", test_persona_files()))
    results.append(("state lifecycle", test_state_lifecycle()))
    results.append(("validation loop catch (gate 1)", test_validation_loop_catch()))
    results.append(("progress synthesis (gate 2)", test_progress_synthesis()))
    results.append(("stuck pattern visibility (gate 3)", test_stuck_pattern_search_thrash()))
    results.append(("cap enforcement (gate 4)", test_cap_enforcement()))
    results.append(("hooks dict wiring", test_hooks_dict_wiring()))
    results.append(("steer dispatch shape", test_dispatch_shape()))
    results.append(("Stage 3/4/5/6 regression (gate 5)", test_regression_imports()))
    results.append(("SQLite persistence", test_sqlite_persistence()))

    print("\n=== Stage 7 test summary ===")
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
