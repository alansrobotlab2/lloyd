"""MCP tools for the self-modification loop.

Mirrors `agent_mcp/autoresearch.py`: the loop's real logic lives under
`scripts/automod/`, and this module is the thin surface the agent calls.

The proposal step is deliberately absent. There is no `automod_write_code`
tool — `automod_start` hands back a worktree path and Lloyd edits it with the
ordinary Edit/Write/Bash tools. The round is a wrapper around normal work, not
a special code-generation mode.

Every mutating tool refuses while `automod.enabled` is false, so the machinery
ships inert.
"""

from __future__ import annotations

import json
import logging
import time

from mcp.types import TextContent, Tool

from agent_mcp._shared import get_bound_session, text_result

logger = logging.getLogger("lloyd-mcp.automod")


def _enabled() -> bool:
    try:
        from app.config import CONFIG
        return bool((CONFIG.get("automod") or {}).get("enabled", False))
    except Exception:
        return False


def _require_inner_voice() -> bool:
    """`automod.require_inner_voice` in config.yaml, default true."""
    try:
        from app.config import CONFIG
        return bool((CONFIG.get("automod") or {}).get("require_inner_voice", True))
    except Exception:
        return True


#: Session-file key holding the turn ids this session has been refused on.
#:
#: A refusal has to outlive the call that issued it, and the session file is
#: the only store both the refusing call and the retry can see: `call_tool`
#: runs its body through `asyncio.to_thread`, so a module-level set is not
#: shared with the dispatch, and the retry can equally well arrive after
#: `lloyd-mcp` restarted. Same file the flag lives in, same write.
IV_REFUSAL_KEY = "inner_voice_refused_turns"

#: How many refused turns to keep. The marker only has to outlive *one* turn,
#: so anything past a handful is bookkeeping nothing reads; the bound is what
#: stops a long-lived session's file growing forever in the case that does
#: accumulate — `inner_voice` cleared and re-refused by a later turn.
IV_REFUSAL_HISTORY = 8


def _current_turn_id() -> str:
    """The turn this dispatch belongs to, or "" for a caller that has none.

    `agent_mcp.main.call_tool` sets it from the harness's `lloyd/turn_id`
    `_meta` (`app/harness/mcp_pool.py` stamps it from `options.turn_id`, which
    `app/routers/messages.py` mints per turn). The CLI, the detached promoter
    and a direct `run_query` caller leave it empty.

    Empty means there is nothing to key a marker on, so the gate keeps today's
    behaviour for those callers instead of locking them out of ever opening a
    round. A sessionless call that can change state never reaches here at all:
    `main.call_tool` refuses it first (#1053).
    """
    from agent_mcp import _task_registry
    try:
        return str(_task_registry.current_turn_id.get() or "")
    except Exception:  # noqa: BLE001
        return ""


def _iv_refusal(action: str, *, enabled_for_next_turn: bool,
                already_refused_this_turn: bool = False) -> dict:
    """The payload handed back for one refusal, first call or repeat.

    `next` stays "end your turn" on the repeat path too. Rewriting it to say
    "call again" is what #837 proposed and #989 rejected: the gate's whole
    claim is that an observer cannot be attached mid-turn, so a payload that
    invited a same-turn retry would formalise the bypass this closes rather
    than the refusal being ignored.
    """
    payload = {
        "error": f"inner voice is not attached to this turn — refusing to {action}",
        "inner_voice_enabled_for_next_turn": enabled_for_next_turn,
        "why": ("A round rewrites production code, and the observer is what catches "
                "the loop drifting. It attaches at turn start, so switching it on "
                "now cannot cover this turn."),
        "next": ("End your turn and open the round again; the retry runs observed."
                 if enabled_for_next_turn else
                 "Could not write the session file — enable Inner Voice on this "
                 "session and retry."),
    }
    if already_refused_this_turn:
        payload["already_refused_this_turn"] = True
    return payload


def _inner_voice_gate(action: str) -> dict | None:
    """Refuse to drive the loop from a turn with no observer attached.

    Returns an error payload to hand back, or None to proceed.

    **Enabling the flag is not sufficient, which is why this refuses rather
    than merely switching it on.** Inner Voice attaches at turn START:
    `attach_observer_for_turn` runs before `run_query` and is the only place
    the observer is installed. Flipping `inner_voice` from inside a tool call
    rewrites the session file for the *next* turn and does nothing for the one
    making the call. Same shape as the position-0 rule for the system prompt.
    A round driven inside a single turn would otherwise report itself observed
    while running blind, which is worse than being plainly unobserved.

    So: enable it, then refuse once. The retry runs observed, and the property
    is real rather than aspirational.

    **The refusal is sticky for the turn that issued it (#1226).** The flag the
    refusing call writes is read by every later call, so checking it first made
    the refusal advice a retry could ignore — on 2026-09-17 the retry came 71
    seconds later in the same turn (`sessions/20260917_104306_autocode_f9fe.json`:
    refused at msg39, `round_id SM_20260917_174742` at msg76, then four gates and
    two land attempts, none of them observed). So the refused-turn marker is
    consulted before the flag: same turn id, same refusal, however many times it
    is asked; a later turn id proceeds, because by then the flag is true and the
    observer really is attached.

    Worth the friction for two independent reasons, and the second is the one
    that survives a quiet round.

    *Live:* a round is the one thing Lloyd does that rewrites production, and
    the observer is what notices the loop drifting off the request, repeating
    itself, or talking its way into a change nobody asked for.

    *Afterwards:* an IV session is the only kind the Inner Voice tab lists, so
    running rounds there is what makes self-modification **reviewable** — the
    transcript, the observations and the interventions all land somewhere a
    human can page back through. A round driven from a plain session leaves
    nothing but a ledger row and a diff, which tells you what changed and
    nothing about how the agent got there.

    No bound session means this is not a chat turn: the CLI, or the detached
    promoter. A human at a terminal is their own observer, so that path passes.
    A caller with a session but no turn id is in the same class for the sticky
    half: nothing keys the marker, so it is refused once and today's behaviour
    is left intact — see `_current_turn_id`.

    **A worker or autonomy session passes too, since 2026-09-12.** Both
    reasons above were written for a *chat* turn, and neither holds for an
    unattended one. *Live:* the observer's measured effect on rounds was
    negative — #874 abandoned at iteration 38 with 44 minutes left on an
    invented premise, sixteen false repetition fires in a day — and the
    drift it was meant to catch is caught deterministically now by the
    anchors and the gate. *Afterwards:* every background run has been
    recorded since 2026-09-10 (`app/run_recorder.py`) and the Background tab
    lists it, so a round driven from a worker session is exactly as
    reviewable as one driven from an IV session. Refusing here would have
    made `inner_voice: false` on the autocode source a switch that stops
    every round from opening, which is the opposite of a switch.
    """
    if not _require_inner_voice():
        return None
    session_id = get_bound_session()
    if not session_id:
        return None

    from app.paths import SESSIONS_DIR
    path = SESSIONS_DIR / f"{session_id}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Fail closed on the observer, not on the round: an unreadable session
        # file should not be able to block self-modification entirely.
        return None
    if not isinstance(data, dict):
        return None
    turn_id = _current_turn_id()
    refused_turns = [t for t in (data.get(IV_REFUSAL_KEY) or []) if isinstance(t, str)]
    try:
        from app.sessions_io import NON_USER_PLATFORMS
    except Exception:  # noqa: BLE001
        NON_USER_PLATFORMS = frozenset()
    # The exemption is checked before both the marker and the flag, so a worker
    # or autonomy session that somehow carries a stale refusal is still exempt:
    # those platforms need no observer at all, and a marker left by an earlier
    # platform value must not act as a switch that stops every round opening.
    if str(data.get("platform") or "") in NON_USER_PLATFORMS:
        return None
    if turn_id and turn_id in refused_turns:
        # The flag is true by now — this turn's own refusal wrote it — so a gate
        # that read the flag first returned None here and the round opened.
        return _iv_refusal(action, enabled_for_next_turn=bool(data.get("inner_voice")),
                           already_refused_this_turn=True)
    if data.get("inner_voice"):
        return None

    data["inner_voice"] = True
    data["inner_voice_evaluate_user_turns"] = True
    if turn_id:
        data[IV_REFUSAL_KEY] = (refused_turns + [turn_id])[-IV_REFUSAL_HISTORY:]
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        enabled = True
    except OSError:
        enabled = False

    return _iv_refusal(action, enabled_for_next_turn=enabled)


def _err(message: str) -> str:
    """Errors are JSON with an `error` key — text_result sniffs that shape."""
    return json.dumps({"error": message})


def _land_detached(round_id: str) -> dict:
    """Start the landing in its own session and return immediately.

    THE reason this tool cannot just call `promote()`. A landing restarts
    `lloyd-mcp` and `lloyd-backend`, and this module runs *inside* lloyd-mcp.
    Both confs set `stopasgroup`/`killasgroup`, so supervisord signals the
    whole process group and kills the promoter partway through its own
    restart sequence. What that leaves is worse than a failed landing: the
    tree already fast-forwarded onto the new commit, the aggregator stopped
    (an intentional stop is not auto-restarted), the backend still serving
    the old code with no tools at all, and `current.json` frozen in `landing`
    — the one state the guardian is explicitly told not to observe. Nothing
    watches it and nothing rolls it back.

    `spawn_detached` puts the promoter in a new session, which is exactly what
    a process-group signal cannot reach. The CLI path survived this only by
    accident, because the Bash tool already spawns that way — so the loop
    worked when a human drove it and would have failed the first time Lloyd
    did, which is the case it exists for.
    """
    from scripts.automod import round as R

    # The checks, the spawn and the marker are `round.land_detached`'s: the
    # reaper starts a landing the same way (a gate that passed after its turn
    # ended), and two copies of "start a landing" would drift.
    started = R.land_detached(round_id, by="automod_land")
    if started.get("error"):
        return started
    pid, log = started["pid"], started["log"]
    return {
        "landing": round_id, "pid": pid, "log": str(log),
        "next": "END YOUR TURN NOW. Do not poll, do not call another tool.",
        # The last thing the model reads before the structured finalizer asks
        # it what happened. #1242 answered `landed: false` and `rejected`
        # because it "could not state landed: true on evidence it did not
        # have", and the item was closed as tried-and-rejected while its change
        # promoted (2026-09-18).
        "outcome": (
            "When you restate the result: `landed` is TRUE — you called automod_land on a "
            "passed gate, and that is what the field asks; the ledger records the promotion, "
            "you are not asked to watch it. Report each clause as the review graded it. "
            "`unnecessary` and `rejected` are for a round that lands nothing; this one is "
            "landing."),
        "note": (
            "Detached, because landing restarts lloyd-mcp and would otherwise kill "
            "the promoter mid-flight. It now waits for the backend to be IDLE — and "
            "your own turn is what is keeping it busy. Polling automod_status in a "
            f"loop starves the gate you are waiting on, and {_landing_minutes_note()}. "
            "Stop talking and the landing proceeds within seconds. "
            "It will restart the backend, which ends this turn anyway. Check "
            "automod_status on your NEXT turn: `current.state` goes landing -> "
            "observing, and the guardian settles it to last-known-good at the end of "
            f"its observation window. {_land_train_note()}Progress is logged to {log}."),
    }


def _landing_minutes_note() -> str:
    """The idle budget and the observation window, read from config at call
    time. Both are `automod.landing` keys that have moved (the window went
    900 s → 450/120 s on 2026-09-20) while this note said "15 minutes" for
    each; a restated number is wrong the day the key changes."""
    try:
        from scripts.automod import promote as P
        budget, _hard = P._idle_budget(None)
        window = P.errors_window(restart=True)
        quiet = P.errors_window(restart=False)
        return (f"after about {round(budget / 60)} minutes of a busy backend the landing "
                f"gives up; once landed, the guardian observes it for about "
                f"{round(window / 60)} minutes ({round(quiet / 60)} when nothing restarted)")
    except Exception:  # noqa: BLE001 — a note is never the landing
        return "after its idle budget the landing gives up"


def _land_train_note() -> str:
    """With the land train on (`automod.landing.defer_restart`), a landing that
    needs no eager restart merges and waits in `pending_restart` for one
    batched flush; say so at call time, because the idle/restart wording above
    describes only the eager path."""
    try:
        from scripts.automod import promote as P
        on, _why = P.restart_deferred()
    except Exception:  # noqa: BLE001 — a note is never the landing
        return ""
    if not on:
        return ""
    return ("With the land train on, a landing that needs no restart of its own "
            "merges without waiting for idle and is listed under "
            "`pending_restart` in automod_status until one batched restart "
            "(a flush) makes it live and opens the window; `current` stays "
            "empty until then. ")


def _background_tasks_for_session() -> list[dict]:
    """Background Bash children the bound session started and has not seen
    finish. A gate that runs while one is in flight kills it at abort or
    restarts under it at landing — #578 lost its only baseline run that way."""
    sid = get_bound_session()
    if not sid:
        return []
    from agent_mcp import _task_registry
    now = time.time()
    return [{"task_id": r.task_id, "description": r.description,
             "elapsed_s": round(now - r.started_at)}
            for r in _task_registry.list_active() if r.session_id == sid]


def _gate_detached(round_id: str, *, skip_smoke: bool = False) -> dict:
    """Start the gate in its own session and return at once.

    A gate with the review rung ran seven to twelve minutes then (sixteen by
    2026-09-18 — `_gate_minutes_note` reads the figure off the ledger, and the
    tool DESCRIPTION carries no number, because it is part of every turn's
    cached prompt prefix and must not change with a median), and the
    2026-09-11 lesson is that one tool call cannot sit silent on the wire
    that long: the transport's read timeout fired, the pool re-sent the
    request, and the same round was gated twice at once — both review
    attempts spent on one commit the model was never shown a finding for.
    So the gate runs detached, `gate.running` says so, and
    `automod_gate_wait` polls in bounded slices.
    """
    from scripts.automod import state as S, worktree as W

    if not W.worktree_path(round_id).exists():
        return {"error": f"no worktree for {round_id}"}
    if S.gate_in_progress(round_id):
        return _gate_wait(round_id, wait_seconds=0,
                          note="a gate is already running for this round; not started again")
    busy = _background_tasks_for_session()
    if busy:
        return {"error": ("refusing to gate while background task(s) of this session are "
                          "still running — a gate that aborts kills them and a landing "
                          "restarts the backend under them. Wait for them (they report on "
                          "a later iteration) or kill them, then gate."),
                "background_tasks": busy}
    from scripts.automod import round as R
    started = R.gate_detached(round_id, by="automod_gate", skip_smoke=skip_smoke)
    if started.get("error"):
        return started
    pid, log = started["pid"], started["log"]
    return {
        "gate_started": round_id, "pid": pid, "log": str(log),
        "next": ("Call automod_gate_wait(round_id). It blocks up to four minutes and "
                 "returns the per-rung report when the ladder finishes, or a progress "
                 "line if it has not — call it again then. Do NOT edit, commit or run "
                 "anything in the worktree while the gate runs: the review grades a "
                 "snapshot of the commit you gated, and every commit it refuses spends "
                 "one of the round's two review attempts."),
        "note": (f"Detached: {_gate_minutes_note()}, longer than one tool call may stay "
                 "silent on the wire. If your turn ends first, a gate that PASSES is landed "
                 "by the loop, one whose grader could not be reached is gated again by the loop, "
                 "and a refused one comes back with its findings."),
    }


def _gate_minutes_note() -> str:
    """How long a full gate takes now, off the ledger. This said "seven to
    twelve minutes" for a week in which the median went from five to sixteen."""
    try:
        from scripts.automod import backlog as B, state as S
        stats = B.gate_duration_stats(S.LEDGER_PATH)
        if stats.get("n"):
            return (f"a full gate takes about {round(stats['median_s'] / 60)} minutes now "
                    f"(median of the last {stats['n']}; the slow ones "
                    f"{round(stats['p90_s'] / 60)})")
    except Exception:  # noqa: BLE001 — a note is never the gate
        pass
    return "a full gate takes many minutes"


def _rungs_since(round_id: str, since_ts: float) -> list[str]:
    from scripts.automod import state as S
    rows = []
    for e in S.read_events(limit=400):
        if e.get("round_id") != round_id or float(e.get("ts") or 0) < since_ts - 1:
            continue
        if e.get("event") == "gate":
            rows.append(f"{'PASS' if e.get('ok') else 'FAIL'} {e.get('rung')} "
                        f"({e.get('seconds', 0)}s): {str(e.get('detail') or '')[:160]}")
        elif e.get("event") == "review":
            rows.append(f"review attempt {e.get('attempt')}: "
                        f"{e.get('kind') or e.get('error') or 'ran'}")
    return rows


def _gate_wait(round_id: str, *, wait_seconds: int = 240, note: str = "") -> dict:
    """Block until the detached gate finishes or `wait_seconds` pass.

    Bounded well under the transport's read timeout, so the call itself can
    never be the thing that times out. A marker whose process is gone with a
    `gate.json` newer than it is a finished gate; gone with no newer report
    is a gate that died, and says so.
    """
    from scripts.automod import state as S

    # `0` is a real answer ("just look"), so no `or 240` here.
    wait = max(0, min(int(240 if wait_seconds is None else wait_seconds), 540))
    deadline = time.time() + wait
    gate_path = S.ROUNDS_DIR / round_id / "gate.json"
    log = S.ROUNDS_DIR / round_id / "gate.log"

    def _report() -> dict:
        rep = json.loads(gate_path.read_text(encoding="utf-8"))
        rep["gate_finished"] = True
        # An external blocker is not a verdict on the diff, and the round's
        # right next move is to wait and gate again — but nothing said so, so
        # round 866-a ended its turn on a grader 503 caused by a SIBLING
        # round's landing and was reaped 30 minutes later with the work
        # intact in its worktree.
        failed = next((r for r in (rep.get("rungs") or [])
                       if not r.get("ok")), None) or {}
        data = failed.get("data") or {}
        if data.get("external_blocker"):
            after = int(data.get("retry_after_s") or 120)
            why = str(data.get("external_reason") or "outside your diff")
            rep["next"] = (
                f"This failure is {why} — outside your diff. Wait {after}s and "
                f"call automod_gate again, up to two more times. Do NOT abort "
                f"and do NOT end the turn on this: nothing about your change "
                f"has been judged yet."
            )
            rep["retry_after_s"] = after
        return _with_headline(rep)

    while True:
        marker = S.read_gate_marker(round_id)
        if marker is None:
            if gate_path.exists():
                return _report()
            return {"error": f"no gate has been started for {round_id} — call automod_gate first"}
        started = float(marker.get("started_at") or 0)
        if not S.pid_alive(marker.get("pid") or 0):
            S.clear_gate_marker(round_id)
            if gate_path.exists() and gate_path.stat().st_mtime >= started - 1:
                return _report()
            return {"error": ("the gate process died before writing a report — read the "
                              "log, then gate again"), "log": str(log), "marker": marker}
        if time.time() >= deadline:
            out = {"running": True, "round_id": round_id, "pid": marker.get("pid"),
                   "elapsed_s": round(time.time() - started),
                   "rungs_so_far": _rungs_since(round_id, started),
                   "next": "call automod_gate_wait(round_id) again"}
            if note:
                out["note"] = note
            return out
        time.sleep(5)


def _with_headline(rep: dict) -> dict:
    """A finished gate report that says what it is before it says anything else.

    The report was `gate.json` verbatim: `"ok": true` on its fifth line, then
    several hundred more, and on a PASS those end in the reviewer's advisory
    findings — "cannot fail", "is still grep-only", "no test opens…" — which
    read exactly like the reasons a refusal gives. Four rounds in four days
    aborted a change 20–90 s after it had passed every rung, each reporting a
    refusal that is on no ledger row: #1131 (2026-09-15, "both review attempts
    spent" on a 5-of-5 pass), #1190 (09-16, "review refused clause 4" on a
    first-attempt pass and a green rollback drill), #1053 twice (09-18). The
    model had the whole report each time; it was not cut. Only an external
    blocker carried a `next`, because only that misreading had happened yet.

    So: `verdict` and `next` first, in words; on a pass the advisories move out
    of the review rung into `notes_that_did_not_block`, labelled as what they
    are (the gate has already written them onto the item); and preflight's
    `allowed` bucket, a second copy of `changed_paths`, is dropped.
    """
    from scripts.automod import review as RV
    rungs = [r for r in (rep.get("rungs") or []) if isinstance(r, dict)]
    failed = next((r for r in rungs if not r.get("ok")), None)
    review = next((r for r in rungs if r.get("name") == "review"), None) or {}
    rdata = review.get("data") if isinstance(review.get("data"), dict) else {}
    notes = None
    if rep.get("ok"):
        graded = str(review.get("detail") or "").split(";", 1)[0].strip()
        verdict = "PASSED — every rung is green" + (f" ({graded})" if graded else "")
        nxt = (f"Call automod_land(\"{rep.get('round_id')}\") now, then end your turn. "
               "Nothing in this report is a refusal, and automod_abort would discard a "
               "change that is ready to land.")
        seams = list(rdata.pop("advisory_seams", None) or [])
        findings = list(rdata.pop("advisory_findings", None) or [])
        if seams or findings:
            notes = {"what": ("The reviewer's notes on a change it PASSED. They did not block, "
                              "the gate has already recorded them on the item, and they are "
                              "not a reason to edit, re-gate or abort."),
                     "seams": seams, "findings": findings}
        # A tests rung that passed over failures predating the round: say whose
        # they are, so the round does not spend itself fixing the tree.
        tests = next((r for r in rungs if r.get("name") == "tests"), None) or {}
        tdata = tests.get("data") if isinstance(tests.get("data"), dict) else {}
        pre_ids = list(tdata.get("pre_existing_failures") or [])
        if pre_ids:
            owner = tdata.get("red_tree_item")
            notes = notes or {"what": ("Notes on a change the gate PASSED. They did not block "
                                       "and are not a reason to edit, re-gate or abort.")}
            notes["pre_existing_failures"] = {
                "ids": pre_ids[:50],
                "note": ((f"these fail at base too and are your own item #{owner}'s contract: "
                          "making them pass is this round's job")
                         if tdata.get("red_tree_item_is_own") else
                         ((f"tracked by item #{owner}" if owner else "tracked on the board as a "
                           "red-tree item") + " — these fail at base too; do not fix them in "
                          "this round"))}
    elif rep.get("next"):
        verdict = (f"NOT JUDGED — the {failed.get('name') if failed else '?'} rung failed for a "
                   f"reason outside your diff")
        nxt = rep["next"]
    elif failed and failed.get("name") == "review":
        fdata = failed.get("data") if isinstance(failed.get("data"), dict) else {}
        attempt = int(fdata.get("review_attempt") or 0)
        if fdata.get("review_premise_unsound"):
            verdict = "REFUSED at review — the reviewer judged the premise unsound"
            nxt = "automod_abort with the reviewer's summary as the reason (branch kept)."
        elif fdata.get("review_retry") and attempt and attempt < RV.REVIEW_MAX_PER_ROUND:
            verdict = f"REFUSED at review — attempt {attempt} of {RV.REVIEW_MAX_PER_ROUND}"
            nxt = ("Fix what the review rung's `detail` names, commit, and call automod_gate "
                   "again. One attempt is left.")
        elif fdata.get("review_retry"):
            verdict = (f"REFUSED at review — attempt {attempt or RV.REVIEW_MAX_PER_ROUND} of "
                       f"{RV.REVIEW_MAX_PER_ROUND}, the last")
            nxt = ("automod_abort with the findings as the reason (branch kept); the item "
                   "comes back with them. No edit, no re-gate, no land.")
        else:
            verdict = "NOT GRADED — the review rung could not run"
            nxt = "Read the review rung's `detail`; it says whether to gate again."
    else:
        name = failed.get("name") if failed else "?"
        verdict = f"FAILED at {name}"
        nxt = f"Fix what the {name} rung's `detail` names, commit, and call automod_gate again."
    for r in rungs:
        data = r.get("data") if isinstance(r.get("data"), dict) else None
        buckets = data.get("buckets") if data else None
        if (r.get("name") == "preflight" and isinstance(buckets, dict)
                and sorted(buckets.get("allowed") or []) == sorted(rep.get("changed_paths") or [])):
            buckets.pop("allowed", None)
    out = {"verdict": verdict, "next": nxt}
    if notes:
        out["notes_that_did_not_block"] = notes
    out.update({k: v for k, v in rep.items() if k != "next"})
    return out


def _abort_refusal(round_id: str, *, discard_passed_gate: bool = False) -> dict | None:
    """Why `automod_abort` will not run as asked, or None.

    An abort removes the worktree. Under a running landing that takes the tree
    out from beneath the promoter; on a round whose gate PASSED at the commit
    it still holds, it throws away a change that needed one more call — which
    is what each of the four rounds in `_with_headline` did. Refused once,
    with the verdict in the refusal; `discard_passed_gate=true` is how a round
    that really means it says so, and the reason goes on the ledger either way.
    The CLI and the reaper call `round.abort` directly and are not asked.
    """
    from scripts.automod import state as S, worktree as W
    if S.land_in_progress(round_id):
        return {"error": (f"a landing is running for {round_id}; aborting would remove the "
                          f"worktree under it. End your turn — the landing reports to the ledger.")}
    gate_path = S.ROUNDS_DIR / round_id / "gate.json"
    if discard_passed_gate or not gate_path.exists():
        return None
    try:
        report = json.loads(gate_path.read_text(encoding="utf-8"))
        head = W.head(W.worktree_path(round_id)) or ""
    except Exception:  # noqa: BLE001 — an unreadable report never blocks an abort
        return None
    if not report.get("ok") or not head or str(report.get("head") or "") != head:
        return None
    review = next((r for r in (report.get("rungs") or [])
                   if isinstance(r, dict) and r.get("name") == "review"), None) or {}
    return {"error": (f"{round_id} PASSED its gate at {head[:8]} — every rung green"
                      + (f" ({str(review.get('detail') or '').split(';', 1)[0].strip()})"
                         if review.get("detail") else "")
                      + ". Aborting discards a change that is ready to land."),
            "next": (f"Call automod_land(\"{round_id}\"), then end your turn. If you read a "
                     f"refusal in the gate report, read its `verdict` line again: advisory notes "
                     f"on a pass are not a refusal. To abandon a passed change anyway, call "
                     f"automod_abort again with discard_passed_gate=true and say why in `reason`.")}


def _amend_clause(round_id: str, clause, text: str, reason: str) -> dict:
    from scripts.automod import backlog as B, state as S
    import yaml
    spec_path = S.ROUNDS_DIR / round_id / "run_spec.yaml"
    if not spec_path.exists():
        return {"error": f"no run spec for {round_id}"}
    run_spec = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    item_id = (run_spec.get("item") or {}).get("id")
    if not item_id:
        return {"error": f"{round_id} is not bound to a backlog item; there is no contract to amend"}
    try:
        rec = B.amend_clause(int(item_id), clause, text, reason, round_id=round_id)
    except ValueError as exc:
        return {"error": str(exc)}
    return {"amended": rec, "item_id": int(item_id),
            "next": ("Gate again. The next review sees the amendment as an <amendments> "
                     "block and ratifies or refuses it — a refusal restores the old text. "
                     "Nothing needs committing for the amendment itself.")}


async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="automod_start",
            description=(
                "Open a self-modification round: creates a git worktree off the current "
                "HEAD and returns its path. Edit that path with the normal Edit/Write/Bash "
                "tools, commit inside it, then call automod_gate. Only one round at a time."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "goal": {"type": "string",
                             "description": "What this round is trying to change."},
                    "item_id": {"type": "integer",
                                "description": ("The backlog item this round implements. "
                                                "Binds the round to its acceptance clauses, "
                                                "which the gate's review rung grades the diff "
                                                "against. Always pass it for an implement round.")},
                    "from_branch": {"type": "string",
                                    "description": ("Resume from this branch (e.g. automod/SM_…) "
                                                    "when a previous round was sent back by review: "
                                                    "the new worktree starts from it, rebased onto "
                                                    "live main, and the old branch is deleted.")},
                },
                "required": ["goal"],
            },
        ),
        Tool(
            name="automod_gate",
            description=(
                "Start the promotion gate on a round: static checks, the full test suite, "
                "the review rung (a second reader grading the diff against the item's "
                "clauses), a candidate venv if requirements changed, a canary boot on "
                "alternate ports, one real agent turn, and a guardian drill if the diff "
                "touches the rollback path. RETURNS IMMEDIATELY — the gate runs detached "
                "for many minutes (the result says how long one takes now); call "
                "automod_gate_wait for the report. Refuses while a gate is already running for the round or while "
                "a background task of yours is in flight."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "round_id": {"type": "string",
                                  "description": "Round id returned by automod_start."},
                    "skip_smoke": {"type": "boolean",
                                   "description": "Skip the live-LLM turn (rung 5). Honoured "
                                                  "ONLY when the model endpoint is actually "
                                                  "unreachable — the gate probes it and runs "
                                                  "the rung anyway if it answers. Either way "
                                                  "the rung is recorded, so a skip is visible "
                                                  "in the promotion record rather than absent "
                                                  "from it."},
                },
                "required": ["round_id"],
            },
        ),
        Tool(
            name="automod_gate_wait",
            description=(
                "Wait for the detached gate started by automod_gate. Blocks up to "
                "wait_seconds (default 240) and returns the per-rung report when the "
                "ladder has finished, or {running: true, rungs_so_far: [...]} if it has "
                "not — call again then. A finished report opens with `verdict` and `next`: "
                "read those first. On a failure the failing rung's detail names the cause."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "round_id": {"type": "string",
                                 "description": "Round id returned by automod_start."},
                    "wait_seconds": {"type": "integer",
                                     "description": "How long to block, 0–540. Default 240."},
                },
                "required": ["round_id"],
            },
        ),
        Tool(
            name="automod_amend_clause",
            description=(
                "Amend ONE acceptance clause the review rung just judged `unsatisfiable` "
                "(no diff could satisfy it as written). Replaces the clause text on the "
                "backlog item and records the amendment as pending; the next review "
                "ratifies it or refuses it and restores the old text. Refused for any "
                "clause the last review did not mark unsatisfiable. Amend to the nearest "
                "clause that is satisfiable and still what the item asked for — never to "
                "something weaker."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "round_id": {"type": "string",
                                 "description": "Round id returned by automod_start."},
                    "clause": {"type": "integer", "description": "1-based clause index."},
                    "text": {"type": "string", "description": "The amended clause."},
                    "reason": {"type": "string",
                               "description": "Why the original could not be satisfied."},
                },
                "required": ["round_id", "clause", "text", "reason"],
            },
        ),
        Tool(
            name="automod_land",
            description=(
                "Promote a round whose gate passed. Returns IMMEDIATELY: the landing "
                "runs detached, because it restarts this very process. It waits for the "
                "backend to go IDLE, fast-forwards, restarts MCP then backend, and "
                "verifies the running commit changed. END YOUR TURN once this returns — "
                "your own turn is what keeps the backend busy, so polling starves the "
                "gate the landing is waiting on. Check automod_status on your next "
                "turn: automod_status shows the landing, then the guardian's observation "
                "window, then the commit settling to last-known-good."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "round_id": {"type": "string",
                                  "description": "Round id whose gate passed."},
                    "dry_run": {"type": "boolean",
                                "description": "Report what would be promoted without "
                                               "touching the live tree."},
                },
                "required": ["round_id"],
            },
        ),
        Tool(
            name="automod_status",
            description=("Current self-modification state: last known good commit, the "
                         "promotion being observed, halt/broken flags, and recent ledger events."),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="automod_abort",
            description=("Abandon a round. Removes its worktree but KEEPS the branch, which "
                         "is the only record of what was attempted. Refused while the round's "
                         "landing is running, and refused once for a round whose gate PASSED "
                         "at its current commit — that round wants automod_land."),
            inputSchema={
                "type": "object",
                "properties": {"round_id": {"type": "string",
                                            "description": "Round id to abandon."},
                               "reason": {"type": "string",
                                          "description": ("Why — one line. When the review rung "
                                                          "sent the round back, its findings.")},
                               "discard_passed_gate": {
                                   "type": "boolean",
                                   "description": ("Only to abandon a change whose gate PASSED at "
                                                   "its current commit. Say why in `reason`.")}},
                "required": ["round_id"],
            },
        ),
        Tool(
            name="automod_vault_land",
            description=("Land a change to the Obsidian vault (~/obsidian) through the "
                         "self-modification loop. The vault is a live tree with no worktree, "
                         "so this runs AFTER you edit: it validates only the paths you name "
                         "(front matter must parse; a touched skill must still load, a touched "
                         "autonomy task must still parse, SOUL.md/memory/skill edits must still "
                         "build the system prompt), commits exactly those paths on the vault's "
                         "main, and records the sha in the automod ledger. If validation fails "
                         "the paths are REVERTED (tracked files back to HEAD, new files deleted) "
                         "and nothing lands. .obsidian/**, .git/** and .trash/** are denied. "
                         "Requires Inner Voice on the turn, like automod_start."),
            inputSchema={
                "type": "object",
                "properties": {
                    "paths": {"type": "array", "items": {"type": "string"},
                              "description": "Vault-relative paths you changed, e.g. "
                                             "skills/foo/SKILL.md"},
                    "message": {"type": "string", "description": "Commit message"},
                    "item_id": {"type": "integer",
                                "description": "Backlog item this implements, if any"},
                },
                "required": ["paths", "message"],
            },
        ),
        Tool(
            name="automod_vault_revert",
            description=("Revert a vault commit made by automod_vault_land (a plain git "
                         "revert), recorded in the ledger. For a landed vault change that "
                         "turned out wrong."),
            inputSchema={
                "type": "object",
                "properties": {"sha": {"type": "string", "description": "The commit to revert"},
                               "reason": {"type": "string",
                                          "description": "Why; recorded in the ledger"}},
                "required": ["sha"],
            },
        ),
        Tool(
            name="automod_rollback",
            description=("Ask the guardian to revert the live tree and restart. Returns "
                         "immediately; the guardian acts within seconds and picks the "
                         "target the same way it would for a crash. Normally it does "
                         "this on its own — use this only when you need to force it."),
            inputSchema={
                "type": "object",
                "properties": {"reason": {"type": "string",
                                          "description": "Why a manual rollback is needed; "
                                                         "recorded in the ledger."}},
                "required": ["reason"],
            },
        ),
    ]


async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    import asyncio

    def _sync():
        from scripts.automod import round as R, state as S

        if name == "automod_status":
            return text_result(json.dumps(R.status(), indent=2, default=str))

        if not _enabled() and name != "automod_abort":
            return text_result(_err(
                "automod.enabled is false in config.yaml. The self-modification loop "
                "ships inert; enabling it is a deliberate human decision."))

        if name == "automod_start":
            goal = (arguments.get("goal") or "").strip()
            if not goal:
                return text_result(_err("goal is required"))
            gate = _inner_voice_gate("open a round")
            if gate:
                return text_result(json.dumps(gate, indent=2))
            item_id = arguments.get("item_id")
            try:
                item_id = int(item_id) if item_id not in (None, "") else None
            except (TypeError, ValueError):
                return text_result(_err(f"item_id must be an integer, got {item_id!r}"))
            from_branch = str(arguments.get("from_branch") or "").strip() or None
            from agent_mcp import _task_registry
            return text_result(json.dumps(
                R.start(goal, item_id=item_id, from_branch=from_branch, opened_by="tool",
                        session_id=str(_task_registry.current_session_id.get("") or "")),
                indent=2))

        if name == "automod_gate":
            rid = arguments.get("round_id") or ""
            return text_result(json.dumps(
                _gate_detached(rid, skip_smoke=bool(arguments.get("skip_smoke"))), indent=2))

        if name == "automod_gate_wait":
            rid = arguments.get("round_id") or ""
            try:
                wait = int(arguments.get("wait_seconds", 240))
            except (TypeError, ValueError):
                wait = 240
            return text_result(json.dumps(_gate_wait(rid, wait_seconds=wait), indent=2))

        if name == "automod_amend_clause":
            return text_result(json.dumps(_amend_clause(
                str(arguments.get("round_id") or ""), arguments.get("clause"),
                str(arguments.get("text") or ""), str(arguments.get("reason") or "")), indent=2))

        if name == "automod_land":
            rid = arguments.get("round_id") or ""
            if bool(arguments.get("dry_run")):
                return text_result(json.dumps(R.land(rid, dry_run=True), indent=2))
            return text_result(json.dumps(_land_detached(rid), indent=2))

        if name == "automod_abort":
            rid = str(arguments.get("round_id") or "")
            refusal = _abort_refusal(
                rid, discard_passed_gate=bool(arguments.get("discard_passed_gate")))
            if refusal:
                return text_result(json.dumps(refusal, indent=2))
            return text_result(json.dumps(R.abort(rid, reason=str(arguments.get("reason") or "")),
                                          indent=2))

        if name == "automod_vault_land":
            from scripts.automod import review as RV, vault_round as VR
            # The second reader for vault rounds, wired here and not at import
            # in vault_round: tests and the CLI must not reach for the network.
            if VR.GRADER is None:
                VR.GRADER = RV.grade_vault
            gate = _inner_voice_gate("land a vault change")
            if gate:
                return text_result(json.dumps(gate, indent=2))
            paths = [str(x) for x in (arguments.get("paths") or []) if str(x).strip()]
            # The turn's own session id, from the `_meta` the harness stamps — not
            # from `arguments`, which the caller controls. It is the attribution
            # that survives an omitted optional `item_id`, and 56 of the 172
            # `vault_land` rows in the ledger on 2026-09-21 carry no item, so a
            # vault round that landed would otherwise be recorded as not landed
            # (`scripts/automod/backlog.py:round_landing_rows`).
            from agent_mcp import _task_registry
            try:
                out = VR.land(paths, str(arguments.get("message") or ""),
                              item_id=arguments.get("item_id"),
                              session_id=str(_task_registry.current_session_id.get("") or ""))
            except VR.VaultRoundError as exc:
                return text_result(_err(str(exc)))
            out["note"] = ("Committed on the vault's main and live already — nothing "
                           "restarts. automod_vault_revert undoes it.")
            return text_result(json.dumps(out, indent=2))

        if name == "automod_vault_revert":
            from scripts.automod import vault_round as VR
            try:
                out = VR.revert(str(arguments.get("sha") or ""),
                                str(arguments.get("reason") or "manual"))
            except VR.VaultRoundError as exc:
                return text_result(_err(str(exc)))
            return text_result(json.dumps(out, indent=2))

        if name == "automod_rollback":
            reason = arguments.get("reason") or "manual"
            # Handed to the guardian, never performed here. A rollback stops
            # lloyd-mcp — this process — so doing it inline meant issuing the
            # stop that kills the caller and then never reaching `git reset`.
            # The stack went down and the tree did not move: the one outcome
            # worse than either doing it or not. The guardian is outside that
            # blast radius and already owns evidence preservation, retries,
            # the denylist and flap protection.
            req = S.request_rollback(reason=str(reason)[:1000], trigger="manual")
            return text_result(json.dumps({
                "requested": True,
                "note": ("The guardian performs this within a few seconds. It picks the "
                         "target the same way it would for a crash: the promotion's own "
                         "recorded rollback_target first, never a stranded pointer. "
                         "Poll automod_status for the outcome."),
                "request": req,
            }, indent=2))

        return text_result(_err(f"Unknown tool: {name}"))

    try:
        return await asyncio.to_thread(_sync)
    except Exception as exc:
        logger.exception("automod tool %s failed", name)
        return text_result(_err(f"{type(exc).__name__}: {exc}"))
