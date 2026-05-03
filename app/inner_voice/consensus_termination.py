"""Inner Voice (#345) Stage 4 — consensus termination + escape hatches.

When agent emits ``SIGNAL:TASK_COMPLETE`` on an ambient turn, this module
runs a last-mile check: does the critic ensemble actually agree that the
task is done? If the aggregated severity crosses the veto threshold, we
inject a ``<inner-voice kind='please-continue'>…</inner-voice>`` ambient and
agent gets one more turn. Otherwise we accept termination.

The four escape hatches that keep this loop bounded:

  1. **critic timeout** — every persona errored (HTTP timeout, parse
     failure-after-retry, etc). We have no signal so we accept. Per
     ``inner_voice.consensus_termination.critic_timeout_defaults_to_agent``.
  2. **Three-strike accept** — three consecutive vetoes for this session
     with no progress signal between them. agent is probably stuck, not
     lying; further nudges won't unstick it. Accept, log, escalate later
     if it persists. Per ``inner_voice.consensus_termination.three_strike_accept``.
  3. **Max nudges per session** — the session's ``nudge_count`` already
     hit ``inner_voice.throughput.max_nudges_per_session``. We escalate
     instead of nudging again: write a backlog task plus emit a push
     notification event. The user gets a real-world signal that an
     autonomous run got stuck.
  4. **Hard max turns** — the autonomy scheduler signals when a session
     has hit ``inner_voice.consensus_termination.hard_max_turns``. We
     accept termination AND escalate so the operator sees the run.

All decisions are written to the event log as
``inner_voice.consensus_termination_proposal`` (the proposal to terminate
the turn) and ``inner_voice.consensus_termination_decision`` (the verdict
plus the specific hatch that fired, if any).

State for this module is in-process only:

  * ``_session_nudge_counts`` — total nudges fired per session_id.
  * ``_session_consecutive_vetoes`` — current streak of vetoes without
    an intervening progress signal.
  * ``_session_escalations`` — how many times we've escalated a session
    (for telemetry; doesn't gate further behavior).

The module never raises into its caller. On any unrecoverable error the
decision falls back to ``accepted`` with an error string in the rationale,
so the chat path keeps moving.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.config import CONFIG
from app import event_log as _event_log
from app.inner_voice.critic import Critique

logger = logging.getLogger("lloyd-server")


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------

# Match SIGNAL:TASK_COMPLETE on its own line OR at the end of the response.
# Stage 1's heuristic uses the same anchor — keep them in sync. The match
# is case-sensitive (the protocol is `SIGNAL:TASK_COMPLETE` exactly).
_TASK_COMPLETE_RE = re.compile(
    r"(?m)^\s*SIGNAL:TASK_COMPLETE\s*$|SIGNAL:TASK_COMPLETE\s*\Z"
)
# Some callers also want to suppress the consensus-termination check when
# the agent emitted BLOCKED — that's a deliberate refusal, not a premature
# stop. STAGE_COMPLETE on a non-final pipeline stage is also a continue
# signal in disguise; we only run termination check on TASK_COMPLETE.
_BLOCKED_RE = re.compile(r"(?m)^\s*SIGNAL:BLOCKED:.*$")


def has_task_complete_signal(response_text: str) -> bool:
    """Return True if the response ends with (or contains a standalone)
    ``SIGNAL:TASK_COMPLETE`` and does NOT also contain ``SIGNAL:BLOCKED``.

    Mirrors the precedence rule from Stage 1's heuristic: a BLOCKED in the
    same response wins (the agent is correctly refusing) — we don't want
    to "veto a refusal" by injecting a please-continue.
    """
    if not response_text:
        return False
    if _BLOCKED_RE.search(response_text):
        return False
    return bool(_TASK_COMPLETE_RE.search(response_text))


# ---------------------------------------------------------------------------
# Per-session state — module-level dicts, in-process only
# ---------------------------------------------------------------------------

# total nudges (please-continue ambient enqueued) for this session
_session_nudge_counts: dict[str, int] = {}
# consecutive vetoes since last progress signal
_session_consecutive_vetoes: dict[str, int] = {}
# escalations recorded (for telemetry)
_session_escalations: dict[str, int] = {}


def get_nudge_count(session_id: str) -> int:
    """Public accessor for `/api/inner_voice/status` — exposes the
    nudge_count the frontend pill renders.
    """
    return int(_session_nudge_counts.get(session_id, 0))


def get_consecutive_veto_count(session_id: str) -> int:
    return int(_session_consecutive_vetoes.get(session_id, 0))


def reset_consecutive_vetoes(session_id: str) -> None:
    """Called when a "progress signal" is seen (a turn that did NOT trigger
    a veto). The streak resets so a later veto starts the count fresh.
    """
    _session_consecutive_vetoes[session_id] = 0


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _ct_cfg() -> dict[str, Any]:
    iv = CONFIG.get("inner_voice") or {}
    ct = dict(iv.get("consensus_termination") or {})
    ct.setdefault("enabled", True)
    ct.setdefault("critic_timeout_defaults_to_agent", True)
    ct.setdefault("three_strike_accept", True)
    ct.setdefault("hard_max_turns", 60)
    return ct


def _max_nudges_per_session() -> int:
    iv = CONFIG.get("inner_voice") or {}
    tp = iv.get("throughput") or {}
    return int(tp.get("max_nudges_per_session", 2))


# ---------------------------------------------------------------------------
# Stage 6 helpers — shared with the steer-dispatch path in messages.py.
# Exposing the per-session counter + cap check + bump op so the steer path
# uses the SAME budget consensus_termination uses. Otherwise both paths
# could each spend max_nudges_per_session on the same session.
# ---------------------------------------------------------------------------


def is_self_correct_on_nudge_enabled() -> bool:
    """Config gate for Stage 6 steer dispatch (auto-fire ambient retry when
    the post-loop ensemble lands in the nudge band [severity_threshold,
    veto_severity_threshold)).

    Default true. Set ``inner_voice.self_correct_on_nudge: false`` in
    config.yaml to fall back to log-only behavior.
    """
    iv = CONFIG.get("inner_voice")
    if not iv:
        return False
    return bool(iv.get("self_correct_on_nudge", True))


def can_consume_nudge_budget(session_id: str) -> bool:
    """Return True if this session has nudge headroom.

    Steer dispatch + consensus_termination veto share this budget. The
    cap is `inner_voice.throughput.max_nudges_per_session` (default 2).
    """
    return _session_nudge_counts.get(session_id, 0) < _max_nudges_per_session()


def consume_nudge_budget(session_id: str) -> int:
    """Increment the nudge counter for this session and return the new
    value. Caller is responsible for having checked `can_consume_nudge_budget`
    first; this only updates state.
    """
    _session_nudge_counts[session_id] = _session_nudge_counts.get(session_id, 0) + 1
    return _session_nudge_counts[session_id]


def _veto_severity_threshold() -> float:
    iv = CONFIG.get("inner_voice") or {}
    dis = iv.get("disagreement") or {}
    return float(dis.get("veto_severity_threshold", 0.85))


def is_enabled() -> bool:
    return bool(_ct_cfg().get("enabled", True))


# ---------------------------------------------------------------------------
# Decision struct
# ---------------------------------------------------------------------------


@dataclass
class TerminationDecision:
    """The verdict on a single ``SIGNAL:TASK_COMPLETE`` event.

    ``action`` is one of:
      - ``accepted`` — termination accepted. No injection.
      - ``vetoed`` — critic ensemble disagrees at veto severity.
        Caller should enqueue ``please_continue_kwargs`` and record the
        intervention.
      - ``accepted_brain2_timeout`` — every persona errored. We have no
        signal, so we accept (default-to-Brain-1).
      - ``accepted_three_strike`` — third consecutive veto without a
        progress signal. Caller does NOT inject; we treat it as stuck.
      - ``escalated_max_nudges`` — max_nudges_per_session reached; caller
        writes a backlog task + emits push-notif event instead of nudging.
      - ``escalated_max_turns`` — caller pre-flagged that hard_max_turns
        was hit; we accept termination AND escalate.

    ``please_continue_kwargs`` is set ONLY when ``action == 'vetoed'``.
    """

    action: str
    rationale: str
    severity_max: float = 0.0
    disagree_count: int = 0
    nudge_count_after: int = 0
    consecutive_vetoes_after: int = 0
    hatch_fired: str | None = None
    please_continue_kwargs: dict[str, Any] | None = None
    escalation_kwargs: dict[str, Any] | None = None
    triggered_by_critique_id_hint: int | None = None  # populated by caller

    def to_event_dict(self, *, ensemble_name: str, turn_id: str) -> dict[str, Any]:
        return {
            "action": self.action,
            "rationale": self.rationale,
            "severity_max": self.severity_max,
            "disagree_count": self.disagree_count,
            "nudge_count_after": self.nudge_count_after,
            "consecutive_vetoes_after": self.consecutive_vetoes_after,
            "hatch_fired": self.hatch_fired,
            "ensemble_name": ensemble_name,
            "turn_id": turn_id,
            "max_nudges_per_session": _max_nudges_per_session(),
            "veto_severity_threshold": _veto_severity_threshold(),
        }


# ---------------------------------------------------------------------------
# Aggregation helpers — duplicated minimally from ensemble._aggregate so
# this module doesn't import ensemble (avoid a cycle).
# ---------------------------------------------------------------------------


def _summarize(critiques: list[Critique]) -> tuple[float, int, int]:
    """Return (severity_max, disagree_count, error_count) for the critique
    list. Empty list returns (0, 0, 0).
    """
    severity_max = 0.0
    disagree_count = 0
    error_count = 0
    for c in critiques:
        if c.error is not None:
            error_count += 1
            continue
        if c.disagrees:
            disagree_count += 1
            if c.severity > severity_max:
                severity_max = c.severity
    return severity_max, disagree_count, error_count


# ---------------------------------------------------------------------------
# Please-continue ambient construction
# ---------------------------------------------------------------------------


def make_please_continue_ambient(
    *,
    turn_id: str,
    severity: float,
    reasons: list[str],
    response_excerpt: str,
) -> dict[str, str]:
    """Build the AmbientPrefetchEntry kwargs that surface a vetoed
    ``SIGNAL:TASK_COMPLETE``.

    Same shape as ``make_completion_nudge_entry`` and
    ``make_drift_cancel_ambient`` — kwargs passable directly into
    ``AmbientPrefetchEntry(**kwargs, enqueued_at=…)``.

    The wrapping ``<inner-voice kind='please-continue'>`` tag is what the
    spec requested. agent sees this as the leading content of its next
    ambient turn (via ``<context>`` injection).
    """
    excerpt = (response_excerpt or "")[:300]
    reason_block = "\n".join(f"- {r}" for r in reasons[:5] if r) or "- (no reasons)"
    return {
        "source": "inner_voice:consensus_termination:vetoed",
        "summary": (
            f"Previous ambient turn ({turn_id}) emitted SIGNAL:TASK_COMPLETE "
            f"but critic ensemble vetoed at severity {severity:.2f}."
        ),
        "content": (
            "<inner-voice kind='please-continue'>\n"
            "Inner Voice (#345 Stage 4) vetoed your last SIGNAL:TASK_COMPLETE. "
            "The critic ensemble disagreed at veto severity — the deliverable "
            "doesn't look done.\n\n"
            "Reasons:\n"
            f"{reason_block}\n\n"
            "Please continue the task. If you genuinely believe it IS done and "
            "you can defend that against the reasons above, emit "
            "SIGNAL:TASK_COMPLETE again with a one-line justification "
            "addressing each reason; Inner Voice will accept the second pass.\n\n"
            f"Tail of your response: {excerpt!r}\n"
            "</inner-voice>"
        ),
        "dedup_key": f"inner_voice:consensus_termination:{turn_id}",
    }


def make_escalation_backlog_kwargs(
    *,
    session_id: str,
    turn_id: str,
    nudge_count: int,
    severity_max: float,
    last_reason: str,
) -> dict[str, Any]:
    """Build the kwargs for a ``backlog_write_task`` call when the
    max_nudges escape hatch fires. The caller is responsible for actually
    making the call (this module avoids importing the MCP layer).

    Returns a flat dict suitable for `record_inner_voice_intervention`'s
    ``content`` field AND for posting to the backlog tool.
    """
    title = (
        f"Inner Voice escalation: session {session_id[:18]} hit "
        f"max_nudges ({nudge_count})"
    )
    body = (
        f"Inner Voice (#345 Stage 4) escalated this session because the "
        f"critic ensemble vetoed termination {nudge_count} times — the "
        f"max_nudges_per_session ceiling. The agent is either confidently "
        f"asserting completion that the ensemble disagrees with, OR genuinely "
        f"stuck.\n\n"
        f"Session: `{session_id}`\n"
        f"Final turn: `{turn_id}`\n"
        f"Severity at last veto: {severity_max:.2f}\n"
        f"Last reason: {last_reason or '(no reason)'}\n\n"
        f"Action: review the session transcript, decide if this is a real "
        f"deliverable gap or a calibration issue with the Stage 4 ensemble. "
        f"If calibration, tune the persona prompts."
    )
    return {
        "title": title,
        "body": body,
        "tags": ["inner-voice", "stage-4", "escalation", "agent-stability"],
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


# Optional external escalation hook — caller can wire a Discord/Email
# pusher that this module fires when an escalation lands. Best-effort
# (no-op if unset).
EscalationFn = Callable[[dict[str, Any]], Awaitable[None]]
_escalation_hook: EscalationFn | None = None


def set_escalation_hook(fn: EscalationFn | None) -> None:
    """Wire (or unwire) the external push-notification fn. Called from
    server.py at startup. None disables.
    """
    global _escalation_hook
    _escalation_hook = fn


async def evaluate(
    *,
    session_id: str,
    turn_id: str,
    response_text: str,
    critiques: list[Critique],
    ensemble_name: str,
    hard_max_turns_hit: bool = False,
) -> TerminationDecision:
    """Evaluate one TASK_COMPLETE event against the ensemble's critiques.

    Caller is expected to:
      1. Have already run the post-loop ensemble (so ``critiques`` is
         the actual critic verdict, not a freshly-fired pass).
      2. Have detected ``SIGNAL:TASK_COMPLETE`` in ``response_text`` via
         ``has_task_complete_signal()``.
      3. Pass ``hard_max_turns_hit=True`` if the autonomy scheduler is
         enforcing a turn ceiling on this session.

    On decision:
      - ``vetoed`` → caller enqueues ``please_continue_kwargs`` ambient,
        records the intervention via ``record_inner_voice_intervention``.
      - ``escalated_*`` → caller writes the backlog task (kwargs in
        ``escalation_kwargs``), invokes the push-notification hook.
      - ``accepted_*`` → caller does nothing; the turn ends normally.

    The function ALWAYS writes the proposal + decision events. Returns
    even on internal failure (with action=accepted, rationale=error).
    """
    # Stage 4 gate — runtime kill switch.
    if not is_enabled():
        return TerminationDecision(
            action="accepted",
            rationale="consensus_termination disabled in config",
        )

    # Always log the proposal. The decision event follows.
    severity_max, disagree_count, error_count = _summarize(critiques)
    veto_floor = _veto_severity_threshold()
    persona_count = len(critiques)

    try:
        _event_log.log_event(
            session_id,
            "inner_voice.consensus_termination_proposal",
            {
                "ensemble_name": ensemble_name,
                "personas_invoked": persona_count,
                "personas_disagreed": disagree_count,
                "personas_errored": error_count,
                "severity_max": severity_max,
                "veto_severity_threshold": veto_floor,
                "would_veto": severity_max >= veto_floor,
                "response_tail": (response_text or "")[-500:],
            },
            turn_id=turn_id,
        )
    except Exception as e:
        logger.warning("consensus_termination proposal event log failed: %s", e)

    # ── Hatch 4: hard_max_turns ──────────────────────────────────────────
    # Caller pre-flagged. We accept termination (no injection) AND
    # escalate so the run gets human visibility. Highest priority hatch
    # because it's an external-system gate (autonomy scheduler told us).
    if hard_max_turns_hit:
        nudges = _session_nudge_counts.get(session_id, 0)
        last_reason = next(
            (c.reason for c in critiques if c.disagrees and c.error is None and c.reason),
            "",
        )
        esc_kwargs = make_escalation_backlog_kwargs(
            session_id=session_id,
            turn_id=turn_id,
            nudge_count=nudges,
            severity_max=severity_max,
            last_reason=last_reason or "(hard_max_turns hit; no recent veto)",
        )
        _session_escalations[session_id] = _session_escalations.get(session_id, 0) + 1
        decision = TerminationDecision(
            action="escalated_max_turns",
            rationale=(
                f"hard_max_turns hit (config: {_ct_cfg().get('hard_max_turns')}); "
                "accepting termination and escalating to backlog + push notif"
            ),
            severity_max=severity_max,
            disagree_count=disagree_count,
            nudge_count_after=nudges,
            consecutive_vetoes_after=_session_consecutive_vetoes.get(session_id, 0),
            hatch_fired="hard_max_turns",
            escalation_kwargs=esc_kwargs,
        )
        await _emit_decision_event(session_id, turn_id, ensemble_name, decision)
        await _maybe_fire_escalation_hook(decision, session_id=session_id, turn_id=turn_id)
        return decision

    # ── Hatch 1: critic timeout ─────────────────────────────────────────
    # Every persona errored. We have no signal — the spec says
    # "default to agent" (accept termination). No streak update; this
    # is an infrastructure failure, not a model decision.
    if persona_count > 0 and error_count == persona_count:
        if _ct_cfg().get("critic_timeout_defaults_to_agent", True):
            decision = TerminationDecision(
                action="accepted_brain2_timeout",
                rationale=(
                    f"all {persona_count} personas errored "
                    f"({error_count} timeouts/parse-failures); "
                    "default-to-Brain-1 accepts termination"
                ),
                severity_max=0.0,
                disagree_count=0,
                nudge_count_after=_session_nudge_counts.get(session_id, 0),
                consecutive_vetoes_after=_session_consecutive_vetoes.get(session_id, 0),
                hatch_fired="brain2_timeout",
            )
            await _emit_decision_event(session_id, turn_id, ensemble_name, decision)
            return decision

    # ── No veto path: severity below threshold OR no disagreement ────────
    if severity_max < veto_floor:
        # The streak resets — this is a "progress signal".
        reset_consecutive_vetoes(session_id)
        decision = TerminationDecision(
            action="accepted",
            rationale=(
                f"severity_max={severity_max:.2f} < veto_floor={veto_floor:.2f}; "
                f"disagree_count={disagree_count}/{persona_count}; "
                "ensemble agrees task is done"
            ),
            severity_max=severity_max,
            disagree_count=disagree_count,
            nudge_count_after=_session_nudge_counts.get(session_id, 0),
            consecutive_vetoes_after=0,
        )
        await _emit_decision_event(session_id, turn_id, ensemble_name, decision)
        return decision

    # ── Veto path: severity above threshold AND disagree ─────────────────
    # Now apply the remaining hatches (max_nudges, three-strike) before
    # actually injecting.

    # Hatch 3: max_nudges_per_session — if the session has already nudged
    # max_nudges times, we DON'T inject again. Escalate instead.
    nudges = _session_nudge_counts.get(session_id, 0)
    cap = _max_nudges_per_session()
    if nudges >= cap:
        last_reason = next(
            (c.reason for c in critiques if c.disagrees and c.error is None and c.reason),
            "",
        )
        esc_kwargs = make_escalation_backlog_kwargs(
            session_id=session_id,
            turn_id=turn_id,
            nudge_count=nudges,
            severity_max=severity_max,
            last_reason=last_reason,
        )
        _session_escalations[session_id] = _session_escalations.get(session_id, 0) + 1
        decision = TerminationDecision(
            action="escalated_max_nudges",
            rationale=(
                f"nudge_count={nudges} >= max_nudges_per_session={cap}; "
                "vetoing further nudge in favor of human escalation"
            ),
            severity_max=severity_max,
            disagree_count=disagree_count,
            nudge_count_after=nudges,
            consecutive_vetoes_after=_session_consecutive_vetoes.get(session_id, 0),
            hatch_fired="max_nudges",
            escalation_kwargs=esc_kwargs,
        )
        await _emit_decision_event(session_id, turn_id, ensemble_name, decision)
        await _maybe_fire_escalation_hook(decision, session_id=session_id, turn_id=turn_id)
        return decision

    # Hatch 2: three-strike — if the streak is already at 2, this would
    # be the third veto in a row with no progress. Accept and emit a
    # `_session_escalations` ping (no backlog write — it's a softer
    # circuit breaker than max_nudges).
    streak = _session_consecutive_vetoes.get(session_id, 0)
    if _ct_cfg().get("three_strike_accept", True) and streak >= 2:
        new_streak = streak + 1
        _session_consecutive_vetoes[session_id] = new_streak
        decision = TerminationDecision(
            action="accepted_three_strike",
            rationale=(
                f"three-strike: consecutive_vetoes={new_streak} >= 3; "
                "agent is stuck or persona is miscalibrated; accepting "
                "termination without further nudge"
            ),
            severity_max=severity_max,
            disagree_count=disagree_count,
            nudge_count_after=nudges,
            consecutive_vetoes_after=new_streak,
            hatch_fired="three_strike",
        )
        await _emit_decision_event(session_id, turn_id, ensemble_name, decision)
        return decision

    # ── Veto: enqueue please-continue ambient ───────────────────────────
    reasons = [c.reason for c in critiques if c.disagrees and c.error is None and c.reason]
    please_kwargs = make_please_continue_ambient(
        turn_id=turn_id,
        severity=severity_max,
        reasons=reasons,
        response_excerpt=response_text or "",
    )
    new_nudges = nudges + 1
    new_streak = streak + 1
    _session_nudge_counts[session_id] = new_nudges
    _session_consecutive_vetoes[session_id] = new_streak

    decision = TerminationDecision(
        action="vetoed",
        rationale=(
            f"severity_max={severity_max:.2f} >= veto_floor={veto_floor:.2f}; "
            f"disagree_count={disagree_count}/{persona_count}; "
            f"injecting please-continue ambient (nudge {new_nudges}/{cap})"
        ),
        severity_max=severity_max,
        disagree_count=disagree_count,
        nudge_count_after=new_nudges,
        consecutive_vetoes_after=new_streak,
        please_continue_kwargs=please_kwargs,
    )
    await _emit_decision_event(session_id, turn_id, ensemble_name, decision)
    return decision


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _emit_decision_event(
    session_id: str, turn_id: str, ensemble_name: str, decision: TerminationDecision
) -> None:
    try:
        _event_log.log_event(
            session_id,
            "inner_voice.consensus_termination_decision",
            decision.to_event_dict(ensemble_name=ensemble_name, turn_id=turn_id),
            turn_id=turn_id,
        )
    except Exception as e:
        logger.warning("consensus_termination decision event log failed: %s", e)


async def _maybe_fire_escalation_hook(
    decision: TerminationDecision,
    *,
    session_id: str,
    turn_id: str,
) -> None:
    """Fire the optional push-notification hook. Never raises."""
    if _escalation_hook is None:
        return
    try:
        payload = {
            "session_id": session_id,
            "turn_id": turn_id,
            "action": decision.action,
            "hatch_fired": decision.hatch_fired,
            "severity_max": decision.severity_max,
            "nudge_count": decision.nudge_count_after,
            "rationale": decision.rationale,
            "escalation_kwargs": decision.escalation_kwargs or {},
        }
        await asyncio.wait_for(_escalation_hook(payload), timeout=5.0)
    except asyncio.TimeoutError:
        logger.warning(
            "consensus_termination escalation hook timeout (5s) session=%s turn=%s",
            session_id, turn_id,
        )
    except Exception as e:
        logger.warning("consensus_termination escalation hook failed: %s", e)


__all__ = [
    "TerminationDecision",
    "evaluate",
    "has_task_complete_signal",
    "make_please_continue_ambient",
    "make_escalation_backlog_kwargs",
    "set_escalation_hook",
    "get_nudge_count",
    "get_consecutive_veto_count",
    "reset_consecutive_vetoes",
    "is_enabled",
]
