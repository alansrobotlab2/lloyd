"""Inner Voice (#345) Stage 5 — intervention grading pass.

When Inner Voice fires an intervention (kind ``continue`` / ``escalate``)
on turn N, the verdict's quality can only be measured *after* turn N+1
lands — did the agent actually address the critique, or did it ignore /
restate / dodge it?

This module is the *grading pass*: a third the critic call that fires after
every ambient turn completes, looks up any ungraded interventions in
the session, and asks the ``grader`` persona to judge whether the
just-finished turn (the "outcome turn") addressed each one.

Output writes back into ``inner_voice_interventions``:

  * ``outcome_turn_id``   — the turn we graded against
  * ``outcome_addressed`` — True / False / None (ambiguous)
  * ``outcome_summary``   — one short sentence describing what the
    outcome turn did relative to the critique
  * ``graded_at``         — timestamp

Only interventions whose ``target_turn_id != outcome_turn_id`` are
candidates — we never grade an intervention against the very turn that
triggered it (the new please-continue ambient is the *next* turn).

Throughput knobs (from ``inner_voice.grading``):

  * ``enabled``                  — kill switch (default true)
  * ``persona``                  — grader persona file (default ``grader``)
  * ``max_per_session_per_run``  — cap grading work per turn (default 3)
  * ``only_grade_within_turns``  — skip if more than N turns elapsed
                                    since the intervention (avoid grading
                                    cold interventions where the trail is
                                    cold)

Failure semantics: every error is caught and logged. The chat path never
sees a grading exception. Failed grades leave ``outcome_addressed`` NULL
so the next sweep can retry, but the ``graded_at`` field is *not* set so
the row remains in the ungraded queue.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from app.config import CONFIG
from app import event_log as _event_log
from app.inner_voice import critic as _critic
from app.inner_voice.ensemble import _read_persona_file, _truncate
from usage_store import (
    list_ungraded_interventions,
    update_intervention_outcome,
)

logger = logging.getLogger("lloyd-server")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _grading_cfg() -> dict[str, Any]:
    """Read ``inner_voice.grading`` with defaults applied per call so
    config edits take effect without a restart.
    """
    iv = CONFIG.get("inner_voice") or {}
    cfg = dict(iv.get("grading") or {})
    cfg.setdefault("enabled", True)
    cfg.setdefault("persona", "grader")
    cfg.setdefault("max_per_session_per_run", 3)
    cfg.setdefault("only_grade_within_turns", 5)
    return cfg


def _is_grading_enabled() -> bool:
    iv = CONFIG.get("inner_voice")
    if not iv:
        return False
    return bool(_grading_cfg().get("enabled", True))


# ---------------------------------------------------------------------------
# JSON parser for the grader's `{"addressed": ...}` output
# ---------------------------------------------------------------------------


_GRADER_PREFILL = '{"addressed":'


def _extract_first_json_object(text: str) -> str | None:
    """Same brace-counting extractor as critic.py, kept local so this module
    isn't coupled to critic's private helpers.
    """
    if not text:
        return None
    in_str = False
    esc = False
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start: i + 1]
            if depth < 0:
                return None
    return None


def _parse_grader_json(raw: str) -> dict[str, Any] | None:
    """Try every reasonable shape for a ``{"addressed": ...}`` object.

    Mirrors `_parse_critic_json` semantics but checks for the ``addressed``
    key instead of ``disagrees``. Returns None on total failure.
    """
    if not raw:
        return None

    # 1. Direct parse.
    try:
        v = json.loads(raw)
        if isinstance(v, dict) and "addressed" in v:
            return v
    except json.JSONDecodeError:
        pass

    # 2. Prefill-continuation parse.
    candidate = (_GRADER_PREFILL + raw).strip()
    try:
        v = json.loads(candidate)
        if isinstance(v, dict) and "addressed" in v:
            return v
    except json.JSONDecodeError:
        pass

    # 3. Embedded JSON object scan.
    blob = _extract_first_json_object(raw)
    if blob:
        try:
            v = json.loads(blob)
            if isinstance(v, dict) and "addressed" in v:
                return v
        except json.JSONDecodeError:
            pass

    # 4. Scalar coercion. raw might be `true,...` or `null,...`.
    head = raw.strip().lstrip(",: ").split(",", 1)[0].rstrip("}").strip().lower()
    if head in ("true", "false", "null"):
        return {
            "addressed": True if head == "true" else (False if head == "false" else None),
            "summary": "scalar fallback",
        }

    return None


def _coerce_addressed(v: Any) -> bool | None:
    """Normalize an ``addressed`` value to True/False/None.

    Strings ('true', 'false', 'null', 'yes', 'no', 'unknown') accepted.
    Anything else → None (ambiguous).
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "addressed", "1"):
            return True
        if s in ("false", "no", "ignored", "0"):
            return False
        if s in ("null", "none", "unknown", "ambiguous", "n/a"):
            return None
    return None


# ---------------------------------------------------------------------------
# Grader prompt assembly
# ---------------------------------------------------------------------------


_INTERVENTION_CONTENT_CAP = 1500
_OUTCOME_TEXT_CAP = 4000
_TASK_INTENT_CAP = 1500


def _summarize_tool_calls(tool_calls: list[dict]) -> str:
    """Short list of tool names from the outcome turn for the prompt.

    Mirrors `ensemble._summarize_tool_calls` but lighter — grader doesn't
    need arg-key fingerprints, just the names.
    """
    out: list[str] = []
    for tc in tool_calls or []:
        name = tc.get("name") or tc.get("function", {}).get("name", "?")
        short = name.rsplit("__", 1)[-1] if "__" in name else name
        out.append(short)
    return ", ".join(out) if out else "(none)"


def _build_grader_prompt(
    *,
    frozen_task_intent: str,
    intervention: dict,
    outcome_response_text: str,
    outcome_tool_calls: list[dict],
) -> str:
    """Assemble the grader's user-message body.

    Tag layout (referenced by examples in `~/obsidian/lloyd/inner_voice/personas/grader.md`):

        <task>...</task>
        <intervention>
        kind: <kind>
        persona: <persona name from triggered_by_critique_id, if known>
        reason: <reason text>
        injected_content: <ambient content snippet>
        </intervention>
        <outcome_response>
        text: ...
        tool_calls: [name1, name2]
        </outcome_response>
    """
    persona = "(unknown)"
    reason = ""
    # The intervention row carries `content` (the injected ambient body).
    # We extract a reason hint by scanning for the persona's reason text in
    # the content — it's typically embedded in the please-continue body.
    body = intervention.get("content") or ""
    # Best-effort persona/reason recovery — the row itself doesn't store
    # them, but the consensus-termination please-continue body usually
    # starts with "Inner Voice ... <persona>, severity ... — <reason>".
    m = re.search(r"\(([\w_]+),\s*severity\s*([\d.]+)\)\s*[—-]\s*(.+?)(?:\.\s|$)", body)
    if m:
        persona = m.group(1)
        reason = m.group(3).strip()
    # Trimmed content for the prompt — full text already lives in the row
    # for forensics.
    content_excerpt = _truncate(body, _INTERVENTION_CONTENT_CAP)
    tool_str = _summarize_tool_calls(outcome_tool_calls)

    return (
        "<task>\n"
        f"{_truncate(frozen_task_intent or '(no task text)', _TASK_INTENT_CAP)}\n"
        "</task>\n\n"
        "<intervention>\n"
        f"kind: {intervention.get('kind', '?')}\n"
        f"persona: {persona}\n"
        f"reason: {reason or '(not extracted from injected content)'}\n"
        f"injected_content: {content_excerpt}\n"
        "</intervention>\n\n"
        "<outcome_response>\n"
        f"text: {_truncate(outcome_response_text or '(empty)', _OUTCOME_TEXT_CAP)}\n"
        f"tool_calls: [{tool_str}]\n"
        "</outcome_response>"
    )


# ---------------------------------------------------------------------------
# Public entry point — wired from messages.py post-loop
# ---------------------------------------------------------------------------


async def grade_outcome_turn(
    *,
    session_id: str,
    outcome_turn_id: str,
    outcome_response_text: str,
    outcome_tool_calls: list[dict],
    frozen_task_intent: str,
) -> dict[str, Any]:
    """Grade any ungraded interventions for `session_id` against the just-
    finished outcome turn. Returns a small summary dict for logging.

    Idempotent: interventions are pulled where ``outcome_turn_id IS NULL``
    so a second call against the same outcome turn finds nothing.

    Best-effort. Catches every exception. The chat path never sees a raise.

    Returns:
        {
          "graded": <int>,           -- count of interventions graded this run
          "skipped": <int>,          -- candidates skipped (e.g. cap reached)
          "errors": <int>,           -- candidates that errored mid-grade
          "total_candidates": <int>, -- candidates surfaced from SQLite
        }
    """
    summary = {
        "graded": 0,
        "skipped": 0,
        "errors": 0,
        "total_candidates": 0,
    }

    if not _is_grading_enabled():
        return summary

    cfg = _grading_cfg()
    persona_name = str(cfg.get("persona") or "grader")
    cap = max(0, int(cfg.get("max_per_session_per_run", 3)))

    try:
        candidates = list_ungraded_interventions(
            session_id,
            exclude_target_turn_id=outcome_turn_id,
            limit=20,  # always pull a small superset — cap applies after
        )
    except Exception as e:
        logger.warning("grading: list_ungraded_interventions failed: %s", e)
        return summary

    summary["total_candidates"] = len(candidates)
    if not candidates:
        return summary

    # Persona prompt is the same for all candidates this run — read once.
    loaded = _read_persona_file(persona_name)
    if loaded is None:
        try:
            _event_log.log_event(
                session_id,
                "inner_voice.grading_skipped",
                {"reason": "grader_persona_missing", "persona": persona_name},
                turn_id=outcome_turn_id,
            )
        except Exception:
            pass
        return summary
    meta, system_prompt = loaded
    persona_version = meta.get("version") if isinstance(meta, dict) else None

    try:
        _event_log.log_event(
            session_id,
            "inner_voice.grading_started",
            {
                "persona": persona_name,
                "persona_version": persona_version,
                "candidate_count": len(candidates),
                "cap": cap,
            },
            turn_id=outcome_turn_id,
        )
    except Exception:
        pass

    for intervention in candidates:
        if summary["graded"] + summary["errors"] >= cap:
            summary["skipped"] += 1
            continue

        intervention_id = intervention.get("id")
        if intervention_id is None:
            summary["errors"] += 1
            continue

        try:
            user_prompt = _build_grader_prompt(
                frozen_task_intent=frozen_task_intent,
                intervention=intervention,
                outcome_response_text=outcome_response_text,
                outcome_tool_calls=outcome_tool_calls,
            )
        except Exception as e:
            logger.warning("grading: prompt build failed (id=%s): %s", intervention_id, e)
            summary["errors"] += 1
            continue

        # Route through the existing critic.call_critic infrastructure —
        # we override the prefill via a one-shot call to call_critic_grader
        # (separate function so we don't pollute the disagrees-prefill code
        # path with a second prefill convention).
        started = time.monotonic()
        addressed: bool | None = None
        grader_summary = ""
        error: str | None = None
        raw_response = ""

        try:
            verdict = await _call_grader(
                persona_system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            raw_response = verdict["raw"]
            error = verdict.get("error")
            if error is None and verdict.get("parsed") is not None:
                parsed = verdict["parsed"]
                addressed = _coerce_addressed(parsed.get("addressed"))
                grader_summary = str(parsed.get("summary") or "").strip()[:500]
        except Exception as e:
            logger.warning("grading: call failed (id=%s): %s", intervention_id, e)
            error = f"unexpected {type(e).__name__}: {e}"

        wall_ms = int((time.monotonic() - started) * 1000)

        # Event log — full prompt + raw response. SQLite gets the verdict.
        try:
            _event_log.log_event(
                session_id,
                "inner_voice.grading_invoked",
                {
                    "intervention_id": intervention_id,
                    "intervention_kind": intervention.get("kind"),
                    "intervention_target_turn_id": intervention.get("target_turn_id"),
                    "outcome_turn_id": outcome_turn_id,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "wall_ms": wall_ms,
                },
                turn_id=outcome_turn_id,
            )
        except Exception:
            pass
        try:
            _event_log.log_event(
                session_id,
                "inner_voice.grading_response_raw",
                {
                    "intervention_id": intervention_id,
                    "raw": raw_response,
                    "error": error,
                    "wall_ms": wall_ms,
                },
                turn_id=outcome_turn_id,
            )
        except Exception:
            pass

        if error is not None:
            summary["errors"] += 1
            try:
                _event_log.log_event(
                    session_id,
                    "inner_voice.grading_failure",
                    {
                        "intervention_id": intervention_id,
                        "error": error,
                    },
                    turn_id=outcome_turn_id,
                )
            except Exception:
                pass
            # Leave the row ungraded — next pass retries.
            continue

        # Persist the verdict. `outcome_addressed=NULL` is a valid value
        # (ambiguous); we still set `outcome_turn_id` and `graded_at` so
        # the row leaves the ungraded queue.
        try:
            update_intervention_outcome(
                int(intervention_id),
                outcome_turn_id=outcome_turn_id,
                outcome_addressed=addressed,
                outcome_summary=grader_summary or "(no summary)",
            )
        except Exception as e:
            logger.warning(
                "grading: persist failed (id=%s): %s", intervention_id, e
            )
            summary["errors"] += 1
            continue

        try:
            _event_log.log_event(
                session_id,
                "inner_voice.grading_decision",
                {
                    "intervention_id": intervention_id,
                    "outcome_turn_id": outcome_turn_id,
                    "addressed": addressed,
                    "summary": grader_summary,
                    "intervention_kind": intervention.get("kind"),
                    "intervention_target_turn_id": intervention.get("target_turn_id"),
                },
                turn_id=outcome_turn_id,
            )
        except Exception:
            pass

        summary["graded"] += 1

    try:
        _event_log.log_event(
            session_id,
            "inner_voice.grading_completed",
            {
                "outcome_turn_id": outcome_turn_id,
                **summary,
            },
            turn_id=outcome_turn_id,
        )
    except Exception:
        pass

    return summary


# ---------------------------------------------------------------------------
# Grader-specific HTTP call
#
# Mirrors `critic._post_chat_completion` + parse pipeline, but with a
# different prefill (`{"addressed":` vs `{"disagrees":`). We don't extend
# critic.call_critic with a prefill arg because the persona contract is
# distinct enough that interleaving them invites bugs.
# ---------------------------------------------------------------------------


async def _call_grader(
    *,
    persona_system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    """Single the critic round-trip for the grader persona.

    Returns:
        {
          "parsed": dict | None,    -- parsed JSON if successful
          "raw": str,               -- raw response text
          "error": str | None,      -- non-None on HTTP / timeout / parse failure
        }
    """
    import asyncio
    import httpx

    # Reuse the critic config block — the critic endpoint, timeout, max_tokens
    # are the same. Only the prefill differs.
    crit_cfg = _critic._critic_cfg()  # type: ignore[attr-defined]
    base_url, model_name = _critic._critic_endpoint()  # type: ignore[attr-defined]
    if not base_url:
        return {"parsed": None, "raw": "", "error": "critic endpoint unconfigured"}

    messages = [
        {"role": "system", "content": persona_system_prompt},
        {"role": "user", "content": user_prompt},
        # Grader-specific prefill — not the disagrees-shaped one.
        {"role": "assistant", "content": _GRADER_PREFILL},
    ]

    max_tokens = int(crit_cfg["max_tokens"])
    timeout_s = float(crit_cfg["timeout_seconds"])
    max_attempts = 1 + max(0, int(crit_cfg.get("json_retry_on_parse_failure", 1)))

    last_raw = ""
    error: str | None = None

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    url = f"{base_url}/v1/chat/completions"

    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                resp = await asyncio.wait_for(
                    client.post(
                        url,
                        headers={
                            "Authorization": "Bearer no-key-required",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    ),
                    timeout=timeout_s + 1.0,
                )
                resp.raise_for_status()
                data = resp.json()
        except asyncio.TimeoutError:
            error = f"timeout after {timeout_s}s"
            break
        except httpx.HTTPError as e:
            error = f"http {type(e).__name__}: {e}"
            break
        except Exception as e:
            error = f"unexpected {type(e).__name__}: {e}"
            break

        raw = (
            (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            or ""
        )
        last_raw = raw

        parsed = _parse_grader_json(raw)
        if parsed is not None:
            return {"parsed": parsed, "raw": raw, "error": None}

        # Parse failed — retry once if attempts remain. Add a short error
        # message to the messages list so the model knows to try again.
        if attempt < max_attempts:
            messages.append({
                "role": "user",
                "content": (
                    "Your previous response did not parse as a valid JSON object "
                    "containing an 'addressed' key. Please return exactly one "
                    'JSON object on a single line: {"addressed": <bool|null>, '
                    '"summary": "..."}'
                ),
            })
            payload["messages"] = messages

    if error is not None:
        return {"parsed": None, "raw": last_raw, "error": error}
    return {
        "parsed": None,
        "raw": last_raw,
        "error": "json_parse_failed_after_retry",
    }


__all__ = [
    "grade_outcome_turn",
    "_is_grading_enabled",
    "_grading_cfg",
    "_build_grader_prompt",
    "_parse_grader_json",
    "_coerce_addressed",
]
