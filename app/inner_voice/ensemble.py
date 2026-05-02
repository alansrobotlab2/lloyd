"""Inner Voice (#345) Stage 2 — single-persona post-loop critic runner.

Stage 2 ships ONE persona (`completion_checker`). Stage 3 grows this into a
true ensemble (3 personas concurrent) and adds aggregation / intervention
dispatch. The current shape — single persona, log-only — is the minimum
viable surface for validating the Brain 2 call path against ambient turns
without changing the agent's behavior.

Responsibilities:

  1. **Build asymmetric context** for Brain 2: frozen task intent + last-N
     transcript window, with thinking blocks stripped. This is the recipe
     from the Disagreement Engineering section (techniques 1+2). Lensed
     past-failure context (technique 3) is reserved for the
     `confidence_calibrator` / `hallucination_flag` personas — Stage 4+.

  2. **Load persona prompts** from the vault. Files at
     `~/obsidian/lloyd/inner_voice/personas/<name>.md` with YAML
     frontmatter for version + ensemble assignments. Cache per-process
     with mtime invalidation so iterative tuning works without restarts.

  3. **Invoke the critic** via `critic.call_critic` and persist the result:
     full prompt + raw response → event log (blob-store deduped); parsed
     verdict + forensic offsets → `inner_voice_critiques` SQLite row;
     summary fields → SSE `inner_voice_critique` to the frontend.

  4. **No interventions yet.** Stage 2's `action_taken` defaults to
     `log_only` regardless of severity. Stage 3 wires the actual nudge /
     interrupt / continue dispatch.

Contract on failure: any unrecoverable error (config missing, persona file
gone, critic call exhausted) writes a forensic-only event and returns
without raising. The chat path must never see an exception from this
module.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Awaitable, Callable

import yaml

from app.config import CONFIG
from app import event_log as _event_log
from app.inner_voice import critic as _critic
from app.inner_voice.critic import Critique
from app.paths import SESSIONS_DIR
from usage_store import record_inner_voice_critique

logger = logging.getLogger("lloyd-server")


# ---------------------------------------------------------------------------
# Persona prompt loading
# ---------------------------------------------------------------------------

_PERSONAS_DIR = Path("/home/alansrobotlab/obsidian/lloyd/inner_voice/personas")
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# (mtime_ns, parsed_meta, body) keyed by persona_name.
_persona_cache: dict[str, tuple[int, dict[str, Any], str]] = {}


def _read_persona_file(persona_name: str) -> tuple[dict[str, Any], str] | None:
    """Read `<personas>/<persona_name>.md` and return (frontmatter, body).

    Returns None if the file is missing or malformed. Caches by mtime so
    edits during a session take effect on the next ambient turn without a
    restart — the spec says budget 5–10 iterations on the prompt.
    """
    path = _PERSONAS_DIR / f"{persona_name}.md"
    if not path.exists():
        logger.warning("persona file missing: %s", path)
        return None
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return None

    cached = _persona_cache.get(persona_name)
    if cached and cached[0] == mtime:
        return cached[1], cached[2]

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("persona read failed (%s): %s", persona_name, e)
        return None

    meta: dict[str, Any] = {}
    body = text
    m = _FRONTMATTER_RE.match(text)
    if m:
        try:
            meta = yaml.safe_load(m.group(1)) or {}
            if not isinstance(meta, dict):
                meta = {}
        except yaml.YAMLError as e:
            logger.warning("persona frontmatter YAML error (%s): %s", persona_name, e)
            meta = {}
        body = text[m.end():]

    _persona_cache[persona_name] = (mtime, meta, body)
    return meta, body


# ---------------------------------------------------------------------------
# Transcript window assembly (asymmetric — no thinking)
# ---------------------------------------------------------------------------


def _extract_text_only(content: Any) -> str:
    """Return the human-readable text from a message's `content` field.

    Skips `thinking` blocks unconditionally — Brain 2 must NOT see Brain 1's
    chain of thought (technique 2 from the Disagreement Engineering recipe).
    Tool-use blocks are summarized as `[tool: name]` (no args) so Brain 2
    knows what fired without being anchored to Brain 1's tool-call shape.
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content is not None else ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype == "text":
            t = (block.get("text") or "").strip()
            if t:
                parts.append(t)
        elif btype == "tool_use":
            name = block.get("name", "?")
            parts.append(f"[tool: {name}]")
        elif btype == "tool_result":
            # Show that a tool returned, but not its content (to keep
            # context tight; Brain 2 can ask via the event log if needed).
            parts.append("[tool_result]")
        # `thinking` and `image` blocks: skip.
    return "\n".join(p for p in parts if p)


def _summarize_tool_calls(tool_calls: list[dict] | None) -> list[str]:
    """Return a list of short tool call descriptors for the current turn.

    Each entry is `<name>(<key=val,...>)` truncated. We include arg keys
    (not values) so the persona prompt's "tool history shows X" rule has
    something to match without the raw payload pollution.
    """
    out: list[str] = []
    for tc in tool_calls or []:
        name = tc.get("name") or tc.get("function", {}).get("name", "?")
        # Normalize MCP-prefixed names for readability.
        short = name.rsplit("__", 1)[-1] if "__" in name else name
        args = tc.get("input") or tc.get("arguments") or tc.get("function", {}).get("arguments", {})
        keys = ""
        if isinstance(args, dict):
            keys = ",".join(sorted(args.keys())[:5])
        elif isinstance(args, str):
            try:
                parsed = json.loads(args)
                if isinstance(parsed, dict):
                    keys = ",".join(sorted(parsed.keys())[:5])
            except json.JSONDecodeError:
                pass
        out.append(f"{short}({keys})" if keys else short)
    return out


def _load_recent_transcript(
    session_id: str,
    *,
    exclude_turn_id: str | None = None,
    window_turns: int = 5,
) -> list[dict[str, Any]]:
    """Return the last `window_turns` user/assistant pairs from the session.

    Excludes `exclude_turn_id` (the turn currently under critique). Drops
    subliminal / system / tool / silent-ambient entries. Each returned dict
    is `{"role", "text", "turn_id_hint"}` — minimal fields for prompt
    assembly.
    """
    meta_path = SESSIONS_DIR / f"{session_id}.json"
    if not meta_path.exists():
        return []
    try:
        data = json.loads(meta_path.read_text())
    except Exception as e:
        logger.warning("transcript load failed (%s): %s", session_id, e)
        return []

    messages = data.get("messages") or []
    if not isinstance(messages, list):
        return []

    convo: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "")
        if role not in ("user", "assistant"):
            continue
        if msg.get("silent"):
            continue
        # Heuristic match for "current turn": the persisted assistant entry
        # for the active turn carries source != 'user' AND was just appended.
        # We cannot match by turn_id directly because the persisted format
        # uses message ids, not turn ids. Best we can do is drop the LAST
        # assistant message — the consumer always ensures the current turn's
        # assistant text is the last persisted entry by the time this runs.
        text = _extract_text_only(msg.get("content"))
        if not text:
            continue
        convo.append({"role": role, "text": text, "turn_id_hint": msg.get("id", "")})

    # Drop the trailing assistant message — that's the turn under critique.
    if exclude_turn_id and convo and convo[-1]["role"] == "assistant":
        convo = convo[:-1]
    elif convo and convo[-1]["role"] == "assistant":
        # Defensive: if no exclude_turn_id given, still drop the last
        # assistant — it's almost certainly the in-flight turn.
        convo = convo[:-1]

    return convo[-window_turns * 2 :]  # window_turns user+assistant pairs ≈ 2x entries


# ---------------------------------------------------------------------------
# User-prompt assembly for Brain 2
# ---------------------------------------------------------------------------


_TRANSCRIPT_TEXT_CAP = 800   # chars per transcript message
_RESPONSE_TEXT_CAP = 4000    # chars of Brain 1's final response
_TASK_INTENT_CAP = 1500      # chars of frozen task intent


def _truncate(s: str, cap: int) -> str:
    if not s:
        return s
    if len(s) <= cap:
        return s
    return s[: cap - 3] + "..."


def _build_user_prompt(
    *,
    frozen_task_intent: str,
    response_text: str,
    tool_calls: list[dict],
    transcript: list[dict[str, Any]],
) -> str:
    """Assemble the Brain 2 user-message body.

    Layout:
        <task>
        ...
        </task>

        <recent_transcript>
        [user] ...
        [assistant] ...
        </recent_transcript>

        <final_response>
        text: ...
        tool_calls: [name1, name2, ...]
        </final_response>

    Fixed layout — the persona prompt's examples reference these tag names.
    Changing them invalidates the calibration on the prompt; do not.
    """
    transcript_lines: list[str] = []
    for entry in transcript:
        prefix = "user" if entry["role"] == "user" else "assistant"
        transcript_lines.append(f"[{prefix}] {_truncate(entry['text'], _TRANSCRIPT_TEXT_CAP)}")
    transcript_block = "\n".join(transcript_lines) if transcript_lines else "(no prior turns)"

    tool_summaries = _summarize_tool_calls(tool_calls)
    tool_str = ", ".join(tool_summaries) if tool_summaries else "(none)"

    return (
        "<task>\n"
        f"{_truncate(frozen_task_intent or '(no task text)', _TASK_INTENT_CAP)}\n"
        "</task>\n\n"
        "<recent_transcript>\n"
        f"{transcript_block}\n"
        "</recent_transcript>\n\n"
        "<final_response>\n"
        f"text: {_truncate(response_text or '(empty)', _RESPONSE_TEXT_CAP)}\n"
        f"tool_calls: [{tool_str}]\n"
        "</final_response>"
    )


# ---------------------------------------------------------------------------
# Throughput guard
# ---------------------------------------------------------------------------


_session_critique_counts: dict[str, int] = {}


def _max_critiques_per_session() -> int:
    iv = CONFIG.get("inner_voice") or {}
    tp = iv.get("throughput") or {}
    return int(tp.get("max_critiques_per_session", 50))


def _under_session_cap(session_id: str) -> bool:
    cap = _max_critiques_per_session()
    if cap <= 0:
        return True
    return _session_critique_counts.get(session_id, 0) < cap


def _bump_session_count(session_id: str) -> None:
    _session_critique_counts[session_id] = _session_critique_counts.get(session_id, 0) + 1


# ---------------------------------------------------------------------------
# Ensemble selection
# ---------------------------------------------------------------------------


def _select_ensemble_for_turn(turn_source: str, frozen_task_intent: str) -> tuple[str, list[str]]:
    """Stage 2: hardcoded to single-persona [completion_checker].

    Stage 3 implements work-type-keyed selection (heuristic on autonomy
    metadata + last tool call); Stage 4 layers safety routing on top.
    Returning the ensemble *name* (`autonomy_default`) alongside the
    persona list lets the SSE/event pipeline already render the eventual
    full label even though only one persona fires.
    """
    iv = CONFIG.get("inner_voice") or {}
    ens = iv.get("ensemble") or {}
    name = ens.get("default") or "autonomy_default"
    # Stage 2 ships only completion_checker. Other personas in the configured
    # set are silently dropped here — Stage 3 lights them up.
    return name, ["completion_checker"]


# ---------------------------------------------------------------------------
# Public entry point — wired from messages.py
# ---------------------------------------------------------------------------


# Type alias for the SSE emit hook. Producers in messages.py pass a callback
# that pushes onto the active turn's event queue.
EmitFn = Callable[[str, dict[str, Any]], Awaitable[None]]


async def run_post_loop_critique(
    *,
    session_id: str,
    turn_id: str,
    turn_source: str,
    frozen_task_intent: str,
    response_text: str,
    tool_calls: list[dict],
    emit_sse: EmitFn | None = None,
) -> list[Critique]:
    """Run Brain 2 on a finished ambient turn. Returns the critiques.

    Spawned via `asyncio.ensure_future` from the ResultMessage branch in
    `_run_turn`, alongside the Stage 1 completion check. Best-effort.
    Catches every exception on its own — never raises into the consumer.

    Stage 2 fires one persona (`completion_checker`). Stage 3 will fan out
    to 3 personas concurrently and add aggregation logic; the current
    return-list shape leaves room for that without changing call sites.
    """
    out: list[Critique] = []
    if not _is_enabled():
        return out
    if not _under_session_cap(session_id):
        try:
            _event_log.log_event(
                session_id,
                "inner_voice.persona_skipped",
                {"reason": "session_critique_cap_reached",
                 "cap": _max_critiques_per_session()},
                turn_id=turn_id,
            )
        except Exception:
            pass
        return out

    ensemble_name, personas = _select_ensemble_for_turn(turn_source, frozen_task_intent)

    try:
        if emit_sse is not None:
            await emit_sse("inner_voice_observation_state", {"state": "critiquing"})
            await emit_sse(
                "inner_voice_ensemble_change",
                {"ensemble_name": ensemble_name, "personas": personas},
            )
    except Exception as e:
        logger.warning("inner_voice ensemble emit failed: %s", e)

    try:
        _event_log.log_event(
            session_id,
            "inner_voice.ensemble_selected",
            {
                "ensemble_name": ensemble_name,
                "personas": personas,
                "selection_rationale": "stage2_single_persona_default",
                "turn_source": turn_source,
            },
            turn_id=turn_id,
        )
    except Exception:
        pass

    transcript_window_turns = int(_critic._critic_cfg().get("transcript_window_turns", 5))
    transcript = _load_recent_transcript(
        session_id,
        exclude_turn_id=turn_id,
        window_turns=transcript_window_turns,
    )
    user_prompt = _build_user_prompt(
        frozen_task_intent=frozen_task_intent,
        response_text=response_text,
        tool_calls=tool_calls,
        transcript=transcript,
    )

    # Stage 2 runs personas sequentially because there's only one. Stage 3
    # switches to `asyncio.gather`. The shape stays compatible.
    for persona_name in personas:
        try:
            critique = await _run_one_persona(
                session_id=session_id,
                turn_id=turn_id,
                persona_name=persona_name,
                user_prompt=user_prompt,
                response_excerpt=response_text or "",
            )
            if critique is not None:
                out.append(critique)
                _bump_session_count(session_id)
                if emit_sse is not None:
                    try:
                        await _emit_critique_sse(emit_sse, critique, turn_id=turn_id)
                    except Exception as e:
                        logger.warning("inner_voice critique emit failed: %s", e)
        except Exception as e:
            logger.warning("inner_voice persona %s failed: %s", persona_name, e)

    # Stage 2 single-persona aggregation — pass-through. Logged anyway so
    # Stage 3's multi-persona aggregation can be A/B'd against the trivial
    # baseline.
    try:
        agg = _aggregate(out)
        _event_log.log_event(
            session_id,
            "inner_voice.aggregation_decision",
            {
                "personas_invoked": len(out),
                "personas_disagreed": agg["disagree_count"],
                "severity_max": agg["severity_max"],
                "action_chosen": agg["action_chosen"],
                "rationale": "stage2_single_persona_passthrough",
            },
            turn_id=turn_id,
        )
    except Exception:
        pass

    try:
        if emit_sse is not None:
            await emit_sse("inner_voice_observation_state", {"state": "observing"})
    except Exception:
        pass

    return out


# ---------------------------------------------------------------------------
# Per-persona invocation — handles all logging/persistence for one critique
# ---------------------------------------------------------------------------


async def _run_one_persona(
    *,
    session_id: str,
    turn_id: str,
    persona_name: str,
    user_prompt: str,
    response_excerpt: str,
) -> Critique | None:
    """Invoke one persona end-to-end: load prompt, call critic, persist."""
    loaded = _read_persona_file(persona_name)
    if loaded is None:
        try:
            _event_log.log_event(
                session_id,
                "inner_voice.persona_skipped",
                {"reason": "persona_file_missing", "persona": persona_name},
                turn_id=turn_id,
            )
        except Exception:
            pass
        return None

    meta, body = loaded
    persona_version = meta.get("version") if isinstance(meta, dict) else None

    # `persona_invoked` is logged BEFORE the call. Captures the full prompt
    # (system + user). Returns the line offset so we can pin the SQLite row
    # to it. The full prompt is automatically blob-stored if it crosses the
    # threshold.
    invoked_offset: int | None = None
    try:
        invoked_offset = _event_log.log_event(
            session_id,
            "inner_voice.persona_invoked",
            {
                "persona": persona_name,
                "persona_version": persona_version,
                "system_prompt": body,
                "user_prompt": user_prompt,
            },
            turn_id=turn_id,
        )
    except Exception as e:
        logger.warning("persona_invoked log failed: %s", e)

    started = time.monotonic()
    critique = await _critic.call_critic(
        persona=persona_name,
        persona_version=persona_version,
        persona_system_prompt=body,
        user_prompt=user_prompt,
        response_excerpt=response_excerpt,
    )
    wall_ms = int((time.monotonic() - started) * 1000)

    # `persona_response_raw` — the model's literal output text (potentially
    # blob-stored). Also used for the `raw_response_offset` link.
    raw_offset: int | None = None
    try:
        raw_offset = _event_log.log_event(
            session_id,
            "inner_voice.persona_response_raw",
            {
                "persona": persona_name,
                "raw": critique.raw_response,
                "input_tokens": critique.input_tokens,
                "output_tokens": critique.output_tokens,
                "latency_ms": critique.latency_ms,
                "wall_ms": wall_ms,
                "error": critique.error,
                "parse_attempts": critique.parse_attempts,
            },
            turn_id=turn_id,
        )
    except Exception as e:
        logger.warning("persona_response_raw log failed: %s", e)

    # `persona_response_parsed` if the call produced a parseable verdict;
    # `persona_parse_failure` otherwise. Both can fire — parse_failure first,
    # parsed second (after retry succeeded), or only parse_failure (after
    # retries exhausted).
    if critique.error is None and critique.parse_attempts > 0:
        try:
            _event_log.log_event(
                session_id,
                "inner_voice.persona_response_parsed",
                {
                    "persona": persona_name,
                    "disagrees": critique.disagrees,
                    "severity": critique.severity,
                    "reason": critique.reason,
                    "suggested_action": critique.suggested_action,
                },
                turn_id=turn_id,
            )
        except Exception as e:
            logger.warning("persona_response_parsed log failed: %s", e)
    elif critique.error is not None:
        try:
            _event_log.log_event(
                session_id,
                "inner_voice.persona_parse_failure",
                {
                    "persona": persona_name,
                    "error": critique.error,
                    "parse_attempts": critique.parse_attempts,
                    "raw_excerpt": (critique.raw_response or "")[:500],
                },
                turn_id=turn_id,
            )
        except Exception as e:
            logger.warning("persona_parse_failure log failed: %s", e)

    # Persist the SQLite row. event_log_offset / raw_response_offset link
    # forward to the event log lines so click-to-detail in the UI can pull
    # the full prompt + raw via /api/inner_voice/event_log?offset=N&limit=1.
    try:
        record_inner_voice_critique(
            session_id=session_id,
            turn_id=turn_id,
            persona=persona_name,
            persona_version=persona_version,
            model=critique.model,
            input_tokens=critique.input_tokens,
            output_tokens=critique.output_tokens,
            latency_ms=critique.latency_ms,
            disagrees=critique.disagrees,
            severity=critique.severity,
            reason=critique.reason,
            suggested_action=critique.suggested_action,
            action_taken=critique.action_taken,
            anchor_response_excerpt=critique.anchor_response_excerpt,
            event_log_offset=invoked_offset,
            raw_response_offset=raw_offset,
            prompt_hash=critique.prompt_hash,
            parse_attempts=critique.parse_attempts,
        )
    except Exception as e:
        logger.warning("inner_voice critique SQLite persist failed: %s", e)

    return critique


# ---------------------------------------------------------------------------
# Aggregation (Stage 2 trivial pass-through)
# ---------------------------------------------------------------------------


def _aggregate(critiques: list[Critique]) -> dict[str, Any]:
    """Return a summary dict over a list of Critiques.

    Stage 2: single-persona, pass-through. Stage 3 will replace this with
    severity-weighted voting. Shape is intentionally Stage-3-compatible so
    the call site doesn't move.
    """
    if not critiques:
        return {
            "disagree_count": 0,
            "severity_max": 0.0,
            "action_chosen": "no_op",
        }
    disagree_count = sum(1 for c in critiques if c.disagrees)
    severity_max = max((c.severity for c in critiques), default=0.0)
    # Stage 2 logs the suggested action but never dispatches; Stage 3
    # promotes this into a real intervention call.
    action_chosen = "log_only"
    return {
        "disagree_count": disagree_count,
        "severity_max": severity_max,
        "action_chosen": action_chosen,
    }


# ---------------------------------------------------------------------------
# SSE emit shim
# ---------------------------------------------------------------------------


async def _emit_critique_sse(
    emit_sse: EmitFn,
    critique: Critique,
    *,
    turn_id: str,
) -> None:
    """Push the user-visible critique summary onto the active turn's events.

    Wire shape (matches `inner_voice_critique` on the frontend):
        {
          persona, severity, reason,
          disagrees, suggested_action, action_taken,
          anchor_turn_id, prompt_hash,
        }
    """
    await emit_sse(
        "inner_voice_critique",
        {
            "persona": critique.persona,
            "persona_version": critique.persona_version,
            "severity": critique.severity,
            "disagrees": critique.disagrees,
            "reason": critique.reason,
            "suggested_action": critique.suggested_action,
            "action_taken": critique.action_taken,
            "anchor_turn_id": turn_id,
            "prompt_hash": critique.prompt_hash,
            "error": critique.error,
        },
    )


# ---------------------------------------------------------------------------
# Config gate
# ---------------------------------------------------------------------------


def _is_enabled() -> bool:
    """Read `inner_voice.critic.enabled` (defaults true) — but only if the
    parent `inner_voice` block is present. Stage 2 is opt-in via the session
    flag; this gate is the kill-switch for emergency disable without a
    redeploy.
    """
    iv = CONFIG.get("inner_voice")
    if not iv:
        return False
    crit = iv.get("critic") or {}
    if "enabled" in crit:
        return bool(crit["enabled"])
    # Default ON when `inner_voice.critic` block exists. Stage 2 acceptance
    # gate requires Brain 2 firing on every Inner Voice ambient turn.
    return True


__all__ = [
    "run_post_loop_critique",
    "_build_user_prompt",
    "_load_recent_transcript",
    "_select_ensemble_for_turn",
    "_aggregate",
]
