"""Inner Voice observer — a thin second-agent harness.

One observer per primary turn. The observer:

1. At turn start, runs ONE goal-extraction LLM call to produce a goal card
   (success criteria, out-of-scope, completion signals).
2. Subscribes to the primary's NormalizedEvent stream via an OnEvent hook.
3. For each significant event, runs a cheap pre-filter; if interesting,
   calls a focused LLM with the goal card + prior decisions threaded in.
4. Has five levers: inject, cancel, ambient, clarify, deny_tool.

All judgment lives in `observer_prompt.SYSTEM_PROMPT`. The Python here is
plumbing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx

from app import event_log as _event_log
from app.config import CONFIG, _get_model_cfg
from app.inner_voice import observer_prompt as _prompt
from usage_store import record_inner_voice_observation

logger = logging.getLogger("lloyd-iv-observer")


# ---------------------------------------------------------------------------
# Decision struct
# ---------------------------------------------------------------------------


@dataclass
class ObserverDecision:
    """Parsed observer verdict."""

    action: str = "noop"
    reason: str = ""
    content: str = ""
    # Forensic / persistence fields
    raw_response: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    error: str | None = None


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _observer_cfg() -> dict[str, Any]:
    iv = CONFIG.get("inner_voice") or {}
    obs = dict(iv.get("observer") or {})
    obs.setdefault("max_tokens", _prompt.DEFAULT_MAX_TOKENS)
    obs.setdefault("timeout_seconds", _prompt.DEFAULT_TIMEOUT_SECONDS)
    obs.setdefault("pretool_timeout_seconds", _prompt.DEFAULT_PRETOOL_TIMEOUT_SECONDS)
    obs.setdefault("intervention_budget", _prompt.DEFAULT_INTERVENTION_BUDGET)
    obs.setdefault("primary_text_window_chars", 4000)
    obs.setdefault("goal_extraction_timeout_seconds", _prompt.DEFAULT_GOAL_EXTRACTION_TIMEOUT_SECONDS)
    obs.setdefault("goal_extraction_max_tokens", _prompt.DEFAULT_GOAL_EXTRACTION_MAX_TOKENS)
    obs.setdefault("goal_extraction_enabled", True)
    obs.setdefault("fast_path_enabled", True)
    return obs


def _resolve_endpoint(model_alias: str | None = None) -> tuple[str, str]:
    """Resolve (base_url, model_name) for the observer's vLLM endpoint."""
    iv = CONFIG.get("inner_voice") or {}
    name = (
        model_alias
        or iv.get("model")
        or CONFIG.get("model", {}).get("default", "")
    )
    cfg = _get_model_cfg(name) or {}
    base = cfg.get("base_url") or cfg.get("env", {}).get("ANTHROPIC_BASE_URL", "")
    return (base.rstrip("/"), name)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _extract_first_json_object(text: str) -> str | None:
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
                return text[start : i + 1]
            if depth < 0:
                return None
    return None


def _parse_observer_json(raw: str) -> dict[str, Any] | None:
    """Parse the observer's JSON response, tolerant of prefill conventions.

    Handles five shapes:
      1. Complete JSON object (model didn't honor prefill).
      2. Prefill continuation (model returned `"value","field":...}`).
      3. Prefill continuation but model dropped the opening quote (returned
         `value","field":...}` — common with local models). We synthesize
         the missing `"`.
      4. JSON object embedded in surrounding prose.
      5. Bare action word (e.g. `noop` or `noop"}`).
    """
    if not raw:
        return None
    candidates = [raw, _prompt.JSON_PREFILL + raw]
    raw_stripped = raw.lstrip()
    if raw_stripped and raw_stripped[0].isalpha():
        candidates.append(_prompt.JSON_PREFILL + '"' + raw)
    for c in candidates:
        try:
            v = json.loads(c.strip())
            if isinstance(v, dict) and "action" in v:
                return v
        except json.JSONDecodeError:
            pass
    blob = _extract_first_json_object(raw)
    if blob:
        try:
            v = json.loads(blob)
            if isinstance(v, dict) and "action" in v:
                return v
        except json.JSONDecodeError:
            pass
    blob2 = _extract_first_json_object(_prompt.JSON_PREFILL + raw)
    if blob2:
        try:
            v = json.loads(blob2)
            if isinstance(v, dict) and "action" in v:
                return v
        except json.JSONDecodeError:
            pass
    bare = raw.strip().strip('",}{ ').lower()
    if bare in ("noop", "allow", "inject", "cancel", "ambient", "clarify", "deny_tool", "deny"):
        return {"action": bare, "reason": "bare-word fallback"}
    return None


def _normalize_action(raw_action: Any, *, allow_deny_tool: bool) -> str:
    """Normalize observer action string. Falls back safely on garbage."""
    if not isinstance(raw_action, str):
        return "noop"
    a = raw_action.strip().lower()
    valid_post = {"noop", "inject", "cancel", "ambient", "clarify"}
    valid_pre = {"allow", "deny_tool"}
    if allow_deny_tool:
        if a in valid_pre:
            return a
        if a == "deny":
            return "deny_tool"
        # Out of context for pretool — coerce to allow (fail-open).
        return "allow"
    if a in valid_post:
        return a
    if a in valid_pre:
        return "noop"
    return "noop"


# ---------------------------------------------------------------------------
# Goal extraction (one call per turn)
# ---------------------------------------------------------------------------


def _parse_goal_card(raw: str) -> dict[str, Any] | None:
    """Parse the goal-extraction JSON response."""
    if not raw:
        return None
    candidates = [raw, _prompt.GOAL_EXTRACTION_PREFILL + raw]
    raw_stripped = raw.lstrip()
    if raw_stripped and raw_stripped[0] == "[":
        candidates.append(_prompt.GOAL_EXTRACTION_PREFILL + raw)
    for c in candidates:
        try:
            v = json.loads(c.strip())
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            pass
    blob = _extract_first_json_object(raw)
    if blob:
        try:
            v = json.loads(blob)
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            pass
    return None


async def _post_chat_completion(
    *,
    base_url: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    timeout_seconds: float,
    prefill: str,
) -> dict[str, Any]:
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": prefill},
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        resp = await client.post(
            url,
            headers={
                "Authorization": "Bearer no-key-required",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()


async def extract_goal_card(
    user_request: str,
    *,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Run one LLM call at turn start to extract the goal card.

    Returns None on any failure. Best-effort — observer falls back to
    "no goal card" mode (lighter-touch oversight) on extraction failure.
    """
    cfg = cfg or _observer_cfg()
    if not cfg.get("goal_extraction_enabled", True):
        return None
    if not user_request or not user_request.strip():
        return None
    base_url, model_name = _resolve_endpoint()
    if not base_url:
        return None
    timeout = float(cfg.get("goal_extraction_timeout_seconds", _prompt.DEFAULT_GOAL_EXTRACTION_TIMEOUT_SECONDS))
    max_tokens = int(cfg.get("goal_extraction_max_tokens", _prompt.DEFAULT_GOAL_EXTRACTION_MAX_TOKENS))
    user_prompt = _prompt.build_goal_extraction_user_prompt(user_request)
    raw = ""
    started = time.perf_counter()
    try:
        body = await _post_chat_completion(
            base_url=base_url,
            model_name=model_name,
            system_prompt=_prompt.GOAL_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            timeout_seconds=timeout,
            prefill=_prompt.GOAL_EXTRACTION_PREFILL,
        )
        choices = body.get("choices") or []
        if choices:
            raw = (choices[0].get("message", {}).get("content") or "").strip()
    except Exception as e:  # noqa: BLE001 — best-effort
        logger.warning("[iv.observer] goal extraction failed: %s", e)
        return None
    parsed = _parse_goal_card(raw)
    latency_ms = int((time.perf_counter() - started) * 1000)
    if parsed is None:
        logger.warning(
            "[iv.observer] goal extraction parse_failed in %dms raw=%r",
            latency_ms, raw[:300],
        )
        return None
    # Normalize lists
    out = {
        "success_criteria": _coerce_str_list(parsed.get("success_criteria")),
        "out_of_scope": _coerce_str_list(parsed.get("out_of_scope")),
        "completion_signals": _coerce_str_list(parsed.get("completion_signals")),
    }
    logger.info(
        "[iv.observer] goal_card extracted in %dms: %d criteria, %d oos, %d signals",
        latency_ms,
        len(out["success_criteria"]),
        len(out["out_of_scope"]),
        len(out["completion_signals"]),
    )
    return out


def _coerce_str_list(v: Any) -> list[str]:
    if not isinstance(v, list):
        return []
    out: list[str] = []
    for item in v:
        if isinstance(item, str) and item.strip():
            out.append(item.strip()[:400])
    return out[:8]  # cap each list at 8 items


# ---------------------------------------------------------------------------
# Tiered triggering — cheap pre-filters
# ---------------------------------------------------------------------------


# First-word allowlist for Bash commands considered safely read-only.
_SAFE_BASH_FIRST_WORDS = frozenset({
    "ls", "cat", "head", "tail", "wc", "grep", "rg", "fgrep", "egrep",
    "find", "fd", "du", "df", "echo", "date", "pwd", "which", "whereis",
    "hostname", "uname", "whoami", "id", "stat", "file", "ps", "uptime",
    "free", "type", "tree", "less", "more", "history", "diff", "comm",
    "sort", "uniq", "tr", "cut", "column", "test", "true", "false",
    "basename", "dirname", "realpath", "readlink", "tac", "nl", "expand",
    "unexpand", "fold", "fmt", "od", "hexdump", "xxd", "strings",
    "printf", "yes", "seq", "env", "printenv", "tty", "groups",
    "ip", "netstat", "ss", "lsof", "curl", "wget",
})

# Patterns that immediately disqualify a Bash command from fast-allow,
# regardless of the first word. Conservative.
_BASH_RISK_PATTERNS = re.compile(
    r"(?:\brm\b|\bmv\b|\bcp\b|\bchmod\b|\bchown\b|\bsudo\b|\bdd\b|\bmkfs\b|"
    r"\bmount\b|\bumount\b|\bkill\b|\bpkill\b|\bkillall\b|\bsystemctl\b|"
    r"\bservice\b|\bdocker\b|\bgit\s+push\b|\bgit\s+reset\b|\bgit\s+rebase\b|"
    r"\bgit\s+checkout\b|--force|--hard|>\s*[^/]|>>\s*[^/]|`|"
    # Pipe to a shell interpreter — `cat foo | bash`, `curl x | sh`, etc.
    # The first-word allowlist would otherwise let this through.
    r"\|\s*(?:bash|sh|zsh|fish|ksh|csh|python3?|node|ruby|perl)\b)",
    re.IGNORECASE,
)


def _bash_command_is_safely_readonly(cmd: str) -> bool:
    """Return True iff a Bash command is plausibly read-only and safe."""
    if not cmd or not cmd.strip():
        return True  # empty / comment — harmless
    if _BASH_RISK_PATTERNS.search(cmd):
        return False
    # Pull the first word, stripping common prefixes like `2>/dev/null`.
    tokens = cmd.split()
    if not tokens:
        return True
    first = tokens[0].lstrip("(").lstrip("\\")
    if first.startswith("#"):
        return True
    return first in _SAFE_BASH_FIRST_WORDS


# Fast-allow for tools whose names imply read/list semantics.
_SAFE_TOOL_NAME_KEYWORDS = (
    "read", "list", "get", "search", "find", "query", "fetch", "view",
    "show", "lookup", "describe", "status", "check", "stat", "summary",
    "count", "watch",
)

# Fast-allow for these explicit non-Bash tool names (built-ins).
_FAST_ALLOW_TOOL_NAMES = frozenset({"Read", "Glob", "Grep"})


def _fast_path_pretool(tool_name: str, tool_args: dict) -> ObserverDecision | None:
    """Cheap deterministic check for benign tool calls. Returns an
    'allow' ObserverDecision when safe, or None to escalate to LLM.
    """
    if not tool_name:
        return None
    if tool_name in _FAST_ALLOW_TOOL_NAMES:
        return ObserverDecision(action="allow", reason="fast-path: read-only tool")
    if tool_name == "Bash":
        cmd = (tool_args.get("command") or "") if isinstance(tool_args, dict) else ""
        if _bash_command_is_safely_readonly(cmd):
            return ObserverDecision(action="allow", reason="fast-path: read-only Bash")
        return None
    # MCP tools: fast-allow when name contains a clearly read/list verb,
    # AND args are small (large args are usually writes).
    if tool_name.startswith("mcp__"):
        name_lower = tool_name.lower()
        if any(kw in name_lower for kw in _SAFE_TOOL_NAME_KEYWORDS):
            args_str = json.dumps(tool_args, default=str) if isinstance(tool_args, dict) else str(tool_args)
            if len(args_str) < 1000:
                return ObserverDecision(action="allow", reason="fast-path: read-shaped MCP tool")
    return None


def _fast_path_tool_result(tool_name: str, content: str, is_error: bool) -> ObserverDecision | None:
    """Cheap check for benign tool results. Returns 'noop' when safe."""
    if not is_error and len(content) < 2000:
        return ObserverDecision(action="noop", reason="fast-path: small benign result")
    # Common "primary will retry" patterns — observer doesn't need to
    # inspect a parse-error message; the primary handles it.
    if is_error and "Tool call arguments could not be parsed as JSON" in content:
        return ObserverDecision(action="noop", reason="fast-path: parse-error, primary will retry")
    return None


# ---------------------------------------------------------------------------
# Observer LLM call wrapper
# ---------------------------------------------------------------------------


async def _call_observer(
    *,
    user_prompt: str,
    allow_deny_tool: bool,
    cfg: dict[str, Any] | None = None,
    timeout_override: float | None = None,
) -> ObserverDecision:
    """One observer LLM call. Returns a parsed ObserverDecision.

    Errors and parse failures fold into a noop / allow decision so the
    primary stream is never blocked by observer faults.
    """
    cfg = cfg or _observer_cfg()
    base_url, model_name = _resolve_endpoint()
    if not base_url:
        return ObserverDecision(
            action=("allow" if allow_deny_tool else "noop"),
            reason="observer endpoint unresolved",
            error="no base_url",
        )
    timeout = float(
        timeout_override
        if timeout_override is not None
        else cfg.get("timeout_seconds", _prompt.DEFAULT_TIMEOUT_SECONDS)
    )
    max_tokens = int(cfg.get("max_tokens", _prompt.DEFAULT_MAX_TOKENS))

    started = time.perf_counter()
    raw = ""
    err: str | None = None
    in_tok = 0
    out_tok = 0
    try:
        body = await _post_chat_completion(
            base_url=base_url,
            model_name=model_name,
            system_prompt=_prompt.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            timeout_seconds=timeout,
            prefill=_prompt.JSON_PREFILL,
        )
        choices = body.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            raw = (msg.get("content") or "").strip()
        usage = body.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        out_tok = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    except asyncio.TimeoutError:
        err = "timeout"
    except httpx.HTTPError as e:
        err = f"http_error: {e}"
    except Exception as e:  # noqa: BLE001
        err = f"exception: {e}"

    latency_ms = int((time.perf_counter() - started) * 1000)

    if err is not None:
        return ObserverDecision(
            action=("allow" if allow_deny_tool else "noop"),
            reason=err,
            raw_response=raw,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            error=err,
        )

    parsed = _parse_observer_json(raw)
    if parsed is None:
        logger.warning(
            "[iv.observer] parse_failed (allow_deny_tool=%s) raw=%r",
            allow_deny_tool, raw[:300],
        )
        return ObserverDecision(
            action=("allow" if allow_deny_tool else "noop"),
            reason="parse_failed",
            raw_response=raw,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=latency_ms,
            error="parse_failed",
        )

    action = _normalize_action(parsed.get("action"), allow_deny_tool=allow_deny_tool)
    reason = str(parsed.get("reason") or "")[:500]
    content = str(parsed.get("content") or "")[:4000]
    return ObserverDecision(
        action=action,
        reason=reason,
        content=content,
        raw_response=raw,
        input_tokens=in_tok,
        output_tokens=out_tok,
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Observer state per turn
# ---------------------------------------------------------------------------


@dataclass
class ObserverState:
    """All per-turn state lives here. Passed by reference into hooks so the
    observer can mutate primary's chat_messages and cancel_event safely."""

    session_id: str
    turn_id: str
    user_request: str
    chat_messages_handle: list[dict[str, Any]]  # observer mutates this
    cancel_event: asyncio.Event
    enqueue_ambient_callback: Callable[[str, str], Awaitable[None]] | None
    clarify_callback: Callable[[str, str], Awaitable[None]] | None = None
    # persist_intervention_callback(kind, content, reason) — writes a
    # user-visible breadcrumb to the session JSON for inject/cancel actions
    # so the user sees what the observer did. Optional; None = no breadcrumb.
    persist_intervention_callback: Callable[[str, str, str], Awaitable[None]] | None = None
    primary_model: str = ""
    accumulated_text: str = ""
    interventions_used: int = 0
    intervention_budget: int = _prompt.DEFAULT_INTERVENTION_BUDGET
    sequence: int = 0
    cfg: dict[str, Any] = field(default_factory=dict)
    closed: bool = False
    # Goal card extracted at turn start (None on extraction failure or
    # when extraction is disabled).
    goal_card: dict[str, Any] | None = None
    # Compact log of decisions made this turn — threaded back into the
    # observer's per-event prompt so it can escalate or back off.
    decisions_this_turn: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Lever dispatch
# ---------------------------------------------------------------------------


async def _apply_lever(
    state: ObserverState, decision: ObserverDecision, trigger: str,
    *, related_tool: str | None = None,
) -> None:
    """Apply the observer's chosen action against primary state."""
    a = decision.action
    if a == "noop":
        return

    # Result-trigger fires AFTER the harness emitted its terminal event;
    # inject can't take effect (no further iteration will read it) and
    # cancel is moot. Translate inject → ambient when callback wired.
    if trigger == "result":
        if a == "inject":
            if state.enqueue_ambient_callback is not None and decision.content.strip():
                a = "ambient"
                decision.action = "ambient"
            else:
                decision.action = "noop_inject_on_result"
                return
        elif a == "cancel":
            decision.action = "noop_cancel_on_result"
            return
        elif a == "clarify":
            # Asking a question after the turn is over is nonsensical.
            decision.action = "noop_clarify_on_result"
            return

    # Budget gate — applies to inject/ambient/clarify only. Cancel is the
    # escape hatch lever: it terminates the loop and exits, so rationing it
    # would prevent recovery from "primary keeps ignoring my injects" cases.
    if a != "cancel" and state.interventions_used >= state.intervention_budget:
        decision.action = "noop_budget_exhausted"
        decision.reason = ((decision.reason or "") + " [budget exhausted]").strip()
        return

    if a == "inject":
        if not decision.content.strip():
            decision.action = "noop_empty_content"
            return
        # role="user" because vLLM rejects mid-stream system messages.
        state.chat_messages_handle.append(
            {"role": "user", "content": "[INNER VOICE] " + decision.content.strip()}
        )
        # Persist a user-visible breadcrumb to the session JSON so the
        # user (and the persisted history) records the intervention. The
        # chat_messages_handle append above is transient — it lives only
        # in the harness's in-memory buffer for the next iteration.
        if state.persist_intervention_callback is not None:
            try:
                await state.persist_intervention_callback(
                    "inject", decision.content, decision.reason or "",
                )
            except Exception as e:
                logger.warning("[iv.observer] inject persist failed: %s", e)
        state.interventions_used += 1
        logger.info(
            "[iv.observer] inject session=%s turn=%s reason=%s",
            state.session_id, state.turn_id, decision.reason,
        )
        _event_log.log_event(
            state.session_id,
            "inner_voice.observer_injected",
            {"trigger": trigger, "reason": decision.reason, "content": decision.content},
            turn_id=state.turn_id,
        )
        return

    if a == "cancel":
        # Cancel does NOT increment interventions_used — see budget gate above.
        # The lever ends the turn; counting it would only matter if it could
        # fire repeatedly, which it can't.
        state.cancel_event.set()
        # Persist a user-visible breadcrumb so the user sees WHY the turn
        # stopped instead of an apparent silent stall.
        if state.persist_intervention_callback is not None:
            try:
                await state.persist_intervention_callback(
                    "cancel", "", decision.reason or "",
                )
            except Exception as e:
                logger.warning("[iv.observer] cancel persist failed: %s", e)
        logger.info(
            "[iv.observer] cancel session=%s turn=%s reason=%s",
            state.session_id, state.turn_id, decision.reason,
        )
        _event_log.log_event(
            state.session_id,
            "inner_voice.observer_cancelled",
            {"trigger": trigger, "reason": decision.reason},
            turn_id=state.turn_id,
        )
        return

    if a == "ambient":
        if state.enqueue_ambient_callback is None or not decision.content.strip():
            decision.action = "noop_no_ambient_channel"
            return
        try:
            await state.enqueue_ambient_callback(decision.content, decision.reason)
        except Exception as e:
            logger.warning("[iv.observer] ambient enqueue failed: %s", e)
            decision.action = "noop_ambient_failed"
            decision.error = (decision.error or "") + f"; ambient_enqueue: {e}"
            return
        state.interventions_used += 1
        logger.info(
            "[iv.observer] ambient session=%s turn=%s reason=%s",
            state.session_id, state.turn_id, decision.reason,
        )
        _event_log.log_event(
            state.session_id,
            "inner_voice.observer_ambient",
            {"trigger": trigger, "reason": decision.reason, "content": decision.content},
            turn_id=state.turn_id,
        )
        return

    if a == "clarify":
        if state.clarify_callback is None or not decision.content.strip():
            decision.action = "noop_no_clarify_channel"
            return
        try:
            await state.clarify_callback(decision.content, decision.reason)
        except Exception as e:
            logger.warning("[iv.observer] clarify callback failed: %s", e)
            decision.action = "noop_clarify_failed"
            decision.error = (decision.error or "") + f"; clarify_cb: {e}"
            return
        # Clarify pauses the primary so the user can answer.
        state.cancel_event.set()
        state.interventions_used += 1
        logger.info(
            "[iv.observer] clarify session=%s turn=%s reason=%s",
            state.session_id, state.turn_id, decision.reason,
        )
        _event_log.log_event(
            state.session_id,
            "inner_voice.observer_clarified",
            {"trigger": trigger, "reason": decision.reason, "content": decision.content},
            turn_id=state.turn_id,
        )
        return


def _persist(
    state: ObserverState,
    decision: ObserverDecision,
    trigger: str,
    *,
    related_tool: str | None = None,
) -> None:
    state.sequence += 1
    try:
        record_inner_voice_observation(
            session_id=state.session_id,
            turn_id=state.turn_id,
            sequence_in_turn=state.sequence,
            trigger=trigger,
            action=decision.action,
            reason=decision.reason or None,
            content=decision.content or None,
            related_tool=related_tool,
            input_tokens=decision.input_tokens,
            output_tokens=decision.output_tokens,
            latency_ms=decision.latency_ms,
            model=state.primary_model,
            error=decision.error,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("[iv.observer] persist failed: %s", e)
    # Append a compact entry to the cross-event memory.
    state.decisions_this_turn.append({
        "trigger": trigger,
        "action": decision.action,
        "reason": decision.reason,
        "related_tool": related_tool,
    })


# ---------------------------------------------------------------------------
# Per-event prompt builder + LLM dispatch
# ---------------------------------------------------------------------------


def _build_event_user_prompt(
    state: ObserverState, event_summary: str,
) -> str:
    return _prompt.build_user_prompt_for_event(
        user_request=state.user_request,
        goal_card=state.goal_card,
        event_summary=event_summary,
        primary_text_so_far=state.accumulated_text[: state.cfg.get("primary_text_window_chars", 4000)],
        interventions_used=state.interventions_used,
        interventions_budget=state.intervention_budget,
        prior_decisions=state.decisions_this_turn,
    )


# ---------------------------------------------------------------------------
# Public API: install_observer
# ---------------------------------------------------------------------------


def install_observer(
    *,
    hooks: Any,  # HookRegistry
    session_id: str,
    turn_id: str,
    user_request: str,
    chat_messages_handle: list[dict[str, Any]],
    cancel_event: asyncio.Event,
    primary_model: str,
    enqueue_ambient_callback: Callable[[str, str], Awaitable[None]] | None = None,
    clarify_callback: Callable[[str, str], Awaitable[None]] | None = None,
    persist_intervention_callback: Callable[[str, str, str], Awaitable[None]] | None = None,
    goal_card: dict[str, Any] | None = None,
) -> ObserverState:
    """Install observer hooks onto a HookRegistry for one primary turn.

    Returns the ObserverState. `goal_card` may be None if extraction was
    skipped or failed; the observer will still run with lighter-touch
    judgment.
    """
    cfg = _observer_cfg()
    state = ObserverState(
        session_id=session_id,
        turn_id=turn_id,
        user_request=user_request,
        chat_messages_handle=chat_messages_handle,
        cancel_event=cancel_event,
        enqueue_ambient_callback=enqueue_ambient_callback,
        clarify_callback=clarify_callback,
        persist_intervention_callback=persist_intervention_callback,
        primary_model=primary_model,
        intervention_budget=int(cfg.get("intervention_budget", _prompt.DEFAULT_INTERVENTION_BUDGET)),
        cfg=cfg,
        goal_card=goal_card,
    )
    fast_path_enabled = bool(cfg.get("fast_path_enabled", True))

    async def pretool_cb(
        input_data: dict[str, Any], tool_use_id: str | None, _ctx: Any
    ) -> dict[str, Any]:
        if state.closed:
            return {}
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input") or {}

        # Tier 1: cheap deterministic check for benign tools.
        if fast_path_enabled:
            fp = _fast_path_pretool(tool_name, tool_input)
            if fp is not None:
                _persist(state, fp, trigger="pretool", related_tool=tool_name)
                return {}  # allow

        # Tier 2: LLM judgment for everything else.
        summary = _prompt.build_pretool_event_summary(tool_name, tool_input)
        user_prompt = _build_event_user_prompt(state, summary)
        decision = await _call_observer(
            user_prompt=user_prompt,
            allow_deny_tool=True,
            cfg=state.cfg,
            timeout_override=float(
                state.cfg.get("pretool_timeout_seconds", _prompt.DEFAULT_PRETOOL_TIMEOUT_SECONDS)
            ),
        )
        _persist(state, decision, trigger="pretool", related_tool=tool_name)
        if decision.action == "deny_tool":
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": decision.content
                    or decision.reason
                    or "Inner Voice denied this tool call.",
                }
            }
        return {}

    async def on_event_cb(evt: dict[str, Any]) -> None:
        if state.closed:
            return
        etype = evt.get("type")
        if etype == "text_delta":
            state.accumulated_text += evt.get("text", "")
            return

        if etype == "assistant_message":
            text = evt.get("text", "") or ""
            tool_calls = evt.get("tool_calls", []) or []
            iteration = int(evt.get("iteration", 0))
            summary = _prompt.build_assistant_message_summary(iteration, text, tool_calls)
            user_prompt = _build_event_user_prompt(state, summary)
            decision = await _call_observer(
                user_prompt=user_prompt, allow_deny_tool=False, cfg=state.cfg,
            )
            await _apply_lever(state, decision, trigger="assistant_message")
            _persist(state, decision, trigger="assistant_message")
            return

        if etype == "tool_result":
            tool_name = evt.get("name", "") or ""
            content = evt.get("content", "") or ""
            is_error = bool(evt.get("is_error", False))

            # Tier 1: cheap check for benign small results / parse-retry errors.
            if fast_path_enabled:
                fp = _fast_path_tool_result(tool_name, content, is_error)
                if fp is not None:
                    _persist(state, fp, trigger="tool_result", related_tool=tool_name)
                    return

            # Tier 2: LLM judgment.
            summary = _prompt.build_tool_result_summary(tool_name, content, is_error)
            user_prompt = _build_event_user_prompt(state, summary)
            decision = await _call_observer(
                user_prompt=user_prompt, allow_deny_tool=False, cfg=state.cfg,
            )
            await _apply_lever(state, decision, trigger="tool_result", related_tool=tool_name)
            _persist(state, decision, trigger="tool_result", related_tool=tool_name)
            return

        if etype == "result":
            stop_reason = evt.get("stop_reason", "") or ""
            response_text = evt.get("response_text", "") or ""
            summary = _prompt.build_result_summary(stop_reason, response_text)
            user_prompt = _build_event_user_prompt(state, summary)
            decision = await _call_observer(
                user_prompt=user_prompt, allow_deny_tool=False, cfg=state.cfg,
            )
            await _apply_lever(state, decision, trigger="result")
            _persist(state, decision, trigger="result")
            state.closed = True
            return

    hooks.add_pre_tool_use(None, pretool_cb)
    hooks.add_on_event(on_event_cb)
    return state


def close_observer(state: ObserverState) -> None:
    """Mark the observer closed. Called from `_run_turn` finally block.

    No-op today (the hook callbacks check state.closed) but reserved for
    future cleanup.
    """
    state.closed = True
