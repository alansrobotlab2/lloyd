"""Inner Voice observer — a thin second-agent harness (v4).

One observer per primary turn. The observer:

1. At turn start, runs ONE goal-extraction tool-call to produce a goal
   card (success criteria, out-of-scope, completion signals).
2. Subscribes to the primary's NormalizedEvent stream via an OnEvent hook.
3. For each significant event, runs a cheap pre-filter; if interesting,
   forces the LLM to call exactly one of the lever tools.
4. Has five soft levers: noop, inject, cancel, ambient, clarify. No
   deny_tool — destructive Bash is hard-blocked deterministically by
   `app/harness/safety.py` (default PreToolUse hook), independent of IV.

All judgment lives in `observer_prompt.SYSTEM_PROMPT` and the tool-schema
descriptions in `lever_tools.LEVER_TOOLS`. The Python here is plumbing.
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
from app.inner_voice.lever_tools import (
    GOAL_EXTRACTION_TOOL_NAME,
    GOAL_EXTRACTION_TOOLS,
    LEVER_NAMES,
    LEVER_TOOLS,
)
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
    cache_read: int = 0
    cache_create: int = 0
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
# Tool-call extraction
# ---------------------------------------------------------------------------


def _extract_tool_call(body: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Pull the single forced tool call out of a vLLM response.

    Returns `(tool_name, args_dict)` or None if vLLM returned no tool call
    or the args don't parse. The caller decides how to fall back.
    """
    choices = body.get("choices") or []
    if not choices:
        return None
    msg = choices[0].get("message") or {}
    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        return None
    tc = tool_calls[0]
    fn = tc.get("function") or {}
    name = fn.get("name") or ""
    if not name:
        return None
    raw_args = fn.get("arguments")
    if isinstance(raw_args, dict):
        return (name, raw_args)
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            return None
        if isinstance(args, dict):
            return (name, args)
    return None


async def _post_chat_completion_with_tools(
    *,
    base_url: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    tools: list[dict[str, Any]],
    max_tokens: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    """POST a chat completion with `tools` + `tool_choice="required"`.

    Returns the raw response body. Caller extracts the single forced tool
    call from `body["choices"][0]["message"]["tool_calls"][0]`. The model
    cannot return free-form content under this contract.
    """
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "tools": tools,
        "tool_choice": "required",
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "priority": 0,
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
    recent_exchanges: list[dict[str, str]] | None = None,
) -> dict[str, Any] | None:
    """Run one LLM call at turn start to extract the goal card.

    Tool-call mode: forces a single `record_goal_card` invocation; reads
    the three array fields directly from tool args. No JSON parsing.

    Returns None on any failure. Best-effort — observer falls back to
    "no goal card" mode (lighter-touch oversight) on extraction failure.

    `recent_exchanges` (last few user/assistant texts) is passed through
    so the extractor can resolve follow-up messages like "yeah do it" or
    "still broken" against the prior thread instead of producing an empty
    goal card.
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
    user_prompt = _prompt.build_goal_extraction_user_prompt(user_request, recent_exchanges)
    started = time.perf_counter()
    try:
        body = await _post_chat_completion_with_tools(
            base_url=base_url,
            model_name=model_name,
            system_prompt=_prompt.GOAL_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tools=GOAL_EXTRACTION_TOOLS,
            max_tokens=max_tokens,
            timeout_seconds=timeout,
        )
    except Exception as e:  # noqa: BLE001 — best-effort
        logger.warning("[iv.observer] goal extraction failed: %s", e)
        return None
    latency_ms = int((time.perf_counter() - started) * 1000)
    extracted = _extract_tool_call(body)
    if extracted is None:
        logger.warning("[iv.observer] goal extraction: no tool_call in %dms", latency_ms)
        return None
    name, args = extracted
    if name != GOAL_EXTRACTION_TOOL_NAME:
        logger.warning(
            "[iv.observer] goal extraction: unexpected tool %r in %dms",
            name, latency_ms,
        )
        return None
    out = {
        "success_criteria": _coerce_str_list(args.get("success_criteria")),
        "out_of_scope": _coerce_str_list(args.get("out_of_scope")),
        "completion_signals": _coerce_str_list(args.get("completion_signals")),
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
    """Cheap deterministic check for benign tool calls. Returns a `noop`
    ObserverDecision (skip the LLM call) when safe, or None to escalate.

    v4: pretool is no longer a gate, just an observation trigger. The
    fast-path is purely a perf optimization that avoids spending an LLM
    call on tools that obviously don't need IV's attention.
    """
    if not tool_name:
        return None
    if tool_name in _FAST_ALLOW_TOOL_NAMES:
        return ObserverDecision(action="noop", reason="fast-path: read-only tool")
    if tool_name == "Bash":
        cmd = (tool_args.get("command") or "") if isinstance(tool_args, dict) else ""
        if _bash_command_is_safely_readonly(cmd):
            return ObserverDecision(action="noop", reason="fast-path: read-only Bash")
        return None
    # MCP tools: fast-noop when name contains a clearly read/list verb,
    # AND args are small (large args are usually writes). Names are bare
    # (no `mcp__server__` prefix) since the harness drops the namespace.
    name_lower = tool_name.lower()
    if any(kw in name_lower for kw in _SAFE_TOOL_NAME_KEYWORDS):
        args_str = json.dumps(tool_args, default=str) if isinstance(tool_args, dict) else str(tool_args)
        if len(args_str) < 1000:
            return ObserverDecision(action="noop", reason="fast-path: read-shaped MCP tool")
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


def _fast_path_assistant_message(text: str, tool_calls: list) -> ObserverDecision | None:
    """Cheap noop for assistant messages that don't need LLM judgment.

    The pretool gate already evaluates each proposed tool with its real
    args, so re-evaluating the same tool call at the assistant_message
    boundary is duplicate work. Skip the LLM call when the iteration is
    pure tool dispatch (no text, just tool_calls). The IV will see the
    tool result on the next event and judge progress then.
    """
    if tool_calls and not text.strip():
        return ObserverDecision(
            action="noop",
            reason="fast-path: tool-dispatch-only iteration; pretool gate handles it",
        )
    return None


# ---------------------------------------------------------------------------
# Observer LLM call wrapper
# ---------------------------------------------------------------------------


async def _call_observer(
    *,
    user_prompt: str,
    cfg: dict[str, Any] | None = None,
    timeout_override: float | None = None,
) -> ObserverDecision:
    """One observer LLM call. Returns a parsed ObserverDecision.

    Tool-call mode: vLLM is forced to emit exactly one of the LEVER_TOOLS
    function calls. The tool name IS the action; args carry reason/content.
    Errors fold into a noop decision so the primary stream is never blocked
    by observer faults.
    """
    cfg = cfg or _observer_cfg()
    base_url, model_name = _resolve_endpoint()
    if not base_url:
        return ObserverDecision(
            action="noop",
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
    body: dict[str, Any] | None = None
    err: str | None = None
    in_tok = 0
    out_tok = 0
    cache_read = 0
    cache_create = 0
    try:
        body = await _post_chat_completion_with_tools(
            base_url=base_url,
            model_name=model_name,
            system_prompt=_prompt.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            tools=LEVER_TOOLS,
            max_tokens=max_tokens,
            timeout_seconds=timeout,
        )
        usage = body.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        out_tok = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        cache_read = int(usage.get("cache_read") or usage.get("prompt_tokens_cached") or 0)
        cache_create = int(usage.get("cache_create") or 0)
    except asyncio.TimeoutError:
        err = "timeout"
    except httpx.HTTPError as e:
        err = f"http_error: {e}"
    except Exception as e:  # noqa: BLE001
        err = f"exception: {e}"

    latency_ms = int((time.perf_counter() - started) * 1000)

    if err is not None:
        return ObserverDecision(
            action="noop", reason=err,
            input_tokens=in_tok, output_tokens=out_tok,
            cache_read=cache_read, cache_create=cache_create,
            latency_ms=latency_ms, error=err,
        )

    extracted = _extract_tool_call(body or {})
    if extracted is None:
        logger.warning(
            "[iv.observer] no_tool_call response in %dms body_keys=%s",
            latency_ms, list((body or {}).keys()),
        )
        return ObserverDecision(
            action="noop", reason="no_tool_call",
            input_tokens=in_tok, output_tokens=out_tok,
            cache_read=cache_read, cache_create=cache_create,
            latency_ms=latency_ms, error="no_tool_call",
        )
    name, args = extracted
    if name not in LEVER_NAMES:
        logger.warning("[iv.observer] unknown lever %r — coercing to noop", name)
        return ObserverDecision(
            action="noop", reason=f"unknown_lever:{name}",
            input_tokens=in_tok, output_tokens=out_tok,
            cache_read=cache_read, cache_create=cache_create,
            latency_ms=latency_ms, error="unknown_lever",
        )
    return ObserverDecision(
        action=name,
        reason=str(args.get("reason") or "")[:500],
        content=str(args.get("content") or "")[:4000],
        input_tokens=in_tok,
        output_tokens=out_tok,
        cache_read=cache_read,
        cache_create=cache_create,
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
    # The subliminal context block the primary saw at turn start
    # (prefetched skills, vault hits, facts, ambient signals). Surfaced
    # to the observer so it can recognize when the primary is following
    # documented procedure rather than freelancing.
    subliminal_context: str = ""


# ---------------------------------------------------------------------------
# Lever dispatch
# ---------------------------------------------------------------------------


# Reason-text patterns that look like a "task complete, stopping early"
# cancel. Still used by the cancel-for-completion guard in `on_event_cb`
# (a cancel claimed-as-complete on the first text-only iteration is
# usually the IV being wrong; the harness terminates naturally on its
# own without needing a force-stop).
_COMPLETION_REASON_PATTERN = re.compile(
    r"\b(complete|completed|done|criteria met|all met|success criteria"
    r"|stopping early|stop early|avoid padding|no more (?:work|tools)"
    r"|nothing more)\b",
    re.IGNORECASE,
)


# Strip a leading [INNER VOICE] prefix that the model sometimes parrots
# from the system prompt. The harness adds its own prefix when injecting,
# so a model-emitted one would double up: "[INNER VOICE] [INNER VOICE] ...".
_INNER_VOICE_PREFIX_RE = re.compile(
    r"^\[\s*INNER\s*VOICE\s*\]\s*", re.IGNORECASE,
)


async def _apply_lever(
    state: ObserverState, decision: ObserverDecision, trigger: str,
    *, related_tool: str | None = None,
) -> None:
    """Apply the observer's chosen action against primary state."""
    a = decision.action
    if a == "noop":
        return
    # Passive observation labels (set by upstream guards). Persist-only;
    # never apply a lever, never consume budget, never gate on trigger.
    if a.startswith("noop_") or a == "acknowledge_complete":
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
        content_clean = _INNER_VOICE_PREFIX_RE.sub("", decision.content.strip()).strip()
        if not content_clean:
            decision.action = "noop_empty_content"
            return
        # Mutate decision.content so the persisted breadcrumb and event log
        # both see the deduped text — otherwise the UI shows the doubled
        # prefix while the chat history shows the clean version.
        decision.content = content_clean
        # role="user" because vLLM rejects mid-stream system messages.
        state.chat_messages_handle.append(
            {"role": "user", "content": "[INNER VOICE] " + content_clean}
        )
        # Persist a user-visible breadcrumb to the session JSON so the
        # user (and the persisted history) records the intervention. The
        # chat_messages_handle append above is transient — it lives only
        # in the harness's in-memory buffer for the next iteration.
        if state.persist_intervention_callback is not None:
            try:
                await state.persist_intervention_callback(
                    "inject", content_clean, decision.reason or "",
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
        # Mark the observer closed so any pretool callbacks that fire
        # concurrently (the harness still dispatches tool_calls already
        # extracted from this iteration) short-circuit instead of running
        # an LLM call that could surface a confusing post-cancel deny.
        state.closed = True
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
            cache_read=decision.cache_read,
            cache_create=decision.cache_create,
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
    # Head+tail windowing instead of head-only truncation. The conclusion is
    # what determines completeness; chopping it off makes every long response
    # look "cut off mid-sentence" to the IV.
    cap = state.cfg.get("primary_text_window_chars", 4000)
    return _prompt.build_user_prompt_for_event(
        user_request=state.user_request,
        goal_card=state.goal_card,
        event_summary=event_summary,
        primary_text_so_far=_prompt.windowed_text(state.accumulated_text, cap),
        interventions_used=state.interventions_used,
        interventions_budget=state.intervention_budget,
        prior_decisions=state.decisions_this_turn,
        subliminal_context=state.subliminal_context,
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
    subliminal_context: str = "",
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
        subliminal_context=subliminal_context or "",
    )
    fast_path_enabled = bool(cfg.get("fast_path_enabled", True))

    async def pretool_cb(
        input_data: dict[str, Any], tool_use_id: str | None, _ctx: Any
    ) -> dict[str, Any]:
        # v4: pretool is observation-only. IV cannot block tool dispatch.
        # Hard safety on destructive Bash is enforced upstream by the
        # harness's default safety hook (app/harness/safety.py), which
        # runs regardless of whether IV is opted in.
        if state.closed or state.cancel_event.is_set():
            return {}
        tool_name = input_data.get("tool_name", "")
        tool_input = input_data.get("tool_input") or {}

        # Tier 1: cheap deterministic noop for benign tools — saves an LLM
        # call. The harness will dispatch the tool either way.
        if fast_path_enabled:
            fp = _fast_path_pretool(tool_name, tool_input)
            if fp is not None:
                _persist(state, fp, trigger="pretool", related_tool=tool_name)
                return {}

        # Tier 2: LLM judgment with soft-lever-only tools. Decision can be
        # noop / inject / cancel / ambient / clarify. An inject here lands
        # as the next user message after the tool dispatches; a cancel
        # ends the turn after this tool finishes. Neither blocks dispatch.
        summary = _prompt.build_pretool_event_summary(tool_name, tool_input)
        user_prompt = _build_event_user_prompt(state, summary)
        decision = await _call_observer(user_prompt=user_prompt, cfg=state.cfg)
        if state.closed or state.cancel_event.is_set():
            decision.action = "noop_pretool_after_cancel"
            decision.reason = (
                (decision.reason or "")
                + " [skipped: turn cancelled while pretool LLM was in flight]"
            ).strip()
            _persist(state, decision, trigger="pretool", related_tool=tool_name)
            return {}
        await _apply_lever(state, decision, trigger="pretool", related_tool=tool_name)
        _persist(state, decision, trigger="pretool", related_tool=tool_name)
        return {}

    async def on_event_cb(evt: dict[str, Any]) -> None:
        # Mirror the pretool guard: if the user (or IV itself) cancelled the
        # turn, stop observing. Otherwise the observer keeps issuing LLM
        # judgment calls and applying levers (injects, ambients) for a turn
        # the user already abandoned.
        if state.closed or state.cancel_event.is_set():
            return
        etype = evt.get("type")
        if etype == "text_delta":
            state.accumulated_text += evt.get("text", "")
            return

        if etype == "assistant_message":
            text = evt.get("text", "") or ""
            tool_calls = evt.get("tool_calls", []) or []
            iteration = int(evt.get("iteration", 0))

            # Tier 1: cheap noop for tool-dispatch-only iterations. The
            # pretool gate evaluates each proposed tool with its real args
            # — running an LLM here too is duplicate work.
            if fast_path_enabled:
                fp = _fast_path_assistant_message(text, tool_calls)
                if fp is not None:
                    _persist(state, fp, trigger="assistant_message")
                    return

            finish_reason = str(evt.get("finish_reason") or "stop")
            summary = _prompt.build_assistant_message_summary(
                iteration, text, tool_calls, finish_reason,
                goal_card=state.goal_card,
            )
            user_prompt = _build_event_user_prompt(state, summary)
            decision = await _call_observer(
                user_prompt=user_prompt, cfg=state.cfg,
            )
            if state.closed or state.cancel_event.is_set():
                decision.action = "noop_assistant_after_cancel"
                decision.reason = (
                    (decision.reason or "")
                    + " [skipped: turn cancelled while observer LLM was in flight]"
                ).strip()
                _persist(state, decision, trigger="assistant_message")
                return
            # Block consecutive injects. The prompt's escalation rule asks the
            # IV to convert a second-on-same-theme decision to cancel/noop, but
            # small models reliably ignore that. If the immediately previous
            # assistant_message decision was also an inject, force this one to
            # noop — give the primary at least one full iteration to act on the
            # earlier nudge before we fire again. The IV can still escalate to
            # cancel through the normal path; this only suppresses inject-on-inject.
            if decision.action == "inject":
                last_inject_idx = -1
                for i in range(len(state.decisions_this_turn) - 1, -1, -1):
                    d = state.decisions_this_turn[i]
                    if d.get("trigger") != "assistant_message":
                        continue
                    if d.get("action") == "inject":
                        last_inject_idx = i
                    break
                if last_inject_idx >= 0:
                    logger.info(
                        "[iv.observer] suppressed consecutive inject session=%s "
                        "turn=%s reason=%r",
                        state.session_id, state.turn_id, decision.reason,
                    )
                    _event_log.log_event(
                        state.session_id,
                        "inner_voice.inject_suppressed_consecutive",
                        {"reason": decision.reason, "content": decision.content},
                        turn_id=state.turn_id,
                    )
                    decision.action = "noop_inject_after_inject"
                    decision.reason = (
                        (decision.reason or "")
                        + " [suppressed: prior assistant_message decision was inject; "
                        "give primary one iteration to respond]"
                    ).strip()
            # Block cancel-for-completion. There are two failure modes the
            # IV has been hitting:
            #
            #   (a) Cancel mid-tool-sequence: the primary just emitted tool
            #       calls, IV claims "complete." Tool calls = work in flight;
            #       cancelling here aborts the work and (because the harness
            #       still dispatches the already-extracted tool_calls before
            #       checking cancel_event) surfaces a confusing post-cancel
            #       "Tool call denied" to the user.
            #
            #   (b) Cancel on the first text-only message: the primary just
            #       delivered its answer in a single text block, IV reads
            #       that as "delivered + now padding" and cancels. The
            #       harness would have terminated naturally on the NEXT
            #       loop iteration (no tool_calls => stop_reason="stop"), so
            #       this cancel adds a confusing red breadcrumb to a turn
            #       that completed successfully.
            #
            # In both cases the harness handles natural termination on its
            # own. Only let cancel-for-completion through after the IV has
            # already intervened this turn (i.e. it's escalating from
            # ignored injects), where force-stopping is the documented
            # escalation path.
            if (
                decision.action == "cancel"
                and _COMPLETION_REASON_PATTERN.search(decision.reason or "")
                and state.interventions_used == 0
            ):
                tool_names = [
                    (tc.get("function") or {}).get("name") or tc.get("name") or "?"
                    for tc in tool_calls
                ]
                if tool_calls:
                    # Mid-tool-sequence cancel — primary still has work in
                    # flight. Block as a guard; this is the IV being wrong,
                    # not a success signal.
                    blocked_label = "noop_cancel_with_pending_tools"
                    blocked_note = "[blocked: primary has pending tool calls; not done]"
                else:
                    # No pending tools, no prior interventions, IV thinks
                    # the answer is complete. The harness terminates the
                    # loop naturally on the next iteration — no cancel
                    # needed. Convert to a positive acknowledgement so the
                    # observations panel can render it as agreement
                    # ("IV reviewed and agrees the answer is complete")
                    # rather than a red force-stop.
                    blocked_label = "acknowledge_complete"
                    blocked_note = "[harness will terminate naturally; recorded as acknowledgement instead of cancel]"
                logger.info(
                    "[iv.observer] blocked cancel-for-completion session=%s "
                    "turn=%s action=%s reason=%r pending_tools=%s",
                    state.session_id, state.turn_id, blocked_label,
                    decision.reason, tool_names,
                )
                _event_log.log_event(
                    state.session_id,
                    "inner_voice.cancel_blocked_completion",
                    {"reason": decision.reason, "pending_tools": tool_names,
                     "interventions_used": state.interventions_used,
                     "downgraded_to": blocked_label},
                    turn_id=state.turn_id,
                )
                decision.action = blocked_label
                decision.reason = ((decision.reason or "") + " " + blocked_note).strip()
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
                user_prompt=user_prompt, cfg=state.cfg,
            )
            if state.closed or state.cancel_event.is_set():
                decision.action = "noop_tool_result_after_cancel"
                decision.reason = (
                    (decision.reason or "")
                    + " [skipped: turn cancelled while observer LLM was in flight]"
                ).strip()
                _persist(state, decision, trigger="tool_result", related_tool=tool_name)
                return
            await _apply_lever(state, decision, trigger="tool_result", related_tool=tool_name)
            _persist(state, decision, trigger="tool_result", related_tool=tool_name)
            return

        if etype == "result":
            stop_reason = evt.get("stop_reason", "") or ""
            response_text = evt.get("response_text", "") or ""
            summary = _prompt.build_result_summary(
                stop_reason, response_text, goal_card=state.goal_card,
            )
            user_prompt = _build_event_user_prompt(state, summary)
            decision = await _call_observer(
                user_prompt=user_prompt, cfg=state.cfg,
            )
            if state.cancel_event.is_set():
                decision.action = "noop_result_after_cancel"
                decision.reason = (
                    (decision.reason or "")
                    + " [skipped: turn cancelled while observer LLM was in flight]"
                ).strip()
                _persist(state, decision, trigger="result")
                state.closed = True
                return
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
