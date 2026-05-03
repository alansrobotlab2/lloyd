"""Inner Voice (#345) Stage 1 — mechanical heuristics, no the critic.

Two responsibilities, both deterministic and runtime-cheap:

  1. **PreToolUse hook callback** that regex-matches dangerous tool calls
     against `inner_voice.pretooluse_deny` rules in config.yaml. Denies
     return an SDK `permissionDecision: "deny"` so the tool never runs.

  2. **Post-loop completion heuristic** — looks at an ambient turn's
     final response and decides whether the agent prematurely stopped
     without emitting `SIGNAL:TASK_COMPLETE` / `SIGNAL:BLOCKED` and
     without a terminal tool call. When premature, builds an ambient
     prefetch entry that nudges the agent on its next user turn.

Stage 2+ replaces the post-loop heuristic with a the critic ensemble call.
The PreToolUse callback stays — it runs ahead of any the critic inference
and never costs an LLM call, so it's the cheapest possible safety layer.

## Rule shape (config.yaml `inner_voice.pretooluse_deny[]`)

    - tool: Bash
      pattern: 'rm\\s+-rf\\s+.*(?:obsidian|\\.openclaw|agent-services|\\.venvs|lloyd)'
      reason: "destructive deletion against guarded path"

`tool` matches the SDK's tool name (e.g. "Bash", "Write", "Edit"). `pattern`
is a Python regex compiled with `IGNORECASE | MULTILINE`. `reason` is the
human-readable string surfaced to the model in the deny response.

## What does NOT cross this threshold

- Bench scores. The bench runner (`scripts/autoresearch/bench_runner.py`)
  hits vLLM directly with a single chat completion — it never touches the
  SDK, so PreToolUse hooks have no effect on bench numbers. Stage 1's
  validation path is runtime smoke testing, not bench score deltas.
- Anything beyond the runtime SDK path. If the agent shells out via a
  subprocess of its own that doesn't route through ClaudeAgentOptions,
  this layer cannot see it.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.config import CONFIG
from app import event_log

logger = logging.getLogger("lloyd-server")


# ---------------------------------------------------------------------------
# Rule loading & compilation
# ---------------------------------------------------------------------------

# Cached compiled rules. Loaded lazily on first use; reset on `reload_rules`.
_compiled_rules_cache: list[dict[str, Any]] | None = None


def _load_pretooluse_deny_rules() -> list[dict[str, Any]]:
    """Read `inner_voice.pretooluse_deny` from CONFIG and compile patterns.

    Each input rule is a dict {"tool", "pattern", "reason"}. Compiled rules
    add a `compiled` key holding the `re.Pattern`. Malformed rules (missing
    fields, bad regex) are logged at WARNING and skipped — this never raises
    into the SDK callback path.
    """
    global _compiled_rules_cache
    if _compiled_rules_cache is not None:
        return _compiled_rules_cache

    raw = (CONFIG.get("inner_voice") or {}).get("pretooluse_deny") or []
    out: list[dict[str, Any]] = []
    for rule in raw:
        if not isinstance(rule, dict):
            logger.warning(
                "inner_voice.pretooluse_deny: skipping non-dict rule: %r", rule
            )
            continue
        tool = rule.get("tool", "")
        pattern = rule.get("pattern", "")
        reason = rule.get("reason", "")
        if not tool or not pattern:
            logger.warning(
                "inner_voice.pretooluse_deny: skipping rule missing tool/pattern: %r",
                rule,
            )
            continue
        try:
            compiled = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        except re.error as e:
            logger.warning(
                "inner_voice.pretooluse_deny: bad regex %r: %s", pattern, e
            )
            continue
        out.append(
            {
                "tool": tool,
                "pattern": pattern,
                "reason": reason,
                "compiled": compiled,
            }
        )
    _compiled_rules_cache = out
    if out:
        logger.info(
            "inner_voice.pretooluse_deny: loaded %d rule(s)", len(out)
        )
    return out


def reload_rules() -> None:
    """Force re-read of rules from CONFIG. Call after editing config.yaml."""
    global _compiled_rules_cache
    _compiled_rules_cache = None


# ---------------------------------------------------------------------------
# PreToolUse hook callback
# ---------------------------------------------------------------------------

# Tool input keys we'll search for the regex haystack. For Bash the field
# is `command`. We union over a small set so future deny rules can target
# Edit/Write content without rewriting the matcher.
_HAYSTACK_FIELDS = ("command", "cmd", "script", "code", "content", "new_string")


def _serialize_input_for_match(tool_input: Any) -> str:
    """Best-effort flattening of `tool_input` to a string for regex match.

    Joins the values of keys in `_HAYSTACK_FIELDS` with newlines. Falls
    through to a dump of every string-valued field if no preferred key is
    present — keeps the matcher useful for tools we haven't named yet.
    """
    if not isinstance(tool_input, dict):
        return str(tool_input)
    parts: list[str] = []
    for key in _HAYSTACK_FIELDS:
        v = tool_input.get(key)
        if isinstance(v, str):
            parts.append(v)
    if not parts:
        for v in tool_input.values():
            if isinstance(v, str):
                parts.append(v)
    return "\n".join(parts)


async def _evaluate_and_log(
    lloyd_session_id: str,
    input_data: dict[str, Any],
    tool_use_id: str | None,
) -> dict[str, Any]:
    """Core evaluation — shared by `pretooluse_callback` (no session bind)
    and `make_pretooluse_callback(lloyd_session_id)` (closure-bound).

    The split exists because `PreToolUseHookInput.session_id` is the
    *Claude CLI* session UUID, not Lloyd's session_id. The SDK can't
    plumb our session_id through, so the messages.py wiring builds a
    closure that captures Lloyd's session_id at options-build time and
    passes it in here.
    """
    sdk_session_id = input_data.get("session_id", "") or ""
    tool_name = input_data.get("tool_name", "") or ""
    tool_input = input_data.get("tool_input", {}) or {}
    haystack = _serialize_input_for_match(tool_input)

    matched_rule: dict[str, Any] | None = None
    rules_evaluated = 0
    try:
        for rule in _load_pretooluse_deny_rules():
            if rule["tool"] != tool_name:
                continue
            rules_evaluated += 1
            if rule["compiled"].search(haystack):
                matched_rule = rule
                break
    except Exception as e:
        # Defensive: never let regex eval take down the chat path. A miss
        # is safer than a crash; the agent retains full tool access.
        logger.warning("pretooluse_callback regex eval failed: %s", e)

    # Log the evaluation regardless of decision. Always write to the
    # Lloyd session_id file — that's where the rest of the brain1.* events
    # for this turn already live. Capture the SDK session_id alongside so
    # the two namespaces stay forensically linked.
    try:
        event_log.log_event(
            lloyd_session_id or sdk_session_id,
            "inner_voice.pre_tool_use_evaluated",
            {
                "tool_name": tool_name,
                "tool_use_id": tool_use_id,
                "sdk_session_id": sdk_session_id,
                "haystack_excerpt": haystack[:200],
                "rules_evaluated": rules_evaluated,
                "matched": bool(matched_rule),
                "matched_pattern": matched_rule["pattern"] if matched_rule else None,
                "matched_reason": matched_rule["reason"] if matched_rule else None,
                "decision": "deny" if matched_rule else "pass",
            },
        )
    except Exception as e:
        logger.warning("pretooluse_callback event_log failed: %s", e)

    if matched_rule:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    "Inner Voice safety rule blocked this call: "
                    f"{matched_rule['reason']}"
                ),
            }
        }
    # Empty dict = pass-through (default permission semantics apply).
    return {}


async def pretooluse_callback(
    input_data: dict[str, Any],  # PreToolUseHookInput TypedDict
    tool_use_id: str | None,
    context: Any,                 # HookContext (unused in Stage 1)
) -> dict[str, Any]:
    """Standalone PreToolUse callback — uses the SDK session_id for logging.

    Kept for unit tests and any caller that doesn't want to thread Lloyd's
    session_id through. Production wiring uses
    `make_pretooluse_callback(lloyd_session_id)` instead so events land in
    the Lloyd-keyed events.jsonl file.
    """
    return await _evaluate_and_log(
        "", input_data, tool_use_id,
    )


def make_pretooluse_callback(lloyd_session_id: str):
    """Return a PreToolUse callback bound to a specific Lloyd session_id.

    Usage at options-build time:
        hooks = {
            "PreToolUse": [HookMatcher(matcher="Bash",
                hooks=[make_pretooluse_callback(session_id)])],
        }
    """
    async def _bound(
        input_data: dict[str, Any],
        tool_use_id: str | None,
        context: Any,
    ) -> dict[str, Any]:
        return await _evaluate_and_log(lloyd_session_id, input_data, tool_use_id)
    return _bound


# ---------------------------------------------------------------------------
# Post-loop completion heuristic
# ---------------------------------------------------------------------------

# Tokens that indicate a turn explicitly signaled completion. Any of these
# anywhere in the response means we DON'T nudge.
_TERMINAL_SIGNALS = (
    "SIGNAL:TASK_COMPLETE",
    "SIGNAL:STAGE_COMPLETE",
    "SIGNAL:BLOCKED",
)

# Tools whose presence in the turn means the agent did real work. Trust
# those terminations even without an explicit SIGNAL token. Names match
# both the bare tool name and the MCP-prefixed form (`mcp__<svr>__<tool>`).
_TERMINAL_TOOL_NAMES = frozenset(
    {
        "memory_add",
        "memory_replace",
        "fact_add",
        "fact_relate",
        "vault_write",
        "backlog_write_task",
        "autonomy_write_task",
        "Edit",
        "Write",
        "NotebookEdit",
        "email_send",
        "email_reply",
        "email_forward",
        "discord_send",
        "calendar_create",
        "ambient_decide",  # explicit "stay silent" is also a closure
    }
)


def _response_signals_completion(response_text: str) -> bool:
    """True if any SIGNAL: token appears anywhere in the response text."""
    if not response_text:
        return False
    return any(sig in response_text for sig in _TERMINAL_SIGNALS)


def _had_terminal_tool_call(tool_calls: list[dict] | None) -> bool:
    """True if any tool call from this turn matches a terminal-tool name.

    Accepts both shapes used in messages.py:
      - `{"name": "<bare-or-mcp-prefixed>", ...}`
      - `{"function": {"name": "..."}, ...}`
    """
    for tc in tool_calls or []:
        name = tc.get("name") or tc.get("function", {}).get("name", "")
        if not name:
            continue
        # Normalize MCP-prefixed names (`mcp__lloyd-mcp__memory_add`).
        short = name.rsplit("__", 1)[-1] if "__" in name else name
        if name in _TERMINAL_TOOL_NAMES or short in _TERMINAL_TOOL_NAMES:
            return True
    return False


def evaluate_completion(
    response_text: str,
    tool_calls: list[dict] | None,
) -> dict[str, Any]:
    """Decide whether an ambient turn ended prematurely.

    Returns a dict with keys:
      - `premature`     (bool)  — True iff this is a Stage 1 nudge target
      - `reason`        (str)   — human-readable explanation
      - `signal_seen`   (bool)
      - `terminal_tool` (bool)
      - `has_content`   (bool)

    Premature iff: response had non-empty content AND no SIGNAL token AND
    no terminal tool call. Empty responses are NOT nudged — that's a
    different failure class (the SDK crashed, or the cancel_event fired
    mid-stream). Stage 2+ may revisit.
    """
    signal_seen = _response_signals_completion(response_text)
    terminal_tool = _had_terminal_tool_call(tool_calls or [])
    has_content = bool((response_text or "").strip())

    if not has_content:
        return {
            "premature": False,
            "reason": "empty response — not a Stage 1 nudge target",
            "signal_seen": False,
            "terminal_tool": False,
            "has_content": False,
        }
    if signal_seen:
        return {
            "premature": False,
            "reason": "SIGNAL token present",
            "signal_seen": True,
            "terminal_tool": terminal_tool,
            "has_content": True,
        }
    if terminal_tool:
        return {
            "premature": False,
            "reason": "terminal tool call present",
            "signal_seen": False,
            "terminal_tool": True,
            "has_content": True,
        }
    return {
        "premature": True,
        "reason": "no SIGNAL token and no terminal tool call",
        "signal_seen": False,
        "terminal_tool": False,
        "has_content": True,
    }


def is_completion_check_enabled() -> bool:
    """Read the `inner_voice.ambient_completion_check` flag from CONFIG.

    Disabling this turns off the post-loop heuristic without touching any
    other Stage 1 wiring (PreToolUse stays armed).
    """
    return bool(
        (CONFIG.get("inner_voice") or {}).get("ambient_completion_check", False)
    )


# ---------------------------------------------------------------------------
# Nudge construction (passive — surfaces via ambient prefetch)
# ---------------------------------------------------------------------------

def make_completion_nudge_entry(
    turn_id: str,
    response_excerpt: str,
) -> dict[str, str]:
    """Build the AmbientPrefetchEntry kwargs that nudge the agent.

    Stage 1 uses the passive prefetch path: the nudge surfaces in the
    `<context>` block of the next user turn rather than firing a fresh
    ambient turn (which would cost an SDK invocation). Stage 2+ may
    upgrade to active interruption when the critic disagreement crosses
    the veto threshold.

    `dedup_key` collapses repeat nudges for the same turn — if the
    completion check fires twice on the same turn (defensive against
    duplicate post-capture), only the latest nudge survives in the
    queue.
    """
    excerpt = (response_excerpt or "")[:300]
    return {
        "source": "inner_voice:completion_check",
        "summary": (
            f"Previous ambient turn ({turn_id}) closed without SIGNAL or "
            "terminal tool call — finish or signal blocked."
        ),
        "content": (
            "Inner Voice (#345 Stage 1, heuristic) flagged your last ambient "
            "turn as ending mid-task: no SIGNAL:TASK_COMPLETE / SIGNAL:BLOCKED "
            "was emitted, and no terminal tool (memory_add, vault_write, "
            "autonomy_write_task, Edit, Write, …) fired. If the task is "
            "genuinely done, signal it. If you stopped early, finish.\n\n"
            f"Tail of last response: {excerpt}"
        ),
        "dedup_key": f"inner_voice:completion_check:{turn_id}",
    }
