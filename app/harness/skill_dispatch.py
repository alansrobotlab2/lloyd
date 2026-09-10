"""Dispatch-time skill delivery — the matched SKILL.md at the tool call, not only at turn start (#536).

Lloyd already has a skill injector. `prefetch.py` scores the *turn text* once,
before the turn, and injects the winner's body (`prefetch.py:40-42` thresholds,
`_search_skills`, rendered in `_format_context`). What it structurally cannot do
is key on the tool the agent is about to call: on a boilerplate worker prompt
the turn-text match has nothing protocol-shaped to match on, so the rule that
matters ("never reach for yt-dlp on this box", "restart this unit, not that
one", "write the note at this path") is absent at the only moment it is
load-bearing. The logged cost is the 08-26 lap rate (1 of 3 transcript sessions
skill-first; the non-skill-first one paid a retry).

This module is the second injector, and it is deliberately not a replacement:

* it runs **inside** the PreToolUse walk, on a drafted call, so it keys on tool
  name plus argument pattern rather than on prose;
* the drafted call is **not executed**. The loop returns a *non-error*
  synthetic tool result carrying the matched SKILL.md body, so the model reads
  the protocol and re-issues the call — the same shape as the synthetic
  non-error `ToolSearch` result the loop already produces
  (`app/harness/loop.py:1501`), not the deny shape (`loop.py:1536-1540`), which
  comes back `is_error=True` and is booked into `tool_errors`
  (`autonomy.py:912`, `:927`) — the very number this feature is meant to
  improve. That second PreToolUse outcome is what `HookRegistry.fire_pre_tool_use`
  now recognises alongside `deny`;
* it is **default-off**, one rule set, and it yields: a safety deny outranks a
  delivery regardless of registration order, and a skill that already reached
  this turn — by the turn-start injector or by an earlier delivery — is not
  delivered a second time.

The rule set is small and argument-shaped on purpose. Every trigger costs the
model one extra round-trip (~2.7 s TTFT here), so a false positive is expensive
and the patterns name failure modes that were actually logged — a `Bash` that
reaches for yt-dlp on a box with no Node runtime, a `supervisorctl` restart
aimed at the wrong unit — not topics.

Recuris (arXiv:2608.24876) calls this call-time invocation. Only the delivery
half is implemented: no E/W/C skill evolution, no meta-agent, no admission gate.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from app.harness.hooks import HookRegistry

logger = logging.getLogger("lloyd-harness-skill-dispatch")

# A delivered body has to be small enough that the extra round-trip it buys is
# not paid for twice in prompt. 6000 chars ≈ 1.5k tokens at the ~4 chars/token
# this repo measures elsewhere (prefetch's SKILL_BODY_MAX is the same size).
MAX_DELIVERY_CHARS = 6000

# Skill bodies are read off disk; this is the cache lifetime, in seconds. The
# library changes on a nightly cadence, not a per-dispatch one.
_SKILL_CACHE_TTL_S = 300.0

_BODY_CACHE: dict[str, tuple[float, str]] = {}


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DispatchRule:
    """One dispatch-time trigger: a tool name plus an argument pattern.

    `fields` names the entries of `tool_input` the patterns are searched in. It
    is a whitelist, not a hint: if the drafted call carries none of those keys
    the rule cannot match, which is what keeps a `Bash` rule from firing on a
    `Write` whose `content` happens to mention `supervisorctl`.

    `fields` empty means "any string-valued argument" — used only by rules whose
    patterns are specific enough to be safe anywhere.
    """

    skill: str
    label: str
    tool: str | None = None  # None = any tool
    fields: tuple[str, ...] = ()
    patterns: tuple[re.Pattern[str], ...] = field(default_factory=tuple)

    def matches(self, tool_name: str, tool_input: dict[str, Any]) -> bool:
        if self.tool is not None and self.tool != tool_name:
            return False
        haystack = _field_text(tool_input, self.fields)
        if not haystack:
            return False
        return any(p.search(haystack) for p in self.patterns)


def _field_text(tool_input: dict[str, Any], fields: tuple[str, ...]) -> str:
    """Concatenate the arguments a rule is allowed to look at, or all of them."""
    if not isinstance(tool_input, dict):
        return ""
    if fields:
        parts = [tool_input.get(f) for f in fields]
    else:
        parts = list(tool_input.values())
    return "\n".join(p for p in parts if isinstance(p, str) and p)


# Ordered: first match wins, so the specific protocol goes before the general
# one it could also describe (restarting agent-tts is a voice-mode event and
# also, loosely, a service restart).
DISPATCH_RULES: tuple[DispatchRule, ...] = (
    DispatchRule(
        skill="voice-mode",
        label="bash drives the voice pipeline",
        tool="Bash",
        fields=("command",),
        patterns=(
            # Action-gated, in either word order. `supervisorctl status
            # agent-tts` is every health check on this box, and delivering a
            # protocol card on a read is a pure false positive — the acceptance
            # bounds spurious triggers at ~10% of dispatches precisely because
            # each one costs a round-trip.
            re.compile(
                r"\b(?:restart|stop|start|enable|disable|signal|kill)\b[^\n]*"
                r"\b(?:agent-(?:tts|livekit-server)|lloyd-voice-mode\.service)\b"
            ),
            re.compile(
                r"\b(?:agent-(?:tts|livekit-server)|lloyd-voice-mode\.service)\b[^\n]*"
                r"\b(?:restart|stop|start|enable|disable)\b"
            ),
            # Launching the script directly is itself the thing the skill
            # forbids, so the invocation is the whole trigger.
            re.compile(r"\bpython\S*[^\n]*\bvoice_mode\.py\b"),
        ),
    ),
    DispatchRule(
        skill="restart-lloyd",
        label="bash restarts a lloyd service",
        tool="Bash",
        fields=("command",),
        patterns=(
            # `supervisorctl ... restart|stop|start` — deliberately not `status`:
            # every health check reads supervisorctl state, and a delivery on a
            # read would be a pure false positive.
            re.compile(r"\bsupervisorctl\b[^\n|;&]*\b(restart|stop|start|signal)\b"),
            re.compile(r"\bsystemctl\s+--user\s+(?:restart|stop|start)\s+\S*lloyd\S*"),
        ),
    ),
    DispatchRule(
        skill="youtube-transcript",
        label="bash reaches for a transcript extractor",
        tool="Bash",
        fields=("command",),
        patterns=(
            # The skill's own HARD guardrail: this box has no Node runtime, so
            # yt-dlp is the wrong tool and has cost two confirmed failures.
            re.compile(r"\byt[-_]dlp\b"),
            re.compile(r"\byoutube[-_]transcript(?:[-_]api)?\b"),
            re.compile(r"\btranscriptExtractor\b"),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _config() -> dict[str, Any]:
    try:
        from app.config import CONFIG

        cfg = CONFIG.get("harness", {}).get("skill_dispatch", {}) or {}
    except Exception:  # pragma: no cover - config load is not under test
        return {}
    return cfg if isinstance(cfg, dict) else {}


def enabled() -> bool:
    """Default-off: an absent key, an unreadable config, or a false flag all
    mean the deliverer is not installed. Nothing about this module is on by
    default, including in tests that build their own rule set.

    The two keys live in the runtime config, which the self-mod preflight
    denies — a human sets them:

        harness:
          skill_dispatch:
            enabled: true      # absent == false
            skills: [youtube-transcript]   # absent/[] == every rule in the table
    """
    return bool(_config().get("enabled", False))


def rules_for(name: str | None) -> tuple[DispatchRule, ...]:
    """The enabled rule set, optionally restricted to one skill.

    `only_skills` is how one source is enabled at a time — the acceptance asks
    for a rollout per protocol, and `skills` in config keeps that out of code.
    """
    rules = DISPATCH_RULES
    only = _config().get("skills")
    if isinstance(only, list) and only:
        wanted = {str(s) for s in only}
        rules = tuple(r for r in rules if r.skill in wanted)
    if name:
        rules = tuple(r for r in rules if r.skill == name)
    return rules


# ---------------------------------------------------------------------------
# Body loading
# ---------------------------------------------------------------------------


def skill_body(name: str) -> str:
    """Full SKILL.md text for one skill, or "" when it is not on disk.

    Read through the same loader the turn-start injector uses, so a quarantined
    skill is unreachable from both paths rather than reachable from one.
    """
    now = time.time()
    hit = _BODY_CACHE.get(name)
    if hit and hit[0] > now:
        return hit[1]
    body = ""
    try:
        from agent_mcp.skills import _iter_skills

        for skill in _iter_skills():
            if skill.get("name") == name:
                body = skill.get("raw") or ""
                break
    except Exception as exc:  # pragma: no cover - disk/IO failure
        logger.warning("skill_dispatch: failed to load skill %r: %s", name, exc)
        return ""
    _BODY_CACHE[name] = (now + _SKILL_CACHE_TTL_S, body)
    return body


def cap_body(body: str, limit: int = MAX_DELIVERY_CHARS) -> str:
    if len(body) <= limit:
        return body
    return body[:limit] + "\n[... truncated for dispatch-time delivery]"


def render_delivery(rule: DispatchRule, body: str) -> str:
    """The synthetic tool_result content.

    It has to say three things unambiguously, because it arrives in the place
    in history where the tool's answer would have been: the call did not run,
    what is being handed over, and that re-issuing is the expected next move.
    """
    return (
        f"[harness] This call was NOT executed. Before you re-issue it, here is "
        f"the protocol Lloyd uses for it (matched rule: {rule.label}; skill: "
        f"{rule.skill}). Follow it, then re-issue the call — with the arguments "
        f"the protocol actually calls for.\n\n"
        f'<skill-dispatch name="{rule.skill}" rule="{rule.label}">\n'
        f"{body}\n"
        f"</skill-dispatch>"
    )


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def match_rule(
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    rules: tuple[DispatchRule, ...] | None = None,
) -> DispatchRule | None:
    """First rule whose tool + argument pattern matches, else None."""
    for rule in rules if rules is not None else DISPATCH_RULES:
        try:
            if rule.matches(tool_name, tool_input):
                return rule
        except Exception as exc:  # pragma: no cover - defensive against a bad rule
            logger.warning("skill_dispatch: rule %s/%s errored: %s", rule.skill, rule.label, exc)
    return None


# Advisory counters for the probe harness and the logs. Not synchronized: the
# numbers are a report, not a ledger, and a lost increment on a contended
# counter is not worth a lock on every dispatch.
STATS: dict[str, int] = {
    "dispatched": 0,
    "delivered": 0,
    "skipped_already_injected": 0,
    "skipped_already_delivered": 0,
    "skipped_no_body": 0,
    "delivered_chars": 0,
}


def stats_snapshot() -> dict[str, int]:
    return dict(STATS)


def reset_stats() -> None:
    for key in STATS:
        STATS[key] = 0


_SKILL_TAG_RE = re.compile(r'<skill(?:-dispatch)? name="([^"]+)"')


def injected_skill_names(context_text: str) -> set[str]:
    """Skills the turn-start prefetch already injected into this turn.

    Parses the `<skill name="...">` blocks `_format_context` renders
    (`prefetch.py:895/:903`) rather than reaching into prefetch's internals, so
    this keeps working if the injector changes shape and cannot accidentally
    diverge from what actually landed in the prompt.
    """
    if not context_text:
        return set()
    return set(_SKILL_TAG_RE.findall(context_text))


# ---------------------------------------------------------------------------
# Hook callback + installer
# ---------------------------------------------------------------------------


def build_pretool_cb(
    *,
    rules: tuple[DispatchRule, ...],
    already_injected: set[str],
    delivered: set[str],
) -> Any:
    """Build the per-turn PreToolUse callback.

    `already_injected` is the turn-start set from prefetch and `delivered` starts
    empty per turn: both make a second delivery of the same skill in the same
    turn pointless, and `delivered` is what stops a re-issued call being held
    back a second time — the loop that would otherwise burn the turn budget on
    one protocol. The pair lives in this closure, not in module state, so it
    cannot leak across turns or sessions.
    """

    async def _skill_dispatch_pretool(
        input_data: dict[str, Any], _tool_use_id: str | None, _ctx: Any
    ) -> dict[str, Any]:
        STATS["dispatched"] += 1
        tool_name = str(input_data.get("tool_name") or "")
        tool_input = input_data.get("tool_input") or {}
        rule = match_rule(tool_name, tool_input if isinstance(tool_input, dict) else {}, rules=rules)
        if rule is None:
            return {}
        if rule.skill in already_injected:
            STATS["skipped_already_injected"] += 1
            return {}
        if rule.skill in delivered:
            STATS["skipped_already_delivered"] += 1
            return {}
        body = cap_body(skill_body(rule.skill))
        if not body.strip():
            # A rule naming a skill that is not on disk must never swallow a
            # call with an empty lesson attached.
            STATS["skipped_no_body"] += 1
            logger.warning(
                "skill_dispatch: rule %s (%s) matched but skill body is empty/missing",
                rule.skill, rule.label,
            )
            return {}
        delivered.add(rule.skill)
        STATS["delivered"] += 1
        STATS["delivered_chars"] += len(body)
        logger.info(
            "[skill_dispatch] delivering skill=%s rule=%r on %s session=%s (%d chars)",
            rule.skill, rule.label, tool_name, input_data.get("session_id"), len(body),
        )
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                # The second PreToolUse outcome (#738): the loop must return a
                # NON-error result, because a denied call is counted as a tool
                # error by the fleet metrics this feature is measured with.
                "skillDeliver": {
                    "skill": rule.skill,
                    "label": rule.label,
                    "content": render_delivery(rule, body),
                },
            }
        }

    return _skill_dispatch_pretool


def install_skill_dispatch_hook(
    hooks: HookRegistry,
    *,
    already_injected: set[str] | None = None,
    force_enabled: bool | None = None,
    rules: tuple[DispatchRule, ...] | None = None,
) -> bool:
    """Install the dispatch-time deliverer on a turn's HookRegistry.

    Returns whether it was installed. Call it AFTER
    `install_default_safety_hook`: registration order is not what makes safety
    win — `fire_pre_tool_use` lets a deny beat a delivery either way — but
    installing the gate first keeps the walk reading in the order it matters.

    `force_enabled` exists for the probe harness and the tests; production
    callers pass nothing and get the config flag.
    """
    on = force_enabled if force_enabled is not None else enabled()
    if not on:
        return False
    active = rules if rules is not None else rules_for(None)
    if not active:
        return False
    hooks.add_pre_tool_use(None, build_pretool_cb(
        rules=active,
        already_injected=set(already_injected or ()),
        delivered=set(),
    ))
    return True
