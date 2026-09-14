"""Default harness safety hook — deterministic destructive-Bash gate.

Replaces the LLM-judgment deny lever that lived in Inner Voice (v3). This
module exposes pattern-based hard denies for catastrophic operations and
installs them as a default `PreToolUse` hook on every primary turn — IV-on
or IV-off. Closes the prior gap where the safety net only ran when a
session opted into Inner Voice.

The patterns are intentionally narrow — only ops with no plausible legitimate
agent use case (sudo, dd-to-device, mkfs, fork bombs, force-push to main,
piping remote content to a shell). Everyday risky-looking commands like
`cp`, `mv`, `chmod` on a single file are *not* denied — those are normal
agent behavior and gating them would break far more than it protects.

Inner Voice still sees pretool events as observations; it just can't block
anymore. This module is the only hard gate on tool dispatch.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.harness.hooks import HookRegistry

logger = logging.getLogger("lloyd-harness-safety")


# ---------------------------------------------------------------------------
# Hard-deny pattern set — catastrophic-only
# ---------------------------------------------------------------------------

# Each entry: (compiled_regex, label).
#
# The label is what appears in the user-visible deny reason, so write it
# like a one-line warning the user will read in chat:
#   "destructive pattern '<label>' on '<excerpt>'".
#
# Ordering doesn't matter for correctness (any match denies) but is roughly
# severity-sorted for readability when scanning the source.
_HARD_DENY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Privilege escalation. No agent use case.
    (re.compile(r"\bsudo\b", re.IGNORECASE), "sudo"),

    # rm -rf on root, home, $HOME, system dirs. Conservative: matches when
    # the rm flags include both -r/-R and -f and the target starts with
    # `/`, `~`, or `$HOME`/`${HOME}`. Single-file rm is fine.
    (
        re.compile(
            r"\brm\s+(?:-[a-zA-Z]*[rR][a-zA-Z]*[fF]|-[a-zA-Z]*[fF][a-zA-Z]*[rR])"
            r"\b\s+(?:/(?!tmp\b|var/tmp\b)|~|\$HOME|\$\{HOME\})",
        ),
        "rm -rf on root/home/system path",
    ),

    # dd writing to a device node.
    (
        re.compile(r"\bdd\s+(?:[\w=/.]+\s+)*of=/dev/", re.IGNORECASE),
        "dd of=/dev/* (raw disk write)",
    ),

    # mkfs anywhere.
    (re.compile(r"\bmkfs(?:\.[a-z0-9]+)?\b", re.IGNORECASE), "mkfs (filesystem create)"),

    # chmod -R 777 / 000 on a root path.
    (
        re.compile(
            r"\bchmod\s+-R\s+(?:777|000)\s+(?:/(?!tmp\b|var/tmp\b)|~|\$HOME)",
            re.IGNORECASE,
        ),
        "chmod -R 777/000 on root/home",
    ),

    # git push --force(-with-lease) to main/master/release/* — destructive
    # to shared history. Local force-push to a feature branch is fine.
    (
        re.compile(
            r"\bgit\s+push\s+(?:[\w./-]+\s+)*"
            r"(?:--force(?:-with-lease)?|-f)\b"
            r"\s+\S+\s+(?:main|master|release/\S+|prod\S*)\b",
            re.IGNORECASE,
        ),
        "git push --force to main/master/release",
    ),

    # Piping curl/wget output directly into a shell interpreter.
    (
        re.compile(
            r"\b(?:curl|wget|fetch)\b[^|;&]*\|\s*(?:sudo\s+)?(?:bash|sh|zsh|fish|ksh)\b",
            re.IGNORECASE,
        ),
        "curl/wget piped to shell interpreter",
    ),

    # Raw write to a disk device node.
    (re.compile(r">\s*/dev/(?:sd[a-z]\d?|nvme\d+n\d+(?:p\d+)?|hd[a-z]\d?)\b"), "write to disk device"),

    # Fork bomb.
    (re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"), "fork bomb"),

    # `> /etc/...` or system config overwrites via redirect.
    (re.compile(r">\s*/etc/(?!tmp/)"), "redirect to /etc"),
]


# Labels the aggregator does not enforce at dispatch. `\bsudo\b` matches the
# word inside grep/echo text, and the paths that never installed this hook
# (autonomy.run_task, run_prompt_on_primary) ran seven such commands harmlessly
# before 2026-09-14 — while sudo itself needs a password on this host, so the
# rule protects nothing there. The hook keeps it for the turns that had it.
_HOOK_ONLY_LABELS = frozenset({"sudo"})


def check_bash_command(command: str, cwd: str | None = None, *,
                       at_dispatch: bool = False) -> tuple[str, str] | None:
    """Return (label, excerpt) if `command` matches a hard-deny pattern,
    else None.

    Two checks, one definition. The regex table above catches the
    catastrophic-anywhere shapes; `protected_paths` parses the command and
    refuses a delete, move or `git clean` that takes out the vault, the lloyd
    tree or $HOME wholesale — the spellings the regex table let through on
    2026-09-10 and 2026-09-12. `cwd` is where the command starts; `None`
    means the aggregator's own directory.

    This is the only definition: the harness PreToolUse hook calls it, and so
    does the aggregator's `call_tool` for every Bash dispatch (`at_dispatch`),
    whatever hooks the caller installed (`agent_mcp/main.py`). Until then the
    autonomy and direct worker paths installed no safety hook at all.
    """
    if not command or not isinstance(command, str):
        return None
    for pattern, label in _HARD_DENY_PATTERNS:
        if at_dispatch and label in _HOOK_ONLY_LABELS:
            continue
        m = pattern.search(command)
        if m:
            excerpt = m.group(0)
            if len(excerpt) > 80:
                excerpt = excerpt[:80] + "..."
            return (label, excerpt)
    from app.harness.protected_paths import check_protected_delete
    why = check_protected_delete(command, cwd)
    if why:
        excerpt = command.strip().splitlines()[0]
        if len(excerpt) > 80:
            excerpt = excerpt[:80] + "..."
        return (f"destructive operation on {why}", excerpt)
    return None


# ---------------------------------------------------------------------------
# Hook callback + installer
# ---------------------------------------------------------------------------


async def _safety_pretool_cb(
    input_data: dict[str, Any], _tool_use_id: str | None, _ctx: Any,
) -> dict[str, Any]:
    """Default PreToolUse hook — hard-deny on catastrophic Bash patterns.

    Non-Bash tools pass through. Bash commands without a matching pattern
    pass through. Matched patterns deny with a user-readable reason.
    """
    tool_name = input_data.get("tool_name", "")
    if tool_name != "Bash":
        return {}
    tool_input = input_data.get("tool_input") or {}
    command = ""
    cwd = None
    if isinstance(tool_input, dict):
        command = tool_input.get("command") or ""
        cwd = tool_input.get("cwd") or None
    match = check_bash_command(command, cwd if isinstance(cwd, str) else None)
    if match is None:
        return {}
    label, excerpt = match
    reason = f"harness safety: blocked {label!r} on {excerpt!r}"
    logger.warning(
        "[harness.safety] hard-deny session=%s pattern=%s excerpt=%r",
        input_data.get("session_id"), label, excerpt,
    )
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def install_default_safety_hook(hooks: HookRegistry) -> None:
    """Install the default destructive-Bash deny hook on a HookRegistry.

    Call this for every primary turn (IV-on or IV-off). Idempotent in
    practice because each turn builds a fresh `HookRegistry`.
    """
    hooks.add_pre_tool_use(None, _safety_pretool_cb)
