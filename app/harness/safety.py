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
from app.harness.outbound_content import install_outbound_content_gate

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

    # The desktop lease is Alan's to grant (app/desktop_lease.py). Lloyd holds
    # Bash, so the human-facing route and the file behind it are refused here:
    # a request to the route, a write to the file, or a Python call into the
    # module that writes it.
    (re.compile(r"\b(curl|wget|xh|http|httpie)\b[^\n]*/api/desktop/lease"),
     "grant the desktop lease"),
    (re.compile(r"(>|\btee\b|\bcp\b|\bmv\b|\bln\b|\binstall\b|\brsync\b)[^\n]*"
                r"desktop/lease\.json"), "write the desktop lease"),
    (re.compile(r"desktop_lease\s*\.\s*(grant|revoke|_store|note_seat)\b"),
     "grant the desktop lease"),

    # The retained desktop frame is Alan's last screen: the JPEG plus the
    # element names written on it, and the gate in front of that route is only
    # the peer address (#1418). Asking the route for it is refused the way the
    # lease above is; naming the path in a grep, an edit or a review note is
    # not a request, and is not denied.
    (re.compile(r"\b(curl|wget|xh|http|httpie)\b[^\n]*/api/desktop/(?:frame|state)"),
     "read the desktop frame mirror"),
]


# Labels the aggregator does not enforce at dispatch. `\bsudo\b` matches the
# word inside grep/echo text, and the paths that never installed this hook
# (autonomy.run_task, run_prompt_on_primary) ran seven such commands harmlessly
# before 2026-09-14 — while sudo itself needs a password on this host, so the
# rule protects nothing there. The hook keeps it for the turns that had it.
_HOOK_ONLY_LABELS = frozenset({"sudo"})


# ---------------------------------------------------------------------------
# Desktop deny-set for tool arguments — scoped to the fields that can arrive
# ---------------------------------------------------------------------------

# What must not be reachable from a tool call: the human-only lease route and
# the file behind it, and the two routes that read or feed the retained frame
# (#1418).
DESKTOP_DENIED_ROUTES: tuple[str, ...] = (
    "/api/desktop/lease",
    "/api/desktop/frame",
    "/api/desktop/state",
)
DESKTOP_DENIED_PATHS: tuple[str, ...] = ("desktop/lease.json",)

# The only argument fields that can put a request on a route or bytes at a
# path: `url` on the request tools, the path fields on the file tools. Scoping
# the match to them is the whole point of this helper. The aggregator used to
# scan the serialized arguments blob for the strings above, so a backlog item
# or a vault note whose body merely *described* the lease route or the lease
# file was denied as if it were an attempt — including the write that
# documented this defect (#1418 clause 4).
DESKTOP_ROUTE_FIELDS: tuple[str, ...] = ("url", "uri", "href", "endpoint")
DESKTOP_PATH_FIELDS: tuple[str, ...] = ("file_path", "path", "paths",
                                        "filename", "destination")

_LEASE_REFUSAL = ("the desktop lease is granted by Alan from Mission Control, "
                  "never by a tool call")
_FRAME_REFUSAL = ("the desktop frame mirror retains Alan's last screen capture "
                  "— a tool call may neither read it nor publish a frame to it "
                  "(#1418)")

# Lease needles first: a call naming the lease keeps the refusal the lease
# tests already pin.
_DESKTOP_NEEDLES: tuple[tuple[str, str], ...] = (
    ("/api/desktop/lease", _LEASE_REFUSAL),
    ("desktop/lease.json", _LEASE_REFUSAL),
    ("/api/desktop/frame", _FRAME_REFUSAL),
    ("/api/desktop/state", _FRAME_REFUSAL),
)


def _string_field_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, str)]
    return []


def desktop_refusal(arguments: Any) -> str | None:
    """Return the reason to refuse `arguments`, or None if they reach nothing.

    Matches only :data:`DESKTOP_ROUTE_FIELDS` and :data:`DESKTOP_PATH_FIELDS`,
    so an `http_request` carrying the lease or frame URL and a `Write`/`Edit`
    carrying the lease path are still refused, while prose in any other field —
    a note describing the route — is not.
    """
    if not isinstance(arguments, dict):
        return None
    for field in DESKTOP_ROUTE_FIELDS + DESKTOP_PATH_FIELDS:
        for value in _string_field_values(arguments.get(field)):
            for needle, reason in _DESKTOP_NEEDLES:
                if needle in value:
                    return reason
    return None


def check_bash_command(command: str, cwd: str | None = None, *,
                       at_dispatch: bool = False, session_id: str | None = None,
                       parent_of=None) -> tuple[str, str] | None:
    """Return (label, excerpt) if `command` matches a hard-deny pattern,
    else None.

    Four checks, one definition. The fourth, `service_control`, needs the
    session: a background session (worker, autonomy, bench — by id shape,
    a `task:*` subagent by its parent through `parent_of`) may not restart
    or stop an engine or a service; a chat session may. Callers that do not
    know the session pass none and that check is skipped. The regex table above catches the
    catastrophic-anywhere shapes; `protected_paths` parses the command and
    refuses a delete, move or `git clean` that takes out the vault, the lloyd
    tree or $HOME wholesale — the spellings the regex table let through on
    2026-09-10 and 2026-09-12; `sync_registration` refuses anything that
    changes the Obsidian Sync registration, which Lloyd deleted twice on
    2026-09-14. `cwd` is where the command starts; `None` means the
    aggregator's own directory.

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
    from app.harness.sync_registration import check_sync_registration
    why = check_sync_registration(command, cwd)
    if why:
        excerpt = command.strip().splitlines()[0]
        if len(excerpt) > 80:
            excerpt = excerpt[:80] + "..."
        return (f"Obsidian Sync registration: {why}", excerpt)
    from app.harness.service_control import check_service_control
    why = check_service_control(command, session_id, parent_of=parent_of)
    if why:
        excerpt = command.strip().splitlines()[0]
        if len(excerpt) > 80:
            excerpt = excerpt[:80] + "..."
        return (f"service control: {why}", excerpt)
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
    match = check_bash_command(command, cwd if isinstance(cwd, str) else None,
                               session_id=str(input_data.get("session_id") or "") or None)
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

    Also arms the outbound content gate (#1136) on the same registry, because
    this is the one installer every production turn already calls and the gate
    is a sibling guard, not a fourth convention: Bash shape and payload
    content are the two halves of "should this call happen", and a path that
    installs one and not the other is what #869 is about. The gate takes no
    scope here — it falls back to `policy.current_scope`, which the worker
    pool and the autonomy runner bind per job, and it is armed explicitly
    with a scope at the two dispatch paths that build a registry without this
    floor (`autonomy.run_task`, `workers/sources/_common.py`).
    """
    hooks.add_pre_tool_use(None, _safety_pretool_cb)
    install_outbound_content_gate(hooks)
