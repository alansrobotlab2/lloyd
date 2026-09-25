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
# Durable-external pattern set — the reversibility TIER axis, not the deny axis
# ---------------------------------------------------------------------------
#
# `_HARD_DENY_PATTERNS` above answers one question: must this command never run?
# This table answers a different one, and the difference is the whole design: is
# this command's effect *outside this machine, or in front of someone else, after
# the fact*? Its answer is a reversibility tier, consumed by
# `app/harness/policy.py`'s grant gate — never a denial. A tier-2 command runs
# from an unattended scope when a human minted a live grant for it (#534), and is
# refused with the exact grant to mint when nobody did. The catastrophic table
# above has no such escape hatch and must never gain one.
#
# Why they cannot be one table: every shape here has an obvious legitimate use —
# `git push` is how work ships, a service restart is how a deploy takes effect —
# which is exactly why the hard-deny rule ("only ops with no plausible legitimate
# agent use case") excludes them. And why the tier ladder cannot answer it by
# tool NAME: one name, `Bash`, covers `ls` and `dd` in the same breath, so a
# name-level tier would deny every worker in the fleet. The command string is the
# only place the distinction exists, which is why the table is here, next to the
# matcher that already reads command strings, and not in the ladder.
#
# Matching is per *segment* and anchored to the segment's program word, after
# quoted arguments are scrubbed (`_scrubbed_segments`). Both halves are
# deliberate: anchoring is what keeps `grep -rn 'git push' tests/` — a command
# that only *talks about* a push, which is what most of the corpus matching these
# strings actually is — at tier 1, and scrubbing is what keeps a heredoc full of
# other people's command strings from tiering the turn that carried it. The cost
# is that `bash -c 'git push origin main'` also reads tier 1: this axis catches
# the ordinary shape of a durable action, not a determined evasion. That is the
# same property the hard-deny table has, and for the same reason — authority, not
# sandboxing.
#
# Each entry is `(compiled_regex, label)`, the same shape as the hard-deny table,
# and the label is user-visible (it appears in the grant denial's reason). Adding
# an entry here is the whole extension point: `bash_command_tier` below is the
# only consumer, and it is what `policy.tool_tier` asks, so a new shape changes
# the tier of a Bash call with no change to `app/harness/policy.py` — which is the
# property `tests/test_grant_bash_tier.py` pins.

#: Program-word anchor: leading whitespace, then any number of `NAME=value`
#: assignments (`LANG=C git push …`). Deliberately nothing else: a wrapper
#: (`timeout 5 git push`) resolves to tier 1, honestly.
_DURABLE_LEAD = r"^\s*(?:[\w.+-]+=\S*\s+)*"

#: Read-only `supervisorctl`/`systemctl` verbs are absent on purpose —
#: `supervisorctl -c …/supervisord.conf status` is the health probe the runbooks
#: tell every unattended turn to run, and tiering it would deny the fleet's own
#: liveness check. The state-changing verb set mirrors `_SUPERVISOR_VERBS` /
#: `_SYSTEMCTL_VERBS` in `app/harness/service_control.py`, which is the *hard-deny*
#: axis for the same shapes: a background session may not restart a service at
#: all, and an attended-or-granted one now pays a grant for it.
_DURABLE_EXTERNAL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # A supervisor program changing state: `supervisorctl [-c <conf>] restart <unit>`.
    (
        re.compile(
            _DURABLE_LEAD
            + r"(?:[\w./-]*supervisorctl)\b(?:\s+\S+)*?"
            r"\s+(?:restart|start|stop|signal|reload|update|remove|add|clear)\b",
        ),
        "supervisorctl state change (restart/stop a service)",
    ),

    # The same thing through systemd (`agent-supervisord.service` and
    # `agent-obsidian-sync` are real units on this box).
    (
        re.compile(
            _DURABLE_LEAD
            + r"(?:[\w./-]*systemctl)\b(?:\s+\S+)*?"
            r"\s+(?:restart|start|stop|kill|reload|reload-or-restart|try-restart"
            r"|daemon-reload|isolate|reset-failed)\b",
        ),
        "systemctl state change (restart/stop a unit)",
    ),

    # The restart form this box actually uses: restarts go through the promoter's
    # own CLI, which shells out to supervisorctl itself
    # (`scripts/automod/round.py`), so a tier keyed on the literal word
    # `supervisorctl` would miss the sanctioned path entirely.
    (
        re.compile(
            _DURABLE_LEAD
            + r"(?:(?:[\w./-]*python[\d.]*\b(?:\s+\S+)*?\s+"
              r"scripts[/\.]automod[/\.]round(?:\.py)?"
              r"|[\w./-]*automod[/\.]round\.py)\b"
              r"(?:\s+\S+)*?\s+(?:restart|recover)\b)",
        ),
        "automod round restart (restarts a live service)",
    ),

    # Publishing a branch: the counterparty is every other clone of the remote.
    # `push` must be the subcommand, not an argument of one — `git grep push` is
    # a search. The global-flag run allows `git -C <path> push`, which is how
    # this repo's own scripts address the tree.
    (
        re.compile(
            _DURABLE_LEAD
            + r"git\b(?:\s+(?:-C\s+\S+|-c\s+\S+|-\S+))*\s+push\b",
        ),
        "git push (publishes to a remote other people clone)",
    ),

    # An HTTP *write* aimed at a host that is not this machine. Two order-free
    # lookaheads: a write verb anywhere in the segment, and a non-loopback URL
    # anywhere in it. Loopback stays tier 1 because every health probe, engine
    # call and queue poke on this box is a loopback POST — measured over the
    # retained transcripts, a deny keyed on "any POST" would have caught real
    # fleet traffic while the durable kind was at zero.
    (
        re.compile(
            _DURABLE_LEAD
            + r"(?:[\w./-]*)(?:curl|wget|xh|httpie)\b"
            + r"(?=[^|;&]*?\s(?:-X\s*(?:POST|PUT|PATCH|DELETE)"
              r"|--request[=\s]+(?:POST|PUT|PATCH|DELETE)"
              r"|-d\b|--data(-raw)?\b|-F\b|--form(-multipart)?\b"
              r"|-T\b|--upload-file\b))"
            + r"(?=[^|;&]*?https?://"
              r"(?!127\.0\.0\.1|localhost|\[?::1\]?|0\.0\.0\.0))",
            re.IGNORECASE,
        ),
        "HTTP write to a non-loopback host (a POST to someone else's API)",
    ),
]

#: Quoted spans become a single space before matching: an argument in quotes is
#: data the command carries, not the shape of the command.
_DURABLE_QUOTED_SPAN_RE = re.compile(r"'[^']*'|\"[^\"\\]*(?:\\.[^\"\\]*)*\"")
#: Shell segment boundaries: `;`, `|`, `||`, `&&`, `&`, newlines, subshell parens.
_DURABLE_SEGMENT_RE = re.compile(r"[\n;|&()]+")


def _scrubbed_segments(command: str) -> list[str]:
    """Quote-scrubbed `command`, split into shell segments.

    Order matters: scrubbing first is what stops a quoted `&&` or a command
    string inside an argument from being read as a segment of its own.
    """
    unquoted = _DURABLE_QUOTED_SPAN_RE.sub(" ", command)
    return [s for s in _DURABLE_SEGMENT_RE.split(unquoted) if s.strip()]


def match_durable_external(command: Any) -> str | None:
    """Label of the durable-external shape this command carries, else None.

    The label exists because a grant denial has to name what it is asking
    permission for: `check_grants` puts this string in the reason a human reads,
    the way the hard-deny table's label reaches a safety refusal.
    """
    if not command or not isinstance(command, str):
        return None
    for segment in _scrubbed_segments(command):
        for pattern, label in _DURABLE_EXTERNAL_PATTERNS:
            if pattern.match(segment):
                return label
    return None


def bash_command_tier(command: Any) -> int:
    """Reversibility tier of one Bash command string: 2 durable-external, else 1.

    The one function `app/harness/policy.py` asks. It answers 1 rather than
    `policy`'s constant because a tier number is the ladder's vocabulary and
    `policy.py` owns the ladder; this module owns the shape. Anything the table
    does not name is tier 1 — the same deliberate empty case the name-level
    ladder uses, because a gate that guesses wrong denies real work and a worker
    that cannot run `ls` burns a run.
    """
    return 2 if match_durable_external(command) else 1


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
    hooks.add_pre_tool_use(None, _safety_pretool_cb, fail_closed=True)
    install_outbound_content_gate(hooks)
