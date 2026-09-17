"""A background turn may not restart or stop an engine or a service.

On 2026-09-17 at 00:37Z an autocode turn, continued by hand after its round
had been reaped, ran `round restart --only agent-llm-primary` from Bash to
fix a grader that had returned empty output. The restart was the right
diagnosis and the wrong actor: it took the primary down for five minutes,
killed every turn in flight including the one that issued it, and left the
round it had just reopened with no owner. The loop then read that round as
open until a human aborted it.

Service control belongs to a person or to the promoter, both of which go
through the guardian's pause lease. A worker, autonomy or bench session gets
a refusal here instead, at both enforcement points `safety.check_bash_command`
has (the harness hook and the aggregator's dispatch), keyed on the session id
the way `_tool_sandbox` keys its read-only rule: a four-part id
(`20260916_165238_autocode_1b1a`) is a background run, a `task:*` id is
whatever its parent is, and a chat's three-part id is never refused.

Read-only forms stay allowed — `supervisorctl status`, `systemctl status`,
`round status` — so a turn can still see what is running.
"""

from __future__ import annotations

import re
from typing import Callable

# The verbs that change a service's state. `status`, `tail`, `pid`, `maintail`
# and the like are not here on purpose.
_SUPERVISOR_VERBS = frozenset({"restart", "stop", "start", "signal", "shutdown", "reload",
                               "update", "remove", "add", "clear"})
_SYSTEMCTL_VERBS = frozenset({"restart", "stop", "start", "kill", "reload", "reload-or-restart",
                              "try-restart", "daemon-reload", "isolate", "reset-failed"})
_ROUND_VERBS = frozenset({"restart", "recover"})
# `round land` in the foreground of a worker turn. Not a service verb, and
# refused for a different reason: the landing waits for the backend to go
# idle, and the turn that ran it IS what keeps it busy, under a Bash timeout
# that then kills it. On 2026-09-17 #1179's turn ran `timeout 120 … round
# land`; the promoter died at 120 s waiting for that very turn, the reaper
# closed a round with nine green rungs, and the item read as `spent`.
# `automod_land` spawns it detached and the turn ends.
_ROUND_LAND_VERB = "land"
_ENGINE_PROCESS_WORDS = ("vllm", "llama-server", "llama_server", "uvicorn", "server.py",
                         "supervisord", "lloyd-mcp", "agent_mcp")
_LAUNCHERS = frozenset({"flash-next-run-arm.sh", "start-qwen38-flash-next.sh"})
_SHELLS = frozenset({"bash", "sh", "zsh", "dash"})
_PYTHONS = frozenset({"python", "python3"})
_MAX_DEPTH = 3
_INTERPRETER_FORM = re.compile(
    r"\b(supervisorctl|systemctl|scripts\.automod\.round|round\.py)\b\W+(?:[\w.:/-]+\W+){0,6}?"
    r"(restart|stop|start|signal|shutdown|reload|update|kill|daemon-reload|recover)\b")


def _segment_label(argv: list[str], depth: int) -> str | None:
    """The service-control form one simple command is, or None. Parsed with
    `protected_paths`' tokeniser so a quoted mention (`grep 'supervisorctl
    restart' CLAUDE.md`) is an argument to grep, not a command."""
    import os
    from app.harness.protected_paths import _strip_wrappers
    argv, _ = _strip_wrappers(list(argv))
    if not argv:
        return None
    base = os.path.basename(argv[0])
    if base in _SHELLS:
        if "-c" in argv[1:]:
            i = argv.index("-c", 1)
            inner = argv[i + 1] if i + 1 < len(argv) else ""
            return find_service_control(inner, _depth=depth + 1)
        return _segment_label(argv[1:], depth) if len(argv) > 1 else None
    if base == "supervisorctl":
        verb = next((a for a in argv[1:] if a in _SUPERVISOR_VERBS), None)
        return f"supervisorctl {verb}" if verb else None
    if base == "systemctl":
        verb = next((a for a in argv[1:] if a in _SYSTEMCTL_VERBS), None)
        return f"systemctl {verb}" if verb else None
    if base in _PYTHONS or base.startswith("python3.") or base.endswith("round.py"):
        rest = argv[1:] if not base.endswith("round.py") else argv
        if "-c" in rest:
            i = rest.index("-c")
            return find_service_control(rest[i + 1] if i + 1 < len(rest) else "", _depth=depth + 1)
        is_round = base.endswith("round.py") or any(
            a == "scripts.automod.round" or a.endswith("/round.py") for a in rest)
        if is_round:
            verb = next((a for a in rest if a in _ROUND_VERBS), None)
            if verb is None and _ROUND_LAND_VERB in rest and "--dry-run" not in rest:
                verb = _ROUND_LAND_VERB
            return f"round {verb}" if verb else None
        return None
    if base in ("pkill", "killall"):
        for a in argv[1:]:
            if any(w in a for w in _ENGINE_PROCESS_WORDS):
                return f"{base} {a[:40]}"
        return None
    if base in _LAUNCHERS:
        return f"engine launcher {base}"
    return None


def is_background_session_id(session_id: str) -> bool:
    """The id-shape rule `sessions_io` uses for its listings: four parts,
    an 8-digit date and a 6-digit time. A chat id has three parts."""
    from app.sessions_io import is_background_session_name
    return is_background_session_name(str(session_id or ""))


def is_background_session(session_id: str, *,
                          parent_of: Callable[[str], str | None] | None = None) -> bool:
    """`is_background_session_id`, plus a `task:*` subagent classified by its
    parent (`parent_of` resolves it; without a resolver a subagent is not
    refused — the parent's own Bash still is)."""
    sid = str(session_id or "")
    if not sid:
        return False
    if sid.startswith("task:"):
        if parent_of is None:
            return False
        parent = parent_of(sid)
        return bool(parent and parent != sid and is_background_session(parent, parent_of=parent_of))
    return is_background_session_id(sid)


def find_service_control(command: str, *, _depth: int = 0) -> str | None:
    """The first service-control form in `command`, as a short label, or None.

    Every simple command in the string is checked (pipes, `&&`, `;`, `&`,
    subshells), wrappers are peeled (`sudo`, `nohup`, `timeout`, env
    assignments), and a `bash -c` / `python -c` one-liner is scanned inside,
    three levels deep."""
    text = str(command or "")
    if not text or _depth > _MAX_DEPTH:
        return None
    from app.harness.protected_paths import _segments, _tokens
    for argv in _segments(_tokens(text)):
        label = _segment_label(argv, _depth)
        if label:
            return label
    if _depth > 0:
        # Inside an interpreter one-liner the command is data, not argv:
        # `subprocess.run(["supervisorctl", "restart", ...])`. A word-distance
        # match on the program name and a verb is the best reading there.
        m = _INTERPRETER_FORM.search(text)
        if m:
            return f"{m.group(1)} {m.group(2)} (in a one-liner)"
    return None


def check_service_control(command: str, session_id: str | None, *,
                          parent_of: Callable[[str], str | None] | None = None) -> str | None:
    """Reason to refuse `command` for `session_id`, or None.

    Only a background session is refused, and only for a state-changing
    service verb. A chat session is never refused here: a person restarting
    the stack from Mission Control is the intended operator.
    """
    if not session_id or not is_background_session(session_id, parent_of=parent_of):
        return None
    label = find_service_control(command)
    if not label:
        return None
    if label == f"round {_ROUND_LAND_VERB}":
        return (f"round land from background session {session_id}: a landing run in the "
                f"foreground of your own turn waits for that turn to end and is killed by "
                f"the Bash timeout first. Call the `automod_land` tool — it spawns the "
                f"landing detached — and END YOUR TURN")
    return (f"{label} from background session {session_id}: a worker turn may not restart "
            f"or stop an engine or a service (it kills every turn in flight, its own "
            f"included); report the need on the item and leave the restart to a person "
            f"or the promoter")
