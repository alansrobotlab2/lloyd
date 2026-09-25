"""Programmatic tool calling, `lloyd_rpc` — the policy half (review 2026-09-24, P9).

A model that needs twenty backlog items or ten files read spends one engine
round trip per call, and every round trip re-sends the whole conversation. P9
lets a `Bash` command make those calls itself: the stdlib client
`agent-services/rpc/lloyd_rpc.py` (`agent-services/bin/lloyd_rpc` on the command
line) POSTs `tools/call` to the aggregator from inside the shell, and only what
the script prints comes back as the Bash result.

The rules, and where each is enforced:

* **v1 is read-only.** A nested call must name a tool the server annotates
  `readOnlyHint` (`agent_mcp.annotations.READ_ONLY`). `harness.rpc.allow_mutating`
  exists and stays closed: a nested call skips every PreToolUse hook the
  harness owns — the grant gate, Inner Voice, the repetition guard — and the
  read-only set is the one class for which none of those decides anything.
  Opening it waits on grants travelling in `_meta`.
* **The turn's deny list travels with the Bash call.** The loop stamps
  `lloyd/rpc_deny` on Bash calls only: this iteration's disallowed set (plan
  mode, surface, the caller's bans) plus `FIXED_DENY`. The aggregator records
  it when the Bash call spawns, and admits nested calls against THAT copy, not
  against whatever a script sends back (`agent_mcp/_rpc.py`).
* **No recursion.** `Bash`, `Task` and `ToolSearch` are denied, so depth is 1.
* **Refused above the effect ledger** in `agent_mcp.main.call_tool`, so a call
  that was never allowed is never ledgered `unknown`.
* **A sandboxed (bench/eval) session gets no env at all**, and the bwrap
  sandbox unsets the credential variables and covers the token file.

Stdlib-only and import-light: the loop, the aggregator, `builtin_bash` and
`prompt_builder` all read it, and none of them may pull another's imports in.
`enabled()` reads config lazily and answers False on any failure, which is
today's behaviour: the feature ships off.
"""
from __future__ import annotations

import contextvars
import fnmatch
import hashlib
import json
import logging
from typing import Any, Iterable

logger = logging.getLogger("lloyd-harness-rpc")

# ── `_meta` keys. Must match agent_mcp/_rpc.py and the client. ──────────────
#: Stamped by the loop on a Bash call: the names a nested call may not use.
META_RPC_DENY = "lloyd/rpc_deny"
#: Sent by the client on a nested call: the Bash call it runs inside.
META_RPC_PARENT_CALL_ID = "lloyd/rpc_parent_call_id"
#: Sent by the client: always 1 in v1 (nothing nested can spawn a shell).
META_RPC_DEPTH = "lloyd/rpc_depth"

# ── Environment handed to the Bash child. Must match the client. ────────────
ENV_URL = "LLOYD_RPC_URL"
ENV_TOKEN_FILE = "LLOYD_RPC_TOKEN_FILE"
ENV_SESSION_ID = "LLOYD_SESSION_ID"
ENV_TURN_ID = "LLOYD_TURN_ID"
ENV_PARENT_CALL_ID = "LLOYD_PARENT_CALL_ID"
ENV_EFFECT_SCOPE = "LLOYD_EFFECT_SCOPE"
ENV_SURFACE = "LLOYD_SURFACE"
ENV_DENY = "LLOYD_RPC_DENY"
ENV_DEADLINE = "LLOYD_RPC_DEADLINE"
ENV_DEPTH = "LLOYD_RPC_DEPTH"

#: Every variable above, for the sandbox to unset and the tests to check.
ENV_NAMES: tuple[str, ...] = (
    ENV_URL, ENV_TOKEN_FILE, ENV_SESSION_ID, ENV_TURN_ID, ENV_PARENT_CALL_ID,
    ENV_EFFECT_SCOPE, ENV_SURFACE, ENV_DENY, ENV_DEADLINE, ENV_DEPTH,
)

#: Denied to every nested call, whatever the turn allows. Patterns are
#: `fnmatch` shapes over the bare tool name. `Bash`/`Task`/`ToolSearch` would
#: make depth > 1 (or change the catalog mid-turn); `automod_*` drive the
#: self-modification loop; `desktop_*` look at Alan's real screen and are for a
#: person's chat turn, not for a script; `_*` are the harness's own internals.
FIXED_DENY: tuple[str, ...] = (
    "Bash", "Task", "ToolSearch", "automod_*", "desktop_*", "_*",
)

#: A client-side call needs at least this long before the Bash deadline.
MIN_REMAINING_S = 1.0
#: The deadline is the Bash call's own timeout less this, so a nested call
#: never outlives the shell it runs in.
DEADLINE_MARGIN_S = 5.0


def _block() -> dict:
    try:
        from app.config import CONFIG

        return ((CONFIG or {}).get("harness") or {}).get("rpc") or {}
    except Exception:  # noqa: BLE001 — unreadable config is "off"
        return {}


#: An eval arm's switch (`eval/run_rpc_eval.py`): set, it wins over config in
#: this context — the prompt paragraph and the loop's stamp both read
#: `enabled()`, and the aggregator serves exactly the Bash calls that carry the
#: stamp, so one process can run an `on` arm beside an `off` one.
_override: contextvars.ContextVar[bool | None] = contextvars.ContextVar(
    "lloyd_rpc_enabled_override", default=None)


def override(value: bool | None) -> contextvars.Token:
    return _override.set(value)


def enabled() -> bool:
    """`harness.rpc.enabled`; False by default and on any failure.

    Read by the side that owns the turn — the loop (whether to stamp a Bash
    call) and `prompt_builder` (whether to say so). The aggregator does not
    read it: it serves a Bash call that carries `lloyd/rpc_deny` and no other,
    so the turn's decision is the one decision.
    """
    forced = _override.get()
    if forced is not None:
        return forced
    return _block().get("enabled", False) is True


def allow_mutating() -> bool:
    """Always False in v1, whatever `harness.rpc.allow_mutating` says.

    The key is read so a `true` is noticed and logged rather than silently
    honoured: a mutating nested call would skip the grant gate, which is a
    PreToolUse hook the aggregator never sees.
    """
    if _block().get("allow_mutating", False) is True:
        logger.error("harness.rpc.allow_mutating is true, but lloyd_rpc v1 is "
                     "read-only until grants travel in _meta; staying closed")
    return False


def bare_name(name: Any) -> str:
    text = str(name or "")
    if text.startswith("mcp__"):
        rest = text[len("mcp__"):]
        if "__" in rest:
            return rest.split("__", 1)[1]
    return text


def bash_deny(disallowed: Iterable[str] | None) -> list[str]:
    """What the loop stamps on a Bash call: the turn's set plus `FIXED_DENY`.

    Sorted and de-duplicated, so the same turn state always produces the same
    `_meta` bytes.
    """
    names = {bare_name(n) for n in (disallowed or ()) if n}
    names.update(FIXED_DENY)
    return sorted(n for n in names if n)


def is_denied(name: str, deny: Iterable[str]) -> bool:
    bare = bare_name(name)
    for pattern in deny:
        pat = bare_name(pattern)
        if pat == bare or (any(c in pat for c in "*?[") and fnmatch.fnmatchcase(bare, pat)):
            return True
    return False


def refusal(name: str, deny: Iterable[str], read_only: Iterable[str]) -> str | None:
    """Why a nested call to `name` may not run, or None. Deny first, so a
    denied read-only tool is named as denied rather than as a writer."""
    bare = bare_name(name)
    denied = set(deny) | set(FIXED_DENY)
    if is_denied(bare, denied):
        return (f"{bare} is not available to lloyd_rpc in this turn (the turn's "
                "deny list, or a tool no nested call may use: Bash, Task, "
                "ToolSearch, automod_*, desktop_*)")
    if bare not in set(read_only) and not allow_mutating():
        return (f"{bare} can change state; lloyd_rpc v1 calls read-only tools "
                "only — call it directly as a tool instead")
    return None


def args_digest(arguments: Any) -> str:
    """A short stable digest of a call's arguments, for the event log."""
    try:
        text = json.dumps(arguments, sort_keys=True, default=str, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        text = repr(arguments)
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:12]


def trailer(counts: dict[str, int], errors: int, seconds: float) -> str:
    """`[lloyd_rpc: 12 calls — Read×10 Grep×2, 0 errors, 3.1 s]`, or "" for none.

    Appended to the Bash result so the model — and anyone reading the
    transcript — can see that one shell call stood for many tool calls.
    """
    total = sum(counts.values())
    if total <= 0:
        return ""
    parts = " ".join(f"{name}×{n}" for name, n in
                     sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
    return (f"[lloyd_rpc: {total} call{'s' if total != 1 else ''} — {parts}, "
            f"{errors} error{'s' if errors != 1 else ''}, {seconds:.1f} s]")
