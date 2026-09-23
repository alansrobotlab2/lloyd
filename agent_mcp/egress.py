"""#628 — destination-scoped network egress: telemetry first, then a default-deny policy.

Lloyd had no destination concept anywhere on its outbound path. `http_fetch` and
`http_request` checked only whether a host was *private* (`http_tools.py:395`,
`:494`), `browser.py:210` mirrored that, `http_search` never received a host at
all, and nothing recorded where a call went: the live queue DB held
`authority_grants grant_dispatch queue runs tool_effects watermarks` and no
destination, the dashboard's eleven sections held no network figure. Every scope
— chat, worker, autonomy task, bench trial — could reach any public host,
permanently. That is the gap #590's EchoLeak fixture describes as "exfiltration
through a legitimate egress path, no dangerous-looking tool involved" and #582's
Risks section names as "a read-only mount stops writes, not exfiltration".

Two layers, deliberately sequenced, because the second one without the first is
guessing at a blast radius:

1. **Telemetry (on by default).** Every call through the four egress tools
   writes one row: the destination as `scheme://host`, the calling scope, and
   the decision. That table is the reviewable inventory the item's step 2 asks
   for; nobody can pick a policy shape without it.
2. **Policy (flag `harness.egress_policy.enforce`, default off).** Default-deny
   against `harness.egress_policy.allow` plus live destination grants. A denial
   renders the exact row that would authorize the call, so the human's answer is
   a batched mint, not a popup on a live run.

What this is NOT, stated here because presenting it wrongly is the failure the
source talk criticises ("either you're approving a bunch of stuff or you're
hoping your auto approval flow is configured properly"): **this is not a
sandbox.** It covers the paths the agent is *given* — which is where an
injection actually lands — and not a subprocess that opens its own socket.
Env-var proxying is advisory. The substrate half (network namespace +
proxy-only route) is #582's step-2 spike and lands behind this, not inside it.

Rules that are not obvious
--------------------------
* **The private/loopback floor is never reopened by the allow-list.** An entry
  naming `127.0.0.1` denies: the floor is evaluated first and a grant cannot
  cover a destination the floor refuses. `http_request` keeps its deliberate
  loopback allowance (it is how the box drives its own services) and that
  allowance is not destination-gated either — refusing loopback there would
  break the sanctioned local-service lane, which is not what #628 asked for.
* **`browser_navigate`'s loopback exemption is a non-enforcing convenience.**
  The browser tab reaches Lloyd's own UI on `127.0.0.1:8080` by design
  (`navigate_from_ui`, pinned by `tests/test_browser_panel.py`). Under an
  enforced policy that exemption closes for the *agent's* tool lane, and only
  there: the URL bar is a human typing, and a person is never refused.
* **An attended scope is never refused.** Enforcement that stops a human from
  fetching the page they asked for would be answered by turning enforcement
  off, so it enforces nothing. An attended call to an unknown destination
  still records `grant-required` — that is what tells a reviewer what
  enforcement *would* have denied — but it proceeds.
* **Attended is the session's platform, not the presence of a scope.**
  `run_prompt_on_primary` turns carry no session and no effect scope at all;
  treating an empty scope as attended would exempt the most unattended class on
  the box. Same three-layer reading `agent_mcp/builtin_grants.py` uses for the
  mint path.
* **The grant scope is derived from the queue row, not guessed from the effect
  scope.** `effect_scope` is `item:<source>:<item id>` — the item id is the
  unit a retry re-runs, which is right for idempotency and useless for
  authority: a grant keyed to it dies with one queue row. `grant_scope` is
  `autonomy-task:<task_id>` or `worker:<source>` and survives every future run
  of the same job. The aggregator gets it by reading the claimed row's payload
  out of the same WAL file it already writes `tool_effects` to, and one
  definition of the mapping lives in `app.harness.policy.grant_scope_for_source`
  — the same function `workers.pool.grant_scope_for` delegates to, so the two
  processes cannot disagree about whose authority a turn is borrowing.
* **Telemetry fails open; the policy fails closed.** A ledger that cannot write
  must not brick a fetch (`_tool_effects`' rule, same reason). A grant store
  that cannot be read during enforcement is a denial, matching
  `check_grants`' "unreadable store" behaviour: a check that cannot see cannot
  allow.
* **Telemetry rows are the measurement, so they outlive the policy.** Retention
  is 90 days against the item's 7-day window; a policy decision reviewed on
  day 8 must still be able to see the week it came from.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import re
import sqlite3
import threading
import time
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("lloyd-egress")

TABLE = "egress_events"

#: The four tools this item pins. Anything else that reaches the network
#: (Bash + curl, a subprocess, a pip install) is out of this change's scope:
#: recording *those* is the Bash-namespace half (#582 step 2), not this one.
EGRESS_TOOLS = frozenset({
    "http_fetch", "http_request", "http_search", "browser_navigate",
})

#: The backend `http_search` actually queries. `ddgs` 9.14.0's ddg text engine
#: posts to `https://html.duckduckgo.com/html/` (see
#: `ddgs/engines/duckduckgo.py`); it is recorded under that literal host because
#: `http_search` takes no url argument and a destination has to be *named* for a
#: grant to cover it. A redirect to a result's own host is a separate call, made
#: by `http_fetch`, and gets its own row and its own decision.
SEARCH_BACKEND_HOST = "html.duckduckgo.com"

DECISION_ALLOW = "allow"
DECISION_DENY = "deny"
DECISION_GRANT = "grant-required"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS egress_events (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  at            TEXT NOT NULL,
  tool          TEXT NOT NULL,
  destination   TEXT NOT NULL,
  host          TEXT NOT NULL,
  decision      TEXT NOT NULL,
  scope         TEXT NOT NULL DEFAULT '',
  session_id    TEXT NOT NULL DEFAULT '',
  worker_source TEXT NOT NULL DEFAULT '',
  task_id       TEXT NOT NULL DEFAULT '',
  reason        TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS egress_events_at_idx ON egress_events(at);
CREATE INDEX IF NOT EXISTS egress_events_host_idx ON egress_events(host, at);
CREATE INDEX IF NOT EXISTS egress_events_scope_idx ON egress_events(scope, at);
"""

_DEFAULTS = {
    "telemetry": True,
    "enforce": False,
    "allow": [],
    "retention_days": 90,
}

PRUNE_INTERVAL_S = 3600.0
_last_prune = 0.0

_init_lock = threading.Lock()
_init_done: set[str] = set()


def config() -> dict:
    """`harness.egress_policy` from config.yaml, read at call time.

    Read through the dict rather than a module-level snapshot so a test can
    `monkeypatch.setitem(CONFIG, ...)` and so the aggregator — a separate
    process from the backend — reads the file for itself. Same shape as
    `agent_mcp._tool_effects.config()`.
    """
    try:
        from app.config import CONFIG
        raw = (CONFIG.get("harness") or {}).get("egress_policy") or {}
    except Exception:  # noqa: BLE001 — a config that cannot be read is defaults
        raw = {}
    out = dict(_DEFAULTS)
    if isinstance(raw, dict):
        out.update({k: v for k, v in raw.items() if v is not None})
    return out


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    val = str(raw).strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    logger.warning("egress: unparseable %s=%r — keeping %s", name, raw, default)
    return default


def telemetry_on() -> bool:
    return _env_flag("LLOYD_EGRESS_TELEMETRY", bool(config().get("telemetry", True)))


def enforce_on() -> bool:
    """The enforcement flag. **Default off**: a human flips it once the
    destination table has been read and the policy shape chosen (#628's own
    post-landing decisions). Telemetry runs with it off."""
    return _env_flag("LLOYD_EGRESS_ENFORCE", bool(config().get("enforce", False)))


# ── destinations ────────────────────────────────────────────────────────────

def host_of(url: str) -> str:
    """The lowercase host a URL names, port and userinfo stripped.

    Not just `urlparse().hostname`: that returns None for `http://::1/`, because
    a bare IPv6 literal outside brackets fails the port parse, and an
    "unparsable destination" is the wrong answer for a destination that is
    plainly the machine itself. The bracketed form `https://[::1]:8080/x` does
    parse, and both must reach the private-host block with the same message.
    """
    try:
        parsed = urllib.parse.urlparse(str(url or "").strip())
    except Exception:  # noqa: BLE001 — urlparse only raises on a bad IPv6 zone
        return ""
    netloc = parsed.netloc or ""
    if parsed.scheme in ("http", "https") and not netloc:
        # `scheme://host` with something colon-shaped in it: urlparse has already
        # put the whole authority in `netloc` when it can, so an empty netloc on
        # an http URL means the host is in `path` (`http://::1/` → path '::1/').
        netloc = (parsed.path or "").split("/", 1)[0]
    if not netloc:
        return ""
    netloc = netloc.rsplit("@", 1)[-1]          # drop userinfo
    if netloc.startswith("["):                   # bracketed IPv6 literal
        return netloc[1:netloc.find("]")].lower() if "]" in netloc else ""
    head, sep, tail = netloc.partition(":")
    if sep and tail.isdigit():                   # real port
        return head.lower()
    return netloc.lower()                        # no port, or an IPv6 literal


def destination_of(url: str) -> tuple[str, str]:
    """`(scheme://host, host)` — the pair every row and every decision keys on.

    The destination is the *scheme and host only*: no port, no path, no query.
    A path-bearing destination would make the allow-list a URL-matching exercise
    nobody maintains, and a port-bearing one would let a second port on an
    approved host read as a different approval.
    """
    host = host_of(url)
    try:
        scheme = (urllib.parse.urlparse(str(url or "").strip()).scheme or "").lower()
    except Exception:  # noqa: BLE001
        scheme = ""
    if not host:
        return (str(url or "")[:200], "")
    return (f"{scheme or 'https'}://{host}", host)


# `validate_destination` and `host_matches` are `app.harness.policy`'s, not
# re-implemented here: the allow-list and the grant store must agree on what a
# destination means, and two definitions of host matching is how one of them
# stops being true. The wrapper only re-raises as ValueError, because an
# allow-list entry is config rather than a grant, and a caller reading config
# should not have to import `GrantError` to handle a bad line.
def validate_destination(text: Any) -> str:
    from app.harness.policy import GrantError, normalize_destination
    try:
        return normalize_destination(text)
    except GrantError as exc:
        raise ValueError(str(exc)) from exc


def host_matches(entry: str, host: str) -> bool:
    from app.harness.policy import destination_covers
    return destination_covers(entry, host)


# ── the private/loopback floor ──────────────────────────────────────────────

def floor_reason(host: str, *, loopback_ok: bool = False) -> str:
    """Why this host may never be fetched, or "" when it may.

    Delegates to `agent_mcp.http_tools._is_private_host` rather than restating
    the ranges: a second definition of the private space is how one of them
    stops being true, and this one's whole job is to be closed.

    `loopback_ok` is `http_request`'s sanctioned local-service allowance
    (`http_tools.py:493`). It is passed in by the caller, never inferred, so the
    floor's answer cannot drift by call site.
    """
    if not host:
        return ""
    from agent_mcp.http_tools import _is_loopback_host, _is_private_host

    if not _is_private_host(host):
        return ""
    if loopback_ok and _is_loopback_host(host):
        return ""
    return f'Blocked — private/internal host "{host}"'


# ── who is asking ───────────────────────────────────────────────────────────

def _effect_scope() -> str:
    try:
        from app.harness.policy import current_effect_scope
        return str(current_effect_scope.get() or "")
    except Exception:  # noqa: BLE001 — no harness bound means no scope
        return ""


def _split_effect(scope: str) -> tuple[str, str]:
    """`item:scheduled-task:4712` → ('scheduled-task', '4712')."""
    parts = str(scope or "").split(":")
    if len(parts) >= 3 and parts[0] == "item":
        return parts[1], parts[2]
    return "", ""


@functools.lru_cache(maxsize=2048)
def _task_id_for_item(item_id: str) -> str:
    """The `task_id` of the claimed queue row an effect scope names.

    A read of the work queue's own row, in the same WAL file this module writes
    its telemetry to, by primary key. Read at call time rather than shipped in
    `_meta`: the pool binds `current_scope` in its own task and the aggregator's
    dispatch runs outside that context — which is why `effect_scope` is a `_meta`
    key and the grant scope is not. The row is claimed and running while any
    egress call from that turn is in flight, so the payload is current.
    """
    if not item_id.isdigit():
        return ""
    conn = _connect(create=False)
    if conn is None:
        return ""
    try:
        row = conn.execute("SELECT payload_json FROM queue WHERE id=?",
                           (int(item_id),)).fetchone()
    except sqlite3.Error:
        # No `queue` table yet — a database this module created itself for the
        # telemetry table has no queue in it until the pool's own init runs, and
        # an egress call answering "no such table: queue" would be a destination
        # check that takes down a fetch over a bookkeeping lookup.
        return ""
    try:
        return str(json.loads(row["payload_json"] or "{}").get("task_id") or "")
    except Exception as exc:  # noqa: BLE001
        logger.debug("egress: queue lookup for item %s failed (%s)", item_id, exc)
        return ""
    finally:
        conn.close()


def scope_context(effect_scope: str = "") -> dict:
    """`{scope, worker_source, task_id}` for the turn making this call.

    `scope` is the **grant scope** — the thing a destination grant is keyed to,
    and the same string a denial tells a human to mint against. Derived through
    `app.harness.policy.grant_scope_for_source`, the one mapping the pool also
    uses, so `check_grants` and this module read the same authority.
    """
    from app.harness.policy import grant_scope_for_source

    effect = effect_scope or _effect_scope()
    source, item_id = _split_effect(effect)
    task_id = _task_id_for_item(item_id) if source == "scheduled-task" else ""
    payload = {"task_id": task_id} if task_id else {}
    return {
        "effect_scope": effect,
        "scope": grant_scope_for_source(source, payload) if source else "",
        "worker_source": source,
        "task_id": task_id,
    }


@functools.lru_cache(maxsize=512)
def _session_is_user(session_id: str) -> bool:
    """Is a human reading this session? Deny-list of platforms, via
    `app.sessions_io.is_user_session` — the same read the grant mint path makes.

    Cached for the process's life: a session's platform never changes, and this
    is a file read on a per-call path.
    """
    if not session_id:
        return False
    try:
        from app.sessions_io import SESSIONS_DIR, is_user_session
        meta = SESSIONS_DIR / f"{session_id}.json"
        if not meta.exists():
            return False
        return bool(is_user_session(json.loads(meta.read_text(encoding="utf-8"))))
    except Exception as exc:  # noqa: BLE001 — unreadable is not "attended"
        logger.debug("egress: session %s platform unreadable (%s)", session_id, exc)
        return False


def attended(session_id: str) -> bool:
    return _session_is_user(str(session_id or ""))


# ── the store ───────────────────────────────────────────────────────────────

def db_path() -> Path:
    """The work queue's own database, the way `agent_mcp._tool_effects.db_path()`
    takes it: the aggregator is a separate process from the backend, so it asks
    for the configured path instead of inventing one. `LLOYD_EGRESS_DB` overrides
    for tests and canaries.
    """
    raw = os.environ.get("LLOYD_EGRESS_DB")
    if raw:
        return Path(raw).expanduser()
    from workers.queue import configured_db_path
    return configured_db_path()


def _connect(create: bool = True) -> sqlite3.Connection | None:
    path = db_path()
    if not create and not path.exists():
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _ensure(conn: sqlite3.Connection) -> None:
    """Create the table once per process per file (keyed by resolved path, so a
    test that moves the DB to a second tmp file still gets its schema)."""
    key = str(db_path().resolve())
    if key in _init_done:
        return
    with _init_lock:
        if key in _init_done:
            return
        conn.executescript(_SCHEMA)
        _init_done.add(key)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(at: datetime) -> str:
    return at.isoformat(timespec="seconds")


# ── the allow-list ──────────────────────────────────────────────────────────

def allow_entries(raw: Any = None) -> list[dict]:
    """Normalized allow-list from config.

    A plain string (`"duckduckgo.com"`) means "any scope, no expiry"; a dict may
    carry `host`, `scope`, `reason`, `expires_at`. An entry whose host will not
    validate is dropped **loudly** rather than applied loosely — a malformed
    entry in a security list is the one input where guessing is worse than
    ignoring.
    """
    if raw is None:
        raw = config().get("allow") or []
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[dict] = []
    for item in raw:
        if isinstance(item, str):
            item = {"host": item}
        if not isinstance(item, dict):
            continue
        try:
            host = validate_destination(item.get("host"))
        except ValueError as exc:
            logger.warning("egress: dropping allow entry %r (%s)", item, exc)
            continue
        out.append({
            "host": host,
            "scope": str(item.get("scope") or ""),
            "reason": str(item.get("reason") or ""),
            "expires_at": str(item.get("expires_at") or ""),
        })
    return out


def _covers(entry: dict, *, host: str, scope: str, at: datetime) -> bool:
    if not host_matches(entry["host"], host):
        return False
    if entry["scope"] and entry["scope"] != scope:
        return False
    exp = entry.get("expires_at") or ""
    if exp:
        try:
            cutoff = datetime.fromisoformat(exp.replace("Z", "+00:00"))
        except ValueError:
            return False
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        return at < cutoff.astimezone(timezone.utc)
    return True


def count_permanent_entries(raw: Any = None) -> int:
    """Allow entries with no expiry — the number #628's acceptance says must be
    reported, not hidden."""
    return sum(1 for e in allow_entries(raw) if not e["expires_at"])


# ── grants ──────────────────────────────────────────────────────────────────
#
# No matcher here on purpose. Whether a grant authorizes this (scope, tool,
# host) is answered by `app.harness.policy.check_grants(..., destination=host)`
# and nowhere else: #534's gate and #628's guard consume the same rows, so a
# destination grant with `quota=20` bounds twenty calls total, not twenty plus
# whatever the other gate happened to allow. A second scan of the table is also
# where a second, looser definition of host matching would grow — the reason
# `host_matches` is a re-export of `policy.destination_covers` rather than a
# local regex.
def grant_covers(*, scope: str, tool: str, host: str,
                 at: datetime) -> tuple[bool, str]:
    """(authorized, reason) from the live grant store — #534's gate, asked directly.

    `record=False`: the decision surface for egress is `egress_events`, and
    writing a `grant_dispatch` row per fetch would make #534's ledger a function
    of how often the web is polled rather than of how often authority was
    actually needed.
    """
    from app.harness.policy import check_grants, default_store
    presented = scope or "worker"   # an unscoped turn borrows 'worker', as the pool's own default does
    decision = check_grants(default_store(), scope=presented, tool_name=tool,
                            tool_input={}, destination=host, now=at,
                            record=False)
    return (decision.allowed, decision.reason)


# ── the decision ────────────────────────────────────────────────────────────

class Decision:
    """The guard's answer, and the row it wrote.

    `allowed` is what the tool does; `decision` is what the table records. They
    differ for one case on purpose: an attended call to an unknown destination
    is recorded `grant-required` and still proceeds, because refusing a human
    would only get enforcement turned off.
    """

    __slots__ = ("allowed", "decision", "reason", "destination", "host", "scope")

    def __init__(self, *, allowed: bool, decision: str, reason: str = "",
                 destination: str = "", host: str = "", scope: str = ""):
        self.allowed = allowed
        self.decision = decision
        self.reason = reason
        self.destination = destination
        self.host = host
        self.scope = scope

    @property
    def refusal(self) -> str:
        return self.reason


def _decide(*, tool: str, destination: str, host: str, scope: str,
            attended_here: bool, at: datetime) -> tuple[str, bool, str]:
    """(decision, allowed, reason) for a destination that cleared the floor.

    Three answers, in this order:

    1. an allow-list entry covering (host, scope) and not expired → allow;
    2. an attended call → `grant-required`, and **allowed**. A human is on the
       other end; the row is still written because "attended scopes are never
       refused" is a statement about the tool's answer, not about whether the
       event happened;
    3. otherwise #534's gate decides: a live destination grant for this exact
       (scope, tool, host) or a refusal that carries the mintable row.

    An attended call does not consult the store at all, so an operator browsing
    to an unknown host can never spend a task's destination quota by accident.
    """
    if any(_covers(e, host=host, scope=scope, at=at) for e in allow_entries()):
        return DECISION_ALLOW, True, ""
    if attended_here:
        return DECISION_GRANT, True, ""
    try:
        allowed, why = grant_covers(scope=scope, tool=tool, host=host, at=at)
    except Exception as exc:  # noqa: BLE001 — an unreadable store is a closed door
        return DECISION_DENY, False, (
            f"egress: refused {tool} to {destination} — the grant store is "
            f"unreadable ({exc.__class__.__name__}: {exc}) and an egress check "
            "that cannot see its own rows cannot allow. No connection was "
            "attempted.")
    if allowed:
        return DECISION_ALLOW, True, ""
    return DECISION_DENY, False, refusal_text(tool=tool, destination=destination,
                                              host=host, scope=scope, now=at)


def refusal_text(*, tool: str, destination: str, host: str, scope: str,
                 now: datetime | None = None) -> str:
    """The denial text, carrying the exact mintable row.

    The grant shape comes from `app.harness.policy.grant_shape`, the same
    renderer #534's dispatch gate uses, extended with the destination axis — one
    issuance UI, not two that can drift.
    """
    from app.harness.policy import grant_shape

    row = grant_shape(scope=scope or "interactive", tool=tool, tool_input={},
                      destination=host, now=now)
    return (
        f"egress: refused {tool} to {destination} — destination is not on the "
        f"allow-list and no live grant covers it, and scope '{scope}' is "
        f"unattended. No connection was attempted. A human issues the row: {row}"
    )


def guard(tool: str, url: str = "", *, host: str = "",
          loopback_ok: bool = False, floor_off: bool = False,
          session_id: str = "") -> Decision:
    """One destination-tagged row, and the policy's answer. Call before any socket.

    `host` is passed directly by `http_search`, which takes no URL and has to
    name its backend explicitly.

    Two inputs carry an existing decision rather than making a new one, and each
    is set by the lane that already had the behaviour:

    * `loopback_ok` — `http_request`'s sanctioned local-service allowance
      (`http_tools.py:500`), which the destination axis must not silently revoke.
    * `floor_off` — the caller's `block_private_hosts` knob resolved. The
      browser has one that ships off-but-settable for a local-only browser, and a
      guard that reinstated what the operator switched off would be a silent
      behavior change wearing a security badge.

    The floor is part of the enforced policy. With `enforce` off the guard only
    records, and each lane's own private-host check below it answers exactly as
    it did before #628 — so the default-off flag changes no tool's behaviour.
    With it on, the floor runs first and its message passes through verbatim.
    """
    if tool not in EGRESS_TOOLS:
        # Not a refusal — an uninstrumented path. Recorded nowhere so the table
        # stays the inventory of the four lanes it claims to cover.
        return Decision(allowed=True, decision=DECISION_ALLOW)
    if host:
        # `http_search` names its backend directly rather than passing a URL.
        destination, host = f"https://{host}", host
    else:
        destination, host = destination_of(url)

    at = _now()
    enforced = enforce_on()
    ctx = scope_context()
    sid = session_id or _session_id()
    attended_here = attended(sid)
    # An attended turn carries no effect scope; name it with #534's interactive
    # scope rather than '', so the inventory separates a human browsing from an
    # unscoped fleet call (`run_prompt_on_primary` turns carry no scope either).
    scope = ctx["scope"] or (_interactive_scope() if attended_here else "")

    decision, reason, allowed = DECISION_ALLOW, "", True
    if host and loopback_ok and _is_loopback(host):
        # `http_request`'s sanctioned local-service lane: neither the floor nor
        # the destination axis applies (see the module docstring).
        decision, reason = DECISION_ALLOW, "loopback allowance"
    elif enforced and not host:
        decision, allowed = DECISION_DENY, False
        reason = f'Unparsable destination: "{url}"'
    elif enforced and not floor_off and floor_reason(host, loopback_ok=loopback_ok):
        # The floor is evaluated before the allow-list, so an entry naming a
        # private address denies rather than reopening it. Its message passes
        # through unchanged: it is the text each lane already returned for the
        # same address, and callers and tests match on it.
        decision, allowed = DECISION_DENY, False
        reason = floor_reason(host, loopback_ok=loopback_ok)
    elif host and (enforced or attended_here):
        # Unenforced, an attended call is still classified — `grant-required`
        # is how the inventory shows a human reaching a host outside policy —
        # and `_decide` never refuses an attended call, so this cannot deny.
        decision, allowed, reason = _decide(
            tool=tool, destination=destination, host=host, scope=scope,
            attended_here=attended_here, at=at)

    # What the table's `reason` column carries. A denial carries the refusal
    # text the caller saw, so the table is the audit of what was said as well as
    # what was decided; a grant-required row carries just the tag, because the
    # full mint shape in every one of those rows would make the table unreadable
    # for the one thing it is looked at — which hosts, from which scopes.
    row_reason = reason or {
        DECISION_GRANT: DECISION_GRANT,
        DECISION_ALLOW: "allow-listed" if enforced else "",
    }.get(decision, "")

    if telemetry_on():
        _record(tool=tool, destination=destination, host=host,
                decision=decision, scope=scope, session_id=sid,
                worker_source=ctx["worker_source"], task_id=ctx["task_id"],
                reason=row_reason, at=at)
    return Decision(allowed=allowed, decision=decision, reason=reason,
                    destination=destination, host=host, scope=scope)


def _interactive_scope() -> str:
    from app.harness.policy import INTERACTIVE_SCOPE
    return INTERACTIVE_SCOPE


def _is_loopback(host: str) -> bool:
    from agent_mcp.http_tools import _is_loopback_host
    return _is_loopback_host(host)


def _session_id() -> str:
    try:
        from agent_mcp._shared import get_bound_session
        return str(get_bound_session() or "")
    except Exception:  # noqa: BLE001
        return ""


def _record(*, tool: str, destination: str, host: str, decision: str,
            scope: str, session_id: str, worker_source: str, task_id: str,
            reason: str, at: datetime) -> None:
    """Write the row, or lose it quietly. See the module docstring on fail-open."""
    try:
        conn = _connect()
        try:
            _ensure(conn)
            conn.execute(
                f"INSERT INTO {TABLE} (at, tool, destination, host, decision,"
                " scope, session_id, worker_source, task_id, reason)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (_iso(at), tool, destination, host, decision, scope,
                 session_id or "", worker_source or "", task_id or "",
                 (reason or "")[:500]))
            _maybe_prune(conn, now=time.monotonic())
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 — telemetry never bricks a fetch
        logger.warning("egress: could not record %s → %s (%s)", tool, destination, exc)


async def aguard(tool: str, url: str = "", *, host: str = "",
                 loopback_ok: bool = False, floor_off: bool = False) -> Decision:
    """`guard` off the event loop.

    The aggregator runs every tool call on one loop, and this path contains two
    SQLite statements and a session-file read. `http_fetch`/`http_request` are
    sync functions that the aggregator calls from a coroutine, so a blocking
    write there would sit on the loop that serves every other turn — the same
    reason `_tool_effects.claim` hops threads.
    """
    # The bound session is resolved here and passed explicitly, so the guard's
    # answer does not rest on context propagation into the worker thread.
    return await asyncio.to_thread(
        guard, tool, url, host=host, loopback_ok=loopback_ok,
        floor_off=floor_off, session_id=_session_id())


def _maybe_prune(conn: sqlite3.Connection, *, now: float) -> None:
    global _last_prune
    if now - _last_prune < PRUNE_INTERVAL_S:
        return
    _last_prune = now
    try:
        days = float(config().get("retention_days") or 90)
        cutoff = _iso(_now() - timedelta(days=days))
        n = conn.execute(f"DELETE FROM {TABLE} WHERE at < ?", (cutoff,)).rowcount
        if n:
            logger.info("egress: pruned %d rows older than %s", n, cutoff)
    except Exception as exc:  # noqa: BLE001
        logger.warning("egress: prune failed (%s)", exc)


# ── the report /api/dashboard reads ─────────────────────────────────────────

def network_report(days: float = 7.0, *, limit: int = 25) -> dict:
    """The destination inventory: per-destination and per-scope counts.

    Reads without creating — a box with workers switched off must not have a
    database materialised to report zero, and a zero-denominator report must
    say so rather than present an empty table as "nothing left".
    """
    since = _iso(_now() - timedelta(days=float(days)))
    path = db_path()
    empty = {
        "window_days": float(days), "since": since, "total": 0,
        "distinct_hosts": 0, "by_decision": {DECISION_ALLOW: 0, DECISION_DENY: 0,
                                             DECISION_GRANT: 0},
        "per_destination": [], "per_scope": [],
        # `database_present` is the zero denominator named beside the zero
        # numerator: `total: 0` with a database means no call was made in the
        # window, and without one means the writer never ran. Reporting the
        # first as the second would make a broken telemetry path look like an
        # idle box, which is the reading that lets it stay broken.
        "database": str(path), "database_present": path.exists(),
        "policy": {"telemetry": telemetry_on(), "enforce": enforce_on(),
                   "allow_entries": len(allow_entries()),
                   "permanent_allow_entries": count_permanent_entries()},
    }
    conn = _connect(create=False)
    if conn is None:
        return empty
    try:
        _ensure(conn)
        total = int(conn.execute(
            f"SELECT COUNT(*) FROM {TABLE} WHERE at>=?", (since,)).fetchone()[0])
        hosts = int(conn.execute(
            f"SELECT COUNT(DISTINCT host) FROM {TABLE} WHERE at>=?",
            (since,)).fetchone()[0])
        by_decision = {DECISION_ALLOW: 0, DECISION_DENY: 0, DECISION_GRANT: 0}
        for row in conn.execute(
                f"SELECT decision, COUNT(*) AS n FROM {TABLE} WHERE at>=?"
                " GROUP BY decision", (since,)):
            if row["decision"] in by_decision:
                by_decision[row["decision"]] = int(row["n"])
        per_destination = [
            {"destination": r["destination"], "host": r["host"],
             "count": int(r["n"]), "denied": int(r["denied"]),
             "grant_required": int(r["grant_required"]),
             "scopes": int(r["scopes"]), "last_at": r["last_at"]}
            for r in conn.execute(
                f"SELECT destination, host, COUNT(*) AS n,"  # noqa: S608 — TABLE is a module constant
                f" SUM(decision='{DECISION_DENY}') AS denied,"
                f" SUM(decision='{DECISION_GRANT}') AS grant_required,"
                " COUNT(DISTINCT CASE WHEN scope<>'' THEN scope END) AS scopes,"
                f" MAX(at) AS last_at FROM {TABLE} WHERE at>=?"
                " GROUP BY destination, host ORDER BY n DESC LIMIT ?",
                (since, limit))]
        per_scope = [
            {"scope": r["scope"] or "(interactive/unscoped)",
             "total": int(r["n"]), "denied": int(r["denied"]),
             "grant_required": int(r["grant_required"]),
             "distinct_hosts": int(r["hosts"])}
            for r in conn.execute(
                f"SELECT scope, COUNT(*) AS n,"
                f" SUM(decision='{DECISION_DENY}') AS denied,"
                f" SUM(decision='{DECISION_GRANT}') AS grant_required,"
                f" COUNT(DISTINCT host) AS hosts FROM {TABLE} WHERE at>=?"
                " GROUP BY scope ORDER BY n DESC LIMIT ?",
                (since, limit))]
        return {
            "window_days": float(days), "since": since, "total": total,
            "distinct_hosts": hosts, "by_decision": by_decision,
            "per_destination": per_destination, "per_scope": per_scope,
            "database": str(path), "database_present": True,
            "policy": {"telemetry": telemetry_on(), "enforce": enforce_on(),
                       "allow_entries": len(allow_entries()),
                       "permanent_allow_entries": count_permanent_entries()},
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("egress: network_report failed (%s)", exc)
        return {**empty, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        conn.close()
