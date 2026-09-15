"""Per-session ceiling on side-effecting tool calls, enforced at dispatch (#584).

Lloyd's guards were all booleans before this file. `policy.py` asks whether this
session may use this tool at all; `_tool_sandbox.py` asks whether this session is
read-only; #582's path rules ask whether this *path* is writable; #534's grants
ask whether this *scope* was given the permission and when it expires; #544 asks
whether this exact effect already happened. None of them asks **how many of
these has this session already done in the last hour**, and the 2026-08-22 loss
is the incident that question answers: a nightly extraction stage that truncated
`entity-aliases.json`, deleted `_relationships.json` and `memory-graph/`, and
rebuilt `facts/` wholesale — one run, inside its own scope, every single call
legal, and `_pipeline/` is gitignored so none of it came back. A boolean cannot
stop that run. A ceiling can: not because any one of those deletes was
disallowed, but because there were hundreds of them in a minute.

The shape is Anthropic's admission-webhook pattern (Sachin Malhotra, "Give the
Agent a Budget, Not a Token"): every mutating class gets a rate limit, only the
size changes, and the bypass is a human-side action rather than a flag the
caller can set. Their incident was a cleanup command whose filter evaluated to
nothing and deleted ~200 workloads in 90 seconds with the author's own token and
fully in scope — the same shape as ours.

What this is *not*, stated where it matters most:

* **A ceiling is not a scope.** One `automod_rollback` inside budget still
  destroys, exactly as one correctly-shaped `rm` inside budget still destroyed
  `_relationships.json`. #582 (substrate enforcement of protected paths) is the
  scope axis and this complements it; a bulk wipe is what only this can stop.
* **A ceiling is not idempotency.** #544 prevents the *same* effect happening
  twice across retries; this prevents *many* different effects in one window.
* **A ceiling is not a permission.** A deny here says "not yet, and a human
  decides whether ever", never "you may not".

Keying
------

The bucket key is **(charge key, op-class, target-scope, run-source)**:

* the **charge key** is the harness session id from `_meta`, never an argument —
  except that a call whose id arrived only as an argument is charged to
  ``UNBOUND`` (see there), so no model-supplied string buys a fresh budget;
* the **op-class** is the *kind* of effect (`fs_delete`, not `email_delete`) —
  a tool-name list is the wrong denominator, because an agent with a goal
  achieves it through several tools and per-tool budgets multiply;
* the **target-scope** is `vault` / `lloyd` / `_pipeline` / `other` / `none`, so
  a burst of `~/lloyd` edits cannot spend the vault's budget;
* the **run-source** is the claimed queue item's source, or the session id's
  producer slug, else `interactive` — the tightest table.

The window is fixed and the ceiling is a total: a burst gets the whole ceiling,
which is the behaviour the runaway cases want (a nightly job gets its 1500
writes and then stops dead) rather than a gradual refill that would let a
runaway trickle for hours. The item's "refilling" is the window resetting.

Fail-open
---------

Every counter write is SQLite on the dispatch path of every mutating call in the
fleet, so a broken counter must degrade to the pre-#584 behaviour rather than
freeze every write in the machine. `harness.mutation_budget.enabled: false` is
the kill switch.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("lloyd-mcp.budget")

# ── the clock ───────────────────────────────────────────────────────────────
#
# Module-level, not `time.time()` at each call site, for two reasons: the refill
# test advances it rather than sleeping (the window is an hour), and the ledger's
# `at` column and the hour bucket must come from the same reading or a run that
# crosses an hour boundary splits its own counts.
#
# `time.time()`, not monotonic: the ledger is read by a person and by the
# dashboard long after the call, and a row whose `at` is seconds-since-boot is
# unreadable. The cost — an NTP step can shift a window — is a ceiling that is
# briefly early or late, which is the right side to be wrong on for a control
# whose whole job is to make a runaway slow down.


def clock() -> float:
    return time.time()


#: `main.META_EFFECT_SCOPE`, repeated because this module is imported by the
#: backend too (`app/routers/workers.py`) and importing `main.py` there would
#: drag in the whole server.
META_EFFECT_SCOPE = "lloyd/effect_scope"

#: The charge key for a call whose session id did NOT arrive in `_meta`.
#:
#: `_bound_session_id` still honours a legacy `_session_id` *argument* so a
#: harness and an aggregator at different versions correlate. That fallback is
#: fine for attribution and fatal for a ceiling: the bucket's entire value is
#: that the model cannot choose it, and a caller-supplied id buys unlimited
#: fresh budgets (`sid-1`, `sid-2`, …, N calls each) — and shaped like a nightly
#: job (`20260915_030000_scheduledtask_ab12`) it would also jump from the
#: interactive table to the batch one. So an unbound call is charged to this one
#: shared key and evaluated as `interactive`: the tightest ceiling, shared,
#: rather than an unlimited supply. Every real caller is bound —
#: `app/harness/mcp_pool.py:474-476` sets `_meta[lloyd/session_id]` on every
#: call — so this closes the only argument-side route to a bigger bucket at no
#: practical cost.
UNBOUND = "unbound"

#: Argument names that read as an attempt to move the ceiling. They are never
#: *read* here — the bucket key comes from `_meta` and config, and an argument
#: contributes only the path it names — but the set is named so the sanitiser in
#: `app/harness/loop.py` can strip it from what gets logged and replayed, and so
#: the test can pin that adding a key to this set cannot change a decision.
#:
#: Deliberately narrow: stripping a key nobody named would start rewriting real
#: payloads, and `file_path`/`path` must survive precisely because they are what
#: the target-scope is derived from.
SanitisedKeys = frozenset({
    "bypass", "ceiling", "mutation_ceiling", "rate_limit", "window_seconds",
    "_session_id",
})

# ── op classes ──────────────────────────────────────────────────────────────
#
# Named for the effect, not the tool. `fs_write` is the same bucket whether the
# call came from `Write`, `Edit`, `vault_write` or `memory_add`, because the
# failure this caps is aggregate: an agent with a goal reaches it through
# several tools, and per-tool budgets multiply into nothing.

_FS_WRITE = frozenset({
    "Write", "Edit", "SearchReplace", "MultiEdit", "NotebookEdit",
    "vault_write", "memory_add", "memory_replace", "memory_remove",
    "automod_vault_land",
})

_FS_DELETE = frozenset({
    "ide_close_tab", "automod_vault_revert", "email_delete",
    "calendar_delete_event", "contacts_delete", "email_empty_trash",
    "email_empty_junk", "email_delete_folder", "email_delete_filter",
    "fact_invalidate", "forget",
})

_SEND = frozenset({
    "email_send", "email_reply", "email_forward", "email_save_draft",
    "calendar_create", "tasks_create", "mc_navigate", "mc_close_modal",
    "ide_open_file", "ide_open_folder", "email_display", "browser_navigate",
})

# The heaviest single effects: each can take down the harness or the knowledge
# graph, so they get a ceiling of their own rather than being pooled with sends.
_DESTROY = frozenset({
    "automod_start", "automod_gate", "automod_amend_clause", "automod_land",
    "automod_rollback", "automod_abort",
    "autonomy_run_task", "autoresearch_round", "autoresearch_promote",
    "autoresearch_rollback",
    "grant_revoke", "ClearGoal", "ExitPlanMode",
    "research_complete", "browser_tabs", "browser_screenshot",
})

_RECORD_WRITE = frozenset({
    "backlog_write_task", "autonomy_write_task", "research_propose",
    "autoresearch_bench_add", "fact_add", "remember", "fact_relate",
    "vault_write",
})

_SPAWN = frozenset({"Task", "autonomy_run_task", "autoresearch_round"})

#: A shell command that is not otherwise a delete, an unknown/unclassified
#: mutating tool, or any of the named classes above. `Bash` is the wildcard
#: tool, so without this bucket its ceiling would be the union of every effect
#: it can have.
_OTHER = "other"

OP_CLASSES = ("fs_write", "fs_delete", "send", "destroy", "record_write",
              "kg_write", "spawn", "shell", "other")

#: Bash is capped by *what its arguments say*, not by its name. The named
#: mutating tools are covered above; this is the shape an autonomous runaway
#: actually takes, because `Bash` can do anything the tools can. Every verb
#: here is a removal or a truncation — `mv` and `>` are excluded deliberately:
#: misclassifying a move as a delete would spend the tight `fs_delete` ceiling
#: on ordinary file motion, and a wrong ceiling that is too *tight* is a
#: different outage, not a safe one.
_CMD_START = r"(?:\A|[;&|\n(]|\$\(|\bxargs\s+)"
_ENV_PREFIX = r"\s*(?:(?:[\w.]+=\S+)\s+)*"
_DELETE_COMMAND = re.compile(
    # A removal verb in *command position*. Matching a bare `\brm\b` anywhere in
    # the string would classify `grep -rn "rm " notes.md` as a delete and charge
    # it to the delete ceiling — which is how a classifier spends its
    # credibility and a batch job runs out of budget on greps.
    _CMD_START + _ENV_PREFIX + r"(?:rm|rmdir|unlink|shred|srm|wipefs)\b"
    r"|" + _CMD_START + _ENV_PREFIX + r"find\b[^|;&\n]{0,200}?\s-delete\b"
    r"|" + _CMD_START + _ENV_PREFIX + r"git\s+(?:clean|restore)\b"
    r"|" + _CMD_START + _ENV_PREFIX + r"truncate\s+(?:-s\s*|--size[= ]*)0\b"
    # `dd of=/dev/null` is a discard, not a destruction.
    r"|" + _CMD_START + _ENV_PREFIX + r"dd\b[^|;&\n]{0,200}?\bof=(?!\s*/dev/(?:null|zero)\b)",
    re.IGNORECASE,
)

#: Path-shaped argument keys: anything naming a place on disk, under whatever
#: name each tool happens to use. Deliberately excludes keys that merely look
#: pathlike (`entity`, `collection`, `source`, `dir`-free names): a knowledge-
#: graph entity name is not a filesystem target, and charging a fact write to
#: whatever bucket its name happens to resemble is how a ceiling stops meaning
#: anything.
_PATH_ARG_KEYS = frozenset({
    "file_path", "path", "paths", "target", "destination", "dir", "directory",
    "folder", "folderPath", "folder_path", "parentFolderPath",
    "parent_folder_path", "newParentPath", "new_parent_path",
    "artifact_path", "root", "output", "output_file", "attachment", "cwd",
})

_VAULT_ROOT = Path(os.path.expanduser("~/obsidian"))
_REPO_ROOT = Path(os.path.expanduser("~/lloyd"))
_PIPELINE_ROOT = _REPO_ROOT / "_pipeline"

#: Tightest-first. The order `_scope_rank` reads.
SCOPE_ORDER = ("vault", "_pipeline", "lloyd", "other")

#: A tool that is mutating but names no filesystem target at all (an email, a
#: calendar entry, a KG fact). Its own scope, and the "*" ceiling row is the one
#: that applies to it.
NO_TARGET_SCOPE = "none"


def classify_path(raw: Any) -> Optional[str]:
    """Which ceiling scope a path argument names, or None if it isn't a path."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip().replace("$HOME", str(Path.home()))
    if text.startswith((".", "~", "/")):
        # Lexical, not `realpath`: the target usually does not exist yet (that
        # is what a write is), and resolving symlinks would make
        # `/tmp/x -> ~/obsidian/x` look like `/tmp`.
        resolved = Path(os.path.normpath(os.path.expanduser(text)))
    else:
        # A bare relative path is the shape `vault_write` takes
        # (`knowledge/foo.md`) and not the shape the file tools take (they
        # require absolute). Guessing `other` here would hand every nightly
        # vault rewrite the loosest bucket, so the guess is `vault`.
        return "vault"
    for scope, root in (("vault", _VAULT_ROOT), ("_pipeline", _PIPELINE_ROOT),
                        ("lloyd", _REPO_ROOT)):
        if resolved == root or root in resolved.parents:
            return scope
    return "other"


def _path_tokens(command: str) -> list[str]:
    """Path-ish words in a shell command.

    `shlex` first, whitespace split as the fallback when it cannot be tokenised
    (unbalanced quotes — a command the harness is about to refuse anyway, and a
    wrong scope there must not become a crash).
    """
    try:
        words = shlex.split(command, posix=True)
    except ValueError:
        words = command.split()
    return [w for w in words
            if w and not w.startswith("-")
            and w.startswith(("~", "/", ".", "$HOME", "$V"))]


def target_scope(name: str, arguments: Any) -> str:
    """The scope a call is charged to: the *tightest* named scope it touches.

    The tightest, not the loosest: a call that touches the vault and a temp file
    is a vault call. Anything else would make a mixed-argument call the cheapest
    route past the vault ceiling, which is the same mistake as a per-tool budget.
    """
    args = arguments if isinstance(arguments, dict) else {}
    scopes: list[str] = []

    if name == "Bash":
        command = str(args.get("command") or "")
        for token in _path_tokens(command):
            scopes.append(classify_path(token) or NO_TARGET_SCOPE)
        # A delete with no path token (`git clean` in cwd) is scoped by cwd,
        # which is exactly the 08-22 shape: the damage was relative to a root.
        scopes.append(classify_path(args.get("cwd")))
    else:
        for key in _PATH_ARG_KEYS:
            if key not in args:
                continue
            value = args[key]
            if isinstance(value, (list, tuple)):
                for item in value:
                    scopes.append(classify_path(item))
            else:
                scopes.append(classify_path(value))
    scopes = [s for s in scopes if s]
    if not scopes:
        return NO_TARGET_SCOPE
    for scope in SCOPE_ORDER:
        if scope in scopes:
            return scope
    return NO_TARGET_SCOPE


def op_class(name: str, arguments: Any) -> Optional[str]:
    """The capped class for a tool call, or None if the call is not mutating.

    Read-only tools (including `Read`/`Grep`/`Glob` and every annotated
    read-only MCP tool) return None and never touch the counter database, so
    normal navigation and the ceiling never compete. The predicate is a superset
    of `annotations.READ_ONLY`, and the test asserts that for every real tool.
    """
    if name == "Bash":
        command = str((arguments or {}).get("command") or "") if isinstance(
            arguments, dict) else ""
        if _DELETE_COMMAND.search(command):
            return "fs_delete"
        # An unrecognised Bash command is still mutating in the general case.
        # The alternative — cap only what the regex knows — is a ceiling that
        # any unseen write walks straight through, and the harness's own
        # destructive-pattern deny has already had its chance at the scary ones.
        return "shell"
    if name in _FS_DELETE:
        return "fs_delete"
    if name in _FS_WRITE and name not in _RECORD_WRITE:
        return "fs_write"
    if name in _RECORD_WRITE:
        return "record_write"
    if name in _DESTROY:
        return "destroy"
    if name in _SPAWN:
        return "spawn"
    if name in _SEND:
        return "send"
    # An unknown tool is capped, not waved through — the same default
    # `annotations.annotations_for` gives an unclassified tool in plan mode. A
    # newly added mutating tool is therefore capped from the commit that added
    # it, without anyone remembering to list it here.
    return _OTHER


def _is_capped(name: str, arguments: Any) -> bool:
    """Whether a call is charged at all: `op_class` returning something."""
    return op_class(name, arguments) is not None


# ── run source ──────────────────────────────────────────────────────────────

_BG_SESSION = re.compile(r"^\d{8}_\d{6}_([a-z0-9]+)_[0-9a-f]{4}$")
_ITEM_SCOPE = re.compile(r"^item:([^:]+):.+$")


def run_source(session_id: str, effect_scope: str = "", *,
               bound: bool = True) -> str:
    """Which run this call belongs to, from harness state only.

    The claimed queue item's scope wins (`item:autonomy-task:39` →
    `autonomy-task`), because that is the unit a ceiling has to be per-source
    for: the same session id can appear in a worker's turn and a person's turn.
    Then `bound=True` and the four-part background session id's producer slug
    (`20260915_030000_scheduledtask_ab12`), which is what a session-backed
    worker turn looks like once its item scope is gone — a source that opens a
    second session, rather than a `Task` subagent inheriting the scope.

    `bound=False` means the id arrived as an *argument*, so its shape is
    model-supplied and is not trusted for source selection either.

    Everything unresolved lands on `interactive`, the tightest table.
    """
    if effect_scope:
        match = _ITEM_SCOPE.match(effect_scope)
        if match:
            return match.group(1)
    if session_id and bound:
        match = _BG_SESSION.match(session_id)
        if match:
            return match.group(1)
    return "interactive"


# ── ceilings ────────────────────────────────────────────────────────────────
#
# Per (source, op-class, target-scope), each level falling back to "*":
# `ceilings[source][class][scope]` → `ceilings[source][class]["*"]` →
# `ceilings["*"][class][scope]` → `ceilings["*"][class]["*"]`. Two tables:
# `interactive`, which is a person who rarely needs more than a few dozen writes
# an hour, and `*` for worker sources — the nightly consolidation and vault
# maintenance jobs that legitimately rewrite hundreds of notes.
#
# The numbers below are a starting point, not a calibrated answer. The item
# names ceiling sizing as a human calibration decision after one night of real
# batch jobs, and `status()` is what that person reads to set a source that
# tripped. What matters mechanically is the *ordering*: interactive tight,
# `_pipeline` tighter than the vault for deletes (that is where 08-22 happened),
# and a batch source much looser than a person for writes.
_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "window_seconds": 3600,
    # Trip wire: a source/class/scope whose ALLOWED calls in the hour reach this
    # fraction of its own ceiling gets a row in `mutation_alerts` (alert, never
    # block). The ratio is below 1.0 because anything at or above 1.0 is
    # unreachable by construction — the ceiling stops the bucket at 1.0 — so a
    # "multiplier" of 2 would be an alert that can never fire, the same shape as
    # a check whose denominator is always zero. 0.9 means "this bucket is within
    # one call of starting to refuse work". Ceiling-relative rather than a
    # rolling historical baseline on purpose — see `_trip_wire`.
    "tripwire_ratio": 0.9,
    "retention_seconds": 7 * 86400,
    "ceilings": {
        "interactive": {
            # A vault note has no git history; a repo file does, so the repo
            # write is the recoverable one and gets the looser number.
            "fs_write": {"vault": 120, "lloyd": 400, "_pipeline": 30, "*": 120},
            # The vault is the unrecoverable substrate and `_pipeline` is where
            # 08-22 happened; `~/lloyd` is git-tracked, so a repo delete is
            # recoverable and gets the loosest of the three.
            "fs_delete": {"vault": 10, "lloyd": 30, "_pipeline": 5, "*": 10},
            "send": {"*": 15},
            "destroy": {"*": 15},
            "record_write": {"*": 40},
            "kg_write": {"*": 120},
            "spawn": {"*": 40},
            "shell": {"*": 300},
            "other": {"*": 1000},
        },
        "*": {
            "fs_write": {"vault": 1500, "lloyd": 600, "_pipeline": 1500, "*": 600},
            "fs_delete": {"vault": 200, "lloyd": 200, "_pipeline": 500, "*": 200},
            "send": {"*": 150},
            "destroy": {"*": 200},
            "record_write": {"*": 400},
            "kg_write": {"*": 5000},
            "spawn": {"*": 100},
            "shell": {"*": 3000},
            "other": {"*": 5000},
        },
    },
}


def config() -> dict:
    """`harness.mutation_budget`, merged over `_DEFAULTS` key by key.

    Merge, not replace: a config that sets only `enabled: false` must not empty
    the ceiling table, because `ceiling_for` would then find no ceilings and
    every call would be uncapped — the silent-open failure of a half-written
    config, which is the failure mode a control like this must not have. The
    `ceilings` sub-tree is merged one level deeper for the same reason.
    """
    from app.config import CONFIG
    raw = (CONFIG.get("harness") or {}).get("mutation_budget")
    cfg = dict(_DEFAULTS)
    if isinstance(raw, dict):
        for key, value in raw.items():
            if key != "ceilings":
                cfg[key] = value
        merged = {k: dict(v) if isinstance(v, dict) else v
                  for k, v in (_DEFAULTS["ceilings"]).items()}
        for source, classes in ((raw.get("ceilings") or {})).items():
            if not isinstance(classes, dict):
                continue
            table = dict(merged.get(source) or {})
            for cls, scopes in classes.items():
                if isinstance(scopes, dict):
                    table[cls] = {**(table.get(cls) or {}), **scopes}
                else:
                    table[cls] = scopes
            merged[source] = table
        cfg["ceilings"] = merged
    return cfg


def enabled() -> bool:
    return bool(config()["enabled"])


def ceiling_for(op_cls: str, source: str, scope: str) -> Optional[int]:
    """The ceiling for one bucket, or None when the class is deliberately
    uncapped. `0` is not the same as None: 0 means block unconditionally."""
    return _ceiling_from(config()["ceilings"], op_cls, source, scope)


def _ceiling_from(ceilings: Any, op_cls: str, source: str, scope: str) -> Optional[int]:
    if not isinstance(ceilings, dict):
        return None
    for src in (source, "*"):
        table = ceilings.get(src)
        if not isinstance(table, dict):
            continue
        scopes = table.get(op_cls)
        if scopes is None:
            continue
        if not isinstance(scopes, dict):
            value = _as_int(scopes)
            if value is not None:
                return value
            continue
        for scl in (scope, "*"):
            if scl in scopes:
                value = _as_int(scopes[scl])
                if value is not None:
                    return value
    return None


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return None


# ── the counter database ────────────────────────────────────────────────────

_SCHEMA = """
-- One row per capped call, allowed or denied. `denied=True` on a read is the
-- calibration signal: a run with denied rows is a run that tried to exceed its
-- ceiling, which is the number a person sets a source's ceiling from.
--   session_id   the CHARGE key, not always the caller's id: an unbound call is
--                filed under 'unbound' (see UNBOUND), with bound=0 to say so.
--                Auditors wanting the claimed id have the aggregator's own log.
--   hour_bucket  the 3600-second bucket the trip wire aggregates over.
--   used_after   the bucket's allowed count *after* this decision, so on a deny
--                it is unchanged — the row says "the count was already at N".
CREATE TABLE IF NOT EXISTS mutation_calls (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  at            REAL NOT NULL,
  hour_bucket   INTEGER NOT NULL,
  session_id    TEXT NOT NULL,
  source        TEXT NOT NULL,
  tool          TEXT NOT NULL,
  op_class      TEXT NOT NULL,
  target_scope  TEXT NOT NULL,
  effect_scope  TEXT NOT NULL DEFAULT '',
  bound         INTEGER NOT NULL DEFAULT 1,
  allowed       INTEGER NOT NULL,
  ceiling       INTEGER,
  used_after    INTEGER
);
CREATE INDEX IF NOT EXISTS mutation_calls_bucket
  ON mutation_calls(op_class, target_scope, session_id, source, at);
CREATE INDEX IF NOT EXISTS mutation_calls_hour
  ON mutation_calls(source, hour_bucket, op_class, allowed);
CREATE INDEX IF NOT EXISTS mutation_calls_run
  ON mutation_calls(effect_scope, op_class) WHERE effect_scope != '';
CREATE TABLE IF NOT EXISTS mutation_alerts (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  hour_bucket   INTEGER NOT NULL,
  at            REAL NOT NULL,
  source        TEXT NOT NULL,
  op_class      TEXT NOT NULL,
  target_scope  TEXT NOT NULL,
  count         INTEGER NOT NULL,
  ratio         REAL NOT NULL,
  UNIQUE(hour_bucket, source, op_class, target_scope)
);
"""

#: Rows older than this are deleted, once per process hour. A table that grows
#: forever is a table that eventually makes the check slow, and the trip wire
#: needs only the current and prior hours while the ledger needs the window.
PRUNE_INTERVAL_S = 3600.0
_last_prune = 0.0
_init_done: set[str] = set()


def db_path() -> Path:
    """The same file `workers/queue.py` opens.

    Deliberately not a sibling file: this ledger is written from the aggregator
    and read by the backend's `/api/workers/runs`, and a second database means a
    second backup story and a second thing that can be missing while the queue
    is fine. `LLOYD_WORKERS_DB` already exists to point at another file — a test
    that overrides only one of the two would silently count into the live tree.
    """
    override = os.environ.get("LLOYD_MUTATION_DB")
    if override:
        return Path(os.path.expanduser(override))
    return Path(os.path.expanduser(
        os.environ.get("LLOYD_WORKERS_DB", "~/lloyd/_pipeline/workers/workers.db")))


def _connect(create: bool = True) -> Optional[sqlite3.Connection]:
    path = db_path()
    try:
        if not path.parent.exists():
            if not create:
                return None
            path.parent.mkdir(parents=True, exist_ok=True)
        # `isolation_level="IMMEDIATE"` is load-bearing, not a tuning knob: it
        # makes the implicit BEGIN that `with conn:` issues a `BEGIN IMMEDIATE`,
        # which takes the write lock *before* the count read. With the default
        # deferred mode each of two racing callers can read `used` below the
        # ceiling, insert, and be allowed — the exact race a reservation is
        # supposed to close, and one that stays invisible until a burst actually
        # races. `timeout=3.0` is the busy_timeout that waits out the other
        # writer rather than erroring.
        conn = sqlite3.connect(str(path), timeout=3.0,
                               isolation_level="IMMEDIATE")
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as exc:
        logger.warning("mutation budget: database unavailable at %s: %s — "
                       "dispatching uncapped this call (%s)",
                       path, exc, "fail-open")
        return None


def _ensure(conn: sqlite3.Connection) -> None:
    key = str(db_path())
    if key in _init_done:
        return
    conn.executescript(_SCHEMA)
    _init_done.add(key)


def _hour_bucket(now: float) -> int:
    return int(now // 3600)


@dataclass
class Decision:
    """What the gate decided, and the numbers the deny message is built from."""
    allowed: bool
    tool: str
    op_class: str
    target_scope: str
    source: str
    ceiling: int
    used: int
    window_seconds: int

    @property
    def remaining(self) -> int:
        return max(0, self.ceiling - self.used)

    @property
    def reason(self) -> str:
        return deny_message(self)


def deny_message(decision: Decision) -> str:
    """The corrective deny.

    Bounce-with-count is deliberate: waiting *is* the correct response, so the
    message has to make a retry in the same window useless rather than merely
    unfashionable. It also has to not read like a permission problem, because
    the two things an agent reaches for when a deny is vague are asking for a
    grant (#534) and rewording the call — and neither is the answer here.
    """
    window = decision.window_seconds
    human = window // 60 if window >= 60 else window
    unit = "minutes" if window >= 60 else "seconds"
    return (
        f"mutation ceiling reached: this session has already made "
        f"{decision.used} of {decision.ceiling} allowed {decision.op_class} "
        f"calls on scope {decision.target_scope} (run-source "
        f"{decision.source}) in the last {human} {unit}, so {decision.remaining} "
        f"remain and nothing ran. Do not retry this call inside the window — "
        f"the ceiling refills on its own, and re-sending the same arguments, or "
        f"a reworded one, draws from the same budget. No argument to this tool "
        f"raises it: to go past the ceiling, ask a human to raise "
        f"harness.mutation_budget in config.yaml. This is a rate ceiling, not a "
        f"permission — the call may be entirely in scope."
    )


def _count(conn: sqlite3.Connection, *, op_cls: str, scope: str, source: str,
           session_id: str) -> int:
    start = clock() - float(config()["window_seconds"])
    row = conn.execute(
        "SELECT COUNT(*) FROM mutation_calls WHERE op_class=? AND"
        " target_scope=? AND source=? AND session_id=? AND allowed=1 AND at>=?",
        (op_cls, scope, source, session_id, start)).fetchone()
    return int(row[0])


def _record(conn: sqlite3.Connection, *, tool: str, op_cls: str, scope: str,
            source: str, session_id: str, effect_scope: str, bound: bool,
            allowed: bool, ceiling: Optional[int], used_after: int,
            now: float) -> None:
    conn.execute(
        "INSERT INTO mutation_calls (at, hour_bucket, session_id, source, tool,"
        " op_class, target_scope, effect_scope, bound, allowed, ceiling,"
        " used_after) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (now, _hour_bucket(now), session_id, source, tool, op_cls, scope,
         effect_scope, 1 if bound else 0, 1 if allowed else 0, ceiling,
         used_after))


def _maybe_prune(conn: sqlite3.Connection, now: float) -> None:
    global _last_prune
    if now - _last_prune < PRUNE_INTERVAL_S:
        return
    _last_prune = now
    keep = now - float(config()["retention_seconds"])
    conn.execute("DELETE FROM mutation_calls WHERE at < ?", (keep,))
    conn.execute("DELETE FROM mutation_alerts WHERE at < ?", (keep,))


def _trip_wire(conn: sqlite3.Connection, cfg: dict, source: str, op_cls: str,
               scope: str, bucket: int) -> None:
    """Alert, never block, when a source is spending a whole ceiling's worth.

    Measured per (source, class, scope) against that bucket's own ceiling, on
    the hour's *allowed* rows: a bucket full of denials is the ceiling doing its
    job and is already in the log line the aggregator writes per denial, whereas
    a source that is *allowed* 2× a ceiling's worth in an hour is the shape the
    talk's trip wire exists to see (an investigation agent whose thread rate
    paged on-call and turned out to be one infra failure; the fix was one line
    of agent context, not a block).

    Ceiling-relative rather than a rolling historical baseline, deliberately: a
    baseline needs N hours of prior rows to average, so it abstains for the
    first N hours of the table's life — and an alert that abstains is an alert
    nobody can distinguish from an alert that is off. This one works on hour
    one. The rolling baseline is the obvious next step and is recorded on the
    item, not folded into this round.

    Counts ALLOWED rows only, and its ratio is below 1.0 (see the config
    comment): a threshold at or above the ceiling can never be reached, because
    the ceiling is what stops the count.

    One row per (hour, source, class, scope), updated in place: alerting per
    call is how an alert becomes noise.
    """
    ratio_cfg = float(cfg["tripwire_ratio"])
    ceiling = ceiling_for(op_cls, source, scope)
    if not ceiling:
        return
    count = int(conn.execute(
        "SELECT COUNT(*) FROM mutation_calls WHERE source=? AND hour_bucket=?"
        " AND op_class=? AND target_scope=? AND allowed=1",
        (source, bucket, op_cls, scope)).fetchone()[0])
    if count < ratio_cfg * ceiling:
        return
    now = clock()
    conn.execute(
        "INSERT INTO mutation_alerts (hour_bucket, at, source, op_class,"
        " target_scope, count, ratio) VALUES (?,?,?,?,?,?,?)"
        " ON CONFLICT(hour_bucket, source, op_class, target_scope)"
        " DO UPDATE SET count=excluded.count, at=excluded.at,"
        " ratio=excluded.ratio",
        (bucket, now, source, op_cls, scope, count, round(count / ceiling, 3)))
    logger.warning(
        "MUTATION RATE ALERT (trip wire, not a block): source %s made %d"
        " allowed %s calls on scope %s in the hour — %.1fx its ceiling of %d."
        " Anything over the ceiling was already denied.",
        source, count, op_cls, scope, count / ceiling, ceiling)


def _decide_sync(tool: str, op_cls: str, scope: str, source: str,
                 session_id: str, effect_scope: str, bound: bool) -> Decision:
    """Reserve a slot, or refuse. Atomic and synchronous: one writer per slot.

    Counted in a transaction that also inserts the row, so two callers cannot
    both see `used == ceiling - 1` and both be allowed. The `BEGIN IMMEDIATE` is
    what buys that, not the single connection: with the default deferred
    transaction a reader takes a *read* snapshot, so two processes can each read
    `used` below the ceiling, each insert, and both be allowed — the race the
    reservation exists to close, and it would be invisible until a burst actually
    raced. IMMEDIATE takes the write lock before the read, and `busy_timeout` in
    `_connect` covers a genuine concurrent writer.
    """
    cfg = config()
    window = int(cfg["window_seconds"])
    ceiling = ceiling_for(op_cls, source, scope)
    conn = _connect()
    if conn is None:
        # Fail open: a ceiling that cannot count must not freeze every write in
        # the fleet. #582's substrate checks stand independently of this one, so
        # an unavailable counter degrades to the pre-#584 behaviour, not to
        # nothing.
        return Decision(True, tool, op_cls, scope, source, int(1e9), 0, window)
    try:
        _ensure(conn)
        now = clock()
        # An unbound call is charged to one shared key, never to the id it
        # claims — see UNBOUND.
        charge = session_id if bound else UNBOUND
        with conn:
            used = _count(conn, op_cls=op_cls, scope=scope, source=source,
                          session_id=charge)
            if ceiling is None:
                # Uncapped class: recorded so the ledger is complete and the
                # trip wire has a denominator, but never refused.
                _record(conn, tool=tool, op_cls=op_cls, scope=scope,
                        source=source, session_id=charge,
                        effect_scope=effect_scope, bound=bound, allowed=True,
                        ceiling=None, used_after=used + 1, now=now)
                return Decision(True, tool, op_cls, scope, source,
                                int(1e9), used + 1, window)
            if used >= ceiling:
                _record(conn, tool=tool, op_cls=op_cls, scope=scope,
                        source=source, session_id=charge,
                        effect_scope=effect_scope, bound=bound, allowed=False,
                        ceiling=ceiling, used_after=used, now=now)
                return Decision(False, tool, op_cls, scope, source, ceiling,
                                used, window)
            _record(conn, tool=tool, op_cls=op_cls, scope=scope, source=source,
                    session_id=charge, effect_scope=effect_scope, bound=bound,
                    allowed=True, ceiling=ceiling, used_after=used + 1, now=now)
            _maybe_prune(conn, now)
            _trip_wire(conn, cfg, source, op_cls, scope, _hour_bucket(now))
            return Decision(True, tool, op_cls, scope, source, ceiling,
                            used + 1, window)
    except Exception as exc:
        logger.warning("mutation budget: decision for %s (%s/%s) failed: %s —"
                       " dispatching uncapped (fail-open)", tool, op_cls, scope,
                       exc)
        return Decision(True, tool, op_cls, scope, source, int(1e9), 0, window)
    finally:
        conn.close()


async def guard(name: str, arguments: Any, *, session_id: str,
                effect_scope: str = "", bound: bool = True) -> Decision:
    """The one entry point `call_tool` calls, and the whole gate.

    `bound` says whether the session id came from `_meta` (True, every real
    harness call) or only from an argument (False — charged to `UNBOUND`,
    evaluated against the interactive table).
    """
    op_cls = op_class(name, arguments)
    if op_cls is None or not enabled():
        return Decision(True, name, "", "", "", int(1e9), 0,
                        int(config()["window_seconds"]))
    scope = target_scope(name, arguments)
    source = run_source(session_id, effect_scope, bound=bound)
    return await asyncio.to_thread(
        _decide_sync, name, op_cls, scope, source, session_id, effect_scope,
        bound)


# ── readers ─────────────────────────────────────────────────────────────────

def run_mutations(effect_scope: str, *, denied: bool = False) -> dict[str, int]:
    """Per-op-class counts for one run, keyed for `runs.mutations_json`.

    Empty scope, empty answer — an interactive turn has no run to report on.
    Denied calls are excluded by default, because the number a reviewer wants is
    the number of effects and a deny is a call that had none; pass
    `denied=True` to see the calls the ceiling refused, which is what a
    calibration pass reads: a run whose denied count is non-zero was a run that
    *tried* to exceed the ceiling, and that is the number that decides whether
    a source's ceiling is set right.
    """
    if not effect_scope:
        return {}
    conn = _connect(create=False)
    if conn is None:
        return {}
    try:
        _ensure(conn)
        rows = conn.execute(
            "SELECT op_class, COUNT(*) AS n FROM mutation_calls WHERE"
            " effect_scope=? AND allowed=? GROUP BY op_class ORDER BY n DESC",
            (effect_scope, 0 if denied else 1)).fetchall()
        return {r["op_class"]: int(r["n"]) for r in rows}
    except Exception as exc:
        logger.warning("mutation budget: run counters for %r unavailable: %s",
                       effect_scope, exc)
        return {}
    finally:
        conn.close()


def recent_alerts(limit: int = 20) -> list[dict]:
    """Trip-wire rows, newest first — the aggregate alert, alerting only."""
    conn = _connect(create=False)
    if conn is None:
        return []
    try:
        _ensure(conn)
        return [dict(r) for r in conn.execute(
            "SELECT hour_bucket, at, source, op_class, target_scope, count,"
            " ratio FROM mutation_alerts ORDER BY id DESC LIMIT ?",
            (max(1, min(200, limit)),)).fetchall()]
    except Exception as exc:
        logger.warning("mutation budget: alerts unavailable: %s", exc)
        return []
    finally:
        conn.close()


def status(hours: int = 3) -> dict:
    """Per-source per-class counts over the last N hours, plus the ceilings.

    This is the surface the human calibration decision reads: which source is
    nearest its ceiling, and which denied. Not a model tool on purpose — the
    override has to be a person editing `harness.mutation_budget`, per the item:
    a model-reachable "raise the ceiling" call would be this round's own
    `skipReview` bug.
    """
    conn = _connect(create=False)
    cfg = config()
    out: dict[str, Any] = {"window_hours": hours, "sources": [],
                           "trip_wires": recent_alerts(20)}
    if conn is None:
        out["error"] = "counter database unavailable"
        return out
    try:
        _ensure(conn)
        bucket_hi = _hour_bucket(clock())
        bucket_lo = bucket_hi - max(0, min(168, int(hours))) + 1
        rows = conn.execute(
            "SELECT source, op_class, target_scope, COUNT(*) AS n"
            " FROM mutation_calls WHERE hour_bucket>=? AND hour_bucket<=?"
            " AND allowed=1 GROUP BY source, op_class, target_scope"
            " ORDER BY source, n DESC", (bucket_lo, bucket_hi)).fetchall()
        denials = dict(conn.execute(
            "SELECT source, COUNT(*) FROM mutation_calls WHERE hour_bucket>=?"
            " AND hour_bucket<=? AND allowed=0 GROUP BY source",
            (bucket_lo, bucket_hi)).fetchall())
        by_source: dict[str, list[dict]] = {}
        for r in rows:
            by_source.setdefault(r["source"], []).append({
                "class": r["op_class"], "scope": r["target_scope"],
                "allowed": int(r["n"]),
                "ceiling": ceiling_for(r["op_class"], r["source"],
                                       r["target_scope"]),
            })
        for source in sorted(set(by_source) | set(denials)):
            out["sources"].append({
                "source": source,
                "denied": int(denials.get(source, 0)),
                "classes": by_source.get(source, []),
            })
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        conn.close()
    return out
