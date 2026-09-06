"""Poison sweep — periodic triage of the work queue's dead letters.

An item that fails `workers.max_attempts` times lands in state `poisoned` and
stays there forever: nothing in the pool ever looks at a poisoned row again.
That is the right default for a *queue* (a bad item must not spin) and the
wrong default for a *fleet* — the poisoned pile is the only place a systemic
failure surfaces, and an unbounded pile of untriaged rows is indistinguishable
from a healthy one.

**It runs from the pool's scheduler loop, not as a work source.** A source that
repairs the queue has to be claimed by a free worker slot, and there are two of
them holding jobs for up to an hour — it would be starved exactly when the
queue is backed up. The scheduler loop is a separate asyncio task on a 60s
tick that no amount of queue backlog can block.

**It is deterministic — no model call.** Autonomy task #76 (Queue Health
Check) is the model-driven analyst layered on top, and its own activity log is
the argument for keeping a deterministic floor underneath it: it timed out at
600s on three of its last six runs, because reaching a model needs the primary
engine, a free worker slot and a healthy queue — the three things in doubt when
items are poisoning.

What a sweep does with each poisoned row:

  transient   a timeout, a dropped connection, a wedged engine. The item is
              probably fine and the world was not. **Revive** it for exactly
              one more claim, after a delay.
  structural  a bad payload, an unknown source, a KeyError. Retrying is pure
              waste and the row needs a human. **Quarantine** it.

An unrecognised error is treated as structural. Retrying an error nobody has
classified spends GPU-hours on a guess; quarantining it costs a line in a
report.

Three things override a revive, all for the same reason — #76's own log records
it as "a weekly reset that does not fix the cause just re-poisons":

  - the item's revive budget (`max_revives`) is spent,
  - the (source, signature) pair has poisoned `repeat_threshold` times across
    sweeps, which is a broken cause rather than a transient one,
  - an equivalent item is already open, because a poisoned row has lost its
    dedup_key and reviving it would duplicate work the source has re-enqueued.

`quarantined` is a distinct terminal state rather than a flag on `poisoned`, so
the dashboard's `poisoned_total` keeps meaning "needs a human" instead of
slowly turning into a lifetime counter nobody reads.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from workers.queue import WorkQueue, new_run_id

logger = logging.getLogger("lloyd-workers.maintenance")

#: Watermark namespace and the source name written to the `runs` table. It is
#: deliberately not a registered work source — nothing enqueues into it.
SOURCE = "queue-maintenance"

_TALLY_PREFIX = "sig:"
_LAST_SWEEP_AT = "last_sweep_at"
_LAST_SWEEP = "last_sweep"

DEFAULTS = {
    "enabled": True,
    "interval_seconds": 900,
    "max_revives": 1,
    "revive_delay_seconds": 300,
    "repeat_threshold": 3,
    "tally_retention_days": 14,
    "scan_limit": 200,
}


# ── Error signatures ──────────────────────────────────────────────────────

# Order matters: paths before hex before digits, or a path's digits are
# rewritten first and the path stops matching.
_SIG_SUBS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?:/[\w.\-]+){2,}/?"), "<path>"),
    (re.compile(r"\b[0-9a-f]{8,}\b", re.IGNORECASE), "<id>"),
    (re.compile(r"\b\d+\b"), "<n>"),
    (re.compile(r"\s+"), " "),
]


def signature(error: Optional[str]) -> str:
    """Collapse an error string to a stable grouping key.

    `exceeded max_duration_seconds=600` and `...=1800` are the same failure and
    must tally together, or a source whose timeout was tuned once looks like
    two unrelated one-off problems.
    """
    text = (error or "").strip()
    if not text:
        return "(no error recorded)"
    text = text.splitlines()[0][:400]
    for pattern, repl in _SIG_SUBS:
        text = pattern.sub(repl, text)
    return text.strip()[:200]


# A failure of the world, not of the item: worth exactly one more try.
_TRANSIENT = re.compile(
    r"""(?ix)
    \b(?:
        TimeoutError | TimedOut | ReadTimeout | WriteTimeout | PoolTimeout
      | ConnectTimeout | CancelledError | StreamStalledError
      | ConnectError | ConnectionError | ConnectionReset | ConnectionRefused
      | ConnectionAborted | RemoteProtocolError | ReadError | WriteError
      | ProtocolError | IncompleteRead | ChunkedEncodingError
      | ServiceUnavailable | BadGateway | GatewayTimeout | TooManyRequests
      | OverloadedError | RateLimit\w* | OSError
    )\b
    | \b(?:429|500|502|503|504)\b
    | server\s+disconnected
    | connection\s+(?:refused|reset|closed|aborted)
    | temporarily\s+unavailable
    | all\s+connection\s+attempts\s+failed
    | (?:engine|model|server)\s+(?:is\s+)?(?:not\s+ready|unavailable|overloaded)
    """
)

# A failure of the item: retrying reproduces it exactly. Checked first, because
# a message can name both (`TypeError` raised while handling a timeout) and the
# deterministic half is the one that decides.
_STRUCTURAL = re.compile(
    r"""(?ix)
    \b(?:
        unknown\s+source
      | KeyError | IndexError | AttributeError | TypeError | NameError
      | ImportError | ModuleNotFoundError | JSONDecodeError | UnicodeDecodeError
      | ValidationError | AssertionError | NotImplementedError
      | FileNotFoundError | NotADirectoryError | IsADirectoryError
      | PermissionError | IntegrityError | OperationalError
    )\b
    """
)


def classify(error: Optional[str]) -> str:
    """`"transient"` or `"structural"` for one error string."""
    text = error or ""
    if _STRUCTURAL.search(text):
        return "structural"
    if _TRANSIENT.search(text):
        return "transient"
    return "structural"


# ── Cross-sweep signature tally ───────────────────────────────────────────


def _tally_key(source: str, sig: str) -> str:
    digest = hashlib.sha1(f"{source}|{sig}".encode("utf-8")).hexdigest()[:12]
    return f"{_TALLY_PREFIX}{digest}"


def _load_tally(queue: WorkQueue, key: str) -> Optional[dict]:
    raw = queue.wm_get(SOURCE, key)
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _prune_tallies(queue: WorkQueue, retention_days: int, now: datetime) -> int:
    """Forget signatures nothing has produced lately.

    Without this the tally is a lifetime counter, and a source that poisoned
    three times last spring would be permanently barred from ever being
    revived again.
    """
    cutoff = now - timedelta(days=max(1, retention_days))
    dropped = 0
    for key, raw in queue.wm_all(SOURCE).items():
        if not key.startswith(_TALLY_PREFIX):
            continue
        try:
            last_seen = datetime.fromisoformat((json.loads(raw) or {}).get("last_seen", ""))
        except (TypeError, ValueError):
            queue.wm_delete(SOURCE, key)
            dropped += 1
            continue
        if last_seen < cutoff:
            queue.wm_delete(SOURCE, key)
            dropped += 1
    return dropped


# ── The sweep ─────────────────────────────────────────────────────────────


def sweep(
    queue: WorkQueue,
    *,
    max_attempts: int = 3,
    max_revives: int = 1,
    revive_delay_seconds: float = 300,
    repeat_threshold: int = 3,
    tally_retention_days: int = 14,
    scan_limit: int = 200,
    now: Optional[datetime] = None,
) -> dict:
    """Triage every poisoned item once. Returns a report dict.

    Pure queue work — no model, no network, no side effects beyond the queue
    and its watermarks. `run_sweep` wraps this with reporting.
    """
    now = now or datetime.now(timezone.utc)
    items = queue.list_items(state="poisoned", limit=scan_limit)

    actions: list[dict] = []
    escalations: list[dict] = []
    revived = quarantined = 0

    for item in items:
        sig = signature(item.error)
        failure_class = classify(item.error)
        key = _tally_key(item.source, sig)

        tally = _load_tally(queue, key) or {
            "source": item.source,
            "signature": sig,
            "count": 0,
            "first_seen": now.isoformat(),
        }
        tally["count"] = int(tally.get("count", 0)) + 1
        tally["last_seen"] = now.isoformat()
        tally["last_error"] = (item.error or "")[:500]
        queue.wm_set(SOURCE, key, json.dumps(tally, default=str))

        revives = int((item.triage or {}).get("revives", 0))
        repeat = tally["count"] >= repeat_threshold

        # Reasons not to revive, most informative first.
        if failure_class == "structural":
            reason = "not retryable — the item reproduces this failure"
        elif repeat:
            reason = (
                f"{tally['count']} poisonings on this signature in "
                f"{tally_retention_days}d — the cause is not transient"
            )
        elif revives >= max_revives:
            reason = f"transient, but already revived {revives}x"
        elif queue.has_open_sibling(item.source, item.kind, item.payload, item.id):
            reason = "an equivalent item is already open — reviving would duplicate it"
        else:
            reason = ""

        triage = {
            "at": now.isoformat(),
            "signature": sig,
            "class": failure_class,
            "occurrences": tally["count"],
            "error": (item.error or "")[:500],
        }

        if reason:
            triage.update({"action": "quarantined", "reason": reason, "revives": revives})
            ok = queue.quarantine(item.id, triage)
            action = "quarantined"
        else:
            triage.update({
                "action": "revived",
                "reason": "transient failure — one more claim",
                "revives": revives + 1,
            })
            # One more claim, not a fresh budget: claim_next increments
            # attempts, so this lands exactly on max_attempts and a second
            # failure re-poisons immediately.
            ok = queue.revive(
                item.id,
                attempts=max(0, max_attempts - 1),
                delay_seconds=revive_delay_seconds,
                triage=triage,
            )
            action = "revived"

        if not ok:
            # Lost a race with a worker or another sweep. The row is no longer
            # poisoned, which is the outcome we wanted anyway.
            logger.debug("Item %d was no longer poisoned — skipped", item.id)
            continue

        if action == "revived":
            revived += 1
        else:
            quarantined += 1
            if repeat:
                # Either class: a transient cause recurring is not transient,
                # and a structural one recurring is a source producing broken
                # items. Both want a human, neither wants another retry.
                escalations.append({
                    "source": item.source,
                    "signature": sig,
                    "count": tally["count"],
                    "class": failure_class,
                })

        actions.append({
            "item_id": item.id,
            "source": item.source,
            "kind": item.kind,
            "attempts": item.attempts,
            "signature": sig,
            "class": failure_class,
            "action": action,
            "reason": triage["reason"],
            "occurrences": tally["count"],
        })

    pruned = _prune_tallies(queue, tally_retention_days, now)

    return {
        "at": now.isoformat(),
        "scanned": len(items),
        "revived": revived,
        "quarantined": quarantined,
        "tallies_pruned": pruned,
        "actions": actions,
        "escalations": escalations,
    }


# ── Reporting ─────────────────────────────────────────────────────────────


def report_dir() -> Path:
    from app.paths import AUTONOMY_RUNS_DIR

    return AUTONOMY_RUNS_DIR / SOURCE


def _render_report(report: dict) -> str:
    lines = [
        "---",
        f"at: '{report['at']}'",
        f"scanned: {report['scanned']}",
        f"revived: {report['revived']}",
        f"quarantined: {report['quarantined']}",
        f"escalations: {len(report['escalations'])}",
        "---",
        "",
        "# Poison sweep",
        "",
    ]
    if report["escalations"]:
        lines.append("## Escalations — a transient-looking cause that keeps recurring")
        lines.append("")
        for esc in report["escalations"]:
            lines.append(
                f"- **{esc['source']}** — {esc['count']}x `{esc['signature']}`"
            )
        lines.append("")
    lines.append("## Actions")
    lines.append("")
    lines.append("| item | source/kind | class | action | why | seen |")
    lines.append("|---|---|---|---|---|---|")
    for a in report["actions"]:
        lines.append(
            f"| {a['item_id']} | {a['source']}/{a['kind']} | {a['class']} | "
            f"{a['action']} | {a['reason']} | {a['occurrences']}x |"
        )
    lines.append("")
    lines.append("## Signatures")
    lines.append("")
    for a in report["actions"]:
        lines.append(f"- `{a['signature']}`")
    return "\n".join(lines) + "\n"


def write_report(report: dict) -> Optional[str]:
    """Write the sweep's triage note. Returns the path, or None on failure.

    A sweep that cannot write its note has still done the useful half of the
    job — the queue is repaired either way — so this never raises.
    """
    try:
        directory = report_dir()
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.fromisoformat(report["at"]).strftime("%Y%m%d_%H%M%S")
        path = directory / f"sweep_{stamp}.md"
        path.write_text(_render_report(report), encoding="utf-8")
        return str(path)
    except Exception as exc:  # noqa: BLE001 — reporting must not break repair
        logger.warning("Poison sweep report not written: %s", exc)
        return None


def run_sweep(queue: WorkQueue, cfg: dict[str, Any], max_attempts: int = 3) -> dict:
    """Sweep, then persist what happened. Safe to call on every scheduler tick.

    A sweep that changed nothing records nothing but its timestamp: at a 15
    minute cadence a run row per tick would be 96 rows a day of "no poisoned
    items", burying the handful that mean something.
    """
    settings = {**DEFAULTS, **(cfg or {})}
    started = datetime.now(timezone.utc)

    report = sweep(
        queue,
        max_attempts=max_attempts,
        max_revives=int(settings["max_revives"]),
        revive_delay_seconds=float(settings["revive_delay_seconds"]),
        repeat_threshold=int(settings["repeat_threshold"]),
        tally_retention_days=int(settings["tally_retention_days"]),
        scan_limit=int(settings["scan_limit"]),
        now=started,
    )

    acted = report["revived"] + report["quarantined"]
    summary = (
        f"swept {report['scanned']} poisoned: "
        f"{report['revived']} revived, {report['quarantined']} quarantined"
    )
    if report["escalations"]:
        summary += f"; {len(report['escalations'])} recurring"

    if acted:
        report["report_path"] = write_report(report)
        completed = datetime.now(timezone.utc)
        queue.record_run(
            run_id=new_run_id(SOURCE),
            queue_id=None,
            source=SOURCE,
            status="success",
            started_at=started.isoformat(),
            completed_at=completed.isoformat(),
            duration_seconds=(completed - started).total_seconds(),
            summary=summary,
            artifact_path=report.get("report_path") or "",
            meta_json=json.dumps(
                {k: report[k] for k in ("revived", "quarantined", "escalations")},
                default=str,
            ),
        )
        for esc in report["escalations"]:
            logger.error(
                "Poison sweep escalation: %s has poisoned %dx on a %s failure "
                "`%s` — quarantined rather than retried; the cause needs fixing",
                esc["source"], esc["count"], esc.get("class", "?"), esc["signature"],
            )
        logger.info("Poison sweep: %s", summary)

    queue.wm_set(SOURCE, _LAST_SWEEP_AT, started.isoformat())
    queue.wm_set(SOURCE, _LAST_SWEEP, json.dumps({
        "at": report["at"],
        "scanned": report["scanned"],
        "revived": report["revived"],
        "quarantined": report["quarantined"],
        "escalations": report["escalations"],
        "report_path": report.get("report_path"),
    }, default=str))
    return report


def last_sweep(queue: WorkQueue) -> Optional[dict]:
    """The previous sweep's summary, for the dashboard. None if never run."""
    raw = queue.wm_get(SOURCE, _LAST_SWEEP)
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None
