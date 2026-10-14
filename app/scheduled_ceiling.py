"""The declared scheduled-task ceiling, wired into the live turn in shadow mode (#2670).

`app/harness/capabilities.py` declared a 33-name ceiling for `scheduled-task` on
2026-10-09 (#2471) but deliberately stopped short of the one turn that runs it:
`app/autonomy.py::run_task` predates the envelope machinery and assembles its own
`RunOptions` off `_get_disallowed_tools()` + the grant-mint ban, so no scheduled
task has ever been under that ceiling and the seven-day check #2471 owes had no
denominator. This module is the wiring, in the one mode that cannot break a job:

* **shadow** (the default, and what this ships in) — the ceiling is COMPILED, and
  the turn is dispatched exactly as it was yesterday: `allowed_tools=None`, so
  the harness's only tool filter stays the deny floor. What the ceiling would
  have refused is written to a durable ledger instead of being enforced.
* **enforce** — `allowed_tools` is the compiled envelope. That is a later commit
  (#2670's owed entry 1), reachable here only by an operator setting
  `LLOYD_SCHEDULED_CEILING_MODE=enforce`, which is recorded on every ledger row
  so a flip that happens outside a review still shows up in the evidence.

Why shadow mode dispatches `None` rather than an allow-list equal to the deny
floor's reachable set: the harness folds `allowed_tools` into the catalog hidden
from the turn (`loop._allow_list_hidden`) and re-derives the dispatch set each
iteration, so an explicit list is a SNAPSHOT. A tool loaded mid-turn by
`tool_search` would be absent from the snapshot and therefore hidden — the exact
"job quietly lost a tool" failure the ceiling work exists to avoid, manufactured
by the shadow pass itself. `None` is byte-identical to yesterday's compile.

The ledger exists because shadow leaves no trace anywhere else. A scheduled
task's transcript records WHICH tools it called but nothing marks which of those
names fell outside the envelope — nothing refused them, so the transcript of a
shadow turn looks identical to one under no ceiling at all. Enforcement needs a
per-name would-strip rate, and that rate is only computable from this file.

The one thing that makes the ceiling safe to enforce at all — the ban name a
per-job scope may be licensed for, so the nightly vault-landing jobs are never
counted as would-stripped — is NOT declared here. It is
`capabilities.SCOPE_BAN_EXCEPTIONS`, next to the widening rule that consults it,
because it is a property of what an envelope may contain and this module is only
one of the two callers that could ever ask for it.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("lloyd.scheduled_ceiling")

SOURCE = "scheduled-task"

MODE_ENV = "LLOYD_SCHEDULED_CEILING_MODE"
LEDGER_ENV = "LLOYD_SCHEDULED_CEILING_LEDGER"

#: The two modes. Anything else in the env var is a typo, and a typo must not
#: silently become enforcement, so an unrecognised value is reported and reads
#: as shadow.
MODE_SHADOW = "shadow"
MODE_ENFORCE = "enforce"


def mode() -> str:
    """`shadow` unless an operator said `enforce`, with the switch named in the row."""
    raw = str(os.environ.get(MODE_ENV) or "").strip().lower()
    return MODE_ENFORCE if raw == MODE_ENFORCE else MODE_SHADOW


def ledger_path() -> Path:
    """`DATA_ROOT/scheduling/scheduled-ceiling.jsonl`, resolved per call.

    Under the data root rather than beside the code, for the reason every other
    witness here has (#2213, #1799): an observation about a run that lives
    inside the tree the run edits is destroyed by whatever destroyed the run.
    """
    raw = str(os.environ.get(LEDGER_ENV) or "").strip()
    if raw:
        return Path(os.path.expanduser(raw))
    from app.paths import DATA_ROOT
    return DATA_ROOT / "scheduling" / "scheduled-ceiling.jsonl"


def read_rows(path: Path | None = None) -> list[dict]:
    """Every readable row, oldest first. A malformed line is skipped, not fatal.

    This is what the owed-check job reads to decide whether enforcement may
    start, so it answers with the rows that exist rather than raising over a
    half-written last line — a truncated tail is normal for an append-only file.
    """
    p = path or ledger_path()
    rows: list[dict] = []
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def would_strip(called_names, envelope) -> list[str]:
    """Which names the turn reached for that the declared ceiling does not contain.

    Normalised on both sides (`mcp__lloyd-mcp__Bash` and `Bash` are one tool),
    de-duplicated, sorted — so a row's name list is a set-valued measurement and
    a loop that called the same tool eleven times contributes one entry.
    """
    allowed = {str(n) for n in (envelope or ())}
    from app.harness.policy import normalize_tool_name
    return sorted({normalize_tool_name(n) for n in (called_names or ())
                   if normalize_tool_name(n)} - allowed)


class TurnSurface:
    """One scheduled-task turn's tool surface: what it is dispatched under, and its witness.

    Built before `RunOptions` so the turn's dispatch set and the row it later
    writes come from one compile. `allowed_tools` is what goes on the options;
    `envelope` is what the ceiling says, which in shadow mode is deliberately
    NOT what the turn is dispatched under.
    """

    def __init__(self, *, mode: str, grant_scope: str, task_id,
                 allowed_tools: list[str] | None,
                 envelope: list[str] | None,
                 compile_error: str = "") -> None:
        self.mode = mode
        self.grant_scope = grant_scope
        self.task_id = task_id
        self.allowed_tools = allowed_tools
        self.envelope = envelope
        #: Non-empty when the ceiling could not be compiled at all. The row then
        #: carries `compile_error` and NO `would_strip` key: an empty list means
        #: "measured, and every call was inside the envelope", and a metric that
        #: reads its own missing input as a zero is the failure mode this box has
        #: been burned on repeatedly.
        self.compile_error = compile_error
        self._written: dict | None = None

    def witness(self, called_names, *, run_id: str = "",
                now: datetime.datetime | None = None) -> dict | None:
        """Append this turn's ONE row. Idempotent, and never raises into the run.

        Exactly one row per turn is the row's whole contract: the owed-check job
        divides would-strip counts by the row count, so a double-write halves
        every rate and a missing write silently removes a turn from the
        denominator. The `_written` guard is what makes the `finally` at the run's
        bottom safe to call from more than one exit path, and the catch is what
        keeps a dead data root from failing a job that already finished its work.
        """
        if self._written is not None:
            return self._written
        row = None
        try:
            ts = now or datetime.datetime.now(datetime.timezone.utc)
            utc = ts.astimezone(datetime.timezone.utc)
            row = {
                "ts": utc.isoformat(),
                "date": utc.date().isoformat(),
                "mode": self.mode,
                "task_id": self.task_id,
                "run_id": run_id,
                "grant_scope": self.grant_scope,
                "n_calls": len({str(n) for n in (called_names or ())}),
                "envelope_size": None if self.envelope is None
                else len(self.envelope),
            }
            if self.compile_error:
                row["compile_error"] = self.compile_error
            else:
                stripped = would_strip(called_names, self.envelope)
                row["would_strip"] = stripped
                row["n_would_strip"] = len(stripped)
            path = ledger_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            self._written = row
        except Exception as exc:  # noqa: BLE001 — a witness must never cost a run
            logger.warning("could not write the scheduled-ceiling shadow row for "
                           "task #%s run %s to %s: %s — this turn is missing from "
                           "the enforcement denominator",
                           self.task_id, run_id, ledger_path(), exc)
            return None
        return row


def compile_surface(*, grant_scope: str, task_id,
                    disallowed_tools) -> TurnSurface:
    """This turn's tool surface, compiled through `capabilities.envelope_for`.

    `disallowed_tools` is the deny floor the caller is already going to put on
    `RunOptions`, passed in so the ceiling is narrowed by it rather than
    contradicting it: `envelope_for`'s `extra_disallowed` argument exists for
    exactly this — a tool the operator switched off is not advertised even under
    a live grant, because no server is serving it and P3 would refuse it.

    The grant widening is read HERE, through `live_grant_names(store,
    grant_scope)`, because this is the layer that knows a run exists and which
    job it is; `grant_scope` is the same string the policy hook is armed with a
    few lines earlier in `run_task`, so the envelope, the grant gate and the
    outbound content gate all answer for one identity.

    A ceiling that cannot be compiled is handled by the mode, and the two modes
    deliberately do opposite things. Shadow catches: the point of shipping
    shadow is that no scheduled job can be broken by this code, so a failure is
    logged, the turn runs on today's compile, and its row says what broke.
    Enforce re-raises: an `enforce` operator who silently got no ceiling would
    be reading a policy that was not applied, which is worse than a run that
    fails and says why — that asymmetry is the reason the switch is off by
    default.
    """
    m = mode()
    try:
        from app.harness.capabilities import envelope_for, live_grant_names
        from app.harness.policy import default_store

        widening = live_grant_names(default_store(), grant_scope)
        envelope = envelope_for(SOURCE, disallowed_tools,
                                grant_widening=widening, scope=grant_scope)
        error = ""
    except Exception as exc:  # noqa: BLE001 — the mode decides who pays
        if m == MODE_ENFORCE:
            raise
        logger.warning("scheduled-task ceiling could not compile for %s (#%s): "
                       "%s: %s — running on today's deny floor; this turn's "
                       "ledger row records the failure and no would-strip "
                       "measurement", grant_scope, task_id,
                       type(exc).__name__, exc)
        envelope, error = None, f"{type(exc).__name__}: {exc}"[:300]
    # Shadow: dispatched set stays the deny floor's, which is what `None` means
    # to the harness — see the module docstring for why not a snapshot list.
    return TurnSurface(
        mode=m, grant_scope=grant_scope, task_id=task_id,
        allowed_tools=envelope if m == MODE_ENFORCE else None,
        envelope=envelope, compile_error=error)
