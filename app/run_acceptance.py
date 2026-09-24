"""Grade an autonomy run against the acceptance its task declares (#623).

A successful autonomy run is recorded `status="success"` as a literal the
moment its turn returns text; nothing re-derives it. This module is the
independent reading beside that literal: a task may declare

    acceptance:
      objective_checks:
        - {type: tool_called, value: Bash}
        - {type: regex, value: "wrote \\d+ entries"}
      rubric: [accuracy]          # recorded, not scored (see below)

and the finished run is graded by `scripts/autoresearch/judge.py`'s objective
layer against the run's own dispatch record — the tool calls the harness
actually dispatched, collected off the event stream by `DispatchTrace` — never
against what the final text says it did. The trace is stamped
`tool_trace_authoritative`, so "I ran Bash" with no Bash dispatch fails
`tool_called: Bash`.

**Record only.** The grade goes onto the run's `meta` (so into `runs.meta_json`
and the run record's front matter) and changes nothing else: not the status,
not the retry, not the task's schedule. Whether a grade may ever gate a run is
a decision for after the grades have been read and hand-checked; the item's
human clauses own it.

**No model call.** Only the deterministic layer runs. A declared `rubric` is
carried onto the grade as `rubric_ungraded` so the ledger says what was not
judged; calling the rubric judge would put an engine call on every pilot run
before anyone has shown the grade is worth one.

Grades:

- `graded_pass` / `graded_fail` — every measurable check passed / one did not.
- `no_acceptance` — the task declares no block. Never counted as a pass.
- `acceptance_invalid` — a block that does not parse into judge check types.
- `not_measurable` — the block declares checks and none could be measured.
- `grader_error` — grading itself raised. Recorded, never a pass.

A task with `grader: false` is not graded at all and its run carries no grade
key — the row it writes is the row it wrote before this module existed.
"""

from __future__ import annotations

import datetime
import json
import logging
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)

# At import, not per call: `DispatchTrace.observe` runs inside a run's event
# loop, and the bench runner's first import costs ~0.7 s — which, taken there,
# would stall every HTTP request and chat stream the backend is serving. A
# failed import is kept rather than raised, so `autonomy` still imports and
# the run records `grader_error` instead of a grade.
try:
    from scripts.autoresearch.bench_runner_sdk import _classify_result
    from scripts.autoresearch.judge import CHECK_TYPES, _score_objective
    _IMPORT_ERROR: Optional[str] = None
except Exception as _e:  # pragma: no cover - exercised only on a broken tree
    _classify_result = CHECK_TYPES = _score_objective = None
    _IMPORT_ERROR = f"{type(_e).__name__}: {_e}"[:300]

#: The key the grade rides under, on the run's `meta` and on the result dict
#: the scheduled-task source hands the pool.
GRADE_KEY = "acceptance_grade"

GRADED = ("graded_pass", "graded_fail")


def grading_enabled(task: dict) -> bool:
    """`grader: false` in the task's front matter switches grading off."""
    raw = task.get("grader", True)
    if isinstance(raw, str):
        return raw.strip().lower() not in ("false", "no", "off", "0")
    return raw is not False


def parse_acceptance(task: dict) -> Optional[dict]:
    """The task's `acceptance:` block as `{checks, rubric, invalid}`, or None.

    A bare list is read as the objective checks. Check types are validated
    against the judge's own `CHECK_TYPES` rather than a list kept here — the
    judge fails closed on a type it does not know, so a type it would silently
    grade False must be refused where a person can see it.
    """
    raw = task.get("acceptance")
    if raw is None or raw == "" or raw == [] or raw == {}:
        return None
    if isinstance(raw, list):
        raw = {"objective_checks": raw}
    if not isinstance(raw, dict):
        return {"checks": [], "rubric": [], "invalid": [repr(raw)[:200]]}
    checks: list[dict] = []
    invalid: list[str] = []
    for c in raw.get("objective_checks") or []:
        if (isinstance(c, dict) and c.get("type") in CHECK_TYPES
                and c.get("value") is not None):
            checks.append({"type": c["type"], "value": c["value"]})
        else:
            invalid.append(repr(c)[:200])
    rubric = raw.get("rubric") or []
    if not isinstance(rubric, list):
        rubric = [rubric]
    return {"checks": checks, "rubric": [str(r) for r in rubric], "invalid": invalid}


class DispatchTrace:
    """The run's tool use, read off the harness event stream.

    A `tool_call` event is emitted for every call the model made, including
    ones a hook then refused, so a call is filed only when its `tool_result`
    arrives: refused ones go to `denied_calls`, classified by the bench
    runner's own `_classify_result` rather than a second list of deny markers.
    A call with no result (the run ended mid-dispatch) is counted as made.

    `observe` sits inside the run's own event loop, so it never raises: a
    failure is kept and the grade becomes `grader_error` instead.
    """

    def __init__(self) -> None:
        self.tool_calls: list[dict] = []
        self.denied_calls: list[dict] = []
        self._pending: dict[str, str] = {}
        self.error: Optional[str] = None

    def observe(self, evt: dict) -> None:
        try:
            self._observe(evt)
        except Exception as e:
            self.error = self.error or f"{type(e).__name__}: {e}"[:300]

    def _observe(self, evt: dict) -> None:
        etype = evt.get("type")
        if etype == "tool_call":
            self._pending[evt.get("call_id", "")] = evt.get("name", "")
        elif etype == "tool_result":
            name = evt.get("name") or self._pending.get(evt.get("call_id", ""), "")
            self._pending.pop(evt.get("call_id", ""), None)
            content = evt.get("content")
            if isinstance(content, list):
                content = " ".join(
                    str(b.get("text", b)) if isinstance(b, dict) else str(b)
                    for b in content)
            kind, _ = _classify_result(str(content or ""))
            if kind:
                self.denied_calls.append({"name": name, "deny_kind": kind})
            else:
                self.tool_calls.append({"name": name})

    def as_trace(self, final_text: str) -> dict:
        unanswered = [{"name": n} for n in self._pending.values()]
        return {"final_text": final_text or "",
                "tool_calls": self.tool_calls + unanswered,
                "denied_calls": list(self.denied_calls),
                "tool_trace_authoritative": True,
                "trace_error": self.error}


def grade_run(task: dict, trace: dict) -> Optional[dict]:
    """The grade for one finished run, or None when grading is switched off."""
    if not grading_enabled(task):
        return None
    try:
        if _IMPORT_ERROR:
            raise RuntimeError(f"judge unavailable: {_IMPORT_ERROR}")
        if trace.get("trace_error"):
            raise RuntimeError(f"dispatch trace incomplete: {trace['trace_error']}")
        spec = parse_acceptance(task)
        if spec is None:
            return {"grade": "no_acceptance"}
        if spec["invalid"] or not spec["checks"]:
            return {"grade": "acceptance_invalid", "invalid": spec["invalid"],
                    "rubric_ungraded": spec["rubric"]}
        score, results = _score_objective(
            {"id": task.get("id"), "objective_checks": spec["checks"]}, trace)
        if score is None:
            grade = "not_measurable"
        else:
            grade = "graded_pass" if all(
                r.get("passed") for r in results if r.get("measured")) else "graded_fail"
        return {"grade": grade, "score": score,
                "checks": [{"type": r.get("type"), "value": r.get("value"),
                            "passed": r.get("passed")} for r in results],
                "rubric_ungraded": spec["rubric"]}
    except Exception as e:  # grading must never cost the run its record
        logger.warning("acceptance grading failed for task %s: %s", task.get("id"), e)
        return {"grade": "grader_error", "error": f"{type(e).__name__}: {e}"[:300]}


def false_completion_rate(conn: sqlite3.Connection, *, since: Optional[str] = None,
                          days: float = 7.0, by: str = "source") -> dict[str, Optional[float]]:
    """Per source, the share of `status='success'` runs graded `graded_fail`.

    `by="task_id"` splits it per task instead: every autonomy run shares the
    one `scheduled-task` source, and a pilot is read task by task.

    Only `graded_pass`/`graded_fail` rows are in the denominator: a run with
    no acceptance, or one whose checks could not be measured, says nothing
    about whether its success was real. A source with success runs but no
    graded ones maps to None — "0% false completions" for a source nobody
    graded is the reading this exists to prevent.
    """
    if since is None:
        since = (datetime.datetime.now(datetime.timezone.utc)
                 - datetime.timedelta(days=days)).isoformat()
    if by not in ("source", "task_id"):
        raise ValueError(f"by must be 'source' or 'task_id', not {by!r}")
    rows = conn.execute(
        f"SELECT {by}, meta_json FROM runs WHERE status = 'success' "
        "AND started_at >= ?", (since,)).fetchall()
    tally: dict[str, list[int]] = {}
    for key, meta_json in rows:
        counts = tally.setdefault(str(key), [0, 0])
        try:
            grade = (json.loads(meta_json or "{}").get(GRADE_KEY) or {}).get("grade")
        except (ValueError, AttributeError):
            continue
        if grade in GRADED:
            counts[0] += 1
            counts[1] += grade == "graded_fail"
    return {s: (fail / graded if graded else None) for s, (graded, fail) in tally.items()}
