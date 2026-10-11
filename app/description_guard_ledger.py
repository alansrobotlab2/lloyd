"""One durable row per call to the #2408 description guard, in one bounded file.

`agent_mcp/autonomy.py::_stale_description_error` is the engine-side half of the rail
that refuses an autonomy task description quoting a tree constant at a value the tree
does not have (`01dea8bc` moved `LEDGER_ARCHIVE_AGE_DAYS` from 30 to 14 and task #79's
prose went on saying 30). It returns a refusal string or None, and until #2534 that
return was its only output — so the two states that matter most read identically to
every observer, including the one that has to answer "does this guard ever fire, and
does it fire correctly?": a description that resolved 8 pairs and agreed, and one that
resolved nothing and therefore could not disagree, both leave the function as None.
`scripts/automod/constant_quotes.py` was written to stop exactly that confusion on the
vault-side rail — `INSTRUMENT_FAILURE`, `Report.resolved`, `Report.is_silent`, and
`scan()`'s `pairs`/`silent` denominators all exist so a sweep cannot read "0
mismatches" as a clean sweep — and the engine half did the same work and threw the
result away. This module is where that half keeps it.

One row per call that reaches the resolver, appended to a JSONL file under the data
root:

    {"at": "...", "task": 79, "op": "update", "resolved": 2, "found": 1, "refused": true}

plus `"error"` when the resolver itself could not answer (`resolved`/`found` are then
null). A call whose description is blank never reaches the resolver and writes no row;
the guard answers None before importing anything. `counts()` turns the file into the
question that motivated it — `refused` / `passed` (compared at least one pair and
agreed) / `zero_resolved` (nothing comparable, so nothing stale either) — and
`instrument_error` for the calls whose resolver raised. Read it with:

    cd ~/lloyd && .venvs/lloyd/bin/python -m app.description_guard_ledger

Traffic is small and was measured before this existed, which is the reason for a
record rather than an inference: 38 task descriptions in the live vault resolve 10
pairs with 0 mismatches, and 34 `autonomy_write_task` calls across the session corpus
carry a non-empty description (2026-10-10, item #2534's triage). A retro-scan of those
descriptions against the current tree resolved 12 pairs and found 4 mismatches, so the
route has subject matter; whether the live guard blocks anything a writer legitimately
needs cannot be answered from a retro-scan, because the live guard judges the caller's
argument and never the record. That is the question this store answers going forward.

Fail-open, exactly like `app/harness/denial_journal.py` and for the same reason: a
witness that can turn its own failure into a refused write, or into an exception out of
the dispatcher, is worse than no witness — the write it would block may be the
`status: draft` park that stops a job. `record` never raises and never changes a
verdict; a lost row costs one warning line. The row is written *before* the refusal is
built, so a refusal is journaled whether or not the caller ever reads it.

Not a decision input. Nothing on the write path reads this file. Its readers are the
owed pass that asks the production question above, and whoever is deciding whether the
guard's threshold is wrong.

Bounded by itself: `_bound` keeps the newest rows that fit in `LOW_WATER_BYTES` once
the file passes `MAX_BYTES`, so the store cannot outgrow the sweep that has never heard
of it — the same reasoning `app/paths.py` records for `TOOL_QUALITY_DIR`. The rewrite
goes through `app.atomic_io.atomic_write_text`, so a reader can never see a half-trimmed
file (`app/atomic_io.py`'s own reason for existing).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("lloyd-description-guard-ledger")

LEDGER_NAME = "autonomy-description-guard.jsonl"
#: Where the ledger lives, for a test or a canary boot. Read on every call, so a
#: monkeypatched environment moves the writer and this module's own CLI together.
LEDGER_ENV = "LLOYD_DESCRIPTION_GUARD_LEDGER"

#: Trim once the file passes this size, and keep only the newest rows that fit in
#: LOW_WATER_BYTES. A row costs 123 bytes when it passed and 138 when it refused or
#: raised (measured over the schema below, the widest field being the `error` class
#: name), so the ceiling is ~470 rows deep — against the 14 calls carrying a
#: non-empty description that the ENTIRE 9,757-file session corpus held when #2534 was
#: triaged (2026-10-10), which is years of history at observed traffic, and small
#: enough that a trim is a 64 KiB rewrite rather than an IO event worth a lock.
MAX_BYTES = 64 * 1024
#: The low-water mark, deliberately below MAX_BYTES: trimming to the ceiling would put
#: the very next append back over it and rewrite the file on every call.
LOW_WATER_BYTES = MAX_BYTES // 2


def ledger_path() -> Path:
    """Where the ledger lives. Env override first, then the data root."""
    override = os.environ.get(LEDGER_ENV, "").strip()
    if override:
        return Path(override)
    try:
        from app.paths import AUTONOMY_DESCRIPTION_LEDGER_PATH
        return Path(AUTONOMY_DESCRIPTION_LEDGER_PATH)
    except Exception:  # noqa: BLE001 — same fail-open as the denial journal
        return Path(os.path.expanduser(f"~/.cache/lloyd/{LEDGER_NAME}"))


def _append(target: Path, line: str) -> None:
    """One whole-line append. Its own function because it is the sink's only syscall."""
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line)


def record(*, task_id: Any = 0, op: str = "", resolved: int | None = 0,
           found: int | None = 0, refused: bool = False,
           resolver_error: str = "", path: Path | None = None) -> bool:
    """Append one row. Returns True when the row was written. Never raises.

    `resolved=None` with `resolver_error` set is the resolver-raised shape: the call
    reached the resolver and the resolver could not answer, which is neither "agreed"
    nor "nothing to compare" and must not be filed under either.
    """
    try:
        target = path if path is not None else ledger_path()
        row = {
            "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "task": int(task_id or 0),
            "op": str(op or ("update" if int(task_id or 0) else "create")),
            "resolved": None if resolved is None else int(resolved),
            "found": None if found is None else int(found),
            "refused": bool(refused),
            "error": str(resolver_error or ""),
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        _append(target, json.dumps(row, sort_keys=True) + "\n")
        _bound(target)
        return True
    except Exception as exc:  # noqa: BLE001 — the guard outlives its own ledger
        logger.warning("description-guard ledger write failed (%s: %s); the write "
                       "proceeds and only this row is lost",
                       type(exc).__name__, str(exc)[:160])
        return False


def _bound(target: Path) -> None:
    """Keep the file at most MAX_BYTES by dropping the oldest whole rows."""
    if target.stat().st_size <= MAX_BYTES:
        return
    with target.open(encoding="utf-8") as handle:
        lines = [ln for ln in handle.read().splitlines() if ln.strip()]
    kept: list[str] = []
    total = 0
    for ln in reversed(lines):
        cost = len(ln.encode("utf-8")) + 1
        if kept and total + cost > LOW_WATER_BYTES:
            break
        kept.append(ln)
        total += cost
    from app.atomic_io import atomic_write_text
    atomic_write_text(target, "".join(ln + "\n" for ln in reversed(kept)))


def rows(path: Path | None = None) -> list[dict[str, Any]]:
    """Every parseable row, oldest first. A missing file is an empty list."""
    target = path if path is not None else ledger_path()
    out: list[dict[str, Any]] = []
    try:
        with target.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except Exception:  # noqa: BLE001 — one bad line is not a lost ledger
                    continue
                if isinstance(parsed, dict):
                    out.append(parsed)
    except FileNotFoundError:
        return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("description-guard ledger unreadable: %s", exc)
    return out


def counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    """The four buckets, in the order the question is asked.

    `refused` first and exclusively: a refusal always resolved at least one pair (a
    mismatch cannot be found without one), so counting it again under `passed` would
    double-book the row and hide the rate the reader is here for. `zero_resolved` is
    the bucket that used to be indistinguishable from `passed`.
    """
    out = {"refused": 0, "passed": 0, "zero_resolved": 0, "instrument_error": 0,
           "total": len(entries)}
    for row in entries:
        if row.get("refused"):
            out["refused"] += 1
        elif row.get("resolved") is None:
            out["instrument_error"] += 1
        elif row.get("resolved", 0) > 0:
            out["passed"] += 1
        else:
            out["zero_resolved"] += 1
    return out


def main(argv: list[str] | None = None) -> int:
    """Print the counts. `argv[0]`, if given, is the ledger to read."""
    args = list(sys.argv[1:] if argv is None else argv)
    target = Path(args[0]) if args else ledger_path()
    tally = counts(rows(target))
    print(f"ledger: {target}")
    print(f"rows={tally['total']} refused={tally['refused']} passed={tally['passed']} "
          f"zero_resolved={tally['zero_resolved']} "
          f"instrument_error={tally['instrument_error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
