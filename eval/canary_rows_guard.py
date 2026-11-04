#!/usr/bin/env python3
"""#3419 — who may append to the TRACKED injection-canary corpus, and what to say if not.

The thing this refuses is a rehearsal. `eval/run_persistence_arms.sh 2` and
`eval/run_egress_arms.sh 1 --on-port …` are the SAME commands the two weekly systemd
window fires run, and with `LLOYD_CANARY_ROWS` unset they appended straight into
`eval/measurements/injection-canary/rows.jsonl` — the tracked corpus the deploy gate
reads its leak rate out of. Measured at 2026-10-26: one rehearsal left 18 uncommitted
rows there (`git diff --numstat -- eval/measurements/injection-canary/rows.jsonl` ->
`18  0`), 15 of them the whole egress scenario set at `arm: enforce-off`, and the 10-26
window then read 30/66 in the working tree against the committed 27/60
(`eval/measurements/injection-canary/run-2026-10-26-persistence.md:10`). Two consecutive
sub-bound windows were being counted off a contaminated file.

The rule, in one line: **the tracked corpus is a window fire's to write.** A caller with
no redirect that is not one of the two window units is refused, exits non-zero, and is
told to set `LLOYD_CANARY_ROWS`. A caller that DID set it is never refused from here —
sending a rehearsal to a scratch file is exactly what #2397 built the variable for.

WHO COUNTS AS A WINDOW FIRE is the predicate
`scripts/maintenance/commit_injection_canary_corpus.py` already owns and tests (#2877):
`INVOCATION_ID` set AND one whole `/proc/self/cgroup` path component equal to a key of
`WINDOW_UNITS` (`lloyd-persistence-window.service`, `lloyd-egress-window.service`).
This module IMPORTS that rather than re-deriving it, because two implementations of "is
this the unit" is how a guard and the corpus commit step start meaning different things
by the same fire — and because `INVOCATION_ID` alone is not the discriminator: an agent
Bash fork on this box inherits the backend unit's value (measured twice in one turn in
that module's docstring, same value both times), so presence would admit exactly the
caller being refused.

Why the cgroup and not "just require LLOYD_CANARY_ROWS": both units deliberately UNSET
that variable (`agent-services/systemd/lloyd-persistence-window.service:25`,
`lloyd-egress-window.service:112,124`) and commit their rows through an `ExecStartPost`,
and `tests/test_persistence_window_unit.py` plus `tests/test_egress_window_unit.py` pin
those absences — a blanket refusal on the unset variable would switch both measurement
windows off and read green doing it.

Stdlib plus that one import, so a driver can call `preflight()` at the top of `main()`
before it pauses the dispatch pool: the #3169 lesson in `eval/run_egress_arms.sh` is that
a window which is going to be refused must be refused before it spends wall clock and a
pool hold on nothing.

    python eval/canary_rows_guard.py     # operator's question: "am I a window fire?"
    echo $?                              # 0 = the tracked corpus is mine to write, 2 = refused
Tests: .venvs/lloyd/bin/python -m pytest tests/test_injection_canary_rows_guard.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
LLOYD_HOME = EVAL_DIR.parent
if str(LLOYD_HOME) not in sys.path:
    sys.path.insert(0, str(LLOYD_HOME))

from scripts.maintenance.commit_injection_canary_corpus import (  # noqa: E402
    PROC_CGROUP, WINDOW_UNITS, authorized_unit, invocation_refusal)

#: The redirect that makes a caller's destination scratch and therefore always allowed.
#: Read from the same name the bench writes (`eval/run_injection_canary.py:ROWS_ENV_VAR`);
#: duplicated as a literal here because importing that module to ask one question would
#: pull the whole bench (and its aggregator client) into a pre-flight.
ROWS_ENV_VAR = "LLOYD_CANARY_ROWS"
REPORT_ENV_VAR = "LLOYD_CANARY_REPORT"

#: The corpus directory, as a display string in the refusal only. The guard deliberately
#: takes a destination decision from its caller and never stats a path: the question it
#: answers is about the CALLER, and a guard that resolved paths would disagree with the
#: caller about which file was meant under a test that monkeypatches `ROWS_PATH`.
TRACKED_CORPUS_ROWS = "eval/measurements/injection-canary/rows.jsonl"

#: The env var naming one systemd invocation, read by `is_systemd_invocation` there.
INVOCATION_ENV_VAR = "INVOCATION_ID"

#: The two row fields carrying the per-fire identity (#3419 step 3 / #3748). `fire_unit`
#: is the KEY of `WINDOW_UNITS` that `authorized_unit` matched — the window key of the
#: authorized unit — and `fire_invocation` is that fire's `INVOCATION_ID`. Groupers
#: (`per_window`, `persistence_windows`) split a UTC date by the pair when both are
#: present and pool by date alone when either is absent, so every identity-less
#: historical row keeps bucketing by `ts[:10]` and no published figure re-derives.
FIRE_UNIT_FIELD = "fire_unit"
FIRE_INVOCATION_FIELD = "fire_invocation"


def window_of(row: dict) -> tuple[str, str]:
    """`(sort key, display label)` of the measurement window one row counts toward.

    Two window fires sharing a UTC date pooled into ONE denominator, because both
    groupers keyed on `ts[:10]` alone: the 2026-11-02 egress fire's two controls
    (`36c3607f`) landed inside the persistence fire's window and turned its published
    34/60 into a pooled 35/72 with a flipped bound flag, and `f3eb9ac3` did the same
    to 2026-10-26 with rehearsal rows, 27/60 → 30/66 (#3419, #3748).

    A row stamped by `append_rows` (see `fire_identity`) sorts and labels by
    `day + fire_unit + fire_invocation`, so one UTC date carrying two fires yields two
    windows, each with its own denominator. A row carrying NEITHER field — every row
    committed before the stamp existed, which is every row on the tracked corpus today —
    keeps the bare date, exactly what both groupers computed on every historical row
    before this function, so no published figure re-derives. The key is a string with
    the day first, so "oldest window first" survives, and an unstamped date's window
    sorts ahead of its own stamped splits.

    Lives here, not in the runner, because the attribution probe imports this module
    and must not pay for the runner's asyncio/http-server harness to learn a bucketing
    rule — and a rule read by two groupers may only be written once.
    """
    day = (row.get("ts") or "?")[:10]
    unit = str(row.get(FIRE_UNIT_FIELD) or "")
    inv = str(row.get(FIRE_INVOCATION_FIELD) or "")
    if unit and inv:
        short = unit.removeprefix("lloyd-").removesuffix(".service")
        return f"{day} {unit} {inv}", f"{day} {short} {inv[:8]}"
    return day, day


def fire_identity(env: dict[str, str] | None = None,
                  cgroup_file: str = PROC_CGROUP) -> dict[str, str]:
    """The per-fire stamp `append_rows` merges into every row of a TRACKED-corpus write.

    Both parts are facts about the caller that nothing in the caller controls per-row:
    the cgroup component systemd wrote when it started the unit, and the `INVOCATION_ID`
    systemd generated for that one invocation. Read through the SAME `authorized_unit`
    the refusal reads, so the identity stamped on a row and the identity that admitted
    the write cannot come apart. No validation here: the only tracked write that reaches
    this is one `preflight` already admitted, which means both reads are non-empty —
    outside a window the dict is simply what the environment says, and `append_rows`
    never stamps a scratch destination with it (#3748 clause 1).
    """
    inv = (os.environ if env is None else env).get(INVOCATION_ENV_VAR) or ""
    return {FIRE_UNIT_FIELD: authorized_unit(cgroup_file) or "",
            FIRE_INVOCATION_FIELD: inv}


class CanaryRowsRefused(RuntimeError):
    """A caller tried to write the tracked corpus without being one of the window fires.

    `RuntimeError` and not `SystemExit` on purpose: the three entry points that catch it
    (`_run_cmd` in the bench, `main()` in both window drivers) each print the reason to
    stderr and return their own 2, and a `SystemExit` raised through an `asyncio.run`
    would unwind the drivers' hold-teardown instead of running it.
    """


def has_redirect(env: dict[str, str] | None = None) -> bool:
    """Whether `LLOYD_CANARY_ROWS` is set to something non-blank.

    Blank counts as unset — the same reading `_env_path` gives in the bench, where an
    empty `LLOYD_CANARY_ROWS=` is the shell's way of saying unset in a wrapper that
    always passes the variable.
    """
    raw = (os.environ if env is None else env).get(ROWS_ENV_VAR) or ""
    return bool(raw.strip())


def unwindowed_write_refusal(env: dict[str, str] | None = None,
                             cgroup_file: str = PROC_CGROUP) -> str | None:
    """Why this caller may not append to the tracked corpus, or None if it may.

    Delegates the conjunction to `invocation_refusal` (#2877) and adds the remedy, because
    a refusal that does not name `LLOYD_CANARY_ROWS` leaves a rehearsal with nowhere to go
    and it will re-run the command with `sudo`-grade confidence instead.
    """
    refused = invocation_refusal(env, cgroup_file)
    if refused is None:
        return None
    units = " or ".join(sorted(WINDOW_UNITS))
    return (f"refusing to write the tracked injection-canary corpus ({TRACKED_CORPUS_ROWS}): "
            f"{refused}. Set {ROWS_ENV_VAR}=<scratch>/rows.jsonl (and "
            f"{REPORT_ENV_VAR}=<scratch>/run-<date>-persistence.md for the report) to "
            f"measure into a scratch file instead; the tracked corpus is appended only by "
            f"a {units} window fire (#3419).")


def preflight(env: dict[str, str] | None = None,
              cgroup_file: str = PROC_CGROUP) -> str | None:
    """The message this caller must be refused with, or None if it may write the corpus.

    `None` when `LLOYD_CANARY_ROWS` names a scratch destination: the guard never refuses a
    redirect, so a rehearsal that follows the message can never be refused twice, and the
    two weekly fires — which set nothing but do run in their unit's cgroup — fall through
    to the tracked corpus exactly as they do today.
    """
    if has_redirect(env):
        return None
    return unwindowed_write_refusal(env, cgroup_file)


def check_or_raise(cgroup_file: str = PROC_CGROUP) -> None:
    """Raise `CanaryRowsRefused` if this process may not write the tracked corpus."""
    refused = preflight(cgroup_file=cgroup_file)
    if refused:
        raise CanaryRowsRefused(refused)


def main(argv: list[str] | None = None) -> int:
    """The operator's question, answered by exit code: is the tracked corpus mine?

    Prints the refusal to stderr and returns 2 for a caller that is not a window fire, so
    the same function the drivers call can be run by hand before typing a rehearsal. No
    argv flags: a switch that took a cgroup path would be one the refused caller could
    pass, which is the reasoning `commit_injection_canary_corpus.py` gives for the same
    choice (`PROC_CGROUP` there).
    """
    refused = preflight()
    if refused is None:
        print(f"canary_rows_guard: this caller may append to {TRACKED_CORPUS_ROWS}")
        return 0
    print(f"canary_rows_guard: {refused}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
