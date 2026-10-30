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
    PROC_CGROUP, WINDOW_UNITS, invocation_refusal)

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
