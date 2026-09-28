"""Accounting for dashboard pins that need a frontend to run (#1691).

A test helper, not a test — the sibling of `board_presence.py`, so the same
`python -m pytest tests/` invocation picks nothing up here.

Why it exists. `tests/test_dashboard_responsive.py` skips all six of its geometry
pins when `web/node_modules` has no vite, and `ssssss / 6 skipped / exit 0` is
indistinguishable from a green run. The gate cannot see it either: its skip budget
is a SUITE total (`scripts/automod/gate.py:77-80`, live runs at 31-32 of a 40
ceiling), the partial-run branch (`gate.py:1518-1529`) applies no floor at all,
and `scripts/automod/review.py:274` flags only a skip call a round ADDS, so the
skip sites this file already had can never be blamed on a later round.

What it does. It keeps a ledger of which pins actually measured something, and
reports the ones that did not:

  * undeclared environment -> a NAMED FINDING on the terminal (`FINDING` below),
    counted from live state, exit code unchanged. A box with no `npm install`
    stays green, as it must: `node_modules` reaches a worktree only through
    `rung_frontend` (`gate.py:1231-1235`) and is gitignored, so red would fail
    every non-frontend round for the reason the file skips in the first place.
  * declared environment   -> a FAILURE naming the pins. The gate sets
    `DECLARED_ENV` exactly where it made `node_modules` reachable, so declared is
    a promise already made to a reviewer that the pins could run. One exception,
    decided by `declared_covers`: a stop the promise never spoke of — no chromium
    download, which the gate's one-file check knows nothing about — stays a SKIP
    with the finding printed. Clause 2 (declared and nothing ran is red) and
    clause 3 (an absent chromium still skips) otherwise collide on that box.

`scripts/automod/gate.py`'s tests rung greps the finding out of pytest's output
and puts it in the rung detail beside the pass counts, so a round that skipped its
dashboard pins says so in the report a reviewer reads (#1691 clause 5).
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

#: Set by the environment that GUARANTEES the frontend dependencies — the automod
#: gate does it in `_run_suite` only for the worktree it symlinked `node_modules`
#: into. Not a general "is this a dev box" flag: an unset variable means nobody
#: promised anything, which is why it must stay a skip there.
#:
#: The gate keeps its own copy of the same spelling (`gate.FRONTEND_PINS_ENV`):
#: this module has to stay importable in a scratch tree with no `scripts/` package
#: (the subprocess nodes in `test_dashboard_pin_accounting.py` run it there), so
#: the two ends cannot share one import. What they share instead is a pin —
#: `test_gate_parallel_tests.py::test_the_two_ends_of_the_seam_are_named_in_both_files`
#: compares them, so a rename on one side reddens a test rather than returning the
#: mechanism to silent-skip with every node green.
DECLARED_ENV = "LLOYD_FRONTEND_PINS_AVAILABLE"

#: The finding's name, and the string the gate's rung detail greps for. One
#: constant on each side of that seam: the pytest text and the gate's grep are the
#: same contract, and a rename has to move both or the finding goes silent again.
FINDING = "DASHBOARD_PINS_NOT_EXECUTED"

#: The fixture that means "this node needs the served dashboard". A pin is a pin
#: because it asks for the thing that may be missing — not because of its name,
#: and not because someone listed it: a hard-coded count goes stale the way four
#: stale counts in this repo's own gate detail strings went stale (#1448/#1541).
PIN_FIXTURE = "dashboard_url"


def declares_frontend_available() -> bool:
    """True when THIS process was told the frontend dependencies are reachable."""
    return os.environ.get(DECLARED_ENV, "").strip().lower() not in ("", "0", "false")


#: The one stop a box is allowed to take even when it was declared. The gate's
#: declaration is a check of ONE file — `web/node_modules/.bin/vite` — so a round that
#: touched `web/` on a machine with no chromium download arrives declared and still
#: cannot run a single pin. Clause 3 of #1691 says an absent chromium SKIPS; clause 2
#: says a declared run that executed nothing fails. Without this split those two
#: clauses contradict each other on exactly that machine, which is the conflict the
#: review rung found in round SM_20260928_005340.
#:
#: It is a constant rather than a substring both sides type, because `_measure` builds
#: the reason from it and `declared_covers` reads it: one rename moves the skipper and
#: the classifier together, and a drift between them is a failing test
#: (`test_a_stop_the_declaration_never_promised_stays_a_skip`, not a silent one).
BROWSER_MISSING = "chromium is not available"


def declared_covers(reason: str | None) -> bool:
    """Would the declaration have prevented this stop? True means the run was promised
    this dependency and did not get it, so the pending pins are a FAILURE. False means
    the stop is a dependency the declaration never spoke of — visible as a named
    finding, never red."""
    if not reason:
        # Nothing noted: the pins went pending with no explanation at all. The
        # declaration was made, so the silence is the failure, not an excuse.
        return True
    return not reason.startswith(BROWSER_MISSING)


class PinLedger:
    """Which of one file's pins measured something, and the report on the ones
    that did not. Kept per module rather than global so a scratch-tree copy of the
    pin file accounts for its own pins (#1691)."""

    def __init__(self, module_path: Path, fixture: str = PIN_FIXTURE):
        self.module = Path(module_path)
        self.fixture = fixture
        self._measured: set[str] = set()
        self._current: Optional[str] = None
        self._reason: Optional[str] = None
        self._reported = False

    # ── what the pin file calls while it runs ────────────────────────────────
    def begin(self, node_name: str) -> None:
        """Which pin is executing, so `_measure` can credit it without threading
        the node through every call site."""
        self._current = node_name

    def measured(self) -> None:
        """Credit the current pin with one real browser measurement."""
        if self._current:
            self._measured.add(self._current)

    def did_measure(self, node_name: str) -> bool:
        """Public read of the ledger, so a test can assert a pin was NOT credited
        without reaching into the set (#1691 clause 3's accounting half)."""
        return node_name in self._measured

    @property
    def reported(self) -> bool:
        """Has this module's finding been spoken yet? Public read of the once-only
        guard, so a node can assert the teardown DID report rather than trusting
        that a `finally` ran (#1691)."""
        return self._reported

    def note(self, reason: str) -> None:
        """Record why pins are about to skip, so the finding names vite or
        chromium instead of saying only that nothing ran. First note wins: the FIRST
        missing dependency is the one worth acting on, and any later one is noise
        behind it.

        Whether that stop makes a declared run red is NOT decided here — see
        `declared_covers`, which reads this back. A note is a fact about the box; the
        classifier is the rule about promises.
        """
        if self._reason is None:
            self._reason = reason

    @property
    def reason(self) -> Optional[str]:
        """Why the pins stopped, or None if nothing ever noted one. Public read, so the
        fixture's teardown does not reach into the ledger's private field (the review
        rung's advisory on round SM_20260928_005340 named that read at :243)."""
        return self._reason

    # ── the accounting ───────────────────────────────────────────────────────
    def pins(self, session) -> list[str]:
        """The file's pin node ids, read off the collected session."""
        out = []
        for item in getattr(session, "items", ()):
            if Path(str(item.fspath)).resolve() == self.module.resolve():
                if self.fixture in (item.fixturenames or ()):
                    out.append(item.name)
        return out

    def pending(self, session) -> list[str]:
        return [p for p in self.pins(session) if p not in self._measured]

    def finding_text(self, pending: list[str], total: int,
                     reason: Optional[str], *, with_names: bool) -> str:
        text = (f"{FINDING}: {len(pending)} of {total} pins in {self.module.name} "
                f"did not run — {reason or 'no reason recorded'}")
        if with_names and pending:
            text += ". Pins: " + ", ".join(p.split("[")[0] for p in
                                           dict.fromkeys(p.split("[")[0] for p in pending))
        return text

    def report(self, session, reason: Optional[str] = None) -> Optional[str]:
        """Emit the finding for whatever has not measured by now.

        Called twice per module at most — once on the dependency-skip path, once
        from the fixture's teardown — and it prints ONCE: six identical lines, one
        per skip site, is how a named finding stops being read. Returns the text
        it printed, or None when there was nothing to say (or it already spoke).
        """
        if self._reported:
            return None
        self._reported = True
        total_all = self.pins(session)
        pending = [p for p in total_all if p not in self._measured]
        if not pending:
            return None
        why = reason or self._reason
        if declares_frontend_available() and declared_covers(why):
            import pytest
            pytest.fail(self.finding_text(pending, len(total_all), why,
                                          with_names=True))
        # Undeclared, or declared but stopped by a dependency the promise never spoke
        # of (no chromium): the finding is printed and the skip stands.
        text = self.finding_text(pending, len(total_all), why, with_names=False)
        warnings.warn(text, stacklevel=2)
        return text

    def recompute(self, session, reason: Optional[str] = None) -> Optional[str]:
        """The same sentence again, recomputed from live state.

        Only exists so a test can prove `report` reads the ledger: after the state
        moves, the number has to move with it. A finding whose count came from
        prose could not.
        """
        total_all = self.pins(session)
        pending = [p for p in total_all if p not in self._measured]
        if not pending:
            return None
        return self.finding_text(pending, len(total_all),
                                 reason or self._reason, with_names=False)


@dataclass(frozen=True)
class ReMeasured:
    """The outcome of `grade_once_then_twice`: a first verdict, and a second only
    when the first disagreed."""

    first: tuple[str, ...]
    second: Optional[tuple[str, ...]] = None
    values: tuple = field(default=(), repr=False)

    @property
    def retried(self) -> bool:
        return self.second is not None

    @property
    def complaints(self) -> list[str]:
        """What the test must assert on: the SECOND measurement's verdict when
        there was one — an empty second verdict is a pass, which is the whole
        point — otherwise the first's, which was already clean."""
        if self.second is not None:
            return list(self.second)
        return list(self.first)

    @property
    def clean(self) -> bool:
        return not self.complaints

    def detail(self) -> str:
        if not self.retried:
            return "measured once"
        return (f"measured twice and the second measurement agreed with the "
                f"first: {list(self.first)}")


def grade_once_then_twice(measure: Callable[[], object],
                          grade: Callable[[object], Iterable[str]]) -> ReMeasured:
    """Measure, grade, and measure ONE more time only if the grade complained.

    `tests/test_dashboard_responsive.py::test_desktop_cards_keep_their_track_widths`
    flinches under xdist-8: the gate's own ledger records it among
    "failed only under parallel load and passed serially" (`promotions.jsonl`,
    rung `tests`, 2026-09-27T23:29:44Z), and until now that serial retry
    (`gate.py:1443-1467`) was the only thing standing between this pin and a red
    gate. Retrying inside the test makes the pin's own flake tolerance explicit
    while keeping the failure real: ONE disagreeing measurement is noise, a SECOND
    one is a layout defect, and the count never exceeds two.
    """
    first_value = measure()
    first = tuple(grade(first_value))
    if not first:
        return ReMeasured(first=first, values=(first_value,))
    second_value = measure()
    return ReMeasured(first=first, second=tuple(grade(second_value)),
                      values=(first_value, second_value))
