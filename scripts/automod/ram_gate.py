"""The one reader of the host-RAM boot-gate definition both routes share.

`agent-services/bin/ram-boot-gate.sh` holds the four thresholds — the landing
route's floor and abort line, the A/B sweep route's wait and abort line — and
the reason each one is what it is. The sweep route sources that file directly;
this module is how the landing route reads the SAME numbers, so a floor that has
to move moves in exactly one place and every reader sees the move. (#1340: the
two pairs lived in two files with nothing between them, and the prose pointer to
the production one was 596 lines off, so a reader could check neither.)

Parsing is deliberately strict. Each name must appear exactly once, as a bare
``NAME=<integer>`` assignment, and each route's abort line must sit below its own
wait/floor line. There is no default, no fallback and no remembered copy: a
missing or malformed definition raises, because a boot gate that quietly falls
back to a number someone typed in here is the exact defect this module removes.

Run it directly to see what the two routes will actually enforce::

    .venvs/lloyd/bin/python -m scripts.automod.ram_gate
"""

from __future__ import annotations

import re
from pathlib import Path

# This module's own tree, not the live checkout: a candidate worktree has to read
# the definition it is carrying, or a gate would test one tree's numbers and land
# another's.
REPO_ROOT = Path(__file__).resolve().parents[2]
RAM_GATE_FILE = REPO_ROOT / "agent-services" / "bin" / "ram-boot-gate.sh"

#: The GiB thresholds. Names are shared with the shell definition verbatim, so a
#: grep for any one of them lands on one file.
GIB_THRESHOLDS = ("PRIMARY_RAM_FLOOR_GIB", "PRIMARY_RAM_ABORT_GIB",
                  "SWEEP_RAM_WAIT_GIB", "SWEEP_RAM_ABORT_GIB")

#: Wait budgets, in whole seconds (also GiB-free of their own).
SECOND_BUDGETS = ("PRIMARY_RAM_WAIT_SECONDS",)

DEFINITION_NAMES = GIB_THRESHOLDS + SECOND_BUDGETS

# Only a bare `NAME=123` line is a definition: the shell file's cadence lines
# (`RAM_GATE_TRIES="${RAM_GATE_TRIES:-60}"`) and its prose must not parse.
_ASSIGNMENT = re.compile(r"^([A-Z][A-Z0-9_]*)=([0-9]+)$")

# Each route's (wait-or-floor, abort) pair. An abort line at or above its own
# wait line makes the gate a no-op or a permanent refusal, whichever the reading
# is, so it is checked on both sides of the seam — the shell checks the same two
# pairs in `ram_gate_numbers_ok` before it will boot anything.
PAIRS = (("PRIMARY_RAM_FLOOR_GIB", "PRIMARY_RAM_ABORT_GIB"),
         ("SWEEP_RAM_WAIT_GIB", "SWEEP_RAM_ABORT_GIB"))


def parse_ram_gate(text: str) -> dict[str, int]:
    """Read the thresholds out of the shared definition's text."""
    found: dict[str, int] = {}
    for lineno, line in enumerate(text.splitlines(), start=1):
        m = _ASSIGNMENT.match(line)
        if not m or m.group(1) not in DEFINITION_NAMES:
            continue
        name = m.group(1)
        if name in found:
            raise ValueError(f"{RAM_GATE_FILE}: {name} is defined twice (line {lineno})")
        found[name] = int(m.group(2))
    missing = [n for n in DEFINITION_NAMES if n not in found]
    if missing:
        raise ValueError(f"{RAM_GATE_FILE}: no definition for {', '.join(missing)}")
    for high, low in PAIRS:
        if found[low] >= found[high]:
            raise ValueError(f"{RAM_GATE_FILE}: {low} ({found[low]}) must sit below "
                             f"{high} ({found[high]}) — an abort line at or above its "
                             "own wait line is not a gate")
    return found


def load_ram_gate(path: Path | None = None) -> dict[str, int]:
    """Parse the shared definition from disk. Raises rather than defaulting."""
    return parse_ram_gate((path or RAM_GATE_FILE).read_text())


_RAM_GATE = load_ram_gate()

# The landing route's two thresholds and its wait budget. `promote.py` binds
# these names into its own module so the existing restart-leg tests can still
# patch them, but the numbers themselves come from the one definition file.
PRIMARY_RAM_FLOOR_GIB = _RAM_GATE["PRIMARY_RAM_FLOOR_GIB"]
PRIMARY_RAM_ABORT_GIB = _RAM_GATE["PRIMARY_RAM_ABORT_GIB"]
PRIMARY_RAM_WAIT_SECONDS = float(_RAM_GATE["PRIMARY_RAM_WAIT_SECONDS"])

# The sweep route's pair, exported so a test can assert the arm script and this
# module still agree. `_restart_primary` does not use them: they are the A/B
# sweep's, and lower by design (see the definition file for why).
SWEEP_RAM_WAIT_GIB = _RAM_GATE["SWEEP_RAM_WAIT_GIB"]
SWEEP_RAM_ABORT_GIB = _RAM_GATE["SWEEP_RAM_ABORT_GIB"]


if __name__ == "__main__":
    for _name in DEFINITION_NAMES:
        print(f"{_name}={_RAM_GATE[_name]}")
