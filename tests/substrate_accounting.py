"""Accounting for the substrate nodes of `tests/test_path_sandbox.py` (#3133).

A test helper, not a test — the sibling of `dashboard_pins.py`, so the same
`python -m pytest tests/` invocation picks nothing up here.

Why it exists. `tests/test_path_sandbox.py` is the ONLY functional proof that the
kernel layer of vault protection (the `--ro-bind` substrate of #2109, `b3b652b6`)
refuses protected-tree writes. Three of its shapes could report all-pass while
pinning nothing:

  * the `skipif_no_bwrap` marker evaluated `P.available()` at module import, so one
    failed 60 s-cached probe skipped every substrate node of the file and the run
    still read green;
  * the 15-vector corpus was ONE node with a `pytest.skip` inside its loop, so one
    missing dependency (`perl`) discarded the verdicts of the ~14 vectors that had
    already executed;
  * the blindness pin asserted only that twelve spellings return `None`, which a
    matcher that returned `None` for everything satisfies.

This module answers the first. It keeps a ledger of which substrate-gated nodes
actually entered a body and reports the ones that did not — from a node INSIDE the
pin file, so a mass skip cannot exit 0 whatever else the run looks at. The other two
are fixed in the pin file itself (one parametrized node per vector; a positive
control beside the blindness pin).

Why the report is a node and not a warning. The gate's only counterweight was the
suite-wide `PYTEST_MAX_SKIPPED = 40` (`scripts/automod/gate.py`), whose own docstring
records a baseline of 31 skips — ten substrate skips fit the moment any unrelated skip
leaves the suite — and the partial-run branch applies no floor at all. A finding on
stdout can be missed the same way; a FAILED node cannot. So `verdict()` is spoken by
`test_the_substrate_nodes_that_never_ran_are_reported`, which the pin file ships with
no skip marker of its own and places LAST, so it speaks after the nodes it counts.

Why an environment declaration is the only quiet route. `dashboard_pins.py` solved the
same problem for the frontend pins: an undeclared box keeps its skips, a declared run
turns a pin that never executed into a FAILURE. Here the polarity is deliberately the
other way round, because the thing at stake is not a measurement of the box but the
proof that the kernel refuses writes into `SOUL.md`: UNDECLARED is the loud state, and
`DECLARED_ENV` exists for the one runner that has decided the substrate is supposed to
be missing here and accepts the skips it declares. Nothing in the tree sets it today
(see the accounting file); it is the seam a later gate change would set, and its
spelling is pinned there so a rename reddens a test instead of returning the mechanism
to silent-skip.
"""

from __future__ import annotations

import ast
import functools
import os
from pathlib import Path

#: The environment variable that DECLARES the substrate unavailable on purpose.
#: Unset (the normal state, and the state of every automod round) means a substrate
#: node that never ran is a FAILURE. Set it only from a runner that knows this box
#: cannot build a namespace and has decided to accept that — the mirror image of
#: `dashboard_pins.DECLARED_ENV`, which declares the opposite promise.
DECLARED_ENV = "LLOYD_PATH_SUBSTRATE_AVAILABLE"

#: The finding's prefix. Same idiom as `dashboard_pins.FINDING`: one constant, so a
#: later grep (a gate rung's detail, a reviewer's search) and the raised message are
#: the same contract.
FINDING = "PATH_SUBSTRATE_NODES_NOT_EXECUTED"

#: The node inside the pin file that speaks for the ledger. It carries no skip marker,
#: so the ledger's own verdict is always executed.
GATE_NODE = "test_the_substrate_nodes_that_never_ran_are_reported"

#: The name of the single substrate gate in the pin file. The ledger counts the nodes
#: it decorates by this NAME, not by a typed-out list, because a hard-coded count is
#: the stale-number class this repo's own gate detail strings keep dying of (#1448).
SUBSTRATE_GATE = "skipif_no_bwrap"


def declares_substrate_available() -> bool:
    """True when THIS process was told the substrate is expected to be missing."""
    return os.environ.get(DECLARED_ENV, "").strip().lower() not in ("", "0", "false")


class SubstrateLedger:
    """Which of the pin file's substrate nodes entered a body, and the report on the
    ones that did not. Kept per module instance so the pin file accounts for its own
    nodes, the way `dashboard_pins.PinLedger` does (#3133)."""

    def __init__(self, gate: str = SUBSTRATE_GATE):
        self.gate = gate
        self.gated: set[str] = set()        # functions carrying the substrate gate
        self.executed: set[str] = set()     # functions whose body was entered
        self.probe_reason: str = ""         # the probe's own explanation, if any

    # ── what the pin file does while it is being imported and run ────────────
    def note_probe(self, reason: str) -> None:
        """The reason the substrate probe gave at import, so the finding can quote
        the kernel's complaint instead of saying only that nothing ran."""
        self.probe_reason = reason or ""

    def expect_gated_from_source(self, module_file: str | Path) -> "set[str]":
        """The names of this file's gated test functions, read off its SOURCE.

        Read at import with `ast`, before a single node has run, because the
        denominator of the finding has to exist even when nothing executes: a ledger
        that learned its own expectation from executed nodes would report "0 of 0 did
        not run" on exactly the box where everything skipped.
        """
        tree = ast.parse(Path(module_file).read_text())
        for node in tree.body:
            if not (isinstance(node, ast.FunctionDef)
                    and node.name.startswith("test_")):
                continue
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                name = getattr(target, "id", None) or getattr(target, "attr", None)
                if name == self.gate:
                    self.gated.add(node.name)
        return self.gated

    def measured(self, fn):
        """Decorator: credit the gated function with having ENTERED its body.

        Entry, not success — a node that ran and failed is already red, and the only
        state that can hide is one that never got as far as the first statement. The
        corpus is parametrized, so this credits a FUNCTION: once any vector of the
        corpus entered the body, the substrate plainly worked, and the vectors that
        did not execute did so for their own per-vector reason (a missing program),
        which their own skip line names. That is clause 2's dependency skip, and it is
        deliberately not counted as a substrate non-execution.
        """
        name = fn.__name__

        @functools.wraps(fn)
        def entered(*args, **kwargs):
            self.executed.add(name)
            return fn(*args, **kwargs)

        return entered

    # ── the accounting ──────────────────────────────────────────────────────
    def collected(self, session) -> "dict[str, list[str]]":
        """Gated node names collected in this session, grouped by function."""
        out: dict[str, list[str]] = {}
        for item in getattr(session, "items", ()):
            base = Path(str(item.fspath)).name
            if base != Path(self.module_name).name:
                continue
            name = item.name
            func = name.split("[")[0]
            if func in self.gated:
                out.setdefault(func, []).append(name)
        return out

    def pending(self, session) -> "tuple[list[str], int]":
        """`(nodes that never ran, total gated nodes collected)`."""
        groups = self.collected(session)
        total = sum(len(names) for names in groups.values())
        pending = [n for func, names in groups.items()
                   if func not in self.executed for n in names]
        return pending, total

    def verdict(self, session) -> "tuple[str, str]":
        """`("pass" | "skip" | "fail", message)` — the three things the ledger can say.

        * **pass** — every substrate node collected entered its body. The run is what
          it claims to be, and the node says so out of the ledger rather than by
          saying nothing.
        * **skip** — nodes are outstanding AND the run declared the substrate missing
          through `DECLARED_ENV`. This is the ONLY quiet branch, and it is available
          to nobody who did not make the declaration.
        * **fail** — nodes are outstanding with no declaration, or not one gated node
          was collected at all (the gate itself went missing, which is the loudest of
          the three because it means the file stopped testing the substrate entirely).
        """
        pending, total = self.pending(session)
        if not total:
            return ("fail",
                    f"{FINDING}: not one node of {self.module_name} whose function "
                    f"carries the `{self.gate}` gate was collected, so the substrate "
                    f"proof is pinning nothing (gated functions known to the ledger: "
                    f"{sorted(self.gated)})")
        if not pending:
            return ("pass", f"all {total} substrate-gated nodes of {self.module_name} "
                            "entered their bodies")
        if declares_substrate_available():
            return ("skip",
                    f"{len(pending)} of {total} substrate nodes of {self.module_name} "
                    f"did not run — declared acceptable by {DECLARED_ENV}: "
                    f"{self.probe_reason or 'no probe reason recorded'}")
        return ("fail",
                f"{FINDING}: {len(pending)} of {total} substrate nodes in "
                f"{self.module_name} did not run — "
                f"{self.probe_reason or 'no probe reason recorded'}"
                f". Nodes: {', '.join(sorted(pending))}")

    module_name = "test_path_sandbox.py"
