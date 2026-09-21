"""The matrix's own rows, runnable as pytest nodes under the fault-injection marker.

`pytest -m "not live_vault and not fault_injection" tests/degradation` — the expression the
gate's tests rung now uses — collects nothing from this file. That is the point: these are the
tests that inject faults, and the selfmod gate runs `pytest -q` on a box where a worker pool
may be live, so no **row** is reachable from a bare run (#644's own blast radius risk). A row
is a fault injected at one of Lloyd's real dependencies — a health endpoint, the vault count,
the voice media table, the scheduler's due gate — judged by what that consumer then reported.

The complement, deliberately *not* marked, is the four seam crossings at the bottom of
`tests/test_degradation_contract.py`. They assert that an injector's mechanism is real — a
child interpreter ran the command, a handshake actually negotiated TLS, `git repack` actually
folded loose objects — against a temp directory, an ephemeral loopback port and a throwaway
certificate, and they judge no consumer's verdict. The two sets answer different questions,
and collapsing them either way loses something: a seam verified only inside a deselected file
is not verified by the gate that is supposed to be guarded, and the full row walk does not
belong in the two-minute rung that runs on a box serving its user.

The marker is deliberately **not** declared in `pytest.ini`, and the reason is a boundary
rather than a trick: `pytest.ini` is a path the self-modification loop may not write at all,
and #644's acceptance check requires this suite to land without editing it. So the exclusion
lives in the gate's `-m` expression and nowhere else.

Registration does not decide selection — `-m` matches the marks applied to an item, whether or
not the name appears in `ini`, so every node in this file is deselected either way; registering the
name would only silence the `PytestUnknownMarkWarning` and satisfy a `--strict-markers` run
(this repo runs without it). What the unregistered name does buy is that the expression stays
harmless where the marked files are absent: in a candidate tree rebased onto a base predating
`tests/degradation/`, `not fault_injection` matches no item and everything else still runs.
`tests/test_degradation_contract.py::test_the_marker_is_unregistered_and_the_gate_carries_the_widened_expression`
pins the exclusion to the gate and `pytest.ini` to unmodified.

The rows are executed here as well as by the runner so a regression shows up as a named node
in the suite output rather than as a line in a subprocess's stdout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.degradation import runner as R  # noqa: E402

pytestmark = pytest.mark.fault_injection

ROWS = {row["id"]: row for row in R.load_matrix()}


@pytest.mark.parametrize("row_id", sorted(ROWS))
def test_degradation_row_holds(row_id):
    """One row, one injected fault, one declared behavior.

    The failure message is the row's evidence line — injected fault, what the consumer
    reported — because that pairing is the artifact this matrix exists to produce. A row that
    fails is either a regression in a consumer's failure handling or a row whose fault stopped
    being injectable, and the evidence says which.
    """
    record = R.run_row(ROWS[row_id])
    assert record["passed"], (
        f"{row_id} | {record['fault']} | reported {record['reported']!r} "
        f"| {record['evidence']}"
        + ("".join(f"\n  - {problem}" for problem in record.get("problems", []))))
