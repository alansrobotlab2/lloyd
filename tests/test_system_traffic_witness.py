"""#2207 clause 6 — the traffic figure the item rests on has to come from bytes
with history, and stay equal to them.

The item's scope ruling ("`/api/system/*` took 0 of the logged requests, so keep
the surface and bound it") was a count over `~/lloyd-data/logs/server.log*`, which
are rotatable files: by the time this round ran, rotation had already swept a
segment and the triage number (1,321,377, measured ~03:00Z on 2026-10-05) could no
longer be reproduced from anything on disk. Clause 6 asks that the report be
re-derived from a committed witness instead, and that the item quote that figure.

So this file is not testing the product; it is testing the evidence the product
decision was made on, in the only way that outlives a log rotation: the committed
witness at `~/obsidian/backlog/data/server.log` (vault commit `eb15a6e4`) versus
the figure written in the item body. A number quoted into prose and never tied to
bytes is a claim nobody can re-check, and the corpus it was counted from was gone
inside a day.

Marked `live_vault` because both paths it reads are outside this repository: the
gate runs `-m "not live_vault"`, which is why clause 6 is cited as the run this
round made (`tests/test_system_traffic_witness.py -k witness`) rather than as a
green rung. Nothing here skips quietly — if the vault is not where the clause says
it is, the node fails, and says which half it could not find.

The cost of that mark is recorded because a review named it on 2026-10-05: while
the mark stands this anti-rot property has no CI, so a drift between the item's
figure and the bytes is caught only by a run that opts in. Dropping it would put
the check in every gate at the price the convention exists to avoid — coupling
every round, on every box, to a live vault and to a backlog file this very loop
appends to. Settling that is a ruling about `live_vault`, not something this diff
can decide by itself.
"""

from __future__ import annotations

import glob
import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.live_vault

VAULT = Path.home() / "obsidian"
WITNESS = VAULT / "backlog" / "data" / "server.log"

#: The four counts clause 6 reports, re-derived from the witness on 2026-10-05:
#: the total request tokens, the zero the ruling rests on, and the three prefixes
#: that prove the extraction pattern is not silently matching nothing.
TOTAL_TOKENS = 1_244_996
SYSTEM_HITS = 0
CONTROLS = {"entity": 10, "lsp": 1, "skills": 1}


def _witness_lines() -> list[str]:
    assert WITNESS.is_file(), (
        f"{WITNESS} is missing — the clause 6 witness was never committed, or was "
        "deleted after vault commit eb15a6e4")
    text = WITNESS.read_text(encoding="utf-8")
    assert text, f"{WITNESS} is empty — an empty file matches no pattern at all"
    return text.splitlines()


def test_the_witness_bytes_reproduce_the_report_they_were_committed_for():
    """The report itself, recounted from the committed lines: the total, the zero,
    and the controls that make the zero mean 'no such request' rather than 'no
    such pattern'."""
    lines = _witness_lines()
    assert len(lines) == TOTAL_TOKENS, (
        f"the witness holds {len(lines):,} request tokens, not the {TOTAL_TOKENS:,} "
        "the clause and the item both quote — a witness whose number moves is not a "
        "witness; re-cut it from the logs and update both together")
    assert all(ln.startswith("/api/") for ln in lines[:200]), (
        "the witness is no longer one API path per line, so the per-prefix counts "
        "below are counting a different shape than the one that was reported")

    assert sum(1 for ln in lines if ln.startswith("/api/system")) == SYSTEM_HITS, (
        "the witness now contains /api/system requests: either someone opened the "
        "cert panel since the witness was cut (in which case the item's scope "
        "ruling needs re-measuring), or the extraction changed")
    for prefix, expected in CONTROLS.items():
        seen = sum(1 for ln in lines if ln == f"/api/{prefix}")
        assert seen == expected, (
            f"control /api/{prefix}: witness holds {seen} lines, the report said "
            f"{expected} — the zero for /api/system is only meaningful while the "
            "pattern is demonstrably firing on other routes")


def test_the_witness_is_committed_and_the_item_quotes_its_number():
    """Both halves of clause 6 together: the bytes are in the vault's history, and
    the number the item quotes is the number those bytes produce.

    The first is `git ls-files` on the vault, not a stat: a file that exists only on
    this box has exactly as much history as the log rotation that ate the last one.
    The second is the anti-rot pin — the item body states a "Witness figure of
    record", and it has to agree with a recount, so a future re-triage that pastes a
    fresh live count without re-cutting the witness breaks here instead of quietly
    replacing the evidence.
    """
    tracked = subprocess.run(
        ["git", "-C", str(VAULT), "ls-files", "--error-unmatch",
         "backlog/data/server.log"],
        capture_output=True, text=True)
    assert tracked.returncode == 0, (
        "the witness is on disk but not tracked by the vault repo — clause 6 exists "
        f"precisely to stop that: {tracked.stderr.strip()}")

    items = sorted(glob.glob(str(VAULT / "backlog" / "2207-*.md")))
    assert len(items) == 1, f"expected one #2207 item file, found {len(items)}"
    body = Path(items[0]).read_text(encoding="utf-8")
    quoted = re.findall(r"Witness figure of record:\s*([\d,]+) request tokens", body)
    assert len(quoted) == 1, (
        f"the item states {len(quoted)} 'Witness figure of record' lines; clause 6's "
        "tail requires exactly one figure for the witness to be checked against")
    n = int(quoted[0].replace(",", ""))
    assert n == len(_witness_lines()), (
        f"the item quotes {n:,} but the committed witness recounts "
        f"{len(_witness_lines()):,} — re-cut the witness from the logs and update "
        "the item together, which is the pair this node exists to keep in step")
