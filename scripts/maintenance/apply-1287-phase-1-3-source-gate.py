#!/usr/bin/env python3
"""Land the #1287 Phase 1.3 source-gate clause in the consolidation runbook.

`nightly-skill-consolidation/SKILL.md` lives in the vault, so it is not in the code
round's diff — the clause has to arrive through the vault route, and this script is what
makes that arrival deterministic instead of hand-typed: exact-anchor insertion, refusing
if either anchor is ambiguous or outside Phase 1.3, and idempotent so a re-run reports
"already applied" rather than doubling the clause.

`tests/test_consolidation_source_gate.py` calls `patched_text()` directly, which is how
one commit can carry both halves: the test asserts the runbook's *final* text while the
clause is still in flight here, and reads the live file once it has landed.

Usage:
    python apply-1287-phase-1-3-source-gate.py            # apply to the live vault
    python apply-1287-phase-1-3-source-gate.py --dry-run  # report, write nothing
"""
from __future__ import annotations

import sys
from pathlib import Path

SKILL = Path.home() / "obsidian" / "skills" / "nightly-skill-consolidation" / "SKILL.md"

MARKER = "consolidation_source_gate.py"

ANCHOR = "- Has 2+ dated snapshots (pattern is persistent,not a one-off)\n"

CLAUSE = """- **Source gate — ask the emitter whether it would still write this file.** A
  candidate sitting on disk is not evidence that `mine-trajectories.py` would write it
  today: `is_emittable()` (`scripts/mine-trajectories.py:215`, #1181) refuses to emit a
  `sequence` pattern flagged `has_error_recovery: false`, and the historical files stay
  where they are. Consult the gate *after* the four filters above and *before* the cap:
  ```bash
  ~/lloyd/.venvs/lloyd/bin/python ~/lloyd/scripts/consolidation_source_gate.py \\
      check --candidates ~/lloyd-data/_pipeline/skills/candidates/ | tee /tmp/source-gate.txt
  ```
  It prints `DROP <pattern> :: not_emittable :: <the emitter's own reason>` per key, and
  its last line is `scanned: N  eligible: N  dropped_by_source_gate: N
  eligible_after_source_gate: N` — the same last-line shape, and for the same reason, as
  Phase 0's `skipped_by_verdict`. Put `dropped_by_source_gate: N` and
  `eligible_after_source_gate: N` in the run report **even when the work list is empty**:
  an unexplained zero reads as a broken job (Phase 0.4, same reason). Record each dropped
  key in the Phase 4.6 ledger as `rejected_artifact_class` with that command as
  `evidence_cmd`.
  The tool **imports** `is_emittable` rather than restating it, so a change to the emitter
  lands here for free; `tests/test_consolidation_source_gate.py` fails if the runbook and
  the emitter ever disagree, which is what stops this bullet becoming a hand-copied `if`.
  If the script is missing — a rolled-back round, a tree that predates it — report
  `dropped_by_source_gate: script_absent` and adjudicate the pre-gate pool as before: the
  gate may be unavailable, it may never be silently skipped.
  Measured 2026-09-20T08:0xZ over the 3,930 candidate files: of the **761** keys that
  clear every filter above, **653** are refused here and **108** remain — at 5 patterns a
  run that was ~152 nights of hand-adjudicating a class the source had already decided
  (`run_58_20260920_104452` spent all five slots that way, and every verdict came back
  `rejected_artifact_class`). `falsifier-20260920-consolidation.py --corpus` still prints
  `eligible_keys=761` by design, because it measures the *pre-gate* pool; only this
  tool's line is evidence about the post-gate one. #1181 recorded this consequence under
  "what the fix does NOT cure" and left it unowned — #1287 is the item that closes it.
  Two things this gate must never do: drop a key for the **age** of its newest snapshot
  (`has_error_recovery` is the discriminator, not a date, and all 108 surviving keys are
  as stale as the 653 dropped ones), or drop one for a field its front matter does not
  record — a key the file cannot decide is a `KEEP`, which is why an `error` key (the
  field is absent on all 126 of them) and a `success` key (whose file has never carried
  `params_signature`, 102 files) are never counted in `dropped_by_source_gate`.
"""


def patched_text(text: str) -> str:
    """The runbook with the clause inserted. Raises ValueError if it cannot be placed."""
    if MARKER in text:
        return text
    if text.count(ANCHOR) != 1:
        raise ValueError(f"anchor found {text.count(ANCHOR)} times, need exactly 1")
    head, tail = text.split(ANCHOR, 1)
    if "### 1.3 Build work list" not in head or "## Phase 2:" not in tail:
        raise ValueError("anchor is not inside Phase 1.3; refusing to insert")
    return head + ANCHOR + CLAUSE + tail


def main(argv: list[str]) -> int:
    dry = "--dry-run" in argv
    try:
        text = SKILL.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"cannot read {SKILL}: {exc}")
        return 1
    if MARKER in text:
        print("already applied")
        return 0
    try:
        new = patched_text(text)
    except ValueError as exc:
        print(exc)
        return 1
    if dry:
        print(f"dry run: would insert {len(CLAUSE)} chars after the Phase 1.3 evidence gates")
        return 0
    SKILL.write_text(new, encoding="utf-8")
    print(f"applied: {SKILL}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
