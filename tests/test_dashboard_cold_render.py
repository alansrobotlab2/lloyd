"""#1204: one cold `/api/dashboard` cycle over the real board, timed.

The page's render is gated on this one request (`DashboardPage.tsx:1146`
renders "Loading dashboard…" until the whole payload arrives), so page-render
time equals API time. Measured at triage, 2026-09-16: a cold cycle cost
**15.22 s over a 1,142-file board**, and the user saw "Loading dashboard…" for
exactly that long, recurring every 60 s on the scorecard TTL rather than once.

The budget is a MEASURED number, not an aspirational one, and since #1858 it
carries the corpus it was measured on with it. It was 6.0 s on the 1,142-file
board of 2026-09-16, then 8.0 s on a 1,450-file board on 2026-09-25, and both
times the board outgrew the bound rather than the code getting faster: the node
became the commonest parallel-only flake in the gate ledger — 11 rows naming it
before 2026-09-28 and 6 more after (`promotions.jsonl`, rung `tests`) — because
a cycle over a bigger corpus costs more and the number did not know it. So now:
`CALIBRATED_BOARD_FILES` records the board the table beside `COLD_BUDGET_S` was
walked over, `cold_budget_for_board` charges a cycle the same per-file allowance
spread over the board it actually walked, and a board that has drifted more than
`BOARD_DRIFT_FRACTION` away reddens
`test_the_board_is_still_the_size_the_budget_was_calibrated_on`, which tells the
reader to RE-BASE the calibration rather than raise the budget. Corpus growth
then costs a measurement, not a flake. (The loader swap alone measured 5.00 s
cold over the 1,142-file board; anything below it needs recommendation B — one
shared board walk, analytics off the read path — which #1199 holds.)

Four things keep this from being a stopwatch that always passes:

* the board file count is printed and asserted > 0 — the walk is the thing
  being timed, so a cycle over zero files is no measurement at all;
* the payload must actually carry the two heavy sections, because a `_gather`
  that swallowed an exception would return `{"error": ...}` in milliseconds and
  read as a speed-up;
* both caches are cleared before EVERY timed cycle, including the confirming
  one. A warm cycle costs 0.06 s and would trivially satisfy the bound while
  proving nothing about the cold path the user pays;
* one breach is a sample, not a verdict. A breaching cycle is re-measured once,
  cold, and the node reddens only if that second cycle breaches too — which is
  the difference between a load spike under `xdist -n 8` and a slow dashboard.
  `enforce_cold_cycle_budget` never runs a third cycle, so it cannot grind its
  way to a green.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

import pytest

from app.routers import dashboard as dash
import board_presence
from board_presence import board_files_or_stop, timed_ledger_or_stop
from scripts.automod import backlog as B

#: The board this calibration was walked over, in item files (`*.md` with a
#: numeric name, which is how the dashboard counts them — the same rule
#: `board_files_or_stop(numeric_names=True)` applies). Re-based 2026-11-07 (item
#: #3867) from 21411475, and all thirteen serial cycles below printed exactly this
#: count, so the denominator and the durations describe one board and not two. The
#: count this replaces (`RETIRED_BOARD_FILES` below) described a board 16.9% smaller
#: than the one the cycle walks today: 10 files past that count's own +20% upper
#: edge, which is this node's arithmetic rather than a census —
#: `assert_board_within_calibration_drift` puts the edge at `int(RETIRED_BOARD_FILES
#: * (1 + BOARD_DRIFT_FRACTION))` = 3,794 and its failure line printed this count
#: against that ceiling. The drift node has no third state between "re-charged" and
#: "re-base", and it was right to ask for one. The 10 and the 3,794 are re-derived
#: by `test_the_re_base_prose_recomputes_from_the_two_calibrations`, because this
#: sentence first shipped as "one file past", which understated its own node by nine.
#:
#: ONE copy of the number lives in this file. `cold_budget_for_board` divides
#: the budget by it to get the per-file allowance the cycle is charged, and
#: `assert_board_within_calibration_drift` builds the +/- band from it, so the
#: two derived quantities can never disagree about which board the budget
#: describes. `test_the_calibrated_board_count_is_written_once_in_this_file`
#: reddens if the digits are ever typed again, in code or in a comment: a prose
#: copy is the thing that goes stale, and this file's last two calibrations each
#: left one behind (the 1,142-file board and the 1,450-file one, both of which
#: the bound then walked past).
CALIBRATED_BOARD_FILES = 3804

#: Bytes of `promotions.jsonl` the calibrated cycle decoded, for the same
#: reason. NOT part of the denominator — the budget scales with the board, not
#: with the ledger, because pinning bytes decoded is a second change and #1858
#: left that ruling open (owed clause 2: the ledger is read by the cycle, grows
#: faster than the board, and is in none of the retention sweep's bounded
#: stores). A reader comparing today's cost against this table has to know both
#: inputs, since ledger growth can exhaust this budget with no code change and
#: no drift-test warning.
CALIBRATED_LEDGER_BYTES = 28_960_660

#: The calibration this one replaces, as DATA rather than as a sentence, and that
#: distinction is the point: every movement this file's prose quotes about a re-base
#: (how far the tail moved, how far the board moved, how much dearer the walk went
#: per item file, how many files past the retired band's edge the new count was) is
#: arithmetic on these two numbers and the constants above, and
#: `test_the_re_base_prose_recomputes_from_the_two_calibrations` re-runs that
#: arithmetic against the prose. Both figures below are quoted in the record above
#: `COLD_P90_S`: the 3,162-item board walked over 2026-10-24 from 66f62027, whose
#: larger set came out at 10.53 s. They are also the numbers a reader needs to check
#: the drift node's own demand: this count against a retired edge of 3,794.
RETIRED_BOARD_FILES = 3162
RETIRED_P90_S = 10.53

#: The cold cycles `COLD_P90_S` is derived from: two serial sets measured over the
#: board named by `CALIBRATED_BOARD_FILES`, each entry a pair of the 1-minute load
#: average the node printed beside them (`machine at measurement: load 1/5/15 =`)
#: and the durations in seconds, in the order they ran.
#:
#: They are data rather than a sentence because the constant used to be a literal
#: typed beside a prose table quoting the same measurements, and the only rail on it
#: read `COLD_BUDGET_S == COLD_P90_S * COLD_BUDGET_MARGIN` — the budget's own
#: definition, so the rail could not fail and could not notice a p90 that had
#: quietly absorbed the margin. `COLD_P90_S` is now computed from these cycles, and
#: `test_the_calibration_table_and_the_cycles_it_records_agree` re-derives every
#: figure the table below quotes from them too, so the prose, the cycles and the
#: number the dashboard is charged are one measurement and not three claims.
_CALIBRATION_CYCLES_S: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("3.69-7.66", (16.11, 16.36, 16.38, 16.50, 16.69, 16.34, 14.51, 14.12)),
    ("3.33-3.91", (15.01, 15.17, 14.71, 14.44, 14.68)),
)


def _median_s(samples):
    """Middle of the sorted samples: the mean of the two middles for an even count."""
    ordered = sorted(samples)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2


def _p90_nearest_rank(samples):
    """The value at rank ceil(0.9 n) of the sorted samples.

    Nearest-rank rather than an interpolation, because that is the convention every
    row of this calibration's table has reported its p90 by: with 8 samples the rank
    is the maximum, and with 6 it is the maximum too. Integer arithmetic so the rank
    cannot land between two samples on a float boundary.
    """
    ordered = sorted(samples)
    rank = (9 * len(ordered) + 9) // 10
    return float(ordered[min(rank, len(ordered)) - 1])


#: The larger of the two measured tails, in seconds.
#:
#:     measured 2026-11-07 from d0857445 in a round worktree, 32 cores
#:     board  = CALIBRATED_BOARD_FILES item files (the numeric-name subset);
#:              all thirteen runs below printed that count
#:     ledger = the live store. `timed_ledger_or_stop` repoints it only when the
#:              configured state dir holds NO ledger, and this round's state dir was
#:              new and held none, so all thirteen cycles printed `REPOINTED: ...
#:              promotions.jsonl has no ledger, so the cycle reads the live one
#:              instead` over a store of 30,770,490 B — +6.25% on
#:              `CALIBRATED_LEDGER_BYTES`, inside the drift band, so no drift fault
#:              was raised and this re-base left the centre alone too.
#:
#:              THAT IS NOT THE SHAPE A GATE RUN DECODES, and the gap between the
#:              two is larger than the store's own growth: that helper takes the
#:              configured dir whenever it holds ANY bytes, and a gate's round state
#:              dir holds a ledger of its own that other tests in the same session
#:              append rows to, so the graded cycle decodes a few kilobytes and not
#:              these 30 MB. Read once on each side of this item: the review rung's
#:              own cycle printed `ledger for this pair: ... (6,140 bytes) | the
#:              configured state dir`, copied from the `review` rung's detail in
#:              `~/.local/state/lloyd-automod/rounds/SM_20261106_233502/gate.json`,
#:              and this round reproduced the substitution over a 6,000-byte scratch
#:              ledger under `LLOYD_AUTOMOD_STATE`, which printed that configured
#:              path beside a cold cycle of 10.848 s — against the 14.12-16.69 s of
#:              the serial runs below. The table therefore prices the live store, and
#:              a cycle inside a gate is cheaper than this table by most of what the
#:              store's decode costs. `live_ledger_bytes()` below deliberately reads
#:              the OPPOSITE precedence from `timed_ledger_or_stop`, for that reason.
#:              The centre is left
#:              alone because it is pinned by equality in
#:              `tests/test_retention_sweep.py` (#2043's whole case is that the fold
#:              lands the store inside the band around THAT number) and because #2071
#:              ruled that the centre is not the drift-drift lever. The fold IS
#:              running — two archived buckets sit beside the store, the older of
#:              them 9,392,103 B written 2026-11-05 (`ls -la
#:              ~/.local/state/lloyd-automod/promotions.jsonl*` re-checks it) — and the
#:              live store is still 30,770,490 B, +6.25% over this centre. That is not
#:              a contradiction, and it is not the store refilling: store 13 of the
#:              sweep is an AGE rule (`LEDGER_ARCHIVE_AGE_DAYS = 14`), not a size rule,
#:              so nothing in it bounds the file's growth at all. At the published
#:              1,992,001 B/day the band's upper edge (`int(centre * 1.2)`) is about two
#:              days off, so this drift node will be asked to re-base again well inside
#:              one band's lifetime — which is #2071's other half, not this file's. The durations
#:              below are of the real store at its real size, and the budget those
#:              thirteen cycles ran against was still the previous 15.83 s: six of
#:              them breached it on the first sample, all six in the first set, and
#:              each was retired by its own confirming cycle, so no node failed. One
#:              breach buying a second cold sample is what the rule is FOR, and the
#:              sentence below is right that six retirements say nothing about the
#:              centre on their own — but the six sit above the very tail that moved,
#:              and the only reading of the two together is that the centre had moved
#:              and not just the denominator.
#:     serial, 8 runs, load 3.69-7.66:  min 14.12  median 16.35  p90 16.69  max 16.69
#:     serial, 5 runs, load 3.33-3.91:  min 14.44  median 14.71  p90 15.17  max 15.17
#:
#:     #3867, 2026-11-07, from d0857445, in a round worktree, 32 cores. What this
#:     re-base is FOR is the board: the count it replaces described a board 16.9%
#:     smaller than the one the cycle walks, 10 files past the retired band's upper
#:     edge, and the drift node asked for exactly this. The measured tail moved
#:     +58.5% while the board moved +20.3%, so the walk itself went from 3.33 ms to
#:     4.39 ms per item file — 31.7% dearer per file than the cycle that came out at
#:     the retired p90. Those five figures are arithmetic on `RETIRED_P90_S` and
#:     `RETIRED_BOARD_FILES`, and this sentence shipped once as "52.2%" — the
#:     PREVIOUS re-base's pair (6.92 to 10.53) pasted into the new record — which is
#:     why `test_the_re_base_prose_recomputes_from_the_two_calibrations` now recomputes
#:     each of them instead of trusting the prose. #1204 asks where the cold cost
#:     goes; this is a third independent sighting that it is not file count, and the
#:     per-file charge is kept rather than a flat number for that reason.
#:
#:     The quieter set (load 3.33-3.91) sits 9.1% below the louder at the tail and
#:     10.0% below it at the middle, which is the direction load points in. Against
#:     that 1.52 s spread between the two sets' own tails, the 25%
#:     margin is 4.17 s of slack over the p90. One `-n 8 --dist loadfile` pass over
#:     this file surfaced a cold cycle at 12.19 s — 27% BELOW this serial p90, so
#:     eight workers are not what pushes a cycle past a serial tail; what does, on
#:     this box, is another agent's gate running alongside this one, and no sample
#:     here models that. A parallel sample is a ceiling to check a budget against, not
#:     a distribution to take a p90 from, and it is in that capacity it is recorded.
#:
#:     To restore the value this one replaced, restore its inputs: the 3 162-item
#:     board and the live ledger at the centre pinned above, walked over 2026-10-24
#:     from 66f62027, which came out at 10.53 s.
#:
#:     What the next re-base will read: at the board's own recent rate the band's
#:     upper edge is days off, not a fortnight. The answer to a third breach in a row
#:     is #1204's cheaper section — one shared walk for the three sections that each
#:     re-walk the board today — and not a wider band and not a bigger margin.
#:
#: Derived rather than typed: the larger of the two rows' p90s, computed from
#: `_CALIBRATION_CYCLES_S` below the table's own numbers. A re-baser replaces the
#: cycles, and the constant, the budget and the per-file charge all follow; the node
#: that reads the table against those cycles is what says so if the prose was left
#: behind.
COLD_P90_S = max(_p90_nearest_rank(cycles) for _, cycles in _CALIBRATION_CYCLES_S)

#: Stated margin over that p90 (#1858 clause 1): 25%, arithmetic rather than the
#: "~30% headroom" the 2026-09-25 re-base claimed in prose. It has to cover the
#: worker contention the re-measurement does not absorb. It used also to have to
#: stay under the cost this check was written against, and the re-base above put the
#: enforced bound past it, so what follows is read rather than repeated, and one
#: node re-derives it from the constants:
#: The enforced bound is 20.86 s, 37.1% ABOVE the 15.22 s triage cost, so a cycle
#: that regressed to the 2026-09-16 shape would now pass this node. That is a fact
#: about the corpus and not a lever, and `TRIAGE_COLD_S` plus
#: `test_the_enforced_bound_is_pinned_to_the_ceiling_it_outran` hold it: the same
#: node keeps the half that did NOT move, which is that per item file the bound is
#: 5.48 ms against the 13.33 ms/file that triage run recorded — 2.4x tighter — so a
#: walk that regressed to the triage rate still breaches this budget by 2.4x. The
#: way back under the absolute ceiling is a cheaper walk, not a thinner margin: the
#: p90 that would put the bound under the triage cost at this margin is 12.18 s, 27%
#: BELOW the tail measured on this board, so #1204's one shared board walk is the
#: route and dropping this number would only redden the node it is meant to protect.
#: A re-baser changes the table and, if they must, this number; the budget follows
#: arithmetically, and so must the sentence above — or that node says so.
COLD_BUDGET_MARGIN = 1.25

#: Budget for one cold cycle over the board named by `CALIBRATED_BOARD_FILES`,
#: charged per file by `cold_budget_for_board`. The value this re-base replaced
#: was 8.65 s, on the 2,628-file board walked over a live ledger of 29,444,519
#: bytes, measured 2026-10-14 from 53a7fd11; to restore it, restore those two
#: inputs and the p90 they sat on. Before that, 9.74 s was the budget on a board of
#: 2,180 item files walked over a live ledger of 29,188,189 bytes, measured
#: 2026-10-05 from 19acea6c. Before that, 9.55 s was the budget on
#: a board of 1,804 item files with the old ledger centre. Before that,
#: 8.0 s was the calibration on a 1,450-file board with a 6,286,192-byte ledger,
#: where the node measured 6.02 s and 6.16 s (round SM_20260925_205059's review
#: run, 2026-09-25, load average 11). Triage baseline: 15.22 s over 1,142 files,
#: 2026-09-16 — and this is the first calibration whose bound does NOT fire short
#: of it. The two before it did: the retired p90 put the bound at 13.16 s, 13.5%
#: under the triage cost, and the one before that at 8.65 s, 43.2% under it. Where
#: this bound actually stands, and the half of the comparison that survived, is the
#: sentence above `COLD_BUDGET_MARGIN` — and that sentence is now re-derived by a
#: node rather than trusted.
COLD_BUDGET_S = COLD_P90_S * COLD_BUDGET_MARGIN

#: How far the live board may move from the calibrated one before the budget
#: stops describing it (#1858 clause 3).
BOARD_DRIFT_FRACTION = 0.20

#: The pain this file exists to prevent, in the unit the user felt it: one cold
#: cycle timed at #1204's triage on 2026-09-16 over that day's board — the run whose
#: 15.22 s the page sat showing "Loading dashboard…" for, and the reason a budget
#: exists at all. It is a constant now rather than a sentence because the enforced
#: bound has just moved ABOVE it, and a ceiling nothing can name is a ceiling nothing
#: can pin: `test_the_enforced_bound_is_pinned_to_the_ceiling_it_outran` computes the
#: relation from these two numbers plus `COLD_BUDGET_MARGIN`, and reddens when the
#: prose above `COLD_BUDGET_MARGIN` disagrees about which side of the ceiling the
#: bound is on, or by how much. The percentage is always relative to the triage cost
#: — `(bound - 15.22) / 15.22` — because the file had two conventions for it and one
#: of them is now retired.
TRIAGE_COLD_S = 15.22
TRIAGE_BOARD_FILES = 1142


def ms_per_item_file(seconds: float, board_files: int) -> float:
    """Milliseconds of cycle per item file: the unit the two corpora share.

    The only unit in which a 1,142-file measurement of 2026-09-16 and a
    board-three-times-that-size measurement of today can be compared at all. The
    absolute seconds cannot be, and this file spent two calibrations saying the
    ceiling they defined was intact while the corpus tripled underneath it.
    """
    return seconds / board_files * 1000.0


def cold_budget_for_board(board_files: int) -> float:
    """The budget owed by a cycle that walked `board_files` item files.

    `COLD_BUDGET_S` is the price of the calibrated board; spread over the files
    it was measured on it is a per-file allowance, and that allowance times the
    board actually walked is what this cycle pays. The floor at `COLD_BUDGET_S`
    means a board *smaller* than the calibrated one does not buy a tighter
    bound — the absolute number is the one the triage baseline is readable
    against, and a shrunken corpus is a calibration event anyway (the drift test
    says so). Growth past `BOARD_DRIFT_FRACTION` is not absorbed forever either:
    it reddens the drift test, which asks for a re-measurement rather than a
    bigger margin.
    """
    per_file = COLD_BUDGET_S / CALIBRATED_BOARD_FILES
    return max(COLD_BUDGET_S, per_file * board_files)


def assert_board_within_calibration_drift(live_board_files: int) -> None:
    """Redden when the board is no longer the corpus this budget measured.

    The band is `CALIBRATED_BOARD_FILES +/- BOARD_DRIFT_FRACTION`, derived from
    the constant rather than restated, so re-basing the calibration moves the
    band with it and cannot leave a stale +/-20% of a dead number behind.
    """
    low = int(CALIBRATED_BOARD_FILES * (1.0 - BOARD_DRIFT_FRACTION))
    high = int(CALIBRATED_BOARD_FILES * (1.0 + BOARD_DRIFT_FRACTION))
    assert low <= live_board_files <= high, (
        f"the cold-cycle budget is calibrated on a board of "
        f"{CALIBRATED_BOARD_FILES} item files and this board holds "
        f"{live_board_files} — outside the "
        f"+/-{BOARD_DRIFT_FRACTION:.0%} band ({low}..{high}). RE-BASE THE "
        f"CALIBRATION, do not raise the budget: re-run the node the way the "
        f"table beside `COLD_BUDGET_S` says (serial runs and `-n 8` runs over "
        f"the real board), then set `COLD_P90_S`, `CALIBRATED_BOARD_FILES` and "
        f"`CALIBRATED_LEDGER_BYTES` to what those runs measured. Raising "
        f"`COLD_BUDGET_MARGIN` instead keeps a green node that no longer "
        f"measures the board anyone has."
    )


def assert_ledger_within_calibration_drift(live_ledger_bytes: int) -> None:
    """Redden when the ledger this cycle decodes is no longer the one measured.

    The same band as the board\'s, applied to the cycle\'s other input. Cost has
    two inputs and only one of them used to be checked: the board walked and the
    ledger decoded, and the unchecked half was the cheaper way to break the
    budget — `scripts.automod.state` re-reads `promotions.jsonl` whole on every
    cycle that misses its cache, and skips that cache entirely above 64 MiB, so a
    store nothing bounds can exhaust this budget with no code change and no
    warning. That is #1858\'s owed clause 2, and #1975 closed it by giving the
    store an owner.

    The remedy points the other way from the board\'s. A board grows only through
    work; this store has a bound now (`retention-sweep.py`\'s promotion ledger
    rung, store thirteen, which archives rows past `LEDGER_ARCHIVE_AGE_DAYS` into
    a gzip beside the live file), so the first question when this reddens is
    whether that rung ran and refused — and only then whether the calibration is
    simply older than the ledger. What is never the remedy is
    `COLD_BUDGET_MARGIN`: raising it accepts the slower cycle and leaves the
    store growing, which is the one thing the bound exists to prevent.

    `live_ledger_bytes` is a parameter and the caller reads it, exactly as the
    board arm does, so the band and the measurement are one comparison rather
    than two claims that have to be kept in agreement by hand.
    """
    low = int(CALIBRATED_LEDGER_BYTES * (1.0 - BOARD_DRIFT_FRACTION))
    high = int(CALIBRATED_LEDGER_BYTES * (1.0 + BOARD_DRIFT_FRACTION))
    assert low <= live_ledger_bytes <= high, (
        f"the cold cycle is calibrated on {CALIBRATED_LEDGER_BYTES:,} decoded "
        f"ledger bytes and the live `promotions.jsonl` holds "
        f"{live_ledger_bytes:,} — outside the "
        f"+/-{BOARD_DRIFT_FRACTION:.0%} band ({low:,}..{high:,}). FOLD THE STORE "
        f"OR RE-MEASURE, do not raise the budget: run "
        f"`scripts/groundskeeper/retention-sweep.py --apply` and read its "
        f"`promotions ledger:` line (a `REFUSED` there is this drift with a named "
        f"cause and the live file untouched), and if the rung reports the window "
        f"empty then `LEDGER_ARCHIVE_AGE_DAYS` needs the owed ruling rather than "
        f"a bigger constant. Re-basing means setting `CALIBRATED_LEDGER_BYTES` to "
        f"what the live file measures now. Raising `COLD_BUDGET_MARGIN` instead "
        f"accepts the slower cycle and leaves the store unbounded."
    )


def live_ledger_bytes() -> tuple[int, str]:
    """Bytes in the promotion ledger, and which file that was.

    Deliberately the OPPOSITE precedence from `board_presence.timed_ledger_or_stop`,
    which the cycle's own timing node uses. That helper takes the configured state
    dir whenever it holds any bytes at all, because what a timing run needs is
    *something to decode*; this measurement needs the *store* the calibration was
    walked over, and the two differ exactly where a sandbox is involved. The gate
    runs a round with `LLOYD_AUTOMOD_STATE` on a scratch dir that other tests in
    the same session append a few rows to, so the configured path there holds a
    kilobyte stub — the first run of the drift node measured it and reddened on
    1,000-odd bytes against a 23 MB floor. A stub is not a shrunken store, and
    reporting one as the live size makes the check fire on the sandbox rather than
    on the world.

    So: the fixed home path (`board_presence.live_ledger()`, the seam the suite
    already uses for exactly this) first, because the calibration table describes
    that file; the configured path only when the home path holds no bytes; and a
    loud failure when neither does. Never 0, which would sit inside the band and
    read as a bounded store.

    Returns `(bytes, provenance)` — the caller prints which file it measured,
    because "inside the band" over the home store and over a configured fallback
    are different sentences.
    """
    live = board_presence.live_ledger()
    if live.is_file() and live.stat().st_size > 0:
        return live.stat().st_size, f"the live store ({live})"

    from scripts.automod import state as st
    configured = Path(st.LEDGER_PATH)
    floor = int(CALIBRATED_LEDGER_BYTES * (1.0 - BOARD_DRIFT_FRACTION))
    if configured.is_file() and configured.stat().st_size > 0:
        size = configured.stat().st_size
        if size >= floor:
            return (size, f"the configured store ({configured}); the live path "
                         f"{live} has no bytes")
        # Below the floor with no live store to compare against, this file is a
        # run's scratch ledger and not a shrunken store: a real trim of this
        # corpus is what the fold rung does, and it says so in its own report
        # line. Naming it a drift would send a reader to look for a rung that
        # truncated a file that was never a store.
        raise AssertionError(
            f"no promotion ledger to measure: {live} is absent and {configured} "
            f"holds {size:,} bytes, under the band's {floor:,}-byte floor — that "
            f"is a run's scratch state dir, not the store "
            f"`CALIBRATED_LEDGER_BYTES` describes. Red rather than skipped: 0 or a "
            f"stub would report an unbounded store as a bounded one")

    raise AssertionError(
        f"no promotion ledger to measure at {live} or {configured}: the cold cycle "
        f"decodes this file, so `CALIBRATED_LEDGER_BYTES` cannot be checked here. "
        f"Red rather than skipped, because 0 bytes would sit inside the band and "
        f"report an unbounded store as a bounded one")


def _audit_nodes(tree: "ast.AST") -> tuple[list[str], list[str], list[str], dict[str, int]]:
    """(node names, nodes retired from a run, `live_vault` nodes, parametrized nodes).

    Read off the syntax tree rather than by grepping the file's text, and that is
    load-bearing: the first draft of this audit looked for the literal spellings
    `pytest.mark.skip` and `pytest.skip(` in the source, and reddened on this node's
    own assertion strings, which quote them. A guard that cannot tell a call from a
    mention of a call refuses the file it is standing in.

    `away` collects the two ways a node stops running without being deleted: an
    `skip`/`skipif`/`xfail` decorator, and a body-level `pytest.skip(...)` /
    `pytest.xfail(...)` / `pytest.importorskip(...)` call. `marked` is the
    `live_vault` list, which the gate deselects with `-m "not live_vault"`.
    """

    def chain(node) -> str:
        parts = []
        while isinstance(node, (ast.Attribute, ast.Name)):
            parts.append(node.attr if isinstance(node, ast.Attribute) else node.id)
            node = node.value if isinstance(node, ast.Attribute) else None
        return ".".join(reversed(parts))

    nodes: list[str] = []
    away: list[str] = []
    marked: list[str] = []
    parametrized: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.name.startswith("test_"):
            continue
        nodes.append(node.name)
        for decorator in node.decorator_list:
            name = chain(decorator.func) if isinstance(decorator, ast.Call) else chain(decorator)
            if name.endswith(("mark.skip", "mark.skipif", "mark.xfail")):
                away.append(f"{node.name}: @{name}")
            elif name.endswith("mark.live_vault"):
                marked.append(node.name)
            elif name.endswith("mark.parametrize"):
                args = getattr(decorator, "args", [])
                cases = len(args[1].elts) if len(args) > 1 and isinstance(
                    args[1], (ast.List, ast.Tuple)) else 0
                parametrized[node.name] = parametrized.get(node.name, 0) + cases
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and chain(inner.func) in (
                    "pytest.skip", "pytest.xfail", "pytest.importorskip"):
                away.append(f"{node.name}: {chain(inner.func)}() in the body")
    return nodes, away, marked, parametrized


def store_fullest_between_folds(*, live_bytes: int, cadence_days: float,
                                rate_bytes_per_day: float) -> float:
    """Bytes the store sits at at its HIGHEST on a fold every `cadence_days`.

    The fold trims the store back to its window each pass, so one cycle is: the size a
    fold leaves, then that plus everything arriving before the next pass. `live_bytes` is
    therefore the FLOOR, and there are two readings of it which #2575 made one agree with
    the other:

      * the size a fold leaves behind — what `sweep_promotions_ledger` would report as
        `after`, and what `_bytes_a_fold_leaves` computes from the fold's own plan. The
        same quantity `CALIBRATED_LEDGER_BYTES` was timed at: a store that had been
        folding for a month, which is what #2071's remedy restored.
      * the size on disk when pytest reached it. That reading is not merely elsewhere on
        the same sawtooth: it is the floor PLUS what has arrived since the last fold, and
        the bound then adds the cadence's growth again — two cycles priced as one. The gap
        is exactly what the last fold dropped, so it is 0 bytes right after a pass and a
        day's growth by the next: measured on the live store on 2026-10-11 it was 1,966,072
        bytes at 11:20Z and 834,006 bytes at 12:20Z, a fold having run in between, against
        growth of about 1.9 MB/day. #2071's own overage — the 27,066 bytes this file exists
        to re-derive — is two orders of magnitude smaller than the term.

    A census of the live file, with no cadence and no floor, answers a third question — the
    size on the day someone looked — which is what the drift node is for and not this one.
    """
    return live_bytes + cadence_days * rate_bytes_per_day


def cadence_outruns_band(*, live_bytes: int, cadence_days: float,
                         rate_bytes_per_day: float, ceiling_bytes: int) -> bool:
    """True when a fold every `cadence_days` lets the store top `ceiling_bytes`.

    This is #2071 in one line. The fold rung had existed since #1204 and
    `LEDGER_ARCHIVE_AGE_DAYS` had not moved, so the corpus was "bounded"; what made
    the bound false was that `autonomy/79-retention-sweep.md` scheduled the fold
    weekly while the store was growing at 2,208,597 B/day (measured 2026-10-02 over
    the row timestamps, 7-day mean). Seven days of that growth is 15,460,179 bytes —
    44% of the 34,752,792-byte ceiling — so the store topped the ceiling on the
    fifth day after a pass and sat 20.1% over it when the node reddened, then came
    back inside the band when the pass ran. A cadence slower than
    `(ceiling - live) / rate` days cannot bound the corpus at any point in its
    cycle, which is why a size check alone leaves the failure invisible between two
    passes and why fixing it by moving the ceiling fixes nothing: no ceiling contains
    a sawtooth whose amplitude is set by the cadence.
    """
    return store_fullest_between_folds(live_bytes=live_bytes, cadence_days=cadence_days,
                                       rate_bytes_per_day=rate_bytes_per_day) > ceiling_bytes


def centre_holds_the_sawtooth(*, floor_bytes: int, added_bytes: float,
                              drift_fraction: float) -> bool:
    """Is there ANY band centre covering a fold cycle of that shape?

    A centre covers it iff it is high enough that the post-fold floor sits above the
    band's bottom AND low enough that the floor plus one cadence's growth stays under
    its top: `(floor + added) / (1 + drift) <= centre <= floor / (1 - drift)`. Those
    two ends meet only when `added <= 2 * drift * floor / (1 - drift)` — a condition on
    the cadence and the growth that does not contain the centre at all.

    So past that cadence no value of `CALIBRATED_LEDGER_BYTES` bounds the store, which
    is why #2071 could not be fixed by moving the calibration, and why
    `test_the_board_is_still_the_size_the_budget_was_calibrated_on` can tell a reader to
    RE-BASE when the board moves while this node cannot: board growth is monotone so a
    centre can chase it, whereas a fold's amplitude is set by its own cadence.
    """
    return added_bytes <= 2 * drift_fraction * floor_bytes / (1 - drift_fraction)


def _sweep_module():
    """`scripts/groundskeeper/retention-sweep.py` loaded as a module.

    The file name has hyphens, so `import` cannot reach it; this is the same loader
    `tests/test_retention_sweep.py:74` uses, restated here rather than imported
    because this suite must not inherit that file's fixtures. Read-only — nothing
    here calls `sweep_promotions_ledger`.
    """
    script = (Path(__file__).resolve().parent.parent
              / "scripts" / "groundskeeper" / "retention-sweep.py")
    spec = importlib.util.spec_from_file_location("retention_sweep_for_cold_render",
                                                 script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _measured_growth_bytes_per_day(ledger: Path, now: float) -> float:
    """Mean bytes/day the store grew over its last 7 days, read on the fold's clock.

    `_ledger_row_seconds` is the sweep's own age rule, so a row this counts as 3 days
    old is 3 days old to the rung that would archive it — one clock, not two. A store
    younger than the window divides by the span it actually covers, so the box's first
    week yields a rate instead of a division by nothing.
    """
    mod = _sweep_module()
    floor = now - 7 * 86400
    oldest: float | None = None
    bytes_in_window = 0
    with ledger.open("rb") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            stamp = mod._ledger_row_seconds(row)
            if stamp is None:
                continue
            oldest = stamp if oldest is None else min(oldest, stamp)
            if stamp > floor:
                bytes_in_window += len(line)
    span_days = max(0.0, now - max(floor, oldest or now)) / 86400
    if span_days <= 0 or bytes_in_window == 0:
        raise AssertionError(
            f"{ledger}: no dated promotion row in the last 7 days, so the drift has no "
            f"measurable rate. Red rather than skipped — a store whose growth cannot be "
            f"read is exactly the case a byte-count check is blind to")
    return bytes_in_window / span_days


def _fold_cadence_days(front: dict) -> float | None:
    """The cadence one task's front matter schedules the fold at, in days.

    Resolved through `app.autonomy._frequency_interval_seconds` — the ONE predicate the
    dispatch loop and the stall alert already share (`next_run_gap`'s own words: two
    surfaces had to agree, so they were given one arithmetic) — rather than against
    `FREQUENCY_INTERVALS` in here. That table is what the WORD resolves to, and the
    scheduler consults `runs_per_day:` FIRST (`_frequency_interval_seconds` itself; and
    `scripts/autonomy/validate_tasks.py` states the same rule while it lints the pair for
    #2445: "#30 declared `frequency: daily` beside `runs_per_day: 3` and the scheduler
    obeys the number"). The word-only read this file shipped with was wrong in both
    directions: a fold scheduled twice a day was charged one day (a false red, since the
    sawtooth halves), and a schedule written the sub-daily way the linter itself names — a
    word outside the four-word map plus a matching `runs_per_day:`, which its own message
    calls "#24's `6x-daily` + `runs_per_day: 6`" (`scripts/autonomy/validate_tasks.py:374-383`)
    — resolved to NO interval at all, so the same read turned the remedy this file recommends
    into "no fold whatsoever, never due, ever".

    `None` means the scheduler resolves no interval at all, which is never due, ever: the
    caller asserts rather than assumes, because that failure is a ledger with no fold.
    """
    from app.autonomy import _frequency_interval_seconds

    seconds = _frequency_interval_seconds({"frequency": front.get("frequency", ""),
                                          "runs_per_day": front.get("runs_per_day")})
    return None if seconds is None else seconds / 86400.0


def _declared_fold_cadence_days() -> float:
    """The interval the retention rung is actually scheduled at, in days."""
    from app.autonomy import AUTONOMY_DIR

    task = AUTONOMY_DIR / "79-retention-sweep.md"
    assert task.is_file(), (
        f"{task} is the schedule the promotions fold runs on; it is not readable, so "
        f"the cadence this node is about cannot be checked")
    import yaml
    front = yaml.safe_load(task.read_text(encoding="utf-8").split("---\n", 2)[1])
    cadence = _fold_cadence_days(front)
    assert cadence is not None, (
        f"{task}: `frequency: {front.get('frequency')!r}` with "
        f"`runs_per_day: {front.get('runs_per_day')!r}` resolves to no interval through "
        f"`app.autonomy._frequency_interval_seconds`, which is never due, ever — the "
        f"scheduler would dispatch this task no times, leaving the ledger with no fold "
        f"whatsoever and no alarm about it")
    return cadence


def test_a_fold_cadence_too_slow_for_the_growth_rate_is_not_bounded():
    """The arithmetic that made #2071 red, pinned where the gate can see it.

    `test_the_ledger_is_still_the_size_the_budget_was_calibrated_on` reddened at
    34,779,858 bytes against a 34,752,792-byte ceiling — 27,066 bytes over, which
    reads like a rounding accident and is not one. The store had grown 2,208,597
    bytes/day while `autonomy/79-retention-sweep.md` declared `frequency: weekly`, so
    15.5 MB of rows arrived between two passes and the band — whose entire width is
    40% of whatever `CALIBRATED_LEDGER_BYTES` says on the day you read this — was
    crossed on the way up rather than by any code change. The two figures in this
    paragraph are that day's; the `ceiling` the node computes is derived from the
    calibration, so it moves whenever `CALIBRATED_LEDGER_BYTES` moves and the assert
    below is what says which way it went. The 2026-10-05 cold-cycle re-base did NOT
    move it: that re-base re-measured the board count and the cycle p90 and left the
    ledger centre at `CALIBRATED_LEDGER_BYTES`, so `ceiling` is still 34,752,792 and
    this assert is the witness that it is, not evidence of a move.

    The two figures it computes with are the ones measured on this box on 2026-10-02
    and pinned here rather than re-read, so the node is decidable on any box and in
    both directions: the weekly cadence that reddened the item is refused, the daily
    one this round schedules is accepted, the boundary sits where the arithmetic puts
    it, and a cadence faster than one that passes is never refused — a check that
    reddens at every cadence points at a lever that cannot fix it. Both this node and
    the live one call `cadence_outruns_band`, so the model cannot drift from its own
    witness.
    """
    floor = 27_265_006          # bytes a pass left behind, 2026-10-02 10:35
    rate = 2_208_597.0          # B/day, 7-day mean of row-dated bytes an hour later
    ceiling = int(CALIBRATED_LEDGER_BYTES * (1.0 + BOARD_DRIFT_FRACTION))
    assert ceiling == 34_752_792, f"ceiling moved to {ceiling:,}: recalibrate here too"

    assert cadence_outruns_band(live_bytes=floor, cadence_days=7.0,
                                rate_bytes_per_day=rate, ceiling_bytes=ceiling), (
        "a weekly fold over a store growing 2.2 MB/day was accepted: that is the "
        "cadence #2071 was filed for, and accepting it re-opens the item")
    assert not cadence_outruns_band(live_bytes=floor, cadence_days=1.0,
                                    rate_bytes_per_day=rate, ceiling_bytes=ceiling), (
        "the daily cadence this round schedules was refused, so the fix does not fix "
        "the arithmetic it names")

    # The boundary to two decimals: (ceiling - floor) / rate = 3.39 days. At or under
    # it the store is inside the band at the top of its cycle; over it, it is not.
    # That is what makes the red deterministic instead of a matter of timing.
    assert (ceiling - floor) / rate == pytest.approx(3.39, abs=0.005), (
        f"boundary moved to {(ceiling - floor) / rate:.2f} d: the pinned figures and "
        "the live measurement have diverged")
    assert not cadence_outruns_band(live_bytes=floor, cadence_days=3.39,
                                    rate_bytes_per_day=rate, ceiling_bytes=ceiling)
    assert cadence_outruns_band(live_bytes=floor, cadence_days=3.40,
                                rate_bytes_per_day=rate, ceiling_bytes=ceiling)

    # A fold that ran with no gap at all must never be refused: if it were, this would
    # be a red no schedule can cure, which is worse than no check.
    assert not cadence_outruns_band(live_bytes=floor, cadence_days=0.0,
                                    rate_bytes_per_day=rate, ceiling_bytes=ceiling), (
        "a fold that ran with no gap at all was refused — the node would then be "
        "unfixable by the only lever it points at")

    # And why this item could not be fixed by re-basing the centre. `floor` is inside
    # the band's bottom at the shipped centre, which is why the drift node was green at
    # 10:36 and red by 10:31; a weekly cycle from that floor tops 42.7 MB. For any
    # centre to cover it the centre would have to be at least 35.6 MB, whose own bottom
    # (28.5 MB) is above the floor — so the admissible interval is empty and re-basing
    # would have bought one green census and lost the check. The board's sibling node
    # tells a reader to RE-BASE when the board grows; a fold's sawtooth is a different
    # shape and the cadence is the only lever.
    assert not centre_holds_the_sawtooth(floor_bytes=floor, added_bytes=7 * rate,
                                         drift_fraction=BOARD_DRIFT_FRACTION), (
        "some centre is claimed to bound a weekly cycle at the pinned growth rate — if "
        "the band's +/-20% or these figures moved, re-derive this before repeating the "
        "claim that re-basing could not have fixed #2071")
    assert centre_holds_the_sawtooth(floor_bytes=floor, added_bytes=1 * rate,
                                     drift_fraction=BOARD_DRIFT_FRACTION), (
        "no centre bounds even a DAILY cycle at the pinned floor: the floor and the "
        "band have come apart, and a re-base will be needed as well as a cadence")
    admits = ((floor + rate) / (1 + BOARD_DRIFT_FRACTION),
              floor / (1 - BOARD_DRIFT_FRACTION))
    assert admits[0] <= CALIBRATED_LEDGER_BYTES <= admits[1], (
        f"a daily cycle at this rate admits a centre in [{admits[0]:,.0f}, "
        f"{admits[1]:,.0f}] and {CALIBRATED_LEDGER_BYTES:,} is outside it, so the "
        "constant #2071 left alone now needs its own ruling")
def _bytes_a_fold_leaves(ledger: Path, now: float, mod) -> int:
    """The size `sweep_promotions_ledger` would leave this store at, run right now.

    Computed by calling the fold's own `_archive_plan` and subtracting the bytes it
    decides to archive — not by re-deriving "older than the window" here. The age is only
    the first of that predicate's terms: an unparseable or unstamped line is kept
    (`_archive_plan:1782-1788`, and a boolean `ts` reads as no age because
    `_ledger_row_seconds` rejects `bool`), and every round's NEWEST row is held back however
    old it is (`:1767-1776` builds the newest map, `:1792-1796` keeps on it) so the board can
    still answer "is this closed?" while that round's record sits in the archive. On the
    live store that protection is not marginal — a dry run over a copy reported 1,376
    candidates, of which 817 survived as newest-of-round and only 559 were actually
    archived (830,001 bytes). A date filter that dropped all 1,376 would understate the
    floor by 1.1 MB and make the bound easier to satisfy than the fold that sets it, the
    exact failure the two-sides-one-predicate rule exists to prevent.

    Compaction (`_compact_oversized_rows`, the other thing the fold does on its apply
    path) shrinks surviving rows and is not modelled, so what this returns is an upper
    bound on the floor: the bound can only be too tight, never too loose.

    Bytes, not characters, and the `(line, row)` pair `_archive_plan` is called with is
    built the way `sweep_promotions_ledger:2019-2027` builds it — unparseable lines arrive
    as `None` rather than being skipped — because the indices it returns are into that list.
    """
    rows = []
    for line in ledger.read_bytes().splitlines(keepends=True):
        body = line.strip()
        try:
            row = json.loads(body.decode("utf-8", "replace")) if body else None
        except ValueError:
            row = None
        rows.append((line, row if isinstance(row, dict) else None))
    archive_idx, _kept = mod._archive_plan(rows, now)
    dropped = set(archive_idx)
    return sum(len(line) for i, (line, _r) in enumerate(rows) if i not in dropped)


def test_the_declared_fold_cadence_keeps_the_store_inside_the_band():
    """The cadence that schedules the fold is part of the bound, and is checked.

    #2071's red was a store 20.1% over its ceiling with nothing wrong in the code: the
    fold rung existed, `LEDGER_ARCHIVE_AGE_DAYS` had not moved, and the sweep still
    trimmed the store back inside the band the moment it ran. What was wrong was the
    schedule — `autonomy/79-retention-sweep.md` declared `frequency: weekly` — and a
    size check alone cannot see that, because between two passes the store is
    legitimately and predictably over the line while the file on disk looks fine.

    Both inputs are read rather than copied so the node goes red when either moves:
    the cadence from the task file that schedules the fold, the growth from the store's
    own row timestamps. The cadence is resolved by `_fold_cadence_days` — the
    scheduler's own predicate, `runs_per_day:` before the word — and the store entering
    the cycle is `_bytes_a_fold_leaves`, the fold's own keep/drop partition, not the
    size on disk: the sawtooth this models starts at the post-fold floor and ends one
    cadence later, and a live size read between two folds is the peak of one cycle
    charged with another's growth, which grows with the gap between fold runs.

    Deliberately NOT marked `live_vault` even though it reads the live store and
    `~/obsidian/autonomy/`, both of which no round controls: that marker is how a node
    opts out of the gate's run, and this check's whole value is that the gate sees it.
    The file's sibling drift nodes are unmarked for the same reason, and
    `test_no_node_of_this_file_is_marked_or_deleted_to_get_it_green` holds the file at
    zero markers so the exemption cannot be added later from inside the file.
    """
    led = board_presence.live_ledger()
    if not led.is_file():
        raise AssertionError(
            f"no promotion ledger at {led}: this node measures the store's growth "
            "rate, and with no store there is no cadence to judge — red, not skipped")
    now = time.time()
    rate = _measured_growth_bytes_per_day(led, now)
    assert rate > 0, (
        f"{led}: no row timestamps landed in the 14-day window, so the growth rate is "
        "0 B/day and every cadence would look safe. A denominator of zero is not a "
        "clean result")
    cadence_days = _declared_fold_cadence_days()

    live_bytes = _bytes_a_fold_leaves(led, now, _sweep_module())
    ceiling = int(CALIBRATED_LEDGER_BYTES * (1.0 + BOARD_DRIFT_FRACTION))
    peak = store_fullest_between_folds(live_bytes=live_bytes,
                                       cadence_days=cadence_days,
                                       rate_bytes_per_day=rate)
    print(f"post-fold floor {live_bytes:,} B (on disk now {led.stat().st_size:,} B); "
          f"growth {rate:,.0f} B/day; fold cadence {cadence_days:g} d; ceiling "
          f"{ceiling:,} B; peak {peak:,.0f} B")
    assert not cadence_outruns_band(live_bytes=live_bytes, cadence_days=cadence_days,
                                    rate_bytes_per_day=rate, ceiling_bytes=ceiling), (
        f"a fold every {cadence_days:g} d leaves the store at {peak:,.0f} bytes at its "
        f"highest against a {ceiling:,}-byte ceiling, so the size band is crossed by "
        f"arithmetic about {cadence_days * rate - (ceiling - live_bytes):,.0f} bytes "
        "before it is crossed by anything the code does. At this growth rate the "
        f"cadence fits in {(ceiling - live_bytes) / rate:.2f} d — schedule the fold "
        "more often, or make a ruling on what the band is worth if it cannot.")

def test_the_cadence_and_the_post_fold_floor_are_read_the_way_their_owners_read_them():
    """Both inputs the cadence bound is built from, pinned against the code that owns them.

    Each is a reader this file wrote for itself, which makes them exactly where a number
    drifts out of agreement with the thing it measures — #1121's two-surfaces failure, and
    the one that made this item red twice from a single cause.

    The `None` and sub-daily cases are the point, not decoration. A sub-daily cadence is
    writable: `scripts/autonomy/validate_tasks.py:374-383` accepts a word outside the
    four-word map so long as `runs_per_day:` carries it, in the reader's own words "the
    `Nx-daily` + `runs_per_day: N` idiom … not a contradiction but a sub-`hourly` cadence
    the word alone cannot express" (#24's `6x-daily` + 6). Verified against the linter for
    this node's two shapes: `2x-daily` + 2 and `every-4h` + 6 pass it, while a word-only
    `2x-daily` is an error — "the scheduler resolves no interval and will never dispatch it"
    — which is exactly the `None` case below. A `runs_per_day:` beside an IN-map word that
    disagrees is also an error (`:387-392`, 5% tolerance at `:124`/`:390`, #30 declaring
    `daily` while running 3×/day), and the scheduler still obeys the number; that shape is
    pinned here too, so what the bound charges never differs from what runs.

    The reader this replaces consulted the four words alone, so it charged a twice-daily
    fold as one day, and resolved both accepted sub-daily shapes to no interval at all —
    reporting "no fold whatsoever" about the very remedy this file's failure text recommends
    ("schedule the fold more often"). Advice and reader disagreed on whether it was
    followable.
    """
    import tempfile

    from app.autonomy import _frequency_interval_seconds

    # ── cadence: the number the scheduler waits, on both spellings ──────────────────
    cases = [
        ({"frequency": "daily"}, 1.0),                          # the word alone
        ({"frequency": "WEEKLY"}, 7.0),                         # the reader lowercases
        ({"frequency": "every-15min"}, 900.0 / 86400.0),
        ({"frequency": "2x-daily", "runs_per_day": 2}, 0.5),   # validate_tasks.py:374-383
        ({"frequency": "every-4h", "runs_per_day": 6}, 1.0 / 6),
        ({"frequency": "daily", "runs_per_day": 2}, 0.5),       # number wins; linter errors
        ({"frequency": "daily", "runs_per_day": 0}, 1.0),       # 0 is absent, not instant
        ({"frequency": "every-third-day"}, None),               # no interval at all
        ({"frequency": ""}, None),
    ]
    for front, expected in cases:
        assert _fold_cadence_days(front) == expected, front
        seconds = _frequency_interval_seconds({"frequency": front.get("frequency", ""),
                                               "runs_per_day": front.get("runs_per_day")})
        charged = None if seconds is None else seconds / 86400.0
        assert charged == (None if expected is None
                           else pytest.approx(expected, abs=1e-9)), (
            f"{front}: the cadence the bound charges is not the interval the scheduler "
            f"waits ({charged} against {expected}) — two surfaces, one arithmetic, #1121")

    assert _fold_cadence_days({"frequency": "daily"}) == 1.0, (
        "a 1-day cadence is the cycle the cold render was timed against (#2071 measured a "
        "20 s cold cycle while the fold ran daily), so the model below stops meaning that "
        "measurement if this stops answering 1.0")

    # ── floor: the fold's own keep/drop partition, on a ledger this node writes ──────
    mod = _sweep_module()
    now = 1_800_000_000.0
    day = 86400.0
    # (row, does the fold keep it). The shapes are the three terms of `_archive_plan`'s
    # predicate plus the protection that makes it not merely a date filter.
    cases = [
        ({"ts": now, "item_id": 7, "filler": "x" * 40}, True),                    # fresh
        ({"ts": now - (mod.LEDGER_ARCHIVE_AGE_DAYS - 0.5) * day, "item_id": 7}, True),
        ({"ts": now - 34 * day, "item_id": 7}, False),                             # expired
        ({"created_at": "2020-01-01T00:00:00+00:00", "item_id": 8}, False),        # ISO age
        ({"ts": True, "item_id": 9}, True),                                       # bool ts
        ({"item_id": 10}, True),                                                  # undated
        # Round 11's newest row survives even though it is 30 days old ("is this closed?"),
        # and its older sibling does not: the pair is what proves the rule bites both ways.
        ({"ts": now - 30 * day, "round_id": "SM_X", "landed_sha": "a" * 40,
          "item_id": 11}, True),
        ({"ts": now - 31 * day, "round_id": "SM_X", "landed_sha": "b" * 40,
          "item_id": 11}, False),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        led = Path(tmp) / "promotions.jsonl"
        kept_bytes = 0
        with led.open("wb") as handle:      # bytes, so `len` is the on-disk size
            for row, keep in cases:
                line = json.dumps(row).encode("utf-8") + b"\n"
                handle.write(line)
                if keep:
                    kept_bytes += len(line)
        modelled = _bytes_a_fold_leaves(led, now, mod)
        assert modelled == kept_bytes, (
            f"the model reports a post-fold floor of {modelled:,} B where the fold's own "
            f"keep/drop partition over the same rows keeps {kept_bytes:,} B")
        assert kept_bytes < led.stat().st_size, (
            "a synthetic ledger in which nothing is archived cannot tell the two readings "
            "of `live_bytes` apart, so the assertion above would prove nothing")
        # Second opinion from the rung itself: `sweep_promotions_ledger` in dry-run over
        # the same copy, whose report says how many bytes it WOULD drop and how many rows
        # survived as newest-of-round. A test calling the sweep must never point it at the
        # live ledger, so this runs only on the file written above.
        report = mod.sweep_promotions_ledger(False, now, ledger=led,
                                             backlog_dir=Path(tmp))
        assert report["refused"] is None, report["refused"]
        assert modelled == report["before"] - report["bytes"], (
            f"the model says {modelled:,} B survives, the fold's own report says "
            f"{report['before']:,} - {report['bytes']:,} B is dropped")
        assert report["kept_newest"] == 1, (
            f"{report['kept_newest']} rows survived as newest-of-round; the synthetic "
            "ledger above is built so exactly one does, and 0 means the protection that "
            "makes the fold more than a date filter is not exercised here")

    live = board_presence.live_ledger()
    if live.is_file():
        floor = _bytes_a_fold_leaves(live, time.time(), mod)
        assert 0 < floor <= live.stat().st_size, (
            f"post-fold floor {floor:,} B against {live.stat().st_size:,} B on disk — the "
            "floor is a partition of the store, so it cannot exceed it")


def test_no_node_of_this_file_is_marked_or_deleted_to_get_it_green():
    """#2071 clause 2: the green run is earned by the named node, not by hiding it.

    A fix that makes the corpus small and a change that removes the only reader of
    the number produce the same green run, so the difference is counted here rather
    than read off a diff. Denominators first, because a check over zero things is not
    a check — the rule this repo's own doc-claim file applies to its extract: `git
    show aaa2a5d0:tests/test_dashboard_cold_render.py | grep -cE '^(async )?def
    test_'` is 20 at #2071's base, so the floor here is 23 and a removal reddens this
    node instead of looking like tidying.

    What is counted is the spellings that retire a node from a green run: `skip`,
    `xfail` and `importorskip` anywhere in the file, and any `live_vault` mark, which
    the gate's `-m "not live_vault"` deselects. The file carried none of the first
    group at base and carries none now, and the file holds no `live_vault` node at all
    (#2071's round added one for a growth measurement and it was since dropped), so
    nothing here is deselected by the gate's marker. The assertions below are what keep
    that from becoming an unverified claim rather than a counted fact.
    """
    source = Path(__file__).read_text(encoding="utf-8")
    nodes, away, marked, parametrized = _audit_nodes(ast.parse(source))

    assert not away, (
        f"{sorted(away)} retire a node from the gate's own run: clause 2 forbids "
        f"reaching green by skipping, xfailing or deselecting one")

    assert "test_the_ledger_is_still_the_size_the_budget_was_calibrated_on" not in marked, (
        "the node #2071 was filed for is marked `live_vault`, which the gate's "
        '`-m "not live_vault"` deselects — the run would be green because the '
        "assertion never ran")
    assert "test_the_ledger_is_still_the_size_the_budget_was_calibrated_on" in nodes, (
        "the node #2071 was filed for is no longer in this file")

    # Counted, not remembered: 20 at #2071's base (the `git show aaa2a5d0 | grep -cE`
    # above), 23 at #2575's base, 24 with the node that round adds. The floor stays at 23
    # rather than rising to the current count, because this rail catches a node VANISHING,
    # and a floor a round raises to its own number satisfies that round and nothing after
    # it. (`pytest --collect-only` gives different numbers — 25 and 26 — because it counts
    # the parametrized cases; what is floored here is node definitions.)
    assert len(nodes) >= 23, (
        f"{len(nodes)} test nodes against a floor of 23: a node was removed rather than "
        f"made to pass")
    assert set(parametrized) == {
        "test_a_ledger_far_smaller_than_the_calibration_reddens_the_drift_check",
        "test_a_board_far_smaller_than_the_calibration_reddens_the_drift_check",
    }, (f"{sorted(parametrized)}: the two parametrized nodes are this file's "
        f"falsification witnesses — they are what prove the drift check can redden — "
        f"so neither may be emptied out or dropped while the file is being tidied")
    assert all(cases == 2 for cases in parametrized.values()), (
        f"{parametrized}: each of those nodes runs two cases (nothing to decode, and "
        f"a corpus truncated to noise); one case is a node that can no longer fail")


def empty_both_caches() -> None:
    """Forget every cache a cold cycle is supposed to miss.

    Both layers, always: the response cache in `app.routers.dashboard` and the
    decoded-ledger cache in `scripts.automod.state`. A cycle that hits either one
    costs milliseconds, so "cold" is a property of two stores and clearing one of
    them is how a warm cycle gets timed while reading as a cold one.
    """
    dash._cache.clear()
    B._ledger_cache_clear()


@pytest.fixture(autouse=True)
def _cold():
    """Every timing test starts from an empty cache, in both layers."""
    empty_both_caches()
    yield
    empty_both_caches()


async def _cold_cycle() -> dict:
    """One cold call through the real route handler, decoded."""
    response = await dash.get_dashboard()
    return json.loads(response.body.decode("utf-8"))


async def enforce_cold_cycle_budget(*, board_files: int, run_cycle, grade,
                                    budget: float | None = None) -> float:
    """Time cold cycles until one is inside the budget — at most two.

    `run_cycle()` performs one timed cold cycle and returns
    `(seconds, payload)`; `grade(payload)` returns the reason that cycle measured
    nothing, or `None` for a real one. Both caches are emptied before the second
    attempt, so the confirming cycle is as cold as the first.

    One breach is not a verdict. Under `xdist -n 8` on a loaded box a single
    sample carries the scheduler's variance, which is how this node became the
    gate's commonest parallel-only flake: a cycle just under the budget at rest
    went over it under eight workers, with no code change behind the red. So the
    first cycle that breaches is followed by exactly one more, cold, and the node
    reddens only when both breach. Two is also the ceiling — a third cycle would
    be a way of grinding to a green, so `run_cycle` is called at most twice and
    `test_a_breaching_cycle_is_confirmed_by_exactly_one_more_cold_cycle` counts
    the calls to prove it.

    A cycle that `grade` refuses is not a breach and gets no retry: a fast error
    is not a slow dashboard, and re-measuring it would let a broken reader pass
    on the luck of its second attempt.
    """
    owed = cold_budget_for_board(board_files) if budget is None else budget
    first_seconds: float | None = None
    for attempt in (1, 2):
        if attempt == 2:
            empty_both_caches()
        seconds, payload = await run_cycle()
        complaint = grade(payload)
        if complaint:
            raise AssertionError(
                f"attempt {attempt} finished in {seconds:.2f}s and so is inside "
                f"the {owed:.2f}s budget, but it measured nothing: {complaint}. A "
                f"cycle that did not walk the board cannot retire a breach or "
                f"prove a speed-up, so this is a failure rather than a retry."
            )
        print(f"cold cycle{'' if attempt == 1 else ' (confirming)'}: "
              f"{seconds:.2f}s over {board_files} files; budget {owed:.2f}s")
        if seconds < owed:
            return seconds
        if attempt == 1:
            first_seconds = seconds
    raise AssertionError(
        f"two consecutive cold /api/dashboard cycles over {board_files} board "
        f"files both breached the {owed:.2f}s budget: {first_seconds:.2f}s then "
        f"{seconds:.2f}s. The second cycle ran with both caches emptied, so this "
        f"is the cold path and not a load spike. The calibration is "
        f"{COLD_BUDGET_S:.2f}s over {CALIBRATED_BOARD_FILES} item files and "
        f"{CALIBRATED_LEDGER_BYTES:,} ledger bytes; the triage baseline was "
        f"15.22 s at 1,142 files. The remaining cost is named on #1204: "
        f"re-walking the board per section (recommendation B) and re-decoding the "
        f"ledger (tests/test_backlog_ledger_cache.py)."
    )


def _grade_a_cold_cycle(payload: dict) -> str | None:
    """Why this payload is not a measurement, or None if it is one.

    The positive control from #1204, kept exactly where it was: the two sections
    that cost the time must have run. A `_gather` that swallows an exception
    returns `{"error": ...}` in milliseconds and reads as a speed-up, and a board
    scan that parses nothing returns `total: 0` after the full walk. Either one
    is a fast *nothing*, and a fast nothing must never retire a budget.
    """
    backlog = payload.get("backlog") or {}
    automod = payload.get("automod") or {}
    if not isinstance(backlog, dict) or "error" in backlog:
        return (f"the backlog section errored out ({backlog}), so the timed cycle "
                f"never walked the board — a fast error is not a fast dashboard")
    if not isinstance(automod, dict) or "error" in automod:
        return (f"the automod section errored out ({automod}), which is the "
                f"section that cost 13.07 s cold at triage; without it the "
                f"measurement is of a different request")
    if not backlog.get("total", 0) > 0:
        return (f"backlog.total reads {backlog.get('total')} — the scan parsed "
                f"nothing, so the budget would be met by a broken reader")
    return None


async def test_one_cold_dashboard_cycle_beats_the_budget(monkeypatch):
    # No conditional-skip marker on the board directory, and none anywhere on
    # this path: with one, a vault root that exists and a board that has been
    # emptied or moved reads as not-run. `board_files_or_stop` fails on every
    # state that leaves it nothing to walk — moved board, emptied board, no
    # vault. See tests/board_presence.py.
    files = board_files_or_stop(what="cold-cycle budget", numeric_names=True)
    ledger = timed_ledger_or_stop(monkeypatch, what="cold-cycle budget")
    ledger_bytes = ledger.path.stat().st_size
    print(f"board: {len(files)} item files; ledger {ledger.path}: "
          f"{ledger_bytes:,} bytes | {ledger.why}")
    assert ledger_bytes > 0, (
        f"{ledger.path} is empty, so this cycle decoded no ledger at all and the "
        f"budget does not constrain the ledger re-decode path"
    )
    budget = cold_budget_for_board(len(files))
    # What the machine was doing while this was timed. The table beside
    # `COLD_BUDGET_S` cannot carry it, and a red that prints only an elapsed against
    # a budget gets read as a regression by a reader with no way to tell a slow
    # dashboard from a busy box. The load average and the xdist worker id together
    # say which one it was. The example breach below is interpolated from `budget`
    # rather than written out: this file re-based `COLD_BUDGET_S` 7.64 -> 7.79, which
    # moved the budget at calibrated board size from 9.55s to 9.74s, and a hard-coded
    # illustration would have kept quoting the figure the same diff retired.
    worker = os.environ.get("PYTEST_XDIST_WORKER", "master")
    print(f"budget for this cycle: {budget:.2f}s = the calibrated "
          f"{COLD_BUDGET_S:.2f}s spread over {CALIBRATED_BOARD_FILES} files and "
          f"re-charged on {len(files)}")
    print(f"machine at measurement: xdist worker {worker}, "
          f"load 1/5/15 = {os.getloadavg()} ")

    async def one_cold_cycle():
        started = time.perf_counter()
        payload = await _cold_cycle()
        return time.perf_counter() - started, payload

    elapsed = await enforce_cold_cycle_budget(
        board_files=len(files),
        budget=budget,
        run_cycle=one_cold_cycle,
        grade=_grade_a_cold_cycle,
    )
    assert elapsed < budget, (
        f"the helper returned {elapsed:.2f}s, outside the {budget:.2f}s it was "
        f"given — it retired a breach it did not measure away"
    )


def test_the_board_is_still_the_size_the_budget_was_calibrated_on():
    """Redden when the corpus the budget measured has drifted out of its band.

    This is the node that turns the old flake into a maintenance signal. The
    budget was 8.0 s across two calibrations while the board grew past it, and
    the only thing that noticed was a parallel-only red on whatever round happened
    to be gated at the time. Now the growth itself reddens, once, with an
    instruction to re-measure.
    """
    files = board_files_or_stop(what="budget calibration drift", numeric_names=True)
    live = len(files)
    assert_board_within_calibration_drift(live)
    print(f"board drift: {live} item files live vs {CALIBRATED_BOARD_FILES} "
          f"calibrated — inside the +/-{BOARD_DRIFT_FRACTION:.0%} band")


def test_the_ledger_is_still_the_size_the_budget_was_calibrated_on():
    """Redden when the ledger this cycle decodes has drifted out of its band.

    The board's twin, and the reason it exists is the sentence in the table beside
    `COLD_P90_S`: a cycle decodes two things, the board and the ledger, and until
    #1975 only the first had a bound and only the first was checked. Ledger growth
    was therefore the silent way to break this budget — no code change, no test
    red, just a slower cycle until the timing node flaked and the flake got blamed
    on the machine.
    """
    size, why = live_ledger_bytes()
    assert_ledger_within_calibration_drift(size)
    print(f"ledger drift: {size:,} bytes live vs {CALIBRATED_LEDGER_BYTES:,} "
          f"calibrated — inside the +/-{BOARD_DRIFT_FRACTION:.0%} band; {why}")


@pytest.mark.parametrize("live_ledger_bytes", [
    pytest.param(0, id="a-ledger-with-nothing-to-decode"),
    pytest.param(1024, id="a-ledger-truncated-to-noise"),
])
def test_a_ledger_far_smaller_than_the_calibration_reddens_the_drift_check(
        live_ledger_bytes):
    """A ledger much smaller than the calibration is as uncalibrated as a grown one.

    Two-sided for the board's reason, and one more reason that is this store's
    own: a truncate is exactly what a badly-written sixth rung would do to this
    file, and a budget that reads a shrunken decode as a win would call that an
    improvement. It is data loss; the node should say so.
    """
    with pytest.raises(AssertionError) as caught:
        assert_ledger_within_calibration_drift(live_ledger_bytes)
    message = str(caught.value)
    assert f"{CALIBRATED_LEDGER_BYTES:,}" in message, message
    assert f"{live_ledger_bytes:,}" in message, message
    assert "FOLD THE STORE" in message and "do not raise the budget" in message, message


def test_a_ledger_that_outgrew_the_calibration_reddens_with_the_rung_named():
    """Past the band upward, the message points at the store's owner before the knob.

    The 7-day mean growth at calibration was about 2 MB/day, so this store leaves
    its band in days, not quarters — a month of unminded growth is enough, which
    is exactly how #1858 found the live file 4.5x its own docstring baseline with
    nothing red anywhere. The message has to name the rung that bounds the file
    first: a reader who reaches for `COLD_BUDGET_MARGIN` here has accepted the
    slower cycle and left the store growing.
    """
    outside = int(CALIBRATED_LEDGER_BYTES * 1.5)
    with pytest.raises(AssertionError) as caught:
        assert_ledger_within_calibration_drift(outside)
    message = str(caught.value)
    assert "retention-sweep.py" in message, message
    assert "promotions ledger:" in message, message
    assert "LEDGER_ARCHIVE_AGE_DAYS" in message, message
    assert "COLD_BUDGET_MARGIN" in message, message
    assert message.index("retention-sweep.py") < message.index("COLD_BUDGET_MARGIN"), (
        "the instruction to bound the store must come before the warning about "
        "the knob that hides the drift")


@pytest.mark.parametrize("live_board_files", [
    pytest.param(0, id="an-empty-board"),
    pytest.param(1, id="a-one-file-board"),
])
def test_a_board_far_smaller_than_the_calibration_reddens_the_drift_check(
        live_board_files):
    """A shrunken board is as uncalibrated as a grown one.

    Drift is two-sided on purpose: a cycle over almost nothing also costs almost
    nothing, and a budget calibrated on the real board would be met by a corpus
    that no longer exists. `board_files_or_stop` stops on a genuinely missing
    board; this catches a board that is present and no longer the one measured.
    """
    with pytest.raises(AssertionError) as caught:
        assert_board_within_calibration_drift(live_board_files)
    message = str(caught.value)
    assert str(CALIBRATED_BOARD_FILES) in message, message
    assert str(live_board_files) in message, message
    assert "RE-BASE" in message and "do not raise the budget" in message, message


def test_a_board_that_outgrew_the_calibration_reddens_the_drift_check():
    """The failure this whole clause exists for, exercised without waiting for it.

    Driven one file past the band rather than at some dramatic multiple, because
    the band's edge is the behaviour under test: 20% of growth is absorbed by the
    per-file charge, and the file past it is a calibration event.
    """
    outside = int(CALIBRATED_BOARD_FILES * (1.0 + BOARD_DRIFT_FRACTION)) + 1
    inside = int(CALIBRATED_BOARD_FILES * (1.0 + BOARD_DRIFT_FRACTION))
    assert_board_within_calibration_drift(inside)   # the band itself still passes
    with pytest.raises(AssertionError) as caught:
        assert_board_within_calibration_drift(outside)
    message = str(caught.value)
    assert f"{CALIBRATED_BOARD_FILES} item files" in message, message
    assert f"{outside}" in message, (
        f"the message must print the count it is reddening on: {message}")
    assert "RE-BASE" in message and "do not raise the budget" in message, message
    assert "COLD_BUDGET_MARGIN" in message, (
        f"the message has to name the dial the reader must NOT turn: {message}")


def test_the_budget_is_the_measured_p90_times_the_stated_margin():
    """Clause 1's arithmetic, pinned: no hand-tuned number sits between them."""
    assert COLD_BUDGET_MARGIN > 1.0, COLD_BUDGET_MARGIN
    assert COLD_BUDGET_S == pytest.approx(COLD_P90_S * COLD_BUDGET_MARGIN), (
        f"{COLD_BUDGET_S} is not {COLD_P90_S} x {COLD_BUDGET_MARGIN}: the budget "
        f"is a typed number again, not a re-based one")
    assert COLD_BUDGET_S >= COLD_P90_S, (
        f"the budget {COLD_BUDGET_S} is below the p90 it was derived from "
        f"{COLD_P90_S}")


@pytest.mark.asyncio
async def test_the_memo_wraps_the_real_state_request_and_counts_it(monkeypatch):
    """The memo is tested across the process boundary it actually sits on.

    The saving this diff made lives on `_agent_state`, an httpx GET of the MCP
    aggregator's `/state` — the boundary the cold cycle crossed twice and the reason
    the dashboard sat on "Loading…" for a minute. Round `SM_20261106_233502` proved it
    in `test_a_warm_cycle_pays_the_aggregator_round_trip_once_per_ttl`, which replaces
    `_agent_state` with a counting stub and drives `get_dashboard()`. That is a real
    test of the memo, but with `_agent_state` itself stubbed it cannot tell you the
    memo wraps THIS call and not some other one: a memo pasted onto `_primary_state`,
    or a route that fetched the registry beside the memo, counts the same there. The
    review rung named that gap on the diff that introduced the memo, so the diff now
    crosses the boundary instead of describing it.

    This node runs the product's own coroutine over the product's own
    `httpx.AsyncClient`, with only the transport replaced — `httpx.MockTransport`, the
    idiom `tests/test_mcp_transport.py` already uses against this same URL. The
    response is parsed by the product's `resp.json()`, the URL and the timeout are the
    product's, and the count is taken on the request as it leaves. A memo pinned onto
    any other call site changes the count and this node says so. The credential leg is
    pinned in `tests/test_mcp_transport.py` against a live loopback server; this node
    makes no claim about it.
    """
    import httpx

    from app.routers import dashboard as D

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"subagents": {}, "tasks": {}})

    real_client = httpx.AsyncClient

    def client_with_test_transport(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client_with_test_transport)
    monkeypatch.setattr(D, "_MCP_STATE_URL", "http://127.0.0.1:1/state")
    monkeypatch.setattr(D, "_cache", {})

    first = await D._agent_states_cached()
    second = await D._agent_states_cached()
    assert first == second == {"subagents": {}, "tasks": {}}, (
        "the memo must hand back what the request returned, parsed by the product")
    assert len(seen) == 1, (
        f"two calls inside the TTL must be one request; the cycle paid {len(seen)}. "
        f"This is the duplicate the cold cycle was, and the memo is only worth what "
        f"it saves if it lands on this call site")
    assert seen[0].url.path == "/state", (
        f"the memo must sit on the STATE endpoint, not some other call: {seen[0].url}")

    stamp, value = D._cache["agents"]
    D._cache["agents"] = (stamp - D._AGENT_STATE_TTL_S - 1, value)
    await D._agent_states_cached()
    assert len(seen) == 2, (
        "an entry older than _AGENT_STATE_TTL_S must go out again, or the memo is not "
        "a TTL and the dashboard stops showing agents that just appeared")


@pytest.mark.asyncio
async def test_a_warm_cycle_pays_the_aggregator_round_trip_once_per_ttl(monkeypatch):
    """The section the warm ratio measures is memoised, and the TTL still binds.

    `test_the_warm_cycle_is_an_order_of_magnitude_cheaper_than_cold` states a
    ratio and names the failure it exists to catch — "a section recomputing on
    every poll regardless of TTL". The section that now does that is `agents`:
    one HTTP GET of the aggregator's `/state` per cycle, measured here at
    0.59-1.39 s per call while every other uncached section sat at or under
    0.163 s, which is what took the ratio from 23x to 8.7x under `-n 8`. Its
    ratio node passes or fails on that one call, so the call itself is pinned
    here, counted rather than timed: two cycles with everything else already
    warm fetch it ONCE, and pushing the memo's stamp one TTL into the past makes
    the next cycle fetch again — the TTL is a bound on staleness, not a way to
    stop asking.

    The rest of the cycle is stubbed to `{}` for this node, the way
    `test_dashboard_engine_metrics._build_client` stubs it, so the count is
    about `agents` alone and the node costs milliseconds instead of the ~16 s a
    real cold cycle over the live board takes.
    """
    for name in ("_network", "_recent_sessions", "_services", "_workers",
                 "_autonomy", "_backlog", "_automod", "_usage"):
        monkeypatch.setattr(dash, name, lambda: {})
    monkeypatch.setattr(dash.host_metrics, "collect", _async_empty)
    monkeypatch.setattr(dash, "_vllm", _async_empty)
    monkeypatch.setattr(dash, "_primary_state", _async_empty)

    fetches = []

    async def _counted_state():
        fetches.append(1)
        return {"subagents": {}, "tasks": {}}

    monkeypatch.setattr(dash, "_agent_state", _counted_state)
    dash._cache.clear()
    try:
        await dash.get_dashboard()
        assert len(fetches) == 1, (
            "the first cycle must fetch the registry once; it fetched "
            f"{len(fetches)}")
        await dash.get_dashboard()
        assert len(fetches) == 1, (
            "a warm cycle re-fetched the aggregator registry — that round trip "
            "is 0.59-1.39 s PER CALL measured here against the 0.019 s a "
            "memo-hit cycle takes, and re-fetching it every poll is what "
            f"puts the ratio under 10x; calls now: {len(fetches)}")
        stamp, value = dash._cache["agents"]
        assert value == {"subagents": {}, "tasks": {}}
        assert 0.0 <= time.monotonic() - stamp < dash._AGENT_STATE_TTL_S, (
            f"the memo is stamped {time.monotonic() - stamp:.3f}s ago; the entry "
            "has to age from when the response arrived, the way `_cached` ages "
            "a value that spent its own fetch time being produced")
        dash._cache["agents"] = (stamp - dash._AGENT_STATE_TTL_S, value)
        await dash.get_dashboard()
        assert len(fetches) == 2, (
            f"a memo aged past `_AGENT_STATE_TTL_S` ({dash._AGENT_STATE_TTL_S} s) "
            f"was still served: fetches {len(fetches)}, so the panel would show "
            "a registry nothing refreshed")
    finally:
        dash._cache.pop("agents", None)


async def _async_empty():
    return {}


def test_the_calibration_table_and_the_cycles_it_records_agree():
    """The `COLD_P90_S` table and `_CALIBRATION_CYCLES_S` are one measurement.

    Three surfaces used to hold this calibration: the measured durations, the prose
    table quoting them, and the constant the budget is charged from — and the only
    rail on that constant read `COLD_BUDGET_S == COLD_P90_S * COLD_BUDGET_MARGIN`,
    which is the budget's own definition, so it could not fail and could not notice a
    p90 typed with the margin already folded into it. The cycles are in the file as
    data now: each table row is re-derived here from the set it names, and the
    constant has to equal the largest p90 this file's own table quotes. A re-baser who
    edits one of the three surfaces and not the others reddens this node instead of
    shipping a budget nothing measures.

    Figures are compared with a 0.011 s tolerance because the table quotes them to
    two decimals and an even-count median of two-decimal samples sits on a rounding
    boundary: the 6-cycle set's median is 6.665 s, which prints as 6.66 or 6.67
    depending on the float, and the row quotes 6.67.
    """
    rows = re.findall(
        r"^#: {5}serial, (\d+) runs, load (\d+\.\d+-\d+\.\d+):"
        r" +min (\d+\.\d+) +median (\d+\.\d+) +p90 (\d+\.\d+) +max (\d+\.\d+)$",
        Path(__file__).read_text(encoding="utf-8"), re.MULTILINE)
    assert len(rows) == len(_CALIBRATION_CYCLES_S), (
        f"the table quotes {len(rows)} serial rows and `_CALIBRATION_CYCLES_S` holds "
        f"{len(_CALIBRATION_CYCLES_S)} sets of cycles: the two halves are compared "
        f"row against set, so they have to be the same size to mean anything")

    quoted_p90 = []
    for (n, load, q_min, q_med, q_p90, q_max), (row_load, cycles) in zip(
            rows, _CALIBRATION_CYCLES_S):
        where = f"the load {row_load} row"
        assert int(n) == len(cycles), (
            f"{where}: it says {n} runs and the recorded set is {len(cycles)} "
            f"durations long")
        assert load == row_load, (
            f"{where}: its load range is {load!r} and the recorded one is "
            f"{row_load!r}")
        for label, have, quoted in (
                ("min", min(cycles), float(q_min)),
                ("median", _median_s(cycles), float(q_med)),
                ("p90", _p90_nearest_rank(cycles), float(q_p90)),
                ("max", max(cycles), float(q_max))):
            assert abs(have - quoted) <= 0.011, (
                f"{where} ({len(cycles)} cycles): its {label} is {quoted} and those "
                f"cycles give {have:.3f}")
        quoted_p90.append(float(q_p90))

    assert COLD_P90_S == pytest.approx(max(quoted_p90), abs=0.011), (
        f"COLD_P90_S is {COLD_P90_S} and the largest p90 this file's table quotes is "
        f"{max(quoted_p90)}: the constant is the larger of the two measured tails, so "
        f"one of the two was changed without the other")


def _comment_prose() -> str:
    """This file's comment prose as one searchable string, markers and wraps folded.

    Every figure the two nodes below pin lives in a `#:` block that the file's own
    line-wrapping breaks across three or four lines, so a sentence a human wrote as
    one thought is not one string on disk. Stripping the `#:` markers and collapsing
    whitespace first is what lets a regex ask about the sentence rather than about the
    editor's line length.

    The result covers docstrings and assert messages too, which is deliberate: a
    figure quoted in a docstring is a claim on the same surface as one in a comment,
    and the `== 1` arms below are loud about a second copy rather than silently
    matching whichever one they find first.
    """
    raw = Path(__file__).read_text(encoding="utf-8")
    return re.sub(r"\s+", " ", re.sub(r"^[ \t]*#:[ \t]?", " ", raw, flags=re.MULTILINE))


def test_the_enforced_bound_is_pinned_to_the_ceiling_it_outran():
    """The bound's relation to the triage cost is arithmetic here, not prose.

    The review rung on round `SM_20261106_233502` refused that diff for a sentence
    that had outlived its number: the re-base raised the enforced bound from 13.16 s
    to 20.86 s, and the file went on claiming the bound stayed short of the triage
    cost it exists to catch, quoting the retired figure. Nothing could notice, because
    no assertion named the ceiling at all. So the ceiling is a constant now
    (`TRIAGE_COLD_S`), and this node computes the relation and compares it with the
    one sentence that states it.

    Two directions make it fail, which is the whole point. Quote a number, a side, or
    a percentage the constants do not give and it reddens — that catches the stale
    sentence. And leave the sentence alone while a re-base moves the budget back under
    the ceiling, so the disclosed breach stops being true, and it reddens too: a
    disclosure is a claim with an expiry date, and this file's last four calibrations
    each rewrote the number without rewriting the sentence beside it.

    The second half is the part that still holds and had no rail either. Absolute
    seconds cannot survive a corpus that triples; milliseconds per item file can, so
    the surviving invariant — the enforced allowance must stay under the per-file rate
    triage recorded — is asserted, and it is the terminal stop on the re-base ladder:
    a re-baser cannot keep absorbing a dearer walk past it.
    """
    prose = _comment_prose()
    bound = cold_budget_for_board(CALIBRATED_BOARD_FILES)
    assert bound == pytest.approx(COLD_BUDGET_S), (
        "a cycle over the calibrated board is charged the calibrated budget; the "
        "ceiling comparison is meaningless if the two disagree")
    pct = (bound - TRIAGE_COLD_S) / TRIAGE_COLD_S * 100.0

    hits = re.findall(
        r"[Tt]he enforced bound is (\d+\.\d+) s, (\d+\.\d)% (ABOVE|SHORT OF) the "
        r"([\d.]+) s triage cost", prose)
    assert len(hits) == 1, (
        f"exactly one sentence states where the enforced bound stands against the "
        f"triage cost, and this file states it {len(hits)} times: the node cannot "
        f"arbitrate between two disclosures")
    q_bound, q_pct, q_side, q_ceiling = hits[0]
    assert float(q_ceiling) == TRIAGE_COLD_S, (
        f"the sentence names a triage cost of {q_ceiling} s and the constant is "
        f"{TRIAGE_COLD_S}: the ceiling was retyped rather than referenced")
    assert abs(float(q_bound) - bound) <= 0.011, (
        f"the sentence quotes a bound of {q_bound} s; the constants give "
        f"{bound:.4f} s ({COLD_P90_S} x {COLD_BUDGET_MARGIN}). This is the arm that "
        f"caught a retired 13.16 s left standing beside a 20.86 s budget")
    assert abs(float(q_pct) - abs(pct)) <= 0.05, (
        f"the sentence says {q_pct}% and the arithmetic gives {abs(pct):.2f}% of "
        f"{TRIAGE_COLD_S} s")
    assert (q_side == "SHORT OF") == (bound < TRIAGE_COLD_S), (
        f"the sentence reads {q_side} while the bound is {bound:.2f} s against a "
        f"ceiling of {TRIAGE_COLD_S} s: the disclosure outlived the fact, which is "
        f"the exact failure this node was written for")

    bound_ms = ms_per_item_file(bound, CALIBRATED_BOARD_FILES)
    triage_ms = ms_per_item_file(TRIAGE_COLD_S, TRIAGE_BOARD_FILES)
    assert bound_ms < triage_ms, (
        f"the enforced allowance is {bound_ms:.2f} ms per item file and the triage "
        f"run recorded {triage_ms:.2f}: the check can no longer catch a walk as dear "
        f"as the one it was written against, in ANY unit. Absolute seconds already "
        f"went past — this is the last comparison left, and reaching it means #1204's "
        f"cheaper walk (one shared board walk), not a wider band, a thinner margin or "
        f"a re-based p90.")
    rates = re.findall(
        r"per item file the bound is (\d+\.\d+) ms against the ([\d.]+) ms/file "
        r"that triage run recorded — ([\d.]+)x tighter", prose)
    assert len(rates) == 1, f"expected one per-file disclosure, found {len(rates)}"
    q_bound_ms, q_triage_ms, q_headroom = rates[0]
    assert abs(float(q_bound_ms) - bound_ms) <= 0.005, (
        f"the prose says {q_bound_ms} ms/file, the constants give {bound_ms:.4f}")
    assert abs(float(q_triage_ms) - triage_ms) <= 0.005, (
        f"the prose says {q_triage_ms} ms/file at triage, the constants give "
        f"{triage_ms:.4f} from {TRIAGE_COLD_S} s over {TRIAGE_BOARD_FILES} files")
    assert abs(float(q_headroom) - triage_ms / bound_ms) <= 0.05, (
        f"the prose claims {q_headroom}x of headroom; the two rates give "
        f"{triage_ms / bound_ms:.2f}x")

    restore_p90 = TRIAGE_COLD_S / COLD_BUDGET_MARGIN
    back = re.search(
        r"the p90 that would put the bound under the triage cost at this margin is "
        r"(\d+\.\d+) s, (\d+)% BELOW the tail measured on this board", prose)
    assert back, (
        "the file must state what would actually put this bound back under the "
        "ceiling, because 'lower the margin' is the wrong remedy and the number is "
        "what says so")
    assert abs(float(back.group(1)) - restore_p90) <= 0.005, (
        f"the p90 that clears the ceiling at this margin is {back.group(1)} s; the "
        f"arithmetic is {TRIAGE_COLD_S} / {COLD_BUDGET_MARGIN} = {restore_p90:.4f}")
    below = (1 - restore_p90 / COLD_P90_S) * 100.0
    assert abs(float(back.group(2)) - below) <= 0.5, (
        f"the prose says the required p90 is {back.group(2)}% below the measured tail "
        f"of {COLD_P90_S} s; it is {below:.1f}% below it")

    retired = re.search(
        r"the retired p90 put the bound at (\d+\.\d+) s, ([\d.]+)% under the "
        r"triage cost", prose)
    assert retired, "the history of the ceiling is stated once, in the budget comment"
    assert abs(float(retired.group(1)) - RETIRED_P90_S * COLD_BUDGET_MARGIN) <= 0.011, (
        f"the retired bound is quoted as {retired.group(1)} s against "
        f"{RETIRED_P90_S} x {COLD_BUDGET_MARGIN} = "
        f"{RETIRED_P90_S * COLD_BUDGET_MARGIN:.4f} s")
    retired_short = (
        (TRIAGE_COLD_S - RETIRED_P90_S * COLD_BUDGET_MARGIN) / TRIAGE_COLD_S * 100)
    assert abs(float(retired.group(2)) - retired_short) <= 0.05, (
        f"the retired shortfall is quoted as {retired.group(2)}%; on this file's "
        f"convention (relative to the triage cost) it is {retired_short:.2f}%")


def test_the_re_base_prose_recomputes_from_the_two_calibrations():
    """Every movement the re-base record quotes is re-derived from both calibrations.

    `test_the_calibration_table_and_the_cycles_it_records_agree` re-derives the
    min/median/p90/max of each row, and the review rung pointed out that it stops
    there: the comparative figures — how far the tail moved, how far the board moved,
    how much dearer the walk went per file, how many files past the retired band the
    new count sat — are the sentences a reader actually acts on, and none of them was
    checked. One of them ("52.2%") was the previous re-base's pair pasted into this
    record, and another ("one file past") understated the drift node's own arithmetic
    by nine. Both shipped green.

    So the pair the record compares itself against is data (`RETIRED_BOARD_FILES`,
    `RETIRED_P90_S`), and each figure below is recomputed from it and matched against
    the sentence. A re-baser who replaces the cycles and the count but leaves the
    sentences now gets a red node naming each figure that no longer follows.
    """
    prose = _comment_prose()

    def one(pattern: str) -> tuple:
        hits = re.findall(pattern, prose)
        assert len(hits) == 1, (
            f"pattern {pattern!r} matched {len(hits)} passages; each pinned figure is "
            f"stated exactly once, and a second copy is how one goes stale")
        first = hits[0]
        return first if isinstance(first, tuple) else (first,)

    moved = one(r"[Tt]he measured tail moved \+([\d.]+)% while the board moved "
                r"\+([\d.]+)%")
    tail_pct = (COLD_P90_S / RETIRED_P90_S - 1) * 100
    board_pct = (CALIBRATED_BOARD_FILES / RETIRED_BOARD_FILES - 1) * 100
    assert round(tail_pct, 1) == float(moved[0]), (
        f"the tail moved from {RETIRED_P90_S} s to {COLD_P90_S} s, which is "
        f"{tail_pct:.2f}%, and the record says {moved[0]}% — the figure that shipped "
        f"first here was the PREVIOUS re-base's pair (6.92 to 10.53), not this one's")
    assert round(board_pct, 1) == float(moved[1]), (
        f"the board moved from {RETIRED_BOARD_FILES} to {CALIBRATED_BOARD_FILES} item "
        f"files, which is {board_pct:.2f}%, and the record says {moved[1]}%")

    dearer = one(r"from (\d+\.\d+) ms to (\d+\.\d+) ms per item file — "
                 r"([\d.]+)% dearer per file")
    ms_then = ms_per_item_file(RETIRED_P90_S, RETIRED_BOARD_FILES)
    ms_now = ms_per_item_file(COLD_P90_S, CALIBRATED_BOARD_FILES)
    assert abs(float(dearer[0]) - ms_then) <= 0.005, (
        f"the retired walk is quoted at {dearer[0]} ms/file and {RETIRED_P90_S} s "
        f"over {RETIRED_BOARD_FILES} files is {ms_then:.4f}")
    assert abs(float(dearer[1]) - ms_now) <= 0.005, (
        f"this walk is quoted at {dearer[1]} ms/file and {COLD_P90_S} s over "
        f"{CALIBRATED_BOARD_FILES} files is {ms_now:.4f}")
    assert round((ms_now / ms_then - 1) * 100, 1) == float(dearer[2]), (
        f"the per-file movement is quoted as {dearer[2]}% and the two rates give "
        f"{(ms_now / ms_then - 1) * 100:.2f}%")

    smaller = re.findall(r"described a board ([\d.]+)% smaller", prose)
    assert smaller, "the record says how much smaller the retired count was"
    smaller_pct = round((1 - RETIRED_BOARD_FILES / CALIBRATED_BOARD_FILES) * 100, 1)
    assert all(smaller_pct == float(s) for s in smaller), (
        f"'described a board X% smaller' reads {smaller}, but the two counts give "
        f"{smaller_pct}%")

    past = re.findall(r"(\d+) files past", prose)
    edge = int(RETIRED_BOARD_FILES * (1.0 + BOARD_DRIFT_FRACTION))
    over = CALIBRATED_BOARD_FILES - edge
    assert past, "the record says how far past the retired band's edge this count sat"
    assert f"{edge:,}" in prose, (
        f"the record should name the retired edge it counted from ({edge:,}), so a "
        f"reader can check the subtraction instead of trusting it")
    assert all(int(p) == over for p in past), (
        f"the record says {past} files past a retired edge of {edge}, and the drift "
        f"node's own arithmetic (`int({RETIRED_BOARD_FILES} * "
        f"(1 + {BOARD_DRIFT_FRACTION}))` = {edge}) puts this count {over} past it. "
        f"The first version of that sentence understated its own node by nine, and "
        f"the edge is derived from a constant rather than remembered.")

    apart = one(r"sits ([\d.]+)% below the louder at the tail and ([\d.]+)% below "
                r"it at the middle")
    (loud_load, loud), (quiet_load, quiet) = _CALIBRATION_CYCLES_S
    tail_gap = (1 - _p90_nearest_rank(quiet) / _p90_nearest_rank(loud)) * 100
    mid_gap = (1 - _median_s(quiet) / _median_s(loud)) * 100
    assert round(tail_gap, 1) == float(apart[0]), (
        f"the {quiet_load} set's tail is {_p90_nearest_rank(quiet)} s against the "
        f"{loud_load} set's {_p90_nearest_rank(loud)} s: {tail_gap:.2f}% below, and "
        f"the record says {apart[0]}%")
    assert round(mid_gap, 1) == float(apart[1]), (
        f"the medians are {_median_s(quiet)} and {_median_s(loud)} s: "
        f"{mid_gap:.2f}% apart, and the record says {apart[1]}%")


def test_the_per_file_charge_reproduces_the_calibration_and_scales_from_it():
    """Clause 2: the budget's margin reads the calibrated count, one copy."""
    assert cold_budget_for_board(CALIBRATED_BOARD_FILES) == pytest.approx(
        COLD_BUDGET_S), "the per-file charge must be exact on its own calibration"
    assert cold_budget_for_board(CALIBRATED_BOARD_FILES * 2) == pytest.approx(
        COLD_BUDGET_S * 2), "a board twice the size is charged twice the budget"
    assert cold_budget_for_board(1) == COLD_BUDGET_S, (
        "a smaller board does not buy a tighter bound: the floor is the "
        "calibrated number, and a shrunken corpus is the drift test's business")


def test_the_calibrated_board_count_is_written_once_in_this_file():
    """No second copy of the calibrated count, in code or in prose.

    Reading the file's own text is how this can be checked over everything the
    next editor might write, including the comments that go stale quietly. The
    historical counts (1,142 and 1,450) are not this count and are allowed to
    stay where the story needs them; what may not exist is a second copy of the
    number the budget and the drift band are both derived from.
    """
    text = _file_text_without_this_node()
    for name, value in (("CALIBRATED_BOARD_FILES", CALIBRATED_BOARD_FILES),
                        ("CALIBRATED_LEDGER_BYTES", CALIBRATED_LEDGER_BYTES)):
        digits = str(value)
        spaced = f"{value:,}"
        # What may not exist is the number written anywhere but its own
        # assignment. An underscored literal (`28_960_660`) is caught because the
        # digits are compared with underscores stripped, and a thousands-separated
        # one in prose is caught through `spaced`. Historical counts (1,142 and
        # 1,450 files) are different numbers and stay where the story needs them.
        definition = f"{name} = "
        hits = [i for i, line in enumerate(text.splitlines(), 1)
                if (digits in line.replace("_", "") or spaced in line)
                and definition not in line]
        assert not hits, (
            f"{value} is written on {len(hits)} lines beyond its own assignment "
            f"({hits}); every one of those is a number nothing re-computes, and "
            f"`{name}`'s band, its budget-table row and its drift message are all "
            f"derived from the single definition")


def _file_text_without_this_node() -> str:
    """This file's text, minus the body of the written-once node itself.

    A node that greps its own file for a number necessarily matches the literals
    it types to do the grep, and counting those as drift makes it redder the more
    carefully it is written — which is how the first version of this check, with
    one constant, came back reporting the number on its own assertion lines. The
    node's own body is therefore cut before searching: what remains is the
    calibration's prose and every other line in the file, which is the surface
    that actually goes stale quietly.
    """
    lines = Path(__file__).read_text(encoding="utf-8").splitlines()
    marker = "def test_the_calibrated_board_count_is_written_once"
    out, skipping = [], False
    for line in lines:
        if line.startswith(marker):
            skipping = True
        elif skipping and (line.startswith("def ") or line.startswith("@")
                           or line.startswith("class ") or line.startswith("async def ")):
            skipping = False
        if not skipping:
            out.append(line)
    assert not any(line.startswith(marker) for line in out), (
        "the written-once node's own body was not excluded from its own search, "
        "which makes it red on its own literals")
    return "\n".join(out)


def _valid_payload() -> dict:
    """A payload the nothing-was-measured guard accepts, for a fake cycle."""
    return {"backlog": {"total": 7}, "automod": {"rounds": 3}}


async def test_a_breaching_cycle_is_confirmed_by_exactly_one_more_cold_cycle():
    """Slow then fast: the second cold sample retires the breach, and it is the
    last one. Clause 4 of #1858, driven by an injected cycle so the policy is
    pinned without a slow machine to demonstrate it on."""
    calls = []

    async def fake_cycle():
        calls.append(len(calls) + 1)
        durations = [COLD_BUDGET_S + 1.0, 0.5]
        return durations[len(calls) - 1], _valid_payload()

    elapsed = await enforce_cold_cycle_budget(
        board_files=CALIBRATED_BOARD_FILES, run_cycle=fake_cycle,
        grade=_grade_a_cold_cycle)
    assert elapsed == 0.5, elapsed
    assert len(calls) == 2, (
        f"a breach must be confirmed by exactly one more cycle; the injected "
        f"cycle ran {len(calls)} times")


async def test_two_breaching_cycles_fail_and_name_both_durations_and_the_board():
    """Slow then slow: a second breach is the verdict, and it is the last sample.

    The message is part of the clause. A reader of a red gate should not have to
    re-run anything to know what it cost and over what corpus.
    """
    calls = []

    async def fake_cycle():
        calls.append(len(calls) + 1)
        return COLD_BUDGET_S + [1.0, 2.5][len(calls) - 1], _valid_payload()

    with pytest.raises(AssertionError) as caught:
        await enforce_cold_cycle_budget(
            board_files=CALIBRATED_BOARD_FILES, run_cycle=fake_cycle,
            grade=_grade_a_cold_cycle)
    assert len(calls) == 2, (
        f"never a third cycle: the helper ran the injected cycle {len(calls)} "
        f"times, and a third sample would be a way of grinding to a green")
    message = str(caught.value)
    for expected in (f"{COLD_BUDGET_S + 1.0:.2f}s", f"{COLD_BUDGET_S + 2.5:.2f}s",
                     f"{CALIBRATED_BOARD_FILES} board files"):
        assert expected in message, (
            f"{expected!r} missing from the failure message: {message}")


async def test_a_cycle_inside_the_budget_is_measured_once():
    """No retry cost on the common path: the second cycle exists only for breaches."""
    calls = []

    async def fake_cycle():
        calls.append(1)
        return 0.5, _valid_payload()

    elapsed = await enforce_cold_cycle_budget(
        board_files=CALIBRATED_BOARD_FILES, run_cycle=fake_cycle,
        grade=_grade_a_cold_cycle)
    assert elapsed == 0.5 and len(calls) == 1, (elapsed, len(calls))


async def test_a_fast_error_fails_at_once_instead_of_buying_a_second_sample():
    """The nothing-was-measured guards stand in front of the retry.

    Clause 5's other half: a payload whose heavy section errored is not a breach
    waiting to be re-measured, it is a cycle that measured nothing. Retrying it
    would let a broken reader pass on the luck of its second attempt.
    """
    calls = []

    async def fake_cycle():
        calls.append(1)
        return 0.01, {"backlog": {"error": "boom"}, "automod": {}}

    with pytest.raises(AssertionError) as caught:
        await enforce_cold_cycle_budget(
            board_files=CALIBRATED_BOARD_FILES, run_cycle=fake_cycle,
            grade=_grade_a_cold_cycle)
    assert len(calls) == 1, (
        f"a nothing-was-measured cycle must not earn a retry; got {len(calls)}")
    assert "measured nothing" in str(caught.value), str(caught.value)


async def test_the_confirming_cycle_starts_from_emptied_caches(tmp_path):
    """The second cycle is cold, witnessed on the real cache objects.

    A breach retired by a warm cycle is worse than the flake this replaced: the
    0.06 s warm path clears any budget instantly. So the fake cycle plays the
    part of a real one — it populates both caches as `get_dashboard` and the
    ledger reader do — and what is asserted is what the *next* attempt finds:
    nothing. The positive control is the seeding itself, done in the first
    attempt, so the assertion cannot be satisfied by caches nobody filled.
    """
    from scripts.automod import state as S

    ledger = tmp_path / "promotions.jsonl"
    ledger.write_text(json.dumps({"event": "gate", "round_id": "SM_FAKE"}) + "\n",
                      encoding="utf-8")
    found_empty = []    # what each attempt observes on entry
    left_warm = []      # what each attempt leaves behind, having run

    async def fake_cycle():
        found_empty.append((dict(dash._cache), S.ledger_read_count(ledger)))
        dash._cache["payload"] = ({"backlog": {"total": 1}}, time.monotonic())
        S.ledger_rows(ledger)                     # warm the decoded-ledger cache
        left_warm.append((dict(dash._cache), S.ledger_read_count(ledger)))
        durations = [COLD_BUDGET_S + 1.0, 0.5]
        return durations[len(found_empty) - 1], _valid_payload()

    elapsed = await enforce_cold_cycle_budget(
        board_files=CALIBRATED_BOARD_FILES, run_cycle=fake_cycle,
        grade=_grade_a_cold_cycle)

    assert len(found_empty) == 2, found_empty
    assert elapsed == 0.5, elapsed
    cache_left, reads_left = left_warm[0]
    assert "payload" in cache_left and reads_left == 1, (
        f"the first attempt was supposed to leave both caches warm so there was "
        f"something for the reset to remove; it left {list(cache_left)} and "
        f"{reads_left} ledger read(s)")
    cache_found, reads_found = found_empty[1]
    assert cache_found == {}, (
        f"the confirming cycle found the response cache still populated: "
        f"{list(cache_found)} — a warm cycle cannot retire a breach")
    assert reads_found == 0, (
        f"the confirming cycle found the decoded ledger still cached "
        f"(reads={reads_found} since the last reset), so it would have been timed "
        f"with the ledger already in memory")


def test_a_sandbox_ledger_stub_never_becomes_the_drift_measurement(tmp_path, monkeypatch):
    """A few rows in the configured state dir is not the store this constant bounds.

    The regression this node exists for: the gate's first run of the drift arm
    reddened on a 1 KB file, because the suite's own `timed_ledger_or_stop` prefers
    the configured path whenever it has any bytes and a gate session seeds that
    path with rows appended by other tests. For a timing run a stub is fine — any
    bytes exercise the decode — but a SIZE assertion measured against a stub is a
    verdict about the sandbox, so the drift arm reads the live store first. Here
    the configured path holds a stub and the live path a full store, and the
    measurement must be the full store's.
    """
    from scripts.automod import state as st
    stub = tmp_path / "scratch" / "promotions.jsonl"
    stub.parent.mkdir(parents=True)
    stub.write_bytes(b'{"ts": 1.0, "event": "gate"}\n')
    full = tmp_path / "live-promotions.jsonl"
    full.write_bytes(b'{"ts": 1.0, "event": "gate"}\n' * 5000)
    monkeypatch.setattr(st, "LEDGER_PATH", stub)
    monkeypatch.setattr(board_presence, "live_ledger", lambda: full)

    size, why = live_ledger_bytes()
    assert size == full.stat().st_size, (
        f"the drift arm measured the {stub.stat().st_size}-byte sandbox stub at "
        f"{stub} instead of the {full.stat().st_size}-byte store the calibration "
        f"describes; the band is about the store, not about the run's scratch dir")
    assert "live store" in why, why

    # The fallback needs the configured file to be plausible as a store, and the
    # only test that can say so is one that sets the bar: with the calibration
    # re-based to the fixture's own size, the configured path IS the store and the
    # number comes from it, with `why` naming which file was read.
    monkeypatch.setattr(board_presence, "live_ledger",
                        lambda: tmp_path / "nothing-here.jsonl")
    real_calibration = CALIBRATED_LEDGER_BYTES
    monkeypatch.setattr(sys.modules[__name__], "CALIBRATED_LEDGER_BYTES",
                        stub.stat().st_size)
    size2, why2 = live_ledger_bytes()
    assert size2 == stub.stat().st_size, size2
    assert "configured store" in why2 and "no bytes" in why2, why2

    # And with the calibration back at the real number, that same stub is refused:
    # below the band's floor with no live store beside it, it is a scratch file,
    # and a node that measured it would report a sandbox as a bounded store. The
    # number is carried in a variable, never retyped — the written-once node below
    # reddens on a second copy of it in this file, including one in a test.
    monkeypatch.setattr(sys.modules[__name__], "CALIBRATED_LEDGER_BYTES",
                        real_calibration)
    with pytest.raises(AssertionError) as caught:
        live_ledger_bytes()
    assert "scratch state dir" in str(caught.value), caught.value


def test_no_ledger_at_all_is_red_and_not_a_zero(tmp_path, monkeypatch):
    """Neither path holding a ledger is red, never a 0 that sits inside the band.

    Zero would clear the band's floor and be reported as a drift, so the outcome
    is red either way; what this node pins is WHICH failure it is — the message has
    to name the missing store, because "0 bytes, outside the band" reads as a
    shrunken corpus and sends the reader looking for the rung that truncated a file
    that was never there.
    """
    from scripts.automod import state as st
    monkeypatch.setattr(st, "LEDGER_PATH", tmp_path / "absent" / "promotions.jsonl")
    monkeypatch.setattr(board_presence, "live_ledger",
                        lambda: tmp_path / "absent-live.jsonl")
    with pytest.raises(AssertionError) as caught:
        live_ledger_bytes()
    message = str(caught.value)
    assert "no promotion ledger to measure" in message, message
    assert "outside the" not in message, (
        f"a missing store must be named as missing, not as a byte drift: {message}")


def test_a_repointed_ledger_says_so(tmp_path, monkeypatch):
    """The `repointed` half of `timed_ledger_or_stop`, exercised on every box.

    Added because the round-SM_20260917_064456 advisory ("the ledger test repoints
    `S.LEDGER_PATH` at the live 6.3 MB file, so the test proves only the read
    shape, not the measured cost") is only answerable if the loud path is shown to
    be loud. On a box whose state dir already holds a ledger the helper returns
    `repointed=False` and the reason string is never read.

    Both sides are exercised here rather than hoped for: the substituted ledger is
    a file this test writes, and the configured path is a file that does not
    exist. No `if this box has a ledger` guard — a conditional green exit is as
    dishonest as a skip, and the round's whole block was about tests that could
    report a thing they did not do.
    """
    from scripts.automod import state as S

    configured = tmp_path / "state-dir" / "promotions.jsonl"      # absent
    stand_in = tmp_path / "promotions.jsonl"                      # the "live" one
    stand_in.write_text(json.dumps({"event": "land", "ok": True}) + "\n",
                        encoding="utf-8")
    monkeypatch.setattr(S, "LEDGER_PATH", configured)
    monkeypatch.setattr(board_presence, "live_ledger", lambda: stand_in)

    src = timed_ledger_or_stop(monkeypatch, what="repoint-branch check")
    assert src.repointed, (
        f"returned repointed=False while the configured path ({configured}) does "
        f"not exist and the live one ({stand_in}) does: the helper substituted a "
        f"ledger and reported it as the configured one")
    assert src.path == stand_in, f"expected the stand-in live ledger, got {src.path}"
    assert "REPOINTED" in src.why and "LLOYD_AUTOMOD_STATE" in src.why, (
        f"the reason a caller prints must name the substitution it made: "
        f"{src.why!r}")

    # And the other arm, same box: a configured ledger that exists is used as-is
    # and must NOT be reported as a substitution.
    monkeypatch.setattr(S, "LEDGER_PATH", stand_in)
    same = timed_ledger_or_stop(monkeypatch, what="non-repoint arm")
    assert not same.repointed and same.path == stand_in, (
        f"the helper called a configured ledger a substitution: {same}")
    print(f"repoint arm: {src.why[:100]}… | non-repoint arm: {same.why}")


def test_a_scan_slower_than_its_ttl_is_still_fresh_when_it_lands():
    """A cached value ages from when it exists, not from when its scan started.

    A scan spends time reading the world, and the value it hands back is that
    much old the moment it exists — nothing later makes it younger. So the entry
    is stamped when `fn()` returns. Stamped when the scan *begins*, the entry
    carries its own scan time as age before it has served anyone, and any scan
    slower than its own TTL is born expired: never served, recomputed by every
    reader for as long as the board lasts. That is not hypothetical here — the
    backlog walk of 3,760 item files costs 3.43 s cold measured 2026-11-05 and is
    assembled behind a 12.81 s health half on `_VAULT_SCAN_TTL_S` = 10.0, so its
    entry was 16.7 s old by the time the response went out and the next page load
    walked the board again. The warm cycle below is the stopwatch that measures
    the consequence; this node pins the cause on a key of its own, with no board
    and no stopwatch to be lenient about.

    The two timings are a scan that takes measurable time (0.05 s) against a TTL
    shorter than it (0.01 s), and the claim is a call count, so a loaded machine
    can only widen the gap between them, never close it.
    """
    dash._cache.clear()
    runs: list[int] = []

    def slow_scan() -> str:
        runs.append(len(runs) + 1)
        time.sleep(0.05)
        return "scanned"

    ttl = 0.01
    first = dash._cached("born_expired_probe", ttl, slow_scan)
    second = dash._cached("born_expired_probe", ttl, slow_scan)

    assert first == second == "scanned", (
        f"a cached entry has to return the value its scan produced, got "
        f"{first!r} then {second!r}")
    assert len(runs) == 1, (
        f"the scan ran {len(runs)} times for two calls made back to back, "
        f"though the value was produced less than {ttl:.2f} s ago and the TTL is "
        f"{ttl:.2f} s. An entry stamped at the start of its scan is born "
        "expired — it arrives already older than the window it is sold on, so "
        "every reader recomputes the whole scan. Stamp it when the value exists.")


def test_the_short_ttl_half_of_a_section_is_measured_last(monkeypatch):
    """A section assembled from two cached halves measures the short-TTL one last.

    `_backlog` builds one payload out of a cached pair: the raw counts on
    `_VAULT_SCAN_TTL_S`, and the `health` sub-object on `_SCORECARD_TTL_S`. Health
    is the slow half — 12.81 s measured 2026-11-05 over the live ledger, longer
    than the walk's 10.0 s TTL on its own — so whichever runs first is the one
    that arrives at the browser older than the window it was sold on, and
    measuring the walk first left its entry stale before the response was even
    assembled. One page load later (the ~0.4 s between the two cycles the warm
    test below times) that entry reads as expired and the board is walked again.

    This node is that defect made measurable without a corpus or a stopwatch:
    `health` is stubbed to sleep well past the walk's TTL — 0.50 s against a TTL
    of 0.20 s, so the wrong order overshoots by 2.5x instead of sitting on the
    boundary — and the claim is that when `_backlog()` returns, the short-TTL
    half is still inside its own TTL, which is only true if it was measured after
    the slow half. Both entries are read back out of `dash._cache`, and the
    payload is checked for both halves so a reorder cannot quietly drop one.
    """
    board = dash.vault_root() / "backlog"
    files = ([p for p in board.glob("*.md") if p.name[:1].isdigit()]
             if board.exists() else [])
    dash._cache.clear()
    ttl, slow_for = 0.20, 0.50

    def slow_health(backlog_dir) -> dict:
        time.sleep(slow_for)
        return {"pinned": "health half"}

    monkeypatch.setattr(dash, "_VAULT_SCAN_TTL_S", ttl)
    monkeypatch.setattr(dash, "_backlog_health", slow_health)
    out = dash._backlog()

    walk_entry = dash._cache.get("backlog")
    health_entry = dash._cache.get("backlog_health")
    assert walk_entry is not None and health_entry is not None, (
        "`_backlog()` must leave both cached halves behind, got "
        f"{'backlog' if walk_entry else '(no backlog entry)'} and "
        f"{'backlog_health' if health_entry else '(no backlog_health entry)'}")
    assert out.get("health") == {"pinned": "health half"}, (
        f"the health half has to reach the payload it was computed for, got "
        f"{out.get('health')!r}")
    assert isinstance(out.get("total"), int), (
        f"the raw-count half has to reach the payload too, got "
        f"{out.get('total')!r}")

    age = time.monotonic() - walk_entry[0]
    assert age < ttl, (
        f"the short-TTL half was {age:.2f} s old when `_backlog()` returned, "
        f"past the {ttl:.2f} s the section sells it on, while the slow half it "
        f"was assembled with took {slow_for:.2f} s to measure. Measuring the "
        "fast half first serves it already stale and leaves its entry expired "
        "for the next reader, which then walks the board again — "
        f"{len(files)} item files on this board. Measure the short-TTL half last.")
    assert walk_entry[0] > health_entry[0], (
        f"the short-TTL entry (stamp {walk_entry[0]:.3f}) has to be taken after "
        f"the long-TTL one ({health_entry[0]:.3f}); otherwise the half of the "
        "payload that promises the least staleness is its oldest measurement")


WARM_CYCLE_SAMPLES = 5


def warm_vs_cold_verdict(cold: float, warms: list[float]) -> tuple[float, float, bool]:
    """Grade a cold/warm pair as the 10x clause states it, and return the readings.

    The warm cost is the CHEAPEST sample in `warms`, never its mean or its last. A
    warm cycle is milliseconds of dict lookups, so a sample of it can only be
    inflated from the true cost — by the scheduler handing the worker away to one of
    the others while the gate's suite runs beside it — and can never be deflated
    below it. The floor is therefore the only estimator here that measures the code
    rather than the box.

    Taking it costs the clause nothing it was watching. The failure this node exists
    for — #1204 section 5's section recomputing on every poll — makes EVERY warm
    cycle expensive, so its floor is as high as its mean and 10x still fails.
    `test_a_dead_cache_still_reddens_under_the_warm_floor` pins that, because an
    estimator that only ever turns a red into a green is a way of grinding to green
    and not a measurement.
    """
    warm = min(warms)
    ratio = cold / warm if warm > 0 else float("inf")
    return warm, ratio, ratio >= 10


def test_a_dead_cache_still_reddens_under_the_warm_floor():
    """The floor estimator must be blind to noise and to nothing else.

    Two shapes, both from measurements this file already carries. The first is the
    sample round `SM_20260917_055508` said the old code let through — cold 1.69 s
    with a 0.9 s warm cycle, 1.9x — where every sample is expensive because the
    cache is not doing its job: the floor cannot rescue it and the verdict is False.
    The second is #3844's actual red, cold 5.548 s with samples scattered from
    0.071 s to 1.10 s by a busy scheduler: the floor reads the 0.071 s cycle, which
    is the only one of those five that was not descheduled, and the clause holds.

    A fully dead cache is the third case: warm identical to cold at 1.69 s, under
    1x, reddened. If a future edit makes this return True for either of the first
    and third shapes, the ratio clause has stopped being a measurement.
    """
    warm, ratio, within = warm_vs_cold_verdict(1.69, [0.90, 0.92, 0.88, 0.95, 0.91])
    assert not within, (
        f"a warm path that is expensive on every sample must redden, got {ratio:.1f}x")
    assert warm == 0.88, f"the floor is the cheapest sample, not its mean: {warm}"

    warm, ratio, within = warm_vs_cold_verdict(
        5.548, [0.840, 0.071, 0.620, 1.100, 0.300])
    assert within, (
        f"one unscheduled sample of a 71 ms warm cycle must satisfy the clause "
        f"despite the inflated ones, got {ratio:.1f}x")
    assert warm == 0.071, f"the floor is the cheapest sample: {warm}"

    _, ratio, within = warm_vs_cold_verdict(1.69, [1.69] * WARM_CYCLE_SAMPLES)
    assert not within, (
        f"warm reading exactly like cold is a dead cache, got {ratio:.1f}x")


def test_the_two_claused_nodes_are_executed_and_neither_was_marked_away():
    """#3844 clause 3: the two nodes this item names reached green by measuring.

    A red node has four cheap endings — `skip`, `xfail`, deletion, or a
    `live_vault` mark that the gate's own selection
    (`-m "not live_vault and not fault_injection"`) drops — and all four leave the
    suite green while the thing they covered stays unmeasured. The clause therefore
    asks that neither node took one, which is checkable without a timing run or a
    live store: the node exists in its file, carries no such decorator, its file
    carries no such module-level `pytestmark`, and its body still contains an
    `assert` rather than a print. The last arm is the one that matters after a fix
    like this round's, because hollowing an assertion out is the fifth cheap ending
    and it does not even need a marker.

    Pinned across both files on purpose: the item's two clauses are one acceptance,
    and a reader who has only this node can see both named nodes are still there,
    still decorated nothing, and still asserting.
    """
    import ast
    from pathlib import Path

    forbidden = ("skip", "skipif", "skip_if", "xfail", "live_vault")
    targets = {
        Path(__file__): "test_the_warm_cycle_is_an_order_of_magnitude_cheaper_than_cold",
        Path(__file__).with_name("test_retrieval_seed_anchoring.py"):
            "test_the_anchorless_residue_survives_the_corpus_growth_and_is_pinned",
    }
    for path, name in targets.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", "") == "pytestmark" for t in node.targets):
                rendered = ast.unparse(node.value)
                assert not any(f in rendered for f in forbidden), (
                    f"{path.name} put {rendered} on every node in the file, which is "
                    "how one marker retires a whole clause")
        fn = next((n for n in tree.body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and n.name == name), None)
        assert fn is not None, (
            f"{name} is not in {path.name} — a deleted or renamed node is not a "
            "passing one, and #3844 names it explicitly")
        marks = [ast.unparse(d) for d in fn.decorator_list]
        assert not any(f in m for m in marks for f in forbidden), (
            f"{name} is marked away by {marks}: the clause asks that it run")
        assert any(isinstance(n, ast.Assert) for n in ast.walk(fn)), (
            f"{name} contains no assert, so it can no longer fail and has stopped "
            "measuring the thing its name states")


async def test_the_warm_cycle_is_an_order_of_magnitude_cheaper_than_cold(monkeypatch):
    """The cold bound is only meaningful against a much cheaper warm baseline.

    Advisory finding on round `SM_20260917_055508`: "Docstring states a 10x
    warm/cold ratio contract but the code asserts only the absolute
    'warm < 1.0'; at the measured cold 1.69 s a 0.9 s warm cycle (1.9x) would
    pass while the stated contract is violated." Correct, and the fix is to
    measure both here rather than assert a number the neighbouring sentence
    never checks. Cold and warm are timed back to back in this one test, on the
    same board and the same ledger, so the ratio is a measured pair and not a
    pair of numbers from two runs.

    Measured on this branch at the 1,141-file board: cold 1.69 s, warm 0.073 s —
    a 23x ratio. The assert asks for 10x, the margin that says the caches are
    doing their job. #1204 section 5 measured the *old* warm path (a 2 s poll
    against `_VAULT_SCAN_TTL_S = 10.0`) and got every fifth request paying a ~2 s
    walk — a warm cycle at 1x the cold one, which is exactly what this fires on.

    The 10x is not a restatement of that 2 s/10 s pairing; it cannot be, since
    TTL-vs-poll is a schedule property and this is a wall-clock measurement. What
    it pins is the failure that pairing produced: a section recomputing on a poll
    whose TTL has not expired. If warm and cold ever read alike, the cache is
    dead and the cold test above has been timing the cache-miss path forever
    without saying so. The absolute `warm < 1.0` stays as the second half — it
    catches a machine where *both* cycles are slow, which a ratio cannot see.

    One warm sample is what #3844 found red at base 173d522c with this file's diff
    absent: cold 5.548 s against warm 0.840 s, 6.6x, on a box running the rest of
    the suite beside the node. Both figures were inflated by that load — the cold
    one 3.3x the 1.69 s baseline, the warm one 11x its 0.073 s — and the ratio is
    the collateral, because a 73 ms cycle absorbs a scheduler stall as a large
    fraction of itself while a 5 s cycle barely notices it. This is the same
    parallel-only flake `enforce_cold_cycle_budget` documents for the cold bound
    and refuses to treat a single sample as a verdict about, so the warm side is
    now sampled `WARM_CYCLE_SAMPLES` times and read at its floor by
    `warm_vs_cold_verdict`. The cold side stays one sample because a cache cannot
    be made cold twice inside one process; `warm < 1.0` now bounds the floor, which
    is the weaker of the two readings of that half and says so: it reddens when even
    the best cycle on a saturated box exceeds a second.
    """
    files = board_files_or_stop(what="warm-vs-cold comparison", numeric_names=True)
    led = timed_ledger_or_stop(monkeypatch, what="warm-vs-cold comparison")
    print(f"ledger for this pair: {led.path} ({led.path.stat().st_size:,} bytes) "
          f"| {led.why}")
    cold_started = time.perf_counter()
    await _cold_cycle()
    cold = time.perf_counter() - cold_started
    warms = []
    for _ in range(WARM_CYCLE_SAMPLES):
        started = time.perf_counter()
        await _cold_cycle()
        warms.append(time.perf_counter() - started)
    warm, ratio, within_ratio = warm_vs_cold_verdict(cold, warms)
    print(f"cold {cold:.3f}s → warm {warm:.3f}s "
          f"(best of {len(warms)}: {', '.join(f'{w:.3f}' for w in warms)}) over the "
          f"same {len(files)} item files: {ratio:.1f}x")
    assert within_ratio, (
        f"the warm cycle took {warm:.3f}s — its cheapest of {len(warms)} samples, "
        f"all of them: {', '.join(f'{w:.3f}' for w in warms)} — against a cold cycle "
        f"of {cold:.3f}s: {ratio:.1f}x, under the 10x this clause states. Baseline on "
        f"this branch: cold 1.69 s, warm 0.073 s, 23x. A warm cycle costs "
        f"milliseconds of dict lookups, so a warm cycle whose BEST sample is "
        f"anywhere near the cold one means a section is recomputing on every poll "
        f"regardless of TTL — the 2 s spike in #1204 section 5, and it also means "
        f"the cold test above has been timing the cache-miss path while reading as a "
        f"bound.")
    assert warm < 1.0, (
        f"the best of {len(warms)} warm cycles, with every section cache warm, took "
        f"{warm:.2f}s. The ratio above can be satisfied by a machine where cold and "
        f"warm are both slow; this is the absolute half of the same claim.")


async def test_the_usage_section_publishes_by_skill_24h_beside_by_model_24h():
    """#783 clause 4: the per-skill breakdown reaches the payload a browser
    decodes, and the model breakdown it sits beside is untouched.

    Read through the real route handler, not by calling `_usage()` directly: the
    section is serialised by `_gather` on a worker thread (`dashboard.py:937`),
    so this is the one assertion that catches a value the JSON encoder refuses
    and a block that never leaves the process. `by_model_24h`'s keys are pinned
    literally, because the frontend types it (`web/src/api.ts:2014`) and the
    clause is that its shape and rows do not change.
    """
    from app import usage_store
    from app.harness import skill_dispatch as sd

    usage_store.record_usage(
        session_id="dash-skill", model="primary", input_tokens=1200,
        output_tokens=60, cache_create=10, cache_read=900,
        skills=sd.skill_deliveries(
            '<skill name="youtube-transcript" score="4.1" excerpt="true">\n'
            "excerpt\n</skill>"),
    )

    payload = await _cold_cycle()
    usage = payload["usage"]
    assert "error" not in usage, f"the usage section errored out: {usage}"

    assert usage["by_model_24h"] == [
        {"model": "primary", "requests": 1, "input_tokens": 1200,
         "output_tokens": 60, "cache_create": 10, "cache_read": 900,
         "cost_usd": 0.0},
    ], usage["by_model_24h"]

    assert usage["by_skill_24h"] == [
        {"skill": "youtube-transcript", "route": "prefetch_excerpt",
         "requests": 1, "input_tokens": 1200, "output_tokens": 60,
         "cache_create": 10, "cache_read": 900},
    ], usage["by_skill_24h"]
    print("usage section: by_model_24h 1 row, by_skill_24h 1 row "
          "(youtube-transcript/prefetch_excerpt)")
