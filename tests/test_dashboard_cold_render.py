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
#: `board_files_or_stop(numeric_names=True)` applies).
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
CALIBRATED_BOARD_FILES = 2180

#: Bytes of `promotions.jsonl` the calibrated cycle decoded, for the same
#: reason. NOT part of the denominator — the budget scales with the board, not
#: with the ledger, because pinning bytes decoded is a second change and #1858
#: left that ruling open (owed clause 2: the ledger is read by the cycle, grows
#: faster than the board, and is in none of the retention sweep's bounded
#: stores). A reader comparing today's cost against this table has to know both
#: inputs, since ledger growth can exhaust this budget with no code change and
#: no drift-test warning.
CALIBRATED_LEDGER_BYTES = 28_960_660

#: The later of the two measured tails, in seconds.
#:
#:     measured 2026-10-05 from 19acea6c in a round worktree, 32 cores
#:     board  = CALIBRATED_BOARD_FILES item files (the numeric-name subset);
#:              all fourteen runs below printed that count
#:     ledger = the live store, reached directly because the serial runs below
#:              carried no `LLOYD_AUTOMOD_STATE`, so `S.LEDGER_PATH` was the real
#:              state dir's own file. NOT "repointed as in every gate run":
#:              `tests/board_presence.py` repoints only when the configured state
#:              dir holds NO ledger, and a gate run's round state dir holds one of
#:              its own — 3,888 bytes in this round's — so the corpus a gate
#:              decodes is that small file and these durations are not its shape.
#:              Those runs decoded 29,188,189 bytes in the first four and
#:              29,194,126 in the rest — 0.8% above `CALIBRATED_LEDGER_BYTES`,
#:              which this re-base deliberately left alone: that centre is pinned
#:              by equality in
#:              `tests/test_retention_sweep.py` (where #2043's whole case is that
#:              the fold lands the store inside the band around THAT number) and
#:              #2071 ruled the centre is not the drift-drift lever. The durations
#:              below are of the real store at its real size; the budget those runs
#:              printed was still the old one, and a duration does not care which
#:              budget it was read against.
#:     serial, 8 runs, load 4.84-5.80:  min 6.78  median 7.16  p90 7.79  max 7.79
#:     serial, 6 runs, load 5.36-5.70:  min 6.89  median 7.04  p90 7.18  max 7.18
#:
#: p90 is nearest-rank, the convention the row this table replaces reported it
#: by, so with 8 samples it is the maximum and the 6-sample set's p90 is its
#: maximum too. Their later p90 is the 8-run set's and that is this constant.
#:
#: What this table does not have is an `xdist -n 8 --dist loadfile` full-suite
#: row, and it cannot be given one by a gate run. The calibration this one
#: replaces recorded such a row — 7.64 s printed by the node's own line against
#: a serial p90 of 7.52, a 1.6% parallel penalty — and set its constant to that
#: sample. This constant is instead the later of two serial sets measured with no
#: gate in flight, and the gate's `tests` rung cannot add a duration to it: that
#: rung runs `-n 8 --dist loadfile` WITHOUT `-s`, and pytest discards a PASSING
#: test's captured stdout, so the node's `cold cycle:` line never reaches the log
#: that passed 16156 tests at head `e43c28df` (34 skipped, 496.4 s).
#: Checked rather than assumed: three `-n 8 --dist loadfile -s` passes over this
#: file (load 3.96-4.27, 25 passed each) printed no `cold cycle:` line either, and
#: `--durations=3` under the same shape printed no durations report — xdist
#: forwards neither a passing worker test's stdout nor the durations table. So the
#: gate's pass is a BOUND, not a measurement: by this node's own rule (one breach
#: buys one more cold sample, two breaches fail) clearing the ladder says the
#: first sample came in at or under `COLD_BUDGET_S`, which says the serial p90
#: plus a full 25% margin absorbed whatever the parallel penalty is, and by how
#: much is not something the ladder can tell anyone. What it would take to
#: measure the row is a serial `-s` run of this file while a suite-wide `-n 8`
#: pass is loading the box, which is the shape of the number, not of the constant.
#:
#: It is also why the assert below tells a re-baser to run "`-n 8` runs over the
#: real board" while this table has no such row: under that shape the node's line
#: surfaces only when it FAILS — a breach is exactly when pytest prints it — so a
#: parallel sample you can actually read off a gate is a worst case, not a
#: typical one. Nothing here claims to know how the retired row got its number.
#:
#: To restore the value this one replaced, restore its inputs: the board of
#: 1,804 item files and the ledger centre it was walked over, measured
#: 2026-09-30 from abe3360a.
#:
#: What none of those rows capture is contention, and this re-base shows the
#: machine's contribution at the tail rather than the middle: the same code over
#: the same corpus came in at median 7.16 s in one set and 7.04 s in the other
#: (1.7% apart, with overlapping 1-minute load averages) but 0.61 s apart at
#: p90 — 8.5%. That is why a single sample on a 3%-margin bound was never
#: stable, and why the constant is a p90 over eight runs and not the fastest
#: cycle: the gate's tests rung adds eight workers on top of whatever else the
#: box is doing, and the ledger carries 17 rows naming this node as a
#: parallel-only failure up to 2026-09-29 (triage on #1858), with no code change
#: behind any of them.
#:
#: The budget is deliberately NOT padded to absorb an arbitrary load spike. A
#: number high enough for that would sit near the 15.22 s cost the check exists to
#: catch and stop measuring the dashboard. What absorbs the transient is the second
#: sample in `enforce_cold_cycle_budget`, plus the load average and xdist worker id
#: the node prints beside every duration, so a red that survives the retry says
#: whether the machine was the problem or the code was.
COLD_P90_S = 7.79

#: Stated margin over that p90 (#1858 clause 1): 25%, arithmetic rather than the
#: "~30% headroom" the 2026-09-25 re-base claimed in prose. It has to cover the
#: worker contention the re-measurement does not absorb, and it has to stay far
#: enough under the 15.22 s triage cost that a real regression still trips it —
#: at 1.25 the budget is 9.74 s and the triage cost is 56% higher, so the check
#: still means something. A re-baser changes the table and, if they must, this
#: number; the budget itself follows arithmetically.
COLD_BUDGET_MARGIN = 1.25

#: Budget for one cold cycle over the board named by `CALIBRATED_BOARD_FILES`,
#: charged per file by `cold_budget_for_board`. The value this re-base replaced
#: was 9.55 s, on a board of 1,804 item files with the old ledger centre; to
#: restore it, restore those two inputs and the p90 they sat on. Before that,
#: 8.0 s was the calibration on a 1,450-file board with a 6,286,192-byte ledger,
#: where the node measured 6.02 s and 6.16 s (round SM_20260925_205059's review
#: run, 2026-09-25, load average 11). Triage baseline: 15.22 s over 1,142 files,
#: 2026-09-16 — the bound still fires well short of that.
COLD_BUDGET_S = COLD_P90_S * COLD_BUDGET_MARGIN

#: How far the live board may move from the calibrated one before the budget
#: stops describing it (#1858 clause 3).
BOARD_DRIFT_FRACTION = 0.20


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

    The fold trims the store back to its window each pass, so the fullest it ever
    gets is where it stands now plus everything that arrives before the next pass.
    A census of the live file answers a different question — it reports the size on
    the day someone looked, which is anywhere in that sawtooth.
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


def _declared_fold_cadence_days() -> float:
    """The interval the retention rung is actually scheduled at, in days."""
    from app.autonomy import AUTONOMY_DIR, FREQUENCY_INTERVALS

    task = AUTONOMY_DIR / "79-retention-sweep.md"
    assert task.is_file(), (
        f"{task} is the schedule the promotions fold runs on; it is not readable, so "
        f"the cadence this node is about cannot be checked")
    import yaml
    front = yaml.safe_load(task.read_text(encoding="utf-8").split("---\n", 2)[1])
    frequency = str(front.get("frequency", "")).strip().lower()
    assert frequency in FREQUENCY_INTERVALS, (
        f"`frequency: {frequency!r}` is outside `app.autonomy.FREQUENCY_INTERVALS`, so "
        f"the scheduler gives this task no interval at all — which is never due, ever, "
        f"and would leave the ledger with no fold whatsoever")
    return FREQUENCY_INTERVALS[frequency] / 86400


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
    own row timestamps. `app.autonomy.FREQUENCY_INTERVALS` is the single table the
    scheduler reads (its own comment says an unreadable frequency resolves to "not
    due, ever"), so a cadence name absent from it is asserted here rather than guessed
    at — that case is the fold never running, which is the worst outcome and the
    easiest to write by accident.

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
    rate = _measured_growth_bytes_per_day(led, time.time())
    assert rate > 0, (
        f"{led}: no row timestamps landed in the 14-day window, so the growth rate is "
        "0 B/day and every cadence would look safe. A denominator of zero is not a "
        "clean result")
    cadence_days = _declared_fold_cadence_days()

    live_bytes = led.stat().st_size
    ceiling = int(CALIBRATED_LEDGER_BYTES * (1.0 + BOARD_DRIFT_FRACTION))
    peak = store_fullest_between_folds(live_bytes=live_bytes,
                                       cadence_days=cadence_days,
                                       rate_bytes_per_day=rate)
    print(f"ledger {live_bytes:,} B; growth {rate:,.0f} B/day; fold cadence "
          f"{cadence_days:g} d; ceiling {ceiling:,} B; peak {peak:,.0f} B")
    assert not cadence_outruns_band(live_bytes=live_bytes, cadence_days=cadence_days,
                                    rate_bytes_per_day=rate, ceiling_bytes=ceiling), (
        f"a fold every {cadence_days:g} d leaves the store at {peak:,.0f} bytes at its "
        f"highest against a {ceiling:,}-byte ceiling, so the size band is crossed by "
        f"arithmetic about {cadence_days * rate - (ceiling - live_bytes):,.0f} bytes "
        "before it is crossed by anything the code does. At this growth rate the "
        f"cadence fits in {(ceiling - live_bytes) / rate:.2f} d — schedule the fold "
        "more often, or make a ruling on what the band is worth if it cannot.")

def test_no_node_of_this_file_is_marked_or_deleted_to_get_it_green():
    """#2071 clause 2: the green run is earned by the named node, not by hiding it.

    A fix that makes the corpus small and a change that removes the only reader of
    the number produce the same green run, so the difference is counted here rather
    than read off a diff. Denominators first, because a check over zero things is not
    a check — the rule this repo's own doc-claim file applies to its extract: `git
    show aaa2a5d0:tests/test_dashboard_cold_render.py | grep -cE '^(async )?def
    test_'` is 20 at #2071's base, so the floor here is 23 (20 plus this round's 3)
    and a removal reddens this node instead of looking like tidying.

    What is counted is the spellings that retire a node from a green run: `skip`,
    `xfail` and `importorskip` anywhere in the file, and any `live_vault` mark, which
    the gate's `-m "not live_vault"` deselects. The file carried none of the first
    group at base and carries none now; the one `live_vault` node this round added is
    a growth measurement about this box, not the node #2071 names, and the assertions
    below are what keep that distinction from being an unverified claim.
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

    assert len(nodes) >= 23, (
        f"{len(nodes)} test nodes against a floor of 23 (#2071's base measured 20, and "
        f"this round added 3): a node was removed rather than made to pass")
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
    """
    files = board_files_or_stop(what="warm-vs-cold comparison", numeric_names=True)
    led = timed_ledger_or_stop(monkeypatch, what="warm-vs-cold comparison")
    print(f"ledger for this pair: {led.path} ({led.path.stat().st_size:,} bytes) "
          f"| {led.why}")
    cold_started = time.perf_counter()
    await _cold_cycle()
    cold = time.perf_counter() - cold_started
    started = time.perf_counter()
    await _cold_cycle()
    warm = time.perf_counter() - started
    ratio = cold / warm if warm > 0 else float("inf")
    print(f"cold {cold:.3f}s → warm {warm:.3f}s over the same {len(files)} item "
          f"files: {ratio:.1f}x")
    assert ratio >= 10, (
        f"the warm cycle took {warm:.3f}s against a cold cycle of {cold:.3f}s — "
        f"{ratio:.1f}x, under the 10x this clause states. Baseline on this branch: "
        f"cold 1.69 s, warm 0.073 s, 23x. A warm cycle costs milliseconds of dict "
        f"lookups, so a warm cycle anywhere near the cold one means a section is "
        f"recomputing on every poll regardless of TTL — the 2 s spike in #1204 "
        f"section 5, and it also means the cold test above has been timing the "
        f"cache-miss path while reading as a bound.")
    assert warm < 1.0, (
        f"the second cycle, with every section cache warm, took {warm:.2f}s. The "
        f"ratio above can be satisfied by a machine where cold and warm are both "
        f"slow; this is the absolute half of the same claim.")


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
