#!/usr/bin/env python3
"""Dose rate and clearance for the three loaded memory files (#4122).

The byte ceilings in `app/prompt_surface.py` are a LEVEL reading, and a level is a
lagging instrument on this surface: `lloyd/USER.md` sat at 16,247 B of a 16,384 B
ceiling on 2026-11-15 — 137 B of clearance — while its level's effective half-life
is ~89 days, so a change in how fast the file is being written takes months to move
the byte count that is supposed to warn about it. And the statistic that looks
healthy at that level is the wrong one: USER.md's 60-day NET is about −2 B/day, so a
report on net reads "fine" forever while every append keeps hitting the ceiling and
every retire only has to be as big as the last one. Net is the residue of two
numbers; the risk lives in the inflow half of them, which no instrument on this box
added. This script is that instrument: gross inflow (`bytes_in`), gross outflow
(`bytes_out`), the level (`pool`), the bound (`ceiling`) and the ratio that actually
decides urgency (`headroom_days`).

    headroom_days = (ceiling − pool) / bytes_in_per_day

`bytes_in_per_day` is the sum of the POSITIVE blob-size deltas inside the window,
divided by the window: `git cat-file -s` on every commit that touched the file, so a
day that adds 400 B and a day that retires 400 B both count 400 into the dose. The
formula deliberately never reads `net`.

What the numbers mean — and what they do not
--------------------------------------------
**A `headroom_days` figure is meaningless without the window it was divided by**, so
the window is a column of the row, not a caption on the print. The same file with the
same formula over 30 / 45 / 60 / 84 days read 12.4 / 8.1 / 5.2 / 0.14 on 2026-11-15.
Any comparison between two rows of the series is valid only at equal `window_days`.

**The first week is expected to be quiet, and a quiet week is not a dead
instrument.** The synthesis that filed this item published a 1.2-day headline for
USER.md and pre-registered its own falsifier — `line 97` of
`knowledge/synthesis/2026-11-15-the-dose-is-not-the-level-loaded-memory-kinetics.md`,
a first-night value over 1.5 retires the headline. That falsifier came out
false — 1.2 days was a per-commit median (+138 B) read as a per-day rate — so what
survives is not a calibration of today's dose but a **regime detector**: over the
84-day window USER.md ran **974.9 B/day** of gross inflow with a **+12,063 B** growth
commit, which is `headroom_days = 0.14` and fires the alarm. That is exactly the
September regime that put USER.md at 95,302 B and MEMORY.md at 242,446 B (the git
maxima for both) with nothing able to refuse it, because a ceiling can only refuse a
write that is already over it. Today's dose reads 5–12 days and the alarm stays shut;
a September-shaped climb reads well under 1 and does not.

Why the cut is 1.0 day
----------------------
One day is the interval this record itself runs on: at a dose that exhausts the
clearance in less than a day, the next reading is taken from a file that is already
over its ceiling, so the alarm would be reporting a condition it could no longer have
caught early. It is a response-time bound, not a forecast — and it says nothing about
what to DO, because of the standing rule below.

The alarm trims nothing (#1881)
-------------------------------
`#1881` — "never make room". Clearance that cannot meet the dose is reported as a
shortfall in bytes and days and nothing else happens: this script opens the three
files to READ their size and never writes to any path under the vault, so a run whose
alarm fires leaves every loaded file byte-identical. That is pinned by
`tests/test_loaded_memory_dose_rate.py`. A future reader who wants the alarm to
trigger a retire has to change that rule, not this script.

SOUL.md has dose and clearance, and can never alarm
---------------------------------------------------
`lloyd/SOUL.md` is loaded FIRST into every prompt (`app/prompt_builder.py:224`,
`:545`, `:558`) but has no byte ceiling by ruling, not by oversight: `#377` settled
its bound as the two shape ratios plus the load-bearing markers, and
`app/prompt_surface.py:84-86` refuses a size constant there as "a second, unblessed
answer to a question #377 already answered". So its row carries `ceiling: null` and
`headroom_days: null` by construction — the instrument reports what a SOUL.md climb
costs without claiming a clearance it has no authority to measure. Whether SOUL.md
should get an alerting-only bound is a re-settlement of #377 and is owed to a person.

Where the row goes, and why it is idempotent
--------------------------------------------
`--record` appends **one row per loaded file per day** to
`loaded-memory-dose-rate.jsonl` in `scripts/automod/state.py`'s `STATE_DIR` — beside
`gate_to_land_lag.jsonl`, `landed_outcomes.jsonl` and `scorecard.jsonl`, in the store
the loop already has. It is not a new store and it is not in the checkout, where the
guardian alerts on runtime data hourly (#1667). The dedup key is `(date, file)`: a
second `--record` on the same UTC date appends nothing, because a series that takes
two rows for one day also silently redefines what "one reading a day" was measuring.
A window comparison is therefore a REPORT (`--window-days` without `--record`), not a
row — otherwise a caller could change the series' denominator by passing a flag.

    python scripts/maintenance/loaded_memory_dose_rate.py [--window-days N]
        [--vault PATH] [--state-dir PATH] [--record]

Exit 0 when no file is under the cut, 1 when at least one is. **The exit code is the
verdict, not the run's health** — `--record` has already appended its rows by the time
either code returns, so a red reading is a reading, and a run that appended rows and
exited 1 must never be re-run to "fix" it.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Appended, not inserted: `app/` holds top-level modules that would shadow a package
# of the same name for anything else in this process (the rule at
# `scripts/maintenance/gate_to_land_lag.py:65-69`). Invoked as a file, neither `app`
# nor `scripts` is importable without this.
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

#: The report's own bucket, beside `gate_to_land_lag.jsonl` in the automod state dir.
RECORD_NAME = "loaded-memory-dose-rate.jsonl"

#: The three files a prompt build reads, in READ ORDER — SOUL.md first
#: (`app/prompt_builder.py:545`, `:558`), then the two members `_memory_files_for`
#: selects. Order matters to the print: a reader scanning the top of a prompt sees
#: these bytes in this order.
LOAD_ORDER: tuple[str, ...] = ("SOUL.md", "MEMORY.md", "USER.md")

#: The loaded files live in the vault's `lloyd/` segment; that prefix is what every
#: `git log --` and `<rev>:<path>` spec below is built from, and `MEMORY_CEILINGS`
#: is keyed by the BARE filename, so a lookup that passed `lloyd/USER.md` would
#: return None and read as "USER.md has no ceiling".
VAULT_SUBDIR = "lloyd"

#: The divisor when the caller does not name one. 30 days is the span the triage
#: measurements of 2026-11-15 were quoted at, so a first row is comparable to the
#: figures in the item — and the row still carries `window_days`, because that span
#: is a convention and not a property of the file.
DEFAULT_WINDOW_DAYS = 30

#: The alarm cut, in days of clearance. See "Why the cut is 1.0 day" above.
HEADROOM_CUT_DAYS = 1.0

#: Printed for a `None` so every column of every line is greppable, including the
#: two that SOUL.md carries as null by construction.
NULL = "null"


def _git(vault: Path, *args: str, stdin: str | None = None) -> str:
    """One git call against `vault`. A non-zero exit raises; it is never swallowed.

    Silence here would be the worst possible failure mode: a report that printed
    `bytes_in=0` because git could not be found is indistinguishable from a file
    nobody has been writing to, which is the exact state this instrument exists to
    tell apart from a dose that went unmeasured.
    """
    r = subprocess.run(["git", "-C", str(vault), *args],
                       capture_output=True, text=True, input=stdin, timeout=180)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} in {vault} exited {r.returncode}: "
                           f"{r.stderr.strip()}")
    return r.stdout


def _commits_touching(vault: Path, rel_path: str) -> list[tuple[str, int]]:
    """`(sha, author-unix-seCONDS)` for commits that touched `rel_path`, newest first."""
    out: list[tuple[str, int]] = []
    for line in _git(vault, "log", "--pretty=%H%x09%at", "--", rel_path).splitlines():
        sha, _, at = line.partition("\t")
        if sha.strip() and at.strip().lstrip("-").isdigit():
            out.append((sha.strip(), int(at)))
    return out


def _blob_sizes(vault: Path, revs: list[str], rel_path: str) -> list[int | None]:
    """Size of `rel_path` at each rev, in the caller's order; None where it is absent.

    One `git cat-file --batch-check` call for the whole walk rather than one
    `cat-file -s` per commit: `lloyd/MEMORY.md` has over a hundred commits touching
    it, and a loop of git processes over a daily job is how a probe becomes the
    slowest thing in its own cron slot. `--batch-check` echoes the spec followed by
    `missing` for a path the rev does not carry, which is how a file's own creation
    is told apart from a size.
    """
    if not revs:
        return []
    raw = _git(vault, "cat-file", "--batch-check",
               stdin="".join(f"{rev}:{rel_path}\n" for rev in revs))
    sizes: list[int | None] = []
    for line in raw.splitlines():
        parts = line.split()
        sizes.append(int(parts[2]) if len(parts) >= 3 and parts[1] == "blob" else None)
    if len(sizes) != len(revs):
        raise RuntimeError(f"git cat-file --batch-check answered {len(sizes)} lines "
                           f"for {len(revs)} specs in {vault}")
    return sizes


def gross_flux(vault: Path, rel_path: str, window_days: int,
               now: datetime) -> tuple[float, float]:
    """`(gross_bytes_in_per_day, gross_bytes_out_per_day)` over the trailing window.

    Walks every commit that ever touched the file and keeps the deltas whose commit
    falls inside the window, so the first in-window delta is still measured against
    the size the file had the day before the window opened — a window that dropped
    that edge would understate the dose by exactly the one change that motivated
    asking. A rev where the path is absent counts as 0 bytes, so the commit that
    CREATES the file lands as pure inflow: it is a real dose, and a report that
    called it nothing would read as a clean start for a file being written at full
    speed.

    Positive deltas go to `bytes_in`, negative ones to `bytes_out`, and both are
    divided by `window_days` — the declared window, not the span between the first
    and last commit inside it. The two differ whenever a window contains no commits,
    and dividing by the span would then divide a small number by a small number and
    call the result a rate.
    """
    if window_days <= 0:
        raise ValueError(f"window_days must be positive, got {window_days}")
    commits = _commits_touching(vault, rel_path)
    ordered = list(reversed(commits))                      # oldest first
    sizes = _blob_sizes(vault, [sha for sha, _ in ordered], rel_path)
    floor_ts = now.timestamp() - window_days * 86_400.0
    gross_in = gross_out = 0
    prev = 0
    for (_sha, ts), size in zip(ordered, sizes):
        cur = 0 if size is None else size
        delta = cur - prev
        prev = cur
        if ts < floor_ts:
            continue
        if delta > 0:
            gross_in += delta
        elif delta < 0:
            gross_out += -delta
    return (gross_in / window_days, gross_out / window_days)


def dose_row(vault: Path, filename: str, window_days: int = DEFAULT_WINDOW_DAYS,
             now: datetime | None = None) -> dict:
    """One file's dose and clearance, as the row `--record` appends.

    `ceiling` comes from `app.prompt_surface.memory_ceiling` — the same accessor the
    two writers call — so the day a ceiling constant moves, this report moves with it
    instead of guarding a second copy of the number that is already wrong. A name it
    returns None for (`SOUL.md`, per #377) gets `headroom_days: None`: there is no
    clearance to divide without a bound, and inventing one here would be the
    byte-constant ruling the module already refused.

    A file that is absent from the vault reports `pool: None` and no `headroom_days`
    rather than `pool: 0`: zero bytes is a legitimately-empty memory file, and a
    missing one is a different fact that must not be printed as that fact.
    """
    now = now or datetime.now(timezone.utc)
    rel_path = f"{VAULT_SUBDIR}/{filename}"
    from app.prompt_surface import memory_ceiling
    bytes_in, bytes_out = gross_flux(vault, rel_path, window_days, now)
    path = vault / rel_path
    pool: int | None = len(path.read_bytes()) if path.is_file() else None
    ceiling = memory_ceiling(filename)
    headroom_days: float | None = None
    if pool is not None and ceiling is not None and bytes_in > 0:
        headroom_days = round((ceiling - pool) / bytes_in, 3)
    return {
        "ts": now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date": now.astimezone(timezone.utc).strftime("%Y-%m-%d"),
        "file": filename,
        "window_days": window_days,
        "bytes_in": round(bytes_in, 3),
        "bytes_out": round(bytes_out, 3),
        "net": round(bytes_in - bytes_out, 3),
        "pool": pool,
        "ceiling": ceiling,
        "headroom_days": headroom_days,
    }


def dose_rows(vault: Path, window_days: int = DEFAULT_WINDOW_DAYS,
              now: datetime | None = None) -> list[dict]:
    """Rows for all three loaded files, in read order."""
    return [dose_row(vault, f, window_days, now) for f in LOAD_ORDER]


def shortfall_bytes(row: dict, cut_days: float = HEADROOM_CUT_DAYS) -> float:
    """How many bytes MORE clearance the file needs to sit at `cut_days` of headroom.

    The alarm has to name a number a person can act on. `ceiling − pool` says how
    little room is left; it does not say how little is *enough*, and at a dose high
    enough to fire the cut the answer is not zero — a file with 137 B left and a dose
    of 1,000 B/day needs 863 B of additional clearance to be one day out of trouble,
    which is the size of the retire that would actually settle it. 0.0 when the row
    is not alarming (or cannot alarm), so a caller can print it unconditionally.
    """
    hd, ceiling, pool = row.get("headroom_days"), row.get("ceiling"), row.get("pool")
    if hd is None or ceiling is None or pool is None or hd >= cut_days:
        return 0.0
    return max(0.0, round(cut_days * row["bytes_in"] - (ceiling - pool), 3))


def alarm_lines(rows: list[dict], cut_days: float = HEADROOM_CUT_DAYS) -> list[str]:
    """One line per alarming file; empty when every file clears the cut.

    Every line names the file, the clearance left, the shortfall in bytes, the days
    and the window they were computed over — a reader who has only this line and the
    row can re-derive the number. `headroom_days` is a float and can be negative,
    which is the honest shape of a file already past its ceiling: `(ceiling − pool)`
    over a positive dose is negative, and clamping it to 0 would hide how far past.
    """
    out: list[str] = []
    for row in rows:
        hd = row.get("headroom_days")
        if hd is None or hd >= cut_days:
            continue
        clearance = row["ceiling"] - row["pool"]
        out.append(
            f"ALARM {row['file']}: headroom_days={hd} under cut={cut_days} "
            f"(clearance_b={clearance} under ceiling_b={row['ceiling']} at "
            f"bytes_in={row['bytes_in']} B/day over window_days={row['window_days']}, "
            f"shortfall_b={shortfall_bytes(row, cut_days)}). "
            "No room is made (#1881): this names the shortfall and trims nothing."
        )
    return out


def _state_dir() -> Path:
    from scripts.automod import state as S
    # Read off the module at call time, never copied at import: `STATE_DIR` is the
    # directory the loop offers and `LLOYD_AUTOMOD_STATE` its one relocation hook
    # (same as `scripts/maintenance/gate_to_land_lag.py::_state_dir`).
    return Path(S.STATE_DIR)


def record_path(state_dir: Path | None = None) -> Path:
    return (Path(state_dir) if state_dir else _state_dir()) / RECORD_NAME


def recorded_keys(path: Path) -> set[tuple[str, str]]:
    """The `(date, file)` pairs already in the journal. A damaged line is skipped,
    not fatal: one torn row must not stop today's reading from being recorded."""
    keys: set[tuple[str, str]] = set()
    if not path.exists():
        return keys
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("date") and row.get("file"):
            keys.add((row["date"], row["file"]))
    return keys


def record(rows: list[dict], state_dir: Path | None = None) -> list[dict]:
    """Append the rows no `(date, file)` is on yet; return the ones appended."""
    path = record_path(state_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = recorded_keys(path)
    fresh = [r for r in rows if (r["date"], r["file"]) not in seen]
    if fresh:
        with path.open("a", encoding="utf-8") as fh:
            for row in fresh:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
    return fresh


def _fmt(value) -> str:
    return NULL if value is None else str(value)


def format_rows(rows: list[dict]) -> str:
    """One line per row, plus the alarming lines. Keys, not columns: a fixed-width
    table reflows when a filename grows and breaks a grep beside it."""
    lines = [f"file={row['file']} bytes_in={row['bytes_in']} "
             f"bytes_out={row['bytes_out']} net={row['net']} pool={_fmt(row['pool'])} "
             f"ceiling={_fmt(row['ceiling'])} "
             f"headroom_days={_fmt(row['headroom_days'])} "
             f"window_days={row['window_days']}" for row in rows]
    return "\n".join(lines + alarm_lines(rows))


def main(argv: list[str] | None = None) -> int:
    from app.paths import VAULT_ROOT
    # The vault and the state dir both come from the resolvers (#1367), never from
    # `$HOME`: under a gate's `HOME` the vault is a round's symlink and a
    # home-computed path silently measures the wrong tree.
    from scripts.automod.state import STATE_DIR
    ap = argparse.ArgumentParser(
        prog="loaded_memory_dose_rate.py",
        description="Dose rate (gross inflow), outflow and clearance-days for the "
                    "three loaded memory files. Report-only: it never trims and "
                    "never writes to the vault (#1881).")
    ap.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS,
                    help=f"trailing window the rates are divided by "
                         f"(default {DEFAULT_WINDOW_DAYS}); the row carries it, "
                         "because a rate without its window is not a rate")
    ap.add_argument("--vault", default=str(VAULT_ROOT),
                    help="vault root holding lloyd/{SOUL,MEMORY,USER}.md "
                         "(default: app.paths.VAULT_ROOT)")
    ap.add_argument("--state-dir", default=None,
                    help="automod state dir to record into (default: "
                         "scripts.automod.state.STATE_DIR)")
    ap.add_argument("--record", action="store_true",
                    help="append one row per loaded file to "
                         f"{RECORD_NAME}; a second run on the same UTC date adds "
                         "nothing")
    args = ap.parse_args(argv)
    now = datetime.now(timezone.utc)
    vault = Path(args.vault)
    state_dir = Path(args.state_dir) if args.state_dir else Path(STATE_DIR)

    rows = dose_rows(vault, args.window_days, now)
    print(f"loaded-memory dose rate  vault={vault}  window_days={args.window_days}  "
          f"files={len(rows)}")
    print(format_rows(rows))
    if args.record:
        fresh = record(rows, state_dir)
        print(f"record={record_path(state_dir)} appended={len(fresh)} "
              f"skipped_already_recorded={len(rows) - len(fresh)} "
              f"(one row per file per UTC date)")
    return 1 if alarm_lines(rows) else 0


if __name__ == "__main__":
    sys.exit(main())
