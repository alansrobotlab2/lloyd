#!/usr/bin/env python3
"""Per-check health census: how often every gate rung and named eval check has run, how
often it has actually FAILED, when it last failed, and what it costs (#3418).

The motivation is a class of failure this tree has been bitten by repeatedly, always the
same shape: a check that reports a verdict it cannot justify. `tests/test_architecture_coverage_doc_claims.py`
passed for a while on a `git grep` proving command that returned ZERO rows (#1511's landed
finding); `scripts/automod/landed_outcomes.py` shipped a `record()` behind a `--record` flag
that NOTHING called, so its journal did not exist on this box (#2378's own landed finding);
and a rung whose every row is `seconds: 0.0` is not 554 fast healthy runs, it is 554 skips.
None of those three was visible in any artifact this tree had, because every existing reader
asks about ONE rung: `scripts/automod/scorecard.py:812` asks `rung == "review"`, `:1024` asks
`rung == "tests" and ok`, `scripts/automod/backlog.py:3919,3925,3926,3983` ask about
`preflight`/`tests`/`review`/`drill`, `workers/sources/autocode.py:403` asks `review`, and
`eval/diagnosis_fanout_replay.py:275,315` asks `review`. Measured before writing this, those
are ALL of the `get("rung")` sites outside `scripts/automod/gate.py` (one exception is noted
in `--help`: `scripts/autonomy/checkpoint.py:53` matches on `rung` only because `GATE_RUNGS`
is one of the names in its own constant tuple).

So the per-check history was already sitting in the ledger, unread, and this module reads all
of it at once. It computes nothing new about a check: pass, fail, n, wall-clock, and the age
of the last real failure.

Two degenerate states become findings, and only one of them is a verdict on its own:

* **never failed in n runs** — a candidate for vacuous, NOT a finding by itself. Every
  zero-fail check at or above its class floor therefore carries an INJECTION VERDICT: one
  seeded fault that the check ought to catch, tried here, recorded as
  `alive (<fault> caught)` or `vacuous (<fault> not caught)` or `not injectable (<reason>)`.
  A green row with no verdict recorded is the acceptance failure this clause exists to
  prevent, and the reason the module ships one BLINDED CONTROL: a probe that cannot tell a
  check that caught a fault from one that was told nothing about it would print `alive` for
  everything, which is the vacuous-report failure mode one level up.
* **never executed** — prints `0/0 (never executed)`, never a dash and never a fake zero.

What this module NEVER does: retire, disable, weaken or re-scope a check. Disposition is
Alan's call (item clause 5). An always-green rung whose injected fault IS caught is a good
row, and with one user and a quiet tree an always-green rung whose fault cannot be seeded
from here is usually just a green check.

Window: the live promotions ledger UNION every monthly archive beside it, the read shape of
`scripts/automod/gate_to_land_lag.py:140`. This is load-bearing and not a style choice.
The retention sweep rolls old rows into `promotions-archive-<YYYYMM>.jsonl.gz` (`scripts/groundskeeper/retention-sweep.py:732`),
and the two live archives were produced by different rounds of that sweep, so any month can
be one file or the other: 2026-09 lives in the gzip, where all 11 `rollback_succeeded` rows
are, while 2026-10's rows are still in the live file. A census reading only the live file
loses whole RUNGS — `venv` has 0 live rows and 554 archived ones, so it would print
`0/0 (never executed)` for a rung that ran 554 times. The same trap is already recorded in
this tree at `scripts/autonomy/health.py:327-330` and `scripts/automod/review.py:1798-1801`.

Cost is WALL-CLOCK ONLY. The promotions ledger has no token or dollar field, so no figure
here may be labelled a spend or a GPU cost, and the addable field set is one named constant,
`ADDABLE_FIELDS`. The reason is #2378's landed finding: `review.rung_wait_s` is queue wait
for a rung that then ran CONCURRENTLY with tests, `review.waited_s` is a retry sleep already
nested inside `seconds`, and `restart_flushed.waited_s` / `pause_set.seconds` carry no
`round_id` at all — summing the whole `data` block produces an overcount with no owner.
`test_a_naive_sum_of_every_duration_field_is_larger_than_the_addable_set` pins that the
distinction is real rather than rhetorical.

Usage:
    python3 scripts/maintenance/check_health_census.py
    python3 scripts/maintenance/check_health_census.py --state-dir /path/to/fixtures
    python3 scripts/maintenance/check_health_census.py --record
    python3 scripts/maintenance/check_health_census.py --json

`--record` appends exactly one dated row beside `landed_outcomes_path()` — never into the
checkout, where the guardian alerts hourly on in-tree runtime data — and prints the receipt
line naming it. `--json` is the same run's row with no journal write.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import json
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LIVE_LEDGER = "promotions.jsonl"
ARCHIVE_GLOB = "promotions-archive-*.jsonl.gz"
JOURNAL_NAME = "check_health_census.jsonl"
#: Classes the report prints, and the floor each one needs before a zero-fail count may be
#: called a flag. Chosen locally, not imported from anywhere: the video this came from gives
#: no definition of "healthy" (a check "passing 99.999% of the time" is the only number in
#: it, offered as an illustration and not a bar), so the choice is ours and is printed. A
#: gate rung runs on every round and can accumulate hundreds of samples; an eval check runs
#: on a cadence and may have a dozen, and refusing the flag below the floor is what stops
#: the census printing a confident "0 fails in 2" about a check that has never been offered
#: anything to fail on.
FLOORS: dict[str, int] = {"gate_rung": 20, "eval_check": 5, "nightly_arm": 5}
#: The only duration fields this census may add up. `seconds` is the rung's own wall-clock;
#: every other `*_s`/`*_seconds` field in a ledger `data` block is either double-counted
#: (queue wait for a concurrently-run rung, a nested retry sleep) or unattributable (no
#: `round_id`). The companion test proves a naive full sum is LARGER, which is the whole
#: reason the constant is a constant.
ADDABLE_FIELDS: tuple[str, ...] = ("seconds",)
NEVER_EXECUTED = "0/0 (never executed)"
BELOW_FLOOR = "below floor"
#: The other two categories `flag_token` can return. Bare tokens, because clause 5 keys the
#: injection requirement off the CATEGORY (`never_failed`) while clause 3 pins the PRINTED form
#: (`never_failed (0 fails in 2466)`), and a string that has to be both is a string that is
#: neither: the first version compared `c["flag"].startswith("never_failed")` at four call
#: sites, which silently stopped matching the moment the printed form changed.
NEVER_FAILED = "never_failed"
FAILING = "failing"
#: The fixed caller vocabulary for the eval tree. Deliberately closed: an open-ended
#: "how is it called?" field is how a census becomes prose.
CALLER_CLASSES: tuple[str, ...] = ("gate rung", "scheduled job", "nightly", "test",
                                   "no caller found")
#: The boundary between "standing check" and "one-off probe", printed BEFORE any health
#: figure and derived from the classification rather than hand-asserted per file. A file
#: whose only caller is itself is a measurement somebody ran once; listing it as
#: never-failed would be noise, which is the risk this item names as its own scope hazard.
STANDING_RULE = (
    "STANDING CHECK if the caller census finds at least one caller OUTSIDE the file itself — "
    "a rung name the ledger actually carries, an autonomy task in the vault naming it, a "
    "nightly baseline written under its name, or a test node that imports it. ONE-OFF PROBE if "
    "nothing outside the file names it: the only hits a search returns are the file's own "
    "bytes. A test-only caller stands only while the suite runs, which on this box means while "
    "a gate runs, and the row says which class it is in. Health figures are computed only from "
    "the execution record (the promotions ledger's gate rows), so a one-off probe can never be "
    "printed as never-failed — the boundary is drawn and printed before any figure, because a "
    "throwaway measured as always-green is the noise this item names as its own scope hazard."
)
#: Who re-runs this. A `--record` flag with no caller is how #2378's landed outcomes module
#: came to have a journal that did not exist on this box, so the cadence is stated here and
#: the caller half is owed on the item (an autonomy task + skill, landed after the code).
CADENCE = "weekly (autonomy task `check-health-census-weekly`; owed, see item #3418)"

GATE_DIR = "scripts/automod/gate.py"
DEFAULT_SEARCH_DIRS: tuple[str, ...] = ("scripts", "app", "agent-services", "workers",
                                        "eval", "tests", "app/harness")
VAULT_AUTONOMY_GLOB = "autonomy/*.md"


# ---------------------------------------------------------------- ledger reading
def state_dir(default: Optional[Path] = None) -> Path:
    """The automod state dir, resolved at call time so a fixture can point elsewhere."""
    if default is not None:
        return Path(default)
    from scripts.automod import state as S
    return S.STATE_DIR


def ledger_files(dirn: Path) -> list[Path]:
    """The live ledger, then every monthly archive beside it, oldest month first.

    The glob is the whole point, and it is the same reason `gate_to_land_lag.py:130-140`
    globs: a census that read only `promotions.jsonl` would print `0/0 (never executed)` for
    `venv`, which has 0 live rows and 554 archived ones. A missing live file is tolerated —
    the sweep can roll its last rows out — and the reader then sees no rows rather than a
    traceback.
    """
    dirn = Path(dirn)
    return [dirn / LIVE_LEDGER, *sorted(dirn.glob(ARCHIVE_GLOB))]


def iter_ledger_rows(files: list[Path]) -> Iterator[tuple[Optional[dict], bool]]:
    """Yield `(row, from_archive)` per decodable ledger line, streaming.

    Streaming because the live ledger is ~32 MB and the archives are gzip; holding decoded
    rows costs more than these figures need. `row` is None for a line that is not a JSON
    object — a truncated final line is normal in an appended JSONL and must not raise, and
    the caller counts those out loud instead of letting a partial read look complete.
    """
    for path in files:
        archived = path.name != LIVE_LEDGER
        if not path.exists():
            continue
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    yield None, archived
                    continue
                yield (row if isinstance(row, dict) else None), archived


def gate_rows(dirn: Path) -> tuple[list[dict], dict[str, Any]]:
    """Every `event: "gate"` row in the window, plus the population it was drawn from.

    `population` travels with the rows because a count alone cannot tell a reader whether
    the archives were there: `files` names each file read, `undecodable` counts the lines
    that were not JSON objects, `archived_rows` separates the two sources.
    """
    files = ledger_files(dirn)
    rows: list[dict] = []
    undecodable = 0
    archived_rows = 0
    for row, was_archived in iter_ledger_rows(files):
        if row is None:
            # A line that is not a JSON object is undecodable wherever it sits, and the
            # reader says so instead of dropping it.
            undecodable += 1
            continue
        if row.get("event") != "gate":
            continue
        if was_archived:
            archived_rows += 1
            # Tagged, not counted here: clause 2's claim is about ONE rung (`venv`: 0 live
            # rows, 554 archived), so the per-check rollup needs to know which of ITS rows
            # came out of a gzip file. An underscore key marks it as reader-added, and it
            # never reaches the journal because the row is not this dict.
            row = {**row, "_from_archive": True}
        rows.append(row)
    return rows, {"files": [f.name for f in files if f.exists()],
                  "gate_rows": len(rows), "archived_rows": archived_rows,
                  "undecodable": undecodable}


# ---------------------------------------------------------------- per-check stats
def addable_seconds(row: dict) -> float:
    """Sum ONLY `ADDABLE_FIELDS` out of one rung row.

    A field is read at the row's TOP level first and from its `data` block second, because
    that is where the ledger actually puts it: `gate.py:2104` appends
    `{"event": "gate", "rung": name, "ok": ok, "seconds": seconds}` with the duration on the
    row, not inside `data`. A reader that looked only in `data` would return 0.0 for every
    rung on the box and the cost half of this census would be a column of clean zeros."""
    # The `data` block is only a source when it is a dict: `gate.py` writes it as `{}` for most
    # rungs, and a `None` there (an older row, or an event that carries no data at all) must read
    # as "no fields here", not as a TypeError on `in`. Written as a branch rather than a ternary
    # so the checker can see both sources are dicts.
    sources: list[dict] = [row]
    data = row.get("data")
    if isinstance(data, dict):
        sources.append(data)
    total = 0.0
    for key in ADDABLE_FIELDS:
        for source in sources:
            if key not in source:
                continue
            try:
                total += float(source.get(key) or 0.0)
            except (TypeError, ValueError):
                continue
            break
    return total


def naive_seconds(row: dict) -> float:
    """EVERY duration-shaped numeric field on the row and in its `data`, for the overcount test.

    Not used by the report. It exists so a test can prove `ADDABLE_FIELDS` excludes a real
    overcount rather than being a stylistic preference, so it has to be the sum a careless reader
    would actually write: the rung's own `seconds` PLUS the nested waits. Matching only the
    `_s`/`_seconds` suffix misses the plain `seconds` key and comes out SMALLER than the addable
    sum, which inverts the very relationship the test pins — 210 against 900 on the fixture — and
    a test that passes on an inverted instrument proves nothing.
    """
    fields: dict[str, object] = dict(row)
    if isinstance(row.get("data"), dict):
        fields.update(row["data"])
    total = 0.0
    for key, value in fields.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if key in ("seconds", "sec") or key.endswith(("_s", "_seconds", "_sec")):
            total += float(value)
    return total


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def check_stats(name: str, class_name: str, rows: list[dict]) -> dict[str, Any]:
    """The per-check figures. n is `len(rows)`, so a check with no rows has n 0.

    `skipped` is a rung row whose `seconds` is 0.0: for `venv` that is all 554 archived rows
    (`gate.py:2030` records 0.0 when requirements are unchanged), and printing those as 554
    healthy passes is the same mistake as a `skipped: true` rung counting as a real pass —
    #1212's landed finding. Hence `zero_second_share` on EVERY row rather than an exception
    list for the one rung known to need it.
    """
    stamps = [float(r["ts"]) for r in rows if isinstance(r.get("ts"), (int, float))]
    passes = sum(1 for r in rows if r.get("ok") is True)
    fails = sum(1 for r in rows if r.get("ok") is False)
    durations = [float(r.get("seconds") or 0.0) for r in rows]
    zero_second = sum(1 for d in durations if d == 0.0)
    archived_n = sum(1 for r in rows if r.get("_from_archive"))
    last_fail = max([float(r["ts"]) for r in rows
                     if r.get("ok") is False and isinstance(r.get("ts"), (int, float))],
                    default=None)
    return {
        "name": name,
        "class": class_name,
        "n": len(rows),
        "passes": passes,
        "fails": fails,
        "other_ok_states": len(rows) - passes - fails,
        "zero_second": zero_second,
        "zero_second_share": round(zero_second / len(rows), 4) if rows else None,
        # How many of THIS check's rows came out of a monthly archive rather than the live
        # ledger, which is what clause 2 is about: `venv` printing n=554 with archived_n=554
        # proves the union was read, where n alone could be either file's doing.
        "archived_n": archived_n,
        "mean_seconds": round(statistics.fmean(durations), 3) if durations else None,
        "median_seconds": round(statistics.median(durations), 3) if durations else None,
        "first_ts": min(stamps) if stamps else None,
        "last_ts": max(stamps) if stamps else None,
        "last_fail_ts": last_fail,
        "window_stamps": (_utc(min(stamps)), _utc(max(stamps))) if stamps else (None, None),
        # Computed once, here, from the class's own floor, and stored as a BARE TOKEN —
        # `never_failed`, `below floor`, `failing`, or `NEVER_EXECUTED`. `flag_of` below reads
        # this field and decorates it, so the flag in the printed line, the flag the injection
        # verdicts are keyed off, and the flag in the journaled row are one measurement: three
        # call sites deriving one string is how a row ends up printed `never_failed` and
        # journaled `below floor`. The token is bare so a caller can ASK which category a check
        # is in (`c["flag"] == NEVER_FAILED`) instead of pattern-matching a printed sentence
        # that carries an n in it.
        "flag": flag_token(len(rows), fails, FLOORS.get(class_name, 0)),
        "floor": FLOORS.get(class_name, 0),
    }


def flag_token(n: int, fails: int, floor: int) -> str:
    """The category a check's pass history falls into, as one of four bare tokens.

    `below floor` is a refusal, not a verdict: the item's thin-denominator risk is a confident
    "0 fails in 2" about a check that was never offered anything to fail on, so under the floor
    the census says it does not know.
    """
    if n == 0:
        return NEVER_EXECUTED
    if fails == 0:
        return BELOW_FLOOR if n < floor else NEVER_FAILED
    return FAILING


def flag_of(check: dict[str, Any], floor: int) -> str:
    """The category, printed — n and all. One derivation (`flag_token`, at the stats); this is
    its only formatter."""
    token = str(check.get("flag") or flag_token(int(check["n"]), int(check["fails"]), floor))
    if token == NEVER_EXECUTED:
        return NEVER_EXECUTED
    if token == BELOW_FLOOR:
        return f"{BELOW_FLOOR} (n={check['n']})"
    if token == NEVER_FAILED:
        return f"{NEVER_FAILED} (0 fails in {check['n']})"
    return f"{check['fails']} fail(s) in {check['n']}"


def days_since_last_fail(check: dict[str, Any], now_ts: float) -> Optional[float]:
    """Days since the last row whose `ok` was False, or None if there was never one.

    None is printed, never a 0 — a check that has never failed is not a check that failed
    today, and collapsing the two is how an age figure becomes a false claim.
    """
    if check["last_fail_ts"] is None:
        return None
    return round((now_ts - check["last_fail_ts"]) / 86400.0, 2)


def render_check_row(check: dict[str, Any], *, floor: int, now_ts: float,
                     verdict: Optional[str] = None) -> str:
    """One row: the pass/fail split over n, the age of the last real failure, the cost, and
    the share of rows that were skips. Every field here is also in the recorded row."""
    if check["n"] == 0:
        # Where the history would live, named on the row: a bare `0/0` about an eval arm reads
        # as "this script has never run anywhere" when all the census can say is that its
        # ledger has no rows for it — a job-driven arm's runs land in `workers.db`, not here.
        where = ("no execution row in the automod ledger, whose unit of execution is a gate "
                 "rung; a job-driven arm's runs are in workers.db"
                 if check["class"] != "gate_rung" else "no caller and no cadence found")
        return (f"{check['class']} {check['name']}: {NEVER_EXECUTED} floor={floor} "
                f"zero rows in the window; {where}")
    age = days_since_last_fail(check, now_ts)
    age_txt = "none (0 fails in window)" if age is None else f"{age}d"
    # The skip share is a COUNT and a percentage: `venv`'s 554 archived rows are all
    # `seconds: 0.0`, and printing them as 554 cheap healthy passes is the #2758 mistake. At
    # 100% the rung never ran at all, so the row says that rather than leaving the reader to
    # divide.
    skipped = int(check["zero_second"])
    pct = round(100.0 * skipped / check["n"])
    skip_txt = f"zero-second={skipped}/{check['n']} ({pct}%" + (
        ": rung never ran)" if pct == 100 else ")")
    first, last = check["window_stamps"]
    return (f"{check['class']} {check['name']}: {check['passes']} pass / {check['fails']} fail "
            f"over n={check['n']} [{flag_of(check, floor)}] "
            f"days_since_last_fail={age_txt} mean={check['mean_seconds']}s "
            f"median={check['median_seconds']}s {skip_txt} "
            f"window={first}..{last}"
            + (f" verdict={verdict}" if verdict else ""))


# ---------------------------------------------------------------- caller census
def eval_files(root: Path) -> list[Path]:
    """The eval tree's top level. Subpackages (`eval/action_equivalence/`, `eval/kvf/`) are
    libraries, not arms; a census of checks must not count a module two times."""
    return sorted(p for p in Path(root).glob("eval/*.py")
                  if p.name != "__init__.py")


#: The three spellings a caller can name an eval script by, and all three have to be searched
#: or the census invents orphans: `python eval/x.py` (a shell step), `python -m eval.x` /
#: `from eval.x import` (a dotted import), and `from eval import x` (a bare stem, which is how
#: `tests/test_alpha.py` names its subject and matches NO path-shaped needle at all). The first
#: version searched only `{stem}.py` and therefore called a test-covered arm an orphan — the
#: exact error direction this item exists to catch, since an orphan list is a deliverable and a
#: wrong one is worse than an empty one.
def _needles(path: Path) -> list[str]:
    """Every way a caller could name this file, and no more than that.

    Four shapes, all fixed strings: `eval/x.py` (a shell line, a systemd unit, a comment),
    `eval.x` (a Python import), and the file's path under `tests/` (the fixture's callers use
    the real convention — the tree's convention is a module path, `tests/test_alpha.py` names
    its subject and matches no path-shaped needle at all). The first version searched only
    `{stem}.py` and therefore called a test-covered arm an orphan, which is the error direction
    this item exists to catch: an orphan list is a deliverable, and a wrong one is worse than an
    empty one.

    The fifth shape — the BARE stem, which is how an autonomy task's prose names a script it
    runs ("run `run_djev_name_prior_probe` weekly") — is included only when the stem contains an
    underscore. Measured on the live tree with the bare stem always included: `eval/stats.py`
    came back called by 136 files, among them `agent-services/bin/start-decider.sh` and three
    unrelated autonomy tasks, because the word "stats" appears in all of them. A needle that
    matches too much manufactures callers, which is #2609's mistake — an 8-char prefix loose
    enough to match every other candidate — in a new costume, and here it would have quietly
    converted orphans into scheduled jobs. `stats`, `memory` and `alpha` are words;
    `djev_name_prior_probe` and `prefetch_calibration` are module names.
    """
    stem = path.stem
    needles = [f"eval/{path.name}", f"eval.{stem}",
               f"tests/eval/{path.name}", f"tests/eval.{stem}",
               f"tests/{path.name}", f"tests/{stem}.py"]
    if "_" in stem:
        needles.append(stem)
    return needles


def _grep_hit(root: Path, needles: list[str], dirs: list[str]) -> list[str]:
    """`git grep -lF -e <each needle> -- <dirs>`, returning the matched paths (possibly []).

    Fixed strings, always (`-F`). Without it the dotted needle `eval.stats` is a PATTERN whose
    `.` matches any character, and it fired on the prose "eval stats" — which classified a
    file with no caller at all as a scheduled job. Fixed strings ask the question actually
    being asked: does this string appear.

    Only TRACKED files match, since this is `git grep` and not `rg`: an untracked file in the
    tree is not part of the system anyone can run, so the exclusion is correct for the live
    tree and the fixture has to `git add` its callers for the same reason.

    [] means both "no matches" and "the grep failed". A census that reported 'unknown' for a
    failed grep would be more honest and useless; what a failure costs here is one file
    classified `no caller found`, the same print a genuinely uncalled file gets, and the
    evidence line prints the exact command so a reader can re-run it.
    """
    if not root.is_dir() or not needles:
        return []
    present = [d for d in dirs if (root / d).is_dir()]
    if not present:
        return []
    try:
        proc = subprocess.run(
            ["git", "grep", "-lF", *[a for n in needles for a in ("-e", n)], "--", *present],
            cwd=str(root), capture_output=True, text=True, timeout=120.0)
    except (OSError, subprocess.SubprocessError):
        return []
    return sorted({ln.strip() for ln in proc.stdout.splitlines() if ln.strip()})


def _vault_named(vault: Path, needles: list[str]) -> list[str]:
    """Vault autonomy-task and skill files naming any needle, by fixed substring.

    The scheduled-job half of the vocabulary lives OUTSIDE the repo: `workers/queue.py` takes
    task definitions from `~/obsidian/autonomy/*.md`, not from `autonomy-tasks/` (a retired
    path whose only occupant is a README). A classifier that searched only the checkout would
    call every scheduled arm an orphan — the precise error direction #2378's `landed_outcomes`
    finding warns about. `rg` is not assumed present and the vault's two directories are small,
    so a bounded walk is 8 ms over the live vault and works in a fixture; `skills/` is included
    because a skill's `command:` line is exactly as much a cadence as a task body is.
    """
    hits: list[str] = []
    for sub in ("autonomy", "skills"):
        d = vault / sub
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.suffix.lower() not in {".md", ".py", ".sh", ".yaml", ".yml", ".json"}:
                continue
            try:
                txt = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if any(n in txt for n in needles):
                hits.append(str(p.relative_to(vault)))
    return hits


def nightly_artifacts(baseline_dir: Path) -> list[str]:
    """The `EVAL_BASELINES_DIR/nightly-*.json` files actually on disk, oldest first.

    The nightly class is keyed on artifacts and not on a hand-list: `workers/queue.py` has zero
    hits for `nightly` and no reflection job exists in `workers/sources/`, so a nightly arm named
    by a constant would be a claim about a scheduler that does not exist.
    """
    return sorted(p.name for p in Path(baseline_dir).glob("nightly-*.json"))


def classify_eval_file(path: Path, *, root: Path, vault: Path, rung_names: set[str],
                       nightly: list[str], baseline_dir: Path,
                       search_dirs: list[str]) -> dict[str, Any]:
    """One file, one class from `CALLER_CLASSES`, beside the command that established it.

    The class is decided by WHICH DIRECTORY the hit is in, because "who calls it" and "what
    cadence re-runs it" are the same question here: a hit in `agent-services/`, `workers/`,
    `scripts/` or `app/` is a job or a service (scheduled job), a hit in `scripts/automod/gate.py`
    is a rung, a hit only in `tests/` is a test node, and a hit only inside `eval/` is a
    cross-reference — which is reported as such rather than dropped, because one eval script
    importing another is not a cadence and silently filtering those hits used to make the
    evidence line lie about a search that had matched.

    Precedence is stated because two classes can both be true of one file: gate rung first (the
    cadence that runs it on every round beats anything rarer), then scheduled job, then nightly,
    then test. `test` is LAST on purpose: it is the weakest standing evidence on this box, where
    a test node runs only while a gate runs.

    Every `nightly` / `vault` / `search_dirs` input is a PARAMETER and not an import-time
    constant, so a caller (including a test) can prove the dirs are consulted per call: an
    import-time snapshot would make the classification of a fixture depend on the live box.
    """
    stem = path.stem
    needles = _needles(path)
    cmd = (f"git grep -lF {' '.join(f'-e {n!r}' for n in needles)} -- "
           f"{' '.join(search_dirs)}")
    hits = _grep_hit(root, needles, search_dirs)
    gate_hits = [h for h in hits if h == GATE_DIR]
    code_hits = [h for h in hits
                 if not h.startswith(("tests/", "eval/")) and h != GATE_DIR]
    test_hits = [h for h in hits if h.startswith("tests/")]
    eval_only = [h for h in hits if h.startswith("eval/")]
    vault_hits = _vault_named(vault, needles)
    baselines = sorted(p.name for p in Path(baseline_dir).glob(f"{stem}*")) if baseline_dir else []

    evidence: list[str] = [f"{cmd} -> {', '.join(hits[:3]) if hits else 'nothing'}"]
    if vault_hits:
        evidence.append(f"{VAULT_AUTONOMY_GLOB} + skills/** naming it -> "
                        f"{', '.join(vault_hits[:3])}")
    if baselines:
        evidence.append(f"ls {Path(baseline_dir).name}/{stem}* -> {', '.join(baselines[:2])}")

    if stem in rung_names or gate_hits:
        cls = "gate rung"
        evidence.append(f"named by {GATE_DIR}"
                        + (f" and carries ledger rows as rung {stem!r}"
                           if stem in rung_names else ""))
    elif code_hits or vault_hits:
        cls = "scheduled job"
    elif baselines:
        cls = "nightly"
        if not nightly:
            evidence.append("no nightly-*.json artifact exists in the baselines dir, so the "
                            "cadence rests on the named artifact alone")
    elif test_hits:
        cls = "test"
    else:
        cls = "no caller found"
        if eval_only:
            evidence.append(f"hits were only inside eval/ ({', '.join(eval_only[:3])}): an eval "
                            "script importing another is a cross-reference, not a cadence")
    return {"file": str(path.relative_to(root)), "name": stem, "class": cls,
            #: STANDING per the printed rule: something outside the file itself re-runs it.
            #: `test` counts as standing but is the weakest kind, which is what the rule text
            #: says out loud rather than hiding in a footnote.
            "standing": cls != "no caller found", "evidence": evidence, "hits": len(hits)}


def classify_eval_tree(root: Path, *, vault: Path, rung_names: set[str],
                       baseline_dir: Path, search_dirs: list[str]) -> list[dict[str, Any]]:
    nightly = nightly_artifacts(baseline_dir)
    return [classify_eval_file(p, root=root, vault=vault, rung_names=rung_names,
                               nightly=nightly, baseline_dir=baseline_dir,
                               search_dirs=search_dirs)
            for p in eval_files(root)]


# ---------------------------------------------------------------- injection harness
class Injection:
    """One seeded fault a check OUGHT to catch, and the probe that tries it.

    `probe()` returns True when the check caught the fault. The signature is a callable
    rather than a config blob because an injection that cannot actually run is not a verdict,
    and a table of strings would be exactly the thing this harness exists to avoid.
    """

    #: `class_name` defaults to `gate_rung` because that is what every shipped probe targets and
    #: what the floor lookup needs; a caller registering a probe for an eval arm passes it. It
    #: is never inferred from the ledger — the check's own record supplies the floor at
    #: `injection_verdicts`, and this field is only ever the fallback for a check that is not in
    #: the census at all (the blinded control).
    def __init__(self, name: str, fault: str, probe, *, class_name: str = "gate_rung",
                 blind: bool = False) -> None:
        self.name = name
        self.class_name = class_name
        self.fault = fault
        self.probe = probe
        #: The blinded control: a probe deliberately given none of the information the check
        #: needs, which therefore MUST come out `vacuous`. Its purpose is to prove the
        #: harness distinguishes "caught" from "was never told", and a harness that printed
        #: `alive` for it would make every other verdict here worthless.
        self.blind = blind


def _ctx(src: str, rel: str = "scripts/maintenance/probe_subject.py"):
    """A `code_rules._FileCtx` for a source string — the production rule input, constructed
    directly so the injection does not need a worktree."""
    from scripts.automod import code_rules as CR
    return CR._FileCtx(root=REPO_ROOT, rel=rel, src=src, lines=src.splitlines(),
                       tree_ast=ast.parse(src), siblings=frozenset())


#: The fault and its benign twin, differing ONLY in the timezone argument. The shape is not
#: free choice: the rule's matcher is `_is_path_expr` (`code_rules.py:453-461`), which fires
#: only on an expression that BUILDS a path (`Path(...)`, `base / name`, `os.path.join(...)`),
#: so a bare `name = datetime.now()...` assignment is correctly not its business, and a probe
#: written that way would report the rung vacuous when it had merely asked the wrong question.
NAIVE_NOW_BAD = ("from datetime import datetime\nfrom pathlib import Path\n"
                 "out_path = Path('run-' + datetime.now().strftime('%F') + '.json')\n")
NAIVE_NOW_BENIGN = ("from datetime import datetime, timezone\nfrom pathlib import Path\n"
                    "out_path = Path('run-' + datetime.now(timezone.utc)"
                    ".strftime('%F') + '.json')\n")


def _inject_code_rules_naive_now() -> bool:
    """Fault: a no-arg `datetime.now()` naming an artifact file — what `code_rules`'s
    `naive_now_in_artifact_name` (`code_rules.py:474-486`) exists to stop. The PRODUCTION
    detector is called, not a reimplementation, and BOTH halves are demanded: it fires on the
    fault AND stays silent on the same expression written with `timezone.utc`. A probe asking
    only the first half would pass on a rule that flags every `Path` call whatsoever."""
    from scripts.automod import code_rules as CR
    return (bool(CR.naive_now_in_artifact_name(_ctx(NAIVE_NOW_BAD)))
            and not CR.naive_now_in_artifact_name(_ctx(NAIVE_NOW_BENIGN)))


def _inject_static_does_not_compile() -> bool:
    """Fault: a file that does not compile. Half of the `static` rung only — the rung also
    runs pyflakes and the census cannot seed a worktree, so the fault string says which half
    was tried rather than claiming the rung."""
    try:
        compile("def f(:\n  pass\n", "probe_subject.py", "exec")
    except SyntaxError:
        return True
    return False


def _inject_vet_empties_a_file() -> bool:
    """Fault: a change set that empties a tracked file — one of `vet`'s three structural
    violations (`vet.py:92 EMPTY_FILE`), and the one a real round has hit before. The
    PRODUCTION entry point runs (`vet.vet_change_set(base, worktree)`), over a throwaway git
    repo this function builds and deletes: `vet_change_set` takes any rev plus any worktree, so
    the census never needs to seed the live tree, and a violation seeded anywhere is the same
    detector deciding. BOTH halves are demanded, exactly as the `code_rules` probe demands
    its benign twin: the emptying commit must be flagged and the untouched repo must not be.
    """
    from scripts.automod import vet as V
    with tempfile.TemporaryDirectory(prefix="census-vet-") as tmp:
        repo = Path(tmp)
        (repo / "victim.py").write_text("print('content')\n", encoding="utf-8")
        for args in (["init", "-q"], ["config", "user.email", "c@localhost.invalid"],
                     ["config", "user.name", "census"], ["add", "-A"],
                     ["commit", "-q", "-m", "seed"]):
            subprocess.run(["git", *args], cwd=str(repo), check=True, capture_output=True, text=True)
        base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo), capture_output=True,
                      text=True).stdout.strip()
        clean = V.vet_change_set(base, repo)
        (repo / "victim.py").write_text("", encoding="utf-8")
        subprocess.run(["git", "commit", "-q", "-am", "empty it"], cwd=str(repo), check=True,
               capture_output=True, text=True)
        seeded = V.vet_change_set(base, repo)
    kinds = {v.kind for v in seeded.violations}
    return V.EMPTY_FILE in kinds and not clean.violations


def _inject_blinded_rung_reader() -> bool:
    """The blinded control, which MUST come out `vacuous`.

    `flag_of` is handed stats built from rows stripped of everything but a name — no `ok`, no
    `ts`, no `seconds` — so it cannot tell a check that failed from one that never did, and it
    reports `never_failed` anyway. That is the failure mode one level up from a vacuous check:
    a harness whose probes could not distinguish "caught the fault" from "was never told"
    would print `alive` for all fifteen rungs and look like a clean bill of health. The
    control is what makes the other verdicts here worth reading."""
    stripped = check_stats("BLINDED-CONTROL", "gate_rung", [{"name": "x"}] * 30)
    return not flag_of(stripped, FLOORS["gate_rung"]).startswith("never_failed")


def default_injections() -> list[Injection]:
    """The checks this module can try something on, and the fault it tries on each.

    Which checks are OWED a verdict is not decided here: that comes from the flags in the
    census, which come from the ledger's own rung values, so a rung absent from this window is
    not quietly handed a verdict and one that appeared last week is not missing. What is
    hand-declared here is only WHICH FAULT would be a fair test of a given check, because that
    is a claim about the check's mechanism and cannot be read out of a ledger row. Checks with
    no entry get `not injectable (<reason>)` from `NOT_INJECTABLE`, printed rather than a
    verdict invented.
    """
    return [
        Injection("BLINDED-CONTROL",
                  "the census's own flag reader, told nothing but a row count",
                  _inject_blinded_rung_reader, blind=True),
        Injection("code_rules",
                  "a naive datetime.now() naming an artifact file, through the production "
                  "detector `code_rules.naive_now_in_artifact_name`",
                  _inject_code_rules_naive_now),
        Injection("static",
                  "a source file that does not compile (the compile half of the rung only; "
                  "the pyflakes half is not seeded from here)",
                  _inject_static_does_not_compile),
        Injection("vet",
                  "a change set that empties a tracked file, through the production "
                  "structural vet `vet.vet_change_set`",
                  _inject_vet_empties_a_file)]


#: Rungs the census cannot inject from here, with the reason printed rather than a verdict
#: invented. Seeding these needs a candidate worktree (a real round), which is a gate's job
#: and not a report's — and the alternative, calling a rung's internals with a fixture built
#: by the report's author, is a check asserting about itself.
#: Why a flagged check has no probe here. Only checks that genuinely cannot be probed from a
#: report belong in this dict: `vet` used to claim it needed a seeded candidate diff and was
#: wrong — `vet.vet_change_set(base, worktree)` takes any rev and any tree, so a throwaway repo
#: is enough, and a rung left listed here as un-probeable is a rung nobody will ever test. A
#: flagged check that is injectable but has no probe registered yet prints
#: `not injectable (no probe registered)` — a hole, printed as a hole.
NOT_INJECTABLE: dict[str, str] = {
    "pyright": "needs a seeded worktree with a type error and the pyright binary",
    "drill": "needs a rollback-path change plus a guardian restart",
    "prompt_surface": "needs a seeded prompt-surface edit",
    "venv": "needs a requirements change to decide the skip; every archived row is seconds 0.0",
    "preflight": "needs an empty round or a non-descendant candidate",
    # #3538: observe-only, and it records a list of files rather than a
    # verdict, so an injected failure here would only re-prove `_rung`
    # reports one — which tests/test_automod_gate.py pins directly.
    "prose_consumers": "observe-only: a hit list, never a verdict",
    # These four fail in every window the ledger holds, so they carry no zero-fail flag to
    # prove. The reasons are stated without counts on purpose: the ledger grows daily and a
    # figure baked into this dict would be a claim the report keeps printing after it stops
    # being true.
    "tests": "fails in the window, so there is no zero-fail flag to prove",
    "review": "fails in the window; it is also gated on LLM access",
    "canary_smoke": "fails in the window; it needs a live engine turn",
    "canary_boot": "fails in the window; it boots an engine",
}


def injection_verdicts(checks: list[dict[str, Any]],
                       injections: list[Injection]) -> list[dict[str, Any]]:
    """Run each probe once, and record what came back.

    Only a check whose flag token is `never_failed` needs a verdict (clause 5's own words:
    "every zero-fail check **at or above its floor**" — `below floor` carries a different
    token, so it is not owed one), but every probe that was TRIED is recorded whether or not it
    was owed, because the item's acceptance is "at least one check is demonstrated `vacuous` by
    a seeded fault the check does not catch, **or** the report prints `no vacuous check found`
    with the injections enumerated": a verdict set that only contains owed checks cannot prove
    a clean result was earned and not merely un-tried. `not injectable (<reason>)` prints for a
    flagged check with no probe rather than leaving the row green and bare.
    """
    by_name = {inj.name: inj for inj in injections}
    out: dict[str, dict[str, Any]] = {}
    for check in checks:
        name = check["name"]
        #: The check's OWN class floor from its own record, not a constant. The floor decides
        #: only whether a verdict is owed; a probe that exists is run regardless, so `tried`
        #: and `required` are separate fields and the render can print both.
        floor = int(check.get("floor") or FLOORS.get(check["class"], 0))
        inj = by_name.get(name)
        flagged = check.get("flag") == NEVER_FAILED
        if name == "BLINDED-CONTROL" and inj is None:
            # The control is a pseudo-check with no ledger record of its own, so without its
            # probe there is nothing here to report on it — and printing `not injectable` for it
            # would put a name in the owed-verdict set that no real check ever owes, which is
            # exactly how a report teaches a reader to ignore that column.
            continue
        if inj is None and not flagged:
            # Nothing owed and nothing tried: no line. A census that invented a verdict for a
            # check nobody flagged and nobody probed would itself be a vacuous check.
            continue
        required = flagged or name == "BLINDED-CONTROL"
        if inj is None:
            out[name] = {"check": name, "class": check["class"],
                         "verdict": "not injectable ("
                                    + NOT_INJECTABLE.get(name, "no probe registered") + ")",
                         "fault": None, "caught": None, "blind": False,
                         "tried": False, "required": flagged}
            continue
        try:
            caught = bool(inj.probe())
        except Exception as exc:      # noqa: BLE001 — a probe that errors is reported, not swallowed
            out[name] = {"check": name, "class": check["class"],
                         "verdict": f"not injectable (probe raised {type(exc).__name__}: {exc})",
                         "fault": inj.fault, "caught": None, "blind": inj.blind,
                         "tried": True, "required": required}
            continue
        out[name] = {"check": name, "class": check["class"],
                     "verdict": (f"alive ({inj.fault} caught)" if caught
                                 else f"vacuous ({inj.fault} not caught)"),
                     "fault": inj.fault, "caught": caught, "blind": inj.blind,
                     "tried": True, "required": required, "floor": floor}
    # A list, sorted by check name: the journaled row and the printed lines are read by a
    # person comparing one run to the next, and a dict's order would make that a diff of
    # hash luck rather than of verdicts.
    return [out[name] for name in sorted(out)]


# ---------------------------------------------------------------- census assembly
def build_census(*, dirn: Path, root: Path, vault: Path, baseline_dir: Path,
                 now_ts: Optional[float] = None,
                 search_dirs: list[str] | None = None,
                 injections: list[Injection] | None = None) -> dict[str, Any]:
    """Assemble the whole report: rung rows from the ledger, eval rows from the tree, the
    standing rule printed first, and an injection verdict for every flagged check."""
    now_ts = now_ts if now_ts is not None else datetime.now(timezone.utc).timestamp()
    search_dirs = search_dirs or list(DEFAULT_SEARCH_DIRS)
    rows, population = gate_rows(dirn)
    rung_names = sorted({str(r["rung"]) for r in rows if r.get("rung")})
    by_rung: dict[str, list[dict]] = {name: [] for name in rung_names}
    for row in rows:
        name = str(row.get("rung"))
        by_rung.setdefault(name, []).append(row)
    # One row per rung name the ledger named, PLUS one per named eval arm. An eval arm has NO
    # execution record in this ledger by construction — `gate_rows` keeps only `event:"gate"`
    # rows, and the ledger's unit of execution is a rung, not a script — so every arm prints the
    # `0/0 (never executed)` clause 3(a) demands rather than being dropped: that print is the
    # finding, and it is the `landed_outcomes.py` shape (#2378: a module with a `--record` flag
    # and no caller, whose journal has never existed on this box). Where an arm's real history
    # WOULD live — `runs.status`/`runs.duration_seconds` in `workers.db` for a scheduled or
    # nightly arm — is named on the row so nobody reads the 0 as "it never ran anywhere".
    checks = [check_stats(name, "gate_rung", by_rung[name]) for name in rung_names]
    eval_arms = sorted(p.stem for p in eval_files(root))
    arm_stats = {arm: check_stats(arm, "eval_check", [])
                 for arm in eval_arms if arm not in rung_names}
    checks += list(arm_stats.values())
    evals = classify_eval_tree(root, vault=vault, rung_names=set(rung_names),
                               baseline_dir=baseline_dir, search_dirs=search_dirs)
    rung_n = {c["name"]: int(c["n"]) for c in checks}
    for entry in evals:
        #: How many execution rows this file's own arm has, so the classification line and the
        #: health line cannot disagree about whether anything ran.
        stem = Path(entry["file"]).stem
        entry["window_rows"] = rung_n.get(stem, int((arm_stats.get(stem) or {}).get("n") or 0))
    # Every check row carries its own `flag` token (set in `check_stats`), so clause 5's journal
    # holds the verdict for exactly the checks the flag named, and a reader can see which those
    # were without re-deriving them from `floors`. An eval FILE carries no flag: its row answers
    # "who re-runs this", and execution history is keyed by rung/arm name, not by file path.
    inj = injections if injections is not None else default_injections()
    #: The blinded control rides as a pseudo-check so it goes through the SAME flagging and the
    #: SAME verdict machinery as a real rung: if the census ever stops flagging a zero-fail
    #: check, the control prints `not injectable` instead of `vacuous` and the report admits its
    #: own brokenness rather than reporting a clean check surface. It is not a rung and is never
    #: printed as one — it appears in `injections`, never in `checks`.
    control = {"name": "BLINDED-CONTROL", "class": "gate_rung", "n": 1, "passes": 1,
               "fails": 0, "flag": NEVER_FAILED, "floor": FLOORS["gate_rung"]}
    injections_out = injection_verdicts(checks + [control], inj)
    stamps = [c["first_ts"] for c in checks if c["first_ts"]] + \
             [c["last_ts"] for c in checks if c["last_ts"]]
    return {
        "kind": "check_health_census",
        "schema": 1,
        "generated_at": _utc(now_ts),
        # The window's own stamps and the files it was read out of, on the row: the print names
        # them and clause 5 requires every printed figure in the journal.
        "window": {"first_ts": _utc(min(stamps)) if stamps else None,
                   "last_ts": _utc(max(stamps)) if stamps else None,
                   "files": list(population["files"])},
        "files": list(population["files"]),
        "ledger": {"dir": str(dirn), "gate_rows": population["gate_rows"],
                   "archived_rows": population["archived_rows"],
                   "undecodable": population["undecodable"]},
        "population": population,
        "counts": {"gate_rungs": len(rung_names), "eval_arms": len(eval_arms),
                   "eval_files": len(evals),
                   "never_failed": sum(1 for c in checks if c["flag"] == NEVER_FAILED),
                   "below_floor": sum(1 for c in checks if c["flag"] == BELOW_FLOOR),
                   "never_executed": sum(1 for c in checks if c["flag"] == NEVER_EXECUTED),
                   "no_caller_found": sum(1 for e in evals
                                          if e["class"] == "no caller found")},
        "floors": dict(FLOORS),
        "addable_fields": list(ADDABLE_FIELDS),
        "cadence": CADENCE,
        "standing_rule": STANDING_RULE,
        "checks": checks,
        "eval_files": evals,
        "injections": injections_out,
    }


def render(census: dict[str, Any], *, now_ts: Optional[float] = None) -> list[str]:
    """The printable report, in the order the item asks for: the window and its population,
    the standing rule BEFORE any health figure, then every rung, then every eval file."""
    now_ts = now_ts if now_ts is not None else (
        datetime.strptime(census["generated_at"], "%Y-%m-%dT%H:%M:%SZ")
        .replace(tzinfo=timezone.utc).timestamp())
    win, pop = census["window"], census["population"]
    out = [
        f"check-health-census: generated {census['generated_at']} "
        f"window {win['first_ts']} .. {win['last_ts']} files={','.join(win['files']) or 'none'} "
        f"gate_rows={pop['gate_rows']} (archived {pop['archived_rows']}, "
        f"undecodable {pop['undecodable']})",
        f"floors: {', '.join(f'{k}>={v}' for k, v in sorted(census['floors'].items()))} "
        f"| cost = wall-clock from {list(census['addable_fields'])} only; no token or "
        "dollar field exists in this ledger, so nothing here is a spend",
        f"standing boundary: {census['standing_rule']}",
    ]
    #: `injections` is a list sorted by check name (the journaled shape), so the print looks a
    #: name up rather than carrying a second copy of the mapping.
    by_check = {str(v.get("check")): v for v in (census.get("injections") or [])}
    for check in census["checks"]:
        verdict = (by_check.get(check["name"]) or {}).get("verdict")
        out.append("  " + render_check_row(
            check, floor=int(check.get("floor") or FLOORS.get(check["class"], 0)),
            now_ts=now_ts, verdict=verdict))
    control = by_check.get("BLINDED-CONTROL") or {}
    out.append(f"  injection control BLINDED-CONTROL: {control.get('verdict', 'NOT RUN')}")
    vacuous = sorted(str(v["check"]) for v in (census.get("injections") or [])
                     if (v.get("verdict") or "").startswith("vacuous")
                     and v["check"] != "BLINDED-CONTROL")
    out.append("  VACUOUS checks (fault seeded, check did not catch it): "
               + (", ".join(vacuous) if vacuous
                  else "no vacuous check found — every seeded fault was caught"))
    out.append("  injections tried: " + ("; ".join(
        f"{v['check']}={v['verdict']}" for v in (census.get("injections") or [])
        if v.get("tried")) or "none"))
    out.append("  verdicts owed but not injectable: " + ("; ".join(
        f"{v['check']}={v['verdict']}" for v in (census.get("injections") or [])
        if not v.get("tried")) or "none"))
    out.append(f"eval tree: {len(census['eval_files'])} file(s), classified from "
               f"{', '.join(CALLER_CLASSES)}")
    for entry in census["eval_files"]:
        rows = int(entry.get("window_rows", 0))
        out.append(f"  {entry['class']:<16} {'STANDING' if entry['standing'] else 'one-off':<9}"
                   f" {entry['file']} — "
                   + (f"{rows} row(s) in the window" if rows
                      else "zero rows in the window"))
        for ev in entry["evidence"]:
            out.append(f"      evidence: {ev}")
    orphans = [e["file"] for e in census["eval_files"] if e["class"] == "no caller found"]
    out.append(f"no caller found: {len(orphans)} file(s)"
               + (": " + ", ".join(orphans) if orphans else " (list empty, printed anyway)"))
    return out


def census_path(state_dir: Optional[Path] = None) -> Path:
    """The journal: beside `landed_outcomes_path()`, in the automod state dir — never inside
    the checkout, where the guardian alerts on in-tree runtime data hourly (#1667).

    `state_dir` is the directory `--state-dir` resolved to, and honouring it is the
    whole of the #3777 fix. `main` used to resolve that directory for READING the
    ledger and then call `record(census)` with nothing, so the write fell through to
    the default resolution below: a `--record --state-dir /tmp/fixtures` run appended
    its row to the journal in the real state dir and left `/tmp/fixtures` empty. The
    row it wrote there is newest and real, but its `window` is the scratch fixtures —
    `first_ts: null`, `population.files: []` — so a reader that takes the last row of
    a live journal reads `never executed` over a history that exists. That is the
    finding this census exists to report about OTHER checks, printed about itself.
    """
    if state_dir is not None:
        return Path(state_dir) / JOURNAL_NAME
    from scripts.automod import state as S
    return S.STATE_DIR / JOURNAL_NAME


def record(row: dict[str, Any], path: Optional[Path] = None) -> Path:
    path = Path(path or census_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    return path


# ---------------------------------------------------------------- cli
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="check_health_census.py",
        description="Per-check pass history, cost and last real failure for every gate rung "
                    "and every named eval check, with an injection verdict for each "
                    "never-failed one. Report-only: retires nothing.")
    ap.add_argument("--state-dir", default=None,
                    help="automod state dir holding promotions.jsonl and its archives")
    # The vault root comes from the resolver, never from `Path.home() / "obsidian"`: this is
    # #1367, and the node that catches it is
    # `tests/test_no_runtime_paths_in_code.py::test_no_module_outside_the_resolver_computes_the_vault_root_from_home`
    # — which is exactly how the first version of this line got caught, by the gate's tests rung
    # rather than by a reader. A module that resolved the vault from `$HOME` also silently reads
    # the WRONG vault inside an automod round, whose `$HOME` is a round home whose `obsidian` is
    # a symlink; the caller census would then report zero autonomy callers for every arm.
    from app.paths import VAULT_ROOT
    ap.add_argument("--vault", default=str(VAULT_ROOT),
                    help="vault root, for the autonomy/*.md caller census "
                         "(default: app.paths.VAULT_ROOT)")
    ap.add_argument("--baselines", default=None,
                    help="nightly baseline dir; default app.paths.EVAL_BASELINES_DIR")
    ap.add_argument("--repo-root", default=None,
                    help="tree to census; default this checkout. A fixture points it at a "
                         "throwaway git tree so the caller census can be tested hermetically")
    ap.add_argument("--record", action="store_true",
                    help="append one dated row to check_health_census.jsonl in the "
                         "state dir just read (default: beside landed_outcomes_path())")
    ap.add_argument("--json", action="store_true", help="print the row as JSON, write nothing")
    args = ap.parse_args(argv)

    dirn = state_dir(Path(args.state_dir) if args.state_dir else None)
    baseline_dir = Path(args.baselines) if args.baselines else None
    if baseline_dir is None:
        from app.paths import EVAL_BASELINES_DIR
        baseline_dir = EVAL_BASELINES_DIR
    try:
        census = build_census(dirn=dirn,
                              root=Path(args.repo_root) if args.repo_root else REPO_ROOT,
                              vault=Path(args.vault), baseline_dir=baseline_dir)
    except Exception as exc:  # noqa: BLE001 — a census that cannot run must say so and exit 2
        print(f"check-health-census: cannot build ({type(exc).__name__}: {exc})",
              file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(census, sort_keys=True, default=str))
        return 0
    for line in render(census):
        print(line)
    if args.record:
        # The row goes into the directory this run was pointed at, exactly as the
        # ledger it describes was read from there — the same shape
        # `gate_to_land_lag.py:382` has always had (`record(row, record_path(state_dir))`).
        # `record(census)` with nothing was the #3777 leak: it journaled a
        # fixture-windowed row into the live journal.
        path = record(census, census_path(dirn))
        print(f"recorded {census['generated_at']} -> {path} "
              f"(check_health_census.jsonl; cadence {CADENCE})")
    print(f"report-only: nothing here retires, disables or weakens a check; disposition of "
          f"any row is Alan's call ({CADENCE})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
