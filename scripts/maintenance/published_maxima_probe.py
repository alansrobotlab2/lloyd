#!/usr/bin/env python3
"""Bound the silent rot in the published paired_check maxima (#3037, #3304).

What it re-derives
------------------
The era-D paired-check population: every `regression_check` ledger row since
era D began (`2026-09-22T04:08:33Z`), restricted to the `agent-qmd-daemon.conf`
pin, over the live `promotions.jsonl` **and every `promotions-archive-*.jsonl.gz`**.
The filter is copied verbatim out of `workers/sources/automod_regression.py`'s
own re-check block (`:319-330`), so the numbers this script prints and publishes
against are the numbers the provenance comment claims. On 2026-10-25 that is
1,204 rows — 471 in the live file, 733 in the two archives — and the live file
alone reads under 40 % of the population, so a reader who forgets the archives
understates the maxima exactly the way #3037 found.

What it appends
---------------
EXACTLY ONE row per invocation, dated with its own UTC stamp, to
`<production data root>/_pipeline/automod/published-maxima.jsonl` (the item's
declared home; the sibling recorder `scripts/maintenance/gate_to_land_lag.py`
writes beside the ledger instead, under `STATE_DIR` — this one keeps the journal
with the other maintenance records, and says so here rather than in a footnote).
The directory is created if absent; it did not exist before this script did, and
a `FileNotFoundError` on a first run reads as a broken probe. The keys are fixed:
`ts, n, fresh_median, fresh_max, graded_median, graded_max, over_bound_count,
published_fresh, published_graded, budget`. Nothing else is ever written.

What makes it non-zero
----------------------
Two thresholds, both declared in `evaluate()` and both visible in the message:

  1. `over_bound_count > 0` — a recorded era-D run over the armed budget.
  2. A published paired_check constant strictly below its arm's ledger max.
     NO MARGIN. #3304 asked for 25 %, and 25 % is the bug: at publication
     `905.3` sits 21.97 % under the ledger's 1,035.3 and `401.3` 16.02 % under
     477.9, so a 25 % rule reads GREEN on 283 ms of real drift — the exact
     silent gap this script exists to close. The comparison is strict equality
     upward: a constant equal to its max passes, one milli-ms under it does not.

Why it is OFF the suite and OFF the gate
----------------------------------------
`tests/conftest.py:93` points `LLOYD_AUTOMOD_STATE` at a scratch directory for
every pytest run, so a ledger-reading test is vacuously green inside a gate round
and red as a rail somewhere else — ruled at #3037's owed entry 2 (2026-10-25),
which is also why the constants stay a dated publication rather than becoming a
re-deriving assertion (#2145's intent). The live guard on the promotion path is
`over_budget()` in `workers/sources/automod_regression.py` (def :421, called
:1200 and :2282); this script's exit code is read by nobody but whoever runs it,
and it must stay that way. `tests/test_automod_regression.py` pins that nothing
under `scripts/automod/` imports it or reads its status.
"""

from __future__ import annotations

import gzip
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Appended, not inserted: `app/` holds top-level modules that would shadow a
# package of the same name for anything else in this process (same rule as
# `scripts/maintenance/gate_to_land_lag.py:65-69`). This script is invoked as a
# file, so without it neither `app` nor `scripts` is importable at all.
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

#: The era-D floor and the one-pin restriction, copied verbatim from
#: `workers/sources/automod_regression.py:322-328`. Widening either one here
#: silently changes which rows the published constants are checked against.
ERA_D_START = "2026-09-22T04:08:33Z"
PIN_MARK = "agent-qmd-daemon.conf"

#: The published constants this probe exists to bound. Values are read out of
#: `tests/test_automod_regression.py` (`PUBLISHED_MAXIMA_MS` and
#: `POST_WIDENING_MS`) at call time, so refreshing the publication cannot leave
#: this script checking a stale figure; these two literals are the fallback when
#: the test file cannot be read, and are deliberately the paired_check arm only.
JOURNAL_NAME = "published-maxima.jsonl"


def _read(path: Path) -> str:
    if path.suffix == ".gz":
        return gzip.decompress(path.read_bytes()).decode()
    return path.read_text(encoding="utf-8")


def ledger_files(state_dir: Path) -> list[Path]:
    """Live ledger first, then every archive. Same shape as gate_to_land_lag."""
    return [state_dir / "promotions.jsonl",
            *sorted(state_dir.glob("promotions-archive-*.jsonl.gz"))]


def era_d_readings(state_dir: Path) -> list[dict]:
    """The era-D paired-check population, over the live ledger AND its archives.

    A row contributes `created_at`, `commit` and its two `latency_ms_avg` arms.
    Rows whose `detail` is not the dict the promotion writer emits are skipped
    by the truthy `detail` clause, exactly as the module's command does.
    """
    out: list[dict] = []
    for path in ledger_files(state_dir):
        if not path.exists():
            continue
        for line in _read(path).splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("event") != "regression_check":
                continue
            if (row.get("created_at") or "") < ERA_D_START:
                continue
            pin = row.get("pin")
            if not isinstance(pin, dict) or PIN_MARK not in str(
                    (pin.get("daemon") or {}).get("source")):
                continue
            detail = row.get("detail")
            if not isinstance(detail, dict):
                continue
            arms = detail.get("latency_ms_avg")
            if not isinstance(arms, dict):
                continue
            out.append({"created_at": row.get("created_at"),
                        "commit": row.get("commit"),
                        "before": arms.get("before"),
                        "after": arms.get("after")})
    return out


def _values(rows: list[dict], arm: str) -> list[float]:
    return [r[arm] for r in rows if isinstance(r.get(arm), (int, float))]


def _nearest_rank_p95(values: list[float]) -> float:
    """p95 by NEAREST RANK, named because it is not the method that produced the
    2026-10-20 figures: over those same 1,035 rows nearest-rank gives 283.9 /
    681.9 and `statistics.quantiles` gives 283.6 / 681.6 inclusive, 284.2 /
    682.1 exclusive — none of them the published 283.4 / 681.5. A shape number
    whose method nobody can re-derive is how 611.9 became load-bearing prose
    without an owner, so the method travels with the figure here.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def _arm_summary(rows: list[dict], arm: str) -> dict:
    vals = _values(rows, arm)
    # One max, then the row that holds it: `max()` inside a comprehension over
    # 1,204 rows is 1.2M comparisons for a number printed once.
    peak_value = max(vals) if vals else None
    peak = next((r for r in rows if r.get(arm) == peak_value), {}) if vals else {}
    return {
        "median": round(statistics.median(vals), 1) if vals else 0.0,
        "p95": round(_nearest_rank_p95(vals), 1),
        "max": round(max(vals), 1) if vals else 0.0,
        "max_exact": round(max(vals), 3) if vals else 0.0,
        "max_created_at": peak.get("created_at"),
        "max_commit": (peak.get("commit") or "")[:9],
    }


def summarize(state_dir: Path) -> dict:
    """One measurement of the live population, including the budget it faces."""
    # The budget is read off the module that defines it — `LATENCY_BUDGET_MS` is
    # `workers/sources/automod_regression.py:395`, not `scripts/automod/state.py`,
    # and the first draft's `state.LATENCY_BUDGET_MS` died with AttributeError on
    # a real run. Same source the suite asserts against.
    from workers.sources import automod_regression as regression
    rows = era_d_readings(state_dir)
    fresh = _arm_summary(rows, "before")
    graded = _arm_summary(rows, "after")
    budget = float(regression.LATENCY_BUDGET_MS[regression.CONTEXT_PAIRED_CHECK])
    over = sum(1 for r in rows
               if isinstance(r.get("before"), (int, float))
               and isinstance(r.get("after"), (int, float))
               and max(r["before"], r["after"]) > budget)
    return {
        "state_dir": str(state_dir),
        "files_read": [p.name for p in ledger_files(state_dir) if p.exists()],
        "n": len(rows),
        # min/max of the stamp, NOT rows[0]/rows[-1]: rows come out in file order
        # (live ledger, then archives), so the first draft printed
        # `2026-10-11T05:17:04Z to 2026-10-11T04:35:31Z` — a span that runs
        # backwards and would outvote the module prose's own span claim.
        "span": ([min(r["created_at"] for r in rows),
                  max(r["created_at"] for r in rows)] if rows else []),
        "fresh_median": fresh["median"], "fresh_p95": fresh["p95"],
        "fresh_max": fresh["max"], "fresh_max_exact": fresh["max_exact"],
        "fresh_max_created_at": fresh["max_created_at"],
        "fresh_max_commit": fresh["max_commit"],
        "graded_median": graded["median"], "graded_p95": graded["p95"],
        "graded_max": graded["max"], "graded_max_exact": graded["max_exact"],
        "graded_max_created_at": graded["max_created_at"],
        "graded_max_commit": graded["max_commit"],
        "over_bound_count": over,
        "budget": budget,
    }


#: Which dict publishes which arm. The names live in the test file
#: (`tests/test_automod_regression.py:111` / `:136`) — `workers/sources/
#: automod_regression.py` carries these figures only as PROSE and an inline
#: comment (`git grep -c PUBLISHED_MAXIMA_MS -- workers/sources/…` → 0), so
#: pointing this at the module would be reading a comment as a constant.
PUBLISHED_DICTS = {"PUBLISHED_MAXIMA_MS": "fresh", "POST_WIDENING_MS": "graded"}


def published_constants(test_file: Path | None = None) -> dict:
    """The paired_check constants as the suite publishes them, read at call time.

    Parsed entry by entry, never the whole dict at once: the keys are the module's
    own constants (`R.CONTEXT_PAIRED_CHECK`), which are `ast.Attribute` nodes, so
    a `literal_eval` over the dict raises `ValueError: malformed node …
    <ast.Attribute object>` before it reaches any figure. Reading one figure at a
    time, with the key matched by its resolved name, is what works.

    There is NO fallback literal, deliberately. A number invented here that
    happens to equal the current truth (the first draft's 1160.2 / 477.9) makes a
    drift read GREEN, which is the failure this probe was built to close. Unreadable
    constants are an alarm in `evaluate`.
    """
    path = Path(test_file) if test_file else (
        REPO_ROOT / "tests" / "test_automod_regression.py")
    import ast
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        out: dict[str, object] = {}
        for node in tree.body:
            if not (isinstance(node, ast.Assign)
                    and isinstance(node.targets[0], ast.Name)):
                continue
            arm = PUBLISHED_DICTS.get(node.targets[0].id)
            if arm is None or not isinstance(node.value, ast.Dict):
                continue
            for key, value in zip(node.value.keys, node.value.values):
                # The keys are NOT string literals: the dicts are written
                # `{R.CONTEXT_NIGHTLY: …, R.CONTEXT_PAIRED_CHECK: 905.3}`
                # (tests/test_automod_regression.py:111-117, :136-139), so
                # `literal_eval(key)` raises on every entry — the first draft's
                # real failure. Match the resolved name instead, and `literal_eval`
                # only the figure.
                if not ast.unparse(key).endswith("CONTEXT_PAIRED_CHECK"):
                    continue
                out[arm] = float(ast.literal_eval(value))
        if set(out) != {"fresh", "graded"}:
            raise ValueError(f"only {sorted(out)} of fresh/graded resolved")
        out["source"] = str(path)
        return out
    except Exception as exc:                                    # noqa: BLE001
        return {"fresh": None, "graded": None,
                "source": f"UNREADABLE {path} ({type(exc).__name__}: {exc})"}


def evaluate(measured: dict, published: dict, budget: float) -> list[str]:
    """The declared thresholds, as messages. An empty list means the run is green.

    1. `over_bound_count > 0` — a recorded era-D paired_check run over the armed
       budget. (That is the guard the LIVE path already enforces through
       `over_budget()`; here it is a tripwire for the record having moved.)
    2. A published paired_check constant STRICTLY below its arm's ledger max.
       NO MARGIN. #3304 asked for 25 % and 25 % is the bug being refused: the
       publication sat at 905.3 against a ledger 1,160.230 — 21.97 % under — and
       401.3 against 477.877 — 16.02 % under — so a 25 % rule reads GREEN across
       254.9 ms and 76.6 ms of real drift, which is the silent gap this script
       exists to close. Equality passes: a constant exactly at its max is a
       current publication, not a stale one.
    3. Constants that cannot be read at all are an alarm, not a green: `None`
       here means `published_constants()` could not resolve the dict, and a probe
       that invents its own expected value has stopped measuring.
    """
    problems: list[str] = []
    if measured["over_bound_count"] > 0:
        problems.append(
            f"over_bound_count={measured['over_bound_count']} of n={measured['n']} "
            f"era-D paired_check rows exceed the armed budget {budget:g} ms")
    drifted: list[str] = []
    for arm_key, label in (("fresh", "fresh (before)"), ("graded", "graded (after)")):
        pub = published[arm_key]
        ledger_max = measured[f"{arm_key}_max"]
        if pub is None:
            problems.append(
                f"{label} arm: cannot read the published constant — {published['source']}"
                " — so the drift check could not run")
            continue
        if pub < ledger_max:
            drifted.append(
                f"{label}: published {pub:g} ms is BELOW the ledger max "
                f"{ledger_max:g} ms (exact {measured[f'{arm_key}_max_exact']:g}, at "
                f"{measured[f'{arm_key}_max_created_at']} commit "
                f"{measured[f'{arm_key}_max_commit']})")
    if drifted:
        # One line naming BOTH arms with BOTH numbers, whatever the caller greps:
        # a per-arm alarm alone lets a reader see half the drift and refresh half
        # the pair, which is how a set gets out of sync with its own stamp.
        problems.append(
            "published paired_check set is below the ledger: "
            f"fresh {published['fresh'] if published['fresh'] is not None else 'unreadable'}"
            f" vs {measured['fresh_max']:g} ms, "
            f"graded {published['graded'] if published['graded'] is not None else 'unreadable'}"
            f" vs {measured['graded_max']:g} ms "
            f"(n={measured['n']}, over_bound_count={measured['over_bound_count']}, "
            f"budget {budget:g} ms) — refresh the published constants beside a "
            "new stamp, or record a ruling not to")
        problems.extend(drifted)
    return problems


def journal_path(data_root: Path) -> Path:
    return Path(data_root) / "_pipeline" / "automod" / JOURNAL_NAME


def append_journal_row(root: Path, row: dict) -> Path:
    """Append exactly one line. The directory is created: it did not exist
    before this probe did, and a first-run FileNotFoundError reads as a broken
    probe rather than an unbuilt home."""
    path = journal_path(Path(root))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def build_row(measured: dict, published: dict, ts: str) -> dict:
    """The journal's fixed key set. Nothing may be added silently: the daily
    reader diffs these keys, so a new one is a schema change, not a bonus."""
    return {
        "ts": ts,
        "n": measured["n"],
        "fresh_median": measured["fresh_median"],
        "fresh_max": measured["fresh_max"],
        "graded_median": measured["graded_median"],
        "graded_max": measured["graded_max"],
        "over_bound_count": measured["over_bound_count"],
        "published_fresh": published["fresh"],
        "published_graded": published["graded"],
        "budget": measured["budget"],
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Default is the ledger directory `scripts/automod/state.py:46` uses when
    # `LLOYD_AUTOMOD_STATE` is unset — read here rather than through that module
    # because the whole point of running this off-suite is to see the REAL ledger,
    # and an inherited `LLOYD_AUTOMOD_STATE` pointing at a scratch copy would make
    # the probe measure nothing and print `n=0` green.
    state = Path.home() / ".local" / "state" / "lloyd-automod"
    if "--state-dir" in argv:
        state = Path(argv[argv.index("--state-dir") + 1])
    from app.data_root import production_data_root
    root = production_data_root()

    measured = summarize(state)
    published = published_constants()
    row = build_row(measured, published,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"))
    path = append_journal_row(root, row)

    print(f"era-D paired_check: n={measured['n']} over "
          f"{' + '.join(measured['files_read'])}")
    if measured["span"]:
        print(f"  span {measured['span'][0]} to {measured['span'][-1]}")
    else:
        print("  span: no era-D rows in this state dir")
    print(f"  fresh (before): median {measured['fresh_median']} "
          f"p95 {measured['fresh_p95']} max {measured['fresh_max']} "
          f"(exact {measured['fresh_max_exact']}, at "
          f"{measured['fresh_max_created_at']} commit "
          f"{measured['fresh_max_commit']})")
    print(f"  graded (after): median {measured['graded_median']} "
          f"p95 {measured['graded_p95']} max {measured['graded_max']} "
          f"(exact {measured['graded_max_exact']}, at "
          f"{measured['graded_max_created_at']} commit "
          f"{measured['graded_max_commit']})")
    print(f"  over {measured['budget']:g} ms: {measured['over_bound_count']} of "
          f"{measured['n']}")
    def _shown(value):
        return "UNREADABLE" if value is None else f"{value:g}"

    print(f"  published: fresh {_shown(published['fresh'])} / "
          f"graded {_shown(published['graded'])} "
          f"(read from {published['source']})")
    print(f"  journal: appended 1 row to {path}")

    problems = evaluate(measured, published, measured["budget"])
    for problem in problems:
        print(f"ALARM: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
