#!/usr/bin/env python3
"""Fact-quality improvement pass — the #376 feedback loop, runnable on its own.

Reads a real feedback signal (the user's corrections log and recent-write
drift), pairs contradictions, and — with --apply — expires the claims an
independent reason condemns. Dry-run by default: without --apply nothing is
written and the run says what it *would* have done.

    scripts/memory/fact-improvement.py                     # plan only
    scripts/memory/fact-improvement.py --apply             # act, capped
    scripts/memory/fact-improvement.py --apply --max-actions 10
    scripts/memory/fact-improvement.py --report-eval       # score the metric too

Exits 2 when the improvement pass could not complete — including when the
knowledge-graph store could not be read (#1383): the run then prints
`[warn] knowledge-graph store unreadable: …` beside its counts and exits 2
even when the markdown-driven half (drift, contradiction pairs) succeeded,
because partial success must be reportable as partial, not as clean. Exits 3
when it completed but claims to have changed more facts than its own budget
allowed — a distinct code, and an overrun keeps it even when the store was
also unreadable. Exits 4 when the apply loop stopped mid-run because ONE
ACTION marked more than one fact (the blast-radius halt, #3006): facts may
already have been retired by the action that tripped it, so a run that
halted can never report itself as a success — and 4 is printed only for that
value, never for the ordinary `run action budget` stop that ends a capped
apply. A consumer that reports success when it could not see the graph is
worse than none — same rule knowledge-health-report.py runs on (it
prints `STORE UNAVAILABLE: …` to stderr; this pass keeps its line beside the
counts on stdout, where a counts-reader cannot miss it).

The 4 path also prints `LLOYD_SCRIPT_EXIT: fact-improvement.py exit=4` on
stdout AND stderr before it leaves (#2619, whose rule this borrows): a
scheduled run's exit code dies inside the agent turn that invoked it, so the
line is the half the fleet can read — `app/autonomy.py` scans each tool result
for it into that run's `meta_json.script_alarm`, which is what
`/api/autonomy/health`'s `script_alarm_runs` counts. The 2 and 3 rungs return
their codes only, as they always have: a code that prints no sentinel is
invisible to that counter, and closing that gap for the other two rungs is a
separate decision, not part of #3006.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from agent_mcp import fact_improvement as fi  # noqa: E402

#: Exit code for a run whose apply loop hit the blast-radius guard
#: (`agent_mcp/fact_improvement.py::run_improvement`, the `changed > 1` branch).
#: 4 because 2 is the unreadable store and argparse's own usage error, 3 is the
#: budget overrun, and 1 is the generic "the pass raised" path.
BLAST_RADIUS_EXIT = 4

#: Marker line naming this script's real exit code, printed just before a
#: non-zero exit. Its format is `app/autonomy.py`'s `EXIT_SENTINEL_RE`, and
#: `tests/test_memory_improvement.py::test_a_halted_apply_prints_the_sentinel_the_fleet_scanner_reads`
#: checks the output against that module's real `_scan_exit_sentinels` rather
#: than a copy of the pattern, so the two cannot drift quietly
#: (#2619, `scripts/memory/knowledge-health-report.py::emit_exit_sentinel`).
EXIT_SENTINEL_PREFIX = "LLOYD_SCRIPT_EXIT: "


def _emit_exit_sentinel(code: int) -> None:
    """Print the exit-code marker line on both streams. See EXIT_SENTINEL_PREFIX.

    Both, and last: the scheduled invocation pipes `2>&1 | tail`, so a line that
    only exists on one stream, or only before the rest of the output, is the
    line that gets lost.
    """
    line = f"{EXIT_SENTINEL_PREFIX}{Path(__file__).name} exit={code}"
    print(line)
    print(line, file=sys.stderr)


def _blast_radius_halt(rec: dict) -> list:
    """Entities this run halted on, for the blast-radius guard (#3006).

    Keys on the VALUE, never on a `stopped` key being present: the same field
    carries `RUN_BUDGET_STOP` on every apply that simply ran out of budget, and
    that is the normal end of a capped run, not a writer that over-reached.
    """
    return [str(entry.get("entity") or "?")
            for entry in (rec.get("per_entity") or [])
            if isinstance(entry, dict)
            and entry.get("stopped") == fi.BLAST_RADIUS_HALT]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true",
                    help="actually expire/invalidate facts (default: plan only)")
    ap.add_argument("--sources", default="corrections,drift",
                    help="comma-separated signals to read: corrections, drift")
    ap.add_argument("--entity", action="append", default=[],
                    help="improve only this entity (repeatable); bypasses signal sources")
    ap.add_argument("--days", type=int, default=fi.DRIFT_WINDOW_DAYS,
                    help=f"drift window in days (default {fi.DRIFT_WINDOW_DAYS})")
    ap.add_argument("--corrections-window-days", type=int,
                    default=fi.CORRECTIONS_WINDOW_DAYS,
                    help=f"corrections window in days (default "
                         f"{fi.CORRECTIONS_WINDOW_DAYS}). An entry older than "
                         "this names no entity; the run then says the log is "
                         "stale since <date> instead of reporting no "
                         "corrections (#802)")
    ap.add_argument("--limit", type=int, default=40,
                    help="max entities to examine (default 40)")
    ap.add_argument("--max-actions", type=int, default=fi.MAX_ACTIONS_PER_RUN,
                    help=f"max facts this run may change (default {fi.MAX_ACTIONS_PER_RUN})")
    ap.add_argument("--report-eval", action="store_true",
                    help="also score fact_entity_recall before and after (slow: ~1s/query)")
    ap.add_argument("--json", action="store_true", help="print the full record as JSON")
    args = ap.parse_args()

    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    unknown = set(sources) - {"corrections", "drift"}
    if unknown:
        print(f"[error] unknown source(s): {', '.join(sorted(unknown))}"
              " (known: corrections, drift)", file=sys.stderr)
        return 2

    before_eval = fi._fact_entity_recall() if args.report_eval else None

    try:
        rec = fi.run_improvement(
            apply=args.apply, sources=sources, entities=args.entity or None,
            days=args.days, limit=args.limit, max_actions=args.max_actions,
            report_eval=args.report_eval,
            corrections_days=args.corrections_window_days,
        )
    except Exception as exc:  # noqa: BLE001 - a consumer must not exit 0 on a failed pass
        print(f"[error] improvement pass failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(rec, indent=2, default=str))
    else:
        mode = "APPLY" if rec["apply"] else "DRY RUN"
        near_dupes = sum(e.get("near_duplicates") or 0 for e in rec["per_entity"])
        # `of N` is the point (#699): a bare `planned=0` was read as a claim
        # about the knowledge graph when it is a claim about the slice this run
        # scanned. None = drift was not consulted, so there is no pool to name;
        # print the count rather than a fraction over a total nobody measured.
        # A refused entity is not a scanned one (#702): `scanned=40` with seven
        # god-nodes refused was a coverage claim about facts nobody compared,
        # so the refusals come off the count and are named beside it.
        refused = [e["entity"] for e in rec["per_entity"] if e.get("refused")]
        total = rec.get("drift_candidates_total")
        n_scanned = len(rec["entities"]) - len(refused)
        scanned = (f"entities scanned={n_scanned} of {total} drift candidates"
                   if total is not None else
                   f"entities scanned={n_scanned}")
        # The pool as a share of the tree (#1461): `of 11806 drift candidates`
        # read as a selection on a night the pool was 11,806 of 11,807 entity
        # dirs. The fraction goes on the same line as the count it qualifies.
        corpus = rec.get("drift_corpus_total")
        if total is not None and corpus:
            scanned += (f" ({total}/{corpus} entity dirs = "
                        f"{100.0 * total / corpus:.2f}% of the corpus)")
        # Facts, not entities (#3007): `entities scanned=40 of 1,383` says where
        # the fraction was taken, not how much of it was read. The 2026-10-18 pass
        # scanned 40 entities and, across its 9 partial ones, 442 of 1,638 facts —
        # 27.0% — and printed none of that, so a hub covered at 3.5% and an entity
        # scanned whole both read as "scanned". The fact fraction goes on this SAME
        # stdout line, on the rule #1383 states for the store verdict: a caveat
        # printed on another line is the false green again. A refused entity is not
        # in the numerator, because on such a row `checked` is the size of the read
        # that was never scanned (`facts.py` returns `checked=len(facts)` with
        # `refused: True`) and counting it would present a refusal as 100%
        # coverage. When nothing at all was scanned no fraction prints at all:
        # there is nothing to divide, and a printed 0.0% over real totals would read
        # as a coverage that was measured rather than an absence of one.
        if n_scanned:
            facts_scanned = sum(e.get("checked", 0) for e in rec["per_entity"]
                                if not e.get("refused"))
            facts_total = sum(e.get("facts_total") or 0 for e in rec["per_entity"])
            if facts_total:
                scanned += (f" facts scanned={facts_scanned:,} of {facts_total:,}"
                            f" = {100.0 * facts_scanned / facts_total:.1f}%")
        if refused:
            scanned += f" refused={len(refused)} ({', '.join(refused)})"
        print(f"[{mode}] signals={rec['signals']} {scanned}"
              f" actions planned={rec['actions_planned']} taken={rec['actions_taken']}"
              f" near-duplicate pairs reported-not-deleted={near_dupes}")
        print(f"[facts] active {rec['before_active']} -> {rec['after_active']}"
              f" (delta {rec['delta_active']})")
        # The store verdict beside the counts (#1383): `-1 -> -1` is the
        # sentinel shape for "the store could not answer", and on its own it
        # read as a zero. It must sit on the SAME output as the counts — a
        # warning on stderr next to a clean stdout is the false green again.
        # A stale corrections log beside the counts, not a bare `signals=0`
        # (#802): `memory/corrections.md` went unwritten on 2026-05-08 and every
        # pass still printed a quiet zero, because a zero from a dead channel and
        # a zero from a quiet fortnight printed identically. Same rule as the
        # store verdict above — the caveat goes on stdout, where the
        # counts-reader is, never only in the JSON nobody opens.
        # Per SOURCE, not one aggregate line (#2789). What this block used to print
        # was a single age — the newest entry across every stale log, from
        # `corrections_stale_since` — for the whole channel. With three logs of
        # three different vintages (a heading log last written 2026-05-08, a
        # standing-prose log whose bullets carry no date, a memory index with
        # in-window `[feedback]` bullets) that printed one file's staleness as the
        # channel's, so a run that had just read four fresh corrections still said
        # "log stale since 2026-05-08" and a run that read none looked identical to
        # it. Each line below carries its own log's entries / in-window / signals
        # counts and, when that log is the dry one, its own newest entry — so the
        # stdout states WHICH log is empty instead of pronouncing the channel dead.
        _srcs = rec.get("corrections_sources") or {}
        for _path, _info in sorted(_srcs.items()):
            _line = (f"[corrections] {_path}: entries={_info.get('entries')} "
                     f"in_window={_info.get('in_window')} "
                     f"signals={_info.get('signals')} status={_info.get('status')}")
            if _info.get("status") == "no_entries_in_window" and _info.get("newest_entry"):
                _line += (f" — log stale since {_info['newest_entry']} "
                          f"(window {rec.get('corrections_window_days')} days)")
            print(_line)
        # The other silent half of the same channel (#2198): entries with no date on
        # the line. The stale line above covers a log that is simply old; it never
        # fired for `lloyd/USER.md`, whose bullets are standing prose with no date at
        # all, so a pass that got 0 signals from both sources printed the half that
        # had a line and read as a two-source pass. This prints the undated count, the
        # fallback date that placed them, and whether that date came from the file's
        # front matter or its mtime — a fallback a reader cannot see is the silent
        # admission this line exists to prevent.
        _fb = rec.get("corrections_undated_fallback") or {}
        if _fb.get("summary") and rec.get("corrections_status") != "signals":
            print(f"[corrections] {_fb['summary']}")
        # The drift twins of that line (#1461). A pool that is the corpus is a
        # sweep, not a recency slice — a rebuild re-stamped every row's
        # `created_at` — and an empty pool is a stopped writer, not a quiet
        # night. Both on stdout beside the counts, for the same reason.
        window = rec.get("drift_window_days")
        if rec.get("drift_status") == "sweep":
            print(f"[drift] sweep: {total} of {corpus} entity dirs "
                  f"({100.0 * (rec.get('drift_pool_fraction') or 0):.2f}%) have a fact "
                  f"created in the last {window} days — the pool is the corpus, not a "
                  f"recency slice (a rebuild re-stamps created_at); the scanned "
                  f"entities are its newest-created, ties by name")
        elif rec.get("drift_status") == "stale":
            print(f"[drift] 0 candidates in window: no fact created in the last "
                  f"{window} days across {corpus} entity dirs; stale since "
                  f"{rec.get('drift_newest_fact') or 'never (no dated fact)'}")
        if rec.get("store_ok") is False:
            print(f"[warn] knowledge-graph store unreadable: {rec.get('store_error')}")
        elif rec.get("store_ok") is True:
            # `is True`, not `else`: a record that carries no verdict at all
            # must not print the ok line either, or the printed half of this
            # fix is once again a default rather than a measurement.
            print(f"[store] ok ({rec.get('kg_db')})")
        if before_eval is not None or rec.get("fact_entity_recall") is not None:
            print(f"[metric] fact_entity_recall {before_eval} -> {rec.get('fact_entity_recall')}")
        if not rec["apply"]:
            print("[note] dry run: nothing was written. --apply to act on the list.")
        for entry in rec["per_entity"]:
            if entry.get("error"):
                print(f"  ! {entry['entity']}: {entry['error']}")
                continue
            for action in entry.get("actions", []):
                flag = ("expired" if action.get("applied") else
                        "would expire" if action.get("planned") else
                        "skipped" if action.get("skipped") else "no-op")
                print(f"  [{flag}] {entry['entity']}: {action['loser_fact'][:70]}")
                print(f"           {action['reason'][:120]}")
            if entry.get("stopped"):
                print(f"  = {entry['entity']}: stopped — {entry['stopped']}")
        print(f"[record] {rec.get('record_path') or rec.get('record_error')}")

    if rec["actions_taken"] > args.max_actions:
        print(f"[error] changed {rec['actions_taken']} facts, over the {args.max_actions}"
              " allowed by this run — inspect the record before trusting it",
              file=sys.stderr)
        return 3
    halted = _blast_radius_halt(rec)
    if halted:
        # Checked after the overrun so a 10-fact night keeps its 3, and before
        # the store rung because a writer that just marked more than it was
        # aimed at is the fact a reader has to see first: the store may be
        # perfectly healthy and the facts still be gone (#3006). The per-entity
        # `= <entity>: stopped — …` line above carries the same fact, but only
        # on the human branch — under `--json` stdout was the record and nothing
        # named the halt at all, which is how a halted apply reached the
        # scheduler looking like a successful run. Like the overrun rung above,
        # the `[error]` line goes to stderr so a `--json` reader still gets a
        # parseable record on stdout; the sentinel below is the one line that
        # goes to both, because the scheduler reads the streams merged.
        print(f"[error] blast radius exceeded for {', '.join(halted)}: one action "
              "marked more than one fact, so the apply stopped for that entity — "
              "facts may already be retired; inspect "
              f"{rec.get('record_path') or rec.get('record_error') or 'the run record'} "
              "before trusting this run", file=sys.stderr)
        _emit_exit_sentinel(BLAST_RADIUS_EXIT)
        return BLAST_RADIUS_EXIT
    if rec.get("store_ok") is False:
        # #1383: the markdown half may have planned and reported everything
        # correctly and the graph still was never seen. Checked after the
        # budget rung so an overrun keeps its distinct 3 — but the pass that
        # could not read the store can no longer exit 0.
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
