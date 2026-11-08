#!/usr/bin/env python3
"""One dated census of the memory store's invalidation streams (#3927).

The talk this came from describes a four-stage loop — update, age, observe,
**grade**, correct — and the stage Lloyd has no counterpart to is *grade*:
snapshot the record as of a past date and compare what it believed against what
happened afterwards. Grading needs a base rate, and the base rate nobody here
has ever computed is the **hazard rate**: of the facts that were active at some
date, how many were invalidated within 30/60/90/180 days. `knowledge/ai/
weibull-decay-for-agent-memory.md` Open Question 1 says no published agent
system reports fitted beta values; #672, #864 and #1067 each need a hazard
curve to be tuned against, and none exists.

So this is the measurement that has to come first, and it is a *count*, not a
fit. It publishes the two invalidation streams separately because they are not
comparable:

  fact side  one pass over the markdown fact tree, read through the shipped
             `app.kg_store.parse_fact_file` — the same parser the index is
             built with, so the census and `facts_idx` cannot disagree about
             what a fact field is. Fact fields live NESTED in a `facts:` YAML
             list, not as top-level frontmatter keys, which is why a
             line-anchored grep over this tree returns nothing and reads as an
             empty store.
  edge side  the store's own `stats()` plus one aggregate read, because edges
             expire rather than delete and are the stream with volume in it.

Two rulings the census takes because leaving them implicit is what makes a
census lie:

  * **The root is resolved, never spelled.** The tree is
    `app.paths.VAULT_FACTS_ROOT`, which the 2026-09-22 moved-roots note at
    `app/paths.py:298-308` exists to keep honest. A literal path to a
    same-named directory under the vault matches zero files and would report a
    clean-looking zero invalidations — the false zero that would have fired
    this item's own stop gate on an artefact. The row therefore carries
    `facts_root` and `facts_root_is_default` (#700's lesson: a record that
    cannot say which tree it described certifies itself as production), and the
    store is reached only through `app.kg_store.store()` — this module never
    opens a database file itself.
  * **An event is a timestamp, not a row.** One bulk job that stamps 442 facts
    with the same `expired_at` is ONE discovery event, not 442. Counted as rows
    it clears the stop gate; counted as distinct stamps it does not, and the
    difference is the whole finding. The gate below is therefore computed on
    `independent_event_count` — the number of DISTINCT `expired_at` values in
    the fact stream — and `facts_with_expired_at` sits beside it, never inside
    it.

What is deliberately NOT here: the hazard arm (step 3) and the prediction arm
(step 4). Both are reported `not_built`, with the independent-event count as
the stated reason when the gate does not pass. Whether a curve could be
measured on the EDGE stream instead — which is where the volume is — is a
scope decision about the arm's substrate, and this census reports both counts
so a human can make it. Nothing here is consumed by any gate, changes
retrieval, or touches decay scoring.

    python -m scripts.maintenance.invalidation_census
    python -m scripts.maintenance.invalidation_census --out-dir /tmp/census

The artifact is report-only and lands under `app.paths.EVAL_BASELINES_DIR`,
next to the nightly baselines; an output directory inside the code tree is a
refusal, not a warning (#2378's ruling on in-tree runtime data).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import kg_store, paths                                # noqa: E402

#: The published stop-gate threshold, named so the artifact can cite it and a
#: later reader can tell a verdict from a vibe. The item's own wording is
#: "fewer than ~200 invalidation events corpus-wide: publish the census as the
#: deliverable … Do not fit anything to 25 positives."
STOP_GATE_NAME = "STOP_GATE_MIN_INDEPENDENT_EVENTS"
STOP_GATE_MIN_INDEPENDENT_EVENTS = 200

#: Half-life horizons the hazard arm would have used, recorded so the artifact
#: says what it declined to measure and in what units.
HAZARD_HORIZON_DAYS = (30, 60, 90, 180)

#: Buckets for (expired_at - created_at), in days. `expired_at` is written by
#: whoever ran the invalidation and defaults to *today*, so it is a discovery
#: time, not a truth time: this distribution is the censoring measurement that
#: makes the hazard curve readable as an UPPER bound on staleness. A negative
#: bucket is not a paradox, it is a backfill that expired a fact older than its
#: own creation stamp, and it is reported rather than clamped away.
DELTA_BUCKETS: tuple[tuple[str, float], ...] = (
    ("negative", 0.0),
    ("same_day", 1.0),
    ("1_to_7_days", 8.0),
    ("8_to_30_days", 31.0),
    ("31_to_90_days", 91.0),
    ("91_to_180_days", 181.0),
    ("181_to_365_days", 366.0),
    ("over_365_days", float("inf")),
)

CENSORING_NOTE = ("expired_at records when a job NOTICEDED a change, not when the world "
                  "changed: it is written at invalidation time and defaults to 'today'. "
                  "Any hazard curve read off it is therefore an upper bound on staleness, "
                  "and a stamp shared by hundreds of facts is one bulk job, not that many "
                  "observations.")


def _parse_ts(value: Any) -> Optional[datetime]:
    """Best-effort ISO-8601 parse. Naive stamps are read as UTC, because every
    writer in this codebase stamps with `datetime.now(timezone.utc)` and a
    naive stamp in a machine-facing payload is a UTC stamp (one-clock rule).
    Anything unparseable comes back None and is counted, never guessed."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip().strip("'\"")
        if not text or text.lower() in {"null", "none"}:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _is_set(value: Any) -> bool:
    """A fact field counts as set when it carries a real value, not a YAML null
    spelled as a string. The tree stores `expired_at: null` on live facts."""
    return _parse_ts(value) is not None or (
        value is not None and str(value).strip().strip("'\"").lower() not in {"", "null", "none"})


def _month(dt: Optional[datetime]) -> str:
    return dt.strftime("%Y-%m") if dt else "unknown"


def scan_fact_tree(facts_root: Path) -> dict[str, Any]:
    """One pass over the markdown fact tree at `facts_root`.

    Returns the fact-side stream: the four separate counts the acceptance asks
    for (total facts, facts carrying `expired_at`, facts carrying `invalid_at`,
    distinct entities affected), plus the distinct-timestamp count the stop gate
    is computed on and the cross-tab of creation month against expiry month.
    """
    root = Path(facts_root)
    total_facts = 0
    files_with_facts = 0
    expired = 0
    invalid = 0
    affected_entities: set[str] = set()
    expired_stamps: Counter[str] = Counter()
    invalid_stamps: Counter[str] = Counter()
    delta_buckets: Counter[str] = Counter()
    delta_days: list[float] = []
    created_x_expired: Counter[str] = Counter()
    unparseable_expired = 0
    unparseable_created = 0

    for path in sorted(root.rglob("*.md")) if root.is_dir() else []:
        fm, facts = kg_store.parse_fact_file(path)
        if not facts:
            continue
        files_with_facts += 1
        file_entity = str(fm.get("entity") or path.parent.name)
        for f in facts:
            if not str(f.get("fact") or "").strip():
                continue
            total_facts += 1
            entity = str(f.get("entity") or file_entity)
            exp_raw, inv_raw = f.get("expired_at"), f.get("invalid_at")
            exp = _parse_ts(exp_raw)

            if _is_set(exp_raw):
                expired += 1
                affected_entities.add(entity)
                expired_stamps[str(exp_raw).strip().strip("'\"")] += 1
                if exp is None:
                    unparseable_expired += 1
                cre = _parse_ts(f.get("created_at"))
                created_x_expired[f"{_month(cre)}_x_{_month(exp)}"] += 1
                if exp is not None and cre is not None:
                    days = (exp - cre).total_seconds() / 86400.0
                    delta_days.append(days)
                    for name, ceiling in DELTA_BUCKETS:
                        if days < ceiling:
                            delta_buckets[name] += 1
                            break
                else:
                    unparseable_created += 1
            if _is_set(inv_raw):
                invalid += 1
                affected_entities.add(entity)
                invalid_stamps[str(inv_raw).strip().strip("'\"")] += 1

    top_stamp, top_count = (expired_stamps.most_common(1)[0] if expired_stamps
                            else (None, 0))
    return {
        "total_facts": total_facts,
        "facts_with_expired_at": expired,
        "facts_with_invalid_at": invalid,
        "distinct_entities_affected": len(affected_entities),
        # The stop gate's input. Rows are what a bulk job writes; distinct
        # stamps are what was actually discovered.
        "distinct_expired_at_values": len(expired_stamps),
        "distinct_invalid_at_values": len(invalid_stamps),
        "files_scanned": (len(sorted(root.rglob("*.md"))) if root.is_dir() else 0),
        "files_with_fact_lists": files_with_facts,
        "unparseable_expired_at": unparseable_expired,
        "expired_facts_missing_created_at": unparseable_created,
        "created_x_expired_month": dict(sorted(created_x_expired.items())),
        "expiry_top_stamps": [{"expired_at": s, "facts": c}
                              for s, c in expired_stamps.most_common(5)],
        # Kept from the same pass `censoring_row` reads, so the tree is walked
        # once: two passes over 47k files for the same numbers is how a census
        # ends up reporting two different snapshots of a live tree.
        "expiry_day_buckets": {name: delta_buckets.get(name, 0) for name, _ in DELTA_BUCKETS},
        "expiry_deltas_measured": len(delta_days),
        "expiry_deltas": sorted(round(d, 4) for d in delta_days),
    }


def censoring_row(fact_stream: dict[str, Any]) -> dict[str, Any]:
    """The censoring measurement, derived from the one tree pass: how old a
    fact was when something stamped it, and how much of the stream a single
    bulk stamp accounts for.

    `expired_at` is a DISCOVERY time, so the day distribution below is the
    reader's correction: a stream that expires everything the day it was
    written is a job rewriting rows, not a corpus changing underneath.
    """
    buckets = fact_stream["expiry_day_buckets"]
    expired = fact_stream["facts_with_expired_at"]
    counted = fact_stream["expiry_deltas_measured"]
    deltas = fact_stream["expiry_deltas"]
    top = fact_stream["expiry_top_stamps"][:1]
    top_stamp = top[0]["expired_at"] if top else None
    top_count = top[0]["facts"] if top else 0
    within_90 = sum(buckets.get(n, 0) for n in
                    ("negative", "same_day", "1_to_7_days", "8_to_30_days", "31_to_90_days"))
    return {
        "expiry_minus_created_day_buckets": dict(buckets),
        "deltas_measured": counted,
        "deltas_unmeasurable_missing_or_unparseable_created_at":
            fact_stream["expired_facts_missing_created_at"],
        "median_days_to_expiry": (round(deltas[counted // 2], 2) if counted else None),
        "share_expiring_same_day": (round(buckets.get("same_day", 0) / counted, 4)
                                    if counted else None),
        "share_expiring_within_90_days": (round(within_90 / counted, 4) if counted else None),
        "most_common_expired_at": top_stamp,
        "most_common_expired_at_facts": top_count,
        "share_carried_by_single_most_common_stamp": (round(top_count / expired, 4)
                                                     if expired else None),
        "note": CENSORING_NOTE,
    }


def scan_edge_stream(kg) -> dict[str, Any]:
    """The edge-side stream, from `store().stats()` plus one aggregate read.

    `stats()` gives the totals; distinct expiry DAYS are the honest unit here
    because a nightly sweep stamps a whole day's expirations with one call, so
    the day count is the independent-event analogue on this stream. Both the
    stats-derived delta and a direct count are reported, and `agree` says
    whether they match.
    """
    stats = kg.stats()
    rows = kg._query("SELECT substr(expired_at, 1, 10) AS day, COUNT(*) AS facts "
                     "FROM edges WHERE expired_at IS NOT NULL GROUP BY day")
    per_day = {r["day"]: int(r["facts"]) for r in rows}
    expired_rows = sum(per_day.values())
    stamps = kg._query("SELECT COUNT(DISTINCT expired_at) AS n "
                       "FROM edges WHERE expired_at IS NOT NULL")[0]
    return {
        "edges_total": int(stats["edges_total"]),
        "edges_active": int(stats["edges_active"]),
        "edges_expired": expired_rows,
        "edges_expired_from_stats_delta": int(stats["edges_total"]) - int(stats["edges_active"]),
        "agree": expired_rows == int(stats["edges_total"]) - int(stats["edges_active"]),
        "expired_edge_distinct_expiry_days": len(per_day),
        "expired_edge_distinct_expiry_timestamps": int(stamps["n"]),
        "expired_edge_top_days": [{"day": d, "edges": c} for d, c in
                                  sorted(per_day.items(), key=lambda kv: (-kv[1], kv[0]))[:5]],
        "store_entities": int(stats["entities"]),
        "store_aliases": int(stats["aliases"]),
    }


def scan_store_fact_index(kg) -> dict[str, Any]:
    """The same fact-side question, asked of the store's index instead of the
    markdown tree.

    Two instruments, both reported, because the index is BUILT from the tree: a
    disagreement is a stale index, and a census that publishes only one of them
    can hand a reader a false zero from whichever is behind. The tree is the
    source of record for the four headline counts; this is the witness that says
    whether the index currently agrees.
    """
    row = kg._query("SELECT COUNT(*) AS n, "
                    "SUM(expired_at IS NOT NULL) AS expired, "
                    "SUM(invalid_at IS NOT NULL) AS invalid, "
                    "COUNT(DISTINCT expired_at) AS stamps FROM facts_idx")[0]
    return {
        "index_rows": int(row["n"]),
        "rows_with_expired_at": int(row["expired"] or 0),
        "rows_with_invalid_at": int(row["invalid"] or 0),
        "distinct_expired_at_values": int(row["stamps"] or 0),
    }


def stop_gate(independent_event_count: int,
              threshold: int = STOP_GATE_MIN_INDEPENDENT_EVENTS) -> dict[str, Any]:
    """The step-2 stop gate, on DISTINCT expiry stamps and never on raw rows."""
    verdict = "build_arms" if independent_event_count >= threshold else "stop_census_only"
    return {
        "name": STOP_GATE_NAME,
        "threshold": int(threshold),
        "independent_event_count": int(independent_event_count),
        "basis": ("count of DISTINCT expired_at values in the fact stream, not the number of "
                  "expired fact rows: one bulk job stamping N facts is one event"),
        "verdict": verdict,
    }


def arms(gate: dict[str, Any], edge_gate: dict[str, Any]) -> dict[str, Any]:
    """What the two unbuilt arms say about themselves, and why."""
    count = gate["independent_event_count"]
    if count < gate["threshold"]:
        reason = (f"{count} independent invalidation events (distinct expiry stamps) is below "
                  f"{gate['threshold']} ({gate['name']}); fitting a hazard curve to this would "
                  f"be fitting noise, which is what the item's stop gate exists to prevent")
    else:
        reason = (f"{count} independent invalidation events clears {gate['threshold']}, but the "
                  f"arm's substrate is unresolved: the fact stream carries {count} distinct "
                  f"expiry stamps against the edge stream's "
                  f"{edge_gate['expired_edge_distinct_expiry_days']} distinct expiry days, and "
                  f"choosing between them changes the contract. Awaiting that ruling")
    return {
        "hazard_rate": {"status": "not_built", "horizons_days": list(HAZARD_HORIZON_DAYS),
                        "reason": reason},
        "prediction": {"status": "not_built",
                       "reason": "measures a flag lift over the invalidation base rate, so it "
                                 "needs the hazard arm's base rate first; not built"},
    }


def build_row(*, facts_root: Optional[Path] = None, store: Any = None,
              threshold: int = STOP_GATE_MIN_INDEPENDENT_EVENTS) -> dict[str, Any]:
    """The census row. Both sources are read through their sanctioned module:
    the tree path comes from `app.paths.VAULT_FACTS_ROOT` (resolved at call
    time, so an override or a test redirect is what the row reports) and the
    database only through `app.kg_store.store()` — `store=` exists so a test can
    hand over a provisioned one, never so a caller can pass a raw connection."""
    root = Path(paths.VAULT_FACTS_ROOT) if facts_root is None else Path(facts_root)
    if store is None:
        store = kg_store.store()
    now = datetime.now(timezone.utc)
    fact_stream = scan_fact_tree(root)
    censoring = censoring_row(fact_stream)
    # The raw per-fact day-deltas stay out of the artifact: the buckets, the
    # median and the shares above are the reading, and a 30k-float array in a
    # dated report row is a blob that hides every other field.
    fact_stream.pop("expiry_deltas", None)
    edge_stream = scan_edge_stream(store)
    store_index = scan_store_fact_index(store)
    edge_gate = {"expired_edge_distinct_expiry_days":
                 edge_stream["expired_edge_distinct_expiry_days"]}
    gate = stop_gate(fact_stream["distinct_expired_at_values"], threshold)
    return {
        "schema": "invalidation-census/1",
        "generated_at": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "facts_root": str(root),
        "facts_root_is_default": root == Path(paths.VAULT_FACTS_ROOT_DEFAULT),
        "kg_db_resolved_by": "app.kg_store.store()",
        "fact_stream": fact_stream,
        "fact_index_from_store": store_index,
        "cross_check": {
            "source_of_record_for_the_four_fact_counts": (
                "the markdown tree at facts_root — `fact_stream` above. "
                "`fact_index_from_store` is the second instrument, not a replacement"),
            "tree_total_facts": fact_stream["total_facts"],
            "store_index_rows": store_index["index_rows"],
            "agree_on_totals": fact_stream["total_facts"] == store_index["index_rows"],
            "agree_on_expired": (fact_stream["facts_with_expired_at"]
                                 == store_index["rows_with_expired_at"]),
            "agree_on_invalid": (fact_stream["facts_with_invalid_at"]
                                 == store_index["rows_with_invalid_at"]),
            "why_two_instruments": (
                "facts_idx is BUILT from the tree, so the two can disagree only by "
                "one of them being stale; publishing only one hands the reader a "
                "false zero from whichever is behind. A disagreement here is an "
                "index that needs `kg_rebuild`, not a different fact store"),
        },
        "edge_stream": edge_stream,
        "censoring": censoring,
        "stop_gate": gate,
        # The same gate applied to the edge stream's independent unit, reported
        # so the substrate question is answerable from the artifact alone.
        "edge_stream_gate": {
            "name": STOP_GATE_NAME,
            "threshold": gate["threshold"],
            "independent_event_count": edge_stream["expired_edge_distinct_expiry_days"],
            "basis": "count of DISTINCT expiry days among expired edges",
            "verdict": ("build_arms" if edge_stream["expired_edge_distinct_expiry_days"]
                        >= gate["threshold"] else "stop_census_only"),
        },
        "arms": arms(gate, edge_gate),
        "open_ruling": ("Whether the hazard and prediction arms are built on the edge stream "
                        "rather than the fact stream. Owed to a human: it changes the arm's "
                        "whole contract, and no round can make it."),
        "regenerate": {
            "command": "python -m scripts.maintenance.invalidation_census",
            "cwd": str(ROOT),
            # `app.paths` anchors DATA_ROOT inside a git worktree, so a round's
            # checkout resolves both the tree and the store into itself and the
            # census correctly refuses a store that is not there. From anywhere
            # but the live checkout, name the data root explicitly.
            "env": {"LLOYD_DATA": "the data root holding the fact tree and the store; "
                                  "required when this code tree is not the running one"},
        },
    }


def resolve_out_dir(raw: Path | str) -> Path:
    """Where the artifact goes: a refusal, not a warning, if it is inside the
    code tree (#2378 — a worktree's copy of runtime state must not reach a
    commit)."""
    out = Path(raw).expanduser().resolve()
    code_root = ROOT.resolve()
    if out == code_root or code_root in out.parents:
        raise ValueError(f"refusing to write runtime data inside the code tree: {out}")
    return out


def artifact_name(date: str) -> str:
    return f"invalidation-census-{date}.json"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Dated census of the memory store's "
                                             "invalidation streams (#3927). Report-only.")
    ap.add_argument("--out-dir", default=None,
                    help=f"Default {paths.EVAL_BASELINES_DIR}")
    ap.add_argument("--facts-root", default=None,
                    help="Default app.paths.VAULT_FACTS_ROOT (resolved, never spelled)")
    ap.add_argument("--threshold", type=int, default=STOP_GATE_MIN_INDEPENDENT_EVENTS,
                    help=f"{STOP_GATE_NAME} override, for a what-if run")
    ap.add_argument("--no-write", action="store_true", help="Print the row, write nothing")
    args = ap.parse_args(argv)

    root = Path(args.facts_root) if args.facts_root else None
    out_dir = resolve_out_dir(args.out_dir or paths.EVAL_BASELINES_DIR)
    row = build_row(facts_root=root, threshold=args.threshold)

    written: Optional[Path] = None
    if not args.no_write:
        out_dir.mkdir(parents=True, exist_ok=True)
        written = out_dir / artifact_name(row["date"])
        written.write_text(json.dumps(row, indent=2, sort_keys=True, default=str) + "\n",
                           encoding="utf-8")
    fs, gate = row["fact_stream"], row["stop_gate"]
    print(f"facts_root={row['facts_root']} (default={row['facts_root_is_default']})")
    print(f"facts: total={fs['total_facts']} expired={fs['facts_with_expired_at']} "
          f"invalid={fs['facts_with_invalid_at']} entities={fs['distinct_entities_affected']}")
    print(f"independent invalidation events (distinct stamps)={gate['independent_event_count']} "
          f"vs {gate['name']}={gate['threshold']} -> {gate['verdict']}")
    print(f"edges: total={row['edge_stream']['edges_total']} active={row['edge_stream']['edges_active']} "
          f"expired={row['edge_stream']['edges_expired']} "
          f"days={row['edge_stream']['expired_edge_distinct_expiry_days']}")
    xc = row["cross_check"]
    print(f"tree vs index: tree={xc['tree_total_facts']} index_rows={xc['store_index_rows']} "
          f"expired_agree={xc['agree_on_expired']} invalid_agree={xc['agree_on_invalid']}")
    print(f"arms: hazard={row['arms']['hazard_rate']['status']} "
          f"prediction={row['arms']['prediction']['status']}")
    print(f"artifact={written}" if written else "artifact=not_written (--no-write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
