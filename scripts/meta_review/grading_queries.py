#!/usr/bin/env python3
"""Inner Voice (#345) Stage 5 — grading-pass meta-review queries.

The Stage 5 cross-stage tooling deliverable: "Notebook adds grading-pass
queries: outcome_addressed rate over time, false-positive rate by
persona, human-vs-Brain-2 grader disagreement."

There is no notebook in this repo yet (the Stage 3 deliverable
``meta_review_template.ipynb`` was never landed), so this is the script
form: each top-level function is a query you can run from the CLI or
import from a Python REPL.

Usage:
    python -m scripts.meta_review.grading_queries
    python scripts/meta_review/grading_queries.py
    python scripts/meta_review/grading_queries.py --hours 168
    python scripts/meta_review/grading_queries.py --json

The default run prints a small report covering the three Stage 5 gates:

  1. ``addressed_rate over time`` — daily buckets of addressed_true /
     (addressed_true + addressed_false). The trend line answers "are
     interventions getting more or less effective as the personas
     iterate?"
  2. ``false-positive rate by persona`` — for each triggering persona,
     the fraction of interventions where the grader said
     ``addressed=true`` (high → persona's signal correlated with real
     follow-up; low → persona may be flagging things Brain 1 didn't
     need to act on).
  3. ``coverage`` — fraction of recent interventions where
     ``outcome_addressed`` is non-null AND graded within 1 outcome
     turn. The Stage 5 acceptance gate requires ≥80%.

The "human-vs-Brain-2 grader disagreement" query needs hand-labeled
data that doesn't exist yet — a stub function (`grader_disagreement`)
is included with the schema it expects so the validation pass has
somewhere to write its CSV.
"""

from __future__ import annotations

import argparse
import json as _json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Allow running directly from any cwd. Walk up to find the lloyd root.
_THIS = Path(__file__).resolve()
_LLOYD_ROOT = _THIS.parent.parent.parent
DB_PATH = _LLOYD_ROOT / "usage.db"


def _conn() -> sqlite3.Connection:
    """Read-only-ish connection. Uses a separate cursor so we don't fight
    the live backend's WAL writer.
    """
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _since_iso(hours: float) -> str:
    return (datetime.utcnow() - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# Query 1: addressed_rate over time
# ---------------------------------------------------------------------------


def addressed_rate_over_time(
    *, hours: float = 24 * 14, bucket_hours: float = 24
) -> list[dict[str, Any]]:
    """Bucket graded interventions by outcome time and return the per-bucket
    addressed_rate.

    Args:
        hours: window to scan. Default 14 days.
        bucket_hours: bucket width. Default 24h (daily).

    Returns: list of dicts, oldest bucket first:
        {
          "bucket_start_iso": ISO timestamp,
          "graded": <int>,
          "addressed_true": <int>,
          "addressed_false": <int>,
          "addressed_null": <int>,
          "addressed_rate": <float>,  -- true / (true+false)
        }
    """
    since = _since_iso(hours)
    conn = _conn()
    rows = conn.execute(
        """SELECT graded_at, outcome_addressed
             FROM inner_voice_interventions
            WHERE graded_at IS NOT NULL
              AND graded_at >= ?
            ORDER BY graded_at ASC""",
        (since,),
    ).fetchall()

    if not rows:
        return []

    bucket_seconds = bucket_hours * 3600
    buckets: dict[int, dict[str, int]] = {}

    def _epoch(ts_iso: str) -> int:
        # Tolerate both ISO with and without microseconds.
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(ts_iso[:19], fmt)
                return int(dt.timestamp())
            except ValueError:
                continue
        # Fallback: best-effort parse of leading 19 chars
        return 0

    for r in rows:
        e = _epoch(str(r["graded_at"]))
        bucket_key = int(e // bucket_seconds) * int(bucket_seconds)
        b = buckets.setdefault(
            bucket_key,
            {
                "graded": 0,
                "addressed_true": 0,
                "addressed_false": 0,
                "addressed_null": 0,
            },
        )
        b["graded"] += 1
        addr = r["outcome_addressed"]
        if addr == 1:
            b["addressed_true"] += 1
        elif addr == 0:
            b["addressed_false"] += 1
        else:
            b["addressed_null"] += 1

    out: list[dict[str, Any]] = []
    for k in sorted(buckets):
        b = buckets[k]
        denom = b["addressed_true"] + b["addressed_false"]
        rate = (b["addressed_true"] / denom) if denom > 0 else 0.0
        out.append(
            {
                "bucket_start_iso": datetime.utcfromtimestamp(k).strftime(
                    "%Y-%m-%dT%H:%M:%S"
                ),
                **b,
                "addressed_rate": round(rate, 4),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Query 2: addressed/false-positive rate by triggering persona
# ---------------------------------------------------------------------------


def addressed_rate_by_persona(*, hours: float = 24 * 14) -> list[dict[str, Any]]:
    """Per-persona breakdown of intervention outcomes.

    The "false-positive rate" reading: a persona that frequently
    triggers interventions where the grader says ``addressed=false``
    (Brain 1 just ignored the critique with no apparent harm) is a
    candidate for prompt tuning OR for a higher severity threshold.

    Returns: list of dicts, sorted by total descending:
        {
          "persona": <str>,
          "total":           <int>,
          "graded":          <int>,
          "addressed_true":  <int>,
          "addressed_false": <int>,
          "addressed_null":  <int>,
          "addressed_rate":  <float>,  -- true / (true + false)
          "false_positive_rate": <float>,  -- false / (true + false), inverse signal
        }
    """
    since = _since_iso(hours)
    conn = _conn()
    rows = conn.execute(
        """SELECT iv.id,
                  iv.outcome_addressed,
                  iv.outcome_turn_id,
                  c.persona AS triggering_persona
             FROM inner_voice_interventions iv
        LEFT JOIN inner_voice_critiques c
               ON c.id = iv.triggered_by_critique_id
            WHERE iv.created_at >= ?""",
        (since,),
    ).fetchall()

    by_persona: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "total": 0,
            "graded": 0,
            "addressed_true": 0,
            "addressed_false": 0,
            "addressed_null": 0,
        }
    )

    for r in rows:
        p = r["triggering_persona"] or "(unknown)"
        b = by_persona[p]
        b["total"] += 1
        if r["outcome_turn_id"] is not None:
            b["graded"] += 1
            addr = r["outcome_addressed"]
            if addr == 1:
                b["addressed_true"] += 1
            elif addr == 0:
                b["addressed_false"] += 1
            else:
                b["addressed_null"] += 1

    out: list[dict[str, Any]] = []
    for p, b in by_persona.items():
        denom = b["addressed_true"] + b["addressed_false"]
        rate = (b["addressed_true"] / denom) if denom > 0 else 0.0
        fp_rate = (b["addressed_false"] / denom) if denom > 0 else 0.0
        out.append(
            {
                "persona": p,
                **b,
                "addressed_rate": round(rate, 4),
                "false_positive_rate": round(fp_rate, 4),
            }
        )
    out.sort(key=lambda r: -r["total"])
    return out


# ---------------------------------------------------------------------------
# Query 3: coverage (Stage 5 gate)
# ---------------------------------------------------------------------------


def coverage_summary(*, hours: float = 24 * 7) -> dict[str, Any]:
    """Fraction of recent interventions where ``outcome_addressed`` is
    non-null. Stage 5 gate: ≥80%.

    Returns:
        {
          "total":          <int>,
          "graded":         <int>,
          "non_null_rate":  <float>,  -- non-null outcome / total
          "graded_rate":    <float>,  -- any verdict (incl. null) / total
          "passes_gate":    <bool>,   -- non_null_rate >= 0.80
        }
    """
    since = _since_iso(hours)
    conn = _conn()
    row = conn.execute(
        """SELECT
              COUNT(*)                                    AS total,
              SUM(CASE WHEN outcome_addressed IS NOT NULL THEN 1 ELSE 0 END) AS non_null,
              SUM(CASE WHEN outcome_turn_id IS NOT NULL THEN 1 ELSE 0 END)   AS graded
            FROM inner_voice_interventions
           WHERE created_at >= ?""",
        (since,),
    ).fetchone()
    total = int(row["total"] or 0)
    non_null = int(row["non_null"] or 0)
    graded = int(row["graded"] or 0)
    return {
        "total": total,
        "graded": graded,
        "non_null_rate": round((non_null / total) if total > 0 else 0.0, 4),
        "graded_rate": round((graded / total) if total > 0 else 0.0, 4),
        "passes_gate": ((non_null / total) >= 0.80) if total > 0 else False,
    }


# ---------------------------------------------------------------------------
# Query 4 (stub): human-vs-Brain-2 grader disagreement
# ---------------------------------------------------------------------------


def grader_disagreement(human_labels_csv: Path | None = None) -> dict[str, Any]:
    """Compare a human-labeled subset of interventions against the
    Brain 2 grader's verdict. Stage 5 spot-check gate: agreement ≥80%.

    Expected CSV schema (one row per labeled intervention):
        intervention_id,human_addressed,human_summary
        42,true,"agent did the missing vault_write"
        43,false,"agent re-asserted same claim, no tool call"
        44,null,"unclear — outcome turn was a tool call probe"

    Returns:
        {
          "labeled_count":      <int>,
          "agree_count":        <int>,
          "disagree_count":     <int>,
          "agreement_rate":     <float>,
          "passes_gate":        <bool>,   -- >= 0.80
          "disagreements": [
            {
              "intervention_id": <int>,
              "human": "true|false|null",
              "brain2": "true|false|null",
              "human_summary": <str>,
              "brain2_summary": <str>,
            }, ...
          ],
        }

    If `human_labels_csv` is None, returns {"labeled_count": 0, ...}.
    Run this after a hand-labeling pass (recommended N ≥ 20).
    """
    if human_labels_csv is None or not human_labels_csv.exists():
        return {
            "labeled_count": 0,
            "agree_count": 0,
            "disagree_count": 0,
            "agreement_rate": 0.0,
            "passes_gate": False,
            "disagreements": [],
            "note": (
                "no human-labels CSV provided; create one with schema "
                "intervention_id,human_addressed,human_summary"
            ),
        }

    import csv as _csv

    by_iv_id_human: dict[int, tuple[str, str]] = {}
    with human_labels_csv.open("r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            try:
                iv_id = int(row["intervention_id"])
                by_iv_id_human[iv_id] = (
                    str(row.get("human_addressed", "")).strip().lower(),
                    str(row.get("human_summary") or "").strip(),
                )
            except (KeyError, ValueError):
                continue

    if not by_iv_id_human:
        return {
            "labeled_count": 0,
            "agree_count": 0,
            "disagree_count": 0,
            "agreement_rate": 0.0,
            "passes_gate": False,
            "disagreements": [],
        }

    placeholders = ",".join("?" for _ in by_iv_id_human)
    conn = _conn()
    rows = conn.execute(
        f"""SELECT id, outcome_addressed, outcome_summary
              FROM inner_voice_interventions
             WHERE id IN ({placeholders})""",
        tuple(by_iv_id_human.keys()),
    ).fetchall()

    def _label(v: Any) -> str:
        if v is True or v == 1:
            return "true"
        if v is False or v == 0:
            return "false"
        return "null"

    agree = 0
    disagree = 0
    disagreements: list[dict[str, Any]] = []
    for r in rows:
        iv_id = int(r["id"])
        human_label, human_summary = by_iv_id_human[iv_id]
        brain2_label = _label(r["outcome_addressed"])
        brain2_summary = str(r["outcome_summary"] or "")
        if human_label == brain2_label:
            agree += 1
        else:
            disagree += 1
            disagreements.append(
                {
                    "intervention_id": iv_id,
                    "human": human_label,
                    "brain2": brain2_label,
                    "human_summary": human_summary,
                    "brain2_summary": brain2_summary,
                }
            )

    total = agree + disagree
    rate = (agree / total) if total > 0 else 0.0
    return {
        "labeled_count": total,
        "agree_count": agree,
        "disagree_count": disagree,
        "agreement_rate": round(rate, 4),
        "passes_gate": rate >= 0.80 if total > 0 else False,
        "disagreements": disagreements,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hours",
        type=float,
        default=24 * 7,
        help="Window in hours (default 7 days)",
    )
    parser.add_argument(
        "--bucket-hours",
        type=float,
        default=24,
        help="Bucket width for over-time query (default 24h)",
    )
    parser.add_argument(
        "--human-labels",
        type=Path,
        default=None,
        help="Path to a CSV of human-labeled grader spot-checks "
             "(intervention_id,human_addressed,human_summary). "
             "Disagreement query is skipped when omitted.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the default text report.",
    )
    args = parser.parse_args(argv)

    cov = coverage_summary(hours=args.hours)
    over_time = addressed_rate_over_time(
        hours=args.hours, bucket_hours=args.bucket_hours
    )
    by_persona = addressed_rate_by_persona(hours=args.hours)
    disagreement = grader_disagreement(args.human_labels)

    out = {
        "window_hours": args.hours,
        "coverage": cov,
        "addressed_rate_over_time": over_time,
        "addressed_rate_by_persona": by_persona,
        "grader_disagreement": disagreement,
    }

    if args.json:
        print(_json.dumps(out, indent=2, default=str))
        return 0

    # Pretty text report
    print(f"=== Inner Voice Stage 5 grading meta-review ({args.hours}h window) ===")
    print()
    print("[1] Coverage (Stage 5 gate ≥ 0.80 non-null)")
    cov_pass = "PASS" if cov["passes_gate"] else "FAIL"
    print(
        f"    total={cov['total']}  graded={cov['graded']}  "
        f"non_null_rate={cov['non_null_rate']}  graded_rate={cov['graded_rate']}  "
        f"=> {cov_pass}"
    )
    print()
    print(f"[2] Addressed rate over time (bucket={args.bucket_hours}h)")
    if not over_time:
        print("    (no graded interventions in window)")
    for b in over_time:
        denom = b["addressed_true"] + b["addressed_false"]
        print(
            f"    {b['bucket_start_iso']}  graded={b['graded']}  "
            f"true={b['addressed_true']}  false={b['addressed_false']}  "
            f"null={b['addressed_null']}  "
            f"rate={b['addressed_rate']}{'' if denom else ' (n/a)'}"
        )
    print()
    print("[3] Addressed rate by triggering persona")
    if not by_persona:
        print("    (no interventions in window)")
    for r in by_persona:
        print(
            f"    {r['persona']:24s}  total={r['total']:3d}  graded={r['graded']:3d}  "
            f"true={r['addressed_true']:3d}  false={r['addressed_false']:3d}  "
            f"null={r['addressed_null']:3d}  "
            f"addr={r['addressed_rate']}  fp={r['false_positive_rate']}"
        )
    print()
    print("[4] Human-vs-Brain-2 grader disagreement (Stage 5 spot-check ≥ 0.80)")
    if disagreement["labeled_count"] == 0:
        note = disagreement.get("note") or "no labeled rows"
        print(f"    (skipped: {note})")
    else:
        d_pass = "PASS" if disagreement["passes_gate"] else "FAIL"
        print(
            f"    labeled={disagreement['labeled_count']}  "
            f"agree={disagreement['agree_count']}  "
            f"disagree={disagreement['disagree_count']}  "
            f"agreement_rate={disagreement['agreement_rate']}  "
            f"=> {d_pass}"
        )
        if disagreement["disagreements"]:
            print("    Top disagreements:")
            for d in disagreement["disagreements"][:5]:
                print(
                    f"      iv#{d['intervention_id']}  human={d['human']}  "
                    f"brain2={d['brain2']}  | human: {d['human_summary'][:80]}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
