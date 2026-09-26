#!/usr/bin/env python3
"""Replay fact writes through the #1487 ADD/UPDATE/NOOP gate, offline.

The gate (`agent_mcp/fact_write_gate.py`) decides from two things only: the
new fact, and the entity's active facts in the same category. Both are on
disk, so a write can be re-decided exactly as the write path would have
decided it — against the facts that existed BEFORE it — without writing
anything. That is this script: it reads a COPY of the store (never the live
`kg.sqlite`; `.backup` one first), rebuilds each sampled write's prior state
from `created_at`, and asks djev the production question.

    # a copy of the store, then a replay of recent writes
    sqlite3 -readonly ~/lloyd-data/_pipeline/vault-derived/kg.sqlite \\
        ".backup ~/lloyd-data/eval/1487/kg-copy.sqlite"
    python eval/run_fact_write_gate_eval.py replay \\
        --db ~/lloyd-data/eval/1487/kg-copy.sqlite --since 2026-09-24 \\
        --n 400 --out ~/lloyd-data/eval/1487/replay.jsonl
    # hand labels (one `<row> <E|S|P|D>` per line), then the numbers
    python eval/run_fact_write_gate_eval.py score \\
        --decisions ~/lloyd-data/eval/1487/replay.jsonl \\
        --labels ~/lloyd-data/eval/1487/replay_labels.txt

Labels, for a (new, existing) pair:
  E  equivalent — the same claim; NOOP right, UPDATE harmless
  S  the new fact is contained in the existing one — NOOP right, UPDATE loses
     information (a false supersede)
  P  the new fact contains the existing one and says more — UPDATE right,
     NOOP loses the new detail
  D  different claims — ADD right; NOOP or UPDATE loses a fact

A false supersede (UPDATE on S or D) destroys a true fact, and is the number
the ship decision weighs first.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LLOYD_DJEV_SHADOW", "0")

from agent_mcp import fact_write_gate as gate  # noqa: E402
from eval.stats import wilson_ci  # noqa: E402

LIVE_DB = Path.home() / "lloyd-data" / "_pipeline" / "vault-derived" / "kg.sqlite"


def _open_copy(db: Path):
    """The copy, through `app.kg_store`, the store's only writer (CLAUDE.md); an
    absent file raises `StoreUnavailable` rather than reading as zero rows. The one
    opener that is not this module is the guardian's read-only row count (#1525)."""
    from app.kg_store import KGStore, _require_database
    if db.resolve() == LIVE_DB.resolve():
        raise SystemExit("refusing the live kg.sqlite: `.backup` a copy and pass that")
    return KGStore(_require_database(db))


def load_rows(db: Path) -> list[dict]:
    s = _open_copy(db)
    cols = ["entity", "category", "fact_id", "text_hash", "fact", "created_at",
            "source_doc", "expired_at", "invalid_at"]
    try:
        rows = [dict(zip(cols, r)) for r in s.conn.execute(
            f"select {', '.join(cols)} from facts_idx")]
    finally:
        s.close()
    return rows


def prior_state(rows: list[dict]) -> dict:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["expired_at"] or r["invalid_at"]:
            continue
        groups[(r["entity"], r["category"])].append(r)
    return groups


def existing_before(groups, r) -> list[dict]:
    """The active facts of r's (entity, category) written before it — what the
    write path would have held when r arrived. A same-text row is left out: the
    #499 verbatim guard answers that case before the gate is reached."""
    ts = r["created_at"] or ""
    return [{"id": o["fact_id"], "fact": o["fact"], "expired_at": None, "invalid_at": None}
            for o in groups[(r["entity"], r["category"])]
            if (o["created_at"] or "") < ts and o["text_hash"] != r["text_hash"]]


def cmd_replay(args) -> int:
    if args.k is not None:
        gate.SHORTLIST_K = args.k
    rows = load_rows(Path(args.db).expanduser())
    groups = prior_state(rows)
    recent = [r for r in rows if (r["created_at"] or "") >= args.since
              and not (r["expired_at"] or r["invalid_at"])]
    if args.exclude:
        # The calibration pairs' new facts: a held-out replay must not grade
        # the thresholds on the rows they were chosen on.
        held = {p["new_hash"] for p in json.load(open(Path(args.exclude).expanduser()))}
        recent = [r for r in recent if r["text_hash"] not in held]
    random.Random(args.seed).shuffle(recent)
    sample = recent[: args.n]
    out = open(Path(args.out).expanduser(), "w", encoding="utf-8")
    counts = Counter()
    for k, r in enumerate(sample):
        existing = existing_before(groups, r)
        d = gate.decide(r["entity"], r["category"], r["fact"], existing)
        counts[d.verdict if d.asked or d.verdict != "add" else f"add:{d.reason}"] += 1
        out.write(json.dumps({"row": k, "entity": r["entity"], "category": r["category"],
                              "fact": r["fact"], "created_at": r["created_at"],
                              "source_doc": r["source_doc"], "n_existing": len(existing),
                              **d.as_dict()}, ensure_ascii=False) + "\n")
        out.flush()
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(sample)} {dict(counts)}", file=sys.stderr)
    print(json.dumps({"sampled": len(sample), "of_recent": len(recent), **counts}))
    return 0


def _labels(path: Path) -> dict[int, str]:
    out = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] in "ESPD":
            out[int(parts[0])] = parts[1]
    return out


def _ci(k: int, n: int) -> str:
    if not n:
        return "n=0"
    lo, hi = wilson_ci(k, n)
    return f"{k}/{n} = {k / n:.3f} [{lo:.3f}, {hi:.3f}]"


def cmd_score(args) -> int:
    decs = [json.loads(l) for l in open(Path(args.decisions).expanduser())]
    labels = _labels(Path(args.labels).expanduser())
    by = Counter(d["verdict"] for d in decs)
    asked = [d for d in decs if d["asked"]]
    lat = sorted(d["latency_ms"] for d in asked if d.get("latency_ms") is not None)
    print(f"writes replayed: {len(decs)}  verdicts: {dict(by)}")
    print(f"djev asked on {len(asked)}/{len(decs)}; failed-open "
          f"{sum(d['reason'] == 'djev_failed' for d in decs)}")
    if lat:
        print(f"djev latency ms p50 {lat[len(lat) // 2]:.0f} p95 {lat[int(len(lat) * .95)]:.0f} "
              f"max {lat[-1]:.0f}")
    noop = [d for d in decs if d["verdict"] == "noop" and d["row"] in labels]
    upd = [d for d in decs if d["verdict"] == "update" and d["row"] in labels]
    print("NOOP right (E|S):      ", _ci(sum(labels[d["row"]] in "ES" for d in noop), len(noop)))
    print("UPDATE safe (E|P):     ", _ci(sum(labels[d["row"]] in "EP" for d in upd), len(upd)))
    print("false supersede (S|D): ", _ci(sum(labels[d["row"]] in "SD" for d in upd), len(upd)))
    acted = noop + upd
    lost = [d for d in acted if (d["verdict"] == "noop" and labels[d["row"]] in "PD")
            or (d["verdict"] == "update" and labels[d["row"]] in "SD")]
    print("any-loss among actions:", _ci(len(lost), len(acted)))
    for d in lost:
        print(f"   LOSS row {d['row']} {d['verdict']} [{labels[d['row']]}] {d['fact'][:90]!r}"
              f" vs {(d['target'] or {}).get('fact', '')[:90]!r}")
    adds = [d for d in decs if d["verdict"] == "add" and d["row"] in labels]
    if adds:
        missed = sum(labels[d["row"]] in "ESP" for d in adds)
        print("ADD with a labelled near-duplicate (missed):", _ci(missed, len(adds)))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--db", required=True)
    r.add_argument("--since", default="2026-09-24")
    r.add_argument("--n", type=int, default=400)
    r.add_argument("--seed", type=int, default=1487)
    r.add_argument("--out", required=True)
    r.add_argument("--exclude", help="pairs.json whose new_hash rows are left out")
    r.add_argument("--k", type=int, default=None,
                   help="candidates per canvas (default: the gate's SHORTLIST_K)")
    s = sub.add_parser("score")
    s.add_argument("--decisions", required=True)
    s.add_argument("--labels", required=True)
    args = ap.parse_args(argv)
    return {"replay": cmd_replay, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
