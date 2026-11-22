"""What each inflow source's items turned into.

The loop counted what it filed and what it closed, per pass. It never asked the
question a self-improving loop has to answer about its own intake: of the items
a source filed, how many became a landed change, and how many were closed
without one? On 2026-11-22 the largest source on the board (owed-check, 112 of
160 open items) was not counted by any gauge at all.

This is a read of the board files only — no ledger, no model, no network:

* **source** is who filed the item, from its tags, and for owed-check's older
  untagged follow-ups from the marker line `owed.apply_verdict` writes into the
  body (`OWED_BODY_MARKER`);
* **outcome** is `open`, or how it closed: `landed` (`automod_landed` is set),
  `retired` (triage judged it stale, already done or a duplicate), `expired`,
  or `closed` for anything else;
* **yield** is landed over everything that closed. It is `None` under
  `MIN_CLOSED`, never a rate off a handful.

**Yield is not value, and nothing may throttle a source on it.** The first
reading (30 days to 2026-11-22) had owed-check at 649 of 1,312 items created,
191 of them only witnessing an earlier item's claim, and an 84 % yield: the
loop lands its own bookkeeping as readily as anything else. What the table is
good for is the other direction — a source whose items are mostly closed
unlanded (step-conformance: 2 landed of 25) is filing work nobody wanted.
`scorecard` row 21 records it daily.

    python -m scripts.automod.source_yield            # last 30 days, table
    python -m scripts.automod.source_yield --days 14 --json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

#: The line `owed.apply_verdict` appends to every follow-up it files. Items
#: filed before 2026-11-22 carry no `spawned-by-owed` tag, so this is the only
#: way to attribute them.
OWED_BODY_MARKER = "Filed by owed-check from #"

#: Below this many closed items a source's yield is not a rate.
MIN_CLOSED = 5

OUTCOMES = ("open", "landed", "retired", "expired", "closed")


def source_of(tags, body: str = "") -> str:
    """Who filed an item. A person's item, and a scheduled task's untagged one,
    are both `other`: the board cannot tell them apart."""
    from scripts.automod import backlog as B
    tags = [str(t) for t in (tags or [])]
    if B.RED_TREE_TAG in tags:
        return "red-tree"
    if B.OWED_SPAWN_TAG in tags or OWED_BODY_MARKER in (body or ""):
        return "owed-check"
    if any(t in B.EVAL_SPAWN_TAGS for t in tags):
        return "youtube-eval"
    for t in tags:
        if t in B.REVIEW_SPAWN_TAGS:
            return "arch-review"
        if t == "spawned-by-triage":
            return "triage"
        if t in B.SPAWN_TAGS:
            return "autocode"
    if any(t.startswith("spawned-by-") for t in tags):
        # `backlog_write_task` stamps `spawned-by-<whatever the session called
        # itself>`: 50-odd names in a month, most used once. One bucket.
        return "scheduled-task"
    if "guardian" in tags:
        return "guardian"
    return "other"


def outcome_of(status: str, fm: dict) -> str:
    from scripts.automod import backlog as B
    if status in B.OPEN_STATUSES:
        return "open"
    if fm.get("automod_landed"):
        return "landed"
    if B.EXPIRED_TAG in [str(t) for t in (fm.get("tags") or [])]:
        return "expired"
    if fm.get("autotriage_retired") or fm.get("duplicate_of"):
        return "retired"
    return "closed"


def compute(*, days: float = 30.0, backlog_dir: Path | None = None,
            now: float | None = None) -> dict[str, Any]:
    """`{"days", "sources": {name: {created, open, landed, retired, expired,
    closed, derived, yield}}, "total": {...}}` for items created in the window."""
    from scripts.automod import backlog as B
    now = time.time() if now is None else now
    since = now - days * 86400
    sources: dict[str, dict[str, Any]] = {}
    for item in B.all_items(None, backlog_dir=backlog_dir):
        ts = B._iso_ts(item.created, legacy_local=True)
        if ts is None or ts < since:
            continue
        try:
            fm, body = B._split_frontmatter(item.path.read_text(encoding="utf-8"))
        except OSError:
            continue
        row = sources.setdefault(source_of(fm.get("tags"), body),
                                 {"created": 0, "derived": 0, **{o: 0 for o in OUTCOMES}})
        row["created"] += 1
        row[outcome_of(item.status, fm)] += 1
        if B.is_derived_name(item.name):
            row["derived"] += 1
    total: dict[str, Any] = {"created": 0, "derived": 0, **{o: 0 for o in OUTCOMES}}
    for row in sources.values():
        for key in total:
            total[key] += row[key]
    for row in [*sources.values(), total]:
        done = row["created"] - row["open"]
        row["yield"] = round(row["landed"] / done, 3) if done >= MIN_CLOSED else None
    return {"days": days, "sources": dict(sorted(sources.items(),
                                                 key=lambda kv: -kv[1]["created"])),
            "total": total}


def render(reading: dict) -> str:
    lines = [f"Inflow yield — items created in the last {reading['days']:g} d",
             "",
             "| source | created | open | landed | retired | expired | closed | derived | yield |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, r in [*reading["sources"].items(), ("total", reading["total"])]:
        y = "—" if r["yield"] is None else f"{r['yield'] * 100:.0f}%"
        lines.append(f"| {name} | {r['created']} | {r['open']} | {r['landed']} | {r['retired']} "
                     f"| {r['expired']} | {r['closed']} | {r['derived']} | {y} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="What each inflow source's items turned into")
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    reading = compute(days=args.days)
    print(json.dumps(reading, indent=2) if args.json else render(reading))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
