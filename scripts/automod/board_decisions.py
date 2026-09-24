"""What the queue decisions produced (#904).

Three things decide what gets worked on next — triage (single, group and the
sweep), the board steward, and the hold that releases a confirmation when the
implement pool has room — and until this module nothing joined any of them to
what happened afterwards. `board_flow` counts created/closed, `board_health`
counts composition, #76 counts run health; a decider that promotes the wrong
item every day, on time and without errors, read the same as one deciding well.

This is a join over the automod ledger, read once
(`backlog._ledger_rows`), with the board files only as a last resort:

* **Promotions.** Every entry of an item into the implement pool (`up_next`,
  or `in_progress` reached without passing through it), attributed to the
  **source event** that moved it — a `backlog_triage` confirm, a
  `backlog_confirm_released`, a `board_steward` applied move, or the
  reconciler's own `status_moved` (a human reopen, a re-offer). Not
  `status_moved.by`: it is NULL on 1,135 of 1,156 rows. An implement round
  that starts on an item no event put in the pool is an entry too, attributed
  `unrecorded` — a person in Mission Control, or autonomy #35 writing through
  the MCP store, leave no ledger row, and pretending they did not happen would
  hide exactly the decisions this was asked to see.
* **Terminal state** is the first exit from the pool after the entry: landed,
  closed, sent to a person, sent back through triage, or returned to `draft`
  for any other reason; `None` while it is still in the pool. Entry and exit
  timestamps are both carried, so dwell is a subtraction.
* **Per-day counts** cover every UTC day in the window, zeros included, so a
  hold streak is a run of days and not an absence of rows.
* **Retire-then-reopen.** Every retirement out of the working pool inside the
  window — parked by the sweep, sent to a person, or closed by the loop — and
  whether the item later came back: into the pool for the two that stay open,
  onto the board at all for a close. The board file is consulted only when the
  ledger records no way back, because a reopen from Mission Control writes no
  ledger row.

Placement (the item's own open question, decided 2026-09-24): a function here,
surfaced through `board_health` — so the dashboard's backlog payload and the
steward's `<board_health>` block carry its summary with no new job, route or
poll — and `round board-decisions` for the full per-item listing. Not a seventh
autonomy job and not an extension of #76: both would put a model turn in front
of a deterministic join. The thresholds that would make a reading actionable
are not set here; they need a week of traffic and a person.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.automod import backlog as B

WINDOW_DAYS = 7
# Below this the outcome is `insufficient` and no rate is emitted: one landing
# out of two promotions is not a 50% rate of anything.
MIN_PROMOTIONS = 5
WORKING = (B.IMPLEMENT_POOL_STATUS, "in_progress")

# Reasons the reconciler writes when it hands an item to a person. Rows from
# 2026-09-24 on carry `needs_human` themselves; these read the ones before.
_NEEDS_HUMAN_REASONS = ("a human decides", "person still owes")
_LANDED_REASON = "landed"


def _decider_of_confirm(row: dict) -> str:
    """Which judgment a `backlog_triage` confirm came from."""
    if "umbrella" in row:
        return "group_triage"
    if row.get("sweep_batch"):
        return "sweep"
    src = row.get("verdict_source")
    if src == "gate" or row.get("red_tree"):
        return "gate"
    if src == "human":
        return "human"
    return "triage"


def _decider_of_move(reason: str) -> str:
    r = (reason or "").lower()
    if r.startswith("offered again — reopened") or "reopened" in r[:40]:
        return "reopen"
    if r.startswith("offered again"):
        return "reoffer"
    if "triage confirmed" in r:
        return "triage"
    if r.startswith("board steward"):
        return "steward"
    return "reconcile"


def _needs_human(row: dict) -> bool:
    if "needs_human" in row:
        return bool(row["needs_human"])
    reason = str(row.get("reason") or "").lower()
    return any(k in reason for k in _NEEDS_HUMAN_REASONS)


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _day(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def board_decisions(ledger: Path, *, items: list | None = None, now: float | None = None,
                    days: int = WINDOW_DAYS, min_promotions: int = MIN_PROMOTIONS) -> dict:
    """The full reading for `days` ending `now`. `items` (the board, any
    status) is optional: it supplies each item's current status and the
    board-side reopen fallback."""
    now = time.time() if now is None else now
    since = now - days * 86400
    # `board_steward` and `backlog_sweep` rows are about many items and carry
    # no `item_id`, so the filter is on the clock only.
    rows = sorted((d for d in B._ledger_rows(ledger) if isinstance(d.get("ts"), (int, float))),
                  key=lambda d: float(d["ts"]))
    board = {int(i.id): i for i in (items or [])}

    entries: list[dict] = []          # every entry into the pool, all time
    in_pool: dict[int, dict] = {}     # item -> its open entry
    held: dict[int, dict] = {}        # item -> the held confirm a release will honour
    last_confirm: dict[int, dict] = {}
    retirements: list[dict] = []      # every retirement, all time
    # (ts, item, what, entry) — the ways back, and the loop's own re-triage
    signals: list[tuple[float, int, str, dict | None]] = []

    def enter(iid: int, ts: float, decider: str, via: str, decision_ts: float | None) -> None:
        if iid in in_pool:
            return
        e = {"item_id": iid, "decider": decider, "via": via,
             "decision_ts": decision_ts, "entry_ts": ts,
             "terminal": None, "terminal_ts": None, "landed": False}
        entries.append(e)
        in_pool[iid] = e
        signals.append((ts, iid, "entered_pool", e))

    def leave(iid: int, ts: float, kind: str) -> None:
        e = in_pool.pop(iid, None)
        if e is None:
            return
        if e["landed"] and kind in ("returned", "needs_human"):
            kind = "landed"
        e["terminal"], e["terminal_ts"] = kind, ts

    def retire(iid: int, ts: float, kind: str, by: str) -> None:
        retirements.append({"item_id": iid, "ts": ts, "kind": kind, "by": by})

    for d in rows:
        ev = d.get("event")
        ts = float(d["ts"])
        if ev == "backlog_sweep":
            # The sweep parks by summary row: a `low` rank on a draft.
            for sid, rank in (d.get("ranked") or {}).items():
                if str(rank).startswith("low/") and str(sid).isdigit() \
                        and int(sid) not in in_pool:
                    retire(int(sid), ts, "parked", "sweep")
            continue
        if ev == "board_steward":
            if not d.get("apply"):
                continue
            for m in d.get("applied") or []:
                if not m.get("applied"):
                    continue
                mid = int(m.get("item_id") or 0)
                if m.get("status") == B.IMPLEMENT_POOL_STATUS:
                    enter(mid, ts, "steward", "steward", ts)
                elif m.get("status") == B.TRIAGE_POOL_STATUS:
                    leave(mid, ts, "returned")
                    if B.NEEDS_HUMAN_TAG in (m.get("tags_add") or []):
                        retire(mid, ts, "needs_human", "steward")
            continue
        try:
            iid = int(d["item_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if ev == "backlog_triage":
            verdict = d.get("verdict")
            if verdict == "confirmed":
                last_confirm[iid] = d
                if d.get("held"):
                    held[iid] = d
                elif not d.get("closed"):
                    enter(iid, ts, _decider_of_confirm(d), "confirm", ts)
            elif d.get("closed"):
                leave(iid, ts, "closed")
                retire(iid, ts, "closed", _decider_of_confirm(d))
        elif ev == "backlog_confirm_released":
            if d.get("moved"):
                c = held.pop(iid, None) or last_confirm.get(iid) or {}
                enter(iid, ts, _decider_of_confirm(c) if c else "triage", "released",
                      float(c["ts"]) if c.get("ts") else None)
        elif ev == "status_moved":
            to, reason = d.get("to"), str(d.get("reason") or "")
            if to in WORKING:
                if d.get("from") in B.CLOSED_STATUSES:
                    signals.append((ts, iid, "reopened_on_board", None))
                if iid not in in_pool:
                    dec = _decider_of_move(reason)
                    c = last_confirm.get(iid) if dec == "triage" else None
                    enter(iid, ts, dec, "status_moved",
                          float(c["ts"]) if c else (None if dec == "reconcile" else ts))
            elif to == B.TRIAGE_POOL_STATUS:
                if d.get("from") in B.CLOSED_STATUSES:
                    signals.append((ts, iid, "reopened_on_board", None))
                nh = _needs_human(d)
                landed = reason.lower().startswith(_LANDED_REASON)
                kind = ("landed" if landed else "needs_human" if nh
                        else "returned")
                leave(iid, ts, kind)
                if nh:
                    retire(iid, ts, "needs_human", "loop")
            elif to in B.CLOSED_STATUSES:
                leave(iid, ts, "closed")
                retire(iid, ts, "closed", "loop")
        elif ev == "item_landed":
            if iid in in_pool:
                in_pool[iid]["landed"] = True
            if d.get("closed"):
                leave(iid, ts, "landed")
                retire(iid, ts, "landed", "loop")
        elif ev == "item_closed":
            leave(iid, ts, "closed")
            retire(iid, ts, "closed", str(d.get("by") or "loop"))
        elif ev == "backlog_expired":
            leave(iid, ts, "closed")
            retire(iid, ts, "expired", "loop")
        elif ev == "backlog_retriage":
            leave(iid, ts, "retriage")
            signals.append((ts, iid, "retriage", None))
        elif ev == "item_reopened":
            signals.append((ts, iid, "reopened_on_board", None))
        elif ev == "backlog_implement" and d.get("phase") == "started":
            enter(iid, ts, "unrecorded", "implement_started", None)

    # ── promotions in the window ────────────────────────────────────────
    window = [e for e in entries if since <= e["entry_ts"] <= now]
    listing = []
    for e in window:
        cur = board.get(e["item_id"])
        if e["terminal"] is None and cur is not None and cur.status in B.CLOSED_STATUSES:
            # Closed by something that writes no ledger row — a person in
            # Mission Control, a hand sweep. Done, with no timestamp to trust.
            e = {**e, "terminal": "closed_off_ledger"}
        listing.append({
            "item_id": e["item_id"], "decider": e["decider"], "via": e["via"],
            "decision_at": _iso(e["decision_ts"]), "entered_at": _iso(e["entry_ts"]),
            "terminal": e["terminal"], "terminal_at": _iso(e["terminal_ts"]),
            "dwell_s": (round(e["terminal_ts"] - e["entry_ts"], 1)
                        if e["terminal_ts"] is not None else None),
            "landed": e["landed"] or e["terminal"] == "landed",
            "status_now": cur.status if cur is not None else None,
        })
    by_decider: dict[str, int] = {}
    for e in listing:
        by_decider[e["decider"]] = by_decider.get(e["decider"], 0) + 1
    n = len(listing)
    if n < min_promotions:
        outcome: dict = {"verdict": "insufficient", "promotions": n, "min": min_promotions}
    else:
        kinds: dict[str, int] = {}
        for e in listing:
            k = e["terminal"] or "open"
            kinds[k] = kinds.get(k, 0) + 1
        dwell = sorted(e["dwell_s"] for e in listing if e["dwell_s"] is not None)
        done = (kinds.get("landed", 0) + kinds.get("closed", 0)
                + kinds.get("closed_off_ledger", 0))
        outcome = {"verdict": "measured", "promotions": n, "terminal": kinds,
                   "done_rate": round(done / n, 3),
                   "landed_rate": round(kinds.get("landed", 0) / n, 3),
                   "dwell_h_median": (round(dwell[len(dwell) // 2] / 3600, 1) if dwell else None)}

    # ── per day, zeros included ─────────────────────────────────────────
    first = datetime.fromtimestamp(since, timezone.utc).date()
    last = datetime.fromtimestamp(now, timezone.utc).date()
    per_day: list[dict] = []
    day = first
    while day <= last:
        per_day.append({"day": day.isoformat(), "promotions": 0, "by_decider": {}})
        day += timedelta(days=1)
    index = {p["day"]: p for p in per_day}
    for e in window:
        p = index.get(_day(e["entry_ts"]))
        if p is not None:
            p["promotions"] += 1
            p["by_decider"][e["decider"]] = p["by_decider"].get(e["decider"], 0) + 1
    longest = run = 0
    for p in per_day:
        run = run + 1 if p["promotions"] == 0 else 0
        longest = max(longest, run)

    # ── retired, then back ──────────────────────────────────────────────
    closing = ("closed", "landed", "expired")
    back_rows: list[dict] = []
    in_window = [r for r in retirements if since <= r["ts"] <= now]
    by_item: dict[int, list] = {}
    for sig in signals:
        by_item.setdefault(sig[1], []).append(sig)
    for r in in_window:
        way = None
        retriaged = False
        for ts, _iid, what, entry in by_item.get(r["item_id"], ()):
            if ts <= r["ts"]:
                continue
            if what == "retriage":
                retriaged = True
                continue
            if what == "entered_pool" or r["kind"] in closing:
                way = {"at": _iso(ts), "how": what,
                       "via": entry["decider"] if entry else None}
                break
        if way is None:
            cur = board.get(r["item_id"])
            if cur is not None and (
                    (r["kind"] in closing and cur.status in B.OPEN_STATUSES)
                    or (r["kind"] not in closing and cur.status in WORKING)):
                way = {"at": None, "how": f"board status {cur.status}", "via": None}
        if way is not None:
            back_rows.append({"item_id": r["item_id"], "kind": r["kind"], "by": r["by"],
                              "retired_at": _iso(r["ts"]), "back_at": way["at"],
                              "how": way["how"], "via": way["via"],
                              # The loop's own second life (`retriage_spent_items`)
                              # sat between the two: a reversal, but not a person's.
                              "after_retriage": retriaged})
    by_kind: dict[str, dict] = {}
    for r in in_window:
        k = by_kind.setdefault(r["kind"], {"retired": 0, "reopened": 0, "after_retriage": 0,
                                           "by_reopen_item": 0})
        k["retired"] += 1
    for r in back_rows:
        k = by_kind[r["kind"]]
        k["reopened"] += 1
        k["after_retriage"] += int(r["after_retriage"])
        k["by_reopen_item"] += int(r["via"] == "reopen")

    return {
        "window": {"days": days, "since": _iso(since), "until": _iso(now)},
        "promotions": {"count": n, "by_decider": by_decider, "outcome": outcome},
        "per_day": per_day,
        "zero_streak": {"longest": longest, "current": run},
        "retire_reopen": {"retired": len(in_window), "reopened": len(back_rows),
                          "ids": sorted({r["item_id"] for r in back_rows}),
                          "by_kind": by_kind},
        "entries": listing,
        "reopened": back_rows,
    }


def summary(reading: dict) -> dict:
    """What `board_health` carries: everything but the per-item listings,
    which a 15-minute steward row and a 2-second dashboard poll do not need.
    The reopened ids stay — a count with no ids cannot be checked."""
    return {k: v for k, v in reading.items() if k not in ("entries", "reopened")}


def health_line(s: dict | None) -> str:
    """The steward's one line for it."""
    if not s:
        return "board decisions: (unavailable)"
    p, rr, z = s.get("promotions") or {}, s.get("retire_reopen") or {}, s.get("zero_streak") or {}
    o = p.get("outcome") or {}
    if o.get("verdict") == "measured":
        out = (f"{o.get('done_rate', 0):.0%} reached done ({o.get('landed_rate', 0):.0%} landed), "
               f"median dwell {o.get('dwell_h_median')} h")
    else:
        out = f"outcome insufficient ({o.get('promotions', 0)} < {o.get('min', MIN_PROMOTIONS)})"
    deciders = ", ".join(f"{k} {v}" for k, v in sorted((p.get("by_decider") or {}).items())) or "none"
    days = " ".join(str(d.get("promotions", 0)) for d in s.get("per_day") or [])
    ids = ", ".join(f"#{i}" for i in (rr.get("ids") or [])[:12]) or "none"
    kinds = "; ".join(f"{k} {v.get('reopened', 0)}/{v.get('retired', 0)}"
                      for k, v in sorted((rr.get("by_kind") or {}).items()))
    return (f"board decisions {(s.get('window') or {}).get('days', WINDOW_DAYS)}d: "
            f"{p.get('count', 0)} promotions into the pool ({deciders}); {out}; per day [{days}], "
            f"zero-promotion streak {z.get('current', 0)} d now, longest {z.get('longest', 0)} d; "
            f"{rr.get('reopened', 0)} of {rr.get('retired', 0)} retirements later reopened "
            f"({kinds or 'none'}; ids {ids})")
