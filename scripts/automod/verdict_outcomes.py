"""What became of the items each decider ruled on, and whether `worth` predicts it.

`board_decisions` joins a *promotion* to its first exit from the pool, pooled
over seven days. `source_yield` reads what each inflow source's items turned
into. Neither answers the question a decider has to be held to: of the items
single triage *confirmed*, how many landed and stayed landed, and of the ones
it *retired*, how many came back? And nothing at all checked the sweep's
`worth` / `size` labels against results, although `backlog.rank_key` orders
every pool on them.

This is a read of the ledger (live file plus rotations, each event once) and
of the board files
of the items that were ruled on. No model, no network, nothing written.

**The verdict vocabulary is the ledger's own**, not invented here:

* `backlog_triage` rows carry `verdict` — `confirmed`, `stale`, `already_done`,
  `folded`, `unverifiable`, `not_code`, `no_measurement` — and the decider is
  read off the row: `group_triage` (it names an umbrella, a group or a group
  verdict), `sweep` (`sweep_batch`), `gate` (a red-tree row), `human`, else
  `triage`.
* `backlog_sweep` rows carry `judged: {id: keep}` and `ranked: {id:
  "<worth>/<size>"}`. A keep ranked `low` is `keep_low` (parked when it was a
  draft); the sweep's retirements are `backlog_triage` rows and are counted
  there, once.
* `owed_check` rows carry `result.moved` — `reopened` or `closed` — which is
  owed-check's ruling on the item itself (`reopen`, `close`). Its per-entry
  outcomes (`settled`, `ruling`, `work`, `recheck`) are about an owed line,
  not the item, and are not verdicts here.
* `steward_move` rows are the board steward's applied moves: `to_up_next`,
  `to_draft`, `to_in_progress`.

A verdict is in one of four **classes** by what it did, not by its name:
`confirm` (it put or kept the item on the way to a round), `retire` (the row
says `closed: true`, or owed-check closed it), `fold`, and `other` (a verdict
that neither promoted nor closed — `unverifiable`, a refused close).

**Outcomes**, from what the ledger says happened to that item *after* the
verdict's timestamp, with the board file as the last resort:

    landed_kept       an `item_landed` row closed it, and nothing reopened it
    landed_reopened   it landed and closed, then a round started on it again, an
                      `item_reopened` / owed-check reopen named it, a guardian
                      rollback took its commit off main, or it is open today
    round_rejected    its round closed it `rejected` or `unnecessary`
    retired_later     a later triage verdict closed it
    expired           `backlog_expired`
    spent             open, and the last thing the reconciler said was that its
                      unattended attempts are spent
    open              still open, for any other reason
    owed_closed       owed-check closed it with no landing: tried and closed, or
                      its remainder handed to a follow-up item
    closed_other      closed with no ledger row that says how (a hand close)
    reopened          retire class only: it was back in a pool, in a round, or
                      open on the board afterwards
    stayed_closed     retire class only: none of that

One item counts once per (decider, verdict): the first such verdict in the
window. An item confirmed three times on its way through three rounds is one
confirmation that did or did not end in a landing.

**What this may be used for.** `source_yield`'s rule stands here too: a rate of
landing is not a rate of value — the loop lands 84 % of its own bookkeeping
items — and nothing may throttle a decider or a source on it. The two figures
worth reading are the ones that can only be bad: a confirmation whose round
found the item *unnecessary*, and a retirement that came back.

    python -m scripts.automod.verdict_outcomes --days 30
    python -m scripts.automod.verdict_outcomes --days 60 --worth
    python -m scripts.automod.verdict_outcomes --json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

#: Below this many resolved verdicts a rate is `None`, never a rate off a handful.
MIN_N = 20

CONFIRM, RETIRE, FOLD, OTHER = "confirm", "retire", "fold", "other"

CONFIRM_OUTCOMES = ("landed_kept", "landed_reopened", "round_rejected", "retired_later",
                    "expired", "owed_closed", "spent", "open", "closed_other")
RETIRE_OUTCOMES = ("stayed_closed", "reopened")
#: Outcomes that are not yet an answer: left out of every denominator.
UNRESOLVED = ("open", "spent")

_ROUND_REJECTIONS = ("rejected", "unnecessary")
_SPENT_MARK = "attempts are spent", "attempt is spent"

WORTH_LEVELS = ("high", "medium", "low")
SIZE_LEVELS = ("small", "medium", "large")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval; `None` at n = 0. Same formula as `eval/stats.py`'s
    `wilson_ci`, restated because `scripts/` does not import `eval/`."""
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def _rate(k: int, n: int, *, min_n: int = MIN_N) -> dict[str, Any]:
    """`{k, n, rate, ci}` with `rate`/`ci` None under `min_n`."""
    if n < min_n:
        return {"k": k, "n": n, "rate": None, "ci": None}
    return {"k": k, "n": n, "rate": round(k / n, 4), "ci": wilson(k, n)}


def separated(a: dict, b: dict) -> bool:
    """True when two `_rate` cells' Wilson intervals do not overlap — the bar
    a difference has to clear before anything is built on it."""
    if not a.get("ci") or not b.get("ci"):
        return False
    return a["ci"][1] < b["ci"][0] or b["ci"][1] < a["ci"][0]


def ledger_rows(ledger: Path | None = None) -> list[dict]:
    """The live ledger plus its rotations, each event once.

    `board_decisions._replay_rows` dedupes on a row's whole canonical JSON, which
    costs ~0.9 s over 91,527 rows (measured 2026-11-22) on every call. The key
    here is `(ts, event, item_id)`: `ts` is a float stamped to the microsecond,
    and the only thing this module would get wrong from a row read twice is an
    attempt count, so the cheap key is enough and it is ~10x faster."""
    from scripts.automod import backlog as B
    from scripts.automod import state as S
    seen: set[tuple] = set()
    out: list[dict] = []
    for d in B._ledger_rows_merged(ledger or S.LEDGER_PATH):
        key = (d.get("ts"), d.get("event"), d.get("item_id"), d.get("batch_id"))
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


# ── reading the verdicts ───────────────────────────────────────────────────

def decider_of_triage(row: dict) -> str:
    """Which judgment wrote a `backlog_triage` row."""
    if "umbrella" in row or row.get("group") or row.get("group_verdict"):
        return "group_triage"
    if row.get("sweep_batch"):
        return "sweep"
    if row.get("verdict_source") == "gate" or row.get("red_tree"):
        return "gate"
    if row.get("verdict_source") == "human" or row.get("by") == "human":
        return "human"
    return "triage"


def _class_of_triage(row: dict) -> str:
    verdict = str(row.get("verdict") or "")
    if verdict == "confirmed":
        return CONFIRM
    if verdict == "folded":
        return FOLD
    return RETIRE if row.get("closed") else OTHER


def _int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def verdicts_in(rows: list[dict], *, since: float, until: float) -> list[dict]:
    """Every verdict in `[since, until]`: `{ts, item_id, decider, verdict,
    cls}` plus `worth`/`size` on a sweep keep and `held` on a held confirm."""
    out: list[dict] = []
    for d in rows:
        ts = d.get("ts")
        if not isinstance(ts, (int, float)) or not since <= ts <= until:
            continue
        ev = d.get("event")
        if ev == "backlog_triage":
            iid = _int(d.get("item_id"))
            verdict = str(d.get("verdict") or "")
            if iid is None or not verdict or verdict == "incomplete":
                continue
            out.append({"ts": float(ts), "item_id": iid, "decider": decider_of_triage(d),
                        "verdict": verdict, "cls": _class_of_triage(d),
                        "held": bool(d.get("held")),
                        "session_id": str(d.get("session_id") or "")})
        elif ev == "backlog_sweep":
            judged, ranked = d.get("judged") or {}, d.get("ranked") or {}
            if not isinstance(judged, dict):
                continue
            for key, v in judged.items():
                iid = _int(key)
                if iid is None or str(v) != "keep":
                    continue
                worth, _, size = str((ranked or {}).get(key) or "").partition("/")
                worth = worth if worth in WORTH_LEVELS else ""
                size = size if size in SIZE_LEVELS else ""
                out.append({"ts": float(ts), "item_id": iid, "decider": "sweep",
                            "verdict": "keep_low" if worth == "low" else "keep",
                            "cls": CONFIRM, "worth": worth, "size": size,
                            "session_id": str(d.get("session_id") or "")})
        elif ev == "owed_check":
            iid = _int(d.get("item_id"))
            moved = str((d.get("result") or {}).get("moved") or "") if d.get("ok") else ""
            if iid is None or moved not in ("reopened", "closed"):
                continue
            out.append({"ts": float(ts), "item_id": iid, "decider": "owed_check",
                        "verdict": "reopen" if moved == "reopened" else "close",
                        "cls": CONFIRM if moved == "reopened" else RETIRE,
                        "session_id": str(d.get("session_id") or "")})
        elif ev == "steward_move":
            iid, to = _int(d.get("item_id")), str(d.get("to") or "")
            if iid is None or not to:
                continue
            out.append({"ts": float(ts), "item_id": iid, "decider": "board_steward",
                        "verdict": f"to_{to}", "cls": CONFIRM if to == "up_next" else OTHER,
                        "session_id": str(d.get("tick") or "")})
    return out


# ── what happened afterwards ───────────────────────────────────────────────

#: The events an item's later history is read from.
_TIMELINE_EVENTS = frozenset({"item_landed", "item_closed", "item_reopened", "backlog_implement",
                              "backlog_triage", "backlog_expired", "owed_check", "status_moved",
                              "review"})


def timelines(rows: list[dict]) -> tuple[dict[int, list[dict]], dict[str, float]]:
    """`({item_id: [rows, oldest first]}, {rolled-back commit: ts})`."""
    by_item: dict[int, list[dict]] = {}
    rolled: dict[str, float] = {}
    for d in rows:
        ts = d.get("ts")
        if not isinstance(ts, (int, float)):
            continue
        ev = d.get("event")
        if ev == "rollback_succeeded" and d.get("commit"):
            rolled[str(d["commit"])] = float(ts)
        elif ev in _TIMELINE_EVENTS:
            iid = _int(d.get("item_id"))
            if iid is not None:
                by_item.setdefault(iid, []).append(d)
    for seq in by_item.values():
        seq.sort(key=lambda d: float(d["ts"]))
    return by_item, rolled


def _is_reopen(d: dict) -> bool:
    ev = d.get("event")
    if ev == "item_reopened":
        return True
    if ev == "backlog_implement":
        return d.get("phase") in ("started", "reopened")
    if ev == "owed_check":
        return bool(d.get("ok")) and (d.get("result") or {}).get("moved") == "reopened"
    return False


def outcome_after(verdict: dict, seq: list[dict], rolled: dict[str, float], *,
                  status: str | None, landed_marker: bool = False,
                  open_statuses: tuple[str, ...] = ("draft", "up_next", "in_progress")
                  ) -> dict[str, Any]:
    """The outcome of one verdict. `seq` is the item's timeline, `status` its
    status on the board today (`None`: no file). Returns `{outcome, detail,
    closed_ts, attempts, refusals}`."""
    ts = verdict["ts"]
    later = [d for d in seq if float(d["ts"]) > ts]
    is_open = status in open_statuses
    attempts = sum(1 for d in later if d.get("event") == "backlog_implement"
                   and d.get("phase") == "started")
    refusals = sum(1 for d in later if d.get("event") == "review" and d.get("blocking"))
    base = {"attempts": attempts, "refusals": refusals, "closed_ts": None, "detail": ""}

    if verdict["cls"] == RETIRE:
        back = next((d for d in later if _is_reopen(d)
                     or (d.get("event") == "backlog_triage" and d.get("verdict") == "confirmed")
                     or (d.get("event") == "item_landed")
                     or (d.get("event") == "status_moved"
                         and d.get("to") in ("up_next", "in_progress"))), None)
        if back is not None or is_open:
            landed = any(d.get("event") == "item_landed" and d.get("closed") for d in later)
            return {**base, "outcome": "reopened",
                    "detail": "landed afterwards" if landed else
                              (str(back.get("event")) if back is not None else "open on the board")}
        return {**base, "outcome": "stayed_closed"}

    landing = next((d for d in later if d.get("event") == "item_landed" and d.get("closed")), None)
    if landing is None and verdict["cls"] == FOLD:
        # A folded member's fate is its umbrella's: the umbrella's landing closes it.
        landing = next((d for d in later if d.get("event") == "item_closed"
                        and d.get("by") == "umbrella"), None)
    if landing is not None:
        lts = float(landing["ts"])
        commit = str(landing.get("commit") or "")
        undone = next((d for d in later if float(d["ts"]) > lts and _is_reopen(d)), None)
        why = ""
        if commit and rolled.get(commit, 0) >= lts - 1:
            why = "guardian rollback"
        elif undone is not None:
            why = f"{undone.get('event')}:{undone.get('phase') or undone.get('by') or 'reopen'}"
        elif is_open:
            why = "open on the board"
        return {**base, "closed_ts": lts, "detail": why,
                "outcome": "landed_reopened" if why else "landed_kept"}
    rejected = next((d for d in later if d.get("event") == "item_closed"
                     and d.get("acceptance") in _ROUND_REJECTIONS), None)
    if rejected is not None and not is_open:
        return {**base, "outcome": "round_rejected", "closed_ts": float(rejected["ts"]),
                "detail": str(rejected.get("acceptance"))}
    retired = next((d for d in later if d.get("event") == "backlog_triage" and d.get("closed")
                    and d.get("verdict") != "confirmed"), None)
    if retired is not None and not is_open:
        return {**base, "outcome": "retired_later", "closed_ts": float(retired["ts"]),
                "detail": str(retired.get("verdict"))}
    if any(d.get("event") == "backlog_expired" for d in later) and not is_open:
        return {**base, "outcome": "expired"}
    if is_open or status is None:
        moves = [d for d in later if d.get("event") == "status_moved"]
        last = str(moves[-1].get("reason") or "") if moves else ""
        if status == "draft" and any(m in last for m in _SPENT_MARK):
            return {**base, "outcome": "spent"}
        return {**base, "outcome": "open", "detail": "" if status else "no board file"}
    if any(d.get("event") == "owed_check" and d.get("ok")
           and ((d.get("result") or {}).get("moved") == "closed"
                or (d.get("result") or {}).get("remaining") == 0) for d in later):
        return {**base, "outcome": "owed_closed"}
    if landed_marker:
        # Closed, and the item says it landed, with no `item_landed` row after the
        # verdict: the landing predates it or was written by a path with no row.
        return {**base, "outcome": "landed_kept", "detail": "board marker only"}
    return {**base, "outcome": "closed_other"}


# ── the board ──────────────────────────────────────────────────────────────

def _board(ids: set[int], backlog_dir: Path | None) -> dict[int, dict]:
    """`{id: {status, source, worth, size, priority, landed_marker}}` for `ids`."""
    from scripts.automod import backlog as B
    from scripts.automod import source_yield as SY
    out: dict[int, dict] = {}
    for item in B.all_items(None, backlog_dir=backlog_dir):
        if item.id not in ids:
            continue
        # No second YAML parse (0.6 s over 1,300 files): `Item` already carries the
        # tags and the body, and the landed marker is a top-level front-matter key.
        try:
            head = item.path.read_text(encoding="utf-8").split("\n---", 1)[0]
        except OSError:
            head = ""
        out[item.id] = {"status": item.status, "source": SY.source_of(item.tags, item.body),
                        "worth": item.worth or "", "size": item.size or "",
                        "priority": str(item.priority or ""),
                        "landed_marker": f"\n{B.LANDED_MARKER}:" in head}
    return out


def _tally(cells: dict, key, cls: str, outcome: str) -> None:
    names = RETIRE_OUTCOMES if cls == RETIRE else CONFIRM_OUTCOMES
    row = cells.setdefault(key, {"cls": cls, "n": 0, **{o: 0 for o in names}})
    row["n"] += 1
    row[outcome] = row.get(outcome, 0) + 1


def _finish(row: dict) -> dict:
    """Add the rates a cell supports."""
    if row["cls"] == RETIRE:
        row["reopened_rate"] = _rate(row["reopened"], row["n"])
        return row
    resolved = row["n"] - sum(row.get(o, 0) for o in UNRESOLVED)
    row["resolved"] = resolved
    row["landed_kept_rate"] = _rate(row["landed_kept"], resolved)
    row["round_rejected_rate"] = _rate(row["round_rejected"], resolved)
    return row


def judged(*, days: float = 30.0, ledger: Path | None = None,
           backlog_dir: Path | None = None, now: float | None = None,
           rows: list[dict] | None = None, since_days_ago: float | None = None,
           until_days_ago: float = 0.0) -> list[dict]:
    """Every verdict in the window with its outcome joined on, deduped to the
    first per (decider, verdict, item). The rows `compute` and `worth_hit_rate`
    count, and the ones `eval/triage_replay/build.py` picks its cases from."""
    from scripts.automod import backlog as B
    now = time.time() if now is None else now
    if rows is None:
        rows = ledger_rows(ledger)
    since = now - (days if since_days_ago is None else since_days_ago) * 86400
    until = now - until_days_ago * 86400
    found = sorted(verdicts_in(rows, since=since, until=until), key=lambda v: v["ts"])
    seen: set[tuple] = set()
    first: list[dict] = []
    for v in found:
        key = (v["decider"], v["verdict"], v["item_id"])
        if key not in seen:
            seen.add(key)
            first.append(v)
    by_item, rolled = timelines(rows)
    board = _board({v["item_id"] for v in first}, backlog_dir)
    for v in first:
        b = board.get(v["item_id"]) or {}
        v["source"] = b.get("source", "unknown")
        v["status"] = b.get("status")
        v.update(outcome_after(v, by_item.get(v["item_id"], []), rolled,
                               status=b.get("status"),
                               landed_marker=bool(b.get("landed_marker")),
                               open_statuses=tuple(B.OPEN_STATUSES)))
    return first


def compute(*, days: float = 30.0, ledger: Path | None = None,
            backlog_dir: Path | None = None, now: float | None = None,
            rows: list[dict] | None = None) -> dict[str, Any]:
    """`{"days", "verdicts", "by_decider": {"<decider>/<verdict>": cell},
    "by_source": {"<source>/<class>": cell}, "deciders": {decider: {confirm,
    retire}}}`. Cheap on the cached ledger: one pass over the rows and one read
    of each ruled-on item's file."""
    rows_out = judged(days=days, ledger=ledger, backlog_dir=backlog_dir, now=now, rows=rows)
    by_decider: dict[str, dict] = {}
    by_source: dict[str, dict] = {}
    pooled: dict[str, dict] = {}
    counted: set[tuple[int, str]] = set()
    for v in rows_out:
        if v["cls"] == OTHER:
            continue
        cls = CONFIRM if v["cls"] == FOLD else v["cls"]
        _tally(by_decider, f"{v['decider']}/{v['verdict']}", cls, v["outcome"])
        # One item, one row per class here: single triage confirming an item and
        # the sweep keeping it are two deciders but one item of that source.
        if (v["item_id"], cls) not in counted:
            counted.add((v["item_id"], cls))
            _tally(by_source, f"{v['source']}/{cls}", cls, v["outcome"])
        # Pooled per decider: a held-or-not confirm, a keep and a keep_low are one
        # decision to let the item live. Folds stay out — their fate is the umbrella's.
        if v["cls"] != FOLD:
            _tally(pooled, f"{v['decider']}/{cls}", cls, v["outcome"])
    deciders: dict[str, dict] = {}
    for key, cell in pooled.items():
        name, _, cls = key.partition("/")
        deciders.setdefault(name, {})[cls] = _finish(cell)
    return {"days": days, "verdicts": len(rows_out), "min_n": MIN_N,
            "by_decider": {k: _finish(c) for k, c in sorted(by_decider.items())},
            "by_source": {k: _finish(c) for k, c in sorted(by_source.items())},
            "deciders": dict(sorted(deciders.items()))}


# ── worth / size hit rate ──────────────────────────────────────────────────

def worth_hit_rate(*, days: float = 60.0, min_age_days: float = 0.0,
                   ledger: Path | None = None, backlog_dir: Path | None = None,
                   now: float | None = None, rows: list[dict] | None = None) -> dict[str, Any]:
    """Outcome rates per `worth` and per `size` label for the sweep's keeps.

    Two denominators, because `rank_key` itself puts `high` first: `ranked` is
    every item the sweep labelled, `attempted` only those a round started on
    afterwards. A label's landed share over `ranked` mostly measures the order
    the label caused; over `attempted` it measures whether the label says
    anything about the item."""
    keeps = [v for v in judged(days=days, until_days_ago=min_age_days, ledger=ledger,
                               backlog_dir=backlog_dir, now=now, rows=rows)
             if v["decider"] == "sweep" and v["cls"] == CONFIRM and v.get("worth")]

    def table(field: str, levels: tuple[str, ...]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for level in levels:
            vs = [v for v in keeps if v.get(field) == level]
            tried = [v for v in vs if v["attempts"] > 0 or v["outcome"].startswith("landed")]
            landed = [v for v in vs if v["outcome"] == "landed_kept"]
            resolved = [v for v in vs if v["outcome"] not in UNRESOLVED]
            waits = sorted((v["closed_ts"] - v["ts"]) / 86400 for v in landed if v["closed_ts"])
            n_landed = len(landed)
            out[level] = {
                "ranked": len(vs), "attempted": len(tried), "resolved": len(resolved),
                "outcomes": {o: sum(1 for v in vs if v["outcome"] == o) for o in CONFIRM_OUTCOMES},
                "landed_kept_of_ranked": _rate(n_landed, len(vs)),
                "landed_kept_of_attempted": _rate(
                    sum(1 for v in tried if v["outcome"] == "landed_kept"), len(tried)),
                "unwanted_of_resolved": _rate(
                    sum(1 for v in resolved
                        if v["outcome"] in ("round_rejected", "retired_later", "expired")),
                    len(resolved)),
                "reopened_of_landed": _rate(
                    sum(1 for v in vs if v["outcome"] == "landed_reopened"),
                    sum(1 for v in vs if v["outcome"].startswith("landed"))),
                "attempts_per_landing": (round(sum(v["attempts"] for v in tried) / n_landed, 2)
                                         if n_landed >= MIN_N else None),
                "refusals_per_landing": (round(sum(v["refusals"] for v in tried) / n_landed, 2)
                                         if n_landed >= MIN_N else None),
                "median_days_to_land": (round(waits[len(waits) // 2], 2)
                                        if len(waits) >= MIN_N else None),
            }
        return out

    worth, size = table("worth", WORTH_LEVELS), table("size", SIZE_LEVELS)
    pairs = {}
    for metric in ("landed_kept_of_attempted", "unwanted_of_resolved", "reopened_of_landed"):
        pairs[metric] = {f"{a}_vs_{b}": separated(worth[a][metric], worth[b][metric])
                         for a, b in (("high", "medium"), ("medium", "low"), ("high", "low"))}
    return {"days": days, "min_age_days": min_age_days, "ranked": len(keeps),
            "worth": worth, "size": size, "worth_separated": pairs}


# ── output ─────────────────────────────────────────────────────────────────

def _cell(r: dict | None) -> str:
    if not r or r.get("rate") is None:
        return f"— ({r['k']}/{r['n']})" if r else "—"
    lo, hi = r["ci"]
    return f"{r['rate'] * 100:.0f}% ({r['k']}/{r['n']}; {lo * 100:.0f}–{hi * 100:.0f})"


def _render_cells(title: str, cells: dict[str, dict]) -> list[str]:
    lines = [title, "",
             "| key | n | landed-kept | landed-reopened | round-rejected | retired-later "
             "| expired | owed-closed | spent | open | closed-other | landed-kept of resolved |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, c in cells.items():
        if c["cls"] == RETIRE:
            continue
        lines.append(f"| {key} | {c['n']} | {c['landed_kept']} | {c['landed_reopened']} "
                     f"| {c['round_rejected']} | {c['retired_later']} | {c['expired']} "
                     f"| {c['owed_closed']} | {c['spent']} | {c['open']} | {c['closed_other']} "
                     f"| {_cell(c['landed_kept_rate'])} |")
    lines += ["", "| key | n | stayed closed | reopened | reopened rate |", "|---|---|---|---|---|"]
    for key, c in cells.items():
        if c["cls"] == RETIRE:
            lines.append(f"| {key} | {c['n']} | {c['stayed_closed']} | {c['reopened']} "
                         f"| {_cell(c['reopened_rate'])} |")
    return lines


def render(reading: dict) -> str:
    lines = [f"Verdict outcomes — verdicts of the last {reading['days']:g} d "
             f"({reading['verdicts']} distinct item verdicts; rates need n ≥ {reading['min_n']}; "
             f"95 % Wilson interval in brackets)", ""]
    lines += _render_cells("By decider and verdict", reading["by_decider"])
    lines += [""] + _render_cells("By inflow source", reading["by_source"])
    return "\n".join(lines)


def render_worth(reading: dict) -> str:
    lines = [f"Worth / size hit rate — sweep keeps of the last {reading['days']:g} d, "
             f"at least {reading['min_age_days']:g} d old ({reading['ranked']} ranked)", ""]
    for field in ("worth", "size"):
        lines += [f"| {field} | ranked | attempted | landed-kept of ranked | landed-kept of attempted "
                  "| unwanted of resolved | reopened of landed | attempts/landing "
                  "| review refusals/landing | median d to land |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for level, r in reading[field].items():
            lines.append(
                f"| {level} | {r['ranked']} | {r['attempted']} | {_cell(r['landed_kept_of_ranked'])} "
                f"| {_cell(r['landed_kept_of_attempted'])} | {_cell(r['unwanted_of_resolved'])} "
                f"| {_cell(r['reopened_of_landed'])} | {r['attempts_per_landing'] or '—'} "
                f"| {r['refusals_per_landing'] if r['refusals_per_landing'] is not None else '—'} "
                f"| {r['median_days_to_land'] if r['median_days_to_land'] is not None else '—'} |")
        lines.append("")
    lines.append("Worth pairs whose 95 % intervals do not overlap: " + (
        "; ".join(f"{m}: {', '.join(k for k, v in p.items() if v)}"
                  for m, p in reading["worth_separated"].items() if any(p.values())) or "none"))
    return "\n".join(lines)


def scorecard_summary(reading: dict) -> dict[str, dict]:
    """Row 23's payload: per decider, the two rates and their n."""
    out: dict[str, dict] = {}
    for name, cells in reading.get("deciders", {}).items():
        c, r = cells.get(CONFIRM), cells.get(RETIRE)
        out[name] = {"confirm_landed_kept": c["landed_kept_rate"] if c else None,
                     "confirm_round_rejected": c["round_rejected_rate"] if c else None,
                     "retire_reopened": r["reopened_rate"] if r else None}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="What became of the items each decider ruled on")
    ap.add_argument("--days", type=float, default=30.0)
    ap.add_argument("--worth", action="store_true", help="the worth/size hit-rate table instead")
    ap.add_argument("--min-age-days", type=float, default=0.0,
                    help="--worth: only keeps at least this old (lets outcomes mature)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.worth:
        reading = worth_hit_rate(days=args.days, min_age_days=args.min_age_days)
        print(json.dumps(reading, indent=2) if args.json else render_worth(reading))
    else:
        reading = compute(days=args.days)
        print(json.dumps(reading, indent=2) if args.json else render(reading))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
