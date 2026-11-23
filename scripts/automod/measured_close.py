"""Measured close: the measurement, not the review alone, settles an improvement.

An item triaged `class: improvement` with a `metric:` naming a fitness bench
(`eval/fitness/manifest.yaml`) claims it will move that bench. The review rung
grades whether the change did what its clauses say; it cannot run the bench.
So when such an item lands `met`, it owes one more thing — the measurement —
and this module settles that debt with no model session:

    ruling(facts, fitness_rows, noise) -> met | rejected | pending

* `met`      — the first fitness row at or after the landing moved in the right
               direction, outside the noise floor, against the last row before it.
* `rejected` — it did not. A clean close ("tried, did not measurably help"):
               the numbers go on the item, and ONE follow-up item proposes
               reverting that landing. Never an automatic revert: nothing inside
               the blast radius performs a rollback, and this is not a rollback
               request either — it is a backlog item a round may take or triage
               may retire.
* `pending`  — no row yet (or no noise floor, so no row could ever rule). Stays
               owed, bounded: after `MAX_PENDING_DAYS` it is released with a
               recorded `unmeasured` note. Nothing parks.

The seam, and what it deliberately is not. The item still closes `done` on the
review, exactly as today: FIVE writers reach `done` on a met landing
(`owed.hand_run_pending`'s docstring enumerates them) and holding the status
open would mean a hook in each. Instead the debt rides the machinery that
already carries "landed, closed, still owes a post-landing check": an entry on
the item's `owed` list, plus a `measured_close:` front-matter block that is this
module's own state. The entry is dated past the bound so the owed-check SESSION
does not rule on it first; if this pass never runs (the switch turned off with
entries outstanding) the entry falls due and owed-check rules on it like any
other — the debt can not be stranded.

No landing writer is touched. `settle_pass` finds its candidates by joining the
ledger's `item_landed` rows to the board, so it opens the debt on its first
pass after a landing instead of from inside the landing. The one hook is
`workers/sources/owed_check.py::enqueue_if_due`, off the event loop, throttled
to one pass per `TICK_EVERY_S`.

Ships off: `automod.landing.measured_close: off`.

    python -m scripts.automod.measured_close report    # what it WOULD rule; writes nothing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVENT = "measured_close"
#: The item's own record of the measurement: `{bench, commit, landed_at, verdict, …}`.
FM_KEY = "measured_close"
MET, REJECTED, PENDING, UNMEASURED, OPENED = "met", "rejected", "pending", "unmeasured", "opened"
TERMINAL = (MET, REJECTED, UNMEASURED)

DAY_S = 86400.0
#: A landing with no ruling after this many days is released `unmeasured`. Two
#: weeks: the nightly run measures every bench daily, so fourteen nights without
#: one usable row means the instrument is off or the bench has no floor, and
#: waiting longer would not change that.
MAX_PENDING_DAYS = 14.0
#: The owed entry is dated this far past the bound, so this pass gets the release
#: before the owed-check session sees the entry as due.
OWED_GRACE_DAYS = 1.0
#: The owed-check tick calls `tick()` every interval; the board join runs this often.
TICK_EVERY_S = 900.0
#: Revert proposals one pass may file.
PROPOSALS_PER_PASS = 1

_last_tick = 0.0


def enabled(cfg: Mapping[str, Any] | None = None) -> bool:
    """`automod.landing.measured_close`. YAML reads a bare `on`/`off` as a
    boolean, so both spellings are accepted; anything else is off."""
    if cfg is None:
        try:
            from scripts.automod import state as S
            cfg = S.landing_cfg()
        except Exception:  # noqa: BLE001 — unreadable config is the shipped default
            return False
    value = (cfg or {}).get("measured_close")
    return value is True or str(value).strip().lower() == "on"


def owed_text(bench: str, commit: str) -> str:
    """The owed entry. Text is the identity `owed.add_owed` dedupes on, so it is
    built in one place; it names this file so `backlog.names_a_demand` keeps it
    and so a session that does meet it knows not to rule early."""
    return (f"Metric `{bench}` must be measured after landing {str(commit)[:8]}: "
            f"scripts/automod/measured_close.py settles this from the first fitness row at "
            f"or after that commit. Until one exists, answer `recheck`")


# ── the ruling (pure) ────────────────────────────────────────────────────────

def _facts_of(row: Mapping[str, Any]) -> dict:
    return {k: row.get(k) for k in ("value", "n", "ci_low", "ci_high", "commit",
                                    "recorded_at", "run_id", "ts")}


def ruling(facts: Mapping[str, Any], rows: Sequence[dict], noise: Mapping[str, Any], *,
           contains: Callable[[Mapping[str, Any]], bool] | None = None) -> dict:
    """`{verdict, why, pre, post, delta, noise_spread, direction}` for one landing.

    `facts` is `{bench, commit, landed_ts}`. `pre` is the newest measured row
    before `landed_ts`; `post` the OLDEST measured row at or after it for which
    `contains(row)` holds (the row's code includes the landing; None = judge by
    time alone). The first row, not the newest: every later night has more
    landings in it that are not this one.

    The bounds are `fitness.bench_delta`'s: a move must be larger than the
    measured spread AND the post row's own Wilson interval must not cover the
    pre value. `met` needs a move in the bench's good direction; anything else
    with both rows and a floor is `rejected`. `pending` is every case where the
    question cannot be asked yet.
    """
    from scripts.automod import fitness as F
    bench = str(facts.get("bench") or "")
    landed = float(facts.get("landed_ts") or 0.0)
    out: dict[str, Any] = {"verdict": PENDING, "why": "", "pre": None, "post": None,
                           "delta": None, "noise_spread": None, "direction": None,
                           "can_rule": True}
    floor = noise.get(bench) if isinstance(noise.get(bench), Mapping) else {}
    status, spread = floor.get("status") or "unmeasured", floor.get("spread")
    if status != "ok" or not isinstance(spread, (int, float)) or isinstance(spread, bool):
        out["can_rule"] = False
        out["why"] = (f"`{bench}` is flagged too-noisy" if status == "too-noisy"
                      else f"`{bench}` has no measured noise floor")
        return out
    out["noise_spread"] = float(spread)
    mine = F.measured_rows(rows, bench)
    before = [r for r in mine if float(r["ts"]) < landed]
    after = [r for r in mine if float(r["ts"]) >= landed
             and (contains is None or contains(r))]
    if before:
        out["pre"] = _facts_of(before[-1])
    if after:
        out["post"] = _facts_of(after[0])
    if not after:
        out["why"] = "no fitness row at or after the landing"
        return out
    if not before:
        out["can_rule"] = False     # no later row can supply an earlier one
        out["why"] = "no fitness row before the landing to compare with"
        return out
    pre, post = before[-1], after[0]
    delta = round(float(post["value"]) - float(pre["value"]), 6)
    higher = bool(post.get("higher_is_better", True))
    better = delta > 0 if higher else delta < 0
    lo, hi = post.get("ci_low"), post.get("ci_high")
    covered = (isinstance(lo, (int, float)) and isinstance(hi, (int, float))
               and lo <= float(pre["value"]) <= hi)
    outside = abs(delta) > float(spread) and not covered
    out["delta"] = delta
    out["direction"] = ("unchanged" if delta == 0 else "improved" if better else "regressed")
    if outside and better:
        out["verdict"], out["why"] = MET, "moved in the right direction, outside the noise floor"
    else:
        out["verdict"] = REJECTED
        out["why"] = ("moved the wrong way, outside the noise floor" if outside
                      else "the post-landing row's 95% interval covers the pre-landing value"
                      if abs(delta) > float(spread) else "the delta is inside the noise floor")
        out["outside_noise"] = outside
    return out


def describe(facts: Mapping[str, Any], r: Mapping[str, Any]) -> str:
    """One line carrying the numbers, for the item and the ledger."""
    def one(row: Mapping[str, Any] | None) -> str:
        if not row:
            return "no row"
        return (f"{float(row['value']):.3f} (n={row.get('n')}, "
                f"`{str(row.get('commit') or 'unknown')[:8]}`, "
                f"{str(row.get('recorded_at') or '')[:10]})")
    delta = "—" if r.get("delta") is None else f"{r['delta']:+.3f}"
    floor = "none" if r.get("noise_spread") is None else f"{r['noise_spread']:.3f}"
    return (f"`{facts['bench']}` after landing {str(facts['commit'])[:8]}: {r['verdict']} — "
            f"{r.get('why') or 'no reason'}; before {one(r.get('pre'))}, after "
            f"{one(r.get('post'))}, delta {delta}, noise floor {floor}")


# ── candidates (pure) ────────────────────────────────────────────────────────

def candidates(events: Iterable[Mapping[str, Any]],
               classes: Mapping[int, tuple[str, str]],
               benches: Iterable[str]) -> list[dict]:
    """One `{item_id, bench, commit, landed_ts, round_id}` per item whose NEWEST
    `item_landed` row is a `met` landing, whose class is `improvement` and whose
    metric names a manifest bench. A vault landing has no code commit to measure
    and is skipped."""
    known = set(benches)
    newest: dict[int, Mapping[str, Any]] = {}
    for e in events:
        if e.get("event") != "item_landed" or not str(e.get("item_id") or "").isdigit():
            continue
        if not isinstance(e.get("ts"), (int, float)):
            continue
        iid = int(e["item_id"])
        if iid not in newest or float(e["ts"]) >= float(newest[iid]["ts"]):
            newest[iid] = e
    out = []
    for iid, e in sorted(newest.items()):
        cls, metric = classes.get(iid, ("", ""))
        if (e.get("acceptance") != "met" or e.get("vault") or not e.get("commit")
                or cls != "improvement" or metric not in known):
            continue
        out.append({"item_id": iid, "bench": metric, "commit": str(e["commit"]),
                    "landed_ts": float(e["ts"]), "round_id": e.get("round_id")})
    return out


def state_of(fm: Mapping[str, Any], commit: str) -> str:
    """The recorded verdict for THIS landing, "" when the item carries none (or
    carries one for an earlier landing it has since been reopened past)."""
    block = fm.get(FM_KEY)
    if not isinstance(block, Mapping) or str(block.get("commit") or "") != str(commit):
        return ""
    return str(block.get("verdict") or "")


def decide(facts: Mapping[str, Any], state: str, r: Mapping[str, Any], *, now: float,
           max_pending_days: float = MAX_PENDING_DAYS) -> str:
    """What the pass does with one candidate: `skip`, `open`, `wait`, or a
    terminal verdict. Pure. A landing already older than the bound that was never
    opened is history from before the switch — left alone, not back-filled."""
    if state in TERMINAL:
        return "skip"
    age_days = (now - float(facts["landed_ts"])) / DAY_S
    if not state and age_days > max_pending_days:
        return "skip"
    if r["verdict"] in (MET, REJECTED):
        return r["verdict"]
    if age_days >= max_pending_days:
        return UNMEASURED
    return "wait" if state else "open"


# ── the revert proposal (pure) ───────────────────────────────────────────────

def proposal(facts: Mapping[str, Any], r: Mapping[str, Any]) -> dict:
    """The one follow-up a `rejected` ruling files. A landing that moved the
    bench the WRONG way outside noise makes the revert an `improvement` on that
    bench; one that moved nothing makes it a `simplify` — same behaviour, less
    code — and that is all the measurement supports."""
    sha, bench, iid = str(facts["commit"])[:8], facts["bench"], facts["item_id"]
    worse = bool(r.get("outside_noise")) and r.get("direction") == "regressed"
    name = (f"Propose reverting {sha} (#{iid}): `{bench}` "
            + ("got worse after it landed" if worse else "did not move after it landed"))
    body = "\n".join([
        f"#{iid} landed as `{sha}` claiming to improve `{bench}`. The measurement says "
        f"otherwise: {describe(facts, r)}.",
        "",
        "This is a proposal, not a rollback request: nothing was reverted. Decide whether the "
        "landing still earns its lines — it may have a second effect the bench does not see — "
        "and if not, revert it in a round (`git revert` of that one commit, with whatever "
        "later landings built on it carried forward).",
        "",
        "## Re-run", "",
        f"    python -m scripts.automod.fitness run --once --only {bench}",
        "",
        "## Acceptance clauses", "",
        (f"1. `{bench}` is back within its noise floor ({float(r['noise_spread']):.3f}) of the "
         f"{float(r['pre']['value']):.3f} it measured before `{sha}`, on a `fitness run` after landing"
         if worse else
         f"1. `{sha}`'s change is reverted (or the item is closed `rejected` with the reason "
         f"it should stay), and `{bench}` is not worse than {float(r['post']['value']):.3f} "
         f"beyond its noise floor on a `fitness run` after landing"),
        "",
        f"Filed by measured close (`scripts/automod/measured_close.py`) from #{iid}.",
    ])
    return {"name": name, "body": body, "bench": bench,
            "cls": "improvement" if worse else "simplify"}


# ── the board and git (impure) ───────────────────────────────────────────────

def make_contains(commit: str, *, repo: Path = ROOT,
                  runner: Callable[..., Any] = subprocess.run
                  ) -> Callable[[Mapping[str, Any]], bool]:
    """Whether a fitness row's code includes `commit`: the checkout the bench ran
    from (`commit`) and, for an engine bench, the backend that answered it
    (`backend_commit`) must both have the landing as an ancestor. A row with no
    recorded commit, or a git that cannot answer, is judged by time alone — an
    unanswerable question must not hold a ruling for ever."""
    memo: dict[str, bool] = {}

    def is_ancestor(descendant: str) -> bool:
        if descendant not in memo:
            try:
                done = runner(["git", "-C", str(repo), "merge-base", "--is-ancestor",
                               commit, descendant], capture_output=True, timeout=20)
                memo[descendant] = done.returncode != 1   # 1 = not an ancestor; else unknown
            except Exception:  # noqa: BLE001
                memo[descendant] = True
        return memo[descendant]

    def contains(row: Mapping[str, Any]) -> bool:
        return all(is_ancestor(str(c)) for c in (row.get("commit"), row.get("backend_commit"))
                   if c)
    return contains


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _block(facts: Mapping[str, Any], verdict: str, r: Mapping[str, Any] | None = None,
           **extra: Any) -> dict:
    block: dict[str, Any] = {"bench": facts["bench"], "commit": facts["commit"],
                             "landed_at": _iso(facts["landed_ts"]), "verdict": verdict}
    if r is not None:
        for key in ("pre", "post"):
            row = r.get(key)
            if row:
                block[key] = {k: row.get(k) for k in ("value", "n", "commit", "recorded_at")}
        for key in ("delta", "noise_spread", "direction", "why"):
            if r.get(key) not in (None, ""):
                block[key] = r[key]
    block.update({k: v for k, v in extra.items() if v not in (None, "")})
    return block


def open_debt(path: Path, facts: Mapping[str, Any], *,
              max_pending_days: float = MAX_PENDING_DAYS) -> None:
    """Record the debt: the state block, and one dated owed entry."""
    from scripts.automod import backlog as B, owed as O
    due = facts["landed_ts"] + (max_pending_days + OWED_GRACE_DAYS) * DAY_S
    B.update_frontmatter(path, {FM_KEY: _block(facts, PENDING)})
    O.add_owed(path, [owed_text(facts["bench"], facts["commit"])], kind="check",
               recheck_after=_iso(due),
               activity=f"measured close: `{facts['bench']}` owed a measurement after "
                        f"landing {facts['commit'][:8]}")


def settle_debt(path: Path, facts: Mapping[str, Any], verdict: str, r: Mapping[str, Any],
                *, follow_up: int | None = None, note: str = "") -> None:
    """Move the entry from `owed` to `owed_settled` and write the verdict block.
    One narrow write, deliberately not `owed.apply_verdict`: that writer re-dates
    every entry it was not given an answer for, and this pass has an answer for
    exactly one."""
    from scripts.automod import backlog as B, owed as O
    from app.backlog_move import now_stamp
    fm = B.front_matter_of(path) or {}
    what = owed_text(facts["bench"], facts["commit"])
    keep = [{k: v for k, v in e.items() if v not in ("", None, 0)}
            for e in O.entries_of(fm, derived=False) if e.get("what") != what]
    evidence = describe(facts, r) + (f" {note}" if note else "")
    record = {"what": what, "outcome": "settled" if verdict == MET else "ruling",
              "at": now_stamp(), "evidence": evidence[:1200]}
    if follow_up:
        record["follow_up"] = int(follow_up)
    settled = list(fm.get(O.SETTLED_KEY) or []) + [record]
    B.update_frontmatter(
        path, {O.OWED_KEY: keep or None, O.SETTLED_KEY: settled,
               FM_KEY: _block(facts, verdict, r, follow_up=follow_up, note=note)},
        activity=(f"measured close: {evidence}"
                  + (f" -> #{follow_up} proposes the revert" if follow_up else ""))[:1500])


def _board() -> tuple[dict[int, Path], dict[int, dict], list[tuple[int, dict]]]:
    """`(paths by id, front matter by id, open fronts)` in one walk."""
    from scripts.automod import backlog as B
    paths: dict[int, Path] = {}
    fronts: dict[int, dict] = {}
    open_fronts: list[tuple[int, dict]] = []
    for item in B.all_items(None):
        fm = B.front_matter_of(item.path) or {}
        paths[item.id], fronts[item.id] = item.path, fm
        if item.status in B.OPEN_STATUSES:
            open_fronts.append((item.id, fm))
    return paths, fronts, open_fronts


def _classes(fronts: Mapping[int, Mapping[str, Any]]) -> dict[int, tuple[str, str]]:
    from scripts.automod.item_class import clean_class, clean_metric
    return {iid: (clean_class(fm.get("class")), clean_metric(fm.get("metric")))
            for iid, fm in fronts.items() if fm.get("class")}


def _bench_names() -> list[str]:
    from scripts.automod import fitness as F
    return [b.name for b in F.load_manifest()]


def settle_pass(*, now: float | None = None, dry_run: bool = False,
                events: Sequence[Mapping[str, Any]] | None = None,
                rows: Sequence[dict] | None = None, noise: Mapping[str, Any] | None = None,
                benches: Sequence[str] | None = None,
                board: tuple[Mapping[int, Path], Mapping[int, Mapping[str, Any]],
                             Sequence[tuple[int, Mapping[str, Any]]]] | None = None,
                contains_for: Callable[[str], Callable[[Mapping[str, Any]], bool] | None] | None = None,
                opener: Callable[..., None] | None = None,
                settler: Callable[..., None] | None = None,
                filer: Callable[..., int] | None = None,
                append: Callable[[dict], None] | None = None,
                max_pending_days: float = MAX_PENDING_DAYS) -> list[dict]:
    """One pass over every landed improvement item. Returns one row per candidate
    it did (or, `dry_run`, would do) something with; each row that is not `wait`
    or `skip` is also appended to the ledger as a `measured_close` event.
    """
    from scripts.automod import fitness as F, fitness_intake as FI, state as S
    now = time.time() if now is None else float(now)
    events = list(S.ledger_rows()) if events is None else list(events)
    rows = F.read_rows() if rows is None else rows
    noise = F.read_noise() if noise is None else noise
    benches = _bench_names() if benches is None else benches
    paths, fronts, open_fronts = _board() if board is None else board
    contains_for = contains_for or make_contains
    open_benches = FI.open_bench_items(open_fronts)
    proposals = 0
    out: list[dict] = []
    for facts in candidates(events, _classes(fronts), benches):
        iid = facts["item_id"]
        path, fm = paths.get(iid), fronts.get(iid) or {}
        if path is None:
            continue
        r = ruling(facts, rows, noise, contains=contains_for(facts["commit"]))
        state = state_of(fm, facts["commit"])
        action = decide(facts, state, r, now=now, max_pending_days=max_pending_days)
        row: dict[str, Any] = {"event": EVENT, "item_id": iid, "bench": facts["bench"],
                               "commit": facts["commit"], "round_id": facts.get("round_id"),
                               "verdict": action, "why": r.get("why"), "delta": r.get("delta"),
                               "noise_spread": r.get("noise_spread"),
                               "value": (r.get("post") or {}).get("value"),
                               "reference_value": (r.get("pre") or {}).get("value"),
                               "summary": describe(facts, r)}
        if action in ("skip", "wait"):
            if dry_run and action == "wait":
                out.append(row)
            continue
        if dry_run:
            out.append(row)
            continue
        if not state:
            # Opened on the pass that first sees the landing, whatever is ruled next:
            # the `opened` row is what the scorecard counts as "landed".
            (opener or open_debt)(path, facts, max_pending_days=max_pending_days)
            (append or S.append_event)({**row, "verdict": OPENED})
        if action == "open":
            out.append({**row, "verdict": OPENED})
            continue
        follow_up, note = None, ""
        if action == UNMEASURED:
            note = (f"Released unmeasured after {max_pending_days:g} d: "
                    f"{r.get('why') or 'no ruling was possible'}.")
        if action == REJECTED:
            holders = open_benches.get(facts["bench"]) or []
            if holders:
                note = (f"No revert proposal filed: open item #{holders[0]} already names "
                        f"`{facts['bench']}`.")
            elif proposals >= PROPOSALS_PER_PASS:
                note = "No revert proposal filed: this pass already filed one."
            else:
                p = proposal(facts, r)
                follow_up = int((filer or FI.file_item)(p["name"], p["body"],
                                                        bench=p["bench"], cls=p["cls"]))
                proposals += 1
                open_benches.setdefault(facts["bench"], []).append(follow_up)
        (settler or settle_debt)(path, facts, action, r, follow_up=follow_up, note=note)
        row.update({"follow_up": follow_up, "note": note or None})
        (append or S.append_event)(row)
        out.append(row)
    return out


def tick(*, apply: bool = True, now: float | None = None,
         monotonic: Callable[[], float] = time.monotonic) -> list[dict]:
    """The owed-check tick's hook: nothing unless the switch is on, the source
    applies its answers, and `TICK_EVERY_S` has passed since the last pass."""
    global _last_tick
    if not apply or not enabled():
        return []
    stamp = monotonic()
    if _last_tick and stamp - _last_tick < TICK_EVERY_S:
        return []
    _last_tick = stamp
    return settle_pass(now=now)


# ── scorecard row 25 ─────────────────────────────────────────────────────────

def scorecard_block(events: Iterable[Mapping[str, Any]], since: float) -> dict:
    """`{landed, met, rejected, unmeasured, pending, proposals}`. `landed`, the
    three verdicts and `proposals` are counted over `measured_close` events at or
    after `since`; `pending` is every landing opened at any time with no verdict
    yet. All zero with `tracked: False` when the switch has never written a row."""
    opened: set[tuple[Any, Any]] = set()
    done: set[tuple[Any, Any]] = set()
    counts = {MET: 0, REJECTED: 0, UNMEASURED: 0}
    landed = proposals = 0
    tracked = False
    for e in events:
        if e.get("event") != EVENT:
            continue
        tracked = True
        key = (e.get("item_id"), e.get("commit"))
        verdict = e.get("verdict")
        recent = isinstance(e.get("ts"), (int, float)) and float(e["ts"]) >= since
        if verdict == OPENED:
            opened.add(key)
            landed += 1 if recent else 0
        elif verdict in counts:
            done.add(key)
            if recent:
                counts[verdict] += 1
                proposals += 1 if e.get("follow_up") else 0
    return {"tracked": tracked, "landed": landed, **counts,
            "pending": len(opened - done), "proposals": proposals}


def render_scorecard_row(block: Mapping[str, Any] | None) -> str:
    if not isinstance(block, Mapping) or not block.get("tracked"):
        return ("| 25 | measured closes | — | no improvement landing has been measured "
                "(`automod.landing.measured_close` off, or none landed) |")
    return (f"| 25 | measured closes | {block.get('met', 0)} met / {block.get('rejected', 0)} "
            f"rejected of {block.get('landed', 0)} landed | improvement items with a manifest "
            f"`metric`: {block.get('pending', 0)} pending a fitness row, "
            f"{block.get('unmeasured', 0)} released unmeasured, "
            f"{block.get('proposals', 0)} revert proposal(s) filed |")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m scripts.automod.measured_close",
        description="What measured close would rule for every landed improvement item. "
                    "Writes nothing, whatever the switch says.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("report")
    rp.add_argument("--json", action="store_true")
    rp.add_argument("--max-pending-days", type=float, default=MAX_PENDING_DAYS)
    args = ap.parse_args(argv)
    out = settle_pass(dry_run=True, max_pending_days=args.max_pending_days)
    if args.json:
        print(json.dumps(out, indent=2, default=str))
        return 0
    print(f"measured close is {'ON' if enabled() else 'off'} "
          f"(automod.landing.measured_close); {len(out)} landing(s) it would act on")
    for row in out:
        print(f"  #{row['item_id']:<5} {row['verdict']:10s} {row['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
