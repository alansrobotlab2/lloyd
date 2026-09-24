"""Read-only replay of the reconciler's needs-human hand-offs over the live
automod ledger: would `desired_statuses` as it stands in this tree still have
made each one? `needs-human-handoffs-2026-09-24.md` is the result.

Nothing is written outside a temp directory. The live ledger and the item
files under ~/obsidian/backlog (tags/members/group/board only) are read; the
reconciler runs over a ledger PREFIX (every row before the hand-off) against a
one-item scratch board, with every state reader it consults pointed at a proxy
built from that prefix:

  worktree_path(rid) exists  <=> round_start(rid) <= t and no close event
                                 (round_aborted / round_abandoned / promoted /
                                 land_abandoned) at or before t
  gate/land markers, current.json -> absent (a gate or a landing needs an
                                 open round, so the proxy above covers them)
  open deferral targets       -> a stub on the scratch board for each target
                                 open at t (not `done` today, or `completed`
                                 after t)

The pre-fix module runs beside it, so a hand-off the old rule no longer
reproduces (the code or the ledger shape has moved since) is flagged.

  (git show <pre-fix rev>:scripts/automod/backlog.py) > /tmp/backlog_old.py
  .venvs/lloyd/bin/python eval/measurements/needs_human_handoff_replay.py 7 /tmp/backlog_old.py [OUT_DIR]
"""
from __future__ import annotations

import collections
import importlib.util
import json
import os
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

SCRATCH = Path(tempfile.mkdtemp(prefix="handoff-replay-"))
os.environ["LLOYD_AUTOMOD_STATE"] = str(SCRATCH / "state")
(SCRATCH / "state").mkdir(exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import yaml  # noqa: E402
from scripts.automod import backlog as B, state as S, worktree as W  # noqa: E402

DAYS = float(sys.argv[1]) if len(sys.argv) > 1 else 7
spec = importlib.util.spec_from_file_location("backlog_old", sys.argv[2])
OLD = importlib.util.module_from_spec(spec)
sys.modules["backlog_old"] = OLD
spec.loader.exec_module(OLD)
OUT_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else SCRATCH

LIVE = Path.home() / ".local/state/lloyd-automod/promotions.jsonl"
VAULT = Path.home() / "obsidian/backlog"
rows = [json.loads(l) for l in LIVE.read_text().splitlines() if l.strip()]
now = rows[-1]["ts"]
HUMAN = ("a human decides", "every implement turn this item got failed on the stack")


def is_handoff(r):
    return (r.get("event") == "status_moved" and r.get("to") == "draft"
            and any(h in (r.get("reason") or "") for h in HUMAN))


byitem = collections.defaultdict(list)
for i, r in enumerate(rows):
    try:
        byitem[int(r["item_id"])].append((i, r))
    except (KeyError, TypeError, ValueError):
        pass

round_open: dict[str, float] = {}
round_close: dict[str, float] = {}
CLOSE = {"round_aborted", "round_abandoned", "promoted", "land_abandoned"}
for r in rows:
    rid = r.get("round_id")
    if not rid:
        continue
    if r.get("event") == "round_start":
        round_open.setdefault(rid, r["ts"])
    elif r.get("event") in CLOSE:
        round_close.setdefault(rid, r["ts"])


def reversal(i, iid):
    for j, x in byitem[iid]:
        if j <= i:
            continue
        e = x.get("event")
        if e == "backlog_retriage":
            return "retriage", x["ts"]
        if e == "backlog_umbrella_unfolded":
            return "unfold", x["ts"]
        if e == "round_abandoned":
            # `reap_abandoned_rounds` then set_status(up_next) with no ledger row
            return "reaper:up_next(unrecorded)", x["ts"]
        if e == "backlog_implement" and x.get("phase") == "reopened":
            return "reopen_item", x["ts"]
        if e == "backlog_implement" and x.get("phase") == "started":
            return "round_restart", x["ts"]
        if e == "status_moved" and x.get("to") in ("up_next", "in_progress"):
            return "moved:" + x["to"], x["ts"]
        if e in ("item_closed", "item_landed") or (e == "status_moved" and x.get("to") == "done"):
            return "closed", x["ts"]
        if is_handoff(x):
            return "re-handoff", x["ts"]
    return None, None


def item_file(iid, status):
    src = next(iter(sorted(VAULT.glob(f"{iid}-*.md"))), None)
    fm = {}
    if src is not None:
        text = src.read_text(encoding="utf-8")
        if text.startswith("---"):
            try:
                fm = yaml.safe_load(text.split("---", 2)[1]) or {}
            except Exception:  # noqa: BLE001
                fm = {}
    keep = {k: fm[k] for k in ("tags", "members", "group", "board", "priority") if k in fm}
    tags = [t for t in (keep.get("tags") or []) if t not in (B.NEEDS_HUMAN_TAG, B.RETRIAGE_TAG)]
    keep["tags"] = tags or ["backlog"]
    keep["status"] = status
    return f"---\n{yaml.dump(keep)}---\n\n# item {iid}\n"


def fm_of(i):
    f = next(iter(sorted(VAULT.glob(f"{i}-*.md"))), None)
    if not f:
        return {}
    try:
        return yaml.safe_load(f.read_text().split("---", 2)[1]) or {}
    except Exception:  # noqa: BLE001
        return {}


def open_at(i, T):
    """Open at T: not `done` today, or closed (its `completed`) after T."""
    fm = fm_of(i)
    if fm.get("status") != "done":
        return True
    try:
        return datetime.fromisoformat(str(fm.get("completed"))).replace(
            tzinfo=timezone.utc).timestamp() > T
    except ValueError:
        return False


def deferral_targets(prefix, iid):
    rows_ = B.implement_history(prefix).get(iid) or []
    last = next((r for r in reversed(rows_) if r.get("phase") == "finished"), None)
    oc = (last or {}).get("outcome")
    if not isinstance(oc, dict):
        return []
    t = B._ints(oc.get("deferred_to"))
    for c in oc.get("clause_outcomes") or []:
        if isinstance(c, dict):
            t += B._ints(c.get("deferred_to"))
    return [x for x in dict.fromkeys(t) if x != iid]


def run(mod, prefix: Path, iid: int, t: float, board: Path, alive_on: bool):
    wt_yes, wt_no = board.parent / "wt_yes", board.parent / "wt_no"
    wt_yes.mkdir(exist_ok=True)

    def wpath(rid):
        o, c = round_open.get(rid), round_close.get(rid)
        live = alive_on and o is not None and o <= t and (c is None or c > t)
        return wt_yes if live else wt_no
    W.worktree_path = wpath
    S.gate_in_progress = lambda rid: None
    S.land_in_progress = lambda rid: None
    S.read_current = lambda: None
    mod.BACKLOG_DIR = board
    return mod.desired_statuses(prefix, None).get(iid)


def main():
    hand = [(i, r) for i, r in enumerate(rows) if is_handoff(r) and r["ts"] >= now - DAYS * 86400]
    out = []
    with tempfile.TemporaryDirectory(dir=SCRATCH) as td:
        td = Path(td)
        board = td / "backlog"
        board.mkdir()
        for i, r in hand:
            iid = int(r["item_id"])
            for f in board.glob("*.md"):
                f.unlink()
            (board / f"{iid}-item-{i}.md").write_text(item_file(iid, r.get("from") or "in_progress"))
            prefix = td / f"prefix_{i}.jsonl"
            prefix.write_text("\n".join(json.dumps(x) for x in rows[:i]) + "\n")
            # The deferral targets that were still open at t get a stub on the
            # scratch board, so `open_items(None)` answers as the board did.
            for tgt in deferral_targets(prefix, iid):
                if open_at(tgt, r["ts"]):
                    (board / f"{tgt}-stub-{i}.md").write_text(
                        "---\nstatus: draft\ntags: [backlog]\n---\n\n# stub\n")
            old = run(OLD, prefix, iid, r["ts"], board, alive_on=False)
            new = run(B, prefix, iid, r["ts"], board, alive_on=True)
            prefix.unlink()
            how, when = reversal(i, iid)
            new_nh = bool(new and len(new) > 2 and new[2])
            if new is None:
                cls = "no-opinion"
            elif new_nh:
                cls = "still-handed-off"
            elif new[0] == "in_progress":
                cls = "held:round-alive"
            elif new[0] == "draft":
                cls = "held:second-life-owed"
            else:
                cls = "other:" + new[0]
            out.append({"i": i, "item": iid, "ts": r["ts"], "how": how,
                        "dt_min": (when - r["ts"]) / 60 if when else None,
                        "old_reproduces": bool(old and len(old) > 2 and old[2]),
                        "fixed": cls, "fixed_why": (new or ("", ""))[1][:90]})
    iso = lambda ts: datetime.fromtimestamp(ts, timezone.utc).strftime("%m-%dT%H:%M:%SZ")
    print(f"{len(out)} hand-offs in {DAYS:g} d; old code reproduces "
          f"{sum(o['old_reproduces'] for o in out)}")
    rev = [o for o in out if o["how"] not in (None, "re-handoff")]
    print("reversed:", len(rev), "median min",
          round(statistics.median([o["dt_min"] for o in rev]), 1) if rev else None)
    tab = collections.Counter((o["how"], o["fixed"]) for o in out)
    for k, v in sorted(tab.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        print(f"  {v:4d}  reversal={k[0]!s:18} fixed={k[1]}")
    print("fixed:", collections.Counter(o["fixed"] for o in out))
    for o in out:
        print(o["item"], iso(o["ts"]), o["how"],
              round(o["dt_min"], 1) if o["dt_min"] is not None else "", o["fixed"],
              "" if o["old_reproduces"] else "(old-does-not-reproduce)", o["fixed_why"])
    (OUT_DIR / f"replay_{DAYS:g}d.json").write_text(json.dumps(out, indent=1))


main()
