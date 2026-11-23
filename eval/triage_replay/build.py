#!/usr/bin/env python3
"""Build `cases.json`: past triage decisions whose outcome is known.

A hand tool. It reads production — the automod ledger and the vault's git
history — and writes one JSON file next to itself; nothing else. Re-run it to
refresh the set; the runner (`run.py`) and its tests only read the file.

**The label is what happened, never what a triage model thought.** Two kinds:

    good_confirm   single triage confirmed it; a round landed it, the item
                   closed, and 14+ days later nothing has reopened it. Items
                   whose name says they only witness, record or pin an earlier
                   item's claim are left out (`backlog.is_derived_name`): the
                   loop lands its own bookkeeping as readily as anything else,
                   and "it landed" is no evidence such an item was wanted.
                   Taken round-robin across inflow sources.
    bad_confirm    single triage confirmed it and a PERSON later closed it as
                   superseded because the problem it described did not occur
                   or had gone. Hand-audited ids (`HUMAN_SUPERSEDED`); the
                   builder takes the reason from the ledger row and refuses an
                   id whose row is not there.

**What is not in the set, and why — the audit of 2026-11-22.** The obvious
negative labels do not survive being read:

* *confirmed, then closed `unnecessary`/`rejected` by its round* — 42 single-
  triage confirmations in 80 days. Read one by one, about 20 say the change was
  complete and never landed (a gate still running at the turn's end, an
  external red tree, a held promotion), 11 say the acceptance was already true
  on main — and in every one of those that names the fixing commit, that commit
  landed 0.1 to 6 hours AFTER the triage verdict (a race with another round or
  a hand merge, not a wrong verdict) — 4 carry no reason, one is a measured
  proposal rejection whose premise was true, one a premise that expired after
  triage. None is a triage error the item's text could have shown.
* *retired, then reopened* — 1 of 895 retirements in 80 days, and that one
  (#4432) is open only because its close was held; its own body says
  "WITHDRAWN — not a blocker". The retirement was right.
* *retired and stayed closed* is not a label at all: nobody reopening an item
  is not evidence the retirement was right.

So the record supports one class well (1,996 candidates) and the other barely
(2). The runner prints `unmeasured` for a label under its minimum n, and a
one-sided set is still worth having for one question — does a change to the
triage prompt stop confirming work that landed and stayed landed? — but it
cannot say whether triage confirms too much. That needs hand labels.

**The text is the item as triage saw it**: the file at the last vault commit
before the verdict's timestamp. When the item was filed and triaged between two
vault commits there is no such version; the earliest committed one is used and
the case says so (`text_as_of: "earliest"`), because that version may already
carry the findings triage appended.

    .venvs/lloyd/bin/python eval/triage_replay/build.py            # from ~/lloyd
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

LLOYD_HOME = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(LLOYD_HOME))

CASES_PATH = Path(__file__).resolve().parent / "cases.json"

#: A landing has to have stood this long before it is called kept.
KEPT_DAYS = 14.0
WINDOW_DAYS = 80.0
PER_LABEL = 40

#: Single-triage confirmations a person later closed as superseded: the problem
#: did not occur in practice (#443) or had already gone (#550). Audited by hand
#: on 2026-11-22 from the 41 human closes that followed a model's confirm; the
#: other 39 were landings, rulings on a clause, or declined options.
HUMAN_SUPERSEDED = (443, 550)
BODY_CHARS = 30_000


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else ""


def text_as_of(vault: Path, item_id: int, ts: float) -> tuple[str, str, str] | None:
    """`(text, vault sha, "before_verdict" | "earliest")` for one item."""
    spec = f"backlog/{int(item_id)}-*.md"
    sha = _git(vault, "log", "-1", "--format=%H", f"--before=@{int(ts)}", "--", spec).strip()
    how = "before_verdict"
    if not sha:
        shas = _git(vault, "log", "--format=%H", "--reverse", "--", spec).split()
        if not shas:
            return None
        sha, how = shas[0], "earliest"
    names = [n for n in _git(vault, "ls-tree", "--name-only", sha, "backlog/").splitlines()
             if n.startswith(f"backlog/{int(item_id)}-") and n.endswith(".md")]
    if not names:
        return None
    text = _git(vault, "show", f"{sha}:{names[0]}")
    return (text, sha, how) if text else None


def item_fields(text: str, item_id: int) -> dict | None:
    """The fields the triage prompt reads, parsed by the board's own loader."""
    from scripts.automod import backlog as B
    with tempfile.TemporaryDirectory(prefix="triage-replay-") as tmp:
        (Path(tmp) / f"{int(item_id)}-case.md").write_text(text, encoding="utf-8")
        items = B.all_items(None, backlog_dir=Path(tmp))
    if not items:
        return None
    i = items[0]
    return {"name": i.name, "priority": str(i.priority or ""), "created": str(i.created or ""),
            "tags": [str(t) for t in i.tags], "parent": i.parent, "worth": i.worth or "",
            "size": i.size or "", "body": (i.body or "")[:BODY_CHARS]}


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _case(vault: Path, label: str, v: dict, row: dict, outcome: dict, source: str,
          *, allow_earliest: bool) -> dict | None:
    got = text_as_of(vault, v["item_id"], v["ts"])
    if got is None:
        return None
    text, vault_sha, how = got
    if how == "earliest" and not allow_earliest:
        return None
    fields = item_fields(text, v["item_id"])
    if fields is None or not fields["body"].strip():
        return None
    tree = _git(LLOYD_HOME, "rev-list", "-1", f"--before=@{int(v['ts'])}", "HEAD").strip()
    return {
        "key": f"{label}-{v['item_id']}", "item_id": v["item_id"], "label": label,
        "expected": "confirmed" if label == "good_confirm" else "not_confirmed",
        "source": source,
        "decision": {"decider": "triage", "verdict": "confirmed", "at": _iso(v["ts"]),
                     "session_id": str(row.get("session_id") or ""),
                     "evidence": " ".join(str(row.get("evidence") or "").split())[:500]},
        "outcome": outcome, "text_as_of": how, "vault_sha": vault_sha, "tree_as_of": tree,
        "age_days": max(0, int((v["ts"] - _created_ts(fields["created"], v["ts"])) // 86400)),
        "item": fields}


def build(*, vault: Path, now: float | None = None, per_label: int = PER_LABEL) -> dict:
    from scripts.automod import backlog as B
    from scripts.automod import verdict_outcomes as VO
    now = time.time() if now is None else now
    rows = VO.ledger_rows()
    by_item, _ = VO.timelines(rows)
    triage_rows = {(int(d["item_id"]), float(d["ts"])): d for d in rows
                   if d.get("event") == "backlog_triage" and d.get("item_id") is not None}
    kept_before = now - KEPT_DAYS * 86400
    pool = [v for v in VO.judged(days=WINDOW_DAYS, rows=rows, now=now)
            if v["decider"] == "triage" and v["verdict"] == "confirmed"
            and v["outcome"] == "landed_kept" and not v["detail"]
            and v["closed_ts"] and v["closed_ts"] <= kept_before]
    skipped = {"no_clean_text": 0, "derived": 0}
    by_source: dict[str, list[dict]] = {}
    for v in sorted(pool, key=lambda v: v["item_id"]):
        by_source.setdefault(v["source"], []).append(v)
    cases: list[dict] = []
    # Round-robin over sources, and within a source an even stride over the
    # window rather than the newest N: the prompt and the board both changed.
    queues = {}
    quota = -(-per_label // max(1, len(by_source)))
    for name, vs in by_source.items():
        stride = max(1, len(vs) // quota)
        queues[name] = vs[::stride] + [p for n, p in enumerate(vs) if n % stride]
    good = 0
    while good < per_label and any(queues.values()):
        for name in sorted(queues):
            if good >= per_label or not queues[name]:
                continue
            v = queues[name].pop(0)
            seq = by_item.get(v["item_id"], [])
            landing = next(x for x in seq if float(x["ts"]) > v["ts"]
                           and x.get("event") == "item_landed" and x.get("closed"))
            outcome = {"name": "landed_kept",
                       "evidence": (f"item_landed {_iso(landing['ts'])} commit "
                                    f"{str(landing.get('commit'))[:12]} round "
                                    f"{landing.get('round_id')}, closed; nothing has reopened "
                                    f"it in the {int((now - landing['ts']) // 86400)} d since")}
            case = _case(vault, "good_confirm", v, triage_rows.get((v["item_id"], v["ts"]), {}),
                         outcome, name, allow_earliest=False)
            if case is None:
                skipped["no_clean_text"] += 1
            elif B.is_derived_name(case["item"]["name"]):
                skipped["derived"] += 1
            else:
                cases.append(case)
                good += 1
    for item_id in HUMAN_SUPERSEDED:
        seq = by_item.get(item_id, [])
        close = next((x for x in seq if x.get("event") == "item_closed" and x.get("by") == "human"), None)
        confirm = next((x for x in seq if x.get("event") == "backlog_triage"
                        and x.get("verdict") == "confirmed"
                        and x.get("verdict_source") != "human"), None)
        if close is None or confirm is None or float(confirm["ts"]) >= float(close["ts"]):
            raise SystemExit(f"#{item_id}: the ledger no longer shows a model confirm followed "
                             f"by a human close — re-audit HUMAN_SUPERSEDED")
        v = {"item_id": item_id, "ts": float(confirm["ts"])}
        outcome = {"name": "human_superseded",
                   "evidence": f"item_closed {_iso(close['ts'])} by human: "
                               + " ".join(str(close.get("reason") or "").split())[:500]}
        case = _case(vault, "bad_confirm", v, confirm, outcome, "audited", allow_earliest=True)
        if case is None:
            raise SystemExit(f"#{item_id}: no text recoverable from the vault's history")
        cases.append(case)
    cases.sort(key=lambda c: c["key"])
    # A quoted `workers/queue.py:<line>` is a citation the tree bans in every tracked
    # path (`tests/test_egress_window_unit.py`): it rots. The file name stays.
    cases = json.loads(re.sub(r"(workers/queue\.py):\d+(?:-\d+)?", r"\1",
                              json.dumps(cases, ensure_ascii=False)))
    return {"built_at": _iso(now), "window_days": WINDOW_DAYS, "kept_days": KEPT_DAYS,
            "available": {"good_confirm": len(pool), "bad_confirm": len(HUMAN_SUPERSEDED)},
            "skipped": skipped, "cases": cases}


def _created_ts(created: str, fallback: float) -> float:
    from scripts.automod import backlog as B
    ts = B._iso_ts(created, legacy_local=True)
    return fallback if ts is None else ts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the triage replay set from production history")
    ap.add_argument("--vault", type=Path, default=None, help="the vault checkout (default: ~/obsidian)")
    ap.add_argument("--out", type=Path, default=CASES_PATH)
    ap.add_argument("--per-label", type=int, default=PER_LABEL)
    args = ap.parse_args(argv)
    from scripts.automod import backlog as B
    vault = args.vault or Path(B.BACKLOG_DIR).parent
    doc = build(vault=vault, per_label=args.per_label)
    args.out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    by = {}
    for c in doc["cases"]:
        by[c["label"]] = by.get(c["label"], 0) + 1
    print(f"{len(doc['cases'])} cases → {args.out}: {by}; available {doc['available']}; "
          f"skipped {doc['skipped']}; "
          f"{sum(1 for c in doc['cases'] if c['text_as_of'] == 'earliest')} use the earliest text")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
