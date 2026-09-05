#!/usr/bin/env python3
"""Inner Voice — retrospective grading over recorded observations.

Read-only analysis. Nothing in the chat path depends on this, and it must
stay that way: the observer's job is to watch the primary, not to be
watched synchronously by a third thing.

The subsystem records everything needed to judge itself — every decision,
its trigger, its content, its cost, and the session it belongs to — and
until now nothing read any of it back. That left no answer to the two
questions that decide whether Inner Voice is worth its cost:

  * PRECISION — when it intervened, did the primary actually change course?
  * RECALL    — when it stayed silent at the end of a turn, did the user
                immediately have to correct the agent?

Both are proxies, and they are labelled as proxies in the output. An
inject "landed" if the primary's next assistant message did something
other than repeat itself; a terminal noop "missed" if the user's very next
message reads like a correction. Neither is ground truth. They are good
enough to spot a regression and to compare two prompts, which is what the
tuning loop actually needs.

Usage:
    python scripts/iv_grade.py                    # all sessions
    python scripts/iv_grade.py --session <id>
    python scripts/iv_grade.py --since 2026-08-01
    python scripts/iv_grade.py --json             # machine-readable
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_LLOYD_HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_LLOYD_HOME))

DB_PATH = _LLOYD_HOME / "usage.db"
SESSIONS_DIR = _LLOYD_HOME / "sessions"

INTERVENTIONS = ("inject", "cancel", "ambient", "clarify")

# A row whose decision came from a deterministic guard rather than the model.
# These cost no LLM call, so `_was_llm_call` already excludes them from the
# cost tables — which also made them invisible in the per-trigger interventions
# column, and `pretool` vanished from the report entirely even on the evening it
# fired 19 injects. They also trivially satisfy the precision proxy: they only
# ever fire mid-turn, so the loop always continues afterwards, and 19 false
# positives read as a 1.0 landed rate. Score them apart.
_DETERMINISTIC_REASON_RE = re.compile(r"^(?:deterministic:|fast-path:)", re.IGNORECASE)


def _is_deterministic(row: dict) -> bool:
    return bool(_DETERMINISTIC_REASON_RE.match(row.get("reason") or ""))

# The user's next message reading as a correction is the strongest cheap
# signal that a terminal noop was wrong. Deliberately narrow: a follow-up
# question is normal conversation, not a correction.
_CORRECTION_RE = re.compile(
    r"\b(?:you (?:didn'?t|did not|never|forgot|missed|skipped)"
    r"|that'?s (?:not|wrong|incorrect)|not what i (?:asked|wanted|meant)"
    r"|try again|you were supposed to|i asked (?:you )?(?:to|for)"
    r"|still (?:broken|failing|not working|wrong)|actually,? no"
    r"|finish (?:it|the)|incomplete|you stopped)\b",
    re.IGNORECASE,
)


def _rows(conn: sqlite3.Connection, where: str, params: list) -> list[dict]:
    conn.row_factory = sqlite3.Row
    sql = f"""SELECT id, session_id, turn_id, sequence_in_turn, trigger, action,
                     reason, content, related_tool, input_tokens, output_tokens,
                     cache_read, latency_ms, model, error, created_at
              FROM inner_voice_observations {where} ORDER BY id"""
    return [dict(r) for r in conn.execute(sql, params)]


def _session_messages(session_id: str) -> list[dict]:
    p = SESSIONS_DIR / f"{session_id}.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text()).get("messages") or []
    except Exception:
        return []


def _text_of(msg: dict) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    out = []
    for chunk in content or []:
        if isinstance(chunk, dict) and chunk.get("type") == "text":
            out.append(chunk.get("text") or "")
    return "".join(out)


def _grade_injects(rows: list[dict]) -> dict[str, Any]:
    """Did an inject change what the primary did next?

    Proxy: an inject that is followed, later in the same turn, by at least
    one more assistant_message decision means the loop continued and the
    primary read the nudge. An inject with nothing after it in the turn
    means the turn ended anyway — the nudge bought nothing.
    """
    by_turn: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_turn[r["turn_id"]].append(r)

    landed = stranded = 0
    deterministic = 0
    deterministic_turns: Counter = Counter()
    stranded_examples: list[dict] = []
    for turn_id, turn_rows in by_turn.items():
        turn_rows.sort(key=lambda r: r["sequence_in_turn"])
        for i, r in enumerate(turn_rows):
            if r["action"] != "inject":
                continue
            if _is_deterministic(r):
                # A guard inject always fires mid-turn, so "the loop continued"
                # is true by construction and says nothing about whether the
                # nudge was right. Count them; do not score them.
                deterministic += 1
                deterministic_turns[turn_id] += 1
                continue
            later = turn_rows[i + 1:]
            continued = any(x["trigger"] == "assistant_message" for x in later)
            if continued:
                landed += 1
            else:
                stranded += 1
                if len(stranded_examples) < 5:
                    stranded_examples.append({
                        "turn_id": turn_id,
                        "trigger": r["trigger"],
                        "reason": (r["reason"] or "")[:110],
                    })
    total = landed + stranded
    worst = deterministic_turns.most_common(3)
    return {
        "injects": total,
        "landed": landed,
        "stranded": stranded,
        "landed_rate": round(landed / total, 3) if total else None,
        "stranded_examples": stranded_examples,
        "deterministic_injects": deterministic,
        # More than a couple of guard injects in one turn is the shape of a
        # miscalibrated guard, not of a primary in trouble.
        "deterministic_worst_turns": [
            {"turn_id": t, "injects": n} for t, n in worst
        ],
    }


def _grade_terminal_noops(rows: list[dict]) -> dict[str, Any]:
    """Did a turn the observer signed off on draw a correction?

    Proxy for recall. Looks at the `result`-trigger decision for each turn
    and asks whether the user's next message reads like a correction.
    """
    by_session: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["trigger"] == "result":
            by_session[r["session_id"]].append(r)

    checked = missed = 0
    missed_examples: list[dict] = []
    for session_id, result_rows in by_session.items():
        msgs = _session_messages(session_id)
        if not msgs:
            continue
        user_texts = [
            (m.get("timestamp") or "", _text_of(m))
            for m in msgs
            if m.get("role") == "user" and not str(m.get("source") or "").startswith("inner_voice_")
        ]
        for r in result_rows:
            if r["action"] not in ("noop", "acknowledge_complete"):
                continue
            ts = r["created_at"] or ""
            nxt = next((t for stamp, t in user_texts if stamp > ts), None)
            if nxt is None:
                continue
            checked += 1
            if _CORRECTION_RE.search(nxt):
                missed += 1
                if len(missed_examples) < 5:
                    missed_examples.append({
                        "session_id": session_id,
                        "turn_id": r["turn_id"],
                        "observer_reason": (r["reason"] or "")[:90],
                        "user_said": nxt.strip()[:130],
                    })
    return {
        "terminal_noops_with_a_following_user_message": checked,
        "followed_by_correction": missed,
        "miss_rate": round(missed / checked, 3) if checked else None,
        "miss_examples": missed_examples,
    }


def _was_llm_call(r: dict) -> bool:
    """Did this row cost a round-trip?

    Keyed on measured spend rather than on the reason text. Reason
    prefixes have drifted (`fast-path:`, `Fast-path:`, and since v5
    `observation-only:` for pretool rows written with no call at all), and
    a text match silently reclassified whole triggers as expensive.
    """
    return bool(r.get("input_tokens") or r.get("latency_ms"))


def _cost(rows: list[dict]) -> dict[str, Any]:
    turns = {r["turn_id"] for r in rows}
    llm = [r for r in rows if _was_llm_call(r)]
    in_tok = sum(r["input_tokens"] or 0 for r in rows)
    cached = sum(r["cache_read"] or 0 for r in rows)
    by_trigger: dict[str, dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "in_tok": 0, "ms": 0, "interventions": 0, "guard": 0}
    )
    for r in llm:
        b = by_trigger[r["trigger"]]
        b["calls"] += 1
        b["in_tok"] += r["input_tokens"] or 0
        b["ms"] += r["latency_ms"] or 0
        if r["action"] in INTERVENTIONS:
            b["interventions"] += 1
    # Guard interventions cost no LLM call, so they are absent from `llm` —
    # which is why `pretool` disappeared from this table entirely on the
    # evening the repetition guard fired 19 times. Count them separately so a
    # trigger that is spending nothing but acting a lot is still visible.
    for r in rows:
        if r["action"] in INTERVENTIONS and _is_deterministic(r):
            by_trigger[r["trigger"]]["guard"] += 1
    return {
        "observations": len(rows),
        "turns": len(turns),
        "llm_calls": len(llm),
        "fast_path_share": round(1 - len(llm) / len(rows), 3) if rows else None,
        "input_tokens": in_tok,
        "cached_input_tokens": cached,
        "cache_hit_rate": round(cached / in_tok, 3) if in_tok else None,
        "input_tokens_per_turn": round(in_tok / len(turns)) if turns else 0,
        "observer_ms_per_turn": round(
            sum(r["latency_ms"] or 0 for r in llm) / len(turns)
        ) if turns else 0,
        "by_trigger": {k: dict(v) for k, v in sorted(by_trigger.items())},
        "models": dict(Counter(r["model"] or "?" for r in rows)),
        "errors": dict(Counter(
            (r["error"] or "").split(":")[0] for r in rows if r["error"]
        )),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", help="restrict to one session id")
    ap.add_argument("--since", help="ISO date lower bound on created_at")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args()

    if not DB_PATH.exists():
        print(f"no usage db at {DB_PATH}", file=sys.stderr)
        return 1

    where, params = [], []
    if args.session:
        where.append("session_id = ?")
        params.append(args.session)
    if args.since:
        where.append("created_at >= ?")
        params.append(args.since)
    clause = (" WHERE " + " AND ".join(where)) if where else ""

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    rows = _rows(conn, clause, params)
    if not rows:
        print("no observations matched")
        return 0

    report = {
        "scope": {
            "session": args.session or "all",
            "since": args.since or "all time",
            "first": rows[0]["created_at"],
            "last": rows[-1]["created_at"],
        },
        "cost": _cost(rows),
        "precision_proxy": _grade_injects(rows),
        "recall_proxy": _grade_terminal_noops(rows),
        "action_mix": dict(Counter(r["action"] for r in rows).most_common()),
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    c, p, r = report["cost"], report["precision_proxy"], report["recall_proxy"]
    print(f"\nInner Voice grading — {report['scope']['session']}, "
          f"{report['scope']['first'][:10]} to {report['scope']['last'][:10]}")
    print("=" * 66)
    print(f"\nCOST  {c['turns']} turns, {c['observations']} observations, "
          f"{c['llm_calls']} LLM calls")
    print(f"  fast-path share        {c['fast_path_share']}")
    print(f"  input tokens / turn    {c['input_tokens_per_turn']:,}")
    print(f"  observer ms / turn     {c['observer_ms_per_turn']:,}")
    print(f"  cache hit rate         {c['cache_hit_rate']}"
          f"{'   (0.0 means vLLM lacks --enable-prompt-tokens-details)' if not c['cache_hit_rate'] else ''}")
    print(f"  served by              {c['models']}")
    if c["errors"]:
        print(f"  errors                 {c['errors']}")
    print("\n  by trigger:")
    print(f"    {'trigger':<20}{'calls':>7}{'interv':>8}{'guard':>7}"
          f"{'in_tok':>12}{'tok/interv':>12}")
    for trig, b in c["by_trigger"].items():
        per = f"{b['in_tok'] // b['interventions']:,}" if b["interventions"] else "—"
        print(f"    {trig:<20}{b['calls']:>7}{b['interventions']:>8}"
              f"{b.get('guard', 0):>7}{b['in_tok']:>12,}{per:>12}")

    print(f"\nPRECISION PROXY  (did an inject keep the primary working?)")
    print(f"  model-judged injects   {p['injects']}")
    print(f"  loop continued after   {p['landed']}")
    print(f"  turn ended anyway      {p['stranded']}")
    print(f"  landed rate            {p['landed_rate']}")
    for ex in p["stranded_examples"]:
        print(f"    stranded [{ex['trigger']}] {ex['reason']}")
    print(f"  guard injects          {p['deterministic_injects']}"
          f"   (not scored — they always fire mid-turn, so 'the loop "
          f"continued' is true by construction)")
    for t in p["deterministic_worst_turns"]:
        if t["injects"] >= 3:
            print(f"    !! {t['injects']} guard injects in turn {t['turn_id']} "
                  f"— check the guard, not the primary")

    print(f"\nRECALL PROXY  (did a signed-off turn draw a correction?)")
    print(f"  terminal noops checked {r['terminal_noops_with_a_following_user_message']}")
    print(f"  followed by correction {r['followed_by_correction']}")
    print(f"  miss rate              {r['miss_rate']}")
    for ex in r["miss_examples"]:
        print(f"    missed: observer said {ex['observer_reason']!r}")
        print(f"            user then said {ex['user_said']!r}")

    print("\nBoth rates are PROXIES, not ground truth — see the module docstring.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
