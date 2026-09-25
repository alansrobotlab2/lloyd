#!/usr/bin/env python3
"""#1484 — would mid-turn recall driven by the tool stream have anticipated anything?

Offline replay, no model. For recorded turns with at least ``--min-calls`` tool
calls, every ``--every`` calls the probe builds the query the proposal would —
the model's own captions (`summary`) for the last ``--every`` calls plus the
basenames of the paths they touched — runs prefetch's hybrid vault leg on it
(`prefetch._search_vault`, the same ~55 ms call), and keeps up to 3 hits the turn
has not already seen (its turn-start `<vault-context>`, anything it read, earlier
anchors), exactly the ephemeral `<context>` the proposal would append.

The score is **anticipation**: a hit counts as useful when a LATER tool call in
the same turn names that document (its vault path or file name in the call's
arguments) — the model went and fetched it anyway, so appending it earlier could
have saved the fetch. The reference rate is the same measure over the turn-start
prefix's own vault hits, which production already injects.

What it cannot see: a hit the model would have used had it been shown, but never
fetched on its own. So anticipation is a floor on usefulness, and the reference
row carries the same floor.

Usage:
    .venvs/lloyd/bin/python eval/run_midturn_recall_probe.py \\
        --sessions-dir ~/lloyd-data/sessions --out ~/lloyd-data/eval/1484/probe.json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LLOYD_HOME = HERE.parent
sys.path.insert(0, str(LLOYD_HOME))
sys.path.insert(0, str(HERE))

from run_prefetch_eval import _quiet_logging  # noqa: E402

with _quiet_logging():
    import prefetch  # noqa: E402
    from agent_mcp.transcript_self_hit import drop_self_hits  # noqa: E402

ANCHOR_HITS = 3
_PATH_TOKEN = re.compile(r"[\w.~/-]+\.(?:md|py|ts|tsx|yaml|yml|json|sh)\b")
_VC_FILE = re.compile(r"file: ([^,)\s]+)")


def _args(tc: dict) -> dict:
    try:
        return json.loads((tc.get("function") or {}).get("arguments") or "{}")
    except ValueError:
        return {}


def _text(m: dict) -> str:
    c = m.get("content", "")
    if isinstance(c, list):
        c = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
    return c if isinstance(c, str) else ""


def turns(data: dict) -> list[dict]:
    """[{prefix_files, calls: [(name, args_json, caption)]}] per user turn."""
    out, cur = [], None
    for m in data.get("messages", []):
        role = m.get("role")
        if role == "user":
            cur = {"prefix_files": [], "calls": []}
            out.append(cur)
        elif cur is None:
            continue
        elif role == "subliminal":
            cur["prefix_files"] += _VC_FILE.findall(_text(m))
        elif role == "assistant":
            for tc in m.get("tool_calls") or []:
                a = _args(tc)
                cap = tc.get("summary") or a.get("summary") or ""
                cur["calls"].append(((tc.get("function") or {}).get("name", ""),
                                     json.dumps(a), str(cap)))
    return out


def _key(path: str) -> str:
    """A document's findable name: its file name without the extension — with its
    folder when the name alone is a date or too short to be specific, so a daily
    note is not "named" by every command that prints today's date."""
    p = Path(path.removeprefix("qmd://"))
    stem = p.stem.lower()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", stem) or len(stem) < 8:
        return f"{p.parent.name}/{stem}".lower()
    return stem


def _named_later(key: str, calls: list) -> bool:
    return bool(key) and any(key in args.lower() for _, args, _ in calls)


def probe_turn(sid: str, turn: dict, every: int) -> dict:
    calls = turn["calls"]
    seen = {_key(f) for f in turn["prefix_files"]}
    anchors = []
    for i in range(every, len(calls), every):
        window = calls[i - every:i]
        for _, args, _ in calls[:i]:
            seen |= {_key(p) for p in _PATH_TOKEN.findall(args)}
        caps = [c for _, _, c in window if c]
        names = sorted({Path(p).name for _, a, _ in window for p in _PATH_TOKEN.findall(a)})
        query = " ".join(caps + names)[:600]
        if len(query) < prefetch.VAULT_MIN_QUERY_LEN:
            continue
        t0 = time.perf_counter()
        hits = prefetch._search_vault(query, None, legs=("lex", "vec"))
        ms = (time.perf_counter() - t0) * 1000
        hits, _ = drop_self_hits(query, hits, sid)
        new = [h for h in hits if _key(h.get("file", "")) not in seen][:ANCHOR_HITS]
        seen |= {_key(h.get("file", "")) for h in new}
        later = calls[i:]
        anchors.append({
            "at_call": i, "query": query[:200], "ms": round(ms, 1),
            "hits": [h.get("file") for h in new],
            "used_later": [h.get("file") for h in new if _named_later(_key(h.get("file", "")), later)],
            "chars": sum(len(h.get("title", "")) + len(h.get("snippet", "")) + 60 for h in new),
        })
    prefix_used = [f for f in turn["prefix_files"] if _named_later(_key(f), calls)]
    return {"n_calls": len(calls), "anchors": anchors,
            "prefix_hits": len(turn["prefix_files"]), "prefix_used_later": len(prefix_used)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions-dir", required=True)
    ap.add_argument("--max-sessions", type=int, default=60)
    ap.add_argument("--min-calls", type=int, default=20)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from stats import wilson_ci

    rows = []
    files = sorted(Path(args.sessions_dir).expanduser().glob("*.json"), reverse=True)
    with _quiet_logging():
        for f in files:
            if len(rows) >= args.max_sessions:
                break
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            long_turns = [t for t in turns(data) if len(t["calls"]) >= args.min_calls]
            if not long_turns:
                continue
            t = max(long_turns, key=lambda t: len(t["calls"]))
            rows.append({"session": f.stem, **probe_turn(f.stem, t, args.every)})
    anchors = [a for r in rows for a in r["anchors"]]
    hits = sum(len(a["hits"]) for a in anchors)
    used = sum(len(a["used_later"]) for a in anchors)
    ph = sum(r["prefix_hits"] for r in rows)
    pu = sum(r["prefix_used_later"] for r in rows)
    summ = {
        "turns": len(rows), "anchors": len(anchors),
        "anchors_with_new_hits": sum(1 for a in anchors if a["hits"]),
        "hits_appended": hits, "hits_used_later": used,
        "anticipation": round(used / hits, 3) if hits else None,
        "anticipation_wilson95": [round(x, 3) for x in wilson_ci(used, hits)] if hits else None,
        "turns_with_any_used_hit": sum(1 for r in rows if any(a["used_later"] for a in r["anchors"])),
        "prefix_reference": {"hits": ph, "used_later": pu,
                             "rate": round(pu / ph, 3) if ph else None,
                             "wilson95": [round(x, 3) for x in wilson_ci(pu, ph)] if ph else None},
        "chars_appended_per_turn_mean": round(statistics.mean(
            [sum(a["chars"] for a in r["anchors"]) for r in rows]), 1) if rows else 0,
        "query_ms_p50": round(statistics.median([a["ms"] for a in anchors]), 1) if anchors else None,
    }
    Path(args.out).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).expanduser().write_text(json.dumps({"summary": summ, "turns": rows}, indent=1))
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
