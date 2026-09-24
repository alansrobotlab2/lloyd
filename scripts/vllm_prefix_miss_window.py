"""The counted prefix-miss reading of `architecture/vllm.md` §10, re-derivable.

§10's only reading was a spot check dated 2026-09-11, taken off the dashboard
and a five-minute in-memory ring, so nothing could say whether a later day was
better or worse. This joins the three surfaces that do keep history, read-only:

* `usage.db` — one row per turn, `prefix_misses` / `reprefill_tokens`
  (`app/prefix_miss.py`); `ts` is UTC.
* `event_logs/*.events.jsonl` — one `brain1.prefix_miss` per miss iteration,
  stamped UTC at the iteration's END with its `duration_ms`.
* the primary engine's own status lines (`agent-llm-primary.log*`, every 10 s
  while it has work): `GPU KV cache usage` and `Running: N reqs`. They are
  stamped `MM-DD HH:MM:SS` in LOCAL time with no year, so the year and zone are
  parameters, not guesses buried in a regex.

`engine_pressure`'s ring cannot stand in for the third: it is a deque in the
backend process, gone at every restart. Nothing else records KV over time.

    python -m scripts.vllm_prefix_miss_window --start 2026-09-23 --end 2026-09-24 \\
        [--write-extract tests/fixtures/vllm_prefix_miss_2026-09-23.json]

`extract` reads the live data; `derive` turns an extract into every number §10
quotes, and `tests/test_vllm_doc_claims.py` runs it over the committed extract.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import statistics
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ENGINE_TZ = "America/Los_Angeles"
# Engine samples kept before the window's start, so the first misses' gaps
# have their lines.
LOOKBACK_S = 3600
STATUS_INTERVAL_S = 10

_STATUS_RE = re.compile(
    r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) \[loggers\.py:\d+\] Engine 000: "
    r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, "
    r"Running: (\d+) reqs, .*?GPU KV cache usage: ([\d.]+)%")
_POOL_RE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")


def session_kind(sid: str) -> str:
    sid = str(sid or "")
    if sid.startswith("task:"):
        return "task"
    parts = sid.split("_")
    return parts[2] if len(parts) >= 4 else "chat"


def _epoch(s: str) -> float:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def extract(start: str, end: str, *, data_root: Path, year: int | None = None,
            engine_tz: str = ENGINE_TZ) -> dict:
    """Everything `derive` needs for [start, end) UTC: turns and misses in
    the window, and engine samples from the earliest miss gap onwards."""
    t0, t1 = _epoch(start), _epoch(end)
    year = year or datetime.fromtimestamp(t0, timezone.utc).year
    db = sqlite3.connect(f"file:{data_root / 'usage.db'}?mode=ro", uri=True)
    turns = [[r[0], session_kind(r[1]), r[2], r[3]] for r in db.execute(
        "SELECT ts, session_id, prefix_misses, reprefill_tokens FROM usage "
        "WHERE ts >= ? AND ts < ? ORDER BY ts",
        (datetime.fromtimestamp(t0, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
         datetime.fromtimestamp(t1, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")))]
    misses = []
    for path in sorted((data_root / "event_logs").glob("*.events.jsonl")):
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if '"brain1.prefix_miss"' not in line:
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                t = _epoch(ev["ts"])
                if t0 <= t < t1:
                    d = ev.get("data") or {}
                    misses.append([round(t, 3), int(d.get("duration_ms") or 0),
                                   int(d.get("iteration") or 0), int(d.get("input_tokens") or 0),
                                   int(d.get("cache_read") or 0), session_kind(ev.get("session_id")),
                                   str(ev.get("turn_id") or "")])
    # The gap a miss's prefix sat unreferenced in: from the previous
    # iteration's last proposed tool call (its request had finished and its
    # blocks were free) to this iteration's request.
    calls: dict[str, list[float]] = {m[6]: [] for m in misses if m[6]}
    for path in sorted((data_root / "event_logs").glob("*.events.jsonl")):
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if '"brain1.tool_call_proposed"' not in line:
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if ev.get("turn_id") in calls:
                    calls[ev["turn_id"]].append(_epoch(ev["ts"]))
    for m in misses:
        began = m[0] - m[1] / 1000
        prior = [c for c in calls.get(m[6], ()) if c < began]
        m[6] = round(max(prior), 3) if prior else None
    misses.sort()
    tz = ZoneInfo(engine_tz)
    samples, pools = {}, set()
    lo = min([t0 - LOOKBACK_S] + [m[6] for m in misses if m[6]])
    for path in sorted((data_root / "logs" / "services").glob("agent-llm-primary.log*")):
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if (p := _POOL_RE.search(line)):
                    pools.add(int(p.group(1).replace(",", "")))
                m = _STATUS_RE.search(line)
                if not m:
                    continue
                local = datetime.strptime(f"{year}-{m.group(1)}", "%Y-%m-%d %H:%M:%S")
                t = local.replace(tzinfo=tz).timestamp()
                if lo <= t < t1 + STATUS_INTERVAL_S:
                    # [t, KV %, running, tokens the engine computed in the
                    # 10 s this line reports on: prompt + generated]
                    samples[t] = [int(t), round(float(m.group(5)), 1), int(m.group(4)),
                                  round((float(m.group(2)) + float(m.group(3))) * STATUS_INTERVAL_S)]
    return {"window": [start, end], "engine_tz": engine_tz, "year": year,
            "pool_tokens": sorted(pools), "turns": turns, "misses": misses,
            "kv_samples": [samples[k] for k in sorted(samples)]}


def _pct(values: list[float], q: float) -> float:
    """Nearest-rank percentile: a value that was actually observed."""
    s = sorted(values)
    return s[max(0, min(len(s) - 1, math.ceil(len(s) * q) - 1))] if s else float("nan")


def derive(ex: dict, *, gate: float) -> dict:
    t0, t1 = _epoch(ex["window"][0]), _epoch(ex["window"][1])
    turns = ex["turns"]
    measured = [t for t in turns if t[2] is not None]
    missing = [t for t in measured if (t[2] or 0) > 0]
    by_kind: dict[str, list[int]] = {}
    for t in missing:
        k = by_kind.setdefault(t[1], [0, 0])
        k[0] += t[2]
        k[1] += t[3] or 0
    samples = ex["kv_samples"]
    in_window = [s for s in samples if t0 <= s[0] < t1]
    kv = [s[1] / 100 for s in in_window]

    def between(a: float, b: float) -> list[list]:
        return [s for s in samples if a <= s[0] <= b]

    # One pool for the whole window, or the free-pool arithmetic below is
    # against the wrong denominator for part of it.
    (pool,) = ex["pool_tokens"]
    during, gap_kv, gaps, churned = [], [], [], 0
    for m in ex["misses"]:
        start = m[0] - m[1] / 1000
        d = between(start, m[0] + STATUS_INTERVAL_S)
        if d:
            during.append(max(s[1] for s in d) / 100)
        if m[6] is None:
            continue
        # Status lines report on the 10 s BEFORE their stamp, so the gap's
        # lines run to one interval past its end.
        g = between(m[6], start + STATUS_INTERVAL_S)
        if not g:
            continue
        gaps.append(start - m[6])
        peak = max(s[1] for s in g) / 100
        gap_kv.append(peak)
        # LRU: a freed prefix is reclaimed once new allocations have gone
        # through every free block ahead of it. Tokens computed in the gap
        # against the free pool at its tightest is the most it could need.
        if sum(s[3] for s in g) >= (1 - peak) * pool:
            churned += 1
    return {
        "turns": len(turns),
        "measured": len(measured),
        "turns_with_misses": len(missing),
        "misses": sum(t[2] for t in missing),
        "reprefill_tokens": sum(t[3] or 0 for t in missing),
        "worst_turn_tokens": max((t[3] or 0 for t in missing), default=0),
        "by_kind": {k: v for k, v in sorted(by_kind.items(), key=lambda kv_: -kv_[1][1])},
        "miss_events": len(ex["misses"]),
        "cold_events": sum(1 for m in ex["misses"] if m[4] == 0),
        "kv_samples": len(kv),
        "kv_p50": _pct(kv, 0.5), "kv_p90": _pct(kv, 0.9), "kv_max": max(kv, default=0.0),
        "running_p50": _pct([s[2] for s in in_window], 0.5),
        "pool_tokens": pool,
        "misses_with_gap": len(gap_kv),
        "gap_s_p50": round(_pct(gaps, 0.5), 1) if gaps else None,
        "miss_kv_during_p50": _pct(during, 0.5), "miss_kv_during_p90": _pct(during, 0.9),
        "miss_kv_gap_p50": _pct(gap_kv, 0.5), "miss_kv_gap_p90": _pct(gap_kv, 0.9),
        "miss_kv_gap_max": max(gap_kv, default=0.0),
        "misses_gap_over_gate": sum(1 for v in gap_kv if v >= gate),
        "misses_gap_over_90": sum(1 for v in gap_kv if v >= 0.9),
        "misses_gap_churned_free_pool": churned,
        "median_turn_misses": statistics.median([t[2] for t in missing]) if missing else 0,
    }


def main(argv: list[str] | None = None) -> int:
    from app.paths import PRODUCTION_DATA_ROOT
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True, help="UTC, inclusive (ISO date or datetime)")
    ap.add_argument("--end", required=True, help="UTC, exclusive")
    ap.add_argument("--data-root", type=Path, default=PRODUCTION_DATA_ROOT)
    ap.add_argument("--gate", type=float, default=0.60)
    ap.add_argument("--write-extract", type=Path)
    a = ap.parse_args(argv)
    ex = extract(a.start, a.end, data_root=a.data_root)
    if a.write_extract:
        a.write_extract.write_text(json.dumps(ex, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps(derive(ex, gate=a.gate), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
