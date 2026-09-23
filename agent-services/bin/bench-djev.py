#!/usr/bin/env python
"""Measure djev's structured-decision server, and keep the numbers the design
rests on reproducible.

Four tables, each one a claim `architecture/djev.md` makes:

  headline   a 3-question ticket warm and from an idle GPU, and one ~8.7k-token
             state cold and warm. The "it is faster than the GB10 reference"
             row.
  batching   1/3/6/12/24 questions in one request. The "≈33 ms fixed + ~1.3 ms
             per extra decision" row, which is why a sweep batch and a clause
             set are one request each.
  prefill    the same question against states of 0.5k/2k/8.7k/22k tokens, cold
             and warm, with the implied prefill rate. The cost model.
  capacity   N candidates scored listwise in one canvas, reporting `label_mass`
             and the server's own chunk split. This is the table the ≤12
             default and the 16 ceiling come from, and the only one whose
             failure mode is invisible without diagnostics: above 32 questions
             the canvas splits into separate shared contexts and the returned
             probabilities still look confident.

Why this file exists: the harness that produced the first set of these numbers
lived in a session scratchpad and no copy survived it, so the design's whole
evidence base was unreproducible the day after it was written. Same reason
`bench-admission-stall.py` exists, and this sits beside it.

COLD IS THE DEFAULT AND IT IS NOT FREE. vLLM serves djev with
`--enable-prefix-caching`, so the second read of one state is 20-40x the first.
Every "cold" figure here is taken by prepending a nonce to the state, which is
the only honest way to get one on a warm engine; a bench that reused its state
would report the cache and call it the model.

Run it with nothing else on GPU 2 — it waits for an idle engine and refuses
otherwise, like its neighbour. Nothing in production calls djev on a schedule,
so "idle" is the normal state rather than something to arrange.

    .venvs/lloyd/bin/python agent-services/bin/bench-djev.py all
    .venvs/lloyd/bin/python agent-services/bin/bench-djev.py headline --idle-probe
    .venvs/lloyd/bin/python agent-services/bin/bench-djev.py capacity --max-n 64
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import string
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app import vllm_metrics  # noqa: E402
from app.paths import SERVICE_LOGS_DIR  # noqa: E402

STRUCTURED = os.environ.get("BENCH_DJEV_URL", "http://127.0.0.1:8011")
ENGINE = os.environ.get("BENCH_DJEV_ENGINE", "http://127.0.0.1:8010")
DEFAULT_OUT = SERVICE_LOGS_DIR / "djev-bench.jsonl"

# Upstream's own GB10 DGX Spark figures, for the comparison column. They are
# the reason the headline table exists: this is an SM86 card running the
# Marlin W4A16 path, and the expectation going in was that it would be slower.
UPSTREAM_GB10_MS = {
    "ticket_warm": 104.3,
    "state_8k_cold": 5420.0,
    "state_8k_warm": 140.0,
}

TICKET = (
    "Subject: cannot log in\n\n"
    "Since the deploy this morning every request to the API returns 500 and "
    "none of our users can sign in. The status page still says operational. "
    "We are losing checkout revenue every minute this is down. Please treat "
    "this as an emergency."
)

TICKET_QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Is this ticket urgent?"},
    "team": {"type": "choice", "instructions": "Which team owns this?",
             "criteria": {"billing": "payments, invoices and refunds",
                          "infra": "servers, deploys and outages",
                          "design": "visual design and copy"}},
    "severity": {"type": "score", "instructions": "How severe is this?",
                 "criteria": ["trivial", "minor", "major", "critical"]},
}

# One question repeated is the wrong shape for the batching table: the server
# builds one answer row per question and the canvas cost is per row, but a
# model asked the same thing twenty times is not what a sweep batch looks
# like. These are distinct yes/no reads over one state, which is.
BATCH_QUESTION_POOL = [
    "Does this mention money?", "Does this mention a deadline?",
    "Is the writer angry?", "Does this name a specific person?",
    "Is this about software?", "Does this ask for a refund?",
    "Is this a first report of the problem?", "Does this mention a deploy?",
    "Is this from a paying customer?", "Does this mention the status page?",
    "Is there a workaround described?", "Does this mention testing?",
    "Is the impact quantified?", "Does this mention security?",
    "Is this reproducible?", "Does this mention a browser?",
    "Is a phone number given?", "Does this mention email?",
    "Is this a duplicate report?", "Does this mention documentation?",
    "Is an account id given?", "Does this mention an outage?",
    "Is a screenshot attached?", "Does this mention monitoring?",
]


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

def post(body: dict, *, timeout: float = 120.0) -> tuple[dict, float]:
    """`(response, wall_ms)`. Raises on anything that is not a 200 — a bench
    that silently scores an error body is worse than one that stops."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        STRUCTURED.rstrip("/") + "/v1/systemone", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read())
    return out, (time.perf_counter() - t0) * 1e3


def nonce(n: int = 12) -> str:
    return "".join(random.choice(string.ascii_lowercase) for _ in range(n))


def cold(state: str) -> str:
    """The same state, with a prefix no prefix-cache block can match.

    It goes at the FRONT. vLLM's prefix cache is a prefix trie over block
    hashes, so a nonce appended to the end still matches every block before
    it and the read is warm for all but its last page — which is exactly the
    measurement error this exists to avoid."""
    return f"[bench {nonce()}]\n{state}"


#: Words per token for this filler, measured against the server's own
#: `usage.input_tokens` rather than assumed. The textbook ~0.75 is for mixed
#: English; this pool is common short words and lands at ~1.0, which put the
#: nominal 8,667-token bucket at 11,646 and made the table hard to compare
#: with the one before it.
FILLER_WORDS_PER_TOKEN = 1.0


def filler(target_tokens: int) -> str:
    """Prose of roughly `target_tokens` tokens.

    The tables report the server's OWN `usage.input_tokens` rather than this
    estimate, so the approximation only decides which buckets get measured,
    never what is reported."""
    words = int(target_tokens * FILLER_WORDS_PER_TOKEN)
    pool = ("the system records every decision it makes so a later reader can "
            "tell what it knew at the time and what it chose to do about it "
            "without having to reconstruct either from the outcome alone ")
    text = (pool * (words // len(pool.split()) + 2)).split()
    return " ".join(text[:words])


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def structured_up() -> bool:
    try:
        with urllib.request.urlopen(STRUCTURED.rstrip("/") + "/health", timeout=3) as r:
            return json.loads(r.read()).get("status") == "ok"
    except Exception:  # noqa: BLE001
        return False


async def require_idle(quiet_s: float = 3.0, limit_s: float = 120.0) -> None:
    """Refuse a busy engine, through the one definition of idle.

    `vllm_metrics.wait_idle` rather than a `num_requests_running == 0` read
    here: a momentary zero is not idle, and the bench and the worker pool
    disagreeing about that is how a shared engine gets measured as a private
    one. djev is `--max-num-seqs 1`, so a neighbour does not merely add noise
    — it serializes in front of every read and the whole table shifts."""
    await vllm_metrics.wait_idle(ENGINE, quiet_s=quiet_s, limit_s=limit_s)


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def p50(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def label_masses(resp: dict) -> list[float]:
    qs = ((resp.get("diagnostics") or {}).get("questions") or {})
    return [float(v.get("label_mass", 0.0)) for v in qs.values()]


def server_ms(resp: dict) -> float:
    return float(((resp.get("diagnostics") or {}).get("timing") or {}).get("total_ms", 0.0))


def repeat(body: dict, n: int, *, warm_first: bool = True) -> list[float]:
    """`n` wall-clock samples of one body.

    The first read of a state pays its prefill, so a p50 over samples that
    includes it measures a mixture of two populations. `warm_first` spends one
    read warming the cache and reports the rest; a caller measuring COLD sets
    it false and passes a body whose state is already nonce'd."""
    if warm_first:
        post(body)
    return [post(body)[1] for _ in range(n)]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def table_headline(samples: int, idle_probe: bool) -> dict:
    rows = []

    body = {"model": "djev", "state": TICKET, "questions": TICKET_QUESTIONS, "samples": 1}
    warm = repeat(body, samples)
    rows.append({"case": "3-question ticket, samples=1", "p50_ms": round(p50(warm), 1),
                 "n": samples, "upstream_gb10_ms": UPSTREAM_GB10_MS["ticket_warm"]})

    if idle_probe:
        # The P8 row. GPU 2 clocks down to 210 MHz with nothing calling it and
        # a sporadic decision finishes BEFORE the clocks ramp, so this is the
        # latency a real advisory call sees — the warm row above is the one
        # that needs the caveat, not this one. It costs `idle_s` of wall time
        # per sample, which is why it is opt-in.
        idle_rows = []
        for _ in range(max(1, samples // 2)):
            time.sleep(float(os.environ.get("BENCH_DJEV_IDLE_S", "45")))
            idle_rows.append(post(body)[1])
        rows.append({"case": "same, after the GPU has idled into P8",
                     "p50_ms": round(p50(idle_rows), 1), "n": len(idle_rows),
                     "upstream_gb10_ms": None})

    state8k = filler(8667)
    q = {"relevant": {"type": "noul", "instructions": "Is this passage about decision records?"}}
    cold_body = {"model": "djev", "state": cold(state8k), "questions": q, "samples": 1}
    resp, cold_ms = post(cold_body)
    tok = int((resp.get("usage") or {}).get("input_tokens") or 0)
    warm_ms = p50(repeat(cold_body, samples, warm_first=False))
    rows.append({"case": f"state ~{tok} tok, cold", "p50_ms": round(cold_ms, 1), "n": 1,
                 "upstream_gb10_ms": UPSTREAM_GB10_MS["state_8k_cold"]})
    rows.append({"case": f"state ~{tok} tok, warm", "p50_ms": round(warm_ms, 1), "n": samples,
                 "upstream_gb10_ms": UPSTREAM_GB10_MS["state_8k_warm"]})

    # The structured server's own overhead, which is the claim "~1 ms of that".
    resp2, wall = post(body)
    rows.append({"case": "structured-server overhead (wall - server total_ms)",
                 "p50_ms": round(wall - server_ms(resp2), 1), "n": 1,
                 "upstream_gb10_ms": None})
    return {"table": "headline", "rows": rows}


def table_batching(samples: int) -> dict:
    rows = []
    for n in (1, 3, 6, 12, 24):
        qs = {f"q{i}": {"type": "noul", "instructions": BATCH_QUESTION_POOL[i]}
              for i in range(n)}
        body = {"model": "djev", "state": TICKET, "questions": qs, "samples": 1}
        ms = repeat(body, samples)
        rows.append({"questions": n, "p50_ms": round(p50(ms), 1), "n": samples,
                     "min_label_mass": round(min(label_masses(post(body)[0])), 3)})
    # The fixed/marginal split the sweep's economics rest on, by least squares
    # over the measured points rather than by eyeballing two of them.
    xs = [r["questions"] for r in rows]
    ys = [r["p50_ms"] for r in rows]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom if denom else float("nan")
    return {"table": "batching", "rows": rows,
            "fit": {"fixed_ms": round(my - slope * mx, 1), "per_question_ms": round(slope, 2)}}


def table_prefill(samples: int) -> dict:
    rows = []
    q = {"relevant": {"type": "noul", "instructions": "Is this passage about decision records?"}}
    for target in (506, 2104, 8667, 21862):
        body = {"model": "djev", "state": cold(filler(target)), "questions": q, "samples": 1}
        resp, cold_ms = post(body)
        tok = int((resp.get("usage") or {}).get("input_tokens") or 0)
        warm_ms = p50(repeat(body, samples, warm_first=False))
        # Prefill is the difference: the warm read does the same decode work
        # off a cached prefix, so what the cold read pays extra IS the prefill.
        implied = (tok / ((cold_ms - warm_ms) / 1e3)) if cold_ms > warm_ms else float("nan")
        rows.append({"state_tokens": tok, "cold_ms": round(cold_ms, 1),
                     "warm_ms": round(warm_ms, 1),
                     "implied_prefill_tok_s": round(implied),
                     "ms_per_token_cold": round(cold_ms / tok, 4) if tok else None})
    return {"table": "prefill", "rows": rows}


def table_capacity(max_n: int) -> dict:
    """N candidates in one canvas: latency, `label_mass`, and the chunk split.

    The two hard limits this table exists to show are both invisible in the
    answers themselves. Above 32 questions the server splits the canvas into
    separate shared contexts, and upstream says plainly that a partitioned
    listwise score is not comparable across chunks — so a reranker that fans a
    pool across them and sorts the union is producing an artefact. And
    `label_mass` collapses as N grows while the returned probabilities stay
    renormalized over the label set, so they keep summing to 1 and keep
    looking like confident scores."""
    rows = []
    levels = ["irrelevant", "tangential", "partly answers it", "directly answers it"]
    query = "How does the system decide whether two backlog items are the same finding?"
    for n in (4, 8, 16, 32, 48, 64):
        if n > max_n:
            break
        parts, qs = [], {}
        for i in range(n):
            parts.append(f"[{i}] " + filler(60))
            qs[f"c{i}"] = {"type": "score", "instructions": f"How well does candidate [{i}] answer the query?",
                           "criteria": levels}
        body = {"model": "djev",
                "state": f"Query: {query}\n\nCandidates:\n" + "\n\n".join(parts),
                "questions": qs, "samples": 1}
        resp, ms = post(body, timeout=300)
        diag = resp.get("diagnostics") or {}
        masses = label_masses(resp)
        scores = [float((a or {}).get("score", 0.0)) for a in (resp.get("answers") or {}).values()]
        rows.append({
            "candidates": n,
            "state_tokens": int((resp.get("usage") or {}).get("input_tokens") or 0),
            "latency_ms": round(ms, 1),
            "ms_per_candidate": round(ms / n, 1),
            "canvas_chunks": len(diag.get("chunks") or []),
            "chunk_sizes": [len(c) for c in (diag.get("chunks") or [])],
            "min_label_mass": round(min(masses), 3) if masses else None,
            # A floor cannot catch this shape: every candidate scoring the same
            # is an empty answer with a perfectly legal label mass.
            "degenerate_all_equal": len(set(round(s, 6) for s in scores)) <= 1,
        })
    return {"table": "capacity", "rows": rows}


TABLES = {
    "headline": lambda a: table_headline(a.samples, a.idle_probe),
    "batching": lambda a: table_batching(a.samples),
    "prefill": lambda a: table_prefill(a.samples),
    "capacity": lambda a: table_capacity(a.max_n),
}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def render(table: dict) -> str:
    rows = table["rows"]
    if not rows:
        return f"{table['table']}: no rows\n"
    cols = list(rows[0].keys())
    widths = {c: max(len(str(c)), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    out = [f"\n== {table['table']} ==",
           "  ".join(str(c).ljust(widths[c]) for c in cols),
           "  ".join("-" * widths[c] for c in cols)]
    for r in rows:
        out.append("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    if "fit" in table:
        out.append(f"fit: {table['fit']}")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", choices=[*TABLES, "all"])
    ap.add_argument("--samples", type=int, default=7,
                    help="Wall-clock samples per warm point (default 7)")
    ap.add_argument("--max-n", type=int, default=64,
                    help="Largest candidate count for `capacity` (default 64)")
    ap.add_argument("--idle-probe", action="store_true",
                    help="Also measure from a clocked-down GPU (sleeps "
                         "BENCH_DJEV_IDLE_S=45 per sample)")
    ap.add_argument("--label", default="", help="Label recorded with the run")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--no-wait-idle", action="store_true",
                    help="Skip the idle check (for a box you know is quiet)")
    args = ap.parse_args()

    if not structured_up():
        print(f"djev structured server is not answering on {STRUCTURED} — "
              f"check `supervisorctl status agent-djev`.", file=sys.stderr)
        return 2
    if not args.no_wait_idle:
        try:
            asyncio.run(require_idle())
        except Exception as exc:  # noqa: BLE001
            print(f"refusing to start: engine is not idle ({exc}). djev serves "
                  f"one sequence at a time, so a neighbour serializes in front "
                  f"of every read here.", file=sys.stderr)
            return 2

    wanted = list(TABLES) if args.table == "all" else [args.table]
    results = []
    for name in wanted:
        table = TABLES[name](args)
        results.append(table)
        print(render(table))

    rec = {"ts": time.time(), "label": args.label, "url": STRUCTURED,
           "tables": results}
    try:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"recorded to {out}")
    except OSError as exc:
        print(f"(not recorded: {exc})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
